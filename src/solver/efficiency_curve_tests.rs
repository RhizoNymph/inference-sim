//! Token-count-dependent compute efficiency in the static roofline and the
//! serving step cost, the prefill weight-read floor, and the parameter-count
//! consistency warning.

use super::*;
use crate::{
    calibration::ComputeEfficiencyCurve,
    config::parse_cluster,
    workload::{DType, ParameterCountSource},
};

fn rtx3090_node() -> Cluster {
    parse_cluster(
        r#"
        schema_version = 1
        [cluster]
        preset = "custom"
        [interconnect]
        kind = "ethernet"
        variant = "10g"
        oversubscription = 1.0
        [[nodes]]
        id = 0
        gpu = "a100_40gb"
        gpu_count = 1
        hbm_gb = 24.0
        hbm_bandwidth_gb_s = 936.0
        peak_f16_tflops = 88.0
        intra = "pcie_gen4"
        nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
        "#,
    )
    .expect("rtx3090 cluster parses")
}

const PEAK_FLOPS: f64 = 88.0e12;
const HBM_BYTES_PER_S: f64 = 936.0e9;
const PARAMETER_COUNT: f64 = 7.615e9;

fn qwen7b() -> ModelSpec {
    ModelSpec {
        layers: 28,
        hidden_size: 3584,
        attention_heads: 28,
        kv_heads: 4,
        vocab_size: 152_064,
        parameters: Bytes::from_gigabytes(15.23),
        parameter_count: Some(PARAMETER_COUNT),
        parameter_count_source: ParameterCountSource::ShapeWithFfnWidth,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn curve() -> ComputeEfficiencyCurve {
    ComputeEfficiencyCurve::new(&[(128, 0.70), (256, 0.66), (512, 0.78), (1024, 0.89)])
        .expect("valid curve")
}

fn scalar_calibration() -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.8494,
        decode_memory_bandwidth_scale: 0.8383,
        ..SimulationCalibration::default()
    }
}

fn curve_calibration() -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency_curve: Some(curve()),
        ..scalar_calibration()
    }
}

fn one_rank() -> ParallelismConfig {
    ParallelismConfig {
        tensor_ranks: 1,
        pipeline_ranks: 1,
        expert_ranks: 1,
        data_ranks: 1,
    }
}

fn request(batch: u32, prompt: u32, decode: u32, phase: InferencePhase) -> InferenceRequest {
    InferenceRequest {
        batch_size: batch,
        prompt_tokens: prompt,
        decode_tokens: decode,
        max_sequence_tokens: prompt + decode,
        phase,
    }
}

fn score_with(
    model: &ModelSpec,
    request: &InferenceRequest,
    calibration: SimulationCalibration,
) -> ScoredParallelismConfig {
    let score = Solver::score_config_with_options(
        &rtx3090_node(),
        model,
        request,
        one_rank(),
        SolverOptions {
            calibration,
            ..SolverOptions::default()
        },
    );
    assert!(score.feasible, "{:?}", score.rejected_reason);
    score
}

fn score(
    request: &InferenceRequest,
    calibration: SimulationCalibration,
) -> ScoredParallelismConfig {
    score_with(&qwen7b(), request, calibration)
}

/// Closed-form compute-bound prefill latency at efficiency `e`.
fn prefill_compute_s(batch: u32, prompt: u32, efficiency: f64) -> f64 {
    let tokens = f64::from(batch) * f64::from(prompt);
    let dense = 2.0 * PARAMETER_COUNT * tokens;
    let attention = 2.0 * 28.0 * 3584.0 * f64::from(prompt) * f64::from(prompt) * f64::from(batch);
    (dense + attention) / (PEAK_FLOPS * efficiency)
}

fn weight_read_s() -> f64 {
    Bytes::from_gigabytes(15.23).as_bytes() as f64 / (HBM_BYTES_PER_S * 0.8383)
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= expected.abs() * 1e-9,
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn prefill_uses_the_curve_at_batch_times_prompt_tokens() {
    for (batch, prompt, efficiency) in [
        (1, 512, 0.78),
        (2, 256, 0.78),
        (1, 1024, 0.89),
        (8, 2048, 0.89),
        (1, 256, 0.66),
    ] {
        let scored = score(
            &request(batch, prompt, 1, InferencePhase::Prefill),
            curve_calibration(),
        );
        assert_close(
            scored.estimated_latency_s,
            prefill_compute_s(batch, prompt, efficiency),
            &format!("{batch}x{prompt}"),
        );
    }
}

#[test]
fn prefill_interpolates_between_curve_points_in_log_tokens() {
    // 362 tokens is close to the geometric midpoint of 256 and 512.
    let tokens = 362.0_f64;
    let fraction = (tokens.ln() - 256.0_f64.ln()) / (512.0_f64.ln() - 256.0_f64.ln());
    let efficiency = 0.66 + fraction * (0.78 - 0.66);
    let scored = score(
        &request(1, 362, 1, InferencePhase::Prefill),
        curve_calibration(),
    );
    assert_close(
        scored.estimated_latency_s,
        prefill_compute_s(1, 362, efficiency),
        "1x362",
    );
}

#[test]
fn without_a_curve_the_scalar_efficiency_is_unchanged() {
    for (batch, prompt) in [(1, 512), (8, 2048), (32, 512)] {
        let scored = score(
            &request(batch, prompt, 1, InferencePhase::Prefill),
            scalar_calibration(),
        );
        assert_close(
            scored.estimated_latency_s,
            prefill_compute_s(batch, prompt, 0.8494),
            &format!("{batch}x{prompt}"),
        );
    }
}

#[test]
fn short_prefill_is_bounded_below_by_one_weight_read() {
    for prompt in [1, 16, 32, 64] {
        let scored = score(
            &request(1, prompt, 1, InferencePhase::Prefill),
            curve_calibration(),
        );
        assert!(prefill_compute_s(1, prompt, 0.70) < weight_read_s());
        assert_close(
            scored.estimated_latency_s,
            weight_read_s(),
            &format!("1x{prompt}"),
        );
    }
}

#[test]
fn memory_bound_decode_is_unaffected_by_the_curve() {
    for (batch, prompt) in [(1, 512), (8, 2048), (32, 512), (64, 512)] {
        let decode = request(batch, prompt, 128, InferencePhase::Decode);
        let with_curve = score(&decode, curve_calibration());
        let scalar = score(&decode, scalar_calibration());
        assert_close(
            with_curve.estimated_latency_s,
            scalar.estimated_latency_s,
            &format!("decode {batch}x{prompt}"),
        );
    }
}

#[test]
fn compute_bound_decode_uses_the_curve_at_the_batch_size() {
    // A tiny weight footprint makes decode compute-bound, exposing the
    // efficiency the decode step uses: the curve at `batch` tokens.
    let model = ModelSpec {
        parameters: Bytes::from_gigabytes(0.01),
        ..qwen7b()
    };
    let at_512 = score_with(
        &model,
        &request(512, 16, 1, InferencePhase::Decode),
        curve_calibration(),
    );
    let at_1024 = score_with(
        &model,
        &request(1024, 16, 1, InferencePhase::Decode),
        curve_calibration(),
    );
    // Doubling the batch doubles FLOPs, but efficiency rises 0.78 -> 0.89.
    let ratio = at_1024.estimated_latency_s / at_512.estimated_latency_s;
    assert!(
        (ratio - 2.0 * 0.78 / 0.89).abs() < 1e-6,
        "decode ratio {ratio}"
    );
}

#[test]
fn step_cost_uses_the_curve_at_the_step_token_count() {
    let cluster = rtx3090_node();
    let calibration = curve_calibration();
    let anchor = score(&request(1, 512, 1, InferencePhase::Decode), calibration);
    let mut model =
        IterationCostModel::new(&cluster, &qwen7b(), &anchor, calibration).expect("placed config");

    // Pure prefill: equals the static prefill at every size, including the
    // memory-bound short prompts.
    for prompt in [16, 128, 512, 1024, 2048] {
        let mut work = StepWork::default();
        work.add_completing_prefill_chunk(1, 0, prompt);
        let step = model.step_latency(&work);
        let scored = score(&request(1, prompt, 1, InferencePhase::Prefill), calibration);
        assert_close(
            step.total_s,
            scored.estimated_latency_s,
            &format!("prefill step {prompt}"),
        );
    }

    // Mixed step: 496 prefill tokens + 16 decodes = 512 tokens -> e(512).
    let mut work = StepWork::default();
    work.add_prefill_chunk(1, 0, 496);
    work.add_decode(16, 600);
    let step = model.step_latency(&work);
    let mut scalar = IterationCostModel::new(
        &cluster,
        &qwen7b(),
        &anchor,
        SimulationCalibration {
            compute_efficiency: 0.78,
            ..scalar_calibration()
        },
    )
    .expect("placed config");
    assert_close(
        step.compute_s,
        scalar.step_latency(&work).compute_s,
        "mixed step",
    );
}

#[test]
fn default_mlp_width_mismatch_emits_a_structured_warning() {
    // Qwen2.5-7B without ffn_hidden_size: the 4 x hidden default derives
    // 6.23B parameters while 15.23 GB of bf16 implies 7.6B (-18%).
    let model = ModelSpec {
        parameter_count: Some(6.23e9),
        parameter_count_source: ParameterCountSource::ShapeWithDefaultFfnWidth,
        ..qwen7b()
    };
    let mismatch = model.parameter_count_mismatch().expect("mismatch");
    assert!((mismatch.relative_difference + 0.182).abs() < 0.01);
    let scored = score_with(
        &model,
        &request(1, 512, 1, InferencePhase::Prefill),
        scalar_calibration(),
    );
    let warning = scored
        .approximations
        .iter()
        .find(|approximation| approximation.code == "model_parameter_count_mismatch")
        .expect("mismatch approximation");
    assert_eq!(warning.phase, "model");
    assert_eq!(warning.category, "model");
    assert!(warning.message.contains("6.230B"), "{}", warning.message);
    assert!(
        warning
            .remediation
            .as_deref()
            .unwrap_or("")
            .contains("ffn_hidden_size")
    );
}

#[test]
fn no_parameter_warning_when_the_count_is_trustworthy() {
    let cases = [
        // Explicit MLP width.
        ModelSpec {
            parameter_count: Some(6.23e9),
            parameter_count_source: ParameterCountSource::ShapeWithFfnWidth,
            ..qwen7b()
        },
        // Explicit parameter count.
        ModelSpec {
            parameter_count: Some(6.23e9),
            parameter_count_source: ParameterCountSource::Explicit,
            ..qwen7b()
        },
        // Default width but within 10% of the bytes.
        ModelSpec {
            parameter_count: Some(7.0e9),
            parameter_count_source: ParameterCountSource::ShapeWithDefaultFfnWidth,
            ..qwen7b()
        },
    ];
    for model in cases {
        assert_eq!(model.parameter_count_mismatch(), None, "{model:?}");
        let scored = score_with(
            &model,
            &request(1, 512, 1, InferencePhase::Prefill),
            scalar_calibration(),
        );
        assert!(
            scored
                .approximations
                .iter()
                .all(|approximation| approximation.code != "model_parameter_count_mismatch")
        );
    }
}
