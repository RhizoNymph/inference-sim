//! Separate KV-read bandwidth scale: KV-cache reads in the decode roofline
//! and the serving step cost are priced at `kv_read_bandwidth_scale` of peak
//! HBM bandwidth, weight reads stay at `decode_memory_bandwidth_scale`, and
//! leaving the key out changes nothing.

use super::*;
use crate::{config::parse_cluster, workload::ParameterCountSource};

const HBM_BYTES_PER_S: f64 = 936.0e9;
const WEIGHT_SCALE: f64 = 0.8383;
const KV_SCALE: f64 = 0.45;
/// 2 (K and V) x 28 layers x 4 KV heads x 128 head dim x 2 bytes (bf16).
const KV_BYTES_PER_TOKEN: f64 = 2.0 * 28.0 * 4.0 * 128.0 * 2.0;

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

fn qwen7b() -> ModelSpec {
    ModelSpec {
        layers: 28,
        hidden_size: 3584,
        attention_heads: 28,
        kv_heads: 4,
        vocab_size: 152_064,
        parameters: Bytes::from_gigabytes(15.23),
        parameter_count: Some(7.615e9),
        parameter_count_source: ParameterCountSource::ShapeWithFfnWidth,
        dtype: crate::workload::DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn shared_scale() -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.8494,
        decode_memory_bandwidth_scale: WEIGHT_SCALE,
        ..SimulationCalibration::default()
    }
}

fn separate_scale() -> SimulationCalibration {
    SimulationCalibration {
        kv_read_bandwidth_scale: Some(KV_SCALE),
        ..shared_scale()
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

fn score(
    request: &InferenceRequest,
    calibration: SimulationCalibration,
) -> ScoredParallelismConfig {
    let score = Solver::score_config_with_options(
        &rtx3090_node(),
        &qwen7b(),
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

fn weight_read_s(scale: f64) -> f64 {
    Bytes::from_gigabytes(15.23).as_bytes() as f64 / (HBM_BYTES_PER_S * scale)
}

fn kv_read_s(context_tokens: f64, scale: f64) -> f64 {
    context_tokens * KV_BYTES_PER_TOKEN / (HBM_BYTES_PER_S * scale)
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= expected.abs() * 1e-9,
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn decode_prices_kv_reads_at_the_kv_scale_and_weights_at_the_weight_scale() {
    for (batch, prompt) in [(1, 512), (8, 2048), (32, 512), (4, 4096)] {
        let decode = 128;
        let scored = score(
            &request(batch, prompt, decode, InferencePhase::Decode),
            separate_scale(),
        );
        let average_context = f64::from(prompt) + (f64::from(decode) + 1.0) / 2.0;
        let kv_tokens = average_context * f64::from(decode) * f64::from(batch);
        let expected =
            weight_read_s(WEIGHT_SCALE) * f64::from(decode) + kv_read_s(kv_tokens, KV_SCALE);
        assert_close(
            scored.estimated_latency_s,
            expected,
            &format!("decode {batch}x{prompt}"),
        );
    }
}

#[test]
fn a_lower_kv_scale_slows_decode_more_as_kv_grows() {
    let slowdown = |batch: u32| {
        let decode = request(batch, 512, 128, InferencePhase::Decode);
        score(&decode, separate_scale()).estimated_latency_s
            / score(&decode, shared_scale()).estimated_latency_s
    };
    let at_1 = slowdown(1);
    let at_32 = slowdown(32);
    assert!(at_1 > 1.0, "batch 1 slowdown {at_1}");
    assert!(at_32 > at_1, "batch 32 slowdown {at_32} vs batch 1 {at_1}");
}

#[test]
fn kv_scale_equal_to_the_weight_scale_matches_the_absent_key() {
    let same = SimulationCalibration {
        kv_read_bandwidth_scale: Some(WEIGHT_SCALE),
        ..shared_scale()
    };
    for phase in [
        InferencePhase::Prefill,
        InferencePhase::Decode,
        InferencePhase::EndToEnd,
    ] {
        let shape = request(8, 2048, 128, phase);
        assert_eq!(
            score(&shape, same).estimated_latency_s,
            score(&shape, shared_scale()).estimated_latency_s,
            "{phase:?}"
        );
    }
}

#[test]
fn static_prefill_ignores_the_kv_scale() {
    // The static prefill roofline has no KV-read term (attention over the
    // prompt is compute-bound, the floor is one weight read), so the KV
    // scale never reaches it, compute-bound or weight-read-bound.
    for prompt in [16, 512, 4096] {
        let shape = request(1, prompt, 1, InferencePhase::Prefill);
        assert_eq!(
            score(&shape, separate_scale()).estimated_latency_s,
            score(&shape, shared_scale()).estimated_latency_s,
            "prefill 1x{prompt}"
        );
    }
}

#[test]
fn approximation_records_the_separate_kv_scale_only_when_set() {
    let code = "kv_read_bandwidth_scale_uniform";
    let decode = request(8, 512, 128, InferencePhase::Decode);
    let scored = score(&decode, separate_scale());
    let record = scored
        .approximations
        .iter()
        .find(|approximation| approximation.code == code)
        .expect("kv-read approximation");
    assert_eq!(record.phase, "decode");
    assert_eq!(record.category, "memory");
    assert!(record.message.contains("0.450"), "{}", record.message);
    assert!(record.message.contains("0.838"), "{}", record.message);

    let end_to_end = score(
        &request(8, 512, 128, InferencePhase::EndToEnd),
        separate_scale(),
    );
    assert!(
        end_to_end
            .approximations
            .iter()
            .any(|approximation| approximation.code == code)
    );

    let prefill = score(
        &request(8, 512, 1, InferencePhase::Prefill),
        separate_scale(),
    );
    assert!(
        !prefill
            .approximations
            .iter()
            .any(|approximation| approximation.code == code)
    );
    let absent = score(&decode, shared_scale());
    assert!(
        !absent
            .approximations
            .iter()
            .any(|approximation| approximation.code == code)
    );
}

#[test]
fn pure_decode_step_equals_the_one_token_decode_with_a_kv_scale() {
    let cluster = rtx3090_node();
    for (batch, prompt) in [(1, 512), (32, 512), (4, 4096)] {
        let anchor = score(
            &request(batch, prompt, 1, InferencePhase::Decode),
            separate_scale(),
        );
        let mut model = IterationCostModel::new(&cluster, &qwen7b(), &anchor, separate_scale())
            .expect("placed config");
        let mut work = StepWork::default();
        work.add_decode(batch, prompt + 1);
        let step = model.step_latency(&work);
        assert_close(
            step.total_s,
            anchor.estimated_latency_s,
            &format!("decode step {batch}x{prompt}"),
        );
        let context = f64::from(batch) * f64::from(prompt + 1);
        assert_close(
            step.memory_s,
            weight_read_s(WEIGHT_SCALE) + kv_read_s(context, KV_SCALE),
            &format!("decode step memory {batch}x{prompt}"),
        );
    }
}

#[test]
fn step_cost_prices_prefill_context_rereads_at_the_kv_scale() {
    // A later prefill chunk re-reads the earlier chunks' KV through the same
    // paged-attention path as decode, so both use the KV-read bandwidth.
    let cluster = rtx3090_node();
    let anchor = score(
        &request(1, 512, 1, InferencePhase::Decode),
        separate_scale(),
    );
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &anchor, separate_scale())
        .expect("placed config");
    let mut work = StepWork::default();
    work.add_prefill_chunk(1, 6000, 16);
    work.add_decode(4, 700);
    let step = model.step_latency(&work);
    let kv_tokens = 6000.0 + 4.0 * 700.0;
    assert_close(
        step.memory_s,
        weight_read_s(WEIGHT_SCALE) + kv_read_s(kv_tokens, KV_SCALE),
        "mixed step memory",
    );
}

#[test]
fn step_cost_without_a_kv_scale_keeps_the_shared_bandwidth_formula() {
    let cluster = rtx3090_node();
    let calibration = shared_scale();
    let anchor = score(&request(1, 512, 1, InferencePhase::Decode), calibration);
    let mut model =
        IterationCostModel::new(&cluster, &qwen7b(), &anchor, calibration).expect("placed config");
    let mut work = StepWork::default();
    work.add_prefill_chunk(2, 300, 64);
    work.add_decode(16, 900);
    let step = model.step_latency(&work);
    let bandwidth = Solver::effective_hbm_bandwidth(&cluster, &anchor.placement, calibration);
    let kv_tokens = 2.0 * 300.0 + 16.0 * 900.0;
    let expected = (Bytes::from_gigabytes(15.23).as_bytes() as f64
        + kv_tokens * KV_BYTES_PER_TOKEN)
        / bandwidth
        * calibration.decode_compute_scale;
    assert_eq!(step.memory_s, expected);
}
