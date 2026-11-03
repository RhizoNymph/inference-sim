//! Locked GPU clocks under the vLLM batch-queue microbatch split: each
//! microbatch pass (and the whole-batch decode) is slowed only on the stage
//! that holds the locked GPU.

use super::*;
use crate::{
    calibration::PipelineMicrobatching,
    config::parse_cluster,
    workload::{DType, ParameterCountSource},
};

const LAYERS: u32 = 28;

/// Two single-GPU nodes; `node1_keys` is appended to node 1 only.
fn cluster(node1_keys: &str) -> Cluster {
    let mut text = r#"
schema_version = 1
[cluster]
preset = "custom"
[interconnect]
kind = "ethernet"
variant = "10g"
oversubscription = 1.0
"#
    .to_string();
    for id in 0..2 {
        let extra = if id == 1 { node1_keys } else { "" };
        text.push_str(&format!(
            r#"
[[nodes]]
id = {id}
gpu = "a100_40gb"
gpu_count = 1
hbm_gb = 24.0
hbm_bandwidth_gb_s = 936.0
peak_f16_tflops = 88.0
intra = "pcie_gen4"
nics = {{ count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }}
sustained_sm_clock_mhz = 1835.0
bandwidth_knee_sm_clock_mhz = 1512.0
{extra}
"#
        ));
    }
    parse_cluster(&text).expect("cluster parses")
}

fn qwen7b() -> ModelSpec {
    ModelSpec {
        layers: LAYERS,
        hidden_size: 3584,
        attention_heads: 28,
        kv_heads: 4,
        vocab_size: 152_064,
        parameters: Bytes::from_gigabytes(15.23),
        parameter_count: Some(7.615e9),
        parameter_count_source: ParameterCountSource::ShapeWithFfnWidth,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn score(cluster: &Cluster) -> ScoredParallelismConfig {
    let score = Solver::score_config_with_options(
        cluster,
        &qwen7b(),
        &InferenceRequest {
            batch_size: 4,
            prompt_tokens: 512,
            decode_tokens: 32,
            max_sequence_tokens: 512 + 32,
            phase: InferencePhase::EndToEnd,
        },
        ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 2,
            expert_ranks: 1,
            data_ranks: 1,
        },
        SolverOptions {
            calibration: SimulationCalibration {
                compute_efficiency: 0.85,
                decode_memory_bandwidth_scale: 0.84,
                pipeline_microbatching: PipelineMicrobatching::VllmBatchQueue,
                ..SimulationCalibration::default()
            },
            ..SolverOptions::default()
        },
    );
    assert!(score.feasible, "{:?}", score.rejected_reason);
    score
}

fn duration(score: &ScoredParallelismConfig, name: &str) -> f64 {
    score
        .operations
        .iter()
        .find(|operation| operation.name == name)
        .unwrap_or_else(|| panic!("missing operation {name}"))
        .duration_s
}

#[test]
fn locked_gpu_slows_only_its_stage_in_every_microbatch_pass() {
    let healthy = score(&cluster(""));
    let locked = score(&cluster("locked_sm_clock_mhz = 1200.0"));
    assert!(locked.estimated_latency_s > healthy.estimated_latency_s);

    for prefix in ["microbatch 0 ", "microbatch 1 ", "decode "] {
        let ratios: Vec<f64> = (0..LAYERS)
            .map(|layer| {
                let name = format!("{prefix}layer {layer} compute");
                duration(&locked, &name) / duration(&healthy, &name)
            })
            .collect();
        let (first, second) = ratios.split_at((LAYERS / 2) as usize);
        let nominal = |stage: &[f64]| stage.iter().all(|ratio| (ratio - 1.0).abs() < 1e-12);
        let slowed = |stage: &[f64]| stage.iter().all(|ratio| *ratio > 1.05);
        assert!(
            (nominal(first) && slowed(second)) || (slowed(first) && nominal(second)),
            "{prefix}: per-layer slowdowns {ratios:?}"
        );
    }
}
