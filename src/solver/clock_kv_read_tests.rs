//! A locked SM clock's bandwidth factor applies to KV-cache reads priced at
//! a separate `kv_read_bandwidth_scale` as well as to weight reads, in the
//! static solver and in the serving step cost.

use super::*;
use crate::{
    config::parse_cluster,
    workload::{DType, ParameterCountSource},
};

const KNEE_MHZ: f64 = 1512.0;
const LOCKED_MHZ: f64 = 1200.0;
const MODEL_KEYS: &str = "sustained_sm_clock_mhz = 1835.0\nbandwidth_knee_sm_clock_mhz = 1512.0";
const LOCK_KEY: &str = "locked_sm_clock_mhz = 1200.0";

fn cluster(clock_keys: &str) -> Cluster {
    parse_cluster(&format!(
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
nics = {{ count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }}
{clock_keys}
"#
    ))
    .expect("cluster parses")
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
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn calibration(kv_read_bandwidth_scale: Option<f64>) -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.8494,
        decode_memory_bandwidth_scale: 0.8383,
        kv_read_bandwidth_scale,
        ..SimulationCalibration::default()
    }
}

fn decode_request() -> InferenceRequest {
    InferenceRequest {
        batch_size: 4,
        prompt_tokens: 4096,
        decode_tokens: 64,
        max_sequence_tokens: 4096 + 64,
        phase: InferencePhase::Decode,
    }
}

fn score(cluster: &Cluster, calibration: SimulationCalibration) -> ScoredParallelismConfig {
    let score = Solver::score_config_with_options(
        cluster,
        &qwen7b(),
        &decode_request(),
        ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        },
        SolverOptions {
            calibration,
            ..SolverOptions::default()
        },
    );
    assert!(score.feasible, "{:?}", score.rejected_reason);
    score
}

#[test]
fn locked_clock_slows_kv_reads_at_the_separate_scale_by_the_knee_ratio() {
    let kv_scaled = calibration(Some(0.4687));
    let healthy = score(&cluster(MODEL_KEYS), kv_scaled);
    let locked = score(&cluster(&format!("{MODEL_KEYS}\n{LOCK_KEY}")), kv_scaled);
    let shared = score(&cluster(MODEL_KEYS), calibration(None));
    assert!(
        healthy.estimated_latency_s > shared.estimated_latency_s,
        "the separate KV-read scale must price long-context decode slower"
    );
    let ratio = locked.estimated_latency_s / healthy.estimated_latency_s;
    assert!(
        (ratio - KNEE_MHZ / LOCKED_MHZ).abs() < 1e-6,
        "decode slowdown {ratio}"
    );
}

#[test]
fn step_cost_slows_kv_reads_at_the_separate_scale_by_the_knee_ratio() {
    let kv_scaled = calibration(Some(0.4687));
    let healthy_cluster = cluster(MODEL_KEYS);
    let locked_cluster = cluster(&format!("{MODEL_KEYS}\n{LOCK_KEY}"));
    let healthy_score = score(&healthy_cluster, kv_scaled);
    let locked_score = score(&locked_cluster, kv_scaled);
    let mut healthy =
        IterationCostModel::new(&healthy_cluster, &qwen7b(), &healthy_score, kv_scaled)
            .expect("placed");
    let mut locked = IterationCostModel::new(&locked_cluster, &qwen7b(), &locked_score, kv_scaled)
        .expect("placed");
    let mut shared = IterationCostModel::new(
        &healthy_cluster,
        &qwen7b(),
        &healthy_score,
        calibration(None),
    )
    .expect("placed");

    let mut decode = StepWork::default();
    decode.add_decode(32, 4096);
    let (h, l, s) = (
        healthy.step_latency(&decode),
        locked.step_latency(&decode),
        shared.step_latency(&decode),
    );
    assert!(
        h.memory_s > s.memory_s,
        "KV reads priced at the separate scale"
    );
    assert!(
        (l.memory_s / h.memory_s - KNEE_MHZ / LOCKED_MHZ).abs() < 1e-9,
        "memory slowdown {}",
        l.memory_s / h.memory_s
    );
}
