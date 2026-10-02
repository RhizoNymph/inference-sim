//! Locked GPU clocks: per-stage compute/bandwidth degradation in the static
//! solver and the serving step cost, placement preference for healthy GPUs,
//! evidence, and bit-identical results without a lock.

use super::*;
use crate::{
    config::parse_cluster,
    types::gpu_clock::GpuClockState,
    workload::{DType, ParameterCountSource},
};

const SUSTAINED_MHZ: f64 = 1835.0;
const KNEE_MHZ: f64 = 1512.0;
const LOCKED_MHZ: f64 = 1200.0;

/// `nodes` RTX-3090-like single-GPU nodes; `clock_keys` is appended to every
/// node and `node1_keys` only to node 1.
fn cluster(nodes: u32, clock_keys: &str, node1_keys: &str) -> Cluster {
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
    for id in 0..nodes {
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
{clock_keys}
{extra}
"#
        ));
    }
    parse_cluster(&text).expect("cluster parses")
}

const MODEL_KEYS: &str = "sustained_sm_clock_mhz = 1835.0\nbandwidth_knee_sm_clock_mhz = 1512.0";
const LOCK_KEY: &str = "locked_sm_clock_mhz = 1200.0";

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

fn calibration() -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.8494,
        decode_memory_bandwidth_scale: 0.8383,
        ..SimulationCalibration::default()
    }
}

fn config(pipeline_ranks: u32) -> ParallelismConfig {
    ParallelismConfig {
        tensor_ranks: 1,
        pipeline_ranks,
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
    cluster: &Cluster,
    request: &InferenceRequest,
    pipeline_ranks: u32,
) -> ScoredParallelismConfig {
    let score = Solver::score_config_with_options(
        cluster,
        &qwen7b(),
        request,
        config(pipeline_ranks),
        SolverOptions {
            calibration: calibration(),
            ..SolverOptions::default()
        },
    );
    assert!(score.feasible, "{:?}", score.rejected_reason);
    score
}

fn layer_compute_durations(score: &ScoredParallelismConfig) -> Vec<f64> {
    score
        .operations
        .iter()
        .filter(|operation| operation.kind == SimOperationKind::Compute)
        .map(|operation| operation.duration_s)
        .collect()
}

fn has_clock_evidence(score: &ScoredParallelismConfig) -> bool {
    score
        .approximations
        .iter()
        .any(|approximation| approximation.code == GPU_CLOCK_DEGRADATION)
}

#[test]
fn unlocked_clock_model_gives_bit_identical_scores() {
    let plain = cluster(2, "", "");
    let declared = cluster(2, MODEL_KEYS, "");
    for phase in [
        InferencePhase::Prefill,
        InferencePhase::Decode,
        InferencePhase::EndToEnd,
    ] {
        for pipeline_ranks in [1, 2] {
            let request = request(8, 512, 128, phase);
            let a = score(&plain, &request, pipeline_ranks);
            let b = score(&declared, &request, pipeline_ranks);
            assert_eq!(a.estimated_latency_s, b.estimated_latency_s);
            assert_eq!(a.operations, b.operations);
            assert_eq!(a.approximations, b.approximations);
            assert!(!has_clock_evidence(&b));
        }
    }
}

#[test]
fn locked_clock_slows_compute_bound_prefill_by_the_sustained_ratio() {
    let healthy = score(
        &cluster(1, MODEL_KEYS, ""),
        &request(8, 2048, 1, InferencePhase::Prefill),
        1,
    );
    let locked = score(
        &cluster(1, &format!("{MODEL_KEYS}\n{LOCK_KEY}"), ""),
        &request(8, 2048, 1, InferencePhase::Prefill),
        1,
    );
    let ratio = locked.estimated_latency_s / healthy.estimated_latency_s;
    assert!(
        (ratio - SUSTAINED_MHZ / LOCKED_MHZ).abs() < 1e-6,
        "prefill slowdown {ratio}"
    );
    assert!(has_clock_evidence(&locked));
}

#[test]
fn locked_clock_slows_memory_bound_decode_by_the_knee_ratio() {
    let request = request(1, 512, 128, InferencePhase::Decode);
    let healthy = score(&cluster(1, MODEL_KEYS, ""), &request, 1);
    let locked = score(
        &cluster(1, &format!("{MODEL_KEYS}\n{LOCK_KEY}"), ""),
        &request,
        1,
    );
    let ratio = locked.estimated_latency_s / healthy.estimated_latency_s;
    assert!(
        (ratio - KNEE_MHZ / LOCKED_MHZ).abs() < 1e-6,
        "decode slowdown {ratio}"
    );
}

#[test]
fn lock_between_knee_and_sustained_leaves_memory_bound_decode_unchanged() {
    let request = request(1, 512, 128, InferencePhase::Decode);
    let healthy = score(&cluster(1, MODEL_KEYS, ""), &request, 1);
    let locked = score(
        &cluster(
            1,
            &format!("{MODEL_KEYS}\nlocked_sm_clock_mhz = 1700.0"),
            "",
        ),
        &request,
        1,
    );
    assert!(
        (locked.estimated_latency_s / healthy.estimated_latency_s - 1.0).abs() < 1e-9,
        "decode should stay memory-bound at full bandwidth"
    );
}

#[test]
fn lock_without_knee_keeps_decode_bandwidth_and_says_so() {
    let request = request(1, 512, 128, InferencePhase::Decode);
    let healthy = score(
        &cluster(1, "sustained_sm_clock_mhz = 1835.0", ""),
        &request,
        1,
    );
    let locked = score(
        &cluster(
            1,
            &format!("sustained_sm_clock_mhz = 1835.0\n{LOCK_KEY}"),
            "",
        ),
        &request,
        1,
    );
    assert!((locked.estimated_latency_s / healthy.estimated_latency_s - 1.0).abs() < 1e-9);
    let evidence = locked
        .approximations
        .iter()
        .find(|approximation| approximation.code == GPU_CLOCK_DEGRADATION)
        .expect("clock evidence");
    assert!(
        evidence.message.contains("bandwidth knee none"),
        "{}",
        evidence.message
    );
    assert!(evidence.message.contains("keeps full HBM bandwidth"));
}

#[test]
fn locked_gpu_in_one_pipeline_stage_slows_only_that_stage() {
    let request = request(8, 512, 1, InferencePhase::Prefill);
    let healthy = score(&cluster(2, MODEL_KEYS, ""), &request, 2);
    let degraded = score(&cluster(2, MODEL_KEYS, LOCK_KEY), &request, 2);
    let stage_of = |score: &ScoredParallelismConfig| {
        score
            .placement
            .rank_to_gpu
            .iter()
            .map(|addr| addr.node_id)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        stage_of(&degraded),
        vec![0, 1],
        "stage 0 on node 0, stage 1 on node 1"
    );
    let healthy_layers = layer_compute_durations(&healthy);
    let degraded_layers = layer_compute_durations(&degraded);
    assert_eq!(healthy_layers.len(), 28);
    for layer in 0..28_u32 {
        let stage = operations::layer_stage(layer, 28, 2);
        let ratio = degraded_layers[layer as usize] / healthy_layers[layer as usize];
        if stage == 0 {
            assert_eq!(ratio, 1.0, "layer {layer} on the healthy stage");
        } else {
            assert!(
                (ratio - SUSTAINED_MHZ / LOCKED_MHZ).abs() < 1e-6,
                "layer {layer} on the locked stage: {ratio}"
            );
        }
    }
    let total_ratio = degraded.estimated_latency_s / healthy.estimated_latency_s;
    assert!(total_ratio > 1.0 && total_ratio < SUSTAINED_MHZ / LOCKED_MHZ);
    let evidence = degraded
        .approximations
        .iter()
        .find(|approximation| approximation.code == GPU_CLOCK_DEGRADATION)
        .expect("clock evidence");
    assert!(
        evidence
            .message
            .contains("stage 1: node 1 gpu 0 locked 1200 MHz"),
        "{}",
        evidence.message
    );
    assert!(!evidence.message.contains("stage 0"));
}

#[test]
fn placement_prefers_the_unlocked_gpu_of_the_same_type() {
    let cluster = parse_cluster(&format!(
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
gpu_count = 2
hbm_gb = 24.0
hbm_bandwidth_gb_s = 936.0
peak_f16_tflops = 88.0
intra = "pcie_gen4"
{MODEL_KEYS}
[[nodes.gpu_profile_overrides]]
gpu = 0
{LOCK_KEY}
"#
    ))
    .expect("cluster parses");
    assert_eq!(
        cluster.nodes[&0]
            .gpu_profile(0)
            .and_then(|profile| profile.clock)
            .map(|clock| clock.state()),
        Some(GpuClockState::Locked(
            crate::types::gpu_clock::SmClockMhz::new(LOCKED_MHZ).expect("clock")
        ))
    );
    let score = score(&cluster, &request(1, 512, 128, InferencePhase::EndToEnd), 1);
    assert_eq!(score.placement.rank_to_gpu[0].local_gpu_id, 1);
    assert!(!has_clock_evidence(&score));
}

#[test]
fn step_cost_is_bit_identical_without_a_lock() {
    let request = request(8, 512, 128, InferencePhase::EndToEnd);
    let plain_cluster = cluster(2, "", "");
    let declared_cluster = cluster(2, MODEL_KEYS, "");
    for pipeline_ranks in [1, 2] {
        let plain_score = score(&plain_cluster, &request, pipeline_ranks);
        let declared_score = score(&declared_cluster, &request, pipeline_ranks);
        let mut plain =
            IterationCostModel::new(&plain_cluster, &qwen7b(), &plain_score, calibration())
                .expect("placed");
        let mut declared =
            IterationCostModel::new(&declared_cluster, &qwen7b(), &declared_score, calibration())
                .expect("placed");
        let mut work = StepWork::default();
        work.add_prefill_chunk(1, 0, 512);
        work.add_decode(16, 700);
        assert_eq!(plain.step_latency(&work), declared.step_latency(&work));
    }
}

#[test]
fn step_cost_scales_decode_memory_and_prefill_compute_on_a_locked_gpu() {
    let request = request(1, 512, 128, InferencePhase::EndToEnd);
    let healthy_cluster = cluster(1, MODEL_KEYS, "");
    let locked_cluster = cluster(1, &format!("{MODEL_KEYS}\n{LOCK_KEY}"), "");
    let healthy_score = score(&healthy_cluster, &request, 1);
    let locked_score = score(&locked_cluster, &request, 1);
    let mut healthy =
        IterationCostModel::new(&healthy_cluster, &qwen7b(), &healthy_score, calibration())
            .expect("placed");
    let mut locked =
        IterationCostModel::new(&locked_cluster, &qwen7b(), &locked_score, calibration())
            .expect("placed");

    let mut decode = StepWork::default();
    decode.add_decode(8, 600);
    let (h, l) = (healthy.step_latency(&decode), locked.step_latency(&decode));
    assert!((l.memory_s / h.memory_s - KNEE_MHZ / LOCKED_MHZ).abs() < 1e-9);
    assert!((l.compute_s / h.compute_s - SUSTAINED_MHZ / LOCKED_MHZ).abs() < 1e-9);

    let mut prefill = StepWork::default();
    prefill.add_completing_prefill_chunk(1, 0, 2048);
    let (h, l) = (
        healthy.step_latency(&prefill),
        locked.step_latency(&prefill),
    );
    let forward_ratio = (l.total_s - l.overhead_s) / (h.total_s - h.overhead_s);
    assert!(
        (forward_ratio - SUSTAINED_MHZ / LOCKED_MHZ).abs() < 1e-9,
        "{forward_ratio}"
    );
}

#[test]
fn step_cost_slows_only_the_locked_stage_share() {
    let request = request(1, 512, 128, InferencePhase::EndToEnd);
    let healthy_cluster = cluster(2, MODEL_KEYS, "");
    let degraded_cluster = cluster(2, MODEL_KEYS, LOCK_KEY);
    let healthy_score = score(&healthy_cluster, &request, 2);
    let degraded_score = score(&degraded_cluster, &request, 2);
    let mut healthy =
        IterationCostModel::new(&healthy_cluster, &qwen7b(), &healthy_score, calibration())
            .expect("placed");
    let mut degraded =
        IterationCostModel::new(&degraded_cluster, &qwen7b(), &degraded_score, calibration())
            .expect("placed");
    let mut decode = StepWork::default();
    decode.add_decode(8, 600);
    let (h, d) = (
        healthy.step_latency(&decode),
        degraded.step_latency(&decode),
    );
    // Half the layers run at full bandwidth, half at locked/knee.
    let expected = 0.5 + 0.5 * KNEE_MHZ / LOCKED_MHZ;
    assert!(
        (d.memory_s / h.memory_s - expected).abs() < 1e-9,
        "{}",
        d.memory_s / h.memory_s
    );
    assert_eq!(d.communication_s, h.communication_s);
}
