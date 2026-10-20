//! End-to-end KV-exhaustion preemption on the lab's RTX 3090 / Qwen2.5-7B
//! decode batch sweep setup (lab-runs/2026-10-01-qwen7b-decode-batch-sweep):
//! a static batch of 512-token prompts with 1 + 128 output tokens, a
//! 32,768-token step budget, 64 sequences, and vLLM's logged KV capacity of
//! 24,896 tokens.

use super::super::super::*;
use crate::config::parse_cluster;

const OUTPUT: u32 = 129;
const KV_TOKENS: u64 = 24_896;

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
        peak_f16_tflops = 71.0
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
        parameter_count: None,
        parameter_count_source: crate::workload::ParameterCountSource::Explicit,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn calibration() -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.850_699,
        decode_memory_bandwidth_scale: 0.838_309,
        ..SimulationCalibration::default()
    }
}

fn request() -> InferenceRequest {
    InferenceRequest {
        batch_size: 1,
        prompt_tokens: 512,
        decode_tokens: OUTPUT,
        max_sequence_tokens: 512 + OUTPUT,
        phase: InferencePhase::EndToEnd,
    }
}

fn one_rank() -> SearchSpace {
    SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    }
}

/// All `batch` requests at t = 0, like `LLM.generate` on a static batch.
fn static_batch_traffic(batch: u32, policy: ServingDecodeCapacityPolicy) -> ServingTraffic {
    ServingTraffic {
        request_count: Some(batch),
        arrival: ServingArrivalPattern::FixedGap,
        arrival_gap_s: Some(0.0),
        shape_seed: 0,
        prefill_batching: ServingPrefillBatching::Continuous {
            max_batch_tokens: Some(32_768),
            chunk_tokens: Some(32_768),
        },
        decode_batching: ServingDecodeBatching::Continuous {
            max_batch_tokens: Some(64),
        },
        decode_capacity_policy: policy,
        max_decode_sequences: Some(64),
        max_resident_tokens: Some(KV_TOKENS),
        kv_block_tokens: Some(16),
        max_kv_blocks: Some(KV_TOKENS / 16),
        prefix_cache_hit_rate: Some(0.0),
        batch_sizes: vec![1],
        prompt_tokens: vec![512],
        decode_tokens: vec![OUTPUT],
        ..ServingTraffic::default()
    }
}

fn colocated(traffic: ServingTraffic) -> DisaggregatedServingConfig {
    DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Colocated,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: one_rank(),
            decode: one_rank(),
        },
        traffic,
        slo_policies: Vec::new(),
    }
}

fn simulate(batch: u32, policy: ServingDecodeCapacityPolicy) -> ScoredServingConfig {
    let results = ServingSolver::rank_disaggregated(
        &rtx3090_node(),
        &qwen7b(),
        &request(),
        &colocated(static_batch_traffic(batch, policy)),
        calibration(),
    );
    results.into_iter().next().expect("one serving result")
}

fn has_approximation(result: &ScoredServingConfig, code: &str) -> bool {
    result
        .approximations
        .iter()
        .any(|approximation| approximation.code == code)
}

#[test]
fn batches_that_fit_the_kv_cache_never_preempt_and_match_reserved() {
    // 32 x 641 = 20,512 tokens fit in 24,896.
    let paged = simulate(32, ServingDecodeCapacityPolicy::Preempt);
    let reserved = simulate(32, ServingDecodeCapacityPolicy::CandidateReject);
    assert!(paged.feasible, "{:?}", paged.rejected_reason);
    assert_eq!(paged.metrics.preemptions, 0);
    assert_eq!(paged.metrics.preemption_recompute_tokens, 0);
    assert_eq!(paged.metrics.completed_requests, 32);
    assert_eq!(paged.metrics.e2el_max_s, reserved.metrics.e2el_max_s);
    assert_eq!(paged.metrics.tpot_p50_s, reserved.metrics.tpot_p50_s);
    assert!(has_approximation(
        &paged,
        "iteration_engine_recompute_preemption"
    ));
    assert!(!has_approximation(
        &paged,
        "iteration_engine_kv_reserved_at_admission"
    ));
    assert!(has_approximation(
        &reserved,
        "iteration_engine_kv_reserved_at_admission"
    ));
}

#[test]
fn kv_exhaustion_preempts_and_stretches_the_static_batch() {
    let fits = simulate(32, ServingDecodeCapacityPolicy::Preempt);
    let exhausted = simulate(48, ServingDecodeCapacityPolicy::Preempt);
    // The engine enforced the KV limit online, so the post-hoc reserved-
    // footprint peak (48 x 641 > 24,896) does not reject the candidate.
    assert!(exhausted.feasible, "{:?}", exhausted.rejected_reason);
    assert_eq!(exhausted.metrics.completed_requests, 48);
    assert_eq!(exhausted.metrics.rejected_requests, 0);
    assert!(exhausted.metrics.preemptions > 0);
    assert!(exhausted.metrics.preempted_requests > 0);
    assert!(
        exhausted.metrics.preemption_recompute_tokens >= 512 * exhausted.metrics.preempted_requests
    );
    // The batch decodes in roughly two waves: well over 1.6x the batch-32
    // makespan (measured decode time 2.4x).
    let ratio = exhausted.metrics.e2el_max_s / fits.metrics.e2el_max_s;
    assert!(ratio > 1.6, "makespan ratio {ratio}");
    // Every request still emits exactly its output tokens.
    assert!(
        exhausted
            .request_observations
            .iter()
            .all(|observation| observation.decode_token_finish_s.len() == OUTPUT as usize)
    );
}

#[test]
fn reserved_allocation_queues_whole_footprints_without_preempting() {
    // Without preemption the 48-request batch queues on whole footprints;
    // it still completes, and the engine never preempts.
    let reserved = simulate(48, ServingDecodeCapacityPolicy::CandidateReject);
    assert_eq!(reserved.metrics.preemptions, 0);
}
