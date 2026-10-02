//! End-to-end disaggregated serving through `ServingSolver` on the RTX 3090
//! lab cluster with its measured network (examples/), Qwen2.5-7B, 512-token
//! prompts and 128 output tokens.

use std::time::Instant;

use super::super::super::*;
use crate::config::parse_cluster;

const LAB_CLUSTER: &str = include_str!("../../../../examples/rtx3090_lab_cluster_measured_curves.toml");
const PROMPT_KV_BYTES: u64 = 512 * 28 * 4 * 128 * 2 * 2;

fn lab_cluster() -> Cluster {
    parse_cluster(LAB_CLUSTER).expect("lab cluster parses")
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

fn request() -> InferenceRequest {
    InferenceRequest {
        batch_size: 1,
        prompt_tokens: 512,
        decode_tokens: 128,
        max_sequence_tokens: 640,
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

/// vLLM per-instance limits: max_num_batched_tokens 2048, max_num_seqs 64,
/// and each GPU's KV cache.
fn traffic(arrival: ServingArrivalPattern) -> ServingTraffic {
    ServingTraffic {
        request_count: Some(200),
        arrival,
        prefill_batching: ServingPrefillBatching::Continuous {
            max_batch_tokens: Some(2048),
            chunk_tokens: Some(2048),
        },
        decode_batching: ServingDecodeBatching::Continuous {
            max_batch_tokens: Some(64),
        },
        max_decode_sequences_per_gpu: Some(64),
        max_resident_tokens_per_gpu: Some(82_864),
        kv_block_tokens: Some(16),
        max_kv_blocks_per_gpu: Some(5_179),
        prefix_cache_hit_rate: Some(0.0),
        batch_sizes: vec![1],
        prompt_tokens: vec![512],
        decode_tokens: vec![128],
        ..ServingTraffic::default()
    }
}

fn config(
    mode: ServingDeploymentMode,
    prefill_nodes: Vec<NodeId>,
    decode_nodes: Vec<NodeId>,
    traffic: ServingTraffic,
) -> DisaggregatedServingConfig {
    DisaggregatedServingConfig {
        deployment_mode: mode,
        prefill_nodes,
        decode_nodes,
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

fn simulate_config(config: &DisaggregatedServingConfig) -> ScoredServingConfig {
    let results = ServingSolver::rank_disaggregated(
        &lab_cluster(),
        &qwen7b(),
        &request(),
        config,
        calibration(),
    );
    assert_eq!(results.len(), 1);
    results.into_iter().next().expect("one serving result")
}

/// Prefill on `prefill`, decode on `decode`.
fn simulate(prefill: NodeId, decode: NodeId, arrival: ServingArrivalPattern) -> ScoredServingConfig {
    simulate_config(&config(
        ServingDeploymentMode::FullyDisaggregated,
        vec![prefill],
        vec![decode],
        traffic(arrival),
    ))
}

fn poisson(rate_per_s: f64) -> ServingArrivalPattern {
    ServingArrivalPattern::Poisson {
        rate_per_s,
        seed: 0,
    }
}

fn has_approximation(result: &ScoredServingConfig, code: &str) -> bool {
    result
        .approximations
        .iter()
        .any(|approximation| approximation.code == code)
}

#[test]
fn lab_disaggregated_pool_runs_on_the_engine_with_kv_pulls() {
    let result = simulate(0, 1, poisson(2.0));
    assert!(result.feasible, "{:?}", result.rejected_reason);
    assert!(has_approximation(&result, "kv_transfer_fifo_link_queues"));
    assert!(has_approximation(&result, "kv_transfer_decode_initiated_pull"));
    assert!(!has_approximation(&result, "phase_pipeline_scheduler"));
    assert_eq!(result.metrics.completed_requests, 200);
    assert_eq!(result.metrics.rejected_requests, 0);
    for observation in &result.request_observations {
        assert_eq!(observation.decode_token_finish_s.len(), 128);
        assert_eq!(observation.kv_transfer_bytes, PROMPT_KV_BYTES);
        assert_eq!(observation.prefill_node, 0);
        assert_eq!(observation.decode_node, 1);
        assert!(observation.kv_start_s + 1e-12 >= observation.prefill_finish_s);
        assert!(observation.first_decode_finish_s + 1e-12 >= observation.kv_finish_s);
        assert!(!observation.kv_transfer_resources.is_empty());
        assert_eq!(observation.kv_transfer_paths.len(), 1);
        let phases = observation
            .worker_assignments
            .iter()
            .map(|assignment| assignment.phase.as_str())
            .collect::<Vec<_>>();
        for phase in ["prefill", "kv_transfer", "decode"] {
            assert!(phases.contains(&phase), "{phases:?}");
        }
    }
    let transfers = result
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("kv-transfer 0->1"))
        .count();
    assert_eq!(transfers, 200);
    let steps_on = |node: NodeId| {
        result
            .scheduled_operations
            .iter()
            .filter(|operation| {
                operation.name.starts_with("engine step")
                    && operation
                        .resources
                        .contains(&format!("gpu compute node {node}"))
            })
            .count()
    };
    assert!(steps_on(0) > 0 && steps_on(1) > 0);
    assert!(!result.kv_route_resource_summary.is_empty());
}

#[test]
fn light_load_ttft_is_prefill_plus_pull_plus_one_decode_step() {
    let result = simulate(0, 1, poisson(1.0));
    let ttft_ms = result.metrics.ttft_p50_s * 1e3;
    // ~98 ms prefill + ~81 ms pull over the measured 0->1 curve + ~20 ms
    // decode step that recomputes the last prompt token.
    assert!((180.0..260.0).contains(&ttft_ms), "ttft p50 {ttft_ms} ms");
    let observation = &result.request_observations[0];
    let pull_ms = (observation.kv_finish_s - observation.kv_start_s) * 1e3;
    assert!((70.0..95.0).contains(&pull_ms), "pull {pull_ms} ms");
    // The decode worker never carries prefill chunks, so TPOT stays at the
    // decode step even while prompts arrive.
    let itl_ms = result.metrics.itl_p50_s * 1e3;
    assert!((18.5..21.5).contains(&itl_ms), "itl p50 {itl_ms} ms");
    assert!(result.metrics.tpot_p50_s < 1.1 * result.metrics.itl_p50_s);
}

#[test]
fn prefill_instance_convention_reports_the_prefill_token() {
    let mut prefill_first = traffic(poisson(1.0));
    prefill_first.disaggregated_first_token = ServingDisaggregatedFirstToken::PrefillInstance;
    let prefill_token = simulate_config(&config(
        ServingDeploymentMode::FullyDisaggregated,
        vec![0],
        vec![1],
        prefill_first,
    ));
    let decode_token = simulate(0, 1, poisson(1.0));
    assert!(prefill_token.feasible);
    let saved_ms = (decode_token.metrics.ttft_p50_s - prefill_token.metrics.ttft_p50_s) * 1e3;
    // The prefill token skips the pull and the recompute step.
    assert!((80.0..130.0).contains(&saved_ms), "saved {saved_ms} ms");
    for observation in &prefill_token.request_observations {
        assert_eq!(observation.decode_token_finish_s.len(), 128);
        assert!((observation.first_decode_finish_s - observation.prefill_finish_s).abs() < 1e-9);
    }
}

#[test]
fn kv_flows_over_the_slow_direction_cost_more() {
    let forward = simulate(0, 1, poisson(1.0));
    let backward = simulate(1, 0, poisson(1.0));
    let pull = |result: &ScoredServingConfig| {
        let observation = &result.request_observations[0];
        observation.kv_finish_s - observation.kv_start_s
    };
    // Measured node0->node1 is ~0.36 GB/s, node1->node0 ~1.2 GB/s.
    assert!(pull(&forward) > 2.5 * pull(&backward));
    assert!(forward.metrics.ttft_p50_s > backward.metrics.ttft_p50_s + 0.04);
}

#[test]
fn burst_load_queues_pulls_on_the_link_one_at_a_time() {
    let result = simulate(0, 1, ServingArrivalPattern::FixedGap);
    assert!(result.feasible, "{:?}", result.rejected_reason);
    assert_eq!(result.metrics.completed_requests, 200);
    let mut windows = result
        .request_observations
        .iter()
        .map(|observation| (observation.kv_start_s, observation.kv_finish_s))
        .collect::<Vec<_>>();
    windows.sort_by(|left, right| left.0.total_cmp(&right.0));
    for pair in windows.windows(2) {
        assert!(pair[1].0 + 1e-9 >= pair[0].1, "overlapping pulls {pair:?}");
    }
    // Each 2048-token prefill step finishes four prompts at once; their
    // ~81 ms pulls then queue behind each other on node0's egress.
    let max_link_wait_s = result
        .request_observations
        .iter()
        .map(|observation| observation.kv_resource_queue_s)
        .fold(0.0, f64::max);
    assert!(max_link_wait_s > 0.15, "max link wait {max_link_wait_s}");
    assert!(result.metrics.peak_decode_sequences <= 128);
}

#[test]
fn partially_disaggregated_pool_mixes_colocated_and_pulled_requests() {
    let result = simulate_config(&config(
        ServingDeploymentMode::PartiallyDisaggregated,
        vec![0, 1],
        vec![1],
        traffic(poisson(2.0)),
    ));
    assert!(result.feasible, "{:?}", result.rejected_reason);
    assert!(!has_approximation(&result, "phase_pipeline_scheduler"));
    assert_eq!(result.metrics.completed_requests, 200);
    let pulled = result
        .request_observations
        .iter()
        .filter(|observation| observation.kv_transfer_bytes > 0)
        .count();
    assert!(pulled > 0 && pulled < 200, "pulled {pulled}");
    for observation in &result.request_observations {
        assert_eq!(observation.decode_node, 1);
        if observation.prefill_node == 1 {
            assert_eq!(observation.kv_transfer_bytes, 0);
        } else {
            assert_eq!(observation.kv_transfer_bytes, PROMPT_KV_BYTES);
        }
    }
}

#[test]
fn independent_batching_keeps_the_phase_pipeline() {
    let mut independent = traffic(poisson(2.0));
    independent.request_count = Some(20);
    independent.prefill_batching = ServingPrefillBatching::Independent;
    let result = simulate_config(&config(
        ServingDeploymentMode::FullyDisaggregated,
        vec![0],
        vec![1],
        independent,
    ));
    assert!(has_approximation(&result, "phase_pipeline_scheduler"));
    assert!(!has_approximation(&result, "kv_transfer_fifo_link_queues"));
}

#[test]
fn two_hundred_request_disaggregated_run_finishes_quickly() {
    let started = Instant::now();
    let result = simulate(0, 1, poisson(4.0));
    let elapsed = started.elapsed();
    assert!(result.feasible, "{:?}", result.rejected_reason);
    assert_eq!(result.metrics.completed_requests, 200);
    let bound_s = if cfg!(debug_assertions) { 10.0 } else { 1.0 };
    assert!(
        elapsed.as_secs_f64() < bound_s,
        "200-request disaggregated simulation took {elapsed:?}"
    );
}
