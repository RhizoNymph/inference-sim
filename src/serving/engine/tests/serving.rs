//! End-to-end checks through `ServingSolver` on the lab's RTX 3090 /
//! Qwen2.5-7B serving setup, with the static-batch-fitted constants.

use std::time::Instant;

use super::super::super::*;
use crate::config::parse_cluster;

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

/// vLLM serving setup of lab-runs/2026-09-30-serving-baseline.
fn vllm_like_traffic(arrival: ServingArrivalPattern) -> ServingTraffic {
    ServingTraffic {
        request_count: Some(200),
        arrival,
        shape_seed: 0,
        prefill_batching: ServingPrefillBatching::Continuous {
            max_batch_tokens: Some(2048),
            chunk_tokens: Some(2048),
        },
        decode_batching: ServingDecodeBatching::Continuous {
            max_batch_tokens: Some(64),
        },
        max_decode_sequences: Some(64),
        max_resident_tokens: Some(82_864),
        kv_block_tokens: Some(16),
        max_kv_blocks: Some(5_179),
        prefix_cache_hit_rate: Some(0.0),
        batch_sizes: vec![1],
        prompt_tokens: vec![512],
        decode_tokens: vec![128],
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

fn simulate(arrival: ServingArrivalPattern) -> ScoredServingConfig {
    let results = ServingSolver::rank_disaggregated(
        &rtx3090_node(),
        &qwen7b(),
        &request(),
        &colocated(vllm_like_traffic(arrival)),
        calibration(),
    );
    assert_eq!(results.len(), 1);
    results.into_iter().next().expect("one serving result")
}

fn poisson(rate_per_s: f64) -> ServingArrivalPattern {
    ServingArrivalPattern::Poisson {
        rate_per_s,
        seed: 0,
    }
}

fn burst() -> ServingArrivalPattern {
    ServingArrivalPattern::FixedGap
}

#[test]
fn colocated_continuous_serving_runs_on_the_iteration_engine() {
    let result = simulate(poisson(2.0));
    assert!(result.feasible, "{:?}", result.rejected_reason);
    assert!(
        result
            .scheduled_operations
            .iter()
            .all(|operation| operation.name.starts_with("engine step"))
    );
    assert!(
        result
            .approximations
            .iter()
            .any(|approximation| approximation.code == "iteration_engine_kv_reserved_at_admission")
    );
    assert_eq!(result.metrics.completed_requests, 200);
    // vLLM semantics: the prefill step emits token 1, so 127 decode steps follow.
    assert!(
        result
            .request_observations
            .iter()
            .all(|observation| observation.decode_token_finish_s.len() == 128)
    );
}

#[test]
fn light_load_decode_steps_match_the_measured_memory_bound_step() {
    let result = simulate(poisson(1.0));
    let itl_p50_ms = result.metrics.itl_p50_s * 1e3;
    // Measured static decode step at batch 1-8: 19.3-19.7 ms; vLLM serving ITL p50 19.4 ms.
    assert!(
        (18.5..21.0).contains(&itl_p50_ms),
        "itl p50 {itl_p50_ms} ms"
    );
    // Prefill-carrying steps inflate TPOT above ITL.
    assert!(result.metrics.tpot_p50_s > result.metrics.itl_p50_s);
}

#[test]
fn overload_queues_requests_and_grows_ttft_instead_of_rejecting() {
    let light = simulate(poisson(1.0));
    let heavy = simulate(burst());
    for result in [&light, &heavy] {
        assert!(result.feasible, "{:?}", result.rejected_reason);
        assert_eq!(result.metrics.rejected_requests, 0);
        assert_eq!(result.metrics.completed_requests, 200);
        assert!(result.metrics.peak_decode_sequences <= 64);
    }
    assert!(
        heavy.metrics.ttft_p50_s > 2.0,
        "burst ttft {}",
        heavy.metrics.ttft_p50_s
    );
    assert!(heavy.metrics.ttft_p50_s > 10.0 * light.metrics.ttft_p50_s);
    // TPOT rises with load while ITL p50 stays near the decode step.
    assert!(heavy.metrics.tpot_p50_s > 2.0 * light.metrics.tpot_p50_s);
    assert!(heavy.metrics.itl_p50_s < 1.5 * light.metrics.itl_p50_s);
    // Saturation: one 3090 serves roughly 800 output tokens/s.
    let throughput = heavy.metrics.throughput_tokens_per_s;
    assert!(
        (600.0..1000.0).contains(&throughput),
        "throughput {throughput}"
    );
}

#[test]
fn two_hundred_request_serving_run_finishes_quickly() {
    let started = Instant::now();
    let result = simulate(poisson(4.0));
    let elapsed = started.elapsed();
    assert!(result.feasible, "{:?}", result.rejected_reason);
    // Release builds take well under a second; debug builds get headroom.
    let bound_s = if cfg!(debug_assertions) { 10.0 } else { 1.0 };
    assert!(
        elapsed.as_secs_f64() < bound_s,
        "200-request serving simulation took {elapsed:?}"
    );
}
