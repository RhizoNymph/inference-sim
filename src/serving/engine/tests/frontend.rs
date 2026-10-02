//! Calibrated API-server (frontend) latency on the iteration engine: it is a
//! constant per-request delay before the engine sees a request, so it adds to
//! TTFT and E2EL and leaves TPOT, ITL, and queueing unchanged.

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
        parameter_count_source: crate::workload::ParameterCountSource::ShapeWithFfnWidth,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn base_calibration() -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.8494,
        decode_memory_bandwidth_scale: 0.8383,
        ..SimulationCalibration::default()
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

fn simulate(rate_per_s: f64, calibration: SimulationCalibration) -> ScoredServingConfig {
    let traffic = ServingTraffic {
        request_count: Some(60),
        arrival: ServingArrivalPattern::Poisson {
            rate_per_s,
            seed: 0,
        },
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
        decode_tokens: vec![32],
        ..ServingTraffic::default()
    };
    let config = DisaggregatedServingConfig {
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
    };
    let request = InferenceRequest {
        batch_size: 1,
        prompt_tokens: 512,
        decode_tokens: 32,
        max_sequence_tokens: 544,
        phase: InferencePhase::EndToEnd,
    };
    let results = ServingSolver::rank_disaggregated(
        &rtx3090_node(),
        &qwen7b(),
        &request,
        &config,
        calibration,
    );
    let result = results.into_iter().next().expect("one serving result");
    assert!(result.feasible, "{:?}", result.rejected_reason);
    result
}

fn codes(result: &ScoredServingConfig) -> Vec<&str> {
    result
        .approximations
        .iter()
        .map(|approximation| approximation.code.as_str())
        .collect()
}

fn assert_shifted(base: &ScoredServingConfig, shifted: &ScoredServingConfig, delay_s: f64) {
    assert_eq!(
        base.request_observations.len(),
        shifted.request_observations.len()
    );
    for (a, b) in base
        .request_observations
        .iter()
        .zip(&shifted.request_observations)
    {
        assert!(
            (b.ttft_s - a.ttft_s - delay_s).abs() < 1e-9,
            "ttft {} -> {}",
            a.ttft_s,
            b.ttft_s
        );
        assert!(
            (b.e2el_s - a.e2el_s - delay_s).abs() < 1e-9,
            "e2el {} -> {}",
            a.e2el_s,
            b.e2el_s
        );
        assert!((b.tpot_s - a.tpot_s).abs() < 1e-12);
        assert!((b.itl_s - a.itl_s).abs() < 1e-12);
    }
}

#[test]
fn fixed_frontend_latency_adds_to_ttft_and_e2el_only() {
    let base = simulate(2.0, base_calibration());
    let shifted = simulate(
        2.0,
        SimulationCalibration {
            frontend_latency_us: 10_000.0,
            ..base_calibration()
        },
    );
    assert_shifted(&base, &shifted, 0.010);
}

#[test]
fn per_prompt_token_frontend_latency_scales_with_prompt_length() {
    let base = simulate(1.0, base_calibration());
    let shifted = simulate(
        1.0,
        SimulationCalibration {
            frontend_latency_us: 5_000.0,
            frontend_latency_per_prompt_token_us: 10.0,
            ..base_calibration()
        },
    );
    // 5 ms + 512 tokens x 10 us.
    assert_shifted(&base, &shifted, 0.010_12);
}

#[test]
fn frontend_latency_holds_under_load() {
    // Near saturation queueing dominates TTFT; a constant ingress delay still
    // shifts every request by exactly the delay.
    let base = simulate(8.0, base_calibration());
    let shifted = simulate(
        8.0,
        SimulationCalibration {
            frontend_latency_us: 7_000.0,
            ..base_calibration()
        },
    );
    assert_shifted(&base, &shifted, 0.007);
}

#[test]
fn approximation_records_whether_frontend_latency_is_modeled() {
    let unset = simulate(1.0, base_calibration());
    assert!(codes(&unset).contains(&"iteration_engine_no_frontend_overhead"));
    assert!(!codes(&unset).contains(&"iteration_engine_constant_frontend_latency"));

    let set = simulate(
        1.0,
        SimulationCalibration {
            frontend_latency_us: 6_000.0,
            ..base_calibration()
        },
    );
    assert!(codes(&set).contains(&"iteration_engine_constant_frontend_latency"));
    assert!(!codes(&set).contains(&"iteration_engine_no_frontend_overhead"));
}
