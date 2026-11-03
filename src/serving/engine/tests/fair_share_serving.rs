//! End-to-end fair-share KV pulls and the proxy constant through
//! `ServingSolver` on the RTX 3090 lab cluster (Qwen2.5-7B, 512/128).

use super::super::super::*;
use crate::calibration::KvLinkSharing;
use crate::config::parse_cluster;

const LAB_CLUSTER: &str =
    include_str!("../../../../examples/rtx3090_lab_cluster_measured_curves.toml");

fn qwen7b() -> ModelSpec {
    ModelSpec {
        layers: 28,
        hidden_size: 3584,
        attention_heads: 28,
        kv_heads: 4,
        vocab_size: 152_064,
        parameters: Bytes::from_gigabytes(15.23),
        parameter_count: Some(7.615e9),
        parameter_count_source: crate::workload::ParameterCountSource::Explicit,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn calibration(sharing: KvLinkSharing, proxy_us: f64) -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.8494,
        decode_memory_bandwidth_scale: 0.8383,
        kv_link_sharing: sharing,
        disaggregation_proxy_latency_us: proxy_us,
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

fn config(arrival: ServingArrivalPattern) -> DisaggregatedServingConfig {
    DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::FullyDisaggregated,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
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
        traffic: ServingTraffic {
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
        },
        slo_policies: Vec::new(),
    }
}

fn simulate(
    arrival: ServingArrivalPattern,
    calibration: SimulationCalibration,
) -> ScoredServingConfig {
    let request = InferenceRequest {
        batch_size: 1,
        prompt_tokens: 512,
        decode_tokens: 128,
        max_sequence_tokens: 640,
        phase: InferencePhase::EndToEnd,
    };
    let cluster = parse_cluster(LAB_CLUSTER).expect("lab cluster parses");
    let result = ServingSolver::rank_disaggregated(
        &cluster,
        &qwen7b(),
        &request,
        &config(arrival),
        calibration,
    )
    .into_iter()
    .next()
    .expect("one serving result");
    assert!(result.feasible, "{:?}", result.rejected_reason);
    assert_eq!(result.metrics.completed_requests, 200);
    result
}

fn has_approximation(result: &ScoredServingConfig, code: &str) -> bool {
    result
        .approximations
        .iter()
        .any(|approximation| approximation.code == code)
}

fn poisson(rate_per_s: f64) -> ServingArrivalPattern {
    ServingArrivalPattern::Poisson {
        rate_per_s,
        seed: 0,
    }
}

#[test]
fn fair_share_burst_overlaps_pulls_at_the_same_aggregate_throughput() {
    let fair = simulate(
        ServingArrivalPattern::FixedGap,
        calibration(KvLinkSharing::FairShare, 0.0),
    );
    assert!(has_approximation(&fair, "kv_transfer_fair_share_links"));
    assert!(!has_approximation(&fair, "kv_transfer_fifo_link_queues"));
    let mut windows = fair
        .request_observations
        .iter()
        .map(|observation| (observation.kv_start_s, observation.kv_finish_s))
        .collect::<Vec<_>>();
    windows.sort_by(|left, right| left.0.total_cmp(&right.0));
    let overlapping = windows
        .windows(2)
        .filter(|pair| pair[1].0 + 1e-9 < pair[0].1)
        .count();
    assert!(overlapping > 50, "only {overlapping} overlapping pulls");
    // Contention shows as resource queueing on top of the ~81 ms price.
    let max_contention_s = fair
        .request_observations
        .iter()
        .map(|observation| observation.kv_resource_queue_s)
        .fold(0.0, f64::max);
    assert!(max_contention_s > 0.15, "max contention {max_contention_s}");
    for observation in &fair.request_observations {
        let sojourn_s = observation.kv_finish_s - observation.kv_start_s;
        assert!(
            (sojourn_s - observation.kv_transfer_s - observation.kv_resource_queue_s).abs() < 1e-9
        );
    }
    // The link moves the same work: the last pull ends near FIFO's.
    let fifo = simulate(
        ServingArrivalPattern::FixedGap,
        calibration(KvLinkSharing::Fifo, 0.0),
    );
    let last = |result: &ScoredServingConfig| {
        result
            .request_observations
            .iter()
            .map(|observation| observation.kv_finish_s)
            .fold(0.0, f64::max)
    };
    let ratio = last(&fair) / last(&fifo);
    assert!((0.9..1.1).contains(&ratio), "last pull ratio {ratio}");
}

#[test]
fn fair_share_leaves_light_load_ttft_unchanged() {
    let fifo = simulate(poisson(1.0), calibration(KvLinkSharing::Fifo, 0.0));
    let fair = simulate(poisson(1.0), calibration(KvLinkSharing::FairShare, 0.0));
    let delta_ms = (fair.metrics.ttft_p50_s - fifo.metrics.ttft_p50_s).abs() * 1e3;
    assert!(delta_ms < 1.0, "TTFT p50 moved {delta_ms} ms");
    assert!((fair.metrics.tpot_p50_s - fifo.metrics.tpot_p50_s).abs() < 2e-4);
}

#[test]
fn proxy_latency_adds_to_light_load_ttft_only() {
    let base = simulate(poisson(1.0), calibration(KvLinkSharing::FairShare, 0.0));
    let proxied = simulate(
        poisson(1.0),
        calibration(KvLinkSharing::FairShare, 40_000.0),
    );
    assert!(has_approximation(
        &base,
        "disaggregated_proxy_hop_not_modeled"
    ));
    assert!(has_approximation(
        &proxied,
        "disaggregated_constant_proxy_latency"
    ));
    let added_ms = (proxied.metrics.ttft_p50_s - base.metrics.ttft_p50_s) * 1e3;
    assert!((35.0..45.0).contains(&added_ms), "TTFT grew {added_ms} ms");
    assert!((proxied.metrics.tpot_p50_s - base.metrics.tpot_p50_s).abs() < 2e-4);
    for observation in &proxied.request_observations {
        assert!(observation.kv_start_s >= observation.prefill_finish_s + 0.04 - 1e-9);
    }
}
