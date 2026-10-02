//! Solver-level tests for measured collective curves and one-way-asymmetric
//! links: which model prices each collective/transfer, which direction a
//! routed transfer uses, and the evidence left behind.

use super::*;
use crate::{config::parse_cluster, workload::DType};

const TWO_NODE_BASE: &str = r#"
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
intra = "pcie_gen4"
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
[[nodes]]
id = 1
gpu = "a100_40gb"
gpu_count = 1
intra = "pcie_gen4"
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
"#;

const ALL_REDUCE_CURVE: &str = r#"
[[collective_curves]]
op = "all_reduce"
scope = "node_group"
nodes = [0, 1]
ranks = 2
source = "test"
points = [[1024, 130.0], [65536, 370.0], [1048576, 1800.0]]
"#;

const ALL_GATHER_CURVE: &str = r#"
[[collective_curves]]
op = "all_gather"
scope = "node_group"
nodes = [0, 1]
ranks = 2
points = [[1024, 90.0], [1048576, 1000.0]]
"#;

fn send_curves(forward_floor_us: f64, reverse_floor_us: f64) -> String {
    format!(
        r#"
[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
points = [[1024, {forward_floor_us}], [67108864, 190000.0]]

[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 1
dst_node = 0
points = [[1024, {reverse_floor_us}], [67108864, 57000.0]]
"#
    )
}

fn cluster(extra: &str) -> Cluster {
    parse_cluster(&format!("{TWO_NODE_BASE}{extra}")).expect("cluster")
}

fn model() -> ModelSpec {
    ModelSpec {
        layers: 4,
        hidden_size: 4096,
        attention_heads: 32,
        kv_heads: 8,
        vocab_size: 32000,
        parameters: Bytes::from_gigabytes(4.0),
        parameter_count: None,
        parameter_count_source: crate::workload::ParameterCountSource::Explicit,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn request(phase: InferencePhase, batch_size: u32) -> InferenceRequest {
    InferenceRequest {
        batch_size,
        prompt_tokens: 128,
        decode_tokens: 16,
        max_sequence_tokens: 256,
        phase,
    }
}

fn config(tensor_ranks: u32, pipeline_ranks: u32) -> ParallelismConfig {
    ParallelismConfig {
        tensor_ranks,
        pipeline_ranks,
        expert_ranks: 1,
        data_ranks: 1,
    }
}

fn score(
    cluster: &Cluster,
    request: &InferenceRequest,
    config: ParallelismConfig,
) -> ScoredParallelismConfig {
    score_with(cluster, request, config, SimulationCalibration::default())
}

fn score_with(
    cluster: &Cluster,
    request: &InferenceRequest,
    config: ParallelismConfig,
    calibration: SimulationCalibration,
) -> ScoredParallelismConfig {
    let scored = Solver::score_config_with_options(
        cluster,
        &model(),
        request,
        config,
        SolverOptions {
            calibration,
            ..SolverOptions::default()
        },
    );
    assert!(scored.feasible, "{:?}", scored.rejected_reason);
    scored
}

fn op_duration(scored: &ScoredParallelismConfig, name: &str) -> f64 {
    scored
        .operations
        .iter()
        .find(|op| op.name == name)
        .unwrap_or_else(|| panic!("missing operation {name}"))
        .duration_s
}

fn codes(scored: &ScoredParallelismConfig) -> Vec<&str> {
    scored
        .approximations
        .iter()
        .map(|approximation| approximation.code.as_str())
        .collect()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs() * 1e-9,
        "expected {expected}, got {actual}"
    );
}

fn decode_token_bytes(batch: u32) -> u64 {
    u64::from(batch) * u64::from(model().hidden_size) * 2
}

#[test]
fn tensor_parallel_all_reduce_is_priced_from_the_matching_curve() {
    let cluster = cluster(&format!("{ALL_REDUCE_CURVE}{ALL_GATHER_CURVE}"));
    let request = request(InferencePhase::Decode, 8);
    let scored = score(&cluster, &request, config(2, 1));
    let curve = cluster.collective_curves.curves()[0].curve();
    let per_step = curve.evaluate(decode_token_bytes(8)).latency_s;
    close(
        op_duration(&scored, "layer 0 tp all-reduce attn group 0"),
        per_step * f64::from(request.decode_tokens),
    );
    let codes = codes(&scored);
    assert!(codes.contains(&"measured_collective_curve"), "{codes:?}");
    assert!(!codes.contains(&"coarse_collective_model"), "{codes:?}");
    assert!(!codes.contains(&"collective_curve_absent"), "{codes:?}");
    assert!(
        !codes.contains(&"collective_curve_extrapolated"),
        "{codes:?}"
    );
}

#[test]
fn tensor_parallel_traces_embedding_all_reduce_and_logits_all_gather() {
    let cluster = cluster(&format!("{ALL_REDUCE_CURVE}{ALL_GATHER_CURVE}"));
    let request = request(InferencePhase::Decode, 8);
    let scored = score(&cluster, &request, config(2, 1));
    let curves = cluster.collective_curves.curves();
    let decode_steps = f64::from(request.decode_tokens);
    close(
        op_duration(&scored, "embedding tp all-reduce group 0"),
        curves[0].curve().evaluate(decode_token_bytes(8)).latency_s * decode_steps,
    );
    // Each rank contributes its vocab shard of the logits for every sequence.
    let shard_bytes = 8 * u64::from(model().vocab_size) / 2 * 2;
    close(
        op_duration(&scored, "lm_head logits all-gather group 0"),
        curves[1].curve().evaluate(shard_bytes).latency_s * decode_steps,
    );
    let embedding = operation_position(&scored, "embedding tp all-reduce group 0");
    let first_compute = operation_position(&scored, "layer 0 compute");
    assert!(
        scored.operations[first_compute]
            .dependencies
            .contains(&embedding)
    );
    let gather = operation_position(&scored, "lm_head logits all-gather group 0");
    let last_reduce = operation_position(&scored, "layer 3 tp all-reduce mlp group 0");
    assert_eq!(scored.operations[gather].dependencies, vec![last_reduce]);

    // Single-rank configs have neither.
    let single = score(&cluster, &request, config(1, 1));
    assert!(
        single
            .operations
            .iter()
            .all(|op| !op.name.contains("embedding") && !op.name.contains("lm_head"))
    );
}

fn operation_position(scored: &ScoredParallelismConfig, name: &str) -> usize {
    scored
        .operations
        .iter()
        .position(|op| op.name == name)
        .unwrap_or_else(|| panic!("missing operation {name}"))
}

#[test]
fn curves_are_not_rescaled_by_collective_calibration() {
    let measured = cluster(ALL_REDUCE_CURVE);
    let request = request(InferencePhase::Decode, 8);
    let base = score(&measured, &request, config(2, 1));
    let scaled = score_with(
        &measured,
        &request,
        config(2, 1),
        SimulationCalibration {
            collective_latency_scale: 3.0,
            collective_bandwidth_scale: 0.5,
            ..SimulationCalibration::default()
        },
    );
    let name = "layer 0 tp all-reduce mlp group 0";
    close(op_duration(&scaled, name), op_duration(&base, name));

    // Without curves the same scalars do change the alpha-beta estimate.
    let plain = cluster("");
    let plain_base = score(&plain, &request, config(2, 1));
    let plain_scaled = score_with(
        &plain,
        &request,
        config(2, 1),
        SimulationCalibration {
            collective_latency_scale: 3.0,
            ..SimulationCalibration::default()
        },
    );
    assert!(op_duration(&plain_scaled, name) > op_duration(&plain_base, name));
}

#[test]
fn prefill_extrapolates_above_the_curve_and_reports_it() {
    let cluster = cluster(ALL_REDUCE_CURVE);
    let request = request(InferencePhase::Prefill, 8);
    let scored = score(&cluster, &request, config(2, 1));
    let bytes = decode_token_bytes(8) * u64::from(request.prompt_tokens);
    assert!(bytes > 1_048_576);
    let curve = cluster.collective_curves.curves()[0].curve();
    close(
        op_duration(&scored, "layer 1 tp all-reduce attn group 0"),
        curve.evaluate(bytes).latency_s,
    );
    let extrapolated = scored
        .approximations
        .iter()
        .find(|approximation| approximation.code == "collective_curve_extrapolated")
        .expect("extrapolation evidence");
    assert!(extrapolated.message.contains("above the measured range"));
    assert!(extrapolated.message.contains(&format!("at {bytes} B")));
}

#[test]
fn unmatched_collectives_fall_back_to_alpha_beta_with_evidence() {
    let curve_only_for_four_ranks = r#"
[[collective_curves]]
op = "all_reduce"
scope = "inter_node"
ranks = 4
points = [[1024, 130.0], [1048576, 1800.0]]
"#;
    let with_curve = cluster(curve_only_for_four_ranks);
    let plain = cluster("");
    let request = request(InferencePhase::Decode, 8);
    let scored = score(&with_curve, &request, config(2, 1));
    let baseline = score(&plain, &request, config(2, 1));
    close(scored.estimated_latency_s, baseline.estimated_latency_s);
    let codes = codes(&scored);
    assert!(codes.contains(&"coarse_collective_model"), "{codes:?}");
    assert!(codes.contains(&"collective_curve_absent"), "{codes:?}");
    let absent = scored
        .approximations
        .iter()
        .find(|approximation| approximation.code == "collective_curve_absent")
        .expect("absent evidence");
    assert!(absent.message.contains("all_reduce nodes=0+1 ranks=2"));
}

#[test]
fn clusters_without_curves_keep_the_original_evidence() {
    let scored = score(
        &cluster(""),
        &request(InferencePhase::Decode, 8),
        config(2, 1),
    );
    let codes = codes(&scored);
    assert!(codes.contains(&"coarse_collective_model"), "{codes:?}");
    assert!(
        !codes.iter().any(|code| code.contains("curve")),
        "{codes:?}"
    );
}

#[test]
fn pipeline_sends_use_the_curve_for_their_direction() {
    let request = request(InferencePhase::Decode, 8);
    let curves = cluster(&send_curves(400.0, 150.0));
    let scored = score(&curves, &request, config(1, 2));
    // Stage 0 sits on node 0, so activations travel node0 -> node1.
    let forward = curves
        .collective_curves
        .curves()
        .iter()
        .find(|curve| curve.label() == "send_recv node0->node1")
        .expect("forward curve")
        .curve()
        .evaluate(decode_token_bytes(8))
        .latency_s;
    close(
        op_duration(&scored, "pipeline sendrecv edge 0"),
        forward * f64::from(request.decode_tokens),
    );
    // Making only the reverse direction slow leaves the forward send as is;
    // making the forward direction fast speeds it up.
    let slow_reverse = score(&cluster(&send_curves(400.0, 900.0)), &request, config(1, 2));
    close(
        op_duration(&slow_reverse, "pipeline sendrecv edge 0"),
        op_duration(&scored, "pipeline sendrecv edge 0"),
    );
    let fast_forward = score(&cluster(&send_curves(150.0, 400.0)), &request, config(1, 2));
    assert!(
        op_duration(&fast_forward, "pipeline sendrecv edge 0")
            < op_duration(&scored, "pipeline sendrecv edge 0")
    );
}

#[test]
fn derived_curve_regions_are_reported() {
    let derived = r#"
[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
derived_below_bytes = 67108864
points = [[1024, 160.0], [67108864, 190000.0], [134217728, 380000.0]]
"#;
    let scored = score(
        &cluster(derived),
        &request(InferencePhase::Decode, 8),
        config(1, 2),
    );
    let codes = codes(&scored);
    assert!(
        codes.contains(&"collective_curve_derived_region"),
        "{codes:?}"
    );
}

fn nic_cluster(node0_caps: &str, node1_caps: &str) -> Cluster {
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
intra = "pcie_gen4"
nics = {{ count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1{node0_caps} }}
[[nodes]]
id = 1
gpu = "a100_40gb"
gpu_count = 1
intra = "pcie_gen4"
nics = {{ count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1{node1_caps} }}
"#
    ))
    .expect("cluster")
}

fn node0_egress_capped() -> Cluster {
    nic_cluster(", egress_bandwidth_gbps = 3.61", "")
}

fn slow_both_ways() -> Cluster {
    cluster("").clone_with_node0_nic_gbps(3.61)
}

trait NicOverride {
    fn clone_with_node0_nic_gbps(self, gbps: f64) -> Cluster;
}

impl NicOverride for Cluster {
    fn clone_with_node0_nic_gbps(mut self, gbps: f64) -> Cluster {
        if let Some(node) = self.nodes.get_mut(&0) {
            node.network.nic_bandwidth = Bandwidth::from_gigabits_per_sec(gbps);
        }
        self
    }
}

fn transfer_s(cluster: &Cluster, src: NodeId, dst: NodeId, bytes: u64) -> f64 {
    Solver::estimate_transfer_between_nodes(
        cluster,
        &[src],
        &[dst],
        Bytes::from_bytes(bytes),
        SimulationCalibration::default(),
    )
    .total_s
}

#[test]
fn asymmetric_nic_slows_only_the_egress_direction() {
    let asymmetric = node0_egress_capped();
    let symmetric = cluster("");
    let slow = slow_both_ways();
    let bytes = 64 * 1024 * 1024;
    let out_of_node0 = transfer_s(&asymmetric, 0, 1, bytes);
    let into_node0 = transfer_s(&asymmetric, 1, 0, bytes);
    close(out_of_node0, transfer_s(&slow, 0, 1, bytes));
    close(into_node0, transfer_s(&symmetric, 1, 0, bytes));
    assert!(out_of_node0 > 2.0 * into_node0);

    let gpu = |node_id| GpuAddr {
        node_id,
        local_gpu_id: 0,
    };
    let gpu_transfer = |src, dst| {
        Solver::estimate_transfer_between_gpus_with_options(
            &asymmetric,
            &[gpu(src)],
            &[gpu(dst)],
            Bytes::from_bytes(bytes),
            SolverOptions::default(),
        )
        .total_s
    };
    assert!(gpu_transfer(0, 1) > 2.0 * gpu_transfer(1, 0));
}

#[test]
fn ring_collectives_are_bounded_by_the_slower_direction() {
    let asymmetric = node0_egress_capped();
    let slow = slow_both_ways();
    let symmetric = cluster("");
    let request = request(InferencePhase::Prefill, 8);
    let name = "layer 0 tp all-reduce attn group 0";
    let asymmetric_s = op_duration(&score(&asymmetric, &request, config(2, 1)), name);
    close(
        asymmetric_s,
        op_duration(&score(&slow, &request, config(2, 1)), name),
    );
    assert!(asymmetric_s > op_duration(&score(&symmetric, &request, config(2, 1)), name));
}

#[test]
fn pipeline_sends_follow_the_asymmetric_direction() {
    let asymmetric = node0_egress_capped();
    let request = request(InferencePhase::Prefill, 8);
    let name = "pipeline sendrecv edge 0";
    let forward = op_duration(&score(&asymmetric, &request, config(1, 2)), name);
    close(
        forward,
        op_duration(&score(&slow_both_ways(), &request, config(1, 2)), name),
    );
    // Ingress caps apply to the receiving side: capping node 1's ingress
    // instead yields the same node0 -> node1 send.
    let ingress = nic_cluster("", ", ingress_bandwidth_gbps = 3.61");
    assert!(ingress.nodes[&0].network.nic_direction_caps.is_empty());
    assert!(!ingress.nodes[&1].network.nic_direction_caps.is_empty());
    close(
        op_duration(&score(&ingress, &request, config(1, 2)), name),
        forward,
    );
}

#[test]
fn rejects_invalid_nic_caps() {
    let error = parse_cluster(&TWO_NODE_BASE.replacen(
        "bandwidth_gbps = 9.41, rail_count = 1",
        "bandwidth_gbps = 9.41, egress_bandwidth_gbps = 0.0, rail_count = 1",
        1,
    ))
    .expect_err("zero cap");
    assert!(
        error
            .to_string()
            .contains("egress_bandwidth_gbps must be finite and positive")
    );
}

const CUSTOM_LINK_BASE: &str = r#"
schema_version = 1
[cluster]
preset = "custom"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 1
intra = "pcie_gen4"
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 100.0, rail_count = 1 }
[[nodes]]
id = 1
gpu = "a100_40gb"
gpu_count = 1
intra = "pcie_gen4"
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 100.0, rail_count = 1 }
[[interconnect.links]]
from = 0
to = 1
kind = "ethernet"
variant = "10g"
"#;

#[test]
fn asymmetric_custom_links_route_by_direction() {
    let symmetric = parse_cluster(CUSTOM_LINK_BASE).expect("cluster");
    let asymmetric = parse_cluster(&format!(
        "{CUSTOM_LINK_BASE}to_from_bandwidth_gbps = 2.5\nto_from_latency_us = 40.0\n"
    ))
    .expect("cluster");
    let bytes = 64 * 1024 * 1024;
    close(
        transfer_s(&asymmetric, 0, 1, bytes),
        transfer_s(&symmetric, 0, 1, bytes),
    );
    let reverse = transfer_s(&asymmetric, 1, 0, bytes);
    assert!(reverse > 3.0 * transfer_s(&symmetric, 1, 0, bytes));
    let InterNodeTopology::Custom(edges) = &asymmetric.inter_node_topology else {
        panic!("custom topology");
    };
    let link = &edges[&UnorderedPair::new(0, 1)][0];
    let reverse_profile = link.direction_profile(1, 0);
    assert!((reverse_profile.bandwidth.as_gigabits_per_sec() - 2.5).abs() < 1e-9);
    assert!((reverse_profile.latency.to_us() - 40.0).abs() < 1e-9);
    assert_eq!(link.slowest_direction(), reverse_profile);

    // A ring over the pair is bounded by the slow reverse direction.
    let request = request(InferencePhase::Prefill, 8);
    let name = "layer 0 tp all-reduce attn group 0";
    assert!(
        op_duration(&score(&asymmetric, &request, config(2, 1)), name)
            > 3.0 * op_duration(&score(&symmetric, &request, config(2, 1)), name)
    );
}

#[test]
fn rejects_invalid_link_direction_overrides() {
    let error = parse_cluster(&format!(
        "{CUSTOM_LINK_BASE}from_to_bandwidth_gbps = -1.0\n"
    ))
    .expect_err("negative");
    assert!(
        error
            .to_string()
            .contains("from_to_bandwidth_gbps must be finite and positive")
    );
    let error = parse_cluster(&format!("{CUSTOM_LINK_BASE}to_from_latency_us = -1.0\n"))
        .expect_err("negative latency");
    assert!(
        error
            .to_string()
            .contains("to_from_latency_us must be finite and non-negative")
    );
}
