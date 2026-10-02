use super::super::*;
use super::*;
use crate::config::parse_cluster;
use crate::types::collective::CollectiveKind;
use crate::types::collective_curves::{CurveLookup, CurveQuery};

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9 * left.abs().max(right.abs()).max(1.0)
}

fn flows(src: KvShardLayout, dst: KvShardLayout) -> Vec<(usize, usize, f64)> {
    kv_shard_flows(src, dst)
        .into_iter()
        .map(|flow| (flow.src_rank, flow.dst_rank, flow.fraction))
        .collect()
}

fn total_fraction(src: KvShardLayout, dst: KvShardLayout) -> f64 {
    kv_shard_flows(src, dst)
        .iter()
        .map(|flow| flow.fraction)
        .sum()
}

// ---- shards ----

#[test]
fn single_gpu_to_single_gpu_moves_everything_in_one_flow() {
    let one = KvShardLayout::new(1, 1, 1, 4);
    assert_eq!(flows(one, one), vec![(0, 0, 1.0)]);
}

#[test]
fn matching_tensor_layouts_move_shard_to_shard() {
    let tp2 = KvShardLayout::new(2, 1, 1, 4);
    assert_eq!(flows(tp2, tp2), vec![(0, 0, 0.5), (1, 1, 0.5)]);
}

#[test]
fn tensor_parallel_prefill_gathers_into_one_decode_gpu() {
    let tp2 = KvShardLayout::new(2, 1, 1, 4);
    let tp1 = KvShardLayout::new(1, 1, 1, 4);
    assert_eq!(flows(tp2, tp1), vec![(0, 0, 0.5), (1, 0, 0.5)]);
    assert_eq!(flows(tp1, tp2), vec![(0, 0, 0.5), (0, 1, 0.5)]);
}

#[test]
fn pipeline_stages_split_by_layers() {
    let pp2 = KvShardLayout::new(1, 2, 1, 4);
    let tp1 = KvShardLayout::new(1, 1, 1, 4);
    assert_eq!(flows(pp2, tp1), vec![(0, 0, 0.5), (1, 0, 0.5)]);
    // TP=2 prefill into PP=2 decode: every (head half, layer half) block moves once.
    let tp2 = KvShardLayout::new(2, 1, 1, 4);
    assert_eq!(
        flows(tp2, pp2),
        vec![(0, 0, 0.25), (0, 1, 0.25), (1, 0, 0.25), (1, 1, 0.25)]
    );
    assert!(close(total_fraction(tp2, pp2), 1.0));
}

#[test]
fn replicated_destination_heads_are_each_transferred() {
    // 4 KV heads on 8 tensor ranks: every head lives on two decode ranks.
    let tp1 = KvShardLayout::new(1, 1, 1, 4);
    let tp8 = KvShardLayout::new(8, 1, 1, 4);
    assert!(close(total_fraction(tp1, tp8), 2.0));
    for flow in kv_shard_flows(tp1, tp8) {
        assert!(close(flow.fraction, 0.25));
    }
}

#[test]
fn replicated_source_heads_are_read_from_one_replica_each() {
    let tp8 = KvShardLayout::new(8, 1, 1, 4);
    let tp4 = KvShardLayout::new(4, 1, 1, 4);
    let moved = flows(tp8, tp4);
    assert!(close(total_fraction(tp8, tp4), 1.0));
    assert_eq!(moved.len(), 4);
    // Decode rank d reads head d from replica d % 2 of {2d, 2d + 1}.
    assert_eq!(
        moved.iter().map(|flow| (flow.0, flow.1)).collect::<Vec<_>>(),
        vec![(0, 0), (3, 1), (4, 2), (7, 3)]
    );
}

// ---- link queues ----

fn plan(flows: &[(&[LinkId], f64)]) -> KvTransferPlan {
    KvTransferPlan::new(
        flows
            .iter()
            .map(|(links, service_s)| KvFlow::new(links.to_vec(), *service_s).expect("valid flow"))
            .collect(),
        1,
    )
}

#[test]
fn flow_rejects_invalid_service_times() {
    assert!(KvFlow::new(vec![0], -1.0).is_err());
    assert!(KvFlow::new(vec![0], f64::NAN).is_err());
    assert!(KvFlow::new(vec![0], f64::INFINITY).is_err());
    assert!(KvFlow::new(vec![0], 0.0).is_ok());
}

#[test]
fn idle_network_starts_a_transfer_when_ready() {
    let mut queues = LinkQueues::new(2);
    let window = queues.reserve(&plan(&[(&[0, 1], 0.08)]), 1.0, 0);
    assert!(close(window.start_s, 1.0));
    assert!(close(window.finish_s, 1.08));
    assert!(window.predecessors.is_empty());
    assert!(queues.busy_at(0, 1.05));
    assert!(!queues.busy_at(0, 1.09));
}

#[test]
fn transfers_sharing_a_link_serialize_in_fifo_order() {
    let mut queues = LinkQueues::new(3);
    let first = queues.reserve(&plan(&[(&[0, 1], 0.1)]), 0.0, 0);
    // Shares link 1 with the first transfer.
    let second = queues.reserve(&plan(&[(&[1, 2], 0.1)]), 0.02, 1);
    assert!(close(first.finish_s, 0.1));
    assert!(close(second.start_s, 0.1));
    assert!(close(second.finish_s, 0.2));
    assert_eq!(second.predecessors, vec![0]);
    // Throughput over the shared link equals its bandwidth: two 0.1 s
    // transfers take 0.2 s back to back.
}

#[test]
fn transfers_on_disjoint_links_overlap() {
    let mut queues = LinkQueues::new(2);
    let first = queues.reserve(&plan(&[(&[0], 0.1)]), 0.0, 0);
    let second = queues.reserve(&plan(&[(&[1], 0.1)]), 0.0, 1);
    assert!(close(first.finish_s, 0.1));
    assert!(close(second.finish_s, 0.1));
    assert!(second.predecessors.is_empty());
}

#[test]
fn opposite_directions_of_a_link_do_not_contend() {
    // Directed resources: 0 = node0 -> node1, 1 = node1 -> node0.
    let mut queues = LinkQueues::new(2);
    let forward = queues.reserve(&plan(&[(&[0], 0.1)]), 0.0, 0);
    let backward = queues.reserve(&plan(&[(&[1], 0.03)]), 0.0, 1);
    assert!(close(forward.finish_s, 0.1));
    assert!(close(backward.finish_s, 0.03));
}

#[test]
fn flows_of_one_transfer_on_a_shared_link_serialize() {
    let mut queues = LinkQueues::new(1);
    let window = queues.reserve(&plan(&[(&[0], 0.05), (&[0], 0.05)]), 0.0, 0);
    assert!(close(window.start_s, 0.0));
    assert!(close(window.finish_s, 0.1));
    assert_eq!(window.flows.len(), 2);
    assert!(window.predecessors.is_empty());
}

#[test]
fn empty_plan_completes_instantly() {
    let mut queues = LinkQueues::new(0);
    let window = queues.reserve(&KvTransferPlan::default(), 2.5, 0);
    assert!(close(window.start_s, 2.5));
    assert!(close(window.finish_s, 2.5));
    assert!(close(KvTransferPlan::default().uncontended_s(), 0.0));
}

#[test]
fn uncontended_duration_overlaps_disjoint_flows() {
    let parallel = plan(&[(&[0], 0.05), (&[1], 0.07)]);
    assert!(close(parallel.uncontended_s(), 0.07));
    let shared = plan(&[(&[0], 0.05), (&[0], 0.07)]);
    assert!(close(shared.uncontended_s(), 0.12));
    assert!(close(shared.scaled(2.0).uncontended_s(), 0.24));
}

// ---- plans over a lab-like cluster ----

const LAB_NODES: &str = r#"
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
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, egress_bandwidth_gbps = 3.61, rail_count = 1 }
[[nodes]]
id = 1
gpu = "a100_40gb"
gpu_count = 1
hbm_gb = 24.0
hbm_bandwidth_gb_s = 936.0
peak_f16_tflops = 71.0
intra = "pcie_gen4"
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
"#;

const SEND_CURVES: &str = r#"
[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
points = [[1048576, 3085.14], [16777216, 46267.16], [33554432, 92344.96], [67108864, 194939.89]]
[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 1
dst_node = 0
points = [[1048576, 1098.30], [16777216, 14477.70], [33554432, 28766.04], [67108864, 57301.68]]
"#;

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

fn gpu(node_id: NodeId) -> GpuAddr {
    GpuAddr {
        node_id,
        local_gpu_id: 0,
    }
}

fn single_gpu(node_id: NodeId) -> PlacedKvLayout {
    PlacedKvLayout::new(KvShardLayout::new(1, 1, 1, 4), vec![gpu(node_id)]).expect("layout")
}

fn planner(cluster: &Cluster) -> KvTransferPlanner<'_> {
    KvTransferPlanner::new(KvPlanContext {
        cluster,
        calibration: SimulationCalibration::default(),
        calibration_profile: None,
        bytes_per_token: KvPlanContext::kv_bytes_per_token(&qwen7b()),
    })
}

const PROMPT_KV_BYTES: u64 = 512 * 28 * 4 * 128 * 2 * 2;

#[test]
fn prompt_kv_bytes_follow_layers_heads_and_dtype() {
    assert_eq!(KvPlanContext::kv_bytes_per_token(&qwen7b()), 57_344);
    let cluster = parse_cluster(LAB_NODES).expect("cluster");
    let planned = planner(&cluster)
        .plan(&single_gpu(0), &single_gpu(1), 1, 512)
        .expect("plan");
    assert_eq!(planned.plan.bytes(), PROMPT_KV_BYTES);
    assert_eq!(planned.plan.flows().len(), 1);
    assert_eq!(planned.paths.len(), 1);
    assert_eq!(planned.paths[0].source, gpu(0));
    assert_eq!(planned.paths[0].destination, gpu(1));
    // Two lockstep sequences move twice the bytes.
    let doubled = planner(&cluster)
        .plan(&single_gpu(0), &single_gpu(1), 2, 512)
        .expect("plan");
    assert_eq!(doubled.plan.bytes(), 2 * PROMPT_KV_BYTES);
}

#[test]
fn alpha_beta_transfer_uses_the_slower_egress_direction() {
    let cluster = parse_cluster(LAB_NODES).expect("cluster");
    let mut planner = planner(&cluster);
    let forward = planner
        .plan(&single_gpu(0), &single_gpu(1), 1, 512)
        .expect("forward");
    let backward = planner
        .plan(&single_gpu(1), &single_gpu(0), 1, 512)
        .expect("backward");
    assert_eq!(forward.pricing, vec![FlowPricing::AlphaBeta]);
    let forward_s = forward.plan.uncontended_s();
    let backward_s = backward.plan.uncontended_s();
    // 29.4 MB at 3.61 Gb/s is ~65 ms; at 9.41 Gb/s ~25 ms.
    let expected_forward_s = PROMPT_KV_BYTES as f64 * 8.0 / 3.61e9;
    assert!(
        (forward_s / expected_forward_s - 1.0).abs() < 0.05,
        "forward {forward_s}"
    );
    assert!(forward_s > 2.0 * backward_s, "{forward_s} vs {backward_s}");
    // Opposite directions are distinct queue resources.
    assert!(
        forward
            .resources
            .iter()
            .all(|resource| !backward.resources.contains(resource))
    );
    assert!(
        forward
            .bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "KV transfer fabric/NIC path")
    );
}

#[test]
fn measured_send_recv_curve_prices_the_directed_pair() {
    let cluster = parse_cluster(&format!("{LAB_NODES}{SEND_CURVES}")).expect("cluster");
    let mut planner = planner(&cluster);
    let forward = planner
        .plan(&single_gpu(0), &single_gpu(1), 1, 512)
        .expect("forward");
    let backward = planner
        .plan(&single_gpu(1), &single_gpu(0), 1, 512)
        .expect("backward");
    let expected = |src: NodeId, dst: NodeId| match cluster.collective_curves.lookup(&CurveQuery {
        kind: CollectiveKind::SendRecv,
        participant_nodes: &[src, dst],
    }) {
        CurveLookup::Matched(curve) => curve.curve().evaluate(PROMPT_KV_BYTES).latency_s,
        _ => panic!("curve configured"),
    };
    assert!(close(forward.plan.uncontended_s(), expected(0, 1)));
    assert!(close(backward.plan.uncontended_s(), expected(1, 0)));
    // ~0.36 GB/s forward: 29.4 MB in roughly 80 ms.
    assert!((0.07..0.09).contains(&forward.plan.uncontended_s()));
    assert!(matches!(
        forward.pricing.as_slice(),
        [FlowPricing::MeasuredCurve { .. }]
    ));
}

#[test]
fn curves_ignore_calibration_scalars_but_alpha_beta_honours_them() {
    let curved = parse_cluster(&format!("{LAB_NODES}{SEND_CURVES}")).expect("cluster");
    let plain = parse_cluster(LAB_NODES).expect("cluster");
    let scaled = SimulationCalibration {
        kv_transfer_scale: 2.0,
        ..SimulationCalibration::default()
    };
    let plan_with = |cluster: &Cluster, calibration| {
        KvTransferPlanner::new(KvPlanContext {
            cluster,
            calibration,
            calibration_profile: None,
            bytes_per_token: KvPlanContext::kv_bytes_per_token(&qwen7b()),
        })
        .plan(&single_gpu(0), &single_gpu(1), 1, 512)
        .expect("plan")
        .plan
        .uncontended_s()
    };
    assert!(close(
        plan_with(&curved, scaled),
        plan_with(&curved, SimulationCalibration::default())
    ));
    assert!(
        plan_with(&plain, scaled) > 1.9 * plan_with(&plain, SimulationCalibration::default())
    );
}

#[test]
fn same_gpu_layouts_move_nothing() {
    let cluster = parse_cluster(LAB_NODES).expect("cluster");
    let planned = planner(&cluster)
        .plan(&single_gpu(0), &single_gpu(0), 1, 512)
        .expect("plan");
    assert!(planned.plan.flows().is_empty());
    assert_eq!(planned.plan.bytes(), 0);
}

#[test]
fn layout_rank_count_must_match_gpus() {
    assert_eq!(
        PlacedKvLayout::new(KvShardLayout::new(2, 1, 1, 4), vec![gpu(0)]),
        Err(KvPlanError::RankCountMismatch {
            expected: 2,
            actual: 1
        })
    );
}
