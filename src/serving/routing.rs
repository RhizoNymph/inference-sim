use super::*;

pub(super) struct RequestRoute {
    pub(super) prefill_node: NodeId,
    pub(super) decode_node: NodeId,
    pub(super) prefill_route_nodes: Vec<NodeId>,
    pub(super) prefill_route_gpus: Vec<GpuAddr>,
    pub(super) decode_route_nodes: Vec<NodeId>,
    pub(super) decode_route_gpus: Vec<GpuAddr>,
    pub(super) routing: RoutingDecision,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RoutingDecision {
    pub(super) policy: ServingRoutingPolicy,
    pub(super) candidate_count: u32,
    pub(super) routable_candidate_count: u32,
    pub(super) candidates: Vec<ServingRouteCandidateObservation>,
    pub(super) estimated_e2el_s: f64,
    pub(super) estimated_kv_transfer_s: f64,
    pub(super) estimated_kv_resource_wait_s: f64,
    pub(super) estimated_prefill_wait_s: f64,
    pub(super) estimated_decode_wait_s: f64,
    pub(super) reason: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct RoutingLoad {
    pub(super) prefill_worker_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    pub(super) decode_worker_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    pub(super) kv_route_ready_s: BTreeMap<String, f64>,
}

pub(super) fn kv_transfer_bytes_for_routes(
    model: &ModelSpec,
    request: &InferenceRequest,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_gpus: &[GpuAddr],
    decode_gpus: &[GpuAddr],
) -> Bytes {
    let prefill_gpu_set: BTreeSet<_> = prefill_gpus.iter().copied().collect();
    let decode_gpu_set: BTreeSet<_> = decode_gpus.iter().copied().collect();
    if !prefill_gpu_set.is_empty() && prefill_gpu_set == decode_gpu_set {
        return Bytes::from_bytes(0);
    }

    let prefill: BTreeSet<_> = prefill_nodes.iter().copied().collect();
    let decode: BTreeSet<_> = decode_nodes.iter().copied().collect();
    if prefill_gpu_set.is_empty() && prefill == decode {
        return Bytes::from_bytes(0);
    }

    let head_dim = model.hidden_size / model.attention_heads.max(1);
    let bytes = request.batch_size as u64
        * request.prompt_tokens as u64
        * model.layers as u64
        * model.kv_heads as u64
        * head_dim as u64
        * 2
        * model.kv_dtype().bytes_per_element();

    Bytes::from_bytes(bytes)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route_request(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    effective_prefill_tokens: u32,
    traffic: &ServingTraffic,
    request_idx: u32,
    arrival_s: f64,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_resource_base_node: Option<NodeId>,
    prefill_placement_nodes: &[NodeId],
    decode_resource_base_node: Option<NodeId>,
    decode_placement_nodes: &[NodeId],
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
    routing_load: &mut RoutingLoad,
) -> RequestRoute {
    match traffic.routing_policy {
        ServingRoutingPolicy::RoundRobin => {
            let prefill_node = routed_serving_node(prefill_nodes, request_idx);
            let decode_node = routed_serving_node(decode_nodes, request_idx);
            let prefill_route_nodes = routed_node_set(
                prefill_resource_base_node,
                prefill_placement_nodes,
                prefill_node,
            );
            let decode_route_nodes = routed_node_set(
                decode_resource_base_node,
                decode_placement_nodes,
                decode_node,
            );
            let prefill_route_gpus = routed_gpu_set(
                prefill_resource_base_node,
                &prefill_score.placement.rank_to_gpu,
                prefill_node,
            );
            let decode_route_gpus = routed_gpu_set(
                decode_resource_base_node,
                &decode_one_score.placement.rank_to_gpu,
                decode_node,
            );
            let candidate_count =
                route_candidate_count(prefill_nodes, decode_nodes).min(u32::MAX as usize) as u32;
            let routing_candidates = vec![round_robin_route_candidate_observation(
                cluster,
                model,
                request,
                &prefill_route_nodes,
                &decode_route_nodes,
                &prefill_route_gpus,
                &decode_route_gpus,
                calibration,
                calibration_profile,
            )];
            RequestRoute {
                prefill_node,
                decode_node,
                prefill_route_nodes,
                prefill_route_gpus,
                decode_route_nodes,
                decode_route_gpus,
                routing: RoutingDecision {
                    policy: ServingRoutingPolicy::RoundRobin,
                    candidate_count,
                    routable_candidate_count: candidate_count,
                    candidates: routing_candidates,
                    estimated_e2el_s: f64::INFINITY,
                    estimated_kv_transfer_s: f64::INFINITY,
                    estimated_kv_resource_wait_s: 0.0,
                    estimated_prefill_wait_s: 0.0,
                    estimated_decode_wait_s: 0.0,
                    reason: format!(
                        "round-robin request index {request_idx} selected prefill node {prefill_node} and decode node {decode_node}"
                    ),
                },
            }
        }
        ServingRoutingPolicy::TopologyAware => route_topology_aware_request(
            cluster,
            model,
            request,
            effective_prefill_tokens,
            arrival_s,
            prefill_nodes,
            decode_nodes,
            prefill_resource_base_node,
            prefill_placement_nodes,
            decode_resource_base_node,
            decode_placement_nodes,
            prefill_score,
            decode_one_score,
            decode_tail_scale,
            base_batch_size,
            base_prompt_tokens,
            prefill_worker_slots_per_gpu(traffic),
            decode_worker_slots_per_gpu(traffic),
            calibration,
            calibration_profile,
            routing_load,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route_topology_aware_request(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    effective_prefill_tokens: u32,
    arrival_s: f64,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_resource_base_node: Option<NodeId>,
    prefill_placement_nodes: &[NodeId],
    decode_resource_base_node: Option<NodeId>,
    decode_placement_nodes: &[NodeId],
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    prefill_worker_slots_per_gpu: usize,
    decode_worker_slots_per_gpu: usize,
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
    routing_load: &mut RoutingLoad,
) -> RequestRoute {
    let prefill_candidates = sorted_unique_nodes(prefill_nodes);
    let decode_candidates = sorted_unique_nodes(decode_nodes);
    let mut best: Option<(RequestRoute, RouteEstimate)> = None;
    let candidate_count = prefill_candidates
        .len()
        .saturating_mul(decode_candidates.len())
        .min(u32::MAX as usize) as u32;
    let mut routable_candidate_count = 0_u32;
    let mut routing_candidates = Vec::new();

    for &prefill_node in &prefill_candidates {
        let prefill_route_nodes = routed_node_set(
            prefill_resource_base_node,
            prefill_placement_nodes,
            prefill_node,
        );
        for &decode_node in &decode_candidates {
            let decode_route_nodes = routed_node_set(
                decode_resource_base_node,
                decode_placement_nodes,
                decode_node,
            );
            let prefill_route_gpus = routed_gpu_set(
                prefill_resource_base_node,
                &prefill_score.placement.rank_to_gpu,
                prefill_node,
            );
            let decode_route_gpus = routed_gpu_set(
                decode_resource_base_node,
                &decode_one_score.placement.rank_to_gpu,
                decode_node,
            );
            let kv_bytes = kv_transfer_bytes_for_routes(
                model,
                request,
                &prefill_route_nodes,
                &decode_route_nodes,
                &prefill_route_gpus,
                &decode_route_gpus,
            );
            if !Solver::transfer_between_gpus_routable(
                cluster,
                &prefill_route_gpus,
                &decode_route_gpus,
                kv_bytes,
            ) {
                routing_candidates.push(ServingRouteCandidateObservation {
                    prefill_node,
                    decode_node,
                    prefill_route_nodes: prefill_route_nodes.clone(),
                    prefill_route_gpus,
                    decode_route_nodes,
                    decode_route_gpus,
                    selected: false,
                    routable: false,
                    rejection_reason: Some(
                        "KV transfer path is not routable between routed prefill and decode GPUs"
                            .to_string(),
                    ),
                    estimated_e2el_s: None,
                    estimated_kv_transfer_s: None,
                    estimated_kv_resource_wait_s: None,
                    estimated_prefill_wait_s: None,
                    estimated_decode_wait_s: None,
                    kv_transfer_bytes: kv_bytes.as_bytes(),
                    kv_transfer_bottlenecks: Vec::new(),
                    kv_transfer_resources: Vec::new(),
                });
                continue;
            }
            routable_candidate_count = routable_candidate_count.saturating_add(1);
            let kv_cost = Solver::estimate_transfer_between_gpus_with_options(
                cluster,
                &prefill_route_gpus,
                &decode_route_gpus,
                kv_bytes,
                SolverOptions {
                    calibration,
                    calibration_profile,
                    max_candidates: None,
                    search_deadline: None,
                    explicit_placement: None,
                },
            );
            let kv_paths =
                kv_transfer_paths(cluster, &prefill_route_gpus, &decode_route_gpus, kv_bytes);
            let kv_resources = kv_transfer_scheduler_resources(&kv_paths, &kv_cost.bottlenecks);
            let estimate = route_estimate(
                request,
                effective_prefill_tokens,
                arrival_s,
                prefill_node,
                decode_node,
                &prefill_route_gpus,
                &decode_route_gpus,
                prefill_score,
                decode_one_score,
                decode_tail_scale,
                base_batch_size,
                base_prompt_tokens,
                prefill_worker_slots_per_gpu,
                decode_worker_slots_per_gpu,
                kv_cost.total_s,
                &kv_resources,
                routing_load,
            );
            routing_candidates.push(ServingRouteCandidateObservation {
                prefill_node,
                decode_node,
                prefill_route_nodes: prefill_route_nodes.clone(),
                prefill_route_gpus: prefill_route_gpus.clone(),
                decode_route_nodes: decode_route_nodes.clone(),
                decode_route_gpus: decode_route_gpus.clone(),
                selected: false,
                routable: true,
                rejection_reason: None,
                estimated_e2el_s: Some(estimate.e2el_proxy_s),
                estimated_kv_transfer_s: Some(estimate.kv_transfer_s),
                estimated_kv_resource_wait_s: Some(estimate.kv_resource_wait_s),
                estimated_prefill_wait_s: Some(estimate.prefill_wait_s),
                estimated_decode_wait_s: Some(estimate.decode_wait_s),
                kv_transfer_bytes: kv_bytes.as_bytes(),
                kv_transfer_bottlenecks: kv_cost.bottlenecks.clone(),
                kv_transfer_resources: kv_resources.clone(),
            });
            let route = RequestRoute {
                prefill_node,
                decode_node,
                prefill_route_nodes: prefill_route_nodes.clone(),
                prefill_route_gpus,
                decode_route_nodes,
                decode_route_gpus,
                routing: RoutingDecision {
                    policy: ServingRoutingPolicy::TopologyAware,
                    candidate_count,
                    routable_candidate_count,
                    candidates: Vec::new(),
                    estimated_e2el_s: estimate.e2el_proxy_s,
                    estimated_kv_transfer_s: estimate.kv_transfer_s,
                    estimated_kv_resource_wait_s: estimate.kv_resource_wait_s,
                    estimated_prefill_wait_s: estimate.prefill_wait_s,
                    estimated_decode_wait_s: estimate.decode_wait_s,
                    reason: String::new(),
                },
            };

            if best
                .as_ref()
                .map(|(_, best_estimate)| estimate.better_than(best_estimate))
                .unwrap_or(true)
            {
                best = Some((route, estimate));
            }
        }
    }

    let (route, estimate) = best.unwrap_or_else(|| {
        let prefill_node = routed_serving_node(prefill_nodes, 0);
        let decode_node = routed_serving_node(decode_nodes, 0);
        let prefill_route_gpus = routed_gpu_set(
            prefill_resource_base_node,
            &prefill_score.placement.rank_to_gpu,
            prefill_node,
        );
        let decode_route_gpus = routed_gpu_set(
            decode_resource_base_node,
            &decode_one_score.placement.rank_to_gpu,
            decode_node,
        );
        let route = RequestRoute {
            prefill_node,
            decode_node,
            prefill_route_nodes: routed_node_set(
                prefill_resource_base_node,
                prefill_placement_nodes,
                prefill_node,
            ),
            decode_route_nodes: routed_node_set(
                decode_resource_base_node,
                decode_placement_nodes,
                decode_node,
            ),
            prefill_route_gpus,
            decode_route_gpus,
            routing: RoutingDecision {
                policy: ServingRoutingPolicy::TopologyAware,
                candidate_count,
                routable_candidate_count,
                candidates: Vec::new(),
                estimated_e2el_s: 0.0,
                estimated_kv_transfer_s: 0.0,
                estimated_kv_resource_wait_s: 0.0,
                estimated_prefill_wait_s: 0.0,
                estimated_decode_wait_s: 0.0,
                reason: "topology-aware router found no routable candidates and fell back to deterministic first nodes".to_string(),
            },
        };
        (
            route,
            RouteEstimate {
                prefill_finish_s: arrival_s,
                decode_finish_s: arrival_s,
                e2el_proxy_s: 0.0,
                kv_transfer_s: 0.0,
                kv_resource_wait_s: 0.0,
                prefill_wait_s: 0.0,
                decode_wait_s: 0.0,
                kv_finish_s: arrival_s,
                kv_resources: Vec::new(),
                prefill_node,
                decode_node,
            },
        )
    });
    mark_selected_routing_candidate(&mut routing_candidates, &route);
    mark_route_workers_ready(
        &mut routing_load.prefill_worker_ready_s,
        &route.prefill_route_gpus,
        route.prefill_node,
        prefill_worker_slots_per_gpu,
        estimate.prefill_finish_s,
    );
    mark_route_workers_ready(
        &mut routing_load.decode_worker_ready_s,
        &route.decode_route_gpus,
        route.decode_node,
        decode_worker_slots_per_gpu,
        estimate.decode_finish_s,
    );
    mark_route_resources_ready(
        &mut routing_load.kv_route_ready_s,
        &estimate.kv_resources,
        estimate.kv_finish_s,
    );
    RequestRoute {
        routing: RoutingDecision {
            policy: route.routing.policy,
            candidate_count,
            routable_candidate_count,
            candidates: routing_candidates,
            estimated_e2el_s: estimate.e2el_proxy_s,
            estimated_kv_transfer_s: estimate.kv_transfer_s,
            estimated_kv_resource_wait_s: estimate.kv_resource_wait_s,
            estimated_prefill_wait_s: estimate.prefill_wait_s,
            estimated_decode_wait_s: estimate.decode_wait_s,
            reason: if routable_candidate_count == 0 {
                route.routing.reason
            } else {
                format!(
                    "topology-aware router selected prefill node {} and decode node {} from {}/{} routable candidates by estimated E2EL {:.6}s, KV {:.6}s, KV resource wait {:.6}s, prefill wait {:.6}s, decode wait {:.6}s",
                    route.prefill_node,
                    route.decode_node,
                    routable_candidate_count,
                    candidate_count,
                    estimate.e2el_proxy_s,
                    estimate.kv_transfer_s,
                    estimate.kv_resource_wait_s,
                    estimate.prefill_wait_s,
                    estimate.decode_wait_s
                )
            },
        },
        ..route
    }
}

pub(super) fn route_candidate_count(prefill_nodes: &[NodeId], decode_nodes: &[NodeId]) -> usize {
    sorted_unique_nodes(prefill_nodes)
        .len()
        .saturating_mul(sorted_unique_nodes(decode_nodes).len())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn round_robin_route_candidate_observation(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    prefill_route_nodes: &[NodeId],
    decode_route_nodes: &[NodeId],
    prefill_route_gpus: &[GpuAddr],
    decode_route_gpus: &[GpuAddr],
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
) -> ServingRouteCandidateObservation {
    let kv_transfer_bytes = kv_transfer_bytes_for_routes(
        model,
        request,
        prefill_route_nodes,
        decode_route_nodes,
        prefill_route_gpus,
        decode_route_gpus,
    );
    let routable = Solver::transfer_between_gpus_routable(
        cluster,
        prefill_route_gpus,
        decode_route_gpus,
        kv_transfer_bytes,
    );
    let (estimated_kv_transfer_s, kv_transfer_bottlenecks, kv_transfer_resources) = if routable {
        let kv_cost = Solver::estimate_transfer_between_gpus_with_options(
            cluster,
            prefill_route_gpus,
            decode_route_gpus,
            kv_transfer_bytes,
            SolverOptions {
                calibration,
                calibration_profile,
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );
        let kv_paths = kv_transfer_paths(
            cluster,
            prefill_route_gpus,
            decode_route_gpus,
            kv_transfer_bytes,
        );
        let kv_resources = kv_transfer_scheduler_resources(&kv_paths, &kv_cost.bottlenecks);
        (Some(kv_cost.total_s), kv_cost.bottlenecks, kv_resources)
    } else {
        (None, Vec::new(), Vec::new())
    };

    ServingRouteCandidateObservation {
        prefill_node: prefill_route_nodes.first().copied().unwrap_or(0),
        decode_node: decode_route_nodes.first().copied().unwrap_or(0),
        prefill_route_nodes: prefill_route_nodes.to_vec(),
        prefill_route_gpus: prefill_route_gpus.to_vec(),
        decode_route_nodes: decode_route_nodes.to_vec(),
        decode_route_gpus: decode_route_gpus.to_vec(),
        selected: true,
        routable,
        rejection_reason: (!routable).then(|| {
            "KV transfer path is not routable between routed prefill and decode GPUs".to_string()
        }),
        estimated_e2el_s: None,
        estimated_kv_transfer_s,
        estimated_kv_resource_wait_s: None,
        estimated_prefill_wait_s: None,
        estimated_decode_wait_s: None,
        kv_transfer_bytes: kv_transfer_bytes.as_bytes(),
        kv_transfer_bottlenecks,
        kv_transfer_resources,
    }
}

pub(super) fn mark_selected_routing_candidate(
    candidates: &mut [ServingRouteCandidateObservation],
    route: &RequestRoute,
) {
    for candidate in candidates {
        candidate.selected = candidate.prefill_node == route.prefill_node
            && candidate.decode_node == route.decode_node
            && candidate.prefill_route_nodes == route.prefill_route_nodes
            && candidate.prefill_route_gpus == route.prefill_route_gpus
            && candidate.decode_route_nodes == route.decode_route_nodes
            && candidate.decode_route_gpus == route.decode_route_gpus;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RouteEstimate {
    pub(super) prefill_finish_s: f64,
    pub(super) kv_finish_s: f64,
    pub(super) decode_finish_s: f64,
    pub(super) e2el_proxy_s: f64,
    pub(super) kv_transfer_s: f64,
    pub(super) kv_resource_wait_s: f64,
    pub(super) prefill_wait_s: f64,
    pub(super) decode_wait_s: f64,
    pub(super) kv_resources: Vec<String>,
    pub(super) prefill_node: NodeId,
    pub(super) decode_node: NodeId,
}

impl RouteEstimate {
    fn better_than(&self, other: &Self) -> bool {
        self.e2el_proxy_s
            .total_cmp(&other.e2el_proxy_s)
            .then_with(|| self.kv_transfer_s.total_cmp(&other.kv_transfer_s))
            .then_with(|| self.kv_resource_wait_s.total_cmp(&other.kv_resource_wait_s))
            .then_with(|| self.decode_wait_s.total_cmp(&other.decode_wait_s))
            .then_with(|| self.prefill_node.cmp(&other.prefill_node))
            .then_with(|| self.decode_node.cmp(&other.decode_node))
            .is_lt()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route_estimate(
    request: &InferenceRequest,
    effective_prefill_tokens: u32,
    arrival_s: f64,
    prefill_node: NodeId,
    decode_node: NodeId,
    prefill_route_gpus: &[GpuAddr],
    decode_route_gpus: &[GpuAddr],
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    prefill_worker_slots_per_gpu: usize,
    decode_worker_slots_per_gpu: usize,
    kv_transfer_s: f64,
    kv_resources: &[String],
    routing_load: &RoutingLoad,
) -> RouteEstimate {
    let prefill_scale = request_token_scale(
        request.batch_size,
        effective_prefill_tokens,
        base_batch_size,
        base_prompt_tokens,
    );
    let prefill_service_s = finite_or_zero(prefill_score.estimated_latency_s) * prefill_scale;
    let first_token_scale = request_token_scale(request.batch_size, 1, base_batch_size, 1);
    let decode_iterations =
        1.0 + f64::from(request.decode_tokens.saturating_sub(1)) * decode_tail_scale;
    let decode_service_s = finite_or_zero(decode_one_score.estimated_latency_s)
        * first_token_scale
        * decode_iterations;
    let prefill_ready_s = route_workers_ready_s(
        &routing_load.prefill_worker_ready_s,
        prefill_route_gpus,
        prefill_node,
        prefill_worker_slots_per_gpu,
    );
    let decode_ready_s = route_workers_ready_s(
        &routing_load.decode_worker_ready_s,
        decode_route_gpus,
        decode_node,
        decode_worker_slots_per_gpu,
    );
    let prefill_start_s = arrival_s.max(prefill_ready_s);
    let prefill_finish_s = prefill_start_s + prefill_service_s;
    let kv_ready_s = route_resources_ready_s(&routing_load.kv_route_ready_s, kv_resources);
    let kv_start_s = prefill_finish_s.max(kv_ready_s);
    let kv_finish_s = kv_start_s + finite_or_zero(kv_transfer_s);
    let decode_start_s = kv_finish_s.max(decode_ready_s);
    let decode_finish_s = decode_start_s + decode_service_s;

    RouteEstimate {
        prefill_finish_s,
        kv_finish_s,
        decode_finish_s,
        e2el_proxy_s: (decode_finish_s - arrival_s).max(0.0),
        kv_transfer_s: finite_or_zero(kv_transfer_s),
        kv_resource_wait_s: (kv_start_s - prefill_finish_s).max(0.0),
        prefill_wait_s: (prefill_start_s - arrival_s).max(0.0),
        decode_wait_s: (decode_start_s - kv_finish_s).max(0.0),
        kv_resources: kv_resources.to_vec(),
        prefill_node,
        decode_node,
    }
}

pub(super) fn route_workers_ready_s(
    ready_s: &BTreeMap<GpuAddr, Vec<f64>>,
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
    worker_slots_per_gpu: usize,
) -> f64 {
    route_worker_gpus(route_gpus, fallback_node)
        .iter()
        .map(|gpu| worker_slot_ready_s(ready_s, *gpu, worker_slots_per_gpu))
        .fold(0.0, f64::max)
}

pub(super) fn route_resources_ready_s(
    ready_s: &BTreeMap<String, f64>,
    resources: &[String],
) -> f64 {
    resources
        .iter()
        .filter_map(|resource| ready_s.get(resource).copied())
        .fold(0.0, f64::max)
}

pub(super) fn mark_route_workers_ready(
    ready_s: &mut BTreeMap<GpuAddr, Vec<f64>>,
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
    worker_slots_per_gpu: usize,
    finish_s: f64,
) {
    for gpu in route_worker_gpus(route_gpus, fallback_node) {
        mark_worker_slot_ready(ready_s, gpu, worker_slots_per_gpu, finish_s);
    }
}

pub(super) fn worker_slot_ready_s(
    ready_s: &BTreeMap<GpuAddr, Vec<f64>>,
    gpu: GpuAddr,
    worker_slots_per_gpu: usize,
) -> f64 {
    let slots = worker_slots_per_gpu.max(1);
    let Some(ready_slots) = ready_s.get(&gpu) else {
        return 0.0;
    };
    if ready_slots.len() < slots {
        return 0.0;
    }
    ready_slots.iter().copied().fold(f64::INFINITY, f64::min)
}

pub(super) fn mark_worker_slot_ready(
    ready_s: &mut BTreeMap<GpuAddr, Vec<f64>>,
    gpu: GpuAddr,
    worker_slots_per_gpu: usize,
    finish_s: f64,
) -> Option<u32> {
    if !finish_s.is_finite() {
        return None;
    }
    let slots = worker_slots_per_gpu.max(1);
    let ready_slots = ready_s.entry(gpu).or_default();
    if ready_slots.len() < slots {
        ready_slots.push(finish_s);
        return Some((ready_slots.len() - 1).min(u32::MAX as usize) as u32);
    }
    if let Some((slot_idx, _)) = ready_slots
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1).then_with(|| left.0.cmp(&right.0)))
    {
        ready_slots[slot_idx] = finish_s;
        return Some(slot_idx.min(u32::MAX as usize) as u32);
    }
    None
}

pub(super) struct WorkerAssignmentSpan<'a> {
    pub(super) phase: &'a str,
    pub(super) start_s: f64,
    pub(super) finish_s: f64,
    pub(super) operation_ids: &'a [usize],
}

pub(super) fn assign_worker_slots(
    ready_s: &mut BTreeMap<GpuAddr, Vec<f64>>,
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
    worker_slots_per_gpu: usize,
    span: WorkerAssignmentSpan<'_>,
) -> Vec<ServingWorkerAssignmentObservation> {
    route_worker_gpus(route_gpus, fallback_node)
        .into_iter()
        .filter_map(|gpu| {
            mark_worker_slot_ready(ready_s, gpu, worker_slots_per_gpu, span.finish_s).map(|slot| {
                ServingWorkerAssignmentObservation {
                    phase: span.phase.to_string(),
                    node_id: gpu.node_id,
                    local_gpu_id: gpu.local_gpu_id,
                    slot,
                    start_s: span.start_s,
                    finish_s: span.finish_s,
                    operation_ids: span.operation_ids.to_vec(),
                }
            })
        })
        .collect()
}

pub(super) fn assignments_for_worker_gpus(
    assignments: &[ServingWorkerAssignmentObservation],
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
) -> Vec<ServingWorkerAssignmentObservation> {
    let route_gpus = route_worker_gpus(route_gpus, fallback_node)
        .into_iter()
        .collect::<BTreeSet<_>>();
    assignments
        .iter()
        .filter(|assignment| {
            route_gpus.contains(&GpuAddr {
                node_id: assignment.node_id,
                local_gpu_id: assignment.local_gpu_id,
            })
        })
        .cloned()
        .collect()
}

pub(super) fn mark_route_resources_ready(
    ready_s: &mut BTreeMap<String, f64>,
    resources: &[String],
    finish_s: f64,
) {
    if !finish_s.is_finite() {
        return;
    }
    for resource in resources {
        let entry = ready_s.entry(resource.clone()).or_insert(0.0);
        *entry = entry.max(finish_s);
    }
}

pub(super) fn route_worker_gpus(route_gpus: &[GpuAddr], fallback_node: NodeId) -> Vec<GpuAddr> {
    let mut gpus = route_gpus.to_vec();
    if gpus.is_empty() {
        gpus.push(GpuAddr {
            node_id: fallback_node,
            local_gpu_id: 0,
        });
    }
    gpus.sort_unstable();
    gpus.dedup();
    gpus
}
