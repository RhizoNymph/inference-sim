use super::*;

pub(super) fn pool_rejection(pool: &ResolvedServingPool, message: String) -> ServingRejection {
    ServingRejection {
        phase: "pool".to_string(),
        category: "pool".to_string(),
        resource: pool_resource(pool),
        code: "pool_invalid".to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: Some(
            "check serving pool node IDs, group labels, overlap policy, and pool search bounds"
                .to_string(),
        ),
        message,
    }
}

pub(super) fn pool_resource(pool: &ResolvedServingPool) -> String {
    pool.label.clone().unwrap_or_else(|| {
        format!(
            "prefill_nodes={:?} decode_nodes={:?}",
            pool.prefill_nodes, pool.decode_nodes
        )
    })
}

pub(super) fn service_health_rejections(
    traffic: &ServingTraffic,
    pool: &ResolvedServingPool,
) -> Vec<ServingRejection> {
    let mut rejections = Vec::new();
    push_service_health_rejection(&mut rejections, "prefill", traffic.services.prefill, true);
    push_service_health_rejection(&mut rejections, "decode", traffic.services.decode, true);
    push_service_health_rejection(
        &mut rejections,
        "kv_transfer",
        traffic.services.kv_transfer,
        !same_u32s(&pool.prefill_nodes, &pool.decode_nodes),
    );
    rejections
}

pub(super) fn push_service_health_rejection(
    rejections: &mut Vec<ServingRejection>,
    phase: &str,
    service: ServingServicePhaseConfig,
    required: bool,
) {
    if !required || service.health.accepts_requests() {
        return;
    }

    let code = match service.health {
        ServingServiceHealth::Healthy => return,
        ServingServiceHealth::Draining => "service_draining",
        ServingServiceHealth::Unavailable => "service_unavailable",
    };
    rejections.push(ServingRejection {
        phase: phase.to_string(),
        category: "service".to_string(),
        resource: format!("{phase}_service"),
        code: code.to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: Some(format!(
            "restore {phase} service health or choose a serving pool whose required services are healthy"
        )),
        message: format!(
            "{phase} service is {} and cannot accept new serving requests",
            service.health.as_str()
        ),
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route_availability_rejections(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    pool: &ResolvedServingPool,
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    calibration: SimulationCalibration,
    route_coverage: ServingRouteCoverage,
) -> Vec<ServingRejection> {
    if !prefill_score.feasible || !decode_one_score.feasible {
        return Vec::new();
    }

    let calibration = calibration.sanitized();
    let prefill_resource_base_node = single_placement_node(prefill_score);
    let prefill_placement_nodes = placement_nodes(prefill_score);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let decode_placement_nodes = placement_nodes(decode_one_score);

    match traffic.routing_policy {
        ServingRoutingPolicy::RoundRobin => {
            let request_count = traffic.request_count(calibration);
            for request_idx in 0..request_count {
                let request_shape = traffic.request_at(request, request_idx);
                let prefill_node = routed_serving_node(&pool.prefill_nodes, request_idx);
                let decode_node = routed_serving_node(&pool.decode_nodes, request_idx);
                let prefill_route_nodes = routed_node_set(
                    prefill_resource_base_node,
                    &prefill_placement_nodes,
                    prefill_node,
                );
                let decode_route_nodes = routed_node_set(
                    decode_resource_base_node,
                    &decode_placement_nodes,
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
                    &request_shape,
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
                    return vec![route_unavailable_rejection(
                        "round_robin",
                        Some(request_idx),
                        &prefill_route_nodes,
                        &decode_route_nodes,
                        route_coverage,
                    )];
                }
            }
            Vec::new()
        }
        ServingRoutingPolicy::TopologyAware => {
            let prefill_candidates = sorted_unique_nodes(&pool.prefill_nodes);
            let decode_candidates = sorted_unique_nodes(&pool.decode_nodes);
            let request_shape = traffic.request_at(request, 0);
            for prefill_node in prefill_candidates {
                let prefill_route_nodes = routed_node_set(
                    prefill_resource_base_node,
                    &prefill_placement_nodes,
                    prefill_node,
                );
                for decode_node in &decode_candidates {
                    let decode_route_nodes = routed_node_set(
                        decode_resource_base_node,
                        &decode_placement_nodes,
                        *decode_node,
                    );
                    let prefill_route_gpus = routed_gpu_set(
                        prefill_resource_base_node,
                        &prefill_score.placement.rank_to_gpu,
                        prefill_node,
                    );
                    let decode_route_gpus = routed_gpu_set(
                        decode_resource_base_node,
                        &decode_one_score.placement.rank_to_gpu,
                        *decode_node,
                    );
                    let kv_bytes = kv_transfer_bytes_for_routes(
                        model,
                        &request_shape,
                        &prefill_route_nodes,
                        &decode_route_nodes,
                        &prefill_route_gpus,
                        &decode_route_gpus,
                    );
                    if Solver::transfer_between_gpus_routable(
                        cluster,
                        &prefill_route_gpus,
                        &decode_route_gpus,
                        kv_bytes,
                    ) {
                        return Vec::new();
                    }
                }
            }

            vec![route_unavailable_rejection(
                "topology_aware",
                None,
                &pool.prefill_nodes,
                &pool.decode_nodes,
                route_coverage,
            )]
        }
    }
}

pub(super) fn route_unavailable_rejection(
    policy: &str,
    request_idx: Option<u32>,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    route_coverage: ServingRouteCoverage,
) -> ServingRejection {
    let request_context = request_idx
        .map(|idx| format!(" for request {idx}"))
        .unwrap_or_default();
    ServingRejection {
        phase: "kv_transfer".to_string(),
        category: "topology".to_string(),
        resource: "kv_transfer_route".to_string(),
        code: "kv_transfer_route_unavailable".to_string(),
        observed: Some(f64::from(route_coverage.routable_candidate_count)),
        limit: Some(1.0),
        unit: Some("routes".to_string()),
        remediation: Some(
            "choose connected prefill/decode pools or add the missing custom inter-node links"
                .to_string(),
        ),
        message: format!(
            "{policy} KV transfer route unavailable{request_context}: route_coverage={}/{} routable, unroutable_candidates={}, prefill_nodes={prefill_nodes:?} decode_nodes={decode_nodes:?}",
            route_coverage.routable_candidate_count,
            route_coverage.candidate_count,
            route_coverage.unroutable_candidate_count,
        ),
    }
}

pub(super) fn route_coverage(
    cluster: &Cluster,
    pool: &ResolvedServingPool,
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
) -> ServingRouteCoverage {
    if !prefill_score.feasible || !decode_one_score.feasible {
        return ServingRouteCoverage::default();
    }

    let prefill_resource_base_node = single_placement_node(prefill_score);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let prefill_candidates = sorted_unique_nodes(&pool.prefill_nodes);
    let decode_candidates = sorted_unique_nodes(&pool.decode_nodes);
    let mut candidate_count = 0usize;
    let mut routable_candidate_count = 0usize;

    for prefill_node in prefill_candidates {
        let prefill_route_gpus = routed_gpu_set(
            prefill_resource_base_node,
            &prefill_score.placement.rank_to_gpu,
            prefill_node,
        );
        for decode_node in &decode_candidates {
            candidate_count += 1;
            let decode_route_gpus = routed_gpu_set(
                decode_resource_base_node,
                &decode_one_score.placement.rank_to_gpu,
                *decode_node,
            );
            if Solver::transfer_between_gpus_routable(
                cluster,
                &prefill_route_gpus,
                &decode_route_gpus,
                Bytes::from_bytes(1),
            ) {
                routable_candidate_count += 1;
            }
        }
    }

    let bounded_candidate_count = candidate_count.min(u32::MAX as usize) as u32;
    let bounded_routable_candidate_count = routable_candidate_count.min(u32::MAX as usize) as u32;
    let fraction = if candidate_count == 0 {
        0.0
    } else {
        routable_candidate_count as f64 / candidate_count as f64
    };

    ServingRouteCoverage {
        candidate_count: bounded_candidate_count,
        routable_candidate_count: bounded_routable_candidate_count,
        unroutable_candidate_count: bounded_candidate_count
            .saturating_sub(bounded_routable_candidate_count),
        fraction,
    }
}

pub(super) fn parallelism_rejections(
    phase: &'static str,
    score: &ScoredParallelismConfig,
) -> Vec<ServingRejection> {
    if score.feasible {
        return Vec::new();
    }

    let placement_rejections: Vec<_> = score
        .placement_evidence
        .iter()
        .filter(|evidence| evidence.decision == "rejected")
        .map(|evidence| placement_evidence_rejection(phase, evidence))
        .collect();
    if !placement_rejections.is_empty() {
        return placement_rejections;
    }

    let message = score
        .rejected_reason
        .clone()
        .unwrap_or_else(|| format!("{phase} parallelism config rejected"));
    let category = classify_rejection_category(&message);
    let code = classify_rejection_code(category, &message);
    vec![ServingRejection {
        phase: phase.to_string(),
        category: category.to_string(),
        resource: classify_rejection_resource(category, &message).to_string(),
        code: code.to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: rejection_remediation(category, code, phase).map(str::to_string),
        message,
    }]
}

pub(super) fn placement_evidence_rejection(
    phase: &'static str,
    evidence: &PlacementEvidence,
) -> ServingRejection {
    ServingRejection {
        phase: phase.to_string(),
        category: placement_evidence_rejection_category(evidence).to_string(),
        resource: evidence.resource.clone(),
        code: evidence.code.clone(),
        observed: evidence.observed,
        limit: evidence.limit,
        unit: evidence.unit.clone(),
        remediation: evidence.remediation.clone().or_else(|| {
            rejection_remediation(
                placement_evidence_rejection_category(evidence),
                &evidence.code,
                phase,
            )
            .map(str::to_string)
        }),
        message: evidence.message.clone(),
    }
}

pub(super) fn placement_evidence_rejection_category(evidence: &PlacementEvidence) -> &'static str {
    let code = evidence.code.as_str();
    let resource = evidence.resource.as_str();
    if code.contains("hbm") || resource == "gpu_hbm" {
        "memory"
    } else if code.contains("dtype") {
        "capability"
    } else if code.contains("available_gpus") || resource == "available_gpus" {
        "capacity"
    } else if code.contains("placement")
        || code.contains("rank_count")
        || code.contains("unknown_gpu")
        || code.contains("duplicate_gpu")
        || code.contains("gpu_unavailable")
    {
        "placement"
    } else {
        classify_rejection_category(&evidence.message)
    }
}

pub(super) fn classify_rejection_category(message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    if message.contains("hbm") || message.contains("memory") || message.contains("gb per gpu") {
        "memory"
    } else if message.contains("link")
        || message.contains("route")
        || message.contains("topology")
        || message.contains("fabric")
    {
        "topology"
    } else if message.contains("gpu") && message.contains("available") {
        "capacity"
    } else {
        "parallelism"
    }
}

pub(super) fn classify_rejection_resource(category: &str, message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    match category {
        "memory" => "gpu_hbm",
        "topology" => "interconnect",
        "capacity" if message.contains("gpu") => "gpu_count",
        "capacity" => "serving_capacity",
        _ => "parallelism_config",
    }
}

pub(super) fn classify_rejection_code(category: &str, message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    match category {
        "memory" => "memory_capacity_exceeded",
        "topology" => "topology_unavailable",
        "capacity" if message.contains("gpu") => "gpu_capacity_insufficient",
        "capacity" => "serving_capacity_exceeded",
        _ => "parallelism_config_invalid",
    }
}

pub(super) fn rejection_remediation(
    category: &str,
    code: &str,
    phase: &str,
) -> Option<&'static str> {
    match (category, code) {
        ("memory", _) => {
            Some("increase sharding, use GPUs with more HBM, or reduce batch/sequence/KV residency")
        }
        ("topology", _) => Some(
            "choose a pool or placement inside connected topology domains, or add the missing links",
        ),
        ("capacity", "gpu_capacity_insufficient") => {
            Some("reduce total ranks or expand the selected GPU/node pool")
        }
        ("capacity", _) => Some("raise serving capacity limits or reduce offered traffic"),
        ("parallelism", _) if phase == "decode_first_token" => {
            Some("adjust decode search space so one-token decode is feasible")
        }
        ("parallelism", _) => Some("adjust tensor/pipeline/expert/data ranks for this phase"),
        _ => None,
    }
}

pub(super) fn joined_rejection_reason(rejections: &[ServingRejection]) -> Option<String> {
    if rejections.is_empty() {
        None
    } else {
        Some(
            rejections
                .iter()
                .map(|rejection| rejection.message.as_str())
                .collect::<Vec<_>>()
                .join("; "),
        )
    }
}
