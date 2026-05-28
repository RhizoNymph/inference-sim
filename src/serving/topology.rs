use super::*;

fn transfer_resources(bottlenecks: &[String]) -> Vec<String> {
    let mut resources = bottlenecks.to_vec();
    if resources.is_empty() {
        resources.push("transfer: local".to_string());
    }
    resources.sort();
    resources.dedup();
    resources
}

pub(super) fn kv_transfer_scheduler_resources(
    paths: &[ServingKvTransferPathObservation],
    bottlenecks: &[String],
) -> Vec<String> {
    let mut resources = paths
        .iter()
        .flat_map(|path| path.resource_details.iter().map(kv_route_resource_id))
        .collect::<Vec<_>>();
    resources.sort();
    resources.dedup();

    if resources.is_empty() {
        transfer_resources(bottlenecks)
    } else {
        resources
    }
}

pub(super) fn kv_transfer_paths(
    cluster: &Cluster,
    source_gpus: &[GpuAddr],
    destination_gpus: &[GpuAddr],
    bytes: Bytes,
) -> Vec<ServingKvTransferPathObservation> {
    if bytes.as_bytes() == 0 || source_gpus.is_empty() || destination_gpus.is_empty() {
        return Vec::new();
    }

    let graph = TopologyGraph::from_cluster(cluster);
    let mut paths = Vec::new();
    for source in source_gpus {
        for destination in destination_gpus {
            if source == destination {
                continue;
            }
            let path = if source.node_id == destination.node_id {
                let (latency_s, bandwidth) =
                    Solver::intra_node_link_profile(cluster, source.node_id);
                ServingKvTransferPathObservation {
                    source: *source,
                    destination: *destination,
                    latency_s,
                    bottleneck_bandwidth_gbps: bandwidth.as_gigabits_per_sec(),
                    resources: vec![format!("node {} intra-node fabric", source.node_id)],
                    resource_details: vec![intra_node_kv_path_resource(
                        *source,
                        *destination,
                        latency_s,
                        bandwidth.as_gigabits_per_sec(),
                    )],
                }
            } else {
                let Some(path) = graph.route_between_gpus(*source, *destination, bytes) else {
                    continue;
                };
                let resource_details = path
                    .resources
                    .iter()
                    .map(kv_path_resource_from_routed)
                    .collect();
                ServingKvTransferPathObservation {
                    source: *source,
                    destination: *destination,
                    latency_s: path.latency_s,
                    bottleneck_bandwidth_gbps: path.bottleneck_bandwidth.as_gigabits_per_sec(),
                    resources: path.labels,
                    resource_details,
                }
            };
            paths.push(path);
        }
    }

    paths.sort_by(|left, right| {
        left.source
            .cmp(&right.source)
            .then_with(|| left.destination.cmp(&right.destination))
    });
    paths
}

fn intra_node_kv_path_resource(
    source: GpuAddr,
    destination: GpuAddr,
    latency_s: f64,
    bandwidth_gbps: f64,
) -> ServingKvTransferPathResourceObservation {
    ServingKvTransferPathResourceObservation {
        kind: "intra_node_fabric".to_string(),
        label: format!("node {} intra-node fabric", source.node_id),
        bandwidth_gbps,
        latency_s,
        rail_id: None,
        from: Some(kv_path_gpu_endpoint(source)),
        to: Some(kv_path_gpu_endpoint(destination)),
    }
}

fn kv_path_resource_from_routed(
    resource: &RoutedResource,
) -> ServingKvTransferPathResourceObservation {
    ServingKvTransferPathResourceObservation {
        kind: resource.kind.as_str().to_string(),
        label: resource.label.clone(),
        bandwidth_gbps: resource.bandwidth.as_gigabits_per_sec(),
        latency_s: resource.latency_s,
        rail_id: resource.rail_id,
        from: Some(kv_path_endpoint_from_graph(&resource.from)),
        to: Some(kv_path_endpoint_from_graph(&resource.to)),
    }
}

fn kv_path_endpoint_from_graph(
    resource: &GraphResource,
) -> ServingKvTransferPathEndpointObservation {
    match resource {
        GraphResource::Gpu(gpu) => kv_path_gpu_endpoint(*gpu),
        GraphResource::IntraNode { node_id } => ServingKvTransferPathEndpointObservation {
            kind: "intra_node".to_string(),
            node_id: Some(*node_id),
            local_gpu_id: None,
            nic_id: None,
            rail_id: None,
        },
        GraphResource::Nic {
            node_id,
            nic_id,
            rail_id,
        } => ServingKvTransferPathEndpointObservation {
            kind: "nic".to_string(),
            node_id: Some(*node_id),
            local_gpu_id: None,
            nic_id: Some(*nic_id),
            rail_id: Some(*rail_id),
        },
    }
}

fn kv_path_gpu_endpoint(gpu: GpuAddr) -> ServingKvTransferPathEndpointObservation {
    ServingKvTransferPathEndpointObservation {
        kind: "gpu".to_string(),
        node_id: Some(gpu.node_id),
        local_gpu_id: Some(gpu.local_gpu_id),
        nic_id: None,
        rail_id: None,
    }
}

#[derive(Clone, Debug)]
struct KvRouteResourceAccumulator {
    resource_id: String,
    kind: String,
    label: String,
    request_indices: BTreeSet<u32>,
    path_observations: u64,
    transfer_bytes: u64,
    estimated_transfer_s: f64,
    min_bandwidth_gbps: f64,
    max_latency_s: f64,
    rail_id: Option<u32>,
    from: Option<ServingKvTransferPathEndpointObservation>,
    to: Option<ServingKvTransferPathEndpointObservation>,
}

pub(super) fn kv_route_resource_summary(
    observations: &[ServingRequestObservation],
) -> Vec<ServingKvRouteResourceSummary> {
    let mut resources = BTreeMap::<String, KvRouteResourceAccumulator>::new();

    for observation in observations {
        if observation.kv_transfer_bytes == 0 || observation.kv_transfer_paths.is_empty() {
            continue;
        }
        let bytes_per_path = observation
            .kv_transfer_bytes
            .div_ceil(observation.kv_transfer_paths.len() as u64);
        for path in &observation.kv_transfer_paths {
            for resource in &path.resource_details {
                let resource_id = kv_route_resource_id(resource);
                let bytes_per_segment = bytes_per_path;
                let estimated_transfer_s = resource.latency_s
                    + transfer_seconds(bytes_per_segment, resource.bandwidth_gbps);
                let entry = resources.entry(resource_id.clone()).or_insert_with(|| {
                    KvRouteResourceAccumulator {
                        resource_id: resource_id.clone(),
                        kind: resource.kind.clone(),
                        label: resource.label.clone(),
                        request_indices: BTreeSet::new(),
                        path_observations: 0,
                        transfer_bytes: 0,
                        estimated_transfer_s: 0.0,
                        min_bandwidth_gbps: resource.bandwidth_gbps,
                        max_latency_s: resource.latency_s,
                        rail_id: resource.rail_id,
                        from: resource.from.clone(),
                        to: resource.to.clone(),
                    }
                });
                entry.request_indices.insert(observation.request_idx);
                entry.path_observations = entry.path_observations.saturating_add(1);
                entry.transfer_bytes = entry.transfer_bytes.saturating_add(bytes_per_segment);
                entry.estimated_transfer_s += estimated_transfer_s;
                if resource.bandwidth_gbps.is_finite()
                    && resource.bandwidth_gbps < entry.min_bandwidth_gbps
                {
                    entry.min_bandwidth_gbps = resource.bandwidth_gbps;
                }
                if resource.latency_s.is_finite() && resource.latency_s > entry.max_latency_s {
                    entry.max_latency_s = resource.latency_s;
                }
            }
        }
    }

    let mut summaries = resources
        .into_values()
        .map(|resource| ServingKvRouteResourceSummary {
            resource_id: resource.resource_id,
            kind: resource.kind,
            label: resource.label,
            request_count: resource.request_indices.len().min(u32::MAX as usize) as u32,
            path_observations: resource.path_observations,
            transfer_bytes: resource.transfer_bytes,
            estimated_transfer_s: resource.estimated_transfer_s,
            min_bandwidth_gbps: resource.min_bandwidth_gbps,
            max_latency_s: resource.max_latency_s,
            rail_id: resource.rail_id,
            from: resource.from,
            to: resource.to,
        })
        .collect::<Vec<_>>();
    summaries.sort_by(|left, right| {
        right
            .estimated_transfer_s
            .total_cmp(&left.estimated_transfer_s)
            .then_with(|| right.transfer_bytes.cmp(&left.transfer_bytes))
            .then_with(|| right.path_observations.cmp(&left.path_observations))
            .then_with(|| left.resource_id.cmp(&right.resource_id))
    });
    summaries
}

pub(super) fn kv_route_topology_summary(
    summaries: &[ServingKvRouteResourceSummary],
) -> ServingKvRouteTopologySummary {
    let mut rail_ids = BTreeSet::new();
    let mut inter_node_route_resource_count = 0u32;
    let mut gpu_nic_route_resource_count = 0u32;
    let mut intra_node_route_resource_count = 0u32;
    let mut unrailed_inter_node_route_resource_count = 0u32;

    for summary in summaries {
        if let Some(rail_id) = summary.rail_id {
            rail_ids.insert(rail_id);
        }
        match summary.kind.as_str() {
            "inter_node_fabric" | "gpu_scoped_inter_node_fabric" => {
                inter_node_route_resource_count = inter_node_route_resource_count.saturating_add(1);
                if summary.rail_id.is_none() {
                    unrailed_inter_node_route_resource_count =
                        unrailed_inter_node_route_resource_count.saturating_add(1);
                }
            }
            "gpu_nic_local" => {
                gpu_nic_route_resource_count = gpu_nic_route_resource_count.saturating_add(1);
            }
            "intra_node_fabric" => {
                intra_node_route_resource_count = intra_node_route_resource_count.saturating_add(1);
            }
            _ => {}
        }
    }

    let rail_ids = rail_ids.into_iter().collect::<Vec<_>>();
    let single_rail_dependency = inter_node_route_resource_count > 0 && rail_ids.len() == 1;
    ServingKvRouteTopologySummary {
        route_resource_count: summaries.len().min(u32::MAX as usize) as u32,
        inter_node_route_resource_count,
        gpu_nic_route_resource_count,
        intra_node_route_resource_count,
        rail_count: rail_ids.len().min(u32::MAX as usize) as u32,
        single_rail_id: single_rail_dependency.then(|| rail_ids[0]),
        single_rail_dependency,
        unrailed_inter_node_route_resource_count,
        rail_ids,
    }
}

pub(super) fn topology_bottleneck_observations(
    route_coverage: ServingRouteCoverage,
    kv_route_topology: &ServingKvRouteTopologySummary,
    kv_route_resources: &[ServingKvRouteResourceSummary],
    request_observations: &[ServingRequestObservation],
    phase_resource_utilization: &[ServingPhaseResourceUtilization],
) -> Vec<ServingTopologyBottleneckObservation> {
    let mut bottlenecks = Vec::new();

    if route_coverage.candidate_count > 0 && route_coverage.unroutable_candidate_count > 0 {
        let severity = if route_coverage.routable_candidate_count == 0 {
            "critical"
        } else {
            "warning"
        };
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "kv_transfer_route".to_string(),
            code: "partial_route_coverage".to_string(),
            severity: severity.to_string(),
            observed: Some(f64::from(route_coverage.routable_candidate_count)),
            limit: Some(f64::from(route_coverage.candidate_count)),
            unit: Some("routes".to_string()),
            message: format!(
                "only {} of {} prefill/decode route candidates are routable; {} candidates are unroutable",
                route_coverage.routable_candidate_count,
                route_coverage.candidate_count,
                route_coverage.unroutable_candidate_count
            ),
            remediation: Some(
                "choose connected prefill/decode pools, add missing inter-node links, or switch to topology-aware routing".to_string(),
            ),
        });
    }

    if kv_route_topology.single_rail_dependency {
        let rail_label = kv_route_topology
            .single_rail_id
            .map(|rail| rail.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "kv_route_rail".to_string(),
            code: "single_rail_dependency".to_string(),
            severity: "warning".to_string(),
            observed: Some(f64::from(kv_route_topology.rail_count)),
            limit: Some(2.0),
            unit: Some("rails".to_string()),
            message: format!(
                "KV transfer routes use a single inter-node rail ({rail_label}); rail failure or congestion can affect all routed KV handoffs"
            ),
            remediation: Some(
                "add rail-diverse custom links, adjust GPU/NIC locality, or choose prefill/decode pools with multiple routable rails".to_string(),
            ),
        });
    }

    if kv_route_topology.unrailed_inter_node_route_resource_count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "kv_route_rail".to_string(),
            code: "unrailed_inter_node_routes".to_string(),
            severity: "info".to_string(),
            observed: Some(f64::from(
                kv_route_topology.unrailed_inter_node_route_resource_count,
            )),
            limit: Some(0.0),
            unit: Some("resources".to_string()),
            message: format!(
                "{} inter-node KV route resources do not carry rail IDs, so rail-diversity risk may be underreported",
                kv_route_topology.unrailed_inter_node_route_resource_count
            ),
            remediation: Some(
                "use NIC rail maps or rail-scoped custom inter-node links when modeling rail-sensitive clusters".to_string(),
            ),
        });
    }

    bottlenecks.extend(route_locality_bottleneck_observations(kv_route_resources));
    bottlenecks.extend(dynamic_route_contention_bottleneck_observations(
        request_observations,
        phase_resource_utilization,
    ));

    bottlenecks
}

pub(super) fn topology_domain_bottleneck_observations(
    cluster: &Cluster,
    prefill_placement: &RankPlacement,
    decode_placement: &RankPlacement,
) -> Vec<ServingTopologyBottleneckObservation> {
    let prefill_nodes = placement_node_ids(prefill_placement);
    let decode_nodes = placement_node_ids(decode_placement);
    let all_nodes = union_node_ids(&prefill_nodes, &decode_nodes);

    let mut bottlenecks = Vec::new();
    append_topology_domain_bottlenecks(cluster, "placement", &all_nodes, &mut bottlenecks);
    append_topology_domain_bottlenecks(cluster, "prefill", &prefill_nodes, &mut bottlenecks);
    append_topology_domain_bottlenecks(cluster, "decode", &decode_nodes, &mut bottlenecks);
    bottlenecks
}

fn placement_node_ids(placement: &RankPlacement) -> Vec<NodeId> {
    placement
        .rank_to_gpu
        .iter()
        .map(|gpu| gpu.node_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn union_node_ids(left: &[NodeId], right: &[NodeId]) -> Vec<NodeId> {
    left.iter()
        .copied()
        .chain(right.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn serving_pool_topology_summary(
    cluster: Option<&Cluster>,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
) -> ServingPoolTopologySummary {
    let prefill_node_set = prefill_nodes.iter().copied().collect::<BTreeSet<_>>();
    let decode_node_set = decode_nodes.iter().copied().collect::<BTreeSet<_>>();
    let prefill_nodes = prefill_node_set.iter().copied().collect::<Vec<_>>();
    let decode_nodes = decode_node_set.iter().copied().collect::<Vec<_>>();
    let shared_node_count = prefill_node_set.intersection(&decode_node_set).count();

    ServingPoolTopologySummary {
        prefill_node_count: prefill_nodes.len(),
        decode_node_count: decode_nodes.len(),
        shared_node_count,
        dedicated_prefill_node_count: prefill_nodes.len().saturating_sub(shared_node_count),
        dedicated_decode_node_count: decode_nodes.len().saturating_sub(shared_node_count),
        prefill_racks: topology_summary_domain_values(cluster, &prefill_nodes, |node| {
            node.topology.rack.as_deref()
        }),
        decode_racks: topology_summary_domain_values(cluster, &decode_nodes, |node| {
            node.topology.rack.as_deref()
        }),
        prefill_islands: topology_summary_domain_values(cluster, &prefill_nodes, |node| {
            node.topology.island.as_deref()
        }),
        decode_islands: topology_summary_domain_values(cluster, &decode_nodes, |node| {
            node.topology.island.as_deref()
        }),
        prefill_failure_domains: topology_summary_domain_values(cluster, &prefill_nodes, |node| {
            node.topology.failure_domain.as_deref()
        }),
        decode_failure_domains: topology_summary_domain_values(cluster, &decode_nodes, |node| {
            node.topology.failure_domain.as_deref()
        }),
        prefill_node_labels: topology_summary_node_labels(cluster, &prefill_nodes),
        decode_node_labels: topology_summary_node_labels(cluster, &decode_nodes),
    }
}

fn topology_summary_domain_values(
    cluster: Option<&Cluster>,
    node_ids: &[NodeId],
    domain: impl Fn(&Node) -> Option<&str>,
) -> Vec<String> {
    let Some(cluster) = cluster else {
        return Vec::new();
    };

    node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id).and_then(&domain))
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn topology_summary_node_labels(cluster: Option<&Cluster>, node_ids: &[NodeId]) -> Vec<String> {
    let Some(cluster) = cluster else {
        return Vec::new();
    };

    node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id))
        .flat_map(|node| node.topology.labels.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn append_topology_domain_bottlenecks(
    cluster: &Cluster,
    phase: &str,
    node_ids: &[NodeId],
    bottlenecks: &mut Vec<ServingTopologyBottleneckObservation>,
) {
    if node_ids.len() < 2 {
        return;
    }
    append_single_domain_bottleneck(
        phase,
        "rack",
        node_ids,
        topology_domain_values(cluster, node_ids, |node| node.topology.rack.as_deref()),
        bottlenecks,
    );
    append_single_domain_bottleneck(
        phase,
        "island",
        node_ids,
        topology_domain_values(cluster, node_ids, |node| node.topology.island.as_deref()),
        bottlenecks,
    );
    append_single_domain_bottleneck(
        phase,
        "failure_domain",
        node_ids,
        topology_domain_values(cluster, node_ids, |node| {
            node.topology.failure_domain.as_deref()
        }),
        bottlenecks,
    );
}

fn topology_domain_values(
    cluster: &Cluster,
    node_ids: &[NodeId],
    domain: impl Fn(&Node) -> Option<&str>,
) -> Option<Vec<String>> {
    let mut values = Vec::new();
    for node_id in node_ids {
        let node = cluster.node(*node_id)?;
        values.push(domain(node)?.to_string());
    }
    Some(values)
}

fn append_single_domain_bottleneck(
    phase: &str,
    domain: &str,
    node_ids: &[NodeId],
    values: Option<Vec<String>>,
    bottlenecks: &mut Vec<ServingTopologyBottleneckObservation>,
) {
    let Some(values) = values else {
        return;
    };
    let distinct = values.iter().collect::<BTreeSet<_>>();
    if distinct.len() != 1 {
        return;
    }
    let domain_value = values
        .first()
        .expect("domain values exist when node_ids has at least two");
    bottlenecks.push(ServingTopologyBottleneckObservation {
        phase: phase.to_string(),
        category: "topology".to_string(),
        resource: format!("{domain}:{domain_value}"),
        code: format!("single_{domain}_placement"),
        severity: "warning".to_string(),
        observed: Some(1.0),
        limit: Some(2.0),
        unit: Some("domains".to_string()),
        message: format!(
            "{phase} placement uses {} selected nodes but only one {domain} ({domain_value}); a domain outage can affect the whole selected placement scope",
            node_ids.len()
        ),
        remediation: Some(format!(
            "spread selected ranks or service pools across multiple {domain}s, add topology-domain placement constraints, or lower topology_risk_penalty_weight if this concentration is intentional"
        )),
    });
}

pub(super) fn kv_route_constraint_rejections(
    topology: &ServingKvRouteTopologySummary,
    resources: &[ServingKvRouteResourceSummary],
    constraints: ServingKvRouteConstraints,
) -> Vec<ServingRejection> {
    if !constraints.any() {
        return Vec::new();
    }

    let mut rejections = Vec::new();
    if topology.inter_node_route_resource_count > 0 {
        if let Some(limit) = constraints.min_inter_node_rail_count
            && topology.rail_count < limit
        {
            rejections.push(ServingRejection {
                phase: "kv_transfer".to_string(),
                category: "topology".to_string(),
                resource: "kv_route_rail".to_string(),
                code: "kv_route_rail_count_below_min".to_string(),
                observed: Some(f64::from(topology.rail_count)),
                limit: Some(f64::from(limit)),
                unit: Some("rails".to_string()),
                remediation: Some(
                    "add rail-diverse custom links, adjust GPU/NIC locality, choose different prefill/decode pools, or lower serving.min_kv_route_rail_count"
                        .to_string(),
                ),
                message: format!(
                    "KV transfer routes use {} inter-node rails, below configured minimum {}",
                    topology.rail_count, limit
                ),
            });
        }
        if constraints.require_inter_node_rail_metadata
            && topology.unrailed_inter_node_route_resource_count > 0
        {
            rejections.push(ServingRejection {
                phase: "kv_transfer".to_string(),
                category: "topology".to_string(),
                resource: "kv_route_rail".to_string(),
                code: "kv_route_rail_metadata_missing".to_string(),
                observed: Some(f64::from(
                    topology.unrailed_inter_node_route_resource_count,
                )),
                limit: Some(0.0),
                unit: Some("resources".to_string()),
                remediation: Some(
                    "add NIC rail maps or rail-scoped custom inter-node links, or disable serving.require_kv_route_rail_metadata"
                        .to_string(),
                ),
                message: format!(
                    "{} inter-node KV route resources lack rail metadata",
                    topology.unrailed_inter_node_route_resource_count
                ),
            });
        }
    }

    if constraints.require_gpudirect {
        let host_staged_count = resources
            .iter()
            .filter(|resource| resource.kind == "gpu_nic_local")
            .filter(|resource| {
                route_label_contains_any(
                    &resource.label,
                    &["host-staged", "host_staged", "no-gpudirect", "no_gpudirect"],
                )
            })
            .count()
            .min(u32::MAX as usize) as u32;
        if host_staged_count > 0 {
            rejections.push(ServingRejection {
                phase: "kv_transfer".to_string(),
                category: "topology".to_string(),
                resource: "gpu_nic_local".to_string(),
                code: "host_staged_kv_path_disallowed".to_string(),
                observed: Some(f64::from(host_staged_count)),
                limit: Some(0.0),
                unit: Some("resources".to_string()),
                remediation: Some(
                    "choose GPUDirect-capable GPU/NIC locality, adjust placement tags, or disable serving.require_gpudirect_kv_paths"
                        .to_string(),
                ),
                message: format!(
                    "{} GPU-to-NIC KV route resources are host-staged or lack GPUDirect",
                    host_staged_count
                ),
            });
        }
    }

    rejections
}

pub(super) fn dynamic_route_contention_bottleneck_observations(
    request_observations: &[ServingRequestObservation],
    phase_resource_utilization: &[ServingPhaseResourceUtilization],
) -> Vec<ServingTopologyBottleneckObservation> {
    let mut bottlenecks = Vec::new();
    let queued_requests = request_observations
        .iter()
        .filter(|observation| observation.kv_resource_queue_s.is_finite())
        .filter(|observation| observation.kv_resource_queue_s > 0.0)
        .collect::<Vec<_>>();
    if !queued_requests.is_empty() {
        let total_queue_s = queued_requests
            .iter()
            .map(|observation| observation.kv_resource_queue_s)
            .sum::<f64>();
        let max_queue_s = queued_requests
            .iter()
            .map(|observation| observation.kv_resource_queue_s)
            .fold(0.0_f64, f64::max);
        let queued_count = queued_requests.len().min(u32::MAX as usize) as u32;
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "contention".to_string(),
            resource: "kv_route_resource".to_string(),
            code: "kv_route_resource_queueing".to_string(),
            severity: "warning".to_string(),
            observed: Some(max_queue_s * 1000.0),
            limit: Some(0.0),
            unit: Some("ms".to_string()),
            message: format!(
                "{} requests queued on modeled KV route resources; max route-resource queue {:.3} ms, mean queued-request route-resource queue {:.3} ms",
                queued_count,
                max_queue_s * 1000.0,
                total_queue_s / queued_requests.len() as f64 * 1000.0
            ),
            remediation: Some(
                "add route diversity, increase KV-transfer worker or fabric capacity, reduce burstiness, or choose prefill/decode pools with less shared-route contention"
                    .to_string(),
            ),
        });
    }

    if let Some(hot_route) = phase_resource_utilization
        .iter()
        .filter(|utilization| utilization.phase == "kv_transfer")
        .filter(|utilization| utilization.resource_kind == "kv_route")
        .filter(|utilization| utilization.operation_count > 1)
        .filter(|utilization| utilization.utilization >= 0.85)
        .max_by(|left, right| {
            left.utilization
                .total_cmp(&right.utilization)
                .then_with(|| left.busy_s.total_cmp(&right.busy_s))
        })
    {
        let severity = if hot_route.utilization >= 0.98 {
            "critical"
        } else {
            "warning"
        };
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "contention".to_string(),
            resource: hot_route.resource.clone(),
            code: "hot_kv_route_resource".to_string(),
            severity: severity.to_string(),
            observed: Some(hot_route.utilization),
            limit: Some(0.85),
            unit: Some("utilization".to_string()),
            message: format!(
                "KV route resource '{}' reached {:.1}% utilization across {} operations",
                hot_route.resource,
                hot_route.utilization * 100.0,
                hot_route.operation_count
            ),
            remediation: Some(
                "spread KV handoffs across more rails or decode replicas, increase interconnect capacity, or lower offered load for this placement"
                    .to_string(),
            ),
        });
    }

    bottlenecks
}

#[derive(Clone, Debug, Default)]
struct RouteLocalityDiagnostics {
    count: u32,
    transfer_bytes: u64,
    min_bandwidth_gbps: Option<f64>,
    max_latency_s: Option<f64>,
}

impl RouteLocalityDiagnostics {
    fn observe(&mut self, summary: &ServingKvRouteResourceSummary) {
        self.count = self.count.saturating_add(1);
        self.transfer_bytes = self.transfer_bytes.saturating_add(summary.transfer_bytes);
        if summary.min_bandwidth_gbps.is_finite() {
            self.min_bandwidth_gbps = Some(
                self.min_bandwidth_gbps
                    .map(|current| current.min(summary.min_bandwidth_gbps))
                    .unwrap_or(summary.min_bandwidth_gbps),
            );
        }
        if summary.max_latency_s.is_finite() {
            self.max_latency_s = Some(
                self.max_latency_s
                    .map(|current| current.max(summary.max_latency_s))
                    .unwrap_or(summary.max_latency_s),
            );
        }
    }
}

fn route_locality_bottleneck_observations(
    kv_route_resources: &[ServingKvRouteResourceSummary],
) -> Vec<ServingTopologyBottleneckObservation> {
    let mut host_staged = RouteLocalityDiagnostics::default();
    let mut cross_socket = RouteLocalityDiagnostics::default();
    let mut slow_gpu_nic = RouteLocalityDiagnostics::default();
    let slow_gpu_nic_bandwidth_gbps = slow_gpu_nic_bandwidth_threshold(kv_route_resources);

    for summary in kv_route_resources {
        if summary.kind != "gpu_nic_local" {
            continue;
        }
        if route_label_contains_any(
            &summary.label,
            &["host-staged", "host_staged", "no-gpudirect", "no_gpudirect"],
        ) {
            host_staged.observe(summary);
        }
        if route_label_contains_any(
            &summary.label,
            &["cross-socket", "cross_socket", "cross socket"],
        ) {
            cross_socket.observe(summary);
        }
        if slow_gpu_nic_bandwidth_gbps.is_some_and(|threshold| {
            summary.min_bandwidth_gbps.is_finite() && summary.min_bandwidth_gbps < threshold
        }) {
            slow_gpu_nic.observe(summary);
        }
    }

    let mut bottlenecks = Vec::new();
    if host_staged.count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "gpu_nic_local".to_string(),
            code: "host_staged_kv_path".to_string(),
            severity: "warning".to_string(),
            observed: host_staged.min_bandwidth_gbps,
            limit: None,
            unit: Some("gbps".to_string()),
            message: format!(
                "{} GPU-to-NIC KV route resources are host-staged or lack GPUDirect; min bandwidth {} Gbps, max latency {} us across {} transfer bytes",
                host_staged.count,
                format_optional_f64(host_staged.min_bandwidth_gbps),
                format_optional_f64(host_staged.max_latency_s.map(|latency_s| latency_s * 1e6)),
                host_staged.transfer_bytes
            ),
            remediation: Some(
                "prefer GPUDirect-capable GPU/NIC paths, adjust GPU-to-NIC locality, or model the host-staged copy engine explicitly".to_string(),
            ),
        });
    }
    if cross_socket.count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "gpu_nic_local".to_string(),
            code: "cross_socket_kv_path".to_string(),
            severity: "warning".to_string(),
            observed: cross_socket.max_latency_s.map(|latency_s| latency_s * 1e6),
            limit: None,
            unit: Some("us".to_string()),
            message: format!(
                "{} GPU-to-NIC KV route resources cross sockets; max local-path latency {} us across {} transfer bytes",
                cross_socket.count,
                format_optional_f64(cross_socket.max_latency_s.map(|latency_s| latency_s * 1e6)),
                cross_socket.transfer_bytes
            ),
            remediation: Some(
                "prefer same-socket GPU/NIC affinity, add explicit placement constraints, or model NUMA/PCIe domains before relying on this route".to_string(),
            ),
        });
    }
    if slow_gpu_nic.count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "gpu_nic_local".to_string(),
            code: "slow_gpu_nic_kv_path".to_string(),
            severity: "info".to_string(),
            observed: slow_gpu_nic.min_bandwidth_gbps,
            limit: slow_gpu_nic_bandwidth_gbps,
            unit: Some("gbps".to_string()),
            message: format!(
                "{} GPU-to-NIC KV route resources are much slower than the selected inter-node fabric; min bandwidth {} Gbps below threshold {} Gbps",
                slow_gpu_nic.count,
                format_optional_f64(slow_gpu_nic.min_bandwidth_gbps),
                format_optional_f64(slow_gpu_nic_bandwidth_gbps)
            ),
            remediation: Some(
                "check GPU/NIC affinity, PCIe generation, GPUDirect availability, and whether local copy paths should be calibrated separately".to_string(),
            ),
        });
    }

    bottlenecks
}

fn slow_gpu_nic_bandwidth_threshold(
    kv_route_resources: &[ServingKvRouteResourceSummary],
) -> Option<f64> {
    kv_route_resources
        .iter()
        .filter(|summary| {
            matches!(
                summary.kind.as_str(),
                "inter_node_fabric" | "gpu_scoped_inter_node_fabric"
            ) && summary.min_bandwidth_gbps.is_finite()
        })
        .map(|summary| summary.min_bandwidth_gbps)
        .max_by(f64::total_cmp)
        .map(|bandwidth_gbps| bandwidth_gbps * 0.5)
}

fn route_label_contains_any(label: &str, needles: &[&str]) -> bool {
    let normalized = label.to_ascii_lowercase();
    needles.iter().any(|needle| normalized.contains(needle))
}

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "unknown".to_string())
}

fn kv_route_resource_id(resource: &ServingKvTransferPathResourceObservation) -> String {
    let left = kv_route_endpoint_key(resource.from.as_ref());
    let right = kv_route_endpoint_key(resource.to.as_ref());
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    format!(
        "kv_route:{}|{}|rail={}|{}<->{}",
        resource.kind,
        resource.label,
        optional_u32_key(resource.rail_id),
        first,
        second
    )
}

fn kv_route_endpoint_key(endpoint: Option<&ServingKvTransferPathEndpointObservation>) -> String {
    let Some(endpoint) = endpoint else {
        return "none".to_string();
    };
    format!(
        "{}:node={}:gpu={}:nic={}:rail={}",
        endpoint.kind,
        optional_u32_key(endpoint.node_id),
        optional_u32_key(endpoint.local_gpu_id),
        optional_u32_key(endpoint.nic_id),
        optional_u32_key(endpoint.rail_id)
    )
}

fn optional_u32_key(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn transfer_seconds(bytes: u64, bandwidth_gbps: f64) -> f64 {
    if bandwidth_gbps.is_finite() && bandwidth_gbps > 0.0 {
        bytes as f64 / (bandwidth_gbps * 1e9 / 8.0)
    } else {
        0.0
    }
}
