use super::*;

impl Solver {
    pub fn estimate_collective(
        cluster: &Cluster,
        placement: &RankPlacement,
        collective: &CollectiveCall,
    ) -> CollectiveCost {
        Self::estimate_collective_with_calibration(
            cluster,
            placement,
            collective,
            SimulationCalibration::default(),
        )
    }

    pub fn estimate_collective_with_calibration(
        cluster: &Cluster,
        placement: &RankPlacement,
        collective: &CollectiveCall,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        let calibration = calibration.sanitized();
        if collective.participants.len() <= 1 || collective.bytes_per_rank.as_bytes() == 0 {
            return CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            };
        }

        let participant_addrs: Vec<_> = collective
            .participants
            .iter()
            .filter_map(|rank| placement.gpu_for_rank(*rank))
            .collect();
        let nodes: BTreeSet<NodeId> = participant_addrs.iter().map(|addr| addr.node_id).collect();
        let rank_count = participant_addrs.len().max(1) as f64;
        let bytes = collective.bytes_per_rank.as_bytes() as f64;

        let mut latency_s = 0.0;
        let mut bandwidth_s = 0.0;
        let mut bottlenecks = Vec::new();

        if nodes.len() <= 1 {
            let node_id = participant_addrs
                .first()
                .map(|addr| addr.node_id)
                .unwrap_or(0);
            let (latency, bandwidth) = Self::intra_node_profile(cluster, node_id);
            let steps = Self::collective_steps(collective.kind, rank_count);
            let traffic_multiplier = Self::traffic_multiplier(collective.kind, rank_count);

            latency_s += steps * latency;
            bandwidth_s += bytes * traffic_multiplier / bandwidth.as_bytes_per_sec();
            bottlenecks.push(format!("node {node_id} intra-node fabric"));
        } else {
            let steps = Self::collective_steps(collective.kind, rank_count);
            let traffic_multiplier = Self::traffic_multiplier(collective.kind, rank_count);
            if let Some(cost) = Self::estimate_graph_collective_cost(
                cluster,
                &participant_addrs,
                Bytes::from_bytes((bytes * traffic_multiplier).ceil() as u64),
                steps,
            ) {
                latency_s += cost.latency_s;
                bandwidth_s += cost.bandwidth_s;
                bottlenecks.extend(cost.bottlenecks);
            } else {
                let (intra_latency, intra_bandwidth) =
                    Self::intra_node_profile(cluster, *nodes.first().unwrap());
                let (fabric_latency, fabric_bandwidth) =
                    Self::inter_node_profile_for_nodes(cluster, &nodes);
                let nic_bandwidth = Self::effective_nic_bandwidth(cluster, &participant_addrs);
                let effective_inter_bandwidth = fabric_bandwidth
                    .as_bytes_per_sec()
                    .min(nic_bandwidth.as_bytes_per_sec());

                latency_s += steps * (intra_latency + fabric_latency);
                bandwidth_s += bytes * traffic_multiplier / intra_bandwidth.as_bytes_per_sec();
                bandwidth_s += bytes * traffic_multiplier / effective_inter_bandwidth;
                bottlenecks.push("inter-node fabric".to_string());
                bottlenecks.push("node NIC bandwidth".to_string());
            }
        }

        latency_s *= calibration.collective_latency_scale;
        bandwidth_s /= calibration.collective_bandwidth_scale;

        CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks,
        }
    }

    pub fn estimate_transfer_between_nodes(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        Self::estimate_transfer_between_nodes_with_options(
            cluster,
            source_nodes,
            dest_nodes,
            bytes,
            SolverOptions {
                calibration,
                calibration_profile: None,
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        )
    }

    pub fn estimate_transfer_between_nodes_with_options(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> CollectiveCost {
        Self::estimate_transfer_between_nodes_with_observation(
            cluster,
            source_nodes,
            dest_nodes,
            bytes,
            options,
        )
        .0
    }

    pub fn estimate_transfer_between_nodes_with_observation(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> (CollectiveCost, Option<CalibrationFitApplication>) {
        let mut cost = Self::estimate_transfer_between_nodes_baseline(
            cluster,
            source_nodes,
            dest_nodes,
            bytes,
            options.calibration,
        );
        if bytes.as_bytes() > 0
            && let Some(evaluation) =
                Self::fitted_kv_transfer_latency(options.calibration_profile, bytes, &cost)
        {
            cost.latency_s = cost.latency_s.min(evaluation.seconds);
            cost.bandwidth_s = (evaluation.seconds - cost.latency_s).max(0.0);
            cost.total_s = evaluation.seconds;
            return (cost, Some(evaluation.application));
        }
        (cost, None)
    }

    pub fn estimate_transfer_between_gpus_with_options(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> CollectiveCost {
        Self::estimate_transfer_between_gpus_with_observation(
            cluster,
            source_gpus,
            dest_gpus,
            bytes,
            options,
        )
        .0
    }

    pub fn estimate_transfer_between_gpus_with_observation(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> (CollectiveCost, Option<CalibrationFitApplication>) {
        let mut cost = Self::estimate_transfer_between_gpus_baseline(
            cluster,
            source_gpus,
            dest_gpus,
            bytes,
            options.calibration,
        );
        if bytes.as_bytes() > 0
            && let Some(evaluation) =
                Self::fitted_kv_transfer_latency(options.calibration_profile, bytes, &cost)
        {
            cost.latency_s = cost.latency_s.min(evaluation.seconds);
            cost.bandwidth_s = (evaluation.seconds - cost.latency_s).max(0.0);
            cost.total_s = evaluation.seconds;
            return (cost, Some(evaluation.application));
        }
        (cost, None)
    }

    fn estimate_transfer_between_nodes_baseline(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        let calibration = calibration.sanitized();
        let nodes: BTreeSet<_> = source_nodes
            .iter()
            .chain(dest_nodes.iter())
            .copied()
            .collect();
        if bytes.as_bytes() == 0 || source_nodes.is_empty() || dest_nodes.is_empty() {
            return CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            };
        }

        if let Some(mut cost) =
            Self::estimate_graph_node_transfer_cost(cluster, source_nodes, dest_nodes, bytes)
        {
            cost.latency_s *= calibration.collective_latency_scale;
            cost.bandwidth_s = cost.bandwidth_s * calibration.kv_transfer_scale
                / calibration.collective_bandwidth_scale;
            cost.total_s = cost.latency_s + cost.bandwidth_s;
            return cost;
        }

        if matches!(&cluster.inter_node_topology, InterNodeTopology::Custom(_)) {
            return CollectiveCost {
                latency_s: f64::INFINITY,
                bandwidth_s: f64::INFINITY,
                total_s: f64::INFINITY,
                bottlenecks: vec!["KV transfer route unavailable".to_string()],
            };
        }

        let (fabric_latency, fabric_bandwidth) =
            Self::inter_node_profile_for_nodes(cluster, &nodes);
        let mut nic_bytes_per_sec = 0.0_f64;
        for node_id in &nodes {
            if let Some(node) = cluster.node(*node_id) {
                nic_bytes_per_sec += node.network.active_nic_bandwidth_bytes_per_sec();
            }
        }
        let effective_bandwidth = fabric_bandwidth
            .as_bytes_per_sec()
            .min(nic_bytes_per_sec.max(1.0));
        let latency_s = fabric_latency * calibration.collective_latency_scale;
        let bandwidth_s = (bytes.as_bytes() as f64 / effective_bandwidth)
            * calibration.kv_transfer_scale
            / calibration.collective_bandwidth_scale;

        CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks: vec!["KV transfer fabric/NIC path".to_string()],
        }
    }

    fn estimate_transfer_between_gpus_baseline(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        let calibration = calibration.sanitized();
        if bytes.as_bytes() == 0 || source_gpus.is_empty() || dest_gpus.is_empty() {
            return CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            };
        }

        let Some(mut cost) =
            Self::estimate_graph_gpu_transfer_cost(cluster, source_gpus, dest_gpus, bytes)
        else {
            return CollectiveCost {
                latency_s: f64::INFINITY,
                bandwidth_s: f64::INFINITY,
                total_s: f64::INFINITY,
                bottlenecks: vec!["KV transfer route unavailable".to_string()],
            };
        };
        cost.latency_s *= calibration.collective_latency_scale;
        cost.bandwidth_s = cost.bandwidth_s * calibration.kv_transfer_scale
            / calibration.collective_bandwidth_scale;
        cost.total_s = cost.latency_s + cost.bandwidth_s;
        cost
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn fitted_phase_latency(
        profile: Option<&CalibrationProfileMetadata>,
        phase: &str,
        targets: &[&str],
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        extra_features: Option<&BTreeMap<String, f64>>,
        baseline_s: Option<f64>,
    ) -> Option<FitEvaluation> {
        let profile = profile?;
        let features = calibration_feature_values(model, request, config, extra_features);
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(fit, phase, targets) {
                return None;
            }
            evaluate_fit(fit, phase, &features, baseline_s)
        })
    }

    pub(crate) fn fitted_latency_from_features(
        profile: Option<&CalibrationProfileMetadata>,
        phase: &str,
        targets: &[&str],
        features: &BTreeMap<String, f64>,
        baseline_s: Option<f64>,
    ) -> Option<CalibrationFitApplication> {
        let profile = profile?;
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(fit, phase, targets) {
                return None;
            }
            evaluate_fit(fit, phase, features, baseline_s).map(|evaluation| evaluation.application)
        })
    }

    pub(crate) fn fitted_value_from_features(
        profile: Option<&CalibrationProfileMetadata>,
        phase: &str,
        targets: &[&str],
        features: &BTreeMap<String, f64>,
        baseline_value: Option<f64>,
        prediction_kind: &str,
        prediction_unit: Option<&str>,
    ) -> Option<CalibrationFitApplication> {
        let profile = profile?;
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(fit, phase, targets) {
                return None;
            }
            evaluate_value_fit(
                fit,
                phase,
                features,
                baseline_value,
                prediction_kind,
                prediction_unit,
            )
        })
    }

    fn fitted_kv_transfer_latency(
        profile: Option<&CalibrationProfileMetadata>,
        bytes: Bytes,
        cost: &CollectiveCost,
    ) -> Option<FitEvaluation> {
        let mut features = BTreeMap::new();
        let bytes = bytes.as_bytes() as f64;
        features.insert("kv_transfer_bytes".to_string(), bytes);
        features.insert("transfer_bytes".to_string(), bytes);
        features.insert("kv_transfer_mb".to_string(), bytes / 1e6);
        features.insert("kv_transfer_gb".to_string(), bytes / 1e9);
        features.insert("transfer_mb".to_string(), bytes / 1e6);
        features.insert("transfer_gb".to_string(), bytes / 1e9);
        features.insert("latency_ms".to_string(), cost.latency_s * 1000.0);
        features.insert("bandwidth_ms".to_string(), cost.bandwidth_s * 1000.0);
        features.insert("baseline_ms".to_string(), cost.total_s * 1000.0);
        features.insert("hop_count".to_string(), cost.bottlenecks.len() as f64);
        let effective_gbps =
            if bytes > 0.0 && cost.bandwidth_s.is_finite() && cost.bandwidth_s > 0.0 {
                bytes * 8.0 / cost.bandwidth_s / 1e9
            } else {
                0.0
            };
        features.insert("rail_bandwidth_gbps".to_string(), effective_gbps);
        features.insert("effective_bandwidth_gbps".to_string(), effective_gbps);

        let profile = profile?;
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(
                fit,
                "kv_transfer",
                &[
                    "kv_transfer_ms",
                    "kv_transfer_latency_ms",
                    "transfer_ms",
                    "transfer_latency_ms",
                    "kv_transfer_s",
                ],
            ) {
                return None;
            }
            evaluate_fit(fit, "kv_transfer", &features, Some(cost.total_s))
        })
    }

    pub fn transfer_between_nodes_routable(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
    ) -> bool {
        if bytes.as_bytes() == 0 || source_nodes.is_empty() || dest_nodes.is_empty() {
            return true;
        }
        match &cluster.inter_node_topology {
            InterNodeTopology::FatTree { .. } | InterNodeTopology::Flat { .. } => true,
            InterNodeTopology::Custom(_) => {
                Self::estimate_graph_node_transfer_cost(cluster, source_nodes, dest_nodes, bytes)
                    .is_some()
            }
        }
    }

    pub fn transfer_between_gpus_routable(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
    ) -> bool {
        if bytes.as_bytes() == 0 || source_gpus.is_empty() || dest_gpus.is_empty() {
            return true;
        }
        Self::estimate_graph_gpu_transfer_cost(cluster, source_gpus, dest_gpus, bytes).is_some()
    }

    fn estimate_graph_collective_cost(
        cluster: &Cluster,
        participant_addrs: &[GpuAddr],
        route_bytes: Bytes,
        latency_steps: f64,
    ) -> Option<CollectiveCost> {
        let graph = TopologyGraph::from_cluster(cluster);
        let mut paths = Vec::new();

        for i in 0..participant_addrs.len() {
            for j in i + 1..participant_addrs.len() {
                let src = participant_addrs[i];
                let dst = participant_addrs[j];
                if src.node_id == dst.node_id {
                    continue;
                }
                paths.push(graph.route_between_gpus(src, dst, route_bytes)?);
            }
        }

        if paths.is_empty() {
            return None;
        }

        let bytes_per_path =
            Bytes::from_bytes(div_ceil(route_bytes.as_bytes(), paths.len() as u64));
        let (latency_s, bandwidth_s, bottlenecks) =
            Self::contended_paths_cost(&paths, bytes_per_path);

        Some(CollectiveCost {
            latency_s: latency_steps * latency_s,
            bandwidth_s,
            total_s: latency_steps * latency_s + bandwidth_s,
            bottlenecks,
        })
    }

    fn estimate_graph_node_transfer_cost(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
    ) -> Option<CollectiveCost> {
        let graph = TopologyGraph::from_cluster(cluster);
        let mut paths = Vec::new();

        for src in source_nodes {
            for dst in dest_nodes {
                if src == dst {
                    continue;
                }
                paths.push(graph.route_between_nodes(*src, *dst, bytes)?);
            }
        }

        if paths.is_empty() {
            return Some(CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            });
        }

        let bytes_per_path = Bytes::from_bytes(div_ceil(bytes.as_bytes(), paths.len() as u64));
        let (latency_s, bandwidth_s, mut bottlenecks) =
            Self::contended_paths_cost(&paths, bytes_per_path);
        if !bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "KV transfer fabric/NIC path")
        {
            bottlenecks.push("KV transfer fabric/NIC path".to_string());
        }

        Some(CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks,
        })
    }

    fn estimate_graph_gpu_transfer_cost(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
    ) -> Option<CollectiveCost> {
        let graph = TopologyGraph::from_cluster(cluster);
        let mut paths = Vec::new();
        let mut has_inter_node_path = false;

        for src in source_gpus {
            for dst in dest_gpus {
                if src == dst {
                    continue;
                }
                if src.node_id == dst.node_id {
                    paths.push(Self::intra_node_gpu_path(cluster, src.node_id));
                } else {
                    has_inter_node_path = true;
                    let path = graph.route_between_gpus(*src, *dst, bytes).or_else(|| {
                        if Self::can_use_node_level_gpu_path(cluster, *src)
                            && Self::can_use_node_level_gpu_path(cluster, *dst)
                        {
                            graph
                                .route_between_nodes(src.node_id, dst.node_id, bytes)
                                .map(|path| {
                                    Self::add_endpoint_intra_node_resources(
                                        cluster,
                                        path,
                                        src.node_id,
                                        dst.node_id,
                                    )
                                })
                        } else {
                            None
                        }
                    })?;
                    paths.push(path);
                }
            }
        }

        if paths.is_empty() {
            return Some(CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            });
        }

        let bytes_per_path = Bytes::from_bytes(div_ceil(bytes.as_bytes(), paths.len() as u64));
        let (latency_s, bandwidth_s, mut bottlenecks) =
            Self::contended_paths_cost(&paths, bytes_per_path);
        if has_inter_node_path
            && !bottlenecks
                .iter()
                .any(|bottleneck| bottleneck == "KV transfer fabric/NIC path")
        {
            bottlenecks.push("KV transfer fabric/NIC path".to_string());
        }

        Some(CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks,
        })
    }

    fn intra_node_gpu_path(cluster: &Cluster, node_id: NodeId) -> RoutedPath {
        let (latency_s, bandwidth) = Self::intra_node_link_profile(cluster, node_id);
        let label = format!("node {node_id} intra-node fabric");
        RoutedPath {
            latency_s,
            bottleneck_bandwidth: bandwidth,
            labels: vec![label.clone()],
            resources: vec![RoutedResource {
                label,
                kind: RoutedResourceKind::IntraNodeFabric,
                from: GraphResource::IntraNode { node_id },
                to: GraphResource::IntraNode { node_id },
                bandwidth,
                latency_s,
                rail_id: None,
            }],
        }
    }

    fn can_use_node_level_gpu_path(cluster: &Cluster, gpu: GpuAddr) -> bool {
        cluster
            .node(gpu.node_id)
            .is_some_and(|node| !node.network.gpu_nic_map.contains_key(&gpu.local_gpu_id))
    }

    fn add_endpoint_intra_node_resources(
        cluster: &Cluster,
        mut path: RoutedPath,
        source_node: NodeId,
        dest_node: NodeId,
    ) -> RoutedPath {
        Self::push_intra_node_resource(cluster, &mut path, source_node);
        Self::push_intra_node_resource(cluster, &mut path, dest_node);
        path
    }

    fn push_intra_node_resource(cluster: &Cluster, path: &mut RoutedPath, node_id: NodeId) {
        let (latency_s, bandwidth) = Self::intra_node_link_profile(cluster, node_id);
        let label = format!("node {node_id} intra-node fabric");
        path.latency_s += latency_s;
        if bandwidth.as_bytes_per_sec() < path.bottleneck_bandwidth.as_bytes_per_sec() {
            path.bottleneck_bandwidth = bandwidth;
        }
        if !path.labels.contains(&label) {
            path.labels.push(label.clone());
        }
        path.resources.push(RoutedResource {
            label,
            kind: RoutedResourceKind::IntraNodeFabric,
            from: GraphResource::IntraNode { node_id },
            to: GraphResource::IntraNode { node_id },
            bandwidth,
            latency_s,
            rail_id: None,
        });
    }

    fn contended_paths_cost(
        paths: &[RoutedPath],
        bytes_per_path: Bytes,
    ) -> (f64, f64, Vec<String>) {
        let mut resource_counts = std::collections::HashMap::new();
        for path in paths {
            let mut seen_in_path = HashSet::new();
            for resource in &path.resources {
                if seen_in_path.insert(resource.label.clone()) {
                    *resource_counts
                        .entry(resource.label.clone())
                        .or_insert(0_u32) += 1;
                }
            }
        }

        let mut max_latency_s = 0.0_f64;
        let mut max_bandwidth_s = 0.0_f64;
        let mut bottlenecks = Vec::new();
        for path in paths {
            max_latency_s = max_latency_s.max(path.latency_s);
            let mut path_bandwidth = f64::INFINITY;
            let mut path_bottleneck = None;
            for resource in &path.resources {
                let count = f64::from(*resource_counts.get(&resource.label).unwrap_or(&1));
                let shared_bandwidth = resource.bandwidth.as_bytes_per_sec() / count.max(1.0);
                if shared_bandwidth < path_bandwidth {
                    path_bandwidth = shared_bandwidth;
                    path_bottleneck = Some(resource.label.clone());
                }
            }

            if let Some(path_bottleneck) = path_bottleneck
                && !bottlenecks.contains(&path_bottleneck)
            {
                bottlenecks.push(path_bottleneck);
            }
            max_bandwidth_s =
                max_bandwidth_s.max(bytes_per_path.as_bytes() as f64 / path_bandwidth.max(1.0));
        }

        (max_latency_s, max_bandwidth_s, bottlenecks)
    }

    fn collective_steps(kind: CollectiveKind, rank_count: f64) -> f64 {
        match kind {
            CollectiveKind::SendRecv | CollectiveKind::Broadcast => 1.0,
            CollectiveKind::AllReduce => 2.0 * (rank_count - 1.0).max(1.0),
            CollectiveKind::AllGather
            | CollectiveKind::ReduceScatter
            | CollectiveKind::AllToAll => (rank_count - 1.0).max(1.0),
        }
    }

    fn traffic_multiplier(kind: CollectiveKind, rank_count: f64) -> f64 {
        match kind {
            CollectiveKind::SendRecv | CollectiveKind::Broadcast => 1.0,
            CollectiveKind::AllReduce => 2.0 * (rank_count - 1.0) / rank_count,
            CollectiveKind::AllGather | CollectiveKind::ReduceScatter => {
                (rank_count - 1.0) / rank_count
            }
            CollectiveKind::AllToAll => (rank_count - 1.0) / rank_count,
        }
        .max(1.0 / rank_count)
    }

    fn intra_node_profile(cluster: &Cluster, node_id: NodeId) -> (f64, Bandwidth) {
        let Some(node) = cluster.node(node_id) else {
            return (0.0, Bandwidth::from_bytes_per_sec(1.0));
        };

        match &node.intra_node_fabric {
            IntraNodeTopology::NvSwitch(profile) | IntraNodeTopology::NvLinkDirect(profile) => {
                (profile.latency.to_us() / 1e6, profile.bw.unidirectional)
            }
            IntraNodeTopology::Pcie(profile) => {
                (profile.latency.to_us() / 1e6, profile.bw.unidirectional)
            }
            IntraNodeTopology::Custom(edges) => {
                let mut min_bw = None;
                let mut max_latency = 0.0_f64;
                for edge in edges.values() {
                    match edge {
                        crate::types::fabric::intra_node::LinkProfile::NvLink(profile) => {
                            min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                            max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                        }
                        crate::types::fabric::intra_node::LinkProfile::Pcie(profile) => {
                            min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                            max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                        }
                    }
                }
                (
                    max_latency,
                    min_bw.unwrap_or_else(|| Bandwidth::from_bytes_per_sec(1.0)),
                )
            }
        }
    }

    pub(crate) fn intra_node_link_profile(cluster: &Cluster, node_id: NodeId) -> (f64, Bandwidth) {
        Self::intra_node_profile(cluster, node_id)
    }

    fn inter_node_profile(cluster: &Cluster) -> (f64, Bandwidth) {
        match &cluster.inter_node_topology {
            InterNodeTopology::FatTree {
                link,
                oversubscription,
                ..
            } => (
                link.latency.to_us() / 1e6,
                link.bw.unidirectional / oversubscription.max(1.0),
            ),
            InterNodeTopology::Flat { link } => {
                (link.latency.to_us() / 1e6, link.bw.unidirectional)
            }
            InterNodeTopology::Custom(edges) => {
                let mut min_bw = None;
                let mut max_latency = 0.0_f64;
                for profile in edges.values().flatten().map(|link| &link.profile) {
                    min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                    max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                }
                (
                    max_latency,
                    min_bw.unwrap_or_else(|| Bandwidth::from_bytes_per_sec(1.0)),
                )
            }
        }
    }

    fn inter_node_profile_for_nodes(
        cluster: &Cluster,
        nodes: &BTreeSet<NodeId>,
    ) -> (f64, Bandwidth) {
        match &cluster.inter_node_topology {
            InterNodeTopology::Custom(edges) if nodes.len() > 1 => {
                let mut min_bw = None;
                let mut max_latency = 0.0_f64;
                let node_ids: Vec<_> = nodes.iter().copied().collect();

                for i in 0..node_ids.len() {
                    for j in i + 1..node_ids.len() {
                        if let Some(links) =
                            edges.get(&UnorderedPair::new(node_ids[i], node_ids[j]))
                        {
                            for profile in links.iter().map(|link| &link.profile) {
                                min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                                max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                            }
                        }
                    }
                }

                if let Some(min_bw) = min_bw {
                    (max_latency, min_bw)
                } else {
                    Self::inter_node_profile(cluster)
                }
            }
            _ => Self::inter_node_profile(cluster),
        }
    }

    fn effective_nic_bandwidth(cluster: &Cluster, addrs: &[GpuAddr]) -> Bandwidth {
        let mut total = 0.0;
        let mut seen = HashSet::new();

        for addr in addrs {
            let Some(node) = cluster.node(addr.node_id) else {
                continue;
            };
            for nic_id in node.network.nic_candidates_for_gpu(addr.local_gpu_id) {
                if seen.insert((addr.node_id, nic_id)) {
                    total += node.network.nic_bandwidth(nic_id).as_bytes_per_sec();
                }
            }
        }

        Bandwidth::from_bytes_per_sec(total.max(1.0))
    }
}
