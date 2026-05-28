use super::*;

impl Solver {
    pub(super) fn reject(
        mut score: ScoredParallelismConfig,
        reason: String,
    ) -> ScoredParallelismConfig {
        score.feasible = false;
        score.estimated_latency_s = f64::INFINITY;
        score.rejected_reason = Some(reason);
        score
    }

    pub(super) fn placement_evidence(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        placement: &RankPlacement,
        required_memory_per_gpu: Bytes,
        strategy_code: &str,
    ) -> Vec<PlacementEvidence> {
        let total_ranks = config.total_ranks();
        let available_gpus = cluster.available_gpus();
        let eligible_gpus =
            Self::sorted_eligible_gpus(cluster, model, required_memory_per_gpu).len() as u32;
        let strategy_message = if strategy_code == "explicit_rank_placement" {
            format!(
                "selected {total_ranks} ranks from explicit rank placement config over {available_gpus} available GPUs"
            )
        } else {
            format!(
                "selected {total_ranks} ranks from {available_gpus} available GPUs using deterministic capability and HBM ordering"
            )
        };
        let mut evidence = vec![
            PlacementEvidence::new(
                "selected",
                "rank_placement",
                "rank_placement",
                strategy_code,
                strategy_message,
            )
            .with_details(
                Some(f64::from(total_ranks)),
                Some(f64::from(available_gpus)),
                Some("ranks"),
                Some(
                    "add topology-aware placement constraints before relying on fine-grained locality",
                ),
            ),
            PlacementEvidence::new(
                "selected",
                "gpu_eligibility",
                "eligible_gpus",
                "hbm_capable_gpus_available",
                format!(
                    "{eligible_gpus} GPUs have enough HBM for {:.2} GB per-rank memory",
                    required_memory_per_gpu.as_gigabytes()
                ),
            )
            .with_details(
                Some(f64::from(eligible_gpus)),
                Some(f64::from(total_ranks)),
                Some("gpus"),
                Some("increase HBM capacity, reduce model/request memory, or reduce rank count"),
            ),
        ];

        if placement_spans_nodes(placement) {
            evidence.push(
                PlacementEvidence::new(
                    "selected",
                    "topology",
                    "inter_node",
                    "placement_spans_nodes",
                    "selected rank placement spans multiple nodes and depends on modeled inter-node communication",
                )
                .with_details(
                    Some(placement_node_count(placement) as f64),
                    None,
                    Some("nodes"),
                    Some(
                        "model explicit NIC, rail, rack, and fabric resources before treating this as exact locality",
                    ),
                ),
            );
        }

        evidence
    }

    pub(super) fn placement_capacity_evidence(
        total_ranks: u32,
        available_gpus: u32,
        reason: &str,
    ) -> PlacementEvidence {
        PlacementEvidence::new(
            "rejected",
            "rank_placement",
            "available_gpus",
            "insufficient_available_gpus",
            reason,
        )
        .with_details(
            Some(f64::from(available_gpus)),
            Some(f64::from(total_ranks)),
            Some("gpus"),
            Some("reduce total ranks, add GPUs, or re-enable unavailable GPUs"),
        )
    }

    pub(super) fn explicit_placement_rejection(
        resource: String,
        code: &str,
        message: String,
        observed: Option<f64>,
        limit: Option<f64>,
        unit: Option<&str>,
        remediation: &str,
    ) -> PlacementEvidence {
        PlacementEvidence::new("rejected", "rank_placement", resource, code, message).with_details(
            observed,
            limit,
            unit,
            Some(remediation),
        )
    }

    pub(super) fn place_explicit_ranks(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        placement: &RankPlacement,
        required_memory_per_gpu: Bytes,
    ) -> Result<RankPlacement, Box<PlacementEvidence>> {
        let total_ranks = config.total_ranks() as usize;
        if placement.rank_to_gpu.len() != total_ranks {
            let observed = placement.rank_to_gpu.len();
            let message = format!(
                "explicit placement has {observed} ranks but config requires {total_ranks} ranks"
            );
            return Err(Box::new(Self::explicit_placement_rejection(
                "rank_count".to_string(),
                "explicit_placement_rank_count_mismatch",
                message,
                Some(observed as f64),
                Some(total_ranks as f64),
                Some("ranks"),
                "match placement.ranks length to tensor*pipeline*expert*data ranks",
            )));
        }

        let mut used_gpus = BTreeSet::new();
        for (rank, addr) in placement.rank_to_gpu.iter().copied().enumerate() {
            let resource = format!(
                "rank {rank} node {} gpu {}",
                addr.node_id, addr.local_gpu_id
            );
            let Some(profile) = cluster.gpu_profile(addr) else {
                let message = format!(
                    "explicit placement rank {rank} references unknown node {} gpu {}",
                    addr.node_id, addr.local_gpu_id
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_unknown_gpu",
                    message,
                    None,
                    None,
                    None,
                    "choose a GPU that exists in the cluster inventory",
                )));
            };
            if !cluster.is_gpu_available(addr) {
                let message = format!(
                    "explicit placement rank {rank} references unavailable node {} gpu {}",
                    addr.node_id, addr.local_gpu_id
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_gpu_unavailable",
                    message,
                    None,
                    None,
                    None,
                    "choose an available GPU or remove the disabled/maintenance overlay",
                )));
            }
            if !Self::gpu_supports_model_dtype(&profile, model) {
                let message = format!(
                    "explicit placement rank {rank} uses node {} gpu {} ({}) which does not support model.dtype {}",
                    addr.node_id,
                    addr.local_gpu_id,
                    profile.label,
                    Self::model_dtype_label(model)
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_dtype_unsupported",
                    message,
                    None,
                    None,
                    None,
                    "choose GPUs whose tensor throughput metadata supports the model dtype",
                )));
            }
            if !used_gpus.insert(addr) {
                let message = format!(
                    "explicit placement maps more than one rank to node {} gpu {}",
                    addr.node_id, addr.local_gpu_id
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_duplicate_gpu",
                    message,
                    None,
                    None,
                    None,
                    "map each rank to a unique GPU in the current rank-placement model",
                )));
            }
            if required_memory_per_gpu > profile.hbm_size {
                let message = format!(
                    "explicit placement rank {rank} requires {:.2} GB but node {} gpu {} has {:.2} GB HBM",
                    required_memory_per_gpu.as_gigabytes(),
                    addr.node_id,
                    addr.local_gpu_id,
                    profile.hbm_size.as_gigabytes()
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_hbm_exceeded",
                    message,
                    Some(required_memory_per_gpu.as_gigabytes()),
                    Some(profile.hbm_size.as_gigabytes()),
                    Some("GB"),
                    "reduce per-rank memory or choose a larger-HBM GPU",
                )));
            }
        }

        Ok(placement.clone())
    }

    pub(super) fn placement_hbm_evidence(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        required_memory_per_gpu: Bytes,
        reason: &str,
    ) -> PlacementEvidence {
        let eligible_gpus =
            Self::sorted_eligible_gpus(cluster, model, required_memory_per_gpu).len() as u32;
        PlacementEvidence::new(
            "rejected",
            "gpu_eligibility",
            "eligible_gpus",
            "insufficient_hbm_eligible_gpus",
            reason,
        )
        .with_details(
            Some(f64::from(eligible_gpus)),
            Some(f64::from(config.total_ranks())),
            Some("gpus"),
            Some("reduce memory per rank, increase tensor/pipeline sharding, or choose larger-HBM GPUs"),
        )
    }

    pub(super) fn placement_memory_rejection_evidence(
        required_memory_per_gpu: Bytes,
        reason: &str,
    ) -> PlacementEvidence {
        PlacementEvidence::new(
            "rejected",
            "memory",
            "gpu_hbm",
            "placement_hbm_headroom_exceeded",
            reason,
        )
        .with_details(
            Some(required_memory_per_gpu.as_gigabytes()),
            None,
            Some("GB"),
            Some("reduce model/request memory or place ranks on GPUs with more HBM"),
        )
    }

    pub(super) fn parallelism_approximations(
        cluster: &Cluster,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        operations: &[SimOperation],
        calibration_fits: &[CalibrationFitApplication],
    ) -> Vec<SimulationApproximation> {
        let mut approximations = Vec::new();
        let phase = phase_label(request.phase);
        push_approximation(
            &mut approximations,
            SimulationApproximation::new(
                phase,
                "placement",
                "rank_placement",
                "capability_ordered_rank_placement",
                "Rank placement is deterministic and capability-aware, but it is not a full topology/NUMA/rail-aware placement search.",
                Some(
                    "supply explicit placement constraints or add topology-aware placement search before relying on fine-grained locality conclusions"
                        .to_string(),
                ),
            ),
        );
        push_approximation(
            &mut approximations,
            SimulationApproximation::new(
                phase,
                "memory",
                "per_gpu_hbm",
                "static_per_gpu_memory_estimate",
                "Memory is estimated as a static per-GPU requirement and does not model allocator timelines, graph capture reserve, KV-block fragmentation, or transient peaks.",
                Some(
                    "calibrate memory components and add placement/time-aware memory accounting for production capacity planning"
                        .to_string(),
                ),
            ),
        );

        if config.total_ranks() > 1
            && operations
                .iter()
                .any(|op| op.kind == SimOperationKind::Collective)
        {
            push_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    phase,
                    "communication",
                    "collectives",
                    "coarse_collective_model",
                    "Collective operations use calibrated coarse formulas and do not fully model backend algorithm auto-selection, channels, protocol thresholds, or concurrent traffic contention.",
                    Some(
                        "fit collective behavior from backend benchmarks and add algorithm/route contention modeling for topology-sensitive decisions"
                            .to_string(),
                    ),
                ),
            );
        }

        if placement_spans_nodes(placement) {
            let (code, message) = match &cluster.inter_node_topology {
                InterNodeTopology::Custom(_) => (
                    "custom_topology_route_approximation",
                    "Custom inter-node links are routed through coarse path bottlenecks and do not yet model per-NIC/per-rail/per-switch contention.",
                ),
                InterNodeTopology::FatTree { .. } | InterNodeTopology::Flat { .. } => (
                    "aggregate_inter_node_topology",
                    "Inter-node topology is represented by aggregate fabric parameters rather than explicit rack switches, rails, NICs, and oversubscribed uplinks.",
                ),
            };
            push_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    phase,
                    "topology",
                    "inter_node",
                    code,
                    message,
                    Some(
                        "model explicit NICs, rails, switches, and route sharing before making fine-grained fabric locality claims"
                            .to_string(),
                    ),
                ),
            );
        }

        if calibration_fits.iter().any(|fit| {
            fit.max_extrapolation_ratio > 0.0 || fit.applicability_status != "interpolated"
        }) {
            push_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    phase,
                    "calibration",
                    "fitted_models",
                    "calibration_fit_extrapolation",
                    "At least one calibration fit is outside its fitted feature range or is not marked interpolated, so ranking may be sensitive to extrapolation error.",
                    Some(
                        "add benchmark coverage for this workload shape or hard-reject extrapolated fits via calibration policy gates"
                            .to_string(),
                    ),
                ),
            );
        }

        approximations
    }

    pub(super) fn place_ranks(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        required_memory_per_gpu: Bytes,
    ) -> Result<RankPlacement, String> {
        let total_ranks = config.total_ranks();
        let eligible_gpus = Self::sorted_eligible_gpus(cluster, model, required_memory_per_gpu);
        let total_gpus = cluster.available_gpus();

        if total_ranks > total_gpus {
            return Err(format!(
                "config requires {total_ranks} ranks but only {} GPUs are available",
                total_gpus
            ));
        }
        if total_ranks as usize > eligible_gpus.len() {
            return Err(format!(
                "estimated {:.2} GB per GPU and model.dtype {} leave {} of {} GPUs ineligible; config requires {total_ranks} ranks but only {} GPUs have enough HBM and dtype support",
                required_memory_per_gpu.as_gigabytes(),
                Self::model_dtype_label(model),
                total_gpus.saturating_sub(eligible_gpus.len() as u32),
                total_gpus,
                eligible_gpus.len()
            ));
        }

        if let Some(placement) =
            Self::place_by_data_replicas(cluster, model, config, required_memory_per_gpu)
        {
            return Ok(placement);
        }

        Ok(RankPlacement {
            rank_to_gpu: eligible_gpus
                .into_iter()
                .take(total_ranks as usize)
                .collect(),
        })
    }

    pub(super) fn place_by_data_replicas(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        required_memory_per_gpu: Bytes,
    ) -> Option<RankPlacement> {
        let ranks_per_replica = config.tensor_ranks * config.pipeline_ranks * config.expert_ranks;
        let mut node_gpus = Self::eligible_node_gpu_candidates(
            cluster,
            model,
            required_memory_per_gpu,
            ranks_per_replica as usize,
        );

        if node_gpus.len() < config.data_ranks as usize {
            return None;
        }
        node_gpus.truncate(config.data_ranks as usize);

        let mut rank_to_gpu = Vec::with_capacity(config.total_ranks() as usize);
        let mut used = HashSet::new();
        for rank in 0..config.total_ranks() {
            let (dp, pp, ep, tp) = decode_rank(config, rank);
            let (node_id, gpus) = &node_gpus[dp as usize];
            let local_index =
                (pp * config.expert_ranks * config.tensor_ranks) + (ep * config.tensor_ranks) + tp;
            let addr = GpuAddr {
                node_id: *node_id,
                local_gpu_id: gpus[local_index as usize],
            };
            if !used.insert(addr) {
                return None;
            }
            rank_to_gpu.push(addr);
        }

        Some(RankPlacement { rank_to_gpu })
    }

    pub(super) fn sorted_eligible_gpus(
        cluster: &Cluster,
        model: &ModelSpec,
        required_memory_per_gpu: Bytes,
    ) -> Vec<GpuAddr> {
        let mut gpus: Vec<_> = cluster
            .sorted_gpu_addrs()
            .into_iter()
            .filter(|addr| {
                cluster.is_gpu_available(*addr)
                    && cluster.gpu_profile(*addr).is_some_and(|profile| {
                        profile.hbm_size >= required_memory_per_gpu
                            && Self::gpu_supports_model_dtype(&profile, model)
                    })
            })
            .collect();
        gpus.sort_by(|left, right| Self::compare_gpu_addrs(cluster, model, left, right));
        gpus
    }

    pub(super) fn eligible_node_gpu_candidates(
        cluster: &Cluster,
        model: &ModelSpec,
        required_memory_per_gpu: Bytes,
        min_gpus_per_node: usize,
    ) -> Vec<(NodeId, Vec<u32>)> {
        let mut candidates: Vec<(NodeId, Vec<u32>)> = Vec::new();
        let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
        node_ids.sort_unstable();

        for node_id in node_ids {
            let mut gpus: Vec<_> = cluster
                .node(node_id)
                .into_iter()
                .flat_map(|node| node.gpus.keys().copied())
                .map(|local_gpu_id| GpuAddr {
                    node_id,
                    local_gpu_id,
                })
                .filter(|addr| {
                    cluster.is_gpu_available(*addr)
                        && cluster.gpu_profile(*addr).is_some_and(|profile| {
                            profile.hbm_size >= required_memory_per_gpu
                                && Self::gpu_supports_model_dtype(&profile, model)
                        })
                })
                .collect();
            gpus.sort_by(|left, right| Self::compare_gpu_addrs(cluster, model, left, right));
            if gpus.len() >= min_gpus_per_node {
                candidates.push((
                    node_id,
                    gpus.into_iter().map(|addr| addr.local_gpu_id).collect(),
                ));
            }
        }

        candidates.sort_by(|left, right| {
            let left_addr = GpuAddr {
                node_id: left.0,
                local_gpu_id: left.1[0],
            };
            let right_addr = GpuAddr {
                node_id: right.0,
                local_gpu_id: right.1[0],
            };
            Self::compare_gpu_addrs(cluster, model, &left_addr, &right_addr)
        });
        candidates
    }

    pub(super) fn compare_gpu_addrs(
        cluster: &Cluster,
        model: &ModelSpec,
        left: &GpuAddr,
        right: &GpuAddr,
    ) -> std::cmp::Ordering {
        let left_profile = cluster.gpu_profile(*left);
        let right_profile = cluster.gpu_profile(*right);
        match (left_profile, right_profile) {
            (Some(left_profile), Some(right_profile)) => {
                Self::gpu_peak_tflops(&right_profile, model)
                    .total_cmp(&Self::gpu_peak_tflops(&left_profile, model))
                    .then_with(|| {
                        right_profile
                            .hbm_bandwidth
                            .as_bytes_per_sec()
                            .total_cmp(&left_profile.hbm_bandwidth.as_bytes_per_sec())
                    })
                    .then_with(|| {
                        right_profile
                            .hbm_size
                            .as_bytes()
                            .cmp(&left_profile.hbm_size.as_bytes())
                    })
                    .then_with(|| left.node_id.cmp(&right.node_id))
                    .then_with(|| left.local_gpu_id.cmp(&right.local_gpu_id))
            }
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left
                .node_id
                .cmp(&right.node_id)
                .then_with(|| left.local_gpu_id.cmp(&right.local_gpu_id)),
        }
    }

    pub(super) fn gpu_peak_tflops(
        profile: &crate::types::gpu::GpuProfile,
        model: &ModelSpec,
    ) -> f64 {
        match model.dtype {
            crate::workload::DType::Fp8 => profile.peak_f8_flops.unwrap_or(0.0),
            crate::workload::DType::Int8 => profile.peak_f8_flops.unwrap_or(profile.peak_f16_flops),
            crate::workload::DType::Fp16 | crate::workload::DType::Bf16 => profile.peak_f16_flops,
        }
    }

    pub(super) fn gpu_supports_model_dtype(
        profile: &crate::types::gpu::GpuProfile,
        model: &ModelSpec,
    ) -> bool {
        match model.dtype {
            crate::workload::DType::Fp8 => profile.peak_f8_flops.is_some(),
            crate::workload::DType::Fp16
            | crate::workload::DType::Bf16
            | crate::workload::DType::Int8 => true,
        }
    }

    pub(super) fn model_dtype_label(model: &ModelSpec) -> &'static str {
        model.dtype.label()
    }
}
