use super::*;

impl Solver {
    #[cfg(test)]
    pub(super) fn build_collectives(
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        groups: &ParallelGroups,
    ) -> Vec<CollectiveCall> {
        let mut calls = Vec::new();
        let activation_bytes = Self::activation_bytes(model, request);
        let tp_bytes = Bytes::from_bytes(div_ceil(
            activation_bytes.as_bytes(),
            config.tensor_ranks.max(1) as u64,
        ));

        if config.tensor_ranks > 1 {
            for _ in 0..model.layers {
                for group in &groups.tensor_groups {
                    calls.push(CollectiveCall {
                        kind: CollectiveKind::AllReduce,
                        participants: group.clone(),
                        bytes_per_rank: tp_bytes,
                        dtype: model.dtype,
                        reduction: Some(ReductionOp::Sum),
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Hierarchical,
                    });
                }
            }
        }

        if config.pipeline_ranks > 1 {
            for window in groups.pipeline_stages.windows(2) {
                if let [left, right] = window
                    && let (Some(&left_rank), Some(&right_rank)) = (left.first(), right.first())
                {
                    calls.push(CollectiveCall {
                        kind: CollectiveKind::SendRecv,
                        participants: vec![left_rank, right_rank],
                        bytes_per_rank: activation_bytes,
                        dtype: model.dtype,
                        reduction: None,
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Auto,
                    });
                }
            }
        }

        if let Some(experts) = model.experts
            && config.expert_ranks > 1
        {
            let expert_bytes = Bytes::from_bytes(
                activation_bytes.as_bytes() * experts.top_k as u64
                    / config.expert_ranks.max(1) as u64,
            );
            for _ in 0..model.layers {
                for group in &groups.expert_groups {
                    calls.push(CollectiveCall {
                        kind: CollectiveKind::AllToAll,
                        participants: group.clone(),
                        bytes_per_rank: expert_bytes,
                        dtype: model.dtype,
                        reduction: None,
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Hierarchical,
                    });
                }
            }
        }

        calls
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_operation_trace(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        groups: &ParallelGroups,
        compute_latency_s: f64,
        calibration: SimulationCalibration,
    ) -> Vec<SimOperation> {
        let mut operations = Vec::new();
        let layers = model.layers.max(1);
        let compute_per_layer_s = compute_latency_s / f64::from(layers);
        let compute_resources = compute_resources(placement, request.phase);
        let mut previous_compute = None;
        let mut previous_layer_barrier = None;

        for layer_idx in 0..layers {
            let mut compute_dependencies = Vec::new();
            if let Some(previous_compute) = previous_compute {
                compute_dependencies.push(previous_compute);
            }
            if !calibration.allow_compute_comm_overlap
                && let Some(previous_layer_barrier) = previous_layer_barrier
            {
                compute_dependencies.push(previous_layer_barrier);
            }

            let compute_idx = operations.len();
            operations.push(SimOperation {
                name: format!("layer {layer_idx} compute"),
                kind: SimOperationKind::Compute,
                duration_s: compute_per_layer_s,
                resources: compute_resources.clone(),
                dependencies: compute_dependencies,
            });
            previous_compute = Some(compute_idx);

            let mut layer_tail = compute_idx;
            if config.tensor_ranks > 1 {
                for (group_idx, group) in groups.tensor_groups.iter().enumerate() {
                    let collective = Self::collective_call(
                        CollectiveKind::AllReduce,
                        group.clone(),
                        model,
                        request,
                        config,
                        Some(ReductionOp::Sum),
                    );
                    let op_idx = Self::push_collective_operation(
                        cluster,
                        placement,
                        &mut operations,
                        format!("layer {layer_idx} tp all-reduce group {group_idx}"),
                        collective,
                        compute_idx,
                        calibration,
                    );
                    layer_tail = op_idx;
                }
            }

            if model.experts.is_some() && config.expert_ranks > 1 {
                for (group_idx, group) in groups.expert_groups.iter().enumerate() {
                    let collective = Self::collective_call(
                        CollectiveKind::AllToAll,
                        group.clone(),
                        model,
                        request,
                        config,
                        None,
                    );
                    let op_idx = Self::push_collective_operation(
                        cluster,
                        placement,
                        &mut operations,
                        format!("layer {layer_idx} ep all-to-all group {group_idx}"),
                        collective,
                        compute_idx,
                        calibration,
                    );
                    layer_tail = op_idx;
                }
            }

            previous_layer_barrier = Some(layer_tail);
        }

        if config.pipeline_ranks > 1 {
            let activation_bytes = Self::activation_bytes(model, request);
            for (edge_idx, window) in groups.pipeline_stages.windows(2).enumerate() {
                if let [left, right] = window
                    && let (Some(&left_rank), Some(&right_rank)) = (left.first(), right.first())
                {
                    let collective = CollectiveCall {
                        kind: CollectiveKind::SendRecv,
                        participants: vec![left_rank, right_rank],
                        bytes_per_rank: activation_bytes,
                        dtype: model.dtype,
                        reduction: None,
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Auto,
                    };
                    let dependency = previous_layer_barrier.or(previous_compute).unwrap_or(0);
                    Self::push_collective_operation(
                        cluster,
                        placement,
                        &mut operations,
                        format!("pipeline sendrecv edge {edge_idx}"),
                        collective,
                        dependency,
                        calibration,
                    );
                }
            }
        }

        operations
    }

    fn collective_call(
        kind: CollectiveKind,
        participants: Vec<RankId>,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        reduction: Option<ReductionOp>,
    ) -> CollectiveCall {
        let activation_bytes = Self::activation_bytes(model, request);
        let bytes_per_rank = match kind {
            CollectiveKind::AllReduce => Bytes::from_bytes(div_ceil(
                activation_bytes.as_bytes(),
                config.tensor_ranks.max(1) as u64,
            )),
            CollectiveKind::AllToAll => {
                let top_k = model.experts.map(|experts| experts.top_k).unwrap_or(1);
                Bytes::from_bytes(
                    activation_bytes.as_bytes() * top_k as u64 / config.expert_ranks.max(1) as u64,
                )
            }
            _ => activation_bytes,
        };

        CollectiveCall {
            kind,
            participants,
            bytes_per_rank,
            dtype: model.dtype,
            reduction,
            root: None,
            phase: request.phase,
            algorithm: CollectiveAlgorithm::Hierarchical,
        }
    }

    fn push_collective_operation(
        cluster: &Cluster,
        placement: &RankPlacement,
        operations: &mut Vec<SimOperation>,
        name: String,
        collective: CollectiveCall,
        dependency: usize,
        calibration: SimulationCalibration,
    ) -> usize {
        let cost = Self::estimate_collective_with_calibration(
            cluster,
            placement,
            &collective,
            calibration,
        );
        let resources = if cost.bottlenecks.is_empty() {
            vec!["communication".to_string()]
        } else {
            cost.bottlenecks
        };
        let op_idx = operations.len();
        operations.push(SimOperation {
            name,
            kind: SimOperationKind::Collective,
            duration_s: cost.total_s,
            resources,
            dependencies: vec![dependency],
        });
        op_idx
    }
}

pub fn schedule_operations(operations: &[SimOperation]) -> Vec<ScheduledOperation> {
    let mut scheduler = ResourceScheduler::new();
    let mut scheduled_ids = Vec::with_capacity(operations.len());

    for operation in operations {
        let dependencies: Vec<_> = operation
            .dependencies
            .iter()
            .filter_map(|idx| scheduled_ids.get(*idx).copied())
            .collect();
        let scheduled_id = scheduler.schedule(
            operation.name.clone(),
            operation.duration_s,
            0.0,
            &dependencies,
            operation.resources.clone(),
        );
        scheduled_ids.push(scheduled_id);
    }

    scheduler.operations().to_vec()
}

fn compute_resources(placement: &RankPlacement, phase: InferencePhase) -> Vec<String> {
    let mut resources: Vec<_> = placement
        .rank_to_gpu
        .iter()
        .map(|addr| format!("gpu compute node {}", addr.node_id))
        .collect();
    if matches!(phase, InferencePhase::Decode | InferencePhase::EndToEnd) {
        resources.extend(
            placement
                .rank_to_gpu
                .iter()
                .map(|addr| format!("gpu HBM node {}", addr.node_id)),
        );
    }
    resources.sort();
    resources.dedup();
    resources
}
