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

        if config.tensor_ranks > 1 {
            for _ in 0..model.layers * 2 {
                for group in &groups.tensor_groups {
                    calls.push(CollectiveCall {
                        kind: CollectiveKind::AllReduce,
                        participants: group.clone(),
                        bytes_per_rank: activation_bytes,
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
    ) -> (Vec<SimOperation>, Vec<PricedCollective>) {
        let mut operations = Vec::new();
        let mut priced = Vec::new();
        let layers = model.layers.max(1);
        let stages = config.pipeline_ranks.max(1);
        let compute_per_layer_s = compute_latency_s / f64::from(layers);
        let stage_plans: Vec<_> = (0..stages)
            .map(|stage| PipelineStagePlan::new(placement, groups, stage, request.phase))
            .collect();
        let mut previous_compute = None;
        let mut previous_layer_barrier = None;
        let mut current_stage = 0;

        // Tensor parallelism shards the embedding table by vocab, so the
        // looked-up activations are all-reduced before the first layer.
        let mut embedding_reductions = Vec::new();
        if config.tensor_ranks > 1 {
            let activation_token_bytes = Self::activation_token_bytes(model, request);
            for (group_idx, group) in groups.tensor_groups.iter().enumerate() {
                if !stage_plans[0].contains_group(group) {
                    continue;
                }
                embedding_reductions.push(Self::push_repeated_collective_operation(
                    cluster,
                    placement,
                    &mut operations,
                    &mut priced,
                    format!("embedding tp all-reduce group {group_idx}"),
                    RepeatedCollective {
                        kind: CollectiveKind::AllReduce,
                        participants: group.clone(),
                        token_bytes: activation_token_bytes,
                        crossings: Crossings::Activations,
                        reduction: Some(ReductionOp::Sum),
                        algorithm: CollectiveAlgorithm::Hierarchical,
                    },
                    model,
                    request,
                    Vec::new(),
                    calibration,
                ));
            }
        }

        for layer_idx in 0..layers {
            let mut compute_dependencies = Vec::new();
            if layer_idx == 0 {
                compute_dependencies.extend(embedding_reductions.iter().copied());
            }
            let layer_stage = layer_stage(layer_idx, layers, stages);
            while current_stage < layer_stage {
                let boundary_dependency = previous_layer_barrier.or(previous_compute).unwrap_or(0);
                let boundary_idx = Self::push_pipeline_boundary_operation(
                    cluster,
                    model,
                    request,
                    placement,
                    groups,
                    current_stage,
                    boundary_dependency,
                    &mut operations,
                    &mut priced,
                    calibration,
                );
                current_stage += 1;
                if let Some(boundary_idx) = boundary_idx {
                    previous_layer_barrier = Some(boundary_idx);
                    compute_dependencies.push(boundary_idx);
                }
            }
            if let Some(previous_compute) = previous_compute {
                compute_dependencies.push(previous_compute);
            }
            if !calibration.allow_compute_comm_overlap
                && let Some(previous_layer_barrier) = previous_layer_barrier
                && !compute_dependencies.contains(&previous_layer_barrier)
            {
                compute_dependencies.push(previous_layer_barrier);
            }

            let stage_plan = &stage_plans[layer_stage as usize];
            let compute_idx = operations.len();
            operations.push(SimOperation {
                name: format!("layer {layer_idx} compute"),
                kind: SimOperationKind::Compute,
                duration_s: compute_per_layer_s,
                resources: stage_plan.compute_resources.clone(),
                dependencies: compute_dependencies,
            });
            previous_compute = Some(compute_idx);

            let mut layer_tail = compute_idx;
            if config.tensor_ranks > 1 {
                let activation_token_bytes = Self::activation_token_bytes(model, request);
                for (group_idx, group) in groups.tensor_groups.iter().enumerate() {
                    if !stage_plan.contains_group(group) {
                        continue;
                    }
                    let mut dependency = compute_idx;
                    for block in ["attn", "mlp"] {
                        dependency = Self::push_repeated_collective_operation(
                            cluster,
                            placement,
                            &mut operations,
                            &mut priced,
                            format!("layer {layer_idx} tp all-reduce {block} group {group_idx}"),
                            RepeatedCollective {
                                kind: CollectiveKind::AllReduce,
                                participants: group.clone(),
                                token_bytes: activation_token_bytes,
                                crossings: Crossings::Activations,
                                reduction: Some(ReductionOp::Sum),
                                algorithm: CollectiveAlgorithm::Hierarchical,
                            },
                            model,
                            request,
                            vec![dependency],
                            calibration,
                        );
                    }
                    layer_tail = dependency;
                }
            }

            if let Some(experts) = model.experts
                && config.expert_ranks > 1
            {
                let expert_token_bytes = Self::activation_token_bytes(model, request)
                    * u64::from(experts.top_k)
                    / u64::from(config.expert_ranks.max(1));
                for (group_idx, group) in groups.expert_groups.iter().enumerate() {
                    if !stage_plan.contains_group(group) {
                        continue;
                    }
                    layer_tail = Self::push_repeated_collective_operation(
                        cluster,
                        placement,
                        &mut operations,
                        &mut priced,
                        format!("layer {layer_idx} ep all-to-all group {group_idx}"),
                        RepeatedCollective {
                            kind: CollectiveKind::AllToAll,
                            participants: group.clone(),
                            token_bytes: expert_token_bytes,
                            crossings: Crossings::Activations,
                            reduction: None,
                            algorithm: CollectiveAlgorithm::Hierarchical,
                        },
                        model,
                        request,
                        vec![layer_tail],
                        calibration,
                    );
                }
            }

            previous_layer_barrier = Some(layer_tail);
        }

        // The LM head is vocab-parallel too: every tensor rank contributes
        // its logits shard for each sampled position.
        if config.tensor_ranks > 1 {
            let last_stage = &stage_plans[stage_plans.len() - 1];
            let shard_bytes = Self::logits_shard_bytes(model, request, config.tensor_ranks);
            let dependency = previous_layer_barrier.or(previous_compute).unwrap_or(0);
            for (group_idx, group) in groups.tensor_groups.iter().enumerate() {
                if !last_stage.contains_group(group) {
                    continue;
                }
                Self::push_repeated_collective_operation(
                    cluster,
                    placement,
                    &mut operations,
                    &mut priced,
                    format!("lm_head logits all-gather group {group_idx}"),
                    RepeatedCollective {
                        kind: CollectiveKind::AllGather,
                        participants: group.clone(),
                        token_bytes: shard_bytes,
                        crossings: Crossings::SampledLogits,
                        reduction: None,
                        algorithm: CollectiveAlgorithm::Ring,
                    },
                    model,
                    request,
                    vec![dependency],
                    calibration,
                );
            }
        }

        (operations, priced)
    }

    // Activations cross a collective or stage boundary once for the prompt
    // and once per generated token, so a collective's latency term repeats per
    // crossing instead of being paid once for the combined volume.
    fn activation_crossings(request: &InferenceRequest, token_bytes: u64) -> Vec<(u64, f64)> {
        let prompt = (token_bytes * u64::from(request.prompt_tokens), 1.0);
        let decode = (token_bytes, f64::from(request.decode_tokens));
        match request.phase {
            InferencePhase::Prefill => vec![prompt],
            InferencePhase::Decode => vec![decode],
            InferencePhase::EndToEnd => vec![prompt, decode],
        }
    }

    // Logits are produced for one position per sequence per forward pass:
    // once for the prompt and once per generated token.
    fn sampled_logit_crossings(request: &InferenceRequest, batch_bytes: u64) -> Vec<(u64, f64)> {
        let prompt = (batch_bytes, 1.0);
        let decode = (batch_bytes, f64::from(request.decode_tokens));
        match request.phase {
            InferencePhase::Prefill => vec![prompt],
            InferencePhase::Decode => vec![decode],
            InferencePhase::EndToEnd => vec![prompt, decode],
        }
    }

    // Each tensor rank holds a vocab shard of the LM-head logits for every
    // sequence; the all-gather contributes that shard.
    fn logits_shard_bytes(model: &ModelSpec, request: &InferenceRequest, tensor_ranks: u32) -> u64 {
        u64::from(request.batch_size)
            * div_ceil(u64::from(model.vocab_size), u64::from(tensor_ranks.max(1)))
            * model.dtype.bytes_per_element()
    }

    fn activation_token_bytes(model: &ModelSpec, request: &InferenceRequest) -> u64 {
        u64::from(request.batch_size)
            * u64::from(model.hidden_size)
            * model.dtype.bytes_per_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn push_repeated_collective_operation(
        cluster: &Cluster,
        placement: &RankPlacement,
        operations: &mut Vec<SimOperation>,
        priced: &mut Vec<PricedCollective>,
        name: String,
        collective: RepeatedCollective,
        model: &ModelSpec,
        request: &InferenceRequest,
        dependencies: Vec<usize>,
        calibration: SimulationCalibration,
    ) -> usize {
        let mut duration_s = 0.0;
        let mut resources = Vec::new();
        let participant_nodes: Vec<NodeId> = collective
            .participants
            .iter()
            .filter_map(|rank| placement.gpu_for_rank(*rank))
            .map(|addr| addr.node_id)
            .collect();
        let crossings = match collective.crossings {
            Crossings::Activations => Self::activation_crossings(request, collective.token_bytes),
            Crossings::SampledLogits => {
                Self::sampled_logit_crossings(request, collective.token_bytes)
            }
        };
        for (bytes, count) in crossings {
            if count <= 0.0 || bytes == 0 {
                continue;
            }
            let call = CollectiveCall {
                kind: collective.kind,
                participants: collective.participants.clone(),
                bytes_per_rank: Bytes::from_bytes(bytes),
                dtype: model.dtype,
                reduction: collective.reduction,
                root: None,
                phase: request.phase,
                algorithm: collective.algorithm,
            };
            let cost =
                Self::estimate_collective_with_calibration(cluster, placement, &call, calibration);
            duration_s += cost.total_s * count;
            resources.extend(cost.bottlenecks);
            priced.push(PricedCollective::new(
                collective.kind,
                &participant_nodes,
                cost.pricing,
            ));
        }
        resources.sort();
        resources.dedup();
        if resources.is_empty() {
            resources.push("communication".to_string());
        }

        let op_idx = operations.len();
        operations.push(SimOperation {
            name,
            kind: SimOperationKind::Collective,
            duration_s,
            resources,
            dependencies,
        });
        op_idx
    }

    #[allow(clippy::too_many_arguments)]
    fn push_pipeline_boundary_operation(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        placement: &RankPlacement,
        groups: &ParallelGroups,
        edge_idx: u32,
        dependency: usize,
        operations: &mut Vec<SimOperation>,
        priced: &mut Vec<PricedCollective>,
        calibration: SimulationCalibration,
    ) -> Option<usize> {
        let left_rank = *groups.pipeline_stages.get(edge_idx as usize)?.first()?;
        let right_rank = *groups.pipeline_stages.get(edge_idx as usize + 1)?.first()?;
        Some(Self::push_repeated_collective_operation(
            cluster,
            placement,
            operations,
            priced,
            format!("pipeline sendrecv edge {edge_idx}"),
            RepeatedCollective {
                kind: CollectiveKind::SendRecv,
                participants: vec![left_rank, right_rank],
                token_bytes: Self::activation_token_bytes(model, request),
                crossings: Crossings::Activations,
                reduction: None,
                algorithm: CollectiveAlgorithm::Auto,
            },
            model,
            request,
            vec![dependency],
            calibration,
        ))
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

// Layers split into contiguous, balanced stages in pipeline order.
pub(super) fn layer_stage(layer_idx: u32, layers: u32, stages: u32) -> u32 {
    (u64::from(layer_idx) * u64::from(stages) / u64::from(layers.max(1))) as u32
}

struct PipelineStagePlan {
    ranks: BTreeSet<RankId>,
    compute_resources: Vec<String>,
}

impl PipelineStagePlan {
    fn new(
        placement: &RankPlacement,
        groups: &ParallelGroups,
        stage: u32,
        phase: InferencePhase,
    ) -> Self {
        let ranks: BTreeSet<RankId> = match groups.pipeline_stages.get(stage as usize) {
            Some(stage_ranks) if !stage_ranks.is_empty() => stage_ranks.iter().copied().collect(),
            _ => (0..placement.rank_to_gpu.len() as RankId).collect(),
        };
        let gpus = ranks
            .iter()
            .filter_map(|rank| placement.gpu_for_rank(*rank))
            .collect::<Vec<_>>();
        Self {
            compute_resources: compute_resources(&gpus, phase),
            ranks,
        }
    }

    fn contains_group(&self, group: &[RankId]) -> bool {
        group.first().is_some_and(|rank| self.ranks.contains(rank))
    }
}

fn compute_resources(gpus: &[GpuAddr], phase: InferencePhase) -> Vec<String> {
    let mut resources: Vec<_> = gpus
        .iter()
        .map(|addr| format!("gpu compute node {}", addr.node_id))
        .collect();
    if matches!(phase, InferencePhase::Decode | InferencePhase::EndToEnd) {
        resources.extend(
            gpus.iter()
                .map(|addr| format!("gpu HBM node {}", addr.node_id)),
        );
    }
    resources.sort();
    resources.dedup();
    resources
}

// A collective issued once per activation crossing; the per-crossing message
// is `token_bytes` times the tokens crossing at once.
struct RepeatedCollective {
    kind: CollectiveKind,
    participants: Vec<RankId>,
    token_bytes: u64,
    crossings: Crossings,
    reduction: Option<ReductionOp>,
    algorithm: CollectiveAlgorithm,
}

/// How often a repeated collective is issued and how big each issue is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Crossings {
    /// Once for the prompt (`token_bytes` x prompt tokens) and once per
    /// generated token (`token_bytes`).
    Activations,
    /// Once per forward pass for the sampled position of every sequence:
    /// `token_bytes` already covers the batch, issued once for the prompt
    /// pass and once per generated token.
    SampledLogits,
}
