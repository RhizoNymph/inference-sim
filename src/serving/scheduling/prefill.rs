use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct PrefillTokenSpan {
    pub(in crate::serving) start_s: f64,
    pub(in crate::serving) finish_s: f64,
    pub(in crate::serving) tokens: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct PrefillAdmissionSpan {
    pub(in crate::serving) finish_s: f64,
    pub(in crate::serving) traffic_class: Option<String>,
    pub(in crate::serving) tokens: u64,
    pub(in crate::serving) node_tokens: Vec<(NodeId, u64)>,
    pub(in crate::serving) gpu_tokens: Vec<(GpuAddr, u64)>,
}

impl PrefillAdmissionSpan {
    pub(in crate::serving) fn for_state_chunk(
        state: &DecodeRequestState,
        chunk_tokens: Option<u32>,
        finish_s: f64,
    ) -> Self {
        let tokens = prefill_work_tokens(state, chunk_tokens);
        Self::for_state_tokens(state, tokens, finish_s)
    }

    pub(in crate::serving) fn for_state_tokens(
        state: &DecodeRequestState,
        tokens: u64,
        finish_s: f64,
    ) -> Self {
        Self {
            finish_s,
            traffic_class: state.traffic_class.clone(),
            tokens,
            node_tokens: prefill_owner_nodes(state)
                .into_iter()
                .map(|node_id| (node_id, tokens))
                .collect(),
            gpu_tokens: prefill_owner_gpus(state)
                .into_iter()
                .map(|gpu| (gpu, tokens))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct DecodeRequestState {
    pub(in crate::serving) request_idx: u32,
    pub(in crate::serving) request_id: Option<String>,
    pub(in crate::serving) tenant: Option<String>,
    pub(in crate::serving) model_id: Option<String>,
    pub(in crate::serving) traffic_class: Option<String>,
    pub(in crate::serving) shape_profile: Option<String>,
    pub(in crate::serving) cache_key: Option<String>,
    pub(in crate::serving) prefill_node: NodeId,
    pub(in crate::serving) decode_node: NodeId,
    pub(in crate::serving) prefill_route_nodes: Vec<NodeId>,
    pub(in crate::serving) prefill_route_gpus: Vec<GpuAddr>,
    pub(in crate::serving) decode_route_nodes: Vec<NodeId>,
    pub(in crate::serving) decode_route_gpus: Vec<GpuAddr>,
    pub(in crate::serving) routing_policy: ServingRoutingPolicy,
    pub(in crate::serving) routing_candidate_count: u32,
    pub(in crate::serving) routing_routable_candidate_count: u32,
    pub(in crate::serving) routing_estimated_e2el_s: f64,
    pub(in crate::serving) routing_estimated_kv_transfer_s: f64,
    pub(in crate::serving) routing_estimated_kv_resource_wait_s: f64,
    pub(in crate::serving) routing_estimated_prefill_wait_s: f64,
    pub(in crate::serving) routing_estimated_decode_wait_s: f64,
    pub(in crate::serving) routing_reason: String,
    pub(in crate::serving) routing_candidates: Vec<ServingRouteCandidateObservation>,
    pub(in crate::serving) arrival_s: f64,
    pub(in crate::serving) priority: i32,
    pub(in crate::serving) batch_size: u32,
    pub(in crate::serving) prompt_tokens: u32,
    pub(in crate::serving) prefix_cache_hit_tokens: u32,
    pub(in crate::serving) effective_prefill_tokens: u32,
    pub(in crate::serving) remaining_prefill_tokens: u32,
    pub(in crate::serving) prefill_chunks: u32,
    pub(in crate::serving) decode_tokens: u32,
    pub(in crate::serving) slo: ServingRequestSlo,
    pub(in crate::serving) max_queue_delay_s: Option<f64>,
    pub(in crate::serving) max_kv_queue_delay_s: Option<f64>,
    pub(in crate::serving) max_decode_queue_delay_s: Option<f64>,
    pub(in crate::serving) max_decode_iteration_queue_delay_s: Option<f64>,
    pub(in crate::serving) request_timeout_s: Option<f64>,
    pub(in crate::serving) deadline_s: Option<f64>,
    pub(in crate::serving) cancellation_s: Option<f64>,
    pub(in crate::serving) remaining_tokens: u32,
    pub(in crate::serving) emitted_tokens: u32,
    pub(in crate::serving) max_sequence_tokens: u32,
    pub(in crate::serving) kv_block_tokens: u32,
    pub(in crate::serving) kv_cache_blocks: u64,
    pub(in crate::serving) kv_allocated_tokens: u64,
    pub(in crate::serving) kv_fragmentation_tokens: u64,
    pub(in crate::serving) prefill_scheduled: bool,
    pub(in crate::serving) dependencies: Vec<usize>,
    pub(in crate::serving) kv_start_s: f64,
    pub(in crate::serving) kv_finish_s: f64,
    pub(in crate::serving) first_decode_start_s: Option<f64>,
    pub(in crate::serving) first_decode_finish_s: Option<f64>,
    pub(in crate::serving) last_decode_finish_s: Option<f64>,
    pub(in crate::serving) decode_token_start_s: Vec<f64>,
    pub(in crate::serving) decode_token_finish_s: Vec<f64>,
    pub(in crate::serving) prefill_start_s: f64,
    pub(in crate::serving) prefill_finish_s: f64,
    pub(in crate::serving) prefill_worker_queue_s: f64,
    pub(in crate::serving) prefill_resource_queue_s: f64,
    pub(in crate::serving) decode_worker_queue_s: f64,
    pub(in crate::serving) decode_resource_queue_s: f64,
    pub(in crate::serving) kv_transfer_bytes: u64,
    pub(in crate::serving) kv_transfer_bottlenecks: Vec<String>,
    pub(in crate::serving) kv_transfer_paths: Vec<ServingKvTransferPathObservation>,
    pub(in crate::serving) kv_transfer_resources: Vec<String>,
    pub(in crate::serving) kv_transfer_resource_dependencies: Vec<usize>,
    pub(in crate::serving) kv_worker_queue_s: f64,
    pub(in crate::serving) kv_resource_queue_s: f64,
    pub(in crate::serving) kv_transfer_s: f64,
    pub(in crate::serving) kv_transfer_fit: Option<CalibrationFitApplication>,
    pub(in crate::serving) prefill_token_spans: Vec<PrefillTokenSpan>,
    pub(in crate::serving) worker_assignments: Vec<ServingWorkerAssignmentObservation>,
    pub(in crate::serving) status: ServingRequestStatus,
    pub(in crate::serving) status_time_s: Option<f64>,
    pub(in crate::serving) failure_reason: Option<String>,
    pub(in crate::serving) failure_rejection: Option<ServingRejection>,
}

pub(in crate::serving) fn compare_request_priority(
    states: &[DecodeRequestState],
    left: usize,
    right: usize,
) -> std::cmp::Ordering {
    states[right]
        .priority
        .cmp(&states[left].priority)
        .then_with(|| states[left].arrival_s.total_cmp(&states[right].arrival_s))
        .then_with(|| states[left].request_idx.cmp(&states[right].request_idx))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::serving) fn schedule_prefills(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    prefill_score: &ScoredParallelismConfig,
    traffic: &ServingTraffic,
    prefill_resource_base_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
) {
    let prefill_worker_slots_per_gpu = prefill_worker_slots_per_gpu(traffic);
    for state in states.iter_mut() {
        if state.status == ServingRequestStatus::Pending && state.remaining_prefill_tokens == 0 {
            state.prefill_scheduled = true;
            state.prefill_start_s = state.arrival_s;
            state.prefill_finish_s = state.arrival_s;
            state.dependencies.clear();
        }
    }

    let mut active_prefills = Vec::new();

    match traffic.prefill_batching {
        ServingPrefillBatching::Independent => schedule_independent_prefills(
            scheduler,
            worker_runtime,
            states,
            &mut active_prefills,
            prefill_score,
            traffic,
            prefill_resource_base_node,
            base_batch_size,
            base_prompt_tokens,
            traffic.max_queue_delay_s,
            prefill_worker_slots_per_gpu,
        ),
        ServingPrefillBatching::Continuous {
            max_batch_tokens,
            chunk_tokens,
        } => schedule_continuous_prefills(
            scheduler,
            worker_runtime,
            states,
            &mut active_prefills,
            prefill_score,
            traffic,
            prefill_resource_base_node,
            base_batch_size,
            base_prompt_tokens,
            max_batch_tokens,
            chunk_tokens,
            traffic.max_queue_delay_s,
            prefill_worker_slots_per_gpu,
        ),
    }
}

pub(in crate::serving) fn prefill_worker_slots_per_gpu(traffic: &ServingTraffic) -> usize {
    scaled_worker_slots(
        traffic.max_prefill_worker_slots_per_gpu.unwrap_or(1),
        traffic.services.prefill,
    )
}

pub(in crate::serving) fn decode_worker_slots_per_gpu(traffic: &ServingTraffic) -> usize {
    scaled_worker_slots(
        traffic.max_decode_worker_slots_per_gpu.unwrap_or(1),
        traffic.services.decode,
    )
}

pub(in crate::serving) fn kv_transfer_worker_slots_per_gpu(
    traffic: &ServingTraffic,
) -> Option<usize> {
    traffic
        .max_kv_transfer_worker_slots_per_gpu
        .map(|slots| scaled_worker_slots(slots, traffic.services.kv_transfer))
}

pub(in crate::serving) fn scaled_worker_slots(
    configured_slots: u32,
    service: ServingServicePhaseConfig,
) -> usize {
    if !service.health.accepts_requests() {
        return 0;
    }
    let scaled = f64::from(configured_slots.max(1)) * service.worker_scale;
    if !scaled.is_finite() || scaled <= 0.0 {
        0
    } else {
        scaled.ceil().min(usize::MAX as f64) as usize
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::serving) fn schedule_independent_prefills(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    active_prefills: &mut Vec<PrefillAdmissionSpan>,
    prefill_score: &ScoredParallelismConfig,
    traffic: &ServingTraffic,
    prefill_resource_base_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    max_queue_delay_s: Option<f64>,
    prefill_worker_slots_per_gpu: usize,
) {
    let mut order = (0..states.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| compare_request_priority(states, *left, *right));
    for idx in order {
        if states[idx].prefill_scheduled || states[idx].status != ServingRequestStatus::Pending {
            continue;
        }
        let prefill_node = states[idx].prefill_node;
        let request_idx = states[idx].request_idx;
        schedule_prefill_batch(
            scheduler,
            worker_runtime,
            states,
            active_prefills,
            traffic,
            &[idx],
            prefill_score,
            prefill_resource_base_node,
            Some(prefill_node),
            base_batch_size,
            base_prompt_tokens,
            None,
            max_queue_delay_s,
            prefill_worker_slots_per_gpu,
            format!("request {request_idx} prefill"),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::serving) fn schedule_continuous_prefills(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    active_prefills: &mut Vec<PrefillAdmissionSpan>,
    prefill_score: &ScoredParallelismConfig,
    traffic: &ServingTraffic,
    prefill_resource_base_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    max_batch_tokens: Option<u64>,
    chunk_tokens: Option<u32>,
    max_queue_delay_s: Option<f64>,
    prefill_worker_slots_per_gpu: usize,
) {
    let max_batch_tokens = max_batch_tokens.unwrap_or(u64::MAX).max(1);
    let mut batch_idx = 0_u32;

    while states
        .iter()
        .any(|state| !state.prefill_scheduled && state.status == ServingRequestStatus::Pending)
    {
        let Some((ready_idx, ready_s)) = states
            .iter()
            .enumerate()
            .filter(|(_, state)| {
                !state.prefill_scheduled && state.status == ServingRequestStatus::Pending
            })
            .map(|(idx, state)| {
                (
                    idx,
                    prefill_candidate_ready_s(
                        state,
                        scheduler,
                        worker_runtime,
                        prefill_worker_slots_per_gpu,
                    ),
                )
            })
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| compare_request_priority(states, left.0, right.0))
            })
        else {
            break;
        };
        let target_prefill_node =
            prefill_resource_base_node.map(|_| states[ready_idx].prefill_node);
        let mut selected = Vec::new();
        let mut selected_tokens = 0_u64;

        let mut candidates = states
            .iter()
            .enumerate()
            .filter_map(|(idx, state)| {
                if state.prefill_scheduled || state.status != ServingRequestStatus::Pending {
                    return None;
                }
                if let Some(target_prefill_node) = target_prefill_node
                    && state.prefill_node != target_prefill_node
                {
                    return None;
                }
                if prefill_candidate_ready_s(
                    state,
                    scheduler,
                    worker_runtime,
                    prefill_worker_slots_per_gpu,
                ) > ready_s + 1e-12
                {
                    return None;
                }
                Some(idx)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| compare_request_priority(states, *left, *right));

        for idx in candidates {
            let state = &states[idx];
            let request_tokens = prefill_work_tokens(state, chunk_tokens);
            if selected.is_empty()
                || selected_tokens.saturating_add(request_tokens) <= max_batch_tokens
            {
                selected.push(idx);
                selected_tokens = selected_tokens.saturating_add(request_tokens);
            }
        }

        let label = if chunk_tokens.is_some() {
            format!("prefill batch {batch_idx} chunk tokens {selected_tokens}")
        } else {
            format!("prefill batch {batch_idx} tokens {selected_tokens}")
        };
        schedule_prefill_batch(
            scheduler,
            worker_runtime,
            states,
            active_prefills,
            traffic,
            &selected,
            prefill_score,
            prefill_resource_base_node,
            target_prefill_node,
            base_batch_size,
            base_prompt_tokens,
            chunk_tokens,
            max_queue_delay_s,
            prefill_worker_slots_per_gpu,
            label,
        );
        batch_idx += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::serving) fn schedule_prefill_batch(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    active_prefills: &mut Vec<PrefillAdmissionSpan>,
    traffic: &ServingTraffic,
    selected: &[usize],
    prefill_score: &ScoredParallelismConfig,
    prefill_resource_base_node: Option<NodeId>,
    target_prefill_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    chunk_tokens: Option<u32>,
    max_queue_delay_s: Option<f64>,
    prefill_worker_slots_per_gpu: usize,
    label: String,
) {
    if selected.is_empty() {
        return;
    }

    let total_tokens = selected
        .iter()
        .map(|idx| prefill_work_tokens(&states[*idx], chunk_tokens))
        .sum::<u64>();
    let base_tokens = u64::from(base_batch_size.max(1)) * u64::from(base_prompt_tokens.max(1));
    let scale = total_tokens as f64 / base_tokens.max(1) as f64;
    let earliest_start_s = selected
        .iter()
        .map(|idx| {
            prefill_candidate_ready_s(
                &states[*idx],
                scheduler,
                worker_runtime,
                prefill_worker_slots_per_gpu,
            )
        })
        .fold(0.0, f64::max);
    let dependencies = selected
        .iter()
        .flat_map(|idx| states[*idx].dependencies.iter().copied())
        .collect::<Vec<_>>();
    let mut prefill_ops = prefill_score.operations.clone();
    scale_operations(&mut prefill_ops, scale);
    if let Some(target_prefill_node) = target_prefill_node {
        remap_operations_to_node(
            &mut prefill_ops,
            prefill_resource_base_node,
            target_prefill_node,
        );
    }
    let has_cancellable_request = selected
        .iter()
        .any(|idx| states[*idx].cancellation_s.is_some());
    let has_queue_limited_request = selected.iter().any(|idx| {
        states[*idx]
            .max_queue_delay_s
            .or(max_queue_delay_s)
            .is_some()
    });
    let has_prefill_capacity_admission = request_level_prefill_capacity_enabled(traffic);
    if has_queue_limited_request || has_cancellable_request || has_prefill_capacity_admission {
        let mut candidate_scheduler = scheduler.clone();
        let candidate_ids = schedule_trace(
            &mut candidate_scheduler,
            &label,
            &prefill_ops,
            earliest_start_s,
            &dependencies,
        );
        let (candidate_start_s, candidate_finish_s) =
            operation_span(&candidate_scheduler, &candidate_ids);
        if has_prefill_capacity_admission {
            retire_prefill_admissions(active_prefills, candidate_start_s);
        }
        let mut candidate_active_prefills = active_prefills.clone();
        let mut admitted = Vec::new();
        let mut rejected = Vec::new();
        let mut capacity_rejected = Vec::new();
        let mut cancelled = Vec::new();
        for idx in selected {
            if let Some(cancellation_s) = states[*idx].cancellation_s
                && cancellation_s <= candidate_start_s + 1e-12
            {
                cancelled.push((*idx, cancellation_s));
                continue;
            }
            if let Some(effective_max_queue_delay_s) =
                states[*idx].max_queue_delay_s.or(max_queue_delay_s)
                && states[*idx].prefill_chunks == 0
            {
                let queue_delay_s = (candidate_start_s - states[*idx].arrival_s).max(0.0);
                if queue_delay_s > effective_max_queue_delay_s + 1e-12 {
                    rejected.push((*idx, queue_delay_s, effective_max_queue_delay_s));
                    continue;
                }
            }
            if has_prefill_capacity_admission {
                let candidate = PrefillAdmissionSpan::for_state_chunk(
                    &states[*idx],
                    chunk_tokens,
                    candidate_finish_s,
                );
                if states[*idx].prefill_chunks == 0
                    && let Some(reason) = prefill_capacity_admission_rejection(
                        &candidate_active_prefills,
                        &candidate,
                        traffic,
                    )
                {
                    capacity_rejected.push((*idx, reason, candidate_start_s));
                    continue;
                }
                candidate_active_prefills.push(candidate);
            }
            admitted.push(*idx);
        }

        if !cancelled.is_empty() || !rejected.is_empty() || !capacity_rejected.is_empty() {
            for (idx, cancellation_s) in cancelled {
                cancel_request(
                    &mut states[idx],
                    cancellation_s,
                    "request cancelled before prefill admission",
                );
            }
            for (idx, queue_delay_s, effective_max_queue_delay_s) in rejected {
                reject_admission(&mut states[idx], queue_delay_s, effective_max_queue_delay_s);
            }
            for (idx, reason, admission_s) in capacity_rejected {
                reject_prefill_capacity_admission(&mut states[idx], admission_s, reason);
            }
            if admitted.is_empty() {
                return;
            }
            schedule_prefill_batch(
                scheduler,
                worker_runtime,
                states,
                active_prefills,
                traffic,
                &admitted,
                prefill_score,
                prefill_resource_base_node,
                target_prefill_node,
                base_batch_size,
                base_prompt_tokens,
                chunk_tokens,
                max_queue_delay_s,
                prefill_worker_slots_per_gpu,
                label,
            );
            return;
        }
    }
    let prefill_ids = schedule_trace(
        scheduler,
        &label,
        &prefill_ops,
        earliest_start_s,
        &dependencies,
    );
    let (prefill_start_s, prefill_finish_s) = operation_span(scheduler, &prefill_ids);
    let prefill_deps = terminal_ids(&prefill_ops, &prefill_ids);

    for idx in selected {
        record_prefill_queue_breakdown(
            &mut states[*idx],
            scheduler,
            worker_runtime,
            prefill_worker_slots_per_gpu,
            prefill_start_s,
        );
    }
    let prefill_worker_assignments = assign_selected_prefill_workers_ready(
        worker_runtime,
        states,
        selected,
        prefill_worker_slots_per_gpu,
        prefill_start_s,
        prefill_finish_s,
        &prefill_ids,
    );

    for idx in selected {
        let state = &mut states[*idx];
        state.worker_assignments.extend(assignments_for_worker_gpus(
            &prefill_worker_assignments,
            &state.prefill_route_gpus,
            state.prefill_node,
        ));
        if state.prefill_chunks == 0 {
            state.prefill_start_s = prefill_start_s;
        }
        state.prefill_finish_s = prefill_finish_s;
        state.dependencies = prefill_deps.clone();
        let chunk_tokens = state_prefill_chunk_tokens(state, chunk_tokens);
        let prefill_tokens =
            u64::from(state.batch_size.max(1)).saturating_mul(u64::from(chunk_tokens));
        if prefill_tokens > 0 && prefill_finish_s > prefill_start_s {
            state.prefill_token_spans.push(PrefillTokenSpan {
                start_s: prefill_start_s,
                finish_s: prefill_finish_s,
                tokens: prefill_tokens,
            });
            if has_prefill_capacity_admission {
                active_prefills.push(PrefillAdmissionSpan::for_state_tokens(
                    state,
                    prefill_tokens,
                    prefill_finish_s,
                ));
            }
        }
        state.remaining_prefill_tokens =
            state.remaining_prefill_tokens.saturating_sub(chunk_tokens);
        state.prefill_chunks = state.prefill_chunks.saturating_add(1);
        state.prefill_scheduled = state.remaining_prefill_tokens == 0;
    }
}

pub(in crate::serving) fn prefill_ready_s(
    state: &DecodeRequestState,
    scheduler: &ResourceScheduler,
) -> f64 {
    if state.prefill_chunks == 0 {
        state.arrival_s
    } else {
        operation_finish_s(scheduler, &state.dependencies)
    }
}

pub(in crate::serving) fn prefill_candidate_ready_s(
    state: &DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    prefill_worker_slots_per_gpu: usize,
) -> f64 {
    prefill_ready_s(state, scheduler).max(prefill_worker_ready_s(
        state,
        worker_runtime,
        prefill_worker_slots_per_gpu,
    ))
}

pub(in crate::serving) fn prefill_worker_ready_s(
    state: &DecodeRequestState,
    worker_runtime: &ServingWorkerRuntime,
    prefill_worker_slots_per_gpu: usize,
) -> f64 {
    route_worker_gpus(&state.prefill_route_gpus, state.prefill_node)
        .iter()
        .map(|gpu| {
            prefill_worker_gpu_ready_s(
                &worker_runtime.prefill_ready_s,
                *gpu,
                prefill_worker_slots_per_gpu,
            )
        })
        .fold(0.0, f64::max)
}

pub(in crate::serving) fn prefill_worker_gpu_ready_s(
    ready_s: &BTreeMap<GpuAddr, Vec<f64>>,
    gpu: GpuAddr,
    prefill_worker_slots_per_gpu: usize,
) -> f64 {
    worker_slot_ready_s(ready_s, gpu, prefill_worker_slots_per_gpu)
}

pub(in crate::serving) fn decode_candidate_ready_s(
    state: &DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    decode_worker_slots_per_gpu: usize,
) -> f64 {
    operation_finish_s(scheduler, &state.dependencies).max(decode_worker_ready_s(
        state,
        worker_runtime,
        decode_worker_slots_per_gpu,
    ))
}

pub(in crate::serving) fn decode_worker_ready_s(
    state: &DecodeRequestState,
    worker_runtime: &ServingWorkerRuntime,
    decode_worker_slots_per_gpu: usize,
) -> f64 {
    route_workers_ready_s(
        &worker_runtime.decode_ready_s,
        &state.decode_route_gpus,
        state.decode_node,
        decode_worker_slots_per_gpu,
    )
}

pub(in crate::serving) fn kv_transfer_worker_ready_s(
    state: &DecodeRequestState,
    worker_runtime: &ServingWorkerRuntime,
    kv_transfer_worker_slots_per_gpu: usize,
) -> f64 {
    kv_transfer_worker_gpus(state)
        .iter()
        .map(|gpu| {
            worker_slot_ready_s(
                &worker_runtime.kv_transfer_ready_s,
                *gpu,
                kv_transfer_worker_slots_per_gpu,
            )
        })
        .fold(0.0, f64::max)
}

pub(in crate::serving) fn record_prefill_queue_breakdown(
    state: &mut DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    prefill_worker_slots_per_gpu: usize,
    prefill_start_s: f64,
) {
    let phase_ready_s = prefill_ready_s(state, scheduler);
    let worker_ready_s =
        prefill_worker_ready_s(state, worker_runtime, prefill_worker_slots_per_gpu);
    let candidate_ready_s = phase_ready_s.max(worker_ready_s);
    state.prefill_worker_queue_s += finite_or_zero(worker_ready_s - phase_ready_s);
    state.prefill_resource_queue_s += finite_or_zero(prefill_start_s - candidate_ready_s);
}

pub(in crate::serving) fn record_decode_queue_breakdown(
    state: &mut DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    decode_worker_slots_per_gpu: usize,
    decode_start_s: f64,
) {
    let phase_ready_s = operation_finish_s(scheduler, &state.dependencies);
    let worker_ready_s = decode_worker_ready_s(state, worker_runtime, decode_worker_slots_per_gpu);
    let candidate_ready_s = phase_ready_s.max(worker_ready_s);
    state.decode_worker_queue_s += finite_or_zero(worker_ready_s - phase_ready_s);
    state.decode_resource_queue_s += finite_or_zero(decode_start_s - candidate_ready_s);
}

pub(in crate::serving) fn assign_selected_prefill_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    states: &[DecodeRequestState],
    selected: &[usize],
    prefill_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<ServingWorkerAssignmentObservation> {
    let mut route_gpus = BTreeSet::new();
    for idx in selected {
        route_gpus.extend(route_worker_gpus(
            &states[*idx].prefill_route_gpus,
            states[*idx].prefill_node,
        ));
    }
    let route_gpus = route_gpus.into_iter().collect::<Vec<_>>();
    assign_worker_slots(
        &mut worker_runtime.prefill_ready_s,
        &route_gpus,
        0,
        prefill_worker_slots_per_gpu,
        WorkerAssignmentSpan {
            phase: "prefill",
            start_s,
            finish_s,
            operation_ids,
        },
    )
}

pub(in crate::serving) fn assign_decode_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    state: &DecodeRequestState,
    decode_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<ServingWorkerAssignmentObservation> {
    assign_worker_slots(
        &mut worker_runtime.decode_ready_s,
        &state.decode_route_gpus,
        state.decode_node,
        decode_worker_slots_per_gpu,
        WorkerAssignmentSpan {
            phase: "decode",
            start_s,
            finish_s,
            operation_ids,
        },
    )
}

pub(in crate::serving) fn assign_selected_decode_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    states: &[DecodeRequestState],
    selected: &[usize],
    decode_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<(usize, Vec<ServingWorkerAssignmentObservation>)> {
    selected
        .iter()
        .map(|idx| {
            (
                *idx,
                assign_decode_workers_ready(
                    worker_runtime,
                    &states[*idx],
                    decode_worker_slots_per_gpu,
                    start_s,
                    finish_s,
                    operation_ids,
                ),
            )
        })
        .collect()
}

pub(in crate::serving) fn assign_kv_transfer_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    state: &DecodeRequestState,
    kv_transfer_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<ServingWorkerAssignmentObservation> {
    let gpus = kv_transfer_worker_gpus(state);
    assign_worker_slots(
        &mut worker_runtime.kv_transfer_ready_s,
        &gpus,
        state.decode_node,
        kv_transfer_worker_slots_per_gpu,
        WorkerAssignmentSpan {
            phase: "kv_transfer",
            start_s,
            finish_s,
            operation_ids,
        },
    )
}

pub(in crate::serving) fn kv_transfer_worker_gpus(state: &DecodeRequestState) -> Vec<GpuAddr> {
    let mut gpus = BTreeSet::new();
    gpus.extend(route_worker_gpus(
        &state.prefill_route_gpus,
        state.prefill_node,
    ));
    gpus.extend(route_worker_gpus(
        &state.decode_route_gpus,
        state.decode_node,
    ));
    gpus.into_iter().collect()
}

pub(in crate::serving) fn prefill_work_tokens(
    state: &DecodeRequestState,
    chunk_tokens: Option<u32>,
) -> u64 {
    u64::from(state.batch_size.max(1)) * u64::from(state_prefill_chunk_tokens(state, chunk_tokens))
}

pub(in crate::serving) fn state_prefill_chunk_tokens(
    state: &DecodeRequestState,
    chunk_tokens: Option<u32>,
) -> u32 {
    chunk_tokens
        .map(|chunk_tokens| state.remaining_prefill_tokens.min(chunk_tokens.max(1)))
        .unwrap_or(state.remaining_prefill_tokens)
}

pub(in crate::serving) fn request_level_prefill_capacity_enabled(traffic: &ServingTraffic) -> bool {
    traffic.decode_capacity_policy == ServingDecodeCapacityPolicy::RequestReject
        && (traffic.max_prefill_tokens.is_some()
            || traffic.max_prefill_tokens_per_node.is_some()
            || traffic.max_prefill_tokens_per_gpu.is_some()
            || traffic
                .traffic_classes
                .iter()
                .any(|class| class.max_prefill_tokens.is_some()))
}

pub(in crate::serving) fn retire_prefill_admissions(
    active: &mut Vec<PrefillAdmissionSpan>,
    now_s: f64,
) {
    active.retain(|span| span.finish_s > now_s + 1e-12);
}

pub(in crate::serving) fn prefill_capacity_admission_rejection(
    active: &[PrefillAdmissionSpan],
    candidate: &PrefillAdmissionSpan,
    traffic: &ServingTraffic,
) -> Option<ServingRejection> {
    if let Some(limit) = traffic.max_prefill_tokens {
        let observed = active
            .iter()
            .map(|span| span.tokens)
            .sum::<u64>()
            .saturating_add(candidate.tokens);
        if observed > limit {
            let message = format!(
                "prefill admission rejected: active prefill tokens {observed} > max_prefill_tokens {limit}"
            );
            return Some(with_rejection_details(
                request_rejection(
                    "prefill",
                    "capacity",
                    "prefill_tokens",
                    "prefill_active_tokens_exceeded",
                    message,
                ),
                Some(observed as f64),
                Some(limit as f64),
                Some("tokens"),
                Some("increase max_prefill_tokens, add prefill capacity, or reduce prompt load"),
            ));
        }
    }

    if let Some(class) = serving_traffic_class(traffic, candidate.traffic_class.as_deref())
        && let Some(limit) = class.max_prefill_tokens
    {
        let observed =
            active_prefill_class_tokens(active, &class.name).saturating_add(candidate.tokens);
        if observed > limit {
            let message = format!(
                "prefill admission rejected: traffic class {} active prefill tokens {} > max_prefill_tokens {}",
                class.name, observed, limit
            );
            return Some(with_rejection_details(
                request_rejection(
                    "prefill",
                    "capacity",
                    &format!("traffic_class {} prefill_tokens", class.name),
                    "traffic_class_prefill_active_tokens_exceeded",
                    message,
                ),
                Some(observed as f64),
                Some(limit as f64),
                Some("tokens"),
                Some(
                    "increase the traffic class max_prefill_tokens, add prefill capacity, or reduce class prompt concurrency",
                ),
            ));
        }
    }

    if let Some(limit) = traffic.max_prefill_tokens_per_node {
        for (node_id, tokens) in &candidate.node_tokens {
            let observed = active_prefill_node_tokens(active, *node_id).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "prefill admission rejected: node {node_id} active prefill tokens {observed} > max_prefill_tokens_per_node {limit}"
                );
                return Some(with_rejection_details(
                    request_rejection(
                        "prefill",
                        "capacity",
                        &format!("node {node_id} prefill_tokens"),
                        "prefill_active_tokens_per_node_exceeded",
                        message,
                    ),
                    Some(observed as f64),
                    Some(limit as f64),
                    Some("tokens"),
                    Some(
                        "increase max_prefill_tokens_per_node, add prefill nodes, or rebalance routing",
                    ),
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_prefill_tokens_per_gpu {
        for (gpu, tokens) in &candidate.gpu_tokens {
            let observed = active_prefill_gpu_tokens(active, *gpu).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "prefill admission rejected: node {} gpu {} active prefill tokens {} > max_prefill_tokens_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(with_rejection_details(
                    request_rejection(
                        "prefill",
                        "capacity",
                        &format!(
                            "node {} gpu {} prefill_tokens",
                            gpu.node_id, gpu.local_gpu_id
                        ),
                        "prefill_active_tokens_per_gpu_exceeded",
                        message,
                    ),
                    Some(observed as f64),
                    Some(limit as f64),
                    Some("tokens"),
                    Some(
                        "increase max_prefill_tokens_per_gpu, add prefill GPUs, or reduce chunk size",
                    ),
                ));
            }
        }
    }

    None
}

pub(in crate::serving) fn active_prefill_class_tokens(
    active: &[PrefillAdmissionSpan],
    class_name: &str,
) -> u64 {
    active
        .iter()
        .filter(|span| span.traffic_class.as_deref() == Some(class_name))
        .map(|span| span.tokens)
        .sum()
}

pub(in crate::serving) fn active_prefill_node_tokens(
    active: &[PrefillAdmissionSpan],
    node_id: NodeId,
) -> u64 {
    active
        .iter()
        .flat_map(|span| &span.node_tokens)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, tokens)| *tokens)
        .sum()
}

pub(in crate::serving) fn active_prefill_gpu_tokens(
    active: &[PrefillAdmissionSpan],
    gpu: GpuAddr,
) -> u64 {
    active
        .iter()
        .flat_map(|span| &span.gpu_tokens)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, tokens)| *tokens)
        .sum()
}
