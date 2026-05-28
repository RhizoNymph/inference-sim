use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestObservation {
    pub request_idx: u32,
    pub request_id: Option<String>,
    pub tenant: Option<String>,
    pub model_id: Option<String>,
    pub traffic_class: Option<String>,
    pub shape_profile: Option<String>,
    pub cache_key: Option<String>,
    pub status: ServingRequestStatus,
    pub status_time_s: Option<f64>,
    pub failure_reason: Option<String>,
    pub rejection: Option<ServingRejection>,
    pub priority: i32,
    pub slo: ServingRequestSlo,
    pub ttft_slo_missed: bool,
    pub tpot_slo_missed: bool,
    pub itl_slo_missed: bool,
    pub e2el_slo_missed: bool,
    pub deadline_s: Option<f64>,
    pub deadline_missed: bool,
    pub cancellation_s: Option<f64>,
    pub prefill_node: NodeId,
    pub decode_node: NodeId,
    pub prefill_route_nodes: Vec<NodeId>,
    pub prefill_route_gpus: Vec<GpuAddr>,
    pub decode_route_nodes: Vec<NodeId>,
    pub decode_route_gpus: Vec<GpuAddr>,
    pub kv_cache_owner_gpus: Vec<GpuAddr>,
    pub routing_policy: ServingRoutingPolicy,
    pub routing_candidate_count: u32,
    pub routing_routable_candidate_count: u32,
    pub routing_estimated_e2el_s: f64,
    pub routing_estimated_kv_transfer_s: f64,
    pub routing_estimated_kv_resource_wait_s: f64,
    pub routing_estimated_prefill_wait_s: f64,
    pub routing_estimated_decode_wait_s: f64,
    pub routing_reason: String,
    pub routing_candidates: Vec<ServingRouteCandidateObservation>,
    pub arrival_s: f64,
    pub batch_size: u32,
    pub prompt_tokens: u32,
    pub prefix_cache_hit_tokens: u32,
    pub effective_prefill_tokens: u32,
    pub prefill_chunks: u32,
    pub prefill_token_start_s: Vec<f64>,
    pub prefill_token_finish_s: Vec<f64>,
    pub decode_tokens: u32,
    pub kv_block_tokens: u32,
    pub kv_cache_blocks: u64,
    pub kv_allocated_tokens: u64,
    pub kv_fragmentation_tokens: u64,
    pub kv_block_ownership: Vec<ServingKvBlockOwnershipObservation>,
    pub prefill_start_s: f64,
    pub prefill_finish_s: f64,
    pub kv_start_s: f64,
    pub kv_finish_s: f64,
    pub first_decode_start_s: f64,
    pub first_decode_finish_s: f64,
    pub last_decode_finish_s: f64,
    pub decode_iterations: u32,
    pub decode_token_start_s: Vec<f64>,
    pub decode_token_finish_s: Vec<f64>,
    pub inter_token_latency_s: Vec<f64>,
    pub ttft_s: f64,
    pub tpot_s: f64,
    pub itl_s: f64,
    pub e2el_s: f64,
    pub service_s: f64,
    pub queue_delay_s: f64,
    pub prefill_s: f64,
    pub prefill_worker_queue_s: f64,
    pub prefill_resource_queue_s: f64,
    pub kv_queue_s: f64,
    pub kv_worker_queue_s: f64,
    pub kv_resource_queue_s: f64,
    pub kv_transfer_bytes: u64,
    pub kv_transfer_bottlenecks: Vec<String>,
    pub kv_transfer_paths: Vec<ServingKvTransferPathObservation>,
    pub kv_transfer_resources: Vec<String>,
    pub kv_transfer_resource_dependencies: Vec<usize>,
    pub kv_transfer_s: f64,
    pub kv_transfer_fit: Option<CalibrationFitApplication>,
    pub decode_queue_s: f64,
    pub decode_worker_queue_s: f64,
    pub decode_resource_queue_s: f64,
    pub decode_s: f64,
    pub phase_breakdown: Vec<ServingRequestPhaseBreakdownObservation>,
    pub lifecycle_events: Vec<ServingRequestEventObservation>,
    pub metric_source: String,
    pub worker_summary: Vec<ServingRequestWorkerSummaryObservation>,
    pub worker_assignments: Vec<ServingWorkerAssignmentObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRouteCandidateObservation {
    pub prefill_node: NodeId,
    pub decode_node: NodeId,
    pub prefill_route_nodes: Vec<NodeId>,
    pub prefill_route_gpus: Vec<GpuAddr>,
    pub decode_route_nodes: Vec<NodeId>,
    pub decode_route_gpus: Vec<GpuAddr>,
    pub selected: bool,
    pub routable: bool,
    pub rejection_reason: Option<String>,
    pub estimated_e2el_s: Option<f64>,
    pub estimated_kv_transfer_s: Option<f64>,
    pub estimated_kv_resource_wait_s: Option<f64>,
    pub estimated_prefill_wait_s: Option<f64>,
    pub estimated_decode_wait_s: Option<f64>,
    pub kv_transfer_bytes: u64,
    pub kv_transfer_bottlenecks: Vec<String>,
    pub kv_transfer_resources: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvTransferPathObservation {
    pub source: GpuAddr,
    pub destination: GpuAddr,
    pub latency_s: f64,
    pub bottleneck_bandwidth_gbps: f64,
    pub resources: Vec<String>,
    pub resource_details: Vec<ServingKvTransferPathResourceObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvTransferPathResourceObservation {
    pub kind: String,
    pub label: String,
    pub bandwidth_gbps: f64,
    pub latency_s: f64,
    pub rail_id: Option<u32>,
    pub from: Option<ServingKvTransferPathEndpointObservation>,
    pub to: Option<ServingKvTransferPathEndpointObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvTransferPathEndpointObservation {
    pub kind: String,
    pub node_id: Option<NodeId>,
    pub local_gpu_id: Option<u32>,
    pub nic_id: Option<u32>,
    pub rail_id: Option<u32>,
}

impl ServingRequestObservation {
    pub(super) fn from_decode_state(state: &DecodeRequestState) -> Self {
        let fallback_first_token_finish_s = state
            .first_decode_finish_s
            .unwrap_or(state.kv_finish_s.max(state.arrival_s));
        let fallback_first_decode_start_s = state
            .first_decode_start_s
            .unwrap_or(state.kv_finish_s.max(state.arrival_s));
        let fallback_last_token_finish_s = state
            .last_decode_finish_s
            .unwrap_or(fallback_first_token_finish_s);
        let fallback_tpot_s = if state.decode_tokens > 1 {
            (fallback_last_token_finish_s - fallback_first_token_finish_s)
                / f64::from(state.decode_tokens - 1)
        } else {
            (fallback_last_token_finish_s - state.kv_finish_s).max(0.0)
        };
        let status_time_s = state.status_time_s.or_else(|| {
            inferred_status_time_s(
                state,
                fallback_first_decode_start_s,
                fallback_first_token_finish_s,
            )
        });
        let lifecycle_events = request_lifecycle_events(state, status_time_s);
        let event_metrics = request_lifecycle_metric_summary(&lifecycle_events);
        let metric_source = if event_metrics.is_some() {
            "request_lifecycle_events"
        } else {
            "decode_state_fallback"
        };
        let first_decode_start_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.first_decode_start_s)
            .unwrap_or(fallback_first_decode_start_s);
        let first_token_finish_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.first_decode_finish_s)
            .unwrap_or(fallback_first_token_finish_s);
        let last_token_finish_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.last_decode_finish_s)
            .unwrap_or(fallback_last_token_finish_s);
        let decode_token_start_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_token_start_s.clone())
            .unwrap_or_else(|| state.decode_token_start_s.clone());
        let decode_token_finish_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_token_finish_s.clone())
            .unwrap_or_else(|| state.decode_token_finish_s.clone());
        let inter_token_latency_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.inter_token_latency_s.clone())
            .unwrap_or_else(|| inter_token_latencies(&decode_token_finish_s));
        let ttft_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.ttft_s)
            .unwrap_or_else(|| (first_token_finish_s - state.arrival_s).max(0.0));
        let tpot_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.tpot_s)
            .unwrap_or(fallback_tpot_s);
        let itl_s = mean(&inter_token_latency_s);
        let e2el_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.e2el_s)
            .unwrap_or_else(|| (last_token_finish_s - state.arrival_s).max(0.0));
        let service_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.service_s)
            .unwrap_or_else(|| (last_token_finish_s - state.prefill_start_s).max(0.0));
        let queue_delay_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.prefill_queue_s)
            .unwrap_or_else(|| (state.prefill_start_s - state.arrival_s).max(0.0));
        let prefill_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.prefill_s)
            .unwrap_or_else(|| (state.prefill_finish_s - state.prefill_start_s).max(0.0));
        let kv_queue_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.kv_queue_s)
            .unwrap_or_else(|| (state.kv_start_s - state.prefill_finish_s).max(0.0));
        let decode_queue_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_queue_s)
            .unwrap_or_else(|| (first_decode_start_s - state.kv_finish_s).max(0.0));
        let decode_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_s)
            .unwrap_or_else(|| (last_token_finish_s - first_decode_start_s).max(0.0));
        let completed = state.status.is_completed();
        let ttft_slo_missed = slo_missed(state.slo.ttft_s, ttft_s, completed);
        let tpot_slo_missed = slo_missed(state.slo.tpot_s, tpot_s, completed);
        let itl_slo_missed = slo_missed(state.slo.itl_s, itl_s, completed);
        let e2el_slo_missed = slo_missed(state.slo.e2el_s, e2el_s, completed);
        let deadline_missed = state.deadline_s.is_some_and(|deadline_s| {
            !state.status.is_completed() || last_token_finish_s > deadline_s + 1e-12
        });
        let phase_breakdown = request_phase_breakdown(
            state,
            status_time_s,
            first_decode_start_s,
            first_token_finish_s,
            last_token_finish_s,
        );
        let kv_block_ownership = request_kv_block_ownership(state);
        let worker_summary = request_worker_summary(state, &kv_block_ownership);
        let prefill_token_spans = prefill_capacity_spans(state);

        Self {
            request_idx: state.request_idx,
            request_id: state.request_id.clone(),
            tenant: state.tenant.clone(),
            model_id: state.model_id.clone(),
            traffic_class: state.traffic_class.clone(),
            shape_profile: state.shape_profile.clone(),
            cache_key: state.cache_key.clone(),
            status: state.status,
            status_time_s,
            failure_reason: state.failure_reason.clone(),
            rejection: state.failure_rejection.clone(),
            priority: state.priority,
            slo: state.slo,
            ttft_slo_missed,
            tpot_slo_missed,
            itl_slo_missed,
            e2el_slo_missed,
            deadline_s: state.deadline_s,
            deadline_missed,
            cancellation_s: state.cancellation_s,
            prefill_node: state.prefill_node,
            decode_node: state.decode_node,
            prefill_route_nodes: state.prefill_route_nodes.clone(),
            prefill_route_gpus: state.prefill_route_gpus.clone(),
            decode_route_nodes: state.decode_route_nodes.clone(),
            decode_route_gpus: state.decode_route_gpus.clone(),
            kv_cache_owner_gpus: state.decode_route_gpus.clone(),
            routing_policy: state.routing_policy,
            routing_candidate_count: state.routing_candidate_count,
            routing_routable_candidate_count: state.routing_routable_candidate_count,
            routing_estimated_e2el_s: state.routing_estimated_e2el_s,
            routing_estimated_kv_transfer_s: state.routing_estimated_kv_transfer_s,
            routing_estimated_kv_resource_wait_s: state.routing_estimated_kv_resource_wait_s,
            routing_estimated_prefill_wait_s: state.routing_estimated_prefill_wait_s,
            routing_estimated_decode_wait_s: state.routing_estimated_decode_wait_s,
            routing_reason: state.routing_reason.clone(),
            routing_candidates: state.routing_candidates.clone(),
            arrival_s: state.arrival_s,
            batch_size: state.batch_size,
            prompt_tokens: state.prompt_tokens,
            prefix_cache_hit_tokens: state.prefix_cache_hit_tokens,
            effective_prefill_tokens: state.effective_prefill_tokens,
            prefill_chunks: state.prefill_chunks,
            prefill_token_start_s: prefill_token_spans
                .iter()
                .map(|span| span.start_s)
                .collect(),
            prefill_token_finish_s: prefill_token_spans
                .iter()
                .map(|span| span.finish_s)
                .collect(),
            decode_tokens: state.decode_tokens,
            kv_block_tokens: state.kv_block_tokens,
            kv_cache_blocks: state.kv_cache_blocks,
            kv_allocated_tokens: state.kv_allocated_tokens,
            kv_fragmentation_tokens: state.kv_fragmentation_tokens,
            kv_block_ownership,
            prefill_start_s: state.prefill_start_s,
            prefill_finish_s: state.prefill_finish_s,
            kv_start_s: state.kv_start_s,
            kv_finish_s: state.kv_finish_s,
            first_decode_start_s,
            first_decode_finish_s: first_token_finish_s,
            last_decode_finish_s: last_token_finish_s,
            decode_iterations: decode_token_finish_s.len().min(u32::MAX as usize) as u32,
            decode_token_start_s,
            decode_token_finish_s,
            inter_token_latency_s,
            ttft_s,
            tpot_s,
            itl_s,
            e2el_s,
            service_s,
            queue_delay_s,
            prefill_s,
            prefill_worker_queue_s: state.prefill_worker_queue_s,
            prefill_resource_queue_s: state.prefill_resource_queue_s,
            kv_queue_s,
            kv_worker_queue_s: state.kv_worker_queue_s,
            kv_resource_queue_s: state.kv_resource_queue_s,
            kv_transfer_bytes: state.kv_transfer_bytes,
            kv_transfer_bottlenecks: state.kv_transfer_bottlenecks.clone(),
            kv_transfer_paths: state.kv_transfer_paths.clone(),
            kv_transfer_resources: state.kv_transfer_resources.clone(),
            kv_transfer_resource_dependencies: state.kv_transfer_resource_dependencies.clone(),
            kv_transfer_s: state.kv_transfer_s,
            kv_transfer_fit: state.kv_transfer_fit.clone(),
            decode_queue_s,
            decode_worker_queue_s: state.decode_worker_queue_s,
            decode_resource_queue_s: state.decode_resource_queue_s,
            decode_s,
            phase_breakdown,
            lifecycle_events,
            metric_source: metric_source.to_string(),
            worker_summary,
            worker_assignments: state.worker_assignments.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct RequestLifecycleMetricSummary {
    first_decode_start_s: f64,
    first_decode_finish_s: f64,
    last_decode_finish_s: f64,
    decode_token_start_s: Vec<f64>,
    decode_token_finish_s: Vec<f64>,
    inter_token_latency_s: Vec<f64>,
    ttft_s: f64,
    tpot_s: f64,
    e2el_s: f64,
    service_s: f64,
    prefill_queue_s: f64,
    prefill_s: f64,
    kv_queue_s: f64,
    decode_queue_s: f64,
    decode_s: f64,
}

fn request_lifecycle_metric_summary(
    events: &[ServingRequestEventObservation],
) -> Option<RequestLifecycleMetricSummary> {
    let arrival_s = first_lifecycle_event_time(events, ServingRequestEventKind::Arrived)?;
    let prefill_start_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::PrefillStarted)?;
    let prefill_finish_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::PrefillFinished)?;
    let kv_start_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::KvTransferStarted)?;
    let kv_finish_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::KvTransferFinished)?;
    let decode_token_start_s =
        lifecycle_event_times(events, ServingRequestEventKind::DecodeIterationStarted);
    let decode_token_finish_s =
        lifecycle_event_times(events, ServingRequestEventKind::DecodeIterationFinished);
    let first_decode_start_s = decode_token_start_s.first().copied()?;
    let first_decode_finish_s = decode_token_finish_s.first().copied()?;
    let last_decode_finish_s = decode_token_finish_s.last().copied()?;
    let inter_token_latency_s = inter_token_latencies(&decode_token_finish_s);
    let tpot_s = if decode_token_finish_s.len() > 1 {
        (last_decode_finish_s - first_decode_finish_s) / (decode_token_finish_s.len() as f64 - 1.0)
    } else {
        (last_decode_finish_s - kv_finish_s).max(0.0)
    };

    Some(RequestLifecycleMetricSummary {
        first_decode_start_s,
        first_decode_finish_s,
        last_decode_finish_s,
        decode_token_start_s,
        decode_token_finish_s,
        inter_token_latency_s,
        ttft_s: (first_decode_finish_s - arrival_s).max(0.0),
        tpot_s,
        e2el_s: (last_decode_finish_s - arrival_s).max(0.0),
        service_s: (last_decode_finish_s - prefill_start_s).max(0.0),
        prefill_queue_s: (prefill_start_s - arrival_s).max(0.0),
        prefill_s: (prefill_finish_s - prefill_start_s).max(0.0),
        kv_queue_s: (kv_start_s - prefill_finish_s).max(0.0),
        decode_queue_s: (first_decode_start_s - kv_finish_s).max(0.0),
        decode_s: (last_decode_finish_s - first_decode_start_s).max(0.0),
    })
}

fn first_lifecycle_event_time(
    events: &[ServingRequestEventObservation],
    kind: ServingRequestEventKind,
) -> Option<f64> {
    lifecycle_event_times(events, kind).into_iter().next()
}

fn lifecycle_event_times(
    events: &[ServingRequestEventObservation],
    kind: ServingRequestEventKind,
) -> Vec<f64> {
    let mut times = events
        .iter()
        .filter(|event| event.kind == kind && event.at_s.is_finite())
        .map(|event| event.at_s)
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    times
}

fn inferred_status_time_s(
    state: &DecodeRequestState,
    first_decode_start_s: f64,
    first_token_finish_s: f64,
) -> Option<f64> {
    match state.status {
        ServingRequestStatus::Pending => None,
        ServingRequestStatus::Completed | ServingRequestStatus::TimedOut => {
            state.last_decode_finish_s.or(Some(first_token_finish_s))
        }
        ServingRequestStatus::RejectedAdmission => Some(
            [
                state.kv_finish_s,
                state.prefill_finish_s,
                state.prefill_start_s,
                state.arrival_s,
            ]
            .into_iter()
            .find(|value| value.is_finite())
            .unwrap_or(state.arrival_s),
        ),
        ServingRequestStatus::Cancelled => state.cancellation_s.or_else(|| {
            [
                state.last_decode_finish_s,
                Some(first_decode_start_s),
                Some(state.kv_finish_s),
                Some(state.prefill_finish_s),
                Some(state.arrival_s),
            ]
            .into_iter()
            .flatten()
            .find(|value| value.is_finite())
        }),
    }
}

fn request_phase_breakdown(
    state: &DecodeRequestState,
    status_time_s: Option<f64>,
    first_decode_start_s: f64,
    first_token_finish_s: f64,
    last_token_finish_s: f64,
) -> Vec<ServingRequestPhaseBreakdownObservation> {
    let terminal_s = status_time_s.unwrap_or(f64::INFINITY);
    let mut phases = Vec::new();
    let prefill_queue_finish_s = if state.prefill_start_s.is_finite() {
        state.prefill_start_s
    } else {
        terminal_s
    };
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "queued_for_prefill",
        ServingRequestPhaseCategory::Queue,
        state.arrival_s,
        prefill_queue_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "prefilling",
        ServingRequestPhaseCategory::Service,
        state.prefill_start_s,
        state.prefill_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "queued_for_kv_transfer",
        ServingRequestPhaseCategory::Queue,
        state.prefill_finish_s,
        state.kv_start_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "transferring_kv",
        ServingRequestPhaseCategory::Transfer,
        state.kv_start_s,
        state.kv_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "queued_for_decode",
        ServingRequestPhaseCategory::Queue,
        state.kv_finish_s,
        first_decode_start_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "first_decode_iteration",
        ServingRequestPhaseCategory::Service,
        first_decode_start_s,
        first_token_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "decode_tail",
        ServingRequestPhaseCategory::Service,
        first_token_finish_s,
        last_token_finish_s,
        false,
        true,
    );
    phases
}

#[allow(clippy::too_many_arguments)]
fn push_phase_breakdown(
    phases: &mut Vec<ServingRequestPhaseBreakdownObservation>,
    terminal_s: f64,
    phase: &str,
    category: ServingRequestPhaseCategory,
    start_s: f64,
    finish_s: f64,
    contributes_to_ttft: bool,
    contributes_to_e2el: bool,
) {
    if !start_s.is_finite() || !finish_s.is_finite() || start_s > terminal_s + 1e-12 {
        return;
    }
    let finish_s = finish_s.min(terminal_s);
    if finish_s < start_s {
        return;
    }
    phases.push(ServingRequestPhaseBreakdownObservation {
        phase: phase.to_string(),
        category,
        start_s,
        finish_s,
        duration_s: (finish_s - start_s).max(0.0),
        contributes_to_ttft,
        contributes_to_e2el,
    });
}

pub(super) fn request_kv_block_ownership(
    state: &DecodeRequestState,
) -> Vec<ServingKvBlockOwnershipObservation> {
    let Some((allocated_at_s, released_at_s)) = kv_residency_window_s(state) else {
        return Vec::new();
    };
    kv_owner_allocations(state)
        .into_iter()
        .map(|allocation| {
            let owner_worker_slots = decode_owner_worker_slots(state, allocation.owner);
            let worker_slot_ownership = kv_worker_slot_ownership(
                &allocation,
                &owner_worker_slots,
                state.batch_size.max(1),
                state.kv_block_tokens,
            );
            ServingKvBlockOwnershipObservation {
                allocation_id: allocation.allocation_id,
                owner: allocation.owner,
                owner_worker_slots,
                worker_slot_ownership,
                decode_operation_ids: decode_owner_operation_ids(state, allocation.owner),
                allocated_at_s,
                released_at_s,
                duration_s: (released_at_s - allocated_at_s).max(0.0),
                block_start: allocation.block_start,
                block_end: allocation.block_end,
                decode_sequences: state.batch_size.max(1),
                resident_tokens: allocation.resident_tokens,
                kv_blocks: allocation.kv_blocks,
                allocated_kv_tokens: allocation.allocated_kv_tokens,
                kv_fragmentation_tokens: allocation.kv_fragmentation_tokens,
                block_table_entries: allocation.block_table_entries,
                block_table_bytes: allocation.block_table_bytes,
            }
        })
        .collect()
}

fn kv_worker_slot_ownership(
    allocation: &KvOwnerAllocation,
    owner_worker_slots: &[u32],
    decode_sequences: u32,
    kv_block_tokens: u32,
) -> Vec<ServingKvWorkerSlotOwnershipObservation> {
    let mut slots = owner_worker_slots.to_vec();
    slots.sort_unstable();
    slots.dedup();
    if slots.is_empty() {
        return Vec::new();
    }

    let block_counts = partition_units(allocation.kv_blocks, slots.len());
    let resident_token_counts = partition_tokens_by_block_capacity(
        allocation.resident_tokens,
        &block_counts,
        u64::from(kv_block_tokens.max(1)),
    );
    let sequence_counts = partition_units(u64::from(decode_sequences), slots.len());
    let mut block_start = allocation.block_start;

    slots
        .into_iter()
        .zip(block_counts)
        .zip(resident_token_counts)
        .zip(sequence_counts)
        .filter_map(|(((slot, kv_blocks), resident_tokens), decode_sequences)| {
            let block_end = block_start.saturating_add(kv_blocks);
            let allocated_kv_tokens = kv_blocks.saturating_mul(u64::from(kv_block_tokens.max(1)));
            let slot_allocation = if kv_blocks == 0 && resident_tokens == 0 && decode_sequences == 0
            {
                None
            } else {
                Some(ServingKvWorkerSlotOwnershipObservation {
                    allocation_id: format!(
                        "{}:slot-{}:blocks-{}-{}",
                        allocation.allocation_id, slot, block_start, block_end
                    ),
                    slot,
                    block_start,
                    block_end,
                    decode_sequences: decode_sequences.min(u64::from(u32::MAX)) as u32,
                    resident_tokens,
                    kv_blocks,
                    allocated_kv_tokens,
                    kv_fragmentation_tokens: allocated_kv_tokens.saturating_sub(resident_tokens),
                    block_table_entries: kv_blocks,
                    block_table_bytes: kv_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
                })
            };
            block_start = block_end;
            slot_allocation
        })
        .collect()
}

fn decode_owner_worker_slots(state: &DecodeRequestState, owner: GpuAddr) -> Vec<u32> {
    let mut slots = state
        .worker_assignments
        .iter()
        .filter(|assignment| {
            assignment.phase == "decode"
                && assignment.node_id == owner.node_id
                && assignment.local_gpu_id == owner.local_gpu_id
        })
        .map(|assignment| assignment.slot)
        .collect::<Vec<_>>();
    slots.sort_unstable();
    slots.dedup();
    slots
}

fn decode_owner_operation_ids(state: &DecodeRequestState, owner: GpuAddr) -> Vec<usize> {
    let mut operation_ids = state
        .worker_assignments
        .iter()
        .filter(|assignment| {
            assignment.phase == "decode"
                && assignment.node_id == owner.node_id
                && assignment.local_gpu_id == owner.local_gpu_id
        })
        .flat_map(|assignment| assignment.operation_ids.iter().copied())
        .collect::<Vec<_>>();
    operation_ids.sort_unstable();
    operation_ids.dedup();
    operation_ids
}

fn request_worker_summary(
    state: &DecodeRequestState,
    kv_block_ownership: &[ServingKvBlockOwnershipObservation],
) -> Vec<ServingRequestWorkerSummaryObservation> {
    let mut summary = Vec::new();
    summary.extend(worker_assignment_summary(
        &state.worker_assignments,
        "prefill",
        "prefill_source",
    ));
    summary.extend(worker_assignment_summary(
        &state.worker_assignments,
        "kv_transfer",
        "kv_transfer",
    ));
    summary.extend(worker_assignment_summary(
        &state.worker_assignments,
        "decode",
        "decode_owner",
    ));
    summary.extend(kv_block_ownership.iter().map(kv_cache_owner_worker_summary));
    summary
}

#[derive(Clone, Debug)]
struct WorkerSummaryAccumulator {
    worker_slots: Vec<u32>,
    operation_ids: Vec<usize>,
    assignment_count: u32,
    start_s: f64,
    finish_s: f64,
}

impl Default for WorkerSummaryAccumulator {
    fn default() -> Self {
        Self {
            worker_slots: Vec::new(),
            operation_ids: Vec::new(),
            assignment_count: 0,
            start_s: f64::INFINITY,
            finish_s: 0.0,
        }
    }
}

fn worker_assignment_summary(
    assignments: &[ServingWorkerAssignmentObservation],
    phase: &str,
    role: &str,
) -> Vec<ServingRequestWorkerSummaryObservation> {
    let mut grouped: BTreeMap<(NodeId, u32), WorkerSummaryAccumulator> = BTreeMap::new();
    for assignment in assignments
        .iter()
        .filter(|assignment| assignment.phase == phase)
    {
        let entry = grouped
            .entry((assignment.node_id, assignment.local_gpu_id))
            .or_default();
        entry.worker_slots.push(assignment.slot);
        entry
            .operation_ids
            .extend(assignment.operation_ids.iter().copied());
        entry.assignment_count = entry.assignment_count.saturating_add(1);
        entry.start_s = entry.start_s.min(assignment.start_s);
        entry.finish_s = entry.finish_s.max(assignment.finish_s);
    }

    grouped
        .into_iter()
        .map(|((node_id, local_gpu_id), mut entry)| {
            entry.worker_slots.sort_unstable();
            entry.worker_slots.dedup();
            entry.operation_ids.sort_unstable();
            entry.operation_ids.dedup();
            ServingRequestWorkerSummaryObservation {
                role: role.to_string(),
                phase: phase.to_string(),
                node_id,
                local_gpu_id,
                worker_slots: entry.worker_slots,
                operation_ids: entry.operation_ids,
                assignment_count: entry.assignment_count,
                start_s: if entry.start_s.is_finite() {
                    entry.start_s
                } else {
                    0.0
                },
                finish_s: entry.finish_s,
                resident_tokens: 0,
                kv_blocks: 0,
            }
        })
        .collect()
}

fn kv_cache_owner_worker_summary(
    ownership: &ServingKvBlockOwnershipObservation,
) -> ServingRequestWorkerSummaryObservation {
    ServingRequestWorkerSummaryObservation {
        role: "kv_cache_owner".to_string(),
        phase: "decode".to_string(),
        node_id: ownership.owner.node_id,
        local_gpu_id: ownership.owner.local_gpu_id,
        worker_slots: ownership.owner_worker_slots.clone(),
        operation_ids: ownership.decode_operation_ids.clone(),
        assignment_count: ownership.owner_worker_slots.len().min(u32::MAX as usize) as u32,
        start_s: ownership.allocated_at_s,
        finish_s: ownership.released_at_s,
        resident_tokens: ownership.resident_tokens,
        kv_blocks: ownership.kv_blocks,
    }
}

pub(super) fn kv_residency_window_s(state: &DecodeRequestState) -> Option<(f64, f64)> {
    if !state.status.is_admitted() {
        return None;
    }
    let allocated_at_s = state.kv_finish_s;
    if !allocated_at_s.is_finite() {
        return None;
    }

    let mut released_at_s = state.last_decode_finish_s?;
    if state.status == ServingRequestStatus::Cancelled
        && let Some(cancellation_s) = state.cancellation_s.or(state.status_time_s)
    {
        released_at_s = released_at_s.min(cancellation_s);
    }
    if state.status == ServingRequestStatus::TimedOut
        && let Some(timeout_s) = state.request_timeout_s
    {
        released_at_s = released_at_s.min(state.arrival_s + timeout_s);
    }
    if state.status == ServingRequestStatus::TimedOut
        && let Some(status_time_s) = state.status_time_s
        && status_time_s > released_at_s
    {
        released_at_s = status_time_s;
    }
    if !released_at_s.is_finite() || released_at_s <= allocated_at_s {
        return None;
    }

    Some((allocated_at_s, released_at_s))
}

fn prefill_terminal_s(state: &DecodeRequestState) -> Option<f64> {
    if state.status == ServingRequestStatus::Cancelled {
        return state.cancellation_s.or(state.status_time_s);
    }
    if state.status == ServingRequestStatus::TimedOut
        && let Some(timeout_s) = state.request_timeout_s
    {
        return Some(state.arrival_s + timeout_s);
    }
    state.status_time_s
}

pub(super) fn prefill_capacity_spans(state: &DecodeRequestState) -> Vec<PrefillTokenSpan> {
    if !state.status.is_admitted() {
        return Vec::new();
    }

    let terminal_s = prefill_terminal_s(state).unwrap_or(f64::INFINITY);
    let mut spans = state
        .prefill_token_spans
        .iter()
        .filter_map(|span| {
            let finish_s = span.finish_s.min(terminal_s);
            if span.tokens == 0
                || !span.start_s.is_finite()
                || !finish_s.is_finite()
                || finish_s <= span.start_s
            {
                return None;
            }
            Some(PrefillTokenSpan {
                start_s: span.start_s,
                finish_s,
                tokens: span.tokens,
            })
        })
        .collect::<Vec<_>>();
    if !spans.is_empty() {
        return spans;
    }

    let start_s = state.prefill_start_s;
    let finish_s = state.prefill_finish_s.min(terminal_s);
    let tokens = u64::from(state.batch_size.max(1))
        .saturating_mul(u64::from(state.effective_prefill_tokens));
    if tokens > 0 && start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
        spans.push(PrefillTokenSpan {
            start_s,
            finish_s,
            tokens,
        });
    }
    spans
}

pub(super) fn prefill_owner_gpus(state: &DecodeRequestState) -> Vec<GpuAddr> {
    if state.prefill_route_gpus.is_empty() {
        vec![GpuAddr {
            node_id: state.prefill_node,
            local_gpu_id: 0,
        }]
    } else {
        state.prefill_route_gpus.clone()
    }
}

pub(super) fn prefill_owner_nodes(state: &DecodeRequestState) -> Vec<NodeId> {
    if state.prefill_route_nodes.is_empty() {
        vec![state.prefill_node]
    } else {
        state.prefill_route_nodes.clone()
    }
}

pub(super) fn decode_owner_gpus(state: &DecodeRequestState) -> Vec<GpuAddr> {
    if state.decode_route_gpus.is_empty() {
        vec![GpuAddr {
            node_id: state.decode_node,
            local_gpu_id: 0,
        }]
    } else {
        state.decode_route_gpus.clone()
    }
}

fn kv_ownership_event_message(state: &DecodeRequestState) -> String {
    let block_table_bytes = state
        .kv_cache_blocks
        .saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES);
    format!(
        "kv_blocks={} resident_tokens={} allocated_kv_tokens={} block_table_bytes={} owners={}",
        state.kv_cache_blocks,
        u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1)),
        state.kv_allocated_tokens,
        block_table_bytes,
        decode_owner_gpus(state).len()
    )
}

fn request_lifecycle_events(
    state: &DecodeRequestState,
    status_time_s: Option<f64>,
) -> Vec<ServingRequestEventObservation> {
    let terminal_s = status_time_s.unwrap_or(f64::INFINITY);
    let mut events = Vec::new();
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::Arrived,
        "arrival",
        state.arrival_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::QueuedForPrefill,
        "prefill",
        state.arrival_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::PrefillStarted,
        "prefill",
        state.prefill_start_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::PrefillFinished,
        "prefill",
        state.prefill_finish_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::QueuedForKvTransfer,
        "kv_transfer",
        state.prefill_finish_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::KvTransferStarted,
        "kv_transfer",
        state.kv_start_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::KvTransferFinished,
        "kv_transfer",
        state.kv_finish_s,
        None,
        None,
    );
    if let Some((allocated_at_s, released_at_s)) = kv_residency_window_s(state) {
        push_lifecycle_event(
            &mut events,
            terminal_s,
            ServingRequestEventKind::KvBlocksAllocated,
            "kv_cache",
            allocated_at_s,
            None,
            Some(kv_ownership_event_message(state)),
        );
        if state.first_decode_start_s.is_some() {
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::QueuedForDecode,
                "decode",
                state.kv_finish_s,
                None,
                None,
            );
        }
        for (idx, start_s) in state.decode_token_start_s.iter().copied().enumerate() {
            let decode_iteration = Some(idx.min(u32::MAX as usize) as u32);
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::DecodeIterationStarted,
                "decode",
                start_s,
                decode_iteration,
                None,
            );
            if let Some(finish_s) = state.decode_token_finish_s.get(idx).copied() {
                push_lifecycle_event(
                    &mut events,
                    terminal_s,
                    ServingRequestEventKind::DecodeIterationFinished,
                    "decode",
                    finish_s,
                    decode_iteration,
                    None,
                );
            }
        }
        push_lifecycle_event(
            &mut events,
            terminal_s,
            ServingRequestEventKind::KvBlocksReleased,
            "kv_cache",
            released_at_s,
            None,
            Some(kv_ownership_event_message(state)),
        );
    } else {
        if state.first_decode_start_s.is_some() {
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::QueuedForDecode,
                "decode",
                state.kv_finish_s,
                None,
                None,
            );
        }
        for (idx, start_s) in state.decode_token_start_s.iter().copied().enumerate() {
            let decode_iteration = Some(idx.min(u32::MAX as usize) as u32);
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::DecodeIterationStarted,
                "decode",
                start_s,
                decode_iteration,
                None,
            );
            if let Some(finish_s) = state.decode_token_finish_s.get(idx).copied() {
                push_lifecycle_event(
                    &mut events,
                    terminal_s,
                    ServingRequestEventKind::DecodeIterationFinished,
                    "decode",
                    finish_s,
                    decode_iteration,
                    None,
                );
            }
        }
    }
    if let Some(status_time_s) = status_time_s
        && let Some(kind) = terminal_event_kind(state.status)
    {
        push_lifecycle_event(
            &mut events,
            f64::INFINITY,
            kind,
            "terminal",
            status_time_s,
            None,
            state.failure_reason.clone(),
        );
    }
    events
}

fn terminal_event_kind(status: ServingRequestStatus) -> Option<ServingRequestEventKind> {
    match status {
        ServingRequestStatus::Pending => None,
        ServingRequestStatus::Completed => Some(ServingRequestEventKind::Completed),
        ServingRequestStatus::RejectedAdmission => Some(ServingRequestEventKind::RejectedAdmission),
        ServingRequestStatus::TimedOut => Some(ServingRequestEventKind::TimedOut),
        ServingRequestStatus::Cancelled => Some(ServingRequestEventKind::Cancelled),
    }
}

fn push_lifecycle_event(
    events: &mut Vec<ServingRequestEventObservation>,
    terminal_s: f64,
    kind: ServingRequestEventKind,
    phase: &str,
    at_s: f64,
    decode_iteration: Option<u32>,
    message: Option<String>,
) {
    if !at_s.is_finite() || at_s > terminal_s + 1e-12 {
        return;
    }
    events.push(ServingRequestEventObservation {
        kind,
        phase: phase.to_string(),
        at_s,
        decode_iteration,
        message,
    });
}
