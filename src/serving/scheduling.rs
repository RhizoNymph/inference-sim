use super::*;

mod decode;
mod pipeline;
mod prefill;
mod summary;
pub(super) use decode::*;
pub(super) use pipeline::*;
pub(super) use prefill::*;
pub(super) use summary::*;

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct ServingWorkerRuntime {
    pub(super) prefill_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    pub(super) decode_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    pub(super) kv_transfer_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ServingSimulation {
    pub(super) metrics: ServingMetrics,
    pub(super) calibration_fits: Vec<CalibrationFitApplication>,
    pub(super) measurement_window: MeasurementWindowSelection,
    pub(super) metric_breakdowns: Vec<ServingMetricBreakdown>,
    pub(super) request_observations: Vec<ServingRequestObservation>,
    pub(super) decode_iterations: Vec<ServingDecodeIterationObservation>,
    pub(super) node_capacity: Vec<ServingNodeCapacityObservation>,
    pub(super) gpu_capacity: Vec<ServingGpuCapacityObservation>,
    pub(super) traffic_class_capacity: Vec<ServingTrafficClassCapacityObservation>,
    pub(super) service_observations: Vec<ServingServiceObservation>,
    pub(super) worker_observations: Vec<ServingWorkerObservation>,
    pub(super) scheduled_operations: Vec<ScheduledOperation>,
    pub(super) resource_utilization: Vec<ResourceUtilization>,
    pub(super) phase_resource_utilization: Vec<ServingPhaseResourceUtilization>,
    pub(super) kv_bottlenecks: Vec<String>,
    /// Scheduler that produced this simulation; `None` when it never ran.
    pub(super) scheduler_model: Option<SchedulerModel>,
}

impl ServingSimulation {
    pub(super) fn rejected() -> Self {
        Self {
            metrics: ServingMetrics::rejected(),
            calibration_fits: Vec::new(),
            measurement_window: MeasurementWindowSelection::rejected(),
            metric_breakdowns: Vec::new(),
            request_observations: Vec::new(),
            decode_iterations: Vec::new(),
            node_capacity: Vec::new(),
            gpu_capacity: Vec::new(),
            traffic_class_capacity: Vec::new(),
            service_observations: Vec::new(),
            worker_observations: Vec::new(),
            scheduled_operations: Vec::new(),
            resource_utilization: Vec::new(),
            phase_resource_utilization: Vec::new(),
            kv_bottlenecks: Vec::new(),
            scheduler_model: None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn schedule_serving_simulation(
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
) -> ServingSimulation {
    let calibration = calibration.sanitized();
    let scheduled_requests = traffic.request_count(calibration);
    let arrival_times = traffic.arrival_times(scheduled_requests, calibration);
    let base_decode_tokens = request.decode_tokens.max(1);
    let decode_total_s = decode_score.estimated_latency_s;
    let decode_one_s = decode_one_score.estimated_latency_s;
    let decode_tail_token_s = if base_decode_tokens > 1 {
        ((decode_total_s - decode_one_s).max(0.0) / f64::from(base_decode_tokens - 1))
            .max(decode_one_s)
    } else {
        decode_one_s
    };
    let decode_tail_scale = decode_tail_token_s / decode_one_s.max(1e-12);

    let prefill_resource_base_node = single_placement_node(prefill_score);
    let prefill_placement_nodes = placement_nodes(prefill_score);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let decode_placement_nodes = placement_nodes(decode_one_score);
    let mut decode_states = Vec::new();
    let mut routing_load = RoutingLoad::default();

    for request_idx in 0..scheduled_requests {
        let arrival_s = arrival_times
            .get(request_idx as usize)
            .copied()
            .unwrap_or(0.0);
        let request_shape = traffic.request_at(request, request_idx);
        let prefix_cache_hit_tokens =
            traffic.prefix_cache_hit_tokens(request_idx, request_shape.prompt_tokens);
        let effective_prefill_tokens = request_shape
            .prompt_tokens
            .saturating_sub(prefix_cache_hit_tokens);
        let route = route_request(
            cluster,
            model,
            &request_shape,
            effective_prefill_tokens,
            traffic,
            request_idx,
            arrival_s,
            prefill_nodes,
            decode_nodes,
            prefill_resource_base_node,
            &prefill_placement_nodes,
            decode_resource_base_node,
            &decode_placement_nodes,
            prefill_score,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            request.prompt_tokens,
            calibration,
            calibration_profile,
            &mut routing_load,
        );
        let decode_tokens = request_shape.decode_tokens.max(1);
        let max_sequence_tokens = request_shape
            .max_sequence_tokens
            .max(request_shape.prompt_tokens.saturating_add(decode_tokens))
            .max(1);
        let kv_block_tokens = kv_block_tokens(traffic);
        let kv_allocation = sequence_kv_allocation(
            request_shape.batch_size.max(1),
            max_sequence_tokens,
            kv_block_tokens,
        );
        decode_states.push(DecodeRequestState {
            request_idx,
            request_id: traffic.request_id(request_idx),
            tenant: traffic.tenant(request_idx),
            model_id: traffic.model_id(request_idx),
            traffic_class: traffic.traffic_class_name(request_idx),
            shape_profile: traffic.shape_profile_name(request_idx),
            cache_key: traffic.cache_key(request_idx),
            prefill_node: route.prefill_node,
            decode_node: route.decode_node,
            prefill_route_nodes: route.prefill_route_nodes,
            prefill_route_gpus: route.prefill_route_gpus,
            decode_route_nodes: route.decode_route_nodes,
            decode_route_gpus: route.decode_route_gpus,
            routing_policy: route.routing.policy,
            routing_candidate_count: route.routing.candidate_count,
            routing_routable_candidate_count: route.routing.routable_candidate_count,
            routing_estimated_e2el_s: route.routing.estimated_e2el_s,
            routing_estimated_kv_transfer_s: route.routing.estimated_kv_transfer_s,
            routing_estimated_kv_resource_wait_s: route.routing.estimated_kv_resource_wait_s,
            routing_estimated_prefill_wait_s: route.routing.estimated_prefill_wait_s,
            routing_estimated_decode_wait_s: route.routing.estimated_decode_wait_s,
            routing_reason: route.routing.reason,
            routing_candidates: route.routing.candidates,
            arrival_s,
            priority: traffic.effective_priority(request_idx),
            batch_size: request_shape.batch_size.max(1),
            prompt_tokens: request_shape.prompt_tokens.max(1),
            prefix_cache_hit_tokens,
            effective_prefill_tokens,
            remaining_prefill_tokens: effective_prefill_tokens,
            prefill_chunks: 0,
            decode_tokens,
            slo: traffic.effective_slo(request_idx),
            max_queue_delay_s: traffic.effective_max_queue_delay_s(request_idx),
            max_kv_queue_delay_s: traffic.effective_max_kv_queue_delay_s(request_idx),
            max_decode_queue_delay_s: traffic.effective_max_decode_queue_delay_s(request_idx),
            max_decode_iteration_queue_delay_s: traffic
                .effective_max_decode_iteration_queue_delay_s(request_idx),
            request_timeout_s: traffic.effective_request_timeout_s(request_idx),
            deadline_s: traffic.deadline_s(request_idx, arrival_s),
            cancellation_s: traffic.cancellation_s(request_idx, arrival_s),
            remaining_tokens: decode_tokens,
            emitted_tokens: 0,
            max_sequence_tokens,
            kv_block_tokens,
            kv_cache_blocks: kv_allocation.blocks,
            kv_allocated_tokens: kv_allocation.allocated_tokens,
            kv_fragmentation_tokens: kv_allocation.fragmentation_tokens,
            prefill_scheduled: false,
            dependencies: Vec::new(),
            kv_start_s: 0.0,
            kv_finish_s: 0.0,
            first_decode_start_s: None,
            first_decode_finish_s: None,
            last_decode_finish_s: None,
            decode_token_start_s: Vec::with_capacity(decode_tokens as usize),
            decode_token_finish_s: Vec::with_capacity(decode_tokens as usize),
            prefill_start_s: 0.0,
            prefill_finish_s: 0.0,
            prefill_worker_queue_s: 0.0,
            prefill_resource_queue_s: 0.0,
            decode_worker_queue_s: 0.0,
            decode_resource_queue_s: 0.0,
            kv_transfer_bytes: 0,
            kv_transfer_bottlenecks: Vec::new(),
            kv_transfer_paths: Vec::new(),
            kv_transfer_resources: Vec::new(),
            kv_transfer_resource_dependencies: Vec::new(),
            kv_worker_queue_s: 0.0,
            kv_resource_queue_s: 0.0,
            kv_transfer_s: 0.0,
            kv_transfer_fit: None,
            prefill_token_spans: Vec::new(),
            worker_assignments: Vec::new(),
            status: ServingRequestStatus::Pending,
            status_time_s: None,
            failure_reason: None,
            failure_rejection: None,
        });
    }

    let scheduler_model =
        select_scheduler_model(traffic, prefill_score, decode_score, &decode_states);
    let (scheduler_model, timeline) = match scheduler_model {
        SchedulerModel::IterationEngine => {
            let mut engine_states = decode_states.clone();
            match run_iteration_engine(
                &mut engine_states,
                cluster,
                model,
                decode_score,
                traffic,
                calibration,
            ) {
                Ok(engine) => {
                    decode_states = engine_states;
                    (
                        SchedulerModel::IterationEngine,
                        ScheduledTimeline {
                            operations: engine.operations,
                            decode_iterations: engine.decode_iterations,
                            kv_bottlenecks: Vec::new(),
                        },
                    )
                }
                // Unreachable for a placed config; keep a result rather than
                // dropping the candidate.
                Err(_) => (
                    SchedulerModel::PhasePipeline(
                        PhasePipelineReason::SplitPrefillDecodeParallelism,
                    ),
                    schedule_phase_pipeline(
                        &mut decode_states,
                        prefill_score,
                        decode_one_score,
                        cluster,
                        model,
                        request,
                        traffic,
                        calibration,
                        calibration_profile,
                        decode_tail_scale,
                    ),
                ),
            }
        }
        SchedulerModel::PhasePipeline(reason) => (
            SchedulerModel::PhasePipeline(reason),
            schedule_phase_pipeline(
                &mut decode_states,
                prefill_score,
                decode_one_score,
                cluster,
                model,
                request,
                traffic,
                calibration,
                calibration_profile,
                decode_tail_scale,
            ),
        ),
    };
    apply_terminal_statuses(&mut decode_states, traffic);
    summarize_serving_simulation(
        decode_states,
        timeline,
        SummaryContext {
            prefill_score,
            decode_score,
            model,
            request,
            traffic,
            prefill_nodes,
            decode_nodes,
            calibration_profile,
            scheduled_requests,
            scheduler_model,
        },
    )
}

pub(super) fn request_rejection(
    phase: &str,
    category: &str,
    resource: &str,
    code: &str,
    message: String,
) -> ServingRejection {
    ServingRejection {
        phase: phase.to_string(),
        category: category.to_string(),
        resource: resource.to_string(),
        code: code.to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: None,
        message,
    }
}

pub(super) fn with_rejection_details(
    mut rejection: ServingRejection,
    observed: Option<f64>,
    limit: Option<f64>,
    unit: Option<&str>,
    remediation: Option<&str>,
) -> ServingRejection {
    rejection.observed = observed;
    rejection.limit = limit;
    rejection.unit = unit.map(str::to_string);
    rejection.remediation = remediation.map(str::to_string);
    rejection
}

pub(super) fn set_request_rejection(state: &mut DecodeRequestState, rejection: ServingRejection) {
    state.failure_reason = Some(rejection.message.clone());
    state.failure_rejection = Some(rejection);
}

pub(super) fn reject_admission(
    state: &mut DecodeRequestState,
    queue_delay_s: f64,
    max_queue_delay_s: f64,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(state.arrival_s + queue_delay_s);
    let message = format!(
        "admission rejected: prefill queue delay {queue_delay_s:.6}s > max_queue_delay {max_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "prefill",
                "queueing",
                "prefill_queue",
                "prefill_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_queue_delay_s),
            Some("s"),
            Some("increase max_queue_delay, add prefill workers, or reduce arrival pressure"),
        ),
    );
    state.dependencies.clear();
    state.prefill_start_s = f64::INFINITY;
    state.prefill_finish_s = f64::INFINITY;
    state.kv_start_s = f64::INFINITY;
    state.kv_finish_s = f64::INFINITY;
    state.kv_worker_queue_s = 0.0;
    state.kv_resource_queue_s = 0.0;
    state.kv_transfer_resources.clear();
    state.kv_transfer_resource_dependencies.clear();
    state.kv_transfer_s = 0.0;
}

pub(super) fn reject_prefill_capacity_admission(
    state: &mut DecodeRequestState,
    admission_s: f64,
    rejection: ServingRejection,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if admission_s.is_finite() {
        admission_s
    } else {
        state.arrival_s
    });
    set_request_rejection(state, rejection);
    state.dependencies.clear();
    state.prefill_start_s = f64::INFINITY;
    state.prefill_finish_s = f64::INFINITY;
    state.kv_start_s = f64::INFINITY;
    state.kv_finish_s = f64::INFINITY;
    state.kv_worker_queue_s = 0.0;
    state.kv_resource_queue_s = 0.0;
    state.kv_transfer_bottlenecks.clear();
    state.kv_transfer_paths.clear();
    state.kv_transfer_resources.clear();
    state.kv_transfer_resource_dependencies.clear();
    state.kv_transfer_s = 0.0;
    state.prefill_token_spans.clear();
}

pub(super) fn reject_kv_queue_admission(
    state: &mut DecodeRequestState,
    ready_s: f64,
    queue_delay_s: f64,
    max_kv_queue_delay_s: f64,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if ready_s.is_finite() {
        ready_s + max_kv_queue_delay_s
    } else {
        state.prefill_finish_s.max(state.arrival_s)
    });
    let message = format!(
        "admission rejected: KV transfer queue delay {queue_delay_s:.6}s > max_kv_queue_delay {max_kv_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "kv_transfer",
                "queueing",
                "kv_transfer_queue",
                "kv_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_kv_queue_delay_s),
            Some("s"),
            Some(
                "increase max_kv_queue_delay, add KV transfer worker slots, or improve KV route capacity",
            ),
        ),
    );
    state.dependencies.clear();
    state.kv_finish_s = f64::INFINITY;
}

pub(super) fn cancel_request(state: &mut DecodeRequestState, cancellation_s: f64, reason: &str) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::Cancelled;
    state.status_time_s = Some(cancellation_s);
    state.failure_reason = Some(format!("{reason}: cancellation at {cancellation_s:.6}s"));
    state.failure_rejection = None;
}
