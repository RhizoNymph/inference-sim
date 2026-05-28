use super::*;

mod decode;
mod prefill;
pub(super) use decode::*;
pub(super) use prefill::*;

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
    let mut scheduler = ResourceScheduler::new();
    let mut decode_states = Vec::new();
    let mut decode_iterations = Vec::new();
    let mut kv_bottlenecks = Vec::new();
    let mut routing_load = RoutingLoad::default();
    let mut worker_runtime = ServingWorkerRuntime::default();

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

    schedule_prefills(
        &mut scheduler,
        &mut worker_runtime,
        &mut decode_states,
        prefill_score,
        traffic,
        prefill_resource_base_node,
        request.batch_size,
        request.prompt_tokens,
    );
    for state in &mut decode_states {
        if state.status == ServingRequestStatus::Pending
            && let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= state.prefill_finish_s + 1e-12
        {
            cancel_request(state, cancellation_s, "request cancelled during prefill");
        }
    }

    for state in &mut decode_states {
        if state.status != ServingRequestStatus::Pending {
            continue;
        }
        let request_shape = InferenceRequest {
            batch_size: state.batch_size,
            prompt_tokens: state.prompt_tokens,
            decode_tokens: state.decode_tokens,
            max_sequence_tokens: state.max_sequence_tokens,
            phase: request.phase,
        };
        let kv_bytes = kv_transfer_bytes_for_routes(
            model,
            &request_shape,
            &state.prefill_route_nodes,
            &state.decode_route_nodes,
            &state.prefill_route_gpus,
            &state.decode_route_gpus,
        );
        let (kv_cost, kv_fit) = Solver::estimate_transfer_between_gpus_with_observation(
            cluster,
            &state.prefill_route_gpus,
            &state.decode_route_gpus,
            kv_bytes,
            SolverOptions {
                calibration,
                calibration_profile,
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );
        for bottleneck in &kv_cost.bottlenecks {
            if !kv_bottlenecks.contains(bottleneck) {
                kv_bottlenecks.push(bottleneck.clone());
            }
        }
        let kv_paths = kv_transfer_paths(
            cluster,
            &state.prefill_route_gpus,
            &state.decode_route_gpus,
            kv_bytes,
        );
        let kv_resources = kv_transfer_scheduler_resources(&kv_paths, &kv_cost.bottlenecks);
        let phase_ready_s = operation_finish_s(&scheduler, &state.dependencies);
        let kv_worker_ready_s = if kv_bytes.as_bytes() > 0 {
            kv_transfer_worker_slots_per_gpu(traffic)
                .map(|slots| kv_transfer_worker_ready_s(state, &worker_runtime, slots))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let kv_candidate_ready_s = phase_ready_s.max(kv_worker_ready_s);
        let kv_preview = scheduler.preview(
            kv_cost.total_s,
            kv_candidate_ready_s,
            &state.dependencies,
            kv_resources.clone(),
        );
        state.kv_start_s = kv_preview.start_s;
        state.kv_worker_queue_s = finite_or_zero(kv_worker_ready_s - phase_ready_s);
        state.kv_resource_queue_s = finite_or_zero(kv_preview.start_s - kv_candidate_ready_s);
        state.kv_transfer_bytes = kv_bytes.as_bytes();
        state.kv_transfer_bottlenecks = kv_cost.bottlenecks.clone();
        state.kv_transfer_paths = kv_paths.clone();
        state.kv_transfer_resources = kv_preview.resources.clone();
        state.kv_transfer_resource_dependencies = kv_preview.resource_dependencies.clone();
        state.kv_transfer_s = kv_cost.total_s;
        state.kv_transfer_fit = kv_fit.clone();
        let kv_queue_s = finite_or_zero(kv_preview.start_s - phase_ready_s);
        if let Some(max_kv_queue_delay_s) = state.max_kv_queue_delay_s
            && kv_queue_s > max_kv_queue_delay_s + 1e-12
        {
            reject_kv_queue_admission(state, phase_ready_s, kv_queue_s, max_kv_queue_delay_s);
            continue;
        }
        let kv_id = scheduler.schedule(
            format!(
                "request {} kv-transfer {}->{}",
                state.request_idx, state.prefill_node, state.decode_node
            ),
            kv_cost.total_s,
            kv_candidate_ready_s,
            &state.dependencies,
            kv_resources,
        );
        state.kv_finish_s = operation_finish_s(&scheduler, &[kv_id]);
        if kv_bytes.as_bytes() > 0
            && let Some(slots) = kv_transfer_worker_slots_per_gpu(traffic)
        {
            let assignments = assign_kv_transfer_workers_ready(
                &mut worker_runtime,
                state,
                slots,
                state.kv_start_s,
                state.kv_finish_s,
                &[kv_id],
            );
            state.worker_assignments.extend(assignments);
        }
        state.dependencies = vec![kv_id];
        if let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= state.kv_finish_s + 1e-12
        {
            cancel_request(
                state,
                cancellation_s,
                "request cancelled during KV transfer",
            );
        }
    }
    apply_decode_capacity_admission(
        &mut decode_states,
        traffic,
        decode_one_score,
        decode_tail_scale,
        request.batch_size,
        decode_worker_slots_per_gpu(traffic),
    );

    match traffic.decode_batching {
        ServingDecodeBatching::Independent => schedule_independent_decodes(
            &mut scheduler,
            &mut worker_runtime,
            &mut decode_states,
            &mut decode_iterations,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            decode_worker_slots_per_gpu(traffic),
        ),
        ServingDecodeBatching::Continuous { max_batch_tokens } => schedule_continuous_decodes(
            &mut scheduler,
            &mut worker_runtime,
            &mut decode_states,
            &mut decode_iterations,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            max_batch_tokens,
            decode_worker_slots_per_gpu(traffic),
        ),
    }
    apply_terminal_statuses(&mut decode_states, traffic);

    let mut calibration_fits = decode_states
        .iter()
        .filter_map(|state| state.kv_transfer_fit.clone())
        .collect::<Vec<_>>();
    let observations: Vec<_> = decode_states
        .iter()
        .map(ServingRequestObservation::from_decode_state)
        .collect();
    let makespan_s = scheduler.makespan_s();
    let measurement_window =
        effective_measurement_window(traffic, &observations, scheduler.operations(), makespan_s);
    let measurement_start_s = measurement_window.start_s;
    let measurement_end_s = measurement_window.end_s;
    let measured_all_observations = observations
        .iter()
        .filter(|observation| {
            observation.arrival_s + 1e-12 >= measurement_start_s
                && observation.arrival_s <= measurement_end_s + 1e-12
        })
        .cloned()
        .collect::<Vec<_>>();
    let measured_observations = measured_all_observations
        .iter()
        .filter(|observation| observation.status.is_completed())
        .cloned()
        .collect::<Vec<_>>();
    let measured_request_indices = measured_observations
        .iter()
        .map(|observation| observation.request_idx)
        .collect::<BTreeSet<_>>();
    let measured_decode_iteration_latencies = decode_iterations
        .iter()
        .filter(|iteration| {
            iteration
                .request_indices
                .iter()
                .any(|request_idx| measured_request_indices.contains(request_idx))
        })
        .map(|iteration| iteration.latency_s)
        .filter(|latency_s| latency_s.is_finite())
        .collect::<Vec<_>>();
    let measured_output_tokens = measured_observations
        .iter()
        .fold(0_u64, |total, observation| {
            total.saturating_add(observation_output_tokens(observation))
        });
    let measured_prompt_tokens = measured_observations
        .iter()
        .map(|observation| u64::from(observation.batch_size) * u64::from(observation.prompt_tokens))
        .sum::<u64>();
    let measured_prefix_cache_hit_tokens = measured_observations
        .iter()
        .map(|observation| {
            u64::from(observation.batch_size) * u64::from(observation.prefix_cache_hit_tokens)
        })
        .sum::<u64>();
    let measured_effective_prefill_tokens = measured_observations
        .iter()
        .map(|observation| {
            u64::from(observation.batch_size) * u64::from(observation.effective_prefill_tokens)
        })
        .sum::<u64>();
    let measured_prefill_chunks = measured_observations
        .iter()
        .map(|observation| u64::from(observation.prefill_chunks))
        .sum::<u64>();
    let measured_prefix_cache_hit_rate = if measured_prompt_tokens > 0 {
        measured_prefix_cache_hit_tokens as f64 / measured_prompt_tokens as f64
    } else {
        0.0
    };
    let measurement_duration_s = (measurement_end_s - measurement_start_s).max(0.0);
    let throughput_tokens_per_s = if measurement_duration_s > 0.0 {
        measured_output_tokens as f64 / measurement_duration_s
    } else {
        0.0
    };

    let ttft = collect_metric(&measured_observations, |observation| observation.ttft_s);
    let tpot = collect_metric(&measured_observations, |observation| observation.tpot_s);
    let itl = measured_observations
        .iter()
        .flat_map(|observation| observation.inter_token_latency_s.iter().copied())
        .collect::<Vec<_>>();
    let e2el = collect_metric(&measured_observations, |observation| observation.e2el_s);
    let service = collect_metric(&measured_observations, |observation| observation.service_s);
    let prefill = collect_metric(&measured_observations, |observation| observation.prefill_s);
    let kv_transfer = collect_metric(&measured_observations, |observation| {
        observation.kv_transfer_s
    });
    let kv_queue = collect_metric(&measured_observations, |observation| observation.kv_queue_s);
    let kv_worker_queue = collect_metric(&measured_observations, |observation| {
        observation.kv_worker_queue_s
    });
    let kv_resource_queue = collect_metric(&measured_observations, |observation| {
        observation.kv_resource_queue_s
    });
    let decode_queue = collect_metric(&measured_observations, |observation| {
        observation.decode_queue_s
    });
    let prefill_worker_queue = collect_metric(&measured_observations, |observation| {
        observation.prefill_worker_queue_s
    });
    let prefill_resource_queue = collect_metric(&measured_observations, |observation| {
        observation.prefill_resource_queue_s
    });
    let decode_worker_queue = collect_metric(&measured_observations, |observation| {
        observation.decode_worker_queue_s
    });
    let decode_resource_queue = collect_metric(&measured_observations, |observation| {
        observation.decode_resource_queue_s
    });
    let decode = collect_metric(&measured_observations, |observation| observation.decode_s);
    let queue_delay = collect_metric(&measured_observations, |observation| {
        observation.queue_delay_s
    });
    let capacity = capacity_profile(&decode_states, traffic);
    let admitted_requests = decode_states
        .iter()
        .filter(|state| state.status.is_admitted())
        .count()
        .min(u32::MAX as usize) as u32;
    let completed_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::Completed)
        .count()
        .min(u32::MAX as usize) as u32;
    let rejected_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::RejectedAdmission)
        .count()
        .min(u32::MAX as usize) as u32;
    let timed_out_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::TimedOut)
        .count()
        .min(u32::MAX as usize) as u32;
    let cancelled_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::Cancelled)
        .count()
        .min(u32::MAX as usize) as u32;
    let deadline_constrained_requests = measured_all_observations
        .iter()
        .filter(|observation| observation.deadline_s.is_some())
        .count();
    let deadline_missed_requests = measured_all_observations
        .iter()
        .filter(|observation| observation.deadline_missed)
        .count();
    let deadline_miss_rate = if deadline_constrained_requests > 0 {
        deadline_missed_requests as f64 / deadline_constrained_requests as f64
    } else {
        0.0
    };
    let ttft_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.ttft_s,
        |observation| observation.ttft_s,
    );
    let tpot_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.tpot_s,
        |observation| observation.tpot_s,
    );
    let itl_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.itl_s,
        |observation| observation.itl_s,
    );
    let e2el_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.e2el_s,
        |observation| observation.e2el_s,
    );

    let mut metrics = ServingMetrics {
        ttft_s: mean(&ttft),
        ttft_p50_s: percentile(ttft.clone(), 0.50),
        ttft_p90_s: percentile(ttft.clone(), 0.90),
        ttft_p95_s: percentile(ttft.clone(), 0.95),
        ttft_p99_s: percentile(ttft.clone(), 0.99),
        ttft_max_s: max_value(&ttft),
        ttft_slo_constrained_requests: ttft_slo_counts.constrained,
        ttft_slo_missed_requests: ttft_slo_counts.missed,
        ttft_slo_miss_rate: ttft_slo_counts.miss_rate,
        tpot_s: mean(&tpot),
        tpot_p50_s: percentile(tpot.clone(), 0.50),
        tpot_p90_s: percentile(tpot.clone(), 0.90),
        tpot_p95_s: percentile(tpot.clone(), 0.95),
        tpot_p99_s: percentile(tpot.clone(), 0.99),
        tpot_max_s: max_value(&tpot),
        tpot_slo_constrained_requests: tpot_slo_counts.constrained,
        tpot_slo_missed_requests: tpot_slo_counts.missed,
        tpot_slo_miss_rate: tpot_slo_counts.miss_rate,
        itl_s: mean(&itl),
        itl_p50_s: percentile(itl.clone(), 0.50),
        itl_p90_s: percentile(itl.clone(), 0.90),
        itl_p95_s: percentile(itl.clone(), 0.95),
        itl_p99_s: percentile(itl.clone(), 0.99),
        itl_max_s: max_value(&itl),
        itl_slo_constrained_requests: itl_slo_counts.constrained,
        itl_slo_missed_requests: itl_slo_counts.missed,
        itl_slo_miss_rate: itl_slo_counts.miss_rate,
        decode_iterations: measured_decode_iteration_latencies
            .len()
            .min(u64::MAX as usize) as u64,
        decode_iteration_s: mean(&measured_decode_iteration_latencies),
        decode_iteration_p50_s: percentile(measured_decode_iteration_latencies.clone(), 0.50),
        decode_iteration_p90_s: percentile(measured_decode_iteration_latencies.clone(), 0.90),
        decode_iteration_p95_s: percentile(measured_decode_iteration_latencies.clone(), 0.95),
        decode_iteration_p99_s: percentile(measured_decode_iteration_latencies.clone(), 0.99),
        decode_iteration_max_s: max_value(&measured_decode_iteration_latencies),
        throughput_tokens_per_s,
        e2el_s: mean(&e2el),
        e2el_p50_s: percentile(e2el.clone(), 0.50),
        e2el_p90_s: percentile(e2el.clone(), 0.90),
        e2el_p95_s: percentile(e2el.clone(), 0.95),
        e2el_p99_s: percentile(e2el.clone(), 0.99),
        e2el_max_s: max_value(&e2el),
        e2el_slo_constrained_requests: e2el_slo_counts.constrained,
        e2el_slo_missed_requests: e2el_slo_counts.missed,
        e2el_slo_miss_rate: e2el_slo_counts.miss_rate,
        deadline_miss_rate,
        service_s: mean(&service),
        prefill_s: mean(&prefill),
        prefill_chunks: measured_prefill_chunks,
        prompt_tokens: measured_prompt_tokens,
        prefix_cache_hit_tokens: measured_prefix_cache_hit_tokens,
        effective_prefill_tokens: measured_effective_prefill_tokens,
        prefix_cache_hit_rate: measured_prefix_cache_hit_rate,
        kv_transfer_s: mean(&kv_transfer),
        kv_queue_s: mean(&kv_queue),
        kv_worker_queue_s: mean(&kv_worker_queue),
        kv_resource_queue_s: mean(&kv_resource_queue),
        decode_queue_s: mean(&decode_queue),
        prefill_worker_queue_s: mean(&prefill_worker_queue),
        prefill_resource_queue_s: mean(&prefill_resource_queue),
        decode_worker_queue_s: mean(&decode_worker_queue),
        decode_resource_queue_s: mean(&decode_resource_queue),
        decode_s: mean(&decode),
        queue_delay_s: mean(&queue_delay),
        queue_delay_p90_s: percentile(queue_delay.clone(), 0.90),
        queue_delay_p95_s: percentile(queue_delay.clone(), 0.95),
        queue_delay_max_s: max_value(&queue_delay),
        peak_prefill_tokens: capacity.peak_prefill_tokens,
        peak_prefill_tokens_per_node: capacity.peak_prefill_tokens_per_node,
        peak_prefill_tokens_per_gpu: capacity.peak_prefill_tokens_per_gpu,
        peak_decode_sequences: capacity.peak_decode_sequences,
        peak_resident_tokens: capacity.peak_resident_tokens,
        peak_decode_sequences_per_node: capacity.peak_decode_sequences_per_node,
        peak_resident_tokens_per_node: capacity.peak_resident_tokens_per_node,
        peak_decode_sequences_per_gpu: capacity.peak_decode_sequences_per_gpu,
        peak_resident_tokens_per_gpu: capacity.peak_resident_tokens_per_gpu,
        peak_kv_blocks: capacity.peak_kv_blocks,
        peak_allocated_kv_tokens: capacity.peak_allocated_kv_tokens,
        peak_kv_fragmentation_tokens: capacity.peak_kv_fragmentation_tokens,
        peak_kv_block_table_bytes: capacity.peak_kv_block_table_bytes,
        peak_kv_blocks_per_node: capacity.peak_kv_blocks_per_node,
        peak_allocated_kv_tokens_per_node: capacity.peak_allocated_kv_tokens_per_node,
        peak_kv_fragmentation_tokens_per_node: capacity.peak_kv_fragmentation_tokens_per_node,
        peak_kv_block_table_bytes_per_node: capacity.peak_kv_block_table_bytes_per_node,
        peak_kv_blocks_per_gpu: capacity.peak_kv_blocks_per_gpu,
        peak_allocated_kv_tokens_per_gpu: capacity.peak_allocated_kv_tokens_per_gpu,
        peak_kv_fragmentation_tokens_per_gpu: capacity.peak_kv_fragmentation_tokens_per_gpu,
        peak_kv_block_table_bytes_per_gpu: capacity.peak_kv_block_table_bytes_per_gpu,
        decode_sequence_utilization: capacity.decode_sequence_utilization,
        resident_token_utilization: capacity.resident_token_utilization,
        kv_block_utilization: capacity.kv_block_utilization,
        decode_sequence_per_node_utilization: capacity.decode_sequence_per_node_utilization,
        resident_token_per_node_utilization: capacity.resident_token_per_node_utilization,
        kv_block_per_node_utilization: capacity.kv_block_per_node_utilization,
        decode_sequence_per_gpu_utilization: capacity.decode_sequence_per_gpu_utilization,
        resident_token_per_gpu_utilization: capacity.resident_token_per_gpu_utilization,
        kv_block_per_gpu_utilization: capacity.kv_block_per_gpu_utilization,
        scheduled_makespan_s: makespan_s,
        scheduled_requests,
        admitted_requests,
        completed_requests,
        rejected_requests,
        timed_out_requests,
        cancelled_requests,
        deadline_constrained_requests: deadline_constrained_requests.min(u32::MAX as usize) as u32,
        deadline_missed_requests: deadline_missed_requests.min(u32::MAX as usize) as u32,
        measured_requests: measured_observations.len().min(u32::MAX as usize) as u32,
        measurement_start_s,
        measurement_end_s,
    };
    calibration_fits.extend(apply_serving_metric_calibration_fits(
        &mut metrics,
        calibration_profile,
        model,
        request,
        traffic,
        prefill_score.config,
        decode_score.config,
    ));
    let metric_breakdowns =
        metric_breakdowns(&observations, measurement_start_s, measurement_end_s);
    let worker_observations = worker_observations(&observations, &capacity.gpus, traffic);
    let service_observations = service_observations(
        traffic,
        prefill_nodes,
        decode_nodes,
        prefill_score,
        decode_score,
        &observations,
        &worker_observations,
    );
    let scheduled_operations = scheduler.operations().to_vec();
    let resource_utilization = resource_utilization(&scheduled_operations, makespan_s);
    let phase_resource_utilization = phase_resource_utilization(&scheduled_operations, makespan_s);

    ServingSimulation {
        metrics,
        calibration_fits,
        measurement_window,
        metric_breakdowns,
        request_observations: observations,
        decode_iterations,
        node_capacity: capacity.nodes,
        gpu_capacity: capacity.gpus,
        traffic_class_capacity: capacity.traffic_classes,
        service_observations,
        worker_observations,
        scheduled_operations,
        resource_utilization,
        phase_resource_utilization,
        kv_bottlenecks,
    }
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
