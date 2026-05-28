use super::*;

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

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PrefillTokenSpan {
    pub(super) start_s: f64,
    pub(super) finish_s: f64,
    pub(super) tokens: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PrefillAdmissionSpan {
    pub(super) finish_s: f64,
    pub(super) traffic_class: Option<String>,
    pub(super) tokens: u64,
    pub(super) node_tokens: Vec<(NodeId, u64)>,
    pub(super) gpu_tokens: Vec<(GpuAddr, u64)>,
}

impl PrefillAdmissionSpan {
    pub(super) fn for_state_chunk(
        state: &DecodeRequestState,
        chunk_tokens: Option<u32>,
        finish_s: f64,
    ) -> Self {
        let tokens = prefill_work_tokens(state, chunk_tokens);
        Self::for_state_tokens(state, tokens, finish_s)
    }

    pub(super) fn for_state_tokens(state: &DecodeRequestState, tokens: u64, finish_s: f64) -> Self {
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
pub(super) struct DecodeRequestState {
    pub(super) request_idx: u32,
    pub(super) request_id: Option<String>,
    pub(super) tenant: Option<String>,
    pub(super) model_id: Option<String>,
    pub(super) traffic_class: Option<String>,
    pub(super) shape_profile: Option<String>,
    pub(super) cache_key: Option<String>,
    pub(super) prefill_node: NodeId,
    pub(super) decode_node: NodeId,
    pub(super) prefill_route_nodes: Vec<NodeId>,
    pub(super) prefill_route_gpus: Vec<GpuAddr>,
    pub(super) decode_route_nodes: Vec<NodeId>,
    pub(super) decode_route_gpus: Vec<GpuAddr>,
    pub(super) routing_policy: ServingRoutingPolicy,
    pub(super) routing_candidate_count: u32,
    pub(super) routing_routable_candidate_count: u32,
    pub(super) routing_estimated_e2el_s: f64,
    pub(super) routing_estimated_kv_transfer_s: f64,
    pub(super) routing_estimated_kv_resource_wait_s: f64,
    pub(super) routing_estimated_prefill_wait_s: f64,
    pub(super) routing_estimated_decode_wait_s: f64,
    pub(super) routing_reason: String,
    pub(super) routing_candidates: Vec<ServingRouteCandidateObservation>,
    pub(super) arrival_s: f64,
    pub(super) priority: i32,
    pub(super) batch_size: u32,
    pub(super) prompt_tokens: u32,
    pub(super) prefix_cache_hit_tokens: u32,
    pub(super) effective_prefill_tokens: u32,
    pub(super) remaining_prefill_tokens: u32,
    pub(super) prefill_chunks: u32,
    pub(super) decode_tokens: u32,
    pub(super) slo: ServingRequestSlo,
    pub(super) max_queue_delay_s: Option<f64>,
    pub(super) max_kv_queue_delay_s: Option<f64>,
    pub(super) max_decode_queue_delay_s: Option<f64>,
    pub(super) max_decode_iteration_queue_delay_s: Option<f64>,
    pub(super) request_timeout_s: Option<f64>,
    pub(super) deadline_s: Option<f64>,
    pub(super) cancellation_s: Option<f64>,
    pub(super) remaining_tokens: u32,
    pub(super) emitted_tokens: u32,
    pub(super) max_sequence_tokens: u32,
    pub(super) kv_block_tokens: u32,
    pub(super) kv_cache_blocks: u64,
    pub(super) kv_allocated_tokens: u64,
    pub(super) kv_fragmentation_tokens: u64,
    pub(super) prefill_scheduled: bool,
    pub(super) dependencies: Vec<usize>,
    pub(super) kv_start_s: f64,
    pub(super) kv_finish_s: f64,
    pub(super) first_decode_start_s: Option<f64>,
    pub(super) first_decode_finish_s: Option<f64>,
    pub(super) last_decode_finish_s: Option<f64>,
    pub(super) decode_token_start_s: Vec<f64>,
    pub(super) decode_token_finish_s: Vec<f64>,
    pub(super) prefill_start_s: f64,
    pub(super) prefill_finish_s: f64,
    pub(super) prefill_worker_queue_s: f64,
    pub(super) prefill_resource_queue_s: f64,
    pub(super) decode_worker_queue_s: f64,
    pub(super) decode_resource_queue_s: f64,
    pub(super) kv_transfer_bytes: u64,
    pub(super) kv_transfer_bottlenecks: Vec<String>,
    pub(super) kv_transfer_paths: Vec<ServingKvTransferPathObservation>,
    pub(super) kv_transfer_resources: Vec<String>,
    pub(super) kv_transfer_resource_dependencies: Vec<usize>,
    pub(super) kv_worker_queue_s: f64,
    pub(super) kv_resource_queue_s: f64,
    pub(super) kv_transfer_s: f64,
    pub(super) kv_transfer_fit: Option<CalibrationFitApplication>,
    pub(super) prefill_token_spans: Vec<PrefillTokenSpan>,
    pub(super) worker_assignments: Vec<ServingWorkerAssignmentObservation>,
    pub(super) status: ServingRequestStatus,
    pub(super) status_time_s: Option<f64>,
    pub(super) failure_reason: Option<String>,
    pub(super) failure_rejection: Option<ServingRejection>,
}

pub(super) fn compare_request_priority(
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
pub(super) fn schedule_prefills(
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

pub(super) fn prefill_worker_slots_per_gpu(traffic: &ServingTraffic) -> usize {
    scaled_worker_slots(
        traffic.max_prefill_worker_slots_per_gpu.unwrap_or(1),
        traffic.services.prefill,
    )
}

pub(super) fn decode_worker_slots_per_gpu(traffic: &ServingTraffic) -> usize {
    scaled_worker_slots(
        traffic.max_decode_worker_slots_per_gpu.unwrap_or(1),
        traffic.services.decode,
    )
}

pub(super) fn kv_transfer_worker_slots_per_gpu(traffic: &ServingTraffic) -> Option<usize> {
    traffic
        .max_kv_transfer_worker_slots_per_gpu
        .map(|slots| scaled_worker_slots(slots, traffic.services.kv_transfer))
}

pub(super) fn scaled_worker_slots(
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
pub(super) fn schedule_independent_prefills(
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
pub(super) fn schedule_continuous_prefills(
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
pub(super) fn schedule_prefill_batch(
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

pub(super) fn prefill_ready_s(state: &DecodeRequestState, scheduler: &ResourceScheduler) -> f64 {
    if state.prefill_chunks == 0 {
        state.arrival_s
    } else {
        operation_finish_s(scheduler, &state.dependencies)
    }
}

pub(super) fn prefill_candidate_ready_s(
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

pub(super) fn prefill_worker_ready_s(
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

pub(super) fn prefill_worker_gpu_ready_s(
    ready_s: &BTreeMap<GpuAddr, Vec<f64>>,
    gpu: GpuAddr,
    prefill_worker_slots_per_gpu: usize,
) -> f64 {
    worker_slot_ready_s(ready_s, gpu, prefill_worker_slots_per_gpu)
}

pub(super) fn decode_candidate_ready_s(
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

pub(super) fn decode_worker_ready_s(
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

pub(super) fn kv_transfer_worker_ready_s(
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

pub(super) fn record_prefill_queue_breakdown(
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

pub(super) fn record_decode_queue_breakdown(
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

pub(super) fn assign_selected_prefill_workers_ready(
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

pub(super) fn assign_decode_workers_ready(
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

pub(super) fn assign_selected_decode_workers_ready(
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

pub(super) fn assign_kv_transfer_workers_ready(
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

pub(super) fn kv_transfer_worker_gpus(state: &DecodeRequestState) -> Vec<GpuAddr> {
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

pub(super) fn prefill_work_tokens(state: &DecodeRequestState, chunk_tokens: Option<u32>) -> u64 {
    u64::from(state.batch_size.max(1)) * u64::from(state_prefill_chunk_tokens(state, chunk_tokens))
}

pub(super) fn state_prefill_chunk_tokens(
    state: &DecodeRequestState,
    chunk_tokens: Option<u32>,
) -> u32 {
    chunk_tokens
        .map(|chunk_tokens| state.remaining_prefill_tokens.min(chunk_tokens.max(1)))
        .unwrap_or(state.remaining_prefill_tokens)
}

pub(super) fn request_level_prefill_capacity_enabled(traffic: &ServingTraffic) -> bool {
    traffic.decode_capacity_policy == ServingDecodeCapacityPolicy::RequestReject
        && (traffic.max_prefill_tokens.is_some()
            || traffic.max_prefill_tokens_per_node.is_some()
            || traffic.max_prefill_tokens_per_gpu.is_some()
            || traffic
                .traffic_classes
                .iter()
                .any(|class| class.max_prefill_tokens.is_some()))
}

pub(super) fn retire_prefill_admissions(active: &mut Vec<PrefillAdmissionSpan>, now_s: f64) {
    active.retain(|span| span.finish_s > now_s + 1e-12);
}

pub(super) fn prefill_capacity_admission_rejection(
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

pub(super) fn active_prefill_class_tokens(
    active: &[PrefillAdmissionSpan],
    class_name: &str,
) -> u64 {
    active
        .iter()
        .filter(|span| span.traffic_class.as_deref() == Some(class_name))
        .map(|span| span.tokens)
        .sum()
}

pub(super) fn active_prefill_node_tokens(active: &[PrefillAdmissionSpan], node_id: NodeId) -> u64 {
    active
        .iter()
        .flat_map(|span| &span.node_tokens)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, tokens)| *tokens)
        .sum()
}

pub(super) fn active_prefill_gpu_tokens(active: &[PrefillAdmissionSpan], gpu: GpuAddr) -> u64 {
    active
        .iter()
        .flat_map(|span| &span.gpu_tokens)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, tokens)| *tokens)
        .sum()
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

pub(super) fn apply_decode_capacity_admission(
    states: &mut [DecodeRequestState],
    traffic: &ServingTraffic,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    decode_worker_slots_per_gpu: usize,
) {
    if traffic.decode_capacity_policy != ServingDecodeCapacityPolicy::RequestReject {
        return;
    }

    let mut order = (0..states.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        states[*left]
            .kv_finish_s
            .total_cmp(&states[*right].kv_finish_s)
            .then_with(|| compare_request_priority(states, *left, *right))
    });
    let mut active = Vec::new();
    let mut decode_worker_ready_s = BTreeMap::new();

    for idx in order {
        if states[idx].status != ServingRequestStatus::Pending || states[idx].remaining_tokens == 0
        {
            continue;
        }
        let admission_s = states[idx].kv_finish_s;
        if !admission_s.is_finite() {
            reject_decode_capacity_admission(
                &mut states[idx],
                with_rejection_details(
                    request_rejection(
                        "decode",
                        "capacity",
                        "kv_residency",
                        "decode_admission_kv_not_ready",
                        "decode admission rejected: request is not ready for finite KV residency"
                            .to_string(),
                    ),
                    None,
                    None,
                    None,
                    Some("ensure prefill and KV transfer complete before decode admission"),
                ),
            );
            continue;
        }

        retire_decode_residency(&mut active, admission_s);
        let decode_start_s = admission_s.max(route_workers_ready_s(
            &decode_worker_ready_s,
            &states[idx].decode_route_gpus,
            states[idx].decode_node,
            decode_worker_slots_per_gpu,
        ));
        let finish_s = decode_start_s
            + decode_service_estimate_s(
                &states[idx],
                decode_one_score,
                decode_tail_scale,
                base_batch_size,
            );
        let candidate = DecodeAdmissionResidency::for_state(&states[idx], finish_s);
        if let Some(rejection) = decode_capacity_admission_rejection(&active, &candidate, traffic) {
            reject_decode_capacity_admission(&mut states[idx], rejection);
            continue;
        }

        mark_route_workers_ready(
            &mut decode_worker_ready_s,
            &states[idx].decode_route_gpus,
            states[idx].decode_node,
            decode_worker_slots_per_gpu,
            finish_s,
        );
        active.push(candidate);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DecodeAdmissionResidency {
    pub(super) finish_s: f64,
    pub(super) traffic_class: Option<String>,
    pub(super) sequences: u32,
    pub(super) resident_tokens: u64,
    pub(super) kv_blocks: u64,
    pub(super) node_sequences: Vec<(NodeId, u32)>,
    pub(super) node_tokens: Vec<(NodeId, u64)>,
    pub(super) node_blocks: Vec<(NodeId, u64)>,
    pub(super) gpu_sequences: Vec<(GpuAddr, u32)>,
    pub(super) gpu_tokens: Vec<(GpuAddr, u64)>,
    pub(super) gpu_blocks: Vec<(GpuAddr, u64)>,
}

impl DecodeAdmissionResidency {
    pub(super) fn for_state(state: &DecodeRequestState, finish_s: f64) -> Self {
        let sequences = state.batch_size.max(1);
        let resident_tokens =
            u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1));
        let allocation = sequence_kv_allocation(
            state.batch_size.max(1),
            state.max_sequence_tokens.max(1),
            state.kv_block_tokens,
        );
        let decode_route_nodes = if state.decode_route_nodes.is_empty() {
            vec![state.decode_node]
        } else {
            state.decode_route_nodes.clone()
        };
        let per_node_tokens = resident_tokens.div_ceil(decode_route_nodes.len().max(1) as u64);
        let per_node_blocks = allocation
            .blocks
            .div_ceil(decode_route_nodes.len().max(1) as u64);
        let node_sequences = decode_route_nodes
            .iter()
            .map(|node_id| (*node_id, sequences))
            .collect::<Vec<_>>();
        let node_tokens = decode_route_nodes
            .iter()
            .map(|node_id| (*node_id, per_node_tokens))
            .collect::<Vec<_>>();
        let node_blocks = decode_route_nodes
            .into_iter()
            .map(|node_id| (node_id, per_node_blocks))
            .collect::<Vec<_>>();
        let decode_route_gpus = if state.decode_route_gpus.is_empty() {
            vec![GpuAddr {
                node_id: state.decode_node,
                local_gpu_id: 0,
            }]
        } else {
            state.decode_route_gpus.clone()
        };
        let per_gpu_tokens = resident_tokens.div_ceil(decode_route_gpus.len().max(1) as u64);
        let per_gpu_blocks = allocation
            .blocks
            .div_ceil(decode_route_gpus.len().max(1) as u64);
        let gpu_sequences = decode_route_gpus
            .iter()
            .map(|gpu| (*gpu, sequences))
            .collect::<Vec<_>>();
        let gpu_tokens = decode_route_gpus
            .iter()
            .map(|gpu| (*gpu, per_gpu_tokens))
            .collect::<Vec<_>>();
        let gpu_blocks = decode_route_gpus
            .into_iter()
            .map(|gpu| (gpu, per_gpu_blocks))
            .collect();

        Self {
            finish_s,
            traffic_class: state.traffic_class.clone(),
            sequences,
            resident_tokens,
            kv_blocks: allocation.blocks,
            node_sequences,
            node_tokens,
            node_blocks,
            gpu_sequences,
            gpu_tokens,
            gpu_blocks,
        }
    }
}

pub(super) fn retire_decode_residency(active: &mut Vec<DecodeAdmissionResidency>, now_s: f64) {
    active.retain(|residency| residency.finish_s > now_s + 1e-12);
}

pub(super) fn decode_service_estimate_s(
    state: &DecodeRequestState,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
) -> f64 {
    let first_token_scale = request_token_scale(state.batch_size, 1, base_batch_size, 1);
    let decode_iterations =
        1.0 + f64::from(state.decode_tokens.saturating_sub(1)) * decode_tail_scale;
    finite_or_zero(decode_one_score.estimated_latency_s) * first_token_scale * decode_iterations
}

pub(super) fn decode_capacity_admission_request_rejection(
    resource: String,
    code: &str,
    message: String,
    observed: f64,
    limit: f64,
    unit: &str,
    remediation: &str,
) -> ServingRejection {
    with_rejection_details(
        request_rejection("decode", "capacity", &resource, code, message),
        Some(observed),
        Some(limit),
        Some(unit),
        Some(remediation),
    )
}

pub(super) fn decode_capacity_admission_rejection(
    active: &[DecodeAdmissionResidency],
    candidate: &DecodeAdmissionResidency,
    traffic: &ServingTraffic,
) -> Option<ServingRejection> {
    let active_sequences = active
        .iter()
        .map(|residency| residency.sequences)
        .sum::<u32>();
    if let Some(limit) = traffic.max_decode_sequences {
        let observed = active_sequences.saturating_add(candidate.sequences);
        if observed > limit {
            let message = format!(
                "decode admission rejected: active decode sequences {observed} > max_decode_sequences {limit}"
            );
            return Some(decode_capacity_admission_request_rejection(
                "decode_sequences".to_string(),
                "decode_capacity_exceeded",
                message,
                f64::from(observed),
                f64::from(limit),
                "sequences",
                "increase max_decode_sequences, add decode capacity, or reduce concurrency",
            ));
        }
    }

    let active_tokens = active
        .iter()
        .map(|residency| residency.resident_tokens)
        .sum::<u64>();
    if let Some(limit) = traffic.max_resident_tokens {
        let observed = active_tokens.saturating_add(candidate.resident_tokens);
        if observed > limit {
            let message = format!(
                "decode admission rejected: resident KV tokens {observed} > max_resident_tokens {limit}"
            );
            return Some(decode_capacity_admission_request_rejection(
                "resident_tokens".to_string(),
                "kv_residency_capacity_exceeded",
                message,
                observed as f64,
                limit as f64,
                "tokens",
                "increase max_resident_tokens, reduce sequence length, or add decode capacity",
            ));
        }
    }

    if let Some(limit) = traffic.max_kv_blocks {
        let active_blocks = active
            .iter()
            .map(|residency| residency.kv_blocks)
            .sum::<u64>();
        let observed = active_blocks.saturating_add(candidate.kv_blocks);
        if observed > limit {
            let message = format!(
                "decode admission rejected: resident KV blocks {observed} > max_kv_blocks {limit}"
            );
            return Some(decode_capacity_admission_request_rejection(
                "kv_blocks".to_string(),
                "kv_block_capacity_exceeded",
                message,
                observed as f64,
                limit as f64,
                "blocks",
                "increase max_kv_blocks, increase KV block capacity, or reduce sequence length",
            ));
        }
    }

    if let Some(class) = serving_traffic_class(traffic, candidate.traffic_class.as_deref()) {
        if let Some(limit) = class.max_decode_sequences {
            let observed =
                active_class_sequences(active, &class.name).saturating_add(candidate.sequences);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: traffic class {} active decode sequences {} > max_decode_sequences {}",
                    class.name, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("traffic_class {} decode_sequences", class.name),
                    "traffic_class_decode_capacity_exceeded",
                    message,
                    f64::from(observed),
                    f64::from(limit),
                    "sequences",
                    "increase the traffic class max_decode_sequences, add class capacity, or reduce class concurrency",
                ));
            }
        }

        if let Some(limit) = class.max_resident_tokens {
            let observed =
                active_class_tokens(active, &class.name).saturating_add(candidate.resident_tokens);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: traffic class {} resident KV tokens {} > max_resident_tokens {}",
                    class.name, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("traffic_class {} resident_tokens", class.name),
                    "traffic_class_kv_residency_capacity_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "tokens",
                    "increase the traffic class max_resident_tokens, reduce class sequence length, or add class capacity",
                ));
            }
        }

        if let Some(limit) = class.max_kv_blocks {
            let observed =
                active_class_blocks(active, &class.name).saturating_add(candidate.kv_blocks);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: traffic class {} resident KV blocks {} > max_kv_blocks {}",
                    class.name, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("traffic_class {} kv_blocks", class.name),
                    "traffic_class_kv_block_capacity_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "blocks",
                    "increase the traffic class max_kv_blocks, reduce class sequence length, or add class capacity",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_decode_sequences_per_node {
        for (node_id, sequences) in &candidate.node_sequences {
            let observed = active_node_sequences(active, *node_id).saturating_add(*sequences);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {node_id} active decode sequences {observed} > max_decode_sequences_per_node {limit}"
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("node {node_id} decode_sequences_per_node"),
                    "decode_capacity_per_node_exceeded",
                    message,
                    f64::from(observed),
                    f64::from(limit),
                    "sequences",
                    "increase max_decode_sequences_per_node, add decode nodes, or rebalance routing",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_resident_tokens_per_node {
        for (node_id, tokens) in &candidate.node_tokens {
            let observed = active_node_tokens(active, *node_id).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {node_id} resident KV tokens {observed} > max_resident_tokens_per_node {limit}"
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("node {node_id} resident_tokens_per_node"),
                    "kv_residency_capacity_per_node_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "tokens",
                    "increase max_resident_tokens_per_node, add decode nodes, or reduce sequence length",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_kv_blocks_per_node {
        for (node_id, blocks) in &candidate.node_blocks {
            let observed = active_node_blocks(active, *node_id).saturating_add(*blocks);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {node_id} resident KV blocks {observed} > max_kv_blocks_per_node {limit}"
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("node {node_id} kv_blocks_per_node"),
                    "kv_block_capacity_per_node_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "blocks",
                    "increase max_kv_blocks_per_node, add decode nodes, or reduce sequence length",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_decode_sequences_per_gpu {
        for (gpu, sequences) in &candidate.gpu_sequences {
            let observed = active_gpu_sequences(active, *gpu).saturating_add(*sequences);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {} gpu {} active decode sequences {} > max_decode_sequences_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!(
                        "node {} gpu {} decode_sequences_per_gpu",
                        gpu.node_id, gpu.local_gpu_id
                    ),
                    "decode_capacity_per_gpu_exceeded",
                    message,
                    f64::from(observed),
                    f64::from(limit),
                    "sequences",
                    "increase max_decode_sequences_per_gpu, add decode GPUs, or rebalance routing",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_resident_tokens_per_gpu {
        for (gpu, tokens) in &candidate.gpu_tokens {
            let observed = active_gpu_tokens(active, *gpu).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {} gpu {} resident KV tokens {} > max_resident_tokens_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!(
                        "node {} gpu {} resident_tokens_per_gpu",
                        gpu.node_id, gpu.local_gpu_id
                    ),
                    "kv_residency_capacity_per_gpu_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "tokens",
                    "increase max_resident_tokens_per_gpu, add decode GPUs, or reduce sequence length",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_kv_blocks_per_gpu {
        for (gpu, blocks) in &candidate.gpu_blocks {
            let observed = active_gpu_blocks(active, *gpu).saturating_add(*blocks);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {} gpu {} resident KV blocks {} > max_kv_blocks_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!(
                        "node {} gpu {} kv_blocks_per_gpu",
                        gpu.node_id, gpu.local_gpu_id
                    ),
                    "kv_block_capacity_per_gpu_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "blocks",
                    "increase max_kv_blocks_per_gpu, add decode GPUs, or reduce sequence length",
                ));
            }
        }
    }

    None
}

pub(super) fn serving_traffic_class<'a>(
    traffic: &'a ServingTraffic,
    class_name: Option<&str>,
) -> Option<&'a ServingTrafficClass> {
    let class_name = class_name?;
    traffic
        .traffic_classes
        .iter()
        .find(|class| class.name == class_name)
}

pub(super) fn active_class_sequences(active: &[DecodeAdmissionResidency], class_name: &str) -> u32 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.sequences)
        .sum()
}

pub(super) fn active_class_tokens(active: &[DecodeAdmissionResidency], class_name: &str) -> u64 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.resident_tokens)
        .sum()
}

pub(super) fn active_class_blocks(active: &[DecodeAdmissionResidency], class_name: &str) -> u64 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.kv_blocks)
        .sum()
}

pub(super) fn active_node_sequences(active: &[DecodeAdmissionResidency], node_id: NodeId) -> u32 {
    active
        .iter()
        .flat_map(|residency| &residency.node_sequences)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, sequences)| *sequences)
        .sum()
}

pub(super) fn active_node_tokens(active: &[DecodeAdmissionResidency], node_id: NodeId) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.node_tokens)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, tokens)| *tokens)
        .sum()
}

pub(super) fn active_node_blocks(active: &[DecodeAdmissionResidency], node_id: NodeId) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.node_blocks)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, blocks)| *blocks)
        .sum()
}

pub(super) fn active_gpu_sequences(active: &[DecodeAdmissionResidency], gpu: GpuAddr) -> u32 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_sequences)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, sequences)| *sequences)
        .sum()
}

pub(super) fn active_gpu_tokens(active: &[DecodeAdmissionResidency], gpu: GpuAddr) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_tokens)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, tokens)| *tokens)
        .sum()
}

pub(super) fn active_gpu_blocks(active: &[DecodeAdmissionResidency], gpu: GpuAddr) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_blocks)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, blocks)| *blocks)
        .sum()
}

pub(super) fn reject_decode_capacity_admission(
    state: &mut DecodeRequestState,
    rejection: ServingRejection,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if state.kv_finish_s.is_finite() {
        state.kv_finish_s
    } else {
        state.arrival_s
    });
    set_request_rejection(state, rejection);
    state.dependencies.clear();
}

pub(super) fn decode_queue_delay_rejection(
    state: &DecodeRequestState,
    decode_start_s: f64,
) -> Option<(f64, f64)> {
    if state.emitted_tokens != 0 {
        return None;
    }
    let max_decode_queue_delay_s = state.max_decode_queue_delay_s?;
    let queue_delay_s = finite_or_zero(decode_start_s - state.kv_finish_s);
    (queue_delay_s > max_decode_queue_delay_s + 1e-12)
        .then_some((queue_delay_s, max_decode_queue_delay_s))
}

pub(super) fn decode_iteration_queue_timeout(
    state: &DecodeRequestState,
    decode_start_s: f64,
) -> Option<(f64, f64)> {
    if state.emitted_tokens == 0 {
        return None;
    }
    let max_queue_delay_s = state.max_decode_iteration_queue_delay_s?;
    let phase_ready_s = state.last_decode_finish_s.unwrap_or(state.kv_finish_s);
    let queue_delay_s = finite_or_zero(decode_start_s - phase_ready_s);
    (queue_delay_s > max_queue_delay_s + 1e-12).then_some((queue_delay_s, max_queue_delay_s))
}

pub(super) fn reject_decode_queue_admission(
    state: &mut DecodeRequestState,
    decode_start_s: f64,
    queue_delay_s: f64,
    max_decode_queue_delay_s: f64,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if state.kv_finish_s.is_finite() {
        state.kv_finish_s + max_decode_queue_delay_s
    } else {
        state.arrival_s
    });
    let message = format!(
        "admission rejected: decode queue delay {queue_delay_s:.6}s > max_decode_queue_delay {max_decode_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "decode",
                "queueing",
                "decode_queue",
                "decode_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_decode_queue_delay_s),
            Some("s"),
            Some("increase max_decode_queue_delay, add decode workers, or reduce decode load"),
        ),
    );
    state.first_decode_start_s = Some(decode_start_s);
    state.dependencies.clear();
}

pub(super) fn timeout_decode_iteration_queue(
    state: &mut DecodeRequestState,
    queue_delay_s: f64,
    max_queue_delay_s: f64,
) {
    let phase_ready_s = state.last_decode_finish_s.unwrap_or(state.kv_finish_s);
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::TimedOut;
    state.status_time_s = Some(if phase_ready_s.is_finite() {
        phase_ready_s + max_queue_delay_s
    } else {
        state.arrival_s
    });
    let message = format!(
        "request timed out: decode iteration queue delay {queue_delay_s:.6}s > max_decode_iteration_queue_delay {max_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "decode",
                "queueing",
                "decode_iteration_queue",
                "decode_iteration_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_queue_delay_s),
            Some("s"),
            Some(
                "increase max_decode_iteration_queue_delay, add decode workers, or reduce decode load",
            ),
        ),
    );
    state.dependencies.clear();
}

#[allow(clippy::too_many_arguments)]
pub(super) fn schedule_independent_decodes(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    decode_iterations: &mut Vec<ServingDecodeIterationObservation>,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    decode_worker_slots_per_gpu: usize,
) {
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let mut order = (0..states.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| compare_request_priority(states, *left, *right));
    for idx in order {
        let state = &mut states[idx];
        while state.remaining_tokens > 0 {
            let ready_s = decode_candidate_ready_s(
                state,
                scheduler,
                worker_runtime,
                decode_worker_slots_per_gpu,
            );
            if let Some(cancellation_s) = state.cancellation_s
                && cancellation_s <= ready_s + 1e-12
            {
                cancel_request(
                    state,
                    cancellation_s,
                    "request cancelled before decode iteration",
                );
                break;
            }
            let mut token_ops = decode_one_score.operations.clone();
            let mut scale = request_token_scale(state.batch_size, 1, base_batch_size, 1);
            if state.emitted_tokens > 0 {
                scale *= decode_tail_scale;
            }
            scale_operations(&mut token_ops, scale);
            remap_operations_to_node(&mut token_ops, decode_resource_base_node, state.decode_node);
            let (preview_decode_start_s, _) =
                preview_trace_span(scheduler, &token_ops, ready_s, &state.dependencies);
            if let Some((queue_delay_s, max_decode_queue_delay_s)) =
                decode_queue_delay_rejection(state, preview_decode_start_s)
            {
                record_decode_queue_breakdown(
                    state,
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                reject_decode_queue_admission(
                    state,
                    preview_decode_start_s,
                    queue_delay_s,
                    max_decode_queue_delay_s,
                );
                break;
            }
            if let Some((queue_delay_s, max_queue_delay_s)) =
                decode_iteration_queue_timeout(state, preview_decode_start_s)
            {
                record_decode_queue_breakdown(
                    state,
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                timeout_decode_iteration_queue(state, queue_delay_s, max_queue_delay_s);
                break;
            }
            let decode_ids = schedule_trace(
                scheduler,
                &format!(
                    "request {} decode token {}",
                    state.request_idx, state.emitted_tokens
                ),
                &token_ops,
                ready_s,
                &state.dependencies,
            );
            let (decode_start_s, _) = operation_span(scheduler, &decode_ids);
            let dependencies = terminal_ids(&token_ops, &decode_ids);
            let finish_s = operation_finish_s(scheduler, &dependencies);
            let is_first_token = state.emitted_tokens == 0;
            record_decode_queue_breakdown(
                state,
                scheduler,
                worker_runtime,
                decode_worker_slots_per_gpu,
                decode_start_s,
            );
            decode_iterations.push(ServingDecodeIterationObservation {
                iteration_idx: decode_iterations.len().min(u32::MAX as usize) as u32,
                decode_nodes: vec![state.decode_node],
                request_indices: vec![state.request_idx],
                operation_ids: decode_ids.clone(),
                start_s: decode_start_s,
                finish_s,
                latency_s: (finish_s - decode_start_s).max(0.0),
                batch_tokens: state.batch_size.max(1),
                first_token_batch_tokens: if is_first_token {
                    state.batch_size.max(1)
                } else {
                    0
                },
                tail_token_batch_tokens: if is_first_token {
                    0
                } else {
                    state.batch_size.max(1)
                },
            });
            record_decode_finish(state, decode_start_s, finish_s, dependencies);
            let assignments = assign_decode_workers_ready(
                worker_runtime,
                state,
                decode_worker_slots_per_gpu,
                decode_start_s,
                finish_s,
                &decode_ids,
            );
            state.worker_assignments.extend(assignments);
            if let Some(cancellation_s) = state.cancellation_s
                && cancellation_s <= finish_s + 1e-12
            {
                cancel_request(state, cancellation_s, "request cancelled during decode");
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn schedule_continuous_decodes(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    decode_iterations: &mut Vec<ServingDecodeIterationObservation>,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    max_batch_tokens: Option<u32>,
    decode_worker_slots_per_gpu: usize,
) {
    let max_batch_tokens = max_batch_tokens.unwrap_or(u32::MAX).max(1);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let mut iteration_idx = 0_u32;

    while states
        .iter()
        .any(|state| state.remaining_tokens > 0 && state.status == ServingRequestStatus::Pending)
    {
        mark_decode_ready_cancellations(
            scheduler,
            worker_runtime,
            states,
            decode_worker_slots_per_gpu,
        );
        let Some((ready_idx, ready_s)) = states
            .iter()
            .enumerate()
            .filter(|(_, state)| {
                state.remaining_tokens > 0 && state.status == ServingRequestStatus::Pending
            })
            .map(|(idx, state)| {
                (
                    idx,
                    decode_candidate_ready_s(
                        state,
                        scheduler,
                        worker_runtime,
                        decode_worker_slots_per_gpu,
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
        let target_decode_node = decode_resource_base_node.map(|_| states[ready_idx].decode_node);
        let mut selected = Vec::new();
        let mut selected_batch_tokens = 0_u32;

        let mut candidates = states
            .iter()
            .enumerate()
            .filter_map(|(idx, state)| {
                if state.remaining_tokens == 0 || state.status != ServingRequestStatus::Pending {
                    return None;
                }
                if let Some(target_decode_node) = target_decode_node
                    && state.decode_node != target_decode_node
                {
                    return None;
                }
                let state_ready_s = decode_candidate_ready_s(
                    state,
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                );
                if state_ready_s > ready_s + 1e-12 {
                    return None;
                }
                if let Some(cancellation_s) = state.cancellation_s
                    && cancellation_s <= state_ready_s + 1e-12
                {
                    return None;
                }
                Some(idx)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| compare_request_priority(states, *left, *right));

        for idx in candidates {
            let state = &states[idx];
            let request_batch = state.batch_size.max(1);
            if selected.is_empty()
                || selected_batch_tokens.saturating_add(request_batch) <= max_batch_tokens
            {
                selected.push(idx);
                selected_batch_tokens = selected_batch_tokens.saturating_add(request_batch);
            }
        }
        if selected.is_empty() {
            continue;
        }

        let mut first_token_batch = 0_u32;
        let mut tail_token_batch = 0_u32;
        let mut dependencies = Vec::new();
        for idx in &selected {
            let state = &states[*idx];
            if state.emitted_tokens == 0 {
                first_token_batch = first_token_batch.saturating_add(state.batch_size);
            } else {
                tail_token_batch = tail_token_batch.saturating_add(state.batch_size);
            }
            dependencies.extend_from_slice(&state.dependencies);
        }
        dependencies.sort_unstable();
        dependencies.dedup();

        let effective_batch =
            f64::from(first_token_batch) + f64::from(tail_token_batch) * decode_tail_scale;
        let scale = effective_batch / f64::from(base_batch_size.max(1));
        let mut token_ops = decode_one_score.operations.clone();
        scale_operations(&mut token_ops, scale);
        if let Some(target_decode_node) = target_decode_node {
            remap_operations_to_node(
                &mut token_ops,
                decode_resource_base_node,
                target_decode_node,
            );
        }
        let (preview_decode_start_s, _) =
            preview_trace_span(scheduler, &token_ops, ready_s, &dependencies);
        let rejected_for_decode_queue = selected
            .iter()
            .filter_map(|idx| {
                decode_queue_delay_rejection(&states[*idx], preview_decode_start_s).map(
                    |(queue_delay_s, max_decode_queue_delay_s)| {
                        (*idx, queue_delay_s, max_decode_queue_delay_s)
                    },
                )
            })
            .collect::<Vec<_>>();
        let timed_out_for_decode_queue = selected
            .iter()
            .filter_map(|idx| {
                decode_iteration_queue_timeout(&states[*idx], preview_decode_start_s).map(
                    |(queue_delay_s, max_queue_delay_s)| (*idx, queue_delay_s, max_queue_delay_s),
                )
            })
            .collect::<Vec<_>>();
        if !rejected_for_decode_queue.is_empty() || !timed_out_for_decode_queue.is_empty() {
            for (idx, queue_delay_s, max_decode_queue_delay_s) in rejected_for_decode_queue {
                record_decode_queue_breakdown(
                    &mut states[idx],
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                reject_decode_queue_admission(
                    &mut states[idx],
                    preview_decode_start_s,
                    queue_delay_s,
                    max_decode_queue_delay_s,
                );
            }
            for (idx, queue_delay_s, max_queue_delay_s) in timed_out_for_decode_queue {
                record_decode_queue_breakdown(
                    &mut states[idx],
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                timeout_decode_iteration_queue(&mut states[idx], queue_delay_s, max_queue_delay_s);
            }
            continue;
        }
        let decode_ids = schedule_trace(
            scheduler,
            &format!(
                "decode iteration {iteration_idx} batch {}",
                selected_batch_tokens
            ),
            &token_ops,
            ready_s,
            &dependencies,
        );
        let (decode_start_s, _) = operation_span(scheduler, &decode_ids);
        let terminal_dependencies = terminal_ids(&token_ops, &decode_ids);
        let finish_s = operation_finish_s(scheduler, &terminal_dependencies);
        let request_indices = selected
            .iter()
            .map(|idx| states[*idx].request_idx)
            .collect::<Vec<_>>();
        let decode_nodes = selected_decode_nodes(states, &selected);
        decode_iterations.push(ServingDecodeIterationObservation {
            iteration_idx,
            decode_nodes,
            request_indices,
            operation_ids: decode_ids.clone(),
            start_s: decode_start_s,
            finish_s,
            latency_s: (finish_s - decode_start_s).max(0.0),
            batch_tokens: selected_batch_tokens,
            first_token_batch_tokens: first_token_batch,
            tail_token_batch_tokens: tail_token_batch,
        });
        for idx in &selected {
            record_decode_queue_breakdown(
                &mut states[*idx],
                scheduler,
                worker_runtime,
                decode_worker_slots_per_gpu,
                decode_start_s,
            );
        }
        let worker_assignments = assign_selected_decode_workers_ready(
            worker_runtime,
            states,
            &selected,
            decode_worker_slots_per_gpu,
            decode_start_s,
            finish_s,
            &decode_ids,
        );

        let worker_assignments_by_idx = worker_assignments.into_iter().collect::<BTreeMap<_, _>>();
        for idx in selected {
            if let Some(assignments) = worker_assignments_by_idx.get(&idx) {
                states[idx].worker_assignments.extend(assignments.clone());
            }
            record_decode_finish(
                &mut states[idx],
                decode_start_s,
                finish_s,
                terminal_dependencies.clone(),
            );
            if let Some(cancellation_s) = states[idx].cancellation_s
                && cancellation_s <= finish_s + 1e-12
            {
                cancel_request(
                    &mut states[idx],
                    cancellation_s,
                    "request cancelled during decode",
                );
            }
        }
        iteration_idx += 1;
    }
}

pub(super) fn mark_decode_ready_cancellations(
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    decode_worker_slots_per_gpu: usize,
) {
    for state in states {
        if state.status != ServingRequestStatus::Pending || state.remaining_tokens == 0 {
            continue;
        }
        let ready_s = decode_candidate_ready_s(
            state,
            scheduler,
            worker_runtime,
            decode_worker_slots_per_gpu,
        );
        if let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= ready_s + 1e-12
        {
            cancel_request(
                state,
                cancellation_s,
                "request cancelled before decode iteration",
            );
        }
    }
}

pub(super) fn selected_decode_nodes(
    states: &[DecodeRequestState],
    selected: &[usize],
) -> Vec<NodeId> {
    let mut nodes = selected
        .iter()
        .map(|idx| states[*idx].decode_node)
        .collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

pub(super) fn record_decode_finish(
    state: &mut DecodeRequestState,
    start_s: f64,
    finish_s: f64,
    dependencies: Vec<usize>,
) {
    if state.first_decode_start_s.is_none() {
        state.first_decode_start_s = Some(start_s);
    }
    if state.first_decode_finish_s.is_none() {
        state.first_decode_finish_s = Some(finish_s);
    }
    state.last_decode_finish_s = Some(finish_s);
    state.decode_token_start_s.push(start_s);
    state.decode_token_finish_s.push(finish_s);
    state.dependencies = dependencies;
    state.emitted_tokens += 1;
    state.remaining_tokens = state.remaining_tokens.saturating_sub(1);
}

pub(super) fn apply_terminal_statuses(states: &mut [DecodeRequestState], traffic: &ServingTraffic) {
    for state in states {
        if state.status != ServingRequestStatus::Pending {
            continue;
        }
        let Some(last_decode_finish_s) = state.last_decode_finish_s else {
            state.status = ServingRequestStatus::TimedOut;
            let status_time_s = [
                state.kv_finish_s,
                state.prefill_finish_s,
                state.prefill_start_s,
                state.arrival_s,
            ]
            .into_iter()
            .find(|value| value.is_finite())
            .unwrap_or(state.arrival_s);
            state.status_time_s = Some(status_time_s);
            set_request_rejection(
                state,
                request_rejection(
                    "decode",
                    "timeout",
                    "decode_scheduler",
                    "decode_not_scheduled",
                    "request did not complete decode scheduling".to_string(),
                ),
            );
            continue;
        };
        if let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= last_decode_finish_s + 1e-12
        {
            cancel_request(state, cancellation_s, "request cancelled before completion");
            continue;
        }
        let e2el_s = (last_decode_finish_s - state.arrival_s).max(0.0);
        let timeout_s = state.request_timeout_s.or(traffic.request_timeout_s);
        if let Some(timeout_s) = timeout_s
            && e2el_s > timeout_s + 1e-12
        {
            state.status = ServingRequestStatus::TimedOut;
            state.status_time_s = Some(last_decode_finish_s);
            let message =
                format!("request timed out: e2el {e2el_s:.6}s > request_timeout {timeout_s:.6}s");
            set_request_rejection(
                state,
                with_rejection_details(
                    request_rejection(
                        "end_to_end",
                        "timeout",
                        "request_timeout",
                        "request_timeout_exceeded",
                        message,
                    ),
                    Some(e2el_s),
                    Some(timeout_s),
                    Some("s"),
                    Some(
                        "increase request_timeout, reduce load, or choose a faster serving config",
                    ),
                ),
            );
        } else {
            state.status = ServingRequestStatus::Completed;
            state.status_time_s = Some(last_decode_finish_s);
            state.failure_reason = None;
            state.failure_rejection = None;
        }
    }
}

pub(super) fn inter_token_latencies(token_finish_s: &[f64]) -> Vec<f64> {
    token_finish_s
        .windows(2)
        .filter_map(|window| {
            let delta_s = window[1] - window[0];
            if delta_s.is_finite() && delta_s >= 0.0 {
                Some(delta_s)
            } else {
                None
            }
        })
        .collect()
}
