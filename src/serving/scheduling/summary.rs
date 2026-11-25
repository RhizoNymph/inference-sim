//! Turns scheduled request states into serving metrics, capacity, worker,
//! service, and utilization observations. Shared by both schedulers.

use super::*;

/// Inputs the summary needs besides the request states and the timeline.
pub(in crate::serving) struct SummaryContext<'a> {
    pub(in crate::serving) prefill_score: &'a ScoredParallelismConfig,
    pub(in crate::serving) decode_score: &'a ScoredParallelismConfig,
    pub(in crate::serving) model: &'a ModelSpec,
    pub(in crate::serving) request: &'a InferenceRequest,
    pub(in crate::serving) traffic: &'a ServingTraffic,
    pub(in crate::serving) prefill_nodes: &'a [NodeId],
    pub(in crate::serving) decode_nodes: &'a [NodeId],
    pub(in crate::serving) calibration_profile: Option<&'a CalibrationProfileMetadata>,
    pub(in crate::serving) scheduled_requests: u32,
    pub(in crate::serving) scheduler_model: SchedulerModel,
}

pub(in crate::serving) fn summarize_serving_simulation(
    decode_states: Vec<DecodeRequestState>,
    timeline: ScheduledTimeline,
    context: SummaryContext<'_>,
) -> ServingSimulation {
    let SummaryContext {
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
    } = context;
    let ScheduledTimeline {
        operations,
        decode_iterations,
        kv_bottlenecks,
    } = timeline;
    let mut calibration_fits = decode_states
        .iter()
        .filter_map(|state| state.kv_transfer_fit.clone())
        .collect::<Vec<_>>();
    let observations: Vec<_> = decode_states
        .iter()
        .map(ServingRequestObservation::from_decode_state)
        .collect();
    let makespan_s = operations
        .iter()
        .map(|operation| operation.finish_s)
        .fold(0.0, f64::max);
    let measurement_window =
        effective_measurement_window(traffic, &observations, &operations, makespan_s);
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
    let scheduled_operations = operations;
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
        scheduler_model: Some(scheduler_model),
    }
}
