//! Phase pipeline scheduler: every prefill, then every KV transfer, then
//! every decode iteration, on the greedy resource scheduler. Used for
//! disaggregated pools, independent batching, and other candidates the
//! iteration engine does not cover.

use super::*;

/// Operations and decode iterations a scheduler produced for one candidate.
pub(in crate::serving) struct ScheduledTimeline {
    pub(in crate::serving) operations: Vec<ScheduledOperation>,
    pub(in crate::serving) decode_iterations: Vec<ServingDecodeIterationObservation>,
    pub(in crate::serving) kv_bottlenecks: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::serving) fn schedule_phase_pipeline(
    decode_states: &mut [DecodeRequestState],
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
    decode_tail_scale: f64,
) -> ScheduledTimeline {
    let prefill_resource_base_node = single_placement_node(prefill_score);
    let mut scheduler = ResourceScheduler::new();
    let mut decode_iterations = Vec::new();
    let mut kv_bottlenecks = Vec::new();
    let mut worker_runtime = ServingWorkerRuntime::default();

    schedule_prefills(
        &mut scheduler,
        &mut worker_runtime,
        decode_states,
        prefill_score,
        traffic,
        prefill_resource_base_node,
        request.batch_size,
        request.prompt_tokens,
    );
    for state in decode_states.iter_mut() {
        if state.status == ServingRequestStatus::Pending
            && let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= state.prefill_finish_s + 1e-12
        {
            cancel_request(state, cancellation_s, "request cancelled during prefill");
        }
    }

    for state in decode_states.iter_mut() {
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
        decode_states,
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
            decode_states,
            &mut decode_iterations,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            decode_worker_slots_per_gpu(traffic),
        ),
        ServingDecodeBatching::Continuous { max_batch_tokens } => schedule_continuous_decodes(
            &mut scheduler,
            &mut worker_runtime,
            decode_states,
            &mut decode_iterations,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            max_batch_tokens,
            decode_worker_slots_per_gpu(traffic),
        ),
    }

    ScheduledTimeline {
        operations: scheduler.operations().to_vec(),
        decode_iterations,
        kv_bottlenecks,
    }
}
