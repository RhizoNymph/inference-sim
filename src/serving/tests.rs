use super::*;
use crate::{
    config::{CalibrationFitFeatureRange, CalibrationFittedModel, parse_cluster},
    types::{
        common::{Bytes, GpuAddr},
        configs::{ParallelGroups, RankPlacement},
        fabric::inter_node::InterNodeTopology,
        fabric::variants::ib::IbVariant,
        topology::Cluster,
    },
    workload::{DType, InferenceRequest},
};

fn model() -> ModelSpec {
    ModelSpec {
        layers: 4,
        hidden_size: 4096,
        attention_heads: 32,
        kv_heads: 8,
        vocab_size: 32000,
        parameters: Bytes::from_gigabytes(16.0),
        parameter_count: None,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn request() -> InferenceRequest {
    InferenceRequest {
        batch_size: 2,
        prompt_tokens: 128,
        decode_tokens: 16,
        max_sequence_tokens: 256,
        phase: InferencePhase::EndToEnd,
    }
}

#[test]
fn kv_dtype_controls_serving_kv_transfer_bytes() {
    let bf16_model = model();
    let mut fp8_kv_model = model();
    fp8_kv_model.kv_dtype = Some(DType::Fp8);
    let request = request();

    let bf16_bytes = kv_transfer_bytes_for_routes(&bf16_model, &request, &[0], &[1], &[], &[]);
    let fp8_bytes = kv_transfer_bytes_for_routes(&fp8_kv_model, &request, &[0], &[1], &[], &[]);

    assert_eq!(fp8_bytes.as_bytes() * 2, bf16_bytes.as_bytes());
}

fn constant_serving_metric_fit(target: &str, latency_ms: f64) -> CalibrationFittedModel {
    CalibrationFittedModel {
        name: Some(format!("{target}-constant")),
        target: target.to_string(),
        phase: Some("serving".to_string()),
        kind: Some("latency".to_string()),
        model: "linear".to_string(),
        unit: Some("ms".to_string()),
        intercept: Some(latency_ms),
        features: vec!["baseline_ms".to_string()],
        coefficients: vec![0.0],
        feature_ranges: Vec::new(),
        r_squared: Some(1.0),
        adjusted_r_squared: Some(1.0),
        rmse: Some(0.0),
        rmse_pct: Some(0.0),
        mean_abs_pct_error: Some(0.0),
        max_abs_pct_error: Some(0.0),
        validation_rmse: None,
        validation_rmse_pct: None,
        validation_mean_abs_pct_error: None,
        validation_max_abs_pct_error: None,
        confidence_interval: None,
        confidence_interval_pct: None,
        confidence_level: None,
        sample_count: Some(8),
        validation_sample_count: Some(2),
        source: Some("unit-test".to_string()),
        notes: None,
    }
}

fn constant_serving_throughput_fit(throughput_tokens_per_s: f64) -> CalibrationFittedModel {
    CalibrationFittedModel {
        name: Some("throughput_tokens_per_s-constant".to_string()),
        target: "throughput_tokens_per_s".to_string(),
        phase: Some("serving".to_string()),
        kind: Some("throughput".to_string()),
        model: "linear".to_string(),
        unit: Some("tokens/s".to_string()),
        intercept: Some(throughput_tokens_per_s),
        features: vec!["baseline_value".to_string()],
        coefficients: vec![0.0],
        feature_ranges: Vec::new(),
        r_squared: Some(1.0),
        adjusted_r_squared: Some(1.0),
        rmse: Some(0.0),
        rmse_pct: Some(0.0),
        mean_abs_pct_error: Some(0.0),
        max_abs_pct_error: Some(0.0),
        validation_rmse: None,
        validation_rmse_pct: None,
        validation_mean_abs_pct_error: None,
        validation_max_abs_pct_error: None,
        confidence_interval: None,
        confidence_interval_pct: None,
        confidence_level: None,
        sample_count: Some(8),
        validation_sample_count: Some(2),
        source: Some("unit-test".to_string()),
        notes: None,
    }
}

fn calibration_profile_with_fits(fits: Vec<CalibrationFittedModel>) -> CalibrationProfileMetadata {
    CalibrationProfileMetadata {
        path: "in-memory".to_string(),
        name: Some("unit-profile".to_string()),
        hardware: Some("h100_sxm".to_string()),
        fabric: Some("ib_ndr".to_string()),
        model: Some("test-model".to_string()),
        dtype: Some("bf16".to_string()),
        serving_stack: Some("unit-test".to_string()),
        serving_runtime_features: Vec::new(),
        backend_version: None,
        driver_version: None,
        cuda_version: None,
        rocm_version: None,
        nccl_version: None,
        rccl_version: None,
        ucx_version: None,
        kernel_settings: Vec::new(),
        environment_hash: None,
        source: Some("unit-test".to_string()),
        date: Some("2026-05-26".to_string()),
        notes: None,
        valid_shape: None,
        invalid_shapes: Vec::new(),
        fits,
        benchmarks: Vec::new(),
    }
}

fn worker_decode_score(duration_s: f64) -> ScoredParallelismConfig {
    ScoredParallelismConfig {
        config: ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        },
        placement: RankPlacement {
            rank_to_gpu: vec![GpuAddr {
                node_id: 0,
                local_gpu_id: 0,
            }],
        },
        placement_evidence: Vec::new(),
        groups: ParallelGroups {
            tensor_groups: Vec::new(),
            pipeline_stages: Vec::new(),
            expert_groups: Vec::new(),
            data_groups: Vec::new(),
        },
        feasible: true,
        estimated_latency_s: duration_s,
        estimated_memory_per_gpu: Bytes::from_bytes(0),
        calibration_fits: Vec::new(),
        calibration_gate_violations: Vec::new(),
        approximations: Vec::new(),
        approximation_policy_violations: Vec::new(),
        bottlenecks: Vec::new(),
        rejected_reason: None,
        operations: vec![SimOperation {
            name: "decode".to_string(),
            kind: crate::solver::SimOperationKind::Compute,
            duration_s,
            resources: Vec::new(),
            dependencies: Vec::new(),
        }],
        scheduled_operations: Vec::new(),
        resource_utilization: Vec::new(),
        operation_makespan_s: duration_s,
    }
}

fn pending_decode_state(request_idx: u32, gpu: GpuAddr) -> DecodeRequestState {
    DecodeRequestState {
        request_idx,
        request_id: None,
        tenant: None,
        model_id: None,
        traffic_class: None,
        shape_profile: None,
        cache_key: None,
        prefill_node: gpu.node_id,
        decode_node: gpu.node_id,
        prefill_route_nodes: vec![gpu.node_id],
        prefill_route_gpus: vec![gpu],
        decode_route_nodes: vec![gpu.node_id],
        decode_route_gpus: vec![gpu],
        routing_policy: ServingRoutingPolicy::RoundRobin,
        routing_candidate_count: 1,
        routing_routable_candidate_count: 1,
        routing_estimated_e2el_s: 0.0,
        routing_estimated_kv_transfer_s: 0.0,
        routing_estimated_kv_resource_wait_s: 0.0,
        routing_estimated_prefill_wait_s: 0.0,
        routing_estimated_decode_wait_s: 0.0,
        routing_reason: String::new(),
        routing_candidates: Vec::new(),
        arrival_s: 0.0,
        priority: 0,
        batch_size: 1,
        prompt_tokens: 1,
        prefix_cache_hit_tokens: 0,
        effective_prefill_tokens: 1,
        remaining_prefill_tokens: 0,
        prefill_chunks: 1,
        decode_tokens: 1,
        slo: ServingRequestSlo::default(),
        max_queue_delay_s: None,
        max_kv_queue_delay_s: None,
        max_decode_queue_delay_s: None,
        max_decode_iteration_queue_delay_s: None,
        request_timeout_s: None,
        deadline_s: None,
        cancellation_s: None,
        remaining_tokens: 1,
        emitted_tokens: 0,
        max_sequence_tokens: 2,
        kv_block_tokens: 16,
        kv_cache_blocks: 1,
        kv_allocated_tokens: 16,
        kv_fragmentation_tokens: 14,
        prefill_scheduled: true,
        dependencies: Vec::new(),
        kv_start_s: 0.0,
        kv_finish_s: 0.0,
        first_decode_start_s: None,
        first_decode_finish_s: None,
        last_decode_finish_s: None,
        decode_token_start_s: Vec::new(),
        decode_token_finish_s: Vec::new(),
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
    }
}

fn pending_prefill_state(request_idx: u32, gpu: GpuAddr) -> DecodeRequestState {
    let mut state = pending_decode_state(request_idx, gpu);
    state.prefill_scheduled = false;
    state.remaining_prefill_tokens = 1;
    state.prefill_chunks = 0;
    state.effective_prefill_tokens = 1;
    state.prefill_start_s = f64::INFINITY;
    state.prefill_finish_s = f64::INFINITY;
    state.dependencies.clear();
    state
}

#[test]
fn route_worker_readiness_uses_slowest_routed_gpu() {
    let gpu0 = GpuAddr {
        node_id: 7,
        local_gpu_id: 0,
    };
    let gpu1 = GpuAddr {
        node_id: 7,
        local_gpu_id: 1,
    };
    let mut ready_s = BTreeMap::new();

    ready_s.insert(gpu0, vec![3.0]);
    ready_s.insert(gpu1, vec![5.0]);

    assert_eq!(route_workers_ready_s(&ready_s, &[gpu0, gpu1], 7, 1), 5.0);
    assert_eq!(route_workers_ready_s(&ready_s, &[gpu1, gpu0], 7, 1), 5.0);
    assert_eq!(route_workers_ready_s(&ready_s, &[], 9, 1), 0.0);

    mark_route_workers_ready(&mut ready_s, &[], 9, 1, 4.0);
    assert_eq!(
        ready_s[&GpuAddr {
            node_id: 9,
            local_gpu_id: 0,
        }],
        vec![4.0]
    );
}

#[test]
fn prefill_worker_slots_use_earliest_busy_slot() {
    let gpu = GpuAddr {
        node_id: 7,
        local_gpu_id: 0,
    };
    let state = pending_prefill_state(0, gpu);
    let mut runtime = ServingWorkerRuntime::default();

    assert_eq!(prefill_worker_ready_s(&state, &runtime, 2), 0.0);
    mark_worker_slot_ready(&mut runtime.prefill_ready_s, gpu, 2, 3.0);
    assert_eq!(prefill_worker_ready_s(&state, &runtime, 2), 0.0);
    mark_worker_slot_ready(&mut runtime.prefill_ready_s, gpu, 2, 5.0);
    assert_eq!(prefill_worker_ready_s(&state, &runtime, 2), 3.0);
    mark_worker_slot_ready(&mut runtime.prefill_ready_s, gpu, 2, 7.0);
    assert_eq!(prefill_worker_ready_s(&state, &runtime, 2), 5.0);
}

#[test]
fn independent_decode_uses_worker_ready_even_without_resource_conflict() {
    let gpu = GpuAddr {
        node_id: 0,
        local_gpu_id: 0,
    };
    let mut scheduler = ResourceScheduler::new();
    let mut worker_runtime = ServingWorkerRuntime::default();
    let mut states = vec![pending_decode_state(0, gpu), pending_decode_state(1, gpu)];
    let mut decode_iterations = Vec::new();
    let decode_score = worker_decode_score(1.0);

    schedule_independent_decodes(
        &mut scheduler,
        &mut worker_runtime,
        &mut states,
        &mut decode_iterations,
        &decode_score,
        1.0,
        1,
        1,
    );

    assert_eq!(decode_iterations.len(), 2);
    assert_eq!(states[0].first_decode_start_s, Some(0.0));
    assert_eq!(states[0].first_decode_finish_s, Some(1.0));
    assert_eq!(states[1].first_decode_start_s, Some(1.0));
    assert_eq!(states[1].first_decode_finish_s, Some(2.0));
    assert_eq!(worker_runtime.decode_ready_s[&gpu], vec![2.0]);
    assert_eq!(states[0].decode_worker_queue_s, 0.0);
    assert_eq!(states[0].decode_resource_queue_s, 0.0);
    assert_eq!(states[1].decode_worker_queue_s, 1.0);
    assert_eq!(states[1].decode_resource_queue_s, 0.0);

    for state in &mut states {
        state.status = ServingRequestStatus::Completed;
    }
    let observations = states
        .iter()
        .map(ServingRequestObservation::from_decode_state)
        .collect::<Vec<_>>();
    assert_eq!(observations[0].status_time_s, Some(1.0));
    assert_eq!(observations[0].metric_source, "request_lifecycle_events");
    assert_eq!(
        observations[0]
            .lifecycle_events
            .last()
            .map(|event| event.kind),
        Some(ServingRequestEventKind::Completed)
    );
    assert!(
        observations[0]
            .lifecycle_events
            .windows(2)
            .all(|window| window[0].at_s <= window[1].at_s + 1e-12)
    );
    assert!(
        observations[0]
            .lifecycle_events
            .iter()
            .any(
                |event| event.kind == ServingRequestEventKind::DecodeIterationStarted
                    && event.decode_iteration == Some(0)
            )
    );
    let first_decode_finish_event = observations[0]
        .lifecycle_events
        .iter()
        .find(|event| event.kind == ServingRequestEventKind::DecodeIterationFinished)
        .expect("first decode finish event");
    assert!((observations[0].ttft_s - first_decode_finish_event.at_s).abs() < 1e-12);
    assert_eq!(
        observations[0].decode_iterations as usize,
        observations[0]
            .lifecycle_events
            .iter()
            .filter(|event| event.kind == ServingRequestEventKind::DecodeIterationFinished)
            .count()
    );
    assert!(
        observations[0]
            .lifecycle_events
            .iter()
            .any(|event| event.kind == ServingRequestEventKind::KvBlocksAllocated)
    );
    assert!(
        observations[0]
            .lifecycle_events
            .iter()
            .any(|event| event.kind == ServingRequestEventKind::KvBlocksReleased)
    );
    assert_eq!(observations[0].kv_block_ownership.len(), 1);
    let ownership = &observations[0].kv_block_ownership[0];
    assert_eq!(ownership.owner, gpu);
    assert_eq!(ownership.allocated_at_s, 0.0);
    assert_eq!(ownership.released_at_s, 1.0);
    assert_eq!(ownership.duration_s, 1.0);
    assert_eq!(ownership.allocation_id, "request-0:node-0:gpu-0:blocks-0-1");
    assert_eq!(ownership.block_start, 0);
    assert_eq!(ownership.block_end, 1);
    assert_eq!(ownership.decode_sequences, 1);
    assert_eq!(ownership.resident_tokens, 2);
    assert_eq!(ownership.kv_blocks, 1);
    assert_eq!(ownership.allocated_kv_tokens, 16);
    assert_eq!(ownership.kv_fragmentation_tokens, 14);
    assert_eq!(ownership.block_table_entries, 1);
    assert_eq!(ownership.block_table_bytes, 16);
    assert_eq!(ownership.worker_slot_ownership.len(), 1);
    assert_eq!(
        ownership.worker_slot_ownership[0].allocation_id,
        "request-0:node-0:gpu-0:blocks-0-1:slot-0:blocks-0-1"
    );
    assert_eq!(ownership.worker_slot_ownership[0].slot, 0);
    assert_eq!(ownership.worker_slot_ownership[0].block_start, 0);
    assert_eq!(ownership.worker_slot_ownership[0].block_end, 1);
    assert_eq!(ownership.worker_slot_ownership[0].decode_sequences, 1);
    assert_eq!(ownership.worker_slot_ownership[0].resident_tokens, 2);
    assert_eq!(ownership.worker_slot_ownership[0].kv_blocks, 1);
    assert_eq!(ownership.worker_slot_ownership[0].block_table_bytes, 16);
    assert!(observations[0].phase_breakdown.iter().any(|phase| {
        phase.phase == "first_decode_iteration"
            && phase.category == ServingRequestPhaseCategory::Service
    }));
    let ttft_breakdown_s = observations[0]
        .phase_breakdown
        .iter()
        .filter(|phase| phase.contributes_to_ttft)
        .map(|phase| phase.duration_s)
        .sum::<f64>();
    let e2el_breakdown_s = observations[0]
        .phase_breakdown
        .iter()
        .filter(|phase| phase.contributes_to_e2el)
        .map(|phase| phase.duration_s)
        .sum::<f64>();
    assert!((ttft_breakdown_s - observations[0].ttft_s).abs() < 1e-12);
    assert!((e2el_breakdown_s - observations[0].e2el_s).abs() < 1e-12);
    let workers = worker_observations(&observations, &[], &ServingTraffic::default());
    let decode_worker = workers
        .iter()
        .find(|observation| {
            observation.phase == "decode"
                && observation.node_id == gpu.node_id
                && observation.local_gpu_id == gpu.local_gpu_id
        })
        .expect("decode worker observation");
    assert!(decode_worker.worker_queue_s > 0.0);
    assert_eq!(decode_worker.resource_queue_s, 0.0);
    assert_eq!(decode_worker.kv_cache_owner_slots.len(), 1);
    assert_eq!(decode_worker.kv_cache_owner_slots[0].slot, 0);
    assert_eq!(decode_worker.kv_cache_owner_slots[0].peak_kv_blocks, 2);
    assert_eq!(
        decode_worker.kv_cache_owner_slots[0].peak_kv_block_table_bytes,
        32
    );
}

#[test]
fn kv_block_ownership_partitions_logical_blocks_without_double_counting() {
    let gpu0 = GpuAddr {
        node_id: 0,
        local_gpu_id: 0,
    };
    let gpu1 = GpuAddr {
        node_id: 0,
        local_gpu_id: 1,
    };
    let mut state = pending_decode_state(3, gpu0);
    state.decode_route_gpus = vec![gpu0, gpu1];
    state.batch_size = 1;
    state.max_sequence_tokens = 33;
    state.kv_block_tokens = 16;
    state.kv_cache_blocks = 3;
    state.kv_allocated_tokens = 48;
    state.kv_fragmentation_tokens = 15;
    state.kv_finish_s = 0.25;
    state.last_decode_finish_s = Some(1.25);
    state.status = ServingRequestStatus::Completed;

    let ownership = request_kv_block_ownership(&state);

    assert_eq!(ownership.len(), 2);
    assert_eq!(
        ownership[0].allocation_id,
        "request-3:node-0:gpu-0:blocks-0-2"
    );
    assert_eq!(ownership[0].block_start, 0);
    assert_eq!(ownership[0].block_end, 2);
    assert_eq!(ownership[0].resident_tokens, 22);
    assert_eq!(ownership[0].kv_blocks, 2);
    assert_eq!(ownership[0].allocated_kv_tokens, 32);
    assert_eq!(ownership[0].kv_fragmentation_tokens, 10);
    assert_eq!(ownership[0].block_table_entries, 2);
    assert_eq!(ownership[0].block_table_bytes, 32);
    assert_eq!(
        ownership[1].allocation_id,
        "request-3:node-0:gpu-1:blocks-2-3"
    );
    assert_eq!(ownership[1].block_start, 2);
    assert_eq!(ownership[1].block_end, 3);
    assert_eq!(ownership[1].resident_tokens, 11);
    assert_eq!(ownership[1].kv_blocks, 1);
    assert_eq!(ownership[1].allocated_kv_tokens, 16);
    assert_eq!(ownership[1].kv_fragmentation_tokens, 5);
    assert_eq!(ownership[1].block_table_entries, 1);
    assert_eq!(ownership[1].block_table_bytes, 16);
    assert_eq!(
        ownership
            .iter()
            .map(|entry| entry.resident_tokens)
            .sum::<u64>(),
        33
    );
    assert_eq!(
        ownership.iter().map(|entry| entry.kv_blocks).sum::<u64>(),
        3
    );
    assert_eq!(
        ownership
            .iter()
            .map(|entry| entry.allocated_kv_tokens)
            .sum::<u64>(),
        48
    );
    assert_eq!(
        ownership
            .iter()
            .map(|entry| entry.kv_fragmentation_tokens)
            .sum::<u64>(),
        15
    );

    let capacity = capacity_profile(&[state], &ServingTraffic::default());
    assert_eq!(capacity.peak_resident_tokens, 33);
    assert_eq!(capacity.peak_kv_blocks, 3);
    assert_eq!(
        capacity
            .gpus
            .iter()
            .map(|gpu| gpu.peak_resident_tokens)
            .sum::<u64>(),
        capacity.peak_resident_tokens
    );
    assert_eq!(
        capacity
            .gpus
            .iter()
            .map(|gpu| gpu.peak_kv_blocks)
            .sum::<u64>(),
        capacity.peak_kv_blocks
    );
    assert_eq!(
        capacity
            .gpus
            .iter()
            .map(|gpu| gpu.peak_kv_fragmentation_tokens)
            .sum::<u64>(),
        capacity.peak_kv_fragmentation_tokens
    );
    assert_eq!(capacity.peak_kv_block_table_bytes, 48);
    assert_eq!(
        capacity
            .gpus
            .iter()
            .map(|gpu| gpu.peak_kv_block_table_bytes)
            .sum::<u64>(),
        capacity.peak_kv_block_table_bytes
    );
}

#[test]
fn decode_worker_slots_control_independent_decode_overlap() {
    let gpu = GpuAddr {
        node_id: 0,
        local_gpu_id: 0,
    };
    let decode_score = worker_decode_score(1.0);
    let mut scheduler = ResourceScheduler::new();
    let mut worker_runtime = ServingWorkerRuntime::default();
    let mut states = vec![pending_decode_state(0, gpu), pending_decode_state(1, gpu)];
    let mut decode_iterations = Vec::new();

    schedule_independent_decodes(
        &mut scheduler,
        &mut worker_runtime,
        &mut states,
        &mut decode_iterations,
        &decode_score,
        1.0,
        1,
        2,
    );

    assert_eq!(decode_iterations.len(), 2);
    assert_eq!(states[0].first_decode_start_s, Some(0.0));
    assert_eq!(states[0].first_decode_finish_s, Some(1.0));
    assert_eq!(states[1].first_decode_start_s, Some(0.0));
    assert_eq!(states[1].first_decode_finish_s, Some(1.0));
    assert_eq!(states[1].decode_worker_queue_s, 0.0);
    assert_eq!(worker_runtime.decode_ready_s[&gpu], vec![1.0, 1.0]);

    let observations = states
        .iter()
        .map(ServingRequestObservation::from_decode_state)
        .collect::<Vec<_>>();
    let traffic = ServingTraffic {
        max_decode_worker_slots_per_gpu: Some(2),
        ..ServingTraffic::default()
    };
    let workers = worker_observations(&observations, &[], &traffic);
    let worker = workers
        .iter()
        .find(|observation| {
            observation.phase == "decode"
                && observation.node_id == gpu.node_id
                && observation.local_gpu_id == gpu.local_gpu_id
        })
        .expect("decode worker observation");
    assert_eq!(worker.configured_worker_slots, 2);
    assert_eq!(worker.peak_active_worker_slots, 2);
    assert!((worker.worker_slot_utilization - 1.0).abs() < 1e-12);
    assert_eq!(
        observations
            .iter()
            .flat_map(|observation| observation.kv_block_ownership.iter())
            .flat_map(|ownership| ownership.worker_slot_ownership.iter())
            .map(|slot| slot.kv_blocks)
            .sum::<u64>(),
        2
    );
    assert!(
        observations
            .iter()
            .flat_map(|observation| observation.kv_block_ownership.iter())
            .all(|ownership| ownership.worker_slot_ownership.len() == 1)
    );
    assert_eq!(worker.kv_cache_owner_slots.len(), 2);
    assert!(
        worker
            .kv_cache_owner_slots
            .iter()
            .all(|slot| slot.peak_kv_blocks == 1
                && slot.peak_kv_block_table_bytes == KV_BLOCK_TABLE_ENTRY_BYTES)
    );
    assert_eq!(
        worker
            .kv_cache_owner_slots
            .iter()
            .map(|slot| slot.peak_kv_blocks)
            .sum::<u64>(),
        2
    );
}

#[test]
fn prefill_worker_slots_control_independent_prefill_overlap() {
    let gpu = GpuAddr {
        node_id: 0,
        local_gpu_id: 0,
    };
    let prefill_score = worker_decode_score(1.0);
    let mut single_slot_scheduler = ResourceScheduler::new();
    let mut single_slot_runtime = ServingWorkerRuntime::default();
    let mut single_slot_states = vec![pending_prefill_state(0, gpu), pending_prefill_state(1, gpu)];
    let single_slot_traffic = ServingTraffic {
        request_count: Some(2),
        prefill_batching: ServingPrefillBatching::Independent,
        max_prefill_worker_slots_per_gpu: Some(1),
        ..ServingTraffic::default()
    };

    schedule_prefills(
        &mut single_slot_scheduler,
        &mut single_slot_runtime,
        &mut single_slot_states,
        &prefill_score,
        &single_slot_traffic,
        Some(0),
        1,
        1,
    );

    assert_eq!(single_slot_states[0].prefill_start_s, 0.0);
    assert_eq!(single_slot_states[0].prefill_finish_s, 1.0);
    assert_eq!(single_slot_states[1].prefill_start_s, 1.0);
    assert_eq!(single_slot_states[1].prefill_finish_s, 2.0);
    assert_eq!(single_slot_states[1].prefill_worker_queue_s, 1.0);

    let mut two_slot_scheduler = ResourceScheduler::new();
    let mut two_slot_runtime = ServingWorkerRuntime::default();
    let mut two_slot_states = vec![pending_prefill_state(0, gpu), pending_prefill_state(1, gpu)];
    let two_slot_traffic = ServingTraffic {
        request_count: Some(2),
        prefill_batching: ServingPrefillBatching::Independent,
        max_prefill_worker_slots_per_gpu: Some(2),
        ..ServingTraffic::default()
    };

    schedule_prefills(
        &mut two_slot_scheduler,
        &mut two_slot_runtime,
        &mut two_slot_states,
        &prefill_score,
        &two_slot_traffic,
        Some(0),
        1,
        1,
    );

    assert_eq!(two_slot_states[0].prefill_start_s, 0.0);
    assert_eq!(two_slot_states[0].prefill_finish_s, 1.0);
    assert_eq!(two_slot_states[1].prefill_start_s, 0.0);
    assert_eq!(two_slot_states[1].prefill_finish_s, 1.0);
    assert_eq!(two_slot_states[1].prefill_worker_queue_s, 0.0);
    assert_eq!(two_slot_runtime.prefill_ready_s[&gpu], vec![1.0, 1.0]);

    let observations = two_slot_states
        .iter()
        .map(ServingRequestObservation::from_decode_state)
        .collect::<Vec<_>>();
    let workers = worker_observations(&observations, &[], &two_slot_traffic);
    let worker = workers
        .iter()
        .find(|observation| {
            observation.phase == "prefill"
                && observation.node_id == gpu.node_id
                && observation.local_gpu_id == gpu.local_gpu_id
        })
        .expect("prefill worker observation");
    assert_eq!(worker.configured_worker_slots, 2);
    assert_eq!(worker.peak_active_worker_slots, 2);
    assert!((worker.worker_slot_utilization - 1.0).abs() < 1e-12);
}

#[test]
fn ranks_disaggregated_serving_configs_with_metrics() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1, 2],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: vec![
            ServingPoolCandidate {
                label: Some("split".to_string()),
                prefill_nodes: vec![0],
                decode_nodes: vec![1],
                prefill_groups: Vec::new(),
                decode_groups: Vec::new(),
                prefill_node_filter: ServingPoolNodeFilter::default(),
                decode_node_filter: ServingPoolNodeFilter::default(),
                domain_spread: ServingPoolDomainSpread::default(),
                prefill_gpu_labels: Vec::new(),
                decode_gpu_labels: Vec::new(),
            },
            ServingPoolCandidate {
                label: Some("colocated".to_string()),
                prefill_nodes: Vec::new(),
                decode_nodes: Vec::new(),
                prefill_groups: vec!["h100".to_string()],
                decode_groups: vec!["h100".to_string()],
                prefill_node_filter: ServingPoolNodeFilter::default(),
                decode_node_filter: ServingPoolNodeFilter::default(),
                domain_spread: ServingPoolDomainSpread::default(),
                prefill_gpu_labels: Vec::new(),
                decode_gpu_labels: Vec::new(),
            },
        ],
        pool_search: Some(ServingPoolSearch {
            prefill_groups: vec!["h100".to_string()],
            decode_groups: vec!["h100".to_string()],
            prefill_node_counts: vec![1],
            decode_node_counts: vec![1],
            prefill_node_filter: ServingPoolNodeFilter::default(),
            decode_node_filter: ServingPoolNodeFilter::default(),
            prefill_gpu_labels: Vec::new(),
            decode_gpu_labels: Vec::new(),
            allow_overlap: false,
            domain_spread: ServingPoolDomainSpread::default(),
            max_candidates: 4,
        }),
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(6),
            arrival_gap_s: Some(0.0005),
            arrival: ServingArrivalPattern::Poisson {
                rate_per_s: 2_000.0,
                seed: 7,
            },
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(512),
                chunk_tokens: None,
            },
            decode_batching: ServingDecodeBatching::Continuous {
                max_batch_tokens: Some(8),
            },
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(16),
            max_resident_tokens: Some(16_384),
            max_decode_sequences_per_node: Some(16),
            max_resident_tokens_per_node: Some(16_384),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: Some(10.0),
            tpot_slo_s: Some(10.0),
            itl_slo_s: Some(10.0),
            e2el_slo_s: Some(10.0),
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: None,
            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1, 2],
            prompt_tokens: vec![64, 128],
            decode_tokens: vec![8, 16],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 12);
    assert!(
        results
            .iter()
            .any(|score| { score.deployment_mode == ServingDeploymentMode::FullyDisaggregated })
    );
    assert!(
        results
            .iter()
            .any(|score| score.deployment_mode == ServingDeploymentMode::Colocated)
    );
    assert!(
        results[0].feasible,
        "top result rejected: {:?}",
        results[0].rejected_reason
    );
    let candidate_ids = results
        .iter()
        .map(|score| score.candidate_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(candidate_ids.len(), results.len());
    assert!(results.iter().all(|score| {
        score.candidate_id.starts_with("serving:mode-")
            && score.candidate_id.contains(":pool-")
            && score.candidate_id.contains(":pre-")
            && score.candidate_id.contains(":dec-")
    }));
    assert!(
        results
            .iter()
            .filter(|score| score.feasible)
            .all(|score| score.pareto.rank.is_some())
    );
    assert!(
        results
            .iter()
            .any(|score| score.feasible && score.pareto.is_frontier)
    );
    assert_eq!(results[0].calibration_summary.status, "uncalibrated");
    assert!(results[0].calibration_summary.active_phase_count > 0);
    assert_eq!(results[0].calibration_summary.fit_count, 0);
    assert!(
        results[0].calibration_summary.uncalibrated_phase_count
            <= results[0].calibration_summary.active_phase_count
    );
    assert_eq!(
        results[0].approximation_summary.approximation_count,
        results[0].approximations.len() as u32
    );
    assert_eq!(results[0].approximation_summary.policy_violation_count, 0);
    assert_eq!(results[0].approximation_summary.status, "calibration_risk");
    assert!(results[0].approximation_summary.runtime_count > 0);
    assert!(results[0].approximation_summary.memory_count > 0);
    assert!(results[0].approximation_summary.approximate_queueing);
    assert!(results[0].approximation_summary.uncalibrated_runtime);
    assert!(
        results[0]
            .approximation_summary
            .top_codes
            .iter()
            .any(|count| count.name == "approximate_serving_event_loop" && count.count == 1)
    );
    assert!(
        results[0]
            .pareto
            .dimensions
            .iter()
            .any(|dimension| dimension.metric == "itl_s"
                && dimension.direction == "minimize"
                && dimension.unit == "seconds")
    );
    assert!(!results[0].bottleneck_summary.is_empty());
    assert!(
        results[0]
            .bottleneck_summary
            .iter()
            .any(|summary| summary.source == "memory_pressure"
                && summary.code == "peak_memory_pressure"
                && summary.observed.is_some_and(|observed| observed > 0.0))
    );
    assert!(
        results[0]
            .bottleneck_summary
            .iter()
            .any(|summary| summary.source == "phase_utilization"
                && summary.code == "hot_phase_resource"
                && summary.observed.is_some_and(|observed| observed > 0.0))
    );
    assert!(results[0].pool_label.is_some());
    assert!(results[0].metrics.ttft_s.is_finite());
    assert!(results[0].metrics.ttft_p90_s >= results[0].metrics.ttft_p50_s);
    assert!(results[0].metrics.ttft_p95_s >= results[0].metrics.ttft_p50_s);
    assert!(results[0].metrics.ttft_max_s >= results[0].metrics.ttft_p99_s);
    assert!(results[0].metrics.tpot_s.is_finite());
    assert!(results[0].metrics.tpot_p90_s >= results[0].metrics.tpot_p50_s);
    assert!(results[0].metrics.tpot_max_s >= results[0].metrics.tpot_p99_s);
    assert!(results[0].metrics.itl_s.is_finite());
    assert!(results[0].metrics.itl_p90_s >= results[0].metrics.itl_p50_s);
    assert!(results[0].metrics.itl_p95_s >= results[0].metrics.itl_p50_s);
    assert!(results[0].metrics.itl_max_s >= results[0].metrics.itl_p99_s);
    assert!(results[0].metrics.decode_iterations > 0);
    assert!(results[0].metrics.decode_iteration_s.is_finite());
    assert!(results[0].metrics.decode_iteration_p90_s >= results[0].metrics.decode_iteration_p50_s);
    assert!(results[0].metrics.decode_iteration_p95_s >= results[0].metrics.decode_iteration_p50_s);
    assert!(results[0].metrics.decode_iteration_max_s >= results[0].metrics.decode_iteration_p99_s);
    assert_eq!(results[0].metrics.ttft_slo_miss_rate, 0.0);
    assert_eq!(results[0].metrics.tpot_slo_miss_rate, 0.0);
    assert_eq!(results[0].metrics.itl_slo_miss_rate, 0.0);
    assert_eq!(results[0].metrics.e2el_slo_miss_rate, 0.0);
    assert!(results[0].metrics.e2el_p90_s >= results[0].metrics.e2el_p50_s);
    assert!(results[0].metrics.e2el_p95_s >= results[0].metrics.e2el_p50_s);
    assert!(results[0].metrics.e2el_max_s >= results[0].metrics.e2el_p99_s);
    assert!(results[0].metrics.queue_delay_p90_s >= results[0].metrics.queue_delay_s);
    assert!(results[0].metrics.queue_delay_max_s >= results[0].metrics.queue_delay_p95_s);
    assert!(results[0].metrics.throughput_tokens_per_s > 0.0);
    assert!(results[0].metrics.service_s.is_finite());
    assert!(results[0].metrics.prefill_s > 0.0);
    assert!(results[0].metrics.decode_s > 0.0);
    assert!(results[0].metrics.peak_prefill_tokens > 0);
    assert!(results[0].metrics.peak_prefill_tokens_per_node > 0);
    assert!(results[0].metrics.peak_prefill_tokens_per_gpu > 0);
    assert!(results[0].metrics.peak_decode_sequences > 0);
    assert!(results[0].metrics.peak_resident_tokens > 0);
    assert!(results[0].metrics.peak_kv_blocks > 0);
    assert!(results[0].metrics.peak_allocated_kv_tokens >= results[0].metrics.peak_resident_tokens);
    assert!(
        results[0].metrics.peak_kv_fragmentation_tokens
            <= results[0].metrics.peak_allocated_kv_tokens
    );
    assert!(results[0].metrics.peak_kv_block_table_bytes > 0);
    assert!(results[0].metrics.peak_decode_sequences_per_node > 0);
    assert!(results[0].metrics.peak_resident_tokens_per_node > 0);
    assert!(results[0].metrics.peak_kv_blocks_per_node > 0);
    assert!(results[0].metrics.decode_sequence_utilization > 0.0);
    assert!(results[0].metrics.resident_token_utilization > 0.0);
    assert!(results[0].metrics.decode_sequence_per_node_utilization > 0.0);
    assert!(results[0].metrics.resident_token_per_node_utilization > 0.0);
    assert!(results[0].metrics.peak_decode_sequences_per_gpu > 0);
    assert!(results[0].metrics.peak_resident_tokens_per_gpu > 0);
    assert!(results[0].metrics.peak_kv_blocks_per_gpu > 0);
    let disaggregated_score = results
        .iter()
        .find(|score| {
            score.feasible && score.deployment_mode == ServingDeploymentMode::FullyDisaggregated
        })
        .expect("feasible disaggregated score");
    assert!(disaggregated_score.approximation_summary.coarse_topology);
    assert_eq!(disaggregated_score.hardware_footprint.prefill_node_count, 1);
    assert_eq!(disaggregated_score.hardware_footprint.decode_node_count, 1);
    assert_eq!(disaggregated_score.hardware_footprint.unique_node_count, 2);
    assert_eq!(disaggregated_score.hardware_footprint.shared_node_count, 0);
    assert_eq!(disaggregated_score.hardware_footprint.shared_gpu_count, 0);
    assert!(disaggregated_score.hardware_footprint.prefill_gpu_count > 0);
    assert!(disaggregated_score.hardware_footprint.decode_gpu_count > 0);
    assert_eq!(
        disaggregated_score.hardware_footprint.prefill_gpu_types,
        vec![ServingGpuTypeCount {
            gpu: "H100 SXM5".to_string(),
            count: disaggregated_score.hardware_footprint.prefill_gpu_count,
        }]
    );
    assert_eq!(
        disaggregated_score.hardware_footprint.decode_gpu_types,
        vec![ServingGpuTypeCount {
            gpu: "H100 SXM5".to_string(),
            count: disaggregated_score.hardware_footprint.decode_gpu_count,
        }]
    );
    assert!(disaggregated_score.hardware_footprint.aggregate_hbm_gb >= 160.0);
    assert!(
        disaggregated_score
            .hardware_footprint
            .aggregate_peak_f16_tflops
            > 0.0
    );
    assert!(
        disaggregated_score
            .hardware_footprint
            .aggregate_effective_peak_tflops
            > 0.0
    );
    assert!(
        disaggregated_score
            .hardware_footprint
            .throughput_tokens_per_s_per_gpu
            > 0.0
    );
    assert!(
        disaggregated_score
            .memory_pressure
            .iter()
            .any(|pressure| pressure.phase == "prefill"
                && pressure.active_requests > 0
                && pressure.active_tokens > 0
                && pressure.duration_s > 0.0
                && pressure.capacity_used_fraction > 0.0
                && pressure.headroom_gb.is_finite()
                && pressure.dominant_component.is_some())
    );
    assert!(
        disaggregated_score
            .memory_pressure
            .iter()
            .any(|pressure| pressure.phase == "kv_transfer"
                && pressure.active_requests > 0
                && pressure.kv_blocks > 0
                && pressure.components.activations_gb == 0.0
                && pressure.components.temporary_gb == 0.0
                && pressure.limiting_gpu.is_some())
    );
    assert!(
        disaggregated_score
            .memory_pressure
            .iter()
            .any(|pressure| pressure.phase == "decode"
                && pressure.active_requests > 0
                && pressure.kv_blocks > 0
                && pressure.estimated_per_gpu_gb >= pressure.components.weights_gb
                && pressure.min_hbm_per_gpu_gb > 0.0)
    );
    assert!(
        results[0]
            .worker_observations
            .iter()
            .any(|worker| worker.phase == "prefill"
                && worker.request_count > 0
                && worker.peak_prefill_tokens > 0)
    );
    assert!(results[0].worker_observations.iter().any(|worker| {
        worker.phase == "decode"
            && worker.request_count > 0
            && worker.peak_decode_sequences > 0
            && worker.peak_resident_tokens > 0
            && worker.peak_kv_blocks > 0
            && !worker.kv_cache_owner_slots.is_empty()
            && worker
                .kv_cache_owner_slots
                .iter()
                .all(|slot| slot.peak_kv_blocks > 0)
    }));
    assert!(
        results[0].request_observations.iter().any(|observation| {
            observation.kv_block_ownership.len() == observation.decode_route_gpus.len()
                && observation.kv_block_ownership.iter().all(|ownership| {
                    observation.kv_cache_owner_gpus.contains(&ownership.owner)
                        && ownership.allocated_at_s == observation.kv_finish_s
                        && ownership.released_at_s <= observation.last_decode_finish_s
                        && ownership.kv_blocks > 0
                        && ownership.allocated_kv_tokens >= ownership.resident_tokens
                        && !ownership.owner_worker_slots.is_empty()
                        && !ownership.decode_operation_ids.is_empty()
                })
        }),
        "expected per-request decode-worker KV block ownership with worker slots"
    );
    assert!(
        results[0]
            .metric_breakdowns
            .iter()
            .any(|breakdown| { breakdown.group == "prefill_node" && breakdown.key == "node-0" })
    );
    assert!(
        results[0]
            .metric_breakdowns
            .iter()
            .any(|breakdown| { breakdown.group == "decode_node" && breakdown.key == "node-1" })
    );
    assert!(
        results[0]
            .metric_breakdowns
            .iter()
            .any(|breakdown| { breakdown.group == "prefill_route" && breakdown.key == "nodes[0]" })
    );
    assert!(
        results[0]
            .metric_breakdowns
            .iter()
            .any(|breakdown| { breakdown.group == "decode_route" && breakdown.key == "nodes[1]" })
    );
    assert!(
        results[0]
            .phase_resource_utilization
            .iter()
            .any(|row| { row.phase == "prefill" && row.resource_kind == "gpu_compute" })
    );
    assert!(
        results[0]
            .phase_resource_utilization
            .iter()
            .any(|row| { row.phase == "decode" && row.resource_kind == "gpu_compute" })
    );
    assert!(results[0].prefill_memory.estimated_per_gpu_gb > 0.0);
    assert!(results[0].prefill_memory.min_hbm_per_gpu_gb >= 80.0);
    assert_eq!(
        results[0].prefill_memory.limiting_gpu,
        Some(GpuAddr {
            node_id: 0,
            local_gpu_id: 0,
        })
    );
    assert!(results[0].prefill_memory.capacity_used_fraction() > 0.0);
    let prefill_dominant_component = results[0]
        .prefill_memory
        .dominant_component()
        .expect("prefill memory dominant component");
    assert_eq!(prefill_dominant_component.name, "weights");
    assert!(prefill_dominant_component.gb > 0.0);
    assert!(prefill_dominant_component.fraction_of_total > 0.0);
    assert!(results[0].prefill_memory.headroom_gb > 0.0);
    assert!(results[0].prefill_memory.components.weights_gb > 0.0);
    assert!(results[0].prefill_memory.components.kv_cache_gb > 0.0);
    assert!(results[0].prefill_memory.components.block_table_gb > 0.0);
    assert!(results[0].prefill_memory.components.activations_gb > 0.0);
    assert_eq!(
        results[0].prefill_memory.estimated_per_gpu_gb,
        results[0].prefill_memory.components.total_gb
    );
    assert!(results[0].decode_memory.estimated_per_gpu_gb > 0.0);
    assert!(results[0].decode_memory.min_hbm_per_gpu_gb >= 80.0);
    assert_eq!(
        results[0].decode_memory.limiting_gpu,
        Some(GpuAddr {
            node_id: 0,
            local_gpu_id: 0,
        })
    );
    assert!(results[0].decode_memory.capacity_used_fraction() > 0.0);
    let decode_dominant_component = results[0]
        .decode_memory
        .dominant_component()
        .expect("decode memory dominant component");
    assert_eq!(decode_dominant_component.name, "weights");
    assert!(decode_dominant_component.gb > 0.0);
    assert!(decode_dominant_component.fraction_of_total > 0.0);
    assert!(results[0].decode_memory.headroom_fraction > 0.0);
    assert!(results[0].decode_memory.components.weights_gb > 0.0);
    assert!(results[0].decode_memory.components.kv_cache_gb > 0.0);
    assert!(results[0].decode_memory.components.block_table_gb > 0.0);
    assert!(results[0].decode_memory.components.activations_gb > 0.0);
    assert_eq!(
        results[0].decode_memory.estimated_per_gpu_gb,
        results[0].decode_memory.components.total_gb
    );
    assert_eq!(results[0].request_observations.len(), 6);
    assert_eq!(results[0].request_observations[0].request_idx, 0);
    assert!(results[0].request_observations[0].prefill_node <= 1);
    assert!(results[0].request_observations[0].decode_node <= 1);
    assert!(results[0].request_observations[0].ttft_s > 0.0);
    assert!(results[0].request_observations[0].itl_s.is_finite());
    assert!(results[0].request_observations[0].decode_s > 0.0);
    assert_eq!(
        results[0].request_observations[0].decode_iterations,
        results[0].request_observations[0].decode_tokens
    );
    assert_eq!(
        results[0].request_observations[0]
            .decode_token_start_s
            .len(),
        results[0].request_observations[0].decode_tokens as usize
    );
    assert_eq!(
        results[0].request_observations[0]
            .decode_token_finish_s
            .len(),
        results[0].request_observations[0].decode_tokens as usize
    );
    assert_eq!(
        results[0].request_observations[0]
            .inter_token_latency_s
            .len(),
        results[0].request_observations[0]
            .decode_tokens
            .saturating_sub(1) as usize
    );
    assert!(
        results[0].request_observations.iter().any(|observation| {
            observation
                .worker_assignments
                .iter()
                .any(|assignment| assignment.phase == "prefill")
                && observation
                    .worker_assignments
                    .iter()
                    .any(|assignment| assignment.phase == "decode")
                && observation.worker_assignments.iter().all(|assignment| {
                    assignment.finish_s >= assignment.start_s
                        && !assignment.operation_ids.is_empty()
                })
        }),
        "expected per-request prefill/decode worker slot assignments"
    );
    assert!(
        results[0].request_observations.iter().any(|observation| {
            let has_prefill_source = observation.worker_summary.iter().any(|summary| {
                summary.role == "prefill_source"
                    && summary.phase == "prefill"
                    && !summary.worker_slots.is_empty()
                    && !summary.operation_ids.is_empty()
            });
            let has_decode_owner = observation.worker_summary.iter().any(|summary| {
                summary.role == "decode_owner"
                    && summary.phase == "decode"
                    && !summary.worker_slots.is_empty()
                    && !summary.operation_ids.is_empty()
            });
            let has_cache_owner = observation.worker_summary.iter().any(|summary| {
                summary.role == "kv_cache_owner"
                    && summary.phase == "decode"
                    && summary.resident_tokens > 0
                    && summary.kv_blocks > 0
                    && !summary.worker_slots.is_empty()
                    && !summary.operation_ids.is_empty()
            });
            has_prefill_source && has_decode_owner && has_cache_owner
        }),
        "expected compact per-request worker/cache owner summary"
    );
    assert!(!results[0].decode_iterations.is_empty());
    assert!(!results[0].node_capacity.is_empty());
    assert!(
        results[0]
            .node_capacity
            .iter()
            .any(|node| node.peak_prefill_tokens > 0)
    );
    assert!(!results[0].gpu_capacity.is_empty());
    assert!(
        results[0]
            .gpu_capacity
            .iter()
            .any(|gpu| gpu.peak_prefill_tokens > 0)
    );
    assert!(!results[0].resource_utilization.is_empty());
    assert!(results[0].resource_utilization[0].utilization > 0.0);
    assert!(
        results[0]
            .approximations
            .iter()
            .any(|approximation| { approximation.code == "approximate_serving_event_loop" })
    );
    assert!(
        results[0]
            .approximations
            .iter()
            .any(|approximation| { approximation.code == "serving_stack_uncalibrated" })
    );
    assert!(results[0].phase_calibration.iter().any(|phase| {
        phase.phase == "prefill"
            && phase.active
            && !phase.calibrated
            && phase.status == "uncalibrated_no_profile"
    }));
    assert!(
        results
            .iter()
            .flat_map(|result| result.approximations.iter())
            .any(|approximation| { approximation.code == "node_set_kv_handoff" })
    );
    assert!(
        results[0]
            .approximations
            .iter()
            .any(|approximation| { approximation.code == "approximate_kv_residency_accounting" })
    );
    assert_eq!(results[0].metrics.scheduled_requests, 6);
    assert!(results[0].metrics.scheduled_makespan_s >= results[0].metrics.e2el_s);

    let bounded = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            max_prefill_candidates: Some(1),
            max_decode_candidates: Some(2),
            max_serving_pairs: Some(3),
            ..ServingSolverOptions::default()
        },
    );

    assert_eq!(bounded.len(), 3);

    let runtime_profile = CalibrationProfileMetadata {
        path: "in-memory".to_string(),
        name: Some("runtime-profile".to_string()),
        hardware: Some("h100_sxm".to_string()),
        fabric: Some("ib_ndr".to_string()),
        model: Some("test-model".to_string()),
        dtype: Some("bf16".to_string()),
        serving_stack: Some("vllm".to_string()),
        serving_runtime_features: vec!["paged_attention".to_string(), "cuda_graphs".to_string()],
        backend_version: None,
        driver_version: None,
        cuda_version: None,
        rocm_version: None,
        nccl_version: None,
        rccl_version: None,
        ucx_version: None,
        kernel_settings: Vec::new(),
        environment_hash: None,
        source: Some("unit-test".to_string()),
        date: Some("2026-05-26".to_string()),
        notes: None,
        valid_shape: None,
        invalid_shapes: Vec::new(),
        fits: Vec::new(),
        benchmarks: Vec::new(),
    };
    let profiled = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&runtime_profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(!profiled[0].approximations.iter().any(|approximation| {
        approximation.code == "serving_stack_uncalibrated"
            || approximation.code == "serving_stack_unspecified"
            || approximation.code == "calibration_profile_hardware_mismatch"
            || approximation.code == "calibration_profile_hardware_unspecified"
            || approximation.code == "calibration_profile_fabric_mismatch"
            || approximation.code == "calibration_profile_fabric_unspecified"
            || approximation.code == "calibration_profile_model_mismatch"
            || approximation.code == "calibration_profile_model_unspecified"
            || approximation.code == "calibration_profile_model_unverified"
            || approximation.code == "calibration_profile_provenance_incomplete"
            || approximation.code == "calibration_profile_dtype_mismatch"
            || approximation.code == "calibration_profile_dtype_unspecified"
    }));
    assert!(profiled[0].approximations.iter().any(|approximation| {
        approximation.phase == "prefill" && approximation.code == "serving_phase_uncalibrated"
    }));
    assert!(profiled[0].approximations.iter().any(|approximation| {
        approximation.phase == "decode" && approximation.code == "serving_phase_uncalibrated"
    }));
    assert!(profiled[0].approximations.iter().any(|approximation| {
        approximation.phase == "kv_transfer" && approximation.code == "serving_phase_uncalibrated"
    }));
    assert!(profiled[0].phase_calibration.iter().any(|phase| {
        phase.phase == "prefill"
            && phase.active
            && !phase.calibrated
            && phase.fit_count == 0
            && phase.status == "uncalibrated_no_fit"
    }));
    assert!(profiled[0].phase_calibration.iter().any(|phase| {
        phase.phase == "kv_transfer"
            && phase.active
            && !phase.calibrated
            && phase.fit_count == 0
            && phase.status == "uncalibrated_no_fit"
    }));

    let mut mismatched_profile = runtime_profile.clone();
    mismatched_profile.dtype = Some("fp16".to_string());
    let dtype_mismatch = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&mismatched_profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        dtype_mismatch[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_dtype_mismatch"
                    && approximation
                        .message
                        .contains("profile declares dtype 'fp16'")
                    && approximation
                        .message
                        .contains("workload model dtype is 'bf16'")
            })
    );

    let stack_mismatch = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&runtime_profile),
            model_id: Some("test-model"),
            serving_stack: Some("tensorrt-llm"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        stack_mismatch[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_serving_stack_mismatch"
                    && approximation.message.contains("serving_stack 'vllm'")
                    && approximation.message.contains("requests 'tensorrt-llm'")
            })
    );

    let covered_runtime_features = vec!["paged_attention".to_string(), "cuda_graphs".to_string()];
    let runtime_feature_covered = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&runtime_profile),
            model_id: Some("test-model"),
            serving_stack: Some("vllm"),
            serving_runtime_features: Some(&covered_runtime_features),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        !runtime_feature_covered[0]
            .approximations
            .iter()
            .any(|approximation| approximation.code.contains("runtime_feature"))
    );

    let missing_runtime_features = vec!["speculative_decode".to_string()];
    let runtime_feature_mismatch = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&runtime_profile),
            model_id: Some("test-model"),
            serving_stack: Some("vllm"),
            serving_runtime_features: Some(&missing_runtime_features),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        runtime_feature_mismatch[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_runtime_feature_mismatch"
                    && approximation.scope == "runtime_feature:speculative_decode"
                    && approximation.message.contains("speculative_decode")
            })
    );

    let mut feature_unspecified_profile = runtime_profile.clone();
    feature_unspecified_profile.serving_runtime_features.clear();
    let runtime_feature_unspecified = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&feature_unspecified_profile),
            model_id: Some("test-model"),
            serving_stack: Some("vllm"),
            serving_runtime_features: Some(&covered_runtime_features),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        runtime_feature_unspecified[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_runtime_feature_unspecified"
                    && approximation.scope == "runtime_feature:paged_attention"
            })
    );

    let mut incomplete_provenance_profile = runtime_profile.clone();
    incomplete_provenance_profile.source = None;
    incomplete_provenance_profile.date = None;
    let incomplete_provenance = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&incomplete_provenance_profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        incomplete_provenance[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_provenance_incomplete"
                    && approximation.message.contains("source and date")
            })
    );

    let mut weak_fit_profile = runtime_profile.clone();
    weak_fit_profile.fits = vec![CalibrationFittedModel {
        name: Some("weak-prefill-fit".to_string()),
        target: "prefill_ms".to_string(),
        phase: Some("prefill".to_string()),
        kind: Some("serving".to_string()),
        model: "linear".to_string(),
        unit: Some("ms".to_string()),
        intercept: Some(5.0),
        features: vec!["batch_size".to_string()],
        coefficients: vec![0.0],
        feature_ranges: vec![CalibrationFitFeatureRange {
            feature: "batch_size".to_string(),
            min: Some(1.0),
            max: Some(8.0),
        }],
        r_squared: Some(1.0),
        adjusted_r_squared: None,
        rmse: None,
        rmse_pct: None,
        mean_abs_pct_error: None,
        max_abs_pct_error: None,
        validation_rmse: None,
        validation_rmse_pct: None,
        validation_mean_abs_pct_error: None,
        validation_max_abs_pct_error: None,
        confidence_interval: None,
        confidence_interval_pct: None,
        confidence_level: None,
        sample_count: None,
        validation_sample_count: None,
        source: None,
        notes: None,
    }];
    let weak_fit_metadata = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&weak_fit_profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        weak_fit_metadata[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "serving_calibration_fit_sample_count_unspecified"
            })
    );
    assert!(
        weak_fit_metadata[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "serving_calibration_fit_holdout_unspecified"
            })
    );
    assert!(
        weak_fit_metadata[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "serving_calibration_fit_source_unspecified"
            })
    );
    assert!(
        weak_fit_metadata[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "serving_calibration_fit_uncertainty_unspecified"
            })
    );
    assert_eq!(
        weak_fit_metadata[0].calibration_fits[0].fit_name.as_deref(),
        Some("weak-prefill-fit")
    );
    assert_eq!(weak_fit_metadata[0].calibration_fits[0].sample_count, None);
    assert_eq!(
        weak_fit_metadata[0].calibration_fits[0].validation_sample_count,
        None
    );

    let mut model_mismatch_profile = runtime_profile.clone();
    model_mismatch_profile.model = Some("llama-70b".to_string());
    let model_mismatch = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&model_mismatch_profile),
            model_id: Some("mixtral-8x7b"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        model_mismatch[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_model_mismatch"
                    && approximation.message.contains("llama-70b")
                    && approximation.message.contains("mixtral-8x7b")
            })
    );

    let model_unverified = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&runtime_profile),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        model_unverified[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_model_unverified"
                    && approximation.message.contains("test-model")
            })
    );

    let mut mixed_model_serving = serving.clone();
    mixed_model_serving.traffic.request_count = Some(2);
    mixed_model_serving.traffic.trace_requests = vec![
        ServingTraceRequest {
            request_id: Some("trace-a".to_string()),
            tenant: None,
            model_id: Some("test-model".to_string()),
            cache_key: None,
            arrival_s: 0.0,
            priority: 0,
            batch_size: 1,
            prompt_tokens: 128,
            decode_tokens: 4,
            max_sequence_tokens: None,
            prefix_cache_hit_tokens: None,
            prefix_cache_hit_rate: None,
            slo: ServingRequestSlo::default(),
            deadline_s: None,
            cancellation_s: None,
        },
        ServingTraceRequest {
            request_id: Some("trace-b".to_string()),
            tenant: None,
            model_id: Some("other-model".to_string()),
            cache_key: None,
            arrival_s: 0.01,
            priority: 0,
            batch_size: 1,
            prompt_tokens: 128,
            decode_tokens: 4,
            max_sequence_tokens: None,
            prefix_cache_hit_tokens: None,
            prefix_cache_hit_rate: None,
            slo: ServingRequestSlo::default(),
            deadline_s: None,
            cancellation_s: None,
        },
    ];
    let mixed_trace_model_profile = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &mixed_model_serving,
        ServingSolverOptions {
            calibration_profile: Some(&runtime_profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        mixed_trace_model_profile[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_multi_model_trace"
                    && approximation.message.contains("test-model")
                    && approximation.message.contains("other-model")
            })
    );
    assert!(
        mixed_trace_model_profile[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_trace_model_mismatch"
                    && approximation.message.contains("other-model")
            })
    );

    let mut topology_mismatch_profile = runtime_profile.clone();
    topology_mismatch_profile.hardware = Some("a100_pcie".to_string());
    topology_mismatch_profile.fabric = Some("ethernet_10g".to_string());
    let topology_mismatch = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&topology_mismatch_profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        topology_mismatch[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_hardware_mismatch"
                    && approximation.message.contains("a100_pcie")
                    && approximation.message.contains("H100")
            })
    );
    assert!(
        topology_mismatch[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_fabric_mismatch"
                    && approximation.message.contains("ethernet_10g")
                    && approximation.message.contains("IB NDR")
            })
    );

    let mut incomplete_profile = runtime_profile;
    incomplete_profile.hardware = None;
    incomplete_profile.fabric = None;
    incomplete_profile.model = None;
    let incomplete_metadata = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&incomplete_profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!(
        incomplete_metadata[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_hardware_unspecified"
            })
    );
    assert!(
        incomplete_metadata[0]
            .approximations
            .iter()
            .any(|approximation| {
                approximation.code == "calibration_profile_fabric_unspecified"
            })
    );
    assert!(
        incomplete_metadata[0]
            .approximations
            .iter()
            .any(|approximation| { approximation.code == "calibration_profile_model_unspecified" })
    );
}

#[test]
fn serving_pool_gpu_labels_constrain_prefill_and_decode_placements() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[interconnect.links]]
        from = 0
        to = 1
        from_gpu_tag = "prefill-fast"
        to_gpu_tag = "decode-fast"
        kind = "ib"
        variant = "ndr"

        [[nodes]]
        id = 0
        node_tags = ["prefill-pool"]
        rack = "rack-a"
        island = "island-a"
        failure_domain = "az-a"
        intra = "pcie_gen5"
        gpus = [
          { start_id = 0, count = 2, gpu = "h100_sxm", labels = ["prefill-fast"] },
          { start_id = 2, count = 2, gpu = "a100_80gb", labels = ["prefill-capacity"] },
        ]
        nics = { count = 4, affinity = "dedicated", rail_count = 4 }

        [[nodes]]
        id = 1
        node_tags = ["decode-pool"]
        rack = "rack-b"
        island = "island-b"
        failure_domain = "az-b"
        intra = "pcie_gen5"
        gpus = [
          { start_id = 0, count = 2, gpu = "h100_sxm", labels = ["decode-fast"] },
          { start_id = 2, count = 2, gpu = "a100_80gb", labels = ["decode-capacity"] },
        ]
        nics = { count = 4, affinity = "dedicated", rail_count = 4 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![2],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::FullyDisaggregated,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: vec![ServingPoolCandidate {
            label: None,
            prefill_nodes: vec![0],
            decode_nodes: vec![1],
            prefill_groups: Vec::new(),
            decode_groups: Vec::new(),
            prefill_node_filter: ServingPoolNodeFilter::default(),
            decode_node_filter: ServingPoolNodeFilter::default(),
            domain_spread: ServingPoolDomainSpread::default(),
            prefill_gpu_labels: vec!["prefill_fast".to_string()],
            decode_gpu_labels: vec!["decode_fast".to_string()],
        }],
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );
    let best = results
        .iter()
        .find(|score| score.feasible)
        .expect("feasible label-constrained serving result");

    assert_eq!(best.prefill_gpu_labels, vec!["prefill_fast"]);
    assert_eq!(best.decode_gpu_labels, vec!["decode_fast"]);
    assert_eq!(best.pool_topology.prefill_node_count, 1);
    assert_eq!(best.pool_topology.decode_node_count, 1);
    assert_eq!(best.pool_topology.shared_node_count, 0);
    assert_eq!(best.pool_topology.dedicated_prefill_node_count, 1);
    assert_eq!(best.pool_topology.dedicated_decode_node_count, 1);
    assert_eq!(best.pool_topology.prefill_racks, vec!["rack_a"]);
    assert_eq!(best.pool_topology.decode_racks, vec!["rack_b"]);
    assert_eq!(best.pool_topology.prefill_islands, vec!["island_a"]);
    assert_eq!(best.pool_topology.decode_islands, vec!["island_b"]);
    assert_eq!(best.pool_topology.prefill_failure_domains, vec!["az_a"]);
    assert_eq!(best.pool_topology.decode_failure_domains, vec!["az_b"]);
    assert_eq!(best.pool_topology.prefill_node_labels, vec!["prefill_pool"]);
    assert_eq!(best.pool_topology.decode_node_labels, vec!["decode_pool"]);
    assert_eq!(
        best.hardware_footprint.prefill_gpu_label_counts,
        vec![ServingGpuLabelCount {
            label: "prefill_fast".to_string(),
            count: 2,
        }]
    );
    assert_eq!(
        best.hardware_footprint.decode_gpu_label_counts,
        vec![ServingGpuLabelCount {
            label: "decode_fast".to_string(),
            count: 2,
        }]
    );
    assert!(
        best.prefill_score
            .placement
            .rank_to_gpu
            .iter()
            .all(|gpu| { gpu.node_id == 0 && gpu.local_gpu_id < 2 })
    );
    assert!(
        best.decode_score
            .placement
            .rank_to_gpu
            .iter()
            .all(|gpu| { gpu.node_id == 1 && gpu.local_gpu_id < 2 })
    );
    assert_eq!(best.route_coverage.routable_candidate_count, 1);
}

#[test]
fn serving_pool_topology_summary_reports_partial_overlap() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[nodes]]
        id = 0
        node_tags = ["prefill-only"]
        rack = "rack-a"
        island = "island-a"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 1
        node_tags = ["shared"]
        rack = "rack-b"
        island = "island-b"
        failure_domain = "az-b"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 2
        node_tags = ["decode-only"]
        rack = "rack-c"
        island = "island-c"
        failure_domain = "az-c"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }
        "#,
    )
    .unwrap();

    let summary = serving_pool_topology_summary(Some(&cluster), &[0, 1], &[1, 2]);

    assert_eq!(summary.prefill_node_count, 2);
    assert_eq!(summary.decode_node_count, 2);
    assert_eq!(summary.shared_node_count, 1);
    assert_eq!(summary.dedicated_prefill_node_count, 1);
    assert_eq!(summary.dedicated_decode_node_count, 1);
    assert_eq!(summary.prefill_racks, vec!["rack_a", "rack_b"]);
    assert_eq!(summary.decode_racks, vec!["rack_b", "rack_c"]);
    assert_eq!(summary.prefill_node_labels, vec!["prefill_only", "shared"]);
    assert_eq!(summary.decode_node_labels, vec!["decode_only", "shared"]);
}

#[test]
fn serving_pool_search_filters_nodes_by_gpu_labels() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[nodes]]
        id = 0
        group = "prefill"
        intra = "pcie_gen5"
        gpu = "a100_80gb"
        gpu_count = 2
        gpu_tags = ["prefill-capacity"]
        nics = { count = 2, affinity = "dedicated", rail_count = 2 }

        [[nodes]]
        id = 1
        group = "prefill"
        intra = "pcie_gen5"
        gpu = "h100_sxm"
        gpu_count = 2
        gpu_tags = ["prefill-fast"]
        nics = { count = 2, affinity = "dedicated", rail_count = 2 }

        [[nodes]]
        id = 2
        group = "decode"
        intra = "pcie_gen5"
        gpu = "h100_sxm"
        gpu_count = 2
        gpu_tags = ["decode-fast"]
        nics = { count = 2, affinity = "dedicated", rail_count = 2 }

        [[nodes]]
        id = 3
        group = "decode"
        intra = "pcie_gen5"
        gpu = "a100_80gb"
        gpu_count = 2
        gpu_tags = ["decode-capacity"]
        nics = { count = 2, affinity = "dedicated", rail_count = 2 }
        "#,
    )
    .unwrap();
    let search = ServingPoolSearch {
        prefill_groups: vec!["prefill".to_string()],
        decode_groups: vec!["decode".to_string()],
        prefill_node_counts: vec![1],
        decode_node_counts: vec![1],
        prefill_node_filter: ServingPoolNodeFilter::default(),
        decode_node_filter: ServingPoolNodeFilter::default(),
        prefill_gpu_labels: vec!["prefill_fast".to_string()],
        decode_gpu_labels: vec!["decode_fast".to_string()],
        allow_overlap: false,
        domain_spread: ServingPoolDomainSpread::default(),
        max_candidates: 8,
    };

    let generated = generate_pool_search_candidates_with_summary(
        &cluster,
        &search,
        ServingDeploymentMode::FullyDisaggregated,
    )
    .unwrap();
    let candidates = generated.candidates;

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].prefill_nodes, vec![1]);
    assert_eq!(candidates[0].decode_nodes, vec![2]);
    assert_eq!(candidates[0].prefill_gpu_labels, vec!["prefill_fast"]);
    assert_eq!(candidates[0].decode_gpu_labels, vec!["decode_fast"]);
}

#[test]
fn serving_pool_candidate_filters_by_topology_node_selectors() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[nodes]]
        id = 0
        group = "prefill"
        node_tags = ["fast"]
        rack = "rack-a"
        island = "island-a"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 1
        group = "prefill"
        node_tags = ["slow"]
        rack = "rack-b"
        island = "island-b"
        failure_domain = "az-a"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 2
        group = "decode"
        node_tags = ["decode"]
        rack = "rack-c"
        island = "island-c"
        failure_domain = "az-b"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 3
        group = "decode"
        node_tags = ["decode"]
        rack = "rack-d"
        island = "island-d"
        failure_domain = "az-c"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }
        "#,
    )
    .unwrap();
    let candidate = ServingPoolCandidate {
        label: Some("filtered".to_string()),
        prefill_nodes: Vec::new(),
        decode_nodes: Vec::new(),
        prefill_groups: vec!["prefill".to_string()],
        decode_groups: vec!["decode".to_string()],
        prefill_node_filter: ServingPoolNodeFilter {
            node_labels: vec!["fast".to_string()],
            racks: vec!["rack_a".to_string()],
            exclude_islands: vec!["island_b".to_string()],
            ..ServingPoolNodeFilter::default()
        },
        decode_node_filter: ServingPoolNodeFilter {
            exclude_failure_domains: vec!["az_c".to_string()],
            ..ServingPoolNodeFilter::default()
        },
        domain_spread: ServingPoolDomainSpread {
            min_prefill_racks: Some(1),
            min_decode_failure_domains: Some(1),
            ..ServingPoolDomainSpread::default()
        },
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
    };
    let mut invalid = candidate.clone();
    invalid.domain_spread.min_decode_failure_domains = Some(2);
    let err = resolve_pool_candidate(&cluster, &invalid).unwrap_err();
    assert!(err.contains("does not satisfy configured topology-domain spread constraints"));

    let resolved = resolve_pool_candidate(&cluster, &candidate).unwrap();

    assert_eq!(resolved.prefill_nodes, vec![0]);
    assert_eq!(resolved.decode_nodes, vec![2]);
    assert_eq!(resolved.label.as_deref(), Some("filtered"));
}

#[test]
fn serving_pool_search_filters_by_topology_node_selectors() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[nodes]]
        id = 0
        group = "prefill"
        node_tags = ["pool-a"]
        rack = "rack-a"
        island = "island-a"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 1
        group = "prefill"
        node_tags = ["pool-b"]
        rack = "rack-b"
        island = "island-b"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 2
        group = "prefill"
        node_tags = ["pool-a"]
        rack = "rack-c"
        island = "island-a"
        failure_domain = "az-c"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 3
        group = "decode"
        node_tags = ["decode-west"]
        rack = "rack-d"
        island = "island-d"
        failure_domain = "az-d"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 4
        group = "decode"
        node_tags = ["decode-east"]
        rack = "rack-e"
        island = "island-e"
        failure_domain = "az-e"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }
        "#,
    )
    .unwrap();
    let search = ServingPoolSearch {
        prefill_groups: vec!["prefill".to_string()],
        decode_groups: vec!["decode".to_string()],
        prefill_node_counts: vec![1],
        decode_node_counts: vec![1],
        prefill_node_filter: ServingPoolNodeFilter {
            node_labels: vec!["pool_a".to_string()],
            racks: vec!["rack_a".to_string(), "rack_c".to_string()],
            exclude_failure_domains: vec!["az_c".to_string()],
            ..ServingPoolNodeFilter::default()
        },
        decode_node_filter: ServingPoolNodeFilter {
            exclude_node_labels: vec!["decode_west".to_string()],
            ..ServingPoolNodeFilter::default()
        },
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
        allow_overlap: false,
        domain_spread: ServingPoolDomainSpread::default(),
        max_candidates: 8,
    };

    let generated = generate_pool_search_candidates_with_summary(
        &cluster,
        &search,
        ServingDeploymentMode::FullyDisaggregated,
    )
    .unwrap();
    let candidates = &generated.candidates;

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].prefill_nodes, vec![0]);
    assert_eq!(candidates[0].decode_nodes, vec![4]);
    assert_eq!(generated.summary.generated_candidate_count, 1);
    assert_eq!(generated.summary.considered_candidate_count(), 1);
    assert_eq!(generated.summary.groups.len(), 1);
    assert_eq!(generated.summary.groups[0].prefill_group_node_count, 3);
    assert_eq!(
        generated.summary.groups[0].prefill_node_filter_node_count,
        1
    );
    assert_eq!(generated.summary.groups[0].decode_group_node_count, 2);
    assert_eq!(generated.summary.groups[0].decode_node_filter_node_count, 1);
}

#[test]
fn serving_pool_search_summary_reports_generated_deployment_modes() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = ServingPoolSearch {
        prefill_groups: vec!["h100".to_string()],
        decode_groups: vec!["h100".to_string()],
        prefill_node_counts: vec![1, 2],
        decode_node_counts: vec![1],
        prefill_node_filter: ServingPoolNodeFilter::default(),
        decode_node_filter: ServingPoolNodeFilter::default(),
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
        allow_overlap: true,
        domain_spread: ServingPoolDomainSpread::default(),
        max_candidates: 16,
    };

    let generated = generate_pool_search_candidates_with_summary(
        &cluster,
        &search,
        ServingDeploymentMode::Flexible,
    )
    .unwrap();

    assert_eq!(generated.candidates.len(), 6);
    assert_eq!(generated.summary.generated_candidate_count, 6);
    assert_eq!(generated.summary.generated_colocated_count(), 2);
    assert_eq!(
        generated.summary.generated_partially_disaggregated_count(),
        2
    );
    assert_eq!(generated.summary.generated_fully_disaggregated_count(), 2);
    assert_eq!(generated.summary.groups.len(), 1);
    assert_eq!(generated.summary.groups[0].generated_colocated_count, 2);
    assert_eq!(
        generated.summary.groups[0].generated_partially_disaggregated_count,
        2
    );
    assert_eq!(
        generated.summary.groups[0].generated_fully_disaggregated_count,
        2
    );
}

#[test]
fn serving_pool_search_filters_by_topology_domain_spread() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[nodes]]
        id = 0
        group = "prefill"
        rack = "rack-a"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 1
        group = "prefill"
        rack = "rack-b"
        failure_domain = "az-b"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 2
        group = "prefill"
        rack = "rack-a"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 3
        group = "decode"
        rack = "rack-c"
        failure_domain = "az-c"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }
        "#,
    )
    .unwrap();
    let search = ServingPoolSearch {
        prefill_groups: vec!["prefill".to_string()],
        decode_groups: vec!["decode".to_string()],
        prefill_node_counts: vec![2],
        decode_node_counts: vec![1],
        prefill_node_filter: ServingPoolNodeFilter::default(),
        decode_node_filter: ServingPoolNodeFilter::default(),
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
        allow_overlap: false,
        domain_spread: ServingPoolDomainSpread {
            min_prefill_racks: Some(2),
            min_prefill_failure_domains: Some(2),
            ..ServingPoolDomainSpread::default()
        },
        max_candidates: 8,
    };

    let generated = generate_pool_search_candidates_with_summary(
        &cluster,
        &search,
        ServingDeploymentMode::FullyDisaggregated,
    )
    .unwrap();
    let candidates = generated.candidates;

    assert_eq!(candidates.len(), 2);
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.prefill_nodes.contains(&1))
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.prefill_nodes != vec![0, 2])
    );
}

#[test]
fn aggregate_serving_metric_fits_adjust_reported_latency_metrics() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.001),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![4],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };
    let raw = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );
    assert!(raw[0].metrics.e2el_s.is_finite());

    let profile = calibration_profile_with_fits(vec![
        constant_serving_metric_fit("ttft_ms", 11.0),
        constant_serving_metric_fit("tpot_ms", 7.0),
        constant_serving_metric_fit("e2el_ms", 123.0),
        constant_serving_throughput_fit(321.0),
    ]);
    let fitted = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&profile),
            model_id: Some("test-model"),
            max_serving_pairs: Some(1),
            ..ServingSolverOptions::default()
        },
    );

    assert!((fitted[0].metrics.ttft_s - 0.011).abs() < 1e-12);
    assert!((fitted[0].metrics.tpot_s - 0.007).abs() < 1e-12);
    assert!((fitted[0].metrics.e2el_s - 0.123).abs() < 1e-12);
    assert!((fitted[0].metrics.throughput_tokens_per_s - 321.0).abs() < 1e-12);
    assert!(fitted[0].calibration_fits.iter().any(|fit| {
        fit.phase == "serving"
            && fit.target == "e2el_ms"
            && fit.baseline_s == Some(raw[0].metrics.e2el_s)
            && fit
                .features
                .iter()
                .any(|feature| feature.name == "baseline_ms" && feature.value > 0.0)
    }));
    assert!(fitted[0].calibration_fits.iter().any(|fit| {
        fit.phase == "serving"
            && fit.target == "throughput_tokens_per_s"
            && fit.prediction_kind == "throughput"
            && fit.prediction_unit.as_deref() == Some("tokens/s")
            && fit.predicted_value == 321.0
            && fit.baseline_value == Some(raw[0].metrics.throughput_tokens_per_s)
            && fit.predicted_s == 0.0
    }));
    assert_eq!(fitted[0].calibration_summary.fit_count, 4);
    assert_eq!(fitted[0].calibration_summary.fit_count_with_uncertainty, 4);
    assert!(fitted[0].approximations.iter().any(|approximation| {
        approximation.code == "aggregate_serving_metric_fit_not_trace_replay"
    }));
}

#[test]
fn serving_mode_rejects_incompatible_pool() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::FullyDisaggregated,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    assert_eq!(results[0].deployment_mode, ServingDeploymentMode::Colocated);
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("serving.mode 'fully_disaggregated'"))
    );
}

#[test]
fn prefix_cache_hits_reduce_prefill_work() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            trace_requests: vec![
                ServingTraceRequest {
                    request_id: Some("cold".to_string()),
                    tenant: None,
                    model_id: None,
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 2,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
                ServingTraceRequest {
                    request_id: Some("cached".to_string()),
                    tenant: None,
                    model_id: None,
                    cache_key: Some("shared-prefix".to_string()),
                    arrival_s: 0.1,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 2,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: Some(96),
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
            ],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    let cold = &score.request_observations[0];
    let cached = &score.request_observations[1];
    assert_eq!(cold.prompt_tokens, 128);
    assert_eq!(cold.prefix_cache_hit_tokens, 0);
    assert_eq!(cold.effective_prefill_tokens, 128);
    assert_eq!(cached.cache_key.as_deref(), Some("shared-prefix"));
    assert_eq!(cached.prompt_tokens, 128);
    assert_eq!(cached.prefix_cache_hit_tokens, 96);
    assert_eq!(cached.effective_prefill_tokens, 32);
    assert!(cached.prefill_s < cold.prefill_s);
    assert_eq!(score.metrics.prompt_tokens, 256);
    assert_eq!(score.metrics.prefix_cache_hit_tokens, 96);
    assert_eq!(score.metrics.effective_prefill_tokens, 160);
    assert!((score.metrics.prefix_cache_hit_rate - 0.375).abs() < 1e-12);
}

#[test]
fn serving_objective_changes_result_ordering() {
    let pool = ResolvedServingPool {
        label: None,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
    };
    let mut low_latency = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights::default(),
            0.0,
        ),
        "test".to_string(),
    );
    low_latency.feasible = true;
    low_latency.metrics = finite_metrics();
    low_latency.metrics.e2el_s = 1.0;
    low_latency.metrics.throughput_tokens_per_s = 10.0;
    low_latency.memory_pressure = vec![memory_pressure_observation("decode", 0.80)];
    low_latency.pool_label = Some("low-latency".to_string());

    let mut high_throughput = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights::default(),
            0.0,
        ),
        "test".to_string(),
    );
    high_throughput.feasible = true;
    high_throughput.metrics = finite_metrics();
    high_throughput.metrics.e2el_s = 2.0;
    high_throughput.metrics.throughput_tokens_per_s = 20.0;
    high_throughput.memory_pressure = vec![memory_pressure_observation("decode", 0.40)];
    high_throughput.pool_label = Some("high-throughput".to_string());

    let mut e2el_results = vec![high_throughput.clone(), low_latency.clone()];
    sort_serving_results(&mut e2el_results, ServingObjective::MinimizeE2el);
    assert_eq!(e2el_results[0].pool_label.as_deref(), Some("low-latency"));

    let mut throughput_results = vec![low_latency.clone(), high_throughput.clone()];
    sort_serving_results(
        &mut throughput_results,
        ServingObjective::MaximizeThroughput,
    );
    assert_eq!(
        throughput_results[0].pool_label.as_deref(),
        Some("high-throughput")
    );

    let mut memory_results = vec![low_latency.clone(), high_throughput.clone()];
    sort_serving_results(
        &mut memory_results,
        ServingObjective::MinimizeMemoryPressure,
    );
    assert_eq!(
        memory_results[0].pool_label.as_deref(),
        Some("high-throughput")
    );

    let mut low_cost = low_latency;
    low_cost.cost_estimate.total_cost_usd = Some(1.0);
    low_cost.cost_estimate.energy_kwh = Some(2.0);
    low_cost.cost_estimate.average_power_watts = Some(400.0);
    low_cost.pool_label = Some("low-cost".to_string());

    let mut high_cost = high_throughput;
    high_cost.cost_estimate.total_cost_usd = Some(10.0);
    high_cost.cost_estimate.energy_kwh = Some(20.0);
    high_cost.cost_estimate.average_power_watts = Some(800.0);
    high_cost.pool_label = Some("high-cost".to_string());

    let mut cost_results = vec![high_cost.clone(), low_cost.clone()];
    sort_serving_results(&mut cost_results, ServingObjective::MinimizeCost);
    assert_eq!(cost_results[0].pool_label.as_deref(), Some("low-cost"));

    let mut energy_results = vec![high_cost.clone(), low_cost.clone()];
    sort_serving_results(&mut energy_results, ServingObjective::MinimizeEnergy);
    assert_eq!(energy_results[0].pool_label.as_deref(), Some("low-cost"));

    let mut power_results = vec![high_cost, low_cost];
    sort_serving_results(&mut power_results, ServingObjective::MinimizePower);
    assert_eq!(power_results[0].pool_label.as_deref(), Some("low-cost"));
}

#[test]
fn slo_miss_penalty_changes_latency_objective_ordering() {
    let pool = ResolvedServingPool {
        label: None,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
    };
    let mut fast_with_misses = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights {
                e2el: 2.0,
                ..ServingSloMissPenaltyWeights::default()
            },
            0.0,
        ),
        "test".to_string(),
    );
    fast_with_misses.feasible = true;
    fast_with_misses.metrics = finite_metrics();
    fast_with_misses.metrics.e2el_s = 1.0;
    fast_with_misses.metrics.e2el_slo_miss_rate = 1.0;
    fast_with_misses.slo_miss_penalty_components = slo_miss_penalty_components_from_metrics(
        &fast_with_misses.metrics,
        fast_with_misses.slo_miss_penalty_weights,
    );
    fast_with_misses.slo_miss_penalty_score = fast_with_misses.slo_miss_penalty_components.total;
    fast_with_misses.pool_label = Some("fast-with-misses".to_string());

    let mut slower_without_misses = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights {
                e2el: 2.0,
                ..ServingSloMissPenaltyWeights::default()
            },
            0.0,
        ),
        "test".to_string(),
    );
    slower_without_misses.feasible = true;
    slower_without_misses.metrics = finite_metrics();
    slower_without_misses.metrics.e2el_s = 2.0;
    slower_without_misses.slo_miss_penalty_components = slo_miss_penalty_components_from_metrics(
        &slower_without_misses.metrics,
        slower_without_misses.slo_miss_penalty_weights,
    );
    slower_without_misses.slo_miss_penalty_score =
        slower_without_misses.slo_miss_penalty_components.total;
    slower_without_misses.pool_label = Some("slower-without-misses".to_string());

    let mut results = vec![fast_with_misses, slower_without_misses];
    sort_serving_results(&mut results, ServingObjective::MinimizeE2el);

    assert_eq!(
        results[0].pool_label.as_deref(),
        Some("slower-without-misses")
    );
    assert_eq!(results[1].slo_miss_penalty_score, 2.0);
}

#[test]
fn topology_risk_penalty_changes_latency_objective_ordering() {
    let pool = ResolvedServingPool {
        label: None,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
    };
    let mut fast_partial_route = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights::default(),
            2.0,
        ),
        "test".to_string(),
    );
    fast_partial_route.feasible = true;
    fast_partial_route.metrics = finite_metrics();
    fast_partial_route.metrics.e2el_s = 1.0;
    fast_partial_route.route_coverage = ServingRouteCoverage {
        candidate_count: 2,
        routable_candidate_count: 1,
        unroutable_candidate_count: 1,
        fraction: 0.5,
    };
    fast_partial_route.topology_risk_penalty_score = topology_risk_penalty_score(
        fast_partial_route.route_coverage,
        &fast_partial_route.topology_bottlenecks,
        fast_partial_route.topology_risk_penalty_weight,
    );
    fast_partial_route.pool_label = Some("fast-partial-route".to_string());

    let mut slower_full_route = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights::default(),
            2.0,
        ),
        "test".to_string(),
    );
    slower_full_route.feasible = true;
    slower_full_route.metrics = finite_metrics();
    slower_full_route.metrics.e2el_s = 1.5;
    slower_full_route.route_coverage = ServingRouteCoverage {
        candidate_count: 2,
        routable_candidate_count: 2,
        unroutable_candidate_count: 0,
        fraction: 1.0,
    };
    slower_full_route.topology_risk_penalty_score = topology_risk_penalty_score(
        slower_full_route.route_coverage,
        &slower_full_route.topology_bottlenecks,
        slower_full_route.topology_risk_penalty_weight,
    );
    slower_full_route.pool_label = Some("slower-full-route".to_string());

    let mut results = vec![fast_partial_route, slower_full_route];
    sort_serving_results(&mut results, ServingObjective::MinimizeE2el);

    assert_eq!(results[0].pool_label.as_deref(), Some("slower-full-route"));
    assert_eq!(results[1].topology_risk_penalty_score, 1.0);
}

#[test]
fn topology_risk_penalty_scores_route_locality_and_contention_bottlenecks() {
    let full_coverage = ServingRouteCoverage {
        candidate_count: 2,
        routable_candidate_count: 2,
        unroutable_candidate_count: 0,
        fraction: 1.0,
    };

    let cross_socket_penalty = topology_risk_penalty_score(
        full_coverage,
        &[test_topology_bottleneck(
            "cross_socket_kv_path",
            "warning",
            None,
        )],
        2.0,
    );
    assert!((cross_socket_penalty - 0.6).abs() < 1e-9);

    let host_staged_penalty = topology_risk_penalty_score(
        full_coverage,
        &[test_topology_bottleneck(
            "host_staged_kv_path",
            "warning",
            None,
        )],
        2.0,
    );
    assert!((host_staged_penalty - 1.0).abs() < 1e-9);

    let hot_route_penalty = topology_risk_penalty_score(
        full_coverage,
        &[test_topology_bottleneck(
            "hot_kv_route_resource",
            "warning",
            Some(0.92),
        )],
        2.0,
    );
    assert!((hot_route_penalty - 1.84).abs() < 1e-9);
}

#[test]
fn topology_bottlenecks_report_domain_concentration_risk() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[nodes]]
        id = 0
        rack = "rack-a"
        island = "island-a"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 1
        rack = "rack-a"
        island = "island-a"
        failure_domain = "az-a"
        gpu = "h100_sxm"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }

        [[nodes]]
        id = 2
        rack = "rack-b"
        island = "island-b"
        failure_domain = "az-b"
        gpu = "a100_80gb"
        gpu_count = 1
        nics = { count = 1, affinity = "dedicated", rail_count = 1 }
        "#,
    )
    .unwrap();
    let prefill_placement = RankPlacement {
        rank_to_gpu: vec![
            GpuAddr {
                node_id: 0,
                local_gpu_id: 0,
            },
            GpuAddr {
                node_id: 1,
                local_gpu_id: 0,
            },
        ],
    };
    let decode_placement = RankPlacement {
        rank_to_gpu: vec![GpuAddr {
            node_id: 2,
            local_gpu_id: 0,
        }],
    };

    let bottlenecks =
        topology_domain_bottleneck_observations(&cluster, &prefill_placement, &decode_placement);

    assert!(bottlenecks.iter().any(|bottleneck| {
        bottleneck.phase == "prefill"
            && bottleneck.code == "single_failure_domain_placement"
            && bottleneck.resource == "failure_domain:az_a"
    }));
    assert!(bottlenecks.iter().any(|bottleneck| {
        bottleneck.phase == "prefill"
            && bottleneck.code == "single_rack_placement"
            && bottleneck.resource == "rack:rack_a"
    }));

    let score = topology_risk_penalty_score(
        ServingRouteCoverage {
            candidate_count: 1,
            routable_candidate_count: 1,
            unroutable_candidate_count: 0,
            fraction: 1.0,
        },
        &bottlenecks,
        2.0,
    );
    assert_eq!(score, 2.0);
}

#[test]
fn service_backpressure_penalty_changes_latency_objective_ordering() {
    let pool = ResolvedServingPool {
        label: None,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        prefill_gpu_labels: Vec::new(),
        decode_gpu_labels: Vec::new(),
    };
    let mut fast_rejecting = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights::default(),
            0.0,
        ),
        "test".to_string(),
    );
    fast_rejecting.feasible = true;
    fast_rejecting.metrics = finite_metrics();
    fast_rejecting.metrics.e2el_s = 1.0;
    fast_rejecting.service_backpressure_penalty_weight = 2.0;
    fast_rejecting.service_observations = vec![service_observation_for_penalty("decode", 2, 1)];
    fast_rejecting.service_backpressure_penalty_score =
        service_backpressure_penalty_score(&fast_rejecting.service_observations, 2.0);
    fast_rejecting.pool_label = Some("fast-rejecting".to_string());

    let mut slower_open = ServingSolver::rejected_pair_for_pool(
        &pool,
        rejected_pair_context(
            ServingObjective::MinimizeE2el,
            ServingSloMissPenaltyWeights::default(),
            0.0,
        ),
        "test".to_string(),
    );
    slower_open.feasible = true;
    slower_open.metrics = finite_metrics();
    slower_open.metrics.e2el_s = 1.4;
    slower_open.service_backpressure_penalty_weight = 2.0;
    slower_open.service_observations = vec![service_observation_for_penalty("decode", 2, 0)];
    slower_open.service_backpressure_penalty_score =
        service_backpressure_penalty_score(&slower_open.service_observations, 2.0);
    slower_open.pool_label = Some("slower-open".to_string());

    let mut results = vec![fast_rejecting, slower_open];
    sort_serving_results(&mut results, ServingObjective::MinimizeE2el);

    assert_eq!(results[0].pool_label.as_deref(), Some("slower-open"));
    assert_eq!(results[1].service_backpressure_penalty_score, 1.0);
}

#[test]
fn traffic_class_slo_miss_penalty_uses_scoped_breakdown() {
    let traffic = ServingTraffic {
        traffic_classes: vec![ServingTrafficClass {
            name: "gold".to_string(),
            group: "tenant".to_string(),
            key: "tenant-a".to_string(),
            slo_miss_penalty_weights: ServingSloMissPenaltyWeights {
                ttft: 2.0,
                e2el: 3.0,
                ..ServingSloMissPenaltyWeights::default()
            },
            ..ServingTrafficClass::default()
        }],
        ..ServingTraffic::default()
    };
    let breakdown = ServingMetricBreakdown {
        group: "tenant".to_string(),
        key: "tenant-a".to_string(),
        request_count: 2,
        completed_requests: 2,
        failed_requests: 0,
        rejected_requests: 0,
        timed_out_requests: 0,
        cancelled_requests: 0,
        output_tokens: 4,
        lifecycle_event_metric_request_count: 2,
        fallback_metric_request_count: 0,
        metric_source_counts: vec![ServingMeasurementMetricSourceCount {
            metric_source: "request_lifecycle_events".to_string(),
            request_count: 2,
        }],
        deadline_constrained_requests: 0,
        deadline_missed_requests: 0,
        deadline_miss_rate: 0.0,
        ttft_slo_constrained_requests: 2,
        ttft_slo_missed_requests: 1,
        ttft_slo_miss_rate: 0.5,
        tpot_slo_constrained_requests: 0,
        tpot_slo_missed_requests: 0,
        tpot_slo_miss_rate: 0.0,
        itl_slo_constrained_requests: 0,
        itl_slo_missed_requests: 0,
        itl_slo_miss_rate: 0.0,
        e2el_slo_constrained_requests: 2,
        e2el_slo_missed_requests: 2,
        e2el_slo_miss_rate: 1.0,
        ttft_s: 1.0,
        ttft_p90_s: 1.0,
        ttft_p95_s: 1.0,
        ttft_max_s: 1.0,
        tpot_s: 1.0,
        tpot_p90_s: 1.0,
        tpot_p95_s: 1.0,
        tpot_max_s: 1.0,
        itl_s: 1.0,
        itl_p90_s: 1.0,
        itl_p95_s: 1.0,
        itl_max_s: 1.0,
        throughput_tokens_per_s: 1.0,
        e2el_s: 1.0,
        e2el_p90_s: 1.0,
        e2el_p95_s: 1.0,
        e2el_max_s: 1.0,
    };

    let penalties = traffic_class_slo_miss_penalties(&traffic, &[breakdown]);

    assert_eq!(penalties.len(), 1);
    assert_eq!(penalties[0].name, "gold");
    assert_eq!(penalties[0].components.ttft, 1.0);
    assert_eq!(penalties[0].components.e2el, 3.0);
    assert_eq!(penalties[0].components.total, 4.0);
}

fn finite_metrics() -> ServingMetrics {
    let mut metrics = ServingMetrics::rejected();
    metrics.ttft_s = 1.0;
    metrics.tpot_s = 1.0;
    metrics.itl_s = 1.0;
    metrics.e2el_s = 1.0;
    metrics.throughput_tokens_per_s = 1.0;
    metrics.ttft_slo_miss_rate = 0.0;
    metrics.tpot_slo_miss_rate = 0.0;
    metrics.itl_slo_miss_rate = 0.0;
    metrics.e2el_slo_miss_rate = 0.0;
    metrics.deadline_miss_rate = 0.0;
    metrics
}

fn test_topology_bottleneck(
    code: &str,
    severity: &str,
    observed: Option<f64>,
) -> ServingTopologyBottleneckObservation {
    ServingTopologyBottleneckObservation {
        phase: "kv_transfer".to_string(),
        category: "topology".to_string(),
        resource: "test".to_string(),
        code: code.to_string(),
        severity: severity.to_string(),
        observed,
        limit: None,
        unit: None,
        message: "test topology bottleneck".to_string(),
        remediation: None,
    }
}

fn rejected_pair_context(
    objective: ServingObjective,
    slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
    topology_risk_penalty_weight: f64,
) -> RejectedPairContext {
    RejectedPairContext {
        objective,
        slo_miss_penalty_weight: slo_miss_penalty_weights.aggregate,
        slo_miss_penalty_weights,
        topology_risk_penalty_weight,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        metric_ceilings: ServingMetricCeilings::default(),
        kv_route_constraints: ServingKvRouteConstraints::default(),
        pool_search_summary: None,
    }
}

fn memory_pressure_observation(
    phase: &str,
    capacity_used_fraction: f64,
) -> ServingMemoryPressureObservation {
    ServingMemoryPressureObservation {
        phase: phase.to_string(),
        estimate_kind: "test".to_string(),
        start_s: 0.0,
        finish_s: 1.0,
        duration_s: 1.0,
        active_requests: 1,
        active_tokens: 1,
        kv_blocks: 1,
        estimated_per_gpu_gb: capacity_used_fraction,
        min_hbm_per_gpu_gb: 1.0,
        capacity_used_fraction,
        headroom_gb: 1.0 - capacity_used_fraction,
        limiting_gpu: Some(GpuAddr {
            node_id: 0,
            local_gpu_id: 0,
        }),
        dominant_component: None,
        components: ServingMemoryComponents {
            weights_gb: capacity_used_fraction,
            kv_cache_gb: 0.0,
            block_table_gb: 0.0,
            activations_gb: 0.0,
            temporary_gb: 0.0,
            communication_gb: 0.0,
            runtime_reserve_gb: 0.0,
            fragmentation_gb: 0.0,
            total_gb: capacity_used_fraction,
        },
    }
}

fn service_observation_for_penalty(
    phase: &str,
    request_count: u32,
    backpressure_rejections: u32,
) -> ServingServiceObservation {
    ServingServiceObservation {
        phase: phase.to_string(),
        health: ServingServiceHealth::Healthy,
        accepts_requests: true,
        worker_scale: 1.0,
        configured_worker_slots_per_gpu: 1,
        effective_worker_slots_per_gpu: 1,
        node_count: 1,
        gpu_count: 1,
        request_count,
        admitted_requests: request_count.saturating_sub(backpressure_rejections),
        completed_requests: request_count.saturating_sub(backpressure_rejections),
        failed_requests: backpressure_rejections,
        rejected_requests: backpressure_rejections,
        timed_out_requests: 0,
        cancelled_requests: 0,
        queue_cap_s: Some(0.0),
        queue_cap_request_count: request_count,
        queue_cap_hit_count: backpressure_rejections,
        decode_iteration_queue_cap_s: None,
        decode_iteration_queue_cap_request_count: 0,
        decode_iteration_queue_cap_hit_count: 0,
        backpressure_rejections,
        timeout_rejections: 0,
        backpressure_state: if backpressure_rejections > 0 {
            "rejecting".to_string()
        } else {
            "open".to_string()
        },
        worker_slot_utilization: 0.0,
        queue_s: 0.0,
        queue_p95_s: 0.0,
        queue_max_s: 0.0,
        worker_queue_s: 0.0,
        resource_queue_s: 0.0,
        service_s: 0.0,
    }
}

#[test]
fn measurement_window_filters_reported_serving_metrics() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let mut serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(3),
            arrival_gap_s: Some(1.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: Some(10.0),
            tpot_slo_s: Some(10.0),
            itl_slo_s: Some(10.0),
            e2el_slo_s: Some(10.0),
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: None,
            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: Some(0.5),
            measurement_end_s: Some(1.5),
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![64],
            decode_tokens: vec![2],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.request_observations.len(), 3);
    assert_eq!(score.metrics.scheduled_requests, 3);
    assert_eq!(score.metrics.measured_requests, 1);
    assert_eq!(score.metrics.measurement_start_s, 0.5);
    assert_eq!(score.metrics.measurement_end_s, 1.5);
    assert_eq!(score.metrics.ttft_s, score.request_observations[1].ttft_s);
    assert_eq!(score.metrics.e2el_s, score.request_observations[1].e2el_s);
    assert_eq!(score.measurement_window.measured_requests, 1);
    assert_eq!(
        score
            .measurement_window
            .lifecycle_event_metric_request_count,
        1
    );
    assert_eq!(score.measurement_window.fallback_metric_request_count, 0);
    assert_eq!(score.measurement_window.metric_source_counts.len(), 1);
    assert_eq!(
        score.measurement_window.metric_source_counts[0].metric_source,
        "request_lifecycle_events"
    );
    assert_eq!(
        score.measurement_window.metric_source_counts[0].request_count,
        1
    );
    let measured_observation = &score.request_observations[1];
    assert_eq!(
        measured_observation.metric_source,
        "request_lifecycle_events"
    );
    assert_eq!(measured_observation.decode_iterations, 2);
    assert_eq!(observation_output_tokens(measured_observation), 2);
    assert!(
        (score.metrics.throughput_tokens_per_s
            - observation_output_tokens(measured_observation) as f64)
            .abs()
            < 1e-12
    );
    let priority_breakdown = score
        .metric_breakdowns
        .iter()
        .find(|breakdown| breakdown.group == "priority" && breakdown.key == "priority-0")
        .expect("priority metric breakdown");
    assert_eq!(
        priority_breakdown.output_tokens,
        observation_output_tokens(measured_observation)
    );
    assert_eq!(
        priority_breakdown.throughput_tokens_per_s,
        score.metrics.throughput_tokens_per_s
    );

    serving.traffic.measurement_start_s = None;
    serving.traffic.measurement_end_s = None;
    serving.traffic.measurement_warmup_s = Some(0.5);
    serving.traffic.measurement_cooldown_s = Some(1.0);

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.measured_requests, 1);
    assert_eq!(score.metrics.measurement_start_s, 0.5);
    assert!(score.metrics.measurement_end_s > 1.0);
    assert!(score.metrics.measurement_end_s < 2.0);
    assert_eq!(score.metrics.ttft_s, score.request_observations[1].ttft_s);
    assert_eq!(score.metrics.e2el_s, score.request_observations[1].e2el_s);

    serving.traffic.measurement_warmup_s = None;
    serving.traffic.measurement_cooldown_s = None;
    serving.traffic.measurement_steady_state = true;
    serving.traffic.measurement_steady_state_min_requests = Some(1);
    serving.traffic.measurement_steady_state_max_cv = Some(0.0);

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert!(
        score
            .approximations
            .iter()
            .any(|approximation| { approximation.code == "steady_state_measurement_window" })
    );
    assert_eq!(score.measurement_window.source, "steady_state");
    assert!(score.measurement_window.steady_state_requested);
    assert!(score.measurement_window.steady_state_applied);
    assert_eq!(
        score.measurement_window.measured_requests,
        score.metrics.measured_requests
    );
    assert!(score.measurement_window.duration_s >= 0.0);
    assert_eq!(score.measurement_window.steady_state_sample_count, 3);
    assert_eq!(
        score
            .measurement_window
            .steady_state_candidate_request_count,
        Some(score.metrics.measured_requests)
    );
    assert_eq!(
        score.measurement_window.steady_state_candidate_e2el_cv,
        Some(0.0)
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_e2el_std_error_s
            .is_some()
    );
    assert!(score.measurement_window.steady_state_candidate_metric_count >= 8);
    assert!(
        score
            .measurement_window
            .steady_state_candidate_worst_cv
            .is_some_and(|cv| cv >= 0.0)
    );
    assert_eq!(
        score
            .measurement_window
            .steady_state_candidate_output_tokens,
        Some(u64::from(score.metrics.measured_requests) * 2)
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_throughput_tokens_per_s
            .is_some_and(|throughput| throughput > 0.0)
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_metrics
            .iter()
            .any(|metric| metric.metric == "ttft")
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_metrics
            .iter()
            .any(|metric| metric.metric == "tpot")
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_metrics
            .iter()
            .any(|metric| metric.metric == "request_throughput")
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_utilization_count
            > 0
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_worst_utilization_cv
            .is_some()
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_utilization
            .iter()
            .any(|utilization| utilization.source == "scheduled_resource")
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_utilization
            .iter()
            .any(|utilization| utilization.source == "worker_slot")
    );
    assert!(
        score
            .measurement_window
            .steady_state_candidate_utilization
            .iter()
            .any(|utilization| utilization.source == "kv_residency")
    );
}

#[test]
fn steady_state_detection_selects_stable_latency_window() {
    let samples = vec![
        (0.0, 2.0),
        (1.0, 1.01),
        (2.0, 1.00),
        (3.0, 0.99),
        (4.0, 1.02),
        (5.0, 0.4),
    ];

    let search = steady_state_measurement_window_from_samples(samples, Some(3), Some(0.05));
    let window = search.selected.unwrap();

    assert_eq!((window.start_s, window.end_s), (1.0, 4.0));
    assert_eq!(window.request_count, 4);
    assert_eq!(search.sample_count, 6);
    assert!(search.matching_window_count > 0);
    assert!(window.cv <= 0.05);
    assert!(window.std_error_s > 0.0);
    assert_eq!(window.output_tokens, 4);
    assert!(window.throughput_tokens_per_s > 0.0);
    assert_eq!(window.metric_count, window.metrics.len() as u32);
    assert!(window.metrics.iter().any(|metric| metric.metric == "ttft"));
    assert!(
        window
            .metrics
            .iter()
            .any(|metric| metric.metric == "request_throughput")
    );
    assert!(window.worst_cv.is_some());
}

#[test]
fn service_health_rejects_and_worker_scale_is_reported() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let mut serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            max_decode_worker_slots_per_gpu: Some(1),
            services: ServingServicesConfig {
                decode: ServingServicePhaseConfig {
                    health: ServingServiceHealth::Unavailable,
                    worker_scale: 1.0,
                },
                ..ServingServicesConfig::default()
            },
            service_backpressure_penalty_weight: 0.0,
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let rejected = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );
    assert!(!rejected[0].feasible);
    assert!(rejected[0].rejections.iter().any(|rejection| {
        rejection.phase == "decode" && rejection.code == "service_unavailable"
    }));

    serving.traffic.services.decode = ServingServicePhaseConfig {
        health: ServingServiceHealth::Healthy,
        worker_scale: 2.0,
    };
    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );
    assert!(results[0].feasible);
    let decode_service = results[0]
        .service_observations
        .iter()
        .find(|service| service.phase == "decode")
        .expect("decode service observation");
    assert_eq!(decode_service.health, ServingServiceHealth::Healthy);
    assert!(decode_service.accepts_requests);
    assert_eq!(decode_service.configured_worker_slots_per_gpu, 1);
    assert_eq!(decode_service.effective_worker_slots_per_gpu, 2);
    assert_eq!(decode_service.worker_scale, 2.0);
    assert!(decode_service.request_count > 0);
}

#[test]
fn admission_control_rejects_requests_that_wait_too_long() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: Some(10.0),
            tpot_slo_s: Some(10.0),
            itl_slo_s: Some(10.0),
            e2el_slo_s: Some(10.0),
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: None,
            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: Some(0.0),
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![2],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.scheduled_requests, 2);
    assert_eq!(score.metrics.admitted_requests, 1);
    assert_eq!(score.metrics.completed_requests, 1);
    assert_eq!(score.metrics.rejected_requests, 1);
    assert_eq!(score.metrics.timed_out_requests, 0);
    assert_eq!(score.metrics.measured_requests, 1);
    assert_eq!(score.metrics.ttft_slo_miss_rate, 0.5);
    assert_eq!(score.metrics.e2el_slo_miss_rate, 0.5);
    assert_eq!(score.request_observations.len(), 2);
    assert_eq!(
        score.request_observations[0].status,
        ServingRequestStatus::Completed
    );
    assert_eq!(
        score.request_observations[1].status,
        ServingRequestStatus::RejectedAdmission
    );
    assert!(
        score.request_observations[1]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("admission rejected"))
    );
    let rejection = score.request_observations[1]
        .rejection
        .as_ref()
        .expect("structured prefill queue rejection");
    assert_eq!(rejection.phase, "prefill");
    assert_eq!(rejection.category, "queueing");
    assert_eq!(rejection.code, "prefill_queue_delay_exceeded");
    assert_eq!(rejection.unit.as_deref(), Some("s"));
    assert!(
        score.request_observations[1]
            .status_time_s
            .is_some_and(f64::is_finite)
    );
    assert!(
        score.request_observations[1]
            .lifecycle_events
            .iter()
            .any(|event| event.kind == ServingRequestEventKind::RejectedAdmission)
    );
    assert!(
        !score.request_observations[1]
            .lifecycle_events
            .iter()
            .any(|event| event.kind == ServingRequestEventKind::PrefillStarted)
    );
    assert!(
        score.request_observations[1]
            .phase_breakdown
            .iter()
            .any(|phase| {
                phase.phase == "queued_for_prefill"
                    && phase.category == ServingRequestPhaseCategory::Queue
            })
    );
    assert!(score.request_observations[1].prefill_start_s.is_infinite());
}

#[test]
fn request_timeout_marks_requests_without_counting_success_latency() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: Some(10.0),
            tpot_slo_s: Some(10.0),
            itl_slo_s: Some(10.0),
            e2el_slo_s: Some(10.0),
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: None,
            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: Some(1e-9),
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![2],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.scheduled_requests, 1);
    assert_eq!(score.metrics.admitted_requests, 1);
    assert_eq!(score.metrics.completed_requests, 0);
    assert_eq!(score.metrics.rejected_requests, 0);
    assert_eq!(score.metrics.timed_out_requests, 1);
    assert_eq!(score.metrics.measured_requests, 0);
    assert_eq!(score.metrics.throughput_tokens_per_s, 0.0);
    assert!(score.metrics.e2el_s.is_infinite());
    assert_eq!(score.metrics.ttft_slo_miss_rate, 1.0);
    assert_eq!(score.metrics.tpot_slo_miss_rate, 1.0);
    assert_eq!(score.metrics.itl_slo_miss_rate, 1.0);
    assert_eq!(score.metrics.e2el_slo_miss_rate, 1.0);
    assert_eq!(
        score.request_observations[0].status,
        ServingRequestStatus::TimedOut
    );
    assert!(
        score.request_observations[0]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("request timed out"))
    );
    let rejection = score.request_observations[0]
        .rejection
        .as_ref()
        .expect("structured request timeout rejection");
    assert_eq!(rejection.phase, "end_to_end");
    assert_eq!(rejection.category, "timeout");
    assert_eq!(rejection.resource, "request_timeout");
    assert_eq!(rejection.code, "request_timeout_exceeded");
    assert_eq!(rejection.limit, Some(1e-9));
    assert_eq!(rejection.unit.as_deref(), Some("s"));
    assert!(
        rejection
            .observed
            .is_some_and(|observed| observed > rejection.limit.unwrap())
    );
    assert!(
        score.request_observations[0]
            .last_decode_finish_s
            .is_finite()
    );
}

#[test]
fn traffic_class_queue_delay_rejects_matching_requests() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            trace_requests: vec![
                ServingTraceRequest {
                    request_id: Some("gold-0".to_string()),
                    tenant: Some("gold".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 2,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
                ServingTraceRequest {
                    request_id: Some("gold-1".to_string()),
                    tenant: Some("gold".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 2,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
            ],
            traffic_classes: vec![ServingTrafficClass {
                name: "gold".to_string(),
                group: "tenant".to_string(),
                key: "gold".to_string(),
                max_queue_delay_s: Some(0.0),
                ..ServingTrafficClass::default()
            }],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert_eq!(score.metrics.admitted_requests, 1);
    assert_eq!(score.metrics.rejected_requests, 1);
    assert_eq!(
        score.request_observations[1].status,
        ServingRequestStatus::RejectedAdmission
    );
    assert_eq!(
        score.request_observations[1].traffic_class.as_deref(),
        Some("gold")
    );
    assert!(
        score.request_observations[1]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("admission rejected"))
    );
}

#[test]
fn traffic_class_timeout_marks_matching_requests() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            trace_requests: vec![ServingTraceRequest {
                request_id: Some("gold-timeout".to_string()),
                tenant: Some("gold".to_string()),
                model_id: Some("model-a".to_string()),
                cache_key: None,
                arrival_s: 0.0,
                priority: 0,
                batch_size: 1,
                prompt_tokens: 128,
                decode_tokens: 2,
                max_sequence_tokens: None,
                prefix_cache_hit_tokens: None,
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo::default(),
                deadline_s: None,
                cancellation_s: None,
            }],
            traffic_classes: vec![ServingTrafficClass {
                name: "gold".to_string(),
                group: "tenant".to_string(),
                key: "gold".to_string(),
                request_timeout_s: Some(1e-9),
                ..ServingTrafficClass::default()
            }],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let observation = &results[0].request_observations[0];
    assert_eq!(observation.status, ServingRequestStatus::TimedOut);
    assert_eq!(observation.traffic_class.as_deref(), Some("gold"));
    assert!(
        observation
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("request timed out"))
    );
}

#[test]
fn slo_miss_rate_limits_reject_serving_candidates() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: None,
            tpot_slo_s: None,
            itl_slo_s: None,
            e2el_slo_s: Some(1e-9),
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: Some(0.0),
            max_deadline_miss_rate: Some(0.0),
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: Vec::new(),
            prompt_tokens: Vec::new(),
            decode_tokens: Vec::new(),
            trace_requests: vec![ServingTraceRequest {
                request_id: Some("deadline-miss".to_string()),
                tenant: None,
                model_id: None,
                cache_key: None,
                arrival_s: 0.0,
                priority: 0,
                batch_size: 1,
                prompt_tokens: 128,
                decode_tokens: 2,
                max_sequence_tokens: None,
                prefix_cache_hit_tokens: None,
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo::default(),
                deadline_s: Some(1e-9),
                cancellation_s: None,
            }],
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(!score.feasible);
    assert_eq!(score.metrics.e2el_slo_miss_rate, 1.0);
    assert_eq!(score.metrics.deadline_miss_rate, 1.0);
    assert!(score.bottlenecks.contains(&"e2el_miss_rate".to_string()));
    assert!(
        score
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("E2EL miss rate exceeded"))
    );

    let e2el_rejection = score
        .rejections
        .iter()
        .find(|rejection| rejection.code == "e2el_miss_rate_exceeded")
        .expect("missing E2EL SLO rejection");
    assert_eq!(e2el_rejection.phase, "serving");
    assert_eq!(e2el_rejection.category, "slo");
    assert_eq!(e2el_rejection.resource, "e2el_miss_rate");
    assert_eq!(e2el_rejection.observed, Some(1.0));
    assert_eq!(e2el_rejection.limit, Some(0.0));
    assert_eq!(e2el_rejection.unit.as_deref(), Some("ratio"));

    let deadline_rejection = score
        .rejections
        .iter()
        .find(|rejection| rejection.code == "deadline_miss_rate_exceeded")
        .expect("missing deadline SLO rejection");
    assert_eq!(deadline_rejection.resource, "deadline_miss_rate");
    assert_eq!(deadline_rejection.observed, Some(1.0));
    assert_eq!(deadline_rejection.limit, Some(0.0));
}

#[test]
fn scoped_slo_policies_reject_priority_miss_rates() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            trace_requests: vec![
                ServingTraceRequest {
                    request_id: Some("priority-miss".to_string()),
                    tenant: Some("gold".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 10,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 2,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo {
                        ttft_s: None,
                        tpot_s: None,
                        itl_s: None,
                        e2el_s: Some(1e-9),
                    },
                    deadline_s: None,
                    cancellation_s: None,
                },
                ServingTraceRequest {
                    request_id: Some("bronze-unconstrained".to_string()),
                    tenant: Some("bronze".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.1,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 2,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
            ],
            ..ServingTraffic::default()
        },
        slo_policies: vec![ServingSloPolicy {
            group: "priority".to_string(),
            key: "priority-10".to_string(),
            max_e2el_slo_miss_rate: Some(0.0),
            ..ServingSloPolicy::default()
        }],
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(!score.feasible);
    let priority_breakdown = score
        .metric_breakdowns
        .iter()
        .find(|breakdown| breakdown.group == "priority" && breakdown.key == "priority-10")
        .expect("priority breakdown");
    assert_eq!(priority_breakdown.e2el_slo_miss_rate, 1.0);
    let rejection = score
        .rejections
        .iter()
        .find(|rejection| rejection.code == "scoped_e2el_miss_rate_exceeded")
        .expect("missing scoped SLO rejection");
    assert_eq!(rejection.resource, "priority:priority-10:e2el_miss_rate");
    assert_eq!(rejection.observed, Some(1.0));
    assert_eq!(rejection.limit, Some(0.0));
    assert!(
        rejection
            .message
            .contains("E2EL miss rate exceeded for priority=priority-10")
    );
}

#[test]
fn traffic_classes_apply_default_slos_to_matching_requests() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            trace_requests: vec![ServingTraceRequest {
                request_id: Some("gold-request".to_string()),
                tenant: Some("gold".to_string()),
                model_id: Some("model-a".to_string()),
                cache_key: None,
                arrival_s: 0.0,
                priority: 0,
                batch_size: 1,
                prompt_tokens: 128,
                decode_tokens: 2,
                max_sequence_tokens: None,
                prefix_cache_hit_tokens: None,
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo::default(),
                deadline_s: None,
                cancellation_s: None,
            }],
            traffic_classes: vec![ServingTrafficClass {
                name: "gold-tenant".to_string(),
                group: "tenant".to_string(),
                key: "gold".to_string(),
                slo: ServingRequestSlo {
                    ttft_s: None,
                    tpot_s: None,
                    itl_s: None,
                    e2el_s: Some(1e-9),
                },
                ..ServingTrafficClass::default()
            }],
            ..ServingTraffic::default()
        },
        slo_policies: vec![ServingSloPolicy {
            group: "tenant".to_string(),
            key: "gold".to_string(),
            max_e2el_slo_miss_rate: Some(0.0),
            ..ServingSloPolicy::default()
        }],
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(!score.feasible);
    assert_eq!(
        score.request_observations[0].traffic_class.as_deref(),
        Some("gold-tenant")
    );
    assert_eq!(score.request_observations[0].slo.e2el_s, Some(1e-9));
    assert!(score.request_observations[0].e2el_slo_missed);
    assert!(
        score
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("tenant=gold"))
    );
}

#[test]
fn cancellation_marks_requests_as_failed_without_decode_residency() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: Some(10.0),
            tpot_slo_s: Some(10.0),
            itl_slo_s: Some(10.0),
            e2el_slo_s: Some(10.0),
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: None,
            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: Vec::new(),
            prompt_tokens: Vec::new(),
            decode_tokens: Vec::new(),
            trace_requests: vec![ServingTraceRequest {
                request_id: Some("cancelled-request".to_string()),
                tenant: Some("tenant-a".to_string()),
                model_id: Some("model-a".to_string()),
                cache_key: None,
                arrival_s: 0.0,
                priority: 0,
                batch_size: 1,
                prompt_tokens: 128,
                decode_tokens: 2,
                max_sequence_tokens: None,
                prefix_cache_hit_tokens: None,
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo {
                    ttft_s: Some(1e-9),
                    tpot_s: Some(1e-9),
                    itl_s: Some(1e-9),
                    e2el_s: Some(1e-9),
                },
                deadline_s: Some(10.0),
                cancellation_s: Some(1e-9),
            }],
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.scheduled_requests, 1);
    assert_eq!(score.metrics.admitted_requests, 1);
    assert_eq!(score.metrics.completed_requests, 0);
    assert_eq!(score.metrics.cancelled_requests, 1);
    assert_eq!(score.metrics.timed_out_requests, 0);
    assert_eq!(score.metrics.deadline_constrained_requests, 1);
    assert_eq!(score.metrics.deadline_missed_requests, 1);
    assert_eq!(score.metrics.deadline_miss_rate, 1.0);
    assert_eq!(score.metrics.measured_requests, 0);
    assert_eq!(score.metrics.peak_decode_sequences, 0);
    assert_eq!(score.metrics.ttft_slo_miss_rate, 1.0);
    assert_eq!(
        score.request_observations[0].status,
        ServingRequestStatus::Cancelled
    );
    assert_eq!(score.request_observations[0].cancellation_s, Some(1e-9));
    assert_eq!(
        score.request_observations[0].request_id.as_deref(),
        Some("cancelled-request")
    );
    assert!(score.request_observations[0].kv_block_ownership.is_empty());
    assert!(
        !score.request_observations[0]
            .lifecycle_events
            .iter()
            .any(|event| event.kind == ServingRequestEventKind::KvBlocksAllocated)
    );
    assert!(score.request_observations[0].deadline_missed);
    assert!(
        score.request_observations[0]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("cancelled"))
    );
}

#[test]
fn priority_orders_ready_prefill_batches() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(128),
                chunk_tokens: None,
            },
            trace_requests: vec![
                ServingTraceRequest {
                    request_id: Some("low-priority".to_string()),
                    tenant: Some("tenant-low".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 1,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
                ServingTraceRequest {
                    request_id: Some("high-priority".to_string()),
                    tenant: Some("tenant-high".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 10,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 1,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
            ],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let observations = &results[0].request_observations;
    assert_eq!(observations[1].request_id.as_deref(), Some("high-priority"));
    assert!(observations[1].prefill_start_s <= observations[0].prefill_start_s + 1e-12);
    assert!(observations[1].prefill_finish_s <= observations[0].prefill_start_s + 1e-12);

    let model_breakdown = results[0]
        .metric_breakdowns
        .iter()
        .find(|breakdown| breakdown.group == "model_id" && breakdown.key == "model-a")
        .expect("model metric breakdown");
    assert_eq!(model_breakdown.request_count, 2);
    assert_eq!(model_breakdown.completed_requests, 2);
    assert_eq!(model_breakdown.output_tokens, 2);
    assert!(model_breakdown.ttft_s.is_finite());
    assert!(model_breakdown.ttft_p90_s >= model_breakdown.ttft_s);
    assert!(model_breakdown.ttft_max_s >= model_breakdown.ttft_p95_s);
    assert!(model_breakdown.e2el_p90_s >= model_breakdown.e2el_s);
    assert!(model_breakdown.e2el_max_s >= model_breakdown.e2el_p95_s);
    assert!(model_breakdown.throughput_tokens_per_s > 0.0);
    assert!(
        results[0]
            .metric_breakdowns
            .iter()
            .any(|breakdown| { breakdown.group == "tenant" && breakdown.key == "tenant-high" })
    );
    let priority_breakdown = results[0]
        .metric_breakdowns
        .iter()
        .find(|breakdown| breakdown.group == "priority" && breakdown.key == "priority-10")
        .expect("priority metric breakdown");
    assert_eq!(priority_breakdown.request_count, 1);
    assert_eq!(priority_breakdown.completed_requests, 1);
}

#[test]
fn traffic_class_admission_priority_orders_ready_prefill_batches() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(128),
                chunk_tokens: None,
            },
            trace_requests: vec![
                ServingTraceRequest {
                    request_id: Some("default-priority".to_string()),
                    tenant: Some("tenant-default".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 1,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
                ServingTraceRequest {
                    request_id: Some("class-priority".to_string()),
                    tenant: Some("tenant-fast".to_string()),
                    model_id: Some("model-a".to_string()),
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 1,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
            ],
            traffic_classes: vec![ServingTrafficClass {
                name: "fast-lane".to_string(),
                group: "tenant".to_string(),
                key: "tenant-fast".to_string(),
                admission_priority: Some(20),
                ..ServingTrafficClass::default()
            }],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let observations = &results[0].request_observations;
    assert_eq!(
        observations[1].request_id.as_deref(),
        Some("class-priority")
    );
    assert_eq!(observations[1].traffic_class.as_deref(), Some("fast-lane"));
    assert_eq!(observations[1].priority, 20);
    assert_eq!(observations[0].priority, 0);
    assert!(observations[1].prefill_start_s <= observations[0].prefill_start_s + 1e-12);
    assert!(observations[1].prefill_finish_s <= observations[0].prefill_start_s + 1e-12);

    let priority_breakdown = results[0]
        .metric_breakdowns
        .iter()
        .find(|breakdown| breakdown.group == "priority" && breakdown.key == "priority-20")
        .expect("traffic-class admission priority metric breakdown");
    assert_eq!(priority_breakdown.request_count, 1);
    assert_eq!(priority_breakdown.completed_requests, 1);
}

#[test]
fn rejects_serving_configs_when_component_memory_exceeds_hbm() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let mut large_model = model();
    large_model.parameters = Bytes::from_gigabytes(76.0);
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &large_model,
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    assert!(
        results[0].prefill_memory.components.total_gb
            > results[0].prefill_memory.min_hbm_per_gpu_gb
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("serving memory estimate"))
    );
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.category == "memory"
            && rejection.resource == "gpu_hbm"
            && rejection.code == "serving_memory_headroom_exceeded"
            && rejection.observed == Some(results[0].prefill_memory.estimated_per_gpu_gb)
            && rejection.limit == Some(results[0].prefill_memory.min_hbm_per_gpu_gb)
            && rejection.unit.as_deref() == Some("GB")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("sharding"))
            && rejection.message.contains("dominant_component=weights")
            && rejection.message.contains("limiting_gpu=node:0 gpu:0")
    }));
}

#[test]
fn serving_memory_uses_configured_kv_block_tokens_for_block_table_overhead() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let request = InferenceRequest {
        batch_size: 3,
        prompt_tokens: 129,
        decode_tokens: 17,
        max_sequence_tokens: 257,
        phase: InferencePhase::EndToEnd,
    };
    let mut serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            kv_block_tokens: Some(8),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let small_blocks = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request,
        &serving,
        SimulationCalibration::default(),
    );
    serving.traffic.kv_block_tokens = Some(64);
    let large_blocks = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request,
        &serving,
        SimulationCalibration::default(),
    );

    let expected_small_blocks =
        u64::from(request.batch_size) * u64::from(request.max_sequence_tokens).div_ceil(8);
    let expected_large_blocks =
        u64::from(request.batch_size) * u64::from(request.max_sequence_tokens).div_ceil(64);
    let expected_small_gb =
        expected_small_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES) as f64 / 1e9;
    let expected_large_gb =
        expected_large_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES) as f64 / 1e9;
    assert!(
        small_blocks[0].decode_memory.components.block_table_gb
            > large_blocks[0].decode_memory.components.block_table_gb
    );
    assert!(
        (small_blocks[0].decode_memory.components.block_table_gb - expected_small_gb).abs() < 1e-15
    );
    assert!(
        (large_blocks[0].decode_memory.components.block_table_gb - expected_large_gb).abs() < 1e-15
    );
}

#[test]
fn serving_memory_overhead_factors_are_calibratable() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let baseline = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );
    let stripped_overheads = SimulationCalibration {
        serving_memory_temporary_fraction: 0.0,
        serving_memory_activation_communication_fraction: 0.0,
        serving_memory_weight_communication_fraction: 0.0,
        serving_memory_runtime_reserve_fraction: 0.0,
        serving_memory_fragmentation_fraction: 0.0,
        ..SimulationCalibration::default()
    };
    let stripped = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        stripped_overheads,
    );

    assert!(baseline[0].prefill_memory.components.temporary_gb > 0.0);
    assert!(baseline[0].prefill_memory.components.communication_gb > 0.0);
    assert!(baseline[0].prefill_memory.components.runtime_reserve_gb > 0.0);
    assert!(baseline[0].prefill_memory.components.fragmentation_gb > 0.0);
    assert_eq!(stripped[0].prefill_memory.components.temporary_gb, 0.0);
    assert_eq!(stripped[0].prefill_memory.components.communication_gb, 0.0);
    assert_eq!(
        stripped[0].prefill_memory.components.runtime_reserve_gb,
        0.0
    );
    assert_eq!(stripped[0].prefill_memory.components.fragmentation_gb, 0.0);
    assert!(
        baseline[0].prefill_memory.components.total_gb
            > stripped[0].prefill_memory.components.total_gb
    );
}

#[test]
fn rejects_serving_configs_when_memory_pressure_limit_is_exceeded() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: Some(0.01),
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].max_memory_pressure_fraction, Some(0.01));
    assert!(!results[0].feasible);
    assert!(results[0].memory_pressure.iter().any(|pressure| {
        pressure.capacity_used_fraction > serving.max_memory_pressure_fraction.unwrap()
    }));
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.category == "memory"
            && rejection.resource == "memory_pressure"
            && rejection.code == "memory_pressure_fraction_exceeded"
            && rejection.limit == Some(0.01)
            && rejection.unit.as_deref() == Some("fraction")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("max_memory_pressure_fraction"))
    }));
}

#[test]
fn rejects_serving_configs_when_unique_gpu_footprint_exceeds_limit() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![2],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: Some(1),
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].max_unique_gpus, Some(1));
    assert!(!results[0].feasible);
    assert!(results[0].hardware_footprint.unique_gpu_count > 1);
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.category == "capacity"
            && rejection.resource == "unique_gpus"
            && rejection.code == "unique_gpu_footprint_exceeded"
            && rejection.limit == Some(1.0)
            && rejection.unit.as_deref() == Some("gpus")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("max_unique_gpus"))
    }));
}

#[test]
fn rejects_serving_configs_when_throughput_is_below_minimum() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: Some(1_000_000.0),
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].min_throughput_tokens_per_s, Some(1_000_000.0));
    assert!(!results[0].feasible);
    assert!(results[0].metrics.throughput_tokens_per_s < 1_000_000.0);
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.category == "throughput"
            && rejection.resource == "throughput_tokens_per_s"
            && rejection.code == "throughput_below_min"
            && rejection.limit == Some(1_000_000.0)
            && rejection.unit.as_deref() == Some("tokens/s")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("min_throughput_tokens_per_s"))
    }));
}

#[test]
fn rejects_serving_configs_when_latency_ceiling_is_exceeded() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            metric_ceilings: ServingMetricCeilings {
                max_e2el_s: Some(0.000_001),
                ..ServingMetricCeilings::default()
            },
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].metric_ceilings.max_e2el_s, Some(0.000_001));
    assert!(!results[0].feasible);
    assert!(results[0].metrics.e2el_s > 0.000_001);
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.category == "latency"
            && rejection.phase == "serving"
            && rejection.resource == "e2el_s"
            && rejection.code == "e2el_above_max"
            && rejection.limit == Some(0.000_001)
            && rejection.unit.as_deref() == Some("s")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("max_e2el_s"))
    }));
}

#[test]
fn reports_serving_cost_and_power_estimates_when_configured() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel {
            default_gpu_hour_usd: Some(4.0),
            node_hour_usd: Some(1.0),
            kwh_usd: Some(0.10),
            default_gpu_watts: Some(700.0),
            node_watts: Some(1000.0),
            gpu_rates: Vec::new(),
        },
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    let estimate = &results[0].cost_estimate;
    assert_eq!(
        estimate.modeled_gpu_count,
        results[0].hardware_footprint.unique_gpu_count
    );
    assert_eq!(
        estimate.modeled_node_count,
        results[0].hardware_footprint.unique_node_count
    );
    assert!(estimate.modeled_duration_s.is_some_and(|value| value > 0.0));
    assert!(estimate.gpu_hours.is_some_and(|value| value > 0.0));
    assert!(estimate.node_hours.is_some_and(|value| value > 0.0));
    assert!(estimate.gpu_hour_cost_usd.is_some_and(|value| value > 0.0));
    assert!(estimate.node_hour_cost_usd.is_some_and(|value| value > 0.0));
    assert!(
        estimate
            .average_power_watts
            .is_some_and(|value| value > 0.0)
    );
    assert!(estimate.energy_kwh.is_some_and(|value| value > 0.0));
    assert!(estimate.energy_cost_usd.is_some_and(|value| value > 0.0));
    assert!(estimate.total_cost_usd.is_some_and(|value| value > 0.0));
    assert!(
        estimate
            .cost_per_1k_output_tokens_usd
            .is_some_and(|value| value > 0.0)
    );
}

#[test]
fn rejects_serving_configs_that_exceed_prefill_capacity() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            max_prefill_tokens: Some(255),
            max_prefill_tokens_per_node: Some(255),
            max_prefill_tokens_per_gpu: Some(255),
            max_prefill_worker_slots_per_gpu: None,
            batch_sizes: vec![2],
            prompt_tokens: vec![128],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    assert_eq!(results[0].metrics.peak_prefill_tokens, 256);
    assert_eq!(results[0].metrics.peak_prefill_tokens_per_node, 256);
    assert_eq!(results[0].metrics.peak_prefill_tokens_per_gpu, 256);
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("prefill capacity exceeded"))
    );
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.phase == "prefill"
            && rejection.category == "capacity"
            && rejection.resource == "prefill_tokens"
            && rejection.code == "prefill_capacity_exceeded"
            && rejection.observed == Some(256.0)
            && rejection.limit == Some(255.0)
            && rejection.unit.as_deref() == Some("tokens")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("prefill capacity"))
    }));
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.phase == "prefill"
            && rejection.resource == "prefill_tokens_per_node"
            && rejection.code == "prefill_capacity_per_node_exceeded"
    }));
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.phase == "prefill"
            && rejection.resource == "prefill_tokens_per_gpu"
            && rejection.code == "prefill_capacity_per_gpu_exceeded"
    }));
}

#[test]
fn parallelism_rejections_preserve_structured_placement_evidence() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let prefill_search = SearchSpace {
        tensor_ranks: vec![9],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let decode_search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: prefill_search,
            decode: decode_search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    let rejection = results[0]
        .rejections
        .iter()
        .find(|rejection| {
            rejection.phase == "prefill" && rejection.code == "insufficient_available_gpus"
        })
        .expect("structured prefill placement rejection");
    assert_eq!(rejection.category, "capacity");
    assert_eq!(rejection.resource, "available_gpus");
    assert_eq!(rejection.observed, Some(8.0));
    assert_eq!(rejection.limit, Some(9.0));
    assert_eq!(rejection.unit.as_deref(), Some("gpus"));
    assert!(
        rejection
            .remediation
            .as_deref()
            .is_some_and(|hint| hint.contains("add GPUs"))
    );
    assert!(
        rejection
            .message
            .contains("config requires 9 ranks but cluster only has 8 available GPUs")
    );
}

#[test]
fn rejects_serving_configs_that_exceed_decode_capacity() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(1),
            max_resident_tokens: Some(1),
            max_decode_sequences_per_node: Some(1),
            max_resident_tokens_per_node: Some(1),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: Some(1),
            max_kv_blocks_per_node: Some(1),
            max_kv_blocks_per_gpu: None,

            ttft_slo_s: None,

            tpot_slo_s: None,

            itl_slo_s: None,

            e2el_slo_s: None,

            max_ttft_slo_miss_rate: None,

            max_tpot_slo_miss_rate: None,

            max_itl_slo_miss_rate: None,

            max_e2el_slo_miss_rate: None,

            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,

            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![2],
            prompt_tokens: vec![128],
            decode_tokens: vec![16],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    assert!(results[0].metrics.peak_decode_sequences > 1);
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("decode capacity exceeded"))
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("KV residency capacity exceeded"))
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("per-node decode capacity exceeded"))
    );
    assert!(results[0].metrics.peak_kv_blocks > 1);
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("KV block capacity exceeded"))
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("per-node KV block capacity exceeded"))
    );
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.phase == "decode"
            && rejection.category == "capacity"
            && rejection.resource == "decode_sequences"
            && rejection.code == "decode_capacity_exceeded"
            && rejection.observed == Some(f64::from(results[0].metrics.peak_decode_sequences))
            && rejection.limit == Some(1.0)
            && rejection.unit.as_deref() == Some("sequences")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("decode replicas"))
            && rejection.message.contains("decode capacity exceeded")
    }));
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.phase == "decode"
            && rejection.category == "capacity"
            && rejection.resource == "resident_tokens"
            && rejection.code == "kv_residency_capacity_exceeded"
            && rejection.observed == Some(results[0].metrics.peak_resident_tokens as f64)
            && rejection.limit == Some(1.0)
            && rejection.unit.as_deref() == Some("tokens")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("KV residency"))
            && rejection.message.contains("KV residency capacity exceeded")
    }));
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.phase == "decode"
            && rejection.category == "capacity"
            && rejection.resource == "kv_blocks"
            && rejection.code == "kv_block_capacity_exceeded"
            && rejection.observed == Some(results[0].metrics.peak_kv_blocks as f64)
            && rejection.limit == Some(1.0)
            && rejection.unit.as_deref() == Some("blocks")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("KV block capacity"))
            && rejection.message.contains("KV block capacity exceeded")
    }));
}

#[test]
fn request_level_decode_capacity_policy_rejects_overflow_requests() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::RequestReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(1),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(1),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: Some(10.0),
            tpot_slo_s: Some(10.0),
            itl_slo_s: Some(10.0),
            e2el_slo_s: Some(10.0),
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: None,
            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![16],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.scheduled_requests, 2);
    assert_eq!(score.metrics.completed_requests, 1);
    assert_eq!(score.metrics.rejected_requests, 1);
    assert_eq!(score.metrics.peak_decode_sequences, 1);
    assert!(score.rejected_reason.is_none());
    assert_eq!(score.request_observations.len(), 2);
    assert_eq!(
        score.request_observations[0].status,
        ServingRequestStatus::Completed
    );
    assert_eq!(
        score.request_observations[1].status,
        ServingRequestStatus::RejectedAdmission
    );
    assert!(
        score.request_observations[1]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("decode admission rejected"))
    );
    let rejection = score.request_observations[1]
        .rejection
        .as_ref()
        .expect("structured decode capacity rejection");
    assert_eq!(rejection.phase, "decode");
    assert_eq!(rejection.category, "capacity");
    assert_eq!(rejection.resource, "decode_sequences");
    assert_eq!(rejection.code, "decode_capacity_exceeded");
    assert_eq!(rejection.observed, Some(2.0));
    assert_eq!(rejection.limit, Some(1.0));
    assert_eq!(rejection.unit.as_deref(), Some("sequences"));
    assert!(
        rejection
            .remediation
            .as_deref()
            .is_some_and(|hint| hint.contains("decode capacity"))
    );
    assert_eq!(score.request_observations[1].decode_iterations, 0);
}

#[test]
fn traffic_class_decode_capacity_rejects_matching_overflow_requests() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::RequestReject,
            trace_requests: vec![
                ServingTraceRequest {
                    request_id: Some("gold-0".to_string()),
                    tenant: Some("gold".to_string()),
                    model_id: None,
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 16,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
                ServingTraceRequest {
                    request_id: Some("gold-1".to_string()),
                    tenant: Some("gold".to_string()),
                    model_id: None,
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 16,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
            ],
            traffic_classes: vec![ServingTrafficClass {
                name: "gold".to_string(),
                group: "tenant".to_string(),
                key: "gold".to_string(),
                max_decode_sequences: Some(1),
                ..ServingTrafficClass::default()
            }],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.scheduled_requests, 2);
    assert_eq!(score.metrics.completed_requests, 1);
    assert_eq!(score.metrics.rejected_requests, 1);
    assert_eq!(score.metrics.peak_decode_sequences, 1);
    assert!(score.rejected_reason.is_none());
    assert_eq!(
        score.request_observations[0].status,
        ServingRequestStatus::Completed
    );
    assert_eq!(
        score.request_observations[1].status,
        ServingRequestStatus::RejectedAdmission
    );
    assert_eq!(
        score.request_observations[1].traffic_class.as_deref(),
        Some("gold")
    );
    let class_capacity = score
        .traffic_class_capacity
        .iter()
        .find(|capacity| capacity.name == "gold")
        .expect("gold class capacity observation");
    assert_eq!(class_capacity.max_decode_sequences, Some(1));
    assert_eq!(class_capacity.peak_decode_sequences, 1);
    assert!((class_capacity.decode_sequence_utilization - 1.0).abs() < 1e-12);
    let rejection = score.request_observations[1]
        .rejection
        .as_ref()
        .expect("structured traffic-class decode capacity rejection");
    assert_eq!(rejection.phase, "decode");
    assert_eq!(rejection.category, "capacity");
    assert_eq!(rejection.resource, "traffic_class gold decode_sequences");
    assert_eq!(rejection.code, "traffic_class_decode_capacity_exceeded");
    assert_eq!(rejection.observed, Some(2.0));
    assert_eq!(rejection.limit, Some(1.0));
    assert_eq!(rejection.unit.as_deref(), Some("sequences"));
    assert!(
        rejection
            .remediation
            .as_deref()
            .is_some_and(|hint| hint.contains("traffic class max_decode_sequences"))
    );
    assert_eq!(score.request_observations[1].decode_iterations, 0);
}

#[test]
fn request_level_prefill_capacity_policy_rejects_overflow_requests() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(1024),
                chunk_tokens: None,
            },
            decode_capacity_policy: ServingDecodeCapacityPolicy::RequestReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: Some(128),
            max_prefill_tokens_per_node: Some(128),
            max_prefill_tokens_per_gpu: Some(128),
            max_prefill_worker_slots_per_gpu: None,
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.completed_requests, 1);
    assert_eq!(score.metrics.rejected_requests, 1);
    assert_eq!(score.metrics.peak_prefill_tokens, 128);
    assert_eq!(score.metrics.peak_prefill_tokens_per_node, 128);
    assert_eq!(score.metrics.peak_prefill_tokens_per_gpu, 128);
    assert!(score.rejected_reason.is_none());
    assert!(
        !score
            .rejections
            .iter()
            .any(|rejection| rejection.code.starts_with("prefill_capacity"))
    );

    let rejected = score
        .request_observations
        .iter()
        .find(|observation| observation.status == ServingRequestStatus::RejectedAdmission)
        .expect("one request should be rejected by prefill admission");
    assert!(
        rejected
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("prefill admission rejected"))
    );
    let rejection = rejected
        .rejection
        .as_ref()
        .expect("structured prefill capacity rejection");
    assert_eq!(rejection.phase, "prefill");
    assert_eq!(rejection.category, "capacity");
    assert_eq!(rejection.code, "prefill_active_tokens_exceeded");
    assert_eq!(rejection.observed, Some(256.0));
    assert_eq!(rejection.limit, Some(128.0));
    assert_eq!(rejection.unit.as_deref(), Some("tokens"));
    assert_eq!(rejected.decode_iterations, 0);
}

#[test]
fn traffic_class_prefill_capacity_rejects_matching_overflow_requests() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(1024),
                chunk_tokens: None,
            },
            decode_capacity_policy: ServingDecodeCapacityPolicy::RequestReject,
            trace_requests: vec![
                ServingTraceRequest {
                    request_id: Some("gold-prefill-0".to_string()),
                    tenant: Some("gold".to_string()),
                    model_id: None,
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 1,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
                ServingTraceRequest {
                    request_id: Some("gold-prefill-1".to_string()),
                    tenant: Some("gold".to_string()),
                    model_id: None,
                    cache_key: None,
                    arrival_s: 0.0,
                    priority: 0,
                    batch_size: 1,
                    prompt_tokens: 128,
                    decode_tokens: 1,
                    max_sequence_tokens: None,
                    prefix_cache_hit_tokens: None,
                    prefix_cache_hit_rate: None,
                    slo: ServingRequestSlo::default(),
                    deadline_s: None,
                    cancellation_s: None,
                },
            ],
            traffic_classes: vec![ServingTrafficClass {
                name: "gold".to_string(),
                group: "tenant".to_string(),
                key: "gold".to_string(),
                max_prefill_tokens: Some(128),
                ..ServingTrafficClass::default()
            }],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    let score = &results[0];
    assert!(score.feasible);
    assert_eq!(score.metrics.scheduled_requests, 2);
    assert_eq!(score.metrics.completed_requests, 1);
    assert_eq!(score.metrics.rejected_requests, 1);
    assert_eq!(score.metrics.peak_prefill_tokens, 128);
    assert!(score.rejected_reason.is_none());
    assert_eq!(
        score.request_observations[1].status,
        ServingRequestStatus::RejectedAdmission
    );
    assert_eq!(
        score.request_observations[1].traffic_class.as_deref(),
        Some("gold")
    );
    let class_capacity = score
        .traffic_class_capacity
        .iter()
        .find(|capacity| capacity.name == "gold")
        .expect("gold class capacity observation");
    assert_eq!(class_capacity.max_prefill_tokens, Some(128));
    assert_eq!(class_capacity.peak_prefill_tokens, 128);
    assert!((class_capacity.prefill_token_utilization - 1.0).abs() < 1e-12);
    let rejection = score.request_observations[1]
        .rejection
        .as_ref()
        .expect("structured traffic-class prefill capacity rejection");
    assert_eq!(rejection.phase, "prefill");
    assert_eq!(rejection.category, "capacity");
    assert_eq!(rejection.resource, "traffic_class gold prefill_tokens");
    assert_eq!(
        rejection.code,
        "traffic_class_prefill_active_tokens_exceeded"
    );
    assert_eq!(rejection.observed, Some(256.0));
    assert_eq!(rejection.limit, Some(128.0));
    assert_eq!(rejection.unit.as_deref(), Some("tokens"));
    assert!(
        rejection
            .remediation
            .as_deref()
            .is_some_and(|hint| hint.contains("traffic class max_prefill_tokens"))
    );
    assert_eq!(score.request_observations[1].decode_iterations, 0);
}

#[test]
fn reports_and_rejects_per_node_decode_capacity() {
    let cluster = Cluster::h100_sxm_nodes(3, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1, 2],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(1),
            max_resident_tokens_per_node: Some(511),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,

            ttft_slo_s: None,

            tpot_slo_s: None,

            itl_slo_s: None,

            e2el_slo_s: None,

            max_ttft_slo_miss_rate: None,

            max_tpot_slo_miss_rate: None,

            max_itl_slo_miss_rate: None,

            max_e2el_slo_miss_rate: None,

            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,

            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![2],
            prompt_tokens: vec![128],
            decode_tokens: vec![16],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    let decode_node_capacity = results[0]
        .node_capacity
        .iter()
        .filter(|node| node.peak_decode_sequences > 0)
        .collect::<Vec<_>>();
    assert_eq!(decode_node_capacity.len(), 2);
    assert_eq!(decode_node_capacity[0].node_id, 1);
    assert_eq!(decode_node_capacity[1].node_id, 2);
    assert!(
        results[0]
            .node_capacity
            .iter()
            .any(|node| node.node_id == 0 && node.peak_prefill_tokens > 0)
    );
    assert!(results[0].metrics.peak_decode_sequences >= 2);
    assert_eq!(results[0].metrics.peak_decode_sequences_per_node, 2);
    assert_eq!(
        results[0]
            .request_observations
            .iter()
            .map(|observation| observation.decode_node)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        results[0]
            .request_observations
            .iter()
            .map(|observation| observation.decode_route_nodes.clone())
            .collect::<Vec<_>>(),
        vec![vec![1], vec![2]]
    );
    assert!(results[0].metrics.kv_transfer_s > 0.0);
    assert!(
        results[0]
            .resource_utilization
            .iter()
            .any(|resource| resource.resource == "gpu compute node 2")
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("per-node decode capacity exceeded"))
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("per-node KV residency capacity exceeded"))
    );
}

#[test]
fn reports_and_rejects_per_gpu_kv_capacity() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![2],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: Some(1),
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: Some(255),
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: None,

            tpot_slo_s: None,

            itl_slo_s: None,

            e2el_slo_s: None,

            max_ttft_slo_miss_rate: None,

            max_tpot_slo_miss_rate: None,

            max_itl_slo_miss_rate: None,

            max_e2el_slo_miss_rate: None,

            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,

            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![2],
            prompt_tokens: vec![128],
            decode_tokens: vec![16],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    let decode_gpu_capacity = results[0]
        .gpu_capacity
        .iter()
        .filter(|gpu| gpu.peak_decode_sequences > 0)
        .collect::<Vec<_>>();
    assert_eq!(decode_gpu_capacity.len(), 2);
    assert_eq!(results[0].metrics.peak_decode_sequences_per_gpu, 2);
    assert_eq!(results[0].metrics.peak_resident_tokens_per_gpu, 256);
    assert!(decode_gpu_capacity.iter().all(|gpu| {
        gpu.node_id == 1
            && gpu.peak_decode_sequences == 2
            && gpu.peak_resident_tokens == 256
            && gpu.decode_sequence_utilization == 2.0
    }));
    assert!(
        results[0]
            .gpu_capacity
            .iter()
            .any(|gpu| gpu.node_id == 0 && gpu.peak_prefill_tokens > 0)
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("per-GPU decode capacity exceeded"))
    );
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("per-GPU KV residency capacity exceeded"))
    );
}

#[test]
fn topology_aware_routing_prefers_lower_kv_transfer_cost() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [interconnect]
        kind = "ethernet"
        variant = "10g"

        [[interconnect.links]]
        from = 0
        to = 1
        kind = "ethernet"
        variant = "10g"

        [[interconnect.links]]
        from = 0
        to = 2
        kind = "ib"
        variant = "ndr"

        [[node_groups]]
        label = "prefill"
        start_id = 0
        count = 1
        gpu = "h100_sxm"
        gpu_count = 8
        intra = "nvlink_v4"
        nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }

        [[node_groups]]
        label = "decode"
        start_id = 1
        count = 2
        gpu = "h100_sxm"
        gpu_count = 8
        intra = "nvlink_v4"
        nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1, 2],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::TopologyAware,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,

            ttft_slo_s: None,

            tpot_slo_s: None,

            itl_slo_s: None,

            e2el_slo_s: None,

            max_ttft_slo_miss_rate: None,

            max_tpot_slo_miss_rate: None,

            max_itl_slo_miss_rate: None,

            max_e2el_slo_miss_rate: None,

            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,

            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![1024],
            decode_tokens: vec![1],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    assert_eq!(results[0].request_observations[0].prefill_node, 0);
    assert_eq!(results[0].request_observations[0].decode_node, 2);
    assert_eq!(
        results[0].request_observations[0].decode_route_nodes,
        vec![2]
    );
    assert_eq!(
        results[0].request_observations[0].routing_policy,
        ServingRoutingPolicy::TopologyAware
    );
    assert_eq!(
        results[0].request_observations[0].routing_candidate_count,
        2
    );
    assert_eq!(
        results[0].request_observations[0].routing_routable_candidate_count,
        2
    );
    assert_eq!(results[0].route_coverage.candidate_count, 2);
    assert_eq!(results[0].route_coverage.routable_candidate_count, 2);
    assert_eq!(results[0].route_coverage.fraction, 1.0);
    assert!(
        results[0].request_observations[0]
            .routing_estimated_e2el_s
            .is_finite()
    );
    assert!(
        results[0].request_observations[0]
            .routing_estimated_kv_transfer_s
            .is_finite()
    );
    assert!(
        results[0].request_observations[0]
            .routing_reason
            .contains("topology-aware router selected")
    );
    assert!(results[0].request_observations[0].kv_transfer_bytes > 0);
    assert!(
        results[0].request_observations[0]
            .kv_transfer_bottlenecks
            .iter()
            .any(|bottleneck| bottleneck.contains("IB NDR"))
    );
    assert_eq!(
        results[0].request_observations[0].prefill_route_gpus,
        vec![GpuAddr {
            node_id: 0,
            local_gpu_id: 0,
        }]
    );
    assert_eq!(
        results[0].request_observations[0].decode_route_gpus,
        vec![GpuAddr {
            node_id: 2,
            local_gpu_id: 0,
        }]
    );
    assert_eq!(
        results[0].request_observations[0].kv_cache_owner_gpus,
        results[0].request_observations[0].decode_route_gpus
    );
    assert!(
        results[0].request_observations[0]
            .kv_transfer_paths
            .iter()
            .any(|path| {
                path.source.node_id == 0
                    && path.destination.node_id == 2
                    && path.bottleneck_bandwidth_gbps > 0.0
                    && path
                        .resources
                        .iter()
                        .any(|resource| resource.contains("IB NDR"))
                    && path.resource_details.iter().any(|resource| {
                        resource.kind == "inter_node_fabric"
                            && resource.rail_id.is_some()
                            && resource
                                .from
                                .as_ref()
                                .is_some_and(|endpoint| endpoint.kind == "nic")
                            && resource
                                .to
                                .as_ref()
                                .is_some_and(|endpoint| endpoint.kind == "nic")
                    })
            })
    );
    assert!(results[0].kv_route_resource_summary.iter().any(|resource| {
        resource.kind == "inter_node_fabric"
            && resource.request_count > 0
            && resource.path_observations > 0
            && resource.transfer_bytes > 0
            && resource.estimated_transfer_s > 0.0
            && resource.rail_id.is_some()
    }));
    assert!(results[0].kv_route_topology_summary.route_resource_count >= 3);
    assert_eq!(
        results[0]
            .kv_route_topology_summary
            .inter_node_route_resource_count,
        1
    );
    assert!(
        results[0]
            .kv_route_topology_summary
            .gpu_nic_route_resource_count
            >= 2
    );
    assert_eq!(results[0].kv_route_topology_summary.rail_ids, vec![0]);
    assert_eq!(results[0].kv_route_topology_summary.rail_count, 1);
    assert!(results[0].kv_route_topology_summary.single_rail_dependency);
    assert_eq!(results[0].kv_route_topology_summary.single_rail_id, Some(0));
    assert!(results[0].topology_bottlenecks.iter().any(|bottleneck| {
        bottleneck.code == "single_rail_dependency"
            && bottleneck.severity == "warning"
            && bottleneck.observed == Some(1.0)
            && bottleneck.limit == Some(2.0)
            && bottleneck.unit.as_deref() == Some("rails")
            && bottleneck.message.contains("single inter-node rail")
    }));
    assert!(
        results[0]
            .bottlenecks
            .contains(&"single_rail_dependency".to_string())
    );
    assert!(
        results[0]
            .resource_utilization
            .iter()
            .any(|resource| resource.resource == "gpu compute node 2")
    );
}

#[test]
fn rejects_serving_configs_when_kv_route_rail_count_is_below_minimum() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [interconnect]
        kind = "ib"
        variant = "ndr"

        [[interconnect.links]]
        from = 0
        to = 1
        kind = "ib"
        variant = "ndr"
        rail = 0

        [[node_groups]]
        label = "prefill"
        start_id = 0
        count = 1
        gpu = "h100_sxm"
        gpu_count = 1
        intra = "nvlink_v4"
        nics = { count = 1, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 1 }

        [[node_groups]]
        label = "decode"
        start_id = 1
        count = 1
        gpu = "h100_sxm"
        gpu_count = 1
        intra = "nvlink_v4"
        nics = { count = 1, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 1 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::FullyDisaggregated,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            kv_route_constraints: ServingKvRouteConstraints {
                min_inter_node_rail_count: Some(2),
                ..ServingKvRouteConstraints::default()
            },
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    assert_eq!(
        results[0].kv_route_constraints.min_inter_node_rail_count,
        Some(2)
    );
    assert_eq!(results[0].kv_route_topology_summary.rail_count, 1);
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.category == "topology"
            && rejection.phase == "kv_transfer"
            && rejection.resource == "kv_route_rail"
            && rejection.code == "kv_route_rail_count_below_min"
            && rejection.observed == Some(1.0)
            && rejection.limit == Some(2.0)
            && rejection.unit.as_deref() == Some("rails")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("min_kv_route_rail_count"))
    }));
}

#[test]
fn kv_route_constraints_reject_missing_rail_metadata_and_host_staged_paths() {
    let topology = ServingKvRouteTopologySummary {
        route_resource_count: 2,
        inter_node_route_resource_count: 1,
        unrailed_inter_node_route_resource_count: 1,
        ..ServingKvRouteTopologySummary::default()
    };
    let resources = vec![ServingKvRouteResourceSummary {
        resource_id: "kv_route:gpu_nic_local|node=0|gpu=0|nic=0".to_string(),
        kind: "gpu_nic_local".to_string(),
        label: "node 0 gpu 0 -> nic 0 path host_staged".to_string(),
        request_count: 1,
        path_observations: 1,
        transfer_bytes: 1024,
        estimated_transfer_s: 0.001,
        min_bandwidth_gbps: 100.0,
        max_latency_s: 0.000_010,
        rail_id: Some(0),
        from: None,
        to: None,
    }];

    let rejections = kv_route_constraint_rejections(
        &topology,
        &resources,
        ServingKvRouteConstraints {
            require_inter_node_rail_metadata: true,
            require_gpudirect: true,
            ..ServingKvRouteConstraints::default()
        },
    );

    assert!(rejections.iter().any(|rejection| {
        rejection.code == "kv_route_rail_metadata_missing"
            && rejection.resource == "kv_route_rail"
            && rejection.limit == Some(0.0)
            && rejection.unit.as_deref() == Some("resources")
    }));
    assert!(rejections.iter().any(|rejection| {
        rejection.code == "host_staged_kv_path_disallowed"
            && rejection.resource == "gpu_nic_local"
            && rejection.observed == Some(1.0)
            && rejection.unit.as_deref() == Some("resources")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("require_gpudirect_kv_paths"))
    }));
}

#[test]
fn topology_bottlenecks_report_route_locality_risks() {
    let resources = vec![
        ServingKvRouteResourceSummary {
            resource_id: "gpu-nic-host-staged".to_string(),
            kind: "gpu_nic_local".to_string(),
            label: "node 0 gpu 0 -> nic 0 rail 0 path host_staged/host-staged".to_string(),
            request_count: 2,
            path_observations: 2,
            transfer_bytes: 4096,
            estimated_transfer_s: 0.0002,
            min_bandwidth_gbps: 20.0,
            max_latency_s: 100e-6,
            rail_id: Some(0),
            from: None,
            to: None,
        },
        ServingKvRouteResourceSummary {
            resource_id: "gpu-nic-cross-socket".to_string(),
            kind: "gpu_nic_local".to_string(),
            label: "node 1 gpu 0 -> nic 1 rail 0 path cross_socket/gpudirect".to_string(),
            request_count: 2,
            path_observations: 2,
            transfer_bytes: 4096,
            estimated_transfer_s: 0.00005,
            min_bandwidth_gbps: 64.0,
            max_latency_s: 8e-6,
            rail_id: Some(0),
            from: None,
            to: None,
        },
        ServingKvRouteResourceSummary {
            resource_id: "inter-node".to_string(),
            kind: "inter_node_fabric".to_string(),
            label: "IB NDR node 0 <-> node 1 rail 0".to_string(),
            request_count: 2,
            path_observations: 2,
            transfer_bytes: 4096,
            estimated_transfer_s: 0.00002,
            min_bandwidth_gbps: 400.0,
            max_latency_s: 3e-6,
            rail_id: Some(0),
            from: None,
            to: None,
        },
    ];
    let route_coverage = ServingRouteCoverage {
        candidate_count: 1,
        routable_candidate_count: 1,
        unroutable_candidate_count: 0,
        fraction: 1.0,
    };
    let topology = kv_route_topology_summary(&resources);
    let bottlenecks =
        topology_bottleneck_observations(route_coverage, &topology, &resources, &[], &[]);

    assert!(bottlenecks.iter().any(|bottleneck| {
        bottleneck.code == "host_staged_kv_path"
            && bottleneck.severity == "warning"
            && bottleneck.observed == Some(20.0)
            && bottleneck.unit.as_deref() == Some("gbps")
            && bottleneck.message.contains("lack GPUDirect")
    }));
    assert!(bottlenecks.iter().any(|bottleneck| {
        bottleneck.code == "cross_socket_kv_path"
            && bottleneck.severity == "warning"
            && bottleneck.observed == Some(8.0)
            && bottleneck.unit.as_deref() == Some("us")
    }));
    assert!(bottlenecks.iter().any(|bottleneck| {
        bottleneck.code == "slow_gpu_nic_kv_path"
            && bottleneck.severity == "info"
            && bottleneck.observed == Some(20.0)
            && bottleneck.limit == Some(200.0)
            && bottleneck.unit.as_deref() == Some("gbps")
    }));
}

#[test]
fn topology_bottlenecks_report_hot_kv_route_resources() {
    let utilization = vec![ServingPhaseResourceUtilization {
        phase: "kv_transfer".to_string(),
        resource_kind: "kv_route".to_string(),
        resource: "kv_route:inter_node_fabric|rail=0".to_string(),
        busy_s: 0.9,
        utilization: 0.9,
        operation_count: 4,
        first_start_s: 0.0,
        last_finish_s: 1.0,
    }];

    let bottlenecks = dynamic_route_contention_bottleneck_observations(&[], &utilization);

    assert!(bottlenecks.iter().any(|bottleneck| {
        bottleneck.code == "hot_kv_route_resource"
            && bottleneck.category == "contention"
            && bottleneck.severity == "warning"
            && bottleneck.observed == Some(0.9)
            && bottleneck.limit == Some(0.85)
            && bottleneck.unit.as_deref() == Some("utilization")
            && bottleneck.message.contains("90.0% utilization")
    }));
}

#[test]
fn kv_transfers_on_disjoint_routes_can_overlap() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[interconnect.links]]
        from = 0
        to = 1
        kind = "ib"
        variant = "ndr"

        [[interconnect.links]]
        from = 2
        to = 3
        kind = "ib"
        variant = "ndr"

        [[node_groups]]
        label = "all"
        start_id = 0
        count = 4
        gpu = "h100_sxm"
        gpu_count = 1
        intra = "nvlink_v4"
        nics = { count = 1, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 1 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0, 2],
        decode_nodes: vec![1, 3],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            batch_sizes: vec![1],
            prompt_tokens: vec![1024],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    let observations = &results[0].request_observations;
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].prefill_node, 0);
    assert_eq!(observations[0].decode_node, 1);
    assert_eq!(observations[1].prefill_node, 2);
    assert_eq!(observations[1].decode_node, 3);
    assert!((observations[0].kv_start_s - observations[1].kv_start_s).abs() < 1e-12);
    assert!((observations[0].kv_finish_s - observations[1].kv_finish_s).abs() < 1e-12);
    assert!(observations.iter().all(|observation| {
        observation.kv_resource_queue_s == 0.0
            && observation.kv_transfer_resource_dependencies.is_empty()
            && observation
                .kv_transfer_resources
                .iter()
                .all(|resource| resource.starts_with("kv_route:"))
    }));

    let kv_operations = results[0]
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("kv-transfer"))
        .collect::<Vec<_>>();
    assert_eq!(kv_operations.len(), 2);
    assert!(kv_operations.iter().all(|operation| {
        operation
            .resources
            .iter()
            .all(|resource| resource.starts_with("kv_route:"))
    }));
    assert!(kv_operations.iter().all(|operation| {
        !operation
            .resources
            .iter()
            .any(|resource| resource == "KV transfer fabric/NIC path")
    }));
    assert!(
        kv_operations
            .iter()
            .all(|operation| operation.resource_dependencies.is_empty())
    );
    assert!(
        results[0]
            .phase_resource_utilization
            .iter()
            .any(|row| row.phase == "kv_transfer" && row.resource_kind == "kv_route")
    );
    assert!(
        results[0]
            .kv_route_resource_summary
            .iter()
            .filter(|resource| resource.kind == "inter_node_fabric")
            .count()
            >= 2
    );
}

#[test]
fn kv_transfer_worker_slots_report_worker_queue() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[interconnect.links]]
        from = 0
        to = 1
        kind = "ib"
        variant = "ndr"

        [[interconnect.links]]
        from = 0
        to = 2
        kind = "ethernet"
        variant = "400g"

        [[node_groups]]
        label = "all"
        start_id = 0
        count = 3
        gpu = "h100_sxm"
        gpu_count = 1
        intra = "nvlink_v4"
        nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 2 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1, 2],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(4096),
                chunk_tokens: None,
            },
            decode_batching: ServingDecodeBatching::Independent,
            max_kv_transfer_worker_slots_per_gpu: Some(1),
            batch_sizes: vec![1],
            prompt_tokens: vec![1024],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    let observations = &results[0].request_observations;
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].prefill_node, 0);
    assert_eq!(observations[0].decode_node, 1);
    assert_eq!(observations[1].prefill_node, 0);
    assert_eq!(observations[1].decode_node, 2);
    assert!((observations[0].prefill_finish_s - observations[1].prefill_finish_s).abs() < 1e-12);
    assert!(
        observations
            .iter()
            .any(|observation| observation.kv_worker_queue_s > 0.0)
    );
    assert!(observations.iter().all(|observation| {
        observation.kv_resource_queue_s == 0.0
            && observation.kv_transfer_resource_dependencies.is_empty()
    }));
    assert!(results[0].metrics.kv_worker_queue_s > 0.0);
    assert_eq!(results[0].metrics.kv_resource_queue_s, 0.0);
    let kv_worker = results[0]
        .worker_observations
        .iter()
        .find(|worker| {
            worker.phase == "kv_transfer" && worker.node_id == 0 && worker.local_gpu_id == 0
        })
        .expect("shared prefill-side KV transfer worker");
    assert_eq!(kv_worker.configured_worker_slots, 1);
    assert_eq!(kv_worker.request_count, 2);
    assert!(kv_worker.worker_queue_s > 0.0);
    assert_eq!(kv_worker.resource_queue_s, 0.0);
    assert_eq!(kv_worker.peak_active_worker_slots, 1);
    assert!(kv_worker.worker_slot_utilization > 0.0);
}

#[test]
fn kv_queue_delay_cap_rejects_before_decode_admission() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[interconnect.links]]
        from = 0
        to = 1
        kind = "ib"
        variant = "ndr"

        [[interconnect.links]]
        from = 0
        to = 2
        kind = "ethernet"
        variant = "400g"

        [[node_groups]]
        label = "all"
        start_id = 0
        count = 3
        gpu = "h100_sxm"
        gpu_count = 1
        intra = "nvlink_v4"
        nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 2 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1, 2],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(4096),
                chunk_tokens: None,
            },
            decode_batching: ServingDecodeBatching::Independent,
            max_kv_transfer_worker_slots_per_gpu: Some(1),
            max_kv_queue_delay_s: Some(0.0),
            service_backpressure_penalty_weight: 3.0,
            batch_sizes: vec![1],
            prompt_tokens: vec![1024],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    let observations = &results[0].request_observations;
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].status, ServingRequestStatus::Completed);
    assert_eq!(
        observations[1].status,
        ServingRequestStatus::RejectedAdmission
    );
    assert!(
        observations[1]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("KV transfer queue delay"))
    );
    let rejection = observations[1]
        .rejection
        .as_ref()
        .expect("structured KV queue rejection");
    assert_eq!(rejection.phase, "kv_transfer");
    assert_eq!(rejection.category, "queueing");
    assert_eq!(rejection.code, "kv_queue_delay_exceeded");
    assert_eq!(rejection.unit.as_deref(), Some("s"));
    assert!(observations[1].kv_worker_queue_s > 0.0);
    assert!(observations[1].first_decode_start_s.is_infinite());

    let kv_operations = results[0]
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("kv-transfer"))
        .count();
    assert_eq!(kv_operations, 1);
    assert_eq!(results[0].metrics.completed_requests, 1);
    assert_eq!(results[0].metrics.rejected_requests, 1);
    let kv_service = results[0]
        .service_observations
        .iter()
        .find(|service| service.phase == "kv_transfer")
        .expect("KV transfer service observation");
    assert_eq!(kv_service.request_count, 2);
    assert_eq!(kv_service.completed_requests, 1);
    assert_eq!(kv_service.rejected_requests, 1);
    assert_eq!(kv_service.queue_cap_request_count, 2);
    assert_eq!(kv_service.queue_cap_hit_count, 1);
    assert_eq!(kv_service.backpressure_rejections, 1);
    assert_eq!(kv_service.backpressure_state, "rejecting");
    assert_eq!(kv_service.queue_cap_s, Some(0.0));
    assert!(kv_service.queue_max_s > 0.0);
    assert_eq!(results[0].service_backpressure_penalty_weight, 3.0);
    assert!(results[0].service_backpressure_penalty_score > 0.0);
    assert!(results[0].approximations.iter().any(|approximation| {
        approximation.code == "kv_transfer_to_prefill_backpressure_not_modeled"
            && approximation
                .message
                .contains("KV-transfer service reported")
    }));
}

#[test]
fn decode_queue_delay_cap_rejects_before_decode_reservation() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(4096),
                chunk_tokens: None,
            },
            decode_batching: ServingDecodeBatching::Independent,
            max_decode_worker_slots_per_gpu: Some(1),
            max_decode_queue_delay_s: Some(0.0),
            batch_sizes: vec![1],
            prompt_tokens: vec![1024],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    let observations = &results[0].request_observations;
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].status, ServingRequestStatus::Completed);
    assert_eq!(
        observations[1].status,
        ServingRequestStatus::RejectedAdmission
    );
    assert!(
        observations[1]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("decode queue delay"))
    );
    let rejection = observations[1]
        .rejection
        .as_ref()
        .expect("structured decode queue rejection");
    assert_eq!(rejection.phase, "decode");
    assert_eq!(rejection.category, "queueing");
    assert_eq!(rejection.code, "decode_queue_delay_exceeded");
    assert_eq!(rejection.unit.as_deref(), Some("s"));
    assert!(observations[1].decode_queue_s > 0.0);
    assert!(observations[1].decode_worker_queue_s > 0.0);

    let decode_operations = results[0]
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("request 1 decode token"))
        .count();
    assert_eq!(decode_operations, 0);
    assert_eq!(results[0].metrics.completed_requests, 1);
    assert_eq!(results[0].metrics.rejected_requests, 1);
    let decode_service = results[0]
        .service_observations
        .iter()
        .find(|service| service.phase == "decode")
        .expect("decode service observation");
    assert_eq!(decode_service.request_count, 2);
    assert_eq!(decode_service.completed_requests, 1);
    assert_eq!(decode_service.rejected_requests, 1);
    assert_eq!(decode_service.queue_cap_request_count, 2);
    assert_eq!(decode_service.queue_cap_hit_count, 1);
    assert_eq!(decode_service.backpressure_rejections, 1);
    assert_eq!(decode_service.backpressure_state, "rejecting");
    assert_eq!(decode_service.queue_cap_s, Some(0.0));
    assert!(decode_service.queue_max_s > 0.0);
    assert!(results[0].approximations.iter().any(|approximation| {
        approximation.code == "decode_to_prefill_backpressure_not_modeled"
            && approximation.message.contains("Decode service reported")
    }));
}

#[test]
fn decode_iteration_queue_delay_cap_times_out_partial_decode() {
    let gpu = GpuAddr {
        node_id: 0,
        local_gpu_id: 0,
    };
    let decode_score = worker_decode_score(1.0);
    let mut scheduler = ResourceScheduler::new();
    let mut worker_runtime = ServingWorkerRuntime::default();
    worker_runtime.decode_ready_s.insert(gpu, vec![1.0]);
    let mut state = pending_decode_state(0, gpu);
    state.decode_tokens = 2;
    state.remaining_tokens = 1;
    state.emitted_tokens = 1;
    state.kv_finish_s = 0.0;
    state.first_decode_start_s = Some(0.0);
    state.first_decode_finish_s = Some(0.0);
    state.last_decode_finish_s = Some(0.0);
    state.decode_token_start_s = vec![0.0];
    state.decode_token_finish_s = vec![0.0];
    state.max_decode_iteration_queue_delay_s = Some(0.5);
    let mut states = vec![state];
    let mut decode_iterations = Vec::new();

    schedule_independent_decodes(
        &mut scheduler,
        &mut worker_runtime,
        &mut states,
        &mut decode_iterations,
        &decode_score,
        1.0,
        1,
        1,
    );

    assert_eq!(states[0].status, ServingRequestStatus::TimedOut);
    assert_eq!(states[0].status_time_s, Some(0.5));
    assert_eq!(states[0].remaining_tokens, 0);
    assert_eq!(states[0].emitted_tokens, 1);
    assert_eq!(states[0].decode_token_finish_s, vec![0.0]);
    assert!(states[0].decode_worker_queue_s > 0.0);
    assert!(
        states[0]
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("decode iteration queue delay"))
    );
    let rejection = states[0]
        .failure_rejection
        .as_ref()
        .expect("structured decode iteration timeout");
    assert_eq!(rejection.phase, "decode");
    assert_eq!(rejection.category, "queueing");
    assert_eq!(rejection.code, "decode_iteration_queue_delay_exceeded");
    assert_eq!(rejection.observed, Some(1.0));
    assert_eq!(rejection.limit, Some(0.5));
    assert_eq!(rejection.unit.as_deref(), Some("s"));
    assert!(decode_iterations.is_empty());
    assert!(scheduler.operations().is_empty());

    let observation = ServingRequestObservation::from_decode_state(&states[0]);
    assert_eq!(observation.status, ServingRequestStatus::TimedOut);
    assert_eq!(
        observation
            .rejection
            .as_ref()
            .map(|rejection| rejection.code.as_str()),
        Some("decode_iteration_queue_delay_exceeded")
    );
    assert_eq!(observation.status_time_s, Some(0.5));
    assert_eq!(observation.decode_iterations, 1);
    assert_eq!(observation.kv_block_ownership.len(), 1);
    assert_eq!(observation.kv_block_ownership[0].released_at_s, 0.5);
}

#[test]
fn kv_transfers_on_shared_route_report_resource_queue() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[interconnect.links]]
        from = 0
        to = 1
        kind = "ib"
        variant = "ndr"

        [[node_groups]]
        label = "all"
        start_id = 0
        count = 2
        gpu = "h100_sxm"
        gpu_count = 1
        intra = "nvlink_v4"
        nics = { count = 1, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 1 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(4096),
                chunk_tokens: None,
            },
            decode_batching: ServingDecodeBatching::Independent,
            batch_sizes: vec![1],
            prompt_tokens: vec![1024],
            decode_tokens: vec![1],
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    let observations = &results[0].request_observations;
    assert_eq!(observations.len(), 2);
    assert!((observations[0].prefill_finish_s - observations[1].prefill_finish_s).abs() < 1e-12);
    assert!(
        observations
            .iter()
            .any(|observation| observation.kv_resource_queue_s > 0.0)
    );
    assert!(results[0].metrics.kv_resource_queue_s > 0.0);
    assert!(results[0].phase_calibration.iter().any(|phase| {
        phase.phase == "kv_route_resource_queue"
            && phase.active
            && !phase.calibrated
            && phase.status == "uncalibrated_no_profile"
            && phase.estimated_s > 0.0
    }));
    assert!(observations.iter().any(|observation| {
        !observation.kv_transfer_resource_dependencies.is_empty()
            && observation
                .kv_transfer_resources
                .iter()
                .all(|resource| resource.starts_with("kv_route:"))
    }));
    assert!(results[0].topology_bottlenecks.iter().any(|bottleneck| {
        bottleneck.code == "kv_route_resource_queueing"
            && bottleneck.category == "contention"
            && bottleneck.severity == "warning"
            && bottleneck.observed.is_some_and(|observed| observed > 0.0)
            && bottleneck.unit.as_deref() == Some("ms")
    }));

    let profile = CalibrationProfileMetadata {
        path: "in-memory".to_string(),
        name: Some("queue-profile".to_string()),
        hardware: Some("h100_sxm".to_string()),
        fabric: Some("ib_ndr".to_string()),
        model: Some("test-model".to_string()),
        dtype: Some("bf16".to_string()),
        serving_stack: Some("unit-test".to_string()),
        serving_runtime_features: Vec::new(),
        backend_version: None,
        driver_version: None,
        cuda_version: None,
        rocm_version: None,
        nccl_version: None,
        rccl_version: None,
        ucx_version: None,
        kernel_settings: Vec::new(),
        environment_hash: None,
        source: Some("unit-test".to_string()),
        date: Some("2026-05-26".to_string()),
        notes: None,
        valid_shape: None,
        invalid_shapes: Vec::new(),
        fits: Vec::new(),
        benchmarks: Vec::new(),
    };
    let profiled = ServingSolver::rank_disaggregated_with_options(
        &cluster,
        &model(),
        &request(),
        &serving,
        ServingSolverOptions {
            calibration_profile: Some(&profile),
            model_id: Some("test-model"),
            ..ServingSolverOptions::default()
        },
    );
    assert!(profiled[0].phase_calibration.iter().any(|phase| {
        phase.phase == "kv_route_resource_queue"
            && phase.active
            && !phase.calibrated
            && phase.status == "uncalibrated_no_fit"
            && phase.estimated_s > 0.0
    }));
    assert!(profiled[0].approximations.iter().any(|approximation| {
        approximation.scope == "kv_route_resource_queue"
            && approximation.code == "serving_queue_component_uncalibrated"
    }));

    let kv_operations = results[0]
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("kv-transfer"))
        .collect::<Vec<_>>();
    assert_eq!(kv_operations.len(), 2);
    assert!(
        kv_operations
            .iter()
            .any(|operation| !operation.resource_dependencies.is_empty())
    );
}

#[test]
fn topology_aware_routing_accounts_for_kv_route_resource_load() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [[interconnect.links]]
        from = 0
        to = 1
        kind = "ethernet"
        variant = "100g"
        rail = 1

        [[interconnect.links]]
        from = 0
        to = 2
        kind = "ib"
        variant = "ndr"
        rail = 0

        [[node_groups]]
        label = "all"
        start_id = 0
        count = 3
        gpu = "h100_sxm"
        gpu_count = 1
        intra = "nvlink_v4"
        nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 2 }
        "#,
    )
    .unwrap();
    let model = model();
    let request_shape = request();
    let score = worker_decode_score(0.001);
    let prefill_route_gpus = vec![GpuAddr {
        node_id: 0,
        local_gpu_id: 0,
    }];
    let fast_decode_route_gpus = vec![GpuAddr {
        node_id: 2,
        local_gpu_id: 0,
    }];
    let fast_kv_bytes = kv_transfer_bytes_for_routes(
        &model,
        &request_shape,
        &[0],
        &[2],
        &prefill_route_gpus,
        &fast_decode_route_gpus,
    );
    let fast_kv_cost = Solver::estimate_transfer_between_gpus_with_options(
        &cluster,
        &prefill_route_gpus,
        &fast_decode_route_gpus,
        fast_kv_bytes,
        SolverOptions {
            calibration: SimulationCalibration::default(),
            calibration_profile: None,
            max_candidates: None,
            search_deadline: None,
            explicit_placement: None,
        },
    );
    let fast_kv_paths = kv_transfer_paths(
        &cluster,
        &prefill_route_gpus,
        &fast_decode_route_gpus,
        fast_kv_bytes,
    );
    let fast_kv_resources =
        kv_transfer_scheduler_resources(&fast_kv_paths, &fast_kv_cost.bottlenecks);
    assert!(!fast_kv_resources.is_empty());

    let mut routing_load = RoutingLoad::default();
    for resource in &fast_kv_resources {
        routing_load
            .kv_route_ready_s
            .insert(resource.clone(), 10_000.0);
    }
    let fast_estimate = route_estimate(
        &request_shape,
        request_shape.prompt_tokens,
        0.0,
        0,
        2,
        &prefill_route_gpus,
        &fast_decode_route_gpus,
        &score,
        &score,
        1.0,
        request_shape.batch_size,
        request_shape.prompt_tokens,
        1,
        1,
        fast_kv_cost.total_s,
        &fast_kv_resources,
        &routing_load,
    );
    assert!(fast_estimate.kv_resource_wait_s > 9_999.0);

    let selected = route_topology_aware_request(
        &cluster,
        &model,
        &request_shape,
        request_shape.prompt_tokens,
        0.0,
        &[0],
        &[1, 2],
        Some(0),
        &[0],
        Some(0),
        &[0],
        &score,
        &score,
        1.0,
        request_shape.batch_size,
        request_shape.prompt_tokens,
        1,
        1,
        SimulationCalibration::default(),
        None,
        &mut routing_load,
    );

    assert_eq!(selected.decode_node, 1);
    assert_eq!(selected.routing.estimated_kv_resource_wait_s, 0.0);
    assert!(selected.routing.reason.contains("KV resource wait"));
    assert_eq!(selected.routing.candidates.len(), 2);
    assert!(selected.routing.candidates.iter().any(|candidate| {
        candidate.selected
            && candidate.routable
            && candidate.decode_node == 1
            && candidate.estimated_kv_resource_wait_s == Some(0.0)
    }));
    assert!(selected.routing.candidates.iter().any(|candidate| {
        !candidate.selected
            && candidate.routable
            && candidate.decode_node == 2
            && candidate
                .estimated_kv_resource_wait_s
                .is_some_and(|wait_s| wait_s > 9_999.0)
            && candidate
                .kv_transfer_resources
                .iter()
                .all(|resource| resource.starts_with("kv_route:"))
    }));
}

#[test]
fn same_node_different_gpu_routes_pay_intra_node_kv_handoff() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![0],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: SearchSpace {
                tensor_ranks: vec![2],
                pipeline_ranks: vec![1],
                expert_ranks: vec![1],
                data_ranks: vec![1],
            },
            decode: SearchSpace {
                tensor_ranks: vec![1],
                pipeline_ranks: vec![1],
                expert_ranks: vec![1],
                data_ranks: vec![1],
            },
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    let observation = &results[0].request_observations[0];
    assert_eq!(observation.prefill_node, 0);
    assert_eq!(observation.decode_node, 0);
    assert_ne!(
        observation.prefill_route_gpus,
        observation.decode_route_gpus
    );
    assert!(observation.kv_transfer_bytes > 0);
    assert!(observation.kv_transfer_s > 0.0);
    assert!(observation.kv_transfer_paths.iter().any(|path| {
        path.source.node_id == 0
            && path.destination.node_id == 0
            && path
                .resources
                .iter()
                .any(|resource| resource.contains("intra-node fabric"))
            && path
                .resource_details
                .iter()
                .any(|resource| resource.kind == "intra_node_fabric")
    }));
}

#[test]
fn route_coverage_reports_partially_routable_disaggregated_pool() {
    let cluster = parse_cluster(
        r#"
        [cluster]
        preset = "custom"

        [interconnect]
        kind = "ethernet"
        variant = "10g"

        [[interconnect.links]]
        from = 0
        to = 2
        kind = "ib"
        variant = "ndr"

        [[node_groups]]
        label = "prefill"
        start_id = 0
        count = 1
        gpu = "h100_sxm"
        gpu_count = 8
        intra = "nvlink_v4"
        nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }

        [[node_groups]]
        label = "decode"
        start_id = 1
        count = 2
        gpu = "h100_sxm"
        gpu_count = 8
        intra = "nvlink_v4"
        nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }
        "#,
    )
    .unwrap();
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1, 2],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            routing_policy: ServingRoutingPolicy::TopologyAware,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    assert_eq!(results[0].route_coverage.candidate_count, 2);
    assert_eq!(results[0].route_coverage.routable_candidate_count, 1);
    assert_eq!(results[0].route_coverage.unroutable_candidate_count, 1);
    assert_eq!(results[0].route_coverage.fraction, 0.5);
    assert!(results[0].topology_bottlenecks.iter().any(|bottleneck| {
        bottleneck.code == "partial_route_coverage"
            && bottleneck.severity == "warning"
            && bottleneck.observed == Some(1.0)
            && bottleneck.limit == Some(2.0)
            && bottleneck.unit.as_deref() == Some("routes")
    }));
    assert!(
        results[0]
            .bottlenecks
            .contains(&"partial_route_coverage".to_string())
    );
    assert_eq!(results[0].request_observations[0].prefill_node, 0);
    assert_eq!(results[0].request_observations[0].decode_node, 2);
    assert_eq!(
        results[0].request_observations[0].routing_candidate_count,
        2
    );
    assert_eq!(
        results[0].request_observations[0].routing_routable_candidate_count,
        1
    );
}

#[test]
fn rejects_disaggregated_pool_without_kv_route() {
    let mut cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    cluster.inter_node_topology = InterNodeTopology::Custom(Default::default());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Independent,
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            ..ServingTraffic::default()
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].feasible);
    assert_eq!(results[0].route_coverage.candidate_count, 1);
    assert_eq!(results[0].route_coverage.routable_candidate_count, 0);
    assert_eq!(results[0].route_coverage.unroutable_candidate_count, 1);
    assert_eq!(results[0].route_coverage.fraction, 0.0);
    assert!(results[0].request_observations.is_empty());
    assert!(
        results[0]
            .rejected_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("KV transfer route unavailable"))
    );
    assert!(
        results[0]
            .bottlenecks
            .contains(&"kv_transfer_route".to_string())
    );
    assert!(results[0].rejections.iter().any(|rejection| {
        rejection.phase == "kv_transfer"
            && rejection.category == "topology"
            && rejection.resource == "kv_transfer_route"
            && rejection.code == "kv_transfer_route_unavailable"
            && rejection.observed == Some(0.0)
            && rejection.limit == Some(1.0)
            && rejection.unit.as_deref() == Some("routes")
            && rejection.message.contains("route_coverage=0/1")
            && rejection.message.contains("unroutable_candidates=1")
            && rejection
                .remediation
                .as_deref()
                .is_some_and(|hint| hint.contains("connected prefill/decode pools"))
    }));
}

#[test]
fn continuous_decode_batches_by_routed_replica_node() {
    let cluster = Cluster::h100_sxm_nodes(3, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![1, 2],
        decode_nodes: vec![1, 2],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(512),
                chunk_tokens: None,
            },
            decode_batching: ServingDecodeBatching::Continuous {
                max_batch_tokens: Some(8),
            },
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(4),
            max_resident_tokens_per_node: Some(4096),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,

            ttft_slo_s: None,

            tpot_slo_s: None,

            itl_slo_s: None,

            e2el_slo_s: None,

            max_ttft_slo_miss_rate: None,

            max_tpot_slo_miss_rate: None,

            max_itl_slo_miss_rate: None,

            max_e2el_slo_miss_rate: None,

            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,

            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![1],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    assert_eq!(
        results[0]
            .request_observations
            .iter()
            .map(|observation| (observation.prefill_node, observation.decode_node))
            .collect::<Vec<_>>(),
        vec![(1, 1), (2, 2)]
    );
    assert_eq!(
        results[0]
            .request_observations
            .iter()
            .map(|observation| {
                (
                    observation.prefill_route_nodes.clone(),
                    observation.decode_route_nodes.clone(),
                )
            })
            .collect::<Vec<_>>(),
        vec![(vec![1], vec![1]), (vec![2], vec![2])]
    );
    assert_eq!(results[0].metrics.kv_transfer_s, 0.0);
    let decode_iteration_count = results[0]
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("decode iteration"))
        .filter(|operation| operation.name.contains("layer 0 compute"))
        .count();
    assert_eq!(decode_iteration_count, 2);
    assert_eq!(results[0].decode_iterations.len(), 2);
    assert_eq!(
        results[0]
            .decode_iterations
            .iter()
            .map(|iteration| iteration.request_indices.clone())
            .collect::<Vec<_>>(),
        vec![vec![0], vec![1]]
    );
    assert!(
        results[0]
            .decode_iterations
            .iter()
            .all(|iteration| iteration.latency_s > 0.0 && iteration.batch_tokens == 1)
    );
    assert!(
        results[0]
            .resource_utilization
            .iter()
            .any(|resource| resource.resource == "gpu compute node 1")
    );
    assert!(
        results[0]
            .resource_utilization
            .iter()
            .any(|resource| resource.resource == "gpu compute node 2")
    );
}

#[test]
fn continuous_prefill_batches_ready_requests() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(2),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(512),
                chunk_tokens: None,
            },
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,

            ttft_slo_s: None,

            tpot_slo_s: None,

            itl_slo_s: None,

            e2el_slo_s: None,

            max_ttft_slo_miss_rate: None,

            max_tpot_slo_miss_rate: None,

            max_itl_slo_miss_rate: None,

            max_e2el_slo_miss_rate: None,

            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,

            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![1],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    let first_layer_prefill_batches = results[0]
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("prefill batch"))
        .filter(|operation| operation.name.contains("layer 0 compute"))
        .count();
    assert_eq!(first_layer_prefill_batches, 1);
    assert_eq!(results[0].request_observations[0].prefill_start_s, 0.0);
    assert_eq!(
        results[0].request_observations[0].prefill_start_s,
        results[0].request_observations[1].prefill_start_s
    );
}

#[test]
fn continuous_prefill_chunks_long_requests() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let search = SearchSpace {
        tensor_ranks: vec![1],
        pipeline_ranks: vec![1],
        expert_ranks: vec![1],
        data_ranks: vec![1],
    };
    let serving = DisaggregatedServingConfig {
        deployment_mode: ServingDeploymentMode::Flexible,
        prefill_nodes: vec![0],
        decode_nodes: vec![1],
        pool_candidates: Vec::new(),
        pool_search: None,
        objective: ServingObjective::MinimizeE2el,
        slo_miss_penalty_weight: 0.0,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights::default(),
        topology_risk_penalty_weight: 0.0,
        max_memory_pressure_fraction: None,
        max_unique_gpus: None,
        min_throughput_tokens_per_s: None,
        cost_model: ServingCostModel::default(),
        search: ServingSearchSpace {
            prefill: search.clone(),
            decode: search,
        },
        traffic: ServingTraffic {
            request_count: Some(1),
            arrival_gap_s: Some(0.0),
            arrival: ServingArrivalPattern::FixedGap,
            routing_policy: ServingRoutingPolicy::RoundRobin,
            prefill_batching: ServingPrefillBatching::Continuous {
                max_batch_tokens: Some(512),
                chunk_tokens: Some(64),
            },
            decode_batching: ServingDecodeBatching::Independent,
            decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
            services: ServingServicesConfig::default(),
            service_backpressure_penalty_weight: 0.0,
            max_prefill_tokens: None,
            max_prefill_tokens_per_node: None,
            max_prefill_tokens_per_gpu: None,
            max_prefill_worker_slots_per_gpu: None,
            max_decode_sequences: Some(8),
            max_resident_tokens: Some(8192),
            max_decode_sequences_per_node: Some(8),
            max_resident_tokens_per_node: Some(8192),
            max_decode_sequences_per_gpu: None,
            max_decode_worker_slots_per_gpu: None,
            max_resident_tokens_per_gpu: None,
            max_kv_transfer_worker_slots_per_gpu: None,
            kv_block_tokens: None,
            max_kv_blocks: None,
            max_kv_blocks_per_node: None,
            max_kv_blocks_per_gpu: None,
            ttft_slo_s: None,
            tpot_slo_s: None,
            itl_slo_s: None,
            e2el_slo_s: None,
            max_ttft_slo_miss_rate: None,
            max_tpot_slo_miss_rate: None,
            max_itl_slo_miss_rate: None,
            max_e2el_slo_miss_rate: None,
            max_deadline_miss_rate: None,
            metric_ceilings: ServingMetricCeilings::default(),
            kv_route_constraints: ServingKvRouteConstraints::default(),
            measurement_start_s: None,
            measurement_end_s: None,
            measurement_warmup_s: None,
            measurement_cooldown_s: None,
            measurement_steady_state: false,
            measurement_steady_state_min_requests: None,
            measurement_steady_state_max_cv: None,
            max_queue_delay_s: None,
            max_kv_queue_delay_s: None,
            max_decode_queue_delay_s: None,
            max_decode_iteration_queue_delay_s: None,
            request_timeout_s: None,
            shape_seed: 1,
            prefix_cache_hit_rate: None,
            batch_size_distribution: None,
            prompt_tokens_distribution: None,
            decode_tokens_distribution: None,
            shape_profiles: Vec::new(),
            batch_sizes: vec![1],
            prompt_tokens: vec![128],
            decode_tokens: vec![1],
            trace_requests: Vec::new(),
            traffic_classes: Vec::new(),
        },
        slo_policies: Vec::new(),
    };

    let results = ServingSolver::rank_disaggregated(
        &cluster,
        &model(),
        &request(),
        &serving,
        SimulationCalibration::default(),
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].feasible);
    assert_eq!(results[0].request_observations[0].prefill_chunks, 2);
    assert_eq!(results[0].metrics.prefill_chunks, 2);
    assert_eq!(results[0].metrics.peak_prefill_tokens, 64);
    assert_eq!(results[0].metrics.peak_prefill_tokens_per_node, 64);
    assert_eq!(results[0].metrics.peak_prefill_tokens_per_gpu, 64);
    let first_layer_prefill_chunks = results[0]
        .scheduled_operations
        .iter()
        .filter(|operation| operation.name.contains("prefill batch"))
        .filter(|operation| operation.name.contains("chunk tokens"))
        .filter(|operation| operation.name.contains("layer 0 compute"))
        .count();
    assert_eq!(first_layer_prefill_chunks, 2);
}

#[test]
fn poisson_arrivals_are_seeded_and_monotonic() {
    let traffic = ServingTraffic {
        request_count: Some(4),
        arrival_gap_s: Some(1.0),
        arrival: ServingArrivalPattern::Poisson {
            rate_per_s: 100.0,
            seed: 42,
        },
        routing_policy: ServingRoutingPolicy::RoundRobin,
        prefill_batching: ServingPrefillBatching::Independent,
        decode_batching: ServingDecodeBatching::Independent,
        decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
        services: ServingServicesConfig::default(),
        service_backpressure_penalty_weight: 0.0,
        max_prefill_tokens: None,
        max_prefill_tokens_per_node: None,
        max_prefill_tokens_per_gpu: None,
        max_prefill_worker_slots_per_gpu: None,
        max_decode_sequences: None,
        max_resident_tokens: None,
        max_decode_sequences_per_node: None,
        max_resident_tokens_per_node: None,
        max_decode_sequences_per_gpu: None,
        max_decode_worker_slots_per_gpu: None,
        max_resident_tokens_per_gpu: None,
        max_kv_transfer_worker_slots_per_gpu: None,
        kv_block_tokens: None,
        max_kv_blocks: None,
        max_kv_blocks_per_node: None,
        max_kv_blocks_per_gpu: None,

        ttft_slo_s: None,

        tpot_slo_s: None,

        itl_slo_s: None,

        e2el_slo_s: None,

        max_ttft_slo_miss_rate: None,

        max_tpot_slo_miss_rate: None,

        max_itl_slo_miss_rate: None,

        max_e2el_slo_miss_rate: None,

        max_deadline_miss_rate: None,
        metric_ceilings: ServingMetricCeilings::default(),
        kv_route_constraints: ServingKvRouteConstraints::default(),
        measurement_start_s: None,
        measurement_end_s: None,
        measurement_warmup_s: None,
        measurement_cooldown_s: None,
        measurement_steady_state: false,
        measurement_steady_state_min_requests: None,
        measurement_steady_state_max_cv: None,
        max_queue_delay_s: None,
        max_kv_queue_delay_s: None,
        max_decode_queue_delay_s: None,
        max_decode_iteration_queue_delay_s: None,
        request_timeout_s: None,

        shape_seed: 1,
        prefix_cache_hit_rate: None,
        batch_size_distribution: None,
        prompt_tokens_distribution: None,
        decode_tokens_distribution: None,
        shape_profiles: Vec::new(),
        batch_sizes: Vec::new(),
        prompt_tokens: Vec::new(),
        decode_tokens: Vec::new(),
        trace_requests: Vec::new(),
        traffic_classes: Vec::new(),
    };

    let first = traffic.arrival_times(4, SimulationCalibration::default());
    let second = traffic.arrival_times(4, SimulationCalibration::default());

    assert_eq!(first, second);
    assert_eq!(first[0], 0.0);
    assert!(first.windows(2).all(|window| window[1] > window[0]));
}

#[test]
fn bursty_arrivals_group_requests_deterministically() {
    let traffic = ServingTraffic {
        request_count: Some(5),
        arrival: ServingArrivalPattern::Bursty {
            burst_size: 2,
            burst_interval_s: 0.010,
            intra_burst_gap_s: 0.001,
        },
        ..ServingTraffic::default()
    };

    let arrivals = traffic.arrival_times(5, SimulationCalibration::default());

    assert_eq!(arrivals, vec![0.0, 0.001, 0.010, 0.011, 0.020]);
}

#[test]
fn diurnal_arrivals_are_seeded_and_monotonic() {
    let traffic = ServingTraffic {
        request_count: Some(8),
        arrival: ServingArrivalPattern::Diurnal {
            min_rate_per_s: 10.0,
            max_rate_per_s: 100.0,
            period_s: 1.0,
            phase_s: 0.25,
            seed: 42,
        },
        ..ServingTraffic::default()
    };

    let first = traffic.arrival_times(8, SimulationCalibration::default());
    let second = traffic.arrival_times(8, SimulationCalibration::default());

    assert_eq!(first, second);
    assert_eq!(first[0], 0.0);
    assert!(first.windows(2).all(|window| window[1] > window[0]));
    assert!(first[7] < 1.0);
}

#[test]
fn diurnal_rate_changes_with_phase() {
    let peak = diurnal_rate_at_s(0.25, 10.0, 100.0, 1.0, 0.0);
    let trough = diurnal_rate_at_s(0.75, 10.0, 100.0, 1.0, 0.0);

    assert!(peak > trough);
    assert!((peak - 100.0).abs() < 1e-9);
    assert!((trough - 10.0).abs() < 1e-9);
}

#[test]
fn self_similar_arrivals_are_seeded_and_monotonic() {
    let traffic = ServingTraffic {
        request_count: Some(8),
        arrival: ServingArrivalPattern::SelfSimilar {
            rate_per_s: 100.0,
            pareto_shape: 1.4,
            max_gap_s: Some(0.2),
            seed: 42,
        },
        ..ServingTraffic::default()
    };

    let first = traffic.arrival_times(8, SimulationCalibration::default());
    let second = traffic.arrival_times(8, SimulationCalibration::default());

    assert_eq!(first, second);
    assert_eq!(first[0], 0.0);
    assert!(first.windows(2).all(|window| window[1] > window[0]));
    let gaps = first
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect::<Vec<_>>();
    let min_gap = gaps.iter().copied().fold(f64::INFINITY, f64::min);
    let max_gap = gaps.iter().copied().fold(0.0, f64::max);
    assert!(max_gap > min_gap);
}

#[test]
fn self_similar_arrivals_honor_max_gap_cap() {
    let arrivals = self_similar_arrival_times(16, 1.0, 1.2, Some(0.05), 7);

    assert_eq!(arrivals[0], 0.0);
    assert!(arrivals.windows(2).all(|window| {
        let gap_s = window[1] - window[0];
        gap_s > 0.0 && gap_s <= 0.050_000_000_001
    }));
}

#[test]
fn request_shape_distributions_are_seeded() {
    let traffic = ServingTraffic {
        request_count: Some(4),
        arrival_gap_s: Some(0.0),
        arrival: ServingArrivalPattern::FixedGap,
        routing_policy: ServingRoutingPolicy::RoundRobin,
        prefill_batching: ServingPrefillBatching::Independent,
        decode_batching: ServingDecodeBatching::Independent,
        decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
        services: ServingServicesConfig::default(),
        service_backpressure_penalty_weight: 0.0,
        max_prefill_tokens: None,
        max_prefill_tokens_per_node: None,
        max_prefill_tokens_per_gpu: None,
        max_prefill_worker_slots_per_gpu: None,
        max_decode_sequences: None,
        max_resident_tokens: None,
        max_decode_sequences_per_node: None,
        max_resident_tokens_per_node: None,
        max_decode_sequences_per_gpu: None,
        max_decode_worker_slots_per_gpu: None,
        max_resident_tokens_per_gpu: None,
        max_kv_transfer_worker_slots_per_gpu: None,
        kv_block_tokens: None,
        max_kv_blocks: None,
        max_kv_blocks_per_node: None,
        max_kv_blocks_per_gpu: None,

        ttft_slo_s: None,

        tpot_slo_s: None,

        itl_slo_s: None,

        e2el_slo_s: None,

        max_ttft_slo_miss_rate: None,

        max_tpot_slo_miss_rate: None,

        max_itl_slo_miss_rate: None,

        max_e2el_slo_miss_rate: None,

        max_deadline_miss_rate: None,
        metric_ceilings: ServingMetricCeilings::default(),
        kv_route_constraints: ServingKvRouteConstraints::default(),
        measurement_start_s: None,
        measurement_end_s: None,
        measurement_warmup_s: None,
        measurement_cooldown_s: None,
        measurement_steady_state: false,
        measurement_steady_state_min_requests: None,
        measurement_steady_state_max_cv: None,
        max_queue_delay_s: None,
        max_kv_queue_delay_s: None,
        max_decode_queue_delay_s: None,
        max_decode_iteration_queue_delay_s: None,
        request_timeout_s: None,

        shape_seed: 99,
        prefix_cache_hit_rate: None,
        batch_size_distribution: Some(ServingValueDistribution::Weighted {
            values: vec![1, 4],
            weights: vec![0.25, 0.75],
        }),
        prompt_tokens_distribution: Some(ServingValueDistribution::LogNormal {
            median: 512.0,
            sigma: 0.6,
            min: 128,
            max: 2048,
        }),
        decode_tokens_distribution: Some(ServingValueDistribution::Uniform { min: 8, max: 32 }),
        shape_profiles: Vec::new(),
        batch_sizes: vec![1],
        prompt_tokens: vec![128],
        decode_tokens: vec![8],
        trace_requests: Vec::new(),
        traffic_classes: Vec::new(),
    };

    let base = request();
    let first = (0..4)
        .map(|idx| traffic.request_at(&base, idx))
        .collect::<Vec<_>>();
    let second = (0..4)
        .map(|idx| traffic.request_at(&base, idx))
        .collect::<Vec<_>>();

    assert_eq!(first, second);
    assert!(
        first
            .iter()
            .all(|request| request.batch_size == 1 || request.batch_size == 4)
    );
    assert!(
        first
            .iter()
            .all(|request| (128..=2048).contains(&request.prompt_tokens))
    );
    assert!(
        first
            .iter()
            .all(|request| (8..=32).contains(&request.decode_tokens))
    );
}

#[test]
fn shape_profiles_keep_synthetic_prompt_decode_shapes_correlated() {
    let traffic = ServingTraffic {
        shape_seed: 7,
        shape_profiles: vec![
            ServingShapeProfile {
                name: "tenant-a-small".to_string(),
                weight: 1.0,
                tenant: Some("tenant-a".to_string()),
                model_id: Some("model-a".to_string()),
                cache_key: Some("shared-prefix-a".to_string()),
                priority: Some(7),
                batch_size: 1,
                prompt_tokens: 128,
                decode_tokens: 16,
                max_sequence_tokens: Some(256),
                prefix_cache_hit_tokens: Some(64),
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo {
                    ttft_s: Some(0.100),
                    tpot_s: Some(0.020),
                    itl_s: None,
                    e2el_s: Some(0.500),
                },
                request_timeout_s: Some(0.750),
                deadline_after_s: Some(0.050),
                cancellation_after_s: None,
            },
            ServingShapeProfile {
                name: "tenant-b-large".to_string(),
                weight: 1.0,
                tenant: Some("tenant-b".to_string()),
                model_id: Some("model-b".to_string()),
                cache_key: Some("shared-prefix-b".to_string()),
                priority: Some(-1),
                batch_size: 4,
                prompt_tokens: 4096,
                decode_tokens: 256,
                max_sequence_tokens: Some(8192),
                prefix_cache_hit_tokens: None,
                prefix_cache_hit_rate: Some(0.25),
                slo: ServingRequestSlo {
                    ttft_s: Some(0.400),
                    tpot_s: Some(0.060),
                    itl_s: Some(0.060),
                    e2el_s: Some(2.000),
                },
                request_timeout_s: Some(3.000),
                deadline_after_s: None,
                cancellation_after_s: Some(0.010),
            },
        ],
        batch_size_distribution: Some(ServingValueDistribution::Uniform { min: 8, max: 16 }),
        prompt_tokens_distribution: Some(ServingValueDistribution::Uniform { min: 8, max: 16 }),
        decode_tokens_distribution: Some(ServingValueDistribution::Uniform { min: 8, max: 16 }),
        ..ServingTraffic::default()
    };

    let base = request();
    let first = (0..8)
        .map(|idx| traffic.request_at(&base, idx))
        .collect::<Vec<_>>();
    let second = (0..8)
        .map(|idx| traffic.request_at(&base, idx))
        .collect::<Vec<_>>();

    assert_eq!(first, second);
    assert!(first.iter().any(|request| request.prompt_tokens == 128));
    assert!(first.iter().any(|request| request.prompt_tokens == 4096));
    for (idx, request) in first.iter().enumerate() {
        match (
            request.batch_size,
            request.prompt_tokens,
            request.decode_tokens,
            request.max_sequence_tokens,
        ) {
            (1, 128, 16, 256) => {
                assert_eq!(
                    traffic.shape_profile_name(idx as u32).as_deref(),
                    Some("tenant-a-small")
                );
                assert_eq!(traffic.tenant(idx as u32).as_deref(), Some("tenant-a"));
                assert_eq!(traffic.model_id(idx as u32).as_deref(), Some("model-a"));
                assert_eq!(
                    traffic.cache_key(idx as u32).as_deref(),
                    Some("shared-prefix-a")
                );
                assert_eq!(traffic.priority(idx as u32), 7);
                assert_eq!(traffic.prefix_cache_hit_tokens(idx as u32, 128), 64);
                assert_eq!(
                    traffic.effective_slo(idx as u32),
                    ServingRequestSlo {
                        ttft_s: Some(0.100),
                        tpot_s: Some(0.020),
                        itl_s: None,
                        e2el_s: Some(0.500),
                    }
                );
                assert_eq!(traffic.effective_request_timeout_s(idx as u32), Some(0.750));
                assert_eq!(traffic.deadline_s(idx as u32, 1.0), Some(1.050));
                assert_eq!(traffic.cancellation_s(idx as u32, 1.0), None);
            }
            (4, 4096, 256, 8192) => {
                assert_eq!(
                    traffic.shape_profile_name(idx as u32).as_deref(),
                    Some("tenant-b-large")
                );
                assert_eq!(traffic.tenant(idx as u32).as_deref(), Some("tenant-b"));
                assert_eq!(traffic.model_id(idx as u32).as_deref(), Some("model-b"));
                assert_eq!(
                    traffic.cache_key(idx as u32).as_deref(),
                    Some("shared-prefix-b")
                );
                assert_eq!(traffic.priority(idx as u32), -1);
                assert_eq!(traffic.prefix_cache_hit_tokens(idx as u32, 4096), 1024);
                assert_eq!(
                    traffic.effective_slo(idx as u32),
                    ServingRequestSlo {
                        ttft_s: Some(0.400),
                        tpot_s: Some(0.060),
                        itl_s: Some(0.060),
                        e2el_s: Some(2.000),
                    }
                );
                assert_eq!(traffic.effective_request_timeout_s(idx as u32), Some(3.000));
                assert_eq!(traffic.deadline_s(idx as u32, 2.0), None);
                assert_eq!(traffic.cancellation_s(idx as u32, 2.0), Some(2.010));
            }
            other => panic!("unexpected shape profile request: {other:?}"),
        }
    }
}

#[test]
fn trace_requests_override_generated_arrivals_and_shapes() {
    let traffic = ServingTraffic {
        request_count: Some(2),
        arrival_gap_s: Some(10.0),
        arrival: ServingArrivalPattern::Poisson {
            rate_per_s: 100.0,
            seed: 42,
        },
        routing_policy: ServingRoutingPolicy::RoundRobin,
        prefill_batching: ServingPrefillBatching::Independent,
        decode_batching: ServingDecodeBatching::Independent,
        decode_capacity_policy: ServingDecodeCapacityPolicy::CandidateReject,
        services: ServingServicesConfig::default(),
        service_backpressure_penalty_weight: 0.0,
        max_prefill_tokens: None,
        max_prefill_tokens_per_node: None,
        max_prefill_tokens_per_gpu: None,
        max_prefill_worker_slots_per_gpu: None,
        max_decode_sequences: None,
        max_resident_tokens: None,
        max_decode_sequences_per_node: None,
        max_resident_tokens_per_node: None,
        max_decode_sequences_per_gpu: None,
        max_decode_worker_slots_per_gpu: None,
        max_resident_tokens_per_gpu: None,
        max_kv_transfer_worker_slots_per_gpu: None,
        kv_block_tokens: None,
        max_kv_blocks: None,
        max_kv_blocks_per_node: None,
        max_kv_blocks_per_gpu: None,
        ttft_slo_s: None,
        tpot_slo_s: None,
        itl_slo_s: None,
        e2el_slo_s: None,
        max_ttft_slo_miss_rate: None,
        max_tpot_slo_miss_rate: None,
        max_itl_slo_miss_rate: None,
        max_e2el_slo_miss_rate: None,
        max_deadline_miss_rate: None,
        metric_ceilings: ServingMetricCeilings::default(),
        kv_route_constraints: ServingKvRouteConstraints::default(),
        measurement_start_s: None,
        measurement_end_s: None,
        measurement_warmup_s: None,
        measurement_cooldown_s: None,
        measurement_steady_state: false,
        measurement_steady_state_min_requests: None,
        measurement_steady_state_max_cv: None,
        max_queue_delay_s: None,
        max_kv_queue_delay_s: None,
        max_decode_queue_delay_s: None,
        max_decode_iteration_queue_delay_s: None,
        request_timeout_s: None,
        shape_seed: 99,
        prefix_cache_hit_rate: None,
        batch_size_distribution: Some(ServingValueDistribution::Uniform { min: 4, max: 8 }),
        prompt_tokens_distribution: None,
        decode_tokens_distribution: None,
        shape_profiles: Vec::new(),
        batch_sizes: vec![4],
        prompt_tokens: vec![1024],
        decode_tokens: vec![64],
        trace_requests: vec![
            ServingTraceRequest {
                request_id: Some("trace-0".to_string()),
                tenant: Some("tenant-a".to_string()),
                model_id: Some("model-a".to_string()),
                cache_key: Some("shared-prefix-a".to_string()),
                arrival_s: 0.0,
                priority: 5,
                batch_size: 1,
                prompt_tokens: 256,
                decode_tokens: 8,
                max_sequence_tokens: Some(1024),
                prefix_cache_hit_tokens: Some(128),
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo::default(),
                deadline_s: Some(10.0),
                cancellation_s: None,
            },
            ServingTraceRequest {
                request_id: Some("trace-1".to_string()),
                tenant: Some("tenant-b".to_string()),
                model_id: Some("model-a".to_string()),
                cache_key: None,
                arrival_s: 0.0025,
                priority: -1,
                batch_size: 2,
                prompt_tokens: 512,
                decode_tokens: 16,
                max_sequence_tokens: None,
                prefix_cache_hit_tokens: None,
                prefix_cache_hit_rate: Some(0.5),
                slo: ServingRequestSlo::default(),
                deadline_s: None,
                cancellation_s: None,
            },
        ],
        traffic_classes: Vec::new(),
    };
    let base = request();

    assert_eq!(traffic.request_count(SimulationCalibration::default()), 2);
    assert_eq!(
        traffic.arrival_times(2, SimulationCalibration::default()),
        vec![0.0, 0.0025]
    );
    let first = traffic.request_at(&base, 0);
    let second = traffic.request_at(&base, 1);

    assert_eq!(first.batch_size, 1);
    assert_eq!(first.prompt_tokens, 256);
    assert_eq!(first.decode_tokens, 8);
    assert_eq!(first.max_sequence_tokens, 1024);
    assert_eq!(traffic.request_id(0).as_deref(), Some("trace-0"));
    assert_eq!(traffic.tenant(0).as_deref(), Some("tenant-a"));
    assert_eq!(traffic.model_id(0).as_deref(), Some("model-a"));
    assert_eq!(traffic.priority(0), 5);
    assert_eq!(traffic.deadline_s(0, 0.0), Some(10.0));
    assert_eq!(second.batch_size, 2);
    assert_eq!(second.prompt_tokens, 512);
    assert_eq!(second.decode_tokens, 16);
    assert_eq!(second.max_sequence_tokens, 528);
}

#[test]
fn trace_derived_arrivals_keep_synthetic_shapes_and_metadata() {
    let traffic = ServingTraffic {
        request_count: Some(2),
        arrival: ServingArrivalPattern::TraceDerived,
        shape_profiles: vec![ServingShapeProfile {
            name: "trace-derived-synthetic".to_string(),
            weight: 1.0,
            tenant: Some("synthetic-tenant".to_string()),
            model_id: Some("synthetic-model".to_string()),
            cache_key: Some("synthetic-prefix".to_string()),
            priority: Some(3),
            batch_size: 4,
            prompt_tokens: 1024,
            decode_tokens: 64,
            max_sequence_tokens: Some(2048),
            prefix_cache_hit_tokens: Some(128),
            prefix_cache_hit_rate: None,
            slo: ServingRequestSlo {
                ttft_s: Some(0.100),
                tpot_s: None,
                itl_s: None,
                e2el_s: Some(1.000),
            },
            request_timeout_s: Some(2.0),
            deadline_after_s: Some(0.250),
            cancellation_after_s: Some(0.500),
        }],
        trace_requests: vec![
            ServingTraceRequest {
                request_id: Some("trace-0".to_string()),
                tenant: Some("trace-tenant".to_string()),
                model_id: Some("trace-model".to_string()),
                cache_key: Some("trace-prefix".to_string()),
                arrival_s: 0.0,
                priority: 9,
                batch_size: 1,
                prompt_tokens: 256,
                decode_tokens: 8,
                max_sequence_tokens: Some(1024),
                prefix_cache_hit_tokens: Some(16),
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo {
                    ttft_s: Some(0.001),
                    tpot_s: None,
                    itl_s: None,
                    e2el_s: Some(0.002),
                },
                deadline_s: Some(10.0),
                cancellation_s: Some(11.0),
            },
            ServingTraceRequest {
                request_id: Some("trace-1".to_string()),
                tenant: Some("trace-tenant".to_string()),
                model_id: Some("trace-model".to_string()),
                cache_key: Some("trace-prefix".to_string()),
                arrival_s: 0.0025,
                priority: 9,
                batch_size: 2,
                prompt_tokens: 512,
                decode_tokens: 16,
                max_sequence_tokens: Some(2048),
                prefix_cache_hit_tokens: Some(32),
                prefix_cache_hit_rate: None,
                slo: ServingRequestSlo::default(),
                deadline_s: None,
                cancellation_s: None,
            },
        ],
        ..ServingTraffic::default()
    };
    let base = request();

    assert_eq!(traffic.request_count(SimulationCalibration::default()), 2);
    assert_eq!(
        traffic.arrival_times(2, SimulationCalibration::default()),
        vec![0.0, 0.0025]
    );

    let first = traffic.request_at(&base, 0);
    assert_eq!(first.batch_size, 4);
    assert_eq!(first.prompt_tokens, 1024);
    assert_eq!(first.decode_tokens, 64);
    assert_eq!(first.max_sequence_tokens, 2048);
    assert_eq!(traffic.request_id(0), None);
    assert_eq!(
        traffic.shape_profile_name(0).as_deref(),
        Some("trace-derived-synthetic")
    );
    assert_eq!(traffic.tenant(0).as_deref(), Some("synthetic-tenant"));
    assert_eq!(traffic.model_id(0).as_deref(), Some("synthetic-model"));
    assert_eq!(traffic.cache_key(0).as_deref(), Some("synthetic-prefix"));
    assert_eq!(traffic.priority(0), 3);
    assert_eq!(traffic.prefix_cache_hit_tokens(0, first.prompt_tokens), 128);
    assert_eq!(traffic.effective_slo(0).ttft_s, Some(0.100));
    assert_eq!(traffic.effective_slo(0).e2el_s, Some(1.000));
    assert_eq!(traffic.effective_request_timeout_s(0), Some(2.0));
    assert_eq!(traffic.deadline_s(0, 0.0), Some(0.250));
    assert_eq!(traffic.cancellation_s(0, 0.0), Some(0.500));
}
