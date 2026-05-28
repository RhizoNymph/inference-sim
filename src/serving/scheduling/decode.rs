use super::*;

pub(in crate::serving) fn apply_decode_capacity_admission(
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
pub(in crate::serving) struct DecodeAdmissionResidency {
    pub(in crate::serving) finish_s: f64,
    pub(in crate::serving) traffic_class: Option<String>,
    pub(in crate::serving) sequences: u32,
    pub(in crate::serving) resident_tokens: u64,
    pub(in crate::serving) kv_blocks: u64,
    pub(in crate::serving) node_sequences: Vec<(NodeId, u32)>,
    pub(in crate::serving) node_tokens: Vec<(NodeId, u64)>,
    pub(in crate::serving) node_blocks: Vec<(NodeId, u64)>,
    pub(in crate::serving) gpu_sequences: Vec<(GpuAddr, u32)>,
    pub(in crate::serving) gpu_tokens: Vec<(GpuAddr, u64)>,
    pub(in crate::serving) gpu_blocks: Vec<(GpuAddr, u64)>,
}

impl DecodeAdmissionResidency {
    pub(in crate::serving) fn for_state(state: &DecodeRequestState, finish_s: f64) -> Self {
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

pub(in crate::serving) fn retire_decode_residency(
    active: &mut Vec<DecodeAdmissionResidency>,
    now_s: f64,
) {
    active.retain(|residency| residency.finish_s > now_s + 1e-12);
}

pub(in crate::serving) fn decode_service_estimate_s(
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

pub(in crate::serving) fn decode_capacity_admission_request_rejection(
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

pub(in crate::serving) fn decode_capacity_admission_rejection(
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

pub(in crate::serving) fn serving_traffic_class<'a>(
    traffic: &'a ServingTraffic,
    class_name: Option<&str>,
) -> Option<&'a ServingTrafficClass> {
    let class_name = class_name?;
    traffic
        .traffic_classes
        .iter()
        .find(|class| class.name == class_name)
}

pub(in crate::serving) fn active_class_sequences(
    active: &[DecodeAdmissionResidency],
    class_name: &str,
) -> u32 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.sequences)
        .sum()
}

pub(in crate::serving) fn active_class_tokens(
    active: &[DecodeAdmissionResidency],
    class_name: &str,
) -> u64 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.resident_tokens)
        .sum()
}

pub(in crate::serving) fn active_class_blocks(
    active: &[DecodeAdmissionResidency],
    class_name: &str,
) -> u64 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.kv_blocks)
        .sum()
}

pub(in crate::serving) fn active_node_sequences(
    active: &[DecodeAdmissionResidency],
    node_id: NodeId,
) -> u32 {
    active
        .iter()
        .flat_map(|residency| &residency.node_sequences)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, sequences)| *sequences)
        .sum()
}

pub(in crate::serving) fn active_node_tokens(
    active: &[DecodeAdmissionResidency],
    node_id: NodeId,
) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.node_tokens)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, tokens)| *tokens)
        .sum()
}

pub(in crate::serving) fn active_node_blocks(
    active: &[DecodeAdmissionResidency],
    node_id: NodeId,
) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.node_blocks)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, blocks)| *blocks)
        .sum()
}

pub(in crate::serving) fn active_gpu_sequences(
    active: &[DecodeAdmissionResidency],
    gpu: GpuAddr,
) -> u32 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_sequences)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, sequences)| *sequences)
        .sum()
}

pub(in crate::serving) fn active_gpu_tokens(
    active: &[DecodeAdmissionResidency],
    gpu: GpuAddr,
) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_tokens)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, tokens)| *tokens)
        .sum()
}

pub(in crate::serving) fn active_gpu_blocks(
    active: &[DecodeAdmissionResidency],
    gpu: GpuAddr,
) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_blocks)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, blocks)| *blocks)
        .sum()
}

pub(in crate::serving) fn reject_decode_capacity_admission(
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

pub(in crate::serving) fn decode_queue_delay_rejection(
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

pub(in crate::serving) fn decode_iteration_queue_timeout(
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

pub(in crate::serving) fn reject_decode_queue_admission(
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

pub(in crate::serving) fn timeout_decode_iteration_queue(
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
pub(in crate::serving) fn schedule_independent_decodes(
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
pub(in crate::serving) fn schedule_continuous_decodes(
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

pub(in crate::serving) fn mark_decode_ready_cancellations(
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

pub(in crate::serving) fn selected_decode_nodes(
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

pub(in crate::serving) fn record_decode_finish(
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

pub(in crate::serving) fn apply_terminal_statuses(
    states: &mut [DecodeRequestState],
    traffic: &ServingTraffic,
) {
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

pub(in crate::serving) fn inter_token_latencies(token_finish_s: &[f64]) -> Vec<f64> {
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
