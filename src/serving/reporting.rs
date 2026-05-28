use super::*;

struct WorkerObservationAccumulator {
    request_count: u32,
    completed_requests: u32,
    failed_requests: u32,
    input_tokens: u64,
    output_tokens: u64,
    queue_s: Vec<f64>,
    worker_queue_s: Vec<f64>,
    resource_queue_s: Vec<f64>,
    service_s: Vec<f64>,
    worker_slot_intervals: Vec<(f64, f64)>,
    first_start_s: f64,
    last_finish_s: f64,
    peak_prefill_tokens: u64,
    peak_decode_sequences: u32,
    peak_resident_tokens: u64,
    peak_kv_blocks: u64,
    peak_allocated_kv_tokens: u64,
    peak_kv_fragmentation_tokens: u64,
    peak_kv_block_table_bytes: u64,
    kv_cache_owner_slots: Vec<ServingWorkerKvSlotObservation>,
    decode_sequence_utilization: f64,
    resident_token_utilization: f64,
    kv_block_utilization: f64,
}

impl Default for WorkerObservationAccumulator {
    fn default() -> Self {
        Self {
            request_count: 0,
            completed_requests: 0,
            failed_requests: 0,
            input_tokens: 0,
            output_tokens: 0,
            queue_s: Vec::new(),
            worker_queue_s: Vec::new(),
            resource_queue_s: Vec::new(),
            service_s: Vec::new(),
            worker_slot_intervals: Vec::new(),
            first_start_s: f64::INFINITY,
            last_finish_s: 0.0,
            peak_prefill_tokens: 0,
            peak_decode_sequences: 0,
            peak_resident_tokens: 0,
            peak_kv_blocks: 0,
            peak_allocated_kv_tokens: 0,
            peak_kv_fragmentation_tokens: 0,
            peak_kv_block_table_bytes: 0,
            kv_cache_owner_slots: Vec::new(),
            decode_sequence_utilization: 0.0,
            resident_token_utilization: 0.0,
            kv_block_utilization: 0.0,
        }
    }
}

impl WorkerObservationAccumulator {
    fn record_common(&mut self, observation: &ServingRequestObservation) {
        self.request_count = self.request_count.saturating_add(1);
        if observation.status.is_completed() {
            self.completed_requests = self.completed_requests.saturating_add(1);
        } else {
            self.failed_requests = self.failed_requests.saturating_add(1);
        }
    }

    fn record_prefill(&mut self, observation: &ServingRequestObservation) {
        self.record_common(observation);
        self.input_tokens = self.input_tokens.saturating_add(
            u64::from(observation.batch_size) * u64::from(observation.effective_prefill_tokens),
        );
        if observation.prefill_start_s.is_finite() {
            push_finite(
                &mut self.queue_s,
                observation.prefill_worker_queue_s + observation.prefill_resource_queue_s,
            );
            push_finite(&mut self.worker_queue_s, observation.prefill_worker_queue_s);
            push_finite(
                &mut self.resource_queue_s,
                observation.prefill_resource_queue_s,
            );
            push_finite(&mut self.service_s, observation.prefill_s);
            self.first_start_s = self.first_start_s.min(observation.prefill_start_s);
        }
        if observation.prefill_finish_s.is_finite() {
            self.last_finish_s = self.last_finish_s.max(observation.prefill_finish_s);
        }
        let mut recorded_chunk_interval = false;
        for (start_s, finish_s) in observation
            .prefill_token_start_s
            .iter()
            .zip(observation.prefill_token_finish_s.iter())
        {
            if push_interval(&mut self.worker_slot_intervals, *start_s, *finish_s) {
                recorded_chunk_interval = true;
            }
        }
        if !recorded_chunk_interval && observation.status.is_admitted() {
            push_interval(
                &mut self.worker_slot_intervals,
                observation.prefill_start_s,
                observation.prefill_finish_s,
            );
        }
    }

    fn record_decode(&mut self, observation: &ServingRequestObservation) {
        self.record_common(observation);
        self.output_tokens = self.output_tokens.saturating_add(
            u64::from(observation.batch_size)
                * observation
                    .decode_token_finish_s
                    .len()
                    .min(u64::MAX as usize) as u64,
        );
        push_finite(
            &mut self.queue_s,
            observation.decode_worker_queue_s + observation.decode_resource_queue_s,
        );
        push_finite(&mut self.worker_queue_s, observation.decode_worker_queue_s);
        push_finite(
            &mut self.resource_queue_s,
            observation.decode_resource_queue_s,
        );
        push_finite(&mut self.service_s, observation.decode_s);
        if observation.first_decode_start_s.is_finite() {
            self.first_start_s = self.first_start_s.min(observation.first_decode_start_s);
        }
        if observation.last_decode_finish_s.is_finite() {
            self.last_finish_s = self.last_finish_s.max(observation.last_decode_finish_s);
        }
        let mut recorded_token_interval = false;
        for (start_s, finish_s) in observation
            .decode_token_start_s
            .iter()
            .zip(observation.decode_token_finish_s.iter())
        {
            if push_interval(&mut self.worker_slot_intervals, *start_s, *finish_s) {
                recorded_token_interval = true;
            }
        }
        if !recorded_token_interval {
            push_interval(
                &mut self.worker_slot_intervals,
                observation.first_decode_start_s,
                observation.last_decode_finish_s,
            );
        }
    }

    fn record_kv_transfer(&mut self, observation: &ServingRequestObservation) {
        self.record_common(observation);
        if observation.kv_start_s.is_finite() {
            push_finite(&mut self.queue_s, observation.kv_queue_s);
            push_finite(&mut self.worker_queue_s, observation.kv_worker_queue_s);
            push_finite(&mut self.resource_queue_s, observation.kv_resource_queue_s);
            self.first_start_s = self.first_start_s.min(observation.kv_start_s);
        }
        if observation.kv_finish_s.is_finite() {
            push_finite(&mut self.service_s, observation.kv_transfer_s);
            self.last_finish_s = self.last_finish_s.max(observation.kv_finish_s);
        }
        push_interval(
            &mut self.worker_slot_intervals,
            observation.kv_start_s,
            observation.kv_finish_s,
        );
    }

    fn apply_decode_capacity(&mut self, capacity: &ServingGpuCapacityObservation) {
        self.peak_decode_sequences = capacity.peak_decode_sequences;
        self.peak_resident_tokens = capacity.peak_resident_tokens;
        self.peak_kv_blocks = capacity.peak_kv_blocks;
        self.peak_allocated_kv_tokens = capacity.peak_allocated_kv_tokens;
        self.peak_kv_fragmentation_tokens = capacity.peak_kv_fragmentation_tokens;
        self.peak_kv_block_table_bytes = capacity.peak_kv_block_table_bytes;
        self.decode_sequence_utilization = capacity.decode_sequence_utilization;
        self.resident_token_utilization = capacity.resident_token_utilization;
        self.kv_block_utilization = capacity.kv_block_utilization;
    }

    fn apply_decode_worker_slot_capacity(&mut self, slots: Vec<ServingWorkerKvSlotObservation>) {
        self.kv_cache_owner_slots = slots;
    }

    fn apply_prefill_capacity(&mut self, capacity: &ServingGpuCapacityObservation) {
        self.peak_prefill_tokens = capacity.peak_prefill_tokens;
    }

    fn into_observation(
        self,
        phase: String,
        gpu: GpuAddr,
        configured_worker_slots: u32,
        deduplicate_intervals: bool,
    ) -> ServingWorkerObservation {
        let worker_slot_intervals = if deduplicate_intervals {
            deduplicate_intervals_by_span(&self.worker_slot_intervals)
        } else {
            self.worker_slot_intervals.clone()
        };
        let peak_active_worker_slots = peak_active_intervals(&worker_slot_intervals);
        let worker_slot_utilization =
            interval_utilization(&worker_slot_intervals, configured_worker_slots);
        ServingWorkerObservation {
            phase,
            node_id: gpu.node_id,
            local_gpu_id: gpu.local_gpu_id,
            configured_worker_slots,
            peak_active_worker_slots,
            worker_slot_utilization,
            request_count: self.request_count,
            completed_requests: self.completed_requests,
            failed_requests: self.failed_requests,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            queue_s: mean(&self.queue_s),
            queue_p95_s: percentile(self.queue_s.clone(), 0.95),
            queue_max_s: max_value(&self.queue_s),
            worker_queue_s: mean(&self.worker_queue_s),
            worker_queue_p95_s: percentile(self.worker_queue_s.clone(), 0.95),
            worker_queue_max_s: max_value(&self.worker_queue_s),
            resource_queue_s: mean(&self.resource_queue_s),
            resource_queue_p95_s: percentile(self.resource_queue_s.clone(), 0.95),
            resource_queue_max_s: max_value(&self.resource_queue_s),
            service_s: mean(&self.service_s),
            first_start_s: if self.first_start_s.is_finite() {
                self.first_start_s
            } else {
                0.0
            },
            last_finish_s: self.last_finish_s,
            peak_prefill_tokens: self.peak_prefill_tokens,
            peak_decode_sequences: self.peak_decode_sequences,
            peak_resident_tokens: self.peak_resident_tokens,
            peak_kv_blocks: self.peak_kv_blocks,
            peak_allocated_kv_tokens: self.peak_allocated_kv_tokens,
            peak_kv_fragmentation_tokens: self.peak_kv_fragmentation_tokens,
            peak_kv_block_table_bytes: self.peak_kv_block_table_bytes,
            kv_cache_owner_slots: self.kv_cache_owner_slots,
            decode_sequence_utilization: self.decode_sequence_utilization,
            resident_token_utilization: self.resident_token_utilization,
            kv_block_utilization: self.kv_block_utilization,
        }
    }
}

fn push_interval(intervals: &mut Vec<(f64, f64)>, start_s: f64, finish_s: f64) -> bool {
    if start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
        intervals.push((start_s, finish_s));
        true
    } else {
        false
    }
}

fn interval_events(intervals: &[(f64, f64)]) -> Vec<(f64, i32)> {
    let mut events = Vec::with_capacity(intervals.len() * 2);
    for (start_s, finish_s) in intervals {
        if start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
            events.push((*start_s, 1));
            events.push((*finish_s, -1));
        }
    }
    events.sort_by(|left, right| left.0.total_cmp(&right.0));
    events
}

fn deduplicate_intervals_by_span(intervals: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut deduplicated = intervals
        .iter()
        .copied()
        .filter(|(start_s, finish_s)| {
            start_s.is_finite() && finish_s.is_finite() && finish_s > start_s
        })
        .collect::<Vec<_>>();
    deduplicated.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    deduplicated.dedup_by(|left, right| {
        left.0.total_cmp(&right.0) == std::cmp::Ordering::Equal
            && left.1.total_cmp(&right.1) == std::cmp::Ordering::Equal
    });
    deduplicated
}

fn peak_active_intervals(intervals: &[(f64, f64)]) -> u32 {
    let events = interval_events(intervals);
    let mut active = 0_i32;
    let mut peak = 0_i32;
    let mut idx = 0;

    while idx < events.len() {
        let time_s = events[idx].0;
        let mut delta = 0_i32;
        while idx < events.len() && events[idx].0.total_cmp(&time_s) == std::cmp::Ordering::Equal {
            delta += events[idx].1;
            idx += 1;
        }
        active = (active + delta).max(0);
        peak = peak.max(active);
    }

    peak.max(0) as u32
}

fn interval_utilization(intervals: &[(f64, f64)], configured_slots: u32) -> f64 {
    if configured_slots == 0 {
        return 0.0;
    }
    let events = interval_events(intervals);
    if events.len() < 2 {
        return 0.0;
    }

    let first_s = events[0].0;
    let mut last_s = first_s;
    let mut active = 0_i32;
    let mut busy_slot_s = 0.0;
    let mut idx = 0;

    while idx < events.len() {
        let time_s = events[idx].0;
        if time_s > last_s && active > 0 {
            busy_slot_s += f64::from(active) * (time_s - last_s);
        }

        let mut delta = 0_i32;
        while idx < events.len() && events[idx].0.total_cmp(&time_s) == std::cmp::Ordering::Equal {
            delta += events[idx].1;
            idx += 1;
        }
        active = (active + delta).max(0);
        last_s = time_s;
    }

    let window_s = (last_s - first_s).max(0.0);
    if window_s <= 0.0 {
        0.0
    } else {
        (busy_slot_s / (window_s * f64::from(configured_slots))).max(0.0)
    }
}

pub(super) fn phase_worker_slots(
    phase: &str,
    prefill_slots: u32,
    decode_slots: u32,
    kv_transfer_slots: Option<u32>,
) -> u32 {
    match phase {
        "prefill" => prefill_slots,
        "decode" => decode_slots,
        "kv_transfer" => kv_transfer_slots.unwrap_or(1),
        _ => 1,
    }
}

fn phase_deduplicates_worker_intervals(phase: &str, traffic: &ServingTraffic) -> bool {
    match phase {
        "prefill" => matches!(
            &traffic.prefill_batching,
            ServingPrefillBatching::Continuous { .. }
        ),
        "decode" => matches!(
            &traffic.decode_batching,
            ServingDecodeBatching::Continuous { .. }
        ),
        _ => false,
    }
}

fn kv_transfer_observation_gpus(observation: &ServingRequestObservation) -> Vec<GpuAddr> {
    let mut gpus = BTreeSet::new();
    gpus.extend(route_worker_gpus(
        &observation.prefill_route_gpus,
        observation.prefill_node,
    ));
    gpus.extend(route_worker_gpus(
        &observation.decode_route_gpus,
        observation.decode_node,
    ));
    gpus.into_iter().collect()
}

fn kv_worker_slot_capacity_observations(
    observations: &[ServingRequestObservation],
    traffic: &ServingTraffic,
) -> BTreeMap<GpuAddr, Vec<ServingWorkerKvSlotObservation>> {
    let mut slot_events: BTreeMap<(GpuAddr, u32), Vec<CapacityEvent>> = BTreeMap::new();
    for observation in observations {
        for ownership in &observation.kv_block_ownership {
            if !ownership.allocated_at_s.is_finite()
                || !ownership.released_at_s.is_finite()
                || ownership.released_at_s < ownership.allocated_at_s
            {
                continue;
            }
            for slot_ownership in &ownership.worker_slot_ownership {
                let allocation = KvAllocation {
                    blocks: slot_ownership.kv_blocks,
                    allocated_tokens: slot_ownership.allocated_kv_tokens,
                    fragmentation_tokens: slot_ownership.kv_fragmentation_tokens,
                    block_table_bytes: slot_ownership.block_table_bytes,
                };
                let events = slot_events
                    .entry((ownership.owner, slot_ownership.slot))
                    .or_default();
                events.push(CapacityEvent::new(
                    ownership.allocated_at_s,
                    i64::from(slot_ownership.decode_sequences),
                    slot_ownership.resident_tokens as i128,
                    allocation,
                    1,
                ));
                events.push(CapacityEvent::new(
                    ownership.released_at_s,
                    -i64::from(slot_ownership.decode_sequences),
                    -(slot_ownership.resident_tokens as i128),
                    allocation,
                    -1,
                ));
            }
        }
    }

    let configured_decode_slots = decode_worker_slots_per_gpu(traffic)
        .min(u32::MAX as usize)
        .max(1) as u32;
    let decode_sequence_capacity = traffic
        .max_decode_sequences_per_gpu
        .map(|capacity| per_worker_slot_capacity_u32(capacity, configured_decode_slots));
    let resident_token_capacity = traffic
        .max_resident_tokens_per_gpu
        .map(|capacity| per_worker_slot_capacity_u64(capacity, configured_decode_slots));
    let kv_block_capacity = traffic
        .max_kv_blocks_per_gpu
        .map(|capacity| per_worker_slot_capacity_u64(capacity, configured_decode_slots));

    let mut by_gpu: BTreeMap<GpuAddr, Vec<ServingWorkerKvSlotObservation>> = BTreeMap::new();
    for ((gpu, slot), events) in slot_events {
        let peaks = capacity_peaks(events);
        by_gpu
            .entry(gpu)
            .or_default()
            .push(ServingWorkerKvSlotObservation {
                slot,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                decode_sequence_utilization: optional_u32_utilization(
                    peaks.decode_sequences,
                    decode_sequence_capacity,
                ),
                resident_token_utilization: optional_u64_utilization(
                    peaks.resident_tokens,
                    resident_token_capacity,
                ),
                kv_block_utilization: optional_u64_utilization(peaks.kv_blocks, kv_block_capacity),
            });
    }
    for slot_observations in by_gpu.values_mut() {
        slot_observations.sort_by_key(|observation| observation.slot);
    }
    by_gpu
}

fn per_worker_slot_capacity_u32(capacity: u32, configured_slots: u32) -> u32 {
    let configured_slots = u64::from(configured_slots.max(1));
    let capacity = u64::from(capacity);
    capacity.div_ceil(configured_slots).min(u64::from(u32::MAX)) as u32
}

fn per_worker_slot_capacity_u64(capacity: u64, configured_slots: u32) -> u64 {
    capacity.div_ceil(u64::from(configured_slots.max(1)))
}

fn optional_u32_utilization(value: u32, capacity: Option<u32>) -> f64 {
    capacity
        .filter(|capacity| *capacity > 0)
        .map(|capacity| f64::from(value) / f64::from(capacity))
        .unwrap_or(0.0)
}

fn optional_u64_utilization(value: u64, capacity: Option<u64>) -> f64 {
    capacity
        .filter(|capacity| *capacity > 0)
        .map(|capacity| value as f64 / capacity as f64)
        .unwrap_or(0.0)
}

pub(super) fn service_observations(
    traffic: &ServingTraffic,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    observations: &[ServingRequestObservation],
    workers: &[ServingWorkerObservation],
) -> Vec<ServingServiceObservation> {
    let prefill_slots = traffic.max_prefill_worker_slots_per_gpu.unwrap_or(1).max(1);
    let decode_slots = traffic.max_decode_worker_slots_per_gpu.unwrap_or(1).max(1);
    let kv_transfer_slots = traffic
        .max_kv_transfer_worker_slots_per_gpu
        .unwrap_or(1)
        .max(1);
    let prefill_gpu_count = unique_placement_gpu_count(prefill_score);
    let decode_gpu_count = unique_placement_gpu_count(decode_score);
    let kv_transfer_node_count = prefill_nodes
        .iter()
        .copied()
        .chain(decode_nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .len()
        .min(u32::MAX as usize) as u32;
    let kv_transfer_gpu_count = prefill_gpu_count.saturating_add(decode_gpu_count);
    let inputs = ServiceObservationInputs {
        traffic,
        observations,
        workers,
    };

    vec![
        service_observation(
            "prefill",
            traffic.services.prefill,
            prefill_slots,
            prefill_worker_slots_per_gpu(traffic),
            prefill_nodes.len().min(u32::MAX as usize) as u32,
            prefill_gpu_count,
            inputs,
        ),
        service_observation(
            "decode",
            traffic.services.decode,
            decode_slots,
            decode_worker_slots_per_gpu(traffic),
            decode_nodes.len().min(u32::MAX as usize) as u32,
            decode_gpu_count,
            inputs,
        ),
        service_observation(
            "kv_transfer",
            traffic.services.kv_transfer,
            kv_transfer_slots,
            kv_transfer_worker_slots_per_gpu(traffic).unwrap_or(0),
            kv_transfer_node_count,
            kv_transfer_gpu_count,
            inputs,
        ),
    ]
}

#[derive(Clone, Copy, Debug)]
struct ServiceObservationInputs<'a> {
    traffic: &'a ServingTraffic,
    observations: &'a [ServingRequestObservation],
    workers: &'a [ServingWorkerObservation],
}

fn service_observation(
    phase: &str,
    service: ServingServicePhaseConfig,
    configured_worker_slots: u32,
    effective_worker_slots: usize,
    node_count: u32,
    gpu_count: u32,
    inputs: ServiceObservationInputs<'_>,
) -> ServingServiceObservation {
    let phase_workers = inputs
        .workers
        .iter()
        .filter(|worker| worker.phase == phase)
        .collect::<Vec<_>>();
    let admission = service_admission_observation(phase, inputs.traffic, inputs.observations);
    let worker_slot_utilization =
        mean_weighted_by_requests(&phase_workers, |worker| worker.worker_slot_utilization);
    let worker_queue_s = mean_weighted_by_requests(&phase_workers, |worker| worker.worker_queue_s);
    let resource_queue_s =
        mean_weighted_by_requests(&phase_workers, |worker| worker.resource_queue_s);
    let service_s = mean_weighted_by_requests(&phase_workers, |worker| worker.service_s);
    let backpressure_state = service_backpressure_state(
        service,
        admission.backpressure_rejections,
        admission.timeout_rejections,
        admission.queue_max_s,
    );

    ServingServiceObservation {
        phase: phase.to_string(),
        health: service.health,
        accepts_requests: service.health.accepts_requests(),
        worker_scale: service.worker_scale,
        configured_worker_slots_per_gpu: configured_worker_slots,
        effective_worker_slots_per_gpu: effective_worker_slots.min(u32::MAX as usize) as u32,
        node_count,
        gpu_count,
        request_count: admission.request_count,
        admitted_requests: admission.admitted_requests,
        completed_requests: admission.completed_requests,
        failed_requests: admission.failed_requests,
        rejected_requests: admission.rejected_requests,
        timed_out_requests: admission.timed_out_requests,
        cancelled_requests: admission.cancelled_requests,
        queue_cap_s: admission.queue_cap_s,
        queue_cap_request_count: admission.queue_cap_request_count,
        queue_cap_hit_count: admission.queue_cap_hit_count,
        decode_iteration_queue_cap_s: admission.decode_iteration_queue_cap_s,
        decode_iteration_queue_cap_request_count: admission
            .decode_iteration_queue_cap_request_count,
        decode_iteration_queue_cap_hit_count: admission.decode_iteration_queue_cap_hit_count,
        backpressure_rejections: admission.backpressure_rejections,
        timeout_rejections: admission.timeout_rejections,
        backpressure_state,
        worker_slot_utilization,
        queue_s: admission.queue_s,
        queue_p95_s: admission.queue_p95_s,
        queue_max_s: admission.queue_max_s,
        worker_queue_s,
        resource_queue_s,
        service_s,
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ServiceAdmissionObservation {
    request_count: u32,
    admitted_requests: u32,
    completed_requests: u32,
    failed_requests: u32,
    rejected_requests: u32,
    timed_out_requests: u32,
    cancelled_requests: u32,
    queue_cap_s: Option<f64>,
    queue_cap_request_count: u32,
    queue_cap_hit_count: u32,
    decode_iteration_queue_cap_s: Option<f64>,
    decode_iteration_queue_cap_request_count: u32,
    decode_iteration_queue_cap_hit_count: u32,
    backpressure_rejections: u32,
    timeout_rejections: u32,
    queue_s: f64,
    queue_p95_s: f64,
    queue_max_s: f64,
}

fn service_admission_observation(
    phase: &str,
    traffic: &ServingTraffic,
    observations: &[ServingRequestObservation],
) -> ServiceAdmissionObservation {
    let mut accumulator = ServiceAdmissionAccumulator::default();
    for observation in observations {
        if !service_phase_attempted(phase, observation) {
            continue;
        }
        accumulator.record(phase, traffic, observation);
    }
    accumulator.into_observation()
}

#[derive(Clone, Debug, Default)]
struct ServiceAdmissionAccumulator {
    request_count: u32,
    admitted_requests: u32,
    completed_requests: u32,
    rejected_requests: u32,
    timed_out_requests: u32,
    cancelled_requests: u32,
    queue_cap_s: Option<f64>,
    queue_cap_request_count: u32,
    queue_cap_hit_count: u32,
    decode_iteration_queue_cap_s: Option<f64>,
    decode_iteration_queue_cap_request_count: u32,
    decode_iteration_queue_cap_hit_count: u32,
    backpressure_rejections: u32,
    timeout_rejections: u32,
    queue_samples_s: Vec<f64>,
}

impl ServiceAdmissionAccumulator {
    fn record(
        &mut self,
        phase: &str,
        traffic: &ServingTraffic,
        observation: &ServingRequestObservation,
    ) {
        self.request_count = self.request_count.saturating_add(1);
        if service_phase_admitted(phase, observation) {
            self.admitted_requests = self.admitted_requests.saturating_add(1);
        }
        if service_phase_completed(phase, observation) {
            self.completed_requests = self.completed_requests.saturating_add(1);
        }
        if observation.status == ServingRequestStatus::Cancelled {
            self.cancelled_requests = self.cancelled_requests.saturating_add(1);
        }

        if let Some(queue_s) = service_phase_queue_s(phase, observation) {
            self.queue_samples_s.push(queue_s);
        }
        if let Some(cap_s) = service_phase_queue_cap_s(phase, traffic, observation.request_idx) {
            self.queue_cap_request_count = self.queue_cap_request_count.saturating_add(1);
            self.queue_cap_s = min_optional_f64(self.queue_cap_s, cap_s);
        }
        if phase == "decode"
            && let Some(cap_s) =
                traffic.effective_max_decode_iteration_queue_delay_s(observation.request_idx)
        {
            self.decode_iteration_queue_cap_request_count = self
                .decode_iteration_queue_cap_request_count
                .saturating_add(1);
            self.decode_iteration_queue_cap_s =
                min_optional_f64(self.decode_iteration_queue_cap_s, cap_s);
        }

        let Some(rejection) = observation.rejection.as_ref() else {
            return;
        };
        if rejection.phase != phase {
            return;
        }
        if observation.status == ServingRequestStatus::RejectedAdmission {
            self.rejected_requests = self.rejected_requests.saturating_add(1);
        }
        if observation.status == ServingRequestStatus::TimedOut {
            self.timed_out_requests = self.timed_out_requests.saturating_add(1);
            self.timeout_rejections = self.timeout_rejections.saturating_add(1);
        }
        if rejection.category == "queueing" && rejection.code.ends_with("queue_delay_exceeded") {
            self.backpressure_rejections = self.backpressure_rejections.saturating_add(1);
            if rejection.code == "decode_iteration_queue_delay_exceeded" {
                self.decode_iteration_queue_cap_hit_count =
                    self.decode_iteration_queue_cap_hit_count.saturating_add(1);
            } else {
                self.queue_cap_hit_count = self.queue_cap_hit_count.saturating_add(1);
            }
        }
    }

    fn into_observation(self) -> ServiceAdmissionObservation {
        let failed_requests = self.request_count.saturating_sub(self.completed_requests);
        ServiceAdmissionObservation {
            request_count: self.request_count,
            admitted_requests: self.admitted_requests,
            completed_requests: self.completed_requests,
            failed_requests,
            rejected_requests: self.rejected_requests,
            timed_out_requests: self.timed_out_requests,
            cancelled_requests: self.cancelled_requests,
            queue_cap_s: self.queue_cap_s,
            queue_cap_request_count: self.queue_cap_request_count,
            queue_cap_hit_count: self.queue_cap_hit_count,
            decode_iteration_queue_cap_s: self.decode_iteration_queue_cap_s,
            decode_iteration_queue_cap_request_count: self.decode_iteration_queue_cap_request_count,
            decode_iteration_queue_cap_hit_count: self.decode_iteration_queue_cap_hit_count,
            backpressure_rejections: self.backpressure_rejections,
            timeout_rejections: self.timeout_rejections,
            queue_s: mean(&self.queue_samples_s),
            queue_p95_s: percentile(self.queue_samples_s.clone(), 0.95),
            queue_max_s: max_value(&self.queue_samples_s),
        }
    }
}

fn service_phase_attempted(phase: &str, observation: &ServingRequestObservation) -> bool {
    match phase {
        "prefill" => observation.arrival_s.is_finite(),
        "kv_transfer" => {
            observation.kv_transfer_bytes > 0
                || observation.kv_start_s.is_finite()
                || observation
                    .rejection
                    .as_ref()
                    .is_some_and(|rejection| rejection.phase == "kv_transfer")
        }
        "decode" => {
            observation.first_decode_start_s.is_finite()
                || !observation.decode_token_finish_s.is_empty()
                || observation.status.is_admitted() && observation.kv_finish_s.is_finite()
                || observation
                    .rejection
                    .as_ref()
                    .is_some_and(|rejection| rejection.phase == "decode")
        }
        _ => false,
    }
}

fn service_phase_admitted(phase: &str, observation: &ServingRequestObservation) -> bool {
    !observation.rejection.as_ref().is_some_and(|rejection| {
        rejection.phase == phase && observation.status == ServingRequestStatus::RejectedAdmission
    })
}

fn service_phase_completed(phase: &str, observation: &ServingRequestObservation) -> bool {
    match phase {
        "prefill" => observation.prefill_finish_s.is_finite(),
        "kv_transfer" => observation.kv_finish_s.is_finite(),
        "decode" => observation.status == ServingRequestStatus::Completed,
        _ => false,
    }
}

fn service_phase_queue_s(phase: &str, observation: &ServingRequestObservation) -> Option<f64> {
    let queue_s = match phase {
        "prefill" => observation.prefill_worker_queue_s + observation.prefill_resource_queue_s,
        "kv_transfer" => observation.kv_queue_s,
        "decode" => {
            let decode_queue_s =
                observation.decode_worker_queue_s + observation.decode_resource_queue_s;
            observation.decode_queue_s.max(decode_queue_s)
        }
        _ => return None,
    };
    queue_s.is_finite().then_some(queue_s.max(0.0))
}

fn service_phase_queue_cap_s(
    phase: &str,
    traffic: &ServingTraffic,
    request_idx: u32,
) -> Option<f64> {
    match phase {
        "prefill" => traffic.effective_max_queue_delay_s(request_idx),
        "kv_transfer" => traffic.effective_max_kv_queue_delay_s(request_idx),
        "decode" => traffic.effective_max_decode_queue_delay_s(request_idx),
        _ => None,
    }
}

fn min_optional_f64(current: Option<f64>, candidate: f64) -> Option<f64> {
    Some(match current {
        Some(current) => current.min(candidate),
        None => candidate,
    })
}

fn service_backpressure_state(
    service: ServingServicePhaseConfig,
    backpressure_rejections: u32,
    timeout_rejections: u32,
    queue_max_s: f64,
) -> String {
    if !service.health.accepts_requests() {
        "closed".to_string()
    } else if timeout_rejections > 0 {
        "timing_out".to_string()
    } else if backpressure_rejections > 0 {
        "rejecting".to_string()
    } else if queue_max_s.is_finite() && queue_max_s > 0.0 {
        "queued".to_string()
    } else {
        "open".to_string()
    }
}

fn unique_placement_gpu_count(score: &ScoredParallelismConfig) -> u32 {
    score
        .placement
        .rank_to_gpu
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        .min(u32::MAX as usize) as u32
}

fn mean_weighted_by_requests(
    workers: &[&ServingWorkerObservation],
    metric: impl Fn(&ServingWorkerObservation) -> f64,
) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0_u64;
    for worker in workers {
        let value = metric(worker);
        if !value.is_finite() {
            continue;
        }
        let weight = u64::from(worker.request_count.max(1));
        weighted_sum += value * weight as f64;
        total_weight = total_weight.saturating_add(weight);
    }
    if total_weight == 0 {
        0.0
    } else {
        weighted_sum / total_weight as f64
    }
}

pub(super) fn worker_observations(
    observations: &[ServingRequestObservation],
    gpu_capacity: &[ServingGpuCapacityObservation],
    traffic: &ServingTraffic,
) -> Vec<ServingWorkerObservation> {
    let mut workers: BTreeMap<(String, GpuAddr), WorkerObservationAccumulator> = BTreeMap::new();
    let prefill_slots = prefill_worker_slots_per_gpu(traffic).min(u32::MAX as usize) as u32;
    let decode_slots = decode_worker_slots_per_gpu(traffic).min(u32::MAX as usize) as u32;
    let kv_transfer_slots =
        kv_transfer_worker_slots_per_gpu(traffic).map(|slots| slots.min(u32::MAX as usize) as u32);
    let kv_worker_slot_capacity = kv_worker_slot_capacity_observations(observations, traffic);

    for observation in observations {
        for gpu in &observation.prefill_route_gpus {
            workers
                .entry(("prefill".to_string(), *gpu))
                .or_default()
                .record_prefill(observation);
        }
        for gpu in &observation.decode_route_gpus {
            workers
                .entry(("decode".to_string(), *gpu))
                .or_default()
                .record_decode(observation);
        }
        if kv_transfer_slots.is_some() && observation.kv_transfer_bytes > 0 {
            for gpu in kv_transfer_observation_gpus(observation) {
                workers
                    .entry(("kv_transfer".to_string(), gpu))
                    .or_default()
                    .record_kv_transfer(observation);
            }
        }
    }

    for capacity in gpu_capacity {
        let gpu = GpuAddr {
            node_id: capacity.node_id,
            local_gpu_id: capacity.local_gpu_id,
        };
        if capacity.peak_prefill_tokens > 0 {
            workers
                .entry(("prefill".to_string(), gpu))
                .or_default()
                .apply_prefill_capacity(capacity);
        }
        if capacity.peak_decode_sequences > 0
            || capacity.peak_resident_tokens > 0
            || capacity.peak_kv_blocks > 0
        {
            workers
                .entry(("decode".to_string(), gpu))
                .or_default()
                .apply_decode_capacity(capacity);
        }
    }
    for (gpu, slot_observations) in kv_worker_slot_capacity {
        workers
            .entry(("decode".to_string(), gpu))
            .or_default()
            .apply_decode_worker_slot_capacity(slot_observations);
    }

    workers
        .into_iter()
        .map(|((phase, gpu), accumulator)| {
            let configured_worker_slots =
                phase_worker_slots(&phase, prefill_slots, decode_slots, kv_transfer_slots);
            let deduplicate_intervals = phase_deduplicates_worker_intervals(&phase, traffic);
            accumulator.into_observation(phase, gpu, configured_worker_slots, deduplicate_intervals)
        })
        .collect()
}

#[derive(Default)]
struct MetricBreakdownAccumulator {
    request_count: u32,
    completed_requests: u32,
    failed_requests: u32,
    rejected_requests: u32,
    timed_out_requests: u32,
    cancelled_requests: u32,
    output_tokens: u64,
    metric_source_counts: BTreeMap<String, u32>,
    deadline_constrained_requests: u32,
    deadline_missed_requests: u32,
    ttft_slo_constrained_requests: u32,
    ttft_slo_missed_requests: u32,
    tpot_slo_constrained_requests: u32,
    tpot_slo_missed_requests: u32,
    itl_slo_constrained_requests: u32,
    itl_slo_missed_requests: u32,
    e2el_slo_constrained_requests: u32,
    e2el_slo_missed_requests: u32,
    ttft: Vec<f64>,
    tpot: Vec<f64>,
    itl: Vec<f64>,
    e2el: Vec<f64>,
}

pub(super) fn metric_breakdowns(
    observations: &[ServingRequestObservation],
    measurement_start_s: f64,
    measurement_end_s: f64,
) -> Vec<ServingMetricBreakdown> {
    let measurement_duration_s = (measurement_end_s - measurement_start_s).max(0.0);
    let mut groups: BTreeMap<(String, String), MetricBreakdownAccumulator> = BTreeMap::new();

    for observation in observations {
        if observation.arrival_s + 1e-12 < measurement_start_s
            || observation.arrival_s > measurement_end_s + 1e-12
        {
            continue;
        }
        if let Some(tenant) = observation.tenant.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "tenant",
                tenant,
                observation,
                measurement_duration_s,
            );
        }
        if let Some(model_id) = observation.model_id.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "model_id",
                model_id,
                observation,
                measurement_duration_s,
            );
        }
        if let Some(traffic_class) = observation.traffic_class.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "traffic_class",
                traffic_class,
                observation,
                measurement_duration_s,
            );
        }
        if let Some(shape_profile) = observation.shape_profile.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "shape_profile",
                shape_profile,
                observation,
                measurement_duration_s,
            );
        }
        accumulate_metric_breakdown(
            &mut groups,
            "priority",
            &priority_key(observation.priority),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "prefill_node",
            &node_key(observation.prefill_node),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "decode_node",
            &node_key(observation.decode_node),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "prefill_route",
            &node_set_key(&observation.prefill_route_nodes),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "decode_route",
            &node_set_key(&observation.decode_route_nodes),
            observation,
            measurement_duration_s,
        );
    }

    groups
        .into_iter()
        .map(|((group, key), accumulator)| {
            accumulator.into_breakdown(group, key, measurement_duration_s)
        })
        .collect()
}

fn node_key(node_id: NodeId) -> String {
    format!("node-{node_id}")
}

pub(super) fn priority_key(priority: i32) -> String {
    format!("priority-{priority}")
}

fn node_set_key(nodes: &[NodeId]) -> String {
    let mut nodes = nodes.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    format!(
        "nodes[{}]",
        nodes
            .iter()
            .map(|node| node.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn accumulate_metric_breakdown(
    groups: &mut BTreeMap<(String, String), MetricBreakdownAccumulator>,
    group: &str,
    key: &str,
    observation: &ServingRequestObservation,
    measurement_duration_s: f64,
) {
    let accumulator = groups
        .entry((group.to_string(), key.to_string()))
        .or_default();
    accumulator.record(observation, measurement_duration_s);
}

impl MetricBreakdownAccumulator {
    fn record(&mut self, observation: &ServingRequestObservation, _measurement_duration_s: f64) {
        self.request_count = self.request_count.saturating_add(1);
        if observation.deadline_s.is_some() {
            self.deadline_constrained_requests =
                self.deadline_constrained_requests.saturating_add(1);
        }
        if observation.deadline_missed {
            self.deadline_missed_requests = self.deadline_missed_requests.saturating_add(1);
        }
        self.record_slo_misses(observation);
        match observation.status {
            ServingRequestStatus::RejectedAdmission => {
                self.rejected_requests = self.rejected_requests.saturating_add(1);
            }
            ServingRequestStatus::TimedOut => {
                self.timed_out_requests = self.timed_out_requests.saturating_add(1);
            }
            ServingRequestStatus::Cancelled => {
                self.cancelled_requests = self.cancelled_requests.saturating_add(1);
            }
            ServingRequestStatus::Pending | ServingRequestStatus::Completed => {}
        }
        if !observation.status.is_completed() {
            self.failed_requests = self.failed_requests.saturating_add(1);
            return;
        }

        self.completed_requests = self.completed_requests.saturating_add(1);
        let metric_source_count = self
            .metric_source_counts
            .entry(observation.metric_source.clone())
            .or_default();
        *metric_source_count = metric_source_count.saturating_add(1);
        self.output_tokens = self
            .output_tokens
            .saturating_add(observation_output_tokens(observation));
        push_finite(&mut self.ttft, observation.ttft_s);
        push_finite(&mut self.tpot, observation.tpot_s);
        push_finite(&mut self.itl, observation.itl_s);
        push_finite(&mut self.e2el, observation.e2el_s);
    }

    fn record_slo_misses(&mut self, observation: &ServingRequestObservation) {
        if observation.slo.ttft_s.is_some() {
            self.ttft_slo_constrained_requests =
                self.ttft_slo_constrained_requests.saturating_add(1);
            if observation.ttft_slo_missed {
                self.ttft_slo_missed_requests = self.ttft_slo_missed_requests.saturating_add(1);
            }
        }
        if observation.slo.tpot_s.is_some() {
            self.tpot_slo_constrained_requests =
                self.tpot_slo_constrained_requests.saturating_add(1);
            if observation.tpot_slo_missed {
                self.tpot_slo_missed_requests = self.tpot_slo_missed_requests.saturating_add(1);
            }
        }
        if observation.slo.itl_s.is_some() {
            self.itl_slo_constrained_requests = self.itl_slo_constrained_requests.saturating_add(1);
            if observation.itl_slo_missed {
                self.itl_slo_missed_requests = self.itl_slo_missed_requests.saturating_add(1);
            }
        }
        if observation.slo.e2el_s.is_some() {
            self.e2el_slo_constrained_requests =
                self.e2el_slo_constrained_requests.saturating_add(1);
            if observation.e2el_slo_missed {
                self.e2el_slo_missed_requests = self.e2el_slo_missed_requests.saturating_add(1);
            }
        }
    }

    fn into_breakdown(
        self,
        group: String,
        key: String,
        measurement_duration_s: f64,
    ) -> ServingMetricBreakdown {
        let deadline_miss_rate = if self.deadline_constrained_requests > 0 {
            f64::from(self.deadline_missed_requests) / f64::from(self.deadline_constrained_requests)
        } else {
            0.0
        };
        let throughput_tokens_per_s = if measurement_duration_s > 0.0 {
            self.output_tokens as f64 / measurement_duration_s
        } else {
            0.0
        };
        let ttft_s = mean(&self.ttft);
        let ttft_p90_s = percentile(self.ttft.clone(), 0.90);
        let ttft_p95_s = percentile(self.ttft.clone(), 0.95);
        let ttft_max_s = max_value(&self.ttft);
        let tpot_s = mean(&self.tpot);
        let tpot_p90_s = percentile(self.tpot.clone(), 0.90);
        let tpot_p95_s = percentile(self.tpot.clone(), 0.95);
        let tpot_max_s = max_value(&self.tpot);
        let itl_s = mean(&self.itl);
        let itl_p90_s = percentile(self.itl.clone(), 0.90);
        let itl_p95_s = percentile(self.itl.clone(), 0.95);
        let itl_max_s = max_value(&self.itl);
        let e2el_s = mean(&self.e2el);
        let e2el_p90_s = percentile(self.e2el.clone(), 0.90);
        let e2el_p95_s = percentile(self.e2el.clone(), 0.95);
        let e2el_max_s = max_value(&self.e2el);
        let metric_source_counts = self
            .metric_source_counts
            .into_iter()
            .map(
                |(metric_source, request_count)| ServingMeasurementMetricSourceCount {
                    metric_source,
                    request_count,
                },
            )
            .collect::<Vec<_>>();
        let lifecycle_event_metric_request_count = metric_source_counts
            .iter()
            .find(|count| count.metric_source == "request_lifecycle_events")
            .map(|count| count.request_count)
            .unwrap_or(0);
        let fallback_metric_request_count = metric_source_counts
            .iter()
            .filter(|count| count.metric_source != "request_lifecycle_events")
            .map(|count| count.request_count)
            .sum();

        ServingMetricBreakdown {
            group,
            key,
            request_count: self.request_count,
            completed_requests: self.completed_requests,
            failed_requests: self.failed_requests,
            rejected_requests: self.rejected_requests,
            timed_out_requests: self.timed_out_requests,
            cancelled_requests: self.cancelled_requests,
            output_tokens: self.output_tokens,
            lifecycle_event_metric_request_count,
            fallback_metric_request_count,
            metric_source_counts,
            deadline_constrained_requests: self.deadline_constrained_requests,
            deadline_missed_requests: self.deadline_missed_requests,
            deadline_miss_rate,
            ttft_slo_constrained_requests: self.ttft_slo_constrained_requests,
            ttft_slo_missed_requests: self.ttft_slo_missed_requests,
            ttft_slo_miss_rate: ratio_or_infinity(
                self.ttft_slo_missed_requests,
                self.ttft_slo_constrained_requests,
            ),
            tpot_slo_constrained_requests: self.tpot_slo_constrained_requests,
            tpot_slo_missed_requests: self.tpot_slo_missed_requests,
            tpot_slo_miss_rate: ratio_or_infinity(
                self.tpot_slo_missed_requests,
                self.tpot_slo_constrained_requests,
            ),
            itl_slo_constrained_requests: self.itl_slo_constrained_requests,
            itl_slo_missed_requests: self.itl_slo_missed_requests,
            itl_slo_miss_rate: ratio_or_infinity(
                self.itl_slo_missed_requests,
                self.itl_slo_constrained_requests,
            ),
            e2el_slo_constrained_requests: self.e2el_slo_constrained_requests,
            e2el_slo_missed_requests: self.e2el_slo_missed_requests,
            e2el_slo_miss_rate: ratio_or_infinity(
                self.e2el_slo_missed_requests,
                self.e2el_slo_constrained_requests,
            ),
            ttft_s,
            ttft_p90_s,
            ttft_p95_s,
            ttft_max_s,
            tpot_s,
            tpot_p90_s,
            tpot_p95_s,
            tpot_max_s,
            itl_s,
            itl_p90_s,
            itl_p95_s,
            itl_max_s,
            throughput_tokens_per_s,
            e2el_s,
            e2el_p90_s,
            e2el_p95_s,
            e2el_max_s,
        }
    }
}

fn push_finite(values: &mut Vec<f64>, value: f64) {
    if value.is_finite() {
        values.push(value);
    }
}
