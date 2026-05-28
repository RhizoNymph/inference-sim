use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CapacityProfile {
    pub(super) peak_prefill_tokens: u64,
    pub(super) peak_prefill_tokens_per_node: u64,
    pub(super) peak_prefill_tokens_per_gpu: u64,
    pub(super) peak_decode_sequences: u32,
    pub(super) peak_resident_tokens: u64,
    pub(super) peak_decode_sequences_per_node: u32,
    pub(super) peak_resident_tokens_per_node: u64,
    pub(super) peak_decode_sequences_per_gpu: u32,
    pub(super) peak_resident_tokens_per_gpu: u64,
    pub(super) peak_kv_blocks: u64,
    pub(super) peak_allocated_kv_tokens: u64,
    pub(super) peak_kv_fragmentation_tokens: u64,
    pub(super) peak_kv_block_table_bytes: u64,
    pub(super) peak_kv_blocks_per_node: u64,
    pub(super) peak_allocated_kv_tokens_per_node: u64,
    pub(super) peak_kv_fragmentation_tokens_per_node: u64,
    pub(super) peak_kv_block_table_bytes_per_node: u64,
    pub(super) peak_kv_blocks_per_gpu: u64,
    pub(super) peak_allocated_kv_tokens_per_gpu: u64,
    pub(super) peak_kv_fragmentation_tokens_per_gpu: u64,
    pub(super) peak_kv_block_table_bytes_per_gpu: u64,
    pub(super) decode_sequence_utilization: f64,
    pub(super) resident_token_utilization: f64,
    pub(super) kv_block_utilization: f64,
    pub(super) decode_sequence_per_node_utilization: f64,
    pub(super) resident_token_per_node_utilization: f64,
    pub(super) kv_block_per_node_utilization: f64,
    pub(super) decode_sequence_per_gpu_utilization: f64,
    pub(super) resident_token_per_gpu_utilization: f64,
    pub(super) kv_block_per_gpu_utilization: f64,
    pub(super) nodes: Vec<ServingNodeCapacityObservation>,
    pub(super) gpus: Vec<ServingGpuCapacityObservation>,
    pub(super) traffic_classes: Vec<ServingTrafficClassCapacityObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CapacityRejection {
    pub(super) reason: String,
    pub(super) bottleneck: String,
    pub(super) resource: String,
    pub(super) code: String,
    pub(super) observed: f64,
    pub(super) limit: f64,
    pub(super) unit: String,
}

impl CapacityRejection {
    pub(super) fn into_serving_rejection(self) -> ServingRejection {
        let remediation = capacity_remediation(&self.code).map(str::to_string);
        let phase = if self.code.starts_with("prefill_") {
            "prefill"
        } else {
            "decode"
        };
        ServingRejection {
            phase: phase.to_string(),
            category: "capacity".to_string(),
            resource: self.resource,
            code: self.code,
            observed: Some(self.observed),
            limit: Some(self.limit),
            unit: Some(self.unit),
            remediation,
            message: self.reason,
        }
    }
}

fn capacity_remediation(code: &str) -> Option<&'static str> {
    match code {
        "prefill_capacity_exceeded"
        | "prefill_capacity_per_node_exceeded"
        | "prefill_capacity_per_gpu_exceeded" => Some(
            "increase prefill capacity, add prefill workers, lower prefill batch/chunk tokens, or reduce prompt/batch concurrency",
        ),
        "decode_capacity_exceeded" | "decode_capacity_per_node_exceeded" => Some(
            "increase decode replicas or decode sequence capacity, or lower arrival concurrency",
        ),
        "decode_capacity_per_gpu_exceeded" => Some(
            "increase decode GPU capacity, use more decode tensor/data ranks, or lower arrival concurrency",
        ),
        "kv_residency_capacity_exceeded"
        | "kv_residency_capacity_per_node_exceeded"
        | "kv_residency_capacity_per_gpu_exceeded" => Some(
            "increase KV residency capacity, add decode workers, or reduce max sequence/batch size",
        ),
        "kv_block_capacity_exceeded"
        | "kv_block_capacity_per_node_exceeded"
        | "kv_block_capacity_per_gpu_exceeded" => Some(
            "increase KV block capacity, add decode workers, increase KV block budget, or reduce max sequence/batch size",
        ),
        _ => None,
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct KvAllocation {
    pub(super) blocks: u64,
    pub(super) allocated_tokens: u64,
    pub(super) fragmentation_tokens: u64,
    pub(super) block_table_bytes: u64,
}

pub(super) const DEFAULT_KV_BLOCK_TOKENS: u32 = 16;
pub(super) const KV_BLOCK_TABLE_ENTRY_BYTES: u64 = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct KvOwnerAllocation {
    pub(super) allocation_id: String,
    pub(super) owner: GpuAddr,
    pub(super) block_start: u64,
    pub(super) block_end: u64,
    pub(super) resident_tokens: u64,
    pub(super) kv_blocks: u64,
    pub(super) allocated_kv_tokens: u64,
    pub(super) kv_fragmentation_tokens: u64,
    pub(super) block_table_entries: u64,
    pub(super) block_table_bytes: u64,
}

pub(super) fn kv_block_tokens(traffic: &ServingTraffic) -> u32 {
    traffic
        .kv_block_tokens
        .unwrap_or(DEFAULT_KV_BLOCK_TOKENS)
        .max(1)
}

pub(super) fn sequence_kv_allocation(
    batch_size: u32,
    max_sequence_tokens: u32,
    block_tokens: u32,
) -> KvAllocation {
    let block_tokens = u64::from(block_tokens.max(1));
    let blocks_per_sequence = u64::from(max_sequence_tokens.max(1)).div_ceil(block_tokens);
    let blocks = u64::from(batch_size.max(1)).saturating_mul(blocks_per_sequence);
    let resident_tokens =
        u64::from(batch_size.max(1)).saturating_mul(u64::from(max_sequence_tokens.max(1)));
    let allocated_tokens = blocks.saturating_mul(block_tokens);
    KvAllocation {
        blocks,
        allocated_tokens,
        fragmentation_tokens: allocated_tokens.saturating_sub(resident_tokens),
        block_table_bytes: blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
    }
}

pub(super) fn kv_owner_allocations(state: &DecodeRequestState) -> Vec<KvOwnerAllocation> {
    let owners = decode_owner_gpus(state);
    if owners.is_empty() || state.kv_cache_blocks == 0 {
        return Vec::new();
    }

    let block_counts = partition_units(state.kv_cache_blocks, owners.len());
    let resident_tokens =
        u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1));
    let token_counts = partition_tokens_by_block_capacity(
        resident_tokens,
        &block_counts,
        u64::from(state.kv_block_tokens.max(1)),
    );
    let mut block_start = 0_u64;
    owners
        .into_iter()
        .zip(block_counts)
        .zip(token_counts)
        .filter_map(|((owner, kv_blocks), resident_tokens)| {
            if kv_blocks == 0 {
                return None;
            }
            let block_end = block_start.saturating_add(kv_blocks);
            let allocated_kv_tokens =
                kv_blocks.saturating_mul(u64::from(state.kv_block_tokens.max(1)));
            let allocation = KvOwnerAllocation {
                allocation_id: format!(
                    "request-{}:node-{}:gpu-{}:blocks-{}-{}",
                    state.request_idx, owner.node_id, owner.local_gpu_id, block_start, block_end
                ),
                owner,
                block_start,
                block_end,
                resident_tokens,
                kv_blocks,
                allocated_kv_tokens,
                kv_fragmentation_tokens: allocated_kv_tokens.saturating_sub(resident_tokens),
                block_table_entries: kv_blocks,
                block_table_bytes: kv_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
            };
            block_start = block_end;
            Some(allocation)
        })
        .collect()
}

pub(super) fn partition_units(total: u64, parts: usize) -> Vec<u64> {
    if parts == 0 {
        return Vec::new();
    }
    let base = total / parts as u64;
    let remainder = total % parts as u64;
    (0..parts)
        .map(|idx| base + u64::from((idx as u64) < remainder))
        .collect()
}

pub(super) fn partition_tokens_by_block_capacity(
    total_tokens: u64,
    block_counts: &[u64],
    block_tokens: u64,
) -> Vec<u64> {
    let capacities = block_counts
        .iter()
        .map(|blocks| blocks.saturating_mul(block_tokens.max(1)))
        .collect::<Vec<_>>();
    let mut remaining_tokens = total_tokens;
    let mut remaining_capacity = capacities.iter().copied().sum::<u64>();
    let mut token_counts = Vec::with_capacity(capacities.len());

    for capacity in capacities {
        if remaining_tokens == 0 || capacity == 0 || remaining_capacity == 0 {
            token_counts.push(0);
            remaining_capacity = remaining_capacity.saturating_sub(capacity);
            continue;
        }
        let tokens = if capacity >= remaining_capacity {
            remaining_tokens.min(capacity)
        } else {
            (((remaining_tokens as u128) * (capacity as u128)) / remaining_capacity as u128)
                .min(capacity as u128)
                .min(remaining_tokens as u128) as u64
        };
        token_counts.push(tokens);
        remaining_tokens = remaining_tokens.saturating_sub(tokens);
        remaining_capacity = remaining_capacity.saturating_sub(capacity);
    }

    token_counts
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct CapacityEvent {
    pub(super) time_s: f64,
    pub(super) sequence_delta: i64,
    pub(super) resident_token_delta: i128,
    pub(super) kv_block_delta: i128,
    pub(super) allocated_token_delta: i128,
    pub(super) fragmentation_token_delta: i128,
    pub(super) block_table_byte_delta: i128,
}

impl CapacityEvent {
    pub(super) fn new(
        time_s: f64,
        sequence_delta: i64,
        resident_token_delta: i128,
        allocation: KvAllocation,
        sign: i128,
    ) -> Self {
        Self {
            time_s,
            sequence_delta,
            resident_token_delta,
            kv_block_delta: sign * i128::from(allocation.blocks),
            allocated_token_delta: sign * i128::from(allocation.allocated_tokens),
            fragmentation_token_delta: sign * i128::from(allocation.fragmentation_tokens),
            block_table_byte_delta: sign * i128::from(allocation.block_table_bytes),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct CapacityPeaks {
    pub(super) decode_sequences: u32,
    pub(super) resident_tokens: u64,
    pub(super) kv_blocks: u64,
    pub(super) allocated_kv_tokens: u64,
    pub(super) kv_fragmentation_tokens: u64,
    pub(super) kv_block_table_bytes: u64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct PrefillCapacityEvent {
    pub(super) time_s: f64,
    pub(super) token_delta: i128,
}

pub(super) fn capacity_profile(
    states: &[DecodeRequestState],
    traffic: &ServingTraffic,
) -> CapacityProfile {
    let mut prefill_events = Vec::with_capacity(states.len() * 2);
    let mut prefill_node_events: BTreeMap<NodeId, Vec<PrefillCapacityEvent>> = BTreeMap::new();
    let mut prefill_gpu_events: BTreeMap<GpuAddr, Vec<PrefillCapacityEvent>> = BTreeMap::new();
    let mut prefill_class_events: BTreeMap<String, Vec<PrefillCapacityEvent>> = BTreeMap::new();
    let mut aggregate_events = Vec::with_capacity(states.len() * 2);
    let mut node_events: BTreeMap<NodeId, Vec<CapacityEvent>> = BTreeMap::new();
    let mut gpu_events: BTreeMap<GpuAddr, Vec<CapacityEvent>> = BTreeMap::new();
    let mut class_events: BTreeMap<String, Vec<CapacityEvent>> = BTreeMap::new();
    for state in states {
        for span in prefill_capacity_spans(state) {
            prefill_events.push(PrefillCapacityEvent {
                time_s: span.start_s,
                token_delta: span.tokens as i128,
            });
            prefill_events.push(PrefillCapacityEvent {
                time_s: span.finish_s,
                token_delta: -(span.tokens as i128),
            });
            for node_id in prefill_owner_nodes(state) {
                let events = prefill_node_events.entry(node_id).or_default();
                events.push(PrefillCapacityEvent {
                    time_s: span.start_s,
                    token_delta: span.tokens as i128,
                });
                events.push(PrefillCapacityEvent {
                    time_s: span.finish_s,
                    token_delta: -(span.tokens as i128),
                });
            }
            for gpu in prefill_owner_gpus(state) {
                let events = prefill_gpu_events.entry(gpu).or_default();
                events.push(PrefillCapacityEvent {
                    time_s: span.start_s,
                    token_delta: span.tokens as i128,
                });
                events.push(PrefillCapacityEvent {
                    time_s: span.finish_s,
                    token_delta: -(span.tokens as i128),
                });
            }
            if let Some(traffic_class) = state.traffic_class.as_deref() {
                let events = prefill_class_events
                    .entry(traffic_class.to_string())
                    .or_default();
                events.push(PrefillCapacityEvent {
                    time_s: span.start_s,
                    token_delta: span.tokens as i128,
                });
                events.push(PrefillCapacityEvent {
                    time_s: span.finish_s,
                    token_delta: -(span.tokens as i128),
                });
            }
        }

        let Some((start_s, finish_s)) = kv_residency_window_s(state) else {
            continue;
        };

        let sequences = i64::from(state.batch_size.max(1));
        let resident_tokens =
            u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1));
        let aggregate_allocation = KvAllocation {
            blocks: state.kv_cache_blocks,
            allocated_tokens: state.kv_allocated_tokens,
            fragmentation_tokens: state.kv_fragmentation_tokens,
            block_table_bytes: state
                .kv_cache_blocks
                .saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
        };
        aggregate_events.push(CapacityEvent::new(
            start_s,
            sequences,
            resident_tokens as i128,
            aggregate_allocation,
            1,
        ));
        aggregate_events.push(CapacityEvent::new(
            finish_s,
            -sequences,
            -(resident_tokens as i128),
            aggregate_allocation,
            -1,
        ));
        if let Some(traffic_class) = state.traffic_class.as_deref() {
            let events = class_events.entry(traffic_class.to_string()).or_default();
            events.push(CapacityEvent::new(
                start_s,
                sequences,
                resident_tokens as i128,
                aggregate_allocation,
                1,
            ));
            events.push(CapacityEvent::new(
                finish_s,
                -sequences,
                -(resident_tokens as i128),
                aggregate_allocation,
                -1,
            ));
        }
        let owner_allocations = kv_owner_allocations(state);
        let mut node_allocations: BTreeMap<NodeId, (u64, KvAllocation)> = BTreeMap::new();
        for allocation in &owner_allocations {
            let entry = node_allocations
                .entry(allocation.owner.node_id)
                .or_insert((0, KvAllocation::default()));
            entry.0 = entry.0.saturating_add(allocation.resident_tokens);
            entry.1.blocks = entry.1.blocks.saturating_add(allocation.kv_blocks);
            entry.1.allocated_tokens = entry
                .1
                .allocated_tokens
                .saturating_add(allocation.allocated_kv_tokens);
            entry.1.fragmentation_tokens = entry
                .1
                .fragmentation_tokens
                .saturating_add(allocation.kv_fragmentation_tokens);
            entry.1.block_table_bytes = entry
                .1
                .block_table_bytes
                .saturating_add(allocation.block_table_bytes);
        }
        for (node_id, (node_tokens, node_allocation)) in node_allocations {
            let events = node_events.entry(node_id).or_default();
            events.push(CapacityEvent::new(
                start_s,
                sequences,
                node_tokens as i128,
                node_allocation,
                1,
            ));
            events.push(CapacityEvent::new(
                finish_s,
                -sequences,
                -(node_tokens as i128),
                node_allocation,
                -1,
            ));
        }
        for allocation in owner_allocations {
            let events = gpu_events.entry(allocation.owner).or_default();
            let gpu_allocation = KvAllocation {
                blocks: allocation.kv_blocks,
                allocated_tokens: allocation.allocated_kv_tokens,
                fragmentation_tokens: allocation.kv_fragmentation_tokens,
                block_table_bytes: allocation.block_table_bytes,
            };
            events.push(CapacityEvent::new(
                start_s,
                sequences,
                allocation.resident_tokens as i128,
                gpu_allocation,
                1,
            ));
            events.push(CapacityEvent::new(
                finish_s,
                -sequences,
                -(allocation.resident_tokens as i128),
                gpu_allocation,
                -1,
            ));
        }
    }

    let peak_prefill_tokens = prefill_token_peak(prefill_events);
    let aggregate_peaks = capacity_peaks(aggregate_events);
    let mut peak_decode_sequences_per_node = 0;
    let mut peak_resident_tokens_per_node = 0;
    let mut peak_kv_blocks_per_node = 0;
    let mut peak_allocated_kv_tokens_per_node = 0;
    let mut peak_kv_fragmentation_tokens_per_node = 0;
    let mut peak_kv_block_table_bytes_per_node = 0;
    let mut peak_prefill_tokens_per_node = 0;
    let mut peak_decode_sequences_per_gpu = 0;
    let mut peak_resident_tokens_per_gpu = 0;
    let mut peak_kv_blocks_per_gpu = 0;
    let mut peak_allocated_kv_tokens_per_gpu = 0;
    let mut peak_kv_fragmentation_tokens_per_gpu = 0;
    let mut peak_kv_block_table_bytes_per_gpu = 0;
    let mut peak_prefill_tokens_per_gpu = 0;
    let node_ids = node_events
        .keys()
        .copied()
        .chain(prefill_node_events.keys().copied())
        .collect::<BTreeSet<_>>();
    let nodes = node_ids
        .into_iter()
        .map(|node_id| {
            let prefill_peak =
                prefill_token_peak(prefill_node_events.remove(&node_id).unwrap_or_default());
            peak_prefill_tokens_per_node = peak_prefill_tokens_per_node.max(prefill_peak);
            let peaks = capacity_peaks(node_events.remove(&node_id).unwrap_or_default());
            peak_decode_sequences_per_node =
                peak_decode_sequences_per_node.max(peaks.decode_sequences);
            peak_resident_tokens_per_node =
                peak_resident_tokens_per_node.max(peaks.resident_tokens);
            peak_kv_blocks_per_node = peak_kv_blocks_per_node.max(peaks.kv_blocks);
            peak_allocated_kv_tokens_per_node =
                peak_allocated_kv_tokens_per_node.max(peaks.allocated_kv_tokens);
            peak_kv_fragmentation_tokens_per_node =
                peak_kv_fragmentation_tokens_per_node.max(peaks.kv_fragmentation_tokens);
            peak_kv_block_table_bytes_per_node =
                peak_kv_block_table_bytes_per_node.max(peaks.kv_block_table_bytes);
            ServingNodeCapacityObservation {
                node_id,
                peak_prefill_tokens: prefill_peak,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                decode_sequence_utilization: traffic
                    .max_decode_sequences_per_node
                    .map(|capacity| f64::from(peaks.decode_sequences) / f64::from(capacity))
                    .unwrap_or(0.0),
                resident_token_utilization: traffic
                    .max_resident_tokens_per_node
                    .map(|capacity| peaks.resident_tokens as f64 / capacity as f64)
                    .unwrap_or(0.0),
                kv_block_utilization: traffic
                    .max_kv_blocks_per_node
                    .map(|capacity| peaks.kv_blocks as f64 / capacity as f64)
                    .unwrap_or(0.0),
            }
        })
        .collect();
    let gpu_ids = gpu_events
        .keys()
        .copied()
        .chain(prefill_gpu_events.keys().copied())
        .collect::<BTreeSet<_>>();
    let gpus = gpu_ids
        .into_iter()
        .map(|gpu| {
            let prefill_peak =
                prefill_token_peak(prefill_gpu_events.remove(&gpu).unwrap_or_default());
            peak_prefill_tokens_per_gpu = peak_prefill_tokens_per_gpu.max(prefill_peak);
            let peaks = capacity_peaks(gpu_events.remove(&gpu).unwrap_or_default());
            peak_decode_sequences_per_gpu =
                peak_decode_sequences_per_gpu.max(peaks.decode_sequences);
            peak_resident_tokens_per_gpu = peak_resident_tokens_per_gpu.max(peaks.resident_tokens);
            peak_kv_blocks_per_gpu = peak_kv_blocks_per_gpu.max(peaks.kv_blocks);
            peak_allocated_kv_tokens_per_gpu =
                peak_allocated_kv_tokens_per_gpu.max(peaks.allocated_kv_tokens);
            peak_kv_fragmentation_tokens_per_gpu =
                peak_kv_fragmentation_tokens_per_gpu.max(peaks.kv_fragmentation_tokens);
            peak_kv_block_table_bytes_per_gpu =
                peak_kv_block_table_bytes_per_gpu.max(peaks.kv_block_table_bytes);
            ServingGpuCapacityObservation {
                node_id: gpu.node_id,
                local_gpu_id: gpu.local_gpu_id,
                peak_prefill_tokens: prefill_peak,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                decode_sequence_utilization: traffic
                    .max_decode_sequences_per_gpu
                    .map(|capacity| f64::from(peaks.decode_sequences) / f64::from(capacity))
                    .unwrap_or(0.0),
                resident_token_utilization: traffic
                    .max_resident_tokens_per_gpu
                    .map(|capacity| peaks.resident_tokens as f64 / capacity as f64)
                    .unwrap_or(0.0),
                kv_block_utilization: traffic
                    .max_kv_blocks_per_gpu
                    .map(|capacity| peaks.kv_blocks as f64 / capacity as f64)
                    .unwrap_or(0.0),
            }
        })
        .collect();
    let traffic_classes: Vec<ServingTrafficClassCapacityObservation> = traffic
        .traffic_classes
        .iter()
        .map(|class| {
            let prefill_peak =
                prefill_token_peak(prefill_class_events.remove(&class.name).unwrap_or_default());
            let peaks = capacity_peaks(class_events.remove(&class.name).unwrap_or_default());
            ServingTrafficClassCapacityObservation {
                name: class.name.clone(),
                group: class.group.clone(),
                key: class.key.clone(),
                max_prefill_tokens: class.max_prefill_tokens,
                max_decode_sequences: class.max_decode_sequences,
                max_resident_tokens: class.max_resident_tokens,
                max_kv_blocks: class.max_kv_blocks,
                peak_prefill_tokens: prefill_peak,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                prefill_token_utilization: class
                    .max_prefill_tokens
                    .map(|capacity| prefill_peak as f64 / capacity as f64)
                    .unwrap_or(0.0),
                decode_sequence_utilization: class
                    .max_decode_sequences
                    .map(|capacity| f64::from(peaks.decode_sequences) / f64::from(capacity))
                    .unwrap_or(0.0),
                resident_token_utilization: class
                    .max_resident_tokens
                    .map(|capacity| peaks.resident_tokens as f64 / capacity as f64)
                    .unwrap_or(0.0),
                kv_block_utilization: class
                    .max_kv_blocks
                    .map(|capacity| peaks.kv_blocks as f64 / capacity as f64)
                    .unwrap_or(0.0),
            }
        })
        .collect();
    CapacityProfile {
        peak_prefill_tokens,
        peak_prefill_tokens_per_node,
        peak_prefill_tokens_per_gpu,
        peak_decode_sequences: aggregate_peaks.decode_sequences,
        peak_resident_tokens: aggregate_peaks.resident_tokens,
        peak_decode_sequences_per_node,
        peak_resident_tokens_per_node,
        peak_decode_sequences_per_gpu,
        peak_resident_tokens_per_gpu,
        peak_kv_blocks: aggregate_peaks.kv_blocks,
        peak_allocated_kv_tokens: aggregate_peaks.allocated_kv_tokens,
        peak_kv_fragmentation_tokens: aggregate_peaks.kv_fragmentation_tokens,
        peak_kv_block_table_bytes: aggregate_peaks.kv_block_table_bytes,
        peak_kv_blocks_per_node,
        peak_allocated_kv_tokens_per_node,
        peak_kv_fragmentation_tokens_per_node,
        peak_kv_block_table_bytes_per_node,
        peak_kv_blocks_per_gpu,
        peak_allocated_kv_tokens_per_gpu,
        peak_kv_fragmentation_tokens_per_gpu,
        peak_kv_block_table_bytes_per_gpu,
        decode_sequence_utilization: traffic
            .max_decode_sequences
            .map(|capacity| f64::from(aggregate_peaks.decode_sequences) / f64::from(capacity))
            .unwrap_or(0.0),
        resident_token_utilization: traffic
            .max_resident_tokens
            .map(|capacity| aggregate_peaks.resident_tokens as f64 / capacity as f64)
            .unwrap_or(0.0),
        kv_block_utilization: traffic
            .max_kv_blocks
            .map(|capacity| aggregate_peaks.kv_blocks as f64 / capacity as f64)
            .unwrap_or(0.0),
        decode_sequence_per_node_utilization: traffic
            .max_decode_sequences_per_node
            .map(|capacity| f64::from(peak_decode_sequences_per_node) / f64::from(capacity))
            .unwrap_or(0.0),
        resident_token_per_node_utilization: traffic
            .max_resident_tokens_per_node
            .map(|capacity| peak_resident_tokens_per_node as f64 / capacity as f64)
            .unwrap_or(0.0),
        kv_block_per_node_utilization: traffic
            .max_kv_blocks_per_node
            .map(|capacity| peak_kv_blocks_per_node as f64 / capacity as f64)
            .unwrap_or(0.0),
        decode_sequence_per_gpu_utilization: traffic
            .max_decode_sequences_per_gpu
            .map(|capacity| f64::from(peak_decode_sequences_per_gpu) / f64::from(capacity))
            .unwrap_or(0.0),
        resident_token_per_gpu_utilization: traffic
            .max_resident_tokens_per_gpu
            .map(|capacity| peak_resident_tokens_per_gpu as f64 / capacity as f64)
            .unwrap_or(0.0),
        kv_block_per_gpu_utilization: traffic
            .max_kv_blocks_per_gpu
            .map(|capacity| peak_kv_blocks_per_gpu as f64 / capacity as f64)
            .unwrap_or(0.0),
        nodes,
        gpus,
        traffic_classes,
    }
}

pub(super) fn capacity_peaks(mut events: Vec<CapacityEvent>) -> CapacityPeaks {
    events.sort_by(|left, right| left.time_s.total_cmp(&right.time_s));

    let mut active_sequences = 0_i64;
    let mut resident_tokens = 0_i128;
    let mut kv_blocks = 0_i128;
    let mut allocated_kv_tokens = 0_i128;
    let mut kv_fragmentation_tokens = 0_i128;
    let mut kv_block_table_bytes = 0_i128;
    let mut peaks = CapacityPeaks::default();
    let mut idx = 0;

    while idx < events.len() {
        let event_time_s = events[idx].time_s;
        let mut sequence_delta = 0_i64;
        let mut token_delta = 0_i128;
        let mut block_delta = 0_i128;
        let mut allocated_delta = 0_i128;
        let mut fragmentation_delta = 0_i128;
        let mut block_table_delta = 0_i128;
        while idx < events.len() && events[idx].time_s.total_cmp(&event_time_s).is_eq() {
            sequence_delta += events[idx].sequence_delta;
            token_delta += events[idx].resident_token_delta;
            block_delta += events[idx].kv_block_delta;
            allocated_delta += events[idx].allocated_token_delta;
            fragmentation_delta += events[idx].fragmentation_token_delta;
            block_table_delta += events[idx].block_table_byte_delta;
            idx += 1;
        }

        active_sequences = (active_sequences + sequence_delta).max(0);
        resident_tokens = (resident_tokens + token_delta).max(0);
        kv_blocks = (kv_blocks + block_delta).max(0);
        allocated_kv_tokens = (allocated_kv_tokens + allocated_delta).max(0);
        kv_fragmentation_tokens = (kv_fragmentation_tokens + fragmentation_delta).max(0);
        kv_block_table_bytes = (kv_block_table_bytes + block_table_delta).max(0);
        peaks.decode_sequences = peaks.decode_sequences.max(active_sequences as u32);
        peaks.resident_tokens = peaks.resident_tokens.max(resident_tokens as u64);
        peaks.kv_blocks = peaks.kv_blocks.max(kv_blocks as u64);
        peaks.allocated_kv_tokens = peaks.allocated_kv_tokens.max(allocated_kv_tokens as u64);
        peaks.kv_fragmentation_tokens = peaks
            .kv_fragmentation_tokens
            .max(kv_fragmentation_tokens as u64);
        peaks.kv_block_table_bytes = peaks.kv_block_table_bytes.max(kv_block_table_bytes as u64);
    }

    peaks
}

fn prefill_token_peak(mut events: Vec<PrefillCapacityEvent>) -> u64 {
    events.sort_by(|left, right| left.time_s.total_cmp(&right.time_s));

    let mut active_tokens = 0_i128;
    let mut peak_tokens = 0_u64;
    let mut idx = 0;

    while idx < events.len() {
        let event_time_s = events[idx].time_s;
        let mut token_delta = 0_i128;
        while idx < events.len() && events[idx].time_s.total_cmp(&event_time_s).is_eq() {
            token_delta += events[idx].token_delta;
            idx += 1;
        }

        active_tokens = (active_tokens + token_delta).max(0);
        peak_tokens = peak_tokens.max(active_tokens as u64);
    }

    peak_tokens
}

pub(super) fn capacity_rejections(
    metrics: &ServingMetrics,
    traffic: &ServingTraffic,
) -> Vec<CapacityRejection> {
    let mut rejections = Vec::new();
    if traffic.decode_capacity_policy == ServingDecodeCapacityPolicy::RequestReject {
        return rejections;
    }

    if let Some(max_prefill_tokens) = traffic.max_prefill_tokens
        && metrics.peak_prefill_tokens > max_prefill_tokens
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "prefill capacity exceeded: peak_prefill_tokens {} > max_prefill_tokens {}",
                metrics.peak_prefill_tokens, max_prefill_tokens
            ),
            bottleneck: "prefill token capacity".to_string(),
            resource: "prefill_tokens".to_string(),
            code: "prefill_capacity_exceeded".to_string(),
            observed: metrics.peak_prefill_tokens as f64,
            limit: max_prefill_tokens as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_prefill_tokens_per_node) = traffic.max_prefill_tokens_per_node
        && metrics.peak_prefill_tokens_per_node > max_prefill_tokens_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node prefill capacity exceeded: peak_prefill_tokens_per_node {} > max_prefill_tokens_per_node {}",
                metrics.peak_prefill_tokens_per_node, max_prefill_tokens_per_node
            ),
            bottleneck: "per-node prefill token capacity".to_string(),
            resource: "prefill_tokens_per_node".to_string(),
            code: "prefill_capacity_per_node_exceeded".to_string(),
            observed: metrics.peak_prefill_tokens_per_node as f64,
            limit: max_prefill_tokens_per_node as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_prefill_tokens_per_gpu) = traffic.max_prefill_tokens_per_gpu
        && metrics.peak_prefill_tokens_per_gpu > max_prefill_tokens_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU prefill capacity exceeded: peak_prefill_tokens_per_gpu {} > max_prefill_tokens_per_gpu {}",
                metrics.peak_prefill_tokens_per_gpu, max_prefill_tokens_per_gpu
            ),
            bottleneck: "per-GPU prefill token capacity".to_string(),
            resource: "prefill_tokens_per_gpu".to_string(),
            code: "prefill_capacity_per_gpu_exceeded".to_string(),
            observed: metrics.peak_prefill_tokens_per_gpu as f64,
            limit: max_prefill_tokens_per_gpu as f64,
            unit: "tokens".to_string(),
        });
    }

    if let Some(max_decode_sequences) = traffic.max_decode_sequences
        && metrics.peak_decode_sequences > max_decode_sequences
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "decode capacity exceeded: peak_decode_sequences {} > max_decode_sequences {}",
                metrics.peak_decode_sequences, max_decode_sequences
            ),
            bottleneck: "decode sequence capacity".to_string(),
            resource: "decode_sequences".to_string(),
            code: "decode_capacity_exceeded".to_string(),
            observed: f64::from(metrics.peak_decode_sequences),
            limit: f64::from(max_decode_sequences),
            unit: "sequences".to_string(),
        });
    }
    if let Some(max_resident_tokens) = traffic.max_resident_tokens
        && metrics.peak_resident_tokens > max_resident_tokens
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "KV residency capacity exceeded: peak_resident_tokens {} > max_resident_tokens {}",
                metrics.peak_resident_tokens, max_resident_tokens
            ),
            bottleneck: "KV residency capacity".to_string(),
            resource: "resident_tokens".to_string(),
            code: "kv_residency_capacity_exceeded".to_string(),
            observed: metrics.peak_resident_tokens as f64,
            limit: max_resident_tokens as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_kv_blocks) = traffic.max_kv_blocks
        && metrics.peak_kv_blocks > max_kv_blocks
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "KV block capacity exceeded: peak_kv_blocks {} > max_kv_blocks {}",
                metrics.peak_kv_blocks, max_kv_blocks
            ),
            bottleneck: "KV block capacity".to_string(),
            resource: "kv_blocks".to_string(),
            code: "kv_block_capacity_exceeded".to_string(),
            observed: metrics.peak_kv_blocks as f64,
            limit: max_kv_blocks as f64,
            unit: "blocks".to_string(),
        });
    }
    if let Some(max_decode_sequences_per_node) = traffic.max_decode_sequences_per_node
        && metrics.peak_decode_sequences_per_node > max_decode_sequences_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node decode capacity exceeded: peak_decode_sequences_per_node {} > max_decode_sequences_per_node {}",
                metrics.peak_decode_sequences_per_node, max_decode_sequences_per_node
            ),
            bottleneck: "per-node decode sequence capacity".to_string(),
            resource: "decode_sequences_per_node".to_string(),
            code: "decode_capacity_per_node_exceeded".to_string(),
            observed: f64::from(metrics.peak_decode_sequences_per_node),
            limit: f64::from(max_decode_sequences_per_node),
            unit: "sequences".to_string(),
        });
    }
    if let Some(max_resident_tokens_per_node) = traffic.max_resident_tokens_per_node
        && metrics.peak_resident_tokens_per_node > max_resident_tokens_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node KV residency capacity exceeded: peak_resident_tokens_per_node {} > max_resident_tokens_per_node {}",
                metrics.peak_resident_tokens_per_node, max_resident_tokens_per_node
            ),
            bottleneck: "per-node KV residency capacity".to_string(),
            resource: "resident_tokens_per_node".to_string(),
            code: "kv_residency_capacity_per_node_exceeded".to_string(),
            observed: metrics.peak_resident_tokens_per_node as f64,
            limit: max_resident_tokens_per_node as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_kv_blocks_per_node) = traffic.max_kv_blocks_per_node
        && metrics.peak_kv_blocks_per_node > max_kv_blocks_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node KV block capacity exceeded: peak_kv_blocks_per_node {} > max_kv_blocks_per_node {}",
                metrics.peak_kv_blocks_per_node, max_kv_blocks_per_node
            ),
            bottleneck: "per-node KV block capacity".to_string(),
            resource: "kv_blocks_per_node".to_string(),
            code: "kv_block_capacity_per_node_exceeded".to_string(),
            observed: metrics.peak_kv_blocks_per_node as f64,
            limit: max_kv_blocks_per_node as f64,
            unit: "blocks".to_string(),
        });
    }
    if let Some(max_decode_sequences_per_gpu) = traffic.max_decode_sequences_per_gpu
        && metrics.peak_decode_sequences_per_gpu > max_decode_sequences_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU decode capacity exceeded: peak_decode_sequences_per_gpu {} > max_decode_sequences_per_gpu {}",
                metrics.peak_decode_sequences_per_gpu, max_decode_sequences_per_gpu
            ),
            bottleneck: "per-GPU decode sequence capacity".to_string(),
            resource: "decode_sequences_per_gpu".to_string(),
            code: "decode_capacity_per_gpu_exceeded".to_string(),
            observed: f64::from(metrics.peak_decode_sequences_per_gpu),
            limit: f64::from(max_decode_sequences_per_gpu),
            unit: "sequences".to_string(),
        });
    }
    if let Some(max_resident_tokens_per_gpu) = traffic.max_resident_tokens_per_gpu
        && metrics.peak_resident_tokens_per_gpu > max_resident_tokens_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU KV residency capacity exceeded: peak_resident_tokens_per_gpu {} > max_resident_tokens_per_gpu {}",
                metrics.peak_resident_tokens_per_gpu, max_resident_tokens_per_gpu
            ),
            bottleneck: "per-GPU KV residency capacity".to_string(),
            resource: "resident_tokens_per_gpu".to_string(),
            code: "kv_residency_capacity_per_gpu_exceeded".to_string(),
            observed: metrics.peak_resident_tokens_per_gpu as f64,
            limit: max_resident_tokens_per_gpu as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_kv_blocks_per_gpu) = traffic.max_kv_blocks_per_gpu
        && metrics.peak_kv_blocks_per_gpu > max_kv_blocks_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU KV block capacity exceeded: peak_kv_blocks_per_gpu {} > max_kv_blocks_per_gpu {}",
                metrics.peak_kv_blocks_per_gpu, max_kv_blocks_per_gpu
            ),
            bottleneck: "per-GPU KV block capacity".to_string(),
            resource: "kv_blocks_per_gpu".to_string(),
            code: "kv_block_capacity_per_gpu_exceeded".to_string(),
            observed: metrics.peak_kv_blocks_per_gpu as f64,
            limit: max_kv_blocks_per_gpu as f64,
            unit: "blocks".to_string(),
        });
    }
    rejections
}
