use super::*;

pub(super) fn memory_headroom(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    phase: InferencePhase,
    score: &ScoredParallelismConfig,
    kv_block_tokens: u32,
    calibration: SimulationCalibration,
) -> ServingMemoryHeadroom {
    let components = serving_memory_components(
        model,
        request,
        phase,
        score.config,
        kv_block_tokens,
        calibration,
    );
    let estimated_per_gpu_gb = components.total_gb;
    let mut min_hbm_per_gpu_gb = f64::INFINITY;
    let mut limiting_gpu = None;
    for addr in &score.placement.rank_to_gpu {
        let Some(profile) = cluster.gpu_profile(*addr) else {
            continue;
        };
        let hbm_gb = profile.hbm_size.as_gigabytes();
        if hbm_gb < min_hbm_per_gpu_gb {
            min_hbm_per_gpu_gb = hbm_gb;
            limiting_gpu = Some(*addr);
        }
    }

    if !estimated_per_gpu_gb.is_finite()
        || !min_hbm_per_gpu_gb.is_finite()
        || min_hbm_per_gpu_gb <= 0.0
    {
        return ServingMemoryHeadroom::unavailable();
    }

    let headroom_gb = min_hbm_per_gpu_gb - estimated_per_gpu_gb;
    ServingMemoryHeadroom {
        estimated_per_gpu_gb,
        min_hbm_per_gpu_gb,
        limiting_gpu,
        headroom_gb,
        headroom_fraction: headroom_gb / min_hbm_per_gpu_gb,
        components,
    }
}

#[derive(Copy, Clone, Debug)]
struct MemoryPressureSpan {
    start_s: f64,
    finish_s: f64,
    active_tokens: u64,
    kv_blocks: u64,
}

pub(super) fn memory_pressure_observations(
    observations: &[ServingRequestObservation],
    prefill_memory: &ServingMemoryHeadroom,
    decode_memory: &ServingMemoryHeadroom,
) -> Vec<ServingMemoryPressureObservation> {
    let mut pressure = Vec::new();
    push_memory_pressure_phase(
        &mut pressure,
        "prefill",
        "active_request_component_scaled_prefill_memory",
        prefill_memory,
        true,
        observations
            .iter()
            .filter_map(|observation| {
                memory_pressure_span(
                    observation.prefill_start_s,
                    observation.prefill_finish_s,
                    u64::from(observation.batch_size)
                        .saturating_mul(u64::from(observation.effective_prefill_tokens.max(1))),
                    0,
                )
            })
            .collect(),
    );

    let kv_memory = kv_transfer_pressure_memory(prefill_memory, decode_memory);
    push_memory_pressure_phase(
        &mut pressure,
        "kv_transfer",
        "active_request_component_scaled_kv_transfer_memory",
        &kv_memory,
        false,
        observations
            .iter()
            .filter_map(|observation| {
                memory_pressure_span(
                    observation.kv_start_s,
                    observation.kv_finish_s,
                    observation.kv_allocated_tokens,
                    observation.kv_cache_blocks,
                )
            })
            .collect(),
    );

    push_memory_pressure_phase(
        &mut pressure,
        "decode",
        "active_request_component_scaled_decode_memory",
        decode_memory,
        true,
        observations
            .iter()
            .filter_map(|observation| {
                memory_pressure_span(
                    observation.first_decode_start_s,
                    observation.last_decode_finish_s,
                    observation.kv_allocated_tokens,
                    observation.kv_cache_blocks,
                )
            })
            .collect(),
    );

    pressure
}

pub(super) fn memory_pressure_rejection(
    observations: &[ServingMemoryPressureObservation],
    max_memory_pressure_fraction: Option<f64>,
) -> Option<ServingRejection> {
    let limit = max_memory_pressure_fraction?;
    let peak = observations
        .iter()
        .filter(|observation| observation.capacity_used_fraction.is_finite())
        .max_by(|left, right| {
            left.capacity_used_fraction
                .total_cmp(&right.capacity_used_fraction)
        })?;

    if peak.capacity_used_fraction <= limit + 1e-12 {
        return None;
    }

    Some(ServingRejection {
        phase: peak.phase.clone(),
        category: "memory".to_string(),
        resource: "memory_pressure".to_string(),
        code: "memory_pressure_fraction_exceeded".to_string(),
        observed: Some(peak.capacity_used_fraction),
        limit: Some(limit),
        unit: Some("fraction".to_string()),
        remediation: Some(
            "increase tensor/pipeline sharding, use GPUs with more HBM, reduce batch/sequence/KV residency, or raise the configured max_memory_pressure_fraction"
                .to_string(),
        ),
        message: format!(
            "{} memory pressure {:.3} exceeds configured limit {:.3}",
            peak.phase, peak.capacity_used_fraction, limit
        ),
    })
}

pub(super) fn gpu_footprint_rejection(
    footprint: &ServingHardwareFootprint,
    max_unique_gpus: Option<u32>,
) -> Option<ServingRejection> {
    let limit = max_unique_gpus?;
    if footprint.unique_gpu_count <= limit {
        return None;
    }

    Some(ServingRejection {
        phase: "placement".to_string(),
        category: "capacity".to_string(),
        resource: "unique_gpus".to_string(),
        code: "unique_gpu_footprint_exceeded".to_string(),
        observed: Some(f64::from(footprint.unique_gpu_count)),
        limit: Some(f64::from(limit)),
        unit: Some("gpus".to_string()),
        remediation: Some(
            "reduce prefill/decode ranks, choose smaller pools, allow colocated placement, or raise serving.max_unique_gpus"
                .to_string(),
        ),
        message: format!(
            "serving candidate uses {} unique GPUs, exceeding configured max_unique_gpus {}",
            footprint.unique_gpu_count, limit
        ),
    })
}

pub(super) fn throughput_floor_rejection(
    metrics: &ServingMetrics,
    min_throughput_tokens_per_s: Option<f64>,
) -> Option<ServingRejection> {
    let limit = min_throughput_tokens_per_s?;
    if metrics.throughput_tokens_per_s.is_finite() && metrics.throughput_tokens_per_s >= limit {
        return None;
    }

    Some(ServingRejection {
        phase: "serving".to_string(),
        category: "throughput".to_string(),
        resource: "throughput_tokens_per_s".to_string(),
        code: "throughput_below_min".to_string(),
        observed: metrics
            .throughput_tokens_per_s
            .is_finite()
            .then_some(metrics.throughput_tokens_per_s),
        limit: Some(limit),
        unit: Some("tokens/s".to_string()),
        remediation: Some(
            "increase serving parallelism or worker capacity, reduce latency/queueing pressure, choose a faster pool, or lower serving.min_throughput_tokens_per_s"
                .to_string(),
        ),
        message: format!(
            "serving throughput {:.3} tokens/s is below configured minimum {:.3} tokens/s",
            metrics.throughput_tokens_per_s, limit
        ),
    })
}

pub(super) fn metric_ceiling_rejections(
    metrics: &ServingMetrics,
    ceilings: ServingMetricCeilings,
) -> Vec<ServingRejection> {
    if !ceilings.any() {
        return Vec::new();
    }

    [
        (
            "prefill",
            "ttft_s",
            "ttft_above_max",
            metrics.ttft_s,
            ceilings.max_ttft_s,
            "serving.max_ttft_s",
        ),
        (
            "decode",
            "tpot_s",
            "tpot_above_max",
            metrics.tpot_s,
            ceilings.max_tpot_s,
            "serving.max_tpot_s",
        ),
        (
            "decode",
            "itl_s",
            "itl_above_max",
            metrics.itl_s,
            ceilings.max_itl_s,
            "serving.max_itl_s",
        ),
        (
            "serving",
            "e2el_s",
            "e2el_above_max",
            metrics.e2el_s,
            ceilings.max_e2el_s,
            "serving.max_e2el_s",
        ),
    ]
    .into_iter()
    .filter_map(
        |(phase, resource, code, observed, limit, config_field)| -> Option<ServingRejection> {
            let limit = limit?;
            if observed.is_finite() && observed <= limit + 1e-12 {
                return None;
            }
            Some(ServingRejection {
                phase: phase.to_string(),
                category: "latency".to_string(),
                resource: resource.to_string(),
                code: code.to_string(),
                observed: observed.is_finite().then_some(observed),
                limit: Some(limit),
                unit: Some("s".to_string()),
                remediation: Some(format!(
                    "reduce queueing/service time, change prefill/decode placement or parallelism, choose a faster pool, or raise {config_field}"
                )),
                message: format!(
                    "{resource} {:.6}s exceeds configured maximum {:.6}s",
                    observed, limit
                ),
            })
        },
    )
    .collect()
}

fn memory_pressure_span(
    start_s: f64,
    finish_s: f64,
    active_tokens: u64,
    kv_blocks: u64,
) -> Option<MemoryPressureSpan> {
    if start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
        Some(MemoryPressureSpan {
            start_s,
            finish_s,
            active_tokens,
            kv_blocks,
        })
    } else {
        None
    }
}

fn kv_transfer_pressure_memory(
    prefill_memory: &ServingMemoryHeadroom,
    decode_memory: &ServingMemoryHeadroom,
) -> ServingMemoryHeadroom {
    let mut memory = *decode_memory;
    if prefill_memory.min_hbm_per_gpu_gb < memory.min_hbm_per_gpu_gb {
        memory.min_hbm_per_gpu_gb = prefill_memory.min_hbm_per_gpu_gb;
        memory.limiting_gpu = prefill_memory.limiting_gpu;
    }
    if prefill_memory.components.weights_gb > memory.components.weights_gb {
        memory.components.weights_gb = prefill_memory.components.weights_gb;
    }
    memory
}

fn push_memory_pressure_phase(
    pressure: &mut Vec<ServingMemoryPressureObservation>,
    phase: &str,
    estimate_kind: &str,
    memory: &ServingMemoryHeadroom,
    include_activation_components: bool,
    spans: Vec<MemoryPressureSpan>,
) {
    if spans.is_empty() {
        return;
    }

    let mut boundaries = spans
        .iter()
        .flat_map(|span| [span.start_s, span.finish_s])
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    boundaries.sort_by(|left, right| left.total_cmp(right));
    boundaries.dedup_by(|left, right| (*left - *right).abs() <= 1e-12);

    for window in boundaries.windows(2) {
        let start_s = window[0];
        let finish_s = window[1];
        if finish_s <= start_s {
            continue;
        }

        let active_spans = spans
            .iter()
            .filter(|span| span.start_s < finish_s - 1e-12 && span.finish_s > start_s + 1e-12)
            .collect::<Vec<_>>();
        if active_spans.is_empty() {
            continue;
        }

        let active_requests = active_spans.len().min(u32::MAX as usize) as u32;
        let active_tokens = active_spans.iter().fold(0_u64, |total, span| {
            total.saturating_add(span.active_tokens)
        });
        let kv_blocks = active_spans
            .iter()
            .fold(0_u64, |total, span| total.saturating_add(span.kv_blocks));
        let components = scaled_memory_pressure_components(
            memory.components,
            f64::from(active_requests),
            include_activation_components,
        );
        let estimated_per_gpu_gb = components.total_gb;
        let capacity_used_fraction = if estimated_per_gpu_gb.is_finite()
            && memory.min_hbm_per_gpu_gb.is_finite()
            && memory.min_hbm_per_gpu_gb > 0.0
        {
            estimated_per_gpu_gb / memory.min_hbm_per_gpu_gb
        } else {
            f64::INFINITY
        };
        let headroom_gb =
            if estimated_per_gpu_gb.is_finite() && memory.min_hbm_per_gpu_gb.is_finite() {
                memory.min_hbm_per_gpu_gb - estimated_per_gpu_gb
            } else {
                f64::INFINITY
            };

        pressure.push(ServingMemoryPressureObservation {
            phase: phase.to_string(),
            estimate_kind: estimate_kind.to_string(),
            start_s,
            finish_s,
            duration_s: finish_s - start_s,
            active_requests,
            active_tokens,
            kv_blocks,
            estimated_per_gpu_gb,
            min_hbm_per_gpu_gb: memory.min_hbm_per_gpu_gb,
            capacity_used_fraction,
            headroom_gb,
            limiting_gpu: memory.limiting_gpu,
            dominant_component: components.dominant_component(),
            components,
        });
    }
}

fn scaled_memory_pressure_components(
    base: ServingMemoryComponents,
    active_requests: f64,
    include_activation_components: bool,
) -> ServingMemoryComponents {
    let scale = active_requests.max(0.0);
    let weights_gb = base.weights_gb;
    let kv_cache_gb = base.kv_cache_gb * scale;
    let block_table_gb = base.block_table_gb * scale;
    let activations_gb = if include_activation_components {
        base.activations_gb * scale
    } else {
        0.0
    };
    let temporary_gb = if include_activation_components {
        base.temporary_gb * scale
    } else {
        0.0
    };
    let communication_gb = base.communication_gb * scale;
    let runtime_reserve_gb = base.runtime_reserve_gb * scale;
    let fragmentation_gb = base.fragmentation_gb * scale;
    let total_gb = weights_gb
        + kv_cache_gb
        + block_table_gb
        + activations_gb
        + temporary_gb
        + communication_gb
        + runtime_reserve_gb
        + fragmentation_gb;

    ServingMemoryComponents {
        weights_gb,
        kv_cache_gb,
        block_table_gb,
        activations_gb,
        temporary_gb,
        communication_gb,
        runtime_reserve_gb,
        fragmentation_gb,
        total_gb,
    }
}

fn serving_memory_components(
    model: &ModelSpec,
    request: &InferenceRequest,
    phase: InferencePhase,
    config: ParallelismConfig,
    kv_block_tokens: u32,
    calibration: SimulationCalibration,
) -> ServingMemoryComponents {
    let calibration = calibration.sanitized();
    let shard_factor =
        f64::from((config.tensor_ranks * config.pipeline_ranks * config.expert_ranks).max(1));
    let tensor_ranks = f64::from(config.tensor_ranks.max(1));

    let weights_gb = model.parameters.as_gigabytes() / shard_factor;
    let kv_cache_gb = kv_cache_gb(model, request, config);
    let block_table_gb = kv_block_table_gb(request, config, kv_block_tokens);
    let activations_gb = activation_memory_gb(model, request, phase, tensor_ranks);
    let temporary_gb = activations_gb * calibration.serving_memory_temporary_fraction;
    let communication_gb = (activations_gb
        * calibration.serving_memory_activation_communication_fraction)
        .max(weights_gb * calibration.serving_memory_weight_communication_fraction);

    let subtotal_gb = weights_gb
        + kv_cache_gb
        + block_table_gb
        + activations_gb
        + temporary_gb
        + communication_gb;
    let runtime_reserve_gb = subtotal_gb * calibration.serving_memory_runtime_reserve_fraction;
    let fragmentation_gb =
        (weights_gb + kv_cache_gb) * calibration.serving_memory_fragmentation_fraction;
    let total_gb = subtotal_gb + runtime_reserve_gb + fragmentation_gb;

    ServingMemoryComponents {
        weights_gb,
        kv_cache_gb,
        block_table_gb,
        activations_gb,
        temporary_gb,
        communication_gb,
        runtime_reserve_gb,
        fragmentation_gb,
        total_gb,
    }
}

fn kv_cache_gb(model: &ModelSpec, request: &InferenceRequest, config: ParallelismConfig) -> f64 {
    let head_dim = f64::from(model.hidden_size) / f64::from(model.attention_heads.max(1));
    let tokens = f64::from(request.max_sequence_tokens) * f64::from(request.batch_size);
    let bytes = tokens
        * f64::from(model.kv_heads)
        * head_dim
        * 2.0
        * model.kv_dtype().bytes_per_element() as f64
        * f64::from(model.layers)
        / f64::from(config.tensor_ranks.max(1));

    bytes / 1e9
}

fn kv_block_table_gb(
    request: &InferenceRequest,
    config: ParallelismConfig,
    block_tokens: u32,
) -> f64 {
    let block_tokens = u64::from(block_tokens.max(1));
    let blocks_per_sequence = u64::from(request.max_sequence_tokens.max(1)).div_ceil(block_tokens);
    let total_blocks = u64::from(request.batch_size.max(1)).saturating_mul(blocks_per_sequence);
    let local_blocks = total_blocks.div_ceil(u64::from(config.tensor_ranks.max(1)));
    local_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES) as f64 / 1e9
}

fn activation_memory_gb(
    model: &ModelSpec,
    request: &InferenceRequest,
    phase: InferencePhase,
    tensor_ranks: f64,
) -> f64 {
    let active_tokens = match phase {
        InferencePhase::Prefill => request.prompt_tokens.max(1),
        InferencePhase::Decode => 1,
        InferencePhase::EndToEnd => request.prompt_tokens.saturating_add(request.decode_tokens),
    };
    let bytes = f64::from(request.batch_size)
        * f64::from(active_tokens)
        * f64::from(model.hidden_size)
        * model.dtype.bytes_per_element() as f64
        / tensor_ranks.max(1.0);

    bytes / 1e9
}

pub(super) fn memory_headroom_rejection(
    phase: &str,
    memory: &ServingMemoryHeadroom,
) -> Option<ServingRejection> {
    if !memory.headroom_gb.is_finite() || memory.headroom_gb >= 0.0 {
        return None;
    }

    let dominant_component = memory
        .dominant_component()
        .map(|component| {
            format!(
                "; dominant_component={} {:.2} GB ({:.1}% of estimate)",
                component.name,
                component.gb,
                component.fraction_of_total * 100.0
            )
        })
        .unwrap_or_default();
    let limiting_gpu = memory
        .limiting_gpu
        .map(|addr| {
            format!(
                "; limiting_gpu=node:{} gpu:{}",
                addr.node_id, addr.local_gpu_id
            )
        })
        .unwrap_or_default();

    Some(ServingRejection {
        phase: phase.to_string(),
        category: "memory".to_string(),
        resource: "gpu_hbm".to_string(),
        code: "serving_memory_headroom_exceeded".to_string(),
        observed: Some(memory.estimated_per_gpu_gb),
        limit: Some(memory.min_hbm_per_gpu_gb),
        unit: Some("GB".to_string()),
        remediation: Some(
            "increase sharding, use GPUs with more HBM, or reduce batch/sequence/KV residency"
                .to_string(),
        ),
        message: format!(
            "{phase} serving memory estimate {:.2} GB per GPU exceeds {:.2} GB HBM{dominant_component}{limiting_gpu}",
            memory.estimated_per_gpu_gb, memory.min_hbm_per_gpu_gb
        ),
    })
}
