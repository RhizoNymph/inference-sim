use super::*;

#[derive(Default)]
struct ServingHardwareCapacity {
    node_count: u32,
    gpu_count: u32,
    hbm_gb: f64,
    hbm_bandwidth_gb_s: f64,
    peak_f16_tflops: f64,
    peak_f8_tflops: Option<f64>,
    effective_peak_tflops: f64,
}

pub(super) fn serving_hardware_footprint(
    cluster: &Cluster,
    model: &ModelSpec,
    prefill_placement: &RankPlacement,
    decode_placement: &RankPlacement,
    throughput_tokens_per_s: f64,
) -> ServingHardwareFootprint {
    let prefill_gpus = placement_gpu_set(prefill_placement);
    let decode_gpus = placement_gpu_set(decode_placement);
    let unique_gpus = prefill_gpus
        .union(&decode_gpus)
        .copied()
        .collect::<BTreeSet<_>>();
    let prefill_nodes = gpu_node_set(&prefill_gpus);
    let decode_nodes = gpu_node_set(&decode_gpus);

    let aggregate = serving_hardware_capacity(cluster, model, &unique_gpus);
    let prefill = serving_hardware_capacity(cluster, model, &prefill_gpus);
    let decode = serving_hardware_capacity(cluster, model, &decode_gpus);

    ServingHardwareFootprint {
        unique_node_count: aggregate.node_count,
        unique_gpu_count: aggregate.gpu_count,
        prefill_node_count: prefill.node_count,
        prefill_gpu_count: prefill.gpu_count,
        decode_node_count: decode.node_count,
        decode_gpu_count: decode.gpu_count,
        shared_node_count: prefill_nodes.intersection(&decode_nodes).count() as u32,
        shared_gpu_count: prefill_gpus.intersection(&decode_gpus).count() as u32,
        aggregate_hbm_gb: aggregate.hbm_gb,
        prefill_hbm_gb: prefill.hbm_gb,
        decode_hbm_gb: decode.hbm_gb,
        aggregate_hbm_bandwidth_gb_s: aggregate.hbm_bandwidth_gb_s,
        prefill_hbm_bandwidth_gb_s: prefill.hbm_bandwidth_gb_s,
        decode_hbm_bandwidth_gb_s: decode.hbm_bandwidth_gb_s,
        aggregate_peak_f16_tflops: aggregate.peak_f16_tflops,
        prefill_peak_f16_tflops: prefill.peak_f16_tflops,
        decode_peak_f16_tflops: decode.peak_f16_tflops,
        aggregate_peak_f8_tflops: aggregate.peak_f8_tflops,
        prefill_peak_f8_tflops: prefill.peak_f8_tflops,
        decode_peak_f8_tflops: decode.peak_f8_tflops,
        aggregate_gpu_types: serving_gpu_type_counts(cluster, &unique_gpus),
        prefill_gpu_types: serving_gpu_type_counts(cluster, &prefill_gpus),
        decode_gpu_types: serving_gpu_type_counts(cluster, &decode_gpus),
        aggregate_gpu_label_counts: serving_gpu_label_counts(cluster, &unique_gpus),
        prefill_gpu_label_counts: serving_gpu_label_counts(cluster, &prefill_gpus),
        decode_gpu_label_counts: serving_gpu_label_counts(cluster, &decode_gpus),
        aggregate_effective_peak_tflops: aggregate.effective_peak_tflops,
        prefill_effective_peak_tflops: prefill.effective_peak_tflops,
        decode_effective_peak_tflops: decode.effective_peak_tflops,
        throughput_tokens_per_s_per_gpu: safe_ratio(
            throughput_tokens_per_s,
            f64::from(aggregate.gpu_count),
        ),
        throughput_tokens_per_s_per_effective_peak_tflop: safe_ratio(
            throughput_tokens_per_s,
            aggregate.effective_peak_tflops,
        ),
        throughput_tokens_per_s_per_hbm_gb: safe_ratio(throughput_tokens_per_s, aggregate.hbm_gb),
    }
}

fn serving_gpu_type_counts(
    cluster: &Cluster,
    gpus: &BTreeSet<GpuAddr>,
) -> Vec<ServingGpuTypeCount> {
    let mut counts = BTreeMap::new();
    for addr in gpus {
        if let Some(profile) = cluster.gpu_profile(*addr) {
            *counts.entry(profile.label.to_string()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|(gpu, count)| ServingGpuTypeCount { gpu, count })
        .collect()
}

fn serving_gpu_label_counts(
    cluster: &Cluster,
    gpus: &BTreeSet<GpuAddr>,
) -> Vec<ServingGpuLabelCount> {
    let mut counts = BTreeMap::new();
    for addr in gpus {
        let Some(node) = cluster.node(addr.node_id) else {
            continue;
        };
        let Some(labels) = node.gpu_labels(addr.local_gpu_id) else {
            continue;
        };
        for label in labels {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|(label, count)| ServingGpuLabelCount { label, count })
        .collect()
}

pub(super) fn serving_cost_estimate(
    cluster: &Cluster,
    prefill_placement: &RankPlacement,
    decode_placement: &RankPlacement,
    metrics: &ServingMetrics,
    cost_model: &ServingCostModel,
) -> ServingCostEstimate {
    let prefill_gpus = placement_gpu_set(prefill_placement);
    let decode_gpus = placement_gpu_set(decode_placement);
    let unique_gpus = prefill_gpus
        .union(&decode_gpus)
        .copied()
        .collect::<BTreeSet<_>>();
    let unique_nodes = gpu_node_set(&unique_gpus);
    let modeled_duration_s = (metrics.scheduled_makespan_s.is_finite()
        && metrics.scheduled_makespan_s > 0.0)
        .then_some(metrics.scheduled_makespan_s);
    let modeled_gpu_count = unique_gpus.len().min(u32::MAX as usize) as u32;
    let modeled_node_count = unique_nodes.len().min(u32::MAX as usize) as u32;
    let gpu_hours =
        modeled_duration_s.map(|duration_s| duration_s / 3600.0 * f64::from(modeled_gpu_count));
    let node_hours =
        modeled_duration_s.map(|duration_s| duration_s / 3600.0 * f64::from(modeled_node_count));

    let gpu_rate_sum = sum_gpu_rate(cluster, &unique_gpus, cost_model, |rate| rate.gpu_hour_usd);
    let gpu_hour_cost_usd = modeled_duration_s
        .zip(gpu_rate_sum)
        .map(|(duration_s, rate_sum)| duration_s / 3600.0 * rate_sum);
    let node_hour_cost_usd = node_hours
        .zip(cost_model.node_hour_usd)
        .map(|(hours, rate)| hours * rate);
    let gpu_power_watts = sum_gpu_rate(cluster, &unique_gpus, cost_model, |rate| rate.watts);
    let node_power_watts = cost_model
        .node_watts
        .map(|watts| watts * f64::from(modeled_node_count));
    let average_power_watts = sum_optional_values([gpu_power_watts, node_power_watts]);
    let energy_kwh = modeled_duration_s
        .zip(average_power_watts)
        .map(|(duration_s, watts)| watts / 1000.0 * duration_s / 3600.0);
    let energy_cost_usd = energy_kwh
        .zip(cost_model.kwh_usd)
        .map(|(kwh, rate)| kwh * rate);
    let total_cost_usd =
        sum_optional_values([gpu_hour_cost_usd, node_hour_cost_usd, energy_cost_usd]);
    let cost_per_1k_output_tokens_usd = total_cost_usd.and_then(|cost| {
        (metrics.decode_iterations > 0).then(|| cost / metrics.decode_iterations as f64 * 1000.0)
    });
    let cost_per_1k_requests_usd = total_cost_usd.and_then(|cost| {
        (metrics.completed_requests > 0)
            .then(|| cost / f64::from(metrics.completed_requests) * 1000.0)
    });

    ServingCostEstimate {
        modeled_duration_s,
        modeled_gpu_count,
        modeled_node_count,
        gpu_hours,
        node_hours,
        gpu_hour_cost_usd,
        node_hour_cost_usd,
        average_power_watts,
        energy_kwh,
        energy_cost_usd,
        total_cost_usd,
        cost_per_1k_output_tokens_usd,
        cost_per_1k_requests_usd,
    }
}

fn sum_gpu_rate<F>(
    cluster: &Cluster,
    gpus: &BTreeSet<GpuAddr>,
    cost_model: &ServingCostModel,
    selector: F,
) -> Option<f64>
where
    F: Fn(&ServingGpuCostRate) -> Option<f64>,
{
    let mut total = 0.0;
    for gpu in gpus {
        let profile = cluster.gpu_profile(*gpu)?;
        let value = cost_model
            .gpu_rates
            .iter()
            .find(|rate| gpu_label_matches(&rate.gpu_label, profile.label))
            .and_then(&selector)
            .or_else(|| {
                let default_rate = ServingGpuCostRate {
                    gpu_label: "default".to_string(),
                    gpu_hour_usd: cost_model.default_gpu_hour_usd,
                    watts: cost_model.default_gpu_watts,
                };
                selector(&default_rate)
            })?;
        total += value;
    }
    Some(total)
}

fn gpu_label_matches(configured: &str, profile_label: &str) -> bool {
    normalize_hardware_label(configured) == normalize_hardware_label(profile_label)
}

fn normalize_hardware_label(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn sum_optional_values<const N: usize>(values: [Option<f64>; N]) -> Option<f64> {
    let mut saw_value = false;
    let mut total = 0.0;
    for value in values.into_iter().flatten() {
        saw_value = true;
        total += value;
    }
    saw_value.then_some(total)
}

fn placement_gpu_set(placement: &RankPlacement) -> BTreeSet<GpuAddr> {
    placement.rank_to_gpu.iter().copied().collect()
}

fn gpu_node_set(gpus: &BTreeSet<GpuAddr>) -> BTreeSet<NodeId> {
    gpus.iter().map(|gpu| gpu.node_id).collect()
}

fn serving_hardware_capacity(
    cluster: &Cluster,
    model: &ModelSpec,
    gpus: &BTreeSet<GpuAddr>,
) -> ServingHardwareCapacity {
    let mut capacity = ServingHardwareCapacity {
        node_count: gpu_node_set(gpus).len() as u32,
        gpu_count: gpus.len() as u32,
        peak_f8_tflops: Some(0.0),
        ..ServingHardwareCapacity::default()
    };

    for gpu in gpus {
        let Some(profile) = cluster.gpu_profile(*gpu) else {
            capacity.peak_f8_tflops = None;
            continue;
        };
        capacity.hbm_gb += profile.hbm_size.as_gigabytes();
        capacity.hbm_bandwidth_gb_s += profile.hbm_bandwidth.as_gigabytes_per_sec();
        capacity.peak_f16_tflops += profile.peak_f16_flops;
        capacity.effective_peak_tflops += effective_gpu_peak_tflops(&profile, model.dtype);
        match (capacity.peak_f8_tflops, profile.peak_f8_flops) {
            (Some(total), Some(peak_f8_tflops)) => {
                capacity.peak_f8_tflops = Some(total + peak_f8_tflops);
            }
            _ => {
                capacity.peak_f8_tflops = None;
            }
        }
    }

    capacity
}

fn effective_gpu_peak_tflops(profile: &crate::types::gpu::GpuProfile, dtype: DType) -> f64 {
    match dtype {
        DType::Fp8 => profile.peak_f8_flops.unwrap_or(0.0),
        DType::Int8 => profile.peak_f8_flops.unwrap_or(profile.peak_f16_flops),
        DType::Fp16 | DType::Bf16 => profile.peak_f16_flops,
    }
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator.is_finite() && denominator.is_finite() && denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}
