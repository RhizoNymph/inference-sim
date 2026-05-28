use super::*;

pub(super) fn write_serving_memory<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    memory: &ServingMemoryHeadroom,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": {{")?;
    writeln!(
        writer,
        "{indent}    \"estimated_per_gpu_gb\": {},",
        json_optional_f64(memory.estimated_per_gpu_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"min_hbm_per_gpu_gb\": {},",
        json_optional_f64(memory.min_hbm_per_gpu_gb)
    )?;
    write_memory_limiter_gpu(writer, indent, memory.limiting_gpu, true)?;
    writeln!(
        writer,
        "{indent}    \"capacity_used_fraction\": {},",
        json_optional_f64(memory.capacity_used_fraction())
    )?;
    writeln!(
        writer,
        "{indent}    \"headroom_gb\": {},",
        json_optional_f64(memory.headroom_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"headroom_fraction\": {},",
        json_optional_f64(memory.headroom_fraction)
    )?;
    write_memory_dominant_component(writer, indent, memory, true)?;
    write_memory_component_fractions(writer, indent, memory, true)?;
    writeln!(writer, "{indent}    \"components\": {{")?;
    writeln!(
        writer,
        "{indent}      \"weights_gb\": {},",
        json_optional_f64(memory.components.weights_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"kv_cache_gb\": {},",
        json_optional_f64(memory.components.kv_cache_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"block_table_gb\": {},",
        json_optional_f64(memory.components.block_table_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"activations_gb\": {},",
        json_optional_f64(memory.components.activations_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"temporary_gb\": {},",
        json_optional_f64(memory.components.temporary_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"communication_gb\": {},",
        json_optional_f64(memory.components.communication_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"runtime_reserve_gb\": {},",
        json_optional_f64(memory.components.runtime_reserve_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"fragmentation_gb\": {},",
        json_optional_f64(memory.components.fragmentation_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"total_gb\": {}",
        json_optional_f64(memory.components.total_gb)
    )?;
    writeln!(writer, "{indent}    }}")?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_serving_hardware_footprint<W: Write>(
    writer: &mut W,
    indent: &str,
    footprint: &ServingHardwareFootprint,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"hardware_footprint\": {{")?;
    writeln!(
        writer,
        "{indent}    \"unique_node_count\": {},",
        footprint.unique_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"unique_gpu_count\": {},",
        footprint.unique_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_node_count\": {},",
        footprint.prefill_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_gpu_count\": {},",
        footprint.prefill_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_node_count\": {},",
        footprint.decode_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_gpu_count\": {},",
        footprint.decode_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"shared_node_count\": {},",
        footprint.shared_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"shared_gpu_count\": {},",
        footprint.shared_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_hbm_gb\": {},",
        json_optional_f64(footprint.aggregate_hbm_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_hbm_gb\": {},",
        json_optional_f64(footprint.prefill_hbm_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_hbm_gb\": {},",
        json_optional_f64(footprint.decode_hbm_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_hbm_bandwidth_gb_s\": {},",
        json_optional_f64(footprint.aggregate_hbm_bandwidth_gb_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_hbm_bandwidth_gb_s\": {},",
        json_optional_f64(footprint.prefill_hbm_bandwidth_gb_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_hbm_bandwidth_gb_s\": {},",
        json_optional_f64(footprint.decode_hbm_bandwidth_gb_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_peak_f16_tflops\": {},",
        json_optional_f64(footprint.aggregate_peak_f16_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_peak_f16_tflops\": {},",
        json_optional_f64(footprint.prefill_peak_f16_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_peak_f16_tflops\": {},",
        json_optional_f64(footprint.decode_peak_f16_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_peak_f8_tflops\": {},",
        json_optional_value(footprint.aggregate_peak_f8_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_peak_f8_tflops\": {},",
        json_optional_value(footprint.prefill_peak_f8_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_peak_f8_tflops\": {},",
        json_optional_value(footprint.decode_peak_f8_tflops)
    )?;
    write_serving_gpu_type_counts(
        writer,
        indent,
        "aggregate_gpu_types",
        &footprint.aggregate_gpu_types,
        true,
    )?;
    write_serving_gpu_type_counts(
        writer,
        indent,
        "prefill_gpu_types",
        &footprint.prefill_gpu_types,
        true,
    )?;
    write_serving_gpu_type_counts(
        writer,
        indent,
        "decode_gpu_types",
        &footprint.decode_gpu_types,
        true,
    )?;
    write_serving_gpu_label_counts(
        writer,
        indent,
        "aggregate_gpu_label_counts",
        &footprint.aggregate_gpu_label_counts,
        true,
    )?;
    write_serving_gpu_label_counts(
        writer,
        indent,
        "prefill_gpu_label_counts",
        &footprint.prefill_gpu_label_counts,
        true,
    )?;
    write_serving_gpu_label_counts(
        writer,
        indent,
        "decode_gpu_label_counts",
        &footprint.decode_gpu_label_counts,
        true,
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_effective_peak_tflops\": {},",
        json_optional_f64(footprint.aggregate_effective_peak_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_effective_peak_tflops\": {},",
        json_optional_f64(footprint.prefill_effective_peak_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_effective_peak_tflops\": {},",
        json_optional_f64(footprint.decode_effective_peak_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s_per_gpu\": {},",
        json_optional_f64(footprint.throughput_tokens_per_s_per_gpu)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s_per_effective_peak_tflop\": {},",
        json_optional_f64(footprint.throughput_tokens_per_s_per_effective_peak_tflop)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s_per_hbm_gb\": {}",
        json_optional_f64(footprint.throughput_tokens_per_s_per_hbm_gb)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_serving_gpu_type_counts<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    counts: &[ServingGpuTypeCount],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"{field}\": [")?;
    for (idx, entry) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"gpu\": {},",
            json_string(&entry.gpu)
        )?;
        writeln!(writer, "{indent}        \"count\": {}", entry.count)?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_serving_gpu_label_counts<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    counts: &[ServingGpuLabelCount],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"{field}\": [")?;
    for (idx, entry) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"label\": {},",
            json_string(&entry.label)
        )?;
        writeln!(writer, "{indent}        \"count\": {}", entry.count)?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

pub(super) fn write_serving_cost_estimate<W: Write>(
    writer: &mut W,
    indent: &str,
    estimate: &ServingCostEstimate,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"cost_estimate\": {{")?;
    writeln!(
        writer,
        "{indent}    \"modeled_duration_s\": {},",
        json_optional_value(estimate.modeled_duration_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"modeled_gpu_count\": {},",
        estimate.modeled_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"modeled_node_count\": {},",
        estimate.modeled_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"gpu_hours\": {},",
        json_optional_value(estimate.gpu_hours)
    )?;
    writeln!(
        writer,
        "{indent}    \"node_hours\": {},",
        json_optional_value(estimate.node_hours)
    )?;
    writeln!(
        writer,
        "{indent}    \"gpu_hour_cost_usd\": {},",
        json_optional_value(estimate.gpu_hour_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"node_hour_cost_usd\": {},",
        json_optional_value(estimate.node_hour_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"average_power_watts\": {},",
        json_optional_value(estimate.average_power_watts)
    )?;
    writeln!(
        writer,
        "{indent}    \"energy_kwh\": {},",
        json_optional_value(estimate.energy_kwh)
    )?;
    writeln!(
        writer,
        "{indent}    \"energy_cost_usd\": {},",
        json_optional_value(estimate.energy_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"total_cost_usd\": {},",
        json_optional_value(estimate.total_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"cost_per_1k_output_tokens_usd\": {},",
        json_optional_value(estimate.cost_per_1k_output_tokens_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"cost_per_1k_requests_usd\": {}",
        json_optional_value(estimate.cost_per_1k_requests_usd)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_memory_limiter_gpu<W: Write>(
    writer: &mut W,
    indent: &str,
    limiting_gpu: Option<GpuAddr>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(gpu) = limiting_gpu else {
        return writeln!(
            writer,
            "{indent}    \"limiting_gpu\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}    \"limiting_gpu\": {{")?;
    writeln!(writer, "{indent}      \"node_id\": {},", gpu.node_id)?;
    writeln!(
        writer,
        "{indent}      \"local_gpu_id\": {}",
        gpu.local_gpu_id
    )?;
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_memory_dominant_component<W: Write>(
    writer: &mut W,
    indent: &str,
    memory: &ServingMemoryHeadroom,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(component) = memory.dominant_component() else {
        return writeln!(
            writer,
            "{indent}    \"dominant_component\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}    \"dominant_component\": {{")?;
    writeln!(
        writer,
        "{indent}      \"name\": {},",
        json_string(component.name)
    )?;
    writeln!(
        writer,
        "{indent}      \"gb\": {},",
        json_optional_f64(component.gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"fraction_of_total\": {}",
        json_optional_f64(component.fraction_of_total)
    )?;
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_memory_component_fractions<W: Write>(
    writer: &mut W,
    indent: &str,
    memory: &ServingMemoryHeadroom,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let contributions = memory.components.component_contributions();
    writeln!(writer, "{indent}    \"component_fractions\": {{")?;
    for (idx, component) in contributions.iter().enumerate() {
        writeln!(
            writer,
            "{indent}      \"{}\": {}{}",
            component.name,
            json_optional_f64(component.fraction_of_total),
            comma(idx + 1 < contributions.len())
        )?;
    }
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

pub(super) fn write_memory_pressure_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingMemoryPressureObservation],
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_observations =
        limit.map_or(observations.len(), |limit| limit.min(observations.len()));
    writeln!(
        writer,
        "{indent}  \"memory_pressure_observation_count\": {},",
        observations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"memory_pressure_observations_truncated\": {},",
        displayed_observations < observations.len()
    )?;
    writeln!(writer, "{indent}  \"memory_pressure\": [")?;
    for (idx, observation) in observations.iter().take(displayed_observations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&observation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"estimate_kind\": {},",
            json_string(&observation.estimate_kind)
        )?;
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_ms(observation.start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_ms(observation.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"duration_ms\": {},",
            json_ms(observation.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"active_requests\": {},",
            observation.active_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"active_tokens\": {},",
            observation.active_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_blocks\": {},",
            observation.kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"estimated_per_gpu_gb\": {},",
            json_optional_f64(observation.estimated_per_gpu_gb)
        )?;
        writeln!(
            writer,
            "{indent}      \"min_hbm_per_gpu_gb\": {},",
            json_optional_f64(observation.min_hbm_per_gpu_gb)
        )?;
        writeln!(
            writer,
            "{indent}      \"capacity_used_fraction\": {},",
            json_optional_f64(observation.capacity_used_fraction)
        )?;
        writeln!(
            writer,
            "{indent}      \"headroom_gb\": {},",
            json_optional_f64(observation.headroom_gb)
        )?;
        write_memory_pressure_limiter_gpu(writer, indent, observation.limiting_gpu, true)?;
        write_memory_pressure_dominant_component(
            writer,
            indent,
            observation.dominant_component,
            true,
        )?;
        write_memory_pressure_component_fractions(writer, indent, observation, true)?;
        write_memory_pressure_components(writer, indent, observation, false)?;
        if idx + 1 < displayed_observations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_memory_pressure_limiter_gpu<W: Write>(
    writer: &mut W,
    indent: &str,
    limiting_gpu: Option<GpuAddr>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(gpu) = limiting_gpu else {
        return writeln!(
            writer,
            "{indent}      \"limiting_gpu\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}      \"limiting_gpu\": {{")?;
    writeln!(writer, "{indent}        \"node_id\": {},", gpu.node_id)?;
    writeln!(
        writer,
        "{indent}        \"local_gpu_id\": {}",
        gpu.local_gpu_id
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_memory_pressure_dominant_component<W: Write>(
    writer: &mut W,
    indent: &str,
    component: Option<crate::serving::ServingMemoryComponentContribution>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(component) = component else {
        return writeln!(
            writer,
            "{indent}      \"dominant_component\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}      \"dominant_component\": {{")?;
    writeln!(
        writer,
        "{indent}        \"name\": {},",
        json_string(component.name)
    )?;
    writeln!(
        writer,
        "{indent}        \"gb\": {},",
        json_optional_f64(component.gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"fraction_of_total\": {}",
        json_optional_f64(component.fraction_of_total)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_memory_pressure_component_fractions<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingMemoryPressureObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let contributions = observation.components.component_contributions();
    writeln!(writer, "{indent}      \"component_fractions\": {{")?;
    for (idx, component) in contributions.iter().enumerate() {
        writeln!(
            writer,
            "{indent}        \"{}\": {}{}",
            component.name,
            json_optional_f64(component.fraction_of_total),
            comma(idx + 1 < contributions.len())
        )?;
    }
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_memory_pressure_components<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingMemoryPressureObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let components = observation.components;
    writeln!(writer, "{indent}      \"components\": {{")?;
    writeln!(
        writer,
        "{indent}        \"weights_gb\": {},",
        json_optional_f64(components.weights_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"kv_cache_gb\": {},",
        json_optional_f64(components.kv_cache_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"block_table_gb\": {},",
        json_optional_f64(components.block_table_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"activations_gb\": {},",
        json_optional_f64(components.activations_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"temporary_gb\": {},",
        json_optional_f64(components.temporary_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"communication_gb\": {},",
        json_optional_f64(components.communication_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"runtime_reserve_gb\": {},",
        json_optional_f64(components.runtime_reserve_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"fragmentation_gb\": {},",
        json_optional_f64(components.fragmentation_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"total_gb\": {}",
        json_optional_f64(components.total_gb)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}
