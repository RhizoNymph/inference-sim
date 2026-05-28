use super::*;

pub(super) fn write_serving_metrics<W: Write>(
    writer: &mut W,
    feasible: bool,
    metrics: &ServingMetrics,
    calibration_fits: &[CalibrationFitApplication],
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let overall_uncertainty = calibration_uncertainty_summary(calibration_fits.iter());
    let prefill_uncertainty =
        calibration_phase_uncertainty_summary(calibration_fits.iter(), "prefill");
    let decode_uncertainty =
        calibration_phase_uncertainty_summary(calibration_fits.iter(), "decode");
    let kv_transfer_uncertainty =
        calibration_phase_uncertainty_summary(calibration_fits.iter(), "kv_transfer");
    writeln!(writer, "{indent}  \"metrics\": {{")?;
    writeln!(
        writer,
        "{indent}    \"ttft_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.ttft_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.ttft_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.ttft_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p50_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p90_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p95_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p99_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_max_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_slo_constrained_requests\": {},",
        metrics.ttft_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_slo_missed_requests\": {},",
        metrics.ttft_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_slo_miss_rate\": {},",
        json_optional_f64(metrics.ttft_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.tpot_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.tpot_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.tpot_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p50_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p90_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p95_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p99_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_max_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_slo_constrained_requests\": {},",
        metrics.tpot_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_slo_missed_requests\": {},",
        metrics.tpot_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_slo_miss_rate\": {},",
        json_optional_f64(metrics.tpot_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_ms\": {},",
        json_optional_ms(feasible, metrics.itl_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.itl_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.itl_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.itl_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p50_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p90_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p95_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p99_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_max_ms\": {},",
        json_optional_ms(feasible, metrics.itl_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_slo_constrained_requests\": {},",
        metrics.itl_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_slo_missed_requests\": {},",
        metrics.itl_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_slo_miss_rate\": {},",
        json_optional_f64(metrics.itl_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iterations\": {},",
        metrics.decode_iterations
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.decode_iteration_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.decode_iteration_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.decode_iteration_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p50_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p90_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p95_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p99_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_max_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s\": {},",
        json_optional_f64(metrics.throughput_tokens_per_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_calibration_uncertainty_tokens_per_s\": {},",
        json_metric_relative_uncertainty_value(
            feasible,
            metrics.throughput_tokens_per_s,
            &overall_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_calibration_lower_tokens_per_s\": {},",
        json_metric_relative_lower_value(
            feasible,
            metrics.throughput_tokens_per_s,
            &overall_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_calibration_upper_tokens_per_s\": {},",
        json_metric_relative_upper_value(
            feasible,
            metrics.throughput_tokens_per_s,
            &overall_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.e2el_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.e2el_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.e2el_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p50_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p90_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p95_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p99_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_max_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_slo_constrained_requests\": {},",
        metrics.e2el_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_slo_missed_requests\": {},",
        metrics.e2el_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_slo_miss_rate\": {},",
        json_optional_f64(metrics.e2el_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_miss_rate\": {},",
        json_optional_f64(metrics.deadline_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"service_ms\": {},",
        json_optional_ms(feasible, metrics.service_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_ms\": {},",
        json_optional_ms(feasible, metrics.prefill_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.prefill_s, &prefill_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.prefill_s, &prefill_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.prefill_s, &prefill_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_chunks\": {},",
        metrics.prefill_chunks
    )?;
    writeln!(
        writer,
        "{indent}    \"prompt_tokens\": {},",
        metrics.prompt_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"prefix_cache_hit_tokens\": {},",
        metrics.prefix_cache_hit_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"effective_prefill_tokens\": {},",
        metrics.effective_prefill_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"prefix_cache_hit_rate\": {},",
        json_optional_f64(metrics.prefix_cache_hit_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_ms\": {},",
        json_optional_ms(feasible, metrics.kv_transfer_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.kv_transfer_s, &kv_transfer_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.kv_transfer_s, &kv_transfer_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.kv_transfer_s, &kv_transfer_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_queue_ms\": {},",
        json_optional_ms(feasible, metrics.kv_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_worker_queue_ms\": {},",
        json_optional_ms(feasible, metrics.kv_worker_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_resource_queue_ms\": {},",
        json_optional_ms(feasible, metrics.kv_resource_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_queue_ms\": {},",
        json_optional_ms(feasible, metrics.decode_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_worker_queue_ms\": {},",
        json_optional_ms(feasible, metrics.prefill_worker_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_resource_queue_ms\": {},",
        json_optional_ms(feasible, metrics.prefill_resource_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_worker_queue_ms\": {},",
        json_optional_ms(feasible, metrics.decode_worker_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_resource_queue_ms\": {},",
        json_optional_ms(feasible, metrics.decode_resource_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_ms\": {},",
        json_optional_ms(feasible, metrics.decode_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.decode_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.decode_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.decode_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_p90_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_p95_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_max_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_prefill_tokens\": {},",
        metrics.peak_prefill_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_prefill_tokens_per_node\": {},",
        metrics.peak_prefill_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_prefill_tokens_per_gpu\": {},",
        metrics.peak_prefill_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_decode_sequences\": {},",
        metrics.peak_decode_sequences
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_resident_tokens\": {},",
        metrics.peak_resident_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_decode_sequences_per_node\": {},",
        metrics.peak_decode_sequences_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_resident_tokens_per_node\": {},",
        metrics.peak_resident_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_decode_sequences_per_gpu\": {},",
        metrics.peak_decode_sequences_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_resident_tokens_per_gpu\": {},",
        metrics.peak_resident_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_blocks\": {},",
        metrics.peak_kv_blocks
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_allocated_kv_tokens\": {},",
        metrics.peak_allocated_kv_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_fragmentation_tokens\": {},",
        metrics.peak_kv_fragmentation_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_block_table_bytes\": {},",
        metrics.peak_kv_block_table_bytes
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_blocks_per_node\": {},",
        metrics.peak_kv_blocks_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_allocated_kv_tokens_per_node\": {},",
        metrics.peak_allocated_kv_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_fragmentation_tokens_per_node\": {},",
        metrics.peak_kv_fragmentation_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_block_table_bytes_per_node\": {},",
        metrics.peak_kv_block_table_bytes_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_blocks_per_gpu\": {},",
        metrics.peak_kv_blocks_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_allocated_kv_tokens_per_gpu\": {},",
        metrics.peak_allocated_kv_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_fragmentation_tokens_per_gpu\": {},",
        metrics.peak_kv_fragmentation_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_block_table_bytes_per_gpu\": {},",
        metrics.peak_kv_block_table_bytes_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_sequence_utilization\": {},",
        json_optional_f64(metrics.decode_sequence_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"resident_token_utilization\": {},",
        json_optional_f64(metrics.resident_token_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_block_utilization\": {},",
        json_optional_f64(metrics.kv_block_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_sequence_per_node_utilization\": {},",
        json_optional_f64(metrics.decode_sequence_per_node_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"resident_token_per_node_utilization\": {},",
        json_optional_f64(metrics.resident_token_per_node_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_block_per_node_utilization\": {},",
        json_optional_f64(metrics.kv_block_per_node_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_sequence_per_gpu_utilization\": {},",
        json_optional_f64(metrics.decode_sequence_per_gpu_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"resident_token_per_gpu_utilization\": {},",
        json_optional_f64(metrics.resident_token_per_gpu_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_block_per_gpu_utilization\": {},",
        json_optional_f64(metrics.kv_block_per_gpu_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_ms\": {},",
        json_optional_ms(feasible, metrics.scheduled_makespan_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.scheduled_makespan_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.scheduled_makespan_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.scheduled_makespan_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_requests\": {},",
        metrics.scheduled_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"admitted_requests\": {},",
        metrics.admitted_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"completed_requests\": {},",
        metrics.completed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_requests\": {},",
        metrics.rejected_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"timed_out_requests\": {},",
        metrics.timed_out_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"cancelled_requests\": {},",
        metrics.cancelled_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_constrained_requests\": {},",
        metrics.deadline_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_missed_requests\": {},",
        metrics.deadline_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"measured_requests\": {},",
        metrics.measured_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"measurement_start_ms\": {},",
        json_optional_ms(feasible, metrics.measurement_start_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"measurement_end_ms\": {}",
        json_optional_ms(feasible, metrics.measurement_end_s)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_measurement_window<W: Write>(
    writer: &mut W,
    feasible: bool,
    window: &ServingMeasurementWindowObservation,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let candidate_start_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_start_s)
    } else {
        "null".to_string()
    };
    let candidate_end_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_end_s)
    } else {
        "null".to_string()
    };
    let steady_state_min_requests = if feasible {
        json_optional_u32(window.steady_state_min_requests)
    } else {
        "null".to_string()
    };
    let steady_state_max_cv = if feasible {
        json_optional_value(window.steady_state_max_cv)
    } else {
        "null".to_string()
    };
    let candidate_request_count = if feasible {
        json_optional_u32(window.steady_state_candidate_request_count)
    } else {
        "null".to_string()
    };
    let candidate_e2el_mean_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_e2el_mean_s)
    } else {
        "null".to_string()
    };
    let candidate_e2el_stddev_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_e2el_stddev_s)
    } else {
        "null".to_string()
    };
    let candidate_e2el_cv = if feasible {
        json_optional_value(window.steady_state_candidate_e2el_cv)
    } else {
        "null".to_string()
    };
    let candidate_e2el_std_error_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_e2el_std_error_s)
    } else {
        "null".to_string()
    };
    let candidate_worst_metric = if feasible {
        json_optional_string(window.steady_state_candidate_worst_metric.as_deref())
    } else {
        "null".to_string()
    };
    let candidate_worst_cv = if feasible {
        json_optional_value(window.steady_state_candidate_worst_cv)
    } else {
        "null".to_string()
    };
    let candidate_output_tokens = if feasible {
        json_optional_u64(window.steady_state_candidate_output_tokens)
    } else {
        "null".to_string()
    };
    let candidate_throughput = if feasible {
        json_optional_value(window.steady_state_candidate_throughput_tokens_per_s)
    } else {
        "null".to_string()
    };
    let candidate_worst_utilization_resource = if feasible {
        json_optional_string(
            window
                .steady_state_candidate_worst_utilization_resource
                .as_deref(),
        )
    } else {
        "null".to_string()
    };
    let candidate_worst_utilization_cv = if feasible {
        json_optional_value(window.steady_state_candidate_worst_utilization_cv)
    } else {
        "null".to_string()
    };
    writeln!(writer, "{indent}  \"measurement_window\": {{")?;
    writeln!(
        writer,
        "{indent}    \"source\": {},",
        json_string(&window.source)
    )?;
    writeln!(
        writer,
        "{indent}    \"start_ms\": {},",
        json_optional_ms(feasible, window.start_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"end_ms\": {},",
        json_optional_ms(feasible, window.end_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"duration_ms\": {},",
        json_optional_ms(feasible, window.duration_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"request_count\": {},",
        window.request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"completed_request_count\": {},",
        window.completed_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"failed_request_count\": {},",
        window.failed_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_request_count\": {},",
        window.rejected_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"timed_out_request_count\": {},",
        window.timed_out_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"cancelled_request_count\": {},",
        window.cancelled_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_constrained_request_count\": {},",
        window.deadline_constrained_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_missed_request_count\": {},",
        window.deadline_missed_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"measured_requests\": {},",
        window.measured_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"lifecycle_event_metric_request_count\": {},",
        window.lifecycle_event_metric_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"fallback_metric_request_count\": {},",
        window.fallback_metric_request_count
    )?;
    write_measurement_metric_source_counts(
        writer,
        feasible.then_some(window.metric_source_counts.as_slice()),
        indent,
        true,
    )?;
    writeln!(
        writer,
        "{indent}    \"configured_start_bound\": {},",
        window.configured_start_bound
    )?;
    writeln!(
        writer,
        "{indent}    \"configured_end_bound\": {},",
        window.configured_end_bound
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_requested\": {},",
        window.steady_state_requested
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_applied\": {},",
        window.steady_state_applied
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_min_requests\": {steady_state_min_requests},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_max_cv\": {steady_state_max_cv},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_sample_count\": {},",
        window.steady_state_sample_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_matching_window_count\": {},",
        window.steady_state_matching_window_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_request_count\": {candidate_request_count},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_start_ms\": {candidate_start_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_end_ms\": {candidate_end_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_mean_ms\": {candidate_e2el_mean_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_stddev_ms\": {candidate_e2el_stddev_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_cv\": {candidate_e2el_cv},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_std_error_ms\": {candidate_e2el_std_error_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_metric_count\": {},",
        window.steady_state_candidate_metric_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_metric\": {candidate_worst_metric},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_cv\": {candidate_worst_cv},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_output_tokens\": {candidate_output_tokens},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_throughput_tokens_per_s\": {candidate_throughput},"
    )?;
    write_steady_state_candidate_metrics(
        writer,
        feasible.then_some(window.steady_state_candidate_metrics.as_slice()),
        indent,
        true,
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_utilization_count\": {},",
        window.steady_state_candidate_utilization_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_utilization_resource\": {candidate_worst_utilization_resource},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_utilization_cv\": {candidate_worst_utilization_cv},"
    )?;
    write_steady_state_candidate_utilization(
        writer,
        feasible.then_some(window.steady_state_candidate_utilization.as_slice()),
        indent,
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_measurement_metric_source_counts<W: Write>(
    writer: &mut W,
    counts: Option<&[ServingMeasurementMetricSourceCount]>,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"metric_source_counts\": [")?;
    let counts = counts.unwrap_or(&[]);
    for (idx, count) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"metric_source\": {},",
            json_string(&count.metric_source)
        )?;
        writeln!(
            writer,
            "{indent}        \"request_count\": {}",
            count.request_count
        )?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_steady_state_candidate_metrics<W: Write>(
    writer: &mut W,
    metrics: Option<&[ServingSteadyStateMetricObservation]>,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"steady_state_candidate_metrics\": [")?;
    if let Some(metrics) = metrics {
        for (idx, metric) in metrics.iter().enumerate() {
            writeln!(writer, "{indent}      {{")?;
            writeln!(
                writer,
                "{indent}        \"metric\": {},",
                json_string(&metric.metric)
            )?;
            writeln!(
                writer,
                "{indent}        \"unit\": {},",
                json_string(&metric.unit)
            )?;
            writeln!(
                writer,
                "{indent}        \"sample_count\": {},",
                metric.sample_count
            )?;
            writeln!(
                writer,
                "{indent}        \"mean\": {},",
                json_optional_f64(metric.mean)
            )?;
            writeln!(
                writer,
                "{indent}        \"stddev\": {},",
                json_optional_f64(metric.stddev)
            )?;
            writeln!(
                writer,
                "{indent}        \"cv\": {},",
                json_optional_f64(metric.cv)
            )?;
            writeln!(
                writer,
                "{indent}        \"std_error\": {}",
                json_optional_f64(metric.std_error)
            )?;
            writeln!(writer, "{indent}      }}{}", comma(idx + 1 < metrics.len()))?;
        }
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_steady_state_candidate_utilization<W: Write>(
    writer: &mut W,
    observations: Option<&[ServingSteadyStateUtilizationObservation]>,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_utilization\": ["
    )?;
    if let Some(observations) = observations {
        for (idx, observation) in observations.iter().enumerate() {
            writeln!(writer, "{indent}      {{")?;
            writeln!(
                writer,
                "{indent}        \"source\": {},",
                json_string(&observation.source)
            )?;
            writeln!(
                writer,
                "{indent}        \"phase\": {},",
                json_string(&observation.phase)
            )?;
            writeln!(
                writer,
                "{indent}        \"resource_kind\": {},",
                json_string(&observation.resource_kind)
            )?;
            writeln!(
                writer,
                "{indent}        \"resource\": {},",
                json_string(&observation.resource)
            )?;
            writeln!(
                writer,
                "{indent}        \"bucket_count\": {},",
                observation.bucket_count
            )?;
            writeln!(
                writer,
                "{indent}        \"active_bucket_count\": {},",
                observation.active_bucket_count
            )?;
            writeln!(
                writer,
                "{indent}        \"event_count\": {},",
                observation.event_count
            )?;
            writeln!(
                writer,
                "{indent}        \"mean_utilization\": {},",
                json_optional_f64(observation.mean_utilization)
            )?;
            writeln!(
                writer,
                "{indent}        \"max_utilization\": {},",
                json_optional_f64(observation.max_utilization)
            )?;
            writeln!(
                writer,
                "{indent}        \"utilization_cv\": {}",
                json_optional_f64(observation.utilization_cv)
            )?;
            writeln!(
                writer,
                "{indent}      }}{}",
                comma(idx + 1 < observations.len())
            )?;
        }
    }
    writeln!(writer, "{indent}    ]")
}

pub(super) fn write_metric_breakdowns<W: Write>(
    writer: &mut W,
    indent: &str,
    breakdowns: &[ServingMetricBreakdown],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"metric_breakdowns\": [")?;
    for (idx, breakdown) in breakdowns.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"group\": {},",
            json_string(&breakdown.group)
        )?;
        writeln!(
            writer,
            "{indent}      \"key\": {},",
            json_string(&breakdown.key)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            breakdown.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"completed_requests\": {},",
            breakdown.completed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"failed_requests\": {},",
            breakdown.failed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"rejected_requests\": {},",
            breakdown.rejected_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"timed_out_requests\": {},",
            breakdown.timed_out_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"cancelled_requests\": {},",
            breakdown.cancelled_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"output_tokens\": {},",
            breakdown.output_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"lifecycle_event_metric_request_count\": {},",
            breakdown.lifecycle_event_metric_request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"fallback_metric_request_count\": {},",
            breakdown.fallback_metric_request_count
        )?;
        write_measurement_metric_source_counts(
            writer,
            Some(breakdown.metric_source_counts.as_slice()),
            &format!("{indent}  "),
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_constrained_requests\": {},",
            breakdown.deadline_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_missed_requests\": {},",
            breakdown.deadline_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_miss_rate\": {},",
            json_optional_f64(breakdown.deadline_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_constrained_requests\": {},",
            breakdown.ttft_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_missed_requests\": {},",
            breakdown.ttft_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_miss_rate\": {},",
            json_optional_f64(breakdown.ttft_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_constrained_requests\": {},",
            breakdown.tpot_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_missed_requests\": {},",
            breakdown.tpot_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_miss_rate\": {},",
            json_optional_f64(breakdown.tpot_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_constrained_requests\": {},",
            breakdown.itl_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_missed_requests\": {},",
            breakdown.itl_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_miss_rate\": {},",
            json_optional_f64(breakdown.itl_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_constrained_requests\": {},",
            breakdown.e2el_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_missed_requests\": {},",
            breakdown.e2el_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_miss_rate\": {},",
            json_optional_f64(breakdown.e2el_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_ms\": {},",
            json_ms(breakdown.ttft_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_p90_ms\": {},",
            json_ms(breakdown.ttft_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_p95_ms\": {},",
            json_ms(breakdown.ttft_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_max_ms\": {},",
            json_ms(breakdown.ttft_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_ms\": {},",
            json_ms(breakdown.tpot_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_p90_ms\": {},",
            json_ms(breakdown.tpot_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_p95_ms\": {},",
            json_ms(breakdown.tpot_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_max_ms\": {},",
            json_ms(breakdown.tpot_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_ms\": {},",
            json_ms(breakdown.itl_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_p90_ms\": {},",
            json_ms(breakdown.itl_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_p95_ms\": {},",
            json_ms(breakdown.itl_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_max_ms\": {},",
            json_ms(breakdown.itl_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"throughput_tokens_per_s\": {},",
            json_optional_f64(breakdown.throughput_tokens_per_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_ms\": {},",
            json_ms(breakdown.e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_p90_ms\": {},",
            json_ms(breakdown.e2el_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_p95_ms\": {},",
            json_ms(breakdown.e2el_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_max_ms\": {}",
            json_ms(breakdown.e2el_max_s)
        )?;
        if idx + 1 < breakdowns.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_node_capacity<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingNodeCapacityObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"decode_node_capacity\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"node_id\": {},",
            observation.node_id
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_gpu_capacity<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingGpuCapacityObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"decode_gpu_capacity\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"node_id\": {},",
            observation.node_id
        )?;
        writeln!(
            writer,
            "{indent}      \"local_gpu_id\": {},",
            observation.local_gpu_id
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_traffic_class_capacity<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingTrafficClassCapacityObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"traffic_class_capacity\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&observation.name)
        )?;
        writeln!(
            writer,
            "{indent}      \"group\": {},",
            json_string(&observation.group)
        )?;
        writeln!(
            writer,
            "{indent}      \"key\": {},",
            json_string(&observation.key)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_prefill_tokens\": {},",
            json_optional_u64(observation.max_prefill_tokens)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_decode_sequences\": {},",
            json_optional_u32(observation.max_decode_sequences)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_resident_tokens\": {},",
            json_optional_u64(observation.max_resident_tokens)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_kv_blocks\": {},",
            json_optional_u64(observation.max_kv_blocks)
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_token_utilization\": {},",
            json_optional_f64(observation.prefill_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_worker_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingWorkerObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"serving_workers\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&observation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"node_id\": {},",
            observation.node_id
        )?;
        writeln!(
            writer,
            "{indent}      \"local_gpu_id\": {},",
            observation.local_gpu_id
        )?;
        writeln!(
            writer,
            "{indent}      \"configured_worker_slots\": {},",
            observation.configured_worker_slots
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_active_worker_slots\": {},",
            observation.peak_active_worker_slots
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_slot_utilization\": {},",
            json_optional_f64(observation.worker_slot_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            observation.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"completed_requests\": {},",
            observation.completed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"failed_requests\": {},",
            observation.failed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"input_tokens\": {},",
            observation.input_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"output_tokens\": {},",
            observation.output_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_ms\": {},",
            json_ms(observation.queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_p95_ms\": {},",
            json_ms(observation.queue_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_max_ms\": {},",
            json_ms(observation.queue_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_ms\": {},",
            json_ms(observation.worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_p95_ms\": {},",
            json_ms(observation.worker_queue_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_max_ms\": {},",
            json_ms(observation.worker_queue_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_ms\": {},",
            json_ms(observation.resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_p95_ms\": {},",
            json_ms(observation.resource_queue_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_max_ms\": {},",
            json_ms(observation.resource_queue_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"service_ms\": {},",
            json_ms(observation.service_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"first_start_ms\": {},",
            json_ms(observation.first_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_finish_ms\": {},",
            json_ms(observation.last_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        write_worker_kv_cache_owner_slots(writer, indent, &observation.kv_cache_owner_slots)?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_worker_kv_cache_owner_slots<W: Write>(
    writer: &mut W,
    indent: &str,
    slots: &[ServingWorkerKvSlotObservation],
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"kv_cache_owner_slots\": [")?;
    for (idx, slot) in slots.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(writer, "{indent}          \"slot\": {},", slot.slot)?;
        writeln!(
            writer,
            "{indent}          \"peak_decode_sequences\": {},",
            slot.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_resident_tokens\": {},",
            slot.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_kv_blocks\": {},",
            slot.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_allocated_kv_tokens\": {},",
            slot.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_kv_fragmentation_tokens\": {},",
            slot.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_kv_block_table_bytes\": {},",
            slot.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_sequence_utilization\": {},",
            json_optional_f64(slot.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}          \"resident_token_utilization\": {},",
            json_optional_f64(slot.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_block_utilization\": {}",
            json_optional_f64(slot.kv_block_utilization)
        )?;
        if idx + 1 < slots.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ],")
}

pub(super) fn write_service_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingServiceObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"serving_services\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&observation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"health\": {},",
            json_string(observation.health.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"accepts_requests\": {},",
            observation.accepts_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_scale\": {},",
            json_optional_f64(observation.worker_scale)
        )?;
        writeln!(
            writer,
            "{indent}      \"configured_worker_slots_per_gpu\": {},",
            observation.configured_worker_slots_per_gpu
        )?;
        writeln!(
            writer,
            "{indent}      \"effective_worker_slots_per_gpu\": {},",
            observation.effective_worker_slots_per_gpu
        )?;
        writeln!(
            writer,
            "{indent}      \"node_count\": {},",
            observation.node_count
        )?;
        writeln!(
            writer,
            "{indent}      \"gpu_count\": {},",
            observation.gpu_count
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            observation.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"admitted_requests\": {},",
            observation.admitted_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"completed_requests\": {},",
            observation.completed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"failed_requests\": {},",
            observation.failed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"rejected_requests\": {},",
            observation.rejected_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"timed_out_requests\": {},",
            observation.timed_out_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"cancelled_requests\": {},",
            observation.cancelled_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_cap_ms\": {},",
            json_optional_value(observation.queue_cap_s.map(|seconds| seconds * 1000.0))
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_cap_request_count\": {},",
            observation.queue_cap_request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_cap_hit_count\": {},",
            observation.queue_cap_hit_count
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iteration_queue_cap_ms\": {},",
            json_optional_value(
                observation
                    .decode_iteration_queue_cap_s
                    .map(|seconds| seconds * 1000.0)
            )
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iteration_queue_cap_request_count\": {},",
            observation.decode_iteration_queue_cap_request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iteration_queue_cap_hit_count\": {},",
            observation.decode_iteration_queue_cap_hit_count
        )?;
        writeln!(
            writer,
            "{indent}      \"backpressure_rejections\": {},",
            observation.backpressure_rejections
        )?;
        writeln!(
            writer,
            "{indent}      \"timeout_rejections\": {},",
            observation.timeout_rejections
        )?;
        writeln!(
            writer,
            "{indent}      \"backpressure_state\": {},",
            json_string(&observation.backpressure_state)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_slot_utilization\": {},",
            json_optional_f64(observation.worker_slot_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_ms\": {},",
            json_optional_f64(observation.queue_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_p95_ms\": {},",
            json_optional_f64(observation.queue_p95_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_max_ms\": {},",
            json_optional_f64(observation.queue_max_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_ms\": {},",
            json_optional_f64(observation.worker_queue_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_ms\": {},",
            json_optional_f64(observation.resource_queue_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"service_ms\": {}",
            json_optional_f64(observation.service_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < observations.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_resource_utilization<W: Write>(
    writer: &mut W,
    indent: &str,
    utilization: &[ResourceUtilization],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"resource_utilization\": [")?;
    for (idx, resource) in utilization.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&resource.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"busy_ms\": {},",
            json_optional_f64(resource.busy_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"utilization\": {},",
            json_optional_f64(resource.utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"operation_count\": {},",
            resource.operation_count
        )?;
        writeln!(
            writer,
            "{indent}      \"first_start_ms\": {},",
            json_optional_f64(resource.first_start_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_finish_ms\": {}",
            json_optional_f64(resource.last_finish_s * 1000.0)
        )?;
        if idx + 1 < utilization.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_serving_bottleneck_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summaries: &[ServingBottleneckSummary],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"bottleneck_summary\": [")?;
    for (idx, summary) in summaries.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"source\": {},",
            json_string(&summary.source)
        )?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&summary.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&summary.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&summary.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&summary.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"severity\": {},",
            json_string(&summary.severity)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(summary.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(summary.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {},",
            summary
                .unit
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&summary.message)
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {}",
            summary
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < summaries.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_pareto_dimensions<W: Write>(
    writer: &mut W,
    indent: &str,
    dimensions: &[ServingParetoDimension],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"pareto_dimensions\": [")?;
    for (idx, dimension) in dimensions.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"metric\": {},",
            json_string(&dimension.metric)
        )?;
        writeln!(
            writer,
            "{indent}      \"direction\": {},",
            json_string(&dimension.direction)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {}",
            json_string(&dimension.unit)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < dimensions.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_phase_resource_utilization<W: Write>(
    writer: &mut W,
    indent: &str,
    utilization: &[ServingPhaseResourceUtilization],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"phase_resource_utilization\": [")?;
    for (idx, resource) in utilization.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&resource.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_kind\": {},",
            json_string(&resource.resource_kind)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&resource.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"busy_ms\": {},",
            json_optional_f64(resource.busy_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"utilization\": {},",
            json_optional_f64(resource.utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"operation_count\": {},",
            resource.operation_count
        )?;
        writeln!(
            writer,
            "{indent}      \"first_start_ms\": {},",
            json_optional_f64(resource.first_start_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_finish_ms\": {}",
            json_optional_f64(resource.last_finish_s * 1000.0)
        )?;
        if idx + 1 < utilization.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}
