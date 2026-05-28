use std::{
    fs::{File, OpenOptions},
    io::Write,
};

use super::*;

pub(super) fn initialize_serving_metrics_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_metrics_csv_path else {
        return Ok(());
    };
    write_serving_metrics_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_metrics_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_metrics_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_metrics_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_metrics_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_metrics_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,status,rejected_reason,deployment_mode,pool,objective,objective_score,objective_base_score,slo_miss_penalty_score,service_backpressure_penalty_score,topology_risk_penalty_score,prefill_nodes,decode_nodes,unique_node_count,unique_gpu_count,prefill_node_count,decode_node_count,shared_node_count,prefill_gpu_count,decode_gpu_count,shared_gpu_count,aggregate_hbm_gb,prefill_hbm_gb,decode_hbm_gb,aggregate_hbm_bandwidth_gb_s,prefill_hbm_bandwidth_gb_s,decode_hbm_bandwidth_gb_s,aggregate_effective_peak_tflops,prefill_effective_peak_tflops,decode_effective_peak_tflops,throughput_tokens_per_s_per_gpu,throughput_tokens_per_s_per_effective_peak_tflop,throughput_tokens_per_s_per_hbm_gb,aggregate_gpu_types,prefill_gpu_types,decode_gpu_types,aggregate_gpu_label_counts,prefill_gpu_label_counts,decode_gpu_label_counts,ttft_s,tpot_s,itl_s,e2el_s,throughput_tokens_per_s,ttft_calibration_uncertainty_s,ttft_calibration_lower_s,ttft_calibration_upper_s,tpot_calibration_uncertainty_s,tpot_calibration_lower_s,tpot_calibration_upper_s,itl_calibration_uncertainty_s,itl_calibration_lower_s,itl_calibration_upper_s,e2el_calibration_uncertainty_s,e2el_calibration_lower_s,e2el_calibration_upper_s,throughput_calibration_uncertainty_tokens_per_s,throughput_calibration_lower_tokens_per_s,throughput_calibration_upper_tokens_per_s,scheduled_makespan_s,admitted_requests,completed_requests,rejected_requests,timed_out_requests,cancelled_requests,measurement_window_request_count,measurement_window_completed_request_count,measurement_window_failed_request_count,measurement_window_rejected_request_count,measurement_window_timed_out_request_count,measurement_window_cancelled_request_count,measurement_window_deadline_constrained_request_count,measurement_window_deadline_missed_request_count,measured_requests,measurement_window_source,measurement_start_s,measurement_end_s,measurement_duration_s,lifecycle_event_metric_request_count,fallback_metric_request_count,metric_source_counts,prefill_s,prefill_queue_s,kv_transfer_s,kv_queue_s,decode_s,decode_queue_s,queue_delay_s,ttft_slo_constrained_requests,ttft_slo_missed_requests,ttft_slo_miss_rate,tpot_slo_constrained_requests,tpot_slo_missed_requests,tpot_slo_miss_rate,itl_slo_constrained_requests,itl_slo_missed_requests,itl_slo_miss_rate,e2el_slo_constrained_requests,e2el_slo_missed_requests,e2el_slo_miss_rate,deadline_constrained_requests,deadline_missed_requests,deadline_miss_rate,peak_prefill_tokens,peak_decode_sequences,peak_resident_tokens,peak_kv_blocks,calibration_status,calibration_active_phase_count,calibration_calibrated_phase_count,calibration_uncalibrated_phase_count,calibration_coverage_fraction,calibration_fit_count,calibration_extrapolated_fit_count,calibration_unbounded_fit_count,calibration_fit_count_with_uncertainty,calibration_min_confidence_score,calibration_max_extrapolation_ratio,calibration_relative_uncertainty_pct,calibration_absolute_uncertainty_s,calibration_gate_violation_count,calibration_hard_gate_violation_count,approximation_status,approximation_count,approximation_policy_violation_count,approximation_calibration_count,approximation_topology_count,approximation_queueing_count,approximation_runtime_count,approximation_memory_count,approximation_capacity_count,approximation_routing_count,approximation_admission_count,approximation_uncalibrated_phase_count,approximation_uncalibrated_queue_component_count,approximation_extrapolated_fit_count,approximation_coarse_topology,approximation_approximate_queueing,approximation_uncalibrated_runtime,approximation_category_counts,approximation_top_codes,topology_bottleneck_count,bottleneck_count,top_bottleneck_source,top_bottleneck_phase,top_bottleneck_category,top_bottleneck_resource,top_bottleneck_code,top_bottleneck_severity,top_bottleneck_unit,top_bottleneck_remediation,rejection_count,top_rejection_phase,top_rejection_category,top_rejection_resource,top_rejection_code,top_rejection_unit,top_rejection_remediation"
    )?;
    Ok(())
}

fn write_serving_metrics_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let metrics = &score.metrics;
        let footprint = &score.hardware_footprint;
        let overall_uncertainty = calibration_uncertainty_summary(score.calibration_fits.iter());
        let decode_uncertainty =
            calibration_phase_uncertainty_summary(score.calibration_fits.iter(), "decode");
        let top_bottleneck = score.bottleneck_summary.first();
        let top_rejection = score.rejections.first();
        let status = if score.feasible {
            "feasible"
        } else {
            "rejected"
        };
        write_csv_record(
            writer,
            &[
                csv_optional_str(scenario_name),
                (candidate_idx + 1).to_string(),
                csv_str(&score.candidate_id),
                score.feasible.to_string(),
                csv_str(status),
                csv_optional_str(score.rejected_reason.as_deref()),
                csv_str(score.deployment_mode.as_str()),
                csv_str(&pool_label(score)),
                csv_str(score.objective.as_str()),
                csv_f64(nominal_serving_objective_score(score)),
                csv_f64(serving_objective_base_score(score)),
                csv_f64(score.slo_miss_penalty_score),
                csv_f64(score.service_backpressure_penalty_score),
                csv_f64(score.topology_risk_penalty_score),
                csv_str(&node_list(&score.prefill_nodes)),
                csv_str(&node_list(&score.decode_nodes)),
                footprint.unique_node_count.to_string(),
                footprint.unique_gpu_count.to_string(),
                footprint.prefill_node_count.to_string(),
                footprint.decode_node_count.to_string(),
                footprint.shared_node_count.to_string(),
                footprint.prefill_gpu_count.to_string(),
                footprint.decode_gpu_count.to_string(),
                footprint.shared_gpu_count.to_string(),
                csv_f64(footprint.aggregate_hbm_gb),
                csv_f64(footprint.prefill_hbm_gb),
                csv_f64(footprint.decode_hbm_gb),
                csv_f64(footprint.aggregate_hbm_bandwidth_gb_s),
                csv_f64(footprint.prefill_hbm_bandwidth_gb_s),
                csv_f64(footprint.decode_hbm_bandwidth_gb_s),
                csv_f64(footprint.aggregate_effective_peak_tflops),
                csv_f64(footprint.prefill_effective_peak_tflops),
                csv_f64(footprint.decode_effective_peak_tflops),
                csv_f64(footprint.throughput_tokens_per_s_per_gpu),
                csv_f64(footprint.throughput_tokens_per_s_per_effective_peak_tflop),
                csv_f64(footprint.throughput_tokens_per_s_per_hbm_gb),
                csv_str(&serving_gpu_type_count_list(&footprint.aggregate_gpu_types)),
                csv_str(&serving_gpu_type_count_list(&footprint.prefill_gpu_types)),
                csv_str(&serving_gpu_type_count_list(&footprint.decode_gpu_types)),
                csv_str(&serving_gpu_label_count_list(
                    &footprint.aggregate_gpu_label_counts,
                )),
                csv_str(&serving_gpu_label_count_list(
                    &footprint.prefill_gpu_label_counts,
                )),
                csv_str(&serving_gpu_label_count_list(
                    &footprint.decode_gpu_label_counts,
                )),
                csv_f64(metrics.ttft_s),
                csv_f64(metrics.tpot_s),
                csv_f64(metrics.itl_s),
                csv_f64(metrics.e2el_s),
                csv_f64(metrics.throughput_tokens_per_s),
                csv_metric_uncertainty_s(score.feasible, metrics.ttft_s, &overall_uncertainty),
                csv_metric_lower_s(score.feasible, metrics.ttft_s, &overall_uncertainty),
                csv_metric_upper_s(score.feasible, metrics.ttft_s, &overall_uncertainty),
                csv_metric_uncertainty_s(score.feasible, metrics.tpot_s, &decode_uncertainty),
                csv_metric_lower_s(score.feasible, metrics.tpot_s, &decode_uncertainty),
                csv_metric_upper_s(score.feasible, metrics.tpot_s, &decode_uncertainty),
                csv_metric_uncertainty_s(score.feasible, metrics.itl_s, &decode_uncertainty),
                csv_metric_lower_s(score.feasible, metrics.itl_s, &decode_uncertainty),
                csv_metric_upper_s(score.feasible, metrics.itl_s, &decode_uncertainty),
                csv_metric_uncertainty_s(score.feasible, metrics.e2el_s, &overall_uncertainty),
                csv_metric_lower_s(score.feasible, metrics.e2el_s, &overall_uncertainty),
                csv_metric_upper_s(score.feasible, metrics.e2el_s, &overall_uncertainty),
                csv_metric_relative_uncertainty(
                    score.feasible,
                    metrics.throughput_tokens_per_s,
                    &overall_uncertainty,
                ),
                csv_metric_relative_lower(
                    score.feasible,
                    metrics.throughput_tokens_per_s,
                    &overall_uncertainty,
                ),
                csv_metric_relative_upper(
                    score.feasible,
                    metrics.throughput_tokens_per_s,
                    &overall_uncertainty,
                ),
                csv_f64(metrics.scheduled_makespan_s),
                metrics.admitted_requests.to_string(),
                metrics.completed_requests.to_string(),
                metrics.rejected_requests.to_string(),
                metrics.timed_out_requests.to_string(),
                metrics.cancelled_requests.to_string(),
                score.measurement_window.request_count.to_string(),
                score.measurement_window.completed_request_count.to_string(),
                score.measurement_window.failed_request_count.to_string(),
                score.measurement_window.rejected_request_count.to_string(),
                score.measurement_window.timed_out_request_count.to_string(),
                score.measurement_window.cancelled_request_count.to_string(),
                score
                    .measurement_window
                    .deadline_constrained_request_count
                    .to_string(),
                score
                    .measurement_window
                    .deadline_missed_request_count
                    .to_string(),
                metrics.measured_requests.to_string(),
                csv_str(&score.measurement_window.source),
                csv_f64(score.measurement_window.start_s),
                csv_f64(score.measurement_window.end_s),
                csv_f64(score.measurement_window.duration_s),
                score
                    .measurement_window
                    .lifecycle_event_metric_request_count
                    .to_string(),
                score
                    .measurement_window
                    .fallback_metric_request_count
                    .to_string(),
                csv_str(&metric_source_count_list(
                    &score.measurement_window.metric_source_counts,
                )),
                csv_f64(metrics.prefill_s),
                csv_f64(metrics.prefill_worker_queue_s + metrics.prefill_resource_queue_s),
                csv_f64(metrics.kv_transfer_s),
                csv_f64(metrics.kv_queue_s),
                csv_f64(metrics.decode_s),
                csv_f64(metrics.decode_queue_s),
                csv_f64(metrics.queue_delay_s),
                metrics.ttft_slo_constrained_requests.to_string(),
                metrics.ttft_slo_missed_requests.to_string(),
                csv_f64(metrics.ttft_slo_miss_rate),
                metrics.tpot_slo_constrained_requests.to_string(),
                metrics.tpot_slo_missed_requests.to_string(),
                csv_f64(metrics.tpot_slo_miss_rate),
                metrics.itl_slo_constrained_requests.to_string(),
                metrics.itl_slo_missed_requests.to_string(),
                csv_f64(metrics.itl_slo_miss_rate),
                metrics.e2el_slo_constrained_requests.to_string(),
                metrics.e2el_slo_missed_requests.to_string(),
                csv_f64(metrics.e2el_slo_miss_rate),
                metrics.deadline_constrained_requests.to_string(),
                metrics.deadline_missed_requests.to_string(),
                csv_f64(metrics.deadline_miss_rate),
                metrics.peak_prefill_tokens.to_string(),
                metrics.peak_decode_sequences.to_string(),
                metrics.peak_resident_tokens.to_string(),
                metrics.peak_kv_blocks.to_string(),
                csv_str(&score.calibration_summary.status),
                score.calibration_summary.active_phase_count.to_string(),
                score.calibration_summary.calibrated_phase_count.to_string(),
                score
                    .calibration_summary
                    .uncalibrated_phase_count
                    .to_string(),
                csv_f64(score.calibration_summary.coverage_fraction),
                score.calibration_summary.fit_count.to_string(),
                score.calibration_summary.extrapolated_fit_count.to_string(),
                score.calibration_summary.unbounded_fit_count.to_string(),
                score
                    .calibration_summary
                    .fit_count_with_uncertainty
                    .to_string(),
                csv_optional_f64(score.calibration_summary.min_confidence_score),
                csv_optional_f64(score.calibration_summary.max_extrapolation_ratio),
                csv_optional_f64(score.calibration_summary.relative_uncertainty_pct),
                csv_optional_f64(score.calibration_summary.absolute_uncertainty_s),
                score.calibration_summary.gate_violation_count.to_string(),
                score
                    .calibration_summary
                    .hard_gate_violation_count
                    .to_string(),
                csv_str(&score.approximation_summary.status),
                score.approximation_summary.approximation_count.to_string(),
                score
                    .approximation_summary
                    .policy_violation_count
                    .to_string(),
                score.approximation_summary.calibration_count.to_string(),
                score.approximation_summary.topology_count.to_string(),
                score.approximation_summary.queueing_count.to_string(),
                score.approximation_summary.runtime_count.to_string(),
                score.approximation_summary.memory_count.to_string(),
                score.approximation_summary.capacity_count.to_string(),
                score.approximation_summary.routing_count.to_string(),
                score.approximation_summary.admission_count.to_string(),
                score
                    .approximation_summary
                    .uncalibrated_phase_count
                    .to_string(),
                score
                    .approximation_summary
                    .uncalibrated_queue_component_count
                    .to_string(),
                score
                    .approximation_summary
                    .extrapolated_fit_count
                    .to_string(),
                score.approximation_summary.coarse_topology.to_string(),
                score.approximation_summary.approximate_queueing.to_string(),
                score.approximation_summary.uncalibrated_runtime.to_string(),
                csv_str(&serving_approximation_count_list(
                    &score.approximation_summary.category_counts,
                )),
                csv_str(&serving_approximation_count_list(
                    &score.approximation_summary.top_codes,
                )),
                score.topology_bottlenecks.len().to_string(),
                score.bottleneck_summary.len().to_string(),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.source.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.phase.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.category.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.resource.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.code.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.severity.as_str())),
                csv_optional_str(top_bottleneck.and_then(|bottleneck| bottleneck.unit.as_deref())),
                csv_optional_str(
                    top_bottleneck.and_then(|bottleneck| bottleneck.remediation.as_deref()),
                ),
                score.rejections.len().to_string(),
                csv_optional_str(top_rejection.map(|rejection| rejection.phase.as_str())),
                csv_optional_str(top_rejection.map(|rejection| rejection.category.as_str())),
                csv_optional_str(top_rejection.map(|rejection| rejection.resource.as_str())),
                csv_optional_str(top_rejection.map(|rejection| rejection.code.as_str())),
                csv_optional_str(top_rejection.and_then(|rejection| rejection.unit.as_deref())),
                csv_optional_str(
                    top_rejection.and_then(|rejection| rejection.remediation.as_deref()),
                ),
            ],
        )?;
    }
    Ok(())
}

pub(super) fn initialize_serving_metric_breakdowns_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_metric_breakdowns_csv_path else {
        return Ok(());
    };
    write_serving_metric_breakdowns_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_metric_breakdowns_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_metric_breakdowns_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_metric_breakdowns_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_metric_breakdowns_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_metric_breakdowns_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,group,key,request_count,completed_requests,failed_requests,rejected_requests,timed_out_requests,cancelled_requests,output_tokens,throughput_tokens_per_s,lifecycle_event_metric_request_count,fallback_metric_request_count,metric_source_counts,deadline_constrained_requests,deadline_missed_requests,deadline_miss_rate,ttft_s,tpot_s,itl_s,e2el_s,ttft_p90_s,tpot_p90_s,itl_p90_s,e2el_p90_s,ttft_p95_s,tpot_p95_s,itl_p95_s,e2el_p95_s,ttft_max_s,tpot_max_s,itl_max_s,e2el_max_s,ttft_slo_constrained_requests,ttft_slo_missed_requests,ttft_slo_miss_rate,tpot_slo_constrained_requests,tpot_slo_missed_requests,tpot_slo_miss_rate,itl_slo_constrained_requests,itl_slo_missed_requests,itl_slo_miss_rate,e2el_slo_constrained_requests,e2el_slo_missed_requests,e2el_slo_miss_rate"
    )?;
    Ok(())
}

fn write_serving_metric_breakdowns_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        for breakdown in &score.metric_breakdowns {
            write_serving_metric_breakdown_csv_row(
                writer,
                scenario_name,
                candidate_idx + 1,
                score,
                breakdown,
            )?;
        }
    }
    Ok(())
}

fn write_serving_metric_breakdown_csv_row<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    breakdown: &ServingMetricBreakdown,
) -> Result<(), CliError> {
    write_csv_record(
        writer,
        &[
            csv_optional_str(scenario_name),
            candidate_rank.to_string(),
            csv_str(&score.candidate_id),
            score.feasible.to_string(),
            csv_str(score.deployment_mode.as_str()),
            csv_str(&pool_label(score)),
            csv_str(&breakdown.group),
            csv_str(&breakdown.key),
            breakdown.request_count.to_string(),
            breakdown.completed_requests.to_string(),
            breakdown.failed_requests.to_string(),
            breakdown.rejected_requests.to_string(),
            breakdown.timed_out_requests.to_string(),
            breakdown.cancelled_requests.to_string(),
            breakdown.output_tokens.to_string(),
            csv_f64(breakdown.throughput_tokens_per_s),
            breakdown.lifecycle_event_metric_request_count.to_string(),
            breakdown.fallback_metric_request_count.to_string(),
            csv_str(&metric_source_count_list(&breakdown.metric_source_counts)),
            breakdown.deadline_constrained_requests.to_string(),
            breakdown.deadline_missed_requests.to_string(),
            csv_f64(breakdown.deadline_miss_rate),
            csv_f64(breakdown.ttft_s),
            csv_f64(breakdown.tpot_s),
            csv_f64(breakdown.itl_s),
            csv_f64(breakdown.e2el_s),
            csv_f64(breakdown.ttft_p90_s),
            csv_f64(breakdown.tpot_p90_s),
            csv_f64(breakdown.itl_p90_s),
            csv_f64(breakdown.e2el_p90_s),
            csv_f64(breakdown.ttft_p95_s),
            csv_f64(breakdown.tpot_p95_s),
            csv_f64(breakdown.itl_p95_s),
            csv_f64(breakdown.e2el_p95_s),
            csv_f64(breakdown.ttft_max_s),
            csv_f64(breakdown.tpot_max_s),
            csv_f64(breakdown.itl_max_s),
            csv_f64(breakdown.e2el_max_s),
            breakdown.ttft_slo_constrained_requests.to_string(),
            breakdown.ttft_slo_missed_requests.to_string(),
            csv_f64(breakdown.ttft_slo_miss_rate),
            breakdown.tpot_slo_constrained_requests.to_string(),
            breakdown.tpot_slo_missed_requests.to_string(),
            csv_f64(breakdown.tpot_slo_miss_rate),
            breakdown.itl_slo_constrained_requests.to_string(),
            breakdown.itl_slo_missed_requests.to_string(),
            csv_f64(breakdown.itl_slo_miss_rate),
            breakdown.e2el_slo_constrained_requests.to_string(),
            breakdown.e2el_slo_missed_requests.to_string(),
            csv_f64(breakdown.e2el_slo_miss_rate),
        ],
    )
}

pub(super) fn initialize_rank_sensitivity_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.rank_sensitivity_csv_path else {
        return Ok(());
    };
    write_rank_sensitivity_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_rank_sensitivity_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    policy: &CalibrationPolicy,
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.rank_sensitivity_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_rank_sensitivity_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_rank_sensitivity_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
        policy.uncertainty_ranking_weight,
    )
}

pub(super) fn write_parallelism_rank_sensitivity_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredParallelismConfig],
    policy: &CalibrationPolicy,
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.rank_sensitivity_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_rank_sensitivity_csv_header(&mut file)?;
        Box::new(file)
    };
    write_parallelism_rank_sensitivity_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
        policy.uncertainty_ranking_weight,
    )
}

fn write_rank_sensitivity_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,mode,candidate_rank,candidate_id,feasible,status,nominal_rank,uncertainty_adjusted_rank,uncertainty_rank_delta,nominal_score,uncertainty_adjusted_score,uncertainty_ranking_weight,calibration_status,calibration_fit_count,calibration_fit_count_with_uncertainty,calibration_min_confidence_score,calibration_max_extrapolation_ratio,calibration_applicability_status,calibration_relative_uncertainty_pct,calibration_absolute_uncertainty_s,approximation_status,approximation_count,approximation_policy_violation_count,approximation_coarse_topology,approximation_approximate_queueing,approximation_aggregate_memory,approximation_uncalibrated_runtime,approximation_unsupported_runtime,approximation_top_codes,rejection_count,top_rejection_phase,top_rejection_category,top_rejection_resource,top_rejection_code,top_rejection_unit,top_rejection_remediation,bottleneck_count,top_bottleneck_source,top_bottleneck_category,top_bottleneck_code,top_bottleneck_severity,hardware_unique_node_count,hardware_unique_gpu_count,hardware_prefill_gpu_count,hardware_decode_gpu_count,hardware_shared_gpu_count,hardware_aggregate_hbm_gb,hardware_prefill_hbm_gb,hardware_decode_hbm_gb,hardware_aggregate_effective_peak_tflops,hardware_prefill_effective_peak_tflops,hardware_decode_effective_peak_tflops,hardware_aggregate_gpu_types,hardware_prefill_gpu_types,hardware_decode_gpu_types,hardware_aggregate_gpu_label_counts,hardware_prefill_gpu_label_counts,hardware_decode_gpu_label_counts,hardware_throughput_tokens_per_s_per_gpu,hardware_throughput_tokens_per_s_per_effective_peak_tflop,hardware_throughput_tokens_per_s_per_hbm_gb,objective,deployment_mode,pool,estimated_latency_s,memory_per_gpu_gb,tp,pp,ep,dp,ttft_s,tpot_s,itl_s,e2el_s,throughput_tokens_per_s"
    )?;
    Ok(())
}

fn write_serving_rank_sensitivity_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
    uncertainty_ranking_weight: f64,
) -> Result<(), CliError> {
    let nominal_ranks = serving_nominal_rank_map(results);
    let uncertainty_adjusted_ranks =
        serving_uncertainty_adjusted_rank_map(results, uncertainty_ranking_weight);
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let calibration = calibration_uncertainty_summary(score.calibration_fits.iter());
        let approximations = rank_sensitivity_approximation_summary(
            &score.approximations,
            &score.approximation_policy_violations,
        );
        let top_rejection = score.rejections.first();
        let top_bottleneck = score.bottleneck_summary.first();
        let footprint = &score.hardware_footprint;
        let nominal_rank = *nominal_ranks
            .get(&score.candidate_id)
            .unwrap_or(&(candidate_idx + 1));
        let uncertainty_adjusted_rank = *uncertainty_adjusted_ranks
            .get(&score.candidate_id)
            .unwrap_or(&(candidate_idx + 1));
        write_csv_record(
            writer,
            &[
                csv_optional_str(scenario_name),
                csv_str("serving"),
                (candidate_idx + 1).to_string(),
                csv_str(&score.candidate_id),
                score.feasible.to_string(),
                csv_str(if score.feasible {
                    "feasible"
                } else {
                    "rejected"
                }),
                nominal_rank.to_string(),
                uncertainty_adjusted_rank.to_string(),
                ((uncertainty_adjusted_rank as i64) - (nominal_rank as i64)).to_string(),
                csv_f64(nominal_serving_objective_score(score)),
                csv_f64(uncertainty_adjusted_serving_objective_score(
                    score,
                    uncertainty_ranking_weight,
                )),
                csv_f64(uncertainty_ranking_weight),
                csv_str(&score.calibration_summary.status),
                score.calibration_summary.fit_count.to_string(),
                calibration.fit_count_with_uncertainty.to_string(),
                csv_optional_f64(calibration.min_confidence_score),
                csv_optional_f64(calibration.max_extrapolation_ratio),
                csv_str(&calibration.applicability_status),
                csv_optional_f64(score.calibration_summary.relative_uncertainty_pct),
                csv_optional_f64(score.calibration_summary.absolute_uncertainty_s),
                csv_str(&approximations.status),
                approximations.approximation_count.to_string(),
                approximations.policy_violation_count.to_string(),
                approximations.coarse_topology.to_string(),
                approximations.approximate_queueing.to_string(),
                approximations.aggregate_memory.to_string(),
                approximations.uncalibrated_runtime.to_string(),
                approximations.unsupported_runtime.to_string(),
                csv_str(&approximations.top_codes),
                score.rejections.len().to_string(),
                csv_optional_str(top_rejection.map(|rejection| rejection.phase.as_str())),
                csv_optional_str(top_rejection.map(|rejection| rejection.category.as_str())),
                csv_optional_str(top_rejection.map(|rejection| rejection.resource.as_str())),
                csv_optional_str(top_rejection.map(|rejection| rejection.code.as_str())),
                csv_optional_str(top_rejection.and_then(|rejection| rejection.unit.as_deref())),
                csv_optional_str(
                    top_rejection.and_then(|rejection| rejection.remediation.as_deref()),
                ),
                score.bottleneck_summary.len().to_string(),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.source.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.category.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.code.as_str())),
                csv_optional_str(top_bottleneck.map(|bottleneck| bottleneck.severity.as_str())),
                footprint.unique_node_count.to_string(),
                footprint.unique_gpu_count.to_string(),
                footprint.prefill_gpu_count.to_string(),
                footprint.decode_gpu_count.to_string(),
                footprint.shared_gpu_count.to_string(),
                csv_f64(footprint.aggregate_hbm_gb),
                csv_f64(footprint.prefill_hbm_gb),
                csv_f64(footprint.decode_hbm_gb),
                csv_f64(footprint.aggregate_effective_peak_tflops),
                csv_f64(footprint.prefill_effective_peak_tflops),
                csv_f64(footprint.decode_effective_peak_tflops),
                csv_str(&serving_gpu_type_count_list(&footprint.aggregate_gpu_types)),
                csv_str(&serving_gpu_type_count_list(&footprint.prefill_gpu_types)),
                csv_str(&serving_gpu_type_count_list(&footprint.decode_gpu_types)),
                csv_str(&serving_gpu_label_count_list(
                    &footprint.aggregate_gpu_label_counts,
                )),
                csv_str(&serving_gpu_label_count_list(
                    &footprint.prefill_gpu_label_counts,
                )),
                csv_str(&serving_gpu_label_count_list(
                    &footprint.decode_gpu_label_counts,
                )),
                csv_f64(footprint.throughput_tokens_per_s_per_gpu),
                csv_f64(footprint.throughput_tokens_per_s_per_effective_peak_tflop),
                csv_f64(footprint.throughput_tokens_per_s_per_hbm_gb),
                csv_str(score.objective.as_str()),
                csv_str(score.deployment_mode.as_str()),
                csv_str(&pool_label(score)),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                csv_f64(score.metrics.ttft_s),
                csv_f64(score.metrics.tpot_s),
                csv_f64(score.metrics.itl_s),
                csv_f64(score.metrics.e2el_s),
                csv_f64(score.metrics.throughput_tokens_per_s),
            ],
        )?;
    }
    Ok(())
}

fn write_parallelism_rank_sensitivity_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredParallelismConfig],
    displayed_results: usize,
    uncertainty_ranking_weight: f64,
) -> Result<(), CliError> {
    let nominal_ranks = parallelism_nominal_rank_map(results);
    let uncertainty_adjusted_ranks =
        parallelism_uncertainty_adjusted_rank_map(results, uncertainty_ranking_weight);
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_id = parallelism_candidate_id(score);
        let calibration = calibration_uncertainty_summary(score.calibration_fits.iter());
        let approximations = rank_sensitivity_approximation_summary(
            &score.approximations,
            &score.approximation_policy_violations,
        );
        let nominal_rank = *nominal_ranks
            .get(&candidate_id)
            .unwrap_or(&(candidate_idx + 1));
        let uncertainty_adjusted_rank = *uncertainty_adjusted_ranks
            .get(&candidate_id)
            .unwrap_or(&(candidate_idx + 1));
        write_csv_record(
            writer,
            &[
                csv_optional_str(scenario_name),
                csv_str("parallelism"),
                (candidate_idx + 1).to_string(),
                csv_str(&candidate_id),
                score.feasible.to_string(),
                csv_str(if score.feasible {
                    "feasible"
                } else {
                    "rejected"
                }),
                nominal_rank.to_string(),
                uncertainty_adjusted_rank.to_string(),
                ((uncertainty_adjusted_rank as i64) - (nominal_rank as i64)).to_string(),
                csv_f64(score.estimated_latency_s),
                csv_f64(uncertainty_adjusted_parallelism_latency_s(
                    score,
                    uncertainty_ranking_weight,
                )),
                csv_f64(uncertainty_ranking_weight),
                csv_str(&calibration.applicability_status),
                calibration.fit_count.to_string(),
                calibration.fit_count_with_uncertainty.to_string(),
                csv_optional_f64(calibration.min_confidence_score),
                csv_optional_f64(calibration.max_extrapolation_ratio),
                csv_str(&calibration.applicability_status),
                csv_optional_f64(calibration.relative_uncertainty_pct),
                csv_optional_f64(calibration.absolute_uncertainty_s),
                csv_str(&approximations.status),
                approximations.approximation_count.to_string(),
                approximations.policy_violation_count.to_string(),
                approximations.coarse_topology.to_string(),
                approximations.approximate_queueing.to_string(),
                approximations.aggregate_memory.to_string(),
                approximations.uncalibrated_runtime.to_string(),
                approximations.unsupported_runtime.to_string(),
                csv_str(&approximations.top_codes),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                csv_f64(score.estimated_latency_s),
                csv_f64(score.estimated_memory_per_gpu.as_gigabytes()),
                score.config.tensor_ranks.to_string(),
                score.config.pipeline_ranks.to_string(),
                score.config.expert_ranks.to_string(),
                score.config.data_ranks.to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ],
        )?;
    }
    Ok(())
}

pub(super) fn initialize_serving_services_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_services_csv_path else {
        return Ok(());
    };
    write_serving_services_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_services_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_services_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_services_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_services_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_services_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,phase,health,accepts_requests,worker_scale,configured_worker_slots_per_gpu,effective_worker_slots_per_gpu,node_count,gpu_count,request_count,admitted_requests,completed_requests,failed_requests,rejected_requests,timed_out_requests,cancelled_requests,queue_cap_s,queue_cap_request_count,queue_cap_hit_count,decode_iteration_queue_cap_s,decode_iteration_queue_cap_request_count,decode_iteration_queue_cap_hit_count,backpressure_rejections,timeout_rejections,backpressure_state,worker_slot_utilization,queue_s,queue_p95_s,queue_max_s,worker_queue_s,resource_queue_s,service_s"
    )?;
    Ok(())
}

fn write_serving_services_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for observation in &score.service_observations {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    csv_str(&observation.phase),
                    csv_str(observation.health.as_str()),
                    observation.accepts_requests.to_string(),
                    csv_f64(observation.worker_scale),
                    observation.configured_worker_slots_per_gpu.to_string(),
                    observation.effective_worker_slots_per_gpu.to_string(),
                    observation.node_count.to_string(),
                    observation.gpu_count.to_string(),
                    observation.request_count.to_string(),
                    observation.admitted_requests.to_string(),
                    observation.completed_requests.to_string(),
                    observation.failed_requests.to_string(),
                    observation.rejected_requests.to_string(),
                    observation.timed_out_requests.to_string(),
                    observation.cancelled_requests.to_string(),
                    csv_optional_f64(observation.queue_cap_s),
                    observation.queue_cap_request_count.to_string(),
                    observation.queue_cap_hit_count.to_string(),
                    csv_optional_f64(observation.decode_iteration_queue_cap_s),
                    observation
                        .decode_iteration_queue_cap_request_count
                        .to_string(),
                    observation.decode_iteration_queue_cap_hit_count.to_string(),
                    observation.backpressure_rejections.to_string(),
                    observation.timeout_rejections.to_string(),
                    csv_str(&observation.backpressure_state),
                    csv_f64(observation.worker_slot_utilization),
                    csv_f64(observation.queue_s),
                    csv_f64(observation.queue_p95_s),
                    csv_f64(observation.queue_max_s),
                    csv_f64(observation.worker_queue_s),
                    csv_f64(observation.resource_queue_s),
                    csv_f64(observation.service_s),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_utilization_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_utilization_csv_path else {
        return Ok(());
    };
    write_serving_utilization_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_utilization_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_utilization_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_utilization_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_utilization_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_utilization_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,source,phase,resource_kind,resource,busy_s,utilization,operation_count,first_start_s,last_finish_s"
    )?;
    Ok(())
}

fn write_serving_utilization_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for resource in &score.phase_resource_utilization {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    csv_str("phase_resource"),
                    csv_str(&resource.phase),
                    csv_str(&resource.resource_kind),
                    csv_str(&resource.resource),
                    csv_f64(resource.busy_s),
                    csv_f64(resource.utilization),
                    resource.operation_count.to_string(),
                    csv_f64(resource.first_start_s),
                    csv_f64(resource.last_finish_s),
                ],
            )?;
        }
        for resource in &score.resource_utilization {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    csv_str("scheduled_resource"),
                    csv_str(""),
                    csv_str("scheduled"),
                    csv_str(&resource.resource),
                    csv_f64(resource.busy_s),
                    csv_f64(resource.utilization),
                    resource.operation_count.to_string(),
                    csv_f64(resource.first_start_s),
                    csv_f64(resource.last_finish_s),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_memory_pressure_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_memory_pressure_csv_path else {
        return Ok(());
    };
    write_serving_memory_pressure_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_memory_pressure_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_memory_pressure_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_memory_pressure_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_memory_pressure_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_memory_pressure_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,observation_idx,phase,estimate_kind,start_s,finish_s,duration_s,active_requests,active_tokens,kv_blocks,estimated_per_gpu_gb,min_hbm_per_gpu_gb,capacity_used_fraction,headroom_gb,limiting_node,limiting_gpu,dominant_component,dominant_component_gb,dominant_component_fraction,weights_gb,kv_cache_gb,block_table_gb,activations_gb,temporary_gb,communication_gb,runtime_reserve_gb,fragmentation_gb,total_gb"
    )?;
    Ok(())
}

fn write_serving_memory_pressure_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for (observation_idx, observation) in score.memory_pressure.iter().enumerate() {
            let components = observation.components;
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    (observation_idx + 1).to_string(),
                    csv_str(&observation.phase),
                    csv_str(&observation.estimate_kind),
                    csv_f64(observation.start_s),
                    csv_f64(observation.finish_s),
                    csv_f64(observation.duration_s),
                    observation.active_requests.to_string(),
                    observation.active_tokens.to_string(),
                    observation.kv_blocks.to_string(),
                    csv_f64(observation.estimated_per_gpu_gb),
                    csv_f64(observation.min_hbm_per_gpu_gb),
                    csv_f64(observation.capacity_used_fraction),
                    csv_f64(observation.headroom_gb),
                    csv_optional_u32(observation.limiting_gpu.map(|gpu| gpu.node_id)),
                    csv_optional_u32(observation.limiting_gpu.map(|gpu| gpu.local_gpu_id)),
                    csv_optional_str(
                        observation
                            .dominant_component
                            .as_ref()
                            .map(|component| component.name),
                    ),
                    csv_optional_f64(observation.dominant_component.map(|component| component.gb)),
                    csv_optional_f64(
                        observation
                            .dominant_component
                            .map(|component| component.fraction_of_total),
                    ),
                    csv_f64(components.weights_gb),
                    csv_f64(components.kv_cache_gb),
                    csv_f64(components.block_table_gb),
                    csv_f64(components.activations_gb),
                    csv_f64(components.temporary_gb),
                    csv_f64(components.communication_gb),
                    csv_f64(components.runtime_reserve_gb),
                    csv_f64(components.fragmentation_gb),
                    csv_f64(components.total_gb),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_timeline_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_timeline_csv_path else {
        return Ok(());
    };
    write_serving_timeline_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_timeline_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_timeline_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_timeline_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_timeline_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_timeline_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,operation_idx,operation_id,phase,name,resource_count,resources,start_s,finish_s,duration_s,explicit_dependencies,resource_dependencies,dependency_count"
    )?;
    Ok(())
}

fn write_serving_timeline_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for (operation_idx, operation) in score.scheduled_operations.iter().enumerate() {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    (operation_idx + 1).to_string(),
                    operation.id.to_string(),
                    csv_str(scheduled_operation_phase(&operation.name)),
                    csv_str(&operation.name),
                    operation.resources.len().to_string(),
                    csv_str(&operation.resources.join("|")),
                    csv_f64(operation.start_s),
                    csv_f64(operation.finish_s),
                    csv_f64((operation.finish_s - operation.start_s).max(0.0)),
                    csv_str(&usize_list(&operation.explicit_dependencies)),
                    csv_str(&usize_list(&operation.resource_dependencies)),
                    (operation.explicit_dependencies.len() + operation.resource_dependencies.len())
                        .to_string(),
                ],
            )?;
        }
    }
    Ok(())
}

fn scheduled_operation_phase(name: &str) -> &'static str {
    let normalized = name.to_ascii_lowercase();
    if normalized.contains("prefill") {
        "prefill"
    } else if normalized.contains("kv-transfer") || normalized.contains("kv transfer") {
        "kv_transfer"
    } else if normalized.contains("decode") {
        "decode"
    } else {
        "other"
    }
}

pub(super) fn initialize_serving_occupancy_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_occupancy_csv_path else {
        return Ok(());
    };
    write_serving_occupancy_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_occupancy_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_occupancy_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_occupancy_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_occupancy_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
        args.occupancy_buckets,
        args.occupancy_resource_limit,
    )
}

fn write_serving_occupancy_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,resource_idx,resource,bucket_idx,start_s,finish_s,busy_s,utilization,operation_count"
    )?;
    Ok(())
}

fn write_serving_occupancy_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
    bucket_count: usize,
    resource_limit: Option<usize>,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        let resources = selected_occupancy_resources(&score.resource_utilization, resource_limit);
        let occupancy = resource_occupancy_buckets(
            &score.scheduled_operations,
            score.metrics.scheduled_makespan_s,
            bucket_count,
            &resources,
        );
        for (resource_idx, series) in occupancy.iter().enumerate() {
            for bucket in &series.buckets {
                write_csv_record(
                    writer,
                    &[
                        csv_optional_str(scenario_name),
                        candidate_rank.to_string(),
                        csv_str(&score.candidate_id),
                        score.feasible.to_string(),
                        csv_str(score.deployment_mode.as_str()),
                        csv_str(&pool),
                        (resource_idx + 1).to_string(),
                        csv_str(&series.resource),
                        bucket.bucket_idx.to_string(),
                        csv_f64(bucket.start_s),
                        csv_f64(bucket.finish_s),
                        csv_f64(bucket.busy_s),
                        csv_f64(bucket.utilization),
                        bucket.operation_count.to_string(),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_placement_evidence_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_placement_evidence_csv_path else {
        return Ok(());
    };
    write_serving_placement_evidence_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_placement_evidence_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_placement_evidence_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_placement_evidence_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_placement_evidence_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_placement_evidence_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,phase,evidence_idx,decision,scope,resource,code,observed,limit,unit,message,remediation"
    )?;
    Ok(())
}

fn write_serving_placement_evidence_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        write_serving_placement_evidence_phase_rows(
            writer,
            scenario_name,
            score,
            candidate_rank,
            &pool,
            "prefill",
            &score.prefill_score.placement_evidence,
        )?;
        write_serving_placement_evidence_phase_rows(
            writer,
            scenario_name,
            score,
            candidate_rank,
            &pool,
            "decode",
            &score.decode_score.placement_evidence,
        )?;
    }
    Ok(())
}

fn write_serving_placement_evidence_phase_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    score: &ScoredServingConfig,
    candidate_rank: usize,
    pool: &str,
    phase: &str,
    evidence_items: &[PlacementEvidence],
) -> Result<(), CliError> {
    for (evidence_idx, evidence) in evidence_items.iter().enumerate() {
        write_csv_record(
            writer,
            &[
                csv_optional_str(scenario_name),
                candidate_rank.to_string(),
                csv_str(&score.candidate_id),
                score.feasible.to_string(),
                csv_str(score.deployment_mode.as_str()),
                csv_str(pool),
                csv_str(phase),
                (evidence_idx + 1).to_string(),
                csv_str(&evidence.decision),
                csv_str(&evidence.scope),
                csv_str(&evidence.resource),
                csv_str(&evidence.code),
                csv_optional_f64(evidence.observed),
                csv_optional_f64(evidence.limit),
                csv_optional_str(evidence.unit.as_deref()),
                csv_str(&evidence.message),
                csv_optional_str(evidence.remediation.as_deref()),
            ],
        )?;
    }
    Ok(())
}

pub(super) fn initialize_serving_worker_evidence_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_worker_evidence_csv_path else {
        return Ok(());
    };
    write_serving_worker_evidence_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_worker_evidence_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_worker_evidence_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_worker_evidence_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_worker_evidence_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_worker_evidence_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,request_idx,request_id,tenant,model_id,traffic_class,shape_profile,status,record_kind,record_idx,role,phase,node_id,local_gpu_id,slot,worker_slots,operation_ids,assignment_count,start_s,finish_s,duration_s,allocation_id,owner_worker_slots,block_start,block_end,decode_sequences,resident_tokens,kv_blocks,allocated_kv_tokens,kv_fragmentation_tokens,block_table_entries,block_table_bytes"
    )?;
    Ok(())
}

fn write_serving_worker_evidence_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for observation in &score.request_observations {
            for (record_idx, summary) in observation.worker_summary.iter().enumerate() {
                write_serving_worker_summary_csv_row(
                    writer,
                    scenario_name,
                    candidate_rank,
                    score,
                    &pool,
                    observation,
                    record_idx + 1,
                    summary,
                )?;
            }
            for (record_idx, assignment) in observation.worker_assignments.iter().enumerate() {
                write_serving_worker_assignment_csv_row(
                    writer,
                    scenario_name,
                    candidate_rank,
                    score,
                    &pool,
                    observation,
                    record_idx + 1,
                    assignment,
                )?;
            }
            for (record_idx, ownership) in observation.kv_block_ownership.iter().enumerate() {
                write_serving_kv_ownership_csv_row(
                    writer,
                    scenario_name,
                    candidate_rank,
                    score,
                    &pool,
                    observation,
                    record_idx + 1,
                    ownership,
                )?;
                for (slot_idx, slot) in ownership.worker_slot_ownership.iter().enumerate() {
                    write_serving_kv_slot_ownership_csv_row(
                        writer,
                        scenario_name,
                        candidate_rank,
                        score,
                        &pool,
                        observation,
                        slot_idx + 1,
                        ownership,
                        slot,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_serving_worker_summary_csv_row<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    record_idx: usize,
    summary: &ServingRequestWorkerSummaryObservation,
) -> Result<(), CliError> {
    write_serving_worker_evidence_csv_record(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        "worker_summary",
        record_idx,
        Some(&summary.role),
        &summary.phase,
        Some(summary.node_id),
        Some(summary.local_gpu_id),
        None,
        &csv_str(&u32_list(&summary.worker_slots)),
        &csv_str(&usize_list(&summary.operation_ids)),
        Some(summary.assignment_count),
        Some(summary.start_s),
        Some(summary.finish_s),
        Some((summary.finish_s - summary.start_s).max(0.0)),
        None,
        "",
        None,
        None,
        None,
        Some(summary.resident_tokens),
        Some(summary.kv_blocks),
        None,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_serving_worker_assignment_csv_row<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    record_idx: usize,
    assignment: &ServingWorkerAssignmentObservation,
) -> Result<(), CliError> {
    write_serving_worker_evidence_csv_record(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        "worker_assignment",
        record_idx,
        None,
        &assignment.phase,
        Some(assignment.node_id),
        Some(assignment.local_gpu_id),
        Some(assignment.slot),
        "",
        &csv_str(&usize_list(&assignment.operation_ids)),
        None,
        Some(assignment.start_s),
        Some(assignment.finish_s),
        Some((assignment.finish_s - assignment.start_s).max(0.0)),
        None,
        "",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_serving_kv_ownership_csv_row<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    record_idx: usize,
    ownership: &ServingKvBlockOwnershipObservation,
) -> Result<(), CliError> {
    write_serving_worker_evidence_csv_record(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        "kv_block_ownership",
        record_idx,
        Some("kv_cache_owner"),
        "kv_cache",
        Some(ownership.owner.node_id),
        Some(ownership.owner.local_gpu_id),
        None,
        "",
        &csv_str(&usize_list(&ownership.decode_operation_ids)),
        None,
        Some(ownership.allocated_at_s),
        Some(ownership.released_at_s),
        Some(ownership.duration_s),
        Some(&ownership.allocation_id),
        &csv_str(&u32_list(&ownership.owner_worker_slots)),
        Some(ownership.block_start),
        Some(ownership.block_end),
        Some(u64::from(ownership.decode_sequences)),
        Some(ownership.resident_tokens),
        Some(ownership.kv_blocks),
        Some(ownership.allocated_kv_tokens),
        Some(ownership.kv_fragmentation_tokens),
        Some(ownership.block_table_entries),
        Some(ownership.block_table_bytes),
    )
}

#[allow(clippy::too_many_arguments)]
fn write_serving_kv_slot_ownership_csv_row<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    record_idx: usize,
    ownership: &ServingKvBlockOwnershipObservation,
    slot: &ServingKvWorkerSlotOwnershipObservation,
) -> Result<(), CliError> {
    write_serving_worker_evidence_csv_record(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        "kv_worker_slot_ownership",
        record_idx,
        Some("kv_cache_owner_slot"),
        "kv_cache",
        Some(ownership.owner.node_id),
        Some(ownership.owner.local_gpu_id),
        Some(slot.slot),
        &csv_str(&slot.slot.to_string()),
        &csv_str(&usize_list(&ownership.decode_operation_ids)),
        None,
        None,
        None,
        None,
        Some(&slot.allocation_id),
        &csv_str(&u32_list(&ownership.owner_worker_slots)),
        Some(slot.block_start),
        Some(slot.block_end),
        Some(u64::from(slot.decode_sequences)),
        Some(slot.resident_tokens),
        Some(slot.kv_blocks),
        Some(slot.allocated_kv_tokens),
        Some(slot.kv_fragmentation_tokens),
        Some(slot.block_table_entries),
        Some(slot.block_table_bytes),
    )
}

#[allow(clippy::too_many_arguments)]
fn write_serving_worker_evidence_csv_record<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    record_kind: &str,
    record_idx: usize,
    role: Option<&str>,
    phase: &str,
    node_id: Option<u32>,
    local_gpu_id: Option<u32>,
    slot: Option<u32>,
    worker_slots: &str,
    operation_ids: &str,
    assignment_count: Option<u32>,
    start_s: Option<f64>,
    finish_s: Option<f64>,
    duration_s: Option<f64>,
    allocation_id: Option<&str>,
    owner_worker_slots: &str,
    block_start: Option<u64>,
    block_end: Option<u64>,
    decode_sequences: Option<u64>,
    resident_tokens: Option<u64>,
    kv_blocks: Option<u64>,
    allocated_kv_tokens: Option<u64>,
    kv_fragmentation_tokens: Option<u64>,
    block_table_entries: Option<u64>,
    block_table_bytes: Option<u64>,
) -> Result<(), CliError> {
    write_csv_record(
        writer,
        &[
            csv_optional_str(scenario_name),
            candidate_rank.to_string(),
            csv_str(&score.candidate_id),
            score.feasible.to_string(),
            csv_str(score.deployment_mode.as_str()),
            csv_str(pool),
            observation.request_idx.to_string(),
            csv_optional_str(observation.request_id.as_deref()),
            csv_optional_str(observation.tenant.as_deref()),
            csv_optional_str(observation.model_id.as_deref()),
            csv_optional_str(observation.traffic_class.as_deref()),
            csv_optional_str(observation.shape_profile.as_deref()),
            csv_str(observation.status.as_str()),
            csv_str(record_kind),
            record_idx.to_string(),
            csv_optional_str(role),
            csv_str(phase),
            node_id.map(|value| value.to_string()).unwrap_or_default(),
            local_gpu_id
                .map(|value| value.to_string())
                .unwrap_or_default(),
            csv_optional_u32(slot),
            worker_slots.to_string(),
            operation_ids.to_string(),
            csv_optional_u32(assignment_count),
            csv_optional_f64(start_s),
            csv_optional_f64(finish_s),
            csv_optional_f64(duration_s),
            csv_optional_str(allocation_id),
            owner_worker_slots.to_string(),
            block_start
                .map(|value| value.to_string())
                .unwrap_or_default(),
            block_end.map(|value| value.to_string()).unwrap_or_default(),
            decode_sequences
                .map(|value| value.to_string())
                .unwrap_or_default(),
            resident_tokens
                .map(|value| value.to_string())
                .unwrap_or_default(),
            kv_blocks.map(|value| value.to_string()).unwrap_or_default(),
            allocated_kv_tokens
                .map(|value| value.to_string())
                .unwrap_or_default(),
            kv_fragmentation_tokens
                .map(|value| value.to_string())
                .unwrap_or_default(),
            block_table_entries
                .map(|value| value.to_string())
                .unwrap_or_default(),
            block_table_bytes
                .map(|value| value.to_string())
                .unwrap_or_default(),
        ],
    )
}

pub(super) fn initialize_serving_rejections_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_rejections_csv_path else {
        return Ok(());
    };
    write_serving_rejections_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_rejections_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_rejections_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_rejections_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_rejections_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_rejections_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,rejection_idx,phase,category,resource,code,observed,limit,unit,message,remediation"
    )?;
    Ok(())
}

fn write_serving_rejections_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for (rejection_idx, rejection) in score.rejections.iter().enumerate() {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    (rejection_idx + 1).to_string(),
                    csv_str(&rejection.phase),
                    csv_str(&rejection.category),
                    csv_str(&rejection.resource),
                    csv_str(&rejection.code),
                    csv_optional_f64(rejection.observed),
                    csv_optional_f64(rejection.limit),
                    csv_optional_str(rejection.unit.as_deref()),
                    csv_str(&rejection.message),
                    csv_optional_str(rejection.remediation.as_deref()),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_route_paths_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_route_paths_csv_path else {
        return Ok(());
    };
    write_serving_route_paths_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_route_paths_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_route_paths_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_route_paths_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_route_paths_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_route_paths_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,request_idx,request_id,tenant,model_id,traffic_class,shape_profile,path_idx,source_node,source_gpu,destination_node,destination_gpu,path_transfer_bytes,path_estimated_transfer_s,path_latency_s,path_bottleneck_bandwidth_gbps,path_resources,resource_idx,resource_id,kind,label,resource_transfer_bytes,resource_estimated_transfer_s,resource_bandwidth_gbps,resource_latency_s,rail_id,dependency_ids,from_kind,from_node,from_gpu,from_nic,from_rail,to_kind,to_node,to_gpu,to_nic,to_rail"
    )?;
    Ok(())
}

fn write_serving_route_paths_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for observation in &score.request_observations {
            if observation.kv_transfer_bytes == 0 || observation.kv_transfer_paths.is_empty() {
                continue;
            }
            let bytes_per_path = observation
                .kv_transfer_bytes
                .div_ceil(observation.kv_transfer_paths.len() as u64);
            for (path_idx, path) in observation.kv_transfer_paths.iter().enumerate() {
                if path.resource_details.is_empty() {
                    write_serving_route_paths_csv_row(
                        writer,
                        scenario_name,
                        candidate_rank,
                        score,
                        &pool,
                        observation,
                        path_idx,
                        path,
                        bytes_per_path,
                        None,
                        None,
                    )?;
                    continue;
                }
                for (resource_idx, resource) in path.resource_details.iter().enumerate() {
                    write_serving_route_paths_csv_row(
                        writer,
                        scenario_name,
                        candidate_rank,
                        score,
                        &pool,
                        observation,
                        path_idx,
                        path,
                        bytes_per_path,
                        Some(resource_idx),
                        Some(resource),
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_serving_route_paths_csv_row<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    path_idx: usize,
    path: &ServingKvTransferPathObservation,
    bytes_per_path: u64,
    resource_idx: Option<usize>,
    resource: Option<&ServingKvTransferPathResourceObservation>,
) -> Result<(), CliError> {
    let path_estimated_transfer_s =
        path.latency_s + route_transfer_seconds(bytes_per_path, path.bottleneck_bandwidth_gbps);
    let resource_estimated_transfer_s = resource.map(|resource| {
        resource.latency_s + route_transfer_seconds(bytes_per_path, resource.bandwidth_gbps)
    });
    write_csv_record(
        writer,
        &[
            csv_optional_str(scenario_name),
            candidate_rank.to_string(),
            csv_str(&score.candidate_id),
            score.feasible.to_string(),
            csv_str(score.deployment_mode.as_str()),
            csv_str(pool),
            observation.request_idx.to_string(),
            csv_optional_str(observation.request_id.as_deref()),
            csv_optional_str(observation.tenant.as_deref()),
            csv_optional_str(observation.model_id.as_deref()),
            csv_optional_str(observation.traffic_class.as_deref()),
            csv_optional_str(observation.shape_profile.as_deref()),
            (path_idx + 1).to_string(),
            path.source.node_id.to_string(),
            path.source.local_gpu_id.to_string(),
            path.destination.node_id.to_string(),
            path.destination.local_gpu_id.to_string(),
            bytes_per_path.to_string(),
            csv_f64(path_estimated_transfer_s),
            csv_f64(path.latency_s),
            csv_f64(path.bottleneck_bandwidth_gbps),
            csv_str(&path.resources.join(";")),
            resource_idx
                .map(|resource_idx| (resource_idx + 1).to_string())
                .unwrap_or_default(),
            resource
                .map(serving_route_path_resource_id)
                .map(|resource_id| csv_str(&resource_id))
                .unwrap_or_default(),
            resource
                .map(|resource| csv_str(&resource.kind))
                .unwrap_or_default(),
            resource
                .map(|resource| csv_str(&resource.label))
                .unwrap_or_default(),
            resource
                .map(|_| bytes_per_path.to_string())
                .unwrap_or_default(),
            csv_optional_f64(resource_estimated_transfer_s),
            csv_optional_f64(resource.map(|resource| resource.bandwidth_gbps)),
            csv_optional_f64(resource.map(|resource| resource.latency_s)),
            csv_optional_u32(resource.and_then(|resource| resource.rail_id)),
            csv_str(
                &observation
                    .kv_transfer_resource_dependencies
                    .iter()
                    .map(|dependency| dependency.to_string())
                    .collect::<Vec<_>>()
                    .join(";"),
            ),
            csv_endpoint_kind(resource.and_then(|resource| resource.from.as_ref())),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.from.as_ref()),
                |endpoint| endpoint.node_id,
            ),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.from.as_ref()),
                |endpoint| endpoint.local_gpu_id,
            ),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.from.as_ref()),
                |endpoint| endpoint.nic_id,
            ),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.from.as_ref()),
                |endpoint| endpoint.rail_id,
            ),
            csv_endpoint_kind(resource.and_then(|resource| resource.to.as_ref())),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.to.as_ref()),
                |endpoint| endpoint.node_id,
            ),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.to.as_ref()),
                |endpoint| endpoint.local_gpu_id,
            ),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.to.as_ref()),
                |endpoint| endpoint.nic_id,
            ),
            csv_endpoint_u32(
                resource.and_then(|resource| resource.to.as_ref()),
                |endpoint| endpoint.rail_id,
            ),
        ],
    )
}

fn route_transfer_seconds(bytes: u64, bandwidth_gbps: f64) -> f64 {
    if bytes == 0 {
        0.0
    } else if bandwidth_gbps.is_finite() && bandwidth_gbps > 0.0 {
        (bytes as f64 * 8.0) / (bandwidth_gbps * 1_000_000_000.0)
    } else {
        f64::INFINITY
    }
}

fn serving_route_path_resource_id(resource: &ServingKvTransferPathResourceObservation) -> String {
    let left = serving_route_path_endpoint_key(resource.from.as_ref());
    let right = serving_route_path_endpoint_key(resource.to.as_ref());
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    format!(
        "kv_route:{}|{}|rail={}|{}<->{}",
        resource.kind,
        resource.label,
        optional_u32_key(resource.rail_id),
        first,
        second
    )
}

fn serving_route_path_endpoint_key(
    endpoint: Option<&ServingKvTransferPathEndpointObservation>,
) -> String {
    let Some(endpoint) = endpoint else {
        return "none".to_string();
    };
    format!(
        "{}:node={}:gpu={}:nic={}:rail={}",
        endpoint.kind,
        optional_u32_key(endpoint.node_id),
        optional_u32_key(endpoint.local_gpu_id),
        optional_u32_key(endpoint.nic_id),
        optional_u32_key(endpoint.rail_id)
    )
}

fn optional_u32_key(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

pub(super) fn initialize_kv_route_resources_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.kv_route_resources_csv_path else {
        return Ok(());
    };
    write_kv_route_resources_csv_header(&mut File::create(path)?)
}

pub(super) fn write_kv_route_resources_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.kv_route_resources_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_kv_route_resources_csv_header(&mut file)?;
        Box::new(file)
    };
    write_kv_route_resources_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_kv_route_resources_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,resource_idx,resource_id,kind,label,request_count,path_observations,transfer_bytes,estimated_transfer_s,min_bandwidth_gbps,max_latency_s,rail_id,from_kind,from_node,from_gpu,from_nic,from_rail,to_kind,to_node,to_gpu,to_nic,to_rail"
    )?;
    Ok(())
}

fn write_kv_route_resources_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for (resource_idx, resource) in score.kv_route_resource_summary.iter().enumerate() {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    (resource_idx + 1).to_string(),
                    csv_str(&resource.resource_id),
                    csv_str(&resource.kind),
                    csv_str(&resource.label),
                    resource.request_count.to_string(),
                    resource.path_observations.to_string(),
                    resource.transfer_bytes.to_string(),
                    csv_f64(resource.estimated_transfer_s),
                    csv_f64(resource.min_bandwidth_gbps),
                    csv_f64(resource.max_latency_s),
                    csv_optional_u32(resource.rail_id),
                    csv_endpoint_kind(resource.from.as_ref()),
                    csv_endpoint_u32(resource.from.as_ref(), |endpoint| endpoint.node_id),
                    csv_endpoint_u32(resource.from.as_ref(), |endpoint| endpoint.local_gpu_id),
                    csv_endpoint_u32(resource.from.as_ref(), |endpoint| endpoint.nic_id),
                    csv_endpoint_u32(resource.from.as_ref(), |endpoint| endpoint.rail_id),
                    csv_endpoint_kind(resource.to.as_ref()),
                    csv_endpoint_u32(resource.to.as_ref(), |endpoint| endpoint.node_id),
                    csv_endpoint_u32(resource.to.as_ref(), |endpoint| endpoint.local_gpu_id),
                    csv_endpoint_u32(resource.to.as_ref(), |endpoint| endpoint.nic_id),
                    csv_endpoint_u32(resource.to.as_ref(), |endpoint| endpoint.rail_id),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_bottlenecks_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_bottlenecks_csv_path else {
        return Ok(());
    };
    write_serving_bottlenecks_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_bottlenecks_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_bottlenecks_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_bottlenecks_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_bottlenecks_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_bottlenecks_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,bottleneck_idx,source,phase,category,resource,code,severity,observed,limit,unit,message,remediation,request_idx,request_id,tenant,model_id,traffic_class,shape_profile"
    )?;
    Ok(())
}

fn write_serving_bottlenecks_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for (bottleneck_idx, bottleneck) in score.bottleneck_summary.iter().enumerate() {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    (bottleneck_idx + 1).to_string(),
                    csv_str(&bottleneck.source),
                    csv_str(&bottleneck.phase),
                    csv_str(&bottleneck.category),
                    csv_str(&bottleneck.resource),
                    csv_str(&bottleneck.code),
                    csv_str(&bottleneck.severity),
                    csv_optional_f64(bottleneck.observed),
                    csv_optional_f64(bottleneck.limit),
                    csv_optional_str(bottleneck.unit.as_deref()),
                    csv_str(&bottleneck.message),
                    csv_optional_str(bottleneck.remediation.as_deref()),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                ],
            )?;
        }
        let mut bottleneck_idx = score.bottleneck_summary.len();
        for observation in &score.request_observations {
            write_serving_request_bottleneck_rows(
                writer,
                scenario_name,
                candidate_rank,
                score,
                &pool,
                observation,
                &mut bottleneck_idx,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_serving_request_bottleneck_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    bottleneck_idx: &mut usize,
) -> Result<(), CliError> {
    if let Some(rejection) = &observation.rejection {
        *bottleneck_idx += 1;
        write_serving_request_bottleneck_csv_record(
            writer,
            scenario_name,
            candidate_rank,
            score,
            pool,
            *bottleneck_idx,
            observation,
            "request",
            &rejection.phase,
            &rejection.category,
            &request_resource_label(observation),
            &format!("request_{}", rejection.code),
            "critical",
            rejection.observed,
            rejection.limit,
            rejection.unit.as_deref(),
            observation
                .failure_reason
                .as_deref()
                .unwrap_or(&rejection.message),
            rejection.remediation.as_deref(),
        )?;
    }

    if observation.deadline_missed {
        *bottleneck_idx += 1;
        write_serving_request_bottleneck_csv_record(
            writer,
            scenario_name,
            candidate_rank,
            score,
            pool,
            *bottleneck_idx,
            observation,
            "request",
            "e2e",
            "deadline",
            &request_resource_label(observation),
            "request_deadline_miss",
            "critical",
            Some(observation.e2el_s),
            observation.deadline_s,
            Some("s"),
            &format!(
                "request {} exceeded its deadline",
                request_display_label(observation)
            ),
            Some(
                "reduce offered load, add serving capacity, or route this class to a lower-latency pool",
            ),
        )?;
    }

    write_serving_request_slo_bottleneck_row(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        bottleneck_idx,
        "ttft",
        observation.ttft_s,
        observation.slo.ttft_s,
        observation.ttft_slo_missed,
    )?;
    write_serving_request_slo_bottleneck_row(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        bottleneck_idx,
        "tpot",
        observation.tpot_s,
        observation.slo.tpot_s,
        observation.tpot_slo_missed,
    )?;
    write_serving_request_slo_bottleneck_row(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        bottleneck_idx,
        "itl",
        observation.itl_s,
        observation.slo.itl_s,
        observation.itl_slo_missed,
    )?;
    write_serving_request_slo_bottleneck_row(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        observation,
        bottleneck_idx,
        "e2el",
        observation.e2el_s,
        observation.slo.e2el_s,
        observation.e2el_slo_missed,
    )?;

    match observation.status.as_str() {
        "timed_out" | "cancelled" => {
            *bottleneck_idx += 1;
            write_serving_request_bottleneck_csv_record(
                writer,
                scenario_name,
                candidate_rank,
                score,
                pool,
                *bottleneck_idx,
                observation,
                "request",
                "lifecycle",
                "request_outcome",
                &request_resource_label(observation),
                &format!("request_{}", observation.status.as_str()),
                "critical",
                observation.status_time_s,
                observation.deadline_s.or(observation.cancellation_s),
                Some("s"),
                observation.failure_reason.as_deref().unwrap_or_else(|| {
                    if observation.status.as_str() == "timed_out" {
                        "request timed out before completion"
                    } else {
                        "request was cancelled before completion"
                    }
                }),
                Some(
                    "review admission policy, timeout/cancellation settings, and class-specific capacity",
                ),
            )?;
        }
        _ => {}
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_serving_request_slo_bottleneck_row<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    observation: &ServingRequestObservation,
    bottleneck_idx: &mut usize,
    phase: &str,
    observed_s: f64,
    limit_s: Option<f64>,
    missed: bool,
) -> Result<(), CliError> {
    if !missed {
        return Ok(());
    }
    *bottleneck_idx += 1;
    write_serving_request_bottleneck_csv_record(
        writer,
        scenario_name,
        candidate_rank,
        score,
        pool,
        *bottleneck_idx,
        observation,
        "request",
        phase,
        "slo",
        &request_resource_label(observation),
        &format!("request_{phase}_slo_miss"),
        "warning",
        Some(observed_s),
        limit_s,
        Some("s"),
        &format!(
            "request {} exceeded its {phase} SLO",
            request_display_label(observation)
        ),
        Some(
            "adjust the serving objective or add class-specific capacity, queue caps, or routing constraints",
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn write_serving_request_bottleneck_csv_record<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    candidate_rank: usize,
    score: &ScoredServingConfig,
    pool: &str,
    bottleneck_idx: usize,
    observation: &ServingRequestObservation,
    source: &str,
    phase: &str,
    category: &str,
    resource: &str,
    code: &str,
    severity: &str,
    observed: Option<f64>,
    limit: Option<f64>,
    unit: Option<&str>,
    message: &str,
    remediation: Option<&str>,
) -> Result<(), CliError> {
    write_csv_record(
        writer,
        &[
            csv_optional_str(scenario_name),
            candidate_rank.to_string(),
            csv_str(&score.candidate_id),
            score.feasible.to_string(),
            csv_str(score.deployment_mode.as_str()),
            csv_str(pool),
            bottleneck_idx.to_string(),
            csv_str(source),
            csv_str(phase),
            csv_str(category),
            csv_str(resource),
            csv_str(code),
            csv_str(severity),
            csv_optional_f64(observed),
            csv_optional_f64(limit),
            csv_optional_str(unit),
            csv_str(message),
            csv_optional_str(remediation),
            observation.request_idx.to_string(),
            csv_optional_str(observation.request_id.as_deref()),
            csv_optional_str(observation.tenant.as_deref()),
            csv_optional_str(observation.model_id.as_deref()),
            csv_optional_str(observation.traffic_class.as_deref()),
            csv_optional_str(observation.shape_profile.as_deref()),
        ],
    )
}

fn request_resource_label(observation: &ServingRequestObservation) -> String {
    observation
        .request_id
        .as_ref()
        .map(|id| format!("request:{id}"))
        .unwrap_or_else(|| format!("request:{}", observation.request_idx))
}

fn request_display_label(observation: &ServingRequestObservation) -> String {
    observation
        .request_id
        .as_ref()
        .map(|id| format!("'{id}'"))
        .unwrap_or_else(|| format!("#{}", observation.request_idx))
}

pub(super) fn initialize_serving_phase_calibration_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_phase_calibration_csv_path else {
        return Ok(());
    };
    write_serving_phase_calibration_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_phase_calibration_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_phase_calibration_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_phase_calibration_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_phase_calibration_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_phase_calibration_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,phase_idx,phase,active,calibrated,status,fit_count,applied_targets,estimated_s,phase_fit_count_with_uncertainty,phase_relative_uncertainty_pct,phase_absolute_uncertainty_s,phase_min_confidence_score,phase_max_extrapolation_ratio,phase_applicability_status,candidate_calibration_status,candidate_coverage_fraction,candidate_fit_count,candidate_relative_uncertainty_pct,candidate_absolute_uncertainty_s"
    )?;
    Ok(())
}

fn write_serving_phase_calibration_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for (phase_idx, phase) in score.phase_calibration.iter().enumerate() {
            let uncertainty =
                calibration_phase_uncertainty_summary(score.calibration_fits.iter(), &phase.phase);
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    (phase_idx + 1).to_string(),
                    csv_str(&phase.phase),
                    phase.active.to_string(),
                    phase.calibrated.to_string(),
                    csv_str(&phase.status),
                    phase.fit_count.to_string(),
                    csv_str(&phase.applied_targets.join("|")),
                    csv_f64(phase.estimated_s),
                    uncertainty.fit_count_with_uncertainty.to_string(),
                    csv_optional_f64(uncertainty.relative_uncertainty_pct),
                    csv_optional_f64(uncertainty.absolute_uncertainty_s),
                    csv_optional_f64(uncertainty.min_confidence_score),
                    csv_optional_f64(uncertainty.max_extrapolation_ratio),
                    csv_str(&uncertainty.applicability_status),
                    csv_str(&score.calibration_summary.status),
                    csv_f64(score.calibration_summary.coverage_fraction),
                    score.calibration_summary.fit_count.to_string(),
                    csv_optional_f64(score.calibration_summary.relative_uncertainty_pct),
                    csv_optional_f64(score.calibration_summary.absolute_uncertainty_s),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_approximations_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.serving_approximations_csv_path else {
        return Ok(());
    };
    write_serving_approximations_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_approximations_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.serving_approximations_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_approximations_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_approximations_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_approximations_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,record_kind,record_idx,action,metric,phase,category,scope,code,message,remediation"
    )?;
    Ok(())
}

fn write_serving_approximations_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for (record_idx, approximation) in score.approximations.iter().enumerate() {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    csv_str("approximation"),
                    (record_idx + 1).to_string(),
                    String::new(),
                    String::new(),
                    csv_str(&approximation.phase),
                    csv_str(&approximation.category),
                    csv_str(&approximation.scope),
                    csv_str(&approximation.code),
                    csv_str(&approximation.message),
                    csv_optional_str(approximation.remediation.as_deref()),
                ],
            )?;
        }
        for (record_idx, violation) in score.approximation_policy_violations.iter().enumerate() {
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    csv_str("policy_violation"),
                    (record_idx + 1).to_string(),
                    csv_str(violation.action.as_str()),
                    csv_optional_str(violation.metric.as_deref()),
                    csv_str(&violation.phase),
                    csv_str(&violation.category),
                    csv_str(&violation.scope),
                    csv_str(&violation.code),
                    csv_str(&violation.message),
                    String::new(),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_request_metrics_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.request_metrics_csv_path else {
        return Ok(());
    };
    write_serving_request_metrics_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_request_metrics_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.request_metrics_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_request_metrics_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_request_metrics_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_request_metrics_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,request_idx,request_id,tenant,model_id,traffic_class,shape_profile,status,metric_source,event_sourced,terminal_event,terminal_event_s,metric_unavailable_reason,ttft_end_event,tpot_start_event,tpot_end_event,e2el_end_event,throughput_duration_end_event,decode_finish_event_count,tpot_sample_count,included_in_measurement_window,measurement_window_source,measurement_start_s,measurement_end_s,arrival_s,batch_size,prompt_tokens,effective_prefill_tokens,decode_tokens,output_tokens,request_output_tokens_per_s,ttft_s,tpot_s,itl_s,e2el_s,prefill_queue_s,prefill_s,kv_queue_s,kv_transfer_s,decode_queue_s,decode_s,queue_delay_s,deadline_s,deadline_missed,ttft_slo_missed,tpot_slo_missed,itl_slo_missed,e2el_slo_missed,prefill_node,decode_node,kv_transfer_bytes"
    )?;
    Ok(())
}

fn write_serving_request_metrics_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for observation in &score.request_observations {
            let included_in_measurement_window = request_in_measurement_window(observation, score);
            let output_tokens = request_observation_output_tokens(observation);
            let request_output_tokens_per_s = request_observation_output_tokens_per_s(observation);
            let derivation = request_metric_derivation(observation);
            write_csv_record(
                writer,
                &[
                    csv_optional_str(scenario_name),
                    candidate_rank.to_string(),
                    csv_str(&score.candidate_id),
                    score.feasible.to_string(),
                    csv_str(score.deployment_mode.as_str()),
                    csv_str(&pool),
                    observation.request_idx.to_string(),
                    csv_optional_str(observation.request_id.as_deref()),
                    csv_optional_str(observation.tenant.as_deref()),
                    csv_optional_str(observation.model_id.as_deref()),
                    csv_optional_str(observation.traffic_class.as_deref()),
                    csv_optional_str(observation.shape_profile.as_deref()),
                    csv_str(observation.status.as_str()),
                    csv_str(&observation.metric_source),
                    derivation.event_sourced.to_string(),
                    csv_optional_str(derivation.terminal_event),
                    csv_optional_f64(derivation.terminal_event_s),
                    csv_optional_str(derivation.metric_unavailable_reason.as_deref()),
                    csv_optional_str(derivation.ttft_end_event),
                    csv_optional_str(derivation.tpot_start_event),
                    csv_optional_str(derivation.tpot_end_event),
                    csv_optional_str(derivation.e2el_end_event),
                    csv_optional_str(derivation.throughput_duration_end_event),
                    derivation.decode_finish_event_count.to_string(),
                    derivation.tpot_sample_count.to_string(),
                    included_in_measurement_window.to_string(),
                    csv_str(&score.measurement_window.source),
                    csv_f64(score.measurement_window.start_s),
                    csv_f64(score.measurement_window.end_s),
                    csv_f64(observation.arrival_s),
                    observation.batch_size.to_string(),
                    observation.prompt_tokens.to_string(),
                    observation.effective_prefill_tokens.to_string(),
                    observation.decode_tokens.to_string(),
                    output_tokens.to_string(),
                    csv_optional_f64(request_output_tokens_per_s),
                    csv_f64(observation.ttft_s),
                    csv_f64(observation.tpot_s),
                    csv_f64(observation.itl_s),
                    csv_f64(observation.e2el_s),
                    csv_f64(observation.queue_delay_s),
                    csv_f64(observation.prefill_s),
                    csv_f64(observation.kv_queue_s),
                    csv_f64(observation.kv_transfer_s),
                    csv_f64(observation.decode_queue_s),
                    csv_f64(observation.decode_s),
                    csv_f64(observation.queue_delay_s),
                    csv_optional_f64(observation.deadline_s),
                    observation.deadline_missed.to_string(),
                    observation.ttft_slo_missed.to_string(),
                    observation.tpot_slo_missed.to_string(),
                    observation.itl_slo_missed.to_string(),
                    observation.e2el_slo_missed.to_string(),
                    observation.prefill_node.to_string(),
                    observation.decode_node.to_string(),
                    observation.kv_transfer_bytes.to_string(),
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn initialize_serving_request_lifecycle_events_csv(
    args: &CliArgs,
) -> Result<(), CliError> {
    let Some(path) = &args.request_lifecycle_events_csv_path else {
        return Ok(());
    };
    write_serving_request_lifecycle_events_csv_header(&mut File::create(path)?)
}

pub(super) fn write_serving_request_lifecycle_events_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.request_lifecycle_events_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_serving_request_lifecycle_events_csv_header(&mut file)?;
        Box::new(file)
    };
    write_serving_request_lifecycle_events_csv_rows(
        &mut file,
        scenario_name,
        results,
        args.top_k.min(results.len()),
    )
}

fn write_serving_request_lifecycle_events_csv_header<W: Write>(
    writer: &mut W,
) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,candidate_rank,candidate_id,feasible,deployment_mode,pool,request_idx,request_id,tenant,model_id,traffic_class,shape_profile,status,metric_source,event_idx,event,phase,at_s,at_ms,decode_iteration,message"
    )?;
    Ok(())
}

fn write_serving_request_lifecycle_events_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    results: &[ScoredServingConfig],
    displayed_results: usize,
) -> Result<(), CliError> {
    for (candidate_idx, score) in results.iter().take(displayed_results).enumerate() {
        let candidate_rank = candidate_idx + 1;
        let pool = pool_label(score);
        for observation in &score.request_observations {
            for (event_idx, event) in observation.lifecycle_events.iter().enumerate() {
                write_csv_record(
                    writer,
                    &[
                        csv_optional_str(scenario_name),
                        candidate_rank.to_string(),
                        csv_str(&score.candidate_id),
                        score.feasible.to_string(),
                        csv_str(score.deployment_mode.as_str()),
                        csv_str(&pool),
                        observation.request_idx.to_string(),
                        csv_optional_str(observation.request_id.as_deref()),
                        csv_optional_str(observation.tenant.as_deref()),
                        csv_optional_str(observation.model_id.as_deref()),
                        csv_optional_str(observation.traffic_class.as_deref()),
                        csv_optional_str(observation.shape_profile.as_deref()),
                        csv_str(observation.status.as_str()),
                        csv_str(&observation.metric_source),
                        (event_idx + 1).to_string(),
                        csv_str(event.kind.as_str()),
                        csv_str(&event.phase),
                        csv_f64(event.at_s),
                        csv_f64(event.at_s * 1000.0),
                        csv_optional_u32(event.decode_iteration),
                        csv_optional_str(event.message.as_deref()),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn initialize_calibration_residuals_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.calibration_residuals_csv_path else {
        return Ok(());
    };
    write_calibration_residuals_csv_header(&mut File::create(path)?)
}

pub(super) fn write_calibration_residuals_csv_if_configured(
    args: &CliArgs,
    scenario_name: Option<&str>,
    profile: Option<&CalibrationProfileMetadata>,
    append: bool,
) -> Result<(), CliError> {
    let Some(path) = &args.calibration_residuals_csv_path else {
        return Ok(());
    };
    let mut file: Box<dyn Write> = if append {
        Box::new(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        let mut file = File::create(path)?;
        write_calibration_residuals_csv_header(&mut file)?;
        Box::new(file)
    };
    if let Some(profile) = profile {
        write_calibration_residuals_csv_rows(&mut file, scenario_name, profile)?;
    }
    Ok(())
}

fn write_calibration_residuals_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario,profile_path,profile_name,benchmark_idx,benchmark_name,kind,phase,hardware,fabric,model,dtype,batch_size,prompt_tokens,decode_tokens,sequence_tokens,tensor_ranks,pipeline_ranks,expert_ranks,data_ranks,measured_ms,predicted_ms,latency_error_ms,latency_abs_error_ms,latency_signed_pct_error,latency_abs_pct_error,latency_residual_status,throughput_tokens_per_s,source,command,notes"
    )?;
    Ok(())
}

fn write_calibration_residuals_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    scenario_name: Option<&str>,
    profile: &CalibrationProfileMetadata,
) -> Result<(), CliError> {
    for (idx, benchmark) in profile.benchmarks.iter().enumerate() {
        let residual = calibration_benchmark_latency_residual(benchmark);
        write_csv_record(
            writer,
            &[
                csv_optional_str(scenario_name),
                csv_str(&profile.path),
                csv_optional_str(profile.name.as_deref()),
                (idx + 1).to_string(),
                csv_optional_str(benchmark.name.as_deref()),
                csv_optional_str(benchmark.kind.as_deref()),
                csv_optional_str(benchmark.phase.as_deref()),
                csv_optional_str(benchmark.hardware.as_deref()),
                csv_optional_str(benchmark.fabric.as_deref()),
                csv_optional_str(benchmark.model.as_deref()),
                csv_optional_str(benchmark.dtype.as_deref()),
                csv_optional_u32(benchmark.batch_size),
                csv_optional_u32(benchmark.prompt_tokens),
                csv_optional_u32(benchmark.decode_tokens),
                csv_optional_u32(benchmark.sequence_tokens),
                csv_optional_u32(benchmark.tensor_ranks),
                csv_optional_u32(benchmark.pipeline_ranks),
                csv_optional_u32(benchmark.expert_ranks),
                csv_optional_u32(benchmark.data_ranks),
                csv_optional_f64(benchmark.measured_ms),
                csv_optional_f64(benchmark.predicted_ms),
                csv_optional_f64(residual.map(|residual| residual.error_ms)),
                csv_optional_f64(residual.map(|residual| residual.abs_error_ms)),
                csv_optional_f64(residual.map(|residual| residual.signed_pct_error)),
                csv_optional_f64(residual.map(|residual| residual.abs_pct_error)),
                csv_str(calibration_benchmark_latency_residual_status(residual)),
                csv_optional_f64(benchmark.throughput_tokens_per_s),
                csv_optional_str(benchmark.source.as_deref()),
                csv_optional_str(benchmark.command.as_deref()),
                csv_optional_str(benchmark.notes.as_deref()),
            ],
        )?;
    }
    Ok(())
}

fn write_csv_record<W: Write + ?Sized>(writer: &mut W, fields: &[String]) -> Result<(), CliError> {
    for (idx, field) in fields.iter().enumerate() {
        if idx > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{field}")?;
    }
    writeln!(writer)?;
    Ok(())
}

fn csv_optional_str(value: Option<&str>) -> String {
    value.map(csv_str).unwrap_or_default()
}

fn csv_str(value: &str) -> String {
    if value
        .chars()
        .any(|ch| matches!(ch, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_optional_f64(value: Option<f64>) -> String {
    value.map(csv_f64).unwrap_or_default()
}

fn csv_optional_u32(value: Option<u32>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn csv_optional_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn csv_optional_bool(value: Option<bool>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn csv_endpoint_kind(endpoint: Option<&ServingKvTransferPathEndpointObservation>) -> String {
    endpoint
        .map(|endpoint| csv_str(&endpoint.kind))
        .unwrap_or_default()
}

fn csv_endpoint_u32(
    endpoint: Option<&ServingKvTransferPathEndpointObservation>,
    field: impl FnOnce(&ServingKvTransferPathEndpointObservation) -> Option<u32>,
) -> String {
    endpoint
        .and_then(field)
        .map(|value| value.to_string())
        .unwrap_or_default()
}

fn csv_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.9}")
    } else {
        String::new()
    }
}

pub(super) fn request_in_measurement_window(
    observation: &ServingRequestObservation,
    score: &ScoredServingConfig,
) -> bool {
    observation.status.as_str() == "completed"
        && observation.arrival_s + 1e-12 >= score.measurement_window.start_s
        && observation.arrival_s <= score.measurement_window.end_s + 1e-12
}

pub(super) fn request_observation_output_tokens(observation: &ServingRequestObservation) -> u64 {
    if observation.status.as_str() != "completed" {
        return 0;
    }

    let decode_iterations = if observation.decode_iterations > 0 {
        observation.decode_iterations
    } else {
        observation.decode_tokens.max(1)
    };
    u64::from(observation.batch_size.max(1)).saturating_mul(u64::from(decode_iterations))
}

pub(super) fn request_observation_output_tokens_per_s(
    observation: &ServingRequestObservation,
) -> Option<f64> {
    if observation.status.as_str() == "completed"
        && observation.e2el_s.is_finite()
        && observation.e2el_s > 0.0
    {
        Some(request_observation_output_tokens(observation) as f64 / observation.e2el_s)
    } else {
        None
    }
}

fn metric_source_count_list(counts: &[ServingMeasurementMetricSourceCount]) -> String {
    counts
        .iter()
        .map(|count| format!("{}:{}", count.metric_source, count.request_count))
        .collect::<Vec<_>>()
        .join("|")
}

fn serving_gpu_type_count_list(counts: &[ServingGpuTypeCount]) -> String {
    counts
        .iter()
        .map(|count| format!("{}:{}", count.gpu, count.count))
        .collect::<Vec<_>>()
        .join("|")
}

fn serving_gpu_label_count_list(counts: &[ServingGpuLabelCount]) -> String {
    counts
        .iter()
        .map(|count| format!("{}:{}", count.label, count.count))
        .collect::<Vec<_>>()
        .join("|")
}

fn serving_approximation_count_list(counts: &[ServingApproximationCount]) -> String {
    counts
        .iter()
        .map(|count| format!("{}:{}", count.name, count.count))
        .collect::<Vec<_>>()
        .join("|")
}

fn csv_metric_uncertainty_s(
    feasible: bool,
    value_s: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value_s.is_finite() {
        return String::new();
    }
    csv_optional_f64(metric_uncertainty_value(value_s, uncertainty))
}

fn csv_metric_lower_s(
    feasible: bool,
    value_s: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value_s.is_finite() {
        return String::new();
    }
    csv_optional_f64(metric_sensitivity_bounds(value_s, uncertainty).map(|(lower, _)| lower))
}

fn csv_metric_upper_s(
    feasible: bool,
    value_s: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value_s.is_finite() {
        return String::new();
    }
    csv_optional_f64(metric_sensitivity_bounds(value_s, uncertainty).map(|(_, upper)| upper))
}

fn csv_metric_relative_uncertainty(
    feasible: bool,
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value.is_finite() {
        return String::new();
    }
    csv_optional_f64(metric_relative_uncertainty_value(value, uncertainty))
}

fn csv_metric_relative_lower(
    feasible: bool,
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value.is_finite() {
        return String::new();
    }
    csv_optional_f64(metric_relative_sensitivity_bounds(value, uncertainty).map(|(lower, _)| lower))
}

fn csv_metric_relative_upper(
    feasible: bool,
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value.is_finite() {
        return String::new();
    }
    csv_optional_f64(metric_relative_sensitivity_bounds(value, uncertainty).map(|(_, upper)| upper))
}

pub(super) fn initialize_scenario_sensitivity_csv(args: &CliArgs) -> Result<(), CliError> {
    let Some(path) = &args.scenario_sensitivity_csv_path else {
        return Ok(());
    };
    write_scenario_sensitivity_csv_header(&mut File::create(path)?)
}

pub(super) fn write_scenario_sensitivity_csv_if_configured(
    args: &CliArgs,
    sensitivity: &[ScenarioSensitivity],
) -> Result<(), CliError> {
    let Some(path) = &args.scenario_sensitivity_csv_path else {
        return Ok(());
    };
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    write_scenario_sensitivity_csv_rows(&mut file, sensitivity)
}

fn write_scenario_sensitivity_csv_header<W: Write>(writer: &mut W) -> Result<(), CliError> {
    writeln!(
        writer,
        "scenario_index,scenario,available,reason,baseline,baseline_name,candidate_id,status,rejected_reason,feasible,objective,deployment_mode,pool,hardware_unique_node_count,hardware_unique_gpu_count,hardware_prefill_gpu_count,hardware_decode_gpu_count,hardware_shared_gpu_count,hardware_aggregate_hbm_gb,hardware_prefill_hbm_gb,hardware_decode_hbm_gb,hardware_aggregate_effective_peak_tflops,hardware_prefill_effective_peak_tflops,hardware_decode_effective_peak_tflops,hardware_aggregate_gpu_types,hardware_prefill_gpu_types,hardware_decode_gpu_types,hardware_aggregate_gpu_label_counts,hardware_prefill_gpu_label_counts,hardware_decode_gpu_label_counts,hardware_throughput_tokens_per_s_per_gpu,hardware_throughput_tokens_per_s_per_effective_peak_tflop,hardware_throughput_tokens_per_s_per_hbm_gb,calibration_status,calibration_coverage_fraction,calibration_fit_count,calibration_fit_count_with_uncertainty,calibration_relative_uncertainty_pct,calibration_absolute_uncertainty_ms,calibration_gate_violation_count,calibration_hard_gate_violation_count,approximation_status,approximation_count,approximation_policy_violation_count,approximation_calibration_count,approximation_topology_count,approximation_queueing_count,approximation_coarse_topology,approximation_approximate_queueing,approximation_uncalibrated_runtime,approximation_category_counts,approximation_top_codes,bottleneck_count,top_bottleneck_source,top_bottleneck_category,top_bottleneck_code,top_bottleneck_severity,rejection_count,top_rejection_phase,top_rejection_category,top_rejection_resource,top_rejection_code,top_rejection_unit,top_rejection_remediation,ttft_ms,ttft_ms_delta,ttft_ms_delta_pct,tpot_ms,tpot_ms_delta,tpot_ms_delta_pct,throughput_tokens_per_s,throughput_tokens_per_s_delta,throughput_tokens_per_s_delta_pct,e2el_ms,e2el_ms_delta,e2el_ms_delta_pct"
    )?;
    Ok(())
}

fn write_scenario_sensitivity_csv_rows<W: Write + ?Sized>(
    writer: &mut W,
    sensitivity: &[ScenarioSensitivity],
) -> Result<(), CliError> {
    let baseline = sensitivity.iter().find(|entry| entry.available);
    for entry in sensitivity {
        let comparison_baseline = if entry.available { baseline } else { None };
        let (ttft_delta, ttft_delta_pct) = scenario_metric_delta_values(
            entry.ttft_ms,
            comparison_baseline.and_then(|baseline| baseline.ttft_ms),
        );
        let (tpot_delta, tpot_delta_pct) = scenario_metric_delta_values(
            entry.tpot_ms,
            comparison_baseline.and_then(|baseline| baseline.tpot_ms),
        );
        let (throughput_delta, throughput_delta_pct) = scenario_metric_delta_values(
            entry.throughput_tokens_per_s,
            comparison_baseline.and_then(|baseline| baseline.throughput_tokens_per_s),
        );
        let (e2el_delta, e2el_delta_pct) = scenario_metric_delta_values(
            entry.e2el_ms,
            comparison_baseline.and_then(|baseline| baseline.e2el_ms),
        );
        write_csv_record(
            writer,
            &[
                entry.index.to_string(),
                csv_str(&entry.name),
                entry.available.to_string(),
                csv_optional_str(entry.reason.as_deref()),
                baseline
                    .is_some_and(|baseline| baseline.index == entry.index)
                    .to_string(),
                csv_optional_str(baseline.map(|baseline| baseline.name.as_str())),
                csv_optional_str(entry.candidate_id.as_deref()),
                csv_optional_str(entry.status.as_deref()),
                csv_optional_str(entry.rejected_reason.as_deref()),
                entry
                    .feasible
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_optional_str(entry.objective.as_deref()),
                csv_optional_str(entry.deployment_mode.as_deref()),
                csv_optional_str(entry.pool.as_deref()),
                csv_optional_u64(entry.hardware_unique_node_count),
                csv_optional_u64(entry.hardware_unique_gpu_count),
                csv_optional_u64(entry.hardware_prefill_gpu_count),
                csv_optional_u64(entry.hardware_decode_gpu_count),
                csv_optional_u64(entry.hardware_shared_gpu_count),
                csv_optional_f64(entry.hardware_aggregate_hbm_gb),
                csv_optional_f64(entry.hardware_prefill_hbm_gb),
                csv_optional_f64(entry.hardware_decode_hbm_gb),
                csv_optional_f64(entry.hardware_aggregate_effective_peak_tflops),
                csv_optional_f64(entry.hardware_prefill_effective_peak_tflops),
                csv_optional_f64(entry.hardware_decode_effective_peak_tflops),
                csv_optional_str(entry.hardware_aggregate_gpu_types.as_deref()),
                csv_optional_str(entry.hardware_prefill_gpu_types.as_deref()),
                csv_optional_str(entry.hardware_decode_gpu_types.as_deref()),
                csv_optional_str(entry.hardware_aggregate_gpu_label_counts.as_deref()),
                csv_optional_str(entry.hardware_prefill_gpu_label_counts.as_deref()),
                csv_optional_str(entry.hardware_decode_gpu_label_counts.as_deref()),
                csv_optional_f64(entry.hardware_throughput_tokens_per_s_per_gpu),
                csv_optional_f64(entry.hardware_throughput_tokens_per_s_per_effective_peak_tflop),
                csv_optional_f64(entry.hardware_throughput_tokens_per_s_per_hbm_gb),
                csv_optional_str(entry.calibration_status.as_deref()),
                csv_optional_f64(entry.calibration_coverage_fraction),
                csv_optional_u64(entry.calibration_fit_count),
                csv_optional_u64(entry.calibration_fit_count_with_uncertainty),
                csv_optional_f64(entry.calibration_relative_uncertainty_pct),
                csv_optional_f64(entry.calibration_absolute_uncertainty_ms),
                csv_optional_u64(entry.calibration_gate_violation_count),
                csv_optional_u64(entry.calibration_hard_gate_violation_count),
                csv_optional_str(entry.approximation_status.as_deref()),
                csv_optional_u64(entry.approximation_count),
                csv_optional_u64(entry.approximation_policy_violation_count),
                csv_optional_u64(entry.approximation_calibration_count),
                csv_optional_u64(entry.approximation_topology_count),
                csv_optional_u64(entry.approximation_queueing_count),
                csv_optional_bool(entry.approximation_coarse_topology),
                csv_optional_bool(entry.approximation_approximate_queueing),
                csv_optional_bool(entry.approximation_uncalibrated_runtime),
                csv_optional_str(entry.approximation_category_counts.as_deref()),
                csv_optional_str(entry.approximation_top_codes.as_deref()),
                csv_optional_u64(entry.bottleneck_count),
                csv_optional_str(entry.top_bottleneck_source.as_deref()),
                csv_optional_str(entry.top_bottleneck_category.as_deref()),
                csv_optional_str(entry.top_bottleneck_code.as_deref()),
                csv_optional_str(entry.top_bottleneck_severity.as_deref()),
                csv_optional_u64(entry.rejection_count),
                csv_optional_str(entry.top_rejection_phase.as_deref()),
                csv_optional_str(entry.top_rejection_category.as_deref()),
                csv_optional_str(entry.top_rejection_resource.as_deref()),
                csv_optional_str(entry.top_rejection_code.as_deref()),
                csv_optional_str(entry.top_rejection_unit.as_deref()),
                csv_optional_str(entry.top_rejection_remediation.as_deref()),
                csv_optional_f64(entry.ttft_ms),
                csv_optional_f64(ttft_delta),
                csv_optional_f64(ttft_delta_pct),
                csv_optional_f64(entry.tpot_ms),
                csv_optional_f64(tpot_delta),
                csv_optional_f64(tpot_delta_pct),
                csv_optional_f64(entry.throughput_tokens_per_s),
                csv_optional_f64(throughput_delta),
                csv_optional_f64(throughput_delta_pct),
                csv_optional_f64(entry.e2el_ms),
                csv_optional_f64(e2el_delta),
                csv_optional_f64(e2el_delta_pct),
            ],
        )?;
    }
    Ok(())
}
