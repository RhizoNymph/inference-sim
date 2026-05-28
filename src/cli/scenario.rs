use super::*;

pub(super) fn write_scenario_sweep<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    base_workload: &WorkloadConfig,
    args: &CliArgs,
) -> Result<(), CliError> {
    if args.request_metrics_csv_path.is_some() {
        initialize_serving_request_metrics_csv(args)?;
    }
    if args.request_lifecycle_events_csv_path.is_some() {
        initialize_serving_request_lifecycle_events_csv(args)?;
    }
    if args.serving_metrics_csv_path.is_some() {
        initialize_serving_metrics_csv(args)?;
    }
    if args.serving_metric_breakdowns_csv_path.is_some() {
        initialize_serving_metric_breakdowns_csv(args)?;
    }
    if args.serving_services_csv_path.is_some() {
        initialize_serving_services_csv(args)?;
    }
    if args.serving_utilization_csv_path.is_some() {
        initialize_serving_utilization_csv(args)?;
    }
    if args.serving_memory_pressure_csv_path.is_some() {
        initialize_serving_memory_pressure_csv(args)?;
    }
    if args.serving_timeline_csv_path.is_some() {
        initialize_serving_timeline_csv(args)?;
    }
    if args.serving_occupancy_csv_path.is_some() {
        initialize_serving_occupancy_csv(args)?;
    }
    if args.serving_placement_evidence_csv_path.is_some() {
        initialize_serving_placement_evidence_csv(args)?;
    }
    if args.serving_worker_evidence_csv_path.is_some() {
        initialize_serving_worker_evidence_csv(args)?;
    }
    if args.serving_rejections_csv_path.is_some() {
        initialize_serving_rejections_csv(args)?;
    }
    if args.serving_route_paths_csv_path.is_some() {
        initialize_serving_route_paths_csv(args)?;
    }
    if args.kv_route_resources_csv_path.is_some() {
        initialize_kv_route_resources_csv(args)?;
    }
    if args.serving_bottlenecks_csv_path.is_some() {
        initialize_serving_bottlenecks_csv(args)?;
    }
    if args.serving_phase_calibration_csv_path.is_some() {
        initialize_serving_phase_calibration_csv(args)?;
    }
    if args.serving_approximations_csv_path.is_some() {
        initialize_serving_approximations_csv(args)?;
    }
    if args.calibration_residuals_csv_path.is_some() {
        initialize_calibration_residuals_csv(args)?;
    }
    if args.scenario_sensitivity_csv_path.is_some() {
        initialize_scenario_sensitivity_csv(args)?;
    }
    if args.rank_sensitivity_csv_path.is_some() {
        initialize_rank_sensitivity_csv(args)?;
    }
    match args.format {
        OutputFormat::Text => {
            let mut scenario_sensitivity = Vec::new();
            for (idx, scenario) in args.scenarios.iter().enumerate() {
                if idx > 0 {
                    writeln!(writer)?;
                }
                let mut cluster = cluster.clone();
                let mut workload = base_workload.clone();
                apply_run_scenario_topology(&mut cluster, scenario)?;
                apply_run_scenario(&mut workload, scenario)?;
                validate_scenario_workload(&cluster, &workload, scenario)?;
                writeln!(writer, "scenario={}", scenario.name)?;
                write_scenario_topology_text(writer, scenario)?;
                solve_and_write(writer, &cluster, &workload, args, Some(&scenario.name))?;
                if args.scenario_sensitivity_csv_path.is_some() {
                    let result_json = scenario_result_json_for_sensitivity(
                        &cluster,
                        &workload,
                        args,
                        Some(&scenario.name),
                    )?;
                    scenario_sensitivity.push(scenario_sensitivity_from_result_json(
                        idx + 1,
                        &scenario.name,
                        &result_json,
                    )?);
                }
            }
            write_scenario_sensitivity_csv_if_configured(args, &scenario_sensitivity)?;
        }
        OutputFormat::Json => {
            let mut scenario_sensitivity = Vec::new();
            writeln!(writer, "{{")?;
            writeln!(writer, "  \"schema_version\": 1,")?;
            writeln!(writer, "  \"mode\": \"scenario_sweep\",")?;
            writeln!(writer, "  \"scenario_count\": {},", args.scenarios.len())?;
            writeln!(writer, "  \"scenarios\": [")?;
            for (idx, scenario) in args.scenarios.iter().enumerate() {
                let mut cluster = cluster.clone();
                let mut workload = base_workload.clone();
                apply_run_scenario_topology(&mut cluster, scenario)?;
                apply_run_scenario(&mut workload, scenario)?;
                validate_scenario_workload(&cluster, &workload, scenario)?;
                let mut output = Vec::new();
                solve_and_write(&mut output, &cluster, &workload, args, Some(&scenario.name))?;
                let result_json = String::from_utf8(output).map_err(|err| {
                    CliError::Usage(format!(
                        "scenario '{}' produced non-UTF-8 JSON output: {err}",
                        scenario.name
                    ))
                })?;
                scenario_sensitivity.push(scenario_sensitivity_from_result_json(
                    idx + 1,
                    &scenario.name,
                    &result_json,
                )?);
                writeln!(writer, "    {{")?;
                writeln!(writer, "      \"name\": {},", json_string(&scenario.name))?;
                writeln!(writer, "      \"index\": {},", idx + 1)?;
                write_scenario_topology_json(writer, scenario)?;
                write!(writer, "      \"result\": ")?;
                writer.write_all(result_json.trim_end().as_bytes())?;
                writeln!(writer)?;
                write!(writer, "    }}")?;
                if idx + 1 < args.scenarios.len() {
                    writeln!(writer, ",")?;
                } else {
                    writeln!(writer)?;
                }
            }
            writeln!(writer, "  ],")?;
            write_scenario_sensitivity_csv_if_configured(args, &scenario_sensitivity)?;
            write_scenario_sensitivity_json(writer, &scenario_sensitivity)?;
            writeln!(writer, "}}")?;
        }
        OutputFormat::Markdown => {
            let mut scenario_sensitivity = Vec::new();
            writeln!(writer, "# Scenario Sweep Summary\n")?;
            for (idx, scenario) in args.scenarios.iter().enumerate() {
                let mut cluster = cluster.clone();
                let mut workload = base_workload.clone();
                apply_run_scenario_topology(&mut cluster, scenario)?;
                apply_run_scenario(&mut workload, scenario)?;
                validate_scenario_workload(&cluster, &workload, scenario)?;
                writeln!(
                    writer,
                    "## Scenario {}: {}\n",
                    idx + 1,
                    markdown_cell(&scenario.name)
                )?;
                let mut output = Vec::new();
                solve_and_write(&mut output, &cluster, &workload, args, Some(&scenario.name))?;
                let result_markdown = String::from_utf8(output).map_err(|err| {
                    CliError::Usage(format!(
                        "scenario '{}' produced non-UTF-8 Markdown output: {err}",
                        scenario.name
                    ))
                })?;
                writer.write_all(
                    demote_markdown_summary_headings(&result_markdown)
                        .trim_start()
                        .as_bytes(),
                )?;
                writeln!(writer)?;
                if args.scenario_sensitivity_csv_path.is_some() {
                    let result_json = scenario_result_json_for_sensitivity(
                        &cluster,
                        &workload,
                        args,
                        Some(&scenario.name),
                    )?;
                    scenario_sensitivity.push(scenario_sensitivity_from_result_json(
                        idx + 1,
                        &scenario.name,
                        &result_json,
                    )?);
                }
            }
            write_scenario_sensitivity_csv_if_configured(args, &scenario_sensitivity)?;
        }
    }
    Ok(())
}

fn scenario_result_json_for_sensitivity(
    cluster: &Cluster,
    workload: &WorkloadConfig,
    args: &CliArgs,
    scenario_name: Option<&str>,
) -> Result<String, CliError> {
    let mut sensitivity_args = args.clone();
    sensitivity_args.format = OutputFormat::Json;
    sensitivity_args.request_metrics_csv_path = None;
    sensitivity_args.request_lifecycle_events_csv_path = None;
    sensitivity_args.serving_metrics_csv_path = None;
    sensitivity_args.serving_metric_breakdowns_csv_path = None;
    sensitivity_args.serving_services_csv_path = None;
    sensitivity_args.serving_utilization_csv_path = None;
    sensitivity_args.serving_memory_pressure_csv_path = None;
    sensitivity_args.serving_timeline_csv_path = None;
    sensitivity_args.serving_occupancy_csv_path = None;
    sensitivity_args.serving_placement_evidence_csv_path = None;
    sensitivity_args.serving_worker_evidence_csv_path = None;
    sensitivity_args.serving_rejections_csv_path = None;
    sensitivity_args.serving_route_paths_csv_path = None;
    sensitivity_args.kv_route_resources_csv_path = None;
    sensitivity_args.serving_bottlenecks_csv_path = None;
    sensitivity_args.serving_phase_calibration_csv_path = None;
    sensitivity_args.serving_approximations_csv_path = None;
    sensitivity_args.calibration_residuals_csv_path = None;
    sensitivity_args.scenario_sensitivity_csv_path = None;
    sensitivity_args.rank_sensitivity_csv_path = None;

    let mut output = Vec::new();
    solve_and_write(
        &mut output,
        cluster,
        workload,
        &sensitivity_args,
        scenario_name,
    )?;
    String::from_utf8(output).map_err(|err| {
        CliError::Usage(format!(
            "scenario sensitivity probe produced non-UTF-8 JSON output: {err}"
        ))
    })
}

#[derive(Clone, Debug, Default)]
pub(super) struct ScenarioSensitivity {
    pub(super) index: usize,
    pub(super) name: String,
    pub(super) available: bool,
    pub(super) reason: Option<String>,
    pub(super) candidate_id: Option<String>,
    pub(super) status: Option<String>,
    pub(super) rejected_reason: Option<String>,
    pub(super) feasible: Option<bool>,
    pub(super) objective: Option<String>,
    pub(super) deployment_mode: Option<String>,
    pub(super) pool: Option<String>,
    pub(super) hardware_unique_node_count: Option<u64>,
    pub(super) hardware_unique_gpu_count: Option<u64>,
    pub(super) hardware_prefill_gpu_count: Option<u64>,
    pub(super) hardware_decode_gpu_count: Option<u64>,
    pub(super) hardware_shared_gpu_count: Option<u64>,
    pub(super) hardware_aggregate_hbm_gb: Option<f64>,
    pub(super) hardware_prefill_hbm_gb: Option<f64>,
    pub(super) hardware_decode_hbm_gb: Option<f64>,
    pub(super) hardware_aggregate_effective_peak_tflops: Option<f64>,
    pub(super) hardware_prefill_effective_peak_tflops: Option<f64>,
    pub(super) hardware_decode_effective_peak_tflops: Option<f64>,
    pub(super) hardware_aggregate_gpu_types: Option<String>,
    pub(super) hardware_prefill_gpu_types: Option<String>,
    pub(super) hardware_decode_gpu_types: Option<String>,
    pub(super) hardware_aggregate_gpu_label_counts: Option<String>,
    pub(super) hardware_prefill_gpu_label_counts: Option<String>,
    pub(super) hardware_decode_gpu_label_counts: Option<String>,
    pub(super) hardware_throughput_tokens_per_s_per_gpu: Option<f64>,
    pub(super) hardware_throughput_tokens_per_s_per_effective_peak_tflop: Option<f64>,
    pub(super) hardware_throughput_tokens_per_s_per_hbm_gb: Option<f64>,
    pub(super) calibration_status: Option<String>,
    pub(super) calibration_coverage_fraction: Option<f64>,
    pub(super) calibration_fit_count: Option<u64>,
    pub(super) calibration_fit_count_with_uncertainty: Option<u64>,
    pub(super) calibration_relative_uncertainty_pct: Option<f64>,
    pub(super) calibration_absolute_uncertainty_ms: Option<f64>,
    pub(super) calibration_gate_violation_count: Option<u64>,
    pub(super) calibration_hard_gate_violation_count: Option<u64>,
    pub(super) approximation_status: Option<String>,
    pub(super) approximation_count: Option<u64>,
    pub(super) approximation_policy_violation_count: Option<u64>,
    pub(super) approximation_calibration_count: Option<u64>,
    pub(super) approximation_topology_count: Option<u64>,
    pub(super) approximation_queueing_count: Option<u64>,
    pub(super) approximation_coarse_topology: Option<bool>,
    pub(super) approximation_approximate_queueing: Option<bool>,
    pub(super) approximation_uncalibrated_runtime: Option<bool>,
    pub(super) approximation_category_counts: Option<String>,
    pub(super) approximation_top_codes: Option<String>,
    pub(super) bottleneck_count: Option<u64>,
    pub(super) top_bottleneck_source: Option<String>,
    pub(super) top_bottleneck_category: Option<String>,
    pub(super) top_bottleneck_code: Option<String>,
    pub(super) top_bottleneck_severity: Option<String>,
    pub(super) rejection_count: Option<u64>,
    pub(super) top_rejection_phase: Option<String>,
    pub(super) top_rejection_category: Option<String>,
    pub(super) top_rejection_resource: Option<String>,
    pub(super) top_rejection_code: Option<String>,
    pub(super) top_rejection_unit: Option<String>,
    pub(super) top_rejection_remediation: Option<String>,
    pub(super) ttft_ms: Option<f64>,
    pub(super) tpot_ms: Option<f64>,
    pub(super) throughput_tokens_per_s: Option<f64>,
    pub(super) e2el_ms: Option<f64>,
}

fn scenario_sensitivity_from_result_json(
    index: usize,
    name: &str,
    result_json: &str,
) -> Result<ScenarioSensitivity, CliError> {
    let value: serde_json::Value = serde_json::from_str(result_json).map_err(|err| {
        CliError::Usage(format!(
            "scenario '{name}' produced JSON that could not be parsed for sensitivity output: {err}"
        ))
    })?;
    if value.get("mode").and_then(serde_json::Value::as_str) != Some("serving") {
        return Ok(ScenarioSensitivity {
            index,
            name: name.to_string(),
            available: false,
            reason: Some("scenario result is not a serving solve".to_string()),
            ..ScenarioSensitivity::default()
        });
    }
    let Some(top) = value
        .get("results")
        .and_then(serde_json::Value::as_array)
        .and_then(|results| results.first())
    else {
        return Ok(ScenarioSensitivity {
            index,
            name: name.to_string(),
            available: false,
            reason: Some("scenario serving solve returned no ranked candidates".to_string()),
            ..ScenarioSensitivity::default()
        });
    };
    let metrics = top.get("metrics").unwrap_or(&serde_json::Value::Null);
    let hardware_footprint = top
        .get("hardware_footprint")
        .unwrap_or(&serde_json::Value::Null);
    let calibration_summary = top
        .get("serving_calibration_summary")
        .unwrap_or(&serde_json::Value::Null);
    let approximation_summary = top
        .get("approximation_summary")
        .unwrap_or(&serde_json::Value::Null);
    let top_bottleneck = top
        .get("bottleneck_summary")
        .and_then(serde_json::Value::as_array)
        .and_then(|bottlenecks| bottlenecks.first())
        .unwrap_or(&serde_json::Value::Null);
    let top_rejection = top
        .get("rejections")
        .and_then(serde_json::Value::as_array)
        .and_then(|rejections| rejections.first())
        .unwrap_or(&serde_json::Value::Null);
    let ttft_ms = json_f64_field(metrics, "ttft_ms");
    let tpot_ms = json_f64_field(metrics, "tpot_ms");
    let throughput_tokens_per_s = json_f64_field(metrics, "throughput_tokens_per_s");
    let e2el_ms = json_f64_field(metrics, "e2el_ms");
    let available = ttft_ms.is_some()
        && tpot_ms.is_some()
        && throughput_tokens_per_s.is_some()
        && e2el_ms.is_some();
    Ok(ScenarioSensitivity {
        index,
        name: name.to_string(),
        available,
        reason: (!available).then_some(
            "scenario top serving candidate did not produce complete comparable metrics"
                .to_string(),
        ),
        candidate_id: json_str_field(top, "candidate_id"),
        status: json_str_field(top, "status"),
        rejected_reason: json_str_field(top, "rejected_reason"),
        feasible: top.get("feasible").and_then(serde_json::Value::as_bool),
        objective: json_str_field(top, "objective"),
        deployment_mode: json_str_field(top, "deployment_mode"),
        pool: json_str_field(top, "pool"),
        hardware_unique_node_count: json_u64_field(hardware_footprint, "unique_node_count"),
        hardware_unique_gpu_count: json_u64_field(hardware_footprint, "unique_gpu_count"),
        hardware_prefill_gpu_count: json_u64_field(hardware_footprint, "prefill_gpu_count"),
        hardware_decode_gpu_count: json_u64_field(hardware_footprint, "decode_gpu_count"),
        hardware_shared_gpu_count: json_u64_field(hardware_footprint, "shared_gpu_count"),
        hardware_aggregate_hbm_gb: json_f64_field(hardware_footprint, "aggregate_hbm_gb"),
        hardware_prefill_hbm_gb: json_f64_field(hardware_footprint, "prefill_hbm_gb"),
        hardware_decode_hbm_gb: json_f64_field(hardware_footprint, "decode_hbm_gb"),
        hardware_aggregate_effective_peak_tflops: json_f64_field(
            hardware_footprint,
            "aggregate_effective_peak_tflops",
        ),
        hardware_prefill_effective_peak_tflops: json_f64_field(
            hardware_footprint,
            "prefill_effective_peak_tflops",
        ),
        hardware_decode_effective_peak_tflops: json_f64_field(
            hardware_footprint,
            "decode_effective_peak_tflops",
        ),
        hardware_aggregate_gpu_types: json_named_count_list_field(
            hardware_footprint,
            "aggregate_gpu_types",
            "gpu",
        ),
        hardware_prefill_gpu_types: json_named_count_list_field(
            hardware_footprint,
            "prefill_gpu_types",
            "gpu",
        ),
        hardware_decode_gpu_types: json_named_count_list_field(
            hardware_footprint,
            "decode_gpu_types",
            "gpu",
        ),
        hardware_aggregate_gpu_label_counts: json_named_count_list_field(
            hardware_footprint,
            "aggregate_gpu_label_counts",
            "label",
        ),
        hardware_prefill_gpu_label_counts: json_named_count_list_field(
            hardware_footprint,
            "prefill_gpu_label_counts",
            "label",
        ),
        hardware_decode_gpu_label_counts: json_named_count_list_field(
            hardware_footprint,
            "decode_gpu_label_counts",
            "label",
        ),
        hardware_throughput_tokens_per_s_per_gpu: json_f64_field(
            hardware_footprint,
            "throughput_tokens_per_s_per_gpu",
        ),
        hardware_throughput_tokens_per_s_per_effective_peak_tflop: json_f64_field(
            hardware_footprint,
            "throughput_tokens_per_s_per_effective_peak_tflop",
        ),
        hardware_throughput_tokens_per_s_per_hbm_gb: json_f64_field(
            hardware_footprint,
            "throughput_tokens_per_s_per_hbm_gb",
        ),
        calibration_status: json_str_field(calibration_summary, "status"),
        calibration_coverage_fraction: json_f64_field(calibration_summary, "coverage_fraction"),
        calibration_fit_count: json_u64_field(calibration_summary, "fit_count"),
        calibration_fit_count_with_uncertainty: json_u64_field(
            calibration_summary,
            "fit_count_with_uncertainty",
        ),
        calibration_relative_uncertainty_pct: json_f64_field(
            calibration_summary,
            "relative_uncertainty_pct",
        ),
        calibration_absolute_uncertainty_ms: json_f64_field(
            calibration_summary,
            "absolute_uncertainty_ms",
        ),
        calibration_gate_violation_count: json_u64_field(
            calibration_summary,
            "gate_violation_count",
        ),
        calibration_hard_gate_violation_count: json_u64_field(
            calibration_summary,
            "hard_gate_violation_count",
        ),
        approximation_status: json_str_field(approximation_summary, "status"),
        approximation_count: json_u64_field(approximation_summary, "approximation_count"),
        approximation_policy_violation_count: json_u64_field(
            approximation_summary,
            "policy_violation_count",
        ),
        approximation_calibration_count: json_u64_field(approximation_summary, "calibration_count"),
        approximation_topology_count: json_u64_field(approximation_summary, "topology_count"),
        approximation_queueing_count: json_u64_field(approximation_summary, "queueing_count"),
        approximation_coarse_topology: json_bool_field(approximation_summary, "coarse_topology"),
        approximation_approximate_queueing: json_bool_field(
            approximation_summary,
            "approximate_queueing",
        ),
        approximation_uncalibrated_runtime: json_bool_field(
            approximation_summary,
            "uncalibrated_runtime",
        ),
        approximation_category_counts: json_count_list_field(
            approximation_summary,
            "category_counts",
        ),
        approximation_top_codes: json_count_list_field(approximation_summary, "top_codes"),
        bottleneck_count: top
            .get("bottleneck_summary")
            .and_then(serde_json::Value::as_array)
            .map(|bottlenecks| bottlenecks.len().min(u64::MAX as usize) as u64),
        top_bottleneck_source: json_str_field(top_bottleneck, "source"),
        top_bottleneck_category: json_str_field(top_bottleneck, "category"),
        top_bottleneck_code: json_str_field(top_bottleneck, "code"),
        top_bottleneck_severity: json_str_field(top_bottleneck, "severity"),
        rejection_count: top
            .get("rejections")
            .and_then(serde_json::Value::as_array)
            .map(|rejections| rejections.len().min(u64::MAX as usize) as u64),
        top_rejection_phase: json_str_field(top_rejection, "phase"),
        top_rejection_category: json_str_field(top_rejection, "category"),
        top_rejection_resource: json_str_field(top_rejection, "resource"),
        top_rejection_code: json_str_field(top_rejection, "code"),
        top_rejection_unit: json_str_field(top_rejection, "unit"),
        top_rejection_remediation: json_str_field(top_rejection, "remediation"),
        ttft_ms,
        tpot_ms,
        throughput_tokens_per_s,
        e2el_ms,
    })
}

fn json_str_field(value: &serde_json::Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn json_f64_field(value: &serde_json::Value, field: &str) -> Option<f64> {
    value
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
}

fn json_u64_field(value: &serde_json::Value, field: &str) -> Option<u64> {
    value.get(field).and_then(serde_json::Value::as_u64)
}

fn json_bool_field(value: &serde_json::Value, field: &str) -> Option<bool> {
    value.get(field).and_then(serde_json::Value::as_bool)
}

fn json_count_list_field(value: &serde_json::Value, field: &str) -> Option<String> {
    json_named_count_list_field(value, field, "name")
}

fn json_named_count_list_field(
    value: &serde_json::Value,
    field: &str,
    name_field: &str,
) -> Option<String> {
    let counts = value.get(field)?.as_array()?;
    Some(
        counts
            .iter()
            .filter_map(|count| {
                let name = count.get(name_field)?.as_str()?;
                let count = count.get("count")?.as_u64()?;
                Some(format!("{name}:{count}"))
            })
            .collect::<Vec<_>>()
            .join("|"),
    )
}

fn write_scenario_sensitivity_json<W: Write>(
    writer: &mut W,
    sensitivity: &[ScenarioSensitivity],
) -> Result<(), CliError> {
    let baseline = sensitivity.iter().find(|entry| entry.available);
    writeln!(writer, "  \"scenario_sensitivity\": [")?;
    for (idx, entry) in sensitivity.iter().enumerate() {
        let comparison_baseline = if entry.available { baseline } else { None };
        writeln!(writer, "    {{")?;
        writeln!(writer, "      \"name\": {},", json_string(&entry.name))?;
        writeln!(writer, "      \"index\": {},", entry.index)?;
        writeln!(writer, "      \"available\": {},", entry.available)?;
        writeln!(
            writer,
            "      \"reason\": {},",
            entry
                .reason
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "      \"baseline\": {},",
            baseline.is_some_and(|baseline| baseline.index == entry.index)
        )?;
        writeln!(
            writer,
            "      \"baseline_name\": {},",
            baseline
                .map(|baseline| json_string(&baseline.name))
                .unwrap_or_else(|| "null".to_string())
        )?;
        write_scenario_sensitivity_string(writer, "candidate_id", entry.candidate_id.as_deref())?;
        write_scenario_sensitivity_string(writer, "status", entry.status.as_deref())?;
        write_scenario_sensitivity_string(
            writer,
            "rejected_reason",
            entry.rejected_reason.as_deref(),
        )?;
        writeln!(
            writer,
            "      \"feasible\": {},",
            entry
                .feasible
                .map(|value| value.to_string())
                .unwrap_or_else(|| "null".to_string())
        )?;
        write_scenario_sensitivity_string(writer, "objective", entry.objective.as_deref())?;
        write_scenario_sensitivity_string(
            writer,
            "deployment_mode",
            entry.deployment_mode.as_deref(),
        )?;
        write_scenario_sensitivity_string(writer, "pool", entry.pool.as_deref())?;
        write_scenario_sensitivity_u64(
            writer,
            "hardware_unique_node_count",
            entry.hardware_unique_node_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "hardware_unique_gpu_count",
            entry.hardware_unique_gpu_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "hardware_prefill_gpu_count",
            entry.hardware_prefill_gpu_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "hardware_decode_gpu_count",
            entry.hardware_decode_gpu_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "hardware_shared_gpu_count",
            entry.hardware_shared_gpu_count,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_aggregate_hbm_gb",
            entry.hardware_aggregate_hbm_gb,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_prefill_hbm_gb",
            entry.hardware_prefill_hbm_gb,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_decode_hbm_gb",
            entry.hardware_decode_hbm_gb,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_aggregate_effective_peak_tflops",
            entry.hardware_aggregate_effective_peak_tflops,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_prefill_effective_peak_tflops",
            entry.hardware_prefill_effective_peak_tflops,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_decode_effective_peak_tflops",
            entry.hardware_decode_effective_peak_tflops,
        )?;
        write_scenario_sensitivity_string(
            writer,
            "hardware_aggregate_gpu_types",
            entry.hardware_aggregate_gpu_types.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "hardware_prefill_gpu_types",
            entry.hardware_prefill_gpu_types.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "hardware_decode_gpu_types",
            entry.hardware_decode_gpu_types.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "hardware_aggregate_gpu_label_counts",
            entry.hardware_aggregate_gpu_label_counts.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "hardware_prefill_gpu_label_counts",
            entry.hardware_prefill_gpu_label_counts.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "hardware_decode_gpu_label_counts",
            entry.hardware_decode_gpu_label_counts.as_deref(),
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_throughput_tokens_per_s_per_gpu",
            entry.hardware_throughput_tokens_per_s_per_gpu,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_throughput_tokens_per_s_per_effective_peak_tflop",
            entry.hardware_throughput_tokens_per_s_per_effective_peak_tflop,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "hardware_throughput_tokens_per_s_per_hbm_gb",
            entry.hardware_throughput_tokens_per_s_per_hbm_gb,
        )?;
        write_scenario_sensitivity_string(
            writer,
            "calibration_status",
            entry.calibration_status.as_deref(),
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "calibration_coverage_fraction",
            entry.calibration_coverage_fraction,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "calibration_fit_count",
            entry.calibration_fit_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "calibration_fit_count_with_uncertainty",
            entry.calibration_fit_count_with_uncertainty,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "calibration_relative_uncertainty_pct",
            entry.calibration_relative_uncertainty_pct,
        )?;
        write_scenario_sensitivity_f64(
            writer,
            "calibration_absolute_uncertainty_ms",
            entry.calibration_absolute_uncertainty_ms,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "calibration_gate_violation_count",
            entry.calibration_gate_violation_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "calibration_hard_gate_violation_count",
            entry.calibration_hard_gate_violation_count,
        )?;
        write_scenario_sensitivity_string(
            writer,
            "approximation_status",
            entry.approximation_status.as_deref(),
        )?;
        write_scenario_sensitivity_u64(writer, "approximation_count", entry.approximation_count)?;
        write_scenario_sensitivity_u64(
            writer,
            "approximation_policy_violation_count",
            entry.approximation_policy_violation_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "approximation_calibration_count",
            entry.approximation_calibration_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "approximation_topology_count",
            entry.approximation_topology_count,
        )?;
        write_scenario_sensitivity_u64(
            writer,
            "approximation_queueing_count",
            entry.approximation_queueing_count,
        )?;
        write_scenario_sensitivity_bool(
            writer,
            "approximation_coarse_topology",
            entry.approximation_coarse_topology,
        )?;
        write_scenario_sensitivity_bool(
            writer,
            "approximation_approximate_queueing",
            entry.approximation_approximate_queueing,
        )?;
        write_scenario_sensitivity_bool(
            writer,
            "approximation_uncalibrated_runtime",
            entry.approximation_uncalibrated_runtime,
        )?;
        write_scenario_sensitivity_string(
            writer,
            "approximation_category_counts",
            entry.approximation_category_counts.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "approximation_top_codes",
            entry.approximation_top_codes.as_deref(),
        )?;
        write_scenario_sensitivity_u64(writer, "bottleneck_count", entry.bottleneck_count)?;
        write_scenario_sensitivity_string(
            writer,
            "top_bottleneck_source",
            entry.top_bottleneck_source.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_bottleneck_category",
            entry.top_bottleneck_category.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_bottleneck_code",
            entry.top_bottleneck_code.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_bottleneck_severity",
            entry.top_bottleneck_severity.as_deref(),
        )?;
        write_scenario_sensitivity_u64(writer, "rejection_count", entry.rejection_count)?;
        write_scenario_sensitivity_string(
            writer,
            "top_rejection_phase",
            entry.top_rejection_phase.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_rejection_category",
            entry.top_rejection_category.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_rejection_resource",
            entry.top_rejection_resource.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_rejection_code",
            entry.top_rejection_code.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_rejection_unit",
            entry.top_rejection_unit.as_deref(),
        )?;
        write_scenario_sensitivity_string(
            writer,
            "top_rejection_remediation",
            entry.top_rejection_remediation.as_deref(),
        )?;
        write_scenario_metric_delta(
            writer,
            "ttft_ms",
            entry.ttft_ms,
            comparison_baseline.and_then(|baseline| baseline.ttft_ms),
            true,
        )?;
        write_scenario_metric_delta(
            writer,
            "tpot_ms",
            entry.tpot_ms,
            comparison_baseline.and_then(|baseline| baseline.tpot_ms),
            true,
        )?;
        write_scenario_metric_delta(
            writer,
            "throughput_tokens_per_s",
            entry.throughput_tokens_per_s,
            comparison_baseline.and_then(|baseline| baseline.throughput_tokens_per_s),
            true,
        )?;
        write_scenario_metric_delta(
            writer,
            "e2el_ms",
            entry.e2el_ms,
            comparison_baseline.and_then(|baseline| baseline.e2el_ms),
            false,
        )?;
        writeln!(writer, "    }}{}", comma(idx + 1 < sensitivity.len()))?;
    }
    writeln!(writer, "  ]")?;
    Ok(())
}

fn write_scenario_sensitivity_string<W: Write>(
    writer: &mut W,
    field: &str,
    value: Option<&str>,
) -> Result<(), CliError> {
    writeln!(
        writer,
        "      \"{field}\": {},",
        value.map(json_string).unwrap_or_else(|| "null".to_string())
    )?;
    Ok(())
}

fn write_scenario_sensitivity_f64<W: Write>(
    writer: &mut W,
    field: &str,
    value: Option<f64>,
) -> Result<(), CliError> {
    writeln!(writer, "      \"{field}\": {},", json_optional_value(value))?;
    Ok(())
}

fn write_scenario_sensitivity_u64<W: Write>(
    writer: &mut W,
    field: &str,
    value: Option<u64>,
) -> Result<(), CliError> {
    writeln!(writer, "      \"{field}\": {},", json_optional_u64(value))?;
    Ok(())
}

fn write_scenario_sensitivity_bool<W: Write>(
    writer: &mut W,
    field: &str,
    value: Option<bool>,
) -> Result<(), CliError> {
    writeln!(
        writer,
        "      \"{field}\": {},",
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string())
    )?;
    Ok(())
}

fn write_scenario_metric_delta<W: Write>(
    writer: &mut W,
    field: &str,
    value: Option<f64>,
    baseline: Option<f64>,
    trailing_comma: bool,
) -> Result<(), CliError> {
    let (delta, delta_pct) = scenario_metric_delta_values(value, baseline);
    writeln!(writer, "      \"{field}\": {},", json_optional_value(value))?;
    writeln!(
        writer,
        "      \"{field}_delta\": {},",
        json_optional_value(delta)
    )?;
    writeln!(
        writer,
        "      \"{field}_delta_pct\": {}{}",
        json_optional_value(delta_pct),
        comma(trailing_comma)
    )?;
    Ok(())
}

pub(super) fn scenario_metric_delta_values(
    value: Option<f64>,
    baseline: Option<f64>,
) -> (Option<f64>, Option<f64>) {
    let delta = value
        .zip(baseline)
        .map(|(value, baseline)| value - baseline);
    let delta_pct = value.zip(baseline).and_then(|(value, baseline)| {
        (baseline.abs() > f64::EPSILON).then_some((value - baseline) / baseline * 100.0)
    });
    (delta, delta_pct)
}

fn validate_scenario_workload(
    cluster: &Cluster,
    workload: &WorkloadConfig,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    validate_workload_for_cluster(cluster, workload).map_err(|err| {
        CliError::Usage(format!(
            "scenario '{}' produced an invalid workload: {err}",
            scenario.name
        ))
    })
}

fn write_scenario_topology_text<W: Write>(
    writer: &mut W,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    if !scenario_topology_has_overrides(&scenario.topology) {
        return Ok(());
    }
    writeln!(
        writer,
        "scenario_topology interconnect_bandwidth_scale={} interconnect_latency_scale={} nic_bandwidth_scale={} node_states={} disabled_gpus={} disabled_nics={} degraded_gpus={} degraded_nics={} degraded_rails={} degraded_links={}",
        scenario
            .topology
            .interconnect_bandwidth_scale
            .map(json_f64)
            .unwrap_or_else(|| "1.000000".to_string()),
        scenario
            .topology
            .interconnect_latency_scale
            .map(json_f64)
            .unwrap_or_else(|| "1.000000".to_string()),
        scenario
            .topology
            .nic_bandwidth_scale
            .map(json_f64)
            .unwrap_or_else(|| "1.000000".to_string()),
        format_scenario_node_state_overlays_text(&scenario.topology.node_states),
        format_scenario_gpu_overlays_text(&scenario.topology.disabled_gpus),
        format_scenario_nic_overlays_text(&scenario.topology.disabled_nics),
        format_scenario_degraded_gpu_overlays_text(&scenario.topology.degraded_gpus),
        format_scenario_degraded_nic_overlays_text(&scenario.topology.degraded_nics),
        format_scenario_degraded_rail_overlays_text(&scenario.topology.degraded_rails),
        format_scenario_degraded_link_overlays_text(&scenario.topology.degraded_links)
    )?;
    Ok(())
}

fn write_scenario_topology_json<W: Write>(
    writer: &mut W,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    if !scenario_topology_has_overrides(&scenario.topology) {
        return Ok(());
    }
    writeln!(writer, "      \"topology\": {{")?;
    writeln!(
        writer,
        "        \"interconnect_bandwidth_scale\": {},",
        json_optional_value(scenario.topology.interconnect_bandwidth_scale)
    )?;
    writeln!(
        writer,
        "        \"interconnect_latency_scale\": {},",
        json_optional_value(scenario.topology.interconnect_latency_scale)
    )?;
    writeln!(
        writer,
        "        \"nic_bandwidth_scale\": {},",
        json_optional_value(scenario.topology.nic_bandwidth_scale)
    )?;
    write_scenario_node_state_overlays_json(writer, &scenario.topology.node_states, true)?;
    write_scenario_gpu_overlays_json(writer, &scenario.topology.disabled_gpus, true)?;
    write_scenario_nic_overlays_json(writer, &scenario.topology.disabled_nics, true)?;
    write_scenario_degraded_gpu_overlays_json(writer, &scenario.topology.degraded_gpus, true)?;
    write_scenario_degraded_nic_overlays_json(writer, &scenario.topology.degraded_nics, true)?;
    write_scenario_degraded_rail_overlays_json(writer, &scenario.topology.degraded_rails, true)?;
    write_scenario_degraded_link_overlays_json(writer, &scenario.topology.degraded_links, false)?;
    writeln!(writer, "      }},")?;
    Ok(())
}

fn write_scenario_node_state_overlays_json<W: Write>(
    writer: &mut W,
    overlays: &[RunScenarioNodeStateOverlay],
    trailing_comma: bool,
) -> Result<(), CliError> {
    writeln!(writer, "        \"node_states\": [")?;
    for (idx, overlay) in overlays.iter().enumerate() {
        writeln!(writer, "          {{")?;
        writeln!(
            writer,
            "            \"node_ids\": [{}],",
            u32_list(&overlay.node_ids)
        )?;
        writeln!(
            writer,
            "            \"node_groups\": [{}],",
            string_list(&overlay.node_groups)
        )?;
        writeln!(
            writer,
            "            \"node_tags\": [{}],",
            string_list(&overlay.node_tags)
        )?;
        writeln!(
            writer,
            "            \"racks\": [{}],",
            string_list(&overlay.racks)
        )?;
        writeln!(
            writer,
            "            \"islands\": [{}],",
            string_list(&overlay.islands)
        )?;
        writeln!(
            writer,
            "            \"failure_domains\": [{}],",
            string_list(&overlay.failure_domains)
        )?;
        writeln!(
            writer,
            "            \"state\": {}",
            json_string(overlay.state.as_str())
        )?;
        if idx + 1 < overlays.len() {
            writeln!(writer, "          }},")?;
        } else {
            writeln!(writer, "          }}")?;
        }
    }
    writeln!(writer, "        ]{}", comma(trailing_comma))?;
    Ok(())
}

fn write_scenario_gpu_overlays_json<W: Write>(
    writer: &mut W,
    overlays: &[RunScenarioGpuResourceOverlay],
    trailing_comma: bool,
) -> Result<(), CliError> {
    writeln!(writer, "        \"disabled_gpus\": [")?;
    for (idx, overlay) in overlays.iter().enumerate() {
        writeln!(writer, "          {{")?;
        writeln!(
            writer,
            "            \"node_ids\": [{}],",
            u32_list(&overlay.node_ids)
        )?;
        writeln!(
            writer,
            "            \"node_groups\": [{}],",
            string_list(&overlay.node_groups)
        )?;
        writeln!(
            writer,
            "            \"node_tags\": [{}],",
            string_list(&overlay.node_tags)
        )?;
        writeln!(
            writer,
            "            \"racks\": [{}],",
            string_list(&overlay.racks)
        )?;
        writeln!(
            writer,
            "            \"islands\": [{}],",
            string_list(&overlay.islands)
        )?;
        writeln!(
            writer,
            "            \"failure_domains\": [{}],",
            string_list(&overlay.failure_domains)
        )?;
        writeln!(
            writer,
            "            \"gpu_ids\": [{}]",
            u32_list(&overlay.gpu_ids)
        )?;
        if idx + 1 < overlays.len() {
            writeln!(writer, "          }},")?;
        } else {
            writeln!(writer, "          }}")?;
        }
    }
    writeln!(writer, "        ]{}", comma(trailing_comma))?;
    Ok(())
}

fn write_scenario_nic_overlays_json<W: Write>(
    writer: &mut W,
    overlays: &[RunScenarioNicResourceOverlay],
    trailing_comma: bool,
) -> Result<(), CliError> {
    writeln!(writer, "        \"disabled_nics\": [")?;
    for (idx, overlay) in overlays.iter().enumerate() {
        writeln!(writer, "          {{")?;
        writeln!(
            writer,
            "            \"node_ids\": [{}],",
            u32_list(&overlay.node_ids)
        )?;
        writeln!(
            writer,
            "            \"node_groups\": [{}],",
            string_list(&overlay.node_groups)
        )?;
        writeln!(
            writer,
            "            \"node_tags\": [{}],",
            string_list(&overlay.node_tags)
        )?;
        writeln!(
            writer,
            "            \"racks\": [{}],",
            string_list(&overlay.racks)
        )?;
        writeln!(
            writer,
            "            \"islands\": [{}],",
            string_list(&overlay.islands)
        )?;
        writeln!(
            writer,
            "            \"failure_domains\": [{}],",
            string_list(&overlay.failure_domains)
        )?;
        writeln!(
            writer,
            "            \"nic_ids\": [{}]",
            u32_list(&overlay.nic_ids)
        )?;
        if idx + 1 < overlays.len() {
            writeln!(writer, "          }},")?;
        } else {
            writeln!(writer, "          }}")?;
        }
    }
    writeln!(writer, "        ]{}", comma(trailing_comma))?;
    Ok(())
}

fn write_scenario_degraded_nic_overlays_json<W: Write>(
    writer: &mut W,
    overlays: &[RunScenarioNicDegradationOverlay],
    trailing_comma: bool,
) -> Result<(), CliError> {
    writeln!(writer, "        \"degraded_nics\": [")?;
    for (idx, overlay) in overlays.iter().enumerate() {
        writeln!(writer, "          {{")?;
        writeln!(
            writer,
            "            \"node_ids\": [{}],",
            u32_list(&overlay.node_ids)
        )?;
        writeln!(
            writer,
            "            \"node_groups\": [{}],",
            string_list(&overlay.node_groups)
        )?;
        writeln!(
            writer,
            "            \"node_tags\": [{}],",
            string_list(&overlay.node_tags)
        )?;
        writeln!(
            writer,
            "            \"racks\": [{}],",
            string_list(&overlay.racks)
        )?;
        writeln!(
            writer,
            "            \"islands\": [{}],",
            string_list(&overlay.islands)
        )?;
        writeln!(
            writer,
            "            \"failure_domains\": [{}],",
            string_list(&overlay.failure_domains)
        )?;
        writeln!(
            writer,
            "            \"nic_ids\": [{}],",
            u32_list(&overlay.nic_ids)
        )?;
        writeln!(
            writer,
            "            \"bandwidth_scale\": {},",
            json_optional_value(overlay.bandwidth_scale)
        )?;
        writeln!(
            writer,
            "            \"latency_scale\": {}",
            json_optional_value(overlay.latency_scale)
        )?;
        if idx + 1 < overlays.len() {
            writeln!(writer, "          }},")?;
        } else {
            writeln!(writer, "          }}")?;
        }
    }
    writeln!(writer, "        ]{}", comma(trailing_comma))?;
    Ok(())
}

fn write_scenario_degraded_rail_overlays_json<W: Write>(
    writer: &mut W,
    overlays: &[RunScenarioRailDegradationOverlay],
    trailing_comma: bool,
) -> Result<(), CliError> {
    writeln!(writer, "        \"degraded_rails\": [")?;
    for (idx, overlay) in overlays.iter().enumerate() {
        writeln!(writer, "          {{")?;
        writeln!(
            writer,
            "            \"node_ids\": [{}],",
            u32_list(&overlay.node_ids)
        )?;
        writeln!(
            writer,
            "            \"node_groups\": [{}],",
            string_list(&overlay.node_groups)
        )?;
        writeln!(
            writer,
            "            \"node_tags\": [{}],",
            string_list(&overlay.node_tags)
        )?;
        writeln!(
            writer,
            "            \"racks\": [{}],",
            string_list(&overlay.racks)
        )?;
        writeln!(
            writer,
            "            \"islands\": [{}],",
            string_list(&overlay.islands)
        )?;
        writeln!(
            writer,
            "            \"failure_domains\": [{}],",
            string_list(&overlay.failure_domains)
        )?;
        writeln!(
            writer,
            "            \"rails\": [{}],",
            u32_list(&overlay.rails)
        )?;
        writeln!(
            writer,
            "            \"bandwidth_scale\": {},",
            json_optional_value(overlay.bandwidth_scale)
        )?;
        writeln!(
            writer,
            "            \"latency_scale\": {}",
            json_optional_value(overlay.latency_scale)
        )?;
        if idx + 1 < overlays.len() {
            writeln!(writer, "          }},")?;
        } else {
            writeln!(writer, "          }}")?;
        }
    }
    writeln!(writer, "        ]{}", comma(trailing_comma))?;
    Ok(())
}

fn write_scenario_degraded_gpu_overlays_json<W: Write>(
    writer: &mut W,
    overlays: &[RunScenarioGpuDegradationOverlay],
    trailing_comma: bool,
) -> Result<(), CliError> {
    writeln!(writer, "        \"degraded_gpus\": [")?;
    for (idx, overlay) in overlays.iter().enumerate() {
        writeln!(writer, "          {{")?;
        writeln!(
            writer,
            "            \"node_ids\": [{}],",
            u32_list(&overlay.node_ids)
        )?;
        writeln!(
            writer,
            "            \"node_groups\": [{}],",
            string_list(&overlay.node_groups)
        )?;
        writeln!(
            writer,
            "            \"node_tags\": [{}],",
            string_list(&overlay.node_tags)
        )?;
        writeln!(
            writer,
            "            \"racks\": [{}],",
            string_list(&overlay.racks)
        )?;
        writeln!(
            writer,
            "            \"islands\": [{}],",
            string_list(&overlay.islands)
        )?;
        writeln!(
            writer,
            "            \"failure_domains\": [{}],",
            string_list(&overlay.failure_domains)
        )?;
        writeln!(
            writer,
            "            \"gpu_ids\": [{}],",
            u32_list(&overlay.gpu_ids)
        )?;
        writeln!(
            writer,
            "            \"compute_scale\": {},",
            json_optional_value(overlay.compute_scale)
        )?;
        writeln!(
            writer,
            "            \"hbm_bandwidth_scale\": {},",
            json_optional_value(overlay.hbm_bandwidth_scale)
        )?;
        writeln!(
            writer,
            "            \"hbm_capacity_scale\": {}",
            json_optional_value(overlay.hbm_capacity_scale)
        )?;
        if idx + 1 < overlays.len() {
            writeln!(writer, "          }},")?;
        } else {
            writeln!(writer, "          }}")?;
        }
    }
    writeln!(writer, "        ]{}", comma(trailing_comma))?;
    Ok(())
}

fn write_scenario_degraded_link_overlays_json<W: Write>(
    writer: &mut W,
    overlays: &[RunScenarioLinkDegradationOverlay],
    trailing_comma: bool,
) -> Result<(), CliError> {
    writeln!(writer, "        \"degraded_links\": [")?;
    for (idx, overlay) in overlays.iter().enumerate() {
        writeln!(writer, "          {{")?;
        writeln!(
            writer,
            "            \"from_node_ids\": [{}],",
            u32_list(&overlay.from_node_ids)
        )?;
        writeln!(
            writer,
            "            \"from_node_groups\": [{}],",
            string_list(&overlay.from_node_groups)
        )?;
        writeln!(
            writer,
            "            \"from_node_tags\": [{}],",
            string_list(&overlay.from_node_tags)
        )?;
        writeln!(
            writer,
            "            \"from_racks\": [{}],",
            string_list(&overlay.from_racks)
        )?;
        writeln!(
            writer,
            "            \"from_islands\": [{}],",
            string_list(&overlay.from_islands)
        )?;
        writeln!(
            writer,
            "            \"from_failure_domains\": [{}],",
            string_list(&overlay.from_failure_domains)
        )?;
        writeln!(
            writer,
            "            \"from_gpus\": [{}],",
            u32_list(&overlay.from_gpus)
        )?;
        writeln!(
            writer,
            "            \"to_node_ids\": [{}],",
            u32_list(&overlay.to_node_ids)
        )?;
        writeln!(
            writer,
            "            \"to_node_groups\": [{}],",
            string_list(&overlay.to_node_groups)
        )?;
        writeln!(
            writer,
            "            \"to_node_tags\": [{}],",
            string_list(&overlay.to_node_tags)
        )?;
        writeln!(
            writer,
            "            \"to_racks\": [{}],",
            string_list(&overlay.to_racks)
        )?;
        writeln!(
            writer,
            "            \"to_islands\": [{}],",
            string_list(&overlay.to_islands)
        )?;
        writeln!(
            writer,
            "            \"to_failure_domains\": [{}],",
            string_list(&overlay.to_failure_domains)
        )?;
        writeln!(
            writer,
            "            \"to_gpus\": [{}],",
            u32_list(&overlay.to_gpus)
        )?;
        writeln!(
            writer,
            "            \"rails\": [{}],",
            u32_list(&overlay.rails)
        )?;
        writeln!(
            writer,
            "            \"bandwidth_scale\": {},",
            json_optional_value(overlay.bandwidth_scale)
        )?;
        writeln!(
            writer,
            "            \"latency_scale\": {}",
            json_optional_value(overlay.latency_scale)
        )?;
        if idx + 1 < overlays.len() {
            writeln!(writer, "          }},")?;
        } else {
            writeln!(writer, "          }}")?;
        }
    }
    writeln!(writer, "        ]{}", comma(trailing_comma))?;
    Ok(())
}

fn format_scenario_node_state_overlays_text(overlays: &[RunScenarioNodeStateOverlay]) -> String {
    if overlays.is_empty() {
        return "[]".to_string();
    }
    overlays
        .iter()
        .map(|overlay| {
            format!(
                "{{nodes={},groups={},tags={},racks={},islands={},failure_domains={},state={}}}",
                text_u32_list(&overlay.node_ids),
                text_string_list(&overlay.node_groups),
                text_string_list(&overlay.node_tags),
                text_string_list(&overlay.racks),
                text_string_list(&overlay.islands),
                text_string_list(&overlay.failure_domains),
                overlay.state.as_str()
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn format_scenario_gpu_overlays_text(overlays: &[RunScenarioGpuResourceOverlay]) -> String {
    if overlays.is_empty() {
        return "[]".to_string();
    }
    overlays
        .iter()
        .map(|overlay| {
            format!(
                "{{nodes={},groups={},tags={},racks={},islands={},failure_domains={},gpus={}}}",
                text_u32_list(&overlay.node_ids),
                text_string_list(&overlay.node_groups),
                text_string_list(&overlay.node_tags),
                text_string_list(&overlay.racks),
                text_string_list(&overlay.islands),
                text_string_list(&overlay.failure_domains),
                text_u32_list(&overlay.gpu_ids)
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn format_scenario_nic_overlays_text(overlays: &[RunScenarioNicResourceOverlay]) -> String {
    if overlays.is_empty() {
        return "[]".to_string();
    }
    overlays
        .iter()
        .map(|overlay| {
            format!(
                "{{nodes={},groups={},tags={},racks={},islands={},failure_domains={},nics={}}}",
                text_u32_list(&overlay.node_ids),
                text_string_list(&overlay.node_groups),
                text_string_list(&overlay.node_tags),
                text_string_list(&overlay.racks),
                text_string_list(&overlay.islands),
                text_string_list(&overlay.failure_domains),
                text_u32_list(&overlay.nic_ids)
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn format_scenario_degraded_nic_overlays_text(
    overlays: &[RunScenarioNicDegradationOverlay],
) -> String {
    if overlays.is_empty() {
        return "[]".to_string();
    }
    overlays
        .iter()
        .map(|overlay| {
            format!(
                "{{nodes={},groups={},tags={},racks={},islands={},failure_domains={},nics={},bandwidth_scale={},latency_scale={}}}",
                text_u32_list(&overlay.node_ids),
                text_string_list(&overlay.node_groups),
                text_string_list(&overlay.node_tags),
                text_string_list(&overlay.racks),
                text_string_list(&overlay.islands),
                text_string_list(&overlay.failure_domains),
                text_u32_list(&overlay.nic_ids),
                json_optional_value(overlay.bandwidth_scale),
                json_optional_value(overlay.latency_scale)
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn format_scenario_degraded_rail_overlays_text(
    overlays: &[RunScenarioRailDegradationOverlay],
) -> String {
    if overlays.is_empty() {
        return "[]".to_string();
    }
    overlays
        .iter()
        .map(|overlay| {
            format!(
                "{{nodes={},groups={},tags={},racks={},islands={},failure_domains={},rails={},bandwidth_scale={},latency_scale={}}}",
                text_u32_list(&overlay.node_ids),
                text_string_list(&overlay.node_groups),
                text_string_list(&overlay.node_tags),
                text_string_list(&overlay.racks),
                text_string_list(&overlay.islands),
                text_string_list(&overlay.failure_domains),
                text_u32_list(&overlay.rails),
                overlay
                    .bandwidth_scale
                    .map(json_f64)
                    .unwrap_or_else(|| "null".to_string()),
                overlay
                    .latency_scale
                    .map(json_f64)
                    .unwrap_or_else(|| "null".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn format_scenario_degraded_gpu_overlays_text(
    overlays: &[RunScenarioGpuDegradationOverlay],
) -> String {
    if overlays.is_empty() {
        return "[]".to_string();
    }
    overlays
        .iter()
        .map(|overlay| {
            format!(
                "{{nodes={},groups={},tags={},racks={},islands={},failure_domains={},gpus={},compute_scale={},hbm_bandwidth_scale={},hbm_capacity_scale={}}}",
                text_u32_list(&overlay.node_ids),
                text_string_list(&overlay.node_groups),
                text_string_list(&overlay.node_tags),
                text_string_list(&overlay.racks),
                text_string_list(&overlay.islands),
                text_string_list(&overlay.failure_domains),
                text_u32_list(&overlay.gpu_ids),
                overlay
                    .compute_scale
                    .map(json_f64)
                    .unwrap_or_else(|| "null".to_string()),
                overlay
                    .hbm_bandwidth_scale
                    .map(json_f64)
                    .unwrap_or_else(|| "null".to_string()),
                overlay
                    .hbm_capacity_scale
                    .map(json_f64)
                    .unwrap_or_else(|| "null".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn format_scenario_degraded_link_overlays_text(
    overlays: &[RunScenarioLinkDegradationOverlay],
) -> String {
    if overlays.is_empty() {
        return "[]".to_string();
    }
    overlays
        .iter()
        .map(|overlay| {
            format!(
                "{{from_nodes={},from_groups={},from_tags={},from_racks={},from_islands={},from_failure_domains={},from_gpus={},to_nodes={},to_groups={},to_tags={},to_racks={},to_islands={},to_failure_domains={},to_gpus={},rails={},bandwidth_scale={},latency_scale={}}}",
                text_u32_list(&overlay.from_node_ids),
                text_string_list(&overlay.from_node_groups),
                text_string_list(&overlay.from_node_tags),
                text_string_list(&overlay.from_racks),
                text_string_list(&overlay.from_islands),
                text_string_list(&overlay.from_failure_domains),
                text_u32_list(&overlay.from_gpus),
                text_u32_list(&overlay.to_node_ids),
                text_string_list(&overlay.to_node_groups),
                text_string_list(&overlay.to_node_tags),
                text_string_list(&overlay.to_racks),
                text_string_list(&overlay.to_islands),
                text_string_list(&overlay.to_failure_domains),
                text_u32_list(&overlay.to_gpus),
                text_u32_list(&overlay.rails),
                overlay
                    .bandwidth_scale
                    .map(json_f64)
                    .unwrap_or_else(|| "null".to_string()),
                overlay
                    .latency_scale
                    .map(json_f64)
                    .unwrap_or_else(|| "null".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

pub(super) fn text_u32_list(values: &[u32]) -> String {
    if values.is_empty() {
        return "-".to_string();
    }
    values
        .iter()
        .copied()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("|")
}

fn text_string_list(values: &[String]) -> String {
    if values.is_empty() {
        return "-".to_string();
    }
    values.join("|")
}

fn scenario_topology_has_overrides(topology: &RunScenarioTopologyConfig) -> bool {
    topology.interconnect_bandwidth_scale.is_some()
        || topology.interconnect_latency_scale.is_some()
        || topology.nic_bandwidth_scale.is_some()
        || !topology.node_states.is_empty()
        || !topology.disabled_gpus.is_empty()
        || !topology.disabled_nics.is_empty()
        || !topology.degraded_gpus.is_empty()
        || !topology.degraded_nics.is_empty()
        || !topology.degraded_rails.is_empty()
        || !topology.degraded_links.is_empty()
}

pub(super) fn apply_run_scenario_topology(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    let topology = &scenario.topology;
    if !scenario_topology_has_overrides(topology) {
        return Ok(());
    }
    if let Some(scale) = topology.nic_bandwidth_scale {
        for node in cluster.nodes.values_mut() {
            node.network.nic_bandwidth = node.network.nic_bandwidth * scale;
        }
    }
    scale_inter_node_topology(&mut cluster.inter_node_topology, topology);
    apply_degraded_gpu_overlays(cluster, scenario)?;
    apply_degraded_rail_overlays(cluster, scenario)?;
    apply_degraded_nic_overlays(cluster, scenario)?;
    apply_degraded_link_overlays(cluster, scenario)?;
    apply_node_state_overlays(cluster, scenario)?;
    apply_disabled_gpu_overlays(cluster, scenario)?;
    apply_disabled_nic_overlays(cluster, scenario)?;
    Ok(())
}

fn apply_node_state_overlays(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    for (idx, overlay) in scenario.topology.node_states.iter().enumerate() {
        let nodes = resolve_scenario_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.node_states[{idx}]"),
            &overlay.node_ids,
            &overlay.node_groups,
            &overlay.node_tags,
            &overlay.racks,
            &overlay.islands,
            &overlay.failure_domains,
        )?;
        for node_id in nodes {
            let node = cluster.nodes.get_mut(&node_id).ok_or_else(|| {
                CliError::Usage(format!(
                    "scenario '{}' topology.node_states[{idx}] references unknown node {node_id}",
                    scenario.name
                ))
            })?;
            apply_node_state(node, overlay.state);
        }
    }
    Ok(())
}

fn apply_node_state(node: &mut crate::types::topology::Node, state: RunScenarioNodeState) {
    node.operational_state = match state {
        RunScenarioNodeState::Disabled => NodeOperationalState::Disabled,
        RunScenarioNodeState::Maintenance => NodeOperationalState::Maintenance,
        RunScenarioNodeState::Draining => NodeOperationalState::Draining,
        RunScenarioNodeState::Reserved => NodeOperationalState::Reserved,
    };
    let resource_state = scenario_node_resource_state(state);
    for gpu_id in node.gpus.keys().copied() {
        node.disabled_gpus.insert(gpu_id);
        node.gpu_operational_states.insert(gpu_id, resource_state);
    }
    for nic_id in 0..u32::from(node.network.nic_count) {
        node.network.disabled_nics.insert(nic_id);
        node.network
            .nic_operational_states
            .insert(nic_id, resource_state);
    }
}

fn scenario_node_resource_state(state: RunScenarioNodeState) -> OperationalState {
    match state {
        RunScenarioNodeState::Disabled => OperationalState::Disabled,
        RunScenarioNodeState::Maintenance => OperationalState::Maintenance,
        RunScenarioNodeState::Draining => OperationalState::Draining,
        RunScenarioNodeState::Reserved => OperationalState::Reserved,
    }
}

fn apply_disabled_gpu_overlays(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    for (idx, overlay) in scenario.topology.disabled_gpus.iter().enumerate() {
        let nodes = resolve_scenario_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.disabled_gpus[{idx}]"),
            &overlay.node_ids,
            &overlay.node_groups,
            &overlay.node_tags,
            &overlay.racks,
            &overlay.islands,
            &overlay.failure_domains,
        )?;
        for node_id in nodes {
            let node = cluster.nodes.get_mut(&node_id).ok_or_else(|| {
                CliError::Usage(format!(
                    "scenario '{}' topology.disabled_gpus[{idx}] references unknown node {node_id}",
                    scenario.name
                ))
            })?;
            for gpu_id in &overlay.gpu_ids {
                if !node.gpus.contains_key(gpu_id) {
                    return Err(CliError::Usage(format!(
                        "scenario '{}' topology.disabled_gpus[{idx}] references node {node_id} GPU {gpu_id}, but that local GPU does not exist",
                        scenario.name
                    )));
                }
                node.disabled_gpus.insert(*gpu_id);
                node.gpu_operational_states
                    .insert(*gpu_id, OperationalState::Disabled);
            }
        }
    }
    Ok(())
}

fn apply_disabled_nic_overlays(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    for (idx, overlay) in scenario.topology.disabled_nics.iter().enumerate() {
        let nodes = resolve_scenario_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.disabled_nics[{idx}]"),
            &overlay.node_ids,
            &overlay.node_groups,
            &overlay.node_tags,
            &overlay.racks,
            &overlay.islands,
            &overlay.failure_domains,
        )?;
        for node_id in nodes {
            let node = cluster.nodes.get_mut(&node_id).ok_or_else(|| {
                CliError::Usage(format!(
                    "scenario '{}' topology.disabled_nics[{idx}] references unknown node {node_id}",
                    scenario.name
                ))
            })?;
            for nic_id in &overlay.nic_ids {
                if *nic_id >= u32::from(node.network.nic_count) {
                    return Err(CliError::Usage(format!(
                        "scenario '{}' topology.disabled_nics[{idx}] references node {node_id} NIC {nic_id}, but that node only has {} NICs",
                        scenario.name, node.network.nic_count
                    )));
                }
                node.network.disabled_nics.insert(*nic_id);
                node.network
                    .nic_operational_states
                    .insert(*nic_id, OperationalState::Disabled);
            }
        }
    }
    Ok(())
}

fn apply_degraded_gpu_overlays(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    for (idx, overlay) in scenario.topology.degraded_gpus.iter().enumerate() {
        let nodes = resolve_scenario_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.degraded_gpus[{idx}]"),
            &overlay.node_ids,
            &overlay.node_groups,
            &overlay.node_tags,
            &overlay.racks,
            &overlay.islands,
            &overlay.failure_domains,
        )?;
        for node_id in nodes {
            let node = cluster.nodes.get_mut(&node_id).ok_or_else(|| {
                CliError::Usage(format!(
                    "scenario '{}' topology.degraded_gpus[{idx}] references unknown node {node_id}",
                    scenario.name
                ))
            })?;
            for gpu_id in &overlay.gpu_ids {
                if !node.gpus.contains_key(gpu_id) {
                    return Err(CliError::Usage(format!(
                        "scenario '{}' topology.degraded_gpus[{idx}] references node {node_id} GPU {gpu_id}, but that local GPU does not exist",
                        scenario.name
                    )));
                }
                let base_profile = node.gpu_profile(*gpu_id).ok_or_else(|| {
                    CliError::Usage(format!(
                        "scenario '{}' topology.degraded_gpus[{idx}] could not resolve node {node_id} GPU {gpu_id}",
                        scenario.name
                    ))
                })?;
                node.gpu_profile_overrides
                    .insert(*gpu_id, degraded_gpu_profile(base_profile, overlay));
            }
        }
    }
    Ok(())
}

fn degraded_gpu_profile(
    mut profile: GpuProfile,
    overlay: &RunScenarioGpuDegradationOverlay,
) -> GpuProfile {
    if let Some(scale) = overlay.compute_scale {
        profile.peak_f16_flops *= scale;
        profile.peak_f8_flops = profile.peak_f8_flops.map(|flops| flops * scale);
    }
    if let Some(scale) = overlay.hbm_bandwidth_scale {
        profile.hbm_bandwidth = profile.hbm_bandwidth * scale;
    }
    if let Some(scale) = overlay.hbm_capacity_scale {
        profile.hbm_size = Bytes::from_bytes(
            ((profile.hbm_size.as_bytes() as f64) * scale)
                .round()
                .max(1.0) as u64,
        );
    }
    profile
}

fn apply_degraded_nic_overlays(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    for (idx, overlay) in scenario.topology.degraded_nics.iter().enumerate() {
        let nodes = resolve_scenario_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.degraded_nics[{idx}]"),
            &overlay.node_ids,
            &overlay.node_groups,
            &overlay.node_tags,
            &overlay.racks,
            &overlay.islands,
            &overlay.failure_domains,
        )?;
        for node_id in nodes {
            let node = cluster.nodes.get_mut(&node_id).ok_or_else(|| {
                CliError::Usage(format!(
                    "scenario '{}' topology.degraded_nics[{idx}] references unknown node {node_id}",
                    scenario.name
                ))
            })?;
            for nic_id in &overlay.nic_ids {
                if *nic_id >= u32::from(node.network.nic_count) {
                    return Err(CliError::Usage(format!(
                        "scenario '{}' topology.degraded_nics[{idx}] references node {node_id} NIC {nic_id}, but that node only has {} NICs",
                        scenario.name, node.network.nic_count
                    )));
                }
                if let Some(bandwidth_scale) = overlay.bandwidth_scale {
                    let degraded = node.network.nic_bandwidth(*nic_id) * bandwidth_scale;
                    node.network
                        .nic_bandwidth_overrides
                        .insert(*nic_id, degraded);
                }
                if let Some(latency_scale) = overlay.latency_scale {
                    let degraded = node.network.nic_latency_scale(*nic_id) * latency_scale;
                    node.network
                        .nic_latency_scale_overrides
                        .insert(*nic_id, degraded);
                }
            }
        }
    }
    Ok(())
}

fn apply_degraded_rail_overlays(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    for (idx, overlay) in scenario.topology.degraded_rails.iter().enumerate() {
        let nodes = resolve_scenario_optional_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.degraded_rails[{idx}]"),
            &overlay.node_ids,
            &overlay.node_groups,
            &overlay.node_tags,
            &overlay.racks,
            &overlay.islands,
            &overlay.failure_domains,
        )?;
        let selected_nodes = nodes.iter().copied().collect::<BTreeSet<_>>();
        let mut matched_nics = 0usize;

        for node_id in &nodes {
            let node = cluster.nodes.get_mut(node_id).ok_or_else(|| {
                CliError::Usage(format!(
                    "scenario '{}' topology.degraded_rails[{idx}] references unknown node {node_id}",
                    scenario.name
                ))
            })?;
            for nic_id in 0..u32::from(node.network.nic_count) {
                let rail_id = node.network.rail_id(nic_id);
                if !overlay.rails.contains(&rail_id) {
                    continue;
                }
                if let Some(bandwidth_scale) = overlay.bandwidth_scale {
                    let degraded = node.network.nic_bandwidth(nic_id) * bandwidth_scale;
                    node.network
                        .nic_bandwidth_overrides
                        .insert(nic_id, degraded);
                }
                if let Some(latency_scale) = overlay.latency_scale {
                    let degraded = node.network.nic_latency_scale(nic_id) * latency_scale;
                    node.network
                        .nic_latency_scale_overrides
                        .insert(nic_id, degraded);
                }
                matched_nics += 1;
            }
        }

        let matched_links = apply_degraded_rail_link_overlays(
            &mut cluster.inter_node_topology,
            &selected_nodes,
            &overlay.rails,
            overlay.bandwidth_scale,
            overlay.latency_scale,
        );

        if matched_nics == 0 && matched_links == 0 {
            return Err(CliError::Usage(format!(
                "scenario '{}' topology.degraded_rails[{idx}] did not match any NICs or custom inter-node links on rails [{}]",
                scenario.name,
                u32_list(&overlay.rails)
            )));
        }
    }
    Ok(())
}

fn apply_degraded_rail_link_overlays(
    topology: &mut InterNodeTopology,
    selected_nodes: &BTreeSet<u32>,
    rails: &[u32],
    bandwidth_scale: Option<f64>,
    latency_scale: Option<f64>,
) -> usize {
    let InterNodeTopology::Custom(edges) = topology else {
        return 0;
    };

    let mut matched = 0usize;
    for (pair, links) in edges.iter_mut() {
        let (left, right) = pair.endpoints();
        if !selected_nodes.contains(left) && !selected_nodes.contains(right) {
            continue;
        }
        for link in links {
            if !link.rail.is_some_and(|rail| rails.contains(&rail)) {
                continue;
            }
            if let Some(scale) = bandwidth_scale {
                link.profile.bw.unidirectional = link.profile.bw.unidirectional * scale;
            }
            if let Some(scale) = latency_scale {
                link.profile.latency = Latency::from_us(link.profile.latency.to_us() * scale);
            }
            matched += 1;
        }
    }
    matched
}

fn apply_degraded_link_overlays(
    cluster: &mut Cluster,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    if scenario.topology.degraded_links.is_empty() {
        return Ok(());
    }

    let mut specs = Vec::with_capacity(scenario.topology.degraded_links.len());
    for (idx, overlay) in scenario.topology.degraded_links.iter().enumerate() {
        let from_nodes = resolve_scenario_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.degraded_links[{idx}].from"),
            &overlay.from_node_ids,
            &overlay.from_node_groups,
            &overlay.from_node_tags,
            &overlay.from_racks,
            &overlay.from_islands,
            &overlay.from_failure_domains,
        )?;
        let to_nodes = resolve_scenario_overlay_nodes_with_topology(
            cluster,
            &scenario.name,
            &format!("topology.degraded_links[{idx}].to"),
            &overlay.to_node_ids,
            &overlay.to_node_groups,
            &overlay.to_node_tags,
            &overlay.to_racks,
            &overlay.to_islands,
            &overlay.to_failure_domains,
        )?;
        validate_degraded_link_endpoint_gpus(
            cluster,
            &scenario.name,
            idx,
            "from",
            &from_nodes,
            &overlay.from_gpus,
        )?;
        validate_degraded_link_endpoint_gpus(
            cluster,
            &scenario.name,
            idx,
            "to",
            &to_nodes,
            &overlay.to_gpus,
        )?;
        specs.push((
            idx,
            from_nodes,
            overlay.from_gpus.clone(),
            to_nodes,
            overlay.to_gpus.clone(),
            overlay.rails.clone(),
            overlay.bandwidth_scale,
            overlay.latency_scale,
        ));
    }

    let InterNodeTopology::Custom(edges) = &mut cluster.inter_node_topology else {
        return Err(CliError::Usage(format!(
            "scenario '{}' topology.degraded_links requires a custom inter-node topology",
            scenario.name
        )));
    };

    for (idx, from_nodes, from_gpus, to_nodes, to_gpus, rails, bandwidth_scale, latency_scale) in
        specs
    {
        let mut matched = 0_usize;
        let mut distinct_pairs = 0_usize;
        for from in &from_nodes {
            for to in &to_nodes {
                if from == to {
                    continue;
                }
                distinct_pairs += 1;
                let Some(links) = edges.get_mut(&UnorderedPair::new(*from, *to)) else {
                    continue;
                };
                let mut pair_matched = 0_usize;
                for link in links {
                    if !link_matches_degraded_link(link, *from, *to, &from_gpus, &to_gpus, &rails) {
                        continue;
                    }
                    if let Some(scale) = bandwidth_scale {
                        link.profile.bw.unidirectional = link.profile.bw.unidirectional * scale;
                    }
                    if let Some(scale) = latency_scale {
                        link.profile.latency =
                            Latency::from_us(link.profile.latency.to_us() * scale);
                    }
                    matched += 1;
                    pair_matched += 1;
                }
                if pair_matched == 0
                    && (!rails.is_empty() || !from_gpus.is_empty() || !to_gpus.is_empty())
                {
                    return Err(CliError::Usage(format!(
                        "scenario '{}' topology.degraded_links[{idx}] matched node pair {from}<->{to}, but none of its custom links match rails [{}] and GPU endpoint filters from_gpus=[{}] to_gpus=[{}]",
                        scenario.name,
                        u32_list(&rails),
                        u32_list(&from_gpus),
                        u32_list(&to_gpus)
                    )));
                }
            }
        }
        if distinct_pairs == 0 {
            return Err(CliError::Usage(format!(
                "scenario '{}' topology.degraded_links[{idx}] must resolve at least one distinct node pair",
                scenario.name
            )));
        }
        if matched == 0 {
            return Err(CliError::Usage(format!(
                "scenario '{}' topology.degraded_links[{idx}] did not match any custom inter-node links",
                scenario.name
            )));
        }
    }

    Ok(())
}

fn validate_degraded_link_endpoint_gpus(
    cluster: &Cluster,
    scenario_name: &str,
    idx: usize,
    endpoint: &str,
    nodes: &[u32],
    gpus: &[u32],
) -> Result<(), CliError> {
    for node_id in nodes {
        let Some(node) = cluster.node(*node_id) else {
            return Err(CliError::Usage(format!(
                "scenario '{scenario_name}' topology.degraded_links[{idx}].{endpoint} references unknown node {node_id}"
            )));
        };
        for gpu_id in gpus {
            if !node.gpus.contains_key(gpu_id) {
                return Err(CliError::Usage(format!(
                    "scenario '{scenario_name}' topology.degraded_links[{idx}].{endpoint}_gpus references node {node_id} GPU {gpu_id}, but that local GPU does not exist"
                )));
            }
        }
    }
    Ok(())
}

fn link_matches_degraded_link(
    link: &CustomInterNodeLink,
    from: u32,
    to: u32,
    from_gpus: &[u32],
    to_gpus: &[u32],
    rails: &[u32],
) -> bool {
    link_matches_degraded_link_rails(link, rails)
        && link_matches_degraded_link_gpus(link, from, to, from_gpus, to_gpus)
}

fn link_matches_degraded_link_rails(link: &CustomInterNodeLink, rails: &[u32]) -> bool {
    rails.is_empty() || link.rail.is_some_and(|rail| rails.contains(&rail))
}

fn link_matches_degraded_link_gpus(
    link: &CustomInterNodeLink,
    from: u32,
    to: u32,
    from_gpus: &[u32],
    to_gpus: &[u32],
) -> bool {
    if from_gpus.is_empty() && to_gpus.is_empty() {
        return true;
    }
    let Some((link_from_gpus, link_to_gpus)) = custom_link_gpu_scope_for_pair(link, from, to)
    else {
        return false;
    };
    gpu_scope_filter_matches(&link_from_gpus, from_gpus)
        && gpu_scope_filter_matches(&link_to_gpus, to_gpus)
}

fn custom_link_gpu_scope_for_pair(
    link: &CustomInterNodeLink,
    from: u32,
    to: u32,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let endpoints = link.endpoints.as_ref()?;
    if endpoints.from_node == from && endpoints.to_node == to {
        Some((endpoints.from_gpus.clone(), endpoints.to_gpus.clone()))
    } else if endpoints.from_node == to && endpoints.to_node == from {
        Some((endpoints.to_gpus.clone(), endpoints.from_gpus.clone()))
    } else {
        None
    }
}

fn gpu_scope_filter_matches(link_scope: &[u32], filter: &[u32]) -> bool {
    filter.is_empty() || link_scope.is_empty() || filter.iter().any(|gpu| link_scope.contains(gpu))
}

fn resolve_scenario_overlay_nodes(
    cluster: &Cluster,
    scenario_name: &str,
    overlay_name: &str,
    node_ids: &[u32],
    node_groups: &[String],
) -> Result<Vec<u32>, CliError> {
    let mut resolved = BTreeSet::new();
    for node_id in node_ids {
        if !cluster.nodes.contains_key(node_id) {
            return Err(CliError::Usage(format!(
                "scenario '{scenario_name}' {overlay_name} references unknown node {node_id}"
            )));
        }
        resolved.insert(*node_id);
    }
    for group in node_groups {
        let Some(nodes) = cluster.node_group(group) else {
            return Err(CliError::Usage(format!(
                "scenario '{scenario_name}' {overlay_name} references unknown node group '{group}'"
            )));
        };
        resolved.extend(nodes.iter().copied());
    }
    Ok(resolved.into_iter().collect())
}

#[allow(clippy::too_many_arguments)]
fn resolve_scenario_overlay_nodes_with_topology(
    cluster: &Cluster,
    scenario_name: &str,
    overlay_name: &str,
    node_ids: &[u32],
    node_groups: &[String],
    node_tags: &[String],
    racks: &[String],
    islands: &[String],
    failure_domains: &[String],
) -> Result<Vec<u32>, CliError> {
    let base_nodes = if node_ids.is_empty() && node_groups.is_empty() {
        let mut nodes = cluster.nodes.keys().copied().collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes
    } else {
        resolve_scenario_overlay_nodes(cluster, scenario_name, overlay_name, node_ids, node_groups)?
    };

    let selected = base_nodes
        .into_iter()
        .filter(|node_id| {
            cluster.node(*node_id).is_some_and(|node| {
                scenario_node_matches_topology_selector(
                    node,
                    node_tags,
                    racks,
                    islands,
                    failure_domains,
                )
            })
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(CliError::Usage(format!(
            "scenario '{scenario_name}' {overlay_name} selector matched no nodes"
        )));
    }
    Ok(selected)
}

fn scenario_node_matches_topology_selector(
    node: &crate::types::topology::Node,
    node_tags: &[String],
    racks: &[String],
    islands: &[String],
    failure_domains: &[String],
) -> bool {
    (node_tags.is_empty()
        || node_tags
            .iter()
            .any(|tag| node.topology.labels.contains(tag)))
        && (racks.is_empty()
            || node
                .topology
                .rack
                .as_ref()
                .is_some_and(|rack| racks.contains(rack)))
        && (islands.is_empty()
            || node
                .topology
                .island
                .as_ref()
                .is_some_and(|island| islands.contains(island)))
        && (failure_domains.is_empty()
            || node
                .topology
                .failure_domain
                .as_ref()
                .is_some_and(|failure_domain| failure_domains.contains(failure_domain)))
}

#[allow(clippy::too_many_arguments)]
fn resolve_scenario_optional_overlay_nodes_with_topology(
    cluster: &Cluster,
    scenario_name: &str,
    overlay_name: &str,
    node_ids: &[u32],
    node_groups: &[String],
    node_tags: &[String],
    racks: &[String],
    islands: &[String],
    failure_domains: &[String],
) -> Result<Vec<u32>, CliError> {
    resolve_scenario_overlay_nodes_with_topology(
        cluster,
        scenario_name,
        overlay_name,
        node_ids,
        node_groups,
        node_tags,
        racks,
        islands,
        failure_domains,
    )
}

fn scale_inter_node_topology(
    topology: &mut InterNodeTopology,
    scenario_topology: &RunScenarioTopologyConfig,
) {
    match topology {
        InterNodeTopology::FatTree { link, .. } | InterNodeTopology::Flat { link } => {
            scale_fabric_profile(link, scenario_topology);
        }
        InterNodeTopology::Custom(links) => {
            for link in links.values_mut().flatten() {
                scale_fabric_profile(&mut link.profile, scenario_topology);
            }
        }
    }
}

fn scale_fabric_profile(
    profile: &mut FabricProfile,
    scenario_topology: &RunScenarioTopologyConfig,
) {
    if let Some(scale) = scenario_topology.interconnect_bandwidth_scale {
        profile.bw.unidirectional = profile.bw.unidirectional * scale;
    }
    if let Some(scale) = scenario_topology.interconnect_latency_scale {
        profile.latency = Latency::from_us(profile.latency.to_us() * scale);
    }
}

fn apply_run_scenario(
    workload: &mut WorkloadConfig,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    apply_scenario_calibration_profile(workload, scenario)?;
    apply_scenario_calibration(&mut workload.calibration, scenario.calibration);
    apply_request_shape_scales(&mut workload.request, scenario);

    let has_serving_only_override = scenario.request_count.is_some()
        || scenario.arrival_gap_scale.is_some()
        || scenario.arrival_rate_scale.is_some();
    let Some(serving) = workload.serving.as_mut() else {
        if has_serving_only_override {
            return Err(CliError::Usage(format!(
                "scenario '{}' uses serving traffic overrides, but the workload has no [serving] section",
                scenario.name
            )));
        }
        refresh_workload_calibration_reports(workload);
        return Ok(());
    };

    if let Some(request_count) = scenario.request_count {
        apply_scenario_request_count(&mut serving.traffic, request_count, scenario)?;
    }
    apply_arrival_scales(&mut serving.traffic, workload.calibration, scenario);
    apply_serving_shape_scales(&mut serving.traffic, scenario);
    refresh_workload_calibration_reports(workload);

    Ok(())
}

fn apply_scenario_calibration_profile(
    workload: &mut WorkloadConfig,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    let Some(path) = scenario.calibration_profile_path.as_ref() else {
        return Ok(());
    };
    let profile = load_calibration_profile_path(path).map_err(|err| {
        CliError::Usage(format!(
            "scenario '{}' calibration_profile {} failed: {err}",
            scenario.name,
            path.display()
        ))
    })?;
    workload.calibration_profile = Some(profile.metadata);
    workload.calibration = profile.calibration;
    apply_scenario_calibration(&mut workload.calibration, workload.calibration_overrides);
    Ok(())
}

fn apply_scenario_calibration(
    calibration: &mut SimulationCalibration,
    override_config: RunScenarioCalibrationConfig,
) {
    if let Some(value) = override_config.compute_efficiency {
        calibration.compute_efficiency = value;
    }
    if let Some(value) = override_config.prefill_compute_scale {
        calibration.prefill_compute_scale = value;
    }
    if let Some(value) = override_config.decode_compute_scale {
        calibration.decode_compute_scale = value;
    }
    if let Some(value) = override_config.decode_memory_bandwidth_scale {
        calibration.decode_memory_bandwidth_scale = value;
    }
    if let Some(value) = override_config.collective_latency_scale {
        calibration.collective_latency_scale = value;
    }
    if let Some(value) = override_config.collective_bandwidth_scale {
        calibration.collective_bandwidth_scale = value;
    }
    if let Some(value) = override_config.kv_transfer_scale {
        calibration.kv_transfer_scale = value;
    }
    if let Some(value) = override_config.scheduler_overhead_us {
        calibration.scheduler_overhead_us = value;
    }
    if let Some(value) = override_config.serving_memory_temporary_fraction {
        calibration.serving_memory_temporary_fraction = value;
    }
    if let Some(value) = override_config.serving_memory_activation_communication_fraction {
        calibration.serving_memory_activation_communication_fraction = value;
    }
    if let Some(value) = override_config.serving_memory_weight_communication_fraction {
        calibration.serving_memory_weight_communication_fraction = value;
    }
    if let Some(value) = override_config.serving_memory_runtime_reserve_fraction {
        calibration.serving_memory_runtime_reserve_fraction = value;
    }
    if let Some(value) = override_config.serving_memory_fragmentation_fraction {
        calibration.serving_memory_fragmentation_fraction = value;
    }
    if let Some(value) = override_config.serving_pipeline_depth {
        calibration.serving_pipeline_depth = value;
    }
    if let Some(value) = override_config.request_arrival_gap_s {
        calibration.request_arrival_gap_s = value;
    }
    if let Some(value) = override_config.allow_compute_comm_overlap {
        calibration.allow_compute_comm_overlap = value;
    }
    *calibration = calibration.sanitized();
}

fn apply_scenario_request_count(
    traffic: &mut ServingTraffic,
    request_count: u32,
    scenario: &RunScenarioConfig,
) -> Result<(), CliError> {
    if !traffic.trace_requests.is_empty() {
        let trace_len = traffic.trace_requests.len();
        let requested = request_count as usize;
        if requested > trace_len {
            return Err(CliError::Usage(format!(
                "scenario '{}' request_count {request_count} exceeds trace length {trace_len}; trace scenarios can truncate but not synthesize extra trace rows",
                scenario.name
            )));
        }
        traffic.trace_requests.truncate(requested);
    }
    traffic.request_count = Some(request_count);
    Ok(())
}

fn apply_arrival_scales(
    traffic: &mut ServingTraffic,
    calibration: SimulationCalibration,
    scenario: &RunScenarioConfig,
) {
    if scenario.arrival_gap_scale.is_none() && scenario.arrival_rate_scale.is_none() {
        return;
    }
    let gap_multiplier =
        scenario.arrival_gap_scale.unwrap_or(1.0) / scenario.arrival_rate_scale.unwrap_or(1.0);
    if !traffic.trace_requests.is_empty() {
        scale_trace_arrivals(traffic, gap_multiplier);
        return;
    }
    match &mut traffic.arrival {
        ServingArrivalPattern::FixedGap => {
            let gap_s = traffic
                .arrival_gap_s
                .unwrap_or(calibration.request_arrival_gap_s);
            traffic.arrival_gap_s = Some(gap_s * gap_multiplier);
        }
        ServingArrivalPattern::Poisson { rate_per_s, .. } => {
            *rate_per_s /= gap_multiplier;
        }
        ServingArrivalPattern::Diurnal {
            min_rate_per_s,
            max_rate_per_s,
            ..
        } => {
            *min_rate_per_s /= gap_multiplier;
            *max_rate_per_s /= gap_multiplier;
        }
        ServingArrivalPattern::SelfSimilar {
            rate_per_s,
            max_gap_s,
            ..
        } => {
            *rate_per_s /= gap_multiplier;
            if let Some(max_gap_s) = max_gap_s {
                *max_gap_s *= gap_multiplier;
            }
        }
        ServingArrivalPattern::TraceDerived => {}
        ServingArrivalPattern::Bursty {
            burst_interval_s,
            intra_burst_gap_s,
            ..
        } => {
            *burst_interval_s *= gap_multiplier;
            *intra_burst_gap_s *= gap_multiplier;
        }
    }
}

fn scale_trace_arrivals(traffic: &mut ServingTraffic, gap_multiplier: f64) {
    for request in &mut traffic.trace_requests {
        let old_arrival_s = request.arrival_s;
        let new_arrival_s = old_arrival_s * gap_multiplier;
        if let Some(deadline_s) = request.deadline_s {
            request.deadline_s = Some(new_arrival_s + (deadline_s - old_arrival_s));
        }
        if let Some(cancellation_s) = request.cancellation_s {
            request.cancellation_s = Some(new_arrival_s + (cancellation_s - old_arrival_s));
        }
        request.arrival_s = new_arrival_s;
    }
    if let Some(start_s) = traffic.measurement_start_s.as_mut() {
        *start_s *= gap_multiplier;
    }
    if let Some(end_s) = traffic.measurement_end_s.as_mut() {
        *end_s *= gap_multiplier;
    }
    if let Some(warmup_s) = traffic.measurement_warmup_s.as_mut() {
        *warmup_s *= gap_multiplier;
    }
    if let Some(cooldown_s) = traffic.measurement_cooldown_s.as_mut() {
        *cooldown_s *= gap_multiplier;
    }
}

fn apply_request_shape_scales(request: &mut InferenceRequest, scenario: &RunScenarioConfig) {
    if let Some(scale) = scenario.batch_size_scale {
        request.batch_size = scaled_u32(request.batch_size, scale);
    }
    let sequence_scale = scenario
        .prompt_tokens_scale
        .into_iter()
        .chain(scenario.decode_tokens_scale)
        .fold(1.0_f64, f64::max);
    if let Some(scale) = scenario.prompt_tokens_scale {
        request.prompt_tokens = scaled_u32(request.prompt_tokens, scale);
    }
    if let Some(scale) = scenario.decode_tokens_scale {
        request.decode_tokens = scaled_u32(request.decode_tokens, scale);
    }
    if sequence_scale != 1.0 {
        request.max_sequence_tokens = scaled_u32(request.max_sequence_tokens, sequence_scale);
    }
    request.max_sequence_tokens = request
        .max_sequence_tokens
        .max(request.prompt_tokens.saturating_add(request.decode_tokens));
}

fn apply_serving_shape_scales(traffic: &mut ServingTraffic, scenario: &RunScenarioConfig) {
    if let Some(scale) = scenario.batch_size_scale {
        scale_values(&mut traffic.batch_sizes, scale);
        scale_distribution(&mut traffic.batch_size_distribution, scale);
        for profile in &mut traffic.shape_profiles {
            profile.batch_size = scaled_u32(profile.batch_size, scale);
        }
        for request in &mut traffic.trace_requests {
            request.batch_size = scaled_u32(request.batch_size, scale);
        }
    }
    if let Some(scale) = scenario.prompt_tokens_scale {
        scale_values(&mut traffic.prompt_tokens, scale);
        scale_distribution(&mut traffic.prompt_tokens_distribution, scale);
        for profile in &mut traffic.shape_profiles {
            profile.prompt_tokens = scaled_u32(profile.prompt_tokens, scale);
        }
        for request in &mut traffic.trace_requests {
            request.prompt_tokens = scaled_u32(request.prompt_tokens, scale);
        }
    }
    if let Some(scale) = scenario.decode_tokens_scale {
        scale_values(&mut traffic.decode_tokens, scale);
        scale_distribution(&mut traffic.decode_tokens_distribution, scale);
        for profile in &mut traffic.shape_profiles {
            profile.decode_tokens = scaled_u32(profile.decode_tokens, scale);
        }
        for request in &mut traffic.trace_requests {
            request.decode_tokens = scaled_u32(request.decode_tokens, scale);
        }
    }
    let sequence_scale = scenario
        .prompt_tokens_scale
        .into_iter()
        .chain(scenario.decode_tokens_scale)
        .fold(1.0_f64, f64::max);
    for request in &mut traffic.trace_requests {
        if let Some(max_sequence_tokens) = request.max_sequence_tokens.as_mut() {
            if sequence_scale != 1.0 {
                *max_sequence_tokens = scaled_u32(*max_sequence_tokens, sequence_scale);
            }
            *max_sequence_tokens = (*max_sequence_tokens)
                .max(request.prompt_tokens.saturating_add(request.decode_tokens));
        }
    }
    for profile in &mut traffic.shape_profiles {
        if let Some(max_sequence_tokens) = profile.max_sequence_tokens.as_mut() {
            if sequence_scale != 1.0 {
                *max_sequence_tokens = scaled_u32(*max_sequence_tokens, sequence_scale);
            }
            *max_sequence_tokens = (*max_sequence_tokens)
                .max(profile.prompt_tokens.saturating_add(profile.decode_tokens));
        }
    }
}

fn scale_values(values: &mut [u32], scale: f64) {
    for value in values {
        *value = scaled_u32(*value, scale);
    }
}

fn scale_distribution(distribution: &mut Option<ServingValueDistribution>, scale: f64) {
    match distribution {
        Some(ServingValueDistribution::Uniform { min, max }) => {
            *min = scaled_u32(*min, scale);
            *max = scaled_u32(*max, scale);
            if *min > *max {
                std::mem::swap(min, max);
            }
        }
        Some(ServingValueDistribution::Weighted { values, .. }) => {
            scale_values(values, scale);
        }
        Some(ServingValueDistribution::LogNormal {
            median, min, max, ..
        }) => {
            *median = (*median * scale).max(1.0);
            *min = scaled_u32(*min, scale);
            *max = scaled_u32(*max, scale);
            if *min > *max {
                std::mem::swap(min, max);
            }
        }
        None => {}
    }
}

fn scaled_u32(value: u32, scale: f64) -> u32 {
    let scaled = (f64::from(value) * scale).round();
    if scaled < 1.0 {
        1
    } else if scaled > f64::from(u32::MAX) {
        u32::MAX
    } else {
        scaled as u32
    }
}
