use super::*;

pub(super) fn write_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredParallelismConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    calibration_gate_violations: &[CalibrationGateViolation],
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "cluster_gpus={} model_params_gb={:.2} model_parameter_count_billion={:.3} searched_configs={} reported_configs={} omitted_rejected_configs={}",
        cluster.total_gpus(),
        model.parameters.as_gigabytes(),
        model.parameter_count_billion(),
        search_diagnostics.searched_candidate_count,
        results.len(),
        search_diagnostics.omitted_rejected_candidate_count
    )?;
    write_cluster_inventory_text(writer, cluster)?;
    write_search_diagnostics_text(writer, search_diagnostics)?;
    write_calibration_gate_text(writer, calibration_gate_violations)?;
    write_trust_boundary_text(writer)?;
    writeln!(
        writer,
        "{:<5} {:<8} {:>12} {:>12} {:>7} {:>4} {:>4} {:>4} {:>4}  notes",
        "rank", "status", "latency_ms", "memory_gb", "ranks", "tp", "pp", "ep", "dp"
    )?;

    for (idx, score) in results.iter().take(top_k).enumerate() {
        let status = if score.feasible { "ok" } else { "reject" };
        let latency_ms = if score.feasible {
            format!("{:.3}", score.estimated_latency_s * 1000.0)
        } else {
            "-".to_string()
        };
        let notes = notes_with_approximations(
            score.rejected_reason.as_deref(),
            &score.bottlenecks,
            &score.approximations,
        );

        writeln!(
            writer,
            "{:<5} {:<8} {:>12} {:>12.2} {:>7} {:>4} {:>4} {:>4} {:>4}  {}",
            idx + 1,
            status,
            latency_ms,
            score.estimated_memory_per_gpu.as_gigabytes(),
            score.config.total_ranks(),
            score.config.tensor_ranks,
            score.config.pipeline_ranks,
            score.config.expert_ranks,
            score.config.data_ranks,
            notes
        )?;
    }

    Ok(())
}

pub(super) fn write_markdown_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredParallelismConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    calibration_gate_violations: &[CalibrationGateViolation],
) -> Result<(), io::Error> {
    writeln!(writer, "# Inference Sim Summary\n")?;
    write_markdown_key_value_section(
        writer,
        "Overview",
        &[
            ("Mode", "parallelism".to_string()),
            ("Cluster GPUs", cluster.total_gpus().to_string()),
            ("Available GPUs", cluster.available_gpus().to_string()),
            (
                "Model Parameters GB",
                format!("{:.2}", model.parameters.as_gigabytes()),
            ),
            (
                "Model Parameter Count B",
                format!("{:.3}", model.parameter_count_billion()),
            ),
            (
                "Searched Candidates",
                search_diagnostics.searched_candidate_count.to_string(),
            ),
            ("Reported Candidates", results.len().to_string()),
            (
                "Omitted Rejected Candidates",
                search_diagnostics
                    .omitted_rejected_candidate_count
                    .to_string(),
            ),
        ],
    )?;
    write_markdown_search_diagnostics(writer, search_diagnostics)?;
    write_markdown_cluster_inventory(writer, cluster)?;
    write_markdown_calibration_gates(writer, calibration_gate_violations)?;

    writeln!(writer, "## Candidates\n")?;
    write_markdown_row(
        writer,
        &[
            "Rank",
            "Status",
            "Latency ms",
            "Memory/GPU GB",
            "Ranks",
            "TP",
            "PP",
            "EP",
            "DP",
            "Notes",
        ],
    )?;
    write_markdown_separator(writer, 10)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[
                (idx + 1).to_string(),
                status(score.feasible).to_string(),
                if score.feasible {
                    format!("{:.3}", score.estimated_latency_s * 1000.0)
                } else {
                    "-".to_string()
                },
                format!("{:.2}", score.estimated_memory_per_gpu.as_gigabytes()),
                score.config.total_ranks().to_string(),
                score.config.tensor_ranks.to_string(),
                score.config.pipeline_ranks.to_string(),
                score.config.expert_ranks.to_string(),
                score.config.data_ranks.to_string(),
                parallelism_candidate_notes(score),
            ],
        )?;
    }
    writeln!(writer)?;
    write_markdown_trust_boundary(writer)
}

pub(super) fn write_serving_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredServingConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    options: ServingTextOutputOptions<'_>,
) -> Result<(), io::Error> {
    let serving_stack_note = options
        .serving_stack
        .map(|stack| format!(" serving_stack={stack}"))
        .unwrap_or_default();
    let runtime_features_note = if options.serving_runtime_features.is_empty() {
        String::new()
    } else {
        format!(
            " runtime_features={}",
            options.serving_runtime_features.join(",")
        )
    };
    writeln!(
        writer,
        "cluster_gpus={} model_params_gb={:.2} model_parameter_count_billion={:.3} searched_serving_pairs={} reported_serving_pairs={} omitted_rejected_serving_pairs={} objective={}{}{}",
        cluster.total_gpus(),
        model.parameters.as_gigabytes(),
        model.parameter_count_billion(),
        search_diagnostics.searched_candidate_count,
        results.len(),
        search_diagnostics.omitted_rejected_candidate_count,
        serving_objective_label(results),
        serving_stack_note,
        runtime_features_note
    )?;
    write_cluster_inventory_text(writer, cluster)?;
    write_search_diagnostics_text(writer, search_diagnostics)?;
    write_calibration_gate_text(writer, options.calibration_gate_violations)?;
    write_serving_rejection_summary_text(writer, results)?;
    write_trust_boundary_text(writer)?;
    writeln!(
        writer,
        "{:<5} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5} {:>5} {:>11} {:>8} {:>11} {:>13} {:>12} {:>12} {:>12} {:>14} {:>8} {:>8}  notes",
        "rank",
        "status",
        "ttft_ms",
        "ttft_p95",
        "tpot_ms",
        "tpot_p95",
        "itl_ms",
        "itl_p95",
        "ttft_miss",
        "tpot_miss",
        "itl_miss",
        "e2el_miss",
        "throughput",
        "e2el_ms",
        "e2el_p95",
        "queue_ms",
        "kv_ms",
        "sched_ms",
        "reqs",
        "meas",
        "pf_tok_peak",
        "seq_peak",
        "kv_tok_peak",
        "seq_node_peak",
        "kv_node_peak",
        "seq_gpu_peak",
        "kv_gpu_peak",
        "pool",
        "prefill",
        "decode"
    )?;

    for (idx, score) in results.iter().take(top_k).enumerate() {
        let status = if score.feasible { "ok" } else { "reject" };
        let mut notes = notes_with_approximations(
            score.rejected_reason.as_deref(),
            &score.bottlenecks,
            &score.approximations,
        );
        if let Some(approximation_note) =
            serving_approximation_summary_note(&score.approximation_summary)
        {
            append_note(&mut notes, &approximation_note);
        }
        if score.slo_miss_penalty_weights.any_nonzero()
            || !score.traffic_class_slo_miss_penalties.is_empty()
        {
            let penalty_note = format!("slo_penalty={:.4}", score.slo_miss_penalty_score);
            if notes == "none" {
                notes = penalty_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&penalty_note);
            }
        }
        if score.route_coverage.candidate_count > 0
            && score.route_coverage.routable_candidate_count < score.route_coverage.candidate_count
        {
            let route_note = format!(
                "route_coverage={}/{}",
                score.route_coverage.routable_candidate_count, score.route_coverage.candidate_count
            );
            if notes == "none" {
                notes = route_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&route_note);
            }
        }
        if let Some(pool_search_note) = pool_search_summary_note(score.pool_search_summary.as_ref())
        {
            append_note(&mut notes, &pool_search_note);
        }
        if let Some(pool_topology_note) = pool_topology_summary_note(&score.pool_topology) {
            append_note(&mut notes, &pool_topology_note);
        }
        if score.topology_risk_penalty_score > 0.0 {
            let penalty_note = format!("topology_penalty={:.4}", score.topology_risk_penalty_score);
            if notes == "none" {
                notes = penalty_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&penalty_note);
            }
        }
        if score.service_backpressure_penalty_score > 0.0 {
            let penalty_note = format!(
                "backpressure_penalty={:.4}",
                score.service_backpressure_penalty_score
            );
            if notes == "none" {
                notes = penalty_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&penalty_note);
            }
        }
        if let Some(route_note) = kv_route_summary_note(score) {
            if notes == "none" {
                notes = route_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&route_note);
            }
        }
        if let Some(calibration_note) = serving_calibration_summary_note(score) {
            if notes == "none" {
                notes = calibration_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&calibration_note);
            }
        }
        let mode_note = format!("mode={}", score.deployment_mode.as_str());
        if notes == "none" {
            notes = mode_note;
        } else {
            notes.push_str("; ");
            notes.push_str(&mode_note);
        }
        if let Some(footprint_note) = serving_footprint_summary_note(score) {
            if notes == "none" {
                notes = footprint_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&footprint_note);
            }
        }
        if let Some(cost_note) = serving_cost_summary_note(score) {
            if notes == "none" {
                notes = cost_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&cost_note);
            }
        }
        if let Some(pareto_note) = serving_pareto_note(score) {
            append_note(&mut notes, &pareto_note);
        }
        if let Some(objective_note) = serving_objective_summary_note(score) {
            append_note(&mut notes, &objective_note);
        }
        if let Some(bottleneck_note) = serving_bottleneck_summary_note(score) {
            append_note(&mut notes, &bottleneck_note);
        }
        let prefill = format!(
            "{}/{}/{}/{}",
            score.prefill_config.tensor_ranks,
            score.prefill_config.pipeline_ranks,
            score.prefill_config.expert_ranks,
            score.prefill_config.data_ranks
        );
        let decode = format!(
            "{}/{}/{}/{}",
            score.decode_config.tensor_ranks,
            score.decode_config.pipeline_ranks,
            score.decode_config.expert_ranks,
            score.decode_config.data_ranks
        );
        let pool = pool_label(score);

        writeln!(
            writer,
            "{:<5} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>12.2} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5} {:>5} {:>11} {:>8} {:>11} {:>13} {:>12} {:>12} {:>12} {:>14} {:>8} {:>8}  {}",
            idx + 1,
            status,
            metric_ms(score.feasible, score.metrics.ttft_s),
            metric_ms(score.feasible, score.metrics.ttft_p95_s),
            metric_ms(score.feasible, score.metrics.tpot_s),
            metric_ms(score.feasible, score.metrics.tpot_p95_s),
            metric_ms(score.feasible, score.metrics.itl_s),
            metric_ms(score.feasible, score.metrics.itl_p95_s),
            metric_percent(score.feasible, score.metrics.ttft_slo_miss_rate),
            metric_percent(score.feasible, score.metrics.tpot_slo_miss_rate),
            metric_percent(score.feasible, score.metrics.itl_slo_miss_rate),
            metric_percent(score.feasible, score.metrics.e2el_slo_miss_rate),
            score.metrics.throughput_tokens_per_s,
            metric_ms(score.feasible, score.metrics.e2el_s),
            metric_ms(score.feasible, score.metrics.e2el_p95_s),
            metric_ms(score.feasible, score.metrics.queue_delay_s),
            metric_ms(score.feasible, score.metrics.kv_transfer_s),
            metric_ms(score.feasible, score.metrics.scheduled_makespan_s),
            score.metrics.scheduled_requests,
            score.metrics.measured_requests,
            score.metrics.peak_prefill_tokens,
            score.metrics.peak_decode_sequences,
            score.metrics.peak_resident_tokens,
            score.metrics.peak_decode_sequences_per_node,
            score.metrics.peak_resident_tokens_per_node,
            score.metrics.peak_decode_sequences_per_gpu,
            score.metrics.peak_resident_tokens_per_gpu,
            pool,
            prefill,
            decode,
            notes
        )?;
    }

    Ok(())
}

pub(super) fn write_serving_markdown_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredServingConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    options: ServingTextOutputOptions<'_>,
) -> Result<(), io::Error> {
    let runtime_features = if options.serving_runtime_features.is_empty() {
        "none".to_string()
    } else {
        options.serving_runtime_features.join(", ")
    };
    write!(writer, "# Inference Sim Summary\n\n")?;
    write_markdown_key_value_section(
        writer,
        "Overview",
        &[
            ("Mode", "serving".to_string()),
            ("Objective", serving_objective_label(results).to_string()),
            (
                "Serving Stack",
                options.serving_stack.unwrap_or("none").to_string(),
            ),
            ("Runtime Features", runtime_features),
            ("Cluster GPUs", cluster.total_gpus().to_string()),
            ("Available GPUs", cluster.available_gpus().to_string()),
            (
                "Model Parameters GB",
                format!("{:.2}", model.parameters.as_gigabytes()),
            ),
            (
                "Model Parameter Count B",
                format!("{:.3}", model.parameter_count_billion()),
            ),
            (
                "Searched Serving Pairs",
                search_diagnostics.searched_candidate_count.to_string(),
            ),
            ("Reported Serving Pairs", results.len().to_string()),
            (
                "Omitted Rejected Serving Pairs",
                search_diagnostics
                    .omitted_rejected_candidate_count
                    .to_string(),
            ),
        ],
    )?;
    write_markdown_search_diagnostics(writer, search_diagnostics)?;
    write_markdown_cluster_inventory(writer, cluster)?;
    write_markdown_calibration_gates(writer, options.calibration_gate_violations)?;
    write_markdown_serving_rejection_summary(writer, results)?;

    writeln!(writer, "## Serving Candidates\n")?;
    write_markdown_row(
        writer,
        &[
            "Rank",
            "Status",
            "TTFT ms",
            "TPOT ms",
            "ITL ms",
            "E2EL ms",
            "Throughput tok/s",
            "Queue ms",
            "KV ms",
            "Reqs",
            "Measured",
            "Pool",
            "Mode",
            "Prefill TP/PP/EP/DP",
            "Decode TP/PP/EP/DP",
        ],
    )?;
    write_markdown_separator(writer, 15)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[
                (idx + 1).to_string(),
                status(score.feasible).to_string(),
                metric_ms(score.feasible, score.metrics.ttft_s),
                metric_ms(score.feasible, score.metrics.tpot_s),
                metric_ms(score.feasible, score.metrics.itl_s),
                metric_ms(score.feasible, score.metrics.e2el_s),
                if score.feasible && score.metrics.throughput_tokens_per_s.is_finite() {
                    format!("{:.2}", score.metrics.throughput_tokens_per_s)
                } else {
                    "-".to_string()
                },
                metric_ms(score.feasible, score.metrics.queue_delay_s),
                metric_ms(score.feasible, score.metrics.kv_transfer_s),
                score.metrics.scheduled_requests.to_string(),
                score.metrics.measured_requests.to_string(),
                pool_label(score),
                score.deployment_mode.as_str().to_string(),
                parallelism_config_summary(&score.prefill_config),
                parallelism_config_summary(&score.decode_config),
            ],
        )?;
    }
    writeln!(writer)?;

    writeln!(writer, "## SLO And Capacity\n")?;
    write_markdown_row(
        writer,
        &[
            "Rank",
            "TTFT Miss",
            "TPOT Miss",
            "ITL Miss",
            "E2EL Miss",
            "Peak Prefill Tokens",
            "Peak Decode Sequences",
            "Peak Resident Tokens",
            "Peak KV Blocks",
        ],
    )?;
    write_markdown_separator(writer, 9)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[
                (idx + 1).to_string(),
                metric_percent(score.feasible, score.metrics.ttft_slo_miss_rate),
                metric_percent(score.feasible, score.metrics.tpot_slo_miss_rate),
                metric_percent(score.feasible, score.metrics.itl_slo_miss_rate),
                metric_percent(score.feasible, score.metrics.e2el_slo_miss_rate),
                score.metrics.peak_prefill_tokens.to_string(),
                score.metrics.peak_decode_sequences.to_string(),
                score.metrics.peak_resident_tokens.to_string(),
                score.metrics.peak_kv_blocks.to_string(),
            ],
        )?;
    }
    writeln!(writer)?;

    writeln!(writer, "## Candidate Notes\n")?;
    write_markdown_row(writer, &["Rank", "Notes"])?;
    write_markdown_separator(writer, 2)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[(idx + 1).to_string(), serving_candidate_notes(score)],
        )?;
    }
    writeln!(writer)?;
    write_markdown_trust_boundary(writer)
}

fn parallelism_config_summary(config: &ParallelismConfig) -> String {
    format!(
        "{}/{}/{}/{}",
        config.tensor_ranks, config.pipeline_ranks, config.expert_ranks, config.data_ranks
    )
}

fn parallelism_candidate_notes(score: &ScoredParallelismConfig) -> String {
    notes_with_approximations(
        score.rejected_reason.as_deref(),
        &score.bottlenecks,
        &score.approximations,
    )
}

fn serving_candidate_notes(score: &ScoredServingConfig) -> String {
    let mut notes = notes_with_approximations(
        score.rejected_reason.as_deref(),
        &score.bottlenecks,
        &score.approximations,
    );
    if let Some(approximation_note) =
        serving_approximation_summary_note(&score.approximation_summary)
    {
        append_note(&mut notes, &approximation_note);
    }
    if score.slo_miss_penalty_weights.any_nonzero()
        || !score.traffic_class_slo_miss_penalties.is_empty()
    {
        append_note(
            &mut notes,
            &format!("slo_penalty={:.4}", score.slo_miss_penalty_score),
        );
    }
    if score.route_coverage.candidate_count > 0
        && score.route_coverage.routable_candidate_count < score.route_coverage.candidate_count
    {
        append_note(
            &mut notes,
            &format!(
                "route_coverage={}/{}",
                score.route_coverage.routable_candidate_count, score.route_coverage.candidate_count
            ),
        );
    }
    if let Some(pool_search_note) = pool_search_summary_note(score.pool_search_summary.as_ref()) {
        append_note(&mut notes, &pool_search_note);
    }
    if let Some(pool_topology_note) = pool_topology_summary_note(&score.pool_topology) {
        append_note(&mut notes, &pool_topology_note);
    }
    if score.topology_risk_penalty_score > 0.0 {
        append_note(
            &mut notes,
            &format!("topology_penalty={:.4}", score.topology_risk_penalty_score),
        );
    }
    if score.service_backpressure_penalty_score > 0.0 {
        append_note(
            &mut notes,
            &format!(
                "backpressure_penalty={:.4}",
                score.service_backpressure_penalty_score
            ),
        );
    }
    if let Some(route_note) = kv_route_summary_note(score) {
        append_note(&mut notes, &route_note);
    }
    if let Some(calibration_note) = serving_calibration_summary_note(score) {
        append_note(&mut notes, &calibration_note);
    }
    append_note(
        &mut notes,
        &format!("mode={}", score.deployment_mode.as_str()),
    );
    if let Some(footprint_note) = serving_footprint_summary_note(score) {
        append_note(&mut notes, &footprint_note);
    }
    if let Some(cost_note) = serving_cost_summary_note(score) {
        append_note(&mut notes, &cost_note);
    }
    if let Some(pareto_note) = serving_pareto_note(score) {
        append_note(&mut notes, &pareto_note);
    }
    if let Some(objective_note) = serving_objective_summary_note(score) {
        append_note(&mut notes, &objective_note);
    }
    if let Some(bottleneck_note) = serving_bottleneck_summary_note(score) {
        append_note(&mut notes, &bottleneck_note);
    }
    notes
}

pub(super) fn write_markdown_key_value_section<W: Write>(
    writer: &mut W,
    title: &str,
    rows: &[(&str, String)],
) -> Result<(), io::Error> {
    writeln!(writer, "## {title}\n")?;
    write_markdown_row(writer, &["Metric", "Value"])?;
    write_markdown_separator(writer, 2)?;
    for (label, value) in rows {
        write_markdown_row(writer, &[label.to_string(), value.clone()])?;
    }
    writeln!(writer)
}

fn write_markdown_calibration_gates<W: Write>(
    writer: &mut W,
    violations: &[CalibrationGateViolation],
) -> Result<(), io::Error> {
    if violations.is_empty() {
        return Ok(());
    }
    writeln!(writer, "## Calibration Gate Violations\n")?;
    write_markdown_row(writer, &["Action", "Code", "Observed", "Limit", "Message"])?;
    write_markdown_separator(writer, 5)?;
    for violation in violations {
        write_markdown_row(
            writer,
            &[
                violation.action.as_str().to_string(),
                violation.code.clone(),
                json_optional_value(violation.observed),
                json_optional_value(violation.limit),
                violation.message.clone(),
            ],
        )?;
    }
    writeln!(writer)
}

fn write_markdown_trust_boundary<W: Write>(writer: &mut W) -> Result<(), io::Error> {
    writeln!(writer, "## Trust Boundary\n")?;
    writeln!(
        writer,
        "This is a v1 planning estimate. Compare candidates with calibration, approximation, rejection, and bottleneck artifacts before treating absolute latency or throughput as credible."
    )
}

pub(super) fn write_markdown_row<W, S>(writer: &mut W, cells: &[S]) -> Result<(), io::Error>
where
    W: Write,
    S: AsRef<str>,
{
    write!(writer, "|")?;
    for cell in cells {
        write!(writer, " {} |", markdown_cell(cell.as_ref()))?;
    }
    writeln!(writer)
}

pub(super) fn write_markdown_separator<W: Write>(
    writer: &mut W,
    column_count: usize,
) -> Result<(), io::Error> {
    write!(writer, "|")?;
    for _ in 0..column_count {
        write!(writer, " --- |")?;
    }
    writeln!(writer)
}

pub(super) fn markdown_cell(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "-".to_string();
    }
    trimmed
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "<br>")
        .replace('\r', "")
}

pub(super) fn demote_markdown_summary_headings(markdown: &str) -> String {
    let mut output = String::new();
    for line in markdown.lines() {
        if line.trim() == "# Inference Sim Summary" {
            continue;
        }
        if line.starts_with('#') {
            output.push('#');
        }
        output.push_str(line);
        output.push('\n');
    }
    output
}

fn serving_footprint_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let footprint = &score.hardware_footprint;
    (footprint.unique_gpu_count > 0).then(|| {
        format!(
            "footprint_gpus={}; footprint_nodes={}; throughput_per_gpu={:.2}",
            footprint.unique_gpu_count,
            footprint.unique_node_count,
            footprint.throughput_tokens_per_s_per_gpu
        )
    })
}

fn serving_cost_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let estimate = &score.cost_estimate;
    let total_cost = estimate.total_cost_usd?;
    Some(match estimate.cost_per_1k_output_tokens_usd {
        Some(per_1k) => format!("cost=${total_cost:.6}; cost_per_1k_out=${per_1k:.6}"),
        None => format!("cost=${total_cost:.6}"),
    })
}

fn serving_calibration_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let summary = &score.calibration_summary;
    (!summary.status.is_empty()).then(|| {
        format!(
            "calibration={}:coverage={}/{}",
            summary.status, summary.calibrated_phase_count, summary.active_phase_count
        )
    })
}

fn serving_pareto_note(score: &ScoredServingConfig) -> Option<String> {
    let rank = score.pareto.rank?;
    if score.pareto.is_frontier {
        Some("pareto=frontier".to_string())
    } else {
        Some(format!("pareto_rank={rank}"))
    }
}

fn serving_bottleneck_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let bottleneck = score
        .bottleneck_summary
        .iter()
        .find(|bottleneck| bottleneck.severity != "info")
        .or_else(|| score.bottleneck_summary.first())?;
    Some(format!(
        "top_bottleneck={}:{}:{}",
        bottleneck.severity, bottleneck.code, bottleneck.resource
    ))
}

fn notes_with_approximations(
    rejected_reason: Option<&str>,
    bottlenecks: &[String],
    approximations: &[SimulationApproximation],
) -> String {
    if let Some(reason) = rejected_reason {
        return reason.to_string();
    }
    let mut notes = if bottlenecks.is_empty() {
        "none".to_string()
    } else {
        bottlenecks.join("; ")
    };
    if !approximations.is_empty() {
        let mut codes: Vec<_> = approximations
            .iter()
            .take(3)
            .map(|approximation| approximation.code.as_str())
            .collect();
        let more = approximations.len().saturating_sub(codes.len());
        if more > 0 {
            codes.push("...");
        }
        notes.push_str("; approximations: ");
        notes.push_str(&codes.join(","));
        if more > 0 {
            notes.push_str(&format!(" (+{more})"));
        }
    }
    notes
}

fn append_note(notes: &mut String, note: &str) {
    if note.is_empty() {
        return;
    }
    if notes == "none" {
        *notes = note.to_string();
    } else {
        notes.push_str("; ");
        notes.push_str(note);
    }
}

struct TrustBoundaryNote {
    code: &'static str,
    scope: &'static str,
    label: &'static str,
    implication: &'static str,
}

const TRUST_BOUNDARY_NOTES: &[TrustBoundaryNote] = &[
    TrustBoundaryNote {
        code: "coarse_topology",
        scope: "topology",
        label: "Route resources are modeled at node, GPU, NIC, rail, and link scope, not full PCIe/NUMA/switch microarchitecture.",
        implication: "Use route evidence for relative placement and rail-risk analysis; calibrate before treating absolute locality costs as production latencies.",
    },
    TrustBoundaryNote {
        code: "approximate_queueing",
        scope: "serving_scheduler",
        label: "Queueing uses scheduled timeline and worker-readiness approximations rather than a production runtime event loop.",
        implication: "TTFT, TPOT, throughput, and E2EL are planner estimates whose accuracy depends on traffic shape and calibration coverage.",
    },
    TrustBoundaryNote {
        code: "aggregate_memory",
        scope: "memory",
        label: "HBM pressure is estimated from aggregate weights, KV cache, block tables, activations, communication, reserves, and fragmentation terms.",
        implication: "Capacity comparisons are useful, but allocator behavior, kernel workspace spikes, and exact fragmentation require measured validation.",
    },
    TrustBoundaryNote {
        code: "calibration_dependent",
        scope: "calibration",
        label: "Absolute latency and throughput are only as trustworthy as the loaded calibration profile and its coverage.",
        implication: "Inspect calibration summaries, residuals, extrapolation warnings, and gate violations before relying on rankings.",
    },
    TrustBoundaryNote {
        code: "ignored_network_congestion",
        scope: "network",
        label: "NIC/link bandwidth, latency, route sharing, and rail locality are modeled, but packet-level congestion effects are not.",
        implication: "Incast, outcast, PFC, head-of-line blocking, and switch-buffer behavior need external calibration or a future congestion model.",
    },
    TrustBoundaryNote {
        code: "unsupported_locality_detail",
        scope: "locality",
        label: "PCIe, NUMA, copy-engine, GPUDirect, and backend-specific locality details are represented through configured resources and route records.",
        implication: "Detailed host/device transfer paths should be encoded as topology overrides and checked against route-path artifacts.",
    },
    TrustBoundaryNote {
        code: "serving_stack_approximation",
        scope: "runtime",
        label: "Runtime behavior is captured through serving stack/features, coarse service models, and calibration evidence, not backend-specific kernel emulation.",
        implication: "Use approximation policy gates for uncalibrated stacks or features that may dominate latency.",
    },
];

fn write_trust_boundary_text<W: Write>(writer: &mut W) -> Result<(), io::Error> {
    writeln!(
        writer,
        "trust_boundary=v1_approximate assumptions={}",
        TRUST_BOUNDARY_NOTES
            .iter()
            .map(|note| note.code)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(super) fn write_trust_boundary_json<W: Write>(
    writer: &mut W,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"trust_boundary\": {{")?;
    writeln!(writer, "{indent}  \"status\": \"v1_approximate\",")?;
    writeln!(
        writer,
        "{indent}  \"summary\": {},",
        json_string(
            "Useful for relative planning and calibration workflows; not a proof of production latency or fine-grained locality behavior."
        )
    )?;
    writeln!(writer, "{indent}  \"assumptions\": [")?;
    for (idx, note) in TRUST_BOUNDARY_NOTES.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(note.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(note.scope)
        )?;
        writeln!(
            writer,
            "{indent}      \"label\": {},",
            json_string(note.label)
        )?;
        writeln!(
            writer,
            "{indent}      \"implication\": {}",
            json_string(note.implication)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < TRUST_BOUNDARY_NOTES.len())
        )?;
    }
    writeln!(writer, "{indent}  ]")?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}
