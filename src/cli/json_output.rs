use super::*;

pub(super) fn write_json_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredParallelismConfig],
    options: JsonOutputOptions<'_>,
) -> Result<(), io::Error> {
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema_version\": 1,")?;
    writeln!(writer, "  \"mode\": \"parallelism\",")?;
    writeln!(writer, "  \"cluster_gpus\": {},", cluster.total_gpus())?;
    write_cluster_inventory(writer, cluster, "  ", true)?;
    writeln!(
        writer,
        "  \"model_params_gb\": {},",
        json_f64(model.parameters.as_gigabytes())
    )?;
    writeln!(
        writer,
        "  \"model_parameter_count_billion\": {},",
        json_f64(model.parameter_count_billion())
    )?;
    writeln!(
        writer,
        "  \"searched_configs\": {},",
        options.searched_candidate_count
    )?;
    writeln!(writer, "  \"reported_configs\": {},", results.len())?;
    writeln!(
        writer,
        "  \"omitted_rejected_configs\": {},",
        options.omitted_rejected_candidate_count
    )?;
    write_search_budget(writer, options.search_budget, "  ", true)?;
    write_search_diagnostics(writer, options.search_diagnostics, "  ", true)?;
    write_calibration(
        writer,
        CalibrationJsonContext {
            calibration: options.calibration,
            policy: options.calibration_policy,
            profile: options.calibration_profile,
            coverage: options.calibration_coverage,
            warnings: options.calibration_warnings,
            invalid_shape_warnings: options.calibration_invalid_shape_warnings,
            gate_violations: options.calibration_gate_violations,
        },
        "  ",
        true,
    )?;
    write_approximation_policy(writer, options.approximation_policy, "  ", true)?;
    write_trust_boundary_json(writer, "  ", true)?;
    writeln!(writer, "  \"results\": [")?;
    let nominal_ranks = parallelism_nominal_rank_map(results);
    let uncertainty_adjusted_ranks = parallelism_uncertainty_adjusted_rank_map(
        results,
        options.calibration_policy.uncertainty_ranking_weight,
    );
    for (idx, score) in results.iter().take(options.top_k).enumerate() {
        let candidate_id = parallelism_candidate_id(score);
        write_parallelism_score_json(
            writer,
            cluster,
            idx + 1,
            RankSensitivity {
                nominal_rank: *nominal_ranks.get(&candidate_id).unwrap_or(&(idx + 1)),
                uncertainty_adjusted_rank: *uncertainty_adjusted_ranks
                    .get(&candidate_id)
                    .unwrap_or(&(idx + 1)),
            },
            score,
            options,
            "    ",
        )?;
        if idx + 1 < results.iter().take(options.top_k).count() {
            writeln!(writer, ",")?;
        } else {
            writeln!(writer)?;
        }
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")
}

pub(super) fn write_serving_json_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredServingConfig],
    options: JsonOutputOptions<'_>,
) -> Result<(), io::Error> {
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema_version\": 1,")?;
    writeln!(writer, "  \"mode\": \"serving\",")?;
    writeln!(writer, "  \"cluster_gpus\": {},", cluster.total_gpus())?;
    write_cluster_inventory(writer, cluster, "  ", true)?;
    writeln!(
        writer,
        "  \"model_params_gb\": {},",
        json_f64(model.parameters.as_gigabytes())
    )?;
    writeln!(
        writer,
        "  \"model_parameter_count_billion\": {},",
        json_f64(model.parameter_count_billion())
    )?;
    writeln!(
        writer,
        "  \"searched_serving_pairs\": {},",
        options.searched_candidate_count
    )?;
    writeln!(writer, "  \"reported_serving_pairs\": {},", results.len())?;
    writeln!(
        writer,
        "  \"omitted_rejected_serving_pairs\": {},",
        options.omitted_rejected_candidate_count
    )?;
    write_search_budget(writer, options.search_budget, "  ", true)?;
    write_search_diagnostics(writer, options.search_diagnostics, "  ", true)?;
    writeln!(
        writer,
        "  \"objective\": {},",
        json_string(options.serving_objective.unwrap_or_default().as_str())
    )?;
    writeln!(
        writer,
        "  \"serving_stack\": {},",
        json_optional_string(options.serving_stack)
    )?;
    write_string_vec(
        writer,
        "  ",
        "serving_runtime_features",
        options.serving_runtime_features,
        true,
    )?;
    write_calibration(
        writer,
        CalibrationJsonContext {
            calibration: options.calibration,
            policy: options.calibration_policy,
            profile: options.calibration_profile,
            coverage: options.calibration_coverage,
            warnings: options.calibration_warnings,
            invalid_shape_warnings: options.calibration_invalid_shape_warnings,
            gate_violations: options.calibration_gate_violations,
        },
        "  ",
        true,
    )?;
    write_approximation_policy(writer, options.approximation_policy, "  ", true)?;
    write_trust_boundary_json(writer, "  ", true)?;
    write_serving_rejection_summary_json(writer, results, "  ", true)?;
    writeln!(writer, "  \"results\": [")?;
    let nominal_ranks = serving_nominal_rank_map(results);
    let uncertainty_adjusted_ranks = serving_uncertainty_adjusted_rank_map(
        results,
        options.calibration_policy.uncertainty_ranking_weight,
    );
    for (idx, score) in results.iter().take(options.top_k).enumerate() {
        let candidate_id = score.candidate_id.clone();
        write_serving_score_json(
            writer,
            cluster,
            idx + 1,
            RankSensitivity {
                nominal_rank: *nominal_ranks.get(&candidate_id).unwrap_or(&(idx + 1)),
                uncertainty_adjusted_rank: *uncertainty_adjusted_ranks
                    .get(&candidate_id)
                    .unwrap_or(&(idx + 1)),
            },
            score,
            options,
            "    ",
        )?;
        if idx + 1 < results.iter().take(options.top_k).count() {
            writeln!(writer, ",")?;
        } else {
            writeln!(writer)?;
        }
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")
}

fn write_parallelism_score_json<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    rank: usize,
    rank_sensitivity: RankSensitivity,
    score: &ScoredParallelismConfig,
    options: JsonOutputOptions<'_>,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}{{")?;
    let calibration_uncertainty = calibration_uncertainty_summary(score.calibration_fits.iter());
    writeln!(writer, "{indent}  \"rank\": {rank},")?;
    writeln!(
        writer,
        "{indent}  \"nominal_rank\": {},",
        rank_sensitivity.nominal_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_adjusted_rank\": {},",
        rank_sensitivity.uncertainty_adjusted_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_rank_delta\": {},",
        rank_sensitivity.delta()
    )?;
    writeln!(
        writer,
        "{indent}  \"candidate_id\": {},",
        json_string(&parallelism_candidate_id(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"status\": \"{}\",",
        status(score.feasible)
    )?;
    writeln!(writer, "{indent}  \"feasible\": {},", score.feasible)?;
    write_config(writer, &score.config, indent, "config", true)?;
    write_placement(writer, cluster, &score.placement, indent, "placement", true)?;
    write_placement_evidence(
        writer,
        indent,
        "placement_evidence",
        &score.placement_evidence,
        true,
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_ms\": {},",
        json_optional_ms(score.feasible, score.estimated_latency_s)
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_calibration_uncertainty_ms\": {},",
        json_optional_uncertainty_ms(
            score.feasible,
            calibration_uncertainty.absolute_uncertainty_s
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_calibration_lower_ms\": {},",
        json_metric_lower_ms(
            score.feasible,
            score.estimated_latency_s,
            &calibration_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_calibration_upper_ms\": {},",
        json_metric_upper_ms(
            score.feasible,
            score.estimated_latency_s,
            &calibration_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_uncertainty_adjusted_ms\": {},",
        json_optional_ms(
            score.feasible,
            uncertainty_adjusted_parallelism_latency_s(
                score,
                options.calibration_policy.uncertainty_ranking_weight
            )
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_memory_per_gpu_gb\": {},",
        json_f64(score.estimated_memory_per_gpu.as_gigabytes())
    )?;
    writeln!(
        writer,
        "{indent}  \"operation_makespan_ms\": {},",
        json_optional_ms(score.feasible, score.operation_makespan_s)
    )?;
    write_calibration_fit_applications(
        writer,
        indent,
        "calibration_fit_applications",
        &score.calibration_fits,
        true,
    )?;
    write_calibration_uncertainty(
        writer,
        indent,
        "calibration_uncertainty",
        &score.calibration_fits,
        true,
    )?;
    write_calibration_gate_violation_array(
        writer,
        &score.calibration_gate_violations,
        indent,
        "calibration_gate_violations",
        true,
    )?;
    write_approximations(writer, indent, &score.approximations, true)?;
    write_approximation_policy_violations(
        writer,
        indent,
        &score.approximation_policy_violations,
        true,
    )?;
    write_resource_utilization(writer, indent, &score.resource_utilization, true)?;
    if options.include_occupancy {
        let resources = selected_occupancy_resources(
            &score.resource_utilization,
            options.occupancy_resource_limit,
        );
        let occupancy = resource_occupancy_buckets(
            &score.scheduled_operations,
            score.operation_makespan_s,
            options.occupancy_buckets,
            &resources,
        );
        write_resource_occupancy(writer, indent, &occupancy, true)?;
    }
    if options.include_critical_path {
        let path = critical_path(&score.scheduled_operations);
        write_critical_path(writer, indent, &path, options.critical_path_limit, true)?;
    }
    write_string_vec(writer, indent, "bottlenecks", &score.bottlenecks, true)?;
    write_optional_string(
        writer,
        indent,
        "rejected_reason",
        score.rejected_reason.as_deref(),
        options.include_trace,
    )?;
    if options.include_trace {
        write_scheduled_operations(
            writer,
            indent,
            "scheduled_operations",
            &score.scheduled_operations,
            options.trace_limit,
            false,
        )?;
    }
    write!(writer, "{indent}}}")
}

fn write_serving_score_json<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    rank: usize,
    rank_sensitivity: RankSensitivity,
    score: &ScoredServingConfig,
    options: JsonOutputOptions<'_>,
    indent: &str,
) -> Result<(), io::Error> {
    let candidate_id = score.candidate_id.clone();
    writeln!(writer, "{indent}{{")?;
    writeln!(writer, "{indent}  \"rank\": {rank},")?;
    writeln!(
        writer,
        "{indent}  \"nominal_rank\": {},",
        rank_sensitivity.nominal_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_adjusted_rank\": {},",
        rank_sensitivity.uncertainty_adjusted_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_rank_delta\": {},",
        rank_sensitivity.delta()
    )?;
    writeln!(
        writer,
        "{indent}  \"candidate_id\": {},",
        json_string(&candidate_id)
    )?;
    writeln!(
        writer,
        "{indent}  \"status\": \"{}\",",
        status(score.feasible)
    )?;
    writeln!(writer, "{indent}  \"feasible\": {},", score.feasible)?;
    writeln!(
        writer,
        "{indent}  \"objective\": {},",
        json_string(score.objective.as_str())
    )?;
    writeln!(
        writer,
        "{indent}  \"objective_base_score\": {},",
        json_optional_f64(serving_objective_base_score(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"pareto_frontier\": {},",
        score.pareto.is_frontier
    )?;
    writeln!(
        writer,
        "{indent}  \"pareto_rank\": {},",
        json_optional_u32(score.pareto.rank)
    )?;
    write_string_vec(
        writer,
        indent,
        "pareto_dominated_by",
        &score.pareto.dominated_by,
        true,
    )?;
    write_pareto_dimensions(writer, indent, &score.pareto.dimensions, true)?;
    writeln!(
        writer,
        "{indent}  \"memory_pressure_peak_fraction\": {},",
        json_optional_f64(serving_peak_memory_pressure_fraction(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"memory_pressure_peak_phase\": {},",
        serving_peak_memory_pressure_phase(score)
            .map(json_string)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}  \"max_memory_pressure_fraction\": {},",
        json_optional_value(score.max_memory_pressure_fraction)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_unique_gpus\": {},",
        json_optional_u32(score.max_unique_gpus)
    )?;
    writeln!(
        writer,
        "{indent}  \"min_throughput_tokens_per_s\": {},",
        json_optional_value(score.min_throughput_tokens_per_s)
    )?;
    write_serving_metric_ceilings_json(writer, indent, score.metric_ceilings)?;
    write_kv_route_constraints_json(writer, indent, score.kv_route_constraints)?;
    writeln!(
        writer,
        "{indent}  \"slo_miss_penalty_weight\": {},",
        json_f64(score.slo_miss_penalty_weight)
    )?;
    write_slo_miss_penalty_weights_json(writer, score, indent)?;
    write_slo_miss_penalty_components_json(writer, score, indent)?;
    write_traffic_class_slo_miss_penalties_json(writer, score, indent)?;
    writeln!(
        writer,
        "{indent}  \"slo_miss_penalty_score\": {},",
        json_optional_f64(score.slo_miss_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}  \"service_backpressure_penalty_weight\": {},",
        json_f64(score.service_backpressure_penalty_weight)
    )?;
    writeln!(
        writer,
        "{indent}  \"service_backpressure_penalty_score\": {},",
        json_optional_f64(score.service_backpressure_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}  \"topology_risk_penalty_weight\": {},",
        json_f64(score.topology_risk_penalty_weight)
    )?;
    writeln!(
        writer,
        "{indent}  \"topology_risk_penalty_score\": {},",
        json_optional_f64(score.topology_risk_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}  \"objective_nominal_score\": {},",
        json_optional_f64(nominal_serving_objective_score(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"objective_uncertainty_adjusted_score\": {},",
        json_optional_f64(uncertainty_adjusted_serving_objective_score(
            score,
            options.calibration_policy.uncertainty_ranking_weight
        ))
    )?;
    write_serving_objective_breakdown(
        writer,
        score,
        options.calibration_policy.uncertainty_ranking_weight,
        indent,
        true,
    )?;
    writeln!(
        writer,
        "{indent}  \"pool\": {},",
        json_string(&pool_label(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"deployment_mode\": {},",
        json_string(score.deployment_mode.as_str())
    )?;
    write_route_coverage(writer, indent, score, true)?;
    write_pool_search_summary(writer, indent, score.pool_search_summary.as_ref(), true)?;
    write_pool_topology_summary(writer, indent, &score.pool_topology, true)?;
    write_u32_vec(writer, indent, "prefill_nodes", &score.prefill_nodes, true)?;
    write_u32_vec(writer, indent, "decode_nodes", &score.decode_nodes, true)?;
    write_string_vec(
        writer,
        indent,
        "prefill_gpu_labels",
        &score.prefill_gpu_labels,
        true,
    )?;
    write_string_vec(
        writer,
        indent,
        "decode_gpu_labels",
        &score.decode_gpu_labels,
        true,
    )?;
    write_serving_hardware_footprint(writer, indent, &score.hardware_footprint, true)?;
    write_serving_cost_estimate(writer, indent, &score.cost_estimate, true)?;
    write_config(
        writer,
        &score.prefill_config,
        indent,
        "prefill_config",
        true,
    )?;
    write_config(writer, &score.decode_config, indent, "decode_config", true)?;
    write_placement(
        writer,
        cluster,
        &score.prefill_score.placement,
        indent,
        "prefill_placement",
        true,
    )?;
    write_placement_evidence(
        writer,
        indent,
        "prefill_placement_evidence",
        &score.prefill_score.placement_evidence,
        true,
    )?;
    write_placement(
        writer,
        cluster,
        &score.decode_score.placement,
        indent,
        "decode_placement",
        true,
    )?;
    write_placement_evidence(
        writer,
        indent,
        "decode_placement_evidence",
        &score.decode_score.placement_evidence,
        true,
    )?;
    write_serving_memory(
        writer,
        indent,
        "prefill_memory",
        &score.prefill_memory,
        true,
    )?;
    write_serving_memory(writer, indent, "decode_memory", &score.decode_memory, true)?;
    write_serving_metrics(
        writer,
        score.feasible,
        &score.metrics,
        &score.calibration_fits,
        indent,
        true,
    )?;
    write_measurement_window(
        writer,
        score.feasible,
        &score.measurement_window,
        indent,
        true,
    )?;
    write_serving_bottleneck_summary(writer, indent, &score.bottleneck_summary, true)?;
    write_memory_pressure_observations(
        writer,
        indent,
        &score.memory_pressure,
        options.request_limit,
        true,
    )?;
    write_serving_calibration_summary(writer, indent, &score.calibration_summary, true)?;
    write_calibration_fit_applications(
        writer,
        indent,
        "calibration_fit_applications",
        &score.calibration_fits,
        true,
    )?;
    write_serving_phase_calibration(writer, indent, &score.phase_calibration, true)?;
    write_calibration_uncertainty(
        writer,
        indent,
        "calibration_uncertainty",
        &score.calibration_fits,
        true,
    )?;
    write_calibration_gate_violation_array(
        writer,
        &score.calibration_gate_violations,
        indent,
        "calibration_gate_violations",
        true,
    )?;
    write_serving_approximation_summary(writer, indent, &score.approximation_summary, true)?;
    write_approximations(writer, indent, &score.approximations, true)?;
    write_approximation_policy_violations(
        writer,
        indent,
        &score.approximation_policy_violations,
        true,
    )?;
    write_request_observations(writer, indent, score, options.request_limit, true)?;
    write_decode_iterations(
        writer,
        indent,
        &score.decode_iterations,
        if options.include_trace {
            options.trace_limit
        } else {
            options.request_limit
        },
        options.include_trace,
        true,
    )?;
    write_metric_breakdowns(writer, indent, &score.metric_breakdowns, true)?;
    write_node_capacity(writer, indent, &score.node_capacity, true)?;
    write_gpu_capacity(writer, indent, &score.gpu_capacity, true)?;
    write_traffic_class_capacity(writer, indent, &score.traffic_class_capacity, true)?;
    write_service_observations(writer, indent, &score.service_observations, true)?;
    write_worker_observations(writer, indent, &score.worker_observations, true)?;
    write_resource_utilization(writer, indent, &score.resource_utilization, true)?;
    write_phase_resource_utilization(writer, indent, &score.phase_resource_utilization, true)?;
    write_kv_route_topology_summary(writer, indent, &score.kv_route_topology_summary, true)?;
    write_topology_bottlenecks(writer, indent, &score.topology_bottlenecks, true)?;
    write_kv_route_resource_summary(writer, indent, &score.kv_route_resource_summary, true)?;
    if options.include_occupancy {
        let resources = selected_occupancy_resources(
            &score.resource_utilization,
            options.occupancy_resource_limit,
        );
        let occupancy = resource_occupancy_buckets(
            &score.scheduled_operations,
            score.metrics.scheduled_makespan_s,
            options.occupancy_buckets,
            &resources,
        );
        write_resource_occupancy(writer, indent, &occupancy, true)?;
    }
    if options.include_critical_path {
        let path = critical_path(&score.scheduled_operations);
        write_critical_path(writer, indent, &path, options.critical_path_limit, true)?;
    }
    write_string_vec(writer, indent, "bottlenecks", &score.bottlenecks, true)?;
    write_serving_rejections(writer, indent, &candidate_id, &score.rejections, true)?;
    write_optional_string(
        writer,
        indent,
        "rejected_reason",
        score.rejected_reason.as_deref(),
        options.include_trace,
    )?;
    if options.include_trace {
        write_scheduled_operations(
            writer,
            indent,
            "scheduled_operations",
            &score.scheduled_operations,
            options.trace_limit,
            false,
        )?;
    }
    write!(writer, "{indent}}}")
}

fn write_serving_metric_ceilings_json<W: Write>(
    writer: &mut W,
    indent: &str,
    ceilings: ServingMetricCeilings,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"metric_ceilings\": {{")?;
    writeln!(
        writer,
        "{indent}    \"max_ttft_s\": {},",
        json_optional_value(ceilings.max_ttft_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_tpot_s\": {},",
        json_optional_value(ceilings.max_tpot_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_itl_s\": {},",
        json_optional_value(ceilings.max_itl_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_e2el_s\": {}",
        json_optional_value(ceilings.max_e2el_s)
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

fn write_serving_objective_breakdown<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    uncertainty_ranking_weight: f64,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let descriptor = serving_objective_metric_descriptor(score);
    let base_score = serving_objective_base_score(score);
    let uncertainty_adjusted_base_score =
        serving_uncertainty_adjusted_base_score(score, uncertainty_ranking_weight);
    let nominal_score = nominal_serving_objective_score(score);
    let uncertainty_adjusted_score =
        uncertainty_adjusted_serving_objective_score(score, uncertainty_ranking_weight);

    writeln!(writer, "{indent}  \"objective_breakdown\": {{")?;
    writeln!(
        writer,
        "{indent}    \"selected_objective\": {},",
        json_string(score.objective.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"score_convention\": \"lower_is_better\","
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric\": {},",
        json_string(descriptor.metric)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric_direction\": {},",
        json_string(descriptor.direction)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric_unit\": {},",
        json_string(descriptor.unit)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric_value\": {},",
        json_optional_f64(descriptor.value)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_score\": {},",
        json_optional_f64(base_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"slo_miss_penalty_score\": {},",
        json_optional_f64(score.slo_miss_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"service_backpressure_penalty_score\": {},",
        json_optional_f64(score.service_backpressure_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"topology_risk_penalty_score\": {},",
        json_optional_f64(score.topology_risk_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"nominal_penalty_score\": {},",
        json_optional_f64(serving_objective_penalty_score(score))
    )?;
    writeln!(
        writer,
        "{indent}    \"nominal_score\": {},",
        json_optional_f64(nominal_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_ranking_weight\": {},",
        json_optional_f64(uncertainty_ranking_weight)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_adjusted_base_score\": {},",
        json_optional_f64(uncertainty_adjusted_base_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_adjusted_score\": {},",
        json_optional_f64(uncertainty_adjusted_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_adjusted_delta\": {},",
        json_optional_f64(uncertainty_adjusted_score - nominal_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"largest_nominal_term\": {}",
        json_string(serving_largest_nominal_objective_term(score))
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_slo_miss_penalty_weights_json<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"slo_miss_penalty_weights\": {{")?;
    write_slo_miss_penalty_weights_fields_json(
        writer,
        score.slo_miss_penalty_weights,
        indent,
        "    ",
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

fn write_slo_miss_penalty_components_json<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"slo_miss_penalty_components\": {{")?;
    write_slo_miss_penalty_components_fields_json(
        writer,
        score.slo_miss_penalty_components,
        indent,
        "    ",
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

fn write_slo_miss_penalty_weights_object_json<W: Write>(
    writer: &mut W,
    weights: ServingSloMissPenaltyWeights,
    indent: &str,
    inner: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}{inner}\"weights\": {{")?;
    let nested = format!("{inner}  ");
    write_slo_miss_penalty_weights_fields_json(writer, weights, indent, &nested)?;
    writeln!(writer, "{indent}{inner}}},")?;
    Ok(())
}

fn write_slo_miss_penalty_weights_fields_json<W: Write>(
    writer: &mut W,
    weights: ServingSloMissPenaltyWeights,
    indent: &str,
    inner: &str,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}{inner}\"aggregate\": {},",
        json_f64(weights.aggregate)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"ttft\": {},",
        json_f64(weights.ttft)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"tpot\": {},",
        json_f64(weights.tpot)
    )?;
    writeln!(writer, "{indent}{inner}\"itl\": {},", json_f64(weights.itl))?;
    writeln!(
        writer,
        "{indent}{inner}\"e2el\": {},",
        json_f64(weights.e2el)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"deadline\": {}",
        json_f64(weights.deadline)
    )?;
    Ok(())
}

fn write_slo_miss_penalty_components_object_json<W: Write>(
    writer: &mut W,
    components: ServingSloMissPenaltyComponents,
    indent: &str,
    inner: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}{inner}\"components\": {{")?;
    let nested = format!("{inner}  ");
    write_slo_miss_penalty_components_fields_json(writer, components, indent, &nested)?;
    writeln!(writer, "{indent}{inner}}}{}", comma(trailing_comma))?;
    Ok(())
}

fn write_slo_miss_penalty_components_fields_json<W: Write>(
    writer: &mut W,
    components: ServingSloMissPenaltyComponents,
    indent: &str,
    inner: &str,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}{inner}\"ttft\": {},",
        json_optional_f64(components.ttft)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"tpot\": {},",
        json_optional_f64(components.tpot)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"itl\": {},",
        json_optional_f64(components.itl)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"e2el\": {},",
        json_optional_f64(components.e2el)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"deadline\": {},",
        json_optional_f64(components.deadline)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"total\": {}",
        json_optional_f64(components.total)
    )?;
    Ok(())
}

fn write_traffic_class_slo_miss_penalties_json<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"traffic_class_slo_miss_penalties\": [")?;
    for (idx, penalty) in score.traffic_class_slo_miss_penalties.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&penalty.name)
        )?;
        writeln!(
            writer,
            "{indent}      \"group\": {},",
            json_string(&penalty.group)
        )?;
        writeln!(
            writer,
            "{indent}      \"key\": {},",
            json_string(&penalty.key)
        )?;
        write_slo_miss_penalty_weights_object_json(writer, penalty.weights, indent, "      ")?;
        write_slo_miss_penalty_components_object_json(
            writer,
            penalty.components,
            indent,
            "      ",
            false,
        )?;
        if idx + 1 < score.traffic_class_slo_miss_penalties.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ],")?;
    Ok(())
}

fn write_config<W: Write>(
    writer: &mut W,
    config: &ParallelismConfig,
    indent: &str,
    field: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": {{")?;
    writeln!(
        writer,
        "{indent}    \"tensor_ranks\": {},",
        config.tensor_ranks
    )?;
    writeln!(
        writer,
        "{indent}    \"pipeline_ranks\": {},",
        config.pipeline_ranks
    )?;
    writeln!(
        writer,
        "{indent}    \"expert_ranks\": {},",
        config.expert_ranks
    )?;
    writeln!(writer, "{indent}    \"data_ranks\": {},", config.data_ranks)?;
    writeln!(
        writer,
        "{indent}    \"total_ranks\": {}",
        config.total_ranks()
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_placement<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    placement: &RankPlacement,
    indent: &str,
    field: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, addr) in placement.rank_to_gpu.iter().enumerate() {
        let gpu_label = cluster
            .gpu_profile(*addr)
            .map(|profile| profile.label)
            .unwrap_or("unknown");
        writeln!(writer, "{indent}    {{")?;
        writeln!(writer, "{indent}      \"rank\": {},", idx)?;
        writeln!(writer, "{indent}      \"node_id\": {},", addr.node_id)?;
        writeln!(
            writer,
            "{indent}      \"local_gpu_id\": {},",
            addr.local_gpu_id
        )?;
        writeln!(writer, "{indent}      \"gpu\": {}", json_string(gpu_label))?;
        if idx + 1 < placement.rank_to_gpu.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_placement_evidence<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    evidence_items: &[PlacementEvidence],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, evidence) in evidence_items.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"decision\": {},",
            json_string(&evidence.decision)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(&evidence.scope)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&evidence.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&evidence.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(evidence.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(evidence.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {},",
            evidence
                .unit
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {},",
            evidence
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {}",
            json_string(&evidence.message)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < evidence_items.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}
