use super::*;

pub(super) fn parse_run_config_with_base_dir(
    contents: &str,
    base_dir: Option<&Path>,
) -> Result<RunConfig, ConfigError> {
    let file: RunFile = toml::from_str(contents)
        .map_err(|err| ConfigError::new(format!("invalid run TOML: {err}")))?;
    validate_schema_version("run", file.schema_version)?;
    let run = file.run;
    let cluster_path = file
        .cluster
        .or_else(|| run.as_ref().and_then(|run| run.cluster.clone()))
        .map(|path| resolve_config_path(&path, base_dir));
    let workload_path = file
        .workload
        .or_else(|| run.as_ref().and_then(|run| run.workload.clone()))
        .map(|path| resolve_config_path(&path, base_dir));
    let output = parse_run_output(file.output, base_dir)?;
    let search_budget = parse_run_search_budget(file.search)?;
    let scenarios = parse_run_scenarios(file.scenarios, base_dir)?;

    Ok(RunConfig {
        cluster_path,
        workload_path,
        output,
        search_budget,
        scenarios,
    })
}

pub(super) fn parse_run_output(
    section: Option<RunOutputSection>,
    base_dir: Option<&Path>,
) -> Result<RunOutputConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunOutputConfig::default());
    };
    if let Some(top_k) = section.top_k
        && top_k == 0
    {
        return Err(ConfigError::new("output.top_k must be greater than zero"));
    }
    if let Some(occupancy_buckets) = section.occupancy_buckets
        && occupancy_buckets == 0
    {
        return Err(ConfigError::new(
            "output.occupancy_buckets must be greater than zero",
        ));
    }
    let output_dir =
        parse_run_output_path("output.output_dir", section.output_dir.as_deref(), base_dir)?;
    let request_metrics_csv_path = parse_run_output_path(
        "output.request_metrics_csv",
        section.request_metrics_csv_path.as_deref(),
        base_dir,
    )?;
    let request_lifecycle_events_csv_path = parse_run_output_path(
        "output.request_lifecycle_events_csv",
        section.request_lifecycle_events_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_metrics_csv_path = parse_run_output_path(
        "output.serving_metrics_csv",
        section.serving_metrics_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_metric_breakdowns_csv_path = parse_run_output_path(
        "output.serving_metric_breakdowns_csv",
        section.serving_metric_breakdowns_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_services_csv_path = parse_run_output_path(
        "output.serving_services_csv",
        section.serving_services_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_utilization_csv_path = parse_run_output_path(
        "output.serving_utilization_csv",
        section.serving_utilization_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_memory_pressure_csv_path = parse_run_output_path(
        "output.serving_memory_pressure_csv",
        section.serving_memory_pressure_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_timeline_csv_path = parse_run_output_path(
        "output.serving_timeline_csv",
        section.serving_timeline_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_occupancy_csv_path = parse_run_output_path(
        "output.serving_occupancy_csv",
        section.serving_occupancy_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_placement_evidence_csv_path = parse_run_output_path(
        "output.serving_placement_evidence_csv",
        section.serving_placement_evidence_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_worker_evidence_csv_path = parse_run_output_path(
        "output.serving_worker_evidence_csv",
        section.serving_worker_evidence_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_rejections_csv_path = parse_run_output_path(
        "output.serving_rejections_csv",
        section.serving_rejections_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_route_paths_csv_path = parse_run_output_path(
        "output.serving_route_paths_csv",
        section.serving_route_paths_csv_path.as_deref(),
        base_dir,
    )?;
    let kv_route_resources_csv_path = parse_run_output_path(
        "output.kv_route_resources_csv",
        section.kv_route_resources_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_bottlenecks_csv_path = parse_run_output_path(
        "output.serving_bottlenecks_csv",
        section.serving_bottlenecks_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_phase_calibration_csv_path = parse_run_output_path(
        "output.serving_phase_calibration_csv",
        section.serving_phase_calibration_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_approximations_csv_path = parse_run_output_path(
        "output.serving_approximations_csv",
        section.serving_approximations_csv_path.as_deref(),
        base_dir,
    )?;
    let calibration_residuals_csv_path = parse_run_output_path(
        "output.calibration_residuals_csv",
        section.calibration_residuals_csv_path.as_deref(),
        base_dir,
    )?;
    let scenario_sensitivity_csv_path = parse_run_output_path(
        "output.scenario_sensitivity_csv",
        section.scenario_sensitivity_csv_path.as_deref(),
        base_dir,
    )?;
    let rank_sensitivity_csv_path = parse_run_output_path(
        "output.rank_sensitivity_csv",
        section.rank_sensitivity_csv_path.as_deref(),
        base_dir,
    )?;

    Ok(RunOutputConfig {
        format: section.format.map(|format| normalize(&format)),
        top_k: section.top_k,
        output_dir,
        output_profile: section.output_profile.map(|profile| normalize(&profile)),
        request_metrics_csv_path,
        request_lifecycle_events_csv_path,
        serving_metrics_csv_path,
        serving_metric_breakdowns_csv_path,
        serving_services_csv_path,
        serving_utilization_csv_path,
        serving_memory_pressure_csv_path,
        serving_timeline_csv_path,
        serving_occupancy_csv_path,
        serving_placement_evidence_csv_path,
        serving_worker_evidence_csv_path,
        serving_rejections_csv_path,
        serving_route_paths_csv_path,
        kv_route_resources_csv_path,
        serving_bottlenecks_csv_path,
        serving_phase_calibration_csv_path,
        serving_approximations_csv_path,
        calibration_residuals_csv_path,
        scenario_sensitivity_csv_path,
        rank_sensitivity_csv_path,
        trace: section.trace,
        trace_limit: section.trace_limit,
        request_limit: section.request_limit,
        occupancy: section.occupancy,
        occupancy_buckets: section.occupancy_buckets,
        occupancy_resource_limit: section.occupancy_resource_limit,
        critical_path: section.critical_path,
        critical_path_limit: section.critical_path_limit,
    })
}

pub(super) fn parse_run_output_path(
    name: &str,
    path: Option<&str>,
    base_dir: Option<&Path>,
) -> Result<Option<PathBuf>, ConfigError> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.trim().is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }
    Ok(Some(resolve_config_path(path, base_dir)))
}

pub(super) fn parse_run_search_budget(
    section: Option<RunSearchBudgetSection>,
) -> Result<RunSearchBudgetConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunSearchBudgetConfig::default());
    };
    validate_positive_usize(
        "search.max_parallelism_candidates",
        section.max_parallelism_candidates,
    )?;
    validate_positive_usize(
        "search.max_prefill_candidates",
        section.max_prefill_candidates,
    )?;
    validate_positive_usize(
        "search.max_decode_candidates",
        section.max_decode_candidates,
    )?;
    validate_positive_usize("search.max_serving_pairs", section.max_serving_pairs)?;

    Ok(RunSearchBudgetConfig {
        max_parallelism_candidates: section.max_parallelism_candidates,
        max_prefill_candidates: section.max_prefill_candidates,
        max_decode_candidates: section.max_decode_candidates,
        max_serving_pairs: section.max_serving_pairs,
        max_runtime_ms: section.max_runtime_ms,
        retain_rejected_candidates: section.retain_rejected_candidates,
    })
}

pub(super) fn validate_positive_usize(name: &str, value: Option<usize>) -> Result<(), ConfigError> {
    if let Some(value) = value
        && value == 0
    {
        return Err(ConfigError::new(format!(
            "{name} must be greater than zero"
        )));
    }
    Ok(())
}

pub(super) fn parse_run_scenarios(
    sections: Option<Vec<RunScenarioSection>>,
    base_dir: Option<&Path>,
) -> Result<Vec<RunScenarioConfig>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut scenarios = Vec::with_capacity(sections.len());
    let mut names = HashSet::new();
    for (idx, section) in sections.into_iter().enumerate() {
        let default_name = format!("scenario-{}", idx + 1);
        let name = section
            .name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(&default_name)
            .to_string();
        if !names.insert(name.clone()) {
            return Err(ConfigError::new(format!(
                "scenarios[{idx}].name '{name}' is duplicated"
            )));
        }
        validate_positive_optional_u32(
            &format!("scenarios[{idx}].request_count"),
            section.request_count,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].arrival_gap_scale"),
            section.arrival_gap_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].arrival_rate_scale"),
            section.arrival_rate_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].batch_size_scale"),
            section.batch_size_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].prompt_tokens_scale"),
            section.prompt_tokens_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].decode_tokens_scale"),
            section.decode_tokens_scale,
        )?;
        let calibration_profile_path = parse_run_scenario_calibration_profile_path(
            idx,
            section.calibration_profile.as_deref(),
            base_dir,
        )?;
        let calibration = parse_run_scenario_calibration(idx, section.calibration.as_ref())?;
        let topology = parse_run_scenario_topology(idx, section.topology.as_ref())?;
        scenarios.push(RunScenarioConfig {
            name,
            request_count: section.request_count,
            arrival_gap_scale: section.arrival_gap_scale,
            arrival_rate_scale: section.arrival_rate_scale,
            batch_size_scale: section.batch_size_scale,
            prompt_tokens_scale: section.prompt_tokens_scale,
            decode_tokens_scale: section.decode_tokens_scale,
            calibration_profile_path,
            calibration,
            topology,
        });
    }
    Ok(scenarios)
}

pub(super) fn parse_run_scenario_calibration_profile_path(
    idx: usize,
    path: Option<&str>,
    base_dir: Option<&Path>,
) -> Result<Option<PathBuf>, ConfigError> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.trim().is_empty() {
        return Err(ConfigError::new(format!(
            "scenarios[{idx}].calibration_profile must not be empty"
        )));
    }
    Ok(Some(resolve_config_path(path, base_dir)))
}

pub(super) fn parse_run_scenario_calibration(
    idx: usize,
    section: Option<&CalibrationSection>,
) -> Result<RunScenarioCalibrationConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunScenarioCalibrationConfig::default());
    };
    let prefix = format!("scenarios[{idx}].calibration");
    validate_positive_fraction_optional_f64(
        &format!("{prefix}.compute_efficiency"),
        section.compute_efficiency,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.prefill_compute_scale"),
        section.prefill_compute_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.decode_compute_scale"),
        section.decode_compute_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.decode_memory_bandwidth_scale"),
        section.decode_memory_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.collective_latency_scale"),
        section.collective_latency_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.collective_bandwidth_scale"),
        section.collective_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.kv_transfer_scale"),
        section.kv_transfer_scale,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.scheduler_overhead_us"),
        section.scheduler_overhead_us,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_temporary_fraction"),
        section.serving_memory_temporary_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_activation_communication_fraction"),
        section.serving_memory_activation_communication_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_weight_communication_fraction"),
        section.serving_memory_weight_communication_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_runtime_reserve_fraction"),
        section.serving_memory_runtime_reserve_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_fragmentation_fraction"),
        section.serving_memory_fragmentation_fraction,
    )?;
    validate_positive_optional_u32(
        &format!("{prefix}.serving_pipeline_depth"),
        section.serving_pipeline_depth,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.request_arrival_gap_s"),
        section.request_arrival_gap_s,
    )?;

    Ok(calibration_overrides_from_section(Some(section)))
}

pub(super) fn parse_run_scenario_topology(
    idx: usize,
    section: Option<&RunScenarioTopologySection>,
) -> Result<RunScenarioTopologyConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunScenarioTopologyConfig::default());
    };
    let prefix = format!("scenarios[{idx}].topology");
    validate_positive_optional_f64(
        &format!("{prefix}.interconnect_bandwidth_scale"),
        section.interconnect_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.interconnect_latency_scale"),
        section.interconnect_latency_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.nic_bandwidth_scale"),
        section.nic_bandwidth_scale,
    )?;
    let node_states = parse_run_scenario_node_state_overlays(
        &format!("{prefix}.node_states"),
        section.node_states.as_deref(),
    )?;
    let disabled_gpus = parse_run_scenario_gpu_overlays(
        &format!("{prefix}.disabled_gpus"),
        section.disabled_gpus.as_deref(),
    )?;
    let disabled_nics = parse_run_scenario_nic_overlays(
        &format!("{prefix}.disabled_nics"),
        section.disabled_nics.as_deref(),
    )?;
    let degraded_gpus = parse_run_scenario_gpu_degradation_overlays(
        &format!("{prefix}.degraded_gpus"),
        section.degraded_gpus.as_deref(),
    )?;
    let degraded_nics = parse_run_scenario_nic_degradation_overlays(
        &format!("{prefix}.degraded_nics"),
        section.degraded_nics.as_deref(),
    )?;
    let degraded_rails = parse_run_scenario_rail_degradation_overlays(
        &format!("{prefix}.degraded_rails"),
        section.degraded_rails.as_deref(),
    )?;
    let degraded_links = parse_run_scenario_link_degradation_overlays(
        &format!("{prefix}.degraded_links"),
        section.degraded_links.as_deref(),
    )?;
    Ok(RunScenarioTopologyConfig {
        interconnect_bandwidth_scale: section.interconnect_bandwidth_scale,
        interconnect_latency_scale: section.interconnect_latency_scale,
        nic_bandwidth_scale: section.nic_bandwidth_scale,
        node_states,
        disabled_gpus,
        disabled_nics,
        degraded_gpus,
        degraded_nics,
        degraded_rails,
        degraded_links,
    })
}

pub(super) fn parse_run_scenario_node_state_overlays(
    name: &str,
    sections: Option<&[RunScenarioNodeStateOverlaySection]>,
) -> Result<Vec<RunScenarioNodeStateOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        let state = parse_run_scenario_node_state(&prefix, section.state.as_deref())?;
        let overlay = RunScenarioNodeStateOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            state,
        };
        validate_run_scenario_node_overlay_selectors(&prefix, &overlay)?;
        overlays.push(overlay);
    }
    Ok(overlays)
}

pub(super) fn parse_run_scenario_node_state(
    prefix: &str,
    state: Option<&str>,
) -> Result<RunScenarioNodeState, ConfigError> {
    let Some(state) = state.map(str::trim).filter(|state| !state.is_empty()) else {
        return Err(ConfigError::new(format!("{prefix}.state is required")));
    };

    match normalize(state).as_str() {
        "disabled" | "offline" | "unavailable" => Ok(RunScenarioNodeState::Disabled),
        "maintenance" | "maintenance_mode" => Ok(RunScenarioNodeState::Maintenance),
        "draining" | "drain" => Ok(RunScenarioNodeState::Draining),
        "reserved" | "reserved_capacity" => Ok(RunScenarioNodeState::Reserved),
        _ => Err(ConfigError::new(format!(
            "{prefix}.state '{state}' is unsupported; supported states are disabled, maintenance, draining, and reserved"
        ))),
    }
}

pub(super) fn parse_run_scenario_gpu_overlays(
    name: &str,
    sections: Option<&[RunScenarioGpuResourceOverlaySection]>,
) -> Result<Vec<RunScenarioGpuResourceOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        overlays.push(RunScenarioGpuResourceOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            gpu_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "gpu",
                section.gpu,
                section.gpus.as_deref(),
            )?,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

pub(super) fn parse_run_scenario_nic_overlays(
    name: &str,
    sections: Option<&[RunScenarioNicResourceOverlaySection]>,
) -> Result<Vec<RunScenarioNicResourceOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        overlays.push(RunScenarioNicResourceOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            nic_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "nic",
                section.nic,
                section.nics.as_deref(),
            )?,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

pub(super) fn parse_run_scenario_gpu_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioGpuDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioGpuDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(&format!("{prefix}.compute_scale"), section.compute_scale)?;
        validate_positive_optional_f64(
            &format!("{prefix}.hbm_bandwidth_scale"),
            section.hbm_bandwidth_scale,
        )?;
        validate_positive_optional_f64(
            &format!("{prefix}.hbm_capacity_scale"),
            section.hbm_capacity_scale,
        )?;
        if section.compute_scale.is_none()
            && section.hbm_bandwidth_scale.is_none()
            && section.hbm_capacity_scale.is_none()
        {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of compute_scale, hbm_bandwidth_scale, or hbm_capacity_scale"
            )));
        }
        overlays.push(RunScenarioGpuDegradationOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            gpu_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "gpu",
                section.gpu,
                section.gpus.as_deref(),
            )?,
            compute_scale: section.compute_scale,
            hbm_bandwidth_scale: section.hbm_bandwidth_scale,
            hbm_capacity_scale: section.hbm_capacity_scale,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

pub(super) fn parse_run_scenario_nic_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioNicDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioNicDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(
            &format!("{prefix}.bandwidth_scale"),
            section.bandwidth_scale,
        )?;
        validate_positive_optional_f64(&format!("{prefix}.latency_scale"), section.latency_scale)?;
        if section.bandwidth_scale.is_none() && section.latency_scale.is_none() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of bandwidth_scale or latency_scale"
            )));
        }
        overlays.push(RunScenarioNicDegradationOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            nic_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "nic",
                section.nic,
                section.nics.as_deref(),
            )?,
            bandwidth_scale: section.bandwidth_scale,
            latency_scale: section.latency_scale,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

pub(super) fn parse_run_scenario_rail_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioRailDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioRailDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(
            &format!("{prefix}.bandwidth_scale"),
            section.bandwidth_scale,
        )?;
        validate_positive_optional_f64(&format!("{prefix}.latency_scale"), section.latency_scale)?;
        if section.bandwidth_scale.is_none() && section.latency_scale.is_none() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of bandwidth_scale or latency_scale"
            )));
        }
        let rails =
            parse_run_scenario_link_overlay_rails(&prefix, section.rail, section.rails.as_deref())?;
        if rails.is_empty() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one rail or rails selector"
            )));
        }
        overlays.push(RunScenarioRailDegradationOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            rails,
            bandwidth_scale: section.bandwidth_scale,
            latency_scale: section.latency_scale,
        });
    }
    Ok(overlays)
}

pub(super) fn parse_run_scenario_link_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioLinkDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioLinkDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(
            &format!("{prefix}.bandwidth_scale"),
            section.bandwidth_scale,
        )?;
        validate_positive_optional_f64(&format!("{prefix}.latency_scale"), section.latency_scale)?;
        if section.bandwidth_scale.is_none() && section.latency_scale.is_none() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of bandwidth_scale or latency_scale"
            )));
        }
        let rails =
            parse_run_scenario_link_overlay_rails(&prefix, section.rail, section.rails.as_deref())?;
        let overlay = RunScenarioLinkDegradationOverlay {
            from_node_ids: parse_run_scenario_overlay_node_ids(
                &format!("{prefix}.from"),
                section.from,
                section.from_nodes.as_deref(),
            )?,
            from_node_groups: parse_run_scenario_overlay_node_groups(
                &format!("{prefix}.from"),
                section.from_group.as_deref(),
                section.from_groups.as_deref(),
            )?,
            from_node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.from_node_tags"),
                section.from_node_tag.as_deref(),
                section.from_node_tags.as_deref(),
            )?,
            from_racks: parse_optional_topology_domains(
                &format!("{prefix}.from_racks"),
                section.from_rack.as_deref(),
                section.from_racks.as_deref(),
            )?,
            from_islands: parse_optional_topology_domains(
                &format!("{prefix}.from_islands"),
                section.from_island.as_deref(),
                section.from_islands.as_deref(),
            )?,
            from_failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.from_failure_domains"),
                section.from_failure_domain.as_deref(),
                section.from_failure_domains.as_deref(),
            )?,
            from_gpus: parse_run_scenario_overlay_resource_ids(
                &format!("{prefix}.from"),
                "gpu",
                section.from_gpu,
                section.from_gpus.as_deref(),
            )?,
            to_node_ids: parse_run_scenario_overlay_node_ids(
                &format!("{prefix}.to"),
                section.to,
                section.to_nodes.as_deref(),
            )?,
            to_node_groups: parse_run_scenario_overlay_node_groups(
                &format!("{prefix}.to"),
                section.to_group.as_deref(),
                section.to_groups.as_deref(),
            )?,
            to_node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.to_node_tags"),
                section.to_node_tag.as_deref(),
                section.to_node_tags.as_deref(),
            )?,
            to_racks: parse_optional_topology_domains(
                &format!("{prefix}.to_racks"),
                section.to_rack.as_deref(),
                section.to_racks.as_deref(),
            )?,
            to_islands: parse_optional_topology_domains(
                &format!("{prefix}.to_islands"),
                section.to_island.as_deref(),
                section.to_islands.as_deref(),
            )?,
            to_failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.to_failure_domains"),
                section.to_failure_domain.as_deref(),
                section.to_failure_domains.as_deref(),
            )?,
            to_gpus: parse_run_scenario_overlay_resource_ids(
                &format!("{prefix}.to"),
                "gpu",
                section.to_gpu,
                section.to_gpus.as_deref(),
            )?,
            rails,
            bandwidth_scale: section.bandwidth_scale,
            latency_scale: section.latency_scale,
        };
        validate_run_scenario_link_overlay_selectors(&prefix, &overlay)?;
        overlays.push(overlay);
    }
    Ok(overlays)
}

pub(super) trait RunScenarioResourceOverlay {
    fn node_ids(&self) -> &[u32];
    fn node_groups(&self) -> &[String];
    fn node_tags(&self) -> &[String];
    fn racks(&self) -> &[String];
    fn islands(&self) -> &[String];
    fn failure_domains(&self) -> &[String];
    fn resource_ids(&self) -> &[u32];
}

impl RunScenarioResourceOverlay for RunScenarioGpuResourceOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.gpu_ids
    }
}

impl RunScenarioResourceOverlay for RunScenarioNicResourceOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.nic_ids
    }
}

impl RunScenarioResourceOverlay for RunScenarioGpuDegradationOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.gpu_ids
    }
}

impl RunScenarioResourceOverlay for RunScenarioNicDegradationOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.nic_ids
    }
}

pub(super) fn validate_run_scenario_overlay_selectors(
    name: &str,
    overlay: &impl RunScenarioResourceOverlay,
) -> Result<(), ConfigError> {
    if overlay.node_ids().is_empty()
        && overlay.node_groups().is_empty()
        && overlay.node_tags().is_empty()
        && overlay.racks().is_empty()
        && overlay.islands().is_empty()
        && overlay.failure_domains().is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one node, nodes, group, groups, node_tag, rack, island, or failure_domain selector"
        )));
    }
    if overlay.resource_ids().is_empty() {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one resource id"
        )));
    }
    Ok(())
}

pub(super) fn validate_optional_placement_for_cluster(
    name: &str,
    cluster: &Cluster,
    placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    let Some(placement) = placement else {
        return Ok(());
    };
    let mut used_gpus = BTreeSet::new();
    for (rank, addr) in placement.rank_to_gpu.iter().copied().enumerate() {
        let node = cluster.nodes.get(&addr.node_id).ok_or_else(|| {
            ConfigError::new(format!(
                "{name}.ranks[{rank}] references unknown node id {}",
                addr.node_id
            ))
        })?;
        if !node.gpus.contains_key(&addr.local_gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.ranks[{rank}] references unknown local GPU id {} on node {}",
                addr.local_gpu_id, addr.node_id
            )));
        }
        if !cluster.is_gpu_available(addr) {
            return Err(ConfigError::new(format!(
                "{name}.ranks[{rank}] references unavailable node {} gpu {}",
                addr.node_id, addr.local_gpu_id
            )));
        }
        if !used_gpus.insert(addr) {
            return Err(ConfigError::new(format!(
                "{name}.ranks maps multiple ranks to node {} gpu {}",
                addr.node_id, addr.local_gpu_id
            )));
        }
    }
    Ok(())
}

pub(super) fn model_dtype_label(dtype: DType) -> &'static str {
    dtype.label()
}

pub(super) fn gpu_profile_supports_dtype(profile: &GpuProfile, dtype: DType) -> bool {
    match dtype {
        DType::Fp8 => profile.peak_f8_flops.is_some(),
        DType::Fp16 | DType::Bf16 | DType::Int8 => true,
    }
}

pub(super) fn available_gpus_supporting_dtype(cluster: &Cluster, dtype: DType) -> u32 {
    cluster
        .nodes
        .iter()
        .flat_map(|(&node_id, node)| {
            node.gpus.keys().copied().map(move |local_gpu_id| GpuAddr {
                node_id,
                local_gpu_id,
            })
        })
        .filter(|addr| {
            cluster.is_gpu_available(*addr)
                && cluster
                    .gpu_profile(*addr)
                    .is_some_and(|profile| gpu_profile_supports_dtype(&profile, dtype))
        })
        .count()
        .min(u32::MAX as usize) as u32
}

pub(super) fn validate_cluster_dtype_support(
    cluster: &Cluster,
    dtype: DType,
) -> Result<u32, ConfigError> {
    let dtype_gpus = available_gpus_supporting_dtype(cluster, dtype);
    if dtype_gpus > 0 {
        return Ok(dtype_gpus);
    }

    Err(ConfigError::new(format!(
        "model.dtype {} requires at least one available GPU with {} tensor throughput, but the cluster has no available {}-capable GPUs",
        model_dtype_label(dtype),
        model_dtype_label(dtype),
        model_dtype_label(dtype)
    )))
}

pub(super) fn validate_optional_placement_dtype_support(
    name: &str,
    cluster: &Cluster,
    placement: Option<&RankPlacement>,
    dtype: DType,
) -> Result<(), ConfigError> {
    let Some(placement) = placement else {
        return Ok(());
    };

    for (rank, addr) in placement.rank_to_gpu.iter().copied().enumerate() {
        let Some(profile) = cluster.gpu_profile(addr) else {
            continue;
        };
        if !gpu_profile_supports_dtype(&profile, dtype) {
            return Err(ConfigError::new(format!(
                "{name}.ranks[{rank}] uses node {} gpu {} ({}) which does not support model.dtype {}",
                addr.node_id,
                addr.local_gpu_id,
                profile.label,
                model_dtype_label(dtype)
            )));
        }
    }

    Ok(())
}

pub(super) fn validate_optional_placement_rank_count(
    name: &str,
    search: &SearchSpace,
    placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    let Some(placement) = placement else {
        return Ok(());
    };
    let placement_ranks = placement.rank_to_gpu.len() as u32;
    if search_space_rank_counts(search).contains(&placement_ranks) {
        return Ok(());
    }
    Err(ConfigError::new(format!(
        "{name}.ranks defines {placement_ranks} ranks but no search candidate has that total rank count"
    )))
}

pub(super) fn search_space_rank_counts(search: &SearchSpace) -> BTreeSet<u32> {
    let mut counts = BTreeSet::new();
    for tensor in &search.tensor_ranks {
        for pipeline in &search.pipeline_ranks {
            for expert in &search.expert_ranks {
                for data in &search.data_ranks {
                    counts.insert(
                        tensor
                            .saturating_mul(*pipeline)
                            .saturating_mul(*expert)
                            .saturating_mul(*data),
                    );
                }
            }
        }
    }
    counts
}

pub(super) fn validate_run_scenario_link_overlay_selectors(
    name: &str,
    overlay: &RunScenarioLinkDegradationOverlay,
) -> Result<(), ConfigError> {
    if overlay.from_node_ids.is_empty()
        && overlay.from_node_groups.is_empty()
        && overlay.from_node_tags.is_empty()
        && overlay.from_racks.is_empty()
        && overlay.from_islands.is_empty()
        && overlay.from_failure_domains.is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one from, from_nodes, from_group, from_groups, from_node_tag, from_rack, from_island, or from_failure_domain selector"
        )));
    }
    if overlay.to_node_ids.is_empty()
        && overlay.to_node_groups.is_empty()
        && overlay.to_node_tags.is_empty()
        && overlay.to_racks.is_empty()
        && overlay.to_islands.is_empty()
        && overlay.to_failure_domains.is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one to, to_nodes, to_group, to_groups, to_node_tag, to_rack, to_island, or to_failure_domain selector"
        )));
    }
    Ok(())
}

pub(super) fn validate_run_scenario_node_overlay_selectors(
    name: &str,
    overlay: &RunScenarioNodeStateOverlay,
) -> Result<(), ConfigError> {
    if overlay.node_ids.is_empty()
        && overlay.node_groups.is_empty()
        && overlay.node_tags.is_empty()
        && overlay.racks.is_empty()
        && overlay.islands.is_empty()
        && overlay.failure_domains.is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one node, nodes, group, groups, node_tag, rack, island, or failure_domain selector"
        )));
    }
    Ok(())
}

pub(super) fn parse_run_scenario_link_overlay_rails(
    name: &str,
    rail: Option<u32>,
    rails: Option<&[u32]>,
) -> Result<Vec<u32>, ConfigError> {
    let mut ids = Vec::new();
    if let Some(rail) = rail {
        ids.push(rail);
    }
    if let Some(rails) = rails {
        ids.extend_from_slice(rails);
    }
    sort_dedup_or_error(&mut ids, &format!("{name}.rails"))?;
    Ok(ids)
}

pub(super) fn parse_run_scenario_overlay_node_ids(
    name: &str,
    node: Option<u32>,
    nodes: Option<&[u32]>,
) -> Result<Vec<u32>, ConfigError> {
    let mut ids = Vec::new();
    if let Some(node) = node {
        ids.push(node);
    }
    if let Some(nodes) = nodes {
        ids.extend_from_slice(nodes);
    }
    sort_dedup_or_error(&mut ids, &format!("{name}.nodes"))?;
    Ok(ids)
}

pub(super) fn parse_run_scenario_overlay_node_groups(
    name: &str,
    group: Option<&str>,
    groups: Option<&[String]>,
) -> Result<Vec<String>, ConfigError> {
    let mut labels = Vec::new();
    if let Some(group) = group {
        labels.push(parse_nonempty_overlay_label(
            &format!("{name}.group"),
            group,
        )?);
    }
    if let Some(groups) = groups {
        for (idx, group) in groups.iter().enumerate() {
            labels.push(parse_nonempty_overlay_label(
                &format!("{name}.groups[{idx}]"),
                group,
            )?);
        }
    }
    labels.sort();
    for pair in labels.windows(2) {
        if pair[0] == pair[1] {
            return Err(ConfigError::new(format!(
                "{name}.groups duplicates node group '{}'",
                pair[0]
            )));
        }
    }
    Ok(labels)
}

pub(super) fn parse_nonempty_overlay_label(name: &str, value: &str) -> Result<String, ConfigError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }
    Ok(value.to_string())
}

pub(super) fn parse_run_scenario_overlay_resource_ids(
    name: &str,
    resource_label: &str,
    resource: Option<u32>,
    resources: Option<&[u32]>,
) -> Result<Vec<u32>, ConfigError> {
    let mut ids = Vec::new();
    if let Some(resource) = resource {
        ids.push(resource);
    }
    if let Some(resources) = resources {
        ids.extend_from_slice(resources);
    }
    sort_dedup_or_error(&mut ids, &format!("{name}.{resource_label}s"))?;
    Ok(ids)
}

pub(super) fn sort_dedup_or_error(ids: &mut [u32], name: &str) -> Result<(), ConfigError> {
    ids.sort_unstable();
    for pair in ids.windows(2) {
        if pair[0] == pair[1] {
            return Err(ConfigError::new(format!(
                "{name} duplicates id {}",
                pair[0]
            )));
        }
    }
    Ok(())
}

pub(super) fn calibration_overrides_from_section(
    section: Option<&CalibrationSection>,
) -> RunScenarioCalibrationConfig {
    let Some(section) = section else {
        return RunScenarioCalibrationConfig::default();
    };

    RunScenarioCalibrationConfig {
        compute_efficiency: section.compute_efficiency,
        prefill_compute_scale: section.prefill_compute_scale,
        decode_compute_scale: section.decode_compute_scale,
        decode_memory_bandwidth_scale: section.decode_memory_bandwidth_scale,
        collective_latency_scale: section.collective_latency_scale,
        collective_bandwidth_scale: section.collective_bandwidth_scale,
        kv_transfer_scale: section.kv_transfer_scale,
        scheduler_overhead_us: section.scheduler_overhead_us,
        serving_memory_temporary_fraction: section.serving_memory_temporary_fraction,
        serving_memory_activation_communication_fraction: section
            .serving_memory_activation_communication_fraction,
        serving_memory_weight_communication_fraction: section
            .serving_memory_weight_communication_fraction,
        serving_memory_runtime_reserve_fraction: section.serving_memory_runtime_reserve_fraction,
        serving_memory_fragmentation_fraction: section.serving_memory_fragmentation_fraction,
        serving_pipeline_depth: section.serving_pipeline_depth,
        request_arrival_gap_s: section.request_arrival_gap_s,
        allow_compute_comm_overlap: section.allow_compute_comm_overlap,
    }
}
