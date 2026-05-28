use super::*;

pub(super) fn parse_serving(
    serving: ServingSection,
    default_search: &SearchSpace,
    base_dir: Option<&Path>,
) -> Result<DisaggregatedServingConfig, ConfigError> {
    let deployment_mode = parse_serving_deployment_mode(serving.mode.as_deref())?;
    let slo_miss_penalty_weights = parse_serving_slo_miss_penalty_weights(&serving)?;
    let objective = parse_serving_objective(serving.objective.as_deref())?;
    let metric_ceilings = parse_serving_metric_ceilings(&serving)?;
    let kv_route_constraints = parse_serving_kv_route_constraints(&serving)?;
    let cost_model = parse_serving_cost_model(serving.cost.as_ref())?;
    let pool_candidates = parse_pool_candidates(&serving)?;
    let pool_search = parse_pool_search(serving.pool_search, deployment_mode)?;
    let traffic_classes = parse_traffic_classes(serving.traffic_classes)?;
    let mut slo_policies = parse_slo_policies(serving.slo_policies)?;
    slo_policies.extend(traffic_class_slo_policies(&traffic_classes));
    if pool_candidates.is_empty() && pool_search.is_none() {
        return Err(ConfigError::new(
            "serving requires prefill/decode nodes, [[serving.pool_candidates]], or [serving.pool_search]",
        ));
    }

    let prefill = match serving.prefill_search {
        Some(search) => parse_search("serving.prefill_search", search)?,
        None => default_search.clone(),
    };
    let decode = match serving.decode_search {
        Some(search) => parse_search("serving.decode_search", search)?,
        None => default_search.clone(),
    };
    let prefill_nodes = pool_candidates
        .first()
        .map(|candidate| candidate.prefill_nodes.clone())
        .unwrap_or_default();
    let decode_nodes = pool_candidates
        .first()
        .map(|candidate| candidate.decode_nodes.clone())
        .unwrap_or_default();
    validate_positive_optional_u32("serving.max_unique_gpus", serving.max_unique_gpus)?;
    let min_throughput_tokens_per_s = parse_non_negative_f64(
        "serving.min_throughput_tokens_per_s",
        serving.min_throughput_tokens_per_s,
    )?;
    Ok(DisaggregatedServingConfig {
        deployment_mode,
        prefill_nodes,
        decode_nodes,
        pool_candidates,
        pool_search,
        objective,
        slo_miss_penalty_weight: slo_miss_penalty_weights.aggregate,
        slo_miss_penalty_weights,
        topology_risk_penalty_weight: parse_non_negative_f64(
            "serving.topology_risk_penalty_weight",
            serving.topology_risk_penalty_weight,
        )?
        .unwrap_or(0.0),
        max_memory_pressure_fraction: parse_fraction(
            "serving.max_memory_pressure_fraction",
            serving.max_memory_pressure_fraction,
        )?,
        max_unique_gpus: serving.max_unique_gpus,
        min_throughput_tokens_per_s,
        cost_model,
        search: ServingSearchSpace { prefill, decode },
        traffic: parse_serving_traffic(
            serving.traffic,
            base_dir,
            traffic_classes,
            ServingServiceSections {
                services: serving.services,
                prefill: serving.prefill_service,
                decode: serving.decode_service,
                kv_transfer: serving.kv_transfer_service,
            },
            metric_ceilings,
            kv_route_constraints,
        )?,
        slo_policies,
    })
}

pub(super) fn parse_serving_runtime_features(
    file: &WorkloadFile,
) -> Result<Vec<String>, ConfigError> {
    let mut features = Vec::new();
    if let Some(values) = file.serving_runtime_features.clone() {
        for feature in normalized_group_labels("serving_runtime_features", values)? {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    if let Some(values) = file
        .serving
        .as_ref()
        .and_then(|serving| serving.serving_runtime_features.clone())
    {
        for feature in normalized_group_labels("serving.runtime_features", values)? {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    Ok(features)
}

pub(super) fn parse_serving_metric_ceilings(
    serving: &ServingSection,
) -> Result<ServingMetricCeilings, ConfigError> {
    Ok(ServingMetricCeilings {
        max_ttft_s: parse_positive_optional_seconds(
            "serving.max_ttft_s",
            serving.max_ttft_s,
            "serving.max_ttft_ms",
            serving.max_ttft_ms,
        )?,
        max_tpot_s: parse_positive_optional_seconds(
            "serving.max_tpot_s",
            serving.max_tpot_s,
            "serving.max_tpot_ms",
            serving.max_tpot_ms,
        )?,
        max_itl_s: parse_positive_optional_seconds(
            "serving.max_itl_s",
            serving.max_itl_s,
            "serving.max_itl_ms",
            serving.max_itl_ms,
        )?,
        max_e2el_s: parse_positive_optional_seconds(
            "serving.max_e2el_s",
            serving.max_e2el_s,
            "serving.max_e2el_ms",
            serving.max_e2el_ms,
        )?,
    })
}

pub(super) fn parse_serving_kv_route_constraints(
    serving: &ServingSection,
) -> Result<ServingKvRouteConstraints, ConfigError> {
    validate_positive_optional_u32(
        "serving.min_kv_route_rail_count",
        serving.min_kv_route_rail_count,
    )?;
    Ok(ServingKvRouteConstraints {
        min_inter_node_rail_count: serving.min_kv_route_rail_count,
        require_inter_node_rail_metadata: serving.require_kv_route_rail_metadata.unwrap_or(false),
        require_gpudirect: serving.require_gpudirect_kv_paths.unwrap_or(false),
    })
}

pub(super) fn parse_serving_cost_model(
    section: Option<&ServingCostSection>,
) -> Result<ServingCostModel, ConfigError> {
    let Some(section) = section else {
        return Ok(ServingCostModel::default());
    };
    let default_gpu_hour_usd = parse_non_negative_f64(
        "serving.cost.default_gpu_hour_usd",
        section.default_gpu_hour_usd,
    )?;
    let node_hour_usd =
        parse_non_negative_f64("serving.cost.node_hour_usd", section.node_hour_usd)?;
    let kwh_usd = parse_non_negative_f64("serving.cost.kwh_usd", section.kwh_usd)?;
    let default_gpu_watts =
        parse_non_negative_f64("serving.cost.default_gpu_watts", section.default_gpu_watts)?;
    let node_watts = parse_non_negative_f64("serving.cost.node_watts", section.node_watts)?;
    let mut gpu_rates = Vec::new();
    for (idx, rate) in section
        .gpu_rates
        .as_deref()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let gpu_label = optional_nonempty_string(rate.gpu_label.clone()).ok_or_else(|| {
            ConfigError::new(format!(
                "serving.cost.gpu_rates[{idx}].gpu_label is required"
            ))
        })?;
        gpu_rates.push(ServingGpuCostRate {
            gpu_label,
            gpu_hour_usd: parse_non_negative_f64(
                &format!("serving.cost.gpu_rates[{idx}].gpu_hour_usd"),
                rate.gpu_hour_usd,
            )?,
            watts: parse_non_negative_f64(
                &format!("serving.cost.gpu_rates[{idx}].watts"),
                rate.watts,
            )?,
        });
    }
    Ok(ServingCostModel {
        default_gpu_hour_usd,
        node_hour_usd,
        kwh_usd,
        default_gpu_watts,
        node_watts,
        gpu_rates,
    })
}

pub(super) fn parse_serving_slo_miss_penalty_weights(
    serving: &ServingSection,
) -> Result<ServingSloMissPenaltyWeights, ConfigError> {
    Ok(ServingSloMissPenaltyWeights {
        aggregate: parse_non_negative_f64(
            "serving.slo_miss_penalty_weight",
            serving.slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        ttft: parse_non_negative_f64(
            "serving.ttft_slo_miss_penalty_weight",
            serving.ttft_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        tpot: parse_non_negative_f64(
            "serving.tpot_slo_miss_penalty_weight",
            serving.tpot_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        itl: parse_non_negative_f64(
            "serving.itl_slo_miss_penalty_weight",
            serving.itl_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        e2el: parse_non_negative_f64(
            "serving.e2el_slo_miss_penalty_weight",
            serving.e2el_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        deadline: parse_non_negative_f64(
            "serving.deadline_miss_penalty_weight",
            serving.deadline_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
    })
}

pub(super) fn parse_serving_objective(
    objective: Option<&str>,
) -> Result<ServingObjective, ConfigError> {
    match objective.map(normalize).as_deref() {
        None
        | Some("e2el")
        | Some("end_to_end")
        | Some("end_to_end_latency")
        | Some("latency")
        | Some("minimize_e2el") => Ok(ServingObjective::MinimizeE2el),
        Some("ttft") | Some("minimize_ttft") => Ok(ServingObjective::MinimizeTtft),
        Some("tpot") | Some("minimize_tpot") => Ok(ServingObjective::MinimizeTpot),
        Some("throughput") | Some("max_throughput") | Some("maximize_throughput") => {
            Ok(ServingObjective::MaximizeThroughput)
        }
        Some("slo") | Some("slo_miss") | Some("slo_miss_rate") | Some("minimize_slo_miss_rate") => {
            Ok(ServingObjective::MinimizeSloMissRate)
        }
        Some("memory")
        | Some("hbm")
        | Some("memory_pressure")
        | Some("hbm_pressure")
        | Some("minimize_memory_pressure")
        | Some("minimize_hbm_pressure") => Ok(ServingObjective::MinimizeMemoryPressure),
        Some("cost")
        | Some("total_cost")
        | Some("cost_usd")
        | Some("minimize_cost")
        | Some("minimize_total_cost")
        | Some("minimize_cost_usd") => Ok(ServingObjective::MinimizeCost),
        Some("energy")
        | Some("kwh")
        | Some("energy_kwh")
        | Some("minimize_energy")
        | Some("minimize_energy_kwh") => Ok(ServingObjective::MinimizeEnergy),
        Some("power")
        | Some("watts")
        | Some("average_power")
        | Some("average_power_watts")
        | Some("minimize_power")
        | Some("minimize_average_power")
        | Some("minimize_average_power_watts") => Ok(ServingObjective::MinimizePower),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.objective '{value}'; use e2el, ttft, tpot, throughput, slo_miss_rate, memory_pressure, cost, energy, or power"
        ))),
    }
}

pub(super) fn parse_serving_deployment_mode(
    mode: Option<&str>,
) -> Result<ServingDeploymentMode, ConfigError> {
    match mode.map(normalize).as_deref() {
        None | Some("flexible") | Some("any") | Some("auto") => Ok(ServingDeploymentMode::Flexible),
        Some("colocated") | Some("co_located") | Some("collocated") => {
            Ok(ServingDeploymentMode::Colocated)
        }
        Some("partial") | Some("partially_disaggregated") | Some("partial_disaggregated") => {
            Ok(ServingDeploymentMode::PartiallyDisaggregated)
        }
        Some("disaggregated")
        | Some("full")
        | Some("fully_disaggregated")
        | Some("full_disaggregated") => Ok(ServingDeploymentMode::FullyDisaggregated),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.mode '{value}'; use flexible, colocated, partially_disaggregated, fully_disaggregated, or disaggregated"
        ))),
    }
}

pub(super) fn parse_pool_candidates(
    serving: &ServingSection,
) -> Result<Vec<ServingPoolCandidate>, ConfigError> {
    let mut candidates = Vec::new();
    let has_top_level_pool = serving.prefill_nodes.is_some()
        || serving.decode_nodes.is_some()
        || serving.prefill_groups.is_some()
        || serving.decode_groups.is_some();
    if has_top_level_pool {
        candidates.push(parse_pool_candidate(
            "serving",
            PoolCandidateInput {
                label: None,
                prefill_nodes: serving.prefill_nodes.clone().unwrap_or_default(),
                decode_nodes: serving.decode_nodes.clone().unwrap_or_default(),
                prefill_groups: serving.prefill_groups.clone().unwrap_or_default(),
                decode_groups: serving.decode_groups.clone().unwrap_or_default(),
                prefill_node_filter: ServingPoolNodeFilter::default(),
                decode_node_filter: ServingPoolNodeFilter::default(),
                domain_spread: ServingPoolDomainSpread::default(),
                prefill_gpu_tag: serving.prefill_gpu_tag.clone(),
                prefill_gpu_tags: serving.prefill_gpu_tags.clone(),
                decode_gpu_tag: serving.decode_gpu_tag.clone(),
                decode_gpu_tags: serving.decode_gpu_tags.clone(),
            },
        )?);
    }

    for (idx, candidate) in serving
        .pool_candidates
        .as_ref()
        .into_iter()
        .flatten()
        .enumerate()
    {
        candidates.push(parse_pool_candidate(
            &format!("serving.pool_candidates[{idx}]"),
            PoolCandidateInput {
                label: candidate.label.clone(),
                prefill_nodes: candidate.prefill_nodes.clone().unwrap_or_default(),
                decode_nodes: candidate.decode_nodes.clone().unwrap_or_default(),
                prefill_groups: candidate.prefill_groups.clone().unwrap_or_default(),
                decode_groups: candidate.decode_groups.clone().unwrap_or_default(),
                prefill_node_filter: parse_serving_pool_candidate_node_filter(
                    &format!("serving.pool_candidates[{idx}].prefill"),
                    candidate.prefill_node_tag.as_deref(),
                    candidate.prefill_node_tags.as_deref(),
                    candidate.prefill_rack.as_deref(),
                    candidate.prefill_racks.as_deref(),
                    candidate.prefill_island.as_deref(),
                    candidate.prefill_islands.as_deref(),
                    candidate.prefill_failure_domain.as_deref(),
                    candidate.prefill_failure_domains.as_deref(),
                    candidate.prefill_exclude_node_tag.as_deref(),
                    candidate.prefill_exclude_node_tags.as_deref(),
                    candidate.prefill_exclude_rack.as_deref(),
                    candidate.prefill_exclude_racks.as_deref(),
                    candidate.prefill_exclude_island.as_deref(),
                    candidate.prefill_exclude_islands.as_deref(),
                    candidate.prefill_exclude_failure_domain.as_deref(),
                    candidate.prefill_exclude_failure_domains.as_deref(),
                )?,
                decode_node_filter: parse_serving_pool_candidate_node_filter(
                    &format!("serving.pool_candidates[{idx}].decode"),
                    candidate.decode_node_tag.as_deref(),
                    candidate.decode_node_tags.as_deref(),
                    candidate.decode_rack.as_deref(),
                    candidate.decode_racks.as_deref(),
                    candidate.decode_island.as_deref(),
                    candidate.decode_islands.as_deref(),
                    candidate.decode_failure_domain.as_deref(),
                    candidate.decode_failure_domains.as_deref(),
                    candidate.decode_exclude_node_tag.as_deref(),
                    candidate.decode_exclude_node_tags.as_deref(),
                    candidate.decode_exclude_rack.as_deref(),
                    candidate.decode_exclude_racks.as_deref(),
                    candidate.decode_exclude_island.as_deref(),
                    candidate.decode_exclude_islands.as_deref(),
                    candidate.decode_exclude_failure_domain.as_deref(),
                    candidate.decode_exclude_failure_domains.as_deref(),
                )?,
                domain_spread: parse_serving_pool_candidate_domain_spread(
                    &format!("serving.pool_candidates[{idx}]"),
                    candidate,
                )?,
                prefill_gpu_tag: candidate.prefill_gpu_tag.clone(),
                prefill_gpu_tags: candidate.prefill_gpu_tags.clone(),
                decode_gpu_tag: candidate.decode_gpu_tag.clone(),
                decode_gpu_tags: candidate.decode_gpu_tags.clone(),
            },
        )?);
    }

    Ok(dedup_pool_candidates(candidates))
}

pub(super) fn parse_pool_search(
    pool_search: Option<ServingPoolSearchSection>,
    deployment_mode: ServingDeploymentMode,
) -> Result<Option<ServingPoolSearch>, ConfigError> {
    let Some(pool_search) = pool_search else {
        return Ok(None);
    };
    let max_candidates = pool_search.max_candidates.unwrap_or(64);
    if max_candidates == 0 {
        return Err(ConfigError::new(
            "serving.pool_search.max_candidates must be greater than zero",
        ));
    }
    let prefill_node_filter = parse_pool_search_node_filter(
        "serving.pool_search.prefill",
        pool_search.prefill_node_tag.as_deref(),
        pool_search.prefill_node_tags.as_deref(),
        pool_search.prefill_rack.as_deref(),
        pool_search.prefill_racks.as_deref(),
        pool_search.prefill_island.as_deref(),
        pool_search.prefill_islands.as_deref(),
        pool_search.prefill_failure_domain.as_deref(),
        pool_search.prefill_failure_domains.as_deref(),
        pool_search.prefill_exclude_node_tag.as_deref(),
        pool_search.prefill_exclude_node_tags.as_deref(),
        pool_search.prefill_exclude_rack.as_deref(),
        pool_search.prefill_exclude_racks.as_deref(),
        pool_search.prefill_exclude_island.as_deref(),
        pool_search.prefill_exclude_islands.as_deref(),
        pool_search.prefill_exclude_failure_domain.as_deref(),
        pool_search.prefill_exclude_failure_domains.as_deref(),
    )?;
    let decode_node_filter = parse_pool_search_node_filter(
        "serving.pool_search.decode",
        pool_search.decode_node_tag.as_deref(),
        pool_search.decode_node_tags.as_deref(),
        pool_search.decode_rack.as_deref(),
        pool_search.decode_racks.as_deref(),
        pool_search.decode_island.as_deref(),
        pool_search.decode_islands.as_deref(),
        pool_search.decode_failure_domain.as_deref(),
        pool_search.decode_failure_domains.as_deref(),
        pool_search.decode_exclude_node_tag.as_deref(),
        pool_search.decode_exclude_node_tags.as_deref(),
        pool_search.decode_exclude_rack.as_deref(),
        pool_search.decode_exclude_racks.as_deref(),
        pool_search.decode_exclude_island.as_deref(),
        pool_search.decode_exclude_islands.as_deref(),
        pool_search.decode_exclude_failure_domain.as_deref(),
        pool_search.decode_exclude_failure_domains.as_deref(),
    )?;
    let domain_spread = parse_pool_search_domain_spread(&pool_search)?;

    Ok(Some(ServingPoolSearch {
        prefill_groups: normalized_group_labels(
            "serving.pool_search.prefill_groups",
            require_nonempty(
                "serving.pool_search.prefill_groups",
                pool_search.prefill_groups,
            )?,
        )?,
        decode_groups: normalized_group_labels(
            "serving.pool_search.decode_groups",
            require_nonempty(
                "serving.pool_search.decode_groups",
                pool_search.decode_groups,
            )?,
        )?,
        prefill_node_counts: positive_values(
            "serving.pool_search.prefill_node_counts",
            pool_search.prefill_node_counts.unwrap_or_default(),
        )?,
        decode_node_counts: positive_values(
            "serving.pool_search.decode_node_counts",
            pool_search.decode_node_counts.unwrap_or_default(),
        )?,
        prefill_node_filter,
        decode_node_filter,
        prefill_gpu_labels: parse_gpu_labels(
            "serving.pool_search.prefill_gpu_tags",
            pool_search.prefill_gpu_tag.as_deref(),
            pool_search.prefill_gpu_tags.as_deref(),
        )?
        .into_iter()
        .collect(),
        decode_gpu_labels: parse_gpu_labels(
            "serving.pool_search.decode_gpu_tags",
            pool_search.decode_gpu_tag.as_deref(),
            pool_search.decode_gpu_tags.as_deref(),
        )?
        .into_iter()
        .collect(),
        allow_overlap: pool_search.allow_overlap.unwrap_or(matches!(
            deployment_mode,
            ServingDeploymentMode::Colocated | ServingDeploymentMode::PartiallyDisaggregated
        )),
        domain_spread,
        max_candidates,
    }))
}

pub(super) fn parse_serving_pool_candidate_domain_spread(
    name: &str,
    candidate: &ServingPoolCandidateSection,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    parse_pool_domain_spread(
        name,
        candidate.min_prefill_racks,
        candidate.min_decode_racks,
        candidate.min_prefill_islands,
        candidate.min_decode_islands,
        candidate.min_prefill_failure_domains,
        candidate.min_decode_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_serving_pool_candidate_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    parse_pool_node_filter(
        name,
        node_tag,
        node_tags,
        rack,
        racks,
        island,
        islands,
        failure_domain,
        failure_domains,
        exclude_node_tag,
        exclude_node_tags,
        exclude_rack,
        exclude_racks,
        exclude_island,
        exclude_islands,
        exclude_failure_domain,
        exclude_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_pool_search_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    parse_pool_node_filter(
        name,
        node_tag,
        node_tags,
        rack,
        racks,
        island,
        islands,
        failure_domain,
        failure_domains,
        exclude_node_tag,
        exclude_node_tags,
        exclude_rack,
        exclude_racks,
        exclude_island,
        exclude_islands,
        exclude_failure_domain,
        exclude_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_pool_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    Ok(ServingPoolNodeFilter {
        node_labels: parse_optional_normalized_labels(
            &format!("{name}_node_tags"),
            node_tag,
            node_tags,
        )?,
        racks: parse_optional_topology_domains(&format!("{name}_racks"), rack, racks)?,
        islands: parse_optional_topology_domains(&format!("{name}_islands"), island, islands)?,
        failure_domains: parse_optional_topology_domains(
            &format!("{name}_failure_domains"),
            failure_domain,
            failure_domains,
        )?,
        exclude_node_labels: parse_optional_normalized_labels(
            &format!("{name}_exclude_node_tags"),
            exclude_node_tag,
            exclude_node_tags,
        )?,
        exclude_racks: parse_optional_topology_domains(
            &format!("{name}_exclude_racks"),
            exclude_rack,
            exclude_racks,
        )?,
        exclude_islands: parse_optional_topology_domains(
            &format!("{name}_exclude_islands"),
            exclude_island,
            exclude_islands,
        )?,
        exclude_failure_domains: parse_optional_topology_domains(
            &format!("{name}_exclude_failure_domains"),
            exclude_failure_domain,
            exclude_failure_domains,
        )?,
    })
}

pub(super) fn parse_pool_search_domain_spread(
    pool_search: &ServingPoolSearchSection,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    parse_pool_domain_spread(
        "serving.pool_search",
        pool_search.min_prefill_racks,
        pool_search.min_decode_racks,
        pool_search.min_prefill_islands,
        pool_search.min_decode_islands,
        pool_search.min_prefill_failure_domains,
        pool_search.min_decode_failure_domains,
    )
}

pub(super) fn parse_pool_domain_spread(
    name: &str,
    min_prefill_racks: Option<u32>,
    min_decode_racks: Option<u32>,
    min_prefill_islands: Option<u32>,
    min_decode_islands: Option<u32>,
    min_prefill_failure_domains: Option<u32>,
    min_decode_failure_domains: Option<u32>,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    validate_positive_optional_u32(&format!("{name}.min_prefill_racks"), min_prefill_racks)?;
    validate_positive_optional_u32(&format!("{name}.min_decode_racks"), min_decode_racks)?;
    validate_positive_optional_u32(&format!("{name}.min_prefill_islands"), min_prefill_islands)?;
    validate_positive_optional_u32(&format!("{name}.min_decode_islands"), min_decode_islands)?;
    validate_positive_optional_u32(
        &format!("{name}.min_prefill_failure_domains"),
        min_prefill_failure_domains,
    )?;
    validate_positive_optional_u32(
        &format!("{name}.min_decode_failure_domains"),
        min_decode_failure_domains,
    )?;
    Ok(ServingPoolDomainSpread {
        min_prefill_racks,
        min_decode_racks,
        min_prefill_islands,
        min_decode_islands,
        min_prefill_failure_domains,
        min_decode_failure_domains,
    })
}

pub(super) fn parse_slo_policies(
    policies: Option<Vec<ServingSloPolicySection>>,
) -> Result<Vec<ServingSloPolicy>, ConfigError> {
    let mut parsed = Vec::new();
    for (idx, policy) in policies.unwrap_or_default().into_iter().enumerate() {
        let name = format!("serving.slo_policies[{idx}]");
        let group = parse_slo_policy_group(&format!("{name}.group"), &policy.group)?;
        let key = policy.key.trim().to_string();
        if key.is_empty() {
            return Err(ConfigError::new(format!("{name}.key must not be empty")));
        }
        let parsed_policy = ServingSloPolicy {
            group,
            key,
            max_ttft_slo_miss_rate: parse_fraction(
                &format!("{name}.max_ttft_slo_miss_rate"),
                policy.max_ttft_slo_miss_rate,
            )?,
            max_tpot_slo_miss_rate: parse_fraction(
                &format!("{name}.max_tpot_slo_miss_rate"),
                policy.max_tpot_slo_miss_rate,
            )?,
            max_itl_slo_miss_rate: parse_fraction(
                &format!("{name}.max_itl_slo_miss_rate"),
                policy.max_itl_slo_miss_rate,
            )?,
            max_e2el_slo_miss_rate: parse_fraction(
                &format!("{name}.max_e2el_slo_miss_rate"),
                policy.max_e2el_slo_miss_rate,
            )?,
            max_deadline_miss_rate: parse_fraction(
                &format!("{name}.max_deadline_miss_rate"),
                policy.max_deadline_miss_rate,
            )?,
        };
        if parsed_policy.max_ttft_slo_miss_rate.is_none()
            && parsed_policy.max_tpot_slo_miss_rate.is_none()
            && parsed_policy.max_itl_slo_miss_rate.is_none()
            && parsed_policy.max_e2el_slo_miss_rate.is_none()
            && parsed_policy.max_deadline_miss_rate.is_none()
        {
            return Err(ConfigError::new(format!(
                "{name} must set at least one max_*_miss_rate field"
            )));
        }
        parsed.push(parsed_policy);
    }
    Ok(parsed)
}

pub(super) fn parse_traffic_classes(
    classes: Option<Vec<ServingTrafficClassSection>>,
) -> Result<Vec<ServingTrafficClass>, ConfigError> {
    let mut parsed = Vec::new();
    let mut names = HashSet::new();
    let mut selectors = BTreeMap::new();
    for (idx, class) in classes.unwrap_or_default().into_iter().enumerate() {
        let name = format!("serving.traffic_classes[{idx}]");
        let class_name = class.name.trim().to_string();
        if class_name.is_empty() {
            return Err(ConfigError::new(format!("{name}.name must not be empty")));
        }
        if !names.insert(class_name.clone()) {
            return Err(ConfigError::new(format!(
                "{name}.name '{class_name}' is duplicated"
            )));
        }
        let group = parse_traffic_class_group(&format!("{name}.group"), &class.group)?;
        let slo_miss_penalty_weights = parse_traffic_class_slo_miss_penalty_weights(&name, &class)?;
        let key = parse_traffic_class_key(&name, &group, class.key, class.priority)?;
        if let Some(first_name) = selectors.insert((group.clone(), key.clone()), class_name.clone())
        {
            return Err(ConfigError::new(format!(
                "{name} selector group='{group}' key='{key}' duplicates traffic class '{first_name}'"
            )));
        }
        if let Some(0) = class.max_prefill_tokens {
            return Err(ConfigError::new(format!(
                "{name}.max_prefill_tokens must be greater than zero"
            )));
        }
        if let Some(0) = class.max_decode_sequences {
            return Err(ConfigError::new(format!(
                "{name}.max_decode_sequences must be greater than zero"
            )));
        }
        if let Some(0) = class.max_resident_tokens {
            return Err(ConfigError::new(format!(
                "{name}.max_resident_tokens must be greater than zero"
            )));
        }
        if let Some(0) = class.max_kv_blocks {
            return Err(ConfigError::new(format!(
                "{name}.max_kv_blocks must be greater than zero"
            )));
        }
        let parsed_class = ServingTrafficClass {
            name: class_name,
            group,
            key,
            admission_priority: class.admission_priority,
            max_prefill_tokens: class.max_prefill_tokens,
            max_decode_sequences: class.max_decode_sequences,
            max_resident_tokens: class.max_resident_tokens,
            max_kv_blocks: class.max_kv_blocks,
            slo: ServingRequestSlo {
                ttft_s: parse_slo_ms(&format!("{name}.ttft_slo_ms"), class.ttft_slo_ms)?,
                tpot_s: parse_slo_ms(&format!("{name}.tpot_slo_ms"), class.tpot_slo_ms)?,
                itl_s: parse_slo_ms(&format!("{name}.itl_slo_ms"), class.itl_slo_ms)?,
                e2el_s: parse_slo_ms(&format!("{name}.e2el_slo_ms"), class.e2el_slo_ms)?,
            },
            slo_miss_penalty_weights,
            max_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_queue_delay_s"),
                class.max_queue_delay_s,
                &format!("{name}.max_queue_delay_ms"),
                class.max_queue_delay_ms,
            )?,
            max_kv_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_kv_queue_delay_s"),
                class.max_kv_queue_delay_s,
                &format!("{name}.max_kv_queue_delay_ms"),
                class.max_kv_queue_delay_ms,
            )?,
            max_decode_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_decode_queue_delay_s"),
                class.max_decode_queue_delay_s,
                &format!("{name}.max_decode_queue_delay_ms"),
                class.max_decode_queue_delay_ms,
            )?,
            max_decode_iteration_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_decode_iteration_queue_delay_s"),
                class.max_decode_iteration_queue_delay_s,
                &format!("{name}.max_decode_iteration_queue_delay_ms"),
                class.max_decode_iteration_queue_delay_ms,
            )?,
            request_timeout_s: parse_positive_optional_seconds(
                &format!("{name}.request_timeout_s"),
                class.request_timeout_s,
                &format!("{name}.request_timeout_ms"),
                class.request_timeout_ms,
            )?,
            max_ttft_slo_miss_rate: parse_fraction(
                &format!("{name}.max_ttft_slo_miss_rate"),
                class.max_ttft_slo_miss_rate,
            )?,
            max_tpot_slo_miss_rate: parse_fraction(
                &format!("{name}.max_tpot_slo_miss_rate"),
                class.max_tpot_slo_miss_rate,
            )?,
            max_itl_slo_miss_rate: parse_fraction(
                &format!("{name}.max_itl_slo_miss_rate"),
                class.max_itl_slo_miss_rate,
            )?,
            max_e2el_slo_miss_rate: parse_fraction(
                &format!("{name}.max_e2el_slo_miss_rate"),
                class.max_e2el_slo_miss_rate,
            )?,
            max_deadline_miss_rate: parse_fraction(
                &format!("{name}.max_deadline_miss_rate"),
                class.max_deadline_miss_rate,
            )?,
        };
        if parsed_class.slo == ServingRequestSlo::default()
            && parsed_class.max_queue_delay_s.is_none()
            && parsed_class.max_kv_queue_delay_s.is_none()
            && parsed_class.max_decode_queue_delay_s.is_none()
            && parsed_class.max_decode_iteration_queue_delay_s.is_none()
            && parsed_class.request_timeout_s.is_none()
            && parsed_class.max_ttft_slo_miss_rate.is_none()
            && parsed_class.max_tpot_slo_miss_rate.is_none()
            && parsed_class.max_itl_slo_miss_rate.is_none()
            && parsed_class.max_e2el_slo_miss_rate.is_none()
            && parsed_class.max_deadline_miss_rate.is_none()
            && !parsed_class.slo_miss_penalty_weights.any_nonzero()
            && parsed_class.admission_priority.is_none()
            && parsed_class.max_prefill_tokens.is_none()
            && parsed_class.max_decode_sequences.is_none()
            && parsed_class.max_resident_tokens.is_none()
            && parsed_class.max_kv_blocks.is_none()
        {
            return Err(ConfigError::new(format!(
                "{name} must set at least one admission_priority, class capacity limit, *_slo_ms, max_*_miss_rate, or *_penalty_weight field"
            )));
        }
        parsed.push(parsed_class);
    }
    Ok(parsed)
}

pub(super) fn parse_traffic_class_slo_miss_penalty_weights(
    name: &str,
    class: &ServingTrafficClassSection,
) -> Result<ServingSloMissPenaltyWeights, ConfigError> {
    Ok(ServingSloMissPenaltyWeights {
        aggregate: parse_non_negative_f64(
            &format!("{name}.slo_miss_penalty_weight"),
            class.slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        ttft: parse_non_negative_f64(
            &format!("{name}.ttft_slo_miss_penalty_weight"),
            class.ttft_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        tpot: parse_non_negative_f64(
            &format!("{name}.tpot_slo_miss_penalty_weight"),
            class.tpot_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        itl: parse_non_negative_f64(
            &format!("{name}.itl_slo_miss_penalty_weight"),
            class.itl_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        e2el: parse_non_negative_f64(
            &format!("{name}.e2el_slo_miss_penalty_weight"),
            class.e2el_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        deadline: parse_non_negative_f64(
            &format!("{name}.deadline_miss_penalty_weight"),
            class.deadline_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
    })
}

pub(super) fn traffic_class_slo_policies(classes: &[ServingTrafficClass]) -> Vec<ServingSloPolicy> {
    classes
        .iter()
        .filter_map(|class| {
            let policy = ServingSloPolicy {
                group: class.group.clone(),
                key: class.key.clone(),
                max_ttft_slo_miss_rate: class.max_ttft_slo_miss_rate,
                max_tpot_slo_miss_rate: class.max_tpot_slo_miss_rate,
                max_itl_slo_miss_rate: class.max_itl_slo_miss_rate,
                max_e2el_slo_miss_rate: class.max_e2el_slo_miss_rate,
                max_deadline_miss_rate: class.max_deadline_miss_rate,
            };
            if policy.max_ttft_slo_miss_rate.is_none()
                && policy.max_tpot_slo_miss_rate.is_none()
                && policy.max_itl_slo_miss_rate.is_none()
                && policy.max_e2el_slo_miss_rate.is_none()
                && policy.max_deadline_miss_rate.is_none()
            {
                None
            } else {
                Some(policy)
            }
        })
        .collect()
}

pub(super) fn parse_traffic_class_group(name: &str, group: &str) -> Result<String, ConfigError> {
    match normalize(group).as_str() {
        "tenant" => Ok("tenant".to_string()),
        "model" | "model_id" => Ok("model_id".to_string()),
        "priority" => Ok("priority".to_string()),
        value => Err(ConfigError::new(format!(
            "unsupported {name} '{value}'; use tenant, model_id, or priority"
        ))),
    }
}

pub(super) fn parse_traffic_class_key(
    name: &str,
    group: &str,
    key: Option<String>,
    priority: Option<i32>,
) -> Result<String, ConfigError> {
    if group == "priority" {
        if let Some(priority) = priority {
            return Ok(format!("priority-{priority}"));
        }
        let key = key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .ok_or_else(|| {
                ConfigError::new(format!("{name}.key or {name}.priority is required"))
            })?;
        if key.starts_with("priority-") {
            return Ok(key.to_string());
        }
        if let Ok(priority) = key.parse::<i32>() {
            return Ok(format!("priority-{priority}"));
        }
        return Err(ConfigError::new(format!(
            "{name}.key for priority classes must be an integer or priority-<integer>"
        )));
    }
    if priority.is_some() {
        return Err(ConfigError::new(format!(
            "{name}.priority is only valid when group = \"priority\""
        )));
    }
    key.as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ConfigError::new(format!("{name}.key must not be empty")))
}

pub(super) fn parse_slo_policy_group(name: &str, group: &str) -> Result<String, ConfigError> {
    let normalized = normalize(group);
    match normalized.as_str() {
        "tenant" => Ok("tenant".to_string()),
        "model" | "model_id" => Ok("model_id".to_string()),
        "priority" => Ok("priority".to_string()),
        "prefill_node" => Ok("prefill_node".to_string()),
        "decode_node" => Ok("decode_node".to_string()),
        "prefill_route" => Ok("prefill_route".to_string()),
        "decode_route" => Ok("decode_route".to_string()),
        "" => Err(ConfigError::new(format!("{name} must not be empty"))),
        value => Err(ConfigError::new(format!(
            "unsupported {name} '{value}'; use tenant, model_id, priority, prefill_node, decode_node, prefill_route, or decode_route"
        ))),
    }
}

pub(super) struct PoolCandidateInput {
    label: Option<String>,
    prefill_nodes: Vec<u32>,
    decode_nodes: Vec<u32>,
    prefill_groups: Vec<String>,
    decode_groups: Vec<String>,
    prefill_node_filter: ServingPoolNodeFilter,
    decode_node_filter: ServingPoolNodeFilter,
    domain_spread: ServingPoolDomainSpread,
    prefill_gpu_tag: Option<String>,
    prefill_gpu_tags: Option<Vec<String>>,
    decode_gpu_tag: Option<String>,
    decode_gpu_tags: Option<Vec<String>>,
}

pub(super) fn parse_pool_candidate(
    name: &str,
    input: PoolCandidateInput,
) -> Result<ServingPoolCandidate, ConfigError> {
    let prefill_groups =
        normalized_group_labels(&format!("{name}.prefill_groups"), input.prefill_groups)?;
    let decode_groups =
        normalized_group_labels(&format!("{name}.decode_groups"), input.decode_groups)?;
    let prefill_gpu_labels = parse_gpu_labels(
        &format!("{name}.prefill_gpu_tags"),
        input.prefill_gpu_tag.as_deref(),
        input.prefill_gpu_tags.as_deref(),
    )?
    .into_iter()
    .collect();
    let decode_gpu_labels = parse_gpu_labels(
        &format!("{name}.decode_gpu_tags"),
        input.decode_gpu_tag.as_deref(),
        input.decode_gpu_tags.as_deref(),
    )?
    .into_iter()
    .collect();
    if input.prefill_nodes.is_empty() && prefill_groups.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} requires prefill_nodes or prefill_groups"
        )));
    }
    if input.decode_nodes.is_empty() && decode_groups.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} requires decode_nodes or decode_groups"
        )));
    }

    Ok(ServingPoolCandidate {
        label: input.label.filter(|label| !label.trim().is_empty()),
        prefill_nodes: input.prefill_nodes,
        decode_nodes: input.decode_nodes,
        prefill_groups,
        decode_groups,
        prefill_node_filter: input.prefill_node_filter,
        decode_node_filter: input.decode_node_filter,
        domain_spread: input.domain_spread,
        prefill_gpu_labels,
        decode_gpu_labels,
    })
}

pub(super) fn normalized_group_labels(
    name: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ConfigError> {
    let mut labels = Vec::new();
    for value in values {
        let label = normalize(&value);
        if label.is_empty() {
            return Err(ConfigError::new(format!("{name} values must not be empty")));
        }
        if !labels.contains(&label) {
            labels.push(label);
        }
    }

    Ok(labels)
}

pub(super) fn dedup_pool_candidates(
    candidates: Vec<ServingPoolCandidate>,
) -> Vec<ServingPoolCandidate> {
    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped.iter().any(|existing: &ServingPoolCandidate| {
            same_nodes(&existing.prefill_nodes, &candidate.prefill_nodes)
                && same_nodes(&existing.decode_nodes, &candidate.decode_nodes)
                && same_strings(&existing.prefill_groups, &candidate.prefill_groups)
                && same_strings(&existing.decode_groups, &candidate.decode_groups)
                && same_pool_node_filter(
                    &existing.prefill_node_filter,
                    &candidate.prefill_node_filter,
                )
                && same_pool_node_filter(
                    &existing.decode_node_filter,
                    &candidate.decode_node_filter,
                )
                && existing.domain_spread == candidate.domain_spread
                && same_strings(&existing.prefill_gpu_labels, &candidate.prefill_gpu_labels)
                && same_strings(&existing.decode_gpu_labels, &candidate.decode_gpu_labels)
        }) {
            deduped.push(candidate);
        }
    }

    deduped
}

pub(super) fn same_nodes(left: &[u32], right: &[u32]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

pub(super) fn same_strings(left: &[String], right: &[String]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort();
    right.sort();
    left == right
}

pub(super) fn same_pool_node_filter(
    left: &ServingPoolNodeFilter,
    right: &ServingPoolNodeFilter,
) -> bool {
    same_strings(&left.node_labels, &right.node_labels)
        && same_strings(&left.racks, &right.racks)
        && same_strings(&left.islands, &right.islands)
        && same_strings(&left.failure_domains, &right.failure_domains)
}

pub(super) fn validate_search_space_capacity(
    name: &str,
    search: &SearchSpace,
    available_gpus: u32,
) -> Result<(), ConfigError> {
    if search.tensor_ranks.iter().any(|&tensor| {
        search.pipeline_ranks.iter().any(|&pipeline| {
            search.expert_ranks.iter().any(|&expert| {
                search.data_ranks.iter().any(|&data| {
                    tensor
                        .saturating_mul(pipeline)
                        .saturating_mul(expert)
                        .saturating_mul(data)
                        <= available_gpus
                })
            })
        })
    }) {
        return Ok(());
    }

    Err(ConfigError::new(format!(
        "{name} has no rank combination that fits {available_gpus} available GPUs"
    )))
}

pub(super) fn validate_pool_candidate_for_cluster(
    name: &str,
    cluster: &Cluster,
    candidate: &ServingPoolCandidate,
    deployment_mode: ServingDeploymentMode,
    search: &ServingSearchSpace,
    require_routable_pools: bool,
    model_dtype: DType,
) -> Result<(), ConfigError> {
    validate_no_duplicate_nodes(&format!("{name}.prefill_nodes"), &candidate.prefill_nodes)?;
    validate_no_duplicate_nodes(&format!("{name}.decode_nodes"), &candidate.decode_nodes)?;
    let prefill_nodes = resolve_configured_pool_nodes(
        &format!("{name}.prefill"),
        cluster,
        &candidate.prefill_nodes,
        &candidate.prefill_groups,
    )?;
    let prefill_nodes =
        nodes_matching_pool_node_filter(cluster, &prefill_nodes, &candidate.prefill_node_filter);
    if prefill_nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name}.prefill node topology filters matched no nodes"
        )));
    }
    let decode_nodes = resolve_configured_pool_nodes(
        &format!("{name}.decode"),
        cluster,
        &candidate.decode_nodes,
        &candidate.decode_groups,
    )?;
    let decode_nodes =
        nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
    if decode_nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name}.decode node topology filters matched no nodes"
        )));
    }
    if !pool_nodes_satisfy_domain_spread(
        cluster,
        &candidate.domain_spread,
        &prefill_nodes,
        &decode_nodes,
    ) {
        return Err(ConfigError::new(format!(
            "{name} does not satisfy configured topology-domain spread constraints"
        )));
    }
    validate_serving_deployment_mode(name, deployment_mode, &prefill_nodes, &decode_nodes)?;
    validate_search_space_capacity(
        &format!(
            "{name}.prefill_search for model.dtype {}",
            model_dtype_label(model_dtype)
        ),
        &search.prefill,
        gpu_count_for_nodes_with_labels_and_dtype(
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            model_dtype,
        ),
    )?;
    validate_search_space_capacity(
        &format!(
            "{name}.decode_search for model.dtype {}",
            model_dtype_label(model_dtype)
        ),
        &search.decode,
        gpu_count_for_nodes_with_labels_and_dtype(
            cluster,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            model_dtype,
        ),
    )?;
    if require_routable_pools {
        validate_serving_pool_route(
            name,
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
        )?;
    }
    Ok(())
}

pub(super) fn validate_pool_search_for_cluster(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    search: &ServingSearchSpace,
    require_routable_pools: bool,
    model_dtype: DType,
) -> Result<(), ConfigError> {
    let mut any_candidate = false;
    let mut any_routable_candidate = false;
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;
        validate_search_space_capacity(
            &format!(
                "serving.pool_search.prefill group '{prefill_group}' for model.dtype {}",
                model_dtype_label(model_dtype)
            ),
            &search.prefill,
            gpu_count_for_nodes_with_labels_and_dtype(
                cluster,
                &prefill_nodes,
                &pool_search.prefill_gpu_labels,
                model_dtype,
            ),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;
            validate_search_space_capacity(
                &format!(
                    "serving.pool_search.decode group '{decode_group}' for model.dtype {}",
                    model_dtype_label(model_dtype)
                ),
                &search.decode,
                gpu_count_for_nodes_with_labels_and_dtype(
                    cluster,
                    &decode_nodes,
                    &pool_search.decode_gpu_labels,
                    model_dtype,
                ),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    if pool_search_can_generate_candidate(
                        cluster,
                        pool_search,
                        &prefill_nodes,
                        prefill_count as usize,
                        &decode_nodes,
                        decode_count as usize,
                        deployment_mode,
                    ) {
                        any_candidate = true;
                        if !require_routable_pools
                            || pool_search_can_generate_routable_candidate(
                                cluster,
                                pool_search,
                                &prefill_nodes,
                                &decode_nodes,
                                (prefill_count as usize, decode_count as usize),
                                (
                                    &pool_search.prefill_gpu_labels,
                                    &pool_search.decode_gpu_labels,
                                ),
                                deployment_mode,
                            )
                        {
                            any_routable_candidate = true;
                        }
                    }
                }
            }
        }
    }

    if !any_candidate {
        Err(ConfigError::new(
            "serving.pool_search cannot generate any prefill/decode pool candidate with the configured groups, counts, node topology filters, GPU labels, overlap policy, and topology-domain spread constraints",
        ))
    } else if require_routable_pools && !any_routable_candidate {
        Err(ConfigError::new(
            "serving.pool_search cannot generate any routable prefill/decode pool candidate with the configured groups, counts, overlap policy, and cluster interconnect; add missing links, choose connected pools, or disable serving.require_routable_pools for failure-scenario sweeps",
        ))
    } else {
        Ok(())
    }
}

pub(super) fn validate_explicit_serving_placements_for_configured_pools(
    cluster: &Cluster,
    serving: &DisaggregatedServingConfig,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    if prefill_placement.is_none() && decode_placement.is_none() {
        return Ok(());
    }

    if serving.pool_candidates.is_empty()
        && serving.pool_search.is_none()
        && !serving.prefill_nodes.is_empty()
        && !serving.decode_nodes.is_empty()
        && explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &serving.prefill_nodes,
            &[],
            &serving.decode_nodes,
            &[],
            prefill_placement,
            decode_placement,
        )
    {
        return Ok(());
    }

    for candidate in &serving.pool_candidates {
        let prefill_nodes = resolve_configured_pool_nodes(
            "serving.pool_candidates.prefill",
            cluster,
            &candidate.prefill_nodes,
            &candidate.prefill_groups,
        )?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &candidate.prefill_node_filter,
        );
        let decode_nodes = resolve_configured_pool_nodes(
            "serving.pool_candidates.decode",
            cluster,
            &candidate.decode_nodes,
            &candidate.decode_groups,
        )?;
        let decode_nodes =
            nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
        if pool_nodes_satisfy_domain_spread(
            cluster,
            &candidate.domain_spread,
            &prefill_nodes,
            &decode_nodes,
        ) && explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            prefill_placement,
            decode_placement,
        ) {
            return Ok(());
        }
    }

    if let Some(pool_search) = &serving.pool_search
        && pool_search_can_generate_explicit_placement_candidate(
            cluster,
            pool_search,
            serving.deployment_mode,
            prefill_placement,
            decode_placement,
        )?
    {
        return Ok(());
    }

    Err(ConfigError::new(
        "serving explicit prefill/decode placements do not fit any configured serving pool; ensure serving.prefill_placement ranks are inside prefill pools and serving.decode_placement ranks are inside decode pools, including GPU label filters, or remove explicit placements",
    ))
}

pub(super) fn validate_serving_route_constraints_for_configured_pools(
    cluster: &Cluster,
    serving: &DisaggregatedServingConfig,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    let constraints = serving.traffic.kv_route_constraints;
    if !constraints.any() {
        return Ok(());
    }

    let placements = ExplicitServingPlacements {
        prefill: prefill_placement,
        decode: decode_placement,
    };
    let mut checked_candidates = 0usize;
    let mut first_failure = None;

    for (idx, candidate) in serving.pool_candidates.iter().enumerate() {
        let prefill_nodes = resolve_configured_pool_nodes(
            &format!("serving.pool_candidates[{idx}].prefill"),
            cluster,
            &candidate.prefill_nodes,
            &candidate.prefill_groups,
        )?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &candidate.prefill_node_filter,
        );
        let decode_nodes = resolve_configured_pool_nodes(
            &format!("serving.pool_candidates[{idx}].decode"),
            cluster,
            &candidate.decode_nodes,
            &candidate.decode_groups,
        )?;
        let decode_nodes =
            nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
        if !pool_nodes_satisfy_domain_spread(
            cluster,
            &candidate.domain_spread,
            &prefill_nodes,
            &decode_nodes,
        ) || !explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            prefill_placement,
            decode_placement,
        ) {
            continue;
        }
        checked_candidates += 1;
        match serving_pool_route_constraints_satisfied(
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            constraints,
            placements,
        ) {
            Ok(()) => return Ok(()),
            Err(err) => first_failure.get_or_insert(err),
        };
    }

    if let Some(pool_search) = &serving.pool_search
        && pool_search_can_generate_route_constraint_candidate(
            cluster,
            pool_search,
            serving.deployment_mode,
            constraints,
            placements,
            &mut checked_candidates,
            &mut first_failure,
        )?
    {
        return Ok(());
    }

    let first_failure = first_failure
        .map(|failure| format!(" first failure: {failure}"))
        .unwrap_or_default();
    Err(ConfigError::new(format!(
        "serving has no configured prefill/decode pool candidate satisfying configured KV route constraints after checking {checked_candidates} candidate(s); add rail-diverse/GPUDirect-capable routes, choose different pools, or lower serving.min_kv_route_rail_count / serving.require_kv_route_rail_metadata / serving.require_gpudirect_kv_paths.{first_failure}"
    )))
}

#[derive(Copy, Clone)]
pub(super) struct ExplicitServingPlacements<'a> {
    prefill: Option<&'a RankPlacement>,
    decode: Option<&'a RankPlacement>,
}

pub(super) fn pool_search_can_generate_route_constraint_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    constraints: ServingKvRouteConstraints,
    placements: ExplicitServingPlacements<'_>,
    checked_candidates: &mut usize,
    first_failure: &mut Option<String>,
) -> Result<bool, ConfigError> {
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    for prefill in combinations(&prefill_nodes, prefill_count as usize) {
                        for decode in combinations(&decode_nodes, decode_count as usize) {
                            if !pool_search_concrete_candidate_satisfies_constraints(
                                cluster,
                                pool_search,
                                &prefill,
                                &decode,
                                deployment_mode,
                            ) || !explicit_serving_placements_match_pool(
                                cluster,
                                deployment_mode,
                                &prefill,
                                &pool_search.prefill_gpu_labels,
                                &decode,
                                &pool_search.decode_gpu_labels,
                                placements.prefill,
                                placements.decode,
                            ) {
                                continue;
                            }
                            *checked_candidates += 1;
                            match serving_pool_route_constraints_satisfied(
                                cluster,
                                &prefill,
                                &pool_search.prefill_gpu_labels,
                                &decode,
                                &pool_search.decode_gpu_labels,
                                constraints,
                                placements,
                            ) {
                                Ok(()) => return Ok(true),
                                Err(err) => {
                                    first_failure.get_or_insert(err);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn explicit_serving_placements_match_pool(
    cluster: &Cluster,
    deployment_mode: ServingDeploymentMode,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> bool {
    deployment_mode.accepts_pool(prefill_nodes, decode_nodes)
        && prefill_placement.is_none_or(|placement| {
            placement_matches_pool_scope(cluster, placement, prefill_nodes, prefill_gpu_labels)
        })
        && decode_placement.is_none_or(|placement| {
            placement_matches_pool_scope(cluster, placement, decode_nodes, decode_gpu_labels)
        })
}

pub(super) fn placement_matches_pool_scope(
    cluster: &Cluster,
    placement: &RankPlacement,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> bool {
    let node_ids = node_ids.iter().copied().collect::<BTreeSet<_>>();
    placement.rank_to_gpu.iter().all(|addr| {
        node_ids.contains(&addr.node_id)
            && cluster.node(addr.node_id).is_some_and(|node| {
                gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
            })
    })
}

pub(super) fn pool_search_can_generate_explicit_placement_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<bool, ConfigError> {
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    if pool_search_can_generate_explicit_placement_candidate_for_counts(
                        cluster,
                        pool_search,
                        &prefill_nodes,
                        prefill_count as usize,
                        &decode_nodes,
                        decode_count as usize,
                        deployment_mode,
                        prefill_placement,
                        decode_placement,
                    ) {
                        return Ok(true);
                    }
                }
            }
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pool_search_can_generate_explicit_placement_candidate_for_counts(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    prefill_count: usize,
    decode_nodes: &[u32],
    decode_count: usize,
    deployment_mode: ServingDeploymentMode,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> bool {
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) && explicit_serving_placements_match_pool(
                cluster,
                deployment_mode,
                &prefill,
                &pool_search.prefill_gpu_labels,
                &decode,
                &pool_search.decode_gpu_labels,
                prefill_placement,
                decode_placement,
            ) {
                return true;
            }
        }
    }
    false
}

pub(super) fn validate_serving_deployment_mode(
    name: &str,
    deployment_mode: ServingDeploymentMode,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
) -> Result<(), ConfigError> {
    if deployment_mode.accepts_pool(prefill_nodes, decode_nodes) {
        return Ok(());
    }

    let effective = ServingDeploymentMode::effective_for_pool(prefill_nodes, decode_nodes);
    Err(ConfigError::new(format!(
        "{name} uses {} prefill/decode nodes, but serving.mode '{}' requires {}",
        effective.as_str(),
        deployment_mode.as_str(),
        deployment_mode.as_str()
    )))
}

pub(super) fn pool_search_can_generate_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    prefill_count: usize,
    decode_nodes: &[u32],
    decode_count: usize,
    deployment_mode: ServingDeploymentMode,
) -> bool {
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) {
                return true;
            }
        }
    }
    false
}

pub(super) fn pool_search_concrete_candidate_satisfies_constraints(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
    deployment_mode: ServingDeploymentMode,
) -> bool {
    if !pool_search.allow_overlap
        && deployment_mode != ServingDeploymentMode::Colocated
        && deployment_mode != ServingDeploymentMode::PartiallyDisaggregated
        && overlaps_nodes(prefill_nodes, decode_nodes)
    {
        return false;
    }

    deployment_mode.accepts_pool(prefill_nodes, decode_nodes)
        && pool_nodes_satisfy_domain_spread(
            cluster,
            &pool_search.domain_spread,
            prefill_nodes,
            decode_nodes,
        )
}

pub(super) fn pool_nodes_satisfy_domain_spread(
    cluster: &Cluster,
    domain_spread: &ServingPoolDomainSpread,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
) -> bool {
    pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    )
}

pub(super) fn pool_nodes_meet_min_domain_count(
    cluster: &Cluster,
    node_ids: &[u32],
    min_count: Option<u32>,
    domain: impl Fn(&Node) -> Option<&str>,
) -> bool {
    let Some(min_count) = min_count else {
        return true;
    };
    let domains = node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id))
        .filter_map(domain)
        .collect::<BTreeSet<_>>();
    domains.len() >= min_count as usize
}

pub(super) fn pool_search_can_generate_routable_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
    counts: (usize, usize),
    gpu_labels: (&[String], &[String]),
    deployment_mode: ServingDeploymentMode,
) -> bool {
    let (prefill_count, decode_count) = counts;
    let (prefill_gpu_labels, decode_gpu_labels) = gpu_labels;
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) && serving_pool_route_is_routable(
                cluster,
                &prefill,
                prefill_gpu_labels,
                &decode,
                decode_gpu_labels,
            ) {
                return true;
            }
        }
    }
    false
}

pub(super) fn validate_serving_pool_route(
    name: &str,
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
) -> Result<(), ConfigError> {
    if serving_pool_route_is_routable(
        cluster,
        prefill_nodes,
        prefill_gpu_labels,
        decode_nodes,
        decode_gpu_labels,
    ) {
        return Ok(());
    }

    Err(ConfigError::new(format!(
        "{name} has no routable KV transfer path between prefill nodes {prefill_nodes:?} labels {prefill_gpu_labels:?} and decode nodes {decode_nodes:?} labels {decode_gpu_labels:?}; add missing custom inter-node links, choose connected prefill/decode pools, adjust GPU tag constraints, or disable serving.require_routable_pools for failure-scenario sweeps"
    )))
}

pub(super) fn serving_pool_route_is_routable(
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
) -> bool {
    let prefill_gpus =
        available_gpu_addrs_for_nodes_with_labels(cluster, prefill_nodes, prefill_gpu_labels);
    let decode_gpus =
        available_gpu_addrs_for_nodes_with_labels(cluster, decode_nodes, decode_gpu_labels);
    if prefill_gpus.is_empty() || decode_gpus.is_empty() {
        return true;
    }
    Solver::transfer_between_gpus_routable(
        cluster,
        &prefill_gpus,
        &decode_gpus,
        Bytes::from_bytes(1),
    )
}

#[derive(Default)]
pub(super) struct ServingRouteConstraintSummary {
    inter_node_pair_count: u32,
    routable_inter_node_path_count: u32,
    inter_node_route_resource_count: u32,
    unrailed_inter_node_route_resource_count: u32,
    host_staged_gpu_nic_resource_count: u32,
    rail_ids: BTreeSet<u32>,
}

pub(super) fn serving_pool_route_constraints_satisfied(
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
    constraints: ServingKvRouteConstraints,
    placements: ExplicitServingPlacements<'_>,
) -> Result<(), String> {
    let prefill_gpus = route_constraint_gpus_for_pool(
        cluster,
        prefill_nodes,
        prefill_gpu_labels,
        placements.prefill,
    );
    let decode_gpus =
        route_constraint_gpus_for_pool(cluster, decode_nodes, decode_gpu_labels, placements.decode);
    if prefill_gpus.is_empty() || decode_gpus.is_empty() {
        return Ok(());
    }

    let summary = serving_route_constraint_summary(cluster, &prefill_gpus, &decode_gpus)?;
    if summary.inter_node_route_resource_count > 0 {
        if let Some(limit) = constraints.min_inter_node_rail_count {
            let rail_count = summary.rail_ids.len().min(u32::MAX as usize) as u32;
            if rail_count < limit {
                return Err(format!(
                    "KV transfer routes expose {rail_count} inter-node rail(s), below configured serving.min_kv_route_rail_count {limit}"
                ));
            }
        }
        if constraints.require_inter_node_rail_metadata
            && summary.unrailed_inter_node_route_resource_count > 0
        {
            return Err(format!(
                "{} inter-node KV route resource(s) lack rail metadata",
                summary.unrailed_inter_node_route_resource_count
            ));
        }
    }
    if constraints.require_gpudirect && summary.host_staged_gpu_nic_resource_count > 0 {
        return Err(format!(
            "{} GPU/NIC KV route resource(s) are host-staged or marked no-GPUDirect",
            summary.host_staged_gpu_nic_resource_count
        ));
    }

    Ok(())
}

pub(super) fn route_constraint_gpus_for_pool(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
    placement: Option<&RankPlacement>,
) -> Vec<GpuAddr> {
    let Some(placement) = placement else {
        return available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels);
    };
    let node_ids = node_ids.iter().copied().collect::<BTreeSet<_>>();
    placement
        .rank_to_gpu
        .iter()
        .copied()
        .filter(|addr| {
            node_ids.contains(&addr.node_id)
                && cluster.node(addr.node_id).is_some_and(|node| {
                    gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
                })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn serving_route_constraint_summary(
    cluster: &Cluster,
    prefill_gpus: &[GpuAddr],
    decode_gpus: &[GpuAddr],
) -> Result<ServingRouteConstraintSummary, String> {
    let graph = TopologyGraph::from_cluster(cluster);
    let mut summary = ServingRouteConstraintSummary::default();
    let mut first_unroutable_path = None;
    for source in prefill_gpus {
        for destination in decode_gpus {
            if source == destination || source.node_id == destination.node_id {
                continue;
            }
            summary.inter_node_pair_count = summary.inter_node_pair_count.saturating_add(1);
            let Some(path) = graph.route_between_gpus(*source, *destination, Bytes::from_bytes(1))
            else {
                first_unroutable_path.get_or_insert_with(|| {
                    format!(
                        "no routable KV transfer path between node {} gpu {} and node {} gpu {}",
                        source.node_id,
                        source.local_gpu_id,
                        destination.node_id,
                        destination.local_gpu_id
                    )
                });
                continue;
            };
            summary.routable_inter_node_path_count =
                summary.routable_inter_node_path_count.saturating_add(1);
            for resource in path.resources {
                match resource.kind {
                    RoutedResourceKind::InterNodeFabric
                    | RoutedResourceKind::GpuScopedInterNodeFabric => {
                        summary.inter_node_route_resource_count =
                            summary.inter_node_route_resource_count.saturating_add(1);
                        if let Some(rail_id) = resource.rail_id {
                            summary.rail_ids.insert(rail_id);
                        } else {
                            summary.unrailed_inter_node_route_resource_count = summary
                                .unrailed_inter_node_route_resource_count
                                .saturating_add(1);
                        }
                    }
                    RoutedResourceKind::GpuNicLocal => {
                        if route_label_contains_any(
                            &resource.label,
                            &["host_staged", "no_gpudirect"],
                        ) {
                            summary.host_staged_gpu_nic_resource_count =
                                summary.host_staged_gpu_nic_resource_count.saturating_add(1);
                        }
                    }
                    RoutedResourceKind::IntraNodeFabric => {}
                }
            }
        }
    }
    if summary.inter_node_pair_count > 0 && summary.routable_inter_node_path_count == 0 {
        return Err(first_unroutable_path.unwrap_or_else(|| {
            "no routable inter-node KV transfer path between prefill and decode pools".to_string()
        }));
    }
    Ok(summary)
}

pub(super) fn route_label_contains_any(label: &str, needles: &[&str]) -> bool {
    let label = normalize(label);
    needles.iter().any(|needle| label.contains(needle))
}

pub(super) fn available_gpu_addrs_for_nodes_with_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> Vec<GpuAddr> {
    let mut addrs = Vec::new();
    for node_id in node_ids {
        let Some(node) = cluster.node(*node_id) else {
            continue;
        };
        let mut gpu_ids: Vec<_> = node.gpus.keys().copied().collect();
        gpu_ids.sort_unstable();
        addrs.extend(
            gpu_ids
                .into_iter()
                .map(|local_gpu_id| GpuAddr {
                    node_id: *node_id,
                    local_gpu_id,
                })
                .filter(|addr| {
                    cluster.is_gpu_available(*addr)
                        && gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
                }),
        );
    }
    addrs
}

pub(super) fn gpu_count_for_nodes_with_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> u32 {
    available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels)
        .len()
        .min(u32::MAX as usize) as u32
}

pub(super) fn gpu_count_for_nodes_with_labels_and_dtype(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
    dtype: DType,
) -> u32 {
    available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels)
        .into_iter()
        .filter(|addr| {
            cluster
                .gpu_profile(*addr)
                .is_some_and(|profile| gpu_profile_supports_dtype(&profile, dtype))
        })
        .count()
        .min(u32::MAX as usize) as u32
}

pub(super) fn nodes_with_gpu_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> Vec<u32> {
    if gpu_labels.is_empty() {
        return node_ids.to_vec();
    }
    node_ids
        .iter()
        .copied()
        .filter(|node_id| gpu_count_for_nodes_with_labels(cluster, &[*node_id], gpu_labels) > 0)
        .collect()
}

pub(super) fn nodes_matching_pool_node_filter(
    cluster: &Cluster,
    node_ids: &[u32],
    filter: &ServingPoolNodeFilter,
) -> Vec<u32> {
    node_ids
        .iter()
        .copied()
        .filter(|node_id| {
            cluster
                .node(*node_id)
                .is_some_and(|node| node_matches_pool_node_filter(node, filter))
        })
        .collect()
}

pub(super) fn node_matches_pool_node_filter(node: &Node, filter: &ServingPoolNodeFilter) -> bool {
    (filter.node_labels.is_empty()
        || filter
            .node_labels
            .iter()
            .any(|label| node.topology.labels.contains(label)))
        && (filter.racks.is_empty()
            || node
                .topology
                .rack
                .as_ref()
                .is_some_and(|rack| filter.racks.contains(rack)))
        && (filter.islands.is_empty()
            || node
                .topology
                .island
                .as_ref()
                .is_some_and(|island| filter.islands.contains(island)))
        && (filter.failure_domains.is_empty()
            || node
                .topology
                .failure_domain
                .as_ref()
                .is_some_and(|failure_domain| filter.failure_domains.contains(failure_domain)))
}

pub(super) fn gpu_labels_match(labels: Option<&BTreeSet<String>>, required: &[String]) -> bool {
    required.is_empty()
        || labels.is_some_and(|labels| required.iter().any(|label| labels.contains(label)))
}

pub(super) fn validate_no_duplicate_nodes(name: &str, nodes: &[u32]) -> Result<(), ConfigError> {
    let mut seen = HashSet::new();
    for node in nodes {
        if !seen.insert(*node) {
            return Err(ConfigError::new(format!(
                "{name} contains duplicate node id {node}"
            )));
        }
    }
    Ok(())
}

pub(super) fn resolve_configured_pool_nodes(
    name: &str,
    cluster: &Cluster,
    explicit_nodes: &[u32],
    groups: &[String],
) -> Result<Vec<u32>, ConfigError> {
    let mut nodes = Vec::new();
    for node_id in explicit_nodes {
        if !cluster.nodes.contains_key(node_id) {
            return Err(ConfigError::new(format!(
                "{name}_nodes references unknown node id {node_id}"
            )));
        }
        nodes.push(*node_id);
    }
    for group in groups {
        nodes.extend_from_slice(&configured_group_nodes(
            &format!("{name}_groups"),
            cluster,
            group,
        )?);
    }
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Err(ConfigError::new(format!("{name} resolves to no nodes")));
    }
    Ok(nodes)
}

pub(super) fn configured_group_nodes(
    name: &str,
    cluster: &Cluster,
    group: &str,
) -> Result<Vec<u32>, ConfigError> {
    let nodes = cluster.node_group(group).ok_or_else(|| {
        ConfigError::new(format!("{name} references unknown node group '{group}'"))
    })?;
    if nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} node group '{group}' contains no nodes"
        )));
    }
    Ok(nodes.to_vec())
}

pub(super) fn valid_pool_search_counts(
    name: &str,
    configured: &[u32],
    available_nodes: usize,
) -> Result<Vec<u32>, ConfigError> {
    let mut counts = if configured.is_empty() {
        vec![available_nodes as u32]
    } else {
        configured.to_vec()
    };
    counts.sort_unstable();
    counts.dedup();
    let counts: Vec<_> = counts
        .into_iter()
        .filter(|count| *count > 0 && (*count as usize) <= available_nodes)
        .collect();
    if counts.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} has no value that fits {available_nodes} available nodes"
        )));
    }
    Ok(counts)
}

pub(super) fn combinations(values: &[u32], count: usize) -> Vec<Vec<u32>> {
    if count == 0 || count > values.len() {
        return Vec::new();
    }
    if count == values.len() {
        return vec![values.to_vec()];
    }

    let mut results = Vec::new();
    let mut current = Vec::with_capacity(count);
    push_combinations(values, count, 0, &mut current, &mut results);
    results
}

pub(super) fn push_combinations(
    values: &[u32],
    count: usize,
    start: usize,
    current: &mut Vec<u32>,
    results: &mut Vec<Vec<u32>>,
) {
    if current.len() == count {
        results.push(current.clone());
        return;
    }

    let needed = count - current.len();
    for idx in start..=values.len() - needed {
        current.push(values[idx]);
        push_combinations(values, count, idx + 1, current, results);
        current.pop();
    }
}

pub(super) fn overlaps_nodes(left: &[u32], right: &[u32]) -> bool {
    left.iter().any(|node| right.contains(node))
}

pub(super) fn parse_serving_traffic(
    traffic: Option<ServingTrafficSection>,
    base_dir: Option<&Path>,
    traffic_classes: Vec<ServingTrafficClass>,
    serving_service_sections: ServingServiceSections,
    metric_ceilings: ServingMetricCeilings,
    kv_route_constraints: ServingKvRouteConstraints,
) -> Result<ServingTraffic, ConfigError> {
    let mut services = parse_serving_services(
        "serving",
        serving_service_sections.services,
        serving_service_sections.prefill,
        serving_service_sections.decode,
        serving_service_sections.kv_transfer,
    )?;
    let Some(traffic) = traffic else {
        return Ok(ServingTraffic {
            services: finalize_serving_services(services),
            metric_ceilings,
            kv_route_constraints,
            traffic_classes,
            ..ServingTraffic::default()
        });
    };
    services = merge_serving_services(
        services,
        parse_serving_services(
            "serving.traffic",
            traffic.services.clone(),
            traffic.prefill_service.clone(),
            traffic.decode_service.clone(),
            traffic.kv_transfer_service.clone(),
        )?,
    );

    if let Some(request_count) = traffic.request_count
        && request_count == 0
    {
        return Err(ConfigError::new(
            "serving.traffic.request_count must be greater than zero",
        ));
    }
    let arrival_gap_s = parse_optional_trace_seconds(
        "serving.traffic.arrival_gap_s",
        traffic.arrival_gap_s,
        "serving.traffic.arrival_gap_ms",
        traffic.arrival_gap_ms,
    )?;
    if let Some(arrival_gap_s) = arrival_gap_s
        && (!arrival_gap_s.is_finite() || arrival_gap_s < 0.0)
    {
        return Err(ConfigError::new(
            "serving.traffic.arrival_gap_s/arrival_gap_ms must be finite and non-negative",
        ));
    }
    if let Some(0) = traffic.max_prefill_batch_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_batch_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_chunk_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_chunk_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_tokens_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_tokens_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_tokens_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_tokens_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_worker_slots_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_worker_slots_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_sequences {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_sequences must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_resident_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_resident_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_sequences_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_sequences_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_resident_tokens_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_resident_tokens_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_sequences_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_sequences_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_worker_slots_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_worker_slots_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_transfer_worker_slots_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_transfer_worker_slots_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_resident_tokens_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_resident_tokens_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.kv_block_tokens {
        return Err(ConfigError::new(
            "serving.traffic.kv_block_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_blocks {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_blocks must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_blocks_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_blocks_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_blocks_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_blocks_per_gpu must be greater than zero",
        ));
    }
    let arrival = parse_arrival_pattern(&traffic)?;
    let prefill_batching = parse_prefill_batching(
        traffic.prefill_batching.as_deref(),
        traffic.max_prefill_batch_tokens,
        traffic.max_prefill_chunk_tokens,
    )?;
    let decode_batching = parse_decode_batching(
        traffic.decode_batching.as_deref(),
        traffic.max_decode_batch_tokens,
    )?;
    let trace_window = parse_trace_window(&traffic)?;
    let trace_replay = parse_trace_replay(&traffic)?;
    let measurement_window = parse_measurement_window(&traffic)?;
    if let Some(0) = traffic.measurement_steady_state_min_requests {
        return Err(ConfigError::new(
            "serving.traffic.measurement_steady_state_min_requests must be greater than zero",
        ));
    }
    let measurement_steady_state_max_cv = parse_non_negative_f64(
        "serving.traffic.measurement_steady_state_max_cv",
        traffic.measurement_steady_state_max_cv,
    )?;
    let max_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_queue_delay_s",
        traffic.max_queue_delay_s,
        "serving.traffic.max_queue_delay_ms",
        traffic.max_queue_delay_ms,
    )?;
    let max_kv_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_kv_queue_delay_s",
        traffic.max_kv_queue_delay_s,
        "serving.traffic.max_kv_queue_delay_ms",
        traffic.max_kv_queue_delay_ms,
    )?;
    let max_decode_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_decode_queue_delay_s",
        traffic.max_decode_queue_delay_s,
        "serving.traffic.max_decode_queue_delay_ms",
        traffic.max_decode_queue_delay_ms,
    )?;
    let max_decode_iteration_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_decode_iteration_queue_delay_s",
        traffic.max_decode_iteration_queue_delay_s,
        "serving.traffic.max_decode_iteration_queue_delay_ms",
        traffic.max_decode_iteration_queue_delay_ms,
    )?;
    let request_timeout_s = parse_positive_optional_seconds(
        "serving.traffic.request_timeout_s",
        traffic.request_timeout_s,
        "serving.traffic.request_timeout_ms",
        traffic.request_timeout_ms,
    )?;
    let trace_requests = parse_trace_requests(
        traffic.requests,
        traffic.trace_csv,
        traffic.trace_jsonl,
        base_dir,
    )?;
    let trace_requests = apply_trace_window(trace_requests, trace_window)?;
    let trace_requests = apply_trace_replay(trace_requests, trace_replay)?;
    validate_unique_trace_request_ids(&trace_requests)?;
    if !trace_requests.is_empty()
        && let Some(request_count) = traffic.request_count
        && request_count as usize != trace_requests.len()
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.request_count must match serving.traffic.requests length ({}) when trace requests are provided",
            trace_requests.len()
        )));
    }
    if matches!(arrival, ServingArrivalPattern::TraceDerived) && trace_requests.is_empty() {
        return Err(ConfigError::new(
            "serving.traffic arrival = 'trace_derived' requires inline requests, trace_csv, or trace_jsonl",
        ));
    }

    Ok(ServingTraffic {
        request_count: traffic.request_count,
        arrival_gap_s,
        arrival,
        routing_policy: parse_routing_policy(traffic.routing_policy.as_deref())?,
        prefill_batching,
        decode_batching,
        decode_capacity_policy: parse_decode_capacity_policy(
            traffic.decode_capacity_policy.as_deref(),
        )?,
        services: finalize_serving_services(services),
        service_backpressure_penalty_weight: parse_non_negative_f64(
            "serving.traffic.service_backpressure_penalty_weight",
            traffic.service_backpressure_penalty_weight,
        )?
        .unwrap_or(0.0),
        max_prefill_tokens: traffic.max_prefill_tokens,
        max_prefill_tokens_per_node: traffic.max_prefill_tokens_per_node,
        max_prefill_tokens_per_gpu: traffic.max_prefill_tokens_per_gpu,
        max_prefill_worker_slots_per_gpu: traffic.max_prefill_worker_slots_per_gpu,
        max_decode_sequences: traffic.max_decode_sequences,
        max_resident_tokens: traffic.max_resident_tokens,
        max_decode_sequences_per_node: traffic.max_decode_sequences_per_node,
        max_resident_tokens_per_node: traffic.max_resident_tokens_per_node,
        max_decode_sequences_per_gpu: traffic.max_decode_sequences_per_gpu,
        max_decode_worker_slots_per_gpu: traffic.max_decode_worker_slots_per_gpu,
        max_resident_tokens_per_gpu: traffic.max_resident_tokens_per_gpu,
        max_kv_transfer_worker_slots_per_gpu: traffic.max_kv_transfer_worker_slots_per_gpu,
        kv_block_tokens: traffic.kv_block_tokens,
        max_kv_blocks: traffic.max_kv_blocks,
        max_kv_blocks_per_node: traffic.max_kv_blocks_per_node,
        max_kv_blocks_per_gpu: traffic.max_kv_blocks_per_gpu,
        ttft_slo_s: parse_slo_ms("serving.traffic.ttft_slo_ms", traffic.ttft_slo_ms)?,
        tpot_slo_s: parse_slo_ms("serving.traffic.tpot_slo_ms", traffic.tpot_slo_ms)?,
        itl_slo_s: parse_slo_ms("serving.traffic.itl_slo_ms", traffic.itl_slo_ms)?,
        e2el_slo_s: parse_slo_ms("serving.traffic.e2el_slo_ms", traffic.e2el_slo_ms)?,
        max_ttft_slo_miss_rate: parse_fraction(
            "serving.traffic.max_ttft_slo_miss_rate",
            traffic.max_ttft_slo_miss_rate,
        )?,
        max_tpot_slo_miss_rate: parse_fraction(
            "serving.traffic.max_tpot_slo_miss_rate",
            traffic.max_tpot_slo_miss_rate,
        )?,
        max_itl_slo_miss_rate: parse_fraction(
            "serving.traffic.max_itl_slo_miss_rate",
            traffic.max_itl_slo_miss_rate,
        )?,
        max_e2el_slo_miss_rate: parse_fraction(
            "serving.traffic.max_e2el_slo_miss_rate",
            traffic.max_e2el_slo_miss_rate,
        )?,
        max_deadline_miss_rate: parse_fraction(
            "serving.traffic.max_deadline_miss_rate",
            traffic.max_deadline_miss_rate,
        )?,
        metric_ceilings,
        kv_route_constraints,
        measurement_start_s: measurement_window.start_s,
        measurement_end_s: measurement_window.end_s,
        measurement_warmup_s: measurement_window.warmup_s,
        measurement_cooldown_s: measurement_window.cooldown_s,
        measurement_steady_state: traffic.measurement_steady_state.unwrap_or(false),
        measurement_steady_state_min_requests: traffic.measurement_steady_state_min_requests,
        measurement_steady_state_max_cv,
        max_queue_delay_s,
        max_kv_queue_delay_s,
        max_decode_queue_delay_s,
        max_decode_iteration_queue_delay_s,
        request_timeout_s,
        shape_seed: traffic.shape_seed.unwrap_or(1),
        prefix_cache_hit_rate: parse_cache_hit_rate(
            "serving.traffic.prefix_cache_hit_rate",
            traffic.prefix_cache_hit_rate,
        )?,
        batch_size_distribution: parse_value_distribution(
            "serving.traffic.batch_size_distribution",
            traffic.batch_size_distribution,
        )?,
        prompt_tokens_distribution: parse_value_distribution(
            "serving.traffic.prompt_tokens_distribution",
            traffic.prompt_tokens_distribution,
        )?,
        decode_tokens_distribution: parse_value_distribution(
            "serving.traffic.decode_tokens_distribution",
            traffic.decode_tokens_distribution,
        )?,
        shape_profiles: parse_shape_profiles(
            "serving.traffic.shape_profiles",
            traffic.shape_profiles,
        )?,
        batch_sizes: positive_values(
            "serving.traffic.batch_sizes",
            traffic.batch_sizes.unwrap_or_default(),
        )?,
        prompt_tokens: positive_values(
            "serving.traffic.prompt_tokens",
            traffic.prompt_tokens.unwrap_or_default(),
        )?,
        decode_tokens: positive_values(
            "serving.traffic.decode_tokens",
            traffic.decode_tokens.unwrap_or_default(),
        )?,
        trace_requests,
        traffic_classes,
    })
}

#[derive(Default)]
pub(super) struct PartialServingServicesConfig {
    prefill: Option<ServingServicePhaseConfig>,
    decode: Option<ServingServicePhaseConfig>,
    kv_transfer: Option<ServingServicePhaseConfig>,
}

pub(super) fn parse_serving_services(
    path: &str,
    services: Option<ServingServicesSection>,
    prefill: Option<ServingServicePhaseSection>,
    decode: Option<ServingServicePhaseSection>,
    kv_transfer: Option<ServingServicePhaseSection>,
) -> Result<PartialServingServicesConfig, ConfigError> {
    let mut parsed = PartialServingServicesConfig::default();
    if let Some(services) = services {
        parsed.prefill =
            parse_serving_service_phase(&format!("{path}.services.prefill"), services.prefill)?;
        parsed.decode =
            parse_serving_service_phase(&format!("{path}.services.decode"), services.decode)?;
        parsed.kv_transfer = parse_serving_service_phase(
            &format!("{path}.services.kv_transfer"),
            services.kv_transfer,
        )?;
    }
    parsed.prefill = parse_serving_service_phase(&format!("{path}.prefill_service"), prefill)?
        .or(parsed.prefill);
    parsed.decode =
        parse_serving_service_phase(&format!("{path}.decode_service"), decode)?.or(parsed.decode);
    parsed.kv_transfer =
        parse_serving_service_phase(&format!("{path}.kv_transfer_service"), kv_transfer)?
            .or(parsed.kv_transfer);
    Ok(parsed)
}

pub(super) fn parse_serving_service_phase(
    path: &str,
    section: Option<ServingServicePhaseSection>,
) -> Result<Option<ServingServicePhaseConfig>, ConfigError> {
    let Some(section) = section else {
        return Ok(None);
    };
    let mut health = parse_serving_service_health(path, section.health.as_deref())?;
    if section.enabled == Some(false) {
        health = ServingServiceHealth::Unavailable;
    }
    let worker_scale = parse_serving_service_worker_scale(path, section.worker_scale)?;
    Ok(Some(ServingServicePhaseConfig {
        health,
        worker_scale,
    }))
}

pub(super) fn parse_serving_service_health(
    path: &str,
    health: Option<&str>,
) -> Result<ServingServiceHealth, ConfigError> {
    match health.map(normalize).as_deref() {
        None | Some("healthy") | Some("available") | Some("enabled") => {
            Ok(ServingServiceHealth::Healthy)
        }
        Some("draining") | Some("drain") => Ok(ServingServiceHealth::Draining),
        Some("unavailable") | Some("disabled") | Some("down") => {
            Ok(ServingServiceHealth::Unavailable)
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported {path}.health '{value}'; use healthy, draining, or unavailable"
        ))),
    }
}

pub(super) fn parse_serving_service_worker_scale(
    path: &str,
    worker_scale: Option<f64>,
) -> Result<f64, ConfigError> {
    let Some(worker_scale) = worker_scale else {
        return Ok(1.0);
    };
    if !worker_scale.is_finite() || worker_scale <= 0.0 {
        return Err(ConfigError::new(format!(
            "{path}.worker_scale must be finite and greater than zero"
        )));
    }
    Ok(worker_scale)
}

pub(super) fn merge_serving_services(
    mut base: PartialServingServicesConfig,
    override_config: PartialServingServicesConfig,
) -> PartialServingServicesConfig {
    if override_config.prefill.is_some() {
        base.prefill = override_config.prefill;
    }
    if override_config.decode.is_some() {
        base.decode = override_config.decode;
    }
    if override_config.kv_transfer.is_some() {
        base.kv_transfer = override_config.kv_transfer;
    }
    base
}

pub(super) fn finalize_serving_services(
    partial: PartialServingServicesConfig,
) -> ServingServicesConfig {
    ServingServicesConfig {
        prefill: partial.prefill.unwrap_or_default(),
        decode: partial.decode.unwrap_or_default(),
        kv_transfer: partial.kv_transfer.unwrap_or_default(),
    }
}

pub(super) fn parse_positive_optional_seconds(
    seconds_name: &str,
    seconds: Option<f64>,
    millis_name: &str,
    millis: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    let value = parse_optional_trace_seconds(seconds_name, seconds, millis_name, millis)?;
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(ConfigError::new(format!(
            "{seconds_name}/{millis_name} must be positive"
        )));
    }
    Ok(value)
}

pub(super) fn parse_value_distribution(
    name: &str,
    distribution: Option<ServingValueDistributionSection>,
) -> Result<Option<ServingValueDistribution>, ConfigError> {
    let Some(distribution) = distribution else {
        return Ok(None);
    };

    match normalize(&distribution.kind).as_str() {
        "uniform" => {
            let min = distribution
                .min
                .ok_or_else(|| ConfigError::new(format!("{name}.min is required")))?;
            let max = distribution
                .max
                .ok_or_else(|| ConfigError::new(format!("{name}.max is required")))?;
            if min == 0 || max == 0 || min > max {
                return Err(ConfigError::new(format!(
                    "{name}.min and {name}.max must be positive with min <= max"
                )));
            }
            Ok(Some(ServingValueDistribution::Uniform { min, max }))
        }
        "weighted" | "categorical" => {
            let values = positive_values(
                &format!("{name}.values"),
                distribution.values.unwrap_or_default(),
            )?;
            let weights = distribution.weights.unwrap_or_default();
            if values.is_empty() {
                return Err(ConfigError::new(format!("{name}.values must not be empty")));
            }
            if values.len() != weights.len() {
                return Err(ConfigError::new(format!(
                    "{name}.values and {name}.weights must have the same length"
                )));
            }
            if !weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0)
            {
                return Err(ConfigError::new(format!(
                    "{name}.weights must be finite and positive"
                )));
            }
            Ok(Some(ServingValueDistribution::Weighted { values, weights }))
        }
        "lognormal" | "log_normal" => {
            let median = distribution
                .median
                .ok_or_else(|| ConfigError::new(format!("{name}.median is required")))?;
            let sigma = distribution
                .sigma
                .ok_or_else(|| ConfigError::new(format!("{name}.sigma is required")))?;
            let min = distribution
                .min
                .ok_or_else(|| ConfigError::new(format!("{name}.min is required")))?;
            let max = distribution
                .max
                .ok_or_else(|| ConfigError::new(format!("{name}.max is required")))?;
            if !median.is_finite() || median <= 0.0 {
                return Err(ConfigError::new(format!(
                    "{name}.median must be finite and positive"
                )));
            }
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(ConfigError::new(format!(
                    "{name}.sigma must be finite and positive"
                )));
            }
            if min == 0 || max == 0 || min > max {
                return Err(ConfigError::new(format!(
                    "{name}.min and {name}.max must be positive with min <= max"
                )));
            }
            Ok(Some(ServingValueDistribution::LogNormal {
                median,
                sigma,
                min,
                max,
            }))
        }
        value => Err(ConfigError::new(format!(
            "unsupported {name}.kind '{value}'; use uniform, weighted, or lognormal"
        ))),
    }
}

pub(super) fn parse_shape_profiles(
    name: &str,
    profiles: Option<Vec<ServingShapeProfileSection>>,
) -> Result<Vec<ServingShapeProfile>, ConfigError> {
    let Some(profiles) = profiles else {
        return Ok(Vec::new());
    };
    if profiles.is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }

    let mut names = HashSet::new();
    profiles
        .into_iter()
        .enumerate()
        .map(|(idx, profile)| {
            let profile_name = format!("{name}[{idx}]");
            let parsed_name = optional_nonempty_string(profile.name.clone())
                .unwrap_or_else(|| format!("profile-{idx}"));
            if !names.insert(parsed_name.clone()) {
                return Err(ConfigError::new(format!(
                    "{profile_name}.name '{parsed_name}' is duplicated"
                )));
            }
            if profile.batch_size == 0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.batch_size must be greater than zero"
                )));
            }
            if profile.prompt_tokens == 0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.prompt_tokens must be greater than zero"
                )));
            }
            if profile.decode_tokens == 0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.decode_tokens must be greater than zero"
                )));
            }
            validate_optional_max_sequence_tokens(
                &format!("{profile_name}.max_sequence_tokens"),
                profile.max_sequence_tokens,
                profile.prompt_tokens,
                profile.decode_tokens,
            )?;
            let weight = profile.weight.unwrap_or(1.0);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.weight must be finite and positive"
                )));
            }
            let prefix_cache_hit_rate = parse_cache_hit_rate(
                &format!("{profile_name}.prefix_cache_hit_rate"),
                profile.prefix_cache_hit_rate,
            )?;
            if profile.prefix_cache_hit_tokens.is_some() && prefix_cache_hit_rate.is_some() {
                return Err(ConfigError::new(format!(
                    "{profile_name} cannot set both prefix_cache_hit_tokens and prefix_cache_hit_rate"
                )));
            }
            if let Some(prefix_cache_hit_tokens) = profile.prefix_cache_hit_tokens
                && prefix_cache_hit_tokens > profile.prompt_tokens
            {
                return Err(ConfigError::new(format!(
                    "{profile_name}.prefix_cache_hit_tokens must be less than or equal to prompt_tokens"
                )));
            }
            let slo = parse_shape_profile_slo(&profile_name, &profile)?;
            let request_timeout_s = parse_positive_optional_seconds(
                &format!("{profile_name}.request_timeout_s"),
                profile.request_timeout_s,
                &format!("{profile_name}.request_timeout_ms"),
                profile.request_timeout_ms,
            )?;
            let deadline_after_s = parse_optional_trace_seconds(
                &format!("{profile_name}.deadline_after_s"),
                profile.deadline_after_s,
                &format!("{profile_name}.deadline_after_ms"),
                profile.deadline_after_ms,
            )?;
            if let Some(deadline_after_s) = deadline_after_s
                && deadline_after_s < 0.0
            {
                return Err(ConfigError::new(format!(
                    "{profile_name}.deadline_after must be non-negative"
                )));
            }
            let cancellation_after_s = parse_optional_trace_seconds(
                &format!("{profile_name}.cancel_after_s"),
                profile.cancel_after_s,
                &format!("{profile_name}.cancel_after_ms"),
                profile.cancel_after_ms,
            )?;
            if let Some(cancellation_after_s) = cancellation_after_s
                && cancellation_after_s < 0.0
            {
                return Err(ConfigError::new(format!(
                    "{profile_name}.cancel_after must be non-negative"
                )));
            }
            Ok(ServingShapeProfile {
                name: parsed_name,
                weight,
                tenant: optional_nonempty_string(profile.tenant),
                model_id: optional_nonempty_string(profile.model_id),
                cache_key: optional_nonempty_string(profile.cache_key),
                priority: profile.priority,
                batch_size: profile.batch_size,
                prompt_tokens: profile.prompt_tokens,
                decode_tokens: profile.decode_tokens,
                max_sequence_tokens: profile.max_sequence_tokens,
                prefix_cache_hit_tokens: profile.prefix_cache_hit_tokens,
                prefix_cache_hit_rate,
                slo,
                request_timeout_s,
                deadline_after_s,
                cancellation_after_s,
            })
        })
        .collect()
}

pub(super) fn parse_shape_profile_slo(
    name: &str,
    profile: &ServingShapeProfileSection,
) -> Result<ServingRequestSlo, ConfigError> {
    Ok(ServingRequestSlo {
        ttft_s: parse_shape_profile_slo_value(
            name,
            "ttft_slo",
            profile.ttft_slo_s,
            profile.ttft_slo_ms,
        )?,
        tpot_s: parse_shape_profile_slo_value(
            name,
            "tpot_slo",
            profile.tpot_slo_s,
            profile.tpot_slo_ms,
        )?,
        itl_s: parse_shape_profile_slo_value(
            name,
            "itl_slo",
            profile.itl_slo_s,
            profile.itl_slo_ms,
        )?,
        e2el_s: parse_shape_profile_slo_value(
            name,
            "e2el_slo",
            profile.e2el_slo_s,
            profile.e2el_slo_ms,
        )?,
    })
}

pub(super) fn parse_shape_profile_slo_value(
    profile_name: &str,
    name: &str,
    seconds: Option<f64>,
    millis: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    let value = parse_optional_trace_seconds(
        &format!("{profile_name}.{name}_s"),
        seconds,
        &format!("{profile_name}.{name}_ms"),
        millis,
    )?;
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(ConfigError::new(format!(
            "{profile_name}.{name} must be positive"
        )));
    }
    Ok(value)
}

pub(super) fn parse_routing_policy(
    routing_policy: Option<&str>,
) -> Result<ServingRoutingPolicy, ConfigError> {
    match routing_policy.map(normalize).as_deref() {
        None | Some("round_robin") | Some("roundrobin") => Ok(ServingRoutingPolicy::RoundRobin),
        Some("topology_aware")
        | Some("topologyaware")
        | Some("load_aware")
        | Some("topology_load_aware") => Ok(ServingRoutingPolicy::TopologyAware),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.routing_policy '{value}'; use round_robin or topology_aware"
        ))),
    }
}

pub(super) fn parse_slo_ms(name: &str, value: Option<f64>) -> Result<Option<f64>, ConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(ConfigError::new(format!(
            "{name} must be finite and positive"
        )));
    }
    Ok(Some(value / 1000.0))
}

pub(super) fn parse_cache_hit_rate(
    name: &str,
    value: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    parse_fraction(name, value)
}

pub(super) fn parse_fraction(name: &str, value: Option<f64>) -> Result<Option<f64>, ConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ConfigError::new(format!(
            "{name} must be finite and between 0.0 and 1.0"
        )));
    }
    Ok(Some(value))
}

pub(super) fn parse_non_negative_f64(
    name: &str,
    value: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() || value < 0.0 {
        return Err(ConfigError::new(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(Some(value))
}

pub(super) fn parse_prefill_batching(
    prefill_batching: Option<&str>,
    max_prefill_batch_tokens: Option<u64>,
    max_prefill_chunk_tokens: Option<u32>,
) -> Result<ServingPrefillBatching, ConfigError> {
    match prefill_batching.map(normalize).as_deref() {
        None | Some("independent") | Some("per_request") => Ok(ServingPrefillBatching::Independent),
        Some("continuous") | Some("continuous_batching") => {
            Ok(ServingPrefillBatching::Continuous {
                max_batch_tokens: max_prefill_batch_tokens,
                chunk_tokens: max_prefill_chunk_tokens,
            })
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.prefill_batching '{value}'; use independent or continuous"
        ))),
    }
}

pub(super) fn parse_decode_batching(
    decode_batching: Option<&str>,
    max_decode_batch_tokens: Option<u32>,
) -> Result<ServingDecodeBatching, ConfigError> {
    if let Some(0) = max_decode_batch_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_batch_tokens must be greater than zero",
        ));
    }

    match decode_batching.map(normalize).as_deref() {
        None | Some("independent") | Some("per_request") => Ok(ServingDecodeBatching::Independent),
        Some("continuous") | Some("continuous_batching") => Ok(ServingDecodeBatching::Continuous {
            max_batch_tokens: max_decode_batch_tokens,
        }),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.decode_batching '{value}'; use independent or continuous"
        ))),
    }
}

pub(super) fn parse_decode_capacity_policy(
    policy: Option<&str>,
) -> Result<ServingDecodeCapacityPolicy, ConfigError> {
    match policy.map(normalize).as_deref() {
        None
        | Some("candidate_reject")
        | Some("candidate")
        | Some("hard")
        | Some("hard_reject")
        | Some("reject_candidate") => Ok(ServingDecodeCapacityPolicy::CandidateReject),
        Some("request_reject")
        | Some("request")
        | Some("admission")
        | Some("admission_reject")
        | Some("reject_request") => Ok(ServingDecodeCapacityPolicy::RequestReject),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.decode_capacity_policy '{value}'; use candidate_reject or request_reject"
        ))),
    }
}

pub(super) fn parse_arrival_pattern(
    traffic: &ServingTrafficSection,
) -> Result<ServingArrivalPattern, ConfigError> {
    match traffic.arrival.as_deref().map(normalize).as_deref() {
        None | Some("fixed") | Some("fixed_gap") | Some("constant") => {
            Ok(ServingArrivalPattern::FixedGap)
        }
        Some("poisson") => {
            let rate_per_s = traffic.arrival_rate_per_s.ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.arrival_rate_per_s is required when arrival = 'poisson'",
                )
            })?;
            if !rate_per_s.is_finite() || rate_per_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.arrival_rate_per_s must be finite and positive",
                ));
            }
            Ok(ServingArrivalPattern::Poisson {
                rate_per_s,
                seed: traffic.arrival_seed.unwrap_or(1),
            })
        }
        Some("bursty") | Some("burst") | Some("bursts") => {
            let burst_size = traffic.burst_size.ok_or_else(|| {
                ConfigError::new("serving.traffic.burst_size is required when arrival = 'bursty'")
            })?;
            if burst_size == 0 {
                return Err(ConfigError::new(
                    "serving.traffic.burst_size must be greater than zero",
                ));
            }
            let burst_interval_s = parse_optional_trace_seconds(
                "serving.traffic.burst_interval_s",
                traffic.burst_interval_s,
                "serving.traffic.burst_interval_ms",
                traffic.burst_interval_ms,
            )?
            .ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.burst_interval_s or burst_interval_ms is required when arrival = 'bursty'",
                )
            })?;
            if burst_interval_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.burst_interval_s/burst_interval_ms must be positive",
                ));
            }
            let intra_burst_gap_s = parse_optional_trace_seconds(
                "serving.traffic.burst_arrival_gap_s",
                traffic.burst_arrival_gap_s,
                "serving.traffic.burst_arrival_gap_ms",
                traffic.burst_arrival_gap_ms,
            )?
            .unwrap_or(0.0);
            if intra_burst_gap_s < 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.burst_arrival_gap_s/burst_arrival_gap_ms must be non-negative",
                ));
            }
            let burst_span_s = f64::from(burst_size.saturating_sub(1)) * intra_burst_gap_s;
            if burst_span_s > burst_interval_s {
                return Err(ConfigError::new(
                    "serving.traffic burst_arrival_gap places requests beyond the next burst interval",
                ));
            }
            Ok(ServingArrivalPattern::Bursty {
                burst_size,
                burst_interval_s,
                intra_burst_gap_s,
            })
        }
        Some("diurnal") | Some("daily") | Some("sinusoidal") => {
            let min_rate_per_s = traffic.diurnal_min_rate_per_s.ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.diurnal_min_rate_per_s is required when arrival = 'diurnal'",
                )
            })?;
            if !min_rate_per_s.is_finite() || min_rate_per_s < 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_min_rate_per_s must be finite and non-negative",
                ));
            }

            let max_rate_per_s = traffic.diurnal_max_rate_per_s.ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.diurnal_max_rate_per_s is required when arrival = 'diurnal'",
                )
            })?;
            if !max_rate_per_s.is_finite() || max_rate_per_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_max_rate_per_s must be finite and positive",
                ));
            }
            if min_rate_per_s > max_rate_per_s {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_min_rate_per_s must be less than or equal to diurnal_max_rate_per_s",
                ));
            }

            let period_s = parse_optional_trace_seconds(
                "serving.traffic.diurnal_period_s",
                traffic.diurnal_period_s,
                "serving.traffic.diurnal_period_ms",
                traffic.diurnal_period_ms,
            )?
            .unwrap_or(86_400.0);
            if period_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_period_s/diurnal_period_ms must be positive",
                ));
            }
            let phase_s = parse_optional_trace_seconds(
                "serving.traffic.diurnal_phase_s",
                traffic.diurnal_phase_s,
                "serving.traffic.diurnal_phase_ms",
                traffic.diurnal_phase_ms,
            )?
            .unwrap_or(0.0);

            Ok(ServingArrivalPattern::Diurnal {
                min_rate_per_s,
                max_rate_per_s,
                period_s,
                phase_s,
                seed: traffic.arrival_seed.unwrap_or(1),
            })
        }
        Some("self_similar") | Some("selfsimilar") | Some("pareto") => {
            let rate_per_s = traffic
                .self_similar_rate_per_s
                .or(traffic.arrival_rate_per_s)
                .ok_or_else(|| {
                    ConfigError::new(
                        "serving.traffic.arrival_rate_per_s or self_similar_rate_per_s is required when arrival = 'self_similar'",
                    )
                })?;
            if !rate_per_s.is_finite() || rate_per_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic self-similar arrival rate must be finite and positive",
                ));
            }

            let pareto_shape = traffic.self_similar_pareto_shape.unwrap_or(1.4);
            if !pareto_shape.is_finite() || pareto_shape <= 1.0 {
                return Err(ConfigError::new(
                    "serving.traffic.self_similar_pareto_shape must be finite and greater than 1.0",
                ));
            }

            let max_gap_s = parse_optional_trace_seconds(
                "serving.traffic.self_similar_max_gap_s",
                traffic.self_similar_max_gap_s,
                "serving.traffic.self_similar_max_gap_ms",
                traffic.self_similar_max_gap_ms,
            )?;
            if max_gap_s.is_some_and(|gap| gap <= 0.0) {
                return Err(ConfigError::new(
                    "serving.traffic.self_similar_max_gap_s/self_similar_max_gap_ms must be positive",
                ));
            }

            Ok(ServingArrivalPattern::SelfSimilar {
                rate_per_s,
                pareto_shape,
                max_gap_s,
                seed: traffic.arrival_seed.unwrap_or(1),
            })
        }
        Some("trace_derived") | Some("trace_arrivals") | Some("trace_arrival") => {
            Ok(ServingArrivalPattern::TraceDerived)
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.arrival '{value}'; use fixed_gap, poisson, bursty, diurnal, self_similar, or trace_derived"
        ))),
    }
}

pub(super) fn positive_values(name: &str, values: Vec<u32>) -> Result<Vec<u32>, ConfigError> {
    if values.contains(&0) {
        return Err(ConfigError::new(format!(
            "{name} values must be greater than zero"
        )));
    }

    Ok(values)
}
