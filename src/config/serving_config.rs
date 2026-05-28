use super::*;

mod pools;
mod traffic;
pub(super) use pools::*;
pub(super) use traffic::*;

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
