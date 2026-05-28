use super::*;

pub(in crate::config) fn parse_serving_traffic(
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
pub(in crate::config) struct PartialServingServicesConfig {
    prefill: Option<ServingServicePhaseConfig>,
    decode: Option<ServingServicePhaseConfig>,
    kv_transfer: Option<ServingServicePhaseConfig>,
}

pub(in crate::config) fn parse_serving_services(
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

pub(in crate::config) fn parse_serving_service_phase(
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

pub(in crate::config) fn parse_serving_service_health(
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

pub(in crate::config) fn parse_serving_service_worker_scale(
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

pub(in crate::config) fn merge_serving_services(
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

pub(in crate::config) fn finalize_serving_services(
    partial: PartialServingServicesConfig,
) -> ServingServicesConfig {
    ServingServicesConfig {
        prefill: partial.prefill.unwrap_or_default(),
        decode: partial.decode.unwrap_or_default(),
        kv_transfer: partial.kv_transfer.unwrap_or_default(),
    }
}

pub(in crate::config) fn parse_positive_optional_seconds(
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

pub(in crate::config) fn parse_value_distribution(
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

pub(in crate::config) fn parse_shape_profiles(
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

pub(in crate::config) fn parse_shape_profile_slo(
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

pub(in crate::config) fn parse_shape_profile_slo_value(
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

pub(in crate::config) fn parse_routing_policy(
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

pub(in crate::config) fn parse_slo_ms(
    name: &str,
    value: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
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

pub(in crate::config) fn parse_cache_hit_rate(
    name: &str,
    value: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    parse_fraction(name, value)
}

pub(in crate::config) fn parse_fraction(
    name: &str,
    value: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
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

pub(in crate::config) fn parse_non_negative_f64(
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

pub(in crate::config) fn parse_prefill_batching(
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

pub(in crate::config) fn parse_decode_batching(
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

pub(in crate::config) fn parse_decode_capacity_policy(
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

pub(in crate::config) fn parse_arrival_pattern(
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

pub(in crate::config) fn positive_values(
    name: &str,
    values: Vec<u32>,
) -> Result<Vec<u32>, ConfigError> {
    if values.contains(&0) {
        return Err(ConfigError::new(format!(
            "{name} values must be greater than zero"
        )));
    }

    Ok(values)
}
