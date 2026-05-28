use super::*;

pub(super) fn parse_calibration_policy(
    section: Option<CalibrationPolicySection>,
) -> Result<CalibrationPolicy, ConfigError> {
    let Some(section) = section else {
        return Ok(CalibrationPolicy::default());
    };
    let defaults = CalibrationPolicy::default();
    let min_coverage_score = section.min_coverage_score.or(defaults.min_coverage_score);
    if let Some(score) = min_coverage_score
        && (!score.is_finite() || !(0.0..=1.0).contains(&score))
    {
        return Err(ConfigError::new(
            "calibration_policy.min_coverage_score must be finite and between 0.0 and 1.0",
        ));
    }
    let min_fit_confidence_score = section
        .min_fit_confidence_score
        .or(defaults.min_fit_confidence_score);
    if let Some(score) = min_fit_confidence_score
        && (!score.is_finite() || !(0.0..=1.0).contains(&score))
    {
        return Err(ConfigError::new(
            "calibration_policy.min_fit_confidence_score must be finite and between 0.0 and 1.0",
        ));
    }
    let min_fit_sample_count = section
        .min_fit_sample_count
        .or(defaults.min_fit_sample_count);
    validate_positive_optional_u32(
        "calibration_policy.min_fit_sample_count",
        min_fit_sample_count,
    )?;
    let min_fit_validation_sample_count = section
        .min_fit_validation_sample_count
        .or(defaults.min_fit_validation_sample_count);
    validate_positive_optional_u32(
        "calibration_policy.min_fit_validation_sample_count",
        min_fit_validation_sample_count,
    )?;
    let min_fit_confidence_level = parse_fraction(
        "calibration_policy.min_fit_confidence_level",
        section
            .min_fit_confidence_level
            .or(defaults.min_fit_confidence_level),
    )?;
    let max_fit_relative_uncertainty_pct = parse_non_negative_f64(
        "calibration_policy.max_fit_relative_uncertainty_pct",
        section
            .max_fit_relative_uncertainty_pct
            .or(defaults.max_fit_relative_uncertainty_pct),
    )?;
    let max_fit_absolute_uncertainty_ms = parse_non_negative_f64(
        "calibration_policy.max_fit_absolute_uncertainty_ms",
        section.max_fit_absolute_uncertainty_ms,
    )?;
    let max_fit_absolute_uncertainty_s = parse_non_negative_f64(
        "calibration_policy.max_fit_absolute_uncertainty_s",
        section
            .max_fit_absolute_uncertainty_s
            .or(defaults.max_fit_absolute_uncertainty_s),
    )?;
    if max_fit_absolute_uncertainty_ms.is_some() && max_fit_absolute_uncertainty_s.is_some() {
        return Err(ConfigError::new(
            "calibration_policy must set only one of max_fit_absolute_uncertainty_ms or max_fit_absolute_uncertainty_s",
        ));
    }
    let max_fit_absolute_uncertainty_s = max_fit_absolute_uncertainty_ms
        .map(|ms| ms / 1000.0)
        .or(max_fit_absolute_uncertainty_s);
    let min_serving_phase_coverage_fraction = parse_fraction(
        "calibration_policy.min_serving_phase_coverage_fraction",
        section
            .min_serving_phase_coverage_fraction
            .or(defaults.min_serving_phase_coverage_fraction),
    )?;
    let uncertainty_ranking_weight = section
        .uncertainty_ranking_weight
        .unwrap_or(defaults.uncertainty_ranking_weight);
    if !uncertainty_ranking_weight.is_finite() || uncertainty_ranking_weight < 0.0 {
        return Err(ConfigError::new(
            "calibration_policy.uncertainty_ranking_weight must be finite and nonnegative",
        ));
    }

    Ok(CalibrationPolicy {
        valid_shape: parse_calibration_gate_mode(
            "calibration_policy.valid_shape",
            section.valid_shape.as_deref(),
        )?,
        invalid_shape: parse_calibration_gate_mode(
            "calibration_policy.invalid_shape",
            section.invalid_shape.as_deref(),
        )?,
        coverage: parse_calibration_gate_mode(
            "calibration_policy.coverage",
            section.coverage.as_deref(),
        )?,
        fit_confidence: parse_calibration_gate_mode(
            "calibration_policy.fit_confidence",
            section.fit_confidence.as_deref(),
        )?,
        fit_extrapolation: parse_calibration_gate_mode(
            "calibration_policy.fit_extrapolation",
            section.fit_extrapolation.as_deref(),
        )?,
        fit_partially_bounded: parse_calibration_gate_mode(
            "calibration_policy.fit_partially_bounded",
            section.fit_partially_bounded.as_deref(),
        )?,
        fit_unbounded: parse_calibration_gate_mode(
            "calibration_policy.fit_unbounded",
            section.fit_unbounded.as_deref(),
        )?,
        fit_sample_count: parse_calibration_gate_mode(
            "calibration_policy.fit_sample_count",
            section.fit_sample_count.as_deref(),
        )?,
        fit_validation_sample_count: parse_calibration_gate_mode(
            "calibration_policy.fit_validation_sample_count",
            section.fit_validation_sample_count.as_deref(),
        )?,
        fit_source: parse_calibration_gate_mode(
            "calibration_policy.fit_source",
            section.fit_source.as_deref(),
        )?,
        fit_uncertainty: parse_calibration_gate_mode(
            "calibration_policy.fit_uncertainty",
            section.fit_uncertainty.as_deref(),
        )?,
        profile_source: parse_calibration_gate_mode(
            "calibration_policy.profile_source",
            section.profile_source.as_deref(),
        )?,
        profile_date: parse_calibration_gate_mode(
            "calibration_policy.profile_date",
            section.profile_date.as_deref(),
        )?,
        profile_runtime: parse_calibration_gate_mode(
            "calibration_policy.profile_runtime",
            section.profile_runtime.as_deref(),
        )?,
        min_coverage_score,
        min_fit_confidence_score,
        min_fit_confidence_level,
        min_fit_sample_count,
        min_fit_validation_sample_count,
        max_fit_relative_uncertainty_pct,
        max_fit_absolute_uncertainty_s,
        min_serving_phase_coverage_fraction,
        uncertainty_ranking_weight,
        require_phase_coverage: section
            .require_phase_coverage
            .unwrap_or(defaults.require_phase_coverage),
    })
}

pub(super) fn parse_calibration_gate_mode(
    name: &str,
    value: Option<&str>,
) -> Result<CalibrationGateMode, ConfigError> {
    match value.map(normalize).as_deref() {
        None | Some("warn") | Some("warning") | Some("soft") | Some("soft_fail") => {
            Ok(CalibrationGateMode::Warn)
        }
        Some("reject") | Some("hard") | Some("hard_fail") | Some("fail") => {
            Ok(CalibrationGateMode::Reject)
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported {name} '{value}'; use warn or reject"
        ))),
    }
}

pub(super) fn parse_approximation_policy(
    section: Option<ApproximationPolicySection>,
) -> Result<ApproximationPolicy, ConfigError> {
    let Some(section) = section else {
        return Ok(ApproximationPolicy::default());
    };
    let preset = section
        .preset
        .as_deref()
        .map(parse_approximation_policy_preset)
        .transpose()?;
    let defaults = preset
        .map(approximation_policy_preset_defaults)
        .unwrap_or_default();
    Ok(ApproximationPolicy {
        preset,
        default_action: match section.default_action.as_deref() {
            Some(value) => {
                parse_calibration_gate_mode("approximation_policy.default_action", Some(value))?
            }
            None => defaults.default_action,
        },
        reject_categories: match section.reject_categories {
            Some(values) => {
                normalized_policy_terms("approximation_policy.reject_categories", values)?
            }
            None => defaults.reject_categories,
        },
        reject_codes: match section.reject_codes {
            Some(values) => normalized_policy_terms("approximation_policy.reject_codes", values)?,
            None => defaults.reject_codes,
        },
        warn_categories: match section.warn_categories {
            Some(values) => {
                normalized_policy_terms("approximation_policy.warn_categories", values)?
            }
            None => defaults.warn_categories,
        },
        warn_codes: match section.warn_codes {
            Some(values) => normalized_policy_terms("approximation_policy.warn_codes", values)?,
            None => defaults.warn_codes,
        },
        metric_gates: if section.metric_gates.is_empty() {
            defaults.metric_gates
        } else {
            parse_approximation_metric_gates(section.metric_gates)?
        },
    })
}

pub(super) fn parse_approximation_metric_gates(
    gates: Vec<ApproximationMetricGateSection>,
) -> Result<Vec<ApproximationMetricGate>, ConfigError> {
    let mut parsed = Vec::new();
    for (idx, gate) in gates.into_iter().enumerate() {
        let mut metrics = Vec::new();
        if let Some(metric) = gate.metric {
            metrics.push(metric);
        }
        metrics.extend(gate.metrics.unwrap_or_default());
        if let Some(objective) = gate.objective {
            metrics.push(objective);
        }
        metrics.extend(gate.objectives.unwrap_or_default());
        let metrics = normalized_approximation_metric_terms(
            &format!("approximation_policy.metric_gates[{idx}].metrics"),
            metrics,
        )?;
        if metrics.is_empty() {
            return Err(ConfigError::new(format!(
                "approximation_policy.metric_gates[{idx}] requires metric, metrics, objective, or objectives"
            )));
        }
        let reject_categories = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].reject_categories"),
            gate.reject_categories.unwrap_or_default(),
        )?;
        let reject_codes = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].reject_codes"),
            gate.reject_codes.unwrap_or_default(),
        )?;
        let warn_categories = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].warn_categories"),
            gate.warn_categories.unwrap_or_default(),
        )?;
        let warn_codes = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].warn_codes"),
            gate.warn_codes.unwrap_or_default(),
        )?;
        if reject_categories.is_empty()
            && reject_codes.is_empty()
            && warn_categories.is_empty()
            && warn_codes.is_empty()
        {
            return Err(ConfigError::new(format!(
                "approximation_policy.metric_gates[{idx}] must set at least one reject_categories, reject_codes, warn_categories, or warn_codes entry"
            )));
        }
        parsed.push(ApproximationMetricGate {
            metrics,
            reject_categories,
            reject_codes,
            warn_categories,
            warn_codes,
        });
    }
    Ok(parsed)
}

pub(super) fn normalized_approximation_metric_terms(
    name: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ConfigError> {
    let mut terms = Vec::new();
    for value in values {
        let term = approximation_metric_term(&value).ok_or_else(|| {
            ConfigError::new(format!(
                "unsupported {name} value '{value}'; use ttft, tpot, throughput, e2el, slo_miss_rate, memory_pressure, cost, energy, or power"
            ))
        })?;
        if !terms.contains(&term) {
            terms.push(term);
        }
    }
    Ok(terms)
}

pub(super) fn approximation_metric_term(value: &str) -> Option<String> {
    let term = normalize(value);
    match term.as_str() {
        "ttft" | "time_to_first_token" | "minimize_ttft" => Some("ttft".to_string()),
        "tpot" | "time_per_output_token" | "minimize_tpot" => Some("tpot".to_string()),
        "throughput"
        | "tokens_per_s"
        | "tokens_per_second"
        | "max_throughput"
        | "maximize_throughput" => Some("throughput".to_string()),
        "e2el" | "latency" | "end_to_end" | "end_to_end_latency" | "minimize_e2el" => {
            Some("e2el".to_string())
        }
        "slo" | "slo_miss" | "slo_miss_rate" | "minimize_slo_miss_rate" => {
            Some("slo_miss_rate".to_string())
        }
        "memory"
        | "hbm"
        | "memory_pressure"
        | "hbm_pressure"
        | "minimize_memory_pressure"
        | "minimize_hbm_pressure" => Some("memory_pressure".to_string()),
        "cost"
        | "total_cost"
        | "cost_usd"
        | "minimize_cost"
        | "minimize_total_cost"
        | "minimize_cost_usd" => Some("cost".to_string()),
        "energy" | "kwh" | "energy_kwh" | "minimize_energy" | "minimize_energy_kwh" => {
            Some("energy".to_string())
        }
        "power"
        | "watts"
        | "average_power"
        | "average_power_watts"
        | "minimize_power"
        | "minimize_average_power"
        | "minimize_average_power_watts" => Some("power".to_string()),
        _ => None,
    }
}

pub(super) fn parse_approximation_policy_preset(
    value: &str,
) -> Result<ApproximationPolicyPreset, ConfigError> {
    match normalize(value).as_str() {
        "mvp" | "exploration" | "mvp_exploration" => Ok(ApproximationPolicyPreset::MvpExploration),
        "topology" | "topology_sensitive" | "topology_sensitive_planning" => {
            Ok(ApproximationPolicyPreset::TopologySensitive)
        }
        "memory" | "capacity" | "memory_capacity" | "memory_capacity_planning" => {
            Ok(ApproximationPolicyPreset::MemoryCapacity)
        }
        "calibration" | "calibrated" | "calibration_only" | "calibrated_only" => {
            Ok(ApproximationPolicyPreset::CalibrationOnly)
        }
        "production" | "strict" | "production_recommendation" => {
            Ok(ApproximationPolicyPreset::ProductionRecommendation)
        }
        value => Err(ConfigError::new(format!(
            "unsupported approximation_policy.preset '{value}'; use mvp_exploration, topology_sensitive, memory_capacity, calibration_only, or production_recommendation"
        ))),
    }
}

pub(super) fn approximation_policy_preset_defaults(
    preset: ApproximationPolicyPreset,
) -> ApproximationPolicy {
    match preset {
        ApproximationPolicyPreset::MvpExploration => ApproximationPolicy {
            preset: Some(preset),
            ..ApproximationPolicy::default()
        },
        ApproximationPolicyPreset::TopologySensitive => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Warn,
            reject_categories: vec![
                "topology".to_string(),
                "routing".to_string(),
                "communication".to_string(),
            ],
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
        ApproximationPolicyPreset::MemoryCapacity => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Warn,
            reject_categories: vec!["memory".to_string(), "capacity".to_string()],
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
        ApproximationPolicyPreset::CalibrationOnly => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Warn,
            reject_categories: vec!["calibration".to_string(), "runtime".to_string()],
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
        ApproximationPolicyPreset::ProductionRecommendation => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Reject,
            reject_categories: Vec::new(),
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
    }
}

pub(super) fn normalized_policy_terms(
    name: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ConfigError> {
    let mut terms = Vec::new();
    for value in values {
        let term = normalize(&value);
        if term.is_empty() {
            return Err(ConfigError::new(format!("{name} values must not be empty")));
        }
        if !terms.contains(&term) {
            terms.push(term);
        }
    }
    Ok(terms)
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedCalibrationProfile {
    pub metadata: CalibrationProfileMetadata,
    pub calibration: SimulationCalibration,
}

pub(super) fn load_calibration_profile(
    reference: Option<CalibrationProfileReferenceSection>,
    base_dir: Option<&Path>,
) -> Result<Option<LoadedCalibrationProfile>, ConfigError> {
    let Some(reference) = reference else {
        return Ok(None);
    };
    if reference.path.trim().is_empty() {
        return Err(ConfigError::new(
            "calibration_profile.path must not be empty",
        ));
    }
    let path = resolve_config_path(&reference.path, base_dir);
    Ok(Some(load_calibration_profile_path(&path)?))
}

pub fn load_calibration_profile_path(path: &Path) -> Result<LoadedCalibrationProfile, ConfigError> {
    let contents = fs::read_to_string(path).map_err(|err| {
        ConfigError::new(format!(
            "failed to read calibration_profile.path {}: {err}",
            path.display()
        ))
    })?;
    let profile_file: CalibrationProfileFile = toml::from_str(&contents).map_err(|err| {
        ConfigError::new(format!(
            "invalid calibration profile TOML {}: {err}",
            path.display()
        ))
    })?;
    validate_schema_version("calibration profile", profile_file.schema_version)?;
    let metadata = calibration_profile_metadata(
        path.to_path_buf(),
        profile_file.profile,
        profile_file.valid_shape,
        profile_file.invalid_shapes.unwrap_or_default(),
        profile_file.fits.unwrap_or_default(),
        profile_file.benchmarks.unwrap_or_default(),
    )?;
    let calibration =
        calibration_with_defaults(profile_file.calibration, SimulationCalibration::default());
    Ok(LoadedCalibrationProfile {
        metadata,
        calibration,
    })
}

pub(super) fn calibration_profile_metadata(
    path: PathBuf,
    section: Option<CalibrationProfileMetadataSection>,
    valid_shape: Option<CalibrationShapeRangeSection>,
    invalid_shapes: Vec<CalibrationInvalidShapeRangeSection>,
    fits: Vec<CalibrationFittedModelSection>,
    benchmarks: Vec<CalibrationBenchmarkPointSection>,
) -> Result<CalibrationProfileMetadata, ConfigError> {
    let section = section.unwrap_or(CalibrationProfileMetadataSection {
        name: None,
        hardware: None,
        fabric: None,
        model: None,
        dtype: None,
        serving_stack: None,
        serving_runtime_features: None,
        backend_version: None,
        driver_version: None,
        cuda_version: None,
        rocm_version: None,
        nccl_version: None,
        rccl_version: None,
        ucx_version: None,
        kernel_settings: Vec::new(),
        environment_hash: None,
        source: None,
        date: None,
        notes: None,
    });
    Ok(CalibrationProfileMetadata {
        path: path.display().to_string(),
        name: nonempty_metadata(section.name),
        hardware: nonempty_metadata(section.hardware),
        fabric: nonempty_metadata(section.fabric),
        model: nonempty_metadata(section.model),
        dtype: nonempty_metadata(section.dtype),
        serving_stack: nonempty_metadata(section.serving_stack),
        serving_runtime_features: section
            .serving_runtime_features
            .map(|features| normalized_group_labels("profile.serving_runtime_features", features))
            .transpose()?
            .unwrap_or_default(),
        backend_version: nonempty_metadata(section.backend_version),
        driver_version: nonempty_metadata(section.driver_version),
        cuda_version: nonempty_metadata(section.cuda_version),
        rocm_version: nonempty_metadata(section.rocm_version),
        nccl_version: nonempty_metadata(section.nccl_version),
        rccl_version: nonempty_metadata(section.rccl_version),
        ucx_version: nonempty_metadata(section.ucx_version),
        kernel_settings: nonempty_string_values(
            "profile.kernel_settings",
            section.kernel_settings,
        )?,
        environment_hash: nonempty_metadata(section.environment_hash),
        source: nonempty_metadata(section.source),
        date: nonempty_metadata(section.date),
        notes: nonempty_metadata(section.notes),
        valid_shape: parse_calibration_shape_range(valid_shape)?,
        invalid_shapes: parse_calibration_invalid_shape_ranges(invalid_shapes)?,
        fits: parse_calibration_fits(fits)?,
        benchmarks: parse_calibration_benchmarks(benchmarks)?,
    })
}

pub(super) fn nonempty_metadata(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

pub(super) fn parse_calibration_shape_range(
    section: Option<CalibrationShapeRangeSection>,
) -> Result<Option<CalibrationShapeRange>, ConfigError> {
    let Some(section) = section else {
        return Ok(None);
    };
    validate_min_max(
        "valid_shape.batch_size",
        section.min_batch_size,
        section.max_batch_size,
    )?;
    validate_min_max(
        "valid_shape.prompt_tokens",
        section.min_prompt_tokens,
        section.max_prompt_tokens,
    )?;
    validate_min_max(
        "valid_shape.decode_tokens",
        section.min_decode_tokens,
        section.max_decode_tokens,
    )?;
    validate_min_max(
        "valid_shape.sequence_tokens",
        section.min_sequence_tokens,
        section.max_sequence_tokens,
    )?;

    Ok(Some(CalibrationShapeRange {
        min_batch_size: section.min_batch_size,
        max_batch_size: section.max_batch_size,
        min_prompt_tokens: section.min_prompt_tokens,
        max_prompt_tokens: section.max_prompt_tokens,
        min_decode_tokens: section.min_decode_tokens,
        max_decode_tokens: section.max_decode_tokens,
        min_sequence_tokens: section.min_sequence_tokens,
        max_sequence_tokens: section.max_sequence_tokens,
    }))
}

pub(super) fn parse_calibration_invalid_shape_ranges(
    invalid_shapes: Vec<CalibrationInvalidShapeRangeSection>,
) -> Result<Vec<CalibrationInvalidShapeRange>, ConfigError> {
    invalid_shapes
        .into_iter()
        .enumerate()
        .map(|(idx, section)| parse_calibration_invalid_shape_range(idx, section))
        .collect()
}

pub(super) fn parse_calibration_invalid_shape_range(
    idx: usize,
    section: CalibrationInvalidShapeRangeSection,
) -> Result<CalibrationInvalidShapeRange, ConfigError> {
    validate_min_max(
        &format!("invalid_shapes[{idx}].batch_size"),
        section.min_batch_size,
        section.max_batch_size,
    )?;
    validate_min_max(
        &format!("invalid_shapes[{idx}].prompt_tokens"),
        section.min_prompt_tokens,
        section.max_prompt_tokens,
    )?;
    validate_min_max(
        &format!("invalid_shapes[{idx}].decode_tokens"),
        section.min_decode_tokens,
        section.max_decode_tokens,
    )?;
    validate_min_max(
        &format!("invalid_shapes[{idx}].sequence_tokens"),
        section.min_sequence_tokens,
        section.max_sequence_tokens,
    )?;
    let shape = CalibrationShapeRange {
        min_batch_size: section.min_batch_size,
        max_batch_size: section.max_batch_size,
        min_prompt_tokens: section.min_prompt_tokens,
        max_prompt_tokens: section.max_prompt_tokens,
        min_decode_tokens: section.min_decode_tokens,
        max_decode_tokens: section.max_decode_tokens,
        min_sequence_tokens: section.min_sequence_tokens,
        max_sequence_tokens: section.max_sequence_tokens,
    };
    if !shape_range_has_any_bound(&shape) {
        return Err(ConfigError::new(format!(
            "invalid_shapes[{idx}] must set at least one shape bound"
        )));
    }

    Ok(CalibrationInvalidShapeRange {
        name: nonempty_metadata(section.name),
        reason: nonempty_metadata(section.reason),
        shape,
    })
}

pub(super) fn shape_range_has_any_bound(shape: &CalibrationShapeRange) -> bool {
    shape.min_batch_size.is_some()
        || shape.max_batch_size.is_some()
        || shape.min_prompt_tokens.is_some()
        || shape.max_prompt_tokens.is_some()
        || shape.min_decode_tokens.is_some()
        || shape.max_decode_tokens.is_some()
        || shape.min_sequence_tokens.is_some()
        || shape.max_sequence_tokens.is_some()
}

pub(super) fn parse_calibration_fits(
    fits: Vec<CalibrationFittedModelSection>,
) -> Result<Vec<CalibrationFittedModel>, ConfigError> {
    fits.into_iter()
        .enumerate()
        .map(|(idx, fit)| parse_calibration_fit(idx, fit))
        .collect()
}

pub(super) fn parse_calibration_fit(
    idx: usize,
    fit: CalibrationFittedModelSection,
) -> Result<CalibrationFittedModel, ConfigError> {
    let target = required_nonempty_metadata(&format!("fits[{idx}].target"), fit.target)?;
    let model = required_nonempty_metadata(&format!("fits[{idx}].model"), fit.model)?;
    let features = nonempty_string_values(&format!("fits[{idx}].features"), fit.features)?;
    if features.is_empty() {
        return Err(ConfigError::new(format!(
            "fits[{idx}].features must not be empty"
        )));
    }
    if fit.coefficients.is_empty() {
        return Err(ConfigError::new(format!(
            "fits[{idx}].coefficients must not be empty"
        )));
    }
    if features.len() != fit.coefficients.len() {
        return Err(ConfigError::new(format!(
            "fits[{idx}].features and coefficients must have the same length"
        )));
    }
    let feature_ranges = parse_calibration_fit_feature_ranges(idx, &features, fit.feature_ranges)?;
    validate_optional_finite_f64(&format!("fits[{idx}].intercept"), fit.intercept)?;
    for (coefficient_idx, coefficient) in fit.coefficients.iter().enumerate() {
        validate_finite_f64(
            &format!("fits[{idx}].coefficients[{coefficient_idx}]"),
            *coefficient,
        )?;
    }
    validate_optional_finite_f64(&format!("fits[{idx}].r_squared"), fit.r_squared)?;
    validate_optional_finite_f64(
        &format!("fits[{idx}].adjusted_r_squared"),
        fit.adjusted_r_squared,
    )?;
    validate_nonnegative_optional_f64(&format!("fits[{idx}].rmse"), fit.rmse)?;
    validate_nonnegative_optional_f64(&format!("fits[{idx}].rmse_pct"), fit.rmse_pct)?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].mean_abs_pct_error"),
        fit.mean_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].max_abs_pct_error"),
        fit.max_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_rmse"),
        fit.validation_rmse,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_rmse_pct"),
        fit.validation_rmse_pct,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_mean_abs_pct_error"),
        fit.validation_mean_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_max_abs_pct_error"),
        fit.validation_max_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].confidence_interval"),
        fit.confidence_interval,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].confidence_interval_pct"),
        fit.confidence_interval_pct,
    )?;
    validate_positive_fraction_optional_f64(
        &format!("fits[{idx}].confidence_level"),
        fit.confidence_level,
    )?;
    validate_positive_optional_u32(&format!("fits[{idx}].sample_count"), fit.sample_count)?;
    validate_positive_optional_u32(
        &format!("fits[{idx}].validation_sample_count"),
        fit.validation_sample_count,
    )?;

    Ok(CalibrationFittedModel {
        name: nonempty_metadata(fit.name),
        target,
        phase: nonempty_metadata(fit.phase).map(|phase| normalize(&phase)),
        kind: nonempty_metadata(fit.kind).map(|kind| normalize(&kind)),
        model,
        unit: nonempty_metadata(fit.unit),
        intercept: fit.intercept,
        features,
        coefficients: fit.coefficients,
        feature_ranges,
        r_squared: fit.r_squared,
        adjusted_r_squared: fit.adjusted_r_squared,
        rmse: fit.rmse,
        rmse_pct: fit.rmse_pct,
        mean_abs_pct_error: fit.mean_abs_pct_error,
        max_abs_pct_error: fit.max_abs_pct_error,
        validation_rmse: fit.validation_rmse,
        validation_rmse_pct: fit.validation_rmse_pct,
        validation_mean_abs_pct_error: fit.validation_mean_abs_pct_error,
        validation_max_abs_pct_error: fit.validation_max_abs_pct_error,
        confidence_interval: fit.confidence_interval,
        confidence_interval_pct: fit.confidence_interval_pct,
        confidence_level: fit.confidence_level,
        sample_count: fit.sample_count,
        validation_sample_count: fit.validation_sample_count,
        source: nonempty_metadata(fit.source),
        notes: nonempty_metadata(fit.notes),
    })
}

pub(super) fn parse_calibration_fit_feature_ranges(
    fit_idx: usize,
    features: &[String],
    ranges: Vec<CalibrationFitFeatureRangeSection>,
) -> Result<Vec<CalibrationFitFeatureRange>, ConfigError> {
    let feature_names = features
        .iter()
        .map(|feature| normalize(feature))
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    ranges
        .into_iter()
        .enumerate()
        .map(|(range_idx, range)| {
            let name = format!("fits[{fit_idx}].feature_ranges[{range_idx}]");
            let feature = required_nonempty_metadata(&format!("{name}.feature"), range.feature)?;
            let normalized_feature = normalize(&feature);
            if !feature_names.contains(&normalized_feature) {
                return Err(ConfigError::new(format!(
                    "{name}.feature must match one of fits[{fit_idx}].features"
                )));
            }
            if !seen.insert(normalized_feature) {
                return Err(ConfigError::new(format!(
                    "{name}.feature duplicates an earlier feature range"
                )));
            }
            validate_optional_finite_f64(&format!("{name}.min"), range.min)?;
            validate_optional_finite_f64(&format!("{name}.max"), range.max)?;
            if let (Some(min), Some(max)) = (range.min, range.max)
                && min > max
            {
                return Err(ConfigError::new(format!("{name}.min must be <= max")));
            }
            if range.min.is_none() && range.max.is_none() {
                return Err(ConfigError::new(format!(
                    "{name} must set at least one range bound"
                )));
            }

            Ok(CalibrationFitFeatureRange {
                feature,
                min: range.min,
                max: range.max,
            })
        })
        .collect()
}

pub(super) fn required_nonempty_metadata(
    name: &str,
    value: Option<String>,
) -> Result<String, ConfigError> {
    nonempty_metadata(value).ok_or_else(|| ConfigError::new(format!("{name} must not be empty")))
}

pub(super) fn nonempty_string_values(
    name: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ConfigError> {
    let mut parsed = Vec::new();
    for value in values {
        let value = value.trim().to_string();
        if value.is_empty() {
            return Err(ConfigError::new(format!("{name} values must not be empty")));
        }
        parsed.push(value);
    }
    Ok(parsed)
}

pub(super) fn parse_calibration_benchmarks(
    benchmarks: Vec<CalibrationBenchmarkPointSection>,
) -> Result<Vec<CalibrationBenchmarkPoint>, ConfigError> {
    benchmarks
        .into_iter()
        .enumerate()
        .map(|(idx, benchmark)| parse_calibration_benchmark(idx, benchmark))
        .collect()
}

pub(super) fn parse_calibration_benchmark(
    idx: usize,
    benchmark: CalibrationBenchmarkPointSection,
) -> Result<CalibrationBenchmarkPoint, ConfigError> {
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].batch_size"),
        benchmark.batch_size,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].prompt_tokens"),
        benchmark.prompt_tokens,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].decode_tokens"),
        benchmark.decode_tokens,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].sequence_tokens"),
        benchmark.sequence_tokens,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].tensor_ranks"),
        benchmark.tensor_ranks,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].pipeline_ranks"),
        benchmark.pipeline_ranks,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].expert_ranks"),
        benchmark.expert_ranks,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].data_ranks"),
        benchmark.data_ranks,
    )?;
    validate_positive_optional_f64(
        &format!("benchmarks[{idx}].measured_ms"),
        benchmark.measured_ms,
    )?;
    validate_positive_optional_f64(
        &format!("benchmarks[{idx}].predicted_ms"),
        benchmark.predicted_ms,
    )?;
    validate_positive_optional_f64(
        &format!("benchmarks[{idx}].throughput_tokens_per_s"),
        benchmark.throughput_tokens_per_s,
    )?;

    Ok(CalibrationBenchmarkPoint {
        name: nonempty_metadata(benchmark.name),
        kind: nonempty_metadata(benchmark.kind),
        phase: nonempty_metadata(benchmark.phase),
        hardware: nonempty_metadata(benchmark.hardware),
        fabric: nonempty_metadata(benchmark.fabric),
        model: nonempty_metadata(benchmark.model),
        dtype: nonempty_metadata(benchmark.dtype),
        batch_size: benchmark.batch_size,
        prompt_tokens: benchmark.prompt_tokens,
        decode_tokens: benchmark.decode_tokens,
        sequence_tokens: benchmark.sequence_tokens,
        tensor_ranks: benchmark.tensor_ranks,
        pipeline_ranks: benchmark.pipeline_ranks,
        expert_ranks: benchmark.expert_ranks,
        data_ranks: benchmark.data_ranks,
        measured_ms: benchmark.measured_ms,
        predicted_ms: benchmark.predicted_ms,
        throughput_tokens_per_s: benchmark.throughput_tokens_per_s,
        command: nonempty_metadata(benchmark.command),
        source: nonempty_metadata(benchmark.source),
        notes: nonempty_metadata(benchmark.notes),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CalibrationShapeEnvelope {
    min_batch_size: u32,
    max_batch_size: u32,
    min_prompt_tokens: u32,
    max_prompt_tokens: u32,
    min_decode_tokens: u32,
    max_decode_tokens: u32,
    min_sequence_tokens: u32,
    max_sequence_tokens: u32,
}

impl CalibrationShapeEnvelope {
    fn from_request(request: &InferenceRequest) -> Self {
        let sequence_tokens = request
            .max_sequence_tokens
            .max(request.prompt_tokens.saturating_add(request.decode_tokens))
            .max(1);
        Self {
            min_batch_size: request.batch_size.max(1),
            max_batch_size: request.batch_size.max(1),
            min_prompt_tokens: request.prompt_tokens.max(1),
            max_prompt_tokens: request.prompt_tokens.max(1),
            min_decode_tokens: request.decode_tokens.max(1),
            max_decode_tokens: request.decode_tokens.max(1),
            min_sequence_tokens: sequence_tokens,
            max_sequence_tokens: sequence_tokens,
        }
    }

    fn include_serving(&mut self, serving: &DisaggregatedServingConfig, base: &InferenceRequest) {
        let traffic = &serving.traffic;
        for request in &traffic.trace_requests {
            self.include_trace_request(request, base);
        }
        self.include_values(ShapeField::BatchSize, &traffic.batch_sizes);
        self.include_values(ShapeField::PromptTokens, &traffic.prompt_tokens);
        self.include_values(ShapeField::DecodeTokens, &traffic.decode_tokens);
        self.include_distribution(
            ShapeField::BatchSize,
            traffic.batch_size_distribution.as_ref(),
        );
        self.include_distribution(
            ShapeField::PromptTokens,
            traffic.prompt_tokens_distribution.as_ref(),
        );
        self.include_distribution(
            ShapeField::DecodeTokens,
            traffic.decode_tokens_distribution.as_ref(),
        );
        for profile in &traffic.shape_profiles {
            let sequence_tokens = profile
                .max_sequence_tokens
                .unwrap_or(base.max_sequence_tokens)
                .max(profile.prompt_tokens.saturating_add(profile.decode_tokens))
                .max(1);
            self.include_request_shape(
                profile.batch_size,
                profile.prompt_tokens,
                profile.decode_tokens,
                sequence_tokens,
            );
        }
    }

    fn include_trace_request(&mut self, request: &ServingTraceRequest, base: &InferenceRequest) {
        let sequence_tokens = request
            .max_sequence_tokens
            .unwrap_or(base.max_sequence_tokens)
            .max(request.prompt_tokens.saturating_add(request.decode_tokens))
            .max(1);
        self.include_request_shape(
            request.batch_size,
            request.prompt_tokens,
            request.decode_tokens,
            sequence_tokens,
        );
    }

    fn include_request_shape(
        &mut self,
        batch_size: u32,
        prompt_tokens: u32,
        decode_tokens: u32,
        sequence_tokens: u32,
    ) {
        self.include_value(ShapeField::BatchSize, batch_size);
        self.include_value(ShapeField::PromptTokens, prompt_tokens);
        self.include_value(ShapeField::DecodeTokens, decode_tokens);
        self.include_value(ShapeField::SequenceTokens, sequence_tokens);
    }

    fn include_values(&mut self, field: ShapeField, values: &[u32]) {
        for value in values {
            self.include_value(field, *value);
        }
    }

    fn include_distribution(
        &mut self,
        field: ShapeField,
        distribution: Option<&ServingValueDistribution>,
    ) {
        let Some((min, max)) = distribution.and_then(serving_distribution_range) else {
            return;
        };
        self.include_value(field, min);
        self.include_value(field, max);
    }

    fn include_value(&mut self, field: ShapeField, value: u32) {
        let value = value.max(1);
        match field {
            ShapeField::BatchSize => {
                self.min_batch_size = self.min_batch_size.min(value);
                self.max_batch_size = self.max_batch_size.max(value);
            }
            ShapeField::PromptTokens => {
                self.min_prompt_tokens = self.min_prompt_tokens.min(value);
                self.max_prompt_tokens = self.max_prompt_tokens.max(value);
            }
            ShapeField::DecodeTokens => {
                self.min_decode_tokens = self.min_decode_tokens.min(value);
                self.max_decode_tokens = self.max_decode_tokens.max(value);
            }
            ShapeField::SequenceTokens => {
                self.min_sequence_tokens = self.min_sequence_tokens.min(value);
                self.max_sequence_tokens = self.max_sequence_tokens.max(value);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ShapeField {
    BatchSize,
    PromptTokens,
    DecodeTokens,
    SequenceTokens,
}

pub(super) fn serving_distribution_range(
    distribution: &ServingValueDistribution,
) -> Option<(u32, u32)> {
    match distribution {
        ServingValueDistribution::Uniform { min, max }
        | ServingValueDistribution::LogNormal { min, max, .. } => Some((*min, *max)),
        ServingValueDistribution::Weighted { values, .. } => {
            let min = values.iter().min().copied()?;
            let max = values.iter().max().copied()?;
            Some((min, max))
        }
    }
}

pub(super) fn workload_shape_envelope(
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> CalibrationShapeEnvelope {
    let mut envelope = CalibrationShapeEnvelope::from_request(request);
    if let Some(serving) = serving {
        envelope.include_serving(serving, request);
    }
    envelope
}

pub(super) fn calibration_coverage_report(
    profile: Option<&CalibrationProfileMetadata>,
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Option<CalibrationCoverageReport> {
    let profile = profile?;
    let envelope = workload_shape_envelope(request, serving);
    let required_phases = required_calibration_phases(request, serving);
    let covered_phases = covered_calibration_phases(&profile.benchmarks, &required_phases);
    let missing_phases: Vec<_> = required_phases
        .iter()
        .filter(|phase| !covered_phases.contains(*phase))
        .cloned()
        .collect();
    let phase_coverage_score = if required_phases.is_empty() {
        None
    } else {
        Some(covered_phases.len() as f64 / required_phases.len() as f64)
    };

    let shape_benchmark_count = profile
        .benchmarks
        .iter()
        .filter(|benchmark| benchmark_has_any_shape(benchmark))
        .count();
    let complete_shape_benchmark_count = profile
        .benchmarks
        .iter()
        .filter(|benchmark| benchmark_has_complete_shape(benchmark))
        .count();
    let batch_size_score = benchmark_field_coverage_score(
        envelope.min_batch_size,
        envelope.max_batch_size,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.batch_size),
    );
    let prompt_tokens_score = benchmark_field_coverage_score(
        envelope.min_prompt_tokens,
        envelope.max_prompt_tokens,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.prompt_tokens),
    );
    let decode_tokens_score = benchmark_field_coverage_score(
        envelope.min_decode_tokens,
        envelope.max_decode_tokens,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.decode_tokens),
    );
    let sequence_tokens_score = benchmark_field_coverage_score(
        envelope.min_sequence_tokens,
        envelope.max_sequence_tokens,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.sequence_tokens),
    );
    let shape_coverage_score = optional_mean_f64(
        [
            batch_size_score,
            prompt_tokens_score,
            decode_tokens_score,
            sequence_tokens_score,
        ]
        .into_iter()
        .flatten(),
    );
    let coverage_score = optional_mean_f64(
        [shape_coverage_score, phase_coverage_score]
            .into_iter()
            .flatten(),
    );
    let (nearest_benchmark, nearest_benchmark_distance) =
        nearest_shape_benchmark(&profile.benchmarks, envelope);
    let status = calibration_coverage_status(
        profile.benchmarks.len(),
        shape_benchmark_count,
        coverage_score,
        &missing_phases,
    )
    .to_string();

    Some(CalibrationCoverageReport {
        benchmark_count: profile.benchmarks.len(),
        shape_benchmark_count,
        complete_shape_benchmark_count,
        required_phases,
        covered_phases,
        missing_phases,
        batch_size_score,
        prompt_tokens_score,
        decode_tokens_score,
        sequence_tokens_score,
        shape_coverage_score,
        phase_coverage_score,
        coverage_score,
        nearest_benchmark,
        nearest_benchmark_distance,
        status,
    })
}

pub(super) fn required_calibration_phases(
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Vec<String> {
    if serving.is_some() {
        return vec![
            "prefill".to_string(),
            "decode".to_string(),
            "kv_transfer".to_string(),
        ];
    }

    match request.phase {
        InferencePhase::Prefill => vec!["prefill".to_string()],
        InferencePhase::Decode => vec!["decode".to_string()],
        InferencePhase::EndToEnd => vec!["prefill".to_string(), "decode".to_string()],
    }
}

pub(super) fn covered_calibration_phases(
    benchmarks: &[CalibrationBenchmarkPoint],
    required_phases: &[String],
) -> Vec<String> {
    let observed: HashSet<_> = benchmarks
        .iter()
        .filter_map(|benchmark| benchmark.phase.as_deref())
        .map(normalize)
        .collect();
    required_phases
        .iter()
        .filter(|phase| observed.contains(*phase))
        .cloned()
        .collect()
}

pub(super) fn benchmark_has_any_shape(benchmark: &CalibrationBenchmarkPoint) -> bool {
    benchmark.batch_size.is_some()
        || benchmark.prompt_tokens.is_some()
        || benchmark.decode_tokens.is_some()
        || benchmark.sequence_tokens.is_some()
}

pub(super) fn benchmark_has_complete_shape(benchmark: &CalibrationBenchmarkPoint) -> bool {
    benchmark.batch_size.is_some()
        && benchmark.prompt_tokens.is_some()
        && benchmark.decode_tokens.is_some()
        && benchmark.sequence_tokens.is_some()
}

pub(super) fn benchmark_field_range(
    benchmarks: &[CalibrationBenchmarkPoint],
    selector: impl Fn(&CalibrationBenchmarkPoint) -> Option<u32>,
) -> Option<(u32, u32)> {
    let mut values = benchmarks.iter().filter_map(selector);
    let first = values.next()?;
    let mut min = first;
    let mut max = first;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    Some((min, max))
}

pub(super) fn benchmark_field_coverage_score(
    observed_min: u32,
    observed_max: u32,
    benchmark_range: Option<(u32, u32)>,
) -> Option<f64> {
    let (benchmark_min, benchmark_max) = benchmark_range?;
    if benchmark_min <= observed_min && observed_max <= benchmark_max {
        return Some(1.0);
    }
    let overlap_min = observed_min.max(benchmark_min);
    let overlap_max = observed_max.min(benchmark_max);
    if overlap_min > overlap_max {
        return Some(0.0);
    }
    let observed_len = observed_max.saturating_sub(observed_min) + 1;
    let overlap_len = overlap_max.saturating_sub(overlap_min) + 1;
    Some(f64::from(overlap_len) / f64::from(observed_len))
}

pub(super) fn nearest_shape_benchmark(
    benchmarks: &[CalibrationBenchmarkPoint],
    envelope: CalibrationShapeEnvelope,
) -> (Option<String>, Option<f64>) {
    let mut nearest = None;
    let mut nearest_distance = f64::INFINITY;
    for (idx, benchmark) in benchmarks.iter().enumerate() {
        let (Some(batch_size), Some(prompt_tokens), Some(decode_tokens), Some(sequence_tokens)) = (
            benchmark.batch_size,
            benchmark.prompt_tokens,
            benchmark.decode_tokens,
            benchmark.sequence_tokens,
        ) else {
            continue;
        };
        let distance = shape_point_distance(
            [
                (batch_size, envelope.min_batch_size, envelope.max_batch_size),
                (
                    prompt_tokens,
                    envelope.min_prompt_tokens,
                    envelope.max_prompt_tokens,
                ),
                (
                    decode_tokens,
                    envelope.min_decode_tokens,
                    envelope.max_decode_tokens,
                ),
                (
                    sequence_tokens,
                    envelope.min_sequence_tokens,
                    envelope.max_sequence_tokens,
                ),
            ]
            .into_iter(),
        );
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest = Some(benchmark_label(idx, benchmark));
        }
    }

    if nearest.is_some() {
        (nearest, Some(nearest_distance))
    } else {
        (None, None)
    }
}

pub(super) fn shape_point_distance(values: impl Iterator<Item = (u32, u32, u32)>) -> f64 {
    values
        .map(|(value, min, max)| {
            if value < min {
                (f64::from(min) / f64::from(value.max(1))).ln()
            } else if value > max {
                (f64::from(value) / f64::from(max.max(1))).ln()
            } else {
                0.0
            }
        })
        .map(|distance| distance * distance)
        .sum::<f64>()
        .sqrt()
}

pub(super) fn benchmark_label(idx: usize, benchmark: &CalibrationBenchmarkPoint) -> String {
    benchmark.name.clone().unwrap_or_else(|| {
        let kind = benchmark.kind.as_deref().unwrap_or("unknown");
        let phase = benchmark.phase.as_deref().unwrap_or("unknown");
        format!("benchmark[{idx}] {kind}/{phase}")
    })
}

pub(super) fn calibration_coverage_status(
    benchmark_count: usize,
    shape_benchmark_count: usize,
    coverage_score: Option<f64>,
    missing_phases: &[String],
) -> &'static str {
    if benchmark_count == 0 {
        return "no_benchmarks";
    }
    if shape_benchmark_count == 0 {
        return "no_shape_benchmarks";
    }
    match coverage_score {
        Some(score) if score >= 0.95 && missing_phases.is_empty() => "covered",
        Some(score) if score >= 0.50 => "partial",
        Some(_) => "weak",
        None => "no_coverage_evidence",
    }
}

pub(super) fn optional_mean_f64(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

pub(super) fn calibration_invalid_shape_warnings(
    profile: Option<&CalibrationProfileMetadata>,
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Vec<CalibrationInvalidShapeWarning> {
    let Some(profile) = profile else {
        return Vec::new();
    };
    let envelope = workload_shape_envelope(request, serving);
    profile
        .invalid_shapes
        .iter()
        .filter(|invalid_shape| invalid_shape_overlaps(envelope, &invalid_shape.shape))
        .map(|invalid_shape| {
            let name = invalid_shape.name.clone();
            let reason = invalid_shape.reason.clone();
            let label = name
                .clone()
                .unwrap_or_else(|| "unnamed invalid shape".to_string());
            let reason_suffix = reason
                .as_deref()
                .map(|reason| format!(": {reason}"))
                .unwrap_or_default();
            CalibrationInvalidShapeWarning {
                name,
                reason,
                shape: invalid_shape.shape.clone(),
                message: format!(
                    "workload shape envelope overlaps calibration invalid shape '{label}'{reason_suffix}"
                ),
            }
        })
        .collect()
}

pub(super) fn invalid_shape_overlaps(
    envelope: CalibrationShapeEnvelope,
    shape: &CalibrationShapeRange,
) -> bool {
    optional_range_overlaps(
        envelope.min_batch_size,
        envelope.max_batch_size,
        shape.min_batch_size,
        shape.max_batch_size,
    ) && optional_range_overlaps(
        envelope.min_prompt_tokens,
        envelope.max_prompt_tokens,
        shape.min_prompt_tokens,
        shape.max_prompt_tokens,
    ) && optional_range_overlaps(
        envelope.min_decode_tokens,
        envelope.max_decode_tokens,
        shape.min_decode_tokens,
        shape.max_decode_tokens,
    ) && optional_range_overlaps(
        envelope.min_sequence_tokens,
        envelope.max_sequence_tokens,
        shape.min_sequence_tokens,
        shape.max_sequence_tokens,
    )
}

pub(super) fn optional_range_overlaps(
    observed_min: u32,
    observed_max: u32,
    invalid_min: Option<u32>,
    invalid_max: Option<u32>,
) -> bool {
    let invalid_min = invalid_min.unwrap_or(1);
    let invalid_max = invalid_max.unwrap_or(u32::MAX);
    observed_min <= invalid_max && invalid_min <= observed_max
}

pub(super) fn calibration_gate_violations(
    policy: &CalibrationPolicy,
    profile: Option<&CalibrationProfileMetadata>,
    coverage: Option<&CalibrationCoverageReport>,
    valid_shape_warnings: &[CalibrationApplicabilityWarning],
    invalid_shape_warnings: &[CalibrationInvalidShapeWarning],
) -> Vec<CalibrationGateViolation> {
    let mut violations = Vec::new();
    for warning in valid_shape_warnings {
        violations.push(CalibrationGateViolation {
            code: format!("valid_shape_{}_outside_range", warning.field),
            action: policy.valid_shape,
            observed: None,
            limit: None,
            message: warning.message.clone(),
        });
    }
    for warning in invalid_shape_warnings {
        violations.push(CalibrationGateViolation {
            code: "invalid_shape_overlap".to_string(),
            action: policy.invalid_shape,
            observed: None,
            limit: None,
            message: warning.message.clone(),
        });
    }

    if profile.is_none() {
        if policy.coverage == CalibrationGateMode::Reject {
            violations.push(CalibrationGateViolation {
                code: "no_calibration_profile".to_string(),
                action: policy.coverage,
                observed: None,
                limit: policy.min_coverage_score,
                message:
                    "calibration coverage policy is reject but workload has no calibration profile"
                        .to_string(),
            });
        }
        return violations;
    }

    let profile = profile.expect("profile checked above");
    if profile.source.is_none() {
        violations.push(CalibrationGateViolation {
            code: "calibration_profile_source_unspecified".to_string(),
            action: policy.profile_source,
            observed: Some(0.0),
            limit: Some(1.0),
            message:
                "calibration profile is missing source metadata, so benchmark provenance is not auditable"
                    .to_string(),
        });
    }
    if profile.date.is_none() {
        violations.push(CalibrationGateViolation {
            code: "calibration_profile_date_unspecified".to_string(),
            action: policy.profile_date,
            observed: Some(0.0),
            limit: Some(1.0),
            message:
                "calibration profile is missing date metadata, so benchmark recency is not auditable"
                    .to_string(),
        });
    }
    let missing_runtime_provenance = missing_calibration_profile_runtime_provenance(profile);
    if !missing_runtime_provenance.is_empty() {
        let observed =
            CALIBRATION_PROFILE_RUNTIME_PROVENANCE_FIELD_COUNT - missing_runtime_provenance.len();
        violations.push(CalibrationGateViolation {
            code: "calibration_profile_runtime_provenance_incomplete".to_string(),
            action: policy.profile_runtime,
            observed: Some(observed as f64),
            limit: Some(CALIBRATION_PROFILE_RUNTIME_PROVENANCE_FIELD_COUNT as f64),
            message: format!(
                "calibration profile is missing runtime provenance fields: {}; benchmark results may not be reproducible across software stacks",
                missing_runtime_provenance.join(", ")
            ),
        });
    }

    let Some(coverage) = coverage else {
        violations.push(CalibrationGateViolation {
            code: "no_calibration_coverage".to_string(),
            action: policy.coverage,
            observed: None,
            limit: policy.min_coverage_score,
            message: "calibration profile did not produce a coverage report".to_string(),
        });
        return violations;
    };

    if let Some(min_coverage_score) = policy.min_coverage_score {
        match coverage.coverage_score {
            Some(score) if score < min_coverage_score => {
                violations.push(CalibrationGateViolation {
                    code: "coverage_score_below_min".to_string(),
                    action: policy.coverage,
                    observed: Some(score),
                    limit: Some(min_coverage_score),
                    message: format!(
                        "calibration coverage score {score:.3} is below required minimum {min_coverage_score:.3}"
                    ),
                });
            }
            None => {
                violations.push(CalibrationGateViolation {
                    code: "coverage_score_missing".to_string(),
                    action: policy.coverage,
                    observed: None,
                    limit: Some(min_coverage_score),
                    message: format!(
                        "calibration coverage score is unavailable but required minimum is {min_coverage_score:.3}"
                    ),
                });
            }
            _ => {}
        }
    }
    if policy.require_phase_coverage && !coverage.missing_phases.is_empty() {
        violations.push(CalibrationGateViolation {
            code: "missing_required_calibration_phases".to_string(),
            action: policy.coverage,
            observed: coverage.phase_coverage_score,
            limit: Some(1.0),
            message: format!(
                "calibration benchmarks are missing required phases: {}",
                coverage.missing_phases.join(", ")
            ),
        });
    }

    violations
}

pub(super) const CALIBRATION_PROFILE_RUNTIME_PROVENANCE_FIELD_COUNT: usize = 6;

pub(super) fn missing_calibration_profile_runtime_provenance(
    profile: &CalibrationProfileMetadata,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if profile.backend_version.is_none() {
        missing.push("backend_version");
    }
    if profile.driver_version.is_none() {
        missing.push("driver_version");
    }
    if profile.cuda_version.is_none() && profile.rocm_version.is_none() {
        missing.push("cuda_version_or_rocm_version");
    }
    if profile.nccl_version.is_none()
        && profile.rccl_version.is_none()
        && profile.ucx_version.is_none()
    {
        missing.push("nccl_version_or_rccl_version_or_ucx_version");
    }
    if profile.kernel_settings.is_empty() {
        missing.push("kernel_settings");
    }
    if profile.environment_hash.is_none() {
        missing.push("environment_hash");
    }
    missing
}

pub(super) fn calibration_applicability_warnings(
    profile: Option<&CalibrationProfileMetadata>,
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Vec<CalibrationApplicabilityWarning> {
    let Some(valid_shape) = profile.and_then(|profile| profile.valid_shape.as_ref()) else {
        return Vec::new();
    };

    let envelope = workload_shape_envelope(request, serving);

    let mut warnings = Vec::new();
    push_shape_warning(
        &mut warnings,
        "batch_size",
        envelope.min_batch_size,
        envelope.max_batch_size,
        valid_shape.min_batch_size,
        valid_shape.max_batch_size,
    );
    push_shape_warning(
        &mut warnings,
        "prompt_tokens",
        envelope.min_prompt_tokens,
        envelope.max_prompt_tokens,
        valid_shape.min_prompt_tokens,
        valid_shape.max_prompt_tokens,
    );
    push_shape_warning(
        &mut warnings,
        "decode_tokens",
        envelope.min_decode_tokens,
        envelope.max_decode_tokens,
        valid_shape.min_decode_tokens,
        valid_shape.max_decode_tokens,
    );
    push_shape_warning(
        &mut warnings,
        "sequence_tokens",
        envelope.min_sequence_tokens,
        envelope.max_sequence_tokens,
        valid_shape.min_sequence_tokens,
        valid_shape.max_sequence_tokens,
    );
    warnings
}

pub(super) fn push_shape_warning(
    warnings: &mut Vec<CalibrationApplicabilityWarning>,
    field: &str,
    observed_min: u32,
    observed_max: u32,
    calibrated_min: Option<u32>,
    calibrated_max: Option<u32>,
) {
    let below_min = calibrated_min.is_some_and(|min| observed_min < min);
    let above_max = calibrated_max.is_some_and(|max| observed_max > max);
    if !below_min && !above_max {
        return;
    }

    warnings.push(CalibrationApplicabilityWarning {
        field: field.to_string(),
        observed_min,
        observed_max,
        calibrated_min,
        calibrated_max,
        message: format!(
            "workload {field} range {} falls outside calibration range {}",
            format_u32_range(Some(observed_min), Some(observed_max)),
            format_u32_range(calibrated_min, calibrated_max)
        ),
    });
}

pub(super) fn format_u32_range(min: Option<u32>, max: Option<u32>) -> String {
    match (min, max) {
        (Some(min), Some(max)) => format!("{min}..{max}"),
        (Some(min), None) => format!("{min}..unbounded"),
        (None, Some(max)) => format!("unbounded..{max}"),
        (None, None) => "unbounded".to_string(),
    }
}

pub(super) fn validate_min_max(
    name: &str,
    min: Option<u32>,
    max: Option<u32>,
) -> Result<(), ConfigError> {
    validate_positive_optional_u32(&format!("{name}.min"), min)?;
    validate_positive_optional_u32(&format!("{name}.max"), max)?;
    if let (Some(min), Some(max)) = (min, max)
        && min > max
    {
        return Err(ConfigError::new(format!(
            "{name} min must be less than or equal to max"
        )));
    }
    Ok(())
}

pub(super) fn validate_positive_optional_u32(
    name: &str,
    value: Option<u32>,
) -> Result<(), ConfigError> {
    if value == Some(0) {
        return Err(ConfigError::new(format!(
            "{name} must be greater than zero"
        )));
    }
    Ok(())
}

pub(super) fn validate_optional_max_sequence_tokens(
    name: &str,
    max_sequence_tokens: Option<u32>,
    prompt_tokens: u32,
    decode_tokens: u32,
) -> Result<(), ConfigError> {
    validate_positive_optional_u32(name, max_sequence_tokens)?;
    if let Some(max_sequence_tokens) = max_sequence_tokens {
        let required_sequence_tokens = u64::from(prompt_tokens) + u64::from(decode_tokens);
        if u64::from(max_sequence_tokens) < required_sequence_tokens {
            return Err(ConfigError::new(format!(
                "{name} must be greater than or equal to prompt_tokens + decode_tokens ({required_sequence_tokens})"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_positive_optional_f64(
    name: &str,
    value: Option<f64>,
) -> Result<(), ConfigError> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(ConfigError::new(format!(
            "{name} must be finite and greater than zero"
        )));
    }
    Ok(())
}

pub(super) fn validate_positive_fraction_optional_f64(
    name: &str,
    value: Option<f64>,
) -> Result<(), ConfigError> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0 || value > 1.0)
    {
        return Err(ConfigError::new(format!(
            "{name} must be finite and greater than 0.0 and less than or equal to 1.0"
        )));
    }
    Ok(())
}

pub(super) fn validate_nonnegative_optional_f64(
    name: &str,
    value: Option<f64>,
) -> Result<(), ConfigError> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        return Err(ConfigError::new(format!(
            "{name} must be finite and greater than or equal to zero"
        )));
    }
    Ok(())
}

pub(super) fn validate_optional_finite_f64(
    name: &str,
    value: Option<f64>,
) -> Result<(), ConfigError> {
    if let Some(value) = value {
        validate_finite_f64(name, value)?;
    }
    Ok(())
}

pub(super) fn validate_finite_f64(name: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(ConfigError::new(format!("{name} must be finite")));
    }
    Ok(())
}

pub(super) fn calibration_with_defaults(
    section: Option<CalibrationSection>,
    defaults: SimulationCalibration,
) -> SimulationCalibration {
    let Some(section) = section else {
        return defaults;
    };

    SimulationCalibration {
        compute_efficiency: section
            .compute_efficiency
            .unwrap_or(defaults.compute_efficiency),
        prefill_compute_scale: section
            .prefill_compute_scale
            .unwrap_or(defaults.prefill_compute_scale),
        decode_compute_scale: section
            .decode_compute_scale
            .unwrap_or(defaults.decode_compute_scale),
        decode_memory_bandwidth_scale: section
            .decode_memory_bandwidth_scale
            .unwrap_or(defaults.decode_memory_bandwidth_scale),
        collective_latency_scale: section
            .collective_latency_scale
            .unwrap_or(defaults.collective_latency_scale),
        collective_bandwidth_scale: section
            .collective_bandwidth_scale
            .unwrap_or(defaults.collective_bandwidth_scale),
        kv_transfer_scale: section
            .kv_transfer_scale
            .unwrap_or(defaults.kv_transfer_scale),
        scheduler_overhead_us: section
            .scheduler_overhead_us
            .unwrap_or(defaults.scheduler_overhead_us),
        serving_memory_temporary_fraction: section
            .serving_memory_temporary_fraction
            .unwrap_or(defaults.serving_memory_temporary_fraction),
        serving_memory_activation_communication_fraction: section
            .serving_memory_activation_communication_fraction
            .unwrap_or(defaults.serving_memory_activation_communication_fraction),
        serving_memory_weight_communication_fraction: section
            .serving_memory_weight_communication_fraction
            .unwrap_or(defaults.serving_memory_weight_communication_fraction),
        serving_memory_runtime_reserve_fraction: section
            .serving_memory_runtime_reserve_fraction
            .unwrap_or(defaults.serving_memory_runtime_reserve_fraction),
        serving_memory_fragmentation_fraction: section
            .serving_memory_fragmentation_fraction
            .unwrap_or(defaults.serving_memory_fragmentation_fraction),
        serving_pipeline_depth: section
            .serving_pipeline_depth
            .unwrap_or(defaults.serving_pipeline_depth),
        request_arrival_gap_s: section
            .request_arrival_gap_s
            .unwrap_or(defaults.request_arrival_gap_s),
        allow_compute_comm_overlap: section
            .allow_compute_comm_overlap
            .unwrap_or(defaults.allow_compute_comm_overlap),
    }
    .sanitized()
}
