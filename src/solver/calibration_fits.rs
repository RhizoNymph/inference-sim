use super::*;

pub(super) fn calibration_feature_values(
    model: &ModelSpec,
    request: &InferenceRequest,
    config: ParallelismConfig,
    extra_features: Option<&BTreeMap<String, f64>>,
) -> BTreeMap<String, f64> {
    let mut features = BTreeMap::new();
    let batch_size = f64::from(request.batch_size.max(1));
    let prompt_tokens = f64::from(request.prompt_tokens.max(1));
    let decode_tokens = f64::from(request.decode_tokens.max(1));
    let sequence_tokens = f64::from(request.max_sequence_tokens.max(1));
    let tensor_ranks = f64::from(config.tensor_ranks.max(1));
    let pipeline_ranks = f64::from(config.pipeline_ranks.max(1));
    let expert_ranks = f64::from(config.expert_ranks.max(1));
    let data_ranks = f64::from(config.data_ranks.max(1));
    let total_ranks = f64::from(config.total_ranks().max(1));

    insert_feature(&mut features, "batch_size", batch_size);
    insert_feature(&mut features, "batch", batch_size);
    insert_feature(&mut features, "prompt_tokens", prompt_tokens);
    insert_feature(&mut features, "prefill_tokens", prompt_tokens);
    insert_feature(&mut features, "effective_prefill_tokens", prompt_tokens);
    insert_feature(&mut features, "decode_tokens", decode_tokens);
    insert_feature(&mut features, "output_tokens", decode_tokens);
    insert_feature(&mut features, "sequence_tokens", sequence_tokens);
    insert_feature(&mut features, "max_sequence_tokens", sequence_tokens);
    insert_feature(&mut features, "batch_tokens", batch_size * prompt_tokens);
    insert_feature(
        &mut features,
        "prefill_batch_tokens",
        batch_size * prompt_tokens,
    );
    insert_feature(
        &mut features,
        "decode_batch_tokens",
        batch_size * decode_tokens,
    );
    insert_feature(&mut features, "tensor_ranks", tensor_ranks);
    insert_feature(&mut features, "tp", tensor_ranks);
    insert_feature(&mut features, "pipeline_ranks", pipeline_ranks);
    insert_feature(&mut features, "pp", pipeline_ranks);
    insert_feature(&mut features, "expert_ranks", expert_ranks);
    insert_feature(&mut features, "ep", expert_ranks);
    insert_feature(&mut features, "data_ranks", data_ranks);
    insert_feature(&mut features, "dp", data_ranks);
    insert_feature(&mut features, "total_ranks", total_ranks);
    insert_feature(
        &mut features,
        "parameters_gb",
        model.parameters.as_gigabytes(),
    );
    insert_feature(
        &mut features,
        "model_params_gb",
        model.parameters.as_gigabytes(),
    );
    insert_feature(
        &mut features,
        "parameter_count_billion",
        model.parameter_count_billion(),
    );
    insert_feature(
        &mut features,
        "model_parameter_count_billion",
        model.parameter_count_billion(),
    );
    insert_feature(&mut features, "layers", f64::from(model.layers));
    insert_feature(&mut features, "hidden_size", f64::from(model.hidden_size));
    insert_feature(
        &mut features,
        "attention_heads",
        f64::from(model.attention_heads.max(1)),
    );
    insert_feature(&mut features, "kv_heads", f64::from(model.kv_heads.max(1)));
    insert_feature(
        &mut features,
        "dtype_bytes",
        model.dtype.bytes_per_element() as f64,
    );
    insert_feature(
        &mut features,
        "kv_dtype_bytes",
        model.kv_dtype().bytes_per_element() as f64,
    );
    insert_feature(
        &mut features,
        "kv_cache_dtype_bytes",
        model.kv_dtype().bytes_per_element() as f64,
    );

    if let Some(extra_features) = extra_features {
        for (name, value) in extra_features {
            insert_feature(&mut features, name, *value);
        }
    }

    features
}

fn insert_feature(features: &mut BTreeMap<String, f64>, name: &str, value: f64) {
    if value.is_finite() {
        features.insert(normalize_fit_name(name), value);
    }
}

pub(super) fn fit_matches(fit: &CalibrationFittedModel, phase: &str, targets: &[&str]) -> bool {
    if !matches!(
        normalize_fit_name(&fit.model).as_str(),
        "linear" | "linear_regression" | "ols" | "ordinary_least_squares"
    ) {
        return false;
    }
    if let Some(fit_phase) = &fit.phase
        && normalize_fit_name(fit_phase) != normalize_fit_name(phase)
    {
        return false;
    }
    let fit_target = normalize_fit_name(&fit.target);
    targets
        .iter()
        .any(|target| normalize_fit_name(target) == fit_target)
}

pub(super) fn evaluate_fit(
    fit: &CalibrationFittedModel,
    phase: &str,
    features: &BTreeMap<String, f64>,
    baseline_s: Option<f64>,
) -> Option<FitEvaluation> {
    let mut prediction = fit.intercept.unwrap_or(0.0);
    let mut feature_values = Vec::with_capacity(fit.features.len());
    let mut has_range = false;
    let mut all_ranged = true;
    let mut has_extrapolation = false;
    let mut max_extrapolation_ratio = 0.0_f64;
    for (feature, coefficient) in fit.features.iter().zip(&fit.coefficients) {
        let value = features.get(&normalize_fit_name(feature))?;
        prediction += coefficient * value;
        let feature_range = fit_feature_range(fit, feature);
        let status = feature_range_status(feature_range, *value);
        has_range |= feature_range.is_some();
        all_ranged &= feature_range.is_some();
        has_extrapolation |= status.status == "extrapolated";
        max_extrapolation_ratio = max_extrapolation_ratio.max(status.extrapolation_ratio);
        feature_values.push(CalibrationFitFeatureValue {
            name: feature.clone(),
            value: *value,
            coefficient: *coefficient,
            range_min: feature_range.and_then(|range| range.min),
            range_max: feature_range.and_then(|range| range.max),
            status: status.status,
            extrapolation_ratio: status.extrapolation_ratio,
        });
    }
    if !prediction.is_finite() || prediction <= 0.0 {
        return None;
    }

    let seconds = fit_value_to_seconds(prediction, fit);
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }

    let applicability_status = if has_extrapolation {
        "extrapolated"
    } else if all_ranged && has_range {
        "interpolated"
    } else if has_range {
        "partially_bounded"
    } else {
        "unbounded"
    };
    let confidence_score = fit_confidence_score(fit, applicability_status, max_extrapolation_ratio);
    let (relative_uncertainty_pct, absolute_uncertainty_s, uncertainty_source) =
        fit_uncertainty(fit, seconds);

    Some(FitEvaluation {
        seconds,
        application: CalibrationFitApplication {
            phase: phase.to_string(),
            target: fit.target.clone(),
            fit_name: fit.name.clone(),
            model: fit.model.clone(),
            unit: fit.unit.clone(),
            intercept: fit.intercept.unwrap_or(0.0),
            raw_prediction: prediction,
            prediction_kind: "latency".to_string(),
            predicted_value: seconds,
            prediction_unit: Some("s".to_string()),
            predicted_s: seconds,
            baseline_value: baseline_s,
            baseline_s,
            applicability_status: applicability_status.to_string(),
            confidence_score,
            max_extrapolation_ratio,
            relative_uncertainty_pct,
            absolute_uncertainty_value: absolute_uncertainty_s,
            absolute_uncertainty_s,
            uncertainty_source,
            validation_rmse: fit.validation_rmse,
            validation_rmse_pct: fit.validation_rmse_pct,
            validation_mean_abs_pct_error: fit.validation_mean_abs_pct_error,
            validation_max_abs_pct_error: fit.validation_max_abs_pct_error,
            confidence_interval: fit.confidence_interval,
            confidence_interval_pct: fit.confidence_interval_pct,
            confidence_level: fit.confidence_level,
            sample_count: fit.sample_count,
            validation_sample_count: fit.validation_sample_count,
            source: fit.source.clone(),
            features: feature_values,
        },
    })
}

pub(super) fn evaluate_value_fit(
    fit: &CalibrationFittedModel,
    phase: &str,
    features: &BTreeMap<String, f64>,
    baseline_value: Option<f64>,
    prediction_kind: &str,
    prediction_unit: Option<&str>,
) -> Option<CalibrationFitApplication> {
    let mut prediction = fit.intercept.unwrap_or(0.0);
    let mut feature_values = Vec::with_capacity(fit.features.len());
    let mut has_range = false;
    let mut all_ranged = true;
    let mut has_extrapolation = false;
    let mut max_extrapolation_ratio = 0.0_f64;
    for (feature, coefficient) in fit.features.iter().zip(&fit.coefficients) {
        let value = features.get(&normalize_fit_name(feature))?;
        prediction += coefficient * value;
        let feature_range = fit_feature_range(fit, feature);
        let status = feature_range_status(feature_range, *value);
        has_range |= feature_range.is_some();
        all_ranged &= feature_range.is_some();
        has_extrapolation |= status.status == "extrapolated";
        max_extrapolation_ratio = max_extrapolation_ratio.max(status.extrapolation_ratio);
        feature_values.push(CalibrationFitFeatureValue {
            name: feature.clone(),
            value: *value,
            coefficient: *coefficient,
            range_min: feature_range.and_then(|range| range.min),
            range_max: feature_range.and_then(|range| range.max),
            status: status.status,
            extrapolation_ratio: status.extrapolation_ratio,
        });
    }
    if !prediction.is_finite() || prediction <= 0.0 {
        return None;
    }

    let applicability_status = if has_extrapolation {
        "extrapolated"
    } else if all_ranged && has_range {
        "interpolated"
    } else if has_range {
        "partially_bounded"
    } else {
        "unbounded"
    };
    let confidence_score = fit_confidence_score(fit, applicability_status, max_extrapolation_ratio);
    let (relative_uncertainty_pct, absolute_uncertainty_value, uncertainty_source) =
        fit_uncertainty_value(fit, prediction);

    Some(CalibrationFitApplication {
        phase: phase.to_string(),
        target: fit.target.clone(),
        fit_name: fit.name.clone(),
        model: fit.model.clone(),
        unit: fit.unit.clone(),
        intercept: fit.intercept.unwrap_or(0.0),
        raw_prediction: prediction,
        prediction_kind: prediction_kind.to_string(),
        predicted_value: prediction,
        prediction_unit: prediction_unit.map(str::to_string),
        predicted_s: 0.0,
        baseline_value,
        baseline_s: None,
        applicability_status: applicability_status.to_string(),
        confidence_score,
        max_extrapolation_ratio,
        relative_uncertainty_pct,
        absolute_uncertainty_value,
        absolute_uncertainty_s: None,
        uncertainty_source,
        validation_rmse: fit.validation_rmse,
        validation_rmse_pct: fit.validation_rmse_pct,
        validation_mean_abs_pct_error: fit.validation_mean_abs_pct_error,
        validation_max_abs_pct_error: fit.validation_max_abs_pct_error,
        confidence_interval: fit.confidence_interval,
        confidence_interval_pct: fit.confidence_interval_pct,
        confidence_level: fit.confidence_level,
        sample_count: fit.sample_count,
        validation_sample_count: fit.validation_sample_count,
        source: fit.source.clone(),
        features: feature_values,
    })
}

struct FeatureRangeStatus {
    status: String,
    extrapolation_ratio: f64,
}

fn fit_feature_range<'a>(
    fit: &'a CalibrationFittedModel,
    feature: &str,
) -> Option<&'a CalibrationFitFeatureRange> {
    let normalized = normalize_fit_name(feature);
    fit.feature_ranges
        .iter()
        .find(|range| normalize_fit_name(&range.feature) == normalized)
}

fn feature_range_status(
    range: Option<&CalibrationFitFeatureRange>,
    value: f64,
) -> FeatureRangeStatus {
    let Some(range) = range else {
        return FeatureRangeStatus {
            status: "unbounded".to_string(),
            extrapolation_ratio: 0.0,
        };
    };
    let lower_excess = range.min.map(|min| (min - value).max(0.0)).unwrap_or(0.0);
    let upper_excess = range.max.map(|max| (value - max).max(0.0)).unwrap_or(0.0);
    let excess = lower_excess.max(upper_excess);
    if excess <= 0.0 {
        return FeatureRangeStatus {
            status: "in_range".to_string(),
            extrapolation_ratio: 0.0,
        };
    }
    let scale = match (range.min, range.max) {
        (Some(min), Some(max)) => (max - min).abs().max(1.0),
        (Some(bound), None) | (None, Some(bound)) => bound.abs().max(1.0),
        (None, None) => 1.0,
    };
    FeatureRangeStatus {
        status: "extrapolated".to_string(),
        extrapolation_ratio: excess / scale,
    }
}

fn fit_confidence_score(
    fit: &CalibrationFittedModel,
    applicability_status: &str,
    max_extrapolation_ratio: f64,
) -> f64 {
    let mut score = fit.r_squared.unwrap_or(1.0).clamp(0.0, 1.0);
    if let Some((error_pct, _)) = fit_relative_error_pct(fit) {
        score *= (1.0 - (error_pct / 100.0).clamp(0.0, 1.0)).max(0.0);
    }
    match applicability_status {
        "interpolated" => {}
        "partially_bounded" => score *= 0.9,
        "unbounded" => score *= 0.75,
        "extrapolated" => score /= 1.0 + max_extrapolation_ratio.max(0.0),
        _ => {}
    }
    score.clamp(0.0, 1.0)
}

fn fit_uncertainty(
    fit: &CalibrationFittedModel,
    predicted_s: f64,
) -> (Option<f64>, Option<f64>, Option<String>) {
    let (relative_uncertainty_pct, absolute_uncertainty_value, uncertainty_source) =
        fit_uncertainty_value(fit, predicted_s);
    let absolute_from_rmse = fit_absolute_uncertainty_value(fit)
        .map(|(value, _)| value)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| fit_value_to_seconds(value, fit))
        .filter(|value| value.is_finite() && *value >= 0.0);
    let absolute_uncertainty_s = absolute_from_rmse.or(absolute_uncertainty_value);

    (
        relative_uncertainty_pct,
        absolute_uncertainty_s,
        uncertainty_source,
    )
}

fn fit_uncertainty_value(
    fit: &CalibrationFittedModel,
    predicted_value: f64,
) -> (Option<f64>, Option<f64>, Option<String>) {
    let relative = fit_uncertainty_pct(fit).filter(|(value, _)| value.is_finite() && *value >= 0.0);
    let relative_uncertainty_pct = relative.map(|(value, _)| value);

    let absolute_from_rmse =
        fit_absolute_uncertainty_value(fit).filter(|(value, _)| value.is_finite() && *value >= 0.0);
    let absolute_from_relative = relative_uncertainty_pct
        .filter(|_| predicted_value.is_finite() && predicted_value >= 0.0)
        .map(|value| predicted_value * value / 100.0);
    let absolute_uncertainty_value = absolute_from_rmse
        .map(|(value, _)| value)
        .or(absolute_from_relative);
    let uncertainty_source = if let Some((_, source)) = absolute_from_rmse {
        Some(source.to_string())
    } else {
        relative.map(|(_, source)| source.to_string())
    };

    (
        relative_uncertainty_pct,
        absolute_uncertainty_value,
        uncertainty_source,
    )
}

fn fit_uncertainty_pct(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.confidence_interval_pct
        .map(|value| (value, "confidence_interval_pct"))
        .or_else(|| fit_relative_error_pct(fit))
}

fn fit_relative_error_pct(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.validation_rmse_pct
        .map(|value| (value, "validation_rmse_pct"))
        .or_else(|| {
            fit.validation_mean_abs_pct_error
                .map(|value| (value, "validation_mean_abs_pct_error"))
        })
        .or_else(|| {
            fit.validation_max_abs_pct_error
                .map(|value| (value, "validation_max_abs_pct_error"))
        })
        .or_else(|| fit.rmse_pct.map(|value| (value, "rmse_pct")))
        .or_else(|| {
            fit.mean_abs_pct_error
                .map(|value| (value, "mean_abs_pct_error"))
        })
        .or_else(|| {
            fit.max_abs_pct_error
                .map(|value| (value, "max_abs_pct_error"))
        })
}

fn fit_absolute_uncertainty_value(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.confidence_interval
        .map(|value| (value, "confidence_interval"))
        .or_else(|| fit_absolute_error_value(fit))
}

fn fit_absolute_error_value(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.validation_rmse
        .map(|value| (value, "validation_rmse"))
        .or_else(|| fit.rmse.map(|value| (value, "rmse")))
}

fn fit_value_to_seconds(value: f64, fit: &CalibrationFittedModel) -> f64 {
    if let Some(unit) = fit.unit.as_deref().map(normalize_fit_name) {
        if unit == "us" || unit.contains("microsecond") {
            return value / 1e6;
        }
        if unit == "ms" || unit.contains("millisecond") {
            return value / 1e3;
        }
        if unit == "s" || unit.contains("second") {
            return value;
        }
    }

    let target = normalize_fit_name(&fit.target);
    if target.ends_with("_us") || target.contains("_us_") {
        value / 1e6
    } else if target.ends_with("_ms") || target.contains("_ms_") {
        value / 1e3
    } else {
        value
    }
}

fn normalize_fit_name(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
