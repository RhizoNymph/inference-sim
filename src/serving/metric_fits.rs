use super::*;

pub(super) fn apply_serving_metric_calibration_fits(
    metrics: &mut ServingMetrics,
    calibration_profile: Option<&CalibrationProfileMetadata>,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
) -> Vec<CalibrationFitApplication> {
    if calibration_profile.is_none() {
        return Vec::new();
    }

    let baseline = *metrics;
    let mut applications = Vec::new();

    if let Some(application) = serving_metric_fit_application(
        calibration_profile,
        &[
            "ttft_ms",
            "ttft_latency_ms",
            "time_to_first_token_ms",
            "ttft_s",
            "time_to_first_token_s",
        ],
        "ttft",
        baseline.ttft_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    ) {
        apply_latency_metric_prediction(
            &mut metrics.ttft_s,
            &mut metrics.ttft_p50_s,
            &mut metrics.ttft_p90_s,
            &mut metrics.ttft_p95_s,
            &mut metrics.ttft_p99_s,
            &mut metrics.ttft_max_s,
            application.predicted_s,
        );
        applications.push(application);
    }

    if let Some(application) = serving_metric_fit_application(
        calibration_profile,
        &[
            "tpot_ms",
            "tpot_latency_ms",
            "time_per_output_token_ms",
            "tpot_s",
            "time_per_output_token_s",
        ],
        "tpot",
        baseline.tpot_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    ) {
        apply_latency_metric_prediction(
            &mut metrics.tpot_s,
            &mut metrics.tpot_p50_s,
            &mut metrics.tpot_p90_s,
            &mut metrics.tpot_p95_s,
            &mut metrics.tpot_p99_s,
            &mut metrics.tpot_max_s,
            application.predicted_s,
        );
        applications.push(application);
    }

    if let Some(application) = serving_metric_fit_application(
        calibration_profile,
        &[
            "e2el_ms",
            "e2el_latency_ms",
            "end_to_end_ms",
            "end_to_end_latency_ms",
            "e2el_s",
            "end_to_end_s",
        ],
        "e2el",
        baseline.e2el_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    ) {
        apply_latency_metric_prediction(
            &mut metrics.e2el_s,
            &mut metrics.e2el_p50_s,
            &mut metrics.e2el_p90_s,
            &mut metrics.e2el_p95_s,
            &mut metrics.e2el_p99_s,
            &mut metrics.e2el_max_s,
            application.predicted_s,
        );
        applications.push(application);
    }

    if let Some(application) = serving_metric_value_fit_application(
        calibration_profile,
        &[
            "throughput_tokens_per_s",
            "output_throughput_tokens_per_s",
            "throughput",
            "tokens_per_s",
        ],
        "throughput",
        baseline.throughput_tokens_per_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
        "throughput",
        Some("tokens/s"),
    ) {
        metrics.throughput_tokens_per_s = application.predicted_value;
        applications.push(application);
    }

    applications
}

#[allow(clippy::too_many_arguments)]
fn serving_metric_fit_application(
    calibration_profile: Option<&CalibrationProfileMetadata>,
    targets: &[&str],
    target_metric: &str,
    baseline_s: f64,
    baseline: &ServingMetrics,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
) -> Option<CalibrationFitApplication> {
    if !baseline_s.is_finite() || baseline_s <= 0.0 {
        return None;
    }
    let features = serving_metric_fit_features(
        target_metric,
        baseline_s,
        baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    );
    Solver::fitted_latency_from_features(
        calibration_profile,
        "serving",
        targets,
        &features,
        Some(baseline_s),
    )
}

#[allow(clippy::too_many_arguments)]
fn serving_metric_value_fit_application(
    calibration_profile: Option<&CalibrationProfileMetadata>,
    targets: &[&str],
    target_metric: &str,
    baseline_value: f64,
    baseline: &ServingMetrics,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
    prediction_kind: &str,
    prediction_unit: Option<&str>,
) -> Option<CalibrationFitApplication> {
    if !baseline_value.is_finite() || baseline_value <= 0.0 {
        return None;
    }
    let features = serving_metric_fit_features(
        target_metric,
        baseline_value,
        baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    );
    Solver::fitted_value_from_features(
        calibration_profile,
        "serving",
        targets,
        &features,
        Some(baseline_value),
        prediction_kind,
        prediction_unit,
    )
}

#[allow(clippy::too_many_arguments)]
fn serving_metric_fit_features(
    target_metric: &str,
    target_baseline_value: f64,
    baseline: &ServingMetrics,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
) -> BTreeMap<String, f64> {
    let mut features = BTreeMap::new();
    insert_serving_fit_feature(&mut features, "baseline_value", target_baseline_value);
    insert_serving_fit_feature(
        &mut features,
        &format!("{target_metric}_baseline_value"),
        target_baseline_value,
    );
    insert_serving_fit_feature(&mut features, "baseline_s", target_baseline_value);
    insert_serving_fit_feature(&mut features, "baseline_ms", target_baseline_value * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        &format!("{target_metric}_baseline_s"),
        target_baseline_value,
    );
    insert_serving_fit_feature(
        &mut features,
        &format!("{target_metric}_baseline_ms"),
        target_baseline_value * 1000.0,
    );
    if target_metric == "throughput" {
        insert_serving_fit_feature(
            &mut features,
            "baseline_throughput_tokens_per_s",
            target_baseline_value,
        );
        insert_serving_fit_feature(
            &mut features,
            "throughput_baseline_tokens_per_s",
            target_baseline_value,
        );
    }

    insert_serving_fit_feature(&mut features, "batch_size", f64::from(request.batch_size));
    insert_serving_fit_feature(
        &mut features,
        "prompt_tokens",
        f64::from(request.prompt_tokens),
    );
    insert_serving_fit_feature(
        &mut features,
        "decode_tokens",
        f64::from(request.decode_tokens),
    );
    insert_serving_fit_feature(
        &mut features,
        "sequence_tokens",
        f64::from(request.prompt_tokens.saturating_add(request.decode_tokens)),
    );
    insert_serving_fit_feature(
        &mut features,
        "max_sequence_tokens",
        f64::from(request.max_sequence_tokens),
    );
    insert_serving_fit_feature(&mut features, "model_layers", f64::from(model.layers));
    insert_serving_fit_feature(
        &mut features,
        "model_hidden_size",
        f64::from(model.hidden_size),
    );
    insert_serving_fit_feature(
        &mut features,
        "model_parameters_gb",
        model.parameters.as_gigabytes(),
    );
    insert_serving_fit_feature(
        &mut features,
        "model_parameter_count_billion",
        model.parameter_count_billion(),
    );
    insert_serving_fit_feature(
        &mut features,
        "request_count",
        f64::from(traffic.request_count.unwrap_or(baseline.scheduled_requests)),
    );
    insert_serving_fit_feature(
        &mut features,
        "scheduled_requests",
        f64::from(baseline.scheduled_requests),
    );
    insert_serving_fit_feature(
        &mut features,
        "admitted_requests",
        f64::from(baseline.admitted_requests),
    );
    insert_serving_fit_feature(
        &mut features,
        "completed_requests",
        f64::from(baseline.completed_requests),
    );
    insert_serving_fit_feature(
        &mut features,
        "measured_requests",
        f64::from(baseline.measured_requests),
    );
    if let Some(arrival_rate_per_s) = serving_arrival_rate_feature(traffic) {
        insert_serving_fit_feature(&mut features, "arrival_rate_per_s", arrival_rate_per_s);
        insert_serving_fit_feature(&mut features, "request_rate_per_s", arrival_rate_per_s);
    }

    insert_serving_fit_feature(
        &mut features,
        "prefill_tensor_ranks",
        f64::from(prefill_config.tensor_ranks),
    );
    insert_serving_fit_feature(
        &mut features,
        "prefill_pipeline_ranks",
        f64::from(prefill_config.pipeline_ranks),
    );
    insert_serving_fit_feature(
        &mut features,
        "decode_tensor_ranks",
        f64::from(decode_config.tensor_ranks),
    );
    insert_serving_fit_feature(
        &mut features,
        "decode_pipeline_ranks",
        f64::from(decode_config.pipeline_ranks),
    );

    insert_serving_fit_feature(&mut features, "simulated_ttft_s", baseline.ttft_s);
    insert_serving_fit_feature(&mut features, "simulated_ttft_ms", baseline.ttft_s * 1000.0);
    insert_serving_fit_feature(&mut features, "simulated_tpot_s", baseline.tpot_s);
    insert_serving_fit_feature(&mut features, "simulated_tpot_ms", baseline.tpot_s * 1000.0);
    insert_serving_fit_feature(&mut features, "simulated_e2el_s", baseline.e2el_s);
    insert_serving_fit_feature(&mut features, "simulated_e2el_ms", baseline.e2el_s * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        "throughput_tokens_per_s",
        baseline.throughput_tokens_per_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "simulated_throughput_tokens_per_s",
        baseline.throughput_tokens_per_s,
    );
    insert_serving_fit_feature(&mut features, "prefill_s", baseline.prefill_s);
    insert_serving_fit_feature(&mut features, "prefill_ms", baseline.prefill_s * 1000.0);
    insert_serving_fit_feature(&mut features, "decode_s", baseline.decode_s);
    insert_serving_fit_feature(&mut features, "decode_ms", baseline.decode_s * 1000.0);
    insert_serving_fit_feature(&mut features, "kv_transfer_s", baseline.kv_transfer_s);
    insert_serving_fit_feature(
        &mut features,
        "kv_transfer_ms",
        baseline.kv_transfer_s * 1000.0,
    );
    insert_serving_fit_feature(&mut features, "queue_s", baseline.queue_delay_s);
    insert_serving_fit_feature(&mut features, "queue_ms", baseline.queue_delay_s * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        "prefill_queue_s",
        baseline.prefill_worker_queue_s + baseline.prefill_resource_queue_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "prefill_queue_ms",
        (baseline.prefill_worker_queue_s + baseline.prefill_resource_queue_s) * 1000.0,
    );
    insert_serving_fit_feature(&mut features, "decode_queue_s", baseline.decode_queue_s);
    insert_serving_fit_feature(
        &mut features,
        "decode_queue_ms",
        baseline.decode_queue_s * 1000.0,
    );
    insert_serving_fit_feature(&mut features, "kv_queue_s", baseline.kv_queue_s);
    insert_serving_fit_feature(&mut features, "kv_queue_ms", baseline.kv_queue_s * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        "kv_worker_queue_s",
        baseline.kv_worker_queue_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "kv_worker_queue_ms",
        baseline.kv_worker_queue_s * 1000.0,
    );
    insert_serving_fit_feature(
        &mut features,
        "kv_route_resource_queue_s",
        baseline.kv_resource_queue_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "kv_route_resource_queue_ms",
        baseline.kv_resource_queue_s * 1000.0,
    );
    insert_serving_fit_feature(
        &mut features,
        "output_tokens",
        baseline.decode_iterations as f64 * f64::from(request.batch_size.max(1)),
    );
    insert_serving_fit_feature(
        &mut features,
        "effective_prefill_tokens",
        baseline.effective_prefill_tokens as f64,
    );

    features
}

fn serving_arrival_rate_feature(traffic: &ServingTraffic) -> Option<f64> {
    match traffic.arrival {
        ServingArrivalPattern::FixedGap => traffic
            .arrival_gap_s
            .filter(|gap_s| gap_s.is_finite() && *gap_s > 0.0)
            .map(|gap_s| 1.0 / gap_s),
        ServingArrivalPattern::Poisson { rate_per_s, .. }
        | ServingArrivalPattern::SelfSimilar { rate_per_s, .. } => {
            (rate_per_s.is_finite() && rate_per_s > 0.0).then_some(rate_per_s)
        }
        ServingArrivalPattern::Bursty {
            burst_size,
            burst_interval_s,
            ..
        } => (burst_interval_s.is_finite() && burst_interval_s > 0.0)
            .then_some(f64::from(burst_size) / burst_interval_s),
        ServingArrivalPattern::Diurnal {
            min_rate_per_s,
            max_rate_per_s,
            ..
        } => {
            let mean = (min_rate_per_s + max_rate_per_s) / 2.0;
            (mean.is_finite() && mean > 0.0).then_some(mean)
        }
        ServingArrivalPattern::TraceDerived => None,
    }
}

fn insert_serving_fit_feature(features: &mut BTreeMap<String, f64>, name: &str, value: f64) {
    if value.is_finite() {
        features.insert(name.to_string(), value);
    }
}

fn apply_latency_metric_prediction(
    mean_s: &mut f64,
    p50_s: &mut f64,
    p90_s: &mut f64,
    p95_s: &mut f64,
    p99_s: &mut f64,
    max_s: &mut f64,
    predicted_s: f64,
) {
    if !predicted_s.is_finite() || predicted_s <= 0.0 || !mean_s.is_finite() || *mean_s <= 0.0 {
        return;
    }
    let delta_s = predicted_s - *mean_s;
    *mean_s = predicted_s;
    for value in [p50_s, p90_s, p95_s, p99_s, max_s] {
        if value.is_finite() {
            *value = (*value + delta_s).max(0.0);
        }
    }
}
