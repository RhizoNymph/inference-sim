use super::*;

pub(super) fn apply_calibration_gates_to_parallelism(
    results: &mut [ScoredParallelismConfig],
    violations: &[CalibrationGateViolation],
) {
    let Some(reason) = hard_calibration_gate_reason(violations) else {
        return;
    };
    for score in results {
        score.feasible = false;
        score.estimated_latency_s = f64::INFINITY;
        score.operation_makespan_s = f64::INFINITY;
        score.rejected_reason = Some(join_optional_reason(
            score.rejected_reason.as_deref(),
            &reason,
        ));
        if !score
            .bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "calibration policy")
        {
            score.bottlenecks.push("calibration policy".to_string());
        }
    }
}

pub(super) fn apply_calibration_fit_gates_to_parallelism(
    results: &mut [ScoredParallelismConfig],
    policy: &CalibrationPolicy,
) {
    for score in results {
        let violations = calibration_fit_gate_violations(policy, &score.calibration_fits);
        let hard_reason = hard_calibration_gate_reason(&violations);
        score.calibration_gate_violations.extend(violations);
        if let Some(reason) = hard_reason {
            score.feasible = false;
            score.estimated_latency_s = f64::INFINITY;
            score.operation_makespan_s = f64::INFINITY;
            score.rejected_reason = Some(join_optional_reason(
                score.rejected_reason.as_deref(),
                &reason,
            ));
            if !score
                .bottlenecks
                .iter()
                .any(|bottleneck| bottleneck == "calibration fit policy")
            {
                score.bottlenecks.push("calibration fit policy".to_string());
            }
        }
    }
}

pub(super) fn apply_calibration_gates_to_serving(
    results: &mut [ScoredServingConfig],
    violations: &[CalibrationGateViolation],
) {
    let hard_violations: Vec<_> = violations
        .iter()
        .filter(|violation| violation.action == CalibrationGateMode::Reject)
        .collect();
    if hard_violations.is_empty() {
        return;
    }
    let reason = hard_violations
        .iter()
        .map(|violation| violation.message.as_str())
        .collect::<Vec<_>>()
        .join("; ");
    for score in results {
        score.feasible = false;
        score.rejected_reason = Some(join_optional_reason(
            score.rejected_reason.as_deref(),
            &reason,
        ));
        if !score
            .bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "calibration policy")
        {
            score.bottlenecks.push("calibration policy".to_string());
        }
        score
            .calibration_gate_violations
            .extend(hard_violations.iter().cloned().cloned());
        for violation in &hard_violations {
            score
                .rejections
                .push(calibration_gate_serving_rejection(violation));
        }
        score.refresh_calibration_summary();
    }
}

pub(super) fn apply_calibration_fit_gates_to_serving(
    results: &mut [ScoredServingConfig],
    policy: &CalibrationPolicy,
) {
    for score in results {
        let mut violations = calibration_fit_gate_violations(policy, &score.calibration_fits);
        violations.extend(calibration_phase_gate_violations(
            policy,
            &score.phase_calibration,
        ));
        let hard_violations: Vec<_> = violations
            .iter()
            .filter(|violation| violation.action == CalibrationGateMode::Reject)
            .cloned()
            .collect();
        score.calibration_gate_violations.extend(violations);
        score.refresh_calibration_summary();
        if hard_violations.is_empty() {
            continue;
        }
        let reason = hard_violations
            .iter()
            .map(|violation| violation.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        score.feasible = false;
        score.rejected_reason = Some(join_optional_reason(
            score.rejected_reason.as_deref(),
            &reason,
        ));
        if !score
            .bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "calibration fit policy")
        {
            score.bottlenecks.push("calibration fit policy".to_string());
        }
        for violation in &hard_violations {
            score
                .rejections
                .push(calibration_gate_serving_rejection(violation));
        }
    }
}

pub(super) fn apply_approximation_policy_to_parallelism(
    results: &mut [ScoredParallelismConfig],
    policy: &ApproximationPolicy,
) {
    for score in results {
        let violations = approximation_policy_violations(policy, &score.approximations, None);
        if violations.is_empty() {
            continue;
        }
        let reason = approximation_policy_reason(&violations);
        score.approximation_policy_violations.extend(violations);
        score.feasible = false;
        score.estimated_latency_s = f64::INFINITY;
        score.operation_makespan_s = f64::INFINITY;
        score.rejected_reason = Some(join_optional_reason(
            score.rejected_reason.as_deref(),
            &reason,
        ));
        if !score
            .bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "approximation policy")
        {
            score.bottlenecks.push("approximation policy".to_string());
        }
    }
}

pub(super) fn apply_approximation_policy_to_serving(
    results: &mut [ScoredServingConfig],
    policy: &ApproximationPolicy,
) {
    for score in results {
        let metric = serving_objective_metric_term(score.objective);
        let violations =
            approximation_policy_violations(policy, &score.approximations, Some(metric));
        if violations.is_empty() {
            continue;
        }
        let reason = approximation_policy_reason(&violations);
        score
            .approximation_policy_violations
            .extend(violations.clone());
        score.feasible = false;
        score.rejected_reason = Some(join_optional_reason(
            score.rejected_reason.as_deref(),
            &reason,
        ));
        if !score
            .bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "approximation policy")
        {
            score.bottlenecks.push("approximation policy".to_string());
        }
        for violation in &violations {
            score
                .rejections
                .push(approximation_policy_serving_rejection(violation));
        }
        score.refresh_approximation_summary();
    }
}

pub(super) fn sort_parallelism_results_after_policy(results: &mut [ScoredParallelismConfig]) {
    results.sort_by(compare_nominal_parallelism);
}

pub(super) fn sort_serving_results_after_policy(results: &mut [ScoredServingConfig]) {
    results.sort_by(compare_nominal_serving);
}

pub(super) fn approximation_policy_violations(
    policy: &ApproximationPolicy,
    approximations: &[SimulationApproximation],
    metric: Option<&str>,
) -> Vec<ApproximationPolicyViolation> {
    approximations
        .iter()
        .filter_map(|approximation| {
            let action = approximation_policy_action(policy, approximation, metric);
            (action == CalibrationGateMode::Reject).then(|| ApproximationPolicyViolation {
                metric: metric.map(str::to_string),
                phase: approximation.phase.clone(),
                category: approximation.category.clone(),
                scope: approximation.scope.clone(),
                code: approximation.code.clone(),
                action,
                message: approximation_policy_violation_message(approximation, metric),
            })
        })
        .collect()
}

pub(super) fn approximation_policy_action(
    policy: &ApproximationPolicy,
    approximation: &SimulationApproximation,
    metric: Option<&str>,
) -> CalibrationGateMode {
    let category = policy_term(&approximation.category);
    let code = policy_term(&approximation.code);
    if let Some(metric) = metric {
        for gate in &policy.metric_gates {
            if !gate.metrics.iter().any(|term| term == metric) {
                continue;
            }
            if gate.reject_codes.iter().any(|term| term == &code)
                || gate.reject_categories.iter().any(|term| term == &category)
            {
                return CalibrationGateMode::Reject;
            }
            if gate.warn_codes.iter().any(|term| term == &code)
                || gate.warn_categories.iter().any(|term| term == &category)
            {
                return CalibrationGateMode::Warn;
            }
        }
    }
    if policy.reject_codes.iter().any(|term| term == &code)
        || policy
            .reject_categories
            .iter()
            .any(|term| term == &category)
    {
        return CalibrationGateMode::Reject;
    }
    if policy.warn_codes.iter().any(|term| term == &code)
        || policy.warn_categories.iter().any(|term| term == &category)
    {
        return CalibrationGateMode::Warn;
    }
    policy.default_action
}

pub(super) fn approximation_policy_violation_message(
    approximation: &SimulationApproximation,
    metric: Option<&str>,
) -> String {
    let metric = metric
        .map(|metric| format!(" while evaluating metric '{metric}'"))
        .unwrap_or_default();
    format!(
        "approximation {} for {}/{}/{}{} is rejected by approximation_policy: {}",
        approximation.code,
        approximation.phase,
        approximation.category,
        approximation.scope,
        metric,
        approximation.message
    )
}

pub(super) fn serving_objective_metric_term(objective: ServingObjective) -> &'static str {
    match objective {
        ServingObjective::MinimizeE2el => "e2el",
        ServingObjective::MinimizeTtft => "ttft",
        ServingObjective::MinimizeTpot => "tpot",
        ServingObjective::MaximizeThroughput => "throughput",
        ServingObjective::MinimizeSloMissRate => "slo_miss_rate",
        ServingObjective::MinimizeMemoryPressure => "memory_pressure",
        ServingObjective::MinimizeCost => "cost",
        ServingObjective::MinimizeEnergy => "energy",
        ServingObjective::MinimizePower => "power",
    }
}

pub(super) fn policy_term(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

pub(super) fn approximation_policy_reason(violations: &[ApproximationPolicyViolation]) -> String {
    violations
        .iter()
        .map(|violation| violation.message.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

pub(super) fn calibration_fit_gate_violations(
    policy: &CalibrationPolicy,
    applications: &[CalibrationFitApplication],
) -> Vec<CalibrationGateViolation> {
    let mut violations = Vec::new();
    for application in applications {
        let fit_name = application
            .fit_name
            .as_deref()
            .unwrap_or(application.target.as_str());
        if let Some(min_score) = policy.min_fit_confidence_score
            && application.confidence_score < min_score
        {
            violations.push(CalibrationGateViolation {
                code: "fit_confidence_below_min".to_string(),
                action: policy.fit_confidence,
                observed: Some(application.confidence_score),
                limit: Some(min_score),
                message: format!(
                    "calibration fit {fit_name} for {} {} confidence {:.3} is below required minimum {:.3}",
                    application.phase,
                    application.target,
                    application.confidence_score,
                    min_score
                ),
            });
        }
        if let Some(min_level) = policy.min_fit_confidence_level
            && application.confidence_level.unwrap_or(0.0) < min_level
        {
            violations.push(CalibrationGateViolation {
                code: "fit_confidence_level_below_min".to_string(),
                action: policy.fit_confidence,
                observed: application.confidence_level,
                limit: Some(min_level),
                message: format!(
                    "calibration fit {fit_name} for {} {} confidence_level {} is below required minimum {:.3}",
                    application.phase,
                    application.target,
                    application
                        .confidence_level
                        .map(|level| format!("{level:.3}"))
                        .unwrap_or_else(|| "missing".to_string()),
                    min_level
                ),
            });
        }
        if application.applicability_status == "extrapolated" {
            violations.push(CalibrationGateViolation {
                code: "fit_extrapolated".to_string(),
                action: policy.fit_extrapolation,
                observed: Some(application.max_extrapolation_ratio),
                limit: Some(0.0),
                message: format!(
                    "calibration fit {fit_name} for {} {} extrapolates beyond feature ranges by max ratio {:.3}",
                    application.phase, application.target, application.max_extrapolation_ratio
                ),
            });
        }
        if application.applicability_status == "partially_bounded" {
            violations.push(CalibrationGateViolation {
                code: "fit_partially_bounded".to_string(),
                action: policy.fit_partially_bounded,
                observed: Some(application.max_extrapolation_ratio),
                limit: Some(0.0),
                message: format!(
                    "calibration fit {fit_name} for {} {} has incomplete feature range metadata and is only partially bounded",
                    application.phase, application.target
                ),
            });
        }
        if application.applicability_status == "unbounded" {
            violations.push(CalibrationGateViolation {
                code: "fit_unbounded".to_string(),
                action: policy.fit_unbounded,
                observed: Some(0.0),
                limit: Some(1.0),
                message: format!(
                    "calibration fit {fit_name} for {} {} has no feature range metadata and cannot prove interpolation",
                    application.phase, application.target
                ),
            });
        }
        if let Some(min_samples) = policy.min_fit_sample_count
            && application.sample_count.unwrap_or(0) < min_samples
        {
            violations.push(CalibrationGateViolation {
                code: "fit_sample_count_below_min".to_string(),
                action: policy.fit_sample_count,
                observed: application.sample_count.map(f64::from),
                limit: Some(f64::from(min_samples)),
                message: format!(
                    "calibration fit {fit_name} for {} {} sample_count {} is below required minimum {}",
                    application.phase,
                    application.target,
                    optional_count_label(application.sample_count),
                    min_samples
                ),
            });
        }
        if let Some(min_validation_samples) = policy.min_fit_validation_sample_count
            && application.validation_sample_count.unwrap_or(0) < min_validation_samples
        {
            violations.push(CalibrationGateViolation {
                code: "fit_validation_sample_count_below_min".to_string(),
                action: policy.fit_validation_sample_count,
                observed: application.validation_sample_count.map(f64::from),
                limit: Some(f64::from(min_validation_samples)),
                message: format!(
                    "calibration fit {fit_name} for {} {} validation_sample_count {} is below required minimum {}",
                    application.phase,
                    application.target,
                    optional_count_label(application.validation_sample_count),
                    min_validation_samples
                ),
            });
        }
        if !has_non_empty_metadata(application.source.as_deref()) {
            violations.push(CalibrationGateViolation {
                code: "fit_source_unspecified".to_string(),
                action: policy.fit_source,
                observed: Some(0.0),
                limit: Some(1.0),
                message: format!(
                    "calibration fit {fit_name} for {} {} is missing source metadata",
                    application.phase, application.target
                ),
            });
        }
        if !has_fit_uncertainty_metadata(application) {
            violations.push(CalibrationGateViolation {
                code: "fit_uncertainty_unspecified".to_string(),
                action: policy.fit_uncertainty,
                observed: Some(0.0),
                limit: Some(1.0),
                message: format!(
                    "calibration fit {fit_name} for {} {} is missing uncertainty metadata",
                    application.phase, application.target
                ),
            });
        }
        if let Some(max_relative_pct) = policy.max_fit_relative_uncertainty_pct
            && let Some(relative_pct) = application.relative_uncertainty_pct
            && relative_pct > max_relative_pct
        {
            violations.push(CalibrationGateViolation {
                code: "fit_relative_uncertainty_above_max".to_string(),
                action: policy.fit_uncertainty,
                observed: Some(relative_pct),
                limit: Some(max_relative_pct),
                message: format!(
                    "calibration fit {fit_name} for {} {} relative uncertainty {:.3}% exceeds configured maximum {:.3}%",
                    application.phase, application.target, relative_pct, max_relative_pct
                ),
            });
        }
        if let Some(max_absolute_s) = policy.max_fit_absolute_uncertainty_s
            && let Some(absolute_s) = application.absolute_uncertainty_s
            && absolute_s > max_absolute_s
        {
            violations.push(CalibrationGateViolation {
                code: "fit_absolute_uncertainty_above_max".to_string(),
                action: policy.fit_uncertainty,
                observed: Some(absolute_s * 1000.0),
                limit: Some(max_absolute_s * 1000.0),
                message: format!(
                    "calibration fit {fit_name} for {} {} absolute uncertainty {:.3} ms exceeds configured maximum {:.3} ms",
                    application.phase,
                    application.target,
                    absolute_s * 1000.0,
                    max_absolute_s * 1000.0
                ),
            });
        }
    }
    violations
}

pub(super) fn calibration_phase_gate_violations(
    policy: &CalibrationPolicy,
    phases: &[ServingPhaseCalibrationObservation],
) -> Vec<CalibrationGateViolation> {
    let active_phase_count = phases.iter().filter(|phase| phase.active).count();
    let calibrated_phase_count = phases
        .iter()
        .filter(|phase| phase.active && phase.calibrated)
        .count();
    let mut violations = Vec::new();

    if let Some(min_coverage) = policy.min_serving_phase_coverage_fraction
        && active_phase_count > 0
    {
        let coverage = calibrated_phase_count as f64 / active_phase_count as f64;
        if coverage < min_coverage {
            violations.push(CalibrationGateViolation {
                code: "serving_phase_coverage_below_min".to_string(),
                action: policy.coverage,
                observed: Some(coverage),
                limit: Some(min_coverage),
                message: format!(
                    "serving phase calibration coverage {coverage:.3} is below required minimum {min_coverage:.3} ({calibrated_phase_count}/{active_phase_count} active components calibrated)"
                ),
            });
        }
    }

    if !policy.require_phase_coverage {
        return violations;
    }

    violations.extend(phases
        .iter()
        .filter(|phase| {
            phase.active
                && !phase.calibrated
                && phase.status != "uncalibrated_no_profile"
                && phase.status != "inactive"
        })
        .map(|phase| {
            let estimated_ms = phase.estimated_s * 1_000.0;
            CalibrationGateViolation {
                code: "active_serving_phase_coverage_missing".to_string(),
                action: policy.coverage,
                observed: Some(f64::from(phase.fit_count)),
                limit: Some(1.0),
                message: format!(
                    "active serving calibration component {} is {}; no applied fit covered its estimated {:.3} ms contribution",
                    phase.phase, phase.status, estimated_ms
                ),
            }
        }));
    violations
}

pub(super) fn has_non_empty_metadata(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.trim().is_empty())
}

pub(super) fn has_fit_uncertainty_metadata(application: &CalibrationFitApplication) -> bool {
    application.relative_uncertainty_pct.is_some()
        || application.absolute_uncertainty_s.is_some()
        || application.absolute_uncertainty_value.is_some()
        || application.confidence_interval.is_some()
        || application.confidence_interval_pct.is_some()
        || application.confidence_level.is_some()
        || has_non_empty_metadata(application.uncertainty_source.as_deref())
}

pub(super) fn optional_count_label(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "missing".to_string())
}

pub(super) fn apply_uncertainty_adjusted_ranking_to_parallelism(
    results: &mut [ScoredParallelismConfig],
    policy: &CalibrationPolicy,
) {
    let weight = policy.uncertainty_ranking_weight;
    if weight <= 0.0 {
        return;
    }
    results.sort_by(|a, b| compare_uncertainty_adjusted_parallelism(a, b, weight));
}

pub(super) fn apply_uncertainty_adjusted_ranking_to_serving(
    results: &mut [ScoredServingConfig],
    policy: &CalibrationPolicy,
) {
    let weight = policy.uncertainty_ranking_weight;
    if weight <= 0.0 {
        return;
    }
    results.sort_by(|a, b| compare_uncertainty_adjusted_serving(a, b, weight));
}

pub(super) fn uncertainty_adjusted_parallelism_latency_s(
    score: &ScoredParallelismConfig,
    weight: f64,
) -> f64 {
    let uncertainty = calibration_uncertainty_summary(score.calibration_fits.iter());
    score.estimated_latency_s + weight * uncertainty.absolute_uncertainty_s.unwrap_or(0.0)
}

pub(super) fn compare_nominal_parallelism(
    a: &ScoredParallelismConfig,
    b: &ScoredParallelismConfig,
) -> Ordering {
    b.feasible
        .cmp(&a.feasible)
        .then_with(|| a.estimated_latency_s.total_cmp(&b.estimated_latency_s))
        .then_with(|| {
            a.estimated_memory_per_gpu
                .as_bytes()
                .cmp(&b.estimated_memory_per_gpu.as_bytes())
        })
        .then_with(|| a.config.total_ranks().cmp(&b.config.total_ranks()))
}

pub(super) fn compare_uncertainty_adjusted_parallelism(
    a: &ScoredParallelismConfig,
    b: &ScoredParallelismConfig,
    weight: f64,
) -> Ordering {
    b.feasible
        .cmp(&a.feasible)
        .then_with(|| {
            uncertainty_adjusted_parallelism_latency_s(a, weight)
                .total_cmp(&uncertainty_adjusted_parallelism_latency_s(b, weight))
        })
        .then_with(|| a.estimated_latency_s.total_cmp(&b.estimated_latency_s))
        .then_with(|| {
            a.estimated_memory_per_gpu
                .as_bytes()
                .cmp(&b.estimated_memory_per_gpu.as_bytes())
        })
        .then_with(|| a.config.total_ranks().cmp(&b.config.total_ranks()))
}

pub(super) fn parallelism_nominal_rank_map(
    results: &[ScoredParallelismConfig],
) -> BTreeMap<String, usize> {
    let mut ordered: Vec<_> = results.iter().collect();
    ordered.sort_by(|a, b| compare_nominal_parallelism(a, b));
    ordered
        .into_iter()
        .enumerate()
        .map(|(idx, score)| (parallelism_candidate_id(score), idx + 1))
        .collect()
}

pub(super) fn parallelism_uncertainty_adjusted_rank_map(
    results: &[ScoredParallelismConfig],
    weight: f64,
) -> BTreeMap<String, usize> {
    let mut ordered: Vec<_> = results.iter().collect();
    ordered.sort_by(|a, b| compare_uncertainty_adjusted_parallelism(a, b, weight));
    ordered
        .into_iter()
        .enumerate()
        .map(|(idx, score)| (parallelism_candidate_id(score), idx + 1))
        .collect()
}

pub(super) fn compare_uncertainty_adjusted_serving_objective(
    a: &ScoredServingConfig,
    b: &ScoredServingConfig,
    weight: f64,
) -> Ordering {
    uncertainty_adjusted_serving_objective_score(a, weight)
        .total_cmp(&uncertainty_adjusted_serving_objective_score(b, weight))
}

pub(super) fn uncertainty_adjusted_serving_objective_score(
    score: &ScoredServingConfig,
    weight: f64,
) -> f64 {
    serving_uncertainty_adjusted_base_score(score, weight)
        + score.slo_miss_penalty_score
        + score.topology_risk_penalty_score
        + score.service_backpressure_penalty_score
}

pub(super) fn serving_uncertainty_adjusted_base_score(
    score: &ScoredServingConfig,
    weight: f64,
) -> f64 {
    match score.objective {
        ServingObjective::MinimizeE2el => {
            uncertainty_adjusted_latency_s(score.metrics.e2el_s, &score.calibration_fits, weight)
        }
        ServingObjective::MinimizeTtft => {
            uncertainty_adjusted_latency_s(score.metrics.ttft_s, &score.calibration_fits, weight)
        }
        ServingObjective::MinimizeTpot => uncertainty_adjusted_phase_latency_s(
            score.metrics.tpot_s,
            &score.calibration_fits,
            "decode",
            weight,
        ),
        ServingObjective::MaximizeThroughput => {
            let uncertainty = calibration_uncertainty_summary(score.calibration_fits.iter());
            let conservative_throughput = score.metrics.throughput_tokens_per_s
                - weight
                    * metric_relative_uncertainty_value(
                        score.metrics.throughput_tokens_per_s,
                        &uncertainty,
                    )
                    .unwrap_or(0.0);
            -conservative_throughput
        }
        ServingObjective::MinimizeSloMissRate => {
            let uncertainty = calibration_uncertainty_summary(score.calibration_fits.iter());
            serving_slo_miss_score(&score.metrics)
                + weight * uncertainty.relative_uncertainty_pct.unwrap_or(0.0) / 100.0
        }
        ServingObjective::MinimizeMemoryPressure => serving_peak_memory_pressure_fraction(score),
        ServingObjective::MinimizeCost
        | ServingObjective::MinimizeEnergy
        | ServingObjective::MinimizePower => serving_objective_base_score(score),
    }
}

pub(super) fn nominal_serving_objective_score(score: &ScoredServingConfig) -> f64 {
    serving_objective_base_score(score)
        + score.slo_miss_penalty_score
        + score.service_backpressure_penalty_score
        + score.topology_risk_penalty_score
}

pub(super) fn serving_objective_base_score(score: &ScoredServingConfig) -> f64 {
    match score.objective {
        ServingObjective::MinimizeE2el => score.metrics.e2el_s,
        ServingObjective::MinimizeTtft => score.metrics.ttft_s,
        ServingObjective::MinimizeTpot => score.metrics.tpot_s,
        ServingObjective::MaximizeThroughput => -score.metrics.throughput_tokens_per_s,
        ServingObjective::MinimizeSloMissRate => serving_slo_miss_score(&score.metrics),
        ServingObjective::MinimizeMemoryPressure => serving_peak_memory_pressure_fraction(score),
        ServingObjective::MinimizeCost => {
            optional_serving_objective_score(score.cost_estimate.total_cost_usd)
        }
        ServingObjective::MinimizeEnergy => {
            optional_serving_objective_score(score.cost_estimate.energy_kwh)
        }
        ServingObjective::MinimizePower => {
            optional_serving_objective_score(score.cost_estimate.average_power_watts)
        }
    }
}

pub(super) struct ServingObjectiveMetricDescriptor {
    pub(super) metric: &'static str,
    pub(super) direction: &'static str,
    pub(super) unit: &'static str,
    pub(super) value: f64,
}

pub(super) fn serving_objective_metric_descriptor(
    score: &ScoredServingConfig,
) -> ServingObjectiveMetricDescriptor {
    match score.objective {
        ServingObjective::MinimizeE2el => ServingObjectiveMetricDescriptor {
            metric: "e2el_s",
            direction: "minimize",
            unit: "seconds",
            value: score.metrics.e2el_s,
        },
        ServingObjective::MinimizeTtft => ServingObjectiveMetricDescriptor {
            metric: "ttft_s",
            direction: "minimize",
            unit: "seconds",
            value: score.metrics.ttft_s,
        },
        ServingObjective::MinimizeTpot => ServingObjectiveMetricDescriptor {
            metric: "tpot_s",
            direction: "minimize",
            unit: "seconds_per_output_token",
            value: score.metrics.tpot_s,
        },
        ServingObjective::MaximizeThroughput => ServingObjectiveMetricDescriptor {
            metric: "throughput_tokens_per_s",
            direction: "maximize",
            unit: "tokens_per_second",
            value: score.metrics.throughput_tokens_per_s,
        },
        ServingObjective::MinimizeSloMissRate => ServingObjectiveMetricDescriptor {
            metric: "aggregate_slo_miss_rate",
            direction: "minimize",
            unit: "fraction",
            value: serving_slo_miss_score(&score.metrics),
        },
        ServingObjective::MinimizeMemoryPressure => ServingObjectiveMetricDescriptor {
            metric: "memory_pressure_peak_fraction",
            direction: "minimize",
            unit: "fraction",
            value: serving_peak_memory_pressure_fraction(score),
        },
        ServingObjective::MinimizeCost => ServingObjectiveMetricDescriptor {
            metric: "total_cost_usd",
            direction: "minimize",
            unit: "usd",
            value: optional_serving_objective_score(score.cost_estimate.total_cost_usd),
        },
        ServingObjective::MinimizeEnergy => ServingObjectiveMetricDescriptor {
            metric: "energy_kwh",
            direction: "minimize",
            unit: "kwh",
            value: optional_serving_objective_score(score.cost_estimate.energy_kwh),
        },
        ServingObjective::MinimizePower => ServingObjectiveMetricDescriptor {
            metric: "average_power_watts",
            direction: "minimize",
            unit: "watts",
            value: optional_serving_objective_score(score.cost_estimate.average_power_watts),
        },
    }
}

pub(super) fn serving_objective_penalty_score(score: &ScoredServingConfig) -> f64 {
    score.slo_miss_penalty_score
        + score.service_backpressure_penalty_score
        + score.topology_risk_penalty_score
}

pub(super) fn serving_largest_nominal_objective_term(score: &ScoredServingConfig) -> &'static str {
    [
        ("base_objective", serving_objective_base_score(score).abs()),
        ("slo_miss_penalty", score.slo_miss_penalty_score.abs()),
        (
            "service_backpressure_penalty",
            score.service_backpressure_penalty_score.abs(),
        ),
        (
            "topology_risk_penalty",
            score.topology_risk_penalty_score.abs(),
        ),
    ]
    .into_iter()
    .filter(|(_, value)| value.is_finite())
    .max_by(|(_, a), (_, b)| a.total_cmp(b))
    .map(|(term, _)| term)
    .unwrap_or("unavailable")
}

pub(super) fn serving_objective_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let nominal_score = nominal_serving_objective_score(score);
    if !nominal_score.is_finite() {
        return None;
    }
    Some(format!(
        "objective_score={:.4}:base={:.4}:penalty={:.4}",
        nominal_score,
        serving_objective_base_score(score),
        serving_objective_penalty_score(score)
    ))
}

pub(super) fn optional_serving_objective_score(value: Option<f64>) -> f64 {
    value
        .filter(|value| value.is_finite())
        .unwrap_or(f64::INFINITY)
}

pub(super) fn serving_peak_memory_pressure_fraction(score: &ScoredServingConfig) -> f64 {
    let peak = score
        .memory_pressure
        .iter()
        .filter_map(|observation| {
            observation
                .capacity_used_fraction
                .is_finite()
                .then_some(observation.capacity_used_fraction)
        })
        .chain([
            score.prefill_memory.capacity_used_fraction(),
            score.decode_memory.capacity_used_fraction(),
        ])
        .filter(|fraction| fraction.is_finite())
        .fold(None, |peak: Option<f64>, fraction| {
            Some(peak.map_or(fraction, |peak| peak.max(fraction)))
        });

    peak.unwrap_or(f64::INFINITY)
}

pub(super) fn serving_peak_memory_pressure_phase(score: &ScoredServingConfig) -> Option<&str> {
    score
        .memory_pressure
        .iter()
        .filter(|observation| observation.capacity_used_fraction.is_finite())
        .max_by(|left, right| {
            left.capacity_used_fraction
                .total_cmp(&right.capacity_used_fraction)
        })
        .map(|observation| observation.phase.as_str())
}

pub(super) fn compare_nominal_serving(
    a: &ScoredServingConfig,
    b: &ScoredServingConfig,
) -> Ordering {
    b.feasible
        .cmp(&a.feasible)
        .then_with(|| {
            nominal_serving_objective_score(a).total_cmp(&nominal_serving_objective_score(b))
        })
        .then_with(|| {
            serving_peak_memory_pressure_fraction(a)
                .total_cmp(&serving_peak_memory_pressure_fraction(b))
        })
        .then_with(|| a.metrics.e2el_s.total_cmp(&b.metrics.e2el_s))
        .then_with(|| a.metrics.tpot_s.total_cmp(&b.metrics.tpot_s))
        .then_with(|| a.metrics.ttft_s.total_cmp(&b.metrics.ttft_s))
        .then_with(|| {
            b.metrics
                .throughput_tokens_per_s
                .total_cmp(&a.metrics.throughput_tokens_per_s)
        })
        .then_with(|| {
            a.prefill_config
                .total_ranks()
                .cmp(&b.prefill_config.total_ranks())
        })
        .then_with(|| {
            a.decode_config
                .total_ranks()
                .cmp(&b.decode_config.total_ranks())
        })
        .then_with(|| a.prefill_nodes.cmp(&b.prefill_nodes))
        .then_with(|| a.decode_nodes.cmp(&b.decode_nodes))
        .then_with(|| a.pool_label.cmp(&b.pool_label))
}

pub(super) fn compare_uncertainty_adjusted_serving(
    a: &ScoredServingConfig,
    b: &ScoredServingConfig,
    weight: f64,
) -> Ordering {
    b.feasible
        .cmp(&a.feasible)
        .then_with(|| compare_uncertainty_adjusted_serving_objective(a, b, weight))
        .then_with(|| {
            serving_peak_memory_pressure_fraction(a)
                .total_cmp(&serving_peak_memory_pressure_fraction(b))
        })
        .then_with(|| a.metrics.e2el_s.total_cmp(&b.metrics.e2el_s))
        .then_with(|| a.metrics.tpot_s.total_cmp(&b.metrics.tpot_s))
        .then_with(|| a.metrics.ttft_s.total_cmp(&b.metrics.ttft_s))
        .then_with(|| {
            b.metrics
                .throughput_tokens_per_s
                .total_cmp(&a.metrics.throughput_tokens_per_s)
        })
        .then_with(|| {
            a.prefill_config
                .total_ranks()
                .cmp(&b.prefill_config.total_ranks())
        })
        .then_with(|| {
            a.decode_config
                .total_ranks()
                .cmp(&b.decode_config.total_ranks())
        })
        .then_with(|| a.prefill_nodes.cmp(&b.prefill_nodes))
        .then_with(|| a.decode_nodes.cmp(&b.decode_nodes))
        .then_with(|| a.pool_label.cmp(&b.pool_label))
}

pub(super) fn serving_nominal_rank_map(results: &[ScoredServingConfig]) -> BTreeMap<String, usize> {
    let mut ordered: Vec<_> = results.iter().collect();
    ordered.sort_by(|a, b| compare_nominal_serving(a, b));
    ordered
        .into_iter()
        .enumerate()
        .map(|(idx, score)| (score.candidate_id.clone(), idx + 1))
        .collect()
}

pub(super) fn serving_uncertainty_adjusted_rank_map(
    results: &[ScoredServingConfig],
    weight: f64,
) -> BTreeMap<String, usize> {
    let mut ordered: Vec<_> = results.iter().collect();
    ordered.sort_by(|a, b| compare_uncertainty_adjusted_serving(a, b, weight));
    ordered
        .into_iter()
        .enumerate()
        .map(|(idx, score)| (score.candidate_id.clone(), idx + 1))
        .collect()
}

pub(super) fn uncertainty_adjusted_latency_s(
    value_s: f64,
    applications: &[CalibrationFitApplication],
    weight: f64,
) -> f64 {
    let uncertainty = calibration_uncertainty_summary(applications.iter());
    value_s + weight * metric_uncertainty_value(value_s, &uncertainty).unwrap_or(0.0)
}

pub(super) fn uncertainty_adjusted_phase_latency_s(
    value_s: f64,
    applications: &[CalibrationFitApplication],
    phase: &str,
    weight: f64,
) -> f64 {
    let uncertainty = calibration_phase_uncertainty_summary(applications.iter(), phase);
    value_s + weight * metric_uncertainty_value(value_s, &uncertainty).unwrap_or(0.0)
}

pub(super) fn serving_slo_miss_score(metrics: &ServingMetrics) -> f64 {
    [
        metrics.ttft_slo_miss_rate,
        metrics.tpot_slo_miss_rate,
        metrics.itl_slo_miss_rate,
        metrics.e2el_slo_miss_rate,
        metrics.deadline_miss_rate,
    ]
    .into_iter()
    .filter(|value| value.is_finite())
    .sum()
}

pub(super) fn hard_calibration_gate_reason(
    violations: &[CalibrationGateViolation],
) -> Option<String> {
    let messages: Vec<_> = violations
        .iter()
        .filter(|violation| violation.action == CalibrationGateMode::Reject)
        .map(|violation| violation.message.as_str())
        .collect();
    if messages.is_empty() {
        None
    } else {
        Some(messages.join("; "))
    }
}

pub(super) fn join_optional_reason(existing: Option<&str>, reason: &str) -> String {
    match existing {
        Some(existing) if !existing.is_empty() => format!("{existing}; {reason}"),
        _ => reason.to_string(),
    }
}

pub(super) fn calibration_gate_serving_rejection(
    violation: &CalibrationGateViolation,
) -> ServingRejection {
    ServingRejection {
        phase: "calibration".to_string(),
        category: "policy".to_string(),
        resource: calibration_gate_resource(&violation.code).to_string(),
        code: violation.code.clone(),
        observed: violation.observed,
        limit: violation.limit,
        unit: violation.limit.map(|_| "score".to_string()),
        remediation: Some(calibration_gate_remediation(&violation.code).to_string()),
        message: violation.message.clone(),
    }
}

pub(super) fn approximation_policy_serving_rejection(
    violation: &ApproximationPolicyViolation,
) -> ServingRejection {
    ServingRejection {
        phase: violation.phase.clone(),
        category: "approximation_policy".to_string(),
        resource: violation.scope.clone(),
        code: format!("approximation_policy_reject_{}", violation.code),
        observed: None,
        limit: None,
        unit: None,
        remediation: Some(
            "allow this approximation in approximation_policy, choose a more detailed model, or adjust the experiment to avoid this approximation"
                .to_string(),
        ),
        message: violation.message.clone(),
    }
}

pub(super) fn calibration_gate_resource(code: &str) -> &'static str {
    if code.contains("coverage") || code.contains("phase") {
        "coverage"
    } else if code.contains("fit") {
        "calibration_fit"
    } else if code.contains("invalid_shape") {
        "invalid_shape"
    } else if code.contains("valid_shape") {
        "valid_shape"
    } else {
        "calibration"
    }
}

pub(super) fn calibration_gate_remediation(code: &str) -> &'static str {
    if code.contains("coverage") || code.contains("phase") {
        "add benchmark points for this workload shape and required serving phases or lower calibration_policy coverage requirements"
    } else if code.contains("fit") {
        "add calibration fit feature ranges for this candidate shape, collect benchmark data for the extrapolated region, or set the relevant calibration_policy fit gate to warn"
    } else if code.contains("invalid_shape") || code.contains("valid_shape") {
        "adjust workload shape, extend the calibration profile envelope, or set the relevant calibration_policy gate to warn"
    } else {
        "provide a calibration profile or set calibration_policy coverage to warn"
    }
}

pub(super) fn write_calibration_gate_text<W: Write>(
    writer: &mut W,
    violations: &[CalibrationGateViolation],
) -> Result<(), io::Error> {
    if violations.is_empty() {
        return Ok(());
    }
    let hard_count = violations
        .iter()
        .filter(|violation| violation.action == CalibrationGateMode::Reject)
        .count();
    writeln!(
        writer,
        "calibration_gate_violations={} hard_rejects={}",
        violations.len(),
        hard_count
    )?;
    for violation in violations {
        writeln!(
            writer,
            "calibration_gate action={} code={} observed={} limit={} message={}",
            violation.action.as_str(),
            violation.code,
            json_optional_value(violation.observed),
            json_optional_value(violation.limit),
            violation.message
        )?;
    }
    Ok(())
}
