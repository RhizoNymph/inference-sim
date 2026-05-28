use super::*;

pub(super) fn serving_calibration_fits(
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    simulation: &ServingSimulation,
) -> Vec<CalibrationFitApplication> {
    let mut fits = Vec::new();
    fits.extend(prefill_score.calibration_fits.clone());
    fits.extend(decode_score.calibration_fits.clone());
    fits.extend(decode_one_score.calibration_fits.clone());
    fits.extend(simulation.calibration_fits.clone());
    fits
}

pub(super) fn serving_phase_calibration(
    calibration_profile: Option<&CalibrationProfileMetadata>,
    metrics: &ServingMetrics,
    observations: &[ServingRequestObservation],
    calibration_fits: &[CalibrationFitApplication],
) -> Vec<ServingPhaseCalibrationObservation> {
    let kv_transfer_active = metrics.kv_transfer_s > 0.0
        || observations
            .iter()
            .any(|observation| observation.kv_transfer_bytes > 0);
    let mut components = vec![
        ("prefill", metrics.prefill_s, metrics.prefill_s > 0.0),
        ("decode", metrics.decode_s, metrics.decode_s > 0.0),
        ("kv_transfer", metrics.kv_transfer_s, kv_transfer_active),
    ];
    push_active_queue_calibration_component(
        &mut components,
        "prefill_queue",
        metrics.prefill_worker_queue_s + metrics.prefill_resource_queue_s,
    );
    push_active_queue_calibration_component(
        &mut components,
        "decode_queue",
        metrics.decode_worker_queue_s + metrics.decode_resource_queue_s,
    );
    push_active_queue_calibration_component(
        &mut components,
        "kv_worker_queue",
        metrics.kv_worker_queue_s,
    );
    push_active_queue_calibration_component(
        &mut components,
        "kv_route_resource_queue",
        metrics.kv_resource_queue_s,
    );

    components
        .into_iter()
        .map(|(phase, estimated_s, active)| {
            let applied_targets = calibration_fits
                .iter()
                .filter(|fit| fit.phase == phase)
                .map(|fit| fit.target.clone())
                .collect::<Vec<_>>();
            let calibrated = active && !applied_targets.is_empty();
            let status = if !active {
                "inactive"
            } else if calibrated {
                "calibrated"
            } else if calibration_profile.is_some() {
                "uncalibrated_no_fit"
            } else {
                "uncalibrated_no_profile"
            };
            ServingPhaseCalibrationObservation {
                phase: phase.to_string(),
                active,
                calibrated,
                fit_count: applied_targets.len().min(u32::MAX as usize) as u32,
                applied_targets,
                estimated_s,
                status: status.to_string(),
            }
        })
        .collect()
}

pub(super) fn serving_calibration_summary(
    phases: &[ServingPhaseCalibrationObservation],
    fits: &[CalibrationFitApplication],
    gate_violations: &[CalibrationGateViolation],
) -> ServingCalibrationSummary {
    let active_phase_count = phases.iter().filter(|phase| phase.active).count() as u32;
    let calibrated_phase_count = phases
        .iter()
        .filter(|phase| phase.active && phase.calibrated)
        .count() as u32;
    let uncalibrated_phase_count = active_phase_count.saturating_sub(calibrated_phase_count);
    let coverage_fraction = if active_phase_count > 0 {
        f64::from(calibrated_phase_count) / f64::from(active_phase_count)
    } else {
        0.0
    };
    let fit_count = fits.len().min(u32::MAX as usize) as u32;
    let extrapolated_fit_count = fits
        .iter()
        .filter(|fit| fit.applicability_status == "extrapolated")
        .count()
        .min(u32::MAX as usize) as u32;
    let unbounded_fit_count = fits
        .iter()
        .filter(|fit| fit.applicability_status == "unbounded")
        .count()
        .min(u32::MAX as usize) as u32;
    let fit_count_with_uncertainty = fits
        .iter()
        .filter(|fit| fit_has_numeric_uncertainty(fit))
        .count()
        .min(u32::MAX as usize) as u32;
    let min_confidence_score = fits
        .iter()
        .filter_map(|fit| {
            fit.confidence_score
                .is_finite()
                .then_some(fit.confidence_score)
        })
        .fold(None, |min_score: Option<f64>, score| {
            Some(min_score.map_or(score, |min_score| min_score.min(score)))
        });
    let max_extrapolation_ratio = fits
        .iter()
        .filter_map(|fit| {
            fit.max_extrapolation_ratio
                .is_finite()
                .then_some(fit.max_extrapolation_ratio)
        })
        .fold(None, |max_ratio: Option<f64>, ratio| {
            Some(max_ratio.map_or(ratio, |max_ratio| max_ratio.max(ratio)))
        });
    let predicted_latency_s = fits
        .iter()
        .filter(|fit| fit.prediction_kind == "latency")
        .filter_map(|fit| {
            fit.predicted_s
                .is_finite()
                .then_some(fit.predicted_s.max(0.0))
        })
        .sum::<f64>();
    let uncertainty_s_squared = fits
        .iter()
        .filter(|fit| fit.prediction_kind == "latency")
        .filter_map(|fit| fit.absolute_uncertainty_s)
        .filter(|uncertainty_s| uncertainty_s.is_finite() && *uncertainty_s >= 0.0)
        .map(|uncertainty_s| uncertainty_s * uncertainty_s)
        .sum::<f64>();
    let latency_fit_count_with_uncertainty = fits
        .iter()
        .filter(|fit| fit.prediction_kind == "latency")
        .filter_map(|fit| fit.absolute_uncertainty_s)
        .filter(|uncertainty_s| uncertainty_s.is_finite() && *uncertainty_s >= 0.0)
        .count();
    let absolute_uncertainty_s =
        (latency_fit_count_with_uncertainty > 0).then_some(uncertainty_s_squared.sqrt());
    let latency_relative_uncertainty_pct = absolute_uncertainty_s
        .filter(|_| predicted_latency_s.is_finite() && predicted_latency_s > 0.0)
        .map(|absolute_uncertainty_s| absolute_uncertainty_s / predicted_latency_s * 100.0);
    let max_fit_relative_uncertainty_pct = fits
        .iter()
        .filter_map(|fit| fit.relative_uncertainty_pct)
        .filter(|relative_pct| relative_pct.is_finite() && *relative_pct >= 0.0)
        .fold(None, |max_pct: Option<f64>, relative_pct| {
            Some(max_pct.map_or(relative_pct, |max_pct| max_pct.max(relative_pct)))
        });
    let relative_uncertainty_pct = max_optional_f64(
        latency_relative_uncertainty_pct,
        max_fit_relative_uncertainty_pct,
    );
    let gate_violation_count = gate_violations.len().min(u32::MAX as usize) as u32;
    let hard_gate_violation_count = gate_violations
        .iter()
        .filter(|violation| violation.action.as_str() == "reject")
        .count()
        .min(u32::MAX as usize) as u32;
    let status = calibration_summary_status(
        active_phase_count,
        uncalibrated_phase_count,
        hard_gate_violation_count,
        gate_violation_count,
        fit_count,
    )
    .to_string();

    ServingCalibrationSummary {
        status,
        active_phase_count,
        calibrated_phase_count,
        uncalibrated_phase_count,
        coverage_fraction,
        fit_count,
        extrapolated_fit_count,
        unbounded_fit_count,
        fit_count_with_uncertainty,
        min_confidence_score,
        max_extrapolation_ratio,
        relative_uncertainty_pct,
        absolute_uncertainty_s,
        gate_violation_count,
        hard_gate_violation_count,
    }
}

pub(super) fn fit_has_numeric_uncertainty(fit: &CalibrationFitApplication) -> bool {
    finite_nonnegative(fit.relative_uncertainty_pct)
        || finite_nonnegative(fit.absolute_uncertainty_s)
        || finite_nonnegative(fit.absolute_uncertainty_value)
}

fn max_optional_f64(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn finite_nonnegative(value: Option<f64>) -> bool {
    value.is_some_and(|value| value.is_finite() && value >= 0.0)
}

fn calibration_summary_status(
    active_phase_count: u32,
    uncalibrated_phase_count: u32,
    hard_gate_violation_count: u32,
    gate_violation_count: u32,
    fit_count: u32,
) -> &'static str {
    if hard_gate_violation_count > 0 {
        "gate_rejected"
    } else if gate_violation_count > 0 {
        "gate_warning"
    } else if active_phase_count == 0 {
        "no_active_phases"
    } else if uncalibrated_phase_count == 0 {
        "fully_calibrated"
    } else if fit_count > 0 {
        "partially_calibrated"
    } else {
        "uncalibrated"
    }
}

fn push_active_queue_calibration_component(
    components: &mut Vec<(&'static str, f64, bool)>,
    phase: &'static str,
    estimated_s: f64,
) {
    if calibration_component_active(estimated_s) {
        components.push((phase, estimated_s, true));
    }
}

pub(super) fn calibration_component_active(estimated_s: f64) -> bool {
    estimated_s.is_finite() && estimated_s > 1e-12
}
