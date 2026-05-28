use super::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct CalibrationJsonContext<'a> {
    pub(super) calibration: SimulationCalibration,
    pub(super) policy: &'a CalibrationPolicy,
    pub(super) profile: Option<&'a CalibrationProfileMetadata>,
    pub(super) coverage: Option<&'a CalibrationCoverageReport>,
    pub(super) warnings: &'a [CalibrationApplicabilityWarning],
    pub(super) invalid_shape_warnings: &'a [CalibrationInvalidShapeWarning],
    pub(super) gate_violations: &'a [CalibrationGateViolation],
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CalibrationUncertaintySummary {
    pub(super) fit_count: usize,
    pub(super) fit_count_with_uncertainty: usize,
    pub(super) relative_uncertainty_pct: Option<f64>,
    pub(super) absolute_uncertainty_s: Option<f64>,
    pub(super) min_confidence_score: Option<f64>,
    pub(super) max_extrapolation_ratio: Option<f64>,
    pub(super) applicability_status: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RankSensitivityApproximationSummary {
    pub(super) status: String,
    pub(super) approximation_count: usize,
    pub(super) policy_violation_count: usize,
    pub(super) coarse_topology: bool,
    pub(super) approximate_queueing: bool,
    pub(super) aggregate_memory: bool,
    pub(super) uncalibrated_runtime: bool,
    pub(super) unsupported_runtime: bool,
    pub(super) top_codes: String,
}

pub(super) fn write_calibration<W: Write>(
    writer: &mut W,
    context: CalibrationJsonContext<'_>,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let calibration = context.calibration.sanitized();
    writeln!(writer, "{indent}\"calibration\": {{")?;
    writeln!(
        writer,
        "{indent}  \"compute_efficiency\": {},",
        json_f64(calibration.compute_efficiency)
    )?;
    writeln!(
        writer,
        "{indent}  \"prefill_compute_scale\": {},",
        json_f64(calibration.prefill_compute_scale)
    )?;
    writeln!(
        writer,
        "{indent}  \"decode_compute_scale\": {},",
        json_f64(calibration.decode_compute_scale)
    )?;
    writeln!(
        writer,
        "{indent}  \"decode_memory_bandwidth_scale\": {},",
        json_f64(calibration.decode_memory_bandwidth_scale)
    )?;
    writeln!(
        writer,
        "{indent}  \"collective_latency_scale\": {},",
        json_f64(calibration.collective_latency_scale)
    )?;
    writeln!(
        writer,
        "{indent}  \"collective_bandwidth_scale\": {},",
        json_f64(calibration.collective_bandwidth_scale)
    )?;
    writeln!(
        writer,
        "{indent}  \"kv_transfer_scale\": {},",
        json_f64(calibration.kv_transfer_scale)
    )?;
    writeln!(
        writer,
        "{indent}  \"scheduler_overhead_us\": {},",
        json_f64(calibration.scheduler_overhead_us)
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_memory_temporary_fraction\": {},",
        json_f64(calibration.serving_memory_temporary_fraction)
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_memory_activation_communication_fraction\": {},",
        json_f64(calibration.serving_memory_activation_communication_fraction)
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_memory_weight_communication_fraction\": {},",
        json_f64(calibration.serving_memory_weight_communication_fraction)
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_memory_runtime_reserve_fraction\": {},",
        json_f64(calibration.serving_memory_runtime_reserve_fraction)
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_memory_fragmentation_fraction\": {},",
        json_f64(calibration.serving_memory_fragmentation_fraction)
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_pipeline_depth\": {},",
        calibration.serving_pipeline_depth
    )?;
    writeln!(
        writer,
        "{indent}  \"request_arrival_gap_s\": {},",
        json_f64(calibration.request_arrival_gap_s)
    )?;
    writeln!(
        writer,
        "{indent}  \"allow_compute_comm_overlap\": {},",
        calibration.allow_compute_comm_overlap
    )?;
    write_calibration_profile(writer, context.profile, indent, true)?;
    write_calibration_coverage(writer, context.coverage, indent, true)?;
    write_calibration_policy(writer, context.policy, indent, true)?;
    writeln!(
        writer,
        "{indent}  \"applicability_status\": {},",
        json_string(calibration_applicability_status(
            context.profile,
            context.warnings,
            context.invalid_shape_warnings
        ))
    )?;
    write_calibration_invalid_shape_warnings(writer, context.invalid_shape_warnings, indent, true)?;
    write_calibration_gate_violations(writer, context.gate_violations, indent, true)?;
    write_calibration_applicability_warnings(writer, context.warnings, indent, false)?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}

fn calibration_applicability_status(
    profile: Option<&CalibrationProfileMetadata>,
    warnings: &[CalibrationApplicabilityWarning],
    invalid_shape_warnings: &[CalibrationInvalidShapeWarning],
) -> &'static str {
    let Some(profile) = profile else {
        return "no_profile";
    };
    if !invalid_shape_warnings.is_empty() {
        return "invalid_shape_overlap";
    }
    if profile.valid_shape.is_none() {
        return "no_valid_shape";
    }
    if warnings.is_empty() {
        "within_valid_shape"
    } else {
        "outside_valid_shape"
    }
}

fn write_calibration_coverage<W: Write>(
    writer: &mut W,
    coverage: Option<&CalibrationCoverageReport>,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(coverage) = coverage else {
        writeln!(
            writer,
            "{indent}  \"coverage\": null{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    writeln!(writer, "{indent}  \"coverage\": {{")?;
    writeln!(
        writer,
        "{indent}    \"benchmark_count\": {},",
        coverage.benchmark_count
    )?;
    writeln!(
        writer,
        "{indent}    \"shape_benchmark_count\": {},",
        coverage.shape_benchmark_count
    )?;
    writeln!(
        writer,
        "{indent}    \"complete_shape_benchmark_count\": {},",
        coverage.complete_shape_benchmark_count
    )?;
    let field_indent = format!("{indent}  ");
    write_string_vec(
        writer,
        &field_indent,
        "required_phases",
        &coverage.required_phases,
        true,
    )?;
    write_string_vec(
        writer,
        &field_indent,
        "covered_phases",
        &coverage.covered_phases,
        true,
    )?;
    write_string_vec(
        writer,
        &field_indent,
        "missing_phases",
        &coverage.missing_phases,
        true,
    )?;
    writeln!(
        writer,
        "{indent}    \"batch_size_score\": {},",
        json_optional_value(coverage.batch_size_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"prompt_tokens_score\": {},",
        json_optional_value(coverage.prompt_tokens_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_tokens_score\": {},",
        json_optional_value(coverage.decode_tokens_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"sequence_tokens_score\": {},",
        json_optional_value(coverage.sequence_tokens_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"shape_coverage_score\": {},",
        json_optional_value(coverage.shape_coverage_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"phase_coverage_score\": {},",
        json_optional_value(coverage.phase_coverage_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"coverage_score\": {},",
        json_optional_value(coverage.coverage_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"nearest_benchmark\": {},",
        json_optional_string(coverage.nearest_benchmark.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}    \"nearest_benchmark_distance\": {},",
        json_optional_value(coverage.nearest_benchmark_distance)
    )?;
    writeln!(
        writer,
        "{indent}    \"status\": {}",
        json_string(&coverage.status)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_calibration_policy<W: Write>(
    writer: &mut W,
    policy: &CalibrationPolicy,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"policy\": {{")?;
    writeln!(
        writer,
        "{indent}    \"valid_shape\": {},",
        json_string(policy.valid_shape.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"invalid_shape\": {},",
        json_string(policy.invalid_shape.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"coverage\": {},",
        json_string(policy.coverage.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_confidence\": {},",
        json_string(policy.fit_confidence.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_extrapolation\": {},",
        json_string(policy.fit_extrapolation.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_partially_bounded\": {},",
        json_string(policy.fit_partially_bounded.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_unbounded\": {},",
        json_string(policy.fit_unbounded.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_sample_count\": {},",
        json_string(policy.fit_sample_count.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_validation_sample_count\": {},",
        json_string(policy.fit_validation_sample_count.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_source\": {},",
        json_string(policy.fit_source.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_uncertainty\": {},",
        json_string(policy.fit_uncertainty.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"profile_source\": {},",
        json_string(policy.profile_source.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"profile_date\": {},",
        json_string(policy.profile_date.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"profile_runtime\": {},",
        json_string(policy.profile_runtime.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"min_coverage_score\": {},",
        json_optional_value(policy.min_coverage_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"min_fit_confidence_score\": {},",
        json_optional_value(policy.min_fit_confidence_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"min_fit_confidence_level\": {},",
        json_optional_value(policy.min_fit_confidence_level)
    )?;
    writeln!(
        writer,
        "{indent}    \"min_fit_sample_count\": {},",
        json_optional_u32(policy.min_fit_sample_count)
    )?;
    writeln!(
        writer,
        "{indent}    \"min_fit_validation_sample_count\": {},",
        json_optional_u32(policy.min_fit_validation_sample_count)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_fit_relative_uncertainty_pct\": {},",
        json_optional_value(policy.max_fit_relative_uncertainty_pct)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_fit_absolute_uncertainty_ms\": {},",
        json_optional_seconds_ms(policy.max_fit_absolute_uncertainty_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"min_serving_phase_coverage_fraction\": {},",
        json_optional_value(policy.min_serving_phase_coverage_fraction)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_ranking_weight\": {},",
        json_f64(policy.uncertainty_ranking_weight)
    )?;
    writeln!(
        writer,
        "{indent}    \"require_phase_coverage\": {}",
        policy.require_phase_coverage
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_approximation_policy<W: Write>(
    writer: &mut W,
    policy: &ApproximationPolicy,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"approximation_policy\": {{")?;
    writeln!(
        writer,
        "{indent}  \"preset\": {},",
        json_optional_string(policy.preset.map(|preset| preset.as_str()))
    )?;
    writeln!(
        writer,
        "{indent}  \"default_action\": {},",
        json_string(policy.default_action.as_str())
    )?;
    write_string_vec(
        writer,
        indent,
        "reject_categories",
        &policy.reject_categories,
        true,
    )?;
    write_string_vec(writer, indent, "reject_codes", &policy.reject_codes, true)?;
    write_string_vec(
        writer,
        indent,
        "warn_categories",
        &policy.warn_categories,
        true,
    )?;
    write_string_vec(writer, indent, "warn_codes", &policy.warn_codes, true)?;
    write_approximation_metric_gates(writer, indent, &policy.metric_gates, false)?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}

fn write_approximation_metric_gates<W: Write>(
    writer: &mut W,
    indent: &str,
    gates: &[ApproximationMetricGate],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"metric_gates\": [")?;
    for (idx, gate) in gates.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        write_string_vec(
            writer,
            &format!("{indent}  "),
            "metrics",
            &gate.metrics,
            true,
        )?;
        write_string_vec(
            writer,
            &format!("{indent}  "),
            "reject_categories",
            &gate.reject_categories,
            true,
        )?;
        write_string_vec(
            writer,
            &format!("{indent}  "),
            "reject_codes",
            &gate.reject_codes,
            true,
        )?;
        write_string_vec(
            writer,
            &format!("{indent}  "),
            "warn_categories",
            &gate.warn_categories,
            true,
        )?;
        write_string_vec(
            writer,
            &format!("{indent}  "),
            "warn_codes",
            &gate.warn_codes,
            false,
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < gates.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_calibration_gate_violations<W: Write>(
    writer: &mut W,
    violations: &[CalibrationGateViolation],
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    write_calibration_gate_violation_array(
        writer,
        violations,
        indent,
        "gate_violations",
        trailing_comma,
    )
}

pub(super) fn write_calibration_gate_violation_array<W: Write>(
    writer: &mut W,
    violations: &[CalibrationGateViolation],
    indent: &str,
    field: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, violation) in violations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&violation.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"action\": {},",
            json_string(violation.action.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(violation.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(violation.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {}",
            json_string(&violation.message)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < violations.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_calibration_uncertainty<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    applications: &[CalibrationFitApplication],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let summary = calibration_uncertainty_summary(applications.iter());
    writeln!(writer, "{indent}  \"{field}\": {{")?;
    write_calibration_uncertainty_fields(writer, &format!("{indent}    "), &summary, true)?;
    writeln!(writer, "{indent}    \"by_phase\": [")?;
    let mut phases: BTreeMap<&str, Vec<&CalibrationFitApplication>> = BTreeMap::new();
    for application in applications {
        phases
            .entry(application.phase.as_str())
            .or_default()
            .push(application);
    }
    for (idx, (phase, phase_applications)) in phases.iter().enumerate() {
        let phase_summary = calibration_uncertainty_summary(phase_applications.iter().copied());
        writeln!(writer, "{indent}      {{")?;
        writeln!(writer, "{indent}        \"phase\": {},", json_string(phase))?;
        write_calibration_uncertainty_fields(
            writer,
            &format!("{indent}        "),
            &phase_summary,
            false,
        )?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < phases.len()))?;
    }
    writeln!(writer, "{indent}    ]")?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_serving_calibration_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &ServingCalibrationSummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"serving_calibration_summary\": {{")?;
    writeln!(
        writer,
        "{indent}    \"status\": {},",
        json_string(&summary.status)
    )?;
    writeln!(
        writer,
        "{indent}    \"active_phase_count\": {},",
        summary.active_phase_count
    )?;
    writeln!(
        writer,
        "{indent}    \"calibrated_phase_count\": {},",
        summary.calibrated_phase_count
    )?;
    writeln!(
        writer,
        "{indent}    \"uncalibrated_phase_count\": {},",
        summary.uncalibrated_phase_count
    )?;
    writeln!(
        writer,
        "{indent}    \"coverage_fraction\": {},",
        json_optional_f64(summary.coverage_fraction)
    )?;
    writeln!(writer, "{indent}    \"fit_count\": {},", summary.fit_count)?;
    writeln!(
        writer,
        "{indent}    \"extrapolated_fit_count\": {},",
        summary.extrapolated_fit_count
    )?;
    writeln!(
        writer,
        "{indent}    \"unbounded_fit_count\": {},",
        summary.unbounded_fit_count
    )?;
    writeln!(
        writer,
        "{indent}    \"fit_count_with_uncertainty\": {},",
        summary.fit_count_with_uncertainty
    )?;
    writeln!(
        writer,
        "{indent}    \"min_confidence_score\": {},",
        json_optional_value(summary.min_confidence_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_extrapolation_ratio\": {},",
        json_optional_value(summary.max_extrapolation_ratio)
    )?;
    writeln!(
        writer,
        "{indent}    \"relative_uncertainty_pct\": {},",
        json_optional_value(summary.relative_uncertainty_pct)
    )?;
    writeln!(
        writer,
        "{indent}    \"absolute_uncertainty_ms\": {},",
        json_optional_seconds_ms(summary.absolute_uncertainty_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"gate_violation_count\": {},",
        summary.gate_violation_count
    )?;
    writeln!(
        writer,
        "{indent}    \"hard_gate_violation_count\": {}",
        summary.hard_gate_violation_count
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_serving_phase_calibration<W: Write>(
    writer: &mut W,
    indent: &str,
    phases: &[ServingPhaseCalibrationObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"serving_phase_calibration\": [")?;
    for (idx, phase) in phases.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&phase.phase)
        )?;
        writeln!(writer, "{indent}      \"active\": {},", phase.active)?;
        writeln!(
            writer,
            "{indent}      \"calibrated\": {},",
            phase.calibrated
        )?;
        writeln!(writer, "{indent}      \"fit_count\": {},", phase.fit_count)?;
        writeln!(
            writer,
            "{indent}      \"applied_targets\": [{}],",
            phase
                .applied_targets
                .iter()
                .map(|target| json_string(target))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(
            writer,
            "{indent}      \"estimated_ms\": {},",
            json_optional_seconds_ms(Some(phase.estimated_s))
        )?;
        writeln!(
            writer,
            "{indent}      \"status\": {}",
            json_string(&phase.status)
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < phases.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_calibration_uncertainty_fields<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &CalibrationUncertaintySummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"fit_count\": {},", summary.fit_count)?;
    writeln!(
        writer,
        "{indent}\"fit_count_with_uncertainty\": {},",
        summary.fit_count_with_uncertainty
    )?;
    writeln!(
        writer,
        "{indent}\"relative_uncertainty_pct\": {},",
        json_optional_value(summary.relative_uncertainty_pct)
    )?;
    writeln!(
        writer,
        "{indent}\"absolute_uncertainty_ms\": {},",
        json_optional_seconds_ms(summary.absolute_uncertainty_s)
    )?;
    writeln!(
        writer,
        "{indent}\"min_confidence_score\": {},",
        json_optional_value(summary.min_confidence_score)
    )?;
    writeln!(
        writer,
        "{indent}\"max_extrapolation_ratio\": {},",
        json_optional_value(summary.max_extrapolation_ratio)
    )?;
    writeln!(
        writer,
        "{indent}\"applicability_status\": {}{}",
        json_string(&summary.applicability_status),
        comma(trailing_comma)
    )
}

pub(super) fn calibration_uncertainty_summary<'a>(
    applications: impl Iterator<Item = &'a CalibrationFitApplication>,
) -> CalibrationUncertaintySummary {
    let mut fit_count = 0_usize;
    let mut fit_count_with_uncertainty = 0_usize;
    let mut predicted_s_total = 0.0_f64;
    let mut uncertainty_s_squared = 0.0_f64;
    let mut latency_fit_count_with_uncertainty = 0_usize;
    let mut max_relative_uncertainty_pct: Option<f64> = None;
    let mut min_confidence_score: Option<f64> = None;
    let mut max_extrapolation_ratio: Option<f64> = None;
    let mut applicability_status = "none".to_string();

    for application in applications {
        fit_count += 1;
        if application_has_numeric_uncertainty(application) {
            fit_count_with_uncertainty += 1;
        }
        let is_latency_prediction = application.prediction_kind == "latency";
        if is_latency_prediction
            && application.predicted_s.is_finite()
            && application.predicted_s > 0.0
        {
            predicted_s_total += application.predicted_s;
        }
        if is_latency_prediction
            && let Some(uncertainty_s) = application.absolute_uncertainty_s
            && uncertainty_s.is_finite()
            && uncertainty_s >= 0.0
        {
            latency_fit_count_with_uncertainty += 1;
            uncertainty_s_squared += uncertainty_s * uncertainty_s;
        }
        if let Some(relative_pct) = application.relative_uncertainty_pct
            && relative_pct.is_finite()
            && relative_pct >= 0.0
        {
            max_relative_uncertainty_pct = Some(
                max_relative_uncertainty_pct.map_or(relative_pct, |max| max.max(relative_pct)),
            );
        }
        if application.confidence_score.is_finite() {
            min_confidence_score = Some(
                min_confidence_score
                    .unwrap_or(application.confidence_score)
                    .min(application.confidence_score),
            );
        }
        if application.max_extrapolation_ratio.is_finite() {
            max_extrapolation_ratio = Some(
                max_extrapolation_ratio
                    .unwrap_or(application.max_extrapolation_ratio)
                    .max(application.max_extrapolation_ratio),
            );
        }
        if applicability_rank(&application.applicability_status)
            > applicability_rank(&applicability_status)
        {
            applicability_status = application.applicability_status.clone();
        }
    }

    let absolute_uncertainty_s = if latency_fit_count_with_uncertainty > 0 {
        Some(uncertainty_s_squared.sqrt())
    } else {
        None
    };
    let latency_relative_uncertainty_pct = absolute_uncertainty_s
        .filter(|_| predicted_s_total.is_finite() && predicted_s_total > 0.0)
        .map(|uncertainty_s| uncertainty_s / predicted_s_total * 100.0);
    let relative_uncertainty_pct = max_optional_f64(
        latency_relative_uncertainty_pct,
        max_relative_uncertainty_pct,
    );

    CalibrationUncertaintySummary {
        fit_count,
        fit_count_with_uncertainty,
        relative_uncertainty_pct,
        absolute_uncertainty_s,
        min_confidence_score,
        max_extrapolation_ratio,
        applicability_status,
    }
}

pub(super) fn rank_sensitivity_approximation_summary(
    approximations: &[SimulationApproximation],
    policy_violations: &[ApproximationPolicyViolation],
) -> RankSensitivityApproximationSummary {
    let mut code_counts = BTreeMap::<String, usize>::new();
    let mut coarse_topology = false;
    let mut approximate_queueing = false;
    let mut aggregate_memory = false;
    let mut uncalibrated_runtime = false;
    let mut unsupported_runtime = false;

    for approximation in approximations {
        *code_counts.entry(approximation.code.clone()).or_default() += 1;
        let code = approximation.code.as_str();
        let category = approximation.category.as_str();

        coarse_topology |= category == "topology" || code.contains("topology");
        approximate_queueing |= matches!(category, "queueing" | "admission" | "routing")
            || code.contains("queue")
            || code.contains("contention")
            || code.ends_with("_not_modeled")
            || code == "approximate_serving_event_loop";
        aggregate_memory |= category == "memory"
            || code.contains("memory")
            || code.contains("kv_residency")
            || code == "static_per_gpu_memory_estimate";
        uncalibrated_runtime |= code.contains("uncalibrated")
            || code == "serving_stack_unspecified"
            || code == "calibration_profile_serving_stack_mismatch"
            || code == "calibration_profile_hardware_mismatch"
            || code == "calibration_profile_fabric_mismatch"
            || code == "calibration_profile_model_mismatch"
            || code == "calibration_profile_dtype_mismatch";
        unsupported_runtime |= category == "runtime"
            || code.contains("runtime_feature")
            || code.contains("not_modeled")
            || code.contains("unsupported");
    }

    let status = if !policy_violations.is_empty() {
        "policy_rejected"
    } else if approximations.is_empty() {
        "no_approximations"
    } else if uncalibrated_runtime {
        "calibration_risk"
    } else if coarse_topology || approximate_queueing || aggregate_memory || unsupported_runtime {
        "model_approximation"
    } else {
        "approximate"
    }
    .to_string();

    RankSensitivityApproximationSummary {
        status,
        approximation_count: approximations.len(),
        policy_violation_count: policy_violations.len(),
        coarse_topology,
        approximate_queueing,
        aggregate_memory,
        uncalibrated_runtime,
        unsupported_runtime,
        top_codes: approximation_top_codes(code_counts, 5),
    }
}

fn approximation_top_codes(counts: BTreeMap<String, usize>, limit: usize) -> String {
    let mut entries = counts.into_iter().collect::<Vec<(String, usize)>>();
    entries.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    entries
        .into_iter()
        .take(limit)
        .map(|(code, count)| format!("{code}:{count}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn application_has_numeric_uncertainty(application: &CalibrationFitApplication) -> bool {
    finite_nonnegative(application.relative_uncertainty_pct)
        || finite_nonnegative(application.absolute_uncertainty_s)
        || finite_nonnegative(application.absolute_uncertainty_value)
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

pub(super) fn calibration_phase_uncertainty_summary<'a>(
    applications: impl Iterator<Item = &'a CalibrationFitApplication>,
    phase: &str,
) -> CalibrationUncertaintySummary {
    let normalized_phase = phase.to_ascii_lowercase();
    calibration_uncertainty_summary(
        applications
            .filter(move |application| application.phase.to_ascii_lowercase() == normalized_phase),
    )
}

fn applicability_rank(status: &str) -> u8 {
    match status {
        "extrapolated" => 4,
        "unbounded" => 3,
        "partially_bounded" => 2,
        "interpolated" => 1,
        "none" => 0,
        _ => 0,
    }
}

fn write_calibration_profile<W: Write>(
    writer: &mut W,
    profile: Option<&CalibrationProfileMetadata>,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(profile) = profile else {
        writeln!(
            writer,
            "{indent}  \"profile\": null{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    writeln!(writer, "{indent}  \"profile\": {{")?;
    writeln!(
        writer,
        "{indent}    \"path\": {},",
        json_string(&profile.path)
    )?;
    write_optional_json_string(writer, indent, "name", profile.name.as_deref(), true)?;
    write_optional_json_string(
        writer,
        indent,
        "hardware",
        profile.hardware.as_deref(),
        true,
    )?;
    write_optional_json_string(writer, indent, "fabric", profile.fabric.as_deref(), true)?;
    write_optional_json_string(writer, indent, "model", profile.model.as_deref(), true)?;
    write_optional_json_string(writer, indent, "dtype", profile.dtype.as_deref(), true)?;
    write_optional_json_string(
        writer,
        indent,
        "serving_stack",
        profile.serving_stack.as_deref(),
        true,
    )?;
    write_string_vec(
        writer,
        indent,
        "serving_runtime_features",
        &profile.serving_runtime_features,
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "backend_version",
        profile.backend_version.as_deref(),
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "driver_version",
        profile.driver_version.as_deref(),
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "cuda_version",
        profile.cuda_version.as_deref(),
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "rocm_version",
        profile.rocm_version.as_deref(),
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "nccl_version",
        profile.nccl_version.as_deref(),
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "rccl_version",
        profile.rccl_version.as_deref(),
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "ucx_version",
        profile.ucx_version.as_deref(),
        true,
    )?;
    write_string_vec(
        writer,
        indent,
        "kernel_settings",
        &profile.kernel_settings,
        true,
    )?;
    write_optional_json_string(
        writer,
        indent,
        "environment_hash",
        profile.environment_hash.as_deref(),
        true,
    )?;
    write_optional_json_string(writer, indent, "source", profile.source.as_deref(), true)?;
    write_optional_json_string(writer, indent, "date", profile.date.as_deref(), true)?;
    write_optional_json_string(writer, indent, "notes", profile.notes.as_deref(), true)?;
    write_calibration_valid_shape(writer, indent, profile.valid_shape.as_ref(), true)?;
    write_calibration_invalid_shapes(writer, indent, &profile.invalid_shapes, true)?;
    write_calibration_fits(writer, indent, &profile.fits, true)?;
    write_calibration_benchmark_summary(writer, indent, &profile.benchmarks, true)?;
    write_calibration_benchmarks(writer, indent, &profile.benchmarks, false)?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_calibration_applicability_warnings<W: Write>(
    writer: &mut W,
    warnings: &[CalibrationApplicabilityWarning],
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"applicability_warnings\": [")?;
    for (idx, warning) in warnings.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"field\": {},",
            json_string(&warning.field)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed_min\": {},",
            warning.observed_min
        )?;
        writeln!(
            writer,
            "{indent}      \"observed_max\": {},",
            warning.observed_max
        )?;
        writeln!(
            writer,
            "{indent}      \"calibrated_min\": {},",
            json_optional_u32(warning.calibrated_min)
        )?;
        writeln!(
            writer,
            "{indent}      \"calibrated_max\": {},",
            json_optional_u32(warning.calibrated_max)
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {}",
            json_string(&warning.message)
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < warnings.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_calibration_valid_shape<W: Write>(
    writer: &mut W,
    indent: &str,
    valid_shape: Option<&CalibrationShapeRange>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(valid_shape) = valid_shape else {
        writeln!(
            writer,
            "{indent}    \"valid_shape\": null{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    writeln!(writer, "{indent}    \"valid_shape\": {{")?;
    let field_indent = format!("{indent}      ");
    write_calibration_shape_range_fields(writer, &field_indent, valid_shape)?;
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_calibration_shape_range_fields<W: Write>(
    writer: &mut W,
    field_indent: &str,
    shape: &CalibrationShapeRange,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{field_indent}\"min_batch_size\": {},",
        json_optional_u32(shape.min_batch_size)
    )?;
    writeln!(
        writer,
        "{field_indent}\"max_batch_size\": {},",
        json_optional_u32(shape.max_batch_size)
    )?;
    writeln!(
        writer,
        "{field_indent}\"min_prompt_tokens\": {},",
        json_optional_u32(shape.min_prompt_tokens)
    )?;
    writeln!(
        writer,
        "{field_indent}\"max_prompt_tokens\": {},",
        json_optional_u32(shape.max_prompt_tokens)
    )?;
    writeln!(
        writer,
        "{field_indent}\"min_decode_tokens\": {},",
        json_optional_u32(shape.min_decode_tokens)
    )?;
    writeln!(
        writer,
        "{field_indent}\"max_decode_tokens\": {},",
        json_optional_u32(shape.max_decode_tokens)
    )?;
    writeln!(
        writer,
        "{field_indent}\"min_sequence_tokens\": {},",
        json_optional_u32(shape.min_sequence_tokens)
    )?;
    writeln!(
        writer,
        "{field_indent}\"max_sequence_tokens\": {}",
        json_optional_u32(shape.max_sequence_tokens)
    )?;
    Ok(())
}

fn write_calibration_invalid_shapes<W: Write>(
    writer: &mut W,
    indent: &str,
    invalid_shapes: &[CalibrationInvalidShapeRange],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"invalid_shapes\": [")?;
    for (idx, invalid_shape) in invalid_shapes.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"name\": {},",
            json_optional_string(invalid_shape.name.as_deref())
        )?;
        writeln!(
            writer,
            "{indent}        \"reason\": {},",
            json_optional_string(invalid_shape.reason.as_deref())
        )?;
        writeln!(writer, "{indent}        \"shape\": {{")?;
        let field_indent = format!("{indent}          ");
        write_calibration_shape_range_fields(writer, &field_indent, &invalid_shape.shape)?;
        writeln!(writer, "{indent}        }}")?;
        writeln!(
            writer,
            "{indent}      }}{}",
            comma(idx + 1 < invalid_shapes.len())
        )?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_calibration_fits<W: Write>(
    writer: &mut W,
    indent: &str,
    fits: &[CalibrationFittedModel],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"fits\": [")?;
    for (idx, fit) in fits.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        write_fit_string(writer, indent, "name", fit.name.as_deref(), true)?;
        writeln!(
            writer,
            "{indent}        \"target\": {},",
            json_string(&fit.target)
        )?;
        write_fit_string(writer, indent, "phase", fit.phase.as_deref(), true)?;
        write_fit_string(writer, indent, "kind", fit.kind.as_deref(), true)?;
        writeln!(
            writer,
            "{indent}        \"model\": {},",
            json_string(&fit.model)
        )?;
        write_fit_string(writer, indent, "unit", fit.unit.as_deref(), true)?;
        write_fit_f64(writer, indent, "intercept", fit.intercept, true)?;
        write_fit_string_array(writer, indent, "features", &fit.features, true)?;
        write_fit_f64_array(writer, indent, "coefficients", &fit.coefficients, true)?;
        write_fit_feature_ranges(writer, indent, &fit.feature_ranges, true)?;
        write_fit_f64(writer, indent, "r_squared", fit.r_squared, true)?;
        write_fit_f64(
            writer,
            indent,
            "adjusted_r_squared",
            fit.adjusted_r_squared,
            true,
        )?;
        write_fit_f64(writer, indent, "rmse", fit.rmse, true)?;
        write_fit_f64(writer, indent, "rmse_pct", fit.rmse_pct, true)?;
        write_fit_f64(
            writer,
            indent,
            "mean_abs_pct_error",
            fit.mean_abs_pct_error,
            true,
        )?;
        write_fit_f64(
            writer,
            indent,
            "max_abs_pct_error",
            fit.max_abs_pct_error,
            true,
        )?;
        write_fit_f64(writer, indent, "validation_rmse", fit.validation_rmse, true)?;
        write_fit_f64(
            writer,
            indent,
            "validation_rmse_pct",
            fit.validation_rmse_pct,
            true,
        )?;
        write_fit_f64(
            writer,
            indent,
            "validation_mean_abs_pct_error",
            fit.validation_mean_abs_pct_error,
            true,
        )?;
        write_fit_f64(
            writer,
            indent,
            "validation_max_abs_pct_error",
            fit.validation_max_abs_pct_error,
            true,
        )?;
        write_fit_f64(
            writer,
            indent,
            "confidence_interval",
            fit.confidence_interval,
            true,
        )?;
        write_fit_f64(
            writer,
            indent,
            "confidence_interval_pct",
            fit.confidence_interval_pct,
            true,
        )?;
        write_fit_f64(
            writer,
            indent,
            "confidence_level",
            fit.confidence_level,
            true,
        )?;
        write_fit_u32(writer, indent, "sample_count", fit.sample_count, true)?;
        write_fit_u32(
            writer,
            indent,
            "validation_sample_count",
            fit.validation_sample_count,
            true,
        )?;
        write_fit_string(writer, indent, "source", fit.source.as_deref(), true)?;
        write_fit_string(writer, indent, "notes", fit.notes.as_deref(), false)?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < fits.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_fit_feature_ranges<W: Write>(
    writer: &mut W,
    indent: &str,
    ranges: &[CalibrationFitFeatureRange],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}        \"feature_ranges\": [")?;
    for (idx, range) in ranges.iter().enumerate() {
        writeln!(writer, "{indent}          {{")?;
        writeln!(
            writer,
            "{indent}            \"feature\": {},",
            json_string(&range.feature)
        )?;
        writeln!(
            writer,
            "{indent}            \"min\": {},",
            json_optional_value(range.min)
        )?;
        writeln!(
            writer,
            "{indent}            \"max\": {}",
            json_optional_value(range.max)
        )?;
        writeln!(
            writer,
            "{indent}          }}{}",
            comma(idx + 1 < ranges.len())
        )?;
    }
    writeln!(writer, "{indent}        ]{}", comma(trailing_comma))
}

pub(super) fn write_calibration_fit_applications<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    applications: &[CalibrationFitApplication],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, application) in applications.iter().enumerate() {
        write_calibration_fit_application_object(
            writer,
            &format!("{indent}    "),
            application,
            idx + 1 < applications.len(),
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_optional_calibration_fit_application<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    application: Option<&CalibrationFitApplication>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let field_indent = format!("{indent}      ");
    if let Some(application) = application {
        writeln!(writer, "{field_indent}\"{field}\": {{")?;
        write_calibration_fit_application_fields(writer, &field_indent, application)?;
        writeln!(writer, "{field_indent}}}{}", comma(trailing_comma))
    } else {
        writeln!(
            writer,
            "{field_indent}\"{field}\": null{}",
            comma(trailing_comma)
        )
    }
}

fn write_calibration_fit_application_object<W: Write>(
    writer: &mut W,
    indent: &str,
    application: &CalibrationFitApplication,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}{{")?;
    write_calibration_fit_application_fields(writer, indent, application)?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}

fn write_calibration_fit_application_fields<W: Write>(
    writer: &mut W,
    indent: &str,
    application: &CalibrationFitApplication,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}  \"phase\": {},",
        json_string(&application.phase)
    )?;
    writeln!(
        writer,
        "{indent}  \"target\": {},",
        json_string(&application.target)
    )?;
    writeln!(
        writer,
        "{indent}  \"fit_name\": {},",
        json_optional_string(application.fit_name.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}  \"model\": {},",
        json_string(&application.model)
    )?;
    writeln!(
        writer,
        "{indent}  \"unit\": {},",
        json_optional_string(application.unit.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}  \"intercept\": {},",
        json_f64(application.intercept)
    )?;
    writeln!(
        writer,
        "{indent}  \"raw_prediction\": {},",
        json_f64(application.raw_prediction)
    )?;
    writeln!(
        writer,
        "{indent}  \"prediction_kind\": {},",
        json_string(&application.prediction_kind)
    )?;
    writeln!(
        writer,
        "{indent}  \"predicted_value\": {},",
        json_f64(application.predicted_value)
    )?;
    writeln!(
        writer,
        "{indent}  \"prediction_unit\": {},",
        json_optional_string(application.prediction_unit.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}  \"predicted_ms\": {},",
        if application.prediction_kind == "latency" {
            json_ms(application.predicted_s)
        } else {
            "null".to_string()
        }
    )?;
    writeln!(
        writer,
        "{indent}  \"baseline_value\": {},",
        application
            .baseline_value
            .map(json_f64)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}  \"baseline_ms\": {},",
        if application.prediction_kind == "latency" {
            application
                .baseline_s
                .map(json_ms)
                .unwrap_or_else(|| "null".to_string())
        } else {
            "null".to_string()
        }
    )?;
    writeln!(
        writer,
        "{indent}  \"applicability_status\": {},",
        json_string(&application.applicability_status)
    )?;
    writeln!(
        writer,
        "{indent}  \"confidence_score\": {},",
        json_f64(application.confidence_score)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_extrapolation_ratio\": {},",
        json_f64(application.max_extrapolation_ratio)
    )?;
    writeln!(
        writer,
        "{indent}  \"relative_uncertainty_pct\": {},",
        json_optional_value(application.relative_uncertainty_pct)
    )?;
    writeln!(
        writer,
        "{indent}  \"absolute_uncertainty_ms\": {},",
        if application.prediction_kind == "latency" {
            json_optional_seconds_ms(application.absolute_uncertainty_s)
        } else {
            "null".to_string()
        }
    )?;
    writeln!(
        writer,
        "{indent}  \"absolute_uncertainty_value\": {},",
        json_optional_value(application.absolute_uncertainty_value)
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_source\": {},",
        json_optional_string(application.uncertainty_source.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}  \"validation_rmse\": {},",
        json_optional_value(application.validation_rmse)
    )?;
    writeln!(
        writer,
        "{indent}  \"validation_rmse_pct\": {},",
        json_optional_value(application.validation_rmse_pct)
    )?;
    writeln!(
        writer,
        "{indent}  \"validation_mean_abs_pct_error\": {},",
        json_optional_value(application.validation_mean_abs_pct_error)
    )?;
    writeln!(
        writer,
        "{indent}  \"validation_max_abs_pct_error\": {},",
        json_optional_value(application.validation_max_abs_pct_error)
    )?;
    writeln!(
        writer,
        "{indent}  \"confidence_interval\": {},",
        json_optional_value(application.confidence_interval)
    )?;
    writeln!(
        writer,
        "{indent}  \"confidence_interval_pct\": {},",
        json_optional_value(application.confidence_interval_pct)
    )?;
    writeln!(
        writer,
        "{indent}  \"confidence_level\": {},",
        json_optional_value(application.confidence_level)
    )?;
    writeln!(
        writer,
        "{indent}  \"sample_count\": {},",
        json_optional_u32(application.sample_count)
    )?;
    writeln!(
        writer,
        "{indent}  \"validation_sample_count\": {},",
        json_optional_u32(application.validation_sample_count)
    )?;
    writeln!(
        writer,
        "{indent}  \"source\": {},",
        json_optional_string(application.source.as_deref())
    )?;
    write_calibration_fit_feature_values(writer, indent, &application.features)
}

fn write_calibration_fit_feature_values<W: Write>(
    writer: &mut W,
    indent: &str,
    features: &[CalibrationFitFeatureValue],
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"features\": [")?;
    for (idx, feature) in features.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&feature.name)
        )?;
        writeln!(
            writer,
            "{indent}      \"value\": {},",
            json_f64(feature.value)
        )?;
        writeln!(
            writer,
            "{indent}      \"coefficient\": {},",
            json_f64(feature.coefficient)
        )?;
        writeln!(
            writer,
            "{indent}      \"range_min\": {},",
            json_optional_value(feature.range_min)
        )?;
        writeln!(
            writer,
            "{indent}      \"range_max\": {},",
            json_optional_value(feature.range_max)
        )?;
        writeln!(
            writer,
            "{indent}      \"status\": {},",
            json_string(&feature.status)
        )?;
        writeln!(
            writer,
            "{indent}      \"extrapolation_ratio\": {}",
            json_f64(feature.extrapolation_ratio)
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < features.len()))?;
    }
    writeln!(writer, "{indent}  ]")
}

fn write_fit_string<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<&str>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}        \"{field}\": {}{}",
        json_optional_string(value),
        comma(trailing_comma)
    )
}

fn write_fit_u32<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<u32>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}        \"{field}\": {}{}",
        json_optional_u32(value),
        comma(trailing_comma)
    )
}

fn write_fit_f64<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<f64>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}        \"{field}\": {}{}",
        json_optional_value(value),
        comma(trailing_comma)
    )
}

fn write_fit_string_array<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    values: &[String],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}        \"{field}\": [{}]{}",
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(", "),
        comma(trailing_comma)
    )
}

fn write_fit_f64_array<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    values: &[f64],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}        \"{field}\": [{}]{}",
        values
            .iter()
            .map(|value| json_f64(*value))
            .collect::<Vec<_>>()
            .join(", "),
        comma(trailing_comma)
    )
}

fn write_calibration_invalid_shape_warnings<W: Write>(
    writer: &mut W,
    warnings: &[CalibrationInvalidShapeWarning],
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"invalid_shape_warnings\": [")?;
    for (idx, warning) in warnings.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_optional_string(warning.name.as_deref())
        )?;
        writeln!(
            writer,
            "{indent}      \"reason\": {},",
            json_optional_string(warning.reason.as_deref())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&warning.message)
        )?;
        writeln!(writer, "{indent}      \"shape\": {{")?;
        let field_indent = format!("{indent}        ");
        write_calibration_shape_range_fields(writer, &field_indent, &warning.shape)?;
        writeln!(writer, "{indent}      }}")?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < warnings.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

#[derive(Clone, Debug, PartialEq)]
struct CalibrationBenchmarkSummary {
    benchmark_count: usize,
    latency_comparison_count: usize,
    mean_abs_pct_error: Option<f64>,
    max_abs_pct_error: Option<f64>,
    rmse_pct_error: Option<f64>,
    mean_signed_pct_error: Option<f64>,
    within_10_percent_count: usize,
    within_20_percent_count: usize,
    worst_benchmark: Option<String>,
    status: &'static str,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct CalibrationBenchmarkLatencyResidual {
    pub(super) error_ms: f64,
    pub(super) abs_error_ms: f64,
    pub(super) signed_pct_error: f64,
    pub(super) abs_pct_error: f64,
}

fn calibration_benchmark_summary(
    benchmarks: &[CalibrationBenchmarkPoint],
) -> CalibrationBenchmarkSummary {
    let mut abs_errors = Vec::new();
    let mut signed_errors = Vec::new();
    let mut within_10_percent_count = 0_usize;
    let mut within_20_percent_count = 0_usize;
    let mut worst_benchmark = None;
    let mut worst_abs_error = f64::NEG_INFINITY;

    for (idx, benchmark) in benchmarks.iter().enumerate() {
        let Some(residual) = calibration_benchmark_latency_residual(benchmark) else {
            continue;
        };
        let signed_error = residual.signed_pct_error;
        let abs_error = residual.abs_pct_error;
        if abs_error <= 10.0 {
            within_10_percent_count += 1;
        }
        if abs_error <= 20.0 {
            within_20_percent_count += 1;
        }
        if abs_error > worst_abs_error {
            worst_abs_error = abs_error;
            worst_benchmark = Some(benchmark_label(idx, benchmark));
        }
        signed_errors.push(signed_error);
        abs_errors.push(abs_error);
    }

    let latency_comparison_count = abs_errors.len();
    let mean_abs_pct_error = optional_mean(&abs_errors);
    let max_abs_pct_error = if abs_errors.is_empty() {
        None
    } else {
        Some(max_value(&abs_errors))
    };
    let rmse_pct_error = if signed_errors.is_empty() {
        None
    } else {
        Some(
            (signed_errors.iter().map(|error| error * error).sum::<f64>()
                / signed_errors.len() as f64)
                .sqrt(),
        )
    };
    let mean_signed_pct_error = optional_mean(&signed_errors);
    let status = calibration_benchmark_summary_status(
        benchmarks.len(),
        latency_comparison_count,
        mean_abs_pct_error,
    );

    CalibrationBenchmarkSummary {
        benchmark_count: benchmarks.len(),
        latency_comparison_count,
        mean_abs_pct_error,
        max_abs_pct_error,
        rmse_pct_error,
        mean_signed_pct_error,
        within_10_percent_count,
        within_20_percent_count,
        worst_benchmark,
        status,
    }
}

fn calibration_benchmark_summary_status(
    benchmark_count: usize,
    latency_comparison_count: usize,
    mean_abs_pct_error: Option<f64>,
) -> &'static str {
    if benchmark_count == 0 {
        return "no_benchmarks";
    }
    if latency_comparison_count == 0 {
        return "no_latency_pairs";
    }
    match mean_abs_pct_error {
        Some(error) if error <= 10.0 => "good",
        Some(error) if error <= 25.0 => "watch",
        Some(_) => "poor",
        None => "no_latency_pairs",
    }
}

pub(super) fn calibration_benchmark_latency_residual(
    benchmark: &CalibrationBenchmarkPoint,
) -> Option<CalibrationBenchmarkLatencyResidual> {
    let (Some(measured_ms), Some(predicted_ms)) = (benchmark.measured_ms, benchmark.predicted_ms)
    else {
        return None;
    };
    if !measured_ms.is_finite()
        || !predicted_ms.is_finite()
        || measured_ms <= 0.0
        || predicted_ms <= 0.0
    {
        return None;
    }
    let error_ms = predicted_ms - measured_ms;
    let signed_pct_error = (error_ms / measured_ms) * 100.0;
    Some(CalibrationBenchmarkLatencyResidual {
        error_ms,
        abs_error_ms: error_ms.abs(),
        signed_pct_error,
        abs_pct_error: signed_pct_error.abs(),
    })
}

pub(super) fn calibration_benchmark_latency_residual_status(
    residual: Option<CalibrationBenchmarkLatencyResidual>,
) -> &'static str {
    match residual.map(|residual| residual.abs_pct_error) {
        Some(error) if error <= 10.0 => "good",
        Some(error) if error <= 25.0 => "watch",
        Some(_) => "poor",
        None => "no_latency_pair",
    }
}

fn benchmark_label(idx: usize, benchmark: &CalibrationBenchmarkPoint) -> String {
    benchmark.name.clone().unwrap_or_else(|| {
        let kind = benchmark.kind.as_deref().unwrap_or("unknown");
        let phase = benchmark.phase.as_deref().unwrap_or("unknown");
        format!("benchmark[{idx}] {kind}/{phase}")
    })
}

fn optional_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn max_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(f64::INFINITY)
}

fn write_calibration_benchmark_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    benchmarks: &[CalibrationBenchmarkPoint],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let summary = calibration_benchmark_summary(benchmarks);
    writeln!(writer, "{indent}    \"benchmark_summary\": {{")?;
    writeln!(
        writer,
        "{indent}      \"benchmark_count\": {},",
        summary.benchmark_count
    )?;
    writeln!(
        writer,
        "{indent}      \"latency_comparison_count\": {},",
        summary.latency_comparison_count
    )?;
    writeln!(
        writer,
        "{indent}      \"mean_abs_pct_error\": {},",
        json_optional_value(summary.mean_abs_pct_error)
    )?;
    writeln!(
        writer,
        "{indent}      \"max_abs_pct_error\": {},",
        json_optional_value(summary.max_abs_pct_error)
    )?;
    writeln!(
        writer,
        "{indent}      \"rmse_pct_error\": {},",
        json_optional_value(summary.rmse_pct_error)
    )?;
    writeln!(
        writer,
        "{indent}      \"mean_signed_pct_error\": {},",
        json_optional_value(summary.mean_signed_pct_error)
    )?;
    writeln!(
        writer,
        "{indent}      \"within_10_percent_count\": {},",
        summary.within_10_percent_count
    )?;
    writeln!(
        writer,
        "{indent}      \"within_20_percent_count\": {},",
        summary.within_20_percent_count
    )?;
    writeln!(
        writer,
        "{indent}      \"worst_benchmark\": {},",
        summary
            .worst_benchmark
            .as_deref()
            .map(json_string)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}      \"status\": {}",
        json_string(summary.status)
    )?;
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_calibration_benchmarks<W: Write>(
    writer: &mut W,
    indent: &str,
    benchmarks: &[CalibrationBenchmarkPoint],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"benchmarks\": [")?;
    for (idx, benchmark) in benchmarks.iter().enumerate() {
        let residual = calibration_benchmark_latency_residual(benchmark);
        writeln!(writer, "{indent}      {{")?;
        write_benchmark_string(writer, indent, "name", benchmark.name.as_deref(), true)?;
        write_benchmark_string(writer, indent, "kind", benchmark.kind.as_deref(), true)?;
        write_benchmark_string(writer, indent, "phase", benchmark.phase.as_deref(), true)?;
        write_benchmark_string(
            writer,
            indent,
            "hardware",
            benchmark.hardware.as_deref(),
            true,
        )?;
        write_benchmark_string(writer, indent, "fabric", benchmark.fabric.as_deref(), true)?;
        write_benchmark_string(writer, indent, "model", benchmark.model.as_deref(), true)?;
        write_benchmark_string(writer, indent, "dtype", benchmark.dtype.as_deref(), true)?;
        write_benchmark_u32(writer, indent, "batch_size", benchmark.batch_size, true)?;
        write_benchmark_u32(
            writer,
            indent,
            "prompt_tokens",
            benchmark.prompt_tokens,
            true,
        )?;
        write_benchmark_u32(
            writer,
            indent,
            "decode_tokens",
            benchmark.decode_tokens,
            true,
        )?;
        write_benchmark_u32(
            writer,
            indent,
            "sequence_tokens",
            benchmark.sequence_tokens,
            true,
        )?;
        write_benchmark_u32(writer, indent, "tensor_ranks", benchmark.tensor_ranks, true)?;
        write_benchmark_u32(
            writer,
            indent,
            "pipeline_ranks",
            benchmark.pipeline_ranks,
            true,
        )?;
        write_benchmark_u32(writer, indent, "expert_ranks", benchmark.expert_ranks, true)?;
        write_benchmark_u32(writer, indent, "data_ranks", benchmark.data_ranks, true)?;
        write_benchmark_f64(writer, indent, "measured_ms", benchmark.measured_ms, true)?;
        write_benchmark_f64(writer, indent, "predicted_ms", benchmark.predicted_ms, true)?;
        write_benchmark_f64(
            writer,
            indent,
            "latency_error_ms",
            residual.map(|residual| residual.error_ms),
            true,
        )?;
        write_benchmark_f64(
            writer,
            indent,
            "latency_abs_error_ms",
            residual.map(|residual| residual.abs_error_ms),
            true,
        )?;
        write_benchmark_f64(
            writer,
            indent,
            "latency_signed_pct_error",
            residual.map(|residual| residual.signed_pct_error),
            true,
        )?;
        write_benchmark_f64(
            writer,
            indent,
            "latency_abs_pct_error",
            residual.map(|residual| residual.abs_pct_error),
            true,
        )?;
        write_benchmark_string(
            writer,
            indent,
            "latency_residual_status",
            Some(calibration_benchmark_latency_residual_status(residual)),
            true,
        )?;
        write_benchmark_f64(
            writer,
            indent,
            "throughput_tokens_per_s",
            benchmark.throughput_tokens_per_s,
            true,
        )?;
        write_benchmark_string(
            writer,
            indent,
            "command",
            benchmark.command.as_deref(),
            true,
        )?;
        write_benchmark_string(writer, indent, "source", benchmark.source.as_deref(), true)?;
        write_benchmark_string(writer, indent, "notes", benchmark.notes.as_deref(), false)?;
        writeln!(
            writer,
            "{indent}      }}{}",
            comma(idx + 1 < benchmarks.len())
        )?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_benchmark_string<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<&str>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let value = value.map(json_string).unwrap_or_else(|| "null".to_string());
    writeln!(
        writer,
        "{indent}        \"{field}\": {value}{}",
        comma(trailing_comma)
    )
}

fn write_benchmark_u32<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<u32>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}        \"{field}\": {}{}",
        json_optional_u32(value),
        comma(trailing_comma)
    )
}

fn write_benchmark_f64<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<f64>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}        \"{field}\": {}{}",
        json_optional_value(value),
        comma(trailing_comma)
    )
}

pub(super) fn json_optional_uncertainty_ms(feasible: bool, seconds: Option<f64>) -> String {
    if feasible {
        json_optional_seconds_ms(seconds)
    } else {
        "null".to_string()
    }
}

pub(super) fn json_optional_seconds_ms(seconds: Option<f64>) -> String {
    seconds.map(json_ms).unwrap_or_else(|| "null".to_string())
}

pub(super) fn json_metric_uncertainty_ms(
    feasible: bool,
    value_s: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value_s.is_finite() {
        return "null".to_string();
    }
    json_optional_seconds_ms(metric_uncertainty_value(value_s, uncertainty))
}

pub(super) fn json_metric_lower_ms(
    feasible: bool,
    value_s: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value_s.is_finite() {
        return "null".to_string();
    }
    metric_sensitivity_bounds(value_s, uncertainty)
        .map(|(lower, _)| json_ms(lower))
        .unwrap_or_else(|| "null".to_string())
}

pub(super) fn json_metric_upper_ms(
    feasible: bool,
    value_s: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value_s.is_finite() {
        return "null".to_string();
    }
    metric_sensitivity_bounds(value_s, uncertainty)
        .map(|(_, upper)| json_ms(upper))
        .unwrap_or_else(|| "null".to_string())
}

pub(super) fn json_metric_relative_uncertainty_value(
    feasible: bool,
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value.is_finite() {
        return "null".to_string();
    }
    json_optional_value(
        uncertainty
            .relative_uncertainty_pct
            .filter(|pct| pct.is_finite() && *pct >= 0.0)
            .map(|pct| value.abs() * pct / 100.0),
    )
}

pub(super) fn json_metric_relative_lower_value(
    feasible: bool,
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value.is_finite() {
        return "null".to_string();
    }
    metric_relative_sensitivity_bounds(value, uncertainty)
        .map(|(lower, _)| json_f64(lower))
        .unwrap_or_else(|| "null".to_string())
}

pub(super) fn json_metric_relative_upper_value(
    feasible: bool,
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> String {
    if !feasible || !value.is_finite() {
        return "null".to_string();
    }
    metric_relative_sensitivity_bounds(value, uncertainty)
        .map(|(_, upper)| json_f64(upper))
        .unwrap_or_else(|| "null".to_string())
}

pub(super) fn metric_uncertainty_value(
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> Option<f64> {
    uncertainty.absolute_uncertainty_s.or_else(|| {
        uncertainty
            .relative_uncertainty_pct
            .filter(|pct| pct.is_finite() && *pct >= 0.0)
            .map(|pct| value.abs() * pct / 100.0)
    })
}

pub(super) fn metric_relative_uncertainty_value(
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> Option<f64> {
    uncertainty
        .relative_uncertainty_pct
        .filter(|pct| pct.is_finite() && *pct >= 0.0)
        .map(|pct| value.abs() * pct / 100.0)
}

pub(super) fn metric_sensitivity_bounds(
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> Option<(f64, f64)> {
    metric_uncertainty_value(value, uncertainty)
        .filter(|delta| delta.is_finite() && *delta >= 0.0)
        .map(|delta| ((value - delta).max(0.0), value + delta))
}

pub(super) fn metric_relative_sensitivity_bounds(
    value: f64,
    uncertainty: &CalibrationUncertaintySummary,
) -> Option<(f64, f64)> {
    metric_relative_uncertainty_value(value, uncertainty)
        .filter(|delta| delta.is_finite() && *delta >= 0.0)
        .map(|delta| ((value - delta).max(0.0), value + delta))
}
