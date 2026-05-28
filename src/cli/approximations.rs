use super::*;

pub(super) fn serving_approximation_summary_note(
    summary: &ServingApproximationSummary,
) -> Option<String> {
    if summary.approximation_count == 0 && summary.policy_violation_count == 0 {
        return None;
    }

    let mut parts = vec![format!(
        "approximation_summary={}:{}",
        summary.status, summary.approximation_count
    )];
    if summary.policy_violation_count > 0 {
        parts.push(format!("policy_rejects={}", summary.policy_violation_count));
    }
    if summary.uncalibrated_runtime {
        parts.push("uncalibrated_runtime".to_string());
    }
    if summary.uncalibrated_phase_count > 0 {
        parts.push(format!(
            "uncalibrated_phases={}",
            summary.uncalibrated_phase_count
        ));
    }
    if summary.uncalibrated_queue_component_count > 0 {
        parts.push(format!(
            "uncalibrated_queues={}",
            summary.uncalibrated_queue_component_count
        ));
    }
    if summary.extrapolated_fit_count > 0 {
        parts.push(format!(
            "extrapolated_fits={}",
            summary.extrapolated_fit_count
        ));
    }
    if summary.coarse_topology {
        parts.push("coarse_topology".to_string());
    }
    if summary.approximate_queueing {
        parts.push("approx_queueing".to_string());
    }

    Some(parts.join(","))
}

pub(super) fn write_approximations<W: Write>(
    writer: &mut W,
    indent: &str,
    approximations: &[SimulationApproximation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"approximations\": [")?;
    for (idx, approximation) in approximations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&approximation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&approximation.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(&approximation.scope)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&approximation.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&approximation.message)
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {}",
            approximation
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < approximations.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_serving_approximation_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &ServingApproximationSummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"approximation_summary\": {{")?;
    writeln!(
        writer,
        "{indent}    \"status\": {},",
        json_string(&summary.status)
    )?;
    writeln!(
        writer,
        "{indent}    \"approximation_count\": {},",
        summary.approximation_count
    )?;
    writeln!(
        writer,
        "{indent}    \"policy_violation_count\": {},",
        summary.policy_violation_count
    )?;
    writeln!(
        writer,
        "{indent}    \"calibration_count\": {},",
        summary.calibration_count
    )?;
    writeln!(
        writer,
        "{indent}    \"topology_count\": {},",
        summary.topology_count
    )?;
    writeln!(
        writer,
        "{indent}    \"queueing_count\": {},",
        summary.queueing_count
    )?;
    writeln!(
        writer,
        "{indent}    \"runtime_count\": {},",
        summary.runtime_count
    )?;
    writeln!(
        writer,
        "{indent}    \"memory_count\": {},",
        summary.memory_count
    )?;
    writeln!(
        writer,
        "{indent}    \"capacity_count\": {},",
        summary.capacity_count
    )?;
    writeln!(
        writer,
        "{indent}    \"routing_count\": {},",
        summary.routing_count
    )?;
    writeln!(
        writer,
        "{indent}    \"admission_count\": {},",
        summary.admission_count
    )?;
    writeln!(
        writer,
        "{indent}    \"uncalibrated_phase_count\": {},",
        summary.uncalibrated_phase_count
    )?;
    writeln!(
        writer,
        "{indent}    \"uncalibrated_queue_component_count\": {},",
        summary.uncalibrated_queue_component_count
    )?;
    writeln!(
        writer,
        "{indent}    \"extrapolated_fit_count\": {},",
        summary.extrapolated_fit_count
    )?;
    writeln!(
        writer,
        "{indent}    \"coarse_topology\": {},",
        summary.coarse_topology
    )?;
    writeln!(
        writer,
        "{indent}    \"approximate_queueing\": {},",
        summary.approximate_queueing
    )?;
    writeln!(
        writer,
        "{indent}    \"uncalibrated_runtime\": {},",
        summary.uncalibrated_runtime
    )?;
    write_serving_approximation_counts(
        writer,
        indent,
        "category_counts",
        &summary.category_counts,
        true,
    )?;
    write_serving_approximation_counts(writer, indent, "top_codes", &summary.top_codes, false)?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_serving_approximation_counts<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    counts: &[ServingApproximationCount],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"{field}\": [")?;
    for (idx, count) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"name\": {},",
            json_string(&count.name)
        )?;
        writeln!(writer, "{indent}        \"count\": {}", count.count)?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

pub(super) fn write_approximation_policy_violations<W: Write>(
    writer: &mut W,
    indent: &str,
    violations: &[ApproximationPolicyViolation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"approximation_policy_violations\": [")?;
    for (idx, violation) in violations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"metric\": {},",
            json_optional_string(violation.metric.as_deref())
        )?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&violation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&violation.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(&violation.scope)
        )?;
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
