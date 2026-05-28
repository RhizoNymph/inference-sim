use super::*;

pub(super) fn selected_occupancy_resources(
    utilization: &[ResourceUtilization],
    limit: Option<usize>,
) -> Vec<String> {
    let count = limit.unwrap_or(utilization.len()).min(utilization.len());
    utilization
        .iter()
        .take(count)
        .map(|resource| resource.resource.clone())
        .collect()
}

pub(super) fn write_resource_occupancy<W: Write>(
    writer: &mut W,
    indent: &str,
    occupancy: &[ResourceOccupancySeries],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let bucket_count = occupancy
        .iter()
        .map(|series| series.buckets.len())
        .max()
        .unwrap_or(0);
    writeln!(
        writer,
        "{indent}  \"resource_occupancy_bucket_count\": {bucket_count},"
    )?;
    writeln!(
        writer,
        "{indent}  \"resource_occupancy_resource_count\": {},",
        occupancy.len()
    )?;
    writeln!(writer, "{indent}  \"resource_occupancy\": [")?;
    for (series_idx, series) in occupancy.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&series.resource)
        )?;
        writeln!(writer, "{indent}      \"buckets\": [")?;
        for (bucket_idx, bucket) in series.buckets.iter().enumerate() {
            writeln!(writer, "{indent}        {{")?;
            writeln!(
                writer,
                "{indent}          \"bucket_idx\": {},",
                bucket.bucket_idx
            )?;
            writeln!(
                writer,
                "{indent}          \"start_ms\": {},",
                json_ms(bucket.start_s)
            )?;
            writeln!(
                writer,
                "{indent}          \"finish_ms\": {},",
                json_ms(bucket.finish_s)
            )?;
            writeln!(
                writer,
                "{indent}          \"busy_ms\": {},",
                json_ms(bucket.busy_s)
            )?;
            writeln!(
                writer,
                "{indent}          \"utilization\": {},",
                json_optional_f64(bucket.utilization)
            )?;
            writeln!(
                writer,
                "{indent}          \"operation_count\": {}",
                bucket.operation_count
            )?;
            if bucket_idx + 1 < series.buckets.len() {
                writeln!(writer, "{indent}        }},")?;
            } else {
                writeln!(writer, "{indent}        }}")?;
            }
        }
        writeln!(writer, "{indent}      ]")?;
        if series_idx + 1 < occupancy.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_critical_path<W: Write>(
    writer: &mut W,
    indent: &str,
    path: &CriticalPath,
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_steps = limit.map_or(path.steps.len(), |limit| limit.min(path.steps.len()));
    writeln!(
        writer,
        "{indent}  \"critical_path_ms\": {},",
        json_ms(path.total_s)
    )?;
    writeln!(
        writer,
        "{indent}  \"critical_path_step_count\": {},",
        path.steps.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"critical_path_truncated\": {},",
        displayed_steps < path.steps.len()
    )?;
    writeln!(writer, "{indent}  \"critical_path\": [")?;
    for (idx, step) in path.steps.iter().take(displayed_steps).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"operation_id\": {},",
            step.operation_id
        )?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&step.name)
        )?;
        write_string_vec(
            writer,
            &format!("{indent}    "),
            "resources",
            &step.resources,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "explicit_dependencies",
            &step.explicit_dependencies,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "resource_dependencies",
            &step.resource_dependencies,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_ms(step.start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_ms(step.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"duration_ms\": {},",
            json_ms(step.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"wait_ms\": {}",
            json_ms(step.wait_s)
        )?;
        if idx + 1 < displayed_steps {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_scheduled_operations<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    operations: &[ScheduledOperation],
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_operations = limit.map_or(operations.len(), |limit| limit.min(operations.len()));
    writeln!(
        writer,
        "{indent}  \"scheduled_operation_count\": {},",
        operations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"scheduled_operations_truncated\": {},",
        displayed_operations < operations.len()
    )?;
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, operation) in operations.iter().take(displayed_operations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(writer, "{indent}      \"id\": {},", operation.id)?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&operation.name)
        )?;
        write_string_vec(
            writer,
            &format!("{indent}    "),
            "resources",
            &operation.resources,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "explicit_dependencies",
            &operation.explicit_dependencies,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "resource_dependencies",
            &operation.resource_dependencies,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_optional_f64(operation.start_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_optional_f64(operation.finish_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"duration_ms\": {}",
            json_optional_f64((operation.finish_s - operation.start_s).max(0.0) * 1000.0)
        )?;
        if idx + 1 < displayed_operations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}
