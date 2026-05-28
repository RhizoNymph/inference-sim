use super::*;

pub(super) fn pool_search_summary_note(
    summary: Option<&ServingPoolSearchSummary>,
) -> Option<String> {
    let summary = summary?;
    let considered = summary.considered_candidate_count();
    if considered == 0 && summary.generated_candidate_count == 0 {
        return Some("pool_search=0 generated".to_string());
    }

    let mut note = format!(
        "pool_search={}/{} generated",
        summary.generated_candidate_count, considered
    );
    let rejected_overlap = summary.rejected_overlap_count();
    let rejected_mode = summary.rejected_mode_count();
    let rejected_spread = summary.rejected_domain_spread_count();
    let duplicates = summary.duplicate_candidate_count();
    let mut details = Vec::new();
    let mode_counts = [
        ("colocated", summary.generated_colocated_count()),
        ("partial", summary.generated_partially_disaggregated_count()),
        ("full", summary.generated_fully_disaggregated_count()),
    ]
    .into_iter()
    .filter(|(_, count)| *count > 0)
    .map(|(label, count)| format!("{label}:{count}"))
    .collect::<Vec<_>>();
    if !mode_counts.is_empty() {
        details.push(format!("modes={}", mode_counts.join("|")));
    }
    if rejected_overlap > 0 {
        details.push(format!("overlap={rejected_overlap}"));
    }
    if rejected_mode > 0 {
        details.push(format!("mode={rejected_mode}"));
    }
    if rejected_spread > 0 {
        details.push(format!("spread={rejected_spread}"));
    }
    if duplicates > 0 {
        details.push(format!("duplicate={duplicates}"));
    }
    if summary.truncated {
        details.push(format!("truncated_at={}", summary.max_candidates));
    }
    if !details.is_empty() {
        note.push_str(" (");
        note.push_str(&details.join(","));
        note.push(')');
    }
    Some(note)
}

pub(super) fn pool_topology_summary_note(summary: &ServingPoolTopologySummary) -> Option<String> {
    let has_metadata = !summary.prefill_racks.is_empty()
        || !summary.decode_racks.is_empty()
        || !summary.prefill_islands.is_empty()
        || !summary.decode_islands.is_empty()
        || !summary.prefill_failure_domains.is_empty()
        || !summary.decode_failure_domains.is_empty()
        || !summary.prefill_node_labels.is_empty()
        || !summary.decode_node_labels.is_empty();
    if !has_metadata && summary.shared_node_count == 0 {
        return None;
    }

    Some(format!(
        "pool_topology=pf[nodes={},dedicated={},racks={},islands={},fds={}] decode[nodes={},dedicated={},racks={},islands={},fds={}] shared={}",
        summary.prefill_node_count,
        summary.dedicated_prefill_node_count,
        summary.prefill_racks.len(),
        summary.prefill_islands.len(),
        summary.prefill_failure_domains.len(),
        summary.decode_node_count,
        summary.dedicated_decode_node_count,
        summary.decode_racks.len(),
        summary.decode_islands.len(),
        summary.decode_failure_domains.len(),
        summary.shared_node_count
    ))
}

pub(super) fn kv_route_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let resource = score.kv_route_resource_summary.first()?;
    let rail_note = if score.kv_route_topology_summary.single_rail_dependency {
        format!(
            " rails=1(single:{})",
            score
                .kv_route_topology_summary
                .single_rail_id
                .map(|rail| rail.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )
    } else if score.kv_route_topology_summary.rail_count > 0 {
        format!(" rails={}", score.kv_route_topology_summary.rail_count)
    } else {
        String::new()
    };
    Some(format!(
        "kv_route_top={}:{}{}",
        resource.kind, resource.label, rail_note
    ))
}

pub(super) fn write_kv_route_constraints_json<W: Write>(
    writer: &mut W,
    indent: &str,
    constraints: ServingKvRouteConstraints,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"kv_route_constraints\": {{")?;
    writeln!(
        writer,
        "{indent}    \"min_inter_node_rail_count\": {},",
        json_optional_u32(constraints.min_inter_node_rail_count)
    )?;
    writeln!(
        writer,
        "{indent}    \"require_inter_node_rail_metadata\": {},",
        constraints.require_inter_node_rail_metadata
    )?;
    writeln!(
        writer,
        "{indent}    \"require_gpudirect\": {}",
        constraints.require_gpudirect
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

pub(super) fn write_route_coverage<W: Write>(
    writer: &mut W,
    indent: &str,
    score: &ScoredServingConfig,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"route_coverage\": {{")?;
    writeln!(
        writer,
        "{indent}    \"candidate_count\": {},",
        score.route_coverage.candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"routable_candidate_count\": {},",
        score.route_coverage.routable_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"unroutable_candidate_count\": {},",
        score.route_coverage.unroutable_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"fraction\": {}",
        json_f64(score.route_coverage.fraction)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_pool_search_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: Option<&ServingPoolSearchSummary>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(summary) = summary else {
        return writeln!(
            writer,
            "{indent}  \"pool_search_summary\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}  \"pool_search_summary\": {{")?;
    writeln!(
        writer,
        "{indent}    \"max_candidates\": {},",
        summary.max_candidates
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_candidate_count\": {},",
        summary.generated_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_colocated_count\": {},",
        summary.generated_colocated_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_partially_disaggregated_count\": {},",
        summary.generated_partially_disaggregated_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_fully_disaggregated_count\": {},",
        summary.generated_fully_disaggregated_count()
    )?;
    writeln!(writer, "{indent}    \"truncated\": {},", summary.truncated)?;
    writeln!(
        writer,
        "{indent}    \"considered_candidate_count\": {},",
        summary.considered_candidate_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_overlap_count\": {},",
        summary.rejected_overlap_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_mode_count\": {},",
        summary.rejected_mode_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_domain_spread_count\": {},",
        summary.rejected_domain_spread_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"duplicate_candidate_count\": {},",
        summary.duplicate_candidate_count()
    )?;
    writeln!(writer, "{indent}    \"groups\": [")?;
    for (idx, group) in summary.groups.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"prefill_group\": {},",
            json_string(&group.prefill_group)
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_group\": {},",
            json_string(&group.decode_group)
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_group_node_count\": {},",
            group.prefill_group_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_node_filter_node_count\": {},",
            group.prefill_node_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_gpu_filter_node_count\": {},",
            group.prefill_gpu_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_group_node_count\": {},",
            group.decode_group_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_node_filter_node_count\": {},",
            group.decode_node_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_gpu_filter_node_count\": {},",
            group.decode_gpu_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_node_counts\": [{}],",
            u32_list(&group.prefill_node_counts)
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_node_counts\": [{}],",
            u32_list(&group.decode_node_counts)
        )?;
        writeln!(
            writer,
            "{indent}        \"considered_candidate_count\": {},",
            group.considered_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}        \"rejected_overlap_count\": {},",
            group.rejected_overlap_count
        )?;
        writeln!(
            writer,
            "{indent}        \"rejected_mode_count\": {},",
            group.rejected_mode_count
        )?;
        writeln!(
            writer,
            "{indent}        \"rejected_domain_spread_count\": {},",
            group.rejected_domain_spread_count
        )?;
        writeln!(
            writer,
            "{indent}        \"duplicate_candidate_count\": {},",
            group.duplicate_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_candidate_count\": {},",
            group.generated_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_colocated_count\": {},",
            group.generated_colocated_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_partially_disaggregated_count\": {},",
            group.generated_partially_disaggregated_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_fully_disaggregated_count\": {}",
            group.generated_fully_disaggregated_count
        )?;
        if idx + 1 < summary.groups.len() {
            writeln!(writer, "{indent}      }},")?;
        } else {
            writeln!(writer, "{indent}      }}")?;
        }
    }
    writeln!(writer, "{indent}    ]")?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_pool_topology_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &ServingPoolTopologySummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"pool_topology\": {{")?;
    writeln!(
        writer,
        "{indent}    \"prefill_node_count\": {},",
        summary.prefill_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_node_count\": {},",
        summary.decode_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"shared_node_count\": {},",
        summary.shared_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"dedicated_prefill_node_count\": {},",
        summary.dedicated_prefill_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"dedicated_decode_node_count\": {},",
        summary.dedicated_decode_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_racks\": {},",
        json_string_array(&summary.prefill_racks)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_racks\": {},",
        json_string_array(&summary.decode_racks)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_islands\": {},",
        json_string_array(&summary.prefill_islands)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_islands\": {},",
        json_string_array(&summary.decode_islands)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_failure_domains\": {},",
        json_string_array(&summary.prefill_failure_domains)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_failure_domains\": {},",
        json_string_array(&summary.decode_failure_domains)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_node_labels\": {},",
        json_string_array(&summary.prefill_node_labels)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_node_labels\": {}",
        json_string_array(&summary.decode_node_labels)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_kv_route_resource_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summaries: &[ServingKvRouteResourceSummary],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"kv_route_resource_summary\": [")?;
    for (idx, summary) in summaries.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"resource_id\": {},",
            json_string(&summary.resource_id)
        )?;
        writeln!(
            writer,
            "{indent}      \"kind\": {},",
            json_string(&summary.kind)
        )?;
        writeln!(
            writer,
            "{indent}      \"label\": {},",
            json_string(&summary.label)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            summary.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"path_observations\": {},",
            summary.path_observations
        )?;
        writeln!(
            writer,
            "{indent}      \"transfer_bytes\": {},",
            summary.transfer_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"estimated_transfer_ms\": {},",
            json_ms(summary.estimated_transfer_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"min_bandwidth_gbps\": {},",
            json_optional_f64(summary.min_bandwidth_gbps)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_latency_ms\": {},",
            json_ms(summary.max_latency_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"rail_id\": {},",
            json_optional_u32(summary.rail_id)
        )?;
        write_kv_transfer_path_endpoint(writer, indent, "from", summary.from.as_ref(), true)?;
        write_kv_transfer_path_endpoint(writer, indent, "to", summary.to.as_ref(), false)?;
        if idx + 1 < summaries.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_kv_route_topology_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &ServingKvRouteTopologySummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"kv_route_topology_summary\": {{")?;
    writeln!(
        writer,
        "{indent}    \"route_resource_count\": {},",
        summary.route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"inter_node_route_resource_count\": {},",
        summary.inter_node_route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"gpu_nic_route_resource_count\": {},",
        summary.gpu_nic_route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"intra_node_route_resource_count\": {},",
        summary.intra_node_route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"rail_count\": {},",
        summary.rail_count
    )?;
    writeln!(
        writer,
        "{indent}    \"rail_ids\": {},",
        json_u32_array(&summary.rail_ids)
    )?;
    writeln!(
        writer,
        "{indent}    \"single_rail_dependency\": {},",
        summary.single_rail_dependency
    )?;
    writeln!(
        writer,
        "{indent}    \"single_rail_id\": {},",
        json_optional_u32(summary.single_rail_id)
    )?;
    writeln!(
        writer,
        "{indent}    \"unrailed_inter_node_route_resource_count\": {}",
        summary.unrailed_inter_node_route_resource_count
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

pub(super) fn write_topology_bottlenecks<W: Write>(
    writer: &mut W,
    indent: &str,
    bottlenecks: &[ServingTopologyBottleneckObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"topology_bottlenecks\": [")?;
    for (idx, bottleneck) in bottlenecks.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&bottleneck.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&bottleneck.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&bottleneck.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&bottleneck.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"severity\": {},",
            json_string(&bottleneck.severity)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(bottleneck.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(bottleneck.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {},",
            bottleneck
                .unit
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&bottleneck.message)
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {}",
            bottleneck
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < bottlenecks.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}
