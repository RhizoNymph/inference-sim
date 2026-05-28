use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ServingPoolSearchResult {
    pub(super) candidates: Vec<ServingPoolCandidate>,
    pub(super) summary: ServingPoolSearchSummary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ServingPoolSearchError {
    pub(super) reason: String,
    pub(super) summary: ServingPoolSearchSummary,
}

pub(super) fn generate_pool_search_candidates_with_summary(
    cluster: &Cluster,
    search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
) -> Result<ServingPoolSearchResult, ServingPoolSearchError> {
    let mut candidates = Vec::new();
    let mut summary = ServingPoolSearchSummary {
        max_candidates: search.max_candidates.max(1),
        ..ServingPoolSearchSummary::default()
    };

    for prefill_group in &search.prefill_groups {
        let prefill_group_nodes = sorted_group_nodes(cluster, prefill_group).map_err(|reason| {
            ServingPoolSearchError {
                reason,
                summary: summary.clone(),
            }
        })?;
        let prefill_group_node_count = prefill_group_nodes.len();
        let prefill_node_filter_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_group_nodes,
            &search.prefill_node_filter,
        );
        let prefill_node_filter_node_count = prefill_node_filter_nodes.len();
        let prefill_group_nodes = nodes_with_gpu_labels(
            cluster,
            &prefill_node_filter_nodes,
            &search.prefill_gpu_labels,
        );
        let prefill_gpu_filter_node_count = prefill_group_nodes.len();
        let prefill_counts =
            effective_counts(&search.prefill_node_counts, prefill_group_nodes.len());
        for decode_group in &search.decode_groups {
            let decode_group_nodes =
                sorted_group_nodes(cluster, decode_group).map_err(|reason| {
                    ServingPoolSearchError {
                        reason,
                        summary: summary.clone(),
                    }
                })?;
            let decode_group_node_count = decode_group_nodes.len();
            let decode_node_filter_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_group_nodes,
                &search.decode_node_filter,
            );
            let decode_node_filter_node_count = decode_node_filter_nodes.len();
            let decode_group_nodes = nodes_with_gpu_labels(
                cluster,
                &decode_node_filter_nodes,
                &search.decode_gpu_labels,
            );
            let decode_gpu_filter_node_count = decode_group_nodes.len();
            let decode_counts =
                effective_counts(&search.decode_node_counts, decode_group_nodes.len());
            let mut group_summary = ServingPoolSearchGroupSummary {
                prefill_group: prefill_group.clone(),
                decode_group: decode_group.clone(),
                prefill_group_node_count,
                prefill_node_filter_node_count,
                prefill_gpu_filter_node_count,
                decode_group_node_count,
                decode_node_filter_node_count,
                decode_gpu_filter_node_count,
                prefill_node_counts: prefill_counts.clone(),
                decode_node_counts: decode_counts.clone(),
                ..ServingPoolSearchGroupSummary::default()
            };

            for &prefill_count in &prefill_counts {
                for prefill_nodes in combinations(&prefill_group_nodes, prefill_count as usize) {
                    for &decode_count in &decode_counts {
                        for decode_nodes in combinations(&decode_group_nodes, decode_count as usize)
                        {
                            group_summary.considered_candidate_count += 1;
                            if !search.allow_overlap
                                && deployment_mode != ServingDeploymentMode::Colocated
                                && deployment_mode != ServingDeploymentMode::PartiallyDisaggregated
                                && overlaps_nodes(&prefill_nodes, &decode_nodes)
                            {
                                group_summary.rejected_overlap_count += 1;
                                continue;
                            }
                            if !deployment_mode.accepts_pool(&prefill_nodes, &decode_nodes) {
                                group_summary.rejected_mode_count += 1;
                                continue;
                            }
                            if !pool_search_candidate_satisfies_domain_spread(
                                cluster,
                                search,
                                &prefill_nodes,
                                &decode_nodes,
                            ) {
                                group_summary.rejected_domain_spread_count += 1;
                                continue;
                            }
                            let effective_mode = ServingDeploymentMode::effective_for_pool(
                                &prefill_nodes,
                                &decode_nodes,
                            );
                            let candidate = ServingPoolCandidate {
                                label: Some(format!(
                                    "auto:{}{}->{}{}",
                                    prefill_group,
                                    node_suffix(&prefill_nodes),
                                    decode_group,
                                    node_suffix(&decode_nodes)
                                )),
                                prefill_nodes: prefill_nodes.clone(),
                                decode_nodes,
                                prefill_groups: Vec::new(),
                                decode_groups: Vec::new(),
                                prefill_node_filter: ServingPoolNodeFilter::default(),
                                decode_node_filter: ServingPoolNodeFilter::default(),
                                domain_spread: search.domain_spread.clone(),
                                prefill_gpu_labels: search.prefill_gpu_labels.clone(),
                                decode_gpu_labels: search.decode_gpu_labels.clone(),
                            };
                            let previous_len = candidates.len();
                            push_unique_pool_candidate(&mut candidates, candidate);
                            if candidates.len() > previous_len {
                                group_summary.generated_candidate_count += 1;
                                match effective_mode {
                                    ServingDeploymentMode::Flexible => {}
                                    ServingDeploymentMode::Colocated => {
                                        group_summary.generated_colocated_count += 1;
                                    }
                                    ServingDeploymentMode::PartiallyDisaggregated => {
                                        group_summary.generated_partially_disaggregated_count += 1;
                                    }
                                    ServingDeploymentMode::FullyDisaggregated => {
                                        group_summary.generated_fully_disaggregated_count += 1;
                                    }
                                }
                            } else {
                                group_summary.duplicate_candidate_count += 1;
                            }
                            if candidates.len() >= search.max_candidates.max(1) {
                                summary.generated_candidate_count = candidates.len();
                                summary.truncated = true;
                                summary.groups.push(group_summary);
                                return Ok(ServingPoolSearchResult {
                                    candidates,
                                    summary,
                                });
                            }
                        }
                    }
                }
            }
            summary.groups.push(group_summary);
        }
    }

    if candidates.is_empty() {
        let reason = if deployment_mode == ServingDeploymentMode::Flexible {
            "serving.pool_search generated no valid pool candidates; check groups, counts, node topology filters, GPU labels, overlap policy, and topology-domain spread constraints".to_string()
        } else {
            format!(
                "serving.pool_search generated no valid pool candidates compatible with serving.mode '{}'; check groups, counts, node topology filters, GPU labels, overlap policy, and topology-domain spread constraints",
                deployment_mode.as_str()
            )
        };
        Err(ServingPoolSearchError { reason, summary })
    } else {
        summary.generated_candidate_count = candidates.len();
        Ok(ServingPoolSearchResult {
            candidates,
            summary,
        })
    }
}

fn pool_search_candidate_satisfies_domain_spread(
    cluster: &Cluster,
    search: &ServingPoolSearch,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
) -> bool {
    pool_nodes_satisfy_domain_spread(cluster, &search.domain_spread, prefill_nodes, decode_nodes)
}

fn pool_nodes_satisfy_domain_spread(
    cluster: &Cluster,
    domain_spread: &ServingPoolDomainSpread,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
) -> bool {
    pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    )
}

fn nodes_matching_pool_node_filter(
    cluster: &Cluster,
    node_ids: &[NodeId],
    filter: &ServingPoolNodeFilter,
) -> Vec<NodeId> {
    node_ids
        .iter()
        .copied()
        .filter(|node_id| {
            cluster
                .node(*node_id)
                .is_some_and(|node| node_matches_pool_node_filter(node, filter))
        })
        .collect()
}

fn node_matches_pool_node_filter(node: &Node, filter: &ServingPoolNodeFilter) -> bool {
    (filter.node_labels.is_empty()
        || filter
            .node_labels
            .iter()
            .any(|label| node.topology.labels.contains(label)))
        && (filter.racks.is_empty()
            || node
                .topology
                .rack
                .as_ref()
                .is_some_and(|rack| filter.racks.contains(rack)))
        && (filter.islands.is_empty()
            || node
                .topology
                .island
                .as_ref()
                .is_some_and(|island| filter.islands.contains(island)))
        && (filter.failure_domains.is_empty()
            || node
                .topology
                .failure_domain
                .as_ref()
                .is_some_and(|failure_domain| filter.failure_domains.contains(failure_domain)))
        && filter
            .exclude_node_labels
            .iter()
            .all(|label| !node.topology.labels.contains(label))
        && !node
            .topology
            .rack
            .as_ref()
            .is_some_and(|rack| filter.exclude_racks.contains(rack))
        && !node
            .topology
            .island
            .as_ref()
            .is_some_and(|island| filter.exclude_islands.contains(island))
        && !node
            .topology
            .failure_domain
            .as_ref()
            .is_some_and(|failure_domain| filter.exclude_failure_domains.contains(failure_domain))
}

fn pool_nodes_meet_min_domain_count(
    cluster: &Cluster,
    node_ids: &[NodeId],
    min_count: Option<u32>,
    domain: impl Fn(&Node) -> Option<&str>,
) -> bool {
    let Some(min_count) = min_count else {
        return true;
    };
    let domains = node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id))
        .filter_map(domain)
        .collect::<BTreeSet<_>>();
    domains.len() >= min_count as usize
}

fn sorted_group_nodes(cluster: &Cluster, group: &str) -> Result<Vec<NodeId>, String> {
    let mut nodes = cluster
        .node_group(group)
        .ok_or_else(|| format!("serving.pool_search references unknown node group '{group}'"))?
        .to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Err(format!(
            "serving.pool_search node group '{group}' contains no nodes"
        ));
    }
    Ok(nodes)
}

fn nodes_with_gpu_labels(
    cluster: &Cluster,
    node_ids: &[NodeId],
    gpu_labels: &[String],
) -> Vec<NodeId> {
    if gpu_labels.is_empty() {
        return node_ids.to_vec();
    }
    node_ids
        .iter()
        .copied()
        .filter(|node_id| {
            cluster.node(*node_id).is_some_and(|node| {
                node.gpus.keys().copied().any(|local_gpu_id| {
                    cluster.is_gpu_available(GpuAddr {
                        node_id: *node_id,
                        local_gpu_id,
                    }) && node
                        .gpu_labels(local_gpu_id)
                        .is_some_and(|labels| gpu_labels.iter().any(|label| labels.contains(label)))
                })
            })
        })
        .collect()
}

fn effective_counts(configured: &[u32], available: usize) -> Vec<u32> {
    let mut counts = if configured.is_empty() {
        vec![available as u32]
    } else {
        configured.to_vec()
    };
    counts.sort_unstable();
    counts.dedup();
    counts
        .into_iter()
        .filter(|count| *count > 0 && (*count as usize) <= available)
        .collect()
}

fn combinations(values: &[NodeId], count: usize) -> Vec<Vec<NodeId>> {
    if count == 0 || count > values.len() {
        return Vec::new();
    }
    if count == values.len() {
        return vec![values.to_vec()];
    }

    let mut results = Vec::new();
    let mut current = Vec::with_capacity(count);
    push_combinations(values, count, 0, &mut current, &mut results);
    results
}

fn push_combinations(
    values: &[NodeId],
    count: usize,
    start: usize,
    current: &mut Vec<NodeId>,
    results: &mut Vec<Vec<NodeId>>,
) {
    if current.len() == count {
        results.push(current.clone());
        return;
    }

    let needed = count - current.len();
    for idx in start..=values.len() - needed {
        current.push(values[idx]);
        push_combinations(values, count, idx + 1, current, results);
        current.pop();
    }
}

pub(super) fn overlaps_nodes(left: &[NodeId], right: &[NodeId]) -> bool {
    left.iter().any(|node| right.contains(node))
}

fn node_suffix(nodes: &[NodeId]) -> String {
    format!(
        "[{}]",
        nodes
            .iter()
            .map(|node| node.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(super) fn node_list(nodes: &[NodeId]) -> String {
    nodes
        .iter()
        .map(|node| node.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn extend_unique_pool_candidates(
    candidates: &mut Vec<ServingPoolCandidate>,
    generated: Vec<ServingPoolCandidate>,
) {
    for candidate in generated {
        push_unique_pool_candidate(candidates, candidate);
    }
}

fn push_unique_pool_candidate(
    candidates: &mut Vec<ServingPoolCandidate>,
    candidate: ServingPoolCandidate,
) {
    if !candidates
        .iter()
        .any(|existing| same_pool_candidate(existing, &candidate))
    {
        candidates.push(candidate);
    }
}

fn same_pool_candidate(left: &ServingPoolCandidate, right: &ServingPoolCandidate) -> bool {
    same_u32s(&left.prefill_nodes, &right.prefill_nodes)
        && same_u32s(&left.decode_nodes, &right.decode_nodes)
        && same_strings(&left.prefill_groups, &right.prefill_groups)
        && same_strings(&left.decode_groups, &right.decode_groups)
        && same_pool_node_filter(&left.prefill_node_filter, &right.prefill_node_filter)
        && same_pool_node_filter(&left.decode_node_filter, &right.decode_node_filter)
        && left.domain_spread == right.domain_spread
        && same_strings(&left.prefill_gpu_labels, &right.prefill_gpu_labels)
        && same_strings(&left.decode_gpu_labels, &right.decode_gpu_labels)
}

fn same_pool_node_filter(left: &ServingPoolNodeFilter, right: &ServingPoolNodeFilter) -> bool {
    same_strings(&left.node_labels, &right.node_labels)
        && same_strings(&left.racks, &right.racks)
        && same_strings(&left.islands, &right.islands)
        && same_strings(&left.failure_domains, &right.failure_domains)
        && same_strings(&left.exclude_node_labels, &right.exclude_node_labels)
        && same_strings(&left.exclude_racks, &right.exclude_racks)
        && same_strings(&left.exclude_islands, &right.exclude_islands)
        && same_strings(
            &left.exclude_failure_domains,
            &right.exclude_failure_domains,
        )
}

pub(super) fn same_u32s(left: &[u32], right: &[u32]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn same_strings(left: &[String], right: &[String]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort();
    right.sort();
    left == right
}

pub(super) fn resolve_pool_candidate(
    cluster: &Cluster,
    candidate: &ServingPoolCandidate,
) -> Result<ResolvedServingPool, String> {
    let prefill_nodes = resolve_pool_nodes(
        cluster,
        "prefill_groups",
        &candidate.prefill_nodes,
        &candidate.prefill_groups,
    )?;
    let prefill_nodes =
        nodes_matching_pool_node_filter(cluster, &prefill_nodes, &candidate.prefill_node_filter);
    if prefill_nodes.is_empty() {
        return Err("serving pool prefill node topology filters matched no nodes".to_string());
    }
    let decode_nodes = resolve_pool_nodes(
        cluster,
        "decode_groups",
        &candidate.decode_nodes,
        &candidate.decode_groups,
    )?;
    let decode_nodes =
        nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
    if decode_nodes.is_empty() {
        return Err("serving pool decode node topology filters matched no nodes".to_string());
    }
    if !pool_nodes_satisfy_domain_spread(
        cluster,
        &candidate.domain_spread,
        &prefill_nodes,
        &decode_nodes,
    ) {
        return Err(
            "serving pool does not satisfy configured topology-domain spread constraints"
                .to_string(),
        );
    }

    Ok(ResolvedServingPool {
        label: candidate.label.clone(),
        prefill_nodes,
        decode_nodes,
        prefill_gpu_labels: candidate.prefill_gpu_labels.clone(),
        decode_gpu_labels: candidate.decode_gpu_labels.clone(),
    })
}

fn resolve_pool_nodes(
    cluster: &Cluster,
    field_name: &str,
    explicit_nodes: &[NodeId],
    groups: &[String],
) -> Result<Vec<NodeId>, String> {
    let mut nodes = explicit_nodes.to_vec();
    for group in groups {
        let group_nodes = cluster.node_group(group).ok_or_else(|| {
            format!("serving pool {field_name} references unknown group '{group}'")
        })?;
        nodes.extend_from_slice(group_nodes);
    }
    nodes.sort_unstable();
    nodes.dedup();

    if nodes.is_empty() {
        return Err(format!("serving pool {field_name} resolved to no nodes"));
    }

    Ok(nodes)
}

pub(super) fn routed_serving_node(nodes: &[NodeId], request_idx: u32) -> NodeId {
    if nodes.is_empty() {
        return 0;
    }

    let mut nodes = nodes.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    nodes[request_idx as usize % nodes.len()]
}

pub(super) fn sorted_unique_nodes(nodes: &[NodeId]) -> Vec<NodeId> {
    let mut nodes = nodes.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        nodes.push(0);
    }
    nodes
}
