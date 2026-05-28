use super::*;

pub(in crate::config) fn parse_pool_candidates(
    serving: &ServingSection,
) -> Result<Vec<ServingPoolCandidate>, ConfigError> {
    let mut candidates = Vec::new();
    let has_top_level_pool = serving.prefill_nodes.is_some()
        || serving.decode_nodes.is_some()
        || serving.prefill_groups.is_some()
        || serving.decode_groups.is_some();
    if has_top_level_pool {
        candidates.push(parse_pool_candidate(
            "serving",
            PoolCandidateInput {
                label: None,
                prefill_nodes: serving.prefill_nodes.clone().unwrap_or_default(),
                decode_nodes: serving.decode_nodes.clone().unwrap_or_default(),
                prefill_groups: serving.prefill_groups.clone().unwrap_or_default(),
                decode_groups: serving.decode_groups.clone().unwrap_or_default(),
                prefill_node_filter: ServingPoolNodeFilter::default(),
                decode_node_filter: ServingPoolNodeFilter::default(),
                domain_spread: ServingPoolDomainSpread::default(),
                prefill_gpu_tag: serving.prefill_gpu_tag.clone(),
                prefill_gpu_tags: serving.prefill_gpu_tags.clone(),
                decode_gpu_tag: serving.decode_gpu_tag.clone(),
                decode_gpu_tags: serving.decode_gpu_tags.clone(),
            },
        )?);
    }

    for (idx, candidate) in serving
        .pool_candidates
        .as_ref()
        .into_iter()
        .flatten()
        .enumerate()
    {
        candidates.push(parse_pool_candidate(
            &format!("serving.pool_candidates[{idx}]"),
            PoolCandidateInput {
                label: candidate.label.clone(),
                prefill_nodes: candidate.prefill_nodes.clone().unwrap_or_default(),
                decode_nodes: candidate.decode_nodes.clone().unwrap_or_default(),
                prefill_groups: candidate.prefill_groups.clone().unwrap_or_default(),
                decode_groups: candidate.decode_groups.clone().unwrap_or_default(),
                prefill_node_filter: parse_serving_pool_candidate_node_filter(
                    &format!("serving.pool_candidates[{idx}].prefill"),
                    candidate.prefill_node_tag.as_deref(),
                    candidate.prefill_node_tags.as_deref(),
                    candidate.prefill_rack.as_deref(),
                    candidate.prefill_racks.as_deref(),
                    candidate.prefill_island.as_deref(),
                    candidate.prefill_islands.as_deref(),
                    candidate.prefill_failure_domain.as_deref(),
                    candidate.prefill_failure_domains.as_deref(),
                    candidate.prefill_exclude_node_tag.as_deref(),
                    candidate.prefill_exclude_node_tags.as_deref(),
                    candidate.prefill_exclude_rack.as_deref(),
                    candidate.prefill_exclude_racks.as_deref(),
                    candidate.prefill_exclude_island.as_deref(),
                    candidate.prefill_exclude_islands.as_deref(),
                    candidate.prefill_exclude_failure_domain.as_deref(),
                    candidate.prefill_exclude_failure_domains.as_deref(),
                )?,
                decode_node_filter: parse_serving_pool_candidate_node_filter(
                    &format!("serving.pool_candidates[{idx}].decode"),
                    candidate.decode_node_tag.as_deref(),
                    candidate.decode_node_tags.as_deref(),
                    candidate.decode_rack.as_deref(),
                    candidate.decode_racks.as_deref(),
                    candidate.decode_island.as_deref(),
                    candidate.decode_islands.as_deref(),
                    candidate.decode_failure_domain.as_deref(),
                    candidate.decode_failure_domains.as_deref(),
                    candidate.decode_exclude_node_tag.as_deref(),
                    candidate.decode_exclude_node_tags.as_deref(),
                    candidate.decode_exclude_rack.as_deref(),
                    candidate.decode_exclude_racks.as_deref(),
                    candidate.decode_exclude_island.as_deref(),
                    candidate.decode_exclude_islands.as_deref(),
                    candidate.decode_exclude_failure_domain.as_deref(),
                    candidate.decode_exclude_failure_domains.as_deref(),
                )?,
                domain_spread: parse_serving_pool_candidate_domain_spread(
                    &format!("serving.pool_candidates[{idx}]"),
                    candidate,
                )?,
                prefill_gpu_tag: candidate.prefill_gpu_tag.clone(),
                prefill_gpu_tags: candidate.prefill_gpu_tags.clone(),
                decode_gpu_tag: candidate.decode_gpu_tag.clone(),
                decode_gpu_tags: candidate.decode_gpu_tags.clone(),
            },
        )?);
    }

    Ok(dedup_pool_candidates(candidates))
}

pub(in crate::config) fn parse_pool_search(
    pool_search: Option<ServingPoolSearchSection>,
    deployment_mode: ServingDeploymentMode,
) -> Result<Option<ServingPoolSearch>, ConfigError> {
    let Some(pool_search) = pool_search else {
        return Ok(None);
    };
    let max_candidates = pool_search.max_candidates.unwrap_or(64);
    if max_candidates == 0 {
        return Err(ConfigError::new(
            "serving.pool_search.max_candidates must be greater than zero",
        ));
    }
    let prefill_node_filter = parse_pool_search_node_filter(
        "serving.pool_search.prefill",
        pool_search.prefill_node_tag.as_deref(),
        pool_search.prefill_node_tags.as_deref(),
        pool_search.prefill_rack.as_deref(),
        pool_search.prefill_racks.as_deref(),
        pool_search.prefill_island.as_deref(),
        pool_search.prefill_islands.as_deref(),
        pool_search.prefill_failure_domain.as_deref(),
        pool_search.prefill_failure_domains.as_deref(),
        pool_search.prefill_exclude_node_tag.as_deref(),
        pool_search.prefill_exclude_node_tags.as_deref(),
        pool_search.prefill_exclude_rack.as_deref(),
        pool_search.prefill_exclude_racks.as_deref(),
        pool_search.prefill_exclude_island.as_deref(),
        pool_search.prefill_exclude_islands.as_deref(),
        pool_search.prefill_exclude_failure_domain.as_deref(),
        pool_search.prefill_exclude_failure_domains.as_deref(),
    )?;
    let decode_node_filter = parse_pool_search_node_filter(
        "serving.pool_search.decode",
        pool_search.decode_node_tag.as_deref(),
        pool_search.decode_node_tags.as_deref(),
        pool_search.decode_rack.as_deref(),
        pool_search.decode_racks.as_deref(),
        pool_search.decode_island.as_deref(),
        pool_search.decode_islands.as_deref(),
        pool_search.decode_failure_domain.as_deref(),
        pool_search.decode_failure_domains.as_deref(),
        pool_search.decode_exclude_node_tag.as_deref(),
        pool_search.decode_exclude_node_tags.as_deref(),
        pool_search.decode_exclude_rack.as_deref(),
        pool_search.decode_exclude_racks.as_deref(),
        pool_search.decode_exclude_island.as_deref(),
        pool_search.decode_exclude_islands.as_deref(),
        pool_search.decode_exclude_failure_domain.as_deref(),
        pool_search.decode_exclude_failure_domains.as_deref(),
    )?;
    let domain_spread = parse_pool_search_domain_spread(&pool_search)?;

    Ok(Some(ServingPoolSearch {
        prefill_groups: normalized_group_labels(
            "serving.pool_search.prefill_groups",
            require_nonempty(
                "serving.pool_search.prefill_groups",
                pool_search.prefill_groups,
            )?,
        )?,
        decode_groups: normalized_group_labels(
            "serving.pool_search.decode_groups",
            require_nonempty(
                "serving.pool_search.decode_groups",
                pool_search.decode_groups,
            )?,
        )?,
        prefill_node_counts: positive_values(
            "serving.pool_search.prefill_node_counts",
            pool_search.prefill_node_counts.unwrap_or_default(),
        )?,
        decode_node_counts: positive_values(
            "serving.pool_search.decode_node_counts",
            pool_search.decode_node_counts.unwrap_or_default(),
        )?,
        prefill_node_filter,
        decode_node_filter,
        prefill_gpu_labels: parse_gpu_labels(
            "serving.pool_search.prefill_gpu_tags",
            pool_search.prefill_gpu_tag.as_deref(),
            pool_search.prefill_gpu_tags.as_deref(),
        )?
        .into_iter()
        .collect(),
        decode_gpu_labels: parse_gpu_labels(
            "serving.pool_search.decode_gpu_tags",
            pool_search.decode_gpu_tag.as_deref(),
            pool_search.decode_gpu_tags.as_deref(),
        )?
        .into_iter()
        .collect(),
        allow_overlap: pool_search.allow_overlap.unwrap_or(matches!(
            deployment_mode,
            ServingDeploymentMode::Colocated | ServingDeploymentMode::PartiallyDisaggregated
        )),
        domain_spread,
        max_candidates,
    }))
}

pub(in crate::config) fn parse_serving_pool_candidate_domain_spread(
    name: &str,
    candidate: &ServingPoolCandidateSection,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    parse_pool_domain_spread(
        name,
        candidate.min_prefill_racks,
        candidate.min_decode_racks,
        candidate.min_prefill_islands,
        candidate.min_decode_islands,
        candidate.min_prefill_failure_domains,
        candidate.min_decode_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::config) fn parse_serving_pool_candidate_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    parse_pool_node_filter(
        name,
        node_tag,
        node_tags,
        rack,
        racks,
        island,
        islands,
        failure_domain,
        failure_domains,
        exclude_node_tag,
        exclude_node_tags,
        exclude_rack,
        exclude_racks,
        exclude_island,
        exclude_islands,
        exclude_failure_domain,
        exclude_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::config) fn parse_pool_search_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    parse_pool_node_filter(
        name,
        node_tag,
        node_tags,
        rack,
        racks,
        island,
        islands,
        failure_domain,
        failure_domains,
        exclude_node_tag,
        exclude_node_tags,
        exclude_rack,
        exclude_racks,
        exclude_island,
        exclude_islands,
        exclude_failure_domain,
        exclude_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::config) fn parse_pool_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    Ok(ServingPoolNodeFilter {
        node_labels: parse_optional_normalized_labels(
            &format!("{name}_node_tags"),
            node_tag,
            node_tags,
        )?,
        racks: parse_optional_topology_domains(&format!("{name}_racks"), rack, racks)?,
        islands: parse_optional_topology_domains(&format!("{name}_islands"), island, islands)?,
        failure_domains: parse_optional_topology_domains(
            &format!("{name}_failure_domains"),
            failure_domain,
            failure_domains,
        )?,
        exclude_node_labels: parse_optional_normalized_labels(
            &format!("{name}_exclude_node_tags"),
            exclude_node_tag,
            exclude_node_tags,
        )?,
        exclude_racks: parse_optional_topology_domains(
            &format!("{name}_exclude_racks"),
            exclude_rack,
            exclude_racks,
        )?,
        exclude_islands: parse_optional_topology_domains(
            &format!("{name}_exclude_islands"),
            exclude_island,
            exclude_islands,
        )?,
        exclude_failure_domains: parse_optional_topology_domains(
            &format!("{name}_exclude_failure_domains"),
            exclude_failure_domain,
            exclude_failure_domains,
        )?,
    })
}

pub(in crate::config) fn parse_pool_search_domain_spread(
    pool_search: &ServingPoolSearchSection,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    parse_pool_domain_spread(
        "serving.pool_search",
        pool_search.min_prefill_racks,
        pool_search.min_decode_racks,
        pool_search.min_prefill_islands,
        pool_search.min_decode_islands,
        pool_search.min_prefill_failure_domains,
        pool_search.min_decode_failure_domains,
    )
}

pub(in crate::config) fn parse_pool_domain_spread(
    name: &str,
    min_prefill_racks: Option<u32>,
    min_decode_racks: Option<u32>,
    min_prefill_islands: Option<u32>,
    min_decode_islands: Option<u32>,
    min_prefill_failure_domains: Option<u32>,
    min_decode_failure_domains: Option<u32>,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    validate_positive_optional_u32(&format!("{name}.min_prefill_racks"), min_prefill_racks)?;
    validate_positive_optional_u32(&format!("{name}.min_decode_racks"), min_decode_racks)?;
    validate_positive_optional_u32(&format!("{name}.min_prefill_islands"), min_prefill_islands)?;
    validate_positive_optional_u32(&format!("{name}.min_decode_islands"), min_decode_islands)?;
    validate_positive_optional_u32(
        &format!("{name}.min_prefill_failure_domains"),
        min_prefill_failure_domains,
    )?;
    validate_positive_optional_u32(
        &format!("{name}.min_decode_failure_domains"),
        min_decode_failure_domains,
    )?;
    Ok(ServingPoolDomainSpread {
        min_prefill_racks,
        min_decode_racks,
        min_prefill_islands,
        min_decode_islands,
        min_prefill_failure_domains,
        min_decode_failure_domains,
    })
}

pub(in crate::config) struct PoolCandidateInput {
    label: Option<String>,
    prefill_nodes: Vec<u32>,
    decode_nodes: Vec<u32>,
    prefill_groups: Vec<String>,
    decode_groups: Vec<String>,
    prefill_node_filter: ServingPoolNodeFilter,
    decode_node_filter: ServingPoolNodeFilter,
    domain_spread: ServingPoolDomainSpread,
    prefill_gpu_tag: Option<String>,
    prefill_gpu_tags: Option<Vec<String>>,
    decode_gpu_tag: Option<String>,
    decode_gpu_tags: Option<Vec<String>>,
}

pub(in crate::config) fn parse_pool_candidate(
    name: &str,
    input: PoolCandidateInput,
) -> Result<ServingPoolCandidate, ConfigError> {
    let prefill_groups =
        normalized_group_labels(&format!("{name}.prefill_groups"), input.prefill_groups)?;
    let decode_groups =
        normalized_group_labels(&format!("{name}.decode_groups"), input.decode_groups)?;
    let prefill_gpu_labels = parse_gpu_labels(
        &format!("{name}.prefill_gpu_tags"),
        input.prefill_gpu_tag.as_deref(),
        input.prefill_gpu_tags.as_deref(),
    )?
    .into_iter()
    .collect();
    let decode_gpu_labels = parse_gpu_labels(
        &format!("{name}.decode_gpu_tags"),
        input.decode_gpu_tag.as_deref(),
        input.decode_gpu_tags.as_deref(),
    )?
    .into_iter()
    .collect();
    if input.prefill_nodes.is_empty() && prefill_groups.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} requires prefill_nodes or prefill_groups"
        )));
    }
    if input.decode_nodes.is_empty() && decode_groups.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} requires decode_nodes or decode_groups"
        )));
    }

    Ok(ServingPoolCandidate {
        label: input.label.filter(|label| !label.trim().is_empty()),
        prefill_nodes: input.prefill_nodes,
        decode_nodes: input.decode_nodes,
        prefill_groups,
        decode_groups,
        prefill_node_filter: input.prefill_node_filter,
        decode_node_filter: input.decode_node_filter,
        domain_spread: input.domain_spread,
        prefill_gpu_labels,
        decode_gpu_labels,
    })
}

pub(in crate::config) fn normalized_group_labels(
    name: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ConfigError> {
    let mut labels = Vec::new();
    for value in values {
        let label = normalize(&value);
        if label.is_empty() {
            return Err(ConfigError::new(format!("{name} values must not be empty")));
        }
        if !labels.contains(&label) {
            labels.push(label);
        }
    }

    Ok(labels)
}

pub(in crate::config) fn dedup_pool_candidates(
    candidates: Vec<ServingPoolCandidate>,
) -> Vec<ServingPoolCandidate> {
    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped.iter().any(|existing: &ServingPoolCandidate| {
            same_nodes(&existing.prefill_nodes, &candidate.prefill_nodes)
                && same_nodes(&existing.decode_nodes, &candidate.decode_nodes)
                && same_strings(&existing.prefill_groups, &candidate.prefill_groups)
                && same_strings(&existing.decode_groups, &candidate.decode_groups)
                && same_pool_node_filter(
                    &existing.prefill_node_filter,
                    &candidate.prefill_node_filter,
                )
                && same_pool_node_filter(
                    &existing.decode_node_filter,
                    &candidate.decode_node_filter,
                )
                && existing.domain_spread == candidate.domain_spread
                && same_strings(&existing.prefill_gpu_labels, &candidate.prefill_gpu_labels)
                && same_strings(&existing.decode_gpu_labels, &candidate.decode_gpu_labels)
        }) {
            deduped.push(candidate);
        }
    }

    deduped
}

pub(in crate::config) fn same_nodes(left: &[u32], right: &[u32]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

pub(in crate::config) fn same_strings(left: &[String], right: &[String]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort();
    right.sort();
    left == right
}

pub(in crate::config) fn same_pool_node_filter(
    left: &ServingPoolNodeFilter,
    right: &ServingPoolNodeFilter,
) -> bool {
    same_strings(&left.node_labels, &right.node_labels)
        && same_strings(&left.racks, &right.racks)
        && same_strings(&left.islands, &right.islands)
        && same_strings(&left.failure_domains, &right.failure_domains)
}

pub(in crate::config) fn validate_search_space_capacity(
    name: &str,
    search: &SearchSpace,
    available_gpus: u32,
) -> Result<(), ConfigError> {
    if search.tensor_ranks.iter().any(|&tensor| {
        search.pipeline_ranks.iter().any(|&pipeline| {
            search.expert_ranks.iter().any(|&expert| {
                search.data_ranks.iter().any(|&data| {
                    tensor
                        .saturating_mul(pipeline)
                        .saturating_mul(expert)
                        .saturating_mul(data)
                        <= available_gpus
                })
            })
        })
    }) {
        return Ok(());
    }

    Err(ConfigError::new(format!(
        "{name} has no rank combination that fits {available_gpus} available GPUs"
    )))
}

pub(in crate::config) fn validate_pool_candidate_for_cluster(
    name: &str,
    cluster: &Cluster,
    candidate: &ServingPoolCandidate,
    deployment_mode: ServingDeploymentMode,
    search: &ServingSearchSpace,
    require_routable_pools: bool,
    model_dtype: DType,
) -> Result<(), ConfigError> {
    validate_no_duplicate_nodes(&format!("{name}.prefill_nodes"), &candidate.prefill_nodes)?;
    validate_no_duplicate_nodes(&format!("{name}.decode_nodes"), &candidate.decode_nodes)?;
    let prefill_nodes = resolve_configured_pool_nodes(
        &format!("{name}.prefill"),
        cluster,
        &candidate.prefill_nodes,
        &candidate.prefill_groups,
    )?;
    let prefill_nodes =
        nodes_matching_pool_node_filter(cluster, &prefill_nodes, &candidate.prefill_node_filter);
    if prefill_nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name}.prefill node topology filters matched no nodes"
        )));
    }
    let decode_nodes = resolve_configured_pool_nodes(
        &format!("{name}.decode"),
        cluster,
        &candidate.decode_nodes,
        &candidate.decode_groups,
    )?;
    let decode_nodes =
        nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
    if decode_nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name}.decode node topology filters matched no nodes"
        )));
    }
    if !pool_nodes_satisfy_domain_spread(
        cluster,
        &candidate.domain_spread,
        &prefill_nodes,
        &decode_nodes,
    ) {
        return Err(ConfigError::new(format!(
            "{name} does not satisfy configured topology-domain spread constraints"
        )));
    }
    validate_serving_deployment_mode(name, deployment_mode, &prefill_nodes, &decode_nodes)?;
    validate_search_space_capacity(
        &format!(
            "{name}.prefill_search for model.dtype {}",
            model_dtype_label(model_dtype)
        ),
        &search.prefill,
        gpu_count_for_nodes_with_labels_and_dtype(
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            model_dtype,
        ),
    )?;
    validate_search_space_capacity(
        &format!(
            "{name}.decode_search for model.dtype {}",
            model_dtype_label(model_dtype)
        ),
        &search.decode,
        gpu_count_for_nodes_with_labels_and_dtype(
            cluster,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            model_dtype,
        ),
    )?;
    if require_routable_pools {
        validate_serving_pool_route(
            name,
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
        )?;
    }
    Ok(())
}

pub(in crate::config) fn validate_pool_search_for_cluster(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    search: &ServingSearchSpace,
    require_routable_pools: bool,
    model_dtype: DType,
) -> Result<(), ConfigError> {
    let mut any_candidate = false;
    let mut any_routable_candidate = false;
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;
        validate_search_space_capacity(
            &format!(
                "serving.pool_search.prefill group '{prefill_group}' for model.dtype {}",
                model_dtype_label(model_dtype)
            ),
            &search.prefill,
            gpu_count_for_nodes_with_labels_and_dtype(
                cluster,
                &prefill_nodes,
                &pool_search.prefill_gpu_labels,
                model_dtype,
            ),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;
            validate_search_space_capacity(
                &format!(
                    "serving.pool_search.decode group '{decode_group}' for model.dtype {}",
                    model_dtype_label(model_dtype)
                ),
                &search.decode,
                gpu_count_for_nodes_with_labels_and_dtype(
                    cluster,
                    &decode_nodes,
                    &pool_search.decode_gpu_labels,
                    model_dtype,
                ),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    if pool_search_can_generate_candidate(
                        cluster,
                        pool_search,
                        &prefill_nodes,
                        prefill_count as usize,
                        &decode_nodes,
                        decode_count as usize,
                        deployment_mode,
                    ) {
                        any_candidate = true;
                        if !require_routable_pools
                            || pool_search_can_generate_routable_candidate(
                                cluster,
                                pool_search,
                                &prefill_nodes,
                                &decode_nodes,
                                (prefill_count as usize, decode_count as usize),
                                (
                                    &pool_search.prefill_gpu_labels,
                                    &pool_search.decode_gpu_labels,
                                ),
                                deployment_mode,
                            )
                        {
                            any_routable_candidate = true;
                        }
                    }
                }
            }
        }
    }

    if !any_candidate {
        Err(ConfigError::new(
            "serving.pool_search cannot generate any prefill/decode pool candidate with the configured groups, counts, node topology filters, GPU labels, overlap policy, and topology-domain spread constraints",
        ))
    } else if require_routable_pools && !any_routable_candidate {
        Err(ConfigError::new(
            "serving.pool_search cannot generate any routable prefill/decode pool candidate with the configured groups, counts, overlap policy, and cluster interconnect; add missing links, choose connected pools, or disable serving.require_routable_pools for failure-scenario sweeps",
        ))
    } else {
        Ok(())
    }
}

pub(in crate::config) fn validate_explicit_serving_placements_for_configured_pools(
    cluster: &Cluster,
    serving: &DisaggregatedServingConfig,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    if prefill_placement.is_none() && decode_placement.is_none() {
        return Ok(());
    }

    if serving.pool_candidates.is_empty()
        && serving.pool_search.is_none()
        && !serving.prefill_nodes.is_empty()
        && !serving.decode_nodes.is_empty()
        && explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &serving.prefill_nodes,
            &[],
            &serving.decode_nodes,
            &[],
            prefill_placement,
            decode_placement,
        )
    {
        return Ok(());
    }

    for candidate in &serving.pool_candidates {
        let prefill_nodes = resolve_configured_pool_nodes(
            "serving.pool_candidates.prefill",
            cluster,
            &candidate.prefill_nodes,
            &candidate.prefill_groups,
        )?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &candidate.prefill_node_filter,
        );
        let decode_nodes = resolve_configured_pool_nodes(
            "serving.pool_candidates.decode",
            cluster,
            &candidate.decode_nodes,
            &candidate.decode_groups,
        )?;
        let decode_nodes =
            nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
        if pool_nodes_satisfy_domain_spread(
            cluster,
            &candidate.domain_spread,
            &prefill_nodes,
            &decode_nodes,
        ) && explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            prefill_placement,
            decode_placement,
        ) {
            return Ok(());
        }
    }

    if let Some(pool_search) = &serving.pool_search
        && pool_search_can_generate_explicit_placement_candidate(
            cluster,
            pool_search,
            serving.deployment_mode,
            prefill_placement,
            decode_placement,
        )?
    {
        return Ok(());
    }

    Err(ConfigError::new(
        "serving explicit prefill/decode placements do not fit any configured serving pool; ensure serving.prefill_placement ranks are inside prefill pools and serving.decode_placement ranks are inside decode pools, including GPU label filters, or remove explicit placements",
    ))
}

pub(in crate::config) fn validate_serving_route_constraints_for_configured_pools(
    cluster: &Cluster,
    serving: &DisaggregatedServingConfig,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    let constraints = serving.traffic.kv_route_constraints;
    if !constraints.any() {
        return Ok(());
    }

    let placements = ExplicitServingPlacements {
        prefill: prefill_placement,
        decode: decode_placement,
    };
    let mut checked_candidates = 0usize;
    let mut first_failure = None;

    for (idx, candidate) in serving.pool_candidates.iter().enumerate() {
        let prefill_nodes = resolve_configured_pool_nodes(
            &format!("serving.pool_candidates[{idx}].prefill"),
            cluster,
            &candidate.prefill_nodes,
            &candidate.prefill_groups,
        )?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &candidate.prefill_node_filter,
        );
        let decode_nodes = resolve_configured_pool_nodes(
            &format!("serving.pool_candidates[{idx}].decode"),
            cluster,
            &candidate.decode_nodes,
            &candidate.decode_groups,
        )?;
        let decode_nodes =
            nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
        if !pool_nodes_satisfy_domain_spread(
            cluster,
            &candidate.domain_spread,
            &prefill_nodes,
            &decode_nodes,
        ) || !explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            prefill_placement,
            decode_placement,
        ) {
            continue;
        }
        checked_candidates += 1;
        match serving_pool_route_constraints_satisfied(
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            constraints,
            placements,
        ) {
            Ok(()) => return Ok(()),
            Err(err) => first_failure.get_or_insert(err),
        };
    }

    if let Some(pool_search) = &serving.pool_search
        && pool_search_can_generate_route_constraint_candidate(
            cluster,
            pool_search,
            serving.deployment_mode,
            constraints,
            placements,
            &mut checked_candidates,
            &mut first_failure,
        )?
    {
        return Ok(());
    }

    let first_failure = first_failure
        .map(|failure| format!(" first failure: {failure}"))
        .unwrap_or_default();
    Err(ConfigError::new(format!(
        "serving has no configured prefill/decode pool candidate satisfying configured KV route constraints after checking {checked_candidates} candidate(s); add rail-diverse/GPUDirect-capable routes, choose different pools, or lower serving.min_kv_route_rail_count / serving.require_kv_route_rail_metadata / serving.require_gpudirect_kv_paths.{first_failure}"
    )))
}

#[derive(Copy, Clone)]
pub(in crate::config) struct ExplicitServingPlacements<'a> {
    prefill: Option<&'a RankPlacement>,
    decode: Option<&'a RankPlacement>,
}

pub(in crate::config) fn pool_search_can_generate_route_constraint_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    constraints: ServingKvRouteConstraints,
    placements: ExplicitServingPlacements<'_>,
    checked_candidates: &mut usize,
    first_failure: &mut Option<String>,
) -> Result<bool, ConfigError> {
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    for prefill in combinations(&prefill_nodes, prefill_count as usize) {
                        for decode in combinations(&decode_nodes, decode_count as usize) {
                            if !pool_search_concrete_candidate_satisfies_constraints(
                                cluster,
                                pool_search,
                                &prefill,
                                &decode,
                                deployment_mode,
                            ) || !explicit_serving_placements_match_pool(
                                cluster,
                                deployment_mode,
                                &prefill,
                                &pool_search.prefill_gpu_labels,
                                &decode,
                                &pool_search.decode_gpu_labels,
                                placements.prefill,
                                placements.decode,
                            ) {
                                continue;
                            }
                            *checked_candidates += 1;
                            match serving_pool_route_constraints_satisfied(
                                cluster,
                                &prefill,
                                &pool_search.prefill_gpu_labels,
                                &decode,
                                &pool_search.decode_gpu_labels,
                                constraints,
                                placements,
                            ) {
                                Ok(()) => return Ok(true),
                                Err(err) => {
                                    first_failure.get_or_insert(err);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::config) fn explicit_serving_placements_match_pool(
    cluster: &Cluster,
    deployment_mode: ServingDeploymentMode,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> bool {
    deployment_mode.accepts_pool(prefill_nodes, decode_nodes)
        && prefill_placement.is_none_or(|placement| {
            placement_matches_pool_scope(cluster, placement, prefill_nodes, prefill_gpu_labels)
        })
        && decode_placement.is_none_or(|placement| {
            placement_matches_pool_scope(cluster, placement, decode_nodes, decode_gpu_labels)
        })
}

pub(in crate::config) fn placement_matches_pool_scope(
    cluster: &Cluster,
    placement: &RankPlacement,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> bool {
    let node_ids = node_ids.iter().copied().collect::<BTreeSet<_>>();
    placement.rank_to_gpu.iter().all(|addr| {
        node_ids.contains(&addr.node_id)
            && cluster.node(addr.node_id).is_some_and(|node| {
                gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
            })
    })
}

pub(in crate::config) fn pool_search_can_generate_explicit_placement_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<bool, ConfigError> {
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    if pool_search_can_generate_explicit_placement_candidate_for_counts(
                        cluster,
                        pool_search,
                        &prefill_nodes,
                        prefill_count as usize,
                        &decode_nodes,
                        decode_count as usize,
                        deployment_mode,
                        prefill_placement,
                        decode_placement,
                    ) {
                        return Ok(true);
                    }
                }
            }
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::config) fn pool_search_can_generate_explicit_placement_candidate_for_counts(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    prefill_count: usize,
    decode_nodes: &[u32],
    decode_count: usize,
    deployment_mode: ServingDeploymentMode,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> bool {
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) && explicit_serving_placements_match_pool(
                cluster,
                deployment_mode,
                &prefill,
                &pool_search.prefill_gpu_labels,
                &decode,
                &pool_search.decode_gpu_labels,
                prefill_placement,
                decode_placement,
            ) {
                return true;
            }
        }
    }
    false
}

pub(in crate::config) fn validate_serving_deployment_mode(
    name: &str,
    deployment_mode: ServingDeploymentMode,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
) -> Result<(), ConfigError> {
    if deployment_mode.accepts_pool(prefill_nodes, decode_nodes) {
        return Ok(());
    }

    let effective = ServingDeploymentMode::effective_for_pool(prefill_nodes, decode_nodes);
    Err(ConfigError::new(format!(
        "{name} uses {} prefill/decode nodes, but serving.mode '{}' requires {}",
        effective.as_str(),
        deployment_mode.as_str(),
        deployment_mode.as_str()
    )))
}

pub(in crate::config) fn pool_search_can_generate_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    prefill_count: usize,
    decode_nodes: &[u32],
    decode_count: usize,
    deployment_mode: ServingDeploymentMode,
) -> bool {
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) {
                return true;
            }
        }
    }
    false
}

pub(in crate::config) fn pool_search_concrete_candidate_satisfies_constraints(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
    deployment_mode: ServingDeploymentMode,
) -> bool {
    if !pool_search.allow_overlap
        && deployment_mode != ServingDeploymentMode::Colocated
        && deployment_mode != ServingDeploymentMode::PartiallyDisaggregated
        && overlaps_nodes(prefill_nodes, decode_nodes)
    {
        return false;
    }

    deployment_mode.accepts_pool(prefill_nodes, decode_nodes)
        && pool_nodes_satisfy_domain_spread(
            cluster,
            &pool_search.domain_spread,
            prefill_nodes,
            decode_nodes,
        )
}

pub(in crate::config) fn pool_nodes_satisfy_domain_spread(
    cluster: &Cluster,
    domain_spread: &ServingPoolDomainSpread,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
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

pub(in crate::config) fn pool_nodes_meet_min_domain_count(
    cluster: &Cluster,
    node_ids: &[u32],
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

pub(in crate::config) fn pool_search_can_generate_routable_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
    counts: (usize, usize),
    gpu_labels: (&[String], &[String]),
    deployment_mode: ServingDeploymentMode,
) -> bool {
    let (prefill_count, decode_count) = counts;
    let (prefill_gpu_labels, decode_gpu_labels) = gpu_labels;
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) && serving_pool_route_is_routable(
                cluster,
                &prefill,
                prefill_gpu_labels,
                &decode,
                decode_gpu_labels,
            ) {
                return true;
            }
        }
    }
    false
}

pub(in crate::config) fn validate_serving_pool_route(
    name: &str,
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
) -> Result<(), ConfigError> {
    if serving_pool_route_is_routable(
        cluster,
        prefill_nodes,
        prefill_gpu_labels,
        decode_nodes,
        decode_gpu_labels,
    ) {
        return Ok(());
    }

    Err(ConfigError::new(format!(
        "{name} has no routable KV transfer path between prefill nodes {prefill_nodes:?} labels {prefill_gpu_labels:?} and decode nodes {decode_nodes:?} labels {decode_gpu_labels:?}; add missing custom inter-node links, choose connected prefill/decode pools, adjust GPU tag constraints, or disable serving.require_routable_pools for failure-scenario sweeps"
    )))
}

pub(in crate::config) fn serving_pool_route_is_routable(
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
) -> bool {
    let prefill_gpus =
        available_gpu_addrs_for_nodes_with_labels(cluster, prefill_nodes, prefill_gpu_labels);
    let decode_gpus =
        available_gpu_addrs_for_nodes_with_labels(cluster, decode_nodes, decode_gpu_labels);
    if prefill_gpus.is_empty() || decode_gpus.is_empty() {
        return true;
    }
    Solver::transfer_between_gpus_routable(
        cluster,
        &prefill_gpus,
        &decode_gpus,
        Bytes::from_bytes(1),
    )
}

#[derive(Default)]
pub(in crate::config) struct ServingRouteConstraintSummary {
    inter_node_pair_count: u32,
    routable_inter_node_path_count: u32,
    inter_node_route_resource_count: u32,
    unrailed_inter_node_route_resource_count: u32,
    host_staged_gpu_nic_resource_count: u32,
    rail_ids: BTreeSet<u32>,
}

pub(in crate::config) fn serving_pool_route_constraints_satisfied(
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
    constraints: ServingKvRouteConstraints,
    placements: ExplicitServingPlacements<'_>,
) -> Result<(), String> {
    let prefill_gpus = route_constraint_gpus_for_pool(
        cluster,
        prefill_nodes,
        prefill_gpu_labels,
        placements.prefill,
    );
    let decode_gpus =
        route_constraint_gpus_for_pool(cluster, decode_nodes, decode_gpu_labels, placements.decode);
    if prefill_gpus.is_empty() || decode_gpus.is_empty() {
        return Ok(());
    }

    let summary = serving_route_constraint_summary(cluster, &prefill_gpus, &decode_gpus)?;
    if summary.inter_node_route_resource_count > 0 {
        if let Some(limit) = constraints.min_inter_node_rail_count {
            let rail_count = summary.rail_ids.len().min(u32::MAX as usize) as u32;
            if rail_count < limit {
                return Err(format!(
                    "KV transfer routes expose {rail_count} inter-node rail(s), below configured serving.min_kv_route_rail_count {limit}"
                ));
            }
        }
        if constraints.require_inter_node_rail_metadata
            && summary.unrailed_inter_node_route_resource_count > 0
        {
            return Err(format!(
                "{} inter-node KV route resource(s) lack rail metadata",
                summary.unrailed_inter_node_route_resource_count
            ));
        }
    }
    if constraints.require_gpudirect && summary.host_staged_gpu_nic_resource_count > 0 {
        return Err(format!(
            "{} GPU/NIC KV route resource(s) are host-staged or marked no-GPUDirect",
            summary.host_staged_gpu_nic_resource_count
        ));
    }

    Ok(())
}

pub(in crate::config) fn route_constraint_gpus_for_pool(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
    placement: Option<&RankPlacement>,
) -> Vec<GpuAddr> {
    let Some(placement) = placement else {
        return available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels);
    };
    let node_ids = node_ids.iter().copied().collect::<BTreeSet<_>>();
    placement
        .rank_to_gpu
        .iter()
        .copied()
        .filter(|addr| {
            node_ids.contains(&addr.node_id)
                && cluster.node(addr.node_id).is_some_and(|node| {
                    gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
                })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(in crate::config) fn serving_route_constraint_summary(
    cluster: &Cluster,
    prefill_gpus: &[GpuAddr],
    decode_gpus: &[GpuAddr],
) -> Result<ServingRouteConstraintSummary, String> {
    let graph = TopologyGraph::from_cluster(cluster);
    let mut summary = ServingRouteConstraintSummary::default();
    let mut first_unroutable_path = None;
    for source in prefill_gpus {
        for destination in decode_gpus {
            if source == destination || source.node_id == destination.node_id {
                continue;
            }
            summary.inter_node_pair_count = summary.inter_node_pair_count.saturating_add(1);
            let Some(path) = graph.route_between_gpus(*source, *destination, Bytes::from_bytes(1))
            else {
                first_unroutable_path.get_or_insert_with(|| {
                    format!(
                        "no routable KV transfer path between node {} gpu {} and node {} gpu {}",
                        source.node_id,
                        source.local_gpu_id,
                        destination.node_id,
                        destination.local_gpu_id
                    )
                });
                continue;
            };
            summary.routable_inter_node_path_count =
                summary.routable_inter_node_path_count.saturating_add(1);
            for resource in path.resources {
                match resource.kind {
                    RoutedResourceKind::InterNodeFabric
                    | RoutedResourceKind::GpuScopedInterNodeFabric => {
                        summary.inter_node_route_resource_count =
                            summary.inter_node_route_resource_count.saturating_add(1);
                        if let Some(rail_id) = resource.rail_id {
                            summary.rail_ids.insert(rail_id);
                        } else {
                            summary.unrailed_inter_node_route_resource_count = summary
                                .unrailed_inter_node_route_resource_count
                                .saturating_add(1);
                        }
                    }
                    RoutedResourceKind::GpuNicLocal => {
                        if route_label_contains_any(
                            &resource.label,
                            &["host_staged", "no_gpudirect"],
                        ) {
                            summary.host_staged_gpu_nic_resource_count =
                                summary.host_staged_gpu_nic_resource_count.saturating_add(1);
                        }
                    }
                    RoutedResourceKind::IntraNodeFabric => {}
                }
            }
        }
    }
    if summary.inter_node_pair_count > 0 && summary.routable_inter_node_path_count == 0 {
        return Err(first_unroutable_path.unwrap_or_else(|| {
            "no routable inter-node KV transfer path between prefill and decode pools".to_string()
        }));
    }
    Ok(summary)
}

pub(in crate::config) fn route_label_contains_any(label: &str, needles: &[&str]) -> bool {
    let label = normalize(label);
    needles.iter().any(|needle| label.contains(needle))
}

pub(in crate::config) fn available_gpu_addrs_for_nodes_with_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> Vec<GpuAddr> {
    let mut addrs = Vec::new();
    for node_id in node_ids {
        let Some(node) = cluster.node(*node_id) else {
            continue;
        };
        let mut gpu_ids: Vec<_> = node.gpus.keys().copied().collect();
        gpu_ids.sort_unstable();
        addrs.extend(
            gpu_ids
                .into_iter()
                .map(|local_gpu_id| GpuAddr {
                    node_id: *node_id,
                    local_gpu_id,
                })
                .filter(|addr| {
                    cluster.is_gpu_available(*addr)
                        && gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
                }),
        );
    }
    addrs
}

pub(in crate::config) fn gpu_count_for_nodes_with_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> u32 {
    available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels)
        .len()
        .min(u32::MAX as usize) as u32
}

pub(in crate::config) fn gpu_count_for_nodes_with_labels_and_dtype(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
    dtype: DType,
) -> u32 {
    available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels)
        .into_iter()
        .filter(|addr| {
            cluster
                .gpu_profile(*addr)
                .is_some_and(|profile| gpu_profile_supports_dtype(&profile, dtype))
        })
        .count()
        .min(u32::MAX as usize) as u32
}

pub(in crate::config) fn nodes_with_gpu_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> Vec<u32> {
    if gpu_labels.is_empty() {
        return node_ids.to_vec();
    }
    node_ids
        .iter()
        .copied()
        .filter(|node_id| gpu_count_for_nodes_with_labels(cluster, &[*node_id], gpu_labels) > 0)
        .collect()
}

pub(in crate::config) fn nodes_matching_pool_node_filter(
    cluster: &Cluster,
    node_ids: &[u32],
    filter: &ServingPoolNodeFilter,
) -> Vec<u32> {
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

pub(in crate::config) fn node_matches_pool_node_filter(
    node: &Node,
    filter: &ServingPoolNodeFilter,
) -> bool {
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
}

pub(in crate::config) fn gpu_labels_match(
    labels: Option<&BTreeSet<String>>,
    required: &[String],
) -> bool {
    required.is_empty()
        || labels.is_some_and(|labels| required.iter().any(|label| labels.contains(label)))
}

pub(in crate::config) fn validate_no_duplicate_nodes(
    name: &str,
    nodes: &[u32],
) -> Result<(), ConfigError> {
    let mut seen = HashSet::new();
    for node in nodes {
        if !seen.insert(*node) {
            return Err(ConfigError::new(format!(
                "{name} contains duplicate node id {node}"
            )));
        }
    }
    Ok(())
}

pub(in crate::config) fn resolve_configured_pool_nodes(
    name: &str,
    cluster: &Cluster,
    explicit_nodes: &[u32],
    groups: &[String],
) -> Result<Vec<u32>, ConfigError> {
    let mut nodes = Vec::new();
    for node_id in explicit_nodes {
        if !cluster.nodes.contains_key(node_id) {
            return Err(ConfigError::new(format!(
                "{name}_nodes references unknown node id {node_id}"
            )));
        }
        nodes.push(*node_id);
    }
    for group in groups {
        nodes.extend_from_slice(&configured_group_nodes(
            &format!("{name}_groups"),
            cluster,
            group,
        )?);
    }
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Err(ConfigError::new(format!("{name} resolves to no nodes")));
    }
    Ok(nodes)
}

pub(in crate::config) fn configured_group_nodes(
    name: &str,
    cluster: &Cluster,
    group: &str,
) -> Result<Vec<u32>, ConfigError> {
    let nodes = cluster.node_group(group).ok_or_else(|| {
        ConfigError::new(format!("{name} references unknown node group '{group}'"))
    })?;
    if nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} node group '{group}' contains no nodes"
        )));
    }
    Ok(nodes.to_vec())
}

pub(in crate::config) fn valid_pool_search_counts(
    name: &str,
    configured: &[u32],
    available_nodes: usize,
) -> Result<Vec<u32>, ConfigError> {
    let mut counts = if configured.is_empty() {
        vec![available_nodes as u32]
    } else {
        configured.to_vec()
    };
    counts.sort_unstable();
    counts.dedup();
    let counts: Vec<_> = counts
        .into_iter()
        .filter(|count| *count > 0 && (*count as usize) <= available_nodes)
        .collect();
    if counts.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} has no value that fits {available_nodes} available nodes"
        )));
    }
    Ok(counts)
}

pub(in crate::config) fn combinations(values: &[u32], count: usize) -> Vec<Vec<u32>> {
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

pub(in crate::config) fn push_combinations(
    values: &[u32],
    count: usize,
    start: usize,
    current: &mut Vec<u32>,
    results: &mut Vec<Vec<u32>>,
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

pub(in crate::config) fn overlaps_nodes(left: &[u32], right: &[u32]) -> bool {
    left.iter().any(|node| right.contains(node))
}
