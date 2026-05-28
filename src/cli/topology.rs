use super::*;

const TEXT_CLUSTER_LINK_LIMIT: usize = 16;

#[derive(Clone, Debug, PartialEq)]
struct ClusterTopologyDiagnostic {
    severity: &'static str,
    code: &'static str,
    message: String,
    group: Option<String>,
    from_node: Option<u32>,
    to_node: Option<u32>,
    rail: Option<u32>,
    components: Vec<Vec<u32>>,
}

pub(super) fn write_markdown_cluster_inventory<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
) -> Result<(), io::Error> {
    write_markdown_key_value_section(
        writer,
        "Cluster",
        &[
            ("Nodes", cluster.nodes.len().to_string()),
            ("Total GPUs", cluster.total_gpus().to_string()),
            ("Available GPUs", cluster.available_gpus().to_string()),
            (
                "Disabled GPUs",
                cluster
                    .total_gpus()
                    .saturating_sub(cluster.available_gpus())
                    .to_string(),
            ),
            ("GPU Types", format_cluster_gpu_type_counts(cluster)),
            ("Node Groups", format_node_groups(cluster)),
            (
                "Interconnect",
                format_inter_node_topology_summary(&cluster.inter_node_topology),
            ),
        ],
    )?;

    writeln!(writer, "## Nodes\n")?;
    write_markdown_row(
        writer,
        &[
            "Node",
            "State",
            "Topology",
            "GPUs",
            "Available",
            "GPU Types",
            "GPU Labels",
            "Intra-node",
            "NICs",
            "Active NICs",
            "Rails",
            "Affinity",
        ],
    )?;
    write_markdown_separator(writer, 12)?;
    let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
    node_ids.sort_unstable();
    for node_id in node_ids {
        let Some(node) = cluster.node(node_id) else {
            continue;
        };
        write_markdown_row(
            writer,
            &[
                node_id.to_string(),
                node.operational_state.as_str().to_string(),
                format_node_topology_metadata(node),
                node.gpus.len().to_string(),
                node.available_gpu_count().to_string(),
                format_node_gpu_type_counts(cluster, node_id),
                format_gpu_labels(node),
                format_intra_node_topology(&node.intra_node_fabric),
                format_nic_capacity(&node.network),
                node.network.active_nic_count().to_string(),
                node.network.rail_count.to_string(),
                format_gpu_nic_affinity(node.network.gpu_to_nic),
            ],
        )?;
    }
    writeln!(writer)?;

    let diagnostics = cluster_topology_diagnostics(cluster);
    if diagnostics.is_empty() {
        return Ok(());
    }
    writeln!(writer, "## Topology Diagnostics\n")?;
    write_markdown_row(
        writer,
        &[
            "Severity",
            "Code",
            "Group",
            "From",
            "To",
            "Rail",
            "Components",
            "Message",
        ],
    )?;
    write_markdown_separator(writer, 8)?;
    for diagnostic in diagnostics {
        write_markdown_row(
            writer,
            &[
                diagnostic.severity.to_string(),
                diagnostic.code.to_string(),
                diagnostic.group.unwrap_or_else(|| "none".to_string()),
                diagnostic
                    .from_node
                    .map(|node| node.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                diagnostic
                    .to_node
                    .map(|node| node.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                format_optional_rail(diagnostic.rail),
                format_topology_components(&diagnostic.components),
                diagnostic.message,
            ],
        )?;
    }
    writeln!(writer)
}

pub(super) fn write_cluster_inventory_text<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "cluster_inventory nodes={} total_gpus={} available_gpus={} disabled_gpus={} gpu_types={} node_groups={} interconnect={}",
        cluster.nodes.len(),
        cluster.total_gpus(),
        cluster.available_gpus(),
        cluster
            .total_gpus()
            .saturating_sub(cluster.available_gpus()),
        format_cluster_gpu_type_counts(cluster),
        format_node_groups(cluster),
        format_inter_node_topology_summary(&cluster.inter_node_topology)
    )?;

    let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
    node_ids.sort_unstable();
    for node_id in node_ids {
        let Some(node) = cluster.node(node_id) else {
            continue;
        };
        writeln!(
            writer,
            "node_inventory node={} state={} topology={} gpus={} available_gpus={} disabled_gpus={} gpu_states={} gpu_types={} gpu_labels={} intra={} nics={} active_nics={} disabled_nics={} nic_states={} rails={} nic_rail_map={} affinity={} gpu_nic_map={} gpu_numa_map={} nic_numa_map={} cross_numa_bandwidth_scale={:.3} cross_numa_latency_scale={:.3} gpu_nic_paths={}",
            node_id,
            node.operational_state.as_str(),
            format_node_topology_metadata(node),
            node.gpus.len(),
            node.available_gpu_count(),
            format_id_set(&node.disabled_gpus),
            format_gpu_operational_states(node),
            format_node_gpu_type_counts(cluster, node_id),
            format_gpu_labels(node),
            format_intra_node_topology(&node.intra_node_fabric),
            format_nic_capacity(&node.network),
            node.network.active_nic_count(),
            format_id_set(&node.network.disabled_nics),
            format_nic_operational_states(&node.network),
            node.network.rail_count,
            format_nic_rail_map(&node.network),
            format_gpu_nic_affinity(node.network.gpu_to_nic),
            format_gpu_nic_map(&node.network),
            format_gpu_numa_map(&node.network),
            format_nic_numa_map(&node.network),
            node.network.cross_numa_bandwidth_scale,
            node.network.cross_numa_latency_scale,
            format_gpu_nic_paths(&node.network)
        )?;
    }

    write_inter_node_topology_text(writer, &cluster.inter_node_topology)?;
    write_topology_diagnostics_text(writer, &cluster_topology_diagnostics(cluster))
}

fn write_inter_node_topology_text<W: Write>(
    writer: &mut W,
    topology: &InterNodeTopology,
) -> Result<(), io::Error> {
    if let InterNodeTopology::Custom(links) = topology {
        let mut custom_links = Vec::new();
        for (pair, profiles) in links {
            let (from, to) = pair.endpoints();
            for link in profiles {
                custom_links.push((*from, *to, link));
            }
        }
        custom_links.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.cmp(&right.1))
                .then(left.2.rail.cmp(&right.2.rail))
                .then(left.2.profile.label.cmp(right.2.profile.label))
        });
        for (idx, (from, to, link)) in custom_links.iter().enumerate() {
            if idx >= TEXT_CLUSTER_LINK_LIMIT {
                writeln!(
                    writer,
                    "interconnect_links_truncated remaining={}",
                    custom_links.len() - TEXT_CLUSTER_LINK_LIMIT
                )?;
                break;
            }
            let (from_gpus, to_gpus) = custom_link_gpu_scope(link, *from, *to);
            writeln!(
                writer,
                "interconnect_link from={} to={} rail={} from_gpus={} to_gpus={} fabric={} kind={} bandwidth_gbps={:.3} latency_us={:.3}",
                from,
                to,
                format_optional_rail(link.rail),
                text_u32_list(&from_gpus),
                text_u32_list(&to_gpus),
                link.profile.label,
                fabric_kind_label(link.profile.kind),
                link.profile.bw.unidirectional.as_gigabits_per_sec(),
                link.profile.latency.to_us()
            )?;
        }
    }

    Ok(())
}

fn write_topology_diagnostics_text<W: Write>(
    writer: &mut W,
    diagnostics: &[ClusterTopologyDiagnostic],
) -> Result<(), io::Error> {
    for diagnostic in diagnostics {
        writeln!(
            writer,
            "topology_diagnostic severity={} code={} group={} from={} to={} rail={} components={} message={}",
            diagnostic.severity,
            diagnostic.code,
            diagnostic.group.as_deref().unwrap_or("none"),
            diagnostic
                .from_node
                .map(|node| node.to_string())
                .unwrap_or_else(|| "none".to_string()),
            diagnostic
                .to_node
                .map(|node| node.to_string())
                .unwrap_or_else(|| "none".to_string()),
            format_optional_rail(diagnostic.rail),
            format_topology_components(&diagnostic.components),
            diagnostic.message
        )?;
    }

    Ok(())
}

fn cluster_topology_diagnostics(cluster: &Cluster) -> Vec<ClusterTopologyDiagnostic> {
    let mut diagnostics = gpu_nic_path_diagnostics(cluster);
    let InterNodeTopology::Custom(links) = &cluster.inter_node_topology else {
        return diagnostics;
    };

    diagnostics.extend(unusable_custom_link_diagnostics(cluster, links));

    let components = topology_connected_components(cluster);
    if components.len() > 1 {
        diagnostics.push(ClusterTopologyDiagnostic {
            severity: "warn",
            code: "disconnected_topology_islands",
            message: format!(
                "custom inter-node topology has {} disconnected islands; cross-island collectives, placements, or prefill/decode routes cannot be modeled as routable",
                components.len()
            ),
            group: None,
            from_node: None,
            to_node: None,
            rail: None,
            components: components.clone(),
        });
    }

    let component_index = component_index_map(&components);
    let mut groups: Vec<_> = cluster.node_groups.iter().collect();
    groups.sort_by_key(|(label, _)| *label);
    for (label, nodes) in groups {
        if label == "all" {
            continue;
        }
        let spanning_components: BTreeSet<_> = nodes
            .iter()
            .filter_map(|node_id| component_index.get(node_id).copied())
            .collect();
        if spanning_components.len() > 1 {
            diagnostics.push(ClusterTopologyDiagnostic {
                severity: "warn",
                code: "node_group_spans_disconnected_islands",
                message: format!(
                    "node group '{label}' spans {} disconnected topology islands; pool search or placement constrained to this group may produce unroutable candidates",
                    spanning_components.len()
                ),
                group: Some(label.clone()),
                from_node: None,
                to_node: None,
                rail: None,
                components: components_for_indexes(&components, &spanning_components),
            });
        }
    }

    diagnostics
}

fn gpu_nic_path_diagnostics(cluster: &Cluster) -> Vec<ClusterTopologyDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
    node_ids.sort_unstable();

    for node_id in node_ids {
        let Some(node) = cluster.node(node_id) else {
            continue;
        };
        for ((gpu_id, nic_id), path) in &node.network.gpu_nic_path_overrides {
            let rail = Some(node.network.rail_id(*nic_id));
            let label = path.label.as_deref().unwrap_or("unlabeled");
            let endpoint = format!("node {node_id} GPU {gpu_id}->NIC {nic_id} ({label})");
            if node.disabled_gpus.contains(gpu_id) {
                diagnostics.push(ClusterTopologyDiagnostic {
                    severity: "info",
                    code: "gpu_nic_path_targets_disabled_gpu",
                    message: format!(
                        "{endpoint} is configured for a disabled GPU; it will not be used by placements or routes"
                    ),
                    group: None,
                    from_node: Some(node_id),
                    to_node: None,
                    rail,
                    components: Vec::new(),
                });
            }
            if node.network.disabled_nics.contains(nic_id) {
                diagnostics.push(ClusterTopologyDiagnostic {
                    severity: "warn",
                    code: "gpu_nic_path_targets_disabled_nic",
                    message: format!(
                        "{endpoint} targets a disabled NIC; routes using this locality are unavailable"
                    ),
                    group: None,
                    from_node: Some(node_id),
                    to_node: None,
                    rail,
                    components: Vec::new(),
                });
            }
            if !path.available {
                diagnostics.push(ClusterTopologyDiagnostic {
                    severity: "warn",
                    code: "gpu_nic_path_unavailable",
                    message: format!(
                        "{endpoint} is configured unavailable; routes using this locality are impossible"
                    ),
                    group: None,
                    from_node: Some(node_id),
                    to_node: None,
                    rail,
                    components: Vec::new(),
                });
            }
            if path.gpudirect == Some(false) {
                diagnostics.push(ClusterTopologyDiagnostic {
                    severity: "warn",
                    code: "gpu_nic_path_host_staged",
                    message: format!(
                        "{endpoint} has GPUDirect disabled; KV transfer paths through it require host staging unless another route is selected"
                    ),
                    group: None,
                    from_node: Some(node_id),
                    to_node: None,
                    rail,
                    components: Vec::new(),
                });
            }
            if let Some(bandwidth) = path.bandwidth {
                let path_gbps = bandwidth.as_gigabits_per_sec();
                let default_gbps = node.network.nic_bandwidth.as_gigabits_per_sec();
                if path_gbps < default_gbps {
                    diagnostics.push(ClusterTopologyDiagnostic {
                        severity: if path_gbps * 2.0 <= default_gbps {
                            "warn"
                        } else {
                            "info"
                        },
                        code: "gpu_nic_path_bandwidth_below_nic",
                        message: format!(
                            "{endpoint} bandwidth {path_gbps:.3}Gbps is below node NIC default {default_gbps:.3}Gbps; routing and KV-transfer estimates depend on this locality override"
                        ),
                        group: None,
                        from_node: Some(node_id),
                        to_node: None,
                        rail,
                        components: Vec::new(),
                    });
                }
            }
        }
    }

    diagnostics
}

fn unusable_custom_link_diagnostics(
    cluster: &Cluster,
    links: &HashMap<UnorderedPair<u32>, Vec<CustomInterNodeLink>>,
) -> Vec<ClusterTopologyDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut entries = Vec::new();
    for (pair, pair_links) in links {
        let (from, to) = pair.endpoints();
        for link in pair_links {
            entries.push((*from, *to, link.rail, link.profile.label));
        }
    }
    entries.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
            .then(left.3.cmp(right.3))
    });

    for (from, to, rail, fabric_label) in entries {
        let Some(rail) = rail else {
            continue;
        };
        let Some(from_node) = cluster.node(from) else {
            continue;
        };
        let Some(to_node) = cluster.node(to) else {
            continue;
        };
        let rail_out_of_range = rail >= u32::from(from_node.network.rail_count)
            || rail >= u32::from(to_node.network.rail_count);
        if rail_out_of_range {
            diagnostics.push(ClusterTopologyDiagnostic {
                severity: "warn",
                code: "custom_link_rail_unusable",
                message: format!(
                    "custom link {from}<->{to} on rail {rail} ({fabric_label}) cannot be used because endpoint rail counts are {} and {}",
                    from_node.network.rail_count, to_node.network.rail_count
                ),
                group: None,
                from_node: Some(from),
                to_node: Some(to),
                rail: Some(rail),
                components: Vec::new(),
            });
            continue;
        }
        if !from_node.network.has_active_rail(rail) || !to_node.network.has_active_rail(rail) {
            diagnostics.push(ClusterTopologyDiagnostic {
                severity: "warn",
                code: "custom_link_rail_unusable",
                message: format!(
                    "custom link {from}<->{to} on rail {rail} ({fabric_label}) cannot be used because active endpoint rails are [{}] and [{}]",
                    text_u32_list(&from_node.network.active_rails().into_iter().collect::<Vec<_>>()),
                    text_u32_list(&to_node.network.active_rails().into_iter().collect::<Vec<_>>())
                ),
                group: None,
                from_node: Some(from),
                to_node: Some(to),
                rail: Some(rail),
                components: Vec::new(),
            });
        }
    }

    diagnostics
}

fn topology_connected_components(cluster: &Cluster) -> Vec<Vec<u32>> {
    let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
    node_ids.sort_unstable();
    let graph = TopologyGraph::from_cluster(cluster);
    let mut visited = BTreeSet::new();
    let mut components = Vec::new();

    for node_id in &node_ids {
        if visited.contains(node_id) {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![*node_id];
        visited.insert(*node_id);
        while let Some(current) = stack.pop() {
            component.push(current);
            for next in &node_ids {
                if visited.contains(next) || *next == current {
                    continue;
                }
                if graph
                    .route_between_nodes(current, *next, Bytes::from_bytes(1))
                    .is_some()
                {
                    visited.insert(*next);
                    stack.push(*next);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort_by(|left, right| left.first().cmp(&right.first()));
    components
}

fn component_index_map(components: &[Vec<u32>]) -> BTreeMap<u32, usize> {
    let mut map = BTreeMap::new();
    for (idx, component) in components.iter().enumerate() {
        for node_id in component {
            map.insert(*node_id, idx);
        }
    }
    map
}

fn components_for_indexes(components: &[Vec<u32>], indexes: &BTreeSet<usize>) -> Vec<Vec<u32>> {
    indexes
        .iter()
        .filter_map(|idx| components.get(*idx).cloned())
        .collect()
}

fn format_cluster_gpu_type_counts(cluster: &Cluster) -> String {
    let counts = cluster_gpu_type_counts(cluster);
    if counts.is_empty() {
        return "none".to_string();
    }
    counts
        .iter()
        .map(|(label, entry)| format!("{label}:{}", entry.count))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_node_gpu_type_counts(cluster: &Cluster, node_id: u32) -> String {
    let counts = node_gpu_type_counts(cluster, node_id);
    if counts.is_empty() {
        return "none".to_string();
    }
    counts
        .iter()
        .map(|(label, count)| format!("{label}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_gpu_labels(node: &crate::types::topology::Node) -> String {
    if node.gpu_labels.is_empty() {
        return "none".to_string();
    }
    let mut entries: Vec<_> = node.gpu_labels.iter().collect();
    entries.sort_by_key(|(gpu_id, _)| *gpu_id);
    entries
        .into_iter()
        .map(|(gpu_id, labels)| {
            format!(
                "{}:[{}]",
                gpu_id,
                labels.iter().cloned().collect::<Vec<_>>().join("|")
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn format_gpu_operational_states(node: &crate::types::topology::Node) -> String {
    let mut entries = node
        .gpus
        .keys()
        .copied()
        .filter_map(|gpu_id| {
            let state = node.gpu_operational_state(gpu_id);
            state_needs_inventory(state).then_some((gpu_id, state))
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|(gpu_id, _)| *gpu_id);
    if entries.is_empty() {
        return "all_healthy".to_string();
    }
    entries
        .into_iter()
        .map(|(gpu_id, state)| format!("{gpu_id}:{}", state.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_nic_operational_states(
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
) -> String {
    let mut entries = (0..u32::from(network.nic_count))
        .filter_map(|nic_id| {
            let state = network.nic_operational_state(nic_id);
            state_needs_inventory(state).then_some((nic_id, state))
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|(nic_id, _)| *nic_id);
    if entries.is_empty() {
        return "all_healthy".to_string();
    }
    entries
        .into_iter()
        .map(|(nic_id, state)| format!("{nic_id}:{}", state.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

fn state_needs_inventory(state: OperationalState) -> bool {
    !matches!(state, OperationalState::Healthy)
}

fn format_node_topology_metadata(node: &crate::types::topology::Node) -> String {
    let mut parts = Vec::new();
    if let Some(rack) = &node.topology.rack {
        parts.push(format!("rack={rack}"));
    }
    if let Some(island) = &node.topology.island {
        parts.push(format!("island={island}"));
    }
    if let Some(failure_domain) = &node.topology.failure_domain {
        parts.push(format!("failure_domain={failure_domain}"));
    }
    if !node.topology.labels.is_empty() {
        parts.push(format!(
            "labels=[{}]",
            node.topology
                .labels
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("|")
        ));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(",")
    }
}

fn format_node_groups(cluster: &Cluster) -> String {
    if cluster.node_groups.is_empty() {
        return "none".to_string();
    }
    let mut groups: Vec<_> = cluster.node_groups.iter().collect();
    groups.sort_by_key(|(label, _)| *label);
    groups
        .iter()
        .map(|(label, nodes)| {
            let mut nodes = (*nodes).clone();
            nodes.sort_unstable();
            nodes.dedup();
            format!(
                "{}:[{}]",
                label,
                nodes
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("|")
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn format_inter_node_topology_summary(topology: &InterNodeTopology) -> String {
    match topology {
        InterNodeTopology::FatTree {
            link,
            oversubscription,
            leaf_size,
        } => format!(
            "fat_tree:{}:{:.3}Gbps:{:.3}us:oversub={:.3}:leaf={}",
            link.label,
            link.bw.unidirectional.as_gigabits_per_sec(),
            link.latency.to_us(),
            oversubscription,
            leaf_size
        ),
        InterNodeTopology::Flat { link } => format!(
            "flat:{}:{:.3}Gbps:{:.3}us",
            link.label,
            link.bw.unidirectional.as_gigabits_per_sec(),
            link.latency.to_us()
        ),
        InterNodeTopology::Custom(links) => {
            let link_count: usize = links.values().map(Vec::len).sum();
            format!("custom:links={link_count}")
        }
    }
}

fn format_intra_node_topology(topology: &IntraNodeTopology) -> String {
    match topology {
        IntraNodeTopology::NvSwitch(profile) => format!(
            "nvswitch:{}:{:.3}Gbps:{:.3}us",
            profile.label,
            profile.bw.unidirectional.as_gigabits_per_sec(),
            profile.latency.to_us()
        ),
        IntraNodeTopology::NvLinkDirect(profile) => format!(
            "nvlink_direct:{}:{:.3}Gbps:{:.3}us",
            profile.label,
            profile.bw.unidirectional.as_gigabits_per_sec(),
            profile.latency.to_us()
        ),
        IntraNodeTopology::Pcie(profile) => format!(
            "pcie:{}:{:.3}Gbps:{:.3}us",
            profile.label,
            profile.bw.unidirectional.as_gigabits_per_sec(),
            profile.latency.to_us()
        ),
        IntraNodeTopology::Custom(links) => format!("custom:links={}", links.len()),
    }
}

fn format_nic_capacity(network: &crate::types::fabric::intra_node::NodeNetworkProfile) -> String {
    let base = format!(
        "{}x{:.3}Gbps",
        network.nic_count,
        network.nic_bandwidth.as_gigabits_per_sec()
    );
    if network.nic_bandwidth_overrides.is_empty() && network.nic_latency_scale_overrides.is_empty()
    {
        return base;
    }
    let mut parts = Vec::new();
    if !network.nic_bandwidth_overrides.is_empty() {
        let overrides = network
            .nic_bandwidth_overrides
            .iter()
            .map(|(nic_id, bandwidth)| {
                format!("{}:{:.3}Gbps", nic_id, bandwidth.as_gigabits_per_sec())
            })
            .collect::<Vec<_>>()
            .join("|");
        parts.push(format!("bandwidth_overrides={overrides}"));
    }
    if !network.nic_latency_scale_overrides.is_empty() {
        let overrides = network
            .nic_latency_scale_overrides
            .iter()
            .map(|(nic_id, scale)| format!("{nic_id}:{scale:.3}x"))
            .collect::<Vec<_>>()
            .join("|");
        parts.push(format!("latency_scale_overrides={overrides}"));
    }
    format!("{base}:{}", parts.join(":"))
}

fn format_gpu_nic_affinity(affinity: GpuNicAffinity) -> String {
    match affinity {
        GpuNicAffinity::Dedicated => "dedicated".to_string(),
        GpuNicAffinity::Uniform => "uniform".to_string(),
        GpuNicAffinity::Shared { gpus_per_nic } => format!("shared:{gpus_per_nic}gpus_per_nic"),
    }
}

fn format_nic_rail_map(network: &crate::types::fabric::intra_node::NodeNetworkProfile) -> String {
    if network.nic_rail_map.is_empty() {
        return "default".to_string();
    }

    network
        .nic_rail_map
        .iter()
        .map(|(nic_id, rail_id)| format!("{nic_id}:{rail_id}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_gpu_nic_map(network: &crate::types::fabric::intra_node::NodeNetworkProfile) -> String {
    if network.gpu_nic_map.is_empty() {
        return "default".to_string();
    }

    network
        .gpu_nic_map
        .iter()
        .map(|(gpu_id, nic_ids)| {
            format!(
                "{}:{}",
                gpu_id,
                nic_ids
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("|")
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn format_gpu_numa_map(network: &crate::types::fabric::intra_node::NodeNetworkProfile) -> String {
    if network.gpu_numa_map.is_empty() {
        return "unspecified".to_string();
    }

    network
        .gpu_numa_map
        .iter()
        .map(|(gpu_id, domain)| format!("{gpu_id}:{domain}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_nic_numa_map(network: &crate::types::fabric::intra_node::NodeNetworkProfile) -> String {
    if network.nic_numa_map.is_empty() {
        return "unspecified".to_string();
    }

    network
        .nic_numa_map
        .iter()
        .map(|(nic_id, domain)| format!("{nic_id}:{domain}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_gpu_nic_paths(network: &crate::types::fabric::intra_node::NodeNetworkProfile) -> String {
    if network.gpu_nic_path_overrides.is_empty() {
        return "default".to_string();
    }

    network
        .gpu_nic_path_overrides
        .iter()
        .map(|((gpu_id, nic_id), path)| {
            let bandwidth = path
                .bandwidth
                .map(|bandwidth| format!("{:.3}Gbps", bandwidth.as_gigabits_per_sec()))
                .unwrap_or_else(|| "-".to_string());
            let latency = path
                .latency
                .map(|latency| format!("{:.3}us", latency.to_us()))
                .unwrap_or_else(|| "-".to_string());
            let gpudirect = path
                .gpudirect
                .map(|gpudirect| gpudirect.to_string())
                .unwrap_or_else(|| "-".to_string());
            format!(
                "{}:{}:{}:{}:{}:{}:{}",
                gpu_id,
                nic_id,
                path.label.as_deref().unwrap_or("-"),
                bandwidth,
                latency,
                gpudirect,
                path.available
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn format_id_set(values: &BTreeSet<u32>) -> String {
    if values.is_empty() {
        return "none".to_string();
    }

    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn format_optional_rail(rail: Option<u32>) -> String {
    rail.map(|rail| rail.to_string())
        .unwrap_or_else(|| "all".to_string())
}

fn custom_link_gpu_scope(link: &CustomInterNodeLink, from: u32, to: u32) -> (Vec<u32>, Vec<u32>) {
    let Some(endpoints) = &link.endpoints else {
        return (Vec::new(), Vec::new());
    };
    if endpoints.from_node == from && endpoints.to_node == to {
        (endpoints.from_gpus.clone(), endpoints.to_gpus.clone())
    } else if endpoints.from_node == to && endpoints.to_node == from {
        (endpoints.to_gpus.clone(), endpoints.from_gpus.clone())
    } else {
        (Vec::new(), Vec::new())
    }
}

fn format_topology_components(components: &[Vec<u32>]) -> String {
    if components.is_empty() {
        return "none".to_string();
    }
    components
        .iter()
        .map(|component| {
            format!(
                "[{}]",
                component
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("|")
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

#[derive(Clone, Debug)]
struct GpuInventoryEntry {
    count: u32,
    profile: GpuProfile,
}

pub(super) fn write_cluster_inventory<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"cluster_inventory\": {{")?;
    writeln!(writer, "{indent}  \"node_count\": {},", cluster.nodes.len())?;
    writeln!(
        writer,
        "{indent}  \"total_gpus\": {},",
        cluster.total_gpus()
    )?;
    writeln!(
        writer,
        "{indent}  \"available_gpus\": {},",
        cluster.available_gpus()
    )?;
    writeln!(
        writer,
        "{indent}  \"disabled_gpus\": {},",
        cluster
            .total_gpus()
            .saturating_sub(cluster.available_gpus())
    )?;
    write_cluster_gpu_types(writer, cluster, indent, true)?;
    write_cluster_nodes(writer, cluster, indent, true)?;
    write_cluster_node_groups(writer, cluster, indent, true)?;
    write_inter_node_topology(writer, cluster, indent, true)?;
    write_cluster_topology_diagnostics(writer, cluster, indent, false)?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}

fn write_cluster_gpu_types<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let gpu_types = cluster_gpu_type_counts(cluster);
    writeln!(writer, "{indent}  \"gpu_types\": [")?;
    for (idx, (label, entry)) in gpu_types.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(writer, "{indent}      \"gpu\": {},", json_string(label))?;
        writeln!(writer, "{indent}      \"count\": {},", entry.count)?;
        writeln!(
            writer,
            "{indent}      \"hbm_gb\": {},",
            json_f64(entry.profile.hbm_size.as_gigabytes())
        )?;
        writeln!(
            writer,
            "{indent}      \"hbm_bandwidth_gb_s\": {},",
            json_f64(entry.profile.hbm_bandwidth.as_gigabytes_per_sec())
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_f16_tflops\": {},",
            json_f64(entry.profile.peak_f16_flops)
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_f8_tflops\": {}",
            json_optional_value(entry.profile.peak_f8_flops)
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < gpu_types.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_cluster_nodes<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
    node_ids.sort_unstable();

    writeln!(writer, "{indent}  \"nodes\": [")?;
    for (idx, node_id) in node_ids.iter().enumerate() {
        let node = cluster
            .nodes
            .get(node_id)
            .expect("node ids are collected from cluster.nodes");
        writeln!(writer, "{indent}    {{")?;
        writeln!(writer, "{indent}      \"node_id\": {node_id},")?;
        writeln!(
            writer,
            "{indent}      \"operational_state\": {},",
            json_string(node.operational_state.as_str())
        )?;
        write_node_topology_metadata(writer, node, indent, true)?;
        writeln!(writer, "{indent}      \"gpu_count\": {},", node.gpus.len())?;
        writeln!(
            writer,
            "{indent}      \"available_gpu_count\": {},",
            node.available_gpu_count()
        )?;
        write_id_set_array(
            writer,
            &format!("{indent}      "),
            "disabled_gpus",
            &node.disabled_gpus,
            true,
        )?;
        write_node_gpu_types(writer, cluster, *node_id, indent, true)?;
        write_node_gpu_inventory(writer, cluster, *node_id, indent, true)?;
        write_intra_node_topology(writer, &node.intra_node_fabric, indent, true)?;
        write_node_nics(writer, &node.network, indent, false)?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < node_ids.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_node_topology_metadata<W: Write>(
    writer: &mut W,
    node: &crate::types::topology::Node,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let labels: Vec<_> = node.topology.labels.iter().cloned().collect();
    writeln!(writer, "{indent}      \"topology\": {{")?;
    writeln!(
        writer,
        "{indent}        \"rack\": {},",
        json_optional_string(node.topology.rack.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}        \"island\": {},",
        json_optional_string(node.topology.island.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}        \"failure_domain\": {},",
        json_optional_string(node.topology.failure_domain.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}        \"labels\": [{}]",
        string_list(&labels)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_node_gpu_types<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    node_id: u32,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let gpu_types = node_gpu_type_counts(cluster, node_id);
    writeln!(writer, "{indent}      \"gpu_types\": [")?;
    for (idx, (label, count)) in gpu_types.iter().enumerate() {
        writeln!(
            writer,
            "{indent}        {{ \"gpu\": {}, \"count\": {} }}{}",
            json_string(label),
            count,
            comma(idx + 1 < gpu_types.len())
        )?;
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_node_gpu_inventory<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    node_id: u32,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(node) = cluster.node(node_id) else {
        writeln!(
            writer,
            "{indent}      \"gpus\": []{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    let mut gpu_ids: Vec<_> = node.gpus.keys().copied().collect();
    gpu_ids.sort_unstable();
    writeln!(writer, "{indent}      \"gpus\": [")?;
    for (idx, local_gpu_id) in gpu_ids.iter().enumerate() {
        let Some(profile) = node.gpu_profile(*local_gpu_id) else {
            continue;
        };
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"local_gpu_id\": {},",
            local_gpu_id
        )?;
        writeln!(
            writer,
            "{indent}          \"gpu\": {},",
            json_string(profile.label)
        )?;
        writeln!(
            writer,
            "{indent}          \"available\": {},",
            node.is_gpu_available(*local_gpu_id)
        )?;
        writeln!(
            writer,
            "{indent}          \"operational_state\": {},",
            json_string(node.gpu_operational_state(*local_gpu_id).as_str())
        )?;
        writeln!(
            writer,
            "{indent}          \"profile_overridden\": {},",
            node.gpu_profile_overrides.contains_key(local_gpu_id)
        )?;
        writeln!(
            writer,
            "{indent}          \"hbm_gb\": {},",
            json_f64(profile.hbm_size.as_gigabytes())
        )?;
        writeln!(
            writer,
            "{indent}          \"hbm_bandwidth_gb_s\": {},",
            json_f64(profile.hbm_bandwidth.as_gigabytes_per_sec())
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_f16_tflops\": {},",
            json_f64(profile.peak_f16_flops)
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_f8_tflops\": {},",
            json_optional_value(profile.peak_f8_flops)
        )?;
        let gpu_labels = node
            .gpu_labels(*local_gpu_id)
            .map(|labels| labels.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        writeln!(
            writer,
            "{indent}          \"labels\": [{}],",
            string_list(&gpu_labels)
        )?;
        write_nic_id_array(
            writer,
            &format!("{indent}          "),
            "nic_ids",
            &node.network.nic_candidates_for_gpu(*local_gpu_id),
            true,
        )?;
        write_rail_id_array(
            writer,
            &format!("{indent}          "),
            "rail_ids",
            &node.network,
            *local_gpu_id,
            false,
        )?;
        writeln!(
            writer,
            "{indent}        }}{}",
            comma(idx + 1 < gpu_ids.len())
        )?;
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_intra_node_topology<W: Write>(
    writer: &mut W,
    topology: &IntraNodeTopology,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"intra_node_topology\": {{")?;
    match topology {
        IntraNodeTopology::NvSwitch(profile) => {
            writeln!(writer, "{indent}        \"kind\": \"nvswitch\",")?;
            writeln!(
                writer,
                "{indent}        \"label\": {},",
                json_string(profile.label)
            )?;
            writeln!(
                writer,
                "{indent}        \"bandwidth_gbps\": {},",
                json_f64(profile.bw.unidirectional.as_gigabits_per_sec())
            )?;
            writeln!(
                writer,
                "{indent}        \"latency_us\": {}",
                json_f64(profile.latency.to_us())
            )?;
        }
        IntraNodeTopology::NvLinkDirect(profile) => {
            writeln!(writer, "{indent}        \"kind\": \"nvlink_direct\",")?;
            writeln!(
                writer,
                "{indent}        \"label\": {},",
                json_string(profile.label)
            )?;
            writeln!(
                writer,
                "{indent}        \"bandwidth_gbps\": {},",
                json_f64(profile.bw.unidirectional.as_gigabits_per_sec())
            )?;
            writeln!(
                writer,
                "{indent}        \"latency_us\": {}",
                json_f64(profile.latency.to_us())
            )?;
        }
        IntraNodeTopology::Pcie(profile) => {
            writeln!(writer, "{indent}        \"kind\": \"pcie\",")?;
            writeln!(
                writer,
                "{indent}        \"label\": {},",
                json_string(profile.label)
            )?;
            writeln!(
                writer,
                "{indent}        \"bandwidth_gbps\": {},",
                json_f64(profile.bw.unidirectional.as_gigabits_per_sec())
            )?;
            writeln!(
                writer,
                "{indent}        \"latency_us\": {}",
                json_f64(profile.latency.to_us())
            )?;
        }
        IntraNodeTopology::Custom(links) => {
            writeln!(writer, "{indent}        \"kind\": \"custom\",")?;
            writeln!(writer, "{indent}        \"link_count\": {}", links.len())?;
        }
    }
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_node_nics<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"nics\": {{")?;
    writeln!(writer, "{indent}        \"count\": {},", network.nic_count)?;
    writeln!(
        writer,
        "{indent}        \"active_count\": {},",
        network.active_nic_count()
    )?;
    write_id_set_array(
        writer,
        &format!("{indent}        "),
        "disabled_nics",
        &network.disabled_nics,
        true,
    )?;
    write_nic_operational_states_json(writer, network, &format!("{indent}        "), true)?;
    writeln!(
        writer,
        "{indent}        \"bandwidth_gbps\": {},",
        json_f64(network.nic_bandwidth.as_gigabits_per_sec())
    )?;
    write_nic_bandwidth_overrides_json(writer, network, &format!("{indent}        "), true)?;
    write_nic_latency_scale_overrides_json(writer, network, &format!("{indent}        "), true)?;
    writeln!(
        writer,
        "{indent}        \"rail_count\": {},",
        network.rail_count
    )?;
    write_nic_rail_map_json(writer, network, &format!("{indent}        "), true)?;
    writeln!(
        writer,
        "{indent}        \"affinity\": {},",
        json_string(gpu_nic_affinity_label(network.gpu_to_nic))
    )?;
    writeln!(
        writer,
        "{indent}        \"gpus_per_nic\": {},",
        json_optional_u32(gpu_nic_gpus_per_nic(network.gpu_to_nic))
    )?;
    write_gpu_nic_map_json(writer, network, &format!("{indent}        "), true)?;
    write_gpu_numa_map_json(writer, network, &format!("{indent}        "), true)?;
    write_nic_numa_map_json(writer, network, &format!("{indent}        "), true)?;
    writeln!(
        writer,
        "{indent}        \"cross_numa_bandwidth_scale\": {},",
        json_f64(network.cross_numa_bandwidth_scale)
    )?;
    writeln!(
        writer,
        "{indent}        \"cross_numa_latency_scale\": {},",
        json_f64(network.cross_numa_latency_scale)
    )?;
    write_gpu_nic_paths_json(writer, network, &format!("{indent}        "), false)?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_nic_operational_states_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"nic_states\": [")?;
    for nic_id in 0..u32::from(network.nic_count) {
        let state = network.nic_operational_state(nic_id);
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"nic_id\": {},", nic_id)?;
        writeln!(
            writer,
            "{indent}    \"operational_state\": {},",
            json_string(state.as_str())
        )?;
        writeln!(
            writer,
            "{indent}    \"available\": {}",
            network.is_nic_available(nic_id)
        )?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(nic_id + 1 < u32::from(network.nic_count))
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_nic_bandwidth_overrides_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"bandwidth_overrides\": [")?;
    for (idx, (nic_id, bandwidth)) in network.nic_bandwidth_overrides.iter().enumerate() {
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"nic_id\": {},", nic_id)?;
        writeln!(
            writer,
            "{indent}    \"bandwidth_gbps\": {}",
            json_f64(bandwidth.as_gigabits_per_sec())
        )?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(idx + 1 < network.nic_bandwidth_overrides.len())
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_nic_latency_scale_overrides_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"latency_scale_overrides\": [")?;
    for (idx, (nic_id, latency_scale)) in network.nic_latency_scale_overrides.iter().enumerate() {
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"nic_id\": {},", nic_id)?;
        writeln!(
            writer,
            "{indent}    \"latency_scale\": {}",
            json_f64(*latency_scale)
        )?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(idx + 1 < network.nic_latency_scale_overrides.len())
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_nic_rail_map_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"nic_rail_map\": [")?;
    for (idx, (nic_id, rail_id)) in network.nic_rail_map.iter().enumerate() {
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"nic_id\": {},", nic_id)?;
        writeln!(writer, "{indent}    \"rail_id\": {}", rail_id)?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(idx + 1 < network.nic_rail_map.len())
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_gpu_nic_map_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"gpu_nic_map\": [")?;
    for (idx, (gpu_id, nic_ids)) in network.gpu_nic_map.iter().enumerate() {
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"local_gpu_id\": {},", gpu_id)?;
        write_nic_id_array(writer, &format!("{indent}    "), "nic_ids", nic_ids, false)?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(idx + 1 < network.gpu_nic_map.len())
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_gpu_numa_map_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"gpu_numa_map\": [")?;
    for (idx, (gpu_id, domain)) in network.gpu_numa_map.iter().enumerate() {
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"local_gpu_id\": {},", gpu_id)?;
        writeln!(writer, "{indent}    \"numa_domain\": {}", domain)?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(idx + 1 < network.gpu_numa_map.len())
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_nic_numa_map_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"nic_numa_map\": [")?;
    for (idx, (nic_id, domain)) in network.nic_numa_map.iter().enumerate() {
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"nic_id\": {},", nic_id)?;
        writeln!(writer, "{indent}    \"numa_domain\": {}", domain)?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(idx + 1 < network.nic_numa_map.len())
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_gpu_nic_paths_json<W: Write>(
    writer: &mut W,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"gpu_nic_paths\": [")?;
    for (idx, ((gpu_id, nic_id), path)) in network.gpu_nic_path_overrides.iter().enumerate() {
        writeln!(writer, "{indent}  {{")?;
        writeln!(writer, "{indent}    \"local_gpu_id\": {},", gpu_id)?;
        writeln!(writer, "{indent}    \"nic_id\": {},", nic_id)?;
        writeln!(
            writer,
            "{indent}    \"label\": {},",
            json_optional_string(path.label.as_deref())
        )?;
        writeln!(
            writer,
            "{indent}    \"bandwidth_gbps\": {},",
            path.bandwidth
                .map(|bandwidth| json_f64(bandwidth.as_gigabits_per_sec()))
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}    \"latency_us\": {},",
            path.latency
                .map(|latency| json_f64(latency.to_us()))
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}    \"gpudirect\": {},",
            path.gpudirect
                .map(|gpudirect| gpudirect.to_string())
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(writer, "{indent}    \"available\": {}", path.available)?;
        writeln!(
            writer,
            "{indent}  }}{}",
            comma(idx + 1 < network.gpu_nic_path_overrides.len())
        )?;
    }
    writeln!(writer, "{indent}]{}", comma(trailing_comma))
}

fn write_nic_id_array<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    values: &[u32],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}\"{field}\": [{}]{}",
        u32_list(values),
        comma(trailing_comma)
    )
}

fn write_id_set_array<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    values: &BTreeSet<u32>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let values = values.iter().copied().collect::<Vec<_>>();
    write_nic_id_array(writer, indent, field, &values, trailing_comma)
}

fn write_rail_id_array<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    network: &crate::types::fabric::intra_node::NodeNetworkProfile,
    local_gpu_id: u32,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let rail_ids = network
        .nic_candidates_for_gpu(local_gpu_id)
        .into_iter()
        .map(|nic_id| network.rail_id(nic_id))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    write_nic_id_array(writer, indent, field, &rail_ids, trailing_comma)
}

fn write_cluster_node_groups<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let mut groups: Vec<_> = cluster.node_groups.iter().collect();
    groups.sort_by_key(|(label, _)| *label);

    writeln!(writer, "{indent}  \"node_groups\": [")?;
    for (idx, (label, nodes)) in groups.iter().enumerate() {
        let mut nodes = (*nodes).clone();
        nodes.sort_unstable();
        nodes.dedup();
        writeln!(writer, "{indent}    {{")?;
        writeln!(writer, "{indent}      \"group\": {},", json_string(label))?;
        writeln!(
            writer,
            "{indent}      \"nodes\": [{}]",
            nodes
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < groups.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_inter_node_topology<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"inter_node_topology\": {{")?;
    match &cluster.inter_node_topology {
        InterNodeTopology::FatTree {
            link,
            oversubscription,
            leaf_size,
        } => {
            writeln!(writer, "{indent}    \"kind\": \"fat_tree\",")?;
            writeln!(
                writer,
                "{indent}    \"oversubscription\": {},",
                json_f64(*oversubscription)
            )?;
            writeln!(writer, "{indent}    \"leaf_size\": {leaf_size},")?;
            write_fabric_profile_object(writer, "default_link", link, indent, false)?;
        }
        InterNodeTopology::Flat { link } => {
            writeln!(writer, "{indent}    \"kind\": \"flat\",")?;
            write_fabric_profile_object(writer, "default_link", link, indent, false)?;
        }
        InterNodeTopology::Custom(links) => {
            let mut custom_links = Vec::new();
            for (pair, profiles) in links {
                let (from, to) = pair.endpoints();
                for link in profiles {
                    custom_links.push((*from, *to, link));
                }
            }
            custom_links.sort_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then(left.1.cmp(&right.1))
                    .then(left.2.rail.cmp(&right.2.rail))
                    .then(left.2.profile.label.cmp(right.2.profile.label))
            });

            writeln!(writer, "{indent}    \"kind\": \"custom\",")?;
            writeln!(
                writer,
                "{indent}    \"link_count\": {},",
                custom_links.len()
            )?;
            writeln!(writer, "{indent}    \"links\": [")?;
            for (idx, (from, to, link)) in custom_links.iter().enumerate() {
                let (from_gpus, to_gpus) = custom_link_gpu_scope(link, *from, *to);
                writeln!(writer, "{indent}      {{")?;
                writeln!(writer, "{indent}        \"from_node\": {from},")?;
                writeln!(writer, "{indent}        \"to_node\": {to},")?;
                writeln!(
                    writer,
                    "{indent}        \"rail\": {},",
                    json_optional_u32(link.rail)
                )?;
                writeln!(
                    writer,
                    "{indent}        \"from_gpus\": [{}],",
                    u32_list(&from_gpus)
                )?;
                writeln!(
                    writer,
                    "{indent}        \"to_gpus\": [{}],",
                    u32_list(&to_gpus)
                )?;
                write_fabric_profile_object(
                    writer,
                    "fabric",
                    &link.profile,
                    &format!("{indent}    "),
                    false,
                )?;
                writeln!(
                    writer,
                    "{indent}      }}{}",
                    comma(idx + 1 < custom_links.len())
                )?;
            }
            writeln!(writer, "{indent}    ]")?;
        }
    }
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_fabric_profile_object<W: Write>(
    writer: &mut W,
    field: &str,
    profile: &FabricProfile,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"{field}\": {{")?;
    writeln!(
        writer,
        "{indent}      \"label\": {},",
        json_string(profile.label)
    )?;
    writeln!(
        writer,
        "{indent}      \"kind\": {},",
        json_string(fabric_kind_label(profile.kind))
    )?;
    writeln!(
        writer,
        "{indent}      \"bandwidth_gbps\": {},",
        json_f64(profile.bw.unidirectional.as_gigabits_per_sec())
    )?;
    writeln!(
        writer,
        "{indent}      \"latency_us\": {},",
        json_f64(profile.latency.to_us())
    )?;
    writeln!(
        writer,
        "{indent}      \"reduction_accelerator\": {},",
        json_string(reduction_accelerator_label(&profile.reduction_accel))
    )?;
    writeln!(
        writer,
        "{indent}      \"reduction_min_message_gb\": {}",
        json_optional_value(reduction_min_message_gb(&profile.reduction_accel))
    )?;
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_cluster_topology_diagnostics<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let diagnostics = cluster_topology_diagnostics(cluster);
    writeln!(writer, "{indent}  \"topology_diagnostics\": [")?;
    for (idx, diagnostic) in diagnostics.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"severity\": {},",
            json_string(diagnostic.severity)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(diagnostic.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&diagnostic.message)
        )?;
        writeln!(
            writer,
            "{indent}      \"group\": {},",
            json_optional_string(diagnostic.group.as_deref())
        )?;
        writeln!(
            writer,
            "{indent}      \"from_node\": {},",
            json_optional_u32(diagnostic.from_node)
        )?;
        writeln!(
            writer,
            "{indent}      \"to_node\": {},",
            json_optional_u32(diagnostic.to_node)
        )?;
        writeln!(
            writer,
            "{indent}      \"rail\": {},",
            json_optional_u32(diagnostic.rail)
        )?;
        writeln!(writer, "{indent}      \"components\": [")?;
        for (component_idx, component) in diagnostic.components.iter().enumerate() {
            writeln!(
                writer,
                "{indent}        {}{}",
                json_u32_array(component),
                comma(component_idx + 1 < diagnostic.components.len())
            )?;
        }
        writeln!(writer, "{indent}      ]")?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < diagnostics.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn cluster_gpu_type_counts(cluster: &Cluster) -> BTreeMap<String, GpuInventoryEntry> {
    let mut counts = BTreeMap::new();
    for node in cluster.nodes.values() {
        for gpu in node.gpus.values() {
            let profile = gpu.profile();
            let entry =
                counts
                    .entry(profile.label.to_string())
                    .or_insert_with(|| GpuInventoryEntry {
                        count: 0,
                        profile: profile.clone(),
                    });
            entry.count += 1;
        }
    }
    counts
}

fn node_gpu_type_counts(cluster: &Cluster, node_id: u32) -> BTreeMap<String, u32> {
    let mut counts = BTreeMap::new();
    if let Some(node) = cluster.nodes.get(&node_id) {
        for gpu in node.gpus.values() {
            *counts.entry(gpu.profile().label.to_string()).or_insert(0) += 1;
        }
    }
    counts
}

fn gpu_nic_affinity_label(affinity: GpuNicAffinity) -> &'static str {
    match affinity {
        GpuNicAffinity::Dedicated => "dedicated",
        GpuNicAffinity::Shared { .. } => "shared",
        GpuNicAffinity::Uniform => "uniform",
    }
}

fn gpu_nic_gpus_per_nic(affinity: GpuNicAffinity) -> Option<u32> {
    match affinity {
        GpuNicAffinity::Shared { gpus_per_nic } => Some(u32::from(gpus_per_nic)),
        GpuNicAffinity::Dedicated | GpuNicAffinity::Uniform => None,
    }
}

fn fabric_kind_label(kind: FabricKind) -> &'static str {
    match kind {
        FabricKind::InfiniBand => "infiniband",
        FabricKind::RoCE => "roce",
        FabricKind::Ethernet => "ethernet",
    }
}

fn reduction_accelerator_label(accelerator: &ReductionAccelerator) -> &'static str {
    match accelerator {
        ReductionAccelerator::None => "none",
        ReductionAccelerator::Supported { .. } => "supported",
    }
}

fn reduction_min_message_gb(accelerator: &ReductionAccelerator) -> Option<f64> {
    match accelerator {
        ReductionAccelerator::None => None,
        ReductionAccelerator::Supported { min_message } => Some(min_message.as_gigabytes()),
    }
}
