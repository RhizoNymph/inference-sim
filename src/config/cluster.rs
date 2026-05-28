use super::*;

pub(super) fn parse_custom_cluster(file: ClusterFile) -> Result<Cluster, ConfigError> {
    let node_sections = file.nodes.unwrap_or_default();
    let node_group_sections = file.node_groups.unwrap_or_default();
    if node_sections.is_empty() && node_group_sections.is_empty() {
        return Err(ConfigError::new(
            "custom clusters require at least one [[nodes]] or [[node_groups]] entry",
        ));
    }

    let default_interconnect = file
        .interconnect
        .as_ref()
        .and_then(|section| {
            if section.kind.is_some() || section.variant.is_some() {
                Some(interconnect_profile(section))
            } else {
                None
            }
        })
        .transpose()?;
    let default_network = match (&file.nics, default_interconnect.clone()) {
        (Some(nics), Some(interconnect)) => Some(nics_profile("nics", nics, interconnect)?),
        (Some(nics), None) => Some(nics_profile("nics", nics, default_nic_reference_profile())?),
        _ => None,
    };

    let mut nodes = HashMap::new();
    let mut node_groups = HashMap::new();
    for section in node_sections {
        let topology = parse_node_topology_metadata(
            &format!("nodes[{}]", section.id),
            section.node_tag.as_deref(),
            section.node_tags.as_deref(),
            section.rack.as_deref(),
            section.island.as_deref(),
            section.failure_domain.as_deref(),
        )?;
        let node = build_node(
            &format!("nodes[{}]", section.id),
            topology,
            section.gpu.as_deref(),
            section.gpu_count,
            GpuProfileOverrideConfig {
                hbm_gb: section.hbm_gb,
                hbm_bandwidth_gb_s: section.hbm_bandwidth_gb_s,
                peak_f16_tflops: section.peak_f16_tflops,
                peak_f8_tflops: section.peak_f8_tflops,
            },
            section.gpu_tag,
            section.gpu_tags,
            section.gpus,
            section.gpu_profile_overrides,
            section.disabled_gpus,
            section.gpu_states,
            section.intra.as_deref(),
            section.nics,
            default_interconnect.clone(),
            default_network.clone(),
        )?;
        insert_node(&mut nodes, section.id, node)?;
        if let Some(group) = section.group {
            add_group_node(&mut node_groups, &group, section.id)?;
        }
    }

    for (group_idx, section) in node_group_sections.into_iter().enumerate() {
        if section.count == 0 {
            return Err(ConfigError::new(format!(
                "node_groups[{group_idx}].count must be nonzero"
            )));
        }
        let topology = parse_node_topology_metadata(
            &format!("node_groups[{group_idx}]"),
            section.node_tag.as_deref(),
            section.node_tags.as_deref(),
            section.rack.as_deref(),
            section.island.as_deref(),
            section.failure_domain.as_deref(),
        )?;
        let start_id = section.start_id.unwrap_or_else(|| next_node_id(&nodes));
        for offset in 0..section.count {
            let node_id = start_id + offset;
            let node = build_node(
                &format!("node_groups[{group_idx}]"),
                topology.clone(),
                Some(&section.gpu),
                Some(section.gpu_count),
                GpuProfileOverrideConfig {
                    hbm_gb: section.hbm_gb,
                    hbm_bandwidth_gb_s: section.hbm_bandwidth_gb_s,
                    peak_f16_tflops: section.peak_f16_tflops,
                    peak_f8_tflops: section.peak_f8_tflops,
                },
                section.gpu_tag.clone(),
                section.gpu_tags.clone(),
                None,
                section.gpu_profile_overrides.clone(),
                section.disabled_gpus.clone(),
                section.gpu_states.clone(),
                section.intra.as_deref(),
                section.nics.clone(),
                default_interconnect.clone(),
                default_network.clone(),
            )?;
            insert_node(&mut nodes, node_id, node)?;
            if let Some(label) = &section.label {
                add_group_node(&mut node_groups, label, node_id)?;
            }
        }
    }

    finalize_node_groups(&mut node_groups, &nodes);

    let inter_node_topology = custom_interconnect_topology(
        file.interconnect,
        default_interconnect,
        &nodes,
        &node_groups,
    )?;

    Ok(Cluster {
        nodes,
        node_groups,
        inter_node_topology,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_node(
    name: &str,
    topology: NodeTopologyMetadata,
    gpu_name: Option<&str>,
    gpu_count: Option<u32>,
    gpu_profile_override: GpuProfileOverrideConfig,
    gpu_tag: Option<String>,
    gpu_tags: Option<Vec<String>>,
    gpu_sections: Option<Vec<NodeGpuSection>>,
    gpu_profile_override_sections: Option<Vec<GpuProfileOverrideSection>>,
    disabled_gpus: Option<Vec<u32>>,
    gpu_states: Option<Vec<GpuStateSection>>,
    intra: Option<&str>,
    nics: Option<NicsSection>,
    default_interconnect: Option<FabricProfile>,
    default_network: Option<NodeNetworkProfile>,
) -> Result<Node, ConfigError> {
    let inventory = build_node_gpus(
        name,
        gpu_name,
        gpu_count,
        gpu_profile_override,
        gpu_tag,
        gpu_tags,
        gpu_sections,
    )?;
    let gpus = inventory.gpus;
    let gpu_labels = inventory.gpu_labels;
    let mut gpu_profile_overrides = inventory.gpu_profile_overrides;
    merge_gpu_profile_overrides(
        name,
        &mut gpu_profile_overrides,
        gpu_profile_override_sections.as_deref(),
        &gpus,
    )?;
    let mut gpu_operational_states = inventory.gpu_operational_states;
    merge_gpu_operational_states(
        name,
        &mut gpu_operational_states,
        gpu_states.as_deref(),
        &gpus,
    )?;
    let disabled_gpus =
        parse_disabled_gpus(name, disabled_gpus, &gpus, &mut gpu_operational_states)?;
    let representative_gpu = representative_gpu_for_intra(name, &gpus, intra)?;
    let node_interconnect = default_interconnect.unwrap_or_else(default_nic_reference_profile);
    let network = match nics {
        Some(nics) => nics_profile(&format!("{name}.nics"), &nics, node_interconnect)?,
        None => default_network.unwrap_or(NodeNetworkProfile {
            nic_count: 1,
            nic_bandwidth: node_interconnect.bw.unidirectional,
            nic_bandwidth_overrides: Default::default(),
            nic_latency_scale_overrides: Default::default(),
            gpu_to_nic: GpuNicAffinity::Uniform,
            rail_count: 1,
            nic_rail_map: Default::default(),
            gpu_nic_map: Default::default(),
            gpu_numa_map: Default::default(),
            nic_numa_map: Default::default(),
            cross_numa_bandwidth_scale: 1.0,
            cross_numa_latency_scale: 1.0,
            gpu_nic_path_overrides: Default::default(),
            disabled_nics: Default::default(),
            nic_operational_states: Default::default(),
        }),
    };
    validate_network_profile(&format!("{name}.nics"), &gpus, &disabled_gpus, &network)?;

    Ok(Node {
        gpus,
        topology,
        gpu_labels,
        gpu_profile_overrides,
        disabled_gpus,
        gpu_operational_states,
        operational_state: NodeOperationalState::Healthy,
        intra_node_fabric: intra_node_topology(intra, representative_gpu)?,
        network,
    })
}

pub(super) struct NodeGpuInventory {
    gpus: HashMap<u32, Gpu>,
    gpu_labels: HashMap<u32, BTreeSet<String>>,
    gpu_profile_overrides: HashMap<u32, GpuProfile>,
    gpu_operational_states: HashMap<u32, OperationalState>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct GpuProfileOverrideConfig {
    hbm_gb: Option<f64>,
    hbm_bandwidth_gb_s: Option<f64>,
    peak_f16_tflops: Option<f64>,
    peak_f8_tflops: Option<f64>,
}

impl GpuProfileOverrideConfig {
    fn has_overrides(&self) -> bool {
        self.hbm_gb.is_some()
            || self.hbm_bandwidth_gb_s.is_some()
            || self.peak_f16_tflops.is_some()
            || self.peak_f8_tflops.is_some()
    }
}

pub(super) fn build_node_gpus(
    name: &str,
    gpu_name: Option<&str>,
    gpu_count: Option<u32>,
    gpu_profile_override: GpuProfileOverrideConfig,
    gpu_tag: Option<String>,
    gpu_tags: Option<Vec<String>>,
    gpu_sections: Option<Vec<NodeGpuSection>>,
) -> Result<NodeGpuInventory, ConfigError> {
    if let Some(sections) = gpu_sections {
        if gpu_name.is_some()
            || gpu_count.is_some()
            || gpu_profile_override.has_overrides()
            || gpu_tag.is_some()
            || gpu_tags.is_some()
        {
            return Err(ConfigError::new(format!(
                "{name} must set either gpu/gpu_count/node-wide gpu profile fields/gpu_tags or gpus, not both"
            )));
        }
        return explicit_node_gpus(name, sections);
    }

    let gpu_name = gpu_name.ok_or_else(|| {
        ConfigError::new(format!(
            "{name}.gpu is required unless {name}.gpus is provided"
        ))
    })?;
    let gpu_count = gpu_count.ok_or_else(|| {
        ConfigError::new(format!(
            "{name}.gpu_count is required unless {name}.gpus is provided"
        ))
    })?;
    if gpu_count == 0 {
        return Err(ConfigError::new(format!(
            "{name}.gpu_count must be nonzero"
        )));
    }

    let gpu = gpu(gpu_name)?;
    let mut gpus = HashMap::new();
    let mut gpu_labels = HashMap::new();
    let mut gpu_profile_overrides = HashMap::new();
    let gpu_operational_states = HashMap::new();
    let labels = parse_gpu_labels(
        &format!("{name}.gpu_tags"),
        gpu_tag.as_deref(),
        gpu_tags.as_deref(),
    )?;
    let profile_override = gpu_profile_override_from_config(name, gpu, &gpu_profile_override)?;
    for local_gpu_id in 0..gpu_count {
        gpus.insert(local_gpu_id, gpu);
        if !labels.is_empty() {
            gpu_labels.insert(local_gpu_id, labels.clone());
        }
        if let Some(profile) = &profile_override {
            gpu_profile_overrides.insert(local_gpu_id, profile.clone());
        }
    }
    Ok(NodeGpuInventory {
        gpus,
        gpu_labels,
        gpu_profile_overrides,
        gpu_operational_states,
    })
}

pub(super) fn explicit_node_gpus(
    name: &str,
    sections: Vec<NodeGpuSection>,
) -> Result<NodeGpuInventory, ConfigError> {
    if sections.is_empty() {
        return Err(ConfigError::new(format!("{name}.gpus must not be empty")));
    }

    let mut gpus = HashMap::new();
    let mut gpu_labels = HashMap::new();
    let mut gpu_profile_overrides = HashMap::new();
    let mut gpu_operational_states = HashMap::new();
    for (idx, section) in sections.into_iter().enumerate() {
        if section.id.is_some() && section.start_id.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpus[{idx}] cannot set both id and start_id"
            )));
        }
        let count = section.count.unwrap_or(1);
        if count == 0 {
            return Err(ConfigError::new(format!(
                "{name}.gpus[{idx}].count must be nonzero"
            )));
        }
        let start_id = section
            .start_id
            .or(section.id)
            .unwrap_or_else(|| next_local_gpu_id(&gpus));
        let gpu = gpu(&section.gpu)?;
        let profile_override = gpu_profile_override_from_config(
            &format!("{name}.gpus[{idx}]"),
            gpu,
            &GpuProfileOverrideConfig {
                hbm_gb: section.hbm_gb,
                hbm_bandwidth_gb_s: section.hbm_bandwidth_gb_s,
                peak_f16_tflops: section.peak_f16_tflops,
                peak_f8_tflops: section.peak_f8_tflops,
            },
        )?;
        let labels = parse_gpu_labels(
            &format!("{name}.gpus[{idx}].gpu_tags"),
            section.gpu_tag.as_deref(),
            section.gpu_tags.as_deref(),
        )?;
        let operational_state = section
            .state
            .as_deref()
            .map(|state| {
                parse_resource_operational_state(&format!("{name}.gpus[{idx}].state"), state)
            })
            .transpose()?;
        for offset in 0..count {
            let local_gpu_id = start_id + offset;
            if gpus.insert(local_gpu_id, gpu).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.gpus[{idx}] duplicates local GPU id {local_gpu_id}"
                )));
            }
            if !labels.is_empty() {
                gpu_labels.insert(local_gpu_id, labels.clone());
            }
            if let Some(profile) = &profile_override {
                gpu_profile_overrides.insert(local_gpu_id, profile.clone());
            }
            if let Some(operational_state) = operational_state {
                gpu_operational_states.insert(local_gpu_id, operational_state);
            }
        }
    }

    Ok(NodeGpuInventory {
        gpus,
        gpu_labels,
        gpu_profile_overrides,
        gpu_operational_states,
    })
}

pub(super) fn gpu_profile_override_from_config(
    name: &str,
    gpu: Gpu,
    config: &GpuProfileOverrideConfig,
) -> Result<Option<GpuProfile>, ConfigError> {
    gpu_profile_override_from_base_config(name, gpu.profile(), config)
}

pub(super) fn gpu_profile_override_from_base_config(
    name: &str,
    mut profile: GpuProfile,
    config: &GpuProfileOverrideConfig,
) -> Result<Option<GpuProfile>, ConfigError> {
    if !config.has_overrides() {
        return Ok(None);
    }

    validate_positive_optional_f64(&format!("{name}.hbm_gb"), config.hbm_gb)?;
    validate_positive_optional_f64(
        &format!("{name}.hbm_bandwidth_gb_s"),
        config.hbm_bandwidth_gb_s,
    )?;
    validate_positive_optional_f64(&format!("{name}.peak_f16_tflops"), config.peak_f16_tflops)?;
    validate_positive_optional_f64(&format!("{name}.peak_f8_tflops"), config.peak_f8_tflops)?;

    if let Some(hbm_gb) = config.hbm_gb {
        profile.hbm_size = Bytes::from_gigabytes(hbm_gb);
    }
    if let Some(hbm_bandwidth_gb_s) = config.hbm_bandwidth_gb_s {
        profile.hbm_bandwidth = Bandwidth::from_gigabytes_per_sec(hbm_bandwidth_gb_s);
    }
    if let Some(peak_f16_tflops) = config.peak_f16_tflops {
        profile.peak_f16_flops = peak_f16_tflops;
    }
    if let Some(peak_f8_tflops) = config.peak_f8_tflops {
        profile.peak_f8_flops = Some(peak_f8_tflops);
    }

    Ok(Some(profile))
}

pub(super) fn merge_gpu_profile_overrides(
    name: &str,
    overrides: &mut HashMap<u32, GpuProfile>,
    entries: Option<&[GpuProfileOverrideSection]>,
    gpus: &HashMap<u32, Gpu>,
) -> Result<(), ConfigError> {
    let Some(entries) = entries else {
        return Ok(());
    };

    let mut seen = BTreeSet::new();
    for (idx, entry) in entries.iter().enumerate() {
        if entry.local_gpu_id.is_some() && entry.local_gpu_ids.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_profile_overrides[{idx}] cannot set both gpu and gpus"
            )));
        }
        let gpu_ids = match (entry.local_gpu_id, entry.local_gpu_ids.as_ref()) {
            (Some(gpu_id), None) => vec![gpu_id],
            (None, Some(gpu_ids)) if !gpu_ids.is_empty() => gpu_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides[{idx}].gpus must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides[{idx}] must set gpu or gpus"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        let config = GpuProfileOverrideConfig {
            hbm_gb: entry.hbm_gb,
            hbm_bandwidth_gb_s: entry.hbm_bandwidth_gb_s,
            peak_f16_tflops: entry.peak_f16_tflops,
            peak_f8_tflops: entry.peak_f8_tflops,
        };
        if !config.has_overrides() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_profile_overrides[{idx}] must specify at least one profile override"
            )));
        }

        for gpu_id in gpu_ids {
            let Some(gpu) = gpus.get(&gpu_id).copied() else {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides[{idx}] references unknown local GPU id {gpu_id}"
                )));
            };
            if !seen.insert(gpu_id) {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides duplicates local GPU id {gpu_id}"
                )));
            }
            let base_profile = overrides
                .get(&gpu_id)
                .cloned()
                .unwrap_or_else(|| gpu.profile());
            let Some(profile) = gpu_profile_override_from_base_config(
                &format!("{name}.gpu_profile_overrides[{idx}]"),
                base_profile,
                &config,
            )?
            else {
                unreachable!("profile override config was checked above");
            };
            overrides.insert(gpu_id, profile);
        }
    }

    Ok(())
}

pub(super) fn parse_gpu_labels(
    name: &str,
    gpu_tag: Option<&str>,
    gpu_tags: Option<&[String]>,
) -> Result<BTreeSet<String>, ConfigError> {
    let mut labels = BTreeSet::new();
    if let Some(tag) = gpu_tag {
        let label = normalize(tag);
        if label.is_empty() {
            return Err(ConfigError::new(format!(
                "{name} must not contain empty labels"
            )));
        }
        labels.insert(label);
    }
    if let Some(tags) = gpu_tags {
        if tags.is_empty() {
            return Err(ConfigError::new(format!("{name} must not be empty")));
        }
        for tag in tags {
            let label = normalize(tag);
            if label.is_empty() {
                return Err(ConfigError::new(format!(
                    "{name} must not contain empty labels"
                )));
            }
            labels.insert(label);
        }
    }
    Ok(labels)
}

pub(super) fn parse_node_topology_metadata(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    island: Option<&str>,
    failure_domain: Option<&str>,
) -> Result<NodeTopologyMetadata, ConfigError> {
    Ok(NodeTopologyMetadata {
        labels: parse_normalized_label_set(&format!("{name}.node_tags"), node_tag, node_tags)?,
        rack: parse_optional_topology_domain(&format!("{name}.rack"), rack)?,
        island: parse_optional_topology_domain(&format!("{name}.island"), island)?,
        failure_domain: parse_optional_topology_domain(
            &format!("{name}.failure_domain"),
            failure_domain,
        )?,
    })
}

pub(super) fn parse_normalized_label_set(
    name: &str,
    label: Option<&str>,
    labels: Option<&[String]>,
) -> Result<BTreeSet<String>, ConfigError> {
    let mut parsed = BTreeSet::new();
    for label in parse_optional_normalized_labels(name, label, labels)? {
        parsed.insert(label);
    }
    Ok(parsed)
}

pub(super) fn parse_optional_normalized_labels(
    name: &str,
    label: Option<&str>,
    labels: Option<&[String]>,
) -> Result<Vec<String>, ConfigError> {
    if label.is_some() && labels.is_some() {
        return Err(ConfigError::new(format!(
            "{name} cannot set both singular and plural labels"
        )));
    }
    let mut parsed = Vec::new();
    if let Some(label) = label {
        parsed.push(parse_topology_label(name, label)?);
    }
    if let Some(labels) = labels {
        if labels.is_empty() {
            return Err(ConfigError::new(format!("{name} must not be empty")));
        }
        for label in labels {
            let parsed_label = parse_topology_label(name, label)?;
            if !parsed.contains(&parsed_label) {
                parsed.push(parsed_label);
            }
        }
    }
    Ok(parsed)
}

pub(super) fn parse_optional_topology_domains(
    name: &str,
    label: Option<&str>,
    labels: Option<&[String]>,
) -> Result<Vec<String>, ConfigError> {
    parse_optional_normalized_labels(name, label, labels)
}

pub(super) fn parse_optional_topology_domain(
    name: &str,
    value: Option<&str>,
) -> Result<Option<String>, ConfigError> {
    value
        .map(|value| parse_topology_label(name, value))
        .transpose()
}

pub(super) fn parse_topology_label(name: &str, value: &str) -> Result<String, ConfigError> {
    let label = normalize(value);
    if label.is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }
    Ok(label)
}

pub(super) fn parse_disabled_gpus(
    name: &str,
    values: Option<Vec<u32>>,
    gpus: &HashMap<u32, Gpu>,
    gpu_operational_states: &mut HashMap<u32, OperationalState>,
) -> Result<BTreeSet<u32>, ConfigError> {
    let mut disabled = BTreeSet::new();
    for gpu_id in values.unwrap_or_default() {
        if !gpus.contains_key(&gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_gpus references unknown local GPU id {gpu_id}"
            )));
        }
        if !disabled.insert(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_gpus duplicates local GPU id {gpu_id}"
            )));
        }
        gpu_operational_states.insert(gpu_id, OperationalState::Disabled);
    }

    disabled.extend(
        gpu_operational_states
            .iter()
            .filter_map(|(gpu_id, state)| (!state.accepts_work()).then_some(*gpu_id)),
    );

    Ok(disabled)
}

pub(super) fn merge_gpu_operational_states(
    name: &str,
    states: &mut HashMap<u32, OperationalState>,
    entries: Option<&[GpuStateSection]>,
    gpus: &HashMap<u32, Gpu>,
) -> Result<(), ConfigError> {
    let Some(entries) = entries else {
        return Ok(());
    };
    for (idx, entry) in entries.iter().enumerate() {
        if entry.local_gpu_id.is_some() && entry.local_gpu_ids.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_states[{idx}] cannot set both gpu and gpus"
            )));
        }
        let gpu_ids = match (entry.local_gpu_id, entry.local_gpu_ids.as_ref()) {
            (Some(gpu_id), None) => vec![gpu_id],
            (None, Some(gpu_ids)) if !gpu_ids.is_empty() => gpu_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states[{idx}].gpus must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states[{idx}] must set gpu or gpus"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        let state = parse_resource_operational_state(
            &format!("{name}.gpu_states[{idx}].state"),
            &entry.state,
        )?;
        for gpu_id in gpu_ids {
            if !gpus.contains_key(&gpu_id) {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states[{idx}] references unknown local GPU id {gpu_id}"
                )));
            }
            if states.insert(gpu_id, state).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states duplicates local GPU id {gpu_id}"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn parse_resource_operational_state(
    name: &str,
    value: &str,
) -> Result<OperationalState, ConfigError> {
    match normalize(value).as_str() {
        "healthy" | "enabled" | "available" | "active" => Ok(OperationalState::Healthy),
        "disabled" | "offline" | "unavailable" | "failed" => Ok(OperationalState::Disabled),
        "maintenance" => Ok(OperationalState::Maintenance),
        "draining" | "drain" => Ok(OperationalState::Draining),
        "reserved" => Ok(OperationalState::Reserved),
        parsed => Err(ConfigError::new(format!(
            "unsupported {name} '{parsed}'; use healthy, disabled, maintenance, draining, or reserved"
        ))),
    }
}

pub(super) fn next_local_gpu_id(gpus: &HashMap<u32, Gpu>) -> u32 {
    gpus.keys().copied().max().map_or(0, |gpu_id| gpu_id + 1)
}

pub(super) fn representative_gpu_for_intra(
    name: &str,
    gpus: &HashMap<u32, Gpu>,
    intra: Option<&str>,
) -> Result<Gpu, ConfigError> {
    let Some(first) = gpus.values().next().copied() else {
        return Err(ConfigError::new(format!("{name}.gpus must not be empty")));
    };
    let mixed_gpu_types = gpus.values().any(|gpu| *gpu != first);
    if intra.is_none() && mixed_gpu_types {
        return Err(ConfigError::new(format!(
            "{name}.intra is required when {name}.gpus mixes GPU types"
        )));
    }

    Ok(first)
}

pub(super) fn insert_node(
    nodes: &mut HashMap<u32, Node>,
    node_id: u32,
    node: Node,
) -> Result<(), ConfigError> {
    if nodes.insert(node_id, node).is_some() {
        return Err(ConfigError::new(format!("duplicate node id {node_id}")));
    }

    Ok(())
}

pub(super) fn next_node_id(nodes: &HashMap<u32, Node>) -> u32 {
    nodes.keys().copied().max().map_or(0, |node_id| node_id + 1)
}

pub(super) fn add_group_node(
    node_groups: &mut HashMap<String, Vec<u32>>,
    label: &str,
    node_id: u32,
) -> Result<(), ConfigError> {
    let key = normalize(label);
    if key.is_empty() {
        return Err(ConfigError::new("node group labels must not be empty"));
    }
    node_groups.entry(key).or_default().push(node_id);
    Ok(())
}

pub(super) fn finalize_node_groups(
    node_groups: &mut HashMap<String, Vec<u32>>,
    nodes: &HashMap<u32, Node>,
) {
    let mut all_nodes: Vec<_> = nodes.keys().copied().collect();
    all_nodes.sort_unstable();
    node_groups
        .entry("all".to_string())
        .or_default()
        .extend(all_nodes);

    for group_nodes in node_groups.values_mut() {
        group_nodes.sort_unstable();
        group_nodes.dedup();
    }
}

pub(super) fn custom_interconnect_topology(
    section: Option<InterconnectSection>,
    default_profile: Option<FabricProfile>,
    nodes: &HashMap<u32, Node>,
    node_groups: &HashMap<String, Vec<u32>>,
) -> Result<InterNodeTopology, ConfigError> {
    let Some(section) = section else {
        return Ok(InterNodeTopology::Custom(HashMap::new()));
    };

    if let Some(links) = section.links {
        let mut edges = HashMap::new();
        for link in links {
            let mut profile = interconnect_link_profile(&link)?;
            if let Some(oversubscription) = link.oversubscription {
                if oversubscription < 1.0 {
                    return Err(ConfigError::new(
                        "interconnect.links[].oversubscription must be greater than or equal to 1.0",
                    ));
                }
                profile.bw.unidirectional = profile.bw.unidirectional / oversubscription;
            }
            let rails = interconnect_link_rails(&link)?;
            for pair in interconnect_link_pairs(&link, nodes, node_groups)? {
                let links = edges
                    .entry(UnorderedPair::new(pair.from, pair.to))
                    .or_insert_with(Vec::new);
                for rail in &rails {
                    links.push(CustomInterNodeLink {
                        profile: profile.clone(),
                        rail: *rail,
                        endpoints: pair.endpoint_scope(),
                    });
                }
            }
        }
        return Ok(InterNodeTopology::Custom(edges));
    }

    Ok(match default_profile {
        Some(link) => InterNodeTopology::FatTree {
            link,
            oversubscription: section.oversubscription.unwrap_or(1.0).max(1.0),
            leaf_size: 0,
        },
        None => InterNodeTopology::Custom(HashMap::new()),
    })
}

pub(super) fn interconnect_link_rails(
    link: &InterconnectLinkSection,
) -> Result<Vec<Option<u32>>, ConfigError> {
    if link.rail.is_some() && link.rails.is_some() {
        return Err(ConfigError::new(
            "interconnect.links[].rail and rails cannot both be set",
        ));
    }
    if let Some(rail) = link.rail {
        return Ok(vec![Some(rail)]);
    }
    if let Some(rails) = &link.rails {
        if rails.is_empty() {
            return Err(ConfigError::new(
                "interconnect.links[].rails must not be empty",
            ));
        }
        let mut values = rails.clone();
        values.sort_unstable();
        values.dedup();
        return Ok(values.into_iter().map(Some).collect());
    }

    Ok(vec![None])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct InterconnectLinkPair {
    from: u32,
    to: u32,
    from_gpus: Vec<u32>,
    to_gpus: Vec<u32>,
}

impl InterconnectLinkPair {
    fn endpoint_scope(&self) -> Option<CustomInterNodeLinkEndpoints> {
        if self.from_gpus.is_empty() && self.to_gpus.is_empty() {
            None
        } else {
            Some(CustomInterNodeLinkEndpoints {
                from_node: self.from,
                from_gpus: self.from_gpus.clone(),
                to_node: self.to,
                to_gpus: self.to_gpus.clone(),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct InterconnectGpuEndpointSelector {
    explicit_gpus: Vec<u32>,
    gpu_types: Vec<Gpu>,
    gpu_labels: Vec<String>,
}

impl InterconnectGpuEndpointSelector {
    fn uses_filtered_selector(&self) -> bool {
        !self.gpu_types.is_empty() || !self.gpu_labels.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct InterconnectNodeEndpointSelector {
    explicit_node: Option<u32>,
    group: Option<String>,
    node_labels: Vec<String>,
    racks: Vec<String>,
    islands: Vec<String>,
    failure_domains: Vec<String>,
}

impl InterconnectNodeEndpointSelector {
    fn uses_selector(&self) -> bool {
        self.explicit_node.is_some()
            || self.group.is_some()
            || !self.node_labels.is_empty()
            || !self.racks.is_empty()
            || !self.islands.is_empty()
            || !self.failure_domains.is_empty()
    }
}

pub(super) fn interconnect_link_pairs(
    link: &InterconnectLinkSection,
    nodes: &HashMap<u32, Node>,
    node_groups: &HashMap<String, Vec<u32>>,
) -> Result<Vec<InterconnectLinkPair>, ConfigError> {
    let from_node_selector = interconnect_endpoint_node_selector(
        "from",
        link.from,
        link.from_group.as_deref(),
        link.from_node_tag.as_deref(),
        link.from_node_tags.as_deref(),
        link.from_rack.as_deref(),
        link.from_racks.as_deref(),
        link.from_island.as_deref(),
        link.from_islands.as_deref(),
        link.from_failure_domain.as_deref(),
        link.from_failure_domains.as_deref(),
    )?;
    let to_node_selector = interconnect_endpoint_node_selector(
        "to",
        link.to,
        link.to_group.as_deref(),
        link.to_node_tag.as_deref(),
        link.to_node_tags.as_deref(),
        link.to_rack.as_deref(),
        link.to_racks.as_deref(),
        link.to_island.as_deref(),
        link.to_islands.as_deref(),
        link.to_failure_domain.as_deref(),
        link.to_failure_domains.as_deref(),
    )?;
    let source_nodes =
        interconnect_endpoint_nodes("from", &from_node_selector, nodes, node_groups)?;
    let dest_nodes = interconnect_endpoint_nodes("to", &to_node_selector, nodes, node_groups)?;
    let from_gpu_selector = interconnect_endpoint_gpu_selector(
        "from",
        link.from_gpu,
        link.from_gpus.as_deref(),
        link.from_gpu_type.as_deref(),
        link.from_gpu_types.as_deref(),
        link.from_gpu_tag.as_deref(),
        link.from_gpu_tags.as_deref(),
    )?;
    let to_gpu_selector = interconnect_endpoint_gpu_selector(
        "to",
        link.to_gpu,
        link.to_gpus.as_deref(),
        link.to_gpu_type.as_deref(),
        link.to_gpu_types.as_deref(),
        link.to_gpu_tag.as_deref(),
        link.to_gpu_tags.as_deref(),
    )?;
    let mut pairs = Vec::new();

    for from in &source_nodes {
        for to in &dest_nodes {
            if from != to && !same_unordered_link_pair_exists(&pairs, *from, *to) {
                let from_gpus =
                    interconnect_endpoint_gpus("from", *from, &from_gpu_selector, nodes)?;
                let to_gpus = interconnect_endpoint_gpus("to", *to, &to_gpu_selector, nodes)?;
                if (from_gpu_selector.uses_filtered_selector() && from_gpus.is_empty())
                    || (to_gpu_selector.uses_filtered_selector() && to_gpus.is_empty())
                {
                    continue;
                }
                pairs.push(InterconnectLinkPair {
                    from: *from,
                    to: *to,
                    from_gpus,
                    to_gpus,
                });
            }
        }
    }

    if pairs.is_empty() {
        return Err(ConfigError::new(
            "interconnect.links[] must connect at least two distinct nodes",
        ));
    }

    Ok(pairs)
}

pub(super) fn interconnect_endpoint_gpu_selector(
    name: &str,
    gpu_id: Option<u32>,
    gpu_ids: Option<&[u32]>,
    gpu_type: Option<&str>,
    gpu_types: Option<&[String]>,
    gpu_tag: Option<&str>,
    gpu_tags: Option<&[String]>,
) -> Result<InterconnectGpuEndpointSelector, ConfigError> {
    if gpu_id.is_some() && gpu_ids.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpu and {name}_gpus cannot both be set"
        )));
    }
    if gpu_type.is_some() && gpu_types.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpu_type and {name}_gpu_types cannot both be set"
        )));
    }
    if gpu_tag.is_some() && gpu_tags.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpu_tag and {name}_gpu_tags cannot both be set"
        )));
    }
    let has_explicit_gpus = gpu_id.is_some() || gpu_ids.is_some();
    let has_gpu_types = gpu_type.is_some() || gpu_types.is_some();
    let has_gpu_tags = gpu_tag.is_some() || gpu_tags.is_some();
    if has_explicit_gpus && (has_gpu_types || has_gpu_tags) {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpus cannot be combined with {name}_gpu_type or {name}_gpu_tag selectors"
        )));
    }
    let mut gpus = Vec::new();
    if let Some(gpu_id) = gpu_id {
        gpus.push(gpu_id);
    }
    if let Some(gpu_ids) = gpu_ids {
        if gpu_ids.is_empty() {
            return Err(ConfigError::new(format!(
                "interconnect.links[].{name}_gpus must not be empty"
            )));
        }
        gpus.extend_from_slice(gpu_ids);
    }
    sort_dedup_or_error(&mut gpus, &format!("interconnect.links[].{name}_gpus"))?;
    let mut selected_gpu_types = Vec::new();
    if let Some(gpu_type) = gpu_type {
        selected_gpu_types.push(gpu(gpu_type)?);
    }
    if let Some(gpu_types) = gpu_types {
        if gpu_types.is_empty() {
            return Err(ConfigError::new(format!(
                "interconnect.links[].{name}_gpu_types must not be empty"
            )));
        }
        for gpu_type in gpu_types {
            let parsed = gpu(gpu_type)?;
            if !selected_gpu_types.contains(&parsed) {
                selected_gpu_types.push(parsed);
            }
        }
    }
    let selected_gpu_labels = parse_gpu_labels(
        &format!("interconnect.links[].{name}_gpu_tags"),
        gpu_tag,
        gpu_tags,
    )?
    .into_iter()
    .collect();
    Ok(InterconnectGpuEndpointSelector {
        explicit_gpus: gpus,
        gpu_types: selected_gpu_types,
        gpu_labels: selected_gpu_labels,
    })
}

pub(super) fn interconnect_endpoint_gpus(
    name: &str,
    node_id: u32,
    selector: &InterconnectGpuEndpointSelector,
    nodes: &HashMap<u32, Node>,
) -> Result<Vec<u32>, ConfigError> {
    let Some(node) = nodes.get(&node_id) else {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name} references unknown node id {node_id}"
        )));
    };
    if !selector.explicit_gpus.is_empty() {
        for gpu_id in &selector.explicit_gpus {
            if !node.gpus.contains_key(gpu_id) {
                return Err(ConfigError::new(format!(
                    "interconnect.links[].{name}_gpus references node {node_id} local GPU id {gpu_id}, but that GPU does not exist"
                )));
            }
        }
        return Ok(selector.explicit_gpus.clone());
    }
    if selector.gpu_types.is_empty() && selector.gpu_labels.is_empty() {
        return Ok(Vec::new());
    }
    let mut gpus = node
        .gpus
        .iter()
        .filter_map(|(gpu_id, gpu)| {
            let matches_type = selector.gpu_types.is_empty() || selector.gpu_types.contains(gpu);
            let matches_label = selector.gpu_labels.is_empty()
                || node.gpu_labels(*gpu_id).is_some_and(|labels| {
                    selector
                        .gpu_labels
                        .iter()
                        .any(|label| labels.contains(label))
                });
            (matches_type && matches_label).then_some(*gpu_id)
        })
        .collect::<Vec<_>>();
    gpus.sort_unstable();
    Ok(gpus)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn interconnect_endpoint_node_selector(
    name: &str,
    node_id: Option<u32>,
    group: Option<&str>,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
) -> Result<InterconnectNodeEndpointSelector, ConfigError> {
    if node_id.is_some() && group.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name} and {name}_group cannot both be set"
        )));
    }
    Ok(InterconnectNodeEndpointSelector {
        explicit_node: node_id,
        group: group.map(normalize),
        node_labels: parse_optional_normalized_labels(
            &format!("interconnect.links[].{name}_node_tags"),
            node_tag,
            node_tags,
        )?,
        racks: parse_optional_topology_domains(
            &format!("interconnect.links[].{name}_racks"),
            rack,
            racks,
        )?,
        islands: parse_optional_topology_domains(
            &format!("interconnect.links[].{name}_islands"),
            island,
            islands,
        )?,
        failure_domains: parse_optional_topology_domains(
            &format!("interconnect.links[].{name}_failure_domains"),
            failure_domain,
            failure_domains,
        )?,
    })
}

pub(super) fn interconnect_endpoint_nodes(
    name: &str,
    selector: &InterconnectNodeEndpointSelector,
    nodes: &HashMap<u32, Node>,
    node_groups: &HashMap<String, Vec<u32>>,
) -> Result<Vec<u32>, ConfigError> {
    if !selector.uses_selector() {
        return Err(ConfigError::new(format!(
            "interconnect.links[] requires {name}, {name}_group, {name}_node_tag, {name}_rack, {name}_island, or {name}_failure_domain"
        )));
    }

    let mut selected = if let Some(node_id) = selector.explicit_node {
        if !nodes.contains_key(&node_id) {
            return Err(ConfigError::new(format!(
                "interconnect.links[].{name} references unknown node id {node_id}"
            )));
        }
        vec![node_id]
    } else if let Some(group) = &selector.group {
        node_groups.get(group).cloned().ok_or_else(|| {
            ConfigError::new(format!(
                "interconnect.links[].{name}_group references unknown group '{group}'"
            ))
        })?
    } else {
        let mut node_ids: Vec<_> = nodes.keys().copied().collect();
        node_ids.sort_unstable();
        node_ids
    };

    selected.retain(|node_id| {
        nodes
            .get(node_id)
            .is_some_and(|node| node_matches_topology_selector(node, selector))
    });
    selected.sort_unstable();
    selected.dedup();
    if selected.is_empty() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name} selector matched no nodes"
        )));
    }
    Ok(selected)
}

pub(super) fn node_matches_topology_selector(
    node: &Node,
    selector: &InterconnectNodeEndpointSelector,
) -> bool {
    (selector.node_labels.is_empty()
        || selector
            .node_labels
            .iter()
            .any(|label| node.topology.labels.contains(label)))
        && (selector.racks.is_empty()
            || node
                .topology
                .rack
                .as_ref()
                .is_some_and(|rack| selector.racks.contains(rack)))
        && (selector.islands.is_empty()
            || node
                .topology
                .island
                .as_ref()
                .is_some_and(|island| selector.islands.contains(island)))
        && (selector.failure_domains.is_empty()
            || node
                .topology
                .failure_domain
                .as_ref()
                .is_some_and(|failure_domain| selector.failure_domains.contains(failure_domain)))
}

pub(super) fn same_unordered_link_pair_exists(
    pairs: &[InterconnectLinkPair],
    from: u32,
    to: u32,
) -> bool {
    pairs
        .iter()
        .any(|pair| (pair.from == from && pair.to == to) || (pair.from == to && pair.to == from))
}

pub(super) fn default_nic_reference_profile() -> FabricProfile {
    FabricProfile {
        kind: crate::types::common::FabricKind::InfiniBand,
        label: "default NIC profile",
        bw: crate::types::common::LinkBandwidth::full_duplex_unidirectional(
            Bandwidth::from_gigabits_per_sec(400.0),
        ),
        latency: crate::types::common::Latency::from_us(1.2),
        reduction_accel: crate::types::common::ReductionAccelerator::None,
    }
}

pub(super) fn interconnect_profile(
    section: &InterconnectSection,
) -> Result<FabricProfile, ConfigError> {
    let raw_kind = section
        .kind
        .as_deref()
        .ok_or_else(|| ConfigError::new("interconnect.kind is required"))?;
    let raw_variant = section
        .variant
        .as_deref()
        .ok_or_else(|| ConfigError::new("interconnect.variant is required"))?;
    let kind = normalize(raw_kind);
    let variant = normalize(raw_variant);

    match kind.as_str() {
        "ib" | "infiniband" => match variant.as_str() {
            "edr" => Ok(IbVariant::Edr.default_profile()),
            "hdr" => Ok(IbVariant::Hdr.default_profile()),
            "ndr" => Ok(IbVariant::Ndr.default_profile()),
            "xdr" => Ok(IbVariant::Xdr.default_profile()),
            _ => Err(ConfigError::new(format!(
                "unsupported InfiniBand variant '{}'; use edr, hdr, ndr, or xdr",
                raw_variant
            ))),
        },
        "roce" | "rocev2" => match variant.as_str() {
            "25g" | "v2_25g" => Ok(RoceVariant::V2_25G.default_profile()),
            "50g" | "v2_50g" => Ok(RoceVariant::V2_50G.default_profile()),
            "100g" | "v2_100g" => Ok(RoceVariant::V2_100G.default_profile()),
            "200g" | "v2_200g" => Ok(RoceVariant::V2_200G.default_profile()),
            "400g" | "v2_400g" => Ok(RoceVariant::V2_400G.default_profile()),
            "800g" | "v2_800g" => Ok(RoceVariant::V2_800G.default_profile()),
            _ => Err(ConfigError::new(format!(
                "unsupported RoCE variant '{}'; use 25g, 50g, 100g, 200g, 400g, or 800g",
                raw_variant
            ))),
        },
        "eth" | "ethernet" => match variant.as_str() {
            "10g" => Ok(EthVariant::E10G.default_profile()),
            "25g" => Ok(EthVariant::E25G.default_profile()),
            "40g" => Ok(EthVariant::E40G.default_profile()),
            "100g" => Ok(EthVariant::E100G.default_profile()),
            "200g" => Ok(EthVariant::E200G.default_profile()),
            "400g" => Ok(EthVariant::E400G.default_profile()),
            "800g" => Ok(EthVariant::E800G.default_profile()),
            _ => Err(ConfigError::new(format!(
                "unsupported Ethernet variant '{}'; use 10g, 25g, 40g, 100g, 200g, 400g, or 800g",
                raw_variant
            ))),
        },
        _ => Err(ConfigError::new(format!(
            "unsupported interconnect kind '{}'; use ib, roce, or ethernet",
            raw_kind
        ))),
    }
}

pub(super) fn interconnect_link_profile(
    section: &InterconnectLinkSection,
) -> Result<FabricProfile, ConfigError> {
    interconnect_profile(&InterconnectSection {
        kind: Some(section.kind.clone()),
        variant: Some(section.variant.clone()),
        oversubscription: None,
        links: None,
    })
}

pub(super) fn gpu(value: &str) -> Result<Gpu, ConfigError> {
    match normalize(value).as_str() {
        "a100_40gb" | "a100_40" => Ok(Gpu::A100_40GB),
        "a100_80gb" | "a100_80" => Ok(Gpu::A100_80GB),
        "h100_sxm" | "h100" => Ok(Gpu::H100_SXM),
        "h200_sxm" | "h200" => Ok(Gpu::H200_SXM),
        "b200" => Ok(Gpu::B200),
        "mi300x" => Ok(Gpu::MI300X),
        _ => Err(ConfigError::new(format!(
            "unsupported node gpu '{value}'; use a100_40gb, a100_80gb, h100_sxm, h200_sxm, b200, or mi300x"
        ))),
    }
}

pub(super) fn intra_node_topology(
    value: Option<&str>,
    gpu: Gpu,
) -> Result<IntraNodeTopology, ConfigError> {
    let value = value.map(normalize).unwrap_or_else(|| match gpu {
        Gpu::A100_40GB | Gpu::A100_80GB => "nvlink_v3".to_string(),
        Gpu::H100_SXM | Gpu::H200_SXM => "nvlink_v4".to_string(),
        Gpu::B200 => "nvlink_v5".to_string(),
        Gpu::MI300X => "pcie_gen5".to_string(),
    });

    match value.as_str() {
        "nvlink_v1" | "nvlink1" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V1.default_profile(),
        )),
        "nvlink_v2" | "nvlink2" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V2.default_profile(),
        )),
        "nvlink_v3" | "nvlink3" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V3.default_profile(),
        )),
        "nvlink_v4" | "nvlink4" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V4.default_profile(),
        )),
        "nvlink_v5" | "nvlink5" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V5.default_profile(),
        )),
        "pcie_gen3" | "gen3x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen3x16.default_profile(),
        )),
        "pcie_gen4" | "gen4x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen4x16.default_profile(),
        )),
        "pcie_gen5" | "gen5x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen5x16.default_profile(),
        )),
        "pcie_gen6" | "gen6x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen6x16.default_profile(),
        )),
        _ => Err(ConfigError::new(format!(
            "unsupported node intra fabric '{value}'; use nvlink_v3, nvlink_v4, nvlink_v5, pcie_gen4, or pcie_gen5"
        ))),
    }
}

pub(super) fn nics_profile(
    name: &str,
    section: &NicsSection,
    interconnect: FabricProfile,
) -> Result<NodeNetworkProfile, ConfigError> {
    let affinity = match section.affinity.as_deref().map(normalize) {
        None => GpuNicAffinity::Dedicated,
        Some(value) if value == "dedicated" => GpuNicAffinity::Dedicated,
        Some(value) if value == "uniform" => GpuNicAffinity::Uniform,
        Some(value) if value == "shared" => GpuNicAffinity::Shared {
            gpus_per_nic: {
                let gpus_per_nic = section.gpus_per_nic.ok_or_else(|| {
                    ConfigError::new(format!(
                        "{name}.gpus_per_nic is required for shared affinity"
                    ))
                })?;
                if gpus_per_nic == 0 {
                    return Err(ConfigError::new(format!(
                        "{name}.gpus_per_nic must be nonzero"
                    )));
                }
                gpus_per_nic
            },
        },
        Some(value) => {
            return Err(ConfigError::new(format!(
                "unsupported {name}.affinity '{value}'; use dedicated, shared, or uniform"
            )));
        }
    };

    let nic_count = section.count.unwrap_or(8);
    if nic_count == 0 {
        return Err(ConfigError::new(format!("{name}.count must be nonzero")));
    }
    let rail_count = match section.rail_count {
        Some(0) => {
            return Err(ConfigError::new(format!(
                "{name}.rail_count must be nonzero"
            )));
        }
        Some(rail_count) => rail_count,
        None => nic_count,
    };
    if rail_count > nic_count {
        return Err(ConfigError::new(format!(
            "{name}.rail_count must be less than or equal to {name}.count"
        )));
    }

    let nic_bandwidth = match section.bandwidth_gbps {
        Some(gbps) if gbps > 0.0 => Bandwidth::from_gigabits_per_sec(gbps),
        Some(_) => {
            return Err(ConfigError::new(format!(
                "{name}.bandwidth_gbps must be positive"
            )));
        }
        None => interconnect.bw.unidirectional,
    };
    let gpu_nic_map = parse_gpu_nic_map(name, section.gpu_nic_map.as_deref())?;
    let gpu_numa_map = parse_gpu_numa_map(name, section.gpu_numa_map.as_deref())?;
    let nic_numa_map = parse_nic_numa_map(name, section.nic_numa_map.as_deref(), nic_count)?;
    validate_positive_optional_f64(
        &format!("{name}.cross_numa_bandwidth_scale"),
        section.cross_numa_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{name}.cross_numa_latency_scale"),
        section.cross_numa_latency_scale,
    )?;
    let cross_numa_bandwidth_scale = section.cross_numa_bandwidth_scale.unwrap_or(1.0);
    let cross_numa_latency_scale = section.cross_numa_latency_scale.unwrap_or(1.0);
    let nic_rail_map =
        parse_nic_rail_map(name, section.nic_rail_map.as_deref(), nic_count, rail_count)?;
    let gpu_nic_path_overrides =
        parse_gpu_nic_paths(name, section.gpu_nic_paths.as_deref(), nic_count)?;
    let nic_bandwidth_overrides =
        parse_nic_bandwidth_overrides(name, section.nic_bandwidth_overrides.as_deref(), nic_count)?;
    let nic_latency_scale_overrides = parse_nic_latency_scale_overrides(
        name,
        section.nic_latency_scale_overrides.as_deref(),
        nic_count,
    )?;
    let mut nic_operational_states =
        parse_nic_operational_states(name, section.nic_states.as_deref(), nic_count)?;
    let disabled_nics = parse_disabled_nics(
        name,
        section.disabled_nics.as_deref(),
        nic_count,
        &mut nic_operational_states,
    )?;

    Ok(NodeNetworkProfile {
        nic_count,
        nic_bandwidth,
        nic_bandwidth_overrides,
        nic_latency_scale_overrides,
        gpu_to_nic: affinity,
        rail_count,
        nic_rail_map,
        gpu_nic_map,
        gpu_numa_map,
        nic_numa_map,
        cross_numa_bandwidth_scale,
        cross_numa_latency_scale,
        gpu_nic_path_overrides,
        disabled_nics,
        nic_operational_states,
    })
}

pub(super) fn parse_nic_rail_map(
    name: &str,
    entries: Option<&[NicRailMapSection]>,
    nic_count: u8,
    rail_count: u8,
) -> Result<BTreeMap<NicId, u32>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        let nic_id = entry.nic.ok_or_else(|| {
            ConfigError::new(format!("{name}.nic_rail_map[{idx}].nic is required"))
        })?;
        let rail_id = entry.rail.ok_or_else(|| {
            ConfigError::new(format!("{name}.nic_rail_map[{idx}].rail is required"))
        })?;
        if nic_id >= u32::from(nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.nic_rail_map[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
            )));
        }
        if rail_id >= u32::from(rail_count) {
            return Err(ConfigError::new(format!(
                "{name}.nic_rail_map[{idx}] references rail {rail_id}, but {name}.rail_count is {rail_count}"
            )));
        }
        if map.insert(nic_id, rail_id).is_some() {
            return Err(ConfigError::new(format!(
                "{name}.nic_rail_map duplicates NIC {nic_id}"
            )));
        }
    }

    Ok(map)
}

pub(super) fn parse_gpu_nic_paths(
    name: &str,
    entries: Option<&[GpuNicPathSection]>,
    nic_count: u8,
) -> Result<BTreeMap<(GpuId, NicId), GpuNicPathOverride>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        let gpu_id = entry.local_gpu_id.ok_or_else(|| {
            ConfigError::new(format!(
                "{name}.gpu_nic_paths[{idx}].local_gpu_id is required"
            ))
        })?;
        let nic_id = entry.nic.ok_or_else(|| {
            ConfigError::new(format!("{name}.gpu_nic_paths[{idx}].nic is required"))
        })?;
        if nic_id >= u32::from(nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
            )));
        }
        let bandwidth = match entry.bandwidth_gbps {
            Some(gbps) if gbps.is_finite() && gbps > 0.0 => {
                Some(Bandwidth::from_gigabits_per_sec(gbps))
            }
            Some(_) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_paths[{idx}].bandwidth_gbps must be finite and positive"
                )));
            }
            None => None,
        };
        let latency = match entry.latency_us {
            Some(us) if us.is_finite() && us >= 0.0 => Some(Latency::from_us(us)),
            Some(_) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_paths[{idx}].latency_us must be finite and nonnegative"
                )));
            }
            None => None,
        };
        let override_profile = GpuNicPathOverride {
            label: entry.label.clone(),
            bandwidth,
            latency,
            gpudirect: entry.gpudirect,
            available: entry.available.unwrap_or(true),
        };
        if map.insert((gpu_id, nic_id), override_profile).is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths duplicates local GPU id {gpu_id} and NIC {nic_id}"
            )));
        }
    }

    Ok(map)
}

pub(super) fn parse_disabled_nics(
    name: &str,
    values: Option<&[u32]>,
    nic_count: u8,
    nic_operational_states: &mut BTreeMap<NicId, OperationalState>,
) -> Result<BTreeSet<NicId>, ConfigError> {
    let mut disabled = BTreeSet::new();
    for nic_id in values.unwrap_or_default() {
        if *nic_id >= u32::from(nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_nics references NIC {nic_id}, but {name}.count is {nic_count}"
            )));
        }
        if !disabled.insert(*nic_id) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_nics duplicates NIC {nic_id}"
            )));
        }
        nic_operational_states.insert(*nic_id, OperationalState::Disabled);
    }

    disabled.extend(
        nic_operational_states
            .iter()
            .filter_map(|(nic_id, state)| (!state.accepts_work()).then_some(*nic_id)),
    );

    Ok(disabled)
}

pub(super) fn parse_nic_operational_states(
    name: &str,
    entries: Option<&[NicStateSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, OperationalState>, ConfigError> {
    let mut states = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(states);
    };
    for (idx, entry) in entries.iter().enumerate() {
        if entry.nic.is_some() && entry.nics.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.nic_states[{idx}] cannot set both nic and nics"
            )));
        }
        let nic_ids = match (entry.nic, entry.nics.as_ref()) {
            (Some(nic_id), None) => vec![nic_id],
            (None, Some(nic_ids)) if !nic_ids.is_empty() => nic_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states[{idx}].nics must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states[{idx}] must set nic or nics"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        let state = parse_resource_operational_state(
            &format!("{name}.nic_states[{idx}].state"),
            &entry.state,
        )?;
        for nic_id in nic_ids {
            if nic_id >= u32::from(nic_count) {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
                )));
            }
            if states.insert(nic_id, state).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states duplicates NIC {nic_id}"
                )));
            }
        }
    }
    Ok(states)
}

pub(super) fn parse_nic_bandwidth_overrides(
    name: &str,
    entries: Option<&[NicBandwidthOverrideSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, Bandwidth>, ConfigError> {
    let mut overrides = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(overrides);
    };
    for (idx, entry) in entries.iter().enumerate() {
        let nic_ids = parse_nic_override_ids(
            name,
            "nic_bandwidth_overrides",
            idx,
            entry.nic,
            entry.nics.as_deref(),
        )?;
        if !entry.bandwidth_gbps.is_finite() || entry.bandwidth_gbps <= 0.0 {
            return Err(ConfigError::new(format!(
                "{name}.nic_bandwidth_overrides[{idx}].bandwidth_gbps must be finite and positive"
            )));
        }
        for nic_id in nic_ids {
            validate_nic_override_id(name, "nic_bandwidth_overrides", idx, nic_id, nic_count)?;
            if overrides
                .insert(
                    nic_id,
                    Bandwidth::from_gigabits_per_sec(entry.bandwidth_gbps),
                )
                .is_some()
            {
                return Err(ConfigError::new(format!(
                    "{name}.nic_bandwidth_overrides duplicates NIC {nic_id}"
                )));
            }
        }
    }
    Ok(overrides)
}

pub(super) fn parse_nic_latency_scale_overrides(
    name: &str,
    entries: Option<&[NicLatencyScaleOverrideSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, f64>, ConfigError> {
    let mut overrides = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(overrides);
    };
    for (idx, entry) in entries.iter().enumerate() {
        let nic_ids = parse_nic_override_ids(
            name,
            "nic_latency_scale_overrides",
            idx,
            entry.nic,
            entry.nics.as_deref(),
        )?;
        if !entry.latency_scale.is_finite() || entry.latency_scale <= 0.0 {
            return Err(ConfigError::new(format!(
                "{name}.nic_latency_scale_overrides[{idx}].latency_scale must be finite and positive"
            )));
        }
        for nic_id in nic_ids {
            validate_nic_override_id(name, "nic_latency_scale_overrides", idx, nic_id, nic_count)?;
            if overrides.insert(nic_id, entry.latency_scale).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.nic_latency_scale_overrides duplicates NIC {nic_id}"
                )));
            }
        }
    }
    Ok(overrides)
}

pub(super) fn parse_nic_override_ids(
    name: &str,
    field: &str,
    idx: usize,
    nic: Option<u32>,
    nics: Option<&[u32]>,
) -> Result<Vec<NicId>, ConfigError> {
    if nic.is_some() && nics.is_some() {
        return Err(ConfigError::new(format!(
            "{name}.{field}[{idx}] cannot set both nic and nics"
        )));
    }
    match (nic, nics) {
        (Some(nic_id), None) => Ok(vec![nic_id]),
        (None, Some(nic_ids)) if !nic_ids.is_empty() => Ok(nic_ids.to_vec()),
        (None, Some(_)) => Err(ConfigError::new(format!(
            "{name}.{field}[{idx}].nics must not be empty"
        ))),
        (None, None) => Err(ConfigError::new(format!(
            "{name}.{field}[{idx}] must set nic or nics"
        ))),
        (Some(_), Some(_)) => unreachable!("checked above"),
    }
}

pub(super) fn validate_nic_override_id(
    name: &str,
    field: &str,
    idx: usize,
    nic_id: NicId,
    nic_count: u8,
) -> Result<(), ConfigError> {
    if nic_id >= u32::from(nic_count) {
        return Err(ConfigError::new(format!(
            "{name}.{field}[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
        )));
    }
    Ok(())
}

pub(super) fn parse_gpu_nic_map(
    name: &str,
    entries: Option<&[GpuNicMapSection]>,
) -> Result<BTreeMap<GpuId, Vec<NicId>>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        let gpu_id = entry.local_gpu_id.ok_or_else(|| {
            ConfigError::new(format!(
                "{name}.gpu_nic_map[{idx}].local_gpu_id is required"
            ))
        })?;
        if entry.nic.is_some() && entry.nics.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map[{idx}] cannot set both nic and nics"
            )));
        }
        let mut nics = match (entry.nic, entry.nics.as_ref()) {
            (Some(nic), None) => vec![nic],
            (None, Some(nics)) => nics.clone(),
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_map[{idx}] must set nic or nics"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        if nics.is_empty() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map[{idx}].nics must not be empty"
            )));
        }
        nics.sort_unstable();
        nics.dedup();
        if map.insert(gpu_id, nics).is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map duplicates local GPU id {gpu_id}"
            )));
        }
    }

    Ok(map)
}

pub(super) fn parse_gpu_numa_map(
    name: &str,
    entries: Option<&[GpuNumaMapSection]>,
) -> Result<BTreeMap<GpuId, u32>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        if entry.local_gpu_id.is_some() && entry.local_gpu_ids.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_numa_map[{idx}] cannot set both gpu and gpus"
            )));
        }
        let gpu_ids = match (entry.local_gpu_id, entry.local_gpu_ids.as_ref()) {
            (Some(gpu_id), None) => vec![gpu_id],
            (None, Some(gpu_ids)) if !gpu_ids.is_empty() => gpu_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_numa_map[{idx}].gpus must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_numa_map[{idx}] must set gpu or gpus"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        for gpu_id in gpu_ids {
            if map.insert(gpu_id, entry.numa_domain).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_numa_map duplicates local GPU id {gpu_id}"
                )));
            }
        }
    }

    Ok(map)
}

pub(super) fn parse_nic_numa_map(
    name: &str,
    entries: Option<&[NicNumaMapSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, u32>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        if entry.nic.is_some() && entry.nics.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.nic_numa_map[{idx}] cannot set both nic and nics"
            )));
        }
        let nic_ids = match (entry.nic, entry.nics.as_ref()) {
            (Some(nic_id), None) => vec![nic_id],
            (None, Some(nic_ids)) if !nic_ids.is_empty() => nic_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map[{idx}].nics must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map[{idx}] must set nic or nics"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        for nic_id in nic_ids {
            if nic_id >= u32::from(nic_count) {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
                )));
            }
            if map.insert(nic_id, entry.numa_domain).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map duplicates NIC {nic_id}"
                )));
            }
        }
    }

    Ok(map)
}

pub(super) fn validate_network_profile(
    name: &str,
    gpus: &HashMap<u32, Gpu>,
    disabled_gpus: &BTreeSet<u32>,
    profile: &NodeNetworkProfile,
) -> Result<(), ConfigError> {
    let gpu_count = gpus.len() as u32;
    match profile.gpu_to_nic {
        GpuNicAffinity::Dedicated if u32::from(profile.nic_count) < gpu_count => {
            Err(ConfigError::new(format!(
                "{name} dedicated affinity requires nics.count >= gpu_count ({gpu_count}), got {}",
                profile.nic_count
            )))
        }
        GpuNicAffinity::Shared { gpus_per_nic } => {
            let required_nics = gpu_count.div_ceil(u32::from(gpus_per_nic));
            if required_nics > u32::from(profile.nic_count) {
                return Err(ConfigError::new(format!(
                    "{name} shared affinity with gpus_per_nic={} requires at least {required_nics} NICs for {gpu_count} GPUs, got {}",
                    gpus_per_nic, profile.nic_count
                )));
            }
            Ok(())
        }
        _ => Ok(()),
    }?;

    for (gpu_id, nic_ids) in &profile.gpu_nic_map {
        if !gpus.contains_key(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map references unknown local GPU id {gpu_id}"
            )));
        }
        for nic_id in nic_ids {
            if *nic_id >= u32::from(profile.nic_count) {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_map for local GPU id {gpu_id} references NIC {nic_id}, but {name}.count is {}",
                    profile.nic_count
                )));
            }
        }
    }

    for (gpu_id, nic_id) in profile.gpu_nic_path_overrides.keys() {
        if !gpus.contains_key(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths references unknown local GPU id {gpu_id}"
            )));
        }
        if *nic_id >= u32::from(profile.nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths for local GPU id {gpu_id} references NIC {nic_id}, but {name}.count is {}",
                profile.nic_count
            )));
        }
    }

    for gpu_id in profile.gpu_numa_map.keys() {
        if !gpus.contains_key(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_numa_map references unknown local GPU id {gpu_id}"
            )));
        }
    }

    for gpu_id in gpus.keys() {
        if disabled_gpus.contains(gpu_id) {
            continue;
        }
        if profile.nic_candidates_for_gpu(*gpu_id).is_empty() {
            return Err(ConfigError::new(format!(
                "{name} leaves local GPU id {gpu_id} without an enabled NIC path"
            )));
        }
    }

    Ok(())
}
