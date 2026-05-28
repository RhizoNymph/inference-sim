use std::collections::{BTreeSet, HashMap, HashSet};

use crate::types::{
    common::{GpuAddr, GpuId, NicId, NodeId, OperationalState},
    fabric::{
        inter_node::{FabricProfile, InterNodeTopology},
        intra_node::{GpuNicAffinity, IntraNodeTopology, NodeNetworkProfile},
        variants::nvlink::NvLinkVariant,
    },
    gpu::{Gpu, GpuProfile},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub gpus: HashMap<GpuId, Gpu>,
    pub topology: NodeTopologyMetadata,
    pub gpu_labels: HashMap<GpuId, BTreeSet<String>>,
    pub gpu_profile_overrides: HashMap<GpuId, GpuProfile>,
    pub disabled_gpus: BTreeSet<GpuId>,
    pub gpu_operational_states: HashMap<GpuId, OperationalState>,
    pub operational_state: NodeOperationalState,
    pub intra_node_fabric: IntraNodeTopology,
    pub network: NodeNetworkProfile,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeTopologyMetadata {
    pub labels: BTreeSet<String>,
    pub rack: Option<String>,
    pub island: Option<String>,
    pub failure_domain: Option<String>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum NodeOperationalState {
    #[default]
    Healthy,
    Disabled,
    Maintenance,
    Draining,
    Reserved,
}

impl NodeOperationalState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Disabled => "disabled",
            Self::Maintenance => "maintenance",
            Self::Draining => "draining",
            Self::Reserved => "reserved",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    pub nodes: HashMap<NodeId, Node>,
    pub node_groups: HashMap<String, Vec<NodeId>>,
    pub inter_node_topology: InterNodeTopology,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    GpuCompute(GpuAddr),
    GpuHbm(GpuAddr),
    IntraNode { node_id: NodeId },
    Nic { node_id: NodeId, nic_id: NicId },
    InterNodeFabric,
}

impl Cluster {
    pub fn h100_sxm_nodes(node_count: u32, interconnect_profile: FabricProfile) -> Self {
        let mut nodes = HashMap::new();
        let nvlink = NvLinkVariant::V4.default_profile();
        let nic_bandwidth = interconnect_profile.bw.unidirectional;

        for node_id in 0..node_count {
            let mut gpus = HashMap::new();
            for local_gpu_id in 0..8 {
                gpus.insert(local_gpu_id, Gpu::H100_SXM);
            }

            nodes.insert(
                node_id,
                Node {
                    gpus,
                    topology: Default::default(),
                    gpu_labels: Default::default(),
                    gpu_profile_overrides: Default::default(),
                    disabled_gpus: Default::default(),
                    gpu_operational_states: Default::default(),
                    operational_state: NodeOperationalState::Healthy,
                    intra_node_fabric: IntraNodeTopology::NvSwitch(nvlink.clone()),
                    network: NodeNetworkProfile {
                        nic_count: 8,
                        nic_bandwidth,
                        nic_bandwidth_overrides: Default::default(),
                        nic_latency_scale_overrides: Default::default(),
                        gpu_to_nic: GpuNicAffinity::Dedicated,
                        rail_count: 8,
                        nic_rail_map: Default::default(),
                        gpu_nic_map: Default::default(),
                        gpu_numa_map: Default::default(),
                        nic_numa_map: Default::default(),
                        cross_numa_bandwidth_scale: 1.0,
                        cross_numa_latency_scale: 1.0,
                        gpu_nic_path_overrides: Default::default(),
                        disabled_nics: Default::default(),
                        nic_operational_states: Default::default(),
                    },
                },
            );
        }

        let node_ids: Vec<_> = (0..node_count).collect();
        let mut node_groups = HashMap::new();
        node_groups.insert("all".to_string(), node_ids.clone());
        node_groups.insert("h100".to_string(), node_ids);

        Self {
            nodes,
            node_groups,
            inter_node_topology: InterNodeTopology::FatTree {
                link: interconnect_profile,
                oversubscription: 1.0,
                leaf_size: node_count.min(u16::MAX as u32) as u16,
            },
        }
    }

    pub fn total_gpus(&self) -> u32 {
        self.nodes.values().map(|node| node.gpus.len() as u32).sum()
    }

    pub fn available_gpus(&self) -> u32 {
        self.nodes
            .values()
            .map(|node| node.available_gpu_count() as u32)
            .sum()
    }

    pub fn sorted_gpu_addrs(&self) -> Vec<GpuAddr> {
        let mut addrs = Vec::new();
        let mut node_ids: Vec<_> = self.nodes.keys().copied().collect();
        node_ids.sort_unstable();

        for node_id in node_ids {
            if let Some(node) = self.nodes.get(&node_id) {
                let mut gpu_ids: Vec<_> = node.gpus.keys().copied().collect();
                gpu_ids.sort_unstable();
                addrs.extend(gpu_ids.into_iter().map(|local_gpu_id| GpuAddr {
                    node_id,
                    local_gpu_id,
                }));
            }
        }

        addrs
    }

    pub fn gpu(&self, addr: GpuAddr) -> Option<Gpu> {
        self.nodes
            .get(&addr.node_id)
            .and_then(|node| node.gpus.get(&addr.local_gpu_id))
            .copied()
    }

    pub fn gpu_profile(&self, addr: GpuAddr) -> Option<GpuProfile> {
        self.nodes
            .get(&addr.node_id)?
            .gpu_profile(addr.local_gpu_id)
    }

    pub fn is_gpu_available(&self, addr: GpuAddr) -> bool {
        self.nodes
            .get(&addr.node_id)
            .is_some_and(|node| node.is_gpu_available(addr.local_gpu_id))
    }

    pub fn node(&self, node_id: NodeId) -> Option<&Node> {
        self.nodes.get(&node_id)
    }

    pub fn node_group(&self, label: &str) -> Option<&[NodeId]> {
        self.node_groups.get(label).map(Vec::as_slice)
    }

    pub fn subset_nodes(&self, node_ids: &[NodeId]) -> Result<Self, String> {
        let mut nodes = HashMap::new();
        let requested: HashSet<_> = node_ids.iter().copied().collect();
        for node_id in node_ids {
            let Some(node) = self.nodes.get(node_id) else {
                return Err(format!("unknown node id {node_id}"));
            };
            nodes.insert(*node_id, node.clone());
        }
        let node_groups = self
            .node_groups
            .iter()
            .filter_map(|(label, group_nodes)| {
                let mut retained: Vec<_> = group_nodes
                    .iter()
                    .copied()
                    .filter(|node_id| requested.contains(node_id))
                    .collect();
                retained.sort_unstable();
                retained.dedup();
                if retained.is_empty() {
                    None
                } else {
                    Some((label.clone(), retained))
                }
            })
            .collect();

        Ok(Self {
            nodes,
            node_groups,
            inter_node_topology: self.inter_node_topology.clone(),
        })
    }

    pub fn subset_nodes_with_gpu_labels(
        &self,
        node_ids: &[NodeId],
        gpu_labels: &[String],
    ) -> Result<Self, String> {
        let mut cluster = self.subset_nodes(node_ids)?;
        if gpu_labels.is_empty() {
            return Ok(cluster);
        }
        for node in cluster.nodes.values_mut() {
            let retained_gpus: BTreeSet<_> = node
                .gpus
                .keys()
                .copied()
                .filter(|gpu_id| {
                    node.gpu_labels(*gpu_id)
                        .is_some_and(|labels| gpu_labels.iter().any(|label| labels.contains(label)))
                })
                .collect();
            node.gpus.retain(|gpu_id, _| retained_gpus.contains(gpu_id));
            node.gpu_profile_overrides
                .retain(|gpu_id, _| retained_gpus.contains(gpu_id));
            node.gpu_labels
                .retain(|gpu_id, _| retained_gpus.contains(gpu_id));
            node.disabled_gpus
                .retain(|gpu_id| retained_gpus.contains(gpu_id));
            node.gpu_operational_states
                .retain(|gpu_id, _| retained_gpus.contains(gpu_id));
        }
        Ok(cluster)
    }
}

impl Node {
    pub fn gpu_profile(&self, local_gpu_id: GpuId) -> Option<GpuProfile> {
        self.gpu_profile_overrides
            .get(&local_gpu_id)
            .cloned()
            .or_else(|| self.gpus.get(&local_gpu_id).map(Gpu::profile))
    }

    pub fn is_gpu_available(&self, local_gpu_id: GpuId) -> bool {
        self.gpus.contains_key(&local_gpu_id)
            && self.gpu_operational_state(local_gpu_id).accepts_work()
    }

    pub fn gpu_operational_state(&self, local_gpu_id: GpuId) -> OperationalState {
        self.gpu_operational_states
            .get(&local_gpu_id)
            .copied()
            .unwrap_or_else(|| {
                if self.disabled_gpus.contains(&local_gpu_id) {
                    OperationalState::Disabled
                } else {
                    OperationalState::Healthy
                }
            })
    }

    pub fn gpu_labels(&self, local_gpu_id: GpuId) -> Option<&BTreeSet<String>> {
        self.gpu_labels.get(&local_gpu_id)
    }

    pub fn available_gpu_count(&self) -> usize {
        self.gpus
            .keys()
            .filter(|gpu_id| self.is_gpu_available(**gpu_id))
            .count()
    }
}
