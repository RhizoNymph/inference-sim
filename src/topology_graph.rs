use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::types::{
    common::{Bandwidth, Bytes, GpuAddr, NicId, NodeId, UnorderedPair},
    fabric::{
        inter_node::{CustomInterNodeLinkEndpoints, FabricProfile, InterNodeTopology},
        intra_node::{GpuNicPathOverride, NodeNetworkProfile},
    },
    topology::Cluster,
};

#[derive(Clone, Debug, PartialEq)]
struct GraphInterNodeLink {
    profile: FabricProfile,
    rail: Option<u32>,
    endpoints: Option<CustomInterNodeLinkEndpoints>,
    label: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GraphResource {
    Gpu(GpuAddr),
    IntraNode {
        node_id: NodeId,
    },
    Nic {
        node_id: NodeId,
        nic_id: NicId,
        rail_id: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceLink {
    pub from: GraphResource,
    pub to: GraphResource,
    pub kind: RoutedResourceKind,
    pub bandwidth: Bandwidth,
    pub latency_s: f64,
    pub rail_id: Option<u32>,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq)]
struct ResourceLinkProfile {
    kind: RoutedResourceKind,
    bandwidth: Bandwidth,
    latency_s: f64,
    rail_id: Option<u32>,
    label: String,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoutedResourceKind {
    IntraNodeFabric,
    GpuNicLocal,
    InterNodeFabric,
    GpuScopedInterNodeFabric,
}

impl RoutedResourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RoutedResourceKind::IntraNodeFabric => "intra_node_fabric",
            RoutedResourceKind::GpuNicLocal => "gpu_nic_local",
            RoutedResourceKind::InterNodeFabric => "inter_node_fabric",
            RoutedResourceKind::GpuScopedInterNodeFabric => "gpu_scoped_inter_node_fabric",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoutedResource {
    pub label: String,
    pub kind: RoutedResourceKind,
    pub from: GraphResource,
    pub to: GraphResource,
    pub bandwidth: Bandwidth,
    pub latency_s: f64,
    pub rail_id: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoutedPath {
    pub latency_s: f64,
    pub bottleneck_bandwidth: Bandwidth,
    pub labels: Vec<String>,
    pub resources: Vec<RoutedResource>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TopologyGraph {
    edges: HashMap<GraphResource, Vec<ResourceLink>>,
}

impl TopologyGraph {
    pub fn from_cluster(cluster: &Cluster) -> Self {
        let mut graph = Self::default();
        graph.add_gpu_nic_edges(cluster);
        graph.add_inter_node_edges(cluster);
        graph
    }

    pub fn route_between_gpus(
        &self,
        src: GpuAddr,
        dst: GpuAddr,
        bytes: Bytes,
    ) -> Option<RoutedPath> {
        self.route(
            GraphResource::Gpu(src),
            GraphResource::Gpu(dst),
            bytes.as_bytes(),
        )
    }

    pub fn route_between_nodes(
        &self,
        src: NodeId,
        dst: NodeId,
        bytes: Bytes,
    ) -> Option<RoutedPath> {
        let src_nics = self.nics_for_node(src);
        let dst_nics = self.nics_for_node(dst);
        let mut best = None;

        for src_nic in &src_nics {
            for dst_nic in &dst_nics {
                let Some(path) = self.route(src_nic.clone(), dst_nic.clone(), bytes.as_bytes())
                else {
                    continue;
                };
                if is_better_path(&path, best.as_ref(), bytes.as_bytes()) {
                    best = Some(path);
                }
            }
        }

        best
    }

    fn route(&self, src: GraphResource, dst: GraphResource, bytes: u64) -> Option<RoutedPath> {
        if src == dst {
            return Some(RoutedPath {
                latency_s: 0.0,
                bottleneck_bandwidth: Bandwidth::from_bytes_per_sec(f64::INFINITY),
                labels: Vec::new(),
                resources: Vec::new(),
            });
        }

        let mut nodes = BTreeSet::new();
        nodes.insert(src.clone());
        nodes.insert(dst.clone());
        for (node, links) in &self.edges {
            nodes.insert(node.clone());
            for link in links {
                nodes.insert(link.to.clone());
            }
        }

        let mut dist: BTreeMap<GraphResource, f64> = nodes
            .iter()
            .cloned()
            .map(|node| (node, f64::INFINITY))
            .collect();
        let mut prev: HashMap<GraphResource, ResourceLink> = HashMap::new();
        let mut visited = HashSet::new();
        dist.insert(src.clone(), 0.0);

        while let Some((current, current_dist)) = dist
            .iter()
            .filter(|(node, _)| !visited.contains(*node))
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(node, dist)| (node.clone(), *dist))
        {
            if !current_dist.is_finite() {
                break;
            }
            if current == dst {
                break;
            }
            visited.insert(current.clone());

            for link in self.edges.get(&current).into_iter().flatten() {
                let transfer_s = bytes as f64 / link.bandwidth.as_bytes_per_sec().max(1.0);
                let next_dist = current_dist + link.latency_s + transfer_s;
                if next_dist < *dist.get(&link.to).unwrap_or(&f64::INFINITY) {
                    dist.insert(link.to.clone(), next_dist);
                    prev.insert(link.to.clone(), link.clone());
                }
            }
        }

        if !prev.contains_key(&dst) {
            return None;
        }

        let mut cursor = dst;
        let mut links = Vec::new();
        while cursor != src {
            let link = prev.get(&cursor)?.clone();
            cursor = link.from.clone();
            links.push(link);
        }
        links.reverse();

        let latency_s = links.iter().map(|link| link.latency_s).sum();
        let bottleneck_bandwidth = links
            .iter()
            .map(|link| link.bandwidth)
            .min_by(|a, b| a.as_bytes_per_sec().total_cmp(&b.as_bytes_per_sec()))
            .unwrap_or_else(|| Bandwidth::from_bytes_per_sec(1.0));
        let mut labels = Vec::new();
        let mut resources = Vec::new();
        for link in links {
            if !labels.contains(&link.label) {
                labels.push(link.label.clone());
            }
            resources.push(RoutedResource {
                label: link.label,
                kind: link.kind,
                from: link.from,
                to: link.to,
                bandwidth: link.bandwidth,
                latency_s: link.latency_s,
                rail_id: link.rail_id,
            });
        }

        Some(RoutedPath {
            latency_s,
            bottleneck_bandwidth,
            labels,
            resources,
        })
    }

    fn add_gpu_nic_edges(&mut self, cluster: &Cluster) {
        for (node_id, node) in &cluster.nodes {
            let (latency_s, bandwidth) =
                crate::solver::Solver::intra_node_link_profile(cluster, *node_id);
            let mut gpu_ids: Vec<_> = node.gpus.keys().copied().collect();
            gpu_ids.sort_unstable();

            for gpu_id in gpu_ids {
                if !node.is_gpu_available(gpu_id) {
                    continue;
                }
                for nic_id in nic_candidates(&node.network, gpu_id) {
                    let (edge_latency_s, edge_bandwidth, path_label) =
                        gpu_nic_edge_profile(&node.network, gpu_id, nic_id, latency_s, bandwidth);
                    let rail_id = node.network.rail_id(nic_id);
                    self.add_bidirectional(
                        GraphResource::Gpu(GpuAddr {
                            node_id: *node_id,
                            local_gpu_id: gpu_id,
                        }),
                        GraphResource::Nic {
                            node_id: *node_id,
                            nic_id,
                            rail_id,
                        },
                        ResourceLinkProfile {
                            kind: RoutedResourceKind::GpuNicLocal,
                            bandwidth: edge_bandwidth,
                            latency_s: edge_latency_s,
                            rail_id: Some(rail_id),
                            label: format!(
                                "node {node_id} gpu {gpu_id} -> nic {nic_id} rail {rail_id}{path_label}"
                            ),
                        },
                    );
                }
            }
        }
    }

    fn add_inter_node_edges(&mut self, cluster: &Cluster) {
        let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
        node_ids.sort_unstable();

        for i in 0..node_ids.len() {
            for j in i + 1..node_ids.len() {
                let src = node_ids[i];
                let dst = node_ids[j];
                let links = inter_node_link_profiles(cluster, src, dst);
                if links.is_empty() {
                    continue;
                };
                let nic_pairs = matching_nic_pairs(cluster, src, dst);
                for link in links {
                    if link.endpoints.is_some() {
                        self.add_gpu_scoped_inter_node_edges(cluster, src, dst, &link);
                        continue;
                    }
                    for (src_nic, dst_nic, rail_id) in &nic_pairs {
                        if link.rail.is_some_and(|link_rail| link_rail != *rail_id) {
                            continue;
                        }
                        self.add_bidirectional(
                            GraphResource::Nic {
                                node_id: src,
                                nic_id: *src_nic,
                                rail_id: *rail_id,
                            },
                            GraphResource::Nic {
                                node_id: dst,
                                nic_id: *dst_nic,
                                rail_id: *rail_id,
                            },
                            ResourceLinkProfile {
                                kind: RoutedResourceKind::InterNodeFabric,
                                bandwidth: effective_inter_node_bandwidth(
                                    cluster,
                                    src,
                                    *src_nic,
                                    dst,
                                    *dst_nic,
                                    link.profile.bw.unidirectional,
                                ),
                                latency_s: effective_inter_node_latency_s(
                                    cluster,
                                    src,
                                    *src_nic,
                                    dst,
                                    *dst_nic,
                                    link.profile.latency.to_us() / 1e6,
                                ),
                                rail_id: Some(*rail_id),
                                label: format!(
                                    "{} node {src} <-> node {dst} rail {rail_id}",
                                    link.label
                                ),
                            },
                        );
                    }
                }
            }
        }
    }

    fn add_gpu_scoped_inter_node_edges(
        &mut self,
        cluster: &Cluster,
        src: NodeId,
        dst: NodeId,
        link: &GraphInterNodeLink,
    ) {
        let Some((src_gpus, dst_gpus)) = scoped_link_gpus(cluster, src, dst, link) else {
            return;
        };
        for src_gpu in &src_gpus {
            for dst_gpu in &dst_gpus {
                let Some((bandwidth, rail_id)) = effective_gpu_scoped_link_bandwidth(
                    cluster, src, *src_gpu, dst, *dst_gpu, link,
                ) else {
                    continue;
                };
                let rail_label = rail_id
                    .map(|rail_id| format!("rail {rail_id}"))
                    .unwrap_or_else(|| "rail unknown".to_string());
                let (src_latency_s, src_bandwidth) =
                    crate::solver::Solver::intra_node_link_profile(cluster, src);
                let (dst_latency_s, dst_bandwidth) =
                    crate::solver::Solver::intra_node_link_profile(cluster, dst);
                let bandwidth = Bandwidth::from_bytes_per_sec(
                    bandwidth
                        .as_bytes_per_sec()
                        .min(src_bandwidth.as_bytes_per_sec())
                        .min(dst_bandwidth.as_bytes_per_sec())
                        .max(1.0),
                );
                self.add_bidirectional(
                    GraphResource::Gpu(GpuAddr {
                        node_id: src,
                        local_gpu_id: *src_gpu,
                    }),
                    GraphResource::Gpu(GpuAddr {
                        node_id: dst,
                        local_gpu_id: *dst_gpu,
                    }),
                    ResourceLinkProfile {
                        kind: RoutedResourceKind::GpuScopedInterNodeFabric,
                        bandwidth,
                        latency_s: src_latency_s
                            + link.profile.latency.to_us() / 1e6
                            + dst_latency_s,
                        rail_id,
                        label: format!(
                            "{} node {src} gpu {src_gpu} <-> node {dst} gpu {dst_gpu} {rail_label}",
                            link.label
                        ),
                    },
                );
            }
        }
    }

    fn add_bidirectional(
        &mut self,
        from: GraphResource,
        to: GraphResource,
        profile: ResourceLinkProfile,
    ) {
        let ResourceLinkProfile {
            kind,
            bandwidth,
            latency_s,
            rail_id,
            label,
        } = profile;
        self.edges
            .entry(from.clone())
            .or_default()
            .push(ResourceLink {
                from: from.clone(),
                to: to.clone(),
                kind,
                bandwidth,
                latency_s,
                rail_id,
                label: label.clone(),
            });
        self.edges
            .entry(to.clone())
            .or_default()
            .push(ResourceLink {
                from: to,
                to: from,
                kind,
                bandwidth,
                latency_s,
                rail_id,
                label,
            });
    }

    fn nics_for_node(&self, node_id: NodeId) -> Vec<GraphResource> {
        let mut nics = BTreeSet::new();
        for node in self.edges.keys() {
            if let GraphResource::Nic {
                node_id: nic_node_id,
                ..
            } = node
                && *nic_node_id == node_id
            {
                nics.insert(node.clone());
            }
        }

        nics.into_iter().collect()
    }
}

fn inter_node_link_profiles(
    cluster: &Cluster,
    src: NodeId,
    dst: NodeId,
) -> Vec<GraphInterNodeLink> {
    match &cluster.inter_node_topology {
        InterNodeTopology::FatTree {
            link,
            oversubscription,
            ..
        } => {
            let mut profile = link.clone();
            profile.bw.unidirectional = profile.bw.unidirectional / oversubscription.max(1.0);
            vec![GraphInterNodeLink {
                profile,
                rail: None,
                endpoints: None,
                label: "fat-tree fabric".to_string(),
            }]
        }
        InterNodeTopology::Flat { link } => vec![GraphInterNodeLink {
            profile: link.clone(),
            rail: None,
            endpoints: None,
            label: "flat fabric".to_string(),
        }],
        InterNodeTopology::Custom(edges) => edges
            .get(&UnorderedPair::new(src, dst))
            .into_iter()
            .flatten()
            .map(|link| GraphInterNodeLink {
                profile: link.profile.clone(),
                rail: link.rail,
                endpoints: link.endpoints.clone(),
                label: format!("custom {}", link.profile.label),
            })
            .collect(),
    }
}

fn scoped_link_gpus(
    cluster: &Cluster,
    src: NodeId,
    dst: NodeId,
    link: &GraphInterNodeLink,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let endpoints = link.endpoints.as_ref()?;
    let (src_scope, dst_scope) = if endpoints.from_node == src && endpoints.to_node == dst {
        (&endpoints.from_gpus, &endpoints.to_gpus)
    } else if endpoints.from_node == dst && endpoints.to_node == src {
        (&endpoints.to_gpus, &endpoints.from_gpus)
    } else {
        return None;
    };
    let src_gpus = scoped_or_available_gpus(cluster, src, src_scope);
    let dst_gpus = scoped_or_available_gpus(cluster, dst, dst_scope);
    if src_gpus.is_empty() || dst_gpus.is_empty() {
        None
    } else {
        Some((src_gpus, dst_gpus))
    }
}

fn scoped_or_available_gpus(cluster: &Cluster, node_id: NodeId, scope: &[u32]) -> Vec<u32> {
    if !scope.is_empty() {
        return scope
            .iter()
            .copied()
            .filter(|gpu_id| {
                cluster.is_gpu_available(GpuAddr {
                    node_id,
                    local_gpu_id: *gpu_id,
                })
            })
            .collect();
    }

    let Some(node) = cluster.node(node_id) else {
        return Vec::new();
    };
    let mut gpus: Vec<_> = node
        .gpus
        .keys()
        .copied()
        .filter(|gpu_id| node.is_gpu_available(*gpu_id))
        .collect();
    gpus.sort_unstable();
    gpus
}

fn effective_gpu_scoped_link_bandwidth(
    cluster: &Cluster,
    src: NodeId,
    src_gpu: u32,
    dst: NodeId,
    dst_gpu: u32,
    link: &GraphInterNodeLink,
) -> Option<(Bandwidth, Option<u32>)> {
    let src_node = cluster.node(src)?;
    let dst_node = cluster.node(dst)?;
    let src_nics = enabled_gpu_nics_on_link_rail(cluster, src, src_gpu, link.rail);
    let dst_nics = enabled_gpu_nics_on_link_rail(cluster, dst, dst_gpu, link.rail);
    let mut best = None;

    for src_nic in &src_nics {
        for dst_nic in &dst_nics {
            let rail = src_node.network.rail_id(*src_nic);
            let dst_rail = dst_node.network.rail_id(*dst_nic);
            if link.rail.is_none() && rail != dst_rail {
                continue;
            }
            let bandwidth = effective_inter_node_bandwidth(
                cluster,
                src,
                *src_nic,
                dst,
                *dst_nic,
                link.profile.bw.unidirectional,
            );
            let rail_id = link.rail.or(Some(rail));
            if best
                .as_ref()
                .is_none_or(|(current, _): &(Bandwidth, Option<u32>)| {
                    bandwidth.as_bytes_per_sec() > current.as_bytes_per_sec()
                })
            {
                best = Some((bandwidth, rail_id));
            }
        }
    }

    if best.is_none() && link.rail.is_none() {
        for src_nic in &src_nics {
            for dst_nic in &dst_nics {
                let bandwidth = effective_inter_node_bandwidth(
                    cluster,
                    src,
                    *src_nic,
                    dst,
                    *dst_nic,
                    link.profile.bw.unidirectional,
                );
                let rail = src_node.network.rail_id(*src_nic);
                if best
                    .as_ref()
                    .is_none_or(|(current, _): &(Bandwidth, Option<u32>)| {
                        bandwidth.as_bytes_per_sec() > current.as_bytes_per_sec()
                    })
                {
                    best = Some((bandwidth, Some(rail)));
                }
            }
        }
    }

    best
}

fn enabled_gpu_nics_on_link_rail(
    cluster: &Cluster,
    node_id: NodeId,
    gpu_id: u32,
    link_rail: Option<u32>,
) -> Vec<NicId> {
    let Some(node) = cluster.node(node_id) else {
        return Vec::new();
    };
    node.network
        .nic_candidates_for_gpu(gpu_id)
        .into_iter()
        .filter(|nic_id| node.network.is_nic_available(*nic_id))
        .filter(|nic_id| link_rail.is_none_or(|rail| node.network.rail_id(*nic_id) == rail))
        .collect()
}

fn matching_nic_pairs(cluster: &Cluster, src: NodeId, dst: NodeId) -> Vec<(NicId, NicId, u32)> {
    let Some(src_node) = cluster.node(src) else {
        return Vec::new();
    };
    let Some(dst_node) = cluster.node(dst) else {
        return Vec::new();
    };

    let mut rail_pairs = Vec::new();
    for src_nic in 0..src_node.network.nic_count as NicId {
        if !src_node.network.is_nic_available(src_nic) {
            continue;
        }
        for dst_nic in 0..dst_node.network.nic_count as NicId {
            if !dst_node.network.is_nic_available(dst_nic) {
                continue;
            }
            let src_rail = src_node.network.rail_id(src_nic);
            let dst_rail = dst_node.network.rail_id(dst_nic);
            if src_rail == dst_rail {
                rail_pairs.push((src_nic, dst_nic, src_rail));
            }
        }
    }

    if !rail_pairs.is_empty() {
        return rail_pairs;
    }

    let mut pairs = Vec::new();
    for src_nic in 0..src_node.network.nic_count as NicId {
        if !src_node.network.is_nic_available(src_nic) {
            continue;
        }
        for dst_nic in 0..dst_node.network.nic_count as NicId {
            if !dst_node.network.is_nic_available(dst_nic) {
                continue;
            }
            pairs.push((src_nic, dst_nic, src_node.network.rail_id(src_nic)));
        }
    }

    pairs
}

fn nic_candidates(network: &NodeNetworkProfile, gpu_id: u32) -> Vec<NicId> {
    network.nic_candidates_for_gpu(gpu_id)
}

fn gpu_nic_edge_profile(
    network: &NodeNetworkProfile,
    gpu_id: u32,
    nic_id: NicId,
    default_latency_s: f64,
    default_bandwidth: Bandwidth,
) -> (f64, Bandwidth, String) {
    let locality = gpu_nic_numa_locality(network, gpu_id, nic_id);
    let Some(path) = network.gpu_nic_path(gpu_id, nic_id) else {
        return (
            default_latency_s * locality.latency_scale * network.nic_latency_scale(nic_id),
            default_bandwidth * locality.bandwidth_scale,
            gpu_nic_locality_label(locality.label.as_deref()),
        );
    };

    let latency_s = path
        .latency
        .map(|latency| latency.to_us() / 1e6)
        .unwrap_or(default_latency_s * locality.latency_scale)
        * network.nic_latency_scale(nic_id);
    let bandwidth = path
        .bandwidth
        .unwrap_or(default_bandwidth * locality.bandwidth_scale);
    let label = gpu_nic_path_label(path, locality.label.as_deref());
    (latency_s, bandwidth, label)
}

#[derive(Clone, Debug, Default, PartialEq)]
struct GpuNicNumaLocality {
    bandwidth_scale: f64,
    latency_scale: f64,
    label: Option<String>,
}

fn gpu_nic_numa_locality(
    network: &NodeNetworkProfile,
    gpu_id: u32,
    nic_id: NicId,
) -> GpuNicNumaLocality {
    let Some((gpu_domain, nic_domain)) = network.gpu_nic_numa_domains(gpu_id, nic_id) else {
        return GpuNicNumaLocality {
            bandwidth_scale: 1.0,
            latency_scale: 1.0,
            label: None,
        };
    };
    if gpu_domain == nic_domain {
        GpuNicNumaLocality {
            bandwidth_scale: 1.0,
            latency_scale: 1.0,
            label: Some(format!("same_numa:{gpu_domain}")),
        }
    } else {
        GpuNicNumaLocality {
            bandwidth_scale: network.cross_numa_bandwidth_scale,
            latency_scale: network.cross_numa_latency_scale,
            label: Some(format!("cross_socket:numa{gpu_domain}->numa{nic_domain}")),
        }
    }
}

fn gpu_nic_locality_label(locality_label: Option<&str>) -> String {
    locality_label
        .filter(|label| !label.trim().is_empty())
        .map(|label| format!(" path {}", label.trim()))
        .unwrap_or_default()
}

fn gpu_nic_path_label(path: &GpuNicPathOverride, locality_label: Option<&str>) -> String {
    let mut parts = Vec::new();
    if let Some(label) = path.label.as_deref()
        && !label.trim().is_empty()
    {
        parts.push(label.trim().to_string());
    }
    if let Some(label) = locality_label
        && !label.trim().is_empty()
    {
        parts.push(label.trim().to_string());
    }
    if let Some(gpudirect) = path.gpudirect {
        parts.push(if gpudirect {
            "gpudirect".to_string()
        } else {
            "host-staged".to_string()
        });
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" path {}", parts.join("/"))
    }
}

fn effective_inter_node_bandwidth(
    cluster: &Cluster,
    src: NodeId,
    src_nic: NicId,
    dst: NodeId,
    dst_nic: NicId,
    fabric_bandwidth: Bandwidth,
) -> Bandwidth {
    let mut bytes_per_sec = fabric_bandwidth.as_bytes_per_sec();
    if let Some(src_node) = cluster.node(src) {
        bytes_per_sec =
            bytes_per_sec.min(src_node.network.nic_bandwidth(src_nic).as_bytes_per_sec());
    }
    if let Some(dst_node) = cluster.node(dst) {
        bytes_per_sec =
            bytes_per_sec.min(dst_node.network.nic_bandwidth(dst_nic).as_bytes_per_sec());
    }
    Bandwidth::from_bytes_per_sec(bytes_per_sec.max(1.0))
}

fn effective_inter_node_latency_s(
    cluster: &Cluster,
    src: NodeId,
    src_nic: NicId,
    dst: NodeId,
    dst_nic: NicId,
    fabric_latency_s: f64,
) -> f64 {
    let src_scale = cluster
        .node(src)
        .map(|node| node.network.nic_latency_scale(src_nic))
        .unwrap_or(1.0);
    let dst_scale = cluster
        .node(dst)
        .map(|node| node.network.nic_latency_scale(dst_nic))
        .unwrap_or(1.0);
    fabric_latency_s * src_scale.max(dst_scale)
}

fn is_better_path(candidate: &RoutedPath, current: Option<&RoutedPath>, bytes: u64) -> bool {
    let Some(current) = current else {
        return true;
    };

    let candidate_cost =
        candidate.latency_s + bytes as f64 / candidate.bottleneck_bandwidth.as_bytes_per_sec();
    let current_cost =
        current.latency_s + bytes as f64 / current.bottleneck_bandwidth.as_bytes_per_sec();

    candidate_cost < current_cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::parse_cluster,
        types::{fabric::variants::ib::IbVariant, topology::Cluster},
    };

    #[test]
    fn derives_gpu_to_gpu_route_with_nic_and_fabric_labels() {
        let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
        let graph = TopologyGraph::from_cluster(&cluster);
        let path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();

        assert!(path.latency_s > 0.0);
        assert!(
            path.labels
                .iter()
                .any(|label| label.contains("fat-tree fabric"))
        );
        assert!(path.resources.iter().any(|resource| {
            resource.kind == RoutedResourceKind::GpuNicLocal && resource.rail_id.is_some()
        }));
        assert!(path.resources.iter().any(|resource| {
            resource.kind == RoutedResourceKind::InterNodeFabric
                && resource.rail_id.is_some()
                && matches!(
                    (&resource.from, &resource.to),
                    (GraphResource::Nic { .. }, GraphResource::Nic { .. })
                )
        }));
    }

    #[test]
    fn routes_custom_heterogeneous_link_by_pair_and_rail() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "nvlink_v4"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "a100_80gb"
            gpu_count = 2
            intra = "nvlink_v3"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 200.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);
        let path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 1,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 1,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();

        assert!(path.labels.iter().any(|label| label.contains("IB HDR")));
        assert!(path.labels.iter().any(|label| label.contains("rail 1")));
    }

    #[test]
    fn routes_custom_links_scoped_to_matching_rails() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ethernet"
            variant = "100g"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "nvlink_v4"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "nvlink_v4"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        let rail_zero_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            rail_zero_path
                .labels
                .iter()
                .any(|label| label.contains("Ethernet 100G"))
        );

        let rail_one_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 1,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 1,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            rail_one_path
                .labels
                .iter()
                .any(|label| label.contains("IB HDR"))
        );
        assert!(
            !rail_one_path
                .labels
                .iter()
                .any(|label| label.contains("Ethernet 100G"))
        );
    }

    #[test]
    fn routes_custom_links_scoped_to_gpu_subsets() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu = 0
            to_gpu = 0
            kind = "ethernet"
            variant = "100g"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu = 1
            to_gpu = 1
            kind = "ib"
            variant = "hdr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "nvlink_v4"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "nvlink_v4"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        let gpu_zero_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            gpu_zero_path
                .labels
                .iter()
                .any(|label| label.contains("Ethernet 100G"))
        );
        assert!(
            !gpu_zero_path
                .labels
                .iter()
                .any(|label| label.contains("IB HDR"))
        );

        let gpu_one_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 1,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 1,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            gpu_one_path
                .labels
                .iter()
                .any(|label| label.contains("IB HDR"))
        );
        assert!(
            !gpu_one_path
                .labels
                .iter()
                .any(|label| label.contains("Ethernet 100G"))
        );

        assert!(
            graph
                .route_between_gpus(
                    GpuAddr {
                        node_id: 0,
                        local_gpu_id: 0,
                    },
                    GpuAddr {
                        node_id: 1,
                        local_gpu_id: 1,
                    },
                    Bytes::from_megabytes(1.0),
                )
                .is_none()
        );
    }

    #[test]
    fn routes_custom_links_scoped_to_gpu_tags() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu_tag = "rail-a"
            to_gpu_tag = "rail-a"
            kind = "ethernet"
            variant = "100g"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu_tag = "rail-b"
            to_gpu_tag = "rail-b"
            kind = "ib"
            variant = "hdr"

            [[nodes]]
            id = 0
            intra = "pcie_gen5"
            gpus = [
              { id = 0, gpu = "h100_sxm", labels = ["rail-a"] },
              { id = 1, gpu = "h100_sxm", labels = ["rail-b"] },
            ]
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            intra = "pcie_gen5"
            gpus = [
              { id = 0, gpu = "h100_sxm", labels = ["rail-a"] },
              { id = 1, gpu = "h100_sxm", labels = ["rail-b"] },
            ]
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        let rail_a_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            rail_a_path
                .labels
                .iter()
                .any(|label| label.contains("Ethernet 100G"))
        );
        assert!(
            !rail_a_path
                .labels
                .iter()
                .any(|label| label.contains("IB HDR"))
        );

        let rail_b_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 1,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 1,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            rail_b_path
                .labels
                .iter()
                .any(|label| label.contains("IB HDR"))
        );
        assert!(
            graph
                .route_between_gpus(
                    GpuAddr {
                        node_id: 0,
                        local_gpu_id: 0,
                    },
                    GpuAddr {
                        node_id: 1,
                        local_gpu_id: 1,
                    },
                    Bytes::from_megabytes(1.0),
                )
                .is_none()
        );
    }

    #[test]
    fn explicit_nic_rail_map_controls_rail_scoped_routes() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ethernet"
            variant = "100g"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "nvlink_v4"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2, nic_rail_map = [{ nic = 0, rail = 1 }, { nic = 1, rail = 0 }] }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "nvlink_v4"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2, nic_rail_map = [{ nic = 0, rail = 1 }, { nic = 1, rail = 0 }] }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        let gpu_zero_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            gpu_zero_path
                .labels
                .iter()
                .any(|label| label.contains("IB HDR"))
        );
        assert!(
            !gpu_zero_path
                .labels
                .iter()
                .any(|label| label.contains("Ethernet 100G"))
        );

        let gpu_one_path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 1,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 1,
                },
                Bytes::from_megabytes(1.0),
            )
            .unwrap();
        assert!(
            gpu_one_path
                .labels
                .iter()
                .any(|label| label.contains("Ethernet 100G"))
        );
        assert!(
            !gpu_one_path
                .labels
                .iter()
                .any(|label| label.contains("IB HDR"))
        );
    }

    #[test]
    fn gpu_nic_path_overrides_affect_route_selection() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 0

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1, gpu_nic_paths = [{ gpu = 0, nic = 0, label = "host_staged", bandwidth_gbps = 10.0, latency_us = 100.0, gpudirect = false }, { gpu = 0, nic = 1, label = "same_pcie", bandwidth_gbps = 300.0, latency_us = 1.0, gpudirect = true }] }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 1
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1, gpu_nic_paths = [{ gpu = 0, nic = 0, label = "host_staged", bandwidth_gbps = 10.0, latency_us = 100.0, gpudirect = false }, { gpu = 0, nic = 1, label = "same_pcie", bandwidth_gbps = 300.0, latency_us = 1.0, gpudirect = true }] }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        let path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
                Bytes::from_megabytes(1024.0),
            )
            .unwrap();

        assert!(
            path.labels
                .iter()
                .any(|label| label.contains("same_pcie/gpudirect"))
        );
        assert!(
            !path
                .labels
                .iter()
                .any(|label| label.contains("host_staged"))
        );
        assert!(
            (path.bottleneck_bandwidth.as_gigabits_per_sec() - 200.0).abs() < 1e-9,
            "expected the IB HDR fabric to bottleneck after choosing the faster GPU-NIC path"
        );
    }

    #[test]
    fn numa_maps_affect_default_gpu_nic_route_selection() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 0

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1, gpu_numa_map = [{ gpu = 0, domain = 0 }], nic_numa_map = [{ nic = 0, domain = 1 }, { nic = 1, domain = 0 }], cross_numa_bandwidth_scale = 0.05, cross_numa_latency_scale = 10.0 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 1
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1, gpu_numa_map = [{ gpu = 0, domain = 0 }], nic_numa_map = [{ nic = 0, domain = 1 }, { nic = 1, domain = 0 }], cross_numa_bandwidth_scale = 0.05, cross_numa_latency_scale = 10.0 }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        let path = graph
            .route_between_gpus(
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
                Bytes::from_megabytes(1024.0),
            )
            .unwrap();

        assert!(path.labels.iter().any(|label| label.contains("same_numa")));
        assert!(
            !path
                .labels
                .iter()
                .any(|label| label.contains("cross_socket"))
        );
    }

    #[test]
    fn explicit_gpu_nic_locality_controls_routability() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2, gpu_nic_map = [{ gpu = 1, nic = 0 }] }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2, gpu_nic_map = [{ gpu = 1, nic = 0 }] }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        assert!(
            graph
                .route_between_gpus(
                    GpuAddr {
                        node_id: 0,
                        local_gpu_id: 1,
                    },
                    GpuAddr {
                        node_id: 1,
                        local_gpu_id: 1,
                    },
                    Bytes::from_megabytes(1.0),
                )
                .is_none()
        );
    }

    #[test]
    fn disabled_nics_are_not_routed() {
        let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2, disabled_nics = [1], gpu_nic_map = [{ gpu = 1, nics = [0, 1] }] }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let graph = TopologyGraph::from_cluster(&cluster);

        assert!(
            graph
                .route_between_gpus(
                    GpuAddr {
                        node_id: 0,
                        local_gpu_id: 1,
                    },
                    GpuAddr {
                        node_id: 1,
                        local_gpu_id: 1,
                    },
                    Bytes::from_megabytes(1.0),
                )
                .is_none()
        );
    }

    #[test]
    fn per_nic_bandwidth_overrides_affect_route_cost() {
        let mut cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();

        cluster
            .nodes
            .get_mut(&0)
            .unwrap()
            .network
            .nic_bandwidth_overrides
            .insert(0, Bandwidth::from_gigabits_per_sec(25.0));
        let graph = TopologyGraph::from_cluster(&cluster);
        let path = graph
            .route_between_nodes(0, 1, Bytes::from_gigabytes(1.0))
            .expect("route should remain available through non-degraded rail");

        assert!(path.labels.iter().any(|label| label.contains("rail 1")));
        assert!(!path.labels.iter().any(|label| label.contains("rail 0")));
        assert!(path.bottleneck_bandwidth.as_gigabits_per_sec() > 300.0);
    }

    #[test]
    fn per_nic_latency_overrides_affect_route_cost() {
        let mut cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();

        cluster
            .nodes
            .get_mut(&0)
            .unwrap()
            .network
            .nic_latency_scale_overrides
            .insert(0, 100.0);
        let graph = TopologyGraph::from_cluster(&cluster);
        let path = graph
            .route_between_nodes(0, 1, Bytes::from_bytes(1))
            .expect("route should remain available through lower-latency rail");

        assert!(path.labels.iter().any(|label| label.contains("rail 1")));
        assert!(!path.labels.iter().any(|label| label.contains("rail 0")));
    }
}
