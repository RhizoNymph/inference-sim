//! `[[collective_curves]]` in the cluster TOML: measured latency curves for
//! collectives and point-to-point transfers. Curves are fabric facts tied to
//! the cluster's node ids, so they live next to the nodes and links they
//! describe rather than in a workload or calibration profile.

use serde::Deserialize;

use super::ConfigError;
use crate::types::{
    collective_curves::{
        CollectiveCurve, CollectiveCurveOp, CollectiveCurveSet, CollectiveScope, CurveTarget,
        DirectedNodePair, IntraNodeSelector, MeasuredCurve, NodeGroup, PointToPointScope,
        RankCount,
    },
    topology::Cluster,
};

/// A latency written as a TOML integer or float.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum LatencyUs {
    Integer(i64),
    Float(f64),
}

impl LatencyUs {
    fn value(self) -> f64 {
        match self {
            Self::Integer(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CollectiveCurveSection {
    pub(super) op: String,
    pub(super) scope: String,
    pub(super) nodes: Option<Vec<u32>>,
    pub(super) src_node: Option<u32>,
    pub(super) dst_node: Option<u32>,
    pub(super) ranks: Option<u32>,
    pub(super) source: Option<String>,
    pub(super) derived_below_bytes: Option<u64>,
    /// `[message_bytes_per_rank, latency_us]` pairs.
    pub(super) points: Vec<(u64, LatencyUs)>,
}

/// Parsed operation of a section: a collective or point-to-point.
enum SectionOp {
    Collective(CollectiveCurveOp),
    PointToPoint,
}

pub(super) fn parse_collective_curves(
    sections: Option<Vec<CollectiveCurveSection>>,
    cluster: &Cluster,
) -> Result<CollectiveCurveSet, ConfigError> {
    let sections = sections.unwrap_or_default();
    let mut curves = Vec::with_capacity(sections.len());
    for (idx, section) in sections.into_iter().enumerate() {
        curves.push(parse_curve(idx, section)?);
    }
    let set = CollectiveCurveSet::new(curves).map_err(|err| ConfigError::new(err.to_string()))?;
    for (idx, node) in set.referenced_nodes() {
        if cluster.node(node).is_none() {
            return Err(ConfigError::new(format!(
                "collective_curves[{idx}] references unknown node {node}"
            )));
        }
    }
    Ok(set)
}

fn parse_curve(
    idx: usize,
    section: CollectiveCurveSection,
) -> Result<CollectiveCurve, ConfigError> {
    let name = format!("collective_curves[{idx}]");
    let error = |message: String| ConfigError::new(format!("{name}: {message}"));
    let op = parse_op(&section.op).ok_or_else(|| {
        error(format!(
            "unsupported op '{}'; use all_reduce, all_gather, reduce_scatter, all_to_all, broadcast, or send_recv",
            section.op
        ))
    })?;
    let target = match op {
        SectionOp::Collective(op) => {
            reject_present(&name, "src_node", section.src_node.is_some(), "collectives")?;
            reject_present(&name, "dst_node", section.dst_node.is_some(), "collectives")?;
            let ranks = section
                .ranks
                .ok_or_else(|| error("ranks is required for collective curves".to_string()))?;
            let ranks = RankCount::new(ranks).map_err(|err| error(err.to_string()))?;
            let scope = match normalize(&section.scope).as_str() {
                "node_group" => {
                    let nodes = section
                        .nodes
                        .as_deref()
                        .ok_or_else(|| error("scope node_group requires nodes".to_string()))?;
                    CollectiveScope::NodeGroup(
                        NodeGroup::new(nodes).map_err(|err| error(err.to_string()))?,
                    )
                }
                "inter_node" => {
                    reject_present(&name, "nodes", section.nodes.is_some(), "scope inter_node")?;
                    CollectiveScope::InterNode
                }
                "intra_node" => CollectiveScope::IntraNode(
                    IntraNodeSelector::new(section.nodes.as_deref())
                        .map_err(|err| error(err.to_string()))?,
                ),
                other => {
                    return Err(error(format!(
                        "unsupported scope '{other}' for {}; use node_group, inter_node, or intra_node",
                        section.op
                    )));
                }
            };
            CurveTarget::Collective { op, scope, ranks }
        }
        SectionOp::PointToPoint => {
            if let Some(ranks) = section.ranks
                && ranks != 2
            {
                return Err(error(format!(
                    "send_recv curves are always 2 ranks; ranks = {ranks} is invalid"
                )));
            }
            let scope = match normalize(&section.scope).as_str() {
                "node_pair" => {
                    reject_present(&name, "nodes", section.nodes.is_some(), "scope node_pair")?;
                    let (Some(src), Some(dst)) = (section.src_node, section.dst_node) else {
                        return Err(error(
                            "scope node_pair requires src_node and dst_node".to_string(),
                        ));
                    };
                    PointToPointScope::NodePair(
                        DirectedNodePair::new(src, dst).map_err(|err| error(err.to_string()))?,
                    )
                }
                "inter_node" => {
                    reject_present(&name, "nodes", section.nodes.is_some(), "scope inter_node")?;
                    reject_endpoints(&name, &section)?;
                    PointToPointScope::InterNode
                }
                "intra_node" => {
                    reject_endpoints(&name, &section)?;
                    PointToPointScope::IntraNode(
                        IntraNodeSelector::new(section.nodes.as_deref())
                            .map_err(|err| error(err.to_string()))?,
                    )
                }
                other => {
                    return Err(error(format!(
                        "unsupported scope '{other}' for send_recv; use node_pair, inter_node, or intra_node"
                    )));
                }
            };
            CurveTarget::PointToPoint { scope }
        }
    };
    let points: Vec<(u64, f64)> = section
        .points
        .iter()
        .map(|(bytes, latency)| (*bytes, latency.value()))
        .collect();
    let curve = MeasuredCurve::from_microseconds(&points, section.derived_below_bytes)
        .map_err(|err| error(err.to_string()))?;
    let source = section
        .source
        .map(|source| source.trim().to_string())
        .filter(|source| !source.is_empty());
    Ok(CollectiveCurve::new(target, curve, source))
}

fn parse_op(value: &str) -> Option<SectionOp> {
    Some(match normalize(value).as_str() {
        "all_reduce" | "allreduce" => SectionOp::Collective(CollectiveCurveOp::AllReduce),
        "all_gather" | "allgather" => SectionOp::Collective(CollectiveCurveOp::AllGather),
        "reduce_scatter" | "reducescatter" => {
            SectionOp::Collective(CollectiveCurveOp::ReduceScatter)
        }
        "all_to_all" | "alltoall" => SectionOp::Collective(CollectiveCurveOp::AllToAll),
        "broadcast" => SectionOp::Collective(CollectiveCurveOp::Broadcast),
        "send_recv" | "sendrecv" | "p2p" | "point_to_point" => SectionOp::PointToPoint,
        _ => return None,
    })
}

fn normalize(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn reject_present(
    name: &str,
    field: &str,
    present: bool,
    context: &str,
) -> Result<(), ConfigError> {
    if present {
        return Err(ConfigError::new(format!(
            "{name}: {field} is not allowed for {context}"
        )));
    }
    Ok(())
}

fn reject_endpoints(name: &str, section: &CollectiveCurveSection) -> Result<(), ConfigError> {
    reject_present(name, "src_node", section.src_node.is_some(), "this scope")?;
    reject_present(name, "dst_node", section.dst_node.is_some(), "this scope")
}

#[cfg(test)]
mod tests {
    use crate::{
        config::parse_cluster,
        types::{
            collective::CollectiveKind,
            collective_curves::{CurveLookup, CurveQuery},
        },
    };

    const BASE: &str = r#"
schema_version = 1
[cluster]
preset = "custom"
[interconnect]
kind = "ethernet"
variant = "10g"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 1
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
[[nodes]]
id = 1
gpu = "a100_40gb"
gpu_count = 1
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
"#;

    fn parse(extra: &str) -> Result<crate::types::topology::Cluster, String> {
        parse_cluster(&format!("{BASE}{extra}")).map_err(|err| err.to_string())
    }

    fn lookup_floor(
        cluster: &crate::types::topology::Cluster,
        kind: CollectiveKind,
        nodes: &[u32],
    ) -> Option<f64> {
        match cluster.collective_curves.lookup(&CurveQuery {
            kind,
            participant_nodes: nodes,
        }) {
            CurveLookup::Matched(curve) => Some(curve.curve().floor_s()),
            _ => None,
        }
    }

    #[test]
    fn clusters_without_curves_parse_unchanged() {
        let cluster = parse("").expect("cluster");
        assert!(cluster.collective_curves.is_empty());
    }

    #[test]
    fn parses_collective_and_directed_point_to_point_curves() {
        let cluster = parse(
            r#"
[[collective_curves]]
op = "all_reduce"
scope = "node_group"
nodes = [0, 1]
ranks = 2
source = "lab-runs/x.jsonl"
points = [[1024, 132.66], [2048, 133]]

[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
derived_below_bytes = 4096
points = [[1024, 160.0], [4096, 170.0], [8192, 200.0]]

[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 1
dst_node = 0
points = [[1024, 150.0], [8192, 170.0]]
"#,
        )
        .expect("cluster");
        let curves = cluster.collective_curves.curves();
        assert_eq!(curves.len(), 3);
        assert_eq!(curves[0].source(), Some("lab-runs/x.jsonl"));
        assert_eq!(curves[0].curve().points()[1].latency_s(), 133.0e-6);
        assert_eq!(curves[1].curve().derived_below_bytes(), Some(4096));
        assert_eq!(
            lookup_floor(&cluster, CollectiveKind::AllReduce, &[0, 1]),
            Some(132.66e-6)
        );
        assert_eq!(
            lookup_floor(&cluster, CollectiveKind::SendRecv, &[0, 1]),
            Some(160.0e-6)
        );
        assert_eq!(
            lookup_floor(&cluster, CollectiveKind::SendRecv, &[1, 0]),
            Some(150.0e-6)
        );
    }

    #[test]
    fn rejects_invalid_curve_sections() {
        let cases = [
            (
                r#"op = "all_reduce"
scope = "node_group"
nodes = [0, 1]
points = [[1024, 1.0], [2048, 2.0]]"#,
                "ranks is required",
            ),
            (
                r#"op = "all_reduce"
scope = "node_group"
nodes = [0]
ranks = 2
points = [[1024, 1.0], [2048, 2.0]]"#,
                "at least 2 distinct nodes",
            ),
            (
                r#"op = "all_reduce"
scope = "node_pair"
ranks = 2
points = [[1024, 1.0], [2048, 2.0]]"#,
                "unsupported scope 'node_pair'",
            ),
            (
                r#"op = "send_recv"
scope = "node_group"
nodes = [0, 1]
points = [[1024, 1.0], [2048, 2.0]]"#,
                "unsupported scope 'node_group'",
            ),
            (
                r#"op = "send_recv"
scope = "node_pair"
src_node = 0
points = [[1024, 1.0], [2048, 2.0]]"#,
                "requires src_node and dst_node",
            ),
            (
                r#"op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 0
points = [[1024, 1.0], [2048, 2.0]]"#,
                "must differ",
            ),
            (
                r#"op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
ranks = 4
points = [[1024, 1.0], [2048, 2.0]]"#,
                "always 2 ranks",
            ),
            (
                r#"op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 7
points = [[1024, 1.0], [2048, 2.0]]"#,
                "unknown node 7",
            ),
            (
                r#"op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
points = [[2048, 1.0], [1024, 2.0]]"#,
                "must be greater than the previous",
            ),
            (
                r#"op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
points = [[1024, 1.0]]"#,
                "at least 2 points",
            ),
            (
                r#"op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
points = [[1024, 0.0], [2048, 1.0]]"#,
                "finite and positive",
            ),
            (
                r#"op = "send_recv"
scope = "inter_node"
src_node = 0
points = [[1024, 1.0], [2048, 1.0]]"#,
                "src_node is not allowed",
            ),
            (
                r#"op = "all_reduce"
scope = "inter_node"
nodes = [0, 1]
ranks = 2
points = [[1024, 1.0], [2048, 1.0]]"#,
                "nodes is not allowed",
            ),
            (
                r#"op = "gather"
scope = "inter_node"
ranks = 2
points = [[1024, 1.0], [2048, 1.0]]"#,
                "unsupported op 'gather'",
            ),
            (
                r#"op = "all_reduce"
scope = "inter_node"
ranks = 2
latency = 3
points = [[1024, 1.0], [2048, 1.0]]"#,
                "unknown field",
            ),
        ];
        for (body, expected) in cases {
            let error = parse(&format!("\n[[collective_curves]]\n{body}\n"))
                .expect_err(&format!("should reject: {body}"));
            assert!(
                error.contains(expected),
                "error for\n{body}\nwas '{error}', expected '{expected}'"
            );
        }
    }

    #[test]
    fn rejects_duplicate_targets() {
        let curve = r#"
[[collective_curves]]
op = "send_recv"
scope = "node_pair"
src_node = 0
dst_node = 1
points = [[1024, 1.0], [2048, 2.0]]
"#;
        let error = parse(&format!("{curve}{curve}")).expect_err("duplicate");
        assert!(
            error.contains("both price 'send_recv node0->node1'"),
            "{error}"
        );
    }

    #[test]
    fn preset_clusters_accept_curves() {
        let cluster = parse_cluster(
            r#"
schema_version = 1
[cluster]
preset = "h100_sxm"
node_count = 2
[interconnect]
kind = "ib"
variant = "ndr"
[[collective_curves]]
op = "all_reduce"
scope = "intra_node"
ranks = 8
points = [[1024, 20.0], [1048576, 60.0]]
"#,
        )
        .expect("cluster");
        assert_eq!(cluster.collective_curves.curves().len(), 1);
    }
}
