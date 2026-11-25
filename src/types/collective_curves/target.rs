//! What a measured curve applies to: an operation, a placement scope, and
//! (for collectives) a rank count. Constructors enforce the invariants so a
//! `CurveTarget` can only describe a placement the simulator can match.

use std::{
    collections::BTreeSet,
    fmt::{Display, Formatter},
};

use crate::types::{collective::CollectiveKind, common::NodeId};

/// Collective operations a curve can describe. Point-to-point transfers are
/// a separate target kind because they are directed and always two ranks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CollectiveCurveOp {
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
    Broadcast,
}

impl CollectiveCurveOp {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllReduce => "all_reduce",
            Self::AllGather => "all_gather",
            Self::ReduceScatter => "reduce_scatter",
            Self::AllToAll => "all_to_all",
            Self::Broadcast => "broadcast",
        }
    }

    /// The curve operation pricing a simulator collective, or `None` for
    /// point-to-point (`SendRecv`), which uses point-to-point targets.
    pub fn for_kind(kind: CollectiveKind) -> Option<Self> {
        match kind {
            CollectiveKind::AllReduce => Some(Self::AllReduce),
            CollectiveKind::AllGather => Some(Self::AllGather),
            CollectiveKind::ReduceScatter => Some(Self::ReduceScatter),
            CollectiveKind::AllToAll => Some(Self::AllToAll),
            CollectiveKind::Broadcast => Some(Self::Broadcast),
            CollectiveKind::SendRecv => None,
        }
    }
}

/// Number of ranks taking part in a collective; at least two.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RankCount(u32);

impl RankCount {
    pub fn new(ranks: u32) -> Result<Self, TargetError> {
        if ranks < 2 {
            return Err(TargetError::TooFewRanks { ranks });
        }
        Ok(Self(ranks))
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// An exact set of at least two distinct nodes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeGroup(BTreeSet<NodeId>);

impl NodeGroup {
    pub fn new(nodes: &[NodeId]) -> Result<Self, TargetError> {
        let set: BTreeSet<_> = nodes.iter().copied().collect();
        if set.len() != nodes.len() {
            return Err(TargetError::DuplicateNode {
                nodes: nodes.to_vec(),
            });
        }
        if set.len() < 2 {
            return Err(TargetError::NodeGroupTooSmall {
                nodes: nodes.to_vec(),
            });
        }
        Ok(Self(set))
    }

    pub fn nodes(&self) -> &BTreeSet<NodeId> {
        &self.0
    }
}

/// Which nodes an intra-node curve applies to.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntraNodeSelector {
    AnyNode,
    Nodes(BTreeSet<NodeId>),
}

impl IntraNodeSelector {
    /// `None` selects every node; an explicit list must be non-empty and
    /// duplicate-free.
    pub fn new(nodes: Option<&[NodeId]>) -> Result<Self, TargetError> {
        let Some(nodes) = nodes else {
            return Ok(Self::AnyNode);
        };
        if nodes.is_empty() {
            return Err(TargetError::EmptyIntraNodeSelector);
        }
        let set: BTreeSet<_> = nodes.iter().copied().collect();
        if set.len() != nodes.len() {
            return Err(TargetError::DuplicateNode {
                nodes: nodes.to_vec(),
            });
        }
        Ok(Self::Nodes(set))
    }

    pub fn contains(&self, node: NodeId) -> bool {
        match self {
            Self::AnyNode => true,
            Self::Nodes(nodes) => nodes.contains(&node),
        }
    }

    fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::AnyNode, Self::AnyNode) => true,
            (Self::AnyNode, Self::Nodes(_)) | (Self::Nodes(_), Self::AnyNode) => false,
            (Self::Nodes(a), Self::Nodes(b)) => !a.is_disjoint(b),
        }
    }
}

/// A directed `src -> dst` pair of distinct nodes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DirectedNodePair {
    src: NodeId,
    dst: NodeId,
}

impl DirectedNodePair {
    pub fn new(src: NodeId, dst: NodeId) -> Result<Self, TargetError> {
        if src == dst {
            return Err(TargetError::SelfPair { node: src });
        }
        Ok(Self { src, dst })
    }

    pub fn src(self) -> NodeId {
        self.src
    }

    pub fn dst(self) -> NodeId {
        self.dst
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CollectiveScope {
    /// Exactly these nodes take part.
    NodeGroup(NodeGroup),
    /// Any participant set spanning two or more nodes.
    InterNode,
    /// All participants on one node.
    IntraNode(IntraNodeSelector),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PointToPointScope {
    /// Exactly this direction between two nodes.
    NodePair(DirectedNodePair),
    /// Any pair of distinct nodes, either direction.
    InterNode,
    /// Source and destination on the same node.
    IntraNode(IntraNodeSelector),
}

/// What a curve prices.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CurveTarget {
    Collective {
        op: CollectiveCurveOp,
        scope: CollectiveScope,
        ranks: RankCount,
    },
    PointToPoint {
        scope: PointToPointScope,
    },
}

/// How specifically a target matched a query; higher wins.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MatchSpecificity {
    InterNodeFabric,
    IntraNodeAny,
    IntraNodeListed,
    ExactNodes,
}

/// A placement to price: the operation and the node of every participant,
/// in rank order (for point-to-point: `[src, dst]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurveQuery<'a> {
    pub kind: CollectiveKind,
    pub participant_nodes: &'a [NodeId],
}

impl CurveTarget {
    /// Whether this target prices `query`, and how specifically.
    pub fn matches(&self, query: &CurveQuery<'_>) -> Option<MatchSpecificity> {
        let nodes: BTreeSet<NodeId> = query.participant_nodes.iter().copied().collect();
        match self {
            Self::Collective { op, scope, ranks } => {
                if CollectiveCurveOp::for_kind(query.kind) != Some(*op)
                    || query.participant_nodes.len() != ranks.get() as usize
                {
                    return None;
                }
                match scope {
                    CollectiveScope::NodeGroup(group) => {
                        (group.nodes() == &nodes).then_some(MatchSpecificity::ExactNodes)
                    }
                    CollectiveScope::InterNode => {
                        (nodes.len() >= 2).then_some(MatchSpecificity::InterNodeFabric)
                    }
                    CollectiveScope::IntraNode(selector) => intra_node_match(selector, &nodes),
                }
            }
            Self::PointToPoint { scope } => {
                let &[src, dst] = query.participant_nodes else {
                    return None;
                };
                if query.kind != CollectiveKind::SendRecv {
                    return None;
                }
                match scope {
                    PointToPointScope::NodePair(pair) => (pair.src() == src && pair.dst() == dst)
                        .then_some(MatchSpecificity::ExactNodes),
                    PointToPointScope::InterNode => {
                        (src != dst).then_some(MatchSpecificity::InterNodeFabric)
                    }
                    PointToPointScope::IntraNode(selector) => intra_node_match(selector, &nodes),
                }
            }
        }
    }

    /// Two targets are ambiguous when some query would match both at the
    /// same specificity, so neither could be chosen deterministically.
    pub fn ambiguous_with(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Collective {
                    op: op_a,
                    scope: scope_a,
                    ranks: ranks_a,
                },
                Self::Collective {
                    op: op_b,
                    scope: scope_b,
                    ranks: ranks_b,
                },
            ) => {
                op_a == op_b
                    && ranks_a == ranks_b
                    && match (scope_a, scope_b) {
                        (CollectiveScope::IntraNode(a), CollectiveScope::IntraNode(b)) => {
                            a.overlaps(b)
                        }
                        (a, b) => a == b,
                    }
            }
            (Self::PointToPoint { scope: a }, Self::PointToPoint { scope: b }) => match (a, b) {
                (PointToPointScope::IntraNode(a), PointToPointScope::IntraNode(b)) => a.overlaps(b),
                (a, b) => a == b,
            },
            _ => false,
        }
    }
}

impl Display for CurveTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Collective { op, scope, ranks } => {
                write!(f, "{} {} ranks={}", op.as_str(), scope, ranks.get())
            }
            Self::PointToPoint { scope } => write!(f, "send_recv {scope}"),
        }
    }
}

impl Display for CollectiveScope {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeGroup(group) => write!(f, "nodes={}", join_nodes(group.nodes())),
            Self::InterNode => write!(f, "inter_node"),
            Self::IntraNode(selector) => write!(f, "intra_node{}", selector_suffix(selector)),
        }
    }
}

impl Display for PointToPointScope {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodePair(pair) => write!(f, "node{}->node{}", pair.src(), pair.dst()),
            Self::InterNode => write!(f, "inter_node"),
            Self::IntraNode(selector) => write!(f, "intra_node{}", selector_suffix(selector)),
        }
    }
}

fn intra_node_match(
    selector: &IntraNodeSelector,
    nodes: &BTreeSet<NodeId>,
) -> Option<MatchSpecificity> {
    let mut iter = nodes.iter();
    let (Some(node), None) = (iter.next(), iter.next()) else {
        return None;
    };
    if !selector.contains(*node) {
        return None;
    }
    Some(match selector {
        IntraNodeSelector::AnyNode => MatchSpecificity::IntraNodeAny,
        IntraNodeSelector::Nodes(_) => MatchSpecificity::IntraNodeListed,
    })
}

fn join_nodes(nodes: &BTreeSet<NodeId>) -> String {
    nodes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join("+")
}

fn selector_suffix(selector: &IntraNodeSelector) -> String {
    match selector {
        IntraNodeSelector::AnyNode => String::new(),
        IntraNodeSelector::Nodes(nodes) => format!(" nodes={}", join_nodes(nodes)),
    }
}

/// Invalid target descriptions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    TooFewRanks { ranks: u32 },
    DuplicateNode { nodes: Vec<NodeId> },
    NodeGroupTooSmall { nodes: Vec<NodeId> },
    EmptyIntraNodeSelector,
    SelfPair { node: NodeId },
}

impl Display for TargetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewRanks { ranks } => write!(f, "ranks must be at least 2, got {ranks}"),
            Self::DuplicateNode { nodes } => write!(f, "nodes {nodes:?} contain duplicates"),
            Self::NodeGroupTooSmall { nodes } => {
                write!(
                    f,
                    "a node_group needs at least 2 distinct nodes, got {nodes:?}"
                )
            }
            Self::EmptyIntraNodeSelector => {
                write!(f, "intra_node nodes must be non-empty when set")
            }
            Self::SelfPair { node } => {
                write!(f, "src_node and dst_node must differ, both are {node}")
            }
        }
    }
}

impl std::error::Error for TargetError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_reduce(scope: CollectiveScope, ranks: u32) -> CurveTarget {
        CurveTarget::Collective {
            op: CollectiveCurveOp::AllReduce,
            scope,
            ranks: RankCount::new(ranks).expect("ranks"),
        }
    }

    fn p2p(scope: PointToPointScope) -> CurveTarget {
        CurveTarget::PointToPoint { scope }
    }

    fn query(kind: CollectiveKind, nodes: &[NodeId]) -> CurveQuery<'_> {
        CurveQuery {
            kind,
            participant_nodes: nodes,
        }
    }

    #[test]
    fn constructors_reject_invalid_targets() {
        assert_eq!(
            RankCount::new(1),
            Err(TargetError::TooFewRanks { ranks: 1 })
        );
        assert!(matches!(
            NodeGroup::new(&[0]),
            Err(TargetError::NodeGroupTooSmall { .. })
        ));
        assert!(matches!(
            NodeGroup::new(&[0, 0]),
            Err(TargetError::DuplicateNode { .. })
        ));
        assert_eq!(
            IntraNodeSelector::new(Some(&[])),
            Err(TargetError::EmptyIntraNodeSelector)
        );
        assert_eq!(
            DirectedNodePair::new(3, 3),
            Err(TargetError::SelfPair { node: 3 })
        );
    }

    #[test]
    fn node_group_matches_exact_node_set_and_rank_count() {
        let target = all_reduce(
            CollectiveScope::NodeGroup(NodeGroup::new(&[0, 1]).expect("group")),
            2,
        );
        assert_eq!(
            target.matches(&query(CollectiveKind::AllReduce, &[1, 0])),
            Some(MatchSpecificity::ExactNodes)
        );
        assert_eq!(
            target.matches(&query(CollectiveKind::AllReduce, &[0, 2])),
            None
        );
        assert_eq!(
            target.matches(&query(CollectiveKind::AllReduce, &[0, 0, 1, 1])),
            None,
            "rank count must match"
        );
        assert_eq!(
            target.matches(&query(CollectiveKind::AllGather, &[0, 1])),
            None
        );
        assert_eq!(
            target.matches(&query(CollectiveKind::SendRecv, &[0, 1])),
            None
        );
    }

    #[test]
    fn inter_node_and_intra_node_scopes() {
        let fabric = all_reduce(CollectiveScope::InterNode, 2);
        assert_eq!(
            fabric.matches(&query(CollectiveKind::AllReduce, &[4, 9])),
            Some(MatchSpecificity::InterNodeFabric)
        );
        assert_eq!(
            fabric.matches(&query(CollectiveKind::AllReduce, &[4, 4])),
            None
        );

        let any = all_reduce(CollectiveScope::IntraNode(IntraNodeSelector::AnyNode), 2);
        let listed = all_reduce(
            CollectiveScope::IntraNode(IntraNodeSelector::new(Some(&[4])).expect("selector")),
            2,
        );
        assert_eq!(
            any.matches(&query(CollectiveKind::AllReduce, &[4, 4])),
            Some(MatchSpecificity::IntraNodeAny)
        );
        assert_eq!(
            listed.matches(&query(CollectiveKind::AllReduce, &[4, 4])),
            Some(MatchSpecificity::IntraNodeListed)
        );
        assert_eq!(
            listed.matches(&query(CollectiveKind::AllReduce, &[5, 5])),
            None
        );
        assert_eq!(
            any.matches(&query(CollectiveKind::AllReduce, &[4, 5])),
            None
        );
    }

    #[test]
    fn point_to_point_pairs_are_directed() {
        let forward = p2p(PointToPointScope::NodePair(
            DirectedNodePair::new(0, 1).expect("pair"),
        ));
        assert_eq!(
            forward.matches(&query(CollectiveKind::SendRecv, &[0, 1])),
            Some(MatchSpecificity::ExactNodes)
        );
        assert_eq!(
            forward.matches(&query(CollectiveKind::SendRecv, &[1, 0])),
            None
        );
        assert_eq!(
            forward.matches(&query(CollectiveKind::AllReduce, &[0, 1])),
            None
        );
        assert_eq!(
            forward.matches(&query(CollectiveKind::SendRecv, &[0, 1, 1])),
            None
        );

        let fabric = p2p(PointToPointScope::InterNode);
        assert_eq!(
            fabric.matches(&query(CollectiveKind::SendRecv, &[1, 0])),
            Some(MatchSpecificity::InterNodeFabric)
        );
        assert_eq!(
            fabric.matches(&query(CollectiveKind::SendRecv, &[1, 1])),
            None
        );
        let local = p2p(PointToPointScope::IntraNode(IntraNodeSelector::AnyNode));
        assert_eq!(
            local.matches(&query(CollectiveKind::SendRecv, &[1, 1])),
            Some(MatchSpecificity::IntraNodeAny)
        );
    }

    #[test]
    fn specificity_orders_exact_over_fabric() {
        assert!(MatchSpecificity::ExactNodes > MatchSpecificity::IntraNodeListed);
        assert!(MatchSpecificity::IntraNodeListed > MatchSpecificity::IntraNodeAny);
        assert!(MatchSpecificity::IntraNodeAny > MatchSpecificity::InterNodeFabric);
    }

    #[test]
    fn ambiguity_detects_identical_and_overlapping_targets() {
        let pair = |src, dst| {
            p2p(PointToPointScope::NodePair(
                DirectedNodePair::new(src, dst).expect("pair"),
            ))
        };
        assert!(pair(0, 1).ambiguous_with(&pair(0, 1)));
        assert!(!pair(0, 1).ambiguous_with(&pair(1, 0)));
        let listed = |nodes: &[NodeId]| {
            all_reduce(
                CollectiveScope::IntraNode(IntraNodeSelector::new(Some(nodes)).expect("sel")),
                2,
            )
        };
        assert!(listed(&[0, 1]).ambiguous_with(&listed(&[1, 2])));
        assert!(!listed(&[0]).ambiguous_with(&listed(&[1])));
        assert!(!listed(&[0]).ambiguous_with(&all_reduce(
            CollectiveScope::IntraNode(IntraNodeSelector::AnyNode),
            2
        )));
        assert!(
            !all_reduce(CollectiveScope::InterNode, 2)
                .ambiguous_with(&all_reduce(CollectiveScope::InterNode, 4))
        );
    }

    #[test]
    fn display_is_stable() {
        let target = all_reduce(
            CollectiveScope::NodeGroup(NodeGroup::new(&[1, 0]).expect("group")),
            2,
        );
        assert_eq!(target.to_string(), "all_reduce nodes=0+1 ranks=2");
        let pair = p2p(PointToPointScope::NodePair(
            DirectedNodePair::new(0, 1).expect("pair"),
        ));
        assert_eq!(pair.to_string(), "send_recv node0->node1");
    }
}
