//! The cluster's set of measured curves and curve lookup.

use std::fmt::{Display, Formatter};

use super::{
    curve::MeasuredCurve,
    target::{CurveQuery, CurveTarget},
};
use crate::types::common::NodeId;

/// One measured curve and what it prices.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectiveCurve {
    target: CurveTarget,
    curve: MeasuredCurve,
    source: Option<String>,
}

impl CollectiveCurve {
    pub fn new(target: CurveTarget, curve: MeasuredCurve, source: Option<String>) -> Self {
        Self {
            target,
            curve,
            source,
        }
    }

    pub fn target(&self) -> &CurveTarget {
        &self.target
    }

    pub fn curve(&self) -> &MeasuredCurve {
        &self.curve
    }

    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Short identifier used in evidence records.
    pub fn label(&self) -> String {
        self.target.to_string()
    }
}

/// Why configured curves are not being used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CurveSuspension {
    /// A scenario changed network bandwidth/latency or disabled network
    /// resources, so curves measured on the unmodified fabric no longer
    /// describe it.
    ScenarioNetworkOverlay { scenario: String },
}

impl Display for CurveSuspension {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScenarioNetworkOverlay { scenario } => write!(
                f,
                "scenario '{scenario}' modifies the network, so curves measured on the unmodified fabric are not used"
            ),
        }
    }
}

/// Outcome of looking up a curve for one collective or transfer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurveLookup<'a> {
    NotConfigured,
    Suspended(&'a CurveSuspension),
    NoMatch,
    Matched(&'a CollectiveCurve),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CurveSetError {
    Ambiguous {
        first: usize,
        second: usize,
        target: String,
    },
}

impl Display for CurveSetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ambiguous {
                first,
                second,
                target,
            } => write!(
                f,
                "collective_curves[{first}] and collective_curves[{second}] both price '{target}'; keep one"
            ),
        }
    }
}

impl std::error::Error for CurveSetError {}

/// Curves keyed by target. No two curves are ambiguous for any query, so a
/// lookup's most specific match is unique.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollectiveCurveSet {
    curves: Vec<CollectiveCurve>,
    suspension: Option<CurveSuspension>,
}

impl CollectiveCurveSet {
    pub fn new(curves: Vec<CollectiveCurve>) -> Result<Self, CurveSetError> {
        for (first, a) in curves.iter().enumerate() {
            for (offset, b) in curves[first + 1..].iter().enumerate() {
                if a.target.ambiguous_with(&b.target) {
                    return Err(CurveSetError::Ambiguous {
                        first,
                        second: first + 1 + offset,
                        target: a.label(),
                    });
                }
            }
        }
        Ok(Self {
            curves,
            suspension: None,
        })
    }

    pub fn curves(&self) -> &[CollectiveCurve] {
        &self.curves
    }

    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    pub fn suspension(&self) -> Option<&CurveSuspension> {
        self.suspension.as_ref()
    }

    /// Stops using the curves (they stay listed for reporting). A no-op on
    /// an empty set.
    pub fn suspend(&mut self, reason: CurveSuspension) {
        if !self.curves.is_empty() {
            self.suspension = Some(reason);
        }
    }

    /// Node ids referenced by any curve scope, for validation against the
    /// cluster's nodes.
    pub fn referenced_nodes(&self) -> Vec<(usize, NodeId)> {
        use super::target::{CollectiveScope, IntraNodeSelector, PointToPointScope};
        let mut nodes = Vec::new();
        for (idx, curve) in self.curves.iter().enumerate() {
            let selector_nodes = |selector: &IntraNodeSelector| match selector {
                IntraNodeSelector::AnyNode => Vec::new(),
                IntraNodeSelector::Nodes(nodes) => nodes.iter().copied().collect(),
            };
            let referenced: Vec<NodeId> = match &curve.target {
                CurveTarget::Collective { scope, .. } => match scope {
                    CollectiveScope::NodeGroup(group) => group.nodes().iter().copied().collect(),
                    CollectiveScope::InterNode => Vec::new(),
                    CollectiveScope::IntraNode(selector) => selector_nodes(selector),
                },
                CurveTarget::PointToPoint { scope } => match scope {
                    PointToPointScope::NodePair(pair) => vec![pair.src(), pair.dst()],
                    PointToPointScope::InterNode => Vec::new(),
                    PointToPointScope::IntraNode(selector) => selector_nodes(selector),
                },
            };
            nodes.extend(referenced.into_iter().map(|node| (idx, node)));
        }
        nodes
    }

    pub fn lookup(&self, query: &CurveQuery<'_>) -> CurveLookup<'_> {
        if self.curves.is_empty() {
            return CurveLookup::NotConfigured;
        }
        if let Some(reason) = &self.suspension {
            return CurveLookup::Suspended(reason);
        }
        self.curves
            .iter()
            .filter_map(|curve| curve.target.matches(query).map(|rank| (rank, curve)))
            .max_by_key(|(rank, _)| *rank)
            .map_or(CurveLookup::NoMatch, |(_, curve)| {
                CurveLookup::Matched(curve)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        collective::CollectiveKind,
        collective_curves::target::{
            CollectiveCurveOp, CollectiveScope, DirectedNodePair, NodeGroup, PointToPointScope,
            RankCount,
        },
    };

    fn curve(base_us: f64) -> MeasuredCurve {
        MeasuredCurve::from_microseconds(&[(1024, base_us), (1_048_576, base_us * 10.0)], None)
            .expect("curve")
    }

    fn all_reduce(scope: CollectiveScope, base_us: f64) -> CollectiveCurve {
        CollectiveCurve::new(
            CurveTarget::Collective {
                op: CollectiveCurveOp::AllReduce,
                scope,
                ranks: RankCount::new(2).expect("ranks"),
            },
            curve(base_us),
            Some("test".to_string()),
        )
    }

    fn send(src: NodeId, dst: NodeId, base_us: f64) -> CollectiveCurve {
        CollectiveCurve::new(
            CurveTarget::PointToPoint {
                scope: PointToPointScope::NodePair(DirectedNodePair::new(src, dst).expect("pair")),
            },
            curve(base_us),
            None,
        )
    }

    fn query(kind: CollectiveKind, nodes: &[NodeId]) -> CurveQuery<'_> {
        CurveQuery {
            kind,
            participant_nodes: nodes,
        }
    }

    #[test]
    fn empty_set_is_not_configured() {
        let set = CollectiveCurveSet::default();
        assert!(set.is_empty());
        assert_eq!(
            set.lookup(&query(CollectiveKind::AllReduce, &[0, 1])),
            CurveLookup::NotConfigured
        );
    }

    #[test]
    fn rejects_ambiguous_curves() {
        let error = CollectiveCurveSet::new(vec![send(0, 1, 50.0), send(0, 1, 60.0)])
            .expect_err("duplicate pair");
        assert_eq!(
            error,
            CurveSetError::Ambiguous {
                first: 0,
                second: 1,
                target: "send_recv node0->node1".to_string()
            }
        );
        CollectiveCurveSet::new(vec![send(0, 1, 50.0), send(1, 0, 60.0)])
            .expect("opposite directions are distinct");
    }

    #[test]
    fn most_specific_match_wins() {
        let set = CollectiveCurveSet::new(vec![
            all_reduce(CollectiveScope::InterNode, 100.0),
            all_reduce(
                CollectiveScope::NodeGroup(NodeGroup::new(&[0, 1]).expect("group")),
                130.0,
            ),
        ])
        .expect("set");
        let CurveLookup::Matched(exact) = set.lookup(&query(CollectiveKind::AllReduce, &[0, 1]))
        else {
            panic!("expected match");
        };
        assert_eq!(exact.curve().floor_s(), 130.0e-6);
        let CurveLookup::Matched(fabric) = set.lookup(&query(CollectiveKind::AllReduce, &[0, 2]))
        else {
            panic!("expected fabric match");
        };
        assert_eq!(fabric.curve().floor_s(), 100.0e-6);
        assert_eq!(
            set.lookup(&query(CollectiveKind::AllGather, &[0, 1])),
            CurveLookup::NoMatch
        );
    }

    #[test]
    fn directed_lookup_distinguishes_directions() {
        let set = CollectiveCurveSet::new(vec![send(0, 1, 50.0), send(1, 0, 150.0)]).expect("set");
        let floor = |nodes: &[NodeId]| match set.lookup(&query(CollectiveKind::SendRecv, nodes)) {
            CurveLookup::Matched(curve) => curve.curve().floor_s(),
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(floor(&[0, 1]), 50.0e-6);
        assert_eq!(floor(&[1, 0]), 150.0e-6);
    }

    #[test]
    fn suspension_disables_lookup_but_keeps_curves() {
        let mut set = CollectiveCurveSet::new(vec![send(0, 1, 50.0)]).expect("set");
        set.suspend(CurveSuspension::ScenarioNetworkOverlay {
            scenario: "degraded".to_string(),
        });
        assert!(matches!(
            set.lookup(&query(CollectiveKind::SendRecv, &[0, 1])),
            CurveLookup::Suspended(_)
        ));
        assert_eq!(set.curves().len(), 1);

        let mut empty = CollectiveCurveSet::default();
        empty.suspend(CurveSuspension::ScenarioNetworkOverlay {
            scenario: "x".to_string(),
        });
        assert_eq!(empty.suspension(), None);
    }

    #[test]
    fn lists_referenced_nodes() {
        let set = CollectiveCurveSet::new(vec![
            send(0, 1, 50.0),
            all_reduce(
                CollectiveScope::NodeGroup(NodeGroup::new(&[2, 3]).expect("group")),
                10.0,
            ),
            all_reduce(CollectiveScope::InterNode, 10.0),
        ])
        .expect("set");
        assert_eq!(set.referenced_nodes(), vec![(0, 0), (0, 1), (1, 2), (1, 3)]);
    }
}
