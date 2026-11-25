use crate::{
    types::{
        collective_curves::CurveExtrapolation,
        common::{Bytes, RankId},
    },
    workload::{DType, InferencePhase},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    AllToAll,
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    SendRecv,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveAlgorithm {
    Auto,
    Ring,
    Tree,
    Hierarchical,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    Sum,
    Max,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectiveCall {
    pub kind: CollectiveKind,
    pub participants: Vec<RankId>,
    pub bytes_per_rank: Bytes,
    pub dtype: DType,
    pub reduction: Option<ReductionOp>,
    pub root: Option<RankId>,
    pub phase: InferencePhase,
    pub algorithm: CollectiveAlgorithm,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectiveCost {
    pub latency_s: f64,
    pub bandwidth_s: f64,
    pub total_s: f64,
    pub bottlenecks: Vec<String>,
    pub pricing: CollectivePricing,
}

/// How a collective's or transfer's time was obtained, so evidence records
/// can say whether a number came from a measured curve or the analytical
/// alpha-beta model.
#[derive(Clone, Debug, PartialEq)]
pub enum CollectivePricing {
    /// Nothing crosses a link (single participant or zero bytes).
    NoTraffic,
    /// Analytical alpha-beta model; the coverage says why no curve applied.
    AlphaBeta(CurveCoverage),
    /// A measured curve priced the call.
    MeasuredCurve(CurveApplication),
}

/// Why the alpha-beta model priced a call instead of a measured curve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CurveCoverage {
    /// The cluster lists no curves.
    NotConfigured,
    /// Curves exist but are suspended (the reason is rendered text).
    Suspended { reason: String },
    /// Curves exist but none matches the call's op, placement, and ranks.
    NoMatch,
    /// This kind of transfer is never priced from curves (KV transfers).
    NotConsulted,
}

/// A measured curve applied to one call.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveApplication {
    pub curve_label: String,
    pub source: Option<String>,
    pub bytes: u64,
    pub extrapolation: CurveExtrapolation,
    pub derived_region: bool,
}
