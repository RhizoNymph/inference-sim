use crate::{
    types::common::{Bytes, RankId},
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
}
