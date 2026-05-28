use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Result},
};

use crate::types::common::{
    FabricKind, GpuId, Latency, LinkBandwidth, NodeId, ReductionAccelerator, UnorderedPair,
};

#[derive(Clone, Debug, PartialEq)]
pub struct FabricProfile {
    pub kind: FabricKind,
    pub label: &'static str,
    pub bw: LinkBandwidth,
    pub latency: Latency,
    pub reduction_accel: ReductionAccelerator,
}

impl Display for FabricProfile {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.label)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CustomInterNodeLink {
    pub profile: FabricProfile,
    pub rail: Option<u32>,
    pub endpoints: Option<CustomInterNodeLinkEndpoints>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomInterNodeLinkEndpoints {
    pub from_node: NodeId,
    pub from_gpus: Vec<GpuId>,
    pub to_node: NodeId,
    pub to_gpus: Vec<GpuId>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InterNodeTopology {
    FatTree {
        link: FabricProfile,
        oversubscription: f64,
        leaf_size: u16,
    },
    Flat {
        link: FabricProfile,
    },
    Custom(HashMap<UnorderedPair<NodeId>, Vec<CustomInterNodeLink>>),
}
