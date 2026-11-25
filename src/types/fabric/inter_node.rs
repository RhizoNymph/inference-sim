use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Result},
};

use crate::types::common::{
    FabricKind, GpuId, Latency, LinkBandwidth, NodeId, ReductionAccelerator, UnorderedPair,
};
use crate::types::fabric::direction::{DirectionProfile, LinkDirectionality};

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
    pub direction: LinkDirectionality,
}

impl CustomInterNodeLink {
    /// Bandwidth and latency for traffic `src -> dst` over this link.
    pub fn direction_profile(&self, src: NodeId, dst: NodeId) -> DirectionProfile {
        match self.direction {
            LinkDirectionality::Asymmetric(link) => link.direction(src, dst),
            LinkDirectionality::Symmetric => None,
        }
        .unwrap_or(DirectionProfile {
            bandwidth: self.profile.bw.unidirectional,
            latency: self.profile.latency,
        })
    }

    /// The slower direction's bandwidth and the larger latency, for
    /// aggregate (direction-agnostic) estimates.
    pub fn slowest_direction(&self) -> DirectionProfile {
        match self.direction {
            LinkDirectionality::Symmetric => DirectionProfile {
                bandwidth: self.profile.bw.unidirectional,
                latency: self.profile.latency,
            },
            LinkDirectionality::Asymmetric(link) => {
                let (forward, reverse) = (link.forward(), link.reverse());
                DirectionProfile {
                    bandwidth: if forward.bandwidth.as_bytes_per_sec()
                        <= reverse.bandwidth.as_bytes_per_sec()
                    {
                        forward.bandwidth
                    } else {
                        reverse.bandwidth
                    },
                    latency: Latency::from_us(forward.latency.to_us().max(reverse.latency.to_us())),
                }
            }
        }
    }

    /// Scales bandwidth and latency of both directions (scenario overlays).
    pub fn scale(&mut self, bandwidth_scale: f64, latency_scale: f64) {
        self.profile.bw.unidirectional = self.profile.bw.unidirectional * bandwidth_scale;
        self.profile.latency = Latency::from_us(self.profile.latency.to_us() * latency_scale);
        if let LinkDirectionality::Asymmetric(link) = self.direction {
            self.direction =
                LinkDirectionality::Asymmetric(link.scaled(bandwidth_scale, latency_scale));
        }
    }
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
