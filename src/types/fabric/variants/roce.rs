use crate::types::{
    common::{Bandwidth, Bytes, FabricKind, Latency, LinkBandwidth, ReductionAccelerator},
    fabric::inter_node::FabricProfile,
};

pub enum RoceVariant {
    V2_25G,
    V2_50G,
    V2_100G,
    V2_200G,
    V2_400G,
    V2_800G,
}

impl RoceVariant {
    pub fn default_profile(&self) -> FabricProfile {
        match self {
            RoceVariant::V2_25G => FabricProfile {
                kind: FabricKind::RoCE,
                label: "RoCEv2 25G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    25.0,
                )),
                latency: Latency::from_us(4.0),
                reduction_accel: ReductionAccelerator::None,
            },
            RoceVariant::V2_50G => FabricProfile {
                kind: FabricKind::RoCE,
                label: "RoCEv2 50G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    50.0,
                )),
                latency: Latency::from_us(3.5),
                reduction_accel: ReductionAccelerator::None,
            },
            RoceVariant::V2_100G => FabricProfile {
                kind: FabricKind::RoCE,
                label: "RoCEv2 100G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    100.0,
                )),
                latency: Latency::from_us(3.0),
                reduction_accel: ReductionAccelerator::None,
            },
            RoceVariant::V2_200G => FabricProfile {
                kind: FabricKind::RoCE,
                label: "RoCEv2 200G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    200.0,
                )),
                latency: Latency::from_us(2.5),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
            RoceVariant::V2_400G => FabricProfile {
                kind: FabricKind::RoCE,
                label: "RoCEv2 400G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    400.0,
                )),
                latency: Latency::from_us(2.0),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
            RoceVariant::V2_800G => FabricProfile {
                kind: FabricKind::RoCE,
                label: "RoCEv2 800G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    800.0,
                )),
                latency: Latency::from_us(1.8),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
        }
    }
}
