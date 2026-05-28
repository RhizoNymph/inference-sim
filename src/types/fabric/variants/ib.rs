use crate::types::{
    common::{Bandwidth, Bytes, FabricKind, Latency, LinkBandwidth, ReductionAccelerator},
    fabric::inter_node::FabricProfile,
};

pub enum IbVariant {
    Edr,
    Hdr,
    Ndr,
    Xdr,
}

impl IbVariant {
    pub fn default_profile(&self) -> FabricProfile {
        match self {
            IbVariant::Edr => FabricProfile {
                kind: FabricKind::InfiniBand,
                label: "IB EDR",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    100.0,
                )),
                latency: Latency::from_us(1.8),
                reduction_accel: ReductionAccelerator::None, //Pre-SHARP
            },
            IbVariant::Hdr => FabricProfile {
                kind: FabricKind::InfiniBand,
                label: "IB HDR",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    200.0,
                )),
                latency: Latency::from_us(1.5),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
            IbVariant::Ndr => FabricProfile {
                kind: FabricKind::InfiniBand,
                label: "IB NDR",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    400.0,
                )),
                latency: Latency::from_us(1.2),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
            IbVariant::Xdr => FabricProfile {
                kind: FabricKind::InfiniBand,
                label: "IB XDR",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    800.0,
                )),
                latency: Latency::from_us(1.0),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
        }
    }
}
