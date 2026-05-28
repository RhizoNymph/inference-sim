use crate::types::{
    common::{Bandwidth, FabricKind, Latency, LinkBandwidth, ReductionAccelerator},
    fabric::inter_node::FabricProfile,
};

pub enum EthVariant {
    E10G,
    E25G,
    E40G,
    E100G,
    E200G,
    E400G,
    E800G,
}

impl EthVariant {
    pub fn default_profile(&self) -> FabricProfile {
        match self {
            EthVariant::E10G => FabricProfile {
                kind: FabricKind::Ethernet,
                label: "Ethernet 10G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    10.0,
                )),
                latency: Latency::from_us(30.0),
                reduction_accel: ReductionAccelerator::None,
            },
            EthVariant::E25G => FabricProfile {
                kind: FabricKind::Ethernet,
                label: "Ethernet 25G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    25.0,
                )),
                latency: Latency::from_us(20.0),
                reduction_accel: ReductionAccelerator::None,
            },
            EthVariant::E40G => FabricProfile {
                kind: FabricKind::Ethernet,
                label: "Ethernet 40G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    40.0,
                )),
                latency: Latency::from_us(15.0),
                reduction_accel: ReductionAccelerator::None,
            },
            EthVariant::E100G => FabricProfile {
                kind: FabricKind::Ethernet,
                label: "Ethernet 100G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    100.0,
                )),
                latency: Latency::from_us(10.0),
                reduction_accel: ReductionAccelerator::None,
            },
            EthVariant::E200G => FabricProfile {
                kind: FabricKind::Ethernet,
                label: "Ethernet 200G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    200.0,
                )),
                latency: Latency::from_us(8.0),
                reduction_accel: ReductionAccelerator::None,
            },
            EthVariant::E400G => FabricProfile {
                kind: FabricKind::Ethernet,
                label: "Ethernet 400G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    400.0,
                )),
                latency: Latency::from_us(6.0),
                reduction_accel: ReductionAccelerator::None,
            },
            EthVariant::E800G => FabricProfile {
                kind: FabricKind::Ethernet,
                label: "Ethernet 800G",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    800.0,
                )),
                latency: Latency::from_us(5.0),
                reduction_accel: ReductionAccelerator::None,
            },
        }
    }
}
