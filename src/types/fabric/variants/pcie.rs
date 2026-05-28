use crate::types::{
    common::{Bandwidth, Latency, LinkBandwidth},
    fabric::intra_node::PcieProfile,
};

pub enum PcieVariant {
    Gen3x16,
    Gen4x16,
    Gen5x16,
    Gen6x16,
}

impl PcieVariant {
    pub fn default_profile(&self) -> PcieProfile {
        match self {
            PcieVariant::Gen3x16 => PcieProfile {
                label: "PCIe Gen3 x16",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    128.0,
                )), // ~16 GB/s
                latency: Latency::from_us(1.5),
            },
            PcieVariant::Gen4x16 => PcieProfile {
                label: "PCIe Gen4 x16",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    256.0,
                )), // ~32 GB/s
                latency: Latency::from_us(1.2),
            },
            PcieVariant::Gen5x16 => PcieProfile {
                label: "PCIe Gen5 x16",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    512.0,
                )), // ~64 GB/s
                latency: Latency::from_us(1.0),
            },
            PcieVariant::Gen6x16 => PcieProfile {
                label: "PCIe Gen6 x16",
                bw: LinkBandwidth::full_duplex_unidirectional(Bandwidth::from_gigabits_per_sec(
                    1024.0,
                )), // ~128 GB/s
                latency: Latency::from_us(0.8),
            },
        }
    }
}
