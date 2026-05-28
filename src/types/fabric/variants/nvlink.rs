use crate::types::{
    common::{Bandwidth, Bytes, Latency, LinkBandwidth, ReductionAccelerator},
    fabric::intra_node::NvLinkProfile,
};

pub enum NvLinkVariant {
    V1, // Pascal (P100)
    V2, // Volta (V100)
    V3, // Ampere (A100)
    V4, // Hopper (H100)
    V5, // Blackwell (B200)
}

impl NvLinkVariant {
    pub fn default_profile(&self) -> NvLinkProfile {
        match self {
            NvLinkVariant::V1 => NvLinkProfile {
                label: "NVLink 1.0",
                bw: LinkBandwidth::full_duplex_bidirectional_aggregate(
                    Bandwidth::from_gigabytes_per_sec(160.0),
                ),
                latency: Latency::from_us(1.5),
                reduction_accel: ReductionAccelerator::None,
            },
            NvLinkVariant::V2 => NvLinkProfile {
                label: "NVLink 2.0",
                bw: LinkBandwidth::full_duplex_bidirectional_aggregate(
                    Bandwidth::from_gigabytes_per_sec(300.0),
                ),
                latency: Latency::from_us(1.5),
                reduction_accel: ReductionAccelerator::None,
            },
            NvLinkVariant::V3 => NvLinkProfile {
                label: "NVLink 3.0",
                bw: LinkBandwidth::full_duplex_bidirectional_aggregate(
                    Bandwidth::from_gigabytes_per_sec(600.0),
                ),
                latency: Latency::from_us(1.0),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
            NvLinkVariant::V4 => NvLinkProfile {
                label: "NVLink 4.0",
                bw: LinkBandwidth::full_duplex_bidirectional_aggregate(
                    Bandwidth::from_gigabytes_per_sec(900.0),
                ),
                latency: Latency::from_us(0.7),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
            NvLinkVariant::V5 => NvLinkProfile {
                label: "NVLink 5.0",
                bw: LinkBandwidth::full_duplex_bidirectional_aggregate(
                    Bandwidth::from_gigabytes_per_sec(1800.0),
                ),
                latency: Latency::from_us(0.7),
                reduction_accel: ReductionAccelerator::Supported {
                    min_message: Bytes::from_bytes(8192),
                },
            },
        }
    }
}
