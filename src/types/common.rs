use crate::types::fabric::intra_node::LinkProfile;

pub type GpuId = u32;
pub type NodeId = u32;
pub type RankId = u32;
pub type NicId = u32;
pub type RailId = u32;
pub type CpuSocketId = u64;
pub type PcieSwitchId = u64;
pub type NvSwitchId = u32;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperationalState {
    #[default]
    Healthy,
    Disabled,
    Maintenance,
    Draining,
    Reserved,
}

impl OperationalState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Disabled => "disabled",
            Self::Maintenance => "maintenance",
            Self::Draining => "draining",
            Self::Reserved => "reserved",
        }
    }

    pub fn accepts_work(self) -> bool {
        matches!(self, Self::Healthy)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GpuAddr {
    pub node_id: NodeId,
    pub local_gpu_id: GpuId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EndpointId {
    Gpu(GpuId),
    Node(NodeId),
    CpuSocket(CpuSocketId),
    PcieSwitch(PcieSwitchId),
    NvSwitch(NvSwitchId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FabricEdge {
    pub endpoints: UnorderedPair<EndpointId>,
    pub profile: LinkProfile,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReductionAccelerator {
    None,
    Supported { min_message: Bytes },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FabricKind {
    InfiniBand,
    RoCE,
    Ethernet,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnorderedPair<T> {
    a: T,
    b: T,
}

impl<T: Ord> UnorderedPair<T> {
    pub fn new(x: T, y: T) -> Self {
        if x <= y {
            Self { a: x, b: y }
        } else {
            Self { a: y, b: x }
        }
    }

    pub fn endpoints(&self) -> (&T, &T) {
        (&self.a, &self.b)
    }

    pub fn into_tuple(self) -> (T, T) {
        (self.a, self.b)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(u64);

impl Bytes {
    pub fn from_bytes(b: u64) -> Self {
        Self(b)
    }
    pub fn from_kilobytes(kb: f64) -> Self {
        Self((kb * 1e3) as u64)
    }
    pub fn from_megabytes(mb: f64) -> Self {
        Self((mb * 1e6) as u64)
    }
    pub fn from_gigabytes(gb: f64) -> Self {
        Self((gb * 1e9) as u64)
    }
    pub fn as_bytes(self) -> u64 {
        self.0
    }
    pub fn as_gigabytes(self) -> f64 {
        self.0 as f64 / 1e9
    }
}

/// Bandwidth stored as bytes per second.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Bandwidth(f64);

impl Bandwidth {
    pub fn from_bytes_per_sec(bytes_per_sec: f64) -> Self {
        Self(bytes_per_sec)
    }
    pub fn from_gigabytes_per_sec(gb_s: f64) -> Self {
        Self(gb_s * 1e9)
    }
    pub fn from_gigabits_per_sec(gb_s: f64) -> Self {
        Self(gb_s * 1e9 / 8.0)
    }
    pub fn as_bytes_per_sec(self) -> f64 {
        self.0
    }
    pub fn as_gigabytes_per_sec(self) -> f64 {
        self.0 / 1e9
    }
    pub fn as_gigabits_per_sec(self) -> f64 {
        self.0 * 8.0 / 1e9
    }
}

impl std::ops::Div<f64> for Bandwidth {
    type Output = Bandwidth;
    fn div(self, rhs: f64) -> Bandwidth {
        Bandwidth(self.0 / rhs)
    }
}

impl std::ops::Mul<f64> for Bandwidth {
    type Output = Bandwidth;
    fn mul(self, rhs: f64) -> Bandwidth {
        Bandwidth(self.0 * rhs)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DuplexMode {
    Full,
    Half,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinkBandwidth {
    pub unidirectional: Bandwidth,
    pub duplex_mode: DuplexMode,
}

impl LinkBandwidth {
    pub fn full_duplex_unidirectional(bw: Bandwidth) -> Self {
        Self {
            unidirectional: bw,
            duplex_mode: DuplexMode::Full,
        }
    }

    pub fn full_duplex_bidirectional_aggregate(bw: Bandwidth) -> Self {
        Self {
            unidirectional: bw / 2.0,
            duplex_mode: DuplexMode::Full,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Latency(f64);

impl Latency {
    pub fn from_ms(ms: f64) -> Self {
        Self(ms / 1e3)
    }
    pub fn from_us(us: f64) -> Self {
        Self(us / 1e6)
    }
    pub fn to_ms(self) -> f64 {
        self.0 * 1e3
    }
    pub fn to_us(self) -> f64 {
        self.0 * 1e6
    }
}
