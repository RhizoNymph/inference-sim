use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Display, Formatter, Result};

use crate::types::common::{
    Bandwidth, GpuId, Latency, LinkBandwidth, NicId, OperationalState, ReductionAccelerator,
    UnorderedPair,
};

#[derive(Clone, Debug, PartialEq)]
pub struct NvLinkProfile {
    pub label: &'static str,
    pub bw: LinkBandwidth,
    pub latency: Latency,
    pub reduction_accel: ReductionAccelerator,
}

impl Display for NvLinkProfile {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.label)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PcieProfile {
    pub label: &'static str,
    pub bw: LinkBandwidth,
    pub latency: Latency,
}

impl Display for PcieProfile {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.label)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LinkProfile {
    NvLink(NvLinkProfile),
    Pcie(PcieProfile),
}

#[derive(Clone, Debug, PartialEq)]
pub enum IntraNodeTopology {
    NvSwitch(NvLinkProfile),
    NvLinkDirect(NvLinkProfile),
    Pcie(PcieProfile),
    Custom(HashMap<UnorderedPair<GpuId>, LinkProfile>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuNicPathOverride {
    pub label: Option<String>,
    pub bandwidth: Option<Bandwidth>,
    pub latency: Option<Latency>,
    pub gpudirect: Option<bool>,
    pub available: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuNicAffinity {
    Dedicated,
    Shared { gpus_per_nic: u8 },
    Uniform,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeNetworkProfile {
    pub nic_count: u8,
    pub nic_bandwidth: Bandwidth,
    pub nic_bandwidth_overrides: BTreeMap<NicId, Bandwidth>,
    pub nic_latency_scale_overrides: BTreeMap<NicId, f64>,
    pub gpu_to_nic: GpuNicAffinity,
    pub rail_count: u8,
    pub nic_rail_map: BTreeMap<NicId, u32>,
    pub gpu_nic_map: BTreeMap<GpuId, Vec<NicId>>,
    pub gpu_numa_map: BTreeMap<GpuId, u32>,
    pub nic_numa_map: BTreeMap<NicId, u32>,
    pub cross_numa_bandwidth_scale: f64,
    pub cross_numa_latency_scale: f64,
    pub gpu_nic_path_overrides: BTreeMap<(GpuId, NicId), GpuNicPathOverride>,
    pub disabled_nics: BTreeSet<NicId>,
    pub nic_operational_states: BTreeMap<NicId, OperationalState>,
}

impl NodeNetworkProfile {
    pub fn nic_candidates_for_gpu(&self, gpu_id: GpuId) -> Vec<NicId> {
        let nics = if let Some(nics) = self.gpu_nic_map.get(&gpu_id) {
            nics.clone()
        } else {
            let nic_count = self.nic_count.max(1) as u32;
            match self.gpu_to_nic {
                GpuNicAffinity::Dedicated => vec![gpu_id.min(nic_count - 1)],
                GpuNicAffinity::Shared { gpus_per_nic } => {
                    vec![(gpu_id / u32::from(gpus_per_nic.max(1))).min(nic_count - 1)]
                }
                GpuNicAffinity::Uniform => (0..nic_count).collect(),
            }
        };

        nics.into_iter()
            .filter(|nic_id| self.is_nic_available(*nic_id))
            .filter(|nic_id| {
                self.gpu_nic_path_overrides
                    .get(&(gpu_id, *nic_id))
                    .is_none_or(|path| path.available)
            })
            .collect()
    }

    pub fn active_nic_count(&self) -> u8 {
        (0..u32::from(self.nic_count))
            .filter(|nic_id| self.is_nic_available(*nic_id))
            .count()
            .min(u8::MAX as usize) as u8
    }

    pub fn nic_operational_state(&self, nic_id: NicId) -> OperationalState {
        self.nic_operational_states
            .get(&nic_id)
            .copied()
            .unwrap_or_else(|| {
                if self.disabled_nics.contains(&nic_id) {
                    OperationalState::Disabled
                } else {
                    OperationalState::Healthy
                }
            })
    }

    pub fn is_nic_available(&self, nic_id: NicId) -> bool {
        nic_id < u32::from(self.nic_count) && self.nic_operational_state(nic_id).accepts_work()
    }

    pub fn nic_bandwidth(&self, nic_id: NicId) -> Bandwidth {
        self.nic_bandwidth_overrides
            .get(&nic_id)
            .copied()
            .unwrap_or(self.nic_bandwidth)
    }

    pub fn nic_latency_scale(&self, nic_id: NicId) -> f64 {
        self.nic_latency_scale_overrides
            .get(&nic_id)
            .copied()
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .unwrap_or(1.0)
    }

    pub fn rail_id(&self, nic_id: NicId) -> u32 {
        self.nic_rail_map
            .get(&nic_id)
            .copied()
            .unwrap_or_else(|| nic_id % u32::from(self.rail_count.max(1)))
    }

    pub fn active_rails(&self) -> BTreeSet<u32> {
        (0..u32::from(self.nic_count))
            .filter(|nic_id| self.is_nic_available(*nic_id))
            .map(|nic_id| self.rail_id(nic_id))
            .collect()
    }

    pub fn has_active_rail(&self, rail_id: u32) -> bool {
        self.active_rails().contains(&rail_id)
    }

    pub fn gpu_nic_path(&self, gpu_id: GpuId, nic_id: NicId) -> Option<&GpuNicPathOverride> {
        self.gpu_nic_path_overrides.get(&(gpu_id, nic_id))
    }

    pub fn gpu_numa_domain(&self, gpu_id: GpuId) -> Option<u32> {
        self.gpu_numa_map.get(&gpu_id).copied()
    }

    pub fn nic_numa_domain(&self, nic_id: NicId) -> Option<u32> {
        self.nic_numa_map.get(&nic_id).copied()
    }

    pub fn gpu_nic_numa_domains(&self, gpu_id: GpuId, nic_id: NicId) -> Option<(u32, u32)> {
        Some((self.gpu_numa_domain(gpu_id)?, self.nic_numa_domain(nic_id)?))
    }

    pub fn active_nic_bandwidth_bytes_per_sec(&self) -> f64 {
        (0..u32::from(self.nic_count))
            .filter(|nic_id| self.is_nic_available(*nic_id))
            .map(|nic_id| self.nic_bandwidth(nic_id).as_bytes_per_sec())
            .sum()
    }
}
