use std::fmt::{Display, Formatter, Result};

use crate::types::{
    common::{Bandwidth, Bytes},
    gpu_clock::{ClockDegradation, GpuClock},
};

#[derive(Clone, Debug, PartialEq)]
pub struct GpuProfile {
    pub label: &'static str,
    pub hbm_size: Bytes,
    pub hbm_bandwidth: Bandwidth,
    pub peak_f16_flops: f64,
    pub peak_f8_flops: Option<f64>,
    /// Declared SM-clock model and lock state; `None` means the peaks are
    /// taken as-is and no clock degradation can apply.
    pub clock: Option<GpuClock>,
}

impl GpuProfile {
    /// Compute and bandwidth multipliers from this GPU's clock state
    /// (`ClockDegradation::NOMINAL` without a clock or when unlocked).
    pub fn clock_degradation(&self) -> ClockDegradation {
        self.clock
            .map(|clock| clock.degradation())
            .unwrap_or(ClockDegradation::NOMINAL)
    }
}

impl Display for GpuProfile {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.label)
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Gpu {
    A100_40GB,
    A100_80GB,
    H100_SXM,
    H200_SXM,
    B200,
    MI300X,
}

impl Gpu {
    pub fn profile(&self) -> GpuProfile {
        match self {
            Gpu::A100_40GB => GpuProfile {
                label: "A100 40GB SXM",
                hbm_size: Bytes::from_gigabytes(40.0),
                hbm_bandwidth: Bandwidth::from_gigabytes_per_sec(1555.0),
                peak_f16_flops: 312.0,
                peak_f8_flops: None,
                clock: None,
            },
            Gpu::A100_80GB => GpuProfile {
                label: "A100 80GB SXM",
                hbm_size: Bytes::from_gigabytes(80.0),
                hbm_bandwidth: Bandwidth::from_gigabytes_per_sec(2039.0),
                peak_f16_flops: 312.0,
                peak_f8_flops: None,
                clock: None,
            },
            Gpu::H100_SXM => GpuProfile {
                label: "H100 SXM5",
                hbm_size: Bytes::from_gigabytes(80.0),
                hbm_bandwidth: Bandwidth::from_gigabytes_per_sec(3350.0),
                peak_f16_flops: 989.5,
                peak_f8_flops: Some(1979.0),
                clock: None,
            },
            Gpu::H200_SXM => GpuProfile {
                label: "H200 SXM",
                hbm_size: Bytes::from_gigabytes(141.0),
                hbm_bandwidth: Bandwidth::from_gigabytes_per_sec(4800.0),
                peak_f16_flops: 989.5,
                peak_f8_flops: Some(1979.0),
                clock: None,
            },
            Gpu::B200 => GpuProfile {
                label: "B200",
                hbm_size: Bytes::from_gigabytes(192.0),
                hbm_bandwidth: Bandwidth::from_gigabytes_per_sec(8000.0),
                peak_f16_flops: 2250.0,
                peak_f8_flops: Some(4500.0),
                clock: None,
            },
            Gpu::MI300X => GpuProfile {
                label: "MI300X",
                hbm_size: Bytes::from_gigabytes(192.0),
                hbm_bandwidth: Bandwidth::from_gigabytes_per_sec(5300.0),
                peak_f16_flops: 1307.4,
                peak_f8_flops: Some(2614.9),
                clock: None,
            },
        }
    }
}
