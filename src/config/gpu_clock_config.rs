//! Cluster-TOML GPU clock keys (`sustained_sm_clock_mhz`,
//! `bandwidth_knee_sm_clock_mhz`, `locked_sm_clock_mhz`) layered onto a GPU
//! profile's typed clock (docs/features/degraded_gpu_model.md).

use super::ConfigError;
use crate::types::gpu_clock::{
    BandwidthClockSensitivity, GpuClock, GpuClockModel, GpuClockState, SmClockMhz,
};

/// Raw clock keys from one profile-override layer (node-wide fields, a
/// `[[nodes.gpus]]` entry, or a `gpu_profile_overrides` entry).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct GpuClockOverrideConfig {
    pub(super) sustained_sm_clock_mhz: Option<f64>,
    pub(super) bandwidth_knee_sm_clock_mhz: Option<f64>,
    pub(super) locked_sm_clock_mhz: Option<f64>,
}

impl GpuClockOverrideConfig {
    pub(super) fn has_overrides(&self) -> bool {
        self.sustained_sm_clock_mhz.is_some()
            || self.bandwidth_knee_sm_clock_mhz.is_some()
            || self.locked_sm_clock_mhz.is_some()
    }
}

/// Overlays `layer` onto `base` key by key (a key the layer sets replaces the
/// base's; unset keys are inherited) and validates the result: a knee or a
/// lock needs a sustained clock, every clock is positive, and the knee is at
/// or below the sustained clock.
pub(super) fn merge_gpu_clock(
    name: &str,
    base: Option<GpuClock>,
    layer: &GpuClockOverrideConfig,
) -> Result<Option<GpuClock>, ConfigError> {
    if !layer.has_overrides() {
        return Ok(base);
    }
    let sustained = layer
        .sustained_sm_clock_mhz
        .map(|mhz| parse_clock(name, "sustained_sm_clock_mhz", mhz))
        .transpose()?
        .or(base.map(|clock| clock.model().sustained()));
    let knee = layer
        .bandwidth_knee_sm_clock_mhz
        .map(|mhz| parse_clock(name, "bandwidth_knee_sm_clock_mhz", mhz))
        .transpose()?
        .or(base.and_then(|clock| clock.model().bandwidth_knee()));
    let locked = layer
        .locked_sm_clock_mhz
        .map(|mhz| parse_clock(name, "locked_sm_clock_mhz", mhz))
        .transpose()?
        .or(base.and_then(|clock| clock.locked_clock()));

    let Some(sustained) = sustained else {
        let key = if locked.is_some() {
            "locked_sm_clock_mhz"
        } else {
            "bandwidth_knee_sm_clock_mhz"
        };
        return Err(ConfigError::new(format!(
            "{name}.{key} requires sustained_sm_clock_mhz (the SM clock the GPU's calibrated peaks correspond to) on the same GPU profile"
        )));
    };
    let bandwidth = knee
        .map(BandwidthClockSensitivity::Knee)
        .unwrap_or(BandwidthClockSensitivity::Insensitive);
    let model = GpuClockModel::new(sustained, bandwidth)
        .map_err(|error| ConfigError::new(format!("{name}: {error}")))?;
    let state = locked
        .map(GpuClockState::Locked)
        .unwrap_or(GpuClockState::Unlocked);
    Ok(Some(GpuClock::new(model, state)))
}

fn parse_clock(name: &str, key: &str, mhz: f64) -> Result<SmClockMhz, ConfigError> {
    SmClockMhz::new(mhz).map_err(|error| ConfigError::new(format!("{name}.{key}: {error}")))
}
