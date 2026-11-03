//! GPU SM-clock state and the compute and memory-bandwidth slowdown it
//! implies.
//!
//! A GPU profile may declare the SM clock its calibrated peaks correspond to
//! (`sustained_sm_clock_mhz`: what the card actually holds under sustained
//! load, which on boost-clocked parts is below the maximum boost clock) and
//! how achievable DRAM bandwidth depends on the SM clock. A per-GPU lock
//! (`locked_sm_clock_mhz`, e.g. `nvidia-smi -lgc`) then derives two factors:
//!
//! - compute: `min(1, locked / sustained)`. Tensor-core throughput scales
//!   with the SM clock; a lock at or above the sustained clock cannot make
//!   the card faster than the clock it sustains.
//! - bandwidth: the memory-system roofline
//!   `bandwidth(f) = min(dram_limit, k * f)`. The SMs, L2 and crossbar that
//!   issue and absorb DRAM requests run on the SM/graphics clock, so below
//!   some knee clock `f_k = dram_limit / k` they cannot keep DRAM saturated
//!   and achievable bandwidth falls linearly with the clock; above the knee
//!   DRAM is the limit and the SM clock does not matter. Relative to the
//!   calibrated (unlocked) bandwidth the factor is `min(1, locked / knee)`.
//!
//! The knee is required to be at or below the sustained clock: the
//! calibrated bandwidth was measured at full clock, so the full-clock point
//! must be DRAM-limited for the factor to be relative to it.

use std::fmt::{Display, Formatter};

/// A strictly positive, finite SM clock frequency in MHz.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct SmClockMhz(f64);

impl SmClockMhz {
    pub fn new(mhz: f64) -> Result<Self, GpuClockError> {
        if mhz.is_finite() && mhz > 0.0 {
            Ok(Self(mhz))
        } else {
            Err(GpuClockError::NonPositiveClock { mhz })
        }
    }

    pub fn mhz(self) -> f64 {
        self.0
    }
}

/// How achievable DRAM bandwidth responds to the SM clock.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BandwidthClockSensitivity {
    /// Bandwidth is assumed independent of the SM clock (no knee declared).
    /// This is the conservative default only in the sense that it adds no
    /// unfitted parameter; on the RTX 3090 it is wrong (validation ledger 17:
    /// decode slowed 1.26-1.34x at 1200 MHz with memory clocks untouched).
    Insensitive,
    /// `bandwidth(f) = min(dram_limit, k * f)` with the knee at this clock.
    Knee(SmClockMhz),
}

/// The declared clock behaviour of a GPU model: the sustained SM clock its
/// calibrated peaks correspond to and its bandwidth sensitivity.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GpuClockModel {
    sustained: SmClockMhz,
    bandwidth: BandwidthClockSensitivity,
}

impl GpuClockModel {
    pub fn new(
        sustained: SmClockMhz,
        bandwidth: BandwidthClockSensitivity,
    ) -> Result<Self, GpuClockError> {
        if let BandwidthClockSensitivity::Knee(knee) = bandwidth
            && knee.mhz() > sustained.mhz()
        {
            return Err(GpuClockError::KneeAboveSustained {
                knee_mhz: knee.mhz(),
                sustained_mhz: sustained.mhz(),
            });
        }
        Ok(Self {
            sustained,
            bandwidth,
        })
    }

    pub fn sustained(&self) -> SmClockMhz {
        self.sustained
    }

    pub fn bandwidth(&self) -> BandwidthClockSensitivity {
        self.bandwidth
    }

    pub fn bandwidth_knee(&self) -> Option<SmClockMhz> {
        match self.bandwidth {
            BandwidthClockSensitivity::Insensitive => None,
            BandwidthClockSensitivity::Knee(knee) => Some(knee),
        }
    }

    /// Factors for a GPU running with its SM clock locked at `locked`.
    pub fn degradation_at(&self, locked: SmClockMhz) -> ClockDegradation {
        let compute = (locked.mhz() / self.sustained.mhz()).min(1.0);
        let bandwidth = match self.bandwidth {
            BandwidthClockSensitivity::Insensitive => 1.0,
            BandwidthClockSensitivity::Knee(knee) => (locked.mhz() / knee.mhz()).min(1.0),
        };
        ClockDegradation { compute, bandwidth }
    }
}

/// Whether a GPU's SM clock is locked.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum GpuClockState {
    /// The card boosts freely and runs at its sustained clock under load.
    #[default]
    Unlocked,
    /// The SM clock is pinned (e.g. `nvidia-smi -lgc f,f`).
    Locked(SmClockMhz),
}

/// A GPU profile's clock: the model plus the GPU's current state. A lock
/// cannot exist without a model, so a locked clock always has a sustained
/// clock to be relative to.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GpuClock {
    model: GpuClockModel,
    state: GpuClockState,
}

impl GpuClock {
    pub fn new(model: GpuClockModel, state: GpuClockState) -> Self {
        Self { model, state }
    }

    pub fn unlocked(model: GpuClockModel) -> Self {
        Self::new(model, GpuClockState::Unlocked)
    }

    pub fn model(&self) -> GpuClockModel {
        self.model
    }

    pub fn state(&self) -> GpuClockState {
        self.state
    }

    pub fn locked_clock(&self) -> Option<SmClockMhz> {
        match self.state {
            GpuClockState::Unlocked => None,
            GpuClockState::Locked(clock) => Some(clock),
        }
    }

    pub fn with_state(self, state: GpuClockState) -> Self {
        Self { state, ..self }
    }

    pub fn degradation(&self) -> ClockDegradation {
        match self.state {
            GpuClockState::Unlocked => ClockDegradation::NOMINAL,
            GpuClockState::Locked(clock) => self.model.degradation_at(clock),
        }
    }
}

/// Multipliers on a GPU's effective peak FLOP/s and HBM bandwidth, each in
/// `(0, 1]`; `NOMINAL` (both exactly 1) means no slowdown.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClockDegradation {
    compute: f64,
    bandwidth: f64,
}

impl ClockDegradation {
    pub const NOMINAL: Self = Self {
        compute: 1.0,
        bandwidth: 1.0,
    };

    pub fn compute(&self) -> f64 {
        self.compute
    }

    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    pub fn is_nominal(&self) -> bool {
        self.compute == 1.0 && self.bandwidth == 1.0
    }

    /// The slowest of two GPUs in lock-step (tensor-parallel ranks, a stage):
    /// each term is bound by whichever GPU is slower at it.
    pub fn slowest(self, other: Self) -> Self {
        Self {
            compute: self.compute.min(other.compute),
            bandwidth: self.bandwidth.min(other.bandwidth),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuClockError {
    NonPositiveClock { mhz: f64 },
    KneeAboveSustained { knee_mhz: f64, sustained_mhz: f64 },
}

impl Display for GpuClockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonPositiveClock { mhz } => {
                write!(f, "SM clock must be positive and finite MHz, got {mhz}")
            }
            Self::KneeAboveSustained {
                knee_mhz,
                sustained_mhz,
            } => write!(
                f,
                "bandwidth knee clock {knee_mhz} MHz exceeds the sustained SM clock {sustained_mhz} MHz; the calibrated bandwidth must be DRAM-limited at the sustained clock"
            ),
        }
    }
}

impl std::error::Error for GpuClockError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(mhz: f64) -> SmClockMhz {
        SmClockMhz::new(mhz).expect("positive clock")
    }

    fn rtx3090_model() -> GpuClockModel {
        GpuClockModel::new(
            clock(1835.0),
            BandwidthClockSensitivity::Knee(clock(1512.0)),
        )
        .expect("knee below sustained")
    }

    #[test]
    fn clock_rejects_non_positive_and_non_finite_values() {
        for mhz in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(SmClockMhz::new(mhz).is_err(), "{mhz} must be rejected");
        }
        assert_eq!(clock(1200.0).mhz(), 1200.0);
    }

    #[test]
    fn knee_above_sustained_is_rejected() {
        let error = GpuClockModel::new(
            clock(1500.0),
            BandwidthClockSensitivity::Knee(clock(1600.0)),
        )
        .unwrap_err();
        assert_eq!(
            error,
            GpuClockError::KneeAboveSustained {
                knee_mhz: 1600.0,
                sustained_mhz: 1500.0
            }
        );
        assert!(error.to_string().contains("1600"));
        assert!(
            GpuClockModel::new(
                clock(1500.0),
                BandwidthClockSensitivity::Knee(clock(1500.0))
            )
            .is_ok()
        );
    }

    #[test]
    fn unlocked_clock_is_nominal() {
        let degradation = GpuClock::unlocked(rtx3090_model()).degradation();
        assert!(degradation.is_nominal());
        assert_eq!(degradation, ClockDegradation::NOMINAL);
    }

    #[test]
    fn locked_clock_scales_compute_by_sustained_ratio_and_bandwidth_below_knee() {
        let degradation =
            GpuClock::new(rtx3090_model(), GpuClockState::Locked(clock(1200.0))).degradation();
        assert!((degradation.compute() - 1200.0 / 1835.0).abs() < 1e-12);
        assert!((degradation.bandwidth() - 1200.0 / 1512.0).abs() < 1e-12);
        assert!(!degradation.is_nominal());
    }

    #[test]
    fn lock_between_knee_and_sustained_slows_compute_only() {
        let degradation = rtx3090_model().degradation_at(clock(1700.0));
        assert!(degradation.compute() < 1.0);
        assert_eq!(degradation.bandwidth(), 1.0);
    }

    #[test]
    fn lock_above_sustained_is_capped_at_nominal() {
        let degradation = rtx3090_model().degradation_at(clock(2100.0));
        assert!(degradation.is_nominal());
    }

    #[test]
    fn insensitive_bandwidth_keeps_bandwidth_factor_one() {
        let model = GpuClockModel::new(clock(1980.0), BandwidthClockSensitivity::Insensitive)
            .expect("valid");
        let degradation = model.degradation_at(clock(990.0));
        assert!((degradation.compute() - 0.5).abs() < 1e-12);
        assert_eq!(degradation.bandwidth(), 1.0);
        assert_eq!(model.bandwidth_knee(), None);
    }

    #[test]
    fn slowest_takes_each_term_from_the_slower_gpu() {
        let compute_bound = rtx3090_model().degradation_at(clock(1000.0));
        let other = GpuClockModel::new(clock(1000.0), BandwidthClockSensitivity::Insensitive)
            .expect("valid")
            .degradation_at(clock(900.0));
        let slowest = compute_bound.slowest(other);
        assert!((slowest.compute() - 1000.0 / 1835.0).abs() < 1e-12);
        assert!((slowest.bandwidth() - 1000.0 / 1512.0).abs() < 1e-12);
        assert!(
            ClockDegradation::NOMINAL
                .slowest(ClockDegradation::NOMINAL)
                .is_nominal()
        );
    }
}
