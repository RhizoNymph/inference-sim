//! Per-pipeline-stage slowdown from locked GPU clocks
//! (docs/features/degraded_gpu_model.md).
//!
//! Every GPU's profile may carry a clock state whose `ClockDegradation`
//! multiplies its effective peak FLOP/s and HBM bandwidth. Ranks inside one
//! pipeline stage run in lock-step (tensor/expert parallel), so a stage is as
//! slow as its slowest GPU at each term; different stages run their own
//! layers, so a degraded GPU slows only the stage it serves.
//!
//! The static solver prices a phase once on the placement's nominal hardware
//! (including any calibration fit) and then scales each stage's per-layer
//! compute by `analytical(stage-degraded) / analytical(nominal)`, so a stage
//! whose GPUs are all unlocked keeps exactly the nominal duration. The serving
//! step cost applies the factors to each stage's share of a step's compute
//! and memory terms.

use super::*;
use crate::types::gpu_clock::{BandwidthClockSensitivity, ClockDegradation, GpuClock};

/// Approximation code reported when any placed GPU runs a locked clock.
pub const GPU_CLOCK_DEGRADATION: &str = "gpu_clock_degradation";

/// One locked GPU inside a stage, kept for evidence.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct LockedGpu {
    pub(super) addr: GpuAddr,
    pub(super) clock: GpuClock,
}

/// The clock degradation of one pipeline stage.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct StageClock {
    pub(super) degradation: ClockDegradation,
    pub(super) locked_gpus: Vec<LockedGpu>,
}

/// Clock degradation per pipeline stage of a placed configuration.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct StageClockDegradation {
    stages: Vec<StageClock>,
}

impl StageClockDegradation {
    pub(super) fn from_placement(
        cluster: &Cluster,
        placement: &RankPlacement,
        groups: &ParallelGroups,
        stages: u32,
    ) -> Self {
        let stages = (0..stages.max(1))
            .map(|stage| {
                let ranks: Vec<RankId> = match groups.pipeline_stages.get(stage as usize) {
                    Some(stage_ranks) if !stage_ranks.is_empty() => stage_ranks.clone(),
                    _ => (0..placement.rank_to_gpu.len() as RankId).collect(),
                };
                let mut degradation = ClockDegradation::NOMINAL;
                let mut locked_gpus = Vec::new();
                for addr in ranks
                    .iter()
                    .filter_map(|rank| placement.gpu_for_rank(*rank))
                    .collect::<BTreeSet<_>>()
                {
                    let Some(profile) = cluster.gpu_profile(addr) else {
                        continue;
                    };
                    let gpu_degradation = profile.clock_degradation();
                    degradation = degradation.slowest(gpu_degradation);
                    if let Some(clock) = profile.clock
                        && clock.locked_clock().is_some()
                    {
                        locked_gpus.push(LockedGpu { addr, clock });
                    }
                }
                StageClock {
                    degradation,
                    locked_gpus,
                }
            })
            .collect();
        Self { stages }
    }

    pub(super) fn is_nominal(&self) -> bool {
        self.stages
            .iter()
            .all(|stage| stage.degradation.is_nominal())
    }

    pub(super) fn stages(&self) -> &[StageClock] {
        &self.stages
    }

    pub(super) fn has_locked_gpus(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| !stage.locked_gpus.is_empty())
    }

    /// Evidence for every stage that carries a locked GPU, with the stage's
    /// factors and the resulting compute-latency scale; `None` when no placed
    /// GPU is locked.
    pub(super) fn approximation(
        &self,
        phase: &str,
        stage_latency_scales: &[f64],
    ) -> Option<SimulationApproximation> {
        if !self.has_locked_gpus() {
            return None;
        }
        let mut parts = Vec::new();
        let mut insensitive = false;
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            if stage.locked_gpus.is_empty() {
                continue;
            }
            let gpus = stage
                .locked_gpus
                .iter()
                .map(|locked| {
                    let model = locked.clock.model();
                    let knee = match model.bandwidth() {
                        BandwidthClockSensitivity::Insensitive => {
                            insensitive = true;
                            "none".to_string()
                        }
                        BandwidthClockSensitivity::Knee(knee) => format!("{:.0}", knee.mhz()),
                    };
                    format!(
                        "node {} gpu {} locked {:.0} MHz (sustained {:.0}, bandwidth knee {knee})",
                        locked.addr.node_id,
                        locked.addr.local_gpu_id,
                        locked.clock.locked_clock().map_or(0.0, |clock| clock.mhz()),
                        model.sustained().mhz(),
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let scale = stage_latency_scales.get(stage_idx).copied().unwrap_or(1.0);
            parts.push(format!(
                "stage {stage_idx}: {gpus}; compute x{:.4}, HBM bandwidth x{:.4}, stage compute latency x{:.4}",
                stage.degradation.compute(),
                stage.degradation.bandwidth(),
                scale
            ));
        }
        let mut message = format!(
            "Locked GPU clocks slow only their own pipeline stages (slowest GPU per stage): {}.",
            parts.join("; ")
        );
        if insensitive {
            message.push_str(
                " A GPU without bandwidth_knee_sm_clock_mhz keeps full HBM bandwidth when locked; on the RTX 3090 that missed a 1.26-1.34x decode slowdown (validation ledger 17).",
            );
        }
        Some(SimulationApproximation::new(
            phase,
            "hardware",
            "gpu_clock",
            GPU_CLOCK_DEGRADATION,
            message,
            Some(
                "the compute factor is locked/sustained clock and the bandwidth factor min(1, locked/knee); both clocks were fitted from one locked clock on one RTX 3090 (validation ledger 22), so measure a second clock before relying on other clocks or GPUs"
                    .to_string(),
            ),
        ))
    }
}

impl Solver {
    /// Per-stage multipliers on the nominal per-layer compute duration:
    /// `analytical(stage-degraded) / analytical(nominal)`, exactly 1.0 for a
    /// stage whose GPUs are all nominal.
    pub(super) fn stage_latency_scales(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        calibration: SimulationCalibration,
        degradation: &StageClockDegradation,
    ) -> Vec<f64> {
        if degradation.is_nominal() {
            return vec![1.0; degradation.stages().len()];
        }
        let peak_flops = Self::peak_flops(cluster, model, placement);
        let hbm_bandwidth = Self::effective_hbm_bandwidth(cluster, placement, calibration);
        let nominal = Self::analytical_phase_latency_s(
            model,
            request,
            config,
            peak_flops,
            hbm_bandwidth,
            calibration,
        );
        degradation
            .stages()
            .iter()
            .map(|stage| {
                if stage.degradation.is_nominal() || nominal <= 0.0 {
                    return 1.0;
                }
                Self::analytical_phase_latency_s(
                    model,
                    request,
                    config,
                    peak_flops * stage.degradation.compute(),
                    hbm_bandwidth * stage.degradation.bandwidth(),
                    calibration,
                ) / nominal
            })
            .collect()
    }

    /// The roofline phase latency without calibration fits, for given
    /// effective peak FLOP/s (before the efficiency factor) and bandwidth.
    fn analytical_phase_latency_s(
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        peak_flops: f64,
        hbm_bandwidth: f64,
        calibration: SimulationCalibration,
    ) -> f64 {
        let shard_factor = Self::latency_shard_factor(config);
        let prefill = || {
            Self::prefill_baseline_s(
                model,
                request,
                config,
                shard_factor,
                peak_flops
                    * calibration.compute_efficiency_at(Self::prefill_tokens_per_pass(request)),
                hbm_bandwidth,
                calibration,
            )
        };
        let decode = || {
            Self::decode_compute_latency_s(
                model,
                request,
                config,
                model.parameter_count(),
                shard_factor,
                peak_flops
                    * calibration.compute_efficiency_at(Self::decode_tokens_per_pass(request)),
                hbm_bandwidth,
                calibration,
                None,
            )
            .0
        };
        match request.phase {
            InferencePhase::Prefill => prefill(),
            InferencePhase::Decode => decode(),
            InferencePhase::EndToEnd => prefill() + decode(),
        }
    }
}
