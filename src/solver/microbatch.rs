//! Pipeline overlap across the engine steps one static batch is split into.
//!
//! vLLM 0.29's V1 engine core keeps `max_concurrent_batches =
//! pipeline_parallel_size` scheduled steps in flight (its batch queue) when
//! pipeline parallelism is on and async scheduling is off, which is the case
//! for the Ray executor. Offline requests reach the engine-core process one
//! at a time, and its busy loop starts a step as soon as the first one
//! arrives: that step carries one request, and because the batch queue is not
//! full the next step is scheduled immediately with the requests that arrived
//! in the meantime, while the first step is still on stage 0. A request whose
//! prompt is in flight is not schedulable again, so the two steps stay apart
//! and overlap across stages. A worker waits for its previous step's pipeline
//! send before it starts the next forward pass on that stage.
//!
//! `MicrobatchPlan` encodes that split; `Solver::build_microbatched_trace`
//! lays the prompt passes of both microbatches into one operation trace so the
//! resource scheduler overlaps them, then runs the whole batch's decode after
//! both. Decode is not split: both microbatches read every stage's weights
//! per token, so alternating them costs about the serial round trip.

use std::num::NonZeroU32;

use super::{
    operations::{PassLinks, TraceBuilder},
    *,
};
use crate::calibration::PipelineMicrobatching;

/// Approximation code recorded when a static batch is priced as overlapping
/// microbatches.
pub(super) const PIPELINE_MICROBATCH_OVERLAP: &str = "pipeline_microbatch_overlap";

/// How one static batch is divided into engine steps that share the pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum MicrobatchPlan {
    /// One forward pass carries the whole batch.
    WholeBatch,
    /// The first step carries `first` requests; the next step, issued while
    /// the first is in flight, carries the remaining `rest`.
    FirstThenRest { first: NonZeroU32, rest: NonZeroU32 },
}

impl MicrobatchPlan {
    /// The split `policy` produces for `batch_size` requests on
    /// `pipeline_ranks` stages. Only multi-request batches on two or more
    /// stages split.
    pub(super) fn for_static_batch(
        policy: PipelineMicrobatching,
        batch_size: u32,
        pipeline_ranks: u32,
    ) -> Self {
        match policy {
            PipelineMicrobatching::WholeBatch => Self::WholeBatch,
            PipelineMicrobatching::VllmBatchQueue => {
                if pipeline_ranks < 2 {
                    return Self::WholeBatch;
                }
                match (
                    NonZeroU32::new(1),
                    NonZeroU32::new(batch_size.saturating_sub(1)),
                ) {
                    (Some(first), Some(rest)) => Self::FirstThenRest { first, rest },
                    _ => Self::WholeBatch,
                }
            }
        }
    }

    /// Microbatch sizes in issue order for a batch of `batch_size`.
    #[cfg(test)]
    pub(super) fn microbatch_sizes(self, batch_size: u32) -> Vec<NonZeroU32> {
        match self {
            Self::WholeBatch => NonZeroU32::new(batch_size).into_iter().collect(),
            Self::FirstThenRest { first, rest } => vec![first, rest],
        }
    }
}

/// Compute latency of each microbatch's prompt pass and, for end-to-end, the
/// whole batch's decode.
pub(super) struct MicrobatchCompute {
    pub(super) prompt_passes: Vec<(InferenceRequest, f64)>,
    pub(super) decode_pass: Option<(InferenceRequest, f64)>,
}

impl Solver {
    /// Prices the passes of a split static batch, or `None` when the batch
    /// runs whole: a single microbatch, a decode-only phase, or a phase whose
    /// latency a calibration fit already supplies (a fit is a measured
    /// whole-batch latency and already contains whatever overlap the backend
    /// achieved).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn microbatch_compute(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        calibration: SimulationCalibration,
        calibration_fits: &[CalibrationFitApplication],
    ) -> Option<MicrobatchCompute> {
        if !calibration_fits.is_empty() || request.phase == InferencePhase::Decode {
            return None;
        }
        let MicrobatchPlan::FirstThenRest { first, rest } = MicrobatchPlan::for_static_batch(
            calibration.pipeline_microbatching,
            request.batch_size,
            config.pipeline_ranks,
        ) else {
            return None;
        };
        let pass = |batch_size: u32, phase: InferencePhase| {
            let pass_request = InferenceRequest {
                batch_size,
                phase,
                ..*request
            };
            let (latency_s, _) = Self::estimate_compute_latency_s(
                cluster,
                model,
                &pass_request,
                config,
                placement,
                calibration,
                None,
            );
            (pass_request, latency_s)
        };
        let prompt_passes = vec![
            pass(first.get(), InferencePhase::Prefill),
            pass(rest.get(), InferencePhase::Prefill),
        ];
        let decode_pass = (request.phase == InferencePhase::EndToEnd)
            .then(|| pass(request.batch_size, InferencePhase::Decode));
        Some(MicrobatchCompute {
            prompt_passes,
            decode_pass,
        })
    }

    /// Operation trace of a split static batch: each microbatch's prompt pass
    /// (names prefixed `microbatch N `), the second waiting on the first's
    /// send out of every stage, then the whole batch's decode pass (prefixed
    /// `decode `) after both.
    pub(super) fn build_microbatched_trace(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        placement: &RankPlacement,
        groups: &ParallelGroups,
        calibration: SimulationCalibration,
        compute: &MicrobatchCompute,
    ) -> (Vec<SimOperation>, Vec<PricedCollective>) {
        let mut trace = TraceBuilder::new(cluster, model, config, placement, groups, calibration);
        let mut prompt_tails = Vec::new();
        let mut previous_sends: Vec<Option<usize>> = Vec::new();
        for (index, (pass_request, latency_s)) in compute.prompt_passes.iter().enumerate() {
            let links = PassLinks {
                name_prefix: format!("microbatch {index} "),
                entry: Vec::new(),
                stage_entry: previous_sends
                    .iter()
                    .map(|send| send.iter().copied().collect())
                    .collect(),
            };
            let pass = trace.push_pass(pass_request, *latency_s, &links);
            prompt_tails.extend(pass.tails);
            previous_sends = pass.stage_sends;
        }
        if let Some((decode_request, latency_s)) = &compute.decode_pass {
            let links = PassLinks {
                name_prefix: "decode ".to_string(),
                entry: prompt_tails,
                stage_entry: Vec::new(),
            };
            trace.push_pass(decode_request, *latency_s, &links);
        }
        trace.finish()
    }
}

pub(super) fn microbatch_overlap_approximation(
    request: &InferenceRequest,
    compute: &MicrobatchCompute,
) -> SimulationApproximation {
    let sizes: Vec<String> = compute
        .prompt_passes
        .iter()
        .map(|(pass, _)| pass.batch_size.to_string())
        .collect();
    SimulationApproximation::new(
        phase_label(request.phase),
        "scheduling",
        "pipeline",
        PIPELINE_MICROBATCH_OVERLAP,
        format!(
            "The batch of {} is priced as vLLM batch-queue steps of {} requests whose prompt passes overlap across pipeline stages; the real split depends on when requests reach the engine core and varies run to run, and decode runs as one whole-batch pass.",
            request.batch_size,
            sizes.join(" + ")
        ),
        Some(
            "set calibration.pipeline_microbatching = \"whole_batch\" for backends that run a batch as one pipeline pass"
                .to_string(),
        ),
    )
}
