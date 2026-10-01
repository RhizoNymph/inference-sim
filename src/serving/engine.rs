//! Iteration-level serving engine for colocated continuous-batching serving.
//!
//! When prefill and decode share one engine (same parallelism config, same
//! routed GPUs, continuous batching on both phases), requests are simulated by
//! a discrete-event loop of engine steps modeled on vLLM V1 (see `core`):
//! each step decodes one token for every running sequence and fills the rest
//! of the token budget with prefill chunks, admitting waiting requests only
//! when their KV blocks fit. Step latency comes from the solver's
//! `IterationCostModel` for the step's actual composition. Everything else
//! (disaggregated pools, independent batching, split prefill/decode configs,
//! data-parallel replicas) keeps the phase pipeline scheduler.

use super::*;

mod capacity;
mod core;
mod limits;
mod record;
#[cfg(test)]
mod tests;
mod types;

use crate::solver::IterationCostModel;
pub(super) use core::EngineError;
use core::run_engine;
use limits::{engine_limits, engine_requests, engine_workers};
use record::record_engine_outcome;
pub(super) use types::*;

/// Which scheduler simulates a serving candidate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum SchedulerModel {
    IterationEngine,
    PhasePipeline(PhasePipelineReason),
}

/// Why a candidate cannot run on the iteration engine.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum PhasePipelineReason {
    IndependentPrefillBatching,
    IndependentDecodeBatching,
    SplitPrefillDecodeParallelism,
    DataParallelReplicas,
    DisaggregatedRoutes,
}

impl PhasePipelineReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::IndependentPrefillBatching => "independent_prefill_batching",
            Self::IndependentDecodeBatching => "independent_decode_batching",
            Self::SplitPrefillDecodeParallelism => "split_prefill_decode_parallelism",
            Self::DataParallelReplicas => "data_parallel_replicas",
            Self::DisaggregatedRoutes => "disaggregated_routes",
        }
    }
}

pub(super) fn select_scheduler_model(
    traffic: &ServingTraffic,
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    states: &[DecodeRequestState],
) -> SchedulerModel {
    use PhasePipelineReason as Reason;
    if matches!(
        traffic.prefill_batching,
        ServingPrefillBatching::Independent
    ) {
        return SchedulerModel::PhasePipeline(Reason::IndependentPrefillBatching);
    }
    if matches!(traffic.decode_batching, ServingDecodeBatching::Independent) {
        return SchedulerModel::PhasePipeline(Reason::IndependentDecodeBatching);
    }
    if prefill_score.config != decode_score.config
        || prefill_score.placement != decode_score.placement
        || decode_score.placement.rank_to_gpu.is_empty()
    {
        return SchedulerModel::PhasePipeline(Reason::SplitPrefillDecodeParallelism);
    }
    if decode_score.config.data_ranks > 1 {
        return SchedulerModel::PhasePipeline(Reason::DataParallelReplicas);
    }
    if states
        .iter()
        .any(|state| state.prefill_route_gpus != state.decode_route_gpus)
    {
        return SchedulerModel::PhasePipeline(Reason::DisaggregatedRoutes);
    }
    SchedulerModel::IterationEngine
}

/// Result of simulating a candidate on the iteration engine.
pub(super) struct EngineTimeline {
    pub(super) operations: Vec<ScheduledOperation>,
    pub(super) decode_iterations: Vec<ServingDecodeIterationObservation>,
}

#[derive(Debug)]
pub(super) enum IterationEngineError {
    StepCost(crate::solver::StepCostError),
    Engine(EngineError),
}

impl std::fmt::Display for IterationEngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepCost(error) => write!(formatter, "iteration engine step cost: {error}"),
            Self::Engine(error) => write!(formatter, "iteration engine: {error}"),
        }
    }
}

impl std::error::Error for IterationEngineError {}

/// Runs every request through the iteration engine and writes its lifecycle
/// into `states`, which the shared summary then turns into metrics.
pub(super) fn run_iteration_engine(
    states: &mut [DecodeRequestState],
    cluster: &Cluster,
    model: &ModelSpec,
    score: &ScoredParallelismConfig,
    traffic: &ServingTraffic,
    calibration: SimulationCalibration,
) -> Result<EngineTimeline, IterationEngineError> {
    let mut cost = IterationCostModel::new(cluster, model, score, calibration)
        .map_err(IterationEngineError::StepCost)?;
    let workers = engine_workers(states);
    let limits = engine_limits(traffic, &workers);
    let requests = engine_requests(states, traffic, &workers, &limits);
    let outcome =
        run_engine(&requests, &limits, &mut cost).map_err(IterationEngineError::Engine)?;
    let (operations, decode_iterations) =
        record_engine_outcome(states, &requests, &workers, &outcome);
    Ok(EngineTimeline {
        operations,
        decode_iterations,
    })
}

/// Approximation records attached to every candidate the engine simulated.
pub(super) fn iteration_engine_approximations(
    traffic: &ServingTraffic,
    score: &ScoredParallelismConfig,
) -> Vec<SimulationApproximation> {
    let mut approximations = vec![
        SimulationApproximation::new(
            "serving",
            "queueing",
            "iteration_engine",
            "iteration_engine_kv_reserved_at_admission",
            "The iteration engine reserves KV blocks for a request's full max_sequence_tokens when it is admitted and never preempts; vLLM allocates blocks as tokens are generated and recomputes preempted requests when blocks run out.",
            Some(
                "keep KV capacity comfortably above max_num_seqs * max_sequence_tokens, or treat queueing near KV exhaustion as optimistic"
                    .to_string(),
            ),
        ),
        SimulationApproximation::new(
            "serving",
            "runtime",
            "iteration_engine",
            "iteration_engine_no_frontend_overhead",
            "Step latency covers the forward pass and the calibrated per-step overhead only; API-server tokenization, detokenization, and HTTP streaming time are not modeled, so client-observed TTFT is underestimated at low load.",
            Some("add a measured per-request frontend latency to TTFT when comparing against client-side benchmarks".to_string()),
        ),
    ];
    if score.config.pipeline_ranks > 1 {
        approximations.push(SimulationApproximation::new(
            "serving",
            "runtime",
            "iteration_engine",
            "iteration_engine_pipeline_stages_serialized",
            "With pipeline parallelism each engine step runs its stages back to back; vLLM keeps several micro-batches in flight across stages, so steady-state throughput is underestimated.",
            Some("use a pipeline-aware engine model before relying on pipeline-parallel serving throughput".to_string()),
        ));
    }
    if traffic.max_prefill_tokens.is_some() {
        approximations.push(SimulationApproximation::new(
            "serving",
            "capacity",
            "iteration_engine",
            "iteration_engine_prefill_cap_per_worker_step",
            "max_prefill_tokens is enforced per worker step rather than across concurrent steps of different workers.",
            None,
        ));
    }
    approximations
}

/// Approximation record for a candidate that kept the phase pipeline.
pub(super) fn phase_pipeline_approximation(reason: PhasePipelineReason) -> SimulationApproximation {
    SimulationApproximation::new(
        "serving",
        "queueing",
        reason.as_str(),
        "phase_pipeline_scheduler",
        format!(
            "This candidate is scheduled phase by phase (all prefills, then KV transfers, then decode iterations priced by scaling a reference batch) instead of the iteration engine because of {}; decode iterations do not interleave with prefill chunks and capacity overruns reject instead of queueing.",
            reason.as_str().replace('_', " ")
        ),
        Some(
            "use continuous prefill and decode batching on a colocated pool with one parallelism config for iteration-level serving predictions"
                .to_string(),
        ),
    )
}
