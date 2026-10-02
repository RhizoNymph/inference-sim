//! Disaggregated and partially disaggregated serving on the iteration engine.
//!
//! Workers are routed GPU sets. A request whose prefill and decode routes
//! name the same GPU set runs colocated on that worker; otherwise it
//! prefills on its prefill worker, hands off, is pulled over the topology,
//! and decodes on its decode worker (`core`). Prefill-only workers are priced
//! with the prefill parallelism config, every other worker with the decode
//! config (a worker serving both phases requires the two configs to match).

use std::collections::BTreeSet;

use super::core::run_engine_jobs;
use super::limits::{EngineWorker, engine_limits, engine_request};
use super::transfer::{
    KvPlanContext, KvPlanError, KvTransferPlanner, PlacedKvLayout, PlannedTransfer,
};
use super::*;
use crate::solver::{IterationCostModel, StepLatency, StepWork};

/// Why a candidate's routed worker sets cannot be expressed as engine workers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::serving) enum WorkerLayoutError {
    /// Two distinct routed GPU sets share some but not all GPUs.
    PartiallyOverlappingWorkers,
    /// One GPU set serves both phases but prefill and decode use different
    /// parallelism configs or placements.
    SharedWorkerWithSplitParallelism,
}

/// Engine workers and each state's prefill and decode worker.
pub(in crate::serving) struct DisaggregatedWorkers {
    pub(in crate::serving) workers: Vec<EngineWorker>,
    /// True when the worker only ever prefills (priced with the prefill config).
    pub(in crate::serving) prefill_only: Vec<bool>,
    pub(in crate::serving) prefill_of: Vec<WorkerId>,
    pub(in crate::serving) decode_of: Vec<WorkerId>,
}

/// Groups states' prefill and decode routes into workers.
pub(in crate::serving) fn disaggregated_workers(
    states: &[DecodeRequestState],
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
) -> Result<DisaggregatedWorkers, WorkerLayoutError> {
    let mut workers: Vec<EngineWorker> = Vec::new();
    let mut prefills = Vec::new();
    let mut decodes = Vec::new();
    let mut prefill_of = Vec::with_capacity(states.len());
    let mut decode_of = Vec::with_capacity(states.len());
    let mut worker_for = |gpus: Vec<GpuAddr>, state_idx: usize| -> WorkerId {
        let worker = match workers.iter().position(|worker| worker.gpus == gpus) {
            Some(worker) => worker,
            None => {
                workers.push(EngineWorker::on_gpus(gpus));
                workers.len() - 1
            }
        };
        if workers[worker].members.last() != Some(&state_idx) {
            workers[worker].members.push(state_idx);
        }
        worker
    };
    for (state_idx, state) in states.iter().enumerate() {
        let prefill = worker_for(
            route_worker_gpus(&state.prefill_route_gpus, state.prefill_node),
            state_idx,
        );
        let decode = worker_for(
            route_worker_gpus(&state.decode_route_gpus, state.decode_node),
            state_idx,
        );
        prefill_of.push(prefill);
        decode_of.push(decode);
        prefills.push(prefill);
        decodes.push(decode);
    }
    for (index, left) in workers.iter().enumerate() {
        let left_gpus = left.gpus.iter().collect::<BTreeSet<_>>();
        if workers[index + 1..]
            .iter()
            .any(|right| right.gpus.iter().any(|gpu| left_gpus.contains(gpu)))
        {
            return Err(WorkerLayoutError::PartiallyOverlappingWorkers);
        }
    }
    let decode_workers = decodes.iter().copied().collect::<BTreeSet<_>>();
    let prefill_workers = prefills.iter().copied().collect::<BTreeSet<_>>();
    let split_config = prefill_score.config != decode_score.config;
    if split_config
        && prefill_workers
            .intersection(&decode_workers)
            .next()
            .is_some()
    {
        return Err(WorkerLayoutError::SharedWorkerWithSplitParallelism);
    }
    let prefill_only = (0..workers.len())
        .map(|worker| !decode_workers.contains(&worker))
        .collect();
    Ok(DisaggregatedWorkers {
        workers,
        prefill_only,
        prefill_of,
        decode_of,
    })
}

/// Prices prefill-only workers with the prefill config, the rest with decode.
struct RoleCosts<'a> {
    prefill: IterationCostModel<'a>,
    decode: IterationCostModel<'a>,
    prefill_only: Vec<bool>,
}

impl WorkerStepCost for RoleCosts<'_> {
    fn worker_step_latency(&mut self, worker: WorkerId, work: &StepWork) -> StepLatency {
        if self.prefill_only.get(worker).copied().unwrap_or(false) {
            self.prefill.step_latency(work)
        } else {
            self.decode.step_latency(work)
        }
    }
}

/// Inputs of one disaggregated engine run.
pub(in crate::serving) struct DisaggregatedRun<'a> {
    pub(in crate::serving) cluster: &'a Cluster,
    pub(in crate::serving) model: &'a ModelSpec,
    pub(in crate::serving) prefill_score: &'a ScoredParallelismConfig,
    pub(in crate::serving) decode_score: &'a ScoredParallelismConfig,
    pub(in crate::serving) traffic: &'a ServingTraffic,
    pub(in crate::serving) calibration: SimulationCalibration,
    pub(in crate::serving) calibration_profile: Option<&'a CalibrationProfileMetadata>,
}

/// KV footprint a prefill worker holds for a disaggregated request: the
/// prompt plus the one token the prefill instance samples (vLLM runs the
/// prefill request with `max_tokens = 1`).
fn prefill_footprint(state: &DecodeRequestState) -> KvFootprint {
    let sequences = u64::from(state.batch_size.max(1));
    let tokens_per_sequence = u64::from(state.prompt_tokens.max(1)) + 1;
    let block_tokens = u64::from(state.kv_block_tokens.max(1));
    KvFootprint {
        sequences,
        tokens: sequences * tokens_per_sequence,
        blocks: sequences * tokens_per_sequence.div_ceil(block_tokens),
    }
}

fn first_token_source(traffic: &ServingTraffic) -> FirstTokenSource {
    match traffic.disaggregated_first_token {
        ServingDisaggregatedFirstToken::DecodeInstance => FirstTokenSource::DecodeInstance,
        ServingDisaggregatedFirstToken::PrefillInstance => FirstTokenSource::PrefillInstance,
    }
}

/// The layout of `score` on a worker; the worker's GPU set is authoritative
/// when routing chose GPUs the placement does not name.
fn worker_layout(
    score: &ScoredParallelismConfig,
    model: &ModelSpec,
    routed_node: NodeId,
    worker: &EngineWorker,
) -> Result<PlacedKvLayout, KvPlanError> {
    let placed = PlacedKvLayout::from_score(score, model.kv_heads, routed_node)?;
    let mut placed_gpus = placed.gpus().to_vec();
    placed_gpus.sort_unstable();
    placed_gpus.dedup();
    if placed_gpus == worker.gpus {
        return Ok(placed);
    }
    let config = score.config;
    PlacedKvLayout::new(
        super::transfer::KvShardLayout::new(
            config.tensor_ranks,
            config.pipeline_ranks,
            config.expert_ranks,
            model.kv_heads,
        ),
        worker.gpus.clone(),
    )
}

/// Runs every request through the engine and writes its lifecycle into
/// `states`.
pub(in crate::serving) fn run_disaggregated_engine(
    states: &mut [DecodeRequestState],
    run: DisaggregatedRun<'_>,
) -> Result<EngineTimeline, IterationEngineError> {
    let layout = disaggregated_workers(states, run.prefill_score, run.decode_score)
        .map_err(IterationEngineError::WorkerLayout)?;
    let limits = engine_limits(run.traffic, &layout.workers);
    let mut planner = KvTransferPlanner::new(KvPlanContext {
        cluster: run.cluster,
        calibration: run.calibration,
        calibration_profile: run.calibration_profile,
        bytes_per_token: KvPlanContext::kv_bytes_per_token(run.model),
    });
    let first_token = first_token_source(run.traffic);
    let mut jobs = Vec::with_capacity(states.len());
    let mut transfers: Vec<Option<PlannedTransfer>> = Vec::with_capacity(states.len());
    for (state_idx, state) in states.iter().enumerate() {
        let prefill_worker = layout.prefill_of[state_idx];
        let decode_worker = layout.decode_of[state_idx];
        let mut request = engine_request(state, run.traffic, &limits, prefill_worker);
        add_frontend_latency(&mut request, state, run.calibration);
        if prefill_worker == decode_worker {
            jobs.push(EngineJob::colocated(request));
            transfers.push(None);
            continue;
        }
        let source = worker_layout(
            run.prefill_score,
            run.model,
            state.prefill_node,
            &layout.workers[prefill_worker],
        )
        .map_err(IterationEngineError::KvPlan)?;
        let destination = worker_layout(
            run.decode_score,
            run.model,
            state.decode_node,
            &layout.workers[decode_worker],
        )
        .map_err(IterationEngineError::KvPlan)?;
        let planned = planner
            .plan(
                &source,
                &destination,
                request.sequences,
                request.prompt_tokens(),
            )
            .map_err(IterationEngineError::KvPlan)?;
        let decode_footprint = request.footprint;
        request.footprint = prefill_footprint(state);
        jobs.push(EngineJob {
            request,
            route: JobRoute::Disaggregated(Handoff {
                decode_worker,
                decode_footprint,
                plan: planned.plan.clone(),
                first_token,
            }),
        });
        transfers.push(Some(planned));
    }
    let mut costs = RoleCosts {
        prefill: IterationCostModel::new(
            run.cluster,
            run.model,
            run.prefill_score,
            run.calibration,
        )
        .map_err(IterationEngineError::StepCost)?,
        decode: IterationCostModel::new(run.cluster, run.model, run.decode_score, run.calibration)
            .map_err(IterationEngineError::StepCost)?,
        prefill_only: layout.prefill_only.clone(),
    };
    let outcome =
        run_engine_jobs(&jobs, &limits, &mut costs).map_err(IterationEngineError::Engine)?;
    let recorded = record_disaggregated_outcome(states, &jobs, &layout, &transfers, &outcome);
    Ok(recorded)
}

fn record_disaggregated_outcome(
    states: &mut [DecodeRequestState],
    jobs: &[EngineJob],
    layout: &DisaggregatedWorkers,
    transfers: &[Option<PlannedTransfer>],
    outcome: &EngineOutcome,
) -> EngineTimeline {
    let recorded = record_engine_jobs(states, jobs, &layout.workers, transfers, outcome);
    EngineTimeline {
        operations: recorded.operations,
        decode_iterations: recorded.decode_iterations,
        kv_bottlenecks: recorded.kv_bottlenecks,
    }
}
