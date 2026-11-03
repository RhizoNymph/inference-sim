//! Test-only entry point that runs the engine with one non-default setting.
//!
//! The preemption tests pass `(KvAllocationMode, cost)` and the fair-share
//! tests `(cost, HandoffModel)`; `EngineRunArgs` accepts exactly those two
//! argument pairs at compile time and fills the other setting with its
//! default. Production code calls `run_engine_jobs_configured` with both.

use super::super::types::{
    EngineJob, EngineLimits, EngineOutcome, HandoffModel, KvAllocationMode, WorkerStepCost,
};
use super::{EngineError, run_engine_jobs_configured};

/// An argument pair `run_engine_jobs_with` accepts.
pub(in crate::serving) trait EngineRunArgs {
    fn run(self, jobs: &[EngineJob], limits: &EngineLimits) -> Result<EngineOutcome, EngineError>;
}

/// KV allocation mode with the default handoff model.
impl<C: WorkerStepCost> EngineRunArgs for (KvAllocationMode, &mut C) {
    fn run(self, jobs: &[EngineJob], limits: &EngineLimits) -> Result<EngineOutcome, EngineError> {
        let (kv, cost) = self;
        run_engine_jobs_configured(jobs, limits, kv, HandoffModel::default(), cost)
    }
}

/// Handoff model with KV reserved at admission.
impl<C: WorkerStepCost> EngineRunArgs for (&mut C, HandoffModel) {
    fn run(self, jobs: &[EngineJob], limits: &EngineLimits) -> Result<EngineOutcome, EngineError> {
        let (cost, handoff) = self;
        run_engine_jobs_configured(
            jobs,
            limits,
            KvAllocationMode::ReservedAtAdmission,
            handoff,
            cost,
        )
    }
}

/// Simulates `jobs` with either a KV allocation mode or a handoff model set
/// (see `EngineRunArgs`).
pub(in crate::serving) fn run_engine_jobs_with<A, B>(
    jobs: &[EngineJob],
    limits: &EngineLimits,
    first: A,
    second: B,
) -> Result<EngineOutcome, EngineError>
where
    (A, B): EngineRunArgs,
{
    (first, second).run(jobs, limits)
}
