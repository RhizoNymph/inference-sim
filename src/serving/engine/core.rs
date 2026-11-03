//! The deterministic discrete-event iteration loop, modeled on vLLM V1's
//! scheduler.
//!
//! Every worker repeatedly runs one engine step:
//!
//! 1. arrivals up to the worker's clock join its waiting queue (priority,
//!    then arrival order); requests that can never fit are rejected;
//! 2. cancelled and queue-timed-out requests leave;
//! 3. running requests are served in admission order: each decoding request
//!    takes one token per lockstep sequence, each prefilling request takes the
//!    largest prefill chunk the remaining token budget allows;
//! 4. while budget remains, the head of the waiting queue is admitted if its
//!    KV footprint fits (otherwise it and everything behind it waits) and takes
//!    a first chunk;
//! 5. the step's latency comes from its composition; at its end decoding
//!    requests emit a token, a request whose prompt completes emits its first
//!    token, and finished requests release their KV at that instant.
//!
//! Disaggregated requests (`JobRoute::Disaggregated`) prefill on one
//! worker and decode on another (vLLM NixlConnector semantics):
//!
//! - the prefill step that completes the prompt samples the prefill
//!   instance's token; the request leaves the prefill worker's running set
//!   (its sequence slot frees) but its prompt KV stays there;
//! - it joins the decode worker's waiting queue at that instant; admission
//!   there (in queue order, when its decode KV fits) reserves the decode KV and
//!   starts the KV pull on the FIFO link queues; it takes no token budget;
//! - when the pull completes, the prefill worker's KV is released and the
//!   request becomes runnable on the decode worker at its next step: under
//!   `FirstTokenSource::DecodeInstance` it first recomputes its last prompt
//!   token (a one-token chunk that samples the client's first token), under
//!   `PrefillInstance` it decodes token 2 directly.
//!
//! Under `KvAllocationMode::PagedRecompute` (vLLM V1 recompute preemption) a
//! request holds blocks only for the tokens it has computed: admission needs
//! blocks for its first chunk, and each scheduled item grows its holding
//! before it runs. When a running request needs a block and none is free,
//! the lowest-priority, then most recently admitted, running request that
//! has not been scheduled in this step is preempted (possibly the requester
//! itself): it frees every block, goes to the front of the waiting queue, and
//! on readmission recomputes its whole context (prompt plus emitted tokens)
//! as prefill through the normal chunked token budget; the pass's last chunk
//! samples its next token. A step that preempted admits nothing new. A
//! disaggregated request preempted on its decode worker recomputes there; it
//! is never pulled again.
//!
//! Workers interact only through shared capacity limits, handoffs, and link
//! queues, so the loop always advances the worker with the earliest clock;
//! transfers are therefore reserved in non-decreasing time order.

mod budget;
mod paged;

use std::collections::VecDeque;

use crate::solver::StepWork;

use super::capacity::{CapacityLedger, HoldingKey};
use super::transfer::LinkQueues;
use super::types::{
    ChunkPass, ChunkRecord, EngineJob, EngineLimits, EngineOutcome, EngineRequest, EngineRequestId,
    EngineStep, FirstTokenSource, HandoffRecord, KvAllocationMode, RequestFate, RequestTimeline,
    TokenRecord, WorkerId, WorkerStepCost,
};
use budget::{StepBudget, add_chunk};
use paged::KvGrant;

const TIME_EPSILON_S: f64 = 1e-12;
/// Nudge for a blocked worker that waits on another worker sharing its clock,
/// so the other worker steps first.
const BLOCKED_WAKE_NUDGE_S: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::serving) enum EngineError {
    /// A request names a worker the limits do not describe.
    UnknownWorker {
        request: EngineRequestId,
        worker: WorkerId,
    },
    /// A disaggregated request's decode worker is its prefill worker.
    HandoffToSameWorker {
        request: EngineRequestId,
        worker: WorkerId,
    },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownWorker { request, worker } => write!(
                formatter,
                "engine request {request} is routed to unknown worker {worker}"
            ),
            Self::HandoffToSameWorker { request, worker } => write!(
                formatter,
                "engine request {request} hands off from worker {worker} to itself"
            ),
        }
    }
}

impl std::error::Error for EngineError {}

/// Which context a prefill pass computes, and what its last chunk does.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrefillPass {
    /// The prompt (`prefill_tokens` beyond the cached prefix); completing it
    /// samples the first token, or hands a disaggregated request off.
    Prompt,
    /// After a preemption during decode: the prompt plus the `emitted`
    /// output tokens, on the worker that decodes; completing it samples
    /// token `emitted + 1`.
    Resume { emitted: u32 },
}

impl PrefillPass {
    /// Tokens per sequence (beyond the cached prompt) the pass computes.
    fn target(self, request: &EngineRequest) -> u32 {
        match self {
            Self::Prompt => request.prefill_tokens,
            Self::Resume { emitted } => request.prefill_tokens.saturating_add(emitted),
        }
    }

    fn chunk_pass(self) -> ChunkPass {
        match self {
            Self::Prompt => ChunkPass::Prompt,
            Self::Resume { .. } => ChunkPass::Resume,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Phase {
    NotArrived,
    /// Waiting for prefill admission (prefill or colocated worker).
    Waiting,
    Prefilling {
        computed: u32,
        pass: PrefillPass,
    },
    /// Disaggregated: prefill done, waiting for decode admission.
    AwaitingDecode,
    /// Disaggregated: admitted on decode, KV pull completes at `ready_s`.
    Pulling {
        ready_s: f64,
    },
    /// Disaggregated, `DecodeInstance`: recomputes the last prompt token.
    Recomputing,
    Decoding {
        emitted: u32,
    },
    /// Preempted under paged allocation: queued at the front of its worker's
    /// waiting queue to redo `pass` from its first token.
    Preempted {
        pass: PrefillPass,
    },
    Finished,
}

#[derive(Clone, Debug, Default)]
struct WorkerQueue {
    clock_s: f64,
    /// Future arrivals (fresh requests and handoffs), by (time, request index).
    pending: VecDeque<(f64, EngineRequestId)>,
    /// Preempted requests first (most recently preempted at the front, as
    /// vLLM prepends them), then everything else in `queue_order`.
    waiting: Vec<EngineRequestId>,
    running: Vec<EngineRequestId>,
}

#[derive(Default)]
struct StepPlan {
    work: StepWork,
    decodes: Vec<EngineRequestId>,
    chunks: Vec<(EngineRequestId, u32)>,
    recomputes: Vec<EngineRequestId>,
    preempted: Vec<EngineRequestId>,
}

impl StepPlan {
    fn is_empty(&self) -> bool {
        self.decodes.is_empty() && self.chunks.is_empty() && self.recomputes.is_empty()
    }
}

/// Simulates every colocated request to completion (or rejection) with KV
/// reserved at admission and returns the step-by-step record.
#[cfg(test)]
pub(in crate::serving) fn run_engine(
    requests: &[EngineRequest],
    limits: &EngineLimits,
    cost: &mut impl super::types::StepCost,
) -> Result<EngineOutcome, EngineError> {
    let jobs = requests
        .iter()
        .cloned()
        .map(EngineJob::colocated)
        .collect::<Vec<_>>();
    run_engine_jobs(&jobs, limits, &mut super::types::UniformCost(cost))
}

/// Simulates colocated and disaggregated jobs together with KV reserved at
/// admission.
#[cfg(test)]
pub(in crate::serving) fn run_engine_jobs(
    jobs: &[EngineJob],
    limits: &EngineLimits,
    cost: &mut impl WorkerStepCost,
) -> Result<EngineOutcome, EngineError> {
    run_engine_jobs_with(jobs, limits, KvAllocationMode::ReservedAtAdmission, cost)
}

/// Simulates colocated and disaggregated jobs together under `kv`.
pub(in crate::serving) fn run_engine_jobs_with(
    jobs: &[EngineJob],
    limits: &EngineLimits,
    kv: KvAllocationMode,
    cost: &mut impl WorkerStepCost,
) -> Result<EngineOutcome, EngineError> {
    let mut engine = Engine::new(jobs, limits, kv)?;
    engine.run(cost);
    Ok(EngineOutcome {
        steps: engine.steps,
        timelines: engine.timelines,
    })
}

struct Engine<'a> {
    jobs: &'a [EngineJob],
    limits: &'a EngineLimits,
    kv: KvAllocationMode,
    workers: Vec<WorkerQueue>,
    phases: Vec<Phase>,
    timelines: Vec<RequestTimeline>,
    steps: Vec<EngineStep>,
    ledger: CapacityLedger,
    links: LinkQueues,
}

impl<'a> Engine<'a> {
    fn new(
        jobs: &'a [EngineJob],
        limits: &'a EngineLimits,
        kv: KvAllocationMode,
    ) -> Result<Self, EngineError> {
        let mut workers = vec![WorkerQueue::default(); limits.workers.len()];
        let mut order = (0..jobs.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            let (left, right) = (&jobs[*left].request, &jobs[*right].request);
            left.arrival_s
                .total_cmp(&right.arrival_s)
                .then_with(|| left.request_idx.cmp(&right.request_idx))
        });
        for id in order {
            let request = &jobs[id].request;
            let worker = request.worker;
            if let Some(handoff) = jobs[id].handoff() {
                if handoff.decode_worker >= workers.len() {
                    return Err(EngineError::UnknownWorker {
                        request: id,
                        worker: handoff.decode_worker,
                    });
                }
                if handoff.decode_worker == worker {
                    return Err(EngineError::HandoffToSameWorker {
                        request: id,
                        worker,
                    });
                }
            }
            let Some(queue) = workers.get_mut(worker) else {
                return Err(EngineError::UnknownWorker {
                    request: id,
                    worker,
                });
            };
            queue.pending.push_back((request.arrival_s, id));
        }
        Ok(Self {
            jobs,
            limits,
            kv,
            workers,
            phases: vec![Phase::NotArrived; jobs.len()],
            timelines: vec![RequestTimeline::default(); jobs.len()],
            steps: Vec::new(),
            ledger: CapacityLedger::new(limits, jobs.len()),
            links: LinkQueues::new(0),
        })
    }

    fn request(&self, id: EngineRequestId) -> &'a EngineRequest {
        &self.jobs[id].request
    }

    fn run(&mut self, cost: &mut impl WorkerStepCost) {
        while let Some((worker, now_s)) = self.next_worker() {
            self.ledger.apply_releases_until(now_s);
            self.workers[worker].clock_s = now_s;
            self.enqueue_arrivals(worker, now_s);
            self.expire(worker, now_s);
            match self.plan_step(worker, now_s) {
                Some(plan) => self.execute(worker, now_s, plan, cost),
                None => self.wait_when_blocked(worker, now_s),
            }
        }
    }

    /// Earliest time a pull into this worker completes.
    fn next_pull_ready_s(&self, worker: WorkerId) -> Option<f64> {
        self.workers[worker]
            .running
            .iter()
            .filter_map(|id| match self.phases[*id] {
                Phase::Pulling { ready_s } => Some(ready_s),
                _ => None,
            })
            .min_by(f64::total_cmp)
    }

    fn next_event_s(&self, worker: WorkerId) -> Option<f64> {
        let queue = &self.workers[worker];
        let runnable = queue
            .running
            .iter()
            .any(|id| !matches!(self.phases[*id], Phase::Pulling { .. }));
        if runnable || !queue.waiting.is_empty() {
            return Some(queue.clock_s);
        }
        [
            queue.pending.front().map(|(time_s, _)| *time_s),
            self.next_pull_ready_s(worker),
        ]
        .into_iter()
        .flatten()
        .min_by(f64::total_cmp)
        .map(|time_s| time_s.max(queue.clock_s))
    }

    /// When `worker` next changes state on its own: now if it can step,
    /// otherwise its next arrival, pull completion, or waiting deadline.
    fn progress_event_s(&self, worker: WorkerId) -> Option<f64> {
        let queue = &self.workers[worker];
        if queue
            .running
            .iter()
            .any(|id| !matches!(self.phases[*id], Phase::Pulling { .. }))
        {
            return Some(queue.clock_s);
        }
        [
            queue.pending.front().map(|(time_s, _)| *time_s),
            self.next_pull_ready_s(worker),
            self.next_waiting_deadline_s(worker),
        ]
        .into_iter()
        .flatten()
        .min_by(f64::total_cmp)
    }

    fn next_waiting_deadline_s(&self, worker: WorkerId) -> Option<f64> {
        self.workers[worker]
            .waiting
            .iter()
            .filter_map(|id| {
                let request = self.request(*id);
                let timeout_s = (self.phases[*id] == Phase::Waiting)
                    .then_some(request.max_queue_delay_s)
                    .flatten()
                    .map(|limit_s| request.arrival_s + limit_s + BLOCKED_WAKE_NUDGE_S);
                match (request.cancellation_s, timeout_s) {
                    (Some(left), Some(right)) => Some(left.min(right)),
                    (left, right) => left.or(right),
                }
            })
            .min_by(f64::total_cmp)
    }

    fn next_worker(&self) -> Option<(WorkerId, f64)> {
        (0..self.workers.len())
            .filter_map(|worker| self.next_event_s(worker).map(|time_s| (worker, time_s)))
            .min_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)))
    }

    fn enqueue_arrivals(&mut self, worker: WorkerId, now_s: f64) {
        while let Some(&(time_s, id)) = self.workers[worker].pending.front() {
            if time_s > now_s + TIME_EPSILON_S {
                break;
            }
            self.workers[worker].pending.pop_front();
            if self.phases[id] == Phase::NotArrived {
                if let Some(excess) = self.never_fits(id) {
                    self.finish_without_running(id, RequestFate::NeverFits(excess), now_s);
                    continue;
                }
                self.phases[id] = Phase::Waiting;
            }
            let jobs = self.jobs;
            let phases = &self.phases;
            let waiting = &mut self.workers[worker].waiting;
            let preempted = waiting
                .iter()
                .take_while(|queued| matches!(phases[**queued], Phase::Preempted { .. }))
                .count();
            let position = preempted
                + waiting[preempted..].partition_point(|queued| {
                    queue_order(&jobs[*queued].request, &jobs[id].request)
                        != std::cmp::Ordering::Greater
                });
            waiting.insert(position, id);
        }
    }

    /// The first limit either of the request's footprints exceeds on its own.
    fn never_fits(&self, id: EngineRequestId) -> Option<super::types::CapacityExcess> {
        if let KvAllocationMode::PagedRecompute { block_tokens } = self.kv {
            return self.never_fits_paged(id, block_tokens);
        }
        let request = self.request(id);
        self.ledger.never_fits(request).or_else(|| {
            let handoff = self.jobs[id].handoff()?;
            self.ledger.never_fits_on(
                handoff.decode_worker,
                request.class,
                handoff.decode_footprint,
            )
        })
    }

    fn expire(&mut self, worker: WorkerId, now_s: f64) {
        let waiting = std::mem::take(&mut self.workers[worker].waiting);
        let mut kept = Vec::with_capacity(waiting.len());
        for id in waiting {
            let request = self.request(id);
            if let Some(cancellation_s) = request.cancellation_s
                && cancellation_s <= now_s + TIME_EPSILON_S
            {
                self.finish_without_running(
                    id,
                    RequestFate::Cancelled {
                        at_s: cancellation_s,
                    },
                    now_s,
                );
            } else if self.phases[id] == Phase::Waiting
                && let Some(limit_s) = request.max_queue_delay_s
                && now_s - request.arrival_s > limit_s + TIME_EPSILON_S
            {
                let waited_s = now_s - request.arrival_s;
                self.finish_without_running(
                    id,
                    RequestFate::QueueTimeout { waited_s, limit_s },
                    now_s,
                );
            } else {
                kept.push(id);
            }
        }
        self.workers[worker].waiting = kept;

        let running = std::mem::take(&mut self.workers[worker].running);
        let mut kept = Vec::with_capacity(running.len());
        for id in running {
            match self.request(id).cancellation_s {
                Some(cancellation_s) if cancellation_s <= now_s + TIME_EPSILON_S => {
                    self.phases[id] = Phase::Finished;
                    self.timelines[id].fate = Some(RequestFate::Cancelled {
                        at_s: cancellation_s,
                    });
                    self.release_all(id, now_s);
                }
                _ => kept.push(id),
            }
        }
        self.workers[worker].running = kept;
        self.ledger.apply_releases_until(now_s);
    }

    fn plan_step(&mut self, worker: WorkerId, now_s: f64) -> Option<StepPlan> {
        self.promote_pulled(worker, now_s);
        let step_limits = self.limits.workers[worker].step;
        let mut budget = StepBudget::new(step_limits, self.limits);
        let mut plan = StepPlan::default();

        let mut index = 0;
        while let Some(&id) = self.workers[worker].running.get(index) {
            let request = self.request(id);
            let sequences = request.sequences.max(1);
            let item = match self.phases[id] {
                Phase::Decoding { emitted } => {
                    if u64::from(sequences) > budget.tokens && !plan.is_empty() {
                        index += 1;
                        continue;
                    }
                    Item::Decode { emitted }
                }
                Phase::Prefilling { computed, pass } => {
                    let remaining = pass.target(request).saturating_sub(computed);
                    let chunk = budget.chunk(request, remaining, plan.is_empty());
                    if chunk == 0 {
                        index += 1;
                        continue;
                    }
                    Item::Chunk {
                        computed,
                        chunk,
                        target: pass.target(request),
                    }
                }
                Phase::Recomputing => {
                    if u64::from(sequences) > budget.tokens && !plan.is_empty() {
                        index += 1;
                        continue;
                    }
                    Item::Recompute
                }
                Phase::NotArrived
                | Phase::Waiting
                | Phase::AwaitingDecode
                | Phase::Pulling { .. }
                | Phase::Preempted { .. }
                | Phase::Finished => {
                    index += 1;
                    continue;
                }
            };
            match self.grant_kv(
                worker,
                index,
                id,
                item.kv_tokens_after(request),
                now_s,
                &mut plan,
            ) {
                KvGrant::Granted => {}
                KvGrant::SelfPreempted => break,
            }
            match item {
                Item::Decode { emitted } => {
                    budget.tokens = budget.tokens.saturating_sub(u64::from(sequences));
                    plan.work
                        .add_decode(sequences, request.prompt_tokens() + emitted);
                    plan.decodes.push(id);
                }
                Item::Chunk {
                    computed,
                    chunk,
                    target,
                } => {
                    budget.consume_prefill(request, chunk);
                    add_chunk(&mut plan.work, request, computed, chunk, target);
                    plan.chunks.push((id, chunk));
                }
                Item::Recompute => {
                    budget.consume_prefill(request, 1);
                    plan.work.add_completing_prefill_chunk(
                        sequences,
                        request.prompt_tokens().saturating_sub(1),
                        1,
                    );
                    plan.recomputes.push(id);
                }
            }
            index += 1;
        }

        // vLLM admits nothing in a step that preempted; a step left empty by
        // its preemptions admits right away instead of idling.
        if plan.preempted.is_empty() || plan.is_empty() {
            self.admit_waiting(worker, now_s, &mut budget, &mut plan);
        }

        (!plan.is_empty()).then_some(plan)
    }

    /// Admits the head of the waiting queue while it fits and budget remains
    /// (head of line: a head that does not fit blocks everything behind it).
    fn admit_waiting(
        &mut self,
        worker: WorkerId,
        now_s: f64,
        budget: &mut StepBudget,
        plan: &mut StepPlan,
    ) {
        while let Some(&id) = self.workers[worker].waiting.first() {
            if budget.tokens == 0 {
                break;
            }
            let pass = match self.phases[id] {
                Phase::AwaitingDecode => {
                    if !self.admit_for_pull(worker, id, now_s) {
                        break;
                    }
                    continue;
                }
                Phase::Preempted { pass } => pass,
                _ => PrefillPass::Prompt,
            };
            let request = self.request(id);
            let target = pass.target(request);
            let chunk = budget.chunk(request, target, plan.is_empty());
            if chunk == 0 {
                break;
            }
            let footprint = self.admission_footprint(request, chunk);
            if !self.ledger.fits_on(worker, request.class, footprint) {
                break;
            }
            self.workers[worker].waiting.remove(0);
            let key = self.holding_key(id, worker);
            self.ledger
                .allocate_on(key, worker, request.class, footprint);
            self.workers[worker].running.push(id);
            if matches!(self.phases[id], Phase::Preempted { .. }) {
                if let Some(record) = self.timelines[id].preemptions.last_mut() {
                    record.resumed_s = Some(now_s);
                }
            } else {
                self.timelines[id].admitted_s = Some(now_s);
            }
            self.phases[id] = Phase::Prefilling { computed: 0, pass };
            budget.consume_prefill(request, chunk);
            add_chunk(&mut plan.work, request, 0, chunk, target);
            plan.chunks.push((id, chunk));
        }
    }

    /// Pulled requests whose KV has arrived become runnable.
    fn promote_pulled(&mut self, worker: WorkerId, now_s: f64) {
        for index in 0..self.workers[worker].running.len() {
            let id = self.workers[worker].running[index];
            let Phase::Pulling { ready_s } = self.phases[id] else {
                continue;
            };
            if ready_s > now_s + TIME_EPSILON_S {
                continue;
            }
            let first_token = self.jobs[id]
                .handoff()
                .map(|handoff| handoff.first_token)
                .unwrap_or_default();
            self.phases[id] = match first_token {
                FirstTokenSource::DecodeInstance => Phase::Recomputing,
                FirstTokenSource::PrefillInstance => Phase::Decoding { emitted: 1 },
            };
        }
    }

    /// Admits the head of a decode worker's queue for its KV pull when its
    /// decode footprint fits; returns whether it was admitted.
    fn admit_for_pull(&mut self, worker: WorkerId, id: EngineRequestId, now_s: f64) -> bool {
        let job = &self.jobs[id];
        let Some(handoff) = job.handoff() else {
            return false;
        };
        let footprint = self.decode_admission_footprint(&job.request, handoff.decode_footprint);
        if !self.ledger.fits_on(worker, job.request.class, footprint) {
            return false;
        }
        self.workers[worker].waiting.remove(0);
        self.ledger
            .allocate_on(HoldingKey::Decode(id), worker, job.request.class, footprint);
        let window = self.links.reserve(&handoff.plan, now_s, id);
        // The prefill worker frees the prompt KV once the decode side has it.
        self.ledger
            .release_at(HoldingKey::Primary(id), window.finish_s);
        self.phases[id] = Phase::Pulling {
            ready_s: window.finish_s,
        };
        if let Some(record) = self.timelines[id].handoff.as_mut() {
            record.decode_admitted_s = Some(now_s);
            record.transfer = Some(window);
        }
        self.workers[worker].running.push(id);
        true
    }

    fn execute(
        &mut self,
        worker: WorkerId,
        now_s: f64,
        plan: StepPlan,
        cost: &mut impl WorkerStepCost,
    ) {
        let latency = cost.worker_step_latency(worker, &plan.work);
        let finish_s = now_s + latency.total_s.max(0.0);
        let step = self.steps.len();
        let running_sequences = self.ledger.worker_sequences(worker);
        let kv_blocks = self.ledger.worker_blocks(worker);
        let mut token_requests =
            Vec::with_capacity(plan.decodes.len() + plan.chunks.len() + plan.recomputes.len());
        let mut first_token_sequences = 0_u64;
        let mut finished = Vec::new();
        let mut handed_off = Vec::new();

        for &id in &plan.decodes {
            let Phase::Decoding { emitted } = self.phases[id] else {
                continue;
            };
            let emitted = emitted + 1;
            self.timelines[id].tokens.push(TokenRecord {
                step,
                start_s: now_s,
                finish_s,
            });
            token_requests.push(id);
            self.phases[id] = Phase::Decoding { emitted };
            if emitted >= self.request(id).output_tokens {
                finished.push(id);
            }
        }
        let mut prefill_requests = Vec::with_capacity(plan.chunks.len());
        for &(id, chunk) in &plan.chunks {
            let Phase::Prefilling { computed, pass } = self.phases[id] else {
                continue;
            };
            let request = self.request(id);
            let computed = computed + chunk;
            self.timelines[id].chunks.push(ChunkRecord {
                step,
                start_s: now_s,
                finish_s,
                tokens: u64::from(request.sequences.max(1)) * u64::from(chunk),
                pass: pass.chunk_pass(),
            });
            prefill_requests.push(id);
            if computed < pass.target(request) {
                self.phases[id] = Phase::Prefilling { computed, pass };
                continue;
            }
            match pass {
                PrefillPass::Prompt => {
                    let handoff = self.jobs[id].handoff();
                    let client_token = handoff.is_none_or(|handoff| {
                        handoff.first_token == FirstTokenSource::PrefillInstance
                    });
                    if client_token {
                        self.timelines[id].tokens.push(TokenRecord {
                            step,
                            start_s: finish_s,
                            finish_s,
                        });
                        token_requests.push(id);
                        first_token_sequences += u64::from(request.sequences.max(1));
                    }
                    self.phases[id] = Phase::Decoding { emitted: 1 };
                    let needs_decode = match handoff {
                        None => request.output_tokens > 1,
                        Some(handoff) => match handoff.first_token {
                            FirstTokenSource::DecodeInstance => true,
                            FirstTokenSource::PrefillInstance => request.output_tokens > 1,
                        },
                    };
                    if !needs_decode {
                        finished.push(id);
                    } else if handoff.is_some() {
                        handed_off.push(id);
                    }
                }
                PrefillPass::Resume { emitted } => {
                    // The recompute's last chunk samples the next token: the
                    // client's first (an instant at the step's end, like any
                    // first token) when nothing was emitted yet.
                    let first = emitted == 0;
                    self.timelines[id].tokens.push(TokenRecord {
                        step,
                        start_s: if first { finish_s } else { now_s },
                        finish_s,
                    });
                    token_requests.push(id);
                    if first {
                        first_token_sequences += u64::from(request.sequences.max(1));
                        if let Some(record) = self.timelines[id].handoff.as_mut() {
                            record.recompute_step = Some(step);
                        }
                    }
                    let emitted = emitted + 1;
                    self.phases[id] = Phase::Decoding { emitted };
                    if emitted >= request.output_tokens {
                        finished.push(id);
                    }
                }
            }
        }
        for &id in &plan.recomputes {
            if self.phases[id] != Phase::Recomputing {
                continue;
            }
            let request = self.request(id);
            self.timelines[id].tokens.push(TokenRecord {
                step,
                start_s: finish_s,
                finish_s,
            });
            token_requests.push(id);
            first_token_sequences += u64::from(request.sequences.max(1));
            if let Some(record) = self.timelines[id].handoff.as_mut() {
                record.recompute_step = Some(step);
            }
            self.phases[id] = Phase::Decoding { emitted: 1 };
            if request.output_tokens <= 1 {
                finished.push(id);
            }
        }
        for id in handed_off {
            self.hand_off(worker, id, finish_s);
        }
        for id in finished {
            self.phases[id] = Phase::Finished;
            self.timelines[id].fate = Some(RequestFate::Completed);
            self.release_all(id, finish_s);
            self.workers[worker]
                .running
                .retain(|running| *running != id);
        }

        self.steps.push(EngineStep {
            worker,
            start_s: now_s,
            finish_s,
            work: plan.work,
            latency,
            prefill_requests,
            token_requests,
            first_token_sequences,
            running_sequences,
            kv_blocks,
            preempted_requests: plan.preempted,
        });
        self.workers[worker].clock_s = finish_s;
    }

    /// Moves a request whose prefill finished at `at_s` to its decode
    /// worker's queue.
    fn hand_off(&mut self, worker: WorkerId, id: EngineRequestId, at_s: f64) {
        let Some(handoff) = self.jobs[id].handoff() else {
            return;
        };
        self.phases[id] = Phase::AwaitingDecode;
        self.workers[worker]
            .running
            .retain(|running| *running != id);
        self.ledger
            .release_sequences_at(HoldingKey::Primary(id), at_s);
        self.timelines[id].handoff = Some(HandoffRecord {
            prefill_finish_s: at_s,
            decode_queued_s: at_s,
            ..HandoffRecord::default()
        });
        let request_idx = self.request(id).request_idx;
        let jobs = self.jobs;
        let pending = &mut self.workers[handoff.decode_worker].pending;
        let position = pending.partition_point(|(time_s, queued)| {
            time_s
                .total_cmp(&at_s)
                .then_with(|| jobs[*queued].request.request_idx.cmp(&request_idx))
                != std::cmp::Ordering::Greater
        });
        pending.insert(position, (at_s, id));
    }

    /// Nothing could run: the worker is idle (its next arrival wakes it), its
    /// running requests are all waiting for KV pulls, or its waiting queue is
    /// blocked on capacity. A blocked worker sleeps until the next arrival,
    /// release, pull completion, waiting deadline, or step elsewhere; with
    /// none left, its waiting requests can never be admitted.
    fn wait_when_blocked(&mut self, worker: WorkerId, now_s: f64) {
        let next_pull_s = self.next_pull_ready_s(worker);
        if self.workers[worker].waiting.is_empty() {
            if let Some(ready_s) = next_pull_s {
                self.workers[worker].clock_s = ready_s.max(now_s + BLOCKED_WAKE_NUDGE_S);
            }
            return;
        }
        let next_arrival_s = self.workers[worker]
            .pending
            .front()
            .map(|(time_s, _)| *time_s);
        let next_deadline_s = self.next_waiting_deadline_s(worker);
        // Another worker's next real event (a step, an arrival, a pull
        // completion, a waiting deadline) may free capacity or schedule a
        // release. Another blocked worker's clock is not an event, so two
        // workers blocked on each other end in starvation, not a livelock.
        let other_worker_s = (0..self.workers.len())
            .filter(|other| *other != worker)
            .filter_map(|other| self.progress_event_s(other))
            .map(|time_s| time_s.max(now_s + BLOCKED_WAKE_NUDGE_S))
            .min_by(f64::total_cmp);
        let wake_s = [
            next_arrival_s,
            self.ledger.next_release_after(now_s),
            next_deadline_s,
            other_worker_s,
            next_pull_s,
        ]
        .into_iter()
        .flatten()
        .filter(|time_s| *time_s > now_s)
        .min_by(f64::total_cmp);
        match wake_s {
            Some(wake_s) => self.workers[worker].clock_s = wake_s,
            None => {
                for id in std::mem::take(&mut self.workers[worker].waiting) {
                    self.finish_without_running(id, RequestFate::Starved { at_s: now_s }, now_s);
                }
            }
        }
    }

    fn finish_without_running(&mut self, id: EngineRequestId, fate: RequestFate, now_s: f64) {
        self.phases[id] = Phase::Finished;
        self.timelines[id].fate = Some(fate);
        self.release_all(id, now_s);
    }

    /// Releases whatever the request still holds on either worker.
    fn release_all(&mut self, id: EngineRequestId, at_s: f64) {
        self.ledger.release_at(HoldingKey::Primary(id), at_s);
        if self.jobs[id].handoff().is_some() {
            self.ledger.release_at(HoldingKey::Decode(id), at_s);
        }
    }
}

/// One scheduled unit of a running request in the step being planned.
#[derive(Copy, Clone, Debug)]
enum Item {
    Decode {
        emitted: u32,
    },
    Chunk {
        computed: u32,
        chunk: u32,
        target: u32,
    },
    Recompute,
}

impl Item {
    /// KV tokens per sequence the request holds once this item has run.
    fn kv_tokens_after(self, request: &EngineRequest) -> u64 {
        match self {
            Self::Decode { emitted } => u64::from(request.prompt_tokens()) + u64::from(emitted),
            Self::Chunk {
                computed, chunk, ..
            } => u64::from(request.cached_prompt_tokens) + u64::from(computed) + u64::from(chunk),
            Self::Recompute => u64::from(request.prompt_tokens()),
        }
    }
}

/// Waiting-queue order: higher priority first, then arrival, then index.
fn queue_order(left: &EngineRequest, right: &EngineRequest) -> std::cmp::Ordering {
    right
        .priority
        .cmp(&left.priority)
        .then_with(|| left.arrival_s.total_cmp(&right.arrival_s))
        .then_with(|| left.request_idx.cmp(&right.request_idx))
}
