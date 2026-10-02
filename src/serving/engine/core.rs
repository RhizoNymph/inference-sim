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
//! Workers interact only through shared capacity limits, handoffs, and link
//! queues, so the loop always advances the worker with the earliest clock;
//! transfers are therefore reserved in non-decreasing time order.

use std::collections::VecDeque;

use crate::solver::StepWork;

use super::capacity::{CapacityLedger, HoldingKey};
use super::transfer::LinkQueues;
use super::types::{
    ChunkRecord, EngineJob, EngineLimits, EngineOutcome, EngineRequest, EngineRequestId,
    EngineStep, FirstTokenSource, HandoffRecord, RequestFate, RequestTimeline, StepCost,
    StepLimits, TokenRecord, UniformCost, WorkerId, WorkerStepCost,
};

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

#[derive(Copy, Clone, Debug, PartialEq)]
enum Phase {
    NotArrived,
    /// Waiting for prefill admission (prefill or colocated worker).
    Waiting,
    Prefilling {
        computed: u32,
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
    Finished,
}

#[derive(Clone, Debug, Default)]
struct WorkerQueue {
    clock_s: f64,
    /// Future arrivals (fresh requests and handoffs), by (time, request index).
    pending: VecDeque<(f64, EngineRequestId)>,
    waiting: Vec<EngineRequestId>,
    running: Vec<EngineRequestId>,
}

#[derive(Default)]
struct StepPlan {
    work: StepWork,
    decodes: Vec<EngineRequestId>,
    chunks: Vec<(EngineRequestId, u32)>,
    recomputes: Vec<EngineRequestId>,
}

impl StepPlan {
    fn is_empty(&self) -> bool {
        self.decodes.is_empty() && self.chunks.is_empty() && self.recomputes.is_empty()
    }
}

/// Simulates every colocated request to completion (or rejection) and
/// returns the step-by-step record.
pub(in crate::serving) fn run_engine(
    requests: &[EngineRequest],
    limits: &EngineLimits,
    cost: &mut impl StepCost,
) -> Result<EngineOutcome, EngineError> {
    let jobs = requests
        .iter()
        .cloned()
        .map(EngineJob::colocated)
        .collect::<Vec<_>>();
    run_engine_jobs(&jobs, limits, &mut UniformCost(cost))
}

/// Simulates colocated and disaggregated jobs together.
pub(in crate::serving) fn run_engine_jobs(
    jobs: &[EngineJob],
    limits: &EngineLimits,
    cost: &mut impl WorkerStepCost,
) -> Result<EngineOutcome, EngineError> {
    let mut engine = Engine::new(jobs, limits)?;
    engine.run(cost);
    Ok(EngineOutcome {
        steps: engine.steps,
        timelines: engine.timelines,
    })
}

struct Engine<'a> {
    jobs: &'a [EngineJob],
    limits: &'a EngineLimits,
    workers: Vec<WorkerQueue>,
    phases: Vec<Phase>,
    timelines: Vec<RequestTimeline>,
    steps: Vec<EngineStep>,
    ledger: CapacityLedger,
    links: LinkQueues,
}

impl<'a> Engine<'a> {
    fn new(jobs: &'a [EngineJob], limits: &'a EngineLimits) -> Result<Self, EngineError> {
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
            let waiting = &mut self.workers[worker].waiting;
            let position = waiting.partition_point(|queued| {
                queue_order(&jobs[*queued].request, &jobs[id].request)
                    != std::cmp::Ordering::Greater
            });
            waiting.insert(position, id);
        }
    }

    /// The first limit either of the request's footprints exceeds on its own.
    fn never_fits(&self, id: EngineRequestId) -> Option<super::types::CapacityExcess> {
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

        for &id in &self.workers[worker].running {
            let request = self.request(id);
            match self.phases[id] {
                Phase::Decoding { emitted } => {
                    let tokens = u64::from(request.sequences.max(1));
                    if tokens > budget.tokens && !plan.is_empty() {
                        continue;
                    }
                    budget.tokens = budget.tokens.saturating_sub(tokens);
                    plan.work
                        .add_decode(request.sequences.max(1), request.prompt_tokens() + emitted);
                    plan.decodes.push(id);
                }
                Phase::Prefilling { computed } => {
                    let chunk = budget.chunk(request, computed, plan.is_empty());
                    if chunk == 0 {
                        continue;
                    }
                    budget.consume_prefill(request, chunk);
                    add_chunk(&mut plan.work, request, computed, chunk);
                    plan.chunks.push((id, chunk));
                }
                Phase::Recomputing => {
                    let tokens = u64::from(request.sequences.max(1));
                    if tokens > budget.tokens && !plan.is_empty() {
                        continue;
                    }
                    budget.consume_prefill(request, 1);
                    plan.work.add_completing_prefill_chunk(
                        request.sequences.max(1),
                        request.prompt_tokens().saturating_sub(1),
                        1,
                    );
                    plan.recomputes.push(id);
                }
                Phase::NotArrived
                | Phase::Waiting
                | Phase::AwaitingDecode
                | Phase::Pulling { .. }
                | Phase::Finished => {}
            }
        }

        while let Some(&id) = self.workers[worker].waiting.first() {
            if budget.tokens == 0 {
                break;
            }
            if self.phases[id] == Phase::AwaitingDecode {
                if !self.admit_for_pull(worker, id, now_s) {
                    break;
                }
                continue;
            }
            let request = self.request(id);
            if !self.ledger.fits(request) {
                break;
            }
            let chunk = budget.chunk(request, 0, plan.is_empty());
            if chunk == 0 {
                break;
            }
            self.workers[worker].waiting.remove(0);
            self.ledger.allocate(id, request);
            self.workers[worker].running.push(id);
            self.phases[id] = Phase::Prefilling { computed: 0 };
            self.timelines[id].admitted_s = Some(now_s);
            budget.consume_prefill(request, chunk);
            add_chunk(&mut plan.work, request, 0, chunk);
            plan.chunks.push((id, chunk));
        }

        (!plan.is_empty()).then_some(plan)
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
        if !self
            .ledger
            .fits_on(worker, job.request.class, handoff.decode_footprint)
        {
            return false;
        }
        self.workers[worker].waiting.remove(0);
        self.ledger.allocate_on(
            HoldingKey::Decode(id),
            worker,
            job.request.class,
            handoff.decode_footprint,
        );
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
            let Phase::Prefilling { computed } = self.phases[id] else {
                continue;
            };
            let request = self.request(id);
            let computed = computed + chunk;
            self.timelines[id].chunks.push(ChunkRecord {
                step,
                start_s: now_s,
                finish_s,
                tokens: u64::from(request.sequences.max(1)) * u64::from(chunk),
            });
            prefill_requests.push(id);
            if computed < request.prefill_tokens {
                self.phases[id] = Phase::Prefilling { computed };
                continue;
            }
            let handoff = self.jobs[id].handoff();
            let client_token = handoff
                .is_none_or(|handoff| handoff.first_token == FirstTokenSource::PrefillInstance);
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

/// Waiting-queue order: higher priority first, then arrival, then index.
fn queue_order(left: &EngineRequest, right: &EngineRequest) -> std::cmp::Ordering {
    right
        .priority
        .cmp(&left.priority)
        .then_with(|| left.arrival_s.total_cmp(&right.arrival_s))
        .then_with(|| left.request_idx.cmp(&right.request_idx))
}

/// Remaining token budget of one step being planned.
struct StepBudget {
    tokens: u64,
    chunk_tokens: Option<u32>,
    prefill_tokens: Option<u64>,
    class_prefill_tokens: Vec<Option<u64>>,
}

impl StepBudget {
    fn new(step: StepLimits, limits: &EngineLimits) -> Self {
        Self {
            tokens: step.token_budget.max(1),
            chunk_tokens: step.chunk_tokens,
            prefill_tokens: step.prefill_tokens,
            class_prefill_tokens: limits
                .classes
                .iter()
                .map(|class| class.prefill_tokens_per_step)
                .collect(),
        }
    }

    /// Prefill tokens (per lockstep sequence) the request may compute now.
    /// The first item of an otherwise empty step always makes progress.
    fn chunk(&self, request: &EngineRequest, computed: u32, first_in_step: bool) -> u32 {
        let remaining = request.prefill_tokens.saturating_sub(computed);
        if remaining == 0 {
            return 0;
        }
        let sequences = u64::from(request.sequences.max(1));
        let mut room = self.tokens;
        if let Some(left) = self.prefill_tokens {
            room = room.min(left);
        }
        if let Some(class) = request.class
            && let Some(Some(left)) = self.class_prefill_tokens.get(class)
        {
            room = room.min(*left);
        }
        let mut chunk = u64::from(remaining).min(room / sequences);
        if let Some(cap) = self.chunk_tokens {
            chunk = chunk.min(u64::from(cap.max(1)));
        }
        if chunk == 0 && first_in_step {
            chunk = 1;
        }
        chunk.min(u64::from(u32::MAX)) as u32
    }

    fn consume_prefill(&mut self, request: &EngineRequest, chunk: u32) {
        let tokens = u64::from(request.sequences.max(1)) * u64::from(chunk);
        self.tokens = self.tokens.saturating_sub(tokens);
        if let Some(left) = self.prefill_tokens.as_mut() {
            *left = left.saturating_sub(tokens);
        }
        if let Some(class) = request.class
            && let Some(Some(left)) = self.class_prefill_tokens.get_mut(class)
        {
            *left = left.saturating_sub(tokens);
        }
    }
}

// A chunk that reaches the end of the prompt samples each sequence's first
// token in the same step, so it carries LM-head logits work.
fn add_chunk(work: &mut StepWork, request: &EngineRequest, computed: u32, chunk: u32) {
    let sequences = request.sequences.max(1);
    let context_before = request.cached_prompt_tokens + computed;
    if computed.saturating_add(chunk) >= request.prefill_tokens {
        work.add_completing_prefill_chunk(sequences, context_before, chunk);
    } else {
        work.add_prefill_chunk(sequences, context_before, chunk);
    }
}
