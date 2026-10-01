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
//! Workers interact only through shared capacity limits, so the loop always
//! advances the worker with the earliest clock.

use std::collections::VecDeque;

use crate::solver::StepWork;

use super::capacity::CapacityLedger;
use super::types::{
    ChunkRecord, EngineLimits, EngineOutcome, EngineRequest, EngineRequestId, EngineStep,
    RequestFate, RequestTimeline, StepCost, StepLimits, TokenRecord, WorkerId,
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
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownWorker { request, worker } => write!(
                formatter,
                "engine request {request} is routed to unknown worker {worker}"
            ),
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Phase {
    NotArrived,
    Waiting,
    Prefilling { computed: u32 },
    Decoding { emitted: u32 },
    Finished,
}

#[derive(Clone, Debug, Default)]
struct WorkerQueue {
    clock_s: f64,
    pending: VecDeque<EngineRequestId>,
    waiting: Vec<EngineRequestId>,
    running: Vec<EngineRequestId>,
}

impl WorkerQueue {
    fn next_event_s(&self, requests: &[EngineRequest]) -> Option<f64> {
        if !self.running.is_empty() || !self.waiting.is_empty() {
            return Some(self.clock_s);
        }
        self.pending
            .front()
            .map(|id| requests[*id].arrival_s.max(self.clock_s))
    }

    fn has_work(&self) -> bool {
        !self.running.is_empty() || !self.waiting.is_empty()
    }
}

#[derive(Default)]
struct StepPlan {
    work: StepWork,
    decodes: Vec<EngineRequestId>,
    chunks: Vec<(EngineRequestId, u32)>,
}

impl StepPlan {
    fn is_empty(&self) -> bool {
        self.decodes.is_empty() && self.chunks.is_empty()
    }
}

/// Simulates every request to completion (or rejection) and returns the
/// step-by-step record.
pub(in crate::serving) fn run_engine(
    requests: &[EngineRequest],
    limits: &EngineLimits,
    cost: &mut impl StepCost,
) -> Result<EngineOutcome, EngineError> {
    let mut engine = Engine::new(requests, limits)?;
    engine.run(cost);
    Ok(EngineOutcome {
        steps: engine.steps,
        timelines: engine.timelines,
    })
}

struct Engine<'a> {
    requests: &'a [EngineRequest],
    limits: &'a EngineLimits,
    workers: Vec<WorkerQueue>,
    phases: Vec<Phase>,
    timelines: Vec<RequestTimeline>,
    steps: Vec<EngineStep>,
    ledger: CapacityLedger,
}

impl<'a> Engine<'a> {
    fn new(requests: &'a [EngineRequest], limits: &'a EngineLimits) -> Result<Self, EngineError> {
        let mut workers = vec![WorkerQueue::default(); limits.workers.len()];
        let mut order = (0..requests.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            requests[*left]
                .arrival_s
                .total_cmp(&requests[*right].arrival_s)
                .then_with(|| {
                    requests[*left]
                        .request_idx
                        .cmp(&requests[*right].request_idx)
                })
        });
        for id in order {
            let worker = requests[id].worker;
            let Some(queue) = workers.get_mut(worker) else {
                return Err(EngineError::UnknownWorker {
                    request: id,
                    worker,
                });
            };
            queue.pending.push_back(id);
        }
        Ok(Self {
            requests,
            limits,
            workers,
            phases: vec![Phase::NotArrived; requests.len()],
            timelines: vec![RequestTimeline::default(); requests.len()],
            steps: Vec::new(),
            ledger: CapacityLedger::new(limits, requests.len()),
        })
    }

    fn run(&mut self, cost: &mut impl StepCost) {
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

    fn next_worker(&self) -> Option<(WorkerId, f64)> {
        self.workers
            .iter()
            .enumerate()
            .filter_map(|(worker, queue)| {
                queue
                    .next_event_s(self.requests)
                    .map(|time_s| (worker, time_s))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)))
    }

    fn enqueue_arrivals(&mut self, worker: WorkerId, now_s: f64) {
        while let Some(&id) = self.workers[worker].pending.front() {
            if self.requests[id].arrival_s > now_s + TIME_EPSILON_S {
                break;
            }
            self.workers[worker].pending.pop_front();
            if let Some(excess) = self.ledger.never_fits(&self.requests[id]) {
                self.finish_without_running(id, RequestFate::NeverFits(excess));
                continue;
            }
            self.phases[id] = Phase::Waiting;
            let requests = self.requests;
            let waiting = &mut self.workers[worker].waiting;
            let position = waiting.partition_point(|queued| {
                queue_order(&requests[*queued], &requests[id]) != std::cmp::Ordering::Greater
            });
            waiting.insert(position, id);
        }
    }

    fn expire(&mut self, worker: WorkerId, now_s: f64) {
        let waiting = std::mem::take(&mut self.workers[worker].waiting);
        let mut kept = Vec::with_capacity(waiting.len());
        for id in waiting {
            let request = &self.requests[id];
            if let Some(cancellation_s) = request.cancellation_s
                && cancellation_s <= now_s + TIME_EPSILON_S
            {
                self.finish_without_running(
                    id,
                    RequestFate::Cancelled {
                        at_s: cancellation_s,
                    },
                );
            } else if let Some(limit_s) = request.max_queue_delay_s
                && now_s - request.arrival_s > limit_s + TIME_EPSILON_S
            {
                let waited_s = now_s - request.arrival_s;
                self.finish_without_running(id, RequestFate::QueueTimeout { waited_s, limit_s });
            } else {
                kept.push(id);
            }
        }
        self.workers[worker].waiting = kept;

        let running = std::mem::take(&mut self.workers[worker].running);
        let mut kept = Vec::with_capacity(running.len());
        for id in running {
            match self.requests[id].cancellation_s {
                Some(cancellation_s) if cancellation_s <= now_s + TIME_EPSILON_S => {
                    self.phases[id] = Phase::Finished;
                    self.timelines[id].fate = Some(RequestFate::Cancelled {
                        at_s: cancellation_s,
                    });
                    self.ledger.release_at(id, now_s);
                }
                _ => kept.push(id),
            }
        }
        self.workers[worker].running = kept;
        self.ledger.apply_releases_until(now_s);
    }

    fn plan_step(&mut self, worker: WorkerId, now_s: f64) -> Option<StepPlan> {
        let step_limits = self.limits.workers[worker].step;
        let mut budget = StepBudget::new(step_limits, self.limits);
        let mut plan = StepPlan::default();

        for &id in &self.workers[worker].running {
            let request = &self.requests[id];
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
                Phase::NotArrived | Phase::Waiting | Phase::Finished => {}
            }
        }

        while let Some(&id) = self.workers[worker].waiting.first() {
            let request = &self.requests[id];
            if budget.tokens == 0 || !self.ledger.fits(request) {
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

    fn execute(&mut self, worker: WorkerId, now_s: f64, plan: StepPlan, cost: &mut impl StepCost) {
        let latency = cost.step_latency(&plan.work);
        let finish_s = now_s + latency.total_s.max(0.0);
        let step = self.steps.len();
        let running_sequences = self.ledger.worker_sequences(worker);
        let mut token_requests = Vec::with_capacity(plan.decodes.len() + plan.chunks.len());
        let mut first_token_sequences = 0_u64;
        let mut finished = Vec::new();

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
            if emitted >= self.requests[id].output_tokens {
                finished.push(id);
            }
        }
        let mut prefill_requests = Vec::with_capacity(plan.chunks.len());
        for &(id, chunk) in &plan.chunks {
            let Phase::Prefilling { computed } = self.phases[id] else {
                continue;
            };
            let request = &self.requests[id];
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
            self.timelines[id].tokens.push(TokenRecord {
                step,
                start_s: finish_s,
                finish_s,
            });
            token_requests.push(id);
            first_token_sequences += u64::from(request.sequences.max(1));
            self.phases[id] = Phase::Decoding { emitted: 1 };
            if request.output_tokens <= 1 {
                finished.push(id);
            }
        }
        for id in finished {
            self.phases[id] = Phase::Finished;
            self.timelines[id].fate = Some(RequestFate::Completed);
            self.ledger.release_at(id, finish_s);
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

    /// Nothing could run: either the worker is idle (its next arrival wakes
    /// it) or its waiting queue is blocked on capacity. A blocked worker sleeps
    /// until the next arrival, release, waiting deadline, or step elsewhere;
    /// with none left, its waiting requests can never be admitted.
    fn wait_when_blocked(&mut self, worker: WorkerId, now_s: f64) {
        if self.workers[worker].waiting.is_empty() {
            return;
        }
        let next_arrival_s = self.workers[worker]
            .pending
            .front()
            .map(|id| self.requests[*id].arrival_s);
        let next_deadline_s = self.workers[worker]
            .waiting
            .iter()
            .filter_map(|id| {
                let request = &self.requests[*id];
                let timeout_s = request
                    .max_queue_delay_s
                    .map(|limit_s| request.arrival_s + limit_s + BLOCKED_WAKE_NUDGE_S);
                match (request.cancellation_s, timeout_s) {
                    (Some(left), Some(right)) => Some(left.min(right)),
                    (left, right) => left.or(right),
                }
            })
            .min_by(f64::total_cmp);
        let other_worker_s = self
            .workers
            .iter()
            .enumerate()
            .filter(|(other, queue)| *other != worker && queue.has_work())
            .map(|(_, queue)| queue.clock_s.max(now_s + BLOCKED_WAKE_NUDGE_S))
            .min_by(f64::total_cmp);
        let wake_s = [
            next_arrival_s,
            self.ledger.next_release_after(now_s),
            next_deadline_s,
            other_worker_s,
        ]
        .into_iter()
        .flatten()
        .filter(|time_s| *time_s > now_s)
        .min_by(f64::total_cmp);
        match wake_s {
            Some(wake_s) => self.workers[worker].clock_s = wake_s,
            None => {
                for id in std::mem::take(&mut self.workers[worker].waiting) {
                    self.finish_without_running(id, RequestFate::Starved { at_s: now_s });
                }
            }
        }
    }

    fn finish_without_running(&mut self, id: EngineRequestId, fate: RequestFate) {
        self.phases[id] = Phase::Finished;
        self.timelines[id].fate = Some(fate);
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
