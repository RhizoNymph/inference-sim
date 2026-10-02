//! Value types of the iteration-level serving engine: what a request asks of
//! the engine, the limits it runs under, and what the loop records.

use crate::solver::{StepLatency, StepWork};

use super::transfer::{KvTransferPlan, TransferWindow};

/// Index of one engine worker (one replica on one routed GPU set).
pub(in crate::serving) type WorkerId = usize;
/// Index of a request inside one engine run.
pub(in crate::serving) type EngineRequestId = usize;
/// Index of a traffic class inside one engine run's limits.
pub(in crate::serving) type ClassId = usize;

/// KV cache a request holds from admission to completion: every lockstep
/// sequence reserves blocks for its full `max_sequence_tokens` up front.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::serving) struct KvFootprint {
    pub(in crate::serving) sequences: u64,
    pub(in crate::serving) tokens: u64,
    pub(in crate::serving) blocks: u64,
}

/// One request as the engine sees it. Built once from routing and traffic.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct EngineRequest {
    pub(in crate::serving) worker: WorkerId,
    pub(in crate::serving) arrival_s: f64,
    pub(in crate::serving) priority: i32,
    pub(in crate::serving) request_idx: u32,
    /// Lockstep sequences (`batch_size`), each with the same shape.
    pub(in crate::serving) sequences: u32,
    /// Prompt tokens already in the KV cache (prefix-cache hits).
    pub(in crate::serving) cached_prompt_tokens: u32,
    /// Prompt tokens the engine must compute; at least 1, because the step
    /// that computes the last prompt token samples the first output token.
    pub(in crate::serving) prefill_tokens: u32,
    /// Output tokens to emit, at least 1. The first comes from the prefill.
    pub(in crate::serving) output_tokens: u32,
    pub(in crate::serving) footprint: KvFootprint,
    pub(in crate::serving) class: Option<ClassId>,
    pub(in crate::serving) cancellation_s: Option<f64>,
    pub(in crate::serving) max_queue_delay_s: Option<f64>,
}

impl EngineRequest {
    pub(in crate::serving) fn prompt_tokens(&self) -> u32 {
        self.cached_prompt_tokens
            .saturating_add(self.prefill_tokens)
    }
}

/// Which instance's token a disaggregated request's client sees first.
///
/// vLLM's disaggregated proxy (the NIXL toy proxy) sends the request to the
/// prefill instance with `max_tokens = 1`, waits for that response, then
/// streams the decode instance's output to the client. The decode instance
/// recomputes the last prompt token (vLLM marks `prompt - 1` tokens as
/// loaded from the remote KV) and samples the client's first token in that
/// step, so `DecodeInstance` is the vLLM convention.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::serving) enum FirstTokenSource {
    /// The client's first token comes from the decode worker's step that
    /// recomputes the last prompt token after the KV pull; the prefill
    /// worker's token is internal. All `output_tokens` come from decode.
    #[default]
    DecodeInstance,
    /// The prefill worker's token is the client's first token (a proxy that
    /// forwards it immediately); decode continues from token 2 without
    /// recomputation.
    PrefillInstance,
}

/// A disaggregated request's move from its prefill worker (the
/// `EngineRequest::worker`) to its decode worker.
///
/// The transfer is decode-initiated (vLLM NixlConnector): when the prefill
/// finishes, the request joins the decode worker's waiting queue; admission
/// there reserves its decode KV blocks and starts the pull; the request
/// computes on decode only after the pull completes. The prefill worker holds
/// the prompt KV (tokens and blocks, not a running sequence) until the pull
/// completes.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct Handoff {
    pub(in crate::serving) decode_worker: WorkerId,
    /// KV the request holds on the decode worker from admission to completion.
    pub(in crate::serving) decode_footprint: KvFootprint,
    pub(in crate::serving) plan: KvTransferPlan,
    pub(in crate::serving) first_token: FirstTokenSource,
}

/// Whether a request runs on one worker or moves between two.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) enum JobRoute {
    Colocated,
    Disaggregated(Handoff),
}

/// One request and its route through the engine.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct EngineJob {
    pub(in crate::serving) request: EngineRequest,
    pub(in crate::serving) route: JobRoute,
}

impl EngineJob {
    pub(in crate::serving) fn colocated(request: EngineRequest) -> Self {
        Self {
            request,
            route: JobRoute::Colocated,
        }
    }

    pub(in crate::serving) fn handoff(&self) -> Option<&Handoff> {
        match &self.route {
            JobRoute::Colocated => None,
            JobRoute::Disaggregated(handoff) => Some(handoff),
        }
    }
}

/// What happened to a disaggregated request between its workers.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::serving) struct HandoffRecord {
    /// End of the prefill worker's step that computed the last prompt token
    /// (and sampled the prefill instance's token).
    pub(in crate::serving) prefill_finish_s: f64,
    /// When the request joined the decode worker's waiting queue.
    pub(in crate::serving) decode_queued_s: f64,
    /// When the decode worker reserved its KV and started the pull.
    pub(in crate::serving) decode_admitted_s: Option<f64>,
    pub(in crate::serving) transfer: Option<TransferWindow>,
    /// Decode step that recomputed the last prompt token (`DecodeInstance`).
    pub(in crate::serving) recompute_step: Option<usize>,
}

/// Sequence, token and KV limits on one scope (all workers, one worker, or
/// one traffic class). `None` means unlimited.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::serving) struct CapacityLimits {
    pub(in crate::serving) sequences: Option<u64>,
    pub(in crate::serving) tokens: Option<u64>,
    pub(in crate::serving) blocks: Option<u64>,
}

/// Per-step scheduling knobs of one worker, vLLM V1 style.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::serving) struct StepLimits {
    /// Tokens one step may carry (`max_num_batched_tokens`), decode tokens included.
    pub(in crate::serving) token_budget: u64,
    /// Largest prefill chunk one request may take in one step.
    pub(in crate::serving) chunk_tokens: Option<u32>,
    /// Prefill tokens one step may carry across all its requests.
    pub(in crate::serving) prefill_tokens: Option<u64>,
}

/// Limits a traffic class adds on top of the global and worker limits.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct ClassLimits {
    pub(in crate::serving) name: String,
    pub(in crate::serving) capacity: CapacityLimits,
    pub(in crate::serving) prefill_tokens_per_step: Option<u64>,
}

/// Everything the loop needs besides the requests and the step cost.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct EngineLimits {
    pub(in crate::serving) global: CapacityLimits,
    pub(in crate::serving) workers: Vec<WorkerLimits>,
    pub(in crate::serving) classes: Vec<ClassLimits>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::serving) struct WorkerLimits {
    pub(in crate::serving) capacity: CapacityLimits,
    pub(in crate::serving) step: StepLimits,
}

/// Which limit a request exceeds on its own, so it can never be admitted.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::serving) enum CapacityScope {
    Global,
    Worker,
    Class(ClassId),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::serving) enum CapacityResource {
    Sequences,
    Tokens,
    Blocks,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::serving) struct CapacityExcess {
    pub(in crate::serving) scope: CapacityScope,
    pub(in crate::serving) resource: CapacityResource,
    pub(in crate::serving) needed: u64,
    pub(in crate::serving) limit: u64,
}

/// How a request left the engine.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) enum RequestFate {
    /// Emitted every output token.
    Completed,
    /// Cancelled at `at_s` (while waiting or running).
    Cancelled { at_s: f64 },
    /// Waited in the queue longer than its `max_queue_delay_s`.
    QueueTimeout { waited_s: f64, limit_s: f64 },
    /// Its own footprint exceeds a limit, so no amount of waiting admits it.
    NeverFits(CapacityExcess),
    /// Still waiting at `at_s` when nothing could ever free capacity for it.
    Starved { at_s: f64 },
}

/// Prefill tokens computed for a request in one step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(in crate::serving) struct ChunkRecord {
    pub(in crate::serving) step: usize,
    pub(in crate::serving) start_s: f64,
    pub(in crate::serving) finish_s: f64,
    /// Tokens across the request's lockstep sequences.
    pub(in crate::serving) tokens: u64,
}

/// One emitted output token. The first token is sampled by the step that
/// finishes the prompt, so it is recorded as an instant at that step's end.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(in crate::serving) struct TokenRecord {
    pub(in crate::serving) step: usize,
    pub(in crate::serving) start_s: f64,
    pub(in crate::serving) finish_s: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::serving) struct RequestTimeline {
    pub(in crate::serving) admitted_s: Option<f64>,
    pub(in crate::serving) chunks: Vec<ChunkRecord>,
    pub(in crate::serving) tokens: Vec<TokenRecord>,
    pub(in crate::serving) fate: Option<RequestFate>,
    /// Set for disaggregated requests whose prefill completed.
    pub(in crate::serving) handoff: Option<HandoffRecord>,
}

/// One forward pass of one worker.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct EngineStep {
    pub(in crate::serving) worker: WorkerId,
    pub(in crate::serving) start_s: f64,
    pub(in crate::serving) finish_s: f64,
    pub(in crate::serving) work: StepWork,
    pub(in crate::serving) latency: StepLatency,
    /// Requests that computed prefill tokens in this step.
    pub(in crate::serving) prefill_requests: Vec<EngineRequestId>,
    /// Requests that emitted a token in this step (decodes and first tokens).
    pub(in crate::serving) token_requests: Vec<EngineRequestId>,
    /// Lockstep sequences whose first token this step sampled.
    pub(in crate::serving) first_token_sequences: u64,
    /// Running sequences (prefilling or decoding) after admission this step.
    pub(in crate::serving) running_sequences: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::serving) struct EngineOutcome {
    pub(in crate::serving) steps: Vec<EngineStep>,
    pub(in crate::serving) timelines: Vec<RequestTimeline>,
}

/// Per-step latency source; the production implementation is the solver's
/// `IterationCostModel`, tests substitute closed-form costs.
pub(in crate::serving) trait StepCost {
    fn step_latency(&mut self, work: &StepWork) -> StepLatency;
}

/// Per-worker step latency: disaggregated runs price prefill workers and
/// decode workers with different parallelism configs.
pub(in crate::serving) trait WorkerStepCost {
    fn worker_step_latency(&mut self, worker: WorkerId, work: &StepWork) -> StepLatency;
}

/// Prices every worker with one cost model.
pub(in crate::serving) struct UniformCost<'c, C>(pub(in crate::serving) &'c mut C);

impl<C: StepCost> WorkerStepCost for UniformCost<'_, C> {
    fn worker_step_latency(&mut self, _worker: WorkerId, work: &StepWork) -> StepLatency {
        self.0.step_latency(work)
    }
}

impl StepCost for crate::solver::IterationCostModel<'_> {
    fn step_latency(&mut self, work: &StepWork) -> StepLatency {
        crate::solver::IterationCostModel::step_latency(self, work)
    }
}
