//! KV holdings under each `KvAllocationMode`, and vLLM V1 recompute preemption
//! under `KvAllocationMode::PagedRecompute`.

use std::num::NonZeroU32;

use super::super::capacity::HoldingKey;
use super::super::types::{
    CapacityExcess, EngineRequest, EngineRequestId, KvAllocationMode, KvFootprint,
    PreemptionRecord, WorkerId,
};
use super::{Engine, Phase, PrefillPass, StepPlan};

/// Whether a running request got the KV its scheduled item needs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum KvGrant {
    Granted,
    /// No victim was left but the requester itself, which is now preempted;
    /// the step schedules no further running requests.
    SelfPreempted,
}

impl Engine<'_> {
    /// Which of the request's holdings lives on `worker`.
    pub(super) fn holding_key(&self, id: EngineRequestId, worker: WorkerId) -> HoldingKey {
        match self.jobs[id].handoff() {
            Some(handoff) if handoff.decode_worker == worker => HoldingKey::Decode(id),
            _ => HoldingKey::Primary(id),
        }
    }

    /// What admitting a request with a first chunk of `chunk` tokens per
    /// sequence allocates.
    pub(super) fn admission_footprint(&self, request: &EngineRequest, chunk: u32) -> KvFootprint {
        match self.kv {
            KvAllocationMode::ReservedAtAdmission => request.footprint,
            KvAllocationMode::PagedRecompute { block_tokens } => KvAllocationMode::paged_footprint(
                block_tokens,
                request.sequences,
                u64::from(request.cached_prompt_tokens) + u64::from(chunk),
            ),
        }
    }

    /// What a disaggregated request's decode admission (before its pull)
    /// allocates: the reserved decode footprint, or under paged allocation
    /// the blocks of the prompt KV it is about to receive.
    pub(super) fn decode_admission_footprint(
        &self,
        request: &EngineRequest,
        reserved: KvFootprint,
    ) -> KvFootprint {
        match self.kv {
            KvAllocationMode::ReservedAtAdmission => reserved,
            KvAllocationMode::PagedRecompute { block_tokens } => KvAllocationMode::paged_footprint(
                block_tokens,
                request.sequences,
                u64::from(request.prompt_tokens()),
            ),
        }
    }

    /// Under paged allocation a request can never run when its peak holding
    /// (block-rounded) exceeds a limit on its own: on the prefill worker the
    /// prompt, on the worker that decodes the prompt plus every output token
    /// but the last.
    pub(super) fn never_fits_paged(
        &self,
        id: EngineRequestId,
        block_tokens: NonZeroU32,
    ) -> Option<CapacityExcess> {
        let request = self.request(id);
        let peak_decode = KvAllocationMode::paged_footprint(
            block_tokens,
            request.sequences,
            u64::from(request.prompt_tokens()) + u64::from(request.output_tokens.max(1)) - 1,
        );
        match self.jobs[id].handoff() {
            None => self
                .ledger
                .never_fits_on(request.worker, request.class, peak_decode),
            Some(handoff) => {
                let prompt = KvAllocationMode::paged_footprint(
                    block_tokens,
                    request.sequences,
                    u64::from(request.prompt_tokens()),
                );
                self.ledger
                    .never_fits_on(request.worker, request.class, prompt)
                    .or_else(|| {
                        self.ledger
                            .never_fits_on(handoff.decode_worker, request.class, peak_decode)
                    })
            }
        }
    }

    /// Grows the holding of the running request at `index` to cover
    /// `kv_tokens` per sequence, preempting running requests not yet
    /// scheduled in this step (lowest priority, then most recently admitted)
    /// until it fits. Reserved allocation already holds everything.
    pub(super) fn grant_kv(
        &mut self,
        worker: WorkerId,
        index: usize,
        id: EngineRequestId,
        kv_tokens: u64,
        now_s: f64,
        plan: &mut StepPlan,
    ) -> KvGrant {
        let KvAllocationMode::PagedRecompute { block_tokens } = self.kv else {
            return KvGrant::Granted;
        };
        let key = self.holding_key(id, worker);
        let wanted =
            KvAllocationMode::paged_footprint(block_tokens, self.request(id).sequences, kv_tokens);
        loop {
            let held = self.ledger.held(key).unwrap_or_default();
            let extra = KvFootprint {
                sequences: 0,
                tokens: wanted.tokens.saturating_sub(held.tokens),
                blocks: wanted.blocks.saturating_sub(held.blocks),
            };
            if extra.blocks == 0 && extra.tokens == 0 {
                return KvGrant::Granted;
            }
            if self.ledger.try_grow(key, extra) {
                return KvGrant::Granted;
            }
            let victim = self.preemption_victim(worker, index).unwrap_or(id);
            self.preempt(worker, victim, now_s, plan);
            if victim == id {
                return KvGrant::SelfPreempted;
            }
        }
    }

    /// The running request at or after `index` that vLLM would preempt:
    /// the lowest priority, then the latest in the running list (vLLM's FCFS
    /// `running.pop()`). Requests still pulling their KV are not running in
    /// vLLM and are never chosen.
    fn preemption_victim(&self, worker: WorkerId, index: usize) -> Option<EngineRequestId> {
        self.workers[worker]
            .running
            .iter()
            .enumerate()
            .skip(index)
            .filter(|(_, id)| !matches!(self.phases[**id], Phase::Pulling { .. }))
            .min_by(|(left_pos, left), (right_pos, right)| {
                self.request(**left)
                    .priority
                    .cmp(&self.request(**right).priority)
                    .then_with(|| right_pos.cmp(left_pos))
            })
            .map(|(_, id)| *id)
    }

    /// Frees every block `id` holds on `worker` and queues it at the front of
    /// the worker's waiting queue to recompute its context.
    fn preempt(&mut self, worker: WorkerId, id: EngineRequestId, now_s: f64, plan: &mut StepPlan) {
        let request = self.request(id);
        let (pass, emitted, discarded_per_sequence) = match self.phases[id] {
            Phase::Prefilling {
                computed,
                pass: PrefillPass::Prompt,
            } => (PrefillPass::Prompt, 0, computed),
            Phase::Prefilling {
                computed,
                pass: PrefillPass::Resume { emitted },
            } => (PrefillPass::Resume { emitted }, emitted, computed),
            // Token `emitted` has been sampled but its KV is not computed yet.
            Phase::Decoding { emitted } => (
                PrefillPass::Resume { emitted },
                emitted,
                request
                    .prefill_tokens
                    .saturating_add(emitted)
                    .saturating_sub(1),
            ),
            // The pulled prompt KV, without the last token it was about to
            // recompute.
            Phase::Recomputing => (
                PrefillPass::Resume { emitted: 0 },
                0,
                request.prefill_tokens.saturating_sub(1),
            ),
            Phase::NotArrived
            | Phase::Waiting
            | Phase::AwaitingDecode
            | Phase::Pulling { .. }
            | Phase::Preempted { .. }
            | Phase::Finished => return,
        };
        let key = self.holding_key(id, worker);
        self.ledger.release_now_all(key);
        self.workers[worker]
            .running
            .retain(|running| *running != id);
        self.workers[worker].waiting.insert(0, id);
        self.phases[id] = Phase::Preempted { pass };
        self.timelines[id].preemptions.push(PreemptionRecord {
            worker,
            step: self.steps.len(),
            at_s: now_s,
            emitted_tokens: emitted,
            discarded_tokens: u64::from(request.sequences.max(1))
                * u64::from(discarded_per_sequence),
            resumed_s: None,
        });
        plan.preempted.push(id);
    }
}
