//! Token budget of one step being planned, and how a prefill chunk enters
//! the step's work.

use crate::solver::StepWork;

use super::super::types::{EngineLimits, EngineRequest, StepLimits};

/// Remaining token budget of one step being planned.
pub(super) struct StepBudget {
    pub(super) tokens: u64,
    chunk_tokens: Option<u32>,
    prefill_tokens: Option<u64>,
    class_prefill_tokens: Vec<Option<u64>>,
}

impl StepBudget {
    pub(super) fn new(step: StepLimits, limits: &EngineLimits) -> Self {
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

    /// Prefill tokens (per lockstep sequence) the request may compute now,
    /// out of `remaining` left in its current pass. The first item of an
    /// otherwise empty step always makes progress.
    pub(super) fn chunk(
        &self,
        request: &EngineRequest,
        remaining: u32,
        first_in_step: bool,
    ) -> u32 {
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

    pub(super) fn consume_prefill(&mut self, request: &EngineRequest, chunk: u32) {
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

// A chunk that reaches the end of its pass (`target` tokens beyond the
// cached prompt) samples each sequence's next token in the same step, so it
// carries LM-head logits work.
pub(super) fn add_chunk(
    work: &mut StepWork,
    request: &EngineRequest,
    computed: u32,
    chunk: u32,
    target: u32,
) {
    let sequences = request.sequences.max(1);
    let context_before = request.cached_prompt_tokens + computed;
    if computed.saturating_add(chunk) >= target {
        work.add_completing_prefill_chunk(sequences, context_before, chunk);
    } else {
        work.add_prefill_chunk(sequences, context_before, chunk);
    }
}
