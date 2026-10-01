//! Latency of one serving-engine iteration (one forward pass) priced from the
//! work actually scheduled in that step.
//!
//! A step carries prefill chunks (each a run of prompt tokens appended to a
//! sequence's existing context) and decode tokens (one per running sequence).
//! Its latency is one forward pass: `max(compute, memory)` plus per-step
//! communication and the calibrated per-step overhead, where
//!
//! - compute = dense parameter FLOPs for every token in the step, causal
//!   attention FLOPs for each prefill chunk over its context, and decode
//!   attention FLOPs over each decoding sequence's context;
//! - memory = the weights read once for the step plus the KV cache read for
//!   every active sequence's current context;
//! - communication = two tensor-parallel all-reduces per layer, the expert
//!   all-to-all per layer, and one pipeline send/recv per stage boundary, each
//!   carrying the step's token count of activations.
//!
//! The formulas are the static solver's roofline primitives applied to a
//! step's composition, so a pure-decode step over a batch at context
//! `prompt + 1` equals the solver's one-token decode latency and a
//! compute-bound pure-prefill step equals the solver's prefill latency.

use std::collections::HashMap;

use super::*;

/// Token work scheduled into one engine step, aggregated over its sequences.
///
/// Only the aggregates the roofline needs are kept, so adding a sequence is
/// O(1) and a step's cost does not depend on how many sequences it carries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepWork {
    prefill_tokens: u64,
    /// Sum over chunks of `sequences * ((context_before + tokens)^2 - context_before^2)`:
    /// the causal attention (query, key) pairs a chunk adds.
    prefill_attention_pairs: f64,
    /// Sum over chunks of `sequences * context_before`: earlier-chunk KV re-read.
    prefill_context_tokens: u64,
    decode_sequences: u64,
    /// Sum over decoding sequences of `sequences * context_tokens`.
    decode_context_tokens: u64,
    /// Sequences that sample a token this step (every decode, plus prefill
    /// chunks that complete their prompt); the LM-head logits scale with it.
    sampling_sequences: u64,
}

impl StepWork {
    /// Adds a prefill chunk of `tokens` prompt tokens for `sequences` lockstep
    /// sequences whose KV cache already holds `context_before` tokens.
    pub fn add_prefill_chunk(&mut self, sequences: u32, context_before: u32, tokens: u32) {
        if sequences == 0 || tokens == 0 {
            return;
        }
        let sequences_u64 = u64::from(sequences);
        let before = f64::from(context_before);
        let after = before + f64::from(tokens);
        self.prefill_tokens += sequences_u64 * u64::from(tokens);
        self.prefill_attention_pairs += f64::from(sequences) * (after * after - before * before);
        self.prefill_context_tokens += sequences_u64 * u64::from(context_before);
    }

    /// Adds a prefill chunk that completes its prompt, so each of its
    /// `sequences` samples its first token in this step.
    pub fn add_completing_prefill_chunk(
        &mut self,
        sequences: u32,
        context_before: u32,
        tokens: u32,
    ) {
        if sequences == 0 || tokens == 0 {
            return;
        }
        self.add_prefill_chunk(sequences, context_before, tokens);
        self.sampling_sequences += u64::from(sequences);
    }

    /// Adds one decode token for each of `sequences` lockstep sequences that
    /// attend over `context_tokens` keys (prompt plus tokens generated so far,
    /// including the token being processed).
    pub fn add_decode(&mut self, sequences: u32, context_tokens: u32) {
        if sequences == 0 {
            return;
        }
        let sequences = u64::from(sequences);
        self.decode_sequences += sequences;
        self.decode_context_tokens += sequences * u64::from(context_tokens);
        self.sampling_sequences += sequences;
    }

    pub fn prefill_tokens(&self) -> u64 {
        self.prefill_tokens
    }

    pub fn decode_sequences(&self) -> u64 {
        self.decode_sequences
    }

    pub fn sampling_sequences(&self) -> u64 {
        self.sampling_sequences
    }

    pub fn total_tokens(&self) -> u64 {
        self.prefill_tokens + self.decode_sequences
    }

    pub fn is_empty(&self) -> bool {
        self.total_tokens() == 0
    }
}

/// Breakdown of one step's latency, in seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepLatency {
    pub compute_s: f64,
    pub memory_s: f64,
    pub communication_s: f64,
    pub overhead_s: f64,
    pub total_s: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepCostError {
    /// The scored configuration has no rank placement to price against.
    EmptyPlacement,
}

impl std::fmt::Display for StepCostError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPlacement => write!(
                formatter,
                "iteration cost model needs a placed parallelism config (empty rank placement)"
            ),
        }
    }
}

impl std::error::Error for StepCostError {}

/// A collective issued `repeats` times per step whose per-rank message is
/// `bytes_per_token` times the step's token count.
#[derive(Clone, Debug, PartialEq)]
struct StepCollective {
    kind: CollectiveKind,
    participants: Vec<RankId>,
    size: StepCollectiveSize,
    reduction: Option<ReductionOp>,
    algorithm: CollectiveAlgorithm,
    repeats: u32,
}

/// What a step collective's message scales with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepCollectiveSize {
    /// Bytes per token processed in the step (activations).
    PerToken(u64),
    /// Bytes per sequence that samples a token in the step (LM-head logits).
    PerSamplingSequence(u64),
}

impl StepCollectiveSize {
    fn bytes(self, tokens: u64, sampling_sequences: u64) -> u64 {
        match self {
            Self::PerToken(bytes) => bytes * tokens,
            Self::PerSamplingSequence(bytes) => bytes * sampling_sequences,
        }
    }
}

/// Prices engine steps for one placed parallelism configuration.
pub struct IterationCostModel<'a> {
    cluster: &'a Cluster,
    placement: &'a RankPlacement,
    calibration: SimulationCalibration,
    dtype: crate::workload::DType,
    peak_flops: f64,
    hbm_bandwidth: f64,
    dense_flops_per_token: f64,
    attention_flops_per_pair: f64,
    decode_attention_flops_per_context_token: f64,
    weight_bytes: f64,
    kv_bytes_per_context_token: f64,
    collectives: Vec<StepCollective>,
    communication_cache: HashMap<(u64, u64), f64>,
}

impl<'a> IterationCostModel<'a> {
    pub fn new(
        cluster: &'a Cluster,
        model: &ModelSpec,
        score: &'a ScoredParallelismConfig,
        calibration: SimulationCalibration,
    ) -> Result<Self, StepCostError> {
        if score.placement.rank_to_gpu.is_empty() {
            return Err(StepCostError::EmptyPlacement);
        }
        let calibration = calibration.sanitized();
        let config = score.config;
        let shard_factor = Solver::latency_shard_factor(config);
        let attention_shard_factor = Solver::attention_shard_factor(config);
        let peak_flops =
            Solver::effective_peak_flops(cluster, model, &score.placement, calibration);
        let hbm_bandwidth = Solver::effective_hbm_bandwidth(cluster, &score.placement, calibration);
        let layers = f64::from(model.layers);
        let hidden = f64::from(model.hidden_size);
        let head_dim = f64::from(model.hidden_size / model.attention_heads.max(1));
        let kv_bytes_per_token = 2.0
            * layers
            * f64::from(model.kv_heads)
            * head_dim
            * model.kv_dtype().bytes_per_element() as f64;

        Ok(Self {
            cluster,
            placement: &score.placement,
            calibration,
            dtype: model.dtype,
            peak_flops,
            hbm_bandwidth,
            dense_flops_per_token: 2.0 * model.parameter_count() / shard_factor,
            // Solver::prefill_attention_flops: 2 * layers * hidden * prompt^2.
            attention_flops_per_pair: 2.0 * layers * hidden / attention_shard_factor,
            // Solver::decode_attention_flops: 4 * layers * hidden per context token.
            decode_attention_flops_per_context_token: 4.0 * layers * hidden
                / attention_shard_factor,
            weight_bytes: model.parameters.as_bytes() as f64 / shard_factor,
            kv_bytes_per_context_token: kv_bytes_per_token / attention_shard_factor,
            collectives: step_collectives(model, config, &score.groups),
            communication_cache: HashMap::new(),
        })
    }

    /// Latency of one step carrying `work`. An empty step costs nothing.
    pub fn step_latency(&mut self, work: &StepWork) -> StepLatency {
        if work.is_empty() {
            return StepLatency::default();
        }
        let prefill_flops = work.prefill_tokens as f64 * self.dense_flops_per_token
            + work.prefill_attention_pairs * self.attention_flops_per_pair;
        let decode_flops = work.decode_sequences as f64 * self.dense_flops_per_token
            + work.decode_context_tokens as f64 * self.decode_attention_flops_per_context_token;
        let compute_s = prefill_flops / self.peak_flops * self.calibration.prefill_compute_scale
            + decode_flops / self.peak_flops * self.calibration.decode_compute_scale;
        let kv_read_tokens = (work.decode_context_tokens + work.prefill_context_tokens) as f64;
        let memory_s = (self.weight_bytes + kv_read_tokens * self.kv_bytes_per_context_token)
            / self.hbm_bandwidth
            * self.calibration.decode_compute_scale;
        let forward_s = compute_s.max(memory_s);
        let communication_s = self.communication_s(work.total_tokens(), work.sampling_sequences());
        let overhead_s = self.calibration.scheduler_overhead_us / 1e6;
        let busy_s = if self.calibration.allow_compute_comm_overlap {
            forward_s.max(communication_s)
        } else {
            forward_s + communication_s
        };
        StepLatency {
            compute_s,
            memory_s,
            communication_s,
            overhead_s,
            total_s: busy_s + overhead_s,
        }
    }

    fn communication_s(&mut self, tokens: u64, sampling_sequences: u64) -> f64 {
        if self.collectives.is_empty() || tokens == 0 {
            return 0.0;
        }
        let key = (tokens, sampling_sequences);
        if let Some(cached) = self.communication_cache.get(&key) {
            return *cached;
        }
        let total_s = self
            .collectives
            .iter()
            .filter_map(|collective| {
                let bytes = collective.size.bytes(tokens, sampling_sequences);
                if bytes == 0 {
                    return None;
                }
                let call = CollectiveCall {
                    kind: collective.kind,
                    participants: collective.participants.clone(),
                    bytes_per_rank: Bytes::from_bytes(bytes),
                    dtype: self.dtype,
                    reduction: collective.reduction,
                    root: None,
                    phase: InferencePhase::EndToEnd,
                    algorithm: collective.algorithm,
                };
                let cost = Solver::estimate_collective_with_calibration(
                    self.cluster,
                    self.placement,
                    &call,
                    self.calibration,
                );
                Some(cost.total_s * f64::from(collective.repeats))
            })
            .sum::<f64>();
        self.communication_cache.insert(key, total_s);
        total_s
    }
}

/// The collectives one step issues, mirroring `build_operation_trace`: an
/// embedding all-reduce per first-stage tensor group; per layer, two
/// all-reduces (attention, MLP) per tensor group in the layer's stage and one
/// all-to-all per expert group; one send/recv per stage edge; and an LM-head
/// logits all-gather per last-stage tensor group, sized by sampling sequences.
fn step_collectives(
    model: &ModelSpec,
    config: ParallelismConfig,
    groups: &ParallelGroups,
) -> Vec<StepCollective> {
    let token_bytes = u64::from(model.hidden_size) * model.dtype.bytes_per_element();
    let layers = model.layers.max(1);
    let stages = config.pipeline_ranks.max(1);
    let mut layers_per_stage = vec![0_u32; stages as usize];
    for layer_idx in 0..layers {
        let stage = operations::layer_stage(layer_idx, layers, stages) as usize;
        if let Some(count) = layers_per_stage.get_mut(stage) {
            *count += 1;
        }
    }
    let stage_of_group = |group: &[RankId]| -> Option<usize> {
        let first = group.first()?;
        groups
            .pipeline_stages
            .iter()
            .position(|stage_ranks| stage_ranks.contains(first))
            .or(Some(0))
    };

    let last_stage = groups.pipeline_stages.len().saturating_sub(1);
    let mut collectives = Vec::new();
    if config.tensor_ranks > 1 {
        let logits_shard_bytes = div_ceil(
            u64::from(model.vocab_size),
            u64::from(config.tensor_ranks.max(1)),
        ) * model.dtype.bytes_per_element();
        for group in &groups.tensor_groups {
            let Some(stage) = stage_of_group(group) else {
                continue;
            };
            if stage == 0 {
                collectives.push(StepCollective {
                    kind: CollectiveKind::AllReduce,
                    participants: group.clone(),
                    size: StepCollectiveSize::PerToken(token_bytes),
                    reduction: Some(ReductionOp::Sum),
                    algorithm: CollectiveAlgorithm::Hierarchical,
                    repeats: 1,
                });
            }
            if stage == last_stage {
                collectives.push(StepCollective {
                    kind: CollectiveKind::AllGather,
                    participants: group.clone(),
                    size: StepCollectiveSize::PerSamplingSequence(logits_shard_bytes),
                    reduction: None,
                    algorithm: CollectiveAlgorithm::Ring,
                    repeats: 1,
                });
            }
        }
        for group in &groups.tensor_groups {
            let Some(stage) = stage_of_group(group) else {
                continue;
            };
            let stage_layers = layers_per_stage.get(stage).copied().unwrap_or(0);
            if stage_layers == 0 {
                continue;
            }
            collectives.push(StepCollective {
                kind: CollectiveKind::AllReduce,
                participants: group.clone(),
                size: StepCollectiveSize::PerToken(token_bytes),
                reduction: Some(ReductionOp::Sum),
                algorithm: CollectiveAlgorithm::Hierarchical,
                repeats: 2 * stage_layers,
            });
        }
    }
    if let Some(experts) = model.experts
        && config.expert_ranks > 1
    {
        let expert_token_bytes =
            token_bytes * u64::from(experts.top_k) / u64::from(config.expert_ranks.max(1));
        for group in &groups.expert_groups {
            let Some(stage) = stage_of_group(group) else {
                continue;
            };
            let stage_layers = layers_per_stage.get(stage).copied().unwrap_or(0);
            if stage_layers == 0 {
                continue;
            }
            collectives.push(StepCollective {
                kind: CollectiveKind::AllToAll,
                participants: group.clone(),
                size: StepCollectiveSize::PerToken(expert_token_bytes),
                reduction: None,
                algorithm: CollectiveAlgorithm::Hierarchical,
                repeats: stage_layers,
            });
        }
    }
    if config.pipeline_ranks > 1 {
        for window in groups.pipeline_stages.windows(2) {
            if let [left, right] = window
                && let (Some(&left_rank), Some(&right_rank)) = (left.first(), right.first())
            {
                collectives.push(StepCollective {
                    kind: CollectiveKind::SendRecv,
                    participants: vec![left_rank, right_rank],
                    size: StepCollectiveSize::PerToken(token_bytes),
                    reduction: None,
                    algorithm: CollectiveAlgorithm::Auto,
                    repeats: 1,
                });
            }
        }
    }
    collectives
}

#[cfg(test)]
mod tests;
