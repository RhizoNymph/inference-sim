//! Maps routed request states and `[serving.traffic]` knobs onto engine
//! workers, limits, and requests.
//!
//! | knob | engine meaning |
//! |---|---|
//! | `max_prefill_batch_tokens` | per-step token budget (vLLM `max_num_batched_tokens`), decode tokens included |
//! | `max_prefill_chunk_tokens` | largest prefill chunk one request takes per step |
//! | `max_decode_batch_tokens`, `max_decode_sequences_per_node`/`_per_gpu` | running sequences per worker (vLLM `max_num_seqs`) |
//! | `max_decode_sequences`, `max_resident_tokens`, `max_kv_blocks` | admission limits across all workers |
//! | `max_resident_tokens_per_node`/`_per_gpu`, `max_kv_blocks_per_node`/`_per_gpu` | per-worker KV limits, scaled by the worker's node/GPU count |
//! | `max_prefill_tokens`, `max_prefill_tokens_per_node`/`_per_gpu` | prefill tokens per worker step |
//! | traffic-class `max_decode_sequences`/`max_resident_tokens`/`max_kv_blocks`/`max_prefill_tokens` | the same limits per class |
//! | `max_queue_delay` (and class/trace overrides) | waiting longer rejects the request |
//!
//! A request whose own footprint exceeds a limit is rejected on arrival;
//! otherwise exceeding a limit means waiting, never rejection.

use super::*;

/// One engine replica: the GPU set its requests are routed to.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct EngineWorker {
    pub(in crate::serving) gpus: Vec<GpuAddr>,
    pub(in crate::serving) nodes: Vec<NodeId>,
    pub(in crate::serving) resources: Vec<String>,
    /// Indices of the request states routed to this worker.
    pub(in crate::serving) members: Vec<usize>,
}

/// Groups request states by routed GPU set, in first-appearance order.
pub(in crate::serving) fn engine_workers(states: &[DecodeRequestState]) -> Vec<EngineWorker> {
    let mut workers: Vec<EngineWorker> = Vec::new();
    for (state_idx, state) in states.iter().enumerate() {
        let gpus = route_worker_gpus(&state.decode_route_gpus, state.decode_node);
        if let Some(worker) = workers.iter_mut().find(|worker| worker.gpus == gpus) {
            worker.members.push(state_idx);
            continue;
        }
        let mut nodes = gpus.iter().map(|gpu| gpu.node_id).collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes.dedup();
        let mut resources = nodes
            .iter()
            .flat_map(|node_id| {
                [
                    format!("gpu compute node {node_id}"),
                    format!("gpu HBM node {node_id}"),
                ]
            })
            .collect::<Vec<_>>();
        resources.sort();
        workers.push(EngineWorker {
            gpus,
            nodes,
            resources,
            members: vec![state_idx],
        });
    }
    workers
}

pub(in crate::serving) fn engine_limits(
    traffic: &ServingTraffic,
    workers: &[EngineWorker],
) -> EngineLimits {
    let (token_budget, chunk_tokens) = match traffic.prefill_batching {
        ServingPrefillBatching::Continuous {
            max_batch_tokens,
            chunk_tokens,
        } => (max_batch_tokens.unwrap_or(u64::MAX).max(1), chunk_tokens),
        ServingPrefillBatching::Independent => (u64::MAX, None),
    };
    let decode_batch_sequences = match traffic.decode_batching {
        ServingDecodeBatching::Continuous { max_batch_tokens } => max_batch_tokens.map(u64::from),
        ServingDecodeBatching::Independent => None,
    };
    let prefill_tokens_per_step = min_option([
        traffic.max_prefill_tokens,
        traffic.max_prefill_tokens_per_node,
        traffic.max_prefill_tokens_per_gpu,
    ]);
    let workers = workers
        .iter()
        .map(|worker| {
            let node_count = worker.nodes.len().max(1) as u64;
            let gpu_count = worker.gpus.len().max(1) as u64;
            WorkerLimits {
                capacity: CapacityLimits {
                    sequences: min_option([
                        decode_batch_sequences,
                        traffic.max_decode_sequences_per_node.map(u64::from),
                        traffic.max_decode_sequences_per_gpu.map(u64::from),
                    ]),
                    tokens: min_option([
                        traffic
                            .max_resident_tokens_per_node
                            .map(|limit| limit.saturating_mul(node_count)),
                        traffic
                            .max_resident_tokens_per_gpu
                            .map(|limit| limit.saturating_mul(gpu_count)),
                    ]),
                    blocks: min_option([
                        traffic
                            .max_kv_blocks_per_node
                            .map(|limit| limit.saturating_mul(node_count)),
                        traffic
                            .max_kv_blocks_per_gpu
                            .map(|limit| limit.saturating_mul(gpu_count)),
                    ]),
                },
                step: StepLimits {
                    token_budget,
                    chunk_tokens,
                    prefill_tokens: prefill_tokens_per_step,
                },
            }
        })
        .collect();
    EngineLimits {
        global: CapacityLimits {
            sequences: traffic.max_decode_sequences.map(u64::from),
            tokens: traffic.max_resident_tokens,
            blocks: traffic.max_kv_blocks,
        },
        workers,
        classes: traffic
            .traffic_classes
            .iter()
            .map(|class| ClassLimits {
                name: class.name.clone(),
                capacity: CapacityLimits {
                    sequences: class.max_decode_sequences.map(u64::from),
                    tokens: class.max_resident_tokens,
                    blocks: class.max_kv_blocks,
                },
                prefill_tokens_per_step: class.max_prefill_tokens,
            })
            .collect(),
    }
}

/// One engine request per routed state, indexed like the states.
pub(in crate::serving) fn engine_requests(
    states: &[DecodeRequestState],
    traffic: &ServingTraffic,
    workers: &[EngineWorker],
    limits: &EngineLimits,
) -> Vec<EngineRequest> {
    let mut worker_of_state = vec![0; states.len()];
    for (worker_idx, worker) in workers.iter().enumerate() {
        for state_idx in &worker.members {
            if let Some(slot) = worker_of_state.get_mut(*state_idx) {
                *slot = worker_idx;
            }
        }
    }
    states
        .iter()
        .enumerate()
        .map(|(state_idx, state)| {
            let sequences = state.batch_size.max(1);
            let prompt_tokens = state.prompt_tokens.max(1);
            let prefill_tokens = state.effective_prefill_tokens.clamp(1, prompt_tokens);
            EngineRequest {
                worker: worker_of_state[state_idx],
                arrival_s: state.arrival_s,
                priority: state.priority,
                request_idx: state.request_idx,
                sequences,
                cached_prompt_tokens: prompt_tokens - prefill_tokens,
                prefill_tokens,
                output_tokens: state.decode_tokens.max(1),
                footprint: KvFootprint {
                    sequences: u64::from(sequences),
                    tokens: u64::from(sequences) * u64::from(state.max_sequence_tokens.max(1)),
                    blocks: state.kv_cache_blocks,
                },
                class: state
                    .traffic_class
                    .as_deref()
                    .and_then(|name| limits.classes.iter().position(|class| class.name == name)),
                cancellation_s: state.cancellation_s,
                max_queue_delay_s: state.max_queue_delay_s.or(traffic.max_queue_delay_s),
            }
        })
        .collect()
}

fn min_option<const N: usize>(values: [Option<u64>; N]) -> Option<u64> {
    values.into_iter().flatten().min()
}
