//! Writes an engine run back into the shared request states, scheduled
//! operations, and decode-iteration observations that the serving summary,
//! metrics, evidence, JSON, and CSV outputs consume.

use super::limits::EngineWorker;
use super::*;

/// Records `outcome` into `states` (indexed like `requests`) and returns the
/// step operations and decode iterations.
pub(in crate::serving) fn record_engine_outcome(
    states: &mut [DecodeRequestState],
    requests: &[EngineRequest],
    workers: &[EngineWorker],
    outcome: &EngineOutcome,
) -> (
    Vec<ScheduledOperation>,
    Vec<ServingDecodeIterationObservation>,
) {
    let operations = step_operations(&outcome.steps, workers);
    let decode_iterations = step_decode_iterations(&outcome.steps, requests, workers);
    for ((state, request), timeline) in states.iter_mut().zip(requests).zip(&outcome.timelines) {
        let gpus = workers
            .get(request.worker)
            .map(|worker| worker.gpus.clone())
            .unwrap_or_default();
        record_request(state, request, timeline, &gpus);
    }
    (operations, decode_iterations)
}

fn step_operations(steps: &[EngineStep], workers: &[EngineWorker]) -> Vec<ScheduledOperation> {
    let mut previous_on_worker: Vec<Option<usize>> = vec![None; workers.len()];
    steps
        .iter()
        .enumerate()
        .map(|(id, step)| {
            let name = if step.work.prefill_tokens() > 0 {
                format!(
                    "engine step {id} prefill {} tokens decode {} sequences",
                    step.work.prefill_tokens(),
                    step.work.decode_sequences()
                )
            } else {
                format!(
                    "engine step {id} decode {} sequences",
                    step.work.decode_sequences()
                )
            };
            let previous = previous_on_worker
                .get_mut(step.worker)
                .and_then(|slot| slot.replace(id));
            let resource_dependencies = previous
                .filter(|previous| {
                    steps
                        .get(*previous)
                        .is_some_and(|prior| prior.finish_s + 1e-12 >= step.start_s)
                })
                .into_iter()
                .collect();
            ScheduledOperation {
                id,
                name,
                resources: workers
                    .get(step.worker)
                    .map(|worker| worker.resources.clone())
                    .unwrap_or_default(),
                start_s: step.start_s,
                finish_s: step.finish_s,
                explicit_dependencies: Vec::new(),
                resource_dependencies,
            }
        })
        .collect()
}

fn step_decode_iterations(
    steps: &[EngineStep],
    requests: &[EngineRequest],
    workers: &[EngineWorker],
) -> Vec<ServingDecodeIterationObservation> {
    let mut iterations = Vec::new();
    for (id, step) in steps.iter().enumerate() {
        if step.token_requests.is_empty() {
            continue;
        }
        let batch_tokens = step
            .token_requests
            .iter()
            .map(|request| u64::from(requests[*request].sequences.max(1)))
            .sum::<u64>();
        let first_token_batch_tokens = step.first_token_sequences.min(batch_tokens);
        iterations.push(ServingDecodeIterationObservation {
            iteration_idx: iterations.len().min(u32::MAX as usize) as u32,
            decode_nodes: workers
                .get(step.worker)
                .map(|worker| worker.nodes.clone())
                .unwrap_or_default(),
            request_indices: step
                .token_requests
                .iter()
                .map(|request| requests[*request].request_idx)
                .collect(),
            operation_ids: vec![id],
            start_s: step.start_s,
            finish_s: step.finish_s,
            latency_s: (step.finish_s - step.start_s).max(0.0),
            batch_tokens: saturating_u32(batch_tokens),
            first_token_batch_tokens: saturating_u32(first_token_batch_tokens),
            tail_token_batch_tokens: saturating_u32(batch_tokens - first_token_batch_tokens),
        });
    }
    iterations
}

fn record_request(
    state: &mut DecodeRequestState,
    request: &EngineRequest,
    timeline: &RequestTimeline,
    gpus: &[GpuAddr],
) {
    let sequences = u64::from(request.sequences.max(1));
    if let (Some(first), Some(last)) = (timeline.chunks.first(), timeline.chunks.last()) {
        let computed_tokens = timeline
            .chunks
            .iter()
            .map(|chunk| chunk.tokens)
            .sum::<u64>()
            / sequences;
        state.prefill_start_s = first.start_s;
        state.prefill_finish_s = last.finish_s;
        state.prefill_chunks = timeline.chunks.len().min(u32::MAX as usize) as u32;
        state.remaining_prefill_tokens = state
            .effective_prefill_tokens
            .saturating_sub(computed_tokens.min(u64::from(u32::MAX)) as u32);
        state.prefill_scheduled = state.remaining_prefill_tokens == 0;
        state.prefill_resource_queue_s =
            finite_or_zero(timeline.admitted_s.unwrap_or(first.start_s) - state.arrival_s);
        state.prefill_token_spans = timeline
            .chunks
            .iter()
            .map(|chunk| PrefillTokenSpan {
                start_s: chunk.start_s,
                finish_s: chunk.finish_s,
                tokens: chunk.tokens,
            })
            .collect();
        state.worker_assignments.extend(worker_assignments(
            "prefill",
            gpus,
            first.start_s,
            last.finish_s,
            timeline.chunks.iter().map(|chunk| chunk.step).collect(),
        ));
    } else {
        state.prefill_start_s = f64::INFINITY;
        state.prefill_finish_s = f64::INFINITY;
    }

    if let Some(first_token) = timeline.tokens.first() {
        // Colocated engine: the KV cache never moves, so the handoff is instant.
        state.kv_start_s = first_token.finish_s;
        state.kv_finish_s = first_token.finish_s;
        state.first_decode_start_s = Some(first_token.start_s);
        state.first_decode_finish_s = Some(first_token.finish_s);
        state.decode_token_start_s = timeline.tokens.iter().map(|token| token.start_s).collect();
        state.decode_token_finish_s = timeline.tokens.iter().map(|token| token.finish_s).collect();
        state.last_decode_finish_s = state.decode_token_finish_s.last().copied();
        state.emitted_tokens = timeline.tokens.len().min(u32::MAX as usize) as u32;
        state.remaining_tokens = state.decode_tokens.saturating_sub(state.emitted_tokens);
        if let (Some(second), Some(last)) = (timeline.tokens.get(1), timeline.tokens.last()) {
            state.worker_assignments.extend(worker_assignments(
                "decode",
                gpus,
                second.start_s,
                last.finish_s,
                timeline.tokens[1..]
                    .iter()
                    .map(|token| token.step)
                    .collect(),
            ));
        }
    } else {
        state.kv_start_s = f64::INFINITY;
        state.kv_finish_s = f64::INFINITY;
    }
    state.dependencies = timeline
        .chunks
        .iter()
        .map(|chunk| chunk.step)
        .chain(timeline.tokens.iter().map(|token| token.step))
        .max()
        .into_iter()
        .collect();

    match &timeline.fate {
        Some(RequestFate::Completed) => {}
        Some(RequestFate::Cancelled { at_s }) => {
            let reason = if timeline.chunks.is_empty() {
                "request cancelled before prefill admission"
            } else if timeline.tokens.is_empty() {
                "request cancelled during prefill"
            } else {
                "request cancelled during decode"
            };
            cancel_request(state, *at_s, reason);
        }
        Some(RequestFate::QueueTimeout { waited_s, limit_s }) => {
            reject_admission(state, *waited_s, *limit_s);
        }
        Some(RequestFate::NeverFits(excess)) => {
            let arrival_s = state.arrival_s;
            reject_prefill_capacity_admission(
                state,
                arrival_s,
                never_fits_rejection(*excess, request),
            );
        }
        Some(RequestFate::Starved { at_s }) => {
            reject_prefill_capacity_admission(state, *at_s, starved_rejection());
        }
        None => {}
    }
}

fn worker_assignments(
    phase: &str,
    gpus: &[GpuAddr],
    start_s: f64,
    finish_s: f64,
    operation_ids: Vec<usize>,
) -> Vec<ServingWorkerAssignmentObservation> {
    gpus.iter()
        .map(|gpu| ServingWorkerAssignmentObservation {
            phase: phase.to_string(),
            node_id: gpu.node_id,
            local_gpu_id: gpu.local_gpu_id,
            slot: 0,
            start_s,
            finish_s,
            operation_ids: operation_ids.clone(),
        })
        .collect()
}

fn never_fits_rejection(excess: CapacityExcess, request: &EngineRequest) -> ServingRejection {
    let (scope_resource, scope_code, scope_label) = match excess.scope {
        CapacityScope::Global => (String::new(), String::new(), String::new()),
        CapacityScope::Worker => (
            "worker ".to_string(),
            "worker_".to_string(),
            " per worker".to_string(),
        ),
        CapacityScope::Class(class) => (
            format!("traffic_class {class} "),
            "traffic_class_".to_string(),
            format!(" for traffic class {class}"),
        ),
    };
    let (resource, code, unit, remediation) = match excess.resource {
        CapacityResource::Sequences => (
            "decode_sequences",
            "decode_capacity_exceeded",
            "sequences",
            "increase max_decode_sequences (or max_decode_batch_tokens), or split the request's batch",
        ),
        CapacityResource::Tokens => (
            "resident_tokens",
            "kv_residency_capacity_exceeded",
            "tokens",
            "increase max_resident_tokens, reduce max_sequence_tokens, or add capacity",
        ),
        CapacityResource::Blocks => (
            "kv_blocks",
            "kv_block_capacity_exceeded",
            "blocks",
            "increase max_kv_blocks, reduce max_sequence_tokens, or add capacity",
        ),
    };
    let message = format!(
        "admission rejected: request {} needs {} {unit}{scope_label} but the limit is {}, so it can never be admitted",
        request.request_idx, excess.needed, excess.limit
    );
    decode_capacity_admission_request_rejection(
        format!("{scope_resource}{resource}"),
        &format!("{scope_code}{code}"),
        message,
        excess.needed as f64,
        excess.limit as f64,
        unit,
        remediation,
    )
}

fn starved_rejection() -> ServingRejection {
    request_rejection(
        "prefill",
        "capacity",
        "engine_admission",
        "engine_admission_starved",
        "admission rejected: capacity held by other workers could never be freed for this request"
            .to_string(),
    )
}

fn saturating_u32(value: u64) -> u32 {
    value.min(u64::from(u32::MAX)) as u32
}
