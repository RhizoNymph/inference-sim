use super::core::run_engine;
use super::*;
use crate::solver::{StepLatency, StepWork};

mod disaggregated;
mod disaggregated_serving;
mod frontend;
mod serving;

/// Closed-form step cost: a flat weight-read floor per step, a small cost per
/// decode sequence, and a per-token prefill cost.
struct FakeCost {
    floor_s: f64,
    per_decode_s: f64,
    per_prefill_token_s: f64,
}

impl FakeCost {
    fn new() -> Self {
        Self {
            floor_s: 0.020,
            per_decode_s: 0.0001,
            per_prefill_token_s: 0.0002,
        }
    }
}

impl StepCost for FakeCost {
    fn step_latency(&mut self, work: &StepWork) -> StepLatency {
        let total_s = self.floor_s
            + self.per_decode_s * work.decode_sequences() as f64
            + self.per_prefill_token_s * work.prefill_tokens() as f64;
        StepLatency {
            compute_s: total_s,
            memory_s: self.floor_s,
            communication_s: 0.0,
            overhead_s: 0.0,
            total_s,
        }
    }
}

fn request(arrival_s: f64, request_idx: u32, prompt: u32, output: u32) -> EngineRequest {
    EngineRequest {
        worker: 0,
        arrival_s,
        priority: 0,
        request_idx,
        sequences: 1,
        cached_prompt_tokens: 0,
        prefill_tokens: prompt,
        output_tokens: output,
        footprint: KvFootprint {
            sequences: 1,
            tokens: u64::from(prompt + output),
            blocks: u64::from(prompt + output).div_ceil(16),
        },
        class: None,
        cancellation_s: None,
        max_queue_delay_s: None,
    }
}

fn limits(token_budget: u64) -> EngineLimits {
    EngineLimits {
        global: CapacityLimits::default(),
        workers: vec![WorkerLimits {
            capacity: CapacityLimits::default(),
            step: StepLimits {
                token_budget,
                chunk_tokens: None,
                prefill_tokens: None,
            },
        }],
        classes: Vec::new(),
    }
}

fn run(requests: &[EngineRequest], limits: &EngineLimits) -> EngineOutcome {
    run_engine(requests, limits, &mut FakeCost::new()).expect("engine runs")
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

fn ttft(outcome: &EngineOutcome, requests: &[EngineRequest], id: usize) -> f64 {
    outcome.timelines[id].tokens[0].finish_s - requests[id].arrival_s
}

fn itls(outcome: &EngineOutcome, id: usize) -> Vec<f64> {
    outcome.timelines[id]
        .tokens
        .windows(2)
        .map(|pair| pair[1].finish_s - pair[0].finish_s)
        .collect()
}

#[test]
fn single_request_first_token_comes_from_the_prefill_step() {
    let requests = [request(0.5, 0, 512, 4)];
    let outcome = run(&requests, &limits(2048));
    let timeline = &outcome.timelines[0];
    assert_eq!(timeline.fate, Some(RequestFate::Completed));
    assert_eq!(timeline.admitted_s, Some(0.5));
    assert_eq!(timeline.chunks.len(), 1);
    assert_eq!(timeline.chunks[0].tokens, 512);
    let prefill_s = 0.020 + 512.0 * 0.0002;
    assert!(close(ttft(&outcome, &requests, 0), prefill_s));
    // First token is an instant at the end of the prefill step.
    assert!(close(
        timeline.tokens[0].start_s,
        timeline.tokens[0].finish_s
    ));
    assert_eq!(timeline.tokens.len(), 4);
    for itl in itls(&outcome, 0) {
        assert!(close(itl, 0.020 + 0.0001));
    }
    // One prefill step plus three decode steps.
    assert_eq!(outcome.steps.len(), 4);
    assert_eq!(outcome.steps[0].first_token_sequences, 1);
}

#[test]
fn long_prompt_is_chunked_by_the_token_budget() {
    let requests = [request(0.0, 0, 5000, 2)];
    let outcome = run(&requests, &limits(2048));
    let chunks = outcome.timelines[0]
        .chunks
        .iter()
        .map(|chunk| chunk.tokens)
        .collect::<Vec<_>>();
    assert_eq!(chunks, vec![2048, 2048, 904]);
    // The first token waits for the last chunk.
    assert!(close(
        outcome.timelines[0].tokens[0].finish_s,
        outcome.steps[2].finish_s
    ));
}

#[test]
fn chunk_cap_limits_one_requests_share_of_a_step() {
    let requests = [request(0.0, 0, 300, 1), request(0.0, 1, 300, 1)];
    let mut limits = limits(512);
    limits.workers[0].step.chunk_tokens = Some(128);
    let outcome = run(&requests, &limits);
    assert_eq!(outcome.steps[0].work.prefill_tokens(), 256);
    assert_eq!(outcome.timelines[0].chunks.len(), 3);
    assert_eq!(outcome.timelines[1].chunks.len(), 3);
}

#[test]
fn decode_steps_that_carry_a_prefill_chunk_are_slower() {
    // Request 0 is decoding when request 1 arrives; the step that prefills
    // request 1 also decodes request 0, so request 0 sees one slow ITL.
    let requests = [request(0.0, 0, 512, 20), request(0.2, 1, 1024, 4)];
    let outcome = run(&requests, &limits(2048));
    let gaps = itls(&outcome, 0);
    let fast = 0.020 + 0.0001;
    let slow = gaps.iter().copied().fold(0.0, f64::max);
    assert!(close(slow, 0.020 + 0.0001 + 1024.0 * 0.0002));
    let median = {
        let mut sorted = gaps.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    assert!(close(median, fast) || close(median, fast + 0.0001));
    let tokens = &outcome.timelines[0].tokens;
    let tpot = (tokens[tokens.len() - 1].finish_s - tokens[0].finish_s) / (tokens.len() - 1) as f64;
    assert!(tpot > median * 1.05, "tpot {tpot} vs median itl {median}");
}

#[test]
fn requests_wait_for_a_running_slot_instead_of_being_rejected() {
    let requests = (0..5)
        .map(|idx| request(0.0, idx, 64, 3))
        .collect::<Vec<_>>();
    let mut limits = limits(2048);
    limits.workers[0].capacity.sequences = Some(2);
    let outcome = run(&requests, &limits);
    assert!(
        outcome
            .timelines
            .iter()
            .all(|timeline| timeline.fate == Some(RequestFate::Completed))
    );
    assert!(outcome.steps.iter().all(|step| step.running_sequences <= 2));
    let first_tokens = (0..5)
        .map(|id| ttft(&outcome, &requests, id))
        .collect::<Vec<_>>();
    assert!(close(first_tokens[0], first_tokens[1]));
    assert!(first_tokens[2] > first_tokens[1] + 0.04);
    assert!(first_tokens[4] > first_tokens[2]);
    // A queued request is admitted in the step after a slot frees.
    let released_s = outcome.timelines[0]
        .tokens
        .last()
        .map(|token| token.finish_s);
    assert_eq!(outcome.timelines[2].admitted_s, released_s);
}

#[test]
fn kv_block_capacity_gates_admission() {
    let requests = (0..3)
        .map(|idx| request(0.0, idx, 100, 4))
        .collect::<Vec<_>>();
    let mut limits = limits(2048);
    // Each request needs ceil(104 / 16) = 7 blocks; 14 fit two at a time.
    limits.global.blocks = Some(14);
    let outcome = run(&requests, &limits);
    assert_eq!(outcome.steps[0].prefill_requests, vec![0, 1]);
    assert!(outcome.timelines[2].admitted_s.unwrap_or(0.0) > 0.0);
    assert_eq!(outcome.timelines[2].fate, Some(RequestFate::Completed));
}

#[test]
fn request_larger_than_capacity_is_rejected_on_arrival() {
    let requests = [request(0.0, 0, 100, 4), request(0.0, 1, 4000, 4)];
    let mut limits = limits(2048);
    limits.global.tokens = Some(1000);
    let outcome = run(&requests, &limits);
    assert_eq!(outcome.timelines[0].fate, Some(RequestFate::Completed));
    match &outcome.timelines[1].fate {
        Some(RequestFate::NeverFits(excess)) => {
            assert_eq!(excess.scope, CapacityScope::Global);
            assert_eq!(excess.resource, CapacityResource::Tokens);
            assert_eq!(excess.needed, 4004);
            assert_eq!(excess.limit, 1000);
        }
        other => panic!("expected NeverFits, got {other:?}"),
    }
    assert!(outcome.timelines[1].chunks.is_empty());
}

#[test]
fn queue_delay_limit_rejects_requests_that_wait_too_long() {
    let mut requests = (0..3)
        .map(|idx| request(0.0, idx, 64, 50))
        .collect::<Vec<_>>();
    for request in &mut requests {
        request.max_queue_delay_s = Some(0.5);
    }
    let mut limits = limits(2048);
    limits.workers[0].capacity.sequences = Some(1);
    let outcome = run(&requests, &limits);
    assert_eq!(outcome.timelines[0].fate, Some(RequestFate::Completed));
    for id in [1, 2] {
        match outcome.timelines[id].fate {
            Some(RequestFate::QueueTimeout { waited_s, limit_s }) => {
                assert!(waited_s > limit_s);
                assert!(close(limit_s, 0.5));
            }
            ref other => panic!("expected QueueTimeout, got {other:?}"),
        }
    }
}

#[test]
fn cancellation_removes_waiting_and_running_requests() {
    let mut requests = vec![
        request(0.0, 0, 64, 100),
        request(0.0, 1, 64, 100),
        request(0.0, 2, 64, 4),
    ];
    requests[0].cancellation_s = Some(0.3);
    requests[1].cancellation_s = Some(0.1);
    let mut limits = limits(2048);
    limits.workers[0].capacity.sequences = Some(1);
    let outcome = run(&requests, &limits);
    assert_eq!(
        outcome.timelines[0].fate,
        Some(RequestFate::Cancelled { at_s: 0.3 })
    );
    assert!(outcome.timelines[0].tokens.len() > 1);
    assert_eq!(
        outcome.timelines[1].fate,
        Some(RequestFate::Cancelled { at_s: 0.1 })
    );
    assert!(outcome.timelines[1].chunks.is_empty());
    assert_eq!(outcome.timelines[2].fate, Some(RequestFate::Completed));
    // The cancelled request's slot frees at the first step boundary after 0.3 s.
    assert!(outcome.timelines[2].admitted_s.unwrap_or(0.0) >= 0.3);
}

#[test]
fn higher_priority_requests_are_admitted_first() {
    let mut requests = (0..3)
        .map(|idx| request(0.0, idx, 64, 2))
        .collect::<Vec<_>>();
    requests[2].priority = 5;
    let mut limits = limits(2048);
    limits.workers[0].capacity.sequences = Some(1);
    let outcome = run(&requests, &limits);
    assert_eq!(outcome.steps[0].prefill_requests, vec![2]);
}

#[test]
fn token_budget_bounds_every_step_and_still_makes_progress() {
    let requests = (0..6)
        .map(|idx| request(0.0, idx, 8, 5))
        .collect::<Vec<_>>();
    let outcome = run(&requests, &limits(4));
    assert!(
        outcome
            .steps
            .iter()
            .all(|step| step.work.total_tokens() <= 4)
    );
    assert!(
        outcome
            .timelines
            .iter()
            .all(|timeline| timeline.fate == Some(RequestFate::Completed))
    );
}

#[test]
fn workers_share_global_capacity_in_time_order() {
    let mut requests = vec![request(0.0, 0, 64, 10), request(0.01, 1, 64, 10)];
    requests[1].worker = 1;
    let mut limits = limits(2048);
    limits.workers.push(limits.workers[0]);
    limits.global.sequences = Some(1);
    let outcome = run(&requests, &limits);
    let first_done = outcome.timelines[0]
        .tokens
        .last()
        .map(|token| token.finish_s)
        .unwrap_or(f64::NAN);
    let second_admitted = outcome.timelines[1].admitted_s.unwrap_or(f64::NAN);
    assert!(second_admitted >= first_done - 1e-12);
    assert!(second_admitted < first_done + 0.001);
    assert_eq!(outcome.timelines[1].fate, Some(RequestFate::Completed));
}

#[test]
fn unknown_worker_is_a_typed_error() {
    let mut requests = vec![request(0.0, 0, 64, 2)];
    requests[0].worker = 3;
    let result = run_engine(&requests, &limits(2048), &mut FakeCost::new());
    assert!(matches!(
        result,
        Err(EngineError::UnknownWorker {
            request: 0,
            worker: 3
        })
    ));
}

#[test]
fn engine_is_deterministic() {
    let requests = (0..40)
        .map(|idx| {
            request(
                f64::from(idx) * 0.013,
                idx,
                200 + (idx % 7) * 90,
                8 + idx % 5,
            )
        })
        .collect::<Vec<_>>();
    let mut limits = limits(512);
    limits.workers[0].capacity.sequences = Some(6);
    limits.workers[0].step.chunk_tokens = Some(256);
    assert_eq!(run(&requests, &limits), run(&requests, &limits));
}
