//! KV-exhaustion preemption under `KvAllocationMode::PagedRecompute` (vLLM V1
//! recompute mode) with the closed-form `FakeCost` (20 ms floor, 0.1 ms per
//! decode sequence, 0.2 ms per prefill token).

use std::num::NonZeroU32;

use super::super::core::run_engine_jobs_with;
use super::super::transfer::{KvFlow, KvTransferPlan};
use super::*;

const BLOCK: u32 = 16;

fn paged() -> KvAllocationMode {
    KvAllocationMode::PagedRecompute {
        block_tokens: NonZeroU32::new(BLOCK).expect("non-zero block"),
    }
}

fn run_jobs_with(jobs: &[EngineJob], limits: &EngineLimits, kv: KvAllocationMode) -> EngineOutcome {
    run_engine_jobs_with(jobs, limits, kv, &mut UniformCost(&mut FakeCost::new()))
        .expect("engine runs")
}

fn run_paged(requests: &[EngineRequest], limits: &EngineLimits) -> EngineOutcome {
    let jobs = requests
        .iter()
        .cloned()
        .map(EngineJob::colocated)
        .collect::<Vec<_>>();
    run_jobs_with(&jobs, limits, paged())
}

fn block_limited(token_budget: u64, blocks: u64) -> EngineLimits {
    let mut limits = limits(token_budget);
    limits.workers[0].capacity.blocks = Some(blocks);
    limits
}

fn all_completed(outcome: &EngineOutcome) -> bool {
    outcome
        .timelines
        .iter()
        .all(|timeline| timeline.fate == Some(RequestFate::Completed))
}

/// Two 32-token prompts with 20 output tokens on a 6-block worker: both fit
/// at admission (2 blocks each) and through decode token 16 (3 blocks each);
/// the decode step that computes context token 49 needs a fourth block each.
fn two_requests() -> Vec<EngineRequest> {
    vec![request(0.0, 0, 32, 20), request(0.0, 1, 32, 20)]
}

const PREFILL_64_S: f64 = 0.020 + 64.0 * 0.0002;
const DECODE_2_S: f64 = 0.020 + 2.0 * 0.0001;
const DECODE_1_S: f64 = 0.020 + 0.0001;

#[test]
fn kv_exhaustion_preempts_the_most_recently_admitted_request() {
    let requests = two_requests();
    let outcome = run_paged(&requests, &block_limited(2048, 6));
    assert!(all_completed(&outcome));
    assert!(outcome.timelines[0].preemptions.is_empty());

    let preempted = &outcome.timelines[1];
    assert_eq!(preempted.preemptions.len(), 1);
    let record = preempted.preemptions[0];
    // Steps 1-16 decode both requests; the 17th decode step needs a fourth
    // block for each and preempts the newer request.
    let preempt_s = PREFILL_64_S + 16.0 * DECODE_2_S;
    assert!(close(record.at_s, preempt_s));
    assert_eq!(record.worker, 0);
    assert_eq!(record.emitted_tokens, 17);
    // Prompt (32) plus the 16 decoded tokens whose KV had been computed.
    assert_eq!(record.discarded_tokens, 48);
    assert_eq!(outcome.steps[record.step].preempted_requests, vec![1]);
    assert_eq!(outcome.steps[record.step].token_requests, vec![0]);

    // The older request runs to completion undisturbed.
    let first_done_s = preempt_s + 3.0 * DECODE_1_S;
    assert!(close(
        outcome.timelines[0].tokens.last().expect("tokens").finish_s,
        first_done_s
    ));
    // The preempted request waits for the four blocks its 49-token context
    // needs, then recomputes it in one chunk, which samples token 18.
    assert_eq!(record.resumed_s, Some(first_done_s));
    let resume = preempted.chunks.last().expect("resume chunk");
    assert_eq!(resume.pass, ChunkPass::Resume);
    assert_eq!(resume.tokens, 49);
    assert_eq!(preempted.tokens.len(), 20);
    let resume_finish_s = first_done_s + 0.020 + 49.0 * 0.0002;
    assert!(close(preempted.tokens[17].finish_s, resume_finish_s));
    assert!(close(preempted.tokens[17].start_s, first_done_s));
    assert!(close(
        preempted.tokens[19].finish_s,
        resume_finish_s + 2.0 * DECODE_1_S
    ));
    assert!(outcome.steps.iter().all(|step| step.kv_blocks <= 6));
    assert_eq!(
        outcome.preemption_stats(),
        PreemptionStats {
            preemptions: 1,
            preempted_requests: 1,
            recompute_tokens: 48,
        }
    );
}

#[test]
fn reserved_allocation_never_preempts_and_admits_whole_footprints() {
    let requests = two_requests();
    // Each request reserves ceil(52 / 16) = 4 blocks, so only one fits.
    let outcome = run(&requests, &block_limited(2048, 6));
    assert!(all_completed(&outcome));
    assert_eq!(outcome.steps[0].prefill_requests, vec![0]);
    assert_eq!(outcome.preemption_stats(), PreemptionStats::default());
    assert!(
        outcome
            .steps
            .iter()
            .all(|step| step.preempted_requests.is_empty())
    );
}

#[test]
fn paged_admission_needs_blocks_only_for_the_first_chunk() {
    let requests = [request(0.0, 0, 32, 40), request(0.0, 1, 32, 40)];
    // Reserved: ceil(72 / 16) = 5 blocks each, one at a time on 6 blocks.
    let reserved = run(&requests, &block_limited(2048, 6));
    assert_eq!(reserved.steps[0].prefill_requests, vec![0]);
    // Paged: the 32-token prompts need 2 blocks each.
    let paged = run_paged(&requests, &block_limited(2048, 6));
    assert_eq!(paged.steps[0].prefill_requests, vec![0, 1]);
    assert_eq!(paged.steps[0].kv_blocks, 4);
    assert!(all_completed(&paged));
    assert!(
        paged
            .timelines
            .iter()
            .all(|timeline| timeline.tokens.len() == 40)
    );
}

#[test]
fn paged_allocation_without_kv_pressure_matches_reserved() {
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
    let reserved = run(&requests, &limits);
    let paged = run_paged(&requests, &limits);
    assert_eq!(reserved.timelines, paged.timelines);
    assert_eq!(reserved.steps.len(), paged.steps.len());
    for (left, right) in reserved.steps.iter().zip(&paged.steps) {
        assert_eq!(left.work, right.work);
        assert_eq!(left.start_s, right.start_s);
        assert_eq!(left.finish_s, right.finish_s);
        assert_eq!(left.token_requests, right.token_requests);
    }
}

#[test]
fn paged_never_fits_uses_the_block_rounded_peak_context() {
    // Peak context 32 + 40 - 1 = 71 tokens = 5 blocks > 4.
    let requests = [request(0.0, 0, 32, 40), request(0.0, 1, 16, 4)];
    let outcome = run_paged(&requests, &block_limited(2048, 4));
    match &outcome.timelines[0].fate {
        Some(RequestFate::NeverFits(excess)) => {
            assert_eq!(excess.scope, CapacityScope::Worker);
            assert_eq!(excess.resource, CapacityResource::Blocks);
            assert_eq!(excess.needed, 5);
            assert_eq!(excess.limit, 4);
        }
        other => panic!("expected NeverFits, got {other:?}"),
    }
    assert_eq!(outcome.timelines[1].fate, Some(RequestFate::Completed));
}

#[test]
fn preempted_request_resumes_ahead_of_requests_that_arrived_earlier_in_the_queue() {
    let mut requests = two_requests();
    requests.push(request(0.1, 2, 16, 2));
    let mut limits = block_limited(2048, 6);
    limits.workers[0].capacity.sequences = Some(2);
    let outcome = run_paged(&requests, &limits);
    assert!(all_completed(&outcome));
    let record = outcome.timelines[1].preemptions[0];
    let resumed_s = record.resumed_s.expect("resumed");
    // The third request needs one block and one slot, both free while the
    // preempted request waits, but it queues behind it (head of line).
    let third_admitted_s = outcome.timelines[2].admitted_s.expect("admitted");
    assert!(third_admitted_s >= resumed_s - 1e-12);
    let first_done_s = outcome.timelines[0]
        .tokens
        .last()
        .map_or(f64::NAN, |token| token.finish_s);
    assert!(third_admitted_s >= first_done_s - 1e-12);
    assert!(record.at_s > 0.1);
}

#[test]
fn recompute_prefill_is_chunked_by_the_token_budget() {
    let requests = (0..6)
        .map(|idx| request(0.0, idx, 40, 30))
        .collect::<Vec<_>>();
    let budget = 24;
    let outcome = run_paged(&requests, &block_limited(budget, 12));
    assert!(all_completed(&outcome));
    assert!(outcome.preemption_stats().preemptions > 0);
    for step in &outcome.steps {
        assert!(step.work.total_tokens() <= budget);
        assert!(step.kv_blocks <= 12);
    }
    for (id, timeline) in outcome.timelines.iter().enumerate() {
        assert_eq!(timeline.tokens.len(), 30, "request {id}");
        let Some(last) = timeline.preemptions.last() else {
            continue;
        };
        if last.emitted_tokens == 0 {
            continue;
        }
        // The pass after the last preemption recomputes the prompt and every
        // emitted token, in budget-sized chunks; its last chunk samples the
        // next token.
        let resume = timeline
            .chunks
            .iter()
            .filter(|chunk| chunk.pass == ChunkPass::Resume && chunk.start_s >= last.at_s)
            .collect::<Vec<_>>();
        let recomputed = resume.iter().map(|chunk| chunk.tokens).sum::<u64>();
        assert_eq!(
            recomputed,
            40 + u64::from(last.emitted_tokens),
            "request {id}"
        );
        let final_chunk = resume.last().expect("resume chunks");
        let next = timeline.tokens[last.emitted_tokens as usize];
        assert_eq!(next.step, final_chunk.step);
        assert!(close(next.finish_s, final_chunk.finish_s));
    }
}

#[test]
fn every_preemption_frees_capacity_and_the_run_stays_deterministic() {
    let requests = (0..24)
        .map(|idx| {
            request(
                f64::from(idx) * 0.005,
                idx,
                48 + (idx % 4) * 16,
                20 + idx % 9,
            )
        })
        .collect::<Vec<_>>();
    let limits = block_limited(256, 30);
    let first = run_paged(&requests, &limits);
    assert_eq!(first, run_paged(&requests, &limits));
    assert!(all_completed(&first));
    assert!(first.preemption_stats().preemptions > 0);
    assert!(first.steps.iter().all(|step| step.kv_blocks <= 30));
    for (id, timeline) in first.timelines.iter().enumerate() {
        assert_eq!(
            timeline.tokens.len() as u32,
            requests[id].output_tokens,
            "request {id}"
        );
        assert!(
            timeline
                .tokens
                .windows(2)
                .all(|pair| pair[0].finish_s <= pair[1].finish_s)
        );
    }
}

#[test]
fn cancelled_preempted_request_leaves_the_queue() {
    let mut requests = two_requests();
    requests[1].cancellation_s = Some(0.38);
    let outcome = run_paged(&requests, &block_limited(2048, 6));
    assert_eq!(outcome.timelines[0].fate, Some(RequestFate::Completed));
    assert_eq!(outcome.timelines[1].preemptions.len(), 1);
    assert_eq!(outcome.timelines[1].preemptions[0].resumed_s, None);
    assert_eq!(
        outcome.timelines[1].fate,
        Some(RequestFate::Cancelled { at_s: 0.38 })
    );
}

const DECODE: WorkerId = 1;

fn disaggregated(arrival_s: f64, request_idx: u32, prompt: u32, output: u32) -> EngineJob {
    let mut base = request(arrival_s, request_idx, prompt, output);
    base.footprint = KvFootprint {
        sequences: 1,
        tokens: u64::from(prompt + 1),
        blocks: u64::from(prompt + 1).div_ceil(16),
    };
    let decode_footprint = KvFootprint {
        sequences: 1,
        tokens: u64::from(prompt + output),
        blocks: u64::from(prompt + output).div_ceil(16),
    };
    EngineJob {
        request: base,
        route: JobRoute::Disaggregated(Handoff {
            decode_worker: DECODE,
            decode_footprint,
            plan: KvTransferPlan::new(
                vec![KvFlow::new(vec![0], 0.01).expect("valid flow")],
                1_000_000,
            ),
            first_token: FirstTokenSource::DecodeInstance,
        }),
    }
}

#[test]
fn disaggregated_decode_worker_preempts_and_recomputes_locally() {
    let jobs = [disaggregated(0.0, 0, 32, 20), disaggregated(0.0, 1, 32, 20)];
    let mut limits = limits(2048);
    limits.workers.push(limits.workers[0]);
    limits.workers[DECODE].capacity.blocks = Some(6);
    let outcome = run_jobs_with(&jobs, &limits, paged());
    assert!(all_completed(&outcome));
    assert!(outcome.timelines[0].preemptions.is_empty());
    let timeline = &outcome.timelines[1];
    assert_eq!(timeline.tokens.len(), 20);
    let record = timeline.preemptions.first().expect("preempted on decode");
    assert_eq!(record.worker, DECODE);
    assert!(record.emitted_tokens > 0);
    // One pull: the decode worker recomputes the context itself.
    let handoff = timeline.handoff.as_ref().expect("handed off");
    assert!(handoff.transfer.is_some());
    let resume = timeline
        .chunks
        .iter()
        .filter(|chunk| chunk.pass == ChunkPass::Resume)
        .collect::<Vec<_>>();
    assert!(!resume.is_empty());
    assert!(
        resume
            .iter()
            .all(|chunk| outcome.steps[chunk.step].worker == DECODE)
    );
    assert_eq!(
        resume.iter().map(|chunk| chunk.tokens).sum::<u64>(),
        32 + u64::from(record.emitted_tokens)
    );
    assert!(
        outcome
            .steps
            .iter()
            .filter(|step| step.worker == DECODE)
            .all(|step| step.kv_blocks <= 6)
    );
    // The prefill worker only ever ran the two prompts.
    assert!(
        outcome
            .steps
            .iter()
            .filter(|step| step.worker == 0)
            .all(|step| step.preempted_requests.is_empty())
    );
}
