//! Disaggregated prefill -> KV pull -> decode semantics with the closed-form
//! `FakeCost` (20 ms floor, 0.1 ms per decode sequence, 0.2 ms per prefill
//! token) and hand-built transfer plans.

use super::super::core::run_engine_jobs;
use super::super::transfer::{KvFlow, KvTransferPlan};
use super::*;

const PREFILL: WorkerId = 0;
const DECODE: WorkerId = 1;

fn link_plan(link: usize, service_s: f64) -> KvTransferPlan {
    KvTransferPlan::new(
        vec![KvFlow::new(vec![link], service_s).expect("valid flow")],
        1_000_000,
    )
}

fn job(
    arrival_s: f64,
    request_idx: u32,
    prompt: u32,
    output: u32,
    plan: KvTransferPlan,
    first_token: FirstTokenSource,
) -> EngineJob {
    let mut base = request(arrival_s, request_idx, prompt, output);
    base.worker = PREFILL;
    base.footprint = KvFootprint {
        sequences: 1,
        tokens: u64::from(prompt + 1),
        blocks: u64::from(prompt + 1).div_ceil(16),
    };
    EngineJob {
        request: base,
        route: JobRoute::Disaggregated(Handoff {
            decode_worker: DECODE,
            decode_footprint: KvFootprint {
                sequences: 1,
                tokens: u64::from(prompt + output),
                blocks: u64::from(prompt + output).div_ceil(16),
            },
            plan,
            first_token,
        }),
    }
}

fn decode_job(arrival_s: f64, request_idx: u32, prompt: u32, output: u32, service_s: f64) -> EngineJob {
    job(
        arrival_s,
        request_idx,
        prompt,
        output,
        link_plan(0, service_s),
        FirstTokenSource::DecodeInstance,
    )
}

fn two_workers(token_budget: u64) -> EngineLimits {
    let mut limits = limits(token_budget);
    limits.workers.push(limits.workers[0]);
    limits
}

fn run_jobs(jobs: &[EngineJob], limits: &EngineLimits) -> EngineOutcome {
    run_engine_jobs(jobs, limits, &mut UniformCost(&mut FakeCost::new())).expect("engine runs")
}

fn handoff(outcome: &EngineOutcome, id: usize) -> &HandoffRecord {
    outcome.timelines[id]
        .handoff
        .as_ref()
        .expect("prefill finished")
}

fn transfer(outcome: &EngineOutcome, id: usize) -> &super::super::transfer::TransferWindow {
    handoff(outcome, id).transfer.as_ref().expect("pull started")
}

const PREFILL_512_S: f64 = 0.020 + 512.0 * 0.0002;
const RECOMPUTE_S: f64 = 0.020 + 0.0002;
const DECODE_1_S: f64 = 0.020 + 0.0001;

#[test]
fn prefill_then_pull_then_decode_with_the_decode_instance_first_token() {
    let jobs = [decode_job(0.0, 0, 512, 4, 0.05)];
    let outcome = run_jobs(&jobs, &two_workers(2048));
    let timeline = &outcome.timelines[0];
    assert_eq!(timeline.fate, Some(RequestFate::Completed));
    assert_eq!(timeline.chunks.len(), 1);
    assert_eq!(timeline.chunks[0].tokens, 512);
    assert!(close(timeline.chunks[0].finish_s, PREFILL_512_S));

    let record = handoff(&outcome, 0);
    assert!(close(record.prefill_finish_s, PREFILL_512_S));
    assert!(close(record.decode_queued_s, PREFILL_512_S));
    assert_eq!(record.decode_admitted_s, Some(record.prefill_finish_s));
    let window = transfer(&outcome, 0);
    assert!(close(window.start_s, PREFILL_512_S));
    assert!(close(window.finish_s, PREFILL_512_S + 0.05));

    // All four client tokens come from the decode worker; the first from the
    // step that recomputes the last prompt token after the pull.
    assert_eq!(timeline.tokens.len(), 4);
    let first_token_s = PREFILL_512_S + 0.05 + RECOMPUTE_S;
    assert!(close(ttft(&outcome, &requests_of(&jobs), 0), first_token_s));
    assert!(close(timeline.tokens[0].start_s, timeline.tokens[0].finish_s));
    for itl in itls(&outcome, 0) {
        assert!(close(itl, DECODE_1_S));
    }
    let recompute = record.recompute_step.expect("recomputed");
    assert_eq!(outcome.steps[recompute].worker, DECODE);
    assert_eq!(outcome.steps[recompute].work.prefill_tokens(), 1);
    assert_eq!(outcome.steps[recompute].first_token_sequences, 1);
    // The prefill step samples an internal token the client never sees.
    assert_eq!(outcome.steps[0].worker, PREFILL);
    assert!(outcome.steps[0].token_requests.is_empty());
    assert_eq!(outcome.steps[0].first_token_sequences, 0);
}

fn requests_of(jobs: &[EngineJob]) -> Vec<EngineRequest> {
    jobs.iter().map(|job| job.request.clone()).collect()
}

#[test]
fn prefill_instance_convention_counts_the_prefill_token_as_first() {
    let jobs = [job(
        0.0,
        0,
        512,
        4,
        link_plan(0, 0.05),
        FirstTokenSource::PrefillInstance,
    )];
    let outcome = run_jobs(&jobs, &two_workers(2048));
    let timeline = &outcome.timelines[0];
    assert_eq!(timeline.fate, Some(RequestFate::Completed));
    assert_eq!(timeline.tokens.len(), 4);
    assert!(close(timeline.tokens[0].finish_s, PREFILL_512_S));
    assert_eq!(outcome.steps[0].first_token_sequences, 1);
    // Token 2 is the decode worker's first step after the pull: no recompute.
    assert!(close(
        timeline.tokens[1].finish_s,
        PREFILL_512_S + 0.05 + DECODE_1_S
    ));
    assert_eq!(handoff(&outcome, 0).recompute_step, None);
    assert!(
        outcome
            .steps
            .iter()
            .filter(|step| step.worker == DECODE)
            .all(|step| step.work.prefill_tokens() == 0)
    );
}

#[test]
fn single_output_token_under_prefill_instance_never_reaches_decode() {
    let jobs = [job(
        0.0,
        0,
        64,
        1,
        link_plan(0, 0.05),
        FirstTokenSource::PrefillInstance,
    )];
    let outcome = run_jobs(&jobs, &two_workers(2048));
    assert_eq!(outcome.timelines[0].fate, Some(RequestFate::Completed));
    assert_eq!(outcome.timelines[0].tokens.len(), 1);
    assert!(outcome.timelines[0].handoff.is_none());
    assert!(outcome.steps.iter().all(|step| step.worker == PREFILL));
}

#[test]
fn decode_admission_waits_for_decode_capacity_before_pulling() {
    let jobs = [decode_job(0.0, 0, 512, 6, 0.05), decode_job(0.0, 1, 512, 6, 0.05)];
    let mut limits = two_workers(2048);
    limits.workers[DECODE].capacity.sequences = Some(1);
    let outcome = run_jobs(&jobs, &limits);
    for id in 0..2 {
        assert_eq!(outcome.timelines[id].fate, Some(RequestFate::Completed));
    }
    // Both prefill in one step and queue on decode at the same instant.
    assert!(close(
        handoff(&outcome, 0).decode_queued_s,
        handoff(&outcome, 1).decode_queued_s
    ));
    let first_done_s = outcome.timelines[0]
        .tokens
        .last()
        .map(|token| token.finish_s)
        .expect("tokens");
    let second_admitted_s = handoff(&outcome, 1).decode_admitted_s.expect("admitted");
    assert!(second_admitted_s >= first_done_s - 1e-12);
    // The pull starts only once the decode worker admits the request.
    assert!(close(transfer(&outcome, 1).start_s, second_admitted_s));
    assert!(
        outcome
            .steps
            .iter()
            .filter(|step| step.worker == DECODE)
            .all(|step| step.running_sequences <= 1)
    );
}

#[test]
fn decode_kv_blocks_gate_the_pull() {
    let jobs = [decode_job(0.0, 0, 512, 6, 0.05), decode_job(0.0, 1, 512, 6, 0.05)];
    let mut limits = two_workers(2048);
    // One request's decode KV (518 tokens) fits, two do not.
    limits.workers[DECODE].capacity.tokens = Some(600);
    let outcome = run_jobs(&jobs, &limits);
    let first_done_s = outcome.timelines[0]
        .tokens
        .last()
        .map(|token| token.finish_s)
        .expect("tokens");
    assert!(handoff(&outcome, 1).decode_admitted_s.expect("admitted") >= first_done_s - 1e-12);
    assert_eq!(outcome.timelines[1].fate, Some(RequestFate::Completed));
}

#[test]
fn concurrent_pulls_over_one_link_serialize() {
    let jobs = [decode_job(0.0, 0, 512, 3, 0.1), decode_job(0.0, 1, 512, 3, 0.1)];
    let outcome = run_jobs(&jobs, &two_workers(2048));
    let (first, second) = (transfer(&outcome, 0), transfer(&outcome, 1));
    assert!(close(first.start_s, second.ready_s));
    assert!(close(second.start_s, first.finish_s));
    assert!(close(second.finish_s, first.finish_s + 0.1));
    assert_eq!(second.predecessors, vec![0]);
    let requests = requests_of(&jobs);
    assert!(ttft(&outcome, &requests, 1) >= ttft(&outcome, &requests, 0) + 0.1 - 1e-9);
}

#[test]
fn pulls_on_disjoint_links_overlap() {
    let jobs = [
        job(0.0, 0, 512, 3, link_plan(0, 0.1), FirstTokenSource::DecodeInstance),
        job(0.0, 1, 512, 3, link_plan(1, 0.1), FirstTokenSource::DecodeInstance),
    ];
    let outcome = run_jobs(&jobs, &two_workers(2048));
    assert!(close(transfer(&outcome, 0).finish_s, transfer(&outcome, 1).finish_s));
    assert!(transfer(&outcome, 1).predecessors.is_empty());
}

#[test]
fn prefill_sequence_slot_frees_at_handoff_but_prompt_kv_waits_for_the_pull() {
    // Sequence-limited prefill worker: the second prefill starts right after
    // the first prompt completes, before its KV is pulled.
    let jobs = [decode_job(0.0, 0, 512, 3, 0.3), decode_job(0.0, 1, 512, 3, 0.3)];
    let mut limits = two_workers(2048);
    limits.workers[PREFILL].capacity.sequences = Some(1);
    let outcome = run_jobs(&jobs, &limits);
    assert!(close(
        outcome.timelines[1].admitted_s.expect("admitted"),
        handoff(&outcome, 0).prefill_finish_s
    ));

    // KV-limited prefill worker: one prompt's KV fits, so the second prefill
    // waits until the first prompt's KV has been pulled away.
    let mut limits = two_workers(2048);
    limits.workers[PREFILL].capacity.tokens = Some(600);
    let outcome = run_jobs(&jobs, &limits);
    let admitted_s = outcome.timelines[1].admitted_s.expect("admitted");
    assert!(admitted_s >= transfer(&outcome, 0).finish_s - 1e-12);
    assert!(admitted_s < transfer(&outcome, 0).finish_s + 0.05);
    assert_eq!(outcome.timelines[1].fate, Some(RequestFate::Completed));
}

#[test]
fn prefill_queue_builds_independently_of_decode() {
    // Prefill takes 122 ms per prompt; arrivals every 50 ms queue on prefill
    // while the decode worker keeps decoding.
    let jobs = (0..6)
        .map(|idx| decode_job(f64::from(idx) * 0.05, idx, 512, 8, 0.01))
        .collect::<Vec<_>>();
    let mut limits = two_workers(512);
    limits.workers[PREFILL].capacity.sequences = Some(1);
    let outcome = run_jobs(&jobs, &limits);
    let requests = requests_of(&jobs);
    let waits = (0..6)
        .map(|id| outcome.timelines[id].admitted_s.expect("admitted") - requests[id].arrival_s)
        .collect::<Vec<_>>();
    assert!(waits[5] > waits[1] + 0.1, "{waits:?}");
    // Decode steps never carry more than the one-token recomputes.
    assert!(
        outcome
            .steps
            .iter()
            .filter(|step| step.worker == DECODE)
            .all(|step| step.work.prefill_tokens() <= step.first_token_sequences)
    );
}

#[test]
fn request_whose_decode_kv_never_fits_is_rejected_on_arrival() {
    let jobs = [decode_job(0.0, 0, 512, 600, 0.05)];
    let mut limits = two_workers(2048);
    limits.workers[DECODE].capacity.tokens = Some(1000);
    let outcome = run_jobs(&jobs, &limits);
    match &outcome.timelines[0].fate {
        Some(RequestFate::NeverFits(excess)) => {
            assert_eq!(excess.scope, CapacityScope::Worker);
            assert_eq!(excess.resource, CapacityResource::Tokens);
            assert_eq!(excess.needed, 1112);
        }
        other => panic!("expected NeverFits, got {other:?}"),
    }
    assert!(outcome.timelines[0].chunks.is_empty());
    assert!(outcome.steps.is_empty());
}

#[test]
fn cancellation_during_the_pull_frees_decode_capacity() {
    let mut jobs = vec![decode_job(0.0, 0, 512, 50, 0.2), decode_job(0.0, 1, 512, 3, 0.01)];
    // Request 0 is cancelled while its KV is in flight.
    jobs[0].request.cancellation_s = Some(PREFILL_512_S + 0.05);
    let mut limits = two_workers(2048);
    limits.workers[DECODE].capacity.sequences = Some(1);
    let outcome = run_jobs(&jobs, &limits);
    assert!(matches!(
        outcome.timelines[0].fate,
        Some(RequestFate::Cancelled { .. })
    ));
    assert!(outcome.timelines[0].tokens.is_empty());
    assert_eq!(outcome.timelines[1].fate, Some(RequestFate::Completed));
    // Request 1 is admitted at the first decode step boundary after the
    // cancellation, well before request 0 would have finished.
    let admitted_s = handoff(&outcome, 1).decode_admitted_s.expect("admitted");
    assert!(admitted_s < PREFILL_512_S + 0.2 + 0.1);
}

#[test]
fn colocated_and_disaggregated_jobs_share_one_run() {
    let colocated = EngineJob::colocated(request(0.0, 0, 256, 4));
    let mut disaggregated = decode_job(0.0, 1, 256, 4, 0.02);
    disaggregated.request.worker = PREFILL;
    let outcome = run_jobs(&[colocated, disaggregated], &two_workers(2048));
    assert!(outcome.timelines[0].handoff.is_none());
    assert!(outcome.timelines[1].handoff.is_some());
    for timeline in &outcome.timelines {
        assert_eq!(timeline.fate, Some(RequestFate::Completed));
        assert_eq!(timeline.tokens.len(), 4);
    }
}

#[test]
fn handoff_to_its_own_prefill_worker_is_a_typed_error() {
    let mut bad = decode_job(0.0, 0, 64, 2, 0.01);
    if let JobRoute::Disaggregated(handoff) = &mut bad.route {
        handoff.decode_worker = PREFILL;
    }
    let result = run_engine_jobs(
        &[bad],
        &two_workers(2048),
        &mut UniformCost(&mut FakeCost::new()),
    );
    assert!(matches!(
        result,
        Err(EngineError::HandoffToSameWorker {
            request: 0,
            worker: PREFILL
        })
    ));
    let mut unknown = decode_job(0.0, 0, 64, 2, 0.01);
    if let JobRoute::Disaggregated(handoff) = &mut unknown.route {
        handoff.decode_worker = 7;
    }
    let result = run_engine_jobs(
        &[unknown],
        &two_workers(2048),
        &mut UniformCost(&mut FakeCost::new()),
    );
    assert!(matches!(
        result,
        Err(EngineError::UnknownWorker {
            request: 0,
            worker: 7
        })
    ));
}

#[test]
fn disaggregated_engine_is_deterministic() {
    let jobs = (0..40)
        .map(|idx| decode_job(f64::from(idx) * 0.03, idx, 200 + (idx % 5) * 100, 6 + idx % 4, 0.04))
        .collect::<Vec<_>>();
    let mut limits = two_workers(1024);
    limits.workers[DECODE].capacity.sequences = Some(5);
    let first = run_jobs(&jobs, &limits);
    assert_eq!(first, run_jobs(&jobs, &limits));
    assert!(
        first
            .timelines
            .iter()
            .all(|timeline| timeline.fate == Some(RequestFate::Completed))
    );
}

#[test]
fn cross_worker_capacity_deadlock_starves_instead_of_looping() {
    // A global token limit that fits either footprint alone but not the
    // prefill-held prompt KV plus the decode reservation: the pull can never
    // start, and nothing else can free capacity.
    let jobs = [decode_job(0.0, 0, 512, 4, 0.05)];
    let mut limits = two_workers(2048);
    limits.global.tokens = Some(600);
    let outcome = run_jobs(&jobs, &limits);
    assert!(matches!(
        outcome.timelines[0].fate,
        Some(RequestFate::Starved { .. })
    ));
    assert!(handoff(&outcome, 0).transfer.is_none());
}
