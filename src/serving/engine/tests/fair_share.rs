//! Fair-share KV pulls and the proxy handoff delay in the engine loop, with
//! the closed-form `FakeCost` (20 ms floor, 0.1 ms per decode sequence,
//! 0.2 ms per prefill token) and hand-built single-link plans.

use super::super::core::{run_engine_jobs, run_engine_jobs_with};
use super::super::transfer::{KvFlow, KvTransferPlan, TransferWindow};
use super::*;
use crate::calibration::KvLinkSharing;

const PREFILL: WorkerId = 0;
const DECODE: WorkerId = 1;
const PREFILL_512_S: f64 = 0.020 + 512.0 * 0.0002;
const RECOMPUTE_S: f64 = 0.020 + 0.0002;

fn link_plan(link: usize, service_s: f64) -> KvTransferPlan {
    KvTransferPlan::new(
        vec![KvFlow::new(vec![link], service_s).expect("valid flow")],
        1_000_000,
    )
}

fn pulled_job(arrival_s: f64, request_idx: u32, output: u32, plan: KvTransferPlan) -> EngineJob {
    let prompt = 512;
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
            first_token: FirstTokenSource::DecodeInstance,
        }),
    }
}

fn two_workers() -> EngineLimits {
    let mut limits = limits(2048);
    limits.workers.push(limits.workers[0]);
    limits
}

fn fair(delay_s: f64) -> HandoffModel {
    HandoffModel {
        link_sharing: KvLinkSharing::FairShare,
        delay: HandoffDelay::new(delay_s).expect("valid delay"),
    }
}

fn fifo(delay_s: f64) -> HandoffModel {
    HandoffModel {
        link_sharing: KvLinkSharing::Fifo,
        delay: HandoffDelay::new(delay_s).expect("valid delay"),
    }
}

fn run_with(jobs: &[EngineJob], limits: &EngineLimits, model: HandoffModel) -> EngineOutcome {
    run_engine_jobs_with(jobs, limits, &mut UniformCost(&mut FakeCost::new()), model)
        .expect("engine runs")
}

fn window(outcome: &EngineOutcome, id: usize) -> &TransferWindow {
    outcome.timelines[id]
        .handoff
        .as_ref()
        .and_then(|handoff| handoff.transfer.as_ref())
        .expect("pull recorded")
}

fn requests_of(jobs: &[EngineJob]) -> Vec<EngineRequest> {
    jobs.iter().map(|job| job.request.clone()).collect()
}

#[test]
fn handoff_delay_rejects_invalid_values() {
    assert!(HandoffDelay::new(-0.001).is_err());
    assert!(HandoffDelay::new(f64::NAN).is_err());
    assert!(HandoffDelay::new(f64::INFINITY).is_err());
    assert_eq!(HandoffDelay::new(0.04).expect("valid").seconds(), 0.04);
    assert_eq!(HandoffDelay::default().seconds(), 0.0);
}

#[test]
fn default_handoff_model_is_the_plain_engine() {
    let jobs = [
        pulled_job(0.0, 0, 4, link_plan(0, 0.1)),
        pulled_job(0.0, 1, 4, link_plan(0, 0.1)),
    ];
    let plain = run_engine_jobs(
        &jobs,
        &two_workers(),
        &mut UniformCost(&mut FakeCost::new()),
    )
    .expect("engine runs");
    assert_eq!(plain, run_with(&jobs, &two_workers(), fifo(0.0)));
}

#[test]
fn a_lone_pull_is_identical_under_fifo_and_fair_share() {
    let jobs = [pulled_job(0.0, 0, 6, link_plan(0, 0.081))];
    let fifo_run = run_with(&jobs, &two_workers(), fifo(0.0));
    let fair_run = run_with(&jobs, &two_workers(), fair(0.0));
    assert_eq!(fifo_run, fair_run);
    assert_eq!(window(&fair_run, 0).finish_s, PREFILL_512_S + 0.081);
}

#[test]
fn spaced_pulls_are_identical_under_fifo_and_fair_share() {
    // Pulls that never overlap give the same run under both disciplines.
    // Only the dependency bookkeeping differs: a FIFO pull follows the last
    // pull on its link, a fair-share pull depends only on concurrent ones.
    let jobs = (0..5)
        .map(|idx| pulled_job(f64::from(idx) * 0.5, idx, 4, link_plan(0, 0.08)))
        .collect::<Vec<_>>();
    let without_predecessors = |mut outcome: EngineOutcome| {
        for timeline in &mut outcome.timelines {
            if let Some(transfer) = timeline
                .handoff
                .as_mut()
                .and_then(|handoff| handoff.transfer.as_mut())
            {
                transfer.predecessors.clear();
            }
        }
        outcome
    };
    let fair_run = run_with(&jobs, &two_workers(), fair(0.0));
    assert!(
        (0..5).all(|id| window(&fair_run, id).predecessors.is_empty()),
        "fair-share pulls never overlapped"
    );
    assert_eq!(
        without_predecessors(run_with(&jobs, &two_workers(), fifo(0.0))),
        without_predecessors(fair_run)
    );
}

#[test]
fn concurrent_pulls_share_the_link_instead_of_queueing() {
    // Both prompts finish in one prefill step, so both pulls start together
    // and each runs at half speed.
    let jobs = [
        pulled_job(0.0, 0, 3, link_plan(0, 0.1)),
        pulled_job(0.0, 1, 3, link_plan(0, 0.1)),
    ];
    let outcome = run_with(&jobs, &two_workers(), fair(0.0));
    let start_s = 0.020 + 1024.0 * 0.0002;
    for id in 0..2 {
        let pull = window(&outcome, id);
        assert!(close(pull.start_s, start_s), "{pull:?}");
        assert!(close(pull.finish_s, start_s + 0.2), "{pull:?}");
        assert!(close(pull.contention_s, 0.1));
        // The decode worker wakes exactly when the shared pulls finish.
        let first_token_s = outcome.timelines[id].tokens[0].finish_s;
        assert!(close(first_token_s, start_s + 0.2 + 0.020 + 2.0 * 0.0002));
    }
    assert_eq!(window(&outcome, 1).predecessors, vec![0]);
    // FIFO serializes the same pulls: same last finish, earlier first one.
    let serialized = run_with(&jobs, &two_workers(), fifo(0.0));
    assert!(close(window(&serialized, 0).finish_s, start_s + 0.1));
    assert!(close(window(&serialized, 1).finish_s, start_s + 0.2));
}

#[test]
fn a_handoff_landing_mid_pull_starts_its_pull_immediately() {
    // The second prompt finishes prefill while the first pull is in flight
    // on a slow link; the sleeping decode worker admits it at once and the
    // two pulls share the link from then on.
    let jobs = [
        pulled_job(0.0, 0, 3, link_plan(0, 1.0)),
        pulled_job(0.0, 1, 3, link_plan(0, 1.0)),
    ];
    let mut limits = two_workers();
    limits.workers[PREFILL].step.token_budget = 512;
    let outcome = run_with(&jobs, &limits, fair(0.0));
    let second_prefill_s = 2.0 * PREFILL_512_S;
    let second = outcome.timelines[1].handoff.as_ref().expect("handed off");
    assert!(close(second.prefill_finish_s, second_prefill_s));
    assert_eq!(second.decode_admitted_s, Some(second.decode_queued_s));
    assert!(close(window(&outcome, 1).start_s, second_prefill_s));
    // First: alone for one prefill step, then half speed until done.
    let first_finish_s = PREFILL_512_S + PREFILL_512_S + 2.0 * (1.0 - PREFILL_512_S);
    assert!(close(window(&outcome, 0).finish_s, first_finish_s));
    assert!(close(window(&outcome, 1).finish_s, PREFILL_512_S + 2.0));
}

#[test]
fn burst_of_pulls_keeps_aggregate_link_throughput() {
    let jobs = (0..8)
        .map(|idx| pulled_job(0.0, idx, 2, link_plan(0, 0.05)))
        .collect::<Vec<_>>();
    // One prefill step carries all eight prompts.
    let mut limits = limits(8 * 512);
    limits.workers.push(limits.workers[0]);
    let fair_run = run_with(&jobs, &limits, fair(0.0));
    let fifo_run = run_with(&jobs, &limits, fifo(0.0));
    let last = |outcome: &EngineOutcome| {
        (0..8)
            .map(|id| window(outcome, id).finish_s)
            .fold(0.0, f64::max)
    };
    assert!(close(last(&fair_run), last(&fifo_run)));
    // Fair sharing finishes every pull at that last instant.
    for id in 0..8 {
        assert!(close(window(&fair_run, id).finish_s, last(&fair_run)));
        assert_eq!(fair_run.timelines[id].fate, Some(RequestFate::Completed));
    }
}

#[test]
fn prefill_kv_is_held_until_the_shared_pull_finishes() {
    // KV-limited prefill: one prompt fits. The next prefill waits for the
    // (fair-share) pull to free the first prompt's KV.
    let jobs = [
        pulled_job(0.0, 0, 3, link_plan(0, 0.3)),
        pulled_job(0.0, 1, 3, link_plan(0, 0.3)),
    ];
    let mut limits = two_workers();
    limits.workers[PREFILL].capacity.tokens = Some(600);
    let outcome = run_with(&jobs, &limits, fair(0.0));
    let admitted_s = outcome.timelines[1].admitted_s.expect("admitted");
    assert!(close(admitted_s, window(&outcome, 0).finish_s));
    assert_eq!(outcome.timelines[1].fate, Some(RequestFate::Completed));
}

#[test]
fn proxy_delay_postpones_decode_queue_entry_and_ttft() {
    let jobs = [pulled_job(0.0, 0, 4, link_plan(0, 0.05))];
    let delay_s = 0.04;
    for model in [fair(delay_s), fifo(delay_s)] {
        let outcome = run_with(&jobs, &two_workers(), model);
        let record = outcome.timelines[0].handoff.as_ref().expect("handed off");
        assert!(close(record.prefill_finish_s, PREFILL_512_S));
        assert!(close(record.decode_queued_s, PREFILL_512_S + delay_s));
        assert_eq!(record.decode_admitted_s, Some(record.decode_queued_s));
        let pull = window(&outcome, 0);
        assert!(close(pull.start_s, PREFILL_512_S + delay_s));
        let ttft_s = ttft(&outcome, &requests_of(&jobs), 0);
        assert!(close(ttft_s, PREFILL_512_S + delay_s + 0.05 + RECOMPUTE_S));
    }
}

#[test]
fn proxy_delay_does_not_touch_colocated_requests() {
    let jobs = [EngineJob::colocated(request(0.0, 0, 512, 4))];
    let mut limits = limits(2048);
    limits.workers.push(limits.workers[0]);
    let plain = run_with(&jobs, &limits, fair(0.0));
    let delayed = run_with(&jobs, &limits, fair(0.5));
    assert_eq!(plain, delayed);
}

#[test]
fn a_pull_cancelled_mid_transfer_still_records_its_window() {
    let mut jobs = vec![
        pulled_job(0.0, 0, 4, link_plan(0, 0.2)),
        pulled_job(0.0, 1, 4, link_plan(0, 0.2)),
    ];
    let start_s = 0.020 + 1024.0 * 0.0002;
    jobs[0].request.cancellation_s = Some(start_s + 0.05);
    let outcome = run_with(&jobs, &two_workers(), fair(0.0));
    assert!(matches!(
        outcome.timelines[0].fate,
        Some(RequestFate::Cancelled { .. })
    ));
    let cancelled = window(&outcome, 0);
    assert!(cancelled.finish_s > start_s + 0.05);
    assert_eq!(outcome.timelines[1].fate, Some(RequestFate::Completed));
}

#[test]
fn fair_share_runs_are_deterministic() {
    let jobs = (0..12)
        .map(|idx| {
            pulled_job(
                f64::from(idx) * 0.03,
                idx,
                6,
                link_plan(idx as usize % 2, 0.07),
            )
        })
        .collect::<Vec<_>>();
    let first = run_with(&jobs, &two_workers(), fair(0.01));
    let second = run_with(&jobs, &two_workers(), fair(0.01));
    assert_eq!(first, second);
    for id in 0..12 {
        assert_eq!(first.timelines[id].fate, Some(RequestFate::Completed));
    }
}
