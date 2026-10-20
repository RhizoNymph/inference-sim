//! Fair-share link progress: max-min rates, exact standalone pricing, event
//! recomputation, projections, and the FIFO/fair-share network facade.

use super::*;
use crate::calibration::KvLinkSharing;

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

fn plan(flows: &[(&[LinkId], f64)]) -> KvTransferPlan {
    KvTransferPlan::new(
        flows
            .iter()
            .map(|(links, service_s)| KvFlow::new(links.to_vec(), *service_s).expect("valid flow"))
            .collect(),
        1,
    )
}

fn take(links: &mut FairShareLinks) -> Vec<(usize, TransferWindow)> {
    links.take_completed()
}

fn window(completed: &[(usize, TransferWindow)], ticket: usize) -> &TransferWindow {
    &completed
        .iter()
        .find(|(done, _)| *done == ticket)
        .expect("transfer completed")
        .1
}

#[test]
fn max_min_rates_split_a_shared_link_evenly() {
    assert_eq!(max_min_rates(&[&[0]]), vec![1.0]);
    assert_eq!(max_min_rates(&[&[0], &[0]]), vec![0.5, 0.5]);
    let three = max_min_rates(&[&[0], &[0], &[0]]);
    assert!(three.iter().all(|rate| close(*rate, 1.0 / 3.0)));
    assert!(max_min_rates(&[]).is_empty());
}

#[test]
fn max_min_rates_give_leftover_capacity_to_less_constrained_flows() {
    // B crosses both links; link 1 carries three flows, so B, C and D get a
    // third and A takes the rest of link 0.
    let rates = max_min_rates(&[&[0], &[0, 1], &[1], &[1]]);
    assert!(close(rates[0], 2.0 / 3.0), "{rates:?}");
    for rate in &rates[1..] {
        assert!(close(*rate, 1.0 / 3.0), "{rates:?}");
    }
}

#[test]
fn max_min_rates_cap_flows_at_their_standalone_speed() {
    // Disjoint links and routeless flows never exceed rate 1.
    assert_eq!(max_min_rates(&[&[0], &[1], &[]]), vec![1.0, 1.0, 1.0]);
    let mixed = max_min_rates(&[&[], &[0], &[0]]);
    assert_eq!(mixed, vec![1.0, 0.5, 0.5]);
}

#[test]
fn lone_flow_takes_exactly_its_uncontended_price() {
    let mut links = FairShareLinks::new();
    let start_s = 0.123_456_789;
    let service_s = 0.081_234_567;
    links.start(&plan(&[(&[0, 3], service_s)]), start_s, 7);
    assert_eq!(links.finish_s(7), Some(start_s + service_s));
    // Advancing through unrelated event times does not touch the flow.
    for at_s in [0.13, 0.15, 0.2] {
        links.advance_to(at_s);
        assert!(take(&mut links).is_empty());
    }
    links.advance_to(1.0);
    let completed = take(&mut links);
    let done = window(&completed, 7);
    assert_eq!(done.start_s, start_s);
    assert_eq!(done.ready_s, start_s);
    assert_eq!(done.finish_s, start_s + service_s);
    assert_eq!(done.contention_s, 0.0);
    assert!(done.predecessors.is_empty());
}

#[test]
fn simultaneous_transfers_on_one_link_share_it_and_finish_together() {
    let mut links = FairShareLinks::new();
    links.start(&plan(&[(&[0], 0.1)]), 1.0, 0);
    links.start(&plan(&[(&[0], 0.1)]), 1.0, 1);
    assert_eq!(links.rates(), vec![(0, 0, 0.5), (1, 0, 0.5)]);
    links.advance_to(2.0);
    let completed = take(&mut links);
    for ticket in [0, 1] {
        let done = window(&completed, ticket);
        assert!(close(done.start_s, 1.0));
        assert!(close(done.finish_s, 1.2), "{done:?}");
        assert!(close(done.contention_s, 0.1));
    }
    assert_eq!(window(&completed, 1).predecessors, vec![0]);
}

#[test]
fn staggered_transfers_share_only_while_overlapping() {
    // A runs alone for 50 ms, then both progress at half speed until A
    // finishes at 150 ms; B finishes its last 50 ms alone at 200 ms. The last
    // finish equals FIFO's (the link stays fully used).
    let mut links = FairShareLinks::new();
    links.start(&plan(&[(&[0], 0.1)]), 0.0, 0);
    assert!(close(links.finish_s(0).expect("projected"), 0.1));
    links.advance_to(0.05);
    links.start(&plan(&[(&[0], 0.1)]), 0.05, 1);
    assert!(close(links.finish_s(0).expect("projected"), 0.15));
    assert!(close(links.finish_s(1).expect("projected"), 0.2));
    assert!(close(links.next_finish_after(0.05).expect("next"), 0.15));
    links.advance_to(0.16);
    let first = take(&mut links);
    assert_eq!(first.len(), 1);
    assert!(close(window(&first, 0).finish_s, 0.15));
    assert_eq!(links.rates(), vec![(1, 0, 1.0)]);
    links.advance_to(1.0);
    let second = take(&mut links);
    let done = window(&second, 1);
    assert!(close(done.start_s, 0.05));
    assert!(close(done.finish_s, 0.2));
    assert!(close(done.contention_s, 0.05));
    assert_eq!(done.predecessors, vec![0]);
}

#[test]
fn projection_equals_the_recorded_finish_without_new_starts() {
    let mut links = FairShareLinks::new();
    links.start(&plan(&[(&[0], 0.3)]), 0.0, 0);
    links.start(&plan(&[(&[0], 0.1)]), 0.01, 1);
    links.start(&plan(&[(&[0, 1], 0.2)]), 0.02, 2);
    let projected = [0, 1, 2].map(|ticket| links.finish_s(ticket).expect("projected"));
    links.advance_to(10.0);
    let completed = take(&mut links);
    for (ticket, finish_s) in projected.iter().enumerate() {
        assert_eq!(window(&completed, ticket).finish_s, *finish_s);
    }
    // Work conservation on link 0: everything is done after 0.6 s of work.
    let last_s = projected.iter().copied().fold(0.0, f64::max);
    assert!(close(last_s, 0.6), "{projected:?}");
}

#[test]
fn a_new_start_delays_projections_of_transfers_sharing_its_links() {
    let mut links = FairShareLinks::new();
    links.start(&plan(&[(&[0], 0.1)]), 0.0, 0);
    links.start(&plan(&[(&[1], 0.1)]), 0.0, 1);
    let before = links.finish_s(0).expect("projected");
    links.start(&plan(&[(&[0], 0.1)]), 0.0, 2);
    assert!(links.finish_s(0).expect("projected") > before + 0.05);
    // The disjoint transfer is unaffected.
    assert!(close(links.finish_s(1).expect("projected"), 0.1));
}

#[test]
fn opposite_directions_and_disjoint_links_do_not_contend() {
    let mut links = FairShareLinks::new();
    links.start(&plan(&[(&[0], 0.1)]), 0.0, 0);
    links.start(&plan(&[(&[1], 0.1)]), 0.0, 1);
    links.advance_to(1.0);
    let completed = take(&mut links);
    assert_eq!(window(&completed, 0).finish_s, 0.1);
    assert_eq!(window(&completed, 1).finish_s, 0.1);
    assert!(window(&completed, 1).predecessors.is_empty());
}

#[test]
fn empty_and_zero_service_plans_complete_when_started() {
    let mut links = FairShareLinks::new();
    links.start(&KvTransferPlan::default(), 0.5, 0);
    links.start(&plan(&[(&[0], 0.0)]), 0.5, 1);
    assert_eq!(links.finish_s(0), Some(0.5));
    let completed = take(&mut links);
    assert_eq!(completed.len(), 2);
    assert_eq!(window(&completed, 1).finish_s, 0.5);
    assert_eq!(links.next_finish_after(0.0), None);
}

#[test]
fn a_multi_flow_transfer_ends_with_its_last_flow_and_matches_its_price() {
    // Two flows of one transfer on a shared link: alone, fair sharing between
    // them ends at the FIFO sum (0.05 + 0.07), the plan's uncontended price.
    let shared = plan(&[(&[0], 0.05), (&[0], 0.07)]);
    let mut links = FairShareLinks::new();
    links.start(&shared, 0.0, 0);
    links.advance_to(1.0);
    let completed = take(&mut links);
    let done = window(&completed, 0);
    assert!(close(done.finish_s, shared.uncontended_s()));
    assert!(close(done.flows[0].finish_s, 0.1));
    assert!(close(done.flows[1].finish_s, 0.12));
    assert_eq!(done.contention_s, 0.0);

    let parallel = plan(&[(&[0], 0.05), (&[1], 0.07)]);
    let mut links = FairShareLinks::new();
    links.start(&parallel, 0.0, 0);
    links.advance_to(1.0);
    assert_eq!(
        window(&take(&mut links), 0).finish_s,
        parallel.uncontended_s()
    );
}

#[test]
fn many_concurrent_pulls_keep_aggregate_throughput_and_stretch_each_transfer() {
    // The ledger-18 burst pattern: n equal pulls on one link at once all
    // finish after n services (aggregate unchanged), each n times slower.
    let mut links = FairShareLinks::new();
    let service_s = 0.081;
    for ticket in 0..32 {
        links.start(&plan(&[(&[2], service_s)]), 0.0, ticket);
    }
    links.advance_to(100.0);
    let completed = take(&mut links);
    assert_eq!(completed.len(), 32);
    for (_, done) in &completed {
        assert!(close(done.finish_s, 32.0 * service_s));
    }
    assert_eq!(completed[0].0, 0);
}

#[test]
fn starts_before_the_network_clock_are_clamped_forward() {
    let mut links = FairShareLinks::new();
    links.start(&plan(&[(&[0], 0.1)]), 1.0, 0);
    links.advance_to(1.05);
    links.start(&plan(&[(&[1], 0.1)]), 0.5, 1);
    links.advance_to(10.0);
    let completed = take(&mut links);
    assert!(close(window(&completed, 1).start_s, 1.05));
    assert!(close(window(&completed, 1).finish_s, 1.15));
}

fn network_windows(sharing: KvLinkSharing) -> Vec<(usize, TransferWindow)> {
    let mut network = KvNetwork::new(sharing);
    network.start(&plan(&[(&[0], 0.1)]), 0.0, 0);
    network.start(&plan(&[(&[0], 0.1)]), 0.0, 1);
    network.start(&plan(&[(&[1], 0.1)]), 0.02, 2);
    let mut completed = Vec::new();
    network.advance_to(0.1);
    completed.extend(network.take_completed());
    completed.extend(network.drain());
    completed
}

#[test]
fn fifo_network_reproduces_link_queue_reservations() {
    let completed = network_windows(KvLinkSharing::Fifo);
    let mut queues = LinkQueues::new(0);
    let expected = [
        queues.reserve(&plan(&[(&[0], 0.1)]), 0.0, 0),
        queues.reserve(&plan(&[(&[0], 0.1)]), 0.0, 1),
        queues.reserve(&plan(&[(&[1], 0.1)]), 0.02, 2),
    ];
    assert_eq!(completed.len(), 3);
    for (ticket, window) in &completed {
        assert_eq!(window, &expected[*ticket]);
    }
    // Completion order: ticket 0 at 0.1, ticket 2 at 0.12, ticket 1 at 0.2.
    let order = completed
        .iter()
        .map(|(ticket, _)| *ticket)
        .collect::<Vec<_>>();
    assert_eq!(order, vec![0, 2, 1]);
}

#[test]
fn fair_share_network_overlaps_what_fifo_serializes() {
    let completed = network_windows(KvLinkSharing::FairShare);
    let finish = |ticket: usize| window(&completed, ticket).finish_s;
    assert!(close(finish(0), 0.2));
    assert!(close(finish(1), 0.2));
    assert!(close(finish(2), 0.12));
}

#[test]
fn network_reports_projected_and_next_finishes() {
    for sharing in [KvLinkSharing::Fifo, KvLinkSharing::FairShare] {
        let mut network = KvNetwork::new(sharing);
        network.start(&plan(&[(&[0], 0.1)]), 0.0, 4);
        assert!(close(network.finish_s(4).expect("finish"), 0.1));
        assert!(close(network.next_finish_after(0.0).expect("next"), 0.1));
        assert_eq!(network.next_finish_after(0.1), None);
        assert_eq!(network.finish_s(5), None);
        network.advance_to(0.1);
        assert_eq!(network.take_completed().len(), 1);
        assert!(network.drain().is_empty());
    }
}
