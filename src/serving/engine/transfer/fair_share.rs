//! Fair-share (processor-sharing) progress of concurrent KV flows.
//!
//! Every directed route resource has a capacity of one "standalone flow":
//! a flow running alone progresses at rate 1 (one second of its uncontended
//! service time per second) and occupies each resource on its route fully.
//! With several active flows, rates are the max-min fair allocation
//! (progressive filling) under two constraints: the rates of the flows
//! crossing a resource sum to at most 1, and no flow exceeds its standalone
//! rate 1. Rates change only when a flow starts or finishes, so between those
//! events every flow's remaining work decreases linearly.
//!
//! Normalizing capacity by each flow's own standalone price (rather than
//! dividing a nominal link bandwidth) is what keeps the measured pricing
//! exact: a flow alone finishes after exactly its curve or alpha-beta price,
//! and a saturated resource moves work at the same aggregate rate as the
//! FIFO model (measured NIXL pulls: aggregate throughput stays at the
//! standalone ~300 MB/s while per-transfer time grows with concurrency).
//!
//! State is updated lazily: the remaining work of every flow is stored at a
//! common epoch (the last start or finish) and only rebased when rates
//! change, so a flow that runs alone finishes at `start + service` bit for
//! bit. Projections (when each active transfer would finish if nothing else
//! started) run the same completion code on a clone, so a projected finish
//! equals the finish later recorded unless a new transfer starts first.

use super::{FlowWindow, KvTransferPlan, LinkId, TransferWindow};

/// Flows finishing within this many seconds of the earliest completion
/// finish together.
const COMPLETION_EPSILON_S: f64 = 1e-12;
/// Contention below this is rounding, not sharing.
const CONTENTION_EPSILON_S: f64 = 1e-12;

#[derive(Clone, Debug)]
struct ActiveFlow {
    ticket: usize,
    flow_index: usize,
    resources: Vec<LinkId>,
    /// Uncontended seconds of work left at `FairShareLinks::epoch_s`.
    remaining_s: f64,
    /// Fraction of standalone speed, in (0, 1].
    rate: f64,
}

impl ActiveFlow {
    fn finish_s(&self, epoch_s: f64) -> f64 {
        epoch_s + self.remaining_s / self.rate
    }
}

#[derive(Clone, Debug)]
struct ActiveTransfer {
    ticket: usize,
    start_s: f64,
    uncontended_s: f64,
    resources: Vec<LinkId>,
    flows: Vec<FlowWindow>,
    open_flows: usize,
    predecessors: Vec<usize>,
}

impl ActiveTransfer {
    fn window(self) -> TransferWindow {
        let finish_s = self
            .flows
            .iter()
            .map(|flow| flow.finish_s)
            .fold(self.start_s, f64::max);
        let contention_s = finish_s - self.start_s - self.uncontended_s;
        TransferWindow {
            ready_s: self.start_s,
            start_s: self.start_s,
            finish_s,
            flows: self.flows,
            predecessors: self.predecessors,
            contention_s: if contention_s > CONTENTION_EPSILON_S {
                contention_s
            } else {
                0.0
            },
        }
    }
}

/// Active flows on shared directed resources, progressing at max-min fair
/// rates.
#[derive(Clone, Debug, Default)]
pub(in crate::serving) struct FairShareLinks {
    /// Latest time the network has been advanced to.
    clock_s: f64,
    /// Time at which every flow's `remaining_s` is stated.
    epoch_s: f64,
    flows: Vec<ActiveFlow>,
    transfers: Vec<ActiveTransfer>,
    completed: Vec<(usize, TransferWindow)>,
    /// Projected finish of every active transfer, in start order.
    projection: Vec<(usize, f64)>,
}

impl FairShareLinks {
    pub(in crate::serving) fn new() -> Self {
        Self::default()
    }

    /// Starts every flow of `plan` at `at_s` (clamped to the network's
    /// clock, which only moves forward). Transfers already sharing a resource
    /// with it become its predecessors.
    pub(in crate::serving) fn start(&mut self, plan: &KvTransferPlan, at_s: f64, ticket: usize) {
        self.advance_to(at_s);
        let at_s = at_s.max(self.clock_s).max(self.epoch_s);
        self.rebase(at_s);
        let mut resources = plan
            .flows
            .iter()
            .flat_map(|flow| flow.resources.iter().copied())
            .collect::<Vec<_>>();
        resources.sort_unstable();
        resources.dedup();
        let mut predecessors = self
            .transfers
            .iter()
            .filter(|transfer| shares_a_resource(&transfer.resources, &resources))
            .map(|transfer| transfer.ticket)
            .filter(|other| *other != ticket)
            .collect::<Vec<_>>();
        predecessors.sort_unstable();
        predecessors.dedup();
        let mut transfer = ActiveTransfer {
            ticket,
            start_s: at_s,
            uncontended_s: plan.uncontended_s(),
            resources,
            flows: Vec::with_capacity(plan.flows.len()),
            open_flows: 0,
            predecessors,
        };
        for (flow_index, flow) in plan.flows.iter().enumerate() {
            transfer.flows.push(FlowWindow {
                start_s: at_s,
                finish_s: at_s,
            });
            if flow.service_s > 0.0 {
                transfer.open_flows += 1;
                self.flows.push(ActiveFlow {
                    ticket,
                    flow_index,
                    resources: flow.resources.clone(),
                    remaining_s: flow.service_s,
                    rate: 1.0,
                });
            }
        }
        if transfer.open_flows == 0 {
            self.completed.push((ticket, transfer.window()));
        } else {
            self.transfers.push(transfer);
        }
        self.reallocate();
        self.reproject();
    }

    /// Completes every flow finishing at or before `until_s`, in time order,
    /// recomputing rates after each completion.
    pub(in crate::serving) fn advance_to(&mut self, until_s: f64) {
        if until_s.is_finite() {
            self.clock_s = self.clock_s.max(until_s);
        }
        let mut changed = false;
        while let Some(next_s) = self.next_flow_finish_s() {
            if next_s > until_s + COMPLETION_EPSILON_S {
                break;
            }
            self.complete_flows_at(next_s);
            changed = true;
        }
        if changed {
            self.reproject();
        }
    }

    /// Transfers that completed since the last call, by (finish, ticket).
    pub(in crate::serving) fn take_completed(&mut self) -> Vec<(usize, TransferWindow)> {
        let mut completed = std::mem::take(&mut self.completed);
        completed.sort_by(|left, right| {
            left.1
                .finish_s
                .total_cmp(&right.1.finish_s)
                .then(left.0.cmp(&right.0))
        });
        completed
    }

    /// When the transfer `ticket` finishes (or finished, if not yet taken)
    /// assuming no other transfer starts in the meantime.
    pub(in crate::serving) fn finish_s(&self, ticket: usize) -> Option<f64> {
        self.projection
            .iter()
            .find(|(projected, _)| *projected == ticket)
            .map(|(_, finish_s)| *finish_s)
            .or_else(|| {
                self.completed
                    .iter()
                    .find(|(done, _)| *done == ticket)
                    .map(|(_, window)| window.finish_s)
            })
    }

    /// The earliest projected transfer finish strictly after `after_s`.
    pub(in crate::serving) fn next_finish_after(&self, after_s: f64) -> Option<f64> {
        self.projection
            .iter()
            .map(|(_, finish_s)| *finish_s)
            .filter(|finish_s| *finish_s > after_s + COMPLETION_EPSILON_S)
            .min_by(f64::total_cmp)
    }

    /// Current rate of every active flow, by (ticket, flow index), for tests.
    #[cfg(test)]
    pub(in crate::serving) fn rates(&self) -> Vec<(usize, usize, f64)> {
        self.flows
            .iter()
            .map(|flow| (flow.ticket, flow.flow_index, flow.rate))
            .collect()
    }

    fn next_flow_finish_s(&self) -> Option<f64> {
        self.flows
            .iter()
            .map(|flow| flow.finish_s(self.epoch_s))
            .min_by(f64::total_cmp)
    }

    /// Finishes the flows that end at `at_s` (within epsilon), rebases the
    /// rest to `at_s`, and reallocates rates.
    fn complete_flows_at(&mut self, at_s: f64) {
        let epoch_s = self.epoch_s;
        let mut kept = Vec::with_capacity(self.flows.len());
        for flow in std::mem::take(&mut self.flows) {
            let finish_s = flow.finish_s(epoch_s);
            if finish_s > at_s + COMPLETION_EPSILON_S {
                kept.push(flow);
                continue;
            }
            let Some(slot) = self
                .transfers
                .iter()
                .position(|transfer| transfer.ticket == flow.ticket)
            else {
                continue;
            };
            let transfer = &mut self.transfers[slot];
            if let Some(window) = transfer.flows.get_mut(flow.flow_index) {
                window.finish_s = finish_s;
            }
            transfer.open_flows = transfer.open_flows.saturating_sub(1);
            if transfer.open_flows == 0 {
                let done = self.transfers.remove(slot);
                self.completed.push((done.ticket, done.window()));
            }
        }
        self.flows = kept;
        self.rebase(at_s);
        self.reallocate();
    }

    /// Moves the epoch to `at_s`, charging every flow for the work done at
    /// its current rate since the previous epoch.
    fn rebase(&mut self, at_s: f64) {
        let elapsed_s = at_s - self.epoch_s;
        if elapsed_s > 0.0 {
            for flow in &mut self.flows {
                flow.remaining_s = (flow.remaining_s - flow.rate * elapsed_s).max(0.0);
            }
        }
        self.epoch_s = self.epoch_s.max(at_s);
    }

    fn reallocate(&mut self) {
        let paths = self
            .flows
            .iter()
            .map(|flow| flow.resources.as_slice())
            .collect::<Vec<_>>();
        let rates = max_min_rates(&paths);
        for (flow, rate) in self.flows.iter_mut().zip(rates) {
            flow.rate = rate;
        }
    }

    fn reproject(&mut self) {
        let mut future = Self {
            clock_s: self.clock_s,
            epoch_s: self.epoch_s,
            flows: self.flows.clone(),
            transfers: self.transfers.clone(),
            completed: Vec::new(),
            projection: Vec::new(),
        };
        while let Some(next_s) = future.next_flow_finish_s() {
            future.complete_flows_at(next_s);
        }
        self.projection = self
            .transfers
            .iter()
            .filter_map(|transfer| {
                future
                    .completed
                    .iter()
                    .find(|(ticket, _)| *ticket == transfer.ticket)
                    .map(|(ticket, window)| (*ticket, window.finish_s))
            })
            .collect();
    }
}

fn shares_a_resource(left: &[LinkId], right: &[LinkId]) -> bool {
    // Both sorted and deduplicated.
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// Max-min fair rates (progressive filling) of flows crossing `paths`, with
/// every resource's capacity and every flow's own cap equal to 1. All
/// unfrozen flows rise together; a flow freezes when a resource it crosses
/// saturates or it reaches its standalone rate. Deterministic: the result
/// depends only on the paths, in order.
pub(in crate::serving) fn max_min_rates(paths: &[&[LinkId]]) -> Vec<f64> {
    let links = paths
        .iter()
        .flat_map(|path| path.iter().copied())
        .max()
        .map_or(0, |max| max + 1);
    let mut capacity = vec![1.0_f64; links];
    let mut saturated = vec![false; links];
    let mut rates = vec![0.0_f64; paths.len()];
    let mut frozen = vec![false; paths.len()];
    loop {
        let mut users = vec![0_usize; links];
        let mut open = 0_usize;
        for (index, path) in paths.iter().enumerate() {
            if frozen[index] {
                continue;
            }
            open += 1;
            for link in *path {
                users[*link] += 1;
            }
        }
        if open == 0 {
            break;
        }
        let headroom = (0..paths.len())
            .filter(|index| !frozen[*index])
            .map(|index| 1.0 - rates[index])
            .fold(f64::INFINITY, f64::min);
        let link_share = (0..links)
            .filter(|link| users[*link] > 0)
            .map(|link| capacity[link] / users[link] as f64)
            .fold(f64::INFINITY, f64::min);
        let step = headroom.min(link_share).max(0.0);
        for link in 0..links {
            if users[link] == 0 {
                continue;
            }
            if capacity[link] / users[link] as f64 <= step * (1.0 + 1e-12) {
                saturated[link] = true;
                capacity[link] = 0.0;
            } else {
                capacity[link] -= step * users[link] as f64;
            }
        }
        let caps_bind = headroom <= link_share;
        for (index, path) in paths.iter().enumerate() {
            if frozen[index] {
                continue;
            }
            rates[index] = (rates[index] + step).min(1.0);
            if (caps_bind && rates[index] >= 1.0 - 1e-12)
                || path.iter().any(|link| saturated[*link])
            {
                frozen[index] = true;
            }
        }
        if caps_bind {
            for (index, rate) in rates.iter_mut().enumerate() {
                if !frozen[index] && *rate >= 1.0 - 1e-12 {
                    *rate = 1.0;
                    frozen[index] = true;
                }
            }
        }
    }
    rates
}
