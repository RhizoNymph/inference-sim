//! One engine run's KV-transfer network under either sharing discipline,
//! behind the event-driven interface the engine loop uses: start a transfer
//! at the current time, advance the clock, take the transfers that finished,
//! and ask when a transfer (or the next one) will finish if nothing else
//! starts.

use crate::calibration::KvLinkSharing;

use super::fair_share::FairShareLinks;
use super::{KvTransferPlan, LinkQueues, TransferWindow};

/// Transfers within this many seconds of the clock count as finished.
const FINISH_EPSILON_S: f64 = 1e-12;

/// FIFO link queues: a transfer's window is fixed when it starts.
#[derive(Clone, Debug, Default)]
pub(in crate::serving) struct FifoLinks {
    queues: LinkQueues,
    in_flight: Vec<(usize, TransferWindow)>,
    completed: Vec<(usize, TransferWindow)>,
}

impl FifoLinks {
    fn start(&mut self, plan: &KvTransferPlan, at_s: f64, ticket: usize) {
        let window = self.queues.reserve(plan, at_s, ticket);
        self.in_flight.push((ticket, window));
    }

    fn advance_to(&mut self, until_s: f64) {
        let (done, pending): (Vec<_>, Vec<_>) = std::mem::take(&mut self.in_flight)
            .into_iter()
            .partition(|(_, window)| window.finish_s <= until_s + FINISH_EPSILON_S);
        self.in_flight = pending;
        self.completed.extend(done);
    }

    fn finish_s(&self, ticket: usize) -> Option<f64> {
        self.in_flight
            .iter()
            .chain(&self.completed)
            .find(|(started, _)| *started == ticket)
            .map(|(_, window)| window.finish_s)
    }

    fn next_finish_after(&self, after_s: f64) -> Option<f64> {
        self.in_flight
            .iter()
            .map(|(_, window)| window.finish_s)
            .filter(|finish_s| *finish_s > after_s + FINISH_EPSILON_S)
            .min_by(f64::total_cmp)
    }

    fn take_completed(&mut self) -> Vec<(usize, TransferWindow)> {
        let mut completed = std::mem::take(&mut self.completed);
        completed.sort_by(|left, right| {
            left.1
                .finish_s
                .total_cmp(&right.1.finish_s)
                .then(left.0.cmp(&right.0))
        });
        completed
    }
}

/// The directed route resources of one engine run and the transfers on them.
/// Tickets name transfers (the engine uses request ids). Callers start
/// transfers in non-decreasing time order and advance the clock
/// monotonically.
#[derive(Clone, Debug)]
pub(in crate::serving) enum KvNetwork {
    Fifo(FifoLinks),
    FairShare(FairShareLinks),
}

impl KvNetwork {
    pub(in crate::serving) fn new(sharing: KvLinkSharing) -> Self {
        match sharing {
            KvLinkSharing::Fifo => Self::Fifo(FifoLinks::default()),
            KvLinkSharing::FairShare => Self::FairShare(FairShareLinks::new()),
        }
    }

    /// Starts `plan` at `at_s`.
    pub(in crate::serving) fn start(&mut self, plan: &KvTransferPlan, at_s: f64, ticket: usize) {
        match self {
            Self::Fifo(links) => links.start(plan, at_s, ticket),
            Self::FairShare(links) => links.start(plan, at_s, ticket),
        }
    }

    /// Completes every transfer finishing at or before `until_s`.
    pub(in crate::serving) fn advance_to(&mut self, until_s: f64) {
        match self {
            Self::Fifo(links) => links.advance_to(until_s),
            Self::FairShare(links) => links.advance_to(until_s),
        }
    }

    /// Transfers completed since the last call, by (finish, ticket).
    pub(in crate::serving) fn take_completed(&mut self) -> Vec<(usize, TransferWindow)> {
        match self {
            Self::Fifo(links) => links.take_completed(),
            Self::FairShare(links) => links.take_completed(),
        }
    }

    /// Completes and returns every remaining transfer.
    pub(in crate::serving) fn drain(&mut self) -> Vec<(usize, TransferWindow)> {
        self.advance_to(f64::INFINITY);
        self.take_completed()
    }

    /// When transfer `ticket` finishes if no other transfer starts first;
    /// `None` for unknown or already taken tickets.
    pub(in crate::serving) fn finish_s(&self, ticket: usize) -> Option<f64> {
        match self {
            Self::Fifo(links) => links.finish_s(ticket),
            Self::FairShare(links) => links.finish_s(ticket),
        }
    }

    /// The earliest transfer finish strictly after `after_s`.
    pub(in crate::serving) fn next_finish_after(&self, after_s: f64) -> Option<f64> {
        match self {
            Self::Fifo(links) => links.next_finish_after(after_s),
            Self::FairShare(links) => links.next_finish_after(after_s),
        }
    }
}
