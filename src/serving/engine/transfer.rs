//! KV-cache transfer model for disaggregated serving.
//!
//! A request's prompt KV moves from the prefill worker's GPUs to the decode
//! worker's GPUs as a set of point-to-point flows, one per (source GPU,
//! destination GPU) pair whose KV shards overlap (`shards`). Each flow is
//! priced once, uncontended, from a measured `send_recv` curve when the
//! cluster has one for that directed node pair and from the routed
//! alpha-beta path otherwise (`plan`).
//!
//! Contention (`LinkQueues`): every directed route resource (one direction of
//! a NIC-to-NIC link, a GPU-to-NIC hop, an intra-node fabric hop) is a FIFO
//! queue. A flow starts when its transfer is ready and every resource on its
//! route is free, then holds all of them for its uncontended service time.
//! Concurrent transfers over a shared link therefore serialize, which keeps
//! the link's aggregate throughput equal to its bandwidth; per-transfer
//! latency is FIFO rather than fair-share (`kv_transfer_fifo_link_queues`).

mod plan;
mod shards;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(in crate::serving) use plan::FlowPricing;
pub(in crate::serving) use plan::{
    KvPlanContext, KvPlanError, KvTransferPlanner, PlacedKvLayout, PlannedTransfer,
};
pub(in crate::serving) use shards::{KvShardLayout, kv_shard_flows};

/// Index of one directed route resource inside one engine run.
pub(in crate::serving) type LinkId = usize;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(in crate::serving) enum TransferError {
    /// A flow's service time is negative, NaN, or infinite.
    InvalidServiceTime { service_s: f64 },
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidServiceTime { service_s } => {
                write!(formatter, "KV flow service time {service_s} is not finite and non-negative")
            }
        }
    }
}

impl std::error::Error for TransferError {}

/// One point-to-point piece of a KV transfer: the directed route resources it
/// occupies and its uncontended duration.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct KvFlow {
    resources: Vec<LinkId>,
    service_s: f64,
}

impl KvFlow {
    pub(in crate::serving) fn new(
        mut resources: Vec<LinkId>,
        service_s: f64,
    ) -> Result<Self, TransferError> {
        if !service_s.is_finite() || service_s < 0.0 {
            return Err(TransferError::InvalidServiceTime { service_s });
        }
        resources.sort_unstable();
        resources.dedup();
        Ok(Self {
            resources,
            service_s,
        })
    }

    #[cfg(test)]
    pub(in crate::serving) fn resources(&self) -> &[LinkId] {
        &self.resources
    }

    #[cfg(test)]
    pub(in crate::serving) fn service_s(&self) -> f64 {
        self.service_s
    }

    fn scaled(&self, factor: f64) -> Self {
        Self {
            resources: self.resources.clone(),
            service_s: self.service_s * factor,
        }
    }
}

/// Every flow one request's KV transfer needs. No flows means nothing moves
/// (the transfer completes the instant it is ready).
#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::serving) struct KvTransferPlan {
    flows: Vec<KvFlow>,
    bytes: u64,
}

impl KvTransferPlan {
    pub(in crate::serving) fn new(flows: Vec<KvFlow>, bytes: u64) -> Self {
        Self { flows, bytes }
    }

    #[cfg(test)]
    pub(in crate::serving) fn flows(&self) -> &[KvFlow] {
        &self.flows
    }

    pub(in crate::serving) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Duration with the network otherwise idle: flows sharing a resource
    /// serialize, disjoint flows overlap.
    pub(in crate::serving) fn uncontended_s(&self) -> f64 {
        let mut queues = LinkQueues::new(self.link_count());
        queues.reserve(self, 0.0, 0).finish_s
    }

    /// The same plan with every flow's service time multiplied by `factor`
    /// (used to apply a calibration-profile KV-transfer fit).
    pub(in crate::serving) fn scaled(&self, factor: f64) -> Self {
        if !factor.is_finite() || factor < 0.0 {
            return self.clone();
        }
        Self {
            flows: self.flows.iter().map(|flow| flow.scaled(factor)).collect(),
            bytes: self.bytes,
        }
    }

    pub(in crate::serving) fn link_count(&self) -> usize {
        self.flows
            .iter()
            .flat_map(|flow| flow.resources.iter().copied())
            .max()
            .map_or(0, |max| max + 1)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(in crate::serving) struct FlowWindow {
    pub(in crate::serving) start_s: f64,
    pub(in crate::serving) finish_s: f64,
}

/// When one reserved transfer ran.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct TransferWindow {
    /// When the transfer could have started on an idle network.
    pub(in crate::serving) ready_s: f64,
    pub(in crate::serving) start_s: f64,
    pub(in crate::serving) finish_s: f64,
    pub(in crate::serving) flows: Vec<FlowWindow>,
    /// Earlier transfers (by ticket) that held a resource this one waited for
    /// or followed on.
    pub(in crate::serving) predecessors: Vec<usize>,
}

/// FIFO occupancy of every directed route resource.
#[derive(Clone, Debug, Default)]
pub(in crate::serving) struct LinkQueues {
    free_at_s: Vec<f64>,
    last_ticket: Vec<Option<usize>>,
}

impl LinkQueues {
    pub(in crate::serving) fn new(links: usize) -> Self {
        Self {
            free_at_s: vec![f64::NEG_INFINITY; links],
            last_ticket: vec![None; links],
        }
    }

    /// Reserves every flow of `plan` in order, starting no earlier than
    /// `ready_s`. Callers reserve in non-decreasing `ready_s` order, so the
    /// queues are FIFO in readiness. `ticket` names this transfer in later
    /// transfers' `predecessors`.
    pub(in crate::serving) fn reserve(
        &mut self,
        plan: &KvTransferPlan,
        ready_s: f64,
        ticket: usize,
    ) -> TransferWindow {
        let needed = plan.link_count();
        if needed > self.free_at_s.len() {
            self.free_at_s.resize(needed, f64::NEG_INFINITY);
            self.last_ticket.resize(needed, None);
        }
        let mut flows = Vec::with_capacity(plan.flows.len());
        let mut predecessors = Vec::new();
        for flow in &plan.flows {
            let start_s = flow
                .resources
                .iter()
                .map(|link| self.free_at_s[*link])
                .fold(ready_s, f64::max);
            let finish_s = start_s + flow.service_s;
            for link in &flow.resources {
                if let Some(previous) = self.last_ticket[*link]
                    && previous != ticket
                    && !predecessors.contains(&previous)
                {
                    predecessors.push(previous);
                }
                self.free_at_s[*link] = finish_s;
                self.last_ticket[*link] = Some(ticket);
            }
            flows.push(FlowWindow { start_s, finish_s });
        }
        let start_s = flows
            .iter()
            .map(|flow| flow.start_s)
            .min_by(f64::total_cmp)
            .unwrap_or(ready_s);
        let finish_s = flows
            .iter()
            .map(|flow| flow.finish_s)
            .max_by(f64::total_cmp)
            .unwrap_or(ready_s);
        predecessors.sort_unstable();
        TransferWindow {
            ready_s,
            start_s,
            finish_s: finish_s.max(start_s),
            flows,
            predecessors,
        }
    }

    /// Whether a resource is busy at `at_s` (for tests and diagnostics).
    #[cfg(test)]
    pub(in crate::serving) fn busy_at(&self, link: LinkId, at_s: f64) -> bool {
        self.free_at_s
            .get(link)
            .is_some_and(|free_at_s| *free_at_s > at_s + 1e-12)
    }
}
