//! KV and sequence capacity ledger shared by every worker of one engine run.
//!
//! Admission reserves a request's whole footprint; completion and
//! cancellation return it. Releases are timestamped and applied lazily, so a
//! worker planning a step at time `t` never sees capacity another worker frees
//! after `t`.
//!
//! Holdings are keyed by `HoldingKey`: a colocated request holds one
//! footprint (its primary key); a disaggregated request holds its prefill
//! footprint under the primary key and its decode footprint under the decode
//! key, both at once while its KV is being pulled.

use std::{cmp::Reverse, collections::BinaryHeap};

use super::types::{
    CapacityExcess, CapacityLimits, CapacityResource, CapacityScope, ClassId, EngineLimits,
    EngineRequest, EngineRequestId, KvFootprint, WorkerId,
};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Usage {
    sequences: u64,
    tokens: u64,
    blocks: u64,
}

impl Usage {
    fn add(&mut self, footprint: KvFootprint) {
        self.sequences += footprint.sequences;
        self.tokens += footprint.tokens;
        self.blocks += footprint.blocks;
    }

    fn remove(&mut self, footprint: KvFootprint) {
        self.sequences = self.sequences.saturating_sub(footprint.sequences);
        self.tokens = self.tokens.saturating_sub(footprint.tokens);
        self.blocks = self.blocks.saturating_sub(footprint.blocks);
    }
}

/// Total order on finite simulation times for the release heap.
#[derive(Copy, Clone, Debug, PartialEq)]
struct TimeKey(f64);

impl Eq for TimeKey {}

impl PartialOrd for TimeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Which of a request's holdings a ledger entry is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::serving) enum HoldingKey {
    /// The colocated footprint, or a disaggregated request's prefill-side KV.
    Primary(EngineRequestId),
    /// A disaggregated request's decode-side KV.
    Decode(EngineRequestId),
}

impl HoldingKey {
    fn slot(self, request_count: usize) -> usize {
        match self {
            Self::Primary(id) => id,
            Self::Decode(id) => request_count + id,
        }
    }
}

/// What a timestamped release returns.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ReleaseKind {
    /// Everything the holding still has.
    All,
    /// Only its running sequences (a finished prefill whose KV stays until
    /// the decode worker has pulled it).
    Sequences,
}

pub(in crate::serving) struct CapacityLedger {
    global_limits: CapacityLimits,
    worker_limits: Vec<CapacityLimits>,
    class_limits: Vec<CapacityLimits>,
    global: Usage,
    workers: Vec<Usage>,
    classes: Vec<Usage>,
    releases: BinaryHeap<Reverse<(TimeKey, HoldingKey, ReleaseKind)>>,
    held: Vec<Option<(WorkerId, Option<ClassId>, KvFootprint)>>,
    request_count: usize,
}

impl CapacityLedger {
    pub(in crate::serving) fn new(limits: &EngineLimits, request_count: usize) -> Self {
        Self {
            global_limits: limits.global,
            worker_limits: limits
                .workers
                .iter()
                .map(|worker| worker.capacity)
                .collect(),
            class_limits: limits.classes.iter().map(|class| class.capacity).collect(),
            global: Usage::default(),
            workers: vec![Usage::default(); limits.workers.len()],
            classes: vec![Usage::default(); limits.classes.len()],
            releases: BinaryHeap::new(),
            held: vec![None; request_count * 2],
            request_count,
        }
    }

    /// The first limit this request exceeds even with the engine empty.
    pub(in crate::serving) fn never_fits(&self, request: &EngineRequest) -> Option<CapacityExcess> {
        self.never_fits_on(request.worker, request.class, request.footprint)
    }

    /// The first limit `footprint` exceeds on `worker` with the engine empty.
    pub(in crate::serving) fn never_fits_on(
        &self,
        worker: WorkerId,
        class: Option<ClassId>,
        footprint: KvFootprint,
    ) -> Option<CapacityExcess> {
        let empty = Usage::default();
        excess(CapacityScope::Global, self.global_limits, empty, footprint)
            .or_else(|| {
                self.worker_limits
                    .get(worker)
                    .and_then(|limits| excess(CapacityScope::Worker, *limits, empty, footprint))
            })
            .or_else(|| {
                let class = class?;
                let limits = self.class_limits.get(class)?;
                excess(CapacityScope::Class(class), *limits, empty, footprint)
            })
    }

    /// Whether the request fits next to everything admitted so far.
    pub(in crate::serving) fn fits(&self, request: &EngineRequest) -> bool {
        self.fits_on(request.worker, request.class, request.footprint)
    }

    /// Whether `footprint` fits on `worker` next to everything admitted so far.
    pub(in crate::serving) fn fits_on(
        &self,
        worker: WorkerId,
        class: Option<ClassId>,
        footprint: KvFootprint,
    ) -> bool {
        if excess(
            CapacityScope::Global,
            self.global_limits,
            self.global,
            footprint,
        )
        .is_some()
        {
            return false;
        }
        if let (Some(limits), Some(usage)) =
            (self.worker_limits.get(worker), self.workers.get(worker))
            && excess(CapacityScope::Worker, *limits, *usage, footprint).is_some()
        {
            return false;
        }
        if let Some(class) = class
            && let (Some(limits), Some(usage)) =
                (self.class_limits.get(class), self.classes.get(class))
            && excess(CapacityScope::Class(class), *limits, *usage, footprint).is_some()
        {
            return false;
        }
        true
    }

    pub(in crate::serving) fn allocate(&mut self, id: EngineRequestId, request: &EngineRequest) {
        self.allocate_on(
            HoldingKey::Primary(id),
            request.worker,
            request.class,
            request.footprint,
        );
    }

    pub(in crate::serving) fn allocate_on(
        &mut self,
        key: HoldingKey,
        worker: WorkerId,
        class: Option<ClassId>,
        footprint: KvFootprint,
    ) {
        self.global.add(footprint);
        if let Some(usage) = self.workers.get_mut(worker) {
            usage.add(footprint);
        }
        if let Some(class) = class
            && let Some(usage) = self.classes.get_mut(class)
        {
            usage.add(footprint);
        }
        if let Some(slot) = self.held.get_mut(key.slot(self.request_count)) {
            *slot = Some((worker, class, footprint));
        }
    }

    /// Returns the holding's whole remaining footprint at `at_s`.
    pub(in crate::serving) fn release_at(&mut self, key: HoldingKey, at_s: f64) {
        self.releases
            .push(Reverse((TimeKey(at_s), key, ReleaseKind::All)));
    }

    /// Returns only the holding's sequences at `at_s`; its tokens and blocks
    /// stay until a later `release_at`.
    pub(in crate::serving) fn release_sequences_at(&mut self, key: HoldingKey, at_s: f64) {
        self.releases
            .push(Reverse((TimeKey(at_s), key, ReleaseKind::Sequences)));
    }

    /// Applies every release scheduled at or before `now_s`.
    pub(in crate::serving) fn apply_releases_until(&mut self, now_s: f64) {
        while let Some(Reverse((TimeKey(at_s), key, kind))) = self.releases.peek().copied() {
            if at_s > now_s + 1e-12 {
                break;
            }
            self.releases.pop();
            self.release_now(key, kind);
        }
    }

    /// Earliest scheduled release strictly after `now_s`.
    pub(in crate::serving) fn next_release_after(&self, now_s: f64) -> Option<f64> {
        self.releases
            .iter()
            .map(|Reverse((TimeKey(at_s), _, _))| *at_s)
            .filter(|at_s| *at_s > now_s + 1e-12)
            .min_by(f64::total_cmp)
    }

    pub(in crate::serving) fn worker_sequences(&self, worker: WorkerId) -> u64 {
        self.workers.get(worker).map_or(0, |usage| usage.sequences)
    }

    fn release_now(&mut self, key: HoldingKey, kind: ReleaseKind) {
        let slot = key.slot(self.request_count);
        let footprint = match kind {
            ReleaseKind::All => self.held.get_mut(slot).and_then(Option::take),
            ReleaseKind::Sequences => {
                self.held
                    .get_mut(slot)
                    .and_then(Option::as_mut)
                    .map(|(worker, class, held)| {
                        let released = KvFootprint {
                            sequences: held.sequences,
                            tokens: 0,
                            blocks: 0,
                        };
                        held.sequences = 0;
                        (*worker, *class, released)
                    })
            }
        };
        let Some((worker, class, footprint)) = footprint else {
            return;
        };
        self.global.remove(footprint);
        if let Some(usage) = self.workers.get_mut(worker) {
            usage.remove(footprint);
        }
        if let Some(class) = class
            && let Some(usage) = self.classes.get_mut(class)
        {
            usage.remove(footprint);
        }
    }
}

fn excess(
    scope: CapacityScope,
    limits: CapacityLimits,
    usage: Usage,
    footprint: KvFootprint,
) -> Option<CapacityExcess> {
    [
        (
            CapacityResource::Sequences,
            limits.sequences,
            usage.sequences,
            footprint.sequences,
        ),
        (
            CapacityResource::Tokens,
            limits.tokens,
            usage.tokens,
            footprint.tokens,
        ),
        (
            CapacityResource::Blocks,
            limits.blocks,
            usage.blocks,
            footprint.blocks,
        ),
    ]
    .into_iter()
    .find_map(|(resource, limit, used, extra)| {
        let limit = limit?;
        let needed = used.saturating_add(extra);
        (needed > limit).then_some(CapacityExcess {
            scope,
            resource,
            needed,
            limit,
        })
    })
}
