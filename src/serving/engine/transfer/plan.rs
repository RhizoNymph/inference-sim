//! Builds priced KV transfer plans from the cluster topology.
//!
//! For each shard flow (`shards::kv_shard_flows`) between two distinct GPUs:
//!
//! - bytes = flow fraction x the request's full prompt KV
//!   (`sequences x prompt_tokens x layers x kv_heads x head_dim x 2 x kv
//!   dtype bytes`);
//! - route = the directed topology route between the two GPUs
//!   (`kv_transfer_paths`, which honours per-direction NIC caps and link
//!   overrides);
//! - service time = a measured `send_recv` curve for the directed node pair
//!   when the cluster has one (no calibration scalars, like every
//!   curve-priced collective), otherwise alpha-beta on the route:
//!   `latency x collective_latency_scale + bytes / bottleneck bandwidth x
//!   kv_transfer_scale / collective_bandwidth_scale`;
//! - resources = one directed queue per route resource.
//!
//! A calibration-profile `kv_transfer` fit, when present, rescales every
//! flow so the plan's uncontended duration equals the fitted time.

use std::collections::{BTreeMap, HashMap};

use super::super::*;
use super::{KvFlow, KvShardLayout, KvTransferPlan, LinkId, TransferError, kv_shard_flows};
use crate::types::collective::{CollectiveCost, CollectiveKind, CollectivePricing, CurveCoverage};
use crate::types::collective_curves::{CurveLookup, CurveQuery};

#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) enum KvPlanError {
    /// A placed layout lists a different number of GPUs than it has ranks.
    RankCountMismatch { expected: usize, actual: usize },
    /// No topology route connects the two GPUs.
    Unroutable { source: GpuAddr, destination: GpuAddr },
    Flow(TransferError),
}

impl std::fmt::Display for KvPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RankCountMismatch { expected, actual } => write!(
                formatter,
                "KV layout has {expected} ranks but {actual} placed GPUs"
            ),
            Self::Unroutable {
                source,
                destination,
            } => write!(
                formatter,
                "no KV transfer route from node {} gpu {} to node {} gpu {}",
                source.node_id, source.local_gpu_id, destination.node_id, destination.local_gpu_id
            ),
            Self::Flow(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for KvPlanError {}

/// A KV layout bound to the GPUs of one routed replica, in rank order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::serving) struct PlacedKvLayout {
    layout: KvShardLayout,
    rank_gpus: Vec<GpuAddr>,
}

impl PlacedKvLayout {
    pub(in crate::serving) fn new(
        layout: KvShardLayout,
        rank_gpus: Vec<GpuAddr>,
    ) -> Result<Self, KvPlanError> {
        if rank_gpus.len() != layout.ranks() {
            return Err(KvPlanError::RankCountMismatch {
                expected: layout.ranks(),
                actual: rank_gpus.len(),
            });
        }
        Ok(Self { layout, rank_gpus })
    }

    /// The first data replica of `score`'s placement, moved to `routed_node`
    /// when the placement fits on one node (the same remapping routing
    /// applies to route GPU sets).
    pub(in crate::serving) fn from_score(
        score: &ScoredParallelismConfig,
        kv_heads: u32,
        routed_node: NodeId,
    ) -> Result<Self, KvPlanError> {
        let config = score.config;
        let layout = KvShardLayout::new(
            config.tensor_ranks,
            config.pipeline_ranks,
            config.expert_ranks,
            kv_heads,
        );
        let single_node = single_placement_node(score).is_some();
        let rank_gpus = score
            .placement
            .rank_to_gpu
            .iter()
            .take(layout.ranks())
            .map(|gpu| {
                if single_node {
                    GpuAddr {
                        node_id: routed_node,
                        local_gpu_id: gpu.local_gpu_id,
                    }
                } else {
                    *gpu
                }
            })
            .collect();
        Self::new(layout, rank_gpus)
    }

    pub(in crate::serving) fn gpus(&self) -> &[GpuAddr] {
        &self.rank_gpus
    }
}

/// Cluster-level inputs every plan shares.
pub(in crate::serving) struct KvPlanContext<'a> {
    pub(in crate::serving) cluster: &'a Cluster,
    pub(in crate::serving) calibration: SimulationCalibration,
    pub(in crate::serving) calibration_profile: Option<&'a CalibrationProfileMetadata>,
    /// KV bytes of one token of one sequence across all layers and heads.
    pub(in crate::serving) bytes_per_token: u64,
}

impl KvPlanContext<'_> {
    pub(in crate::serving) fn kv_bytes_per_token(model: &ModelSpec) -> u64 {
        let head_dim = u64::from(model.hidden_size / model.attention_heads.max(1));
        u64::from(model.layers)
            * u64::from(model.kv_heads)
            * head_dim
            * 2
            * model.kv_dtype().bytes_per_element()
    }
}

/// How one flow's uncontended time was obtained.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) enum FlowPricing {
    MeasuredCurve {
        curve_label: String,
        extrapolation: &'static str,
        derived_region: bool,
    },
    AlphaBeta,
}

/// A plan plus the evidence the request states and outputs carry.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::serving) struct PlannedTransfer {
    pub(in crate::serving) plan: KvTransferPlan,
    pub(in crate::serving) paths: Vec<ServingKvTransferPathObservation>,
    /// Directed route resource ids, sorted and unique.
    pub(in crate::serving) resources: Vec<String>,
    pub(in crate::serving) bottlenecks: Vec<String>,
    pub(in crate::serving) pricing: Vec<FlowPricing>,
    pub(in crate::serving) fit: Option<CalibrationFitApplication>,
}

/// Builds plans, interning directed resources as `LinkId`s and caching
/// identical requests' plans.
pub(in crate::serving) struct KvTransferPlanner<'a> {
    context: KvPlanContext<'a>,
    links: HashMap<String, LinkId>,
    cache: HashMap<(PlacedKvLayout, PlacedKvLayout, u64), PlannedTransfer>,
}

impl<'a> KvTransferPlanner<'a> {
    pub(in crate::serving) fn new(context: KvPlanContext<'a>) -> Self {
        Self {
            context,
            links: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    /// The transfer of `sequences x prompt_tokens` tokens of KV from `source`
    /// to `destination`.
    pub(in crate::serving) fn plan(
        &mut self,
        source: &PlacedKvLayout,
        destination: &PlacedKvLayout,
        sequences: u32,
        prompt_tokens: u32,
    ) -> Result<PlannedTransfer, KvPlanError> {
        let total_bytes = u64::from(sequences.max(1))
            * u64::from(prompt_tokens)
            * self.context.bytes_per_token;
        let key = (source.clone(), destination.clone(), total_bytes);
        if let Some(planned) = self.cache.get(&key) {
            return Ok(planned.clone());
        }
        let planned = self.build(source, destination, total_bytes)?;
        self.cache.insert(key, planned.clone());
        Ok(planned)
    }

    fn build(
        &mut self,
        source: &PlacedKvLayout,
        destination: &PlacedKvLayout,
        total_bytes: u64,
    ) -> Result<PlannedTransfer, KvPlanError> {
        // Merge shard flows that land on the same GPU pair.
        let mut pair_bytes: BTreeMap<(GpuAddr, GpuAddr), f64> = BTreeMap::new();
        for flow in kv_shard_flows(source.layout, destination.layout) {
            let (Some(src), Some(dst)) = (
                source.rank_gpus.get(flow.src_rank),
                destination.rank_gpus.get(flow.dst_rank),
            ) else {
                continue;
            };
            if src == dst {
                continue;
            }
            *pair_bytes.entry((*src, *dst)).or_insert(0.0) += flow.fraction * total_bytes as f64;
        }

        let calibration = self.context.calibration.sanitized();
        let mut flows = Vec::with_capacity(pair_bytes.len());
        let mut paths = Vec::with_capacity(pair_bytes.len());
        let mut resources = Vec::new();
        let mut bottlenecks = Vec::new();
        let mut pricing = Vec::with_capacity(pair_bytes.len());
        let mut moved_bytes = 0_u64;
        let mut max_latency_s = 0.0_f64;
        let mut inter_node = false;
        for ((src, dst), bytes) in pair_bytes {
            let bytes = bytes.round().max(0.0) as u64;
            if bytes == 0 {
                continue;
            }
            moved_bytes += bytes;
            let path = kv_transfer_paths(
                self.context.cluster,
                &[src],
                &[dst],
                Bytes::from_bytes(bytes),
            )
            .into_iter()
            .next()
            .ok_or(KvPlanError::Unroutable {
                source: src,
                destination: dst,
            })?;
            inter_node |= src.node_id != dst.node_id;
            let query_nodes = [src.node_id, dst.node_id];
            let lookup = self
                .context
                .cluster
                .collective_curves
                .lookup(&CurveQuery {
                    kind: CollectiveKind::SendRecv,
                    participant_nodes: &query_nodes,
                });
            let (service_s, flow_pricing) = match lookup {
                CurveLookup::Matched(curve) => {
                    let evaluation = curve.curve().evaluate(bytes);
                    max_latency_s = max_latency_s.max(evaluation.floor_s);
                    (
                        evaluation.latency_s,
                        FlowPricing::MeasuredCurve {
                            curve_label: curve.label(),
                            extrapolation: evaluation.extrapolation.as_str(),
                            derived_region: evaluation.derived_region,
                        },
                    )
                }
                CurveLookup::NotConfigured | CurveLookup::Suspended(_) | CurveLookup::NoMatch => {
                    let latency_s = path.latency_s * calibration.collective_latency_scale;
                    let bandwidth_bytes_per_s = path.bottleneck_bandwidth_gbps * 1e9 / 8.0;
                    let bandwidth_s = if bandwidth_bytes_per_s.is_finite()
                        && bandwidth_bytes_per_s > 0.0
                    {
                        bytes as f64 / bandwidth_bytes_per_s * calibration.kv_transfer_scale
                            / calibration.collective_bandwidth_scale
                    } else {
                        f64::INFINITY
                    };
                    max_latency_s = max_latency_s.max(latency_s);
                    (latency_s + bandwidth_s, FlowPricing::AlphaBeta)
                }
            };
            if !service_s.is_finite() {
                return Err(KvPlanError::Unroutable {
                    source: src,
                    destination: dst,
                });
            }
            let mut link_ids = Vec::with_capacity(path.resource_details.len());
            for detail in &path.resource_details {
                let id = kv_route_directed_resource_id(detail);
                let next = self.links.len();
                link_ids.push(*self.links.entry(id.clone()).or_insert(next));
                resources.push(id);
            }
            if let Some(slowest) = path.resource_details.iter().min_by(|left, right| {
                left.bandwidth_gbps.total_cmp(&right.bandwidth_gbps)
            }) {
                bottlenecks.push(slowest.label.clone());
            }
            flows.push(KvFlow::new(link_ids, service_s).map_err(KvPlanError::Flow)?);
            paths.push(path);
            pricing.push(flow_pricing);
        }
        if inter_node {
            bottlenecks.push("KV transfer fabric/NIC path".to_string());
        }
        resources.sort();
        resources.dedup();
        bottlenecks.sort();
        bottlenecks.dedup();

        let mut plan = KvTransferPlan::new(flows, moved_bytes);
        let mut fit = None;
        if moved_bytes > 0 {
            let uncontended_s = plan.uncontended_s();
            let cost = CollectiveCost {
                latency_s: max_latency_s.min(uncontended_s),
                bandwidth_s: (uncontended_s - max_latency_s).max(0.0),
                total_s: uncontended_s,
                bottlenecks: bottlenecks.clone(),
                pricing: CollectivePricing::AlphaBeta(CurveCoverage::NotConsulted),
            };
            if let Some((seconds, application)) = Solver::fitted_kv_transfer_seconds(
                self.context.calibration_profile,
                Bytes::from_bytes(moved_bytes),
                &cost,
            ) && uncontended_s > 0.0
            {
                plan = plan.scaled(seconds / uncontended_s);
                fit = Some(application);
            }
        }
        Ok(PlannedTransfer {
            plan,
            paths,
            resources,
            bottlenecks,
            pricing,
            fit,
        })
    }
}
