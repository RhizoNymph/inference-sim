# Measured collective curves and one-way-asymmetric links

## Scope

- **Measured curves.** A cluster TOML may list `[[collective_curves]]`:
  measured latency versus message size for a collective (all_reduce,
  all_gather, reduce_scatter, all_to_all, broadcast) over a scope and rank
  count, or for a directed point-to-point transfer (send_recv). When a curve
  matches a collective the solver prices it from the curve (log-log
  interpolation, floor below range, last segment's bandwidth above range)
  instead of the analytical alpha-beta model, and records evidence.
- **One-way asymmetry.** NICs can cap egress and/or ingress below their line
  rate (`egress_bandwidth_gbps`, `ingress_bandwidth_gbps`), and custom
  inter-node links can override bandwidth and latency per direction
  (`from_to_*`, `to_from_*`). The topology graph has directed edges, so
  routed transfers (KV transfer, pipeline sends priced through the graph) use
  the direction they travel, and multi-node collectives are bounded by each
  pair's slower direction.
- **Evidence.** Each priced collective yields a `PricedCollective`; the solver
  turns them into approximation records (`measured_collective_curve`,
  `collective_curve_extrapolated`, `collective_curve_derived_region`,
  `collective_curve_absent`, `collective_curves_suspended`) and keeps
  `coarse_collective_model` whenever any traffic-carrying collective still
  used alpha-beta. Approximation policies can warn on or reject any of these
  codes.

## Non-scope

- KV-cache transfers scheduled by the phase pipeline are never priced from
  curves (they use the routed graph plus any `kv_transfer` calibration fit);
  they do honour asymmetric links. Their pricing is
  `AlphaBeta(NotConsulted)`. KV pulls on the disaggregated serving engine do
  use directed `send_recv` curves per GPU-pair flow
  (docs/features/disaggregated_serving_engine.md).
- No curve scaling across rank counts, dtypes, or scopes: a curve prices only
  the exact op / scope / rank count it was measured for. Unmatched calls fall
  back to alpha-beta with `collective_curve_absent`.
- No in-situ effects: curves are standalone benchmarks, so concurrent
  traffic, compute overlap, rank skew, and CUDA-graph launch savings are not
  modelled (the `measured_collective_curve` record says so).
- Per-NIC directional caps (caps are node-wide, applied to every NIC of the
  node) and per-direction asymmetry for fat-tree/flat fabrics (only custom
  links and NICs carry direction) are not supported.
- The CLI topology summary does not yet print curves or directional caps.

## Configuration

```toml
# Collective over an exact node set.
[[collective_curves]]
op = "all_reduce"        # all_reduce | all_gather | reduce_scatter | all_to_all | broadcast
scope = "node_group"     # node_group | inter_node | intra_node
nodes = [0, 1]           # node_group: >= 2 distinct ids; intra_node: optional list; inter_node: forbidden
ranks = 2                # required, >= 2
source = "lab-runs/2026-09-30-nccl-curve/collective_curve.jsonl"   # optional provenance
points = [[1024, 132.66], [2048, 133.47]]   # [message_bytes_per_rank, latency_us]

# Directed point-to-point.
[[collective_curves]]
op = "send_recv"
scope = "node_pair"      # node_pair (src_node -> dst_node) | inter_node | intra_node
src_node = 0
dst_node = 1
derived_below_bytes = 67108864   # optional: points below this were derived, not measured
points = [[1024, 151.81], [67108864, 194939.89]]

# One-way-slow NIC (cap on transmit; ingress_bandwidth_gbps caps receive).
[[nodes]]
id = 0
nics = { count = 1, bandwidth_gbps = 9.41, egress_bandwidth_gbps = 3.61 }

# Asymmetric custom link: from -> to keeps the variant's profile here,
# to -> from is overridden.
[[interconnect.links]]
from = 0
to = 1
kind = "ethernet"
variant = "10g"
to_from_bandwidth_gbps = 2.5
to_from_latency_us = 40.0
```

Message bytes are the tensor each rank contributes, matching
`CollectiveCall::bytes_per_rank`: the full buffer for all_reduce, the per-rank
shard for all_gather / reduce_scatter (nccl-tests reports all_gather size as
the total; divide by ranks), the sent tensor for send_recv. Generate curves
with `tools/lab/lab.py curves` (docs/features/lab_harness.md), which screens
sender-timed point-to-point rows and fills `derived_below_bytes`.

Why the cluster TOML: curves are facts about a fabric between specific node
ids, exactly like links and NICs, so they live where those ids are defined,
are validated against them (unknown node -> config error), and travel with
`Cluster` into every solver and serving path. A calibration profile is tied
to a model/backend shape and is the wrong owner for fabric measurements.

## Data and control flow

1. `config::parse_cluster` deserializes `ClusterFile.collective_curves`
   (`CollectiveCurveSection`, `deny_unknown_fields`) and calls
   `config::collective_curves::parse_collective_curves` for both preset and
   custom clusters. Each section becomes a `CurveTarget` (validated through
   `RankCount`, `NodeGroup`, `IntraNodeSelector`, `DirectedNodePair`) plus a
   `MeasuredCurve` (`from_microseconds`: >= 2 points, strictly increasing
   nonzero bytes, finite positive latency, `derived_below_bytes` inside the
   range). `CollectiveCurveSet::new` rejects ambiguous targets (two curves a
   query could match at the same specificity); node ids are checked against
   the cluster. The set is stored in `Cluster.collective_curves`.
2. NIC caps: `nics_profile` → `parse_nic_direction_caps` fills
   `NodeNetworkProfile.nic_direction_caps` (`NicDirectionCaps` for every NIC
   of the node). Link overrides: `custom_interconnect_topology` →
   `interconnect_link_direction_overrides` → `LinkDirectionOverrides::resolve`
   produces `LinkDirectionality::{Symmetric, Asymmetric(AsymmetricLink)}`
   per resolved pair (oriented by the pair's `from`/`to`), stored on
   `CustomInterNodeLink.direction`.
3. Scenario overlays (`cli::scenario::apply_run_scenario_topology`): any
   overlay that scales interconnect/NIC bandwidth or latency, disables NICs,
   or degrades NICs/rails/links suspends the curve set
   (`CurveSuspension::ScenarioNetworkOverlay`); node and GPU overlays do not.
   NIC caps and directional link profiles scale with the same overlays
   (`NicDirectionCaps::scaled`, `CustomInterNodeLink::scale`).
4. `TopologyGraph::from_cluster` adds inter-node NIC→NIC and GPU-scoped edges
   per direction (`add_directed`): bandwidth = min(link direction bandwidth,
   source NIC egress, destination NIC ingress)
   (`effective_inter_node_bandwidth`), latency = the direction's latency. Both
   directions share one resource label, so contention accounting is unchanged.
   Symmetric configs produce identical edges both ways.
5. `Solver::build_operation_trace` (src/solver/operations.rs) issues each
   collective per activation crossing through
   `push_repeated_collective_operation`, which calls
   `Solver::estimate_collective_with_calibration`
   (src/solver/network_cost.rs). That computes the alpha-beta cost and its
   bottleneck resources as before (multi-node pairs route both directions and
   keep the slower via `slower_path`), then asks `Solver::curve_pricing`
   (src/solver/collective_pricing.rs) with the call's kind and the node of
   every participant in rank order. On `CurveLookup::Matched` the curve's
   evaluation replaces the time (`latency_s` = floor, `bandwidth_s` = rest)
   and the collective calibration scalars are **not** applied; otherwise
   alpha-beta stands, scaled as before. The resulting
   `CollectiveCost.pricing` (`CollectivePricing`) is appended to the trace's
   `PricedCollective` list.
6. `Solver::parallelism_approximations` (src/solver/placement.rs) emits
   `coarse_collective_model` unless every traffic-carrying collective was
   curve-priced (`all_traffic_priced_by_curves`), then appends
   `collective_pricing_approximations` (one record per code, listing the
   affected calls/curves and sizes).

## Files

| file | role | key items |
|---|---|---|
| src/types/collective_curves/curve.rs | curve math | `MeasuredCurve::{from_microseconds, evaluate, tail_bandwidth_bytes_per_s}`, `CurveEvaluation`, `CurveExtrapolation`, `CurveError` |
| src/types/collective_curves/target.rs | what a curve prices | `CurveTarget`, `CollectiveCurveOp`, `CollectiveScope`, `PointToPointScope`, `RankCount`, `NodeGroup`, `IntraNodeSelector`, `DirectedNodePair`, `CurveQuery`, `MatchSpecificity`, `TargetError` |
| src/types/collective_curves/set.rs | curve set + lookup | `CollectiveCurveSet::{new, lookup, suspend, referenced_nodes}`, `CollectiveCurve`, `CurveLookup`, `CurveSuspension`, `CurveSetError` |
| src/types/collective.rs | cost evidence | `CollectiveCost.pricing`, `CollectivePricing`, `CurveCoverage`, `CurveApplication` |
| src/types/fabric/direction.rs | asymmetry types | `NicDirectionCaps`, `AsymmetricLink`, `DirectionProfile`, `LinkDirectionality`, `DirectionError` |
| src/types/fabric/intra_node.rs | NIC rates | `NodeNetworkProfile.nic_direction_caps`, `nic_egress_bandwidth`, `nic_ingress_bandwidth` |
| src/types/fabric/inter_node.rs | link direction | `CustomInterNodeLink.direction`, `direction_profile`, `slowest_direction`, `scale` |
| src/config/collective_curves.rs | TOML parsing | `CollectiveCurveSection`, `parse_collective_curves` |
| src/config/cluster.rs | NIC/link parsing | `parse_nic_direction_caps`, `LinkDirectionOverrides`, `interconnect_link_direction_overrides` |
| src/topology_graph.rs | directed routing | `add_directed`, `GraphInterNodeLink::directed`, directional `effective_inter_node_bandwidth` |
| src/solver/network_cost.rs | collective cost | curve hook in `estimate_collective_with_calibration`, `slower_path`, all_gather/reduce_scatter shard semantics in `traffic_multiplier` |
| src/solver/collective_pricing.rs | curve pricing + evidence | `Solver::curve_pricing`, `PricedCollective`, `all_traffic_priced_by_curves`, `collective_pricing_approximations` |
| src/solver/operations.rs | trace | records `PricedCollective` per crossing |
| src/cli/scenario.rs | overlays | `scenario_modifies_network`, curve suspension, cap/link scaling |
| src/solver/curve_tests.rs, src/cli/curve_scenario_tests.rs | tests | solver-level and scenario-level behaviour |

## Invariants and constraints

- A `MeasuredCurve` always has >= 2 points with strictly increasing nonzero
  bytes and finite positive latencies; `derived_below_bytes` lies in
  `(min_bytes, max_bytes]`. Evaluation is continuous at measured points, the
  floor is constant below range, and the tail grows linearly above range at
  the last segment's marginal bandwidth (average bandwidth if that segment is
  non-increasing).
- No two curves in a set are ambiguous, so the most specific match is unique:
  exact node set / directed pair > listed intra-node > any-node intra-node >
  inter-node fabric.
- Point-to-point curves are directed: a `node_pair` curve for 0 -> 1 never
  prices 1 -> 0. Send participants are `[src, dst]` in rank order.
- Curve-priced costs ignore `collective_latency_scale` and
  `collective_bandwidth_scale`; alpha-beta costs keep them.
- Clusters without curves or asymmetry behave exactly as before (same edges,
  same costs, same approximation codes).
- A suspended set never prices anything but stays listed for reporting.
- NIC caps only lower a direction's rate (`min(line rate, cap)`); a cap above
  line rate has no effect.
