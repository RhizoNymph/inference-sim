//! Measured-curve pricing for collectives and the evidence it leaves.
//!
//! `Solver::curve_pricing` looks a call up in the cluster's
//! `CollectiveCurveSet`; a match replaces the alpha-beta estimate (and is not
//! rescaled by the collective calibration scalars, since the curve already is
//! a measurement). Every priced call is recorded as a `PricedCollective`, and
//! `collective_pricing_approximations` turns those records into approximation
//! entries: curve use, extrapolation, derived regions, missing curves, and
//! suspended curve sets.

use std::collections::BTreeSet;

use super::*;
use crate::types::{
    collective::CurveApplication,
    collective_curves::{CurveExtrapolation, CurveLookup, CurveQuery},
};

/// Result of consulting the curve set for one call.
pub(super) enum CurvePricing {
    Measured {
        latency_s: f64,
        bandwidth_s: f64,
        application: CurveApplication,
    },
    Uncovered(CurveCoverage),
}

/// One priced call in an operation trace.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PricedCollective {
    /// What was priced, e.g. `all_reduce nodes=0+1 ranks=2` or
    /// `send_recv node0->node1`.
    pub(super) call_label: String,
    pub(super) pricing: CollectivePricing,
}

impl PricedCollective {
    pub(super) fn new(
        kind: CollectiveKind,
        participant_nodes: &[NodeId],
        pricing: CollectivePricing,
    ) -> Self {
        Self {
            call_label: call_label(kind, participant_nodes),
            pricing,
        }
    }
}

impl Solver {
    pub(super) fn curve_pricing(
        cluster: &Cluster,
        kind: CollectiveKind,
        participant_nodes: &[NodeId],
        bytes: u64,
    ) -> CurvePricing {
        let query = CurveQuery {
            kind,
            participant_nodes,
        };
        match cluster.collective_curves.lookup(&query) {
            CurveLookup::NotConfigured => CurvePricing::Uncovered(CurveCoverage::NotConfigured),
            CurveLookup::Suspended(reason) => CurvePricing::Uncovered(CurveCoverage::Suspended {
                reason: reason.to_string(),
            }),
            CurveLookup::NoMatch => CurvePricing::Uncovered(CurveCoverage::NoMatch),
            CurveLookup::Matched(curve) => {
                let evaluation = curve.curve().evaluate(bytes);
                CurvePricing::Measured {
                    latency_s: evaluation.floor_s,
                    bandwidth_s: evaluation.latency_s - evaluation.floor_s,
                    application: CurveApplication {
                        curve_label: curve.label(),
                        source: curve.source().map(str::to_string),
                        bytes,
                        extrapolation: evaluation.extrapolation,
                        derived_region: evaluation.derived_region,
                    },
                }
            }
        }
    }
}

fn kind_label(kind: CollectiveKind) -> &'static str {
    match kind {
        CollectiveKind::AllToAll => "all_to_all",
        CollectiveKind::AllReduce => "all_reduce",
        CollectiveKind::AllGather => "all_gather",
        CollectiveKind::ReduceScatter => "reduce_scatter",
        CollectiveKind::Broadcast => "broadcast",
        CollectiveKind::SendRecv => "send_recv",
    }
}

fn call_label(kind: CollectiveKind, participant_nodes: &[NodeId]) -> String {
    if let (CollectiveKind::SendRecv, &[src, dst]) = (kind, participant_nodes) {
        return format!("send_recv node{src}->node{dst}");
    }
    let nodes: BTreeSet<_> = participant_nodes.iter().copied().collect();
    let nodes = nodes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join("+");
    format!(
        "{} nodes={nodes} ranks={}",
        kind_label(kind),
        participant_nodes.len()
    )
}

fn join(labels: &BTreeSet<String>) -> String {
    labels.iter().cloned().collect::<Vec<_>>().join(", ")
}

/// True when at least one call carried traffic and every such call was
/// priced from a measured curve, so the coarse alpha-beta caveat no longer
/// applies to this trace.
pub(super) fn all_traffic_priced_by_curves(priced: &[PricedCollective]) -> bool {
    let mut traffic = priced
        .iter()
        .filter(|record| !matches!(record.pricing, CollectivePricing::NoTraffic))
        .peekable();
    traffic.peek().is_some()
        && traffic.all(|record| matches!(record.pricing, CollectivePricing::MeasuredCurve(_)))
}

/// Curve-specific approximation entries for the trace's collectives (the
/// coarse alpha-beta entry is emitted by the caller).
pub(super) fn collective_pricing_approximations(
    phase: &str,
    priced: &[PricedCollective],
) -> Vec<SimulationApproximation> {
    let mut absent = BTreeSet::new();
    let mut suspended = BTreeSet::new();
    let mut measured = BTreeSet::new();
    let mut below = BTreeSet::new();
    let mut above = BTreeSet::new();
    let mut derived = BTreeSet::new();
    for record in priced {
        match &record.pricing {
            CollectivePricing::NoTraffic => {}
            CollectivePricing::AlphaBeta(coverage) => match coverage {
                CurveCoverage::NoMatch => {
                    absent.insert(record.call_label.clone());
                }
                CurveCoverage::Suspended { reason } => {
                    suspended.insert(reason.clone());
                }
                CurveCoverage::NotConfigured | CurveCoverage::NotConsulted => {}
            },
            CollectivePricing::MeasuredCurve(application) => {
                let curve = match &application.source {
                    Some(source) => format!("{} ({source})", application.curve_label),
                    None => application.curve_label.clone(),
                };
                measured.insert(curve);
                let sized = format!("{} at {} B", application.curve_label, application.bytes);
                match application.extrapolation {
                    CurveExtrapolation::Interpolated => {}
                    CurveExtrapolation::BelowRange => {
                        below.insert(sized.clone());
                    }
                    CurveExtrapolation::AboveRange => {
                        above.insert(sized.clone());
                    }
                }
                if application.derived_region {
                    derived.insert(sized);
                }
            }
        }
    }

    let mut approximations = Vec::new();
    let mut push = |code: &str, message: String, remediation: &str| {
        approximations.push(SimulationApproximation::new(
            phase,
            "communication",
            "collectives",
            code,
            message,
            Some(remediation.to_string()),
        ));
    };
    if !absent.is_empty() {
        push(
            "collective_curve_absent",
            format!(
                "The cluster lists measured collective curves, but no curve matches {} (op, placement scope, and rank count must match), so the alpha-beta model priced them.",
                join(&absent)
            ),
            "measure a collective curve for these operations and placements (tools/lab/lab.py curves) and add it to the cluster's [[collective_curves]]",
        );
    }
    if !suspended.is_empty() {
        push(
            "collective_curves_suspended",
            format!(
                "Measured collective curves are not used: {}.",
                join(&suspended)
            ),
            "re-measure the curves on the modified fabric, or compare against the unmodified topology",
        );
    }
    if !measured.is_empty() {
        push(
            "measured_collective_curve",
            format!(
                "Collectives are priced from measured curves ({}); curves are measured standalone, so concurrent traffic, compute overlap, and contention absent from the benchmark are not modeled.",
                join(&measured)
            ),
            "validate end-to-end against a real run that exercises the same collectives under load",
        );
    }
    if !below.is_empty() || !above.is_empty() {
        let mut parts = Vec::new();
        if !below.is_empty() {
            parts.push(format!(
                "below the measured range (curve floor used): {}",
                join(&below)
            ));
        }
        if !above.is_empty() {
            parts.push(format!(
                "above the measured range (last segment's bandwidth extended): {}",
                join(&above)
            ));
        }
        push(
            "collective_curve_extrapolated",
            format!(
                "Measured collective curves were extrapolated {}.",
                parts.join("; ")
            ),
            "extend the benchmark's message-size range to cover these sizes",
        );
    }
    if !derived.is_empty() {
        push(
            "collective_curve_derived_region",
            format!(
                "Collective curves were evaluated where their points were derived rather than measured (below derived_below_bytes): {}.",
                join(&derived)
            ),
            "re-measure these sizes with receiver-side timing (tools/lab/remote/collective_bench.py)",
        );
    }
    approximations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn application(extrapolation: CurveExtrapolation, derived_region: bool) -> CollectivePricing {
        CollectivePricing::MeasuredCurve(CurveApplication {
            curve_label: "send_recv node0->node1".to_string(),
            source: Some("lab.jsonl".to_string()),
            bytes: 512,
            extrapolation,
            derived_region,
        })
    }

    fn codes(approximations: &[SimulationApproximation]) -> Vec<&str> {
        approximations.iter().map(|a| a.code.as_str()).collect()
    }

    #[test]
    fn labels_calls_by_op_and_placement() {
        assert_eq!(
            call_label(CollectiveKind::SendRecv, &[1, 0]),
            "send_recv node1->node0"
        );
        assert_eq!(
            call_label(CollectiveKind::AllReduce, &[1, 0, 1, 0]),
            "all_reduce nodes=0+1 ranks=4"
        );
    }

    #[test]
    fn alpha_beta_without_curves_adds_no_curve_entries() {
        let priced = [PricedCollective::new(
            CollectiveKind::AllReduce,
            &[0, 1],
            CollectivePricing::AlphaBeta(CurveCoverage::NotConfigured),
        )];
        assert!(collective_pricing_approximations("decode", &priced).is_empty());
        assert!(!all_traffic_priced_by_curves(&priced));
        let none = [PricedCollective::new(
            CollectiveKind::AllReduce,
            &[0],
            CollectivePricing::NoTraffic,
        )];
        assert!(collective_pricing_approximations("decode", &none).is_empty());
        assert!(!all_traffic_priced_by_curves(&none));
        assert!(!all_traffic_priced_by_curves(&[]));
    }

    #[test]
    fn all_curve_pricing_requires_every_traffic_call_measured() {
        let measured = PricedCollective::new(
            CollectiveKind::SendRecv,
            &[0, 1],
            application(CurveExtrapolation::Interpolated, false),
        );
        let quiet = PricedCollective::new(
            CollectiveKind::SendRecv,
            &[0, 0],
            CollectivePricing::NoTraffic,
        );
        let fallback = PricedCollective::new(
            CollectiveKind::AllGather,
            &[0, 1],
            CollectivePricing::AlphaBeta(CurveCoverage::NoMatch),
        );
        assert!(all_traffic_priced_by_curves(&[measured.clone(), quiet]));
        assert!(!all_traffic_priced_by_curves(&[measured, fallback]));
    }

    #[test]
    fn measured_curves_report_use_extrapolation_and_derived_regions() {
        let priced = [
            PricedCollective::new(
                CollectiveKind::SendRecv,
                &[0, 1],
                application(CurveExtrapolation::Interpolated, false),
            ),
            PricedCollective::new(
                CollectiveKind::SendRecv,
                &[0, 1],
                application(CurveExtrapolation::BelowRange, true),
            ),
        ];
        let approximations = collective_pricing_approximations("decode", &priced);
        assert_eq!(
            codes(&approximations),
            vec![
                "measured_collective_curve",
                "collective_curve_extrapolated",
                "collective_curve_derived_region"
            ]
        );
        assert!(
            approximations[0]
                .message
                .contains("send_recv node0->node1 (lab.jsonl)")
        );
        assert!(
            approximations[1]
                .message
                .contains("below the measured range")
        );
        assert!(approximations[1].message.contains("at 512 B"));
    }

    #[test]
    fn missing_and_suspended_curves_are_reported_with_the_fallback() {
        let priced = [
            PricedCollective::new(
                CollectiveKind::AllGather,
                &[0, 1],
                CollectivePricing::AlphaBeta(CurveCoverage::NoMatch),
            ),
            PricedCollective::new(
                CollectiveKind::AllReduce,
                &[0, 1],
                CollectivePricing::AlphaBeta(CurveCoverage::Suspended {
                    reason: "scenario 'x' modifies the network".to_string(),
                }),
            ),
        ];
        let approximations = collective_pricing_approximations("prefill", &priced);
        assert_eq!(
            codes(&approximations),
            vec!["collective_curve_absent", "collective_curves_suspended"]
        );
        assert!(
            approximations[0]
                .message
                .contains("all_gather nodes=0+1 ranks=2")
        );
        assert!(approximations[1].message.contains("scenario 'x'"));
        assert!(approximations.iter().all(|a| a.category == "communication"));
    }
}
