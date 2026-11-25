//! Measured collective-performance curves: fabric facts loaded from the
//! cluster TOML (`[[collective_curves]]`) that price collectives and
//! point-to-point transfers in place of the analytical alpha-beta model.
//! See docs/features/collective_curves.md.

pub mod curve;
pub mod set;
pub mod target;

pub use curve::{CurveError, CurveEvaluation, CurveExtrapolation, CurvePoint, MeasuredCurve};
pub use set::{CollectiveCurve, CollectiveCurveSet, CurveLookup, CurveSetError, CurveSuspension};
pub use target::{
    CollectiveCurveOp, CollectiveScope, CurveQuery, CurveTarget, DirectedNodePair,
    IntraNodeSelector, MatchSpecificity, NodeGroup, PointToPointScope, RankCount, TargetError,
};
