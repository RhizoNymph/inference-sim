//! Compute efficiency as a function of tokens per forward pass.
//!
//! GEMM efficiency on a GPU depends on how many tokens a forward pass carries:
//! small passes leave tensor cores under-occupied, large ones saturate them.
//! A [`ComputeEfficiencyCurve`] holds measured `(tokens, efficiency)` points
//! and interpolates efficiency piecewise-linearly in `ln(tokens)`, clamping to
//! the first point's efficiency below the measured range and to the last
//! point's above it.
//!
//! The curve is stored inline in a fixed-capacity array so it stays `Copy`,
//! like every other calibration scalar. Construction validates the points, so
//! a curve value always has at least two points, strictly increasing positive
//! token counts, and efficiencies in `(0, 1]`.

use std::fmt::{Display, Formatter};

/// Maximum number of points a curve can hold.
pub const MAX_EFFICIENCY_CURVE_POINTS: usize = 32;

/// One validated `(tokens per forward pass, compute efficiency)` sample.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EfficiencyPoint {
    tokens: u64,
    efficiency: f64,
}

impl EfficiencyPoint {
    const PLACEHOLDER: Self = Self {
        tokens: 0,
        efficiency: 0.0,
    };

    pub fn tokens(self) -> u64 {
        self.tokens
    }

    pub fn efficiency(self) -> f64 {
        self.efficiency
    }
}

/// Why a list of raw points cannot form a curve.
#[derive(Clone, Debug, PartialEq)]
pub enum EfficiencyCurveError {
    TooFewPoints {
        count: usize,
    },
    TooManyPoints {
        count: usize,
        max: usize,
    },
    ZeroTokens {
        index: usize,
    },
    TokensNotIncreasing {
        index: usize,
        previous: u64,
        current: u64,
    },
    InvalidEfficiency {
        index: usize,
        efficiency: f64,
    },
}

impl Display for EfficiencyCurveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { count } => write!(
                f,
                "a compute efficiency curve needs at least 2 points, got {count}"
            ),
            Self::TooManyPoints { count, max } => write!(
                f,
                "a compute efficiency curve holds at most {max} points, got {count}"
            ),
            Self::ZeroTokens { index } => write!(f, "points[{index}] has zero tokens"),
            Self::TokensNotIncreasing {
                index,
                previous,
                current,
            } => write!(
                f,
                "points[{index}] tokens {current} must be greater than the previous point's {previous}"
            ),
            Self::InvalidEfficiency { index, efficiency } => write!(
                f,
                "points[{index}] efficiency {efficiency} must be finite, greater than 0.0 and at most 1.0"
            ),
        }
    }
}

impl std::error::Error for EfficiencyCurveError {}

/// Compute efficiency versus tokens per forward pass (see the module docs).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComputeEfficiencyCurve {
    points: [EfficiencyPoint; MAX_EFFICIENCY_CURVE_POINTS],
    len: usize,
}

impl ComputeEfficiencyCurve {
    /// Builds a curve from `(tokens, efficiency)` pairs, validating every
    /// invariant.
    pub fn new(raw: &[(u64, f64)]) -> Result<Self, EfficiencyCurveError> {
        if raw.len() < 2 {
            return Err(EfficiencyCurveError::TooFewPoints { count: raw.len() });
        }
        if raw.len() > MAX_EFFICIENCY_CURVE_POINTS {
            return Err(EfficiencyCurveError::TooManyPoints {
                count: raw.len(),
                max: MAX_EFFICIENCY_CURVE_POINTS,
            });
        }
        let mut points = [EfficiencyPoint::PLACEHOLDER; MAX_EFFICIENCY_CURVE_POINTS];
        let mut previous: Option<u64> = None;
        for (index, (&(tokens, efficiency), slot)) in raw.iter().zip(points.iter_mut()).enumerate()
        {
            if tokens == 0 {
                return Err(EfficiencyCurveError::ZeroTokens { index });
            }
            if let Some(previous) = previous
                && tokens <= previous
            {
                return Err(EfficiencyCurveError::TokensNotIncreasing {
                    index,
                    previous,
                    current: tokens,
                });
            }
            if !efficiency.is_finite() || efficiency <= 0.0 || efficiency > 1.0 {
                return Err(EfficiencyCurveError::InvalidEfficiency { index, efficiency });
            }
            *slot = EfficiencyPoint { tokens, efficiency };
            previous = Some(tokens);
        }
        Ok(Self {
            points,
            len: raw.len(),
        })
    }

    /// The validated points, in increasing token order.
    pub fn points(&self) -> &[EfficiencyPoint] {
        &self.points[..self.len]
    }

    /// Efficiency for a forward pass of `tokens` tokens: piecewise-linear in
    /// `ln(tokens)` between points, clamped at both ends. Non-finite or
    /// non-positive token counts take the first point's efficiency.
    pub fn efficiency_at(&self, tokens: f64) -> f64 {
        let points = self.points();
        let (Some(first), Some(last)) = (points.first(), points.last()) else {
            return 1.0;
        };
        if !tokens.is_finite() || tokens <= first.tokens as f64 {
            return first.efficiency;
        }
        if tokens >= last.tokens as f64 {
            return last.efficiency;
        }
        let x = tokens.ln();
        for pair in points.windows(2) {
            let [left, right] = pair else {
                continue;
            };
            let right_tokens = right.tokens as f64;
            if tokens <= right_tokens {
                let x0 = (left.tokens as f64).ln();
                let x1 = right_tokens.ln();
                let fraction = (x - x0) / (x1 - x0);
                return left.efficiency + fraction * (right.efficiency - left.efficiency);
            }
        }
        last.efficiency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(points: &[(u64, f64)]) -> ComputeEfficiencyCurve {
        match ComputeEfficiencyCurve::new(points) {
            Ok(curve) => curve,
            Err(err) => panic!("valid curve rejected: {err}"),
        }
    }

    #[test]
    fn rejects_fewer_than_two_points() {
        assert_eq!(
            ComputeEfficiencyCurve::new(&[(128, 0.7)]),
            Err(EfficiencyCurveError::TooFewPoints { count: 1 })
        );
        assert_eq!(
            ComputeEfficiencyCurve::new(&[]),
            Err(EfficiencyCurveError::TooFewPoints { count: 0 })
        );
    }

    #[test]
    fn rejects_too_many_points() {
        let raw: Vec<(u64, f64)> = (1..=33).map(|tokens| (tokens, 0.5)).collect();
        assert_eq!(
            ComputeEfficiencyCurve::new(&raw),
            Err(EfficiencyCurveError::TooManyPoints { count: 33, max: 32 })
        );
        let raw: Vec<(u64, f64)> = (1..=32).map(|tokens| (tokens, 0.5)).collect();
        assert!(ComputeEfficiencyCurve::new(&raw).is_ok());
    }

    #[test]
    fn rejects_zero_tokens() {
        assert_eq!(
            ComputeEfficiencyCurve::new(&[(0, 0.5), (16, 0.6)]),
            Err(EfficiencyCurveError::ZeroTokens { index: 0 })
        );
    }

    #[test]
    fn rejects_non_increasing_tokens() {
        assert_eq!(
            ComputeEfficiencyCurve::new(&[(128, 0.5), (128, 0.6)]),
            Err(EfficiencyCurveError::TokensNotIncreasing {
                index: 1,
                previous: 128,
                current: 128,
            })
        );
        assert_eq!(
            ComputeEfficiencyCurve::new(&[(256, 0.5), (128, 0.6)]),
            Err(EfficiencyCurveError::TokensNotIncreasing {
                index: 1,
                previous: 256,
                current: 128,
            })
        );
    }

    #[test]
    fn rejects_efficiency_outside_unit_interval() {
        for bad in [0.0, -0.1, 1.000_001, f64::NAN, f64::INFINITY] {
            let result = ComputeEfficiencyCurve::new(&[(128, 0.5), (256, bad)]);
            assert!(
                matches!(
                    result,
                    Err(EfficiencyCurveError::InvalidEfficiency { index: 1, .. })
                ),
                "efficiency {bad} accepted: {result:?}"
            );
        }
        assert!(ComputeEfficiencyCurve::new(&[(128, 0.5), (256, 1.0)]).is_ok());
    }

    #[test]
    fn returns_exact_values_at_points() {
        let curve = curve(&[(128, 0.70), (512, 0.78), (1024, 0.89)]);
        assert_eq!(curve.efficiency_at(128.0), 0.70);
        assert_eq!(curve.efficiency_at(512.0), 0.78);
        assert_eq!(curve.efficiency_at(1024.0), 0.89);
        assert_eq!(curve.points().len(), 3);
        assert_eq!(curve.points()[1].tokens(), 512);
        assert_eq!(curve.points()[1].efficiency(), 0.78);
    }

    #[test]
    fn interpolates_linearly_in_log_tokens() {
        let curve = curve(&[(100, 0.5), (10_000, 0.9)]);
        // ln(1000) is halfway between ln(100) and ln(10000).
        assert!((curve.efficiency_at(1000.0) - 0.7).abs() < 1e-12);
        // Geometric quarter point.
        let quarter = (100.0_f64.ln() * 0.75 + 10_000.0_f64.ln() * 0.25).exp();
        assert!((curve.efficiency_at(quarter) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn interpolates_non_monotonic_curves() {
        let curve = curve(&[(128, 0.70), (256, 0.66), (512, 0.78)]);
        let mid = (128.0_f64 * 256.0).sqrt();
        assert!((curve.efficiency_at(mid) - 0.68).abs() < 1e-12);
    }

    #[test]
    fn clamps_outside_the_measured_range() {
        let curve = curve(&[(128, 0.70), (4096, 0.89)]);
        assert_eq!(curve.efficiency_at(1.0), 0.70);
        assert_eq!(curve.efficiency_at(0.0), 0.70);
        assert_eq!(curve.efficiency_at(-5.0), 0.70);
        assert_eq!(curve.efficiency_at(f64::NAN), 0.70);
        assert_eq!(curve.efficiency_at(1e9), 0.89);
    }

    #[test]
    fn curve_is_copy_and_comparable() {
        let a = curve(&[(1, 0.1), (2, 0.2)]);
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, curve(&[(1, 0.1), (3, 0.2)]));
    }
}
