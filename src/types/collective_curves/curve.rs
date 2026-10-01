//! A measured latency-versus-message-size curve and its evaluation.
//!
//! Points are interpolated piecewise-linearly in log-log space. Below the
//! smallest measured size the curve returns its floor (the smallest size's
//! latency); above the largest it extends the last segment at that segment's
//! marginal bandwidth. Every evaluation reports whether it extrapolated and
//! whether it landed in a region whose points were derived rather than
//! measured, so callers can attach evidence.

use std::fmt::{Display, Formatter};

/// One validated `(message bytes, latency)` sample.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CurvePoint {
    bytes: u64,
    latency_s: f64,
}

impl CurvePoint {
    pub fn bytes(self) -> u64 {
        self.bytes
    }

    pub fn latency_s(self) -> f64 {
        self.latency_s
    }
}

/// Why a list of raw points cannot form a curve.
#[derive(Clone, Debug, PartialEq)]
pub enum CurveError {
    TooFewPoints {
        count: usize,
    },
    ZeroBytes {
        index: usize,
    },
    BytesNotIncreasing {
        index: usize,
        previous: u64,
        current: u64,
    },
    InvalidLatency {
        index: usize,
        latency_us: f64,
    },
    DerivedThresholdOutsideRange {
        derived_below_bytes: u64,
        min_bytes: u64,
        max_bytes: u64,
    },
}

impl Display for CurveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { count } => {
                write!(f, "a curve needs at least 2 points, got {count}")
            }
            Self::ZeroBytes { index } => write!(f, "points[{index}] has zero message bytes"),
            Self::BytesNotIncreasing {
                index,
                previous,
                current,
            } => write!(
                f,
                "points[{index}] message bytes {current} must be greater than the previous point's {previous}"
            ),
            Self::InvalidLatency { index, latency_us } => write!(
                f,
                "points[{index}] latency {latency_us} us must be finite and positive"
            ),
            Self::DerivedThresholdOutsideRange {
                derived_below_bytes,
                min_bytes,
                max_bytes,
            } => write!(
                f,
                "derived_below_bytes {derived_below_bytes} must lie within the measured range ({min_bytes}..={max_bytes}]"
            ),
        }
    }
}

impl std::error::Error for CurveError {}

/// Where an evaluated size fell relative to the measured points.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CurveExtrapolation {
    /// Within `[min_bytes, max_bytes]`: log-log interpolation.
    Interpolated,
    /// Below `min_bytes`: the curve's latency floor.
    BelowRange,
    /// Above `max_bytes`: last segment's marginal bandwidth.
    AboveRange,
}

impl CurveExtrapolation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interpolated => "interpolated",
            Self::BelowRange => "below_range",
            Self::AboveRange => "above_range",
        }
    }
}

/// Result of evaluating a curve at one message size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CurveEvaluation {
    pub latency_s: f64,
    /// The curve's latency floor (the smallest measured size's latency),
    /// capped at `latency_s`; the rest of `latency_s` is size-dependent.
    pub floor_s: f64,
    pub extrapolation: CurveExtrapolation,
    /// True when the size lies below the curve's `derived_below_bytes`, i.e.
    /// in a range whose points were derived rather than measured.
    pub derived_region: bool,
}

/// A validated measured curve: at least two points, strictly increasing
/// nonzero message sizes, finite positive latencies.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredCurve {
    points: Vec<CurvePoint>,
    derived_below_bytes: Option<u64>,
}

impl MeasuredCurve {
    /// Builds a curve from `(message bytes, latency in microseconds)` pairs.
    /// `derived_below_bytes`, when set, marks points smaller than it as
    /// derived; it must lie in `(min_bytes, max_bytes]`.
    pub fn from_microseconds(
        points_us: &[(u64, f64)],
        derived_below_bytes: Option<u64>,
    ) -> Result<Self, CurveError> {
        if points_us.len() < 2 {
            return Err(CurveError::TooFewPoints {
                count: points_us.len(),
            });
        }
        let mut points = Vec::with_capacity(points_us.len());
        for (index, &(bytes, latency_us)) in points_us.iter().enumerate() {
            if bytes == 0 {
                return Err(CurveError::ZeroBytes { index });
            }
            if !latency_us.is_finite() || latency_us <= 0.0 {
                return Err(CurveError::InvalidLatency { index, latency_us });
            }
            if let Some(previous) = points.last().map(|point: &CurvePoint| point.bytes)
                && bytes <= previous
            {
                return Err(CurveError::BytesNotIncreasing {
                    index,
                    previous,
                    current: bytes,
                });
            }
            points.push(CurvePoint {
                bytes,
                latency_s: latency_us / 1e6,
            });
        }
        let curve = Self {
            points,
            derived_below_bytes: None,
        };
        if let Some(threshold) = derived_below_bytes
            && (threshold <= curve.min_bytes() || threshold > curve.max_bytes())
        {
            return Err(CurveError::DerivedThresholdOutsideRange {
                derived_below_bytes: threshold,
                min_bytes: curve.min_bytes(),
                max_bytes: curve.max_bytes(),
            });
        }
        Ok(Self {
            derived_below_bytes,
            ..curve
        })
    }

    pub fn points(&self) -> &[CurvePoint] {
        &self.points
    }

    pub fn derived_below_bytes(&self) -> Option<u64> {
        self.derived_below_bytes
    }

    pub fn min_bytes(&self) -> u64 {
        self.first().bytes
    }

    pub fn max_bytes(&self) -> u64 {
        self.last().bytes
    }

    /// Latency at the smallest measured size.
    pub fn floor_s(&self) -> f64 {
        self.first().latency_s
    }

    /// Marginal bandwidth (bytes/s) of the last segment, used above range.
    /// Falls back to the largest point's average bandwidth when noise makes
    /// the last segment non-increasing.
    pub fn tail_bandwidth_bytes_per_s(&self) -> f64 {
        let last = self.last();
        let previous = self.points[self.points.len() - 2];
        let delta_s = last.latency_s - previous.latency_s;
        if delta_s > 0.0 {
            (last.bytes - previous.bytes) as f64 / delta_s
        } else {
            last.bytes as f64 / last.latency_s
        }
    }

    pub fn evaluate(&self, bytes: u64) -> CurveEvaluation {
        let (latency_s, extrapolation) = if bytes < self.min_bytes() {
            (self.floor_s(), CurveExtrapolation::BelowRange)
        } else if bytes > self.max_bytes() {
            let last = self.last();
            let extra = (bytes - last.bytes) as f64 / self.tail_bandwidth_bytes_per_s();
            (last.latency_s + extra, CurveExtrapolation::AboveRange)
        } else {
            (self.interpolate(bytes), CurveExtrapolation::Interpolated)
        };
        CurveEvaluation {
            latency_s,
            floor_s: self.floor_s().min(latency_s),
            extrapolation,
            derived_region: self
                .derived_below_bytes
                .is_some_and(|threshold| bytes < threshold),
        }
    }

    fn interpolate(&self, bytes: u64) -> f64 {
        let upper = self
            .points
            .partition_point(|point| point.bytes < bytes)
            .min(self.points.len() - 1);
        let hi = self.points[upper];
        if hi.bytes == bytes || upper == 0 {
            return hi.latency_s;
        }
        let lo = self.points[upper - 1];
        let (log_lo, log_hi, log_x) = (
            (lo.bytes as f64).ln(),
            (hi.bytes as f64).ln(),
            (bytes as f64).ln(),
        );
        let fraction = (log_x - log_lo) / (log_hi - log_lo);
        (lo.latency_s.ln() + fraction * (hi.latency_s.ln() - lo.latency_s.ln())).exp()
    }

    fn first(&self) -> CurvePoint {
        self.points[0]
    }

    fn last(&self) -> CurvePoint {
        self.points[self.points.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        let tolerance = expected.abs() * 1e-9 + 1e-15;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn lab_all_reduce() -> MeasuredCurve {
        MeasuredCurve::from_microseconds(
            &[
                (16_384, 195.4),
                (32_768, 225.57),
                (65_536, 371.83),
                (131_072, 370.47),
            ],
            None,
        )
        .expect("valid curve")
    }

    #[test]
    fn rejects_fewer_than_two_points() {
        assert_eq!(
            MeasuredCurve::from_microseconds(&[(1024, 10.0)], None),
            Err(CurveError::TooFewPoints { count: 1 })
        );
        assert_eq!(
            MeasuredCurve::from_microseconds(&[], None),
            Err(CurveError::TooFewPoints { count: 0 })
        );
    }

    #[test]
    fn rejects_zero_bytes_unsorted_duplicate_and_bad_latency() {
        assert_eq!(
            MeasuredCurve::from_microseconds(&[(0, 1.0), (8, 2.0)], None),
            Err(CurveError::ZeroBytes { index: 0 })
        );
        assert_eq!(
            MeasuredCurve::from_microseconds(&[(8, 1.0), (8, 2.0)], None),
            Err(CurveError::BytesNotIncreasing {
                index: 1,
                previous: 8,
                current: 8
            })
        );
        assert_eq!(
            MeasuredCurve::from_microseconds(&[(16, 1.0), (8, 2.0)], None),
            Err(CurveError::BytesNotIncreasing {
                index: 1,
                previous: 16,
                current: 8
            })
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                MeasuredCurve::from_microseconds(&[(8, 1.0), (16, bad)], None),
                Err(CurveError::InvalidLatency { index: 1, .. })
            ));
        }
    }

    #[test]
    fn derived_threshold_must_lie_inside_the_measured_range() {
        let points = [(1024, 10.0), (4096, 20.0)];
        for bad in [1, 1024, 4097] {
            assert!(matches!(
                MeasuredCurve::from_microseconds(&points, Some(bad)),
                Err(CurveError::DerivedThresholdOutsideRange { .. })
            ));
        }
        let curve = MeasuredCurve::from_microseconds(&points, Some(4096)).expect("valid");
        assert_eq!(curve.derived_below_bytes(), Some(4096));
    }

    #[test]
    fn evaluates_measured_points_exactly() {
        let curve = lab_all_reduce();
        for point in curve.points().to_vec() {
            let evaluation = curve.evaluate(point.bytes());
            close(evaluation.latency_s, point.latency_s());
            assert_eq!(evaluation.extrapolation, CurveExtrapolation::Interpolated);
            assert!(!evaluation.derived_region);
        }
    }

    #[test]
    fn interpolates_in_log_log_space() {
        let curve = lab_all_reduce();
        // 57,344 B (batch-8 Qwen2.5-7B decode all-reduce) sits between the
        // 32 KiB and 64 KiB points, past NCCL's protocol step.
        let bytes = 57_344_u64;
        let fraction =
            ((bytes as f64).ln() - 32_768_f64.ln()) / (65_536_f64.ln() - 32_768_f64.ln());
        let expected_us = (225.57_f64.ln() + fraction * (371.83_f64.ln() - 225.57_f64.ln())).exp();
        let evaluation = curve.evaluate(bytes);
        close(evaluation.latency_s, expected_us / 1e6);
        assert_eq!(evaluation.extrapolation, CurveExtrapolation::Interpolated);
        assert!(evaluation.latency_s > 225.57e-6 && evaluation.latency_s < 371.83e-6);
        // A power law between two points is reproduced exactly.
        let power = MeasuredCurve::from_microseconds(&[(1_000, 10.0), (100_000, 1_000.0)], None)
            .expect("valid");
        close(power.evaluate(10_000).latency_s, 100.0e-6);
    }

    #[test]
    fn non_monotone_measurements_interpolate_without_overshoot() {
        let curve = lab_all_reduce();
        let evaluation = curve.evaluate(100_000);
        assert!(evaluation.latency_s <= 371.83e-6 && evaluation.latency_s >= 370.47e-6);
    }

    #[test]
    fn below_range_returns_the_floor() {
        let curve = lab_all_reduce();
        let evaluation = curve.evaluate(1);
        close(evaluation.latency_s, 195.4e-6);
        close(evaluation.floor_s, 195.4e-6);
        assert_eq!(evaluation.extrapolation, CurveExtrapolation::BelowRange);
    }

    #[test]
    fn above_range_extends_the_last_segment_bandwidth() {
        let curve =
            MeasuredCurve::from_microseconds(&[(1_000_000, 1_000.0), (2_000_000, 1_800.0)], None)
                .expect("valid");
        // Last segment: 1e6 bytes in 800 us -> 1.25e9 B/s.
        close(curve.tail_bandwidth_bytes_per_s(), 1.25e9);
        let evaluation = curve.evaluate(4_000_000);
        close(evaluation.latency_s, 1_800e-6 + 2_000_000.0 / 1.25e9);
        assert_eq!(evaluation.extrapolation, CurveExtrapolation::AboveRange);
        close(evaluation.floor_s, 1_000e-6);
    }

    #[test]
    fn above_range_falls_back_to_average_bandwidth_on_a_flat_tail() {
        let curve = lab_all_reduce();
        // Last segment decreases (371.83 -> 370.47 us), so use 131072 B / 370.47 us.
        let bandwidth = 131_072.0 / 370.47e-6;
        close(curve.tail_bandwidth_bytes_per_s(), bandwidth);
        close(
            curve.evaluate(262_144).latency_s,
            370.47e-6 + 131_072.0 / bandwidth,
        );
    }

    #[test]
    fn reports_derived_region_below_threshold_only() {
        let curve = MeasuredCurve::from_microseconds(
            &[(1024, 100.0), (65_536, 400.0), (1_048_576, 2_000.0)],
            Some(65_536),
        )
        .expect("valid");
        assert!(curve.evaluate(512).derived_region);
        assert!(curve.evaluate(4096).derived_region);
        assert!(!curve.evaluate(65_536).derived_region);
        assert!(!curve.evaluate(4_000_000).derived_region);
    }

    #[test]
    fn floor_never_exceeds_total() {
        let curve =
            MeasuredCurve::from_microseconds(&[(1024, 500.0), (2048, 100.0)], None).expect("valid");
        let evaluation = curve.evaluate(2048);
        close(evaluation.floor_s, 100.0e-6);
        assert!(evaluation.floor_s <= evaluation.latency_s);
    }
}
