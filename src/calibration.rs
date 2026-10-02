mod efficiency_curve;

pub use efficiency_curve::{
    ComputeEfficiencyCurve, EfficiencyCurveError, EfficiencyPoint, MAX_EFFICIENCY_CURVE_POINTS,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimulationCalibration {
    /// Fraction of peak FLOPs achieved. Used for every forward pass unless
    /// `compute_efficiency_curve` is set, which then takes precedence.
    pub compute_efficiency: f64,
    /// Optional efficiency versus tokens per forward pass; see
    /// [`SimulationCalibration::compute_efficiency_at`].
    pub compute_efficiency_curve: Option<ComputeEfficiencyCurve>,
    pub prefill_compute_scale: f64,
    pub decode_compute_scale: f64,
    pub decode_memory_bandwidth_scale: f64,
    pub collective_latency_scale: f64,
    pub collective_bandwidth_scale: f64,
    pub kv_transfer_scale: f64,
    pub scheduler_overhead_us: f64,
    pub serving_memory_temporary_fraction: f64,
    pub serving_memory_activation_communication_fraction: f64,
    pub serving_memory_weight_communication_fraction: f64,
    pub serving_memory_runtime_reserve_fraction: f64,
    pub serving_memory_fragmentation_fraction: f64,
    pub serving_pipeline_depth: u32,
    pub request_arrival_gap_s: f64,
    pub allow_compute_comm_overlap: bool,
    /// Fixed per-request API-server latency (tokenization, request handling,
    /// response streaming) the serving iteration engine adds between a
    /// request's arrival and the engine seeing it.
    pub frontend_latency_us: f64,
    /// Additional per-request API-server latency per prompt token.
    pub frontend_latency_per_prompt_token_us: f64,
}

impl Default for SimulationCalibration {
    fn default() -> Self {
        Self {
            compute_efficiency: 0.35,
            compute_efficiency_curve: None,
            prefill_compute_scale: 1.0,
            decode_compute_scale: 1.0,
            decode_memory_bandwidth_scale: 1.0,
            collective_latency_scale: 1.0,
            collective_bandwidth_scale: 1.0,
            kv_transfer_scale: 1.0,
            scheduler_overhead_us: 0.0,
            serving_memory_temporary_fraction: 0.5,
            serving_memory_activation_communication_fraction: 0.25,
            serving_memory_weight_communication_fraction: 0.005,
            serving_memory_runtime_reserve_fraction: 0.05,
            serving_memory_fragmentation_fraction: 0.03,
            serving_pipeline_depth: 4,
            request_arrival_gap_s: 0.0,
            allow_compute_comm_overlap: false,
            frontend_latency_us: 0.0,
            frontend_latency_per_prompt_token_us: 0.0,
        }
    }
}

impl SimulationCalibration {
    pub fn sanitized(self) -> Self {
        Self {
            compute_efficiency: positive_or(self.compute_efficiency, 0.35),
            compute_efficiency_curve: self.compute_efficiency_curve,
            prefill_compute_scale: positive_or(self.prefill_compute_scale, 1.0),
            decode_compute_scale: positive_or(self.decode_compute_scale, 1.0),
            decode_memory_bandwidth_scale: positive_or(self.decode_memory_bandwidth_scale, 1.0),
            collective_latency_scale: positive_or(self.collective_latency_scale, 1.0),
            collective_bandwidth_scale: positive_or(self.collective_bandwidth_scale, 1.0),
            kv_transfer_scale: positive_or(self.kv_transfer_scale, 1.0),
            scheduler_overhead_us: self.scheduler_overhead_us.max(0.0),
            serving_memory_temporary_fraction: nonnegative_or(
                self.serving_memory_temporary_fraction,
                0.5,
            ),
            serving_memory_activation_communication_fraction: nonnegative_or(
                self.serving_memory_activation_communication_fraction,
                0.25,
            ),
            serving_memory_weight_communication_fraction: nonnegative_or(
                self.serving_memory_weight_communication_fraction,
                0.005,
            ),
            serving_memory_runtime_reserve_fraction: nonnegative_or(
                self.serving_memory_runtime_reserve_fraction,
                0.05,
            ),
            serving_memory_fragmentation_fraction: nonnegative_or(
                self.serving_memory_fragmentation_fraction,
                0.03,
            ),
            serving_pipeline_depth: self.serving_pipeline_depth.max(1),
            request_arrival_gap_s: self.request_arrival_gap_s.max(0.0),
            allow_compute_comm_overlap: self.allow_compute_comm_overlap,
            frontend_latency_us: nonnegative_or(self.frontend_latency_us, 0.0),
            frontend_latency_per_prompt_token_us: nonnegative_or(
                self.frontend_latency_per_prompt_token_us,
                0.0,
            ),
        }
    }

    /// Compute efficiency for one forward pass carrying `tokens_per_pass`
    /// tokens: the curve's interpolated value when a curve is configured,
    /// otherwise the scalar `compute_efficiency`.
    pub fn compute_efficiency_at(&self, tokens_per_pass: f64) -> f64 {
        match &self.compute_efficiency_curve {
            Some(curve) => curve.efficiency_at(tokens_per_pass),
            None => self.compute_efficiency,
        }
    }

    /// Client-observed API-server latency for one request with
    /// `prompt_tokens` prompt tokens, in seconds.
    pub fn frontend_latency_s(&self, prompt_tokens: u32) -> f64 {
        (self.frontend_latency_us
            + self.frontend_latency_per_prompt_token_us * f64::from(prompt_tokens))
            / 1e6
    }

    /// Whether any frontend latency is configured.
    pub fn models_frontend_latency(&self) -> bool {
        self.frontend_latency_us > 0.0 || self.frontend_latency_per_prompt_token_us > 0.0
    }
}

fn positive_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn nonnegative_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_efficiency_applies_without_a_curve() {
        let calibration = SimulationCalibration {
            compute_efficiency: 0.8,
            ..SimulationCalibration::default()
        };
        assert_eq!(calibration.compute_efficiency_at(1.0), 0.8);
        assert_eq!(calibration.compute_efficiency_at(1e6), 0.8);
    }

    #[test]
    fn curve_takes_precedence_over_the_scalar() {
        let curve = match ComputeEfficiencyCurve::new(&[(128, 0.7), (1024, 0.9)]) {
            Ok(curve) => curve,
            Err(err) => panic!("valid curve rejected: {err}"),
        };
        let calibration = SimulationCalibration {
            compute_efficiency: 0.8,
            compute_efficiency_curve: Some(curve),
            ..SimulationCalibration::default()
        };
        assert_eq!(calibration.compute_efficiency_at(16.0), 0.7);
        assert_eq!(calibration.compute_efficiency_at(1024.0), 0.9);
        assert_eq!(
            calibration.sanitized().compute_efficiency_curve,
            Some(curve)
        );
    }

    #[test]
    fn frontend_latency_defaults_to_zero() {
        let calibration = SimulationCalibration::default();
        assert!(!calibration.models_frontend_latency());
        assert_eq!(calibration.frontend_latency_s(512), 0.0);
    }

    #[test]
    fn frontend_latency_has_fixed_and_per_token_terms() {
        let calibration = SimulationCalibration {
            frontend_latency_us: 5_000.0,
            frontend_latency_per_prompt_token_us: 10.0,
            ..SimulationCalibration::default()
        };
        assert!(calibration.models_frontend_latency());
        assert!((calibration.frontend_latency_s(512) - 0.010_12).abs() < 1e-12);
        assert!((calibration.frontend_latency_s(0) - 0.005).abs() < 1e-12);
    }

    #[test]
    fn sanitized_clamps_negative_frontend_latency() {
        let calibration = SimulationCalibration {
            frontend_latency_us: -1.0,
            frontend_latency_per_prompt_token_us: f64::NAN,
            ..SimulationCalibration::default()
        }
        .sanitized();
        assert_eq!(calibration.frontend_latency_us, 0.0);
        assert_eq!(calibration.frontend_latency_per_prompt_token_us, 0.0);
    }
}
