#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimulationCalibration {
    pub compute_efficiency: f64,
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
}

impl Default for SimulationCalibration {
    fn default() -> Self {
        Self {
            compute_efficiency: 0.35,
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
            allow_compute_comm_overlap: true,
        }
    }
}

impl SimulationCalibration {
    pub fn sanitized(self) -> Self {
        Self {
            compute_efficiency: positive_or(self.compute_efficiency, 0.35),
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
        }
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
