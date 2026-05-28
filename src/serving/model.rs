use super::*;

mod metrics;
mod pools;
mod traffic;
pub use metrics::*;
pub use pools::*;
pub use traffic::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingSearchSpace {
    pub prefill: SearchSpace,
    pub decode: SearchSpace,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingObjective {
    #[default]
    MinimizeE2el,
    MinimizeTtft,
    MinimizeTpot,
    MaximizeThroughput,
    MinimizeSloMissRate,
    MinimizeMemoryPressure,
    MinimizeCost,
    MinimizeEnergy,
    MinimizePower,
}

impl ServingObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingObjective::MinimizeE2el => "minimize_e2el",
            ServingObjective::MinimizeTtft => "minimize_ttft",
            ServingObjective::MinimizeTpot => "minimize_tpot",
            ServingObjective::MaximizeThroughput => "maximize_throughput",
            ServingObjective::MinimizeSloMissRate => "minimize_slo_miss_rate",
            ServingObjective::MinimizeMemoryPressure => "minimize_memory_pressure",
            ServingObjective::MinimizeCost => "minimize_cost",
            ServingObjective::MinimizeEnergy => "minimize_energy",
            ServingObjective::MinimizePower => "minimize_power",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredServingConfig {
    pub candidate_id: String,
    pub objective: ServingObjective,
    pub deployment_mode: ServingDeploymentMode,
    pub slo_miss_penalty_weight: f64,
    pub slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
    pub slo_miss_penalty_components: ServingSloMissPenaltyComponents,
    pub traffic_class_slo_miss_penalties: Vec<ServingTrafficClassSloMissPenalty>,
    pub slo_miss_penalty_score: f64,
    pub service_backpressure_penalty_weight: f64,
    pub service_backpressure_penalty_score: f64,
    pub topology_risk_penalty_weight: f64,
    pub topology_risk_penalty_score: f64,
    pub max_memory_pressure_fraction: Option<f64>,
    pub max_unique_gpus: Option<u32>,
    pub min_throughput_tokens_per_s: Option<f64>,
    pub cost_model: ServingCostModel,
    pub metric_ceilings: ServingMetricCeilings,
    pub kv_route_constraints: ServingKvRouteConstraints,
    pub pool_label: Option<String>,
    pub prefill_nodes: Vec<NodeId>,
    pub decode_nodes: Vec<NodeId>,
    pub prefill_gpu_labels: Vec<String>,
    pub decode_gpu_labels: Vec<String>,
    pub pool_topology: ServingPoolTopologySummary,
    pub pool_search_summary: Option<ServingPoolSearchSummary>,
    pub route_coverage: ServingRouteCoverage,
    pub prefill_config: ParallelismConfig,
    pub decode_config: ParallelismConfig,
    pub prefill_score: ScoredParallelismConfig,
    pub decode_score: ScoredParallelismConfig,
    pub prefill_memory: ServingMemoryHeadroom,
    pub decode_memory: ServingMemoryHeadroom,
    pub calibration_summary: ServingCalibrationSummary,
    pub calibration_fits: Vec<CalibrationFitApplication>,
    pub phase_calibration: Vec<ServingPhaseCalibrationObservation>,
    pub calibration_gate_violations: Vec<CalibrationGateViolation>,
    pub approximation_summary: ServingApproximationSummary,
    pub approximations: Vec<SimulationApproximation>,
    pub approximation_policy_violations: Vec<ApproximationPolicyViolation>,
    pub metrics: ServingMetrics,
    pub measurement_window: ServingMeasurementWindowObservation,
    pub metric_breakdowns: Vec<ServingMetricBreakdown>,
    pub hardware_footprint: ServingHardwareFootprint,
    pub cost_estimate: ServingCostEstimate,
    pub memory_pressure: Vec<ServingMemoryPressureObservation>,
    pub kv_route_resource_summary: Vec<ServingKvRouteResourceSummary>,
    pub kv_route_topology_summary: ServingKvRouteTopologySummary,
    pub topology_bottlenecks: Vec<ServingTopologyBottleneckObservation>,
    pub bottleneck_summary: Vec<ServingBottleneckSummary>,
    pub pareto: ServingParetoFrontier,
    pub request_observations: Vec<ServingRequestObservation>,
    pub decode_iterations: Vec<ServingDecodeIterationObservation>,
    pub node_capacity: Vec<ServingNodeCapacityObservation>,
    pub gpu_capacity: Vec<ServingGpuCapacityObservation>,
    pub traffic_class_capacity: Vec<ServingTrafficClassCapacityObservation>,
    pub service_observations: Vec<ServingServiceObservation>,
    pub worker_observations: Vec<ServingWorkerObservation>,
    pub scheduled_operations: Vec<ScheduledOperation>,
    pub resource_utilization: Vec<ResourceUtilization>,
    pub phase_resource_utilization: Vec<ServingPhaseResourceUtilization>,
    pub feasible: bool,
    pub bottlenecks: Vec<String>,
    pub rejections: Vec<ServingRejection>,
    pub rejected_reason: Option<String>,
}

impl ScoredServingConfig {
    pub fn refresh_calibration_summary(&mut self) {
        self.calibration_summary = serving_calibration_summary(
            &self.phase_calibration,
            &self.calibration_fits,
            &self.calibration_gate_violations,
        );
    }

    pub fn refresh_approximation_summary(&mut self) {
        self.approximation_summary = serving_approximation_summary(
            &self.approximations,
            &self.approximation_policy_violations,
        );
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingApproximationSummary {
    pub status: String,
    pub approximation_count: u32,
    pub policy_violation_count: u32,
    pub category_counts: Vec<ServingApproximationCount>,
    pub top_codes: Vec<ServingApproximationCount>,
    pub calibration_count: u32,
    pub topology_count: u32,
    pub queueing_count: u32,
    pub runtime_count: u32,
    pub memory_count: u32,
    pub capacity_count: u32,
    pub routing_count: u32,
    pub admission_count: u32,
    pub uncalibrated_phase_count: u32,
    pub uncalibrated_queue_component_count: u32,
    pub extrapolated_fit_count: u32,
    pub coarse_topology: bool,
    pub approximate_queueing: bool,
    pub uncalibrated_runtime: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingApproximationCount {
    pub name: String,
    pub count: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingCalibrationSummary {
    pub status: String,
    pub active_phase_count: u32,
    pub calibrated_phase_count: u32,
    pub uncalibrated_phase_count: u32,
    pub coverage_fraction: f64,
    pub fit_count: u32,
    pub extrapolated_fit_count: u32,
    pub unbounded_fit_count: u32,
    pub fit_count_with_uncertainty: u32,
    pub min_confidence_score: Option<f64>,
    pub max_extrapolation_ratio: Option<f64>,
    pub relative_uncertainty_pct: Option<f64>,
    pub absolute_uncertainty_s: Option<f64>,
    pub gate_violation_count: u32,
    pub hard_gate_violation_count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingParetoFrontier {
    pub is_frontier: bool,
    pub rank: Option<u32>,
    pub dominated_by: Vec<String>,
    pub dimensions: Vec<ServingParetoDimension>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingParetoDimension {
    pub metric: String,
    pub direction: String,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingBottleneckSummary {
    pub source: String,
    pub phase: String,
    pub category: String,
    pub resource: String,
    pub code: String,
    pub severity: String,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub unit: Option<String>,
    pub message: String,
    pub remediation: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingHardwareFootprint {
    pub unique_node_count: u32,
    pub unique_gpu_count: u32,
    pub prefill_node_count: u32,
    pub prefill_gpu_count: u32,
    pub decode_node_count: u32,
    pub decode_gpu_count: u32,
    pub shared_node_count: u32,
    pub shared_gpu_count: u32,
    pub aggregate_hbm_gb: f64,
    pub prefill_hbm_gb: f64,
    pub decode_hbm_gb: f64,
    pub aggregate_hbm_bandwidth_gb_s: f64,
    pub prefill_hbm_bandwidth_gb_s: f64,
    pub decode_hbm_bandwidth_gb_s: f64,
    pub aggregate_peak_f16_tflops: f64,
    pub prefill_peak_f16_tflops: f64,
    pub decode_peak_f16_tflops: f64,
    pub aggregate_peak_f8_tflops: Option<f64>,
    pub prefill_peak_f8_tflops: Option<f64>,
    pub decode_peak_f8_tflops: Option<f64>,
    pub aggregate_gpu_types: Vec<ServingGpuTypeCount>,
    pub prefill_gpu_types: Vec<ServingGpuTypeCount>,
    pub decode_gpu_types: Vec<ServingGpuTypeCount>,
    pub aggregate_gpu_label_counts: Vec<ServingGpuLabelCount>,
    pub prefill_gpu_label_counts: Vec<ServingGpuLabelCount>,
    pub decode_gpu_label_counts: Vec<ServingGpuLabelCount>,
    pub aggregate_effective_peak_tflops: f64,
    pub prefill_effective_peak_tflops: f64,
    pub decode_effective_peak_tflops: f64,
    pub throughput_tokens_per_s_per_gpu: f64,
    pub throughput_tokens_per_s_per_effective_peak_tflop: f64,
    pub throughput_tokens_per_s_per_hbm_gb: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingGpuTypeCount {
    pub gpu: String,
    pub count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingGpuLabelCount {
    pub label: String,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingTrafficClassSloMissPenalty {
    pub name: String,
    pub group: String,
    pub key: String,
    pub weights: ServingSloMissPenaltyWeights,
    pub components: ServingSloMissPenaltyComponents,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingRouteCoverage {
    pub candidate_count: u32,
    pub routable_candidate_count: u32,
    pub unroutable_candidate_count: u32,
    pub fraction: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingKvRouteTopologySummary {
    pub route_resource_count: u32,
    pub inter_node_route_resource_count: u32,
    pub gpu_nic_route_resource_count: u32,
    pub intra_node_route_resource_count: u32,
    pub rail_ids: Vec<u32>,
    pub rail_count: u32,
    pub single_rail_dependency: bool,
    pub single_rail_id: Option<u32>,
    pub unrailed_inter_node_route_resource_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingTopologyBottleneckObservation {
    pub phase: String,
    pub category: String,
    pub resource: String,
    pub code: String,
    pub severity: String,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub unit: Option<String>,
    pub message: String,
    pub remediation: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingPhaseCalibrationObservation {
    pub phase: String,
    pub active: bool,
    pub calibrated: bool,
    pub fit_count: u32,
    pub applied_targets: Vec<String>,
    pub estimated_s: f64,
    pub status: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvRouteResourceSummary {
    pub resource_id: String,
    pub kind: String,
    pub label: String,
    pub request_count: u32,
    pub path_observations: u64,
    pub transfer_bytes: u64,
    pub estimated_transfer_s: f64,
    pub min_bandwidth_gbps: f64,
    pub max_latency_s: f64,
    pub rail_id: Option<u32>,
    pub from: Option<ServingKvTransferPathEndpointObservation>,
    pub to: Option<ServingKvTransferPathEndpointObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingPhaseResourceUtilization {
    pub phase: String,
    pub resource_kind: String,
    pub resource: String,
    pub busy_s: f64,
    pub utilization: f64,
    pub operation_count: usize,
    pub first_start_s: f64,
    pub last_finish_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRejection {
    pub phase: String,
    pub category: String,
    pub resource: String,
    pub code: String,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub unit: Option<String>,
    pub remediation: Option<String>,
    pub message: String,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingMemoryHeadroom {
    pub estimated_per_gpu_gb: f64,
    pub min_hbm_per_gpu_gb: f64,
    pub limiting_gpu: Option<GpuAddr>,
    pub headroom_gb: f64,
    pub headroom_fraction: f64,
    pub components: ServingMemoryComponents,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingMemoryComponentContribution {
    pub name: &'static str,
    pub gb: f64,
    pub fraction_of_total: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingMemoryPressureObservation {
    pub phase: String,
    pub estimate_kind: String,
    pub start_s: f64,
    pub finish_s: f64,
    pub duration_s: f64,
    pub active_requests: u32,
    pub active_tokens: u64,
    pub kv_blocks: u64,
    pub estimated_per_gpu_gb: f64,
    pub min_hbm_per_gpu_gb: f64,
    pub capacity_used_fraction: f64,
    pub headroom_gb: f64,
    pub limiting_gpu: Option<GpuAddr>,
    pub dominant_component: Option<ServingMemoryComponentContribution>,
    pub components: ServingMemoryComponents,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingMemoryComponents {
    pub weights_gb: f64,
    pub kv_cache_gb: f64,
    pub block_table_gb: f64,
    pub activations_gb: f64,
    pub temporary_gb: f64,
    pub communication_gb: f64,
    pub runtime_reserve_gb: f64,
    pub fragmentation_gb: f64,
    pub total_gb: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServingRequestStatus {
    Pending,
    Completed,
    RejectedAdmission,
    TimedOut,
    Cancelled,
}

impl ServingRequestStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingRequestStatus::Pending => "pending",
            ServingRequestStatus::Completed => "completed",
            ServingRequestStatus::RejectedAdmission => "rejected_admission",
            ServingRequestStatus::TimedOut => "timed_out",
            ServingRequestStatus::Cancelled => "cancelled",
        }
    }

    pub(super) fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }

    pub(super) fn is_admitted(self) -> bool {
        !matches!(self, Self::RejectedAdmission)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServingRequestEventKind {
    Arrived,
    QueuedForPrefill,
    PrefillStarted,
    PrefillFinished,
    QueuedForKvTransfer,
    KvTransferStarted,
    KvTransferFinished,
    KvBlocksAllocated,
    QueuedForDecode,
    DecodeIterationStarted,
    DecodeIterationFinished,
    KvBlocksReleased,
    Completed,
    RejectedAdmission,
    TimedOut,
    Cancelled,
}

impl ServingRequestEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingRequestEventKind::Arrived => "arrived",
            ServingRequestEventKind::QueuedForPrefill => "queued_for_prefill",
            ServingRequestEventKind::PrefillStarted => "prefill_started",
            ServingRequestEventKind::PrefillFinished => "prefill_finished",
            ServingRequestEventKind::QueuedForKvTransfer => "queued_for_kv_transfer",
            ServingRequestEventKind::KvTransferStarted => "kv_transfer_started",
            ServingRequestEventKind::KvTransferFinished => "kv_transfer_finished",
            ServingRequestEventKind::KvBlocksAllocated => "kv_blocks_allocated",
            ServingRequestEventKind::QueuedForDecode => "queued_for_decode",
            ServingRequestEventKind::DecodeIterationStarted => "decode_iteration_started",
            ServingRequestEventKind::DecodeIterationFinished => "decode_iteration_finished",
            ServingRequestEventKind::KvBlocksReleased => "kv_blocks_released",
            ServingRequestEventKind::Completed => "completed",
            ServingRequestEventKind::RejectedAdmission => "rejected_admission",
            ServingRequestEventKind::TimedOut => "timed_out",
            ServingRequestEventKind::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestEventObservation {
    pub kind: ServingRequestEventKind,
    pub phase: String,
    pub at_s: f64,
    pub decode_iteration: Option<u32>,
    pub message: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServingRequestPhaseCategory {
    Queue,
    Service,
    Transfer,
}

impl ServingRequestPhaseCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingRequestPhaseCategory::Queue => "queue",
            ServingRequestPhaseCategory::Service => "service",
            ServingRequestPhaseCategory::Transfer => "transfer",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestPhaseBreakdownObservation {
    pub phase: String,
    pub category: ServingRequestPhaseCategory,
    pub start_s: f64,
    pub finish_s: f64,
    pub duration_s: f64,
    pub contributes_to_ttft: bool,
    pub contributes_to_e2el: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvBlockOwnershipObservation {
    pub allocation_id: String,
    pub owner: GpuAddr,
    pub owner_worker_slots: Vec<u32>,
    pub worker_slot_ownership: Vec<ServingKvWorkerSlotOwnershipObservation>,
    pub decode_operation_ids: Vec<usize>,
    pub allocated_at_s: f64,
    pub released_at_s: f64,
    pub duration_s: f64,
    pub block_start: u64,
    pub block_end: u64,
    pub decode_sequences: u32,
    pub resident_tokens: u64,
    pub kv_blocks: u64,
    pub allocated_kv_tokens: u64,
    pub kv_fragmentation_tokens: u64,
    pub block_table_entries: u64,
    pub block_table_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvWorkerSlotOwnershipObservation {
    pub allocation_id: String,
    pub slot: u32,
    pub block_start: u64,
    pub block_end: u64,
    pub decode_sequences: u32,
    pub resident_tokens: u64,
    pub kv_blocks: u64,
    pub allocated_kv_tokens: u64,
    pub kv_fragmentation_tokens: u64,
    pub block_table_entries: u64,
    pub block_table_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestWorkerSummaryObservation {
    pub role: String,
    pub phase: String,
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub worker_slots: Vec<u32>,
    pub operation_ids: Vec<usize>,
    pub assignment_count: u32,
    pub start_s: f64,
    pub finish_s: f64,
    pub resident_tokens: u64,
    pub kv_blocks: u64,
}

impl ServingMemoryHeadroom {
    pub(super) fn unavailable() -> Self {
        Self {
            estimated_per_gpu_gb: f64::INFINITY,
            min_hbm_per_gpu_gb: f64::INFINITY,
            limiting_gpu: None,
            headroom_gb: f64::INFINITY,
            headroom_fraction: f64::INFINITY,
            components: ServingMemoryComponents::unavailable(),
        }
    }

    pub fn capacity_used_fraction(&self) -> f64 {
        if self.estimated_per_gpu_gb.is_finite()
            && self.min_hbm_per_gpu_gb.is_finite()
            && self.min_hbm_per_gpu_gb > 0.0
        {
            self.estimated_per_gpu_gb / self.min_hbm_per_gpu_gb
        } else {
            f64::INFINITY
        }
    }

    pub fn dominant_component(&self) -> Option<ServingMemoryComponentContribution> {
        self.components.dominant_component()
    }
}

impl ServingMemoryComponents {
    pub(super) fn unavailable() -> Self {
        Self {
            weights_gb: f64::INFINITY,
            kv_cache_gb: f64::INFINITY,
            block_table_gb: f64::INFINITY,
            activations_gb: f64::INFINITY,
            temporary_gb: f64::INFINITY,
            communication_gb: f64::INFINITY,
            runtime_reserve_gb: f64::INFINITY,
            fragmentation_gb: f64::INFINITY,
            total_gb: f64::INFINITY,
        }
    }

    pub fn component_contributions(&self) -> [ServingMemoryComponentContribution; 8] {
        let total_gb = self.total_gb;
        [
            self.component_contribution("weights", self.weights_gb, total_gb),
            self.component_contribution("kv_cache", self.kv_cache_gb, total_gb),
            self.component_contribution("block_table", self.block_table_gb, total_gb),
            self.component_contribution("activations", self.activations_gb, total_gb),
            self.component_contribution("temporary", self.temporary_gb, total_gb),
            self.component_contribution("communication", self.communication_gb, total_gb),
            self.component_contribution("runtime_reserve", self.runtime_reserve_gb, total_gb),
            self.component_contribution("fragmentation", self.fragmentation_gb, total_gb),
        ]
    }

    pub fn dominant_component(&self) -> Option<ServingMemoryComponentContribution> {
        if !self.total_gb.is_finite() || self.total_gb <= 0.0 {
            return None;
        }

        self.component_contributions()
            .into_iter()
            .filter(|component| component.gb.is_finite() && component.gb >= 0.0)
            .max_by(|left, right| left.gb.total_cmp(&right.gb))
    }

    pub(super) fn component_contribution(
        &self,
        name: &'static str,
        gb: f64,
        total_gb: f64,
    ) -> ServingMemoryComponentContribution {
        let fraction_of_total = if gb.is_finite() && total_gb.is_finite() && total_gb > 0.0 {
            gb / total_gb
        } else {
            f64::INFINITY
        };

        ServingMemoryComponentContribution {
            name,
            gb,
            fraction_of_total,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingNodeCapacityObservation {
    pub node_id: NodeId,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingTrafficClassCapacityObservation {
    pub name: String,
    pub group: String,
    pub key: String,
    pub max_prefill_tokens: Option<u64>,
    pub max_decode_sequences: Option<u32>,
    pub max_resident_tokens: Option<u64>,
    pub max_kv_blocks: Option<u64>,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub prefill_token_utilization: f64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingDecodeIterationObservation {
    pub iteration_idx: u32,
    pub decode_nodes: Vec<NodeId>,
    pub request_indices: Vec<u32>,
    pub operation_ids: Vec<usize>,
    pub start_s: f64,
    pub finish_s: f64,
    pub latency_s: f64,
    pub batch_tokens: u32,
    pub first_token_batch_tokens: u32,
    pub tail_token_batch_tokens: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingGpuCapacityObservation {
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingWorkerObservation {
    pub phase: String,
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub configured_worker_slots: u32,
    pub peak_active_worker_slots: u32,
    pub worker_slot_utilization: f64,
    pub request_count: u32,
    pub completed_requests: u32,
    pub failed_requests: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub queue_s: f64,
    pub queue_p95_s: f64,
    pub queue_max_s: f64,
    pub worker_queue_s: f64,
    pub worker_queue_p95_s: f64,
    pub worker_queue_max_s: f64,
    pub resource_queue_s: f64,
    pub resource_queue_p95_s: f64,
    pub resource_queue_max_s: f64,
    pub service_s: f64,
    pub first_start_s: f64,
    pub last_finish_s: f64,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub kv_cache_owner_slots: Vec<ServingWorkerKvSlotObservation>,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingWorkerKvSlotObservation {
    pub slot: u32,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingWorkerAssignmentObservation {
    pub phase: String,
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub slot: u32,
    pub start_s: f64,
    pub finish_s: f64,
    pub operation_ids: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingServiceObservation {
    pub phase: String,
    pub health: ServingServiceHealth,
    pub accepts_requests: bool,
    pub worker_scale: f64,
    pub configured_worker_slots_per_gpu: u32,
    pub effective_worker_slots_per_gpu: u32,
    pub node_count: u32,
    pub gpu_count: u32,
    pub request_count: u32,
    pub admitted_requests: u32,
    pub completed_requests: u32,
    pub failed_requests: u32,
    pub rejected_requests: u32,
    pub timed_out_requests: u32,
    pub cancelled_requests: u32,
    pub queue_cap_s: Option<f64>,
    pub queue_cap_request_count: u32,
    pub queue_cap_hit_count: u32,
    pub decode_iteration_queue_cap_s: Option<f64>,
    pub decode_iteration_queue_cap_request_count: u32,
    pub decode_iteration_queue_cap_hit_count: u32,
    pub backpressure_rejections: u32,
    pub timeout_rejections: u32,
    pub backpressure_state: String,
    pub worker_slot_utilization: f64,
    pub queue_s: f64,
    pub queue_p95_s: f64,
    pub queue_max_s: f64,
    pub worker_queue_s: f64,
    pub resource_queue_s: f64,
    pub service_s: f64,
}

pub struct ServingSolver;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingSolverOptions<'a> {
    pub calibration: SimulationCalibration,
    pub calibration_profile: Option<&'a CalibrationProfileMetadata>,
    pub model_id: Option<&'a str>,
    pub serving_stack: Option<&'a str>,
    pub serving_runtime_features: Option<&'a [String]>,
    pub max_prefill_candidates: Option<usize>,
    pub max_decode_candidates: Option<usize>,
    pub max_serving_pairs: Option<usize>,
    pub search_deadline: Option<Instant>,
    pub explicit_prefill_placement: Option<&'a RankPlacement>,
    pub explicit_decode_placement: Option<&'a RankPlacement>,
}
