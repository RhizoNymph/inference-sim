use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    error::Error,
    fmt::Display,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

mod trace;
use trace::*;

use crate::{
    DisaggregatedServingConfig, SearchSpace, ServingArrivalPattern, ServingCostModel,
    ServingDecodeBatching, ServingDecodeCapacityPolicy, ServingDeploymentMode, ServingGpuCostRate,
    ServingKvRouteConstraints, ServingMetricCeilings, ServingObjective, ServingPoolCandidate,
    ServingPoolDomainSpread, ServingPoolNodeFilter, ServingPoolSearch, ServingPrefillBatching,
    ServingRequestSlo, ServingRoutingPolicy, ServingSearchSpace, ServingServiceHealth,
    ServingServicePhaseConfig, ServingServicesConfig, ServingShapeProfile,
    ServingSloMissPenaltyWeights, ServingSloPolicy, ServingTraceRequest, ServingTraffic,
    ServingTrafficClass, ServingValueDistribution, SimulationCalibration, Solver,
    topology_graph::{RoutedResourceKind, TopologyGraph},
    types::{
        common::{
            Bandwidth, Bytes, GpuAddr, GpuId, Latency, NicId, OperationalState, UnorderedPair,
        },
        configs::RankPlacement,
        fabric::{
            inter_node::{
                CustomInterNodeLink, CustomInterNodeLinkEndpoints, FabricProfile, InterNodeTopology,
            },
            intra_node::{
                GpuNicAffinity, GpuNicPathOverride, IntraNodeTopology, NodeNetworkProfile,
            },
            variants::{
                eth::EthVariant, ib::IbVariant, nvlink::NvLinkVariant, pcie::PcieVariant,
                roce::RoceVariant,
            },
        },
        gpu::{Gpu, GpuProfile},
        topology::{Cluster, Node, NodeOperationalState, NodeTopologyMetadata},
    },
    workload::{DType, ExpertSpec, InferencePhase, InferenceRequest, ModelSpec},
};

const SUPPORTED_SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub struct ConfigError(String);

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ConfigError {}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkloadConfig {
    pub model_id: Option<String>,
    pub serving_stack: Option<String>,
    pub serving_runtime_features: Vec<String>,
    pub model: ModelSpec,
    pub request: InferenceRequest,
    pub search_space: SearchSpace,
    pub placement: Option<RankPlacement>,
    pub serving_prefill_placement: Option<RankPlacement>,
    pub serving_decode_placement: Option<RankPlacement>,
    pub calibration: SimulationCalibration,
    pub calibration_overrides: RunScenarioCalibrationConfig,
    pub calibration_policy: CalibrationPolicy,
    pub approximation_policy: ApproximationPolicy,
    pub calibration_profile: Option<CalibrationProfileMetadata>,
    pub calibration_coverage: Option<CalibrationCoverageReport>,
    pub calibration_warnings: Vec<CalibrationApplicabilityWarning>,
    pub calibration_invalid_shape_warnings: Vec<CalibrationInvalidShapeWarning>,
    pub calibration_gate_violations: Vec<CalibrationGateViolation>,
    pub require_routable_serving_pools: bool,
    pub serving: Option<DisaggregatedServingConfig>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunConfig {
    pub cluster_path: Option<PathBuf>,
    pub workload_path: Option<PathBuf>,
    pub output: RunOutputConfig,
    pub search_budget: RunSearchBudgetConfig,
    pub scenarios: Vec<RunScenarioConfig>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunOutputConfig {
    pub format: Option<String>,
    pub top_k: Option<usize>,
    pub output_dir: Option<PathBuf>,
    pub output_profile: Option<String>,
    pub request_metrics_csv_path: Option<PathBuf>,
    pub request_lifecycle_events_csv_path: Option<PathBuf>,
    pub serving_metrics_csv_path: Option<PathBuf>,
    pub serving_metric_breakdowns_csv_path: Option<PathBuf>,
    pub serving_services_csv_path: Option<PathBuf>,
    pub serving_utilization_csv_path: Option<PathBuf>,
    pub serving_memory_pressure_csv_path: Option<PathBuf>,
    pub serving_timeline_csv_path: Option<PathBuf>,
    pub serving_occupancy_csv_path: Option<PathBuf>,
    pub serving_placement_evidence_csv_path: Option<PathBuf>,
    pub serving_worker_evidence_csv_path: Option<PathBuf>,
    pub serving_rejections_csv_path: Option<PathBuf>,
    pub serving_route_paths_csv_path: Option<PathBuf>,
    pub kv_route_resources_csv_path: Option<PathBuf>,
    pub serving_bottlenecks_csv_path: Option<PathBuf>,
    pub serving_phase_calibration_csv_path: Option<PathBuf>,
    pub serving_approximations_csv_path: Option<PathBuf>,
    pub calibration_residuals_csv_path: Option<PathBuf>,
    pub scenario_sensitivity_csv_path: Option<PathBuf>,
    pub rank_sensitivity_csv_path: Option<PathBuf>,
    pub trace: Option<bool>,
    pub trace_limit: Option<usize>,
    pub request_limit: Option<usize>,
    pub occupancy: Option<bool>,
    pub occupancy_buckets: Option<usize>,
    pub occupancy_resource_limit: Option<usize>,
    pub critical_path: Option<bool>,
    pub critical_path_limit: Option<usize>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RunSearchBudgetConfig {
    pub max_parallelism_candidates: Option<usize>,
    pub max_prefill_candidates: Option<usize>,
    pub max_decode_candidates: Option<usize>,
    pub max_serving_pairs: Option<usize>,
    pub max_runtime_ms: Option<u64>,
    pub retain_rejected_candidates: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunScenarioConfig {
    pub name: String,
    pub request_count: Option<u32>,
    pub arrival_gap_scale: Option<f64>,
    pub arrival_rate_scale: Option<f64>,
    pub batch_size_scale: Option<f64>,
    pub prompt_tokens_scale: Option<f64>,
    pub decode_tokens_scale: Option<f64>,
    pub calibration_profile_path: Option<PathBuf>,
    pub calibration: RunScenarioCalibrationConfig,
    pub topology: RunScenarioTopologyConfig,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RunScenarioCalibrationConfig {
    pub compute_efficiency: Option<f64>,
    pub prefill_compute_scale: Option<f64>,
    pub decode_compute_scale: Option<f64>,
    pub decode_memory_bandwidth_scale: Option<f64>,
    pub collective_latency_scale: Option<f64>,
    pub collective_bandwidth_scale: Option<f64>,
    pub kv_transfer_scale: Option<f64>,
    pub scheduler_overhead_us: Option<f64>,
    pub serving_memory_temporary_fraction: Option<f64>,
    pub serving_memory_activation_communication_fraction: Option<f64>,
    pub serving_memory_weight_communication_fraction: Option<f64>,
    pub serving_memory_runtime_reserve_fraction: Option<f64>,
    pub serving_memory_fragmentation_fraction: Option<f64>,
    pub serving_pipeline_depth: Option<u32>,
    pub request_arrival_gap_s: Option<f64>,
    pub allow_compute_comm_overlap: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunScenarioTopologyConfig {
    pub interconnect_bandwidth_scale: Option<f64>,
    pub interconnect_latency_scale: Option<f64>,
    pub nic_bandwidth_scale: Option<f64>,
    pub node_states: Vec<RunScenarioNodeStateOverlay>,
    pub disabled_gpus: Vec<RunScenarioGpuResourceOverlay>,
    pub disabled_nics: Vec<RunScenarioNicResourceOverlay>,
    pub degraded_gpus: Vec<RunScenarioGpuDegradationOverlay>,
    pub degraded_nics: Vec<RunScenarioNicDegradationOverlay>,
    pub degraded_rails: Vec<RunScenarioRailDegradationOverlay>,
    pub degraded_links: Vec<RunScenarioLinkDegradationOverlay>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RunScenarioNodeState {
    Disabled,
    Maintenance,
    Draining,
    Reserved,
}

impl RunScenarioNodeState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Maintenance => "maintenance",
            Self::Draining => "draining",
            Self::Reserved => "reserved",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunScenarioNodeStateOverlay {
    pub node_ids: Vec<u32>,
    pub node_groups: Vec<String>,
    pub node_tags: Vec<String>,
    pub racks: Vec<String>,
    pub islands: Vec<String>,
    pub failure_domains: Vec<String>,
    pub state: RunScenarioNodeState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunScenarioGpuResourceOverlay {
    pub node_ids: Vec<u32>,
    pub node_groups: Vec<String>,
    pub node_tags: Vec<String>,
    pub racks: Vec<String>,
    pub islands: Vec<String>,
    pub failure_domains: Vec<String>,
    pub gpu_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunScenarioNicResourceOverlay {
    pub node_ids: Vec<u32>,
    pub node_groups: Vec<String>,
    pub node_tags: Vec<String>,
    pub racks: Vec<String>,
    pub islands: Vec<String>,
    pub failure_domains: Vec<String>,
    pub nic_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunScenarioGpuDegradationOverlay {
    pub node_ids: Vec<u32>,
    pub node_groups: Vec<String>,
    pub node_tags: Vec<String>,
    pub racks: Vec<String>,
    pub islands: Vec<String>,
    pub failure_domains: Vec<String>,
    pub gpu_ids: Vec<u32>,
    pub compute_scale: Option<f64>,
    pub hbm_bandwidth_scale: Option<f64>,
    pub hbm_capacity_scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunScenarioNicDegradationOverlay {
    pub node_ids: Vec<u32>,
    pub node_groups: Vec<String>,
    pub node_tags: Vec<String>,
    pub racks: Vec<String>,
    pub islands: Vec<String>,
    pub failure_domains: Vec<String>,
    pub nic_ids: Vec<u32>,
    pub bandwidth_scale: Option<f64>,
    pub latency_scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunScenarioRailDegradationOverlay {
    pub node_ids: Vec<u32>,
    pub node_groups: Vec<String>,
    pub node_tags: Vec<String>,
    pub racks: Vec<String>,
    pub islands: Vec<String>,
    pub failure_domains: Vec<String>,
    pub rails: Vec<u32>,
    pub bandwidth_scale: Option<f64>,
    pub latency_scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunScenarioLinkDegradationOverlay {
    pub from_node_ids: Vec<u32>,
    pub from_node_groups: Vec<String>,
    pub from_node_tags: Vec<String>,
    pub from_racks: Vec<String>,
    pub from_islands: Vec<String>,
    pub from_failure_domains: Vec<String>,
    pub from_gpus: Vec<u32>,
    pub to_node_ids: Vec<u32>,
    pub to_node_groups: Vec<String>,
    pub to_node_tags: Vec<String>,
    pub to_racks: Vec<String>,
    pub to_islands: Vec<String>,
    pub to_failure_domains: Vec<String>,
    pub to_gpus: Vec<u32>,
    pub rails: Vec<u32>,
    pub bandwidth_scale: Option<f64>,
    pub latency_scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationApplicabilityWarning {
    pub field: String,
    pub observed_min: u32,
    pub observed_max: u32,
    pub calibrated_min: Option<u32>,
    pub calibrated_max: Option<u32>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationProfileMetadata {
    pub path: String,
    pub name: Option<String>,
    pub hardware: Option<String>,
    pub fabric: Option<String>,
    pub model: Option<String>,
    pub dtype: Option<String>,
    pub serving_stack: Option<String>,
    pub serving_runtime_features: Vec<String>,
    pub backend_version: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub rocm_version: Option<String>,
    pub nccl_version: Option<String>,
    pub rccl_version: Option<String>,
    pub ucx_version: Option<String>,
    pub kernel_settings: Vec<String>,
    pub environment_hash: Option<String>,
    pub source: Option<String>,
    pub date: Option<String>,
    pub notes: Option<String>,
    pub valid_shape: Option<CalibrationShapeRange>,
    pub invalid_shapes: Vec<CalibrationInvalidShapeRange>,
    pub fits: Vec<CalibrationFittedModel>,
    pub benchmarks: Vec<CalibrationBenchmarkPoint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationShapeRange {
    pub min_batch_size: Option<u32>,
    pub max_batch_size: Option<u32>,
    pub min_prompt_tokens: Option<u32>,
    pub max_prompt_tokens: Option<u32>,
    pub min_decode_tokens: Option<u32>,
    pub max_decode_tokens: Option<u32>,
    pub min_sequence_tokens: Option<u32>,
    pub max_sequence_tokens: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationInvalidShapeRange {
    pub name: Option<String>,
    pub reason: Option<String>,
    pub shape: CalibrationShapeRange,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationInvalidShapeWarning {
    pub name: Option<String>,
    pub reason: Option<String>,
    pub shape: CalibrationShapeRange,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationCoverageReport {
    pub benchmark_count: usize,
    pub shape_benchmark_count: usize,
    pub complete_shape_benchmark_count: usize,
    pub required_phases: Vec<String>,
    pub covered_phases: Vec<String>,
    pub missing_phases: Vec<String>,
    pub batch_size_score: Option<f64>,
    pub prompt_tokens_score: Option<f64>,
    pub decode_tokens_score: Option<f64>,
    pub sequence_tokens_score: Option<f64>,
    pub shape_coverage_score: Option<f64>,
    pub phase_coverage_score: Option<f64>,
    pub coverage_score: Option<f64>,
    pub nearest_benchmark: Option<String>,
    pub nearest_benchmark_distance: Option<f64>,
    pub status: String,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CalibrationGateMode {
    #[default]
    Warn,
    Reject,
}

impl CalibrationGateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warn => "warn",
            Self::Reject => "reject",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationPolicy {
    pub valid_shape: CalibrationGateMode,
    pub invalid_shape: CalibrationGateMode,
    pub coverage: CalibrationGateMode,
    pub fit_confidence: CalibrationGateMode,
    pub fit_extrapolation: CalibrationGateMode,
    pub fit_partially_bounded: CalibrationGateMode,
    pub fit_unbounded: CalibrationGateMode,
    pub fit_sample_count: CalibrationGateMode,
    pub fit_validation_sample_count: CalibrationGateMode,
    pub fit_source: CalibrationGateMode,
    pub fit_uncertainty: CalibrationGateMode,
    pub profile_source: CalibrationGateMode,
    pub profile_date: CalibrationGateMode,
    pub profile_runtime: CalibrationGateMode,
    pub min_coverage_score: Option<f64>,
    pub min_fit_confidence_score: Option<f64>,
    pub min_fit_confidence_level: Option<f64>,
    pub min_fit_sample_count: Option<u32>,
    pub min_fit_validation_sample_count: Option<u32>,
    pub max_fit_relative_uncertainty_pct: Option<f64>,
    pub max_fit_absolute_uncertainty_s: Option<f64>,
    pub min_serving_phase_coverage_fraction: Option<f64>,
    pub uncertainty_ranking_weight: f64,
    pub require_phase_coverage: bool,
}

impl Default for CalibrationPolicy {
    fn default() -> Self {
        Self {
            valid_shape: CalibrationGateMode::Warn,
            invalid_shape: CalibrationGateMode::Warn,
            coverage: CalibrationGateMode::Warn,
            fit_confidence: CalibrationGateMode::Warn,
            fit_extrapolation: CalibrationGateMode::Warn,
            fit_partially_bounded: CalibrationGateMode::Warn,
            fit_unbounded: CalibrationGateMode::Warn,
            fit_sample_count: CalibrationGateMode::Warn,
            fit_validation_sample_count: CalibrationGateMode::Warn,
            fit_source: CalibrationGateMode::Warn,
            fit_uncertainty: CalibrationGateMode::Warn,
            profile_source: CalibrationGateMode::Warn,
            profile_date: CalibrationGateMode::Warn,
            profile_runtime: CalibrationGateMode::Warn,
            min_coverage_score: Some(0.5),
            min_fit_confidence_score: Some(0.5),
            min_fit_confidence_level: None,
            min_fit_sample_count: None,
            min_fit_validation_sample_count: None,
            max_fit_relative_uncertainty_pct: None,
            max_fit_absolute_uncertainty_s: None,
            min_serving_phase_coverage_fraction: None,
            uncertainty_ranking_weight: 0.0,
            require_phase_coverage: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApproximationPolicy {
    pub preset: Option<ApproximationPolicyPreset>,
    pub default_action: CalibrationGateMode,
    pub reject_categories: Vec<String>,
    pub reject_codes: Vec<String>,
    pub warn_categories: Vec<String>,
    pub warn_codes: Vec<String>,
    pub metric_gates: Vec<ApproximationMetricGate>,
}

impl Default for ApproximationPolicy {
    fn default() -> Self {
        Self {
            preset: None,
            default_action: CalibrationGateMode::Warn,
            reject_categories: Vec::new(),
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApproximationMetricGate {
    pub metrics: Vec<String>,
    pub reject_categories: Vec<String>,
    pub reject_codes: Vec<String>,
    pub warn_categories: Vec<String>,
    pub warn_codes: Vec<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ApproximationPolicyPreset {
    MvpExploration,
    TopologySensitive,
    MemoryCapacity,
    CalibrationOnly,
    ProductionRecommendation,
}

impl ApproximationPolicyPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MvpExploration => "mvp_exploration",
            Self::TopologySensitive => "topology_sensitive",
            Self::MemoryCapacity => "memory_capacity",
            Self::CalibrationOnly => "calibration_only",
            Self::ProductionRecommendation => "production_recommendation",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApproximationPolicyViolation {
    pub metric: Option<String>,
    pub phase: String,
    pub category: String,
    pub scope: String,
    pub code: String,
    pub action: CalibrationGateMode,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationGateViolation {
    pub code: String,
    pub action: CalibrationGateMode,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationFittedModel {
    pub name: Option<String>,
    pub target: String,
    pub phase: Option<String>,
    pub kind: Option<String>,
    pub model: String,
    pub unit: Option<String>,
    pub intercept: Option<f64>,
    pub features: Vec<String>,
    pub coefficients: Vec<f64>,
    pub feature_ranges: Vec<CalibrationFitFeatureRange>,
    pub r_squared: Option<f64>,
    pub adjusted_r_squared: Option<f64>,
    pub rmse: Option<f64>,
    pub rmse_pct: Option<f64>,
    pub mean_abs_pct_error: Option<f64>,
    pub max_abs_pct_error: Option<f64>,
    pub validation_rmse: Option<f64>,
    pub validation_rmse_pct: Option<f64>,
    pub validation_mean_abs_pct_error: Option<f64>,
    pub validation_max_abs_pct_error: Option<f64>,
    pub confidence_interval: Option<f64>,
    pub confidence_interval_pct: Option<f64>,
    pub confidence_level: Option<f64>,
    pub sample_count: Option<u32>,
    pub validation_sample_count: Option<u32>,
    pub source: Option<String>,
    pub notes: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationFitFeatureRange {
    pub feature: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationBenchmarkPoint {
    pub name: Option<String>,
    pub kind: Option<String>,
    pub phase: Option<String>,
    pub hardware: Option<String>,
    pub fabric: Option<String>,
    pub model: Option<String>,
    pub dtype: Option<String>,
    pub batch_size: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub decode_tokens: Option<u32>,
    pub sequence_tokens: Option<u32>,
    pub tensor_ranks: Option<u32>,
    pub pipeline_ranks: Option<u32>,
    pub expert_ranks: Option<u32>,
    pub data_ranks: Option<u32>,
    pub measured_ms: Option<f64>,
    pub predicted_ms: Option<f64>,
    pub throughput_tokens_per_s: Option<f64>,
    pub command: Option<String>,
    pub source: Option<String>,
    pub notes: Option<String>,
}

#[derive(Deserialize)]
struct ClusterFile {
    schema_version: Option<u32>,
    cluster: ClusterSection,
    interconnect: Option<InterconnectSection>,
    nics: Option<NicsSection>,
    nodes: Option<Vec<NodeSection>>,
    node_groups: Option<Vec<NodeGroupSection>>,
}

#[derive(Deserialize)]
struct ClusterSection {
    preset: String,
    node_count: Option<u32>,
}

#[derive(Deserialize)]
struct InterconnectSection {
    kind: Option<String>,
    variant: Option<String>,
    oversubscription: Option<f64>,
    links: Option<Vec<InterconnectLinkSection>>,
}

#[derive(Deserialize)]
struct InterconnectLinkSection {
    from: Option<u32>,
    to: Option<u32>,
    from_group: Option<String>,
    to_group: Option<String>,
    #[serde(alias = "from_node_label", alias = "from_node_tag")]
    from_node_tag: Option<String>,
    #[serde(
        alias = "from_node_labels",
        alias = "from_node_tags",
        alias = "from_labels",
        alias = "from_tags"
    )]
    from_node_tags: Option<Vec<String>>,
    #[serde(alias = "from_rack_id")]
    from_rack: Option<String>,
    #[serde(alias = "from_rack_ids")]
    from_racks: Option<Vec<String>>,
    #[serde(alias = "from_island_id", alias = "from_topology_domain")]
    from_island: Option<String>,
    #[serde(alias = "from_island_ids", alias = "from_topology_domains")]
    from_islands: Option<Vec<String>>,
    #[serde(alias = "from_failure_domain_id")]
    from_failure_domain: Option<String>,
    #[serde(alias = "from_failure_domain_ids")]
    from_failure_domains: Option<Vec<String>>,
    #[serde(alias = "to_node_label", alias = "to_node_tag")]
    to_node_tag: Option<String>,
    #[serde(
        alias = "to_node_labels",
        alias = "to_node_tags",
        alias = "to_labels",
        alias = "to_tags"
    )]
    to_node_tags: Option<Vec<String>>,
    #[serde(alias = "to_rack_id")]
    to_rack: Option<String>,
    #[serde(alias = "to_rack_ids")]
    to_racks: Option<Vec<String>>,
    #[serde(alias = "to_island_id", alias = "to_topology_domain")]
    to_island: Option<String>,
    #[serde(alias = "to_island_ids", alias = "to_topology_domains")]
    to_islands: Option<Vec<String>>,
    #[serde(alias = "to_failure_domain_id")]
    to_failure_domain: Option<String>,
    #[serde(alias = "to_failure_domain_ids")]
    to_failure_domains: Option<Vec<String>>,
    #[serde(alias = "from_gpu", alias = "from_gpu_id", alias = "from_local_gpu")]
    from_gpu: Option<u32>,
    #[serde(
        alias = "from_gpu_ids",
        alias = "from_local_gpus",
        alias = "from_local_gpu_ids"
    )]
    from_gpus: Option<Vec<u32>>,
    #[serde(alias = "from_gpu_label", alias = "from_gpu_type")]
    from_gpu_type: Option<String>,
    #[serde(alias = "from_gpu_labels", alias = "from_gpu_types")]
    from_gpu_types: Option<Vec<String>>,
    #[serde(alias = "from_gpu_topology_label")]
    from_gpu_tag: Option<String>,
    #[serde(alias = "from_gpu_topology_labels")]
    from_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "to_gpu", alias = "to_gpu_id", alias = "to_local_gpu")]
    to_gpu: Option<u32>,
    #[serde(
        alias = "to_gpu_ids",
        alias = "to_local_gpus",
        alias = "to_local_gpu_ids"
    )]
    to_gpus: Option<Vec<u32>>,
    #[serde(alias = "to_gpu_label", alias = "to_gpu_type")]
    to_gpu_type: Option<String>,
    #[serde(alias = "to_gpu_labels", alias = "to_gpu_types")]
    to_gpu_types: Option<Vec<String>>,
    #[serde(alias = "to_gpu_topology_label")]
    to_gpu_tag: Option<String>,
    #[serde(alias = "to_gpu_topology_labels")]
    to_gpu_tags: Option<Vec<String>>,
    kind: String,
    variant: String,
    oversubscription: Option<f64>,
    rail: Option<u32>,
    rails: Option<Vec<u32>>,
}

#[derive(Clone, Deserialize)]
struct NicsSection {
    count: Option<u8>,
    bandwidth_gbps: Option<f64>,
    affinity: Option<String>,
    gpus_per_nic: Option<u8>,
    rail_count: Option<u8>,
    #[serde(alias = "nic_rails", alias = "nic_to_rail_map", alias = "rail_map")]
    nic_rail_map: Option<Vec<NicRailMapSection>>,
    #[serde(alias = "gpu_to_nic_map", alias = "gpu_nic_locality")]
    gpu_nic_map: Option<Vec<GpuNicMapSection>>,
    #[serde(
        alias = "gpu_socket_map",
        alias = "gpu_numa_domains",
        alias = "gpu_socket_domains"
    )]
    gpu_numa_map: Option<Vec<GpuNumaMapSection>>,
    #[serde(
        alias = "nic_socket_map",
        alias = "nic_numa_domains",
        alias = "nic_socket_domains"
    )]
    nic_numa_map: Option<Vec<NicNumaMapSection>>,
    #[serde(alias = "cross_socket_bandwidth_scale")]
    cross_numa_bandwidth_scale: Option<f64>,
    #[serde(alias = "cross_socket_latency_scale")]
    cross_numa_latency_scale: Option<f64>,
    #[serde(
        alias = "gpu_nic_path_overrides",
        alias = "gpu_to_nic_paths",
        alias = "gpu_nic_locality_overrides"
    )]
    gpu_nic_paths: Option<Vec<GpuNicPathSection>>,
    #[serde(alias = "bandwidth_overrides", alias = "per_nic_bandwidth")]
    nic_bandwidth_overrides: Option<Vec<NicBandwidthOverrideSection>>,
    #[serde(alias = "latency_overrides", alias = "per_nic_latency")]
    nic_latency_scale_overrides: Option<Vec<NicLatencyScaleOverrideSection>>,
    #[serde(alias = "offline_nics", alias = "unavailable_nics")]
    disabled_nics: Option<Vec<u32>>,
    #[serde(alias = "nic_state", alias = "nics_state", alias = "nic_health")]
    nic_states: Option<Vec<NicStateSection>>,
}

#[derive(Clone, Deserialize)]
struct NicRailMapSection {
    #[serde(alias = "nic_id", alias = "id")]
    nic: Option<u32>,
    #[serde(alias = "rail_id")]
    rail: Option<u32>,
}

#[derive(Clone, Deserialize)]
struct GpuNicMapSection {
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_id")]
    local_gpu_id: Option<u32>,
    nic: Option<u32>,
    nics: Option<Vec<u32>>,
}

#[derive(Clone, Deserialize)]
struct GpuNumaMapSection {
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_id")]
    local_gpu_id: Option<u32>,
    #[serde(alias = "gpus", alias = "gpu_ids", alias = "local_gpu_ids")]
    local_gpu_ids: Option<Vec<u32>>,
    #[serde(
        alias = "domain",
        alias = "numa",
        alias = "numa_id",
        alias = "socket",
        alias = "socket_id"
    )]
    numa_domain: u32,
}

#[derive(Clone, Deserialize)]
struct NicNumaMapSection {
    #[serde(alias = "nic_id", alias = "id")]
    nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    nics: Option<Vec<u32>>,
    #[serde(
        alias = "domain",
        alias = "numa",
        alias = "numa_id",
        alias = "socket",
        alias = "socket_id"
    )]
    numa_domain: u32,
}

#[derive(Clone, Deserialize)]
struct GpuNicPathSection {
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_id")]
    local_gpu_id: Option<u32>,
    nic: Option<u32>,
    label: Option<String>,
    bandwidth_gbps: Option<f64>,
    latency_us: Option<f64>,
    gpudirect: Option<bool>,
    available: Option<bool>,
}

#[derive(Clone, Deserialize)]
struct NicStateSection {
    #[serde(alias = "nic_id", alias = "id")]
    nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    nics: Option<Vec<u32>>,
    state: String,
}

#[derive(Clone, Deserialize)]
struct NicBandwidthOverrideSection {
    #[serde(alias = "nic_id", alias = "id")]
    nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    nics: Option<Vec<u32>>,
    bandwidth_gbps: f64,
}

#[derive(Clone, Deserialize)]
struct NicLatencyScaleOverrideSection {
    #[serde(alias = "nic_id", alias = "id")]
    nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    nics: Option<Vec<u32>>,
    #[serde(alias = "scale")]
    latency_scale: f64,
}

#[derive(Clone, Deserialize)]
struct NodeSection {
    id: u32,
    group: Option<String>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(
        alias = "labels",
        alias = "node_labels",
        alias = "tags",
        alias = "node_tags"
    )]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    gpu: Option<String>,
    gpu_count: Option<u32>,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    peak_f8_tflops: Option<f64>,
    #[serde(alias = "gpu_label", alias = "gpu_tag")]
    gpu_tag: Option<String>,
    #[serde(alias = "gpu_labels", alias = "gpu_tags")]
    gpu_tags: Option<Vec<String>>,
    gpus: Option<Vec<NodeGpuSection>>,
    #[serde(alias = "gpu_overrides", alias = "per_gpu_profiles")]
    gpu_profile_overrides: Option<Vec<GpuProfileOverrideSection>>,
    #[serde(alias = "offline_gpus", alias = "unavailable_gpus")]
    disabled_gpus: Option<Vec<u32>>,
    #[serde(alias = "gpu_state", alias = "gpus_state", alias = "gpu_health")]
    gpu_states: Option<Vec<GpuStateSection>>,
    intra: Option<String>,
    nics: Option<NicsSection>,
}

#[derive(Clone, Deserialize)]
struct NodeGpuSection {
    #[serde(alias = "local_id", alias = "local_gpu_id")]
    id: Option<u32>,
    #[serde(alias = "start", alias = "start_id", alias = "id_start")]
    start_id: Option<u32>,
    #[serde(alias = "type")]
    gpu: String,
    count: Option<u32>,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    peak_f8_tflops: Option<f64>,
    #[serde(alias = "label", alias = "gpu_label", alias = "tag", alias = "gpu_tag")]
    gpu_tag: Option<String>,
    #[serde(
        alias = "labels",
        alias = "gpu_labels",
        alias = "tags",
        alias = "gpu_tags"
    )]
    gpu_tags: Option<Vec<String>>,
    #[serde(alias = "health")]
    state: Option<String>,
}

#[derive(Clone, Deserialize)]
struct GpuProfileOverrideSection {
    #[serde(
        alias = "gpu",
        alias = "gpu_id",
        alias = "local_id",
        alias = "local_gpu_id"
    )]
    local_gpu_id: Option<u32>,
    #[serde(alias = "gpus", alias = "gpu_ids", alias = "local_gpu_ids")]
    local_gpu_ids: Option<Vec<u32>>,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    peak_f8_tflops: Option<f64>,
}

#[derive(Clone, Deserialize)]
struct GpuStateSection {
    #[serde(
        alias = "gpu",
        alias = "gpu_id",
        alias = "local_id",
        alias = "local_gpu_id"
    )]
    local_gpu_id: Option<u32>,
    #[serde(alias = "gpus", alias = "gpu_ids", alias = "local_gpu_ids")]
    local_gpu_ids: Option<Vec<u32>>,
    state: String,
}

#[derive(Clone, Deserialize)]
struct NodeGroupSection {
    label: Option<String>,
    #[serde(alias = "node_count")]
    count: u32,
    #[serde(alias = "id_start")]
    start_id: Option<u32>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(
        alias = "labels",
        alias = "node_labels",
        alias = "tags",
        alias = "node_tags"
    )]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    gpu: String,
    gpu_count: u32,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    peak_f8_tflops: Option<f64>,
    #[serde(alias = "gpu_label", alias = "gpu_tag")]
    gpu_tag: Option<String>,
    #[serde(alias = "gpu_labels", alias = "gpu_tags")]
    gpu_tags: Option<Vec<String>>,
    #[serde(alias = "gpu_overrides", alias = "per_gpu_profiles")]
    gpu_profile_overrides: Option<Vec<GpuProfileOverrideSection>>,
    #[serde(alias = "offline_gpus", alias = "unavailable_gpus")]
    disabled_gpus: Option<Vec<u32>>,
    #[serde(alias = "gpu_state", alias = "gpus_state", alias = "gpu_health")]
    gpu_states: Option<Vec<GpuStateSection>>,
    intra: Option<String>,
    nics: Option<NicsSection>,
}

#[derive(Deserialize)]
struct WorkloadFile {
    schema_version: Option<u32>,
    #[serde(alias = "runtime", alias = "backend", alias = "stack")]
    serving_stack: Option<String>,
    #[serde(
        alias = "runtime_features",
        alias = "backend_features",
        alias = "stack_features"
    )]
    serving_runtime_features: Option<Vec<String>>,
    model: ModelSection,
    request: RequestSection,
    search: Option<SearchSection>,
    placement: Option<PlacementSection>,
    serving: Option<ServingSection>,
    calibration_profile: Option<CalibrationProfileReferenceSection>,
    calibration: Option<CalibrationSection>,
    calibration_policy: Option<CalibrationPolicySection>,
    approximation_policy: Option<ApproximationPolicySection>,
}

#[derive(Deserialize)]
struct CalibrationProfileReferenceSection {
    path: String,
}

#[derive(Deserialize)]
struct RunFile {
    schema_version: Option<u32>,
    cluster: Option<String>,
    workload: Option<String>,
    run: Option<RunSection>,
    output: Option<RunOutputSection>,
    #[serde(alias = "budget", alias = "search_budget")]
    search: Option<RunSearchBudgetSection>,
    scenarios: Option<Vec<RunScenarioSection>>,
}

#[derive(Deserialize)]
struct RunSection {
    #[serde(alias = "cluster_path")]
    cluster: Option<String>,
    #[serde(alias = "workload_path")]
    workload: Option<String>,
}

#[derive(Deserialize)]
struct RunOutputSection {
    format: Option<String>,
    top_k: Option<usize>,
    #[serde(alias = "dir", alias = "directory")]
    output_dir: Option<String>,
    output_profile: Option<String>,
    #[serde(
        alias = "request_metrics_csv",
        alias = "serving_request_metrics_csv",
        alias = "metrics_csv"
    )]
    request_metrics_csv_path: Option<String>,
    #[serde(
        alias = "request_lifecycle_events_csv",
        alias = "serving_request_lifecycle_events_csv",
        alias = "lifecycle_events_csv",
        alias = "serving_lifecycle_events_csv"
    )]
    request_lifecycle_events_csv_path: Option<String>,
    #[serde(
        alias = "serving_metrics_csv",
        alias = "serving_candidate_metrics_csv",
        alias = "candidate_metrics_csv"
    )]
    serving_metrics_csv_path: Option<String>,
    #[serde(
        alias = "serving_metric_breakdowns_csv",
        alias = "serving_breakdowns_csv",
        alias = "metric_breakdowns_csv",
        alias = "breakdowns_csv"
    )]
    serving_metric_breakdowns_csv_path: Option<String>,
    #[serde(
        alias = "serving_services_csv",
        alias = "serving_service_metrics_csv",
        alias = "service_metrics_csv",
        alias = "services_csv"
    )]
    serving_services_csv_path: Option<String>,
    #[serde(
        alias = "serving_utilization_csv",
        alias = "serving_resource_utilization_csv",
        alias = "resource_utilization_csv",
        alias = "utilization_csv"
    )]
    serving_utilization_csv_path: Option<String>,
    #[serde(
        alias = "serving_memory_pressure_csv",
        alias = "serving_hbm_pressure_csv",
        alias = "memory_pressure_csv",
        alias = "hbm_pressure_csv"
    )]
    serving_memory_pressure_csv_path: Option<String>,
    #[serde(
        alias = "serving_timeline_csv",
        alias = "serving_scheduled_operations_csv",
        alias = "scheduled_operations_csv",
        alias = "timeline_csv"
    )]
    serving_timeline_csv_path: Option<String>,
    #[serde(
        alias = "serving_occupancy_csv",
        alias = "serving_resource_occupancy_csv",
        alias = "resource_occupancy_csv",
        alias = "occupancy_csv"
    )]
    serving_occupancy_csv_path: Option<String>,
    #[serde(
        alias = "serving_placement_evidence_csv",
        alias = "serving_placement_csv",
        alias = "placement_evidence_csv",
        alias = "placement_csv"
    )]
    serving_placement_evidence_csv_path: Option<String>,
    #[serde(
        alias = "serving_worker_evidence_csv",
        alias = "serving_worker_assignments_csv",
        alias = "worker_evidence_csv",
        alias = "worker_assignments_csv"
    )]
    serving_worker_evidence_csv_path: Option<String>,
    #[serde(
        alias = "serving_rejections_csv",
        alias = "serving_rejection_evidence_csv",
        alias = "rejections_csv",
        alias = "rejection_evidence_csv"
    )]
    serving_rejections_csv_path: Option<String>,
    #[serde(
        alias = "serving_route_paths_csv",
        alias = "serving_kv_route_paths_csv",
        alias = "kv_route_paths_csv",
        alias = "route_paths_csv"
    )]
    serving_route_paths_csv_path: Option<String>,
    #[serde(
        alias = "kv_route_resources_csv",
        alias = "kv_route_resource_csv",
        alias = "route_resources_csv",
        alias = "serving_route_resources_csv"
    )]
    kv_route_resources_csv_path: Option<String>,
    #[serde(
        alias = "serving_bottlenecks_csv",
        alias = "serving_bottleneck_summary_csv",
        alias = "bottlenecks_csv"
    )]
    serving_bottlenecks_csv_path: Option<String>,
    #[serde(
        alias = "serving_phase_calibration_csv",
        alias = "serving_calibration_phases_csv",
        alias = "phase_calibration_csv"
    )]
    serving_phase_calibration_csv_path: Option<String>,
    #[serde(
        alias = "serving_approximations_csv",
        alias = "serving_approximation_evidence_csv",
        alias = "approximations_csv",
        alias = "approximation_evidence_csv"
    )]
    serving_approximations_csv_path: Option<String>,
    #[serde(
        alias = "calibration_residuals_csv",
        alias = "calibration_benchmark_residuals_csv",
        alias = "calibration_benchmarks_csv"
    )]
    calibration_residuals_csv_path: Option<String>,
    #[serde(
        alias = "scenario_sensitivity_csv",
        alias = "serving_scenario_sensitivity_csv",
        alias = "sensitivity_csv"
    )]
    scenario_sensitivity_csv_path: Option<String>,
    #[serde(
        alias = "rank_sensitivity_csv",
        alias = "solver_rank_sensitivity_csv",
        alias = "serving_rank_sensitivity_csv"
    )]
    rank_sensitivity_csv_path: Option<String>,
    trace: Option<bool>,
    trace_limit: Option<usize>,
    request_limit: Option<usize>,
    occupancy: Option<bool>,
    occupancy_buckets: Option<usize>,
    occupancy_resource_limit: Option<usize>,
    critical_path: Option<bool>,
    critical_path_limit: Option<usize>,
}

#[derive(Deserialize)]
struct RunSearchBudgetSection {
    #[serde(
        alias = "max_candidates",
        alias = "max_configs",
        alias = "max_rank_configs"
    )]
    max_parallelism_candidates: Option<usize>,
    #[serde(alias = "max_prefill_configs")]
    max_prefill_candidates: Option<usize>,
    #[serde(alias = "max_decode_configs")]
    max_decode_candidates: Option<usize>,
    #[serde(alias = "max_pairs", alias = "max_pool_pairs")]
    max_serving_pairs: Option<usize>,
    #[serde(
        alias = "max_runtime_milliseconds",
        alias = "max_search_runtime_ms",
        alias = "max_search_runtime_milliseconds"
    )]
    max_runtime_ms: Option<u64>,
    #[serde(
        alias = "retain_rejected",
        alias = "include_rejected_candidates",
        alias = "include_rejections",
        alias = "keep_rejected_candidates"
    )]
    retain_rejected_candidates: Option<bool>,
}

#[derive(Deserialize)]
struct RunScenarioSection {
    name: Option<String>,
    #[serde(alias = "requests")]
    request_count: Option<u32>,
    #[serde(alias = "interarrival_scale")]
    arrival_gap_scale: Option<f64>,
    #[serde(alias = "traffic_rate_scale", alias = "qps_scale")]
    arrival_rate_scale: Option<f64>,
    batch_size_scale: Option<f64>,
    prompt_tokens_scale: Option<f64>,
    decode_tokens_scale: Option<f64>,
    #[serde(alias = "calibration_profile_path")]
    calibration_profile: Option<String>,
    calibration: Option<CalibrationSection>,
    topology: Option<RunScenarioTopologySection>,
}

#[derive(Deserialize)]
struct RunScenarioTopologySection {
    #[serde(alias = "fabric_bandwidth_scale")]
    interconnect_bandwidth_scale: Option<f64>,
    #[serde(alias = "fabric_latency_scale")]
    interconnect_latency_scale: Option<f64>,
    #[serde(alias = "node_nic_bandwidth_scale")]
    nic_bandwidth_scale: Option<f64>,
    #[serde(
        alias = "node_state",
        alias = "nodes",
        alias = "node_overlays",
        alias = "unavailable_nodes",
        alias = "offline_nodes"
    )]
    node_states: Option<Vec<RunScenarioNodeStateOverlaySection>>,
    #[serde(alias = "offline_gpus", alias = "unavailable_gpus")]
    disabled_gpus: Option<Vec<RunScenarioGpuResourceOverlaySection>>,
    #[serde(alias = "offline_nics", alias = "unavailable_nics")]
    disabled_nics: Option<Vec<RunScenarioNicResourceOverlaySection>>,
    degraded_gpus: Option<Vec<RunScenarioGpuDegradationOverlaySection>>,
    degraded_nics: Option<Vec<RunScenarioNicDegradationOverlaySection>>,
    #[serde(alias = "rail_degradations", alias = "degraded_fabric_rails")]
    degraded_rails: Option<Vec<RunScenarioRailDegradationOverlaySection>>,
    degraded_links: Option<Vec<RunScenarioLinkDegradationOverlaySection>>,
}

#[derive(Deserialize)]
struct RunScenarioNodeStateOverlaySection {
    #[serde(alias = "node_id")]
    node: Option<u32>,
    #[serde(alias = "node_ids")]
    nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    group: Option<String>,
    #[serde(alias = "node_groups")]
    groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "rack_ids")]
    racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    failure_domains: Option<Vec<String>>,
    #[serde(alias = "status", alias = "health")]
    state: Option<String>,
}

#[derive(Deserialize)]
struct RunScenarioGpuResourceOverlaySection {
    #[serde(alias = "node_id")]
    node: Option<u32>,
    #[serde(alias = "node_ids")]
    nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    group: Option<String>,
    #[serde(alias = "node_groups")]
    groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "rack_ids")]
    racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    failure_domains: Option<Vec<String>>,
    #[serde(alias = "gpu_id", alias = "local_gpu_id")]
    gpu: Option<u32>,
    #[serde(alias = "gpu_ids", alias = "local_gpu_ids", alias = "local_gpus")]
    gpus: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct RunScenarioNicResourceOverlaySection {
    #[serde(alias = "node_id")]
    node: Option<u32>,
    #[serde(alias = "node_ids")]
    nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    group: Option<String>,
    #[serde(alias = "node_groups")]
    groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "rack_ids")]
    racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    failure_domains: Option<Vec<String>>,
    #[serde(alias = "nic_id")]
    nic: Option<u32>,
    #[serde(alias = "nic_ids")]
    nics: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct RunScenarioGpuDegradationOverlaySection {
    #[serde(alias = "node_id")]
    node: Option<u32>,
    #[serde(alias = "node_ids")]
    nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    group: Option<String>,
    #[serde(alias = "node_groups")]
    groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "rack_ids")]
    racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    failure_domains: Option<Vec<String>>,
    #[serde(alias = "gpu_id", alias = "local_gpu_id")]
    gpu: Option<u32>,
    #[serde(alias = "gpu_ids", alias = "local_gpu_ids", alias = "local_gpus")]
    gpus: Option<Vec<u32>>,
    #[serde(alias = "flops_scale", alias = "peak_flops_scale")]
    compute_scale: Option<f64>,
    #[serde(alias = "memory_bandwidth_scale")]
    hbm_bandwidth_scale: Option<f64>,
    #[serde(alias = "hbm_size_scale", alias = "memory_capacity_scale")]
    hbm_capacity_scale: Option<f64>,
}

#[derive(Deserialize)]
struct RunScenarioNicDegradationOverlaySection {
    #[serde(alias = "node_id")]
    node: Option<u32>,
    #[serde(alias = "node_ids")]
    nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    group: Option<String>,
    #[serde(alias = "node_groups")]
    groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "rack_ids")]
    racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    failure_domains: Option<Vec<String>>,
    #[serde(alias = "nic_id")]
    nic: Option<u32>,
    #[serde(alias = "nic_ids")]
    nics: Option<Vec<u32>>,
    #[serde(alias = "scale", alias = "nic_bandwidth_scale")]
    bandwidth_scale: Option<f64>,
    #[serde(alias = "nic_latency_scale")]
    latency_scale: Option<f64>,
}

#[derive(Deserialize)]
struct RunScenarioRailDegradationOverlaySection {
    #[serde(alias = "node_id")]
    node: Option<u32>,
    #[serde(alias = "node_ids")]
    nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    group: Option<String>,
    #[serde(alias = "node_groups")]
    groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    rack: Option<String>,
    #[serde(alias = "rack_ids")]
    racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    failure_domains: Option<Vec<String>>,
    rail: Option<u32>,
    rails: Option<Vec<u32>>,
    #[serde(
        alias = "scale",
        alias = "rail_bandwidth_scale",
        alias = "fabric_bandwidth_scale",
        alias = "nic_bandwidth_scale"
    )]
    bandwidth_scale: Option<f64>,
    #[serde(
        alias = "rail_latency_scale",
        alias = "fabric_latency_scale",
        alias = "nic_latency_scale"
    )]
    latency_scale: Option<f64>,
}

#[derive(Deserialize)]
struct RunScenarioLinkDegradationOverlaySection {
    #[serde(alias = "from_node", alias = "from_node_id")]
    from: Option<u32>,
    #[serde(alias = "from_node_ids")]
    from_nodes: Option<Vec<u32>>,
    #[serde(alias = "from_node_group")]
    from_group: Option<String>,
    #[serde(alias = "from_node_groups")]
    from_groups: Option<Vec<String>>,
    #[serde(alias = "from_node_label", alias = "from_node_tag")]
    from_node_tag: Option<String>,
    #[serde(
        alias = "from_node_labels",
        alias = "from_node_label_ids",
        alias = "from_node_tags"
    )]
    from_node_tags: Option<Vec<String>>,
    #[serde(alias = "from_rack_id")]
    from_rack: Option<String>,
    #[serde(alias = "from_rack_ids")]
    from_racks: Option<Vec<String>>,
    #[serde(alias = "from_island_id", alias = "from_topology_domain")]
    from_island: Option<String>,
    #[serde(alias = "from_island_ids", alias = "from_topology_domains")]
    from_islands: Option<Vec<String>>,
    #[serde(alias = "from_failure_domain_id")]
    from_failure_domain: Option<String>,
    #[serde(alias = "from_failure_domain_ids")]
    from_failure_domains: Option<Vec<String>>,
    #[serde(alias = "from_gpu", alias = "from_gpu_id", alias = "from_local_gpu")]
    from_gpu: Option<u32>,
    #[serde(
        alias = "from_gpu_ids",
        alias = "from_local_gpus",
        alias = "from_local_gpu_ids"
    )]
    from_gpus: Option<Vec<u32>>,
    #[serde(alias = "to_node", alias = "to_node_id")]
    to: Option<u32>,
    #[serde(alias = "to_node_ids")]
    to_nodes: Option<Vec<u32>>,
    #[serde(alias = "to_node_group")]
    to_group: Option<String>,
    #[serde(alias = "to_node_groups")]
    to_groups: Option<Vec<String>>,
    #[serde(alias = "to_node_label", alias = "to_node_tag")]
    to_node_tag: Option<String>,
    #[serde(
        alias = "to_node_labels",
        alias = "to_node_label_ids",
        alias = "to_node_tags"
    )]
    to_node_tags: Option<Vec<String>>,
    #[serde(alias = "to_rack_id")]
    to_rack: Option<String>,
    #[serde(alias = "to_rack_ids")]
    to_racks: Option<Vec<String>>,
    #[serde(alias = "to_island_id", alias = "to_topology_domain")]
    to_island: Option<String>,
    #[serde(alias = "to_island_ids", alias = "to_topology_domains")]
    to_islands: Option<Vec<String>>,
    #[serde(alias = "to_failure_domain_id")]
    to_failure_domain: Option<String>,
    #[serde(alias = "to_failure_domain_ids")]
    to_failure_domains: Option<Vec<String>>,
    #[serde(alias = "to_gpu", alias = "to_gpu_id", alias = "to_local_gpu")]
    to_gpu: Option<u32>,
    #[serde(
        alias = "to_gpu_ids",
        alias = "to_local_gpus",
        alias = "to_local_gpu_ids"
    )]
    to_gpus: Option<Vec<u32>>,
    rail: Option<u32>,
    rails: Option<Vec<u32>>,
    #[serde(
        alias = "scale",
        alias = "link_bandwidth_scale",
        alias = "fabric_bandwidth_scale"
    )]
    bandwidth_scale: Option<f64>,
    #[serde(alias = "link_latency_scale", alias = "fabric_latency_scale")]
    latency_scale: Option<f64>,
}

#[derive(Deserialize)]
struct ModelSection {
    #[serde(alias = "name", alias = "model_id")]
    id: Option<String>,
    layers: u32,
    hidden_size: u32,
    attention_heads: u32,
    kv_heads: u32,
    vocab_size: u32,
    parameters_gb: f64,
    #[serde(
        alias = "parameter_count_b",
        alias = "params_billion",
        alias = "params_b"
    )]
    parameter_count_billion: Option<f64>,
    dtype: String,
    #[serde(alias = "cache_dtype", alias = "kv_cache_dtype")]
    kv_dtype: Option<String>,
    experts: Option<ExpertSection>,
}

#[derive(Deserialize)]
struct ExpertSection {
    expert_count: u32,
    top_k: u32,
}

#[derive(Deserialize)]
struct RequestSection {
    batch_size: u32,
    prompt_tokens: u32,
    decode_tokens: u32,
    max_sequence_tokens: u32,
    phase: String,
}

#[derive(Clone, Deserialize)]
struct SearchSection {
    tensor_ranks: Vec<u32>,
    pipeline_ranks: Vec<u32>,
    expert_ranks: Vec<u32>,
    data_ranks: Vec<u32>,
}

#[derive(Clone, Deserialize)]
struct PlacementSection {
    ranks: Vec<PlacementRankSection>,
}

#[derive(Clone, Deserialize)]
struct PlacementRankSection {
    #[serde(alias = "rank_id")]
    rank: Option<u32>,
    #[serde(alias = "node_id")]
    node: Option<u32>,
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_gpu_id")]
    local_gpu_id: Option<u32>,
}

#[derive(Deserialize)]
struct ServingSection {
    mode: Option<String>,
    #[serde(alias = "runtime", alias = "backend", alias = "stack")]
    serving_stack: Option<String>,
    #[serde(
        alias = "runtime_features",
        alias = "backend_features",
        alias = "stack_features"
    )]
    serving_runtime_features: Option<Vec<String>>,
    objective: Option<String>,
    #[serde(alias = "slo_penalty_weight", alias = "slo_objective_penalty_weight")]
    slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "ttft_slo_penalty_weight", alias = "ttft_penalty_weight")]
    ttft_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "tpot_slo_penalty_weight", alias = "tpot_penalty_weight")]
    tpot_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "itl_slo_penalty_weight", alias = "itl_penalty_weight")]
    itl_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "e2el_slo_penalty_weight", alias = "e2el_penalty_weight")]
    e2el_slo_miss_penalty_weight: Option<f64>,
    #[serde(
        alias = "deadline_slo_miss_penalty_weight",
        alias = "deadline_penalty_weight"
    )]
    deadline_miss_penalty_weight: Option<f64>,
    #[serde(alias = "route_risk_penalty_weight", alias = "topology_penalty_weight")]
    topology_risk_penalty_weight: Option<f64>,
    #[serde(alias = "max_hbm_pressure", alias = "max_hbm_utilization")]
    max_memory_pressure_fraction: Option<f64>,
    #[serde(
        alias = "max_gpus",
        alias = "max_gpu_footprint",
        alias = "max_serving_gpus"
    )]
    max_unique_gpus: Option<u32>,
    #[serde(
        alias = "min_throughput",
        alias = "min_tokens_per_s",
        alias = "min_output_tokens_per_s"
    )]
    min_throughput_tokens_per_s: Option<f64>,
    #[serde(
        alias = "ttft_ceiling_s",
        alias = "max_ttft_latency_s",
        alias = "max_time_to_first_token_s"
    )]
    max_ttft_s: Option<f64>,
    #[serde(
        alias = "ttft_ceiling_ms",
        alias = "max_ttft_latency_ms",
        alias = "max_time_to_first_token_ms"
    )]
    max_ttft_ms: Option<f64>,
    #[serde(
        alias = "tpot_ceiling_s",
        alias = "max_tpot_latency_s",
        alias = "max_time_per_output_token_s"
    )]
    max_tpot_s: Option<f64>,
    #[serde(
        alias = "tpot_ceiling_ms",
        alias = "max_tpot_latency_ms",
        alias = "max_time_per_output_token_ms"
    )]
    max_tpot_ms: Option<f64>,
    #[serde(alias = "itl_ceiling_s", alias = "max_itl_latency_s")]
    max_itl_s: Option<f64>,
    #[serde(alias = "itl_ceiling_ms", alias = "max_itl_latency_ms")]
    max_itl_ms: Option<f64>,
    #[serde(
        alias = "e2el_ceiling_s",
        alias = "max_e2el_latency_s",
        alias = "max_end_to_end_latency_s"
    )]
    max_e2el_s: Option<f64>,
    #[serde(
        alias = "e2el_ceiling_ms",
        alias = "max_e2el_latency_ms",
        alias = "max_end_to_end_latency_ms"
    )]
    max_e2el_ms: Option<f64>,
    #[serde(
        alias = "min_kv_route_rails",
        alias = "min_kv_transfer_rails",
        alias = "min_kv_transfer_rail_count"
    )]
    min_kv_route_rail_count: Option<u32>,
    #[serde(
        alias = "require_rail_metadata",
        alias = "require_kv_transfer_rail_metadata"
    )]
    require_kv_route_rail_metadata: Option<bool>,
    #[serde(
        alias = "require_kv_gpudirect",
        alias = "require_kv_transfer_gpudirect"
    )]
    require_gpudirect_kv_paths: Option<bool>,
    #[serde(alias = "economics", alias = "cost_model")]
    cost: Option<ServingCostSection>,
    #[serde(
        alias = "require_routable_pool",
        alias = "validate_routable_pool",
        alias = "validate_routable_pools"
    )]
    require_routable_pools: Option<bool>,
    prefill_nodes: Option<Vec<u32>>,
    decode_nodes: Option<Vec<u32>>,
    prefill_groups: Option<Vec<String>>,
    decode_groups: Option<Vec<String>>,
    #[serde(alias = "prefill_gpu_label", alias = "prefill_gpu_tag")]
    prefill_gpu_tag: Option<String>,
    #[serde(alias = "prefill_gpu_labels", alias = "prefill_gpu_tags")]
    prefill_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "decode_gpu_label", alias = "decode_gpu_tag")]
    decode_gpu_tag: Option<String>,
    #[serde(alias = "decode_gpu_labels", alias = "decode_gpu_tags")]
    decode_gpu_tags: Option<Vec<String>>,
    pool_candidates: Option<Vec<ServingPoolCandidateSection>>,
    pool_search: Option<ServingPoolSearchSection>,
    slo_policies: Option<Vec<ServingSloPolicySection>>,
    #[serde(alias = "classes")]
    traffic_classes: Option<Vec<ServingTrafficClassSection>>,
    prefill_search: Option<SearchSection>,
    decode_search: Option<SearchSection>,
    prefill_placement: Option<PlacementSection>,
    decode_placement: Option<PlacementSection>,
    services: Option<ServingServicesSection>,
    prefill_service: Option<ServingServicePhaseSection>,
    decode_service: Option<ServingServicePhaseSection>,
    kv_transfer_service: Option<ServingServicePhaseSection>,
    traffic: Option<ServingTrafficSection>,
}

#[derive(Deserialize)]
struct ServingCostSection {
    #[serde(alias = "gpu_hour_usd", alias = "usd_per_gpu_hour")]
    default_gpu_hour_usd: Option<f64>,
    #[serde(alias = "usd_per_node_hour")]
    node_hour_usd: Option<f64>,
    #[serde(alias = "usd_per_kwh", alias = "energy_usd_per_kwh")]
    kwh_usd: Option<f64>,
    #[serde(alias = "gpu_watts", alias = "watts_per_gpu")]
    default_gpu_watts: Option<f64>,
    #[serde(alias = "watts_per_node")]
    node_watts: Option<f64>,
    gpu_rates: Option<Vec<ServingGpuCostRateSection>>,
}

#[derive(Deserialize)]
struct ServingGpuCostRateSection {
    #[serde(alias = "gpu", alias = "gpu_type", alias = "label")]
    gpu_label: Option<String>,
    #[serde(
        alias = "gpu_hour_usd",
        alias = "usd_per_hour",
        alias = "usd_per_gpu_hour"
    )]
    gpu_hour_usd: Option<f64>,
    #[serde(alias = "gpu_watts")]
    watts: Option<f64>,
}

#[derive(Clone, Deserialize)]
struct ServingServicesSection {
    prefill: Option<ServingServicePhaseSection>,
    decode: Option<ServingServicePhaseSection>,
    kv_transfer: Option<ServingServicePhaseSection>,
}

struct ServingServiceSections {
    services: Option<ServingServicesSection>,
    prefill: Option<ServingServicePhaseSection>,
    decode: Option<ServingServicePhaseSection>,
    kv_transfer: Option<ServingServicePhaseSection>,
}

#[derive(Clone, Deserialize)]
struct ServingServicePhaseSection {
    health: Option<String>,
    enabled: Option<bool>,
    #[serde(alias = "scale", alias = "worker_slots_scale")]
    worker_scale: Option<f64>,
}

#[derive(Deserialize)]
struct ServingSloPolicySection {
    group: String,
    key: String,
    max_ttft_slo_miss_rate: Option<f64>,
    max_tpot_slo_miss_rate: Option<f64>,
    max_itl_slo_miss_rate: Option<f64>,
    max_e2el_slo_miss_rate: Option<f64>,
    max_deadline_miss_rate: Option<f64>,
}

#[derive(Deserialize)]
struct ServingTrafficClassSection {
    name: String,
    group: String,
    key: Option<String>,
    priority: Option<i32>,
    #[serde(
        alias = "priority_override",
        alias = "queue_priority",
        alias = "scheduling_priority"
    )]
    admission_priority: Option<i32>,
    #[serde(alias = "max_active_prefill_tokens")]
    max_prefill_tokens: Option<u64>,
    max_decode_sequences: Option<u32>,
    max_resident_tokens: Option<u64>,
    max_kv_blocks: Option<u64>,
    ttft_slo_ms: Option<f64>,
    tpot_slo_ms: Option<f64>,
    itl_slo_ms: Option<f64>,
    e2el_slo_ms: Option<f64>,
    max_queue_delay_s: Option<f64>,
    max_queue_delay_ms: Option<f64>,
    max_kv_queue_delay_s: Option<f64>,
    max_kv_queue_delay_ms: Option<f64>,
    max_decode_queue_delay_s: Option<f64>,
    max_decode_queue_delay_ms: Option<f64>,
    max_decode_iteration_queue_delay_s: Option<f64>,
    max_decode_iteration_queue_delay_ms: Option<f64>,
    request_timeout_s: Option<f64>,
    request_timeout_ms: Option<f64>,
    #[serde(alias = "slo_penalty_weight", alias = "slo_objective_penalty_weight")]
    slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "ttft_slo_penalty_weight", alias = "ttft_penalty_weight")]
    ttft_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "tpot_slo_penalty_weight", alias = "tpot_penalty_weight")]
    tpot_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "itl_slo_penalty_weight", alias = "itl_penalty_weight")]
    itl_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "e2el_slo_penalty_weight", alias = "e2el_penalty_weight")]
    e2el_slo_miss_penalty_weight: Option<f64>,
    #[serde(
        alias = "deadline_slo_miss_penalty_weight",
        alias = "deadline_penalty_weight"
    )]
    deadline_miss_penalty_weight: Option<f64>,
    max_ttft_slo_miss_rate: Option<f64>,
    max_tpot_slo_miss_rate: Option<f64>,
    max_itl_slo_miss_rate: Option<f64>,
    max_e2el_slo_miss_rate: Option<f64>,
    max_deadline_miss_rate: Option<f64>,
}

#[derive(Deserialize)]
struct ServingPoolCandidateSection {
    label: Option<String>,
    prefill_nodes: Option<Vec<u32>>,
    decode_nodes: Option<Vec<u32>>,
    prefill_groups: Option<Vec<String>>,
    decode_groups: Option<Vec<String>>,
    #[serde(alias = "prefill_gpu_label", alias = "prefill_gpu_tag")]
    prefill_gpu_tag: Option<String>,
    #[serde(alias = "prefill_gpu_labels", alias = "prefill_gpu_tags")]
    prefill_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "decode_gpu_label", alias = "decode_gpu_tag")]
    decode_gpu_tag: Option<String>,
    #[serde(alias = "decode_gpu_labels", alias = "decode_gpu_tags")]
    decode_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_node_label", alias = "prefill_node_tag")]
    prefill_node_tag: Option<String>,
    #[serde(alias = "prefill_node_labels", alias = "prefill_node_tags")]
    prefill_node_tags: Option<Vec<String>>,
    #[serde(alias = "decode_node_label", alias = "decode_node_tag")]
    decode_node_tag: Option<String>,
    #[serde(alias = "decode_node_labels", alias = "decode_node_tags")]
    decode_node_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_rack_id")]
    prefill_rack: Option<String>,
    #[serde(alias = "prefill_rack_ids")]
    prefill_racks: Option<Vec<String>>,
    #[serde(alias = "decode_rack_id")]
    decode_rack: Option<String>,
    #[serde(alias = "decode_rack_ids")]
    decode_racks: Option<Vec<String>>,
    #[serde(alias = "prefill_island_id")]
    prefill_island: Option<String>,
    #[serde(alias = "prefill_island_ids")]
    prefill_islands: Option<Vec<String>>,
    #[serde(alias = "decode_island_id")]
    decode_island: Option<String>,
    #[serde(alias = "decode_island_ids")]
    decode_islands: Option<Vec<String>>,
    #[serde(alias = "prefill_failure_domain_id")]
    prefill_failure_domain: Option<String>,
    #[serde(alias = "prefill_failure_domain_ids")]
    prefill_failure_domains: Option<Vec<String>>,
    #[serde(alias = "decode_failure_domain_id")]
    decode_failure_domain: Option<String>,
    #[serde(alias = "decode_failure_domain_ids")]
    decode_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_node_label",
        alias = "exclude_prefill_node_label",
        alias = "exclude_prefill_node_tag"
    )]
    prefill_exclude_node_tag: Option<String>,
    #[serde(
        alias = "prefill_exclude_node_labels",
        alias = "exclude_prefill_node_labels",
        alias = "exclude_prefill_node_tags"
    )]
    prefill_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_node_label",
        alias = "exclude_decode_node_label",
        alias = "exclude_decode_node_tag"
    )]
    decode_exclude_node_tag: Option<String>,
    #[serde(
        alias = "decode_exclude_node_labels",
        alias = "exclude_decode_node_labels",
        alias = "exclude_decode_node_tags"
    )]
    decode_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_rack_id",
        alias = "exclude_prefill_rack",
        alias = "exclude_prefill_rack_id"
    )]
    prefill_exclude_rack: Option<String>,
    #[serde(
        alias = "prefill_exclude_rack_ids",
        alias = "exclude_prefill_racks",
        alias = "exclude_prefill_rack_ids"
    )]
    prefill_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_rack_id",
        alias = "exclude_decode_rack",
        alias = "exclude_decode_rack_id"
    )]
    decode_exclude_rack: Option<String>,
    #[serde(
        alias = "decode_exclude_rack_ids",
        alias = "exclude_decode_racks",
        alias = "exclude_decode_rack_ids"
    )]
    decode_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_island_id",
        alias = "exclude_prefill_island",
        alias = "exclude_prefill_island_id"
    )]
    prefill_exclude_island: Option<String>,
    #[serde(
        alias = "prefill_exclude_island_ids",
        alias = "exclude_prefill_islands",
        alias = "exclude_prefill_island_ids"
    )]
    prefill_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_island_id",
        alias = "exclude_decode_island",
        alias = "exclude_decode_island_id"
    )]
    decode_exclude_island: Option<String>,
    #[serde(
        alias = "decode_exclude_island_ids",
        alias = "exclude_decode_islands",
        alias = "exclude_decode_island_ids"
    )]
    decode_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_failure_domain_id",
        alias = "exclude_prefill_failure_domain",
        alias = "exclude_prefill_failure_domain_id"
    )]
    prefill_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "prefill_exclude_failure_domain_ids",
        alias = "exclude_prefill_failure_domains",
        alias = "exclude_prefill_failure_domain_ids"
    )]
    prefill_exclude_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_failure_domain_id",
        alias = "exclude_decode_failure_domain",
        alias = "exclude_decode_failure_domain_id"
    )]
    decode_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "decode_exclude_failure_domain_ids",
        alias = "exclude_decode_failure_domains",
        alias = "exclude_decode_failure_domain_ids"
    )]
    decode_exclude_failure_domains: Option<Vec<String>>,
    #[serde(alias = "min_prefill_rack_count")]
    min_prefill_racks: Option<u32>,
    #[serde(alias = "min_decode_rack_count")]
    min_decode_racks: Option<u32>,
    #[serde(alias = "min_prefill_island_count")]
    min_prefill_islands: Option<u32>,
    #[serde(alias = "min_decode_island_count")]
    min_decode_islands: Option<u32>,
    #[serde(alias = "min_prefill_failure_domain_count")]
    min_prefill_failure_domains: Option<u32>,
    #[serde(alias = "min_decode_failure_domain_count")]
    min_decode_failure_domains: Option<u32>,
}

#[derive(Deserialize)]
struct ServingPoolSearchSection {
    prefill_groups: Vec<String>,
    decode_groups: Vec<String>,
    prefill_node_counts: Option<Vec<u32>>,
    decode_node_counts: Option<Vec<u32>>,
    #[serde(alias = "prefill_gpu_label", alias = "prefill_gpu_tag")]
    prefill_gpu_tag: Option<String>,
    #[serde(alias = "prefill_gpu_labels", alias = "prefill_gpu_tags")]
    prefill_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "decode_gpu_label", alias = "decode_gpu_tag")]
    decode_gpu_tag: Option<String>,
    #[serde(alias = "decode_gpu_labels", alias = "decode_gpu_tags")]
    decode_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_node_label", alias = "prefill_node_tag")]
    prefill_node_tag: Option<String>,
    #[serde(alias = "prefill_node_labels", alias = "prefill_node_tags")]
    prefill_node_tags: Option<Vec<String>>,
    #[serde(alias = "decode_node_label", alias = "decode_node_tag")]
    decode_node_tag: Option<String>,
    #[serde(alias = "decode_node_labels", alias = "decode_node_tags")]
    decode_node_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_rack_id")]
    prefill_rack: Option<String>,
    #[serde(alias = "prefill_rack_ids")]
    prefill_racks: Option<Vec<String>>,
    #[serde(alias = "decode_rack_id")]
    decode_rack: Option<String>,
    #[serde(alias = "decode_rack_ids")]
    decode_racks: Option<Vec<String>>,
    #[serde(alias = "prefill_island_id")]
    prefill_island: Option<String>,
    #[serde(alias = "prefill_island_ids")]
    prefill_islands: Option<Vec<String>>,
    #[serde(alias = "decode_island_id")]
    decode_island: Option<String>,
    #[serde(alias = "decode_island_ids")]
    decode_islands: Option<Vec<String>>,
    #[serde(alias = "prefill_failure_domain_id")]
    prefill_failure_domain: Option<String>,
    #[serde(alias = "prefill_failure_domain_ids")]
    prefill_failure_domains: Option<Vec<String>>,
    #[serde(alias = "decode_failure_domain_id")]
    decode_failure_domain: Option<String>,
    #[serde(alias = "decode_failure_domain_ids")]
    decode_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_node_label",
        alias = "exclude_prefill_node_label",
        alias = "exclude_prefill_node_tag"
    )]
    prefill_exclude_node_tag: Option<String>,
    #[serde(
        alias = "prefill_exclude_node_labels",
        alias = "exclude_prefill_node_labels",
        alias = "exclude_prefill_node_tags"
    )]
    prefill_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_node_label",
        alias = "exclude_decode_node_label",
        alias = "exclude_decode_node_tag"
    )]
    decode_exclude_node_tag: Option<String>,
    #[serde(
        alias = "decode_exclude_node_labels",
        alias = "exclude_decode_node_labels",
        alias = "exclude_decode_node_tags"
    )]
    decode_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_rack_id",
        alias = "exclude_prefill_rack",
        alias = "exclude_prefill_rack_id"
    )]
    prefill_exclude_rack: Option<String>,
    #[serde(
        alias = "prefill_exclude_rack_ids",
        alias = "exclude_prefill_racks",
        alias = "exclude_prefill_rack_ids"
    )]
    prefill_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_rack_id",
        alias = "exclude_decode_rack",
        alias = "exclude_decode_rack_id"
    )]
    decode_exclude_rack: Option<String>,
    #[serde(
        alias = "decode_exclude_rack_ids",
        alias = "exclude_decode_racks",
        alias = "exclude_decode_rack_ids"
    )]
    decode_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_island_id",
        alias = "exclude_prefill_island",
        alias = "exclude_prefill_island_id"
    )]
    prefill_exclude_island: Option<String>,
    #[serde(
        alias = "prefill_exclude_island_ids",
        alias = "exclude_prefill_islands",
        alias = "exclude_prefill_island_ids"
    )]
    prefill_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_island_id",
        alias = "exclude_decode_island",
        alias = "exclude_decode_island_id"
    )]
    decode_exclude_island: Option<String>,
    #[serde(
        alias = "decode_exclude_island_ids",
        alias = "exclude_decode_islands",
        alias = "exclude_decode_island_ids"
    )]
    decode_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_failure_domain_id",
        alias = "exclude_prefill_failure_domain",
        alias = "exclude_prefill_failure_domain_id"
    )]
    prefill_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "prefill_exclude_failure_domain_ids",
        alias = "exclude_prefill_failure_domains",
        alias = "exclude_prefill_failure_domain_ids"
    )]
    prefill_exclude_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_failure_domain_id",
        alias = "exclude_decode_failure_domain",
        alias = "exclude_decode_failure_domain_id"
    )]
    decode_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "decode_exclude_failure_domain_ids",
        alias = "exclude_decode_failure_domains",
        alias = "exclude_decode_failure_domain_ids"
    )]
    decode_exclude_failure_domains: Option<Vec<String>>,
    #[serde(alias = "min_prefill_rack_count")]
    min_prefill_racks: Option<u32>,
    #[serde(alias = "min_decode_rack_count")]
    min_decode_racks: Option<u32>,
    #[serde(alias = "min_prefill_island_count")]
    min_prefill_islands: Option<u32>,
    #[serde(alias = "min_decode_island_count")]
    min_decode_islands: Option<u32>,
    #[serde(alias = "min_prefill_failure_domain_count")]
    min_prefill_failure_domains: Option<u32>,
    #[serde(alias = "min_decode_failure_domain_count")]
    min_decode_failure_domains: Option<u32>,
    allow_overlap: Option<bool>,
    max_candidates: Option<usize>,
}

#[derive(Deserialize)]
struct ServingTrafficSection {
    request_count: Option<u32>,
    arrival: Option<String>,
    arrival_gap_s: Option<f64>,
    arrival_gap_ms: Option<f64>,
    arrival_rate_per_s: Option<f64>,
    arrival_seed: Option<u64>,
    burst_size: Option<u32>,
    burst_interval_s: Option<f64>,
    burst_interval_ms: Option<f64>,
    #[serde(alias = "burst_gap_s", alias = "intra_burst_gap_s")]
    burst_arrival_gap_s: Option<f64>,
    #[serde(alias = "burst_gap_ms", alias = "intra_burst_gap_ms")]
    burst_arrival_gap_ms: Option<f64>,
    #[serde(alias = "min_arrival_rate_per_s", alias = "diurnal_trough_rate_per_s")]
    diurnal_min_rate_per_s: Option<f64>,
    #[serde(alias = "max_arrival_rate_per_s", alias = "diurnal_peak_rate_per_s")]
    diurnal_max_rate_per_s: Option<f64>,
    diurnal_period_s: Option<f64>,
    diurnal_period_ms: Option<f64>,
    diurnal_phase_s: Option<f64>,
    diurnal_phase_ms: Option<f64>,
    #[serde(alias = "selfsimilar_rate_per_s")]
    self_similar_rate_per_s: Option<f64>,
    #[serde(
        alias = "self_similar_shape",
        alias = "self_similar_alpha",
        alias = "self_similar_pareto_alpha",
        alias = "pareto_shape",
        alias = "pareto_alpha"
    )]
    self_similar_pareto_shape: Option<f64>,
    self_similar_max_gap_s: Option<f64>,
    self_similar_max_gap_ms: Option<f64>,
    routing_policy: Option<String>,
    prefill_batching: Option<String>,
    max_prefill_batch_tokens: Option<u64>,
    #[serde(alias = "prefill_chunk_tokens", alias = "chunk_prefill_tokens")]
    max_prefill_chunk_tokens: Option<u32>,
    decode_batching: Option<String>,
    #[serde(alias = "decode_admission_policy", alias = "capacity_policy")]
    decode_capacity_policy: Option<String>,
    #[serde(
        alias = "backpressure_penalty_weight",
        alias = "queue_backpressure_penalty_weight"
    )]
    service_backpressure_penalty_weight: Option<f64>,
    #[serde(alias = "max_active_prefill_tokens")]
    max_prefill_tokens: Option<u64>,
    #[serde(alias = "max_active_prefill_tokens_per_node")]
    max_prefill_tokens_per_node: Option<u64>,
    #[serde(alias = "max_active_prefill_tokens_per_gpu")]
    max_prefill_tokens_per_gpu: Option<u64>,
    #[serde(alias = "prefill_worker_slots_per_gpu", alias = "prefill_worker_slots")]
    max_prefill_worker_slots_per_gpu: Option<u32>,
    max_decode_batch_tokens: Option<u32>,
    max_decode_sequences: Option<u32>,
    max_resident_tokens: Option<u64>,
    max_decode_sequences_per_node: Option<u32>,
    max_resident_tokens_per_node: Option<u64>,
    max_decode_sequences_per_gpu: Option<u32>,
    #[serde(alias = "decode_worker_slots_per_gpu", alias = "decode_worker_slots")]
    max_decode_worker_slots_per_gpu: Option<u32>,
    max_resident_tokens_per_gpu: Option<u64>,
    #[serde(
        alias = "kv_transfer_worker_slots_per_gpu",
        alias = "kv_transfer_worker_slots"
    )]
    max_kv_transfer_worker_slots_per_gpu: Option<u32>,
    kv_block_tokens: Option<u32>,
    max_kv_blocks: Option<u64>,
    max_kv_blocks_per_node: Option<u64>,
    max_kv_blocks_per_gpu: Option<u64>,
    ttft_slo_ms: Option<f64>,
    tpot_slo_ms: Option<f64>,
    itl_slo_ms: Option<f64>,
    e2el_slo_ms: Option<f64>,
    max_ttft_slo_miss_rate: Option<f64>,
    max_tpot_slo_miss_rate: Option<f64>,
    max_itl_slo_miss_rate: Option<f64>,
    max_e2el_slo_miss_rate: Option<f64>,
    max_deadline_miss_rate: Option<f64>,
    #[serde(alias = "prefix_cache_hit_ratio")]
    prefix_cache_hit_rate: Option<f64>,
    shape_seed: Option<u64>,
    batch_size_distribution: Option<ServingValueDistributionSection>,
    prompt_tokens_distribution: Option<ServingValueDistributionSection>,
    decode_tokens_distribution: Option<ServingValueDistributionSection>,
    shape_profiles: Option<Vec<ServingShapeProfileSection>>,
    batch_sizes: Option<Vec<u32>>,
    prompt_tokens: Option<Vec<u32>>,
    decode_tokens: Option<Vec<u32>>,
    requests: Option<Vec<ServingTraceRequestSection>>,
    #[serde(alias = "trace_path", alias = "trace_file")]
    trace_csv: Option<String>,
    #[serde(alias = "trace_json", alias = "trace_jsonl_path")]
    trace_jsonl: Option<String>,
    trace_start_s: Option<f64>,
    trace_start_ms: Option<f64>,
    trace_end_s: Option<f64>,
    trace_end_ms: Option<f64>,
    trace_time_scale: Option<f64>,
    trace_arrival_offset_s: Option<f64>,
    trace_arrival_offset_ms: Option<f64>,
    #[serde(
        alias = "trace_repeat",
        alias = "trace_repeats",
        alias = "trace_replay_count"
    )]
    trace_repeat_count: Option<u32>,
    #[serde(alias = "trace_replay_interval_s")]
    trace_repeat_interval_s: Option<f64>,
    #[serde(alias = "trace_replay_interval_ms")]
    trace_repeat_interval_ms: Option<f64>,
    #[serde(alias = "metric_start_s")]
    measurement_start_s: Option<f64>,
    #[serde(alias = "metric_start_ms")]
    measurement_start_ms: Option<f64>,
    #[serde(alias = "metric_end_s")]
    measurement_end_s: Option<f64>,
    #[serde(alias = "metric_end_ms")]
    measurement_end_ms: Option<f64>,
    #[serde(alias = "metric_warmup_s", alias = "warmup_s")]
    measurement_warmup_s: Option<f64>,
    #[serde(alias = "metric_warmup_ms", alias = "warmup_ms")]
    measurement_warmup_ms: Option<f64>,
    #[serde(alias = "metric_cooldown_s", alias = "cooldown_s")]
    measurement_cooldown_s: Option<f64>,
    #[serde(alias = "metric_cooldown_ms", alias = "cooldown_ms")]
    measurement_cooldown_ms: Option<f64>,
    #[serde(
        alias = "auto_steady_state",
        alias = "steady_state",
        alias = "measurement_auto_steady_state"
    )]
    measurement_steady_state: Option<bool>,
    measurement_steady_state_min_requests: Option<u32>,
    measurement_steady_state_max_cv: Option<f64>,
    max_queue_delay_s: Option<f64>,
    max_queue_delay_ms: Option<f64>,
    max_kv_queue_delay_s: Option<f64>,
    max_kv_queue_delay_ms: Option<f64>,
    max_decode_queue_delay_s: Option<f64>,
    max_decode_queue_delay_ms: Option<f64>,
    max_decode_iteration_queue_delay_s: Option<f64>,
    max_decode_iteration_queue_delay_ms: Option<f64>,
    request_timeout_s: Option<f64>,
    request_timeout_ms: Option<f64>,
    services: Option<ServingServicesSection>,
    prefill_service: Option<ServingServicePhaseSection>,
    decode_service: Option<ServingServicePhaseSection>,
    kv_transfer_service: Option<ServingServicePhaseSection>,
}

#[derive(Deserialize)]
struct ServingValueDistributionSection {
    kind: String,
    min: Option<u32>,
    max: Option<u32>,
    median: Option<f64>,
    sigma: Option<f64>,
    values: Option<Vec<u32>>,
    weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct ServingShapeProfileSection {
    #[serde(alias = "label")]
    name: Option<String>,
    weight: Option<f64>,
    #[serde(alias = "tenant_id")]
    tenant: Option<String>,
    #[serde(alias = "model")]
    model_id: Option<String>,
    #[serde(alias = "prefix_cache_key")]
    cache_key: Option<String>,
    priority: Option<i32>,
    ttft_slo_s: Option<f64>,
    ttft_slo_ms: Option<f64>,
    tpot_slo_s: Option<f64>,
    tpot_slo_ms: Option<f64>,
    itl_slo_s: Option<f64>,
    itl_slo_ms: Option<f64>,
    e2el_slo_s: Option<f64>,
    e2el_slo_ms: Option<f64>,
    request_timeout_s: Option<f64>,
    request_timeout_ms: Option<f64>,
    batch_size: u32,
    prompt_tokens: u32,
    decode_tokens: u32,
    max_sequence_tokens: Option<u32>,
    prefix_cache_hit_tokens: Option<u32>,
    #[serde(alias = "prefix_cache_hit_ratio")]
    prefix_cache_hit_rate: Option<f64>,
    deadline_after_s: Option<f64>,
    deadline_after_ms: Option<f64>,
    #[serde(alias = "cancellation_after_s")]
    cancel_after_s: Option<f64>,
    #[serde(alias = "cancellation_after_ms")]
    cancel_after_ms: Option<f64>,
}

#[derive(Deserialize)]
struct ServingTraceRequestSection {
    #[serde(alias = "id")]
    request_id: Option<String>,
    #[serde(alias = "tenant_id")]
    tenant: Option<String>,
    #[serde(alias = "model")]
    model_id: Option<String>,
    #[serde(alias = "prefix_cache_key")]
    cache_key: Option<String>,
    #[serde(alias = "arrival", alias = "arrival_time_s")]
    arrival_s: Option<f64>,
    #[serde(alias = "arrival_time_ms", alias = "timestamp_ms")]
    arrival_ms: Option<f64>,
    priority: Option<i32>,
    ttft_slo_s: Option<f64>,
    ttft_slo_ms: Option<f64>,
    tpot_slo_s: Option<f64>,
    tpot_slo_ms: Option<f64>,
    itl_slo_s: Option<f64>,
    itl_slo_ms: Option<f64>,
    e2el_slo_s: Option<f64>,
    e2el_slo_ms: Option<f64>,
    deadline_s: Option<f64>,
    deadline_ms: Option<f64>,
    deadline_after_s: Option<f64>,
    deadline_after_ms: Option<f64>,
    cancellation_s: Option<f64>,
    cancellation_ms: Option<f64>,
    cancel_after_s: Option<f64>,
    cancel_after_ms: Option<f64>,
    #[serde(alias = "batch")]
    batch_size: u32,
    #[serde(alias = "input_tokens", alias = "prefill_tokens")]
    prompt_tokens: u32,
    #[serde(alias = "output_tokens", alias = "max_new_tokens")]
    decode_tokens: u32,
    #[serde(alias = "sequence_tokens")]
    max_sequence_tokens: Option<u32>,
    prefix_cache_hit_tokens: Option<u32>,
    #[serde(alias = "prefix_cache_hit_ratio")]
    prefix_cache_hit_rate: Option<f64>,
}

#[derive(Deserialize)]
struct CalibrationSection {
    compute_efficiency: Option<f64>,
    prefill_compute_scale: Option<f64>,
    decode_compute_scale: Option<f64>,
    decode_memory_bandwidth_scale: Option<f64>,
    collective_latency_scale: Option<f64>,
    collective_bandwidth_scale: Option<f64>,
    kv_transfer_scale: Option<f64>,
    scheduler_overhead_us: Option<f64>,
    serving_memory_temporary_fraction: Option<f64>,
    serving_memory_activation_communication_fraction: Option<f64>,
    serving_memory_weight_communication_fraction: Option<f64>,
    serving_memory_runtime_reserve_fraction: Option<f64>,
    serving_memory_fragmentation_fraction: Option<f64>,
    serving_pipeline_depth: Option<u32>,
    request_arrival_gap_s: Option<f64>,
    allow_compute_comm_overlap: Option<bool>,
}

#[derive(Deserialize)]
struct CalibrationPolicySection {
    valid_shape: Option<String>,
    invalid_shape: Option<String>,
    coverage: Option<String>,
    fit_confidence: Option<String>,
    fit_extrapolation: Option<String>,
    fit_partially_bounded: Option<String>,
    fit_unbounded: Option<String>,
    fit_sample_count: Option<String>,
    fit_validation_sample_count: Option<String>,
    fit_source: Option<String>,
    fit_uncertainty: Option<String>,
    profile_source: Option<String>,
    profile_date: Option<String>,
    #[serde(alias = "profile_runtime_provenance", alias = "profile_stack")]
    profile_runtime: Option<String>,
    min_coverage_score: Option<f64>,
    min_fit_confidence_score: Option<f64>,
    min_fit_confidence_level: Option<f64>,
    min_fit_sample_count: Option<u32>,
    min_fit_validation_sample_count: Option<u32>,
    max_fit_relative_uncertainty_pct: Option<f64>,
    max_fit_absolute_uncertainty_ms: Option<f64>,
    max_fit_absolute_uncertainty_s: Option<f64>,
    min_serving_phase_coverage_fraction: Option<f64>,
    uncertainty_ranking_weight: Option<f64>,
    require_phase_coverage: Option<bool>,
}

#[derive(Deserialize)]
struct ApproximationPolicySection {
    #[serde(alias = "profile")]
    preset: Option<String>,
    #[serde(alias = "default", alias = "mode")]
    default_action: Option<String>,
    reject_categories: Option<Vec<String>>,
    reject_codes: Option<Vec<String>>,
    warn_categories: Option<Vec<String>>,
    warn_codes: Option<Vec<String>>,
    #[serde(
        default,
        alias = "metric_gate",
        alias = "precision_gates",
        alias = "precision_gate",
        alias = "metric_precision_gates",
        alias = "metric_precision_gate"
    )]
    metric_gates: Vec<ApproximationMetricGateSection>,
}

#[derive(Deserialize)]
struct ApproximationMetricGateSection {
    metric: Option<String>,
    metrics: Option<Vec<String>>,
    #[serde(alias = "objective")]
    objective: Option<String>,
    #[serde(alias = "objectives")]
    objectives: Option<Vec<String>>,
    reject_categories: Option<Vec<String>>,
    reject_codes: Option<Vec<String>>,
    warn_categories: Option<Vec<String>>,
    warn_codes: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct CalibrationProfileFile {
    schema_version: Option<u32>,
    profile: Option<CalibrationProfileMetadataSection>,
    calibration: Option<CalibrationSection>,
    valid_shape: Option<CalibrationShapeRangeSection>,
    invalid_shapes: Option<Vec<CalibrationInvalidShapeRangeSection>>,
    fits: Option<Vec<CalibrationFittedModelSection>>,
    benchmarks: Option<Vec<CalibrationBenchmarkPointSection>>,
}

#[derive(Deserialize)]
struct CalibrationProfileMetadataSection {
    name: Option<String>,
    hardware: Option<String>,
    fabric: Option<String>,
    model: Option<String>,
    dtype: Option<String>,
    serving_stack: Option<String>,
    #[serde(
        alias = "runtime_features",
        alias = "backend_features",
        alias = "stack_features"
    )]
    serving_runtime_features: Option<Vec<String>>,
    #[serde(alias = "serving_stack_version", alias = "runtime_version")]
    backend_version: Option<String>,
    #[serde(alias = "gpu_driver_version")]
    driver_version: Option<String>,
    #[serde(alias = "cuda")]
    cuda_version: Option<String>,
    #[serde(alias = "rocm")]
    rocm_version: Option<String>,
    #[serde(alias = "nccl")]
    nccl_version: Option<String>,
    #[serde(alias = "rccl")]
    rccl_version: Option<String>,
    #[serde(alias = "ucx")]
    ucx_version: Option<String>,
    #[serde(default)]
    kernel_settings: Vec<String>,
    environment_hash: Option<String>,
    source: Option<String>,
    date: Option<String>,
    notes: Option<String>,
}

#[derive(Deserialize)]
struct CalibrationShapeRangeSection {
    min_batch_size: Option<u32>,
    max_batch_size: Option<u32>,
    min_prompt_tokens: Option<u32>,
    max_prompt_tokens: Option<u32>,
    min_decode_tokens: Option<u32>,
    max_decode_tokens: Option<u32>,
    min_sequence_tokens: Option<u32>,
    max_sequence_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct CalibrationInvalidShapeRangeSection {
    name: Option<String>,
    reason: Option<String>,
    min_batch_size: Option<u32>,
    max_batch_size: Option<u32>,
    min_prompt_tokens: Option<u32>,
    max_prompt_tokens: Option<u32>,
    min_decode_tokens: Option<u32>,
    max_decode_tokens: Option<u32>,
    min_sequence_tokens: Option<u32>,
    max_sequence_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct CalibrationFittedModelSection {
    name: Option<String>,
    target: Option<String>,
    phase: Option<String>,
    kind: Option<String>,
    model: Option<String>,
    unit: Option<String>,
    intercept: Option<f64>,
    #[serde(default, alias = "feature_names")]
    features: Vec<String>,
    #[serde(default)]
    coefficients: Vec<f64>,
    #[serde(default)]
    feature_ranges: Vec<CalibrationFitFeatureRangeSection>,
    r_squared: Option<f64>,
    adjusted_r_squared: Option<f64>,
    rmse: Option<f64>,
    rmse_pct: Option<f64>,
    mean_abs_pct_error: Option<f64>,
    max_abs_pct_error: Option<f64>,
    #[serde(alias = "holdout_rmse")]
    validation_rmse: Option<f64>,
    #[serde(alias = "holdout_rmse_pct")]
    validation_rmse_pct: Option<f64>,
    #[serde(alias = "holdout_mean_abs_pct_error")]
    validation_mean_abs_pct_error: Option<f64>,
    #[serde(alias = "holdout_max_abs_pct_error")]
    validation_max_abs_pct_error: Option<f64>,
    #[serde(alias = "ci", alias = "ci95", alias = "confidence_interval_value")]
    confidence_interval: Option<f64>,
    #[serde(
        alias = "ci_pct",
        alias = "ci95_pct",
        alias = "confidence_interval_percent"
    )]
    confidence_interval_pct: Option<f64>,
    #[serde(
        alias = "ci_level",
        alias = "confidence",
        alias = "coverage_probability"
    )]
    confidence_level: Option<f64>,
    sample_count: Option<u32>,
    validation_sample_count: Option<u32>,
    source: Option<String>,
    notes: Option<String>,
}

#[derive(Deserialize)]
struct CalibrationFitFeatureRangeSection {
    feature: Option<String>,
    min: Option<f64>,
    max: Option<f64>,
}

#[derive(Deserialize)]
struct CalibrationBenchmarkPointSection {
    name: Option<String>,
    kind: Option<String>,
    phase: Option<String>,
    hardware: Option<String>,
    fabric: Option<String>,
    model: Option<String>,
    dtype: Option<String>,
    batch_size: Option<u32>,
    prompt_tokens: Option<u32>,
    decode_tokens: Option<u32>,
    sequence_tokens: Option<u32>,
    tensor_ranks: Option<u32>,
    pipeline_ranks: Option<u32>,
    expert_ranks: Option<u32>,
    data_ranks: Option<u32>,
    measured_ms: Option<f64>,
    predicted_ms: Option<f64>,
    throughput_tokens_per_s: Option<f64>,
    command: Option<String>,
    source: Option<String>,
    notes: Option<String>,
}

pub fn load_cluster(path: impl AsRef<Path>) -> Result<Cluster, ConfigError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path).map_err(|err| {
        ConfigError::new(format!(
            "failed to read cluster config {}: {err}",
            path.display()
        ))
    })?;
    parse_cluster(&contents)
}

pub fn load_workload(path: impl AsRef<Path>) -> Result<WorkloadConfig, ConfigError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path).map_err(|err| {
        ConfigError::new(format!(
            "failed to read workload config {}: {err}",
            path.display()
        ))
    })?;
    parse_workload_with_base_dir(&contents, path.parent())
}

pub fn load_run_config(path: impl AsRef<Path>) -> Result<RunConfig, ConfigError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path).map_err(|err| {
        ConfigError::new(format!(
            "failed to read run config {}: {err}",
            path.display()
        ))
    })?;
    parse_run_config_with_base_dir(&contents, path.parent())
}

pub fn parse_run_config(contents: &str) -> Result<RunConfig, ConfigError> {
    parse_run_config_with_base_dir(contents, None)
}

pub fn parse_cluster(contents: &str) -> Result<Cluster, ConfigError> {
    let file: ClusterFile = toml::from_str(contents)
        .map_err(|err| ConfigError::new(format!("invalid cluster TOML: {err}")))?;
    validate_schema_version("cluster", file.schema_version)?;
    let preset = normalize(&file.cluster.preset);

    if preset == "custom" {
        return parse_custom_cluster(file);
    }

    if preset != "h100_sxm" && preset != "h100sxm" {
        return Err(ConfigError::new(format!(
            "unsupported cluster preset '{}'; supported presets: h100_sxm, custom",
            file.cluster.preset
        )));
    }

    let node_count = file
        .cluster
        .node_count
        .ok_or_else(|| ConfigError::new("cluster.node_count is required for h100_sxm"))?;
    if node_count == 0 {
        return Err(ConfigError::new("cluster.node_count must be nonzero"));
    }

    let interconnect = interconnect_profile(
        file.interconnect
            .as_ref()
            .ok_or_else(|| ConfigError::new("[interconnect] is required for h100_sxm"))?,
    )?;
    let mut cluster = Cluster::h100_sxm_nodes(node_count, interconnect.clone());

    if let Some(oversubscription) = file
        .interconnect
        .as_ref()
        .and_then(|interconnect| interconnect.oversubscription)
    {
        if oversubscription < 1.0 {
            return Err(ConfigError::new(
                "interconnect.oversubscription must be greater than or equal to 1.0",
            ));
        }
        if let InterNodeTopology::FatTree {
            oversubscription: existing,
            ..
        } = &mut cluster.inter_node_topology
        {
            *existing = oversubscription;
        }
    }

    if let Some(nics) = file.nics {
        let profile = nics_profile("nics", &nics, interconnect)?;
        for node in cluster.nodes.values_mut() {
            validate_network_profile("nics", &node.gpus, &node.disabled_gpus, &profile)?;
            node.network = profile.clone();
        }
    }

    Ok(cluster)
}

pub fn parse_workload(contents: &str) -> Result<WorkloadConfig, ConfigError> {
    parse_workload_with_base_dir(contents, None)
}

pub fn refresh_workload_calibration_reports(workload: &mut WorkloadConfig) {
    workload.calibration_coverage = calibration_coverage_report(
        workload.calibration_profile.as_ref(),
        &workload.request,
        workload.serving.as_ref(),
    );
    workload.calibration_warnings = calibration_applicability_warnings(
        workload.calibration_profile.as_ref(),
        &workload.request,
        workload.serving.as_ref(),
    );
    workload.calibration_invalid_shape_warnings = calibration_invalid_shape_warnings(
        workload.calibration_profile.as_ref(),
        &workload.request,
        workload.serving.as_ref(),
    );
    workload.calibration_gate_violations = calibration_gate_violations(
        &workload.calibration_policy,
        workload.calibration_profile.as_ref(),
        workload.calibration_coverage.as_ref(),
        &workload.calibration_warnings,
        &workload.calibration_invalid_shape_warnings,
    );
}

fn parse_run_config_with_base_dir(
    contents: &str,
    base_dir: Option<&Path>,
) -> Result<RunConfig, ConfigError> {
    let file: RunFile = toml::from_str(contents)
        .map_err(|err| ConfigError::new(format!("invalid run TOML: {err}")))?;
    validate_schema_version("run", file.schema_version)?;
    let run = file.run;
    let cluster_path = file
        .cluster
        .or_else(|| run.as_ref().and_then(|run| run.cluster.clone()))
        .map(|path| resolve_config_path(&path, base_dir));
    let workload_path = file
        .workload
        .or_else(|| run.as_ref().and_then(|run| run.workload.clone()))
        .map(|path| resolve_config_path(&path, base_dir));
    let output = parse_run_output(file.output, base_dir)?;
    let search_budget = parse_run_search_budget(file.search)?;
    let scenarios = parse_run_scenarios(file.scenarios, base_dir)?;

    Ok(RunConfig {
        cluster_path,
        workload_path,
        output,
        search_budget,
        scenarios,
    })
}

fn parse_run_output(
    section: Option<RunOutputSection>,
    base_dir: Option<&Path>,
) -> Result<RunOutputConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunOutputConfig::default());
    };
    if let Some(top_k) = section.top_k
        && top_k == 0
    {
        return Err(ConfigError::new("output.top_k must be greater than zero"));
    }
    if let Some(occupancy_buckets) = section.occupancy_buckets
        && occupancy_buckets == 0
    {
        return Err(ConfigError::new(
            "output.occupancy_buckets must be greater than zero",
        ));
    }
    let output_dir =
        parse_run_output_path("output.output_dir", section.output_dir.as_deref(), base_dir)?;
    let request_metrics_csv_path = parse_run_output_path(
        "output.request_metrics_csv",
        section.request_metrics_csv_path.as_deref(),
        base_dir,
    )?;
    let request_lifecycle_events_csv_path = parse_run_output_path(
        "output.request_lifecycle_events_csv",
        section.request_lifecycle_events_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_metrics_csv_path = parse_run_output_path(
        "output.serving_metrics_csv",
        section.serving_metrics_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_metric_breakdowns_csv_path = parse_run_output_path(
        "output.serving_metric_breakdowns_csv",
        section.serving_metric_breakdowns_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_services_csv_path = parse_run_output_path(
        "output.serving_services_csv",
        section.serving_services_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_utilization_csv_path = parse_run_output_path(
        "output.serving_utilization_csv",
        section.serving_utilization_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_memory_pressure_csv_path = parse_run_output_path(
        "output.serving_memory_pressure_csv",
        section.serving_memory_pressure_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_timeline_csv_path = parse_run_output_path(
        "output.serving_timeline_csv",
        section.serving_timeline_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_occupancy_csv_path = parse_run_output_path(
        "output.serving_occupancy_csv",
        section.serving_occupancy_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_placement_evidence_csv_path = parse_run_output_path(
        "output.serving_placement_evidence_csv",
        section.serving_placement_evidence_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_worker_evidence_csv_path = parse_run_output_path(
        "output.serving_worker_evidence_csv",
        section.serving_worker_evidence_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_rejections_csv_path = parse_run_output_path(
        "output.serving_rejections_csv",
        section.serving_rejections_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_route_paths_csv_path = parse_run_output_path(
        "output.serving_route_paths_csv",
        section.serving_route_paths_csv_path.as_deref(),
        base_dir,
    )?;
    let kv_route_resources_csv_path = parse_run_output_path(
        "output.kv_route_resources_csv",
        section.kv_route_resources_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_bottlenecks_csv_path = parse_run_output_path(
        "output.serving_bottlenecks_csv",
        section.serving_bottlenecks_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_phase_calibration_csv_path = parse_run_output_path(
        "output.serving_phase_calibration_csv",
        section.serving_phase_calibration_csv_path.as_deref(),
        base_dir,
    )?;
    let serving_approximations_csv_path = parse_run_output_path(
        "output.serving_approximations_csv",
        section.serving_approximations_csv_path.as_deref(),
        base_dir,
    )?;
    let calibration_residuals_csv_path = parse_run_output_path(
        "output.calibration_residuals_csv",
        section.calibration_residuals_csv_path.as_deref(),
        base_dir,
    )?;
    let scenario_sensitivity_csv_path = parse_run_output_path(
        "output.scenario_sensitivity_csv",
        section.scenario_sensitivity_csv_path.as_deref(),
        base_dir,
    )?;
    let rank_sensitivity_csv_path = parse_run_output_path(
        "output.rank_sensitivity_csv",
        section.rank_sensitivity_csv_path.as_deref(),
        base_dir,
    )?;

    Ok(RunOutputConfig {
        format: section.format.map(|format| normalize(&format)),
        top_k: section.top_k,
        output_dir,
        output_profile: section.output_profile.map(|profile| normalize(&profile)),
        request_metrics_csv_path,
        request_lifecycle_events_csv_path,
        serving_metrics_csv_path,
        serving_metric_breakdowns_csv_path,
        serving_services_csv_path,
        serving_utilization_csv_path,
        serving_memory_pressure_csv_path,
        serving_timeline_csv_path,
        serving_occupancy_csv_path,
        serving_placement_evidence_csv_path,
        serving_worker_evidence_csv_path,
        serving_rejections_csv_path,
        serving_route_paths_csv_path,
        kv_route_resources_csv_path,
        serving_bottlenecks_csv_path,
        serving_phase_calibration_csv_path,
        serving_approximations_csv_path,
        calibration_residuals_csv_path,
        scenario_sensitivity_csv_path,
        rank_sensitivity_csv_path,
        trace: section.trace,
        trace_limit: section.trace_limit,
        request_limit: section.request_limit,
        occupancy: section.occupancy,
        occupancy_buckets: section.occupancy_buckets,
        occupancy_resource_limit: section.occupancy_resource_limit,
        critical_path: section.critical_path,
        critical_path_limit: section.critical_path_limit,
    })
}

fn parse_run_output_path(
    name: &str,
    path: Option<&str>,
    base_dir: Option<&Path>,
) -> Result<Option<PathBuf>, ConfigError> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.trim().is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }
    Ok(Some(resolve_config_path(path, base_dir)))
}

fn parse_run_search_budget(
    section: Option<RunSearchBudgetSection>,
) -> Result<RunSearchBudgetConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunSearchBudgetConfig::default());
    };
    validate_positive_usize(
        "search.max_parallelism_candidates",
        section.max_parallelism_candidates,
    )?;
    validate_positive_usize(
        "search.max_prefill_candidates",
        section.max_prefill_candidates,
    )?;
    validate_positive_usize(
        "search.max_decode_candidates",
        section.max_decode_candidates,
    )?;
    validate_positive_usize("search.max_serving_pairs", section.max_serving_pairs)?;

    Ok(RunSearchBudgetConfig {
        max_parallelism_candidates: section.max_parallelism_candidates,
        max_prefill_candidates: section.max_prefill_candidates,
        max_decode_candidates: section.max_decode_candidates,
        max_serving_pairs: section.max_serving_pairs,
        max_runtime_ms: section.max_runtime_ms,
        retain_rejected_candidates: section.retain_rejected_candidates,
    })
}

fn validate_positive_usize(name: &str, value: Option<usize>) -> Result<(), ConfigError> {
    if let Some(value) = value
        && value == 0
    {
        return Err(ConfigError::new(format!(
            "{name} must be greater than zero"
        )));
    }
    Ok(())
}

fn parse_run_scenarios(
    sections: Option<Vec<RunScenarioSection>>,
    base_dir: Option<&Path>,
) -> Result<Vec<RunScenarioConfig>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut scenarios = Vec::with_capacity(sections.len());
    let mut names = HashSet::new();
    for (idx, section) in sections.into_iter().enumerate() {
        let default_name = format!("scenario-{}", idx + 1);
        let name = section
            .name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(&default_name)
            .to_string();
        if !names.insert(name.clone()) {
            return Err(ConfigError::new(format!(
                "scenarios[{idx}].name '{name}' is duplicated"
            )));
        }
        validate_positive_optional_u32(
            &format!("scenarios[{idx}].request_count"),
            section.request_count,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].arrival_gap_scale"),
            section.arrival_gap_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].arrival_rate_scale"),
            section.arrival_rate_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].batch_size_scale"),
            section.batch_size_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].prompt_tokens_scale"),
            section.prompt_tokens_scale,
        )?;
        validate_positive_optional_f64(
            &format!("scenarios[{idx}].decode_tokens_scale"),
            section.decode_tokens_scale,
        )?;
        let calibration_profile_path = parse_run_scenario_calibration_profile_path(
            idx,
            section.calibration_profile.as_deref(),
            base_dir,
        )?;
        let calibration = parse_run_scenario_calibration(idx, section.calibration.as_ref())?;
        let topology = parse_run_scenario_topology(idx, section.topology.as_ref())?;
        scenarios.push(RunScenarioConfig {
            name,
            request_count: section.request_count,
            arrival_gap_scale: section.arrival_gap_scale,
            arrival_rate_scale: section.arrival_rate_scale,
            batch_size_scale: section.batch_size_scale,
            prompt_tokens_scale: section.prompt_tokens_scale,
            decode_tokens_scale: section.decode_tokens_scale,
            calibration_profile_path,
            calibration,
            topology,
        });
    }
    Ok(scenarios)
}

fn parse_run_scenario_calibration_profile_path(
    idx: usize,
    path: Option<&str>,
    base_dir: Option<&Path>,
) -> Result<Option<PathBuf>, ConfigError> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.trim().is_empty() {
        return Err(ConfigError::new(format!(
            "scenarios[{idx}].calibration_profile must not be empty"
        )));
    }
    Ok(Some(resolve_config_path(path, base_dir)))
}

fn parse_run_scenario_calibration(
    idx: usize,
    section: Option<&CalibrationSection>,
) -> Result<RunScenarioCalibrationConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunScenarioCalibrationConfig::default());
    };
    let prefix = format!("scenarios[{idx}].calibration");
    validate_positive_fraction_optional_f64(
        &format!("{prefix}.compute_efficiency"),
        section.compute_efficiency,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.prefill_compute_scale"),
        section.prefill_compute_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.decode_compute_scale"),
        section.decode_compute_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.decode_memory_bandwidth_scale"),
        section.decode_memory_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.collective_latency_scale"),
        section.collective_latency_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.collective_bandwidth_scale"),
        section.collective_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.kv_transfer_scale"),
        section.kv_transfer_scale,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.scheduler_overhead_us"),
        section.scheduler_overhead_us,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_temporary_fraction"),
        section.serving_memory_temporary_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_activation_communication_fraction"),
        section.serving_memory_activation_communication_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_weight_communication_fraction"),
        section.serving_memory_weight_communication_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_runtime_reserve_fraction"),
        section.serving_memory_runtime_reserve_fraction,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.serving_memory_fragmentation_fraction"),
        section.serving_memory_fragmentation_fraction,
    )?;
    validate_positive_optional_u32(
        &format!("{prefix}.serving_pipeline_depth"),
        section.serving_pipeline_depth,
    )?;
    validate_nonnegative_optional_f64(
        &format!("{prefix}.request_arrival_gap_s"),
        section.request_arrival_gap_s,
    )?;

    Ok(calibration_overrides_from_section(Some(section)))
}

fn parse_run_scenario_topology(
    idx: usize,
    section: Option<&RunScenarioTopologySection>,
) -> Result<RunScenarioTopologyConfig, ConfigError> {
    let Some(section) = section else {
        return Ok(RunScenarioTopologyConfig::default());
    };
    let prefix = format!("scenarios[{idx}].topology");
    validate_positive_optional_f64(
        &format!("{prefix}.interconnect_bandwidth_scale"),
        section.interconnect_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.interconnect_latency_scale"),
        section.interconnect_latency_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{prefix}.nic_bandwidth_scale"),
        section.nic_bandwidth_scale,
    )?;
    let node_states = parse_run_scenario_node_state_overlays(
        &format!("{prefix}.node_states"),
        section.node_states.as_deref(),
    )?;
    let disabled_gpus = parse_run_scenario_gpu_overlays(
        &format!("{prefix}.disabled_gpus"),
        section.disabled_gpus.as_deref(),
    )?;
    let disabled_nics = parse_run_scenario_nic_overlays(
        &format!("{prefix}.disabled_nics"),
        section.disabled_nics.as_deref(),
    )?;
    let degraded_gpus = parse_run_scenario_gpu_degradation_overlays(
        &format!("{prefix}.degraded_gpus"),
        section.degraded_gpus.as_deref(),
    )?;
    let degraded_nics = parse_run_scenario_nic_degradation_overlays(
        &format!("{prefix}.degraded_nics"),
        section.degraded_nics.as_deref(),
    )?;
    let degraded_rails = parse_run_scenario_rail_degradation_overlays(
        &format!("{prefix}.degraded_rails"),
        section.degraded_rails.as_deref(),
    )?;
    let degraded_links = parse_run_scenario_link_degradation_overlays(
        &format!("{prefix}.degraded_links"),
        section.degraded_links.as_deref(),
    )?;
    Ok(RunScenarioTopologyConfig {
        interconnect_bandwidth_scale: section.interconnect_bandwidth_scale,
        interconnect_latency_scale: section.interconnect_latency_scale,
        nic_bandwidth_scale: section.nic_bandwidth_scale,
        node_states,
        disabled_gpus,
        disabled_nics,
        degraded_gpus,
        degraded_nics,
        degraded_rails,
        degraded_links,
    })
}

fn parse_run_scenario_node_state_overlays(
    name: &str,
    sections: Option<&[RunScenarioNodeStateOverlaySection]>,
) -> Result<Vec<RunScenarioNodeStateOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        let state = parse_run_scenario_node_state(&prefix, section.state.as_deref())?;
        let overlay = RunScenarioNodeStateOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            state,
        };
        validate_run_scenario_node_overlay_selectors(&prefix, &overlay)?;
        overlays.push(overlay);
    }
    Ok(overlays)
}

fn parse_run_scenario_node_state(
    prefix: &str,
    state: Option<&str>,
) -> Result<RunScenarioNodeState, ConfigError> {
    let Some(state) = state.map(str::trim).filter(|state| !state.is_empty()) else {
        return Err(ConfigError::new(format!("{prefix}.state is required")));
    };

    match normalize(state).as_str() {
        "disabled" | "offline" | "unavailable" => Ok(RunScenarioNodeState::Disabled),
        "maintenance" | "maintenance_mode" => Ok(RunScenarioNodeState::Maintenance),
        "draining" | "drain" => Ok(RunScenarioNodeState::Draining),
        "reserved" | "reserved_capacity" => Ok(RunScenarioNodeState::Reserved),
        _ => Err(ConfigError::new(format!(
            "{prefix}.state '{state}' is unsupported; supported states are disabled, maintenance, draining, and reserved"
        ))),
    }
}

fn parse_run_scenario_gpu_overlays(
    name: &str,
    sections: Option<&[RunScenarioGpuResourceOverlaySection]>,
) -> Result<Vec<RunScenarioGpuResourceOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        overlays.push(RunScenarioGpuResourceOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            gpu_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "gpu",
                section.gpu,
                section.gpus.as_deref(),
            )?,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

fn parse_run_scenario_nic_overlays(
    name: &str,
    sections: Option<&[RunScenarioNicResourceOverlaySection]>,
) -> Result<Vec<RunScenarioNicResourceOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        overlays.push(RunScenarioNicResourceOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            nic_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "nic",
                section.nic,
                section.nics.as_deref(),
            )?,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

fn parse_run_scenario_gpu_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioGpuDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioGpuDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(&format!("{prefix}.compute_scale"), section.compute_scale)?;
        validate_positive_optional_f64(
            &format!("{prefix}.hbm_bandwidth_scale"),
            section.hbm_bandwidth_scale,
        )?;
        validate_positive_optional_f64(
            &format!("{prefix}.hbm_capacity_scale"),
            section.hbm_capacity_scale,
        )?;
        if section.compute_scale.is_none()
            && section.hbm_bandwidth_scale.is_none()
            && section.hbm_capacity_scale.is_none()
        {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of compute_scale, hbm_bandwidth_scale, or hbm_capacity_scale"
            )));
        }
        overlays.push(RunScenarioGpuDegradationOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            gpu_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "gpu",
                section.gpu,
                section.gpus.as_deref(),
            )?,
            compute_scale: section.compute_scale,
            hbm_bandwidth_scale: section.hbm_bandwidth_scale,
            hbm_capacity_scale: section.hbm_capacity_scale,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

fn parse_run_scenario_nic_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioNicDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioNicDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(
            &format!("{prefix}.bandwidth_scale"),
            section.bandwidth_scale,
        )?;
        validate_positive_optional_f64(&format!("{prefix}.latency_scale"), section.latency_scale)?;
        if section.bandwidth_scale.is_none() && section.latency_scale.is_none() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of bandwidth_scale or latency_scale"
            )));
        }
        overlays.push(RunScenarioNicDegradationOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            nic_ids: parse_run_scenario_overlay_resource_ids(
                &prefix,
                "nic",
                section.nic,
                section.nics.as_deref(),
            )?,
            bandwidth_scale: section.bandwidth_scale,
            latency_scale: section.latency_scale,
        });
        validate_run_scenario_overlay_selectors(&prefix, overlays.last().unwrap())?;
    }
    Ok(overlays)
}

fn parse_run_scenario_rail_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioRailDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioRailDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(
            &format!("{prefix}.bandwidth_scale"),
            section.bandwidth_scale,
        )?;
        validate_positive_optional_f64(&format!("{prefix}.latency_scale"), section.latency_scale)?;
        if section.bandwidth_scale.is_none() && section.latency_scale.is_none() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of bandwidth_scale or latency_scale"
            )));
        }
        let rails =
            parse_run_scenario_link_overlay_rails(&prefix, section.rail, section.rails.as_deref())?;
        if rails.is_empty() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one rail or rails selector"
            )));
        }
        overlays.push(RunScenarioRailDegradationOverlay {
            node_ids: parse_run_scenario_overlay_node_ids(
                &prefix,
                section.node,
                section.nodes.as_deref(),
            )?,
            node_groups: parse_run_scenario_overlay_node_groups(
                &prefix,
                section.group.as_deref(),
                section.groups.as_deref(),
            )?,
            node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.node_tags"),
                section.node_tag.as_deref(),
                section.node_tags.as_deref(),
            )?,
            racks: parse_optional_topology_domains(
                &format!("{prefix}.racks"),
                section.rack.as_deref(),
                section.racks.as_deref(),
            )?,
            islands: parse_optional_topology_domains(
                &format!("{prefix}.islands"),
                section.island.as_deref(),
                section.islands.as_deref(),
            )?,
            failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.failure_domains"),
                section.failure_domain.as_deref(),
                section.failure_domains.as_deref(),
            )?,
            rails,
            bandwidth_scale: section.bandwidth_scale,
            latency_scale: section.latency_scale,
        });
    }
    Ok(overlays)
}

fn parse_run_scenario_link_degradation_overlays(
    name: &str,
    sections: Option<&[RunScenarioLinkDegradationOverlaySection]>,
) -> Result<Vec<RunScenarioLinkDegradationOverlay>, ConfigError> {
    let Some(sections) = sections else {
        return Ok(Vec::new());
    };
    let mut overlays = Vec::with_capacity(sections.len());
    for (idx, section) in sections.iter().enumerate() {
        let prefix = format!("{name}[{idx}]");
        validate_positive_optional_f64(
            &format!("{prefix}.bandwidth_scale"),
            section.bandwidth_scale,
        )?;
        validate_positive_optional_f64(&format!("{prefix}.latency_scale"), section.latency_scale)?;
        if section.bandwidth_scale.is_none() && section.latency_scale.is_none() {
            return Err(ConfigError::new(format!(
                "{prefix} must specify at least one of bandwidth_scale or latency_scale"
            )));
        }
        let rails =
            parse_run_scenario_link_overlay_rails(&prefix, section.rail, section.rails.as_deref())?;
        let overlay = RunScenarioLinkDegradationOverlay {
            from_node_ids: parse_run_scenario_overlay_node_ids(
                &format!("{prefix}.from"),
                section.from,
                section.from_nodes.as_deref(),
            )?,
            from_node_groups: parse_run_scenario_overlay_node_groups(
                &format!("{prefix}.from"),
                section.from_group.as_deref(),
                section.from_groups.as_deref(),
            )?,
            from_node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.from_node_tags"),
                section.from_node_tag.as_deref(),
                section.from_node_tags.as_deref(),
            )?,
            from_racks: parse_optional_topology_domains(
                &format!("{prefix}.from_racks"),
                section.from_rack.as_deref(),
                section.from_racks.as_deref(),
            )?,
            from_islands: parse_optional_topology_domains(
                &format!("{prefix}.from_islands"),
                section.from_island.as_deref(),
                section.from_islands.as_deref(),
            )?,
            from_failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.from_failure_domains"),
                section.from_failure_domain.as_deref(),
                section.from_failure_domains.as_deref(),
            )?,
            from_gpus: parse_run_scenario_overlay_resource_ids(
                &format!("{prefix}.from"),
                "gpu",
                section.from_gpu,
                section.from_gpus.as_deref(),
            )?,
            to_node_ids: parse_run_scenario_overlay_node_ids(
                &format!("{prefix}.to"),
                section.to,
                section.to_nodes.as_deref(),
            )?,
            to_node_groups: parse_run_scenario_overlay_node_groups(
                &format!("{prefix}.to"),
                section.to_group.as_deref(),
                section.to_groups.as_deref(),
            )?,
            to_node_tags: parse_optional_normalized_labels(
                &format!("{prefix}.to_node_tags"),
                section.to_node_tag.as_deref(),
                section.to_node_tags.as_deref(),
            )?,
            to_racks: parse_optional_topology_domains(
                &format!("{prefix}.to_racks"),
                section.to_rack.as_deref(),
                section.to_racks.as_deref(),
            )?,
            to_islands: parse_optional_topology_domains(
                &format!("{prefix}.to_islands"),
                section.to_island.as_deref(),
                section.to_islands.as_deref(),
            )?,
            to_failure_domains: parse_optional_topology_domains(
                &format!("{prefix}.to_failure_domains"),
                section.to_failure_domain.as_deref(),
                section.to_failure_domains.as_deref(),
            )?,
            to_gpus: parse_run_scenario_overlay_resource_ids(
                &format!("{prefix}.to"),
                "gpu",
                section.to_gpu,
                section.to_gpus.as_deref(),
            )?,
            rails,
            bandwidth_scale: section.bandwidth_scale,
            latency_scale: section.latency_scale,
        };
        validate_run_scenario_link_overlay_selectors(&prefix, &overlay)?;
        overlays.push(overlay);
    }
    Ok(overlays)
}

trait RunScenarioResourceOverlay {
    fn node_ids(&self) -> &[u32];
    fn node_groups(&self) -> &[String];
    fn node_tags(&self) -> &[String];
    fn racks(&self) -> &[String];
    fn islands(&self) -> &[String];
    fn failure_domains(&self) -> &[String];
    fn resource_ids(&self) -> &[u32];
}

impl RunScenarioResourceOverlay for RunScenarioGpuResourceOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.gpu_ids
    }
}

impl RunScenarioResourceOverlay for RunScenarioNicResourceOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.nic_ids
    }
}

impl RunScenarioResourceOverlay for RunScenarioGpuDegradationOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.gpu_ids
    }
}

impl RunScenarioResourceOverlay for RunScenarioNicDegradationOverlay {
    fn node_ids(&self) -> &[u32] {
        &self.node_ids
    }

    fn node_groups(&self) -> &[String] {
        &self.node_groups
    }

    fn node_tags(&self) -> &[String] {
        &self.node_tags
    }

    fn racks(&self) -> &[String] {
        &self.racks
    }

    fn islands(&self) -> &[String] {
        &self.islands
    }

    fn failure_domains(&self) -> &[String] {
        &self.failure_domains
    }

    fn resource_ids(&self) -> &[u32] {
        &self.nic_ids
    }
}

fn validate_run_scenario_overlay_selectors(
    name: &str,
    overlay: &impl RunScenarioResourceOverlay,
) -> Result<(), ConfigError> {
    if overlay.node_ids().is_empty()
        && overlay.node_groups().is_empty()
        && overlay.node_tags().is_empty()
        && overlay.racks().is_empty()
        && overlay.islands().is_empty()
        && overlay.failure_domains().is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one node, nodes, group, groups, node_tag, rack, island, or failure_domain selector"
        )));
    }
    if overlay.resource_ids().is_empty() {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one resource id"
        )));
    }
    Ok(())
}

fn validate_optional_placement_for_cluster(
    name: &str,
    cluster: &Cluster,
    placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    let Some(placement) = placement else {
        return Ok(());
    };
    let mut used_gpus = BTreeSet::new();
    for (rank, addr) in placement.rank_to_gpu.iter().copied().enumerate() {
        let node = cluster.nodes.get(&addr.node_id).ok_or_else(|| {
            ConfigError::new(format!(
                "{name}.ranks[{rank}] references unknown node id {}",
                addr.node_id
            ))
        })?;
        if !node.gpus.contains_key(&addr.local_gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.ranks[{rank}] references unknown local GPU id {} on node {}",
                addr.local_gpu_id, addr.node_id
            )));
        }
        if !cluster.is_gpu_available(addr) {
            return Err(ConfigError::new(format!(
                "{name}.ranks[{rank}] references unavailable node {} gpu {}",
                addr.node_id, addr.local_gpu_id
            )));
        }
        if !used_gpus.insert(addr) {
            return Err(ConfigError::new(format!(
                "{name}.ranks maps multiple ranks to node {} gpu {}",
                addr.node_id, addr.local_gpu_id
            )));
        }
    }
    Ok(())
}

fn model_dtype_label(dtype: DType) -> &'static str {
    dtype.label()
}

fn gpu_profile_supports_dtype(profile: &GpuProfile, dtype: DType) -> bool {
    match dtype {
        DType::Fp8 => profile.peak_f8_flops.is_some(),
        DType::Fp16 | DType::Bf16 | DType::Int8 => true,
    }
}

fn available_gpus_supporting_dtype(cluster: &Cluster, dtype: DType) -> u32 {
    cluster
        .nodes
        .iter()
        .flat_map(|(&node_id, node)| {
            node.gpus.keys().copied().map(move |local_gpu_id| GpuAddr {
                node_id,
                local_gpu_id,
            })
        })
        .filter(|addr| {
            cluster.is_gpu_available(*addr)
                && cluster
                    .gpu_profile(*addr)
                    .is_some_and(|profile| gpu_profile_supports_dtype(&profile, dtype))
        })
        .count()
        .min(u32::MAX as usize) as u32
}

fn validate_cluster_dtype_support(cluster: &Cluster, dtype: DType) -> Result<u32, ConfigError> {
    let dtype_gpus = available_gpus_supporting_dtype(cluster, dtype);
    if dtype_gpus > 0 {
        return Ok(dtype_gpus);
    }

    Err(ConfigError::new(format!(
        "model.dtype {} requires at least one available GPU with {} tensor throughput, but the cluster has no available {}-capable GPUs",
        model_dtype_label(dtype),
        model_dtype_label(dtype),
        model_dtype_label(dtype)
    )))
}

fn validate_optional_placement_dtype_support(
    name: &str,
    cluster: &Cluster,
    placement: Option<&RankPlacement>,
    dtype: DType,
) -> Result<(), ConfigError> {
    let Some(placement) = placement else {
        return Ok(());
    };

    for (rank, addr) in placement.rank_to_gpu.iter().copied().enumerate() {
        let Some(profile) = cluster.gpu_profile(addr) else {
            continue;
        };
        if !gpu_profile_supports_dtype(&profile, dtype) {
            return Err(ConfigError::new(format!(
                "{name}.ranks[{rank}] uses node {} gpu {} ({}) which does not support model.dtype {}",
                addr.node_id,
                addr.local_gpu_id,
                profile.label,
                model_dtype_label(dtype)
            )));
        }
    }

    Ok(())
}

fn validate_optional_placement_rank_count(
    name: &str,
    search: &SearchSpace,
    placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    let Some(placement) = placement else {
        return Ok(());
    };
    let placement_ranks = placement.rank_to_gpu.len() as u32;
    if search_space_rank_counts(search).contains(&placement_ranks) {
        return Ok(());
    }
    Err(ConfigError::new(format!(
        "{name}.ranks defines {placement_ranks} ranks but no search candidate has that total rank count"
    )))
}

fn search_space_rank_counts(search: &SearchSpace) -> BTreeSet<u32> {
    let mut counts = BTreeSet::new();
    for tensor in &search.tensor_ranks {
        for pipeline in &search.pipeline_ranks {
            for expert in &search.expert_ranks {
                for data in &search.data_ranks {
                    counts.insert(
                        tensor
                            .saturating_mul(*pipeline)
                            .saturating_mul(*expert)
                            .saturating_mul(*data),
                    );
                }
            }
        }
    }
    counts
}

fn validate_run_scenario_link_overlay_selectors(
    name: &str,
    overlay: &RunScenarioLinkDegradationOverlay,
) -> Result<(), ConfigError> {
    if overlay.from_node_ids.is_empty()
        && overlay.from_node_groups.is_empty()
        && overlay.from_node_tags.is_empty()
        && overlay.from_racks.is_empty()
        && overlay.from_islands.is_empty()
        && overlay.from_failure_domains.is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one from, from_nodes, from_group, from_groups, from_node_tag, from_rack, from_island, or from_failure_domain selector"
        )));
    }
    if overlay.to_node_ids.is_empty()
        && overlay.to_node_groups.is_empty()
        && overlay.to_node_tags.is_empty()
        && overlay.to_racks.is_empty()
        && overlay.to_islands.is_empty()
        && overlay.to_failure_domains.is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one to, to_nodes, to_group, to_groups, to_node_tag, to_rack, to_island, or to_failure_domain selector"
        )));
    }
    Ok(())
}

fn validate_run_scenario_node_overlay_selectors(
    name: &str,
    overlay: &RunScenarioNodeStateOverlay,
) -> Result<(), ConfigError> {
    if overlay.node_ids.is_empty()
        && overlay.node_groups.is_empty()
        && overlay.node_tags.is_empty()
        && overlay.racks.is_empty()
        && overlay.islands.is_empty()
        && overlay.failure_domains.is_empty()
    {
        return Err(ConfigError::new(format!(
            "{name} must specify at least one node, nodes, group, groups, node_tag, rack, island, or failure_domain selector"
        )));
    }
    Ok(())
}

fn parse_run_scenario_link_overlay_rails(
    name: &str,
    rail: Option<u32>,
    rails: Option<&[u32]>,
) -> Result<Vec<u32>, ConfigError> {
    let mut ids = Vec::new();
    if let Some(rail) = rail {
        ids.push(rail);
    }
    if let Some(rails) = rails {
        ids.extend_from_slice(rails);
    }
    sort_dedup_or_error(&mut ids, &format!("{name}.rails"))?;
    Ok(ids)
}

fn parse_run_scenario_overlay_node_ids(
    name: &str,
    node: Option<u32>,
    nodes: Option<&[u32]>,
) -> Result<Vec<u32>, ConfigError> {
    let mut ids = Vec::new();
    if let Some(node) = node {
        ids.push(node);
    }
    if let Some(nodes) = nodes {
        ids.extend_from_slice(nodes);
    }
    sort_dedup_or_error(&mut ids, &format!("{name}.nodes"))?;
    Ok(ids)
}

fn parse_run_scenario_overlay_node_groups(
    name: &str,
    group: Option<&str>,
    groups: Option<&[String]>,
) -> Result<Vec<String>, ConfigError> {
    let mut labels = Vec::new();
    if let Some(group) = group {
        labels.push(parse_nonempty_overlay_label(
            &format!("{name}.group"),
            group,
        )?);
    }
    if let Some(groups) = groups {
        for (idx, group) in groups.iter().enumerate() {
            labels.push(parse_nonempty_overlay_label(
                &format!("{name}.groups[{idx}]"),
                group,
            )?);
        }
    }
    labels.sort();
    for pair in labels.windows(2) {
        if pair[0] == pair[1] {
            return Err(ConfigError::new(format!(
                "{name}.groups duplicates node group '{}'",
                pair[0]
            )));
        }
    }
    Ok(labels)
}

fn parse_nonempty_overlay_label(name: &str, value: &str) -> Result<String, ConfigError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }
    Ok(value.to_string())
}

fn parse_run_scenario_overlay_resource_ids(
    name: &str,
    resource_label: &str,
    resource: Option<u32>,
    resources: Option<&[u32]>,
) -> Result<Vec<u32>, ConfigError> {
    let mut ids = Vec::new();
    if let Some(resource) = resource {
        ids.push(resource);
    }
    if let Some(resources) = resources {
        ids.extend_from_slice(resources);
    }
    sort_dedup_or_error(&mut ids, &format!("{name}.{resource_label}s"))?;
    Ok(ids)
}

fn sort_dedup_or_error(ids: &mut [u32], name: &str) -> Result<(), ConfigError> {
    ids.sort_unstable();
    for pair in ids.windows(2) {
        if pair[0] == pair[1] {
            return Err(ConfigError::new(format!(
                "{name} duplicates id {}",
                pair[0]
            )));
        }
    }
    Ok(())
}

fn calibration_overrides_from_section(
    section: Option<&CalibrationSection>,
) -> RunScenarioCalibrationConfig {
    let Some(section) = section else {
        return RunScenarioCalibrationConfig::default();
    };

    RunScenarioCalibrationConfig {
        compute_efficiency: section.compute_efficiency,
        prefill_compute_scale: section.prefill_compute_scale,
        decode_compute_scale: section.decode_compute_scale,
        decode_memory_bandwidth_scale: section.decode_memory_bandwidth_scale,
        collective_latency_scale: section.collective_latency_scale,
        collective_bandwidth_scale: section.collective_bandwidth_scale,
        kv_transfer_scale: section.kv_transfer_scale,
        scheduler_overhead_us: section.scheduler_overhead_us,
        serving_memory_temporary_fraction: section.serving_memory_temporary_fraction,
        serving_memory_activation_communication_fraction: section
            .serving_memory_activation_communication_fraction,
        serving_memory_weight_communication_fraction: section
            .serving_memory_weight_communication_fraction,
        serving_memory_runtime_reserve_fraction: section.serving_memory_runtime_reserve_fraction,
        serving_memory_fragmentation_fraction: section.serving_memory_fragmentation_fraction,
        serving_pipeline_depth: section.serving_pipeline_depth,
        request_arrival_gap_s: section.request_arrival_gap_s,
        allow_compute_comm_overlap: section.allow_compute_comm_overlap,
    }
}

pub fn validate_workload_for_cluster(
    cluster: &Cluster,
    workload: &WorkloadConfig,
) -> Result<(), ConfigError> {
    let cluster_gpus = cluster.available_gpus();
    if cluster_gpus == 0 {
        return Err(ConfigError::new(
            "cluster must contain at least one available GPU for workload validation",
        ));
    }
    let dtype_gpus = validate_cluster_dtype_support(cluster, workload.model.dtype)?;
    validate_search_space_capacity(
        &format!(
            "search for model.dtype {}",
            model_dtype_label(workload.model.dtype)
        ),
        &workload.search_space,
        dtype_gpus,
    )?;
    validate_optional_placement_for_cluster("placement", cluster, workload.placement.as_ref())?;
    validate_optional_placement_dtype_support(
        "placement",
        cluster,
        workload.placement.as_ref(),
        workload.model.dtype,
    )?;
    validate_optional_placement_rank_count(
        "placement",
        &workload.search_space,
        workload.placement.as_ref(),
    )?;

    let Some(serving) = &workload.serving else {
        return Ok(());
    };
    validate_optional_placement_for_cluster(
        "serving.prefill_placement",
        cluster,
        workload.serving_prefill_placement.as_ref(),
    )?;
    validate_optional_placement_dtype_support(
        "serving.prefill_placement",
        cluster,
        workload.serving_prefill_placement.as_ref(),
        workload.model.dtype,
    )?;
    validate_optional_placement_for_cluster(
        "serving.decode_placement",
        cluster,
        workload.serving_decode_placement.as_ref(),
    )?;
    validate_optional_placement_dtype_support(
        "serving.decode_placement",
        cluster,
        workload.serving_decode_placement.as_ref(),
        workload.model.dtype,
    )?;
    validate_optional_placement_rank_count(
        "serving.prefill_placement",
        &serving.search.prefill,
        workload.serving_prefill_placement.as_ref(),
    )?;
    validate_optional_placement_rank_count(
        "serving.decode_placement",
        &serving.search.decode,
        workload.serving_decode_placement.as_ref(),
    )?;

    for (idx, candidate) in serving.pool_candidates.iter().enumerate() {
        validate_pool_candidate_for_cluster(
            &format!("serving.pool_candidates[{idx}]"),
            cluster,
            candidate,
            serving.deployment_mode,
            &serving.search,
            workload.require_routable_serving_pools,
            workload.model.dtype,
        )?;
    }
    if let Some(pool_search) = &serving.pool_search {
        validate_pool_search_for_cluster(
            cluster,
            pool_search,
            serving.deployment_mode,
            &serving.search,
            workload.require_routable_serving_pools,
            workload.model.dtype,
        )?;
    }
    validate_explicit_serving_placements_for_configured_pools(
        cluster,
        serving,
        workload.serving_prefill_placement.as_ref(),
        workload.serving_decode_placement.as_ref(),
    )?;
    validate_serving_route_constraints_for_configured_pools(
        cluster,
        serving,
        workload.serving_prefill_placement.as_ref(),
        workload.serving_decode_placement.as_ref(),
    )?;

    Ok(())
}

fn parse_workload_with_base_dir(
    contents: &str,
    base_dir: Option<&Path>,
) -> Result<WorkloadConfig, ConfigError> {
    let file: WorkloadFile = toml::from_str(contents)
        .map_err(|err| ConfigError::new(format!("invalid workload TOML: {err}")))?;
    validate_schema_version("workload", file.schema_version)?;

    let serving_stack = nonempty_metadata(file.serving_stack.clone()).or_else(|| {
        file.serving
            .as_ref()
            .and_then(|serving| nonempty_metadata(serving.serving_stack.clone()))
    });
    let serving_runtime_features = parse_serving_runtime_features(&file)?;
    let search_space = match file.search {
        Some(search) => parse_search("search", search)?,
        None => {
            let serving = file
                .serving
                .as_ref()
                .ok_or_else(|| ConfigError::new("either [search] or [serving] must be provided"))?;
            parse_search(
                "serving.prefill_search",
                serving.prefill_search.clone().ok_or_else(|| {
                    ConfigError::new("serving.prefill_search is required when [search] is absent")
                })?,
            )?
        }
    };
    let placement = parse_optional_placement("placement", file.placement.as_ref())?;
    let serving_prefill_placement = parse_optional_placement(
        "serving.prefill_placement",
        file.serving
            .as_ref()
            .and_then(|serving| serving.prefill_placement.as_ref()),
    )?;
    let serving_decode_placement = parse_optional_placement(
        "serving.decode_placement",
        file.serving
            .as_ref()
            .and_then(|serving| serving.decode_placement.as_ref()),
    )?;
    let require_routable_serving_pools = file
        .serving
        .as_ref()
        .and_then(|serving| serving.require_routable_pools)
        .unwrap_or(false);
    let serving = file
        .serving
        .map(|serving| parse_serving(serving, &search_space, base_dir))
        .transpose()?;
    let calibration_overrides = calibration_overrides_from_section(file.calibration.as_ref());
    let calibration_profile = load_calibration_profile(file.calibration_profile, base_dir)?;
    let calibration_defaults = calibration_profile
        .as_ref()
        .map(|profile| profile.calibration)
        .unwrap_or_default();
    validate_model_section(&file.model)?;
    validate_positive_optional_f64(
        "model.parameter_count_billion",
        file.model.parameter_count_billion,
    )?;
    let request_phase = inference_phase(&file.request.phase)?;
    validate_request_section(&file.request, request_phase)?;
    let model = ModelSpec {
        layers: file.model.layers,
        hidden_size: file.model.hidden_size,
        attention_heads: file.model.attention_heads,
        kv_heads: file.model.kv_heads,
        vocab_size: file.model.vocab_size,
        parameters: Bytes::from_gigabytes(file.model.parameters_gb),
        parameter_count: file.model.parameter_count_billion.map(|count| count * 1e9),
        dtype: dtype("model.dtype", &file.model.dtype)?,
        kv_dtype: file
            .model
            .kv_dtype
            .as_deref()
            .map(|value| dtype("model.kv_dtype", value))
            .transpose()?,
        experts: file.model.experts.map(|experts| ExpertSpec {
            expert_count: experts.expert_count,
            top_k: experts.top_k,
        }),
    };
    let request = InferenceRequest {
        batch_size: file.request.batch_size,
        prompt_tokens: file.request.prompt_tokens,
        decode_tokens: file.request.decode_tokens,
        max_sequence_tokens: file.request.max_sequence_tokens,
        phase: request_phase,
    };
    let calibration_profile = calibration_profile.map(|profile| profile.metadata);
    let calibration_policy = parse_calibration_policy(file.calibration_policy)?;
    let approximation_policy = parse_approximation_policy(file.approximation_policy)?;
    let calibration_coverage =
        calibration_coverage_report(calibration_profile.as_ref(), &request, serving.as_ref());
    let calibration_warnings = calibration_applicability_warnings(
        calibration_profile.as_ref(),
        &request,
        serving.as_ref(),
    );
    let calibration_invalid_shape_warnings = calibration_invalid_shape_warnings(
        calibration_profile.as_ref(),
        &request,
        serving.as_ref(),
    );
    let calibration_gate_violations = calibration_gate_violations(
        &calibration_policy,
        calibration_profile.as_ref(),
        calibration_coverage.as_ref(),
        &calibration_warnings,
        &calibration_invalid_shape_warnings,
    );

    Ok(WorkloadConfig {
        model_id: optional_nonempty_string(file.model.id),
        serving_stack,
        serving_runtime_features,
        model,
        request,
        search_space,
        placement,
        serving_prefill_placement,
        serving_decode_placement,
        calibration: calibration_with_defaults(file.calibration, calibration_defaults),
        calibration_overrides,
        calibration_policy,
        approximation_policy,
        calibration_profile,
        calibration_coverage,
        calibration_warnings,
        calibration_invalid_shape_warnings,
        calibration_gate_violations,
        require_routable_serving_pools,
        serving,
    })
}

fn validate_schema_version(kind: &str, version: Option<u32>) -> Result<(), ConfigError> {
    match version {
        None | Some(SUPPORTED_SCHEMA_VERSION) => Ok(()),
        Some(version) => Err(ConfigError::new(format!(
            "unsupported {kind} schema_version {version}; supported schema_version is {SUPPORTED_SCHEMA_VERSION}"
        ))),
    }
}

fn parse_custom_cluster(file: ClusterFile) -> Result<Cluster, ConfigError> {
    let node_sections = file.nodes.unwrap_or_default();
    let node_group_sections = file.node_groups.unwrap_or_default();
    if node_sections.is_empty() && node_group_sections.is_empty() {
        return Err(ConfigError::new(
            "custom clusters require at least one [[nodes]] or [[node_groups]] entry",
        ));
    }

    let default_interconnect = file
        .interconnect
        .as_ref()
        .and_then(|section| {
            if section.kind.is_some() || section.variant.is_some() {
                Some(interconnect_profile(section))
            } else {
                None
            }
        })
        .transpose()?;
    let default_network = match (&file.nics, default_interconnect.clone()) {
        (Some(nics), Some(interconnect)) => Some(nics_profile("nics", nics, interconnect)?),
        (Some(nics), None) => Some(nics_profile("nics", nics, default_nic_reference_profile())?),
        _ => None,
    };

    let mut nodes = HashMap::new();
    let mut node_groups = HashMap::new();
    for section in node_sections {
        let topology = parse_node_topology_metadata(
            &format!("nodes[{}]", section.id),
            section.node_tag.as_deref(),
            section.node_tags.as_deref(),
            section.rack.as_deref(),
            section.island.as_deref(),
            section.failure_domain.as_deref(),
        )?;
        let node = build_node(
            &format!("nodes[{}]", section.id),
            topology,
            section.gpu.as_deref(),
            section.gpu_count,
            GpuProfileOverrideConfig {
                hbm_gb: section.hbm_gb,
                hbm_bandwidth_gb_s: section.hbm_bandwidth_gb_s,
                peak_f16_tflops: section.peak_f16_tflops,
                peak_f8_tflops: section.peak_f8_tflops,
            },
            section.gpu_tag,
            section.gpu_tags,
            section.gpus,
            section.gpu_profile_overrides,
            section.disabled_gpus,
            section.gpu_states,
            section.intra.as_deref(),
            section.nics,
            default_interconnect.clone(),
            default_network.clone(),
        )?;
        insert_node(&mut nodes, section.id, node)?;
        if let Some(group) = section.group {
            add_group_node(&mut node_groups, &group, section.id)?;
        }
    }

    for (group_idx, section) in node_group_sections.into_iter().enumerate() {
        if section.count == 0 {
            return Err(ConfigError::new(format!(
                "node_groups[{group_idx}].count must be nonzero"
            )));
        }
        let topology = parse_node_topology_metadata(
            &format!("node_groups[{group_idx}]"),
            section.node_tag.as_deref(),
            section.node_tags.as_deref(),
            section.rack.as_deref(),
            section.island.as_deref(),
            section.failure_domain.as_deref(),
        )?;
        let start_id = section.start_id.unwrap_or_else(|| next_node_id(&nodes));
        for offset in 0..section.count {
            let node_id = start_id + offset;
            let node = build_node(
                &format!("node_groups[{group_idx}]"),
                topology.clone(),
                Some(&section.gpu),
                Some(section.gpu_count),
                GpuProfileOverrideConfig {
                    hbm_gb: section.hbm_gb,
                    hbm_bandwidth_gb_s: section.hbm_bandwidth_gb_s,
                    peak_f16_tflops: section.peak_f16_tflops,
                    peak_f8_tflops: section.peak_f8_tflops,
                },
                section.gpu_tag.clone(),
                section.gpu_tags.clone(),
                None,
                section.gpu_profile_overrides.clone(),
                section.disabled_gpus.clone(),
                section.gpu_states.clone(),
                section.intra.as_deref(),
                section.nics.clone(),
                default_interconnect.clone(),
                default_network.clone(),
            )?;
            insert_node(&mut nodes, node_id, node)?;
            if let Some(label) = &section.label {
                add_group_node(&mut node_groups, label, node_id)?;
            }
        }
    }

    finalize_node_groups(&mut node_groups, &nodes);

    let inter_node_topology = custom_interconnect_topology(
        file.interconnect,
        default_interconnect,
        &nodes,
        &node_groups,
    )?;

    Ok(Cluster {
        nodes,
        node_groups,
        inter_node_topology,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    name: &str,
    topology: NodeTopologyMetadata,
    gpu_name: Option<&str>,
    gpu_count: Option<u32>,
    gpu_profile_override: GpuProfileOverrideConfig,
    gpu_tag: Option<String>,
    gpu_tags: Option<Vec<String>>,
    gpu_sections: Option<Vec<NodeGpuSection>>,
    gpu_profile_override_sections: Option<Vec<GpuProfileOverrideSection>>,
    disabled_gpus: Option<Vec<u32>>,
    gpu_states: Option<Vec<GpuStateSection>>,
    intra: Option<&str>,
    nics: Option<NicsSection>,
    default_interconnect: Option<FabricProfile>,
    default_network: Option<NodeNetworkProfile>,
) -> Result<Node, ConfigError> {
    let inventory = build_node_gpus(
        name,
        gpu_name,
        gpu_count,
        gpu_profile_override,
        gpu_tag,
        gpu_tags,
        gpu_sections,
    )?;
    let gpus = inventory.gpus;
    let gpu_labels = inventory.gpu_labels;
    let mut gpu_profile_overrides = inventory.gpu_profile_overrides;
    merge_gpu_profile_overrides(
        name,
        &mut gpu_profile_overrides,
        gpu_profile_override_sections.as_deref(),
        &gpus,
    )?;
    let mut gpu_operational_states = inventory.gpu_operational_states;
    merge_gpu_operational_states(
        name,
        &mut gpu_operational_states,
        gpu_states.as_deref(),
        &gpus,
    )?;
    let disabled_gpus =
        parse_disabled_gpus(name, disabled_gpus, &gpus, &mut gpu_operational_states)?;
    let representative_gpu = representative_gpu_for_intra(name, &gpus, intra)?;
    let node_interconnect = default_interconnect.unwrap_or_else(default_nic_reference_profile);
    let network = match nics {
        Some(nics) => nics_profile(&format!("{name}.nics"), &nics, node_interconnect)?,
        None => default_network.unwrap_or(NodeNetworkProfile {
            nic_count: 1,
            nic_bandwidth: node_interconnect.bw.unidirectional,
            nic_bandwidth_overrides: Default::default(),
            nic_latency_scale_overrides: Default::default(),
            gpu_to_nic: GpuNicAffinity::Uniform,
            rail_count: 1,
            nic_rail_map: Default::default(),
            gpu_nic_map: Default::default(),
            gpu_numa_map: Default::default(),
            nic_numa_map: Default::default(),
            cross_numa_bandwidth_scale: 1.0,
            cross_numa_latency_scale: 1.0,
            gpu_nic_path_overrides: Default::default(),
            disabled_nics: Default::default(),
            nic_operational_states: Default::default(),
        }),
    };
    validate_network_profile(&format!("{name}.nics"), &gpus, &disabled_gpus, &network)?;

    Ok(Node {
        gpus,
        topology,
        gpu_labels,
        gpu_profile_overrides,
        disabled_gpus,
        gpu_operational_states,
        operational_state: NodeOperationalState::Healthy,
        intra_node_fabric: intra_node_topology(intra, representative_gpu)?,
        network,
    })
}

struct NodeGpuInventory {
    gpus: HashMap<u32, Gpu>,
    gpu_labels: HashMap<u32, BTreeSet<String>>,
    gpu_profile_overrides: HashMap<u32, GpuProfile>,
    gpu_operational_states: HashMap<u32, OperationalState>,
}

#[derive(Clone, Debug, Default)]
struct GpuProfileOverrideConfig {
    hbm_gb: Option<f64>,
    hbm_bandwidth_gb_s: Option<f64>,
    peak_f16_tflops: Option<f64>,
    peak_f8_tflops: Option<f64>,
}

impl GpuProfileOverrideConfig {
    fn has_overrides(&self) -> bool {
        self.hbm_gb.is_some()
            || self.hbm_bandwidth_gb_s.is_some()
            || self.peak_f16_tflops.is_some()
            || self.peak_f8_tflops.is_some()
    }
}

fn build_node_gpus(
    name: &str,
    gpu_name: Option<&str>,
    gpu_count: Option<u32>,
    gpu_profile_override: GpuProfileOverrideConfig,
    gpu_tag: Option<String>,
    gpu_tags: Option<Vec<String>>,
    gpu_sections: Option<Vec<NodeGpuSection>>,
) -> Result<NodeGpuInventory, ConfigError> {
    if let Some(sections) = gpu_sections {
        if gpu_name.is_some()
            || gpu_count.is_some()
            || gpu_profile_override.has_overrides()
            || gpu_tag.is_some()
            || gpu_tags.is_some()
        {
            return Err(ConfigError::new(format!(
                "{name} must set either gpu/gpu_count/node-wide gpu profile fields/gpu_tags or gpus, not both"
            )));
        }
        return explicit_node_gpus(name, sections);
    }

    let gpu_name = gpu_name.ok_or_else(|| {
        ConfigError::new(format!(
            "{name}.gpu is required unless {name}.gpus is provided"
        ))
    })?;
    let gpu_count = gpu_count.ok_or_else(|| {
        ConfigError::new(format!(
            "{name}.gpu_count is required unless {name}.gpus is provided"
        ))
    })?;
    if gpu_count == 0 {
        return Err(ConfigError::new(format!(
            "{name}.gpu_count must be nonzero"
        )));
    }

    let gpu = gpu(gpu_name)?;
    let mut gpus = HashMap::new();
    let mut gpu_labels = HashMap::new();
    let mut gpu_profile_overrides = HashMap::new();
    let gpu_operational_states = HashMap::new();
    let labels = parse_gpu_labels(
        &format!("{name}.gpu_tags"),
        gpu_tag.as_deref(),
        gpu_tags.as_deref(),
    )?;
    let profile_override = gpu_profile_override_from_config(name, gpu, &gpu_profile_override)?;
    for local_gpu_id in 0..gpu_count {
        gpus.insert(local_gpu_id, gpu);
        if !labels.is_empty() {
            gpu_labels.insert(local_gpu_id, labels.clone());
        }
        if let Some(profile) = &profile_override {
            gpu_profile_overrides.insert(local_gpu_id, profile.clone());
        }
    }
    Ok(NodeGpuInventory {
        gpus,
        gpu_labels,
        gpu_profile_overrides,
        gpu_operational_states,
    })
}

fn explicit_node_gpus(
    name: &str,
    sections: Vec<NodeGpuSection>,
) -> Result<NodeGpuInventory, ConfigError> {
    if sections.is_empty() {
        return Err(ConfigError::new(format!("{name}.gpus must not be empty")));
    }

    let mut gpus = HashMap::new();
    let mut gpu_labels = HashMap::new();
    let mut gpu_profile_overrides = HashMap::new();
    let mut gpu_operational_states = HashMap::new();
    for (idx, section) in sections.into_iter().enumerate() {
        if section.id.is_some() && section.start_id.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpus[{idx}] cannot set both id and start_id"
            )));
        }
        let count = section.count.unwrap_or(1);
        if count == 0 {
            return Err(ConfigError::new(format!(
                "{name}.gpus[{idx}].count must be nonzero"
            )));
        }
        let start_id = section
            .start_id
            .or(section.id)
            .unwrap_or_else(|| next_local_gpu_id(&gpus));
        let gpu = gpu(&section.gpu)?;
        let profile_override = gpu_profile_override_from_config(
            &format!("{name}.gpus[{idx}]"),
            gpu,
            &GpuProfileOverrideConfig {
                hbm_gb: section.hbm_gb,
                hbm_bandwidth_gb_s: section.hbm_bandwidth_gb_s,
                peak_f16_tflops: section.peak_f16_tflops,
                peak_f8_tflops: section.peak_f8_tflops,
            },
        )?;
        let labels = parse_gpu_labels(
            &format!("{name}.gpus[{idx}].gpu_tags"),
            section.gpu_tag.as_deref(),
            section.gpu_tags.as_deref(),
        )?;
        let operational_state = section
            .state
            .as_deref()
            .map(|state| {
                parse_resource_operational_state(&format!("{name}.gpus[{idx}].state"), state)
            })
            .transpose()?;
        for offset in 0..count {
            let local_gpu_id = start_id + offset;
            if gpus.insert(local_gpu_id, gpu).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.gpus[{idx}] duplicates local GPU id {local_gpu_id}"
                )));
            }
            if !labels.is_empty() {
                gpu_labels.insert(local_gpu_id, labels.clone());
            }
            if let Some(profile) = &profile_override {
                gpu_profile_overrides.insert(local_gpu_id, profile.clone());
            }
            if let Some(operational_state) = operational_state {
                gpu_operational_states.insert(local_gpu_id, operational_state);
            }
        }
    }

    Ok(NodeGpuInventory {
        gpus,
        gpu_labels,
        gpu_profile_overrides,
        gpu_operational_states,
    })
}

fn gpu_profile_override_from_config(
    name: &str,
    gpu: Gpu,
    config: &GpuProfileOverrideConfig,
) -> Result<Option<GpuProfile>, ConfigError> {
    gpu_profile_override_from_base_config(name, gpu.profile(), config)
}

fn gpu_profile_override_from_base_config(
    name: &str,
    mut profile: GpuProfile,
    config: &GpuProfileOverrideConfig,
) -> Result<Option<GpuProfile>, ConfigError> {
    if !config.has_overrides() {
        return Ok(None);
    }

    validate_positive_optional_f64(&format!("{name}.hbm_gb"), config.hbm_gb)?;
    validate_positive_optional_f64(
        &format!("{name}.hbm_bandwidth_gb_s"),
        config.hbm_bandwidth_gb_s,
    )?;
    validate_positive_optional_f64(&format!("{name}.peak_f16_tflops"), config.peak_f16_tflops)?;
    validate_positive_optional_f64(&format!("{name}.peak_f8_tflops"), config.peak_f8_tflops)?;

    if let Some(hbm_gb) = config.hbm_gb {
        profile.hbm_size = Bytes::from_gigabytes(hbm_gb);
    }
    if let Some(hbm_bandwidth_gb_s) = config.hbm_bandwidth_gb_s {
        profile.hbm_bandwidth = Bandwidth::from_gigabytes_per_sec(hbm_bandwidth_gb_s);
    }
    if let Some(peak_f16_tflops) = config.peak_f16_tflops {
        profile.peak_f16_flops = peak_f16_tflops;
    }
    if let Some(peak_f8_tflops) = config.peak_f8_tflops {
        profile.peak_f8_flops = Some(peak_f8_tflops);
    }

    Ok(Some(profile))
}

fn merge_gpu_profile_overrides(
    name: &str,
    overrides: &mut HashMap<u32, GpuProfile>,
    entries: Option<&[GpuProfileOverrideSection]>,
    gpus: &HashMap<u32, Gpu>,
) -> Result<(), ConfigError> {
    let Some(entries) = entries else {
        return Ok(());
    };

    let mut seen = BTreeSet::new();
    for (idx, entry) in entries.iter().enumerate() {
        if entry.local_gpu_id.is_some() && entry.local_gpu_ids.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_profile_overrides[{idx}] cannot set both gpu and gpus"
            )));
        }
        let gpu_ids = match (entry.local_gpu_id, entry.local_gpu_ids.as_ref()) {
            (Some(gpu_id), None) => vec![gpu_id],
            (None, Some(gpu_ids)) if !gpu_ids.is_empty() => gpu_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides[{idx}].gpus must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides[{idx}] must set gpu or gpus"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        let config = GpuProfileOverrideConfig {
            hbm_gb: entry.hbm_gb,
            hbm_bandwidth_gb_s: entry.hbm_bandwidth_gb_s,
            peak_f16_tflops: entry.peak_f16_tflops,
            peak_f8_tflops: entry.peak_f8_tflops,
        };
        if !config.has_overrides() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_profile_overrides[{idx}] must specify at least one profile override"
            )));
        }

        for gpu_id in gpu_ids {
            let Some(gpu) = gpus.get(&gpu_id).copied() else {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides[{idx}] references unknown local GPU id {gpu_id}"
                )));
            };
            if !seen.insert(gpu_id) {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_profile_overrides duplicates local GPU id {gpu_id}"
                )));
            }
            let base_profile = overrides
                .get(&gpu_id)
                .cloned()
                .unwrap_or_else(|| gpu.profile());
            let Some(profile) = gpu_profile_override_from_base_config(
                &format!("{name}.gpu_profile_overrides[{idx}]"),
                base_profile,
                &config,
            )?
            else {
                unreachable!("profile override config was checked above");
            };
            overrides.insert(gpu_id, profile);
        }
    }

    Ok(())
}

fn parse_gpu_labels(
    name: &str,
    gpu_tag: Option<&str>,
    gpu_tags: Option<&[String]>,
) -> Result<BTreeSet<String>, ConfigError> {
    let mut labels = BTreeSet::new();
    if let Some(tag) = gpu_tag {
        let label = normalize(tag);
        if label.is_empty() {
            return Err(ConfigError::new(format!(
                "{name} must not contain empty labels"
            )));
        }
        labels.insert(label);
    }
    if let Some(tags) = gpu_tags {
        if tags.is_empty() {
            return Err(ConfigError::new(format!("{name} must not be empty")));
        }
        for tag in tags {
            let label = normalize(tag);
            if label.is_empty() {
                return Err(ConfigError::new(format!(
                    "{name} must not contain empty labels"
                )));
            }
            labels.insert(label);
        }
    }
    Ok(labels)
}

fn parse_node_topology_metadata(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    island: Option<&str>,
    failure_domain: Option<&str>,
) -> Result<NodeTopologyMetadata, ConfigError> {
    Ok(NodeTopologyMetadata {
        labels: parse_normalized_label_set(&format!("{name}.node_tags"), node_tag, node_tags)?,
        rack: parse_optional_topology_domain(&format!("{name}.rack"), rack)?,
        island: parse_optional_topology_domain(&format!("{name}.island"), island)?,
        failure_domain: parse_optional_topology_domain(
            &format!("{name}.failure_domain"),
            failure_domain,
        )?,
    })
}

fn parse_normalized_label_set(
    name: &str,
    label: Option<&str>,
    labels: Option<&[String]>,
) -> Result<BTreeSet<String>, ConfigError> {
    let mut parsed = BTreeSet::new();
    for label in parse_optional_normalized_labels(name, label, labels)? {
        parsed.insert(label);
    }
    Ok(parsed)
}

fn parse_optional_normalized_labels(
    name: &str,
    label: Option<&str>,
    labels: Option<&[String]>,
) -> Result<Vec<String>, ConfigError> {
    if label.is_some() && labels.is_some() {
        return Err(ConfigError::new(format!(
            "{name} cannot set both singular and plural labels"
        )));
    }
    let mut parsed = Vec::new();
    if let Some(label) = label {
        parsed.push(parse_topology_label(name, label)?);
    }
    if let Some(labels) = labels {
        if labels.is_empty() {
            return Err(ConfigError::new(format!("{name} must not be empty")));
        }
        for label in labels {
            let parsed_label = parse_topology_label(name, label)?;
            if !parsed.contains(&parsed_label) {
                parsed.push(parsed_label);
            }
        }
    }
    Ok(parsed)
}

fn parse_optional_topology_domains(
    name: &str,
    label: Option<&str>,
    labels: Option<&[String]>,
) -> Result<Vec<String>, ConfigError> {
    parse_optional_normalized_labels(name, label, labels)
}

fn parse_optional_topology_domain(
    name: &str,
    value: Option<&str>,
) -> Result<Option<String>, ConfigError> {
    value
        .map(|value| parse_topology_label(name, value))
        .transpose()
}

fn parse_topology_label(name: &str, value: &str) -> Result<String, ConfigError> {
    let label = normalize(value);
    if label.is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }
    Ok(label)
}

fn parse_disabled_gpus(
    name: &str,
    values: Option<Vec<u32>>,
    gpus: &HashMap<u32, Gpu>,
    gpu_operational_states: &mut HashMap<u32, OperationalState>,
) -> Result<BTreeSet<u32>, ConfigError> {
    let mut disabled = BTreeSet::new();
    for gpu_id in values.unwrap_or_default() {
        if !gpus.contains_key(&gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_gpus references unknown local GPU id {gpu_id}"
            )));
        }
        if !disabled.insert(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_gpus duplicates local GPU id {gpu_id}"
            )));
        }
        gpu_operational_states.insert(gpu_id, OperationalState::Disabled);
    }

    disabled.extend(
        gpu_operational_states
            .iter()
            .filter_map(|(gpu_id, state)| (!state.accepts_work()).then_some(*gpu_id)),
    );

    Ok(disabled)
}

fn merge_gpu_operational_states(
    name: &str,
    states: &mut HashMap<u32, OperationalState>,
    entries: Option<&[GpuStateSection]>,
    gpus: &HashMap<u32, Gpu>,
) -> Result<(), ConfigError> {
    let Some(entries) = entries else {
        return Ok(());
    };
    for (idx, entry) in entries.iter().enumerate() {
        if entry.local_gpu_id.is_some() && entry.local_gpu_ids.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_states[{idx}] cannot set both gpu and gpus"
            )));
        }
        let gpu_ids = match (entry.local_gpu_id, entry.local_gpu_ids.as_ref()) {
            (Some(gpu_id), None) => vec![gpu_id],
            (None, Some(gpu_ids)) if !gpu_ids.is_empty() => gpu_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states[{idx}].gpus must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states[{idx}] must set gpu or gpus"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        let state = parse_resource_operational_state(
            &format!("{name}.gpu_states[{idx}].state"),
            &entry.state,
        )?;
        for gpu_id in gpu_ids {
            if !gpus.contains_key(&gpu_id) {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states[{idx}] references unknown local GPU id {gpu_id}"
                )));
            }
            if states.insert(gpu_id, state).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_states duplicates local GPU id {gpu_id}"
                )));
            }
        }
    }
    Ok(())
}

fn parse_resource_operational_state(
    name: &str,
    value: &str,
) -> Result<OperationalState, ConfigError> {
    match normalize(value).as_str() {
        "healthy" | "enabled" | "available" | "active" => Ok(OperationalState::Healthy),
        "disabled" | "offline" | "unavailable" | "failed" => Ok(OperationalState::Disabled),
        "maintenance" => Ok(OperationalState::Maintenance),
        "draining" | "drain" => Ok(OperationalState::Draining),
        "reserved" => Ok(OperationalState::Reserved),
        parsed => Err(ConfigError::new(format!(
            "unsupported {name} '{parsed}'; use healthy, disabled, maintenance, draining, or reserved"
        ))),
    }
}

fn next_local_gpu_id(gpus: &HashMap<u32, Gpu>) -> u32 {
    gpus.keys().copied().max().map_or(0, |gpu_id| gpu_id + 1)
}

fn representative_gpu_for_intra(
    name: &str,
    gpus: &HashMap<u32, Gpu>,
    intra: Option<&str>,
) -> Result<Gpu, ConfigError> {
    let Some(first) = gpus.values().next().copied() else {
        return Err(ConfigError::new(format!("{name}.gpus must not be empty")));
    };
    let mixed_gpu_types = gpus.values().any(|gpu| *gpu != first);
    if intra.is_none() && mixed_gpu_types {
        return Err(ConfigError::new(format!(
            "{name}.intra is required when {name}.gpus mixes GPU types"
        )));
    }

    Ok(first)
}

fn insert_node(
    nodes: &mut HashMap<u32, Node>,
    node_id: u32,
    node: Node,
) -> Result<(), ConfigError> {
    if nodes.insert(node_id, node).is_some() {
        return Err(ConfigError::new(format!("duplicate node id {node_id}")));
    }

    Ok(())
}

fn next_node_id(nodes: &HashMap<u32, Node>) -> u32 {
    nodes.keys().copied().max().map_or(0, |node_id| node_id + 1)
}

fn add_group_node(
    node_groups: &mut HashMap<String, Vec<u32>>,
    label: &str,
    node_id: u32,
) -> Result<(), ConfigError> {
    let key = normalize(label);
    if key.is_empty() {
        return Err(ConfigError::new("node group labels must not be empty"));
    }
    node_groups.entry(key).or_default().push(node_id);
    Ok(())
}

fn finalize_node_groups(node_groups: &mut HashMap<String, Vec<u32>>, nodes: &HashMap<u32, Node>) {
    let mut all_nodes: Vec<_> = nodes.keys().copied().collect();
    all_nodes.sort_unstable();
    node_groups
        .entry("all".to_string())
        .or_default()
        .extend(all_nodes);

    for group_nodes in node_groups.values_mut() {
        group_nodes.sort_unstable();
        group_nodes.dedup();
    }
}

fn custom_interconnect_topology(
    section: Option<InterconnectSection>,
    default_profile: Option<FabricProfile>,
    nodes: &HashMap<u32, Node>,
    node_groups: &HashMap<String, Vec<u32>>,
) -> Result<InterNodeTopology, ConfigError> {
    let Some(section) = section else {
        return Ok(InterNodeTopology::Custom(HashMap::new()));
    };

    if let Some(links) = section.links {
        let mut edges = HashMap::new();
        for link in links {
            let mut profile = interconnect_link_profile(&link)?;
            if let Some(oversubscription) = link.oversubscription {
                if oversubscription < 1.0 {
                    return Err(ConfigError::new(
                        "interconnect.links[].oversubscription must be greater than or equal to 1.0",
                    ));
                }
                profile.bw.unidirectional = profile.bw.unidirectional / oversubscription;
            }
            let rails = interconnect_link_rails(&link)?;
            for pair in interconnect_link_pairs(&link, nodes, node_groups)? {
                let links = edges
                    .entry(UnorderedPair::new(pair.from, pair.to))
                    .or_insert_with(Vec::new);
                for rail in &rails {
                    links.push(CustomInterNodeLink {
                        profile: profile.clone(),
                        rail: *rail,
                        endpoints: pair.endpoint_scope(),
                    });
                }
            }
        }
        return Ok(InterNodeTopology::Custom(edges));
    }

    Ok(match default_profile {
        Some(link) => InterNodeTopology::FatTree {
            link,
            oversubscription: section.oversubscription.unwrap_or(1.0).max(1.0),
            leaf_size: 0,
        },
        None => InterNodeTopology::Custom(HashMap::new()),
    })
}

fn interconnect_link_rails(
    link: &InterconnectLinkSection,
) -> Result<Vec<Option<u32>>, ConfigError> {
    if link.rail.is_some() && link.rails.is_some() {
        return Err(ConfigError::new(
            "interconnect.links[].rail and rails cannot both be set",
        ));
    }
    if let Some(rail) = link.rail {
        return Ok(vec![Some(rail)]);
    }
    if let Some(rails) = &link.rails {
        if rails.is_empty() {
            return Err(ConfigError::new(
                "interconnect.links[].rails must not be empty",
            ));
        }
        let mut values = rails.clone();
        values.sort_unstable();
        values.dedup();
        return Ok(values.into_iter().map(Some).collect());
    }

    Ok(vec![None])
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InterconnectLinkPair {
    from: u32,
    to: u32,
    from_gpus: Vec<u32>,
    to_gpus: Vec<u32>,
}

impl InterconnectLinkPair {
    fn endpoint_scope(&self) -> Option<CustomInterNodeLinkEndpoints> {
        if self.from_gpus.is_empty() && self.to_gpus.is_empty() {
            None
        } else {
            Some(CustomInterNodeLinkEndpoints {
                from_node: self.from,
                from_gpus: self.from_gpus.clone(),
                to_node: self.to,
                to_gpus: self.to_gpus.clone(),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InterconnectGpuEndpointSelector {
    explicit_gpus: Vec<u32>,
    gpu_types: Vec<Gpu>,
    gpu_labels: Vec<String>,
}

impl InterconnectGpuEndpointSelector {
    fn uses_filtered_selector(&self) -> bool {
        !self.gpu_types.is_empty() || !self.gpu_labels.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InterconnectNodeEndpointSelector {
    explicit_node: Option<u32>,
    group: Option<String>,
    node_labels: Vec<String>,
    racks: Vec<String>,
    islands: Vec<String>,
    failure_domains: Vec<String>,
}

impl InterconnectNodeEndpointSelector {
    fn uses_selector(&self) -> bool {
        self.explicit_node.is_some()
            || self.group.is_some()
            || !self.node_labels.is_empty()
            || !self.racks.is_empty()
            || !self.islands.is_empty()
            || !self.failure_domains.is_empty()
    }
}

fn interconnect_link_pairs(
    link: &InterconnectLinkSection,
    nodes: &HashMap<u32, Node>,
    node_groups: &HashMap<String, Vec<u32>>,
) -> Result<Vec<InterconnectLinkPair>, ConfigError> {
    let from_node_selector = interconnect_endpoint_node_selector(
        "from",
        link.from,
        link.from_group.as_deref(),
        link.from_node_tag.as_deref(),
        link.from_node_tags.as_deref(),
        link.from_rack.as_deref(),
        link.from_racks.as_deref(),
        link.from_island.as_deref(),
        link.from_islands.as_deref(),
        link.from_failure_domain.as_deref(),
        link.from_failure_domains.as_deref(),
    )?;
    let to_node_selector = interconnect_endpoint_node_selector(
        "to",
        link.to,
        link.to_group.as_deref(),
        link.to_node_tag.as_deref(),
        link.to_node_tags.as_deref(),
        link.to_rack.as_deref(),
        link.to_racks.as_deref(),
        link.to_island.as_deref(),
        link.to_islands.as_deref(),
        link.to_failure_domain.as_deref(),
        link.to_failure_domains.as_deref(),
    )?;
    let source_nodes =
        interconnect_endpoint_nodes("from", &from_node_selector, nodes, node_groups)?;
    let dest_nodes = interconnect_endpoint_nodes("to", &to_node_selector, nodes, node_groups)?;
    let from_gpu_selector = interconnect_endpoint_gpu_selector(
        "from",
        link.from_gpu,
        link.from_gpus.as_deref(),
        link.from_gpu_type.as_deref(),
        link.from_gpu_types.as_deref(),
        link.from_gpu_tag.as_deref(),
        link.from_gpu_tags.as_deref(),
    )?;
    let to_gpu_selector = interconnect_endpoint_gpu_selector(
        "to",
        link.to_gpu,
        link.to_gpus.as_deref(),
        link.to_gpu_type.as_deref(),
        link.to_gpu_types.as_deref(),
        link.to_gpu_tag.as_deref(),
        link.to_gpu_tags.as_deref(),
    )?;
    let mut pairs = Vec::new();

    for from in &source_nodes {
        for to in &dest_nodes {
            if from != to && !same_unordered_link_pair_exists(&pairs, *from, *to) {
                let from_gpus =
                    interconnect_endpoint_gpus("from", *from, &from_gpu_selector, nodes)?;
                let to_gpus = interconnect_endpoint_gpus("to", *to, &to_gpu_selector, nodes)?;
                if (from_gpu_selector.uses_filtered_selector() && from_gpus.is_empty())
                    || (to_gpu_selector.uses_filtered_selector() && to_gpus.is_empty())
                {
                    continue;
                }
                pairs.push(InterconnectLinkPair {
                    from: *from,
                    to: *to,
                    from_gpus,
                    to_gpus,
                });
            }
        }
    }

    if pairs.is_empty() {
        return Err(ConfigError::new(
            "interconnect.links[] must connect at least two distinct nodes",
        ));
    }

    Ok(pairs)
}

fn interconnect_endpoint_gpu_selector(
    name: &str,
    gpu_id: Option<u32>,
    gpu_ids: Option<&[u32]>,
    gpu_type: Option<&str>,
    gpu_types: Option<&[String]>,
    gpu_tag: Option<&str>,
    gpu_tags: Option<&[String]>,
) -> Result<InterconnectGpuEndpointSelector, ConfigError> {
    if gpu_id.is_some() && gpu_ids.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpu and {name}_gpus cannot both be set"
        )));
    }
    if gpu_type.is_some() && gpu_types.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpu_type and {name}_gpu_types cannot both be set"
        )));
    }
    if gpu_tag.is_some() && gpu_tags.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpu_tag and {name}_gpu_tags cannot both be set"
        )));
    }
    let has_explicit_gpus = gpu_id.is_some() || gpu_ids.is_some();
    let has_gpu_types = gpu_type.is_some() || gpu_types.is_some();
    let has_gpu_tags = gpu_tag.is_some() || gpu_tags.is_some();
    if has_explicit_gpus && (has_gpu_types || has_gpu_tags) {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name}_gpus cannot be combined with {name}_gpu_type or {name}_gpu_tag selectors"
        )));
    }
    let mut gpus = Vec::new();
    if let Some(gpu_id) = gpu_id {
        gpus.push(gpu_id);
    }
    if let Some(gpu_ids) = gpu_ids {
        if gpu_ids.is_empty() {
            return Err(ConfigError::new(format!(
                "interconnect.links[].{name}_gpus must not be empty"
            )));
        }
        gpus.extend_from_slice(gpu_ids);
    }
    sort_dedup_or_error(&mut gpus, &format!("interconnect.links[].{name}_gpus"))?;
    let mut selected_gpu_types = Vec::new();
    if let Some(gpu_type) = gpu_type {
        selected_gpu_types.push(gpu(gpu_type)?);
    }
    if let Some(gpu_types) = gpu_types {
        if gpu_types.is_empty() {
            return Err(ConfigError::new(format!(
                "interconnect.links[].{name}_gpu_types must not be empty"
            )));
        }
        for gpu_type in gpu_types {
            let parsed = gpu(gpu_type)?;
            if !selected_gpu_types.contains(&parsed) {
                selected_gpu_types.push(parsed);
            }
        }
    }
    let selected_gpu_labels = parse_gpu_labels(
        &format!("interconnect.links[].{name}_gpu_tags"),
        gpu_tag,
        gpu_tags,
    )?
    .into_iter()
    .collect();
    Ok(InterconnectGpuEndpointSelector {
        explicit_gpus: gpus,
        gpu_types: selected_gpu_types,
        gpu_labels: selected_gpu_labels,
    })
}

fn interconnect_endpoint_gpus(
    name: &str,
    node_id: u32,
    selector: &InterconnectGpuEndpointSelector,
    nodes: &HashMap<u32, Node>,
) -> Result<Vec<u32>, ConfigError> {
    let Some(node) = nodes.get(&node_id) else {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name} references unknown node id {node_id}"
        )));
    };
    if !selector.explicit_gpus.is_empty() {
        for gpu_id in &selector.explicit_gpus {
            if !node.gpus.contains_key(gpu_id) {
                return Err(ConfigError::new(format!(
                    "interconnect.links[].{name}_gpus references node {node_id} local GPU id {gpu_id}, but that GPU does not exist"
                )));
            }
        }
        return Ok(selector.explicit_gpus.clone());
    }
    if selector.gpu_types.is_empty() && selector.gpu_labels.is_empty() {
        return Ok(Vec::new());
    }
    let mut gpus = node
        .gpus
        .iter()
        .filter_map(|(gpu_id, gpu)| {
            let matches_type = selector.gpu_types.is_empty() || selector.gpu_types.contains(gpu);
            let matches_label = selector.gpu_labels.is_empty()
                || node.gpu_labels(*gpu_id).is_some_and(|labels| {
                    selector
                        .gpu_labels
                        .iter()
                        .any(|label| labels.contains(label))
                });
            (matches_type && matches_label).then_some(*gpu_id)
        })
        .collect::<Vec<_>>();
    gpus.sort_unstable();
    Ok(gpus)
}

#[allow(clippy::too_many_arguments)]
fn interconnect_endpoint_node_selector(
    name: &str,
    node_id: Option<u32>,
    group: Option<&str>,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
) -> Result<InterconnectNodeEndpointSelector, ConfigError> {
    if node_id.is_some() && group.is_some() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name} and {name}_group cannot both be set"
        )));
    }
    Ok(InterconnectNodeEndpointSelector {
        explicit_node: node_id,
        group: group.map(normalize),
        node_labels: parse_optional_normalized_labels(
            &format!("interconnect.links[].{name}_node_tags"),
            node_tag,
            node_tags,
        )?,
        racks: parse_optional_topology_domains(
            &format!("interconnect.links[].{name}_racks"),
            rack,
            racks,
        )?,
        islands: parse_optional_topology_domains(
            &format!("interconnect.links[].{name}_islands"),
            island,
            islands,
        )?,
        failure_domains: parse_optional_topology_domains(
            &format!("interconnect.links[].{name}_failure_domains"),
            failure_domain,
            failure_domains,
        )?,
    })
}

fn interconnect_endpoint_nodes(
    name: &str,
    selector: &InterconnectNodeEndpointSelector,
    nodes: &HashMap<u32, Node>,
    node_groups: &HashMap<String, Vec<u32>>,
) -> Result<Vec<u32>, ConfigError> {
    if !selector.uses_selector() {
        return Err(ConfigError::new(format!(
            "interconnect.links[] requires {name}, {name}_group, {name}_node_tag, {name}_rack, {name}_island, or {name}_failure_domain"
        )));
    }

    let mut selected = if let Some(node_id) = selector.explicit_node {
        if !nodes.contains_key(&node_id) {
            return Err(ConfigError::new(format!(
                "interconnect.links[].{name} references unknown node id {node_id}"
            )));
        }
        vec![node_id]
    } else if let Some(group) = &selector.group {
        node_groups.get(group).cloned().ok_or_else(|| {
            ConfigError::new(format!(
                "interconnect.links[].{name}_group references unknown group '{group}'"
            ))
        })?
    } else {
        let mut node_ids: Vec<_> = nodes.keys().copied().collect();
        node_ids.sort_unstable();
        node_ids
    };

    selected.retain(|node_id| {
        nodes
            .get(node_id)
            .is_some_and(|node| node_matches_topology_selector(node, selector))
    });
    selected.sort_unstable();
    selected.dedup();
    if selected.is_empty() {
        return Err(ConfigError::new(format!(
            "interconnect.links[].{name} selector matched no nodes"
        )));
    }
    Ok(selected)
}

fn node_matches_topology_selector(
    node: &Node,
    selector: &InterconnectNodeEndpointSelector,
) -> bool {
    (selector.node_labels.is_empty()
        || selector
            .node_labels
            .iter()
            .any(|label| node.topology.labels.contains(label)))
        && (selector.racks.is_empty()
            || node
                .topology
                .rack
                .as_ref()
                .is_some_and(|rack| selector.racks.contains(rack)))
        && (selector.islands.is_empty()
            || node
                .topology
                .island
                .as_ref()
                .is_some_and(|island| selector.islands.contains(island)))
        && (selector.failure_domains.is_empty()
            || node
                .topology
                .failure_domain
                .as_ref()
                .is_some_and(|failure_domain| selector.failure_domains.contains(failure_domain)))
}

fn same_unordered_link_pair_exists(pairs: &[InterconnectLinkPair], from: u32, to: u32) -> bool {
    pairs
        .iter()
        .any(|pair| (pair.from == from && pair.to == to) || (pair.from == to && pair.to == from))
}

fn default_nic_reference_profile() -> FabricProfile {
    FabricProfile {
        kind: crate::types::common::FabricKind::InfiniBand,
        label: "default NIC profile",
        bw: crate::types::common::LinkBandwidth::full_duplex_unidirectional(
            Bandwidth::from_gigabits_per_sec(400.0),
        ),
        latency: crate::types::common::Latency::from_us(1.2),
        reduction_accel: crate::types::common::ReductionAccelerator::None,
    }
}

fn interconnect_profile(section: &InterconnectSection) -> Result<FabricProfile, ConfigError> {
    let raw_kind = section
        .kind
        .as_deref()
        .ok_or_else(|| ConfigError::new("interconnect.kind is required"))?;
    let raw_variant = section
        .variant
        .as_deref()
        .ok_or_else(|| ConfigError::new("interconnect.variant is required"))?;
    let kind = normalize(raw_kind);
    let variant = normalize(raw_variant);

    match kind.as_str() {
        "ib" | "infiniband" => match variant.as_str() {
            "edr" => Ok(IbVariant::Edr.default_profile()),
            "hdr" => Ok(IbVariant::Hdr.default_profile()),
            "ndr" => Ok(IbVariant::Ndr.default_profile()),
            "xdr" => Ok(IbVariant::Xdr.default_profile()),
            _ => Err(ConfigError::new(format!(
                "unsupported InfiniBand variant '{}'; use edr, hdr, ndr, or xdr",
                raw_variant
            ))),
        },
        "roce" | "rocev2" => match variant.as_str() {
            "25g" | "v2_25g" => Ok(RoceVariant::V2_25G.default_profile()),
            "50g" | "v2_50g" => Ok(RoceVariant::V2_50G.default_profile()),
            "100g" | "v2_100g" => Ok(RoceVariant::V2_100G.default_profile()),
            "200g" | "v2_200g" => Ok(RoceVariant::V2_200G.default_profile()),
            "400g" | "v2_400g" => Ok(RoceVariant::V2_400G.default_profile()),
            "800g" | "v2_800g" => Ok(RoceVariant::V2_800G.default_profile()),
            _ => Err(ConfigError::new(format!(
                "unsupported RoCE variant '{}'; use 25g, 50g, 100g, 200g, 400g, or 800g",
                raw_variant
            ))),
        },
        "eth" | "ethernet" => match variant.as_str() {
            "10g" => Ok(EthVariant::E10G.default_profile()),
            "25g" => Ok(EthVariant::E25G.default_profile()),
            "40g" => Ok(EthVariant::E40G.default_profile()),
            "100g" => Ok(EthVariant::E100G.default_profile()),
            "200g" => Ok(EthVariant::E200G.default_profile()),
            "400g" => Ok(EthVariant::E400G.default_profile()),
            "800g" => Ok(EthVariant::E800G.default_profile()),
            _ => Err(ConfigError::new(format!(
                "unsupported Ethernet variant '{}'; use 10g, 25g, 40g, 100g, 200g, 400g, or 800g",
                raw_variant
            ))),
        },
        _ => Err(ConfigError::new(format!(
            "unsupported interconnect kind '{}'; use ib, roce, or ethernet",
            raw_kind
        ))),
    }
}

fn interconnect_link_profile(
    section: &InterconnectLinkSection,
) -> Result<FabricProfile, ConfigError> {
    interconnect_profile(&InterconnectSection {
        kind: Some(section.kind.clone()),
        variant: Some(section.variant.clone()),
        oversubscription: None,
        links: None,
    })
}

fn gpu(value: &str) -> Result<Gpu, ConfigError> {
    match normalize(value).as_str() {
        "a100_40gb" | "a100_40" => Ok(Gpu::A100_40GB),
        "a100_80gb" | "a100_80" => Ok(Gpu::A100_80GB),
        "h100_sxm" | "h100" => Ok(Gpu::H100_SXM),
        "h200_sxm" | "h200" => Ok(Gpu::H200_SXM),
        "b200" => Ok(Gpu::B200),
        "mi300x" => Ok(Gpu::MI300X),
        _ => Err(ConfigError::new(format!(
            "unsupported node gpu '{value}'; use a100_40gb, a100_80gb, h100_sxm, h200_sxm, b200, or mi300x"
        ))),
    }
}

fn intra_node_topology(value: Option<&str>, gpu: Gpu) -> Result<IntraNodeTopology, ConfigError> {
    let value = value.map(normalize).unwrap_or_else(|| match gpu {
        Gpu::A100_40GB | Gpu::A100_80GB => "nvlink_v3".to_string(),
        Gpu::H100_SXM | Gpu::H200_SXM => "nvlink_v4".to_string(),
        Gpu::B200 => "nvlink_v5".to_string(),
        Gpu::MI300X => "pcie_gen5".to_string(),
    });

    match value.as_str() {
        "nvlink_v1" | "nvlink1" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V1.default_profile(),
        )),
        "nvlink_v2" | "nvlink2" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V2.default_profile(),
        )),
        "nvlink_v3" | "nvlink3" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V3.default_profile(),
        )),
        "nvlink_v4" | "nvlink4" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V4.default_profile(),
        )),
        "nvlink_v5" | "nvlink5" => Ok(IntraNodeTopology::NvSwitch(
            NvLinkVariant::V5.default_profile(),
        )),
        "pcie_gen3" | "gen3x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen3x16.default_profile(),
        )),
        "pcie_gen4" | "gen4x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen4x16.default_profile(),
        )),
        "pcie_gen5" | "gen5x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen5x16.default_profile(),
        )),
        "pcie_gen6" | "gen6x16" => Ok(IntraNodeTopology::Pcie(
            PcieVariant::Gen6x16.default_profile(),
        )),
        _ => Err(ConfigError::new(format!(
            "unsupported node intra fabric '{value}'; use nvlink_v3, nvlink_v4, nvlink_v5, pcie_gen4, or pcie_gen5"
        ))),
    }
}

fn nics_profile(
    name: &str,
    section: &NicsSection,
    interconnect: FabricProfile,
) -> Result<NodeNetworkProfile, ConfigError> {
    let affinity = match section.affinity.as_deref().map(normalize) {
        None => GpuNicAffinity::Dedicated,
        Some(value) if value == "dedicated" => GpuNicAffinity::Dedicated,
        Some(value) if value == "uniform" => GpuNicAffinity::Uniform,
        Some(value) if value == "shared" => GpuNicAffinity::Shared {
            gpus_per_nic: {
                let gpus_per_nic = section.gpus_per_nic.ok_or_else(|| {
                    ConfigError::new(format!(
                        "{name}.gpus_per_nic is required for shared affinity"
                    ))
                })?;
                if gpus_per_nic == 0 {
                    return Err(ConfigError::new(format!(
                        "{name}.gpus_per_nic must be nonzero"
                    )));
                }
                gpus_per_nic
            },
        },
        Some(value) => {
            return Err(ConfigError::new(format!(
                "unsupported {name}.affinity '{value}'; use dedicated, shared, or uniform"
            )));
        }
    };

    let nic_count = section.count.unwrap_or(8);
    if nic_count == 0 {
        return Err(ConfigError::new(format!("{name}.count must be nonzero")));
    }
    let rail_count = match section.rail_count {
        Some(0) => {
            return Err(ConfigError::new(format!(
                "{name}.rail_count must be nonzero"
            )));
        }
        Some(rail_count) => rail_count,
        None => nic_count,
    };
    if rail_count > nic_count {
        return Err(ConfigError::new(format!(
            "{name}.rail_count must be less than or equal to {name}.count"
        )));
    }

    let nic_bandwidth = match section.bandwidth_gbps {
        Some(gbps) if gbps > 0.0 => Bandwidth::from_gigabits_per_sec(gbps),
        Some(_) => {
            return Err(ConfigError::new(format!(
                "{name}.bandwidth_gbps must be positive"
            )));
        }
        None => interconnect.bw.unidirectional,
    };
    let gpu_nic_map = parse_gpu_nic_map(name, section.gpu_nic_map.as_deref())?;
    let gpu_numa_map = parse_gpu_numa_map(name, section.gpu_numa_map.as_deref())?;
    let nic_numa_map = parse_nic_numa_map(name, section.nic_numa_map.as_deref(), nic_count)?;
    validate_positive_optional_f64(
        &format!("{name}.cross_numa_bandwidth_scale"),
        section.cross_numa_bandwidth_scale,
    )?;
    validate_positive_optional_f64(
        &format!("{name}.cross_numa_latency_scale"),
        section.cross_numa_latency_scale,
    )?;
    let cross_numa_bandwidth_scale = section.cross_numa_bandwidth_scale.unwrap_or(1.0);
    let cross_numa_latency_scale = section.cross_numa_latency_scale.unwrap_or(1.0);
    let nic_rail_map =
        parse_nic_rail_map(name, section.nic_rail_map.as_deref(), nic_count, rail_count)?;
    let gpu_nic_path_overrides =
        parse_gpu_nic_paths(name, section.gpu_nic_paths.as_deref(), nic_count)?;
    let nic_bandwidth_overrides =
        parse_nic_bandwidth_overrides(name, section.nic_bandwidth_overrides.as_deref(), nic_count)?;
    let nic_latency_scale_overrides = parse_nic_latency_scale_overrides(
        name,
        section.nic_latency_scale_overrides.as_deref(),
        nic_count,
    )?;
    let mut nic_operational_states =
        parse_nic_operational_states(name, section.nic_states.as_deref(), nic_count)?;
    let disabled_nics = parse_disabled_nics(
        name,
        section.disabled_nics.as_deref(),
        nic_count,
        &mut nic_operational_states,
    )?;

    Ok(NodeNetworkProfile {
        nic_count,
        nic_bandwidth,
        nic_bandwidth_overrides,
        nic_latency_scale_overrides,
        gpu_to_nic: affinity,
        rail_count,
        nic_rail_map,
        gpu_nic_map,
        gpu_numa_map,
        nic_numa_map,
        cross_numa_bandwidth_scale,
        cross_numa_latency_scale,
        gpu_nic_path_overrides,
        disabled_nics,
        nic_operational_states,
    })
}

fn parse_nic_rail_map(
    name: &str,
    entries: Option<&[NicRailMapSection]>,
    nic_count: u8,
    rail_count: u8,
) -> Result<BTreeMap<NicId, u32>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        let nic_id = entry.nic.ok_or_else(|| {
            ConfigError::new(format!("{name}.nic_rail_map[{idx}].nic is required"))
        })?;
        let rail_id = entry.rail.ok_or_else(|| {
            ConfigError::new(format!("{name}.nic_rail_map[{idx}].rail is required"))
        })?;
        if nic_id >= u32::from(nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.nic_rail_map[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
            )));
        }
        if rail_id >= u32::from(rail_count) {
            return Err(ConfigError::new(format!(
                "{name}.nic_rail_map[{idx}] references rail {rail_id}, but {name}.rail_count is {rail_count}"
            )));
        }
        if map.insert(nic_id, rail_id).is_some() {
            return Err(ConfigError::new(format!(
                "{name}.nic_rail_map duplicates NIC {nic_id}"
            )));
        }
    }

    Ok(map)
}

fn parse_gpu_nic_paths(
    name: &str,
    entries: Option<&[GpuNicPathSection]>,
    nic_count: u8,
) -> Result<BTreeMap<(GpuId, NicId), GpuNicPathOverride>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        let gpu_id = entry.local_gpu_id.ok_or_else(|| {
            ConfigError::new(format!(
                "{name}.gpu_nic_paths[{idx}].local_gpu_id is required"
            ))
        })?;
        let nic_id = entry.nic.ok_or_else(|| {
            ConfigError::new(format!("{name}.gpu_nic_paths[{idx}].nic is required"))
        })?;
        if nic_id >= u32::from(nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
            )));
        }
        let bandwidth = match entry.bandwidth_gbps {
            Some(gbps) if gbps.is_finite() && gbps > 0.0 => {
                Some(Bandwidth::from_gigabits_per_sec(gbps))
            }
            Some(_) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_paths[{idx}].bandwidth_gbps must be finite and positive"
                )));
            }
            None => None,
        };
        let latency = match entry.latency_us {
            Some(us) if us.is_finite() && us >= 0.0 => Some(Latency::from_us(us)),
            Some(_) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_paths[{idx}].latency_us must be finite and nonnegative"
                )));
            }
            None => None,
        };
        let override_profile = GpuNicPathOverride {
            label: entry.label.clone(),
            bandwidth,
            latency,
            gpudirect: entry.gpudirect,
            available: entry.available.unwrap_or(true),
        };
        if map.insert((gpu_id, nic_id), override_profile).is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths duplicates local GPU id {gpu_id} and NIC {nic_id}"
            )));
        }
    }

    Ok(map)
}

fn parse_disabled_nics(
    name: &str,
    values: Option<&[u32]>,
    nic_count: u8,
    nic_operational_states: &mut BTreeMap<NicId, OperationalState>,
) -> Result<BTreeSet<NicId>, ConfigError> {
    let mut disabled = BTreeSet::new();
    for nic_id in values.unwrap_or_default() {
        if *nic_id >= u32::from(nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_nics references NIC {nic_id}, but {name}.count is {nic_count}"
            )));
        }
        if !disabled.insert(*nic_id) {
            return Err(ConfigError::new(format!(
                "{name}.disabled_nics duplicates NIC {nic_id}"
            )));
        }
        nic_operational_states.insert(*nic_id, OperationalState::Disabled);
    }

    disabled.extend(
        nic_operational_states
            .iter()
            .filter_map(|(nic_id, state)| (!state.accepts_work()).then_some(*nic_id)),
    );

    Ok(disabled)
}

fn parse_nic_operational_states(
    name: &str,
    entries: Option<&[NicStateSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, OperationalState>, ConfigError> {
    let mut states = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(states);
    };
    for (idx, entry) in entries.iter().enumerate() {
        if entry.nic.is_some() && entry.nics.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.nic_states[{idx}] cannot set both nic and nics"
            )));
        }
        let nic_ids = match (entry.nic, entry.nics.as_ref()) {
            (Some(nic_id), None) => vec![nic_id],
            (None, Some(nic_ids)) if !nic_ids.is_empty() => nic_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states[{idx}].nics must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states[{idx}] must set nic or nics"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        let state = parse_resource_operational_state(
            &format!("{name}.nic_states[{idx}].state"),
            &entry.state,
        )?;
        for nic_id in nic_ids {
            if nic_id >= u32::from(nic_count) {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
                )));
            }
            if states.insert(nic_id, state).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.nic_states duplicates NIC {nic_id}"
                )));
            }
        }
    }
    Ok(states)
}

fn parse_nic_bandwidth_overrides(
    name: &str,
    entries: Option<&[NicBandwidthOverrideSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, Bandwidth>, ConfigError> {
    let mut overrides = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(overrides);
    };
    for (idx, entry) in entries.iter().enumerate() {
        let nic_ids = parse_nic_override_ids(
            name,
            "nic_bandwidth_overrides",
            idx,
            entry.nic,
            entry.nics.as_deref(),
        )?;
        if !entry.bandwidth_gbps.is_finite() || entry.bandwidth_gbps <= 0.0 {
            return Err(ConfigError::new(format!(
                "{name}.nic_bandwidth_overrides[{idx}].bandwidth_gbps must be finite and positive"
            )));
        }
        for nic_id in nic_ids {
            validate_nic_override_id(name, "nic_bandwidth_overrides", idx, nic_id, nic_count)?;
            if overrides
                .insert(
                    nic_id,
                    Bandwidth::from_gigabits_per_sec(entry.bandwidth_gbps),
                )
                .is_some()
            {
                return Err(ConfigError::new(format!(
                    "{name}.nic_bandwidth_overrides duplicates NIC {nic_id}"
                )));
            }
        }
    }
    Ok(overrides)
}

fn parse_nic_latency_scale_overrides(
    name: &str,
    entries: Option<&[NicLatencyScaleOverrideSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, f64>, ConfigError> {
    let mut overrides = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(overrides);
    };
    for (idx, entry) in entries.iter().enumerate() {
        let nic_ids = parse_nic_override_ids(
            name,
            "nic_latency_scale_overrides",
            idx,
            entry.nic,
            entry.nics.as_deref(),
        )?;
        if !entry.latency_scale.is_finite() || entry.latency_scale <= 0.0 {
            return Err(ConfigError::new(format!(
                "{name}.nic_latency_scale_overrides[{idx}].latency_scale must be finite and positive"
            )));
        }
        for nic_id in nic_ids {
            validate_nic_override_id(name, "nic_latency_scale_overrides", idx, nic_id, nic_count)?;
            if overrides.insert(nic_id, entry.latency_scale).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.nic_latency_scale_overrides duplicates NIC {nic_id}"
                )));
            }
        }
    }
    Ok(overrides)
}

fn parse_nic_override_ids(
    name: &str,
    field: &str,
    idx: usize,
    nic: Option<u32>,
    nics: Option<&[u32]>,
) -> Result<Vec<NicId>, ConfigError> {
    if nic.is_some() && nics.is_some() {
        return Err(ConfigError::new(format!(
            "{name}.{field}[{idx}] cannot set both nic and nics"
        )));
    }
    match (nic, nics) {
        (Some(nic_id), None) => Ok(vec![nic_id]),
        (None, Some(nic_ids)) if !nic_ids.is_empty() => Ok(nic_ids.to_vec()),
        (None, Some(_)) => Err(ConfigError::new(format!(
            "{name}.{field}[{idx}].nics must not be empty"
        ))),
        (None, None) => Err(ConfigError::new(format!(
            "{name}.{field}[{idx}] must set nic or nics"
        ))),
        (Some(_), Some(_)) => unreachable!("checked above"),
    }
}

fn validate_nic_override_id(
    name: &str,
    field: &str,
    idx: usize,
    nic_id: NicId,
    nic_count: u8,
) -> Result<(), ConfigError> {
    if nic_id >= u32::from(nic_count) {
        return Err(ConfigError::new(format!(
            "{name}.{field}[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
        )));
    }
    Ok(())
}

fn parse_gpu_nic_map(
    name: &str,
    entries: Option<&[GpuNicMapSection]>,
) -> Result<BTreeMap<GpuId, Vec<NicId>>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        let gpu_id = entry.local_gpu_id.ok_or_else(|| {
            ConfigError::new(format!(
                "{name}.gpu_nic_map[{idx}].local_gpu_id is required"
            ))
        })?;
        if entry.nic.is_some() && entry.nics.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map[{idx}] cannot set both nic and nics"
            )));
        }
        let mut nics = match (entry.nic, entry.nics.as_ref()) {
            (Some(nic), None) => vec![nic],
            (None, Some(nics)) => nics.clone(),
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_map[{idx}] must set nic or nics"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        if nics.is_empty() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map[{idx}].nics must not be empty"
            )));
        }
        nics.sort_unstable();
        nics.dedup();
        if map.insert(gpu_id, nics).is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map duplicates local GPU id {gpu_id}"
            )));
        }
    }

    Ok(map)
}

fn parse_gpu_numa_map(
    name: &str,
    entries: Option<&[GpuNumaMapSection]>,
) -> Result<BTreeMap<GpuId, u32>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        if entry.local_gpu_id.is_some() && entry.local_gpu_ids.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.gpu_numa_map[{idx}] cannot set both gpu and gpus"
            )));
        }
        let gpu_ids = match (entry.local_gpu_id, entry.local_gpu_ids.as_ref()) {
            (Some(gpu_id), None) => vec![gpu_id],
            (None, Some(gpu_ids)) if !gpu_ids.is_empty() => gpu_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_numa_map[{idx}].gpus must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_numa_map[{idx}] must set gpu or gpus"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        for gpu_id in gpu_ids {
            if map.insert(gpu_id, entry.numa_domain).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_numa_map duplicates local GPU id {gpu_id}"
                )));
            }
        }
    }

    Ok(map)
}

fn parse_nic_numa_map(
    name: &str,
    entries: Option<&[NicNumaMapSection]>,
    nic_count: u8,
) -> Result<BTreeMap<NicId, u32>, ConfigError> {
    let mut map = BTreeMap::new();
    let Some(entries) = entries else {
        return Ok(map);
    };

    for (idx, entry) in entries.iter().enumerate() {
        if entry.nic.is_some() && entry.nics.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.nic_numa_map[{idx}] cannot set both nic and nics"
            )));
        }
        let nic_ids = match (entry.nic, entry.nics.as_ref()) {
            (Some(nic_id), None) => vec![nic_id],
            (None, Some(nic_ids)) if !nic_ids.is_empty() => nic_ids.clone(),
            (None, Some(_)) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map[{idx}].nics must not be empty"
                )));
            }
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map[{idx}] must set nic or nics"
                )));
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        };
        for nic_id in nic_ids {
            if nic_id >= u32::from(nic_count) {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map[{idx}] references NIC {nic_id}, but {name}.count is {nic_count}"
                )));
            }
            if map.insert(nic_id, entry.numa_domain).is_some() {
                return Err(ConfigError::new(format!(
                    "{name}.nic_numa_map duplicates NIC {nic_id}"
                )));
            }
        }
    }

    Ok(map)
}

fn validate_network_profile(
    name: &str,
    gpus: &HashMap<u32, Gpu>,
    disabled_gpus: &BTreeSet<u32>,
    profile: &NodeNetworkProfile,
) -> Result<(), ConfigError> {
    let gpu_count = gpus.len() as u32;
    match profile.gpu_to_nic {
        GpuNicAffinity::Dedicated if u32::from(profile.nic_count) < gpu_count => {
            Err(ConfigError::new(format!(
                "{name} dedicated affinity requires nics.count >= gpu_count ({gpu_count}), got {}",
                profile.nic_count
            )))
        }
        GpuNicAffinity::Shared { gpus_per_nic } => {
            let required_nics = gpu_count.div_ceil(u32::from(gpus_per_nic));
            if required_nics > u32::from(profile.nic_count) {
                return Err(ConfigError::new(format!(
                    "{name} shared affinity with gpus_per_nic={} requires at least {required_nics} NICs for {gpu_count} GPUs, got {}",
                    gpus_per_nic, profile.nic_count
                )));
            }
            Ok(())
        }
        _ => Ok(()),
    }?;

    for (gpu_id, nic_ids) in &profile.gpu_nic_map {
        if !gpus.contains_key(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_map references unknown local GPU id {gpu_id}"
            )));
        }
        for nic_id in nic_ids {
            if *nic_id >= u32::from(profile.nic_count) {
                return Err(ConfigError::new(format!(
                    "{name}.gpu_nic_map for local GPU id {gpu_id} references NIC {nic_id}, but {name}.count is {}",
                    profile.nic_count
                )));
            }
        }
    }

    for (gpu_id, nic_id) in profile.gpu_nic_path_overrides.keys() {
        if !gpus.contains_key(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths references unknown local GPU id {gpu_id}"
            )));
        }
        if *nic_id >= u32::from(profile.nic_count) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_nic_paths for local GPU id {gpu_id} references NIC {nic_id}, but {name}.count is {}",
                profile.nic_count
            )));
        }
    }

    for gpu_id in profile.gpu_numa_map.keys() {
        if !gpus.contains_key(gpu_id) {
            return Err(ConfigError::new(format!(
                "{name}.gpu_numa_map references unknown local GPU id {gpu_id}"
            )));
        }
    }

    for gpu_id in gpus.keys() {
        if disabled_gpus.contains(gpu_id) {
            continue;
        }
        if profile.nic_candidates_for_gpu(*gpu_id).is_empty() {
            return Err(ConfigError::new(format!(
                "{name} leaves local GPU id {gpu_id} without an enabled NIC path"
            )));
        }
    }

    Ok(())
}

fn require_nonempty<T>(name: &str, values: Vec<T>) -> Result<Vec<T>, ConfigError> {
    if values.is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }

    Ok(values)
}

fn parse_search(name: &str, search: SearchSection) -> Result<SearchSpace, ConfigError> {
    Ok(SearchSpace {
        tensor_ranks: require_nonempty(&format!("{name}.tensor_ranks"), search.tensor_ranks)?,
        pipeline_ranks: require_nonempty(&format!("{name}.pipeline_ranks"), search.pipeline_ranks)?,
        expert_ranks: require_nonempty(&format!("{name}.expert_ranks"), search.expert_ranks)?,
        data_ranks: require_nonempty(&format!("{name}.data_ranks"), search.data_ranks)?,
    })
}

fn parse_optional_placement(
    name: &str,
    section: Option<&PlacementSection>,
) -> Result<Option<RankPlacement>, ConfigError> {
    section
        .map(|section| parse_rank_placement(name, section))
        .transpose()
}

fn parse_rank_placement(
    name: &str,
    section: &PlacementSection,
) -> Result<RankPlacement, ConfigError> {
    if section.ranks.is_empty() {
        return Err(ConfigError::new(format!("{name}.ranks must not be empty")));
    }

    let max_rank = section
        .ranks
        .iter()
        .enumerate()
        .map(|(idx, rank)| {
            rank.rank
                .ok_or_else(|| ConfigError::new(format!("{name}.ranks[{idx}].rank is required")))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let mut rank_to_gpu = vec![None; max_rank as usize + 1];

    for (idx, rank) in section.ranks.iter().enumerate() {
        let rank_id = rank
            .rank
            .ok_or_else(|| ConfigError::new(format!("{name}.ranks[{idx}].rank is required")))?;
        let node_id = rank
            .node
            .ok_or_else(|| ConfigError::new(format!("{name}.ranks[{idx}].node is required")))?;
        let local_gpu_id = rank.local_gpu_id.ok_or_else(|| {
            ConfigError::new(format!(
                "{name}.ranks[{idx}].gpu or local_gpu_id is required"
            ))
        })?;
        let slot = rank_to_gpu
            .get_mut(rank_id as usize)
            .expect("rank_id must be within max rank");
        if slot.is_some() {
            return Err(ConfigError::new(format!(
                "{name}.ranks duplicates rank {rank_id}"
            )));
        }
        *slot = Some(GpuAddr {
            node_id,
            local_gpu_id,
        });
    }

    let mut placement = Vec::with_capacity(rank_to_gpu.len());
    for (rank_id, gpu) in rank_to_gpu.into_iter().enumerate() {
        let Some(gpu) = gpu else {
            return Err(ConfigError::new(format!(
                "{name}.ranks must define contiguous ranks; missing rank {rank_id}"
            )));
        };
        placement.push(gpu);
    }

    Ok(RankPlacement {
        rank_to_gpu: placement,
    })
}

fn parse_serving(
    serving: ServingSection,
    default_search: &SearchSpace,
    base_dir: Option<&Path>,
) -> Result<DisaggregatedServingConfig, ConfigError> {
    let deployment_mode = parse_serving_deployment_mode(serving.mode.as_deref())?;
    let slo_miss_penalty_weights = parse_serving_slo_miss_penalty_weights(&serving)?;
    let objective = parse_serving_objective(serving.objective.as_deref())?;
    let metric_ceilings = parse_serving_metric_ceilings(&serving)?;
    let kv_route_constraints = parse_serving_kv_route_constraints(&serving)?;
    let cost_model = parse_serving_cost_model(serving.cost.as_ref())?;
    let pool_candidates = parse_pool_candidates(&serving)?;
    let pool_search = parse_pool_search(serving.pool_search, deployment_mode)?;
    let traffic_classes = parse_traffic_classes(serving.traffic_classes)?;
    let mut slo_policies = parse_slo_policies(serving.slo_policies)?;
    slo_policies.extend(traffic_class_slo_policies(&traffic_classes));
    if pool_candidates.is_empty() && pool_search.is_none() {
        return Err(ConfigError::new(
            "serving requires prefill/decode nodes, [[serving.pool_candidates]], or [serving.pool_search]",
        ));
    }

    let prefill = match serving.prefill_search {
        Some(search) => parse_search("serving.prefill_search", search)?,
        None => default_search.clone(),
    };
    let decode = match serving.decode_search {
        Some(search) => parse_search("serving.decode_search", search)?,
        None => default_search.clone(),
    };
    let prefill_nodes = pool_candidates
        .first()
        .map(|candidate| candidate.prefill_nodes.clone())
        .unwrap_or_default();
    let decode_nodes = pool_candidates
        .first()
        .map(|candidate| candidate.decode_nodes.clone())
        .unwrap_or_default();
    validate_positive_optional_u32("serving.max_unique_gpus", serving.max_unique_gpus)?;
    let min_throughput_tokens_per_s = parse_non_negative_f64(
        "serving.min_throughput_tokens_per_s",
        serving.min_throughput_tokens_per_s,
    )?;
    Ok(DisaggregatedServingConfig {
        deployment_mode,
        prefill_nodes,
        decode_nodes,
        pool_candidates,
        pool_search,
        objective,
        slo_miss_penalty_weight: slo_miss_penalty_weights.aggregate,
        slo_miss_penalty_weights,
        topology_risk_penalty_weight: parse_non_negative_f64(
            "serving.topology_risk_penalty_weight",
            serving.topology_risk_penalty_weight,
        )?
        .unwrap_or(0.0),
        max_memory_pressure_fraction: parse_fraction(
            "serving.max_memory_pressure_fraction",
            serving.max_memory_pressure_fraction,
        )?,
        max_unique_gpus: serving.max_unique_gpus,
        min_throughput_tokens_per_s,
        cost_model,
        search: ServingSearchSpace { prefill, decode },
        traffic: parse_serving_traffic(
            serving.traffic,
            base_dir,
            traffic_classes,
            ServingServiceSections {
                services: serving.services,
                prefill: serving.prefill_service,
                decode: serving.decode_service,
                kv_transfer: serving.kv_transfer_service,
            },
            metric_ceilings,
            kv_route_constraints,
        )?,
        slo_policies,
    })
}

fn parse_serving_runtime_features(file: &WorkloadFile) -> Result<Vec<String>, ConfigError> {
    let mut features = Vec::new();
    if let Some(values) = file.serving_runtime_features.clone() {
        for feature in normalized_group_labels("serving_runtime_features", values)? {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    if let Some(values) = file
        .serving
        .as_ref()
        .and_then(|serving| serving.serving_runtime_features.clone())
    {
        for feature in normalized_group_labels("serving.runtime_features", values)? {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    Ok(features)
}

fn parse_serving_metric_ceilings(
    serving: &ServingSection,
) -> Result<ServingMetricCeilings, ConfigError> {
    Ok(ServingMetricCeilings {
        max_ttft_s: parse_positive_optional_seconds(
            "serving.max_ttft_s",
            serving.max_ttft_s,
            "serving.max_ttft_ms",
            serving.max_ttft_ms,
        )?,
        max_tpot_s: parse_positive_optional_seconds(
            "serving.max_tpot_s",
            serving.max_tpot_s,
            "serving.max_tpot_ms",
            serving.max_tpot_ms,
        )?,
        max_itl_s: parse_positive_optional_seconds(
            "serving.max_itl_s",
            serving.max_itl_s,
            "serving.max_itl_ms",
            serving.max_itl_ms,
        )?,
        max_e2el_s: parse_positive_optional_seconds(
            "serving.max_e2el_s",
            serving.max_e2el_s,
            "serving.max_e2el_ms",
            serving.max_e2el_ms,
        )?,
    })
}

fn parse_serving_kv_route_constraints(
    serving: &ServingSection,
) -> Result<ServingKvRouteConstraints, ConfigError> {
    validate_positive_optional_u32(
        "serving.min_kv_route_rail_count",
        serving.min_kv_route_rail_count,
    )?;
    Ok(ServingKvRouteConstraints {
        min_inter_node_rail_count: serving.min_kv_route_rail_count,
        require_inter_node_rail_metadata: serving.require_kv_route_rail_metadata.unwrap_or(false),
        require_gpudirect: serving.require_gpudirect_kv_paths.unwrap_or(false),
    })
}

fn parse_serving_cost_model(
    section: Option<&ServingCostSection>,
) -> Result<ServingCostModel, ConfigError> {
    let Some(section) = section else {
        return Ok(ServingCostModel::default());
    };
    let default_gpu_hour_usd = parse_non_negative_f64(
        "serving.cost.default_gpu_hour_usd",
        section.default_gpu_hour_usd,
    )?;
    let node_hour_usd =
        parse_non_negative_f64("serving.cost.node_hour_usd", section.node_hour_usd)?;
    let kwh_usd = parse_non_negative_f64("serving.cost.kwh_usd", section.kwh_usd)?;
    let default_gpu_watts =
        parse_non_negative_f64("serving.cost.default_gpu_watts", section.default_gpu_watts)?;
    let node_watts = parse_non_negative_f64("serving.cost.node_watts", section.node_watts)?;
    let mut gpu_rates = Vec::new();
    for (idx, rate) in section
        .gpu_rates
        .as_deref()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let gpu_label = optional_nonempty_string(rate.gpu_label.clone()).ok_or_else(|| {
            ConfigError::new(format!(
                "serving.cost.gpu_rates[{idx}].gpu_label is required"
            ))
        })?;
        gpu_rates.push(ServingGpuCostRate {
            gpu_label,
            gpu_hour_usd: parse_non_negative_f64(
                &format!("serving.cost.gpu_rates[{idx}].gpu_hour_usd"),
                rate.gpu_hour_usd,
            )?,
            watts: parse_non_negative_f64(
                &format!("serving.cost.gpu_rates[{idx}].watts"),
                rate.watts,
            )?,
        });
    }
    Ok(ServingCostModel {
        default_gpu_hour_usd,
        node_hour_usd,
        kwh_usd,
        default_gpu_watts,
        node_watts,
        gpu_rates,
    })
}

fn parse_serving_slo_miss_penalty_weights(
    serving: &ServingSection,
) -> Result<ServingSloMissPenaltyWeights, ConfigError> {
    Ok(ServingSloMissPenaltyWeights {
        aggregate: parse_non_negative_f64(
            "serving.slo_miss_penalty_weight",
            serving.slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        ttft: parse_non_negative_f64(
            "serving.ttft_slo_miss_penalty_weight",
            serving.ttft_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        tpot: parse_non_negative_f64(
            "serving.tpot_slo_miss_penalty_weight",
            serving.tpot_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        itl: parse_non_negative_f64(
            "serving.itl_slo_miss_penalty_weight",
            serving.itl_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        e2el: parse_non_negative_f64(
            "serving.e2el_slo_miss_penalty_weight",
            serving.e2el_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        deadline: parse_non_negative_f64(
            "serving.deadline_miss_penalty_weight",
            serving.deadline_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
    })
}

fn parse_serving_objective(objective: Option<&str>) -> Result<ServingObjective, ConfigError> {
    match objective.map(normalize).as_deref() {
        None
        | Some("e2el")
        | Some("end_to_end")
        | Some("end_to_end_latency")
        | Some("latency")
        | Some("minimize_e2el") => Ok(ServingObjective::MinimizeE2el),
        Some("ttft") | Some("minimize_ttft") => Ok(ServingObjective::MinimizeTtft),
        Some("tpot") | Some("minimize_tpot") => Ok(ServingObjective::MinimizeTpot),
        Some("throughput") | Some("max_throughput") | Some("maximize_throughput") => {
            Ok(ServingObjective::MaximizeThroughput)
        }
        Some("slo") | Some("slo_miss") | Some("slo_miss_rate") | Some("minimize_slo_miss_rate") => {
            Ok(ServingObjective::MinimizeSloMissRate)
        }
        Some("memory")
        | Some("hbm")
        | Some("memory_pressure")
        | Some("hbm_pressure")
        | Some("minimize_memory_pressure")
        | Some("minimize_hbm_pressure") => Ok(ServingObjective::MinimizeMemoryPressure),
        Some("cost")
        | Some("total_cost")
        | Some("cost_usd")
        | Some("minimize_cost")
        | Some("minimize_total_cost")
        | Some("minimize_cost_usd") => Ok(ServingObjective::MinimizeCost),
        Some("energy")
        | Some("kwh")
        | Some("energy_kwh")
        | Some("minimize_energy")
        | Some("minimize_energy_kwh") => Ok(ServingObjective::MinimizeEnergy),
        Some("power")
        | Some("watts")
        | Some("average_power")
        | Some("average_power_watts")
        | Some("minimize_power")
        | Some("minimize_average_power")
        | Some("minimize_average_power_watts") => Ok(ServingObjective::MinimizePower),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.objective '{value}'; use e2el, ttft, tpot, throughput, slo_miss_rate, memory_pressure, cost, energy, or power"
        ))),
    }
}

fn parse_serving_deployment_mode(mode: Option<&str>) -> Result<ServingDeploymentMode, ConfigError> {
    match mode.map(normalize).as_deref() {
        None | Some("flexible") | Some("any") | Some("auto") => Ok(ServingDeploymentMode::Flexible),
        Some("colocated") | Some("co_located") | Some("collocated") => {
            Ok(ServingDeploymentMode::Colocated)
        }
        Some("partial") | Some("partially_disaggregated") | Some("partial_disaggregated") => {
            Ok(ServingDeploymentMode::PartiallyDisaggregated)
        }
        Some("disaggregated")
        | Some("full")
        | Some("fully_disaggregated")
        | Some("full_disaggregated") => Ok(ServingDeploymentMode::FullyDisaggregated),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.mode '{value}'; use flexible, colocated, partially_disaggregated, fully_disaggregated, or disaggregated"
        ))),
    }
}

fn parse_pool_candidates(
    serving: &ServingSection,
) -> Result<Vec<ServingPoolCandidate>, ConfigError> {
    let mut candidates = Vec::new();
    let has_top_level_pool = serving.prefill_nodes.is_some()
        || serving.decode_nodes.is_some()
        || serving.prefill_groups.is_some()
        || serving.decode_groups.is_some();
    if has_top_level_pool {
        candidates.push(parse_pool_candidate(
            "serving",
            PoolCandidateInput {
                label: None,
                prefill_nodes: serving.prefill_nodes.clone().unwrap_or_default(),
                decode_nodes: serving.decode_nodes.clone().unwrap_or_default(),
                prefill_groups: serving.prefill_groups.clone().unwrap_or_default(),
                decode_groups: serving.decode_groups.clone().unwrap_or_default(),
                prefill_node_filter: ServingPoolNodeFilter::default(),
                decode_node_filter: ServingPoolNodeFilter::default(),
                domain_spread: ServingPoolDomainSpread::default(),
                prefill_gpu_tag: serving.prefill_gpu_tag.clone(),
                prefill_gpu_tags: serving.prefill_gpu_tags.clone(),
                decode_gpu_tag: serving.decode_gpu_tag.clone(),
                decode_gpu_tags: serving.decode_gpu_tags.clone(),
            },
        )?);
    }

    for (idx, candidate) in serving
        .pool_candidates
        .as_ref()
        .into_iter()
        .flatten()
        .enumerate()
    {
        candidates.push(parse_pool_candidate(
            &format!("serving.pool_candidates[{idx}]"),
            PoolCandidateInput {
                label: candidate.label.clone(),
                prefill_nodes: candidate.prefill_nodes.clone().unwrap_or_default(),
                decode_nodes: candidate.decode_nodes.clone().unwrap_or_default(),
                prefill_groups: candidate.prefill_groups.clone().unwrap_or_default(),
                decode_groups: candidate.decode_groups.clone().unwrap_or_default(),
                prefill_node_filter: parse_serving_pool_candidate_node_filter(
                    &format!("serving.pool_candidates[{idx}].prefill"),
                    candidate.prefill_node_tag.as_deref(),
                    candidate.prefill_node_tags.as_deref(),
                    candidate.prefill_rack.as_deref(),
                    candidate.prefill_racks.as_deref(),
                    candidate.prefill_island.as_deref(),
                    candidate.prefill_islands.as_deref(),
                    candidate.prefill_failure_domain.as_deref(),
                    candidate.prefill_failure_domains.as_deref(),
                    candidate.prefill_exclude_node_tag.as_deref(),
                    candidate.prefill_exclude_node_tags.as_deref(),
                    candidate.prefill_exclude_rack.as_deref(),
                    candidate.prefill_exclude_racks.as_deref(),
                    candidate.prefill_exclude_island.as_deref(),
                    candidate.prefill_exclude_islands.as_deref(),
                    candidate.prefill_exclude_failure_domain.as_deref(),
                    candidate.prefill_exclude_failure_domains.as_deref(),
                )?,
                decode_node_filter: parse_serving_pool_candidate_node_filter(
                    &format!("serving.pool_candidates[{idx}].decode"),
                    candidate.decode_node_tag.as_deref(),
                    candidate.decode_node_tags.as_deref(),
                    candidate.decode_rack.as_deref(),
                    candidate.decode_racks.as_deref(),
                    candidate.decode_island.as_deref(),
                    candidate.decode_islands.as_deref(),
                    candidate.decode_failure_domain.as_deref(),
                    candidate.decode_failure_domains.as_deref(),
                    candidate.decode_exclude_node_tag.as_deref(),
                    candidate.decode_exclude_node_tags.as_deref(),
                    candidate.decode_exclude_rack.as_deref(),
                    candidate.decode_exclude_racks.as_deref(),
                    candidate.decode_exclude_island.as_deref(),
                    candidate.decode_exclude_islands.as_deref(),
                    candidate.decode_exclude_failure_domain.as_deref(),
                    candidate.decode_exclude_failure_domains.as_deref(),
                )?,
                domain_spread: parse_serving_pool_candidate_domain_spread(
                    &format!("serving.pool_candidates[{idx}]"),
                    candidate,
                )?,
                prefill_gpu_tag: candidate.prefill_gpu_tag.clone(),
                prefill_gpu_tags: candidate.prefill_gpu_tags.clone(),
                decode_gpu_tag: candidate.decode_gpu_tag.clone(),
                decode_gpu_tags: candidate.decode_gpu_tags.clone(),
            },
        )?);
    }

    Ok(dedup_pool_candidates(candidates))
}

fn parse_pool_search(
    pool_search: Option<ServingPoolSearchSection>,
    deployment_mode: ServingDeploymentMode,
) -> Result<Option<ServingPoolSearch>, ConfigError> {
    let Some(pool_search) = pool_search else {
        return Ok(None);
    };
    let max_candidates = pool_search.max_candidates.unwrap_or(64);
    if max_candidates == 0 {
        return Err(ConfigError::new(
            "serving.pool_search.max_candidates must be greater than zero",
        ));
    }
    let prefill_node_filter = parse_pool_search_node_filter(
        "serving.pool_search.prefill",
        pool_search.prefill_node_tag.as_deref(),
        pool_search.prefill_node_tags.as_deref(),
        pool_search.prefill_rack.as_deref(),
        pool_search.prefill_racks.as_deref(),
        pool_search.prefill_island.as_deref(),
        pool_search.prefill_islands.as_deref(),
        pool_search.prefill_failure_domain.as_deref(),
        pool_search.prefill_failure_domains.as_deref(),
        pool_search.prefill_exclude_node_tag.as_deref(),
        pool_search.prefill_exclude_node_tags.as_deref(),
        pool_search.prefill_exclude_rack.as_deref(),
        pool_search.prefill_exclude_racks.as_deref(),
        pool_search.prefill_exclude_island.as_deref(),
        pool_search.prefill_exclude_islands.as_deref(),
        pool_search.prefill_exclude_failure_domain.as_deref(),
        pool_search.prefill_exclude_failure_domains.as_deref(),
    )?;
    let decode_node_filter = parse_pool_search_node_filter(
        "serving.pool_search.decode",
        pool_search.decode_node_tag.as_deref(),
        pool_search.decode_node_tags.as_deref(),
        pool_search.decode_rack.as_deref(),
        pool_search.decode_racks.as_deref(),
        pool_search.decode_island.as_deref(),
        pool_search.decode_islands.as_deref(),
        pool_search.decode_failure_domain.as_deref(),
        pool_search.decode_failure_domains.as_deref(),
        pool_search.decode_exclude_node_tag.as_deref(),
        pool_search.decode_exclude_node_tags.as_deref(),
        pool_search.decode_exclude_rack.as_deref(),
        pool_search.decode_exclude_racks.as_deref(),
        pool_search.decode_exclude_island.as_deref(),
        pool_search.decode_exclude_islands.as_deref(),
        pool_search.decode_exclude_failure_domain.as_deref(),
        pool_search.decode_exclude_failure_domains.as_deref(),
    )?;
    let domain_spread = parse_pool_search_domain_spread(&pool_search)?;

    Ok(Some(ServingPoolSearch {
        prefill_groups: normalized_group_labels(
            "serving.pool_search.prefill_groups",
            require_nonempty(
                "serving.pool_search.prefill_groups",
                pool_search.prefill_groups,
            )?,
        )?,
        decode_groups: normalized_group_labels(
            "serving.pool_search.decode_groups",
            require_nonempty(
                "serving.pool_search.decode_groups",
                pool_search.decode_groups,
            )?,
        )?,
        prefill_node_counts: positive_values(
            "serving.pool_search.prefill_node_counts",
            pool_search.prefill_node_counts.unwrap_or_default(),
        )?,
        decode_node_counts: positive_values(
            "serving.pool_search.decode_node_counts",
            pool_search.decode_node_counts.unwrap_or_default(),
        )?,
        prefill_node_filter,
        decode_node_filter,
        prefill_gpu_labels: parse_gpu_labels(
            "serving.pool_search.prefill_gpu_tags",
            pool_search.prefill_gpu_tag.as_deref(),
            pool_search.prefill_gpu_tags.as_deref(),
        )?
        .into_iter()
        .collect(),
        decode_gpu_labels: parse_gpu_labels(
            "serving.pool_search.decode_gpu_tags",
            pool_search.decode_gpu_tag.as_deref(),
            pool_search.decode_gpu_tags.as_deref(),
        )?
        .into_iter()
        .collect(),
        allow_overlap: pool_search.allow_overlap.unwrap_or(matches!(
            deployment_mode,
            ServingDeploymentMode::Colocated | ServingDeploymentMode::PartiallyDisaggregated
        )),
        domain_spread,
        max_candidates,
    }))
}

fn parse_serving_pool_candidate_domain_spread(
    name: &str,
    candidate: &ServingPoolCandidateSection,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    parse_pool_domain_spread(
        name,
        candidate.min_prefill_racks,
        candidate.min_decode_racks,
        candidate.min_prefill_islands,
        candidate.min_decode_islands,
        candidate.min_prefill_failure_domains,
        candidate.min_decode_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
fn parse_serving_pool_candidate_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    parse_pool_node_filter(
        name,
        node_tag,
        node_tags,
        rack,
        racks,
        island,
        islands,
        failure_domain,
        failure_domains,
        exclude_node_tag,
        exclude_node_tags,
        exclude_rack,
        exclude_racks,
        exclude_island,
        exclude_islands,
        exclude_failure_domain,
        exclude_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
fn parse_pool_search_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    parse_pool_node_filter(
        name,
        node_tag,
        node_tags,
        rack,
        racks,
        island,
        islands,
        failure_domain,
        failure_domains,
        exclude_node_tag,
        exclude_node_tags,
        exclude_rack,
        exclude_racks,
        exclude_island,
        exclude_islands,
        exclude_failure_domain,
        exclude_failure_domains,
    )
}

#[allow(clippy::too_many_arguments)]
fn parse_pool_node_filter(
    name: &str,
    node_tag: Option<&str>,
    node_tags: Option<&[String]>,
    rack: Option<&str>,
    racks: Option<&[String]>,
    island: Option<&str>,
    islands: Option<&[String]>,
    failure_domain: Option<&str>,
    failure_domains: Option<&[String]>,
    exclude_node_tag: Option<&str>,
    exclude_node_tags: Option<&[String]>,
    exclude_rack: Option<&str>,
    exclude_racks: Option<&[String]>,
    exclude_island: Option<&str>,
    exclude_islands: Option<&[String]>,
    exclude_failure_domain: Option<&str>,
    exclude_failure_domains: Option<&[String]>,
) -> Result<ServingPoolNodeFilter, ConfigError> {
    Ok(ServingPoolNodeFilter {
        node_labels: parse_optional_normalized_labels(
            &format!("{name}_node_tags"),
            node_tag,
            node_tags,
        )?,
        racks: parse_optional_topology_domains(&format!("{name}_racks"), rack, racks)?,
        islands: parse_optional_topology_domains(&format!("{name}_islands"), island, islands)?,
        failure_domains: parse_optional_topology_domains(
            &format!("{name}_failure_domains"),
            failure_domain,
            failure_domains,
        )?,
        exclude_node_labels: parse_optional_normalized_labels(
            &format!("{name}_exclude_node_tags"),
            exclude_node_tag,
            exclude_node_tags,
        )?,
        exclude_racks: parse_optional_topology_domains(
            &format!("{name}_exclude_racks"),
            exclude_rack,
            exclude_racks,
        )?,
        exclude_islands: parse_optional_topology_domains(
            &format!("{name}_exclude_islands"),
            exclude_island,
            exclude_islands,
        )?,
        exclude_failure_domains: parse_optional_topology_domains(
            &format!("{name}_exclude_failure_domains"),
            exclude_failure_domain,
            exclude_failure_domains,
        )?,
    })
}

fn parse_pool_search_domain_spread(
    pool_search: &ServingPoolSearchSection,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    parse_pool_domain_spread(
        "serving.pool_search",
        pool_search.min_prefill_racks,
        pool_search.min_decode_racks,
        pool_search.min_prefill_islands,
        pool_search.min_decode_islands,
        pool_search.min_prefill_failure_domains,
        pool_search.min_decode_failure_domains,
    )
}

fn parse_pool_domain_spread(
    name: &str,
    min_prefill_racks: Option<u32>,
    min_decode_racks: Option<u32>,
    min_prefill_islands: Option<u32>,
    min_decode_islands: Option<u32>,
    min_prefill_failure_domains: Option<u32>,
    min_decode_failure_domains: Option<u32>,
) -> Result<ServingPoolDomainSpread, ConfigError> {
    validate_positive_optional_u32(&format!("{name}.min_prefill_racks"), min_prefill_racks)?;
    validate_positive_optional_u32(&format!("{name}.min_decode_racks"), min_decode_racks)?;
    validate_positive_optional_u32(&format!("{name}.min_prefill_islands"), min_prefill_islands)?;
    validate_positive_optional_u32(&format!("{name}.min_decode_islands"), min_decode_islands)?;
    validate_positive_optional_u32(
        &format!("{name}.min_prefill_failure_domains"),
        min_prefill_failure_domains,
    )?;
    validate_positive_optional_u32(
        &format!("{name}.min_decode_failure_domains"),
        min_decode_failure_domains,
    )?;
    Ok(ServingPoolDomainSpread {
        min_prefill_racks,
        min_decode_racks,
        min_prefill_islands,
        min_decode_islands,
        min_prefill_failure_domains,
        min_decode_failure_domains,
    })
}

fn parse_slo_policies(
    policies: Option<Vec<ServingSloPolicySection>>,
) -> Result<Vec<ServingSloPolicy>, ConfigError> {
    let mut parsed = Vec::new();
    for (idx, policy) in policies.unwrap_or_default().into_iter().enumerate() {
        let name = format!("serving.slo_policies[{idx}]");
        let group = parse_slo_policy_group(&format!("{name}.group"), &policy.group)?;
        let key = policy.key.trim().to_string();
        if key.is_empty() {
            return Err(ConfigError::new(format!("{name}.key must not be empty")));
        }
        let parsed_policy = ServingSloPolicy {
            group,
            key,
            max_ttft_slo_miss_rate: parse_fraction(
                &format!("{name}.max_ttft_slo_miss_rate"),
                policy.max_ttft_slo_miss_rate,
            )?,
            max_tpot_slo_miss_rate: parse_fraction(
                &format!("{name}.max_tpot_slo_miss_rate"),
                policy.max_tpot_slo_miss_rate,
            )?,
            max_itl_slo_miss_rate: parse_fraction(
                &format!("{name}.max_itl_slo_miss_rate"),
                policy.max_itl_slo_miss_rate,
            )?,
            max_e2el_slo_miss_rate: parse_fraction(
                &format!("{name}.max_e2el_slo_miss_rate"),
                policy.max_e2el_slo_miss_rate,
            )?,
            max_deadline_miss_rate: parse_fraction(
                &format!("{name}.max_deadline_miss_rate"),
                policy.max_deadline_miss_rate,
            )?,
        };
        if parsed_policy.max_ttft_slo_miss_rate.is_none()
            && parsed_policy.max_tpot_slo_miss_rate.is_none()
            && parsed_policy.max_itl_slo_miss_rate.is_none()
            && parsed_policy.max_e2el_slo_miss_rate.is_none()
            && parsed_policy.max_deadline_miss_rate.is_none()
        {
            return Err(ConfigError::new(format!(
                "{name} must set at least one max_*_miss_rate field"
            )));
        }
        parsed.push(parsed_policy);
    }
    Ok(parsed)
}

fn parse_traffic_classes(
    classes: Option<Vec<ServingTrafficClassSection>>,
) -> Result<Vec<ServingTrafficClass>, ConfigError> {
    let mut parsed = Vec::new();
    let mut names = HashSet::new();
    let mut selectors = BTreeMap::new();
    for (idx, class) in classes.unwrap_or_default().into_iter().enumerate() {
        let name = format!("serving.traffic_classes[{idx}]");
        let class_name = class.name.trim().to_string();
        if class_name.is_empty() {
            return Err(ConfigError::new(format!("{name}.name must not be empty")));
        }
        if !names.insert(class_name.clone()) {
            return Err(ConfigError::new(format!(
                "{name}.name '{class_name}' is duplicated"
            )));
        }
        let group = parse_traffic_class_group(&format!("{name}.group"), &class.group)?;
        let slo_miss_penalty_weights = parse_traffic_class_slo_miss_penalty_weights(&name, &class)?;
        let key = parse_traffic_class_key(&name, &group, class.key, class.priority)?;
        if let Some(first_name) = selectors.insert((group.clone(), key.clone()), class_name.clone())
        {
            return Err(ConfigError::new(format!(
                "{name} selector group='{group}' key='{key}' duplicates traffic class '{first_name}'"
            )));
        }
        if let Some(0) = class.max_prefill_tokens {
            return Err(ConfigError::new(format!(
                "{name}.max_prefill_tokens must be greater than zero"
            )));
        }
        if let Some(0) = class.max_decode_sequences {
            return Err(ConfigError::new(format!(
                "{name}.max_decode_sequences must be greater than zero"
            )));
        }
        if let Some(0) = class.max_resident_tokens {
            return Err(ConfigError::new(format!(
                "{name}.max_resident_tokens must be greater than zero"
            )));
        }
        if let Some(0) = class.max_kv_blocks {
            return Err(ConfigError::new(format!(
                "{name}.max_kv_blocks must be greater than zero"
            )));
        }
        let parsed_class = ServingTrafficClass {
            name: class_name,
            group,
            key,
            admission_priority: class.admission_priority,
            max_prefill_tokens: class.max_prefill_tokens,
            max_decode_sequences: class.max_decode_sequences,
            max_resident_tokens: class.max_resident_tokens,
            max_kv_blocks: class.max_kv_blocks,
            slo: ServingRequestSlo {
                ttft_s: parse_slo_ms(&format!("{name}.ttft_slo_ms"), class.ttft_slo_ms)?,
                tpot_s: parse_slo_ms(&format!("{name}.tpot_slo_ms"), class.tpot_slo_ms)?,
                itl_s: parse_slo_ms(&format!("{name}.itl_slo_ms"), class.itl_slo_ms)?,
                e2el_s: parse_slo_ms(&format!("{name}.e2el_slo_ms"), class.e2el_slo_ms)?,
            },
            slo_miss_penalty_weights,
            max_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_queue_delay_s"),
                class.max_queue_delay_s,
                &format!("{name}.max_queue_delay_ms"),
                class.max_queue_delay_ms,
            )?,
            max_kv_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_kv_queue_delay_s"),
                class.max_kv_queue_delay_s,
                &format!("{name}.max_kv_queue_delay_ms"),
                class.max_kv_queue_delay_ms,
            )?,
            max_decode_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_decode_queue_delay_s"),
                class.max_decode_queue_delay_s,
                &format!("{name}.max_decode_queue_delay_ms"),
                class.max_decode_queue_delay_ms,
            )?,
            max_decode_iteration_queue_delay_s: parse_positive_optional_seconds(
                &format!("{name}.max_decode_iteration_queue_delay_s"),
                class.max_decode_iteration_queue_delay_s,
                &format!("{name}.max_decode_iteration_queue_delay_ms"),
                class.max_decode_iteration_queue_delay_ms,
            )?,
            request_timeout_s: parse_positive_optional_seconds(
                &format!("{name}.request_timeout_s"),
                class.request_timeout_s,
                &format!("{name}.request_timeout_ms"),
                class.request_timeout_ms,
            )?,
            max_ttft_slo_miss_rate: parse_fraction(
                &format!("{name}.max_ttft_slo_miss_rate"),
                class.max_ttft_slo_miss_rate,
            )?,
            max_tpot_slo_miss_rate: parse_fraction(
                &format!("{name}.max_tpot_slo_miss_rate"),
                class.max_tpot_slo_miss_rate,
            )?,
            max_itl_slo_miss_rate: parse_fraction(
                &format!("{name}.max_itl_slo_miss_rate"),
                class.max_itl_slo_miss_rate,
            )?,
            max_e2el_slo_miss_rate: parse_fraction(
                &format!("{name}.max_e2el_slo_miss_rate"),
                class.max_e2el_slo_miss_rate,
            )?,
            max_deadline_miss_rate: parse_fraction(
                &format!("{name}.max_deadline_miss_rate"),
                class.max_deadline_miss_rate,
            )?,
        };
        if parsed_class.slo == ServingRequestSlo::default()
            && parsed_class.max_queue_delay_s.is_none()
            && parsed_class.max_kv_queue_delay_s.is_none()
            && parsed_class.max_decode_queue_delay_s.is_none()
            && parsed_class.max_decode_iteration_queue_delay_s.is_none()
            && parsed_class.request_timeout_s.is_none()
            && parsed_class.max_ttft_slo_miss_rate.is_none()
            && parsed_class.max_tpot_slo_miss_rate.is_none()
            && parsed_class.max_itl_slo_miss_rate.is_none()
            && parsed_class.max_e2el_slo_miss_rate.is_none()
            && parsed_class.max_deadline_miss_rate.is_none()
            && !parsed_class.slo_miss_penalty_weights.any_nonzero()
            && parsed_class.admission_priority.is_none()
            && parsed_class.max_prefill_tokens.is_none()
            && parsed_class.max_decode_sequences.is_none()
            && parsed_class.max_resident_tokens.is_none()
            && parsed_class.max_kv_blocks.is_none()
        {
            return Err(ConfigError::new(format!(
                "{name} must set at least one admission_priority, class capacity limit, *_slo_ms, max_*_miss_rate, or *_penalty_weight field"
            )));
        }
        parsed.push(parsed_class);
    }
    Ok(parsed)
}

fn parse_traffic_class_slo_miss_penalty_weights(
    name: &str,
    class: &ServingTrafficClassSection,
) -> Result<ServingSloMissPenaltyWeights, ConfigError> {
    Ok(ServingSloMissPenaltyWeights {
        aggregate: parse_non_negative_f64(
            &format!("{name}.slo_miss_penalty_weight"),
            class.slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        ttft: parse_non_negative_f64(
            &format!("{name}.ttft_slo_miss_penalty_weight"),
            class.ttft_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        tpot: parse_non_negative_f64(
            &format!("{name}.tpot_slo_miss_penalty_weight"),
            class.tpot_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        itl: parse_non_negative_f64(
            &format!("{name}.itl_slo_miss_penalty_weight"),
            class.itl_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        e2el: parse_non_negative_f64(
            &format!("{name}.e2el_slo_miss_penalty_weight"),
            class.e2el_slo_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
        deadline: parse_non_negative_f64(
            &format!("{name}.deadline_miss_penalty_weight"),
            class.deadline_miss_penalty_weight,
        )?
        .unwrap_or(0.0),
    })
}

fn traffic_class_slo_policies(classes: &[ServingTrafficClass]) -> Vec<ServingSloPolicy> {
    classes
        .iter()
        .filter_map(|class| {
            let policy = ServingSloPolicy {
                group: class.group.clone(),
                key: class.key.clone(),
                max_ttft_slo_miss_rate: class.max_ttft_slo_miss_rate,
                max_tpot_slo_miss_rate: class.max_tpot_slo_miss_rate,
                max_itl_slo_miss_rate: class.max_itl_slo_miss_rate,
                max_e2el_slo_miss_rate: class.max_e2el_slo_miss_rate,
                max_deadline_miss_rate: class.max_deadline_miss_rate,
            };
            if policy.max_ttft_slo_miss_rate.is_none()
                && policy.max_tpot_slo_miss_rate.is_none()
                && policy.max_itl_slo_miss_rate.is_none()
                && policy.max_e2el_slo_miss_rate.is_none()
                && policy.max_deadline_miss_rate.is_none()
            {
                None
            } else {
                Some(policy)
            }
        })
        .collect()
}

fn parse_traffic_class_group(name: &str, group: &str) -> Result<String, ConfigError> {
    match normalize(group).as_str() {
        "tenant" => Ok("tenant".to_string()),
        "model" | "model_id" => Ok("model_id".to_string()),
        "priority" => Ok("priority".to_string()),
        value => Err(ConfigError::new(format!(
            "unsupported {name} '{value}'; use tenant, model_id, or priority"
        ))),
    }
}

fn parse_traffic_class_key(
    name: &str,
    group: &str,
    key: Option<String>,
    priority: Option<i32>,
) -> Result<String, ConfigError> {
    if group == "priority" {
        if let Some(priority) = priority {
            return Ok(format!("priority-{priority}"));
        }
        let key = key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .ok_or_else(|| {
                ConfigError::new(format!("{name}.key or {name}.priority is required"))
            })?;
        if key.starts_with("priority-") {
            return Ok(key.to_string());
        }
        if let Ok(priority) = key.parse::<i32>() {
            return Ok(format!("priority-{priority}"));
        }
        return Err(ConfigError::new(format!(
            "{name}.key for priority classes must be an integer or priority-<integer>"
        )));
    }
    if priority.is_some() {
        return Err(ConfigError::new(format!(
            "{name}.priority is only valid when group = \"priority\""
        )));
    }
    key.as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ConfigError::new(format!("{name}.key must not be empty")))
}

fn parse_slo_policy_group(name: &str, group: &str) -> Result<String, ConfigError> {
    let normalized = normalize(group);
    match normalized.as_str() {
        "tenant" => Ok("tenant".to_string()),
        "model" | "model_id" => Ok("model_id".to_string()),
        "priority" => Ok("priority".to_string()),
        "prefill_node" => Ok("prefill_node".to_string()),
        "decode_node" => Ok("decode_node".to_string()),
        "prefill_route" => Ok("prefill_route".to_string()),
        "decode_route" => Ok("decode_route".to_string()),
        "" => Err(ConfigError::new(format!("{name} must not be empty"))),
        value => Err(ConfigError::new(format!(
            "unsupported {name} '{value}'; use tenant, model_id, priority, prefill_node, decode_node, prefill_route, or decode_route"
        ))),
    }
}

struct PoolCandidateInput {
    label: Option<String>,
    prefill_nodes: Vec<u32>,
    decode_nodes: Vec<u32>,
    prefill_groups: Vec<String>,
    decode_groups: Vec<String>,
    prefill_node_filter: ServingPoolNodeFilter,
    decode_node_filter: ServingPoolNodeFilter,
    domain_spread: ServingPoolDomainSpread,
    prefill_gpu_tag: Option<String>,
    prefill_gpu_tags: Option<Vec<String>>,
    decode_gpu_tag: Option<String>,
    decode_gpu_tags: Option<Vec<String>>,
}

fn parse_pool_candidate(
    name: &str,
    input: PoolCandidateInput,
) -> Result<ServingPoolCandidate, ConfigError> {
    let prefill_groups =
        normalized_group_labels(&format!("{name}.prefill_groups"), input.prefill_groups)?;
    let decode_groups =
        normalized_group_labels(&format!("{name}.decode_groups"), input.decode_groups)?;
    let prefill_gpu_labels = parse_gpu_labels(
        &format!("{name}.prefill_gpu_tags"),
        input.prefill_gpu_tag.as_deref(),
        input.prefill_gpu_tags.as_deref(),
    )?
    .into_iter()
    .collect();
    let decode_gpu_labels = parse_gpu_labels(
        &format!("{name}.decode_gpu_tags"),
        input.decode_gpu_tag.as_deref(),
        input.decode_gpu_tags.as_deref(),
    )?
    .into_iter()
    .collect();
    if input.prefill_nodes.is_empty() && prefill_groups.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} requires prefill_nodes or prefill_groups"
        )));
    }
    if input.decode_nodes.is_empty() && decode_groups.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} requires decode_nodes or decode_groups"
        )));
    }

    Ok(ServingPoolCandidate {
        label: input.label.filter(|label| !label.trim().is_empty()),
        prefill_nodes: input.prefill_nodes,
        decode_nodes: input.decode_nodes,
        prefill_groups,
        decode_groups,
        prefill_node_filter: input.prefill_node_filter,
        decode_node_filter: input.decode_node_filter,
        domain_spread: input.domain_spread,
        prefill_gpu_labels,
        decode_gpu_labels,
    })
}

fn normalized_group_labels(name: &str, values: Vec<String>) -> Result<Vec<String>, ConfigError> {
    let mut labels = Vec::new();
    for value in values {
        let label = normalize(&value);
        if label.is_empty() {
            return Err(ConfigError::new(format!("{name} values must not be empty")));
        }
        if !labels.contains(&label) {
            labels.push(label);
        }
    }

    Ok(labels)
}

fn dedup_pool_candidates(candidates: Vec<ServingPoolCandidate>) -> Vec<ServingPoolCandidate> {
    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped.iter().any(|existing: &ServingPoolCandidate| {
            same_nodes(&existing.prefill_nodes, &candidate.prefill_nodes)
                && same_nodes(&existing.decode_nodes, &candidate.decode_nodes)
                && same_strings(&existing.prefill_groups, &candidate.prefill_groups)
                && same_strings(&existing.decode_groups, &candidate.decode_groups)
                && same_pool_node_filter(
                    &existing.prefill_node_filter,
                    &candidate.prefill_node_filter,
                )
                && same_pool_node_filter(
                    &existing.decode_node_filter,
                    &candidate.decode_node_filter,
                )
                && existing.domain_spread == candidate.domain_spread
                && same_strings(&existing.prefill_gpu_labels, &candidate.prefill_gpu_labels)
                && same_strings(&existing.decode_gpu_labels, &candidate.decode_gpu_labels)
        }) {
            deduped.push(candidate);
        }
    }

    deduped
}

fn same_nodes(left: &[u32], right: &[u32]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn same_strings(left: &[String], right: &[String]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort();
    right.sort();
    left == right
}

fn same_pool_node_filter(left: &ServingPoolNodeFilter, right: &ServingPoolNodeFilter) -> bool {
    same_strings(&left.node_labels, &right.node_labels)
        && same_strings(&left.racks, &right.racks)
        && same_strings(&left.islands, &right.islands)
        && same_strings(&left.failure_domains, &right.failure_domains)
}

fn validate_search_space_capacity(
    name: &str,
    search: &SearchSpace,
    available_gpus: u32,
) -> Result<(), ConfigError> {
    if search.tensor_ranks.iter().any(|&tensor| {
        search.pipeline_ranks.iter().any(|&pipeline| {
            search.expert_ranks.iter().any(|&expert| {
                search.data_ranks.iter().any(|&data| {
                    tensor
                        .saturating_mul(pipeline)
                        .saturating_mul(expert)
                        .saturating_mul(data)
                        <= available_gpus
                })
            })
        })
    }) {
        return Ok(());
    }

    Err(ConfigError::new(format!(
        "{name} has no rank combination that fits {available_gpus} available GPUs"
    )))
}

fn validate_pool_candidate_for_cluster(
    name: &str,
    cluster: &Cluster,
    candidate: &ServingPoolCandidate,
    deployment_mode: ServingDeploymentMode,
    search: &ServingSearchSpace,
    require_routable_pools: bool,
    model_dtype: DType,
) -> Result<(), ConfigError> {
    validate_no_duplicate_nodes(&format!("{name}.prefill_nodes"), &candidate.prefill_nodes)?;
    validate_no_duplicate_nodes(&format!("{name}.decode_nodes"), &candidate.decode_nodes)?;
    let prefill_nodes = resolve_configured_pool_nodes(
        &format!("{name}.prefill"),
        cluster,
        &candidate.prefill_nodes,
        &candidate.prefill_groups,
    )?;
    let prefill_nodes =
        nodes_matching_pool_node_filter(cluster, &prefill_nodes, &candidate.prefill_node_filter);
    if prefill_nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name}.prefill node topology filters matched no nodes"
        )));
    }
    let decode_nodes = resolve_configured_pool_nodes(
        &format!("{name}.decode"),
        cluster,
        &candidate.decode_nodes,
        &candidate.decode_groups,
    )?;
    let decode_nodes =
        nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
    if decode_nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name}.decode node topology filters matched no nodes"
        )));
    }
    if !pool_nodes_satisfy_domain_spread(
        cluster,
        &candidate.domain_spread,
        &prefill_nodes,
        &decode_nodes,
    ) {
        return Err(ConfigError::new(format!(
            "{name} does not satisfy configured topology-domain spread constraints"
        )));
    }
    validate_serving_deployment_mode(name, deployment_mode, &prefill_nodes, &decode_nodes)?;
    validate_search_space_capacity(
        &format!(
            "{name}.prefill_search for model.dtype {}",
            model_dtype_label(model_dtype)
        ),
        &search.prefill,
        gpu_count_for_nodes_with_labels_and_dtype(
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            model_dtype,
        ),
    )?;
    validate_search_space_capacity(
        &format!(
            "{name}.decode_search for model.dtype {}",
            model_dtype_label(model_dtype)
        ),
        &search.decode,
        gpu_count_for_nodes_with_labels_and_dtype(
            cluster,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            model_dtype,
        ),
    )?;
    if require_routable_pools {
        validate_serving_pool_route(
            name,
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
        )?;
    }
    Ok(())
}

fn validate_pool_search_for_cluster(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    search: &ServingSearchSpace,
    require_routable_pools: bool,
    model_dtype: DType,
) -> Result<(), ConfigError> {
    let mut any_candidate = false;
    let mut any_routable_candidate = false;
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;
        validate_search_space_capacity(
            &format!(
                "serving.pool_search.prefill group '{prefill_group}' for model.dtype {}",
                model_dtype_label(model_dtype)
            ),
            &search.prefill,
            gpu_count_for_nodes_with_labels_and_dtype(
                cluster,
                &prefill_nodes,
                &pool_search.prefill_gpu_labels,
                model_dtype,
            ),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;
            validate_search_space_capacity(
                &format!(
                    "serving.pool_search.decode group '{decode_group}' for model.dtype {}",
                    model_dtype_label(model_dtype)
                ),
                &search.decode,
                gpu_count_for_nodes_with_labels_and_dtype(
                    cluster,
                    &decode_nodes,
                    &pool_search.decode_gpu_labels,
                    model_dtype,
                ),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    if pool_search_can_generate_candidate(
                        cluster,
                        pool_search,
                        &prefill_nodes,
                        prefill_count as usize,
                        &decode_nodes,
                        decode_count as usize,
                        deployment_mode,
                    ) {
                        any_candidate = true;
                        if !require_routable_pools
                            || pool_search_can_generate_routable_candidate(
                                cluster,
                                pool_search,
                                &prefill_nodes,
                                &decode_nodes,
                                (prefill_count as usize, decode_count as usize),
                                (
                                    &pool_search.prefill_gpu_labels,
                                    &pool_search.decode_gpu_labels,
                                ),
                                deployment_mode,
                            )
                        {
                            any_routable_candidate = true;
                        }
                    }
                }
            }
        }
    }

    if !any_candidate {
        Err(ConfigError::new(
            "serving.pool_search cannot generate any prefill/decode pool candidate with the configured groups, counts, node topology filters, GPU labels, overlap policy, and topology-domain spread constraints",
        ))
    } else if require_routable_pools && !any_routable_candidate {
        Err(ConfigError::new(
            "serving.pool_search cannot generate any routable prefill/decode pool candidate with the configured groups, counts, overlap policy, and cluster interconnect; add missing links, choose connected pools, or disable serving.require_routable_pools for failure-scenario sweeps",
        ))
    } else {
        Ok(())
    }
}

fn validate_explicit_serving_placements_for_configured_pools(
    cluster: &Cluster,
    serving: &DisaggregatedServingConfig,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    if prefill_placement.is_none() && decode_placement.is_none() {
        return Ok(());
    }

    if serving.pool_candidates.is_empty()
        && serving.pool_search.is_none()
        && !serving.prefill_nodes.is_empty()
        && !serving.decode_nodes.is_empty()
        && explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &serving.prefill_nodes,
            &[],
            &serving.decode_nodes,
            &[],
            prefill_placement,
            decode_placement,
        )
    {
        return Ok(());
    }

    for candidate in &serving.pool_candidates {
        let prefill_nodes = resolve_configured_pool_nodes(
            "serving.pool_candidates.prefill",
            cluster,
            &candidate.prefill_nodes,
            &candidate.prefill_groups,
        )?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &candidate.prefill_node_filter,
        );
        let decode_nodes = resolve_configured_pool_nodes(
            "serving.pool_candidates.decode",
            cluster,
            &candidate.decode_nodes,
            &candidate.decode_groups,
        )?;
        let decode_nodes =
            nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
        if pool_nodes_satisfy_domain_spread(
            cluster,
            &candidate.domain_spread,
            &prefill_nodes,
            &decode_nodes,
        ) && explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            prefill_placement,
            decode_placement,
        ) {
            return Ok(());
        }
    }

    if let Some(pool_search) = &serving.pool_search
        && pool_search_can_generate_explicit_placement_candidate(
            cluster,
            pool_search,
            serving.deployment_mode,
            prefill_placement,
            decode_placement,
        )?
    {
        return Ok(());
    }

    Err(ConfigError::new(
        "serving explicit prefill/decode placements do not fit any configured serving pool; ensure serving.prefill_placement ranks are inside prefill pools and serving.decode_placement ranks are inside decode pools, including GPU label filters, or remove explicit placements",
    ))
}

fn validate_serving_route_constraints_for_configured_pools(
    cluster: &Cluster,
    serving: &DisaggregatedServingConfig,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<(), ConfigError> {
    let constraints = serving.traffic.kv_route_constraints;
    if !constraints.any() {
        return Ok(());
    }

    let placements = ExplicitServingPlacements {
        prefill: prefill_placement,
        decode: decode_placement,
    };
    let mut checked_candidates = 0usize;
    let mut first_failure = None;

    for (idx, candidate) in serving.pool_candidates.iter().enumerate() {
        let prefill_nodes = resolve_configured_pool_nodes(
            &format!("serving.pool_candidates[{idx}].prefill"),
            cluster,
            &candidate.prefill_nodes,
            &candidate.prefill_groups,
        )?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &candidate.prefill_node_filter,
        );
        let decode_nodes = resolve_configured_pool_nodes(
            &format!("serving.pool_candidates[{idx}].decode"),
            cluster,
            &candidate.decode_nodes,
            &candidate.decode_groups,
        )?;
        let decode_nodes =
            nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
        if !pool_nodes_satisfy_domain_spread(
            cluster,
            &candidate.domain_spread,
            &prefill_nodes,
            &decode_nodes,
        ) || !explicit_serving_placements_match_pool(
            cluster,
            serving.deployment_mode,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            prefill_placement,
            decode_placement,
        ) {
            continue;
        }
        checked_candidates += 1;
        match serving_pool_route_constraints_satisfied(
            cluster,
            &prefill_nodes,
            &candidate.prefill_gpu_labels,
            &decode_nodes,
            &candidate.decode_gpu_labels,
            constraints,
            placements,
        ) {
            Ok(()) => return Ok(()),
            Err(err) => first_failure.get_or_insert(err),
        };
    }

    if let Some(pool_search) = &serving.pool_search
        && pool_search_can_generate_route_constraint_candidate(
            cluster,
            pool_search,
            serving.deployment_mode,
            constraints,
            placements,
            &mut checked_candidates,
            &mut first_failure,
        )?
    {
        return Ok(());
    }

    let first_failure = first_failure
        .map(|failure| format!(" first failure: {failure}"))
        .unwrap_or_default();
    Err(ConfigError::new(format!(
        "serving has no configured prefill/decode pool candidate satisfying configured KV route constraints after checking {checked_candidates} candidate(s); add rail-diverse/GPUDirect-capable routes, choose different pools, or lower serving.min_kv_route_rail_count / serving.require_kv_route_rail_metadata / serving.require_gpudirect_kv_paths.{first_failure}"
    )))
}

#[derive(Copy, Clone)]
struct ExplicitServingPlacements<'a> {
    prefill: Option<&'a RankPlacement>,
    decode: Option<&'a RankPlacement>,
}

fn pool_search_can_generate_route_constraint_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    constraints: ServingKvRouteConstraints,
    placements: ExplicitServingPlacements<'_>,
    checked_candidates: &mut usize,
    first_failure: &mut Option<String>,
) -> Result<bool, ConfigError> {
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    for prefill in combinations(&prefill_nodes, prefill_count as usize) {
                        for decode in combinations(&decode_nodes, decode_count as usize) {
                            if !pool_search_concrete_candidate_satisfies_constraints(
                                cluster,
                                pool_search,
                                &prefill,
                                &decode,
                                deployment_mode,
                            ) || !explicit_serving_placements_match_pool(
                                cluster,
                                deployment_mode,
                                &prefill,
                                &pool_search.prefill_gpu_labels,
                                &decode,
                                &pool_search.decode_gpu_labels,
                                placements.prefill,
                                placements.decode,
                            ) {
                                continue;
                            }
                            *checked_candidates += 1;
                            match serving_pool_route_constraints_satisfied(
                                cluster,
                                &prefill,
                                &pool_search.prefill_gpu_labels,
                                &decode,
                                &pool_search.decode_gpu_labels,
                                constraints,
                                placements,
                            ) {
                                Ok(()) => return Ok(true),
                                Err(err) => {
                                    first_failure.get_or_insert(err);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn explicit_serving_placements_match_pool(
    cluster: &Cluster,
    deployment_mode: ServingDeploymentMode,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> bool {
    deployment_mode.accepts_pool(prefill_nodes, decode_nodes)
        && prefill_placement.is_none_or(|placement| {
            placement_matches_pool_scope(cluster, placement, prefill_nodes, prefill_gpu_labels)
        })
        && decode_placement.is_none_or(|placement| {
            placement_matches_pool_scope(cluster, placement, decode_nodes, decode_gpu_labels)
        })
}

fn placement_matches_pool_scope(
    cluster: &Cluster,
    placement: &RankPlacement,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> bool {
    let node_ids = node_ids.iter().copied().collect::<BTreeSet<_>>();
    placement.rank_to_gpu.iter().all(|addr| {
        node_ids.contains(&addr.node_id)
            && cluster.node(addr.node_id).is_some_and(|node| {
                gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
            })
    })
}

fn pool_search_can_generate_explicit_placement_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> Result<bool, ConfigError> {
    for prefill_group in &pool_search.prefill_groups {
        let prefill_nodes =
            configured_group_nodes("serving.pool_search.prefill_groups", cluster, prefill_group)?;
        let prefill_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_nodes,
            &pool_search.prefill_node_filter,
        );
        let prefill_nodes =
            nodes_with_gpu_labels(cluster, &prefill_nodes, &pool_search.prefill_gpu_labels);
        let prefill_counts = valid_pool_search_counts(
            "serving.pool_search.prefill_node_counts",
            &pool_search.prefill_node_counts,
            prefill_nodes.len(),
        )?;

        for decode_group in &pool_search.decode_groups {
            let decode_nodes =
                configured_group_nodes("serving.pool_search.decode_groups", cluster, decode_group)?;
            let decode_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_nodes,
                &pool_search.decode_node_filter,
            );
            let decode_nodes =
                nodes_with_gpu_labels(cluster, &decode_nodes, &pool_search.decode_gpu_labels);
            let decode_counts = valid_pool_search_counts(
                "serving.pool_search.decode_node_counts",
                &pool_search.decode_node_counts,
                decode_nodes.len(),
            )?;

            for &prefill_count in &prefill_counts {
                for &decode_count in &decode_counts {
                    if pool_search_can_generate_explicit_placement_candidate_for_counts(
                        cluster,
                        pool_search,
                        &prefill_nodes,
                        prefill_count as usize,
                        &decode_nodes,
                        decode_count as usize,
                        deployment_mode,
                        prefill_placement,
                        decode_placement,
                    ) {
                        return Ok(true);
                    }
                }
            }
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn pool_search_can_generate_explicit_placement_candidate_for_counts(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    prefill_count: usize,
    decode_nodes: &[u32],
    decode_count: usize,
    deployment_mode: ServingDeploymentMode,
    prefill_placement: Option<&RankPlacement>,
    decode_placement: Option<&RankPlacement>,
) -> bool {
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) && explicit_serving_placements_match_pool(
                cluster,
                deployment_mode,
                &prefill,
                &pool_search.prefill_gpu_labels,
                &decode,
                &pool_search.decode_gpu_labels,
                prefill_placement,
                decode_placement,
            ) {
                return true;
            }
        }
    }
    false
}

fn validate_serving_deployment_mode(
    name: &str,
    deployment_mode: ServingDeploymentMode,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
) -> Result<(), ConfigError> {
    if deployment_mode.accepts_pool(prefill_nodes, decode_nodes) {
        return Ok(());
    }

    let effective = ServingDeploymentMode::effective_for_pool(prefill_nodes, decode_nodes);
    Err(ConfigError::new(format!(
        "{name} uses {} prefill/decode nodes, but serving.mode '{}' requires {}",
        effective.as_str(),
        deployment_mode.as_str(),
        deployment_mode.as_str()
    )))
}

fn pool_search_can_generate_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    prefill_count: usize,
    decode_nodes: &[u32],
    decode_count: usize,
    deployment_mode: ServingDeploymentMode,
) -> bool {
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) {
                return true;
            }
        }
    }
    false
}

fn pool_search_concrete_candidate_satisfies_constraints(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
    deployment_mode: ServingDeploymentMode,
) -> bool {
    if !pool_search.allow_overlap
        && deployment_mode != ServingDeploymentMode::Colocated
        && deployment_mode != ServingDeploymentMode::PartiallyDisaggregated
        && overlaps_nodes(prefill_nodes, decode_nodes)
    {
        return false;
    }

    deployment_mode.accepts_pool(prefill_nodes, decode_nodes)
        && pool_nodes_satisfy_domain_spread(
            cluster,
            &pool_search.domain_spread,
            prefill_nodes,
            decode_nodes,
        )
}

fn pool_nodes_satisfy_domain_spread(
    cluster: &Cluster,
    domain_spread: &ServingPoolDomainSpread,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
) -> bool {
    pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    )
}

fn pool_nodes_meet_min_domain_count(
    cluster: &Cluster,
    node_ids: &[u32],
    min_count: Option<u32>,
    domain: impl Fn(&Node) -> Option<&str>,
) -> bool {
    let Some(min_count) = min_count else {
        return true;
    };
    let domains = node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id))
        .filter_map(domain)
        .collect::<BTreeSet<_>>();
    domains.len() >= min_count as usize
}

fn pool_search_can_generate_routable_candidate(
    cluster: &Cluster,
    pool_search: &ServingPoolSearch,
    prefill_nodes: &[u32],
    decode_nodes: &[u32],
    counts: (usize, usize),
    gpu_labels: (&[String], &[String]),
    deployment_mode: ServingDeploymentMode,
) -> bool {
    let (prefill_count, decode_count) = counts;
    let (prefill_gpu_labels, decode_gpu_labels) = gpu_labels;
    for prefill in combinations(prefill_nodes, prefill_count) {
        for decode in combinations(decode_nodes, decode_count) {
            if pool_search_concrete_candidate_satisfies_constraints(
                cluster,
                pool_search,
                &prefill,
                &decode,
                deployment_mode,
            ) && serving_pool_route_is_routable(
                cluster,
                &prefill,
                prefill_gpu_labels,
                &decode,
                decode_gpu_labels,
            ) {
                return true;
            }
        }
    }
    false
}

fn validate_serving_pool_route(
    name: &str,
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
) -> Result<(), ConfigError> {
    if serving_pool_route_is_routable(
        cluster,
        prefill_nodes,
        prefill_gpu_labels,
        decode_nodes,
        decode_gpu_labels,
    ) {
        return Ok(());
    }

    Err(ConfigError::new(format!(
        "{name} has no routable KV transfer path between prefill nodes {prefill_nodes:?} labels {prefill_gpu_labels:?} and decode nodes {decode_nodes:?} labels {decode_gpu_labels:?}; add missing custom inter-node links, choose connected prefill/decode pools, adjust GPU tag constraints, or disable serving.require_routable_pools for failure-scenario sweeps"
    )))
}

fn serving_pool_route_is_routable(
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
) -> bool {
    let prefill_gpus =
        available_gpu_addrs_for_nodes_with_labels(cluster, prefill_nodes, prefill_gpu_labels);
    let decode_gpus =
        available_gpu_addrs_for_nodes_with_labels(cluster, decode_nodes, decode_gpu_labels);
    if prefill_gpus.is_empty() || decode_gpus.is_empty() {
        return true;
    }
    Solver::transfer_between_gpus_routable(
        cluster,
        &prefill_gpus,
        &decode_gpus,
        Bytes::from_bytes(1),
    )
}

#[derive(Default)]
struct ServingRouteConstraintSummary {
    inter_node_pair_count: u32,
    routable_inter_node_path_count: u32,
    inter_node_route_resource_count: u32,
    unrailed_inter_node_route_resource_count: u32,
    host_staged_gpu_nic_resource_count: u32,
    rail_ids: BTreeSet<u32>,
}

fn serving_pool_route_constraints_satisfied(
    cluster: &Cluster,
    prefill_nodes: &[u32],
    prefill_gpu_labels: &[String],
    decode_nodes: &[u32],
    decode_gpu_labels: &[String],
    constraints: ServingKvRouteConstraints,
    placements: ExplicitServingPlacements<'_>,
) -> Result<(), String> {
    let prefill_gpus = route_constraint_gpus_for_pool(
        cluster,
        prefill_nodes,
        prefill_gpu_labels,
        placements.prefill,
    );
    let decode_gpus =
        route_constraint_gpus_for_pool(cluster, decode_nodes, decode_gpu_labels, placements.decode);
    if prefill_gpus.is_empty() || decode_gpus.is_empty() {
        return Ok(());
    }

    let summary = serving_route_constraint_summary(cluster, &prefill_gpus, &decode_gpus)?;
    if summary.inter_node_route_resource_count > 0 {
        if let Some(limit) = constraints.min_inter_node_rail_count {
            let rail_count = summary.rail_ids.len().min(u32::MAX as usize) as u32;
            if rail_count < limit {
                return Err(format!(
                    "KV transfer routes expose {rail_count} inter-node rail(s), below configured serving.min_kv_route_rail_count {limit}"
                ));
            }
        }
        if constraints.require_inter_node_rail_metadata
            && summary.unrailed_inter_node_route_resource_count > 0
        {
            return Err(format!(
                "{} inter-node KV route resource(s) lack rail metadata",
                summary.unrailed_inter_node_route_resource_count
            ));
        }
    }
    if constraints.require_gpudirect && summary.host_staged_gpu_nic_resource_count > 0 {
        return Err(format!(
            "{} GPU/NIC KV route resource(s) are host-staged or marked no-GPUDirect",
            summary.host_staged_gpu_nic_resource_count
        ));
    }

    Ok(())
}

fn route_constraint_gpus_for_pool(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
    placement: Option<&RankPlacement>,
) -> Vec<GpuAddr> {
    let Some(placement) = placement else {
        return available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels);
    };
    let node_ids = node_ids.iter().copied().collect::<BTreeSet<_>>();
    placement
        .rank_to_gpu
        .iter()
        .copied()
        .filter(|addr| {
            node_ids.contains(&addr.node_id)
                && cluster.node(addr.node_id).is_some_and(|node| {
                    gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
                })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn serving_route_constraint_summary(
    cluster: &Cluster,
    prefill_gpus: &[GpuAddr],
    decode_gpus: &[GpuAddr],
) -> Result<ServingRouteConstraintSummary, String> {
    let graph = TopologyGraph::from_cluster(cluster);
    let mut summary = ServingRouteConstraintSummary::default();
    let mut first_unroutable_path = None;
    for source in prefill_gpus {
        for destination in decode_gpus {
            if source == destination || source.node_id == destination.node_id {
                continue;
            }
            summary.inter_node_pair_count = summary.inter_node_pair_count.saturating_add(1);
            let Some(path) = graph.route_between_gpus(*source, *destination, Bytes::from_bytes(1))
            else {
                first_unroutable_path.get_or_insert_with(|| {
                    format!(
                        "no routable KV transfer path between node {} gpu {} and node {} gpu {}",
                        source.node_id,
                        source.local_gpu_id,
                        destination.node_id,
                        destination.local_gpu_id
                    )
                });
                continue;
            };
            summary.routable_inter_node_path_count =
                summary.routable_inter_node_path_count.saturating_add(1);
            for resource in path.resources {
                match resource.kind {
                    RoutedResourceKind::InterNodeFabric
                    | RoutedResourceKind::GpuScopedInterNodeFabric => {
                        summary.inter_node_route_resource_count =
                            summary.inter_node_route_resource_count.saturating_add(1);
                        if let Some(rail_id) = resource.rail_id {
                            summary.rail_ids.insert(rail_id);
                        } else {
                            summary.unrailed_inter_node_route_resource_count = summary
                                .unrailed_inter_node_route_resource_count
                                .saturating_add(1);
                        }
                    }
                    RoutedResourceKind::GpuNicLocal => {
                        if route_label_contains_any(
                            &resource.label,
                            &["host_staged", "no_gpudirect"],
                        ) {
                            summary.host_staged_gpu_nic_resource_count =
                                summary.host_staged_gpu_nic_resource_count.saturating_add(1);
                        }
                    }
                    RoutedResourceKind::IntraNodeFabric => {}
                }
            }
        }
    }
    if summary.inter_node_pair_count > 0 && summary.routable_inter_node_path_count == 0 {
        return Err(first_unroutable_path.unwrap_or_else(|| {
            "no routable inter-node KV transfer path between prefill and decode pools".to_string()
        }));
    }
    Ok(summary)
}

fn route_label_contains_any(label: &str, needles: &[&str]) -> bool {
    let label = normalize(label);
    needles.iter().any(|needle| label.contains(needle))
}

fn available_gpu_addrs_for_nodes_with_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> Vec<GpuAddr> {
    let mut addrs = Vec::new();
    for node_id in node_ids {
        let Some(node) = cluster.node(*node_id) else {
            continue;
        };
        let mut gpu_ids: Vec<_> = node.gpus.keys().copied().collect();
        gpu_ids.sort_unstable();
        addrs.extend(
            gpu_ids
                .into_iter()
                .map(|local_gpu_id| GpuAddr {
                    node_id: *node_id,
                    local_gpu_id,
                })
                .filter(|addr| {
                    cluster.is_gpu_available(*addr)
                        && gpu_labels_match(node.gpu_labels(addr.local_gpu_id), gpu_labels)
                }),
        );
    }
    addrs
}

fn gpu_count_for_nodes_with_labels(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
) -> u32 {
    available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels)
        .len()
        .min(u32::MAX as usize) as u32
}

fn gpu_count_for_nodes_with_labels_and_dtype(
    cluster: &Cluster,
    node_ids: &[u32],
    gpu_labels: &[String],
    dtype: DType,
) -> u32 {
    available_gpu_addrs_for_nodes_with_labels(cluster, node_ids, gpu_labels)
        .into_iter()
        .filter(|addr| {
            cluster
                .gpu_profile(*addr)
                .is_some_and(|profile| gpu_profile_supports_dtype(&profile, dtype))
        })
        .count()
        .min(u32::MAX as usize) as u32
}

fn nodes_with_gpu_labels(cluster: &Cluster, node_ids: &[u32], gpu_labels: &[String]) -> Vec<u32> {
    if gpu_labels.is_empty() {
        return node_ids.to_vec();
    }
    node_ids
        .iter()
        .copied()
        .filter(|node_id| gpu_count_for_nodes_with_labels(cluster, &[*node_id], gpu_labels) > 0)
        .collect()
}

fn nodes_matching_pool_node_filter(
    cluster: &Cluster,
    node_ids: &[u32],
    filter: &ServingPoolNodeFilter,
) -> Vec<u32> {
    node_ids
        .iter()
        .copied()
        .filter(|node_id| {
            cluster
                .node(*node_id)
                .is_some_and(|node| node_matches_pool_node_filter(node, filter))
        })
        .collect()
}

fn node_matches_pool_node_filter(node: &Node, filter: &ServingPoolNodeFilter) -> bool {
    (filter.node_labels.is_empty()
        || filter
            .node_labels
            .iter()
            .any(|label| node.topology.labels.contains(label)))
        && (filter.racks.is_empty()
            || node
                .topology
                .rack
                .as_ref()
                .is_some_and(|rack| filter.racks.contains(rack)))
        && (filter.islands.is_empty()
            || node
                .topology
                .island
                .as_ref()
                .is_some_and(|island| filter.islands.contains(island)))
        && (filter.failure_domains.is_empty()
            || node
                .topology
                .failure_domain
                .as_ref()
                .is_some_and(|failure_domain| filter.failure_domains.contains(failure_domain)))
}

fn gpu_labels_match(labels: Option<&BTreeSet<String>>, required: &[String]) -> bool {
    required.is_empty()
        || labels.is_some_and(|labels| required.iter().any(|label| labels.contains(label)))
}

fn validate_no_duplicate_nodes(name: &str, nodes: &[u32]) -> Result<(), ConfigError> {
    let mut seen = HashSet::new();
    for node in nodes {
        if !seen.insert(*node) {
            return Err(ConfigError::new(format!(
                "{name} contains duplicate node id {node}"
            )));
        }
    }
    Ok(())
}

fn resolve_configured_pool_nodes(
    name: &str,
    cluster: &Cluster,
    explicit_nodes: &[u32],
    groups: &[String],
) -> Result<Vec<u32>, ConfigError> {
    let mut nodes = Vec::new();
    for node_id in explicit_nodes {
        if !cluster.nodes.contains_key(node_id) {
            return Err(ConfigError::new(format!(
                "{name}_nodes references unknown node id {node_id}"
            )));
        }
        nodes.push(*node_id);
    }
    for group in groups {
        nodes.extend_from_slice(&configured_group_nodes(
            &format!("{name}_groups"),
            cluster,
            group,
        )?);
    }
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Err(ConfigError::new(format!("{name} resolves to no nodes")));
    }
    Ok(nodes)
}

fn configured_group_nodes(
    name: &str,
    cluster: &Cluster,
    group: &str,
) -> Result<Vec<u32>, ConfigError> {
    let nodes = cluster.node_group(group).ok_or_else(|| {
        ConfigError::new(format!("{name} references unknown node group '{group}'"))
    })?;
    if nodes.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} node group '{group}' contains no nodes"
        )));
    }
    Ok(nodes.to_vec())
}

fn valid_pool_search_counts(
    name: &str,
    configured: &[u32],
    available_nodes: usize,
) -> Result<Vec<u32>, ConfigError> {
    let mut counts = if configured.is_empty() {
        vec![available_nodes as u32]
    } else {
        configured.to_vec()
    };
    counts.sort_unstable();
    counts.dedup();
    let counts: Vec<_> = counts
        .into_iter()
        .filter(|count| *count > 0 && (*count as usize) <= available_nodes)
        .collect();
    if counts.is_empty() {
        return Err(ConfigError::new(format!(
            "{name} has no value that fits {available_nodes} available nodes"
        )));
    }
    Ok(counts)
}

fn combinations(values: &[u32], count: usize) -> Vec<Vec<u32>> {
    if count == 0 || count > values.len() {
        return Vec::new();
    }
    if count == values.len() {
        return vec![values.to_vec()];
    }

    let mut results = Vec::new();
    let mut current = Vec::with_capacity(count);
    push_combinations(values, count, 0, &mut current, &mut results);
    results
}

fn push_combinations(
    values: &[u32],
    count: usize,
    start: usize,
    current: &mut Vec<u32>,
    results: &mut Vec<Vec<u32>>,
) {
    if current.len() == count {
        results.push(current.clone());
        return;
    }

    let needed = count - current.len();
    for idx in start..=values.len() - needed {
        current.push(values[idx]);
        push_combinations(values, count, idx + 1, current, results);
        current.pop();
    }
}

fn overlaps_nodes(left: &[u32], right: &[u32]) -> bool {
    left.iter().any(|node| right.contains(node))
}

fn parse_serving_traffic(
    traffic: Option<ServingTrafficSection>,
    base_dir: Option<&Path>,
    traffic_classes: Vec<ServingTrafficClass>,
    serving_service_sections: ServingServiceSections,
    metric_ceilings: ServingMetricCeilings,
    kv_route_constraints: ServingKvRouteConstraints,
) -> Result<ServingTraffic, ConfigError> {
    let mut services = parse_serving_services(
        "serving",
        serving_service_sections.services,
        serving_service_sections.prefill,
        serving_service_sections.decode,
        serving_service_sections.kv_transfer,
    )?;
    let Some(traffic) = traffic else {
        return Ok(ServingTraffic {
            services: finalize_serving_services(services),
            metric_ceilings,
            kv_route_constraints,
            traffic_classes,
            ..ServingTraffic::default()
        });
    };
    services = merge_serving_services(
        services,
        parse_serving_services(
            "serving.traffic",
            traffic.services.clone(),
            traffic.prefill_service.clone(),
            traffic.decode_service.clone(),
            traffic.kv_transfer_service.clone(),
        )?,
    );

    if let Some(request_count) = traffic.request_count
        && request_count == 0
    {
        return Err(ConfigError::new(
            "serving.traffic.request_count must be greater than zero",
        ));
    }
    let arrival_gap_s = parse_optional_trace_seconds(
        "serving.traffic.arrival_gap_s",
        traffic.arrival_gap_s,
        "serving.traffic.arrival_gap_ms",
        traffic.arrival_gap_ms,
    )?;
    if let Some(arrival_gap_s) = arrival_gap_s
        && (!arrival_gap_s.is_finite() || arrival_gap_s < 0.0)
    {
        return Err(ConfigError::new(
            "serving.traffic.arrival_gap_s/arrival_gap_ms must be finite and non-negative",
        ));
    }
    if let Some(0) = traffic.max_prefill_batch_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_batch_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_chunk_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_chunk_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_tokens_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_tokens_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_tokens_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_tokens_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_prefill_worker_slots_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_prefill_worker_slots_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_sequences {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_sequences must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_resident_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_resident_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_sequences_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_sequences_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_resident_tokens_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_resident_tokens_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_sequences_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_sequences_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_decode_worker_slots_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_worker_slots_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_transfer_worker_slots_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_transfer_worker_slots_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_resident_tokens_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_resident_tokens_per_gpu must be greater than zero",
        ));
    }
    if let Some(0) = traffic.kv_block_tokens {
        return Err(ConfigError::new(
            "serving.traffic.kv_block_tokens must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_blocks {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_blocks must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_blocks_per_node {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_blocks_per_node must be greater than zero",
        ));
    }
    if let Some(0) = traffic.max_kv_blocks_per_gpu {
        return Err(ConfigError::new(
            "serving.traffic.max_kv_blocks_per_gpu must be greater than zero",
        ));
    }
    let arrival = parse_arrival_pattern(&traffic)?;
    let prefill_batching = parse_prefill_batching(
        traffic.prefill_batching.as_deref(),
        traffic.max_prefill_batch_tokens,
        traffic.max_prefill_chunk_tokens,
    )?;
    let decode_batching = parse_decode_batching(
        traffic.decode_batching.as_deref(),
        traffic.max_decode_batch_tokens,
    )?;
    let trace_window = parse_trace_window(&traffic)?;
    let trace_replay = parse_trace_replay(&traffic)?;
    let measurement_window = parse_measurement_window(&traffic)?;
    if let Some(0) = traffic.measurement_steady_state_min_requests {
        return Err(ConfigError::new(
            "serving.traffic.measurement_steady_state_min_requests must be greater than zero",
        ));
    }
    let measurement_steady_state_max_cv = parse_non_negative_f64(
        "serving.traffic.measurement_steady_state_max_cv",
        traffic.measurement_steady_state_max_cv,
    )?;
    let max_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_queue_delay_s",
        traffic.max_queue_delay_s,
        "serving.traffic.max_queue_delay_ms",
        traffic.max_queue_delay_ms,
    )?;
    let max_kv_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_kv_queue_delay_s",
        traffic.max_kv_queue_delay_s,
        "serving.traffic.max_kv_queue_delay_ms",
        traffic.max_kv_queue_delay_ms,
    )?;
    let max_decode_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_decode_queue_delay_s",
        traffic.max_decode_queue_delay_s,
        "serving.traffic.max_decode_queue_delay_ms",
        traffic.max_decode_queue_delay_ms,
    )?;
    let max_decode_iteration_queue_delay_s = parse_positive_optional_seconds(
        "serving.traffic.max_decode_iteration_queue_delay_s",
        traffic.max_decode_iteration_queue_delay_s,
        "serving.traffic.max_decode_iteration_queue_delay_ms",
        traffic.max_decode_iteration_queue_delay_ms,
    )?;
    let request_timeout_s = parse_positive_optional_seconds(
        "serving.traffic.request_timeout_s",
        traffic.request_timeout_s,
        "serving.traffic.request_timeout_ms",
        traffic.request_timeout_ms,
    )?;
    let trace_requests = parse_trace_requests(
        traffic.requests,
        traffic.trace_csv,
        traffic.trace_jsonl,
        base_dir,
    )?;
    let trace_requests = apply_trace_window(trace_requests, trace_window)?;
    let trace_requests = apply_trace_replay(trace_requests, trace_replay)?;
    validate_unique_trace_request_ids(&trace_requests)?;
    if !trace_requests.is_empty()
        && let Some(request_count) = traffic.request_count
        && request_count as usize != trace_requests.len()
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.request_count must match serving.traffic.requests length ({}) when trace requests are provided",
            trace_requests.len()
        )));
    }
    if matches!(arrival, ServingArrivalPattern::TraceDerived) && trace_requests.is_empty() {
        return Err(ConfigError::new(
            "serving.traffic arrival = 'trace_derived' requires inline requests, trace_csv, or trace_jsonl",
        ));
    }

    Ok(ServingTraffic {
        request_count: traffic.request_count,
        arrival_gap_s,
        arrival,
        routing_policy: parse_routing_policy(traffic.routing_policy.as_deref())?,
        prefill_batching,
        decode_batching,
        decode_capacity_policy: parse_decode_capacity_policy(
            traffic.decode_capacity_policy.as_deref(),
        )?,
        services: finalize_serving_services(services),
        service_backpressure_penalty_weight: parse_non_negative_f64(
            "serving.traffic.service_backpressure_penalty_weight",
            traffic.service_backpressure_penalty_weight,
        )?
        .unwrap_or(0.0),
        max_prefill_tokens: traffic.max_prefill_tokens,
        max_prefill_tokens_per_node: traffic.max_prefill_tokens_per_node,
        max_prefill_tokens_per_gpu: traffic.max_prefill_tokens_per_gpu,
        max_prefill_worker_slots_per_gpu: traffic.max_prefill_worker_slots_per_gpu,
        max_decode_sequences: traffic.max_decode_sequences,
        max_resident_tokens: traffic.max_resident_tokens,
        max_decode_sequences_per_node: traffic.max_decode_sequences_per_node,
        max_resident_tokens_per_node: traffic.max_resident_tokens_per_node,
        max_decode_sequences_per_gpu: traffic.max_decode_sequences_per_gpu,
        max_decode_worker_slots_per_gpu: traffic.max_decode_worker_slots_per_gpu,
        max_resident_tokens_per_gpu: traffic.max_resident_tokens_per_gpu,
        max_kv_transfer_worker_slots_per_gpu: traffic.max_kv_transfer_worker_slots_per_gpu,
        kv_block_tokens: traffic.kv_block_tokens,
        max_kv_blocks: traffic.max_kv_blocks,
        max_kv_blocks_per_node: traffic.max_kv_blocks_per_node,
        max_kv_blocks_per_gpu: traffic.max_kv_blocks_per_gpu,
        ttft_slo_s: parse_slo_ms("serving.traffic.ttft_slo_ms", traffic.ttft_slo_ms)?,
        tpot_slo_s: parse_slo_ms("serving.traffic.tpot_slo_ms", traffic.tpot_slo_ms)?,
        itl_slo_s: parse_slo_ms("serving.traffic.itl_slo_ms", traffic.itl_slo_ms)?,
        e2el_slo_s: parse_slo_ms("serving.traffic.e2el_slo_ms", traffic.e2el_slo_ms)?,
        max_ttft_slo_miss_rate: parse_fraction(
            "serving.traffic.max_ttft_slo_miss_rate",
            traffic.max_ttft_slo_miss_rate,
        )?,
        max_tpot_slo_miss_rate: parse_fraction(
            "serving.traffic.max_tpot_slo_miss_rate",
            traffic.max_tpot_slo_miss_rate,
        )?,
        max_itl_slo_miss_rate: parse_fraction(
            "serving.traffic.max_itl_slo_miss_rate",
            traffic.max_itl_slo_miss_rate,
        )?,
        max_e2el_slo_miss_rate: parse_fraction(
            "serving.traffic.max_e2el_slo_miss_rate",
            traffic.max_e2el_slo_miss_rate,
        )?,
        max_deadline_miss_rate: parse_fraction(
            "serving.traffic.max_deadline_miss_rate",
            traffic.max_deadline_miss_rate,
        )?,
        metric_ceilings,
        kv_route_constraints,
        measurement_start_s: measurement_window.start_s,
        measurement_end_s: measurement_window.end_s,
        measurement_warmup_s: measurement_window.warmup_s,
        measurement_cooldown_s: measurement_window.cooldown_s,
        measurement_steady_state: traffic.measurement_steady_state.unwrap_or(false),
        measurement_steady_state_min_requests: traffic.measurement_steady_state_min_requests,
        measurement_steady_state_max_cv,
        max_queue_delay_s,
        max_kv_queue_delay_s,
        max_decode_queue_delay_s,
        max_decode_iteration_queue_delay_s,
        request_timeout_s,
        shape_seed: traffic.shape_seed.unwrap_or(1),
        prefix_cache_hit_rate: parse_cache_hit_rate(
            "serving.traffic.prefix_cache_hit_rate",
            traffic.prefix_cache_hit_rate,
        )?,
        batch_size_distribution: parse_value_distribution(
            "serving.traffic.batch_size_distribution",
            traffic.batch_size_distribution,
        )?,
        prompt_tokens_distribution: parse_value_distribution(
            "serving.traffic.prompt_tokens_distribution",
            traffic.prompt_tokens_distribution,
        )?,
        decode_tokens_distribution: parse_value_distribution(
            "serving.traffic.decode_tokens_distribution",
            traffic.decode_tokens_distribution,
        )?,
        shape_profiles: parse_shape_profiles(
            "serving.traffic.shape_profiles",
            traffic.shape_profiles,
        )?,
        batch_sizes: positive_values(
            "serving.traffic.batch_sizes",
            traffic.batch_sizes.unwrap_or_default(),
        )?,
        prompt_tokens: positive_values(
            "serving.traffic.prompt_tokens",
            traffic.prompt_tokens.unwrap_or_default(),
        )?,
        decode_tokens: positive_values(
            "serving.traffic.decode_tokens",
            traffic.decode_tokens.unwrap_or_default(),
        )?,
        trace_requests,
        traffic_classes,
    })
}

#[derive(Default)]
struct PartialServingServicesConfig {
    prefill: Option<ServingServicePhaseConfig>,
    decode: Option<ServingServicePhaseConfig>,
    kv_transfer: Option<ServingServicePhaseConfig>,
}

fn parse_serving_services(
    path: &str,
    services: Option<ServingServicesSection>,
    prefill: Option<ServingServicePhaseSection>,
    decode: Option<ServingServicePhaseSection>,
    kv_transfer: Option<ServingServicePhaseSection>,
) -> Result<PartialServingServicesConfig, ConfigError> {
    let mut parsed = PartialServingServicesConfig::default();
    if let Some(services) = services {
        parsed.prefill =
            parse_serving_service_phase(&format!("{path}.services.prefill"), services.prefill)?;
        parsed.decode =
            parse_serving_service_phase(&format!("{path}.services.decode"), services.decode)?;
        parsed.kv_transfer = parse_serving_service_phase(
            &format!("{path}.services.kv_transfer"),
            services.kv_transfer,
        )?;
    }
    parsed.prefill = parse_serving_service_phase(&format!("{path}.prefill_service"), prefill)?
        .or(parsed.prefill);
    parsed.decode =
        parse_serving_service_phase(&format!("{path}.decode_service"), decode)?.or(parsed.decode);
    parsed.kv_transfer =
        parse_serving_service_phase(&format!("{path}.kv_transfer_service"), kv_transfer)?
            .or(parsed.kv_transfer);
    Ok(parsed)
}

fn parse_serving_service_phase(
    path: &str,
    section: Option<ServingServicePhaseSection>,
) -> Result<Option<ServingServicePhaseConfig>, ConfigError> {
    let Some(section) = section else {
        return Ok(None);
    };
    let mut health = parse_serving_service_health(path, section.health.as_deref())?;
    if section.enabled == Some(false) {
        health = ServingServiceHealth::Unavailable;
    }
    let worker_scale = parse_serving_service_worker_scale(path, section.worker_scale)?;
    Ok(Some(ServingServicePhaseConfig {
        health,
        worker_scale,
    }))
}

fn parse_serving_service_health(
    path: &str,
    health: Option<&str>,
) -> Result<ServingServiceHealth, ConfigError> {
    match health.map(normalize).as_deref() {
        None | Some("healthy") | Some("available") | Some("enabled") => {
            Ok(ServingServiceHealth::Healthy)
        }
        Some("draining") | Some("drain") => Ok(ServingServiceHealth::Draining),
        Some("unavailable") | Some("disabled") | Some("down") => {
            Ok(ServingServiceHealth::Unavailable)
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported {path}.health '{value}'; use healthy, draining, or unavailable"
        ))),
    }
}

fn parse_serving_service_worker_scale(
    path: &str,
    worker_scale: Option<f64>,
) -> Result<f64, ConfigError> {
    let Some(worker_scale) = worker_scale else {
        return Ok(1.0);
    };
    if !worker_scale.is_finite() || worker_scale <= 0.0 {
        return Err(ConfigError::new(format!(
            "{path}.worker_scale must be finite and greater than zero"
        )));
    }
    Ok(worker_scale)
}

fn merge_serving_services(
    mut base: PartialServingServicesConfig,
    override_config: PartialServingServicesConfig,
) -> PartialServingServicesConfig {
    if override_config.prefill.is_some() {
        base.prefill = override_config.prefill;
    }
    if override_config.decode.is_some() {
        base.decode = override_config.decode;
    }
    if override_config.kv_transfer.is_some() {
        base.kv_transfer = override_config.kv_transfer;
    }
    base
}

fn finalize_serving_services(partial: PartialServingServicesConfig) -> ServingServicesConfig {
    ServingServicesConfig {
        prefill: partial.prefill.unwrap_or_default(),
        decode: partial.decode.unwrap_or_default(),
        kv_transfer: partial.kv_transfer.unwrap_or_default(),
    }
}

fn parse_positive_optional_seconds(
    seconds_name: &str,
    seconds: Option<f64>,
    millis_name: &str,
    millis: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    let value = parse_optional_trace_seconds(seconds_name, seconds, millis_name, millis)?;
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(ConfigError::new(format!(
            "{seconds_name}/{millis_name} must be positive"
        )));
    }
    Ok(value)
}

fn resolve_config_path(path: &str, base_dir: Option<&Path>) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else if let Some(base_dir) = base_dir {
        base_dir.join(path)
    } else {
        path
    }
}

fn parse_value_distribution(
    name: &str,
    distribution: Option<ServingValueDistributionSection>,
) -> Result<Option<ServingValueDistribution>, ConfigError> {
    let Some(distribution) = distribution else {
        return Ok(None);
    };

    match normalize(&distribution.kind).as_str() {
        "uniform" => {
            let min = distribution
                .min
                .ok_or_else(|| ConfigError::new(format!("{name}.min is required")))?;
            let max = distribution
                .max
                .ok_or_else(|| ConfigError::new(format!("{name}.max is required")))?;
            if min == 0 || max == 0 || min > max {
                return Err(ConfigError::new(format!(
                    "{name}.min and {name}.max must be positive with min <= max"
                )));
            }
            Ok(Some(ServingValueDistribution::Uniform { min, max }))
        }
        "weighted" | "categorical" => {
            let values = positive_values(
                &format!("{name}.values"),
                distribution.values.unwrap_or_default(),
            )?;
            let weights = distribution.weights.unwrap_or_default();
            if values.is_empty() {
                return Err(ConfigError::new(format!("{name}.values must not be empty")));
            }
            if values.len() != weights.len() {
                return Err(ConfigError::new(format!(
                    "{name}.values and {name}.weights must have the same length"
                )));
            }
            if !weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0)
            {
                return Err(ConfigError::new(format!(
                    "{name}.weights must be finite and positive"
                )));
            }
            Ok(Some(ServingValueDistribution::Weighted { values, weights }))
        }
        "lognormal" | "log_normal" => {
            let median = distribution
                .median
                .ok_or_else(|| ConfigError::new(format!("{name}.median is required")))?;
            let sigma = distribution
                .sigma
                .ok_or_else(|| ConfigError::new(format!("{name}.sigma is required")))?;
            let min = distribution
                .min
                .ok_or_else(|| ConfigError::new(format!("{name}.min is required")))?;
            let max = distribution
                .max
                .ok_or_else(|| ConfigError::new(format!("{name}.max is required")))?;
            if !median.is_finite() || median <= 0.0 {
                return Err(ConfigError::new(format!(
                    "{name}.median must be finite and positive"
                )));
            }
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(ConfigError::new(format!(
                    "{name}.sigma must be finite and positive"
                )));
            }
            if min == 0 || max == 0 || min > max {
                return Err(ConfigError::new(format!(
                    "{name}.min and {name}.max must be positive with min <= max"
                )));
            }
            Ok(Some(ServingValueDistribution::LogNormal {
                median,
                sigma,
                min,
                max,
            }))
        }
        value => Err(ConfigError::new(format!(
            "unsupported {name}.kind '{value}'; use uniform, weighted, or lognormal"
        ))),
    }
}

fn parse_shape_profiles(
    name: &str,
    profiles: Option<Vec<ServingShapeProfileSection>>,
) -> Result<Vec<ServingShapeProfile>, ConfigError> {
    let Some(profiles) = profiles else {
        return Ok(Vec::new());
    };
    if profiles.is_empty() {
        return Err(ConfigError::new(format!("{name} must not be empty")));
    }

    let mut names = HashSet::new();
    profiles
        .into_iter()
        .enumerate()
        .map(|(idx, profile)| {
            let profile_name = format!("{name}[{idx}]");
            let parsed_name = optional_nonempty_string(profile.name.clone())
                .unwrap_or_else(|| format!("profile-{idx}"));
            if !names.insert(parsed_name.clone()) {
                return Err(ConfigError::new(format!(
                    "{profile_name}.name '{parsed_name}' is duplicated"
                )));
            }
            if profile.batch_size == 0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.batch_size must be greater than zero"
                )));
            }
            if profile.prompt_tokens == 0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.prompt_tokens must be greater than zero"
                )));
            }
            if profile.decode_tokens == 0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.decode_tokens must be greater than zero"
                )));
            }
            validate_optional_max_sequence_tokens(
                &format!("{profile_name}.max_sequence_tokens"),
                profile.max_sequence_tokens,
                profile.prompt_tokens,
                profile.decode_tokens,
            )?;
            let weight = profile.weight.unwrap_or(1.0);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(ConfigError::new(format!(
                    "{profile_name}.weight must be finite and positive"
                )));
            }
            let prefix_cache_hit_rate = parse_cache_hit_rate(
                &format!("{profile_name}.prefix_cache_hit_rate"),
                profile.prefix_cache_hit_rate,
            )?;
            if profile.prefix_cache_hit_tokens.is_some() && prefix_cache_hit_rate.is_some() {
                return Err(ConfigError::new(format!(
                    "{profile_name} cannot set both prefix_cache_hit_tokens and prefix_cache_hit_rate"
                )));
            }
            if let Some(prefix_cache_hit_tokens) = profile.prefix_cache_hit_tokens
                && prefix_cache_hit_tokens > profile.prompt_tokens
            {
                return Err(ConfigError::new(format!(
                    "{profile_name}.prefix_cache_hit_tokens must be less than or equal to prompt_tokens"
                )));
            }
            let slo = parse_shape_profile_slo(&profile_name, &profile)?;
            let request_timeout_s = parse_positive_optional_seconds(
                &format!("{profile_name}.request_timeout_s"),
                profile.request_timeout_s,
                &format!("{profile_name}.request_timeout_ms"),
                profile.request_timeout_ms,
            )?;
            let deadline_after_s = parse_optional_trace_seconds(
                &format!("{profile_name}.deadline_after_s"),
                profile.deadline_after_s,
                &format!("{profile_name}.deadline_after_ms"),
                profile.deadline_after_ms,
            )?;
            if let Some(deadline_after_s) = deadline_after_s
                && deadline_after_s < 0.0
            {
                return Err(ConfigError::new(format!(
                    "{profile_name}.deadline_after must be non-negative"
                )));
            }
            let cancellation_after_s = parse_optional_trace_seconds(
                &format!("{profile_name}.cancel_after_s"),
                profile.cancel_after_s,
                &format!("{profile_name}.cancel_after_ms"),
                profile.cancel_after_ms,
            )?;
            if let Some(cancellation_after_s) = cancellation_after_s
                && cancellation_after_s < 0.0
            {
                return Err(ConfigError::new(format!(
                    "{profile_name}.cancel_after must be non-negative"
                )));
            }
            Ok(ServingShapeProfile {
                name: parsed_name,
                weight,
                tenant: optional_nonempty_string(profile.tenant),
                model_id: optional_nonempty_string(profile.model_id),
                cache_key: optional_nonempty_string(profile.cache_key),
                priority: profile.priority,
                batch_size: profile.batch_size,
                prompt_tokens: profile.prompt_tokens,
                decode_tokens: profile.decode_tokens,
                max_sequence_tokens: profile.max_sequence_tokens,
                prefix_cache_hit_tokens: profile.prefix_cache_hit_tokens,
                prefix_cache_hit_rate,
                slo,
                request_timeout_s,
                deadline_after_s,
                cancellation_after_s,
            })
        })
        .collect()
}

fn parse_shape_profile_slo(
    name: &str,
    profile: &ServingShapeProfileSection,
) -> Result<ServingRequestSlo, ConfigError> {
    Ok(ServingRequestSlo {
        ttft_s: parse_shape_profile_slo_value(
            name,
            "ttft_slo",
            profile.ttft_slo_s,
            profile.ttft_slo_ms,
        )?,
        tpot_s: parse_shape_profile_slo_value(
            name,
            "tpot_slo",
            profile.tpot_slo_s,
            profile.tpot_slo_ms,
        )?,
        itl_s: parse_shape_profile_slo_value(
            name,
            "itl_slo",
            profile.itl_slo_s,
            profile.itl_slo_ms,
        )?,
        e2el_s: parse_shape_profile_slo_value(
            name,
            "e2el_slo",
            profile.e2el_slo_s,
            profile.e2el_slo_ms,
        )?,
    })
}

fn parse_shape_profile_slo_value(
    profile_name: &str,
    name: &str,
    seconds: Option<f64>,
    millis: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    let value = parse_optional_trace_seconds(
        &format!("{profile_name}.{name}_s"),
        seconds,
        &format!("{profile_name}.{name}_ms"),
        millis,
    )?;
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(ConfigError::new(format!(
            "{profile_name}.{name} must be positive"
        )));
    }
    Ok(value)
}

fn parse_routing_policy(routing_policy: Option<&str>) -> Result<ServingRoutingPolicy, ConfigError> {
    match routing_policy.map(normalize).as_deref() {
        None | Some("round_robin") | Some("roundrobin") => Ok(ServingRoutingPolicy::RoundRobin),
        Some("topology_aware")
        | Some("topologyaware")
        | Some("load_aware")
        | Some("topology_load_aware") => Ok(ServingRoutingPolicy::TopologyAware),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.routing_policy '{value}'; use round_robin or topology_aware"
        ))),
    }
}

fn parse_slo_ms(name: &str, value: Option<f64>) -> Result<Option<f64>, ConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(ConfigError::new(format!(
            "{name} must be finite and positive"
        )));
    }
    Ok(Some(value / 1000.0))
}

fn parse_cache_hit_rate(name: &str, value: Option<f64>) -> Result<Option<f64>, ConfigError> {
    parse_fraction(name, value)
}

fn parse_fraction(name: &str, value: Option<f64>) -> Result<Option<f64>, ConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ConfigError::new(format!(
            "{name} must be finite and between 0.0 and 1.0"
        )));
    }
    Ok(Some(value))
}

fn parse_non_negative_f64(name: &str, value: Option<f64>) -> Result<Option<f64>, ConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() || value < 0.0 {
        return Err(ConfigError::new(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(Some(value))
}

fn parse_prefill_batching(
    prefill_batching: Option<&str>,
    max_prefill_batch_tokens: Option<u64>,
    max_prefill_chunk_tokens: Option<u32>,
) -> Result<ServingPrefillBatching, ConfigError> {
    match prefill_batching.map(normalize).as_deref() {
        None | Some("independent") | Some("per_request") => Ok(ServingPrefillBatching::Independent),
        Some("continuous") | Some("continuous_batching") => {
            Ok(ServingPrefillBatching::Continuous {
                max_batch_tokens: max_prefill_batch_tokens,
                chunk_tokens: max_prefill_chunk_tokens,
            })
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.prefill_batching '{value}'; use independent or continuous"
        ))),
    }
}

fn parse_decode_batching(
    decode_batching: Option<&str>,
    max_decode_batch_tokens: Option<u32>,
) -> Result<ServingDecodeBatching, ConfigError> {
    if let Some(0) = max_decode_batch_tokens {
        return Err(ConfigError::new(
            "serving.traffic.max_decode_batch_tokens must be greater than zero",
        ));
    }

    match decode_batching.map(normalize).as_deref() {
        None | Some("independent") | Some("per_request") => Ok(ServingDecodeBatching::Independent),
        Some("continuous") | Some("continuous_batching") => Ok(ServingDecodeBatching::Continuous {
            max_batch_tokens: max_decode_batch_tokens,
        }),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.decode_batching '{value}'; use independent or continuous"
        ))),
    }
}

fn parse_decode_capacity_policy(
    policy: Option<&str>,
) -> Result<ServingDecodeCapacityPolicy, ConfigError> {
    match policy.map(normalize).as_deref() {
        None
        | Some("candidate_reject")
        | Some("candidate")
        | Some("hard")
        | Some("hard_reject")
        | Some("reject_candidate") => Ok(ServingDecodeCapacityPolicy::CandidateReject),
        Some("request_reject")
        | Some("request")
        | Some("admission")
        | Some("admission_reject")
        | Some("reject_request") => Ok(ServingDecodeCapacityPolicy::RequestReject),
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.decode_capacity_policy '{value}'; use candidate_reject or request_reject"
        ))),
    }
}

fn parse_arrival_pattern(
    traffic: &ServingTrafficSection,
) -> Result<ServingArrivalPattern, ConfigError> {
    match traffic.arrival.as_deref().map(normalize).as_deref() {
        None | Some("fixed") | Some("fixed_gap") | Some("constant") => {
            Ok(ServingArrivalPattern::FixedGap)
        }
        Some("poisson") => {
            let rate_per_s = traffic.arrival_rate_per_s.ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.arrival_rate_per_s is required when arrival = 'poisson'",
                )
            })?;
            if !rate_per_s.is_finite() || rate_per_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.arrival_rate_per_s must be finite and positive",
                ));
            }
            Ok(ServingArrivalPattern::Poisson {
                rate_per_s,
                seed: traffic.arrival_seed.unwrap_or(1),
            })
        }
        Some("bursty") | Some("burst") | Some("bursts") => {
            let burst_size = traffic.burst_size.ok_or_else(|| {
                ConfigError::new("serving.traffic.burst_size is required when arrival = 'bursty'")
            })?;
            if burst_size == 0 {
                return Err(ConfigError::new(
                    "serving.traffic.burst_size must be greater than zero",
                ));
            }
            let burst_interval_s = parse_optional_trace_seconds(
                "serving.traffic.burst_interval_s",
                traffic.burst_interval_s,
                "serving.traffic.burst_interval_ms",
                traffic.burst_interval_ms,
            )?
            .ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.burst_interval_s or burst_interval_ms is required when arrival = 'bursty'",
                )
            })?;
            if burst_interval_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.burst_interval_s/burst_interval_ms must be positive",
                ));
            }
            let intra_burst_gap_s = parse_optional_trace_seconds(
                "serving.traffic.burst_arrival_gap_s",
                traffic.burst_arrival_gap_s,
                "serving.traffic.burst_arrival_gap_ms",
                traffic.burst_arrival_gap_ms,
            )?
            .unwrap_or(0.0);
            if intra_burst_gap_s < 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.burst_arrival_gap_s/burst_arrival_gap_ms must be non-negative",
                ));
            }
            let burst_span_s = f64::from(burst_size.saturating_sub(1)) * intra_burst_gap_s;
            if burst_span_s > burst_interval_s {
                return Err(ConfigError::new(
                    "serving.traffic burst_arrival_gap places requests beyond the next burst interval",
                ));
            }
            Ok(ServingArrivalPattern::Bursty {
                burst_size,
                burst_interval_s,
                intra_burst_gap_s,
            })
        }
        Some("diurnal") | Some("daily") | Some("sinusoidal") => {
            let min_rate_per_s = traffic.diurnal_min_rate_per_s.ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.diurnal_min_rate_per_s is required when arrival = 'diurnal'",
                )
            })?;
            if !min_rate_per_s.is_finite() || min_rate_per_s < 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_min_rate_per_s must be finite and non-negative",
                ));
            }

            let max_rate_per_s = traffic.diurnal_max_rate_per_s.ok_or_else(|| {
                ConfigError::new(
                    "serving.traffic.diurnal_max_rate_per_s is required when arrival = 'diurnal'",
                )
            })?;
            if !max_rate_per_s.is_finite() || max_rate_per_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_max_rate_per_s must be finite and positive",
                ));
            }
            if min_rate_per_s > max_rate_per_s {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_min_rate_per_s must be less than or equal to diurnal_max_rate_per_s",
                ));
            }

            let period_s = parse_optional_trace_seconds(
                "serving.traffic.diurnal_period_s",
                traffic.diurnal_period_s,
                "serving.traffic.diurnal_period_ms",
                traffic.diurnal_period_ms,
            )?
            .unwrap_or(86_400.0);
            if period_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic.diurnal_period_s/diurnal_period_ms must be positive",
                ));
            }
            let phase_s = parse_optional_trace_seconds(
                "serving.traffic.diurnal_phase_s",
                traffic.diurnal_phase_s,
                "serving.traffic.diurnal_phase_ms",
                traffic.diurnal_phase_ms,
            )?
            .unwrap_or(0.0);

            Ok(ServingArrivalPattern::Diurnal {
                min_rate_per_s,
                max_rate_per_s,
                period_s,
                phase_s,
                seed: traffic.arrival_seed.unwrap_or(1),
            })
        }
        Some("self_similar") | Some("selfsimilar") | Some("pareto") => {
            let rate_per_s = traffic
                .self_similar_rate_per_s
                .or(traffic.arrival_rate_per_s)
                .ok_or_else(|| {
                    ConfigError::new(
                        "serving.traffic.arrival_rate_per_s or self_similar_rate_per_s is required when arrival = 'self_similar'",
                    )
                })?;
            if !rate_per_s.is_finite() || rate_per_s <= 0.0 {
                return Err(ConfigError::new(
                    "serving.traffic self-similar arrival rate must be finite and positive",
                ));
            }

            let pareto_shape = traffic.self_similar_pareto_shape.unwrap_or(1.4);
            if !pareto_shape.is_finite() || pareto_shape <= 1.0 {
                return Err(ConfigError::new(
                    "serving.traffic.self_similar_pareto_shape must be finite and greater than 1.0",
                ));
            }

            let max_gap_s = parse_optional_trace_seconds(
                "serving.traffic.self_similar_max_gap_s",
                traffic.self_similar_max_gap_s,
                "serving.traffic.self_similar_max_gap_ms",
                traffic.self_similar_max_gap_ms,
            )?;
            if max_gap_s.is_some_and(|gap| gap <= 0.0) {
                return Err(ConfigError::new(
                    "serving.traffic.self_similar_max_gap_s/self_similar_max_gap_ms must be positive",
                ));
            }

            Ok(ServingArrivalPattern::SelfSimilar {
                rate_per_s,
                pareto_shape,
                max_gap_s,
                seed: traffic.arrival_seed.unwrap_or(1),
            })
        }
        Some("trace_derived") | Some("trace_arrivals") | Some("trace_arrival") => {
            Ok(ServingArrivalPattern::TraceDerived)
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported serving.traffic.arrival '{value}'; use fixed_gap, poisson, bursty, diurnal, self_similar, or trace_derived"
        ))),
    }
}

fn positive_values(name: &str, values: Vec<u32>) -> Result<Vec<u32>, ConfigError> {
    if values.contains(&0) {
        return Err(ConfigError::new(format!(
            "{name} values must be greater than zero"
        )));
    }

    Ok(values)
}

fn parse_calibration_policy(
    section: Option<CalibrationPolicySection>,
) -> Result<CalibrationPolicy, ConfigError> {
    let Some(section) = section else {
        return Ok(CalibrationPolicy::default());
    };
    let defaults = CalibrationPolicy::default();
    let min_coverage_score = section.min_coverage_score.or(defaults.min_coverage_score);
    if let Some(score) = min_coverage_score
        && (!score.is_finite() || !(0.0..=1.0).contains(&score))
    {
        return Err(ConfigError::new(
            "calibration_policy.min_coverage_score must be finite and between 0.0 and 1.0",
        ));
    }
    let min_fit_confidence_score = section
        .min_fit_confidence_score
        .or(defaults.min_fit_confidence_score);
    if let Some(score) = min_fit_confidence_score
        && (!score.is_finite() || !(0.0..=1.0).contains(&score))
    {
        return Err(ConfigError::new(
            "calibration_policy.min_fit_confidence_score must be finite and between 0.0 and 1.0",
        ));
    }
    let min_fit_sample_count = section
        .min_fit_sample_count
        .or(defaults.min_fit_sample_count);
    validate_positive_optional_u32(
        "calibration_policy.min_fit_sample_count",
        min_fit_sample_count,
    )?;
    let min_fit_validation_sample_count = section
        .min_fit_validation_sample_count
        .or(defaults.min_fit_validation_sample_count);
    validate_positive_optional_u32(
        "calibration_policy.min_fit_validation_sample_count",
        min_fit_validation_sample_count,
    )?;
    let min_fit_confidence_level = parse_fraction(
        "calibration_policy.min_fit_confidence_level",
        section
            .min_fit_confidence_level
            .or(defaults.min_fit_confidence_level),
    )?;
    let max_fit_relative_uncertainty_pct = parse_non_negative_f64(
        "calibration_policy.max_fit_relative_uncertainty_pct",
        section
            .max_fit_relative_uncertainty_pct
            .or(defaults.max_fit_relative_uncertainty_pct),
    )?;
    let max_fit_absolute_uncertainty_ms = parse_non_negative_f64(
        "calibration_policy.max_fit_absolute_uncertainty_ms",
        section.max_fit_absolute_uncertainty_ms,
    )?;
    let max_fit_absolute_uncertainty_s = parse_non_negative_f64(
        "calibration_policy.max_fit_absolute_uncertainty_s",
        section
            .max_fit_absolute_uncertainty_s
            .or(defaults.max_fit_absolute_uncertainty_s),
    )?;
    if max_fit_absolute_uncertainty_ms.is_some() && max_fit_absolute_uncertainty_s.is_some() {
        return Err(ConfigError::new(
            "calibration_policy must set only one of max_fit_absolute_uncertainty_ms or max_fit_absolute_uncertainty_s",
        ));
    }
    let max_fit_absolute_uncertainty_s = max_fit_absolute_uncertainty_ms
        .map(|ms| ms / 1000.0)
        .or(max_fit_absolute_uncertainty_s);
    let min_serving_phase_coverage_fraction = parse_fraction(
        "calibration_policy.min_serving_phase_coverage_fraction",
        section
            .min_serving_phase_coverage_fraction
            .or(defaults.min_serving_phase_coverage_fraction),
    )?;
    let uncertainty_ranking_weight = section
        .uncertainty_ranking_weight
        .unwrap_or(defaults.uncertainty_ranking_weight);
    if !uncertainty_ranking_weight.is_finite() || uncertainty_ranking_weight < 0.0 {
        return Err(ConfigError::new(
            "calibration_policy.uncertainty_ranking_weight must be finite and nonnegative",
        ));
    }

    Ok(CalibrationPolicy {
        valid_shape: parse_calibration_gate_mode(
            "calibration_policy.valid_shape",
            section.valid_shape.as_deref(),
        )?,
        invalid_shape: parse_calibration_gate_mode(
            "calibration_policy.invalid_shape",
            section.invalid_shape.as_deref(),
        )?,
        coverage: parse_calibration_gate_mode(
            "calibration_policy.coverage",
            section.coverage.as_deref(),
        )?,
        fit_confidence: parse_calibration_gate_mode(
            "calibration_policy.fit_confidence",
            section.fit_confidence.as_deref(),
        )?,
        fit_extrapolation: parse_calibration_gate_mode(
            "calibration_policy.fit_extrapolation",
            section.fit_extrapolation.as_deref(),
        )?,
        fit_partially_bounded: parse_calibration_gate_mode(
            "calibration_policy.fit_partially_bounded",
            section.fit_partially_bounded.as_deref(),
        )?,
        fit_unbounded: parse_calibration_gate_mode(
            "calibration_policy.fit_unbounded",
            section.fit_unbounded.as_deref(),
        )?,
        fit_sample_count: parse_calibration_gate_mode(
            "calibration_policy.fit_sample_count",
            section.fit_sample_count.as_deref(),
        )?,
        fit_validation_sample_count: parse_calibration_gate_mode(
            "calibration_policy.fit_validation_sample_count",
            section.fit_validation_sample_count.as_deref(),
        )?,
        fit_source: parse_calibration_gate_mode(
            "calibration_policy.fit_source",
            section.fit_source.as_deref(),
        )?,
        fit_uncertainty: parse_calibration_gate_mode(
            "calibration_policy.fit_uncertainty",
            section.fit_uncertainty.as_deref(),
        )?,
        profile_source: parse_calibration_gate_mode(
            "calibration_policy.profile_source",
            section.profile_source.as_deref(),
        )?,
        profile_date: parse_calibration_gate_mode(
            "calibration_policy.profile_date",
            section.profile_date.as_deref(),
        )?,
        profile_runtime: parse_calibration_gate_mode(
            "calibration_policy.profile_runtime",
            section.profile_runtime.as_deref(),
        )?,
        min_coverage_score,
        min_fit_confidence_score,
        min_fit_confidence_level,
        min_fit_sample_count,
        min_fit_validation_sample_count,
        max_fit_relative_uncertainty_pct,
        max_fit_absolute_uncertainty_s,
        min_serving_phase_coverage_fraction,
        uncertainty_ranking_weight,
        require_phase_coverage: section
            .require_phase_coverage
            .unwrap_or(defaults.require_phase_coverage),
    })
}

fn parse_calibration_gate_mode(
    name: &str,
    value: Option<&str>,
) -> Result<CalibrationGateMode, ConfigError> {
    match value.map(normalize).as_deref() {
        None | Some("warn") | Some("warning") | Some("soft") | Some("soft_fail") => {
            Ok(CalibrationGateMode::Warn)
        }
        Some("reject") | Some("hard") | Some("hard_fail") | Some("fail") => {
            Ok(CalibrationGateMode::Reject)
        }
        Some(value) => Err(ConfigError::new(format!(
            "unsupported {name} '{value}'; use warn or reject"
        ))),
    }
}

fn parse_approximation_policy(
    section: Option<ApproximationPolicySection>,
) -> Result<ApproximationPolicy, ConfigError> {
    let Some(section) = section else {
        return Ok(ApproximationPolicy::default());
    };
    let preset = section
        .preset
        .as_deref()
        .map(parse_approximation_policy_preset)
        .transpose()?;
    let defaults = preset
        .map(approximation_policy_preset_defaults)
        .unwrap_or_default();
    Ok(ApproximationPolicy {
        preset,
        default_action: match section.default_action.as_deref() {
            Some(value) => {
                parse_calibration_gate_mode("approximation_policy.default_action", Some(value))?
            }
            None => defaults.default_action,
        },
        reject_categories: match section.reject_categories {
            Some(values) => {
                normalized_policy_terms("approximation_policy.reject_categories", values)?
            }
            None => defaults.reject_categories,
        },
        reject_codes: match section.reject_codes {
            Some(values) => normalized_policy_terms("approximation_policy.reject_codes", values)?,
            None => defaults.reject_codes,
        },
        warn_categories: match section.warn_categories {
            Some(values) => {
                normalized_policy_terms("approximation_policy.warn_categories", values)?
            }
            None => defaults.warn_categories,
        },
        warn_codes: match section.warn_codes {
            Some(values) => normalized_policy_terms("approximation_policy.warn_codes", values)?,
            None => defaults.warn_codes,
        },
        metric_gates: if section.metric_gates.is_empty() {
            defaults.metric_gates
        } else {
            parse_approximation_metric_gates(section.metric_gates)?
        },
    })
}

fn parse_approximation_metric_gates(
    gates: Vec<ApproximationMetricGateSection>,
) -> Result<Vec<ApproximationMetricGate>, ConfigError> {
    let mut parsed = Vec::new();
    for (idx, gate) in gates.into_iter().enumerate() {
        let mut metrics = Vec::new();
        if let Some(metric) = gate.metric {
            metrics.push(metric);
        }
        metrics.extend(gate.metrics.unwrap_or_default());
        if let Some(objective) = gate.objective {
            metrics.push(objective);
        }
        metrics.extend(gate.objectives.unwrap_or_default());
        let metrics = normalized_approximation_metric_terms(
            &format!("approximation_policy.metric_gates[{idx}].metrics"),
            metrics,
        )?;
        if metrics.is_empty() {
            return Err(ConfigError::new(format!(
                "approximation_policy.metric_gates[{idx}] requires metric, metrics, objective, or objectives"
            )));
        }
        let reject_categories = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].reject_categories"),
            gate.reject_categories.unwrap_or_default(),
        )?;
        let reject_codes = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].reject_codes"),
            gate.reject_codes.unwrap_or_default(),
        )?;
        let warn_categories = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].warn_categories"),
            gate.warn_categories.unwrap_or_default(),
        )?;
        let warn_codes = normalized_policy_terms(
            &format!("approximation_policy.metric_gates[{idx}].warn_codes"),
            gate.warn_codes.unwrap_or_default(),
        )?;
        if reject_categories.is_empty()
            && reject_codes.is_empty()
            && warn_categories.is_empty()
            && warn_codes.is_empty()
        {
            return Err(ConfigError::new(format!(
                "approximation_policy.metric_gates[{idx}] must set at least one reject_categories, reject_codes, warn_categories, or warn_codes entry"
            )));
        }
        parsed.push(ApproximationMetricGate {
            metrics,
            reject_categories,
            reject_codes,
            warn_categories,
            warn_codes,
        });
    }
    Ok(parsed)
}

fn normalized_approximation_metric_terms(
    name: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ConfigError> {
    let mut terms = Vec::new();
    for value in values {
        let term = approximation_metric_term(&value).ok_or_else(|| {
            ConfigError::new(format!(
                "unsupported {name} value '{value}'; use ttft, tpot, throughput, e2el, slo_miss_rate, memory_pressure, cost, energy, or power"
            ))
        })?;
        if !terms.contains(&term) {
            terms.push(term);
        }
    }
    Ok(terms)
}

fn approximation_metric_term(value: &str) -> Option<String> {
    let term = normalize(value);
    match term.as_str() {
        "ttft" | "time_to_first_token" | "minimize_ttft" => Some("ttft".to_string()),
        "tpot" | "time_per_output_token" | "minimize_tpot" => Some("tpot".to_string()),
        "throughput"
        | "tokens_per_s"
        | "tokens_per_second"
        | "max_throughput"
        | "maximize_throughput" => Some("throughput".to_string()),
        "e2el" | "latency" | "end_to_end" | "end_to_end_latency" | "minimize_e2el" => {
            Some("e2el".to_string())
        }
        "slo" | "slo_miss" | "slo_miss_rate" | "minimize_slo_miss_rate" => {
            Some("slo_miss_rate".to_string())
        }
        "memory"
        | "hbm"
        | "memory_pressure"
        | "hbm_pressure"
        | "minimize_memory_pressure"
        | "minimize_hbm_pressure" => Some("memory_pressure".to_string()),
        "cost"
        | "total_cost"
        | "cost_usd"
        | "minimize_cost"
        | "minimize_total_cost"
        | "minimize_cost_usd" => Some("cost".to_string()),
        "energy" | "kwh" | "energy_kwh" | "minimize_energy" | "minimize_energy_kwh" => {
            Some("energy".to_string())
        }
        "power"
        | "watts"
        | "average_power"
        | "average_power_watts"
        | "minimize_power"
        | "minimize_average_power"
        | "minimize_average_power_watts" => Some("power".to_string()),
        _ => None,
    }
}

fn parse_approximation_policy_preset(
    value: &str,
) -> Result<ApproximationPolicyPreset, ConfigError> {
    match normalize(value).as_str() {
        "mvp" | "exploration" | "mvp_exploration" => Ok(ApproximationPolicyPreset::MvpExploration),
        "topology" | "topology_sensitive" | "topology_sensitive_planning" => {
            Ok(ApproximationPolicyPreset::TopologySensitive)
        }
        "memory" | "capacity" | "memory_capacity" | "memory_capacity_planning" => {
            Ok(ApproximationPolicyPreset::MemoryCapacity)
        }
        "calibration" | "calibrated" | "calibration_only" | "calibrated_only" => {
            Ok(ApproximationPolicyPreset::CalibrationOnly)
        }
        "production" | "strict" | "production_recommendation" => {
            Ok(ApproximationPolicyPreset::ProductionRecommendation)
        }
        value => Err(ConfigError::new(format!(
            "unsupported approximation_policy.preset '{value}'; use mvp_exploration, topology_sensitive, memory_capacity, calibration_only, or production_recommendation"
        ))),
    }
}

fn approximation_policy_preset_defaults(preset: ApproximationPolicyPreset) -> ApproximationPolicy {
    match preset {
        ApproximationPolicyPreset::MvpExploration => ApproximationPolicy {
            preset: Some(preset),
            ..ApproximationPolicy::default()
        },
        ApproximationPolicyPreset::TopologySensitive => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Warn,
            reject_categories: vec![
                "topology".to_string(),
                "routing".to_string(),
                "communication".to_string(),
            ],
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
        ApproximationPolicyPreset::MemoryCapacity => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Warn,
            reject_categories: vec!["memory".to_string(), "capacity".to_string()],
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
        ApproximationPolicyPreset::CalibrationOnly => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Warn,
            reject_categories: vec!["calibration".to_string(), "runtime".to_string()],
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
        ApproximationPolicyPreset::ProductionRecommendation => ApproximationPolicy {
            preset: Some(preset),
            default_action: CalibrationGateMode::Reject,
            reject_categories: Vec::new(),
            reject_codes: Vec::new(),
            warn_categories: Vec::new(),
            warn_codes: Vec::new(),
            metric_gates: Vec::new(),
        },
    }
}

fn normalized_policy_terms(name: &str, values: Vec<String>) -> Result<Vec<String>, ConfigError> {
    let mut terms = Vec::new();
    for value in values {
        let term = normalize(&value);
        if term.is_empty() {
            return Err(ConfigError::new(format!("{name} values must not be empty")));
        }
        if !terms.contains(&term) {
            terms.push(term);
        }
    }
    Ok(terms)
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedCalibrationProfile {
    pub metadata: CalibrationProfileMetadata,
    pub calibration: SimulationCalibration,
}

fn load_calibration_profile(
    reference: Option<CalibrationProfileReferenceSection>,
    base_dir: Option<&Path>,
) -> Result<Option<LoadedCalibrationProfile>, ConfigError> {
    let Some(reference) = reference else {
        return Ok(None);
    };
    if reference.path.trim().is_empty() {
        return Err(ConfigError::new(
            "calibration_profile.path must not be empty",
        ));
    }
    let path = resolve_config_path(&reference.path, base_dir);
    Ok(Some(load_calibration_profile_path(&path)?))
}

pub fn load_calibration_profile_path(path: &Path) -> Result<LoadedCalibrationProfile, ConfigError> {
    let contents = fs::read_to_string(path).map_err(|err| {
        ConfigError::new(format!(
            "failed to read calibration_profile.path {}: {err}",
            path.display()
        ))
    })?;
    let profile_file: CalibrationProfileFile = toml::from_str(&contents).map_err(|err| {
        ConfigError::new(format!(
            "invalid calibration profile TOML {}: {err}",
            path.display()
        ))
    })?;
    validate_schema_version("calibration profile", profile_file.schema_version)?;
    let metadata = calibration_profile_metadata(
        path.to_path_buf(),
        profile_file.profile,
        profile_file.valid_shape,
        profile_file.invalid_shapes.unwrap_or_default(),
        profile_file.fits.unwrap_or_default(),
        profile_file.benchmarks.unwrap_or_default(),
    )?;
    let calibration =
        calibration_with_defaults(profile_file.calibration, SimulationCalibration::default());
    Ok(LoadedCalibrationProfile {
        metadata,
        calibration,
    })
}

fn calibration_profile_metadata(
    path: PathBuf,
    section: Option<CalibrationProfileMetadataSection>,
    valid_shape: Option<CalibrationShapeRangeSection>,
    invalid_shapes: Vec<CalibrationInvalidShapeRangeSection>,
    fits: Vec<CalibrationFittedModelSection>,
    benchmarks: Vec<CalibrationBenchmarkPointSection>,
) -> Result<CalibrationProfileMetadata, ConfigError> {
    let section = section.unwrap_or(CalibrationProfileMetadataSection {
        name: None,
        hardware: None,
        fabric: None,
        model: None,
        dtype: None,
        serving_stack: None,
        serving_runtime_features: None,
        backend_version: None,
        driver_version: None,
        cuda_version: None,
        rocm_version: None,
        nccl_version: None,
        rccl_version: None,
        ucx_version: None,
        kernel_settings: Vec::new(),
        environment_hash: None,
        source: None,
        date: None,
        notes: None,
    });
    Ok(CalibrationProfileMetadata {
        path: path.display().to_string(),
        name: nonempty_metadata(section.name),
        hardware: nonempty_metadata(section.hardware),
        fabric: nonempty_metadata(section.fabric),
        model: nonempty_metadata(section.model),
        dtype: nonempty_metadata(section.dtype),
        serving_stack: nonempty_metadata(section.serving_stack),
        serving_runtime_features: section
            .serving_runtime_features
            .map(|features| normalized_group_labels("profile.serving_runtime_features", features))
            .transpose()?
            .unwrap_or_default(),
        backend_version: nonempty_metadata(section.backend_version),
        driver_version: nonempty_metadata(section.driver_version),
        cuda_version: nonempty_metadata(section.cuda_version),
        rocm_version: nonempty_metadata(section.rocm_version),
        nccl_version: nonempty_metadata(section.nccl_version),
        rccl_version: nonempty_metadata(section.rccl_version),
        ucx_version: nonempty_metadata(section.ucx_version),
        kernel_settings: nonempty_string_values(
            "profile.kernel_settings",
            section.kernel_settings,
        )?,
        environment_hash: nonempty_metadata(section.environment_hash),
        source: nonempty_metadata(section.source),
        date: nonempty_metadata(section.date),
        notes: nonempty_metadata(section.notes),
        valid_shape: parse_calibration_shape_range(valid_shape)?,
        invalid_shapes: parse_calibration_invalid_shape_ranges(invalid_shapes)?,
        fits: parse_calibration_fits(fits)?,
        benchmarks: parse_calibration_benchmarks(benchmarks)?,
    })
}

fn nonempty_metadata(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn parse_calibration_shape_range(
    section: Option<CalibrationShapeRangeSection>,
) -> Result<Option<CalibrationShapeRange>, ConfigError> {
    let Some(section) = section else {
        return Ok(None);
    };
    validate_min_max(
        "valid_shape.batch_size",
        section.min_batch_size,
        section.max_batch_size,
    )?;
    validate_min_max(
        "valid_shape.prompt_tokens",
        section.min_prompt_tokens,
        section.max_prompt_tokens,
    )?;
    validate_min_max(
        "valid_shape.decode_tokens",
        section.min_decode_tokens,
        section.max_decode_tokens,
    )?;
    validate_min_max(
        "valid_shape.sequence_tokens",
        section.min_sequence_tokens,
        section.max_sequence_tokens,
    )?;

    Ok(Some(CalibrationShapeRange {
        min_batch_size: section.min_batch_size,
        max_batch_size: section.max_batch_size,
        min_prompt_tokens: section.min_prompt_tokens,
        max_prompt_tokens: section.max_prompt_tokens,
        min_decode_tokens: section.min_decode_tokens,
        max_decode_tokens: section.max_decode_tokens,
        min_sequence_tokens: section.min_sequence_tokens,
        max_sequence_tokens: section.max_sequence_tokens,
    }))
}

fn parse_calibration_invalid_shape_ranges(
    invalid_shapes: Vec<CalibrationInvalidShapeRangeSection>,
) -> Result<Vec<CalibrationInvalidShapeRange>, ConfigError> {
    invalid_shapes
        .into_iter()
        .enumerate()
        .map(|(idx, section)| parse_calibration_invalid_shape_range(idx, section))
        .collect()
}

fn parse_calibration_invalid_shape_range(
    idx: usize,
    section: CalibrationInvalidShapeRangeSection,
) -> Result<CalibrationInvalidShapeRange, ConfigError> {
    validate_min_max(
        &format!("invalid_shapes[{idx}].batch_size"),
        section.min_batch_size,
        section.max_batch_size,
    )?;
    validate_min_max(
        &format!("invalid_shapes[{idx}].prompt_tokens"),
        section.min_prompt_tokens,
        section.max_prompt_tokens,
    )?;
    validate_min_max(
        &format!("invalid_shapes[{idx}].decode_tokens"),
        section.min_decode_tokens,
        section.max_decode_tokens,
    )?;
    validate_min_max(
        &format!("invalid_shapes[{idx}].sequence_tokens"),
        section.min_sequence_tokens,
        section.max_sequence_tokens,
    )?;
    let shape = CalibrationShapeRange {
        min_batch_size: section.min_batch_size,
        max_batch_size: section.max_batch_size,
        min_prompt_tokens: section.min_prompt_tokens,
        max_prompt_tokens: section.max_prompt_tokens,
        min_decode_tokens: section.min_decode_tokens,
        max_decode_tokens: section.max_decode_tokens,
        min_sequence_tokens: section.min_sequence_tokens,
        max_sequence_tokens: section.max_sequence_tokens,
    };
    if !shape_range_has_any_bound(&shape) {
        return Err(ConfigError::new(format!(
            "invalid_shapes[{idx}] must set at least one shape bound"
        )));
    }

    Ok(CalibrationInvalidShapeRange {
        name: nonempty_metadata(section.name),
        reason: nonempty_metadata(section.reason),
        shape,
    })
}

fn shape_range_has_any_bound(shape: &CalibrationShapeRange) -> bool {
    shape.min_batch_size.is_some()
        || shape.max_batch_size.is_some()
        || shape.min_prompt_tokens.is_some()
        || shape.max_prompt_tokens.is_some()
        || shape.min_decode_tokens.is_some()
        || shape.max_decode_tokens.is_some()
        || shape.min_sequence_tokens.is_some()
        || shape.max_sequence_tokens.is_some()
}

fn parse_calibration_fits(
    fits: Vec<CalibrationFittedModelSection>,
) -> Result<Vec<CalibrationFittedModel>, ConfigError> {
    fits.into_iter()
        .enumerate()
        .map(|(idx, fit)| parse_calibration_fit(idx, fit))
        .collect()
}

fn parse_calibration_fit(
    idx: usize,
    fit: CalibrationFittedModelSection,
) -> Result<CalibrationFittedModel, ConfigError> {
    let target = required_nonempty_metadata(&format!("fits[{idx}].target"), fit.target)?;
    let model = required_nonempty_metadata(&format!("fits[{idx}].model"), fit.model)?;
    let features = nonempty_string_values(&format!("fits[{idx}].features"), fit.features)?;
    if features.is_empty() {
        return Err(ConfigError::new(format!(
            "fits[{idx}].features must not be empty"
        )));
    }
    if fit.coefficients.is_empty() {
        return Err(ConfigError::new(format!(
            "fits[{idx}].coefficients must not be empty"
        )));
    }
    if features.len() != fit.coefficients.len() {
        return Err(ConfigError::new(format!(
            "fits[{idx}].features and coefficients must have the same length"
        )));
    }
    let feature_ranges = parse_calibration_fit_feature_ranges(idx, &features, fit.feature_ranges)?;
    validate_optional_finite_f64(&format!("fits[{idx}].intercept"), fit.intercept)?;
    for (coefficient_idx, coefficient) in fit.coefficients.iter().enumerate() {
        validate_finite_f64(
            &format!("fits[{idx}].coefficients[{coefficient_idx}]"),
            *coefficient,
        )?;
    }
    validate_optional_finite_f64(&format!("fits[{idx}].r_squared"), fit.r_squared)?;
    validate_optional_finite_f64(
        &format!("fits[{idx}].adjusted_r_squared"),
        fit.adjusted_r_squared,
    )?;
    validate_nonnegative_optional_f64(&format!("fits[{idx}].rmse"), fit.rmse)?;
    validate_nonnegative_optional_f64(&format!("fits[{idx}].rmse_pct"), fit.rmse_pct)?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].mean_abs_pct_error"),
        fit.mean_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].max_abs_pct_error"),
        fit.max_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_rmse"),
        fit.validation_rmse,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_rmse_pct"),
        fit.validation_rmse_pct,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_mean_abs_pct_error"),
        fit.validation_mean_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].validation_max_abs_pct_error"),
        fit.validation_max_abs_pct_error,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].confidence_interval"),
        fit.confidence_interval,
    )?;
    validate_nonnegative_optional_f64(
        &format!("fits[{idx}].confidence_interval_pct"),
        fit.confidence_interval_pct,
    )?;
    validate_positive_fraction_optional_f64(
        &format!("fits[{idx}].confidence_level"),
        fit.confidence_level,
    )?;
    validate_positive_optional_u32(&format!("fits[{idx}].sample_count"), fit.sample_count)?;
    validate_positive_optional_u32(
        &format!("fits[{idx}].validation_sample_count"),
        fit.validation_sample_count,
    )?;

    Ok(CalibrationFittedModel {
        name: nonempty_metadata(fit.name),
        target,
        phase: nonempty_metadata(fit.phase).map(|phase| normalize(&phase)),
        kind: nonempty_metadata(fit.kind).map(|kind| normalize(&kind)),
        model,
        unit: nonempty_metadata(fit.unit),
        intercept: fit.intercept,
        features,
        coefficients: fit.coefficients,
        feature_ranges,
        r_squared: fit.r_squared,
        adjusted_r_squared: fit.adjusted_r_squared,
        rmse: fit.rmse,
        rmse_pct: fit.rmse_pct,
        mean_abs_pct_error: fit.mean_abs_pct_error,
        max_abs_pct_error: fit.max_abs_pct_error,
        validation_rmse: fit.validation_rmse,
        validation_rmse_pct: fit.validation_rmse_pct,
        validation_mean_abs_pct_error: fit.validation_mean_abs_pct_error,
        validation_max_abs_pct_error: fit.validation_max_abs_pct_error,
        confidence_interval: fit.confidence_interval,
        confidence_interval_pct: fit.confidence_interval_pct,
        confidence_level: fit.confidence_level,
        sample_count: fit.sample_count,
        validation_sample_count: fit.validation_sample_count,
        source: nonempty_metadata(fit.source),
        notes: nonempty_metadata(fit.notes),
    })
}

fn parse_calibration_fit_feature_ranges(
    fit_idx: usize,
    features: &[String],
    ranges: Vec<CalibrationFitFeatureRangeSection>,
) -> Result<Vec<CalibrationFitFeatureRange>, ConfigError> {
    let feature_names = features
        .iter()
        .map(|feature| normalize(feature))
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    ranges
        .into_iter()
        .enumerate()
        .map(|(range_idx, range)| {
            let name = format!("fits[{fit_idx}].feature_ranges[{range_idx}]");
            let feature = required_nonempty_metadata(&format!("{name}.feature"), range.feature)?;
            let normalized_feature = normalize(&feature);
            if !feature_names.contains(&normalized_feature) {
                return Err(ConfigError::new(format!(
                    "{name}.feature must match one of fits[{fit_idx}].features"
                )));
            }
            if !seen.insert(normalized_feature) {
                return Err(ConfigError::new(format!(
                    "{name}.feature duplicates an earlier feature range"
                )));
            }
            validate_optional_finite_f64(&format!("{name}.min"), range.min)?;
            validate_optional_finite_f64(&format!("{name}.max"), range.max)?;
            if let (Some(min), Some(max)) = (range.min, range.max)
                && min > max
            {
                return Err(ConfigError::new(format!("{name}.min must be <= max")));
            }
            if range.min.is_none() && range.max.is_none() {
                return Err(ConfigError::new(format!(
                    "{name} must set at least one range bound"
                )));
            }

            Ok(CalibrationFitFeatureRange {
                feature,
                min: range.min,
                max: range.max,
            })
        })
        .collect()
}

fn required_nonempty_metadata(name: &str, value: Option<String>) -> Result<String, ConfigError> {
    nonempty_metadata(value).ok_or_else(|| ConfigError::new(format!("{name} must not be empty")))
}

fn nonempty_string_values(name: &str, values: Vec<String>) -> Result<Vec<String>, ConfigError> {
    let mut parsed = Vec::new();
    for value in values {
        let value = value.trim().to_string();
        if value.is_empty() {
            return Err(ConfigError::new(format!("{name} values must not be empty")));
        }
        parsed.push(value);
    }
    Ok(parsed)
}

fn parse_calibration_benchmarks(
    benchmarks: Vec<CalibrationBenchmarkPointSection>,
) -> Result<Vec<CalibrationBenchmarkPoint>, ConfigError> {
    benchmarks
        .into_iter()
        .enumerate()
        .map(|(idx, benchmark)| parse_calibration_benchmark(idx, benchmark))
        .collect()
}

fn parse_calibration_benchmark(
    idx: usize,
    benchmark: CalibrationBenchmarkPointSection,
) -> Result<CalibrationBenchmarkPoint, ConfigError> {
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].batch_size"),
        benchmark.batch_size,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].prompt_tokens"),
        benchmark.prompt_tokens,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].decode_tokens"),
        benchmark.decode_tokens,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].sequence_tokens"),
        benchmark.sequence_tokens,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].tensor_ranks"),
        benchmark.tensor_ranks,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].pipeline_ranks"),
        benchmark.pipeline_ranks,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].expert_ranks"),
        benchmark.expert_ranks,
    )?;
    validate_positive_optional_u32(
        &format!("benchmarks[{idx}].data_ranks"),
        benchmark.data_ranks,
    )?;
    validate_positive_optional_f64(
        &format!("benchmarks[{idx}].measured_ms"),
        benchmark.measured_ms,
    )?;
    validate_positive_optional_f64(
        &format!("benchmarks[{idx}].predicted_ms"),
        benchmark.predicted_ms,
    )?;
    validate_positive_optional_f64(
        &format!("benchmarks[{idx}].throughput_tokens_per_s"),
        benchmark.throughput_tokens_per_s,
    )?;

    Ok(CalibrationBenchmarkPoint {
        name: nonempty_metadata(benchmark.name),
        kind: nonempty_metadata(benchmark.kind),
        phase: nonempty_metadata(benchmark.phase),
        hardware: nonempty_metadata(benchmark.hardware),
        fabric: nonempty_metadata(benchmark.fabric),
        model: nonempty_metadata(benchmark.model),
        dtype: nonempty_metadata(benchmark.dtype),
        batch_size: benchmark.batch_size,
        prompt_tokens: benchmark.prompt_tokens,
        decode_tokens: benchmark.decode_tokens,
        sequence_tokens: benchmark.sequence_tokens,
        tensor_ranks: benchmark.tensor_ranks,
        pipeline_ranks: benchmark.pipeline_ranks,
        expert_ranks: benchmark.expert_ranks,
        data_ranks: benchmark.data_ranks,
        measured_ms: benchmark.measured_ms,
        predicted_ms: benchmark.predicted_ms,
        throughput_tokens_per_s: benchmark.throughput_tokens_per_s,
        command: nonempty_metadata(benchmark.command),
        source: nonempty_metadata(benchmark.source),
        notes: nonempty_metadata(benchmark.notes),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CalibrationShapeEnvelope {
    min_batch_size: u32,
    max_batch_size: u32,
    min_prompt_tokens: u32,
    max_prompt_tokens: u32,
    min_decode_tokens: u32,
    max_decode_tokens: u32,
    min_sequence_tokens: u32,
    max_sequence_tokens: u32,
}

impl CalibrationShapeEnvelope {
    fn from_request(request: &InferenceRequest) -> Self {
        let sequence_tokens = request
            .max_sequence_tokens
            .max(request.prompt_tokens.saturating_add(request.decode_tokens))
            .max(1);
        Self {
            min_batch_size: request.batch_size.max(1),
            max_batch_size: request.batch_size.max(1),
            min_prompt_tokens: request.prompt_tokens.max(1),
            max_prompt_tokens: request.prompt_tokens.max(1),
            min_decode_tokens: request.decode_tokens.max(1),
            max_decode_tokens: request.decode_tokens.max(1),
            min_sequence_tokens: sequence_tokens,
            max_sequence_tokens: sequence_tokens,
        }
    }

    fn include_serving(&mut self, serving: &DisaggregatedServingConfig, base: &InferenceRequest) {
        let traffic = &serving.traffic;
        for request in &traffic.trace_requests {
            self.include_trace_request(request, base);
        }
        self.include_values(ShapeField::BatchSize, &traffic.batch_sizes);
        self.include_values(ShapeField::PromptTokens, &traffic.prompt_tokens);
        self.include_values(ShapeField::DecodeTokens, &traffic.decode_tokens);
        self.include_distribution(
            ShapeField::BatchSize,
            traffic.batch_size_distribution.as_ref(),
        );
        self.include_distribution(
            ShapeField::PromptTokens,
            traffic.prompt_tokens_distribution.as_ref(),
        );
        self.include_distribution(
            ShapeField::DecodeTokens,
            traffic.decode_tokens_distribution.as_ref(),
        );
        for profile in &traffic.shape_profiles {
            let sequence_tokens = profile
                .max_sequence_tokens
                .unwrap_or(base.max_sequence_tokens)
                .max(profile.prompt_tokens.saturating_add(profile.decode_tokens))
                .max(1);
            self.include_request_shape(
                profile.batch_size,
                profile.prompt_tokens,
                profile.decode_tokens,
                sequence_tokens,
            );
        }
    }

    fn include_trace_request(&mut self, request: &ServingTraceRequest, base: &InferenceRequest) {
        let sequence_tokens = request
            .max_sequence_tokens
            .unwrap_or(base.max_sequence_tokens)
            .max(request.prompt_tokens.saturating_add(request.decode_tokens))
            .max(1);
        self.include_request_shape(
            request.batch_size,
            request.prompt_tokens,
            request.decode_tokens,
            sequence_tokens,
        );
    }

    fn include_request_shape(
        &mut self,
        batch_size: u32,
        prompt_tokens: u32,
        decode_tokens: u32,
        sequence_tokens: u32,
    ) {
        self.include_value(ShapeField::BatchSize, batch_size);
        self.include_value(ShapeField::PromptTokens, prompt_tokens);
        self.include_value(ShapeField::DecodeTokens, decode_tokens);
        self.include_value(ShapeField::SequenceTokens, sequence_tokens);
    }

    fn include_values(&mut self, field: ShapeField, values: &[u32]) {
        for value in values {
            self.include_value(field, *value);
        }
    }

    fn include_distribution(
        &mut self,
        field: ShapeField,
        distribution: Option<&ServingValueDistribution>,
    ) {
        let Some((min, max)) = distribution.and_then(serving_distribution_range) else {
            return;
        };
        self.include_value(field, min);
        self.include_value(field, max);
    }

    fn include_value(&mut self, field: ShapeField, value: u32) {
        let value = value.max(1);
        match field {
            ShapeField::BatchSize => {
                self.min_batch_size = self.min_batch_size.min(value);
                self.max_batch_size = self.max_batch_size.max(value);
            }
            ShapeField::PromptTokens => {
                self.min_prompt_tokens = self.min_prompt_tokens.min(value);
                self.max_prompt_tokens = self.max_prompt_tokens.max(value);
            }
            ShapeField::DecodeTokens => {
                self.min_decode_tokens = self.min_decode_tokens.min(value);
                self.max_decode_tokens = self.max_decode_tokens.max(value);
            }
            ShapeField::SequenceTokens => {
                self.min_sequence_tokens = self.min_sequence_tokens.min(value);
                self.max_sequence_tokens = self.max_sequence_tokens.max(value);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapeField {
    BatchSize,
    PromptTokens,
    DecodeTokens,
    SequenceTokens,
}

fn serving_distribution_range(distribution: &ServingValueDistribution) -> Option<(u32, u32)> {
    match distribution {
        ServingValueDistribution::Uniform { min, max }
        | ServingValueDistribution::LogNormal { min, max, .. } => Some((*min, *max)),
        ServingValueDistribution::Weighted { values, .. } => {
            let min = values.iter().min().copied()?;
            let max = values.iter().max().copied()?;
            Some((min, max))
        }
    }
}

fn workload_shape_envelope(
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> CalibrationShapeEnvelope {
    let mut envelope = CalibrationShapeEnvelope::from_request(request);
    if let Some(serving) = serving {
        envelope.include_serving(serving, request);
    }
    envelope
}

fn calibration_coverage_report(
    profile: Option<&CalibrationProfileMetadata>,
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Option<CalibrationCoverageReport> {
    let profile = profile?;
    let envelope = workload_shape_envelope(request, serving);
    let required_phases = required_calibration_phases(request, serving);
    let covered_phases = covered_calibration_phases(&profile.benchmarks, &required_phases);
    let missing_phases: Vec<_> = required_phases
        .iter()
        .filter(|phase| !covered_phases.contains(*phase))
        .cloned()
        .collect();
    let phase_coverage_score = if required_phases.is_empty() {
        None
    } else {
        Some(covered_phases.len() as f64 / required_phases.len() as f64)
    };

    let shape_benchmark_count = profile
        .benchmarks
        .iter()
        .filter(|benchmark| benchmark_has_any_shape(benchmark))
        .count();
    let complete_shape_benchmark_count = profile
        .benchmarks
        .iter()
        .filter(|benchmark| benchmark_has_complete_shape(benchmark))
        .count();
    let batch_size_score = benchmark_field_coverage_score(
        envelope.min_batch_size,
        envelope.max_batch_size,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.batch_size),
    );
    let prompt_tokens_score = benchmark_field_coverage_score(
        envelope.min_prompt_tokens,
        envelope.max_prompt_tokens,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.prompt_tokens),
    );
    let decode_tokens_score = benchmark_field_coverage_score(
        envelope.min_decode_tokens,
        envelope.max_decode_tokens,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.decode_tokens),
    );
    let sequence_tokens_score = benchmark_field_coverage_score(
        envelope.min_sequence_tokens,
        envelope.max_sequence_tokens,
        benchmark_field_range(&profile.benchmarks, |benchmark| benchmark.sequence_tokens),
    );
    let shape_coverage_score = optional_mean_f64(
        [
            batch_size_score,
            prompt_tokens_score,
            decode_tokens_score,
            sequence_tokens_score,
        ]
        .into_iter()
        .flatten(),
    );
    let coverage_score = optional_mean_f64(
        [shape_coverage_score, phase_coverage_score]
            .into_iter()
            .flatten(),
    );
    let (nearest_benchmark, nearest_benchmark_distance) =
        nearest_shape_benchmark(&profile.benchmarks, envelope);
    let status = calibration_coverage_status(
        profile.benchmarks.len(),
        shape_benchmark_count,
        coverage_score,
        &missing_phases,
    )
    .to_string();

    Some(CalibrationCoverageReport {
        benchmark_count: profile.benchmarks.len(),
        shape_benchmark_count,
        complete_shape_benchmark_count,
        required_phases,
        covered_phases,
        missing_phases,
        batch_size_score,
        prompt_tokens_score,
        decode_tokens_score,
        sequence_tokens_score,
        shape_coverage_score,
        phase_coverage_score,
        coverage_score,
        nearest_benchmark,
        nearest_benchmark_distance,
        status,
    })
}

fn required_calibration_phases(
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Vec<String> {
    if serving.is_some() {
        return vec![
            "prefill".to_string(),
            "decode".to_string(),
            "kv_transfer".to_string(),
        ];
    }

    match request.phase {
        InferencePhase::Prefill => vec!["prefill".to_string()],
        InferencePhase::Decode => vec!["decode".to_string()],
        InferencePhase::EndToEnd => vec!["prefill".to_string(), "decode".to_string()],
    }
}

fn covered_calibration_phases(
    benchmarks: &[CalibrationBenchmarkPoint],
    required_phases: &[String],
) -> Vec<String> {
    let observed: HashSet<_> = benchmarks
        .iter()
        .filter_map(|benchmark| benchmark.phase.as_deref())
        .map(normalize)
        .collect();
    required_phases
        .iter()
        .filter(|phase| observed.contains(*phase))
        .cloned()
        .collect()
}

fn benchmark_has_any_shape(benchmark: &CalibrationBenchmarkPoint) -> bool {
    benchmark.batch_size.is_some()
        || benchmark.prompt_tokens.is_some()
        || benchmark.decode_tokens.is_some()
        || benchmark.sequence_tokens.is_some()
}

fn benchmark_has_complete_shape(benchmark: &CalibrationBenchmarkPoint) -> bool {
    benchmark.batch_size.is_some()
        && benchmark.prompt_tokens.is_some()
        && benchmark.decode_tokens.is_some()
        && benchmark.sequence_tokens.is_some()
}

fn benchmark_field_range(
    benchmarks: &[CalibrationBenchmarkPoint],
    selector: impl Fn(&CalibrationBenchmarkPoint) -> Option<u32>,
) -> Option<(u32, u32)> {
    let mut values = benchmarks.iter().filter_map(selector);
    let first = values.next()?;
    let mut min = first;
    let mut max = first;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    Some((min, max))
}

fn benchmark_field_coverage_score(
    observed_min: u32,
    observed_max: u32,
    benchmark_range: Option<(u32, u32)>,
) -> Option<f64> {
    let (benchmark_min, benchmark_max) = benchmark_range?;
    if benchmark_min <= observed_min && observed_max <= benchmark_max {
        return Some(1.0);
    }
    let overlap_min = observed_min.max(benchmark_min);
    let overlap_max = observed_max.min(benchmark_max);
    if overlap_min > overlap_max {
        return Some(0.0);
    }
    let observed_len = observed_max.saturating_sub(observed_min) + 1;
    let overlap_len = overlap_max.saturating_sub(overlap_min) + 1;
    Some(f64::from(overlap_len) / f64::from(observed_len))
}

fn nearest_shape_benchmark(
    benchmarks: &[CalibrationBenchmarkPoint],
    envelope: CalibrationShapeEnvelope,
) -> (Option<String>, Option<f64>) {
    let mut nearest = None;
    let mut nearest_distance = f64::INFINITY;
    for (idx, benchmark) in benchmarks.iter().enumerate() {
        let (Some(batch_size), Some(prompt_tokens), Some(decode_tokens), Some(sequence_tokens)) = (
            benchmark.batch_size,
            benchmark.prompt_tokens,
            benchmark.decode_tokens,
            benchmark.sequence_tokens,
        ) else {
            continue;
        };
        let distance = shape_point_distance(
            [
                (batch_size, envelope.min_batch_size, envelope.max_batch_size),
                (
                    prompt_tokens,
                    envelope.min_prompt_tokens,
                    envelope.max_prompt_tokens,
                ),
                (
                    decode_tokens,
                    envelope.min_decode_tokens,
                    envelope.max_decode_tokens,
                ),
                (
                    sequence_tokens,
                    envelope.min_sequence_tokens,
                    envelope.max_sequence_tokens,
                ),
            ]
            .into_iter(),
        );
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest = Some(benchmark_label(idx, benchmark));
        }
    }

    if nearest.is_some() {
        (nearest, Some(nearest_distance))
    } else {
        (None, None)
    }
}

fn shape_point_distance(values: impl Iterator<Item = (u32, u32, u32)>) -> f64 {
    values
        .map(|(value, min, max)| {
            if value < min {
                (f64::from(min) / f64::from(value.max(1))).ln()
            } else if value > max {
                (f64::from(value) / f64::from(max.max(1))).ln()
            } else {
                0.0
            }
        })
        .map(|distance| distance * distance)
        .sum::<f64>()
        .sqrt()
}

fn benchmark_label(idx: usize, benchmark: &CalibrationBenchmarkPoint) -> String {
    benchmark.name.clone().unwrap_or_else(|| {
        let kind = benchmark.kind.as_deref().unwrap_or("unknown");
        let phase = benchmark.phase.as_deref().unwrap_or("unknown");
        format!("benchmark[{idx}] {kind}/{phase}")
    })
}

fn calibration_coverage_status(
    benchmark_count: usize,
    shape_benchmark_count: usize,
    coverage_score: Option<f64>,
    missing_phases: &[String],
) -> &'static str {
    if benchmark_count == 0 {
        return "no_benchmarks";
    }
    if shape_benchmark_count == 0 {
        return "no_shape_benchmarks";
    }
    match coverage_score {
        Some(score) if score >= 0.95 && missing_phases.is_empty() => "covered",
        Some(score) if score >= 0.50 => "partial",
        Some(_) => "weak",
        None => "no_coverage_evidence",
    }
}

fn optional_mean_f64(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn calibration_invalid_shape_warnings(
    profile: Option<&CalibrationProfileMetadata>,
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Vec<CalibrationInvalidShapeWarning> {
    let Some(profile) = profile else {
        return Vec::new();
    };
    let envelope = workload_shape_envelope(request, serving);
    profile
        .invalid_shapes
        .iter()
        .filter(|invalid_shape| invalid_shape_overlaps(envelope, &invalid_shape.shape))
        .map(|invalid_shape| {
            let name = invalid_shape.name.clone();
            let reason = invalid_shape.reason.clone();
            let label = name
                .clone()
                .unwrap_or_else(|| "unnamed invalid shape".to_string());
            let reason_suffix = reason
                .as_deref()
                .map(|reason| format!(": {reason}"))
                .unwrap_or_default();
            CalibrationInvalidShapeWarning {
                name,
                reason,
                shape: invalid_shape.shape.clone(),
                message: format!(
                    "workload shape envelope overlaps calibration invalid shape '{label}'{reason_suffix}"
                ),
            }
        })
        .collect()
}

fn invalid_shape_overlaps(
    envelope: CalibrationShapeEnvelope,
    shape: &CalibrationShapeRange,
) -> bool {
    optional_range_overlaps(
        envelope.min_batch_size,
        envelope.max_batch_size,
        shape.min_batch_size,
        shape.max_batch_size,
    ) && optional_range_overlaps(
        envelope.min_prompt_tokens,
        envelope.max_prompt_tokens,
        shape.min_prompt_tokens,
        shape.max_prompt_tokens,
    ) && optional_range_overlaps(
        envelope.min_decode_tokens,
        envelope.max_decode_tokens,
        shape.min_decode_tokens,
        shape.max_decode_tokens,
    ) && optional_range_overlaps(
        envelope.min_sequence_tokens,
        envelope.max_sequence_tokens,
        shape.min_sequence_tokens,
        shape.max_sequence_tokens,
    )
}

fn optional_range_overlaps(
    observed_min: u32,
    observed_max: u32,
    invalid_min: Option<u32>,
    invalid_max: Option<u32>,
) -> bool {
    let invalid_min = invalid_min.unwrap_or(1);
    let invalid_max = invalid_max.unwrap_or(u32::MAX);
    observed_min <= invalid_max && invalid_min <= observed_max
}

fn calibration_gate_violations(
    policy: &CalibrationPolicy,
    profile: Option<&CalibrationProfileMetadata>,
    coverage: Option<&CalibrationCoverageReport>,
    valid_shape_warnings: &[CalibrationApplicabilityWarning],
    invalid_shape_warnings: &[CalibrationInvalidShapeWarning],
) -> Vec<CalibrationGateViolation> {
    let mut violations = Vec::new();
    for warning in valid_shape_warnings {
        violations.push(CalibrationGateViolation {
            code: format!("valid_shape_{}_outside_range", warning.field),
            action: policy.valid_shape,
            observed: None,
            limit: None,
            message: warning.message.clone(),
        });
    }
    for warning in invalid_shape_warnings {
        violations.push(CalibrationGateViolation {
            code: "invalid_shape_overlap".to_string(),
            action: policy.invalid_shape,
            observed: None,
            limit: None,
            message: warning.message.clone(),
        });
    }

    if profile.is_none() {
        if policy.coverage == CalibrationGateMode::Reject {
            violations.push(CalibrationGateViolation {
                code: "no_calibration_profile".to_string(),
                action: policy.coverage,
                observed: None,
                limit: policy.min_coverage_score,
                message:
                    "calibration coverage policy is reject but workload has no calibration profile"
                        .to_string(),
            });
        }
        return violations;
    }

    let profile = profile.expect("profile checked above");
    if profile.source.is_none() {
        violations.push(CalibrationGateViolation {
            code: "calibration_profile_source_unspecified".to_string(),
            action: policy.profile_source,
            observed: Some(0.0),
            limit: Some(1.0),
            message:
                "calibration profile is missing source metadata, so benchmark provenance is not auditable"
                    .to_string(),
        });
    }
    if profile.date.is_none() {
        violations.push(CalibrationGateViolation {
            code: "calibration_profile_date_unspecified".to_string(),
            action: policy.profile_date,
            observed: Some(0.0),
            limit: Some(1.0),
            message:
                "calibration profile is missing date metadata, so benchmark recency is not auditable"
                    .to_string(),
        });
    }
    let missing_runtime_provenance = missing_calibration_profile_runtime_provenance(profile);
    if !missing_runtime_provenance.is_empty() {
        let observed =
            CALIBRATION_PROFILE_RUNTIME_PROVENANCE_FIELD_COUNT - missing_runtime_provenance.len();
        violations.push(CalibrationGateViolation {
            code: "calibration_profile_runtime_provenance_incomplete".to_string(),
            action: policy.profile_runtime,
            observed: Some(observed as f64),
            limit: Some(CALIBRATION_PROFILE_RUNTIME_PROVENANCE_FIELD_COUNT as f64),
            message: format!(
                "calibration profile is missing runtime provenance fields: {}; benchmark results may not be reproducible across software stacks",
                missing_runtime_provenance.join(", ")
            ),
        });
    }

    let Some(coverage) = coverage else {
        violations.push(CalibrationGateViolation {
            code: "no_calibration_coverage".to_string(),
            action: policy.coverage,
            observed: None,
            limit: policy.min_coverage_score,
            message: "calibration profile did not produce a coverage report".to_string(),
        });
        return violations;
    };

    if let Some(min_coverage_score) = policy.min_coverage_score {
        match coverage.coverage_score {
            Some(score) if score < min_coverage_score => {
                violations.push(CalibrationGateViolation {
                    code: "coverage_score_below_min".to_string(),
                    action: policy.coverage,
                    observed: Some(score),
                    limit: Some(min_coverage_score),
                    message: format!(
                        "calibration coverage score {score:.3} is below required minimum {min_coverage_score:.3}"
                    ),
                });
            }
            None => {
                violations.push(CalibrationGateViolation {
                    code: "coverage_score_missing".to_string(),
                    action: policy.coverage,
                    observed: None,
                    limit: Some(min_coverage_score),
                    message: format!(
                        "calibration coverage score is unavailable but required minimum is {min_coverage_score:.3}"
                    ),
                });
            }
            _ => {}
        }
    }
    if policy.require_phase_coverage && !coverage.missing_phases.is_empty() {
        violations.push(CalibrationGateViolation {
            code: "missing_required_calibration_phases".to_string(),
            action: policy.coverage,
            observed: coverage.phase_coverage_score,
            limit: Some(1.0),
            message: format!(
                "calibration benchmarks are missing required phases: {}",
                coverage.missing_phases.join(", ")
            ),
        });
    }

    violations
}

const CALIBRATION_PROFILE_RUNTIME_PROVENANCE_FIELD_COUNT: usize = 6;

fn missing_calibration_profile_runtime_provenance(
    profile: &CalibrationProfileMetadata,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if profile.backend_version.is_none() {
        missing.push("backend_version");
    }
    if profile.driver_version.is_none() {
        missing.push("driver_version");
    }
    if profile.cuda_version.is_none() && profile.rocm_version.is_none() {
        missing.push("cuda_version_or_rocm_version");
    }
    if profile.nccl_version.is_none()
        && profile.rccl_version.is_none()
        && profile.ucx_version.is_none()
    {
        missing.push("nccl_version_or_rccl_version_or_ucx_version");
    }
    if profile.kernel_settings.is_empty() {
        missing.push("kernel_settings");
    }
    if profile.environment_hash.is_none() {
        missing.push("environment_hash");
    }
    missing
}

fn calibration_applicability_warnings(
    profile: Option<&CalibrationProfileMetadata>,
    request: &InferenceRequest,
    serving: Option<&DisaggregatedServingConfig>,
) -> Vec<CalibrationApplicabilityWarning> {
    let Some(valid_shape) = profile.and_then(|profile| profile.valid_shape.as_ref()) else {
        return Vec::new();
    };

    let envelope = workload_shape_envelope(request, serving);

    let mut warnings = Vec::new();
    push_shape_warning(
        &mut warnings,
        "batch_size",
        envelope.min_batch_size,
        envelope.max_batch_size,
        valid_shape.min_batch_size,
        valid_shape.max_batch_size,
    );
    push_shape_warning(
        &mut warnings,
        "prompt_tokens",
        envelope.min_prompt_tokens,
        envelope.max_prompt_tokens,
        valid_shape.min_prompt_tokens,
        valid_shape.max_prompt_tokens,
    );
    push_shape_warning(
        &mut warnings,
        "decode_tokens",
        envelope.min_decode_tokens,
        envelope.max_decode_tokens,
        valid_shape.min_decode_tokens,
        valid_shape.max_decode_tokens,
    );
    push_shape_warning(
        &mut warnings,
        "sequence_tokens",
        envelope.min_sequence_tokens,
        envelope.max_sequence_tokens,
        valid_shape.min_sequence_tokens,
        valid_shape.max_sequence_tokens,
    );
    warnings
}

fn push_shape_warning(
    warnings: &mut Vec<CalibrationApplicabilityWarning>,
    field: &str,
    observed_min: u32,
    observed_max: u32,
    calibrated_min: Option<u32>,
    calibrated_max: Option<u32>,
) {
    let below_min = calibrated_min.is_some_and(|min| observed_min < min);
    let above_max = calibrated_max.is_some_and(|max| observed_max > max);
    if !below_min && !above_max {
        return;
    }

    warnings.push(CalibrationApplicabilityWarning {
        field: field.to_string(),
        observed_min,
        observed_max,
        calibrated_min,
        calibrated_max,
        message: format!(
            "workload {field} range {} falls outside calibration range {}",
            format_u32_range(Some(observed_min), Some(observed_max)),
            format_u32_range(calibrated_min, calibrated_max)
        ),
    });
}

fn format_u32_range(min: Option<u32>, max: Option<u32>) -> String {
    match (min, max) {
        (Some(min), Some(max)) => format!("{min}..{max}"),
        (Some(min), None) => format!("{min}..unbounded"),
        (None, Some(max)) => format!("unbounded..{max}"),
        (None, None) => "unbounded".to_string(),
    }
}

fn validate_min_max(name: &str, min: Option<u32>, max: Option<u32>) -> Result<(), ConfigError> {
    validate_positive_optional_u32(&format!("{name}.min"), min)?;
    validate_positive_optional_u32(&format!("{name}.max"), max)?;
    if let (Some(min), Some(max)) = (min, max)
        && min > max
    {
        return Err(ConfigError::new(format!(
            "{name} min must be less than or equal to max"
        )));
    }
    Ok(())
}

fn validate_positive_optional_u32(name: &str, value: Option<u32>) -> Result<(), ConfigError> {
    if value == Some(0) {
        return Err(ConfigError::new(format!(
            "{name} must be greater than zero"
        )));
    }
    Ok(())
}

fn validate_optional_max_sequence_tokens(
    name: &str,
    max_sequence_tokens: Option<u32>,
    prompt_tokens: u32,
    decode_tokens: u32,
) -> Result<(), ConfigError> {
    validate_positive_optional_u32(name, max_sequence_tokens)?;
    if let Some(max_sequence_tokens) = max_sequence_tokens {
        let required_sequence_tokens = u64::from(prompt_tokens) + u64::from(decode_tokens);
        if u64::from(max_sequence_tokens) < required_sequence_tokens {
            return Err(ConfigError::new(format!(
                "{name} must be greater than or equal to prompt_tokens + decode_tokens ({required_sequence_tokens})"
            )));
        }
    }
    Ok(())
}

fn validate_positive_optional_f64(name: &str, value: Option<f64>) -> Result<(), ConfigError> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(ConfigError::new(format!(
            "{name} must be finite and greater than zero"
        )));
    }
    Ok(())
}

fn validate_positive_fraction_optional_f64(
    name: &str,
    value: Option<f64>,
) -> Result<(), ConfigError> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0 || value > 1.0)
    {
        return Err(ConfigError::new(format!(
            "{name} must be finite and greater than 0.0 and less than or equal to 1.0"
        )));
    }
    Ok(())
}

fn validate_nonnegative_optional_f64(name: &str, value: Option<f64>) -> Result<(), ConfigError> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        return Err(ConfigError::new(format!(
            "{name} must be finite and greater than or equal to zero"
        )));
    }
    Ok(())
}

fn validate_optional_finite_f64(name: &str, value: Option<f64>) -> Result<(), ConfigError> {
    if let Some(value) = value {
        validate_finite_f64(name, value)?;
    }
    Ok(())
}

fn validate_finite_f64(name: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(ConfigError::new(format!("{name} must be finite")));
    }
    Ok(())
}

fn calibration_with_defaults(
    section: Option<CalibrationSection>,
    defaults: SimulationCalibration,
) -> SimulationCalibration {
    let Some(section) = section else {
        return defaults;
    };

    SimulationCalibration {
        compute_efficiency: section
            .compute_efficiency
            .unwrap_or(defaults.compute_efficiency),
        prefill_compute_scale: section
            .prefill_compute_scale
            .unwrap_or(defaults.prefill_compute_scale),
        decode_compute_scale: section
            .decode_compute_scale
            .unwrap_or(defaults.decode_compute_scale),
        decode_memory_bandwidth_scale: section
            .decode_memory_bandwidth_scale
            .unwrap_or(defaults.decode_memory_bandwidth_scale),
        collective_latency_scale: section
            .collective_latency_scale
            .unwrap_or(defaults.collective_latency_scale),
        collective_bandwidth_scale: section
            .collective_bandwidth_scale
            .unwrap_or(defaults.collective_bandwidth_scale),
        kv_transfer_scale: section
            .kv_transfer_scale
            .unwrap_or(defaults.kv_transfer_scale),
        scheduler_overhead_us: section
            .scheduler_overhead_us
            .unwrap_or(defaults.scheduler_overhead_us),
        serving_memory_temporary_fraction: section
            .serving_memory_temporary_fraction
            .unwrap_or(defaults.serving_memory_temporary_fraction),
        serving_memory_activation_communication_fraction: section
            .serving_memory_activation_communication_fraction
            .unwrap_or(defaults.serving_memory_activation_communication_fraction),
        serving_memory_weight_communication_fraction: section
            .serving_memory_weight_communication_fraction
            .unwrap_or(defaults.serving_memory_weight_communication_fraction),
        serving_memory_runtime_reserve_fraction: section
            .serving_memory_runtime_reserve_fraction
            .unwrap_or(defaults.serving_memory_runtime_reserve_fraction),
        serving_memory_fragmentation_fraction: section
            .serving_memory_fragmentation_fraction
            .unwrap_or(defaults.serving_memory_fragmentation_fraction),
        serving_pipeline_depth: section
            .serving_pipeline_depth
            .unwrap_or(defaults.serving_pipeline_depth),
        request_arrival_gap_s: section
            .request_arrival_gap_s
            .unwrap_or(defaults.request_arrival_gap_s),
        allow_compute_comm_overlap: section
            .allow_compute_comm_overlap
            .unwrap_or(defaults.allow_compute_comm_overlap),
    }
    .sanitized()
}

fn validate_model_section(model: &ModelSection) -> Result<(), ConfigError> {
    if model.layers == 0 {
        return Err(ConfigError::new("model.layers must be greater than zero"));
    }
    if model.hidden_size == 0 {
        return Err(ConfigError::new(
            "model.hidden_size must be greater than zero",
        ));
    }
    if model.attention_heads == 0 {
        return Err(ConfigError::new(
            "model.attention_heads must be greater than zero",
        ));
    }
    if model.kv_heads == 0 {
        return Err(ConfigError::new("model.kv_heads must be greater than zero"));
    }
    if model.vocab_size == 0 {
        return Err(ConfigError::new(
            "model.vocab_size must be greater than zero",
        ));
    }
    validate_positive_optional_f64("model.parameters_gb", Some(model.parameters_gb))?;

    if !model.hidden_size.is_multiple_of(model.attention_heads) {
        return Err(ConfigError::new(
            "model.hidden_size must be divisible by model.attention_heads",
        ));
    }
    if model.kv_heads > model.attention_heads {
        return Err(ConfigError::new(
            "model.kv_heads must be less than or equal to model.attention_heads",
        ));
    }
    if !model.attention_heads.is_multiple_of(model.kv_heads) {
        return Err(ConfigError::new(
            "model.attention_heads must be divisible by model.kv_heads",
        ));
    }

    Ok(())
}

fn validate_request_section(
    request: &RequestSection,
    phase: InferencePhase,
) -> Result<(), ConfigError> {
    if request.batch_size == 0 {
        return Err(ConfigError::new(
            "request.batch_size must be greater than zero",
        ));
    }
    if matches!(phase, InferencePhase::Prefill | InferencePhase::EndToEnd)
        && request.prompt_tokens == 0
    {
        return Err(ConfigError::new(
            "request.prompt_tokens must be greater than zero for prefill and end_to_end phases",
        ));
    }
    if matches!(phase, InferencePhase::Decode | InferencePhase::EndToEnd)
        && request.decode_tokens == 0
    {
        return Err(ConfigError::new(
            "request.decode_tokens must be greater than zero for decode and end_to_end phases",
        ));
    }
    validate_optional_max_sequence_tokens(
        "request.max_sequence_tokens",
        Some(request.max_sequence_tokens),
        request.prompt_tokens,
        request.decode_tokens,
    )?;

    Ok(())
}

fn dtype(name: &str, value: &str) -> Result<DType, ConfigError> {
    match normalize(value).as_str() {
        "fp16" | "f16" => Ok(DType::Fp16),
        "bf16" | "bfloat16" => Ok(DType::Bf16),
        "fp8" | "f8" => Ok(DType::Fp8),
        "int8" | "i8" => Ok(DType::Int8),
        _ => Err(ConfigError::new(format!(
            "unsupported {name} '{value}'; use fp16, bf16, fp8, or int8"
        ))),
    }
}

fn inference_phase(value: &str) -> Result<InferencePhase, ConfigError> {
    match normalize(value).as_str() {
        "prefill" => Ok(InferencePhase::Prefill),
        "decode" => Ok(InferencePhase::Decode),
        "endtoend" | "end_to_end" => Ok(InferencePhase::EndToEnd),
        _ => Err(ConfigError::new(format!(
            "unsupported request.phase '{value}'; use prefill, decode, or end_to_end"
        ))),
    }
}

fn normalize(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

#[cfg(test)]
mod tests;
