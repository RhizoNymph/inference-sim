use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    error::Error,
    fmt::Display,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

mod calibration_config;
mod cluster;
mod run_config;
mod sections;
mod serving_config;
mod trace;
pub use calibration_config::{LoadedCalibrationProfile, load_calibration_profile_path};
use calibration_config::{
    calibration_applicability_warnings, calibration_coverage_report, calibration_gate_violations,
    calibration_invalid_shape_warnings, calibration_with_defaults, load_calibration_profile,
    nonempty_metadata, parse_approximation_policy, parse_calibration_policy,
    validate_nonnegative_optional_f64, validate_optional_max_sequence_tokens,
    validate_positive_fraction_optional_f64, validate_positive_optional_f64,
    validate_positive_optional_u32,
};
use cluster::*;
use run_config::*;
use sections::*;
use serving_config::*;
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
