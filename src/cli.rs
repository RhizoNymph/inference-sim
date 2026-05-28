use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error,
    fmt::Display,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::{
    CalibrationFitApplication, CalibrationFitFeatureValue, DisaggregatedServingConfig,
    PlacementEvidence, ScoredParallelismConfig, ScoredServingConfig, SearchSpace,
    ServingArrivalPattern, ServingCostEstimate, ServingKvRouteConstraints, ServingMetricCeilings,
    ServingMetrics, ServingObjective, ServingSolver, ServingSolverOptions, ServingTraffic,
    ServingValueDistribution, SimulationApproximation, Solver, SolverOptions,
    calibration::SimulationCalibration,
    config::{
        ApproximationMetricGate, ApproximationPolicy, ApproximationPolicyViolation,
        CalibrationApplicabilityWarning, CalibrationBenchmarkPoint, CalibrationCoverageReport,
        CalibrationFitFeatureRange, CalibrationFittedModel, CalibrationGateMode,
        CalibrationGateViolation, CalibrationInvalidShapeRange, CalibrationInvalidShapeWarning,
        CalibrationPolicy, CalibrationProfileMetadata, CalibrationShapeRange, ConfigError,
        RunConfig, RunScenarioCalibrationConfig, RunScenarioConfig,
        RunScenarioGpuDegradationOverlay, RunScenarioGpuResourceOverlay,
        RunScenarioLinkDegradationOverlay, RunScenarioNicDegradationOverlay,
        RunScenarioNicResourceOverlay, RunScenarioNodeState, RunScenarioNodeStateOverlay,
        RunScenarioRailDegradationOverlay, RunScenarioTopologyConfig, RunSearchBudgetConfig,
        WorkloadConfig, load_calibration_profile_path, load_cluster, load_run_config,
        load_workload, refresh_workload_calibration_reports, validate_workload_for_cluster,
    },
    scheduler::{
        CriticalPath, ResourceOccupancySeries, ResourceUtilization, ScheduledOperation,
        critical_path, resource_occupancy_buckets,
    },
    serving::{
        ServingApproximationCount, ServingApproximationSummary, ServingBottleneckSummary,
        ServingCalibrationSummary, ServingDecodeIterationObservation,
        ServingGpuCapacityObservation, ServingGpuLabelCount, ServingGpuTypeCount,
        ServingHardwareFootprint, ServingKvBlockOwnershipObservation,
        ServingKvRouteResourceSummary, ServingKvRouteTopologySummary,
        ServingKvTransferPathEndpointObservation, ServingKvTransferPathObservation,
        ServingKvTransferPathResourceObservation, ServingKvWorkerSlotOwnershipObservation,
        ServingMeasurementMetricSourceCount, ServingMeasurementWindowObservation,
        ServingMemoryHeadroom, ServingMemoryPressureObservation, ServingMetricBreakdown,
        ServingNodeCapacityObservation, ServingParetoDimension, ServingPhaseCalibrationObservation,
        ServingPhaseResourceUtilization, ServingPoolSearchSummary, ServingPoolTopologySummary,
        ServingRejection, ServingRequestObservation, ServingRequestWorkerSummaryObservation,
        ServingRouteCandidateObservation, ServingServiceObservation,
        ServingSloMissPenaltyComponents, ServingSloMissPenaltyWeights,
        ServingSteadyStateMetricObservation, ServingSteadyStateUtilizationObservation,
        ServingTopologyBottleneckObservation, ServingTrafficClassCapacityObservation,
        ServingWorkerAssignmentObservation, ServingWorkerKvSlotObservation,
        ServingWorkerObservation,
    },
    topology_graph::TopologyGraph,
    types::{
        common::{
            Bytes, FabricKind, GpuAddr, Latency, OperationalState, ReductionAccelerator,
            UnorderedPair,
        },
        configs::{ParallelismConfig, RankPlacement},
        fabric::{
            inter_node::{CustomInterNodeLink, FabricProfile, InterNodeTopology},
            intra_node::{GpuNicAffinity, IntraNodeTopology},
        },
        gpu::GpuProfile,
        topology::{Cluster, NodeOperationalState},
    },
    workload::{InferenceRequest, ModelSpec},
};

mod approximations;
mod args;
mod calibration;
mod csv;
mod diagnostics;
mod json_output;
mod metrics;
mod policy;
mod presentation;
mod rejections;
mod requests;
mod resources;
mod routes;
mod scenario;
mod search;
mod topology;

use approximations::*;
use args::*;
use calibration::*;
use csv::*;
use diagnostics::*;
use json_output::*;
use metrics::*;
use policy::*;
use presentation::*;
use rejections::*;
use requests::*;
use resources::*;
use routes::*;
use scenario::*;
use search::*;
use topology::*;

const DEFAULT_TOP_K: usize = 10;
const DEFAULT_TRACE_LIMIT: usize = 200;
const DEFAULT_REQUEST_OBSERVATION_LIMIT: usize = 200;
const DEFAULT_OCCUPANCY_BUCKETS: usize = 20;
const DEFAULT_OCCUPANCY_RESOURCE_LIMIT: usize = 10;
const DEFAULT_CRITICAL_PATH_LIMIT: usize = 200;
#[derive(Debug)]
pub enum CliError {
    Help(String),
    Usage(String),
    Config(ConfigError),
    Io(io::Error),
}

impl Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Help(message) => write!(f, "{message}"),
            CliError::Usage(message) => write!(f, "{message}"),
            CliError::Config(err) => write!(f, "{err}"),
            CliError::Io(err) => write!(f, "{err}"),
        }
    }
}

impl Error for CliError {}

impl From<ConfigError> for CliError {
    fn from(value: ConfigError) -> Self {
        CliError::Config(value)
    }
}

impl From<io::Error> for CliError {
    fn from(value: io::Error) -> Self {
        CliError::Io(value)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct JsonOutputOptions<'a> {
    calibration: SimulationCalibration,
    calibration_policy: &'a CalibrationPolicy,
    approximation_policy: &'a ApproximationPolicy,
    calibration_profile: Option<&'a CalibrationProfileMetadata>,
    calibration_coverage: Option<&'a CalibrationCoverageReport>,
    calibration_warnings: &'a [CalibrationApplicabilityWarning],
    calibration_invalid_shape_warnings: &'a [CalibrationInvalidShapeWarning],
    calibration_gate_violations: &'a [CalibrationGateViolation],
    serving_objective: Option<ServingObjective>,
    serving_stack: Option<&'a str>,
    serving_runtime_features: &'a [String],
    top_k: usize,
    include_trace: bool,
    trace_limit: Option<usize>,
    request_limit: Option<usize>,
    include_occupancy: bool,
    occupancy_buckets: usize,
    occupancy_resource_limit: Option<usize>,
    include_critical_path: bool,
    critical_path_limit: Option<usize>,
    search_budget: RunSearchBudgetConfig,
    search_diagnostics: SearchDiagnostics,
    searched_candidate_count: usize,
    omitted_rejected_candidate_count: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct ServingTextOutputOptions<'a> {
    calibration_gate_violations: &'a [CalibrationGateViolation],
    serving_stack: Option<&'a str>,
    serving_runtime_features: &'a [String],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct RankSensitivity {
    nominal_rank: usize,
    uncertainty_adjusted_rank: usize,
}

impl RankSensitivity {
    fn delta(self) -> i64 {
        self.uncertainty_adjusted_rank as i64 - self.nominal_rank as i64
    }
}

pub fn run_from_env() -> Result<(), CliError> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    run_with_args(std::env::args(), &mut handle)
}

pub fn run_with_args<I, S, W>(args: I, writer: &mut W) -> Result<(), CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
    W: Write,
{
    let args = CliArgs::parse(args)?;
    let cluster = load_cluster(&args.cluster_path)?;
    let workload = load_workload(&args.workload_path)?;

    if args.output_dir.is_some() {
        return write_output_profiles(writer, &cluster, &workload, &args);
    }

    run_to_writer(writer, &cluster, &workload, &args)
}

fn run_to_writer<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    workload: &WorkloadConfig,
    args: &CliArgs,
) -> Result<(), CliError> {
    if args.scenarios.is_empty() {
        return solve_and_write(writer, cluster, workload, args, None);
    }

    write_scenario_sweep(writer, cluster, workload, args)
}

fn write_output_profiles<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    workload: &WorkloadConfig,
    args: &CliArgs,
) -> Result<(), CliError> {
    let output_dir = args
        .output_dir
        .as_ref()
        .expect("output_dir is checked by caller");
    fs::create_dir_all(output_dir)?;
    let manifest_path = output_dir.join("manifest.txt");
    let mut manifest = File::create(&manifest_path)?;
    writeln!(manifest, "output_dir={}", output_dir.display())?;
    writeln!(writer, "output_dir={}", output_dir.display())?;
    writeln!(manifest, "manifest={}", manifest_path.display())?;
    writeln!(writer, "manifest={}", manifest_path.display())?;

    for profile in args.output_profile.materialized_profiles() {
        let profile_dir = output_dir.join(profile.as_str());
        fs::create_dir_all(&profile_dir)?;
        let mut profile_args = args.clone();
        profile_args.output_dir = None;
        profile_args.output_profile = *profile;
        apply_output_profile_defaults(&mut profile_args);
        apply_output_profile_paths(&mut profile_args, &profile_dir);
        let primary_path = profile_dir.join(primary_output_filename(profile_args.format));
        let mut output = File::create(&primary_path)?;
        run_to_writer(&mut output, cluster, workload, &profile_args)?;
        writeln!(manifest)?;
        writeln!(manifest, "[{}]", profile.as_str())?;
        writeln!(manifest, "dir={}", profile_dir.display())?;
        writeln!(manifest, "primary={}", primary_path.display())?;
        writeln!(writer)?;
        writeln!(writer, "profile={}", profile.as_str())?;
        writeln!(writer, "dir={}", profile_dir.display())?;
        writeln!(writer, "primary={}", primary_path.display())?;
    }

    Ok(())
}

fn solve_and_write<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    workload: &WorkloadConfig,
    args: &CliArgs,
    scenario_name: Option<&str>,
) -> Result<(), CliError> {
    validate_workload_for_cluster(cluster, workload)?;
    write_calibration_residuals_csv_if_configured(
        args,
        scenario_name,
        workload.calibration_profile.as_ref(),
        scenario_name.is_some(),
    )?;

    if let Some(serving) = &workload.serving {
        let search_budget = serving_search_budget(args.search_budget);
        let search_runtime = SearchRuntime::start(search_budget);
        let mut results = ServingSolver::rank_disaggregated_with_options(
            cluster,
            &workload.model,
            &workload.request,
            serving,
            ServingSolverOptions {
                calibration: workload.calibration,
                calibration_profile: workload.calibration_profile.as_ref(),
                model_id: workload.model_id.as_deref(),
                serving_stack: workload.serving_stack.as_deref(),
                serving_runtime_features: Some(&workload.serving_runtime_features),
                max_prefill_candidates: search_budget.max_prefill_candidates,
                max_decode_candidates: search_budget.max_decode_candidates,
                max_serving_pairs: search_budget.max_serving_pairs,
                search_deadline: search_runtime.deadline,
                explicit_prefill_placement: workload.serving_prefill_placement.as_ref(),
                explicit_decode_placement: workload.serving_decode_placement.as_ref(),
            },
        );
        apply_calibration_fit_gates_to_serving(&mut results, &workload.calibration_policy);
        apply_calibration_gates_to_serving(&mut results, &workload.calibration_gate_violations);
        apply_approximation_policy_to_serving(&mut results, &workload.approximation_policy);
        sort_serving_results_after_policy(&mut results);
        apply_uncertainty_adjusted_ranking_to_serving(&mut results, &workload.calibration_policy);
        let searched_candidate_count = results.len();
        let runtime_elapsed_ms = search_runtime.elapsed_ms();
        let truncated_by_runtime_budget = search_runtime.truncated(
            searched_candidate_count,
            effective_serving_pair_budgeted_count(serving, search_budget),
        );
        let omitted_rejected_candidate_count = if retain_rejected_candidates(search_budget) {
            0
        } else {
            rejected_candidate_count(&results, |score: &ScoredServingConfig| score.feasible)
        };
        let results = reported_serving_results(&results, search_budget);
        let search_diagnostics = serving_search_diagnostics(
            serving,
            search_budget,
            searched_candidate_count,
            results.len(),
            omitted_rejected_candidate_count,
            runtime_elapsed_ms,
            truncated_by_runtime_budget,
        );
        write_serving_metrics_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_metric_breakdowns_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_services_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_utilization_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_memory_pressure_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_timeline_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_occupancy_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_placement_evidence_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_worker_evidence_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_rejections_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_route_paths_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_kv_route_resources_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_bottlenecks_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_phase_calibration_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_approximations_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_request_metrics_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_request_lifecycle_events_csv_if_configured(
            args,
            scenario_name,
            &results,
            scenario_name.is_some(),
        )?;
        write_serving_rank_sensitivity_csv_if_configured(
            args,
            scenario_name,
            &results,
            &workload.calibration_policy,
            scenario_name.is_some(),
        )?;
        match args.format {
            OutputFormat::Text => {
                write_serving_results(
                    writer,
                    cluster,
                    &workload.model,
                    &results,
                    search_diagnostics,
                    args.top_k,
                    ServingTextOutputOptions {
                        calibration_gate_violations: &workload.calibration_gate_violations,
                        serving_stack: workload.serving_stack.as_deref(),
                        serving_runtime_features: &workload.serving_runtime_features,
                    },
                )?;
            }
            OutputFormat::Json => write_serving_json_results(
                writer,
                cluster,
                &workload.model,
                &results,
                JsonOutputOptions {
                    calibration: workload.calibration,
                    calibration_policy: &workload.calibration_policy,
                    approximation_policy: &workload.approximation_policy,
                    calibration_profile: workload.calibration_profile.as_ref(),
                    calibration_coverage: workload.calibration_coverage.as_ref(),
                    calibration_warnings: &workload.calibration_warnings,
                    calibration_invalid_shape_warnings: &workload
                        .calibration_invalid_shape_warnings,
                    calibration_gate_violations: &workload.calibration_gate_violations,
                    serving_objective: Some(serving.objective),
                    serving_stack: workload.serving_stack.as_deref(),
                    serving_runtime_features: &workload.serving_runtime_features,
                    top_k: args.top_k,
                    include_trace: args.trace,
                    trace_limit: args.trace_limit,
                    request_limit: args.request_limit,
                    include_occupancy: args.occupancy,
                    occupancy_buckets: args.occupancy_buckets,
                    occupancy_resource_limit: args.occupancy_resource_limit,
                    include_critical_path: args.critical_path,
                    critical_path_limit: args.critical_path_limit,
                    search_budget,
                    search_diagnostics,
                    searched_candidate_count,
                    omitted_rejected_candidate_count,
                },
            )?,
            OutputFormat::Markdown => write_serving_markdown_results(
                writer,
                cluster,
                &workload.model,
                &results,
                search_diagnostics,
                args.top_k,
                ServingTextOutputOptions {
                    calibration_gate_violations: &workload.calibration_gate_violations,
                    serving_stack: workload.serving_stack.as_deref(),
                    serving_runtime_features: &workload.serving_runtime_features,
                },
            )?,
        }
    } else {
        let search_budget = parallelism_search_budget(args.search_budget);
        let search_runtime = SearchRuntime::start(search_budget);
        let mut results = Solver::rank_configs_with_options(
            cluster,
            &workload.model,
            &workload.request,
            &workload.search_space,
            SolverOptions {
                calibration: workload.calibration,
                calibration_profile: workload.calibration_profile.as_ref(),
                max_candidates: search_budget.max_parallelism_candidates,
                search_deadline: search_runtime.deadline,
                explicit_placement: workload.placement.as_ref(),
            },
        );
        apply_calibration_fit_gates_to_parallelism(&mut results, &workload.calibration_policy);
        apply_calibration_gates_to_parallelism(&mut results, &workload.calibration_gate_violations);
        apply_approximation_policy_to_parallelism(&mut results, &workload.approximation_policy);
        sort_parallelism_results_after_policy(&mut results);
        apply_uncertainty_adjusted_ranking_to_parallelism(
            &mut results,
            &workload.calibration_policy,
        );
        let searched_candidate_count = results.len();
        let runtime_elapsed_ms = search_runtime.elapsed_ms();
        let truncated_by_runtime_budget = search_runtime.truncated(
            searched_candidate_count,
            effective_budgeted_count(
                search_space_candidate_count(&workload.search_space),
                search_budget.max_parallelism_candidates,
            ),
        );
        let omitted_rejected_candidate_count = if retain_rejected_candidates(search_budget) {
            0
        } else {
            rejected_candidate_count(&results, |score: &ScoredParallelismConfig| score.feasible)
        };
        let results = reported_parallelism_results(&results, search_budget);
        let search_diagnostics = parallelism_search_diagnostics(
            &workload.search_space,
            search_budget,
            searched_candidate_count,
            results.len(),
            omitted_rejected_candidate_count,
            runtime_elapsed_ms,
            truncated_by_runtime_budget,
        );
        write_parallelism_rank_sensitivity_csv_if_configured(
            args,
            scenario_name,
            &results,
            &workload.calibration_policy,
            scenario_name.is_some(),
        )?;
        match args.format {
            OutputFormat::Text => {
                write_results(
                    writer,
                    cluster,
                    &workload.model,
                    &results,
                    search_diagnostics,
                    args.top_k,
                    &workload.calibration_gate_violations,
                )?;
            }
            OutputFormat::Json => write_json_results(
                writer,
                cluster,
                &workload.model,
                &results,
                JsonOutputOptions {
                    calibration: workload.calibration,
                    calibration_policy: &workload.calibration_policy,
                    approximation_policy: &workload.approximation_policy,
                    calibration_profile: workload.calibration_profile.as_ref(),
                    calibration_coverage: workload.calibration_coverage.as_ref(),
                    calibration_warnings: &workload.calibration_warnings,
                    calibration_invalid_shape_warnings: &workload
                        .calibration_invalid_shape_warnings,
                    calibration_gate_violations: &workload.calibration_gate_violations,
                    serving_objective: None,
                    serving_stack: None,
                    serving_runtime_features: &[],
                    top_k: args.top_k,
                    include_trace: args.trace,
                    trace_limit: args.trace_limit,
                    request_limit: args.request_limit,
                    include_occupancy: args.occupancy,
                    occupancy_buckets: args.occupancy_buckets,
                    occupancy_resource_limit: args.occupancy_resource_limit,
                    include_critical_path: args.critical_path,
                    critical_path_limit: args.critical_path_limit,
                    search_budget,
                    search_diagnostics,
                    searched_candidate_count,
                    omitted_rejected_candidate_count,
                },
            )?,
            OutputFormat::Markdown => write_markdown_results(
                writer,
                cluster,
                &workload.model,
                &results,
                search_diagnostics,
                args.top_k,
                &workload.calibration_gate_violations,
            )?,
        }
    }

    Ok(())
}

fn write_u32_vec<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    values: &[u32],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}  \"{field}\": [{}]{}",
        values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        comma(trailing_comma)
    )
}

fn write_usize_vec<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    values: &[usize],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}  \"{field}\": [{}]{}",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        comma(trailing_comma)
    )
}

fn write_string_vec<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    values: &[String],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}  \"{field}\": [{}]{}",
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(", "),
        comma(trailing_comma)
    )
}

fn write_optional_string<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<&str>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let value = value.map(json_string).unwrap_or_else(|| "null".to_string());
    writeln!(
        writer,
        "{indent}  \"{field}\": {value}{}",
        comma(trailing_comma)
    )
}

fn write_optional_json_string<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    value: Option<&str>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let value = value.map(json_string).unwrap_or_else(|| "null".to_string());
    writeln!(
        writer,
        "{indent}    \"{field}\": {value}{}",
        comma(trailing_comma)
    )
}

fn pool_label(score: &ScoredServingConfig) -> String {
    score.pool_label.clone().unwrap_or_else(|| {
        format!(
            "p{}{}->d{}{}",
            node_list(&score.prefill_nodes),
            gpu_label_suffix(&score.prefill_gpu_labels),
            node_list(&score.decode_nodes),
            gpu_label_suffix(&score.decode_gpu_labels),
        )
    })
}

fn gpu_label_suffix(labels: &[String]) -> String {
    if labels.is_empty() {
        String::new()
    } else {
        format!(":gpu_labels[{}]", labels.join("|"))
    }
}

fn parallelism_candidate_id(score: &ScoredParallelismConfig) -> String {
    parallelism_config_id(&score.config)
}

fn parallelism_config_id(config: &ParallelismConfig) -> String {
    format!(
        "tp{}-pp{}-ep{}-dp{}",
        config.tensor_ranks, config.pipeline_ranks, config.expert_ranks, config.data_ranks
    )
}

fn serving_objective_label(results: &[ScoredServingConfig]) -> &'static str {
    results
        .first()
        .map(|score| score.objective.as_str())
        .unwrap_or_else(|| ServingObjective::default().as_str())
}

fn node_list(nodes: &[u32]) -> String {
    let mut values = nodes.to_vec();
    values.sort_unstable();
    values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn u32_list(values: &[u32]) -> String {
    values
        .iter()
        .copied()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn string_list(values: &[String]) -> String {
    values
        .iter()
        .map(|value| json_string(value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn usize_list(values: &[usize]) -> String {
    values
        .iter()
        .copied()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn ms_list(values: &[f64]) -> String {
    values
        .iter()
        .copied()
        .map(json_ms)
        .collect::<Vec<_>>()
        .join(", ")
}

fn metric_ms(feasible: bool, seconds: f64) -> String {
    if feasible && seconds.is_finite() {
        format!("{:.3}", seconds * 1000.0)
    } else {
        "-".to_string()
    }
}

fn metric_percent(feasible: bool, rate: f64) -> String {
    if feasible && rate.is_finite() {
        format!("{:.1}%", rate * 100.0)
    } else {
        "-".to_string()
    }
}

fn status(feasible: bool) -> &'static str {
    if feasible { "ok" } else { "reject" }
}

fn json_optional_ms(feasible: bool, seconds: f64) -> String {
    if feasible {
        json_ms(seconds)
    } else {
        "null".to_string()
    }
}

fn json_ms(seconds: f64) -> String {
    json_optional_f64(seconds * 1000.0)
}

fn json_optional_f64(value: f64) -> String {
    if value.is_finite() {
        json_f64(value)
    } else {
        "null".to_string()
    }
}

fn json_optional_value(value: Option<f64>) -> String {
    value.map(json_f64).unwrap_or_else(|| "null".to_string())
}

fn json_optional_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_optional_string(value: Option<&str>) -> String {
    value.map(json_string).unwrap_or_else(|| "null".to_string())
}

fn json_u32_array(values: &[u32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn json_string_array(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn json_f64(value: f64) -> String {
    format!("{value:.6}")
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn comma(trailing_comma: bool) -> &'static str {
    if trailing_comma { "," } else { "" }
}

#[cfg(test)]
mod tests;
