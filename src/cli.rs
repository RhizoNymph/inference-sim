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

mod args;
mod calibration;
mod csv;
mod policy;
mod rejections;
mod scenario;
mod search;
mod topology;

use args::*;
use calibration::*;
use csv::*;
use policy::*;
use rejections::*;
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

fn write_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredParallelismConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    calibration_gate_violations: &[CalibrationGateViolation],
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "cluster_gpus={} model_params_gb={:.2} model_parameter_count_billion={:.3} searched_configs={} reported_configs={} omitted_rejected_configs={}",
        cluster.total_gpus(),
        model.parameters.as_gigabytes(),
        model.parameter_count_billion(),
        search_diagnostics.searched_candidate_count,
        results.len(),
        search_diagnostics.omitted_rejected_candidate_count
    )?;
    write_cluster_inventory_text(writer, cluster)?;
    write_search_diagnostics_text(writer, search_diagnostics)?;
    write_calibration_gate_text(writer, calibration_gate_violations)?;
    write_trust_boundary_text(writer)?;
    writeln!(
        writer,
        "{:<5} {:<8} {:>12} {:>12} {:>7} {:>4} {:>4} {:>4} {:>4}  notes",
        "rank", "status", "latency_ms", "memory_gb", "ranks", "tp", "pp", "ep", "dp"
    )?;

    for (idx, score) in results.iter().take(top_k).enumerate() {
        let status = if score.feasible { "ok" } else { "reject" };
        let latency_ms = if score.feasible {
            format!("{:.3}", score.estimated_latency_s * 1000.0)
        } else {
            "-".to_string()
        };
        let notes = notes_with_approximations(
            score.rejected_reason.as_deref(),
            &score.bottlenecks,
            &score.approximations,
        );

        writeln!(
            writer,
            "{:<5} {:<8} {:>12} {:>12.2} {:>7} {:>4} {:>4} {:>4} {:>4}  {}",
            idx + 1,
            status,
            latency_ms,
            score.estimated_memory_per_gpu.as_gigabytes(),
            score.config.total_ranks(),
            score.config.tensor_ranks,
            score.config.pipeline_ranks,
            score.config.expert_ranks,
            score.config.data_ranks,
            notes
        )?;
    }

    Ok(())
}

fn write_markdown_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredParallelismConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    calibration_gate_violations: &[CalibrationGateViolation],
) -> Result<(), io::Error> {
    writeln!(writer, "# Inference Sim Summary\n")?;
    write_markdown_key_value_section(
        writer,
        "Overview",
        &[
            ("Mode", "parallelism".to_string()),
            ("Cluster GPUs", cluster.total_gpus().to_string()),
            ("Available GPUs", cluster.available_gpus().to_string()),
            (
                "Model Parameters GB",
                format!("{:.2}", model.parameters.as_gigabytes()),
            ),
            (
                "Model Parameter Count B",
                format!("{:.3}", model.parameter_count_billion()),
            ),
            (
                "Searched Candidates",
                search_diagnostics.searched_candidate_count.to_string(),
            ),
            ("Reported Candidates", results.len().to_string()),
            (
                "Omitted Rejected Candidates",
                search_diagnostics
                    .omitted_rejected_candidate_count
                    .to_string(),
            ),
        ],
    )?;
    write_markdown_search_diagnostics(writer, search_diagnostics)?;
    write_markdown_cluster_inventory(writer, cluster)?;
    write_markdown_calibration_gates(writer, calibration_gate_violations)?;

    writeln!(writer, "## Candidates\n")?;
    write_markdown_row(
        writer,
        &[
            "Rank",
            "Status",
            "Latency ms",
            "Memory/GPU GB",
            "Ranks",
            "TP",
            "PP",
            "EP",
            "DP",
            "Notes",
        ],
    )?;
    write_markdown_separator(writer, 10)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[
                (idx + 1).to_string(),
                status(score.feasible).to_string(),
                if score.feasible {
                    format!("{:.3}", score.estimated_latency_s * 1000.0)
                } else {
                    "-".to_string()
                },
                format!("{:.2}", score.estimated_memory_per_gpu.as_gigabytes()),
                score.config.total_ranks().to_string(),
                score.config.tensor_ranks.to_string(),
                score.config.pipeline_ranks.to_string(),
                score.config.expert_ranks.to_string(),
                score.config.data_ranks.to_string(),
                parallelism_candidate_notes(score),
            ],
        )?;
    }
    writeln!(writer)?;
    write_markdown_trust_boundary(writer)
}

fn write_serving_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredServingConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    options: ServingTextOutputOptions<'_>,
) -> Result<(), io::Error> {
    let serving_stack_note = options
        .serving_stack
        .map(|stack| format!(" serving_stack={stack}"))
        .unwrap_or_default();
    let runtime_features_note = if options.serving_runtime_features.is_empty() {
        String::new()
    } else {
        format!(
            " runtime_features={}",
            options.serving_runtime_features.join(",")
        )
    };
    writeln!(
        writer,
        "cluster_gpus={} model_params_gb={:.2} model_parameter_count_billion={:.3} searched_serving_pairs={} reported_serving_pairs={} omitted_rejected_serving_pairs={} objective={}{}{}",
        cluster.total_gpus(),
        model.parameters.as_gigabytes(),
        model.parameter_count_billion(),
        search_diagnostics.searched_candidate_count,
        results.len(),
        search_diagnostics.omitted_rejected_candidate_count,
        serving_objective_label(results),
        serving_stack_note,
        runtime_features_note
    )?;
    write_cluster_inventory_text(writer, cluster)?;
    write_search_diagnostics_text(writer, search_diagnostics)?;
    write_calibration_gate_text(writer, options.calibration_gate_violations)?;
    write_serving_rejection_summary_text(writer, results)?;
    write_trust_boundary_text(writer)?;
    writeln!(
        writer,
        "{:<5} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5} {:>5} {:>11} {:>8} {:>11} {:>13} {:>12} {:>12} {:>12} {:>14} {:>8} {:>8}  notes",
        "rank",
        "status",
        "ttft_ms",
        "ttft_p95",
        "tpot_ms",
        "tpot_p95",
        "itl_ms",
        "itl_p95",
        "ttft_miss",
        "tpot_miss",
        "itl_miss",
        "e2el_miss",
        "throughput",
        "e2el_ms",
        "e2el_p95",
        "queue_ms",
        "kv_ms",
        "sched_ms",
        "reqs",
        "meas",
        "pf_tok_peak",
        "seq_peak",
        "kv_tok_peak",
        "seq_node_peak",
        "kv_node_peak",
        "seq_gpu_peak",
        "kv_gpu_peak",
        "pool",
        "prefill",
        "decode"
    )?;

    for (idx, score) in results.iter().take(top_k).enumerate() {
        let status = if score.feasible { "ok" } else { "reject" };
        let mut notes = notes_with_approximations(
            score.rejected_reason.as_deref(),
            &score.bottlenecks,
            &score.approximations,
        );
        if let Some(approximation_note) =
            serving_approximation_summary_note(&score.approximation_summary)
        {
            append_note(&mut notes, &approximation_note);
        }
        if score.slo_miss_penalty_weights.any_nonzero()
            || !score.traffic_class_slo_miss_penalties.is_empty()
        {
            let penalty_note = format!("slo_penalty={:.4}", score.slo_miss_penalty_score);
            if notes == "none" {
                notes = penalty_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&penalty_note);
            }
        }
        if score.route_coverage.candidate_count > 0
            && score.route_coverage.routable_candidate_count < score.route_coverage.candidate_count
        {
            let route_note = format!(
                "route_coverage={}/{}",
                score.route_coverage.routable_candidate_count, score.route_coverage.candidate_count
            );
            if notes == "none" {
                notes = route_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&route_note);
            }
        }
        if let Some(pool_search_note) = pool_search_summary_note(score.pool_search_summary.as_ref())
        {
            append_note(&mut notes, &pool_search_note);
        }
        if let Some(pool_topology_note) = pool_topology_summary_note(&score.pool_topology) {
            append_note(&mut notes, &pool_topology_note);
        }
        if score.topology_risk_penalty_score > 0.0 {
            let penalty_note = format!("topology_penalty={:.4}", score.topology_risk_penalty_score);
            if notes == "none" {
                notes = penalty_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&penalty_note);
            }
        }
        if score.service_backpressure_penalty_score > 0.0 {
            let penalty_note = format!(
                "backpressure_penalty={:.4}",
                score.service_backpressure_penalty_score
            );
            if notes == "none" {
                notes = penalty_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&penalty_note);
            }
        }
        if let Some(route_note) = kv_route_summary_note(score) {
            if notes == "none" {
                notes = route_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&route_note);
            }
        }
        if let Some(calibration_note) = serving_calibration_summary_note(score) {
            if notes == "none" {
                notes = calibration_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&calibration_note);
            }
        }
        let mode_note = format!("mode={}", score.deployment_mode.as_str());
        if notes == "none" {
            notes = mode_note;
        } else {
            notes.push_str("; ");
            notes.push_str(&mode_note);
        }
        if let Some(footprint_note) = serving_footprint_summary_note(score) {
            if notes == "none" {
                notes = footprint_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&footprint_note);
            }
        }
        if let Some(cost_note) = serving_cost_summary_note(score) {
            if notes == "none" {
                notes = cost_note;
            } else {
                notes.push_str("; ");
                notes.push_str(&cost_note);
            }
        }
        if let Some(pareto_note) = serving_pareto_note(score) {
            append_note(&mut notes, &pareto_note);
        }
        if let Some(objective_note) = serving_objective_summary_note(score) {
            append_note(&mut notes, &objective_note);
        }
        if let Some(bottleneck_note) = serving_bottleneck_summary_note(score) {
            append_note(&mut notes, &bottleneck_note);
        }
        let prefill = format!(
            "{}/{}/{}/{}",
            score.prefill_config.tensor_ranks,
            score.prefill_config.pipeline_ranks,
            score.prefill_config.expert_ranks,
            score.prefill_config.data_ranks
        );
        let decode = format!(
            "{}/{}/{}/{}",
            score.decode_config.tensor_ranks,
            score.decode_config.pipeline_ranks,
            score.decode_config.expert_ranks,
            score.decode_config.data_ranks
        );
        let pool = pool_label(score);

        writeln!(
            writer,
            "{:<5} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>12.2} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5} {:>5} {:>11} {:>8} {:>11} {:>13} {:>12} {:>12} {:>12} {:>14} {:>8} {:>8}  {}",
            idx + 1,
            status,
            metric_ms(score.feasible, score.metrics.ttft_s),
            metric_ms(score.feasible, score.metrics.ttft_p95_s),
            metric_ms(score.feasible, score.metrics.tpot_s),
            metric_ms(score.feasible, score.metrics.tpot_p95_s),
            metric_ms(score.feasible, score.metrics.itl_s),
            metric_ms(score.feasible, score.metrics.itl_p95_s),
            metric_percent(score.feasible, score.metrics.ttft_slo_miss_rate),
            metric_percent(score.feasible, score.metrics.tpot_slo_miss_rate),
            metric_percent(score.feasible, score.metrics.itl_slo_miss_rate),
            metric_percent(score.feasible, score.metrics.e2el_slo_miss_rate),
            score.metrics.throughput_tokens_per_s,
            metric_ms(score.feasible, score.metrics.e2el_s),
            metric_ms(score.feasible, score.metrics.e2el_p95_s),
            metric_ms(score.feasible, score.metrics.queue_delay_s),
            metric_ms(score.feasible, score.metrics.kv_transfer_s),
            metric_ms(score.feasible, score.metrics.scheduled_makespan_s),
            score.metrics.scheduled_requests,
            score.metrics.measured_requests,
            score.metrics.peak_prefill_tokens,
            score.metrics.peak_decode_sequences,
            score.metrics.peak_resident_tokens,
            score.metrics.peak_decode_sequences_per_node,
            score.metrics.peak_resident_tokens_per_node,
            score.metrics.peak_decode_sequences_per_gpu,
            score.metrics.peak_resident_tokens_per_gpu,
            pool,
            prefill,
            decode,
            notes
        )?;
    }

    Ok(())
}

fn write_serving_markdown_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredServingConfig],
    search_diagnostics: SearchDiagnostics,
    top_k: usize,
    options: ServingTextOutputOptions<'_>,
) -> Result<(), io::Error> {
    let runtime_features = if options.serving_runtime_features.is_empty() {
        "none".to_string()
    } else {
        options.serving_runtime_features.join(", ")
    };
    write!(writer, "# Inference Sim Summary\n\n")?;
    write_markdown_key_value_section(
        writer,
        "Overview",
        &[
            ("Mode", "serving".to_string()),
            ("Objective", serving_objective_label(results).to_string()),
            (
                "Serving Stack",
                options.serving_stack.unwrap_or("none").to_string(),
            ),
            ("Runtime Features", runtime_features),
            ("Cluster GPUs", cluster.total_gpus().to_string()),
            ("Available GPUs", cluster.available_gpus().to_string()),
            (
                "Model Parameters GB",
                format!("{:.2}", model.parameters.as_gigabytes()),
            ),
            (
                "Model Parameter Count B",
                format!("{:.3}", model.parameter_count_billion()),
            ),
            (
                "Searched Serving Pairs",
                search_diagnostics.searched_candidate_count.to_string(),
            ),
            ("Reported Serving Pairs", results.len().to_string()),
            (
                "Omitted Rejected Serving Pairs",
                search_diagnostics
                    .omitted_rejected_candidate_count
                    .to_string(),
            ),
        ],
    )?;
    write_markdown_search_diagnostics(writer, search_diagnostics)?;
    write_markdown_cluster_inventory(writer, cluster)?;
    write_markdown_calibration_gates(writer, options.calibration_gate_violations)?;
    write_markdown_serving_rejection_summary(writer, results)?;

    writeln!(writer, "## Serving Candidates\n")?;
    write_markdown_row(
        writer,
        &[
            "Rank",
            "Status",
            "TTFT ms",
            "TPOT ms",
            "ITL ms",
            "E2EL ms",
            "Throughput tok/s",
            "Queue ms",
            "KV ms",
            "Reqs",
            "Measured",
            "Pool",
            "Mode",
            "Prefill TP/PP/EP/DP",
            "Decode TP/PP/EP/DP",
        ],
    )?;
    write_markdown_separator(writer, 15)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[
                (idx + 1).to_string(),
                status(score.feasible).to_string(),
                metric_ms(score.feasible, score.metrics.ttft_s),
                metric_ms(score.feasible, score.metrics.tpot_s),
                metric_ms(score.feasible, score.metrics.itl_s),
                metric_ms(score.feasible, score.metrics.e2el_s),
                if score.feasible && score.metrics.throughput_tokens_per_s.is_finite() {
                    format!("{:.2}", score.metrics.throughput_tokens_per_s)
                } else {
                    "-".to_string()
                },
                metric_ms(score.feasible, score.metrics.queue_delay_s),
                metric_ms(score.feasible, score.metrics.kv_transfer_s),
                score.metrics.scheduled_requests.to_string(),
                score.metrics.measured_requests.to_string(),
                pool_label(score),
                score.deployment_mode.as_str().to_string(),
                parallelism_config_summary(&score.prefill_config),
                parallelism_config_summary(&score.decode_config),
            ],
        )?;
    }
    writeln!(writer)?;

    writeln!(writer, "## SLO And Capacity\n")?;
    write_markdown_row(
        writer,
        &[
            "Rank",
            "TTFT Miss",
            "TPOT Miss",
            "ITL Miss",
            "E2EL Miss",
            "Peak Prefill Tokens",
            "Peak Decode Sequences",
            "Peak Resident Tokens",
            "Peak KV Blocks",
        ],
    )?;
    write_markdown_separator(writer, 9)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[
                (idx + 1).to_string(),
                metric_percent(score.feasible, score.metrics.ttft_slo_miss_rate),
                metric_percent(score.feasible, score.metrics.tpot_slo_miss_rate),
                metric_percent(score.feasible, score.metrics.itl_slo_miss_rate),
                metric_percent(score.feasible, score.metrics.e2el_slo_miss_rate),
                score.metrics.peak_prefill_tokens.to_string(),
                score.metrics.peak_decode_sequences.to_string(),
                score.metrics.peak_resident_tokens.to_string(),
                score.metrics.peak_kv_blocks.to_string(),
            ],
        )?;
    }
    writeln!(writer)?;

    writeln!(writer, "## Candidate Notes\n")?;
    write_markdown_row(writer, &["Rank", "Notes"])?;
    write_markdown_separator(writer, 2)?;
    for (idx, score) in results.iter().take(top_k).enumerate() {
        write_markdown_row(
            writer,
            &[(idx + 1).to_string(), serving_candidate_notes(score)],
        )?;
    }
    writeln!(writer)?;
    write_markdown_trust_boundary(writer)
}

fn parallelism_config_summary(config: &ParallelismConfig) -> String {
    format!(
        "{}/{}/{}/{}",
        config.tensor_ranks, config.pipeline_ranks, config.expert_ranks, config.data_ranks
    )
}

fn parallelism_candidate_notes(score: &ScoredParallelismConfig) -> String {
    notes_with_approximations(
        score.rejected_reason.as_deref(),
        &score.bottlenecks,
        &score.approximations,
    )
}

fn serving_candidate_notes(score: &ScoredServingConfig) -> String {
    let mut notes = notes_with_approximations(
        score.rejected_reason.as_deref(),
        &score.bottlenecks,
        &score.approximations,
    );
    if let Some(approximation_note) =
        serving_approximation_summary_note(&score.approximation_summary)
    {
        append_note(&mut notes, &approximation_note);
    }
    if score.slo_miss_penalty_weights.any_nonzero()
        || !score.traffic_class_slo_miss_penalties.is_empty()
    {
        append_note(
            &mut notes,
            &format!("slo_penalty={:.4}", score.slo_miss_penalty_score),
        );
    }
    if score.route_coverage.candidate_count > 0
        && score.route_coverage.routable_candidate_count < score.route_coverage.candidate_count
    {
        append_note(
            &mut notes,
            &format!(
                "route_coverage={}/{}",
                score.route_coverage.routable_candidate_count, score.route_coverage.candidate_count
            ),
        );
    }
    if let Some(pool_search_note) = pool_search_summary_note(score.pool_search_summary.as_ref()) {
        append_note(&mut notes, &pool_search_note);
    }
    if let Some(pool_topology_note) = pool_topology_summary_note(&score.pool_topology) {
        append_note(&mut notes, &pool_topology_note);
    }
    if score.topology_risk_penalty_score > 0.0 {
        append_note(
            &mut notes,
            &format!("topology_penalty={:.4}", score.topology_risk_penalty_score),
        );
    }
    if score.service_backpressure_penalty_score > 0.0 {
        append_note(
            &mut notes,
            &format!(
                "backpressure_penalty={:.4}",
                score.service_backpressure_penalty_score
            ),
        );
    }
    if let Some(route_note) = kv_route_summary_note(score) {
        append_note(&mut notes, &route_note);
    }
    if let Some(calibration_note) = serving_calibration_summary_note(score) {
        append_note(&mut notes, &calibration_note);
    }
    append_note(
        &mut notes,
        &format!("mode={}", score.deployment_mode.as_str()),
    );
    if let Some(footprint_note) = serving_footprint_summary_note(score) {
        append_note(&mut notes, &footprint_note);
    }
    if let Some(cost_note) = serving_cost_summary_note(score) {
        append_note(&mut notes, &cost_note);
    }
    if let Some(pareto_note) = serving_pareto_note(score) {
        append_note(&mut notes, &pareto_note);
    }
    if let Some(objective_note) = serving_objective_summary_note(score) {
        append_note(&mut notes, &objective_note);
    }
    if let Some(bottleneck_note) = serving_bottleneck_summary_note(score) {
        append_note(&mut notes, &bottleneck_note);
    }
    notes
}

fn write_markdown_key_value_section<W: Write>(
    writer: &mut W,
    title: &str,
    rows: &[(&str, String)],
) -> Result<(), io::Error> {
    writeln!(writer, "## {title}\n")?;
    write_markdown_row(writer, &["Metric", "Value"])?;
    write_markdown_separator(writer, 2)?;
    for (label, value) in rows {
        write_markdown_row(writer, &[label.to_string(), value.clone()])?;
    }
    writeln!(writer)
}

fn write_markdown_calibration_gates<W: Write>(
    writer: &mut W,
    violations: &[CalibrationGateViolation],
) -> Result<(), io::Error> {
    if violations.is_empty() {
        return Ok(());
    }
    writeln!(writer, "## Calibration Gate Violations\n")?;
    write_markdown_row(writer, &["Action", "Code", "Observed", "Limit", "Message"])?;
    write_markdown_separator(writer, 5)?;
    for violation in violations {
        write_markdown_row(
            writer,
            &[
                violation.action.as_str().to_string(),
                violation.code.clone(),
                json_optional_value(violation.observed),
                json_optional_value(violation.limit),
                violation.message.clone(),
            ],
        )?;
    }
    writeln!(writer)
}

fn write_markdown_trust_boundary<W: Write>(writer: &mut W) -> Result<(), io::Error> {
    writeln!(writer, "## Trust Boundary\n")?;
    writeln!(
        writer,
        "This is a v1 planning estimate. Compare candidates with calibration, approximation, rejection, and bottleneck artifacts before treating absolute latency or throughput as credible."
    )
}

fn write_markdown_row<W, S>(writer: &mut W, cells: &[S]) -> Result<(), io::Error>
where
    W: Write,
    S: AsRef<str>,
{
    write!(writer, "|")?;
    for cell in cells {
        write!(writer, " {} |", markdown_cell(cell.as_ref()))?;
    }
    writeln!(writer)
}

fn write_markdown_separator<W: Write>(
    writer: &mut W,
    column_count: usize,
) -> Result<(), io::Error> {
    write!(writer, "|")?;
    for _ in 0..column_count {
        write!(writer, " --- |")?;
    }
    writeln!(writer)
}

fn markdown_cell(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "-".to_string();
    }
    trimmed
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "<br>")
        .replace('\r', "")
}

fn demote_markdown_summary_headings(markdown: &str) -> String {
    let mut output = String::new();
    for line in markdown.lines() {
        if line.trim() == "# Inference Sim Summary" {
            continue;
        }
        if line.starts_with('#') {
            output.push('#');
        }
        output.push_str(line);
        output.push('\n');
    }
    output
}

fn pool_search_summary_note(summary: Option<&ServingPoolSearchSummary>) -> Option<String> {
    let summary = summary?;
    let considered = summary.considered_candidate_count();
    if considered == 0 && summary.generated_candidate_count == 0 {
        return Some("pool_search=0 generated".to_string());
    }

    let mut note = format!(
        "pool_search={}/{} generated",
        summary.generated_candidate_count, considered
    );
    let rejected_overlap = summary.rejected_overlap_count();
    let rejected_mode = summary.rejected_mode_count();
    let rejected_spread = summary.rejected_domain_spread_count();
    let duplicates = summary.duplicate_candidate_count();
    let mut details = Vec::new();
    let mode_counts = [
        ("colocated", summary.generated_colocated_count()),
        ("partial", summary.generated_partially_disaggregated_count()),
        ("full", summary.generated_fully_disaggregated_count()),
    ]
    .into_iter()
    .filter(|(_, count)| *count > 0)
    .map(|(label, count)| format!("{label}:{count}"))
    .collect::<Vec<_>>();
    if !mode_counts.is_empty() {
        details.push(format!("modes={}", mode_counts.join("|")));
    }
    if rejected_overlap > 0 {
        details.push(format!("overlap={rejected_overlap}"));
    }
    if rejected_mode > 0 {
        details.push(format!("mode={rejected_mode}"));
    }
    if rejected_spread > 0 {
        details.push(format!("spread={rejected_spread}"));
    }
    if duplicates > 0 {
        details.push(format!("duplicate={duplicates}"));
    }
    if summary.truncated {
        details.push(format!("truncated_at={}", summary.max_candidates));
    }
    if !details.is_empty() {
        note.push_str(" (");
        note.push_str(&details.join(","));
        note.push(')');
    }
    Some(note)
}

fn pool_topology_summary_note(summary: &ServingPoolTopologySummary) -> Option<String> {
    let has_metadata = !summary.prefill_racks.is_empty()
        || !summary.decode_racks.is_empty()
        || !summary.prefill_islands.is_empty()
        || !summary.decode_islands.is_empty()
        || !summary.prefill_failure_domains.is_empty()
        || !summary.decode_failure_domains.is_empty()
        || !summary.prefill_node_labels.is_empty()
        || !summary.decode_node_labels.is_empty();
    if !has_metadata && summary.shared_node_count == 0 {
        return None;
    }

    Some(format!(
        "pool_topology=pf[nodes={},dedicated={},racks={},islands={},fds={}] decode[nodes={},dedicated={},racks={},islands={},fds={}] shared={}",
        summary.prefill_node_count,
        summary.dedicated_prefill_node_count,
        summary.prefill_racks.len(),
        summary.prefill_islands.len(),
        summary.prefill_failure_domains.len(),
        summary.decode_node_count,
        summary.dedicated_decode_node_count,
        summary.decode_racks.len(),
        summary.decode_islands.len(),
        summary.decode_failure_domains.len(),
        summary.shared_node_count
    ))
}

fn serving_footprint_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let footprint = &score.hardware_footprint;
    (footprint.unique_gpu_count > 0).then(|| {
        format!(
            "footprint_gpus={}; footprint_nodes={}; throughput_per_gpu={:.2}",
            footprint.unique_gpu_count,
            footprint.unique_node_count,
            footprint.throughput_tokens_per_s_per_gpu
        )
    })
}

fn serving_cost_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let estimate = &score.cost_estimate;
    let total_cost = estimate.total_cost_usd?;
    Some(match estimate.cost_per_1k_output_tokens_usd {
        Some(per_1k) => format!("cost=${total_cost:.6}; cost_per_1k_out=${per_1k:.6}"),
        None => format!("cost=${total_cost:.6}"),
    })
}

fn serving_calibration_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let summary = &score.calibration_summary;
    (!summary.status.is_empty()).then(|| {
        format!(
            "calibration={}:coverage={}/{}",
            summary.status, summary.calibrated_phase_count, summary.active_phase_count
        )
    })
}

fn serving_pareto_note(score: &ScoredServingConfig) -> Option<String> {
    let rank = score.pareto.rank?;
    if score.pareto.is_frontier {
        Some("pareto=frontier".to_string())
    } else {
        Some(format!("pareto_rank={rank}"))
    }
}

fn serving_bottleneck_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let bottleneck = score
        .bottleneck_summary
        .iter()
        .find(|bottleneck| bottleneck.severity != "info")
        .or_else(|| score.bottleneck_summary.first())?;
    Some(format!(
        "top_bottleneck={}:{}:{}",
        bottleneck.severity, bottleneck.code, bottleneck.resource
    ))
}

fn kv_route_summary_note(score: &ScoredServingConfig) -> Option<String> {
    let resource = score.kv_route_resource_summary.first()?;
    let rail_note = if score.kv_route_topology_summary.single_rail_dependency {
        format!(
            " rails=1(single:{})",
            score
                .kv_route_topology_summary
                .single_rail_id
                .map(|rail| rail.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )
    } else if score.kv_route_topology_summary.rail_count > 0 {
        format!(" rails={}", score.kv_route_topology_summary.rail_count)
    } else {
        String::new()
    };
    Some(format!(
        "kv_route_top={}:{}{}",
        resource.kind, resource.label, rail_note
    ))
}

fn notes_with_approximations(
    rejected_reason: Option<&str>,
    bottlenecks: &[String],
    approximations: &[SimulationApproximation],
) -> String {
    if let Some(reason) = rejected_reason {
        return reason.to_string();
    }
    let mut notes = if bottlenecks.is_empty() {
        "none".to_string()
    } else {
        bottlenecks.join("; ")
    };
    if !approximations.is_empty() {
        let mut codes: Vec<_> = approximations
            .iter()
            .take(3)
            .map(|approximation| approximation.code.as_str())
            .collect();
        let more = approximations.len().saturating_sub(codes.len());
        if more > 0 {
            codes.push("...");
        }
        notes.push_str("; approximations: ");
        notes.push_str(&codes.join(","));
        if more > 0 {
            notes.push_str(&format!(" (+{more})"));
        }
    }
    notes
}

fn append_note(notes: &mut String, note: &str) {
    if note.is_empty() {
        return;
    }
    if notes == "none" {
        *notes = note.to_string();
    } else {
        notes.push_str("; ");
        notes.push_str(note);
    }
}

fn serving_approximation_summary_note(summary: &ServingApproximationSummary) -> Option<String> {
    if summary.approximation_count == 0 && summary.policy_violation_count == 0 {
        return None;
    }

    let mut parts = vec![format!(
        "approximation_summary={}:{}",
        summary.status, summary.approximation_count
    )];
    if summary.policy_violation_count > 0 {
        parts.push(format!("policy_rejects={}", summary.policy_violation_count));
    }
    if summary.uncalibrated_runtime {
        parts.push("uncalibrated_runtime".to_string());
    }
    if summary.uncalibrated_phase_count > 0 {
        parts.push(format!(
            "uncalibrated_phases={}",
            summary.uncalibrated_phase_count
        ));
    }
    if summary.uncalibrated_queue_component_count > 0 {
        parts.push(format!(
            "uncalibrated_queues={}",
            summary.uncalibrated_queue_component_count
        ));
    }
    if summary.extrapolated_fit_count > 0 {
        parts.push(format!(
            "extrapolated_fits={}",
            summary.extrapolated_fit_count
        ));
    }
    if summary.coarse_topology {
        parts.push("coarse_topology".to_string());
    }
    if summary.approximate_queueing {
        parts.push("approx_queueing".to_string());
    }

    Some(parts.join(","))
}

struct TrustBoundaryNote {
    code: &'static str,
    scope: &'static str,
    label: &'static str,
    implication: &'static str,
}

const TRUST_BOUNDARY_NOTES: &[TrustBoundaryNote] = &[
    TrustBoundaryNote {
        code: "coarse_topology",
        scope: "topology",
        label: "Route resources are modeled at node, GPU, NIC, rail, and link scope, not full PCIe/NUMA/switch microarchitecture.",
        implication: "Use route evidence for relative placement and rail-risk analysis; calibrate before treating absolute locality costs as production latencies.",
    },
    TrustBoundaryNote {
        code: "approximate_queueing",
        scope: "serving_scheduler",
        label: "Queueing uses scheduled timeline and worker-readiness approximations rather than a production runtime event loop.",
        implication: "TTFT, TPOT, throughput, and E2EL are planner estimates whose accuracy depends on traffic shape and calibration coverage.",
    },
    TrustBoundaryNote {
        code: "aggregate_memory",
        scope: "memory",
        label: "HBM pressure is estimated from aggregate weights, KV cache, block tables, activations, communication, reserves, and fragmentation terms.",
        implication: "Capacity comparisons are useful, but allocator behavior, kernel workspace spikes, and exact fragmentation require measured validation.",
    },
    TrustBoundaryNote {
        code: "calibration_dependent",
        scope: "calibration",
        label: "Absolute latency and throughput are only as trustworthy as the loaded calibration profile and its coverage.",
        implication: "Inspect calibration summaries, residuals, extrapolation warnings, and gate violations before relying on rankings.",
    },
    TrustBoundaryNote {
        code: "ignored_network_congestion",
        scope: "network",
        label: "NIC/link bandwidth, latency, route sharing, and rail locality are modeled, but packet-level congestion effects are not.",
        implication: "Incast, outcast, PFC, head-of-line blocking, and switch-buffer behavior need external calibration or a future congestion model.",
    },
    TrustBoundaryNote {
        code: "unsupported_locality_detail",
        scope: "locality",
        label: "PCIe, NUMA, copy-engine, GPUDirect, and backend-specific locality details are represented through configured resources and route records.",
        implication: "Detailed host/device transfer paths should be encoded as topology overrides and checked against route-path artifacts.",
    },
    TrustBoundaryNote {
        code: "serving_stack_approximation",
        scope: "runtime",
        label: "Runtime behavior is captured through serving stack/features, coarse service models, and calibration evidence, not backend-specific kernel emulation.",
        implication: "Use approximation policy gates for uncalibrated stacks or features that may dominate latency.",
    },
];

fn write_trust_boundary_text<W: Write>(writer: &mut W) -> Result<(), io::Error> {
    writeln!(
        writer,
        "trust_boundary=v1_approximate assumptions={}",
        TRUST_BOUNDARY_NOTES
            .iter()
            .map(|note| note.code)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn write_trust_boundary_json<W: Write>(
    writer: &mut W,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"trust_boundary\": {{")?;
    writeln!(writer, "{indent}  \"status\": \"v1_approximate\",")?;
    writeln!(
        writer,
        "{indent}  \"summary\": {},",
        json_string(
            "Useful for relative planning and calibration workflows; not a proof of production latency or fine-grained locality behavior."
        )
    )?;
    writeln!(writer, "{indent}  \"assumptions\": [")?;
    for (idx, note) in TRUST_BOUNDARY_NOTES.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(note.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(note.scope)
        )?;
        writeln!(
            writer,
            "{indent}      \"label\": {},",
            json_string(note.label)
        )?;
        writeln!(
            writer,
            "{indent}      \"implication\": {}",
            json_string(note.implication)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < TRUST_BOUNDARY_NOTES.len())
        )?;
    }
    writeln!(writer, "{indent}  ]")?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}

fn write_json_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredParallelismConfig],
    options: JsonOutputOptions<'_>,
) -> Result<(), io::Error> {
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema_version\": 1,")?;
    writeln!(writer, "  \"mode\": \"parallelism\",")?;
    writeln!(writer, "  \"cluster_gpus\": {},", cluster.total_gpus())?;
    write_cluster_inventory(writer, cluster, "  ", true)?;
    writeln!(
        writer,
        "  \"model_params_gb\": {},",
        json_f64(model.parameters.as_gigabytes())
    )?;
    writeln!(
        writer,
        "  \"model_parameter_count_billion\": {},",
        json_f64(model.parameter_count_billion())
    )?;
    writeln!(
        writer,
        "  \"searched_configs\": {},",
        options.searched_candidate_count
    )?;
    writeln!(writer, "  \"reported_configs\": {},", results.len())?;
    writeln!(
        writer,
        "  \"omitted_rejected_configs\": {},",
        options.omitted_rejected_candidate_count
    )?;
    write_search_budget(writer, options.search_budget, "  ", true)?;
    write_search_diagnostics(writer, options.search_diagnostics, "  ", true)?;
    write_calibration(
        writer,
        CalibrationJsonContext {
            calibration: options.calibration,
            policy: options.calibration_policy,
            profile: options.calibration_profile,
            coverage: options.calibration_coverage,
            warnings: options.calibration_warnings,
            invalid_shape_warnings: options.calibration_invalid_shape_warnings,
            gate_violations: options.calibration_gate_violations,
        },
        "  ",
        true,
    )?;
    write_approximation_policy(writer, options.approximation_policy, "  ", true)?;
    write_trust_boundary_json(writer, "  ", true)?;
    writeln!(writer, "  \"results\": [")?;
    let nominal_ranks = parallelism_nominal_rank_map(results);
    let uncertainty_adjusted_ranks = parallelism_uncertainty_adjusted_rank_map(
        results,
        options.calibration_policy.uncertainty_ranking_weight,
    );
    for (idx, score) in results.iter().take(options.top_k).enumerate() {
        let candidate_id = parallelism_candidate_id(score);
        write_parallelism_score_json(
            writer,
            cluster,
            idx + 1,
            RankSensitivity {
                nominal_rank: *nominal_ranks.get(&candidate_id).unwrap_or(&(idx + 1)),
                uncertainty_adjusted_rank: *uncertainty_adjusted_ranks
                    .get(&candidate_id)
                    .unwrap_or(&(idx + 1)),
            },
            score,
            options,
            "    ",
        )?;
        if idx + 1 < results.iter().take(options.top_k).count() {
            writeln!(writer, ",")?;
        } else {
            writeln!(writer)?;
        }
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")
}

fn write_serving_json_results<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    model: &ModelSpec,
    results: &[ScoredServingConfig],
    options: JsonOutputOptions<'_>,
) -> Result<(), io::Error> {
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema_version\": 1,")?;
    writeln!(writer, "  \"mode\": \"serving\",")?;
    writeln!(writer, "  \"cluster_gpus\": {},", cluster.total_gpus())?;
    write_cluster_inventory(writer, cluster, "  ", true)?;
    writeln!(
        writer,
        "  \"model_params_gb\": {},",
        json_f64(model.parameters.as_gigabytes())
    )?;
    writeln!(
        writer,
        "  \"model_parameter_count_billion\": {},",
        json_f64(model.parameter_count_billion())
    )?;
    writeln!(
        writer,
        "  \"searched_serving_pairs\": {},",
        options.searched_candidate_count
    )?;
    writeln!(writer, "  \"reported_serving_pairs\": {},", results.len())?;
    writeln!(
        writer,
        "  \"omitted_rejected_serving_pairs\": {},",
        options.omitted_rejected_candidate_count
    )?;
    write_search_budget(writer, options.search_budget, "  ", true)?;
    write_search_diagnostics(writer, options.search_diagnostics, "  ", true)?;
    writeln!(
        writer,
        "  \"objective\": {},",
        json_string(options.serving_objective.unwrap_or_default().as_str())
    )?;
    writeln!(
        writer,
        "  \"serving_stack\": {},",
        json_optional_string(options.serving_stack)
    )?;
    write_string_vec(
        writer,
        "  ",
        "serving_runtime_features",
        options.serving_runtime_features,
        true,
    )?;
    write_calibration(
        writer,
        CalibrationJsonContext {
            calibration: options.calibration,
            policy: options.calibration_policy,
            profile: options.calibration_profile,
            coverage: options.calibration_coverage,
            warnings: options.calibration_warnings,
            invalid_shape_warnings: options.calibration_invalid_shape_warnings,
            gate_violations: options.calibration_gate_violations,
        },
        "  ",
        true,
    )?;
    write_approximation_policy(writer, options.approximation_policy, "  ", true)?;
    write_trust_boundary_json(writer, "  ", true)?;
    write_serving_rejection_summary_json(writer, results, "  ", true)?;
    writeln!(writer, "  \"results\": [")?;
    let nominal_ranks = serving_nominal_rank_map(results);
    let uncertainty_adjusted_ranks = serving_uncertainty_adjusted_rank_map(
        results,
        options.calibration_policy.uncertainty_ranking_weight,
    );
    for (idx, score) in results.iter().take(options.top_k).enumerate() {
        let candidate_id = score.candidate_id.clone();
        write_serving_score_json(
            writer,
            cluster,
            idx + 1,
            RankSensitivity {
                nominal_rank: *nominal_ranks.get(&candidate_id).unwrap_or(&(idx + 1)),
                uncertainty_adjusted_rank: *uncertainty_adjusted_ranks
                    .get(&candidate_id)
                    .unwrap_or(&(idx + 1)),
            },
            score,
            options,
            "    ",
        )?;
        if idx + 1 < results.iter().take(options.top_k).count() {
            writeln!(writer, ",")?;
        } else {
            writeln!(writer)?;
        }
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")
}

fn write_parallelism_score_json<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    rank: usize,
    rank_sensitivity: RankSensitivity,
    score: &ScoredParallelismConfig,
    options: JsonOutputOptions<'_>,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}{{")?;
    let calibration_uncertainty = calibration_uncertainty_summary(score.calibration_fits.iter());
    writeln!(writer, "{indent}  \"rank\": {rank},")?;
    writeln!(
        writer,
        "{indent}  \"nominal_rank\": {},",
        rank_sensitivity.nominal_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_adjusted_rank\": {},",
        rank_sensitivity.uncertainty_adjusted_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_rank_delta\": {},",
        rank_sensitivity.delta()
    )?;
    writeln!(
        writer,
        "{indent}  \"candidate_id\": {},",
        json_string(&parallelism_candidate_id(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"status\": \"{}\",",
        status(score.feasible)
    )?;
    writeln!(writer, "{indent}  \"feasible\": {},", score.feasible)?;
    write_config(writer, &score.config, indent, "config", true)?;
    write_placement(writer, cluster, &score.placement, indent, "placement", true)?;
    write_placement_evidence(
        writer,
        indent,
        "placement_evidence",
        &score.placement_evidence,
        true,
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_ms\": {},",
        json_optional_ms(score.feasible, score.estimated_latency_s)
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_calibration_uncertainty_ms\": {},",
        json_optional_uncertainty_ms(
            score.feasible,
            calibration_uncertainty.absolute_uncertainty_s
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_calibration_lower_ms\": {},",
        json_metric_lower_ms(
            score.feasible,
            score.estimated_latency_s,
            &calibration_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_calibration_upper_ms\": {},",
        json_metric_upper_ms(
            score.feasible,
            score.estimated_latency_s,
            &calibration_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_latency_uncertainty_adjusted_ms\": {},",
        json_optional_ms(
            score.feasible,
            uncertainty_adjusted_parallelism_latency_s(
                score,
                options.calibration_policy.uncertainty_ranking_weight
            )
        )
    )?;
    writeln!(
        writer,
        "{indent}  \"estimated_memory_per_gpu_gb\": {},",
        json_f64(score.estimated_memory_per_gpu.as_gigabytes())
    )?;
    writeln!(
        writer,
        "{indent}  \"operation_makespan_ms\": {},",
        json_optional_ms(score.feasible, score.operation_makespan_s)
    )?;
    write_calibration_fit_applications(
        writer,
        indent,
        "calibration_fit_applications",
        &score.calibration_fits,
        true,
    )?;
    write_calibration_uncertainty(
        writer,
        indent,
        "calibration_uncertainty",
        &score.calibration_fits,
        true,
    )?;
    write_calibration_gate_violation_array(
        writer,
        &score.calibration_gate_violations,
        indent,
        "calibration_gate_violations",
        true,
    )?;
    write_approximations(writer, indent, &score.approximations, true)?;
    write_approximation_policy_violations(
        writer,
        indent,
        &score.approximation_policy_violations,
        true,
    )?;
    write_resource_utilization(writer, indent, &score.resource_utilization, true)?;
    if options.include_occupancy {
        let resources = selected_occupancy_resources(
            &score.resource_utilization,
            options.occupancy_resource_limit,
        );
        let occupancy = resource_occupancy_buckets(
            &score.scheduled_operations,
            score.operation_makespan_s,
            options.occupancy_buckets,
            &resources,
        );
        write_resource_occupancy(writer, indent, &occupancy, true)?;
    }
    if options.include_critical_path {
        let path = critical_path(&score.scheduled_operations);
        write_critical_path(writer, indent, &path, options.critical_path_limit, true)?;
    }
    write_string_vec(writer, indent, "bottlenecks", &score.bottlenecks, true)?;
    write_optional_string(
        writer,
        indent,
        "rejected_reason",
        score.rejected_reason.as_deref(),
        options.include_trace,
    )?;
    if options.include_trace {
        write_scheduled_operations(
            writer,
            indent,
            "scheduled_operations",
            &score.scheduled_operations,
            options.trace_limit,
            false,
        )?;
    }
    write!(writer, "{indent}}}")
}

fn write_serving_score_json<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    rank: usize,
    rank_sensitivity: RankSensitivity,
    score: &ScoredServingConfig,
    options: JsonOutputOptions<'_>,
    indent: &str,
) -> Result<(), io::Error> {
    let candidate_id = score.candidate_id.clone();
    writeln!(writer, "{indent}{{")?;
    writeln!(writer, "{indent}  \"rank\": {rank},")?;
    writeln!(
        writer,
        "{indent}  \"nominal_rank\": {},",
        rank_sensitivity.nominal_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_adjusted_rank\": {},",
        rank_sensitivity.uncertainty_adjusted_rank
    )?;
    writeln!(
        writer,
        "{indent}  \"uncertainty_rank_delta\": {},",
        rank_sensitivity.delta()
    )?;
    writeln!(
        writer,
        "{indent}  \"candidate_id\": {},",
        json_string(&candidate_id)
    )?;
    writeln!(
        writer,
        "{indent}  \"status\": \"{}\",",
        status(score.feasible)
    )?;
    writeln!(writer, "{indent}  \"feasible\": {},", score.feasible)?;
    writeln!(
        writer,
        "{indent}  \"objective\": {},",
        json_string(score.objective.as_str())
    )?;
    writeln!(
        writer,
        "{indent}  \"objective_base_score\": {},",
        json_optional_f64(serving_objective_base_score(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"pareto_frontier\": {},",
        score.pareto.is_frontier
    )?;
    writeln!(
        writer,
        "{indent}  \"pareto_rank\": {},",
        json_optional_u32(score.pareto.rank)
    )?;
    write_string_vec(
        writer,
        indent,
        "pareto_dominated_by",
        &score.pareto.dominated_by,
        true,
    )?;
    write_pareto_dimensions(writer, indent, &score.pareto.dimensions, true)?;
    writeln!(
        writer,
        "{indent}  \"memory_pressure_peak_fraction\": {},",
        json_optional_f64(serving_peak_memory_pressure_fraction(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"memory_pressure_peak_phase\": {},",
        serving_peak_memory_pressure_phase(score)
            .map(json_string)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}  \"max_memory_pressure_fraction\": {},",
        json_optional_value(score.max_memory_pressure_fraction)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_unique_gpus\": {},",
        json_optional_u32(score.max_unique_gpus)
    )?;
    writeln!(
        writer,
        "{indent}  \"min_throughput_tokens_per_s\": {},",
        json_optional_value(score.min_throughput_tokens_per_s)
    )?;
    write_serving_metric_ceilings_json(writer, indent, score.metric_ceilings)?;
    write_kv_route_constraints_json(writer, indent, score.kv_route_constraints)?;
    writeln!(
        writer,
        "{indent}  \"slo_miss_penalty_weight\": {},",
        json_f64(score.slo_miss_penalty_weight)
    )?;
    write_slo_miss_penalty_weights_json(writer, score, indent)?;
    write_slo_miss_penalty_components_json(writer, score, indent)?;
    write_traffic_class_slo_miss_penalties_json(writer, score, indent)?;
    writeln!(
        writer,
        "{indent}  \"slo_miss_penalty_score\": {},",
        json_optional_f64(score.slo_miss_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}  \"service_backpressure_penalty_weight\": {},",
        json_f64(score.service_backpressure_penalty_weight)
    )?;
    writeln!(
        writer,
        "{indent}  \"service_backpressure_penalty_score\": {},",
        json_optional_f64(score.service_backpressure_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}  \"topology_risk_penalty_weight\": {},",
        json_f64(score.topology_risk_penalty_weight)
    )?;
    writeln!(
        writer,
        "{indent}  \"topology_risk_penalty_score\": {},",
        json_optional_f64(score.topology_risk_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}  \"objective_nominal_score\": {},",
        json_optional_f64(nominal_serving_objective_score(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"objective_uncertainty_adjusted_score\": {},",
        json_optional_f64(uncertainty_adjusted_serving_objective_score(
            score,
            options.calibration_policy.uncertainty_ranking_weight
        ))
    )?;
    write_serving_objective_breakdown(
        writer,
        score,
        options.calibration_policy.uncertainty_ranking_weight,
        indent,
        true,
    )?;
    writeln!(
        writer,
        "{indent}  \"pool\": {},",
        json_string(&pool_label(score))
    )?;
    writeln!(
        writer,
        "{indent}  \"deployment_mode\": {},",
        json_string(score.deployment_mode.as_str())
    )?;
    write_route_coverage(writer, indent, score, true)?;
    write_pool_search_summary(writer, indent, score.pool_search_summary.as_ref(), true)?;
    write_pool_topology_summary(writer, indent, &score.pool_topology, true)?;
    write_u32_vec(writer, indent, "prefill_nodes", &score.prefill_nodes, true)?;
    write_u32_vec(writer, indent, "decode_nodes", &score.decode_nodes, true)?;
    write_string_vec(
        writer,
        indent,
        "prefill_gpu_labels",
        &score.prefill_gpu_labels,
        true,
    )?;
    write_string_vec(
        writer,
        indent,
        "decode_gpu_labels",
        &score.decode_gpu_labels,
        true,
    )?;
    write_serving_hardware_footprint(writer, indent, &score.hardware_footprint, true)?;
    write_serving_cost_estimate(writer, indent, &score.cost_estimate, true)?;
    write_config(
        writer,
        &score.prefill_config,
        indent,
        "prefill_config",
        true,
    )?;
    write_config(writer, &score.decode_config, indent, "decode_config", true)?;
    write_placement(
        writer,
        cluster,
        &score.prefill_score.placement,
        indent,
        "prefill_placement",
        true,
    )?;
    write_placement_evidence(
        writer,
        indent,
        "prefill_placement_evidence",
        &score.prefill_score.placement_evidence,
        true,
    )?;
    write_placement(
        writer,
        cluster,
        &score.decode_score.placement,
        indent,
        "decode_placement",
        true,
    )?;
    write_placement_evidence(
        writer,
        indent,
        "decode_placement_evidence",
        &score.decode_score.placement_evidence,
        true,
    )?;
    write_serving_memory(
        writer,
        indent,
        "prefill_memory",
        &score.prefill_memory,
        true,
    )?;
    write_serving_memory(writer, indent, "decode_memory", &score.decode_memory, true)?;
    write_serving_metrics(
        writer,
        score.feasible,
        &score.metrics,
        &score.calibration_fits,
        indent,
        true,
    )?;
    write_measurement_window(
        writer,
        score.feasible,
        &score.measurement_window,
        indent,
        true,
    )?;
    write_serving_bottleneck_summary(writer, indent, &score.bottleneck_summary, true)?;
    write_memory_pressure_observations(
        writer,
        indent,
        &score.memory_pressure,
        options.request_limit,
        true,
    )?;
    write_serving_calibration_summary(writer, indent, &score.calibration_summary, true)?;
    write_calibration_fit_applications(
        writer,
        indent,
        "calibration_fit_applications",
        &score.calibration_fits,
        true,
    )?;
    write_serving_phase_calibration(writer, indent, &score.phase_calibration, true)?;
    write_calibration_uncertainty(
        writer,
        indent,
        "calibration_uncertainty",
        &score.calibration_fits,
        true,
    )?;
    write_calibration_gate_violation_array(
        writer,
        &score.calibration_gate_violations,
        indent,
        "calibration_gate_violations",
        true,
    )?;
    write_serving_approximation_summary(writer, indent, &score.approximation_summary, true)?;
    write_approximations(writer, indent, &score.approximations, true)?;
    write_approximation_policy_violations(
        writer,
        indent,
        &score.approximation_policy_violations,
        true,
    )?;
    write_request_observations(writer, indent, score, options.request_limit, true)?;
    write_decode_iterations(
        writer,
        indent,
        &score.decode_iterations,
        if options.include_trace {
            options.trace_limit
        } else {
            options.request_limit
        },
        options.include_trace,
        true,
    )?;
    write_metric_breakdowns(writer, indent, &score.metric_breakdowns, true)?;
    write_node_capacity(writer, indent, &score.node_capacity, true)?;
    write_gpu_capacity(writer, indent, &score.gpu_capacity, true)?;
    write_traffic_class_capacity(writer, indent, &score.traffic_class_capacity, true)?;
    write_service_observations(writer, indent, &score.service_observations, true)?;
    write_worker_observations(writer, indent, &score.worker_observations, true)?;
    write_resource_utilization(writer, indent, &score.resource_utilization, true)?;
    write_phase_resource_utilization(writer, indent, &score.phase_resource_utilization, true)?;
    write_kv_route_topology_summary(writer, indent, &score.kv_route_topology_summary, true)?;
    write_topology_bottlenecks(writer, indent, &score.topology_bottlenecks, true)?;
    write_kv_route_resource_summary(writer, indent, &score.kv_route_resource_summary, true)?;
    if options.include_occupancy {
        let resources = selected_occupancy_resources(
            &score.resource_utilization,
            options.occupancy_resource_limit,
        );
        let occupancy = resource_occupancy_buckets(
            &score.scheduled_operations,
            score.metrics.scheduled_makespan_s,
            options.occupancy_buckets,
            &resources,
        );
        write_resource_occupancy(writer, indent, &occupancy, true)?;
    }
    if options.include_critical_path {
        let path = critical_path(&score.scheduled_operations);
        write_critical_path(writer, indent, &path, options.critical_path_limit, true)?;
    }
    write_string_vec(writer, indent, "bottlenecks", &score.bottlenecks, true)?;
    write_serving_rejections(writer, indent, &candidate_id, &score.rejections, true)?;
    write_optional_string(
        writer,
        indent,
        "rejected_reason",
        score.rejected_reason.as_deref(),
        options.include_trace,
    )?;
    if options.include_trace {
        write_scheduled_operations(
            writer,
            indent,
            "scheduled_operations",
            &score.scheduled_operations,
            options.trace_limit,
            false,
        )?;
    }
    write!(writer, "{indent}}}")
}

fn write_serving_metric_ceilings_json<W: Write>(
    writer: &mut W,
    indent: &str,
    ceilings: ServingMetricCeilings,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"metric_ceilings\": {{")?;
    writeln!(
        writer,
        "{indent}    \"max_ttft_s\": {},",
        json_optional_value(ceilings.max_ttft_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_tpot_s\": {},",
        json_optional_value(ceilings.max_tpot_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_itl_s\": {},",
        json_optional_value(ceilings.max_itl_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"max_e2el_s\": {}",
        json_optional_value(ceilings.max_e2el_s)
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

fn write_serving_objective_breakdown<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    uncertainty_ranking_weight: f64,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let descriptor = serving_objective_metric_descriptor(score);
    let base_score = serving_objective_base_score(score);
    let uncertainty_adjusted_base_score =
        serving_uncertainty_adjusted_base_score(score, uncertainty_ranking_weight);
    let nominal_score = nominal_serving_objective_score(score);
    let uncertainty_adjusted_score =
        uncertainty_adjusted_serving_objective_score(score, uncertainty_ranking_weight);

    writeln!(writer, "{indent}  \"objective_breakdown\": {{")?;
    writeln!(
        writer,
        "{indent}    \"selected_objective\": {},",
        json_string(score.objective.as_str())
    )?;
    writeln!(
        writer,
        "{indent}    \"score_convention\": \"lower_is_better\","
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric\": {},",
        json_string(descriptor.metric)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric_direction\": {},",
        json_string(descriptor.direction)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric_unit\": {},",
        json_string(descriptor.unit)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_metric_value\": {},",
        json_optional_f64(descriptor.value)
    )?;
    writeln!(
        writer,
        "{indent}    \"base_score\": {},",
        json_optional_f64(base_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"slo_miss_penalty_score\": {},",
        json_optional_f64(score.slo_miss_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"service_backpressure_penalty_score\": {},",
        json_optional_f64(score.service_backpressure_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"topology_risk_penalty_score\": {},",
        json_optional_f64(score.topology_risk_penalty_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"nominal_penalty_score\": {},",
        json_optional_f64(serving_objective_penalty_score(score))
    )?;
    writeln!(
        writer,
        "{indent}    \"nominal_score\": {},",
        json_optional_f64(nominal_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_ranking_weight\": {},",
        json_optional_f64(uncertainty_ranking_weight)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_adjusted_base_score\": {},",
        json_optional_f64(uncertainty_adjusted_base_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_adjusted_score\": {},",
        json_optional_f64(uncertainty_adjusted_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"uncertainty_adjusted_delta\": {},",
        json_optional_f64(uncertainty_adjusted_score - nominal_score)
    )?;
    writeln!(
        writer,
        "{indent}    \"largest_nominal_term\": {}",
        json_string(serving_largest_nominal_objective_term(score))
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_kv_route_constraints_json<W: Write>(
    writer: &mut W,
    indent: &str,
    constraints: ServingKvRouteConstraints,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"kv_route_constraints\": {{")?;
    writeln!(
        writer,
        "{indent}    \"min_inter_node_rail_count\": {},",
        json_optional_u32(constraints.min_inter_node_rail_count)
    )?;
    writeln!(
        writer,
        "{indent}    \"require_inter_node_rail_metadata\": {},",
        constraints.require_inter_node_rail_metadata
    )?;
    writeln!(
        writer,
        "{indent}    \"require_gpudirect\": {}",
        constraints.require_gpudirect
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

fn write_slo_miss_penalty_weights_json<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"slo_miss_penalty_weights\": {{")?;
    write_slo_miss_penalty_weights_fields_json(
        writer,
        score.slo_miss_penalty_weights,
        indent,
        "    ",
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

fn write_route_coverage<W: Write>(
    writer: &mut W,
    indent: &str,
    score: &ScoredServingConfig,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"route_coverage\": {{")?;
    writeln!(
        writer,
        "{indent}    \"candidate_count\": {},",
        score.route_coverage.candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"routable_candidate_count\": {},",
        score.route_coverage.routable_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"unroutable_candidate_count\": {},",
        score.route_coverage.unroutable_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"fraction\": {}",
        json_f64(score.route_coverage.fraction)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_pool_search_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: Option<&ServingPoolSearchSummary>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(summary) = summary else {
        return writeln!(
            writer,
            "{indent}  \"pool_search_summary\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}  \"pool_search_summary\": {{")?;
    writeln!(
        writer,
        "{indent}    \"max_candidates\": {},",
        summary.max_candidates
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_candidate_count\": {},",
        summary.generated_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_colocated_count\": {},",
        summary.generated_colocated_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_partially_disaggregated_count\": {},",
        summary.generated_partially_disaggregated_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"generated_fully_disaggregated_count\": {},",
        summary.generated_fully_disaggregated_count()
    )?;
    writeln!(writer, "{indent}    \"truncated\": {},", summary.truncated)?;
    writeln!(
        writer,
        "{indent}    \"considered_candidate_count\": {},",
        summary.considered_candidate_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_overlap_count\": {},",
        summary.rejected_overlap_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_mode_count\": {},",
        summary.rejected_mode_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_domain_spread_count\": {},",
        summary.rejected_domain_spread_count()
    )?;
    writeln!(
        writer,
        "{indent}    \"duplicate_candidate_count\": {},",
        summary.duplicate_candidate_count()
    )?;
    writeln!(writer, "{indent}    \"groups\": [")?;
    for (idx, group) in summary.groups.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"prefill_group\": {},",
            json_string(&group.prefill_group)
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_group\": {},",
            json_string(&group.decode_group)
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_group_node_count\": {},",
            group.prefill_group_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_node_filter_node_count\": {},",
            group.prefill_node_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_gpu_filter_node_count\": {},",
            group.prefill_gpu_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_group_node_count\": {},",
            group.decode_group_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_node_filter_node_count\": {},",
            group.decode_node_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_gpu_filter_node_count\": {},",
            group.decode_gpu_filter_node_count
        )?;
        writeln!(
            writer,
            "{indent}        \"prefill_node_counts\": [{}],",
            u32_list(&group.prefill_node_counts)
        )?;
        writeln!(
            writer,
            "{indent}        \"decode_node_counts\": [{}],",
            u32_list(&group.decode_node_counts)
        )?;
        writeln!(
            writer,
            "{indent}        \"considered_candidate_count\": {},",
            group.considered_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}        \"rejected_overlap_count\": {},",
            group.rejected_overlap_count
        )?;
        writeln!(
            writer,
            "{indent}        \"rejected_mode_count\": {},",
            group.rejected_mode_count
        )?;
        writeln!(
            writer,
            "{indent}        \"rejected_domain_spread_count\": {},",
            group.rejected_domain_spread_count
        )?;
        writeln!(
            writer,
            "{indent}        \"duplicate_candidate_count\": {},",
            group.duplicate_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_candidate_count\": {},",
            group.generated_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_colocated_count\": {},",
            group.generated_colocated_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_partially_disaggregated_count\": {},",
            group.generated_partially_disaggregated_count
        )?;
        writeln!(
            writer,
            "{indent}        \"generated_fully_disaggregated_count\": {}",
            group.generated_fully_disaggregated_count
        )?;
        if idx + 1 < summary.groups.len() {
            writeln!(writer, "{indent}      }},")?;
        } else {
            writeln!(writer, "{indent}      }}")?;
        }
    }
    writeln!(writer, "{indent}    ]")?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_pool_topology_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &ServingPoolTopologySummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"pool_topology\": {{")?;
    writeln!(
        writer,
        "{indent}    \"prefill_node_count\": {},",
        summary.prefill_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_node_count\": {},",
        summary.decode_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"shared_node_count\": {},",
        summary.shared_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"dedicated_prefill_node_count\": {},",
        summary.dedicated_prefill_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"dedicated_decode_node_count\": {},",
        summary.dedicated_decode_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_racks\": {},",
        json_string_array(&summary.prefill_racks)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_racks\": {},",
        json_string_array(&summary.decode_racks)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_islands\": {},",
        json_string_array(&summary.prefill_islands)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_islands\": {},",
        json_string_array(&summary.decode_islands)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_failure_domains\": {},",
        json_string_array(&summary.prefill_failure_domains)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_failure_domains\": {},",
        json_string_array(&summary.decode_failure_domains)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_node_labels\": {},",
        json_string_array(&summary.prefill_node_labels)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_node_labels\": {}",
        json_string_array(&summary.decode_node_labels)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_slo_miss_penalty_components_json<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"slo_miss_penalty_components\": {{")?;
    write_slo_miss_penalty_components_fields_json(
        writer,
        score.slo_miss_penalty_components,
        indent,
        "    ",
    )?;
    writeln!(writer, "{indent}  }},")?;
    Ok(())
}

fn write_slo_miss_penalty_weights_object_json<W: Write>(
    writer: &mut W,
    weights: ServingSloMissPenaltyWeights,
    indent: &str,
    inner: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}{inner}\"weights\": {{")?;
    let nested = format!("{inner}  ");
    write_slo_miss_penalty_weights_fields_json(writer, weights, indent, &nested)?;
    writeln!(writer, "{indent}{inner}}},")?;
    Ok(())
}

fn write_slo_miss_penalty_weights_fields_json<W: Write>(
    writer: &mut W,
    weights: ServingSloMissPenaltyWeights,
    indent: &str,
    inner: &str,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}{inner}\"aggregate\": {},",
        json_f64(weights.aggregate)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"ttft\": {},",
        json_f64(weights.ttft)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"tpot\": {},",
        json_f64(weights.tpot)
    )?;
    writeln!(writer, "{indent}{inner}\"itl\": {},", json_f64(weights.itl))?;
    writeln!(
        writer,
        "{indent}{inner}\"e2el\": {},",
        json_f64(weights.e2el)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"deadline\": {}",
        json_f64(weights.deadline)
    )?;
    Ok(())
}

fn write_slo_miss_penalty_components_object_json<W: Write>(
    writer: &mut W,
    components: ServingSloMissPenaltyComponents,
    indent: &str,
    inner: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}{inner}\"components\": {{")?;
    let nested = format!("{inner}  ");
    write_slo_miss_penalty_components_fields_json(writer, components, indent, &nested)?;
    writeln!(writer, "{indent}{inner}}}{}", comma(trailing_comma))?;
    Ok(())
}

fn write_slo_miss_penalty_components_fields_json<W: Write>(
    writer: &mut W,
    components: ServingSloMissPenaltyComponents,
    indent: &str,
    inner: &str,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}{inner}\"ttft\": {},",
        json_optional_f64(components.ttft)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"tpot\": {},",
        json_optional_f64(components.tpot)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"itl\": {},",
        json_optional_f64(components.itl)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"e2el\": {},",
        json_optional_f64(components.e2el)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"deadline\": {},",
        json_optional_f64(components.deadline)
    )?;
    writeln!(
        writer,
        "{indent}{inner}\"total\": {}",
        json_optional_f64(components.total)
    )?;
    Ok(())
}

fn write_traffic_class_slo_miss_penalties_json<W: Write>(
    writer: &mut W,
    score: &ScoredServingConfig,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"traffic_class_slo_miss_penalties\": [")?;
    for (idx, penalty) in score.traffic_class_slo_miss_penalties.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&penalty.name)
        )?;
        writeln!(
            writer,
            "{indent}      \"group\": {},",
            json_string(&penalty.group)
        )?;
        writeln!(
            writer,
            "{indent}      \"key\": {},",
            json_string(&penalty.key)
        )?;
        write_slo_miss_penalty_weights_object_json(writer, penalty.weights, indent, "      ")?;
        write_slo_miss_penalty_components_object_json(
            writer,
            penalty.components,
            indent,
            "      ",
            false,
        )?;
        if idx + 1 < score.traffic_class_slo_miss_penalties.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ],")?;
    Ok(())
}

fn write_config<W: Write>(
    writer: &mut W,
    config: &ParallelismConfig,
    indent: &str,
    field: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": {{")?;
    writeln!(
        writer,
        "{indent}    \"tensor_ranks\": {},",
        config.tensor_ranks
    )?;
    writeln!(
        writer,
        "{indent}    \"pipeline_ranks\": {},",
        config.pipeline_ranks
    )?;
    writeln!(
        writer,
        "{indent}    \"expert_ranks\": {},",
        config.expert_ranks
    )?;
    writeln!(writer, "{indent}    \"data_ranks\": {},", config.data_ranks)?;
    writeln!(
        writer,
        "{indent}    \"total_ranks\": {}",
        config.total_ranks()
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_placement<W: Write>(
    writer: &mut W,
    cluster: &Cluster,
    placement: &RankPlacement,
    indent: &str,
    field: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, addr) in placement.rank_to_gpu.iter().enumerate() {
        let gpu_label = cluster
            .gpu_profile(*addr)
            .map(|profile| profile.label)
            .unwrap_or("unknown");
        writeln!(writer, "{indent}    {{")?;
        writeln!(writer, "{indent}      \"rank\": {},", idx)?;
        writeln!(writer, "{indent}      \"node_id\": {},", addr.node_id)?;
        writeln!(
            writer,
            "{indent}      \"local_gpu_id\": {},",
            addr.local_gpu_id
        )?;
        writeln!(writer, "{indent}      \"gpu\": {}", json_string(gpu_label))?;
        if idx + 1 < placement.rank_to_gpu.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_placement_evidence<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    evidence_items: &[PlacementEvidence],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, evidence) in evidence_items.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"decision\": {},",
            json_string(&evidence.decision)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(&evidence.scope)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&evidence.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&evidence.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(evidence.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(evidence.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {},",
            evidence
                .unit
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {},",
            evidence
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {}",
            json_string(&evidence.message)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < evidence_items.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_serving_metrics<W: Write>(
    writer: &mut W,
    feasible: bool,
    metrics: &ServingMetrics,
    calibration_fits: &[CalibrationFitApplication],
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let overall_uncertainty = calibration_uncertainty_summary(calibration_fits.iter());
    let prefill_uncertainty =
        calibration_phase_uncertainty_summary(calibration_fits.iter(), "prefill");
    let decode_uncertainty =
        calibration_phase_uncertainty_summary(calibration_fits.iter(), "decode");
    let kv_transfer_uncertainty =
        calibration_phase_uncertainty_summary(calibration_fits.iter(), "kv_transfer");
    writeln!(writer, "{indent}  \"metrics\": {{")?;
    writeln!(
        writer,
        "{indent}    \"ttft_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.ttft_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.ttft_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.ttft_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p50_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p90_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p95_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_p99_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_max_ms\": {},",
        json_optional_ms(feasible, metrics.ttft_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_slo_constrained_requests\": {},",
        metrics.ttft_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_slo_missed_requests\": {},",
        metrics.ttft_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"ttft_slo_miss_rate\": {},",
        json_optional_f64(metrics.ttft_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.tpot_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.tpot_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.tpot_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p50_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p90_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p95_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_p99_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_max_ms\": {},",
        json_optional_ms(feasible, metrics.tpot_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_slo_constrained_requests\": {},",
        metrics.tpot_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_slo_missed_requests\": {},",
        metrics.tpot_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"tpot_slo_miss_rate\": {},",
        json_optional_f64(metrics.tpot_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_ms\": {},",
        json_optional_ms(feasible, metrics.itl_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.itl_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.itl_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.itl_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p50_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p90_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p95_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_p99_ms\": {},",
        json_optional_ms(feasible, metrics.itl_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_max_ms\": {},",
        json_optional_ms(feasible, metrics.itl_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_slo_constrained_requests\": {},",
        metrics.itl_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_slo_missed_requests\": {},",
        metrics.itl_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"itl_slo_miss_rate\": {},",
        json_optional_f64(metrics.itl_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iterations\": {},",
        metrics.decode_iterations
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.decode_iteration_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.decode_iteration_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.decode_iteration_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p50_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p90_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p95_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_p99_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_iteration_max_ms\": {},",
        json_optional_ms(feasible, metrics.decode_iteration_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s\": {},",
        json_optional_f64(metrics.throughput_tokens_per_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_calibration_uncertainty_tokens_per_s\": {},",
        json_metric_relative_uncertainty_value(
            feasible,
            metrics.throughput_tokens_per_s,
            &overall_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_calibration_lower_tokens_per_s\": {},",
        json_metric_relative_lower_value(
            feasible,
            metrics.throughput_tokens_per_s,
            &overall_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_calibration_upper_tokens_per_s\": {},",
        json_metric_relative_upper_value(
            feasible,
            metrics.throughput_tokens_per_s,
            &overall_uncertainty
        )
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.e2el_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.e2el_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.e2el_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p50_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p50_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p90_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p95_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_p99_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_p99_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_max_ms\": {},",
        json_optional_ms(feasible, metrics.e2el_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_slo_constrained_requests\": {},",
        metrics.e2el_slo_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_slo_missed_requests\": {},",
        metrics.e2el_slo_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"e2el_slo_miss_rate\": {},",
        json_optional_f64(metrics.e2el_slo_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_miss_rate\": {},",
        json_optional_f64(metrics.deadline_miss_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"service_ms\": {},",
        json_optional_ms(feasible, metrics.service_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_ms\": {},",
        json_optional_ms(feasible, metrics.prefill_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.prefill_s, &prefill_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.prefill_s, &prefill_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.prefill_s, &prefill_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_chunks\": {},",
        metrics.prefill_chunks
    )?;
    writeln!(
        writer,
        "{indent}    \"prompt_tokens\": {},",
        metrics.prompt_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"prefix_cache_hit_tokens\": {},",
        metrics.prefix_cache_hit_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"effective_prefill_tokens\": {},",
        metrics.effective_prefill_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"prefix_cache_hit_rate\": {},",
        json_optional_f64(metrics.prefix_cache_hit_rate)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_ms\": {},",
        json_optional_ms(feasible, metrics.kv_transfer_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.kv_transfer_s, &kv_transfer_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.kv_transfer_s, &kv_transfer_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_transfer_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.kv_transfer_s, &kv_transfer_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_queue_ms\": {},",
        json_optional_ms(feasible, metrics.kv_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_worker_queue_ms\": {},",
        json_optional_ms(feasible, metrics.kv_worker_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_resource_queue_ms\": {},",
        json_optional_ms(feasible, metrics.kv_resource_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_queue_ms\": {},",
        json_optional_ms(feasible, metrics.decode_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_worker_queue_ms\": {},",
        json_optional_ms(feasible, metrics.prefill_worker_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_resource_queue_ms\": {},",
        json_optional_ms(feasible, metrics.prefill_resource_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_worker_queue_ms\": {},",
        json_optional_ms(feasible, metrics.decode_worker_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_resource_queue_ms\": {},",
        json_optional_ms(feasible, metrics.decode_resource_queue_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_ms\": {},",
        json_optional_ms(feasible, metrics.decode_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.decode_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.decode_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.decode_s, &decode_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_p90_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_p90_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_p95_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_p95_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"queue_delay_max_ms\": {},",
        json_optional_ms(feasible, metrics.queue_delay_max_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_prefill_tokens\": {},",
        metrics.peak_prefill_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_prefill_tokens_per_node\": {},",
        metrics.peak_prefill_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_prefill_tokens_per_gpu\": {},",
        metrics.peak_prefill_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_decode_sequences\": {},",
        metrics.peak_decode_sequences
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_resident_tokens\": {},",
        metrics.peak_resident_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_decode_sequences_per_node\": {},",
        metrics.peak_decode_sequences_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_resident_tokens_per_node\": {},",
        metrics.peak_resident_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_decode_sequences_per_gpu\": {},",
        metrics.peak_decode_sequences_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_resident_tokens_per_gpu\": {},",
        metrics.peak_resident_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_blocks\": {},",
        metrics.peak_kv_blocks
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_allocated_kv_tokens\": {},",
        metrics.peak_allocated_kv_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_fragmentation_tokens\": {},",
        metrics.peak_kv_fragmentation_tokens
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_block_table_bytes\": {},",
        metrics.peak_kv_block_table_bytes
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_blocks_per_node\": {},",
        metrics.peak_kv_blocks_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_allocated_kv_tokens_per_node\": {},",
        metrics.peak_allocated_kv_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_fragmentation_tokens_per_node\": {},",
        metrics.peak_kv_fragmentation_tokens_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_block_table_bytes_per_node\": {},",
        metrics.peak_kv_block_table_bytes_per_node
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_blocks_per_gpu\": {},",
        metrics.peak_kv_blocks_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_allocated_kv_tokens_per_gpu\": {},",
        metrics.peak_allocated_kv_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_fragmentation_tokens_per_gpu\": {},",
        metrics.peak_kv_fragmentation_tokens_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"peak_kv_block_table_bytes_per_gpu\": {},",
        metrics.peak_kv_block_table_bytes_per_gpu
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_sequence_utilization\": {},",
        json_optional_f64(metrics.decode_sequence_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"resident_token_utilization\": {},",
        json_optional_f64(metrics.resident_token_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_block_utilization\": {},",
        json_optional_f64(metrics.kv_block_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_sequence_per_node_utilization\": {},",
        json_optional_f64(metrics.decode_sequence_per_node_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"resident_token_per_node_utilization\": {},",
        json_optional_f64(metrics.resident_token_per_node_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_block_per_node_utilization\": {},",
        json_optional_f64(metrics.kv_block_per_node_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_sequence_per_gpu_utilization\": {},",
        json_optional_f64(metrics.decode_sequence_per_gpu_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"resident_token_per_gpu_utilization\": {},",
        json_optional_f64(metrics.resident_token_per_gpu_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"kv_block_per_gpu_utilization\": {},",
        json_optional_f64(metrics.kv_block_per_gpu_utilization)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_ms\": {},",
        json_optional_ms(feasible, metrics.scheduled_makespan_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_calibration_uncertainty_ms\": {},",
        json_metric_uncertainty_ms(feasible, metrics.scheduled_makespan_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_calibration_lower_ms\": {},",
        json_metric_lower_ms(feasible, metrics.scheduled_makespan_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_makespan_calibration_upper_ms\": {},",
        json_metric_upper_ms(feasible, metrics.scheduled_makespan_s, &overall_uncertainty)
    )?;
    writeln!(
        writer,
        "{indent}    \"scheduled_requests\": {},",
        metrics.scheduled_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"admitted_requests\": {},",
        metrics.admitted_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"completed_requests\": {},",
        metrics.completed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_requests\": {},",
        metrics.rejected_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"timed_out_requests\": {},",
        metrics.timed_out_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"cancelled_requests\": {},",
        metrics.cancelled_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_constrained_requests\": {},",
        metrics.deadline_constrained_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_missed_requests\": {},",
        metrics.deadline_missed_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"measured_requests\": {},",
        metrics.measured_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"measurement_start_ms\": {},",
        json_optional_ms(feasible, metrics.measurement_start_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"measurement_end_ms\": {}",
        json_optional_ms(feasible, metrics.measurement_end_s)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_measurement_window<W: Write>(
    writer: &mut W,
    feasible: bool,
    window: &ServingMeasurementWindowObservation,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let candidate_start_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_start_s)
    } else {
        "null".to_string()
    };
    let candidate_end_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_end_s)
    } else {
        "null".to_string()
    };
    let steady_state_min_requests = if feasible {
        json_optional_u32(window.steady_state_min_requests)
    } else {
        "null".to_string()
    };
    let steady_state_max_cv = if feasible {
        json_optional_value(window.steady_state_max_cv)
    } else {
        "null".to_string()
    };
    let candidate_request_count = if feasible {
        json_optional_u32(window.steady_state_candidate_request_count)
    } else {
        "null".to_string()
    };
    let candidate_e2el_mean_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_e2el_mean_s)
    } else {
        "null".to_string()
    };
    let candidate_e2el_stddev_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_e2el_stddev_s)
    } else {
        "null".to_string()
    };
    let candidate_e2el_cv = if feasible {
        json_optional_value(window.steady_state_candidate_e2el_cv)
    } else {
        "null".to_string()
    };
    let candidate_e2el_std_error_ms = if feasible {
        json_optional_seconds_ms(window.steady_state_candidate_e2el_std_error_s)
    } else {
        "null".to_string()
    };
    let candidate_worst_metric = if feasible {
        json_optional_string(window.steady_state_candidate_worst_metric.as_deref())
    } else {
        "null".to_string()
    };
    let candidate_worst_cv = if feasible {
        json_optional_value(window.steady_state_candidate_worst_cv)
    } else {
        "null".to_string()
    };
    let candidate_output_tokens = if feasible {
        json_optional_u64(window.steady_state_candidate_output_tokens)
    } else {
        "null".to_string()
    };
    let candidate_throughput = if feasible {
        json_optional_value(window.steady_state_candidate_throughput_tokens_per_s)
    } else {
        "null".to_string()
    };
    let candidate_worst_utilization_resource = if feasible {
        json_optional_string(
            window
                .steady_state_candidate_worst_utilization_resource
                .as_deref(),
        )
    } else {
        "null".to_string()
    };
    let candidate_worst_utilization_cv = if feasible {
        json_optional_value(window.steady_state_candidate_worst_utilization_cv)
    } else {
        "null".to_string()
    };
    writeln!(writer, "{indent}  \"measurement_window\": {{")?;
    writeln!(
        writer,
        "{indent}    \"source\": {},",
        json_string(&window.source)
    )?;
    writeln!(
        writer,
        "{indent}    \"start_ms\": {},",
        json_optional_ms(feasible, window.start_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"end_ms\": {},",
        json_optional_ms(feasible, window.end_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"duration_ms\": {},",
        json_optional_ms(feasible, window.duration_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"request_count\": {},",
        window.request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"completed_request_count\": {},",
        window.completed_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"failed_request_count\": {},",
        window.failed_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"rejected_request_count\": {},",
        window.rejected_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"timed_out_request_count\": {},",
        window.timed_out_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"cancelled_request_count\": {},",
        window.cancelled_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_constrained_request_count\": {},",
        window.deadline_constrained_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"deadline_missed_request_count\": {},",
        window.deadline_missed_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"measured_requests\": {},",
        window.measured_requests
    )?;
    writeln!(
        writer,
        "{indent}    \"lifecycle_event_metric_request_count\": {},",
        window.lifecycle_event_metric_request_count
    )?;
    writeln!(
        writer,
        "{indent}    \"fallback_metric_request_count\": {},",
        window.fallback_metric_request_count
    )?;
    write_measurement_metric_source_counts(
        writer,
        feasible.then_some(window.metric_source_counts.as_slice()),
        indent,
        true,
    )?;
    writeln!(
        writer,
        "{indent}    \"configured_start_bound\": {},",
        window.configured_start_bound
    )?;
    writeln!(
        writer,
        "{indent}    \"configured_end_bound\": {},",
        window.configured_end_bound
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_requested\": {},",
        window.steady_state_requested
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_applied\": {},",
        window.steady_state_applied
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_min_requests\": {steady_state_min_requests},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_max_cv\": {steady_state_max_cv},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_sample_count\": {},",
        window.steady_state_sample_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_matching_window_count\": {},",
        window.steady_state_matching_window_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_request_count\": {candidate_request_count},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_start_ms\": {candidate_start_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_end_ms\": {candidate_end_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_mean_ms\": {candidate_e2el_mean_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_stddev_ms\": {candidate_e2el_stddev_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_cv\": {candidate_e2el_cv},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_e2el_std_error_ms\": {candidate_e2el_std_error_ms},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_metric_count\": {},",
        window.steady_state_candidate_metric_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_metric\": {candidate_worst_metric},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_cv\": {candidate_worst_cv},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_output_tokens\": {candidate_output_tokens},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_throughput_tokens_per_s\": {candidate_throughput},"
    )?;
    write_steady_state_candidate_metrics(
        writer,
        feasible.then_some(window.steady_state_candidate_metrics.as_slice()),
        indent,
        true,
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_utilization_count\": {},",
        window.steady_state_candidate_utilization_count
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_utilization_resource\": {candidate_worst_utilization_resource},"
    )?;
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_worst_utilization_cv\": {candidate_worst_utilization_cv},"
    )?;
    write_steady_state_candidate_utilization(
        writer,
        feasible.then_some(window.steady_state_candidate_utilization.as_slice()),
        indent,
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_measurement_metric_source_counts<W: Write>(
    writer: &mut W,
    counts: Option<&[ServingMeasurementMetricSourceCount]>,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"metric_source_counts\": [")?;
    let counts = counts.unwrap_or(&[]);
    for (idx, count) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"metric_source\": {},",
            json_string(&count.metric_source)
        )?;
        writeln!(
            writer,
            "{indent}        \"request_count\": {}",
            count.request_count
        )?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_steady_state_candidate_metrics<W: Write>(
    writer: &mut W,
    metrics: Option<&[ServingSteadyStateMetricObservation]>,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"steady_state_candidate_metrics\": [")?;
    if let Some(metrics) = metrics {
        for (idx, metric) in metrics.iter().enumerate() {
            writeln!(writer, "{indent}      {{")?;
            writeln!(
                writer,
                "{indent}        \"metric\": {},",
                json_string(&metric.metric)
            )?;
            writeln!(
                writer,
                "{indent}        \"unit\": {},",
                json_string(&metric.unit)
            )?;
            writeln!(
                writer,
                "{indent}        \"sample_count\": {},",
                metric.sample_count
            )?;
            writeln!(
                writer,
                "{indent}        \"mean\": {},",
                json_optional_f64(metric.mean)
            )?;
            writeln!(
                writer,
                "{indent}        \"stddev\": {},",
                json_optional_f64(metric.stddev)
            )?;
            writeln!(
                writer,
                "{indent}        \"cv\": {},",
                json_optional_f64(metric.cv)
            )?;
            writeln!(
                writer,
                "{indent}        \"std_error\": {}",
                json_optional_f64(metric.std_error)
            )?;
            writeln!(writer, "{indent}      }}{}", comma(idx + 1 < metrics.len()))?;
        }
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_steady_state_candidate_utilization<W: Write>(
    writer: &mut W,
    observations: Option<&[ServingSteadyStateUtilizationObservation]>,
    indent: &str,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "{indent}    \"steady_state_candidate_utilization\": ["
    )?;
    if let Some(observations) = observations {
        for (idx, observation) in observations.iter().enumerate() {
            writeln!(writer, "{indent}      {{")?;
            writeln!(
                writer,
                "{indent}        \"source\": {},",
                json_string(&observation.source)
            )?;
            writeln!(
                writer,
                "{indent}        \"phase\": {},",
                json_string(&observation.phase)
            )?;
            writeln!(
                writer,
                "{indent}        \"resource_kind\": {},",
                json_string(&observation.resource_kind)
            )?;
            writeln!(
                writer,
                "{indent}        \"resource\": {},",
                json_string(&observation.resource)
            )?;
            writeln!(
                writer,
                "{indent}        \"bucket_count\": {},",
                observation.bucket_count
            )?;
            writeln!(
                writer,
                "{indent}        \"active_bucket_count\": {},",
                observation.active_bucket_count
            )?;
            writeln!(
                writer,
                "{indent}        \"event_count\": {},",
                observation.event_count
            )?;
            writeln!(
                writer,
                "{indent}        \"mean_utilization\": {},",
                json_optional_f64(observation.mean_utilization)
            )?;
            writeln!(
                writer,
                "{indent}        \"max_utilization\": {},",
                json_optional_f64(observation.max_utilization)
            )?;
            writeln!(
                writer,
                "{indent}        \"utilization_cv\": {}",
                json_optional_f64(observation.utilization_cv)
            )?;
            writeln!(
                writer,
                "{indent}      }}{}",
                comma(idx + 1 < observations.len())
            )?;
        }
    }
    writeln!(writer, "{indent}    ]")
}

fn write_approximations<W: Write>(
    writer: &mut W,
    indent: &str,
    approximations: &[SimulationApproximation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"approximations\": [")?;
    for (idx, approximation) in approximations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&approximation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&approximation.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(&approximation.scope)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&approximation.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&approximation.message)
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {}",
            approximation
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < approximations.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_serving_approximation_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &ServingApproximationSummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"approximation_summary\": {{")?;
    writeln!(
        writer,
        "{indent}    \"status\": {},",
        json_string(&summary.status)
    )?;
    writeln!(
        writer,
        "{indent}    \"approximation_count\": {},",
        summary.approximation_count
    )?;
    writeln!(
        writer,
        "{indent}    \"policy_violation_count\": {},",
        summary.policy_violation_count
    )?;
    writeln!(
        writer,
        "{indent}    \"calibration_count\": {},",
        summary.calibration_count
    )?;
    writeln!(
        writer,
        "{indent}    \"topology_count\": {},",
        summary.topology_count
    )?;
    writeln!(
        writer,
        "{indent}    \"queueing_count\": {},",
        summary.queueing_count
    )?;
    writeln!(
        writer,
        "{indent}    \"runtime_count\": {},",
        summary.runtime_count
    )?;
    writeln!(
        writer,
        "{indent}    \"memory_count\": {},",
        summary.memory_count
    )?;
    writeln!(
        writer,
        "{indent}    \"capacity_count\": {},",
        summary.capacity_count
    )?;
    writeln!(
        writer,
        "{indent}    \"routing_count\": {},",
        summary.routing_count
    )?;
    writeln!(
        writer,
        "{indent}    \"admission_count\": {},",
        summary.admission_count
    )?;
    writeln!(
        writer,
        "{indent}    \"uncalibrated_phase_count\": {},",
        summary.uncalibrated_phase_count
    )?;
    writeln!(
        writer,
        "{indent}    \"uncalibrated_queue_component_count\": {},",
        summary.uncalibrated_queue_component_count
    )?;
    writeln!(
        writer,
        "{indent}    \"extrapolated_fit_count\": {},",
        summary.extrapolated_fit_count
    )?;
    writeln!(
        writer,
        "{indent}    \"coarse_topology\": {},",
        summary.coarse_topology
    )?;
    writeln!(
        writer,
        "{indent}    \"approximate_queueing\": {},",
        summary.approximate_queueing
    )?;
    writeln!(
        writer,
        "{indent}    \"uncalibrated_runtime\": {},",
        summary.uncalibrated_runtime
    )?;
    write_serving_approximation_counts(
        writer,
        indent,
        "category_counts",
        &summary.category_counts,
        true,
    )?;
    write_serving_approximation_counts(writer, indent, "top_codes", &summary.top_codes, false)?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_serving_approximation_counts<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    counts: &[ServingApproximationCount],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"{field}\": [")?;
    for (idx, count) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"name\": {},",
            json_string(&count.name)
        )?;
        writeln!(writer, "{indent}        \"count\": {}", count.count)?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_approximation_policy_violations<W: Write>(
    writer: &mut W,
    indent: &str,
    violations: &[ApproximationPolicyViolation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"approximation_policy_violations\": [")?;
    for (idx, violation) in violations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"metric\": {},",
            json_optional_string(violation.metric.as_deref())
        )?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&violation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&violation.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"scope\": {},",
            json_string(&violation.scope)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&violation.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"action\": {},",
            json_string(violation.action.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {}",
            json_string(&violation.message)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < violations.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_serving_memory<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    memory: &ServingMemoryHeadroom,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"{field}\": {{")?;
    writeln!(
        writer,
        "{indent}    \"estimated_per_gpu_gb\": {},",
        json_optional_f64(memory.estimated_per_gpu_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"min_hbm_per_gpu_gb\": {},",
        json_optional_f64(memory.min_hbm_per_gpu_gb)
    )?;
    write_memory_limiter_gpu(writer, indent, memory.limiting_gpu, true)?;
    writeln!(
        writer,
        "{indent}    \"capacity_used_fraction\": {},",
        json_optional_f64(memory.capacity_used_fraction())
    )?;
    writeln!(
        writer,
        "{indent}    \"headroom_gb\": {},",
        json_optional_f64(memory.headroom_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"headroom_fraction\": {},",
        json_optional_f64(memory.headroom_fraction)
    )?;
    write_memory_dominant_component(writer, indent, memory, true)?;
    write_memory_component_fractions(writer, indent, memory, true)?;
    writeln!(writer, "{indent}    \"components\": {{")?;
    writeln!(
        writer,
        "{indent}      \"weights_gb\": {},",
        json_optional_f64(memory.components.weights_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"kv_cache_gb\": {},",
        json_optional_f64(memory.components.kv_cache_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"block_table_gb\": {},",
        json_optional_f64(memory.components.block_table_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"activations_gb\": {},",
        json_optional_f64(memory.components.activations_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"temporary_gb\": {},",
        json_optional_f64(memory.components.temporary_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"communication_gb\": {},",
        json_optional_f64(memory.components.communication_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"runtime_reserve_gb\": {},",
        json_optional_f64(memory.components.runtime_reserve_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"fragmentation_gb\": {},",
        json_optional_f64(memory.components.fragmentation_gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"total_gb\": {}",
        json_optional_f64(memory.components.total_gb)
    )?;
    writeln!(writer, "{indent}    }}")?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_serving_hardware_footprint<W: Write>(
    writer: &mut W,
    indent: &str,
    footprint: &ServingHardwareFootprint,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"hardware_footprint\": {{")?;
    writeln!(
        writer,
        "{indent}    \"unique_node_count\": {},",
        footprint.unique_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"unique_gpu_count\": {},",
        footprint.unique_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_node_count\": {},",
        footprint.prefill_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_gpu_count\": {},",
        footprint.prefill_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_node_count\": {},",
        footprint.decode_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_gpu_count\": {},",
        footprint.decode_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"shared_node_count\": {},",
        footprint.shared_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"shared_gpu_count\": {},",
        footprint.shared_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_hbm_gb\": {},",
        json_optional_f64(footprint.aggregate_hbm_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_hbm_gb\": {},",
        json_optional_f64(footprint.prefill_hbm_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_hbm_gb\": {},",
        json_optional_f64(footprint.decode_hbm_gb)
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_hbm_bandwidth_gb_s\": {},",
        json_optional_f64(footprint.aggregate_hbm_bandwidth_gb_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_hbm_bandwidth_gb_s\": {},",
        json_optional_f64(footprint.prefill_hbm_bandwidth_gb_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_hbm_bandwidth_gb_s\": {},",
        json_optional_f64(footprint.decode_hbm_bandwidth_gb_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_peak_f16_tflops\": {},",
        json_optional_f64(footprint.aggregate_peak_f16_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_peak_f16_tflops\": {},",
        json_optional_f64(footprint.prefill_peak_f16_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_peak_f16_tflops\": {},",
        json_optional_f64(footprint.decode_peak_f16_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_peak_f8_tflops\": {},",
        json_optional_value(footprint.aggregate_peak_f8_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_peak_f8_tflops\": {},",
        json_optional_value(footprint.prefill_peak_f8_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_peak_f8_tflops\": {},",
        json_optional_value(footprint.decode_peak_f8_tflops)
    )?;
    write_serving_gpu_type_counts(
        writer,
        indent,
        "aggregate_gpu_types",
        &footprint.aggregate_gpu_types,
        true,
    )?;
    write_serving_gpu_type_counts(
        writer,
        indent,
        "prefill_gpu_types",
        &footprint.prefill_gpu_types,
        true,
    )?;
    write_serving_gpu_type_counts(
        writer,
        indent,
        "decode_gpu_types",
        &footprint.decode_gpu_types,
        true,
    )?;
    write_serving_gpu_label_counts(
        writer,
        indent,
        "aggregate_gpu_label_counts",
        &footprint.aggregate_gpu_label_counts,
        true,
    )?;
    write_serving_gpu_label_counts(
        writer,
        indent,
        "prefill_gpu_label_counts",
        &footprint.prefill_gpu_label_counts,
        true,
    )?;
    write_serving_gpu_label_counts(
        writer,
        indent,
        "decode_gpu_label_counts",
        &footprint.decode_gpu_label_counts,
        true,
    )?;
    writeln!(
        writer,
        "{indent}    \"aggregate_effective_peak_tflops\": {},",
        json_optional_f64(footprint.aggregate_effective_peak_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"prefill_effective_peak_tflops\": {},",
        json_optional_f64(footprint.prefill_effective_peak_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"decode_effective_peak_tflops\": {},",
        json_optional_f64(footprint.decode_effective_peak_tflops)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s_per_gpu\": {},",
        json_optional_f64(footprint.throughput_tokens_per_s_per_gpu)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s_per_effective_peak_tflop\": {},",
        json_optional_f64(footprint.throughput_tokens_per_s_per_effective_peak_tflop)
    )?;
    writeln!(
        writer,
        "{indent}    \"throughput_tokens_per_s_per_hbm_gb\": {}",
        json_optional_f64(footprint.throughput_tokens_per_s_per_hbm_gb)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_serving_gpu_type_counts<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    counts: &[ServingGpuTypeCount],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"{field}\": [")?;
    for (idx, entry) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"gpu\": {},",
            json_string(&entry.gpu)
        )?;
        writeln!(writer, "{indent}        \"count\": {}", entry.count)?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_serving_gpu_label_counts<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    counts: &[ServingGpuLabelCount],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}    \"{field}\": [")?;
    for (idx, entry) in counts.iter().enumerate() {
        writeln!(writer, "{indent}      {{")?;
        writeln!(
            writer,
            "{indent}        \"label\": {},",
            json_string(&entry.label)
        )?;
        writeln!(writer, "{indent}        \"count\": {}", entry.count)?;
        writeln!(writer, "{indent}      }}{}", comma(idx + 1 < counts.len()))?;
    }
    writeln!(writer, "{indent}    ]{}", comma(trailing_comma))
}

fn write_serving_cost_estimate<W: Write>(
    writer: &mut W,
    indent: &str,
    estimate: &ServingCostEstimate,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"cost_estimate\": {{")?;
    writeln!(
        writer,
        "{indent}    \"modeled_duration_s\": {},",
        json_optional_value(estimate.modeled_duration_s)
    )?;
    writeln!(
        writer,
        "{indent}    \"modeled_gpu_count\": {},",
        estimate.modeled_gpu_count
    )?;
    writeln!(
        writer,
        "{indent}    \"modeled_node_count\": {},",
        estimate.modeled_node_count
    )?;
    writeln!(
        writer,
        "{indent}    \"gpu_hours\": {},",
        json_optional_value(estimate.gpu_hours)
    )?;
    writeln!(
        writer,
        "{indent}    \"node_hours\": {},",
        json_optional_value(estimate.node_hours)
    )?;
    writeln!(
        writer,
        "{indent}    \"gpu_hour_cost_usd\": {},",
        json_optional_value(estimate.gpu_hour_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"node_hour_cost_usd\": {},",
        json_optional_value(estimate.node_hour_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"average_power_watts\": {},",
        json_optional_value(estimate.average_power_watts)
    )?;
    writeln!(
        writer,
        "{indent}    \"energy_kwh\": {},",
        json_optional_value(estimate.energy_kwh)
    )?;
    writeln!(
        writer,
        "{indent}    \"energy_cost_usd\": {},",
        json_optional_value(estimate.energy_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"total_cost_usd\": {},",
        json_optional_value(estimate.total_cost_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"cost_per_1k_output_tokens_usd\": {},",
        json_optional_value(estimate.cost_per_1k_output_tokens_usd)
    )?;
    writeln!(
        writer,
        "{indent}    \"cost_per_1k_requests_usd\": {}",
        json_optional_value(estimate.cost_per_1k_requests_usd)
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_memory_limiter_gpu<W: Write>(
    writer: &mut W,
    indent: &str,
    limiting_gpu: Option<GpuAddr>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(gpu) = limiting_gpu else {
        return writeln!(
            writer,
            "{indent}    \"limiting_gpu\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}    \"limiting_gpu\": {{")?;
    writeln!(writer, "{indent}      \"node_id\": {},", gpu.node_id)?;
    writeln!(
        writer,
        "{indent}      \"local_gpu_id\": {}",
        gpu.local_gpu_id
    )?;
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_memory_dominant_component<W: Write>(
    writer: &mut W,
    indent: &str,
    memory: &ServingMemoryHeadroom,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(component) = memory.dominant_component() else {
        return writeln!(
            writer,
            "{indent}    \"dominant_component\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}    \"dominant_component\": {{")?;
    writeln!(
        writer,
        "{indent}      \"name\": {},",
        json_string(component.name)
    )?;
    writeln!(
        writer,
        "{indent}      \"gb\": {},",
        json_optional_f64(component.gb)
    )?;
    writeln!(
        writer,
        "{indent}      \"fraction_of_total\": {}",
        json_optional_f64(component.fraction_of_total)
    )?;
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_memory_component_fractions<W: Write>(
    writer: &mut W,
    indent: &str,
    memory: &ServingMemoryHeadroom,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let contributions = memory.components.component_contributions();
    writeln!(writer, "{indent}    \"component_fractions\": {{")?;
    for (idx, component) in contributions.iter().enumerate() {
        writeln!(
            writer,
            "{indent}      \"{}\": {}{}",
            component.name,
            json_optional_f64(component.fraction_of_total),
            comma(idx + 1 < contributions.len())
        )?;
    }
    writeln!(writer, "{indent}    }}{}", comma(trailing_comma))
}

fn write_memory_pressure_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingMemoryPressureObservation],
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_observations =
        limit.map_or(observations.len(), |limit| limit.min(observations.len()));
    writeln!(
        writer,
        "{indent}  \"memory_pressure_observation_count\": {},",
        observations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"memory_pressure_observations_truncated\": {},",
        displayed_observations < observations.len()
    )?;
    writeln!(writer, "{indent}  \"memory_pressure\": [")?;
    for (idx, observation) in observations.iter().take(displayed_observations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&observation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"estimate_kind\": {},",
            json_string(&observation.estimate_kind)
        )?;
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_ms(observation.start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_ms(observation.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"duration_ms\": {},",
            json_ms(observation.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"active_requests\": {},",
            observation.active_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"active_tokens\": {},",
            observation.active_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_blocks\": {},",
            observation.kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"estimated_per_gpu_gb\": {},",
            json_optional_f64(observation.estimated_per_gpu_gb)
        )?;
        writeln!(
            writer,
            "{indent}      \"min_hbm_per_gpu_gb\": {},",
            json_optional_f64(observation.min_hbm_per_gpu_gb)
        )?;
        writeln!(
            writer,
            "{indent}      \"capacity_used_fraction\": {},",
            json_optional_f64(observation.capacity_used_fraction)
        )?;
        writeln!(
            writer,
            "{indent}      \"headroom_gb\": {},",
            json_optional_f64(observation.headroom_gb)
        )?;
        write_memory_pressure_limiter_gpu(writer, indent, observation.limiting_gpu, true)?;
        write_memory_pressure_dominant_component(
            writer,
            indent,
            observation.dominant_component,
            true,
        )?;
        write_memory_pressure_component_fractions(writer, indent, observation, true)?;
        write_memory_pressure_components(writer, indent, observation, false)?;
        if idx + 1 < displayed_observations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_memory_pressure_limiter_gpu<W: Write>(
    writer: &mut W,
    indent: &str,
    limiting_gpu: Option<GpuAddr>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(gpu) = limiting_gpu else {
        return writeln!(
            writer,
            "{indent}      \"limiting_gpu\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}      \"limiting_gpu\": {{")?;
    writeln!(writer, "{indent}        \"node_id\": {},", gpu.node_id)?;
    writeln!(
        writer,
        "{indent}        \"local_gpu_id\": {}",
        gpu.local_gpu_id
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_memory_pressure_dominant_component<W: Write>(
    writer: &mut W,
    indent: &str,
    component: Option<crate::serving::ServingMemoryComponentContribution>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(component) = component else {
        return writeln!(
            writer,
            "{indent}      \"dominant_component\": null{}",
            comma(trailing_comma)
        );
    };

    writeln!(writer, "{indent}      \"dominant_component\": {{")?;
    writeln!(
        writer,
        "{indent}        \"name\": {},",
        json_string(component.name)
    )?;
    writeln!(
        writer,
        "{indent}        \"gb\": {},",
        json_optional_f64(component.gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"fraction_of_total\": {}",
        json_optional_f64(component.fraction_of_total)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_memory_pressure_component_fractions<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingMemoryPressureObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let contributions = observation.components.component_contributions();
    writeln!(writer, "{indent}      \"component_fractions\": {{")?;
    for (idx, component) in contributions.iter().enumerate() {
        writeln!(
            writer,
            "{indent}        \"{}\": {}{}",
            component.name,
            json_optional_f64(component.fraction_of_total),
            comma(idx + 1 < contributions.len())
        )?;
    }
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_memory_pressure_components<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingMemoryPressureObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let components = observation.components;
    writeln!(writer, "{indent}      \"components\": {{")?;
    writeln!(
        writer,
        "{indent}        \"weights_gb\": {},",
        json_optional_f64(components.weights_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"kv_cache_gb\": {},",
        json_optional_f64(components.kv_cache_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"block_table_gb\": {},",
        json_optional_f64(components.block_table_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"activations_gb\": {},",
        json_optional_f64(components.activations_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"temporary_gb\": {},",
        json_optional_f64(components.temporary_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"communication_gb\": {},",
        json_optional_f64(components.communication_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"runtime_reserve_gb\": {},",
        json_optional_f64(components.runtime_reserve_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"fragmentation_gb\": {},",
        json_optional_f64(components.fragmentation_gb)
    )?;
    writeln!(
        writer,
        "{indent}        \"total_gb\": {}",
        json_optional_f64(components.total_gb)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
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

fn write_request_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    score: &ScoredServingConfig,
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let observations = &score.request_observations;
    let displayed_observations =
        limit.map_or(observations.len(), |limit| limit.min(observations.len()));
    writeln!(
        writer,
        "{indent}  \"request_observation_count\": {},",
        observations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"request_observations_truncated\": {},",
        displayed_observations < observations.len()
    )?;
    writeln!(writer, "{indent}  \"request_observations\": [")?;
    for (idx, observation) in observations.iter().take(displayed_observations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"request_idx\": {},",
            observation.request_idx
        )?;
        writeln!(
            writer,
            "{indent}      \"request_id\": {},",
            observation
                .request_id
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"tenant\": {},",
            observation
                .tenant
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"model_id\": {},",
            observation
                .model_id
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"traffic_class\": {},",
            observation
                .traffic_class
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"shape_profile\": {},",
            observation
                .shape_profile
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"cache_key\": {},",
            observation
                .cache_key
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"status\": {},",
            json_string(observation.status.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"status_time_ms\": {},",
            observation
                .status_time_s
                .map(json_ms)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"failure_reason\": {},",
            observation
                .failure_reason
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        write_request_rejection(writer, indent, observation.rejection.as_ref(), true)?;
        writeln!(
            writer,
            "{indent}      \"priority\": {},",
            observation.priority
        )?;
        write_request_slo(writer, indent, &observation.slo, true)?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_missed\": {},",
            observation.ttft_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_missed\": {},",
            observation.tpot_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_missed\": {},",
            observation.itl_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_missed\": {},",
            observation.e2el_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_ms\": {},",
            observation
                .deadline_s
                .map(json_ms)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_missed\": {},",
            observation.deadline_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"cancellation_ms\": {},",
            observation
                .cancellation_s
                .map(json_ms)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_node\": {},",
            observation.prefill_node
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_route_nodes\": [{}],",
            u32_list(&observation.prefill_route_nodes)
        )?;
        write_gpu_addr_list(
            writer,
            indent,
            "prefill_route_gpus",
            &observation.prefill_route_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_node\": {},",
            observation.decode_node
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_route_nodes\": [{}],",
            u32_list(&observation.decode_route_nodes)
        )?;
        write_gpu_addr_list(
            writer,
            indent,
            "decode_route_gpus",
            &observation.decode_route_gpus,
            true,
        )?;
        write_gpu_addr_list(
            writer,
            indent,
            "kv_cache_owner_gpus",
            &observation.kv_cache_owner_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_policy\": {},",
            json_string(observation.routing_policy.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_candidate_count\": {},",
            observation.routing_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_routable_candidate_count\": {},",
            observation.routing_routable_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_e2el_ms\": {},",
            json_ms(observation.routing_estimated_e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_kv_transfer_ms\": {},",
            json_ms(observation.routing_estimated_kv_transfer_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_kv_resource_wait_ms\": {},",
            json_ms(observation.routing_estimated_kv_resource_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_prefill_wait_ms\": {},",
            json_ms(observation.routing_estimated_prefill_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_decode_wait_ms\": {},",
            json_ms(observation.routing_estimated_decode_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_reason\": {},",
            json_string(&observation.routing_reason)
        )?;
        write_routing_candidates(writer, indent, &observation.routing_candidates, true)?;
        writeln!(
            writer,
            "{indent}      \"batch_size\": {},",
            observation.batch_size
        )?;
        writeln!(
            writer,
            "{indent}      \"prompt_tokens\": {},",
            observation.prompt_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"prefix_cache_hit_tokens\": {},",
            observation.prefix_cache_hit_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"effective_prefill_tokens\": {},",
            observation.effective_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_chunks\": {},",
            observation.prefill_chunks
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_tokens\": {},",
            observation.decode_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_tokens\": {},",
            observation.kv_block_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_cache_blocks\": {},",
            observation.kv_cache_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_allocated_tokens\": {},",
            observation.kv_allocated_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_fragmentation_tokens\": {},",
            observation.kv_fragmentation_tokens
        )?;
        write_kv_block_ownership(writer, indent, observation, true)?;
        writeln!(
            writer,
            "{indent}      \"arrival_ms\": {},",
            json_ms(observation.arrival_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_start_ms\": {},",
            json_ms(observation.prefill_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_finish_ms\": {},",
            json_ms(observation.prefill_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_start_ms\": {},",
            json_ms(observation.kv_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_finish_ms\": {},",
            json_ms(observation.kv_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"first_decode_start_ms\": {},",
            json_ms(observation.first_decode_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"first_decode_finish_ms\": {},",
            json_ms(observation.first_decode_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_decode_finish_ms\": {},",
            json_ms(observation.last_decode_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iterations\": {},",
            observation.decode_iterations
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_token_start_ms\": [{}],",
            ms_list(&observation.decode_token_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_token_finish_ms\": [{}],",
            ms_list(&observation.decode_token_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"inter_token_latency_ms\": [{}],",
            ms_list(&observation.inter_token_latency_s)
        )?;
        write_request_phase_spans(writer, indent, observation, true)?;
        write_request_phase_breakdown(writer, indent, observation, true)?;
        write_request_lifecycle_events(writer, indent, observation, true)?;
        writeln!(
            writer,
            "{indent}      \"metric_source\": {},",
            json_string(&observation.metric_source)
        )?;
        writeln!(
            writer,
            "{indent}      \"included_in_measurement_window\": {},",
            request_in_measurement_window(observation, score)
        )?;
        writeln!(
            writer,
            "{indent}      \"measurement_window_source\": {},",
            json_string(&score.measurement_window.source)
        )?;
        writeln!(
            writer,
            "{indent}      \"output_tokens\": {},",
            request_observation_output_tokens(observation)
        )?;
        write_request_metric_derivation(writer, indent, observation, score, true)?;
        write_request_worker_summary(writer, indent, &observation.worker_summary, true)?;
        write_worker_assignments(writer, indent, &observation.worker_assignments, true)?;
        writeln!(
            writer,
            "{indent}      \"ttft_ms\": {},",
            json_ms(observation.ttft_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_ms\": {},",
            json_ms(observation.tpot_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_ms\": {},",
            json_ms(observation.itl_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_ms\": {},",
            json_ms(observation.e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"service_ms\": {},",
            json_ms(observation.service_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_delay_ms\": {},",
            json_ms(observation.queue_delay_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_ms\": {},",
            json_ms(observation.prefill_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_worker_queue_ms\": {},",
            json_ms(observation.prefill_worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_resource_queue_ms\": {},",
            json_ms(observation.prefill_resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_queue_ms\": {},",
            json_ms(observation.kv_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_worker_queue_ms\": {},",
            json_ms(observation.kv_worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_resource_queue_ms\": {},",
            json_ms(observation.kv_resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_bytes\": {},",
            observation.kv_transfer_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_bottlenecks\": [{}],",
            string_list(&observation.kv_transfer_bottlenecks)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_resources\": [{}],",
            string_list(&observation.kv_transfer_resources)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_resource_dependencies\": [{}],",
            usize_list(&observation.kv_transfer_resource_dependencies)
        )?;
        write_kv_transfer_paths(writer, indent, observation, true)?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_ms\": {},",
            json_ms(observation.kv_transfer_s)
        )?;
        write_optional_calibration_fit_application(
            writer,
            indent,
            "kv_transfer_fit",
            observation.kv_transfer_fit.as_ref(),
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_queue_ms\": {},",
            json_ms(observation.decode_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_worker_queue_ms\": {},",
            json_ms(observation.decode_worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_resource_queue_ms\": {},",
            json_ms(observation.decode_resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_ms\": {}",
            json_ms(observation.decode_s)
        )?;
        if idx + 1 < displayed_observations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

#[derive(Debug)]
struct RequestMetricDerivation {
    completed: bool,
    event_sourced: bool,
    terminal_event: Option<&'static str>,
    terminal_event_s: Option<f64>,
    metric_unavailable_reason: Option<String>,
    ttft_end_event: Option<&'static str>,
    tpot_start_event: Option<&'static str>,
    tpot_end_event: Option<&'static str>,
    e2el_end_event: Option<&'static str>,
    throughput_duration_start_event: Option<&'static str>,
    throughput_duration_end_event: Option<&'static str>,
    decode_finish_event_count: usize,
    tpot_sample_count: usize,
}

fn request_metric_derivation(observation: &ServingRequestObservation) -> RequestMetricDerivation {
    let completed = observation.status.as_str() == "completed";
    let event_sourced = observation.metric_source == "request_lifecycle_events";
    let decode_finish_event_count = observation.decode_token_finish_s.len();
    let tpot_sample_count = if decode_finish_event_count > 1 {
        decode_finish_event_count - 1
    } else if completed && decode_finish_event_count == 1 {
        1
    } else {
        0
    };
    let first_decode_finish_event =
        (decode_finish_event_count > 0).then_some("decode_iteration_finished:first");
    let last_decode_finish_event =
        (decode_finish_event_count > 0).then_some("decode_iteration_finished:last");
    let tpot_end_event = if decode_finish_event_count > 1 {
        "decode_iteration_finished:last"
    } else {
        "decode_iteration_finished:first"
    };
    let tpot_event = (tpot_sample_count > 0).then_some(tpot_end_event);
    let terminal_event = match observation.status.as_str() {
        "pending" => None,
        status => Some(status),
    };
    let metric_unavailable_reason = if completed {
        None
    } else {
        Some(format!(
            "request_not_completed:{}",
            observation.status.as_str()
        ))
    };
    let e2el_end_event = if completed {
        last_decode_finish_event
    } else {
        terminal_event
    };

    RequestMetricDerivation {
        completed,
        event_sourced,
        terminal_event,
        terminal_event_s: observation.status_time_s,
        metric_unavailable_reason,
        ttft_end_event: first_decode_finish_event,
        tpot_start_event: tpot_event.map(|_| "decode_iteration_finished:first"),
        tpot_end_event: tpot_event,
        e2el_end_event,
        throughput_duration_start_event: completed.then_some("arrived"),
        throughput_duration_end_event: completed.then_some(last_decode_finish_event).flatten(),
        decode_finish_event_count,
        tpot_sample_count,
    }
}

fn write_request_metric_derivation<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    score: &ScoredServingConfig,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let derivation = request_metric_derivation(observation);

    writeln!(writer, "{indent}      \"metric_derivation\": {{")?;
    writeln!(
        writer,
        "{indent}        \"metric_source\": {},",
        json_string(&observation.metric_source)
    )?;
    writeln!(
        writer,
        "{indent}        \"event_sourced\": {},",
        derivation.event_sourced
    )?;
    writeln!(
        writer,
        "{indent}        \"completed\": {},",
        derivation.completed
    )?;
    writeln!(
        writer,
        "{indent}        \"terminal_event\": {},",
        json_optional_string(derivation.terminal_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"terminal_event_ms\": {},",
        derivation
            .terminal_event_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"metric_unavailable_reason\": {},",
        json_optional_string(derivation.metric_unavailable_reason.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}        \"included_in_measurement_window\": {},",
        request_in_measurement_window(observation, score)
    )?;
    writeln!(
        writer,
        "{indent}        \"measurement_window_source\": {},",
        json_string(&score.measurement_window.source)
    )?;
    writeln!(
        writer,
        "{indent}        \"measurement_start_ms\": {},",
        json_ms(score.measurement_window.start_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"measurement_end_ms\": {},",
        json_ms(score.measurement_window.end_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"arrival_ms\": {},",
        json_ms(observation.arrival_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"prefill_start_ms\": {},",
        json_ms(observation.prefill_start_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"kv_finish_ms\": {},",
        json_ms(observation.kv_finish_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"first_decode_start_ms\": {},",
        json_ms(observation.first_decode_start_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"first_decode_finish_ms\": {},",
        json_ms(observation.first_decode_finish_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"last_decode_finish_ms\": {},",
        json_ms(observation.last_decode_finish_s)
    )?;
    writeln!(writer, "{indent}        \"ttft_start_event\": \"arrived\",")?;
    writeln!(
        writer,
        "{indent}        \"ttft_end_event\": {},",
        json_optional_string(derivation.ttft_end_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_start_event\": {},",
        json_optional_string(derivation.tpot_start_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_end_event\": {},",
        json_optional_string(derivation.tpot_end_event)
    )?;
    writeln!(writer, "{indent}        \"e2el_start_event\": \"arrived\",")?;
    writeln!(
        writer,
        "{indent}        \"e2el_end_event\": {},",
        json_optional_string(derivation.e2el_end_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"throughput_duration_start_event\": {},",
        json_optional_string(derivation.throughput_duration_start_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"throughput_duration_end_event\": {},",
        json_optional_string(derivation.throughput_duration_end_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"decode_iteration_count\": {},",
        observation.decode_iterations
    )?;
    writeln!(
        writer,
        "{indent}        \"decode_finish_event_count\": {},",
        derivation.decode_finish_event_count
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_sample_count\": {},",
        derivation.tpot_sample_count
    )?;
    writeln!(
        writer,
        "{indent}        \"output_tokens\": {},",
        request_observation_output_tokens(observation)
    )?;
    writeln!(
        writer,
        "{indent}        \"request_output_tokens_per_s\": {},",
        json_optional_value(request_observation_output_tokens_per_s(observation))
    )?;
    writeln!(
        writer,
        "{indent}        \"ttft_ms\": {},",
        json_ms(observation.ttft_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_ms\": {},",
        json_ms(observation.tpot_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"itl_ms\": {},",
        json_ms(observation.itl_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"e2el_ms\": {}",
        json_ms(observation.e2el_s)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_request_rejection<W: Write>(
    writer: &mut W,
    indent: &str,
    rejection: Option<&ServingRejection>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(rejection) = rejection else {
        writeln!(
            writer,
            "{indent}      \"rejection\": null{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    writeln!(writer, "{indent}      \"rejection\": {{")?;
    writeln!(
        writer,
        "{indent}        \"phase\": {},",
        json_string(&rejection.phase)
    )?;
    writeln!(
        writer,
        "{indent}        \"category\": {},",
        json_string(&rejection.category)
    )?;
    writeln!(
        writer,
        "{indent}        \"resource\": {},",
        json_string(&rejection.resource)
    )?;
    writeln!(
        writer,
        "{indent}        \"code\": {},",
        json_string(&rejection.code)
    )?;
    writeln!(
        writer,
        "{indent}        \"observed\": {},",
        json_optional_value(rejection.observed)
    )?;
    writeln!(
        writer,
        "{indent}        \"limit\": {},",
        json_optional_value(rejection.limit)
    )?;
    writeln!(
        writer,
        "{indent}        \"unit\": {},",
        rejection
            .unit
            .as_deref()
            .map(json_string)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"remediation\": {},",
        rejection
            .remediation
            .as_deref()
            .map(json_string)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"message\": {}",
        json_string(&rejection.message)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_routing_candidates<W: Write>(
    writer: &mut W,
    indent: &str,
    candidates: &[ServingRouteCandidateObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"routing_candidates\": [")?;
    for (idx, candidate) in candidates.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"prefill_node\": {},",
            candidate.prefill_node
        )?;
        writeln!(
            writer,
            "{indent}          \"prefill_route_nodes\": [{}],",
            u32_list(&candidate.prefill_route_nodes)
        )?;
        write_nested_gpu_addr_list(
            writer,
            &format!("{indent}          "),
            "prefill_route_gpus",
            &candidate.prefill_route_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_node\": {},",
            candidate.decode_node
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_route_nodes\": [{}],",
            u32_list(&candidate.decode_route_nodes)
        )?;
        write_nested_gpu_addr_list(
            writer,
            &format!("{indent}          "),
            "decode_route_gpus",
            &candidate.decode_route_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}          \"selected\": {},",
            candidate.selected
        )?;
        writeln!(
            writer,
            "{indent}          \"routable\": {},",
            candidate.routable
        )?;
        writeln!(
            writer,
            "{indent}          \"rejection_reason\": {},",
            candidate
                .rejection_reason
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_e2el_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_kv_transfer_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_kv_transfer_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_kv_resource_wait_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_kv_resource_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_prefill_wait_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_prefill_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_decode_wait_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_decode_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_transfer_bytes\": {},",
            candidate.kv_transfer_bytes
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_transfer_bottlenecks\": [{}],",
            string_list(&candidate.kv_transfer_bottlenecks)
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_transfer_resources\": [{}]",
            string_list(&candidate.kv_transfer_resources)
        )?;
        if idx + 1 < candidates.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_nested_gpu_addr_list<W: Write>(
    writer: &mut W,
    field_indent: &str,
    name: &str,
    gpus: &[GpuAddr],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{field_indent}\"{name}\": [")?;
    for (idx, gpu) in gpus.iter().enumerate() {
        writeln!(writer, "{field_indent}  {{")?;
        writeln!(writer, "{field_indent}    \"node_id\": {},", gpu.node_id)?;
        writeln!(
            writer,
            "{field_indent}    \"local_gpu_id\": {}",
            gpu.local_gpu_id
        )?;
        if idx + 1 < gpus.len() {
            writeln!(writer, "{field_indent}  }},")?;
        } else {
            writeln!(writer, "{field_indent}  }}")?;
        }
    }
    writeln!(writer, "{field_indent}]{}", comma(trailing_comma))
}

fn write_gpu_addr_list<W: Write>(
    writer: &mut W,
    indent: &str,
    name: &str,
    gpus: &[GpuAddr],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"{name}\": [")?;
    for (idx, gpu) in gpus.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(writer, "{indent}          \"node_id\": {},", gpu.node_id)?;
        writeln!(
            writer,
            "{indent}          \"local_gpu_id\": {}",
            gpu.local_gpu_id
        )?;
        if idx + 1 < gpus.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_kv_block_ownership<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"kv_block_ownership\": [")?;
    for (idx, ownership) in observation.kv_block_ownership.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"allocation_id\": {},",
            json_string(&ownership.allocation_id)
        )?;
        writeln!(writer, "{indent}          \"owner\": {{")?;
        writeln!(
            writer,
            "{indent}            \"node_id\": {},",
            ownership.owner.node_id
        )?;
        writeln!(
            writer,
            "{indent}            \"local_gpu_id\": {}",
            ownership.owner.local_gpu_id
        )?;
        writeln!(writer, "{indent}          }},")?;
        writeln!(
            writer,
            "{indent}          \"owner_worker_slots\": [{}],",
            u32_list(&ownership.owner_worker_slots)
        )?;
        write_kv_worker_slot_ownership(writer, indent, ownership, true)?;
        writeln!(
            writer,
            "{indent}          \"decode_operation_ids\": [{}],",
            usize_list(&ownership.decode_operation_ids)
        )?;
        writeln!(
            writer,
            "{indent}          \"allocated_at_ms\": {},",
            json_ms(ownership.allocated_at_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"released_at_ms\": {},",
            json_ms(ownership.released_at_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"duration_ms\": {},",
            json_ms(ownership.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"block_start\": {},",
            ownership.block_start
        )?;
        writeln!(
            writer,
            "{indent}          \"block_end\": {},",
            ownership.block_end
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_sequences\": {},",
            ownership.decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}          \"resident_tokens\": {},",
            ownership.resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_blocks\": {},",
            ownership.kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}          \"allocated_kv_tokens\": {},",
            ownership.allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_fragmentation_tokens\": {},",
            ownership.kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"block_table_entries\": {},",
            ownership.block_table_entries
        )?;
        writeln!(
            writer,
            "{indent}          \"block_table_bytes\": {}",
            ownership.block_table_bytes
        )?;
        if idx + 1 < observation.kv_block_ownership.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_kv_worker_slot_ownership<W: Write>(
    writer: &mut W,
    indent: &str,
    ownership: &ServingKvBlockOwnershipObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}          \"worker_slot_ownership\": [")?;
    for (idx, slot) in ownership.worker_slot_ownership.iter().enumerate() {
        writeln!(writer, "{indent}            {{")?;
        writeln!(
            writer,
            "{indent}              \"allocation_id\": {},",
            json_string(&slot.allocation_id)
        )?;
        writeln!(writer, "{indent}              \"slot\": {},", slot.slot)?;
        writeln!(
            writer,
            "{indent}              \"block_start\": {},",
            slot.block_start
        )?;
        writeln!(
            writer,
            "{indent}              \"block_end\": {},",
            slot.block_end
        )?;
        writeln!(
            writer,
            "{indent}              \"decode_sequences\": {},",
            slot.decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}              \"resident_tokens\": {},",
            slot.resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}              \"kv_blocks\": {},",
            slot.kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}              \"allocated_kv_tokens\": {},",
            slot.allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}              \"kv_fragmentation_tokens\": {},",
            slot.kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}              \"block_table_entries\": {},",
            slot.block_table_entries
        )?;
        writeln!(
            writer,
            "{indent}              \"block_table_bytes\": {}",
            slot.block_table_bytes
        )?;
        if idx + 1 < ownership.worker_slot_ownership.len() {
            writeln!(writer, "{indent}            }},")?;
        } else {
            writeln!(writer, "{indent}            }}")?;
        }
    }
    writeln!(writer, "{indent}          ]{}", comma(trailing_comma))
}

fn write_kv_transfer_paths<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"kv_transfer_paths\": [")?;
    for (idx, path) in observation.kv_transfer_paths.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(writer, "{indent}          \"source\": {{")?;
        writeln!(
            writer,
            "{indent}            \"node_id\": {},",
            path.source.node_id
        )?;
        writeln!(
            writer,
            "{indent}            \"local_gpu_id\": {}",
            path.source.local_gpu_id
        )?;
        writeln!(writer, "{indent}          }},")?;
        writeln!(writer, "{indent}          \"destination\": {{")?;
        writeln!(
            writer,
            "{indent}            \"node_id\": {},",
            path.destination.node_id
        )?;
        writeln!(
            writer,
            "{indent}            \"local_gpu_id\": {}",
            path.destination.local_gpu_id
        )?;
        writeln!(writer, "{indent}          }},")?;
        writeln!(
            writer,
            "{indent}          \"latency_ms\": {},",
            json_ms(path.latency_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"bottleneck_bandwidth_gbps\": {},",
            json_optional_f64(path.bottleneck_bandwidth_gbps)
        )?;
        writeln!(
            writer,
            "{indent}          \"resources\": [{}],",
            string_list(&path.resources)
        )?;
        write_kv_transfer_path_resource_details(writer, indent, &path.resource_details)?;
        if idx + 1 < observation.kv_transfer_paths.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_kv_transfer_path_resource_details<W: Write>(
    writer: &mut W,
    indent: &str,
    resources: &[ServingKvTransferPathResourceObservation],
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}          \"resource_details\": [")?;
    for (idx, resource) in resources.iter().enumerate() {
        writeln!(writer, "{indent}            {{")?;
        writeln!(
            writer,
            "{indent}              \"kind\": {},",
            json_string(&resource.kind)
        )?;
        writeln!(
            writer,
            "{indent}              \"label\": {},",
            json_string(&resource.label)
        )?;
        writeln!(
            writer,
            "{indent}              \"bandwidth_gbps\": {},",
            json_optional_f64(resource.bandwidth_gbps)
        )?;
        writeln!(
            writer,
            "{indent}              \"latency_ms\": {},",
            json_ms(resource.latency_s)
        )?;
        writeln!(
            writer,
            "{indent}              \"rail_id\": {},",
            json_optional_u32(resource.rail_id)
        )?;
        write_kv_transfer_path_endpoint(writer, indent, "from", resource.from.as_ref(), true)?;
        write_kv_transfer_path_endpoint(writer, indent, "to", resource.to.as_ref(), false)?;
        if idx + 1 < resources.len() {
            writeln!(writer, "{indent}            }},")?;
        } else {
            writeln!(writer, "{indent}            }}")?;
        }
    }
    writeln!(writer, "{indent}          ]")
}

fn write_kv_transfer_path_endpoint<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    endpoint: Option<&ServingKvTransferPathEndpointObservation>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(endpoint) = endpoint else {
        writeln!(
            writer,
            "{indent}              \"{field}\": null{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    writeln!(writer, "{indent}              \"{field}\": {{")?;
    writeln!(
        writer,
        "{indent}                \"kind\": {},",
        json_string(&endpoint.kind)
    )?;
    writeln!(
        writer,
        "{indent}                \"node_id\": {},",
        json_optional_u32(endpoint.node_id)
    )?;
    writeln!(
        writer,
        "{indent}                \"local_gpu_id\": {},",
        json_optional_u32(endpoint.local_gpu_id)
    )?;
    writeln!(
        writer,
        "{indent}                \"nic_id\": {},",
        json_optional_u32(endpoint.nic_id)
    )?;
    writeln!(
        writer,
        "{indent}                \"rail_id\": {}",
        json_optional_u32(endpoint.rail_id)
    )?;
    writeln!(writer, "{indent}              }}{}", comma(trailing_comma))
}

fn write_request_phase_spans<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let spans = request_phase_spans(observation);
    writeln!(writer, "{indent}      \"phase_spans\": [")?;
    for (idx, (phase, start_s, finish_s)) in spans.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(*start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(*finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"duration_ms\": {}",
            json_ms((finish_s - start_s).max(0.0))
        )?;
        if idx + 1 < spans.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn request_phase_spans(observation: &ServingRequestObservation) -> Vec<(&'static str, f64, f64)> {
    [
        (
            "queued_for_prefill",
            observation.arrival_s,
            observation.prefill_start_s,
        ),
        (
            "prefilling",
            observation.prefill_start_s,
            observation.prefill_finish_s,
        ),
        (
            "queued_for_kv_transfer",
            observation.prefill_finish_s,
            observation.kv_start_s,
        ),
        (
            "transferring_kv",
            observation.kv_start_s,
            observation.kv_finish_s,
        ),
        (
            "queued_for_decode",
            observation.kv_finish_s,
            observation.first_decode_start_s,
        ),
        (
            "decoding",
            observation.first_decode_start_s,
            observation.last_decode_finish_s,
        ),
    ]
    .into_iter()
    .filter(|(_, start_s, finish_s)| {
        start_s.is_finite() && finish_s.is_finite() && *finish_s >= *start_s
    })
    .collect()
}

fn write_request_phase_breakdown<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"phase_breakdown\": [")?;
    for (idx, phase) in observation.phase_breakdown.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&phase.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"category\": {},",
            json_string(phase.category.as_str())
        )?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(phase.start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(phase.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"duration_ms\": {},",
            json_ms(phase.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"contributes_to_ttft\": {},",
            phase.contributes_to_ttft
        )?;
        writeln!(
            writer,
            "{indent}          \"contributes_to_e2el\": {}",
            phase.contributes_to_e2el
        )?;
        if idx + 1 < observation.phase_breakdown.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_request_lifecycle_events<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"lifecycle_events\": [")?;
    for (idx, event) in observation.lifecycle_events.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"event\": {},",
            json_string(event.kind.as_str())
        )?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&event.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"at_ms\": {},",
            json_ms(event.at_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_iteration\": {},",
            event
                .decode_iteration
                .map(|iteration| iteration.to_string())
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}          \"message\": {}",
            event
                .message
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        if idx + 1 < observation.lifecycle_events.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_worker_assignments<W: Write>(
    writer: &mut W,
    indent: &str,
    assignments: &[ServingWorkerAssignmentObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"worker_assignments\": [")?;
    for (idx, assignment) in assignments.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&assignment.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"node_id\": {},",
            assignment.node_id
        )?;
        writeln!(
            writer,
            "{indent}          \"local_gpu_id\": {},",
            assignment.local_gpu_id
        )?;
        writeln!(writer, "{indent}          \"slot\": {},", assignment.slot)?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(assignment.start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(assignment.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"operation_ids\": [{}]",
            usize_list(&assignment.operation_ids)
        )?;
        if idx + 1 < assignments.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_request_worker_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summaries: &[ServingRequestWorkerSummaryObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"worker_summary\": [")?;
    for (idx, summary) in summaries.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"role\": {},",
            json_string(&summary.role)
        )?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&summary.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"node_id\": {},",
            summary.node_id
        )?;
        writeln!(
            writer,
            "{indent}          \"local_gpu_id\": {},",
            summary.local_gpu_id
        )?;
        writeln!(
            writer,
            "{indent}          \"worker_slots\": [{}],",
            u32_list(&summary.worker_slots)
        )?;
        writeln!(
            writer,
            "{indent}          \"operation_ids\": [{}],",
            usize_list(&summary.operation_ids)
        )?;
        writeln!(
            writer,
            "{indent}          \"assignment_count\": {},",
            summary.assignment_count
        )?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(summary.start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(summary.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"resident_tokens\": {},",
            summary.resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_blocks\": {}",
            summary.kv_blocks
        )?;
        if idx + 1 < summaries.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_decode_iterations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingDecodeIterationObservation],
    limit: Option<usize>,
    include_operation_ids: bool,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_observations =
        limit.map_or(observations.len(), |limit| limit.min(observations.len()));
    writeln!(
        writer,
        "{indent}  \"decode_iteration_count\": {},",
        observations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"decode_iterations_truncated\": {},",
        displayed_observations < observations.len()
    )?;
    writeln!(writer, "{indent}  \"decode_iterations\": [")?;
    for (idx, observation) in observations.iter().take(displayed_observations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"iteration_idx\": {},",
            observation.iteration_idx
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_nodes\": [{}],",
            u32_list(&observation.decode_nodes)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_indices\": [{}],",
            u32_list(&observation.request_indices)
        )?;
        writeln!(
            writer,
            "{indent}      \"operation_count\": {},",
            observation.operation_ids.len()
        )?;
        if include_operation_ids {
            writeln!(
                writer,
                "{indent}      \"operation_ids\": [{}],",
                usize_list(&observation.operation_ids)
            )?;
        }
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_ms(observation.start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_ms(observation.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"latency_ms\": {},",
            json_ms(observation.latency_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"batch_tokens\": {},",
            observation.batch_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"first_token_batch_tokens\": {},",
            observation.first_token_batch_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"tail_token_batch_tokens\": {}",
            observation.tail_token_batch_tokens
        )?;
        if idx + 1 < displayed_observations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_request_slo<W: Write>(
    writer: &mut W,
    indent: &str,
    slo: &crate::ServingRequestSlo,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"slo\": {{")?;
    writeln!(
        writer,
        "{indent}        \"ttft_ms\": {},",
        slo.ttft_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_ms\": {},",
        slo.tpot_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"itl_ms\": {},",
        slo.itl_s.map(json_ms).unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"e2el_ms\": {}",
        slo.e2el_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_metric_breakdowns<W: Write>(
    writer: &mut W,
    indent: &str,
    breakdowns: &[ServingMetricBreakdown],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"metric_breakdowns\": [")?;
    for (idx, breakdown) in breakdowns.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"group\": {},",
            json_string(&breakdown.group)
        )?;
        writeln!(
            writer,
            "{indent}      \"key\": {},",
            json_string(&breakdown.key)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            breakdown.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"completed_requests\": {},",
            breakdown.completed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"failed_requests\": {},",
            breakdown.failed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"rejected_requests\": {},",
            breakdown.rejected_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"timed_out_requests\": {},",
            breakdown.timed_out_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"cancelled_requests\": {},",
            breakdown.cancelled_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"output_tokens\": {},",
            breakdown.output_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"lifecycle_event_metric_request_count\": {},",
            breakdown.lifecycle_event_metric_request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"fallback_metric_request_count\": {},",
            breakdown.fallback_metric_request_count
        )?;
        write_measurement_metric_source_counts(
            writer,
            Some(breakdown.metric_source_counts.as_slice()),
            &format!("{indent}  "),
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_constrained_requests\": {},",
            breakdown.deadline_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_missed_requests\": {},",
            breakdown.deadline_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_miss_rate\": {},",
            json_optional_f64(breakdown.deadline_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_constrained_requests\": {},",
            breakdown.ttft_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_missed_requests\": {},",
            breakdown.ttft_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_miss_rate\": {},",
            json_optional_f64(breakdown.ttft_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_constrained_requests\": {},",
            breakdown.tpot_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_missed_requests\": {},",
            breakdown.tpot_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_miss_rate\": {},",
            json_optional_f64(breakdown.tpot_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_constrained_requests\": {},",
            breakdown.itl_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_missed_requests\": {},",
            breakdown.itl_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_miss_rate\": {},",
            json_optional_f64(breakdown.itl_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_constrained_requests\": {},",
            breakdown.e2el_slo_constrained_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_missed_requests\": {},",
            breakdown.e2el_slo_missed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_miss_rate\": {},",
            json_optional_f64(breakdown.e2el_slo_miss_rate)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_ms\": {},",
            json_ms(breakdown.ttft_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_p90_ms\": {},",
            json_ms(breakdown.ttft_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_p95_ms\": {},",
            json_ms(breakdown.ttft_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"ttft_max_ms\": {},",
            json_ms(breakdown.ttft_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_ms\": {},",
            json_ms(breakdown.tpot_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_p90_ms\": {},",
            json_ms(breakdown.tpot_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_p95_ms\": {},",
            json_ms(breakdown.tpot_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_max_ms\": {},",
            json_ms(breakdown.tpot_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_ms\": {},",
            json_ms(breakdown.itl_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_p90_ms\": {},",
            json_ms(breakdown.itl_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_p95_ms\": {},",
            json_ms(breakdown.itl_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_max_ms\": {},",
            json_ms(breakdown.itl_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"throughput_tokens_per_s\": {},",
            json_optional_f64(breakdown.throughput_tokens_per_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_ms\": {},",
            json_ms(breakdown.e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_p90_ms\": {},",
            json_ms(breakdown.e2el_p90_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_p95_ms\": {},",
            json_ms(breakdown.e2el_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_max_ms\": {}",
            json_ms(breakdown.e2el_max_s)
        )?;
        if idx + 1 < breakdowns.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_node_capacity<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingNodeCapacityObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"decode_node_capacity\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"node_id\": {},",
            observation.node_id
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_gpu_capacity<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingGpuCapacityObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"decode_gpu_capacity\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"node_id\": {},",
            observation.node_id
        )?;
        writeln!(
            writer,
            "{indent}      \"local_gpu_id\": {},",
            observation.local_gpu_id
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_traffic_class_capacity<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingTrafficClassCapacityObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"traffic_class_capacity\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&observation.name)
        )?;
        writeln!(
            writer,
            "{indent}      \"group\": {},",
            json_string(&observation.group)
        )?;
        writeln!(
            writer,
            "{indent}      \"key\": {},",
            json_string(&observation.key)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_prefill_tokens\": {},",
            json_optional_u64(observation.max_prefill_tokens)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_decode_sequences\": {},",
            json_optional_u32(observation.max_decode_sequences)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_resident_tokens\": {},",
            json_optional_u64(observation.max_resident_tokens)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_kv_blocks\": {},",
            json_optional_u64(observation.max_kv_blocks)
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_token_utilization\": {},",
            json_optional_f64(observation.prefill_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_worker_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingWorkerObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"serving_workers\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&observation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"node_id\": {},",
            observation.node_id
        )?;
        writeln!(
            writer,
            "{indent}      \"local_gpu_id\": {},",
            observation.local_gpu_id
        )?;
        writeln!(
            writer,
            "{indent}      \"configured_worker_slots\": {},",
            observation.configured_worker_slots
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_active_worker_slots\": {},",
            observation.peak_active_worker_slots
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_slot_utilization\": {},",
            json_optional_f64(observation.worker_slot_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            observation.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"completed_requests\": {},",
            observation.completed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"failed_requests\": {},",
            observation.failed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"input_tokens\": {},",
            observation.input_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"output_tokens\": {},",
            observation.output_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_ms\": {},",
            json_ms(observation.queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_p95_ms\": {},",
            json_ms(observation.queue_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_max_ms\": {},",
            json_ms(observation.queue_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_ms\": {},",
            json_ms(observation.worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_p95_ms\": {},",
            json_ms(observation.worker_queue_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_max_ms\": {},",
            json_ms(observation.worker_queue_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_ms\": {},",
            json_ms(observation.resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_p95_ms\": {},",
            json_ms(observation.resource_queue_p95_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_max_ms\": {},",
            json_ms(observation.resource_queue_max_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"service_ms\": {},",
            json_ms(observation.service_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"first_start_ms\": {},",
            json_ms(observation.first_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_finish_ms\": {},",
            json_ms(observation.last_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_prefill_tokens\": {},",
            observation.peak_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_decode_sequences\": {},",
            observation.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_resident_tokens\": {},",
            observation.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_blocks\": {},",
            observation.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_allocated_kv_tokens\": {},",
            observation.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_fragmentation_tokens\": {},",
            observation.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"peak_kv_block_table_bytes\": {},",
            observation.peak_kv_block_table_bytes
        )?;
        write_worker_kv_cache_owner_slots(writer, indent, &observation.kv_cache_owner_slots)?;
        writeln!(
            writer,
            "{indent}      \"decode_sequence_utilization\": {},",
            json_optional_f64(observation.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"resident_token_utilization\": {},",
            json_optional_f64(observation.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_utilization\": {}",
            json_optional_f64(observation.kv_block_utilization)
        )?;
        if idx + 1 < observations.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_worker_kv_cache_owner_slots<W: Write>(
    writer: &mut W,
    indent: &str,
    slots: &[ServingWorkerKvSlotObservation],
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"kv_cache_owner_slots\": [")?;
    for (idx, slot) in slots.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(writer, "{indent}          \"slot\": {},", slot.slot)?;
        writeln!(
            writer,
            "{indent}          \"peak_decode_sequences\": {},",
            slot.peak_decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_resident_tokens\": {},",
            slot.peak_resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_kv_blocks\": {},",
            slot.peak_kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_allocated_kv_tokens\": {},",
            slot.peak_allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_kv_fragmentation_tokens\": {},",
            slot.peak_kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"peak_kv_block_table_bytes\": {},",
            slot.peak_kv_block_table_bytes
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_sequence_utilization\": {},",
            json_optional_f64(slot.decode_sequence_utilization)
        )?;
        writeln!(
            writer,
            "{indent}          \"resident_token_utilization\": {},",
            json_optional_f64(slot.resident_token_utilization)
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_block_utilization\": {}",
            json_optional_f64(slot.kv_block_utilization)
        )?;
        if idx + 1 < slots.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ],")
}

fn write_service_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingServiceObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"serving_services\": [")?;
    for (idx, observation) in observations.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&observation.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"health\": {},",
            json_string(observation.health.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"accepts_requests\": {},",
            observation.accepts_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_scale\": {},",
            json_optional_f64(observation.worker_scale)
        )?;
        writeln!(
            writer,
            "{indent}      \"configured_worker_slots_per_gpu\": {},",
            observation.configured_worker_slots_per_gpu
        )?;
        writeln!(
            writer,
            "{indent}      \"effective_worker_slots_per_gpu\": {},",
            observation.effective_worker_slots_per_gpu
        )?;
        writeln!(
            writer,
            "{indent}      \"node_count\": {},",
            observation.node_count
        )?;
        writeln!(
            writer,
            "{indent}      \"gpu_count\": {},",
            observation.gpu_count
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            observation.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"admitted_requests\": {},",
            observation.admitted_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"completed_requests\": {},",
            observation.completed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"failed_requests\": {},",
            observation.failed_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"rejected_requests\": {},",
            observation.rejected_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"timed_out_requests\": {},",
            observation.timed_out_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"cancelled_requests\": {},",
            observation.cancelled_requests
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_cap_ms\": {},",
            json_optional_value(observation.queue_cap_s.map(|seconds| seconds * 1000.0))
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_cap_request_count\": {},",
            observation.queue_cap_request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_cap_hit_count\": {},",
            observation.queue_cap_hit_count
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iteration_queue_cap_ms\": {},",
            json_optional_value(
                observation
                    .decode_iteration_queue_cap_s
                    .map(|seconds| seconds * 1000.0)
            )
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iteration_queue_cap_request_count\": {},",
            observation.decode_iteration_queue_cap_request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iteration_queue_cap_hit_count\": {},",
            observation.decode_iteration_queue_cap_hit_count
        )?;
        writeln!(
            writer,
            "{indent}      \"backpressure_rejections\": {},",
            observation.backpressure_rejections
        )?;
        writeln!(
            writer,
            "{indent}      \"timeout_rejections\": {},",
            observation.timeout_rejections
        )?;
        writeln!(
            writer,
            "{indent}      \"backpressure_state\": {},",
            json_string(&observation.backpressure_state)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_slot_utilization\": {},",
            json_optional_f64(observation.worker_slot_utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_ms\": {},",
            json_optional_f64(observation.queue_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_p95_ms\": {},",
            json_optional_f64(observation.queue_p95_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_max_ms\": {},",
            json_optional_f64(observation.queue_max_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"worker_queue_ms\": {},",
            json_optional_f64(observation.worker_queue_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_queue_ms\": {},",
            json_optional_f64(observation.resource_queue_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"service_ms\": {}",
            json_optional_f64(observation.service_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < observations.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_resource_utilization<W: Write>(
    writer: &mut W,
    indent: &str,
    utilization: &[ResourceUtilization],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"resource_utilization\": [")?;
    for (idx, resource) in utilization.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&resource.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"busy_ms\": {},",
            json_optional_f64(resource.busy_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"utilization\": {},",
            json_optional_f64(resource.utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"operation_count\": {},",
            resource.operation_count
        )?;
        writeln!(
            writer,
            "{indent}      \"first_start_ms\": {},",
            json_optional_f64(resource.first_start_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_finish_ms\": {}",
            json_optional_f64(resource.last_finish_s * 1000.0)
        )?;
        if idx + 1 < utilization.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_serving_bottleneck_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summaries: &[ServingBottleneckSummary],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"bottleneck_summary\": [")?;
    for (idx, summary) in summaries.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"source\": {},",
            json_string(&summary.source)
        )?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&summary.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&summary.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&summary.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&summary.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"severity\": {},",
            json_string(&summary.severity)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(summary.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(summary.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {},",
            summary
                .unit
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&summary.message)
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {}",
            summary
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(writer, "{indent}    }}{}", comma(idx + 1 < summaries.len()))?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_pareto_dimensions<W: Write>(
    writer: &mut W,
    indent: &str,
    dimensions: &[ServingParetoDimension],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"pareto_dimensions\": [")?;
    for (idx, dimension) in dimensions.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"metric\": {},",
            json_string(&dimension.metric)
        )?;
        writeln!(
            writer,
            "{indent}      \"direction\": {},",
            json_string(&dimension.direction)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {}",
            json_string(&dimension.unit)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < dimensions.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_phase_resource_utilization<W: Write>(
    writer: &mut W,
    indent: &str,
    utilization: &[ServingPhaseResourceUtilization],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"phase_resource_utilization\": [")?;
    for (idx, resource) in utilization.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&resource.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource_kind\": {},",
            json_string(&resource.resource_kind)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&resource.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"busy_ms\": {},",
            json_optional_f64(resource.busy_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"utilization\": {},",
            json_optional_f64(resource.utilization)
        )?;
        writeln!(
            writer,
            "{indent}      \"operation_count\": {},",
            resource.operation_count
        )?;
        writeln!(
            writer,
            "{indent}      \"first_start_ms\": {},",
            json_optional_f64(resource.first_start_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_finish_ms\": {}",
            json_optional_f64(resource.last_finish_s * 1000.0)
        )?;
        if idx + 1 < utilization.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_kv_route_resource_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summaries: &[ServingKvRouteResourceSummary],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"kv_route_resource_summary\": [")?;
    for (idx, summary) in summaries.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"resource_id\": {},",
            json_string(&summary.resource_id)
        )?;
        writeln!(
            writer,
            "{indent}      \"kind\": {},",
            json_string(&summary.kind)
        )?;
        writeln!(
            writer,
            "{indent}      \"label\": {},",
            json_string(&summary.label)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_count\": {},",
            summary.request_count
        )?;
        writeln!(
            writer,
            "{indent}      \"path_observations\": {},",
            summary.path_observations
        )?;
        writeln!(
            writer,
            "{indent}      \"transfer_bytes\": {},",
            summary.transfer_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"estimated_transfer_ms\": {},",
            json_ms(summary.estimated_transfer_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"min_bandwidth_gbps\": {},",
            json_optional_f64(summary.min_bandwidth_gbps)
        )?;
        writeln!(
            writer,
            "{indent}      \"max_latency_ms\": {},",
            json_ms(summary.max_latency_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"rail_id\": {},",
            json_optional_u32(summary.rail_id)
        )?;
        write_kv_transfer_path_endpoint(writer, indent, "from", summary.from.as_ref(), true)?;
        write_kv_transfer_path_endpoint(writer, indent, "to", summary.to.as_ref(), false)?;
        if idx + 1 < summaries.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_kv_route_topology_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summary: &ServingKvRouteTopologySummary,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"kv_route_topology_summary\": {{")?;
    writeln!(
        writer,
        "{indent}    \"route_resource_count\": {},",
        summary.route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"inter_node_route_resource_count\": {},",
        summary.inter_node_route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"gpu_nic_route_resource_count\": {},",
        summary.gpu_nic_route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"intra_node_route_resource_count\": {},",
        summary.intra_node_route_resource_count
    )?;
    writeln!(
        writer,
        "{indent}    \"rail_count\": {},",
        summary.rail_count
    )?;
    writeln!(
        writer,
        "{indent}    \"rail_ids\": {},",
        json_u32_array(&summary.rail_ids)
    )?;
    writeln!(
        writer,
        "{indent}    \"single_rail_dependency\": {},",
        summary.single_rail_dependency
    )?;
    writeln!(
        writer,
        "{indent}    \"single_rail_id\": {},",
        json_optional_u32(summary.single_rail_id)
    )?;
    writeln!(
        writer,
        "{indent}    \"unrailed_inter_node_route_resource_count\": {}",
        summary.unrailed_inter_node_route_resource_count
    )?;
    writeln!(writer, "{indent}  }}{}", comma(trailing_comma))
}

fn write_topology_bottlenecks<W: Write>(
    writer: &mut W,
    indent: &str,
    bottlenecks: &[ServingTopologyBottleneckObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"topology_bottlenecks\": [")?;
    for (idx, bottleneck) in bottlenecks.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&bottleneck.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&bottleneck.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&bottleneck.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&bottleneck.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"severity\": {},",
            json_string(&bottleneck.severity)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(bottleneck.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(bottleneck.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {},",
            bottleneck
                .unit
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {},",
            json_string(&bottleneck.message)
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {}",
            bottleneck
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < bottlenecks.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn selected_occupancy_resources(
    utilization: &[ResourceUtilization],
    limit: Option<usize>,
) -> Vec<String> {
    let count = limit.unwrap_or(utilization.len()).min(utilization.len());
    utilization
        .iter()
        .take(count)
        .map(|resource| resource.resource.clone())
        .collect()
}

fn write_resource_occupancy<W: Write>(
    writer: &mut W,
    indent: &str,
    occupancy: &[ResourceOccupancySeries],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let bucket_count = occupancy
        .iter()
        .map(|series| series.buckets.len())
        .max()
        .unwrap_or(0);
    writeln!(
        writer,
        "{indent}  \"resource_occupancy_bucket_count\": {bucket_count},"
    )?;
    writeln!(
        writer,
        "{indent}  \"resource_occupancy_resource_count\": {},",
        occupancy.len()
    )?;
    writeln!(writer, "{indent}  \"resource_occupancy\": [")?;
    for (series_idx, series) in occupancy.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&series.resource)
        )?;
        writeln!(writer, "{indent}      \"buckets\": [")?;
        for (bucket_idx, bucket) in series.buckets.iter().enumerate() {
            writeln!(writer, "{indent}        {{")?;
            writeln!(
                writer,
                "{indent}          \"bucket_idx\": {},",
                bucket.bucket_idx
            )?;
            writeln!(
                writer,
                "{indent}          \"start_ms\": {},",
                json_ms(bucket.start_s)
            )?;
            writeln!(
                writer,
                "{indent}          \"finish_ms\": {},",
                json_ms(bucket.finish_s)
            )?;
            writeln!(
                writer,
                "{indent}          \"busy_ms\": {},",
                json_ms(bucket.busy_s)
            )?;
            writeln!(
                writer,
                "{indent}          \"utilization\": {},",
                json_optional_f64(bucket.utilization)
            )?;
            writeln!(
                writer,
                "{indent}          \"operation_count\": {}",
                bucket.operation_count
            )?;
            if bucket_idx + 1 < series.buckets.len() {
                writeln!(writer, "{indent}        }},")?;
            } else {
                writeln!(writer, "{indent}        }}")?;
            }
        }
        writeln!(writer, "{indent}      ]")?;
        if series_idx + 1 < occupancy.len() {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_critical_path<W: Write>(
    writer: &mut W,
    indent: &str,
    path: &CriticalPath,
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_steps = limit.map_or(path.steps.len(), |limit| limit.min(path.steps.len()));
    writeln!(
        writer,
        "{indent}  \"critical_path_ms\": {},",
        json_ms(path.total_s)
    )?;
    writeln!(
        writer,
        "{indent}  \"critical_path_step_count\": {},",
        path.steps.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"critical_path_truncated\": {},",
        displayed_steps < path.steps.len()
    )?;
    writeln!(writer, "{indent}  \"critical_path\": [")?;
    for (idx, step) in path.steps.iter().take(displayed_steps).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"operation_id\": {},",
            step.operation_id
        )?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&step.name)
        )?;
        write_string_vec(
            writer,
            &format!("{indent}    "),
            "resources",
            &step.resources,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "explicit_dependencies",
            &step.explicit_dependencies,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "resource_dependencies",
            &step.resource_dependencies,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_ms(step.start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_ms(step.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"duration_ms\": {},",
            json_ms(step.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"wait_ms\": {}",
            json_ms(step.wait_s)
        )?;
        if idx + 1 < displayed_steps {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_scheduled_operations<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    operations: &[ScheduledOperation],
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_operations = limit.map_or(operations.len(), |limit| limit.min(operations.len()));
    writeln!(
        writer,
        "{indent}  \"scheduled_operation_count\": {},",
        operations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"scheduled_operations_truncated\": {},",
        displayed_operations < operations.len()
    )?;
    writeln!(writer, "{indent}  \"{field}\": [")?;
    for (idx, operation) in operations.iter().take(displayed_operations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(writer, "{indent}      \"id\": {},", operation.id)?;
        writeln!(
            writer,
            "{indent}      \"name\": {},",
            json_string(&operation.name)
        )?;
        write_string_vec(
            writer,
            &format!("{indent}    "),
            "resources",
            &operation.resources,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "explicit_dependencies",
            &operation.explicit_dependencies,
            true,
        )?;
        write_usize_vec(
            writer,
            &format!("{indent}    "),
            "resource_dependencies",
            &operation.resource_dependencies,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_optional_f64(operation.start_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_optional_f64(operation.finish_s * 1000.0)
        )?;
        writeln!(
            writer,
            "{indent}      \"duration_ms\": {}",
            json_optional_f64((operation.finish_s - operation.start_s).max(0.0) * 1000.0)
        )?;
        if idx + 1 < displayed_operations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
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
mod tests {
    use super::*;
    use std::{fs, time::SystemTime};

    #[test]
    fn parses_required_args() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--top-k",
            "3",
        ])
        .unwrap();

        assert_eq!(args.cluster_path, PathBuf::from("cluster.toml"));
        assert_eq!(args.workload_path, PathBuf::from("workload.toml"));
        assert_eq!(args.top_k, 3);
        assert_eq!(args.request_metrics_csv_path, None);
        assert_eq!(args.request_lifecycle_events_csv_path, None);
        assert_eq!(args.serving_metrics_csv_path, None);
        assert_eq!(args.serving_metric_breakdowns_csv_path, None);
        assert_eq!(args.serving_services_csv_path, None);
        assert_eq!(args.serving_utilization_csv_path, None);
        assert_eq!(args.serving_memory_pressure_csv_path, None);
        assert_eq!(args.serving_timeline_csv_path, None);
        assert_eq!(args.serving_occupancy_csv_path, None);
        assert_eq!(args.serving_placement_evidence_csv_path, None);
        assert_eq!(args.serving_worker_evidence_csv_path, None);
        assert_eq!(args.serving_rejections_csv_path, None);
        assert_eq!(args.serving_route_paths_csv_path, None);
        assert_eq!(args.kv_route_resources_csv_path, None);
        assert_eq!(args.serving_bottlenecks_csv_path, None);
        assert_eq!(args.serving_phase_calibration_csv_path, None);
        assert_eq!(args.serving_approximations_csv_path, None);
        assert_eq!(args.calibration_residuals_csv_path, None);
        assert_eq!(args.scenario_sensitivity_csv_path, None);
        assert_eq!(args.rank_sensitivity_csv_path, None);
        assert_eq!(args.output_dir, None);
        assert_eq!(args.output_profile, OutputProfile::Summary);
        assert_eq!(args.format, OutputFormat::Text);
        assert!(!args.trace);
        assert_eq!(args.trace_limit, Some(DEFAULT_TRACE_LIMIT));
        assert_eq!(args.request_limit, Some(DEFAULT_REQUEST_OBSERVATION_LIMIT));
        assert!(!args.occupancy);
        assert_eq!(args.occupancy_buckets, DEFAULT_OCCUPANCY_BUCKETS);
        assert_eq!(
            args.occupancy_resource_limit,
            Some(DEFAULT_OCCUPANCY_RESOURCE_LIMIT)
        );
        assert!(!args.critical_path);
        assert_eq!(args.critical_path_limit, Some(DEFAULT_CRITICAL_PATH_LIMIT));
        assert_eq!(args.search_budget, RunSearchBudgetConfig::default());
        assert!(args.scenarios.is_empty());
    }

    #[test]
    fn parses_json_output_format() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--format",
            "json",
        ])
        .unwrap();

        assert_eq!(args.format, OutputFormat::Json);
    }

    #[test]
    fn parses_markdown_output_format() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--format",
            "markdown",
        ])
        .unwrap();

        assert_eq!(args.format, OutputFormat::Markdown);
    }

    #[test]
    fn trace_output_implies_json_format() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--trace",
        ])
        .unwrap();

        assert_eq!(args.format, OutputFormat::Json);
        assert!(args.trace);
        assert_eq!(args.trace_limit, Some(DEFAULT_TRACE_LIMIT));
    }

    #[test]
    fn parses_trace_limit_and_enables_trace() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--trace-limit",
            "3",
        ])
        .unwrap();

        assert_eq!(args.format, OutputFormat::Json);
        assert!(args.trace);
        assert_eq!(args.trace_limit, Some(3));
    }

    #[test]
    fn parses_request_limit() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--request-limit",
            "7",
        ])
        .unwrap();

        assert_eq!(args.request_limit, Some(7));
    }

    #[test]
    fn parses_request_metrics_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--request-metrics-csv",
            "request-metrics.csv",
        ])
        .unwrap();

        assert_eq!(
            args.request_metrics_csv_path,
            Some(PathBuf::from("request-metrics.csv"))
        );
    }

    #[test]
    fn parses_request_lifecycle_events_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--request-lifecycle-events-csv",
            "request-lifecycle-events.csv",
        ])
        .unwrap();

        assert_eq!(
            args.request_lifecycle_events_csv_path,
            Some(PathBuf::from("request-lifecycle-events.csv"))
        );
    }

    #[test]
    fn parses_serving_metrics_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-metrics-csv",
            "serving-metrics.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_metrics_csv_path,
            Some(PathBuf::from("serving-metrics.csv"))
        );
    }

    #[test]
    fn parses_serving_metric_breakdowns_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-metric-breakdowns-csv",
            "serving-metric-breakdowns.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_metric_breakdowns_csv_path,
            Some(PathBuf::from("serving-metric-breakdowns.csv"))
        );
    }

    #[test]
    fn parses_serving_services_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-services-csv",
            "serving-services.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_services_csv_path,
            Some(PathBuf::from("serving-services.csv"))
        );
    }

    #[test]
    fn parses_serving_utilization_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-utilization-csv",
            "serving-utilization.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_utilization_csv_path,
            Some(PathBuf::from("serving-utilization.csv"))
        );
    }

    #[test]
    fn parses_serving_memory_pressure_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-memory-pressure-csv",
            "serving-memory-pressure.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_memory_pressure_csv_path,
            Some(PathBuf::from("serving-memory-pressure.csv"))
        );
    }

    #[test]
    fn parses_serving_timeline_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-timeline-csv",
            "serving-timeline.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_timeline_csv_path,
            Some(PathBuf::from("serving-timeline.csv"))
        );
    }

    #[test]
    fn parses_serving_occupancy_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-occupancy-csv",
            "serving-occupancy.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_occupancy_csv_path,
            Some(PathBuf::from("serving-occupancy.csv"))
        );
    }

    #[test]
    fn parses_serving_placement_evidence_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-placement-evidence-csv",
            "serving-placement-evidence.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_placement_evidence_csv_path,
            Some(PathBuf::from("serving-placement-evidence.csv"))
        );
    }

    #[test]
    fn parses_serving_worker_evidence_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-worker-evidence-csv",
            "serving-worker-evidence.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_worker_evidence_csv_path,
            Some(PathBuf::from("serving-worker-evidence.csv"))
        );
    }

    #[test]
    fn parses_serving_rejections_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-rejections-csv",
            "serving-rejections.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_rejections_csv_path,
            Some(PathBuf::from("serving-rejections.csv"))
        );
    }

    #[test]
    fn parses_serving_route_paths_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-route-paths-csv",
            "serving-route-paths.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_route_paths_csv_path,
            Some(PathBuf::from("serving-route-paths.csv"))
        );
    }

    #[test]
    fn parses_kv_route_resources_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--kv-route-resources-csv",
            "kv-route-resources.csv",
        ])
        .unwrap();

        assert_eq!(
            args.kv_route_resources_csv_path,
            Some(PathBuf::from("kv-route-resources.csv"))
        );
    }

    #[test]
    fn parses_serving_bottlenecks_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-bottlenecks-csv",
            "serving-bottlenecks.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_bottlenecks_csv_path,
            Some(PathBuf::from("serving-bottlenecks.csv"))
        );
    }

    #[test]
    fn parses_serving_phase_calibration_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-phase-calibration-csv",
            "serving-phase-calibration.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_phase_calibration_csv_path,
            Some(PathBuf::from("serving-phase-calibration.csv"))
        );
    }

    #[test]
    fn parses_serving_approximations_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--serving-approximations-csv",
            "serving-approximations.csv",
        ])
        .unwrap();

        assert_eq!(
            args.serving_approximations_csv_path,
            Some(PathBuf::from("serving-approximations.csv"))
        );
    }

    #[test]
    fn parses_calibration_residuals_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--calibration-residuals-csv",
            "calibration-residuals.csv",
        ])
        .unwrap();

        assert_eq!(
            args.calibration_residuals_csv_path,
            Some(PathBuf::from("calibration-residuals.csv"))
        );
    }

    #[test]
    fn parses_scenario_sensitivity_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--scenario-sensitivity-csv",
            "scenario-sensitivity.csv",
        ])
        .unwrap();

        assert_eq!(
            args.scenario_sensitivity_csv_path,
            Some(PathBuf::from("scenario-sensitivity.csv"))
        );
    }

    #[test]
    fn parses_rank_sensitivity_csv_arg() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--rank-sensitivity-csv",
            "rank-sensitivity.csv",
        ])
        .unwrap();

        assert_eq!(
            args.rank_sensitivity_csv_path,
            Some(PathBuf::from("rank-sensitivity.csv"))
        );
    }

    #[test]
    fn parses_output_dir_and_profile() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--output-dir",
            "out/run-1",
            "--output-profile",
            "audit",
        ])
        .unwrap();

        assert_eq!(args.output_dir, Some(PathBuf::from("out/run-1")));
        assert_eq!(args.output_profile, OutputProfile::Audit);
    }

    #[test]
    fn output_dir_defaults_to_all_profiles() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--output-dir",
            "out/run-1",
        ])
        .unwrap();

        assert_eq!(args.output_dir, Some(PathBuf::from("out/run-1")));
        assert_eq!(args.output_profile, OutputProfile::All);
    }

    #[test]
    fn rejects_output_profile_without_output_dir() {
        let err = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--output-profile",
            "compare",
        ])
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("--output-profile requires --output-dir")
        );
    }

    #[test]
    fn output_profile_writes_files_under_output_directory() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let output_dir =
            std::env::temp_dir().join(format!("inference-sim-output-profile-compare-{nanos}"));
        let mut output = Vec::new();

        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                root.join("examples/h100_cluster.toml")
                    .display()
                    .to_string(),
                "--workload".to_string(),
                root.join("examples/homogeneous_serving_workload.toml")
                    .display()
                    .to_string(),
                "--output-dir".to_string(),
                output_dir.display().to_string(),
                "--output-profile".to_string(),
                "compare".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-prefill-candidates".to_string(),
                "1".to_string(),
                "--max-decode-candidates".to_string(),
                "1".to_string(),
                "--max-serving-pairs".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let manifest = String::from_utf8(output).unwrap();
        assert!(manifest.contains("profile=compare"));
        assert!(manifest.contains("primary="));
        assert!(output_dir.join("manifest.txt").exists());
        assert!(output_dir.join("compare/results.txt").exists());
        assert!(output_dir.join("compare/serving_metrics.csv").exists());
        assert!(output_dir.join("compare/metric_breakdowns.csv").exists());
        assert!(output_dir.join("compare/rejections.csv").exists());
        assert!(output_dir.join("compare/bottlenecks.csv").exists());
        assert!(output_dir.join("compare/rank_sensitivity.csv").exists());

        let _ = fs::remove_dir_all(output_dir);
    }

    #[test]
    fn output_dir_writes_each_profile_directory_by_default() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let output_dir =
            std::env::temp_dir().join(format!("inference-sim-output-profile-all-{nanos}"));
        let mut output = Vec::new();

        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                root.join("examples/h100_cluster.toml")
                    .display()
                    .to_string(),
                "--workload".to_string(),
                root.join("examples/homogeneous_serving_workload.toml")
                    .display()
                    .to_string(),
                "--output-dir".to_string(),
                output_dir.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-prefill-candidates".to_string(),
                "1".to_string(),
                "--max-decode-candidates".to_string(),
                "1".to_string(),
                "--max-serving-pairs".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let manifest = String::from_utf8(output).unwrap();
        assert!(manifest.contains("profile=summary"));
        assert!(manifest.contains("profile=compare"));
        assert!(manifest.contains("profile=calibration"));
        assert!(manifest.contains("profile=audit"));
        assert!(output_dir.join("summary/results.md").exists());
        assert!(output_dir.join("compare/results.txt").exists());
        assert!(output_dir.join("calibration/results.json").exists());
        assert!(output_dir.join("audit/results.json").exists());
        let summary = fs::read_to_string(output_dir.join("summary/results.md")).unwrap();
        assert!(summary.starts_with("# Inference Sim Summary"));
        assert!(summary.contains("## Overview"));
        assert!(summary.contains("| Rank | Status | TTFT ms | TPOT ms |"));
        assert!(summary.contains("## Candidate Notes"));

        let _ = fs::remove_dir_all(output_dir);
    }

    #[test]
    fn writes_calibration_residuals_csv_artifact() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("inference-sim-calibration-residuals-{nanos}.csv"));
        let args = CliArgs::parse([
            "inference-sim".to_string(),
            "--cluster".to_string(),
            "cluster.toml".to_string(),
            "--workload".to_string(),
            "workload.toml".to_string(),
            "--calibration-residuals-csv".to_string(),
            path.display().to_string(),
        ])
        .unwrap();
        let profile = CalibrationProfileMetadata {
            path: "profile.toml".to_string(),
            name: Some("profile,one".to_string()),
            hardware: None,
            fabric: None,
            model: None,
            dtype: None,
            serving_stack: None,
            serving_runtime_features: Vec::new(),
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
            valid_shape: None,
            invalid_shapes: Vec::new(),
            fits: Vec::new(),
            benchmarks: vec![CalibrationBenchmarkPoint {
                name: Some("decode \"b4\"".to_string()),
                kind: Some("serving".to_string()),
                phase: Some("decode".to_string()),
                hardware: Some("a100".to_string()),
                fabric: Some("hdr".to_string()),
                model: Some("test-model".to_string()),
                dtype: Some("bf16".to_string()),
                batch_size: Some(4),
                prompt_tokens: Some(1024),
                decode_tokens: Some(32),
                sequence_tokens: Some(2048),
                tensor_ranks: Some(4),
                pipeline_ranks: Some(1),
                expert_ranks: Some(1),
                data_ranks: Some(1),
                measured_ms: Some(10.0),
                predicted_ms: Some(12.0),
                throughput_tokens_per_s: Some(256.0),
                command: Some("bench decode".to_string()),
                source: Some("unit-test".to_string()),
                notes: Some("quoted csv fields".to_string()),
            }],
        };

        write_calibration_residuals_csv_if_configured(
            &args,
            Some("baseline"),
            Some(&profile),
            false,
        )
        .unwrap();

        let csv = fs::read_to_string(&path).unwrap();
        assert!(csv.starts_with("scenario,profile_path,profile_name"));
        assert!(csv.contains("baseline,profile.toml,\"profile,one\",1,\"decode \"\"b4\"\"\""));
        assert!(csv.contains(",10.000000000,12.000000000,2.000000000,2.000000000"));
        assert!(csv.contains(",20.000000000,20.000000000,watch,256.000000000,unit-test"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_occupancy_options_and_implies_json() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--occupancy-buckets",
            "4",
            "--occupancy-resource-limit",
            "2",
        ])
        .unwrap();

        assert_eq!(args.format, OutputFormat::Json);
        assert!(args.occupancy);
        assert_eq!(args.occupancy_buckets, 4);
        assert_eq!(args.occupancy_resource_limit, Some(2));
    }

    #[test]
    fn parses_critical_path_options_and_implies_json() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--critical-path-limit",
            "5",
        ])
        .unwrap();

        assert_eq!(args.format, OutputFormat::Json);
        assert!(args.critical_path);
        assert_eq!(args.critical_path_limit, Some(5));
    }

    #[test]
    fn parses_search_budget_options() {
        let args = CliArgs::parse([
            "inference-sim",
            "--cluster",
            "cluster.toml",
            "--workload",
            "workload.toml",
            "--max-candidates",
            "5",
            "--max-prefill-candidates",
            "2",
            "--max-decode-candidates",
            "3",
            "--max-serving-pairs",
            "4",
            "--max-search-runtime-ms",
            "25",
            "--drop-rejected-candidates",
        ])
        .unwrap();

        assert_eq!(args.search_budget.max_parallelism_candidates, Some(5));
        assert_eq!(args.search_budget.max_prefill_candidates, Some(2));
        assert_eq!(args.search_budget.max_decode_candidates, Some(3));
        assert_eq!(args.search_budget.max_serving_pairs, Some(4));
        assert_eq!(args.search_budget.max_runtime_ms, Some(25));
        assert_eq!(args.search_budget.retain_rejected_candidates, Some(false));
    }

    #[test]
    fn parses_run_config_and_cli_overrides() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let run_path = dir.join(format!("inference-sim-run-{nanos}.toml"));
        fs::write(
            &run_path,
            r#"
            schema_version = 1

            [run]
            cluster = "cluster.toml"
            workload = "workload.toml"

            [output]
            format = "json"
            top_k = 2
            output_dir = "artifacts"
            output_profile = "compare"
            request_metrics_csv = "request-metrics.csv"
            request_lifecycle_events_csv = "request-lifecycle-events.csv"
            serving_metrics_csv = "serving-metrics.csv"
            serving_metric_breakdowns_csv = "serving-metric-breakdowns.csv"
            serving_services_csv = "serving-services.csv"
            serving_utilization_csv = "serving-utilization.csv"
            serving_memory_pressure_csv = "serving-memory-pressure.csv"
            serving_timeline_csv = "serving-timeline.csv"
            serving_occupancy_csv = "serving-occupancy.csv"
            serving_placement_evidence_csv = "serving-placement-evidence.csv"
            serving_worker_evidence_csv = "serving-worker-evidence.csv"
            serving_rejections_csv = "serving-rejections.csv"
            serving_route_paths_csv = "serving-route-paths.csv"
            kv_route_resources_csv = "kv-route-resources.csv"
            serving_bottlenecks_csv = "serving-bottlenecks.csv"
            serving_phase_calibration_csv = "serving-phase-calibration.csv"
            serving_approximations_csv = "serving-approximations.csv"
            calibration_residuals_csv = "calibration-residuals.csv"
            scenario_sensitivity_csv = "scenario-sensitivity.csv"
            rank_sensitivity_csv = "rank-sensitivity.csv"
            trace = true
            trace_limit = 5
            request_limit = 0
            occupancy = true
            occupancy_buckets = 4
            occupancy_resource_limit = 0
            critical_path = true
            critical_path_limit = 7

            [search]
            max_parallelism_candidates = 2
            max_prefill_candidates = 3
            max_decode_candidates = 4
            max_serving_pairs = 5
            max_runtime_ms = 250
            retain_rejected_candidates = false

            [[scenarios]]
            name = "baseline"
            request_count = 2

            [[scenarios]]
            name = "burst"
            arrival_rate_scale = 2.0
            calibration_profile = "scenario-profile.toml"

            [scenarios.topology]
            interconnect_bandwidth_scale = 0.5
            interconnect_latency_scale = 2.0
            nic_bandwidth_scale = 0.75

            [[scenarios.topology.node_states]]
            group = "spare"
            state = "draining"

            [[scenarios.topology.disabled_gpus]]
            node = 0
            gpus = [1, 2]

            [[scenarios.topology.disabled_nics]]
            group = "h100"
            nic = 0

            [[scenarios.topology.degraded_gpus]]
            node = 0
            gpu = 3
            compute_scale = 0.5
            hbm_bandwidth_scale = 0.75

            [[scenarios.topology.degraded_nics]]
            node = 0
            nic = 1
            bandwidth_scale = 0.5
            latency_scale = 1.25

            [[scenarios.topology.degraded_rails]]
            rails = [2]
            bandwidth_scale = 0.6
            latency_scale = 1.4

            [[scenarios.topology.degraded_links]]
            from = 0
            to = 1
            from_gpu = 0
            to_gpu = 1
            rail = 0
            bandwidth_scale = 0.4
            latency_scale = 1.5
            "#,
        )
        .unwrap();

        let args = CliArgs::parse([
            "inference-sim".to_string(),
            "--run".to_string(),
            run_path.display().to_string(),
            "--top-k".to_string(),
            "3".to_string(),
            "--request-limit".to_string(),
            "9".to_string(),
            "--max-serving-pairs".to_string(),
            "6".to_string(),
            "--max-runtime-ms".to_string(),
            "50".to_string(),
            "--retain-rejected-candidates".to_string(),
        ])
        .unwrap();

        assert_eq!(args.cluster_path, dir.join("cluster.toml"));
        assert_eq!(args.workload_path, dir.join("workload.toml"));
        assert_eq!(args.top_k, 3);
        assert_eq!(
            args.request_metrics_csv_path,
            Some(dir.join("request-metrics.csv"))
        );
        assert_eq!(
            args.request_lifecycle_events_csv_path,
            Some(dir.join("request-lifecycle-events.csv"))
        );
        assert_eq!(
            args.serving_metrics_csv_path,
            Some(dir.join("serving-metrics.csv"))
        );
        assert_eq!(
            args.serving_metric_breakdowns_csv_path,
            Some(dir.join("serving-metric-breakdowns.csv"))
        );
        assert_eq!(
            args.serving_services_csv_path,
            Some(dir.join("serving-services.csv"))
        );
        assert_eq!(
            args.serving_utilization_csv_path,
            Some(dir.join("serving-utilization.csv"))
        );
        assert_eq!(
            args.serving_memory_pressure_csv_path,
            Some(dir.join("serving-memory-pressure.csv"))
        );
        assert_eq!(
            args.serving_timeline_csv_path,
            Some(dir.join("serving-timeline.csv"))
        );
        assert_eq!(
            args.serving_occupancy_csv_path,
            Some(dir.join("serving-occupancy.csv"))
        );
        assert_eq!(
            args.serving_placement_evidence_csv_path,
            Some(dir.join("serving-placement-evidence.csv"))
        );
        assert_eq!(
            args.serving_worker_evidence_csv_path,
            Some(dir.join("serving-worker-evidence.csv"))
        );
        assert_eq!(
            args.serving_rejections_csv_path,
            Some(dir.join("serving-rejections.csv"))
        );
        assert_eq!(
            args.serving_route_paths_csv_path,
            Some(dir.join("serving-route-paths.csv"))
        );
        assert_eq!(
            args.kv_route_resources_csv_path,
            Some(dir.join("kv-route-resources.csv"))
        );
        assert_eq!(
            args.serving_bottlenecks_csv_path,
            Some(dir.join("serving-bottlenecks.csv"))
        );
        assert_eq!(
            args.serving_phase_calibration_csv_path,
            Some(dir.join("serving-phase-calibration.csv"))
        );
        assert_eq!(
            args.serving_approximations_csv_path,
            Some(dir.join("serving-approximations.csv"))
        );
        assert_eq!(
            args.calibration_residuals_csv_path,
            Some(dir.join("calibration-residuals.csv"))
        );
        assert_eq!(
            args.scenario_sensitivity_csv_path,
            Some(dir.join("scenario-sensitivity.csv"))
        );
        assert_eq!(
            args.rank_sensitivity_csv_path,
            Some(dir.join("rank-sensitivity.csv"))
        );
        assert_eq!(args.output_dir, Some(dir.join("artifacts")));
        assert_eq!(args.output_profile, OutputProfile::Compare);
        assert_eq!(args.format, OutputFormat::Json);
        assert!(args.trace);
        assert_eq!(args.trace_limit, Some(5));
        assert_eq!(args.request_limit, Some(9));
        assert!(args.occupancy);
        assert_eq!(args.occupancy_buckets, 4);
        assert_eq!(args.occupancy_resource_limit, None);
        assert!(args.critical_path);
        assert_eq!(args.critical_path_limit, Some(7));
        assert_eq!(args.search_budget.max_parallelism_candidates, Some(2));
        assert_eq!(args.search_budget.max_prefill_candidates, Some(3));
        assert_eq!(args.search_budget.max_decode_candidates, Some(4));
        assert_eq!(args.search_budget.max_serving_pairs, Some(6));
        assert_eq!(args.search_budget.max_runtime_ms, Some(50));
        assert_eq!(args.search_budget.retain_rejected_candidates, Some(true));
        assert_eq!(args.scenarios.len(), 2);
        assert_eq!(args.scenarios[0].name, "baseline");
        assert_eq!(args.scenarios[0].request_count, Some(2));
        assert_eq!(args.scenarios[1].name, "burst");
        assert_eq!(args.scenarios[1].arrival_rate_scale, Some(2.0));
        assert_eq!(
            args.scenarios[1].calibration_profile_path,
            Some(dir.join("scenario-profile.toml"))
        );
        assert_eq!(
            args.scenarios[1].topology.interconnect_bandwidth_scale,
            Some(0.5)
        );
        assert_eq!(
            args.scenarios[1].topology.interconnect_latency_scale,
            Some(2.0)
        );
        assert_eq!(args.scenarios[1].topology.nic_bandwidth_scale, Some(0.75));
        assert_eq!(args.scenarios[1].topology.node_states.len(), 1);
        assert_eq!(
            args.scenarios[1].topology.node_states[0].node_groups,
            vec!["spare".to_string()]
        );
        assert_eq!(
            args.scenarios[1].topology.node_states[0].state,
            RunScenarioNodeState::Draining
        );
        assert_eq!(args.scenarios[1].topology.disabled_gpus.len(), 1);
        assert_eq!(
            args.scenarios[1].topology.disabled_gpus[0].node_ids,
            vec![0]
        );
        assert_eq!(
            args.scenarios[1].topology.disabled_gpus[0].gpu_ids,
            vec![1, 2]
        );
        assert_eq!(args.scenarios[1].topology.disabled_nics.len(), 1);
        assert_eq!(
            args.scenarios[1].topology.disabled_nics[0].node_groups,
            vec!["h100".to_string()]
        );
        assert_eq!(args.scenarios[1].topology.disabled_nics[0].nic_ids, vec![0]);
        assert_eq!(args.scenarios[1].topology.degraded_gpus.len(), 1);
        assert_eq!(
            args.scenarios[1].topology.degraded_gpus[0].node_ids,
            vec![0]
        );
        assert_eq!(args.scenarios[1].topology.degraded_gpus[0].gpu_ids, vec![3]);
        assert_eq!(
            args.scenarios[1].topology.degraded_gpus[0].compute_scale,
            Some(0.5)
        );
        assert_eq!(
            args.scenarios[1].topology.degraded_gpus[0].hbm_bandwidth_scale,
            Some(0.75)
        );
        assert_eq!(args.scenarios[1].topology.degraded_nics.len(), 1);
        assert_eq!(
            args.scenarios[1].topology.degraded_nics[0].node_ids,
            vec![0]
        );
        assert_eq!(args.scenarios[1].topology.degraded_nics[0].nic_ids, vec![1]);
        assert_eq!(
            args.scenarios[1].topology.degraded_nics[0].bandwidth_scale,
            Some(0.5)
        );
        assert_eq!(
            args.scenarios[1].topology.degraded_nics[0].latency_scale,
            Some(1.25)
        );
        assert_eq!(args.scenarios[1].topology.degraded_rails.len(), 1);
        assert_eq!(args.scenarios[1].topology.degraded_rails[0].rails, vec![2]);
        assert_eq!(
            args.scenarios[1].topology.degraded_rails[0].bandwidth_scale,
            Some(0.6)
        );
        assert_eq!(
            args.scenarios[1].topology.degraded_rails[0].latency_scale,
            Some(1.4)
        );
        assert_eq!(args.scenarios[1].topology.degraded_links.len(), 1);
        assert_eq!(
            args.scenarios[1].topology.degraded_links[0].from_node_ids,
            vec![0]
        );
        assert_eq!(
            args.scenarios[1].topology.degraded_links[0].to_node_ids,
            vec![1]
        );
        assert_eq!(
            args.scenarios[1].topology.degraded_links[0].from_gpus,
            vec![0]
        );
        assert_eq!(
            args.scenarios[1].topology.degraded_links[0].to_gpus,
            vec![1]
        );
        assert_eq!(args.scenarios[1].topology.degraded_links[0].rails, vec![0]);
        assert_eq!(
            args.scenarios[1].topology.degraded_links[0].bandwidth_scale,
            Some(0.4)
        );
        assert_eq!(
            args.scenarios[1].topology.degraded_links[0].latency_scale,
            Some(1.5)
        );

        let _ = fs::remove_file(run_path);
    }

    #[test]
    fn rejects_missing_workload_arg() {
        let err = CliArgs::parse(["inference-sim", "--cluster", "cluster.toml"]).unwrap_err();

        assert!(err.to_string().contains("missing required --workload"));
    }

    #[test]
    fn scenario_node_state_overlay_disables_whole_node_resources() {
        let mut cluster = crate::config::parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[nodes]]
            id = 0
            group = "prefill"
            node_tags = ["prefill-pool"]
            rack = "rack-a"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            group = "decode"
            node_tags = ["decode-pool"]
            rack = "rack-b"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let scenario = RunScenarioConfig {
            name: "maintenance".to_string(),
            request_count: None,
            arrival_gap_scale: None,
            arrival_rate_scale: None,
            batch_size_scale: None,
            prompt_tokens_scale: None,
            decode_tokens_scale: None,
            calibration_profile_path: None,
            calibration: RunScenarioCalibrationConfig::default(),
            topology: RunScenarioTopologyConfig {
                node_states: vec![RunScenarioNodeStateOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: vec!["decode_pool".to_string()],
                    racks: vec!["rack_b".to_string()],
                    islands: Vec::new(),
                    failure_domains: Vec::new(),
                    state: RunScenarioNodeState::Maintenance,
                }],
                ..RunScenarioTopologyConfig::default()
            },
        };

        apply_run_scenario_topology(&mut cluster, &scenario).unwrap();

        assert_eq!(cluster.available_gpus(), 2);
        assert_eq!(cluster.node(0).unwrap().available_gpu_count(), 2);
        let decode_node = cluster.node(1).unwrap();
        assert_eq!(
            decode_node.operational_state,
            NodeOperationalState::Maintenance
        );
        assert_eq!(decode_node.available_gpu_count(), 0);
        assert_eq!(decode_node.network.disabled_nics.len(), 2);
        let mut inventory = Vec::new();
        write_cluster_inventory_text(&mut inventory, &cluster).unwrap();
        let inventory = String::from_utf8(inventory).unwrap();
        assert!(inventory.contains("node_inventory node=1 state=maintenance"));

        let graph = TopologyGraph::from_cluster(&cluster);
        assert!(
            graph
                .route_between_gpus(
                    GpuAddr {
                        node_id: 0,
                        local_gpu_id: 0,
                    },
                    GpuAddr {
                        node_id: 1,
                        local_gpu_id: 0,
                    },
                    Bytes::from_megabytes(1.0),
                )
                .is_none()
        );
    }

    #[test]
    fn degraded_nic_overlay_scales_route_latency() {
        let mut cluster = crate::config::parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let scenario = RunScenarioConfig {
            name: "slow-nic".to_string(),
            request_count: None,
            arrival_gap_scale: None,
            arrival_rate_scale: None,
            batch_size_scale: None,
            prompt_tokens_scale: None,
            decode_tokens_scale: None,
            calibration_profile_path: None,
            calibration: RunScenarioCalibrationConfig::default(),
            topology: RunScenarioTopologyConfig {
                degraded_nics: vec![RunScenarioNicDegradationOverlay {
                    node_ids: vec![0],
                    node_groups: Vec::new(),
                    node_tags: Vec::new(),
                    racks: Vec::new(),
                    islands: Vec::new(),
                    failure_domains: Vec::new(),
                    nic_ids: vec![0],
                    bandwidth_scale: None,
                    latency_scale: Some(100.0),
                }],
                ..RunScenarioTopologyConfig::default()
            },
        };

        apply_run_scenario_topology(&mut cluster, &scenario).unwrap();

        assert_eq!(cluster.node(0).unwrap().network.nic_latency_scale(0), 100.0);
        let graph = TopologyGraph::from_cluster(&cluster);
        let path = graph
            .route_between_nodes(0, 1, Bytes::from_bytes(1))
            .expect("route should remain available through non-degraded NIC");
        assert!(path.labels.iter().any(|label| label.contains("rail 1")));
        assert!(!path.labels.iter().any(|label| label.contains("rail 0")));
    }

    #[test]
    fn degraded_rail_overlay_scales_matching_nics_and_links() {
        let mut cluster = crate::config::parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let scenario = RunScenarioConfig {
            name: "rail-brownout".to_string(),
            request_count: None,
            arrival_gap_scale: None,
            arrival_rate_scale: None,
            batch_size_scale: None,
            prompt_tokens_scale: None,
            decode_tokens_scale: None,
            calibration_profile_path: None,
            calibration: RunScenarioCalibrationConfig::default(),
            topology: RunScenarioTopologyConfig {
                degraded_rails: vec![RunScenarioRailDegradationOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: Vec::new(),
                    racks: Vec::new(),
                    islands: Vec::new(),
                    failure_domains: Vec::new(),
                    rails: vec![0],
                    bandwidth_scale: Some(0.01),
                    latency_scale: Some(100.0),
                }],
                ..RunScenarioTopologyConfig::default()
            },
        };

        apply_run_scenario_topology(&mut cluster, &scenario).unwrap();

        let node = cluster.node(0).unwrap();
        assert!(
            (node.network.nic_bandwidth(0).as_gigabits_per_sec() - 4.0).abs() < 1e-9,
            "rail 0 NIC bandwidth should be degraded"
        );
        assert_eq!(node.network.nic_latency_scale(0), 100.0);
        assert_eq!(node.network.nic_bandwidth(1).as_gigabits_per_sec(), 400.0);
        assert_eq!(node.network.nic_latency_scale(1), 1.0);

        let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
            panic!("expected custom topology");
        };
        let links = edges.get(&UnorderedPair::new(0, 1)).unwrap();
        let rail_zero = links.iter().find(|link| link.rail == Some(0)).unwrap();
        let rail_one = links.iter().find(|link| link.rail == Some(1)).unwrap();
        assert!((rail_zero.profile.bw.unidirectional.as_gigabits_per_sec() - 4.0).abs() < 1e-9);
        assert!((rail_zero.profile.latency.to_us() - 120.0).abs() < 1e-9);
        assert_eq!(
            rail_one.profile.bw.unidirectional.as_gigabits_per_sec(),
            400.0
        );

        let graph = TopologyGraph::from_cluster(&cluster);
        let path = graph
            .route_between_nodes(0, 1, Bytes::from_bytes(1))
            .expect("route should remain available through non-degraded rail");
        assert!(path.labels.iter().any(|label| label.contains("rail 1")));
        assert!(!path.labels.iter().any(|label| label.contains("rail 0")));
    }

    #[test]
    fn node_scoped_scenario_overlays_can_target_topology_selectors() {
        let mut cluster = crate::config::parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 1

            [[nodes]]
            id = 0
            group = "prefill"
            node_tags = ["prefill-pool"]
            rack = "rack-a"
            island = "island-a"
            failure_domain = "az-a"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            group = "decode"
            node_tags = ["decode-pool"]
            rack = "rack-b"
            island = "island-b"
            failure_domain = "az-b"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 2
            group = "spare"
            node_tags = ["spare-pool"]
            rack = "rack-c"
            island = "island-c"
            failure_domain = "az-c"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let scenario = RunScenarioConfig {
            name: "topology-targets".to_string(),
            request_count: None,
            arrival_gap_scale: None,
            arrival_rate_scale: None,
            batch_size_scale: None,
            prompt_tokens_scale: None,
            decode_tokens_scale: None,
            calibration_profile_path: None,
            calibration: RunScenarioCalibrationConfig::default(),
            topology: RunScenarioTopologyConfig {
                node_states: vec![RunScenarioNodeStateOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: Vec::new(),
                    racks: Vec::new(),
                    islands: Vec::new(),
                    failure_domains: vec!["az_c".to_string()],
                    state: RunScenarioNodeState::Reserved,
                }],
                disabled_gpus: vec![RunScenarioGpuResourceOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: vec!["prefill_pool".to_string()],
                    racks: Vec::new(),
                    islands: Vec::new(),
                    failure_domains: Vec::new(),
                    gpu_ids: vec![1],
                }],
                disabled_nics: vec![RunScenarioNicResourceOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: Vec::new(),
                    racks: vec!["rack_a".to_string()],
                    islands: Vec::new(),
                    failure_domains: Vec::new(),
                    nic_ids: vec![1],
                }],
                degraded_gpus: vec![RunScenarioGpuDegradationOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: Vec::new(),
                    racks: Vec::new(),
                    islands: vec!["island_b".to_string()],
                    failure_domains: Vec::new(),
                    gpu_ids: vec![0],
                    compute_scale: Some(0.5),
                    hbm_bandwidth_scale: None,
                    hbm_capacity_scale: None,
                }],
                degraded_nics: vec![RunScenarioNicDegradationOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: Vec::new(),
                    racks: Vec::new(),
                    islands: Vec::new(),
                    failure_domains: vec!["az_b".to_string()],
                    nic_ids: vec![0],
                    bandwidth_scale: None,
                    latency_scale: Some(2.0),
                }],
                degraded_rails: vec![RunScenarioRailDegradationOverlay {
                    node_ids: Vec::new(),
                    node_groups: Vec::new(),
                    node_tags: vec!["decode_pool".to_string()],
                    racks: Vec::new(),
                    islands: Vec::new(),
                    failure_domains: Vec::new(),
                    rails: vec![1],
                    bandwidth_scale: Some(0.5),
                    latency_scale: Some(3.0),
                }],
                ..RunScenarioTopologyConfig::default()
            },
        };

        apply_run_scenario_topology(&mut cluster, &scenario).unwrap();

        let prefill_node = cluster.node(0).unwrap();
        assert!(prefill_node.disabled_gpus.contains(&1));
        assert!(prefill_node.network.disabled_nics.contains(&1));
        assert_eq!(
            prefill_node.network.nic_bandwidth(1).as_gigabits_per_sec(),
            400.0
        );

        let decode_node = cluster.node(1).unwrap();
        let degraded_gpu = decode_node.gpu_profile(0).unwrap();
        let base_gpu = decode_node.gpu_profile(1).unwrap();
        assert!((degraded_gpu.peak_f16_flops - base_gpu.peak_f16_flops * 0.5).abs() < 1e-9);
        assert_eq!(decode_node.network.nic_latency_scale(0), 2.0);
        assert_eq!(
            decode_node.network.nic_bandwidth(0).as_gigabits_per_sec(),
            400.0
        );
        assert_eq!(decode_node.network.nic_latency_scale(1), 3.0);
        assert_eq!(
            decode_node.network.nic_bandwidth(1).as_gigabits_per_sec(),
            200.0
        );

        let spare_node = cluster.node(2).unwrap();
        assert_eq!(spare_node.operational_state, NodeOperationalState::Reserved);
        assert_eq!(spare_node.available_gpu_count(), 0);
        assert_eq!(spare_node.network.disabled_nics.len(), 2);

        let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
            panic!("expected custom topology");
        };
        let links = edges.get(&UnorderedPair::new(0, 1)).unwrap();
        let rail_zero = links.iter().find(|link| link.rail == Some(0)).unwrap();
        let rail_one = links.iter().find(|link| link.rail == Some(1)).unwrap();
        assert_eq!(
            rail_zero.profile.bw.unidirectional.as_gigabits_per_sec(),
            400.0
        );
        assert_eq!(
            rail_one.profile.bw.unidirectional.as_gigabits_per_sec(),
            200.0
        );
    }

    #[test]
    fn degraded_link_overlay_can_target_gpu_scoped_custom_links() {
        let mut cluster = crate::config::parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu = 0
            to_gpu = 0
            kind = "ethernet"
            variant = "100g"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu = 1
            to_gpu = 1
            kind = "ib"
            variant = "hdr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let scenario = RunScenarioConfig {
            name: "slow-gpu-link".to_string(),
            request_count: None,
            arrival_gap_scale: None,
            arrival_rate_scale: None,
            batch_size_scale: None,
            prompt_tokens_scale: None,
            decode_tokens_scale: None,
            calibration_profile_path: None,
            calibration: RunScenarioCalibrationConfig::default(),
            topology: RunScenarioTopologyConfig {
                degraded_links: vec![RunScenarioLinkDegradationOverlay {
                    from_node_ids: vec![0],
                    from_node_groups: Vec::new(),
                    from_node_tags: Vec::new(),
                    from_racks: Vec::new(),
                    from_islands: Vec::new(),
                    from_failure_domains: Vec::new(),
                    from_gpus: vec![1],
                    to_node_ids: vec![1],
                    to_node_groups: Vec::new(),
                    to_node_tags: Vec::new(),
                    to_racks: Vec::new(),
                    to_islands: Vec::new(),
                    to_failure_domains: Vec::new(),
                    to_gpus: vec![1],
                    rails: Vec::new(),
                    bandwidth_scale: Some(0.5),
                    latency_scale: Some(2.0),
                }],
                ..RunScenarioTopologyConfig::default()
            },
        };

        apply_run_scenario_topology(&mut cluster, &scenario).unwrap();

        let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
            panic!("expected custom topology");
        };
        let links = edges.get(&UnorderedPair::new(0, 1)).unwrap();
        let ethernet = links
            .iter()
            .find(|link| link.profile.label == "Ethernet 100G")
            .unwrap();
        let hdr = links
            .iter()
            .find(|link| link.profile.label == "IB HDR")
            .unwrap();

        assert_eq!(
            ethernet.profile.bw.unidirectional.as_gigabits_per_sec(),
            100.0
        );
        assert_eq!(ethernet.profile.latency.to_us(), 10.0);
        assert_eq!(hdr.profile.bw.unidirectional.as_gigabits_per_sec(), 100.0);
        assert_eq!(hdr.profile.latency.to_us(), 3.0);
    }

    #[test]
    fn degraded_link_overlay_can_target_topology_selectors() {
        let mut cluster = crate::config::parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"

            [[interconnect.links]]
            from = 0
            to = 2
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            node_tags = ["prefill-pool"]
            rack = "rack-a"
            island = "island-a"
            failure_domain = "az-a"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 1
            node_tags = ["decode-pool"]
            rack = "rack-b"
            island = "island-b"
            failure_domain = "az-b"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }

            [[nodes]]
            id = 2
            node_tags = ["decode-pool"]
            rack = "rack-c"
            island = "island-c"
            failure_domain = "az-c"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 2 }
            "#,
        )
        .unwrap();
        let scenario = RunScenarioConfig {
            name: "rack-b-slowdown".to_string(),
            request_count: None,
            arrival_gap_scale: None,
            arrival_rate_scale: None,
            batch_size_scale: None,
            prompt_tokens_scale: None,
            decode_tokens_scale: None,
            calibration_profile_path: None,
            calibration: RunScenarioCalibrationConfig::default(),
            topology: RunScenarioTopologyConfig {
                degraded_links: vec![RunScenarioLinkDegradationOverlay {
                    from_node_ids: Vec::new(),
                    from_node_groups: Vec::new(),
                    from_node_tags: vec!["prefill_pool".to_string()],
                    from_racks: Vec::new(),
                    from_islands: Vec::new(),
                    from_failure_domains: Vec::new(),
                    from_gpus: Vec::new(),
                    to_node_ids: Vec::new(),
                    to_node_groups: Vec::new(),
                    to_node_tags: vec!["decode_pool".to_string()],
                    to_racks: vec!["rack_b".to_string()],
                    to_islands: Vec::new(),
                    to_failure_domains: vec!["az_b".to_string()],
                    to_gpus: Vec::new(),
                    rails: Vec::new(),
                    bandwidth_scale: Some(0.5),
                    latency_scale: Some(2.0),
                }],
                ..RunScenarioTopologyConfig::default()
            },
        };

        apply_run_scenario_topology(&mut cluster, &scenario).unwrap();

        let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
            panic!("expected custom topology");
        };
        let rack_b_link = edges
            .get(&UnorderedPair::new(0, 1))
            .unwrap()
            .first()
            .unwrap();
        let rack_c_link = edges
            .get(&UnorderedPair::new(0, 2))
            .unwrap()
            .first()
            .unwrap();

        assert_eq!(
            rack_b_link.profile.bw.unidirectional.as_gigabits_per_sec(),
            100.0
        );
        assert_eq!(rack_b_link.profile.latency.to_us(), 3.0);
        assert_eq!(
            rack_c_link.profile.bw.unidirectional.as_gigabits_per_sec(),
            400.0
        );
        assert_eq!(rack_c_link.profile.latency.to_us(), 1.2);
    }

    #[test]
    fn runs_solver_from_toml_files() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-workload-{nanos}.toml"));

        fs::write(
            &cluster_path,
            r#"
            schema_version = 1

            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[node_groups]]
            label = "h100"
            start_id = 0
            count = 1
            gpu = "h100_sxm"
            gpu_count = 8
            nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 2
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1, 2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("cluster_gpus=8"));
        assert!(output.contains("cluster_inventory"));
        assert!(output.contains("node_inventory node=0"));
        assert!(output.contains("H100 SXM5:8"));
        assert!(output.contains("trust_boundary=v1_approximate"));
        assert!(output.contains("coarse_topology"));
        assert!(output.contains("calibration_dependent"));
        assert!(output.contains("searched_configs=2"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn emits_runtime_search_budget_diagnostics_from_cli() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-runtime-budget-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!(
            "inference-sim-runtime-budget-workload-{nanos}.toml"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 2
            hidden_size = 1024
            attention_heads = 8
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 1.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 32
            decode_tokens = 4
            max_sequence_tokens = 64
            phase = "end_to_end"

            [search]
            tensor_ranks = [1, 2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut text_output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--max-search-runtime-ms".to_string(),
                "0".to_string(),
            ],
            &mut text_output,
        )
        .unwrap();
        let text_output = String::from_utf8(text_output).unwrap();
        assert!(text_output.contains("searched_configs=0"));
        assert!(text_output.contains("max_runtime_ms=0"));
        assert!(text_output.contains("truncated=true"));
        assert!(text_output.contains("truncated_runtime=true"));

        let mut json_output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--max-search-runtime-ms".to_string(),
                "0".to_string(),
            ],
            &mut json_output,
        )
        .unwrap();
        let json_output = String::from_utf8(json_output).unwrap();
        assert!(json_output.contains("\"searched_configs\": 0"));
        assert!(json_output.contains("\"max_runtime_ms\": 0"));
        assert!(json_output.contains("\"truncated\": true"));
        assert!(json_output.contains("\"truncated_by_runtime_budget\": true"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn runtime_search_budget_stops_serving_pair_search() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!(
            "inference-sim-serving-runtime-budget-cluster-{nanos}.toml"
        ));
        let workload_path = dir.join(format!(
            "inference-sim-serving-runtime-budget-workload-{nanos}.toml"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 2
            hidden_size = 1024
            attention_heads = 8
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 1.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 32
            decode_tokens = 4
            max_sequence_tokens = 64
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 1
            arrival = "fixed"
            arrival_gap_ms = 1.0
            batch_sizes = [1]
            prompt_tokens = [32]
            decode_tokens = [4]

            [serving.prefill_search]
            tensor_ranks = [1, 2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving.decode_search]
            tensor_ranks = [1, 2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--max-search-runtime-ms".to_string(),
                "0".to_string(),
            ],
            &mut output,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("searched_serving_pairs=0"));
        assert!(output.contains("mode=serving"));
        assert!(output.contains("prefill_candidate_space=2"));
        assert!(output.contains("decode_candidate_space=2"));
        assert!(output.contains("max_runtime_ms=0"));
        assert!(output.contains("truncated_runtime=true"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn runs_solver_from_run_toml_file() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-run-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-run-workload-{nanos}.toml"));
        let run_path = dir.join(format!("inference-sim-run-config-{nanos}.toml"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 8
            nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 2
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();
        fs::write(
            &run_path,
            format!(
                r#"
            schema_version = 1
            cluster = "{}"
            workload = "{}"

            [output]
            top_k = 1
            format = "text"

            [search]
            max_candidates = 1
            "#,
                cluster_path.display(),
                workload_path.display()
            ),
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--run".to_string(),
                run_path.display().to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("cluster_gpus=8"));
        assert!(output.contains("cluster_inventory"));
        assert!(output.contains("searched_configs=1"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
        let _ = fs::remove_file(run_path);
    }

    #[test]
    fn runs_serving_scenario_sweep_from_run_toml_file() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-sweep-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-sweep-workload-{nanos}.toml"));
        let run_path = dir.join(format!("inference-sim-sweep-run-{nanos}.toml"));
        let profile_path = dir.join(format!("inference-sim-sweep-profile-{nanos}.toml"));
        let scenario_sensitivity_path =
            dir.join(format!("inference-sim-sweep-sensitivity-{nanos}.csv"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 8
            nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 8
            nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 64
            decode_tokens = 8
            max_sequence_tokens = 128
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 2
            arrival_gap_s = 0.001
            "#,
        )
        .unwrap();
        fs::write(
            &run_path,
            format!(
                r#"
            schema_version = 1
            cluster = "{}"
            workload = "{}"

            [output]
            top_k = 1
            format = "json"
            scenario_sensitivity_csv = "{}"

            [[scenarios]]
            name = "baseline"
            request_count = 2

            [[scenarios]]
            name = "burst"
            request_count = 3
            arrival_rate_scale = 2.0
            prompt_tokens_scale = 2.0
            calibration_profile = "{}"

            [scenarios.calibration]
            decode_memory_bandwidth_scale = 0.8
            serving_memory_runtime_reserve_fraction = 0.09

            [scenarios.topology]
            interconnect_bandwidth_scale = 0.5
            interconnect_latency_scale = 2.0
            nic_bandwidth_scale = 0.75

            [[scenarios.topology.disabled_gpus]]
            node = 0
            gpu = 0

            [[scenarios.topology.disabled_nics]]
            node = 0
            nic = 0

            [[scenarios.topology.degraded_gpus]]
            node = 1
            gpu = 0
            compute_scale = 0.5
            hbm_bandwidth_scale = 0.75

            [[scenarios.topology.degraded_nics]]
            node = 0
            nic = 1
            bandwidth_scale = 0.5
            latency_scale = 1.25

            [[scenarios.topology.degraded_rails]]
            rails = [2]
            bandwidth_scale = 0.6
            latency_scale = 1.4

            [[scenarios.topology.degraded_links]]
            from = 0
            to = 1
            rail = 0
            bandwidth_scale = 0.4
            latency_scale = 1.5
            "#,
                cluster_path.display(),
                workload_path.display(),
                scenario_sensitivity_path.display(),
                profile_path.display()
            ),
        )
        .unwrap();
        fs::write(
            &profile_path,
            r#"
            [profile]
            name = "scenario-slow-decode"

            [calibration]
            decode_compute_scale = 1.7
            kv_transfer_scale = 1.3
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--run".to_string(),
                run_path.display().to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"mode\": \"scenario_sweep\""));
        assert!(output.contains("\"scenario_count\": 2"));
        assert!(output.contains("\"name\": \"baseline\""));
        assert!(output.contains("\"name\": \"burst\""));
        assert!(output.contains("\"name\": \"scenario-slow-decode\""));
        assert!(output.contains("\"decode_compute_scale\": 1.7"));
        assert!(output.contains("\"decode_memory_bandwidth_scale\": 0.8"));
        assert!(output.contains("\"serving_memory_runtime_reserve_fraction\": 0.09"));
        assert!(output.contains("\"kv_transfer_scale\": 1.3"));
        assert!(output.contains("\"topology\""));
        assert!(output.contains("\"interconnect_bandwidth_scale\": 0.5"));
        assert!(output.contains("\"interconnect_latency_scale\": 2"));
        assert!(output.contains("\"nic_bandwidth_scale\": 0.75"));
        assert!(output.contains("\"disabled_gpus\""));
        assert!(output.contains("\"gpu_ids\": [0]"));
        assert!(output.contains("\"disabled_nics\""));
        assert!(output.contains("\"nic_ids\": [0]"));
        assert!(output.contains("\"degraded_gpus\""));
        assert!(output.contains("\"compute_scale\": 0.5"));
        assert!(output.contains("\"degraded_nics\""));
        assert!(output.contains("\"bandwidth_scale\": 0.5"));
        assert!(output.contains("\"latency_scale\": 1.25"));
        assert!(output.contains("\"degraded_rails\""));
        assert!(output.contains("\"rails\": [2]"));
        assert!(output.contains("\"latency_scale\": 1.4"));
        assert!(output.contains("\"degraded_links\""));
        assert!(output.contains("\"from_node_ids\": [0]"));
        assert!(output.contains("\"from_node_tags\""));
        assert!(output.contains("\"from_racks\""));
        assert!(output.contains("\"to_node_ids\": [1]"));
        assert!(output.contains("\"to_node_tags\""));
        assert!(output.contains("\"to_racks\""));
        assert!(output.contains("\"latency_scale\": 1.5"));
        assert!(output.contains("\"bandwidth_overrides\""));
        assert!(output.contains("\"latency_scale_overrides\""));
        assert!(output.contains("\"nic_id\": 1"));
        assert!(output.contains("\"nic_id\": 2"));
        assert!(output.contains("\"hbm_bandwidth_gb_s\": 2512.500000"));
        assert!(output.contains("\"peak_f16_tflops\": 494.750000"));
        assert!(output.contains("\"bandwidth_gbps\": 80.000000"));
        assert!(output.contains("\"latency_us\": 3.600000"));
        assert!(output.contains("\"available_gpus\": 15"));
        assert_eq!(output.matches("\"mode\": \"serving\"").count(), 2);
        assert_eq!(output.matches("\"searched_serving_pairs\": 1").count(), 2);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let sensitivity = parsed["scenario_sensitivity"].as_array().unwrap();
        assert_eq!(sensitivity.len(), 2);
        assert_eq!(sensitivity[0]["name"].as_str(), Some("baseline"));
        assert_eq!(sensitivity[0]["baseline"].as_bool(), Some(true));
        assert_eq!(sensitivity[0]["baseline_name"].as_str(), Some("baseline"));
        assert_eq!(sensitivity[0]["available"].as_bool(), Some(true));
        assert_eq!(sensitivity[0]["ttft_ms_delta"].as_f64(), Some(0.0));
        assert_eq!(
            sensitivity[0]["throughput_tokens_per_s_delta"].as_f64(),
            Some(0.0)
        );
        assert!(
            sensitivity[0]["hardware_unique_gpu_count"]
                .as_u64()
                .is_some()
        );
        assert!(
            sensitivity[0]["hardware_prefill_gpu_count"]
                .as_u64()
                .is_some()
        );
        assert!(
            sensitivity[0]["hardware_decode_gpu_count"]
                .as_u64()
                .is_some()
        );
        assert!(
            sensitivity[0]["hardware_aggregate_gpu_types"]
                .as_str()
                .is_some()
        );
        assert!(
            sensitivity[0]["hardware_throughput_tokens_per_s_per_gpu"]
                .as_f64()
                .is_some()
        );
        assert!(sensitivity[0]["calibration_status"].as_str().is_some());
        assert!(
            sensitivity[0]["calibration_coverage_fraction"]
                .as_f64()
                .is_some()
        );
        assert!(sensitivity[0]["calibration_fit_count"].as_u64().is_some());
        assert!(sensitivity[0]["approximation_status"].as_str().is_some());
        assert!(sensitivity[0]["approximation_count"].as_u64().is_some());
        assert!(
            sensitivity[0]["approximation_coarse_topology"]
                .as_bool()
                .is_some()
        );
        assert!(
            sensitivity[0]["approximation_approximate_queueing"]
                .as_bool()
                .is_some()
        );
        assert!(
            sensitivity[0]["approximation_category_counts"]
                .as_str()
                .is_some()
        );
        assert!(sensitivity[0]["approximation_top_codes"].as_str().is_some());
        assert!(sensitivity[0]["bottleneck_count"].as_u64().is_some());
        assert!(sensitivity[0]["rejection_count"].as_u64().is_some());
        assert_eq!(sensitivity[1]["name"].as_str(), Some("burst"));
        assert_eq!(sensitivity[1]["baseline"].as_bool(), Some(false));
        assert_eq!(sensitivity[1]["baseline_name"].as_str(), Some("baseline"));
        assert_eq!(sensitivity[1]["available"].as_bool(), Some(false));
        assert_eq!(sensitivity[1]["ttft_ms"], serde_json::Value::Null);
        assert_eq!(sensitivity[1]["ttft_ms_delta"], serde_json::Value::Null);
        assert_eq!(sensitivity[1]["tpot_ms_delta"], serde_json::Value::Null);
        assert_eq!(
            sensitivity[1]["throughput_tokens_per_s_delta"],
            serde_json::Value::Null
        );
        assert_eq!(sensitivity[1]["e2el_ms_delta"], serde_json::Value::Null);
        assert!(sensitivity[1]["reason"].as_str().is_some());
        assert!(sensitivity[1]["calibration_status"].as_str().is_some());
        assert!(sensitivity[1]["approximation_status"].as_str().is_some());
        assert!(sensitivity[1]["rejection_count"].as_u64().is_some());
        assert!(
            sensitivity[1]["top_rejection_code"].as_str().is_some()
                || sensitivity[1]["rejected_reason"].as_str().is_some()
        );
        let scenario_sensitivity_csv = fs::read_to_string(&scenario_sensitivity_path).unwrap();
        assert!(scenario_sensitivity_csv.starts_with("scenario_index,scenario,available"));
        assert!(scenario_sensitivity_csv.contains("1,baseline,true"));
        assert!(scenario_sensitivity_csv.contains("2,burst,false"));
        assert!(scenario_sensitivity_csv.contains(",baseline,"));
        assert!(scenario_sensitivity_csv.contains("rejected_reason"));
        assert!(scenario_sensitivity_csv.contains("hardware_unique_gpu_count"));
        assert!(scenario_sensitivity_csv.contains("hardware_aggregate_gpu_types"));
        assert!(scenario_sensitivity_csv.contains("hardware_throughput_tokens_per_s_per_gpu"));
        assert!(scenario_sensitivity_csv.contains("H100 SXM5"));
        assert!(scenario_sensitivity_csv.contains("calibration_status"));
        assert!(scenario_sensitivity_csv.contains("calibration_fit_count_with_uncertainty"));
        assert!(scenario_sensitivity_csv.contains("approximation_status"));
        assert!(scenario_sensitivity_csv.contains("approximation_coarse_topology"));
        assert!(scenario_sensitivity_csv.contains("approximation_category_counts"));
        assert!(scenario_sensitivity_csv.contains("approximation_top_codes"));
        assert!(scenario_sensitivity_csv.contains("approximate_serving_event_loop"));
        assert!(scenario_sensitivity_csv.contains("bottleneck_count"));
        assert!(scenario_sensitivity_csv.contains("top_bottleneck_code"));
        assert!(scenario_sensitivity_csv.contains("rejection_count"));
        assert!(scenario_sensitivity_csv.contains("top_rejection_code"));
        assert!(scenario_sensitivity_csv.contains("throughput_tokens_per_s_delta"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
        let _ = fs::remove_file(run_path);
        let _ = fs::remove_file(profile_path);
        let _ = fs::remove_file(scenario_sensitivity_path);
    }

    #[test]
    fn emits_json_solver_results_from_toml_files() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-json-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-json-workload-{nanos}.toml"));
        let rank_sensitivity_path =
            dir.join(format!("inference-sim-json-rank-sensitivity-{nanos}.csv"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 2
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1, 2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--format".to_string(),
                "json".to_string(),
                "--trace".to_string(),
                "--occupancy-buckets".to_string(),
                "2".to_string(),
                "--occupancy-resource-limit".to_string(),
                "1".to_string(),
                "--critical-path-limit".to_string(),
                "2".to_string(),
                "--rank-sensitivity-csv".to_string(),
                rank_sensitivity_path.display().to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.starts_with("{\n"));
        assert!(output.contains("\"mode\": \"parallelism\""));
        assert!(output.contains("\"cluster_inventory\""));
        assert!(output.contains("\"gpu_types\""));
        assert!(output.contains("\"H100 SXM5\""));
        assert!(output.contains("\"inter_node_topology\""));
        assert!(output.contains("\"trust_boundary\""));
        assert!(output.contains("\"status\": \"v1_approximate\""));
        assert!(output.contains("\"code\": \"coarse_topology\""));
        assert!(output.contains("\"code\": \"calibration_dependent\""));
        assert!(output.contains("\"code\": \"unsupported_locality_detail\""));
        assert!(output.contains("\"candidate_id\""));
        assert!(output.contains("\"candidate_id\": \"tp1-pp1-ep1-dp1\""));
        assert!(output.contains("\"nominal_rank\""));
        assert!(output.contains("\"uncertainty_adjusted_rank\""));
        assert!(output.contains("\"uncertainty_rank_delta\""));
        assert!(output.contains("\"calibration\""));
        assert!(output.contains("\"placement\""));
        assert!(output.contains("\"gpu\""));
        assert!(output.contains("\"estimated_latency_ms\""));
        assert!(output.contains("\"estimated_latency_calibration_uncertainty_ms\""));
        assert!(output.contains("\"estimated_latency_calibration_lower_ms\""));
        assert!(output.contains("\"estimated_latency_calibration_upper_ms\""));
        assert!(output.contains("\"estimated_latency_uncertainty_adjusted_ms\""));
        assert!(output.contains("\"calibration_uncertainty\""));
        assert!(output.contains("\"approximation_policy\""));
        assert!(output.contains("\"search_budget\""));
        assert!(output.contains("\"max_parallelism_candidates\": 1"));
        assert!(output.contains("\"search_diagnostics\""));
        assert!(output.contains("\"search_mode\": \"parallelism\""));
        assert!(output.contains("\"candidate_space_count\": 2"));
        assert!(output.contains("\"searched_candidate_count\": 1"));
        assert!(output.contains("\"reported_candidate_count\": 1"));
        assert!(output.contains("\"truncated\": true"));
        assert!(output.contains("\"truncated_by_parallelism_budget\": true"));
        assert!(output.contains("\"default_action\": \"warn\""));
        assert!(output.contains("\"approximations\""));
        assert!(output.contains("\"approximation_policy_violations\""));
        assert!(output.contains("\"code\": \"capability_ordered_rank_placement\""));
        assert!(output.contains("\"code\": \"static_per_gpu_memory_estimate\""));
        assert!(output.contains("\"placement_evidence\""));
        assert!(output.contains("\"decision\": \"selected\""));
        assert!(output.contains("\"resource\": \"rank_placement\""));
        assert!(output.contains("\"code\": \"global_capability_ordered_placement\""));
        assert!(output.contains("\"resource\": \"eligible_gpus\""));
        assert!(output.contains("\"code\": \"hbm_capable_gpus_available\""));
        assert!(output.contains("\"resource_utilization\""));
        assert!(output.contains("\"resource_occupancy_bucket_count\": 2"));
        assert!(output.contains("\"resource_occupancy_resource_count\": 1"));
        assert!(output.contains("\"resource_occupancy\""));
        assert!(output.contains("\"critical_path_ms\""));
        assert!(output.contains("\"critical_path_step_count\""));
        assert!(output.contains("\"critical_path\""));
        assert!(output.contains("\"searched_configs\": 1"));
        assert!(output.contains("\"scheduled_operation_count\""));
        assert!(output.contains("\"scheduled_operations_truncated\""));
        assert!(output.contains("\"scheduled_operations\""));
        assert!(output.contains("\"start_ms\""));
        let rank_sensitivity = fs::read_to_string(&rank_sensitivity_path).unwrap();
        assert!(rank_sensitivity.starts_with("scenario,mode,candidate_rank"));
        assert!(rank_sensitivity.contains(",parallelism,1,tp1-pp1-ep1-dp1,true"));
        assert!(rank_sensitivity.contains("nominal_rank"));
        assert!(rank_sensitivity.contains("uncertainty_adjusted_rank"));
        assert!(rank_sensitivity.contains("calibration_fit_count_with_uncertainty"));
        assert!(rank_sensitivity.contains("calibration_min_confidence_score"));
        assert!(rank_sensitivity.contains("calibration_max_extrapolation_ratio"));
        assert!(rank_sensitivity.contains("calibration_applicability_status"));
        assert!(rank_sensitivity.contains("approximation_status"));
        assert!(rank_sensitivity.contains("approximation_aggregate_memory"));
        assert!(rank_sensitivity.contains("approximation_top_codes"));
        assert!(rank_sensitivity.contains("rejection_count"));
        assert!(rank_sensitivity.contains("top_rejection_code"));
        assert!(rank_sensitivity.contains("bottleneck_count"));
        assert!(rank_sensitivity.contains("top_bottleneck_code"));
        assert!(rank_sensitivity.contains("hardware_unique_gpu_count"));
        assert!(rank_sensitivity.contains("hardware_aggregate_gpu_types"));
        assert!(rank_sensitivity.contains("hardware_throughput_tokens_per_s_per_gpu"));
        assert!(rank_sensitivity.contains("static_per_gpu_memory_estimate"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
        let _ = fs::remove_file(rank_sensitivity_path);
    }

    #[test]
    fn emits_mixed_gpu_cluster_inventory_in_json_results() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!(
            "inference-sim-mixed-inventory-cluster-{nanos}.toml"
        ));
        let workload_path = dir.join(format!(
            "inference-sim-mixed-inventory-workload-{nanos}.toml"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from_group = "mixed_prefill"
            to_group = "decode"
            from_gpus = [0, 1]
            to_gpus = [0, 1]
            kind = "ib"
            variant = "hdr"
            rails = [0, 1]

            [[interconnect.links]]
            from_group = "mixed_prefill"
            to_group = "decode"
            kind = "ethernet"
            variant = "100g"
            rail = 2

            [[nodes]]
            id = 0
            group = "mixed_prefill"
            node_tags = ["prefill", "rack-local"]
            rack = "rack-a"
            island = "island-a"
            failure_domain = "az-a"
            disabled_gpus = [3]
            gpu_states = [{ gpu = 2, state = "maintenance" }]
            intra = "pcie_gen5"
            nics = { count = 4, affinity = "shared", gpus_per_nic = 2, bandwidth_gbps = 400.0, rail_count = 4, disabled_nics = [2], nic_states = [{ nic = 1, state = "reserved" }], nic_bandwidth_overrides = [{ nic = 0, bandwidth_gbps = 250.0 }], nic_latency_scale_overrides = [{ nic = 0, latency_scale = 1.75 }], nic_rail_map = [{ nic = 3, rail = 1 }], gpu_nic_map = [{ gpu = 1, nic = 3 }], gpu_nic_paths = [{ gpu = 1, nic = 3, label = "cross_socket", bandwidth_gbps = 100.0, latency_us = 12.0, gpudirect = false }] }
            gpus = [
              { start_id = 0, count = 2, gpu = "h200_sxm", labels = ["fast-nic"] },
              { start_id = 2, count = 2, gpu = "h100_sxm" },
            ]

            [[nodes]]
            id = 1
            group = "decode"
            node_tag = "decode"
            rack = "rack-b"
            island = "island-b"
            failure_domain = "az-b"
            gpu = "a100_80gb"
            gpu_count = 4
            gpu_profile_overrides = [{ gpu = 0, hbm_gb = 72.0, hbm_bandwidth_gb_s = 1800.0, peak_f16_tflops = 240.0 }]
            intra = "nvlink_v3"
            nics = { count = 4, affinity = "dedicated", bandwidth_gbps = 100.0, rail_count = 4, gpu_numa_map = [{ gpus = [0, 1], domain = 0 }, { gpus = [2, 3], domain = 1 }], nic_numa_map = [{ nics = [0, 1], domain = 0 }, { nics = [2, 3], domain = 1 }], cross_numa_bandwidth_scale = 0.5, cross_numa_latency_scale = 2.0 }
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"cluster_inventory\""));
        assert!(output.contains("\"total_gpus\": 8"));
        assert!(output.contains("\"available_gpus\": 6"));
        assert!(output.contains("\"disabled_gpus\": 2"));
        assert!(output.contains("\"H200 SXM\""));
        assert!(output.contains("\"H100 SXM5\""));
        assert!(output.contains("\"A100 80GB SXM\""));
        assert!(output.contains("\"profile_overridden\": true"));
        assert!(output.contains("\"hbm_gb\": 72.000000"));
        assert!(output.contains("\"hbm_bandwidth_gb_s\": 1800.000000"));
        assert!(output.contains("\"peak_f16_tflops\": 240.000000"));
        assert!(output.contains("\"node_groups\""));
        assert!(output.contains("\"mixed_prefill\""));
        assert!(output.contains("\"decode\""));
        assert!(output.contains("\"nics\""));
        assert!(output.contains("\"operational_state\": \"healthy\""));
        assert!(output.contains("\"operational_state\": \"maintenance\""));
        assert!(output.contains("\"operational_state\": \"reserved\""));
        assert!(output.contains("\"topology\""));
        assert!(output.contains("\"rack\": \"rack_a\""));
        assert!(output.contains("\"island\": \"island_a\""));
        assert!(output.contains("\"failure_domain\": \"az_a\""));
        assert!(output.contains("\"labels\": [\"prefill\", \"rack_local\"]"));
        assert!(output.contains("\"available\": false"));
        assert!(output.contains("\"available_gpu_count\": 2"));
        assert!(output.contains("\"disabled_gpus\": [2, 3]"));
        assert!(output.contains("\"active_count\": 2"));
        assert!(output.contains("\"disabled_nics\": [1, 2]"));
        assert!(output.contains("\"nic_states\""));
        assert!(output.contains("\"bandwidth_overrides\""));
        assert!(output.contains("\"bandwidth_gbps\": 250.000000"));
        assert!(output.contains("\"latency_scale_overrides\""));
        assert!(output.contains("\"latency_scale\": 1.750000"));
        assert!(output.contains("\"affinity\": \"shared\""));
        assert!(output.contains("\"gpus_per_nic\": 2"));
        assert!(output.contains("\"nic_rail_map\""));
        assert!(output.contains("\"nic_id\": 3"));
        assert!(output.contains("\"rail_id\": 1"));
        assert!(output.contains("\"gpu_nic_map\""));
        assert!(output.contains("\"local_gpu_id\": 1"));
        assert!(output.contains("\"nic_ids\": [3]"));
        assert!(output.contains("\"gpu_numa_map\""));
        assert!(output.contains("\"nic_numa_map\""));
        assert!(output.contains("\"numa_domain\": 1"));
        assert!(output.contains("\"cross_numa_bandwidth_scale\": 0.500000"));
        assert!(output.contains("\"cross_numa_latency_scale\": 2.000000"));
        assert!(output.contains("\"rail_ids\": [1]"));
        assert!(output.contains("\"labels\": [\"fast_nic\"]"));
        assert!(output.contains("\"gpu_nic_paths\""));
        assert!(output.contains("\"label\": \"cross_socket\""));
        assert!(output.contains("\"bandwidth_gbps\": 100.000000"));
        assert!(output.contains("\"latency_us\": 12.000000"));
        assert!(output.contains("\"gpudirect\": false"));
        assert!(output.contains("\"inter_node_topology\""));
        assert!(output.contains("\"link_count\": 3"));
        assert!(output.contains("\"rail\": 0"));
        assert!(output.contains("\"rail\": 1"));
        assert!(output.contains("\"rail\": 2"));
        assert!(output.contains("\"from_gpus\": [0, 1]"));
        assert!(output.contains("\"to_gpus\": [0, 1]"));
        assert!(output.contains("\"kind\": \"infiniband\""));
        assert!(output.contains("\"kind\": \"ethernet\""));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn solver_uses_explicit_rank_placement_from_toml() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!(
            "inference-sim-explicit-placement-cluster-{nanos}.toml"
        ));
        let workload_path = dir.join(format!(
            "inference-sim-explicit-placement-workload-{nanos}.toml"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 2

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [[placement.ranks]]
            rank = 0
            node = 1
            gpu = 3
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"placement_evidence\""));
        assert!(output.contains("\"code\": \"explicit_rank_placement\""));
        assert!(output.contains("\"node_id\": 1"));
        assert!(output.contains("\"local_gpu_id\": 3"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn emits_mixed_gpu_cluster_inventory_in_text_results() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!(
            "inference-sim-mixed-text-inventory-cluster-{nanos}.toml"
        ));
        let workload_path = dir.join(format!(
            "inference-sim-mixed-text-inventory-workload-{nanos}.toml"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from_group = "mixed_prefill"
            to_group = "decode"
            kind = "ib"
            variant = "hdr"
            rails = [0, 1]

            [[nodes]]
            id = 0
            group = "mixed_prefill"
            node_tags = ["prefill", "rack-local"]
            rack = "rack-a"
            island = "island-a"
            failure_domain = "az-a"
            disabled_gpus = [3]
            intra = "pcie_gen5"
            nics = { count = 4, affinity = "shared", gpus_per_nic = 2, bandwidth_gbps = 400.0, rail_count = 4, disabled_nics = [2], nic_rail_map = [{ nic = 3, rail = 1 }], gpu_nic_map = [{ gpu = 1, nic = 3 }], gpu_nic_paths = [{ gpu = 1, nic = 3, label = "cross_socket", bandwidth_gbps = 100.0, latency_us = 12.0, gpudirect = false }] }
            gpus = [
              { start_id = 0, count = 2, gpu = "h200_sxm", labels = ["fast-nic"] },
              { start_id = 2, count = 2, gpu = "h100_sxm" },
            ]

            [[nodes]]
            id = 1
            group = "decode"
            node_tag = "decode"
            rack = "rack-b"
            island = "island-b"
            failure_domain = "az-b"
            gpu = "a100_80gb"
            gpu_count = 4
            intra = "nvlink_v3"
            nics = { count = 4, affinity = "dedicated", bandwidth_gbps = 100.0, rail_count = 4 }
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("cluster_inventory"));
        assert!(output.contains("total_gpus=8"));
        assert!(output.contains("available_gpus=7"));
        assert!(output.contains("disabled_gpus=1"));
        assert!(output.contains("gpu_types=A100 80GB SXM:4,H100 SXM5:2,H200 SXM:2"));
        assert!(output.contains("node_groups=all:[0|1],decode:[1],mixed_prefill:[0]"));
        assert!(output.contains("node_inventory node=0"));
        assert!(output.contains("node_inventory node=0 state=healthy"));
        assert!(output.contains(
            "topology=rack=rack_a,island=island_a,failure_domain=az_a,labels=[prefill|rack_local]"
        ));
        assert!(output.contains("available_gpus=3"));
        assert!(output.contains("disabled_gpus=3"));
        assert!(output.contains("gpu_labels=0:[fast_nic],1:[fast_nic]"));
        assert!(output.contains("active_nics=3"));
        assert!(output.contains("disabled_nics=2"));
        assert!(output.contains("nic_rail_map=3:1"));
        assert!(output.contains("affinity=shared:2gpus_per_nic"));
        assert!(output.contains("gpu_nic_map=1:3"));
        assert!(output.contains("gpu_nic_paths=1:3:cross_socket:100.000Gbps:12.000us:false:true"));
        assert!(output.contains("interconnect=custom:links=2"));
        assert!(
            output.contains(
                "interconnect_link from=0 to=1 rail=0 from_gpus=- to_gpus=- fabric=IB HDR"
            )
        );
        assert!(
            output.contains(
                "interconnect_link from=0 to=1 rail=1 from_gpus=- to_gpus=- fabric=IB HDR"
            )
        );

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn emits_topology_diagnostics_for_disconnected_custom_cluster() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!(
            "inference-sim-topology-diagnostics-cluster-{nanos}.toml"
        ));
        let workload_path = dir.join(format!(
            "inference-sim-topology-diagnostics-workload-{nanos}.toml"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 3

            [[nodes]]
            id = 0
            group = "islanded"
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }

            [[nodes]]
            id = 1
            group = "bridge"
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }

            [[nodes]]
            id = 2
            group = "islanded"
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 2
            hidden_size = 1024
            attention_heads = 8
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 1.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 32
            decode_tokens = 4
            max_sequence_tokens = 64
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut text_output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut text_output,
        )
        .unwrap();
        let text_output = String::from_utf8(text_output).unwrap();
        assert!(text_output.contains("topology_diagnostic"));
        assert!(text_output.contains("code=custom_link_rail_unusable"));
        assert!(text_output.contains("code=disconnected_topology_islands"));
        assert!(text_output.contains("code=node_group_spans_disconnected_islands"));
        assert!(text_output.contains("group=islanded"));
        assert!(text_output.contains("components=[0];[1];[2]"));

        let mut json_output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut json_output,
        )
        .unwrap();
        let json_output = String::from_utf8(json_output).unwrap();
        assert!(json_output.contains("\"topology_diagnostics\""));
        assert!(json_output.contains("\"code\": \"custom_link_rail_unusable\""));
        assert!(json_output.contains("\"code\": \"disconnected_topology_islands\""));
        assert!(json_output.contains("\"code\": \"node_group_spans_disconnected_islands\""));
        assert!(json_output.contains("\"group\": \"islanded\""));
        assert!(json_output.contains("\"rail\": 3"));
        assert!(json_output.contains("\"components\""));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn emits_topology_diagnostics_for_gpu_nic_locality_risks() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!(
            "inference-sim-gpu-nic-diagnostics-cluster-{nanos}.toml"
        ));
        let workload_path = dir.join(format!(
            "inference-sim-gpu-nic-diagnostics-workload-{nanos}.toml"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 2, disabled_nics = [1], gpu_nic_paths = [{ gpu = 0, nic = 0, label = "host_staged", bandwidth_gbps = 100.0, latency_us = 12.0, gpudirect = false }, { gpu = 0, nic = 1, label = "disabled_nic_path", available = false }] }
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 2
            hidden_size = 1024
            attention_heads = 8
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 1.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 32
            decode_tokens = 4
            max_sequence_tokens = 64
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
        )
        .unwrap();

        let mut text_output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut text_output,
        )
        .unwrap();
        let text_output = String::from_utf8(text_output).unwrap();
        assert!(text_output.contains("code=gpu_nic_path_host_staged"));
        assert!(text_output.contains("code=gpu_nic_path_bandwidth_below_nic"));
        assert!(text_output.contains("code=gpu_nic_path_targets_disabled_nic"));
        assert!(text_output.contains("code=gpu_nic_path_unavailable"));
        assert!(text_output.contains("from=0"));
        assert!(text_output.contains("rail=0"));
        assert!(text_output.contains("rail=1"));
        assert!(text_output.contains("GPU 0->NIC 0"));
        assert!(text_output.contains("GPU 0->NIC 1"));

        let mut json_output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
            ],
            &mut json_output,
        )
        .unwrap();
        let json_output = String::from_utf8(json_output).unwrap();
        assert!(json_output.contains("\"code\": \"gpu_nic_path_host_staged\""));
        assert!(json_output.contains("\"code\": \"gpu_nic_path_bandwidth_below_nic\""));
        assert!(json_output.contains("\"code\": \"gpu_nic_path_targets_disabled_nic\""));
        assert!(json_output.contains("\"code\": \"gpu_nic_path_unavailable\""));
        assert!(json_output.contains("GPUDirect disabled"));
        assert!(json_output.contains("\"from_node\": 0"));
        assert!(json_output.contains("\"rail\": 0"));
        assert!(json_output.contains("\"rail\": 1"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn calibration_coverage_policy_can_reject_solver_results() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-gated-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-gated-workload-{nanos}.toml"));
        let profile_path = dir.join(format!("inference-sim-gated-profile-{nanos}.toml"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &profile_path,
            r#"
            [profile]
            name = "empty-profile"

            [valid_shape]
            min_batch_size = 1
            max_batch_size = 8
            min_prompt_tokens = 1
            max_prompt_tokens = 4096
            min_decode_tokens = 1
            max_decode_tokens = 128
            min_sequence_tokens = 1
            max_sequence_tokens = 8192
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            format!(
                r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 2
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [calibration_profile]
            path = "{}"

            [calibration_policy]
            coverage = "reject"
            min_coverage_score = 0.50

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
                profile_path.display()
            ),
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--format".to_string(),
                "json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"coverage\": \"reject\""));
        assert!(output.contains("\"code\": \"coverage_score_below_min\""));
        assert!(output.contains("\"action\": \"reject\""));
        assert!(output.contains("\"status\": \"reject\""));
        assert!(output.contains("\"feasible\": false"));
        assert!(output.contains("calibration coverage score 0.000 is below required minimum"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
        let _ = fs::remove_file(profile_path);
    }

    #[test]
    fn calibration_fit_policy_can_reject_extrapolated_solver_results() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-fit-policy-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-fit-policy-workload-{nanos}.toml"));
        let profile_path = dir.join(format!("inference-sim-fit-policy-profile-{nanos}.toml"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &profile_path,
            r#"
            [profile]
            name = "fit-policy-test"

            [[fits]]
            name = "prefill-fit"
            target = "prefill_ms"
            phase = "prefill"
            model = "linear"
            unit = "ms"
            intercept = 0.0
            features = ["effective_prefill_tokens"]
            coefficients = [0.001]
            feature_ranges = [
              { feature = "effective_prefill_tokens", min = 1, max = 64 }
            ]
            r_squared = 1.0
            rmse = 0.5
            rmse_pct = 10.0
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            format!(
                r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 1
            max_sequence_tokens = 256
            phase = "prefill"

            [calibration_profile]
            path = "{}"

            [calibration_policy]
            fit_extrapolation = "reject"
            min_fit_confidence_score = 0.0

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
                profile_path.display()
            ),
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--format".to_string(),
                "json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"fit_extrapolation\": \"reject\""));
        assert!(output.contains("\"calibration_gate_violations\""));
        assert!(output.contains("\"calibration_uncertainty\""));
        assert!(output.contains("\"relative_uncertainty_pct\""));
        assert!(output.contains("\"absolute_uncertainty_ms\""));
        assert!(output.contains("\"uncertainty_source\": \"rmse\""));
        assert!(output.contains("\"sample_count\""));
        assert!(output.contains("\"validation_sample_count\""));
        assert!(output.contains("\"source\""));
        assert!(output.contains("\"code\": \"fit_extrapolated\""));
        assert!(output.contains("\"action\": \"reject\""));
        assert!(output.contains("\"status\": \"reject\""));
        assert!(output.contains("\"feasible\": false"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
        let _ = fs::remove_file(profile_path);
    }

    #[test]
    fn runs_disaggregated_serving_solver_from_toml_files() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-serving-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-serving-workload-{nanos}.toml"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 2

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 2
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]
            max_e2el_ms = 100000.0
            min_kv_route_rail_count = 1

            [serving.pool_search]
            prefill_groups = ["all"]
            decode_groups = ["all"]
            prefill_node_counts = [1]
            decode_node_counts = [1]
            allow_overlap = false
            max_candidates = 1

            [serving.cost]
            default_gpu_hour_usd = 4.0
            node_hour_usd = 1.0
            kwh_usd = 0.12
            default_gpu_watts = 700.0
            node_watts = 1000.0

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 0
            gpu = 2

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            gpu = 2

            [[serving.traffic_classes]]
            name = "tenant-a-soft-penalty"
            group = "tenant"
            key = "tenant-a"
            max_prefill_tokens = 1024
            max_decode_sequences = 4
            max_resident_tokens = 4096
            max_kv_blocks = 512
            e2el_slo_ms = 0.001
            e2el_slo_miss_penalty_weight = 2.0

            [serving.traffic]
            request_count = 2

            [[serving.traffic.requests]]
            request_id = "json-0"
            tenant = "tenant-a"
            model_id = "model-a"
            arrival_ms = 0.0
            deadline_after_ms = 1000.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8

            [[serving.traffic.requests]]
            request_id = "json-1"
            tenant = "tenant-b"
            model_id = "model-a"
            arrival_ms = 1.0
            priority = 1
            deadline_after_ms = 1000.0
            batch_size = 1
            prompt_tokens = 256
            decode_tokens = 8
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("cluster_inventory"));
        assert!(output.contains("node_inventory node=0"));
        assert!(output.contains("node_inventory node=1"));
        assert!(output.contains("searched_serving_pairs=1"));
        assert!(output.contains("trust_boundary=v1_approximate"));
        assert!(output.contains("approximate_queueing"));
        assert!(output.contains("serving_stack_approximation"));
        assert!(output.contains("objective=minimize_e2el"));
        assert!(output.contains("ttft_ms"));
        assert!(output.contains("tpot_ms"));
        assert!(output.contains("pool"));
        assert!(output.contains("meas"));
        assert!(output.contains("sched_ms"));
        assert!(output.contains("seq_peak"));
        assert!(output.contains("kv_tok_peak"));
        assert!(output.contains("seq_node_peak"));
        assert!(output.contains("kv_node_peak"));
        assert!(output.contains("mode=fully_disaggregated"));
        assert!(output.contains("modes=full:1"));
        assert!(output.contains("calibration=uncalibrated"));
        assert!(output.contains("approximation_summary=calibration_risk"));
        assert!(output.contains("uncalibrated_runtime"));
        assert!(output.contains("footprint_gpus="));
        assert!(output.contains("throughput_per_gpu="));
        assert!(output.contains("pareto="));
        assert!(output.contains("objective_score="));
        assert!(output.contains("top_bottleneck="));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn emits_json_disaggregated_serving_results_from_toml_files() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-serving-json-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-serving-json-workload-{nanos}.toml"));
        let request_metrics_path = dir.join(format!(
            "inference-sim-serving-json-request-metrics-{nanos}.csv"
        ));
        let request_lifecycle_path = dir.join(format!(
            "inference-sim-serving-json-request-lifecycle-{nanos}.csv"
        ));
        let serving_metrics_path = dir.join(format!(
            "inference-sim-serving-json-candidate-metrics-{nanos}.csv"
        ));
        let metric_breakdowns_path = dir.join(format!(
            "inference-sim-serving-json-metric-breakdowns-{nanos}.csv"
        ));
        let services_path = dir.join(format!("inference-sim-serving-json-services-{nanos}.csv"));
        let utilization_path = dir.join(format!(
            "inference-sim-serving-json-utilization-{nanos}.csv"
        ));
        let memory_pressure_path = dir.join(format!(
            "inference-sim-serving-json-memory-pressure-{nanos}.csv"
        ));
        let timeline_path = dir.join(format!("inference-sim-serving-json-timeline-{nanos}.csv"));
        let occupancy_path = dir.join(format!("inference-sim-serving-json-occupancy-{nanos}.csv"));
        let placement_evidence_path = dir.join(format!(
            "inference-sim-serving-json-placement-evidence-{nanos}.csv"
        ));
        let worker_evidence_path = dir.join(format!(
            "inference-sim-serving-json-worker-evidence-{nanos}.csv"
        ));
        let rejections_path =
            dir.join(format!("inference-sim-serving-json-rejections-{nanos}.csv"));
        let route_paths_path = dir.join(format!(
            "inference-sim-serving-json-route-paths-{nanos}.csv"
        ));
        let kv_route_resources_path = dir.join(format!(
            "inference-sim-serving-json-kv-route-resources-{nanos}.csv"
        ));
        let bottlenecks_path = dir.join(format!(
            "inference-sim-serving-json-bottlenecks-{nanos}.csv"
        ));
        let phase_calibration_path = dir.join(format!(
            "inference-sim-serving-json-phase-calibration-{nanos}.csv"
        ));
        let approximations_path = dir.join(format!(
            "inference-sim-serving-json-approximations-{nanos}.csv"
        ));
        let rank_sensitivity_path = dir.join(format!(
            "inference-sim-serving-json-rank-sensitivity-{nanos}.csv"
        ));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 2

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 2
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1, 2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            objective = "minimize_cost"
            serving_stack = "vllm"
            runtime_features = ["paged_attention", "cuda_graphs"]
            prefill_nodes = [0]
            decode_nodes = [1]
            max_e2el_ms = 100000.0
            min_kv_route_rail_count = 1

            [serving.pool_search]
            prefill_groups = ["all"]
            decode_groups = ["all"]
            prefill_node_counts = [1]
            decode_node_counts = [1]
            allow_overlap = false
            max_candidates = 1

            [serving.cost]
            default_gpu_hour_usd = 4.0
            node_hour_usd = 1.0
            kwh_usd = 0.12
            default_gpu_watts = 700.0
            node_watts = 1000.0

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 0
            gpu = 0

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            gpu = 0

            [[serving.traffic_classes]]
            name = "tenant-a-soft-penalty"
            group = "tenant"
            key = "tenant-a"
            e2el_slo_ms = 0.001
            e2el_slo_miss_penalty_weight = 2.0

            [serving.traffic]
            request_count = 2

            [[serving.traffic.requests]]
            request_id = "json-0"
            tenant = "tenant-a"
            model_id = "model-a"
            arrival_ms = 0.0
            deadline_after_ms = 1000.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8

            [[serving.traffic.requests]]
            request_id = "json-1"
            tenant = "tenant-b"
            model_id = "model-a"
            arrival_ms = 1.0
            priority = 1
            deadline_after_ms = 1000.0
            batch_size = 1
            prompt_tokens = 256
            decode_tokens = 8
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--request-metrics-csv".to_string(),
                request_metrics_path.display().to_string(),
                "--request-lifecycle-events-csv".to_string(),
                request_lifecycle_path.display().to_string(),
                "--serving-metrics-csv".to_string(),
                serving_metrics_path.display().to_string(),
                "--serving-metric-breakdowns-csv".to_string(),
                metric_breakdowns_path.display().to_string(),
                "--serving-services-csv".to_string(),
                services_path.display().to_string(),
                "--serving-utilization-csv".to_string(),
                utilization_path.display().to_string(),
                "--serving-memory-pressure-csv".to_string(),
                memory_pressure_path.display().to_string(),
                "--serving-timeline-csv".to_string(),
                timeline_path.display().to_string(),
                "--serving-occupancy-csv".to_string(),
                occupancy_path.display().to_string(),
                "--serving-placement-evidence-csv".to_string(),
                placement_evidence_path.display().to_string(),
                "--serving-worker-evidence-csv".to_string(),
                worker_evidence_path.display().to_string(),
                "--serving-rejections-csv".to_string(),
                rejections_path.display().to_string(),
                "--serving-route-paths-csv".to_string(),
                route_paths_path.display().to_string(),
                "--kv-route-resources-csv".to_string(),
                kv_route_resources_path.display().to_string(),
                "--serving-bottlenecks-csv".to_string(),
                bottlenecks_path.display().to_string(),
                "--serving-phase-calibration-csv".to_string(),
                phase_calibration_path.display().to_string(),
                "--serving-approximations-csv".to_string(),
                approximations_path.display().to_string(),
                "--rank-sensitivity-csv".to_string(),
                rank_sensitivity_path.display().to_string(),
                "--trace".to_string(),
                "--occupancy-buckets".to_string(),
                "2".to_string(),
                "--occupancy-resource-limit".to_string(),
                "1".to_string(),
                "--critical-path-limit".to_string(),
                "2".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
                "--max-serving-pairs".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        let request_metrics = fs::read_to_string(&request_metrics_path).unwrap();
        let request_lifecycle = fs::read_to_string(&request_lifecycle_path).unwrap();
        let serving_metrics = fs::read_to_string(&serving_metrics_path).unwrap();
        let metric_breakdowns = fs::read_to_string(&metric_breakdowns_path).unwrap();
        let services = fs::read_to_string(&services_path).unwrap();
        let utilization = fs::read_to_string(&utilization_path).unwrap();
        let memory_pressure = fs::read_to_string(&memory_pressure_path).unwrap();
        let timeline = fs::read_to_string(&timeline_path).unwrap();
        let occupancy = fs::read_to_string(&occupancy_path).unwrap();
        let placement_evidence = fs::read_to_string(&placement_evidence_path).unwrap();
        let worker_evidence = fs::read_to_string(&worker_evidence_path).unwrap();
        let rejections = fs::read_to_string(&rejections_path).unwrap();
        let route_paths = fs::read_to_string(&route_paths_path).unwrap();
        let kv_route_resources = fs::read_to_string(&kv_route_resources_path).unwrap();
        let bottlenecks = fs::read_to_string(&bottlenecks_path).unwrap();
        let phase_calibration = fs::read_to_string(&phase_calibration_path).unwrap();
        let approximations = fs::read_to_string(&approximations_path).unwrap();
        let rank_sensitivity = fs::read_to_string(&rank_sensitivity_path).unwrap();
        assert!(request_metrics.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(request_metrics.contains("traffic_class,shape_profile,status"));
        assert!(request_metrics.contains("included_in_measurement_window"));
        assert!(request_metrics.contains("measurement_window_source"));
        assert!(request_metrics.contains("measurement_start_s"));
        assert!(request_metrics.contains("measurement_end_s"));
        assert!(request_metrics.contains("metric_unavailable_reason"));
        assert!(request_metrics.contains("json-0"));
        assert!(request_metrics.contains("request_lifecycle_events"));
        assert_csv_field(
            &request_metrics,
            "metric_source",
            "request_lifecycle_events",
        );
        assert_csv_field(&request_metrics, "event_sourced", "true");
        assert_csv_field(&request_metrics, "included_in_measurement_window", "true");
        assert_csv_field(&request_metrics, "measurement_window_source", "default");
        assert_csv_field(&request_metrics, "terminal_event", "completed");
        assert_csv_field(&request_metrics, "metric_unavailable_reason", "");
        assert_csv_field(
            &request_metrics,
            "ttft_end_event",
            "decode_iteration_finished:first",
        );
        assert_csv_field(
            &request_metrics,
            "tpot_start_event",
            "decode_iteration_finished:first",
        );
        assert_csv_field(
            &request_metrics,
            "tpot_end_event",
            "decode_iteration_finished:last",
        );
        assert_csv_field(
            &request_metrics,
            "e2el_end_event",
            "decode_iteration_finished:last",
        );
        assert_csv_field(
            &request_metrics,
            "throughput_duration_end_event",
            "decode_iteration_finished:last",
        );
        assert_csv_field(&request_metrics, "decode_finish_event_count", "8");
        assert_csv_field(&request_metrics, "tpot_sample_count", "7");
        assert!(request_metrics.contains("fully_disaggregated"));
        assert!(request_lifecycle.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(request_lifecycle.contains("json-0"));
        assert!(request_lifecycle.contains("request_lifecycle_events"));
        assert!(request_lifecycle.contains(",arrived,arrival,"));
        assert!(request_lifecycle.contains(",decode_iteration_finished,decode,"));
        assert!(request_lifecycle.contains(",completed,terminal,"));
        assert!(serving_metrics.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(serving_metrics.contains("fully_disaggregated"));
        assert!(serving_metrics.contains(",minimize_cost,"));
        assert!(serving_metrics.contains(",feasible,"));
        assert!(serving_metrics.contains("ttft_calibration_uncertainty_s"));
        assert!(serving_metrics.contains("tpot_calibration_lower_s"));
        assert!(serving_metrics.contains("itl_calibration_upper_s"));
        assert!(serving_metrics.contains("e2el_calibration_uncertainty_s"));
        assert!(serving_metrics.contains("throughput_calibration_lower_tokens_per_s"));
        assert!(serving_metrics.contains("aggregate_gpu_types"));
        assert!(serving_metrics.contains("prefill_gpu_types"));
        assert!(serving_metrics.contains("decode_gpu_types"));
        assert!(serving_metrics.contains("aggregate_hbm_gb"));
        assert!(serving_metrics.contains("throughput_tokens_per_s_per_gpu"));
        assert!(serving_metrics.contains("measurement_window_source"));
        assert!(serving_metrics.contains("measurement_duration_s"));
        assert!(serving_metrics.contains("lifecycle_event_metric_request_count"));
        assert!(serving_metrics.contains("fallback_metric_request_count"));
        assert!(serving_metrics.contains("metric_source_counts"));
        assert!(serving_metrics.contains("ttft_slo_constrained_requests"));
        assert!(serving_metrics.contains("e2el_slo_missed_requests"));
        assert!(serving_metrics.contains("deadline_constrained_requests"));
        assert!(serving_metrics.contains("calibration_active_phase_count"));
        assert!(serving_metrics.contains("calibration_uncalibrated_phase_count"));
        assert!(serving_metrics.contains("calibration_extrapolated_fit_count"));
        assert!(serving_metrics.contains("calibration_fit_count_with_uncertainty"));
        assert!(serving_metrics.contains("calibration_gate_violation_count"));
        assert!(serving_metrics.contains("approximation_policy_violation_count"));
        assert!(serving_metrics.contains("approximation_uncalibrated_runtime"));
        assert!(serving_metrics.contains("approximation_category_counts"));
        assert!(serving_metrics.contains("approximation_top_codes"));
        assert!(serving_metrics.contains("bottleneck_count"));
        assert!(serving_metrics.contains("top_bottleneck_code"));
        assert!(serving_metrics.contains("rejection_count"));
        assert!(serving_metrics.contains("top_rejection_code"));
        assert!(serving_metrics.contains("request_lifecycle_events:2"));
        assert_csv_field(&serving_metrics, "aggregate_gpu_types", "H100 SXM5:2");
        assert_csv_field(&serving_metrics, "prefill_gpu_types", "H100 SXM5:1");
        assert_csv_field(&serving_metrics, "decode_gpu_types", "H100 SXM5:1");
        assert_csv_field(&serving_metrics, "aggregate_hbm_gb", "160.000000000");
        assert_csv_field(&serving_metrics, "prefill_hbm_gb", "80.000000000");
        assert_csv_field(&serving_metrics, "decode_hbm_gb", "80.000000000");
        assert_csv_field_nonempty(&serving_metrics, "aggregate_hbm_bandwidth_gb_s");
        assert_csv_field_nonempty(&serving_metrics, "aggregate_effective_peak_tflops");
        assert_csv_field_nonempty(&serving_metrics, "throughput_tokens_per_s_per_gpu");
        assert_csv_field(&serving_metrics, "measurement_window_request_count", "2");
        assert_csv_field(
            &serving_metrics,
            "measurement_window_completed_request_count",
            "2",
        );
        assert_csv_field(
            &serving_metrics,
            "measurement_window_failed_request_count",
            "0",
        );
        assert_csv_field(
            &serving_metrics,
            "measurement_window_rejected_request_count",
            "0",
        );
        assert_csv_field(
            &serving_metrics,
            "measurement_window_deadline_constrained_request_count",
            "2",
        );
        assert_csv_field(
            &serving_metrics,
            "measurement_window_deadline_missed_request_count",
            "0",
        );
        assert_csv_field(&serving_metrics, "ttft_slo_constrained_requests", "0");
        assert_csv_field(&serving_metrics, "ttft_slo_missed_requests", "0");
        assert_csv_field(&serving_metrics, "e2el_slo_constrained_requests", "1");
        assert_csv_field(&serving_metrics, "e2el_slo_missed_requests", "1");
        assert_csv_field(&serving_metrics, "deadline_constrained_requests", "2");
        assert_csv_field(&serving_metrics, "deadline_missed_requests", "0");
        assert_csv_field(&serving_metrics, "calibration_fit_count", "0");
        assert_csv_field(&serving_metrics, "calibration_extrapolated_fit_count", "0");
        assert_csv_field(
            &serving_metrics,
            "calibration_fit_count_with_uncertainty",
            "0",
        );
        assert_csv_field(&serving_metrics, "calibration_gate_violation_count", "0");
        assert_csv_field(
            &serving_metrics,
            "calibration_hard_gate_violation_count",
            "0",
        );
        assert_csv_field(
            &serving_metrics,
            "approximation_policy_violation_count",
            "0",
        );
        assert_csv_field(
            &serving_metrics,
            "approximation_uncalibrated_runtime",
            "true",
        );
        assert_csv_field_nonempty(&serving_metrics, "approximation_category_counts");
        assert_csv_field_nonempty(&serving_metrics, "approximation_top_codes");
        assert_csv_field_nonempty(&serving_metrics, "bottleneck_count");
        assert_csv_field_nonempty(&serving_metrics, "top_bottleneck_code");
        assert_csv_field(&serving_metrics, "rejection_count", "0");
        assert!(metric_breakdowns.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(metric_breakdowns.contains(",tenant,tenant-a,"));
        assert!(metric_breakdowns.contains(",model_id,model-a,"));
        assert!(metric_breakdowns.contains(",traffic_class,tenant-a-soft-penalty,"));
        assert!(metric_breakdowns.contains("throughput_tokens_per_s"));
        assert!(metric_breakdowns.contains("e2el_slo_miss_rate"));
        assert!(metric_breakdowns.contains("rejected_requests"));
        assert!(metric_breakdowns.contains("lifecycle_event_metric_request_count"));
        assert!(metric_breakdowns.contains("metric_source_counts"));
        assert!(metric_breakdowns.contains("e2el_slo_constrained_requests"));
        assert_csv_row_field(
            &metric_breakdowns,
            "key",
            "tenant-a-soft-penalty",
            "request_count",
            "1",
        );
        assert_csv_row_field(
            &metric_breakdowns,
            "key",
            "tenant-a-soft-penalty",
            "completed_requests",
            "1",
        );
        assert_csv_row_field(
            &metric_breakdowns,
            "key",
            "tenant-a-soft-penalty",
            "rejected_requests",
            "0",
        );
        assert_csv_row_field(
            &metric_breakdowns,
            "key",
            "tenant-a-soft-penalty",
            "lifecycle_event_metric_request_count",
            "1",
        );
        assert_csv_row_field(
            &metric_breakdowns,
            "key",
            "tenant-a-soft-penalty",
            "metric_source_counts",
            "request_lifecycle_events:1",
        );
        assert_csv_row_field(
            &metric_breakdowns,
            "key",
            "tenant-a-soft-penalty",
            "e2el_slo_constrained_requests",
            "1",
        );
        assert_csv_row_field(
            &metric_breakdowns,
            "key",
            "tenant-a-soft-penalty",
            "e2el_slo_missed_requests",
            "1",
        );
        assert!(services.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(services.contains("serving:mode-fully-disaggregated"));
        assert!(services.contains("fully_disaggregated"));
        assert!(services.contains(",prefill,healthy,true,"));
        assert!(services.contains(",decode,healthy,true,"));
        assert!(services.contains(",kv_transfer,healthy,true,"));
        assert!(services.contains("backpressure_state"));
        assert!(services.contains("worker_slot_utilization"));
        assert!(utilization.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(utilization.contains("serving:mode-fully-disaggregated"));
        assert!(utilization.contains("fully_disaggregated"));
        assert!(utilization.contains("phase_resource"));
        assert!(utilization.contains("scheduled_resource"));
        assert!(memory_pressure.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(memory_pressure.contains("serving:mode-fully-disaggregated"));
        assert!(memory_pressure.contains("fully_disaggregated"));
        assert!(memory_pressure.contains("capacity_used_fraction"));
        assert!(memory_pressure.contains("kv_cache_gb"));
        assert!(timeline.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(timeline.contains("serving:mode-fully-disaggregated"));
        assert!(timeline.contains("fully_disaggregated"));
        assert!(timeline.contains("request 0 prefill"));
        assert!(timeline.contains("kv_transfer"));
        assert!(occupancy.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(occupancy.contains("serving:mode-fully-disaggregated"));
        assert!(occupancy.contains("fully_disaggregated"));
        assert!(occupancy.contains("utilization"));
        assert!(placement_evidence.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(placement_evidence.contains("serving:mode-fully-disaggregated"));
        assert!(placement_evidence.contains("fully_disaggregated"));
        assert!(placement_evidence.contains("prefill"));
        assert!(placement_evidence.contains("decode"));
        assert!(placement_evidence.contains("explicit_rank_placement"));
        assert!(worker_evidence.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(worker_evidence.contains("serving:mode-fully-disaggregated"));
        assert!(worker_evidence.contains("fully_disaggregated"));
        assert!(worker_evidence.contains("json-0"));
        assert!(worker_evidence.contains("worker_summary"));
        assert!(worker_evidence.contains("worker_assignment"));
        assert!(worker_evidence.contains("kv_block_ownership"));
        assert!(worker_evidence.contains("kv_worker_slot_ownership"));
        assert!(worker_evidence.contains("kv_cache_owner"));
        assert!(rejections.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(route_paths.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(route_paths.contains("serving:mode-fully-disaggregated"));
        assert!(route_paths.contains("fully_disaggregated"));
        assert!(route_paths.contains("json-0"));
        assert!(route_paths.contains("inter_node_fabric"));
        assert!(route_paths.contains("kv_route:"));
        assert!(kv_route_resources.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(kv_route_resources.contains("serving:mode-fully-disaggregated"));
        assert!(kv_route_resources.contains("fully_disaggregated"));
        assert!(bottlenecks.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(bottlenecks.contains("serving:mode-fully-disaggregated"));
        assert!(bottlenecks.contains("fully_disaggregated"));
        assert!(bottlenecks.contains("request_idx,request_id,tenant,model_id,traffic_class"));
        assert!(bottlenecks.contains("request_e2el_slo_miss"));
        assert!(bottlenecks.contains(",objective,all,objective,"));
        assert!(bottlenecks.contains("objective_slo_miss_penalty"));
        assert!(bottlenecks.contains("json-0"));
        assert!(bottlenecks.contains("tenant-a"));
        assert!(bottlenecks.contains("model-a"));
        assert!(phase_calibration.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(phase_calibration.contains("phase_fit_count_with_uncertainty"));
        assert!(phase_calibration.contains("candidate_calibration_status"));
        assert!(phase_calibration.contains("prefill"));
        assert!(phase_calibration.contains("decode"));
        assert!(approximations.starts_with("scenario,candidate_rank,candidate_id"));
        assert!(approximations.contains("serving:mode-fully-disaggregated"));
        assert!(approximations.contains("fully_disaggregated"));
        assert!(approximations.contains("approximation"));
        assert!(approximations.contains("approximate_serving_event_loop"));
        assert!(approximations.contains("serving_stack_uncalibrated"));
        assert!(approximations.contains("node_set_kv_handoff"));
        assert!(rank_sensitivity.starts_with("scenario,mode,candidate_rank"));
        assert!(rank_sensitivity.contains(",serving,1,serving:mode-fully-disaggregated"));
        assert!(rank_sensitivity.contains("fully_disaggregated"));
        assert!(rank_sensitivity.contains("minimize_cost"));
        assert!(rank_sensitivity.contains("uncertainty_adjusted_rank"));
        assert!(rank_sensitivity.contains("calibration_fit_count_with_uncertainty"));
        assert!(rank_sensitivity.contains("calibration_min_confidence_score"));
        assert!(rank_sensitivity.contains("calibration_max_extrapolation_ratio"));
        assert!(rank_sensitivity.contains("calibration_applicability_status"));
        assert!(rank_sensitivity.contains("approximation_status"));
        assert!(rank_sensitivity.contains("approximation_coarse_topology"));
        assert!(rank_sensitivity.contains("approximation_approximate_queueing"));
        assert!(rank_sensitivity.contains("approximation_uncalibrated_runtime"));
        assert!(rank_sensitivity.contains("rejection_count"));
        assert!(rank_sensitivity.contains("top_rejection_code"));
        assert!(rank_sensitivity.contains("bottleneck_count"));
        assert!(rank_sensitivity.contains("top_bottleneck_code"));
        assert!(rank_sensitivity.contains("hardware_unique_gpu_count"));
        assert!(rank_sensitivity.contains("hardware_aggregate_gpu_types"));
        assert!(rank_sensitivity.contains("hardware_throughput_tokens_per_s_per_gpu"));
        assert!(rank_sensitivity.contains("H100 SXM5"));
        assert!(rank_sensitivity.contains("approximate_serving_event_loop"));
        assert!(output.starts_with("{\n"));
        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"cluster_inventory\""));
        assert!(output.contains("\"gpu_types\""));
        assert!(output.contains("\"H100 SXM5\""));
        assert!(output.contains("\"inter_node_topology\""));
        assert!(output.contains("\"trust_boundary\""));
        assert!(output.contains("\"status\": \"v1_approximate\""));
        assert!(output.contains("\"code\": \"approximate_queueing\""));
        assert!(output.contains("\"code\": \"ignored_network_congestion\""));
        assert!(output.contains("\"code\": \"serving_stack_approximation\""));
        assert!(output.contains("\"candidate_id\""));
        assert!(output.contains("serving:mode-fully-disaggregated:pool-"));
        assert!(output.contains("\"deployment_mode\": \"fully_disaggregated\""));
        assert!(output.contains("\"nominal_rank\""));
        assert!(output.contains("\"uncertainty_adjusted_rank\""));
        assert!(output.contains("\"uncertainty_rank_delta\""));
        assert!(output.contains("\"objective\": \"minimize_cost\""));
        assert!(output.contains("\"serving_stack\": \"vllm\""));
        assert!(
            output.contains("\"serving_runtime_features\": [\"paged_attention\", \"cuda_graphs\"]")
        );
        assert!(output.contains("\"objective_base_score\""));
        assert!(output.contains("\"objective_breakdown\""));
        assert!(output.contains("\"selected_objective\": \"minimize_cost\""));
        assert!(output.contains("\"score_convention\": \"lower_is_better\""));
        assert!(output.contains("\"base_metric\": \"total_cost_usd\""));
        assert!(output.contains("\"base_metric_direction\": \"minimize\""));
        assert!(output.contains("\"base_metric_unit\": \"usd\""));
        assert!(output.contains("\"nominal_penalty_score\""));
        assert!(output.contains("\"uncertainty_adjusted_delta\""));
        assert!(output.contains("\"largest_nominal_term\""));
        assert!(output.contains("\"rejection_summary\""));
        assert!(output.contains("\"total_rejection_count\""));
        assert!(output.contains("\"pareto_frontier\""));
        assert!(output.contains("\"pareto_rank\""));
        assert!(output.contains("\"pareto_dominated_by\""));
        assert!(output.contains("\"pareto_dimensions\""));
        assert!(output.contains("\"metric\": \"itl_s\""));
        assert!(output.contains("\"metric\": \"total_cost_usd\""));
        assert!(output.contains("\"metric\": \"energy_kwh\""));
        assert!(output.contains("\"metric\": \"average_power_watts\""));
        assert!(output.contains("\"direction\": \"minimize\""));
        assert!(output.contains("\"memory_pressure_peak_fraction\""));
        assert!(output.contains("\"memory_pressure_peak_phase\""));
        assert!(output.contains("\"max_memory_pressure_fraction\""));
        assert!(output.contains("\"max_unique_gpus\""));
        assert!(output.contains("\"min_throughput_tokens_per_s\""));
        assert!(output.contains("\"metric_ceilings\""));
        assert!(output.contains("\"max_e2el_s\": 100"));
        assert!(output.contains("\"kv_route_constraints\""));
        assert!(output.contains("\"min_inter_node_rail_count\": 1"));
        assert!(output.contains("\"cost_estimate\""));
        assert!(output.contains("\"total_cost_usd\""));
        assert!(output.contains("\"cost_per_1k_output_tokens_usd\""));
        assert!(output.contains("\"slo_miss_penalty_weight\""));
        assert!(output.contains("\"slo_miss_penalty_weights\""));
        assert!(output.contains("\"slo_miss_penalty_components\""));
        assert!(output.contains("\"traffic_class_slo_miss_penalties\""));
        assert!(output.contains("\"traffic_class_capacity\""));
        assert!(output.contains("\"serving_services\""));
        assert!(output.contains("\"health\""));
        assert!(output.contains("\"effective_worker_slots_per_gpu\""));
        assert!(output.contains("\"tenant-a-soft-penalty\""));
        assert!(output.contains("\"max_prefill_tokens\""));
        assert!(output.contains("\"max_decode_sequences\""));
        assert!(output.contains("\"slo_miss_penalty_score\""));
        assert!(output.contains("\"service_backpressure_penalty_weight\""));
        assert!(output.contains("\"service_backpressure_penalty_score\""));
        assert!(output.contains("\"topology_risk_penalty_weight\""));
        assert!(output.contains("\"topology_risk_penalty_score\""));
        assert!(output.contains("\"objective_nominal_score\""));
        assert!(output.contains("\"objective_uncertainty_adjusted_score\""));
        assert!(output.contains("\"route_coverage\""));
        assert!(output.contains("\"candidate_count\""));
        assert!(output.contains("\"routable_candidate_count\""));
        assert!(output.contains("\"unroutable_candidate_count\""));
        assert!(output.contains("\"pool_search_summary\""));
        assert!(output.contains("\"generated_candidate_count\": 1"));
        assert!(output.contains("\"generated_colocated_count\": 0"));
        assert!(output.contains("\"generated_partially_disaggregated_count\": 0"));
        assert!(output.contains("\"generated_fully_disaggregated_count\": 1"));
        assert!(output.contains("\"considered_candidate_count\""));
        assert!(output.contains("\"rejected_overlap_count\""));
        assert!(output.contains("\"prefill_node_filter_node_count\""));
        assert!(output.contains("\"decode_gpu_filter_node_count\""));
        assert!(output.contains("\"pool_topology\""));
        assert!(output.contains("\"prefill_node_count\": 1"));
        assert!(output.contains("\"decode_node_count\": 1"));
        assert!(output.contains("\"shared_node_count\": 0"));
        assert!(output.contains("\"dedicated_prefill_node_count\": 1"));
        assert!(output.contains("\"dedicated_decode_node_count\": 1"));
        assert!(output.contains("\"prefill_racks\""));
        assert!(output.contains("\"decode_node_labels\""));
        assert!(output.contains("\"searched_serving_pairs\": 1"));
        assert!(output.contains("\"search_budget\""));
        assert!(output.contains("\"max_parallelism_candidates\": 1"));
        assert!(output.contains("\"max_prefill_candidates\": 1"));
        assert!(output.contains("\"max_decode_candidates\": 1"));
        assert!(output.contains("\"max_serving_pairs\": 1"));
        assert!(output.contains("\"search_diagnostics\""));
        assert!(output.contains("\"search_mode\": \"serving\""));
        assert!(output.contains("\"prefill_candidate_space_count\": 2"));
        assert!(output.contains("\"decode_candidate_space_count\": 2"));
        assert!(output.contains("\"serving_pair_space_lower_bound_per_pool\": 1"));
        assert!(output.contains("\"truncated_by_prefill_budget\": true"));
        assert!(output.contains("\"truncated_by_decode_budget\": true"));
        assert!(output.contains("\"serving_pair_budget_exhausted\": true"));
        assert!(output.contains("\"hardware_footprint\""));
        assert!(output.contains("\"unique_gpu_count\""));
        assert!(output.contains("\"aggregate_gpu_types\""));
        assert!(output.contains("\"prefill_gpu_types\""));
        assert!(output.contains("\"decode_gpu_types\""));
        assert!(output.contains("\"aggregate_gpu_label_counts\""));
        assert!(output.contains("\"prefill_gpu_label_counts\""));
        assert!(output.contains("\"decode_gpu_label_counts\""));
        assert!(output.contains("\"aggregate_hbm_gb\""));
        assert!(output.contains("\"aggregate_effective_peak_tflops\""));
        assert!(output.contains("\"throughput_tokens_per_s_per_gpu\""));
        assert!(output.contains("\"bottleneck_summary\""));
        assert!(output.contains("\"source\": \"objective\""));
        assert!(output.contains("\"code\": \"objective_slo_miss_penalty\""));
        assert!(output.contains("\"source\": \"memory_pressure\""));
        assert!(output.contains("\"code\": \"peak_memory_pressure\""));
        assert!(output.contains("\"source\": \"phase_utilization\""));
        assert!(output.contains("\"prefill_placement\""));
        assert!(output.contains("\"prefill_placement_evidence\""));
        assert!(output.contains("\"decode_placement\""));
        assert!(output.contains("\"decode_placement_evidence\""));
        assert!(output.contains("\"code\": \"explicit_rank_placement\""));
        assert!(output.contains("\"code\": \"hbm_capable_gpus_available\""));
        assert!(output.contains("\"gpu\""));
        assert!(output.contains("\"prefill_memory\""));
        assert!(output.contains("\"decode_memory\""));
        assert!(output.contains("\"limiting_gpu\""));
        assert!(output.contains("\"capacity_used_fraction\""));
        assert!(output.contains("\"headroom_gb\""));
        assert!(output.contains("\"headroom_fraction\""));
        assert!(output.contains("\"dominant_component\""));
        assert!(output.contains("\"component_fractions\""));
        assert!(output.contains("\"components\""));
        assert!(output.contains("\"weights_gb\""));
        assert!(output.contains("\"kv_cache_gb\""));
        assert!(output.contains("\"block_table_gb\""));
        assert!(output.contains("\"activations_gb\""));
        assert!(output.contains("\"temporary_gb\""));
        assert!(output.contains("\"communication_gb\""));
        assert!(output.contains("\"runtime_reserve_gb\""));
        assert!(output.contains("\"fragmentation_gb\""));
        assert!(output.contains("\"serving_memory_temporary_fraction\""));
        assert!(output.contains("\"serving_memory_activation_communication_fraction\""));
        assert!(output.contains("\"serving_memory_weight_communication_fraction\""));
        assert!(output.contains("\"serving_memory_runtime_reserve_fraction\""));
        assert!(output.contains("\"serving_memory_fragmentation_fraction\""));
        assert!(output.contains("\"total_gb\""));
        assert!(output.contains("\"memory_pressure_observation_count\""));
        assert!(output.contains("\"memory_pressure_observations_truncated\""));
        assert!(output.contains("\"serving_calibration_summary\""));
        assert!(output.contains("\"active_phase_count\""));
        assert!(output.contains("\"calibrated_phase_count\""));
        assert!(output.contains("\"coverage_fraction\""));
        assert!(output.contains("\"hard_gate_violation_count\""));
        assert!(output.contains("\"memory_pressure\""));
        assert!(output.contains("\"estimate_kind\""));
        assert!(output.contains("\"active_requests\""));
        assert!(output.contains("\"active_tokens\""));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"ttft_calibration_uncertainty_ms\""));
        assert!(output.contains("\"ttft_calibration_lower_ms\""));
        assert!(output.contains("\"ttft_calibration_upper_ms\""));
        assert!(output.contains("\"ttft_p90_ms\""));
        assert!(output.contains("\"ttft_max_ms\""));
        assert!(output.contains("\"itl_ms\""));
        assert!(output.contains("\"itl_calibration_uncertainty_ms\""));
        assert!(output.contains("\"itl_calibration_lower_ms\""));
        assert!(output.contains("\"itl_calibration_upper_ms\""));
        assert!(output.contains("\"tpot_calibration_uncertainty_ms\""));
        assert!(output.contains("\"tpot_calibration_lower_ms\""));
        assert!(output.contains("\"tpot_calibration_upper_ms\""));
        assert!(output.contains("\"tpot_p90_ms\""));
        assert!(output.contains("\"tpot_max_ms\""));
        assert!(output.contains("\"itl_p90_ms\""));
        assert!(output.contains("\"itl_p95_ms\""));
        assert!(output.contains("\"itl_max_ms\""));
        assert!(output.contains("\"decode_iterations\""));
        assert!(output.contains("\"decode_iteration_ms\""));
        assert!(output.contains("\"decode_iteration_calibration_uncertainty_ms\""));
        assert!(output.contains("\"decode_iteration_calibration_lower_ms\""));
        assert!(output.contains("\"decode_iteration_calibration_upper_ms\""));
        assert!(output.contains("\"decode_iteration_p90_ms\""));
        assert!(output.contains("\"decode_iteration_p95_ms\""));
        assert!(output.contains("\"decode_iteration_max_ms\""));
        assert!(output.contains("\"throughput_calibration_uncertainty_tokens_per_s\""));
        assert!(output.contains("\"throughput_calibration_lower_tokens_per_s\""));
        assert!(output.contains("\"throughput_calibration_upper_tokens_per_s\""));
        assert!(output.contains("\"e2el_calibration_uncertainty_ms\""));
        assert!(output.contains("\"e2el_calibration_lower_ms\""));
        assert!(output.contains("\"e2el_calibration_upper_ms\""));
        assert!(output.contains("\"e2el_p90_ms\""));
        assert!(output.contains("\"e2el_max_ms\""));
        assert!(output.contains("\"ttft_slo_miss_rate\""));
        assert!(output.contains("\"tpot_slo_miss_rate\""));
        assert!(output.contains("\"itl_slo_miss_rate\""));
        assert!(output.contains("\"e2el_slo_miss_rate\""));
        assert!(output.contains("\"deadline_miss_rate\""));
        assert!(output.contains("\"service_ms\""));
        assert!(output.contains("\"prefill_ms\""));
        assert!(output.contains("\"prefill_calibration_uncertainty_ms\""));
        assert!(output.contains("\"prefill_calibration_lower_ms\""));
        assert!(output.contains("\"prefill_calibration_upper_ms\""));
        assert!(output.contains("\"kv_transfer_calibration_lower_ms\""));
        assert!(output.contains("\"kv_transfer_calibration_upper_ms\""));
        assert!(output.contains("\"decode_ms\""));
        assert!(output.contains("\"decode_calibration_uncertainty_ms\""));
        assert!(output.contains("\"decode_calibration_lower_ms\""));
        assert!(output.contains("\"decode_calibration_upper_ms\""));
        assert!(output.contains("\"scheduled_makespan_calibration_lower_ms\""));
        assert!(output.contains("\"scheduled_makespan_calibration_upper_ms\""));
        assert!(output.contains("\"calibration_fit_applications\""));
        assert!(output.contains("\"serving_phase_calibration\""));
        assert!(output.contains("\"calibrated\""));
        assert!(output.contains("\"applied_targets\""));
        assert!(output.contains("\"uncalibrated_no_profile\""));
        assert!(output.contains("\"calibration_uncertainty\""));
        assert!(output.contains("\"approximation_policy\""));
        assert!(output.contains("\"approximation_summary\""));
        assert!(output.contains("\"status\": \"calibration_risk\""));
        assert!(output.contains("\"category_counts\""));
        assert!(output.contains("\"top_codes\""));
        assert!(output.contains("\"uncalibrated_runtime\""));
        assert!(output.contains("\"approximations\""));
        assert!(output.contains("\"approximation_policy_violations\""));
        assert!(output.contains("\"code\": \"approximate_serving_event_loop\""));
        assert!(output.contains("\"code\": \"serving_stack_uncalibrated\""));
        assert!(output.contains("\"code\": \"node_set_kv_handoff\""));
        assert!(output.contains("\"queue_delay_p90_ms\""));
        assert!(output.contains("\"queue_delay_max_ms\""));
        assert!(output.contains("\"peak_prefill_tokens\""));
        assert!(output.contains("\"peak_prefill_tokens_per_node\""));
        assert!(output.contains("\"peak_prefill_tokens_per_gpu\""));
        assert!(output.contains("\"peak_decode_sequences\""));
        assert!(output.contains("\"peak_resident_tokens\""));
        assert!(output.contains("\"peak_decode_sequences_per_node\""));
        assert!(output.contains("\"peak_resident_tokens_per_node\""));
        assert!(output.contains("\"peak_decode_sequences_per_gpu\""));
        assert!(output.contains("\"peak_resident_tokens_per_gpu\""));
        assert!(output.contains("\"peak_kv_blocks\""));
        assert!(output.contains("\"peak_allocated_kv_tokens\""));
        assert!(output.contains("\"peak_kv_fragmentation_tokens\""));
        assert!(output.contains("\"peak_kv_block_table_bytes\""));
        assert!(output.contains("\"peak_kv_blocks_per_node\""));
        assert!(output.contains("\"peak_kv_block_table_bytes_per_node\""));
        assert!(output.contains("\"peak_kv_blocks_per_gpu\""));
        assert!(output.contains("\"peak_kv_block_table_bytes_per_gpu\""));
        assert!(output.contains("\"kv_cache_owner_slots\""));
        assert!(output.contains("\"slot\""));
        assert!(output.contains("\"admitted_requests\""));
        assert!(output.contains("\"completed_requests\""));
        assert!(output.contains("\"rejected_requests\""));
        assert!(output.contains("\"timed_out_requests\""));
        assert!(output.contains("\"cancelled_requests\""));
        assert!(output.contains("\"queue_cap_ms\""));
        assert!(output.contains("\"queue_cap_request_count\""));
        assert!(output.contains("\"queue_cap_hit_count\""));
        assert!(output.contains("\"decode_iteration_queue_cap_ms\""));
        assert!(output.contains("\"decode_iteration_queue_cap_request_count\""));
        assert!(output.contains("\"decode_iteration_queue_cap_hit_count\""));
        assert!(output.contains("\"backpressure_rejections\""));
        assert!(output.contains("\"timeout_rejections\""));
        assert!(output.contains("\"backpressure_state\""));
        assert!(output.contains("\"deadline_constrained_requests\""));
        assert!(output.contains("\"deadline_missed_requests\""));
        assert!(output.contains("\"measured_requests\""));
        assert!(output.contains("\"measurement_start_ms\""));
        assert!(output.contains("\"measurement_end_ms\""));
        assert!(output.contains("\"measurement_window\""));
        assert!(output.contains("\"lifecycle_event_metric_request_count\""));
        assert!(output.contains("\"fallback_metric_request_count\""));
        assert!(output.contains("\"metric_source_counts\""));
        assert!(output.contains("\"metric_source\": \"request_lifecycle_events\""));
        assert!(output.contains("\"included_in_measurement_window\""));
        assert!(output.contains("\"measurement_window_source\": \"default\""));
        assert!(output.contains("\"steady_state_requested\""));
        assert!(output.contains("\"steady_state_candidate_start_ms\""));
        assert!(output.contains("\"steady_state_sample_count\""));
        assert!(output.contains("\"steady_state_candidate_e2el_cv\""));
        assert!(output.contains("\"steady_state_candidate_metric_count\""));
        assert!(output.contains("\"steady_state_candidate_worst_metric\""));
        assert!(output.contains("\"steady_state_candidate_output_tokens\""));
        assert!(output.contains("\"steady_state_candidate_throughput_tokens_per_s\""));
        assert!(output.contains("\"steady_state_candidate_metrics\""));
        assert!(output.contains("\"steady_state_candidate_utilization_count\""));
        assert!(output.contains("\"steady_state_candidate_worst_utilization_resource\""));
        assert!(output.contains("\"steady_state_candidate_worst_utilization_cv\""));
        assert!(output.contains("\"steady_state_candidate_utilization\""));
        assert!(output.contains("\"decode_sequence_per_node_utilization\""));
        assert!(output.contains("\"resident_token_per_node_utilization\""));
        assert!(output.contains("\"decode_sequence_per_gpu_utilization\""));
        assert!(output.contains("\"resident_token_per_gpu_utilization\""));
        assert!(output.contains("\"kv_block_utilization\""));
        assert!(output.contains("\"kv_block_per_node_utilization\""));
        assert!(output.contains("\"kv_block_per_gpu_utilization\""));
        assert!(output.contains("\"prefill_worker_queue_ms\""));
        assert!(output.contains("\"prefill_resource_queue_ms\""));
        assert!(output.contains("\"kv_worker_queue_ms\""));
        assert!(output.contains("\"kv_resource_queue_ms\""));
        assert!(output.contains("\"decode_worker_queue_ms\""));
        assert!(output.contains("\"decode_resource_queue_ms\""));
        assert!(output.contains("\"queue_p95_ms\""));
        assert!(output.contains("\"queue_max_ms\""));
        assert!(output.contains("\"worker_queue_ms\""));
        assert!(output.contains("\"resource_queue_ms\""));
        assert!(output.contains("\"phase_resource_utilization\""));
        assert!(output.contains("\"phase\": \"prefill\""));
        assert!(output.contains("\"phase\": \"kv_transfer\""));
        assert!(output.contains("\"phase\": \"decode\""));
        assert!(output.contains("\"resource_kind\""));
        assert!(output.contains("\"resource_kind\": \"kv_route\""));
        assert!(output.contains("\"kv_route_topology_summary\""));
        assert!(output.contains("\"route_resource_count\""));
        assert!(output.contains("\"rail_count\""));
        assert!(output.contains("\"rail_ids\""));
        assert!(output.contains("\"single_rail_dependency\""));
        assert!(output.contains("\"single_rail_id\""));
        assert!(output.contains("\"topology_bottlenecks\""));
        assert!(output.contains("\"severity\""));
        assert!(output.contains("\"single_rail_dependency\""));
        assert!(output.contains("\"kv_route_resource_summary\""));
        assert!(output.contains("\"resource_id\""));
        assert!(output.contains("\"kv_route:inter_node_fabric"));
        assert!(output.contains("\"path_observations\""));
        assert!(output.contains("\"estimated_transfer_ms\""));
        assert!(output.contains("\"request_observation_count\""));
        assert!(output.contains("\"request_observations\""));
        assert!(output.contains("\"metric_breakdowns\""));
        assert!(output.contains("\"group\": \"prefill_node\""));
        assert!(output.contains("\"group\": \"decode_node\""));
        assert!(output.contains("\"group\": \"prefill_route\""));
        assert!(output.contains("\"group\": \"decode_route\""));
        assert!(output.contains("\"request_count\""));
        assert!(output.contains("\"output_tokens\""));
        assert!(output.contains("\"request_id\""));
        assert!(output.contains("\"tenant\""));
        assert!(output.contains("\"model_id\""));
        assert!(output.contains("\"traffic_class\""));
        assert!(output.contains("\"shape_profile\""));
        assert!(output.contains("\"status\""));
        assert!(output.contains("\"status_time_ms\""));
        assert!(output.contains("\"failure_reason\""));
        assert!(output.contains("\"rejection\""));
        assert!(output.contains("\"priority\""));
        assert!(output.contains("\"slo\""));
        assert!(output.contains("\"ttft_slo_missed\""));
        assert!(output.contains("\"tpot_slo_missed\""));
        assert!(output.contains("\"itl_slo_missed\""));
        assert!(output.contains("\"e2el_slo_missed\""));
        assert!(output.contains("\"deadline_ms\""));
        assert!(output.contains("\"deadline_missed\""));
        assert!(output.contains("\"cancellation_ms\""));
        assert!(output.contains("\"completed\""));
        assert!(output.contains("\"prefill_node\""));
        assert!(output.contains("\"prefill_route_nodes\""));
        assert!(output.contains("\"prefill_route_gpus\""));
        assert!(output.contains("\"decode_node\""));
        assert!(output.contains("\"decode_route_nodes\""));
        assert!(output.contains("\"decode_route_gpus\""));
        assert!(output.contains("\"kv_cache_owner_gpus\""));
        assert!(output.contains("\"kv_block_tokens\""));
        assert!(output.contains("\"kv_cache_blocks\""));
        assert!(output.contains("\"kv_allocated_tokens\""));
        assert!(output.contains("\"kv_fragmentation_tokens\""));
        assert!(output.contains("\"kv_block_ownership\""));
        assert!(output.contains("\"allocation_id\""));
        assert!(output.contains("\"block_start\""));
        assert!(output.contains("\"block_end\""));
        assert!(output.contains("\"allocated_at_ms\""));
        assert!(output.contains("\"released_at_ms\""));
        assert!(output.contains("\"owner_worker_slots\""));
        assert!(output.contains("\"worker_slot_ownership\""));
        assert!(output.contains("\"decode_operation_ids\""));
        assert!(output.contains("\"decode_sequences\""));
        assert!(output.contains("\"allocated_kv_tokens\""));
        assert!(output.contains("\"block_table_entries\""));
        assert!(output.contains("\"block_table_bytes\""));
        assert!(output.contains("\"routing_policy\""));
        assert!(output.contains("\"routing_candidate_count\""));
        assert!(output.contains("\"routing_routable_candidate_count\""));
        assert!(output.contains("\"routing_estimated_e2el_ms\""));
        assert!(output.contains("\"routing_estimated_kv_resource_wait_ms\""));
        assert!(output.contains("\"routing_reason\""));
        assert!(output.contains("\"routing_candidates\""));
        assert!(output.contains("\"selected\""));
        assert!(output.contains("\"routable\""));
        assert!(output.contains("\"estimated_kv_resource_wait_ms\""));
        assert!(output.contains("\"kv_transfer_bytes\""));
        assert!(output.contains("\"kv_transfer_bottlenecks\""));
        assert!(output.contains("\"kv_transfer_resources\""));
        assert!(output.contains("\"kv_transfer_resource_dependencies\""));
        assert!(output.contains("\"kv_transfer_paths\""));
        assert!(output.contains("\"bottleneck_bandwidth_gbps\""));
        assert!(output.contains("\"resource_details\""));
        assert!(output.contains("\"kind\": \"inter_node_fabric\""));
        assert!(output.contains("\"rail_id\""));
        assert!(output.contains("\"kv_transfer_fit\""));
        assert!(output.contains("\"decode_token_start_ms\""));
        assert!(output.contains("\"decode_token_finish_ms\""));
        assert!(output.contains("\"inter_token_latency_ms\""));
        assert!(output.contains("\"phase_spans\""));
        assert!(output.contains("\"phase_breakdown\""));
        assert!(output.contains("\"category\": \"service\""));
        assert!(output.contains("\"first_decode_iteration\""));
        assert!(output.contains("\"contributes_to_ttft\""));
        assert!(output.contains("\"contributes_to_e2el\""));
        assert!(output.contains("\"queued_for_prefill\""));
        assert!(output.contains("\"lifecycle_events\""));
        assert!(output.contains("\"event\": \"arrived\""));
        assert!(output.contains("\"event\": \"kv_blocks_allocated\""));
        assert!(output.contains("\"event\": \"kv_blocks_released\""));
        assert!(output.contains("\"event\": \"completed\""));
        assert!(output.contains("\"decode_iteration\""));
        assert!(output.contains("\"metric_source\": \"request_lifecycle_events\""));
        assert!(output.contains("\"metric_derivation\""));
        assert!(output.contains("\"event_sourced\": true"));
        assert!(output.contains("\"ttft_start_event\": \"arrived\""));
        assert!(output.contains("\"ttft_end_event\": \"decode_iteration_finished:first\""));
        assert!(output.contains("\"tpot_sample_count\""));
        assert!(output.contains("\"request_output_tokens_per_s\""));
        assert!(output.contains("\"decode_iteration_count\""));
        assert!(output.contains("\"decode_iterations_truncated\""));
        assert!(output.contains("\"worker_summary\""));
        assert!(output.contains("\"role\": \"prefill_source\""));
        assert!(output.contains("\"role\": \"decode_owner\""));
        assert!(output.contains("\"role\": \"kv_cache_owner\""));
        assert!(output.contains("\"worker_slots\""));
        assert!(output.contains("\"assignment_count\""));
        assert!(output.contains("\"worker_assignments\""));
        assert!(output.contains("\"slot\""));
        assert!(output.contains("\"operation_ids\""));
        assert!(output.contains("\"decode_node_capacity\""));
        assert!(output.contains("\"decode_gpu_capacity\""));
        assert!(output.contains("\"serving_workers\""));
        assert!(output.contains("\"phase\": \"prefill\""));
        assert!(output.contains("\"phase\": \"decode\""));
        assert!(output.contains("\"configured_worker_slots\""));
        assert!(output.contains("\"peak_active_worker_slots\""));
        assert!(output.contains("\"worker_slot_utilization\""));
        assert!(output.contains("\"worker_queue_p95_ms\""));
        assert!(output.contains("\"resource_queue_p95_ms\""));
        assert!(output.contains("\"resource_utilization\""));
        assert!(output.contains("\"resource_occupancy_bucket_count\": 2"));
        assert!(output.contains("\"resource_occupancy_resource_count\": 1"));
        assert!(output.contains("\"resource_occupancy\""));
        assert!(output.contains("\"critical_path_ms\""));
        assert!(output.contains("\"critical_path_step_count\""));
        assert!(output.contains("\"critical_path\""));
        assert!(output.contains("\"rejections\""));
        assert!(output.contains("\"scheduled_operation_count\""));
        assert!(output.contains("\"scheduled_operations_truncated\""));
        assert!(output.contains("\"scheduled_operations\""));
        assert!(output.contains("request 0 prefill"));

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let request_observation = &parsed["results"][0]["request_observations"][0];
        let metric_derivation = &request_observation["metric_derivation"];
        assert_eq!(
            metric_derivation["metric_source"],
            request_observation["metric_source"]
        );
        assert_eq!(
            metric_derivation["output_tokens"],
            request_observation["output_tokens"]
        );
        assert_eq!(metric_derivation["event_sourced"].as_bool(), Some(true));
        assert_eq!(
            metric_derivation["ttft_end_event"].as_str(),
            Some("decode_iteration_finished:first")
        );
        assert!(
            metric_derivation["request_output_tokens_per_s"]
                .as_f64()
                .is_some_and(|throughput| throughput > 0.0)
        );
        let aggregate_metrics = &parsed["results"][0]["metrics"];
        assert_eq!(
            aggregate_metrics["e2el_slo_constrained_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(
            aggregate_metrics["e2el_slo_missed_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(
            aggregate_metrics["deadline_constrained_requests"].as_u64(),
            Some(2)
        );
        let traffic_breakdown =
            metric_breakdown_by_group_key(&parsed, "traffic_class", "tenant-a-soft-penalty");
        assert_eq!(traffic_breakdown["request_count"].as_u64(), Some(1));
        assert_eq!(traffic_breakdown["rejected_requests"].as_u64(), Some(0));
        assert_eq!(
            traffic_breakdown["lifecycle_event_metric_request_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            traffic_breakdown["metric_source_counts"][0]["metric_source"].as_str(),
            Some("request_lifecycle_events")
        );
        assert_eq!(
            traffic_breakdown["e2el_slo_constrained_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(
            traffic_breakdown["e2el_slo_missed_requests"].as_u64(),
            Some(1)
        );

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
        let _ = fs::remove_file(request_metrics_path);
        let _ = fs::remove_file(request_lifecycle_path);
        let _ = fs::remove_file(serving_metrics_path);
        let _ = fs::remove_file(metric_breakdowns_path);
        let _ = fs::remove_file(services_path);
        let _ = fs::remove_file(utilization_path);
        let _ = fs::remove_file(memory_pressure_path);
        let _ = fs::remove_file(timeline_path);
        let _ = fs::remove_file(occupancy_path);
        let _ = fs::remove_file(placement_evidence_path);
        let _ = fs::remove_file(worker_evidence_path);
        let _ = fs::remove_file(rejections_path);
        let _ = fs::remove_file(route_paths_path);
        let _ = fs::remove_file(kv_route_resources_path);
        let _ = fs::remove_file(bottlenecks_path);
        let _ = fs::remove_file(phase_calibration_path);
        let _ = fs::remove_file(approximations_path);
        let _ = fs::remove_file(rank_sensitivity_path);
    }

    #[test]
    fn runs_heterogeneous_disaggregated_example_with_topology_evidence() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cluster_path = root.join("examples/heterogeneous_cluster.toml");
        let workload_path = root.join("examples/heterogeneous_disaggregated_workload.toml");

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                "2".to_string(),
                "--max-candidates".to_string(),
                "4".to_string(),
                "--max-serving-pairs".to_string(),
                "4".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"feasible\": true"));
        assert!(output.contains("\"A100 80GB SXM\""));
        assert!(output.contains("\"H100 SXM5\""));
        assert!(output.contains("\"deployment_mode\": \"fully_disaggregated\""));
        assert!(output.contains("\"pool_search_summary\""));
        assert!(output.contains("\"generated_fully_disaggregated_count\""));
        assert!(output.contains("\"pool_topology\""));
        assert!(output.contains("\"dedicated_prefill_node_count\""));
        assert!(output.contains("\"dedicated_decode_node_count\""));
        assert!(output.contains("\"prefill_racks\": [\"rack_a\"]"));
        assert!(output.contains("\"decode_racks\": [\"rack_b\"]"));
        assert!(output.contains("\"prefill_failure_domains\": [\"az_a\"]"));
        assert!(output.contains("\"decode_failure_domains\": [\"az_b\"]"));
        assert!(output.contains("\"prefill_node_labels\": [\"prefill\", \"rack_a\"]"));
        assert!(output.contains("\"decode_node_labels\": [\"decode\", \"rack_b\"]"));
        assert!(output.contains("\"route_coverage\""));
        assert!(output.contains("\"kv_route_topology_summary\""));
        assert!(output.contains("\"rail_ids\": [0]"));
        assert!(output.contains("\"prefill_gpu_types\""));
        assert!(output.contains("\"decode_gpu_types\""));
        assert!(output.contains("\"prefill_gpu_label_counts\""));
        assert!(output.contains("\"decode_gpu_label_counts\""));
        assert!(output.contains("\"metrics\""));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"tpot_ms\""));
        assert!(output.contains("\"throughput_tokens_per_s\""));
        assert!(output.contains("\"e2el_ms\""));
    }

    #[test]
    fn runs_heterogeneous_colocated_example_with_serving_metrics() {
        let output = run_heterogeneous_example("heterogeneous_colocated_workload.toml");

        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"feasible\": true"));
        assert!(output.contains("\"deployment_mode\": \"colocated\""));
        assert!(output.contains("\"shared_node_count\": 1"));
        assert!(output.contains("\"dedicated_prefill_node_count\": 0"));
        assert!(output.contains("\"dedicated_decode_node_count\": 0"));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"tpot_ms\""));
        assert!(output.contains("\"throughput_tokens_per_s\""));
        assert!(output.contains("\"e2el_ms\""));
    }

    #[test]
    fn runs_homogeneous_serving_example_with_colocated_metrics() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cluster_path = root.join("examples/h100_cluster.toml");
        let workload_path = root.join("examples/homogeneous_serving_workload.toml");

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
                "--max-serving-pairs".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"feasible\": true"));
        assert!(output.contains("\"gpu\": \"H100 SXM5\""));
        assert!(output.contains("\"kind\": \"fat_tree\""));
        assert!(output.contains("\"deployment_mode\": \"colocated\""));
        assert!(output.contains("\"shared_node_count\": 1"));
        assert!(output.contains("\"serving_services\""));
        assert!(output.contains("\"phase\": \"prefill\""));
        assert!(output.contains("\"phase\": \"decode\""));
        assert!(output.contains("\"transfer: local\""));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"tpot_ms\""));
        assert!(output.contains("\"throughput_tokens_per_s\""));
        assert!(output.contains("\"e2el_ms\""));
    }

    #[test]
    fn runs_heterogeneous_partially_disaggregated_example_with_serving_metrics() {
        let output =
            run_heterogeneous_example("heterogeneous_partially_disaggregated_workload.toml");

        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"feasible\": true"));
        assert!(output.contains("\"deployment_mode\": \"partially_disaggregated\""));
        assert!(output.contains("\"shared_node_count\": 1"));
        assert!(output.contains("\"dedicated_prefill_node_count\": 1"));
        assert!(output.contains("\"dedicated_decode_node_count\": 1"));
        assert!(output.contains("\"route_coverage\""));
        assert!(output.contains("\"kv_route_topology_summary\""));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"tpot_ms\""));
        assert!(output.contains("\"throughput_tokens_per_s\""));
        assert!(output.contains("\"e2el_ms\""));
    }

    #[test]
    fn runs_heterogeneous_rail_island_example_with_interconnect_evidence() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cluster_path = root.join("examples/heterogeneous_rail_island_cluster.toml");
        let workload_path = root.join("examples/heterogeneous_rail_island_workload.toml");

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "2".to_string(),
                "--request-limit".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "2".to_string(),
                "--max-serving-pairs".to_string(),
                "2".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"feasible\": true"));
        assert!(output.contains("\"link_count\": 3"));
        assert!(output.contains("\"label\": \"IB NDR\""));
        assert!(output.contains("\"label\": \"Ethernet 100G\""));
        assert!(output.contains("\"bandwidth_gbps\": 25.000000"));
        assert!(output.contains("\"rack\": \"rack_east\""));
        assert!(output.contains("\"rack\": \"rack_west\""));
        assert!(output.contains("\"rack\": \"rack_cold\""));
        assert!(output.contains("\"island\": \"island_east\""));
        assert!(output.contains("\"island\": \"island_west\""));
        assert!(output.contains("\"island\": \"island_cold\""));
        assert!(output.contains("\"failure_domain\": \"az_a\""));
        assert!(output.contains("\"failure_domain\": \"az_b\""));
        assert!(output.contains("\"failure_domain\": \"az_c\""));
        assert!(output.contains("pool-fast-rail-prefill-decode"));
        assert!(output.contains("pool-slow-ethernet-prefill-decode"));
        assert!(output.contains("\"deployment_mode\": \"fully_disaggregated\""));
        assert!(output.contains("\"route_coverage\""));
        assert!(output.contains("\"kv_route_topology_summary\""));
        assert!(output.contains("\"inter_node_route_resource_count\": 1"));
        assert!(output.contains("\"custom Ethernet 100G node 0 <-> node 2 rail 0\""));
        assert!(output.contains("\"topology_bottlenecks\""));
        assert!(output.contains("\"single_rail_dependency\""));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"tpot_ms\""));
        assert!(output.contains("\"throughput_tokens_per_s\""));
        assert!(output.contains("\"e2el_ms\""));
    }

    #[test]
    fn runs_heterogeneous_oversubscribed_example_with_contention_evidence() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cluster_path = root.join("examples/heterogeneous_oversubscribed_cluster.toml");
        let workload_path = root.join("examples/heterogeneous_oversubscribed_workload.toml");

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                "1".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
                "--max-serving-pairs".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"feasible\": true"));
        assert!(output.contains("\"link_count\": 1"));
        assert!(output.contains("\"label\": \"IB NDR\""));
        assert!(output.contains("\"bandwidth_gbps\": 25.000000"));
        assert!(output.contains("pool-oversubscribed-rack-uplink"));
        assert!(output.contains("\"deployment_mode\": \"fully_disaggregated\""));
        assert!(output.contains("\"route_coverage\""));
        assert!(output.contains("\"kv_route_topology_summary\""));
        assert!(output.contains("\"single_rail_dependency\""));
        assert!(output.contains("\"kv_route_resource_queueing\""));
        assert!(output.contains("\"kv_route_resource_summary\""));
        assert!(output.contains("\"estimated_transfer_ms\""));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"tpot_ms\""));
        assert!(output.contains("\"throughput_tokens_per_s\""));
        assert!(output.contains("\"e2el_ms\""));
    }

    #[test]
    fn runs_calibrated_trace_run_example_with_profile_evidence() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let run_path = root.join("examples/trace_run.toml");

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--run".to_string(),
                run_path.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                "2".to_string(),
                "--max-candidates".to_string(),
                "2".to_string(),
                "--max-prefill-candidates".to_string(),
                "2".to_string(),
                "--max-decode-candidates".to_string(),
                "2".to_string(),
                "--max-serving-pairs".to_string(),
                "2".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"mode\": \"serving\""));
        assert!(output.contains("\"objective\": \"minimize_slo_miss_rate\""));
        assert!(output.contains("\"name\": \"h100-a100-disaggregated-mvp\""));
        assert!(output.contains("\"backend_version\": \"generic-continuous-batching 0.1\""));
        assert!(output.contains("\"profile_runtime\": \"warn\""));
        assert!(output.contains("\"calibration_fit_applications\""));
        assert!(output.contains("\"fit_name\": \"h100-prefill-latency-fit\""));
        assert!(output.contains("\"confidence_interval\""));
        assert!(output.contains("\"confidence_level\""));
        assert!(output.contains("\"serving_phase_calibration\""));
        assert!(output.contains("\"phase\": \"prefill\""));
        assert!(output.contains("\"phase\": \"decode\""));
        assert!(output.contains("\"calibration_uncertainty\""));
        assert!(output.contains("\"ttft_ms\""));
        assert!(output.contains("\"tpot_ms\""));
        assert!(output.contains("\"throughput_tokens_per_s\""));
        assert!(output.contains("\"e2el_ms\""));
    }

    #[test]
    fn checked_example_json_outputs_are_deterministic() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let examples = root.join("examples");
        let cases = [
            (
                "homogeneous colocated",
                examples.join("h100_cluster.toml"),
                examples.join("homogeneous_serving_workload.toml"),
                "colocated",
                "1",
                "1",
                "1",
            ),
            (
                "heterogeneous colocated",
                examples.join("heterogeneous_cluster.toml"),
                examples.join("heterogeneous_colocated_workload.toml"),
                "colocated",
                "1",
                "4",
                "4",
            ),
            (
                "heterogeneous partially disaggregated",
                examples.join("heterogeneous_cluster.toml"),
                examples.join("heterogeneous_partially_disaggregated_workload.toml"),
                "partially_disaggregated",
                "2",
                "4",
                "4",
            ),
            (
                "heterogeneous fully disaggregated",
                examples.join("heterogeneous_cluster.toml"),
                examples.join("heterogeneous_disaggregated_workload.toml"),
                "fully_disaggregated",
                "2",
                "4",
                "4",
            ),
            (
                "rail island fully disaggregated",
                examples.join("heterogeneous_rail_island_cluster.toml"),
                examples.join("heterogeneous_rail_island_workload.toml"),
                "fully_disaggregated",
                "1",
                "2",
                "2",
            ),
            (
                "oversubscribed fully disaggregated",
                examples.join("heterogeneous_oversubscribed_cluster.toml"),
                examples.join("heterogeneous_oversubscribed_workload.toml"),
                "fully_disaggregated",
                "1",
                "1",
                "1",
            ),
        ];

        for (
            name,
            cluster_path,
            workload_path,
            deployment_mode,
            request_limit,
            max_candidates,
            max_serving_pairs,
        ) in cases
        {
            let first = run_example_json(
                &cluster_path,
                &workload_path,
                request_limit,
                max_candidates,
                max_serving_pairs,
            );
            let second = run_example_json(
                &cluster_path,
                &workload_path,
                request_limit,
                max_candidates,
                max_serving_pairs,
            );
            let mut first_json: serde_json::Value = serde_json::from_str(&first).unwrap();
            let mut second_json: serde_json::Value = serde_json::from_str(&second).unwrap();
            normalize_runtime_elapsed_ms(&mut first_json);
            normalize_runtime_elapsed_ms(&mut second_json);

            assert_eq!(
                first_json, second_json,
                "{name} example JSON should be deterministic except runtime_elapsed_ms"
            );
            assert!(
                first.contains("\"mode\": \"serving\""),
                "{name} should run the serving solver"
            );
            assert!(
                first.contains("\"feasible\": true"),
                "{name} should produce a feasible candidate"
            );
            assert!(
                first.contains(&format!("\"deployment_mode\": \"{deployment_mode}\"")),
                "{name} should preserve its expected disaggregation mode"
            );
            assert!(first.contains("\"metrics\""), "{name} should emit metrics");
            assert!(first.contains("\"ttft_ms\""), "{name} should emit TTFT");
            assert!(first.contains("\"tpot_ms\""), "{name} should emit TPOT");
            assert!(
                first.contains("\"throughput_tokens_per_s\""),
                "{name} should emit throughput"
            );
            assert!(first.contains("\"e2el_ms\""), "{name} should emit E2EL");
            assert!(
                first.contains("\"measurement_window\""),
                "{name} should emit measurement-window provenance"
            );
            assert!(
                first.contains("\"metric_source_counts\""),
                "{name} should emit metric-source provenance"
            );
            assert!(
                first.contains("\"request_observations\""),
                "{name} should emit request observations"
            );
            assert!(
                first.contains("\"included_in_measurement_window\""),
                "{name} should emit per-request measurement inclusion"
            );
            assert!(
                first.contains("\"serving_services\""),
                "{name} should emit service evidence"
            );
            assert!(
                first.contains("\"route_coverage\""),
                "{name} should emit route coverage"
            );
            assert!(
                first.contains("\"approximations\""),
                "{name} should emit approximation evidence"
            );
        }
    }

    #[test]
    fn cli_edge_request_outcome_json_is_deterministic_and_auditable() {
        let capacity = run_temp_cli_json_twice(
            "edge-decode-capacity",
            &h100_single_node_cluster(),
            &edge_case_serving_workload(
                r#"
                request_count = 2
                arrival_gap_s = 0.0
                decode_capacity_policy = "request_reject"
                max_decode_sequences = 1
                max_resident_tokens = 8192
                max_decode_sequences_per_node = 1
                max_resident_tokens_per_node = 8192

                [[serving.traffic.requests]]
                request_id = "capacity-completed"
                arrival_s = 0.0
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 16
                max_sequence_tokens = 512

                [[serving.traffic.requests]]
                request_id = "capacity-rejected"
                arrival_s = 0.0
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 16
                max_sequence_tokens = 512
                "#,
            ),
        );
        assert_metric_count(&capacity, "scheduled_requests", 2);
        assert_metric_count(&capacity, "completed_requests", 1);
        assert_metric_count(&capacity, "rejected_requests", 1);
        assert_measurement_window_counts(&capacity, 2, 1, 1, 1, 0, 0);
        let capacity_priority = metric_breakdown_by_group_key(&capacity, "priority", "priority-0");
        assert_eq!(capacity_priority["request_count"].as_u64(), Some(2));
        assert_eq!(capacity_priority["completed_requests"].as_u64(), Some(1));
        assert_eq!(capacity_priority["failed_requests"].as_u64(), Some(1));
        assert_eq!(capacity_priority["rejected_requests"].as_u64(), Some(1));
        assert_eq!(capacity_priority["timed_out_requests"].as_u64(), Some(0));
        assert_eq!(capacity_priority["cancelled_requests"].as_u64(), Some(0));
        assert_eq!(
            capacity_priority["lifecycle_event_metric_request_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            capacity_priority["fallback_metric_request_count"].as_u64(),
            Some(0)
        );
        assert_eq!(
            capacity_priority["metric_source_counts"][0]["metric_source"].as_str(),
            Some("request_lifecycle_events")
        );
        assert_eq!(
            capacity_priority["metric_source_counts"][0]["request_count"].as_u64(),
            Some(1)
        );
        let capacity_observations = request_observations(&capacity);
        assert_eq!(capacity_observations.len(), 2);
        let completed = observation_by_request_id(capacity_observations, "capacity-completed");
        assert_request_status(completed, "completed");
        assert_lifecycle_event(completed, "completed");
        assert_request_metric_derivation(completed, true, 16);
        let rejected = observation_by_request_id(capacity_observations, "capacity-rejected");
        assert_request_status(rejected, "rejected_admission");
        assert_lifecycle_event(rejected, "rejected_admission");
        assert_rejection_code(rejected, "decode", "decode_capacity_exceeded");
        assert_request_metric_derivation(rejected, false, 0);
        assert!(rejected["metric_derivation"]["ttft_end_event"].is_null());
        assert!(rejected["metric_derivation"]["tpot_start_event"].is_null());
        assert!(rejected["metric_derivation"]["tpot_end_event"].is_null());

        let timeout = run_temp_cli_json_twice(
            "edge-timeout",
            &h100_single_node_cluster(),
            &edge_case_serving_workload(
                r#"
                request_count = 1
                arrival_gap_s = 0.0
                request_timeout_s = 0.000000001
                max_decode_sequences = 8
                max_resident_tokens = 8192
                max_decode_sequences_per_node = 8
                max_resident_tokens_per_node = 8192

                [[serving.traffic.requests]]
                request_id = "timeout-request"
                arrival_s = 0.0
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 2
                max_sequence_tokens = 512
                "#,
            ),
        );
        assert_metric_count(&timeout, "scheduled_requests", 1);
        assert_metric_count(&timeout, "completed_requests", 0);
        assert_metric_count(&timeout, "timed_out_requests", 1);
        assert_metric_count(&timeout, "measured_requests", 0);
        assert_measurement_window_counts(&timeout, 1, 0, 1, 0, 1, 0);
        let timeout_priority = metric_breakdown_by_group_key(&timeout, "priority", "priority-0");
        assert_eq!(timeout_priority["request_count"].as_u64(), Some(1));
        assert_eq!(timeout_priority["failed_requests"].as_u64(), Some(1));
        assert_eq!(timeout_priority["timed_out_requests"].as_u64(), Some(1));
        assert_eq!(
            timeout_priority["lifecycle_event_metric_request_count"].as_u64(),
            Some(0)
        );
        let timeout_observation =
            observation_by_request_id(request_observations(&timeout), "timeout-request");
        assert_request_status(timeout_observation, "timed_out");
        assert_lifecycle_event(timeout_observation, "timed_out");
        assert_rejection_code(
            timeout_observation,
            "end_to_end",
            "request_timeout_exceeded",
        );
        assert_request_metric_derivation(timeout_observation, false, 0);
        assert_eq!(
            timeout_observation["metric_derivation"]["e2el_end_event"].as_str(),
            Some("timed_out")
        );

        let cancellation = run_temp_cli_json_twice(
            "edge-cancellation",
            &h100_single_node_cluster(),
            &edge_case_serving_workload(
                r#"
                request_count = 1
                arrival_gap_s = 0.0
                max_decode_sequences = 8
                max_resident_tokens = 8192
                max_decode_sequences_per_node = 8
                max_resident_tokens_per_node = 8192

                [[serving.traffic.requests]]
                request_id = "cancelled-request"
                arrival_s = 0.0
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 2
                max_sequence_tokens = 512
                cancel_after_s = 0.000000001
                deadline_after_s = 10.0
                "#,
            ),
        );
        assert_metric_count(&cancellation, "scheduled_requests", 1);
        assert_metric_count(&cancellation, "completed_requests", 0);
        assert_metric_count(&cancellation, "cancelled_requests", 1);
        assert_metric_count(&cancellation, "measured_requests", 0);
        assert_measurement_window_counts(&cancellation, 1, 0, 1, 0, 0, 1);
        let cancellation_priority =
            metric_breakdown_by_group_key(&cancellation, "priority", "priority-0");
        assert_eq!(cancellation_priority["request_count"].as_u64(), Some(1));
        assert_eq!(cancellation_priority["failed_requests"].as_u64(), Some(1));
        assert_eq!(
            cancellation_priority["cancelled_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(
            cancellation_priority["lifecycle_event_metric_request_count"].as_u64(),
            Some(0)
        );
        let cancellation_observation =
            observation_by_request_id(request_observations(&cancellation), "cancelled-request");
        assert_request_status(cancellation_observation, "cancelled");
        assert_lifecycle_event(cancellation_observation, "cancelled");
        assert_eq!(
            cancellation_observation["kv_block_ownership"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert_request_metric_derivation(cancellation_observation, false, 0);
        assert!(cancellation_observation["metric_derivation"]["ttft_end_event"].is_null());
        assert!(cancellation_observation["metric_derivation"]["tpot_start_event"].is_null());
        assert!(cancellation_observation["metric_derivation"]["tpot_end_event"].is_null());

        let slo_deadline = run_temp_cli_json_twice(
            "edge-slo-deadline",
            &h100_single_node_cluster(),
            &edge_case_serving_workload(
                r#"
                request_count = 1
                arrival_gap_s = 0.0
                max_decode_sequences = 8
                max_resident_tokens = 8192
                max_decode_sequences_per_node = 8
                max_resident_tokens_per_node = 8192

                [[serving.traffic.requests]]
                request_id = "slo-deadline-miss"
                arrival_s = 0.0
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 2
                max_sequence_tokens = 512
                ttft_slo_s = 0.000000001
                tpot_slo_s = 0.000000001
                itl_slo_s = 0.000000001
                e2el_slo_s = 0.000000001
                deadline_after_s = 0.000000001
                "#,
            ),
        );
        assert_metric_count(&slo_deadline, "scheduled_requests", 1);
        assert_metric_count(&slo_deadline, "completed_requests", 1);
        assert_metric_count(&slo_deadline, "deadline_missed_requests", 1);
        assert_metric_count(&slo_deadline, "measured_requests", 1);
        assert_metric_count(&slo_deadline, "ttft_slo_constrained_requests", 1);
        assert_metric_count(&slo_deadline, "ttft_slo_missed_requests", 1);
        assert_metric_count(&slo_deadline, "tpot_slo_constrained_requests", 1);
        assert_metric_count(&slo_deadline, "tpot_slo_missed_requests", 1);
        assert_metric_count(&slo_deadline, "itl_slo_constrained_requests", 1);
        assert_metric_count(&slo_deadline, "itl_slo_missed_requests", 1);
        assert_metric_count(&slo_deadline, "e2el_slo_constrained_requests", 1);
        assert_metric_count(&slo_deadline, "e2el_slo_missed_requests", 1);
        assert_measurement_window_counts(&slo_deadline, 1, 1, 0, 0, 0, 0);
        assert_eq!(
            slo_deadline["results"][0]["measurement_window"]["deadline_constrained_request_count"]
                .as_u64(),
            Some(1)
        );
        assert_eq!(
            slo_deadline["results"][0]["measurement_window"]["deadline_missed_request_count"]
                .as_u64(),
            Some(1)
        );
        let slo_priority = metric_breakdown_by_group_key(&slo_deadline, "priority", "priority-0");
        assert_eq!(
            slo_priority["deadline_constrained_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(slo_priority["deadline_missed_requests"].as_u64(), Some(1));
        assert_eq!(
            slo_priority["ttft_slo_constrained_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(slo_priority["ttft_slo_missed_requests"].as_u64(), Some(1));
        assert_eq!(
            slo_priority["tpot_slo_constrained_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(slo_priority["tpot_slo_missed_requests"].as_u64(), Some(1));
        assert_eq!(
            slo_priority["itl_slo_constrained_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(slo_priority["itl_slo_missed_requests"].as_u64(), Some(1));
        assert_eq!(
            slo_priority["e2el_slo_constrained_requests"].as_u64(),
            Some(1)
        );
        assert_eq!(slo_priority["e2el_slo_missed_requests"].as_u64(), Some(1));
        let slo_observation =
            observation_by_request_id(request_observations(&slo_deadline), "slo-deadline-miss");
        assert_request_status(slo_observation, "completed");
        assert_lifecycle_event(slo_observation, "completed");
        assert_eq!(slo_observation["ttft_slo_missed"].as_bool(), Some(true));
        assert_eq!(slo_observation["tpot_slo_missed"].as_bool(), Some(true));
        assert_eq!(slo_observation["itl_slo_missed"].as_bool(), Some(true));
        assert_eq!(slo_observation["e2el_slo_missed"].as_bool(), Some(true));
        assert_eq!(slo_observation["deadline_missed"].as_bool(), Some(true));
        assert_request_metric_derivation(slo_observation, true, 2);
    }

    #[test]
    fn request_metrics_csv_reports_terminal_metric_provenance() {
        let workload = edge_case_serving_workload(
            r#"
            request_count = 2
            arrival_gap_s = 0.0
            decode_capacity_policy = "request_reject"
            max_decode_sequences = 1
            max_resident_tokens = 8192
            max_decode_sequences_per_node = 1
            max_resident_tokens_per_node = 8192

            [[serving.traffic.requests]]
            request_id = "csv-completed"
            arrival_s = 0.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 2
            max_sequence_tokens = 512

            [[serving.traffic.requests]]
            request_id = "csv-rejected"
            arrival_s = 0.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 2
            max_sequence_tokens = 512
            "#,
        );
        let csv = run_temp_request_metrics_csv(
            "edge-request-metrics-provenance",
            &h100_single_node_cluster(),
            &workload,
        );

        assert_csv_row_field(&csv, "request_id", "csv-completed", "event_sourced", "true");
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-completed",
            "terminal_event",
            "completed",
        );
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-completed",
            "metric_unavailable_reason",
            "",
        );
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-completed",
            "ttft_end_event",
            "decode_iteration_finished:first",
        );
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-completed",
            "throughput_duration_end_event",
            "decode_iteration_finished:last",
        );
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-completed",
            "decode_finish_event_count",
            "2",
        );
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-completed",
            "tpot_sample_count",
            "1",
        );

        assert_csv_row_field(&csv, "request_id", "csv-rejected", "event_sourced", "false");
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-rejected",
            "terminal_event",
            "rejected_admission",
        );
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-rejected",
            "metric_unavailable_reason",
            "request_not_completed:rejected_admission",
        );
        assert_csv_row_field(&csv, "request_id", "csv-rejected", "ttft_end_event", "");
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-rejected",
            "throughput_duration_end_event",
            "",
        );
        assert_csv_row_field(
            &csv,
            "request_id",
            "csv-rejected",
            "decode_finish_event_count",
            "0",
        );
        assert_csv_row_field(&csv, "request_id", "csv-rejected", "tpot_sample_count", "0");
    }

    #[test]
    fn cli_serving_control_plane_json_is_deterministic_and_auditable() {
        let fixed = run_temp_cli_json_twice(
            "control-fixed-arrivals",
            &h100_single_node_cluster(),
            &control_plane_serving_workload(
                r#"
                request_count = 3
                arrival = "fixed"
                arrival_gap_ms = 1.0
                routing_policy = "round_robin"
                prefill_batching = "independent"
                decode_batching = "independent"
                max_decode_sequences = 8
                max_resident_tokens = 8192
                max_decode_sequences_per_node = 8
                max_resident_tokens_per_node = 8192
                batch_sizes = [1]
                prompt_tokens = [128]
                decode_tokens = [2]
                "#,
            ),
        );
        assert_arrivals_ms(&fixed, &[0.0, 1.0, 2.0]);

        let poisson = run_temp_cli_json_twice(
            "control-poisson-arrivals",
            &h100_single_node_cluster(),
            &control_plane_serving_workload(
                r#"
                request_count = 4
                arrival = "poisson"
                arrival_rate_per_s = 1000.0
                arrival_seed = 42
                routing_policy = "topology_aware"
                prefill_batching = "independent"
                decode_batching = "independent"
                max_decode_sequences = 8
                max_resident_tokens = 8192
                max_decode_sequences_per_node = 8
                max_resident_tokens_per_node = 8192
                batch_sizes = [1]
                prompt_tokens = [128]
                decode_tokens = [2]
                "#,
            ),
        );
        let poisson_arrivals = arrival_ms_values(&poisson);
        assert_eq!(poisson_arrivals.len(), 4);
        assert_monotonic_arrivals(&poisson_arrivals);
        assert_ne!(poisson_arrivals, vec![0.0, 1.0, 2.0, 3.0]);
        assert!(
            request_observations(&poisson).iter().all(|observation| {
                observation["routing_policy"].as_str() == Some("topology_aware")
                    && observation["routing_candidate_count"].as_u64().unwrap_or(0) > 0
            }),
            "poisson topology-aware run should keep routing evidence per request"
        );

        let safe_case_name = "control-trace-replay";
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let trace_path =
            std::env::temp_dir().join(format!("inference-sim-{safe_case_name}-{nanos}.csv"));
        fs::write(
            &trace_path,
            "\
request_id,tenant,model_id,arrival_ms,priority,batch_size,prompt_tokens,decode_tokens,max_sequence_tokens
trace-a,tenant-a,model-a,0.0,2,1,256,4,512
trace-b,tenant-b,model-a,0.0,1,1,256,4,512
",
        )
        .unwrap();
        let trace = run_temp_cli_json_twice(
            safe_case_name,
            &h100_single_node_cluster(),
            &control_plane_serving_workload(&format!(
                r#"
                request_count = 2
                trace_csv = "{}"
                routing_policy = "round_robin"
                prefill_batching = "continuous"
                max_prefill_batch_tokens = 128
                max_prefill_chunk_tokens = 64
                max_prefill_worker_slots_per_gpu = 1
                decode_batching = "continuous"
                max_decode_batch_tokens = 2
                max_decode_worker_slots_per_gpu = 1
                max_decode_sequences = 8
                max_resident_tokens = 8192
                max_decode_sequences_per_node = 8
                max_resident_tokens_per_node = 8192
                "#,
                trace_path.display()
            )),
        );
        let _ = fs::remove_file(&trace_path);
        let trace_observations = request_observations(&trace);
        assert_eq!(trace_observations.len(), 2);
        assert_eq!(
            trace_observations[0]["request_id"].as_str(),
            Some("trace-a")
        );
        assert_eq!(
            trace_observations[1]["request_id"].as_str(),
            Some("trace-b")
        );
        assert_arrivals_ms(&trace, &[0.0, 0.0]);
        assert!(
            trace_observations
                .iter()
                .all(|observation| observation["prefill_chunks"].as_u64().unwrap_or(0) > 1),
            "continuous prefill with chunk limit should expose chunked batching evidence"
        );

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let topology = run_cli_json_paths_twice(
            "control-topology-diagnostics",
            &root.join("examples/heterogeneous_cluster.toml"),
            &root.join("examples/heterogeneous_disaggregated_workload.toml"),
        );
        assert!(
            topology["cluster_inventory"]["topology_diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diagnostic| {
                    diagnostic["code"].as_str() == Some("gpu_nic_path_host_staged")
                }),
            "heterogeneous cluster should emit topology diagnostics"
        );
        let topology_result = &topology["results"][0];
        assert_eq!(topology_result["feasible"].as_bool(), Some(true));
        assert!(
            request_observations(&topology).iter().any(|observation| {
                observation["routing_policy"].as_str() == Some("topology_aware")
                    && observation["routing_candidate_count"].as_u64().unwrap_or(0) > 0
                    && !observation["kv_transfer_resources"]
                        .as_array()
                        .unwrap()
                        .is_empty()
            }),
            "request observations should tie topology-aware routing to selected KV path resources"
        );

        let route_contention = run_cli_json_paths_twice(
            "control-route-contention",
            &root.join("examples/heterogeneous_oversubscribed_cluster.toml"),
            &root.join("examples/heterogeneous_oversubscribed_workload.toml"),
        );
        let result = &route_contention["results"][0];
        assert_eq!(result["feasible"].as_bool(), Some(true));
        assert!(json_number(&result["metrics"]["kv_resource_queue_ms"]) > 0.0);
        assert!(
            result["topology_bottlenecks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|bottleneck| {
                    bottleneck["code"].as_str() == Some("kv_route_resource_queueing")
                }),
            "disaggregated example should report route-resource contention"
        );
        assert!(
            result["topology_bottlenecks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|bottleneck| bottleneck["code"].as_str() == Some("single_rail_dependency")),
            "disaggregated example should report rail-locality topology risk"
        );
        assert!(
            request_observations(&route_contention)
                .iter()
                .any(|observation| {
                    observation["routing_policy"].as_str() == Some("topology_aware")
                        && observation["routing_candidate_count"].as_u64().unwrap_or(0) > 0
                        && !observation["kv_transfer_resources"]
                            .as_array()
                            .unwrap()
                            .is_empty()
                        && json_number(&observation["kv_resource_queue_ms"]) > 0.0
                }),
            "request observations should tie routing, KV path resources, and route queueing together"
        );
    }

    #[test]
    fn cli_rejects_invalid_v1_toml_configs_before_solving() {
        struct InvalidCase {
            name: &'static str,
            cluster: String,
            workload: String,
            expected_error: &'static str,
        }

        let cases = vec![
            InvalidCase {
                name: "duplicate-node-id",
                cluster: r#"
                    schema_version = 1

                    [cluster]
                    preset = "custom"

                    [[nodes]]
                    id = 0
                    gpu = "h100_sxm"
                    gpu_count = 1

                    [[nodes]]
                    id = 0
                    gpu = "h100_sxm"
                    gpu_count = 1
                "#
                .to_string(),
                workload: minimal_parallelism_workload("bf16", ""),
                expected_error: "duplicate node id 0",
            },
            InvalidCase {
                name: "invalid-nic-affinity",
                cluster: r#"
                    schema_version = 1

                    [cluster]
                    preset = "custom"

                    [[nodes]]
                    id = 0
                    gpu = "h100_sxm"
                    gpu_count = 2
                    nics = { count = 1, affinity = "dedicated" }
                "#
                .to_string(),
                workload: minimal_parallelism_workload("bf16", ""),
                expected_error: "dedicated affinity requires nics.count >= gpu_count",
            },
            InvalidCase {
                name: "invalid-rail-count",
                cluster: r#"
                    schema_version = 1

                    [cluster]
                    preset = "h100_sxm"
                    node_count = 1

                    [interconnect]
                    kind = "ib"
                    variant = "ndr"

                    [nics]
                    count = 4
                    rail_count = 5
                    affinity = "uniform"
                "#
                .to_string(),
                workload: minimal_parallelism_workload("bf16", ""),
                expected_error: "nics.rail_count must be less than or equal to nics.count",
            },
            InvalidCase {
                name: "invalid-gpu-nic-path",
                cluster: r#"
                    schema_version = 1

                    [cluster]
                    preset = "custom"

                    [[nodes]]
                    id = 0
                    gpu = "h100_sxm"
                    gpu_count = 1
                    nics = { count = 1, affinity = "uniform", gpu_nic_paths = [{ gpu = 7, nic = 0, bandwidth_gbps = 100.0 }] }
                "#
                .to_string(),
                workload: minimal_parallelism_workload("bf16", ""),
                expected_error: "gpu_nic_paths references unknown local GPU id 7",
            },
            InvalidCase {
                name: "impossible-rank-placement",
                cluster: h100_single_node_cluster(),
                workload: minimal_parallelism_workload(
                    "bf16",
                    r#"
                    [[placement.ranks]]
                    rank = 0
                    node = 0
                    gpu = 0
                    "#,
                ),
                expected_error: "placement.ranks defines 1 ranks but no search candidate has that total rank count",
            },
            InvalidCase {
                name: "unsupported-dtype-hardware",
                cluster: r#"
                    schema_version = 1

                    [cluster]
                    preset = "custom"

                    [[nodes]]
                    id = 0
                    gpu = "a100_80gb"
                    gpu_count = 1
                "#
                .to_string(),
                workload: minimal_parallelism_workload("fp8", ""),
                expected_error: "no available fp8-capable GPUs",
            },
            InvalidCase {
                name: "disconnected-serving-pools",
                cluster: r#"
                    schema_version = 1

                    [cluster]
                    preset = "custom"

                    [[nodes]]
                    id = 0
                    gpu = "h100_sxm"
                    gpu_count = 1
                    nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }

                    [[nodes]]
                    id = 1
                    gpu = "h100_sxm"
                    gpu_count = 1
                    nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }
                "#
                .to_string(),
                workload: minimal_serving_workload(
                    "bf16",
                    r#"
                    [serving]
                    mode = "disaggregated"
                    require_routable_pools = true
                    prefill_nodes = [0]
                    decode_nodes = [1]
                    "#,
                ),
                expected_error: "has no routable KV transfer path",
            },
            InvalidCase {
                name: "invalid-disaggregation-route",
                cluster: h100_two_node_cluster(),
                workload: minimal_serving_workload(
                    "bf16",
                    r#"
                    [serving]
                    mode = "fully_disaggregated"
                    prefill_nodes = [0]
                    decode_nodes = [0]
                    "#,
                ),
                expected_error: "uses colocated prefill/decode nodes",
            },
        ];

        for case in cases {
            assert_cli_config_error_contains(
                case.name,
                &case.cluster,
                &case.workload,
                case.expected_error,
            );
        }
    }

    fn run_heterogeneous_example(workload_file: &str) -> String {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cluster_path = root.join("examples/heterogeneous_cluster.toml");
        let workload_path = root.join("examples").join(workload_file);

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                "2".to_string(),
                "--max-candidates".to_string(),
                "4".to_string(),
                "--max-serving-pairs".to_string(),
                "4".to_string(),
            ],
            &mut output,
        )
        .unwrap();
        String::from_utf8(output).unwrap()
    }

    fn assert_cli_config_error_contains(
        case_name: &str,
        cluster: &str,
        workload: &str,
        expected_error: &str,
    ) {
        let safe_case_name = case_name.replace(|ch: char| !ch.is_ascii_alphanumeric(), "-");
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cluster_path = std::env::temp_dir().join(format!(
            "inference-sim-{safe_case_name}-{nanos}.cluster.toml"
        ));
        let workload_path = std::env::temp_dir().join(format!(
            "inference-sim-{safe_case_name}-{nanos}.workload.toml"
        ));
        fs::write(&cluster_path, cluster).unwrap();
        fs::write(&workload_path, workload).unwrap();

        let mut output = Vec::new();
        let err = run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap_err();

        let _ = fs::remove_file(&cluster_path);
        let _ = fs::remove_file(&workload_path);
        assert!(
            err.to_string().contains(expected_error),
            "{case_name} expected error containing '{expected_error}', got '{err}'"
        );
        assert!(
            output.is_empty(),
            "{case_name} should fail before writing solver output"
        );
    }

    fn run_cli_json_paths_twice(
        case_name: &str,
        cluster_path: &std::path::Path,
        workload_path: &std::path::Path,
    ) -> serde_json::Value {
        let first = run_cli_json_paths(cluster_path, workload_path);
        let second = run_cli_json_paths(cluster_path, workload_path);
        assert_eq!(
            first, second,
            "{case_name} CLI JSON should be deterministic except runtime_elapsed_ms"
        );
        first
    }

    fn run_temp_cli_json_twice(
        case_name: &str,
        cluster: &str,
        workload: &str,
    ) -> serde_json::Value {
        let safe_case_name = case_name.replace(|ch: char| !ch.is_ascii_alphanumeric(), "-");
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cluster_path = std::env::temp_dir().join(format!(
            "inference-sim-{safe_case_name}-{nanos}.cluster.toml"
        ));
        let workload_path = std::env::temp_dir().join(format!(
            "inference-sim-{safe_case_name}-{nanos}.workload.toml"
        ));
        fs::write(&cluster_path, cluster).unwrap();
        fs::write(&workload_path, workload).unwrap();

        let first = run_cli_json_paths(&cluster_path, &workload_path);
        let second = run_cli_json_paths(&cluster_path, &workload_path);

        let _ = fs::remove_file(&cluster_path);
        let _ = fs::remove_file(&workload_path);

        assert_eq!(
            first, second,
            "{case_name} CLI JSON should be deterministic except runtime_elapsed_ms"
        );
        first
    }

    fn run_temp_request_metrics_csv(case_name: &str, cluster: &str, workload: &str) -> String {
        let safe_case_name = case_name.replace(|ch: char| !ch.is_ascii_alphanumeric(), "-");
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cluster_path = std::env::temp_dir().join(format!(
            "inference-sim-{safe_case_name}-{nanos}.cluster.toml"
        ));
        let workload_path = std::env::temp_dir().join(format!(
            "inference-sim-{safe_case_name}-{nanos}.workload.toml"
        ));
        let request_metrics_path = std::env::temp_dir().join(format!(
            "inference-sim-{safe_case_name}-{nanos}.request-metrics.csv"
        ));
        fs::write(&cluster_path, cluster).unwrap();
        fs::write(&workload_path, workload).unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--request-metrics-csv".to_string(),
                request_metrics_path.display().to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                "0".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
                "--max-serving-pairs".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();
        let csv = fs::read_to_string(&request_metrics_path).unwrap();

        let _ = fs::remove_file(&cluster_path);
        let _ = fs::remove_file(&workload_path);
        let _ = fs::remove_file(&request_metrics_path);

        csv
    }

    fn run_cli_json_paths(
        cluster_path: &std::path::Path,
        workload_path: &std::path::Path,
    ) -> serde_json::Value {
        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                "0".to_string(),
                "--max-candidates".to_string(),
                "1".to_string(),
                "--max-serving-pairs".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();
        let mut parsed: serde_json::Value = serde_json::from_slice(&output).unwrap();
        normalize_runtime_elapsed_ms(&mut parsed);
        parsed
    }

    fn h100_single_node_cluster() -> String {
        r#"
        schema_version = 1

        [cluster]
        preset = "h100_sxm"
        node_count = 1

        [interconnect]
        kind = "ib"
        variant = "ndr"
        "#
        .to_string()
    }

    fn h100_two_node_cluster() -> String {
        r#"
        schema_version = 1

        [cluster]
        preset = "h100_sxm"
        node_count = 2

        [interconnect]
        kind = "ib"
        variant = "ndr"
        "#
        .to_string()
    }

    fn minimal_parallelism_workload(dtype: &str, extra: &str) -> String {
        format!(
            r#"
            schema_version = 1

            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "{dtype}"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            {extra}
            "#
        )
    }

    fn minimal_serving_workload(dtype: &str, serving: &str) -> String {
        format!(
            r#"
            schema_version = 1

            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "{dtype}"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            {serving}
            "#
        )
    }

    fn control_plane_serving_workload(traffic: &str) -> String {
        minimal_serving_workload(
            "bf16",
            &format!(
                r#"
                [serving]
                mode = "colocated"
                prefill_nodes = [0]
                decode_nodes = [0]

                [serving.traffic]
                {traffic}
                "#
            ),
        )
    }

    fn edge_case_serving_workload(traffic: &str) -> String {
        minimal_serving_workload(
            "bf16",
            &format!(
                r#"
                [serving]
                mode = "colocated"
                prefill_nodes = [0]
                decode_nodes = [0]

                [serving.traffic]
                routing_policy = "round_robin"
                prefill_batching = "independent"
                decode_batching = "independent"

                {traffic}
                "#
            ),
        )
    }

    fn arrival_ms_values(json: &serde_json::Value) -> Vec<f64> {
        request_observations(json)
            .iter()
            .map(|observation| json_number(&observation["arrival_ms"]))
            .collect()
    }

    fn assert_arrivals_ms(json: &serde_json::Value, expected: &[f64]) {
        let actual = arrival_ms_values(json);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - *expected).abs() < 1e-9,
                "expected arrival {expected} ms, got {actual} ms"
            );
        }
    }

    fn assert_monotonic_arrivals(arrivals: &[f64]) {
        assert!(
            arrivals
                .windows(2)
                .all(|window| window[1] + 1e-12 >= window[0]),
            "arrivals should be monotonic: {arrivals:?}"
        );
    }

    fn json_number(value: &serde_json::Value) -> f64 {
        value
            .as_f64()
            .unwrap_or_else(|| panic!("expected JSON number, got {value}"))
    }

    fn request_observations(json: &serde_json::Value) -> &[serde_json::Value] {
        json["results"][0]["request_observations"]
            .as_array()
            .unwrap()
    }

    fn observation_by_request_id<'a>(
        observations: &'a [serde_json::Value],
        request_id: &str,
    ) -> &'a serde_json::Value {
        observations
            .iter()
            .find(|observation| observation["request_id"].as_str() == Some(request_id))
            .unwrap_or_else(|| panic!("missing request observation for {request_id}"))
    }

    fn metric_breakdown_by_group_key<'a>(
        json: &'a serde_json::Value,
        group: &str,
        key: &str,
    ) -> &'a serde_json::Value {
        json["results"][0]["metric_breakdowns"]
            .as_array()
            .expect("metric_breakdowns array")
            .iter()
            .find(|breakdown| {
                breakdown["group"].as_str() == Some(group) && breakdown["key"].as_str() == Some(key)
            })
            .unwrap_or_else(|| panic!("missing metric breakdown group={group} key={key}"))
    }

    fn assert_metric_count(json: &serde_json::Value, field: &str, expected: u64) {
        assert_eq!(
            json["results"][0]["metrics"][field].as_u64(),
            Some(expected),
            "unexpected metrics.{field}"
        );
    }

    fn assert_csv_field(csv: &str, field: &str, expected: &str) {
        let mut lines = csv.lines();
        let header = lines.next().expect("CSV header");
        let row = lines.next().expect("CSV data row");
        let headers = parse_test_csv_record(header);
        let values = parse_test_csv_record(row);
        let field_idx = headers
            .iter()
            .position(|name| name == field)
            .unwrap_or_else(|| panic!("missing CSV field {field}"));
        let actual = values
            .get(field_idx)
            .unwrap_or_else(|| panic!("missing CSV value for {field}"));
        assert_eq!(actual, expected, "unexpected CSV {field}");
    }

    fn assert_csv_field_nonempty(csv: &str, field: &str) {
        let mut lines = csv.lines();
        let header = lines.next().expect("CSV header");
        let row = lines.next().expect("CSV data row");
        let headers = parse_test_csv_record(header);
        let values = parse_test_csv_record(row);
        let field_idx = headers
            .iter()
            .position(|name| name == field)
            .unwrap_or_else(|| panic!("missing CSV field {field}"));
        let actual = values
            .get(field_idx)
            .unwrap_or_else(|| panic!("missing CSV value for {field}"));
        assert!(!actual.is_empty(), "expected nonempty CSV {field}");
    }

    fn assert_csv_row_field(
        csv: &str,
        selector_field: &str,
        selector_value: &str,
        field: &str,
        expected: &str,
    ) {
        let mut lines = csv.lines();
        let header = lines.next().expect("CSV header");
        let headers = parse_test_csv_record(header);
        let selector_idx = headers
            .iter()
            .position(|name| name == selector_field)
            .unwrap_or_else(|| panic!("missing CSV selector field {selector_field}"));
        let field_idx = headers
            .iter()
            .position(|name| name == field)
            .unwrap_or_else(|| panic!("missing CSV field {field}"));
        for row in lines {
            let values = parse_test_csv_record(row);
            if values
                .get(selector_idx)
                .is_some_and(|value| value == selector_value)
            {
                let actual = values
                    .get(field_idx)
                    .unwrap_or_else(|| panic!("missing CSV value for {field}"));
                assert_eq!(actual, expected, "unexpected CSV {field}");
                return;
            }
        }
        panic!("missing CSV row where {selector_field}={selector_value}");
    }

    fn parse_test_csv_record(row: &str) -> Vec<String> {
        let mut fields = Vec::new();
        let mut field = String::new();
        let mut chars = row.chars().peekable();
        let mut in_quotes = false;
        while let Some(ch) = chars.next() {
            match ch {
                '"' if in_quotes && chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => {
                    fields.push(std::mem::take(&mut field));
                }
                _ => field.push(ch),
            }
        }
        fields.push(field);
        fields
    }

    fn assert_measurement_window_counts(
        json: &serde_json::Value,
        request_count: u64,
        completed_request_count: u64,
        failed_request_count: u64,
        rejected_request_count: u64,
        timed_out_request_count: u64,
        cancelled_request_count: u64,
    ) {
        let window = &json["results"][0]["measurement_window"];
        assert_eq!(window["request_count"].as_u64(), Some(request_count));
        assert_eq!(
            window["completed_request_count"].as_u64(),
            Some(completed_request_count)
        );
        assert_eq!(
            window["failed_request_count"].as_u64(),
            Some(failed_request_count)
        );
        assert_eq!(
            window["rejected_request_count"].as_u64(),
            Some(rejected_request_count)
        );
        assert_eq!(
            window["timed_out_request_count"].as_u64(),
            Some(timed_out_request_count)
        );
        assert_eq!(
            window["cancelled_request_count"].as_u64(),
            Some(cancelled_request_count)
        );
        assert_eq!(
            window["measured_requests"].as_u64(),
            Some(completed_request_count)
        );
    }

    fn assert_request_status(observation: &serde_json::Value, expected: &str) {
        assert_eq!(observation["status"].as_str(), Some(expected));
    }

    fn assert_lifecycle_event(observation: &serde_json::Value, expected: &str) {
        assert!(
            observation["lifecycle_events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event["event"].as_str() == Some(expected)),
            "missing lifecycle event {expected}"
        );
    }

    fn assert_rejection_code(
        observation: &serde_json::Value,
        expected_phase: &str,
        expected_code: &str,
    ) {
        let rejection = &observation["rejection"];
        assert_eq!(rejection["phase"].as_str(), Some(expected_phase));
        assert_eq!(rejection["code"].as_str(), Some(expected_code));
    }

    fn assert_request_metric_derivation(
        observation: &serde_json::Value,
        completed: bool,
        output_tokens: u64,
    ) {
        let derivation = &observation["metric_derivation"];
        let status = observation["status"].as_str().unwrap();
        assert_eq!(derivation["completed"].as_bool(), Some(completed));
        assert_eq!(derivation["terminal_event"].as_str(), Some(status));
        assert_eq!(derivation["output_tokens"].as_u64(), Some(output_tokens));
        assert_eq!(
            derivation["included_in_measurement_window"].as_bool(),
            Some(completed)
        );
        if completed {
            assert!(derivation["metric_unavailable_reason"].is_null());
            assert_eq!(derivation["ttft_start_event"].as_str(), Some("arrived"));
            assert_eq!(
                derivation["ttft_end_event"].as_str(),
                Some("decode_iteration_finished:first")
            );
            assert_eq!(derivation["e2el_start_event"].as_str(), Some("arrived"));
            assert_eq!(
                derivation["e2el_end_event"].as_str(),
                Some("decode_iteration_finished:last")
            );
            assert_eq!(
                derivation["throughput_duration_start_event"].as_str(),
                Some("arrived")
            );
            assert_eq!(
                derivation["throughput_duration_end_event"].as_str(),
                Some("decode_iteration_finished:last")
            );
        } else {
            let expected_reason = format!("request_not_completed:{status}");
            assert_eq!(
                derivation["metric_unavailable_reason"].as_str(),
                Some(expected_reason.as_str())
            );
            assert_eq!(derivation["e2el_start_event"].as_str(), Some("arrived"));
            assert_eq!(derivation["e2el_end_event"].as_str(), Some(status));
            assert!(derivation["throughput_duration_start_event"].is_null());
            assert!(derivation["throughput_duration_end_event"].is_null());
        }
        assert!(derivation["metric_source"].as_str().is_some());
        assert!(derivation["measurement_window_source"].as_str().is_some());
        assert!(derivation["arrival_ms"].is_number());
    }

    fn run_example_json(
        cluster_path: &std::path::Path,
        workload_path: &std::path::Path,
        request_limit: &str,
        max_candidates: &str,
        max_serving_pairs: &str,
    ) -> String {
        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--request-limit".to_string(),
                request_limit.to_string(),
                "--max-candidates".to_string(),
                max_candidates.to_string(),
                "--max-serving-pairs".to_string(),
                max_serving_pairs.to_string(),
            ],
            &mut output,
        )
        .unwrap();
        String::from_utf8(output).unwrap()
    }

    fn normalize_runtime_elapsed_ms(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(fields) => {
                if fields.contains_key("runtime_elapsed_ms") {
                    fields.insert("runtime_elapsed_ms".to_string(), serde_json::json!(0));
                }
                for child in fields.values_mut() {
                    normalize_runtime_elapsed_ms(child);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    normalize_runtime_elapsed_ms(child);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn emits_candidate_id_and_remediation_in_serving_rejections() {
        let rejection = ServingRejection {
            phase: "decode".to_string(),
            category: "capacity".to_string(),
            resource: "resident_tokens".to_string(),
            code: "kv_residency_capacity_exceeded".to_string(),
            observed: Some(128.0),
            limit: Some(64.0),
            unit: Some("tokens".to_string()),
            remediation: Some("add decode workers".to_string()),
            message: "KV residency capacity exceeded".to_string(),
        };
        let mut output = Vec::new();
        write_serving_rejections(&mut output, "", "candidate-1", &[rejection], false).unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"candidate_id\": \"candidate-1\""));
        assert!(output.contains("\"remediation\": \"add decode workers\""));
        assert!(output.contains("\"code\": \"kv_residency_capacity_exceeded\""));
    }

    #[test]
    fn groups_serving_rejections_by_root_cause() {
        let capacity_a = ServingRejection {
            phase: "decode".to_string(),
            category: "capacity".to_string(),
            resource: "resident_tokens".to_string(),
            code: "kv_residency_capacity_exceeded".to_string(),
            observed: Some(128.0),
            limit: Some(64.0),
            unit: Some("tokens".to_string()),
            remediation: Some("add decode workers".to_string()),
            message: "KV residency capacity exceeded".to_string(),
        };
        let mut capacity_b = capacity_a.clone();
        capacity_b.observed = Some(256.0);
        let route = ServingRejection {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "route".to_string(),
            code: "missing_kv_route".to_string(),
            observed: None,
            limit: None,
            unit: None,
            remediation: Some("connect prefill and decode pools".to_string()),
            message: "No routable KV handoff path".to_string(),
        };
        let candidate_a = vec![capacity_a, route];
        let candidate_b = vec![capacity_b];
        let candidate_c = Vec::new();

        let summary = serving_rejection_summary_from_records([
            ("candidate-a", candidate_a.as_slice()),
            ("candidate-b", candidate_b.as_slice()),
            ("candidate-c", candidate_c.as_slice()),
        ]);

        assert_eq!(summary.candidate_with_rejections_count, 2);
        assert_eq!(summary.total_rejection_count, 3);
        assert_eq!(summary.groups.len(), 2);
        assert_eq!(summary.groups[0].key.code, "kv_residency_capacity_exceeded");
        assert_eq!(summary.groups[0].candidate_count, 2);
        assert_eq!(summary.groups[0].rejection_count, 2);
        assert_eq!(
            summary.groups[0].example_candidate_ids,
            vec!["candidate-a".to_string(), "candidate-b".to_string()]
        );
        assert_eq!(summary.groups[1].key.code, "missing_kv_route");
        assert_eq!(summary.groups[1].candidate_count, 1);
        assert_eq!(summary.groups[1].rejection_count, 1);
    }

    #[test]
    fn approximation_policy_can_reject_serving_candidate() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-approx-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-approx-workload-{nanos}.toml"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 2

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 4
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [approximation_policy]
            preset = "topology_sensitive"

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 1
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--format".to_string(),
                "json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"preset\": \"topology_sensitive\""));
        assert!(output.contains("\"status\": \"reject\""));
        assert!(output.contains("\"feasible\": false"));
        assert!(output.contains("\"approximation_summary\""));
        assert!(output.contains("\"status\": \"policy_rejected\""));
        assert!(output.contains("\"policy_violation_count\": 1"));
        assert!(output.contains("\"approximation_policy_violations\""));
        assert!(output.contains("\"code\": \"node_set_kv_handoff\""));
        assert!(output.contains("\"action\": \"reject\""));
        assert!(output.contains("approximation_policy_reject_node_set_kv_handoff"));

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--format".to_string(),
                "json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
                "--drop-rejected-candidates".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"searched_serving_pairs\": 1"));
        assert!(output.contains("\"reported_serving_pairs\": 0"));
        assert!(output.contains("\"omitted_rejected_serving_pairs\": 1"));
        assert!(output.contains("\"retain_rejected_candidates\": false"));
        assert!(!output.contains("\"candidate_id\":"));
        assert!(!output.contains("approximation_policy_reject_node_set_kv_handoff"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn approximation_metric_gate_rejects_only_matching_serving_objective() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let cluster_path = dir.join(format!("inference-sim-metric-gate-cluster-{nanos}.toml"));
        let workload_path = dir.join(format!("inference-sim-metric-gate-workload-{nanos}.toml"));

        fs::write(
            &cluster_path,
            r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 2

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
        )
        .unwrap();
        fs::write(
            &workload_path,
            r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 4
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [approximation_policy]
            default = "warn"

            [[approximation_policy.metric_gates]]
            metric = "e2el"
            reject_categories = ["topology"]

            [serving]
            mode = "disaggregated"
            objective = "e2el"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 1
            "#,
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--format".to_string(),
                "json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"metric_gates\""));
        assert!(output.contains("\"metrics\": [\"e2el\"]"));
        assert!(output.contains("\"feasible\": false"));
        assert!(output.contains("\"metric\": \"e2el\""));
        assert!(output.contains("while evaluating metric 'e2el'"));
        assert!(output.contains("\"code\": \"node_set_kv_handoff\""));

        fs::write(
            &workload_path,
            fs::read_to_string(&workload_path)
                .unwrap()
                .replace("objective = \"e2el\"", "objective = \"tpot\""),
        )
        .unwrap();

        let mut output = Vec::new();
        run_with_args(
            [
                "inference-sim".to_string(),
                "--cluster".to_string(),
                cluster_path.display().to_string(),
                "--workload".to_string(),
                workload_path.display().to_string(),
                "--format".to_string(),
                "json".to_string(),
                "--top-k".to_string(),
                "1".to_string(),
            ],
            &mut output,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"objective\": \"minimize_tpot\""));
        assert!(output.contains("\"feasible\": true"));
        assert!(output.contains("\"approximation_policy_violations\": ["));
        assert!(!output.contains("while evaluating metric 'tpot'"));

        let _ = fs::remove_file(cluster_path);
        let _ = fs::remove_file(workload_path);
    }

    #[test]
    fn emits_calibration_profile_shape_and_benchmarks() {
        let profile = CalibrationProfileMetadata {
            path: "profile.toml".to_string(),
            name: Some("bench-profile".to_string()),
            hardware: Some("h100+a100".to_string()),
            fabric: Some("ndr".to_string()),
            model: Some("test-model".to_string()),
            dtype: Some("bf16".to_string()),
            serving_stack: Some("test-stack".to_string()),
            serving_runtime_features: vec!["paged_attention".to_string()],
            backend_version: Some("test-stack 1.2.3".to_string()),
            driver_version: Some("550.54.15".to_string()),
            cuda_version: Some("12.4".to_string()),
            rocm_version: None,
            nccl_version: Some("2.21.5".to_string()),
            rccl_version: None,
            ucx_version: Some("1.16.0".to_string()),
            kernel_settings: vec!["cuda_graphs=true".to_string(), "block_size=16".to_string()],
            environment_hash: Some("sha256:unit-test".to_string()),
            source: Some("unit-test".to_string()),
            date: Some("2026-05-25".to_string()),
            notes: None,
            valid_shape: Some(CalibrationShapeRange {
                min_batch_size: Some(1),
                max_batch_size: Some(8),
                min_prompt_tokens: Some(128),
                max_prompt_tokens: Some(4096),
                min_decode_tokens: Some(1),
                max_decode_tokens: Some(128),
                min_sequence_tokens: Some(512),
                max_sequence_tokens: Some(8192),
            }),
            invalid_shapes: vec![CalibrationInvalidShapeRange {
                name: Some("uncalibrated-long-context".to_string()),
                reason: Some("no benchmark coverage beyond 16k sequence tokens".to_string()),
                shape: CalibrationShapeRange {
                    min_batch_size: None,
                    max_batch_size: None,
                    min_prompt_tokens: None,
                    max_prompt_tokens: None,
                    min_decode_tokens: None,
                    max_decode_tokens: None,
                    min_sequence_tokens: Some(16385),
                    max_sequence_tokens: None,
                },
            }],
            fits: vec![CalibrationFittedModel {
                name: Some("decode-latency-fit".to_string()),
                target: "decode_ms".to_string(),
                phase: Some("decode".to_string()),
                kind: Some("serving".to_string()),
                model: "linear".to_string(),
                unit: Some("ms".to_string()),
                intercept: Some(1.25),
                features: vec![
                    "batch_size".to_string(),
                    "decode_tokens".to_string(),
                    "sequence_tokens".to_string(),
                ],
                coefficients: vec![0.5, 3.1, 0.002],
                feature_ranges: vec![
                    CalibrationFitFeatureRange {
                        feature: "batch_size".to_string(),
                        min: Some(1.0),
                        max: Some(8.0),
                    },
                    CalibrationFitFeatureRange {
                        feature: "decode_tokens".to_string(),
                        min: Some(1.0),
                        max: Some(128.0),
                    },
                    CalibrationFitFeatureRange {
                        feature: "sequence_tokens".to_string(),
                        min: Some(512.0),
                        max: Some(8192.0),
                    },
                ],
                r_squared: Some(0.98),
                adjusted_r_squared: Some(0.97),
                rmse: Some(2.5),
                rmse_pct: Some(3.0),
                mean_abs_pct_error: Some(2.0),
                max_abs_pct_error: Some(7.5),
                validation_rmse: Some(3.5),
                validation_rmse_pct: Some(4.0),
                validation_mean_abs_pct_error: Some(3.0),
                validation_max_abs_pct_error: Some(9.5),
                confidence_interval: Some(4.25),
                confidence_interval_pct: Some(5.0),
                confidence_level: Some(0.95),
                sample_count: Some(24),
                validation_sample_count: Some(6),
                source: Some("unit-test".to_string()),
                notes: None,
            }],
            benchmarks: vec![CalibrationBenchmarkPoint {
                name: Some("decode-b4".to_string()),
                kind: Some("serving".to_string()),
                phase: Some("decode".to_string()),
                hardware: Some("a100".to_string()),
                fabric: Some("hdr".to_string()),
                model: Some("test-model".to_string()),
                dtype: Some("bf16".to_string()),
                batch_size: Some(4),
                prompt_tokens: Some(1024),
                decode_tokens: Some(32),
                sequence_tokens: Some(2048),
                tensor_ranks: Some(4),
                pipeline_ranks: Some(1),
                expert_ranks: Some(1),
                data_ranks: Some(1),
                measured_ms: Some(12.5),
                predicted_ms: Some(13.0),
                throughput_tokens_per_s: Some(256.0),
                command: Some("bench decode".to_string()),
                source: Some("unit-test".to_string()),
                notes: None,
            }],
        };
        let warnings = vec![CalibrationApplicabilityWarning {
            field: "prompt_tokens".to_string(),
            observed_min: 128,
            observed_max: 8192,
            calibrated_min: Some(128),
            calibrated_max: Some(4096),
            message:
                "workload prompt_tokens range 128..8192 falls outside calibration range 128..4096"
                    .to_string(),
        }];
        let coverage = CalibrationCoverageReport {
            benchmark_count: 1,
            shape_benchmark_count: 1,
            complete_shape_benchmark_count: 1,
            required_phases: vec!["prefill".to_string(), "decode".to_string()],
            covered_phases: vec!["decode".to_string()],
            missing_phases: vec!["prefill".to_string()],
            batch_size_score: Some(0.0),
            prompt_tokens_score: Some(1.0),
            decode_tokens_score: Some(1.0),
            sequence_tokens_score: Some(1.0),
            shape_coverage_score: Some(0.75),
            phase_coverage_score: Some(0.5),
            coverage_score: Some(0.625),
            nearest_benchmark: Some("decode-b4".to_string()),
            nearest_benchmark_distance: Some(0.0),
            status: "partial".to_string(),
        };
        let mut output = Vec::new();
        write_calibration(
            &mut output,
            CalibrationJsonContext {
                calibration: SimulationCalibration::default(),
                policy: &CalibrationPolicy::default(),
                profile: Some(&profile),
                coverage: Some(&coverage),
                warnings: &warnings,
                invalid_shape_warnings: &[],
                gate_violations: &[],
            },
            "",
            false,
        )
        .unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"valid_shape\""));
        assert!(output.contains("\"serving_runtime_features\": [\"paged_attention\"]"));
        assert!(output.contains("\"backend_version\": \"test-stack 1.2.3\""));
        assert!(output.contains("\"driver_version\": \"550.54.15\""));
        assert!(output.contains("\"cuda_version\": \"12.4\""));
        assert!(output.contains("\"nccl_version\": \"2.21.5\""));
        assert!(output.contains("\"ucx_version\": \"1.16.0\""));
        assert!(output.contains("\"kernel_settings\": [\"cuda_graphs=true\", \"block_size=16\"]"));
        assert!(output.contains("\"environment_hash\": \"sha256:unit-test\""));
        assert!(output.contains("\"max_sequence_tokens\": 8192"));
        assert!(output.contains("\"invalid_shapes\""));
        assert!(output.contains("\"uncalibrated-long-context\""));
        assert!(output.contains("\"fits\""));
        assert!(output.contains("\"feature_ranges\""));
        assert!(output.contains("\"target\": \"decode_ms\""));
        assert!(
            output
                .contains("\"features\": [\"batch_size\", \"decode_tokens\", \"sequence_tokens\"]")
        );
        assert!(output.contains("\"coefficients\": [0.500000, 3.100000, 0.002000]"));
        assert!(output.contains("\"r_squared\": 0.980000"));
        assert!(output.contains("\"validation_rmse\": 3.500000"));
        assert!(output.contains("\"validation_rmse_pct\": 4.000000"));
        assert!(output.contains("\"validation_mean_abs_pct_error\": 3.000000"));
        assert!(output.contains("\"validation_max_abs_pct_error\": 9.500000"));
        assert!(output.contains("\"confidence_interval\": 4.250000"));
        assert!(output.contains("\"confidence_interval_pct\": 5.000000"));
        assert!(output.contains("\"confidence_level\": 0.950000"));
        assert!(output.contains("\"sample_count\": 24"));
        assert!(output.contains("\"benchmark_summary\""));
        assert!(output.contains("\"latency_comparison_count\": 1"));
        assert!(output.contains("\"mean_abs_pct_error\": 4.000000"));
        assert!(output.contains("\"max_abs_pct_error\": 4.000000"));
        assert!(output.contains("\"rmse_pct_error\": 4.000000"));
        assert!(output.contains("\"mean_signed_pct_error\": 4.000000"));
        assert!(output.contains("\"within_10_percent_count\": 1"));
        assert!(output.contains("\"within_20_percent_count\": 1"));
        assert!(output.contains("\"worst_benchmark\": \"decode-b4\""));
        assert!(output.contains("\"status\": \"good\""));
        assert!(output.contains("\"coverage\""));
        assert!(output.contains("\"coverage_score\": 0.625000"));
        assert!(output.contains("\"nearest_benchmark\": \"decode-b4\""));
        assert!(output.contains("\"missing_phases\": [\"prefill\"]"));
        assert!(output.contains("\"policy\""));
        assert!(output.contains("\"valid_shape\": \"warn\""));
        assert!(output.contains("\"fit_confidence\": \"warn\""));
        assert!(output.contains("\"fit_extrapolation\": \"warn\""));
        assert!(output.contains("\"fit_partially_bounded\": \"warn\""));
        assert!(output.contains("\"fit_unbounded\": \"warn\""));
        assert!(output.contains("\"fit_sample_count\": \"warn\""));
        assert!(output.contains("\"fit_validation_sample_count\": \"warn\""));
        assert!(output.contains("\"fit_source\": \"warn\""));
        assert!(output.contains("\"fit_uncertainty\": \"warn\""));
        assert!(output.contains("\"profile_source\": \"warn\""));
        assert!(output.contains("\"profile_date\": \"warn\""));
        assert!(output.contains("\"profile_runtime\": \"warn\""));
        assert!(output.contains("\"min_fit_confidence_score\": 0.500000"));
        assert!(output.contains("\"min_fit_confidence_level\": null"));
        assert!(output.contains("\"min_fit_sample_count\": null"));
        assert!(output.contains("\"min_fit_validation_sample_count\": null"));
        assert!(output.contains("\"max_fit_relative_uncertainty_pct\": null"));
        assert!(output.contains("\"max_fit_absolute_uncertainty_ms\": null"));
        assert!(output.contains("\"min_serving_phase_coverage_fraction\": null"));
        assert!(output.contains("\"uncertainty_ranking_weight\": 0.000000"));
        assert!(output.contains("\"gate_violations\""));
        assert!(output.contains("\"benchmarks\""));
        assert!(output.contains("\"name\": \"decode-b4\""));
        assert!(output.contains("\"measured_ms\": 12.500000"));
        assert!(output.contains("\"latency_error_ms\": 0.500000"));
        assert!(output.contains("\"latency_abs_error_ms\": 0.500000"));
        assert!(output.contains("\"latency_signed_pct_error\": 4.000000"));
        assert!(output.contains("\"latency_abs_pct_error\": 4.000000"));
        assert!(output.contains("\"latency_residual_status\": \"good\""));
        assert!(output.contains("\"throughput_tokens_per_s\": 256.000000"));
        assert!(output.contains("\"applicability_status\": \"outside_valid_shape\""));
        assert!(output.contains("\"applicability_warnings\""));
        assert!(output.contains("\"field\": \"prompt_tokens\""));
        assert!(output.contains("\"observed_max\": 8192"));
        assert!(output.contains("\"calibrated_max\": 4096"));
    }

    #[test]
    fn calibration_fit_policy_flags_low_confidence_and_extrapolated_fits() {
        let policy = CalibrationPolicy {
            fit_confidence: CalibrationGateMode::Reject,
            fit_extrapolation: CalibrationGateMode::Reject,
            fit_sample_count: CalibrationGateMode::Reject,
            fit_validation_sample_count: CalibrationGateMode::Warn,
            fit_uncertainty: CalibrationGateMode::Reject,
            min_fit_confidence_score: Some(0.75),
            min_fit_confidence_level: Some(0.95),
            min_fit_sample_count: Some(32),
            min_fit_validation_sample_count: Some(8),
            max_fit_relative_uncertainty_pct: Some(5.0),
            max_fit_absolute_uncertainty_s: Some(0.0005),
            ..CalibrationPolicy::default()
        };
        let application = CalibrationFitApplication {
            phase: "decode".to_string(),
            target: "decode_ms".to_string(),
            fit_name: Some("decode-fit".to_string()),
            model: "linear".to_string(),
            unit: Some("ms".to_string()),
            intercept: 1.0,
            raw_prediction: 10.0,
            prediction_kind: "latency".to_string(),
            predicted_value: 0.010,
            prediction_unit: Some("s".to_string()),
            predicted_s: 0.010,
            baseline_value: Some(0.012),
            baseline_s: Some(0.012),
            applicability_status: "extrapolated".to_string(),
            confidence_score: 0.40,
            max_extrapolation_ratio: 1.5,
            relative_uncertainty_pct: Some(10.0),
            absolute_uncertainty_value: Some(0.001),
            absolute_uncertainty_s: Some(0.001),
            uncertainty_source: Some("rmse_pct".to_string()),
            validation_rmse: None,
            validation_rmse_pct: None,
            validation_mean_abs_pct_error: None,
            validation_max_abs_pct_error: None,
            confidence_interval: None,
            confidence_interval_pct: None,
            confidence_level: Some(0.80),
            sample_count: Some(24),
            validation_sample_count: Some(6),
            source: Some("unit-test".to_string()),
            features: Vec::new(),
        };

        let violations = calibration_fit_gate_violations(&policy, &[application]);

        assert_eq!(violations.len(), 7);
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_confidence_below_min"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(0.40)
                && violation.limit == Some(0.75)
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_confidence_level_below_min"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(0.80)
                && violation.limit == Some(0.95)
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_extrapolated"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(1.5)
                && violation.limit == Some(0.0)
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_sample_count_below_min"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(24.0)
                && violation.limit == Some(32.0)
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_validation_sample_count_below_min"
                && violation.action == CalibrationGateMode::Warn
                && violation.observed == Some(6.0)
                && violation.limit == Some(8.0)
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_relative_uncertainty_above_max"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(10.0)
                && violation.limit == Some(5.0)
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_absolute_uncertainty_above_max"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(1.0)
                && violation.limit == Some(0.5)
        }));
    }

    #[test]
    fn calibration_fit_policy_flags_unbounded_and_partially_bounded_fits() {
        let policy = CalibrationPolicy {
            fit_partially_bounded: CalibrationGateMode::Warn,
            fit_unbounded: CalibrationGateMode::Reject,
            ..CalibrationPolicy::default()
        };
        let partially_bounded = CalibrationFitApplication {
            phase: "prefill".to_string(),
            target: "prefill_ms".to_string(),
            fit_name: Some("prefill-fit".to_string()),
            model: "linear".to_string(),
            unit: Some("ms".to_string()),
            intercept: 1.0,
            raw_prediction: 10.0,
            prediction_kind: "latency".to_string(),
            predicted_value: 0.010,
            prediction_unit: Some("s".to_string()),
            predicted_s: 0.010,
            baseline_value: Some(0.012),
            baseline_s: Some(0.012),
            applicability_status: "partially_bounded".to_string(),
            confidence_score: 0.95,
            max_extrapolation_ratio: 0.0,
            relative_uncertainty_pct: Some(2.0),
            absolute_uncertainty_value: Some(0.0002),
            absolute_uncertainty_s: Some(0.0002),
            uncertainty_source: Some("confidence_interval".to_string()),
            validation_rmse: None,
            validation_rmse_pct: None,
            validation_mean_abs_pct_error: None,
            validation_max_abs_pct_error: None,
            confidence_interval: Some(0.2),
            confidence_interval_pct: Some(2.0),
            confidence_level: Some(0.95),
            sample_count: Some(64),
            validation_sample_count: Some(16),
            source: Some("unit-test".to_string()),
            features: Vec::new(),
        };
        let mut unbounded = partially_bounded.clone();
        unbounded.target = "decode_ms".to_string();
        unbounded.fit_name = Some("decode-fit".to_string());
        unbounded.applicability_status = "unbounded".to_string();

        let violations = calibration_fit_gate_violations(&policy, &[partially_bounded, unbounded]);

        assert_eq!(violations.len(), 2);
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_partially_bounded"
                && violation.action == CalibrationGateMode::Warn
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_unbounded" && violation.action == CalibrationGateMode::Reject
        }));
    }

    #[test]
    fn calibration_uncertainty_summary_counts_value_fit_uncertainty() {
        let application = CalibrationFitApplication {
            phase: "serving".to_string(),
            target: "throughput_tokens_per_s".to_string(),
            fit_name: Some("throughput-fit".to_string()),
            model: "linear".to_string(),
            unit: Some("tokens/s".to_string()),
            intercept: 1.0,
            raw_prediction: 1000.0,
            prediction_kind: "throughput".to_string(),
            predicted_value: 1000.0,
            prediction_unit: Some("tokens/s".to_string()),
            predicted_s: 0.0,
            baseline_value: Some(900.0),
            baseline_s: None,
            applicability_status: "interpolated".to_string(),
            confidence_score: 0.98,
            max_extrapolation_ratio: 0.0,
            relative_uncertainty_pct: Some(7.0),
            absolute_uncertainty_value: Some(70.0),
            absolute_uncertainty_s: None,
            uncertainty_source: Some("confidence_interval_pct".to_string()),
            validation_rmse: None,
            validation_rmse_pct: Some(4.0),
            validation_mean_abs_pct_error: None,
            validation_max_abs_pct_error: None,
            confidence_interval: None,
            confidence_interval_pct: Some(7.0),
            confidence_level: Some(0.95),
            sample_count: Some(64),
            validation_sample_count: Some(16),
            source: Some("unit-test".to_string()),
            features: Vec::new(),
        };

        let summary = calibration_uncertainty_summary([&application].into_iter());

        assert_eq!(summary.fit_count, 1);
        assert_eq!(summary.fit_count_with_uncertainty, 1);
        assert_eq!(summary.relative_uncertainty_pct, Some(7.0));
        assert_eq!(summary.absolute_uncertainty_s, None);
        assert_eq!(summary.min_confidence_score, Some(0.98));
        assert_eq!(summary.applicability_status, "interpolated");
    }

    #[test]
    fn calibration_fit_policy_flags_missing_fit_provenance() {
        let policy = CalibrationPolicy {
            fit_source: CalibrationGateMode::Reject,
            fit_uncertainty: CalibrationGateMode::Warn,
            ..CalibrationPolicy::default()
        };
        let application = CalibrationFitApplication {
            phase: "prefill".to_string(),
            target: "prefill_ms".to_string(),
            fit_name: Some("prefill-fit".to_string()),
            model: "linear".to_string(),
            unit: Some("ms".to_string()),
            intercept: 1.0,
            raw_prediction: 10.0,
            prediction_kind: "latency".to_string(),
            predicted_value: 0.010,
            prediction_unit: Some("s".to_string()),
            predicted_s: 0.010,
            baseline_value: Some(0.012),
            baseline_s: Some(0.012),
            applicability_status: "interpolated".to_string(),
            confidence_score: 1.0,
            max_extrapolation_ratio: 0.0,
            relative_uncertainty_pct: None,
            absolute_uncertainty_value: None,
            absolute_uncertainty_s: None,
            uncertainty_source: None,
            validation_rmse: None,
            validation_rmse_pct: None,
            validation_mean_abs_pct_error: None,
            validation_max_abs_pct_error: None,
            confidence_interval: None,
            confidence_interval_pct: None,
            confidence_level: None,
            sample_count: Some(64),
            validation_sample_count: Some(16),
            source: Some("  ".to_string()),
            features: Vec::new(),
        };

        let violations = calibration_fit_gate_violations(&policy, &[application]);

        assert_eq!(violations.len(), 2);
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_source_unspecified"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(0.0)
                && violation.limit == Some(1.0)
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "fit_uncertainty_unspecified"
                && violation.action == CalibrationGateMode::Warn
                && violation.observed == Some(0.0)
                && violation.limit == Some(1.0)
        }));
    }

    #[test]
    fn calibration_phase_policy_flags_active_uncalibrated_serving_components() {
        let policy = CalibrationPolicy {
            coverage: CalibrationGateMode::Reject,
            require_phase_coverage: true,
            ..CalibrationPolicy::default()
        };
        let phases = vec![
            ServingPhaseCalibrationObservation {
                phase: "prefill".to_string(),
                active: true,
                calibrated: false,
                fit_count: 0,
                applied_targets: Vec::new(),
                estimated_s: 0.012,
                status: "uncalibrated_no_fit".to_string(),
            },
            ServingPhaseCalibrationObservation {
                phase: "decode".to_string(),
                active: true,
                calibrated: true,
                fit_count: 2,
                applied_targets: vec!["decode_ms".to_string()],
                estimated_s: 0.020,
                status: "calibrated".to_string(),
            },
            ServingPhaseCalibrationObservation {
                phase: "kv_transfer".to_string(),
                active: false,
                calibrated: false,
                fit_count: 0,
                applied_targets: Vec::new(),
                estimated_s: 0.0,
                status: "inactive".to_string(),
            },
            ServingPhaseCalibrationObservation {
                phase: "decode_queue".to_string(),
                active: true,
                calibrated: false,
                fit_count: 0,
                applied_targets: Vec::new(),
                estimated_s: 0.004,
                status: "uncalibrated_no_fit".to_string(),
            },
            ServingPhaseCalibrationObservation {
                phase: "prefill_queue".to_string(),
                active: true,
                calibrated: false,
                fit_count: 0,
                applied_targets: Vec::new(),
                estimated_s: 0.003,
                status: "uncalibrated_no_profile".to_string(),
            },
        ];

        let violations = calibration_phase_gate_violations(&policy, &phases);

        assert_eq!(violations.len(), 2);
        assert!(violations.iter().any(|violation| {
            violation.code == "active_serving_phase_coverage_missing"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(0.0)
                && violation.limit == Some(1.0)
                && violation.message.contains("prefill")
                && violation.message.contains("12.000 ms")
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "active_serving_phase_coverage_missing"
                && violation.action == CalibrationGateMode::Reject
                && violation.message.contains("decode_queue")
                && violation.message.contains("4.000 ms")
        }));
    }

    #[test]
    fn calibration_phase_policy_flags_low_serving_phase_coverage() {
        let policy = CalibrationPolicy {
            coverage: CalibrationGateMode::Reject,
            require_phase_coverage: false,
            min_serving_phase_coverage_fraction: Some(0.75),
            ..CalibrationPolicy::default()
        };
        let phases = vec![
            ServingPhaseCalibrationObservation {
                phase: "prefill".to_string(),
                active: true,
                calibrated: true,
                fit_count: 1,
                applied_targets: vec!["prefill_ms".to_string()],
                estimated_s: 0.012,
                status: "calibrated".to_string(),
            },
            ServingPhaseCalibrationObservation {
                phase: "decode".to_string(),
                active: true,
                calibrated: false,
                fit_count: 0,
                applied_targets: Vec::new(),
                estimated_s: 0.020,
                status: "uncalibrated_no_fit".to_string(),
            },
            ServingPhaseCalibrationObservation {
                phase: "kv_transfer".to_string(),
                active: false,
                calibrated: false,
                fit_count: 0,
                applied_targets: Vec::new(),
                estimated_s: 0.0,
                status: "inactive".to_string(),
            },
        ];

        let violations = calibration_phase_gate_violations(&policy, &phases);

        assert_eq!(violations.len(), 1);
        assert!(violations.iter().any(|violation| {
            violation.code == "serving_phase_coverage_below_min"
                && violation.action == CalibrationGateMode::Reject
                && violation.observed == Some(0.5)
                && violation.limit == Some(0.75)
                && violation
                    .message
                    .contains("1/2 active components calibrated")
        }));
    }

    #[test]
    fn calibration_phase_policy_can_be_disabled_for_serving_components() {
        let policy = CalibrationPolicy {
            coverage: CalibrationGateMode::Reject,
            require_phase_coverage: false,
            ..CalibrationPolicy::default()
        };
        let phases = vec![ServingPhaseCalibrationObservation {
            phase: "prefill".to_string(),
            active: true,
            calibrated: false,
            fit_count: 0,
            applied_targets: Vec::new(),
            estimated_s: 0.012,
            status: "uncalibrated_no_fit".to_string(),
        }];

        let violations = calibration_phase_gate_violations(&policy, &phases);

        assert!(violations.is_empty());
    }

    #[test]
    fn uncertainty_ranking_weight_prefers_conservative_parallelism_candidate() {
        let mut results = vec![
            parallelism_score_with_uncertainty(1, 0.010, 0.020),
            parallelism_score_with_uncertainty(2, 0.015, 0.0),
        ];
        let policy = CalibrationPolicy {
            uncertainty_ranking_weight: 1.0,
            ..CalibrationPolicy::default()
        };

        apply_uncertainty_adjusted_ranking_to_parallelism(&mut results, &policy);

        assert_eq!(results[0].config.tensor_ranks, 2);
        assert_eq!(results[1].config.tensor_ranks, 1);
        let nominal_ranks = parallelism_nominal_rank_map(&results);
        let adjusted_ranks =
            parallelism_uncertainty_adjusted_rank_map(&results, policy.uncertainty_ranking_weight);
        assert_eq!(nominal_ranks["tp1-pp1-ep1-dp1"], 1);
        assert_eq!(nominal_ranks["tp2-pp1-ep1-dp1"], 2);
        assert_eq!(adjusted_ranks["tp1-pp1-ep1-dp1"], 2);
        assert_eq!(adjusted_ranks["tp2-pp1-ep1-dp1"], 1);
    }

    fn parallelism_score_with_uncertainty(
        tensor_ranks: u32,
        estimated_latency_s: f64,
        absolute_uncertainty_s: f64,
    ) -> ScoredParallelismConfig {
        ScoredParallelismConfig {
            config: ParallelismConfig {
                tensor_ranks,
                pipeline_ranks: 1,
                expert_ranks: 1,
                data_ranks: 1,
            },
            placement: RankPlacement {
                rank_to_gpu: Vec::new(),
            },
            placement_evidence: Vec::new(),
            groups: crate::types::configs::ParallelGroups {
                tensor_groups: Vec::new(),
                pipeline_stages: Vec::new(),
                expert_groups: Vec::new(),
                data_groups: Vec::new(),
            },
            feasible: true,
            estimated_latency_s,
            estimated_memory_per_gpu: crate::types::common::Bytes::from_bytes(0),
            calibration_fits: vec![CalibrationFitApplication {
                phase: "prefill".to_string(),
                target: "prefill_ms".to_string(),
                fit_name: Some(format!("tp{tensor_ranks}-fit")),
                model: "linear".to_string(),
                unit: Some("ms".to_string()),
                intercept: 1.0,
                raw_prediction: estimated_latency_s * 1000.0,
                prediction_kind: "latency".to_string(),
                predicted_value: estimated_latency_s,
                prediction_unit: Some("s".to_string()),
                predicted_s: estimated_latency_s,
                baseline_value: None,
                baseline_s: None,
                applicability_status: "interpolated".to_string(),
                confidence_score: 1.0,
                max_extrapolation_ratio: 0.0,
                relative_uncertainty_pct: Some(0.0),
                absolute_uncertainty_value: Some(absolute_uncertainty_s),
                absolute_uncertainty_s: Some(absolute_uncertainty_s),
                uncertainty_source: Some("unit-test".to_string()),
                validation_rmse: None,
                validation_rmse_pct: None,
                validation_mean_abs_pct_error: None,
                validation_max_abs_pct_error: None,
                confidence_interval: None,
                confidence_interval_pct: None,
                confidence_level: None,
                sample_count: Some(24),
                validation_sample_count: Some(6),
                source: Some("unit-test".to_string()),
                features: Vec::new(),
            }],
            calibration_gate_violations: Vec::new(),
            approximations: Vec::new(),
            approximation_policy_violations: Vec::new(),
            bottlenecks: Vec::new(),
            rejected_reason: None,
            operations: Vec::new(),
            scheduled_operations: Vec::new(),
            resource_utilization: Vec::new(),
            operation_makespan_s: estimated_latency_s,
        }
    }
}
