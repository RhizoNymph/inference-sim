use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CliArgs {
    pub(super) cluster_path: PathBuf,
    pub(super) workload_path: PathBuf,
    pub(super) top_k: usize,
    pub(super) request_metrics_csv_path: Option<PathBuf>,
    pub(super) request_lifecycle_events_csv_path: Option<PathBuf>,
    pub(super) serving_metrics_csv_path: Option<PathBuf>,
    pub(super) serving_metric_breakdowns_csv_path: Option<PathBuf>,
    pub(super) serving_services_csv_path: Option<PathBuf>,
    pub(super) serving_utilization_csv_path: Option<PathBuf>,
    pub(super) serving_memory_pressure_csv_path: Option<PathBuf>,
    pub(super) serving_timeline_csv_path: Option<PathBuf>,
    pub(super) serving_occupancy_csv_path: Option<PathBuf>,
    pub(super) serving_placement_evidence_csv_path: Option<PathBuf>,
    pub(super) serving_worker_evidence_csv_path: Option<PathBuf>,
    pub(super) serving_rejections_csv_path: Option<PathBuf>,
    pub(super) serving_route_paths_csv_path: Option<PathBuf>,
    pub(super) kv_route_resources_csv_path: Option<PathBuf>,
    pub(super) serving_bottlenecks_csv_path: Option<PathBuf>,
    pub(super) serving_phase_calibration_csv_path: Option<PathBuf>,
    pub(super) serving_approximations_csv_path: Option<PathBuf>,
    pub(super) calibration_residuals_csv_path: Option<PathBuf>,
    pub(super) scenario_sensitivity_csv_path: Option<PathBuf>,
    pub(super) rank_sensitivity_csv_path: Option<PathBuf>,
    pub(super) output_dir: Option<PathBuf>,
    pub(super) output_profile: OutputProfile,
    pub(super) format: OutputFormat,
    pub(super) trace: bool,
    pub(super) trace_limit: Option<usize>,
    pub(super) request_limit: Option<usize>,
    pub(super) occupancy: bool,
    pub(super) occupancy_buckets: usize,
    pub(super) occupancy_resource_limit: Option<usize>,
    pub(super) critical_path: bool,
    pub(super) critical_path_limit: Option<usize>,
    pub(super) search_budget: RunSearchBudgetConfig,
    pub(super) scenarios: Vec<RunScenarioConfig>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum OutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum OutputProfile {
    #[default]
    Summary,
    Compare,
    Calibration,
    Audit,
    All,
}

impl OutputProfile {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Compare => "compare",
            Self::Calibration => "calibration",
            Self::Audit => "audit",
            Self::All => "all",
        }
    }

    pub(super) fn materialized_profiles(self) -> &'static [OutputProfile] {
        match self {
            Self::Summary => &[OutputProfile::Summary],
            Self::Compare => &[OutputProfile::Compare],
            Self::Calibration => &[OutputProfile::Calibration],
            Self::Audit => &[OutputProfile::Audit],
            Self::All => &[
                OutputProfile::Summary,
                OutputProfile::Compare,
                OutputProfile::Calibration,
                OutputProfile::Audit,
            ],
        }
    }
}

pub(super) fn primary_output_filename(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Text => "results.txt",
        OutputFormat::Json => "results.json",
        OutputFormat::Markdown => "results.md",
    }
}

pub(super) fn apply_output_profile_defaults(args: &mut CliArgs) {
    match args.output_profile {
        OutputProfile::Summary => {
            args.format = OutputFormat::Markdown;
            args.trace = false;
            args.occupancy = false;
            args.critical_path = false;
        }
        OutputProfile::Compare => {
            args.format = OutputFormat::Text;
            args.trace = false;
            args.occupancy = false;
            args.critical_path = false;
        }
        OutputProfile::Calibration => {
            args.format = OutputFormat::Json;
            args.trace = false;
            args.occupancy = false;
            args.critical_path = false;
        }
        OutputProfile::Audit => {
            args.format = OutputFormat::Json;
            args.trace = true;
            args.trace_limit = None;
            args.request_limit = None;
            args.occupancy = true;
            args.occupancy_resource_limit = None;
            args.critical_path = true;
            args.critical_path_limit = None;
        }
        OutputProfile::All => {}
    }
}

pub(super) fn apply_output_profile_paths(args: &mut CliArgs, profile_dir: &Path) {
    match args.output_profile {
        OutputProfile::Summary => {}
        OutputProfile::Compare => {
            set_output_path(
                &mut args.serving_metrics_csv_path,
                profile_dir,
                "serving_metrics.csv",
            );
            set_output_path(
                &mut args.serving_metric_breakdowns_csv_path,
                profile_dir,
                "metric_breakdowns.csv",
            );
            set_output_path(
                &mut args.serving_rejections_csv_path,
                profile_dir,
                "rejections.csv",
            );
            set_output_path(
                &mut args.serving_bottlenecks_csv_path,
                profile_dir,
                "bottlenecks.csv",
            );
            set_output_path(
                &mut args.rank_sensitivity_csv_path,
                profile_dir,
                "rank_sensitivity.csv",
            );
            if !args.scenarios.is_empty() {
                set_output_path(
                    &mut args.scenario_sensitivity_csv_path,
                    profile_dir,
                    "scenario_sensitivity.csv",
                );
            }
        }
        OutputProfile::Calibration => {
            set_output_path(
                &mut args.serving_metrics_csv_path,
                profile_dir,
                "serving_metrics.csv",
            );
            set_output_path(
                &mut args.serving_phase_calibration_csv_path,
                profile_dir,
                "phase_calibration.csv",
            );
            set_output_path(
                &mut args.serving_approximations_csv_path,
                profile_dir,
                "approximations.csv",
            );
            set_output_path(
                &mut args.calibration_residuals_csv_path,
                profile_dir,
                "calibration_residuals.csv",
            );
            set_output_path(
                &mut args.request_metrics_csv_path,
                profile_dir,
                "request_metrics.csv",
            );
        }
        OutputProfile::Audit => {
            set_output_path(
                &mut args.request_metrics_csv_path,
                profile_dir,
                "request_metrics.csv",
            );
            set_output_path(
                &mut args.request_lifecycle_events_csv_path,
                profile_dir,
                "request_lifecycle_events.csv",
            );
            set_output_path(
                &mut args.serving_metrics_csv_path,
                profile_dir,
                "serving_metrics.csv",
            );
            set_output_path(
                &mut args.serving_metric_breakdowns_csv_path,
                profile_dir,
                "metric_breakdowns.csv",
            );
            set_output_path(
                &mut args.serving_services_csv_path,
                profile_dir,
                "services.csv",
            );
            set_output_path(
                &mut args.serving_utilization_csv_path,
                profile_dir,
                "utilization.csv",
            );
            set_output_path(
                &mut args.serving_memory_pressure_csv_path,
                profile_dir,
                "memory_pressure.csv",
            );
            set_output_path(
                &mut args.serving_timeline_csv_path,
                profile_dir,
                "timeline.csv",
            );
            set_output_path(
                &mut args.serving_occupancy_csv_path,
                profile_dir,
                "occupancy.csv",
            );
            set_output_path(
                &mut args.serving_placement_evidence_csv_path,
                profile_dir,
                "placement_evidence.csv",
            );
            set_output_path(
                &mut args.serving_worker_evidence_csv_path,
                profile_dir,
                "worker_evidence.csv",
            );
            set_output_path(
                &mut args.serving_rejections_csv_path,
                profile_dir,
                "rejections.csv",
            );
            set_output_path(
                &mut args.serving_route_paths_csv_path,
                profile_dir,
                "route_paths.csv",
            );
            set_output_path(
                &mut args.kv_route_resources_csv_path,
                profile_dir,
                "kv_route_resources.csv",
            );
            set_output_path(
                &mut args.serving_bottlenecks_csv_path,
                profile_dir,
                "bottlenecks.csv",
            );
            set_output_path(
                &mut args.serving_phase_calibration_csv_path,
                profile_dir,
                "phase_calibration.csv",
            );
            set_output_path(
                &mut args.serving_approximations_csv_path,
                profile_dir,
                "approximations.csv",
            );
            set_output_path(
                &mut args.calibration_residuals_csv_path,
                profile_dir,
                "calibration_residuals.csv",
            );
            if !args.scenarios.is_empty() {
                set_output_path(
                    &mut args.scenario_sensitivity_csv_path,
                    profile_dir,
                    "scenario_sensitivity.csv",
                );
            }
            set_output_path(
                &mut args.rank_sensitivity_csv_path,
                profile_dir,
                "rank_sensitivity.csv",
            );
        }
        OutputProfile::All => {}
    }
}

fn set_output_path(path: &mut Option<PathBuf>, profile_dir: &Path, filename: &str) {
    if path.is_none() {
        *path = Some(profile_dir.join(filename));
    }
}

impl CliArgs {
    pub(super) fn parse<I, S>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut args = args.into_iter().map(Into::into);
        let _bin = args.next();
        let mut run_path = None;
        let mut cluster_path = None;
        let mut workload_path = None;
        let mut top_k = None;
        let mut request_metrics_csv_path = None;
        let mut request_lifecycle_events_csv_path = None;
        let mut serving_metrics_csv_path = None;
        let mut serving_metric_breakdowns_csv_path = None;
        let mut serving_services_csv_path = None;
        let mut serving_utilization_csv_path = None;
        let mut serving_memory_pressure_csv_path = None;
        let mut serving_timeline_csv_path = None;
        let mut serving_occupancy_csv_path = None;
        let mut serving_placement_evidence_csv_path = None;
        let mut serving_worker_evidence_csv_path = None;
        let mut serving_rejections_csv_path = None;
        let mut serving_route_paths_csv_path = None;
        let mut kv_route_resources_csv_path = None;
        let mut serving_bottlenecks_csv_path = None;
        let mut serving_phase_calibration_csv_path = None;
        let mut serving_approximations_csv_path = None;
        let mut calibration_residuals_csv_path = None;
        let mut scenario_sensitivity_csv_path = None;
        let mut rank_sensitivity_csv_path = None;
        let mut output_dir = None;
        let mut output_profile = None;
        let mut format = None;
        let mut trace = None;
        let mut trace_limit = None;
        let mut request_limit = None;
        let mut occupancy = None;
        let mut occupancy_buckets = None;
        let mut occupancy_resource_limit = None;
        let mut critical_path = None;
        let mut critical_path_limit = None;
        let mut max_parallelism_candidates = None;
        let mut max_prefill_candidates = None;
        let mut max_decode_candidates = None;
        let mut max_serving_pairs = None;
        let mut max_runtime_ms = None;
        let mut retain_rejected_candidates = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => return Err(CliError::Help(usage())),
                "-r" | "--run" => {
                    run_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "-c" | "--cluster" => {
                    cluster_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "-w" | "--workload" => {
                    workload_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "-n" | "--top-k" => {
                    let value = next_value(&mut args, &arg)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        CliError::Usage(format!("invalid --top-k value '{value}'\n\n{}", usage()))
                    })?;
                    if parsed == 0 {
                        return Err(CliError::Usage(format!(
                            "--top-k must be greater than zero\n\n{}",
                            usage()
                        )));
                    }
                    top_k = Some(parsed);
                }
                "--request-metrics-csv" | "--serving-request-metrics-csv" => {
                    request_metrics_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--request-lifecycle-events-csv"
                | "--serving-request-lifecycle-events-csv"
                | "--lifecycle-events-csv"
                | "--serving-lifecycle-events-csv" => {
                    request_lifecycle_events_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-metrics-csv"
                | "--serving-candidate-metrics-csv"
                | "--candidate-metrics-csv" => {
                    serving_metrics_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-metric-breakdowns-csv"
                | "--serving-breakdowns-csv"
                | "--metric-breakdowns-csv"
                | "--breakdowns-csv" => {
                    serving_metric_breakdowns_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-services-csv"
                | "--serving-service-metrics-csv"
                | "--service-metrics-csv"
                | "--services-csv" => {
                    serving_services_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-utilization-csv"
                | "--serving-resource-utilization-csv"
                | "--resource-utilization-csv"
                | "--utilization-csv" => {
                    serving_utilization_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-memory-pressure-csv"
                | "--serving-hbm-pressure-csv"
                | "--memory-pressure-csv"
                | "--hbm-pressure-csv" => {
                    serving_memory_pressure_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-timeline-csv"
                | "--serving-scheduled-operations-csv"
                | "--scheduled-operations-csv"
                | "--timeline-csv" => {
                    serving_timeline_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-occupancy-csv"
                | "--serving-resource-occupancy-csv"
                | "--resource-occupancy-csv"
                | "--occupancy-csv" => {
                    serving_occupancy_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-placement-evidence-csv"
                | "--serving-placement-csv"
                | "--placement-evidence-csv"
                | "--placement-csv" => {
                    serving_placement_evidence_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-worker-evidence-csv"
                | "--serving-worker-assignments-csv"
                | "--worker-evidence-csv"
                | "--worker-assignments-csv" => {
                    serving_worker_evidence_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-rejections-csv"
                | "--serving-rejection-evidence-csv"
                | "--rejections-csv"
                | "--rejection-evidence-csv" => {
                    serving_rejections_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-route-paths-csv"
                | "--serving-kv-route-paths-csv"
                | "--kv-route-paths-csv"
                | "--route-paths-csv" => {
                    serving_route_paths_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--kv-route-resources-csv"
                | "--kv-route-resource-csv"
                | "--route-resources-csv"
                | "--serving-route-resources-csv" => {
                    kv_route_resources_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-bottlenecks-csv"
                | "--serving-bottleneck-summary-csv"
                | "--bottlenecks-csv" => {
                    serving_bottlenecks_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-phase-calibration-csv"
                | "--serving-calibration-phases-csv"
                | "--phase-calibration-csv" => {
                    serving_phase_calibration_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--serving-approximations-csv"
                | "--serving-approximation-evidence-csv"
                | "--approximations-csv"
                | "--approximation-evidence-csv" => {
                    serving_approximations_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--calibration-residuals-csv"
                | "--calibration-benchmark-residuals-csv"
                | "--calibration-benchmarks-csv" => {
                    calibration_residuals_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--scenario-sensitivity-csv"
                | "--serving-scenario-sensitivity-csv"
                | "--sensitivity-csv" => {
                    scenario_sensitivity_csv_path =
                        Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--rank-sensitivity-csv"
                | "--solver-rank-sensitivity-csv"
                | "--serving-rank-sensitivity-csv" => {
                    rank_sensitivity_csv_path = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--output-dir" | "--output-directory" => {
                    output_dir = Some(PathBuf::from(next_value(&mut args, &arg)?));
                }
                "--output-profile" => {
                    let value = next_value(&mut args, &arg)?;
                    output_profile = Some(parse_output_profile(&value)?);
                }
                "--format" => {
                    let value = next_value(&mut args, &arg)?;
                    format = Some(parse_output_format(&value)?);
                }
                "--json" => {
                    format = Some(OutputFormat::Json);
                }
                "--trace" => {
                    trace = Some(true);
                }
                "--trace-limit" => {
                    let value = next_value(&mut args, &arg)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        CliError::Usage(format!(
                            "invalid --trace-limit value '{value}'\n\n{}",
                            usage()
                        ))
                    })?;
                    trace = Some(true);
                    trace_limit = Some(if parsed == 0 { None } else { Some(parsed) });
                }
                "--request-limit" => {
                    let value = next_value(&mut args, &arg)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        CliError::Usage(format!(
                            "invalid --request-limit value '{value}'\n\n{}",
                            usage()
                        ))
                    })?;
                    request_limit = Some(if parsed == 0 { None } else { Some(parsed) });
                }
                "--occupancy" => {
                    occupancy = Some(true);
                }
                "--occupancy-buckets" => {
                    let value = next_value(&mut args, &arg)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        CliError::Usage(format!(
                            "invalid --occupancy-buckets value '{value}'\n\n{}",
                            usage()
                        ))
                    })?;
                    if parsed == 0 {
                        return Err(CliError::Usage(format!(
                            "--occupancy-buckets must be greater than zero\n\n{}",
                            usage()
                        )));
                    }
                    occupancy = Some(true);
                    occupancy_buckets = Some(parsed);
                }
                "--occupancy-resource-limit" => {
                    let value = next_value(&mut args, &arg)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        CliError::Usage(format!(
                            "invalid --occupancy-resource-limit value '{value}'\n\n{}",
                            usage()
                        ))
                    })?;
                    occupancy = Some(true);
                    occupancy_resource_limit = Some(if parsed == 0 { None } else { Some(parsed) });
                }
                "--critical-path" => {
                    critical_path = Some(true);
                }
                "--critical-path-limit" => {
                    let value = next_value(&mut args, &arg)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        CliError::Usage(format!(
                            "invalid --critical-path-limit value '{value}'\n\n{}",
                            usage()
                        ))
                    })?;
                    critical_path = Some(true);
                    critical_path_limit = Some(if parsed == 0 { None } else { Some(parsed) });
                }
                "--max-candidates" | "--max-parallelism-candidates" => {
                    max_parallelism_candidates = Some(parse_positive_usize_arg(
                        &next_value(&mut args, &arg)?,
                        &arg,
                    )?);
                }
                "--max-prefill-candidates" => {
                    max_prefill_candidates = Some(parse_positive_usize_arg(
                        &next_value(&mut args, &arg)?,
                        &arg,
                    )?);
                }
                "--max-decode-candidates" => {
                    max_decode_candidates = Some(parse_positive_usize_arg(
                        &next_value(&mut args, &arg)?,
                        &arg,
                    )?);
                }
                "--max-serving-pairs" => {
                    max_serving_pairs = Some(parse_positive_usize_arg(
                        &next_value(&mut args, &arg)?,
                        &arg,
                    )?);
                }
                "--max-search-runtime-ms" | "--max-runtime-ms" => {
                    max_runtime_ms = Some(parse_u64_arg(&next_value(&mut args, &arg)?, &arg)?);
                }
                "--retain-rejected-candidates" | "--include-rejected-candidates" => {
                    retain_rejected_candidates = Some(true);
                }
                "--drop-rejected-candidates" | "--omit-rejected-candidates" => {
                    retain_rejected_candidates = Some(false);
                }
                unknown => {
                    return Err(CliError::Usage(format!(
                        "unknown argument '{unknown}'\n\n{}",
                        usage()
                    )));
                }
            }
        }

        let run_config = if let Some(path) = run_path {
            load_run_config(path)?
        } else {
            RunConfig::default()
        };

        let cluster_path = cluster_path.or(run_config.cluster_path);
        let workload_path = workload_path.or(run_config.workload_path);
        let Some(cluster_path) = cluster_path else {
            return Err(CliError::Usage(format!(
                "missing required --cluster path or run.cluster\n\n{}",
                usage()
            )));
        };
        let Some(workload_path) = workload_path else {
            return Err(CliError::Usage(format!(
                "missing required --workload path or run.workload\n\n{}",
                usage()
            )));
        };
        let mut format = format
            .or(run_config
                .output
                .format
                .as_deref()
                .map(parse_output_format)
                .transpose()?)
            .unwrap_or(OutputFormat::Text);
        let top_k = top_k.or(run_config.output.top_k).unwrap_or(DEFAULT_TOP_K);
        let request_metrics_csv_path =
            request_metrics_csv_path.or(run_config.output.request_metrics_csv_path);
        let request_lifecycle_events_csv_path = request_lifecycle_events_csv_path
            .or(run_config.output.request_lifecycle_events_csv_path);
        let serving_metrics_csv_path =
            serving_metrics_csv_path.or(run_config.output.serving_metrics_csv_path);
        let serving_metric_breakdowns_csv_path = serving_metric_breakdowns_csv_path
            .or(run_config.output.serving_metric_breakdowns_csv_path);
        let serving_services_csv_path =
            serving_services_csv_path.or(run_config.output.serving_services_csv_path);
        let serving_utilization_csv_path =
            serving_utilization_csv_path.or(run_config.output.serving_utilization_csv_path);
        let serving_memory_pressure_csv_path =
            serving_memory_pressure_csv_path.or(run_config.output.serving_memory_pressure_csv_path);
        let serving_timeline_csv_path =
            serving_timeline_csv_path.or(run_config.output.serving_timeline_csv_path);
        let serving_occupancy_csv_path =
            serving_occupancy_csv_path.or(run_config.output.serving_occupancy_csv_path);
        let serving_placement_evidence_csv_path = serving_placement_evidence_csv_path
            .or(run_config.output.serving_placement_evidence_csv_path);
        let serving_worker_evidence_csv_path =
            serving_worker_evidence_csv_path.or(run_config.output.serving_worker_evidence_csv_path);
        let serving_rejections_csv_path =
            serving_rejections_csv_path.or(run_config.output.serving_rejections_csv_path);
        let serving_route_paths_csv_path =
            serving_route_paths_csv_path.or(run_config.output.serving_route_paths_csv_path);
        let kv_route_resources_csv_path =
            kv_route_resources_csv_path.or(run_config.output.kv_route_resources_csv_path);
        let serving_bottlenecks_csv_path =
            serving_bottlenecks_csv_path.or(run_config.output.serving_bottlenecks_csv_path);
        let serving_phase_calibration_csv_path = serving_phase_calibration_csv_path
            .or(run_config.output.serving_phase_calibration_csv_path);
        let serving_approximations_csv_path =
            serving_approximations_csv_path.or(run_config.output.serving_approximations_csv_path);
        let calibration_residuals_csv_path =
            calibration_residuals_csv_path.or(run_config.output.calibration_residuals_csv_path);
        let scenario_sensitivity_csv_path =
            scenario_sensitivity_csv_path.or(run_config.output.scenario_sensitivity_csv_path);
        let rank_sensitivity_csv_path =
            rank_sensitivity_csv_path.or(run_config.output.rank_sensitivity_csv_path);
        let output_dir = output_dir.or(run_config.output.output_dir);
        let output_profile_was_configured =
            output_profile.is_some() || run_config.output.output_profile.is_some();
        let output_profile = output_profile
            .or(run_config
                .output
                .output_profile
                .as_deref()
                .map(parse_output_profile)
                .transpose()?)
            .unwrap_or(if output_dir.is_some() {
                OutputProfile::All
            } else {
                OutputProfile::Summary
            });
        if output_profile_was_configured && output_dir.is_none() {
            return Err(CliError::Usage(format!(
                "--output-profile requires --output-dir or output.output_dir\n\n{}",
                usage()
            )));
        }
        let trace = trace.or(run_config.output.trace).unwrap_or(false);
        let trace_limit = trace_limit
            .or(run_config.output.trace_limit.map(nonzero_limit))
            .unwrap_or(Some(DEFAULT_TRACE_LIMIT));
        let request_limit = request_limit
            .or(run_config.output.request_limit.map(nonzero_limit))
            .unwrap_or(Some(DEFAULT_REQUEST_OBSERVATION_LIMIT));
        let occupancy = occupancy.or(run_config.output.occupancy).unwrap_or(false);
        let occupancy_buckets = occupancy_buckets
            .or(run_config.output.occupancy_buckets)
            .unwrap_or(DEFAULT_OCCUPANCY_BUCKETS);
        let occupancy_resource_limit = occupancy_resource_limit
            .or(run_config
                .output
                .occupancy_resource_limit
                .map(nonzero_limit))
            .unwrap_or(Some(DEFAULT_OCCUPANCY_RESOURCE_LIMIT));
        let critical_path = critical_path
            .or(run_config.output.critical_path)
            .unwrap_or(false);
        let critical_path_limit = critical_path_limit
            .or(run_config.output.critical_path_limit.map(nonzero_limit))
            .unwrap_or(Some(DEFAULT_CRITICAL_PATH_LIMIT));
        let search_budget = RunSearchBudgetConfig {
            max_parallelism_candidates: max_parallelism_candidates
                .or(run_config.search_budget.max_parallelism_candidates),
            max_prefill_candidates: max_prefill_candidates
                .or(run_config.search_budget.max_prefill_candidates),
            max_decode_candidates: max_decode_candidates
                .or(run_config.search_budget.max_decode_candidates),
            max_serving_pairs: max_serving_pairs.or(run_config.search_budget.max_serving_pairs),
            max_runtime_ms: max_runtime_ms.or(run_config.search_budget.max_runtime_ms),
            retain_rejected_candidates: retain_rejected_candidates
                .or(run_config.search_budget.retain_rejected_candidates),
        };
        let scenarios = run_config.scenarios;
        if trace || occupancy || critical_path {
            format = OutputFormat::Json;
        }

        Ok(Self {
            cluster_path,
            workload_path,
            top_k,
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
            output_dir,
            output_profile,
            format,
            trace,
            trace_limit,
            request_limit,
            occupancy,
            occupancy_buckets,
            occupancy_resource_limit,
            critical_path,
            critical_path_limit,
            search_budget,
            scenarios,
        })
    }
}

fn parse_output_format(value: &str) -> Result<OutputFormat, CliError> {
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "text" => Ok(OutputFormat::Text),
        "json" => Ok(OutputFormat::Json),
        "markdown" | "md" => Ok(OutputFormat::Markdown),
        _ => Err(CliError::Usage(format!(
            "invalid --format value '{value}'; use text, json, or markdown\n\n{}",
            usage()
        ))),
    }
}

fn parse_output_profile(value: &str) -> Result<OutputProfile, CliError> {
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "summary" => Ok(OutputProfile::Summary),
        "compare" | "comparison" => Ok(OutputProfile::Compare),
        "calibration" | "calibrate" => Ok(OutputProfile::Calibration),
        "audit" | "debug" | "full" => Ok(OutputProfile::Audit),
        "all" => Ok(OutputProfile::All),
        _ => Err(CliError::Usage(format!(
            "invalid --output-profile value '{value}'; use summary, compare, calibration, audit, or all\n\n{}",
            usage()
        ))),
    }
}

fn nonzero_limit(value: usize) -> Option<usize> {
    if value == 0 { None } else { Some(value) }
}

fn parse_positive_usize_arg(value: &str, flag: &str) -> Result<usize, CliError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| CliError::Usage(format!("invalid {flag} value '{value}'\n\n{}", usage())))?;
    if parsed == 0 {
        return Err(CliError::Usage(format!(
            "{flag} must be greater than zero\n\n{}",
            usage()
        )));
    }
    Ok(parsed)
}

fn parse_u64_arg(value: &str, flag: &str) -> Result<u64, CliError> {
    value
        .parse::<u64>()
        .map_err(|_| CliError::Usage(format!("invalid {flag} value '{value}'\n\n{}", usage())))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, CliError> {
    args.next()
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| CliError::Usage(format!("missing value for {flag}\n\n{}", usage())))
}

fn usage() -> String {
    "usage: inference-sim [--run <run.toml>] [--cluster <cluster.toml>] [--workload <workload.toml>] [--top-k <n>] [--request-metrics-csv <path>] [--request-lifecycle-events-csv <path>] [--serving-metrics-csv <path>] [--serving-metric-breakdowns-csv <path>] [--serving-services-csv <path>] [--serving-utilization-csv <path>] [--serving-memory-pressure-csv <path>] [--serving-timeline-csv <path>] [--serving-occupancy-csv <path>] [--serving-placement-evidence-csv <path>] [--serving-worker-evidence-csv <path>] [--serving-rejections-csv <path>] [--serving-route-paths-csv <path>] [--kv-route-resources-csv <path>] [--serving-bottlenecks-csv <path>] [--serving-phase-calibration-csv <path>] [--serving-approximations-csv <path>] [--calibration-residuals-csv <path>] [--scenario-sensitivity-csv <path>] [--rank-sensitivity-csv <path>] [--output-dir <dir>] [--output-profile summary|compare|calibration|audit|all] [--format text|json|markdown] [--trace] [--trace-limit <n>] [--request-limit <n>] [--occupancy] [--occupancy-buckets <n>] [--occupancy-resource-limit <n>] [--critical-path] [--critical-path-limit <n>] [--max-candidates <n>] [--max-prefill-candidates <n>] [--max-decode-candidates <n>] [--max-serving-pairs <n>] [--max-search-runtime-ms <ms>] [--retain-rejected-candidates|--drop-rejected-candidates]"
        .to_string()
}
