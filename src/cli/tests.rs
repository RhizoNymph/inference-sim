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
    let output_dir = std::env::temp_dir().join(format!("inference-sim-output-profile-all-{nanos}"));
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

    write_calibration_residuals_csv_if_configured(&args, Some("baseline"), Some(&profile), false)
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
        output.contains("interconnect_link from=0 to=1 rail=0 from_gpus=- to_gpus=- fabric=IB HDR")
    );
    assert!(
        output.contains("interconnect_link from=0 to=1 rail=1 from_gpus=- to_gpus=- fabric=IB HDR")
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
    let rejections_path = dir.join(format!("inference-sim-serving-json-rejections-{nanos}.csv"));
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
    let output = run_heterogeneous_example("heterogeneous_partially_disaggregated_workload.toml");

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
        slo_deadline["results"][0]["measurement_window"]["deadline_missed_request_count"].as_u64(),
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
            .any(|diagnostic| { diagnostic["code"].as_str() == Some("gpu_nic_path_host_staged") }),
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

fn run_temp_cli_json_twice(case_name: &str, cluster: &str, workload: &str) -> serde_json::Value {
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
        message: "workload prompt_tokens range 128..8192 falls outside calibration range 128..4096"
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
        output.contains("\"features\": [\"batch_size\", \"decode_tokens\", \"sequence_tokens\"]")
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
        violation.code == "fit_partially_bounded" && violation.action == CalibrationGateMode::Warn
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
