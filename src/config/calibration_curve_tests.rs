//! TOML parsing and validation of `compute_efficiency_curve` and the
//! frontend-latency calibration keys, and the parameter-count source.

use super::*;

const MODEL_AND_REQUEST: &str = r#"
[model]
layers = 28
hidden_size = 3584
attention_heads = 28
kv_heads = 4
vocab_size = 152064
ffn_hidden_size = 18944
parameters_gb = 15.23
dtype = "bf16"

[request]
batch_size = 1
prompt_tokens = 512
decode_tokens = 128
max_sequence_tokens = 640
phase = "end_to_end"
"#;

fn workload_with_calibration(calibration: &str) -> Result<WorkloadConfig, ConfigError> {
    parse_workload(&format!(
        "{MODEL_AND_REQUEST}\n[calibration]\n{calibration}\n"
    ))
}

fn error_text(result: Result<WorkloadConfig, ConfigError>) -> String {
    match result {
        Ok(_) => panic!("expected a config error"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn parses_a_compute_efficiency_curve() {
    let workload = workload_with_calibration(
        "compute_efficiency = 0.85\ncompute_efficiency_curve = [[128, 0.70], [512, 0.78], [1024, 1]]",
    )
    .expect("valid curve");
    let curve = workload
        .calibration
        .compute_efficiency_curve
        .expect("curve parsed");
    let points: Vec<(u64, f64)> = curve
        .points()
        .iter()
        .map(|point| (point.tokens(), point.efficiency()))
        .collect();
    assert_eq!(points, vec![(128, 0.70), (512, 0.78), (1024, 1.0)]);
    assert_eq!(workload.calibration.compute_efficiency, 0.85);
    assert_eq!(workload.calibration.compute_efficiency_at(256.0), 0.74);
    assert_eq!(
        workload.calibration_overrides.compute_efficiency_curve,
        Some(curve)
    );
}

#[test]
fn absent_curve_keeps_the_scalar() {
    let workload = workload_with_calibration("compute_efficiency = 0.85").expect("valid");
    assert_eq!(workload.calibration.compute_efficiency_curve, None);
    assert_eq!(workload.calibration.compute_efficiency_at(16.0), 0.85);
}

#[test]
fn rejects_a_single_point_curve() {
    let err = error_text(workload_with_calibration(
        "compute_efficiency_curve = [[128, 0.7]]",
    ));
    assert!(
        err.contains("calibration.compute_efficiency_curve") && err.contains("at least 2 points"),
        "{err}"
    );
}

#[test]
fn rejects_non_increasing_curve_tokens() {
    let err = error_text(workload_with_calibration(
        "compute_efficiency_curve = [[512, 0.7], [256, 0.8]]",
    ));
    assert!(err.contains("points[1] tokens 256"), "{err}");
}

#[test]
fn rejects_curve_efficiency_above_one() {
    let err = error_text(workload_with_calibration(
        "compute_efficiency_curve = [[128, 0.7], [256, 1.2]]",
    ));
    assert!(err.contains("points[1] efficiency 1.2"), "{err}");
}

#[test]
fn rejects_zero_curve_tokens() {
    let err = error_text(workload_with_calibration(
        "compute_efficiency_curve = [[0, 0.7], [256, 0.8]]",
    ));
    assert!(err.contains("points[0] has zero tokens"), "{err}");
}

#[test]
fn parses_frontend_latency() {
    let workload = workload_with_calibration(
        "frontend_latency_us = 5070\nfrontend_latency_per_prompt_token_us = 14.5",
    )
    .expect("valid");
    assert_eq!(workload.calibration.frontend_latency_us, 5070.0);
    assert_eq!(
        workload.calibration.frontend_latency_per_prompt_token_us,
        14.5
    );
    assert!(workload.calibration.models_frontend_latency());
}

#[test]
fn rejects_negative_frontend_latency() {
    let err = error_text(workload_with_calibration("frontend_latency_us = -1.0"));
    assert!(err.contains("calibration.frontend_latency_us"), "{err}");
    let err = error_text(workload_with_calibration(
        "frontend_latency_per_prompt_token_us = -0.5",
    ));
    assert!(
        err.contains("calibration.frontend_latency_per_prompt_token_us"),
        "{err}"
    );
}

#[test]
fn calibration_profile_carries_the_curve_and_frontend_latency() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("inference_sim_efficiency_curve_{unique}"));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let profile = dir.join("profile.toml");
    std::fs::write(
        &profile,
        r#"
schema_version = 1
[profile]
name = "curve-profile"
[calibration]
compute_efficiency = 0.85
decode_memory_bandwidth_scale = 0.84
compute_efficiency_curve = [[128, 0.70], [1024, 0.89]]
frontend_latency_us = 5000.0
"#,
    )
    .expect("write profile");
    let loaded = load_calibration_profile_path(&profile).expect("profile loads");
    assert_eq!(loaded.calibration.compute_efficiency_at(1024.0), 0.89);
    assert_eq!(loaded.calibration.frontend_latency_us, 5000.0);

    // The workload inherits both, and can override the curve inline.
    let workload = parse_workload_with_base_dir(
        &format!("{MODEL_AND_REQUEST}\n[calibration_profile]\npath = \"profile.toml\"\n"),
        Some(&dir),
    )
    .expect("workload loads");
    assert_eq!(workload.calibration.compute_efficiency_at(128.0), 0.70);
    assert_eq!(workload.calibration.frontend_latency_us, 5000.0);
    let workload = parse_workload_with_base_dir(
        &format!(
            "{MODEL_AND_REQUEST}\n[calibration_profile]\npath = \"profile.toml\"\n[calibration]\ncompute_efficiency_curve = [[16, 0.5], [32, 0.6]]\n"
        ),
        Some(&dir),
    )
    .expect("workload loads");
    assert_eq!(workload.calibration.compute_efficiency_at(1024.0), 0.6);

    std::fs::write(
        &profile,
        "schema_version = 1\n[calibration]\ncompute_efficiency_curve = [[128, 0.7]]\n",
    )
    .expect("write profile");
    let err = load_calibration_profile_path(&profile)
        .map(|_| ())
        .expect_err("invalid curve rejected");
    assert!(
        err.to_string()
            .contains("calibration profile calibration.compute_efficiency_curve"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scenario_calibration_validates_the_curve_and_frontend_latency() {
    let config = |calibration: &str| {
        parse_run_config(&format!(
            "schema_version = 1\ncluster = \"cluster.toml\"\nworkload = \"workload.toml\"\n[[scenarios]]\nname = \"s\"\n[scenarios.calibration]\n{calibration}\n"
        ))
    };
    let parsed =
        config("compute_efficiency_curve = [[128, 0.7], [256, 0.8]]\nfrontend_latency_us = 10.0")
            .expect("valid scenario");
    let overrides = parsed.scenarios[0].calibration;
    assert!(overrides.compute_efficiency_curve.is_some());
    assert_eq!(overrides.frontend_latency_us, Some(10.0));

    let err = config("compute_efficiency_curve = [[128, 0.7]]")
        .map(|_| ())
        .expect_err("invalid curve");
    assert!(
        err.to_string()
            .contains("scenarios[0].calibration.compute_efficiency_curve"),
        "{err}"
    );
    let err = config("frontend_latency_us = -3.0")
        .map(|_| ())
        .expect_err("negative latency");
    assert!(
        err.to_string()
            .contains("scenarios[0].calibration.frontend_latency_us"),
        "{err}"
    );
}

#[test]
fn records_where_the_parameter_count_came_from() {
    let with_ffn = parse_workload(MODEL_AND_REQUEST).expect("valid");
    assert_eq!(
        with_ffn.model.parameter_count_source,
        ParameterCountSource::ShapeWithFfnWidth
    );
    assert_eq!(with_ffn.model.parameter_count_mismatch(), None);

    let without_ffn =
        parse_workload(&MODEL_AND_REQUEST.replace("ffn_hidden_size = 18944\n", "")).expect("valid");
    assert_eq!(
        without_ffn.model.parameter_count_source,
        ParameterCountSource::ShapeWithDefaultFfnWidth
    );
    let mismatch = without_ffn
        .model
        .parameter_count_mismatch()
        .expect("4x-hidden default undercounts Qwen2.5-7B");
    assert!(mismatch.relative_difference < -0.10);

    let explicit = parse_workload(&MODEL_AND_REQUEST.replace(
        "ffn_hidden_size = 18944\n",
        "parameter_count_billion = 7.6\n",
    ))
    .expect("valid");
    assert_eq!(
        explicit.model.parameter_count_source,
        ParameterCountSource::Explicit
    );
    assert_eq!(explicit.model.parameter_count_mismatch(), None);
}
