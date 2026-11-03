//! TOML parsing and validation of `kv_read_bandwidth_scale` in workload,
//! calibration-profile, and scenario `[calibration]` sections.

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
phase = "decode"
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
fn absent_kv_read_scale_falls_back_to_the_weight_read_scale() {
    let workload =
        workload_with_calibration("decode_memory_bandwidth_scale = 0.8383").expect("valid");
    assert_eq!(workload.calibration.kv_read_bandwidth_scale, None);
    assert_eq!(
        workload.calibration.effective_kv_read_bandwidth_scale(),
        0.8383
    );
    assert_eq!(workload.calibration_overrides.kv_read_bandwidth_scale, None);
}

#[test]
fn parses_a_kv_read_scale() {
    let workload = workload_with_calibration(
        "decode_memory_bandwidth_scale = 0.8383\nkv_read_bandwidth_scale = 0.452",
    )
    .expect("valid");
    assert_eq!(workload.calibration.kv_read_bandwidth_scale, Some(0.452));
    assert_eq!(workload.calibration.decode_memory_bandwidth_scale, 0.8383);
    assert_eq!(
        workload.calibration_overrides.kv_read_bandwidth_scale,
        Some(0.452)
    );
}

#[test]
fn rejects_a_non_positive_kv_read_scale() {
    for value in ["0.0", "-0.5", "nan", "inf"] {
        let err = error_text(workload_with_calibration(&format!(
            "kv_read_bandwidth_scale = {value}"
        )));
        assert!(err.contains("calibration.kv_read_bandwidth_scale"), "{err}");
    }
}

#[test]
fn calibration_profile_carries_the_kv_read_scale() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("inference_sim_kv_read_scale_{unique}"));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let profile = dir.join("profile.toml");
    std::fs::write(
        &profile,
        "schema_version = 1\n[profile]\nname = \"kv\"\n[calibration]\ndecode_memory_bandwidth_scale = 0.84\nkv_read_bandwidth_scale = 0.45\n",
    )
    .expect("write profile");
    let loaded = load_calibration_profile_path(&profile).expect("profile loads");
    assert_eq!(loaded.calibration.kv_read_bandwidth_scale, Some(0.45));

    // Inherited by a workload, and overridable inline.
    let inherited = parse_workload_with_base_dir(
        &format!("{MODEL_AND_REQUEST}\n[calibration_profile]\npath = \"profile.toml\"\n"),
        Some(&dir),
    )
    .expect("workload loads");
    assert_eq!(inherited.calibration.kv_read_bandwidth_scale, Some(0.45));
    let overridden = parse_workload_with_base_dir(
        &format!(
            "{MODEL_AND_REQUEST}\n[calibration_profile]\npath = \"profile.toml\"\n[calibration]\nkv_read_bandwidth_scale = 0.5\n"
        ),
        Some(&dir),
    )
    .expect("workload loads");
    assert_eq!(overridden.calibration.kv_read_bandwidth_scale, Some(0.5));
    assert_eq!(overridden.calibration.decode_memory_bandwidth_scale, 0.84);

    std::fs::write(
        &profile,
        "schema_version = 1\n[calibration]\nkv_read_bandwidth_scale = -1.0\n",
    )
    .expect("write profile");
    let err = load_calibration_profile_path(&profile)
        .map(|_| ())
        .expect_err("negative scale rejected");
    assert!(
        err.to_string()
            .contains("calibration profile calibration.kv_read_bandwidth_scale"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scenario_calibration_validates_the_kv_read_scale() {
    let config = |calibration: &str| {
        parse_run_config(&format!(
            "schema_version = 1\ncluster = \"cluster.toml\"\nworkload = \"workload.toml\"\n[[scenarios]]\nname = \"s\"\n[scenarios.calibration]\n{calibration}\n"
        ))
    };
    let parsed = config("kv_read_bandwidth_scale = 0.4").expect("valid scenario");
    assert_eq!(
        parsed.scenarios[0].calibration.kv_read_bandwidth_scale,
        Some(0.4)
    );
    let err = config("kv_read_bandwidth_scale = 0.0")
        .map(|_| ())
        .expect_err("zero scale");
    assert!(
        err.to_string()
            .contains("scenarios[0].calibration.kv_read_bandwidth_scale"),
        "{err}"
    );
}
