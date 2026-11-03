//! The JSON calibration block reports `kv_read_bandwidth_scale` (null when
//! absent), a scenario `[scenarios.calibration]` overrides it, and the
//! separate scale slows a decode-phase scenario.

use super::*;
use std::{fs, time::SystemTime};

const CLUSTER: &str = r#"
schema_version = 1
[cluster]
preset = "custom"
[interconnect]
kind = "ethernet"
variant = "10g"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 1
hbm_gb = 24.0
hbm_bandwidth_gb_s = 936.0
peak_f16_tflops = 88.0
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
"#;

const WORKLOAD: &str = r#"
schema_version = 1
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
batch_size = 32
prompt_tokens = 512
decode_tokens = 128
max_sequence_tokens = 640
phase = "decode"
[calibration]
compute_efficiency = 0.8494
decode_memory_bandwidth_scale = 0.8383
[search]
tensor_ranks = [1]
pipeline_ranks = [1]
expert_ranks = [1]
data_ranks = [1]
"#;

fn run_scenarios(scenarios: &str) -> serde_json::Value {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir();
    let cluster_path = dir.join(format!("inference-sim-kv-read-cluster-{nanos}.toml"));
    let workload_path = dir.join(format!("inference-sim-kv-read-workload-{nanos}.toml"));
    let run_path = dir.join(format!("inference-sim-kv-read-run-{nanos}.toml"));
    fs::write(&cluster_path, CLUSTER).unwrap();
    fs::write(&workload_path, WORKLOAD).unwrap();
    fs::write(
        &run_path,
        format!(
            "schema_version = 1\ncluster = \"{}\"\nworkload = \"{}\"\n[output]\ntop_k = 1\nformat = \"json\"\n{scenarios}",
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
    for path in [&cluster_path, &workload_path, &run_path] {
        let _ = fs::remove_file(path);
    }
    serde_json::from_str(&String::from_utf8(output).unwrap()).unwrap()
}

fn scenario<'a>(parsed: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    parsed["scenarios"]
        .as_array()
        .expect("scenarios array")
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("scenario {name} missing"))
}

fn find_key<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    match value {
        serde_json::Value::Object(map) => map
            .get(key)
            .or_else(|| map.values().find_map(|value| find_key(value, key))),
        serde_json::Value::Array(items) => items.iter().find_map(|value| find_key(value, key)),
        _ => None,
    }
}

#[test]
fn scenario_override_reports_and_applies_the_kv_read_scale() {
    let parsed = run_scenarios(
        r#"
[[scenarios]]
name = "shared"

[[scenarios]]
name = "paged-kv"
[scenarios.calibration]
kv_read_bandwidth_scale = 0.45
"#,
    );
    let shared = scenario(&parsed, "shared");
    let paged = scenario(&parsed, "paged-kv");
    assert_eq!(
        find_key(shared, "kv_read_bandwidth_scale"),
        Some(&serde_json::Value::Null)
    );
    assert_eq!(
        find_key(paged, "kv_read_bandwidth_scale").and_then(serde_json::Value::as_f64),
        Some(0.45)
    );
    let latency = |entry: &serde_json::Value| {
        find_key(entry, "estimated_latency_ms")
            .and_then(serde_json::Value::as_f64)
            .expect("latency")
    };
    assert!(
        latency(paged) > latency(shared),
        "{} vs {}",
        latency(paged),
        latency(shared)
    );
}
