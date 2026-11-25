//! Scenario overlays that modify the network suspend measured collective
//! curves; overlays that leave the fabric intact keep them.

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
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
[[nodes]]
id = 1
gpu = "a100_40gb"
gpu_count = 1
nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
[[collective_curves]]
op = "all_reduce"
scope = "node_group"
nodes = [0, 1]
ranks = 2
points = [[1024, 130.0], [1048576, 1800.0]]
"#;

const WORKLOAD: &str = r#"
schema_version = 1
[model]
layers = 4
hidden_size = 4096
attention_heads = 32
kv_heads = 8
vocab_size = 32000
parameters_gb = 4.0
dtype = "bf16"
[request]
batch_size = 8
prompt_tokens = 64
decode_tokens = 8
max_sequence_tokens = 128
phase = "decode"
[search]
tensor_ranks = [2]
pipeline_ranks = [1]
expert_ranks = [1]
data_ranks = [1]
"#;

fn run_scenarios(scenarios: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir();
    let cluster_path = dir.join(format!("inference-sim-curve-cluster-{nanos}.toml"));
    let workload_path = dir.join(format!("inference-sim-curve-workload-{nanos}.toml"));
    let run_path = dir.join(format!("inference-sim-curve-run-{nanos}.toml"));
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
    String::from_utf8(output).unwrap()
}

fn scenario_codes(output: &str, scenario: &str) -> Vec<String> {
    let parsed: serde_json::Value = serde_json::from_str(output).unwrap();
    let scenarios = parsed["scenarios"].as_array().expect("scenarios array");
    let entry = scenarios
        .iter()
        .find(|entry| entry["name"] == scenario)
        .unwrap_or_else(|| panic!("scenario {scenario} missing"));
    let mut codes = Vec::new();
    collect_codes(entry, &mut codes);
    codes
}

fn collect_codes(value: &serde_json::Value, codes: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            if let (Some(serde_json::Value::String(code)), Some(_)) =
                (map.get("code"), map.get("category"))
            {
                codes.push(code.clone());
            }
            map.values().for_each(|value| collect_codes(value, codes));
        }
        serde_json::Value::Array(items) => {
            items.iter().for_each(|value| collect_codes(value, codes))
        }
        _ => {}
    }
}

#[test]
fn network_overlays_suspend_curves_and_node_overlays_keep_them() {
    let output = run_scenarios(
        r#"
[[scenarios]]
name = "baseline"

[[scenarios]]
name = "slow-nics"
[scenarios.topology]
nic_bandwidth_scale = 0.5
"#,
    );
    let baseline = scenario_codes(&output, "baseline");
    assert!(
        baseline
            .iter()
            .any(|code| code == "measured_collective_curve"),
        "{baseline:?}"
    );
    assert!(
        !baseline
            .iter()
            .any(|code| code == "collective_curves_suspended")
    );
    let slow = scenario_codes(&output, "slow-nics");
    assert!(
        slow.iter()
            .any(|code| code == "collective_curves_suspended"),
        "{slow:?}"
    );
    assert!(slow.iter().any(|code| code == "coarse_collective_model"));
    assert!(!slow.iter().any(|code| code == "measured_collective_curve"));
}
