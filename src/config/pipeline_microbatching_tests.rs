//! TOML parsing of `calibration.pipeline_microbatching`.

use super::*;
use crate::calibration::PipelineMicrobatching;

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
batch_size = 8
prompt_tokens = 512
decode_tokens = 1
max_sequence_tokens = 513
phase = "prefill"
"#;

fn workload_with_calibration(calibration: &str) -> Result<WorkloadConfig, ConfigError> {
    parse_workload(&format!(
        "{MODEL_AND_REQUEST}\n[calibration]\n{calibration}\n"
    ))
}

#[test]
fn pipeline_microbatching_defaults_to_whole_batch() {
    let workload = workload_with_calibration("compute_efficiency = 0.85").expect("valid");
    assert_eq!(
        workload.calibration.pipeline_microbatching,
        PipelineMicrobatching::WholeBatch
    );
    assert_eq!(workload.calibration_overrides.pipeline_microbatching, None);
}

#[test]
fn parses_vllm_batch_queue_microbatching() {
    let workload = workload_with_calibration("pipeline_microbatching = \"vllm_batch_queue\"")
        .expect("valid policy");
    assert_eq!(
        workload.calibration.pipeline_microbatching,
        PipelineMicrobatching::VllmBatchQueue
    );
    assert_eq!(
        workload.calibration_overrides.pipeline_microbatching,
        Some(PipelineMicrobatching::VllmBatchQueue)
    );
    let workload = workload_with_calibration("pipeline_microbatching = \"whole_batch\"")
        .expect("valid policy");
    assert_eq!(
        workload.calibration.pipeline_microbatching,
        PipelineMicrobatching::WholeBatch
    );
}

#[test]
fn rejects_an_unknown_microbatching_policy() {
    let err = match workload_with_calibration("pipeline_microbatching = \"gpipe\"") {
        Ok(_) => panic!("unknown policy accepted"),
        Err(err) => err.to_string(),
    };
    assert!(
        err.contains("pipeline_microbatching") || err.contains("gpipe"),
        "{err}"
    );
}

#[test]
fn scenario_calibration_overrides_the_policy() {
    let parsed = parse_run_config(
        "schema_version = 1\ncluster = \"cluster.toml\"\nworkload = \"workload.toml\"\n[[scenarios]]\nname = \"s\"\n[scenarios.calibration]\npipeline_microbatching = \"vllm_batch_queue\"\n",
    )
    .expect("valid scenario");
    assert_eq!(
        parsed.scenarios[0].calibration.pipeline_microbatching,
        Some(PipelineMicrobatching::VllmBatchQueue)
    );
}

#[test]
fn policy_names_round_trip() {
    for policy in [
        PipelineMicrobatching::WholeBatch,
        PipelineMicrobatching::VllmBatchQueue,
    ] {
        let workload =
            workload_with_calibration(&format!("pipeline_microbatching = \"{}\"", policy.as_str()))
                .expect("valid policy");
        assert_eq!(workload.calibration.pipeline_microbatching, policy);
    }
}
