//! TOML parsing and validation of the disaggregated handoff calibration
//! keys: `kv_link_sharing` and `disaggregation_proxy_latency_us`.

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
fn handoff_keys_default_to_fifo_without_proxy_latency() {
    let workload = workload_with_calibration("compute_efficiency = 0.8").expect("parses");
    assert_eq!(workload.calibration.kv_link_sharing, KvLinkSharing::Fifo);
    assert_eq!(workload.calibration.disaggregation_proxy_latency_us, 0.0);
}

#[test]
fn parses_fair_share_links_and_proxy_latency() {
    let workload = workload_with_calibration(
        "kv_link_sharing = \"fair_share\"\ndisaggregation_proxy_latency_us = 40000",
    )
    .expect("parses");
    assert_eq!(
        workload.calibration.kv_link_sharing,
        KvLinkSharing::FairShare
    );
    assert_eq!(
        workload.calibration.disaggregation_proxy_latency_us,
        40_000.0
    );
    let fifo = workload_with_calibration("kv_link_sharing = \"fifo\"").expect("parses");
    assert_eq!(fifo.calibration.kv_link_sharing, KvLinkSharing::Fifo);
}

#[test]
fn rejects_unknown_link_sharing() {
    let err = error_text(workload_with_calibration(
        "kv_link_sharing = \"round_robin\"",
    ));
    assert!(err.contains("calibration.kv_link_sharing"), "{err}");
    assert!(err.contains("fair_share"), "{err}");
}

#[test]
fn rejects_negative_proxy_latency() {
    let err = error_text(workload_with_calibration(
        "disaggregation_proxy_latency_us = -1.0",
    ));
    assert!(
        err.contains("calibration.disaggregation_proxy_latency_us"),
        "{err}"
    );
}
