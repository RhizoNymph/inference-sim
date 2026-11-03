//! Cluster-TOML parsing of the GPU clock keys: sustained clock, bandwidth
//! knee, and per-GPU locks layered through node, `gpus`, and
//! `gpu_profile_overrides` entries.

use super::*;
use crate::types::gpu_clock::{BandwidthClockSensitivity, GpuClockState};

const HEADER: &str = r#"
schema_version = 1

[cluster]
preset = "custom"

[interconnect]
kind = "ethernet"
variant = "10g"
oversubscription = 1.0
"#;

fn cluster_with_node(node: &str) -> Result<Cluster, ConfigError> {
    parse_cluster(&format!("{HEADER}\n{node}\n"))
}

fn error_text(result: Result<Cluster, ConfigError>) -> String {
    match result {
        Ok(_) => panic!("expected a config error"),
        Err(err) => err.to_string(),
    }
}

const RTX3090_NODE: &str = r#"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 2
hbm_gb = 24.0
hbm_bandwidth_gb_s = 936.0
peak_f16_tflops = 88.0
sustained_sm_clock_mhz = 1835.0
bandwidth_knee_sm_clock_mhz = 1512.0
intra = "pcie_gen4"
"#;

#[test]
fn absent_clock_keys_leave_the_profile_without_a_clock() {
    let cluster = cluster_with_node(
        r#"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 1
peak_f16_tflops = 88.0
"#,
    )
    .expect("valid cluster");
    let profile = cluster.nodes[&0].gpu_profile(0).expect("gpu 0");
    assert_eq!(profile.clock, None);
    assert!(profile.clock_degradation().is_nominal());
}

#[test]
fn node_wide_sustained_clock_and_knee_parse_as_an_unlocked_model() {
    let cluster = cluster_with_node(RTX3090_NODE).expect("valid cluster");
    for gpu in 0..2 {
        let clock = cluster.nodes[&0]
            .gpu_profile(gpu)
            .and_then(|profile| profile.clock)
            .expect("clock model");
        assert_eq!(clock.model().sustained().mhz(), 1835.0);
        assert_eq!(
            clock.model().bandwidth_knee().map(|knee| knee.mhz()),
            Some(1512.0)
        );
        assert_eq!(clock.state(), GpuClockState::Unlocked);
        assert!(clock.degradation().is_nominal());
    }
}

#[test]
fn per_gpu_override_locks_only_that_gpu_and_inherits_the_node_model() {
    let cluster = cluster_with_node(&format!(
        "{RTX3090_NODE}\n[[nodes.gpu_profile_overrides]]\ngpu = 1\nlocked_sm_clock_mhz = 1200.0\n"
    ))
    .expect("valid cluster");
    let node = &cluster.nodes[&0];
    let healthy = node.gpu_profile(0).expect("gpu 0");
    let locked = node.gpu_profile(1).expect("gpu 1");
    assert!(healthy.clock_degradation().is_nominal());
    let clock = locked.clock.expect("clock");
    assert_eq!(clock.locked_clock().map(|mhz| mhz.mhz()), Some(1200.0));
    assert_eq!(clock.model().sustained().mhz(), 1835.0);
    let degradation = locked.clock_degradation();
    assert!((degradation.compute() - 1200.0 / 1835.0).abs() < 1e-12);
    assert!((degradation.bandwidth() - 1200.0 / 1512.0).abs() < 1e-12);
    // The override keeps the node-wide peaks.
    assert_eq!(locked.peak_f16_flops, 88.0);
}

#[test]
fn explicit_gpu_entries_accept_clock_keys() {
    let cluster = cluster_with_node(
        r#"
[[nodes]]
id = 0
intra = "pcie_gen4"
[[nodes.gpus]]
id = 0
gpu = "a100_40gb"
peak_f16_tflops = 88.0
sustained_sm_clock_mhz = 1835.0
locked_sm_clock_mhz = 1200.0
"#,
    )
    .expect("valid cluster");
    let clock = cluster.nodes[&0]
        .gpu_profile(0)
        .and_then(|profile| profile.clock)
        .expect("clock");
    assert_eq!(
        clock.model().bandwidth(),
        BandwidthClockSensitivity::Insensitive
    );
    assert_eq!(clock.degradation().bandwidth(), 1.0);
}

#[test]
fn node_groups_accept_clock_keys() {
    let cluster = cluster_with_node(
        r#"
[[node_groups]]
label = "slow"
count = 2
gpu = "a100_40gb"
gpu_count = 1
sustained_sm_clock_mhz = 1400.0
locked_sm_clock_mhz = 700.0
"#,
    )
    .expect("valid cluster");
    for node in cluster.nodes.values() {
        let degradation = node.gpu_profile(0).expect("gpu").clock_degradation();
        assert!((degradation.compute() - 0.5).abs() < 1e-12);
    }
}

#[test]
fn lock_without_sustained_clock_is_rejected() {
    let text = error_text(cluster_with_node(
        r#"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 1
locked_sm_clock_mhz = 1200.0
"#,
    ));
    assert!(
        text.contains("locked_sm_clock_mhz requires sustained_sm_clock_mhz"),
        "{text}"
    );
}

#[test]
fn knee_without_sustained_clock_is_rejected() {
    let text = error_text(cluster_with_node(
        r#"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 1
bandwidth_knee_sm_clock_mhz = 1500.0
"#,
    ));
    assert!(
        text.contains("bandwidth_knee_sm_clock_mhz requires sustained_sm_clock_mhz"),
        "{text}"
    );
}

#[test]
fn knee_above_sustained_clock_is_rejected() {
    let text = error_text(cluster_with_node(
        r#"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 1
sustained_sm_clock_mhz = 1500.0
bandwidth_knee_sm_clock_mhz = 1700.0
"#,
    ));
    assert!(text.contains("exceeds the sustained SM clock"), "{text}");
}

#[test]
fn non_positive_clocks_are_rejected() {
    for key in [
        "sustained_sm_clock_mhz = 0.0",
        "sustained_sm_clock_mhz = 1800.0\nlocked_sm_clock_mhz = -5.0",
        "sustained_sm_clock_mhz = 1800.0\nbandwidth_knee_sm_clock_mhz = 0.0",
    ] {
        let text = error_text(cluster_with_node(&format!(
            "[[nodes]]\nid = 0\ngpu = \"a100_40gb\"\ngpu_count = 1\n{key}\n"
        )));
        assert!(text.contains("must be positive"), "{key}: {text}");
    }
}

#[test]
fn override_lock_without_any_node_model_is_rejected() {
    let text = error_text(cluster_with_node(
        r#"
[[nodes]]
id = 0
gpu = "a100_40gb"
gpu_count = 2
[[nodes.gpu_profile_overrides]]
gpu = 1
locked_sm_clock_mhz = 1200.0
"#,
    ));
    assert!(
        text.contains("gpu_profile_overrides[0].locked_sm_clock_mhz"),
        "{text}"
    );
}
