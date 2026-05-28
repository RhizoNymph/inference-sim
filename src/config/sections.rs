use super::*;

mod serving;
pub(super) use serving::*;

#[derive(Deserialize)]
pub(super) struct ClusterFile {
    pub(super) schema_version: Option<u32>,
    pub(super) cluster: ClusterSection,
    pub(super) interconnect: Option<InterconnectSection>,
    pub(super) nics: Option<NicsSection>,
    pub(super) nodes: Option<Vec<NodeSection>>,
    pub(super) node_groups: Option<Vec<NodeGroupSection>>,
}

#[derive(Deserialize)]
pub(super) struct ClusterSection {
    pub(super) preset: String,
    pub(super) node_count: Option<u32>,
}

#[derive(Deserialize)]
pub(super) struct InterconnectSection {
    pub(super) kind: Option<String>,
    pub(super) variant: Option<String>,
    pub(super) oversubscription: Option<f64>,
    pub(super) links: Option<Vec<InterconnectLinkSection>>,
}

#[derive(Deserialize)]
pub(super) struct InterconnectLinkSection {
    pub(super) from: Option<u32>,
    pub(super) to: Option<u32>,
    pub(super) from_group: Option<String>,
    pub(super) to_group: Option<String>,
    #[serde(alias = "from_node_label", alias = "from_node_tag")]
    pub(super) from_node_tag: Option<String>,
    #[serde(
        alias = "from_node_labels",
        alias = "from_node_tags",
        alias = "from_labels",
        alias = "from_tags"
    )]
    pub(super) from_node_tags: Option<Vec<String>>,
    #[serde(alias = "from_rack_id")]
    pub(super) from_rack: Option<String>,
    #[serde(alias = "from_rack_ids")]
    pub(super) from_racks: Option<Vec<String>>,
    #[serde(alias = "from_island_id", alias = "from_topology_domain")]
    pub(super) from_island: Option<String>,
    #[serde(alias = "from_island_ids", alias = "from_topology_domains")]
    pub(super) from_islands: Option<Vec<String>>,
    #[serde(alias = "from_failure_domain_id")]
    pub(super) from_failure_domain: Option<String>,
    #[serde(alias = "from_failure_domain_ids")]
    pub(super) from_failure_domains: Option<Vec<String>>,
    #[serde(alias = "to_node_label", alias = "to_node_tag")]
    pub(super) to_node_tag: Option<String>,
    #[serde(
        alias = "to_node_labels",
        alias = "to_node_tags",
        alias = "to_labels",
        alias = "to_tags"
    )]
    pub(super) to_node_tags: Option<Vec<String>>,
    #[serde(alias = "to_rack_id")]
    pub(super) to_rack: Option<String>,
    #[serde(alias = "to_rack_ids")]
    pub(super) to_racks: Option<Vec<String>>,
    #[serde(alias = "to_island_id", alias = "to_topology_domain")]
    pub(super) to_island: Option<String>,
    #[serde(alias = "to_island_ids", alias = "to_topology_domains")]
    pub(super) to_islands: Option<Vec<String>>,
    #[serde(alias = "to_failure_domain_id")]
    pub(super) to_failure_domain: Option<String>,
    #[serde(alias = "to_failure_domain_ids")]
    pub(super) to_failure_domains: Option<Vec<String>>,
    #[serde(alias = "from_gpu", alias = "from_gpu_id", alias = "from_local_gpu")]
    pub(super) from_gpu: Option<u32>,
    #[serde(
        alias = "from_gpu_ids",
        alias = "from_local_gpus",
        alias = "from_local_gpu_ids"
    )]
    pub(super) from_gpus: Option<Vec<u32>>,
    #[serde(alias = "from_gpu_label", alias = "from_gpu_type")]
    pub(super) from_gpu_type: Option<String>,
    #[serde(alias = "from_gpu_labels", alias = "from_gpu_types")]
    pub(super) from_gpu_types: Option<Vec<String>>,
    #[serde(alias = "from_gpu_topology_label")]
    pub(super) from_gpu_tag: Option<String>,
    #[serde(alias = "from_gpu_topology_labels")]
    pub(super) from_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "to_gpu", alias = "to_gpu_id", alias = "to_local_gpu")]
    pub(super) to_gpu: Option<u32>,
    #[serde(
        alias = "to_gpu_ids",
        alias = "to_local_gpus",
        alias = "to_local_gpu_ids"
    )]
    pub(super) to_gpus: Option<Vec<u32>>,
    #[serde(alias = "to_gpu_label", alias = "to_gpu_type")]
    pub(super) to_gpu_type: Option<String>,
    #[serde(alias = "to_gpu_labels", alias = "to_gpu_types")]
    pub(super) to_gpu_types: Option<Vec<String>>,
    #[serde(alias = "to_gpu_topology_label")]
    pub(super) to_gpu_tag: Option<String>,
    #[serde(alias = "to_gpu_topology_labels")]
    pub(super) to_gpu_tags: Option<Vec<String>>,
    pub(super) kind: String,
    pub(super) variant: String,
    pub(super) oversubscription: Option<f64>,
    pub(super) rail: Option<u32>,
    pub(super) rails: Option<Vec<u32>>,
}

#[derive(Clone, Deserialize)]
pub(super) struct NicsSection {
    pub(super) count: Option<u8>,
    pub(super) bandwidth_gbps: Option<f64>,
    pub(super) affinity: Option<String>,
    pub(super) gpus_per_nic: Option<u8>,
    pub(super) rail_count: Option<u8>,
    #[serde(alias = "nic_rails", alias = "nic_to_rail_map", alias = "rail_map")]
    pub(super) nic_rail_map: Option<Vec<NicRailMapSection>>,
    #[serde(alias = "gpu_to_nic_map", alias = "gpu_nic_locality")]
    pub(super) gpu_nic_map: Option<Vec<GpuNicMapSection>>,
    #[serde(
        alias = "gpu_socket_map",
        alias = "gpu_numa_domains",
        alias = "gpu_socket_domains"
    )]
    pub(super) gpu_numa_map: Option<Vec<GpuNumaMapSection>>,
    #[serde(
        alias = "nic_socket_map",
        alias = "nic_numa_domains",
        alias = "nic_socket_domains"
    )]
    pub(super) nic_numa_map: Option<Vec<NicNumaMapSection>>,
    #[serde(alias = "cross_socket_bandwidth_scale")]
    pub(super) cross_numa_bandwidth_scale: Option<f64>,
    #[serde(alias = "cross_socket_latency_scale")]
    pub(super) cross_numa_latency_scale: Option<f64>,
    #[serde(
        alias = "gpu_nic_path_overrides",
        alias = "gpu_to_nic_paths",
        alias = "gpu_nic_locality_overrides"
    )]
    pub(super) gpu_nic_paths: Option<Vec<GpuNicPathSection>>,
    #[serde(alias = "bandwidth_overrides", alias = "per_nic_bandwidth")]
    pub(super) nic_bandwidth_overrides: Option<Vec<NicBandwidthOverrideSection>>,
    #[serde(alias = "latency_overrides", alias = "per_nic_latency")]
    pub(super) nic_latency_scale_overrides: Option<Vec<NicLatencyScaleOverrideSection>>,
    #[serde(alias = "offline_nics", alias = "unavailable_nics")]
    pub(super) disabled_nics: Option<Vec<u32>>,
    #[serde(alias = "nic_state", alias = "nics_state", alias = "nic_health")]
    pub(super) nic_states: Option<Vec<NicStateSection>>,
}

#[derive(Clone, Deserialize)]
pub(super) struct NicRailMapSection {
    #[serde(alias = "nic_id", alias = "id")]
    pub(super) nic: Option<u32>,
    #[serde(alias = "rail_id")]
    pub(super) rail: Option<u32>,
}

#[derive(Clone, Deserialize)]
pub(super) struct GpuNicMapSection {
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_id")]
    pub(super) local_gpu_id: Option<u32>,
    pub(super) nic: Option<u32>,
    pub(super) nics: Option<Vec<u32>>,
}

#[derive(Clone, Deserialize)]
pub(super) struct GpuNumaMapSection {
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_id")]
    pub(super) local_gpu_id: Option<u32>,
    #[serde(alias = "gpus", alias = "gpu_ids", alias = "local_gpu_ids")]
    pub(super) local_gpu_ids: Option<Vec<u32>>,
    #[serde(
        alias = "domain",
        alias = "numa",
        alias = "numa_id",
        alias = "socket",
        alias = "socket_id"
    )]
    pub(super) numa_domain: u32,
}

#[derive(Clone, Deserialize)]
pub(super) struct NicNumaMapSection {
    #[serde(alias = "nic_id", alias = "id")]
    pub(super) nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    pub(super) nics: Option<Vec<u32>>,
    #[serde(
        alias = "domain",
        alias = "numa",
        alias = "numa_id",
        alias = "socket",
        alias = "socket_id"
    )]
    pub(super) numa_domain: u32,
}

#[derive(Clone, Deserialize)]
pub(super) struct GpuNicPathSection {
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_id")]
    pub(super) local_gpu_id: Option<u32>,
    pub(super) nic: Option<u32>,
    pub(super) label: Option<String>,
    pub(super) bandwidth_gbps: Option<f64>,
    pub(super) latency_us: Option<f64>,
    pub(super) gpudirect: Option<bool>,
    pub(super) available: Option<bool>,
}

#[derive(Clone, Deserialize)]
pub(super) struct NicStateSection {
    #[serde(alias = "nic_id", alias = "id")]
    pub(super) nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    pub(super) nics: Option<Vec<u32>>,
    pub(super) state: String,
}

#[derive(Clone, Deserialize)]
pub(super) struct NicBandwidthOverrideSection {
    #[serde(alias = "nic_id", alias = "id")]
    pub(super) nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    pub(super) nics: Option<Vec<u32>>,
    pub(super) bandwidth_gbps: f64,
}

#[derive(Clone, Deserialize)]
pub(super) struct NicLatencyScaleOverrideSection {
    #[serde(alias = "nic_id", alias = "id")]
    pub(super) nic: Option<u32>,
    #[serde(alias = "nic_ids", alias = "ids")]
    pub(super) nics: Option<Vec<u32>>,
    #[serde(alias = "scale")]
    pub(super) latency_scale: f64,
}

#[derive(Clone, Deserialize)]
pub(super) struct NodeSection {
    pub(super) id: u32,
    pub(super) group: Option<String>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(
        alias = "labels",
        alias = "node_labels",
        alias = "tags",
        alias = "node_tags"
    )]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    pub(super) gpu: Option<String>,
    pub(super) gpu_count: Option<u32>,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    pub(super) hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    pub(super) hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    pub(super) peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    pub(super) peak_f8_tflops: Option<f64>,
    #[serde(alias = "gpu_label", alias = "gpu_tag")]
    pub(super) gpu_tag: Option<String>,
    #[serde(alias = "gpu_labels", alias = "gpu_tags")]
    pub(super) gpu_tags: Option<Vec<String>>,
    pub(super) gpus: Option<Vec<NodeGpuSection>>,
    #[serde(alias = "gpu_overrides", alias = "per_gpu_profiles")]
    pub(super) gpu_profile_overrides: Option<Vec<GpuProfileOverrideSection>>,
    #[serde(alias = "offline_gpus", alias = "unavailable_gpus")]
    pub(super) disabled_gpus: Option<Vec<u32>>,
    #[serde(alias = "gpu_state", alias = "gpus_state", alias = "gpu_health")]
    pub(super) gpu_states: Option<Vec<GpuStateSection>>,
    pub(super) intra: Option<String>,
    pub(super) nics: Option<NicsSection>,
}

#[derive(Clone, Deserialize)]
pub(super) struct NodeGpuSection {
    #[serde(alias = "local_id", alias = "local_gpu_id")]
    pub(super) id: Option<u32>,
    #[serde(alias = "start", alias = "start_id", alias = "id_start")]
    pub(super) start_id: Option<u32>,
    #[serde(alias = "type")]
    pub(super) gpu: String,
    pub(super) count: Option<u32>,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    pub(super) hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    pub(super) hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    pub(super) peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    pub(super) peak_f8_tflops: Option<f64>,
    #[serde(alias = "label", alias = "gpu_label", alias = "tag", alias = "gpu_tag")]
    pub(super) gpu_tag: Option<String>,
    #[serde(
        alias = "labels",
        alias = "gpu_labels",
        alias = "tags",
        alias = "gpu_tags"
    )]
    pub(super) gpu_tags: Option<Vec<String>>,
    #[serde(alias = "health")]
    pub(super) state: Option<String>,
}

#[derive(Clone, Deserialize)]
pub(super) struct GpuProfileOverrideSection {
    #[serde(
        alias = "gpu",
        alias = "gpu_id",
        alias = "local_id",
        alias = "local_gpu_id"
    )]
    pub(super) local_gpu_id: Option<u32>,
    #[serde(alias = "gpus", alias = "gpu_ids", alias = "local_gpu_ids")]
    pub(super) local_gpu_ids: Option<Vec<u32>>,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    pub(super) hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    pub(super) hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    pub(super) peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    pub(super) peak_f8_tflops: Option<f64>,
}

#[derive(Clone, Deserialize)]
pub(super) struct GpuStateSection {
    #[serde(
        alias = "gpu",
        alias = "gpu_id",
        alias = "local_id",
        alias = "local_gpu_id"
    )]
    pub(super) local_gpu_id: Option<u32>,
    #[serde(alias = "gpus", alias = "gpu_ids", alias = "local_gpu_ids")]
    pub(super) local_gpu_ids: Option<Vec<u32>>,
    pub(super) state: String,
}

#[derive(Clone, Deserialize)]
pub(super) struct NodeGroupSection {
    pub(super) label: Option<String>,
    #[serde(alias = "node_count")]
    pub(super) count: u32,
    #[serde(alias = "id_start")]
    pub(super) start_id: Option<u32>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(
        alias = "labels",
        alias = "node_labels",
        alias = "tags",
        alias = "node_tags"
    )]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    pub(super) gpu: String,
    pub(super) gpu_count: u32,
    #[serde(alias = "hbm_size_gb", alias = "memory_gb")]
    pub(super) hbm_gb: Option<f64>,
    #[serde(alias = "hbm_bandwidth_gbs")]
    pub(super) hbm_bandwidth_gb_s: Option<f64>,
    #[serde(
        alias = "f16_tflops",
        alias = "bf16_tflops",
        alias = "peak_bf16_tflops"
    )]
    pub(super) peak_f16_tflops: Option<f64>,
    #[serde(alias = "f8_tflops", alias = "fp8_tflops", alias = "peak_fp8_tflops")]
    pub(super) peak_f8_tflops: Option<f64>,
    #[serde(alias = "gpu_label", alias = "gpu_tag")]
    pub(super) gpu_tag: Option<String>,
    #[serde(alias = "gpu_labels", alias = "gpu_tags")]
    pub(super) gpu_tags: Option<Vec<String>>,
    #[serde(alias = "gpu_overrides", alias = "per_gpu_profiles")]
    pub(super) gpu_profile_overrides: Option<Vec<GpuProfileOverrideSection>>,
    #[serde(alias = "offline_gpus", alias = "unavailable_gpus")]
    pub(super) disabled_gpus: Option<Vec<u32>>,
    #[serde(alias = "gpu_state", alias = "gpus_state", alias = "gpu_health")]
    pub(super) gpu_states: Option<Vec<GpuStateSection>>,
    pub(super) intra: Option<String>,
    pub(super) nics: Option<NicsSection>,
}

#[derive(Deserialize)]
pub(super) struct WorkloadFile {
    pub(super) schema_version: Option<u32>,
    #[serde(alias = "runtime", alias = "backend", alias = "stack")]
    pub(super) serving_stack: Option<String>,
    #[serde(
        alias = "runtime_features",
        alias = "backend_features",
        alias = "stack_features"
    )]
    pub(super) serving_runtime_features: Option<Vec<String>>,
    pub(super) model: ModelSection,
    pub(super) request: RequestSection,
    pub(super) search: Option<SearchSection>,
    pub(super) placement: Option<PlacementSection>,
    pub(super) serving: Option<ServingSection>,
    pub(super) calibration_profile: Option<CalibrationProfileReferenceSection>,
    pub(super) calibration: Option<CalibrationSection>,
    pub(super) calibration_policy: Option<CalibrationPolicySection>,
    pub(super) approximation_policy: Option<ApproximationPolicySection>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationProfileReferenceSection {
    pub(super) path: String,
}

#[derive(Deserialize)]
pub(super) struct RunFile {
    pub(super) schema_version: Option<u32>,
    pub(super) cluster: Option<String>,
    pub(super) workload: Option<String>,
    pub(super) run: Option<RunSection>,
    pub(super) output: Option<RunOutputSection>,
    #[serde(alias = "budget", alias = "search_budget")]
    pub(super) search: Option<RunSearchBudgetSection>,
    pub(super) scenarios: Option<Vec<RunScenarioSection>>,
}

#[derive(Deserialize)]
pub(super) struct RunSection {
    #[serde(alias = "cluster_path")]
    pub(super) cluster: Option<String>,
    #[serde(alias = "workload_path")]
    pub(super) workload: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct RunOutputSection {
    pub(super) format: Option<String>,
    pub(super) top_k: Option<usize>,
    #[serde(alias = "dir", alias = "directory")]
    pub(super) output_dir: Option<String>,
    pub(super) output_profile: Option<String>,
    #[serde(
        alias = "request_metrics_csv",
        alias = "serving_request_metrics_csv",
        alias = "metrics_csv"
    )]
    pub(super) request_metrics_csv_path: Option<String>,
    #[serde(
        alias = "request_lifecycle_events_csv",
        alias = "serving_request_lifecycle_events_csv",
        alias = "lifecycle_events_csv",
        alias = "serving_lifecycle_events_csv"
    )]
    pub(super) request_lifecycle_events_csv_path: Option<String>,
    #[serde(
        alias = "serving_metrics_csv",
        alias = "serving_candidate_metrics_csv",
        alias = "candidate_metrics_csv"
    )]
    pub(super) serving_metrics_csv_path: Option<String>,
    #[serde(
        alias = "serving_metric_breakdowns_csv",
        alias = "serving_breakdowns_csv",
        alias = "metric_breakdowns_csv",
        alias = "breakdowns_csv"
    )]
    pub(super) serving_metric_breakdowns_csv_path: Option<String>,
    #[serde(
        alias = "serving_services_csv",
        alias = "serving_service_metrics_csv",
        alias = "service_metrics_csv",
        alias = "services_csv"
    )]
    pub(super) serving_services_csv_path: Option<String>,
    #[serde(
        alias = "serving_utilization_csv",
        alias = "serving_resource_utilization_csv",
        alias = "resource_utilization_csv",
        alias = "utilization_csv"
    )]
    pub(super) serving_utilization_csv_path: Option<String>,
    #[serde(
        alias = "serving_memory_pressure_csv",
        alias = "serving_hbm_pressure_csv",
        alias = "memory_pressure_csv",
        alias = "hbm_pressure_csv"
    )]
    pub(super) serving_memory_pressure_csv_path: Option<String>,
    #[serde(
        alias = "serving_timeline_csv",
        alias = "serving_scheduled_operations_csv",
        alias = "scheduled_operations_csv",
        alias = "timeline_csv"
    )]
    pub(super) serving_timeline_csv_path: Option<String>,
    #[serde(
        alias = "serving_occupancy_csv",
        alias = "serving_resource_occupancy_csv",
        alias = "resource_occupancy_csv",
        alias = "occupancy_csv"
    )]
    pub(super) serving_occupancy_csv_path: Option<String>,
    #[serde(
        alias = "serving_placement_evidence_csv",
        alias = "serving_placement_csv",
        alias = "placement_evidence_csv",
        alias = "placement_csv"
    )]
    pub(super) serving_placement_evidence_csv_path: Option<String>,
    #[serde(
        alias = "serving_worker_evidence_csv",
        alias = "serving_worker_assignments_csv",
        alias = "worker_evidence_csv",
        alias = "worker_assignments_csv"
    )]
    pub(super) serving_worker_evidence_csv_path: Option<String>,
    #[serde(
        alias = "serving_rejections_csv",
        alias = "serving_rejection_evidence_csv",
        alias = "rejections_csv",
        alias = "rejection_evidence_csv"
    )]
    pub(super) serving_rejections_csv_path: Option<String>,
    #[serde(
        alias = "serving_route_paths_csv",
        alias = "serving_kv_route_paths_csv",
        alias = "kv_route_paths_csv",
        alias = "route_paths_csv"
    )]
    pub(super) serving_route_paths_csv_path: Option<String>,
    #[serde(
        alias = "kv_route_resources_csv",
        alias = "kv_route_resource_csv",
        alias = "route_resources_csv",
        alias = "serving_route_resources_csv"
    )]
    pub(super) kv_route_resources_csv_path: Option<String>,
    #[serde(
        alias = "serving_bottlenecks_csv",
        alias = "serving_bottleneck_summary_csv",
        alias = "bottlenecks_csv"
    )]
    pub(super) serving_bottlenecks_csv_path: Option<String>,
    #[serde(
        alias = "serving_phase_calibration_csv",
        alias = "serving_calibration_phases_csv",
        alias = "phase_calibration_csv"
    )]
    pub(super) serving_phase_calibration_csv_path: Option<String>,
    #[serde(
        alias = "serving_approximations_csv",
        alias = "serving_approximation_evidence_csv",
        alias = "approximations_csv",
        alias = "approximation_evidence_csv"
    )]
    pub(super) serving_approximations_csv_path: Option<String>,
    #[serde(
        alias = "calibration_residuals_csv",
        alias = "calibration_benchmark_residuals_csv",
        alias = "calibration_benchmarks_csv"
    )]
    pub(super) calibration_residuals_csv_path: Option<String>,
    #[serde(
        alias = "scenario_sensitivity_csv",
        alias = "serving_scenario_sensitivity_csv",
        alias = "sensitivity_csv"
    )]
    pub(super) scenario_sensitivity_csv_path: Option<String>,
    #[serde(
        alias = "rank_sensitivity_csv",
        alias = "solver_rank_sensitivity_csv",
        alias = "serving_rank_sensitivity_csv"
    )]
    pub(super) rank_sensitivity_csv_path: Option<String>,
    pub(super) trace: Option<bool>,
    pub(super) trace_limit: Option<usize>,
    pub(super) request_limit: Option<usize>,
    pub(super) occupancy: Option<bool>,
    pub(super) occupancy_buckets: Option<usize>,
    pub(super) occupancy_resource_limit: Option<usize>,
    pub(super) critical_path: Option<bool>,
    pub(super) critical_path_limit: Option<usize>,
}

#[derive(Deserialize)]
pub(super) struct RunSearchBudgetSection {
    #[serde(
        alias = "max_candidates",
        alias = "max_configs",
        alias = "max_rank_configs"
    )]
    pub(super) max_parallelism_candidates: Option<usize>,
    #[serde(alias = "max_prefill_configs")]
    pub(super) max_prefill_candidates: Option<usize>,
    #[serde(alias = "max_decode_configs")]
    pub(super) max_decode_candidates: Option<usize>,
    #[serde(alias = "max_pairs", alias = "max_pool_pairs")]
    pub(super) max_serving_pairs: Option<usize>,
    #[serde(
        alias = "max_runtime_milliseconds",
        alias = "max_search_runtime_ms",
        alias = "max_search_runtime_milliseconds"
    )]
    pub(super) max_runtime_ms: Option<u64>,
    #[serde(
        alias = "retain_rejected",
        alias = "include_rejected_candidates",
        alias = "include_rejections",
        alias = "keep_rejected_candidates"
    )]
    pub(super) retain_rejected_candidates: Option<bool>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioSection {
    pub(super) name: Option<String>,
    #[serde(alias = "requests")]
    pub(super) request_count: Option<u32>,
    #[serde(alias = "interarrival_scale")]
    pub(super) arrival_gap_scale: Option<f64>,
    #[serde(alias = "traffic_rate_scale", alias = "qps_scale")]
    pub(super) arrival_rate_scale: Option<f64>,
    pub(super) batch_size_scale: Option<f64>,
    pub(super) prompt_tokens_scale: Option<f64>,
    pub(super) decode_tokens_scale: Option<f64>,
    #[serde(alias = "calibration_profile_path")]
    pub(super) calibration_profile: Option<String>,
    pub(super) calibration: Option<CalibrationSection>,
    pub(super) topology: Option<RunScenarioTopologySection>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioTopologySection {
    #[serde(alias = "fabric_bandwidth_scale")]
    pub(super) interconnect_bandwidth_scale: Option<f64>,
    #[serde(alias = "fabric_latency_scale")]
    pub(super) interconnect_latency_scale: Option<f64>,
    #[serde(alias = "node_nic_bandwidth_scale")]
    pub(super) nic_bandwidth_scale: Option<f64>,
    #[serde(
        alias = "node_state",
        alias = "nodes",
        alias = "node_overlays",
        alias = "unavailable_nodes",
        alias = "offline_nodes"
    )]
    pub(super) node_states: Option<Vec<RunScenarioNodeStateOverlaySection>>,
    #[serde(alias = "offline_gpus", alias = "unavailable_gpus")]
    pub(super) disabled_gpus: Option<Vec<RunScenarioGpuResourceOverlaySection>>,
    #[serde(alias = "offline_nics", alias = "unavailable_nics")]
    pub(super) disabled_nics: Option<Vec<RunScenarioNicResourceOverlaySection>>,
    pub(super) degraded_gpus: Option<Vec<RunScenarioGpuDegradationOverlaySection>>,
    pub(super) degraded_nics: Option<Vec<RunScenarioNicDegradationOverlaySection>>,
    #[serde(alias = "rail_degradations", alias = "degraded_fabric_rails")]
    pub(super) degraded_rails: Option<Vec<RunScenarioRailDegradationOverlaySection>>,
    pub(super) degraded_links: Option<Vec<RunScenarioLinkDegradationOverlaySection>>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioNodeStateOverlaySection {
    #[serde(alias = "node_id")]
    pub(super) node: Option<u32>,
    #[serde(alias = "node_ids")]
    pub(super) nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    pub(super) group: Option<String>,
    #[serde(alias = "node_groups")]
    pub(super) groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "rack_ids")]
    pub(super) racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    pub(super) islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    pub(super) failure_domains: Option<Vec<String>>,
    #[serde(alias = "status", alias = "health")]
    pub(super) state: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioGpuResourceOverlaySection {
    #[serde(alias = "node_id")]
    pub(super) node: Option<u32>,
    #[serde(alias = "node_ids")]
    pub(super) nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    pub(super) group: Option<String>,
    #[serde(alias = "node_groups")]
    pub(super) groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "rack_ids")]
    pub(super) racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    pub(super) islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    pub(super) failure_domains: Option<Vec<String>>,
    #[serde(alias = "gpu_id", alias = "local_gpu_id")]
    pub(super) gpu: Option<u32>,
    #[serde(alias = "gpu_ids", alias = "local_gpu_ids", alias = "local_gpus")]
    pub(super) gpus: Option<Vec<u32>>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioNicResourceOverlaySection {
    #[serde(alias = "node_id")]
    pub(super) node: Option<u32>,
    #[serde(alias = "node_ids")]
    pub(super) nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    pub(super) group: Option<String>,
    #[serde(alias = "node_groups")]
    pub(super) groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "rack_ids")]
    pub(super) racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    pub(super) islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    pub(super) failure_domains: Option<Vec<String>>,
    #[serde(alias = "nic_id")]
    pub(super) nic: Option<u32>,
    #[serde(alias = "nic_ids")]
    pub(super) nics: Option<Vec<u32>>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioGpuDegradationOverlaySection {
    #[serde(alias = "node_id")]
    pub(super) node: Option<u32>,
    #[serde(alias = "node_ids")]
    pub(super) nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    pub(super) group: Option<String>,
    #[serde(alias = "node_groups")]
    pub(super) groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "rack_ids")]
    pub(super) racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    pub(super) islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    pub(super) failure_domains: Option<Vec<String>>,
    #[serde(alias = "gpu_id", alias = "local_gpu_id")]
    pub(super) gpu: Option<u32>,
    #[serde(alias = "gpu_ids", alias = "local_gpu_ids", alias = "local_gpus")]
    pub(super) gpus: Option<Vec<u32>>,
    #[serde(alias = "flops_scale", alias = "peak_flops_scale")]
    pub(super) compute_scale: Option<f64>,
    #[serde(alias = "memory_bandwidth_scale")]
    pub(super) hbm_bandwidth_scale: Option<f64>,
    #[serde(alias = "hbm_size_scale", alias = "memory_capacity_scale")]
    pub(super) hbm_capacity_scale: Option<f64>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioNicDegradationOverlaySection {
    #[serde(alias = "node_id")]
    pub(super) node: Option<u32>,
    #[serde(alias = "node_ids")]
    pub(super) nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    pub(super) group: Option<String>,
    #[serde(alias = "node_groups")]
    pub(super) groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "rack_ids")]
    pub(super) racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    pub(super) islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    pub(super) failure_domains: Option<Vec<String>>,
    #[serde(alias = "nic_id")]
    pub(super) nic: Option<u32>,
    #[serde(alias = "nic_ids")]
    pub(super) nics: Option<Vec<u32>>,
    #[serde(alias = "scale", alias = "nic_bandwidth_scale")]
    pub(super) bandwidth_scale: Option<f64>,
    #[serde(alias = "nic_latency_scale")]
    pub(super) latency_scale: Option<f64>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioRailDegradationOverlaySection {
    #[serde(alias = "node_id")]
    pub(super) node: Option<u32>,
    #[serde(alias = "node_ids")]
    pub(super) nodes: Option<Vec<u32>>,
    #[serde(alias = "node_group")]
    pub(super) group: Option<String>,
    #[serde(alias = "node_groups")]
    pub(super) groups: Option<Vec<String>>,
    #[serde(alias = "node_label", alias = "node_tag")]
    pub(super) node_tag: Option<String>,
    #[serde(alias = "node_labels", alias = "node_label_ids", alias = "node_tags")]
    pub(super) node_tags: Option<Vec<String>>,
    #[serde(alias = "rack_id")]
    pub(super) rack: Option<String>,
    #[serde(alias = "rack_ids")]
    pub(super) racks: Option<Vec<String>>,
    #[serde(alias = "island_id", alias = "topology_domain")]
    pub(super) island: Option<String>,
    #[serde(alias = "island_ids", alias = "topology_domains")]
    pub(super) islands: Option<Vec<String>>,
    #[serde(alias = "failure_domain_id")]
    pub(super) failure_domain: Option<String>,
    #[serde(alias = "failure_domain_ids")]
    pub(super) failure_domains: Option<Vec<String>>,
    pub(super) rail: Option<u32>,
    pub(super) rails: Option<Vec<u32>>,
    #[serde(
        alias = "scale",
        alias = "rail_bandwidth_scale",
        alias = "fabric_bandwidth_scale",
        alias = "nic_bandwidth_scale"
    )]
    pub(super) bandwidth_scale: Option<f64>,
    #[serde(
        alias = "rail_latency_scale",
        alias = "fabric_latency_scale",
        alias = "nic_latency_scale"
    )]
    pub(super) latency_scale: Option<f64>,
}

#[derive(Deserialize)]
pub(super) struct RunScenarioLinkDegradationOverlaySection {
    #[serde(alias = "from_node", alias = "from_node_id")]
    pub(super) from: Option<u32>,
    #[serde(alias = "from_node_ids")]
    pub(super) from_nodes: Option<Vec<u32>>,
    #[serde(alias = "from_node_group")]
    pub(super) from_group: Option<String>,
    #[serde(alias = "from_node_groups")]
    pub(super) from_groups: Option<Vec<String>>,
    #[serde(alias = "from_node_label", alias = "from_node_tag")]
    pub(super) from_node_tag: Option<String>,
    #[serde(
        alias = "from_node_labels",
        alias = "from_node_label_ids",
        alias = "from_node_tags"
    )]
    pub(super) from_node_tags: Option<Vec<String>>,
    #[serde(alias = "from_rack_id")]
    pub(super) from_rack: Option<String>,
    #[serde(alias = "from_rack_ids")]
    pub(super) from_racks: Option<Vec<String>>,
    #[serde(alias = "from_island_id", alias = "from_topology_domain")]
    pub(super) from_island: Option<String>,
    #[serde(alias = "from_island_ids", alias = "from_topology_domains")]
    pub(super) from_islands: Option<Vec<String>>,
    #[serde(alias = "from_failure_domain_id")]
    pub(super) from_failure_domain: Option<String>,
    #[serde(alias = "from_failure_domain_ids")]
    pub(super) from_failure_domains: Option<Vec<String>>,
    #[serde(alias = "from_gpu", alias = "from_gpu_id", alias = "from_local_gpu")]
    pub(super) from_gpu: Option<u32>,
    #[serde(
        alias = "from_gpu_ids",
        alias = "from_local_gpus",
        alias = "from_local_gpu_ids"
    )]
    pub(super) from_gpus: Option<Vec<u32>>,
    #[serde(alias = "to_node", alias = "to_node_id")]
    pub(super) to: Option<u32>,
    #[serde(alias = "to_node_ids")]
    pub(super) to_nodes: Option<Vec<u32>>,
    #[serde(alias = "to_node_group")]
    pub(super) to_group: Option<String>,
    #[serde(alias = "to_node_groups")]
    pub(super) to_groups: Option<Vec<String>>,
    #[serde(alias = "to_node_label", alias = "to_node_tag")]
    pub(super) to_node_tag: Option<String>,
    #[serde(
        alias = "to_node_labels",
        alias = "to_node_label_ids",
        alias = "to_node_tags"
    )]
    pub(super) to_node_tags: Option<Vec<String>>,
    #[serde(alias = "to_rack_id")]
    pub(super) to_rack: Option<String>,
    #[serde(alias = "to_rack_ids")]
    pub(super) to_racks: Option<Vec<String>>,
    #[serde(alias = "to_island_id", alias = "to_topology_domain")]
    pub(super) to_island: Option<String>,
    #[serde(alias = "to_island_ids", alias = "to_topology_domains")]
    pub(super) to_islands: Option<Vec<String>>,
    #[serde(alias = "to_failure_domain_id")]
    pub(super) to_failure_domain: Option<String>,
    #[serde(alias = "to_failure_domain_ids")]
    pub(super) to_failure_domains: Option<Vec<String>>,
    #[serde(alias = "to_gpu", alias = "to_gpu_id", alias = "to_local_gpu")]
    pub(super) to_gpu: Option<u32>,
    #[serde(
        alias = "to_gpu_ids",
        alias = "to_local_gpus",
        alias = "to_local_gpu_ids"
    )]
    pub(super) to_gpus: Option<Vec<u32>>,
    pub(super) rail: Option<u32>,
    pub(super) rails: Option<Vec<u32>>,
    #[serde(
        alias = "scale",
        alias = "link_bandwidth_scale",
        alias = "fabric_bandwidth_scale"
    )]
    pub(super) bandwidth_scale: Option<f64>,
    #[serde(alias = "link_latency_scale", alias = "fabric_latency_scale")]
    pub(super) latency_scale: Option<f64>,
}

#[derive(Deserialize)]
pub(super) struct ModelSection {
    #[serde(alias = "name", alias = "model_id")]
    pub(super) id: Option<String>,
    pub(super) layers: u32,
    pub(super) hidden_size: u32,
    pub(super) attention_heads: u32,
    pub(super) kv_heads: u32,
    pub(super) vocab_size: u32,
    pub(super) parameters_gb: f64,
    #[serde(
        alias = "parameter_count_b",
        alias = "params_billion",
        alias = "params_b"
    )]
    pub(super) parameter_count_billion: Option<f64>,
    pub(super) dtype: String,
    #[serde(alias = "cache_dtype", alias = "kv_cache_dtype")]
    pub(super) kv_dtype: Option<String>,
    pub(super) experts: Option<ExpertSection>,
}

#[derive(Deserialize)]
pub(super) struct ExpertSection {
    pub(super) expert_count: u32,
    pub(super) top_k: u32,
}

#[derive(Deserialize)]
pub(super) struct RequestSection {
    pub(super) batch_size: u32,
    pub(super) prompt_tokens: u32,
    pub(super) decode_tokens: u32,
    pub(super) max_sequence_tokens: u32,
    pub(super) phase: String,
}

#[derive(Clone, Deserialize)]
pub(super) struct SearchSection {
    pub(super) tensor_ranks: Vec<u32>,
    pub(super) pipeline_ranks: Vec<u32>,
    pub(super) expert_ranks: Vec<u32>,
    pub(super) data_ranks: Vec<u32>,
}

#[derive(Clone, Deserialize)]
pub(super) struct PlacementSection {
    pub(super) ranks: Vec<PlacementRankSection>,
}

#[derive(Clone, Deserialize)]
pub(super) struct PlacementRankSection {
    #[serde(alias = "rank_id")]
    pub(super) rank: Option<u32>,
    #[serde(alias = "node_id")]
    pub(super) node: Option<u32>,
    #[serde(alias = "gpu", alias = "gpu_id", alias = "local_gpu_id")]
    pub(super) local_gpu_id: Option<u32>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationSection {
    pub(super) compute_efficiency: Option<f64>,
    pub(super) prefill_compute_scale: Option<f64>,
    pub(super) decode_compute_scale: Option<f64>,
    pub(super) decode_memory_bandwidth_scale: Option<f64>,
    pub(super) collective_latency_scale: Option<f64>,
    pub(super) collective_bandwidth_scale: Option<f64>,
    pub(super) kv_transfer_scale: Option<f64>,
    pub(super) scheduler_overhead_us: Option<f64>,
    pub(super) serving_memory_temporary_fraction: Option<f64>,
    pub(super) serving_memory_activation_communication_fraction: Option<f64>,
    pub(super) serving_memory_weight_communication_fraction: Option<f64>,
    pub(super) serving_memory_runtime_reserve_fraction: Option<f64>,
    pub(super) serving_memory_fragmentation_fraction: Option<f64>,
    pub(super) serving_pipeline_depth: Option<u32>,
    pub(super) request_arrival_gap_s: Option<f64>,
    pub(super) allow_compute_comm_overlap: Option<bool>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationPolicySection {
    pub(super) valid_shape: Option<String>,
    pub(super) invalid_shape: Option<String>,
    pub(super) coverage: Option<String>,
    pub(super) fit_confidence: Option<String>,
    pub(super) fit_extrapolation: Option<String>,
    pub(super) fit_partially_bounded: Option<String>,
    pub(super) fit_unbounded: Option<String>,
    pub(super) fit_sample_count: Option<String>,
    pub(super) fit_validation_sample_count: Option<String>,
    pub(super) fit_source: Option<String>,
    pub(super) fit_uncertainty: Option<String>,
    pub(super) profile_source: Option<String>,
    pub(super) profile_date: Option<String>,
    #[serde(alias = "profile_runtime_provenance", alias = "profile_stack")]
    pub(super) profile_runtime: Option<String>,
    pub(super) min_coverage_score: Option<f64>,
    pub(super) min_fit_confidence_score: Option<f64>,
    pub(super) min_fit_confidence_level: Option<f64>,
    pub(super) min_fit_sample_count: Option<u32>,
    pub(super) min_fit_validation_sample_count: Option<u32>,
    pub(super) max_fit_relative_uncertainty_pct: Option<f64>,
    pub(super) max_fit_absolute_uncertainty_ms: Option<f64>,
    pub(super) max_fit_absolute_uncertainty_s: Option<f64>,
    pub(super) min_serving_phase_coverage_fraction: Option<f64>,
    pub(super) uncertainty_ranking_weight: Option<f64>,
    pub(super) require_phase_coverage: Option<bool>,
}

#[derive(Deserialize)]
pub(super) struct ApproximationPolicySection {
    #[serde(alias = "profile")]
    pub(super) preset: Option<String>,
    #[serde(alias = "default", alias = "mode")]
    pub(super) default_action: Option<String>,
    pub(super) reject_categories: Option<Vec<String>>,
    pub(super) reject_codes: Option<Vec<String>>,
    pub(super) warn_categories: Option<Vec<String>>,
    pub(super) warn_codes: Option<Vec<String>>,
    #[serde(
        default,
        alias = "metric_gate",
        alias = "precision_gates",
        alias = "precision_gate",
        alias = "metric_precision_gates",
        alias = "metric_precision_gate"
    )]
    pub(super) metric_gates: Vec<ApproximationMetricGateSection>,
}

#[derive(Deserialize)]
pub(super) struct ApproximationMetricGateSection {
    pub(super) metric: Option<String>,
    pub(super) metrics: Option<Vec<String>>,
    #[serde(alias = "objective")]
    pub(super) objective: Option<String>,
    #[serde(alias = "objectives")]
    pub(super) objectives: Option<Vec<String>>,
    pub(super) reject_categories: Option<Vec<String>>,
    pub(super) reject_codes: Option<Vec<String>>,
    pub(super) warn_categories: Option<Vec<String>>,
    pub(super) warn_codes: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationProfileFile {
    pub(super) schema_version: Option<u32>,
    pub(super) profile: Option<CalibrationProfileMetadataSection>,
    pub(super) calibration: Option<CalibrationSection>,
    pub(super) valid_shape: Option<CalibrationShapeRangeSection>,
    pub(super) invalid_shapes: Option<Vec<CalibrationInvalidShapeRangeSection>>,
    pub(super) fits: Option<Vec<CalibrationFittedModelSection>>,
    pub(super) benchmarks: Option<Vec<CalibrationBenchmarkPointSection>>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationProfileMetadataSection {
    pub(super) name: Option<String>,
    pub(super) hardware: Option<String>,
    pub(super) fabric: Option<String>,
    pub(super) model: Option<String>,
    pub(super) dtype: Option<String>,
    pub(super) serving_stack: Option<String>,
    #[serde(
        alias = "runtime_features",
        alias = "backend_features",
        alias = "stack_features"
    )]
    pub(super) serving_runtime_features: Option<Vec<String>>,
    #[serde(alias = "serving_stack_version", alias = "runtime_version")]
    pub(super) backend_version: Option<String>,
    #[serde(alias = "gpu_driver_version")]
    pub(super) driver_version: Option<String>,
    #[serde(alias = "cuda")]
    pub(super) cuda_version: Option<String>,
    #[serde(alias = "rocm")]
    pub(super) rocm_version: Option<String>,
    #[serde(alias = "nccl")]
    pub(super) nccl_version: Option<String>,
    #[serde(alias = "rccl")]
    pub(super) rccl_version: Option<String>,
    #[serde(alias = "ucx")]
    pub(super) ucx_version: Option<String>,
    #[serde(default)]
    pub(super) kernel_settings: Vec<String>,
    pub(super) environment_hash: Option<String>,
    pub(super) source: Option<String>,
    pub(super) date: Option<String>,
    pub(super) notes: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationShapeRangeSection {
    pub(super) min_batch_size: Option<u32>,
    pub(super) max_batch_size: Option<u32>,
    pub(super) min_prompt_tokens: Option<u32>,
    pub(super) max_prompt_tokens: Option<u32>,
    pub(super) min_decode_tokens: Option<u32>,
    pub(super) max_decode_tokens: Option<u32>,
    pub(super) min_sequence_tokens: Option<u32>,
    pub(super) max_sequence_tokens: Option<u32>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationInvalidShapeRangeSection {
    pub(super) name: Option<String>,
    pub(super) reason: Option<String>,
    pub(super) min_batch_size: Option<u32>,
    pub(super) max_batch_size: Option<u32>,
    pub(super) min_prompt_tokens: Option<u32>,
    pub(super) max_prompt_tokens: Option<u32>,
    pub(super) min_decode_tokens: Option<u32>,
    pub(super) max_decode_tokens: Option<u32>,
    pub(super) min_sequence_tokens: Option<u32>,
    pub(super) max_sequence_tokens: Option<u32>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationFittedModelSection {
    pub(super) name: Option<String>,
    pub(super) target: Option<String>,
    pub(super) phase: Option<String>,
    pub(super) kind: Option<String>,
    pub(super) model: Option<String>,
    pub(super) unit: Option<String>,
    pub(super) intercept: Option<f64>,
    #[serde(default, alias = "feature_names")]
    pub(super) features: Vec<String>,
    #[serde(default)]
    pub(super) coefficients: Vec<f64>,
    #[serde(default)]
    pub(super) feature_ranges: Vec<CalibrationFitFeatureRangeSection>,
    pub(super) r_squared: Option<f64>,
    pub(super) adjusted_r_squared: Option<f64>,
    pub(super) rmse: Option<f64>,
    pub(super) rmse_pct: Option<f64>,
    pub(super) mean_abs_pct_error: Option<f64>,
    pub(super) max_abs_pct_error: Option<f64>,
    #[serde(alias = "holdout_rmse")]
    pub(super) validation_rmse: Option<f64>,
    #[serde(alias = "holdout_rmse_pct")]
    pub(super) validation_rmse_pct: Option<f64>,
    #[serde(alias = "holdout_mean_abs_pct_error")]
    pub(super) validation_mean_abs_pct_error: Option<f64>,
    #[serde(alias = "holdout_max_abs_pct_error")]
    pub(super) validation_max_abs_pct_error: Option<f64>,
    #[serde(alias = "ci", alias = "ci95", alias = "confidence_interval_value")]
    pub(super) confidence_interval: Option<f64>,
    #[serde(
        alias = "ci_pct",
        alias = "ci95_pct",
        alias = "confidence_interval_percent"
    )]
    pub(super) confidence_interval_pct: Option<f64>,
    #[serde(
        alias = "ci_level",
        alias = "confidence",
        alias = "coverage_probability"
    )]
    pub(super) confidence_level: Option<f64>,
    pub(super) sample_count: Option<u32>,
    pub(super) validation_sample_count: Option<u32>,
    pub(super) source: Option<String>,
    pub(super) notes: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationFitFeatureRangeSection {
    pub(super) feature: Option<String>,
    pub(super) min: Option<f64>,
    pub(super) max: Option<f64>,
}

#[derive(Deserialize)]
pub(super) struct CalibrationBenchmarkPointSection {
    pub(super) name: Option<String>,
    pub(super) kind: Option<String>,
    pub(super) phase: Option<String>,
    pub(super) hardware: Option<String>,
    pub(super) fabric: Option<String>,
    pub(super) model: Option<String>,
    pub(super) dtype: Option<String>,
    pub(super) batch_size: Option<u32>,
    pub(super) prompt_tokens: Option<u32>,
    pub(super) decode_tokens: Option<u32>,
    pub(super) sequence_tokens: Option<u32>,
    pub(super) tensor_ranks: Option<u32>,
    pub(super) pipeline_ranks: Option<u32>,
    pub(super) expert_ranks: Option<u32>,
    pub(super) data_ranks: Option<u32>,
    pub(super) measured_ms: Option<f64>,
    pub(super) predicted_ms: Option<f64>,
    pub(super) throughput_tokens_per_s: Option<f64>,
    pub(super) command: Option<String>,
    pub(super) source: Option<String>,
    pub(super) notes: Option<String>,
}
