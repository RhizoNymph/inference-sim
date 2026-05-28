use super::*;

#[derive(Deserialize)]
pub(in crate::config) struct ServingSection {
    pub(in crate::config) mode: Option<String>,
    #[serde(alias = "runtime", alias = "backend", alias = "stack")]
    pub(in crate::config) serving_stack: Option<String>,
    #[serde(
        alias = "runtime_features",
        alias = "backend_features",
        alias = "stack_features"
    )]
    pub(in crate::config) serving_runtime_features: Option<Vec<String>>,
    pub(in crate::config) objective: Option<String>,
    #[serde(alias = "slo_penalty_weight", alias = "slo_objective_penalty_weight")]
    pub(in crate::config) slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "ttft_slo_penalty_weight", alias = "ttft_penalty_weight")]
    pub(in crate::config) ttft_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "tpot_slo_penalty_weight", alias = "tpot_penalty_weight")]
    pub(in crate::config) tpot_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "itl_slo_penalty_weight", alias = "itl_penalty_weight")]
    pub(in crate::config) itl_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "e2el_slo_penalty_weight", alias = "e2el_penalty_weight")]
    pub(in crate::config) e2el_slo_miss_penalty_weight: Option<f64>,
    #[serde(
        alias = "deadline_slo_miss_penalty_weight",
        alias = "deadline_penalty_weight"
    )]
    pub(in crate::config) deadline_miss_penalty_weight: Option<f64>,
    #[serde(alias = "route_risk_penalty_weight", alias = "topology_penalty_weight")]
    pub(in crate::config) topology_risk_penalty_weight: Option<f64>,
    #[serde(alias = "max_hbm_pressure", alias = "max_hbm_utilization")]
    pub(in crate::config) max_memory_pressure_fraction: Option<f64>,
    #[serde(
        alias = "max_gpus",
        alias = "max_gpu_footprint",
        alias = "max_serving_gpus"
    )]
    pub(in crate::config) max_unique_gpus: Option<u32>,
    #[serde(
        alias = "min_throughput",
        alias = "min_tokens_per_s",
        alias = "min_output_tokens_per_s"
    )]
    pub(in crate::config) min_throughput_tokens_per_s: Option<f64>,
    #[serde(
        alias = "ttft_ceiling_s",
        alias = "max_ttft_latency_s",
        alias = "max_time_to_first_token_s"
    )]
    pub(in crate::config) max_ttft_s: Option<f64>,
    #[serde(
        alias = "ttft_ceiling_ms",
        alias = "max_ttft_latency_ms",
        alias = "max_time_to_first_token_ms"
    )]
    pub(in crate::config) max_ttft_ms: Option<f64>,
    #[serde(
        alias = "tpot_ceiling_s",
        alias = "max_tpot_latency_s",
        alias = "max_time_per_output_token_s"
    )]
    pub(in crate::config) max_tpot_s: Option<f64>,
    #[serde(
        alias = "tpot_ceiling_ms",
        alias = "max_tpot_latency_ms",
        alias = "max_time_per_output_token_ms"
    )]
    pub(in crate::config) max_tpot_ms: Option<f64>,
    #[serde(alias = "itl_ceiling_s", alias = "max_itl_latency_s")]
    pub(in crate::config) max_itl_s: Option<f64>,
    #[serde(alias = "itl_ceiling_ms", alias = "max_itl_latency_ms")]
    pub(in crate::config) max_itl_ms: Option<f64>,
    #[serde(
        alias = "e2el_ceiling_s",
        alias = "max_e2el_latency_s",
        alias = "max_end_to_end_latency_s"
    )]
    pub(in crate::config) max_e2el_s: Option<f64>,
    #[serde(
        alias = "e2el_ceiling_ms",
        alias = "max_e2el_latency_ms",
        alias = "max_end_to_end_latency_ms"
    )]
    pub(in crate::config) max_e2el_ms: Option<f64>,
    #[serde(
        alias = "min_kv_route_rails",
        alias = "min_kv_transfer_rails",
        alias = "min_kv_transfer_rail_count"
    )]
    pub(in crate::config) min_kv_route_rail_count: Option<u32>,
    #[serde(
        alias = "require_rail_metadata",
        alias = "require_kv_transfer_rail_metadata"
    )]
    pub(in crate::config) require_kv_route_rail_metadata: Option<bool>,
    #[serde(
        alias = "require_kv_gpudirect",
        alias = "require_kv_transfer_gpudirect"
    )]
    pub(in crate::config) require_gpudirect_kv_paths: Option<bool>,
    #[serde(alias = "economics", alias = "cost_model")]
    pub(in crate::config) cost: Option<ServingCostSection>,
    #[serde(
        alias = "require_routable_pool",
        alias = "validate_routable_pool",
        alias = "validate_routable_pools"
    )]
    pub(in crate::config) require_routable_pools: Option<bool>,
    pub(in crate::config) prefill_nodes: Option<Vec<u32>>,
    pub(in crate::config) decode_nodes: Option<Vec<u32>>,
    pub(in crate::config) prefill_groups: Option<Vec<String>>,
    pub(in crate::config) decode_groups: Option<Vec<String>>,
    #[serde(alias = "prefill_gpu_label", alias = "prefill_gpu_tag")]
    pub(in crate::config) prefill_gpu_tag: Option<String>,
    #[serde(alias = "prefill_gpu_labels", alias = "prefill_gpu_tags")]
    pub(in crate::config) prefill_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "decode_gpu_label", alias = "decode_gpu_tag")]
    pub(in crate::config) decode_gpu_tag: Option<String>,
    #[serde(alias = "decode_gpu_labels", alias = "decode_gpu_tags")]
    pub(in crate::config) decode_gpu_tags: Option<Vec<String>>,
    pub(in crate::config) pool_candidates: Option<Vec<ServingPoolCandidateSection>>,
    pub(in crate::config) pool_search: Option<ServingPoolSearchSection>,
    pub(in crate::config) slo_policies: Option<Vec<ServingSloPolicySection>>,
    #[serde(alias = "classes")]
    pub(in crate::config) traffic_classes: Option<Vec<ServingTrafficClassSection>>,
    pub(in crate::config) prefill_search: Option<SearchSection>,
    pub(in crate::config) decode_search: Option<SearchSection>,
    pub(in crate::config) prefill_placement: Option<PlacementSection>,
    pub(in crate::config) decode_placement: Option<PlacementSection>,
    pub(in crate::config) services: Option<ServingServicesSection>,
    pub(in crate::config) prefill_service: Option<ServingServicePhaseSection>,
    pub(in crate::config) decode_service: Option<ServingServicePhaseSection>,
    pub(in crate::config) kv_transfer_service: Option<ServingServicePhaseSection>,
    pub(in crate::config) traffic: Option<ServingTrafficSection>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingCostSection {
    #[serde(alias = "gpu_hour_usd", alias = "usd_per_gpu_hour")]
    pub(in crate::config) default_gpu_hour_usd: Option<f64>,
    #[serde(alias = "usd_per_node_hour")]
    pub(in crate::config) node_hour_usd: Option<f64>,
    #[serde(alias = "usd_per_kwh", alias = "energy_usd_per_kwh")]
    pub(in crate::config) kwh_usd: Option<f64>,
    #[serde(alias = "gpu_watts", alias = "watts_per_gpu")]
    pub(in crate::config) default_gpu_watts: Option<f64>,
    #[serde(alias = "watts_per_node")]
    pub(in crate::config) node_watts: Option<f64>,
    pub(in crate::config) gpu_rates: Option<Vec<ServingGpuCostRateSection>>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingGpuCostRateSection {
    #[serde(alias = "gpu", alias = "gpu_type", alias = "label")]
    pub(in crate::config) gpu_label: Option<String>,
    #[serde(
        alias = "gpu_hour_usd",
        alias = "usd_per_hour",
        alias = "usd_per_gpu_hour"
    )]
    pub(in crate::config) gpu_hour_usd: Option<f64>,
    #[serde(alias = "gpu_watts")]
    pub(in crate::config) watts: Option<f64>,
}

#[derive(Clone, Deserialize)]
pub(in crate::config) struct ServingServicesSection {
    pub(in crate::config) prefill: Option<ServingServicePhaseSection>,
    pub(in crate::config) decode: Option<ServingServicePhaseSection>,
    pub(in crate::config) kv_transfer: Option<ServingServicePhaseSection>,
}

pub(in crate::config) struct ServingServiceSections {
    pub(in crate::config) services: Option<ServingServicesSection>,
    pub(in crate::config) prefill: Option<ServingServicePhaseSection>,
    pub(in crate::config) decode: Option<ServingServicePhaseSection>,
    pub(in crate::config) kv_transfer: Option<ServingServicePhaseSection>,
}

#[derive(Clone, Deserialize)]
pub(in crate::config) struct ServingServicePhaseSection {
    pub(in crate::config) health: Option<String>,
    pub(in crate::config) enabled: Option<bool>,
    #[serde(alias = "scale", alias = "worker_slots_scale")]
    pub(in crate::config) worker_scale: Option<f64>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingSloPolicySection {
    pub(in crate::config) group: String,
    pub(in crate::config) key: String,
    pub(in crate::config) max_ttft_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_tpot_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_itl_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_e2el_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_deadline_miss_rate: Option<f64>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingTrafficClassSection {
    pub(in crate::config) name: String,
    pub(in crate::config) group: String,
    pub(in crate::config) key: Option<String>,
    pub(in crate::config) priority: Option<i32>,
    #[serde(
        alias = "priority_override",
        alias = "queue_priority",
        alias = "scheduling_priority"
    )]
    pub(in crate::config) admission_priority: Option<i32>,
    #[serde(alias = "max_active_prefill_tokens")]
    pub(in crate::config) max_prefill_tokens: Option<u64>,
    pub(in crate::config) max_decode_sequences: Option<u32>,
    pub(in crate::config) max_resident_tokens: Option<u64>,
    pub(in crate::config) max_kv_blocks: Option<u64>,
    pub(in crate::config) ttft_slo_ms: Option<f64>,
    pub(in crate::config) tpot_slo_ms: Option<f64>,
    pub(in crate::config) itl_slo_ms: Option<f64>,
    pub(in crate::config) e2el_slo_ms: Option<f64>,
    pub(in crate::config) max_queue_delay_s: Option<f64>,
    pub(in crate::config) max_queue_delay_ms: Option<f64>,
    pub(in crate::config) max_kv_queue_delay_s: Option<f64>,
    pub(in crate::config) max_kv_queue_delay_ms: Option<f64>,
    pub(in crate::config) max_decode_queue_delay_s: Option<f64>,
    pub(in crate::config) max_decode_queue_delay_ms: Option<f64>,
    pub(in crate::config) max_decode_iteration_queue_delay_s: Option<f64>,
    pub(in crate::config) max_decode_iteration_queue_delay_ms: Option<f64>,
    pub(in crate::config) request_timeout_s: Option<f64>,
    pub(in crate::config) request_timeout_ms: Option<f64>,
    #[serde(alias = "slo_penalty_weight", alias = "slo_objective_penalty_weight")]
    pub(in crate::config) slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "ttft_slo_penalty_weight", alias = "ttft_penalty_weight")]
    pub(in crate::config) ttft_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "tpot_slo_penalty_weight", alias = "tpot_penalty_weight")]
    pub(in crate::config) tpot_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "itl_slo_penalty_weight", alias = "itl_penalty_weight")]
    pub(in crate::config) itl_slo_miss_penalty_weight: Option<f64>,
    #[serde(alias = "e2el_slo_penalty_weight", alias = "e2el_penalty_weight")]
    pub(in crate::config) e2el_slo_miss_penalty_weight: Option<f64>,
    #[serde(
        alias = "deadline_slo_miss_penalty_weight",
        alias = "deadline_penalty_weight"
    )]
    pub(in crate::config) deadline_miss_penalty_weight: Option<f64>,
    pub(in crate::config) max_ttft_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_tpot_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_itl_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_e2el_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_deadline_miss_rate: Option<f64>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingPoolCandidateSection {
    pub(in crate::config) label: Option<String>,
    pub(in crate::config) prefill_nodes: Option<Vec<u32>>,
    pub(in crate::config) decode_nodes: Option<Vec<u32>>,
    pub(in crate::config) prefill_groups: Option<Vec<String>>,
    pub(in crate::config) decode_groups: Option<Vec<String>>,
    #[serde(alias = "prefill_gpu_label", alias = "prefill_gpu_tag")]
    pub(in crate::config) prefill_gpu_tag: Option<String>,
    #[serde(alias = "prefill_gpu_labels", alias = "prefill_gpu_tags")]
    pub(in crate::config) prefill_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "decode_gpu_label", alias = "decode_gpu_tag")]
    pub(in crate::config) decode_gpu_tag: Option<String>,
    #[serde(alias = "decode_gpu_labels", alias = "decode_gpu_tags")]
    pub(in crate::config) decode_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_node_label", alias = "prefill_node_tag")]
    pub(in crate::config) prefill_node_tag: Option<String>,
    #[serde(alias = "prefill_node_labels", alias = "prefill_node_tags")]
    pub(in crate::config) prefill_node_tags: Option<Vec<String>>,
    #[serde(alias = "decode_node_label", alias = "decode_node_tag")]
    pub(in crate::config) decode_node_tag: Option<String>,
    #[serde(alias = "decode_node_labels", alias = "decode_node_tags")]
    pub(in crate::config) decode_node_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_rack_id")]
    pub(in crate::config) prefill_rack: Option<String>,
    #[serde(alias = "prefill_rack_ids")]
    pub(in crate::config) prefill_racks: Option<Vec<String>>,
    #[serde(alias = "decode_rack_id")]
    pub(in crate::config) decode_rack: Option<String>,
    #[serde(alias = "decode_rack_ids")]
    pub(in crate::config) decode_racks: Option<Vec<String>>,
    #[serde(alias = "prefill_island_id")]
    pub(in crate::config) prefill_island: Option<String>,
    #[serde(alias = "prefill_island_ids")]
    pub(in crate::config) prefill_islands: Option<Vec<String>>,
    #[serde(alias = "decode_island_id")]
    pub(in crate::config) decode_island: Option<String>,
    #[serde(alias = "decode_island_ids")]
    pub(in crate::config) decode_islands: Option<Vec<String>>,
    #[serde(alias = "prefill_failure_domain_id")]
    pub(in crate::config) prefill_failure_domain: Option<String>,
    #[serde(alias = "prefill_failure_domain_ids")]
    pub(in crate::config) prefill_failure_domains: Option<Vec<String>>,
    #[serde(alias = "decode_failure_domain_id")]
    pub(in crate::config) decode_failure_domain: Option<String>,
    #[serde(alias = "decode_failure_domain_ids")]
    pub(in crate::config) decode_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_node_label",
        alias = "exclude_prefill_node_label",
        alias = "exclude_prefill_node_tag"
    )]
    pub(in crate::config) prefill_exclude_node_tag: Option<String>,
    #[serde(
        alias = "prefill_exclude_node_labels",
        alias = "exclude_prefill_node_labels",
        alias = "exclude_prefill_node_tags"
    )]
    pub(in crate::config) prefill_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_node_label",
        alias = "exclude_decode_node_label",
        alias = "exclude_decode_node_tag"
    )]
    pub(in crate::config) decode_exclude_node_tag: Option<String>,
    #[serde(
        alias = "decode_exclude_node_labels",
        alias = "exclude_decode_node_labels",
        alias = "exclude_decode_node_tags"
    )]
    pub(in crate::config) decode_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_rack_id",
        alias = "exclude_prefill_rack",
        alias = "exclude_prefill_rack_id"
    )]
    pub(in crate::config) prefill_exclude_rack: Option<String>,
    #[serde(
        alias = "prefill_exclude_rack_ids",
        alias = "exclude_prefill_racks",
        alias = "exclude_prefill_rack_ids"
    )]
    pub(in crate::config) prefill_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_rack_id",
        alias = "exclude_decode_rack",
        alias = "exclude_decode_rack_id"
    )]
    pub(in crate::config) decode_exclude_rack: Option<String>,
    #[serde(
        alias = "decode_exclude_rack_ids",
        alias = "exclude_decode_racks",
        alias = "exclude_decode_rack_ids"
    )]
    pub(in crate::config) decode_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_island_id",
        alias = "exclude_prefill_island",
        alias = "exclude_prefill_island_id"
    )]
    pub(in crate::config) prefill_exclude_island: Option<String>,
    #[serde(
        alias = "prefill_exclude_island_ids",
        alias = "exclude_prefill_islands",
        alias = "exclude_prefill_island_ids"
    )]
    pub(in crate::config) prefill_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_island_id",
        alias = "exclude_decode_island",
        alias = "exclude_decode_island_id"
    )]
    pub(in crate::config) decode_exclude_island: Option<String>,
    #[serde(
        alias = "decode_exclude_island_ids",
        alias = "exclude_decode_islands",
        alias = "exclude_decode_island_ids"
    )]
    pub(in crate::config) decode_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_failure_domain_id",
        alias = "exclude_prefill_failure_domain",
        alias = "exclude_prefill_failure_domain_id"
    )]
    pub(in crate::config) prefill_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "prefill_exclude_failure_domain_ids",
        alias = "exclude_prefill_failure_domains",
        alias = "exclude_prefill_failure_domain_ids"
    )]
    pub(in crate::config) prefill_exclude_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_failure_domain_id",
        alias = "exclude_decode_failure_domain",
        alias = "exclude_decode_failure_domain_id"
    )]
    pub(in crate::config) decode_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "decode_exclude_failure_domain_ids",
        alias = "exclude_decode_failure_domains",
        alias = "exclude_decode_failure_domain_ids"
    )]
    pub(in crate::config) decode_exclude_failure_domains: Option<Vec<String>>,
    #[serde(alias = "min_prefill_rack_count")]
    pub(in crate::config) min_prefill_racks: Option<u32>,
    #[serde(alias = "min_decode_rack_count")]
    pub(in crate::config) min_decode_racks: Option<u32>,
    #[serde(alias = "min_prefill_island_count")]
    pub(in crate::config) min_prefill_islands: Option<u32>,
    #[serde(alias = "min_decode_island_count")]
    pub(in crate::config) min_decode_islands: Option<u32>,
    #[serde(alias = "min_prefill_failure_domain_count")]
    pub(in crate::config) min_prefill_failure_domains: Option<u32>,
    #[serde(alias = "min_decode_failure_domain_count")]
    pub(in crate::config) min_decode_failure_domains: Option<u32>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingPoolSearchSection {
    pub(in crate::config) prefill_groups: Vec<String>,
    pub(in crate::config) decode_groups: Vec<String>,
    pub(in crate::config) prefill_node_counts: Option<Vec<u32>>,
    pub(in crate::config) decode_node_counts: Option<Vec<u32>>,
    #[serde(alias = "prefill_gpu_label", alias = "prefill_gpu_tag")]
    pub(in crate::config) prefill_gpu_tag: Option<String>,
    #[serde(alias = "prefill_gpu_labels", alias = "prefill_gpu_tags")]
    pub(in crate::config) prefill_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "decode_gpu_label", alias = "decode_gpu_tag")]
    pub(in crate::config) decode_gpu_tag: Option<String>,
    #[serde(alias = "decode_gpu_labels", alias = "decode_gpu_tags")]
    pub(in crate::config) decode_gpu_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_node_label", alias = "prefill_node_tag")]
    pub(in crate::config) prefill_node_tag: Option<String>,
    #[serde(alias = "prefill_node_labels", alias = "prefill_node_tags")]
    pub(in crate::config) prefill_node_tags: Option<Vec<String>>,
    #[serde(alias = "decode_node_label", alias = "decode_node_tag")]
    pub(in crate::config) decode_node_tag: Option<String>,
    #[serde(alias = "decode_node_labels", alias = "decode_node_tags")]
    pub(in crate::config) decode_node_tags: Option<Vec<String>>,
    #[serde(alias = "prefill_rack_id")]
    pub(in crate::config) prefill_rack: Option<String>,
    #[serde(alias = "prefill_rack_ids")]
    pub(in crate::config) prefill_racks: Option<Vec<String>>,
    #[serde(alias = "decode_rack_id")]
    pub(in crate::config) decode_rack: Option<String>,
    #[serde(alias = "decode_rack_ids")]
    pub(in crate::config) decode_racks: Option<Vec<String>>,
    #[serde(alias = "prefill_island_id")]
    pub(in crate::config) prefill_island: Option<String>,
    #[serde(alias = "prefill_island_ids")]
    pub(in crate::config) prefill_islands: Option<Vec<String>>,
    #[serde(alias = "decode_island_id")]
    pub(in crate::config) decode_island: Option<String>,
    #[serde(alias = "decode_island_ids")]
    pub(in crate::config) decode_islands: Option<Vec<String>>,
    #[serde(alias = "prefill_failure_domain_id")]
    pub(in crate::config) prefill_failure_domain: Option<String>,
    #[serde(alias = "prefill_failure_domain_ids")]
    pub(in crate::config) prefill_failure_domains: Option<Vec<String>>,
    #[serde(alias = "decode_failure_domain_id")]
    pub(in crate::config) decode_failure_domain: Option<String>,
    #[serde(alias = "decode_failure_domain_ids")]
    pub(in crate::config) decode_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_node_label",
        alias = "exclude_prefill_node_label",
        alias = "exclude_prefill_node_tag"
    )]
    pub(in crate::config) prefill_exclude_node_tag: Option<String>,
    #[serde(
        alias = "prefill_exclude_node_labels",
        alias = "exclude_prefill_node_labels",
        alias = "exclude_prefill_node_tags"
    )]
    pub(in crate::config) prefill_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_node_label",
        alias = "exclude_decode_node_label",
        alias = "exclude_decode_node_tag"
    )]
    pub(in crate::config) decode_exclude_node_tag: Option<String>,
    #[serde(
        alias = "decode_exclude_node_labels",
        alias = "exclude_decode_node_labels",
        alias = "exclude_decode_node_tags"
    )]
    pub(in crate::config) decode_exclude_node_tags: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_rack_id",
        alias = "exclude_prefill_rack",
        alias = "exclude_prefill_rack_id"
    )]
    pub(in crate::config) prefill_exclude_rack: Option<String>,
    #[serde(
        alias = "prefill_exclude_rack_ids",
        alias = "exclude_prefill_racks",
        alias = "exclude_prefill_rack_ids"
    )]
    pub(in crate::config) prefill_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_rack_id",
        alias = "exclude_decode_rack",
        alias = "exclude_decode_rack_id"
    )]
    pub(in crate::config) decode_exclude_rack: Option<String>,
    #[serde(
        alias = "decode_exclude_rack_ids",
        alias = "exclude_decode_racks",
        alias = "exclude_decode_rack_ids"
    )]
    pub(in crate::config) decode_exclude_racks: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_island_id",
        alias = "exclude_prefill_island",
        alias = "exclude_prefill_island_id"
    )]
    pub(in crate::config) prefill_exclude_island: Option<String>,
    #[serde(
        alias = "prefill_exclude_island_ids",
        alias = "exclude_prefill_islands",
        alias = "exclude_prefill_island_ids"
    )]
    pub(in crate::config) prefill_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_island_id",
        alias = "exclude_decode_island",
        alias = "exclude_decode_island_id"
    )]
    pub(in crate::config) decode_exclude_island: Option<String>,
    #[serde(
        alias = "decode_exclude_island_ids",
        alias = "exclude_decode_islands",
        alias = "exclude_decode_island_ids"
    )]
    pub(in crate::config) decode_exclude_islands: Option<Vec<String>>,
    #[serde(
        alias = "prefill_exclude_failure_domain_id",
        alias = "exclude_prefill_failure_domain",
        alias = "exclude_prefill_failure_domain_id"
    )]
    pub(in crate::config) prefill_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "prefill_exclude_failure_domain_ids",
        alias = "exclude_prefill_failure_domains",
        alias = "exclude_prefill_failure_domain_ids"
    )]
    pub(in crate::config) prefill_exclude_failure_domains: Option<Vec<String>>,
    #[serde(
        alias = "decode_exclude_failure_domain_id",
        alias = "exclude_decode_failure_domain",
        alias = "exclude_decode_failure_domain_id"
    )]
    pub(in crate::config) decode_exclude_failure_domain: Option<String>,
    #[serde(
        alias = "decode_exclude_failure_domain_ids",
        alias = "exclude_decode_failure_domains",
        alias = "exclude_decode_failure_domain_ids"
    )]
    pub(in crate::config) decode_exclude_failure_domains: Option<Vec<String>>,
    #[serde(alias = "min_prefill_rack_count")]
    pub(in crate::config) min_prefill_racks: Option<u32>,
    #[serde(alias = "min_decode_rack_count")]
    pub(in crate::config) min_decode_racks: Option<u32>,
    #[serde(alias = "min_prefill_island_count")]
    pub(in crate::config) min_prefill_islands: Option<u32>,
    #[serde(alias = "min_decode_island_count")]
    pub(in crate::config) min_decode_islands: Option<u32>,
    #[serde(alias = "min_prefill_failure_domain_count")]
    pub(in crate::config) min_prefill_failure_domains: Option<u32>,
    #[serde(alias = "min_decode_failure_domain_count")]
    pub(in crate::config) min_decode_failure_domains: Option<u32>,
    pub(in crate::config) allow_overlap: Option<bool>,
    pub(in crate::config) max_candidates: Option<usize>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingTrafficSection {
    pub(in crate::config) request_count: Option<u32>,
    pub(in crate::config) arrival: Option<String>,
    pub(in crate::config) arrival_gap_s: Option<f64>,
    pub(in crate::config) arrival_gap_ms: Option<f64>,
    pub(in crate::config) arrival_rate_per_s: Option<f64>,
    pub(in crate::config) arrival_seed: Option<u64>,
    pub(in crate::config) burst_size: Option<u32>,
    pub(in crate::config) burst_interval_s: Option<f64>,
    pub(in crate::config) burst_interval_ms: Option<f64>,
    #[serde(alias = "burst_gap_s", alias = "intra_burst_gap_s")]
    pub(in crate::config) burst_arrival_gap_s: Option<f64>,
    #[serde(alias = "burst_gap_ms", alias = "intra_burst_gap_ms")]
    pub(in crate::config) burst_arrival_gap_ms: Option<f64>,
    #[serde(alias = "min_arrival_rate_per_s", alias = "diurnal_trough_rate_per_s")]
    pub(in crate::config) diurnal_min_rate_per_s: Option<f64>,
    #[serde(alias = "max_arrival_rate_per_s", alias = "diurnal_peak_rate_per_s")]
    pub(in crate::config) diurnal_max_rate_per_s: Option<f64>,
    pub(in crate::config) diurnal_period_s: Option<f64>,
    pub(in crate::config) diurnal_period_ms: Option<f64>,
    pub(in crate::config) diurnal_phase_s: Option<f64>,
    pub(in crate::config) diurnal_phase_ms: Option<f64>,
    #[serde(alias = "selfsimilar_rate_per_s")]
    pub(in crate::config) self_similar_rate_per_s: Option<f64>,
    #[serde(
        alias = "self_similar_shape",
        alias = "self_similar_alpha",
        alias = "self_similar_pareto_alpha",
        alias = "pareto_shape",
        alias = "pareto_alpha"
    )]
    pub(in crate::config) self_similar_pareto_shape: Option<f64>,
    pub(in crate::config) self_similar_max_gap_s: Option<f64>,
    pub(in crate::config) self_similar_max_gap_ms: Option<f64>,
    pub(in crate::config) routing_policy: Option<String>,
    pub(in crate::config) prefill_batching: Option<String>,
    pub(in crate::config) max_prefill_batch_tokens: Option<u64>,
    #[serde(alias = "prefill_chunk_tokens", alias = "chunk_prefill_tokens")]
    pub(in crate::config) max_prefill_chunk_tokens: Option<u32>,
    pub(in crate::config) decode_batching: Option<String>,
    #[serde(alias = "decode_admission_policy", alias = "capacity_policy")]
    pub(in crate::config) decode_capacity_policy: Option<String>,
    #[serde(
        alias = "backpressure_penalty_weight",
        alias = "queue_backpressure_penalty_weight"
    )]
    pub(in crate::config) service_backpressure_penalty_weight: Option<f64>,
    #[serde(alias = "max_active_prefill_tokens")]
    pub(in crate::config) max_prefill_tokens: Option<u64>,
    #[serde(alias = "max_active_prefill_tokens_per_node")]
    pub(in crate::config) max_prefill_tokens_per_node: Option<u64>,
    #[serde(alias = "max_active_prefill_tokens_per_gpu")]
    pub(in crate::config) max_prefill_tokens_per_gpu: Option<u64>,
    #[serde(alias = "prefill_worker_slots_per_gpu", alias = "prefill_worker_slots")]
    pub(in crate::config) max_prefill_worker_slots_per_gpu: Option<u32>,
    pub(in crate::config) max_decode_batch_tokens: Option<u32>,
    pub(in crate::config) max_decode_sequences: Option<u32>,
    pub(in crate::config) max_resident_tokens: Option<u64>,
    pub(in crate::config) max_decode_sequences_per_node: Option<u32>,
    pub(in crate::config) max_resident_tokens_per_node: Option<u64>,
    pub(in crate::config) max_decode_sequences_per_gpu: Option<u32>,
    #[serde(alias = "decode_worker_slots_per_gpu", alias = "decode_worker_slots")]
    pub(in crate::config) max_decode_worker_slots_per_gpu: Option<u32>,
    pub(in crate::config) max_resident_tokens_per_gpu: Option<u64>,
    #[serde(
        alias = "kv_transfer_worker_slots_per_gpu",
        alias = "kv_transfer_worker_slots"
    )]
    pub(in crate::config) max_kv_transfer_worker_slots_per_gpu: Option<u32>,
    pub(in crate::config) kv_block_tokens: Option<u32>,
    pub(in crate::config) max_kv_blocks: Option<u64>,
    pub(in crate::config) max_kv_blocks_per_node: Option<u64>,
    pub(in crate::config) max_kv_blocks_per_gpu: Option<u64>,
    pub(in crate::config) ttft_slo_ms: Option<f64>,
    pub(in crate::config) tpot_slo_ms: Option<f64>,
    pub(in crate::config) itl_slo_ms: Option<f64>,
    pub(in crate::config) e2el_slo_ms: Option<f64>,
    pub(in crate::config) max_ttft_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_tpot_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_itl_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_e2el_slo_miss_rate: Option<f64>,
    pub(in crate::config) max_deadline_miss_rate: Option<f64>,
    #[serde(alias = "prefix_cache_hit_ratio")]
    pub(in crate::config) prefix_cache_hit_rate: Option<f64>,
    pub(in crate::config) shape_seed: Option<u64>,
    pub(in crate::config) batch_size_distribution: Option<ServingValueDistributionSection>,
    pub(in crate::config) prompt_tokens_distribution: Option<ServingValueDistributionSection>,
    pub(in crate::config) decode_tokens_distribution: Option<ServingValueDistributionSection>,
    pub(in crate::config) shape_profiles: Option<Vec<ServingShapeProfileSection>>,
    pub(in crate::config) batch_sizes: Option<Vec<u32>>,
    pub(in crate::config) prompt_tokens: Option<Vec<u32>>,
    pub(in crate::config) decode_tokens: Option<Vec<u32>>,
    pub(in crate::config) requests: Option<Vec<ServingTraceRequestSection>>,
    #[serde(alias = "trace_path", alias = "trace_file")]
    pub(in crate::config) trace_csv: Option<String>,
    #[serde(alias = "trace_json", alias = "trace_jsonl_path")]
    pub(in crate::config) trace_jsonl: Option<String>,
    pub(in crate::config) trace_start_s: Option<f64>,
    pub(in crate::config) trace_start_ms: Option<f64>,
    pub(in crate::config) trace_end_s: Option<f64>,
    pub(in crate::config) trace_end_ms: Option<f64>,
    pub(in crate::config) trace_time_scale: Option<f64>,
    pub(in crate::config) trace_arrival_offset_s: Option<f64>,
    pub(in crate::config) trace_arrival_offset_ms: Option<f64>,
    #[serde(
        alias = "trace_repeat",
        alias = "trace_repeats",
        alias = "trace_replay_count"
    )]
    pub(in crate::config) trace_repeat_count: Option<u32>,
    #[serde(alias = "trace_replay_interval_s")]
    pub(in crate::config) trace_repeat_interval_s: Option<f64>,
    #[serde(alias = "trace_replay_interval_ms")]
    pub(in crate::config) trace_repeat_interval_ms: Option<f64>,
    #[serde(alias = "metric_start_s")]
    pub(in crate::config) measurement_start_s: Option<f64>,
    #[serde(alias = "metric_start_ms")]
    pub(in crate::config) measurement_start_ms: Option<f64>,
    #[serde(alias = "metric_end_s")]
    pub(in crate::config) measurement_end_s: Option<f64>,
    #[serde(alias = "metric_end_ms")]
    pub(in crate::config) measurement_end_ms: Option<f64>,
    #[serde(alias = "metric_warmup_s", alias = "warmup_s")]
    pub(in crate::config) measurement_warmup_s: Option<f64>,
    #[serde(alias = "metric_warmup_ms", alias = "warmup_ms")]
    pub(in crate::config) measurement_warmup_ms: Option<f64>,
    #[serde(alias = "metric_cooldown_s", alias = "cooldown_s")]
    pub(in crate::config) measurement_cooldown_s: Option<f64>,
    #[serde(alias = "metric_cooldown_ms", alias = "cooldown_ms")]
    pub(in crate::config) measurement_cooldown_ms: Option<f64>,
    #[serde(
        alias = "auto_steady_state",
        alias = "steady_state",
        alias = "measurement_auto_steady_state"
    )]
    pub(in crate::config) measurement_steady_state: Option<bool>,
    pub(in crate::config) measurement_steady_state_min_requests: Option<u32>,
    pub(in crate::config) measurement_steady_state_max_cv: Option<f64>,
    pub(in crate::config) max_queue_delay_s: Option<f64>,
    pub(in crate::config) max_queue_delay_ms: Option<f64>,
    pub(in crate::config) max_kv_queue_delay_s: Option<f64>,
    pub(in crate::config) max_kv_queue_delay_ms: Option<f64>,
    pub(in crate::config) max_decode_queue_delay_s: Option<f64>,
    pub(in crate::config) max_decode_queue_delay_ms: Option<f64>,
    pub(in crate::config) max_decode_iteration_queue_delay_s: Option<f64>,
    pub(in crate::config) max_decode_iteration_queue_delay_ms: Option<f64>,
    pub(in crate::config) request_timeout_s: Option<f64>,
    pub(in crate::config) request_timeout_ms: Option<f64>,
    pub(in crate::config) services: Option<ServingServicesSection>,
    pub(in crate::config) prefill_service: Option<ServingServicePhaseSection>,
    pub(in crate::config) decode_service: Option<ServingServicePhaseSection>,
    pub(in crate::config) kv_transfer_service: Option<ServingServicePhaseSection>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingValueDistributionSection {
    pub(in crate::config) kind: String,
    pub(in crate::config) min: Option<u32>,
    pub(in crate::config) max: Option<u32>,
    pub(in crate::config) median: Option<f64>,
    pub(in crate::config) sigma: Option<f64>,
    pub(in crate::config) values: Option<Vec<u32>>,
    pub(in crate::config) weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingShapeProfileSection {
    #[serde(alias = "label")]
    pub(in crate::config) name: Option<String>,
    pub(in crate::config) weight: Option<f64>,
    #[serde(alias = "tenant_id")]
    pub(in crate::config) tenant: Option<String>,
    #[serde(alias = "model")]
    pub(in crate::config) model_id: Option<String>,
    #[serde(alias = "prefix_cache_key")]
    pub(in crate::config) cache_key: Option<String>,
    pub(in crate::config) priority: Option<i32>,
    pub(in crate::config) ttft_slo_s: Option<f64>,
    pub(in crate::config) ttft_slo_ms: Option<f64>,
    pub(in crate::config) tpot_slo_s: Option<f64>,
    pub(in crate::config) tpot_slo_ms: Option<f64>,
    pub(in crate::config) itl_slo_s: Option<f64>,
    pub(in crate::config) itl_slo_ms: Option<f64>,
    pub(in crate::config) e2el_slo_s: Option<f64>,
    pub(in crate::config) e2el_slo_ms: Option<f64>,
    pub(in crate::config) request_timeout_s: Option<f64>,
    pub(in crate::config) request_timeout_ms: Option<f64>,
    pub(in crate::config) batch_size: u32,
    pub(in crate::config) prompt_tokens: u32,
    pub(in crate::config) decode_tokens: u32,
    pub(in crate::config) max_sequence_tokens: Option<u32>,
    pub(in crate::config) prefix_cache_hit_tokens: Option<u32>,
    #[serde(alias = "prefix_cache_hit_ratio")]
    pub(in crate::config) prefix_cache_hit_rate: Option<f64>,
    pub(in crate::config) deadline_after_s: Option<f64>,
    pub(in crate::config) deadline_after_ms: Option<f64>,
    #[serde(alias = "cancellation_after_s")]
    pub(in crate::config) cancel_after_s: Option<f64>,
    #[serde(alias = "cancellation_after_ms")]
    pub(in crate::config) cancel_after_ms: Option<f64>,
}

#[derive(Deserialize)]
pub(in crate::config) struct ServingTraceRequestSection {
    #[serde(alias = "id")]
    pub(in crate::config) request_id: Option<String>,
    #[serde(alias = "tenant_id")]
    pub(in crate::config) tenant: Option<String>,
    #[serde(alias = "model")]
    pub(in crate::config) model_id: Option<String>,
    #[serde(alias = "prefix_cache_key")]
    pub(in crate::config) cache_key: Option<String>,
    #[serde(alias = "arrival", alias = "arrival_time_s")]
    pub(in crate::config) arrival_s: Option<f64>,
    #[serde(alias = "arrival_time_ms", alias = "timestamp_ms")]
    pub(in crate::config) arrival_ms: Option<f64>,
    pub(in crate::config) priority: Option<i32>,
    pub(in crate::config) ttft_slo_s: Option<f64>,
    pub(in crate::config) ttft_slo_ms: Option<f64>,
    pub(in crate::config) tpot_slo_s: Option<f64>,
    pub(in crate::config) tpot_slo_ms: Option<f64>,
    pub(in crate::config) itl_slo_s: Option<f64>,
    pub(in crate::config) itl_slo_ms: Option<f64>,
    pub(in crate::config) e2el_slo_s: Option<f64>,
    pub(in crate::config) e2el_slo_ms: Option<f64>,
    pub(in crate::config) deadline_s: Option<f64>,
    pub(in crate::config) deadline_ms: Option<f64>,
    pub(in crate::config) deadline_after_s: Option<f64>,
    pub(in crate::config) deadline_after_ms: Option<f64>,
    pub(in crate::config) cancellation_s: Option<f64>,
    pub(in crate::config) cancellation_ms: Option<f64>,
    pub(in crate::config) cancel_after_s: Option<f64>,
    pub(in crate::config) cancel_after_ms: Option<f64>,
    #[serde(alias = "batch")]
    pub(in crate::config) batch_size: u32,
    #[serde(alias = "input_tokens", alias = "prefill_tokens")]
    pub(in crate::config) prompt_tokens: u32,
    #[serde(alias = "output_tokens", alias = "max_new_tokens")]
    pub(in crate::config) decode_tokens: u32,
    #[serde(alias = "sequence_tokens")]
    pub(in crate::config) max_sequence_tokens: Option<u32>,
    pub(in crate::config) prefix_cache_hit_tokens: Option<u32>,
    #[serde(alias = "prefix_cache_hit_ratio")]
    pub(in crate::config) prefix_cache_hit_rate: Option<f64>,
}
