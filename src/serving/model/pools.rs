use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingPoolCandidate {
    pub label: Option<String>,
    pub prefill_nodes: Vec<NodeId>,
    pub decode_nodes: Vec<NodeId>,
    pub prefill_groups: Vec<String>,
    pub decode_groups: Vec<String>,
    pub prefill_node_filter: ServingPoolNodeFilter,
    pub decode_node_filter: ServingPoolNodeFilter,
    pub domain_spread: ServingPoolDomainSpread,
    pub prefill_gpu_labels: Vec<String>,
    pub decode_gpu_labels: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingPoolSearch {
    pub prefill_groups: Vec<String>,
    pub decode_groups: Vec<String>,
    pub prefill_node_counts: Vec<u32>,
    pub decode_node_counts: Vec<u32>,
    pub prefill_node_filter: ServingPoolNodeFilter,
    pub decode_node_filter: ServingPoolNodeFilter,
    pub prefill_gpu_labels: Vec<String>,
    pub decode_gpu_labels: Vec<String>,
    pub allow_overlap: bool,
    pub domain_spread: ServingPoolDomainSpread,
    pub max_candidates: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingPoolNodeFilter {
    pub node_labels: Vec<String>,
    pub racks: Vec<String>,
    pub islands: Vec<String>,
    pub failure_domains: Vec<String>,
    pub exclude_node_labels: Vec<String>,
    pub exclude_racks: Vec<String>,
    pub exclude_islands: Vec<String>,
    pub exclude_failure_domains: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingPoolDomainSpread {
    pub min_prefill_racks: Option<u32>,
    pub min_decode_racks: Option<u32>,
    pub min_prefill_islands: Option<u32>,
    pub min_decode_islands: Option<u32>,
    pub min_prefill_failure_domains: Option<u32>,
    pub min_decode_failure_domains: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingPoolSearchSummary {
    pub max_candidates: usize,
    pub generated_candidate_count: usize,
    pub truncated: bool,
    pub groups: Vec<ServingPoolSearchGroupSummary>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingPoolTopologySummary {
    pub prefill_node_count: usize,
    pub decode_node_count: usize,
    pub shared_node_count: usize,
    pub dedicated_prefill_node_count: usize,
    pub dedicated_decode_node_count: usize,
    pub prefill_racks: Vec<String>,
    pub decode_racks: Vec<String>,
    pub prefill_islands: Vec<String>,
    pub decode_islands: Vec<String>,
    pub prefill_failure_domains: Vec<String>,
    pub decode_failure_domains: Vec<String>,
    pub prefill_node_labels: Vec<String>,
    pub decode_node_labels: Vec<String>,
}

impl ServingPoolSearchSummary {
    pub fn considered_candidate_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.considered_candidate_count)
            .sum()
    }

    pub fn rejected_overlap_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.rejected_overlap_count)
            .sum()
    }

    pub fn rejected_mode_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.rejected_mode_count)
            .sum()
    }

    pub fn rejected_domain_spread_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.rejected_domain_spread_count)
            .sum()
    }

    pub fn duplicate_candidate_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.duplicate_candidate_count)
            .sum()
    }

    pub fn generated_colocated_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.generated_colocated_count)
            .sum()
    }

    pub fn generated_partially_disaggregated_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.generated_partially_disaggregated_count)
            .sum()
    }

    pub fn generated_fully_disaggregated_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.generated_fully_disaggregated_count)
            .sum()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingPoolSearchGroupSummary {
    pub prefill_group: String,
    pub decode_group: String,
    pub prefill_group_node_count: usize,
    pub prefill_node_filter_node_count: usize,
    pub prefill_gpu_filter_node_count: usize,
    pub decode_group_node_count: usize,
    pub decode_node_filter_node_count: usize,
    pub decode_gpu_filter_node_count: usize,
    pub prefill_node_counts: Vec<u32>,
    pub decode_node_counts: Vec<u32>,
    pub considered_candidate_count: usize,
    pub rejected_overlap_count: usize,
    pub rejected_mode_count: usize,
    pub rejected_domain_spread_count: usize,
    pub duplicate_candidate_count: usize,
    pub generated_candidate_count: usize,
    pub generated_colocated_count: usize,
    pub generated_partially_disaggregated_count: usize,
    pub generated_fully_disaggregated_count: usize,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingDeploymentMode {
    #[default]
    Flexible,
    Colocated,
    PartiallyDisaggregated,
    FullyDisaggregated,
}

impl ServingDeploymentMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Flexible => "flexible",
            Self::Colocated => "colocated",
            Self::PartiallyDisaggregated => "partially_disaggregated",
            Self::FullyDisaggregated => "fully_disaggregated",
        }
    }

    pub fn effective_for_pool(prefill_nodes: &[NodeId], decode_nodes: &[NodeId]) -> Self {
        if same_u32s(prefill_nodes, decode_nodes) {
            Self::Colocated
        } else if overlaps_nodes(prefill_nodes, decode_nodes) {
            Self::PartiallyDisaggregated
        } else {
            Self::FullyDisaggregated
        }
    }

    pub fn accepts_pool(self, prefill_nodes: &[NodeId], decode_nodes: &[NodeId]) -> bool {
        self == Self::Flexible || self == Self::effective_for_pool(prefill_nodes, decode_nodes)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DisaggregatedServingConfig {
    pub deployment_mode: ServingDeploymentMode,
    pub prefill_nodes: Vec<NodeId>,
    pub decode_nodes: Vec<NodeId>,
    pub pool_candidates: Vec<ServingPoolCandidate>,
    pub pool_search: Option<ServingPoolSearch>,
    pub objective: ServingObjective,
    pub slo_miss_penalty_weight: f64,
    pub slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
    pub topology_risk_penalty_weight: f64,
    pub max_memory_pressure_fraction: Option<f64>,
    pub max_unique_gpus: Option<u32>,
    pub min_throughput_tokens_per_s: Option<f64>,
    pub cost_model: ServingCostModel,
    pub search: ServingSearchSpace,
    pub traffic: ServingTraffic,
    pub slo_policies: Vec<ServingSloPolicy>,
}
