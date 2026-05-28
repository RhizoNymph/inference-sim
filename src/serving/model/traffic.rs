use super::*;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingMetricCeilings {
    pub max_ttft_s: Option<f64>,
    pub max_tpot_s: Option<f64>,
    pub max_itl_s: Option<f64>,
    pub max_e2el_s: Option<f64>,
}

impl ServingMetricCeilings {
    pub fn any(self) -> bool {
        self.max_ttft_s.is_some()
            || self.max_tpot_s.is_some()
            || self.max_itl_s.is_some()
            || self.max_e2el_s.is_some()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingKvRouteConstraints {
    pub min_inter_node_rail_count: Option<u32>,
    pub require_inter_node_rail_metadata: bool,
    pub require_gpudirect: bool,
}

impl ServingKvRouteConstraints {
    pub fn any(self) -> bool {
        self.min_inter_node_rail_count.is_some()
            || self.require_inter_node_rail_metadata
            || self.require_gpudirect
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingGpuCostRate {
    pub gpu_label: String,
    pub gpu_hour_usd: Option<f64>,
    pub watts: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingCostModel {
    pub default_gpu_hour_usd: Option<f64>,
    pub node_hour_usd: Option<f64>,
    pub kwh_usd: Option<f64>,
    pub default_gpu_watts: Option<f64>,
    pub node_watts: Option<f64>,
    pub gpu_rates: Vec<ServingGpuCostRate>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingCostEstimate {
    pub modeled_duration_s: Option<f64>,
    pub modeled_gpu_count: u32,
    pub modeled_node_count: u32,
    pub gpu_hours: Option<f64>,
    pub node_hours: Option<f64>,
    pub gpu_hour_cost_usd: Option<f64>,
    pub node_hour_cost_usd: Option<f64>,
    pub average_power_watts: Option<f64>,
    pub energy_kwh: Option<f64>,
    pub energy_cost_usd: Option<f64>,
    pub total_cost_usd: Option<f64>,
    pub cost_per_1k_output_tokens_usd: Option<f64>,
    pub cost_per_1k_requests_usd: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingTraffic {
    pub request_count: Option<u32>,
    pub arrival_gap_s: Option<f64>,
    pub arrival: ServingArrivalPattern,
    pub routing_policy: ServingRoutingPolicy,
    pub prefill_batching: ServingPrefillBatching,
    pub decode_batching: ServingDecodeBatching,
    pub decode_capacity_policy: ServingDecodeCapacityPolicy,
    pub services: ServingServicesConfig,
    pub service_backpressure_penalty_weight: f64,
    pub max_prefill_tokens: Option<u64>,
    pub max_prefill_tokens_per_node: Option<u64>,
    pub max_prefill_tokens_per_gpu: Option<u64>,
    pub max_prefill_worker_slots_per_gpu: Option<u32>,
    pub max_decode_sequences: Option<u32>,
    pub max_resident_tokens: Option<u64>,
    pub max_decode_sequences_per_node: Option<u32>,
    pub max_resident_tokens_per_node: Option<u64>,
    pub max_decode_sequences_per_gpu: Option<u32>,
    pub max_decode_worker_slots_per_gpu: Option<u32>,
    pub max_resident_tokens_per_gpu: Option<u64>,
    pub max_kv_transfer_worker_slots_per_gpu: Option<u32>,
    pub kv_block_tokens: Option<u32>,
    pub max_kv_blocks: Option<u64>,
    pub max_kv_blocks_per_node: Option<u64>,
    pub max_kv_blocks_per_gpu: Option<u64>,
    pub ttft_slo_s: Option<f64>,
    pub tpot_slo_s: Option<f64>,
    pub itl_slo_s: Option<f64>,
    pub e2el_slo_s: Option<f64>,
    pub max_ttft_slo_miss_rate: Option<f64>,
    pub max_tpot_slo_miss_rate: Option<f64>,
    pub max_itl_slo_miss_rate: Option<f64>,
    pub max_e2el_slo_miss_rate: Option<f64>,
    pub max_deadline_miss_rate: Option<f64>,
    pub metric_ceilings: ServingMetricCeilings,
    pub kv_route_constraints: ServingKvRouteConstraints,
    pub measurement_start_s: Option<f64>,
    pub measurement_end_s: Option<f64>,
    pub measurement_warmup_s: Option<f64>,
    pub measurement_cooldown_s: Option<f64>,
    pub measurement_steady_state: bool,
    pub measurement_steady_state_min_requests: Option<u32>,
    pub measurement_steady_state_max_cv: Option<f64>,
    pub max_queue_delay_s: Option<f64>,
    pub max_kv_queue_delay_s: Option<f64>,
    pub max_decode_queue_delay_s: Option<f64>,
    pub max_decode_iteration_queue_delay_s: Option<f64>,
    pub request_timeout_s: Option<f64>,
    pub shape_seed: u64,
    pub prefix_cache_hit_rate: Option<f64>,
    pub batch_size_distribution: Option<ServingValueDistribution>,
    pub prompt_tokens_distribution: Option<ServingValueDistribution>,
    pub decode_tokens_distribution: Option<ServingValueDistribution>,
    pub shape_profiles: Vec<ServingShapeProfile>,
    pub batch_sizes: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub decode_tokens: Vec<u32>,
    pub trace_requests: Vec<ServingTraceRequest>,
    pub traffic_classes: Vec<ServingTrafficClass>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingServiceHealth {
    #[default]
    Healthy,
    Draining,
    Unavailable,
}

impl ServingServiceHealth {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Draining => "draining",
            Self::Unavailable => "unavailable",
        }
    }

    pub(in crate::serving) fn accepts_requests(self) -> bool {
        matches!(self, Self::Healthy)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingServicePhaseConfig {
    pub health: ServingServiceHealth,
    pub worker_scale: f64,
}

impl Default for ServingServicePhaseConfig {
    fn default() -> Self {
        Self {
            health: ServingServiceHealth::Healthy,
            worker_scale: 1.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingServicesConfig {
    pub prefill: ServingServicePhaseConfig,
    pub decode: ServingServicePhaseConfig,
    pub kv_transfer: ServingServicePhaseConfig,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingSloPolicy {
    pub group: String,
    pub key: String,
    pub max_ttft_slo_miss_rate: Option<f64>,
    pub max_tpot_slo_miss_rate: Option<f64>,
    pub max_itl_slo_miss_rate: Option<f64>,
    pub max_e2el_slo_miss_rate: Option<f64>,
    pub max_deadline_miss_rate: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingTrafficClass {
    pub name: String,
    pub group: String,
    pub key: String,
    pub admission_priority: Option<i32>,
    pub max_prefill_tokens: Option<u64>,
    pub max_decode_sequences: Option<u32>,
    pub max_resident_tokens: Option<u64>,
    pub max_kv_blocks: Option<u64>,
    pub slo: ServingRequestSlo,
    pub slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
    pub max_queue_delay_s: Option<f64>,
    pub max_kv_queue_delay_s: Option<f64>,
    pub max_decode_queue_delay_s: Option<f64>,
    pub max_decode_iteration_queue_delay_s: Option<f64>,
    pub request_timeout_s: Option<f64>,
    pub max_ttft_slo_miss_rate: Option<f64>,
    pub max_tpot_slo_miss_rate: Option<f64>,
    pub max_itl_slo_miss_rate: Option<f64>,
    pub max_e2el_slo_miss_rate: Option<f64>,
    pub max_deadline_miss_rate: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingTraceRequest {
    pub request_id: Option<String>,
    pub tenant: Option<String>,
    pub model_id: Option<String>,
    pub cache_key: Option<String>,
    pub arrival_s: f64,
    pub priority: i32,
    pub batch_size: u32,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub max_sequence_tokens: Option<u32>,
    pub prefix_cache_hit_tokens: Option<u32>,
    pub prefix_cache_hit_rate: Option<f64>,
    pub slo: ServingRequestSlo,
    pub deadline_s: Option<f64>,
    pub cancellation_s: Option<f64>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingRequestSlo {
    pub ttft_s: Option<f64>,
    pub tpot_s: Option<f64>,
    pub itl_s: Option<f64>,
    pub e2el_s: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum ServingArrivalPattern {
    #[default]
    FixedGap,
    Poisson {
        rate_per_s: f64,
        seed: u64,
    },
    Bursty {
        burst_size: u32,
        burst_interval_s: f64,
        intra_burst_gap_s: f64,
    },
    Diurnal {
        min_rate_per_s: f64,
        max_rate_per_s: f64,
        period_s: f64,
        phase_s: f64,
        seed: u64,
    },
    SelfSimilar {
        rate_per_s: f64,
        pareto_shape: f64,
        max_gap_s: Option<f64>,
        seed: u64,
    },
    TraceDerived,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingRoutingPolicy {
    #[default]
    RoundRobin,
    TopologyAware,
}

impl ServingRoutingPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::TopologyAware => "topology_aware",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingPrefillBatching {
    #[default]
    Independent,
    Continuous {
        max_batch_tokens: Option<u64>,
        chunk_tokens: Option<u32>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingDecodeBatching {
    #[default]
    Independent,
    Continuous {
        max_batch_tokens: Option<u32>,
    },
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingDecodeCapacityPolicy {
    #[default]
    CandidateReject,
    RequestReject,
}

impl ServingDecodeCapacityPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CandidateReject => "candidate_reject",
            Self::RequestReject => "request_reject",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ServingValueDistribution {
    Uniform {
        min: u32,
        max: u32,
    },
    Weighted {
        values: Vec<u32>,
        weights: Vec<f64>,
    },
    LogNormal {
        median: f64,
        sigma: f64,
        min: u32,
        max: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingShapeProfile {
    pub name: String,
    pub weight: f64,
    pub tenant: Option<String>,
    pub model_id: Option<String>,
    pub cache_key: Option<String>,
    pub priority: Option<i32>,
    pub batch_size: u32,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub max_sequence_tokens: Option<u32>,
    pub prefix_cache_hit_tokens: Option<u32>,
    pub prefix_cache_hit_rate: Option<f64>,
    pub slo: ServingRequestSlo,
    pub request_timeout_s: Option<f64>,
    pub deadline_after_s: Option<f64>,
    pub cancellation_after_s: Option<f64>,
}

impl ServingTraffic {
    pub(in crate::serving) fn request_count(&self, calibration: SimulationCalibration) -> u32 {
        if !self.trace_requests.is_empty() {
            return self.trace_requests.len().min(u32::MAX as usize) as u32;
        }
        self.request_count
            .unwrap_or(calibration.serving_pipeline_depth)
            .max(1)
    }

    pub(in crate::serving) fn arrival_gap_s(&self, calibration: SimulationCalibration) -> f64 {
        self.arrival_gap_s
            .unwrap_or(calibration.request_arrival_gap_s)
            .max(0.0)
    }

    pub(in crate::serving) fn arrival_times(
        &self,
        request_count: u32,
        calibration: SimulationCalibration,
    ) -> Vec<f64> {
        if !self.trace_requests.is_empty() {
            return self
                .trace_requests
                .iter()
                .take(request_count as usize)
                .map(|request| request.arrival_s)
                .collect();
        }
        match self.arrival {
            ServingArrivalPattern::FixedGap => {
                let gap_s = self.arrival_gap_s(calibration);
                (0..request_count)
                    .map(|idx| f64::from(idx) * gap_s)
                    .collect()
            }
            ServingArrivalPattern::Poisson { rate_per_s, seed } => {
                poisson_arrival_times(request_count, rate_per_s, seed)
            }
            ServingArrivalPattern::Bursty {
                burst_size,
                burst_interval_s,
                intra_burst_gap_s,
            } => bursty_arrival_times(
                request_count,
                burst_size,
                burst_interval_s,
                intra_burst_gap_s,
            ),
            ServingArrivalPattern::Diurnal {
                min_rate_per_s,
                max_rate_per_s,
                period_s,
                phase_s,
                seed,
            } => diurnal_arrival_times(
                request_count,
                min_rate_per_s,
                max_rate_per_s,
                period_s,
                phase_s,
                seed,
            ),
            ServingArrivalPattern::SelfSimilar {
                rate_per_s,
                pareto_shape,
                max_gap_s,
                seed,
            } => {
                self_similar_arrival_times(request_count, rate_per_s, pareto_shape, max_gap_s, seed)
            }
            ServingArrivalPattern::TraceDerived => {
                let gap_s = self.arrival_gap_s(calibration);
                (0..request_count)
                    .map(|idx| f64::from(idx) * gap_s)
                    .collect()
            }
        }
    }

    pub(in crate::serving) fn request_at(
        &self,
        base: &InferenceRequest,
        idx: u32,
    ) -> InferenceRequest {
        if let Some(trace_request) = self.trace_request_at(idx) {
            return InferenceRequest {
                batch_size: trace_request.batch_size.max(1),
                prompt_tokens: trace_request.prompt_tokens.max(1),
                decode_tokens: trace_request.decode_tokens.max(1),
                max_sequence_tokens: trace_request
                    .max_sequence_tokens
                    .unwrap_or(base.max_sequence_tokens)
                    .max(
                        trace_request
                            .prompt_tokens
                            .saturating_add(trace_request.decode_tokens),
                    )
                    .max(1),
                phase: base.phase,
            };
        }
        if let Some(profile) = self.shape_profile_at(idx) {
            let batch_size = profile.batch_size.max(1);
            let prompt_tokens = profile.prompt_tokens.max(1);
            let decode_tokens = profile.decode_tokens.max(1);
            return InferenceRequest {
                batch_size,
                prompt_tokens,
                decode_tokens,
                max_sequence_tokens: profile
                    .max_sequence_tokens
                    .unwrap_or(base.max_sequence_tokens)
                    .max(prompt_tokens.saturating_add(decode_tokens))
                    .max(1),
                phase: base.phase,
            };
        }
        InferenceRequest {
            batch_size: sample_or_cyclic(
                self.batch_size_distribution.as_ref(),
                &self.batch_sizes,
                self.shape_seed,
                0xB47C_5512_AE4F_10C1,
                idx,
                base.batch_size,
            ),
            prompt_tokens: sample_or_cyclic(
                self.prompt_tokens_distribution.as_ref(),
                &self.prompt_tokens,
                self.shape_seed,
                0xA11C_7E57_9D13_04F1,
                idx,
                base.prompt_tokens,
            ),
            decode_tokens: sample_or_cyclic(
                self.decode_tokens_distribution.as_ref(),
                &self.decode_tokens,
                self.shape_seed,
                0xDEC0_DE70_4E51_A123,
                idx,
                base.decode_tokens,
            ),
            max_sequence_tokens: base.max_sequence_tokens,
            phase: base.phase,
        }
    }

    pub(in crate::serving) fn trace_arrivals_only(&self) -> bool {
        matches!(self.arrival, ServingArrivalPattern::TraceDerived)
    }

    pub(in crate::serving) fn trace_request_at(&self, idx: u32) -> Option<&ServingTraceRequest> {
        if self.trace_arrivals_only() {
            None
        } else {
            self.trace_requests.get(idx as usize)
        }
    }

    pub(in crate::serving) fn shape_profile_at(&self, idx: u32) -> Option<&ServingShapeProfile> {
        if self.shape_profiles.is_empty() {
            return None;
        }
        let total_weight = self
            .shape_profiles
            .iter()
            .map(|profile| profile.weight)
            .filter(|weight| weight.is_finite() && *weight > 0.0)
            .sum::<f64>();
        if total_weight <= 0.0 {
            let profile_idx = idx as usize % self.shape_profiles.len();
            return self.shape_profiles.get(profile_idx);
        }

        let mut rng = LcgRng::new(
            self.shape_seed
                ^ 0x5A4A_9E57_20B7_4C13
                ^ u64::from(idx).wrapping_mul(0xD1B5_4A32_D192_ED03),
        );
        let mut target = rng.next_open_unit_f64() * total_weight;
        for profile in &self.shape_profiles {
            let weight = profile.weight;
            if !weight.is_finite() || weight <= 0.0 {
                continue;
            }
            if target <= weight {
                return Some(profile);
            }
            target -= weight;
        }
        self.shape_profiles
            .iter()
            .rev()
            .find(|profile| profile.weight.is_finite() && profile.weight > 0.0)
    }

    pub(in crate::serving) fn shape_profile_name(&self, idx: u32) -> Option<String> {
        self.shape_profile_at(idx)
            .map(|profile| profile.name.clone())
    }

    pub(in crate::serving) fn cancellation_s(&self, idx: u32, arrival_s: f64) -> Option<f64> {
        self.trace_request_at(idx)
            .and_then(|request| request.cancellation_s)
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.cancellation_after_s)
                    .map(|offset_s| arrival_s + offset_s)
            })
    }

    pub(in crate::serving) fn deadline_s(&self, idx: u32, arrival_s: f64) -> Option<f64> {
        self.trace_request_at(idx)
            .and_then(|request| request.deadline_s)
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.deadline_after_s)
                    .map(|offset_s| arrival_s + offset_s)
            })
    }

    pub(in crate::serving) fn priority(&self, idx: u32) -> i32 {
        self.trace_request_at(idx)
            .map(|request| request.priority)
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.priority)
            })
            .unwrap_or(0)
    }

    pub(in crate::serving) fn effective_priority(&self, idx: u32) -> i32 {
        self.traffic_class(idx)
            .and_then(|class| class.admission_priority)
            .unwrap_or_else(|| self.priority(idx))
    }

    pub(in crate::serving) fn request_id(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.request_id.clone())
    }

    pub(in crate::serving) fn tenant(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.tenant.clone())
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.tenant.clone())
            })
    }

    pub(in crate::serving) fn model_id(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.model_id.clone())
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.model_id.clone())
            })
    }

    pub(in crate::serving) fn traffic_class(&self, idx: u32) -> Option<&ServingTrafficClass> {
        let tenant = self.tenant(idx);
        let model_id = self.model_id(idx);
        let priority = self.priority(idx);
        self.traffic_classes.iter().find(|class| {
            traffic_class_matches(
                &class.group,
                &class.key,
                tenant.as_deref(),
                model_id.as_deref(),
                priority,
            )
        })
    }

    pub(in crate::serving) fn traffic_class_name(&self, idx: u32) -> Option<String> {
        self.traffic_class(idx).map(|class| class.name.clone())
    }

    pub(in crate::serving) fn cache_key(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.cache_key.clone())
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.cache_key.clone())
            })
    }

    pub(in crate::serving) fn prefix_cache_hit_tokens(&self, idx: u32, prompt_tokens: u32) -> u32 {
        let prompt_tokens = prompt_tokens.max(1);
        let hit_tokens = self
            .trace_request_at(idx)
            .and_then(|request| request.prefix_cache_hit_tokens)
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.prefix_cache_hit_tokens)
            })
            .or_else(|| {
                let rate = self
                    .trace_request_at(idx)
                    .and_then(|request| request.prefix_cache_hit_rate)
                    .or_else(|| {
                        self.shape_profile_at(idx)
                            .and_then(|profile| profile.prefix_cache_hit_rate)
                    })
                    .or(self.prefix_cache_hit_rate)?;
                Some((f64::from(prompt_tokens) * rate.clamp(0.0, 1.0)).floor() as u32)
            })
            .unwrap_or(0);
        hit_tokens.min(prompt_tokens)
    }

    pub(in crate::serving) fn effective_slo(&self, idx: u32) -> ServingRequestSlo {
        let request_slo = self
            .trace_request_at(idx)
            .map(|request| request.slo)
            .unwrap_or_default();
        let profile_slo = self
            .shape_profile_at(idx)
            .map(|profile| profile.slo)
            .unwrap_or_default();
        let class_slo = self
            .traffic_class(idx)
            .map(|class| class.slo)
            .unwrap_or_default();
        ServingRequestSlo {
            ttft_s: request_slo
                .ttft_s
                .or(profile_slo.ttft_s)
                .or(class_slo.ttft_s)
                .or(self.ttft_slo_s),
            tpot_s: request_slo
                .tpot_s
                .or(profile_slo.tpot_s)
                .or(class_slo.tpot_s)
                .or(self.tpot_slo_s),
            itl_s: request_slo
                .itl_s
                .or(profile_slo.itl_s)
                .or(class_slo.itl_s)
                .or(self.itl_slo_s),
            e2el_s: request_slo
                .e2el_s
                .or(profile_slo.e2el_s)
                .or(class_slo.e2el_s)
                .or(self.e2el_slo_s),
        }
    }

    pub(in crate::serving) fn effective_max_queue_delay_s(&self, idx: u32) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_queue_delay_s)
            .or(self.max_queue_delay_s)
    }

    pub(in crate::serving) fn effective_max_kv_queue_delay_s(&self, idx: u32) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_kv_queue_delay_s)
            .or(self.max_kv_queue_delay_s)
    }

    pub(in crate::serving) fn effective_max_decode_queue_delay_s(&self, idx: u32) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_decode_queue_delay_s)
            .or(self.max_decode_queue_delay_s)
    }

    pub(in crate::serving) fn effective_max_decode_iteration_queue_delay_s(
        &self,
        idx: u32,
    ) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_decode_iteration_queue_delay_s)
            .or(self.max_decode_iteration_queue_delay_s)
    }

    pub(in crate::serving) fn effective_request_timeout_s(&self, idx: u32) -> Option<f64> {
        self.shape_profile_at(idx)
            .and_then(|profile| profile.request_timeout_s)
            .or_else(|| {
                self.traffic_class(idx)
                    .and_then(|class| class.request_timeout_s)
            })
            .or(self.request_timeout_s)
    }
}

fn traffic_class_matches(
    group: &str,
    key: &str,
    tenant: Option<&str>,
    model_id: Option<&str>,
    priority: i32,
) -> bool {
    match group {
        "tenant" => tenant == Some(key),
        "model_id" => model_id == Some(key),
        "priority" => key == priority_key(priority),
        _ => false,
    }
}
