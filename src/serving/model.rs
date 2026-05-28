use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingSearchSpace {
    pub prefill: SearchSpace,
    pub decode: SearchSpace,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ServingObjective {
    #[default]
    MinimizeE2el,
    MinimizeTtft,
    MinimizeTpot,
    MaximizeThroughput,
    MinimizeSloMissRate,
    MinimizeMemoryPressure,
    MinimizeCost,
    MinimizeEnergy,
    MinimizePower,
}

impl ServingObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingObjective::MinimizeE2el => "minimize_e2el",
            ServingObjective::MinimizeTtft => "minimize_ttft",
            ServingObjective::MinimizeTpot => "minimize_tpot",
            ServingObjective::MaximizeThroughput => "maximize_throughput",
            ServingObjective::MinimizeSloMissRate => "minimize_slo_miss_rate",
            ServingObjective::MinimizeMemoryPressure => "minimize_memory_pressure",
            ServingObjective::MinimizeCost => "minimize_cost",
            ServingObjective::MinimizeEnergy => "minimize_energy",
            ServingObjective::MinimizePower => "minimize_power",
        }
    }
}

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

    pub(super) fn accepts_requests(self) -> bool {
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
    pub(super) fn request_count(&self, calibration: SimulationCalibration) -> u32 {
        if !self.trace_requests.is_empty() {
            return self.trace_requests.len().min(u32::MAX as usize) as u32;
        }
        self.request_count
            .unwrap_or(calibration.serving_pipeline_depth)
            .max(1)
    }

    pub(super) fn arrival_gap_s(&self, calibration: SimulationCalibration) -> f64 {
        self.arrival_gap_s
            .unwrap_or(calibration.request_arrival_gap_s)
            .max(0.0)
    }

    pub(super) fn arrival_times(
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

    pub(super) fn request_at(&self, base: &InferenceRequest, idx: u32) -> InferenceRequest {
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

    pub(super) fn trace_arrivals_only(&self) -> bool {
        matches!(self.arrival, ServingArrivalPattern::TraceDerived)
    }

    pub(super) fn trace_request_at(&self, idx: u32) -> Option<&ServingTraceRequest> {
        if self.trace_arrivals_only() {
            None
        } else {
            self.trace_requests.get(idx as usize)
        }
    }

    pub(super) fn shape_profile_at(&self, idx: u32) -> Option<&ServingShapeProfile> {
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

    pub(super) fn shape_profile_name(&self, idx: u32) -> Option<String> {
        self.shape_profile_at(idx)
            .map(|profile| profile.name.clone())
    }

    pub(super) fn cancellation_s(&self, idx: u32, arrival_s: f64) -> Option<f64> {
        self.trace_request_at(idx)
            .and_then(|request| request.cancellation_s)
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.cancellation_after_s)
                    .map(|offset_s| arrival_s + offset_s)
            })
    }

    pub(super) fn deadline_s(&self, idx: u32, arrival_s: f64) -> Option<f64> {
        self.trace_request_at(idx)
            .and_then(|request| request.deadline_s)
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.deadline_after_s)
                    .map(|offset_s| arrival_s + offset_s)
            })
    }

    pub(super) fn priority(&self, idx: u32) -> i32 {
        self.trace_request_at(idx)
            .map(|request| request.priority)
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.priority)
            })
            .unwrap_or(0)
    }

    pub(super) fn effective_priority(&self, idx: u32) -> i32 {
        self.traffic_class(idx)
            .and_then(|class| class.admission_priority)
            .unwrap_or_else(|| self.priority(idx))
    }

    pub(super) fn request_id(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.request_id.clone())
    }

    pub(super) fn tenant(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.tenant.clone())
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.tenant.clone())
            })
    }

    pub(super) fn model_id(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.model_id.clone())
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.model_id.clone())
            })
    }

    pub(super) fn traffic_class(&self, idx: u32) -> Option<&ServingTrafficClass> {
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

    pub(super) fn traffic_class_name(&self, idx: u32) -> Option<String> {
        self.traffic_class(idx).map(|class| class.name.clone())
    }

    pub(super) fn cache_key(&self, idx: u32) -> Option<String> {
        self.trace_request_at(idx)
            .and_then(|request| request.cache_key.clone())
            .or_else(|| {
                self.shape_profile_at(idx)
                    .and_then(|profile| profile.cache_key.clone())
            })
    }

    pub(super) fn prefix_cache_hit_tokens(&self, idx: u32, prompt_tokens: u32) -> u32 {
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

    pub(super) fn effective_slo(&self, idx: u32) -> ServingRequestSlo {
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

    pub(super) fn effective_max_queue_delay_s(&self, idx: u32) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_queue_delay_s)
            .or(self.max_queue_delay_s)
    }

    pub(super) fn effective_max_kv_queue_delay_s(&self, idx: u32) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_kv_queue_delay_s)
            .or(self.max_kv_queue_delay_s)
    }

    pub(super) fn effective_max_decode_queue_delay_s(&self, idx: u32) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_decode_queue_delay_s)
            .or(self.max_decode_queue_delay_s)
    }

    pub(super) fn effective_max_decode_iteration_queue_delay_s(&self, idx: u32) -> Option<f64> {
        self.traffic_class(idx)
            .and_then(|class| class.max_decode_iteration_queue_delay_s)
            .or(self.max_decode_iteration_queue_delay_s)
    }

    pub(super) fn effective_request_timeout_s(&self, idx: u32) -> Option<f64> {
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

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingSloMissPenaltyWeights {
    pub aggregate: f64,
    pub ttft: f64,
    pub tpot: f64,
    pub itl: f64,
    pub e2el: f64,
    pub deadline: f64,
}

impl ServingSloMissPenaltyWeights {
    pub fn from_aggregate(aggregate: f64) -> Self {
        Self {
            aggregate,
            ..Self::default()
        }
    }

    pub fn any_nonzero(self) -> bool {
        self.aggregate > 0.0
            || self.ttft > 0.0
            || self.tpot > 0.0
            || self.itl > 0.0
            || self.e2el > 0.0
            || self.deadline > 0.0
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingSloMissPenaltyComponents {
    pub ttft: f64,
    pub tpot: f64,
    pub itl: f64,
    pub e2el: f64,
    pub deadline: f64,
    pub total: f64,
}

impl ServingSloMissPenaltyComponents {
    pub(super) fn add_assign(&mut self, other: Self) {
        self.ttft += other.ttft;
        self.tpot += other.tpot;
        self.itl += other.itl;
        self.e2el += other.e2el;
        self.deadline += other.deadline;
        self.total += other.total;
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingMetrics {
    pub ttft_s: f64,
    pub ttft_p50_s: f64,
    pub ttft_p90_s: f64,
    pub ttft_p95_s: f64,
    pub ttft_p99_s: f64,
    pub ttft_max_s: f64,
    pub ttft_slo_constrained_requests: u32,
    pub ttft_slo_missed_requests: u32,
    pub ttft_slo_miss_rate: f64,
    pub tpot_s: f64,
    pub tpot_p50_s: f64,
    pub tpot_p90_s: f64,
    pub tpot_p95_s: f64,
    pub tpot_p99_s: f64,
    pub tpot_max_s: f64,
    pub tpot_slo_constrained_requests: u32,
    pub tpot_slo_missed_requests: u32,
    pub tpot_slo_miss_rate: f64,
    pub itl_s: f64,
    pub itl_p50_s: f64,
    pub itl_p90_s: f64,
    pub itl_p95_s: f64,
    pub itl_p99_s: f64,
    pub itl_max_s: f64,
    pub itl_slo_constrained_requests: u32,
    pub itl_slo_missed_requests: u32,
    pub itl_slo_miss_rate: f64,
    pub decode_iterations: u64,
    pub decode_iteration_s: f64,
    pub decode_iteration_p50_s: f64,
    pub decode_iteration_p90_s: f64,
    pub decode_iteration_p95_s: f64,
    pub decode_iteration_p99_s: f64,
    pub decode_iteration_max_s: f64,
    pub throughput_tokens_per_s: f64,
    pub e2el_s: f64,
    pub e2el_p50_s: f64,
    pub e2el_p90_s: f64,
    pub e2el_p95_s: f64,
    pub e2el_p99_s: f64,
    pub e2el_max_s: f64,
    pub e2el_slo_constrained_requests: u32,
    pub e2el_slo_missed_requests: u32,
    pub e2el_slo_miss_rate: f64,
    pub deadline_miss_rate: f64,
    pub service_s: f64,
    pub prefill_s: f64,
    pub prefill_chunks: u64,
    pub prompt_tokens: u64,
    pub prefix_cache_hit_tokens: u64,
    pub effective_prefill_tokens: u64,
    pub prefix_cache_hit_rate: f64,
    pub kv_transfer_s: f64,
    pub kv_queue_s: f64,
    pub kv_worker_queue_s: f64,
    pub kv_resource_queue_s: f64,
    pub decode_queue_s: f64,
    pub prefill_worker_queue_s: f64,
    pub prefill_resource_queue_s: f64,
    pub decode_worker_queue_s: f64,
    pub decode_resource_queue_s: f64,
    pub decode_s: f64,
    pub queue_delay_s: f64,
    pub queue_delay_p90_s: f64,
    pub queue_delay_p95_s: f64,
    pub queue_delay_max_s: f64,
    pub peak_prefill_tokens: u64,
    pub peak_prefill_tokens_per_node: u64,
    pub peak_prefill_tokens_per_gpu: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_decode_sequences_per_node: u32,
    pub peak_resident_tokens_per_node: u64,
    pub peak_decode_sequences_per_gpu: u32,
    pub peak_resident_tokens_per_gpu: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub peak_kv_blocks_per_node: u64,
    pub peak_allocated_kv_tokens_per_node: u64,
    pub peak_kv_fragmentation_tokens_per_node: u64,
    pub peak_kv_block_table_bytes_per_node: u64,
    pub peak_kv_blocks_per_gpu: u64,
    pub peak_allocated_kv_tokens_per_gpu: u64,
    pub peak_kv_fragmentation_tokens_per_gpu: u64,
    pub peak_kv_block_table_bytes_per_gpu: u64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
    pub decode_sequence_per_node_utilization: f64,
    pub resident_token_per_node_utilization: f64,
    pub kv_block_per_node_utilization: f64,
    pub decode_sequence_per_gpu_utilization: f64,
    pub resident_token_per_gpu_utilization: f64,
    pub kv_block_per_gpu_utilization: f64,
    pub scheduled_makespan_s: f64,
    pub scheduled_requests: u32,
    pub admitted_requests: u32,
    pub completed_requests: u32,
    pub rejected_requests: u32,
    pub timed_out_requests: u32,
    pub cancelled_requests: u32,
    pub deadline_constrained_requests: u32,
    pub deadline_missed_requests: u32,
    pub measured_requests: u32,
    pub measurement_start_s: f64,
    pub measurement_end_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingMetricBreakdown {
    pub group: String,
    pub key: String,
    pub request_count: u32,
    pub completed_requests: u32,
    pub failed_requests: u32,
    pub rejected_requests: u32,
    pub timed_out_requests: u32,
    pub cancelled_requests: u32,
    pub output_tokens: u64,
    pub lifecycle_event_metric_request_count: u32,
    pub fallback_metric_request_count: u32,
    pub metric_source_counts: Vec<ServingMeasurementMetricSourceCount>,
    pub deadline_constrained_requests: u32,
    pub deadline_missed_requests: u32,
    pub deadline_miss_rate: f64,
    pub ttft_slo_constrained_requests: u32,
    pub ttft_slo_missed_requests: u32,
    pub ttft_slo_miss_rate: f64,
    pub tpot_slo_constrained_requests: u32,
    pub tpot_slo_missed_requests: u32,
    pub tpot_slo_miss_rate: f64,
    pub itl_slo_constrained_requests: u32,
    pub itl_slo_missed_requests: u32,
    pub itl_slo_miss_rate: f64,
    pub e2el_slo_constrained_requests: u32,
    pub e2el_slo_missed_requests: u32,
    pub e2el_slo_miss_rate: f64,
    pub ttft_s: f64,
    pub ttft_p90_s: f64,
    pub ttft_p95_s: f64,
    pub ttft_max_s: f64,
    pub tpot_s: f64,
    pub tpot_p90_s: f64,
    pub tpot_p95_s: f64,
    pub tpot_max_s: f64,
    pub itl_s: f64,
    pub itl_p90_s: f64,
    pub itl_p95_s: f64,
    pub itl_max_s: f64,
    pub throughput_tokens_per_s: f64,
    pub e2el_s: f64,
    pub e2el_p90_s: f64,
    pub e2el_p95_s: f64,
    pub e2el_max_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingSteadyStateMetricObservation {
    pub metric: String,
    pub unit: String,
    pub sample_count: u32,
    pub mean: f64,
    pub stddev: f64,
    pub cv: f64,
    pub std_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingSteadyStateUtilizationObservation {
    pub source: String,
    pub phase: String,
    pub resource_kind: String,
    pub resource: String,
    pub bucket_count: u32,
    pub active_bucket_count: u32,
    pub event_count: u32,
    pub mean_utilization: f64,
    pub max_utilization: f64,
    pub utilization_cv: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingMeasurementMetricSourceCount {
    pub metric_source: String,
    pub request_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingMeasurementWindowObservation {
    pub source: String,
    pub start_s: f64,
    pub end_s: f64,
    pub duration_s: f64,
    pub request_count: u32,
    pub completed_request_count: u32,
    pub failed_request_count: u32,
    pub rejected_request_count: u32,
    pub timed_out_request_count: u32,
    pub cancelled_request_count: u32,
    pub deadline_constrained_request_count: u32,
    pub deadline_missed_request_count: u32,
    pub measured_requests: u32,
    pub lifecycle_event_metric_request_count: u32,
    pub fallback_metric_request_count: u32,
    pub metric_source_counts: Vec<ServingMeasurementMetricSourceCount>,
    pub configured_start_bound: bool,
    pub configured_end_bound: bool,
    pub steady_state_requested: bool,
    pub steady_state_applied: bool,
    pub steady_state_candidate_start_s: Option<f64>,
    pub steady_state_candidate_end_s: Option<f64>,
    pub steady_state_min_requests: Option<u32>,
    pub steady_state_max_cv: Option<f64>,
    pub steady_state_sample_count: u32,
    pub steady_state_matching_window_count: u32,
    pub steady_state_candidate_request_count: Option<u32>,
    pub steady_state_candidate_e2el_mean_s: Option<f64>,
    pub steady_state_candidate_e2el_stddev_s: Option<f64>,
    pub steady_state_candidate_e2el_cv: Option<f64>,
    pub steady_state_candidate_e2el_std_error_s: Option<f64>,
    pub steady_state_candidate_metric_count: u32,
    pub steady_state_candidate_worst_metric: Option<String>,
    pub steady_state_candidate_worst_cv: Option<f64>,
    pub steady_state_candidate_output_tokens: Option<u64>,
    pub steady_state_candidate_throughput_tokens_per_s: Option<f64>,
    pub steady_state_candidate_metrics: Vec<ServingSteadyStateMetricObservation>,
    pub steady_state_candidate_utilization_count: u32,
    pub steady_state_candidate_worst_utilization_resource: Option<String>,
    pub steady_state_candidate_worst_utilization_cv: Option<f64>,
    pub steady_state_candidate_utilization: Vec<ServingSteadyStateUtilizationObservation>,
}

impl ServingMeasurementWindowObservation {
    pub(super) fn rejected() -> Self {
        Self {
            source: "unavailable".to_string(),
            start_s: f64::INFINITY,
            end_s: f64::INFINITY,
            duration_s: f64::INFINITY,
            request_count: 0,
            completed_request_count: 0,
            failed_request_count: 0,
            rejected_request_count: 0,
            timed_out_request_count: 0,
            cancelled_request_count: 0,
            deadline_constrained_request_count: 0,
            deadline_missed_request_count: 0,
            measured_requests: 0,
            lifecycle_event_metric_request_count: 0,
            fallback_metric_request_count: 0,
            metric_source_counts: Vec::new(),
            configured_start_bound: false,
            configured_end_bound: false,
            steady_state_requested: false,
            steady_state_applied: false,
            steady_state_candidate_start_s: None,
            steady_state_candidate_end_s: None,
            steady_state_min_requests: None,
            steady_state_max_cv: None,
            steady_state_sample_count: 0,
            steady_state_matching_window_count: 0,
            steady_state_candidate_request_count: None,
            steady_state_candidate_e2el_mean_s: None,
            steady_state_candidate_e2el_stddev_s: None,
            steady_state_candidate_e2el_cv: None,
            steady_state_candidate_e2el_std_error_s: None,
            steady_state_candidate_metric_count: 0,
            steady_state_candidate_worst_metric: None,
            steady_state_candidate_worst_cv: None,
            steady_state_candidate_output_tokens: None,
            steady_state_candidate_throughput_tokens_per_s: None,
            steady_state_candidate_metrics: Vec::new(),
            steady_state_candidate_utilization_count: 0,
            steady_state_candidate_worst_utilization_resource: None,
            steady_state_candidate_worst_utilization_cv: None,
            steady_state_candidate_utilization: Vec::new(),
        }
    }
}

impl ServingMetrics {
    pub(super) fn rejected() -> Self {
        Self {
            ttft_s: f64::INFINITY,
            ttft_p50_s: f64::INFINITY,
            ttft_p90_s: f64::INFINITY,
            ttft_p95_s: f64::INFINITY,
            ttft_p99_s: f64::INFINITY,
            ttft_max_s: f64::INFINITY,
            ttft_slo_constrained_requests: 0,
            ttft_slo_missed_requests: 0,
            ttft_slo_miss_rate: f64::INFINITY,
            tpot_s: f64::INFINITY,
            tpot_p50_s: f64::INFINITY,
            tpot_p90_s: f64::INFINITY,
            tpot_p95_s: f64::INFINITY,
            tpot_p99_s: f64::INFINITY,
            tpot_max_s: f64::INFINITY,
            tpot_slo_constrained_requests: 0,
            tpot_slo_missed_requests: 0,
            tpot_slo_miss_rate: f64::INFINITY,
            itl_s: f64::INFINITY,
            itl_p50_s: f64::INFINITY,
            itl_p90_s: f64::INFINITY,
            itl_p95_s: f64::INFINITY,
            itl_p99_s: f64::INFINITY,
            itl_max_s: f64::INFINITY,
            itl_slo_constrained_requests: 0,
            itl_slo_missed_requests: 0,
            itl_slo_miss_rate: f64::INFINITY,
            decode_iterations: 0,
            decode_iteration_s: f64::INFINITY,
            decode_iteration_p50_s: f64::INFINITY,
            decode_iteration_p90_s: f64::INFINITY,
            decode_iteration_p95_s: f64::INFINITY,
            decode_iteration_p99_s: f64::INFINITY,
            decode_iteration_max_s: f64::INFINITY,
            throughput_tokens_per_s: 0.0,
            e2el_s: f64::INFINITY,
            e2el_p50_s: f64::INFINITY,
            e2el_p90_s: f64::INFINITY,
            e2el_p95_s: f64::INFINITY,
            e2el_p99_s: f64::INFINITY,
            e2el_max_s: f64::INFINITY,
            e2el_slo_constrained_requests: 0,
            e2el_slo_missed_requests: 0,
            e2el_slo_miss_rate: f64::INFINITY,
            deadline_miss_rate: f64::INFINITY,
            service_s: f64::INFINITY,
            prefill_s: f64::INFINITY,
            prefill_chunks: 0,
            prompt_tokens: 0,
            prefix_cache_hit_tokens: 0,
            effective_prefill_tokens: 0,
            prefix_cache_hit_rate: 0.0,
            kv_transfer_s: 0.0,
            kv_queue_s: f64::INFINITY,
            kv_worker_queue_s: f64::INFINITY,
            kv_resource_queue_s: f64::INFINITY,
            decode_queue_s: f64::INFINITY,
            prefill_worker_queue_s: f64::INFINITY,
            prefill_resource_queue_s: f64::INFINITY,
            decode_worker_queue_s: f64::INFINITY,
            decode_resource_queue_s: f64::INFINITY,
            decode_s: f64::INFINITY,
            queue_delay_s: f64::INFINITY,
            queue_delay_p90_s: f64::INFINITY,
            queue_delay_p95_s: f64::INFINITY,
            queue_delay_max_s: f64::INFINITY,
            peak_prefill_tokens: 0,
            peak_prefill_tokens_per_node: 0,
            peak_prefill_tokens_per_gpu: 0,
            peak_decode_sequences: 0,
            peak_resident_tokens: 0,
            peak_decode_sequences_per_node: 0,
            peak_resident_tokens_per_node: 0,
            peak_decode_sequences_per_gpu: 0,
            peak_resident_tokens_per_gpu: 0,
            peak_kv_blocks: 0,
            peak_allocated_kv_tokens: 0,
            peak_kv_fragmentation_tokens: 0,
            peak_kv_block_table_bytes: 0,
            peak_kv_blocks_per_node: 0,
            peak_allocated_kv_tokens_per_node: 0,
            peak_kv_fragmentation_tokens_per_node: 0,
            peak_kv_block_table_bytes_per_node: 0,
            peak_kv_blocks_per_gpu: 0,
            peak_allocated_kv_tokens_per_gpu: 0,
            peak_kv_fragmentation_tokens_per_gpu: 0,
            peak_kv_block_table_bytes_per_gpu: 0,
            decode_sequence_utilization: 0.0,
            resident_token_utilization: 0.0,
            kv_block_utilization: 0.0,
            decode_sequence_per_node_utilization: 0.0,
            resident_token_per_node_utilization: 0.0,
            kv_block_per_node_utilization: 0.0,
            decode_sequence_per_gpu_utilization: 0.0,
            resident_token_per_gpu_utilization: 0.0,
            kv_block_per_gpu_utilization: 0.0,
            scheduled_makespan_s: f64::INFINITY,
            scheduled_requests: 0,
            admitted_requests: 0,
            completed_requests: 0,
            rejected_requests: 0,
            timed_out_requests: 0,
            cancelled_requests: 0,
            deadline_constrained_requests: 0,
            deadline_missed_requests: 0,
            measured_requests: 0,
            measurement_start_s: f64::INFINITY,
            measurement_end_s: f64::INFINITY,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredServingConfig {
    pub candidate_id: String,
    pub objective: ServingObjective,
    pub deployment_mode: ServingDeploymentMode,
    pub slo_miss_penalty_weight: f64,
    pub slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
    pub slo_miss_penalty_components: ServingSloMissPenaltyComponents,
    pub traffic_class_slo_miss_penalties: Vec<ServingTrafficClassSloMissPenalty>,
    pub slo_miss_penalty_score: f64,
    pub service_backpressure_penalty_weight: f64,
    pub service_backpressure_penalty_score: f64,
    pub topology_risk_penalty_weight: f64,
    pub topology_risk_penalty_score: f64,
    pub max_memory_pressure_fraction: Option<f64>,
    pub max_unique_gpus: Option<u32>,
    pub min_throughput_tokens_per_s: Option<f64>,
    pub cost_model: ServingCostModel,
    pub metric_ceilings: ServingMetricCeilings,
    pub kv_route_constraints: ServingKvRouteConstraints,
    pub pool_label: Option<String>,
    pub prefill_nodes: Vec<NodeId>,
    pub decode_nodes: Vec<NodeId>,
    pub prefill_gpu_labels: Vec<String>,
    pub decode_gpu_labels: Vec<String>,
    pub pool_topology: ServingPoolTopologySummary,
    pub pool_search_summary: Option<ServingPoolSearchSummary>,
    pub route_coverage: ServingRouteCoverage,
    pub prefill_config: ParallelismConfig,
    pub decode_config: ParallelismConfig,
    pub prefill_score: ScoredParallelismConfig,
    pub decode_score: ScoredParallelismConfig,
    pub prefill_memory: ServingMemoryHeadroom,
    pub decode_memory: ServingMemoryHeadroom,
    pub calibration_summary: ServingCalibrationSummary,
    pub calibration_fits: Vec<CalibrationFitApplication>,
    pub phase_calibration: Vec<ServingPhaseCalibrationObservation>,
    pub calibration_gate_violations: Vec<CalibrationGateViolation>,
    pub approximation_summary: ServingApproximationSummary,
    pub approximations: Vec<SimulationApproximation>,
    pub approximation_policy_violations: Vec<ApproximationPolicyViolation>,
    pub metrics: ServingMetrics,
    pub measurement_window: ServingMeasurementWindowObservation,
    pub metric_breakdowns: Vec<ServingMetricBreakdown>,
    pub hardware_footprint: ServingHardwareFootprint,
    pub cost_estimate: ServingCostEstimate,
    pub memory_pressure: Vec<ServingMemoryPressureObservation>,
    pub kv_route_resource_summary: Vec<ServingKvRouteResourceSummary>,
    pub kv_route_topology_summary: ServingKvRouteTopologySummary,
    pub topology_bottlenecks: Vec<ServingTopologyBottleneckObservation>,
    pub bottleneck_summary: Vec<ServingBottleneckSummary>,
    pub pareto: ServingParetoFrontier,
    pub request_observations: Vec<ServingRequestObservation>,
    pub decode_iterations: Vec<ServingDecodeIterationObservation>,
    pub node_capacity: Vec<ServingNodeCapacityObservation>,
    pub gpu_capacity: Vec<ServingGpuCapacityObservation>,
    pub traffic_class_capacity: Vec<ServingTrafficClassCapacityObservation>,
    pub service_observations: Vec<ServingServiceObservation>,
    pub worker_observations: Vec<ServingWorkerObservation>,
    pub scheduled_operations: Vec<ScheduledOperation>,
    pub resource_utilization: Vec<ResourceUtilization>,
    pub phase_resource_utilization: Vec<ServingPhaseResourceUtilization>,
    pub feasible: bool,
    pub bottlenecks: Vec<String>,
    pub rejections: Vec<ServingRejection>,
    pub rejected_reason: Option<String>,
}

impl ScoredServingConfig {
    pub fn refresh_calibration_summary(&mut self) {
        self.calibration_summary = serving_calibration_summary(
            &self.phase_calibration,
            &self.calibration_fits,
            &self.calibration_gate_violations,
        );
    }

    pub fn refresh_approximation_summary(&mut self) {
        self.approximation_summary = serving_approximation_summary(
            &self.approximations,
            &self.approximation_policy_violations,
        );
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingApproximationSummary {
    pub status: String,
    pub approximation_count: u32,
    pub policy_violation_count: u32,
    pub category_counts: Vec<ServingApproximationCount>,
    pub top_codes: Vec<ServingApproximationCount>,
    pub calibration_count: u32,
    pub topology_count: u32,
    pub queueing_count: u32,
    pub runtime_count: u32,
    pub memory_count: u32,
    pub capacity_count: u32,
    pub routing_count: u32,
    pub admission_count: u32,
    pub uncalibrated_phase_count: u32,
    pub uncalibrated_queue_component_count: u32,
    pub extrapolated_fit_count: u32,
    pub coarse_topology: bool,
    pub approximate_queueing: bool,
    pub uncalibrated_runtime: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingApproximationCount {
    pub name: String,
    pub count: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingCalibrationSummary {
    pub status: String,
    pub active_phase_count: u32,
    pub calibrated_phase_count: u32,
    pub uncalibrated_phase_count: u32,
    pub coverage_fraction: f64,
    pub fit_count: u32,
    pub extrapolated_fit_count: u32,
    pub unbounded_fit_count: u32,
    pub fit_count_with_uncertainty: u32,
    pub min_confidence_score: Option<f64>,
    pub max_extrapolation_ratio: Option<f64>,
    pub relative_uncertainty_pct: Option<f64>,
    pub absolute_uncertainty_s: Option<f64>,
    pub gate_violation_count: u32,
    pub hard_gate_violation_count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingParetoFrontier {
    pub is_frontier: bool,
    pub rank: Option<u32>,
    pub dominated_by: Vec<String>,
    pub dimensions: Vec<ServingParetoDimension>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingParetoDimension {
    pub metric: String,
    pub direction: String,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingBottleneckSummary {
    pub source: String,
    pub phase: String,
    pub category: String,
    pub resource: String,
    pub code: String,
    pub severity: String,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub unit: Option<String>,
    pub message: String,
    pub remediation: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingHardwareFootprint {
    pub unique_node_count: u32,
    pub unique_gpu_count: u32,
    pub prefill_node_count: u32,
    pub prefill_gpu_count: u32,
    pub decode_node_count: u32,
    pub decode_gpu_count: u32,
    pub shared_node_count: u32,
    pub shared_gpu_count: u32,
    pub aggregate_hbm_gb: f64,
    pub prefill_hbm_gb: f64,
    pub decode_hbm_gb: f64,
    pub aggregate_hbm_bandwidth_gb_s: f64,
    pub prefill_hbm_bandwidth_gb_s: f64,
    pub decode_hbm_bandwidth_gb_s: f64,
    pub aggregate_peak_f16_tflops: f64,
    pub prefill_peak_f16_tflops: f64,
    pub decode_peak_f16_tflops: f64,
    pub aggregate_peak_f8_tflops: Option<f64>,
    pub prefill_peak_f8_tflops: Option<f64>,
    pub decode_peak_f8_tflops: Option<f64>,
    pub aggregate_gpu_types: Vec<ServingGpuTypeCount>,
    pub prefill_gpu_types: Vec<ServingGpuTypeCount>,
    pub decode_gpu_types: Vec<ServingGpuTypeCount>,
    pub aggregate_gpu_label_counts: Vec<ServingGpuLabelCount>,
    pub prefill_gpu_label_counts: Vec<ServingGpuLabelCount>,
    pub decode_gpu_label_counts: Vec<ServingGpuLabelCount>,
    pub aggregate_effective_peak_tflops: f64,
    pub prefill_effective_peak_tflops: f64,
    pub decode_effective_peak_tflops: f64,
    pub throughput_tokens_per_s_per_gpu: f64,
    pub throughput_tokens_per_s_per_effective_peak_tflop: f64,
    pub throughput_tokens_per_s_per_hbm_gb: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingGpuTypeCount {
    pub gpu: String,
    pub count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingGpuLabelCount {
    pub label: String,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingTrafficClassSloMissPenalty {
    pub name: String,
    pub group: String,
    pub key: String,
    pub weights: ServingSloMissPenaltyWeights,
    pub components: ServingSloMissPenaltyComponents,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingRouteCoverage {
    pub candidate_count: u32,
    pub routable_candidate_count: u32,
    pub unroutable_candidate_count: u32,
    pub fraction: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingKvRouteTopologySummary {
    pub route_resource_count: u32,
    pub inter_node_route_resource_count: u32,
    pub gpu_nic_route_resource_count: u32,
    pub intra_node_route_resource_count: u32,
    pub rail_ids: Vec<u32>,
    pub rail_count: u32,
    pub single_rail_dependency: bool,
    pub single_rail_id: Option<u32>,
    pub unrailed_inter_node_route_resource_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingTopologyBottleneckObservation {
    pub phase: String,
    pub category: String,
    pub resource: String,
    pub code: String,
    pub severity: String,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub unit: Option<String>,
    pub message: String,
    pub remediation: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingPhaseCalibrationObservation {
    pub phase: String,
    pub active: bool,
    pub calibrated: bool,
    pub fit_count: u32,
    pub applied_targets: Vec<String>,
    pub estimated_s: f64,
    pub status: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvRouteResourceSummary {
    pub resource_id: String,
    pub kind: String,
    pub label: String,
    pub request_count: u32,
    pub path_observations: u64,
    pub transfer_bytes: u64,
    pub estimated_transfer_s: f64,
    pub min_bandwidth_gbps: f64,
    pub max_latency_s: f64,
    pub rail_id: Option<u32>,
    pub from: Option<ServingKvTransferPathEndpointObservation>,
    pub to: Option<ServingKvTransferPathEndpointObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingPhaseResourceUtilization {
    pub phase: String,
    pub resource_kind: String,
    pub resource: String,
    pub busy_s: f64,
    pub utilization: f64,
    pub operation_count: usize,
    pub first_start_s: f64,
    pub last_finish_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRejection {
    pub phase: String,
    pub category: String,
    pub resource: String,
    pub code: String,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub unit: Option<String>,
    pub remediation: Option<String>,
    pub message: String,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingMemoryHeadroom {
    pub estimated_per_gpu_gb: f64,
    pub min_hbm_per_gpu_gb: f64,
    pub limiting_gpu: Option<GpuAddr>,
    pub headroom_gb: f64,
    pub headroom_fraction: f64,
    pub components: ServingMemoryComponents,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingMemoryComponentContribution {
    pub name: &'static str,
    pub gb: f64,
    pub fraction_of_total: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingMemoryPressureObservation {
    pub phase: String,
    pub estimate_kind: String,
    pub start_s: f64,
    pub finish_s: f64,
    pub duration_s: f64,
    pub active_requests: u32,
    pub active_tokens: u64,
    pub kv_blocks: u64,
    pub estimated_per_gpu_gb: f64,
    pub min_hbm_per_gpu_gb: f64,
    pub capacity_used_fraction: f64,
    pub headroom_gb: f64,
    pub limiting_gpu: Option<GpuAddr>,
    pub dominant_component: Option<ServingMemoryComponentContribution>,
    pub components: ServingMemoryComponents,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ServingMemoryComponents {
    pub weights_gb: f64,
    pub kv_cache_gb: f64,
    pub block_table_gb: f64,
    pub activations_gb: f64,
    pub temporary_gb: f64,
    pub communication_gb: f64,
    pub runtime_reserve_gb: f64,
    pub fragmentation_gb: f64,
    pub total_gb: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServingRequestStatus {
    Pending,
    Completed,
    RejectedAdmission,
    TimedOut,
    Cancelled,
}

impl ServingRequestStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingRequestStatus::Pending => "pending",
            ServingRequestStatus::Completed => "completed",
            ServingRequestStatus::RejectedAdmission => "rejected_admission",
            ServingRequestStatus::TimedOut => "timed_out",
            ServingRequestStatus::Cancelled => "cancelled",
        }
    }

    pub(super) fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }

    pub(super) fn is_admitted(self) -> bool {
        !matches!(self, Self::RejectedAdmission)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServingRequestEventKind {
    Arrived,
    QueuedForPrefill,
    PrefillStarted,
    PrefillFinished,
    QueuedForKvTransfer,
    KvTransferStarted,
    KvTransferFinished,
    KvBlocksAllocated,
    QueuedForDecode,
    DecodeIterationStarted,
    DecodeIterationFinished,
    KvBlocksReleased,
    Completed,
    RejectedAdmission,
    TimedOut,
    Cancelled,
}

impl ServingRequestEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingRequestEventKind::Arrived => "arrived",
            ServingRequestEventKind::QueuedForPrefill => "queued_for_prefill",
            ServingRequestEventKind::PrefillStarted => "prefill_started",
            ServingRequestEventKind::PrefillFinished => "prefill_finished",
            ServingRequestEventKind::QueuedForKvTransfer => "queued_for_kv_transfer",
            ServingRequestEventKind::KvTransferStarted => "kv_transfer_started",
            ServingRequestEventKind::KvTransferFinished => "kv_transfer_finished",
            ServingRequestEventKind::KvBlocksAllocated => "kv_blocks_allocated",
            ServingRequestEventKind::QueuedForDecode => "queued_for_decode",
            ServingRequestEventKind::DecodeIterationStarted => "decode_iteration_started",
            ServingRequestEventKind::DecodeIterationFinished => "decode_iteration_finished",
            ServingRequestEventKind::KvBlocksReleased => "kv_blocks_released",
            ServingRequestEventKind::Completed => "completed",
            ServingRequestEventKind::RejectedAdmission => "rejected_admission",
            ServingRequestEventKind::TimedOut => "timed_out",
            ServingRequestEventKind::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestEventObservation {
    pub kind: ServingRequestEventKind,
    pub phase: String,
    pub at_s: f64,
    pub decode_iteration: Option<u32>,
    pub message: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServingRequestPhaseCategory {
    Queue,
    Service,
    Transfer,
}

impl ServingRequestPhaseCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            ServingRequestPhaseCategory::Queue => "queue",
            ServingRequestPhaseCategory::Service => "service",
            ServingRequestPhaseCategory::Transfer => "transfer",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestPhaseBreakdownObservation {
    pub phase: String,
    pub category: ServingRequestPhaseCategory,
    pub start_s: f64,
    pub finish_s: f64,
    pub duration_s: f64,
    pub contributes_to_ttft: bool,
    pub contributes_to_e2el: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvBlockOwnershipObservation {
    pub allocation_id: String,
    pub owner: GpuAddr,
    pub owner_worker_slots: Vec<u32>,
    pub worker_slot_ownership: Vec<ServingKvWorkerSlotOwnershipObservation>,
    pub decode_operation_ids: Vec<usize>,
    pub allocated_at_s: f64,
    pub released_at_s: f64,
    pub duration_s: f64,
    pub block_start: u64,
    pub block_end: u64,
    pub decode_sequences: u32,
    pub resident_tokens: u64,
    pub kv_blocks: u64,
    pub allocated_kv_tokens: u64,
    pub kv_fragmentation_tokens: u64,
    pub block_table_entries: u64,
    pub block_table_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvWorkerSlotOwnershipObservation {
    pub allocation_id: String,
    pub slot: u32,
    pub block_start: u64,
    pub block_end: u64,
    pub decode_sequences: u32,
    pub resident_tokens: u64,
    pub kv_blocks: u64,
    pub allocated_kv_tokens: u64,
    pub kv_fragmentation_tokens: u64,
    pub block_table_entries: u64,
    pub block_table_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestWorkerSummaryObservation {
    pub role: String,
    pub phase: String,
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub worker_slots: Vec<u32>,
    pub operation_ids: Vec<usize>,
    pub assignment_count: u32,
    pub start_s: f64,
    pub finish_s: f64,
    pub resident_tokens: u64,
    pub kv_blocks: u64,
}

impl ServingMemoryHeadroom {
    pub(super) fn unavailable() -> Self {
        Self {
            estimated_per_gpu_gb: f64::INFINITY,
            min_hbm_per_gpu_gb: f64::INFINITY,
            limiting_gpu: None,
            headroom_gb: f64::INFINITY,
            headroom_fraction: f64::INFINITY,
            components: ServingMemoryComponents::unavailable(),
        }
    }

    pub fn capacity_used_fraction(&self) -> f64 {
        if self.estimated_per_gpu_gb.is_finite()
            && self.min_hbm_per_gpu_gb.is_finite()
            && self.min_hbm_per_gpu_gb > 0.0
        {
            self.estimated_per_gpu_gb / self.min_hbm_per_gpu_gb
        } else {
            f64::INFINITY
        }
    }

    pub fn dominant_component(&self) -> Option<ServingMemoryComponentContribution> {
        self.components.dominant_component()
    }
}

impl ServingMemoryComponents {
    pub(super) fn unavailable() -> Self {
        Self {
            weights_gb: f64::INFINITY,
            kv_cache_gb: f64::INFINITY,
            block_table_gb: f64::INFINITY,
            activations_gb: f64::INFINITY,
            temporary_gb: f64::INFINITY,
            communication_gb: f64::INFINITY,
            runtime_reserve_gb: f64::INFINITY,
            fragmentation_gb: f64::INFINITY,
            total_gb: f64::INFINITY,
        }
    }

    pub fn component_contributions(&self) -> [ServingMemoryComponentContribution; 8] {
        let total_gb = self.total_gb;
        [
            self.component_contribution("weights", self.weights_gb, total_gb),
            self.component_contribution("kv_cache", self.kv_cache_gb, total_gb),
            self.component_contribution("block_table", self.block_table_gb, total_gb),
            self.component_contribution("activations", self.activations_gb, total_gb),
            self.component_contribution("temporary", self.temporary_gb, total_gb),
            self.component_contribution("communication", self.communication_gb, total_gb),
            self.component_contribution("runtime_reserve", self.runtime_reserve_gb, total_gb),
            self.component_contribution("fragmentation", self.fragmentation_gb, total_gb),
        ]
    }

    pub fn dominant_component(&self) -> Option<ServingMemoryComponentContribution> {
        if !self.total_gb.is_finite() || self.total_gb <= 0.0 {
            return None;
        }

        self.component_contributions()
            .into_iter()
            .filter(|component| component.gb.is_finite() && component.gb >= 0.0)
            .max_by(|left, right| left.gb.total_cmp(&right.gb))
    }

    pub(super) fn component_contribution(
        &self,
        name: &'static str,
        gb: f64,
        total_gb: f64,
    ) -> ServingMemoryComponentContribution {
        let fraction_of_total = if gb.is_finite() && total_gb.is_finite() && total_gb > 0.0 {
            gb / total_gb
        } else {
            f64::INFINITY
        };

        ServingMemoryComponentContribution {
            name,
            gb,
            fraction_of_total,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingNodeCapacityObservation {
    pub node_id: NodeId,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingTrafficClassCapacityObservation {
    pub name: String,
    pub group: String,
    pub key: String,
    pub max_prefill_tokens: Option<u64>,
    pub max_decode_sequences: Option<u32>,
    pub max_resident_tokens: Option<u64>,
    pub max_kv_blocks: Option<u64>,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub prefill_token_utilization: f64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingDecodeIterationObservation {
    pub iteration_idx: u32,
    pub decode_nodes: Vec<NodeId>,
    pub request_indices: Vec<u32>,
    pub operation_ids: Vec<usize>,
    pub start_s: f64,
    pub finish_s: f64,
    pub latency_s: f64,
    pub batch_tokens: u32,
    pub first_token_batch_tokens: u32,
    pub tail_token_batch_tokens: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingGpuCapacityObservation {
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingWorkerObservation {
    pub phase: String,
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub configured_worker_slots: u32,
    pub peak_active_worker_slots: u32,
    pub worker_slot_utilization: f64,
    pub request_count: u32,
    pub completed_requests: u32,
    pub failed_requests: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub queue_s: f64,
    pub queue_p95_s: f64,
    pub queue_max_s: f64,
    pub worker_queue_s: f64,
    pub worker_queue_p95_s: f64,
    pub worker_queue_max_s: f64,
    pub resource_queue_s: f64,
    pub resource_queue_p95_s: f64,
    pub resource_queue_max_s: f64,
    pub service_s: f64,
    pub first_start_s: f64,
    pub last_finish_s: f64,
    pub peak_prefill_tokens: u64,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub kv_cache_owner_slots: Vec<ServingWorkerKvSlotObservation>,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingWorkerKvSlotObservation {
    pub slot: u32,
    pub peak_decode_sequences: u32,
    pub peak_resident_tokens: u64,
    pub peak_kv_blocks: u64,
    pub peak_allocated_kv_tokens: u64,
    pub peak_kv_fragmentation_tokens: u64,
    pub peak_kv_block_table_bytes: u64,
    pub decode_sequence_utilization: f64,
    pub resident_token_utilization: f64,
    pub kv_block_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingWorkerAssignmentObservation {
    pub phase: String,
    pub node_id: NodeId,
    pub local_gpu_id: u32,
    pub slot: u32,
    pub start_s: f64,
    pub finish_s: f64,
    pub operation_ids: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingServiceObservation {
    pub phase: String,
    pub health: ServingServiceHealth,
    pub accepts_requests: bool,
    pub worker_scale: f64,
    pub configured_worker_slots_per_gpu: u32,
    pub effective_worker_slots_per_gpu: u32,
    pub node_count: u32,
    pub gpu_count: u32,
    pub request_count: u32,
    pub admitted_requests: u32,
    pub completed_requests: u32,
    pub failed_requests: u32,
    pub rejected_requests: u32,
    pub timed_out_requests: u32,
    pub cancelled_requests: u32,
    pub queue_cap_s: Option<f64>,
    pub queue_cap_request_count: u32,
    pub queue_cap_hit_count: u32,
    pub decode_iteration_queue_cap_s: Option<f64>,
    pub decode_iteration_queue_cap_request_count: u32,
    pub decode_iteration_queue_cap_hit_count: u32,
    pub backpressure_rejections: u32,
    pub timeout_rejections: u32,
    pub backpressure_state: String,
    pub worker_slot_utilization: f64,
    pub queue_s: f64,
    pub queue_p95_s: f64,
    pub queue_max_s: f64,
    pub worker_queue_s: f64,
    pub resource_queue_s: f64,
    pub service_s: f64,
}

pub struct ServingSolver;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ServingSolverOptions<'a> {
    pub calibration: SimulationCalibration,
    pub calibration_profile: Option<&'a CalibrationProfileMetadata>,
    pub model_id: Option<&'a str>,
    pub serving_stack: Option<&'a str>,
    pub serving_runtime_features: Option<&'a [String]>,
    pub max_prefill_candidates: Option<usize>,
    pub max_decode_candidates: Option<usize>,
    pub max_serving_pairs: Option<usize>,
    pub search_deadline: Option<Instant>,
    pub explicit_prefill_placement: Option<&'a RankPlacement>,
    pub explicit_decode_placement: Option<&'a RankPlacement>,
}
