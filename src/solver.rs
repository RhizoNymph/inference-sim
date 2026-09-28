use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    time::Instant,
};

mod calibration_fits;
mod network_cost;
mod operations;
mod placement;
use calibration_fits::{calibration_feature_values, evaluate_fit, evaluate_value_fit, fit_matches};
pub use operations::schedule_operations;

use crate::{
    calibration::SimulationCalibration,
    config::{
        ApproximationPolicyViolation, CalibrationFitFeatureRange, CalibrationFittedModel,
        CalibrationGateViolation, CalibrationProfileMetadata,
    },
    scheduler::{ResourceScheduler, ResourceUtilization, ScheduledOperation, resource_utilization},
    topology_graph::{
        GraphResource, RoutedPath, RoutedResource, RoutedResourceKind, TopologyGraph,
    },
    types::{
        collective::{
            CollectiveAlgorithm, CollectiveCall, CollectiveCost, CollectiveKind, ReductionOp,
        },
        common::{Bandwidth, Bytes, GpuAddr, NodeId, RankId, UnorderedPair},
        configs::{ParallelGroups, ParallelismConfig, RankPlacement},
        fabric::{inter_node::InterNodeTopology, intra_node::IntraNodeTopology},
        topology::Cluster,
    },
    workload::{InferencePhase, InferenceRequest, ModelSpec},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchSpace {
    pub tensor_ranks: Vec<u32>,
    pub pipeline_ranks: Vec<u32>,
    pub expert_ranks: Vec<u32>,
    pub data_ranks: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredParallelismConfig {
    pub config: ParallelismConfig,
    pub placement: RankPlacement,
    pub placement_evidence: Vec<PlacementEvidence>,
    pub groups: ParallelGroups,
    pub feasible: bool,
    pub estimated_latency_s: f64,
    pub estimated_memory_per_gpu: Bytes,
    pub calibration_fits: Vec<CalibrationFitApplication>,
    pub calibration_gate_violations: Vec<CalibrationGateViolation>,
    pub approximations: Vec<SimulationApproximation>,
    pub approximation_policy_violations: Vec<ApproximationPolicyViolation>,
    pub bottlenecks: Vec<String>,
    pub rejected_reason: Option<String>,
    pub operations: Vec<SimOperation>,
    pub scheduled_operations: Vec<ScheduledOperation>,
    pub resource_utilization: Vec<ResourceUtilization>,
    pub operation_makespan_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacementEvidence {
    pub decision: String,
    pub scope: String,
    pub resource: String,
    pub code: String,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub unit: Option<String>,
    pub message: String,
    pub remediation: Option<String>,
}

impl PlacementEvidence {
    fn new(
        decision: impl Into<String>,
        scope: impl Into<String>,
        resource: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            decision: decision.into(),
            scope: scope.into(),
            resource: resource.into(),
            code: code.into(),
            observed: None,
            limit: None,
            unit: None,
            message: message.into(),
            remediation: None,
        }
    }

    fn with_details(
        mut self,
        observed: Option<f64>,
        limit: Option<f64>,
        unit: Option<&str>,
        remediation: Option<&str>,
    ) -> Self {
        self.observed = observed;
        self.limit = limit;
        self.unit = unit.map(str::to_string);
        self.remediation = remediation.map(str::to_string);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimulationApproximation {
    pub phase: String,
    pub category: String,
    pub scope: String,
    pub code: String,
    pub message: String,
    pub remediation: Option<String>,
}

impl SimulationApproximation {
    pub fn new(
        phase: impl Into<String>,
        category: impl Into<String>,
        scope: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
        remediation: Option<String>,
    ) -> Self {
        Self {
            phase: phase.into(),
            category: category.into(),
            scope: scope.into(),
            code: code.into(),
            message: message.into(),
            remediation,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SimOperationKind {
    Compute,
    Collective,
    Transfer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimOperation {
    pub name: String,
    pub kind: SimOperationKind,
    pub duration_s: f64,
    pub resources: Vec<String>,
    pub dependencies: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationFitApplication {
    pub phase: String,
    pub target: String,
    pub fit_name: Option<String>,
    pub model: String,
    pub unit: Option<String>,
    pub intercept: f64,
    pub raw_prediction: f64,
    pub prediction_kind: String,
    pub predicted_value: f64,
    pub prediction_unit: Option<String>,
    pub predicted_s: f64,
    pub baseline_value: Option<f64>,
    pub baseline_s: Option<f64>,
    pub applicability_status: String,
    pub confidence_score: f64,
    pub max_extrapolation_ratio: f64,
    pub relative_uncertainty_pct: Option<f64>,
    pub absolute_uncertainty_value: Option<f64>,
    pub absolute_uncertainty_s: Option<f64>,
    pub uncertainty_source: Option<String>,
    pub validation_rmse: Option<f64>,
    pub validation_rmse_pct: Option<f64>,
    pub validation_mean_abs_pct_error: Option<f64>,
    pub validation_max_abs_pct_error: Option<f64>,
    pub confidence_interval: Option<f64>,
    pub confidence_interval_pct: Option<f64>,
    pub confidence_level: Option<f64>,
    pub sample_count: Option<u32>,
    pub validation_sample_count: Option<u32>,
    pub source: Option<String>,
    pub features: Vec<CalibrationFitFeatureValue>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationFitFeatureValue {
    pub name: String,
    pub value: f64,
    pub coefficient: f64,
    pub range_min: Option<f64>,
    pub range_max: Option<f64>,
    pub status: String,
    pub extrapolation_ratio: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct FitEvaluation {
    seconds: f64,
    application: CalibrationFitApplication,
}

pub struct Solver;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SolverOptions<'a> {
    pub calibration: SimulationCalibration,
    pub calibration_profile: Option<&'a CalibrationProfileMetadata>,
    pub max_candidates: Option<usize>,
    pub search_deadline: Option<Instant>,
    pub explicit_placement: Option<&'a RankPlacement>,
}

impl Solver {
    pub fn rank_configs(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        search_space: &SearchSpace,
    ) -> Vec<ScoredParallelismConfig> {
        Self::rank_configs_with_options(
            cluster,
            model,
            request,
            search_space,
            SolverOptions::default(),
        )
    }

    pub fn rank_configs_with_options(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        search_space: &SearchSpace,
        options: SolverOptions,
    ) -> Vec<ScoredParallelismConfig> {
        let mut results = Vec::new();
        let configs = ordered_parallelism_candidates(search_space);

        for config in configs {
            if options
                .max_candidates
                .is_some_and(|max_candidates| results.len() >= max_candidates)
                || search_deadline_expired(options.search_deadline)
            {
                break;
            }
            results.push(Self::score_config_with_options(
                cluster, model, request, config, options,
            ));
        }

        results.sort_by(|a, b| {
            b.feasible
                .cmp(&a.feasible)
                .then_with(|| a.estimated_latency_s.total_cmp(&b.estimated_latency_s))
                .then_with(|| {
                    a.estimated_memory_per_gpu
                        .as_bytes()
                        .cmp(&b.estimated_memory_per_gpu.as_bytes())
                })
                .then_with(|| a.config.total_ranks().cmp(&b.config.total_ranks()))
        });

        results
    }

    pub fn score_config(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
    ) -> ScoredParallelismConfig {
        Self::score_config_with_options(cluster, model, request, config, SolverOptions::default())
    }

    pub fn score_config_with_options(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        options: SolverOptions,
    ) -> ScoredParallelismConfig {
        let calibration = options.calibration.sanitized();
        let empty = ScoredParallelismConfig {
            config,
            placement: RankPlacement {
                rank_to_gpu: Vec::new(),
            },
            placement_evidence: Vec::new(),
            groups: ParallelGroups {
                tensor_groups: Vec::new(),
                pipeline_stages: Vec::new(),
                expert_groups: Vec::new(),
                data_groups: Vec::new(),
            },
            feasible: false,
            estimated_latency_s: f64::INFINITY,
            estimated_memory_per_gpu: Bytes::from_bytes(0),
            calibration_fits: Vec::new(),
            calibration_gate_violations: Vec::new(),
            approximations: Vec::new(),
            approximation_policy_violations: Vec::new(),
            bottlenecks: Vec::new(),
            rejected_reason: None,
            operations: Vec::new(),
            scheduled_operations: Vec::new(),
            resource_utilization: Vec::new(),
            operation_makespan_s: 0.0,
        };

        if let Err(reason) = config.validate_dimensions() {
            return Self::reject(empty, reason);
        }

        if config.expert_ranks > 1 && model.experts.is_none() {
            return Self::reject(
                empty,
                "expert parallelism requires a model expert spec".to_string(),
            );
        }

        let total_ranks = config.total_ranks();
        if total_ranks > cluster.available_gpus() {
            let available_gpus = cluster.available_gpus();
            let reason = format!(
                "config requires {total_ranks} ranks but cluster only has {available_gpus} available GPUs"
            );
            return Self::reject(
                ScoredParallelismConfig {
                    placement_evidence: vec![Self::placement_capacity_evidence(
                        total_ranks,
                        available_gpus,
                        &reason,
                    )],
                    ..empty
                },
                reason,
            );
        }

        let estimated_memory_per_gpu = Self::estimate_memory_per_gpu(model, request, config);
        let (placement, placement_strategy_code) = match options.explicit_placement {
            Some(explicit_placement) => match Self::place_explicit_ranks(
                cluster,
                model,
                config,
                explicit_placement,
                estimated_memory_per_gpu,
            ) {
                Ok(placement) => (placement, "explicit_rank_placement"),
                Err(evidence) => {
                    let reason = evidence.message.clone();
                    return Self::reject(
                        ScoredParallelismConfig {
                            estimated_memory_per_gpu,
                            placement_evidence: vec![*evidence],
                            ..empty
                        },
                        reason,
                    );
                }
            },
            None => match Self::place_ranks(cluster, model, config, estimated_memory_per_gpu) {
                Ok(placement) => {
                    let strategy_code = if data_replicas_are_node_packed(config, &placement) {
                        "data_replica_node_packed_placement"
                    } else {
                        "global_capability_ordered_placement"
                    };
                    (placement, strategy_code)
                }
                Err(reason) => {
                    return Self::reject(
                        ScoredParallelismConfig {
                            estimated_memory_per_gpu,
                            placement_evidence: vec![Self::placement_hbm_evidence(
                                cluster,
                                model,
                                config,
                                estimated_memory_per_gpu,
                                &reason,
                            )],
                            ..empty
                        },
                        reason,
                    );
                }
            },
        };
        let groups = Self::build_groups(config);

        if let Some(reason) =
            Self::memory_rejection_reason(cluster, &placement, estimated_memory_per_gpu)
        {
            return Self::reject(
                ScoredParallelismConfig {
                    placement,
                    placement_evidence: vec![Self::placement_memory_rejection_evidence(
                        estimated_memory_per_gpu,
                        &reason,
                    )],
                    groups,
                    estimated_memory_per_gpu,
                    ..empty
                },
                reason,
            );
        }

        let (compute_latency_s, calibration_fits) = Self::estimate_compute_latency_s(
            cluster,
            model,
            request,
            config,
            &placement,
            calibration,
            options.calibration_profile,
        );
        let operations = Self::build_operation_trace(
            cluster,
            model,
            request,
            config,
            &placement,
            &groups,
            compute_latency_s,
            calibration,
        );
        let scheduled_operations = schedule_operations(&operations);
        let estimated_latency_s = scheduled_operations
            .iter()
            .map(|operation| operation.finish_s)
            .fold(0.0, f64::max)
            + calibration.scheduler_overhead_us / 1e6;
        let resource_utilization = resource_utilization(&scheduled_operations, estimated_latency_s);
        let mut bottlenecks = Vec::new();
        let mut seen_bottlenecks = HashSet::new();

        for operation in &operations {
            for resource in &operation.resources {
                if seen_bottlenecks.insert(resource.clone()) {
                    bottlenecks.push(resource.clone());
                }
            }
        }
        if bottlenecks.is_empty() {
            bottlenecks.push("GPU compute".to_string());
        }
        let approximations = Self::parallelism_approximations(
            cluster,
            request,
            config,
            &placement,
            &operations,
            &calibration_fits,
        );
        let placement_evidence = Self::placement_evidence(
            cluster,
            model,
            config,
            &placement,
            estimated_memory_per_gpu,
            placement_strategy_code,
        );

        ScoredParallelismConfig {
            config,
            placement,
            placement_evidence,
            groups,
            feasible: true,
            estimated_latency_s,
            estimated_memory_per_gpu,
            calibration_fits,
            calibration_gate_violations: Vec::new(),
            approximations,
            approximation_policy_violations: Vec::new(),
            bottlenecks,
            rejected_reason: None,
            operations,
            scheduled_operations,
            resource_utilization,
            operation_makespan_s: estimated_latency_s,
        }
    }

    fn build_groups(config: ParallelismConfig) -> ParallelGroups {
        let tp = config.tensor_ranks;
        let pp = config.pipeline_ranks;
        let ep = config.expert_ranks;
        let dp = config.data_ranks;

        let mut tensor_groups = Vec::new();
        for dp_idx in 0..dp {
            for pp_idx in 0..pp {
                for ep_idx in 0..ep {
                    let group = (0..tp)
                        .map(|tp_idx| rank_id(config, dp_idx, pp_idx, ep_idx, tp_idx))
                        .collect();
                    tensor_groups.push(group);
                }
            }
        }

        let mut pipeline_stages = Vec::new();
        for pp_idx in 0..pp {
            let mut stage = Vec::new();
            for dp_idx in 0..dp {
                for ep_idx in 0..ep {
                    for tp_idx in 0..tp {
                        stage.push(rank_id(config, dp_idx, pp_idx, ep_idx, tp_idx));
                    }
                }
            }
            pipeline_stages.push(stage);
        }

        let mut expert_groups = Vec::new();
        if ep > 1 {
            for dp_idx in 0..dp {
                for pp_idx in 0..pp {
                    for tp_idx in 0..tp {
                        let group = (0..ep)
                            .map(|ep_idx| rank_id(config, dp_idx, pp_idx, ep_idx, tp_idx))
                            .collect();
                        expert_groups.push(group);
                    }
                }
            }
        }

        let mut data_groups = Vec::new();
        if dp > 1 {
            for pp_idx in 0..pp {
                for ep_idx in 0..ep {
                    for tp_idx in 0..tp {
                        let group = (0..dp)
                            .map(|dp_idx| rank_id(config, dp_idx, pp_idx, ep_idx, tp_idx))
                            .collect();
                        data_groups.push(group);
                    }
                }
            }
        }

        ParallelGroups {
            tensor_groups,
            pipeline_stages,
            expert_groups,
            data_groups,
        }
    }

    fn estimate_memory_per_gpu(
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
    ) -> Bytes {
        let shard_factor =
            (config.tensor_ranks * config.pipeline_ranks * config.expert_ranks).max(1) as u64;
        let parameter_bytes = div_ceil(model.parameters.as_bytes(), shard_factor);
        let kv_bytes = Self::kv_cache_bytes(model, request, config);

        Bytes::from_bytes(parameter_bytes + kv_bytes)
    }

    fn kv_cache_bytes(
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
    ) -> u64 {
        let head_dim = (model.hidden_size / model.attention_heads.max(1)) as u64;
        let kv_heads = model.kv_heads as u64;
        let tokens = request.max_sequence_tokens as u64 * request.batch_size as u64;
        let bytes_per_element = model.kv_dtype().bytes_per_element();
        let per_layer = tokens * kv_heads * head_dim * 2 * bytes_per_element;
        let total = per_layer * model.layers as u64;

        div_ceil(total, config.tensor_ranks.max(1) as u64)
    }

    fn memory_rejection_reason(
        cluster: &Cluster,
        placement: &RankPlacement,
        memory_per_gpu: Bytes,
    ) -> Option<String> {
        let min_hbm = placement
            .rank_to_gpu
            .iter()
            .filter_map(|addr| cluster.gpu_profile(*addr).map(|profile| profile.hbm_size))
            .min_by_key(|bytes| bytes.as_bytes())?;

        if memory_per_gpu > min_hbm {
            Some(format!(
                "estimated {:.2} GB per GPU exceeds {:.2} GB HBM",
                memory_per_gpu.as_gigabytes(),
                min_hbm.as_gigabytes()
            ))
        } else {
            None
        }
    }

    fn estimate_compute_latency_s(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        calibration: SimulationCalibration,
        calibration_profile: Option<&CalibrationProfileMetadata>,
    ) -> (f64, Vec<CalibrationFitApplication>) {
        let parameter_count = model.parameter_count();
        let shard_factor =
            (config.tensor_ranks * config.pipeline_ranks * config.expert_ranks).max(1) as f64;
        let peak_flops = Self::effective_peak_flops(cluster, model, placement, calibration);

        match request.phase {
            InferencePhase::Prefill => {
                let baseline = Self::prefill_baseline_s(
                    model,
                    request,
                    config,
                    shard_factor,
                    peak_flops,
                    calibration,
                );
                if let Some(evaluation) = Self::fitted_phase_latency(
                    calibration_profile,
                    "prefill",
                    &["prefill_ms", "prefill_latency_ms", "prefill_s"],
                    model,
                    request,
                    config,
                    None,
                    Some(baseline),
                ) {
                    (evaluation.seconds, vec![evaluation.application])
                } else {
                    (baseline, Vec::new())
                }
            }
            InferencePhase::Decode => {
                let (decode_s, application) = Self::decode_compute_latency_s(
                    cluster,
                    model,
                    request,
                    config,
                    placement,
                    parameter_count,
                    shard_factor,
                    peak_flops,
                    calibration,
                    calibration_profile,
                );
                (decode_s, application.into_iter().collect())
            }
            InferencePhase::EndToEnd => {
                let prefill_baseline = Self::prefill_baseline_s(
                    model,
                    request,
                    config,
                    shard_factor,
                    peak_flops,
                    calibration,
                );
                let mut applications = Vec::new();
                let prefill_s = if let Some(evaluation) = Self::fitted_phase_latency(
                    calibration_profile,
                    "prefill",
                    &["prefill_ms", "prefill_latency_ms", "prefill_s"],
                    model,
                    request,
                    config,
                    None,
                    Some(prefill_baseline),
                ) {
                    applications.push(evaluation.application);
                    evaluation.seconds
                } else {
                    prefill_baseline
                };
                let (decode_s, decode_application) = Self::decode_compute_latency_s(
                    cluster,
                    model,
                    request,
                    config,
                    placement,
                    parameter_count,
                    shard_factor,
                    peak_flops,
                    calibration,
                    calibration_profile,
                );
                applications.extend(decode_application);
                (prefill_s + decode_s, applications)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_compute_latency_s(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        parameter_count: f64,
        shard_factor: f64,
        peak_flops: f64,
        calibration: SimulationCalibration,
        calibration_profile: Option<&CalibrationProfileMetadata>,
    ) -> (f64, Option<CalibrationFitApplication>) {
        let decode_tokens = request.decode_tokens as f64;
        let flop_tokens = decode_tokens * request.batch_size as f64;
        let attention_shard_factor = Self::attention_shard_factor(config);
        let flop_latency_s =
            Self::flop_latency_s(parameter_count, flop_tokens, shard_factor, peak_flops)
                + Self::decode_attention_flops(model, request)
                    / attention_shard_factor
                    / peak_flops;
        let parameter_bytes_per_rank = model.parameters.as_bytes() as f64 / shard_factor;
        let hbm_bandwidth = Self::effective_hbm_bandwidth(cluster, placement, calibration);
        let kv_read_latency_s =
            Self::decode_kv_read_bytes(model, request) / attention_shard_factor / hbm_bandwidth;
        let memory_latency_s =
            parameter_bytes_per_rank * decode_tokens / hbm_bandwidth + kv_read_latency_s;

        let baseline = flop_latency_s.max(memory_latency_s) * calibration.decode_compute_scale;
        if let Some(evaluation) = Self::fitted_phase_latency(
            calibration_profile,
            "decode",
            &["decode_ms", "decode_latency_ms", "decode_s"],
            model,
            request,
            config,
            None,
            Some(baseline),
        ) {
            (evaluation.seconds, Some(evaluation.application))
        } else {
            (baseline, None)
        }
    }

    fn flop_latency_s(
        parameter_count: f64,
        active_tokens: f64,
        shard_factor: f64,
        peak_flops: f64,
    ) -> f64 {
        let total_flops = 2.0 * parameter_count * active_tokens;
        let flops_per_rank = total_flops / shard_factor;
        flops_per_rank / peak_flops
    }

    // Attention score/value work shards across tensor ranks (heads) and
    // pipeline ranks (layers) but not expert ranks, which only partition MLP
    // expert weights.
    fn attention_shard_factor(config: ParallelismConfig) -> f64 {
        (config.tensor_ranks * config.pipeline_ranks).max(1) as f64
    }

    // Causal QK^T and AV bilinear FLOPs for prefilling the prompt. The
    // attention projection GEMMs are already covered by the parameter-count
    // FLOP term.
    fn prefill_attention_flops(model: &ModelSpec, request: &InferenceRequest) -> f64 {
        let prompt_tokens = request.prompt_tokens as f64;
        2.0 * model.layers as f64
            * model.hidden_size as f64
            * prompt_tokens
            * prompt_tokens
            * request.batch_size as f64
    }

    // Decode steps see contexts prompt+1 ..= prompt+decode; per-request cost
    // terms use the mean context so total decode work stays closed-form.
    fn average_decode_context_tokens(request: &InferenceRequest) -> f64 {
        request.prompt_tokens as f64 + (request.decode_tokens as f64 + 1.0) / 2.0
    }

    fn decode_attention_flops(model: &ModelSpec, request: &InferenceRequest) -> f64 {
        4.0 * model.layers as f64
            * model.hidden_size as f64
            * Self::average_decode_context_tokens(request)
            * request.decode_tokens as f64
            * request.batch_size as f64
    }

    // Bytes of KV cache streamed from HBM across all decode iterations: every
    // step re-reads each sequence's K and V for its current context length.
    fn decode_kv_read_bytes(model: &ModelSpec, request: &InferenceRequest) -> f64 {
        let head_dim = (model.hidden_size / model.attention_heads.max(1)) as f64;
        let bytes_per_element = model.kv_dtype().bytes_per_element() as f64;
        2.0 * model.layers as f64
            * model.kv_heads as f64
            * head_dim
            * bytes_per_element
            * Self::average_decode_context_tokens(request)
            * request.decode_tokens as f64
            * request.batch_size as f64
    }

    fn prefill_baseline_s(
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        shard_factor: f64,
        peak_flops: f64,
        calibration: SimulationCalibration,
    ) -> f64 {
        let active_tokens = request.prompt_tokens as f64 * request.batch_size as f64;
        let dense_s = Self::flop_latency_s(
            model.parameter_count(),
            active_tokens,
            shard_factor,
            peak_flops,
        );
        let attention_s = Self::prefill_attention_flops(model, request)
            / Self::attention_shard_factor(config)
            / peak_flops;
        (dense_s + attention_s) * calibration.prefill_compute_scale
    }

    fn effective_peak_flops(
        cluster: &Cluster,
        model: &ModelSpec,
        placement: &RankPlacement,
        calibration: SimulationCalibration,
    ) -> f64 {
        let peak_tflops = placement
            .rank_to_gpu
            .iter()
            .filter_map(|addr| cluster.gpu_profile(*addr))
            .map(|profile| match model.dtype {
                crate::workload::DType::Fp8 | crate::workload::DType::Int8 => {
                    profile.peak_f8_flops.unwrap_or(profile.peak_f16_flops)
                }
                crate::workload::DType::Fp16 | crate::workload::DType::Bf16 => {
                    profile.peak_f16_flops
                }
            })
            .fold(f64::INFINITY, f64::min);

        peak_tflops.max(1.0) * 1e12 * calibration.compute_efficiency
    }

    fn effective_hbm_bandwidth(
        cluster: &Cluster,
        placement: &RankPlacement,
        calibration: SimulationCalibration,
    ) -> f64 {
        let hbm_bytes_per_sec = placement
            .rank_to_gpu
            .iter()
            .filter_map(|addr| cluster.gpu_profile(*addr))
            .map(|profile| profile.hbm_bandwidth.as_bytes_per_sec())
            .fold(f64::INFINITY, f64::min);

        hbm_bytes_per_sec.max(1.0) * calibration.decode_memory_bandwidth_scale
    }

    fn activation_bytes(model: &ModelSpec, request: &InferenceRequest) -> Bytes {
        let tokens = Self::active_tokens(request);
        let bytes = request.batch_size as u64
            * tokens
            * model.hidden_size as u64
            * model.dtype.bytes_per_element();

        Bytes::from_bytes(bytes)
    }

    fn active_tokens(request: &InferenceRequest) -> u64 {
        match request.phase {
            InferencePhase::Prefill => request.prompt_tokens as u64,
            InferencePhase::Decode => request.decode_tokens as u64,
            InferencePhase::EndToEnd => request.prompt_tokens as u64 + request.decode_tokens as u64,
        }
    }
}

fn search_deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

fn ordered_parallelism_candidates(search_space: &SearchSpace) -> Vec<ParallelismConfig> {
    let mut by_total_ranks = BTreeMap::<u32, Vec<ParallelismConfig>>::new();
    for &tensor_ranks in &search_space.tensor_ranks {
        for &pipeline_ranks in &search_space.pipeline_ranks {
            for &expert_ranks in &search_space.expert_ranks {
                for &data_ranks in &search_space.data_ranks {
                    let config = ParallelismConfig {
                        tensor_ranks,
                        pipeline_ranks,
                        expert_ranks,
                        data_ranks,
                    };
                    by_total_ranks
                        .entry(config.total_ranks())
                        .or_default()
                        .push(config);
                }
            }
        }
    }

    for configs in by_total_ranks.values_mut() {
        configs.sort_by_key(|config| {
            (
                dimension_spread(*config),
                config.tensor_ranks,
                config.pipeline_ranks,
                config.expert_ranks,
                config.data_ranks,
            )
        });
    }

    let rank_order = diverse_rank_order(by_total_ranks.keys().copied().collect());
    let mut ordered = Vec::new();
    loop {
        let mut advanced = false;
        for total_ranks in &rank_order {
            let Some(configs) = by_total_ranks.get_mut(total_ranks) else {
                continue;
            };
            if configs.is_empty() {
                continue;
            }
            ordered.push(configs.remove(0));
            advanced = true;
        }
        if !advanced {
            break;
        }
    }
    ordered
}

fn diverse_rank_order(ranks: Vec<u32>) -> Vec<u32> {
    if ranks.is_empty() {
        return Vec::new();
    }

    let mut ordered = Vec::with_capacity(ranks.len());
    let mut low = 0usize;
    let mut high = ranks.len() - 1;
    while low < high {
        ordered.push(ranks[low]);
        ordered.push(ranks[high]);
        low += 1;
        high -= 1;
    }
    if low == high {
        ordered.push(ranks[low]);
    }
    ordered
}

fn dimension_spread(config: ParallelismConfig) -> u32 {
    let dims = [
        config.tensor_ranks,
        config.pipeline_ranks,
        config.expert_ranks,
        config.data_ranks,
    ];
    dims.iter().copied().max().unwrap_or(1) - dims.iter().copied().min().unwrap_or(1)
}

fn div_ceil(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        value
    } else {
        value.div_ceil(divisor)
    }
}

fn min_bandwidth(current: Option<Bandwidth>, candidate: Bandwidth) -> Bandwidth {
    match current {
        Some(current) if current.as_bytes_per_sec() <= candidate.as_bytes_per_sec() => current,
        _ => candidate,
    }
}

fn push_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    approximation: SimulationApproximation,
) {
    if approximations.iter().any(|existing| {
        existing.phase == approximation.phase && existing.code == approximation.code
    }) {
        return;
    }
    approximations.push(approximation);
}

fn phase_label(phase: InferencePhase) -> &'static str {
    match phase {
        InferencePhase::Prefill => "prefill",
        InferencePhase::Decode => "decode",
        InferencePhase::EndToEnd => "end_to_end",
    }
}

fn placement_spans_nodes(placement: &RankPlacement) -> bool {
    let Some(first) = placement.rank_to_gpu.first() else {
        return false;
    };
    placement
        .rank_to_gpu
        .iter()
        .any(|addr| addr.node_id != first.node_id)
}

fn placement_node_count(placement: &RankPlacement) -> usize {
    placement
        .rank_to_gpu
        .iter()
        .map(|addr| addr.node_id)
        .collect::<BTreeSet<_>>()
        .len()
}

fn data_replicas_are_node_packed(config: ParallelismConfig, placement: &RankPlacement) -> bool {
    if config.data_ranks <= 1 || placement.rank_to_gpu.len() < config.total_ranks() as usize {
        return false;
    }

    for data_idx in 0..config.data_ranks {
        let nodes = (0..config.pipeline_ranks)
            .flat_map(|pipeline_idx| {
                (0..config.expert_ranks).flat_map(move |expert_idx| {
                    (0..config.tensor_ranks).map(move |tensor_idx| {
                        rank_id(config, data_idx, pipeline_idx, expert_idx, tensor_idx)
                    })
                })
            })
            .filter_map(|rank| placement.gpu_for_rank(rank).map(|gpu| gpu.node_id))
            .collect::<BTreeSet<_>>();
        if nodes.len() != 1 {
            return false;
        }
    }

    true
}

fn rank_id(
    config: ParallelismConfig,
    data_idx: u32,
    pipeline_idx: u32,
    expert_idx: u32,
    tensor_idx: u32,
) -> RankId {
    (((data_idx * config.pipeline_ranks + pipeline_idx) * config.expert_ranks + expert_idx)
        * config.tensor_ranks)
        + tensor_idx
}

fn decode_rank(config: ParallelismConfig, rank: RankId) -> (u32, u32, u32, u32) {
    let tensor_idx = rank % config.tensor_ranks;
    let rank = rank / config.tensor_ranks;
    let expert_idx = rank % config.expert_ranks;
    let rank = rank / config.expert_ranks;
    let pipeline_idx = rank % config.pipeline_ranks;
    let data_idx = rank / config.pipeline_ranks;

    (data_idx, pipeline_idx, expert_idx, tensor_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{CalibrationFitFeatureRange, CalibrationFittedModel, CalibrationProfileMetadata},
        types::{
            collective::{CollectiveAlgorithm, CollectiveKind, ReductionOp},
            fabric::variants::ib::IbVariant,
            gpu::Gpu,
        },
        workload::{DType, ExpertSpec},
    };

    fn cluster(node_count: u32) -> Cluster {
        Cluster::h100_sxm_nodes(node_count, IbVariant::Ndr.default_profile())
    }

    fn model() -> ModelSpec {
        ModelSpec {
            layers: 4,
            hidden_size: 4096,
            attention_heads: 32,
            kv_heads: 8,
            vocab_size: 32000,
            parameters: Bytes::from_gigabytes(16.0),
            parameter_count: None,
            dtype: DType::Bf16,
            kv_dtype: None,
            experts: None,
        }
    }

    fn cluster_with_gpu(gpu: Gpu) -> Cluster {
        let mut cluster = cluster(1);
        for node in cluster.nodes.values_mut() {
            for node_gpu in node.gpus.values_mut() {
                *node_gpu = gpu;
            }
        }
        cluster
    }

    fn request() -> InferenceRequest {
        InferenceRequest {
            batch_size: 4,
            prompt_tokens: 128,
            decode_tokens: 16,
            max_sequence_tokens: 256,
            phase: InferencePhase::EndToEnd,
        }
    }

    fn constant_latency_fit(target: &str, phase: &str, latency_ms: f64) -> CalibrationFittedModel {
        constant_fit(target, phase, latency_ms, "batch_size")
    }

    fn constant_fit(
        target: &str,
        phase: &str,
        latency_ms: f64,
        feature: &str,
    ) -> CalibrationFittedModel {
        CalibrationFittedModel {
            name: Some(format!("{phase}-{target}-fit")),
            target: target.to_string(),
            phase: Some(phase.to_string()),
            kind: Some("latency".to_string()),
            model: "linear".to_string(),
            unit: Some("ms".to_string()),
            intercept: Some(latency_ms),
            features: vec![feature.to_string()],
            coefficients: vec![0.0],
            feature_ranges: vec![CalibrationFitFeatureRange {
                feature: feature.to_string(),
                min: Some(0.0),
                max: Some(10_000.0),
            }],
            r_squared: Some(1.0),
            adjusted_r_squared: Some(1.0),
            rmse: Some(0.0),
            rmse_pct: Some(0.0),
            mean_abs_pct_error: Some(0.0),
            max_abs_pct_error: Some(0.0),
            validation_rmse: None,
            validation_rmse_pct: None,
            validation_mean_abs_pct_error: None,
            validation_max_abs_pct_error: None,
            confidence_interval: None,
            confidence_interval_pct: None,
            confidence_level: None,
            sample_count: Some(1),
            validation_sample_count: None,
            source: Some("unit-test".to_string()),
            notes: None,
        }
    }

    fn basis_fit(
        target: &str,
        phase: &str,
        intercept_ms: f64,
        terms: &[(&str, f64, Option<f64>, Option<f64>)],
    ) -> CalibrationFittedModel {
        let mut fit = constant_fit(target, phase, intercept_ms, "batch_size");
        fit.features = terms.iter().map(|(name, ..)| name.to_string()).collect();
        fit.coefficients = terms
            .iter()
            .map(|(_, coefficient, ..)| *coefficient)
            .collect();
        fit.feature_ranges = terms
            .iter()
            .map(|(name, _, min, max)| CalibrationFitFeatureRange {
                feature: name.to_string(),
                min: *min,
                max: *max,
            })
            .collect();
        fit
    }

    fn profile_with_fits(fits: Vec<CalibrationFittedModel>) -> CalibrationProfileMetadata {
        CalibrationProfileMetadata {
            path: "unit-test-profile.toml".to_string(),
            name: Some("unit-test-profile".to_string()),
            hardware: None,
            fabric: None,
            model: None,
            dtype: None,
            serving_stack: None,
            serving_runtime_features: Vec::new(),
            backend_version: None,
            driver_version: None,
            cuda_version: None,
            rocm_version: None,
            nccl_version: None,
            rccl_version: None,
            ucx_version: None,
            kernel_settings: Vec::new(),
            environment_hash: None,
            source: Some("unit-test".to_string()),
            date: None,
            notes: None,
            valid_shape: None,
            invalid_shapes: Vec::new(),
            fits,
            benchmarks: Vec::new(),
        }
    }

    #[test]
    fn validates_parallelism_dimensions() {
        let config = ParallelismConfig {
            tensor_ranks: 0,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        assert!(config.validate_dimensions().is_err());
    }

    #[test]
    fn tensor_parallel_group_packs_inside_one_h100_node() {
        let cluster = cluster(1);
        let config = ParallelismConfig {
            tensor_ranks: 8,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        let placement = Solver::place_ranks(
            &cluster,
            &model(),
            config,
            Solver::estimate_memory_per_gpu(&model(), &request(), config),
        )
        .unwrap();
        let groups = Solver::build_groups(config);

        assert_eq!(groups.tensor_groups, vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);
        assert_eq!(
            placement.rank_to_gpu,
            (0..8)
                .map(|local_gpu_id| GpuAddr {
                    node_id: 0,
                    local_gpu_id
                })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rank_placement_skips_disabled_gpus() {
        let mut cluster = cluster(1);
        let node = cluster.nodes.get_mut(&0).unwrap();
        node.disabled_gpus.extend(0..7);
        node.gpu_operational_states
            .insert(7, crate::types::common::OperationalState::Maintenance);
        node.disabled_gpus.insert(7);
        node.gpu_operational_states
            .insert(6, crate::types::common::OperationalState::Healthy);
        node.disabled_gpus.remove(&6);
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        let placement = Solver::place_ranks(
            &cluster,
            &model(),
            config,
            Solver::estimate_memory_per_gpu(&model(), &request(), config),
        )
        .unwrap();

        assert_eq!(
            placement.rank_to_gpu,
            vec![GpuAddr {
                node_id: 0,
                local_gpu_id: 6
            }]
        );
    }

    #[test]
    fn data_parallel_replicas_spread_across_nodes() {
        let cluster = cluster(2);
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 2,
        };

        let placement = Solver::place_ranks(
            &cluster,
            &model(),
            config,
            Solver::estimate_memory_per_gpu(&model(), &request(), config),
        )
        .unwrap();
        let groups = Solver::build_groups(config);

        assert_eq!(groups.data_groups, vec![vec![0, 1]]);
        assert_eq!(placement.gpu_for_rank(0).unwrap().node_id, 0);
        assert_eq!(placement.gpu_for_rank(1).unwrap().node_id, 1);
    }

    #[test]
    fn heterogeneous_placement_prefers_faster_gpus_over_node_order() {
        let mut cluster = cluster(2);
        for gpu in cluster.nodes.get_mut(&0).unwrap().gpus.values_mut() {
            *gpu = Gpu::A100_40GB;
        }
        for gpu in cluster.nodes.get_mut(&1).unwrap().gpus.values_mut() {
            *gpu = Gpu::H100_SXM;
        }
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        let placement = Solver::place_ranks(
            &cluster,
            &model(),
            config,
            Solver::estimate_memory_per_gpu(&model(), &request(), config),
        )
        .unwrap();

        assert_eq!(placement.gpu_for_rank(0).unwrap().node_id, 1);
    }

    #[test]
    fn placement_uses_effective_gpu_profile_overrides() {
        let mut cluster = cluster(2);
        let node = cluster.nodes.get_mut(&0).unwrap();
        for local_gpu_id in 0..8 {
            let mut profile = node.gpu_profile(local_gpu_id).unwrap();
            profile.peak_f16_flops *= 0.1;
            profile.peak_f8_flops = profile.peak_f8_flops.map(|flops| flops * 0.1);
            node.gpu_profile_overrides.insert(local_gpu_id, profile);
        }
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        let placement = Solver::place_ranks(
            &cluster,
            &model(),
            config,
            Solver::estimate_memory_per_gpu(&model(), &request(), config),
        )
        .unwrap();

        assert_eq!(placement.gpu_for_rank(0).unwrap().node_id, 1);
    }

    #[test]
    fn fp8_placement_requires_fp8_capable_gpus() {
        let cluster = cluster_with_gpu(Gpu::A100_80GB);
        let mut fp8_model = model();
        fp8_model.dtype = DType::Fp8;
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        let score = Solver::score_config(&cluster, &fp8_model, &request(), config);

        assert!(!score.feasible);
        assert!(score.rejected_reason.unwrap().contains("model.dtype fp8"));
    }

    #[test]
    fn kv_dtype_controls_kv_cache_memory_estimate() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let bf16_model = model();
        let mut fp8_kv_model = model();
        fp8_kv_model.kv_dtype = Some(DType::Fp8);

        let bf16_kv = Solver::kv_cache_bytes(&bf16_model, &request(), config);
        let fp8_kv = Solver::kv_cache_bytes(&fp8_kv_model, &request(), config);

        assert_eq!(fp8_kv * 2, bf16_kv);
        assert!(
            Solver::estimate_memory_per_gpu(&fp8_kv_model, &request(), config)
                < Solver::estimate_memory_per_gpu(&bf16_model, &request(), config)
        );
    }

    #[test]
    fn explicit_parameter_count_controls_compute_estimate_without_changing_weight_memory() {
        let cluster = cluster(1);
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let baseline_model = model();
        let mut larger_compute_model = model();
        larger_compute_model.parameter_count = Some(baseline_model.parameter_count() * 2.0);

        let baseline = Solver::score_config(&cluster, &baseline_model, &request(), config);
        let larger_compute =
            Solver::score_config(&cluster, &larger_compute_model, &request(), config);

        assert_eq!(
            baseline.estimated_memory_per_gpu,
            larger_compute.estimated_memory_per_gpu
        );
        assert!(larger_compute.estimated_latency_s > baseline.estimated_latency_s);
    }

    #[test]
    fn heterogeneous_placement_uses_larger_hbm_when_required() {
        let mut cluster = cluster(2);
        for gpu in cluster.nodes.get_mut(&0).unwrap().gpus.values_mut() {
            *gpu = Gpu::H100_SXM;
        }
        for gpu in cluster.nodes.get_mut(&1).unwrap().gpus.values_mut() {
            *gpu = Gpu::H200_SXM;
        }
        let mut large_model = model();
        large_model.parameters = Bytes::from_gigabytes(100.0);
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        let score = Solver::score_config(&cluster, &large_model, &request(), config);

        assert!(score.feasible);
        assert_eq!(score.placement.gpu_for_rank(0).unwrap().node_id, 1);
    }

    #[test]
    fn score_reports_selected_placement_evidence() {
        let cluster = cluster(2);
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 2,
        };

        let score = Solver::score_config(&cluster, &model(), &request(), config);

        assert!(score.feasible);
        assert!(score.placement_evidence.iter().any(|evidence| {
            evidence.decision == "selected"
                && evidence.scope == "rank_placement"
                && evidence.resource == "rank_placement"
                && evidence.code == "data_replica_node_packed_placement"
                && evidence.observed == Some(2.0)
                && evidence.unit.as_deref() == Some("ranks")
        }));
        assert!(score.placement_evidence.iter().any(|evidence| {
            evidence.decision == "selected"
                && evidence.resource == "inter_node"
                && evidence.code == "placement_spans_nodes"
                && evidence.unit.as_deref() == Some("nodes")
        }));
    }

    #[test]
    fn score_reports_rejected_placement_evidence() {
        let cluster = cluster(1);
        let mut oversized_model = model();
        oversized_model.parameters = Bytes::from_gigabytes(10_000.0);
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };

        let score = Solver::score_config(&cluster, &oversized_model, &request(), config);

        assert!(!score.feasible);
        assert!(score.placement_evidence.iter().any(|evidence| {
            evidence.decision == "rejected"
                && evidence.scope == "gpu_eligibility"
                && evidence.resource == "eligible_gpus"
                && evidence.code == "insufficient_hbm_eligible_gpus"
                && evidence.observed == Some(0.0)
                && evidence.limit == Some(1.0)
                && evidence.unit.as_deref() == Some("gpus")
        }));
    }

    #[test]
    fn explicit_placement_overrides_capability_ordering() {
        let mut cluster = cluster(2);
        for gpu in cluster.nodes.get_mut(&0).unwrap().gpus.values_mut() {
            *gpu = Gpu::A100_40GB;
        }
        for gpu in cluster.nodes.get_mut(&1).unwrap().gpus.values_mut() {
            *gpu = Gpu::H100_SXM;
        }
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let placement = RankPlacement {
            rank_to_gpu: vec![GpuAddr {
                node_id: 0,
                local_gpu_id: 0,
            }],
        };

        let score = Solver::score_config_with_options(
            &cluster,
            &model(),
            &request(),
            config,
            SolverOptions {
                explicit_placement: Some(&placement),
                ..SolverOptions::default()
            },
        );

        assert!(score.feasible);
        assert_eq!(score.placement.gpu_for_rank(0).unwrap().node_id, 0);
        assert!(score.placement_evidence.iter().any(|evidence| {
            evidence.decision == "selected"
                && evidence.resource == "rank_placement"
                && evidence.code == "explicit_rank_placement"
        }));
    }

    #[test]
    fn explicit_placement_rejects_rank_count_mismatch() {
        let config = ParallelismConfig {
            tensor_ranks: 2,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let placement = RankPlacement {
            rank_to_gpu: vec![GpuAddr {
                node_id: 0,
                local_gpu_id: 0,
            }],
        };

        let score = Solver::score_config_with_options(
            &cluster(1),
            &model(),
            &request(),
            config,
            SolverOptions {
                explicit_placement: Some(&placement),
                ..SolverOptions::default()
            },
        );

        assert!(!score.feasible);
        assert!(score.placement_evidence.iter().any(|evidence| {
            evidence.decision == "rejected"
                && evidence.resource == "rank_count"
                && evidence.code == "explicit_placement_rank_count_mismatch"
                && evidence.observed == Some(1.0)
                && evidence.limit == Some(2.0)
                && evidence.unit.as_deref() == Some("ranks")
        }));
    }

    #[test]
    fn decode_latency_uses_hbm_bandwidth_floor() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let decode_request = InferenceRequest {
            batch_size: 1,
            prompt_tokens: 128,
            decode_tokens: 32,
            max_sequence_tokens: 256,
            phase: InferencePhase::Decode,
        };

        let h100_score = Solver::score_config(
            &cluster_with_gpu(Gpu::H100_SXM),
            &model(),
            &decode_request,
            config,
        );
        let h200_score = Solver::score_config(
            &cluster_with_gpu(Gpu::H200_SXM),
            &model(),
            &decode_request,
            config,
        );

        assert!(h100_score.feasible);
        assert!(h200_score.feasible);
        assert!(h200_score.estimated_latency_s < h100_score.estimated_latency_s);
    }

    #[test]
    fn decode_memory_bandwidth_scale_is_calibratable() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let decode_request = InferenceRequest {
            batch_size: 1,
            prompt_tokens: 128,
            decode_tokens: 32,
            max_sequence_tokens: 256,
            phase: InferencePhase::Decode,
        };
        let cluster = cluster_with_gpu(Gpu::H100_SXM);
        let baseline = Solver::score_config(&cluster, &model(), &decode_request, config);
        let slower_hbm = Solver::score_config_with_options(
            &cluster,
            &model(),
            &decode_request,
            config,
            SolverOptions {
                calibration: SimulationCalibration {
                    decode_memory_bandwidth_scale: 0.5,
                    ..SimulationCalibration::default()
                },
                calibration_profile: None,
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );

        assert!(baseline.feasible);
        assert!(slower_hbm.feasible);
        assert!(slower_hbm.estimated_latency_s > baseline.estimated_latency_s * 1.5);
        assert!(
            slower_hbm
                .bottlenecks
                .iter()
                .any(|bottleneck| bottleneck.contains("gpu HBM"))
        );
    }

    #[test]
    fn fitted_phase_latencies_override_compute_estimates() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let request = InferenceRequest {
            batch_size: 1,
            prompt_tokens: 128,
            decode_tokens: 2,
            max_sequence_tokens: 256,
            phase: InferencePhase::EndToEnd,
        };
        let profile = profile_with_fits(vec![
            constant_latency_fit("prefill_ms", "prefill", 5.0),
            constant_latency_fit("decode_ms", "decode", 7.0),
        ]);
        let score = Solver::score_config_with_options(
            &cluster(1),
            &model(),
            &request,
            config,
            SolverOptions {
                calibration: SimulationCalibration::default(),
                calibration_profile: Some(&profile),
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );

        assert!(score.feasible);
        assert!((score.estimated_latency_s - 0.012).abs() < 1e-9);
        assert_eq!(score.calibration_fits.len(), 2);
        assert_eq!(score.calibration_fits[0].phase, "prefill");
        assert_eq!(
            score.calibration_fits[0].fit_name.as_deref(),
            Some("prefill-prefill_ms-fit")
        );
        assert!(score.calibration_fits[0].baseline_s.is_some());
        assert_eq!(
            score.calibration_fits[0].applicability_status,
            "interpolated"
        );
        assert_eq!(score.calibration_fits[0].confidence_score, 1.0);
        assert_eq!(
            score.calibration_fits[0].relative_uncertainty_pct,
            Some(0.0)
        );
        assert_eq!(score.calibration_fits[0].absolute_uncertainty_s, Some(0.0));
        assert_eq!(
            score.calibration_fits[0].uncertainty_source.as_deref(),
            Some("rmse")
        );
        assert_eq!(score.calibration_fits[0].features[0].name, "batch_size");
        assert_eq!(score.calibration_fits[0].features[0].status, "in_range");
        assert_eq!(score.calibration_fits[1].phase, "decode");
    }

    #[test]
    fn fit_applications_report_extrapolation_confidence() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let request = InferenceRequest {
            batch_size: 4,
            prompt_tokens: 128,
            decode_tokens: 1,
            max_sequence_tokens: 256,
            phase: InferencePhase::Prefill,
        };
        let mut fit = constant_latency_fit("prefill_ms", "prefill", 5.0);
        fit.feature_ranges[0].max = Some(2.0);
        fit.rmse = Some(0.5);
        fit.rmse_pct = Some(10.0);
        fit.validation_rmse = Some(0.25);
        fit.validation_rmse_pct = Some(5.0);
        let profile = profile_with_fits(vec![fit]);
        let score = Solver::score_config_with_options(
            &cluster(1),
            &model(),
            &request,
            config,
            SolverOptions {
                calibration: SimulationCalibration::default(),
                calibration_profile: Some(&profile),
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );

        let application = &score.calibration_fits[0];
        assert_eq!(application.applicability_status, "extrapolated");
        assert!(application.confidence_score < 1.0);
        assert!(application.max_extrapolation_ratio > 0.0);
        assert_eq!(application.features[0].status, "extrapolated");
        assert_eq!(application.relative_uncertainty_pct, Some(5.0));
        assert_eq!(application.validation_rmse, Some(0.25));
        assert_eq!(application.validation_rmse_pct, Some(5.0));
        assert!((application.absolute_uncertainty_s.unwrap() - 0.00025).abs() < 1e-12);
        assert_eq!(
            application.uncertainty_source.as_deref(),
            Some("validation_rmse")
        );
    }

    #[test]
    fn fit_applications_prefer_confidence_interval_uncertainty() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let request = InferenceRequest {
            batch_size: 1,
            prompt_tokens: 128,
            decode_tokens: 1,
            max_sequence_tokens: 256,
            phase: InferencePhase::Prefill,
        };
        let mut fit = constant_latency_fit("prefill_ms", "prefill", 5.0);
        fit.rmse = Some(0.1);
        fit.rmse_pct = Some(2.0);
        fit.validation_rmse = Some(0.2);
        fit.validation_rmse_pct = Some(4.0);
        fit.confidence_interval = Some(0.5);
        fit.confidence_interval_pct = Some(12.0);
        fit.confidence_level = Some(0.95);
        let profile = profile_with_fits(vec![fit]);
        let score = Solver::score_config_with_options(
            &cluster(1),
            &model(),
            &request,
            config,
            SolverOptions {
                calibration: SimulationCalibration::default(),
                calibration_profile: Some(&profile),
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );

        let application = &score.calibration_fits[0];
        assert_eq!(application.relative_uncertainty_pct, Some(12.0));
        assert!((application.absolute_uncertainty_s.unwrap() - 0.0005).abs() < 1e-12);
        assert_eq!(
            application.uncertainty_source.as_deref(),
            Some("confidence_interval")
        );
        assert_eq!(application.confidence_interval, Some(0.5));
        assert_eq!(application.confidence_interval_pct, Some(12.0));
        assert_eq!(application.confidence_level, Some(0.95));
    }

    #[test]
    fn derived_basis_features_have_exact_values() {
        let config = ParallelismConfig {
            tensor_ranks: 2,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        // batch_size = 4, prompt_tokens = 128, decode_tokens = 16, tensor_ranks = 2
        // mean decode context = 128 + (16 + 1) / 2 = 136.5
        let features = calibration_feature_values(&model(), &request(), config, None);

        let expected = [
            ("prompt_tokens_squared", 16_384.0),
            ("batch_prompt_tokens_squared", 65_536.0),
            ("decode_context_tokens", 136.5),
            ("batch_decode_context_tokens", 546.0),
            ("decode_batch_context_tokens", 8_736.0),
            ("inv_tensor_ranks", 0.5),
            ("batch_prompt_tokens_per_tensor_rank", 256.0),
            ("batch_prompt_tokens_squared_per_tensor_rank", 32_768.0),
            ("decode_tokens_per_tensor_rank", 8.0),
            ("decode_batch_tokens_per_tensor_rank", 32.0),
            ("decode_batch_context_tokens_per_tensor_rank", 4_368.0),
        ];

        for (name, value) in expected {
            let actual = features
                .get(name)
                .unwrap_or_else(|| panic!("missing calibration basis feature {name}"));
            assert!(
                (actual - value).abs() < 1e-9,
                "feature {name} expected {value} but was {actual}"
            );
        }
    }

    #[test]
    fn quadratic_basis_fit_predicts_prefill_latency() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let request = InferenceRequest {
            batch_size: 4,
            prompt_tokens: 128,
            decode_tokens: 16,
            max_sequence_tokens: 256,
            phase: InferencePhase::Prefill,
        };
        let fit = basis_fit(
            "prefill_ms",
            "prefill",
            1.0,
            &[
                (
                    "batch_prompt_tokens_per_tensor_rank",
                    0.01,
                    Some(0.0),
                    Some(4_096.0),
                ),
                (
                    "batch_prompt_tokens_squared_per_tensor_rank",
                    0.0001,
                    Some(0.0),
                    Some(1_000_000.0),
                ),
            ],
        );
        let profile = profile_with_fits(vec![fit]);
        let score = Solver::score_config_with_options(
            &cluster(1),
            &model(),
            &request,
            config,
            SolverOptions {
                calibration: SimulationCalibration::default(),
                calibration_profile: Some(&profile),
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );

        // b * s / tp = 512, b * s^2 / tp = 65536
        let expected_ms = 1.0 + 0.01 * 512.0 + 0.0001 * 65_536.0;
        assert!(score.feasible);
        assert_eq!(score.calibration_fits.len(), 1);
        let application = &score.calibration_fits[0];
        assert!((application.raw_prediction - expected_ms).abs() < 1e-9);
        assert!((score.estimated_latency_s - expected_ms / 1e3).abs() < 1e-12);
        assert_eq!(application.applicability_status, "interpolated");
        assert_eq!(
            application.features[0].name,
            "batch_prompt_tokens_per_tensor_rank"
        );
        assert_eq!(application.features[0].status, "in_range");
        assert_eq!(application.features[1].status, "in_range");
    }

    #[test]
    fn per_tensor_rank_basis_fit_scales_across_tensor_parallel_sweep() {
        let request = InferenceRequest {
            batch_size: 4,
            prompt_tokens: 128,
            decode_tokens: 16,
            max_sequence_tokens: 256,
            phase: InferencePhase::Prefill,
        };
        let fit = basis_fit(
            "prefill_ms",
            "prefill",
            0.0,
            &[
                (
                    "batch_prompt_tokens_per_tensor_rank",
                    0.01,
                    Some(0.0),
                    Some(4_096.0),
                ),
                (
                    "batch_prompt_tokens_squared_per_tensor_rank",
                    0.0001,
                    Some(0.0),
                    Some(1_000_000.0),
                ),
            ],
        );
        let profile = profile_with_fits(vec![fit]);
        let prediction_ms = |tensor_ranks: u32| {
            let score = Solver::score_config_with_options(
                &cluster(1),
                &model(),
                &request,
                ParallelismConfig {
                    tensor_ranks,
                    pipeline_ranks: 1,
                    expert_ranks: 1,
                    data_ranks: 1,
                },
                SolverOptions {
                    calibration: SimulationCalibration::default(),
                    calibration_profile: Some(&profile),
                    max_candidates: None,
                    search_deadline: None,
                    explicit_placement: None,
                },
            );
            assert!(score.feasible);
            assert_eq!(score.calibration_fits.len(), 1);
            score.calibration_fits[0].raw_prediction
        };

        let tp1 = prediction_ms(1);
        let tp2 = prediction_ms(2);

        assert!((tp1 - (0.01 * 512.0 + 0.0001 * 65_536.0)).abs() < 1e-9);
        assert!((tp2 - (0.01 * 256.0 + 0.0001 * 32_768.0)).abs() < 1e-9);
        assert!((tp1 - 2.0 * tp2).abs() < 1e-9);
    }

    #[test]
    fn basis_feature_range_bounds_report_extrapolation() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        };
        let request = InferenceRequest {
            batch_size: 4,
            prompt_tokens: 128,
            decode_tokens: 16,
            max_sequence_tokens: 256,
            phase: InferencePhase::Prefill,
        };
        let fit = basis_fit(
            "prefill_ms",
            "prefill",
            1.0,
            &[
                (
                    "batch_prompt_tokens_per_tensor_rank",
                    0.01,
                    Some(0.0),
                    Some(4_096.0),
                ),
                (
                    "batch_prompt_tokens_squared_per_tensor_rank",
                    0.0001,
                    Some(0.0),
                    // b * s^2 / tp is 65536 for this request, well above the bound.
                    Some(16_384.0),
                ),
            ],
        );
        let profile = profile_with_fits(vec![fit]);
        let score = Solver::score_config_with_options(
            &cluster(1),
            &model(),
            &request,
            config,
            SolverOptions {
                calibration: SimulationCalibration::default(),
                calibration_profile: Some(&profile),
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );

        let application = &score.calibration_fits[0];
        assert_eq!(application.applicability_status, "extrapolated");
        assert_eq!(application.features[0].status, "in_range");
        assert_eq!(application.features[1].status, "extrapolated");
        assert!(application.max_extrapolation_ratio > 0.0);
    }

    #[test]
    fn fitted_kv_transfer_latency_overrides_transfer_estimate() {
        let profile = profile_with_fits(vec![constant_fit(
            "kv_transfer_ms",
            "kv_transfer",
            7.0,
            "kv_transfer_gb",
        )]);
        let (cost, fit) = Solver::estimate_transfer_between_nodes_with_observation(
            &cluster(2),
            &[0],
            &[1],
            Bytes::from_gigabytes(1.0),
            SolverOptions {
                calibration: SimulationCalibration::default(),
                calibration_profile: Some(&profile),
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );
        let fit = fit.expect("expected KV transfer fit application");

        assert!((cost.total_s - 0.007).abs() < 1e-9);
        assert_eq!(fit.phase, "kv_transfer");
        assert_eq!(fit.target, "kv_transfer_ms");
        assert!(fit.baseline_s.is_some());
        assert_eq!(fit.applicability_status, "interpolated");
        assert_eq!(fit.features[0].name, "kv_transfer_gb");
    }

    #[test]
    fn over_capacity_config_is_rejected() {
        let results = Solver::rank_configs(
            &cluster(1),
            &model(),
            &request(),
            &SearchSpace {
                tensor_ranks: vec![16],
                pipeline_ranks: vec![1],
                expert_ranks: vec![1],
                data_ranks: vec![1],
            },
        );

        assert_eq!(results.len(), 1);
        assert!(!results[0].feasible);
        assert!(
            results[0]
                .rejected_reason
                .as_ref()
                .unwrap()
                .contains("requires 16 ranks")
        );
    }

    #[test]
    fn same_node_collective_is_cheaper_than_cross_node_collective() {
        let cluster = cluster(2);
        let call = CollectiveCall {
            kind: CollectiveKind::AllReduce,
            participants: vec![0, 1],
            bytes_per_rank: Bytes::from_megabytes(64.0),
            dtype: DType::Bf16,
            reduction: Some(ReductionOp::Sum),
            root: None,
            phase: InferencePhase::Prefill,
            algorithm: CollectiveAlgorithm::Hierarchical,
        };
        let same_node = RankPlacement {
            rank_to_gpu: vec![
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 1,
                },
            ],
        };
        let cross_node = RankPlacement {
            rank_to_gpu: vec![
                GpuAddr {
                    node_id: 0,
                    local_gpu_id: 0,
                },
                GpuAddr {
                    node_id: 1,
                    local_gpu_id: 0,
                },
            ],
        };

        let same_cost = Solver::estimate_collective(&cluster, &same_node, &call);
        let cross_cost = Solver::estimate_collective(&cluster, &cross_node, &call);

        assert!(same_cost.total_s < cross_cost.total_s);
        assert!(
            cross_cost
                .bottlenecks
                .iter()
                .any(|bottleneck| bottleneck.contains("fabric"))
        );
        assert!(!cross_cost.bottlenecks.is_empty());
    }

    #[test]
    fn all_to_all_is_generated_only_for_expert_models() {
        let config = ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 2,
            data_ranks: 1,
        };
        let groups = Solver::build_groups(config);
        let mut expert_model = model();
        expert_model.experts = Some(ExpertSpec {
            expert_count: 8,
            top_k: 2,
        });

        let calls = Solver::build_collectives(&expert_model, &request(), config, &groups);

        assert!(
            calls
                .iter()
                .any(|call| call.kind == CollectiveKind::AllToAll)
        );
    }

    #[test]
    fn solver_ranks_feasible_configs_deterministically() {
        let search_space = SearchSpace {
            tensor_ranks: vec![1, 2],
            pipeline_ranks: vec![1],
            expert_ranks: vec![1],
            data_ranks: vec![1],
        };

        let first = Solver::rank_configs(&cluster(1), &model(), &request(), &search_space);
        let second = Solver::rank_configs(&cluster(1), &model(), &request(), &search_space);

        assert_eq!(first, second);
        assert!(first.iter().all(|score| score.feasible));
        assert!(!first[0].resource_utilization.is_empty());
        assert!(first[0].resource_utilization[0].utilization > 0.0);
    }

    #[test]
    fn rank_configs_respects_candidate_budget() {
        let search_space = SearchSpace {
            tensor_ranks: vec![1, 2, 4],
            pipeline_ranks: vec![1, 2],
            expert_ranks: vec![1],
            data_ranks: vec![1],
        };

        let results = Solver::rank_configs_with_options(
            &cluster(4),
            &model(),
            &request(),
            &search_space,
            SolverOptions {
                max_candidates: Some(3),
                ..SolverOptions::default()
            },
        );

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn rank_config_candidate_order_covers_small_and_large_configs() {
        let search_space = SearchSpace {
            tensor_ranks: vec![1, 4],
            pipeline_ranks: vec![1, 2],
            expert_ranks: vec![1],
            data_ranks: vec![1, 8],
        };

        let ordered = ordered_parallelism_candidates(&search_space);

        assert_eq!(ordered[0].total_ranks(), 1);
        assert_eq!(ordered[1].total_ranks(), 64);
        assert_eq!(ordered[2].total_ranks(), 2);
    }

    #[test]
    fn rank_config_budget_uses_diverse_candidate_order() {
        let search_space = SearchSpace {
            tensor_ranks: vec![1, 4],
            pipeline_ranks: vec![1, 2],
            expert_ranks: vec![1],
            data_ranks: vec![1, 8],
        };

        let results = Solver::rank_configs_with_options(
            &cluster(8),
            &model(),
            &request(),
            &search_space,
            SolverOptions {
                max_candidates: Some(2),
                ..SolverOptions::default()
            },
        );

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|score| score.config.total_ranks() == 64));
    }

    #[test]
    fn memory_infeasible_model_is_rejected() {
        let mut huge_model = model();
        huge_model.parameters = Bytes::from_gigabytes(200.0);

        let results = Solver::rank_configs(
            &cluster(1),
            &huge_model,
            &request(),
            &SearchSpace {
                tensor_ranks: vec![1],
                pipeline_ranks: vec![1],
                expert_ranks: vec![1],
                data_ranks: vec![1],
            },
        );

        assert!(!results[0].feasible);
        assert!(
            results[0]
                .rejected_reason
                .as_ref()
                .unwrap()
                .contains("have enough HBM and dtype support")
        );
    }

    fn single_rank_config() -> ParallelismConfig {
        ParallelismConfig {
            tensor_ranks: 1,
            pipeline_ranks: 1,
            expert_ranks: 1,
            data_ranks: 1,
        }
    }

    fn default_options() -> SolverOptions<'static> {
        SolverOptions {
            calibration: SimulationCalibration::default(),
            calibration_profile: None,
            max_candidates: None,
            search_deadline: None,
            explicit_placement: None,
        }
    }

    #[test]
    fn prefill_attention_flops_follow_causal_quadratic_formula() {
        let spec = model();
        let request = InferenceRequest {
            batch_size: 4,
            prompt_tokens: 128,
            decode_tokens: 16,
            max_sequence_tokens: 256,
            phase: InferencePhase::Prefill,
        };

        let expected = 2.0 * 4.0 * 4096.0 * 128.0 * 128.0 * 4.0;
        assert!((Solver::prefill_attention_flops(&spec, &request) - expected).abs() < 1.0);
    }

    #[test]
    fn decode_kv_read_bytes_scale_with_context_and_kv_heads() {
        let spec = model();
        let request = InferenceRequest {
            batch_size: 4,
            prompt_tokens: 128,
            decode_tokens: 16,
            max_sequence_tokens: 256,
            phase: InferencePhase::Decode,
        };

        let average_context = 128.0 + (16.0 + 1.0) / 2.0;
        let head_dim = 4096.0 / 32.0;
        let expected = 2.0 * 4.0 * 8.0 * head_dim * 2.0 * average_context * 16.0 * 4.0;
        let baseline = Solver::decode_kv_read_bytes(&spec, &request);
        assert!((baseline - expected).abs() < 1.0);

        let mut gqa = model();
        gqa.kv_heads = 4;
        let halved = Solver::decode_kv_read_bytes(&gqa, &request);
        assert!((halved - baseline / 2.0).abs() < 1.0);
    }

    #[test]
    fn attention_terms_ignore_expert_sharding() {
        let config = ParallelismConfig {
            tensor_ranks: 2,
            pipeline_ranks: 2,
            expert_ranks: 8,
            data_ranks: 1,
        };

        assert!((Solver::attention_shard_factor(config) - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn prefill_latency_grows_superlinearly_with_prompt_tokens() {
        let mut spec = model();
        spec.layers = 32;
        let short_request = InferenceRequest {
            batch_size: 1,
            prompt_tokens: 16_384,
            decode_tokens: 1,
            max_sequence_tokens: 40_000,
            phase: InferencePhase::Prefill,
        };
        let long_request = InferenceRequest {
            prompt_tokens: 32_768,
            ..short_request
        };

        let short_score = Solver::score_config_with_options(
            &cluster(1),
            &spec,
            &short_request,
            single_rank_config(),
            default_options(),
        );
        let long_score = Solver::score_config_with_options(
            &cluster(1),
            &spec,
            &long_request,
            single_rank_config(),
            default_options(),
        );

        assert!(short_score.feasible);
        assert!(long_score.feasible);
        assert!(long_score.estimated_latency_s > short_score.estimated_latency_s * 2.1);
    }

    #[test]
    fn decode_latency_increases_with_context_length() {
        let spec = model();
        let short_context = InferenceRequest {
            batch_size: 16,
            prompt_tokens: 512,
            decode_tokens: 32,
            max_sequence_tokens: 1_024,
            phase: InferencePhase::Decode,
        };
        let long_context = InferenceRequest {
            prompt_tokens: 16_384,
            max_sequence_tokens: 16_512,
            ..short_context
        };

        let short_score = Solver::score_config_with_options(
            &cluster(1),
            &spec,
            &short_context,
            single_rank_config(),
            default_options(),
        );
        let long_score = Solver::score_config_with_options(
            &cluster(1),
            &spec,
            &long_context,
            single_rank_config(),
            default_options(),
        );

        assert!(short_score.feasible);
        assert!(long_score.feasible);
        assert!(long_score.estimated_latency_s > short_score.estimated_latency_s * 1.05);
    }

    #[test]
    fn decode_memory_latency_includes_weight_and_kv_reads() {
        let spec = model();
        let request = InferenceRequest {
            batch_size: 16,
            prompt_tokens: 16_384,
            decode_tokens: 32,
            max_sequence_tokens: 16_512,
            phase: InferencePhase::Decode,
        };

        let score = Solver::score_config_with_options(
            &cluster(1),
            &spec,
            &request,
            single_rank_config(),
            default_options(),
        );

        let profile = Gpu::H100_SXM.profile();
        let bandwidth = profile.hbm_bandwidth.as_bytes_per_sec();
        let weight_read_s = spec.parameters.as_bytes() as f64 * 32.0 / bandwidth;
        let kv_read_s = Solver::decode_kv_read_bytes(&spec, &request) / bandwidth;
        let expected = weight_read_s + kv_read_s;

        assert!(score.feasible);
        assert!((score.estimated_latency_s - expected).abs() < expected * 1e-9);
    }
}
