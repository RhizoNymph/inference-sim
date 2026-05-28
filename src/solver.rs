use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    time::Instant,
};

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

        'search: for &tensor_ranks in &search_space.tensor_ranks {
            for &pipeline_ranks in &search_space.pipeline_ranks {
                for &expert_ranks in &search_space.expert_ranks {
                    for &data_ranks in &search_space.data_ranks {
                        if options
                            .max_candidates
                            .is_some_and(|max_candidates| results.len() >= max_candidates)
                            || search_deadline_expired(options.search_deadline)
                        {
                            break 'search;
                        }
                        let config = ParallelismConfig {
                            tensor_ranks,
                            pipeline_ranks,
                            expert_ranks,
                            data_ranks,
                        };
                        results.push(Self::score_config_with_options(
                            cluster, model, request, config, options,
                        ));
                    }
                }
            }
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

    fn reject(mut score: ScoredParallelismConfig, reason: String) -> ScoredParallelismConfig {
        score.feasible = false;
        score.estimated_latency_s = f64::INFINITY;
        score.rejected_reason = Some(reason);
        score
    }

    fn placement_evidence(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        placement: &RankPlacement,
        required_memory_per_gpu: Bytes,
        strategy_code: &str,
    ) -> Vec<PlacementEvidence> {
        let total_ranks = config.total_ranks();
        let available_gpus = cluster.available_gpus();
        let eligible_gpus =
            Self::sorted_eligible_gpus(cluster, model, required_memory_per_gpu).len() as u32;
        let strategy_message = if strategy_code == "explicit_rank_placement" {
            format!(
                "selected {total_ranks} ranks from explicit rank placement config over {available_gpus} available GPUs"
            )
        } else {
            format!(
                "selected {total_ranks} ranks from {available_gpus} available GPUs using deterministic capability and HBM ordering"
            )
        };
        let mut evidence = vec![
            PlacementEvidence::new(
                "selected",
                "rank_placement",
                "rank_placement",
                strategy_code,
                strategy_message,
            )
            .with_details(
                Some(f64::from(total_ranks)),
                Some(f64::from(available_gpus)),
                Some("ranks"),
                Some(
                    "add topology-aware placement constraints before relying on fine-grained locality",
                ),
            ),
            PlacementEvidence::new(
                "selected",
                "gpu_eligibility",
                "eligible_gpus",
                "hbm_capable_gpus_available",
                format!(
                    "{eligible_gpus} GPUs have enough HBM for {:.2} GB per-rank memory",
                    required_memory_per_gpu.as_gigabytes()
                ),
            )
            .with_details(
                Some(f64::from(eligible_gpus)),
                Some(f64::from(total_ranks)),
                Some("gpus"),
                Some("increase HBM capacity, reduce model/request memory, or reduce rank count"),
            ),
        ];

        if placement_spans_nodes(placement) {
            evidence.push(
                PlacementEvidence::new(
                    "selected",
                    "topology",
                    "inter_node",
                    "placement_spans_nodes",
                    "selected rank placement spans multiple nodes and depends on modeled inter-node communication",
                )
                .with_details(
                    Some(placement_node_count(placement) as f64),
                    None,
                    Some("nodes"),
                    Some(
                        "model explicit NIC, rail, rack, and fabric resources before treating this as exact locality",
                    ),
                ),
            );
        }

        evidence
    }

    fn placement_capacity_evidence(
        total_ranks: u32,
        available_gpus: u32,
        reason: &str,
    ) -> PlacementEvidence {
        PlacementEvidence::new(
            "rejected",
            "rank_placement",
            "available_gpus",
            "insufficient_available_gpus",
            reason,
        )
        .with_details(
            Some(f64::from(available_gpus)),
            Some(f64::from(total_ranks)),
            Some("gpus"),
            Some("reduce total ranks, add GPUs, or re-enable unavailable GPUs"),
        )
    }

    fn explicit_placement_rejection(
        resource: String,
        code: &str,
        message: String,
        observed: Option<f64>,
        limit: Option<f64>,
        unit: Option<&str>,
        remediation: &str,
    ) -> PlacementEvidence {
        PlacementEvidence::new("rejected", "rank_placement", resource, code, message).with_details(
            observed,
            limit,
            unit,
            Some(remediation),
        )
    }

    fn place_explicit_ranks(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        placement: &RankPlacement,
        required_memory_per_gpu: Bytes,
    ) -> Result<RankPlacement, Box<PlacementEvidence>> {
        let total_ranks = config.total_ranks() as usize;
        if placement.rank_to_gpu.len() != total_ranks {
            let observed = placement.rank_to_gpu.len();
            let message = format!(
                "explicit placement has {observed} ranks but config requires {total_ranks} ranks"
            );
            return Err(Box::new(Self::explicit_placement_rejection(
                "rank_count".to_string(),
                "explicit_placement_rank_count_mismatch",
                message,
                Some(observed as f64),
                Some(total_ranks as f64),
                Some("ranks"),
                "match placement.ranks length to tensor*pipeline*expert*data ranks",
            )));
        }

        let mut used_gpus = BTreeSet::new();
        for (rank, addr) in placement.rank_to_gpu.iter().copied().enumerate() {
            let resource = format!(
                "rank {rank} node {} gpu {}",
                addr.node_id, addr.local_gpu_id
            );
            let Some(profile) = cluster.gpu_profile(addr) else {
                let message = format!(
                    "explicit placement rank {rank} references unknown node {} gpu {}",
                    addr.node_id, addr.local_gpu_id
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_unknown_gpu",
                    message,
                    None,
                    None,
                    None,
                    "choose a GPU that exists in the cluster inventory",
                )));
            };
            if !cluster.is_gpu_available(addr) {
                let message = format!(
                    "explicit placement rank {rank} references unavailable node {} gpu {}",
                    addr.node_id, addr.local_gpu_id
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_gpu_unavailable",
                    message,
                    None,
                    None,
                    None,
                    "choose an available GPU or remove the disabled/maintenance overlay",
                )));
            }
            if !Self::gpu_supports_model_dtype(&profile, model) {
                let message = format!(
                    "explicit placement rank {rank} uses node {} gpu {} ({}) which does not support model.dtype {}",
                    addr.node_id,
                    addr.local_gpu_id,
                    profile.label,
                    Self::model_dtype_label(model)
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_dtype_unsupported",
                    message,
                    None,
                    None,
                    None,
                    "choose GPUs whose tensor throughput metadata supports the model dtype",
                )));
            }
            if !used_gpus.insert(addr) {
                let message = format!(
                    "explicit placement maps more than one rank to node {} gpu {}",
                    addr.node_id, addr.local_gpu_id
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_duplicate_gpu",
                    message,
                    None,
                    None,
                    None,
                    "map each rank to a unique GPU in the current rank-placement model",
                )));
            }
            if required_memory_per_gpu > profile.hbm_size {
                let message = format!(
                    "explicit placement rank {rank} requires {:.2} GB but node {} gpu {} has {:.2} GB HBM",
                    required_memory_per_gpu.as_gigabytes(),
                    addr.node_id,
                    addr.local_gpu_id,
                    profile.hbm_size.as_gigabytes()
                );
                return Err(Box::new(Self::explicit_placement_rejection(
                    resource,
                    "explicit_placement_hbm_exceeded",
                    message,
                    Some(required_memory_per_gpu.as_gigabytes()),
                    Some(profile.hbm_size.as_gigabytes()),
                    Some("GB"),
                    "reduce per-rank memory or choose a larger-HBM GPU",
                )));
            }
        }

        Ok(placement.clone())
    }

    fn placement_hbm_evidence(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        required_memory_per_gpu: Bytes,
        reason: &str,
    ) -> PlacementEvidence {
        let eligible_gpus =
            Self::sorted_eligible_gpus(cluster, model, required_memory_per_gpu).len() as u32;
        PlacementEvidence::new(
            "rejected",
            "gpu_eligibility",
            "eligible_gpus",
            "insufficient_hbm_eligible_gpus",
            reason,
        )
        .with_details(
            Some(f64::from(eligible_gpus)),
            Some(f64::from(config.total_ranks())),
            Some("gpus"),
            Some("reduce memory per rank, increase tensor/pipeline sharding, or choose larger-HBM GPUs"),
        )
    }

    fn placement_memory_rejection_evidence(
        required_memory_per_gpu: Bytes,
        reason: &str,
    ) -> PlacementEvidence {
        PlacementEvidence::new(
            "rejected",
            "memory",
            "gpu_hbm",
            "placement_hbm_headroom_exceeded",
            reason,
        )
        .with_details(
            Some(required_memory_per_gpu.as_gigabytes()),
            None,
            Some("GB"),
            Some("reduce model/request memory or place ranks on GPUs with more HBM"),
        )
    }

    fn parallelism_approximations(
        cluster: &Cluster,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        operations: &[SimOperation],
        calibration_fits: &[CalibrationFitApplication],
    ) -> Vec<SimulationApproximation> {
        let mut approximations = Vec::new();
        let phase = phase_label(request.phase);
        push_approximation(
            &mut approximations,
            SimulationApproximation::new(
                phase,
                "placement",
                "rank_placement",
                "capability_ordered_rank_placement",
                "Rank placement is deterministic and capability-aware, but it is not a full topology/NUMA/rail-aware placement search.",
                Some(
                    "supply explicit placement constraints or add topology-aware placement search before relying on fine-grained locality conclusions"
                        .to_string(),
                ),
            ),
        );
        push_approximation(
            &mut approximations,
            SimulationApproximation::new(
                phase,
                "memory",
                "per_gpu_hbm",
                "static_per_gpu_memory_estimate",
                "Memory is estimated as a static per-GPU requirement and does not model allocator timelines, graph capture reserve, KV-block fragmentation, or transient peaks.",
                Some(
                    "calibrate memory components and add placement/time-aware memory accounting for production capacity planning"
                        .to_string(),
                ),
            ),
        );

        if config.total_ranks() > 1
            && operations
                .iter()
                .any(|op| op.kind == SimOperationKind::Collective)
        {
            push_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    phase,
                    "communication",
                    "collectives",
                    "coarse_collective_model",
                    "Collective operations use calibrated coarse formulas and do not fully model backend algorithm auto-selection, channels, protocol thresholds, or concurrent traffic contention.",
                    Some(
                        "fit collective behavior from backend benchmarks and add algorithm/route contention modeling for topology-sensitive decisions"
                            .to_string(),
                    ),
                ),
            );
        }

        if placement_spans_nodes(placement) {
            let (code, message) = match &cluster.inter_node_topology {
                InterNodeTopology::Custom(_) => (
                    "custom_topology_route_approximation",
                    "Custom inter-node links are routed through coarse path bottlenecks and do not yet model per-NIC/per-rail/per-switch contention.",
                ),
                InterNodeTopology::FatTree { .. } | InterNodeTopology::Flat { .. } => (
                    "aggregate_inter_node_topology",
                    "Inter-node topology is represented by aggregate fabric parameters rather than explicit rack switches, rails, NICs, and oversubscribed uplinks.",
                ),
            };
            push_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    phase,
                    "topology",
                    "inter_node",
                    code,
                    message,
                    Some(
                        "model explicit NICs, rails, switches, and route sharing before making fine-grained fabric locality claims"
                            .to_string(),
                    ),
                ),
            );
        }

        if calibration_fits.iter().any(|fit| {
            fit.max_extrapolation_ratio > 0.0 || fit.applicability_status != "interpolated"
        }) {
            push_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    phase,
                    "calibration",
                    "fitted_models",
                    "calibration_fit_extrapolation",
                    "At least one calibration fit is outside its fitted feature range or is not marked interpolated, so ranking may be sensitive to extrapolation error.",
                    Some(
                        "add benchmark coverage for this workload shape or hard-reject extrapolated fits via calibration policy gates"
                            .to_string(),
                    ),
                ),
            );
        }

        approximations
    }

    fn place_ranks(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        required_memory_per_gpu: Bytes,
    ) -> Result<RankPlacement, String> {
        let total_ranks = config.total_ranks();
        let eligible_gpus = Self::sorted_eligible_gpus(cluster, model, required_memory_per_gpu);
        let total_gpus = cluster.available_gpus();

        if total_ranks > total_gpus {
            return Err(format!(
                "config requires {total_ranks} ranks but only {} GPUs are available",
                total_gpus
            ));
        }
        if total_ranks as usize > eligible_gpus.len() {
            return Err(format!(
                "estimated {:.2} GB per GPU and model.dtype {} leave {} of {} GPUs ineligible; config requires {total_ranks} ranks but only {} GPUs have enough HBM and dtype support",
                required_memory_per_gpu.as_gigabytes(),
                Self::model_dtype_label(model),
                total_gpus.saturating_sub(eligible_gpus.len() as u32),
                total_gpus,
                eligible_gpus.len()
            ));
        }

        if let Some(placement) =
            Self::place_by_data_replicas(cluster, model, config, required_memory_per_gpu)
        {
            return Ok(placement);
        }

        Ok(RankPlacement {
            rank_to_gpu: eligible_gpus
                .into_iter()
                .take(total_ranks as usize)
                .collect(),
        })
    }

    fn place_by_data_replicas(
        cluster: &Cluster,
        model: &ModelSpec,
        config: ParallelismConfig,
        required_memory_per_gpu: Bytes,
    ) -> Option<RankPlacement> {
        let ranks_per_replica = config.tensor_ranks * config.pipeline_ranks * config.expert_ranks;
        let mut node_gpus = Self::eligible_node_gpu_candidates(
            cluster,
            model,
            required_memory_per_gpu,
            ranks_per_replica as usize,
        );

        if node_gpus.len() < config.data_ranks as usize {
            return None;
        }
        node_gpus.truncate(config.data_ranks as usize);

        let mut rank_to_gpu = Vec::with_capacity(config.total_ranks() as usize);
        let mut used = HashSet::new();
        for rank in 0..config.total_ranks() {
            let (dp, pp, ep, tp) = decode_rank(config, rank);
            let (node_id, gpus) = &node_gpus[dp as usize];
            let local_index =
                (pp * config.expert_ranks * config.tensor_ranks) + (ep * config.tensor_ranks) + tp;
            let addr = GpuAddr {
                node_id: *node_id,
                local_gpu_id: gpus[local_index as usize],
            };
            if !used.insert(addr) {
                return None;
            }
            rank_to_gpu.push(addr);
        }

        Some(RankPlacement { rank_to_gpu })
    }

    fn sorted_eligible_gpus(
        cluster: &Cluster,
        model: &ModelSpec,
        required_memory_per_gpu: Bytes,
    ) -> Vec<GpuAddr> {
        let mut gpus: Vec<_> = cluster
            .sorted_gpu_addrs()
            .into_iter()
            .filter(|addr| {
                cluster.is_gpu_available(*addr)
                    && cluster.gpu_profile(*addr).is_some_and(|profile| {
                        profile.hbm_size >= required_memory_per_gpu
                            && Self::gpu_supports_model_dtype(&profile, model)
                    })
            })
            .collect();
        gpus.sort_by(|left, right| Self::compare_gpu_addrs(cluster, model, left, right));
        gpus
    }

    fn eligible_node_gpu_candidates(
        cluster: &Cluster,
        model: &ModelSpec,
        required_memory_per_gpu: Bytes,
        min_gpus_per_node: usize,
    ) -> Vec<(NodeId, Vec<u32>)> {
        let mut candidates: Vec<(NodeId, Vec<u32>)> = Vec::new();
        let mut node_ids: Vec<_> = cluster.nodes.keys().copied().collect();
        node_ids.sort_unstable();

        for node_id in node_ids {
            let mut gpus: Vec<_> = cluster
                .node(node_id)
                .into_iter()
                .flat_map(|node| node.gpus.keys().copied())
                .map(|local_gpu_id| GpuAddr {
                    node_id,
                    local_gpu_id,
                })
                .filter(|addr| {
                    cluster.is_gpu_available(*addr)
                        && cluster.gpu_profile(*addr).is_some_and(|profile| {
                            profile.hbm_size >= required_memory_per_gpu
                                && Self::gpu_supports_model_dtype(&profile, model)
                        })
                })
                .collect();
            gpus.sort_by(|left, right| Self::compare_gpu_addrs(cluster, model, left, right));
            if gpus.len() >= min_gpus_per_node {
                candidates.push((
                    node_id,
                    gpus.into_iter().map(|addr| addr.local_gpu_id).collect(),
                ));
            }
        }

        candidates.sort_by(|left, right| {
            let left_addr = GpuAddr {
                node_id: left.0,
                local_gpu_id: left.1[0],
            };
            let right_addr = GpuAddr {
                node_id: right.0,
                local_gpu_id: right.1[0],
            };
            Self::compare_gpu_addrs(cluster, model, &left_addr, &right_addr)
        });
        candidates
    }

    fn compare_gpu_addrs(
        cluster: &Cluster,
        model: &ModelSpec,
        left: &GpuAddr,
        right: &GpuAddr,
    ) -> std::cmp::Ordering {
        let left_profile = cluster.gpu_profile(*left);
        let right_profile = cluster.gpu_profile(*right);
        match (left_profile, right_profile) {
            (Some(left_profile), Some(right_profile)) => {
                Self::gpu_peak_tflops(&right_profile, model)
                    .total_cmp(&Self::gpu_peak_tflops(&left_profile, model))
                    .then_with(|| {
                        right_profile
                            .hbm_bandwidth
                            .as_bytes_per_sec()
                            .total_cmp(&left_profile.hbm_bandwidth.as_bytes_per_sec())
                    })
                    .then_with(|| {
                        right_profile
                            .hbm_size
                            .as_bytes()
                            .cmp(&left_profile.hbm_size.as_bytes())
                    })
                    .then_with(|| left.node_id.cmp(&right.node_id))
                    .then_with(|| left.local_gpu_id.cmp(&right.local_gpu_id))
            }
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left
                .node_id
                .cmp(&right.node_id)
                .then_with(|| left.local_gpu_id.cmp(&right.local_gpu_id)),
        }
    }

    fn gpu_peak_tflops(profile: &crate::types::gpu::GpuProfile, model: &ModelSpec) -> f64 {
        match model.dtype {
            crate::workload::DType::Fp8 => profile.peak_f8_flops.unwrap_or(0.0),
            crate::workload::DType::Int8 => profile.peak_f8_flops.unwrap_or(profile.peak_f16_flops),
            crate::workload::DType::Fp16 | crate::workload::DType::Bf16 => profile.peak_f16_flops,
        }
    }

    fn gpu_supports_model_dtype(
        profile: &crate::types::gpu::GpuProfile,
        model: &ModelSpec,
    ) -> bool {
        match model.dtype {
            crate::workload::DType::Fp8 => profile.peak_f8_flops.is_some(),
            crate::workload::DType::Fp16
            | crate::workload::DType::Bf16
            | crate::workload::DType::Int8 => true,
        }
    }

    fn model_dtype_label(model: &ModelSpec) -> &'static str {
        model.dtype.label()
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

    #[cfg(test)]
    fn build_collectives(
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        groups: &ParallelGroups,
    ) -> Vec<CollectiveCall> {
        let mut calls = Vec::new();
        let activation_bytes = Self::activation_bytes(model, request);
        let tp_bytes = Bytes::from_bytes(div_ceil(
            activation_bytes.as_bytes(),
            config.tensor_ranks.max(1) as u64,
        ));

        if config.tensor_ranks > 1 {
            for _ in 0..model.layers {
                for group in &groups.tensor_groups {
                    calls.push(CollectiveCall {
                        kind: CollectiveKind::AllReduce,
                        participants: group.clone(),
                        bytes_per_rank: tp_bytes,
                        dtype: model.dtype,
                        reduction: Some(ReductionOp::Sum),
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Hierarchical,
                    });
                }
            }
        }

        if config.pipeline_ranks > 1 {
            for window in groups.pipeline_stages.windows(2) {
                if let [left, right] = window
                    && let (Some(&left_rank), Some(&right_rank)) = (left.first(), right.first())
                {
                    calls.push(CollectiveCall {
                        kind: CollectiveKind::SendRecv,
                        participants: vec![left_rank, right_rank],
                        bytes_per_rank: activation_bytes,
                        dtype: model.dtype,
                        reduction: None,
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Auto,
                    });
                }
            }
        }

        if let Some(experts) = model.experts
            && config.expert_ranks > 1
        {
            let expert_bytes = Bytes::from_bytes(
                activation_bytes.as_bytes() * experts.top_k as u64
                    / config.expert_ranks.max(1) as u64,
            );
            for _ in 0..model.layers {
                for group in &groups.expert_groups {
                    calls.push(CollectiveCall {
                        kind: CollectiveKind::AllToAll,
                        participants: group.clone(),
                        bytes_per_rank: expert_bytes,
                        dtype: model.dtype,
                        reduction: None,
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Hierarchical,
                    });
                }
            }
        }

        calls
    }

    #[allow(clippy::too_many_arguments)]
    fn build_operation_trace(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        placement: &RankPlacement,
        groups: &ParallelGroups,
        compute_latency_s: f64,
        calibration: SimulationCalibration,
    ) -> Vec<SimOperation> {
        let mut operations = Vec::new();
        let layers = model.layers.max(1);
        let compute_per_layer_s = compute_latency_s / f64::from(layers);
        let compute_resources = compute_resources(placement, request.phase);
        let mut previous_compute = None;
        let mut previous_layer_barrier = None;

        for layer_idx in 0..layers {
            let mut compute_dependencies = Vec::new();
            if let Some(previous_compute) = previous_compute {
                compute_dependencies.push(previous_compute);
            }
            if !calibration.allow_compute_comm_overlap
                && let Some(previous_layer_barrier) = previous_layer_barrier
            {
                compute_dependencies.push(previous_layer_barrier);
            }

            let compute_idx = operations.len();
            operations.push(SimOperation {
                name: format!("layer {layer_idx} compute"),
                kind: SimOperationKind::Compute,
                duration_s: compute_per_layer_s,
                resources: compute_resources.clone(),
                dependencies: compute_dependencies,
            });
            previous_compute = Some(compute_idx);

            let mut layer_tail = compute_idx;
            if config.tensor_ranks > 1 {
                for (group_idx, group) in groups.tensor_groups.iter().enumerate() {
                    let collective = Self::collective_call(
                        CollectiveKind::AllReduce,
                        group.clone(),
                        model,
                        request,
                        config,
                        Some(ReductionOp::Sum),
                    );
                    let op_idx = Self::push_collective_operation(
                        cluster,
                        placement,
                        &mut operations,
                        format!("layer {layer_idx} tp all-reduce group {group_idx}"),
                        collective,
                        compute_idx,
                        calibration,
                    );
                    layer_tail = op_idx;
                }
            }

            if model.experts.is_some() && config.expert_ranks > 1 {
                for (group_idx, group) in groups.expert_groups.iter().enumerate() {
                    let collective = Self::collective_call(
                        CollectiveKind::AllToAll,
                        group.clone(),
                        model,
                        request,
                        config,
                        None,
                    );
                    let op_idx = Self::push_collective_operation(
                        cluster,
                        placement,
                        &mut operations,
                        format!("layer {layer_idx} ep all-to-all group {group_idx}"),
                        collective,
                        compute_idx,
                        calibration,
                    );
                    layer_tail = op_idx;
                }
            }

            previous_layer_barrier = Some(layer_tail);
        }

        if config.pipeline_ranks > 1 {
            let activation_bytes = Self::activation_bytes(model, request);
            for (edge_idx, window) in groups.pipeline_stages.windows(2).enumerate() {
                if let [left, right] = window
                    && let (Some(&left_rank), Some(&right_rank)) = (left.first(), right.first())
                {
                    let collective = CollectiveCall {
                        kind: CollectiveKind::SendRecv,
                        participants: vec![left_rank, right_rank],
                        bytes_per_rank: activation_bytes,
                        dtype: model.dtype,
                        reduction: None,
                        root: None,
                        phase: request.phase,
                        algorithm: CollectiveAlgorithm::Auto,
                    };
                    let dependency = previous_layer_barrier.or(previous_compute).unwrap_or(0);
                    Self::push_collective_operation(
                        cluster,
                        placement,
                        &mut operations,
                        format!("pipeline sendrecv edge {edge_idx}"),
                        collective,
                        dependency,
                        calibration,
                    );
                }
            }
        }

        operations
    }

    fn collective_call(
        kind: CollectiveKind,
        participants: Vec<RankId>,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        reduction: Option<ReductionOp>,
    ) -> CollectiveCall {
        let activation_bytes = Self::activation_bytes(model, request);
        let bytes_per_rank = match kind {
            CollectiveKind::AllReduce => Bytes::from_bytes(div_ceil(
                activation_bytes.as_bytes(),
                config.tensor_ranks.max(1) as u64,
            )),
            CollectiveKind::AllToAll => {
                let top_k = model.experts.map(|experts| experts.top_k).unwrap_or(1);
                Bytes::from_bytes(
                    activation_bytes.as_bytes() * top_k as u64 / config.expert_ranks.max(1) as u64,
                )
            }
            _ => activation_bytes,
        };

        CollectiveCall {
            kind,
            participants,
            bytes_per_rank,
            dtype: model.dtype,
            reduction,
            root: None,
            phase: request.phase,
            algorithm: CollectiveAlgorithm::Hierarchical,
        }
    }

    fn push_collective_operation(
        cluster: &Cluster,
        placement: &RankPlacement,
        operations: &mut Vec<SimOperation>,
        name: String,
        collective: CollectiveCall,
        dependency: usize,
        calibration: SimulationCalibration,
    ) -> usize {
        let cost = Self::estimate_collective_with_calibration(
            cluster,
            placement,
            &collective,
            calibration,
        );
        let resources = if cost.bottlenecks.is_empty() {
            vec!["communication".to_string()]
        } else {
            cost.bottlenecks
        };
        let op_idx = operations.len();
        operations.push(SimOperation {
            name,
            kind: SimOperationKind::Collective,
            duration_s: cost.total_s,
            resources,
            dependencies: vec![dependency],
        });
        op_idx
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
                let active_tokens = request.prompt_tokens as f64 * request.batch_size as f64;
                let baseline =
                    Self::flop_latency_s(parameter_count, active_tokens, shard_factor, peak_flops)
                        * calibration.prefill_compute_scale;
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
                let prefill_tokens = request.prompt_tokens as f64 * request.batch_size as f64;
                let prefill_baseline =
                    Self::flop_latency_s(parameter_count, prefill_tokens, shard_factor, peak_flops)
                        * calibration.prefill_compute_scale;
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
        let flop_latency_s =
            Self::flop_latency_s(parameter_count, flop_tokens, shard_factor, peak_flops);
        let parameter_bytes_per_rank = model.parameters.as_bytes() as f64 / shard_factor;
        let hbm_bandwidth = Self::effective_hbm_bandwidth(cluster, placement, calibration);
        let memory_latency_s = parameter_bytes_per_rank * decode_tokens / hbm_bandwidth;

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

    pub fn estimate_collective(
        cluster: &Cluster,
        placement: &RankPlacement,
        collective: &CollectiveCall,
    ) -> CollectiveCost {
        Self::estimate_collective_with_calibration(
            cluster,
            placement,
            collective,
            SimulationCalibration::default(),
        )
    }

    pub fn estimate_collective_with_calibration(
        cluster: &Cluster,
        placement: &RankPlacement,
        collective: &CollectiveCall,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        let calibration = calibration.sanitized();
        if collective.participants.len() <= 1 || collective.bytes_per_rank.as_bytes() == 0 {
            return CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            };
        }

        let participant_addrs: Vec<_> = collective
            .participants
            .iter()
            .filter_map(|rank| placement.gpu_for_rank(*rank))
            .collect();
        let nodes: BTreeSet<NodeId> = participant_addrs.iter().map(|addr| addr.node_id).collect();
        let rank_count = participant_addrs.len().max(1) as f64;
        let bytes = collective.bytes_per_rank.as_bytes() as f64;

        let mut latency_s = 0.0;
        let mut bandwidth_s = 0.0;
        let mut bottlenecks = Vec::new();

        if nodes.len() <= 1 {
            let node_id = participant_addrs
                .first()
                .map(|addr| addr.node_id)
                .unwrap_or(0);
            let (latency, bandwidth) = Self::intra_node_profile(cluster, node_id);
            let steps = Self::collective_steps(collective.kind, rank_count);
            let traffic_multiplier = Self::traffic_multiplier(collective.kind, rank_count);

            latency_s += steps * latency;
            bandwidth_s += bytes * traffic_multiplier / bandwidth.as_bytes_per_sec();
            bottlenecks.push(format!("node {node_id} intra-node fabric"));
        } else {
            let steps = Self::collective_steps(collective.kind, rank_count);
            let traffic_multiplier = Self::traffic_multiplier(collective.kind, rank_count);
            if let Some(cost) = Self::estimate_graph_collective_cost(
                cluster,
                &participant_addrs,
                Bytes::from_bytes((bytes * traffic_multiplier).ceil() as u64),
                steps,
            ) {
                latency_s += cost.latency_s;
                bandwidth_s += cost.bandwidth_s;
                bottlenecks.extend(cost.bottlenecks);
            } else {
                let (intra_latency, intra_bandwidth) =
                    Self::intra_node_profile(cluster, *nodes.first().unwrap());
                let (fabric_latency, fabric_bandwidth) =
                    Self::inter_node_profile_for_nodes(cluster, &nodes);
                let nic_bandwidth = Self::effective_nic_bandwidth(cluster, &participant_addrs);
                let effective_inter_bandwidth = fabric_bandwidth
                    .as_bytes_per_sec()
                    .min(nic_bandwidth.as_bytes_per_sec());

                latency_s += steps * (intra_latency + fabric_latency);
                bandwidth_s += bytes * traffic_multiplier / intra_bandwidth.as_bytes_per_sec();
                bandwidth_s += bytes * traffic_multiplier / effective_inter_bandwidth;
                bottlenecks.push("inter-node fabric".to_string());
                bottlenecks.push("node NIC bandwidth".to_string());
            }
        }

        latency_s *= calibration.collective_latency_scale;
        bandwidth_s /= calibration.collective_bandwidth_scale;

        CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks,
        }
    }

    pub fn estimate_transfer_between_nodes(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        Self::estimate_transfer_between_nodes_with_options(
            cluster,
            source_nodes,
            dest_nodes,
            bytes,
            SolverOptions {
                calibration,
                calibration_profile: None,
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        )
    }

    pub fn estimate_transfer_between_nodes_with_options(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> CollectiveCost {
        Self::estimate_transfer_between_nodes_with_observation(
            cluster,
            source_nodes,
            dest_nodes,
            bytes,
            options,
        )
        .0
    }

    pub fn estimate_transfer_between_nodes_with_observation(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> (CollectiveCost, Option<CalibrationFitApplication>) {
        let mut cost = Self::estimate_transfer_between_nodes_baseline(
            cluster,
            source_nodes,
            dest_nodes,
            bytes,
            options.calibration,
        );
        if bytes.as_bytes() > 0
            && let Some(evaluation) =
                Self::fitted_kv_transfer_latency(options.calibration_profile, bytes, &cost)
        {
            cost.latency_s = cost.latency_s.min(evaluation.seconds);
            cost.bandwidth_s = (evaluation.seconds - cost.latency_s).max(0.0);
            cost.total_s = evaluation.seconds;
            return (cost, Some(evaluation.application));
        }
        (cost, None)
    }

    pub fn estimate_transfer_between_gpus_with_options(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> CollectiveCost {
        Self::estimate_transfer_between_gpus_with_observation(
            cluster,
            source_gpus,
            dest_gpus,
            bytes,
            options,
        )
        .0
    }

    pub fn estimate_transfer_between_gpus_with_observation(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
        options: SolverOptions<'_>,
    ) -> (CollectiveCost, Option<CalibrationFitApplication>) {
        let mut cost = Self::estimate_transfer_between_gpus_baseline(
            cluster,
            source_gpus,
            dest_gpus,
            bytes,
            options.calibration,
        );
        if bytes.as_bytes() > 0
            && let Some(evaluation) =
                Self::fitted_kv_transfer_latency(options.calibration_profile, bytes, &cost)
        {
            cost.latency_s = cost.latency_s.min(evaluation.seconds);
            cost.bandwidth_s = (evaluation.seconds - cost.latency_s).max(0.0);
            cost.total_s = evaluation.seconds;
            return (cost, Some(evaluation.application));
        }
        (cost, None)
    }

    fn estimate_transfer_between_nodes_baseline(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        let calibration = calibration.sanitized();
        let nodes: BTreeSet<_> = source_nodes
            .iter()
            .chain(dest_nodes.iter())
            .copied()
            .collect();
        if bytes.as_bytes() == 0 || source_nodes.is_empty() || dest_nodes.is_empty() {
            return CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            };
        }

        if let Some(mut cost) =
            Self::estimate_graph_node_transfer_cost(cluster, source_nodes, dest_nodes, bytes)
        {
            cost.latency_s *= calibration.collective_latency_scale;
            cost.bandwidth_s = cost.bandwidth_s * calibration.kv_transfer_scale
                / calibration.collective_bandwidth_scale;
            cost.total_s = cost.latency_s + cost.bandwidth_s;
            return cost;
        }

        if matches!(&cluster.inter_node_topology, InterNodeTopology::Custom(_)) {
            return CollectiveCost {
                latency_s: f64::INFINITY,
                bandwidth_s: f64::INFINITY,
                total_s: f64::INFINITY,
                bottlenecks: vec!["KV transfer route unavailable".to_string()],
            };
        }

        let (fabric_latency, fabric_bandwidth) =
            Self::inter_node_profile_for_nodes(cluster, &nodes);
        let mut nic_bytes_per_sec = 0.0_f64;
        for node_id in &nodes {
            if let Some(node) = cluster.node(*node_id) {
                nic_bytes_per_sec += node.network.active_nic_bandwidth_bytes_per_sec();
            }
        }
        let effective_bandwidth = fabric_bandwidth
            .as_bytes_per_sec()
            .min(nic_bytes_per_sec.max(1.0));
        let latency_s = fabric_latency * calibration.collective_latency_scale;
        let bandwidth_s = (bytes.as_bytes() as f64 / effective_bandwidth)
            * calibration.kv_transfer_scale
            / calibration.collective_bandwidth_scale;

        CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks: vec!["KV transfer fabric/NIC path".to_string()],
        }
    }

    fn estimate_transfer_between_gpus_baseline(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
        calibration: SimulationCalibration,
    ) -> CollectiveCost {
        let calibration = calibration.sanitized();
        if bytes.as_bytes() == 0 || source_gpus.is_empty() || dest_gpus.is_empty() {
            return CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            };
        }

        let Some(mut cost) =
            Self::estimate_graph_gpu_transfer_cost(cluster, source_gpus, dest_gpus, bytes)
        else {
            return CollectiveCost {
                latency_s: f64::INFINITY,
                bandwidth_s: f64::INFINITY,
                total_s: f64::INFINITY,
                bottlenecks: vec!["KV transfer route unavailable".to_string()],
            };
        };
        cost.latency_s *= calibration.collective_latency_scale;
        cost.bandwidth_s = cost.bandwidth_s * calibration.kv_transfer_scale
            / calibration.collective_bandwidth_scale;
        cost.total_s = cost.latency_s + cost.bandwidth_s;
        cost
    }

    #[allow(clippy::too_many_arguments)]
    fn fitted_phase_latency(
        profile: Option<&CalibrationProfileMetadata>,
        phase: &str,
        targets: &[&str],
        model: &ModelSpec,
        request: &InferenceRequest,
        config: ParallelismConfig,
        extra_features: Option<&BTreeMap<String, f64>>,
        baseline_s: Option<f64>,
    ) -> Option<FitEvaluation> {
        let profile = profile?;
        let features = calibration_feature_values(model, request, config, extra_features);
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(fit, phase, targets) {
                return None;
            }
            evaluate_fit(fit, phase, &features, baseline_s)
        })
    }

    pub(crate) fn fitted_latency_from_features(
        profile: Option<&CalibrationProfileMetadata>,
        phase: &str,
        targets: &[&str],
        features: &BTreeMap<String, f64>,
        baseline_s: Option<f64>,
    ) -> Option<CalibrationFitApplication> {
        let profile = profile?;
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(fit, phase, targets) {
                return None;
            }
            evaluate_fit(fit, phase, features, baseline_s).map(|evaluation| evaluation.application)
        })
    }

    pub(crate) fn fitted_value_from_features(
        profile: Option<&CalibrationProfileMetadata>,
        phase: &str,
        targets: &[&str],
        features: &BTreeMap<String, f64>,
        baseline_value: Option<f64>,
        prediction_kind: &str,
        prediction_unit: Option<&str>,
    ) -> Option<CalibrationFitApplication> {
        let profile = profile?;
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(fit, phase, targets) {
                return None;
            }
            evaluate_value_fit(
                fit,
                phase,
                features,
                baseline_value,
                prediction_kind,
                prediction_unit,
            )
        })
    }

    fn fitted_kv_transfer_latency(
        profile: Option<&CalibrationProfileMetadata>,
        bytes: Bytes,
        cost: &CollectiveCost,
    ) -> Option<FitEvaluation> {
        let mut features = BTreeMap::new();
        let bytes = bytes.as_bytes() as f64;
        features.insert("kv_transfer_bytes".to_string(), bytes);
        features.insert("transfer_bytes".to_string(), bytes);
        features.insert("kv_transfer_mb".to_string(), bytes / 1e6);
        features.insert("kv_transfer_gb".to_string(), bytes / 1e9);
        features.insert("transfer_mb".to_string(), bytes / 1e6);
        features.insert("transfer_gb".to_string(), bytes / 1e9);
        features.insert("latency_ms".to_string(), cost.latency_s * 1000.0);
        features.insert("bandwidth_ms".to_string(), cost.bandwidth_s * 1000.0);
        features.insert("baseline_ms".to_string(), cost.total_s * 1000.0);
        features.insert("hop_count".to_string(), cost.bottlenecks.len() as f64);
        let effective_gbps =
            if bytes > 0.0 && cost.bandwidth_s.is_finite() && cost.bandwidth_s > 0.0 {
                bytes * 8.0 / cost.bandwidth_s / 1e9
            } else {
                0.0
            };
        features.insert("rail_bandwidth_gbps".to_string(), effective_gbps);
        features.insert("effective_bandwidth_gbps".to_string(), effective_gbps);

        let profile = profile?;
        profile.fits.iter().find_map(|fit| {
            if !fit_matches(
                fit,
                "kv_transfer",
                &[
                    "kv_transfer_ms",
                    "kv_transfer_latency_ms",
                    "transfer_ms",
                    "transfer_latency_ms",
                    "kv_transfer_s",
                ],
            ) {
                return None;
            }
            evaluate_fit(fit, "kv_transfer", &features, Some(cost.total_s))
        })
    }

    pub fn transfer_between_nodes_routable(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
    ) -> bool {
        if bytes.as_bytes() == 0 || source_nodes.is_empty() || dest_nodes.is_empty() {
            return true;
        }
        match &cluster.inter_node_topology {
            InterNodeTopology::FatTree { .. } | InterNodeTopology::Flat { .. } => true,
            InterNodeTopology::Custom(_) => {
                Self::estimate_graph_node_transfer_cost(cluster, source_nodes, dest_nodes, bytes)
                    .is_some()
            }
        }
    }

    pub fn transfer_between_gpus_routable(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
    ) -> bool {
        if bytes.as_bytes() == 0 || source_gpus.is_empty() || dest_gpus.is_empty() {
            return true;
        }
        Self::estimate_graph_gpu_transfer_cost(cluster, source_gpus, dest_gpus, bytes).is_some()
    }

    fn estimate_graph_collective_cost(
        cluster: &Cluster,
        participant_addrs: &[GpuAddr],
        route_bytes: Bytes,
        latency_steps: f64,
    ) -> Option<CollectiveCost> {
        let graph = TopologyGraph::from_cluster(cluster);
        let mut paths = Vec::new();

        for i in 0..participant_addrs.len() {
            for j in i + 1..participant_addrs.len() {
                let src = participant_addrs[i];
                let dst = participant_addrs[j];
                if src.node_id == dst.node_id {
                    continue;
                }
                paths.push(graph.route_between_gpus(src, dst, route_bytes)?);
            }
        }

        if paths.is_empty() {
            return None;
        }

        let bytes_per_path =
            Bytes::from_bytes(div_ceil(route_bytes.as_bytes(), paths.len() as u64));
        let (latency_s, bandwidth_s, bottlenecks) =
            Self::contended_paths_cost(&paths, bytes_per_path);

        Some(CollectiveCost {
            latency_s: latency_steps * latency_s,
            bandwidth_s,
            total_s: latency_steps * latency_s + bandwidth_s,
            bottlenecks,
        })
    }

    fn estimate_graph_node_transfer_cost(
        cluster: &Cluster,
        source_nodes: &[NodeId],
        dest_nodes: &[NodeId],
        bytes: Bytes,
    ) -> Option<CollectiveCost> {
        let graph = TopologyGraph::from_cluster(cluster);
        let mut paths = Vec::new();

        for src in source_nodes {
            for dst in dest_nodes {
                if src == dst {
                    continue;
                }
                paths.push(graph.route_between_nodes(*src, *dst, bytes)?);
            }
        }

        if paths.is_empty() {
            return Some(CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            });
        }

        let bytes_per_path = Bytes::from_bytes(div_ceil(bytes.as_bytes(), paths.len() as u64));
        let (latency_s, bandwidth_s, mut bottlenecks) =
            Self::contended_paths_cost(&paths, bytes_per_path);
        if !bottlenecks
            .iter()
            .any(|bottleneck| bottleneck == "KV transfer fabric/NIC path")
        {
            bottlenecks.push("KV transfer fabric/NIC path".to_string());
        }

        Some(CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks,
        })
    }

    fn estimate_graph_gpu_transfer_cost(
        cluster: &Cluster,
        source_gpus: &[GpuAddr],
        dest_gpus: &[GpuAddr],
        bytes: Bytes,
    ) -> Option<CollectiveCost> {
        let graph = TopologyGraph::from_cluster(cluster);
        let mut paths = Vec::new();
        let mut has_inter_node_path = false;

        for src in source_gpus {
            for dst in dest_gpus {
                if src == dst {
                    continue;
                }
                if src.node_id == dst.node_id {
                    paths.push(Self::intra_node_gpu_path(cluster, src.node_id));
                } else {
                    has_inter_node_path = true;
                    let path = graph.route_between_gpus(*src, *dst, bytes).or_else(|| {
                        if Self::can_use_node_level_gpu_path(cluster, *src)
                            && Self::can_use_node_level_gpu_path(cluster, *dst)
                        {
                            graph
                                .route_between_nodes(src.node_id, dst.node_id, bytes)
                                .map(|path| {
                                    Self::add_endpoint_intra_node_resources(
                                        cluster,
                                        path,
                                        src.node_id,
                                        dst.node_id,
                                    )
                                })
                        } else {
                            None
                        }
                    })?;
                    paths.push(path);
                }
            }
        }

        if paths.is_empty() {
            return Some(CollectiveCost {
                latency_s: 0.0,
                bandwidth_s: 0.0,
                total_s: 0.0,
                bottlenecks: Vec::new(),
            });
        }

        let bytes_per_path = Bytes::from_bytes(div_ceil(bytes.as_bytes(), paths.len() as u64));
        let (latency_s, bandwidth_s, mut bottlenecks) =
            Self::contended_paths_cost(&paths, bytes_per_path);
        if has_inter_node_path
            && !bottlenecks
                .iter()
                .any(|bottleneck| bottleneck == "KV transfer fabric/NIC path")
        {
            bottlenecks.push("KV transfer fabric/NIC path".to_string());
        }

        Some(CollectiveCost {
            latency_s,
            bandwidth_s,
            total_s: latency_s + bandwidth_s,
            bottlenecks,
        })
    }

    fn intra_node_gpu_path(cluster: &Cluster, node_id: NodeId) -> RoutedPath {
        let (latency_s, bandwidth) = Self::intra_node_link_profile(cluster, node_id);
        let label = format!("node {node_id} intra-node fabric");
        RoutedPath {
            latency_s,
            bottleneck_bandwidth: bandwidth,
            labels: vec![label.clone()],
            resources: vec![RoutedResource {
                label,
                kind: RoutedResourceKind::IntraNodeFabric,
                from: GraphResource::IntraNode { node_id },
                to: GraphResource::IntraNode { node_id },
                bandwidth,
                latency_s,
                rail_id: None,
            }],
        }
    }

    fn can_use_node_level_gpu_path(cluster: &Cluster, gpu: GpuAddr) -> bool {
        cluster
            .node(gpu.node_id)
            .is_some_and(|node| !node.network.gpu_nic_map.contains_key(&gpu.local_gpu_id))
    }

    fn add_endpoint_intra_node_resources(
        cluster: &Cluster,
        mut path: RoutedPath,
        source_node: NodeId,
        dest_node: NodeId,
    ) -> RoutedPath {
        Self::push_intra_node_resource(cluster, &mut path, source_node);
        Self::push_intra_node_resource(cluster, &mut path, dest_node);
        path
    }

    fn push_intra_node_resource(cluster: &Cluster, path: &mut RoutedPath, node_id: NodeId) {
        let (latency_s, bandwidth) = Self::intra_node_link_profile(cluster, node_id);
        let label = format!("node {node_id} intra-node fabric");
        path.latency_s += latency_s;
        if bandwidth.as_bytes_per_sec() < path.bottleneck_bandwidth.as_bytes_per_sec() {
            path.bottleneck_bandwidth = bandwidth;
        }
        if !path.labels.contains(&label) {
            path.labels.push(label.clone());
        }
        path.resources.push(RoutedResource {
            label,
            kind: RoutedResourceKind::IntraNodeFabric,
            from: GraphResource::IntraNode { node_id },
            to: GraphResource::IntraNode { node_id },
            bandwidth,
            latency_s,
            rail_id: None,
        });
    }

    fn contended_paths_cost(
        paths: &[RoutedPath],
        bytes_per_path: Bytes,
    ) -> (f64, f64, Vec<String>) {
        let mut resource_counts = std::collections::HashMap::new();
        for path in paths {
            let mut seen_in_path = HashSet::new();
            for resource in &path.resources {
                if seen_in_path.insert(resource.label.clone()) {
                    *resource_counts
                        .entry(resource.label.clone())
                        .or_insert(0_u32) += 1;
                }
            }
        }

        let mut max_latency_s = 0.0_f64;
        let mut max_bandwidth_s = 0.0_f64;
        let mut bottlenecks = Vec::new();
        for path in paths {
            max_latency_s = max_latency_s.max(path.latency_s);
            let mut path_bandwidth = f64::INFINITY;
            let mut path_bottleneck = None;
            for resource in &path.resources {
                let count = f64::from(*resource_counts.get(&resource.label).unwrap_or(&1));
                let shared_bandwidth = resource.bandwidth.as_bytes_per_sec() / count.max(1.0);
                if shared_bandwidth < path_bandwidth {
                    path_bandwidth = shared_bandwidth;
                    path_bottleneck = Some(resource.label.clone());
                }
            }

            if let Some(path_bottleneck) = path_bottleneck
                && !bottlenecks.contains(&path_bottleneck)
            {
                bottlenecks.push(path_bottleneck);
            }
            max_bandwidth_s =
                max_bandwidth_s.max(bytes_per_path.as_bytes() as f64 / path_bandwidth.max(1.0));
        }

        (max_latency_s, max_bandwidth_s, bottlenecks)
    }

    fn collective_steps(kind: CollectiveKind, rank_count: f64) -> f64 {
        match kind {
            CollectiveKind::SendRecv | CollectiveKind::Broadcast => 1.0,
            CollectiveKind::AllReduce => 2.0 * (rank_count - 1.0).max(1.0),
            CollectiveKind::AllGather
            | CollectiveKind::ReduceScatter
            | CollectiveKind::AllToAll => (rank_count - 1.0).max(1.0),
        }
    }

    fn traffic_multiplier(kind: CollectiveKind, rank_count: f64) -> f64 {
        match kind {
            CollectiveKind::SendRecv | CollectiveKind::Broadcast => 1.0,
            CollectiveKind::AllReduce => 2.0 * (rank_count - 1.0) / rank_count,
            CollectiveKind::AllGather | CollectiveKind::ReduceScatter => {
                (rank_count - 1.0) / rank_count
            }
            CollectiveKind::AllToAll => (rank_count - 1.0) / rank_count,
        }
        .max(1.0 / rank_count)
    }

    fn intra_node_profile(cluster: &Cluster, node_id: NodeId) -> (f64, Bandwidth) {
        let Some(node) = cluster.node(node_id) else {
            return (0.0, Bandwidth::from_bytes_per_sec(1.0));
        };

        match &node.intra_node_fabric {
            IntraNodeTopology::NvSwitch(profile) | IntraNodeTopology::NvLinkDirect(profile) => {
                (profile.latency.to_us() / 1e6, profile.bw.unidirectional)
            }
            IntraNodeTopology::Pcie(profile) => {
                (profile.latency.to_us() / 1e6, profile.bw.unidirectional)
            }
            IntraNodeTopology::Custom(edges) => {
                let mut min_bw = None;
                let mut max_latency = 0.0_f64;
                for edge in edges.values() {
                    match edge {
                        crate::types::fabric::intra_node::LinkProfile::NvLink(profile) => {
                            min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                            max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                        }
                        crate::types::fabric::intra_node::LinkProfile::Pcie(profile) => {
                            min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                            max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                        }
                    }
                }
                (
                    max_latency,
                    min_bw.unwrap_or_else(|| Bandwidth::from_bytes_per_sec(1.0)),
                )
            }
        }
    }

    pub(crate) fn intra_node_link_profile(cluster: &Cluster, node_id: NodeId) -> (f64, Bandwidth) {
        Self::intra_node_profile(cluster, node_id)
    }

    fn inter_node_profile(cluster: &Cluster) -> (f64, Bandwidth) {
        match &cluster.inter_node_topology {
            InterNodeTopology::FatTree {
                link,
                oversubscription,
                ..
            } => (
                link.latency.to_us() / 1e6,
                link.bw.unidirectional / oversubscription.max(1.0),
            ),
            InterNodeTopology::Flat { link } => {
                (link.latency.to_us() / 1e6, link.bw.unidirectional)
            }
            InterNodeTopology::Custom(edges) => {
                let mut min_bw = None;
                let mut max_latency = 0.0_f64;
                for profile in edges.values().flatten().map(|link| &link.profile) {
                    min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                    max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                }
                (
                    max_latency,
                    min_bw.unwrap_or_else(|| Bandwidth::from_bytes_per_sec(1.0)),
                )
            }
        }
    }

    fn inter_node_profile_for_nodes(
        cluster: &Cluster,
        nodes: &BTreeSet<NodeId>,
    ) -> (f64, Bandwidth) {
        match &cluster.inter_node_topology {
            InterNodeTopology::Custom(edges) if nodes.len() > 1 => {
                let mut min_bw = None;
                let mut max_latency = 0.0_f64;
                let node_ids: Vec<_> = nodes.iter().copied().collect();

                for i in 0..node_ids.len() {
                    for j in i + 1..node_ids.len() {
                        if let Some(links) =
                            edges.get(&UnorderedPair::new(node_ids[i], node_ids[j]))
                        {
                            for profile in links.iter().map(|link| &link.profile) {
                                min_bw = Some(min_bandwidth(min_bw, profile.bw.unidirectional));
                                max_latency = max_latency.max(profile.latency.to_us() / 1e6);
                            }
                        }
                    }
                }

                if let Some(min_bw) = min_bw {
                    (max_latency, min_bw)
                } else {
                    Self::inter_node_profile(cluster)
                }
            }
            _ => Self::inter_node_profile(cluster),
        }
    }

    fn effective_nic_bandwidth(cluster: &Cluster, addrs: &[GpuAddr]) -> Bandwidth {
        let mut total = 0.0;
        let mut seen = HashSet::new();

        for addr in addrs {
            let Some(node) = cluster.node(addr.node_id) else {
                continue;
            };
            for nic_id in node.network.nic_candidates_for_gpu(addr.local_gpu_id) {
                if seen.insert((addr.node_id, nic_id)) {
                    total += node.network.nic_bandwidth(nic_id).as_bytes_per_sec();
                }
            }
        }

        Bandwidth::from_bytes_per_sec(total.max(1.0))
    }
}

fn search_deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
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

fn calibration_feature_values(
    model: &ModelSpec,
    request: &InferenceRequest,
    config: ParallelismConfig,
    extra_features: Option<&BTreeMap<String, f64>>,
) -> BTreeMap<String, f64> {
    let mut features = BTreeMap::new();
    let batch_size = f64::from(request.batch_size.max(1));
    let prompt_tokens = f64::from(request.prompt_tokens.max(1));
    let decode_tokens = f64::from(request.decode_tokens.max(1));
    let sequence_tokens = f64::from(request.max_sequence_tokens.max(1));
    let tensor_ranks = f64::from(config.tensor_ranks.max(1));
    let pipeline_ranks = f64::from(config.pipeline_ranks.max(1));
    let expert_ranks = f64::from(config.expert_ranks.max(1));
    let data_ranks = f64::from(config.data_ranks.max(1));
    let total_ranks = f64::from(config.total_ranks().max(1));

    insert_feature(&mut features, "batch_size", batch_size);
    insert_feature(&mut features, "batch", batch_size);
    insert_feature(&mut features, "prompt_tokens", prompt_tokens);
    insert_feature(&mut features, "prefill_tokens", prompt_tokens);
    insert_feature(&mut features, "effective_prefill_tokens", prompt_tokens);
    insert_feature(&mut features, "decode_tokens", decode_tokens);
    insert_feature(&mut features, "output_tokens", decode_tokens);
    insert_feature(&mut features, "sequence_tokens", sequence_tokens);
    insert_feature(&mut features, "max_sequence_tokens", sequence_tokens);
    insert_feature(&mut features, "batch_tokens", batch_size * prompt_tokens);
    insert_feature(
        &mut features,
        "prefill_batch_tokens",
        batch_size * prompt_tokens,
    );
    insert_feature(
        &mut features,
        "decode_batch_tokens",
        batch_size * decode_tokens,
    );
    insert_feature(&mut features, "tensor_ranks", tensor_ranks);
    insert_feature(&mut features, "tp", tensor_ranks);
    insert_feature(&mut features, "pipeline_ranks", pipeline_ranks);
    insert_feature(&mut features, "pp", pipeline_ranks);
    insert_feature(&mut features, "expert_ranks", expert_ranks);
    insert_feature(&mut features, "ep", expert_ranks);
    insert_feature(&mut features, "data_ranks", data_ranks);
    insert_feature(&mut features, "dp", data_ranks);
    insert_feature(&mut features, "total_ranks", total_ranks);
    insert_feature(
        &mut features,
        "parameters_gb",
        model.parameters.as_gigabytes(),
    );
    insert_feature(
        &mut features,
        "model_params_gb",
        model.parameters.as_gigabytes(),
    );
    insert_feature(
        &mut features,
        "parameter_count_billion",
        model.parameter_count_billion(),
    );
    insert_feature(
        &mut features,
        "model_parameter_count_billion",
        model.parameter_count_billion(),
    );
    insert_feature(&mut features, "layers", f64::from(model.layers));
    insert_feature(&mut features, "hidden_size", f64::from(model.hidden_size));
    insert_feature(
        &mut features,
        "attention_heads",
        f64::from(model.attention_heads.max(1)),
    );
    insert_feature(&mut features, "kv_heads", f64::from(model.kv_heads.max(1)));
    insert_feature(
        &mut features,
        "dtype_bytes",
        model.dtype.bytes_per_element() as f64,
    );
    insert_feature(
        &mut features,
        "kv_dtype_bytes",
        model.kv_dtype().bytes_per_element() as f64,
    );
    insert_feature(
        &mut features,
        "kv_cache_dtype_bytes",
        model.kv_dtype().bytes_per_element() as f64,
    );

    if let Some(extra_features) = extra_features {
        for (name, value) in extra_features {
            insert_feature(&mut features, name, *value);
        }
    }

    features
}

fn insert_feature(features: &mut BTreeMap<String, f64>, name: &str, value: f64) {
    if value.is_finite() {
        features.insert(normalize_fit_name(name), value);
    }
}

fn fit_matches(fit: &CalibrationFittedModel, phase: &str, targets: &[&str]) -> bool {
    if !matches!(
        normalize_fit_name(&fit.model).as_str(),
        "linear" | "linear_regression" | "ols" | "ordinary_least_squares"
    ) {
        return false;
    }
    if let Some(fit_phase) = &fit.phase
        && normalize_fit_name(fit_phase) != normalize_fit_name(phase)
    {
        return false;
    }
    let fit_target = normalize_fit_name(&fit.target);
    targets
        .iter()
        .any(|target| normalize_fit_name(target) == fit_target)
}

fn evaluate_fit(
    fit: &CalibrationFittedModel,
    phase: &str,
    features: &BTreeMap<String, f64>,
    baseline_s: Option<f64>,
) -> Option<FitEvaluation> {
    let mut prediction = fit.intercept.unwrap_or(0.0);
    let mut feature_values = Vec::with_capacity(fit.features.len());
    let mut has_range = false;
    let mut all_ranged = true;
    let mut has_extrapolation = false;
    let mut max_extrapolation_ratio = 0.0_f64;
    for (feature, coefficient) in fit.features.iter().zip(&fit.coefficients) {
        let value = features.get(&normalize_fit_name(feature))?;
        prediction += coefficient * value;
        let feature_range = fit_feature_range(fit, feature);
        let status = feature_range_status(feature_range, *value);
        has_range |= feature_range.is_some();
        all_ranged &= feature_range.is_some();
        has_extrapolation |= status.status == "extrapolated";
        max_extrapolation_ratio = max_extrapolation_ratio.max(status.extrapolation_ratio);
        feature_values.push(CalibrationFitFeatureValue {
            name: feature.clone(),
            value: *value,
            coefficient: *coefficient,
            range_min: feature_range.and_then(|range| range.min),
            range_max: feature_range.and_then(|range| range.max),
            status: status.status,
            extrapolation_ratio: status.extrapolation_ratio,
        });
    }
    if !prediction.is_finite() || prediction <= 0.0 {
        return None;
    }

    let seconds = fit_value_to_seconds(prediction, fit);
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }

    let applicability_status = if has_extrapolation {
        "extrapolated"
    } else if all_ranged && has_range {
        "interpolated"
    } else if has_range {
        "partially_bounded"
    } else {
        "unbounded"
    };
    let confidence_score = fit_confidence_score(fit, applicability_status, max_extrapolation_ratio);
    let (relative_uncertainty_pct, absolute_uncertainty_s, uncertainty_source) =
        fit_uncertainty(fit, seconds);

    Some(FitEvaluation {
        seconds,
        application: CalibrationFitApplication {
            phase: phase.to_string(),
            target: fit.target.clone(),
            fit_name: fit.name.clone(),
            model: fit.model.clone(),
            unit: fit.unit.clone(),
            intercept: fit.intercept.unwrap_or(0.0),
            raw_prediction: prediction,
            prediction_kind: "latency".to_string(),
            predicted_value: seconds,
            prediction_unit: Some("s".to_string()),
            predicted_s: seconds,
            baseline_value: baseline_s,
            baseline_s,
            applicability_status: applicability_status.to_string(),
            confidence_score,
            max_extrapolation_ratio,
            relative_uncertainty_pct,
            absolute_uncertainty_value: absolute_uncertainty_s,
            absolute_uncertainty_s,
            uncertainty_source,
            validation_rmse: fit.validation_rmse,
            validation_rmse_pct: fit.validation_rmse_pct,
            validation_mean_abs_pct_error: fit.validation_mean_abs_pct_error,
            validation_max_abs_pct_error: fit.validation_max_abs_pct_error,
            confidence_interval: fit.confidence_interval,
            confidence_interval_pct: fit.confidence_interval_pct,
            confidence_level: fit.confidence_level,
            sample_count: fit.sample_count,
            validation_sample_count: fit.validation_sample_count,
            source: fit.source.clone(),
            features: feature_values,
        },
    })
}

fn evaluate_value_fit(
    fit: &CalibrationFittedModel,
    phase: &str,
    features: &BTreeMap<String, f64>,
    baseline_value: Option<f64>,
    prediction_kind: &str,
    prediction_unit: Option<&str>,
) -> Option<CalibrationFitApplication> {
    let mut prediction = fit.intercept.unwrap_or(0.0);
    let mut feature_values = Vec::with_capacity(fit.features.len());
    let mut has_range = false;
    let mut all_ranged = true;
    let mut has_extrapolation = false;
    let mut max_extrapolation_ratio = 0.0_f64;
    for (feature, coefficient) in fit.features.iter().zip(&fit.coefficients) {
        let value = features.get(&normalize_fit_name(feature))?;
        prediction += coefficient * value;
        let feature_range = fit_feature_range(fit, feature);
        let status = feature_range_status(feature_range, *value);
        has_range |= feature_range.is_some();
        all_ranged &= feature_range.is_some();
        has_extrapolation |= status.status == "extrapolated";
        max_extrapolation_ratio = max_extrapolation_ratio.max(status.extrapolation_ratio);
        feature_values.push(CalibrationFitFeatureValue {
            name: feature.clone(),
            value: *value,
            coefficient: *coefficient,
            range_min: feature_range.and_then(|range| range.min),
            range_max: feature_range.and_then(|range| range.max),
            status: status.status,
            extrapolation_ratio: status.extrapolation_ratio,
        });
    }
    if !prediction.is_finite() || prediction <= 0.0 {
        return None;
    }

    let applicability_status = if has_extrapolation {
        "extrapolated"
    } else if all_ranged && has_range {
        "interpolated"
    } else if has_range {
        "partially_bounded"
    } else {
        "unbounded"
    };
    let confidence_score = fit_confidence_score(fit, applicability_status, max_extrapolation_ratio);
    let (relative_uncertainty_pct, absolute_uncertainty_value, uncertainty_source) =
        fit_uncertainty_value(fit, prediction);

    Some(CalibrationFitApplication {
        phase: phase.to_string(),
        target: fit.target.clone(),
        fit_name: fit.name.clone(),
        model: fit.model.clone(),
        unit: fit.unit.clone(),
        intercept: fit.intercept.unwrap_or(0.0),
        raw_prediction: prediction,
        prediction_kind: prediction_kind.to_string(),
        predicted_value: prediction,
        prediction_unit: prediction_unit.map(str::to_string),
        predicted_s: 0.0,
        baseline_value,
        baseline_s: None,
        applicability_status: applicability_status.to_string(),
        confidence_score,
        max_extrapolation_ratio,
        relative_uncertainty_pct,
        absolute_uncertainty_value,
        absolute_uncertainty_s: None,
        uncertainty_source,
        validation_rmse: fit.validation_rmse,
        validation_rmse_pct: fit.validation_rmse_pct,
        validation_mean_abs_pct_error: fit.validation_mean_abs_pct_error,
        validation_max_abs_pct_error: fit.validation_max_abs_pct_error,
        confidence_interval: fit.confidence_interval,
        confidence_interval_pct: fit.confidence_interval_pct,
        confidence_level: fit.confidence_level,
        sample_count: fit.sample_count,
        validation_sample_count: fit.validation_sample_count,
        source: fit.source.clone(),
        features: feature_values,
    })
}

struct FeatureRangeStatus {
    status: String,
    extrapolation_ratio: f64,
}

fn fit_feature_range<'a>(
    fit: &'a CalibrationFittedModel,
    feature: &str,
) -> Option<&'a CalibrationFitFeatureRange> {
    let normalized = normalize_fit_name(feature);
    fit.feature_ranges
        .iter()
        .find(|range| normalize_fit_name(&range.feature) == normalized)
}

fn feature_range_status(
    range: Option<&CalibrationFitFeatureRange>,
    value: f64,
) -> FeatureRangeStatus {
    let Some(range) = range else {
        return FeatureRangeStatus {
            status: "unbounded".to_string(),
            extrapolation_ratio: 0.0,
        };
    };
    let lower_excess = range.min.map(|min| (min - value).max(0.0)).unwrap_or(0.0);
    let upper_excess = range.max.map(|max| (value - max).max(0.0)).unwrap_or(0.0);
    let excess = lower_excess.max(upper_excess);
    if excess <= 0.0 {
        return FeatureRangeStatus {
            status: "in_range".to_string(),
            extrapolation_ratio: 0.0,
        };
    }
    let scale = match (range.min, range.max) {
        (Some(min), Some(max)) => (max - min).abs().max(1.0),
        (Some(bound), None) | (None, Some(bound)) => bound.abs().max(1.0),
        (None, None) => 1.0,
    };
    FeatureRangeStatus {
        status: "extrapolated".to_string(),
        extrapolation_ratio: excess / scale,
    }
}

fn fit_confidence_score(
    fit: &CalibrationFittedModel,
    applicability_status: &str,
    max_extrapolation_ratio: f64,
) -> f64 {
    let mut score = fit.r_squared.unwrap_or(1.0).clamp(0.0, 1.0);
    if let Some((error_pct, _)) = fit_relative_error_pct(fit) {
        score *= (1.0 - (error_pct / 100.0).clamp(0.0, 1.0)).max(0.0);
    }
    match applicability_status {
        "interpolated" => {}
        "partially_bounded" => score *= 0.9,
        "unbounded" => score *= 0.75,
        "extrapolated" => score /= 1.0 + max_extrapolation_ratio.max(0.0),
        _ => {}
    }
    score.clamp(0.0, 1.0)
}

fn fit_uncertainty(
    fit: &CalibrationFittedModel,
    predicted_s: f64,
) -> (Option<f64>, Option<f64>, Option<String>) {
    let (relative_uncertainty_pct, absolute_uncertainty_value, uncertainty_source) =
        fit_uncertainty_value(fit, predicted_s);
    let absolute_from_rmse = fit_absolute_uncertainty_value(fit)
        .map(|(value, _)| value)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| fit_value_to_seconds(value, fit))
        .filter(|value| value.is_finite() && *value >= 0.0);
    let absolute_uncertainty_s = absolute_from_rmse.or(absolute_uncertainty_value);

    (
        relative_uncertainty_pct,
        absolute_uncertainty_s,
        uncertainty_source,
    )
}

fn fit_uncertainty_value(
    fit: &CalibrationFittedModel,
    predicted_value: f64,
) -> (Option<f64>, Option<f64>, Option<String>) {
    let relative = fit_uncertainty_pct(fit).filter(|(value, _)| value.is_finite() && *value >= 0.0);
    let relative_uncertainty_pct = relative.map(|(value, _)| value);

    let absolute_from_rmse =
        fit_absolute_uncertainty_value(fit).filter(|(value, _)| value.is_finite() && *value >= 0.0);
    let absolute_from_relative = relative_uncertainty_pct
        .filter(|_| predicted_value.is_finite() && predicted_value >= 0.0)
        .map(|value| predicted_value * value / 100.0);
    let absolute_uncertainty_value = absolute_from_rmse
        .map(|(value, _)| value)
        .or(absolute_from_relative);
    let uncertainty_source = if let Some((_, source)) = absolute_from_rmse {
        Some(source.to_string())
    } else {
        relative.map(|(_, source)| source.to_string())
    };

    (
        relative_uncertainty_pct,
        absolute_uncertainty_value,
        uncertainty_source,
    )
}

fn fit_uncertainty_pct(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.confidence_interval_pct
        .map(|value| (value, "confidence_interval_pct"))
        .or_else(|| fit_relative_error_pct(fit))
}

fn fit_relative_error_pct(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.validation_rmse_pct
        .map(|value| (value, "validation_rmse_pct"))
        .or_else(|| {
            fit.validation_mean_abs_pct_error
                .map(|value| (value, "validation_mean_abs_pct_error"))
        })
        .or_else(|| {
            fit.validation_max_abs_pct_error
                .map(|value| (value, "validation_max_abs_pct_error"))
        })
        .or_else(|| fit.rmse_pct.map(|value| (value, "rmse_pct")))
        .or_else(|| {
            fit.mean_abs_pct_error
                .map(|value| (value, "mean_abs_pct_error"))
        })
        .or_else(|| {
            fit.max_abs_pct_error
                .map(|value| (value, "max_abs_pct_error"))
        })
}

fn fit_absolute_uncertainty_value(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.confidence_interval
        .map(|value| (value, "confidence_interval"))
        .or_else(|| fit_absolute_error_value(fit))
}

fn fit_absolute_error_value(fit: &CalibrationFittedModel) -> Option<(f64, &'static str)> {
    fit.validation_rmse
        .map(|value| (value, "validation_rmse"))
        .or_else(|| fit.rmse.map(|value| (value, "rmse")))
}

fn fit_value_to_seconds(value: f64, fit: &CalibrationFittedModel) -> f64 {
    if let Some(unit) = fit.unit.as_deref().map(normalize_fit_name) {
        if unit == "us" || unit.contains("microsecond") {
            return value / 1e6;
        }
        if unit == "ms" || unit.contains("millisecond") {
            return value / 1e3;
        }
        if unit == "s" || unit.contains("second") {
            return value;
        }
    }

    let target = normalize_fit_name(&fit.target);
    if target.ends_with("_us") || target.contains("_us_") {
        value / 1e6
    } else if target.ends_with("_ms") || target.contains("_ms_") {
        value / 1e3
    } else {
        value
    }
}

fn normalize_fit_name(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
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

pub fn schedule_operations(operations: &[SimOperation]) -> Vec<ScheduledOperation> {
    let mut scheduler = ResourceScheduler::new();
    let mut scheduled_ids = Vec::with_capacity(operations.len());

    for operation in operations {
        let dependencies: Vec<_> = operation
            .dependencies
            .iter()
            .filter_map(|idx| scheduled_ids.get(*idx).copied())
            .collect();
        let scheduled_id = scheduler.schedule(
            operation.name.clone(),
            operation.duration_s,
            0.0,
            &dependencies,
            operation.resources.clone(),
        );
        scheduled_ids.push(scheduled_id);
    }

    scheduler.operations().to_vec()
}

fn compute_resources(placement: &RankPlacement, phase: InferencePhase) -> Vec<String> {
    let mut resources: Vec<_> = placement
        .rank_to_gpu
        .iter()
        .map(|addr| format!("gpu compute node {}", addr.node_id))
        .collect();
    if matches!(phase, InferencePhase::Decode | InferencePhase::EndToEnd) {
        resources.extend(
            placement
                .rank_to_gpu
                .iter()
                .map(|addr| format!("gpu HBM node {}", addr.node_id)),
        );
    }
    resources.sort();
    resources.dedup();
    resources
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
}
