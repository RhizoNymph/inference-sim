use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use crate::{
    calibration::SimulationCalibration,
    config::{ApproximationPolicyViolation, CalibrationGateViolation, CalibrationProfileMetadata},
    scheduler::{ResourceScheduler, ResourceUtilization, ScheduledOperation, resource_utilization},
    solver::{
        CalibrationFitApplication, PlacementEvidence, ScoredParallelismConfig, SearchSpace,
        SimOperation, SimulationApproximation, Solver, SolverOptions,
    },
    topology_graph::{GraphResource, RoutedResource, TopologyGraph},
    types::{
        common::{Bytes, FabricKind, GpuAddr, NodeId},
        configs::{ParallelismConfig, RankPlacement},
        fabric::inter_node::{FabricProfile, InterNodeTopology},
        topology::{Cluster, Node},
    },
    workload::{DType, InferencePhase, InferenceRequest, ModelSpec},
};

mod approximations;
mod arrivals;
mod calibration;
mod hardware;
mod measurement;
mod memory;
mod metric_fits;
mod model;
mod observations;
mod pools;
mod ranking;
mod rejections;
mod routing;
use approximations::*;
use arrivals::*;
use calibration::*;
use hardware::*;
use measurement::*;
use memory::*;
use metric_fits::*;
pub use model::*;
#[cfg(test)]
use observations::request_kv_block_ownership;
pub use observations::{
    ServingKvTransferPathEndpointObservation, ServingKvTransferPathObservation,
    ServingKvTransferPathResourceObservation, ServingRequestObservation,
    ServingRouteCandidateObservation,
};
use observations::{
    decode_owner_gpus, kv_residency_window_s, prefill_capacity_spans, prefill_owner_gpus,
    prefill_owner_nodes,
};
use pools::*;
use ranking::*;
use rejections::*;
use routing::*;

struct ScorePairContext<'a> {
    cluster: &'a Cluster,
    model: &'a ModelSpec,
    request: &'a InferenceRequest,
    traffic: &'a ServingTraffic,
    slo_policies: &'a [ServingSloPolicy],
    pool: &'a ResolvedServingPool,
    calibration: SimulationCalibration,
    calibration_profile: Option<&'a CalibrationProfileMetadata>,
    model_id: Option<&'a str>,
    serving_stack: Option<&'a str>,
    serving_runtime_features: &'a [String],
    objective: ServingObjective,
    slo_miss_penalty_weight: f64,
    slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
    topology_risk_penalty_weight: f64,
    max_memory_pressure_fraction: Option<f64>,
    max_unique_gpus: Option<u32>,
    min_throughput_tokens_per_s: Option<f64>,
    cost_model: &'a ServingCostModel,
    pool_search_summary: Option<&'a ServingPoolSearchSummary>,
}

#[derive(Clone)]
struct RejectedPairContext {
    objective: ServingObjective,
    slo_miss_penalty_weight: f64,
    slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
    topology_risk_penalty_weight: f64,
    max_memory_pressure_fraction: Option<f64>,
    max_unique_gpus: Option<u32>,
    min_throughput_tokens_per_s: Option<f64>,
    cost_model: ServingCostModel,
    metric_ceilings: ServingMetricCeilings,
    kv_route_constraints: ServingKvRouteConstraints,
    pool_search_summary: Option<ServingPoolSearchSummary>,
}

impl RejectedPairContext {
    fn from_config(
        config: &DisaggregatedServingConfig,
        pool_search_summary: Option<ServingPoolSearchSummary>,
    ) -> Self {
        Self {
            objective: config.objective,
            slo_miss_penalty_weight: config.slo_miss_penalty_weight,
            slo_miss_penalty_weights: config.slo_miss_penalty_weights,
            topology_risk_penalty_weight: config.topology_risk_penalty_weight,
            max_memory_pressure_fraction: config.max_memory_pressure_fraction,
            max_unique_gpus: config.max_unique_gpus,
            min_throughput_tokens_per_s: config.min_throughput_tokens_per_s,
            cost_model: config.cost_model.clone(),
            metric_ceilings: config.traffic.metric_ceilings,
            kv_route_constraints: config.traffic.kv_route_constraints,
            pool_search_summary,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedServingPool {
    label: Option<String>,
    prefill_nodes: Vec<NodeId>,
    decode_nodes: Vec<NodeId>,
    prefill_gpu_labels: Vec<String>,
    decode_gpu_labels: Vec<String>,
}

impl ServingSolver {
    pub fn rank_disaggregated(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: &DisaggregatedServingConfig,
        calibration: SimulationCalibration,
    ) -> Vec<ScoredServingConfig> {
        Self::rank_disaggregated_with_options(
            cluster,
            model,
            request,
            config,
            ServingSolverOptions {
                calibration,
                calibration_profile: None,
                model_id: None,
                serving_stack: None,
                serving_runtime_features: None,
                max_prefill_candidates: None,
                max_decode_candidates: None,
                max_serving_pairs: None,
                search_deadline: None,
                explicit_prefill_placement: None,
                explicit_decode_placement: None,
            },
        )
    }

    pub fn rank_disaggregated_with_options(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        config: &DisaggregatedServingConfig,
        options: ServingSolverOptions<'_>,
    ) -> Vec<ScoredServingConfig> {
        let mut results = Vec::new();
        let mut pool_candidates = config.pool_candidates.clone();
        let mut pool_search_summary = None;
        if let Some(pool_search) = &config.pool_search {
            match generate_pool_search_candidates_with_summary(
                cluster,
                pool_search,
                config.deployment_mode,
            ) {
                Ok(generated) => {
                    pool_search_summary = Some(generated.summary);
                    extend_unique_pool_candidates(&mut pool_candidates, generated.candidates);
                }
                Err(error) => {
                    pool_search_summary = Some(error.summary.clone());
                    results.push(Self::rejected_pair_for_pool(
                        &ResolvedServingPool {
                            label: Some("pool-search".to_string()),
                            prefill_nodes: Vec::new(),
                            decode_nodes: Vec::new(),
                            prefill_gpu_labels: Vec::new(),
                            decode_gpu_labels: Vec::new(),
                        },
                        RejectedPairContext::from_config(config, Some(error.summary)),
                        error.reason,
                    ));
                }
            }
        }

        if pool_candidates.is_empty()
            && !config.prefill_nodes.is_empty()
            && !config.decode_nodes.is_empty()
        {
            pool_candidates.push(ServingPoolCandidate {
                label: None,
                prefill_nodes: config.prefill_nodes.clone(),
                decode_nodes: config.decode_nodes.clone(),
                prefill_groups: Vec::new(),
                decode_groups: Vec::new(),
                prefill_node_filter: ServingPoolNodeFilter::default(),
                decode_node_filter: ServingPoolNodeFilter::default(),
                domain_spread: ServingPoolDomainSpread::default(),
                prefill_gpu_labels: Vec::new(),
                decode_gpu_labels: Vec::new(),
            });
        }

        let mut remaining_serving_pairs = options.max_serving_pairs;
        for pool in &pool_candidates {
            if remaining_serving_pairs == Some(0)
                || search_deadline_expired(options.search_deadline)
            {
                break;
            }
            let resolved_pool = match resolve_pool_candidate(cluster, pool) {
                Ok(resolved_pool) => resolved_pool,
                Err(reason) => {
                    results.push(Self::rejected_pair_for_pool(
                        &ResolvedServingPool {
                            label: pool.label.clone(),
                            prefill_nodes: pool.prefill_nodes.clone(),
                            decode_nodes: pool.decode_nodes.clone(),
                            prefill_gpu_labels: pool.prefill_gpu_labels.clone(),
                            decode_gpu_labels: pool.decode_gpu_labels.clone(),
                        },
                        RejectedPairContext::from_config(config, pool_search_summary.clone()),
                        reason,
                    ));
                    continue;
                }
            };
            if !config
                .deployment_mode
                .accepts_pool(&resolved_pool.prefill_nodes, &resolved_pool.decode_nodes)
            {
                let effective = ServingDeploymentMode::effective_for_pool(
                    &resolved_pool.prefill_nodes,
                    &resolved_pool.decode_nodes,
                );
                results.push(Self::rejected_pair_for_pool(
                    &resolved_pool,
                    RejectedPairContext::from_config(config, pool_search_summary.clone()),
                    format!(
                        "serving.mode '{}' does not allow {} pool prefill_nodes=[{}] decode_nodes=[{}]",
                        config.deployment_mode.as_str(),
                        effective.as_str(),
                        node_list(&resolved_pool.prefill_nodes),
                        node_list(&resolved_pool.decode_nodes)
                    ),
                ));
                continue;
            }
            let pool_options = ServingSolverOptions {
                max_serving_pairs: remaining_serving_pairs,
                ..options
            };
            let pool_results = Self::rank_pool_candidate(
                cluster,
                model,
                request,
                &config.search,
                &config.traffic,
                &config.slo_policies,
                &resolved_pool,
                config.objective,
                config.slo_miss_penalty_weight,
                config.slo_miss_penalty_weights,
                config.topology_risk_penalty_weight,
                config.max_memory_pressure_fraction,
                config.max_unique_gpus,
                config.min_throughput_tokens_per_s,
                &config.cost_model,
                pool_search_summary.as_ref(),
                pool_options,
            );
            if let Some(remaining) = remaining_serving_pairs.as_mut() {
                *remaining = remaining.saturating_sub(pool_results.len());
            }
            results.extend(pool_results);
        }

        annotate_serving_pareto(&mut results);
        sort_serving_results(&mut results, config.objective);
        results
    }

    #[allow(clippy::too_many_arguments)]
    fn rank_pool_candidate(
        cluster: &Cluster,
        model: &ModelSpec,
        request: &InferenceRequest,
        search: &ServingSearchSpace,
        traffic: &ServingTraffic,
        slo_policies: &[ServingSloPolicy],
        pool: &ResolvedServingPool,
        objective: ServingObjective,
        slo_miss_penalty_weight: f64,
        slo_miss_penalty_weights: ServingSloMissPenaltyWeights,
        topology_risk_penalty_weight: f64,
        max_memory_pressure_fraction: Option<f64>,
        max_unique_gpus: Option<u32>,
        min_throughput_tokens_per_s: Option<f64>,
        cost_model: &ServingCostModel,
        pool_search_summary: Option<&ServingPoolSearchSummary>,
        options: ServingSolverOptions<'_>,
    ) -> Vec<ScoredServingConfig> {
        let Ok(prefill_cluster) =
            cluster.subset_nodes_with_gpu_labels(&pool.prefill_nodes, &pool.prefill_gpu_labels)
        else {
            return vec![Self::rejected_pair_for_pool(
                pool,
                RejectedPairContext {
                    objective,
                    slo_miss_penalty_weight,
                    slo_miss_penalty_weights,
                    topology_risk_penalty_weight,
                    max_memory_pressure_fraction,
                    max_unique_gpus,
                    min_throughput_tokens_per_s,
                    cost_model: cost_model.clone(),
                    metric_ceilings: traffic.metric_ceilings,
                    kv_route_constraints: traffic.kv_route_constraints,
                    pool_search_summary: pool_search_summary.cloned(),
                },
                "serving pool prefill_nodes contains an unknown node id".to_string(),
            )];
        };
        let Ok(decode_cluster) =
            cluster.subset_nodes_with_gpu_labels(&pool.decode_nodes, &pool.decode_gpu_labels)
        else {
            return vec![Self::rejected_pair_for_pool(
                pool,
                RejectedPairContext {
                    objective,
                    slo_miss_penalty_weight,
                    slo_miss_penalty_weights,
                    topology_risk_penalty_weight,
                    max_memory_pressure_fraction,
                    max_unique_gpus,
                    min_throughput_tokens_per_s,
                    cost_model: cost_model.clone(),
                    metric_ceilings: traffic.metric_ceilings,
                    kv_route_constraints: traffic.kv_route_constraints,
                    pool_search_summary: pool_search_summary.cloned(),
                },
                "serving pool decode_nodes contains an unknown node id".to_string(),
            )];
        };

        let prefill_solver_options = SolverOptions {
            calibration: options.calibration,
            calibration_profile: options.calibration_profile,
            max_candidates: options.max_prefill_candidates,
            search_deadline: options.search_deadline,
            explicit_placement: options.explicit_prefill_placement,
        };
        let decode_solver_options = SolverOptions {
            calibration: options.calibration,
            calibration_profile: options.calibration_profile,
            max_candidates: options.max_decode_candidates,
            search_deadline: options.search_deadline,
            explicit_placement: options.explicit_decode_placement,
        };
        let prefill_request = InferenceRequest {
            phase: InferencePhase::Prefill,
            ..*request
        };
        let decode_request = InferenceRequest {
            phase: InferencePhase::Decode,
            ..*request
        };
        let decode_one_request = InferenceRequest {
            decode_tokens: 1,
            phase: InferencePhase::Decode,
            ..*request
        };

        let prefill_scores = Solver::rank_configs_with_options(
            &prefill_cluster,
            model,
            &prefill_request,
            &search.prefill,
            prefill_solver_options,
        );
        let decode_scores = Solver::rank_configs_with_options(
            &decode_cluster,
            model,
            &decode_request,
            &search.decode,
            decode_solver_options,
        );
        let mut results = Vec::new();
        let pair_budget = options.max_serving_pairs.unwrap_or(usize::MAX);
        'pair_search: for prefill_score in &prefill_scores {
            for decode_score in &decode_scores {
                if results.len() >= pair_budget || search_deadline_expired(options.search_deadline)
                {
                    break 'pair_search;
                }
                let decode_one_score = Solver::score_config_with_options(
                    &decode_cluster,
                    model,
                    &decode_one_request,
                    decode_score.config,
                    decode_solver_options,
                );
                results.push(Self::score_pair(
                    prefill_score.clone(),
                    decode_score.clone(),
                    decode_one_score,
                    ScorePairContext {
                        cluster,
                        model,
                        request,
                        traffic,
                        slo_policies,
                        pool,
                        calibration: options.calibration,
                        calibration_profile: options.calibration_profile,
                        model_id: options.model_id,
                        serving_stack: options.serving_stack,
                        serving_runtime_features: options.serving_runtime_features.unwrap_or(&[]),
                        objective,
                        slo_miss_penalty_weight,
                        slo_miss_penalty_weights,
                        topology_risk_penalty_weight,
                        max_memory_pressure_fraction,
                        max_unique_gpus,
                        min_throughput_tokens_per_s,
                        cost_model,
                        pool_search_summary,
                    },
                ));
            }
        }

        results
    }

    fn score_pair(
        prefill_score: ScoredParallelismConfig,
        decode_score: ScoredParallelismConfig,
        decode_one_score: ScoredParallelismConfig,
        context: ScorePairContext<'_>,
    ) -> ScoredServingConfig {
        let mut feasible =
            prefill_score.feasible && decode_score.feasible && decode_one_score.feasible;
        let mut rejections = Vec::new();
        rejections.extend(parallelism_rejections("prefill", &prefill_score));
        rejections.extend(parallelism_rejections("decode", &decode_score));
        rejections.extend(parallelism_rejections(
            "decode_first_token",
            &decode_one_score,
        ));
        let service_rejections = service_health_rejections(context.traffic, context.pool);
        if !service_rejections.is_empty() {
            feasible = false;
            rejections.extend(service_rejections);
        }
        let route_coverage = route_coverage(
            context.cluster,
            context.pool,
            &prefill_score,
            &decode_one_score,
        );
        let route_rejections = route_availability_rejections(
            context.cluster,
            context.model,
            context.request,
            context.traffic,
            context.pool,
            &prefill_score,
            &decode_one_score,
            context.calibration,
            route_coverage,
        );
        if !route_rejections.is_empty() {
            feasible = false;
            rejections.extend(route_rejections);
        }

        let simulation = if feasible {
            schedule_serving_simulation(
                &prefill_score,
                &decode_score,
                &decode_one_score,
                context.cluster,
                context.model,
                context.request,
                context.traffic,
                &context.pool.prefill_nodes,
                &context.pool.decode_nodes,
                context.calibration,
                context.calibration_profile,
            )
        } else {
            ServingSimulation::rejected()
        };
        let metrics = simulation.metrics;
        let kv_route_resource_summary = kv_route_resource_summary(&simulation.request_observations);
        let kv_route_topology_summary = kv_route_topology_summary(&kv_route_resource_summary);
        let topology_bottlenecks = topology_bottleneck_observations(
            route_coverage,
            &kv_route_topology_summary,
            &kv_route_resource_summary,
            &simulation.request_observations,
            &simulation.phase_resource_utilization,
        )
        .into_iter()
        .chain(topology_domain_bottleneck_observations(
            context.cluster,
            &prefill_score.placement,
            &decode_score.placement,
        ))
        .collect::<Vec<_>>();

        let mut bottlenecks = Vec::new();
        for bottleneck in prefill_score
            .bottlenecks
            .iter()
            .chain(decode_score.bottlenecks.iter())
            .chain(simulation.kv_bottlenecks.iter())
        {
            if !bottlenecks.contains(bottleneck) {
                bottlenecks.push(bottleneck.clone());
            }
        }
        for topology_bottleneck in &topology_bottlenecks {
            if topology_bottleneck.severity != "info"
                && !bottlenecks.contains(&topology_bottleneck.code)
            {
                bottlenecks.push(topology_bottleneck.code.clone());
            }
        }
        let kv_route_constraint_rejections = kv_route_constraint_rejections(
            &kv_route_topology_summary,
            &kv_route_resource_summary,
            context.traffic.kv_route_constraints,
        );
        if !kv_route_constraint_rejections.is_empty() {
            feasible = false;
            for rejection in kv_route_constraint_rejections {
                if !bottlenecks.contains(&rejection.resource) {
                    bottlenecks.push(rejection.resource.clone());
                }
                rejections.push(rejection);
            }
        }
        for rejection in &rejections {
            if !bottlenecks.contains(&rejection.resource) {
                bottlenecks.push(rejection.resource.clone());
            }
        }
        let capacity_rejections = capacity_rejections(&metrics, context.traffic);
        if !capacity_rejections.is_empty() {
            feasible = false;
            for rejection in capacity_rejections {
                if !bottlenecks.contains(&rejection.bottleneck) {
                    bottlenecks.push(rejection.bottleneck.clone());
                }
                rejections.push(rejection.into_serving_rejection());
            }
        }
        let slo_rejections = slo_rejections(
            &metrics,
            &simulation.metric_breakdowns,
            context.traffic,
            context.slo_policies,
        );
        if !slo_rejections.is_empty() {
            feasible = false;
            for rejection in slo_rejections {
                if !bottlenecks.contains(&rejection.resource) {
                    bottlenecks.push(rejection.resource.clone());
                }
                rejections.push(rejection);
            }
        }
        let prefill_memory = memory_headroom(
            context.cluster,
            context.model,
            context.request,
            InferencePhase::Prefill,
            &prefill_score,
            kv_block_tokens(context.traffic),
            context.calibration,
        );
        let decode_memory = memory_headroom(
            context.cluster,
            context.model,
            context.request,
            InferencePhase::Decode,
            &decode_score,
            kv_block_tokens(context.traffic),
            context.calibration,
        );
        if let Some(rejection) = memory_headroom_rejection("prefill", &prefill_memory) {
            feasible = false;
            if !bottlenecks.contains(&rejection.resource) {
                bottlenecks.push(rejection.resource.clone());
            }
            rejections.push(rejection);
        }
        if let Some(rejection) = memory_headroom_rejection("decode", &decode_memory) {
            feasible = false;
            if !bottlenecks.contains(&rejection.resource) {
                bottlenecks.push(rejection.resource.clone());
            }
            rejections.push(rejection);
        }
        let calibration_fits = serving_calibration_fits(
            &prefill_score,
            &decode_score,
            &decode_one_score,
            &simulation,
        );
        let phase_calibration = serving_phase_calibration(
            context.calibration_profile,
            &metrics,
            &simulation.request_observations,
            &calibration_fits,
        );
        let calibration_summary =
            serving_calibration_summary(&phase_calibration, &calibration_fits, &[]);
        let memory_pressure = memory_pressure_observations(
            &simulation.request_observations,
            &prefill_memory,
            &decode_memory,
        );
        if let Some(rejection) =
            memory_pressure_rejection(&memory_pressure, context.max_memory_pressure_fraction)
        {
            feasible = false;
            if !bottlenecks.contains(&rejection.resource) {
                bottlenecks.push(rejection.resource.clone());
            }
            rejections.push(rejection);
        }
        let hardware_footprint = serving_hardware_footprint(
            context.cluster,
            context.model,
            &prefill_score.placement,
            &decode_score.placement,
            metrics.throughput_tokens_per_s,
        );
        let cost_estimate = serving_cost_estimate(
            context.cluster,
            &prefill_score.placement,
            &decode_score.placement,
            &metrics,
            context.cost_model,
        );
        if let Some(rejection) =
            gpu_footprint_rejection(&hardware_footprint, context.max_unique_gpus)
        {
            feasible = false;
            if !bottlenecks.contains(&rejection.resource) {
                bottlenecks.push(rejection.resource.clone());
            }
            rejections.push(rejection);
        }
        if let Some(rejection) =
            throughput_floor_rejection(&metrics, context.min_throughput_tokens_per_s)
        {
            feasible = false;
            if !bottlenecks.contains(&rejection.resource) {
                bottlenecks.push(rejection.resource.clone());
            }
            rejections.push(rejection);
        }
        for rejection in metric_ceiling_rejections(&metrics, context.traffic.metric_ceilings) {
            feasible = false;
            if !bottlenecks.contains(&rejection.resource) {
                bottlenecks.push(rejection.resource.clone());
            }
            rejections.push(rejection);
        }
        let approximations = serving_approximations(
            context.traffic,
            context.pool,
            context.cluster,
            context.model,
            context.model_id,
            context.serving_stack,
            context.serving_runtime_features,
            context.calibration_profile,
            &prefill_score,
            &decode_score,
            &decode_one_score,
            &simulation,
            &calibration_fits,
            feasible,
        );
        let approximation_policy_violations = Vec::new();
        let approximation_summary =
            serving_approximation_summary(&approximations, &approximation_policy_violations);
        let rejected_reason = joined_rejection_reason(&rejections);
        let mut slo_miss_penalty_components =
            slo_miss_penalty_components_from_metrics(&metrics, context.slo_miss_penalty_weights);
        let traffic_class_slo_miss_penalties =
            traffic_class_slo_miss_penalties(context.traffic, &simulation.metric_breakdowns);
        for penalty in &traffic_class_slo_miss_penalties {
            slo_miss_penalty_components.add_assign(penalty.components);
        }
        let slo_miss_penalty_score = slo_miss_penalty_components.total;
        let service_backpressure_penalty_weight =
            context.traffic.service_backpressure_penalty_weight.max(0.0);
        let service_backpressure_penalty_score = service_backpressure_penalty_score(
            &simulation.service_observations,
            service_backpressure_penalty_weight,
        );
        let topology_risk_penalty_score = topology_risk_penalty_score(
            route_coverage,
            &topology_bottlenecks,
            context.topology_risk_penalty_weight,
        );
        let measurement_request_counts = measurement_window_request_counts(
            &simulation.request_observations,
            metrics.measurement_start_s,
            metrics.measurement_end_s,
        );
        let metric_source_counts = measurement_metric_source_counts(
            &simulation.request_observations,
            metrics.measurement_start_s,
            metrics.measurement_end_s,
        );
        let measurement_window = simulation
            .measurement_window
            .into_observation(measurement_request_counts, metric_source_counts);
        let objective_bottlenecks = serving_objective_bottleneck_summaries(
            context.objective,
            &metrics,
            &cost_estimate,
            &memory_pressure,
            &prefill_memory,
            &decode_memory,
            slo_miss_penalty_score,
            service_backpressure_penalty_score,
            topology_risk_penalty_score,
        );
        let bottleneck_summary = serving_bottleneck_summary(
            &rejections,
            &topology_bottlenecks,
            &memory_pressure,
            &simulation.resource_utilization,
            &simulation.phase_resource_utilization,
            &objective_bottlenecks,
        );
        let deployment_mode = ServingDeploymentMode::effective_for_pool(
            &context.pool.prefill_nodes,
            &context.pool.decode_nodes,
        );
        let candidate_id = serving_candidate_id(
            deployment_mode,
            context.pool,
            &prefill_score.config,
            &decode_score.config,
        );

        ScoredServingConfig {
            candidate_id,
            objective: context.objective,
            deployment_mode,
            slo_miss_penalty_weight: context.slo_miss_penalty_weight,
            slo_miss_penalty_weights: context.slo_miss_penalty_weights,
            slo_miss_penalty_components,
            traffic_class_slo_miss_penalties,
            slo_miss_penalty_score,
            service_backpressure_penalty_weight,
            service_backpressure_penalty_score,
            topology_risk_penalty_weight: context.topology_risk_penalty_weight,
            topology_risk_penalty_score,
            max_memory_pressure_fraction: context.max_memory_pressure_fraction,
            max_unique_gpus: context.max_unique_gpus,
            min_throughput_tokens_per_s: context.min_throughput_tokens_per_s,
            cost_model: context.cost_model.clone(),
            metric_ceilings: context.traffic.metric_ceilings,
            kv_route_constraints: context.traffic.kv_route_constraints,
            pool_label: context.pool.label.clone(),
            prefill_nodes: context.pool.prefill_nodes.clone(),
            decode_nodes: context.pool.decode_nodes.clone(),
            prefill_gpu_labels: context.pool.prefill_gpu_labels.clone(),
            decode_gpu_labels: context.pool.decode_gpu_labels.clone(),
            pool_topology: serving_pool_topology_summary(
                Some(context.cluster),
                &context.pool.prefill_nodes,
                &context.pool.decode_nodes,
            ),
            pool_search_summary: context.pool_search_summary.cloned(),
            route_coverage,
            prefill_config: prefill_score.config,
            decode_config: decode_score.config,
            prefill_memory,
            decode_memory,
            calibration_summary,
            calibration_fits,
            phase_calibration,
            calibration_gate_violations: Vec::new(),
            approximation_summary,
            approximations,
            approximation_policy_violations,
            prefill_score,
            decode_score,
            metrics,
            measurement_window,
            metric_breakdowns: simulation.metric_breakdowns,
            hardware_footprint,
            cost_estimate,
            memory_pressure,
            kv_route_resource_summary,
            kv_route_topology_summary,
            topology_bottlenecks,
            bottleneck_summary,
            pareto: ServingParetoFrontier::default(),
            request_observations: simulation.request_observations,
            decode_iterations: simulation.decode_iterations,
            node_capacity: simulation.node_capacity,
            gpu_capacity: simulation.gpu_capacity,
            traffic_class_capacity: simulation.traffic_class_capacity,
            service_observations: simulation.service_observations,
            worker_observations: simulation.worker_observations,
            scheduled_operations: simulation.scheduled_operations,
            resource_utilization: simulation.resource_utilization,
            phase_resource_utilization: simulation.phase_resource_utilization,
            feasible,
            bottlenecks,
            rejections,
            rejected_reason,
        }
    }

    fn rejected_pair_for_pool(
        pool: &ResolvedServingPool,
        context: RejectedPairContext,
        reason: String,
    ) -> ScoredServingConfig {
        let empty_config = ParallelismConfig {
            tensor_ranks: 0,
            pipeline_ranks: 0,
            expert_ranks: 0,
            data_ranks: 0,
        };
        let empty_score = Solver::score_config(
            &Cluster {
                nodes: Default::default(),
                node_groups: Default::default(),
                inter_node_topology: crate::types::fabric::inter_node::InterNodeTopology::Custom(
                    Default::default(),
                ),
            },
            &ModelSpec {
                layers: 0,
                hidden_size: 0,
                attention_heads: 1,
                kv_heads: 0,
                vocab_size: 0,
                parameters: Bytes::from_bytes(0),
                parameter_count: None,
                dtype: crate::workload::DType::Bf16,
                kv_dtype: None,
                experts: None,
            },
            &InferenceRequest {
                batch_size: 0,
                prompt_tokens: 0,
                decode_tokens: 0,
                max_sequence_tokens: 0,
                phase: InferencePhase::EndToEnd,
            },
            empty_config,
        );

        let deployment_mode =
            ServingDeploymentMode::effective_for_pool(&pool.prefill_nodes, &pool.decode_nodes);
        let candidate_id =
            serving_candidate_id(deployment_mode, pool, &empty_config, &empty_config);

        let rejections = vec![pool_rejection(pool, reason.clone())];
        let bottleneck_summary = serving_bottleneck_summary(&rejections, &[], &[], &[], &[], &[]);
        let approximations = Vec::new();
        let approximation_policy_violations = Vec::new();
        let approximation_summary =
            serving_approximation_summary(&approximations, &approximation_policy_violations);

        ScoredServingConfig {
            candidate_id,
            objective: context.objective,
            deployment_mode,
            slo_miss_penalty_weight: context.slo_miss_penalty_weight,
            slo_miss_penalty_weights: context.slo_miss_penalty_weights,
            slo_miss_penalty_components: ServingSloMissPenaltyComponents::default(),
            traffic_class_slo_miss_penalties: Vec::new(),
            slo_miss_penalty_score: 0.0,
            service_backpressure_penalty_weight: 0.0,
            service_backpressure_penalty_score: 0.0,
            topology_risk_penalty_weight: context.topology_risk_penalty_weight,
            topology_risk_penalty_score: 0.0,
            max_memory_pressure_fraction: context.max_memory_pressure_fraction,
            max_unique_gpus: context.max_unique_gpus,
            min_throughput_tokens_per_s: context.min_throughput_tokens_per_s,
            cost_model: context.cost_model,
            metric_ceilings: context.metric_ceilings,
            kv_route_constraints: context.kv_route_constraints,
            pool_label: pool.label.clone(),
            prefill_nodes: pool.prefill_nodes.clone(),
            decode_nodes: pool.decode_nodes.clone(),
            prefill_gpu_labels: pool.prefill_gpu_labels.clone(),
            decode_gpu_labels: pool.decode_gpu_labels.clone(),
            pool_topology: serving_pool_topology_summary(
                None,
                &pool.prefill_nodes,
                &pool.decode_nodes,
            ),
            pool_search_summary: context.pool_search_summary,
            route_coverage: ServingRouteCoverage::default(),
            prefill_config: empty_config,
            decode_config: empty_config,
            prefill_memory: ServingMemoryHeadroom::unavailable(),
            decode_memory: ServingMemoryHeadroom::unavailable(),
            calibration_summary: serving_calibration_summary(&[], &[], &[]),
            calibration_fits: Vec::new(),
            phase_calibration: Vec::new(),
            calibration_gate_violations: Vec::new(),
            approximation_summary,
            approximations,
            approximation_policy_violations,
            prefill_score: empty_score.clone(),
            decode_score: empty_score,
            metrics: ServingMetrics::rejected(),
            measurement_window: ServingMeasurementWindowObservation::rejected(),
            metric_breakdowns: Vec::new(),
            hardware_footprint: ServingHardwareFootprint::default(),
            cost_estimate: ServingCostEstimate::default(),
            memory_pressure: Vec::new(),
            kv_route_resource_summary: Vec::new(),
            kv_route_topology_summary: ServingKvRouteTopologySummary::default(),
            topology_bottlenecks: Vec::new(),
            bottleneck_summary,
            pareto: ServingParetoFrontier::default(),
            request_observations: Vec::new(),
            decode_iterations: Vec::new(),
            node_capacity: Vec::new(),
            gpu_capacity: Vec::new(),
            traffic_class_capacity: Vec::new(),
            service_observations: Vec::new(),
            worker_observations: Vec::new(),
            scheduled_operations: Vec::new(),
            resource_utilization: Vec::new(),
            phase_resource_utilization: Vec::new(),
            feasible: false,
            bottlenecks: Vec::new(),
            rejections,
            rejected_reason: Some(reason),
        }
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ServingWorkerRuntime {
    prefill_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    decode_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    kv_transfer_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
struct ServingSimulation {
    metrics: ServingMetrics,
    calibration_fits: Vec<CalibrationFitApplication>,
    measurement_window: MeasurementWindowSelection,
    metric_breakdowns: Vec<ServingMetricBreakdown>,
    request_observations: Vec<ServingRequestObservation>,
    decode_iterations: Vec<ServingDecodeIterationObservation>,
    node_capacity: Vec<ServingNodeCapacityObservation>,
    gpu_capacity: Vec<ServingGpuCapacityObservation>,
    traffic_class_capacity: Vec<ServingTrafficClassCapacityObservation>,
    service_observations: Vec<ServingServiceObservation>,
    worker_observations: Vec<ServingWorkerObservation>,
    scheduled_operations: Vec<ScheduledOperation>,
    resource_utilization: Vec<ResourceUtilization>,
    phase_resource_utilization: Vec<ServingPhaseResourceUtilization>,
    kv_bottlenecks: Vec<String>,
}

impl ServingSimulation {
    fn rejected() -> Self {
        Self {
            metrics: ServingMetrics::rejected(),
            calibration_fits: Vec::new(),
            measurement_window: MeasurementWindowSelection::rejected(),
            metric_breakdowns: Vec::new(),
            request_observations: Vec::new(),
            decode_iterations: Vec::new(),
            node_capacity: Vec::new(),
            gpu_capacity: Vec::new(),
            traffic_class_capacity: Vec::new(),
            service_observations: Vec::new(),
            worker_observations: Vec::new(),
            scheduled_operations: Vec::new(),
            resource_utilization: Vec::new(),
            phase_resource_utilization: Vec::new(),
            kv_bottlenecks: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule_serving_simulation(
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
) -> ServingSimulation {
    let calibration = calibration.sanitized();
    let scheduled_requests = traffic.request_count(calibration);
    let arrival_times = traffic.arrival_times(scheduled_requests, calibration);
    let base_decode_tokens = request.decode_tokens.max(1);
    let decode_total_s = decode_score.estimated_latency_s;
    let decode_one_s = decode_one_score.estimated_latency_s;
    let decode_tail_token_s = if base_decode_tokens > 1 {
        ((decode_total_s - decode_one_s).max(0.0) / f64::from(base_decode_tokens - 1))
            .max(decode_one_s)
    } else {
        decode_one_s
    };
    let decode_tail_scale = decode_tail_token_s / decode_one_s.max(1e-12);

    let prefill_resource_base_node = single_placement_node(prefill_score);
    let prefill_placement_nodes = placement_nodes(prefill_score);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let decode_placement_nodes = placement_nodes(decode_one_score);
    let mut scheduler = ResourceScheduler::new();
    let mut decode_states = Vec::new();
    let mut decode_iterations = Vec::new();
    let mut kv_bottlenecks = Vec::new();
    let mut routing_load = RoutingLoad::default();
    let mut worker_runtime = ServingWorkerRuntime::default();

    for request_idx in 0..scheduled_requests {
        let arrival_s = arrival_times
            .get(request_idx as usize)
            .copied()
            .unwrap_or(0.0);
        let request_shape = traffic.request_at(request, request_idx);
        let prefix_cache_hit_tokens =
            traffic.prefix_cache_hit_tokens(request_idx, request_shape.prompt_tokens);
        let effective_prefill_tokens = request_shape
            .prompt_tokens
            .saturating_sub(prefix_cache_hit_tokens);
        let route = route_request(
            cluster,
            model,
            &request_shape,
            effective_prefill_tokens,
            traffic,
            request_idx,
            arrival_s,
            prefill_nodes,
            decode_nodes,
            prefill_resource_base_node,
            &prefill_placement_nodes,
            decode_resource_base_node,
            &decode_placement_nodes,
            prefill_score,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            request.prompt_tokens,
            calibration,
            calibration_profile,
            &mut routing_load,
        );
        let decode_tokens = request_shape.decode_tokens.max(1);
        let max_sequence_tokens = request_shape
            .max_sequence_tokens
            .max(request_shape.prompt_tokens.saturating_add(decode_tokens))
            .max(1);
        let kv_block_tokens = kv_block_tokens(traffic);
        let kv_allocation = sequence_kv_allocation(
            request_shape.batch_size.max(1),
            max_sequence_tokens,
            kv_block_tokens,
        );
        decode_states.push(DecodeRequestState {
            request_idx,
            request_id: traffic.request_id(request_idx),
            tenant: traffic.tenant(request_idx),
            model_id: traffic.model_id(request_idx),
            traffic_class: traffic.traffic_class_name(request_idx),
            shape_profile: traffic.shape_profile_name(request_idx),
            cache_key: traffic.cache_key(request_idx),
            prefill_node: route.prefill_node,
            decode_node: route.decode_node,
            prefill_route_nodes: route.prefill_route_nodes,
            prefill_route_gpus: route.prefill_route_gpus,
            decode_route_nodes: route.decode_route_nodes,
            decode_route_gpus: route.decode_route_gpus,
            routing_policy: route.routing.policy,
            routing_candidate_count: route.routing.candidate_count,
            routing_routable_candidate_count: route.routing.routable_candidate_count,
            routing_estimated_e2el_s: route.routing.estimated_e2el_s,
            routing_estimated_kv_transfer_s: route.routing.estimated_kv_transfer_s,
            routing_estimated_kv_resource_wait_s: route.routing.estimated_kv_resource_wait_s,
            routing_estimated_prefill_wait_s: route.routing.estimated_prefill_wait_s,
            routing_estimated_decode_wait_s: route.routing.estimated_decode_wait_s,
            routing_reason: route.routing.reason,
            routing_candidates: route.routing.candidates,
            arrival_s,
            priority: traffic.effective_priority(request_idx),
            batch_size: request_shape.batch_size.max(1),
            prompt_tokens: request_shape.prompt_tokens.max(1),
            prefix_cache_hit_tokens,
            effective_prefill_tokens,
            remaining_prefill_tokens: effective_prefill_tokens,
            prefill_chunks: 0,
            decode_tokens,
            slo: traffic.effective_slo(request_idx),
            max_queue_delay_s: traffic.effective_max_queue_delay_s(request_idx),
            max_kv_queue_delay_s: traffic.effective_max_kv_queue_delay_s(request_idx),
            max_decode_queue_delay_s: traffic.effective_max_decode_queue_delay_s(request_idx),
            max_decode_iteration_queue_delay_s: traffic
                .effective_max_decode_iteration_queue_delay_s(request_idx),
            request_timeout_s: traffic.effective_request_timeout_s(request_idx),
            deadline_s: traffic.deadline_s(request_idx, arrival_s),
            cancellation_s: traffic.cancellation_s(request_idx, arrival_s),
            remaining_tokens: decode_tokens,
            emitted_tokens: 0,
            max_sequence_tokens,
            kv_block_tokens,
            kv_cache_blocks: kv_allocation.blocks,
            kv_allocated_tokens: kv_allocation.allocated_tokens,
            kv_fragmentation_tokens: kv_allocation.fragmentation_tokens,
            prefill_scheduled: false,
            dependencies: Vec::new(),
            kv_start_s: 0.0,
            kv_finish_s: 0.0,
            first_decode_start_s: None,
            first_decode_finish_s: None,
            last_decode_finish_s: None,
            decode_token_start_s: Vec::with_capacity(decode_tokens as usize),
            decode_token_finish_s: Vec::with_capacity(decode_tokens as usize),
            prefill_start_s: 0.0,
            prefill_finish_s: 0.0,
            prefill_worker_queue_s: 0.0,
            prefill_resource_queue_s: 0.0,
            decode_worker_queue_s: 0.0,
            decode_resource_queue_s: 0.0,
            kv_transfer_bytes: 0,
            kv_transfer_bottlenecks: Vec::new(),
            kv_transfer_paths: Vec::new(),
            kv_transfer_resources: Vec::new(),
            kv_transfer_resource_dependencies: Vec::new(),
            kv_worker_queue_s: 0.0,
            kv_resource_queue_s: 0.0,
            kv_transfer_s: 0.0,
            kv_transfer_fit: None,
            prefill_token_spans: Vec::new(),
            worker_assignments: Vec::new(),
            status: ServingRequestStatus::Pending,
            status_time_s: None,
            failure_reason: None,
            failure_rejection: None,
        });
    }

    schedule_prefills(
        &mut scheduler,
        &mut worker_runtime,
        &mut decode_states,
        prefill_score,
        traffic,
        prefill_resource_base_node,
        request.batch_size,
        request.prompt_tokens,
    );
    for state in &mut decode_states {
        if state.status == ServingRequestStatus::Pending
            && let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= state.prefill_finish_s + 1e-12
        {
            cancel_request(state, cancellation_s, "request cancelled during prefill");
        }
    }

    for state in &mut decode_states {
        if state.status != ServingRequestStatus::Pending {
            continue;
        }
        let request_shape = InferenceRequest {
            batch_size: state.batch_size,
            prompt_tokens: state.prompt_tokens,
            decode_tokens: state.decode_tokens,
            max_sequence_tokens: state.max_sequence_tokens,
            phase: request.phase,
        };
        let kv_bytes = kv_transfer_bytes_for_routes(
            model,
            &request_shape,
            &state.prefill_route_nodes,
            &state.decode_route_nodes,
            &state.prefill_route_gpus,
            &state.decode_route_gpus,
        );
        let (kv_cost, kv_fit) = Solver::estimate_transfer_between_gpus_with_observation(
            cluster,
            &state.prefill_route_gpus,
            &state.decode_route_gpus,
            kv_bytes,
            SolverOptions {
                calibration,
                calibration_profile,
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );
        for bottleneck in &kv_cost.bottlenecks {
            if !kv_bottlenecks.contains(bottleneck) {
                kv_bottlenecks.push(bottleneck.clone());
            }
        }
        let kv_paths = kv_transfer_paths(
            cluster,
            &state.prefill_route_gpus,
            &state.decode_route_gpus,
            kv_bytes,
        );
        let kv_resources = kv_transfer_scheduler_resources(&kv_paths, &kv_cost.bottlenecks);
        let phase_ready_s = operation_finish_s(&scheduler, &state.dependencies);
        let kv_worker_ready_s = if kv_bytes.as_bytes() > 0 {
            kv_transfer_worker_slots_per_gpu(traffic)
                .map(|slots| kv_transfer_worker_ready_s(state, &worker_runtime, slots))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let kv_candidate_ready_s = phase_ready_s.max(kv_worker_ready_s);
        let kv_preview = scheduler.preview(
            kv_cost.total_s,
            kv_candidate_ready_s,
            &state.dependencies,
            kv_resources.clone(),
        );
        state.kv_start_s = kv_preview.start_s;
        state.kv_worker_queue_s = finite_or_zero(kv_worker_ready_s - phase_ready_s);
        state.kv_resource_queue_s = finite_or_zero(kv_preview.start_s - kv_candidate_ready_s);
        state.kv_transfer_bytes = kv_bytes.as_bytes();
        state.kv_transfer_bottlenecks = kv_cost.bottlenecks.clone();
        state.kv_transfer_paths = kv_paths.clone();
        state.kv_transfer_resources = kv_preview.resources.clone();
        state.kv_transfer_resource_dependencies = kv_preview.resource_dependencies.clone();
        state.kv_transfer_s = kv_cost.total_s;
        state.kv_transfer_fit = kv_fit.clone();
        let kv_queue_s = finite_or_zero(kv_preview.start_s - phase_ready_s);
        if let Some(max_kv_queue_delay_s) = state.max_kv_queue_delay_s
            && kv_queue_s > max_kv_queue_delay_s + 1e-12
        {
            reject_kv_queue_admission(state, phase_ready_s, kv_queue_s, max_kv_queue_delay_s);
            continue;
        }
        let kv_id = scheduler.schedule(
            format!(
                "request {} kv-transfer {}->{}",
                state.request_idx, state.prefill_node, state.decode_node
            ),
            kv_cost.total_s,
            kv_candidate_ready_s,
            &state.dependencies,
            kv_resources,
        );
        state.kv_finish_s = operation_finish_s(&scheduler, &[kv_id]);
        if kv_bytes.as_bytes() > 0
            && let Some(slots) = kv_transfer_worker_slots_per_gpu(traffic)
        {
            let assignments = assign_kv_transfer_workers_ready(
                &mut worker_runtime,
                state,
                slots,
                state.kv_start_s,
                state.kv_finish_s,
                &[kv_id],
            );
            state.worker_assignments.extend(assignments);
        }
        state.dependencies = vec![kv_id];
        if let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= state.kv_finish_s + 1e-12
        {
            cancel_request(
                state,
                cancellation_s,
                "request cancelled during KV transfer",
            );
        }
    }
    apply_decode_capacity_admission(
        &mut decode_states,
        traffic,
        decode_one_score,
        decode_tail_scale,
        request.batch_size,
        decode_worker_slots_per_gpu(traffic),
    );

    match traffic.decode_batching {
        ServingDecodeBatching::Independent => schedule_independent_decodes(
            &mut scheduler,
            &mut worker_runtime,
            &mut decode_states,
            &mut decode_iterations,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            decode_worker_slots_per_gpu(traffic),
        ),
        ServingDecodeBatching::Continuous { max_batch_tokens } => schedule_continuous_decodes(
            &mut scheduler,
            &mut worker_runtime,
            &mut decode_states,
            &mut decode_iterations,
            decode_one_score,
            decode_tail_scale,
            request.batch_size,
            max_batch_tokens,
            decode_worker_slots_per_gpu(traffic),
        ),
    }
    apply_terminal_statuses(&mut decode_states, traffic);

    let mut calibration_fits = decode_states
        .iter()
        .filter_map(|state| state.kv_transfer_fit.clone())
        .collect::<Vec<_>>();
    let observations: Vec<_> = decode_states
        .iter()
        .map(ServingRequestObservation::from_decode_state)
        .collect();
    let makespan_s = scheduler.makespan_s();
    let measurement_window =
        effective_measurement_window(traffic, &observations, scheduler.operations(), makespan_s);
    let measurement_start_s = measurement_window.start_s;
    let measurement_end_s = measurement_window.end_s;
    let measured_all_observations = observations
        .iter()
        .filter(|observation| {
            observation.arrival_s + 1e-12 >= measurement_start_s
                && observation.arrival_s <= measurement_end_s + 1e-12
        })
        .cloned()
        .collect::<Vec<_>>();
    let measured_observations = measured_all_observations
        .iter()
        .filter(|observation| observation.status.is_completed())
        .cloned()
        .collect::<Vec<_>>();
    let measured_request_indices = measured_observations
        .iter()
        .map(|observation| observation.request_idx)
        .collect::<BTreeSet<_>>();
    let measured_decode_iteration_latencies = decode_iterations
        .iter()
        .filter(|iteration| {
            iteration
                .request_indices
                .iter()
                .any(|request_idx| measured_request_indices.contains(request_idx))
        })
        .map(|iteration| iteration.latency_s)
        .filter(|latency_s| latency_s.is_finite())
        .collect::<Vec<_>>();
    let measured_output_tokens = measured_observations
        .iter()
        .fold(0_u64, |total, observation| {
            total.saturating_add(observation_output_tokens(observation))
        });
    let measured_prompt_tokens = measured_observations
        .iter()
        .map(|observation| u64::from(observation.batch_size) * u64::from(observation.prompt_tokens))
        .sum::<u64>();
    let measured_prefix_cache_hit_tokens = measured_observations
        .iter()
        .map(|observation| {
            u64::from(observation.batch_size) * u64::from(observation.prefix_cache_hit_tokens)
        })
        .sum::<u64>();
    let measured_effective_prefill_tokens = measured_observations
        .iter()
        .map(|observation| {
            u64::from(observation.batch_size) * u64::from(observation.effective_prefill_tokens)
        })
        .sum::<u64>();
    let measured_prefill_chunks = measured_observations
        .iter()
        .map(|observation| u64::from(observation.prefill_chunks))
        .sum::<u64>();
    let measured_prefix_cache_hit_rate = if measured_prompt_tokens > 0 {
        measured_prefix_cache_hit_tokens as f64 / measured_prompt_tokens as f64
    } else {
        0.0
    };
    let measurement_duration_s = (measurement_end_s - measurement_start_s).max(0.0);
    let throughput_tokens_per_s = if measurement_duration_s > 0.0 {
        measured_output_tokens as f64 / measurement_duration_s
    } else {
        0.0
    };

    let ttft = collect_metric(&measured_observations, |observation| observation.ttft_s);
    let tpot = collect_metric(&measured_observations, |observation| observation.tpot_s);
    let itl = measured_observations
        .iter()
        .flat_map(|observation| observation.inter_token_latency_s.iter().copied())
        .collect::<Vec<_>>();
    let e2el = collect_metric(&measured_observations, |observation| observation.e2el_s);
    let service = collect_metric(&measured_observations, |observation| observation.service_s);
    let prefill = collect_metric(&measured_observations, |observation| observation.prefill_s);
    let kv_transfer = collect_metric(&measured_observations, |observation| {
        observation.kv_transfer_s
    });
    let kv_queue = collect_metric(&measured_observations, |observation| observation.kv_queue_s);
    let kv_worker_queue = collect_metric(&measured_observations, |observation| {
        observation.kv_worker_queue_s
    });
    let kv_resource_queue = collect_metric(&measured_observations, |observation| {
        observation.kv_resource_queue_s
    });
    let decode_queue = collect_metric(&measured_observations, |observation| {
        observation.decode_queue_s
    });
    let prefill_worker_queue = collect_metric(&measured_observations, |observation| {
        observation.prefill_worker_queue_s
    });
    let prefill_resource_queue = collect_metric(&measured_observations, |observation| {
        observation.prefill_resource_queue_s
    });
    let decode_worker_queue = collect_metric(&measured_observations, |observation| {
        observation.decode_worker_queue_s
    });
    let decode_resource_queue = collect_metric(&measured_observations, |observation| {
        observation.decode_resource_queue_s
    });
    let decode = collect_metric(&measured_observations, |observation| observation.decode_s);
    let queue_delay = collect_metric(&measured_observations, |observation| {
        observation.queue_delay_s
    });
    let capacity = capacity_profile(&decode_states, traffic);
    let admitted_requests = decode_states
        .iter()
        .filter(|state| state.status.is_admitted())
        .count()
        .min(u32::MAX as usize) as u32;
    let completed_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::Completed)
        .count()
        .min(u32::MAX as usize) as u32;
    let rejected_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::RejectedAdmission)
        .count()
        .min(u32::MAX as usize) as u32;
    let timed_out_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::TimedOut)
        .count()
        .min(u32::MAX as usize) as u32;
    let cancelled_requests = decode_states
        .iter()
        .filter(|state| state.status == ServingRequestStatus::Cancelled)
        .count()
        .min(u32::MAX as usize) as u32;
    let deadline_constrained_requests = measured_all_observations
        .iter()
        .filter(|observation| observation.deadline_s.is_some())
        .count();
    let deadline_missed_requests = measured_all_observations
        .iter()
        .filter(|observation| observation.deadline_missed)
        .count();
    let deadline_miss_rate = if deadline_constrained_requests > 0 {
        deadline_missed_requests as f64 / deadline_constrained_requests as f64
    } else {
        0.0
    };
    let ttft_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.ttft_s,
        |observation| observation.ttft_s,
    );
    let tpot_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.tpot_s,
        |observation| observation.tpot_s,
    );
    let itl_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.itl_s,
        |observation| observation.itl_s,
    );
    let e2el_slo_counts = observation_slo_miss_counts(
        &measured_all_observations,
        |observation| observation.slo.e2el_s,
        |observation| observation.e2el_s,
    );

    let mut metrics = ServingMetrics {
        ttft_s: mean(&ttft),
        ttft_p50_s: percentile(ttft.clone(), 0.50),
        ttft_p90_s: percentile(ttft.clone(), 0.90),
        ttft_p95_s: percentile(ttft.clone(), 0.95),
        ttft_p99_s: percentile(ttft.clone(), 0.99),
        ttft_max_s: max_value(&ttft),
        ttft_slo_constrained_requests: ttft_slo_counts.constrained,
        ttft_slo_missed_requests: ttft_slo_counts.missed,
        ttft_slo_miss_rate: ttft_slo_counts.miss_rate,
        tpot_s: mean(&tpot),
        tpot_p50_s: percentile(tpot.clone(), 0.50),
        tpot_p90_s: percentile(tpot.clone(), 0.90),
        tpot_p95_s: percentile(tpot.clone(), 0.95),
        tpot_p99_s: percentile(tpot.clone(), 0.99),
        tpot_max_s: max_value(&tpot),
        tpot_slo_constrained_requests: tpot_slo_counts.constrained,
        tpot_slo_missed_requests: tpot_slo_counts.missed,
        tpot_slo_miss_rate: tpot_slo_counts.miss_rate,
        itl_s: mean(&itl),
        itl_p50_s: percentile(itl.clone(), 0.50),
        itl_p90_s: percentile(itl.clone(), 0.90),
        itl_p95_s: percentile(itl.clone(), 0.95),
        itl_p99_s: percentile(itl.clone(), 0.99),
        itl_max_s: max_value(&itl),
        itl_slo_constrained_requests: itl_slo_counts.constrained,
        itl_slo_missed_requests: itl_slo_counts.missed,
        itl_slo_miss_rate: itl_slo_counts.miss_rate,
        decode_iterations: measured_decode_iteration_latencies
            .len()
            .min(u64::MAX as usize) as u64,
        decode_iteration_s: mean(&measured_decode_iteration_latencies),
        decode_iteration_p50_s: percentile(measured_decode_iteration_latencies.clone(), 0.50),
        decode_iteration_p90_s: percentile(measured_decode_iteration_latencies.clone(), 0.90),
        decode_iteration_p95_s: percentile(measured_decode_iteration_latencies.clone(), 0.95),
        decode_iteration_p99_s: percentile(measured_decode_iteration_latencies.clone(), 0.99),
        decode_iteration_max_s: max_value(&measured_decode_iteration_latencies),
        throughput_tokens_per_s,
        e2el_s: mean(&e2el),
        e2el_p50_s: percentile(e2el.clone(), 0.50),
        e2el_p90_s: percentile(e2el.clone(), 0.90),
        e2el_p95_s: percentile(e2el.clone(), 0.95),
        e2el_p99_s: percentile(e2el.clone(), 0.99),
        e2el_max_s: max_value(&e2el),
        e2el_slo_constrained_requests: e2el_slo_counts.constrained,
        e2el_slo_missed_requests: e2el_slo_counts.missed,
        e2el_slo_miss_rate: e2el_slo_counts.miss_rate,
        deadline_miss_rate,
        service_s: mean(&service),
        prefill_s: mean(&prefill),
        prefill_chunks: measured_prefill_chunks,
        prompt_tokens: measured_prompt_tokens,
        prefix_cache_hit_tokens: measured_prefix_cache_hit_tokens,
        effective_prefill_tokens: measured_effective_prefill_tokens,
        prefix_cache_hit_rate: measured_prefix_cache_hit_rate,
        kv_transfer_s: mean(&kv_transfer),
        kv_queue_s: mean(&kv_queue),
        kv_worker_queue_s: mean(&kv_worker_queue),
        kv_resource_queue_s: mean(&kv_resource_queue),
        decode_queue_s: mean(&decode_queue),
        prefill_worker_queue_s: mean(&prefill_worker_queue),
        prefill_resource_queue_s: mean(&prefill_resource_queue),
        decode_worker_queue_s: mean(&decode_worker_queue),
        decode_resource_queue_s: mean(&decode_resource_queue),
        decode_s: mean(&decode),
        queue_delay_s: mean(&queue_delay),
        queue_delay_p90_s: percentile(queue_delay.clone(), 0.90),
        queue_delay_p95_s: percentile(queue_delay.clone(), 0.95),
        queue_delay_max_s: max_value(&queue_delay),
        peak_prefill_tokens: capacity.peak_prefill_tokens,
        peak_prefill_tokens_per_node: capacity.peak_prefill_tokens_per_node,
        peak_prefill_tokens_per_gpu: capacity.peak_prefill_tokens_per_gpu,
        peak_decode_sequences: capacity.peak_decode_sequences,
        peak_resident_tokens: capacity.peak_resident_tokens,
        peak_decode_sequences_per_node: capacity.peak_decode_sequences_per_node,
        peak_resident_tokens_per_node: capacity.peak_resident_tokens_per_node,
        peak_decode_sequences_per_gpu: capacity.peak_decode_sequences_per_gpu,
        peak_resident_tokens_per_gpu: capacity.peak_resident_tokens_per_gpu,
        peak_kv_blocks: capacity.peak_kv_blocks,
        peak_allocated_kv_tokens: capacity.peak_allocated_kv_tokens,
        peak_kv_fragmentation_tokens: capacity.peak_kv_fragmentation_tokens,
        peak_kv_block_table_bytes: capacity.peak_kv_block_table_bytes,
        peak_kv_blocks_per_node: capacity.peak_kv_blocks_per_node,
        peak_allocated_kv_tokens_per_node: capacity.peak_allocated_kv_tokens_per_node,
        peak_kv_fragmentation_tokens_per_node: capacity.peak_kv_fragmentation_tokens_per_node,
        peak_kv_block_table_bytes_per_node: capacity.peak_kv_block_table_bytes_per_node,
        peak_kv_blocks_per_gpu: capacity.peak_kv_blocks_per_gpu,
        peak_allocated_kv_tokens_per_gpu: capacity.peak_allocated_kv_tokens_per_gpu,
        peak_kv_fragmentation_tokens_per_gpu: capacity.peak_kv_fragmentation_tokens_per_gpu,
        peak_kv_block_table_bytes_per_gpu: capacity.peak_kv_block_table_bytes_per_gpu,
        decode_sequence_utilization: capacity.decode_sequence_utilization,
        resident_token_utilization: capacity.resident_token_utilization,
        kv_block_utilization: capacity.kv_block_utilization,
        decode_sequence_per_node_utilization: capacity.decode_sequence_per_node_utilization,
        resident_token_per_node_utilization: capacity.resident_token_per_node_utilization,
        kv_block_per_node_utilization: capacity.kv_block_per_node_utilization,
        decode_sequence_per_gpu_utilization: capacity.decode_sequence_per_gpu_utilization,
        resident_token_per_gpu_utilization: capacity.resident_token_per_gpu_utilization,
        kv_block_per_gpu_utilization: capacity.kv_block_per_gpu_utilization,
        scheduled_makespan_s: makespan_s,
        scheduled_requests,
        admitted_requests,
        completed_requests,
        rejected_requests,
        timed_out_requests,
        cancelled_requests,
        deadline_constrained_requests: deadline_constrained_requests.min(u32::MAX as usize) as u32,
        deadline_missed_requests: deadline_missed_requests.min(u32::MAX as usize) as u32,
        measured_requests: measured_observations.len().min(u32::MAX as usize) as u32,
        measurement_start_s,
        measurement_end_s,
    };
    calibration_fits.extend(apply_serving_metric_calibration_fits(
        &mut metrics,
        calibration_profile,
        model,
        request,
        traffic,
        prefill_score.config,
        decode_score.config,
    ));
    let metric_breakdowns =
        metric_breakdowns(&observations, measurement_start_s, measurement_end_s);
    let worker_observations = worker_observations(&observations, &capacity.gpus, traffic);
    let service_observations = service_observations(
        traffic,
        prefill_nodes,
        decode_nodes,
        prefill_score,
        decode_score,
        &observations,
        &worker_observations,
    );
    let scheduled_operations = scheduler.operations().to_vec();
    let resource_utilization = resource_utilization(&scheduled_operations, makespan_s);
    let phase_resource_utilization = phase_resource_utilization(&scheduled_operations, makespan_s);

    ServingSimulation {
        metrics,
        calibration_fits,
        measurement_window,
        metric_breakdowns,
        request_observations: observations,
        decode_iterations,
        node_capacity: capacity.nodes,
        gpu_capacity: capacity.gpus,
        traffic_class_capacity: capacity.traffic_classes,
        service_observations,
        worker_observations,
        scheduled_operations,
        resource_utilization,
        phase_resource_utilization,
        kv_bottlenecks,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CapacityProfile {
    peak_prefill_tokens: u64,
    peak_prefill_tokens_per_node: u64,
    peak_prefill_tokens_per_gpu: u64,
    peak_decode_sequences: u32,
    peak_resident_tokens: u64,
    peak_decode_sequences_per_node: u32,
    peak_resident_tokens_per_node: u64,
    peak_decode_sequences_per_gpu: u32,
    peak_resident_tokens_per_gpu: u64,
    peak_kv_blocks: u64,
    peak_allocated_kv_tokens: u64,
    peak_kv_fragmentation_tokens: u64,
    peak_kv_block_table_bytes: u64,
    peak_kv_blocks_per_node: u64,
    peak_allocated_kv_tokens_per_node: u64,
    peak_kv_fragmentation_tokens_per_node: u64,
    peak_kv_block_table_bytes_per_node: u64,
    peak_kv_blocks_per_gpu: u64,
    peak_allocated_kv_tokens_per_gpu: u64,
    peak_kv_fragmentation_tokens_per_gpu: u64,
    peak_kv_block_table_bytes_per_gpu: u64,
    decode_sequence_utilization: f64,
    resident_token_utilization: f64,
    kv_block_utilization: f64,
    decode_sequence_per_node_utilization: f64,
    resident_token_per_node_utilization: f64,
    kv_block_per_node_utilization: f64,
    decode_sequence_per_gpu_utilization: f64,
    resident_token_per_gpu_utilization: f64,
    kv_block_per_gpu_utilization: f64,
    nodes: Vec<ServingNodeCapacityObservation>,
    gpus: Vec<ServingGpuCapacityObservation>,
    traffic_classes: Vec<ServingTrafficClassCapacityObservation>,
}

#[derive(Clone, Debug, PartialEq)]
struct CapacityRejection {
    reason: String,
    bottleneck: String,
    resource: String,
    code: String,
    observed: f64,
    limit: f64,
    unit: String,
}

impl CapacityRejection {
    fn into_serving_rejection(self) -> ServingRejection {
        let remediation = capacity_remediation(&self.code).map(str::to_string);
        let phase = if self.code.starts_with("prefill_") {
            "prefill"
        } else {
            "decode"
        };
        ServingRejection {
            phase: phase.to_string(),
            category: "capacity".to_string(),
            resource: self.resource,
            code: self.code,
            observed: Some(self.observed),
            limit: Some(self.limit),
            unit: Some(self.unit),
            remediation,
            message: self.reason,
        }
    }
}

fn capacity_remediation(code: &str) -> Option<&'static str> {
    match code {
        "prefill_capacity_exceeded"
        | "prefill_capacity_per_node_exceeded"
        | "prefill_capacity_per_gpu_exceeded" => Some(
            "increase prefill capacity, add prefill workers, lower prefill batch/chunk tokens, or reduce prompt/batch concurrency",
        ),
        "decode_capacity_exceeded" | "decode_capacity_per_node_exceeded" => Some(
            "increase decode replicas or decode sequence capacity, or lower arrival concurrency",
        ),
        "decode_capacity_per_gpu_exceeded" => Some(
            "increase decode GPU capacity, use more decode tensor/data ranks, or lower arrival concurrency",
        ),
        "kv_residency_capacity_exceeded"
        | "kv_residency_capacity_per_node_exceeded"
        | "kv_residency_capacity_per_gpu_exceeded" => Some(
            "increase KV residency capacity, add decode workers, or reduce max sequence/batch size",
        ),
        "kv_block_capacity_exceeded"
        | "kv_block_capacity_per_node_exceeded"
        | "kv_block_capacity_per_gpu_exceeded" => Some(
            "increase KV block capacity, add decode workers, increase KV block budget, or reduce max sequence/batch size",
        ),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PrefillTokenSpan {
    start_s: f64,
    finish_s: f64,
    tokens: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct PrefillAdmissionSpan {
    finish_s: f64,
    traffic_class: Option<String>,
    tokens: u64,
    node_tokens: Vec<(NodeId, u64)>,
    gpu_tokens: Vec<(GpuAddr, u64)>,
}

impl PrefillAdmissionSpan {
    fn for_state_chunk(
        state: &DecodeRequestState,
        chunk_tokens: Option<u32>,
        finish_s: f64,
    ) -> Self {
        let tokens = prefill_work_tokens(state, chunk_tokens);
        Self::for_state_tokens(state, tokens, finish_s)
    }

    fn for_state_tokens(state: &DecodeRequestState, tokens: u64, finish_s: f64) -> Self {
        Self {
            finish_s,
            traffic_class: state.traffic_class.clone(),
            tokens,
            node_tokens: prefill_owner_nodes(state)
                .into_iter()
                .map(|node_id| (node_id, tokens))
                .collect(),
            gpu_tokens: prefill_owner_gpus(state)
                .into_iter()
                .map(|gpu| (gpu, tokens))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct DecodeRequestState {
    request_idx: u32,
    request_id: Option<String>,
    tenant: Option<String>,
    model_id: Option<String>,
    traffic_class: Option<String>,
    shape_profile: Option<String>,
    cache_key: Option<String>,
    prefill_node: NodeId,
    decode_node: NodeId,
    prefill_route_nodes: Vec<NodeId>,
    prefill_route_gpus: Vec<GpuAddr>,
    decode_route_nodes: Vec<NodeId>,
    decode_route_gpus: Vec<GpuAddr>,
    routing_policy: ServingRoutingPolicy,
    routing_candidate_count: u32,
    routing_routable_candidate_count: u32,
    routing_estimated_e2el_s: f64,
    routing_estimated_kv_transfer_s: f64,
    routing_estimated_kv_resource_wait_s: f64,
    routing_estimated_prefill_wait_s: f64,
    routing_estimated_decode_wait_s: f64,
    routing_reason: String,
    routing_candidates: Vec<ServingRouteCandidateObservation>,
    arrival_s: f64,
    priority: i32,
    batch_size: u32,
    prompt_tokens: u32,
    prefix_cache_hit_tokens: u32,
    effective_prefill_tokens: u32,
    remaining_prefill_tokens: u32,
    prefill_chunks: u32,
    decode_tokens: u32,
    slo: ServingRequestSlo,
    max_queue_delay_s: Option<f64>,
    max_kv_queue_delay_s: Option<f64>,
    max_decode_queue_delay_s: Option<f64>,
    max_decode_iteration_queue_delay_s: Option<f64>,
    request_timeout_s: Option<f64>,
    deadline_s: Option<f64>,
    cancellation_s: Option<f64>,
    remaining_tokens: u32,
    emitted_tokens: u32,
    max_sequence_tokens: u32,
    kv_block_tokens: u32,
    kv_cache_blocks: u64,
    kv_allocated_tokens: u64,
    kv_fragmentation_tokens: u64,
    prefill_scheduled: bool,
    dependencies: Vec<usize>,
    kv_start_s: f64,
    kv_finish_s: f64,
    first_decode_start_s: Option<f64>,
    first_decode_finish_s: Option<f64>,
    last_decode_finish_s: Option<f64>,
    decode_token_start_s: Vec<f64>,
    decode_token_finish_s: Vec<f64>,
    prefill_start_s: f64,
    prefill_finish_s: f64,
    prefill_worker_queue_s: f64,
    prefill_resource_queue_s: f64,
    decode_worker_queue_s: f64,
    decode_resource_queue_s: f64,
    kv_transfer_bytes: u64,
    kv_transfer_bottlenecks: Vec<String>,
    kv_transfer_paths: Vec<ServingKvTransferPathObservation>,
    kv_transfer_resources: Vec<String>,
    kv_transfer_resource_dependencies: Vec<usize>,
    kv_worker_queue_s: f64,
    kv_resource_queue_s: f64,
    kv_transfer_s: f64,
    kv_transfer_fit: Option<CalibrationFitApplication>,
    prefill_token_spans: Vec<PrefillTokenSpan>,
    worker_assignments: Vec<ServingWorkerAssignmentObservation>,
    status: ServingRequestStatus,
    status_time_s: Option<f64>,
    failure_reason: Option<String>,
    failure_rejection: Option<ServingRejection>,
}

fn compare_request_priority(
    states: &[DecodeRequestState],
    left: usize,
    right: usize,
) -> std::cmp::Ordering {
    states[right]
        .priority
        .cmp(&states[left].priority)
        .then_with(|| states[left].arrival_s.total_cmp(&states[right].arrival_s))
        .then_with(|| states[left].request_idx.cmp(&states[right].request_idx))
}

#[allow(clippy::too_many_arguments)]
fn schedule_prefills(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    prefill_score: &ScoredParallelismConfig,
    traffic: &ServingTraffic,
    prefill_resource_base_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
) {
    let prefill_worker_slots_per_gpu = prefill_worker_slots_per_gpu(traffic);
    for state in states.iter_mut() {
        if state.status == ServingRequestStatus::Pending && state.remaining_prefill_tokens == 0 {
            state.prefill_scheduled = true;
            state.prefill_start_s = state.arrival_s;
            state.prefill_finish_s = state.arrival_s;
            state.dependencies.clear();
        }
    }

    let mut active_prefills = Vec::new();

    match traffic.prefill_batching {
        ServingPrefillBatching::Independent => schedule_independent_prefills(
            scheduler,
            worker_runtime,
            states,
            &mut active_prefills,
            prefill_score,
            traffic,
            prefill_resource_base_node,
            base_batch_size,
            base_prompt_tokens,
            traffic.max_queue_delay_s,
            prefill_worker_slots_per_gpu,
        ),
        ServingPrefillBatching::Continuous {
            max_batch_tokens,
            chunk_tokens,
        } => schedule_continuous_prefills(
            scheduler,
            worker_runtime,
            states,
            &mut active_prefills,
            prefill_score,
            traffic,
            prefill_resource_base_node,
            base_batch_size,
            base_prompt_tokens,
            max_batch_tokens,
            chunk_tokens,
            traffic.max_queue_delay_s,
            prefill_worker_slots_per_gpu,
        ),
    }
}

fn prefill_worker_slots_per_gpu(traffic: &ServingTraffic) -> usize {
    scaled_worker_slots(
        traffic.max_prefill_worker_slots_per_gpu.unwrap_or(1),
        traffic.services.prefill,
    )
}

fn decode_worker_slots_per_gpu(traffic: &ServingTraffic) -> usize {
    scaled_worker_slots(
        traffic.max_decode_worker_slots_per_gpu.unwrap_or(1),
        traffic.services.decode,
    )
}

fn kv_transfer_worker_slots_per_gpu(traffic: &ServingTraffic) -> Option<usize> {
    traffic
        .max_kv_transfer_worker_slots_per_gpu
        .map(|slots| scaled_worker_slots(slots, traffic.services.kv_transfer))
}

fn scaled_worker_slots(configured_slots: u32, service: ServingServicePhaseConfig) -> usize {
    if !service.health.accepts_requests() {
        return 0;
    }
    let scaled = f64::from(configured_slots.max(1)) * service.worker_scale;
    if !scaled.is_finite() || scaled <= 0.0 {
        0
    } else {
        scaled.ceil().min(usize::MAX as f64) as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule_independent_prefills(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    active_prefills: &mut Vec<PrefillAdmissionSpan>,
    prefill_score: &ScoredParallelismConfig,
    traffic: &ServingTraffic,
    prefill_resource_base_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    max_queue_delay_s: Option<f64>,
    prefill_worker_slots_per_gpu: usize,
) {
    let mut order = (0..states.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| compare_request_priority(states, *left, *right));
    for idx in order {
        if states[idx].prefill_scheduled || states[idx].status != ServingRequestStatus::Pending {
            continue;
        }
        let prefill_node = states[idx].prefill_node;
        let request_idx = states[idx].request_idx;
        schedule_prefill_batch(
            scheduler,
            worker_runtime,
            states,
            active_prefills,
            traffic,
            &[idx],
            prefill_score,
            prefill_resource_base_node,
            Some(prefill_node),
            base_batch_size,
            base_prompt_tokens,
            None,
            max_queue_delay_s,
            prefill_worker_slots_per_gpu,
            format!("request {request_idx} prefill"),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule_continuous_prefills(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    active_prefills: &mut Vec<PrefillAdmissionSpan>,
    prefill_score: &ScoredParallelismConfig,
    traffic: &ServingTraffic,
    prefill_resource_base_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    max_batch_tokens: Option<u64>,
    chunk_tokens: Option<u32>,
    max_queue_delay_s: Option<f64>,
    prefill_worker_slots_per_gpu: usize,
) {
    let max_batch_tokens = max_batch_tokens.unwrap_or(u64::MAX).max(1);
    let mut batch_idx = 0_u32;

    while states
        .iter()
        .any(|state| !state.prefill_scheduled && state.status == ServingRequestStatus::Pending)
    {
        let Some((ready_idx, ready_s)) = states
            .iter()
            .enumerate()
            .filter(|(_, state)| {
                !state.prefill_scheduled && state.status == ServingRequestStatus::Pending
            })
            .map(|(idx, state)| {
                (
                    idx,
                    prefill_candidate_ready_s(
                        state,
                        scheduler,
                        worker_runtime,
                        prefill_worker_slots_per_gpu,
                    ),
                )
            })
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| compare_request_priority(states, left.0, right.0))
            })
        else {
            break;
        };
        let target_prefill_node =
            prefill_resource_base_node.map(|_| states[ready_idx].prefill_node);
        let mut selected = Vec::new();
        let mut selected_tokens = 0_u64;

        let mut candidates = states
            .iter()
            .enumerate()
            .filter_map(|(idx, state)| {
                if state.prefill_scheduled || state.status != ServingRequestStatus::Pending {
                    return None;
                }
                if let Some(target_prefill_node) = target_prefill_node
                    && state.prefill_node != target_prefill_node
                {
                    return None;
                }
                if prefill_candidate_ready_s(
                    state,
                    scheduler,
                    worker_runtime,
                    prefill_worker_slots_per_gpu,
                ) > ready_s + 1e-12
                {
                    return None;
                }
                Some(idx)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| compare_request_priority(states, *left, *right));

        for idx in candidates {
            let state = &states[idx];
            let request_tokens = prefill_work_tokens(state, chunk_tokens);
            if selected.is_empty()
                || selected_tokens.saturating_add(request_tokens) <= max_batch_tokens
            {
                selected.push(idx);
                selected_tokens = selected_tokens.saturating_add(request_tokens);
            }
        }

        let label = if chunk_tokens.is_some() {
            format!("prefill batch {batch_idx} chunk tokens {selected_tokens}")
        } else {
            format!("prefill batch {batch_idx} tokens {selected_tokens}")
        };
        schedule_prefill_batch(
            scheduler,
            worker_runtime,
            states,
            active_prefills,
            traffic,
            &selected,
            prefill_score,
            prefill_resource_base_node,
            target_prefill_node,
            base_batch_size,
            base_prompt_tokens,
            chunk_tokens,
            max_queue_delay_s,
            prefill_worker_slots_per_gpu,
            label,
        );
        batch_idx += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule_prefill_batch(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    active_prefills: &mut Vec<PrefillAdmissionSpan>,
    traffic: &ServingTraffic,
    selected: &[usize],
    prefill_score: &ScoredParallelismConfig,
    prefill_resource_base_node: Option<NodeId>,
    target_prefill_node: Option<NodeId>,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    chunk_tokens: Option<u32>,
    max_queue_delay_s: Option<f64>,
    prefill_worker_slots_per_gpu: usize,
    label: String,
) {
    if selected.is_empty() {
        return;
    }

    let total_tokens = selected
        .iter()
        .map(|idx| prefill_work_tokens(&states[*idx], chunk_tokens))
        .sum::<u64>();
    let base_tokens = u64::from(base_batch_size.max(1)) * u64::from(base_prompt_tokens.max(1));
    let scale = total_tokens as f64 / base_tokens.max(1) as f64;
    let earliest_start_s = selected
        .iter()
        .map(|idx| {
            prefill_candidate_ready_s(
                &states[*idx],
                scheduler,
                worker_runtime,
                prefill_worker_slots_per_gpu,
            )
        })
        .fold(0.0, f64::max);
    let dependencies = selected
        .iter()
        .flat_map(|idx| states[*idx].dependencies.iter().copied())
        .collect::<Vec<_>>();
    let mut prefill_ops = prefill_score.operations.clone();
    scale_operations(&mut prefill_ops, scale);
    if let Some(target_prefill_node) = target_prefill_node {
        remap_operations_to_node(
            &mut prefill_ops,
            prefill_resource_base_node,
            target_prefill_node,
        );
    }
    let has_cancellable_request = selected
        .iter()
        .any(|idx| states[*idx].cancellation_s.is_some());
    let has_queue_limited_request = selected.iter().any(|idx| {
        states[*idx]
            .max_queue_delay_s
            .or(max_queue_delay_s)
            .is_some()
    });
    let has_prefill_capacity_admission = request_level_prefill_capacity_enabled(traffic);
    if has_queue_limited_request || has_cancellable_request || has_prefill_capacity_admission {
        let mut candidate_scheduler = scheduler.clone();
        let candidate_ids = schedule_trace(
            &mut candidate_scheduler,
            &label,
            &prefill_ops,
            earliest_start_s,
            &dependencies,
        );
        let (candidate_start_s, candidate_finish_s) =
            operation_span(&candidate_scheduler, &candidate_ids);
        if has_prefill_capacity_admission {
            retire_prefill_admissions(active_prefills, candidate_start_s);
        }
        let mut candidate_active_prefills = active_prefills.clone();
        let mut admitted = Vec::new();
        let mut rejected = Vec::new();
        let mut capacity_rejected = Vec::new();
        let mut cancelled = Vec::new();
        for idx in selected {
            if let Some(cancellation_s) = states[*idx].cancellation_s
                && cancellation_s <= candidate_start_s + 1e-12
            {
                cancelled.push((*idx, cancellation_s));
                continue;
            }
            if let Some(effective_max_queue_delay_s) =
                states[*idx].max_queue_delay_s.or(max_queue_delay_s)
                && states[*idx].prefill_chunks == 0
            {
                let queue_delay_s = (candidate_start_s - states[*idx].arrival_s).max(0.0);
                if queue_delay_s > effective_max_queue_delay_s + 1e-12 {
                    rejected.push((*idx, queue_delay_s, effective_max_queue_delay_s));
                    continue;
                }
            }
            if has_prefill_capacity_admission {
                let candidate = PrefillAdmissionSpan::for_state_chunk(
                    &states[*idx],
                    chunk_tokens,
                    candidate_finish_s,
                );
                if states[*idx].prefill_chunks == 0
                    && let Some(reason) = prefill_capacity_admission_rejection(
                        &candidate_active_prefills,
                        &candidate,
                        traffic,
                    )
                {
                    capacity_rejected.push((*idx, reason, candidate_start_s));
                    continue;
                }
                candidate_active_prefills.push(candidate);
            }
            admitted.push(*idx);
        }

        if !cancelled.is_empty() || !rejected.is_empty() || !capacity_rejected.is_empty() {
            for (idx, cancellation_s) in cancelled {
                cancel_request(
                    &mut states[idx],
                    cancellation_s,
                    "request cancelled before prefill admission",
                );
            }
            for (idx, queue_delay_s, effective_max_queue_delay_s) in rejected {
                reject_admission(&mut states[idx], queue_delay_s, effective_max_queue_delay_s);
            }
            for (idx, reason, admission_s) in capacity_rejected {
                reject_prefill_capacity_admission(&mut states[idx], admission_s, reason);
            }
            if admitted.is_empty() {
                return;
            }
            schedule_prefill_batch(
                scheduler,
                worker_runtime,
                states,
                active_prefills,
                traffic,
                &admitted,
                prefill_score,
                prefill_resource_base_node,
                target_prefill_node,
                base_batch_size,
                base_prompt_tokens,
                chunk_tokens,
                max_queue_delay_s,
                prefill_worker_slots_per_gpu,
                label,
            );
            return;
        }
    }
    let prefill_ids = schedule_trace(
        scheduler,
        &label,
        &prefill_ops,
        earliest_start_s,
        &dependencies,
    );
    let (prefill_start_s, prefill_finish_s) = operation_span(scheduler, &prefill_ids);
    let prefill_deps = terminal_ids(&prefill_ops, &prefill_ids);

    for idx in selected {
        record_prefill_queue_breakdown(
            &mut states[*idx],
            scheduler,
            worker_runtime,
            prefill_worker_slots_per_gpu,
            prefill_start_s,
        );
    }
    let prefill_worker_assignments = assign_selected_prefill_workers_ready(
        worker_runtime,
        states,
        selected,
        prefill_worker_slots_per_gpu,
        prefill_start_s,
        prefill_finish_s,
        &prefill_ids,
    );

    for idx in selected {
        let state = &mut states[*idx];
        state.worker_assignments.extend(assignments_for_worker_gpus(
            &prefill_worker_assignments,
            &state.prefill_route_gpus,
            state.prefill_node,
        ));
        if state.prefill_chunks == 0 {
            state.prefill_start_s = prefill_start_s;
        }
        state.prefill_finish_s = prefill_finish_s;
        state.dependencies = prefill_deps.clone();
        let chunk_tokens = state_prefill_chunk_tokens(state, chunk_tokens);
        let prefill_tokens =
            u64::from(state.batch_size.max(1)).saturating_mul(u64::from(chunk_tokens));
        if prefill_tokens > 0 && prefill_finish_s > prefill_start_s {
            state.prefill_token_spans.push(PrefillTokenSpan {
                start_s: prefill_start_s,
                finish_s: prefill_finish_s,
                tokens: prefill_tokens,
            });
            if has_prefill_capacity_admission {
                active_prefills.push(PrefillAdmissionSpan::for_state_tokens(
                    state,
                    prefill_tokens,
                    prefill_finish_s,
                ));
            }
        }
        state.remaining_prefill_tokens =
            state.remaining_prefill_tokens.saturating_sub(chunk_tokens);
        state.prefill_chunks = state.prefill_chunks.saturating_add(1);
        state.prefill_scheduled = state.remaining_prefill_tokens == 0;
    }
}

fn prefill_ready_s(state: &DecodeRequestState, scheduler: &ResourceScheduler) -> f64 {
    if state.prefill_chunks == 0 {
        state.arrival_s
    } else {
        operation_finish_s(scheduler, &state.dependencies)
    }
}

fn prefill_candidate_ready_s(
    state: &DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    prefill_worker_slots_per_gpu: usize,
) -> f64 {
    prefill_ready_s(state, scheduler).max(prefill_worker_ready_s(
        state,
        worker_runtime,
        prefill_worker_slots_per_gpu,
    ))
}

fn prefill_worker_ready_s(
    state: &DecodeRequestState,
    worker_runtime: &ServingWorkerRuntime,
    prefill_worker_slots_per_gpu: usize,
) -> f64 {
    route_worker_gpus(&state.prefill_route_gpus, state.prefill_node)
        .iter()
        .map(|gpu| {
            prefill_worker_gpu_ready_s(
                &worker_runtime.prefill_ready_s,
                *gpu,
                prefill_worker_slots_per_gpu,
            )
        })
        .fold(0.0, f64::max)
}

fn prefill_worker_gpu_ready_s(
    ready_s: &BTreeMap<GpuAddr, Vec<f64>>,
    gpu: GpuAddr,
    prefill_worker_slots_per_gpu: usize,
) -> f64 {
    worker_slot_ready_s(ready_s, gpu, prefill_worker_slots_per_gpu)
}

fn decode_candidate_ready_s(
    state: &DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    decode_worker_slots_per_gpu: usize,
) -> f64 {
    operation_finish_s(scheduler, &state.dependencies).max(decode_worker_ready_s(
        state,
        worker_runtime,
        decode_worker_slots_per_gpu,
    ))
}

fn decode_worker_ready_s(
    state: &DecodeRequestState,
    worker_runtime: &ServingWorkerRuntime,
    decode_worker_slots_per_gpu: usize,
) -> f64 {
    route_workers_ready_s(
        &worker_runtime.decode_ready_s,
        &state.decode_route_gpus,
        state.decode_node,
        decode_worker_slots_per_gpu,
    )
}

fn kv_transfer_worker_ready_s(
    state: &DecodeRequestState,
    worker_runtime: &ServingWorkerRuntime,
    kv_transfer_worker_slots_per_gpu: usize,
) -> f64 {
    kv_transfer_worker_gpus(state)
        .iter()
        .map(|gpu| {
            worker_slot_ready_s(
                &worker_runtime.kv_transfer_ready_s,
                *gpu,
                kv_transfer_worker_slots_per_gpu,
            )
        })
        .fold(0.0, f64::max)
}

fn record_prefill_queue_breakdown(
    state: &mut DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    prefill_worker_slots_per_gpu: usize,
    prefill_start_s: f64,
) {
    let phase_ready_s = prefill_ready_s(state, scheduler);
    let worker_ready_s =
        prefill_worker_ready_s(state, worker_runtime, prefill_worker_slots_per_gpu);
    let candidate_ready_s = phase_ready_s.max(worker_ready_s);
    state.prefill_worker_queue_s += finite_or_zero(worker_ready_s - phase_ready_s);
    state.prefill_resource_queue_s += finite_or_zero(prefill_start_s - candidate_ready_s);
}

fn record_decode_queue_breakdown(
    state: &mut DecodeRequestState,
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    decode_worker_slots_per_gpu: usize,
    decode_start_s: f64,
) {
    let phase_ready_s = operation_finish_s(scheduler, &state.dependencies);
    let worker_ready_s = decode_worker_ready_s(state, worker_runtime, decode_worker_slots_per_gpu);
    let candidate_ready_s = phase_ready_s.max(worker_ready_s);
    state.decode_worker_queue_s += finite_or_zero(worker_ready_s - phase_ready_s);
    state.decode_resource_queue_s += finite_or_zero(decode_start_s - candidate_ready_s);
}

fn assign_selected_prefill_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    states: &[DecodeRequestState],
    selected: &[usize],
    prefill_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<ServingWorkerAssignmentObservation> {
    let mut route_gpus = BTreeSet::new();
    for idx in selected {
        route_gpus.extend(route_worker_gpus(
            &states[*idx].prefill_route_gpus,
            states[*idx].prefill_node,
        ));
    }
    let route_gpus = route_gpus.into_iter().collect::<Vec<_>>();
    assign_worker_slots(
        &mut worker_runtime.prefill_ready_s,
        &route_gpus,
        0,
        prefill_worker_slots_per_gpu,
        WorkerAssignmentSpan {
            phase: "prefill",
            start_s,
            finish_s,
            operation_ids,
        },
    )
}

fn assign_decode_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    state: &DecodeRequestState,
    decode_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<ServingWorkerAssignmentObservation> {
    assign_worker_slots(
        &mut worker_runtime.decode_ready_s,
        &state.decode_route_gpus,
        state.decode_node,
        decode_worker_slots_per_gpu,
        WorkerAssignmentSpan {
            phase: "decode",
            start_s,
            finish_s,
            operation_ids,
        },
    )
}

fn assign_selected_decode_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    states: &[DecodeRequestState],
    selected: &[usize],
    decode_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<(usize, Vec<ServingWorkerAssignmentObservation>)> {
    selected
        .iter()
        .map(|idx| {
            (
                *idx,
                assign_decode_workers_ready(
                    worker_runtime,
                    &states[*idx],
                    decode_worker_slots_per_gpu,
                    start_s,
                    finish_s,
                    operation_ids,
                ),
            )
        })
        .collect()
}

fn assign_kv_transfer_workers_ready(
    worker_runtime: &mut ServingWorkerRuntime,
    state: &DecodeRequestState,
    kv_transfer_worker_slots_per_gpu: usize,
    start_s: f64,
    finish_s: f64,
    operation_ids: &[usize],
) -> Vec<ServingWorkerAssignmentObservation> {
    let gpus = kv_transfer_worker_gpus(state);
    assign_worker_slots(
        &mut worker_runtime.kv_transfer_ready_s,
        &gpus,
        state.decode_node,
        kv_transfer_worker_slots_per_gpu,
        WorkerAssignmentSpan {
            phase: "kv_transfer",
            start_s,
            finish_s,
            operation_ids,
        },
    )
}

fn kv_transfer_worker_gpus(state: &DecodeRequestState) -> Vec<GpuAddr> {
    let mut gpus = BTreeSet::new();
    gpus.extend(route_worker_gpus(
        &state.prefill_route_gpus,
        state.prefill_node,
    ));
    gpus.extend(route_worker_gpus(
        &state.decode_route_gpus,
        state.decode_node,
    ));
    gpus.into_iter().collect()
}

fn prefill_work_tokens(state: &DecodeRequestState, chunk_tokens: Option<u32>) -> u64 {
    u64::from(state.batch_size.max(1)) * u64::from(state_prefill_chunk_tokens(state, chunk_tokens))
}

fn state_prefill_chunk_tokens(state: &DecodeRequestState, chunk_tokens: Option<u32>) -> u32 {
    chunk_tokens
        .map(|chunk_tokens| state.remaining_prefill_tokens.min(chunk_tokens.max(1)))
        .unwrap_or(state.remaining_prefill_tokens)
}

fn request_level_prefill_capacity_enabled(traffic: &ServingTraffic) -> bool {
    traffic.decode_capacity_policy == ServingDecodeCapacityPolicy::RequestReject
        && (traffic.max_prefill_tokens.is_some()
            || traffic.max_prefill_tokens_per_node.is_some()
            || traffic.max_prefill_tokens_per_gpu.is_some()
            || traffic
                .traffic_classes
                .iter()
                .any(|class| class.max_prefill_tokens.is_some()))
}

fn retire_prefill_admissions(active: &mut Vec<PrefillAdmissionSpan>, now_s: f64) {
    active.retain(|span| span.finish_s > now_s + 1e-12);
}

fn prefill_capacity_admission_rejection(
    active: &[PrefillAdmissionSpan],
    candidate: &PrefillAdmissionSpan,
    traffic: &ServingTraffic,
) -> Option<ServingRejection> {
    if let Some(limit) = traffic.max_prefill_tokens {
        let observed = active
            .iter()
            .map(|span| span.tokens)
            .sum::<u64>()
            .saturating_add(candidate.tokens);
        if observed > limit {
            let message = format!(
                "prefill admission rejected: active prefill tokens {observed} > max_prefill_tokens {limit}"
            );
            return Some(with_rejection_details(
                request_rejection(
                    "prefill",
                    "capacity",
                    "prefill_tokens",
                    "prefill_active_tokens_exceeded",
                    message,
                ),
                Some(observed as f64),
                Some(limit as f64),
                Some("tokens"),
                Some("increase max_prefill_tokens, add prefill capacity, or reduce prompt load"),
            ));
        }
    }

    if let Some(class) = serving_traffic_class(traffic, candidate.traffic_class.as_deref())
        && let Some(limit) = class.max_prefill_tokens
    {
        let observed =
            active_prefill_class_tokens(active, &class.name).saturating_add(candidate.tokens);
        if observed > limit {
            let message = format!(
                "prefill admission rejected: traffic class {} active prefill tokens {} > max_prefill_tokens {}",
                class.name, observed, limit
            );
            return Some(with_rejection_details(
                request_rejection(
                    "prefill",
                    "capacity",
                    &format!("traffic_class {} prefill_tokens", class.name),
                    "traffic_class_prefill_active_tokens_exceeded",
                    message,
                ),
                Some(observed as f64),
                Some(limit as f64),
                Some("tokens"),
                Some(
                    "increase the traffic class max_prefill_tokens, add prefill capacity, or reduce class prompt concurrency",
                ),
            ));
        }
    }

    if let Some(limit) = traffic.max_prefill_tokens_per_node {
        for (node_id, tokens) in &candidate.node_tokens {
            let observed = active_prefill_node_tokens(active, *node_id).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "prefill admission rejected: node {node_id} active prefill tokens {observed} > max_prefill_tokens_per_node {limit}"
                );
                return Some(with_rejection_details(
                    request_rejection(
                        "prefill",
                        "capacity",
                        &format!("node {node_id} prefill_tokens"),
                        "prefill_active_tokens_per_node_exceeded",
                        message,
                    ),
                    Some(observed as f64),
                    Some(limit as f64),
                    Some("tokens"),
                    Some(
                        "increase max_prefill_tokens_per_node, add prefill nodes, or rebalance routing",
                    ),
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_prefill_tokens_per_gpu {
        for (gpu, tokens) in &candidate.gpu_tokens {
            let observed = active_prefill_gpu_tokens(active, *gpu).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "prefill admission rejected: node {} gpu {} active prefill tokens {} > max_prefill_tokens_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(with_rejection_details(
                    request_rejection(
                        "prefill",
                        "capacity",
                        &format!(
                            "node {} gpu {} prefill_tokens",
                            gpu.node_id, gpu.local_gpu_id
                        ),
                        "prefill_active_tokens_per_gpu_exceeded",
                        message,
                    ),
                    Some(observed as f64),
                    Some(limit as f64),
                    Some("tokens"),
                    Some(
                        "increase max_prefill_tokens_per_gpu, add prefill GPUs, or reduce chunk size",
                    ),
                ));
            }
        }
    }

    None
}

fn active_prefill_class_tokens(active: &[PrefillAdmissionSpan], class_name: &str) -> u64 {
    active
        .iter()
        .filter(|span| span.traffic_class.as_deref() == Some(class_name))
        .map(|span| span.tokens)
        .sum()
}

fn active_prefill_node_tokens(active: &[PrefillAdmissionSpan], node_id: NodeId) -> u64 {
    active
        .iter()
        .flat_map(|span| &span.node_tokens)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, tokens)| *tokens)
        .sum()
}

fn active_prefill_gpu_tokens(active: &[PrefillAdmissionSpan], gpu: GpuAddr) -> u64 {
    active
        .iter()
        .flat_map(|span| &span.gpu_tokens)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, tokens)| *tokens)
        .sum()
}

fn request_rejection(
    phase: &str,
    category: &str,
    resource: &str,
    code: &str,
    message: String,
) -> ServingRejection {
    ServingRejection {
        phase: phase.to_string(),
        category: category.to_string(),
        resource: resource.to_string(),
        code: code.to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: None,
        message,
    }
}

fn with_rejection_details(
    mut rejection: ServingRejection,
    observed: Option<f64>,
    limit: Option<f64>,
    unit: Option<&str>,
    remediation: Option<&str>,
) -> ServingRejection {
    rejection.observed = observed;
    rejection.limit = limit;
    rejection.unit = unit.map(str::to_string);
    rejection.remediation = remediation.map(str::to_string);
    rejection
}

fn set_request_rejection(state: &mut DecodeRequestState, rejection: ServingRejection) {
    state.failure_reason = Some(rejection.message.clone());
    state.failure_rejection = Some(rejection);
}

fn reject_admission(state: &mut DecodeRequestState, queue_delay_s: f64, max_queue_delay_s: f64) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(state.arrival_s + queue_delay_s);
    let message = format!(
        "admission rejected: prefill queue delay {queue_delay_s:.6}s > max_queue_delay {max_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "prefill",
                "queueing",
                "prefill_queue",
                "prefill_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_queue_delay_s),
            Some("s"),
            Some("increase max_queue_delay, add prefill workers, or reduce arrival pressure"),
        ),
    );
    state.dependencies.clear();
    state.prefill_start_s = f64::INFINITY;
    state.prefill_finish_s = f64::INFINITY;
    state.kv_start_s = f64::INFINITY;
    state.kv_finish_s = f64::INFINITY;
    state.kv_worker_queue_s = 0.0;
    state.kv_resource_queue_s = 0.0;
    state.kv_transfer_resources.clear();
    state.kv_transfer_resource_dependencies.clear();
    state.kv_transfer_s = 0.0;
}

fn reject_prefill_capacity_admission(
    state: &mut DecodeRequestState,
    admission_s: f64,
    rejection: ServingRejection,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if admission_s.is_finite() {
        admission_s
    } else {
        state.arrival_s
    });
    set_request_rejection(state, rejection);
    state.dependencies.clear();
    state.prefill_start_s = f64::INFINITY;
    state.prefill_finish_s = f64::INFINITY;
    state.kv_start_s = f64::INFINITY;
    state.kv_finish_s = f64::INFINITY;
    state.kv_worker_queue_s = 0.0;
    state.kv_resource_queue_s = 0.0;
    state.kv_transfer_bottlenecks.clear();
    state.kv_transfer_paths.clear();
    state.kv_transfer_resources.clear();
    state.kv_transfer_resource_dependencies.clear();
    state.kv_transfer_s = 0.0;
    state.prefill_token_spans.clear();
}

fn reject_kv_queue_admission(
    state: &mut DecodeRequestState,
    ready_s: f64,
    queue_delay_s: f64,
    max_kv_queue_delay_s: f64,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if ready_s.is_finite() {
        ready_s + max_kv_queue_delay_s
    } else {
        state.prefill_finish_s.max(state.arrival_s)
    });
    let message = format!(
        "admission rejected: KV transfer queue delay {queue_delay_s:.6}s > max_kv_queue_delay {max_kv_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "kv_transfer",
                "queueing",
                "kv_transfer_queue",
                "kv_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_kv_queue_delay_s),
            Some("s"),
            Some(
                "increase max_kv_queue_delay, add KV transfer worker slots, or improve KV route capacity",
            ),
        ),
    );
    state.dependencies.clear();
    state.kv_finish_s = f64::INFINITY;
}

fn cancel_request(state: &mut DecodeRequestState, cancellation_s: f64, reason: &str) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::Cancelled;
    state.status_time_s = Some(cancellation_s);
    state.failure_reason = Some(format!("{reason}: cancellation at {cancellation_s:.6}s"));
    state.failure_rejection = None;
}

fn apply_decode_capacity_admission(
    states: &mut [DecodeRequestState],
    traffic: &ServingTraffic,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    decode_worker_slots_per_gpu: usize,
) {
    if traffic.decode_capacity_policy != ServingDecodeCapacityPolicy::RequestReject {
        return;
    }

    let mut order = (0..states.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        states[*left]
            .kv_finish_s
            .total_cmp(&states[*right].kv_finish_s)
            .then_with(|| compare_request_priority(states, *left, *right))
    });
    let mut active = Vec::new();
    let mut decode_worker_ready_s = BTreeMap::new();

    for idx in order {
        if states[idx].status != ServingRequestStatus::Pending || states[idx].remaining_tokens == 0
        {
            continue;
        }
        let admission_s = states[idx].kv_finish_s;
        if !admission_s.is_finite() {
            reject_decode_capacity_admission(
                &mut states[idx],
                with_rejection_details(
                    request_rejection(
                        "decode",
                        "capacity",
                        "kv_residency",
                        "decode_admission_kv_not_ready",
                        "decode admission rejected: request is not ready for finite KV residency"
                            .to_string(),
                    ),
                    None,
                    None,
                    None,
                    Some("ensure prefill and KV transfer complete before decode admission"),
                ),
            );
            continue;
        }

        retire_decode_residency(&mut active, admission_s);
        let decode_start_s = admission_s.max(route_workers_ready_s(
            &decode_worker_ready_s,
            &states[idx].decode_route_gpus,
            states[idx].decode_node,
            decode_worker_slots_per_gpu,
        ));
        let finish_s = decode_start_s
            + decode_service_estimate_s(
                &states[idx],
                decode_one_score,
                decode_tail_scale,
                base_batch_size,
            );
        let candidate = DecodeAdmissionResidency::for_state(&states[idx], finish_s);
        if let Some(rejection) = decode_capacity_admission_rejection(&active, &candidate, traffic) {
            reject_decode_capacity_admission(&mut states[idx], rejection);
            continue;
        }

        mark_route_workers_ready(
            &mut decode_worker_ready_s,
            &states[idx].decode_route_gpus,
            states[idx].decode_node,
            decode_worker_slots_per_gpu,
            finish_s,
        );
        active.push(candidate);
    }
}

#[derive(Clone, Debug, PartialEq)]
struct DecodeAdmissionResidency {
    finish_s: f64,
    traffic_class: Option<String>,
    sequences: u32,
    resident_tokens: u64,
    kv_blocks: u64,
    node_sequences: Vec<(NodeId, u32)>,
    node_tokens: Vec<(NodeId, u64)>,
    node_blocks: Vec<(NodeId, u64)>,
    gpu_sequences: Vec<(GpuAddr, u32)>,
    gpu_tokens: Vec<(GpuAddr, u64)>,
    gpu_blocks: Vec<(GpuAddr, u64)>,
}

impl DecodeAdmissionResidency {
    fn for_state(state: &DecodeRequestState, finish_s: f64) -> Self {
        let sequences = state.batch_size.max(1);
        let resident_tokens =
            u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1));
        let allocation = sequence_kv_allocation(
            state.batch_size.max(1),
            state.max_sequence_tokens.max(1),
            state.kv_block_tokens,
        );
        let decode_route_nodes = if state.decode_route_nodes.is_empty() {
            vec![state.decode_node]
        } else {
            state.decode_route_nodes.clone()
        };
        let per_node_tokens = resident_tokens.div_ceil(decode_route_nodes.len().max(1) as u64);
        let per_node_blocks = allocation
            .blocks
            .div_ceil(decode_route_nodes.len().max(1) as u64);
        let node_sequences = decode_route_nodes
            .iter()
            .map(|node_id| (*node_id, sequences))
            .collect::<Vec<_>>();
        let node_tokens = decode_route_nodes
            .iter()
            .map(|node_id| (*node_id, per_node_tokens))
            .collect::<Vec<_>>();
        let node_blocks = decode_route_nodes
            .into_iter()
            .map(|node_id| (node_id, per_node_blocks))
            .collect::<Vec<_>>();
        let decode_route_gpus = if state.decode_route_gpus.is_empty() {
            vec![GpuAddr {
                node_id: state.decode_node,
                local_gpu_id: 0,
            }]
        } else {
            state.decode_route_gpus.clone()
        };
        let per_gpu_tokens = resident_tokens.div_ceil(decode_route_gpus.len().max(1) as u64);
        let per_gpu_blocks = allocation
            .blocks
            .div_ceil(decode_route_gpus.len().max(1) as u64);
        let gpu_sequences = decode_route_gpus
            .iter()
            .map(|gpu| (*gpu, sequences))
            .collect::<Vec<_>>();
        let gpu_tokens = decode_route_gpus
            .iter()
            .map(|gpu| (*gpu, per_gpu_tokens))
            .collect::<Vec<_>>();
        let gpu_blocks = decode_route_gpus
            .into_iter()
            .map(|gpu| (gpu, per_gpu_blocks))
            .collect();

        Self {
            finish_s,
            traffic_class: state.traffic_class.clone(),
            sequences,
            resident_tokens,
            kv_blocks: allocation.blocks,
            node_sequences,
            node_tokens,
            node_blocks,
            gpu_sequences,
            gpu_tokens,
            gpu_blocks,
        }
    }
}

fn retire_decode_residency(active: &mut Vec<DecodeAdmissionResidency>, now_s: f64) {
    active.retain(|residency| residency.finish_s > now_s + 1e-12);
}

fn decode_service_estimate_s(
    state: &DecodeRequestState,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
) -> f64 {
    let first_token_scale = request_token_scale(state.batch_size, 1, base_batch_size, 1);
    let decode_iterations =
        1.0 + f64::from(state.decode_tokens.saturating_sub(1)) * decode_tail_scale;
    finite_or_zero(decode_one_score.estimated_latency_s) * first_token_scale * decode_iterations
}

fn decode_capacity_admission_request_rejection(
    resource: String,
    code: &str,
    message: String,
    observed: f64,
    limit: f64,
    unit: &str,
    remediation: &str,
) -> ServingRejection {
    with_rejection_details(
        request_rejection("decode", "capacity", &resource, code, message),
        Some(observed),
        Some(limit),
        Some(unit),
        Some(remediation),
    )
}

fn decode_capacity_admission_rejection(
    active: &[DecodeAdmissionResidency],
    candidate: &DecodeAdmissionResidency,
    traffic: &ServingTraffic,
) -> Option<ServingRejection> {
    let active_sequences = active
        .iter()
        .map(|residency| residency.sequences)
        .sum::<u32>();
    if let Some(limit) = traffic.max_decode_sequences {
        let observed = active_sequences.saturating_add(candidate.sequences);
        if observed > limit {
            let message = format!(
                "decode admission rejected: active decode sequences {observed} > max_decode_sequences {limit}"
            );
            return Some(decode_capacity_admission_request_rejection(
                "decode_sequences".to_string(),
                "decode_capacity_exceeded",
                message,
                f64::from(observed),
                f64::from(limit),
                "sequences",
                "increase max_decode_sequences, add decode capacity, or reduce concurrency",
            ));
        }
    }

    let active_tokens = active
        .iter()
        .map(|residency| residency.resident_tokens)
        .sum::<u64>();
    if let Some(limit) = traffic.max_resident_tokens {
        let observed = active_tokens.saturating_add(candidate.resident_tokens);
        if observed > limit {
            let message = format!(
                "decode admission rejected: resident KV tokens {observed} > max_resident_tokens {limit}"
            );
            return Some(decode_capacity_admission_request_rejection(
                "resident_tokens".to_string(),
                "kv_residency_capacity_exceeded",
                message,
                observed as f64,
                limit as f64,
                "tokens",
                "increase max_resident_tokens, reduce sequence length, or add decode capacity",
            ));
        }
    }

    if let Some(limit) = traffic.max_kv_blocks {
        let active_blocks = active
            .iter()
            .map(|residency| residency.kv_blocks)
            .sum::<u64>();
        let observed = active_blocks.saturating_add(candidate.kv_blocks);
        if observed > limit {
            let message = format!(
                "decode admission rejected: resident KV blocks {observed} > max_kv_blocks {limit}"
            );
            return Some(decode_capacity_admission_request_rejection(
                "kv_blocks".to_string(),
                "kv_block_capacity_exceeded",
                message,
                observed as f64,
                limit as f64,
                "blocks",
                "increase max_kv_blocks, increase KV block capacity, or reduce sequence length",
            ));
        }
    }

    if let Some(class) = serving_traffic_class(traffic, candidate.traffic_class.as_deref()) {
        if let Some(limit) = class.max_decode_sequences {
            let observed =
                active_class_sequences(active, &class.name).saturating_add(candidate.sequences);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: traffic class {} active decode sequences {} > max_decode_sequences {}",
                    class.name, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("traffic_class {} decode_sequences", class.name),
                    "traffic_class_decode_capacity_exceeded",
                    message,
                    f64::from(observed),
                    f64::from(limit),
                    "sequences",
                    "increase the traffic class max_decode_sequences, add class capacity, or reduce class concurrency",
                ));
            }
        }

        if let Some(limit) = class.max_resident_tokens {
            let observed =
                active_class_tokens(active, &class.name).saturating_add(candidate.resident_tokens);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: traffic class {} resident KV tokens {} > max_resident_tokens {}",
                    class.name, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("traffic_class {} resident_tokens", class.name),
                    "traffic_class_kv_residency_capacity_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "tokens",
                    "increase the traffic class max_resident_tokens, reduce class sequence length, or add class capacity",
                ));
            }
        }

        if let Some(limit) = class.max_kv_blocks {
            let observed =
                active_class_blocks(active, &class.name).saturating_add(candidate.kv_blocks);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: traffic class {} resident KV blocks {} > max_kv_blocks {}",
                    class.name, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("traffic_class {} kv_blocks", class.name),
                    "traffic_class_kv_block_capacity_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "blocks",
                    "increase the traffic class max_kv_blocks, reduce class sequence length, or add class capacity",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_decode_sequences_per_node {
        for (node_id, sequences) in &candidate.node_sequences {
            let observed = active_node_sequences(active, *node_id).saturating_add(*sequences);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {node_id} active decode sequences {observed} > max_decode_sequences_per_node {limit}"
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("node {node_id} decode_sequences_per_node"),
                    "decode_capacity_per_node_exceeded",
                    message,
                    f64::from(observed),
                    f64::from(limit),
                    "sequences",
                    "increase max_decode_sequences_per_node, add decode nodes, or rebalance routing",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_resident_tokens_per_node {
        for (node_id, tokens) in &candidate.node_tokens {
            let observed = active_node_tokens(active, *node_id).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {node_id} resident KV tokens {observed} > max_resident_tokens_per_node {limit}"
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("node {node_id} resident_tokens_per_node"),
                    "kv_residency_capacity_per_node_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "tokens",
                    "increase max_resident_tokens_per_node, add decode nodes, or reduce sequence length",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_kv_blocks_per_node {
        for (node_id, blocks) in &candidate.node_blocks {
            let observed = active_node_blocks(active, *node_id).saturating_add(*blocks);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {node_id} resident KV blocks {observed} > max_kv_blocks_per_node {limit}"
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!("node {node_id} kv_blocks_per_node"),
                    "kv_block_capacity_per_node_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "blocks",
                    "increase max_kv_blocks_per_node, add decode nodes, or reduce sequence length",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_decode_sequences_per_gpu {
        for (gpu, sequences) in &candidate.gpu_sequences {
            let observed = active_gpu_sequences(active, *gpu).saturating_add(*sequences);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {} gpu {} active decode sequences {} > max_decode_sequences_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!(
                        "node {} gpu {} decode_sequences_per_gpu",
                        gpu.node_id, gpu.local_gpu_id
                    ),
                    "decode_capacity_per_gpu_exceeded",
                    message,
                    f64::from(observed),
                    f64::from(limit),
                    "sequences",
                    "increase max_decode_sequences_per_gpu, add decode GPUs, or rebalance routing",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_resident_tokens_per_gpu {
        for (gpu, tokens) in &candidate.gpu_tokens {
            let observed = active_gpu_tokens(active, *gpu).saturating_add(*tokens);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {} gpu {} resident KV tokens {} > max_resident_tokens_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!(
                        "node {} gpu {} resident_tokens_per_gpu",
                        gpu.node_id, gpu.local_gpu_id
                    ),
                    "kv_residency_capacity_per_gpu_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "tokens",
                    "increase max_resident_tokens_per_gpu, add decode GPUs, or reduce sequence length",
                ));
            }
        }
    }

    if let Some(limit) = traffic.max_kv_blocks_per_gpu {
        for (gpu, blocks) in &candidate.gpu_blocks {
            let observed = active_gpu_blocks(active, *gpu).saturating_add(*blocks);
            if observed > limit {
                let message = format!(
                    "decode admission rejected: node {} gpu {} resident KV blocks {} > max_kv_blocks_per_gpu {}",
                    gpu.node_id, gpu.local_gpu_id, observed, limit
                );
                return Some(decode_capacity_admission_request_rejection(
                    format!(
                        "node {} gpu {} kv_blocks_per_gpu",
                        gpu.node_id, gpu.local_gpu_id
                    ),
                    "kv_block_capacity_per_gpu_exceeded",
                    message,
                    observed as f64,
                    limit as f64,
                    "blocks",
                    "increase max_kv_blocks_per_gpu, add decode GPUs, or reduce sequence length",
                ));
            }
        }
    }

    None
}

fn serving_traffic_class<'a>(
    traffic: &'a ServingTraffic,
    class_name: Option<&str>,
) -> Option<&'a ServingTrafficClass> {
    let class_name = class_name?;
    traffic
        .traffic_classes
        .iter()
        .find(|class| class.name == class_name)
}

fn active_class_sequences(active: &[DecodeAdmissionResidency], class_name: &str) -> u32 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.sequences)
        .sum()
}

fn active_class_tokens(active: &[DecodeAdmissionResidency], class_name: &str) -> u64 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.resident_tokens)
        .sum()
}

fn active_class_blocks(active: &[DecodeAdmissionResidency], class_name: &str) -> u64 {
    active
        .iter()
        .filter(|residency| residency.traffic_class.as_deref() == Some(class_name))
        .map(|residency| residency.kv_blocks)
        .sum()
}

fn active_node_sequences(active: &[DecodeAdmissionResidency], node_id: NodeId) -> u32 {
    active
        .iter()
        .flat_map(|residency| &residency.node_sequences)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, sequences)| *sequences)
        .sum()
}

fn active_node_tokens(active: &[DecodeAdmissionResidency], node_id: NodeId) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.node_tokens)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, tokens)| *tokens)
        .sum()
}

fn active_node_blocks(active: &[DecodeAdmissionResidency], node_id: NodeId) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.node_blocks)
        .filter(|(active_node_id, _)| *active_node_id == node_id)
        .map(|(_, blocks)| *blocks)
        .sum()
}

fn active_gpu_sequences(active: &[DecodeAdmissionResidency], gpu: GpuAddr) -> u32 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_sequences)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, sequences)| *sequences)
        .sum()
}

fn active_gpu_tokens(active: &[DecodeAdmissionResidency], gpu: GpuAddr) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_tokens)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, tokens)| *tokens)
        .sum()
}

fn active_gpu_blocks(active: &[DecodeAdmissionResidency], gpu: GpuAddr) -> u64 {
    active
        .iter()
        .flat_map(|residency| &residency.gpu_blocks)
        .filter(|(active_gpu, _)| *active_gpu == gpu)
        .map(|(_, blocks)| *blocks)
        .sum()
}

fn reject_decode_capacity_admission(state: &mut DecodeRequestState, rejection: ServingRejection) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if state.kv_finish_s.is_finite() {
        state.kv_finish_s
    } else {
        state.arrival_s
    });
    set_request_rejection(state, rejection);
    state.dependencies.clear();
}

fn decode_queue_delay_rejection(
    state: &DecodeRequestState,
    decode_start_s: f64,
) -> Option<(f64, f64)> {
    if state.emitted_tokens != 0 {
        return None;
    }
    let max_decode_queue_delay_s = state.max_decode_queue_delay_s?;
    let queue_delay_s = finite_or_zero(decode_start_s - state.kv_finish_s);
    (queue_delay_s > max_decode_queue_delay_s + 1e-12)
        .then_some((queue_delay_s, max_decode_queue_delay_s))
}

fn decode_iteration_queue_timeout(
    state: &DecodeRequestState,
    decode_start_s: f64,
) -> Option<(f64, f64)> {
    if state.emitted_tokens == 0 {
        return None;
    }
    let max_queue_delay_s = state.max_decode_iteration_queue_delay_s?;
    let phase_ready_s = state.last_decode_finish_s.unwrap_or(state.kv_finish_s);
    let queue_delay_s = finite_or_zero(decode_start_s - phase_ready_s);
    (queue_delay_s > max_queue_delay_s + 1e-12).then_some((queue_delay_s, max_queue_delay_s))
}

fn reject_decode_queue_admission(
    state: &mut DecodeRequestState,
    decode_start_s: f64,
    queue_delay_s: f64,
    max_decode_queue_delay_s: f64,
) {
    state.prefill_scheduled = true;
    state.remaining_prefill_tokens = 0;
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::RejectedAdmission;
    state.status_time_s = Some(if state.kv_finish_s.is_finite() {
        state.kv_finish_s + max_decode_queue_delay_s
    } else {
        state.arrival_s
    });
    let message = format!(
        "admission rejected: decode queue delay {queue_delay_s:.6}s > max_decode_queue_delay {max_decode_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "decode",
                "queueing",
                "decode_queue",
                "decode_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_decode_queue_delay_s),
            Some("s"),
            Some("increase max_decode_queue_delay, add decode workers, or reduce decode load"),
        ),
    );
    state.first_decode_start_s = Some(decode_start_s);
    state.dependencies.clear();
}

fn timeout_decode_iteration_queue(
    state: &mut DecodeRequestState,
    queue_delay_s: f64,
    max_queue_delay_s: f64,
) {
    let phase_ready_s = state.last_decode_finish_s.unwrap_or(state.kv_finish_s);
    state.remaining_tokens = 0;
    state.status = ServingRequestStatus::TimedOut;
    state.status_time_s = Some(if phase_ready_s.is_finite() {
        phase_ready_s + max_queue_delay_s
    } else {
        state.arrival_s
    });
    let message = format!(
        "request timed out: decode iteration queue delay {queue_delay_s:.6}s > max_decode_iteration_queue_delay {max_queue_delay_s:.6}s"
    );
    set_request_rejection(
        state,
        with_rejection_details(
            request_rejection(
                "decode",
                "queueing",
                "decode_iteration_queue",
                "decode_iteration_queue_delay_exceeded",
                message,
            ),
            Some(queue_delay_s),
            Some(max_queue_delay_s),
            Some("s"),
            Some(
                "increase max_decode_iteration_queue_delay, add decode workers, or reduce decode load",
            ),
        ),
    );
    state.dependencies.clear();
}

#[allow(clippy::too_many_arguments)]
fn schedule_independent_decodes(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    decode_iterations: &mut Vec<ServingDecodeIterationObservation>,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    decode_worker_slots_per_gpu: usize,
) {
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let mut order = (0..states.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| compare_request_priority(states, *left, *right));
    for idx in order {
        let state = &mut states[idx];
        while state.remaining_tokens > 0 {
            let ready_s = decode_candidate_ready_s(
                state,
                scheduler,
                worker_runtime,
                decode_worker_slots_per_gpu,
            );
            if let Some(cancellation_s) = state.cancellation_s
                && cancellation_s <= ready_s + 1e-12
            {
                cancel_request(
                    state,
                    cancellation_s,
                    "request cancelled before decode iteration",
                );
                break;
            }
            let mut token_ops = decode_one_score.operations.clone();
            let mut scale = request_token_scale(state.batch_size, 1, base_batch_size, 1);
            if state.emitted_tokens > 0 {
                scale *= decode_tail_scale;
            }
            scale_operations(&mut token_ops, scale);
            remap_operations_to_node(&mut token_ops, decode_resource_base_node, state.decode_node);
            let (preview_decode_start_s, _) =
                preview_trace_span(scheduler, &token_ops, ready_s, &state.dependencies);
            if let Some((queue_delay_s, max_decode_queue_delay_s)) =
                decode_queue_delay_rejection(state, preview_decode_start_s)
            {
                record_decode_queue_breakdown(
                    state,
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                reject_decode_queue_admission(
                    state,
                    preview_decode_start_s,
                    queue_delay_s,
                    max_decode_queue_delay_s,
                );
                break;
            }
            if let Some((queue_delay_s, max_queue_delay_s)) =
                decode_iteration_queue_timeout(state, preview_decode_start_s)
            {
                record_decode_queue_breakdown(
                    state,
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                timeout_decode_iteration_queue(state, queue_delay_s, max_queue_delay_s);
                break;
            }
            let decode_ids = schedule_trace(
                scheduler,
                &format!(
                    "request {} decode token {}",
                    state.request_idx, state.emitted_tokens
                ),
                &token_ops,
                ready_s,
                &state.dependencies,
            );
            let (decode_start_s, _) = operation_span(scheduler, &decode_ids);
            let dependencies = terminal_ids(&token_ops, &decode_ids);
            let finish_s = operation_finish_s(scheduler, &dependencies);
            let is_first_token = state.emitted_tokens == 0;
            record_decode_queue_breakdown(
                state,
                scheduler,
                worker_runtime,
                decode_worker_slots_per_gpu,
                decode_start_s,
            );
            decode_iterations.push(ServingDecodeIterationObservation {
                iteration_idx: decode_iterations.len().min(u32::MAX as usize) as u32,
                decode_nodes: vec![state.decode_node],
                request_indices: vec![state.request_idx],
                operation_ids: decode_ids.clone(),
                start_s: decode_start_s,
                finish_s,
                latency_s: (finish_s - decode_start_s).max(0.0),
                batch_tokens: state.batch_size.max(1),
                first_token_batch_tokens: if is_first_token {
                    state.batch_size.max(1)
                } else {
                    0
                },
                tail_token_batch_tokens: if is_first_token {
                    0
                } else {
                    state.batch_size.max(1)
                },
            });
            record_decode_finish(state, decode_start_s, finish_s, dependencies);
            let assignments = assign_decode_workers_ready(
                worker_runtime,
                state,
                decode_worker_slots_per_gpu,
                decode_start_s,
                finish_s,
                &decode_ids,
            );
            state.worker_assignments.extend(assignments);
            if let Some(cancellation_s) = state.cancellation_s
                && cancellation_s <= finish_s + 1e-12
            {
                cancel_request(state, cancellation_s, "request cancelled during decode");
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule_continuous_decodes(
    scheduler: &mut ResourceScheduler,
    worker_runtime: &mut ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    decode_iterations: &mut Vec<ServingDecodeIterationObservation>,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    max_batch_tokens: Option<u32>,
    decode_worker_slots_per_gpu: usize,
) {
    let max_batch_tokens = max_batch_tokens.unwrap_or(u32::MAX).max(1);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let mut iteration_idx = 0_u32;

    while states
        .iter()
        .any(|state| state.remaining_tokens > 0 && state.status == ServingRequestStatus::Pending)
    {
        mark_decode_ready_cancellations(
            scheduler,
            worker_runtime,
            states,
            decode_worker_slots_per_gpu,
        );
        let Some((ready_idx, ready_s)) = states
            .iter()
            .enumerate()
            .filter(|(_, state)| {
                state.remaining_tokens > 0 && state.status == ServingRequestStatus::Pending
            })
            .map(|(idx, state)| {
                (
                    idx,
                    decode_candidate_ready_s(
                        state,
                        scheduler,
                        worker_runtime,
                        decode_worker_slots_per_gpu,
                    ),
                )
            })
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| compare_request_priority(states, left.0, right.0))
            })
        else {
            break;
        };
        let target_decode_node = decode_resource_base_node.map(|_| states[ready_idx].decode_node);
        let mut selected = Vec::new();
        let mut selected_batch_tokens = 0_u32;

        let mut candidates = states
            .iter()
            .enumerate()
            .filter_map(|(idx, state)| {
                if state.remaining_tokens == 0 || state.status != ServingRequestStatus::Pending {
                    return None;
                }
                if let Some(target_decode_node) = target_decode_node
                    && state.decode_node != target_decode_node
                {
                    return None;
                }
                let state_ready_s = decode_candidate_ready_s(
                    state,
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                );
                if state_ready_s > ready_s + 1e-12 {
                    return None;
                }
                if let Some(cancellation_s) = state.cancellation_s
                    && cancellation_s <= state_ready_s + 1e-12
                {
                    return None;
                }
                Some(idx)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| compare_request_priority(states, *left, *right));

        for idx in candidates {
            let state = &states[idx];
            let request_batch = state.batch_size.max(1);
            if selected.is_empty()
                || selected_batch_tokens.saturating_add(request_batch) <= max_batch_tokens
            {
                selected.push(idx);
                selected_batch_tokens = selected_batch_tokens.saturating_add(request_batch);
            }
        }
        if selected.is_empty() {
            continue;
        }

        let mut first_token_batch = 0_u32;
        let mut tail_token_batch = 0_u32;
        let mut dependencies = Vec::new();
        for idx in &selected {
            let state = &states[*idx];
            if state.emitted_tokens == 0 {
                first_token_batch = first_token_batch.saturating_add(state.batch_size);
            } else {
                tail_token_batch = tail_token_batch.saturating_add(state.batch_size);
            }
            dependencies.extend_from_slice(&state.dependencies);
        }
        dependencies.sort_unstable();
        dependencies.dedup();

        let effective_batch =
            f64::from(first_token_batch) + f64::from(tail_token_batch) * decode_tail_scale;
        let scale = effective_batch / f64::from(base_batch_size.max(1));
        let mut token_ops = decode_one_score.operations.clone();
        scale_operations(&mut token_ops, scale);
        if let Some(target_decode_node) = target_decode_node {
            remap_operations_to_node(
                &mut token_ops,
                decode_resource_base_node,
                target_decode_node,
            );
        }
        let (preview_decode_start_s, _) =
            preview_trace_span(scheduler, &token_ops, ready_s, &dependencies);
        let rejected_for_decode_queue = selected
            .iter()
            .filter_map(|idx| {
                decode_queue_delay_rejection(&states[*idx], preview_decode_start_s).map(
                    |(queue_delay_s, max_decode_queue_delay_s)| {
                        (*idx, queue_delay_s, max_decode_queue_delay_s)
                    },
                )
            })
            .collect::<Vec<_>>();
        let timed_out_for_decode_queue = selected
            .iter()
            .filter_map(|idx| {
                decode_iteration_queue_timeout(&states[*idx], preview_decode_start_s).map(
                    |(queue_delay_s, max_queue_delay_s)| (*idx, queue_delay_s, max_queue_delay_s),
                )
            })
            .collect::<Vec<_>>();
        if !rejected_for_decode_queue.is_empty() || !timed_out_for_decode_queue.is_empty() {
            for (idx, queue_delay_s, max_decode_queue_delay_s) in rejected_for_decode_queue {
                record_decode_queue_breakdown(
                    &mut states[idx],
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                reject_decode_queue_admission(
                    &mut states[idx],
                    preview_decode_start_s,
                    queue_delay_s,
                    max_decode_queue_delay_s,
                );
            }
            for (idx, queue_delay_s, max_queue_delay_s) in timed_out_for_decode_queue {
                record_decode_queue_breakdown(
                    &mut states[idx],
                    scheduler,
                    worker_runtime,
                    decode_worker_slots_per_gpu,
                    preview_decode_start_s,
                );
                timeout_decode_iteration_queue(&mut states[idx], queue_delay_s, max_queue_delay_s);
            }
            continue;
        }
        let decode_ids = schedule_trace(
            scheduler,
            &format!(
                "decode iteration {iteration_idx} batch {}",
                selected_batch_tokens
            ),
            &token_ops,
            ready_s,
            &dependencies,
        );
        let (decode_start_s, _) = operation_span(scheduler, &decode_ids);
        let terminal_dependencies = terminal_ids(&token_ops, &decode_ids);
        let finish_s = operation_finish_s(scheduler, &terminal_dependencies);
        let request_indices = selected
            .iter()
            .map(|idx| states[*idx].request_idx)
            .collect::<Vec<_>>();
        let decode_nodes = selected_decode_nodes(states, &selected);
        decode_iterations.push(ServingDecodeIterationObservation {
            iteration_idx,
            decode_nodes,
            request_indices,
            operation_ids: decode_ids.clone(),
            start_s: decode_start_s,
            finish_s,
            latency_s: (finish_s - decode_start_s).max(0.0),
            batch_tokens: selected_batch_tokens,
            first_token_batch_tokens: first_token_batch,
            tail_token_batch_tokens: tail_token_batch,
        });
        for idx in &selected {
            record_decode_queue_breakdown(
                &mut states[*idx],
                scheduler,
                worker_runtime,
                decode_worker_slots_per_gpu,
                decode_start_s,
            );
        }
        let worker_assignments = assign_selected_decode_workers_ready(
            worker_runtime,
            states,
            &selected,
            decode_worker_slots_per_gpu,
            decode_start_s,
            finish_s,
            &decode_ids,
        );

        let worker_assignments_by_idx = worker_assignments.into_iter().collect::<BTreeMap<_, _>>();
        for idx in selected {
            if let Some(assignments) = worker_assignments_by_idx.get(&idx) {
                states[idx].worker_assignments.extend(assignments.clone());
            }
            record_decode_finish(
                &mut states[idx],
                decode_start_s,
                finish_s,
                terminal_dependencies.clone(),
            );
            if let Some(cancellation_s) = states[idx].cancellation_s
                && cancellation_s <= finish_s + 1e-12
            {
                cancel_request(
                    &mut states[idx],
                    cancellation_s,
                    "request cancelled during decode",
                );
            }
        }
        iteration_idx += 1;
    }
}

fn mark_decode_ready_cancellations(
    scheduler: &ResourceScheduler,
    worker_runtime: &ServingWorkerRuntime,
    states: &mut [DecodeRequestState],
    decode_worker_slots_per_gpu: usize,
) {
    for state in states {
        if state.status != ServingRequestStatus::Pending || state.remaining_tokens == 0 {
            continue;
        }
        let ready_s = decode_candidate_ready_s(
            state,
            scheduler,
            worker_runtime,
            decode_worker_slots_per_gpu,
        );
        if let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= ready_s + 1e-12
        {
            cancel_request(
                state,
                cancellation_s,
                "request cancelled before decode iteration",
            );
        }
    }
}

fn selected_decode_nodes(states: &[DecodeRequestState], selected: &[usize]) -> Vec<NodeId> {
    let mut nodes = selected
        .iter()
        .map(|idx| states[*idx].decode_node)
        .collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

fn record_decode_finish(
    state: &mut DecodeRequestState,
    start_s: f64,
    finish_s: f64,
    dependencies: Vec<usize>,
) {
    if state.first_decode_start_s.is_none() {
        state.first_decode_start_s = Some(start_s);
    }
    if state.first_decode_finish_s.is_none() {
        state.first_decode_finish_s = Some(finish_s);
    }
    state.last_decode_finish_s = Some(finish_s);
    state.decode_token_start_s.push(start_s);
    state.decode_token_finish_s.push(finish_s);
    state.dependencies = dependencies;
    state.emitted_tokens += 1;
    state.remaining_tokens = state.remaining_tokens.saturating_sub(1);
}

fn apply_terminal_statuses(states: &mut [DecodeRequestState], traffic: &ServingTraffic) {
    for state in states {
        if state.status != ServingRequestStatus::Pending {
            continue;
        }
        let Some(last_decode_finish_s) = state.last_decode_finish_s else {
            state.status = ServingRequestStatus::TimedOut;
            let status_time_s = [
                state.kv_finish_s,
                state.prefill_finish_s,
                state.prefill_start_s,
                state.arrival_s,
            ]
            .into_iter()
            .find(|value| value.is_finite())
            .unwrap_or(state.arrival_s);
            state.status_time_s = Some(status_time_s);
            set_request_rejection(
                state,
                request_rejection(
                    "decode",
                    "timeout",
                    "decode_scheduler",
                    "decode_not_scheduled",
                    "request did not complete decode scheduling".to_string(),
                ),
            );
            continue;
        };
        if let Some(cancellation_s) = state.cancellation_s
            && cancellation_s <= last_decode_finish_s + 1e-12
        {
            cancel_request(state, cancellation_s, "request cancelled before completion");
            continue;
        }
        let e2el_s = (last_decode_finish_s - state.arrival_s).max(0.0);
        let timeout_s = state.request_timeout_s.or(traffic.request_timeout_s);
        if let Some(timeout_s) = timeout_s
            && e2el_s > timeout_s + 1e-12
        {
            state.status = ServingRequestStatus::TimedOut;
            state.status_time_s = Some(last_decode_finish_s);
            let message =
                format!("request timed out: e2el {e2el_s:.6}s > request_timeout {timeout_s:.6}s");
            set_request_rejection(
                state,
                with_rejection_details(
                    request_rejection(
                        "end_to_end",
                        "timeout",
                        "request_timeout",
                        "request_timeout_exceeded",
                        message,
                    ),
                    Some(e2el_s),
                    Some(timeout_s),
                    Some("s"),
                    Some(
                        "increase request_timeout, reduce load, or choose a faster serving config",
                    ),
                ),
            );
        } else {
            state.status = ServingRequestStatus::Completed;
            state.status_time_s = Some(last_decode_finish_s);
            state.failure_reason = None;
            state.failure_rejection = None;
        }
    }
}

fn inter_token_latencies(token_finish_s: &[f64]) -> Vec<f64> {
    token_finish_s
        .windows(2)
        .filter_map(|window| {
            let delta_s = window[1] - window[0];
            if delta_s.is_finite() && delta_s >= 0.0 {
                Some(delta_s)
            } else {
                None
            }
        })
        .collect()
}

fn transfer_resources(bottlenecks: &[String]) -> Vec<String> {
    let mut resources = bottlenecks.to_vec();
    if resources.is_empty() {
        resources.push("transfer: local".to_string());
    }
    resources.sort();
    resources.dedup();
    resources
}

fn kv_transfer_scheduler_resources(
    paths: &[ServingKvTransferPathObservation],
    bottlenecks: &[String],
) -> Vec<String> {
    let mut resources = paths
        .iter()
        .flat_map(|path| path.resource_details.iter().map(kv_route_resource_id))
        .collect::<Vec<_>>();
    resources.sort();
    resources.dedup();

    if resources.is_empty() {
        transfer_resources(bottlenecks)
    } else {
        resources
    }
}

fn kv_transfer_paths(
    cluster: &Cluster,
    source_gpus: &[GpuAddr],
    destination_gpus: &[GpuAddr],
    bytes: Bytes,
) -> Vec<ServingKvTransferPathObservation> {
    if bytes.as_bytes() == 0 || source_gpus.is_empty() || destination_gpus.is_empty() {
        return Vec::new();
    }

    let graph = TopologyGraph::from_cluster(cluster);
    let mut paths = Vec::new();
    for source in source_gpus {
        for destination in destination_gpus {
            if source == destination {
                continue;
            }
            let path = if source.node_id == destination.node_id {
                let (latency_s, bandwidth) =
                    Solver::intra_node_link_profile(cluster, source.node_id);
                ServingKvTransferPathObservation {
                    source: *source,
                    destination: *destination,
                    latency_s,
                    bottleneck_bandwidth_gbps: bandwidth.as_gigabits_per_sec(),
                    resources: vec![format!("node {} intra-node fabric", source.node_id)],
                    resource_details: vec![intra_node_kv_path_resource(
                        *source,
                        *destination,
                        latency_s,
                        bandwidth.as_gigabits_per_sec(),
                    )],
                }
            } else {
                let Some(path) = graph.route_between_gpus(*source, *destination, bytes) else {
                    continue;
                };
                let resource_details = path
                    .resources
                    .iter()
                    .map(kv_path_resource_from_routed)
                    .collect();
                ServingKvTransferPathObservation {
                    source: *source,
                    destination: *destination,
                    latency_s: path.latency_s,
                    bottleneck_bandwidth_gbps: path.bottleneck_bandwidth.as_gigabits_per_sec(),
                    resources: path.labels,
                    resource_details,
                }
            };
            paths.push(path);
        }
    }

    paths.sort_by(|left, right| {
        left.source
            .cmp(&right.source)
            .then_with(|| left.destination.cmp(&right.destination))
    });
    paths
}

fn intra_node_kv_path_resource(
    source: GpuAddr,
    destination: GpuAddr,
    latency_s: f64,
    bandwidth_gbps: f64,
) -> ServingKvTransferPathResourceObservation {
    ServingKvTransferPathResourceObservation {
        kind: "intra_node_fabric".to_string(),
        label: format!("node {} intra-node fabric", source.node_id),
        bandwidth_gbps,
        latency_s,
        rail_id: None,
        from: Some(kv_path_gpu_endpoint(source)),
        to: Some(kv_path_gpu_endpoint(destination)),
    }
}

fn kv_path_resource_from_routed(
    resource: &RoutedResource,
) -> ServingKvTransferPathResourceObservation {
    ServingKvTransferPathResourceObservation {
        kind: resource.kind.as_str().to_string(),
        label: resource.label.clone(),
        bandwidth_gbps: resource.bandwidth.as_gigabits_per_sec(),
        latency_s: resource.latency_s,
        rail_id: resource.rail_id,
        from: Some(kv_path_endpoint_from_graph(&resource.from)),
        to: Some(kv_path_endpoint_from_graph(&resource.to)),
    }
}

fn kv_path_endpoint_from_graph(
    resource: &GraphResource,
) -> ServingKvTransferPathEndpointObservation {
    match resource {
        GraphResource::Gpu(gpu) => kv_path_gpu_endpoint(*gpu),
        GraphResource::IntraNode { node_id } => ServingKvTransferPathEndpointObservation {
            kind: "intra_node".to_string(),
            node_id: Some(*node_id),
            local_gpu_id: None,
            nic_id: None,
            rail_id: None,
        },
        GraphResource::Nic {
            node_id,
            nic_id,
            rail_id,
        } => ServingKvTransferPathEndpointObservation {
            kind: "nic".to_string(),
            node_id: Some(*node_id),
            local_gpu_id: None,
            nic_id: Some(*nic_id),
            rail_id: Some(*rail_id),
        },
    }
}

fn kv_path_gpu_endpoint(gpu: GpuAddr) -> ServingKvTransferPathEndpointObservation {
    ServingKvTransferPathEndpointObservation {
        kind: "gpu".to_string(),
        node_id: Some(gpu.node_id),
        local_gpu_id: Some(gpu.local_gpu_id),
        nic_id: None,
        rail_id: None,
    }
}

#[derive(Clone, Debug)]
struct KvRouteResourceAccumulator {
    resource_id: String,
    kind: String,
    label: String,
    request_indices: BTreeSet<u32>,
    path_observations: u64,
    transfer_bytes: u64,
    estimated_transfer_s: f64,
    min_bandwidth_gbps: f64,
    max_latency_s: f64,
    rail_id: Option<u32>,
    from: Option<ServingKvTransferPathEndpointObservation>,
    to: Option<ServingKvTransferPathEndpointObservation>,
}

fn kv_route_resource_summary(
    observations: &[ServingRequestObservation],
) -> Vec<ServingKvRouteResourceSummary> {
    let mut resources = BTreeMap::<String, KvRouteResourceAccumulator>::new();

    for observation in observations {
        if observation.kv_transfer_bytes == 0 || observation.kv_transfer_paths.is_empty() {
            continue;
        }
        let bytes_per_path = observation
            .kv_transfer_bytes
            .div_ceil(observation.kv_transfer_paths.len() as u64);
        for path in &observation.kv_transfer_paths {
            for resource in &path.resource_details {
                let resource_id = kv_route_resource_id(resource);
                let bytes_per_segment = bytes_per_path;
                let estimated_transfer_s = resource.latency_s
                    + transfer_seconds(bytes_per_segment, resource.bandwidth_gbps);
                let entry = resources.entry(resource_id.clone()).or_insert_with(|| {
                    KvRouteResourceAccumulator {
                        resource_id: resource_id.clone(),
                        kind: resource.kind.clone(),
                        label: resource.label.clone(),
                        request_indices: BTreeSet::new(),
                        path_observations: 0,
                        transfer_bytes: 0,
                        estimated_transfer_s: 0.0,
                        min_bandwidth_gbps: resource.bandwidth_gbps,
                        max_latency_s: resource.latency_s,
                        rail_id: resource.rail_id,
                        from: resource.from.clone(),
                        to: resource.to.clone(),
                    }
                });
                entry.request_indices.insert(observation.request_idx);
                entry.path_observations = entry.path_observations.saturating_add(1);
                entry.transfer_bytes = entry.transfer_bytes.saturating_add(bytes_per_segment);
                entry.estimated_transfer_s += estimated_transfer_s;
                if resource.bandwidth_gbps.is_finite()
                    && resource.bandwidth_gbps < entry.min_bandwidth_gbps
                {
                    entry.min_bandwidth_gbps = resource.bandwidth_gbps;
                }
                if resource.latency_s.is_finite() && resource.latency_s > entry.max_latency_s {
                    entry.max_latency_s = resource.latency_s;
                }
            }
        }
    }

    let mut summaries = resources
        .into_values()
        .map(|resource| ServingKvRouteResourceSummary {
            resource_id: resource.resource_id,
            kind: resource.kind,
            label: resource.label,
            request_count: resource.request_indices.len().min(u32::MAX as usize) as u32,
            path_observations: resource.path_observations,
            transfer_bytes: resource.transfer_bytes,
            estimated_transfer_s: resource.estimated_transfer_s,
            min_bandwidth_gbps: resource.min_bandwidth_gbps,
            max_latency_s: resource.max_latency_s,
            rail_id: resource.rail_id,
            from: resource.from,
            to: resource.to,
        })
        .collect::<Vec<_>>();
    summaries.sort_by(|left, right| {
        right
            .estimated_transfer_s
            .total_cmp(&left.estimated_transfer_s)
            .then_with(|| right.transfer_bytes.cmp(&left.transfer_bytes))
            .then_with(|| right.path_observations.cmp(&left.path_observations))
            .then_with(|| left.resource_id.cmp(&right.resource_id))
    });
    summaries
}

fn kv_route_topology_summary(
    summaries: &[ServingKvRouteResourceSummary],
) -> ServingKvRouteTopologySummary {
    let mut rail_ids = BTreeSet::new();
    let mut inter_node_route_resource_count = 0u32;
    let mut gpu_nic_route_resource_count = 0u32;
    let mut intra_node_route_resource_count = 0u32;
    let mut unrailed_inter_node_route_resource_count = 0u32;

    for summary in summaries {
        if let Some(rail_id) = summary.rail_id {
            rail_ids.insert(rail_id);
        }
        match summary.kind.as_str() {
            "inter_node_fabric" | "gpu_scoped_inter_node_fabric" => {
                inter_node_route_resource_count = inter_node_route_resource_count.saturating_add(1);
                if summary.rail_id.is_none() {
                    unrailed_inter_node_route_resource_count =
                        unrailed_inter_node_route_resource_count.saturating_add(1);
                }
            }
            "gpu_nic_local" => {
                gpu_nic_route_resource_count = gpu_nic_route_resource_count.saturating_add(1);
            }
            "intra_node_fabric" => {
                intra_node_route_resource_count = intra_node_route_resource_count.saturating_add(1);
            }
            _ => {}
        }
    }

    let rail_ids = rail_ids.into_iter().collect::<Vec<_>>();
    let single_rail_dependency = inter_node_route_resource_count > 0 && rail_ids.len() == 1;
    ServingKvRouteTopologySummary {
        route_resource_count: summaries.len().min(u32::MAX as usize) as u32,
        inter_node_route_resource_count,
        gpu_nic_route_resource_count,
        intra_node_route_resource_count,
        rail_count: rail_ids.len().min(u32::MAX as usize) as u32,
        single_rail_id: single_rail_dependency.then(|| rail_ids[0]),
        single_rail_dependency,
        unrailed_inter_node_route_resource_count,
        rail_ids,
    }
}

fn topology_bottleneck_observations(
    route_coverage: ServingRouteCoverage,
    kv_route_topology: &ServingKvRouteTopologySummary,
    kv_route_resources: &[ServingKvRouteResourceSummary],
    request_observations: &[ServingRequestObservation],
    phase_resource_utilization: &[ServingPhaseResourceUtilization],
) -> Vec<ServingTopologyBottleneckObservation> {
    let mut bottlenecks = Vec::new();

    if route_coverage.candidate_count > 0 && route_coverage.unroutable_candidate_count > 0 {
        let severity = if route_coverage.routable_candidate_count == 0 {
            "critical"
        } else {
            "warning"
        };
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "kv_transfer_route".to_string(),
            code: "partial_route_coverage".to_string(),
            severity: severity.to_string(),
            observed: Some(f64::from(route_coverage.routable_candidate_count)),
            limit: Some(f64::from(route_coverage.candidate_count)),
            unit: Some("routes".to_string()),
            message: format!(
                "only {} of {} prefill/decode route candidates are routable; {} candidates are unroutable",
                route_coverage.routable_candidate_count,
                route_coverage.candidate_count,
                route_coverage.unroutable_candidate_count
            ),
            remediation: Some(
                "choose connected prefill/decode pools, add missing inter-node links, or switch to topology-aware routing".to_string(),
            ),
        });
    }

    if kv_route_topology.single_rail_dependency {
        let rail_label = kv_route_topology
            .single_rail_id
            .map(|rail| rail.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "kv_route_rail".to_string(),
            code: "single_rail_dependency".to_string(),
            severity: "warning".to_string(),
            observed: Some(f64::from(kv_route_topology.rail_count)),
            limit: Some(2.0),
            unit: Some("rails".to_string()),
            message: format!(
                "KV transfer routes use a single inter-node rail ({rail_label}); rail failure or congestion can affect all routed KV handoffs"
            ),
            remediation: Some(
                "add rail-diverse custom links, adjust GPU/NIC locality, or choose prefill/decode pools with multiple routable rails".to_string(),
            ),
        });
    }

    if kv_route_topology.unrailed_inter_node_route_resource_count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "kv_route_rail".to_string(),
            code: "unrailed_inter_node_routes".to_string(),
            severity: "info".to_string(),
            observed: Some(f64::from(
                kv_route_topology.unrailed_inter_node_route_resource_count,
            )),
            limit: Some(0.0),
            unit: Some("resources".to_string()),
            message: format!(
                "{} inter-node KV route resources do not carry rail IDs, so rail-diversity risk may be underreported",
                kv_route_topology.unrailed_inter_node_route_resource_count
            ),
            remediation: Some(
                "use NIC rail maps or rail-scoped custom inter-node links when modeling rail-sensitive clusters".to_string(),
            ),
        });
    }

    bottlenecks.extend(route_locality_bottleneck_observations(kv_route_resources));
    bottlenecks.extend(dynamic_route_contention_bottleneck_observations(
        request_observations,
        phase_resource_utilization,
    ));

    bottlenecks
}

fn topology_domain_bottleneck_observations(
    cluster: &Cluster,
    prefill_placement: &RankPlacement,
    decode_placement: &RankPlacement,
) -> Vec<ServingTopologyBottleneckObservation> {
    let prefill_nodes = placement_node_ids(prefill_placement);
    let decode_nodes = placement_node_ids(decode_placement);
    let all_nodes = union_node_ids(&prefill_nodes, &decode_nodes);

    let mut bottlenecks = Vec::new();
    append_topology_domain_bottlenecks(cluster, "placement", &all_nodes, &mut bottlenecks);
    append_topology_domain_bottlenecks(cluster, "prefill", &prefill_nodes, &mut bottlenecks);
    append_topology_domain_bottlenecks(cluster, "decode", &decode_nodes, &mut bottlenecks);
    bottlenecks
}

fn placement_node_ids(placement: &RankPlacement) -> Vec<NodeId> {
    placement
        .rank_to_gpu
        .iter()
        .map(|gpu| gpu.node_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn union_node_ids(left: &[NodeId], right: &[NodeId]) -> Vec<NodeId> {
    left.iter()
        .copied()
        .chain(right.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn serving_pool_topology_summary(
    cluster: Option<&Cluster>,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
) -> ServingPoolTopologySummary {
    let prefill_node_set = prefill_nodes.iter().copied().collect::<BTreeSet<_>>();
    let decode_node_set = decode_nodes.iter().copied().collect::<BTreeSet<_>>();
    let prefill_nodes = prefill_node_set.iter().copied().collect::<Vec<_>>();
    let decode_nodes = decode_node_set.iter().copied().collect::<Vec<_>>();
    let shared_node_count = prefill_node_set.intersection(&decode_node_set).count();

    ServingPoolTopologySummary {
        prefill_node_count: prefill_nodes.len(),
        decode_node_count: decode_nodes.len(),
        shared_node_count,
        dedicated_prefill_node_count: prefill_nodes.len().saturating_sub(shared_node_count),
        dedicated_decode_node_count: decode_nodes.len().saturating_sub(shared_node_count),
        prefill_racks: topology_summary_domain_values(cluster, &prefill_nodes, |node| {
            node.topology.rack.as_deref()
        }),
        decode_racks: topology_summary_domain_values(cluster, &decode_nodes, |node| {
            node.topology.rack.as_deref()
        }),
        prefill_islands: topology_summary_domain_values(cluster, &prefill_nodes, |node| {
            node.topology.island.as_deref()
        }),
        decode_islands: topology_summary_domain_values(cluster, &decode_nodes, |node| {
            node.topology.island.as_deref()
        }),
        prefill_failure_domains: topology_summary_domain_values(cluster, &prefill_nodes, |node| {
            node.topology.failure_domain.as_deref()
        }),
        decode_failure_domains: topology_summary_domain_values(cluster, &decode_nodes, |node| {
            node.topology.failure_domain.as_deref()
        }),
        prefill_node_labels: topology_summary_node_labels(cluster, &prefill_nodes),
        decode_node_labels: topology_summary_node_labels(cluster, &decode_nodes),
    }
}

fn topology_summary_domain_values(
    cluster: Option<&Cluster>,
    node_ids: &[NodeId],
    domain: impl Fn(&Node) -> Option<&str>,
) -> Vec<String> {
    let Some(cluster) = cluster else {
        return Vec::new();
    };

    node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id).and_then(&domain))
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn topology_summary_node_labels(cluster: Option<&Cluster>, node_ids: &[NodeId]) -> Vec<String> {
    let Some(cluster) = cluster else {
        return Vec::new();
    };

    node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id))
        .flat_map(|node| node.topology.labels.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn append_topology_domain_bottlenecks(
    cluster: &Cluster,
    phase: &str,
    node_ids: &[NodeId],
    bottlenecks: &mut Vec<ServingTopologyBottleneckObservation>,
) {
    if node_ids.len() < 2 {
        return;
    }
    append_single_domain_bottleneck(
        phase,
        "rack",
        node_ids,
        topology_domain_values(cluster, node_ids, |node| node.topology.rack.as_deref()),
        bottlenecks,
    );
    append_single_domain_bottleneck(
        phase,
        "island",
        node_ids,
        topology_domain_values(cluster, node_ids, |node| node.topology.island.as_deref()),
        bottlenecks,
    );
    append_single_domain_bottleneck(
        phase,
        "failure_domain",
        node_ids,
        topology_domain_values(cluster, node_ids, |node| {
            node.topology.failure_domain.as_deref()
        }),
        bottlenecks,
    );
}

fn topology_domain_values(
    cluster: &Cluster,
    node_ids: &[NodeId],
    domain: impl Fn(&Node) -> Option<&str>,
) -> Option<Vec<String>> {
    let mut values = Vec::new();
    for node_id in node_ids {
        let node = cluster.node(*node_id)?;
        values.push(domain(node)?.to_string());
    }
    Some(values)
}

fn append_single_domain_bottleneck(
    phase: &str,
    domain: &str,
    node_ids: &[NodeId],
    values: Option<Vec<String>>,
    bottlenecks: &mut Vec<ServingTopologyBottleneckObservation>,
) {
    let Some(values) = values else {
        return;
    };
    let distinct = values.iter().collect::<BTreeSet<_>>();
    if distinct.len() != 1 {
        return;
    }
    let domain_value = values
        .first()
        .expect("domain values exist when node_ids has at least two");
    bottlenecks.push(ServingTopologyBottleneckObservation {
        phase: phase.to_string(),
        category: "topology".to_string(),
        resource: format!("{domain}:{domain_value}"),
        code: format!("single_{domain}_placement"),
        severity: "warning".to_string(),
        observed: Some(1.0),
        limit: Some(2.0),
        unit: Some("domains".to_string()),
        message: format!(
            "{phase} placement uses {} selected nodes but only one {domain} ({domain_value}); a domain outage can affect the whole selected placement scope",
            node_ids.len()
        ),
        remediation: Some(format!(
            "spread selected ranks or service pools across multiple {domain}s, add topology-domain placement constraints, or lower topology_risk_penalty_weight if this concentration is intentional"
        )),
    });
}

fn kv_route_constraint_rejections(
    topology: &ServingKvRouteTopologySummary,
    resources: &[ServingKvRouteResourceSummary],
    constraints: ServingKvRouteConstraints,
) -> Vec<ServingRejection> {
    if !constraints.any() {
        return Vec::new();
    }

    let mut rejections = Vec::new();
    if topology.inter_node_route_resource_count > 0 {
        if let Some(limit) = constraints.min_inter_node_rail_count
            && topology.rail_count < limit
        {
            rejections.push(ServingRejection {
                phase: "kv_transfer".to_string(),
                category: "topology".to_string(),
                resource: "kv_route_rail".to_string(),
                code: "kv_route_rail_count_below_min".to_string(),
                observed: Some(f64::from(topology.rail_count)),
                limit: Some(f64::from(limit)),
                unit: Some("rails".to_string()),
                remediation: Some(
                    "add rail-diverse custom links, adjust GPU/NIC locality, choose different prefill/decode pools, or lower serving.min_kv_route_rail_count"
                        .to_string(),
                ),
                message: format!(
                    "KV transfer routes use {} inter-node rails, below configured minimum {}",
                    topology.rail_count, limit
                ),
            });
        }
        if constraints.require_inter_node_rail_metadata
            && topology.unrailed_inter_node_route_resource_count > 0
        {
            rejections.push(ServingRejection {
                phase: "kv_transfer".to_string(),
                category: "topology".to_string(),
                resource: "kv_route_rail".to_string(),
                code: "kv_route_rail_metadata_missing".to_string(),
                observed: Some(f64::from(
                    topology.unrailed_inter_node_route_resource_count,
                )),
                limit: Some(0.0),
                unit: Some("resources".to_string()),
                remediation: Some(
                    "add NIC rail maps or rail-scoped custom inter-node links, or disable serving.require_kv_route_rail_metadata"
                        .to_string(),
                ),
                message: format!(
                    "{} inter-node KV route resources lack rail metadata",
                    topology.unrailed_inter_node_route_resource_count
                ),
            });
        }
    }

    if constraints.require_gpudirect {
        let host_staged_count = resources
            .iter()
            .filter(|resource| resource.kind == "gpu_nic_local")
            .filter(|resource| {
                route_label_contains_any(
                    &resource.label,
                    &["host-staged", "host_staged", "no-gpudirect", "no_gpudirect"],
                )
            })
            .count()
            .min(u32::MAX as usize) as u32;
        if host_staged_count > 0 {
            rejections.push(ServingRejection {
                phase: "kv_transfer".to_string(),
                category: "topology".to_string(),
                resource: "gpu_nic_local".to_string(),
                code: "host_staged_kv_path_disallowed".to_string(),
                observed: Some(f64::from(host_staged_count)),
                limit: Some(0.0),
                unit: Some("resources".to_string()),
                remediation: Some(
                    "choose GPUDirect-capable GPU/NIC locality, adjust placement tags, or disable serving.require_gpudirect_kv_paths"
                        .to_string(),
                ),
                message: format!(
                    "{} GPU-to-NIC KV route resources are host-staged or lack GPUDirect",
                    host_staged_count
                ),
            });
        }
    }

    rejections
}

fn dynamic_route_contention_bottleneck_observations(
    request_observations: &[ServingRequestObservation],
    phase_resource_utilization: &[ServingPhaseResourceUtilization],
) -> Vec<ServingTopologyBottleneckObservation> {
    let mut bottlenecks = Vec::new();
    let queued_requests = request_observations
        .iter()
        .filter(|observation| observation.kv_resource_queue_s.is_finite())
        .filter(|observation| observation.kv_resource_queue_s > 0.0)
        .collect::<Vec<_>>();
    if !queued_requests.is_empty() {
        let total_queue_s = queued_requests
            .iter()
            .map(|observation| observation.kv_resource_queue_s)
            .sum::<f64>();
        let max_queue_s = queued_requests
            .iter()
            .map(|observation| observation.kv_resource_queue_s)
            .fold(0.0_f64, f64::max);
        let queued_count = queued_requests.len().min(u32::MAX as usize) as u32;
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "contention".to_string(),
            resource: "kv_route_resource".to_string(),
            code: "kv_route_resource_queueing".to_string(),
            severity: "warning".to_string(),
            observed: Some(max_queue_s * 1000.0),
            limit: Some(0.0),
            unit: Some("ms".to_string()),
            message: format!(
                "{} requests queued on modeled KV route resources; max route-resource queue {:.3} ms, mean queued-request route-resource queue {:.3} ms",
                queued_count,
                max_queue_s * 1000.0,
                total_queue_s / queued_requests.len() as f64 * 1000.0
            ),
            remediation: Some(
                "add route diversity, increase KV-transfer worker or fabric capacity, reduce burstiness, or choose prefill/decode pools with less shared-route contention"
                    .to_string(),
            ),
        });
    }

    if let Some(hot_route) = phase_resource_utilization
        .iter()
        .filter(|utilization| utilization.phase == "kv_transfer")
        .filter(|utilization| utilization.resource_kind == "kv_route")
        .filter(|utilization| utilization.operation_count > 1)
        .filter(|utilization| utilization.utilization >= 0.85)
        .max_by(|left, right| {
            left.utilization
                .total_cmp(&right.utilization)
                .then_with(|| left.busy_s.total_cmp(&right.busy_s))
        })
    {
        let severity = if hot_route.utilization >= 0.98 {
            "critical"
        } else {
            "warning"
        };
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "contention".to_string(),
            resource: hot_route.resource.clone(),
            code: "hot_kv_route_resource".to_string(),
            severity: severity.to_string(),
            observed: Some(hot_route.utilization),
            limit: Some(0.85),
            unit: Some("utilization".to_string()),
            message: format!(
                "KV route resource '{}' reached {:.1}% utilization across {} operations",
                hot_route.resource,
                hot_route.utilization * 100.0,
                hot_route.operation_count
            ),
            remediation: Some(
                "spread KV handoffs across more rails or decode replicas, increase interconnect capacity, or lower offered load for this placement"
                    .to_string(),
            ),
        });
    }

    bottlenecks
}

#[derive(Clone, Debug, Default)]
struct RouteLocalityDiagnostics {
    count: u32,
    transfer_bytes: u64,
    min_bandwidth_gbps: Option<f64>,
    max_latency_s: Option<f64>,
}

impl RouteLocalityDiagnostics {
    fn observe(&mut self, summary: &ServingKvRouteResourceSummary) {
        self.count = self.count.saturating_add(1);
        self.transfer_bytes = self.transfer_bytes.saturating_add(summary.transfer_bytes);
        if summary.min_bandwidth_gbps.is_finite() {
            self.min_bandwidth_gbps = Some(
                self.min_bandwidth_gbps
                    .map(|current| current.min(summary.min_bandwidth_gbps))
                    .unwrap_or(summary.min_bandwidth_gbps),
            );
        }
        if summary.max_latency_s.is_finite() {
            self.max_latency_s = Some(
                self.max_latency_s
                    .map(|current| current.max(summary.max_latency_s))
                    .unwrap_or(summary.max_latency_s),
            );
        }
    }
}

fn route_locality_bottleneck_observations(
    kv_route_resources: &[ServingKvRouteResourceSummary],
) -> Vec<ServingTopologyBottleneckObservation> {
    let mut host_staged = RouteLocalityDiagnostics::default();
    let mut cross_socket = RouteLocalityDiagnostics::default();
    let mut slow_gpu_nic = RouteLocalityDiagnostics::default();
    let slow_gpu_nic_bandwidth_gbps = slow_gpu_nic_bandwidth_threshold(kv_route_resources);

    for summary in kv_route_resources {
        if summary.kind != "gpu_nic_local" {
            continue;
        }
        if route_label_contains_any(
            &summary.label,
            &["host-staged", "host_staged", "no-gpudirect", "no_gpudirect"],
        ) {
            host_staged.observe(summary);
        }
        if route_label_contains_any(
            &summary.label,
            &["cross-socket", "cross_socket", "cross socket"],
        ) {
            cross_socket.observe(summary);
        }
        if slow_gpu_nic_bandwidth_gbps.is_some_and(|threshold| {
            summary.min_bandwidth_gbps.is_finite() && summary.min_bandwidth_gbps < threshold
        }) {
            slow_gpu_nic.observe(summary);
        }
    }

    let mut bottlenecks = Vec::new();
    if host_staged.count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "gpu_nic_local".to_string(),
            code: "host_staged_kv_path".to_string(),
            severity: "warning".to_string(),
            observed: host_staged.min_bandwidth_gbps,
            limit: None,
            unit: Some("gbps".to_string()),
            message: format!(
                "{} GPU-to-NIC KV route resources are host-staged or lack GPUDirect; min bandwidth {} Gbps, max latency {} us across {} transfer bytes",
                host_staged.count,
                format_optional_f64(host_staged.min_bandwidth_gbps),
                format_optional_f64(host_staged.max_latency_s.map(|latency_s| latency_s * 1e6)),
                host_staged.transfer_bytes
            ),
            remediation: Some(
                "prefer GPUDirect-capable GPU/NIC paths, adjust GPU-to-NIC locality, or model the host-staged copy engine explicitly".to_string(),
            ),
        });
    }
    if cross_socket.count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "gpu_nic_local".to_string(),
            code: "cross_socket_kv_path".to_string(),
            severity: "warning".to_string(),
            observed: cross_socket.max_latency_s.map(|latency_s| latency_s * 1e6),
            limit: None,
            unit: Some("us".to_string()),
            message: format!(
                "{} GPU-to-NIC KV route resources cross sockets; max local-path latency {} us across {} transfer bytes",
                cross_socket.count,
                format_optional_f64(cross_socket.max_latency_s.map(|latency_s| latency_s * 1e6)),
                cross_socket.transfer_bytes
            ),
            remediation: Some(
                "prefer same-socket GPU/NIC affinity, add explicit placement constraints, or model NUMA/PCIe domains before relying on this route".to_string(),
            ),
        });
    }
    if slow_gpu_nic.count > 0 {
        bottlenecks.push(ServingTopologyBottleneckObservation {
            phase: "kv_transfer".to_string(),
            category: "topology".to_string(),
            resource: "gpu_nic_local".to_string(),
            code: "slow_gpu_nic_kv_path".to_string(),
            severity: "info".to_string(),
            observed: slow_gpu_nic.min_bandwidth_gbps,
            limit: slow_gpu_nic_bandwidth_gbps,
            unit: Some("gbps".to_string()),
            message: format!(
                "{} GPU-to-NIC KV route resources are much slower than the selected inter-node fabric; min bandwidth {} Gbps below threshold {} Gbps",
                slow_gpu_nic.count,
                format_optional_f64(slow_gpu_nic.min_bandwidth_gbps),
                format_optional_f64(slow_gpu_nic_bandwidth_gbps)
            ),
            remediation: Some(
                "check GPU/NIC affinity, PCIe generation, GPUDirect availability, and whether local copy paths should be calibrated separately".to_string(),
            ),
        });
    }

    bottlenecks
}

fn slow_gpu_nic_bandwidth_threshold(
    kv_route_resources: &[ServingKvRouteResourceSummary],
) -> Option<f64> {
    kv_route_resources
        .iter()
        .filter(|summary| {
            matches!(
                summary.kind.as_str(),
                "inter_node_fabric" | "gpu_scoped_inter_node_fabric"
            ) && summary.min_bandwidth_gbps.is_finite()
        })
        .map(|summary| summary.min_bandwidth_gbps)
        .max_by(f64::total_cmp)
        .map(|bandwidth_gbps| bandwidth_gbps * 0.5)
}

fn route_label_contains_any(label: &str, needles: &[&str]) -> bool {
    let normalized = label.to_ascii_lowercase();
    needles.iter().any(|needle| normalized.contains(needle))
}

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "unknown".to_string())
}

fn kv_route_resource_id(resource: &ServingKvTransferPathResourceObservation) -> String {
    let left = kv_route_endpoint_key(resource.from.as_ref());
    let right = kv_route_endpoint_key(resource.to.as_ref());
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    format!(
        "kv_route:{}|{}|rail={}|{}<->{}",
        resource.kind,
        resource.label,
        optional_u32_key(resource.rail_id),
        first,
        second
    )
}

fn kv_route_endpoint_key(endpoint: Option<&ServingKvTransferPathEndpointObservation>) -> String {
    let Some(endpoint) = endpoint else {
        return "none".to_string();
    };
    format!(
        "{}:node={}:gpu={}:nic={}:rail={}",
        endpoint.kind,
        optional_u32_key(endpoint.node_id),
        optional_u32_key(endpoint.local_gpu_id),
        optional_u32_key(endpoint.nic_id),
        optional_u32_key(endpoint.rail_id)
    )
}

fn optional_u32_key(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn transfer_seconds(bytes: u64, bandwidth_gbps: f64) -> f64 {
    if bandwidth_gbps.is_finite() && bandwidth_gbps > 0.0 {
        bytes as f64 / (bandwidth_gbps * 1e9 / 8.0)
    } else {
        0.0
    }
}

fn schedule_trace(
    scheduler: &mut ResourceScheduler,
    prefix: &str,
    operations: &[SimOperation],
    earliest_start_s: f64,
    external_dependencies: &[usize],
) -> Vec<usize> {
    let mut scheduled_ids = Vec::with_capacity(operations.len());
    for operation in operations {
        let mut dependencies: Vec<_> = operation
            .dependencies
            .iter()
            .filter_map(|idx| scheduled_ids.get(*idx).copied())
            .collect();
        if operation.dependencies.is_empty() {
            dependencies.extend_from_slice(external_dependencies);
        }

        let scheduled_id = scheduler.schedule(
            format!("{prefix} {}", operation.name),
            operation.duration_s,
            earliest_start_s,
            &dependencies,
            operation.resources.clone(),
        );
        scheduled_ids.push(scheduled_id);
    }

    scheduled_ids
}

fn preview_trace_span(
    scheduler: &ResourceScheduler,
    operations: &[SimOperation],
    earliest_start_s: f64,
    external_dependencies: &[usize],
) -> (f64, f64) {
    let mut preview_scheduler = scheduler.clone();
    let scheduled_ids = schedule_trace(
        &mut preview_scheduler,
        "preview",
        operations,
        earliest_start_s,
        external_dependencies,
    );
    operation_span(&preview_scheduler, &scheduled_ids)
}

fn terminal_ids(operations: &[SimOperation], scheduled_ids: &[usize]) -> Vec<usize> {
    let mut depended_on = std::collections::HashSet::new();
    for operation in operations {
        for dependency in &operation.dependencies {
            depended_on.insert(*dependency);
        }
    }

    scheduled_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, scheduled_id)| {
            if depended_on.contains(&idx) {
                None
            } else {
                Some(*scheduled_id)
            }
        })
        .collect()
}

fn operation_span(scheduler: &ResourceScheduler, ids: &[usize]) -> (f64, f64) {
    let mut start_s = f64::INFINITY;
    let mut finish_s = 0.0_f64;
    for id in ids {
        if let Some(operation) = scheduler.operation(*id) {
            start_s = start_s.min(operation.start_s);
            finish_s = finish_s.max(operation.finish_s);
        }
    }

    if start_s.is_finite() {
        (start_s, finish_s)
    } else {
        (0.0, 0.0)
    }
}

fn operation_finish_s(scheduler: &ResourceScheduler, ids: &[usize]) -> f64 {
    ids.iter()
        .filter_map(|id| scheduler.operation(*id))
        .map(|operation| operation.finish_s)
        .fold(0.0, f64::max)
}

fn scale_operations(operations: &mut [SimOperation], scale: f64) {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    for operation in operations {
        operation.duration_s *= scale;
    }
}

fn single_placement_node(score: &ScoredParallelismConfig) -> Option<NodeId> {
    let nodes: BTreeSet<_> = placement_nodes(score).into_iter().collect();
    if nodes.len() == 1 {
        nodes.first().copied()
    } else {
        None
    }
}

fn placement_nodes(score: &ScoredParallelismConfig) -> Vec<NodeId> {
    let mut nodes: Vec<_> = score
        .placement
        .rank_to_gpu
        .iter()
        .map(|addr| addr.node_id)
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

fn routed_node_set(
    resource_base_node: Option<NodeId>,
    placement_nodes: &[NodeId],
    routed_node: NodeId,
) -> Vec<NodeId> {
    if resource_base_node.is_some() || placement_nodes.is_empty() {
        vec![routed_node]
    } else {
        placement_nodes.to_vec()
    }
}

fn routed_gpu_set(
    resource_base_node: Option<NodeId>,
    placement_gpus: &[GpuAddr],
    routed_node: NodeId,
) -> Vec<GpuAddr> {
    let mut gpus = if resource_base_node.is_some() || placement_gpus.is_empty() {
        placement_gpus
            .iter()
            .map(|addr| GpuAddr {
                node_id: routed_node,
                local_gpu_id: addr.local_gpu_id,
            })
            .collect::<Vec<_>>()
    } else {
        placement_gpus.to_vec()
    };
    if gpus.is_empty() {
        gpus.push(GpuAddr {
            node_id: routed_node,
            local_gpu_id: 0,
        });
    }
    gpus.sort_unstable();
    gpus.dedup();
    gpus
}

fn remap_operations_to_node(
    operations: &mut [SimOperation],
    base_node: Option<NodeId>,
    target_node: NodeId,
) {
    let Some(base_node) = base_node else {
        return;
    };
    if base_node == target_node {
        return;
    }

    for operation in operations {
        for resource in &mut operation.resources {
            *resource = remap_resource_to_node(resource, base_node, target_node);
        }
    }
}

fn remap_resource_to_node(resource: &str, base_node: NodeId, target_node: NodeId) -> String {
    let gpu_compute = format!("gpu compute node {base_node}");
    if resource == gpu_compute {
        return format!("gpu compute node {target_node}");
    }

    let gpu_hbm = format!("gpu HBM node {base_node}");
    if resource == gpu_hbm {
        return format!("gpu HBM node {target_node}");
    }

    let intra_node = format!("node {base_node} intra-node fabric");
    if resource == intra_node {
        return format!("node {target_node} intra-node fabric");
    }

    resource.to_string()
}

fn collect_metric(
    observations: &[ServingRequestObservation],
    metric: impl Fn(&ServingRequestObservation) -> f64,
) -> Vec<f64> {
    observations
        .iter()
        .map(metric)
        .filter(|value| value.is_finite())
        .collect()
}

struct PhaseResourceUtilizationAccumulator {
    busy_s: f64,
    operation_count: usize,
    first_start_s: f64,
    last_finish_s: f64,
}

impl Default for PhaseResourceUtilizationAccumulator {
    fn default() -> Self {
        Self {
            busy_s: 0.0,
            operation_count: 0,
            first_start_s: f64::INFINITY,
            last_finish_s: 0.0,
        }
    }
}

fn phase_resource_utilization(
    operations: &[ScheduledOperation],
    window_s: f64,
) -> Vec<ServingPhaseResourceUtilization> {
    let mut utilization: BTreeMap<(String, String, String), PhaseResourceUtilizationAccumulator> =
        BTreeMap::new();

    for operation in operations {
        let phase = operation_phase(&operation.name).to_string();
        let duration_s = (operation.finish_s - operation.start_s).max(0.0);
        for resource in &operation.resources {
            let resource_kind = resource_kind(resource).to_string();
            let entry = utilization
                .entry((phase.clone(), resource_kind, resource.clone()))
                .or_default();
            entry.busy_s += duration_s;
            entry.operation_count += 1;
            entry.first_start_s = entry.first_start_s.min(operation.start_s);
            entry.last_finish_s = entry.last_finish_s.max(operation.finish_s);
        }
    }

    let window_s = if window_s.is_finite() && window_s > 0.0 {
        window_s
    } else {
        0.0
    };
    let mut rows = utilization
        .into_iter()
        .map(
            |((phase, resource_kind, resource), accumulator)| ServingPhaseResourceUtilization {
                phase,
                resource_kind,
                resource,
                busy_s: accumulator.busy_s,
                utilization: if window_s > 0.0 {
                    (accumulator.busy_s / window_s).min(1.0)
                } else {
                    0.0
                },
                operation_count: accumulator.operation_count,
                first_start_s: if accumulator.first_start_s.is_finite() {
                    accumulator.first_start_s
                } else {
                    0.0
                },
                last_finish_s: accumulator.last_finish_s,
            },
        )
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .utilization
            .total_cmp(&left.utilization)
            .then_with(|| right.busy_s.total_cmp(&left.busy_s))
            .then_with(|| left.phase.cmp(&right.phase))
            .then_with(|| left.resource_kind.cmp(&right.resource_kind))
            .then_with(|| left.resource.cmp(&right.resource))
    });
    rows
}

fn operation_phase(name: &str) -> &'static str {
    if name.contains("kv-transfer") {
        "kv_transfer"
    } else if name.contains("prefill") {
        "prefill"
    } else if name.contains("decode") {
        "decode"
    } else {
        "other"
    }
}

fn resource_kind(resource: &str) -> &'static str {
    if resource.starts_with("gpu compute") {
        "gpu_compute"
    } else if resource.starts_with("gpu HBM") {
        "gpu_hbm"
    } else if resource.starts_with("kv_route:") {
        "kv_route"
    } else if resource.contains("intra-node fabric") {
        "intra_node_fabric"
    } else if resource == "KV transfer fabric/NIC path" {
        "kv_transfer_fabric"
    } else if resource.starts_with("custom ") {
        "inter_node_link"
    } else if resource.contains("fabric") {
        "fabric"
    } else if resource.contains("NIC") || resource.contains("nic") {
        "nic"
    } else {
        "other"
    }
}

struct WorkerObservationAccumulator {
    request_count: u32,
    completed_requests: u32,
    failed_requests: u32,
    input_tokens: u64,
    output_tokens: u64,
    queue_s: Vec<f64>,
    worker_queue_s: Vec<f64>,
    resource_queue_s: Vec<f64>,
    service_s: Vec<f64>,
    worker_slot_intervals: Vec<(f64, f64)>,
    first_start_s: f64,
    last_finish_s: f64,
    peak_prefill_tokens: u64,
    peak_decode_sequences: u32,
    peak_resident_tokens: u64,
    peak_kv_blocks: u64,
    peak_allocated_kv_tokens: u64,
    peak_kv_fragmentation_tokens: u64,
    peak_kv_block_table_bytes: u64,
    kv_cache_owner_slots: Vec<ServingWorkerKvSlotObservation>,
    decode_sequence_utilization: f64,
    resident_token_utilization: f64,
    kv_block_utilization: f64,
}

impl Default for WorkerObservationAccumulator {
    fn default() -> Self {
        Self {
            request_count: 0,
            completed_requests: 0,
            failed_requests: 0,
            input_tokens: 0,
            output_tokens: 0,
            queue_s: Vec::new(),
            worker_queue_s: Vec::new(),
            resource_queue_s: Vec::new(),
            service_s: Vec::new(),
            worker_slot_intervals: Vec::new(),
            first_start_s: f64::INFINITY,
            last_finish_s: 0.0,
            peak_prefill_tokens: 0,
            peak_decode_sequences: 0,
            peak_resident_tokens: 0,
            peak_kv_blocks: 0,
            peak_allocated_kv_tokens: 0,
            peak_kv_fragmentation_tokens: 0,
            peak_kv_block_table_bytes: 0,
            kv_cache_owner_slots: Vec::new(),
            decode_sequence_utilization: 0.0,
            resident_token_utilization: 0.0,
            kv_block_utilization: 0.0,
        }
    }
}

impl WorkerObservationAccumulator {
    fn record_common(&mut self, observation: &ServingRequestObservation) {
        self.request_count = self.request_count.saturating_add(1);
        if observation.status.is_completed() {
            self.completed_requests = self.completed_requests.saturating_add(1);
        } else {
            self.failed_requests = self.failed_requests.saturating_add(1);
        }
    }

    fn record_prefill(&mut self, observation: &ServingRequestObservation) {
        self.record_common(observation);
        self.input_tokens = self.input_tokens.saturating_add(
            u64::from(observation.batch_size) * u64::from(observation.effective_prefill_tokens),
        );
        if observation.prefill_start_s.is_finite() {
            push_finite(
                &mut self.queue_s,
                observation.prefill_worker_queue_s + observation.prefill_resource_queue_s,
            );
            push_finite(&mut self.worker_queue_s, observation.prefill_worker_queue_s);
            push_finite(
                &mut self.resource_queue_s,
                observation.prefill_resource_queue_s,
            );
            push_finite(&mut self.service_s, observation.prefill_s);
            self.first_start_s = self.first_start_s.min(observation.prefill_start_s);
        }
        if observation.prefill_finish_s.is_finite() {
            self.last_finish_s = self.last_finish_s.max(observation.prefill_finish_s);
        }
        let mut recorded_chunk_interval = false;
        for (start_s, finish_s) in observation
            .prefill_token_start_s
            .iter()
            .zip(observation.prefill_token_finish_s.iter())
        {
            if push_interval(&mut self.worker_slot_intervals, *start_s, *finish_s) {
                recorded_chunk_interval = true;
            }
        }
        if !recorded_chunk_interval && observation.status.is_admitted() {
            push_interval(
                &mut self.worker_slot_intervals,
                observation.prefill_start_s,
                observation.prefill_finish_s,
            );
        }
    }

    fn record_decode(&mut self, observation: &ServingRequestObservation) {
        self.record_common(observation);
        self.output_tokens = self.output_tokens.saturating_add(
            u64::from(observation.batch_size)
                * observation
                    .decode_token_finish_s
                    .len()
                    .min(u64::MAX as usize) as u64,
        );
        push_finite(
            &mut self.queue_s,
            observation.decode_worker_queue_s + observation.decode_resource_queue_s,
        );
        push_finite(&mut self.worker_queue_s, observation.decode_worker_queue_s);
        push_finite(
            &mut self.resource_queue_s,
            observation.decode_resource_queue_s,
        );
        push_finite(&mut self.service_s, observation.decode_s);
        if observation.first_decode_start_s.is_finite() {
            self.first_start_s = self.first_start_s.min(observation.first_decode_start_s);
        }
        if observation.last_decode_finish_s.is_finite() {
            self.last_finish_s = self.last_finish_s.max(observation.last_decode_finish_s);
        }
        let mut recorded_token_interval = false;
        for (start_s, finish_s) in observation
            .decode_token_start_s
            .iter()
            .zip(observation.decode_token_finish_s.iter())
        {
            if push_interval(&mut self.worker_slot_intervals, *start_s, *finish_s) {
                recorded_token_interval = true;
            }
        }
        if !recorded_token_interval {
            push_interval(
                &mut self.worker_slot_intervals,
                observation.first_decode_start_s,
                observation.last_decode_finish_s,
            );
        }
    }

    fn record_kv_transfer(&mut self, observation: &ServingRequestObservation) {
        self.record_common(observation);
        if observation.kv_start_s.is_finite() {
            push_finite(&mut self.queue_s, observation.kv_queue_s);
            push_finite(&mut self.worker_queue_s, observation.kv_worker_queue_s);
            push_finite(&mut self.resource_queue_s, observation.kv_resource_queue_s);
            self.first_start_s = self.first_start_s.min(observation.kv_start_s);
        }
        if observation.kv_finish_s.is_finite() {
            push_finite(&mut self.service_s, observation.kv_transfer_s);
            self.last_finish_s = self.last_finish_s.max(observation.kv_finish_s);
        }
        push_interval(
            &mut self.worker_slot_intervals,
            observation.kv_start_s,
            observation.kv_finish_s,
        );
    }

    fn apply_decode_capacity(&mut self, capacity: &ServingGpuCapacityObservation) {
        self.peak_decode_sequences = capacity.peak_decode_sequences;
        self.peak_resident_tokens = capacity.peak_resident_tokens;
        self.peak_kv_blocks = capacity.peak_kv_blocks;
        self.peak_allocated_kv_tokens = capacity.peak_allocated_kv_tokens;
        self.peak_kv_fragmentation_tokens = capacity.peak_kv_fragmentation_tokens;
        self.peak_kv_block_table_bytes = capacity.peak_kv_block_table_bytes;
        self.decode_sequence_utilization = capacity.decode_sequence_utilization;
        self.resident_token_utilization = capacity.resident_token_utilization;
        self.kv_block_utilization = capacity.kv_block_utilization;
    }

    fn apply_decode_worker_slot_capacity(&mut self, slots: Vec<ServingWorkerKvSlotObservation>) {
        self.kv_cache_owner_slots = slots;
    }

    fn apply_prefill_capacity(&mut self, capacity: &ServingGpuCapacityObservation) {
        self.peak_prefill_tokens = capacity.peak_prefill_tokens;
    }

    fn into_observation(
        self,
        phase: String,
        gpu: GpuAddr,
        configured_worker_slots: u32,
        deduplicate_intervals: bool,
    ) -> ServingWorkerObservation {
        let worker_slot_intervals = if deduplicate_intervals {
            deduplicate_intervals_by_span(&self.worker_slot_intervals)
        } else {
            self.worker_slot_intervals.clone()
        };
        let peak_active_worker_slots = peak_active_intervals(&worker_slot_intervals);
        let worker_slot_utilization =
            interval_utilization(&worker_slot_intervals, configured_worker_slots);
        ServingWorkerObservation {
            phase,
            node_id: gpu.node_id,
            local_gpu_id: gpu.local_gpu_id,
            configured_worker_slots,
            peak_active_worker_slots,
            worker_slot_utilization,
            request_count: self.request_count,
            completed_requests: self.completed_requests,
            failed_requests: self.failed_requests,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            queue_s: mean(&self.queue_s),
            queue_p95_s: percentile(self.queue_s.clone(), 0.95),
            queue_max_s: max_value(&self.queue_s),
            worker_queue_s: mean(&self.worker_queue_s),
            worker_queue_p95_s: percentile(self.worker_queue_s.clone(), 0.95),
            worker_queue_max_s: max_value(&self.worker_queue_s),
            resource_queue_s: mean(&self.resource_queue_s),
            resource_queue_p95_s: percentile(self.resource_queue_s.clone(), 0.95),
            resource_queue_max_s: max_value(&self.resource_queue_s),
            service_s: mean(&self.service_s),
            first_start_s: if self.first_start_s.is_finite() {
                self.first_start_s
            } else {
                0.0
            },
            last_finish_s: self.last_finish_s,
            peak_prefill_tokens: self.peak_prefill_tokens,
            peak_decode_sequences: self.peak_decode_sequences,
            peak_resident_tokens: self.peak_resident_tokens,
            peak_kv_blocks: self.peak_kv_blocks,
            peak_allocated_kv_tokens: self.peak_allocated_kv_tokens,
            peak_kv_fragmentation_tokens: self.peak_kv_fragmentation_tokens,
            peak_kv_block_table_bytes: self.peak_kv_block_table_bytes,
            kv_cache_owner_slots: self.kv_cache_owner_slots,
            decode_sequence_utilization: self.decode_sequence_utilization,
            resident_token_utilization: self.resident_token_utilization,
            kv_block_utilization: self.kv_block_utilization,
        }
    }
}

fn push_interval(intervals: &mut Vec<(f64, f64)>, start_s: f64, finish_s: f64) -> bool {
    if start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
        intervals.push((start_s, finish_s));
        true
    } else {
        false
    }
}

fn interval_events(intervals: &[(f64, f64)]) -> Vec<(f64, i32)> {
    let mut events = Vec::with_capacity(intervals.len() * 2);
    for (start_s, finish_s) in intervals {
        if start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
            events.push((*start_s, 1));
            events.push((*finish_s, -1));
        }
    }
    events.sort_by(|left, right| left.0.total_cmp(&right.0));
    events
}

fn deduplicate_intervals_by_span(intervals: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut deduplicated = intervals
        .iter()
        .copied()
        .filter(|(start_s, finish_s)| {
            start_s.is_finite() && finish_s.is_finite() && finish_s > start_s
        })
        .collect::<Vec<_>>();
    deduplicated.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    deduplicated.dedup_by(|left, right| {
        left.0.total_cmp(&right.0) == std::cmp::Ordering::Equal
            && left.1.total_cmp(&right.1) == std::cmp::Ordering::Equal
    });
    deduplicated
}

fn peak_active_intervals(intervals: &[(f64, f64)]) -> u32 {
    let events = interval_events(intervals);
    let mut active = 0_i32;
    let mut peak = 0_i32;
    let mut idx = 0;

    while idx < events.len() {
        let time_s = events[idx].0;
        let mut delta = 0_i32;
        while idx < events.len() && events[idx].0.total_cmp(&time_s) == std::cmp::Ordering::Equal {
            delta += events[idx].1;
            idx += 1;
        }
        active = (active + delta).max(0);
        peak = peak.max(active);
    }

    peak.max(0) as u32
}

fn interval_utilization(intervals: &[(f64, f64)], configured_slots: u32) -> f64 {
    if configured_slots == 0 {
        return 0.0;
    }
    let events = interval_events(intervals);
    if events.len() < 2 {
        return 0.0;
    }

    let first_s = events[0].0;
    let mut last_s = first_s;
    let mut active = 0_i32;
    let mut busy_slot_s = 0.0;
    let mut idx = 0;

    while idx < events.len() {
        let time_s = events[idx].0;
        if time_s > last_s && active > 0 {
            busy_slot_s += f64::from(active) * (time_s - last_s);
        }

        let mut delta = 0_i32;
        while idx < events.len() && events[idx].0.total_cmp(&time_s) == std::cmp::Ordering::Equal {
            delta += events[idx].1;
            idx += 1;
        }
        active = (active + delta).max(0);
        last_s = time_s;
    }

    let window_s = (last_s - first_s).max(0.0);
    if window_s <= 0.0 {
        0.0
    } else {
        (busy_slot_s / (window_s * f64::from(configured_slots))).max(0.0)
    }
}

fn phase_worker_slots(
    phase: &str,
    prefill_slots: u32,
    decode_slots: u32,
    kv_transfer_slots: Option<u32>,
) -> u32 {
    match phase {
        "prefill" => prefill_slots,
        "decode" => decode_slots,
        "kv_transfer" => kv_transfer_slots.unwrap_or(1),
        _ => 1,
    }
}

fn phase_deduplicates_worker_intervals(phase: &str, traffic: &ServingTraffic) -> bool {
    match phase {
        "prefill" => matches!(
            &traffic.prefill_batching,
            ServingPrefillBatching::Continuous { .. }
        ),
        "decode" => matches!(
            &traffic.decode_batching,
            ServingDecodeBatching::Continuous { .. }
        ),
        _ => false,
    }
}

fn kv_transfer_observation_gpus(observation: &ServingRequestObservation) -> Vec<GpuAddr> {
    let mut gpus = BTreeSet::new();
    gpus.extend(route_worker_gpus(
        &observation.prefill_route_gpus,
        observation.prefill_node,
    ));
    gpus.extend(route_worker_gpus(
        &observation.decode_route_gpus,
        observation.decode_node,
    ));
    gpus.into_iter().collect()
}

fn kv_worker_slot_capacity_observations(
    observations: &[ServingRequestObservation],
    traffic: &ServingTraffic,
) -> BTreeMap<GpuAddr, Vec<ServingWorkerKvSlotObservation>> {
    let mut slot_events: BTreeMap<(GpuAddr, u32), Vec<CapacityEvent>> = BTreeMap::new();
    for observation in observations {
        for ownership in &observation.kv_block_ownership {
            if !ownership.allocated_at_s.is_finite()
                || !ownership.released_at_s.is_finite()
                || ownership.released_at_s < ownership.allocated_at_s
            {
                continue;
            }
            for slot_ownership in &ownership.worker_slot_ownership {
                let allocation = KvAllocation {
                    blocks: slot_ownership.kv_blocks,
                    allocated_tokens: slot_ownership.allocated_kv_tokens,
                    fragmentation_tokens: slot_ownership.kv_fragmentation_tokens,
                    block_table_bytes: slot_ownership.block_table_bytes,
                };
                let events = slot_events
                    .entry((ownership.owner, slot_ownership.slot))
                    .or_default();
                events.push(CapacityEvent::new(
                    ownership.allocated_at_s,
                    i64::from(slot_ownership.decode_sequences),
                    slot_ownership.resident_tokens as i128,
                    allocation,
                    1,
                ));
                events.push(CapacityEvent::new(
                    ownership.released_at_s,
                    -i64::from(slot_ownership.decode_sequences),
                    -(slot_ownership.resident_tokens as i128),
                    allocation,
                    -1,
                ));
            }
        }
    }

    let configured_decode_slots = decode_worker_slots_per_gpu(traffic)
        .min(u32::MAX as usize)
        .max(1) as u32;
    let decode_sequence_capacity = traffic
        .max_decode_sequences_per_gpu
        .map(|capacity| per_worker_slot_capacity_u32(capacity, configured_decode_slots));
    let resident_token_capacity = traffic
        .max_resident_tokens_per_gpu
        .map(|capacity| per_worker_slot_capacity_u64(capacity, configured_decode_slots));
    let kv_block_capacity = traffic
        .max_kv_blocks_per_gpu
        .map(|capacity| per_worker_slot_capacity_u64(capacity, configured_decode_slots));

    let mut by_gpu: BTreeMap<GpuAddr, Vec<ServingWorkerKvSlotObservation>> = BTreeMap::new();
    for ((gpu, slot), events) in slot_events {
        let peaks = capacity_peaks(events);
        by_gpu
            .entry(gpu)
            .or_default()
            .push(ServingWorkerKvSlotObservation {
                slot,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                decode_sequence_utilization: optional_u32_utilization(
                    peaks.decode_sequences,
                    decode_sequence_capacity,
                ),
                resident_token_utilization: optional_u64_utilization(
                    peaks.resident_tokens,
                    resident_token_capacity,
                ),
                kv_block_utilization: optional_u64_utilization(peaks.kv_blocks, kv_block_capacity),
            });
    }
    for slot_observations in by_gpu.values_mut() {
        slot_observations.sort_by_key(|observation| observation.slot);
    }
    by_gpu
}

fn per_worker_slot_capacity_u32(capacity: u32, configured_slots: u32) -> u32 {
    let configured_slots = u64::from(configured_slots.max(1));
    let capacity = u64::from(capacity);
    capacity.div_ceil(configured_slots).min(u64::from(u32::MAX)) as u32
}

fn per_worker_slot_capacity_u64(capacity: u64, configured_slots: u32) -> u64 {
    capacity.div_ceil(u64::from(configured_slots.max(1)))
}

fn optional_u32_utilization(value: u32, capacity: Option<u32>) -> f64 {
    capacity
        .filter(|capacity| *capacity > 0)
        .map(|capacity| f64::from(value) / f64::from(capacity))
        .unwrap_or(0.0)
}

fn optional_u64_utilization(value: u64, capacity: Option<u64>) -> f64 {
    capacity
        .filter(|capacity| *capacity > 0)
        .map(|capacity| value as f64 / capacity as f64)
        .unwrap_or(0.0)
}

fn service_observations(
    traffic: &ServingTraffic,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    observations: &[ServingRequestObservation],
    workers: &[ServingWorkerObservation],
) -> Vec<ServingServiceObservation> {
    let prefill_slots = traffic.max_prefill_worker_slots_per_gpu.unwrap_or(1).max(1);
    let decode_slots = traffic.max_decode_worker_slots_per_gpu.unwrap_or(1).max(1);
    let kv_transfer_slots = traffic
        .max_kv_transfer_worker_slots_per_gpu
        .unwrap_or(1)
        .max(1);
    let prefill_gpu_count = unique_placement_gpu_count(prefill_score);
    let decode_gpu_count = unique_placement_gpu_count(decode_score);
    let kv_transfer_node_count = prefill_nodes
        .iter()
        .copied()
        .chain(decode_nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .len()
        .min(u32::MAX as usize) as u32;
    let kv_transfer_gpu_count = prefill_gpu_count.saturating_add(decode_gpu_count);
    let inputs = ServiceObservationInputs {
        traffic,
        observations,
        workers,
    };

    vec![
        service_observation(
            "prefill",
            traffic.services.prefill,
            prefill_slots,
            prefill_worker_slots_per_gpu(traffic),
            prefill_nodes.len().min(u32::MAX as usize) as u32,
            prefill_gpu_count,
            inputs,
        ),
        service_observation(
            "decode",
            traffic.services.decode,
            decode_slots,
            decode_worker_slots_per_gpu(traffic),
            decode_nodes.len().min(u32::MAX as usize) as u32,
            decode_gpu_count,
            inputs,
        ),
        service_observation(
            "kv_transfer",
            traffic.services.kv_transfer,
            kv_transfer_slots,
            kv_transfer_worker_slots_per_gpu(traffic).unwrap_or(0),
            kv_transfer_node_count,
            kv_transfer_gpu_count,
            inputs,
        ),
    ]
}

#[derive(Clone, Copy, Debug)]
struct ServiceObservationInputs<'a> {
    traffic: &'a ServingTraffic,
    observations: &'a [ServingRequestObservation],
    workers: &'a [ServingWorkerObservation],
}

fn service_observation(
    phase: &str,
    service: ServingServicePhaseConfig,
    configured_worker_slots: u32,
    effective_worker_slots: usize,
    node_count: u32,
    gpu_count: u32,
    inputs: ServiceObservationInputs<'_>,
) -> ServingServiceObservation {
    let phase_workers = inputs
        .workers
        .iter()
        .filter(|worker| worker.phase == phase)
        .collect::<Vec<_>>();
    let admission = service_admission_observation(phase, inputs.traffic, inputs.observations);
    let worker_slot_utilization =
        mean_weighted_by_requests(&phase_workers, |worker| worker.worker_slot_utilization);
    let worker_queue_s = mean_weighted_by_requests(&phase_workers, |worker| worker.worker_queue_s);
    let resource_queue_s =
        mean_weighted_by_requests(&phase_workers, |worker| worker.resource_queue_s);
    let service_s = mean_weighted_by_requests(&phase_workers, |worker| worker.service_s);
    let backpressure_state = service_backpressure_state(
        service,
        admission.backpressure_rejections,
        admission.timeout_rejections,
        admission.queue_max_s,
    );

    ServingServiceObservation {
        phase: phase.to_string(),
        health: service.health,
        accepts_requests: service.health.accepts_requests(),
        worker_scale: service.worker_scale,
        configured_worker_slots_per_gpu: configured_worker_slots,
        effective_worker_slots_per_gpu: effective_worker_slots.min(u32::MAX as usize) as u32,
        node_count,
        gpu_count,
        request_count: admission.request_count,
        admitted_requests: admission.admitted_requests,
        completed_requests: admission.completed_requests,
        failed_requests: admission.failed_requests,
        rejected_requests: admission.rejected_requests,
        timed_out_requests: admission.timed_out_requests,
        cancelled_requests: admission.cancelled_requests,
        queue_cap_s: admission.queue_cap_s,
        queue_cap_request_count: admission.queue_cap_request_count,
        queue_cap_hit_count: admission.queue_cap_hit_count,
        decode_iteration_queue_cap_s: admission.decode_iteration_queue_cap_s,
        decode_iteration_queue_cap_request_count: admission
            .decode_iteration_queue_cap_request_count,
        decode_iteration_queue_cap_hit_count: admission.decode_iteration_queue_cap_hit_count,
        backpressure_rejections: admission.backpressure_rejections,
        timeout_rejections: admission.timeout_rejections,
        backpressure_state,
        worker_slot_utilization,
        queue_s: admission.queue_s,
        queue_p95_s: admission.queue_p95_s,
        queue_max_s: admission.queue_max_s,
        worker_queue_s,
        resource_queue_s,
        service_s,
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ServiceAdmissionObservation {
    request_count: u32,
    admitted_requests: u32,
    completed_requests: u32,
    failed_requests: u32,
    rejected_requests: u32,
    timed_out_requests: u32,
    cancelled_requests: u32,
    queue_cap_s: Option<f64>,
    queue_cap_request_count: u32,
    queue_cap_hit_count: u32,
    decode_iteration_queue_cap_s: Option<f64>,
    decode_iteration_queue_cap_request_count: u32,
    decode_iteration_queue_cap_hit_count: u32,
    backpressure_rejections: u32,
    timeout_rejections: u32,
    queue_s: f64,
    queue_p95_s: f64,
    queue_max_s: f64,
}

fn service_admission_observation(
    phase: &str,
    traffic: &ServingTraffic,
    observations: &[ServingRequestObservation],
) -> ServiceAdmissionObservation {
    let mut accumulator = ServiceAdmissionAccumulator::default();
    for observation in observations {
        if !service_phase_attempted(phase, observation) {
            continue;
        }
        accumulator.record(phase, traffic, observation);
    }
    accumulator.into_observation()
}

#[derive(Clone, Debug, Default)]
struct ServiceAdmissionAccumulator {
    request_count: u32,
    admitted_requests: u32,
    completed_requests: u32,
    rejected_requests: u32,
    timed_out_requests: u32,
    cancelled_requests: u32,
    queue_cap_s: Option<f64>,
    queue_cap_request_count: u32,
    queue_cap_hit_count: u32,
    decode_iteration_queue_cap_s: Option<f64>,
    decode_iteration_queue_cap_request_count: u32,
    decode_iteration_queue_cap_hit_count: u32,
    backpressure_rejections: u32,
    timeout_rejections: u32,
    queue_samples_s: Vec<f64>,
}

impl ServiceAdmissionAccumulator {
    fn record(
        &mut self,
        phase: &str,
        traffic: &ServingTraffic,
        observation: &ServingRequestObservation,
    ) {
        self.request_count = self.request_count.saturating_add(1);
        if service_phase_admitted(phase, observation) {
            self.admitted_requests = self.admitted_requests.saturating_add(1);
        }
        if service_phase_completed(phase, observation) {
            self.completed_requests = self.completed_requests.saturating_add(1);
        }
        if observation.status == ServingRequestStatus::Cancelled {
            self.cancelled_requests = self.cancelled_requests.saturating_add(1);
        }

        if let Some(queue_s) = service_phase_queue_s(phase, observation) {
            self.queue_samples_s.push(queue_s);
        }
        if let Some(cap_s) = service_phase_queue_cap_s(phase, traffic, observation.request_idx) {
            self.queue_cap_request_count = self.queue_cap_request_count.saturating_add(1);
            self.queue_cap_s = min_optional_f64(self.queue_cap_s, cap_s);
        }
        if phase == "decode"
            && let Some(cap_s) =
                traffic.effective_max_decode_iteration_queue_delay_s(observation.request_idx)
        {
            self.decode_iteration_queue_cap_request_count = self
                .decode_iteration_queue_cap_request_count
                .saturating_add(1);
            self.decode_iteration_queue_cap_s =
                min_optional_f64(self.decode_iteration_queue_cap_s, cap_s);
        }

        let Some(rejection) = observation.rejection.as_ref() else {
            return;
        };
        if rejection.phase != phase {
            return;
        }
        if observation.status == ServingRequestStatus::RejectedAdmission {
            self.rejected_requests = self.rejected_requests.saturating_add(1);
        }
        if observation.status == ServingRequestStatus::TimedOut {
            self.timed_out_requests = self.timed_out_requests.saturating_add(1);
            self.timeout_rejections = self.timeout_rejections.saturating_add(1);
        }
        if rejection.category == "queueing" && rejection.code.ends_with("queue_delay_exceeded") {
            self.backpressure_rejections = self.backpressure_rejections.saturating_add(1);
            if rejection.code == "decode_iteration_queue_delay_exceeded" {
                self.decode_iteration_queue_cap_hit_count =
                    self.decode_iteration_queue_cap_hit_count.saturating_add(1);
            } else {
                self.queue_cap_hit_count = self.queue_cap_hit_count.saturating_add(1);
            }
        }
    }

    fn into_observation(self) -> ServiceAdmissionObservation {
        let failed_requests = self.request_count.saturating_sub(self.completed_requests);
        ServiceAdmissionObservation {
            request_count: self.request_count,
            admitted_requests: self.admitted_requests,
            completed_requests: self.completed_requests,
            failed_requests,
            rejected_requests: self.rejected_requests,
            timed_out_requests: self.timed_out_requests,
            cancelled_requests: self.cancelled_requests,
            queue_cap_s: self.queue_cap_s,
            queue_cap_request_count: self.queue_cap_request_count,
            queue_cap_hit_count: self.queue_cap_hit_count,
            decode_iteration_queue_cap_s: self.decode_iteration_queue_cap_s,
            decode_iteration_queue_cap_request_count: self.decode_iteration_queue_cap_request_count,
            decode_iteration_queue_cap_hit_count: self.decode_iteration_queue_cap_hit_count,
            backpressure_rejections: self.backpressure_rejections,
            timeout_rejections: self.timeout_rejections,
            queue_s: mean(&self.queue_samples_s),
            queue_p95_s: percentile(self.queue_samples_s.clone(), 0.95),
            queue_max_s: max_value(&self.queue_samples_s),
        }
    }
}

fn service_phase_attempted(phase: &str, observation: &ServingRequestObservation) -> bool {
    match phase {
        "prefill" => observation.arrival_s.is_finite(),
        "kv_transfer" => {
            observation.kv_transfer_bytes > 0
                || observation.kv_start_s.is_finite()
                || observation
                    .rejection
                    .as_ref()
                    .is_some_and(|rejection| rejection.phase == "kv_transfer")
        }
        "decode" => {
            observation.first_decode_start_s.is_finite()
                || !observation.decode_token_finish_s.is_empty()
                || observation.status.is_admitted() && observation.kv_finish_s.is_finite()
                || observation
                    .rejection
                    .as_ref()
                    .is_some_and(|rejection| rejection.phase == "decode")
        }
        _ => false,
    }
}

fn service_phase_admitted(phase: &str, observation: &ServingRequestObservation) -> bool {
    !observation.rejection.as_ref().is_some_and(|rejection| {
        rejection.phase == phase && observation.status == ServingRequestStatus::RejectedAdmission
    })
}

fn service_phase_completed(phase: &str, observation: &ServingRequestObservation) -> bool {
    match phase {
        "prefill" => observation.prefill_finish_s.is_finite(),
        "kv_transfer" => observation.kv_finish_s.is_finite(),
        "decode" => observation.status == ServingRequestStatus::Completed,
        _ => false,
    }
}

fn service_phase_queue_s(phase: &str, observation: &ServingRequestObservation) -> Option<f64> {
    let queue_s = match phase {
        "prefill" => observation.prefill_worker_queue_s + observation.prefill_resource_queue_s,
        "kv_transfer" => observation.kv_queue_s,
        "decode" => {
            let decode_queue_s =
                observation.decode_worker_queue_s + observation.decode_resource_queue_s;
            observation.decode_queue_s.max(decode_queue_s)
        }
        _ => return None,
    };
    queue_s.is_finite().then_some(queue_s.max(0.0))
}

fn service_phase_queue_cap_s(
    phase: &str,
    traffic: &ServingTraffic,
    request_idx: u32,
) -> Option<f64> {
    match phase {
        "prefill" => traffic.effective_max_queue_delay_s(request_idx),
        "kv_transfer" => traffic.effective_max_kv_queue_delay_s(request_idx),
        "decode" => traffic.effective_max_decode_queue_delay_s(request_idx),
        _ => None,
    }
}

fn min_optional_f64(current: Option<f64>, candidate: f64) -> Option<f64> {
    Some(match current {
        Some(current) => current.min(candidate),
        None => candidate,
    })
}

fn service_backpressure_state(
    service: ServingServicePhaseConfig,
    backpressure_rejections: u32,
    timeout_rejections: u32,
    queue_max_s: f64,
) -> String {
    if !service.health.accepts_requests() {
        "closed".to_string()
    } else if timeout_rejections > 0 {
        "timing_out".to_string()
    } else if backpressure_rejections > 0 {
        "rejecting".to_string()
    } else if queue_max_s.is_finite() && queue_max_s > 0.0 {
        "queued".to_string()
    } else {
        "open".to_string()
    }
}

fn unique_placement_gpu_count(score: &ScoredParallelismConfig) -> u32 {
    score
        .placement
        .rank_to_gpu
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        .min(u32::MAX as usize) as u32
}

fn mean_weighted_by_requests(
    workers: &[&ServingWorkerObservation],
    metric: impl Fn(&ServingWorkerObservation) -> f64,
) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0_u64;
    for worker in workers {
        let value = metric(worker);
        if !value.is_finite() {
            continue;
        }
        let weight = u64::from(worker.request_count.max(1));
        weighted_sum += value * weight as f64;
        total_weight = total_weight.saturating_add(weight);
    }
    if total_weight == 0 {
        0.0
    } else {
        weighted_sum / total_weight as f64
    }
}

fn worker_observations(
    observations: &[ServingRequestObservation],
    gpu_capacity: &[ServingGpuCapacityObservation],
    traffic: &ServingTraffic,
) -> Vec<ServingWorkerObservation> {
    let mut workers: BTreeMap<(String, GpuAddr), WorkerObservationAccumulator> = BTreeMap::new();
    let prefill_slots = prefill_worker_slots_per_gpu(traffic).min(u32::MAX as usize) as u32;
    let decode_slots = decode_worker_slots_per_gpu(traffic).min(u32::MAX as usize) as u32;
    let kv_transfer_slots =
        kv_transfer_worker_slots_per_gpu(traffic).map(|slots| slots.min(u32::MAX as usize) as u32);
    let kv_worker_slot_capacity = kv_worker_slot_capacity_observations(observations, traffic);

    for observation in observations {
        for gpu in &observation.prefill_route_gpus {
            workers
                .entry(("prefill".to_string(), *gpu))
                .or_default()
                .record_prefill(observation);
        }
        for gpu in &observation.decode_route_gpus {
            workers
                .entry(("decode".to_string(), *gpu))
                .or_default()
                .record_decode(observation);
        }
        if kv_transfer_slots.is_some() && observation.kv_transfer_bytes > 0 {
            for gpu in kv_transfer_observation_gpus(observation) {
                workers
                    .entry(("kv_transfer".to_string(), gpu))
                    .or_default()
                    .record_kv_transfer(observation);
            }
        }
    }

    for capacity in gpu_capacity {
        let gpu = GpuAddr {
            node_id: capacity.node_id,
            local_gpu_id: capacity.local_gpu_id,
        };
        if capacity.peak_prefill_tokens > 0 {
            workers
                .entry(("prefill".to_string(), gpu))
                .or_default()
                .apply_prefill_capacity(capacity);
        }
        if capacity.peak_decode_sequences > 0
            || capacity.peak_resident_tokens > 0
            || capacity.peak_kv_blocks > 0
        {
            workers
                .entry(("decode".to_string(), gpu))
                .or_default()
                .apply_decode_capacity(capacity);
        }
    }
    for (gpu, slot_observations) in kv_worker_slot_capacity {
        workers
            .entry(("decode".to_string(), gpu))
            .or_default()
            .apply_decode_worker_slot_capacity(slot_observations);
    }

    workers
        .into_iter()
        .map(|((phase, gpu), accumulator)| {
            let configured_worker_slots =
                phase_worker_slots(&phase, prefill_slots, decode_slots, kv_transfer_slots);
            let deduplicate_intervals = phase_deduplicates_worker_intervals(&phase, traffic);
            accumulator.into_observation(phase, gpu, configured_worker_slots, deduplicate_intervals)
        })
        .collect()
}

#[derive(Default)]
struct MetricBreakdownAccumulator {
    request_count: u32,
    completed_requests: u32,
    failed_requests: u32,
    rejected_requests: u32,
    timed_out_requests: u32,
    cancelled_requests: u32,
    output_tokens: u64,
    metric_source_counts: BTreeMap<String, u32>,
    deadline_constrained_requests: u32,
    deadline_missed_requests: u32,
    ttft_slo_constrained_requests: u32,
    ttft_slo_missed_requests: u32,
    tpot_slo_constrained_requests: u32,
    tpot_slo_missed_requests: u32,
    itl_slo_constrained_requests: u32,
    itl_slo_missed_requests: u32,
    e2el_slo_constrained_requests: u32,
    e2el_slo_missed_requests: u32,
    ttft: Vec<f64>,
    tpot: Vec<f64>,
    itl: Vec<f64>,
    e2el: Vec<f64>,
}

fn metric_breakdowns(
    observations: &[ServingRequestObservation],
    measurement_start_s: f64,
    measurement_end_s: f64,
) -> Vec<ServingMetricBreakdown> {
    let measurement_duration_s = (measurement_end_s - measurement_start_s).max(0.0);
    let mut groups: BTreeMap<(String, String), MetricBreakdownAccumulator> = BTreeMap::new();

    for observation in observations {
        if observation.arrival_s + 1e-12 < measurement_start_s
            || observation.arrival_s > measurement_end_s + 1e-12
        {
            continue;
        }
        if let Some(tenant) = observation.tenant.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "tenant",
                tenant,
                observation,
                measurement_duration_s,
            );
        }
        if let Some(model_id) = observation.model_id.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "model_id",
                model_id,
                observation,
                measurement_duration_s,
            );
        }
        if let Some(traffic_class) = observation.traffic_class.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "traffic_class",
                traffic_class,
                observation,
                measurement_duration_s,
            );
        }
        if let Some(shape_profile) = observation.shape_profile.as_deref() {
            accumulate_metric_breakdown(
                &mut groups,
                "shape_profile",
                shape_profile,
                observation,
                measurement_duration_s,
            );
        }
        accumulate_metric_breakdown(
            &mut groups,
            "priority",
            &priority_key(observation.priority),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "prefill_node",
            &node_key(observation.prefill_node),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "decode_node",
            &node_key(observation.decode_node),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "prefill_route",
            &node_set_key(&observation.prefill_route_nodes),
            observation,
            measurement_duration_s,
        );
        accumulate_metric_breakdown(
            &mut groups,
            "decode_route",
            &node_set_key(&observation.decode_route_nodes),
            observation,
            measurement_duration_s,
        );
    }

    groups
        .into_iter()
        .map(|((group, key), accumulator)| {
            accumulator.into_breakdown(group, key, measurement_duration_s)
        })
        .collect()
}

fn node_key(node_id: NodeId) -> String {
    format!("node-{node_id}")
}

fn priority_key(priority: i32) -> String {
    format!("priority-{priority}")
}

fn node_set_key(nodes: &[NodeId]) -> String {
    let mut nodes = nodes.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    format!(
        "nodes[{}]",
        nodes
            .iter()
            .map(|node| node.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn accumulate_metric_breakdown(
    groups: &mut BTreeMap<(String, String), MetricBreakdownAccumulator>,
    group: &str,
    key: &str,
    observation: &ServingRequestObservation,
    measurement_duration_s: f64,
) {
    let accumulator = groups
        .entry((group.to_string(), key.to_string()))
        .or_default();
    accumulator.record(observation, measurement_duration_s);
}

impl MetricBreakdownAccumulator {
    fn record(&mut self, observation: &ServingRequestObservation, _measurement_duration_s: f64) {
        self.request_count = self.request_count.saturating_add(1);
        if observation.deadline_s.is_some() {
            self.deadline_constrained_requests =
                self.deadline_constrained_requests.saturating_add(1);
        }
        if observation.deadline_missed {
            self.deadline_missed_requests = self.deadline_missed_requests.saturating_add(1);
        }
        self.record_slo_misses(observation);
        match observation.status {
            ServingRequestStatus::RejectedAdmission => {
                self.rejected_requests = self.rejected_requests.saturating_add(1);
            }
            ServingRequestStatus::TimedOut => {
                self.timed_out_requests = self.timed_out_requests.saturating_add(1);
            }
            ServingRequestStatus::Cancelled => {
                self.cancelled_requests = self.cancelled_requests.saturating_add(1);
            }
            ServingRequestStatus::Pending | ServingRequestStatus::Completed => {}
        }
        if !observation.status.is_completed() {
            self.failed_requests = self.failed_requests.saturating_add(1);
            return;
        }

        self.completed_requests = self.completed_requests.saturating_add(1);
        let metric_source_count = self
            .metric_source_counts
            .entry(observation.metric_source.clone())
            .or_default();
        *metric_source_count = metric_source_count.saturating_add(1);
        self.output_tokens = self
            .output_tokens
            .saturating_add(observation_output_tokens(observation));
        push_finite(&mut self.ttft, observation.ttft_s);
        push_finite(&mut self.tpot, observation.tpot_s);
        push_finite(&mut self.itl, observation.itl_s);
        push_finite(&mut self.e2el, observation.e2el_s);
    }

    fn record_slo_misses(&mut self, observation: &ServingRequestObservation) {
        if observation.slo.ttft_s.is_some() {
            self.ttft_slo_constrained_requests =
                self.ttft_slo_constrained_requests.saturating_add(1);
            if observation.ttft_slo_missed {
                self.ttft_slo_missed_requests = self.ttft_slo_missed_requests.saturating_add(1);
            }
        }
        if observation.slo.tpot_s.is_some() {
            self.tpot_slo_constrained_requests =
                self.tpot_slo_constrained_requests.saturating_add(1);
            if observation.tpot_slo_missed {
                self.tpot_slo_missed_requests = self.tpot_slo_missed_requests.saturating_add(1);
            }
        }
        if observation.slo.itl_s.is_some() {
            self.itl_slo_constrained_requests = self.itl_slo_constrained_requests.saturating_add(1);
            if observation.itl_slo_missed {
                self.itl_slo_missed_requests = self.itl_slo_missed_requests.saturating_add(1);
            }
        }
        if observation.slo.e2el_s.is_some() {
            self.e2el_slo_constrained_requests =
                self.e2el_slo_constrained_requests.saturating_add(1);
            if observation.e2el_slo_missed {
                self.e2el_slo_missed_requests = self.e2el_slo_missed_requests.saturating_add(1);
            }
        }
    }

    fn into_breakdown(
        self,
        group: String,
        key: String,
        measurement_duration_s: f64,
    ) -> ServingMetricBreakdown {
        let deadline_miss_rate = if self.deadline_constrained_requests > 0 {
            f64::from(self.deadline_missed_requests) / f64::from(self.deadline_constrained_requests)
        } else {
            0.0
        };
        let throughput_tokens_per_s = if measurement_duration_s > 0.0 {
            self.output_tokens as f64 / measurement_duration_s
        } else {
            0.0
        };
        let ttft_s = mean(&self.ttft);
        let ttft_p90_s = percentile(self.ttft.clone(), 0.90);
        let ttft_p95_s = percentile(self.ttft.clone(), 0.95);
        let ttft_max_s = max_value(&self.ttft);
        let tpot_s = mean(&self.tpot);
        let tpot_p90_s = percentile(self.tpot.clone(), 0.90);
        let tpot_p95_s = percentile(self.tpot.clone(), 0.95);
        let tpot_max_s = max_value(&self.tpot);
        let itl_s = mean(&self.itl);
        let itl_p90_s = percentile(self.itl.clone(), 0.90);
        let itl_p95_s = percentile(self.itl.clone(), 0.95);
        let itl_max_s = max_value(&self.itl);
        let e2el_s = mean(&self.e2el);
        let e2el_p90_s = percentile(self.e2el.clone(), 0.90);
        let e2el_p95_s = percentile(self.e2el.clone(), 0.95);
        let e2el_max_s = max_value(&self.e2el);
        let metric_source_counts = self
            .metric_source_counts
            .into_iter()
            .map(
                |(metric_source, request_count)| ServingMeasurementMetricSourceCount {
                    metric_source,
                    request_count,
                },
            )
            .collect::<Vec<_>>();
        let lifecycle_event_metric_request_count = metric_source_counts
            .iter()
            .find(|count| count.metric_source == "request_lifecycle_events")
            .map(|count| count.request_count)
            .unwrap_or(0);
        let fallback_metric_request_count = metric_source_counts
            .iter()
            .filter(|count| count.metric_source != "request_lifecycle_events")
            .map(|count| count.request_count)
            .sum();

        ServingMetricBreakdown {
            group,
            key,
            request_count: self.request_count,
            completed_requests: self.completed_requests,
            failed_requests: self.failed_requests,
            rejected_requests: self.rejected_requests,
            timed_out_requests: self.timed_out_requests,
            cancelled_requests: self.cancelled_requests,
            output_tokens: self.output_tokens,
            lifecycle_event_metric_request_count,
            fallback_metric_request_count,
            metric_source_counts,
            deadline_constrained_requests: self.deadline_constrained_requests,
            deadline_missed_requests: self.deadline_missed_requests,
            deadline_miss_rate,
            ttft_slo_constrained_requests: self.ttft_slo_constrained_requests,
            ttft_slo_missed_requests: self.ttft_slo_missed_requests,
            ttft_slo_miss_rate: ratio_or_infinity(
                self.ttft_slo_missed_requests,
                self.ttft_slo_constrained_requests,
            ),
            tpot_slo_constrained_requests: self.tpot_slo_constrained_requests,
            tpot_slo_missed_requests: self.tpot_slo_missed_requests,
            tpot_slo_miss_rate: ratio_or_infinity(
                self.tpot_slo_missed_requests,
                self.tpot_slo_constrained_requests,
            ),
            itl_slo_constrained_requests: self.itl_slo_constrained_requests,
            itl_slo_missed_requests: self.itl_slo_missed_requests,
            itl_slo_miss_rate: ratio_or_infinity(
                self.itl_slo_missed_requests,
                self.itl_slo_constrained_requests,
            ),
            e2el_slo_constrained_requests: self.e2el_slo_constrained_requests,
            e2el_slo_missed_requests: self.e2el_slo_missed_requests,
            e2el_slo_miss_rate: ratio_or_infinity(
                self.e2el_slo_missed_requests,
                self.e2el_slo_constrained_requests,
            ),
            ttft_s,
            ttft_p90_s,
            ttft_p95_s,
            ttft_max_s,
            tpot_s,
            tpot_p90_s,
            tpot_p95_s,
            tpot_max_s,
            itl_s,
            itl_p90_s,
            itl_p95_s,
            itl_max_s,
            throughput_tokens_per_s,
            e2el_s,
            e2el_p90_s,
            e2el_p95_s,
            e2el_max_s,
        }
    }
}

fn push_finite(values: &mut Vec<f64>, value: f64) {
    if value.is_finite() {
        values.push(value);
    }
}

fn slo_missed(slo_s: Option<f64>, value_s: f64, completed: bool) -> bool {
    slo_s.is_some_and(|slo_s| !completed || !value_s.is_finite() || value_s > slo_s + 1e-12)
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct SloMissCounts {
    constrained: u32,
    missed: u32,
    miss_rate: f64,
}

fn observation_slo_miss_counts(
    observations: &[ServingRequestObservation],
    slo: impl Fn(&ServingRequestObservation) -> Option<f64>,
    value: impl Fn(&ServingRequestObservation) -> f64,
) -> SloMissCounts {
    let mut constrained = 0_u32;
    let mut missed = 0_u32;
    for observation in observations {
        if let Some(slo_s) = slo(observation) {
            constrained = constrained.saturating_add(1);
            let value_s = value(observation);
            if !observation.status.is_completed() || !value_s.is_finite() || value_s > slo_s + 1e-12
            {
                missed = missed.saturating_add(1);
            }
        }
    }
    SloMissCounts {
        constrained,
        missed,
        miss_rate: ratio_or_infinity(missed, constrained),
    }
}

fn ratio_or_infinity(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        f64::INFINITY
    } else {
        f64::from(numerator) / f64::from(denominator)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct KvAllocation {
    blocks: u64,
    allocated_tokens: u64,
    fragmentation_tokens: u64,
    block_table_bytes: u64,
}

const DEFAULT_KV_BLOCK_TOKENS: u32 = 16;
const KV_BLOCK_TABLE_ENTRY_BYTES: u64 = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
struct KvOwnerAllocation {
    allocation_id: String,
    owner: GpuAddr,
    block_start: u64,
    block_end: u64,
    resident_tokens: u64,
    kv_blocks: u64,
    allocated_kv_tokens: u64,
    kv_fragmentation_tokens: u64,
    block_table_entries: u64,
    block_table_bytes: u64,
}

fn kv_block_tokens(traffic: &ServingTraffic) -> u32 {
    traffic
        .kv_block_tokens
        .unwrap_or(DEFAULT_KV_BLOCK_TOKENS)
        .max(1)
}

fn sequence_kv_allocation(
    batch_size: u32,
    max_sequence_tokens: u32,
    block_tokens: u32,
) -> KvAllocation {
    let block_tokens = u64::from(block_tokens.max(1));
    let blocks_per_sequence = u64::from(max_sequence_tokens.max(1)).div_ceil(block_tokens);
    let blocks = u64::from(batch_size.max(1)).saturating_mul(blocks_per_sequence);
    let resident_tokens =
        u64::from(batch_size.max(1)).saturating_mul(u64::from(max_sequence_tokens.max(1)));
    let allocated_tokens = blocks.saturating_mul(block_tokens);
    KvAllocation {
        blocks,
        allocated_tokens,
        fragmentation_tokens: allocated_tokens.saturating_sub(resident_tokens),
        block_table_bytes: blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
    }
}

fn kv_owner_allocations(state: &DecodeRequestState) -> Vec<KvOwnerAllocation> {
    let owners = decode_owner_gpus(state);
    if owners.is_empty() || state.kv_cache_blocks == 0 {
        return Vec::new();
    }

    let block_counts = partition_units(state.kv_cache_blocks, owners.len());
    let resident_tokens =
        u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1));
    let token_counts = partition_tokens_by_block_capacity(
        resident_tokens,
        &block_counts,
        u64::from(state.kv_block_tokens.max(1)),
    );
    let mut block_start = 0_u64;
    owners
        .into_iter()
        .zip(block_counts)
        .zip(token_counts)
        .filter_map(|((owner, kv_blocks), resident_tokens)| {
            if kv_blocks == 0 {
                return None;
            }
            let block_end = block_start.saturating_add(kv_blocks);
            let allocated_kv_tokens =
                kv_blocks.saturating_mul(u64::from(state.kv_block_tokens.max(1)));
            let allocation = KvOwnerAllocation {
                allocation_id: format!(
                    "request-{}:node-{}:gpu-{}:blocks-{}-{}",
                    state.request_idx, owner.node_id, owner.local_gpu_id, block_start, block_end
                ),
                owner,
                block_start,
                block_end,
                resident_tokens,
                kv_blocks,
                allocated_kv_tokens,
                kv_fragmentation_tokens: allocated_kv_tokens.saturating_sub(resident_tokens),
                block_table_entries: kv_blocks,
                block_table_bytes: kv_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
            };
            block_start = block_end;
            Some(allocation)
        })
        .collect()
}

fn partition_units(total: u64, parts: usize) -> Vec<u64> {
    if parts == 0 {
        return Vec::new();
    }
    let base = total / parts as u64;
    let remainder = total % parts as u64;
    (0..parts)
        .map(|idx| base + u64::from((idx as u64) < remainder))
        .collect()
}

fn partition_tokens_by_block_capacity(
    total_tokens: u64,
    block_counts: &[u64],
    block_tokens: u64,
) -> Vec<u64> {
    let capacities = block_counts
        .iter()
        .map(|blocks| blocks.saturating_mul(block_tokens.max(1)))
        .collect::<Vec<_>>();
    let mut remaining_tokens = total_tokens;
    let mut remaining_capacity = capacities.iter().copied().sum::<u64>();
    let mut token_counts = Vec::with_capacity(capacities.len());

    for capacity in capacities {
        if remaining_tokens == 0 || capacity == 0 || remaining_capacity == 0 {
            token_counts.push(0);
            remaining_capacity = remaining_capacity.saturating_sub(capacity);
            continue;
        }
        let tokens = if capacity >= remaining_capacity {
            remaining_tokens.min(capacity)
        } else {
            (((remaining_tokens as u128) * (capacity as u128)) / remaining_capacity as u128)
                .min(capacity as u128)
                .min(remaining_tokens as u128) as u64
        };
        token_counts.push(tokens);
        remaining_tokens = remaining_tokens.saturating_sub(tokens);
        remaining_capacity = remaining_capacity.saturating_sub(capacity);
    }

    token_counts
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct CapacityEvent {
    time_s: f64,
    sequence_delta: i64,
    resident_token_delta: i128,
    kv_block_delta: i128,
    allocated_token_delta: i128,
    fragmentation_token_delta: i128,
    block_table_byte_delta: i128,
}

impl CapacityEvent {
    fn new(
        time_s: f64,
        sequence_delta: i64,
        resident_token_delta: i128,
        allocation: KvAllocation,
        sign: i128,
    ) -> Self {
        Self {
            time_s,
            sequence_delta,
            resident_token_delta,
            kv_block_delta: sign * i128::from(allocation.blocks),
            allocated_token_delta: sign * i128::from(allocation.allocated_tokens),
            fragmentation_token_delta: sign * i128::from(allocation.fragmentation_tokens),
            block_table_byte_delta: sign * i128::from(allocation.block_table_bytes),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct CapacityPeaks {
    decode_sequences: u32,
    resident_tokens: u64,
    kv_blocks: u64,
    allocated_kv_tokens: u64,
    kv_fragmentation_tokens: u64,
    kv_block_table_bytes: u64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct PrefillCapacityEvent {
    time_s: f64,
    token_delta: i128,
}

fn capacity_profile(states: &[DecodeRequestState], traffic: &ServingTraffic) -> CapacityProfile {
    let mut prefill_events = Vec::with_capacity(states.len() * 2);
    let mut prefill_node_events: BTreeMap<NodeId, Vec<PrefillCapacityEvent>> = BTreeMap::new();
    let mut prefill_gpu_events: BTreeMap<GpuAddr, Vec<PrefillCapacityEvent>> = BTreeMap::new();
    let mut prefill_class_events: BTreeMap<String, Vec<PrefillCapacityEvent>> = BTreeMap::new();
    let mut aggregate_events = Vec::with_capacity(states.len() * 2);
    let mut node_events: BTreeMap<NodeId, Vec<CapacityEvent>> = BTreeMap::new();
    let mut gpu_events: BTreeMap<GpuAddr, Vec<CapacityEvent>> = BTreeMap::new();
    let mut class_events: BTreeMap<String, Vec<CapacityEvent>> = BTreeMap::new();
    for state in states {
        for span in prefill_capacity_spans(state) {
            prefill_events.push(PrefillCapacityEvent {
                time_s: span.start_s,
                token_delta: span.tokens as i128,
            });
            prefill_events.push(PrefillCapacityEvent {
                time_s: span.finish_s,
                token_delta: -(span.tokens as i128),
            });
            for node_id in prefill_owner_nodes(state) {
                let events = prefill_node_events.entry(node_id).or_default();
                events.push(PrefillCapacityEvent {
                    time_s: span.start_s,
                    token_delta: span.tokens as i128,
                });
                events.push(PrefillCapacityEvent {
                    time_s: span.finish_s,
                    token_delta: -(span.tokens as i128),
                });
            }
            for gpu in prefill_owner_gpus(state) {
                let events = prefill_gpu_events.entry(gpu).or_default();
                events.push(PrefillCapacityEvent {
                    time_s: span.start_s,
                    token_delta: span.tokens as i128,
                });
                events.push(PrefillCapacityEvent {
                    time_s: span.finish_s,
                    token_delta: -(span.tokens as i128),
                });
            }
            if let Some(traffic_class) = state.traffic_class.as_deref() {
                let events = prefill_class_events
                    .entry(traffic_class.to_string())
                    .or_default();
                events.push(PrefillCapacityEvent {
                    time_s: span.start_s,
                    token_delta: span.tokens as i128,
                });
                events.push(PrefillCapacityEvent {
                    time_s: span.finish_s,
                    token_delta: -(span.tokens as i128),
                });
            }
        }

        let Some((start_s, finish_s)) = kv_residency_window_s(state) else {
            continue;
        };

        let sequences = i64::from(state.batch_size.max(1));
        let resident_tokens =
            u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1));
        let aggregate_allocation = KvAllocation {
            blocks: state.kv_cache_blocks,
            allocated_tokens: state.kv_allocated_tokens,
            fragmentation_tokens: state.kv_fragmentation_tokens,
            block_table_bytes: state
                .kv_cache_blocks
                .saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
        };
        aggregate_events.push(CapacityEvent::new(
            start_s,
            sequences,
            resident_tokens as i128,
            aggregate_allocation,
            1,
        ));
        aggregate_events.push(CapacityEvent::new(
            finish_s,
            -sequences,
            -(resident_tokens as i128),
            aggregate_allocation,
            -1,
        ));
        if let Some(traffic_class) = state.traffic_class.as_deref() {
            let events = class_events.entry(traffic_class.to_string()).or_default();
            events.push(CapacityEvent::new(
                start_s,
                sequences,
                resident_tokens as i128,
                aggregate_allocation,
                1,
            ));
            events.push(CapacityEvent::new(
                finish_s,
                -sequences,
                -(resident_tokens as i128),
                aggregate_allocation,
                -1,
            ));
        }
        let owner_allocations = kv_owner_allocations(state);
        let mut node_allocations: BTreeMap<NodeId, (u64, KvAllocation)> = BTreeMap::new();
        for allocation in &owner_allocations {
            let entry = node_allocations
                .entry(allocation.owner.node_id)
                .or_insert((0, KvAllocation::default()));
            entry.0 = entry.0.saturating_add(allocation.resident_tokens);
            entry.1.blocks = entry.1.blocks.saturating_add(allocation.kv_blocks);
            entry.1.allocated_tokens = entry
                .1
                .allocated_tokens
                .saturating_add(allocation.allocated_kv_tokens);
            entry.1.fragmentation_tokens = entry
                .1
                .fragmentation_tokens
                .saturating_add(allocation.kv_fragmentation_tokens);
            entry.1.block_table_bytes = entry
                .1
                .block_table_bytes
                .saturating_add(allocation.block_table_bytes);
        }
        for (node_id, (node_tokens, node_allocation)) in node_allocations {
            let events = node_events.entry(node_id).or_default();
            events.push(CapacityEvent::new(
                start_s,
                sequences,
                node_tokens as i128,
                node_allocation,
                1,
            ));
            events.push(CapacityEvent::new(
                finish_s,
                -sequences,
                -(node_tokens as i128),
                node_allocation,
                -1,
            ));
        }
        for allocation in owner_allocations {
            let events = gpu_events.entry(allocation.owner).or_default();
            let gpu_allocation = KvAllocation {
                blocks: allocation.kv_blocks,
                allocated_tokens: allocation.allocated_kv_tokens,
                fragmentation_tokens: allocation.kv_fragmentation_tokens,
                block_table_bytes: allocation.block_table_bytes,
            };
            events.push(CapacityEvent::new(
                start_s,
                sequences,
                allocation.resident_tokens as i128,
                gpu_allocation,
                1,
            ));
            events.push(CapacityEvent::new(
                finish_s,
                -sequences,
                -(allocation.resident_tokens as i128),
                gpu_allocation,
                -1,
            ));
        }
    }

    let peak_prefill_tokens = prefill_token_peak(prefill_events);
    let aggregate_peaks = capacity_peaks(aggregate_events);
    let mut peak_decode_sequences_per_node = 0;
    let mut peak_resident_tokens_per_node = 0;
    let mut peak_kv_blocks_per_node = 0;
    let mut peak_allocated_kv_tokens_per_node = 0;
    let mut peak_kv_fragmentation_tokens_per_node = 0;
    let mut peak_kv_block_table_bytes_per_node = 0;
    let mut peak_prefill_tokens_per_node = 0;
    let mut peak_decode_sequences_per_gpu = 0;
    let mut peak_resident_tokens_per_gpu = 0;
    let mut peak_kv_blocks_per_gpu = 0;
    let mut peak_allocated_kv_tokens_per_gpu = 0;
    let mut peak_kv_fragmentation_tokens_per_gpu = 0;
    let mut peak_kv_block_table_bytes_per_gpu = 0;
    let mut peak_prefill_tokens_per_gpu = 0;
    let node_ids = node_events
        .keys()
        .copied()
        .chain(prefill_node_events.keys().copied())
        .collect::<BTreeSet<_>>();
    let nodes = node_ids
        .into_iter()
        .map(|node_id| {
            let prefill_peak =
                prefill_token_peak(prefill_node_events.remove(&node_id).unwrap_or_default());
            peak_prefill_tokens_per_node = peak_prefill_tokens_per_node.max(prefill_peak);
            let peaks = capacity_peaks(node_events.remove(&node_id).unwrap_or_default());
            peak_decode_sequences_per_node =
                peak_decode_sequences_per_node.max(peaks.decode_sequences);
            peak_resident_tokens_per_node =
                peak_resident_tokens_per_node.max(peaks.resident_tokens);
            peak_kv_blocks_per_node = peak_kv_blocks_per_node.max(peaks.kv_blocks);
            peak_allocated_kv_tokens_per_node =
                peak_allocated_kv_tokens_per_node.max(peaks.allocated_kv_tokens);
            peak_kv_fragmentation_tokens_per_node =
                peak_kv_fragmentation_tokens_per_node.max(peaks.kv_fragmentation_tokens);
            peak_kv_block_table_bytes_per_node =
                peak_kv_block_table_bytes_per_node.max(peaks.kv_block_table_bytes);
            ServingNodeCapacityObservation {
                node_id,
                peak_prefill_tokens: prefill_peak,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                decode_sequence_utilization: traffic
                    .max_decode_sequences_per_node
                    .map(|capacity| f64::from(peaks.decode_sequences) / f64::from(capacity))
                    .unwrap_or(0.0),
                resident_token_utilization: traffic
                    .max_resident_tokens_per_node
                    .map(|capacity| peaks.resident_tokens as f64 / capacity as f64)
                    .unwrap_or(0.0),
                kv_block_utilization: traffic
                    .max_kv_blocks_per_node
                    .map(|capacity| peaks.kv_blocks as f64 / capacity as f64)
                    .unwrap_or(0.0),
            }
        })
        .collect();
    let gpu_ids = gpu_events
        .keys()
        .copied()
        .chain(prefill_gpu_events.keys().copied())
        .collect::<BTreeSet<_>>();
    let gpus = gpu_ids
        .into_iter()
        .map(|gpu| {
            let prefill_peak =
                prefill_token_peak(prefill_gpu_events.remove(&gpu).unwrap_or_default());
            peak_prefill_tokens_per_gpu = peak_prefill_tokens_per_gpu.max(prefill_peak);
            let peaks = capacity_peaks(gpu_events.remove(&gpu).unwrap_or_default());
            peak_decode_sequences_per_gpu =
                peak_decode_sequences_per_gpu.max(peaks.decode_sequences);
            peak_resident_tokens_per_gpu = peak_resident_tokens_per_gpu.max(peaks.resident_tokens);
            peak_kv_blocks_per_gpu = peak_kv_blocks_per_gpu.max(peaks.kv_blocks);
            peak_allocated_kv_tokens_per_gpu =
                peak_allocated_kv_tokens_per_gpu.max(peaks.allocated_kv_tokens);
            peak_kv_fragmentation_tokens_per_gpu =
                peak_kv_fragmentation_tokens_per_gpu.max(peaks.kv_fragmentation_tokens);
            peak_kv_block_table_bytes_per_gpu =
                peak_kv_block_table_bytes_per_gpu.max(peaks.kv_block_table_bytes);
            ServingGpuCapacityObservation {
                node_id: gpu.node_id,
                local_gpu_id: gpu.local_gpu_id,
                peak_prefill_tokens: prefill_peak,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                decode_sequence_utilization: traffic
                    .max_decode_sequences_per_gpu
                    .map(|capacity| f64::from(peaks.decode_sequences) / f64::from(capacity))
                    .unwrap_or(0.0),
                resident_token_utilization: traffic
                    .max_resident_tokens_per_gpu
                    .map(|capacity| peaks.resident_tokens as f64 / capacity as f64)
                    .unwrap_or(0.0),
                kv_block_utilization: traffic
                    .max_kv_blocks_per_gpu
                    .map(|capacity| peaks.kv_blocks as f64 / capacity as f64)
                    .unwrap_or(0.0),
            }
        })
        .collect();
    let traffic_classes: Vec<ServingTrafficClassCapacityObservation> = traffic
        .traffic_classes
        .iter()
        .map(|class| {
            let prefill_peak =
                prefill_token_peak(prefill_class_events.remove(&class.name).unwrap_or_default());
            let peaks = capacity_peaks(class_events.remove(&class.name).unwrap_or_default());
            ServingTrafficClassCapacityObservation {
                name: class.name.clone(),
                group: class.group.clone(),
                key: class.key.clone(),
                max_prefill_tokens: class.max_prefill_tokens,
                max_decode_sequences: class.max_decode_sequences,
                max_resident_tokens: class.max_resident_tokens,
                max_kv_blocks: class.max_kv_blocks,
                peak_prefill_tokens: prefill_peak,
                peak_decode_sequences: peaks.decode_sequences,
                peak_resident_tokens: peaks.resident_tokens,
                peak_kv_blocks: peaks.kv_blocks,
                peak_allocated_kv_tokens: peaks.allocated_kv_tokens,
                peak_kv_fragmentation_tokens: peaks.kv_fragmentation_tokens,
                peak_kv_block_table_bytes: peaks.kv_block_table_bytes,
                prefill_token_utilization: class
                    .max_prefill_tokens
                    .map(|capacity| prefill_peak as f64 / capacity as f64)
                    .unwrap_or(0.0),
                decode_sequence_utilization: class
                    .max_decode_sequences
                    .map(|capacity| f64::from(peaks.decode_sequences) / f64::from(capacity))
                    .unwrap_or(0.0),
                resident_token_utilization: class
                    .max_resident_tokens
                    .map(|capacity| peaks.resident_tokens as f64 / capacity as f64)
                    .unwrap_or(0.0),
                kv_block_utilization: class
                    .max_kv_blocks
                    .map(|capacity| peaks.kv_blocks as f64 / capacity as f64)
                    .unwrap_or(0.0),
            }
        })
        .collect();
    CapacityProfile {
        peak_prefill_tokens,
        peak_prefill_tokens_per_node,
        peak_prefill_tokens_per_gpu,
        peak_decode_sequences: aggregate_peaks.decode_sequences,
        peak_resident_tokens: aggregate_peaks.resident_tokens,
        peak_decode_sequences_per_node,
        peak_resident_tokens_per_node,
        peak_decode_sequences_per_gpu,
        peak_resident_tokens_per_gpu,
        peak_kv_blocks: aggregate_peaks.kv_blocks,
        peak_allocated_kv_tokens: aggregate_peaks.allocated_kv_tokens,
        peak_kv_fragmentation_tokens: aggregate_peaks.kv_fragmentation_tokens,
        peak_kv_block_table_bytes: aggregate_peaks.kv_block_table_bytes,
        peak_kv_blocks_per_node,
        peak_allocated_kv_tokens_per_node,
        peak_kv_fragmentation_tokens_per_node,
        peak_kv_block_table_bytes_per_node,
        peak_kv_blocks_per_gpu,
        peak_allocated_kv_tokens_per_gpu,
        peak_kv_fragmentation_tokens_per_gpu,
        peak_kv_block_table_bytes_per_gpu,
        decode_sequence_utilization: traffic
            .max_decode_sequences
            .map(|capacity| f64::from(aggregate_peaks.decode_sequences) / f64::from(capacity))
            .unwrap_or(0.0),
        resident_token_utilization: traffic
            .max_resident_tokens
            .map(|capacity| aggregate_peaks.resident_tokens as f64 / capacity as f64)
            .unwrap_or(0.0),
        kv_block_utilization: traffic
            .max_kv_blocks
            .map(|capacity| aggregate_peaks.kv_blocks as f64 / capacity as f64)
            .unwrap_or(0.0),
        decode_sequence_per_node_utilization: traffic
            .max_decode_sequences_per_node
            .map(|capacity| f64::from(peak_decode_sequences_per_node) / f64::from(capacity))
            .unwrap_or(0.0),
        resident_token_per_node_utilization: traffic
            .max_resident_tokens_per_node
            .map(|capacity| peak_resident_tokens_per_node as f64 / capacity as f64)
            .unwrap_or(0.0),
        kv_block_per_node_utilization: traffic
            .max_kv_blocks_per_node
            .map(|capacity| peak_kv_blocks_per_node as f64 / capacity as f64)
            .unwrap_or(0.0),
        decode_sequence_per_gpu_utilization: traffic
            .max_decode_sequences_per_gpu
            .map(|capacity| f64::from(peak_decode_sequences_per_gpu) / f64::from(capacity))
            .unwrap_or(0.0),
        resident_token_per_gpu_utilization: traffic
            .max_resident_tokens_per_gpu
            .map(|capacity| peak_resident_tokens_per_gpu as f64 / capacity as f64)
            .unwrap_or(0.0),
        kv_block_per_gpu_utilization: traffic
            .max_kv_blocks_per_gpu
            .map(|capacity| peak_kv_blocks_per_gpu as f64 / capacity as f64)
            .unwrap_or(0.0),
        nodes,
        gpus,
        traffic_classes,
    }
}

fn capacity_peaks(mut events: Vec<CapacityEvent>) -> CapacityPeaks {
    events.sort_by(|left, right| left.time_s.total_cmp(&right.time_s));

    let mut active_sequences = 0_i64;
    let mut resident_tokens = 0_i128;
    let mut kv_blocks = 0_i128;
    let mut allocated_kv_tokens = 0_i128;
    let mut kv_fragmentation_tokens = 0_i128;
    let mut kv_block_table_bytes = 0_i128;
    let mut peaks = CapacityPeaks::default();
    let mut idx = 0;

    while idx < events.len() {
        let event_time_s = events[idx].time_s;
        let mut sequence_delta = 0_i64;
        let mut token_delta = 0_i128;
        let mut block_delta = 0_i128;
        let mut allocated_delta = 0_i128;
        let mut fragmentation_delta = 0_i128;
        let mut block_table_delta = 0_i128;
        while idx < events.len() && events[idx].time_s.total_cmp(&event_time_s).is_eq() {
            sequence_delta += events[idx].sequence_delta;
            token_delta += events[idx].resident_token_delta;
            block_delta += events[idx].kv_block_delta;
            allocated_delta += events[idx].allocated_token_delta;
            fragmentation_delta += events[idx].fragmentation_token_delta;
            block_table_delta += events[idx].block_table_byte_delta;
            idx += 1;
        }

        active_sequences = (active_sequences + sequence_delta).max(0);
        resident_tokens = (resident_tokens + token_delta).max(0);
        kv_blocks = (kv_blocks + block_delta).max(0);
        allocated_kv_tokens = (allocated_kv_tokens + allocated_delta).max(0);
        kv_fragmentation_tokens = (kv_fragmentation_tokens + fragmentation_delta).max(0);
        kv_block_table_bytes = (kv_block_table_bytes + block_table_delta).max(0);
        peaks.decode_sequences = peaks.decode_sequences.max(active_sequences as u32);
        peaks.resident_tokens = peaks.resident_tokens.max(resident_tokens as u64);
        peaks.kv_blocks = peaks.kv_blocks.max(kv_blocks as u64);
        peaks.allocated_kv_tokens = peaks.allocated_kv_tokens.max(allocated_kv_tokens as u64);
        peaks.kv_fragmentation_tokens = peaks
            .kv_fragmentation_tokens
            .max(kv_fragmentation_tokens as u64);
        peaks.kv_block_table_bytes = peaks.kv_block_table_bytes.max(kv_block_table_bytes as u64);
    }

    peaks
}

fn prefill_token_peak(mut events: Vec<PrefillCapacityEvent>) -> u64 {
    events.sort_by(|left, right| left.time_s.total_cmp(&right.time_s));

    let mut active_tokens = 0_i128;
    let mut peak_tokens = 0_u64;
    let mut idx = 0;

    while idx < events.len() {
        let event_time_s = events[idx].time_s;
        let mut token_delta = 0_i128;
        while idx < events.len() && events[idx].time_s.total_cmp(&event_time_s).is_eq() {
            token_delta += events[idx].token_delta;
            idx += 1;
        }

        active_tokens = (active_tokens + token_delta).max(0);
        peak_tokens = peak_tokens.max(active_tokens as u64);
    }

    peak_tokens
}

fn capacity_rejections(
    metrics: &ServingMetrics,
    traffic: &ServingTraffic,
) -> Vec<CapacityRejection> {
    let mut rejections = Vec::new();
    if traffic.decode_capacity_policy == ServingDecodeCapacityPolicy::RequestReject {
        return rejections;
    }

    if let Some(max_prefill_tokens) = traffic.max_prefill_tokens
        && metrics.peak_prefill_tokens > max_prefill_tokens
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "prefill capacity exceeded: peak_prefill_tokens {} > max_prefill_tokens {}",
                metrics.peak_prefill_tokens, max_prefill_tokens
            ),
            bottleneck: "prefill token capacity".to_string(),
            resource: "prefill_tokens".to_string(),
            code: "prefill_capacity_exceeded".to_string(),
            observed: metrics.peak_prefill_tokens as f64,
            limit: max_prefill_tokens as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_prefill_tokens_per_node) = traffic.max_prefill_tokens_per_node
        && metrics.peak_prefill_tokens_per_node > max_prefill_tokens_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node prefill capacity exceeded: peak_prefill_tokens_per_node {} > max_prefill_tokens_per_node {}",
                metrics.peak_prefill_tokens_per_node, max_prefill_tokens_per_node
            ),
            bottleneck: "per-node prefill token capacity".to_string(),
            resource: "prefill_tokens_per_node".to_string(),
            code: "prefill_capacity_per_node_exceeded".to_string(),
            observed: metrics.peak_prefill_tokens_per_node as f64,
            limit: max_prefill_tokens_per_node as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_prefill_tokens_per_gpu) = traffic.max_prefill_tokens_per_gpu
        && metrics.peak_prefill_tokens_per_gpu > max_prefill_tokens_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU prefill capacity exceeded: peak_prefill_tokens_per_gpu {} > max_prefill_tokens_per_gpu {}",
                metrics.peak_prefill_tokens_per_gpu, max_prefill_tokens_per_gpu
            ),
            bottleneck: "per-GPU prefill token capacity".to_string(),
            resource: "prefill_tokens_per_gpu".to_string(),
            code: "prefill_capacity_per_gpu_exceeded".to_string(),
            observed: metrics.peak_prefill_tokens_per_gpu as f64,
            limit: max_prefill_tokens_per_gpu as f64,
            unit: "tokens".to_string(),
        });
    }

    if let Some(max_decode_sequences) = traffic.max_decode_sequences
        && metrics.peak_decode_sequences > max_decode_sequences
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "decode capacity exceeded: peak_decode_sequences {} > max_decode_sequences {}",
                metrics.peak_decode_sequences, max_decode_sequences
            ),
            bottleneck: "decode sequence capacity".to_string(),
            resource: "decode_sequences".to_string(),
            code: "decode_capacity_exceeded".to_string(),
            observed: f64::from(metrics.peak_decode_sequences),
            limit: f64::from(max_decode_sequences),
            unit: "sequences".to_string(),
        });
    }
    if let Some(max_resident_tokens) = traffic.max_resident_tokens
        && metrics.peak_resident_tokens > max_resident_tokens
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "KV residency capacity exceeded: peak_resident_tokens {} > max_resident_tokens {}",
                metrics.peak_resident_tokens, max_resident_tokens
            ),
            bottleneck: "KV residency capacity".to_string(),
            resource: "resident_tokens".to_string(),
            code: "kv_residency_capacity_exceeded".to_string(),
            observed: metrics.peak_resident_tokens as f64,
            limit: max_resident_tokens as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_kv_blocks) = traffic.max_kv_blocks
        && metrics.peak_kv_blocks > max_kv_blocks
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "KV block capacity exceeded: peak_kv_blocks {} > max_kv_blocks {}",
                metrics.peak_kv_blocks, max_kv_blocks
            ),
            bottleneck: "KV block capacity".to_string(),
            resource: "kv_blocks".to_string(),
            code: "kv_block_capacity_exceeded".to_string(),
            observed: metrics.peak_kv_blocks as f64,
            limit: max_kv_blocks as f64,
            unit: "blocks".to_string(),
        });
    }
    if let Some(max_decode_sequences_per_node) = traffic.max_decode_sequences_per_node
        && metrics.peak_decode_sequences_per_node > max_decode_sequences_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node decode capacity exceeded: peak_decode_sequences_per_node {} > max_decode_sequences_per_node {}",
                metrics.peak_decode_sequences_per_node, max_decode_sequences_per_node
            ),
            bottleneck: "per-node decode sequence capacity".to_string(),
            resource: "decode_sequences_per_node".to_string(),
            code: "decode_capacity_per_node_exceeded".to_string(),
            observed: f64::from(metrics.peak_decode_sequences_per_node),
            limit: f64::from(max_decode_sequences_per_node),
            unit: "sequences".to_string(),
        });
    }
    if let Some(max_resident_tokens_per_node) = traffic.max_resident_tokens_per_node
        && metrics.peak_resident_tokens_per_node > max_resident_tokens_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node KV residency capacity exceeded: peak_resident_tokens_per_node {} > max_resident_tokens_per_node {}",
                metrics.peak_resident_tokens_per_node, max_resident_tokens_per_node
            ),
            bottleneck: "per-node KV residency capacity".to_string(),
            resource: "resident_tokens_per_node".to_string(),
            code: "kv_residency_capacity_per_node_exceeded".to_string(),
            observed: metrics.peak_resident_tokens_per_node as f64,
            limit: max_resident_tokens_per_node as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_kv_blocks_per_node) = traffic.max_kv_blocks_per_node
        && metrics.peak_kv_blocks_per_node > max_kv_blocks_per_node
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-node KV block capacity exceeded: peak_kv_blocks_per_node {} > max_kv_blocks_per_node {}",
                metrics.peak_kv_blocks_per_node, max_kv_blocks_per_node
            ),
            bottleneck: "per-node KV block capacity".to_string(),
            resource: "kv_blocks_per_node".to_string(),
            code: "kv_block_capacity_per_node_exceeded".to_string(),
            observed: metrics.peak_kv_blocks_per_node as f64,
            limit: max_kv_blocks_per_node as f64,
            unit: "blocks".to_string(),
        });
    }
    if let Some(max_decode_sequences_per_gpu) = traffic.max_decode_sequences_per_gpu
        && metrics.peak_decode_sequences_per_gpu > max_decode_sequences_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU decode capacity exceeded: peak_decode_sequences_per_gpu {} > max_decode_sequences_per_gpu {}",
                metrics.peak_decode_sequences_per_gpu, max_decode_sequences_per_gpu
            ),
            bottleneck: "per-GPU decode sequence capacity".to_string(),
            resource: "decode_sequences_per_gpu".to_string(),
            code: "decode_capacity_per_gpu_exceeded".to_string(),
            observed: f64::from(metrics.peak_decode_sequences_per_gpu),
            limit: f64::from(max_decode_sequences_per_gpu),
            unit: "sequences".to_string(),
        });
    }
    if let Some(max_resident_tokens_per_gpu) = traffic.max_resident_tokens_per_gpu
        && metrics.peak_resident_tokens_per_gpu > max_resident_tokens_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU KV residency capacity exceeded: peak_resident_tokens_per_gpu {} > max_resident_tokens_per_gpu {}",
                metrics.peak_resident_tokens_per_gpu, max_resident_tokens_per_gpu
            ),
            bottleneck: "per-GPU KV residency capacity".to_string(),
            resource: "resident_tokens_per_gpu".to_string(),
            code: "kv_residency_capacity_per_gpu_exceeded".to_string(),
            observed: metrics.peak_resident_tokens_per_gpu as f64,
            limit: max_resident_tokens_per_gpu as f64,
            unit: "tokens".to_string(),
        });
    }
    if let Some(max_kv_blocks_per_gpu) = traffic.max_kv_blocks_per_gpu
        && metrics.peak_kv_blocks_per_gpu > max_kv_blocks_per_gpu
    {
        rejections.push(CapacityRejection {
            reason: format!(
                "per-GPU KV block capacity exceeded: peak_kv_blocks_per_gpu {} > max_kv_blocks_per_gpu {}",
                metrics.peak_kv_blocks_per_gpu, max_kv_blocks_per_gpu
            ),
            bottleneck: "per-GPU KV block capacity".to_string(),
            resource: "kv_blocks_per_gpu".to_string(),
            code: "kv_block_capacity_per_gpu_exceeded".to_string(),
            observed: metrics.peak_kv_blocks_per_gpu as f64,
            limit: max_kv_blocks_per_gpu as f64,
            unit: "blocks".to_string(),
        });
    }
    rejections
}

fn slo_rejections(
    metrics: &ServingMetrics,
    breakdowns: &[ServingMetricBreakdown],
    traffic: &ServingTraffic,
    policies: &[ServingSloPolicy],
) -> Vec<ServingRejection> {
    let mut rejections = Vec::new();
    push_slo_rejection(
        &mut rejections,
        "ttft",
        "TTFT",
        metrics.ttft_slo_miss_rate,
        traffic.max_ttft_slo_miss_rate,
    );
    push_slo_rejection(
        &mut rejections,
        "tpot",
        "TPOT",
        metrics.tpot_slo_miss_rate,
        traffic.max_tpot_slo_miss_rate,
    );
    push_slo_rejection(
        &mut rejections,
        "itl",
        "ITL",
        metrics.itl_slo_miss_rate,
        traffic.max_itl_slo_miss_rate,
    );
    push_slo_rejection(
        &mut rejections,
        "e2el",
        "E2EL",
        metrics.e2el_slo_miss_rate,
        traffic.max_e2el_slo_miss_rate,
    );
    push_slo_rejection(
        &mut rejections,
        "deadline",
        "deadline",
        metrics.deadline_miss_rate,
        traffic.max_deadline_miss_rate,
    );
    for policy in policies {
        push_scoped_slo_rejections(&mut rejections, breakdowns, policy);
    }
    rejections
}

fn push_scoped_slo_rejections(
    rejections: &mut Vec<ServingRejection>,
    breakdowns: &[ServingMetricBreakdown],
    policy: &ServingSloPolicy,
) {
    let Some(breakdown) = breakdowns
        .iter()
        .find(|breakdown| breakdown.group == policy.group && breakdown.key == policy.key)
    else {
        rejections.push(ServingRejection {
            phase: "serving".to_string(),
            category: "slo".to_string(),
            resource: format!("{}:{}:slo_policy", policy.group, policy.key),
            code: "scoped_slo_policy_scope_missing".to_string(),
            observed: None,
            limit: None,
            unit: None,
            remediation: Some(
                "check the policy group/key, trace metadata, and measurement window".to_string(),
            ),
            message: format!(
                "SLO policy scope missing: group={} key={}",
                policy.group, policy.key
            ),
        });
        return;
    };

    push_scoped_slo_rejection(
        rejections,
        policy,
        "ttft",
        "TTFT",
        breakdown.ttft_slo_miss_rate,
        policy.max_ttft_slo_miss_rate,
    );
    push_scoped_slo_rejection(
        rejections,
        policy,
        "tpot",
        "TPOT",
        breakdown.tpot_slo_miss_rate,
        policy.max_tpot_slo_miss_rate,
    );
    push_scoped_slo_rejection(
        rejections,
        policy,
        "itl",
        "ITL",
        breakdown.itl_slo_miss_rate,
        policy.max_itl_slo_miss_rate,
    );
    push_scoped_slo_rejection(
        rejections,
        policy,
        "e2el",
        "E2EL",
        breakdown.e2el_slo_miss_rate,
        policy.max_e2el_slo_miss_rate,
    );
    push_scoped_slo_rejection(
        rejections,
        policy,
        "deadline",
        "deadline",
        breakdown.deadline_miss_rate,
        policy.max_deadline_miss_rate,
    );
}

fn push_scoped_slo_rejection(
    rejections: &mut Vec<ServingRejection>,
    policy: &ServingSloPolicy,
    key: &str,
    label: &str,
    observed: f64,
    limit: Option<f64>,
) {
    let Some(limit) = limit else {
        return;
    };
    if observed.is_finite() && observed <= limit + 1e-12 {
        return;
    }
    let observed_label = if observed.is_finite() {
        format!("{observed:.3}")
    } else {
        "unavailable".to_string()
    };
    rejections.push(ServingRejection {
        phase: "serving".to_string(),
        category: "slo".to_string(),
        resource: format!("{}:{}:{key}_miss_rate", policy.group, policy.key),
        code: format!("scoped_{key}_miss_rate_exceeded"),
        observed: observed.is_finite().then_some(observed),
        limit: Some(limit),
        unit: Some("ratio".to_string()),
        remediation: Some(
            "increase scoped serving capacity, adjust routing/batching, relax the scoped SLO, or raise the scoped miss-rate limit"
                .to_string(),
        ),
        message: format!(
            "{label} miss rate exceeded for {}={}: observed {observed_label} > max {limit:.3}",
            policy.group, policy.key
        ),
    });
}

fn push_slo_rejection(
    rejections: &mut Vec<ServingRejection>,
    key: &str,
    label: &str,
    observed: f64,
    limit: Option<f64>,
) {
    let Some(limit) = limit else {
        return;
    };
    if observed.is_finite() && observed <= limit + 1e-12 {
        return;
    }
    let unavailable = if observed.is_finite() {
        String::new()
    } else {
        " is unavailable and".to_string()
    };
    let observed_label = if observed.is_finite() {
        format!("{observed:.3}")
    } else {
        "unavailable".to_string()
    };
    rejections.push(ServingRejection {
        phase: "serving".to_string(),
        category: "slo".to_string(),
        resource: format!("{key}_miss_rate"),
        code: format!("{key}_miss_rate_exceeded"),
        observed: observed.is_finite().then_some(observed),
        limit: Some(limit),
        unit: Some("ratio".to_string()),
        remediation: Some(
            "increase serving capacity, adjust routing/batching, relax SLOs, or raise the configured miss-rate limit"
                .to_string(),
        ),
        message: format!(
            "{label} miss rate{unavailable} exceeded: observed {observed_label} > max {limit:.3}"
        ),
    });
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }

    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }

    values.sort_by(f64::total_cmp);
    let bounded = quantile.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * bounded).ceil() as usize;
    values[idx]
}

fn max_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(f64::INFINITY)
}

#[cfg(test)]
mod tests;
