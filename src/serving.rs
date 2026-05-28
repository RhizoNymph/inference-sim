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

mod model;
pub use model::*;

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

fn memory_headroom(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    phase: InferencePhase,
    score: &ScoredParallelismConfig,
    kv_block_tokens: u32,
    calibration: SimulationCalibration,
) -> ServingMemoryHeadroom {
    let components = serving_memory_components(
        model,
        request,
        phase,
        score.config,
        kv_block_tokens,
        calibration,
    );
    let estimated_per_gpu_gb = components.total_gb;
    let mut min_hbm_per_gpu_gb = f64::INFINITY;
    let mut limiting_gpu = None;
    for addr in &score.placement.rank_to_gpu {
        let Some(profile) = cluster.gpu_profile(*addr) else {
            continue;
        };
        let hbm_gb = profile.hbm_size.as_gigabytes();
        if hbm_gb < min_hbm_per_gpu_gb {
            min_hbm_per_gpu_gb = hbm_gb;
            limiting_gpu = Some(*addr);
        }
    }

    if !estimated_per_gpu_gb.is_finite()
        || !min_hbm_per_gpu_gb.is_finite()
        || min_hbm_per_gpu_gb <= 0.0
    {
        return ServingMemoryHeadroom::unavailable();
    }

    let headroom_gb = min_hbm_per_gpu_gb - estimated_per_gpu_gb;
    ServingMemoryHeadroom {
        estimated_per_gpu_gb,
        min_hbm_per_gpu_gb,
        limiting_gpu,
        headroom_gb,
        headroom_fraction: headroom_gb / min_hbm_per_gpu_gb,
        components,
    }
}

#[derive(Copy, Clone, Debug)]
struct MemoryPressureSpan {
    start_s: f64,
    finish_s: f64,
    active_tokens: u64,
    kv_blocks: u64,
}

fn memory_pressure_observations(
    observations: &[ServingRequestObservation],
    prefill_memory: &ServingMemoryHeadroom,
    decode_memory: &ServingMemoryHeadroom,
) -> Vec<ServingMemoryPressureObservation> {
    let mut pressure = Vec::new();
    push_memory_pressure_phase(
        &mut pressure,
        "prefill",
        "active_request_component_scaled_prefill_memory",
        prefill_memory,
        true,
        observations
            .iter()
            .filter_map(|observation| {
                memory_pressure_span(
                    observation.prefill_start_s,
                    observation.prefill_finish_s,
                    u64::from(observation.batch_size)
                        .saturating_mul(u64::from(observation.effective_prefill_tokens.max(1))),
                    0,
                )
            })
            .collect(),
    );

    let kv_memory = kv_transfer_pressure_memory(prefill_memory, decode_memory);
    push_memory_pressure_phase(
        &mut pressure,
        "kv_transfer",
        "active_request_component_scaled_kv_transfer_memory",
        &kv_memory,
        false,
        observations
            .iter()
            .filter_map(|observation| {
                memory_pressure_span(
                    observation.kv_start_s,
                    observation.kv_finish_s,
                    observation.kv_allocated_tokens,
                    observation.kv_cache_blocks,
                )
            })
            .collect(),
    );

    push_memory_pressure_phase(
        &mut pressure,
        "decode",
        "active_request_component_scaled_decode_memory",
        decode_memory,
        true,
        observations
            .iter()
            .filter_map(|observation| {
                memory_pressure_span(
                    observation.first_decode_start_s,
                    observation.last_decode_finish_s,
                    observation.kv_allocated_tokens,
                    observation.kv_cache_blocks,
                )
            })
            .collect(),
    );

    pressure
}

fn memory_pressure_rejection(
    observations: &[ServingMemoryPressureObservation],
    max_memory_pressure_fraction: Option<f64>,
) -> Option<ServingRejection> {
    let limit = max_memory_pressure_fraction?;
    let peak = observations
        .iter()
        .filter(|observation| observation.capacity_used_fraction.is_finite())
        .max_by(|left, right| {
            left.capacity_used_fraction
                .total_cmp(&right.capacity_used_fraction)
        })?;

    if peak.capacity_used_fraction <= limit + 1e-12 {
        return None;
    }

    Some(ServingRejection {
        phase: peak.phase.clone(),
        category: "memory".to_string(),
        resource: "memory_pressure".to_string(),
        code: "memory_pressure_fraction_exceeded".to_string(),
        observed: Some(peak.capacity_used_fraction),
        limit: Some(limit),
        unit: Some("fraction".to_string()),
        remediation: Some(
            "increase tensor/pipeline sharding, use GPUs with more HBM, reduce batch/sequence/KV residency, or raise the configured max_memory_pressure_fraction"
                .to_string(),
        ),
        message: format!(
            "{} memory pressure {:.3} exceeds configured limit {:.3}",
            peak.phase, peak.capacity_used_fraction, limit
        ),
    })
}

fn gpu_footprint_rejection(
    footprint: &ServingHardwareFootprint,
    max_unique_gpus: Option<u32>,
) -> Option<ServingRejection> {
    let limit = max_unique_gpus?;
    if footprint.unique_gpu_count <= limit {
        return None;
    }

    Some(ServingRejection {
        phase: "placement".to_string(),
        category: "capacity".to_string(),
        resource: "unique_gpus".to_string(),
        code: "unique_gpu_footprint_exceeded".to_string(),
        observed: Some(f64::from(footprint.unique_gpu_count)),
        limit: Some(f64::from(limit)),
        unit: Some("gpus".to_string()),
        remediation: Some(
            "reduce prefill/decode ranks, choose smaller pools, allow colocated placement, or raise serving.max_unique_gpus"
                .to_string(),
        ),
        message: format!(
            "serving candidate uses {} unique GPUs, exceeding configured max_unique_gpus {}",
            footprint.unique_gpu_count, limit
        ),
    })
}

fn throughput_floor_rejection(
    metrics: &ServingMetrics,
    min_throughput_tokens_per_s: Option<f64>,
) -> Option<ServingRejection> {
    let limit = min_throughput_tokens_per_s?;
    if metrics.throughput_tokens_per_s.is_finite() && metrics.throughput_tokens_per_s >= limit {
        return None;
    }

    Some(ServingRejection {
        phase: "serving".to_string(),
        category: "throughput".to_string(),
        resource: "throughput_tokens_per_s".to_string(),
        code: "throughput_below_min".to_string(),
        observed: metrics
            .throughput_tokens_per_s
            .is_finite()
            .then_some(metrics.throughput_tokens_per_s),
        limit: Some(limit),
        unit: Some("tokens/s".to_string()),
        remediation: Some(
            "increase serving parallelism or worker capacity, reduce latency/queueing pressure, choose a faster pool, or lower serving.min_throughput_tokens_per_s"
                .to_string(),
        ),
        message: format!(
            "serving throughput {:.3} tokens/s is below configured minimum {:.3} tokens/s",
            metrics.throughput_tokens_per_s, limit
        ),
    })
}

fn metric_ceiling_rejections(
    metrics: &ServingMetrics,
    ceilings: ServingMetricCeilings,
) -> Vec<ServingRejection> {
    if !ceilings.any() {
        return Vec::new();
    }

    [
        (
            "prefill",
            "ttft_s",
            "ttft_above_max",
            metrics.ttft_s,
            ceilings.max_ttft_s,
            "serving.max_ttft_s",
        ),
        (
            "decode",
            "tpot_s",
            "tpot_above_max",
            metrics.tpot_s,
            ceilings.max_tpot_s,
            "serving.max_tpot_s",
        ),
        (
            "decode",
            "itl_s",
            "itl_above_max",
            metrics.itl_s,
            ceilings.max_itl_s,
            "serving.max_itl_s",
        ),
        (
            "serving",
            "e2el_s",
            "e2el_above_max",
            metrics.e2el_s,
            ceilings.max_e2el_s,
            "serving.max_e2el_s",
        ),
    ]
    .into_iter()
    .filter_map(
        |(phase, resource, code, observed, limit, config_field)| -> Option<ServingRejection> {
            let limit = limit?;
            if observed.is_finite() && observed <= limit + 1e-12 {
                return None;
            }
            Some(ServingRejection {
                phase: phase.to_string(),
                category: "latency".to_string(),
                resource: resource.to_string(),
                code: code.to_string(),
                observed: observed.is_finite().then_some(observed),
                limit: Some(limit),
                unit: Some("s".to_string()),
                remediation: Some(format!(
                    "reduce queueing/service time, change prefill/decode placement or parallelism, choose a faster pool, or raise {config_field}"
                )),
                message: format!(
                    "{resource} {:.6}s exceeds configured maximum {:.6}s",
                    observed, limit
                ),
            })
        },
    )
    .collect()
}

fn memory_pressure_span(
    start_s: f64,
    finish_s: f64,
    active_tokens: u64,
    kv_blocks: u64,
) -> Option<MemoryPressureSpan> {
    if start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
        Some(MemoryPressureSpan {
            start_s,
            finish_s,
            active_tokens,
            kv_blocks,
        })
    } else {
        None
    }
}

fn kv_transfer_pressure_memory(
    prefill_memory: &ServingMemoryHeadroom,
    decode_memory: &ServingMemoryHeadroom,
) -> ServingMemoryHeadroom {
    let mut memory = *decode_memory;
    if prefill_memory.min_hbm_per_gpu_gb < memory.min_hbm_per_gpu_gb {
        memory.min_hbm_per_gpu_gb = prefill_memory.min_hbm_per_gpu_gb;
        memory.limiting_gpu = prefill_memory.limiting_gpu;
    }
    if prefill_memory.components.weights_gb > memory.components.weights_gb {
        memory.components.weights_gb = prefill_memory.components.weights_gb;
    }
    memory
}

fn push_memory_pressure_phase(
    pressure: &mut Vec<ServingMemoryPressureObservation>,
    phase: &str,
    estimate_kind: &str,
    memory: &ServingMemoryHeadroom,
    include_activation_components: bool,
    spans: Vec<MemoryPressureSpan>,
) {
    if spans.is_empty() {
        return;
    }

    let mut boundaries = spans
        .iter()
        .flat_map(|span| [span.start_s, span.finish_s])
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    boundaries.sort_by(|left, right| left.total_cmp(right));
    boundaries.dedup_by(|left, right| (*left - *right).abs() <= 1e-12);

    for window in boundaries.windows(2) {
        let start_s = window[0];
        let finish_s = window[1];
        if finish_s <= start_s {
            continue;
        }

        let active_spans = spans
            .iter()
            .filter(|span| span.start_s < finish_s - 1e-12 && span.finish_s > start_s + 1e-12)
            .collect::<Vec<_>>();
        if active_spans.is_empty() {
            continue;
        }

        let active_requests = active_spans.len().min(u32::MAX as usize) as u32;
        let active_tokens = active_spans.iter().fold(0_u64, |total, span| {
            total.saturating_add(span.active_tokens)
        });
        let kv_blocks = active_spans
            .iter()
            .fold(0_u64, |total, span| total.saturating_add(span.kv_blocks));
        let components = scaled_memory_pressure_components(
            memory.components,
            f64::from(active_requests),
            include_activation_components,
        );
        let estimated_per_gpu_gb = components.total_gb;
        let capacity_used_fraction = if estimated_per_gpu_gb.is_finite()
            && memory.min_hbm_per_gpu_gb.is_finite()
            && memory.min_hbm_per_gpu_gb > 0.0
        {
            estimated_per_gpu_gb / memory.min_hbm_per_gpu_gb
        } else {
            f64::INFINITY
        };
        let headroom_gb =
            if estimated_per_gpu_gb.is_finite() && memory.min_hbm_per_gpu_gb.is_finite() {
                memory.min_hbm_per_gpu_gb - estimated_per_gpu_gb
            } else {
                f64::INFINITY
            };

        pressure.push(ServingMemoryPressureObservation {
            phase: phase.to_string(),
            estimate_kind: estimate_kind.to_string(),
            start_s,
            finish_s,
            duration_s: finish_s - start_s,
            active_requests,
            active_tokens,
            kv_blocks,
            estimated_per_gpu_gb,
            min_hbm_per_gpu_gb: memory.min_hbm_per_gpu_gb,
            capacity_used_fraction,
            headroom_gb,
            limiting_gpu: memory.limiting_gpu,
            dominant_component: components.dominant_component(),
            components,
        });
    }
}

fn scaled_memory_pressure_components(
    base: ServingMemoryComponents,
    active_requests: f64,
    include_activation_components: bool,
) -> ServingMemoryComponents {
    let scale = active_requests.max(0.0);
    let weights_gb = base.weights_gb;
    let kv_cache_gb = base.kv_cache_gb * scale;
    let block_table_gb = base.block_table_gb * scale;
    let activations_gb = if include_activation_components {
        base.activations_gb * scale
    } else {
        0.0
    };
    let temporary_gb = if include_activation_components {
        base.temporary_gb * scale
    } else {
        0.0
    };
    let communication_gb = base.communication_gb * scale;
    let runtime_reserve_gb = base.runtime_reserve_gb * scale;
    let fragmentation_gb = base.fragmentation_gb * scale;
    let total_gb = weights_gb
        + kv_cache_gb
        + block_table_gb
        + activations_gb
        + temporary_gb
        + communication_gb
        + runtime_reserve_gb
        + fragmentation_gb;

    ServingMemoryComponents {
        weights_gb,
        kv_cache_gb,
        block_table_gb,
        activations_gb,
        temporary_gb,
        communication_gb,
        runtime_reserve_gb,
        fragmentation_gb,
        total_gb,
    }
}

fn serving_memory_components(
    model: &ModelSpec,
    request: &InferenceRequest,
    phase: InferencePhase,
    config: ParallelismConfig,
    kv_block_tokens: u32,
    calibration: SimulationCalibration,
) -> ServingMemoryComponents {
    let calibration = calibration.sanitized();
    let shard_factor =
        f64::from((config.tensor_ranks * config.pipeline_ranks * config.expert_ranks).max(1));
    let tensor_ranks = f64::from(config.tensor_ranks.max(1));

    let weights_gb = model.parameters.as_gigabytes() / shard_factor;
    let kv_cache_gb = kv_cache_gb(model, request, config);
    let block_table_gb = kv_block_table_gb(request, config, kv_block_tokens);
    let activations_gb = activation_memory_gb(model, request, phase, tensor_ranks);
    let temporary_gb = activations_gb * calibration.serving_memory_temporary_fraction;
    let communication_gb = (activations_gb
        * calibration.serving_memory_activation_communication_fraction)
        .max(weights_gb * calibration.serving_memory_weight_communication_fraction);

    let subtotal_gb = weights_gb
        + kv_cache_gb
        + block_table_gb
        + activations_gb
        + temporary_gb
        + communication_gb;
    let runtime_reserve_gb = subtotal_gb * calibration.serving_memory_runtime_reserve_fraction;
    let fragmentation_gb =
        (weights_gb + kv_cache_gb) * calibration.serving_memory_fragmentation_fraction;
    let total_gb = subtotal_gb + runtime_reserve_gb + fragmentation_gb;

    ServingMemoryComponents {
        weights_gb,
        kv_cache_gb,
        block_table_gb,
        activations_gb,
        temporary_gb,
        communication_gb,
        runtime_reserve_gb,
        fragmentation_gb,
        total_gb,
    }
}

fn kv_cache_gb(model: &ModelSpec, request: &InferenceRequest, config: ParallelismConfig) -> f64 {
    let head_dim = f64::from(model.hidden_size) / f64::from(model.attention_heads.max(1));
    let tokens = f64::from(request.max_sequence_tokens) * f64::from(request.batch_size);
    let bytes = tokens
        * f64::from(model.kv_heads)
        * head_dim
        * 2.0
        * model.kv_dtype().bytes_per_element() as f64
        * f64::from(model.layers)
        / f64::from(config.tensor_ranks.max(1));

    bytes / 1e9
}

fn kv_block_table_gb(
    request: &InferenceRequest,
    config: ParallelismConfig,
    block_tokens: u32,
) -> f64 {
    let block_tokens = u64::from(block_tokens.max(1));
    let blocks_per_sequence = u64::from(request.max_sequence_tokens.max(1)).div_ceil(block_tokens);
    let total_blocks = u64::from(request.batch_size.max(1)).saturating_mul(blocks_per_sequence);
    let local_blocks = total_blocks.div_ceil(u64::from(config.tensor_ranks.max(1)));
    local_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES) as f64 / 1e9
}

fn activation_memory_gb(
    model: &ModelSpec,
    request: &InferenceRequest,
    phase: InferencePhase,
    tensor_ranks: f64,
) -> f64 {
    let active_tokens = match phase {
        InferencePhase::Prefill => request.prompt_tokens.max(1),
        InferencePhase::Decode => 1,
        InferencePhase::EndToEnd => request.prompt_tokens.saturating_add(request.decode_tokens),
    };
    let bytes = f64::from(request.batch_size)
        * f64::from(active_tokens)
        * f64::from(model.hidden_size)
        * model.dtype.bytes_per_element() as f64
        / tensor_ranks.max(1.0);

    bytes / 1e9
}

fn memory_headroom_rejection(
    phase: &str,
    memory: &ServingMemoryHeadroom,
) -> Option<ServingRejection> {
    if !memory.headroom_gb.is_finite() || memory.headroom_gb >= 0.0 {
        return None;
    }

    let dominant_component = memory
        .dominant_component()
        .map(|component| {
            format!(
                "; dominant_component={} {:.2} GB ({:.1}% of estimate)",
                component.name,
                component.gb,
                component.fraction_of_total * 100.0
            )
        })
        .unwrap_or_default();
    let limiting_gpu = memory
        .limiting_gpu
        .map(|addr| {
            format!(
                "; limiting_gpu=node:{} gpu:{}",
                addr.node_id, addr.local_gpu_id
            )
        })
        .unwrap_or_default();

    Some(ServingRejection {
        phase: phase.to_string(),
        category: "memory".to_string(),
        resource: "gpu_hbm".to_string(),
        code: "serving_memory_headroom_exceeded".to_string(),
        observed: Some(memory.estimated_per_gpu_gb),
        limit: Some(memory.min_hbm_per_gpu_gb),
        unit: Some("GB".to_string()),
        remediation: Some(
            "increase sharding, use GPUs with more HBM, or reduce batch/sequence/KV residency"
                .to_string(),
        ),
        message: format!(
            "{phase} serving memory estimate {:.2} GB per GPU exceeds {:.2} GB HBM{dominant_component}{limiting_gpu}",
            memory.estimated_per_gpu_gb, memory.min_hbm_per_gpu_gb
        ),
    })
}

fn pool_rejection(pool: &ResolvedServingPool, message: String) -> ServingRejection {
    ServingRejection {
        phase: "pool".to_string(),
        category: "pool".to_string(),
        resource: pool_resource(pool),
        code: "pool_invalid".to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: Some(
            "check serving pool node IDs, group labels, overlap policy, and pool search bounds"
                .to_string(),
        ),
        message,
    }
}

fn pool_resource(pool: &ResolvedServingPool) -> String {
    pool.label.clone().unwrap_or_else(|| {
        format!(
            "prefill_nodes={:?} decode_nodes={:?}",
            pool.prefill_nodes, pool.decode_nodes
        )
    })
}

fn service_health_rejections(
    traffic: &ServingTraffic,
    pool: &ResolvedServingPool,
) -> Vec<ServingRejection> {
    let mut rejections = Vec::new();
    push_service_health_rejection(&mut rejections, "prefill", traffic.services.prefill, true);
    push_service_health_rejection(&mut rejections, "decode", traffic.services.decode, true);
    push_service_health_rejection(
        &mut rejections,
        "kv_transfer",
        traffic.services.kv_transfer,
        !same_u32s(&pool.prefill_nodes, &pool.decode_nodes),
    );
    rejections
}

fn push_service_health_rejection(
    rejections: &mut Vec<ServingRejection>,
    phase: &str,
    service: ServingServicePhaseConfig,
    required: bool,
) {
    if !required || service.health.accepts_requests() {
        return;
    }

    let code = match service.health {
        ServingServiceHealth::Healthy => return,
        ServingServiceHealth::Draining => "service_draining",
        ServingServiceHealth::Unavailable => "service_unavailable",
    };
    rejections.push(ServingRejection {
        phase: phase.to_string(),
        category: "service".to_string(),
        resource: format!("{phase}_service"),
        code: code.to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: Some(format!(
            "restore {phase} service health or choose a serving pool whose required services are healthy"
        )),
        message: format!(
            "{phase} service is {} and cannot accept new serving requests",
            service.health.as_str()
        ),
    });
}

#[allow(clippy::too_many_arguments)]
fn route_availability_rejections(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    pool: &ResolvedServingPool,
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    calibration: SimulationCalibration,
    route_coverage: ServingRouteCoverage,
) -> Vec<ServingRejection> {
    if !prefill_score.feasible || !decode_one_score.feasible {
        return Vec::new();
    }

    let calibration = calibration.sanitized();
    let prefill_resource_base_node = single_placement_node(prefill_score);
    let prefill_placement_nodes = placement_nodes(prefill_score);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let decode_placement_nodes = placement_nodes(decode_one_score);

    match traffic.routing_policy {
        ServingRoutingPolicy::RoundRobin => {
            let request_count = traffic.request_count(calibration);
            for request_idx in 0..request_count {
                let request_shape = traffic.request_at(request, request_idx);
                let prefill_node = routed_serving_node(&pool.prefill_nodes, request_idx);
                let decode_node = routed_serving_node(&pool.decode_nodes, request_idx);
                let prefill_route_nodes = routed_node_set(
                    prefill_resource_base_node,
                    &prefill_placement_nodes,
                    prefill_node,
                );
                let decode_route_nodes = routed_node_set(
                    decode_resource_base_node,
                    &decode_placement_nodes,
                    decode_node,
                );
                let prefill_route_gpus = routed_gpu_set(
                    prefill_resource_base_node,
                    &prefill_score.placement.rank_to_gpu,
                    prefill_node,
                );
                let decode_route_gpus = routed_gpu_set(
                    decode_resource_base_node,
                    &decode_one_score.placement.rank_to_gpu,
                    decode_node,
                );
                let kv_bytes = kv_transfer_bytes_for_routes(
                    model,
                    &request_shape,
                    &prefill_route_nodes,
                    &decode_route_nodes,
                    &prefill_route_gpus,
                    &decode_route_gpus,
                );
                if !Solver::transfer_between_gpus_routable(
                    cluster,
                    &prefill_route_gpus,
                    &decode_route_gpus,
                    kv_bytes,
                ) {
                    return vec![route_unavailable_rejection(
                        "round_robin",
                        Some(request_idx),
                        &prefill_route_nodes,
                        &decode_route_nodes,
                        route_coverage,
                    )];
                }
            }
            Vec::new()
        }
        ServingRoutingPolicy::TopologyAware => {
            let prefill_candidates = sorted_unique_nodes(&pool.prefill_nodes);
            let decode_candidates = sorted_unique_nodes(&pool.decode_nodes);
            let request_shape = traffic.request_at(request, 0);
            for prefill_node in prefill_candidates {
                let prefill_route_nodes = routed_node_set(
                    prefill_resource_base_node,
                    &prefill_placement_nodes,
                    prefill_node,
                );
                for decode_node in &decode_candidates {
                    let decode_route_nodes = routed_node_set(
                        decode_resource_base_node,
                        &decode_placement_nodes,
                        *decode_node,
                    );
                    let prefill_route_gpus = routed_gpu_set(
                        prefill_resource_base_node,
                        &prefill_score.placement.rank_to_gpu,
                        prefill_node,
                    );
                    let decode_route_gpus = routed_gpu_set(
                        decode_resource_base_node,
                        &decode_one_score.placement.rank_to_gpu,
                        *decode_node,
                    );
                    let kv_bytes = kv_transfer_bytes_for_routes(
                        model,
                        &request_shape,
                        &prefill_route_nodes,
                        &decode_route_nodes,
                        &prefill_route_gpus,
                        &decode_route_gpus,
                    );
                    if Solver::transfer_between_gpus_routable(
                        cluster,
                        &prefill_route_gpus,
                        &decode_route_gpus,
                        kv_bytes,
                    ) {
                        return Vec::new();
                    }
                }
            }

            vec![route_unavailable_rejection(
                "topology_aware",
                None,
                &pool.prefill_nodes,
                &pool.decode_nodes,
                route_coverage,
            )]
        }
    }
}

fn route_unavailable_rejection(
    policy: &str,
    request_idx: Option<u32>,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    route_coverage: ServingRouteCoverage,
) -> ServingRejection {
    let request_context = request_idx
        .map(|idx| format!(" for request {idx}"))
        .unwrap_or_default();
    ServingRejection {
        phase: "kv_transfer".to_string(),
        category: "topology".to_string(),
        resource: "kv_transfer_route".to_string(),
        code: "kv_transfer_route_unavailable".to_string(),
        observed: Some(f64::from(route_coverage.routable_candidate_count)),
        limit: Some(1.0),
        unit: Some("routes".to_string()),
        remediation: Some(
            "choose connected prefill/decode pools or add the missing custom inter-node links"
                .to_string(),
        ),
        message: format!(
            "{policy} KV transfer route unavailable{request_context}: route_coverage={}/{} routable, unroutable_candidates={}, prefill_nodes={prefill_nodes:?} decode_nodes={decode_nodes:?}",
            route_coverage.routable_candidate_count,
            route_coverage.candidate_count,
            route_coverage.unroutable_candidate_count,
        ),
    }
}

fn route_coverage(
    cluster: &Cluster,
    pool: &ResolvedServingPool,
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
) -> ServingRouteCoverage {
    if !prefill_score.feasible || !decode_one_score.feasible {
        return ServingRouteCoverage::default();
    }

    let prefill_resource_base_node = single_placement_node(prefill_score);
    let decode_resource_base_node = single_placement_node(decode_one_score);
    let prefill_candidates = sorted_unique_nodes(&pool.prefill_nodes);
    let decode_candidates = sorted_unique_nodes(&pool.decode_nodes);
    let mut candidate_count = 0usize;
    let mut routable_candidate_count = 0usize;

    for prefill_node in prefill_candidates {
        let prefill_route_gpus = routed_gpu_set(
            prefill_resource_base_node,
            &prefill_score.placement.rank_to_gpu,
            prefill_node,
        );
        for decode_node in &decode_candidates {
            candidate_count += 1;
            let decode_route_gpus = routed_gpu_set(
                decode_resource_base_node,
                &decode_one_score.placement.rank_to_gpu,
                *decode_node,
            );
            if Solver::transfer_between_gpus_routable(
                cluster,
                &prefill_route_gpus,
                &decode_route_gpus,
                Bytes::from_bytes(1),
            ) {
                routable_candidate_count += 1;
            }
        }
    }

    let bounded_candidate_count = candidate_count.min(u32::MAX as usize) as u32;
    let bounded_routable_candidate_count = routable_candidate_count.min(u32::MAX as usize) as u32;
    let fraction = if candidate_count == 0 {
        0.0
    } else {
        routable_candidate_count as f64 / candidate_count as f64
    };

    ServingRouteCoverage {
        candidate_count: bounded_candidate_count,
        routable_candidate_count: bounded_routable_candidate_count,
        unroutable_candidate_count: bounded_candidate_count
            .saturating_sub(bounded_routable_candidate_count),
        fraction,
    }
}

fn parallelism_rejections(
    phase: &'static str,
    score: &ScoredParallelismConfig,
) -> Vec<ServingRejection> {
    if score.feasible {
        return Vec::new();
    }

    let placement_rejections: Vec<_> = score
        .placement_evidence
        .iter()
        .filter(|evidence| evidence.decision == "rejected")
        .map(|evidence| placement_evidence_rejection(phase, evidence))
        .collect();
    if !placement_rejections.is_empty() {
        return placement_rejections;
    }

    let message = score
        .rejected_reason
        .clone()
        .unwrap_or_else(|| format!("{phase} parallelism config rejected"));
    let category = classify_rejection_category(&message);
    let code = classify_rejection_code(category, &message);
    vec![ServingRejection {
        phase: phase.to_string(),
        category: category.to_string(),
        resource: classify_rejection_resource(category, &message).to_string(),
        code: code.to_string(),
        observed: None,
        limit: None,
        unit: None,
        remediation: rejection_remediation(category, code, phase).map(str::to_string),
        message,
    }]
}

fn placement_evidence_rejection(
    phase: &'static str,
    evidence: &PlacementEvidence,
) -> ServingRejection {
    ServingRejection {
        phase: phase.to_string(),
        category: placement_evidence_rejection_category(evidence).to_string(),
        resource: evidence.resource.clone(),
        code: evidence.code.clone(),
        observed: evidence.observed,
        limit: evidence.limit,
        unit: evidence.unit.clone(),
        remediation: evidence.remediation.clone().or_else(|| {
            rejection_remediation(
                placement_evidence_rejection_category(evidence),
                &evidence.code,
                phase,
            )
            .map(str::to_string)
        }),
        message: evidence.message.clone(),
    }
}

fn placement_evidence_rejection_category(evidence: &PlacementEvidence) -> &'static str {
    let code = evidence.code.as_str();
    let resource = evidence.resource.as_str();
    if code.contains("hbm") || resource == "gpu_hbm" {
        "memory"
    } else if code.contains("dtype") {
        "capability"
    } else if code.contains("available_gpus") || resource == "available_gpus" {
        "capacity"
    } else if code.contains("placement")
        || code.contains("rank_count")
        || code.contains("unknown_gpu")
        || code.contains("duplicate_gpu")
        || code.contains("gpu_unavailable")
    {
        "placement"
    } else {
        classify_rejection_category(&evidence.message)
    }
}

fn classify_rejection_category(message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    if message.contains("hbm") || message.contains("memory") || message.contains("gb per gpu") {
        "memory"
    } else if message.contains("link")
        || message.contains("route")
        || message.contains("topology")
        || message.contains("fabric")
    {
        "topology"
    } else if message.contains("gpu") && message.contains("available") {
        "capacity"
    } else {
        "parallelism"
    }
}

fn classify_rejection_resource(category: &str, message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    match category {
        "memory" => "gpu_hbm",
        "topology" => "interconnect",
        "capacity" if message.contains("gpu") => "gpu_count",
        "capacity" => "serving_capacity",
        _ => "parallelism_config",
    }
}

fn classify_rejection_code(category: &str, message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    match category {
        "memory" => "memory_capacity_exceeded",
        "topology" => "topology_unavailable",
        "capacity" if message.contains("gpu") => "gpu_capacity_insufficient",
        "capacity" => "serving_capacity_exceeded",
        _ => "parallelism_config_invalid",
    }
}

fn rejection_remediation(category: &str, code: &str, phase: &str) -> Option<&'static str> {
    match (category, code) {
        ("memory", _) => {
            Some("increase sharding, use GPUs with more HBM, or reduce batch/sequence/KV residency")
        }
        ("topology", _) => Some(
            "choose a pool or placement inside connected topology domains, or add the missing links",
        ),
        ("capacity", "gpu_capacity_insufficient") => {
            Some("reduce total ranks or expand the selected GPU/node pool")
        }
        ("capacity", _) => Some("raise serving capacity limits or reduce offered traffic"),
        ("parallelism", _) if phase == "decode_first_token" => {
            Some("adjust decode search space so one-token decode is feasible")
        }
        ("parallelism", _) => Some("adjust tensor/pipeline/expert/data ranks for this phase"),
        _ => None,
    }
}

fn joined_rejection_reason(rejections: &[ServingRejection]) -> Option<String> {
    if rejections.is_empty() {
        None
    } else {
        Some(
            rejections
                .iter()
                .map(|rejection| rejection.message.as_str())
                .collect::<Vec<_>>()
                .join("; "),
        )
    }
}

#[derive(Default)]
struct ServingHardwareCapacity {
    node_count: u32,
    gpu_count: u32,
    hbm_gb: f64,
    hbm_bandwidth_gb_s: f64,
    peak_f16_tflops: f64,
    peak_f8_tflops: Option<f64>,
    effective_peak_tflops: f64,
}

fn serving_hardware_footprint(
    cluster: &Cluster,
    model: &ModelSpec,
    prefill_placement: &RankPlacement,
    decode_placement: &RankPlacement,
    throughput_tokens_per_s: f64,
) -> ServingHardwareFootprint {
    let prefill_gpus = placement_gpu_set(prefill_placement);
    let decode_gpus = placement_gpu_set(decode_placement);
    let unique_gpus = prefill_gpus
        .union(&decode_gpus)
        .copied()
        .collect::<BTreeSet<_>>();
    let prefill_nodes = gpu_node_set(&prefill_gpus);
    let decode_nodes = gpu_node_set(&decode_gpus);

    let aggregate = serving_hardware_capacity(cluster, model, &unique_gpus);
    let prefill = serving_hardware_capacity(cluster, model, &prefill_gpus);
    let decode = serving_hardware_capacity(cluster, model, &decode_gpus);

    ServingHardwareFootprint {
        unique_node_count: aggregate.node_count,
        unique_gpu_count: aggregate.gpu_count,
        prefill_node_count: prefill.node_count,
        prefill_gpu_count: prefill.gpu_count,
        decode_node_count: decode.node_count,
        decode_gpu_count: decode.gpu_count,
        shared_node_count: prefill_nodes.intersection(&decode_nodes).count() as u32,
        shared_gpu_count: prefill_gpus.intersection(&decode_gpus).count() as u32,
        aggregate_hbm_gb: aggregate.hbm_gb,
        prefill_hbm_gb: prefill.hbm_gb,
        decode_hbm_gb: decode.hbm_gb,
        aggregate_hbm_bandwidth_gb_s: aggregate.hbm_bandwidth_gb_s,
        prefill_hbm_bandwidth_gb_s: prefill.hbm_bandwidth_gb_s,
        decode_hbm_bandwidth_gb_s: decode.hbm_bandwidth_gb_s,
        aggregate_peak_f16_tflops: aggregate.peak_f16_tflops,
        prefill_peak_f16_tflops: prefill.peak_f16_tflops,
        decode_peak_f16_tflops: decode.peak_f16_tflops,
        aggregate_peak_f8_tflops: aggregate.peak_f8_tflops,
        prefill_peak_f8_tflops: prefill.peak_f8_tflops,
        decode_peak_f8_tflops: decode.peak_f8_tflops,
        aggregate_gpu_types: serving_gpu_type_counts(cluster, &unique_gpus),
        prefill_gpu_types: serving_gpu_type_counts(cluster, &prefill_gpus),
        decode_gpu_types: serving_gpu_type_counts(cluster, &decode_gpus),
        aggregate_gpu_label_counts: serving_gpu_label_counts(cluster, &unique_gpus),
        prefill_gpu_label_counts: serving_gpu_label_counts(cluster, &prefill_gpus),
        decode_gpu_label_counts: serving_gpu_label_counts(cluster, &decode_gpus),
        aggregate_effective_peak_tflops: aggregate.effective_peak_tflops,
        prefill_effective_peak_tflops: prefill.effective_peak_tflops,
        decode_effective_peak_tflops: decode.effective_peak_tflops,
        throughput_tokens_per_s_per_gpu: safe_ratio(
            throughput_tokens_per_s,
            f64::from(aggregate.gpu_count),
        ),
        throughput_tokens_per_s_per_effective_peak_tflop: safe_ratio(
            throughput_tokens_per_s,
            aggregate.effective_peak_tflops,
        ),
        throughput_tokens_per_s_per_hbm_gb: safe_ratio(throughput_tokens_per_s, aggregate.hbm_gb),
    }
}

fn serving_gpu_type_counts(
    cluster: &Cluster,
    gpus: &BTreeSet<GpuAddr>,
) -> Vec<ServingGpuTypeCount> {
    let mut counts = BTreeMap::new();
    for addr in gpus {
        if let Some(profile) = cluster.gpu_profile(*addr) {
            *counts.entry(profile.label.to_string()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|(gpu, count)| ServingGpuTypeCount { gpu, count })
        .collect()
}

fn serving_gpu_label_counts(
    cluster: &Cluster,
    gpus: &BTreeSet<GpuAddr>,
) -> Vec<ServingGpuLabelCount> {
    let mut counts = BTreeMap::new();
    for addr in gpus {
        let Some(node) = cluster.node(addr.node_id) else {
            continue;
        };
        let Some(labels) = node.gpu_labels(addr.local_gpu_id) else {
            continue;
        };
        for label in labels {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|(label, count)| ServingGpuLabelCount { label, count })
        .collect()
}

fn serving_cost_estimate(
    cluster: &Cluster,
    prefill_placement: &RankPlacement,
    decode_placement: &RankPlacement,
    metrics: &ServingMetrics,
    cost_model: &ServingCostModel,
) -> ServingCostEstimate {
    let prefill_gpus = placement_gpu_set(prefill_placement);
    let decode_gpus = placement_gpu_set(decode_placement);
    let unique_gpus = prefill_gpus
        .union(&decode_gpus)
        .copied()
        .collect::<BTreeSet<_>>();
    let unique_nodes = gpu_node_set(&unique_gpus);
    let modeled_duration_s = (metrics.scheduled_makespan_s.is_finite()
        && metrics.scheduled_makespan_s > 0.0)
        .then_some(metrics.scheduled_makespan_s);
    let modeled_gpu_count = unique_gpus.len().min(u32::MAX as usize) as u32;
    let modeled_node_count = unique_nodes.len().min(u32::MAX as usize) as u32;
    let gpu_hours =
        modeled_duration_s.map(|duration_s| duration_s / 3600.0 * f64::from(modeled_gpu_count));
    let node_hours =
        modeled_duration_s.map(|duration_s| duration_s / 3600.0 * f64::from(modeled_node_count));

    let gpu_rate_sum = sum_gpu_rate(cluster, &unique_gpus, cost_model, |rate| rate.gpu_hour_usd);
    let gpu_hour_cost_usd = modeled_duration_s
        .zip(gpu_rate_sum)
        .map(|(duration_s, rate_sum)| duration_s / 3600.0 * rate_sum);
    let node_hour_cost_usd = node_hours
        .zip(cost_model.node_hour_usd)
        .map(|(hours, rate)| hours * rate);
    let gpu_power_watts = sum_gpu_rate(cluster, &unique_gpus, cost_model, |rate| rate.watts);
    let node_power_watts = cost_model
        .node_watts
        .map(|watts| watts * f64::from(modeled_node_count));
    let average_power_watts = sum_optional_values([gpu_power_watts, node_power_watts]);
    let energy_kwh = modeled_duration_s
        .zip(average_power_watts)
        .map(|(duration_s, watts)| watts / 1000.0 * duration_s / 3600.0);
    let energy_cost_usd = energy_kwh
        .zip(cost_model.kwh_usd)
        .map(|(kwh, rate)| kwh * rate);
    let total_cost_usd =
        sum_optional_values([gpu_hour_cost_usd, node_hour_cost_usd, energy_cost_usd]);
    let cost_per_1k_output_tokens_usd = total_cost_usd.and_then(|cost| {
        (metrics.decode_iterations > 0).then(|| cost / metrics.decode_iterations as f64 * 1000.0)
    });
    let cost_per_1k_requests_usd = total_cost_usd.and_then(|cost| {
        (metrics.completed_requests > 0)
            .then(|| cost / f64::from(metrics.completed_requests) * 1000.0)
    });

    ServingCostEstimate {
        modeled_duration_s,
        modeled_gpu_count,
        modeled_node_count,
        gpu_hours,
        node_hours,
        gpu_hour_cost_usd,
        node_hour_cost_usd,
        average_power_watts,
        energy_kwh,
        energy_cost_usd,
        total_cost_usd,
        cost_per_1k_output_tokens_usd,
        cost_per_1k_requests_usd,
    }
}

fn sum_gpu_rate<F>(
    cluster: &Cluster,
    gpus: &BTreeSet<GpuAddr>,
    cost_model: &ServingCostModel,
    selector: F,
) -> Option<f64>
where
    F: Fn(&ServingGpuCostRate) -> Option<f64>,
{
    let mut total = 0.0;
    for gpu in gpus {
        let profile = cluster.gpu_profile(*gpu)?;
        let value = cost_model
            .gpu_rates
            .iter()
            .find(|rate| gpu_label_matches(&rate.gpu_label, profile.label))
            .and_then(&selector)
            .or_else(|| {
                let default_rate = ServingGpuCostRate {
                    gpu_label: "default".to_string(),
                    gpu_hour_usd: cost_model.default_gpu_hour_usd,
                    watts: cost_model.default_gpu_watts,
                };
                selector(&default_rate)
            })?;
        total += value;
    }
    Some(total)
}

fn gpu_label_matches(configured: &str, profile_label: &str) -> bool {
    normalize_hardware_label(configured) == normalize_hardware_label(profile_label)
}

fn normalize_hardware_label(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn sum_optional_values<const N: usize>(values: [Option<f64>; N]) -> Option<f64> {
    let mut saw_value = false;
    let mut total = 0.0;
    for value in values.into_iter().flatten() {
        saw_value = true;
        total += value;
    }
    saw_value.then_some(total)
}

fn placement_gpu_set(placement: &RankPlacement) -> BTreeSet<GpuAddr> {
    placement.rank_to_gpu.iter().copied().collect()
}

fn gpu_node_set(gpus: &BTreeSet<GpuAddr>) -> BTreeSet<NodeId> {
    gpus.iter().map(|gpu| gpu.node_id).collect()
}

fn serving_hardware_capacity(
    cluster: &Cluster,
    model: &ModelSpec,
    gpus: &BTreeSet<GpuAddr>,
) -> ServingHardwareCapacity {
    let mut capacity = ServingHardwareCapacity {
        node_count: gpu_node_set(gpus).len() as u32,
        gpu_count: gpus.len() as u32,
        peak_f8_tflops: Some(0.0),
        ..ServingHardwareCapacity::default()
    };

    for gpu in gpus {
        let Some(profile) = cluster.gpu_profile(*gpu) else {
            capacity.peak_f8_tflops = None;
            continue;
        };
        capacity.hbm_gb += profile.hbm_size.as_gigabytes();
        capacity.hbm_bandwidth_gb_s += profile.hbm_bandwidth.as_gigabytes_per_sec();
        capacity.peak_f16_tflops += profile.peak_f16_flops;
        capacity.effective_peak_tflops += effective_gpu_peak_tflops(&profile, model.dtype);
        match (capacity.peak_f8_tflops, profile.peak_f8_flops) {
            (Some(total), Some(peak_f8_tflops)) => {
                capacity.peak_f8_tflops = Some(total + peak_f8_tflops);
            }
            _ => {
                capacity.peak_f8_tflops = None;
            }
        }
    }

    capacity
}

fn effective_gpu_peak_tflops(profile: &crate::types::gpu::GpuProfile, dtype: DType) -> f64 {
    match dtype {
        DType::Fp8 => profile.peak_f8_flops.unwrap_or(0.0),
        DType::Int8 => profile.peak_f8_flops.unwrap_or(profile.peak_f16_flops),
        DType::Fp16 | DType::Bf16 => profile.peak_f16_flops,
    }
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator.is_finite() && denominator.is_finite() && denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn serving_candidate_id(
    deployment_mode: ServingDeploymentMode,
    pool: &ResolvedServingPool,
    prefill_config: &ParallelismConfig,
    decode_config: &ParallelismConfig,
) -> String {
    format!(
        "serving:mode-{}:pool-{}:pre-n{}:dec-n{}:pre-{}:dec-{}",
        candidate_segment(deployment_mode.as_str()),
        candidate_segment(&serving_pool_label(
            pool.label.as_deref(),
            &pool.prefill_nodes,
            &pool.decode_nodes,
            &pool.prefill_gpu_labels,
            &pool.decode_gpu_labels,
        )),
        sorted_u32_list(&pool.prefill_nodes),
        sorted_u32_list(&pool.decode_nodes),
        parallelism_config_id(prefill_config),
        parallelism_config_id(decode_config)
    )
}

fn serving_pool_label(
    label: Option<&str>,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_gpu_labels: &[String],
    decode_gpu_labels: &[String],
) -> String {
    label.map(str::to_string).unwrap_or_else(|| {
        format!(
            "p{}{}->d{}{}",
            sorted_u32_list(prefill_nodes),
            gpu_label_suffix(prefill_gpu_labels),
            sorted_u32_list(decode_nodes),
            gpu_label_suffix(decode_gpu_labels),
        )
    })
}

fn gpu_label_suffix(labels: &[String]) -> String {
    if labels.is_empty() {
        String::new()
    } else {
        format!(":gpu_labels[{}]", labels.join("|"))
    }
}

fn parallelism_config_id(config: &ParallelismConfig) -> String {
    format!(
        "tp{}-pp{}-ep{}-dp{}",
        config.tensor_ranks, config.pipeline_ranks, config.expert_ranks, config.data_ranks
    )
}

fn sorted_u32_list(values: &[u32]) -> String {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn candidate_segment(value: &str) -> String {
    let mut segment = String::with_capacity(value.len());
    let mut previous_was_separator = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            segment.push(ch.to_ascii_lowercase());
            previous_was_separator = false;
        } else if !previous_was_separator {
            segment.push('-');
            previous_was_separator = true;
        }
    }
    let segment = segment.trim_matches('-');
    if segment.is_empty() {
        "unnamed".to_string()
    } else {
        segment.to_string()
    }
}

const SERVING_BOTTLENECK_SUMMARY_LIMIT: usize = 12;

fn serving_bottleneck_summary(
    rejections: &[ServingRejection],
    topology_bottlenecks: &[ServingTopologyBottleneckObservation],
    memory_pressure: &[ServingMemoryPressureObservation],
    resource_utilization: &[ResourceUtilization],
    phase_resource_utilization: &[ServingPhaseResourceUtilization],
    objective_bottlenecks: &[ServingBottleneckSummary],
) -> Vec<ServingBottleneckSummary> {
    let mut summaries = Vec::new();

    summaries.extend(rejections.iter().map(|rejection| ServingBottleneckSummary {
        source: "rejection".to_string(),
        phase: rejection.phase.clone(),
        category: rejection.category.clone(),
        resource: rejection.resource.clone(),
        code: rejection.code.clone(),
        severity: "critical".to_string(),
        observed: rejection.observed,
        limit: rejection.limit,
        unit: rejection.unit.clone(),
        message: rejection.message.clone(),
        remediation: rejection.remediation.clone(),
    }));

    summaries.extend(
        topology_bottlenecks
            .iter()
            .map(|bottleneck| ServingBottleneckSummary {
                source: "topology".to_string(),
                phase: bottleneck.phase.clone(),
                category: bottleneck.category.clone(),
                resource: bottleneck.resource.clone(),
                code: bottleneck.code.clone(),
                severity: bottleneck.severity.clone(),
                observed: bottleneck.observed,
                limit: bottleneck.limit,
                unit: bottleneck.unit.clone(),
                message: bottleneck.message.clone(),
                remediation: bottleneck.remediation.clone(),
            }),
    );
    summaries.extend(objective_bottlenecks.iter().cloned());

    if let Some(peak) = peak_memory_pressure_observation(memory_pressure) {
        let dominant = peak
            .dominant_component
            .map(|component| component.name)
            .unwrap_or("unknown");
        summaries.push(ServingBottleneckSummary {
            source: "memory_pressure".to_string(),
            phase: peak.phase.clone(),
            category: "memory".to_string(),
            resource: peak
                .limiting_gpu
                .map(|gpu| format!("gpu:{}:{}", gpu.node_id, gpu.local_gpu_id))
                .unwrap_or_else(|| "gpu_hbm".to_string()),
            code: "peak_memory_pressure".to_string(),
            severity: pressure_severity(peak.capacity_used_fraction).to_string(),
            observed: Some(peak.capacity_used_fraction),
            limit: Some(1.0),
            unit: Some("fraction".to_string()),
            message: format!(
                "{} memory pressure reached {:.1}% of limiting HBM; dominant component is {dominant}",
                peak.phase,
                peak.capacity_used_fraction * 100.0
            ),
            remediation: Some(
                "reduce batch/sequence pressure, add GPUs, change prefill/decode placement, or apply a max_memory_pressure_fraction constraint"
                    .to_string(),
            ),
        });
    }

    if let Some(resource) = resource_utilization
        .iter()
        .filter(|resource| resource.utilization.is_finite() && resource.utilization > 0.0)
        .max_by(|left, right| {
            left.utilization
                .total_cmp(&right.utilization)
                .then_with(|| left.busy_s.total_cmp(&right.busy_s))
        })
    {
        summaries.push(ServingBottleneckSummary {
            source: "utilization".to_string(),
            phase: "all".to_string(),
            category: "scheduler".to_string(),
            resource: resource.resource.clone(),
            code: "hot_scheduled_resource".to_string(),
            severity: utilization_severity(resource.utilization).to_string(),
            observed: Some(resource.utilization),
            limit: Some(0.85),
            unit: Some("utilization".to_string()),
            message: format!(
                "scheduled resource '{}' reached {:.1}% utilization across {} operations",
                resource.resource,
                resource.utilization * 100.0,
                resource.operation_count
            ),
            remediation: Some(
                "spread work across additional workers/resources, reduce offered load, or choose a less contended placement"
                    .to_string(),
            ),
        });
    }

    let mut phase_resources = phase_resource_utilization
        .iter()
        .filter(|resource| resource.utilization.is_finite() && resource.utilization > 0.0)
        .collect::<Vec<_>>();
    phase_resources.sort_by(|left, right| {
        right
            .utilization
            .total_cmp(&left.utilization)
            .then_with(|| right.busy_s.total_cmp(&left.busy_s))
            .then_with(|| left.phase.cmp(&right.phase))
            .then_with(|| left.resource.cmp(&right.resource))
    });
    summaries.extend(phase_resources.into_iter().take(4).map(|resource| {
        ServingBottleneckSummary {
            source: "phase_utilization".to_string(),
            phase: resource.phase.clone(),
            category: resource.resource_kind.clone(),
            resource: resource.resource.clone(),
            code: "hot_phase_resource".to_string(),
            severity: utilization_severity(resource.utilization).to_string(),
            observed: Some(resource.utilization),
            limit: Some(0.85),
            unit: Some("utilization".to_string()),
            message: format!(
                "{} {} resource '{}' reached {:.1}% utilization across {} operations",
                resource.phase,
                resource.resource_kind,
                resource.resource,
                resource.utilization * 100.0,
                resource.operation_count
            ),
            remediation: Some(
                "rebalance routing, increase parallelism or worker slots, or choose resources with lower phase contention"
                    .to_string(),
            ),
        }
    }));

    summaries.sort_by(compare_bottleneck_summary);
    summaries.truncate(SERVING_BOTTLENECK_SUMMARY_LIMIT);
    summaries
}

struct ServingObjectiveBaseTerm {
    metric: &'static str,
    direction: &'static str,
    unit: &'static str,
    value: f64,
    score: f64,
}

#[allow(clippy::too_many_arguments)]
fn serving_objective_bottleneck_summaries(
    objective: ServingObjective,
    metrics: &ServingMetrics,
    cost_estimate: &ServingCostEstimate,
    memory_pressure: &[ServingMemoryPressureObservation],
    prefill_memory: &ServingMemoryHeadroom,
    decode_memory: &ServingMemoryHeadroom,
    slo_miss_penalty_score: f64,
    service_backpressure_penalty_score: f64,
    topology_risk_penalty_score: f64,
) -> Vec<ServingBottleneckSummary> {
    let base = serving_objective_base_term(
        objective,
        metrics,
        cost_estimate,
        memory_pressure,
        prefill_memory,
        decode_memory,
    );
    let mut summaries = vec![ServingBottleneckSummary {
        source: "objective".to_string(),
        phase: "all".to_string(),
        category: "objective".to_string(),
        resource: "base_objective".to_string(),
        code: if base.score.is_finite() {
            "objective_base_metric".to_string()
        } else {
            "objective_base_metric_unavailable".to_string()
        },
        severity: if base.score.is_finite() {
            "info".to_string()
        } else {
            "warning".to_string()
        },
        observed: base.value.is_finite().then_some(base.value),
        limit: None,
        unit: Some(base.unit.to_string()),
        message: if base.score.is_finite() {
            format!(
                "selected objective '{}' uses {} base metric '{}' with value {:.6} {} and lower-is-better score {:.6}",
                objective.as_str(),
                base.direction,
                base.metric,
                base.value,
                base.unit,
                base.score
            )
        } else {
            format!(
                "selected objective '{}' uses base metric '{}' but the metric value is unavailable",
                objective.as_str(),
                base.metric
            )
        },
        remediation: Some(
            "inspect the objective breakdown, metric ceilings, and candidate metric CSV before comparing this candidate against alternatives"
                .to_string(),
        ),
    }];

    push_objective_penalty_bottleneck(
        &mut summaries,
        "slo_miss_penalty",
        "objective_slo_miss_penalty",
        slo_miss_penalty_score,
        "SLO or deadline miss penalties contribute to this candidate's objective score.",
        "relax SLO policy, adjust traffic classes, increase capacity, or optimize the selected latency objective",
    );
    push_objective_penalty_bottleneck(
        &mut summaries,
        "service_backpressure_penalty",
        "objective_service_backpressure_penalty",
        service_backpressure_penalty_score,
        "Service backpressure penalties contribute to this candidate's objective score.",
        "increase worker slots, reduce offered load, tune queue caps, or select a pool with lower service pressure",
    );
    push_objective_penalty_bottleneck(
        &mut summaries,
        "topology_risk_penalty",
        "objective_topology_risk_penalty",
        topology_risk_penalty_score,
        "Topology risk penalties contribute to this candidate's objective score.",
        "select pools with better route coverage, spread across topology domains, or adjust topology_risk_penalty_weight",
    );

    summaries
}

fn serving_objective_base_term(
    objective: ServingObjective,
    metrics: &ServingMetrics,
    cost_estimate: &ServingCostEstimate,
    memory_pressure: &[ServingMemoryPressureObservation],
    prefill_memory: &ServingMemoryHeadroom,
    decode_memory: &ServingMemoryHeadroom,
) -> ServingObjectiveBaseTerm {
    match objective {
        ServingObjective::MinimizeE2el => ServingObjectiveBaseTerm {
            metric: "e2el_s",
            direction: "minimize",
            unit: "seconds",
            value: metrics.e2el_s,
            score: metrics.e2el_s,
        },
        ServingObjective::MinimizeTtft => ServingObjectiveBaseTerm {
            metric: "ttft_s",
            direction: "minimize",
            unit: "seconds",
            value: metrics.ttft_s,
            score: metrics.ttft_s,
        },
        ServingObjective::MinimizeTpot => ServingObjectiveBaseTerm {
            metric: "tpot_s",
            direction: "minimize",
            unit: "seconds_per_output_token",
            value: metrics.tpot_s,
            score: metrics.tpot_s,
        },
        ServingObjective::MaximizeThroughput => ServingObjectiveBaseTerm {
            metric: "throughput_tokens_per_s",
            direction: "maximize",
            unit: "tokens_per_second",
            value: metrics.throughput_tokens_per_s,
            score: -metrics.throughput_tokens_per_s,
        },
        ServingObjective::MinimizeSloMissRate => {
            let value = slo_miss_score(metrics);
            ServingObjectiveBaseTerm {
                metric: "aggregate_slo_miss_rate",
                direction: "minimize",
                unit: "fraction",
                value,
                score: value,
            }
        }
        ServingObjective::MinimizeMemoryPressure => {
            let value = serving_peak_memory_pressure_fraction_from_parts(
                memory_pressure,
                prefill_memory,
                decode_memory,
            );
            ServingObjectiveBaseTerm {
                metric: "memory_pressure_peak_fraction",
                direction: "minimize",
                unit: "fraction",
                value,
                score: value,
            }
        }
        ServingObjective::MinimizeCost => {
            let value = optional_finite_or_infinity(cost_estimate.total_cost_usd);
            ServingObjectiveBaseTerm {
                metric: "total_cost_usd",
                direction: "minimize",
                unit: "usd",
                value,
                score: value,
            }
        }
        ServingObjective::MinimizeEnergy => {
            let value = optional_finite_or_infinity(cost_estimate.energy_kwh);
            ServingObjectiveBaseTerm {
                metric: "energy_kwh",
                direction: "minimize",
                unit: "kwh",
                value,
                score: value,
            }
        }
        ServingObjective::MinimizePower => {
            let value = optional_finite_or_infinity(cost_estimate.average_power_watts);
            ServingObjectiveBaseTerm {
                metric: "average_power_watts",
                direction: "minimize",
                unit: "watts",
                value,
                score: value,
            }
        }
    }
}

fn push_objective_penalty_bottleneck(
    summaries: &mut Vec<ServingBottleneckSummary>,
    resource: &'static str,
    code: &'static str,
    score: f64,
    message: &'static str,
    remediation: &'static str,
) {
    if !score.is_finite() || score <= 0.0 {
        return;
    }
    summaries.push(ServingBottleneckSummary {
        source: "objective".to_string(),
        phase: "all".to_string(),
        category: "objective".to_string(),
        resource: resource.to_string(),
        code: code.to_string(),
        severity: "warning".to_string(),
        observed: Some(score),
        limit: Some(0.0),
        unit: Some("score".to_string()),
        message: format!("{message} penalty_score={score:.6}"),
        remediation: Some(remediation.to_string()),
    });
}

fn peak_memory_pressure_observation(
    observations: &[ServingMemoryPressureObservation],
) -> Option<&ServingMemoryPressureObservation> {
    observations
        .iter()
        .filter(|observation| observation.capacity_used_fraction.is_finite())
        .max_by(|left, right| {
            left.capacity_used_fraction
                .total_cmp(&right.capacity_used_fraction)
                .then_with(|| left.duration_s.total_cmp(&right.duration_s))
        })
}

fn pressure_severity(fraction: f64) -> &'static str {
    if fraction >= 1.0 {
        "critical"
    } else if fraction >= 0.85 {
        "warning"
    } else {
        "info"
    }
}

fn utilization_severity(utilization: f64) -> &'static str {
    if utilization >= 0.98 {
        "critical"
    } else if utilization >= 0.85 {
        "warning"
    } else {
        "info"
    }
}

fn compare_bottleneck_summary(
    left: &ServingBottleneckSummary,
    right: &ServingBottleneckSummary,
) -> std::cmp::Ordering {
    severity_rank(&right.severity)
        .cmp(&severity_rank(&left.severity))
        .then_with(|| bottleneck_magnitude(right).total_cmp(&bottleneck_magnitude(left)))
        .then_with(|| left.source.cmp(&right.source))
        .then_with(|| left.phase.cmp(&right.phase))
        .then_with(|| left.code.cmp(&right.code))
        .then_with(|| left.resource.cmp(&right.resource))
}

fn severity_rank(severity: &str) -> u8 {
    match severity {
        "critical" => 3,
        "warning" => 2,
        "info" => 1,
        _ => 0,
    }
}

fn bottleneck_magnitude(summary: &ServingBottleneckSummary) -> f64 {
    match (summary.observed, summary.limit) {
        (Some(observed), Some(limit)) if limit.is_finite() && limit > 0.0 => observed / limit,
        (Some(observed), _) if observed.is_finite() => observed.abs(),
        _ => 0.0,
    }
}

#[derive(Copy, Clone, Debug)]
struct ServingParetoPoint {
    ttft_s: f64,
    tpot_s: f64,
    itl_s: f64,
    e2el_s: f64,
    throughput_tokens_per_s: f64,
    memory_pressure_fraction: f64,
    unique_gpu_count: f64,
    total_cost_usd: f64,
    energy_kwh: f64,
    average_power_watts: f64,
}

fn search_deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

fn annotate_serving_pareto(results: &mut [ScoredServingConfig]) {
    let dimensions = serving_pareto_dimensions();
    for result in results.iter_mut() {
        result.pareto = ServingParetoFrontier {
            dimensions: dimensions.clone(),
            ..ServingParetoFrontier::default()
        };
    }

    let points = results
        .iter()
        .enumerate()
        .filter(|(_, result)| result.feasible)
        .map(|(idx, result)| {
            (
                idx,
                ServingParetoPoint {
                    ttft_s: finite_or_infinity(result.metrics.ttft_s),
                    tpot_s: finite_or_infinity(result.metrics.tpot_s),
                    itl_s: finite_or_infinity(result.metrics.itl_s),
                    e2el_s: finite_or_infinity(result.metrics.e2el_s),
                    throughput_tokens_per_s: finite_or_negative_infinity(
                        result.metrics.throughput_tokens_per_s,
                    ),
                    memory_pressure_fraction: finite_or_infinity(
                        serving_peak_memory_pressure_fraction(result),
                    ),
                    unique_gpu_count: f64::from(result.hardware_footprint.unique_gpu_count),
                    total_cost_usd: optional_finite_or_infinity(
                        result.cost_estimate.total_cost_usd,
                    ),
                    energy_kwh: optional_finite_or_infinity(result.cost_estimate.energy_kwh),
                    average_power_watts: optional_finite_or_infinity(
                        result.cost_estimate.average_power_watts,
                    ),
                },
            )
        })
        .collect::<Vec<_>>();
    let point_by_idx = points.iter().copied().collect::<BTreeMap<_, _>>();
    let mut remaining = points.iter().map(|(idx, _)| *idx).collect::<BTreeSet<_>>();
    let mut rank = 1_u32;

    while !remaining.is_empty() {
        let front = remaining
            .iter()
            .copied()
            .filter(|candidate_idx| {
                let candidate = point_by_idx[candidate_idx];
                !remaining.iter().copied().any(|other_idx| {
                    other_idx != *candidate_idx
                        && pareto_dominates(point_by_idx[&other_idx], candidate)
                })
            })
            .collect::<Vec<_>>();

        if front.is_empty() {
            break;
        }

        for idx in &front {
            results[*idx].pareto.rank = Some(rank);
            results[*idx].pareto.is_frontier = rank == 1;
        }
        for idx in front {
            remaining.remove(&idx);
        }
        rank = rank.saturating_add(1);
    }

    let ranks = results
        .iter()
        .enumerate()
        .filter_map(|(idx, result)| result.pareto.rank.map(|rank| (idx, rank)))
        .collect::<BTreeMap<_, _>>();
    for idx in points.iter().map(|(idx, _)| *idx) {
        let Some(rank) = ranks.get(&idx).copied() else {
            continue;
        };
        let dominated_by = points
            .iter()
            .filter_map(|(other_idx, other)| {
                let other_rank = ranks.get(other_idx).copied()?;
                (other_rank < rank && pareto_dominates(*other, point_by_idx[&idx]))
                    .then(|| results[*other_idx].candidate_id.clone())
            })
            .take(4)
            .collect();
        results[idx].pareto.dominated_by = dominated_by;
    }
}

fn pareto_dominates(left: ServingParetoPoint, right: ServingParetoPoint) -> bool {
    let no_worse = left.ttft_s <= right.ttft_s
        && left.tpot_s <= right.tpot_s
        && left.itl_s <= right.itl_s
        && left.e2el_s <= right.e2el_s
        && left.throughput_tokens_per_s >= right.throughput_tokens_per_s
        && left.memory_pressure_fraction <= right.memory_pressure_fraction
        && left.unique_gpu_count <= right.unique_gpu_count
        && left.total_cost_usd <= right.total_cost_usd
        && left.energy_kwh <= right.energy_kwh
        && left.average_power_watts <= right.average_power_watts;
    let strictly_better = left.ttft_s < right.ttft_s
        || left.tpot_s < right.tpot_s
        || left.itl_s < right.itl_s
        || left.e2el_s < right.e2el_s
        || left.throughput_tokens_per_s > right.throughput_tokens_per_s
        || left.memory_pressure_fraction < right.memory_pressure_fraction
        || left.unique_gpu_count < right.unique_gpu_count
        || left.total_cost_usd < right.total_cost_usd
        || left.energy_kwh < right.energy_kwh
        || left.average_power_watts < right.average_power_watts;
    no_worse && strictly_better
}

fn serving_pareto_dimensions() -> Vec<ServingParetoDimension> {
    [
        ("ttft_s", "minimize", "seconds"),
        ("tpot_s", "minimize", "seconds_per_token"),
        ("itl_s", "minimize", "seconds"),
        ("e2el_s", "minimize", "seconds"),
        ("throughput_tokens_per_s", "maximize", "tokens_per_second"),
        ("memory_pressure_fraction", "minimize", "fraction"),
        ("unique_gpu_count", "minimize", "gpus"),
        ("total_cost_usd", "minimize", "usd"),
        ("energy_kwh", "minimize", "kwh"),
        ("average_power_watts", "minimize", "watts"),
    ]
    .into_iter()
    .map(|(metric, direction, unit)| ServingParetoDimension {
        metric: metric.to_string(),
        direction: direction.to_string(),
        unit: unit.to_string(),
    })
    .collect()
}

fn finite_or_infinity(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::INFINITY
    }
}

fn finite_or_negative_infinity(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::NEG_INFINITY
    }
}

fn optional_finite_or_infinity(value: Option<f64>) -> f64 {
    value
        .filter(|value| value.is_finite())
        .unwrap_or(f64::INFINITY)
}

fn sort_serving_results(results: &mut [ScoredServingConfig], objective: ServingObjective) {
    results.sort_by(|a, b| {
        b.feasible
            .cmp(&a.feasible)
            .then_with(|| compare_serving_objective(a, b, objective))
            .then_with(|| {
                serving_peak_memory_pressure_fraction(a)
                    .total_cmp(&serving_peak_memory_pressure_fraction(b))
            })
            .then_with(|| a.metrics.e2el_s.total_cmp(&b.metrics.e2el_s))
            .then_with(|| a.metrics.tpot_s.total_cmp(&b.metrics.tpot_s))
            .then_with(|| a.metrics.ttft_s.total_cmp(&b.metrics.ttft_s))
            .then_with(|| {
                b.metrics
                    .throughput_tokens_per_s
                    .total_cmp(&a.metrics.throughput_tokens_per_s)
            })
            .then_with(|| {
                a.prefill_config
                    .total_ranks()
                    .cmp(&b.prefill_config.total_ranks())
            })
            .then_with(|| {
                a.decode_config
                    .total_ranks()
                    .cmp(&b.decode_config.total_ranks())
            })
            .then_with(|| a.prefill_nodes.cmp(&b.prefill_nodes))
            .then_with(|| a.decode_nodes.cmp(&b.decode_nodes))
            .then_with(|| a.pool_label.cmp(&b.pool_label))
    });
}

fn compare_serving_objective(
    a: &ScoredServingConfig,
    b: &ScoredServingConfig,
    objective: ServingObjective,
) -> std::cmp::Ordering {
    serving_objective_score(a, objective).total_cmp(&serving_objective_score(b, objective))
}

fn serving_objective_score(score: &ScoredServingConfig, objective: ServingObjective) -> f64 {
    let base_score = match objective {
        ServingObjective::MinimizeE2el => score.metrics.e2el_s,
        ServingObjective::MinimizeTtft => score.metrics.ttft_s,
        ServingObjective::MinimizeTpot => score.metrics.tpot_s,
        ServingObjective::MaximizeThroughput => -score.metrics.throughput_tokens_per_s,
        ServingObjective::MinimizeSloMissRate => slo_miss_score(&score.metrics),
        ServingObjective::MinimizeMemoryPressure => serving_peak_memory_pressure_fraction(score),
        ServingObjective::MinimizeCost => {
            optional_finite_or_infinity(score.cost_estimate.total_cost_usd)
        }
        ServingObjective::MinimizeEnergy => {
            optional_finite_or_infinity(score.cost_estimate.energy_kwh)
        }
        ServingObjective::MinimizePower => {
            optional_finite_or_infinity(score.cost_estimate.average_power_watts)
        }
    };
    base_score
        + score.slo_miss_penalty_score
        + score.service_backpressure_penalty_score
        + score.topology_risk_penalty_score
}

fn serving_peak_memory_pressure_fraction(score: &ScoredServingConfig) -> f64 {
    serving_peak_memory_pressure_fraction_from_parts(
        &score.memory_pressure,
        &score.prefill_memory,
        &score.decode_memory,
    )
}

fn serving_peak_memory_pressure_fraction_from_parts(
    memory_pressure: &[ServingMemoryPressureObservation],
    prefill_memory: &ServingMemoryHeadroom,
    decode_memory: &ServingMemoryHeadroom,
) -> f64 {
    let peak = memory_pressure
        .iter()
        .filter_map(|observation| {
            observation
                .capacity_used_fraction
                .is_finite()
                .then_some(observation.capacity_used_fraction)
        })
        .chain([
            prefill_memory.capacity_used_fraction(),
            decode_memory.capacity_used_fraction(),
        ])
        .filter(|fraction| fraction.is_finite())
        .fold(None, |peak: Option<f64>, fraction| {
            Some(peak.map_or(fraction, |peak| peak.max(fraction)))
        });
    peak.unwrap_or(f64::INFINITY)
}

fn service_backpressure_penalty_score(
    observations: &[ServingServiceObservation],
    weight: f64,
) -> f64 {
    if weight <= 0.0 || !weight.is_finite() {
        return 0.0;
    }
    let request_count = observations
        .iter()
        .map(|observation| u64::from(observation.request_count))
        .sum::<u64>();
    if request_count == 0 {
        return 0.0;
    }
    let backpressure_rejections = observations
        .iter()
        .map(|observation| u64::from(observation.backpressure_rejections))
        .sum::<u64>();
    weight * (backpressure_rejections as f64 / request_count as f64)
}

fn topology_risk_penalty_score(
    coverage: ServingRouteCoverage,
    topology_bottlenecks: &[ServingTopologyBottleneckObservation],
    weight: f64,
) -> f64 {
    if weight <= 0.0 || !weight.is_finite() {
        return 0.0;
    }
    let route_risk = if coverage.candidate_count == 0 || !coverage.fraction.is_finite() {
        0.0
    } else {
        1.0 - coverage.fraction.clamp(0.0, 1.0)
    };
    let domain_risk = topology_domain_risk_fraction(topology_bottlenecks);
    let bottleneck_risk = topology_bottleneck_risk_fraction(topology_bottlenecks);
    weight * route_risk.max(domain_risk).max(bottleneck_risk)
}

fn topology_domain_risk_fraction(
    topology_bottlenecks: &[ServingTopologyBottleneckObservation],
) -> f64 {
    topology_bottlenecks
        .iter()
        .filter_map(|bottleneck| match bottleneck.code.as_str() {
            "single_failure_domain_placement" => Some(1.0),
            "single_rack_placement" | "single_island_placement" => Some(0.5),
            _ => None,
        })
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn topology_bottleneck_risk_fraction(
    topology_bottlenecks: &[ServingTopologyBottleneckObservation],
) -> f64 {
    topology_bottlenecks
        .iter()
        .filter_map(topology_bottleneck_risk)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn topology_bottleneck_risk(bottleneck: &ServingTopologyBottleneckObservation) -> Option<f64> {
    match bottleneck.code.as_str() {
        "partial_route_coverage" => None,
        "single_failure_domain_placement" | "single_rack_placement" | "single_island_placement" => {
            None
        }
        "hot_kv_route_resource" => bottleneck
            .observed
            .filter(|observed| observed.is_finite())
            .map(|observed| observed.clamp(0.0, 1.0))
            .or_else(|| Some(severity_topology_risk(&bottleneck.severity))),
        "kv_route_resource_queueing" => Some(severity_topology_risk(&bottleneck.severity).max(0.5)),
        "single_rail_dependency" => Some(0.35),
        "host_staged_kv_path" | "host_staged_kv_path_disallowed" => Some(0.5),
        "cross_socket_kv_path" => Some(0.3),
        "slow_gpu_nic_kv_path" => Some(0.15),
        "unrailed_inter_node_routes" | "kv_route_rail_metadata_missing" => Some(0.1),
        _ => {
            let risk = severity_topology_risk(&bottleneck.severity);
            (risk > 0.0).then_some(risk)
        }
    }
}

fn severity_topology_risk(severity: &str) -> f64 {
    match severity.trim().to_ascii_lowercase().as_str() {
        "critical" | "error" => 1.0,
        "warning" | "warn" => 0.5,
        "info" => 0.1,
        _ => 0.0,
    }
}

fn slo_miss_score(metrics: &ServingMetrics) -> f64 {
    [
        metrics.ttft_slo_miss_rate,
        metrics.tpot_slo_miss_rate,
        metrics.itl_slo_miss_rate,
        metrics.e2el_slo_miss_rate,
        metrics.deadline_miss_rate,
    ]
    .into_iter()
    .filter(|value| value.is_finite())
    .sum()
}

fn slo_miss_penalty_components_from_metrics(
    metrics: &ServingMetrics,
    weights: ServingSloMissPenaltyWeights,
) -> ServingSloMissPenaltyComponents {
    let ttft = finite_or_zero(metrics.ttft_slo_miss_rate) * (weights.aggregate + weights.ttft);
    let tpot = finite_or_zero(metrics.tpot_slo_miss_rate) * (weights.aggregate + weights.tpot);
    let itl = finite_or_zero(metrics.itl_slo_miss_rate) * (weights.aggregate + weights.itl);
    let e2el = finite_or_zero(metrics.e2el_slo_miss_rate) * (weights.aggregate + weights.e2el);
    let deadline =
        finite_or_zero(metrics.deadline_miss_rate) * (weights.aggregate + weights.deadline);
    ServingSloMissPenaltyComponents {
        ttft,
        tpot,
        itl,
        e2el,
        deadline,
        total: ttft + tpot + itl + e2el + deadline,
    }
}

fn slo_miss_penalty_components_from_breakdown(
    breakdown: &ServingMetricBreakdown,
    weights: ServingSloMissPenaltyWeights,
) -> ServingSloMissPenaltyComponents {
    let ttft = finite_or_zero(breakdown.ttft_slo_miss_rate) * (weights.aggregate + weights.ttft);
    let tpot = finite_or_zero(breakdown.tpot_slo_miss_rate) * (weights.aggregate + weights.tpot);
    let itl = finite_or_zero(breakdown.itl_slo_miss_rate) * (weights.aggregate + weights.itl);
    let e2el = finite_or_zero(breakdown.e2el_slo_miss_rate) * (weights.aggregate + weights.e2el);
    let deadline =
        finite_or_zero(breakdown.deadline_miss_rate) * (weights.aggregate + weights.deadline);
    ServingSloMissPenaltyComponents {
        ttft,
        tpot,
        itl,
        e2el,
        deadline,
        total: ttft + tpot + itl + e2el + deadline,
    }
}

fn traffic_class_slo_miss_penalties(
    traffic: &ServingTraffic,
    breakdowns: &[ServingMetricBreakdown],
) -> Vec<ServingTrafficClassSloMissPenalty> {
    traffic
        .traffic_classes
        .iter()
        .filter(|class| class.slo_miss_penalty_weights.any_nonzero())
        .filter_map(|class| {
            let breakdown = breakdowns.iter().find(|breakdown| {
                (breakdown.group == class.group && breakdown.key == class.key)
                    || (breakdown.group == "traffic_class" && breakdown.key == class.name)
            })?;
            let components = slo_miss_penalty_components_from_breakdown(
                breakdown,
                class.slo_miss_penalty_weights,
            );
            Some(ServingTrafficClassSloMissPenalty {
                name: class.name.clone(),
                group: class.group.clone(),
                key: class.key.clone(),
                weights: class.slo_miss_penalty_weights,
                components,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ServingPoolSearchResult {
    candidates: Vec<ServingPoolCandidate>,
    summary: ServingPoolSearchSummary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ServingPoolSearchError {
    reason: String,
    summary: ServingPoolSearchSummary,
}

fn generate_pool_search_candidates_with_summary(
    cluster: &Cluster,
    search: &ServingPoolSearch,
    deployment_mode: ServingDeploymentMode,
) -> Result<ServingPoolSearchResult, ServingPoolSearchError> {
    let mut candidates = Vec::new();
    let mut summary = ServingPoolSearchSummary {
        max_candidates: search.max_candidates.max(1),
        ..ServingPoolSearchSummary::default()
    };

    for prefill_group in &search.prefill_groups {
        let prefill_group_nodes = sorted_group_nodes(cluster, prefill_group).map_err(|reason| {
            ServingPoolSearchError {
                reason,
                summary: summary.clone(),
            }
        })?;
        let prefill_group_node_count = prefill_group_nodes.len();
        let prefill_node_filter_nodes = nodes_matching_pool_node_filter(
            cluster,
            &prefill_group_nodes,
            &search.prefill_node_filter,
        );
        let prefill_node_filter_node_count = prefill_node_filter_nodes.len();
        let prefill_group_nodes = nodes_with_gpu_labels(
            cluster,
            &prefill_node_filter_nodes,
            &search.prefill_gpu_labels,
        );
        let prefill_gpu_filter_node_count = prefill_group_nodes.len();
        let prefill_counts =
            effective_counts(&search.prefill_node_counts, prefill_group_nodes.len());
        for decode_group in &search.decode_groups {
            let decode_group_nodes =
                sorted_group_nodes(cluster, decode_group).map_err(|reason| {
                    ServingPoolSearchError {
                        reason,
                        summary: summary.clone(),
                    }
                })?;
            let decode_group_node_count = decode_group_nodes.len();
            let decode_node_filter_nodes = nodes_matching_pool_node_filter(
                cluster,
                &decode_group_nodes,
                &search.decode_node_filter,
            );
            let decode_node_filter_node_count = decode_node_filter_nodes.len();
            let decode_group_nodes = nodes_with_gpu_labels(
                cluster,
                &decode_node_filter_nodes,
                &search.decode_gpu_labels,
            );
            let decode_gpu_filter_node_count = decode_group_nodes.len();
            let decode_counts =
                effective_counts(&search.decode_node_counts, decode_group_nodes.len());
            let mut group_summary = ServingPoolSearchGroupSummary {
                prefill_group: prefill_group.clone(),
                decode_group: decode_group.clone(),
                prefill_group_node_count,
                prefill_node_filter_node_count,
                prefill_gpu_filter_node_count,
                decode_group_node_count,
                decode_node_filter_node_count,
                decode_gpu_filter_node_count,
                prefill_node_counts: prefill_counts.clone(),
                decode_node_counts: decode_counts.clone(),
                ..ServingPoolSearchGroupSummary::default()
            };

            for &prefill_count in &prefill_counts {
                for prefill_nodes in combinations(&prefill_group_nodes, prefill_count as usize) {
                    for &decode_count in &decode_counts {
                        for decode_nodes in combinations(&decode_group_nodes, decode_count as usize)
                        {
                            group_summary.considered_candidate_count += 1;
                            if !search.allow_overlap
                                && deployment_mode != ServingDeploymentMode::Colocated
                                && deployment_mode != ServingDeploymentMode::PartiallyDisaggregated
                                && overlaps_nodes(&prefill_nodes, &decode_nodes)
                            {
                                group_summary.rejected_overlap_count += 1;
                                continue;
                            }
                            if !deployment_mode.accepts_pool(&prefill_nodes, &decode_nodes) {
                                group_summary.rejected_mode_count += 1;
                                continue;
                            }
                            if !pool_search_candidate_satisfies_domain_spread(
                                cluster,
                                search,
                                &prefill_nodes,
                                &decode_nodes,
                            ) {
                                group_summary.rejected_domain_spread_count += 1;
                                continue;
                            }
                            let effective_mode = ServingDeploymentMode::effective_for_pool(
                                &prefill_nodes,
                                &decode_nodes,
                            );
                            let candidate = ServingPoolCandidate {
                                label: Some(format!(
                                    "auto:{}{}->{}{}",
                                    prefill_group,
                                    node_suffix(&prefill_nodes),
                                    decode_group,
                                    node_suffix(&decode_nodes)
                                )),
                                prefill_nodes: prefill_nodes.clone(),
                                decode_nodes,
                                prefill_groups: Vec::new(),
                                decode_groups: Vec::new(),
                                prefill_node_filter: ServingPoolNodeFilter::default(),
                                decode_node_filter: ServingPoolNodeFilter::default(),
                                domain_spread: search.domain_spread.clone(),
                                prefill_gpu_labels: search.prefill_gpu_labels.clone(),
                                decode_gpu_labels: search.decode_gpu_labels.clone(),
                            };
                            let previous_len = candidates.len();
                            push_unique_pool_candidate(&mut candidates, candidate);
                            if candidates.len() > previous_len {
                                group_summary.generated_candidate_count += 1;
                                match effective_mode {
                                    ServingDeploymentMode::Flexible => {}
                                    ServingDeploymentMode::Colocated => {
                                        group_summary.generated_colocated_count += 1;
                                    }
                                    ServingDeploymentMode::PartiallyDisaggregated => {
                                        group_summary.generated_partially_disaggregated_count += 1;
                                    }
                                    ServingDeploymentMode::FullyDisaggregated => {
                                        group_summary.generated_fully_disaggregated_count += 1;
                                    }
                                }
                            } else {
                                group_summary.duplicate_candidate_count += 1;
                            }
                            if candidates.len() >= search.max_candidates.max(1) {
                                summary.generated_candidate_count = candidates.len();
                                summary.truncated = true;
                                summary.groups.push(group_summary);
                                return Ok(ServingPoolSearchResult {
                                    candidates,
                                    summary,
                                });
                            }
                        }
                    }
                }
            }
            summary.groups.push(group_summary);
        }
    }

    if candidates.is_empty() {
        let reason = if deployment_mode == ServingDeploymentMode::Flexible {
            "serving.pool_search generated no valid pool candidates; check groups, counts, node topology filters, GPU labels, overlap policy, and topology-domain spread constraints".to_string()
        } else {
            format!(
                "serving.pool_search generated no valid pool candidates compatible with serving.mode '{}'; check groups, counts, node topology filters, GPU labels, overlap policy, and topology-domain spread constraints",
                deployment_mode.as_str()
            )
        };
        Err(ServingPoolSearchError { reason, summary })
    } else {
        summary.generated_candidate_count = candidates.len();
        Ok(ServingPoolSearchResult {
            candidates,
            summary,
        })
    }
}

fn pool_search_candidate_satisfies_domain_spread(
    cluster: &Cluster,
    search: &ServingPoolSearch,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
) -> bool {
    pool_nodes_satisfy_domain_spread(cluster, &search.domain_spread, prefill_nodes, decode_nodes)
}

fn pool_nodes_satisfy_domain_spread(
    cluster: &Cluster,
    domain_spread: &ServingPoolDomainSpread,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
) -> bool {
    pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_racks,
        |node| node.topology.rack.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_islands,
        |node| node.topology.island.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        prefill_nodes,
        domain_spread.min_prefill_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    ) && pool_nodes_meet_min_domain_count(
        cluster,
        decode_nodes,
        domain_spread.min_decode_failure_domains,
        |node| node.topology.failure_domain.as_deref(),
    )
}

fn nodes_matching_pool_node_filter(
    cluster: &Cluster,
    node_ids: &[NodeId],
    filter: &ServingPoolNodeFilter,
) -> Vec<NodeId> {
    node_ids
        .iter()
        .copied()
        .filter(|node_id| {
            cluster
                .node(*node_id)
                .is_some_and(|node| node_matches_pool_node_filter(node, filter))
        })
        .collect()
}

fn node_matches_pool_node_filter(node: &Node, filter: &ServingPoolNodeFilter) -> bool {
    (filter.node_labels.is_empty()
        || filter
            .node_labels
            .iter()
            .any(|label| node.topology.labels.contains(label)))
        && (filter.racks.is_empty()
            || node
                .topology
                .rack
                .as_ref()
                .is_some_and(|rack| filter.racks.contains(rack)))
        && (filter.islands.is_empty()
            || node
                .topology
                .island
                .as_ref()
                .is_some_and(|island| filter.islands.contains(island)))
        && (filter.failure_domains.is_empty()
            || node
                .topology
                .failure_domain
                .as_ref()
                .is_some_and(|failure_domain| filter.failure_domains.contains(failure_domain)))
        && filter
            .exclude_node_labels
            .iter()
            .all(|label| !node.topology.labels.contains(label))
        && !node
            .topology
            .rack
            .as_ref()
            .is_some_and(|rack| filter.exclude_racks.contains(rack))
        && !node
            .topology
            .island
            .as_ref()
            .is_some_and(|island| filter.exclude_islands.contains(island))
        && !node
            .topology
            .failure_domain
            .as_ref()
            .is_some_and(|failure_domain| filter.exclude_failure_domains.contains(failure_domain))
}

fn pool_nodes_meet_min_domain_count(
    cluster: &Cluster,
    node_ids: &[NodeId],
    min_count: Option<u32>,
    domain: impl Fn(&Node) -> Option<&str>,
) -> bool {
    let Some(min_count) = min_count else {
        return true;
    };
    let domains = node_ids
        .iter()
        .filter_map(|node_id| cluster.node(*node_id))
        .filter_map(domain)
        .collect::<BTreeSet<_>>();
    domains.len() >= min_count as usize
}

fn sorted_group_nodes(cluster: &Cluster, group: &str) -> Result<Vec<NodeId>, String> {
    let mut nodes = cluster
        .node_group(group)
        .ok_or_else(|| format!("serving.pool_search references unknown node group '{group}'"))?
        .to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Err(format!(
            "serving.pool_search node group '{group}' contains no nodes"
        ));
    }
    Ok(nodes)
}

fn nodes_with_gpu_labels(
    cluster: &Cluster,
    node_ids: &[NodeId],
    gpu_labels: &[String],
) -> Vec<NodeId> {
    if gpu_labels.is_empty() {
        return node_ids.to_vec();
    }
    node_ids
        .iter()
        .copied()
        .filter(|node_id| {
            cluster.node(*node_id).is_some_and(|node| {
                node.gpus.keys().copied().any(|local_gpu_id| {
                    cluster.is_gpu_available(GpuAddr {
                        node_id: *node_id,
                        local_gpu_id,
                    }) && node
                        .gpu_labels(local_gpu_id)
                        .is_some_and(|labels| gpu_labels.iter().any(|label| labels.contains(label)))
                })
            })
        })
        .collect()
}

fn effective_counts(configured: &[u32], available: usize) -> Vec<u32> {
    let mut counts = if configured.is_empty() {
        vec![available as u32]
    } else {
        configured.to_vec()
    };
    counts.sort_unstable();
    counts.dedup();
    counts
        .into_iter()
        .filter(|count| *count > 0 && (*count as usize) <= available)
        .collect()
}

fn combinations(values: &[NodeId], count: usize) -> Vec<Vec<NodeId>> {
    if count == 0 || count > values.len() {
        return Vec::new();
    }
    if count == values.len() {
        return vec![values.to_vec()];
    }

    let mut results = Vec::new();
    let mut current = Vec::with_capacity(count);
    push_combinations(values, count, 0, &mut current, &mut results);
    results
}

fn push_combinations(
    values: &[NodeId],
    count: usize,
    start: usize,
    current: &mut Vec<NodeId>,
    results: &mut Vec<Vec<NodeId>>,
) {
    if current.len() == count {
        results.push(current.clone());
        return;
    }

    let needed = count - current.len();
    for idx in start..=values.len() - needed {
        current.push(values[idx]);
        push_combinations(values, count, idx + 1, current, results);
        current.pop();
    }
}

fn overlaps_nodes(left: &[NodeId], right: &[NodeId]) -> bool {
    left.iter().any(|node| right.contains(node))
}

fn node_suffix(nodes: &[NodeId]) -> String {
    format!(
        "[{}]",
        nodes
            .iter()
            .map(|node| node.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn node_list(nodes: &[NodeId]) -> String {
    nodes
        .iter()
        .map(|node| node.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn extend_unique_pool_candidates(
    candidates: &mut Vec<ServingPoolCandidate>,
    generated: Vec<ServingPoolCandidate>,
) {
    for candidate in generated {
        push_unique_pool_candidate(candidates, candidate);
    }
}

fn push_unique_pool_candidate(
    candidates: &mut Vec<ServingPoolCandidate>,
    candidate: ServingPoolCandidate,
) {
    if !candidates
        .iter()
        .any(|existing| same_pool_candidate(existing, &candidate))
    {
        candidates.push(candidate);
    }
}

fn same_pool_candidate(left: &ServingPoolCandidate, right: &ServingPoolCandidate) -> bool {
    same_u32s(&left.prefill_nodes, &right.prefill_nodes)
        && same_u32s(&left.decode_nodes, &right.decode_nodes)
        && same_strings(&left.prefill_groups, &right.prefill_groups)
        && same_strings(&left.decode_groups, &right.decode_groups)
        && same_pool_node_filter(&left.prefill_node_filter, &right.prefill_node_filter)
        && same_pool_node_filter(&left.decode_node_filter, &right.decode_node_filter)
        && left.domain_spread == right.domain_spread
        && same_strings(&left.prefill_gpu_labels, &right.prefill_gpu_labels)
        && same_strings(&left.decode_gpu_labels, &right.decode_gpu_labels)
}

fn same_pool_node_filter(left: &ServingPoolNodeFilter, right: &ServingPoolNodeFilter) -> bool {
    same_strings(&left.node_labels, &right.node_labels)
        && same_strings(&left.racks, &right.racks)
        && same_strings(&left.islands, &right.islands)
        && same_strings(&left.failure_domains, &right.failure_domains)
        && same_strings(&left.exclude_node_labels, &right.exclude_node_labels)
        && same_strings(&left.exclude_racks, &right.exclude_racks)
        && same_strings(&left.exclude_islands, &right.exclude_islands)
        && same_strings(
            &left.exclude_failure_domains,
            &right.exclude_failure_domains,
        )
}

fn same_u32s(left: &[u32], right: &[u32]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn same_strings(left: &[String], right: &[String]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort();
    right.sort();
    left == right
}

fn resolve_pool_candidate(
    cluster: &Cluster,
    candidate: &ServingPoolCandidate,
) -> Result<ResolvedServingPool, String> {
    let prefill_nodes = resolve_pool_nodes(
        cluster,
        "prefill_groups",
        &candidate.prefill_nodes,
        &candidate.prefill_groups,
    )?;
    let prefill_nodes =
        nodes_matching_pool_node_filter(cluster, &prefill_nodes, &candidate.prefill_node_filter);
    if prefill_nodes.is_empty() {
        return Err("serving pool prefill node topology filters matched no nodes".to_string());
    }
    let decode_nodes = resolve_pool_nodes(
        cluster,
        "decode_groups",
        &candidate.decode_nodes,
        &candidate.decode_groups,
    )?;
    let decode_nodes =
        nodes_matching_pool_node_filter(cluster, &decode_nodes, &candidate.decode_node_filter);
    if decode_nodes.is_empty() {
        return Err("serving pool decode node topology filters matched no nodes".to_string());
    }
    if !pool_nodes_satisfy_domain_spread(
        cluster,
        &candidate.domain_spread,
        &prefill_nodes,
        &decode_nodes,
    ) {
        return Err(
            "serving pool does not satisfy configured topology-domain spread constraints"
                .to_string(),
        );
    }

    Ok(ResolvedServingPool {
        label: candidate.label.clone(),
        prefill_nodes,
        decode_nodes,
        prefill_gpu_labels: candidate.prefill_gpu_labels.clone(),
        decode_gpu_labels: candidate.decode_gpu_labels.clone(),
    })
}

fn resolve_pool_nodes(
    cluster: &Cluster,
    field_name: &str,
    explicit_nodes: &[NodeId],
    groups: &[String],
) -> Result<Vec<NodeId>, String> {
    let mut nodes = explicit_nodes.to_vec();
    for group in groups {
        let group_nodes = cluster.node_group(group).ok_or_else(|| {
            format!("serving pool {field_name} references unknown group '{group}'")
        })?;
        nodes.extend_from_slice(group_nodes);
    }
    nodes.sort_unstable();
    nodes.dedup();

    if nodes.is_empty() {
        return Err(format!("serving pool {field_name} resolved to no nodes"));
    }

    Ok(nodes)
}

fn routed_serving_node(nodes: &[NodeId], request_idx: u32) -> NodeId {
    if nodes.is_empty() {
        return 0;
    }

    let mut nodes = nodes.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    nodes[request_idx as usize % nodes.len()]
}

fn sorted_unique_nodes(nodes: &[NodeId]) -> Vec<NodeId> {
    let mut nodes = nodes.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        nodes.push(0);
    }
    nodes
}

#[derive(Clone, Debug, PartialEq)]
struct RequestRoute {
    prefill_node: NodeId,
    decode_node: NodeId,
    prefill_route_nodes: Vec<NodeId>,
    prefill_route_gpus: Vec<GpuAddr>,
    decode_route_nodes: Vec<NodeId>,
    decode_route_gpus: Vec<GpuAddr>,
    routing: RoutingDecision,
}

#[derive(Clone, Debug, PartialEq)]
struct RoutingDecision {
    policy: ServingRoutingPolicy,
    candidate_count: u32,
    routable_candidate_count: u32,
    candidates: Vec<ServingRouteCandidateObservation>,
    estimated_e2el_s: f64,
    estimated_kv_transfer_s: f64,
    estimated_kv_resource_wait_s: f64,
    estimated_prefill_wait_s: f64,
    estimated_decode_wait_s: f64,
    reason: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct RoutingLoad {
    prefill_worker_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    decode_worker_ready_s: BTreeMap<GpuAddr, Vec<f64>>,
    kv_route_ready_s: BTreeMap<String, f64>,
}

fn kv_transfer_bytes_for_routes(
    model: &ModelSpec,
    request: &InferenceRequest,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_gpus: &[GpuAddr],
    decode_gpus: &[GpuAddr],
) -> Bytes {
    let prefill_gpu_set: BTreeSet<_> = prefill_gpus.iter().copied().collect();
    let decode_gpu_set: BTreeSet<_> = decode_gpus.iter().copied().collect();
    if !prefill_gpu_set.is_empty() && prefill_gpu_set == decode_gpu_set {
        return Bytes::from_bytes(0);
    }

    let prefill: BTreeSet<_> = prefill_nodes.iter().copied().collect();
    let decode: BTreeSet<_> = decode_nodes.iter().copied().collect();
    if prefill_gpu_set.is_empty() && prefill == decode {
        return Bytes::from_bytes(0);
    }

    let head_dim = model.hidden_size / model.attention_heads.max(1);
    let bytes = request.batch_size as u64
        * request.prompt_tokens as u64
        * model.layers as u64
        * model.kv_heads as u64
        * head_dim as u64
        * 2
        * model.kv_dtype().bytes_per_element();

    Bytes::from_bytes(bytes)
}

#[allow(clippy::too_many_arguments)]
fn route_request(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    effective_prefill_tokens: u32,
    traffic: &ServingTraffic,
    request_idx: u32,
    arrival_s: f64,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_resource_base_node: Option<NodeId>,
    prefill_placement_nodes: &[NodeId],
    decode_resource_base_node: Option<NodeId>,
    decode_placement_nodes: &[NodeId],
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
    routing_load: &mut RoutingLoad,
) -> RequestRoute {
    match traffic.routing_policy {
        ServingRoutingPolicy::RoundRobin => {
            let prefill_node = routed_serving_node(prefill_nodes, request_idx);
            let decode_node = routed_serving_node(decode_nodes, request_idx);
            let prefill_route_nodes = routed_node_set(
                prefill_resource_base_node,
                prefill_placement_nodes,
                prefill_node,
            );
            let decode_route_nodes = routed_node_set(
                decode_resource_base_node,
                decode_placement_nodes,
                decode_node,
            );
            let prefill_route_gpus = routed_gpu_set(
                prefill_resource_base_node,
                &prefill_score.placement.rank_to_gpu,
                prefill_node,
            );
            let decode_route_gpus = routed_gpu_set(
                decode_resource_base_node,
                &decode_one_score.placement.rank_to_gpu,
                decode_node,
            );
            let candidate_count =
                route_candidate_count(prefill_nodes, decode_nodes).min(u32::MAX as usize) as u32;
            let routing_candidates = vec![round_robin_route_candidate_observation(
                cluster,
                model,
                request,
                &prefill_route_nodes,
                &decode_route_nodes,
                &prefill_route_gpus,
                &decode_route_gpus,
                calibration,
                calibration_profile,
            )];
            RequestRoute {
                prefill_node,
                decode_node,
                prefill_route_nodes,
                prefill_route_gpus,
                decode_route_nodes,
                decode_route_gpus,
                routing: RoutingDecision {
                    policy: ServingRoutingPolicy::RoundRobin,
                    candidate_count,
                    routable_candidate_count: candidate_count,
                    candidates: routing_candidates,
                    estimated_e2el_s: f64::INFINITY,
                    estimated_kv_transfer_s: f64::INFINITY,
                    estimated_kv_resource_wait_s: 0.0,
                    estimated_prefill_wait_s: 0.0,
                    estimated_decode_wait_s: 0.0,
                    reason: format!(
                        "round-robin request index {request_idx} selected prefill node {prefill_node} and decode node {decode_node}"
                    ),
                },
            }
        }
        ServingRoutingPolicy::TopologyAware => route_topology_aware_request(
            cluster,
            model,
            request,
            effective_prefill_tokens,
            arrival_s,
            prefill_nodes,
            decode_nodes,
            prefill_resource_base_node,
            prefill_placement_nodes,
            decode_resource_base_node,
            decode_placement_nodes,
            prefill_score,
            decode_one_score,
            decode_tail_scale,
            base_batch_size,
            base_prompt_tokens,
            prefill_worker_slots_per_gpu(traffic),
            decode_worker_slots_per_gpu(traffic),
            calibration,
            calibration_profile,
            routing_load,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn route_topology_aware_request(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    effective_prefill_tokens: u32,
    arrival_s: f64,
    prefill_nodes: &[NodeId],
    decode_nodes: &[NodeId],
    prefill_resource_base_node: Option<NodeId>,
    prefill_placement_nodes: &[NodeId],
    decode_resource_base_node: Option<NodeId>,
    decode_placement_nodes: &[NodeId],
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    prefill_worker_slots_per_gpu: usize,
    decode_worker_slots_per_gpu: usize,
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
    routing_load: &mut RoutingLoad,
) -> RequestRoute {
    let prefill_candidates = sorted_unique_nodes(prefill_nodes);
    let decode_candidates = sorted_unique_nodes(decode_nodes);
    let mut best: Option<(RequestRoute, RouteEstimate)> = None;
    let candidate_count = prefill_candidates
        .len()
        .saturating_mul(decode_candidates.len())
        .min(u32::MAX as usize) as u32;
    let mut routable_candidate_count = 0_u32;
    let mut routing_candidates = Vec::new();

    for &prefill_node in &prefill_candidates {
        let prefill_route_nodes = routed_node_set(
            prefill_resource_base_node,
            prefill_placement_nodes,
            prefill_node,
        );
        for &decode_node in &decode_candidates {
            let decode_route_nodes = routed_node_set(
                decode_resource_base_node,
                decode_placement_nodes,
                decode_node,
            );
            let prefill_route_gpus = routed_gpu_set(
                prefill_resource_base_node,
                &prefill_score.placement.rank_to_gpu,
                prefill_node,
            );
            let decode_route_gpus = routed_gpu_set(
                decode_resource_base_node,
                &decode_one_score.placement.rank_to_gpu,
                decode_node,
            );
            let kv_bytes = kv_transfer_bytes_for_routes(
                model,
                request,
                &prefill_route_nodes,
                &decode_route_nodes,
                &prefill_route_gpus,
                &decode_route_gpus,
            );
            if !Solver::transfer_between_gpus_routable(
                cluster,
                &prefill_route_gpus,
                &decode_route_gpus,
                kv_bytes,
            ) {
                routing_candidates.push(ServingRouteCandidateObservation {
                    prefill_node,
                    decode_node,
                    prefill_route_nodes: prefill_route_nodes.clone(),
                    prefill_route_gpus,
                    decode_route_nodes,
                    decode_route_gpus,
                    selected: false,
                    routable: false,
                    rejection_reason: Some(
                        "KV transfer path is not routable between routed prefill and decode GPUs"
                            .to_string(),
                    ),
                    estimated_e2el_s: None,
                    estimated_kv_transfer_s: None,
                    estimated_kv_resource_wait_s: None,
                    estimated_prefill_wait_s: None,
                    estimated_decode_wait_s: None,
                    kv_transfer_bytes: kv_bytes.as_bytes(),
                    kv_transfer_bottlenecks: Vec::new(),
                    kv_transfer_resources: Vec::new(),
                });
                continue;
            }
            routable_candidate_count = routable_candidate_count.saturating_add(1);
            let kv_cost = Solver::estimate_transfer_between_gpus_with_options(
                cluster,
                &prefill_route_gpus,
                &decode_route_gpus,
                kv_bytes,
                SolverOptions {
                    calibration,
                    calibration_profile,
                    max_candidates: None,
                    search_deadline: None,
                    explicit_placement: None,
                },
            );
            let kv_paths =
                kv_transfer_paths(cluster, &prefill_route_gpus, &decode_route_gpus, kv_bytes);
            let kv_resources = kv_transfer_scheduler_resources(&kv_paths, &kv_cost.bottlenecks);
            let estimate = route_estimate(
                request,
                effective_prefill_tokens,
                arrival_s,
                prefill_node,
                decode_node,
                &prefill_route_gpus,
                &decode_route_gpus,
                prefill_score,
                decode_one_score,
                decode_tail_scale,
                base_batch_size,
                base_prompt_tokens,
                prefill_worker_slots_per_gpu,
                decode_worker_slots_per_gpu,
                kv_cost.total_s,
                &kv_resources,
                routing_load,
            );
            routing_candidates.push(ServingRouteCandidateObservation {
                prefill_node,
                decode_node,
                prefill_route_nodes: prefill_route_nodes.clone(),
                prefill_route_gpus: prefill_route_gpus.clone(),
                decode_route_nodes: decode_route_nodes.clone(),
                decode_route_gpus: decode_route_gpus.clone(),
                selected: false,
                routable: true,
                rejection_reason: None,
                estimated_e2el_s: Some(estimate.e2el_proxy_s),
                estimated_kv_transfer_s: Some(estimate.kv_transfer_s),
                estimated_kv_resource_wait_s: Some(estimate.kv_resource_wait_s),
                estimated_prefill_wait_s: Some(estimate.prefill_wait_s),
                estimated_decode_wait_s: Some(estimate.decode_wait_s),
                kv_transfer_bytes: kv_bytes.as_bytes(),
                kv_transfer_bottlenecks: kv_cost.bottlenecks.clone(),
                kv_transfer_resources: kv_resources.clone(),
            });
            let route = RequestRoute {
                prefill_node,
                decode_node,
                prefill_route_nodes: prefill_route_nodes.clone(),
                prefill_route_gpus,
                decode_route_nodes,
                decode_route_gpus,
                routing: RoutingDecision {
                    policy: ServingRoutingPolicy::TopologyAware,
                    candidate_count,
                    routable_candidate_count,
                    candidates: Vec::new(),
                    estimated_e2el_s: estimate.e2el_proxy_s,
                    estimated_kv_transfer_s: estimate.kv_transfer_s,
                    estimated_kv_resource_wait_s: estimate.kv_resource_wait_s,
                    estimated_prefill_wait_s: estimate.prefill_wait_s,
                    estimated_decode_wait_s: estimate.decode_wait_s,
                    reason: String::new(),
                },
            };

            if best
                .as_ref()
                .map(|(_, best_estimate)| estimate.better_than(best_estimate))
                .unwrap_or(true)
            {
                best = Some((route, estimate));
            }
        }
    }

    let (route, estimate) = best.unwrap_or_else(|| {
        let prefill_node = routed_serving_node(prefill_nodes, 0);
        let decode_node = routed_serving_node(decode_nodes, 0);
        let prefill_route_gpus = routed_gpu_set(
            prefill_resource_base_node,
            &prefill_score.placement.rank_to_gpu,
            prefill_node,
        );
        let decode_route_gpus = routed_gpu_set(
            decode_resource_base_node,
            &decode_one_score.placement.rank_to_gpu,
            decode_node,
        );
        let route = RequestRoute {
            prefill_node,
            decode_node,
            prefill_route_nodes: routed_node_set(
                prefill_resource_base_node,
                prefill_placement_nodes,
                prefill_node,
            ),
            decode_route_nodes: routed_node_set(
                decode_resource_base_node,
                decode_placement_nodes,
                decode_node,
            ),
            prefill_route_gpus,
            decode_route_gpus,
            routing: RoutingDecision {
                policy: ServingRoutingPolicy::TopologyAware,
                candidate_count,
                routable_candidate_count,
                candidates: Vec::new(),
                estimated_e2el_s: 0.0,
                estimated_kv_transfer_s: 0.0,
                estimated_kv_resource_wait_s: 0.0,
                estimated_prefill_wait_s: 0.0,
                estimated_decode_wait_s: 0.0,
                reason: "topology-aware router found no routable candidates and fell back to deterministic first nodes".to_string(),
            },
        };
        (
            route,
            RouteEstimate {
                prefill_finish_s: arrival_s,
                decode_finish_s: arrival_s,
                e2el_proxy_s: 0.0,
                kv_transfer_s: 0.0,
                kv_resource_wait_s: 0.0,
                prefill_wait_s: 0.0,
                decode_wait_s: 0.0,
                kv_finish_s: arrival_s,
                kv_resources: Vec::new(),
                prefill_node,
                decode_node,
            },
        )
    });
    mark_selected_routing_candidate(&mut routing_candidates, &route);
    mark_route_workers_ready(
        &mut routing_load.prefill_worker_ready_s,
        &route.prefill_route_gpus,
        route.prefill_node,
        prefill_worker_slots_per_gpu,
        estimate.prefill_finish_s,
    );
    mark_route_workers_ready(
        &mut routing_load.decode_worker_ready_s,
        &route.decode_route_gpus,
        route.decode_node,
        decode_worker_slots_per_gpu,
        estimate.decode_finish_s,
    );
    mark_route_resources_ready(
        &mut routing_load.kv_route_ready_s,
        &estimate.kv_resources,
        estimate.kv_finish_s,
    );
    RequestRoute {
        routing: RoutingDecision {
            policy: route.routing.policy,
            candidate_count,
            routable_candidate_count,
            candidates: routing_candidates,
            estimated_e2el_s: estimate.e2el_proxy_s,
            estimated_kv_transfer_s: estimate.kv_transfer_s,
            estimated_kv_resource_wait_s: estimate.kv_resource_wait_s,
            estimated_prefill_wait_s: estimate.prefill_wait_s,
            estimated_decode_wait_s: estimate.decode_wait_s,
            reason: if routable_candidate_count == 0 {
                route.routing.reason
            } else {
                format!(
                    "topology-aware router selected prefill node {} and decode node {} from {}/{} routable candidates by estimated E2EL {:.6}s, KV {:.6}s, KV resource wait {:.6}s, prefill wait {:.6}s, decode wait {:.6}s",
                    route.prefill_node,
                    route.decode_node,
                    routable_candidate_count,
                    candidate_count,
                    estimate.e2el_proxy_s,
                    estimate.kv_transfer_s,
                    estimate.kv_resource_wait_s,
                    estimate.prefill_wait_s,
                    estimate.decode_wait_s
                )
            },
        },
        ..route
    }
}

fn route_candidate_count(prefill_nodes: &[NodeId], decode_nodes: &[NodeId]) -> usize {
    sorted_unique_nodes(prefill_nodes)
        .len()
        .saturating_mul(sorted_unique_nodes(decode_nodes).len())
}

#[allow(clippy::too_many_arguments)]
fn round_robin_route_candidate_observation(
    cluster: &Cluster,
    model: &ModelSpec,
    request: &InferenceRequest,
    prefill_route_nodes: &[NodeId],
    decode_route_nodes: &[NodeId],
    prefill_route_gpus: &[GpuAddr],
    decode_route_gpus: &[GpuAddr],
    calibration: SimulationCalibration,
    calibration_profile: Option<&CalibrationProfileMetadata>,
) -> ServingRouteCandidateObservation {
    let kv_transfer_bytes = kv_transfer_bytes_for_routes(
        model,
        request,
        prefill_route_nodes,
        decode_route_nodes,
        prefill_route_gpus,
        decode_route_gpus,
    );
    let routable = Solver::transfer_between_gpus_routable(
        cluster,
        prefill_route_gpus,
        decode_route_gpus,
        kv_transfer_bytes,
    );
    let (estimated_kv_transfer_s, kv_transfer_bottlenecks, kv_transfer_resources) = if routable {
        let kv_cost = Solver::estimate_transfer_between_gpus_with_options(
            cluster,
            prefill_route_gpus,
            decode_route_gpus,
            kv_transfer_bytes,
            SolverOptions {
                calibration,
                calibration_profile,
                max_candidates: None,
                search_deadline: None,
                explicit_placement: None,
            },
        );
        let kv_paths = kv_transfer_paths(
            cluster,
            prefill_route_gpus,
            decode_route_gpus,
            kv_transfer_bytes,
        );
        let kv_resources = kv_transfer_scheduler_resources(&kv_paths, &kv_cost.bottlenecks);
        (Some(kv_cost.total_s), kv_cost.bottlenecks, kv_resources)
    } else {
        (None, Vec::new(), Vec::new())
    };

    ServingRouteCandidateObservation {
        prefill_node: prefill_route_nodes.first().copied().unwrap_or(0),
        decode_node: decode_route_nodes.first().copied().unwrap_or(0),
        prefill_route_nodes: prefill_route_nodes.to_vec(),
        prefill_route_gpus: prefill_route_gpus.to_vec(),
        decode_route_nodes: decode_route_nodes.to_vec(),
        decode_route_gpus: decode_route_gpus.to_vec(),
        selected: true,
        routable,
        rejection_reason: (!routable).then(|| {
            "KV transfer path is not routable between routed prefill and decode GPUs".to_string()
        }),
        estimated_e2el_s: None,
        estimated_kv_transfer_s,
        estimated_kv_resource_wait_s: None,
        estimated_prefill_wait_s: None,
        estimated_decode_wait_s: None,
        kv_transfer_bytes: kv_transfer_bytes.as_bytes(),
        kv_transfer_bottlenecks,
        kv_transfer_resources,
    }
}

fn mark_selected_routing_candidate(
    candidates: &mut [ServingRouteCandidateObservation],
    route: &RequestRoute,
) {
    for candidate in candidates {
        candidate.selected = candidate.prefill_node == route.prefill_node
            && candidate.decode_node == route.decode_node
            && candidate.prefill_route_nodes == route.prefill_route_nodes
            && candidate.prefill_route_gpus == route.prefill_route_gpus
            && candidate.decode_route_nodes == route.decode_route_nodes
            && candidate.decode_route_gpus == route.decode_route_gpus;
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RouteEstimate {
    prefill_finish_s: f64,
    kv_finish_s: f64,
    decode_finish_s: f64,
    e2el_proxy_s: f64,
    kv_transfer_s: f64,
    kv_resource_wait_s: f64,
    prefill_wait_s: f64,
    decode_wait_s: f64,
    kv_resources: Vec<String>,
    prefill_node: NodeId,
    decode_node: NodeId,
}

impl RouteEstimate {
    fn better_than(&self, other: &Self) -> bool {
        self.e2el_proxy_s
            .total_cmp(&other.e2el_proxy_s)
            .then_with(|| self.kv_transfer_s.total_cmp(&other.kv_transfer_s))
            .then_with(|| self.kv_resource_wait_s.total_cmp(&other.kv_resource_wait_s))
            .then_with(|| self.decode_wait_s.total_cmp(&other.decode_wait_s))
            .then_with(|| self.prefill_node.cmp(&other.prefill_node))
            .then_with(|| self.decode_node.cmp(&other.decode_node))
            .is_lt()
    }
}

#[allow(clippy::too_many_arguments)]
fn route_estimate(
    request: &InferenceRequest,
    effective_prefill_tokens: u32,
    arrival_s: f64,
    prefill_node: NodeId,
    decode_node: NodeId,
    prefill_route_gpus: &[GpuAddr],
    decode_route_gpus: &[GpuAddr],
    prefill_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    decode_tail_scale: f64,
    base_batch_size: u32,
    base_prompt_tokens: u32,
    prefill_worker_slots_per_gpu: usize,
    decode_worker_slots_per_gpu: usize,
    kv_transfer_s: f64,
    kv_resources: &[String],
    routing_load: &RoutingLoad,
) -> RouteEstimate {
    let prefill_scale = request_token_scale(
        request.batch_size,
        effective_prefill_tokens,
        base_batch_size,
        base_prompt_tokens,
    );
    let prefill_service_s = finite_or_zero(prefill_score.estimated_latency_s) * prefill_scale;
    let first_token_scale = request_token_scale(request.batch_size, 1, base_batch_size, 1);
    let decode_iterations =
        1.0 + f64::from(request.decode_tokens.saturating_sub(1)) * decode_tail_scale;
    let decode_service_s = finite_or_zero(decode_one_score.estimated_latency_s)
        * first_token_scale
        * decode_iterations;
    let prefill_ready_s = route_workers_ready_s(
        &routing_load.prefill_worker_ready_s,
        prefill_route_gpus,
        prefill_node,
        prefill_worker_slots_per_gpu,
    );
    let decode_ready_s = route_workers_ready_s(
        &routing_load.decode_worker_ready_s,
        decode_route_gpus,
        decode_node,
        decode_worker_slots_per_gpu,
    );
    let prefill_start_s = arrival_s.max(prefill_ready_s);
    let prefill_finish_s = prefill_start_s + prefill_service_s;
    let kv_ready_s = route_resources_ready_s(&routing_load.kv_route_ready_s, kv_resources);
    let kv_start_s = prefill_finish_s.max(kv_ready_s);
    let kv_finish_s = kv_start_s + finite_or_zero(kv_transfer_s);
    let decode_start_s = kv_finish_s.max(decode_ready_s);
    let decode_finish_s = decode_start_s + decode_service_s;

    RouteEstimate {
        prefill_finish_s,
        kv_finish_s,
        decode_finish_s,
        e2el_proxy_s: (decode_finish_s - arrival_s).max(0.0),
        kv_transfer_s: finite_or_zero(kv_transfer_s),
        kv_resource_wait_s: (kv_start_s - prefill_finish_s).max(0.0),
        prefill_wait_s: (prefill_start_s - arrival_s).max(0.0),
        decode_wait_s: (decode_start_s - kv_finish_s).max(0.0),
        kv_resources: kv_resources.to_vec(),
        prefill_node,
        decode_node,
    }
}

fn route_workers_ready_s(
    ready_s: &BTreeMap<GpuAddr, Vec<f64>>,
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
    worker_slots_per_gpu: usize,
) -> f64 {
    route_worker_gpus(route_gpus, fallback_node)
        .iter()
        .map(|gpu| worker_slot_ready_s(ready_s, *gpu, worker_slots_per_gpu))
        .fold(0.0, f64::max)
}

fn route_resources_ready_s(ready_s: &BTreeMap<String, f64>, resources: &[String]) -> f64 {
    resources
        .iter()
        .filter_map(|resource| ready_s.get(resource).copied())
        .fold(0.0, f64::max)
}

fn mark_route_workers_ready(
    ready_s: &mut BTreeMap<GpuAddr, Vec<f64>>,
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
    worker_slots_per_gpu: usize,
    finish_s: f64,
) {
    for gpu in route_worker_gpus(route_gpus, fallback_node) {
        mark_worker_slot_ready(ready_s, gpu, worker_slots_per_gpu, finish_s);
    }
}

fn worker_slot_ready_s(
    ready_s: &BTreeMap<GpuAddr, Vec<f64>>,
    gpu: GpuAddr,
    worker_slots_per_gpu: usize,
) -> f64 {
    let slots = worker_slots_per_gpu.max(1);
    let Some(ready_slots) = ready_s.get(&gpu) else {
        return 0.0;
    };
    if ready_slots.len() < slots {
        return 0.0;
    }
    ready_slots.iter().copied().fold(f64::INFINITY, f64::min)
}

fn mark_worker_slot_ready(
    ready_s: &mut BTreeMap<GpuAddr, Vec<f64>>,
    gpu: GpuAddr,
    worker_slots_per_gpu: usize,
    finish_s: f64,
) -> Option<u32> {
    if !finish_s.is_finite() {
        return None;
    }
    let slots = worker_slots_per_gpu.max(1);
    let ready_slots = ready_s.entry(gpu).or_default();
    if ready_slots.len() < slots {
        ready_slots.push(finish_s);
        return Some((ready_slots.len() - 1).min(u32::MAX as usize) as u32);
    }
    if let Some((slot_idx, _)) = ready_slots
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1).then_with(|| left.0.cmp(&right.0)))
    {
        ready_slots[slot_idx] = finish_s;
        return Some(slot_idx.min(u32::MAX as usize) as u32);
    }
    None
}

struct WorkerAssignmentSpan<'a> {
    phase: &'a str,
    start_s: f64,
    finish_s: f64,
    operation_ids: &'a [usize],
}

fn assign_worker_slots(
    ready_s: &mut BTreeMap<GpuAddr, Vec<f64>>,
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
    worker_slots_per_gpu: usize,
    span: WorkerAssignmentSpan<'_>,
) -> Vec<ServingWorkerAssignmentObservation> {
    route_worker_gpus(route_gpus, fallback_node)
        .into_iter()
        .filter_map(|gpu| {
            mark_worker_slot_ready(ready_s, gpu, worker_slots_per_gpu, span.finish_s).map(|slot| {
                ServingWorkerAssignmentObservation {
                    phase: span.phase.to_string(),
                    node_id: gpu.node_id,
                    local_gpu_id: gpu.local_gpu_id,
                    slot,
                    start_s: span.start_s,
                    finish_s: span.finish_s,
                    operation_ids: span.operation_ids.to_vec(),
                }
            })
        })
        .collect()
}

fn assignments_for_worker_gpus(
    assignments: &[ServingWorkerAssignmentObservation],
    route_gpus: &[GpuAddr],
    fallback_node: NodeId,
) -> Vec<ServingWorkerAssignmentObservation> {
    let route_gpus = route_worker_gpus(route_gpus, fallback_node)
        .into_iter()
        .collect::<BTreeSet<_>>();
    assignments
        .iter()
        .filter(|assignment| {
            route_gpus.contains(&GpuAddr {
                node_id: assignment.node_id,
                local_gpu_id: assignment.local_gpu_id,
            })
        })
        .cloned()
        .collect()
}

fn mark_route_resources_ready(
    ready_s: &mut BTreeMap<String, f64>,
    resources: &[String],
    finish_s: f64,
) {
    if !finish_s.is_finite() {
        return;
    }
    for resource in resources {
        let entry = ready_s.entry(resource.clone()).or_insert(0.0);
        *entry = entry.max(finish_s);
    }
}

fn route_worker_gpus(route_gpus: &[GpuAddr], fallback_node: NodeId) -> Vec<GpuAddr> {
    let mut gpus = route_gpus.to_vec();
    if gpus.is_empty() {
        gpus.push(GpuAddr {
            node_id: fallback_node,
            local_gpu_id: 0,
        });
    }
    gpus.sort_unstable();
    gpus.dedup();
    gpus
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
struct MeasurementWindowSelection {
    start_s: f64,
    end_s: f64,
    configured_start_bound: bool,
    configured_end_bound: bool,
    steady_state_requested: bool,
    steady_state_applied: bool,
    steady_state_candidate_start_s: Option<f64>,
    steady_state_candidate_end_s: Option<f64>,
    steady_state_min_requests: Option<u32>,
    steady_state_max_cv: Option<f64>,
    steady_state_sample_count: u32,
    steady_state_matching_window_count: u32,
    steady_state_candidate_request_count: Option<u32>,
    steady_state_candidate_e2el_mean_s: Option<f64>,
    steady_state_candidate_e2el_stddev_s: Option<f64>,
    steady_state_candidate_e2el_cv: Option<f64>,
    steady_state_candidate_e2el_std_error_s: Option<f64>,
    steady_state_candidate_metric_count: u32,
    steady_state_candidate_worst_metric: Option<String>,
    steady_state_candidate_worst_cv: Option<f64>,
    steady_state_candidate_output_tokens: Option<u64>,
    steady_state_candidate_throughput_tokens_per_s: Option<f64>,
    steady_state_candidate_metrics: Vec<ServingSteadyStateMetricObservation>,
    steady_state_candidate_utilization_count: u32,
    steady_state_candidate_worst_utilization_resource: Option<String>,
    steady_state_candidate_worst_utilization_cv: Option<f64>,
    steady_state_candidate_utilization: Vec<ServingSteadyStateUtilizationObservation>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct MeasurementWindowRequestCounts {
    request_count: u32,
    completed_request_count: u32,
    failed_request_count: u32,
    rejected_request_count: u32,
    timed_out_request_count: u32,
    cancelled_request_count: u32,
    deadline_constrained_request_count: u32,
    deadline_missed_request_count: u32,
}

impl MeasurementWindowSelection {
    fn rejected() -> Self {
        Self {
            start_s: f64::INFINITY,
            end_s: f64::INFINITY,
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

    fn into_observation(
        self,
        request_counts: MeasurementWindowRequestCounts,
        metric_source_counts: Vec<ServingMeasurementMetricSourceCount>,
    ) -> ServingMeasurementWindowObservation {
        let source = if !self.start_s.is_finite() || !self.end_s.is_finite() {
            "unavailable"
        } else if self.steady_state_applied {
            "steady_state"
        } else if self.configured_start_bound || self.configured_end_bound {
            "configured"
        } else if self.steady_state_requested {
            "steady_state_unavailable"
        } else {
            "default"
        };
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
        ServingMeasurementWindowObservation {
            source: source.to_string(),
            start_s: self.start_s,
            end_s: self.end_s,
            duration_s: (self.end_s - self.start_s).max(0.0),
            request_count: request_counts.request_count,
            completed_request_count: request_counts.completed_request_count,
            failed_request_count: request_counts.failed_request_count,
            rejected_request_count: request_counts.rejected_request_count,
            timed_out_request_count: request_counts.timed_out_request_count,
            cancelled_request_count: request_counts.cancelled_request_count,
            deadline_constrained_request_count: request_counts.deadline_constrained_request_count,
            deadline_missed_request_count: request_counts.deadline_missed_request_count,
            measured_requests: request_counts.completed_request_count,
            lifecycle_event_metric_request_count,
            fallback_metric_request_count,
            metric_source_counts,
            configured_start_bound: self.configured_start_bound,
            configured_end_bound: self.configured_end_bound,
            steady_state_requested: self.steady_state_requested,
            steady_state_applied: self.steady_state_applied,
            steady_state_candidate_start_s: self.steady_state_candidate_start_s,
            steady_state_candidate_end_s: self.steady_state_candidate_end_s,
            steady_state_min_requests: self.steady_state_min_requests,
            steady_state_max_cv: self.steady_state_max_cv,
            steady_state_sample_count: self.steady_state_sample_count,
            steady_state_matching_window_count: self.steady_state_matching_window_count,
            steady_state_candidate_request_count: self.steady_state_candidate_request_count,
            steady_state_candidate_e2el_mean_s: self.steady_state_candidate_e2el_mean_s,
            steady_state_candidate_e2el_stddev_s: self.steady_state_candidate_e2el_stddev_s,
            steady_state_candidate_e2el_cv: self.steady_state_candidate_e2el_cv,
            steady_state_candidate_e2el_std_error_s: self.steady_state_candidate_e2el_std_error_s,
            steady_state_candidate_metric_count: self.steady_state_candidate_metric_count,
            steady_state_candidate_worst_metric: self.steady_state_candidate_worst_metric,
            steady_state_candidate_worst_cv: self.steady_state_candidate_worst_cv,
            steady_state_candidate_output_tokens: self.steady_state_candidate_output_tokens,
            steady_state_candidate_throughput_tokens_per_s: self
                .steady_state_candidate_throughput_tokens_per_s,
            steady_state_candidate_metrics: self.steady_state_candidate_metrics,
            steady_state_candidate_utilization_count: self.steady_state_candidate_utilization_count,
            steady_state_candidate_worst_utilization_resource: self
                .steady_state_candidate_worst_utilization_resource,
            steady_state_candidate_worst_utilization_cv: self
                .steady_state_candidate_worst_utilization_cv,
            steady_state_candidate_utilization: self.steady_state_candidate_utilization,
        }
    }
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

fn serving_calibration_fits(
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    simulation: &ServingSimulation,
) -> Vec<CalibrationFitApplication> {
    let mut fits = Vec::new();
    fits.extend(prefill_score.calibration_fits.clone());
    fits.extend(decode_score.calibration_fits.clone());
    fits.extend(decode_one_score.calibration_fits.clone());
    fits.extend(simulation.calibration_fits.clone());
    fits
}

fn serving_phase_calibration(
    calibration_profile: Option<&CalibrationProfileMetadata>,
    metrics: &ServingMetrics,
    observations: &[ServingRequestObservation],
    calibration_fits: &[CalibrationFitApplication],
) -> Vec<ServingPhaseCalibrationObservation> {
    let kv_transfer_active = metrics.kv_transfer_s > 0.0
        || observations
            .iter()
            .any(|observation| observation.kv_transfer_bytes > 0);
    let mut components = vec![
        ("prefill", metrics.prefill_s, metrics.prefill_s > 0.0),
        ("decode", metrics.decode_s, metrics.decode_s > 0.0),
        ("kv_transfer", metrics.kv_transfer_s, kv_transfer_active),
    ];
    push_active_queue_calibration_component(
        &mut components,
        "prefill_queue",
        metrics.prefill_worker_queue_s + metrics.prefill_resource_queue_s,
    );
    push_active_queue_calibration_component(
        &mut components,
        "decode_queue",
        metrics.decode_worker_queue_s + metrics.decode_resource_queue_s,
    );
    push_active_queue_calibration_component(
        &mut components,
        "kv_worker_queue",
        metrics.kv_worker_queue_s,
    );
    push_active_queue_calibration_component(
        &mut components,
        "kv_route_resource_queue",
        metrics.kv_resource_queue_s,
    );

    components
        .into_iter()
        .map(|(phase, estimated_s, active)| {
            let applied_targets = calibration_fits
                .iter()
                .filter(|fit| fit.phase == phase)
                .map(|fit| fit.target.clone())
                .collect::<Vec<_>>();
            let calibrated = active && !applied_targets.is_empty();
            let status = if !active {
                "inactive"
            } else if calibrated {
                "calibrated"
            } else if calibration_profile.is_some() {
                "uncalibrated_no_fit"
            } else {
                "uncalibrated_no_profile"
            };
            ServingPhaseCalibrationObservation {
                phase: phase.to_string(),
                active,
                calibrated,
                fit_count: applied_targets.len().min(u32::MAX as usize) as u32,
                applied_targets,
                estimated_s,
                status: status.to_string(),
            }
        })
        .collect()
}

fn serving_calibration_summary(
    phases: &[ServingPhaseCalibrationObservation],
    fits: &[CalibrationFitApplication],
    gate_violations: &[CalibrationGateViolation],
) -> ServingCalibrationSummary {
    let active_phase_count = phases.iter().filter(|phase| phase.active).count() as u32;
    let calibrated_phase_count = phases
        .iter()
        .filter(|phase| phase.active && phase.calibrated)
        .count() as u32;
    let uncalibrated_phase_count = active_phase_count.saturating_sub(calibrated_phase_count);
    let coverage_fraction = if active_phase_count > 0 {
        f64::from(calibrated_phase_count) / f64::from(active_phase_count)
    } else {
        0.0
    };
    let fit_count = fits.len().min(u32::MAX as usize) as u32;
    let extrapolated_fit_count = fits
        .iter()
        .filter(|fit| fit.applicability_status == "extrapolated")
        .count()
        .min(u32::MAX as usize) as u32;
    let unbounded_fit_count = fits
        .iter()
        .filter(|fit| fit.applicability_status == "unbounded")
        .count()
        .min(u32::MAX as usize) as u32;
    let fit_count_with_uncertainty = fits
        .iter()
        .filter(|fit| fit_has_numeric_uncertainty(fit))
        .count()
        .min(u32::MAX as usize) as u32;
    let min_confidence_score = fits
        .iter()
        .filter_map(|fit| {
            fit.confidence_score
                .is_finite()
                .then_some(fit.confidence_score)
        })
        .fold(None, |min_score: Option<f64>, score| {
            Some(min_score.map_or(score, |min_score| min_score.min(score)))
        });
    let max_extrapolation_ratio = fits
        .iter()
        .filter_map(|fit| {
            fit.max_extrapolation_ratio
                .is_finite()
                .then_some(fit.max_extrapolation_ratio)
        })
        .fold(None, |max_ratio: Option<f64>, ratio| {
            Some(max_ratio.map_or(ratio, |max_ratio| max_ratio.max(ratio)))
        });
    let predicted_latency_s = fits
        .iter()
        .filter(|fit| fit.prediction_kind == "latency")
        .filter_map(|fit| {
            fit.predicted_s
                .is_finite()
                .then_some(fit.predicted_s.max(0.0))
        })
        .sum::<f64>();
    let uncertainty_s_squared = fits
        .iter()
        .filter(|fit| fit.prediction_kind == "latency")
        .filter_map(|fit| fit.absolute_uncertainty_s)
        .filter(|uncertainty_s| uncertainty_s.is_finite() && *uncertainty_s >= 0.0)
        .map(|uncertainty_s| uncertainty_s * uncertainty_s)
        .sum::<f64>();
    let latency_fit_count_with_uncertainty = fits
        .iter()
        .filter(|fit| fit.prediction_kind == "latency")
        .filter_map(|fit| fit.absolute_uncertainty_s)
        .filter(|uncertainty_s| uncertainty_s.is_finite() && *uncertainty_s >= 0.0)
        .count();
    let absolute_uncertainty_s =
        (latency_fit_count_with_uncertainty > 0).then_some(uncertainty_s_squared.sqrt());
    let latency_relative_uncertainty_pct = absolute_uncertainty_s
        .filter(|_| predicted_latency_s.is_finite() && predicted_latency_s > 0.0)
        .map(|absolute_uncertainty_s| absolute_uncertainty_s / predicted_latency_s * 100.0);
    let max_fit_relative_uncertainty_pct = fits
        .iter()
        .filter_map(|fit| fit.relative_uncertainty_pct)
        .filter(|relative_pct| relative_pct.is_finite() && *relative_pct >= 0.0)
        .fold(None, |max_pct: Option<f64>, relative_pct| {
            Some(max_pct.map_or(relative_pct, |max_pct| max_pct.max(relative_pct)))
        });
    let relative_uncertainty_pct = max_optional_f64(
        latency_relative_uncertainty_pct,
        max_fit_relative_uncertainty_pct,
    );
    let gate_violation_count = gate_violations.len().min(u32::MAX as usize) as u32;
    let hard_gate_violation_count = gate_violations
        .iter()
        .filter(|violation| violation.action.as_str() == "reject")
        .count()
        .min(u32::MAX as usize) as u32;
    let status = calibration_summary_status(
        active_phase_count,
        uncalibrated_phase_count,
        hard_gate_violation_count,
        gate_violation_count,
        fit_count,
    )
    .to_string();

    ServingCalibrationSummary {
        status,
        active_phase_count,
        calibrated_phase_count,
        uncalibrated_phase_count,
        coverage_fraction,
        fit_count,
        extrapolated_fit_count,
        unbounded_fit_count,
        fit_count_with_uncertainty,
        min_confidence_score,
        max_extrapolation_ratio,
        relative_uncertainty_pct,
        absolute_uncertainty_s,
        gate_violation_count,
        hard_gate_violation_count,
    }
}

fn fit_has_numeric_uncertainty(fit: &CalibrationFitApplication) -> bool {
    finite_nonnegative(fit.relative_uncertainty_pct)
        || finite_nonnegative(fit.absolute_uncertainty_s)
        || finite_nonnegative(fit.absolute_uncertainty_value)
}

fn max_optional_f64(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn finite_nonnegative(value: Option<f64>) -> bool {
    value.is_some_and(|value| value.is_finite() && value >= 0.0)
}

fn calibration_summary_status(
    active_phase_count: u32,
    uncalibrated_phase_count: u32,
    hard_gate_violation_count: u32,
    gate_violation_count: u32,
    fit_count: u32,
) -> &'static str {
    if hard_gate_violation_count > 0 {
        "gate_rejected"
    } else if gate_violation_count > 0 {
        "gate_warning"
    } else if active_phase_count == 0 {
        "no_active_phases"
    } else if uncalibrated_phase_count == 0 {
        "fully_calibrated"
    } else if fit_count > 0 {
        "partially_calibrated"
    } else {
        "uncalibrated"
    }
}

fn push_active_queue_calibration_component(
    components: &mut Vec<(&'static str, f64, bool)>,
    phase: &'static str,
    estimated_s: f64,
) {
    if calibration_component_active(estimated_s) {
        components.push((phase, estimated_s, true));
    }
}

fn calibration_component_active(estimated_s: f64) -> bool {
    estimated_s.is_finite() && estimated_s > 1e-12
}

#[allow(clippy::too_many_arguments)]
fn serving_approximations(
    traffic: &ServingTraffic,
    pool: &ResolvedServingPool,
    cluster: &Cluster,
    model: &ModelSpec,
    model_id: Option<&str>,
    serving_stack: Option<&str>,
    serving_runtime_features: &[String],
    calibration_profile: Option<&CalibrationProfileMetadata>,
    prefill_score: &ScoredParallelismConfig,
    decode_score: &ScoredParallelismConfig,
    decode_one_score: &ScoredParallelismConfig,
    simulation: &ServingSimulation,
    calibration_fits: &[CalibrationFitApplication],
    feasible: bool,
) -> Vec<SimulationApproximation> {
    let mut approximations = Vec::new();
    for approximation in prefill_score
        .approximations
        .iter()
        .chain(decode_score.approximations.iter())
        .chain(decode_one_score.approximations.iter())
    {
        push_serving_approximation(&mut approximations, approximation.clone());
    }

    if !feasible {
        return approximations;
    }

    push_serving_approximation(
        &mut approximations,
        SimulationApproximation::new(
            "serving",
            "queueing",
            format!(
                "prefill_batching={},decode_batching={}",
                prefill_batching_label(&traffic.prefill_batching),
                decode_batching_label(&traffic.decode_batching)
            ),
            "approximate_serving_event_loop",
            "Serving requests are scheduled with an approximate event timeline that tracks routed prefill/decode worker readiness, not a production event loop with full worker queues, preemption, backpressure, CUDA stream, and control-plane effects.",
            Some(
                "add a worker-local online scheduler with queue state and backpressure before using the result to make fine-grained latency SLO claims"
                    .to_string(),
            ),
        ),
    );

    if calibration_profile.is_none() {
        push_serving_approximation(
            &mut approximations,
            SimulationApproximation::new(
                "serving",
                "runtime",
                "calibration_profile",
                "serving_stack_uncalibrated",
                "No calibration profile is loaded, so runtime-specific serving behavior such as paged attention, chunked prefill, CUDA graphs, continuous batching, and KV-transfer implementation is not tied to measured backend data.",
                Some(
                    "load a calibration profile with serving_stack metadata and backend-specific benchmark fits, or reject this approximation for runtime-sensitive comparisons"
                        .to_string(),
                ),
            ),
        );
    } else if let Some(profile) = calibration_profile {
        match (
            non_empty_metadata(serving_stack),
            non_empty_metadata(profile.serving_stack.as_deref()),
        ) {
            (_, None) => {
                push_serving_approximation(
                    &mut approximations,
                    SimulationApproximation::new(
                        "serving",
                        "runtime",
                        "calibration_profile",
                        "serving_stack_unspecified",
                        "The loaded calibration profile does not declare a serving_stack, so backend/runtime-specific effects are not explicit in the candidate evidence.",
                        Some(
                            "set profile.serving_stack in the calibration profile, for example vLLM, TensorRT-LLM, SGLang, Dynamo, Ray Serve, Triton, or a custom runtime"
                                .to_string(),
                        ),
                    ),
                );
            }
            (Some(workload_stack), Some(profile_stack))
                if !serving_stack_metadata_matches(profile_stack, workload_stack) =>
            {
                push_serving_approximation(
                    &mut approximations,
                    SimulationApproximation::new(
                        "serving",
                        "runtime",
                        "calibration_profile",
                        "calibration_profile_serving_stack_mismatch",
                        format!(
                            "The calibration profile declares serving_stack '{}' but the workload requests '{}', so runtime-specific behavior may not transfer cleanly.",
                            profile_stack, workload_stack
                        ),
                        Some(
                            "use a calibration profile measured for the requested serving stack, or split profiles by backend/runtime before comparing runtime-sensitive candidates"
                                .to_string(),
                        ),
                    ),
                );
            }
            _ => {}
        }
    }

    if calibration_profile.is_some() && non_empty_metadata(serving_stack).is_none() {
        push_serving_approximation(
            &mut approximations,
            SimulationApproximation::new(
                "serving",
                "runtime",
                "workload",
                "workload_serving_stack_unspecified",
                "The workload does not declare a serving_stack, so runtime-specific behavior is assumed from the calibration profile and is not explicit in the workload TOML.",
                Some(
                    "set serving_stack in the workload or [serving] section, for example vLLM, TensorRT-LLM, SGLang, Dynamo, Ray Serve, Triton, or a custom runtime"
                        .to_string(),
                ),
            ),
        );
    }

    push_serving_runtime_feature_approximations(
        &mut approximations,
        serving_runtime_features,
        calibration_profile,
    );

    if let Some(profile) = calibration_profile {
        push_calibration_profile_topology_approximations(&mut approximations, profile, cluster);
        push_calibration_profile_model_approximation(
            &mut approximations,
            profile,
            model_id,
            traffic,
        );
        push_calibration_profile_provenance_approximation(&mut approximations, profile);

        match profile.dtype.as_deref() {
            Some(profile_dtype) if !profile_dtype_matches_model(profile_dtype, model.dtype) => {
                push_serving_approximation(
                    &mut approximations,
                    SimulationApproximation::new(
                        "serving",
                        "calibration",
                        "calibration_profile",
                        "calibration_profile_dtype_mismatch",
                        format!(
                            "The calibration profile declares dtype '{}' but the workload model dtype is '{}', so fitted serving latencies and memory assumptions may not be applicable.",
                            profile_dtype,
                            model_dtype_label(model.dtype)
                        ),
                        Some(
                            "use a calibration profile measured for the workload dtype, or split profiles by dtype before comparing runtime-sensitive candidates"
                                .to_string(),
                        ),
                    ),
                );
            }
            None => {
                push_serving_approximation(
                    &mut approximations,
                    SimulationApproximation::new(
                        "serving",
                        "calibration",
                        "calibration_profile",
                        "calibration_profile_dtype_unspecified",
                        "The loaded calibration profile does not declare a dtype, so dtype-specific serving latency, memory bandwidth, and KV-cache behavior are implicit.",
                        Some(
                            "set profile.dtype in the calibration profile, for example bf16, fp16, fp8, or int8"
                                .to_string(),
                        ),
                    ),
                );
            }
            _ => {}
        }
    }

    if calibration_profile.is_some() {
        push_uncalibrated_serving_phase_approximation(
            &mut approximations,
            "prefill",
            simulation.metrics.prefill_s,
            calibration_fits,
        );
        push_uncalibrated_serving_phase_approximation(
            &mut approximations,
            "decode",
            simulation.metrics.decode_s,
            calibration_fits,
        );
        let kv_transfer_active = simulation.metrics.kv_transfer_s > 0.0
            || simulation
                .request_observations
                .iter()
                .any(|observation| observation.kv_transfer_bytes > 0);
        if kv_transfer_active {
            push_uncalibrated_serving_phase_approximation(
                &mut approximations,
                "kv_transfer",
                simulation.metrics.kv_transfer_s,
                calibration_fits,
            );
        }
        push_uncalibrated_serving_queue_component_approximation(
            &mut approximations,
            "prefill",
            "prefill_queue",
            simulation.metrics.prefill_worker_queue_s + simulation.metrics.prefill_resource_queue_s,
            calibration_fits,
        );
        push_uncalibrated_serving_queue_component_approximation(
            &mut approximations,
            "decode",
            "decode_queue",
            simulation.metrics.decode_worker_queue_s + simulation.metrics.decode_resource_queue_s,
            calibration_fits,
        );
        push_uncalibrated_serving_queue_component_approximation(
            &mut approximations,
            "kv_transfer",
            "kv_worker_queue",
            simulation.metrics.kv_worker_queue_s,
            calibration_fits,
        );
        push_uncalibrated_serving_queue_component_approximation(
            &mut approximations,
            "kv_transfer",
            "kv_route_resource_queue",
            simulation.metrics.kv_resource_queue_s,
            calibration_fits,
        );
    }

    match traffic.routing_policy {
        ServingRoutingPolicy::RoundRobin
            if route_candidate_count(&pool.prefill_nodes, &pool.decode_nodes) > 1 =>
        {
            push_serving_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    "serving",
                    "routing",
                    "prefill_decode_replicas",
                    "round_robin_ignores_load_and_locality",
                    "Round-robin routing is deterministic and does not react to queue state, cache affinity, KV ownership, or route contention.",
                    Some(
                        "use topology-aware routing or add cache/load-aware routing before comparing replica placement policies"
                            .to_string(),
                    ),
                ),
            );
        }
        ServingRoutingPolicy::TopologyAware => {
            push_serving_approximation(
                &mut approximations,
                SimulationApproximation::new(
                    "serving",
                    "routing",
                    "prefill_decode_replicas",
                    "approximate_topology_load_routing",
                    "Topology-aware routing uses estimated wait and KV-transfer costs, but it is not a full online router with measured worker load, cache affinity, or shared-route contention.",
                    Some(
                        "calibrate router decisions against serving traces before relying on placement/routing deltas"
                            .to_string(),
                    ),
                ),
            );
        }
        ServingRoutingPolicy::RoundRobin => {}
    }

    if is_disaggregated_pool(pool) || simulation_has_kv_handoff(simulation) {
        push_serving_approximation(
            &mut approximations,
            SimulationApproximation::new(
                "kv_transfer",
                "topology",
                "prefill_decode_handoff",
                "node_set_kv_handoff",
                "KV handoff is estimated between routed prefill/decode node sets, not exact source and destination GPUs with PCIe/NVLink/NIC locality.",
                Some(
                    "add per-worker KV ownership and GPU-to-NIC path modeling before making GPUDirect or rail-pinning claims"
                        .to_string(),
                ),
            ),
        );
    }

    if simulation.metrics.peak_resident_tokens > 0 {
        push_serving_approximation(
            &mut approximations,
            SimulationApproximation::new(
                "decode",
                "capacity",
                "kv_residency",
                "approximate_kv_residency_accounting",
                "KV residency records approximate per-request decode-worker KV block ownership and aggregate/per-node/per-GPU peaks, but it does not yet model a production allocator with eviction, migration, spill, or prefix-cache residency.",
                Some(
                    "add allocator-specific block tables and calibrated eviction/reuse behavior before sizing tight decode residency or cache policies"
                        .to_string(),
                ),
            ),
        );
    }

    push_serving_approximation(
        &mut approximations,
        SimulationApproximation::new(
            "serving",
            "memory",
            "component_headroom",
            "component_memory_estimate",
            "Serving memory headroom is componentized, but still estimated per phase and per GPU rather than tracked as an exact time-varying allocator state.",
            Some(
                "calibrate component memory and add time-aware worker/GPU memory accounting for production OOM analysis"
                    .to_string(),
            ),
        ),
    );

    if matches!(
        traffic.decode_capacity_policy,
        ServingDecodeCapacityPolicy::RequestReject
    ) {
        push_serving_approximation(
            &mut approximations,
            SimulationApproximation::new(
                "serving",
                "admission",
                "prefill_decode_capacity",
                "request_level_capacity_admission",
                "Request-level prefill/decode capacity admission approximates active-token and KV residency limits before worker-local queues, backpressure, and allocator eviction decisions are modeled.",
                Some(
                    "add worker-local admission, queue state, and backpressure before evaluating overload-control policies"
                        .to_string(),
                ),
            ),
        );
    }
    push_downstream_prefill_backpressure_approximations(&mut approximations, simulation);

    if let Some(approximation) = steady_state_measurement_approximation(traffic, simulation) {
        push_serving_approximation(&mut approximations, approximation);
    }

    if calibration_fits
        .iter()
        .any(|fit| fit.max_extrapolation_ratio > 0.0 || fit.applicability_status != "interpolated")
    {
        push_serving_approximation(
            &mut approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "fitted_models",
                "serving_calibration_fit_extrapolation",
                "At least one serving-phase calibration fit is outside its fitted feature range or is not marked interpolated.",
                Some(
                    "add benchmark coverage for this serving shape or hard-reject extrapolated fits via calibration policy gates"
                        .to_string(),
                ),
            ),
        );
    }
    if calibration_fits
        .iter()
        .any(is_serving_metric_fit_application)
    {
        push_serving_approximation(
            &mut approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "serving_metric_summary",
                "aggregate_serving_metric_fit_not_trace_replay",
                "An aggregate serving-metric calibration fit adjusted reported TTFT, TPOT, throughput, or E2EL summary metrics, but per-request observations, metric breakdowns, and the scheduled operation timeline remain the simulator timeline.",
                Some(
                    "use this as calibrated summary evidence for candidate ranking, and collect request-level serving traces before treating per-request timings as calibrated"
                        .to_string(),
                ),
            ),
        );
    }
    push_serving_fit_metadata_approximations(&mut approximations, calibration_fits);

    approximations
}

fn push_serving_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    approximation: SimulationApproximation,
) {
    if approximations.iter().any(|existing| {
        existing.phase == approximation.phase
            && existing.category == approximation.category
            && existing.scope == approximation.scope
            && existing.code == approximation.code
    }) {
        return;
    }
    approximations.push(approximation);
}

fn push_serving_runtime_feature_approximations(
    approximations: &mut Vec<SimulationApproximation>,
    features: &[String],
    calibration_profile: Option<&CalibrationProfileMetadata>,
) {
    for feature in features {
        let Some(feature) = non_empty_metadata(Some(feature.as_str())) else {
            continue;
        };
        if let Some(profile) = calibration_profile {
            if profile.serving_runtime_features.is_empty() {
                push_serving_approximation(
                    approximations,
                    SimulationApproximation::new(
                        "serving",
                        "runtime",
                        format!("runtime_feature:{feature}"),
                        "calibration_profile_runtime_feature_unspecified",
                        format!(
                            "The workload declares serving runtime feature '{feature}', but the loaded calibration profile does not declare serving_runtime_features, so feature-specific runtime effects are not covered by profile metadata."
                        ),
                        Some(
                            "set profile.serving_runtime_features for measured backend features, or split profiles by runtime feature before comparing feature-sensitive candidates"
                                .to_string(),
                        ),
                    ),
                );
                continue;
            }
            if !profile
                .serving_runtime_features
                .iter()
                .any(|profile_feature| serving_runtime_feature_matches(profile_feature, feature))
            {
                push_serving_approximation(
                    approximations,
                    SimulationApproximation::new(
                        "serving",
                        "runtime",
                        format!("runtime_feature:{feature}"),
                        "calibration_profile_runtime_feature_mismatch",
                        format!(
                            "The workload declares serving runtime feature '{feature}', but the loaded calibration profile declares features [{}], so feature-specific runtime effects may not transfer cleanly.",
                            profile.serving_runtime_features.join(", ")
                        ),
                        Some(
                            "use a calibration profile measured with the requested runtime feature, or remove the workload feature before comparing runtime-sensitive candidates"
                                .to_string(),
                        ),
                    ),
                );
            }
            continue;
        }
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "runtime",
                format!("runtime_feature:{feature}"),
                "serving_runtime_feature_assumption",
                format!(
                    "The workload declares serving runtime feature '{feature}', but v1 treats this as an explicit runtime assumption unless a calibration fit captures its backend-specific latency, memory, scheduler, or KV-cache behavior."
                ),
                Some(
                    "use serving-stack calibration profiles with measured fits for runtime features that materially affect ranking, or reject runtime assumptions through approximation policy"
                        .to_string(),
                ),
            ),
        );
    }
}

fn serving_approximation_summary(
    approximations: &[SimulationApproximation],
    policy_violations: &[ApproximationPolicyViolation],
) -> ServingApproximationSummary {
    let mut category_counts = BTreeMap::<String, u32>::new();
    let mut code_counts = BTreeMap::<String, u32>::new();
    let mut uncalibrated_phase_count = 0_u32;
    let mut uncalibrated_queue_component_count = 0_u32;
    let mut extrapolated_fit_count = 0_u32;
    let mut coarse_topology = false;
    let mut approximate_queueing = false;
    let mut uncalibrated_runtime = false;

    for approximation in approximations {
        *category_counts
            .entry(approximation.category.clone())
            .or_default() += 1;
        *code_counts.entry(approximation.code.clone()).or_default() += 1;

        coarse_topology |= approximation.category == "topology";
        approximate_queueing |= matches!(
            approximation.category.as_str(),
            "queueing" | "admission" | "routing"
        ) || approximation.code.contains("queue")
            || approximation.code == "approximate_serving_event_loop";
        uncalibrated_runtime |= approximation.code == "serving_stack_uncalibrated"
            || approximation.code == "serving_stack_unspecified"
            || approximation.code == "calibration_profile_serving_stack_mismatch"
            || approximation.code == "serving_runtime_feature_assumption"
            || approximation.code == "calibration_profile_runtime_feature_unspecified"
            || approximation.code == "calibration_profile_runtime_feature_mismatch";

        match approximation.code.as_str() {
            "serving_phase_uncalibrated" => uncalibrated_phase_count += 1,
            "serving_queue_component_uncalibrated" => {
                uncalibrated_queue_component_count += 1;
            }
            "serving_calibration_fit_extrapolation" => extrapolated_fit_count += 1,
            _ => {}
        }
    }

    let category_counts = approximation_count_entries(category_counts, usize::MAX);
    let top_codes = approximation_count_entries(code_counts, 5);
    let count_for = |category: &str| {
        category_counts
            .iter()
            .find(|entry| entry.name == category)
            .map(|entry| entry.count)
            .unwrap_or(0)
    };
    let status = if !policy_violations.is_empty() {
        "policy_rejected"
    } else if approximations.is_empty() {
        "no_approximations"
    } else if uncalibrated_runtime || uncalibrated_phase_count > 0 || extrapolated_fit_count > 0 {
        "calibration_risk"
    } else if coarse_topology || approximate_queueing {
        "model_approximation"
    } else {
        "approximate"
    }
    .to_string();

    ServingApproximationSummary {
        status,
        approximation_count: approximations.len() as u32,
        policy_violation_count: policy_violations.len() as u32,
        calibration_count: count_for("calibration"),
        topology_count: count_for("topology"),
        queueing_count: count_for("queueing"),
        runtime_count: count_for("runtime"),
        memory_count: count_for("memory"),
        capacity_count: count_for("capacity"),
        routing_count: count_for("routing"),
        admission_count: count_for("admission"),
        category_counts,
        top_codes,
        uncalibrated_phase_count,
        uncalibrated_queue_component_count,
        extrapolated_fit_count,
        coarse_topology,
        approximate_queueing,
        uncalibrated_runtime,
    }
}

fn approximation_count_entries(
    counts: BTreeMap<String, u32>,
    limit: usize,
) -> Vec<ServingApproximationCount> {
    let mut entries = counts
        .into_iter()
        .map(|(name, count)| ServingApproximationCount { name, count })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.name.cmp(&right.name))
    });
    entries.truncate(limit);
    entries
}

fn push_downstream_prefill_backpressure_approximations(
    approximations: &mut Vec<SimulationApproximation>,
    simulation: &ServingSimulation,
) {
    for service in &simulation.service_observations {
        let queue_cap_hits = service
            .queue_cap_hit_count
            .saturating_add(service.decode_iteration_queue_cap_hit_count);
        let pressure_events = service
            .backpressure_rejections
            .saturating_add(queue_cap_hits);
        if pressure_events == 0 {
            continue;
        }
        match service.phase.as_str() {
            "kv_transfer" => push_serving_approximation(
                approximations,
                SimulationApproximation::new(
                    "prefill",
                    "backpressure",
                    "kv_transfer_service",
                    "kv_transfer_to_prefill_backpressure_not_modeled",
                    format!(
                        "KV-transfer service reported {pressure_events} queue-cap/backpressure event(s), but the current serving timeline handles them at KV handoff instead of feeding KV pressure back into prefill admission or scheduling."
                    ),
                    Some(
                        "add downstream-to-prefill backpressure state so KV route or KV worker pressure can throttle prefill before prefill work and KV allocation are committed"
                            .to_string(),
                    ),
                ),
            ),
            "decode" => push_serving_approximation(
                approximations,
                SimulationApproximation::new(
                    "prefill",
                    "backpressure",
                    "decode_service",
                    "decode_to_prefill_backpressure_not_modeled",
                    format!(
                        "Decode service reported {pressure_events} queue-cap/backpressure event(s), but the current serving timeline handles them at decode admission/iteration time instead of feeding decode pressure back into prefill admission or scheduling."
                    ),
                    Some(
                        "add downstream-to-prefill backpressure state so decode queue pressure, active decode sets, and tail-token stalls can throttle prefill before new KV is produced"
                            .to_string(),
                    ),
                ),
            ),
            _ => {}
        }
    }
}

fn push_calibration_profile_provenance_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    profile: &CalibrationProfileMetadata,
) {
    let mut missing = Vec::new();
    if non_empty_metadata(profile.source.as_deref()).is_none() {
        missing.push("source");
    }
    if non_empty_metadata(profile.date.as_deref()).is_none() {
        missing.push("date");
    }
    if missing.is_empty() {
        return;
    }

    push_serving_approximation(
        approximations,
        SimulationApproximation::new(
            "serving",
            "calibration",
            "calibration_profile",
            "calibration_profile_provenance_incomplete",
            format!(
                "The loaded calibration profile is missing {} metadata, so benchmark provenance is not fully auditable from the result.",
                missing.join(" and ")
            ),
            Some(
                "set profile.source and profile.date in calibration profiles, and preserve benchmark command/source metadata for reproducible calibration"
                    .to_string(),
            ),
        ),
    );
}

fn push_serving_fit_metadata_approximations(
    approximations: &mut Vec<SimulationApproximation>,
    calibration_fits: &[CalibrationFitApplication],
) {
    if calibration_fits.is_empty() {
        return;
    }

    if calibration_fits
        .iter()
        .any(|fit| fit.sample_count.is_none())
    {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "fitted_models",
                "serving_calibration_fit_sample_count_unspecified",
                "At least one applied serving calibration fit does not report sample_count, so fit coverage strength is not visible in candidate evidence.",
                Some(
                    "include sample_count on calibration fits and collect enough benchmark points to support the fitted model"
                        .to_string(),
                ),
            ),
        );
    }

    if calibration_fits
        .iter()
        .any(|fit| fit.validation_sample_count.is_none())
    {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "fitted_models",
                "serving_calibration_fit_holdout_unspecified",
                "At least one applied serving calibration fit does not report validation_sample_count, so holdout validation coverage is not visible in candidate evidence.",
                Some(
                    "include validation_sample_count or separate holdout metrics for each calibration fit before relying on fit quality for capacity decisions"
                        .to_string(),
                ),
            ),
        );
    }

    if calibration_fits
        .iter()
        .any(|fit| non_empty_metadata(fit.source.as_deref()).is_none())
    {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "fitted_models",
                "serving_calibration_fit_source_unspecified",
                "At least one applied serving calibration fit does not report source metadata, so the benchmark artifact or command behind the fit is not auditable from the result.",
                Some(
                    "include source metadata on calibration fits, such as benchmark suite, command, artifact URI, or measurement run id"
                        .to_string(),
                ),
            ),
        );
    }

    if calibration_fits
        .iter()
        .any(|fit| !fit_has_numeric_uncertainty(fit) && fit.uncertainty_source.is_none())
    {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "fitted_models",
                "serving_calibration_fit_uncertainty_unspecified",
                "At least one applied serving calibration fit does not report uncertainty metadata such as RMSE, relative error, or absolute error.",
                Some(
                    "include rmse, rmse_pct, mean_abs_pct_error, or max_abs_pct_error on calibration fits so ranking uncertainty can be propagated"
                        .to_string(),
                ),
            ),
        );
    }
}

fn push_uncalibrated_serving_phase_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    phase: &'static str,
    phase_seconds: f64,
    calibration_fits: &[CalibrationFitApplication],
) {
    if !phase_seconds.is_finite()
        || phase_seconds <= 0.0
        || calibration_phase_has_fit(calibration_fits, phase)
    {
        return;
    }

    push_serving_approximation(
        approximations,
        SimulationApproximation::new(
            phase,
            "calibration",
            "calibration_profile",
            "serving_phase_uncalibrated",
            format!(
                "A calibration profile is loaded, but no applied fit calibrated the active {phase} serving phase; this phase still uses the simulator's coarse estimate."
            ),
            Some(format!(
                "add a {phase} fit to the calibration profile or reject serving_phase_uncalibrated through approximation_policy for calibrated-only comparisons"
            )),
        ),
    );
}

fn push_uncalibrated_serving_queue_component_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    phase: &'static str,
    component: &'static str,
    component_seconds: f64,
    calibration_fits: &[CalibrationFitApplication],
) {
    if !calibration_component_active(component_seconds)
        || calibration_phase_has_fit(calibration_fits, component)
    {
        return;
    }

    push_serving_approximation(
        approximations,
        SimulationApproximation::new(
            phase,
            "calibration",
            component,
            "serving_queue_component_uncalibrated",
            format!(
                "A calibration profile is loaded, but no applied fit calibrated active {component} delay; this queue component still comes from the simulator's approximate event timeline."
            ),
            Some(
                "treat this component as approximation evidence, collect serving traces for queueing behavior, or reject serving_queue_component_uncalibrated through approximation_policy for calibrated-only comparisons"
                    .to_string(),
            ),
        ),
    );
}

fn is_serving_metric_fit_application(fit: &CalibrationFitApplication) -> bool {
    fit.phase == "serving"
        && matches!(
            fit_target_family(&fit.target),
            Some("ttft" | "tpot" | "throughput" | "e2el")
        )
}

fn fit_target_family(target: &str) -> Option<&'static str> {
    let normalized = target
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    if normalized.starts_with("ttft") || normalized.contains("time_to_first_token") {
        Some("ttft")
    } else if normalized.starts_with("tpot") || normalized.contains("time_per_output_token") {
        Some("tpot")
    } else if normalized.contains("throughput") || normalized.contains("tokens_per_s") {
        Some("throughput")
    } else if normalized.starts_with("e2el") || normalized.contains("end_to_end") {
        Some("e2el")
    } else {
        None
    }
}

fn calibration_phase_has_fit(calibration_fits: &[CalibrationFitApplication], phase: &str) -> bool {
    calibration_fits.iter().any(|fit| fit.phase == phase)
}

fn push_calibration_profile_topology_approximations(
    approximations: &mut Vec<SimulationApproximation>,
    profile: &CalibrationProfileMetadata,
    cluster: &Cluster,
) {
    push_calibration_profile_hardware_approximation(approximations, profile, cluster);
    if cluster.nodes.len() > 1 {
        push_calibration_profile_fabric_approximation(approximations, profile, cluster);
    }
}

fn push_calibration_profile_hardware_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    profile: &CalibrationProfileMetadata,
    cluster: &Cluster,
) {
    let Some(profile_hardware) = non_empty_metadata(profile.hardware.as_deref()) else {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_hardware_unspecified",
                "The loaded calibration profile does not declare hardware, so hardware-specific compute, memory bandwidth, and KV-transfer fits are implicit.",
                Some(
                    "set profile.hardware in the calibration profile, for example h100_sxm, h200_sxm, b200, mi300x, or a cluster-specific hardware label"
                        .to_string(),
                ),
            ),
        );
        return;
    };

    let cluster_terms = cluster_hardware_terms(cluster);
    if !cluster_terms.is_empty() && !metadata_value_matches_terms(profile_hardware, &cluster_terms)
    {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_hardware_mismatch",
                format!(
                    "The calibration profile declares hardware '{}' but the candidate cluster exposes hardware such as {}, so fitted serving behavior may not transfer cleanly.",
                    profile_hardware,
                    summarize_metadata_labels(&cluster_hardware_labels(cluster))
                ),
                Some(
                    "use a calibration profile measured on matching accelerator hardware, split profiles by hardware family, or reject this approximation for calibrated-only comparisons"
                        .to_string(),
                ),
            ),
        );
    }
}

fn push_calibration_profile_fabric_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    profile: &CalibrationProfileMetadata,
    cluster: &Cluster,
) {
    let Some(profile_fabric) = non_empty_metadata(profile.fabric.as_deref()) else {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_fabric_unspecified",
                "The loaded calibration profile does not declare inter-node fabric, so collective, KV-transfer, and routing fits are not tied to a measured network.",
                Some(
                    "set profile.fabric in the calibration profile, for example ib_ndr, rocev2_400g, ethernet_800g, or a cluster-specific fabric label"
                        .to_string(),
                ),
            ),
        );
        return;
    };

    let cluster_terms = cluster_fabric_terms(cluster);
    if !cluster_terms.is_empty() && !metadata_value_matches_terms(profile_fabric, &cluster_terms) {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_fabric_mismatch",
                format!(
                    "The calibration profile declares fabric '{}' but the candidate cluster exposes inter-node fabric such as {}, so collective and KV-transfer fits may not be applicable.",
                    profile_fabric,
                    summarize_metadata_labels(&cluster_fabric_labels(cluster))
                ),
                Some(
                    "use a calibration profile measured on matching inter-node fabric, split profiles by fabric family, or reject this approximation for topology-sensitive comparisons"
                        .to_string(),
                ),
            ),
        );
    }
}

fn push_calibration_profile_model_approximation(
    approximations: &mut Vec<SimulationApproximation>,
    profile: &CalibrationProfileMetadata,
    workload_model_id: Option<&str>,
    traffic: &ServingTraffic,
) {
    let Some(profile_model) = non_empty_metadata(profile.model.as_deref()) else {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_model_unspecified",
                "The loaded calibration profile does not declare a model or model family, so model-specific serving behavior is implicit.",
                Some(
                    "set profile.model in the calibration profile and set model.id, model.name, or model_id in the workload [model] section"
                        .to_string(),
                ),
            ),
        );
        return;
    };

    let workload_model_ids = workload_model_ids(workload_model_id, traffic);
    if workload_model_ids.is_empty() {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_model_unverified",
                format!(
                    "The calibration profile declares model '{}' but the workload [model] section has no id/name/model_id metadata, so model-family applicability cannot be checked.",
                    profile_model
                ),
                Some(
                    "set model.id, model.name, or model_id in the workload [model] section to match the calibration profile's model metadata"
                        .to_string(),
                ),
            ),
        );
        return;
    };

    let matching_model_count = workload_model_ids
        .iter()
        .filter(|model_id| model_metadata_values_match(profile_model, model_id))
        .count();
    if matching_model_count == 0 {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_model_mismatch",
                format!(
                    "The calibration profile declares model '{}' but the workload/trace model metadata is {}, so model-family-specific serving fits may not transfer cleanly.",
                    profile_model,
                    summarize_metadata_labels(&workload_model_ids)
                ),
                Some(
                    "use a calibration profile measured for the workload model family, or split profiles by model id/family before comparing runtime-sensitive candidates"
                        .to_string(),
                ),
            ),
        );
    }

    let trace_model_ids = trace_model_ids(traffic);
    if trace_model_ids.len() > 1 {
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_multi_model_trace",
                format!(
                    "The serving trace contains {} model ids ({}) but the run applies one calibration profile declaring model '{}'.",
                    trace_model_ids.len(),
                    summarize_metadata_labels(&trace_model_ids),
                    profile_model
                ),
                Some(
                    "split mixed-model traces into per-model scenarios, add model-specific calibration profiles, or extend the simulator with multi-model serving calibration"
                        .to_string(),
                ),
            ),
        );
    }

    if matching_model_count > 0 && matching_model_count < workload_model_ids.len() {
        let unmatched = workload_model_ids
            .iter()
            .filter(|model_id| !model_metadata_values_match(profile_model, model_id))
            .cloned()
            .collect::<BTreeSet<_>>();
        push_serving_approximation(
            approximations,
            SimulationApproximation::new(
                "serving",
                "calibration",
                "calibration_profile",
                "calibration_profile_trace_model_mismatch",
                format!(
                    "The calibration profile declares model '{}' and matches part of the workload, but trace/workload model ids {} do not match the profile.",
                    profile_model,
                    summarize_metadata_labels(&unmatched)
                ),
                Some(
                    "split the trace by model id or attach per-model calibration profiles before making model-mix capacity claims"
                        .to_string(),
                ),
            ),
        );
    }
}

fn non_empty_metadata(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn metadata_value_matches_terms(value: &str, candidate_terms: &BTreeSet<String>) -> bool {
    metadata_terms(value)
        .iter()
        .any(|term| candidate_terms.contains(term))
}

fn model_metadata_values_match(profile_model: &str, workload_model_id: &str) -> bool {
    let profile_terms = model_metadata_terms(profile_model);
    let workload_terms = model_metadata_terms(workload_model_id);
    !profile_terms.is_empty()
        && profile_terms
            .iter()
            .any(|term| workload_terms.contains(term))
}

fn serving_stack_metadata_matches(profile_stack: &str, workload_stack: &str) -> bool {
    let profile_terms = metadata_terms(profile_stack);
    let workload_terms = metadata_terms(workload_stack);
    !profile_terms.is_empty()
        && profile_terms
            .iter()
            .any(|term| workload_terms.contains(term))
}

fn serving_runtime_feature_matches(profile_feature: &str, workload_feature: &str) -> bool {
    normalize_runtime_feature(profile_feature) == normalize_runtime_feature(workload_feature)
}

fn normalize_runtime_feature(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn workload_model_ids(
    workload_model_id: Option<&str>,
    traffic: &ServingTraffic,
) -> BTreeSet<String> {
    let mut model_ids = BTreeSet::new();
    if let Some(model_id) = non_empty_metadata(workload_model_id) {
        model_ids.insert(model_id.to_string());
    }
    model_ids.extend(trace_model_ids(traffic));
    model_ids
}

fn trace_model_ids(traffic: &ServingTraffic) -> BTreeSet<String> {
    traffic
        .trace_requests
        .iter()
        .filter_map(|request| non_empty_metadata(request.model_id.as_deref()).map(str::to_string))
        .collect()
}

fn model_metadata_terms(value: &str) -> BTreeSet<String> {
    metadata_terms(value)
        .into_iter()
        .filter(|term| !is_generic_model_metadata_term(term))
        .collect()
}

fn is_generic_model_metadata_term(term: &str) -> bool {
    matches!(
        term,
        "model"
            | "llm"
            | "lm"
            | "transformer"
            | "base"
            | "chat"
            | "instruct"
            | "sft"
            | "rlhf"
            | "fp16"
            | "f16"
            | "bf16"
            | "bfloat16"
            | "fp8"
            | "f8"
            | "int8"
            | "i8"
    )
}

fn cluster_hardware_terms(cluster: &Cluster) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    for node in cluster.nodes.values() {
        add_node_hardware_terms(&mut terms, node);
    }
    for group in cluster.node_groups.keys() {
        add_metadata_terms(&mut terms, group);
    }
    terms
}

fn cluster_hardware_labels(cluster: &Cluster) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();
    for node in cluster.nodes.values() {
        add_node_hardware_labels(&mut labels, node);
    }
    labels
}

fn add_node_hardware_terms(terms: &mut BTreeSet<String>, node: &Node) {
    for (gpu_id, gpu) in &node.gpus {
        let profile = node.gpu_profile(*gpu_id).unwrap_or_else(|| gpu.profile());
        add_metadata_terms(terms, profile.label);
        add_metadata_terms(terms, &format!("{gpu:?}"));
    }
}

fn add_node_hardware_labels(labels: &mut BTreeSet<String>, node: &Node) {
    for (gpu_id, gpu) in &node.gpus {
        let profile = node.gpu_profile(*gpu_id).unwrap_or_else(|| gpu.profile());
        labels.insert(profile.label.to_string());
    }
}

fn cluster_fabric_terms(cluster: &Cluster) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    for profile in cluster_fabric_profiles(cluster) {
        add_fabric_profile_terms(&mut terms, &profile);
    }
    terms
}

fn cluster_fabric_labels(cluster: &Cluster) -> BTreeSet<String> {
    cluster_fabric_profiles(cluster)
        .into_iter()
        .map(|profile| profile.label.to_string())
        .collect()
}

fn cluster_fabric_profiles(cluster: &Cluster) -> Vec<FabricProfile> {
    match &cluster.inter_node_topology {
        InterNodeTopology::FatTree { link, .. } | InterNodeTopology::Flat { link } => {
            vec![link.clone()]
        }
        InterNodeTopology::Custom(links) => links
            .values()
            .flat_map(|links| links.iter().map(|link| link.profile.clone()))
            .collect(),
    }
}

fn add_fabric_profile_terms(terms: &mut BTreeSet<String>, profile: &FabricProfile) {
    add_metadata_terms(terms, profile.label);
    match profile.kind {
        FabricKind::InfiniBand => {
            terms.insert("ib".to_string());
            terms.insert("infiniband".to_string());
        }
        FabricKind::RoCE => {
            terms.insert("roce".to_string());
            terms.insert("rocev2".to_string());
        }
        FabricKind::Ethernet => {
            terms.insert("eth".to_string());
            terms.insert("ethernet".to_string());
        }
    }

    let gbps = profile.bw.unidirectional.as_gigabits_per_sec().round();
    if gbps.is_finite() && gbps > 0.0 && gbps <= u64::MAX as f64 {
        terms.insert(format!("{}g", gbps as u64));
    }
}

fn metadata_terms(value: &str) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    let mut token = String::new();
    for c in value.chars() {
        if c.is_ascii_alphanumeric() {
            token.push(c.to_ascii_lowercase());
        } else if !token.is_empty() {
            insert_metadata_term(&mut terms, std::mem::take(&mut token));
        }
    }
    if !token.is_empty() {
        insert_metadata_term(&mut terms, token);
    }

    let compact = value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect::<String>();
    insert_metadata_term(&mut terms, compact);
    terms
}

fn add_metadata_terms(terms: &mut BTreeSet<String>, value: &str) {
    terms.extend(metadata_terms(value));
}

fn insert_metadata_term(terms: &mut BTreeSet<String>, term: String) {
    if term.is_empty() {
        return;
    }
    if let Some(family) = accelerator_family_term(&term) {
        terms.insert(family);
    }
    terms.insert(term);
}

fn accelerator_family_term(term: &str) -> Option<String> {
    let mut end = term.len();
    let mut removed_suffix = false;
    for (idx, c) in term.char_indices().rev() {
        if c.is_ascii_alphabetic() {
            end = idx;
            removed_suffix = true;
        } else {
            break;
        }
    }
    if !removed_suffix || end == 0 {
        return None;
    }
    let prefix = &term[..end];
    if prefix.chars().last().is_some_and(|c| c.is_ascii_digit())
        && prefix.chars().any(|c| c.is_ascii_alphabetic())
    {
        Some(prefix.to_string())
    } else {
        None
    }
}

fn summarize_metadata_labels(labels: &BTreeSet<String>) -> String {
    let summary = labels.iter().take(6).cloned().collect::<Vec<_>>();
    if summary.is_empty() {
        return "unknown".to_string();
    }
    let remaining = labels.len().saturating_sub(summary.len());
    let mut text = summary.join(", ");
    if remaining > 0 {
        text.push_str(&format!(", +{remaining} more"));
    }
    text
}

fn profile_dtype_matches_model(profile_dtype: &str, model_dtype: DType) -> bool {
    match normalize_profile_dtype(profile_dtype) {
        Some(normalized) => normalized == model_dtype_label(model_dtype),
        None => false,
    }
}

fn normalize_profile_dtype(profile_dtype: &str) -> Option<&'static str> {
    match profile_dtype
        .trim()
        .to_ascii_lowercase()
        .replace(['_', '-', ' '], "")
        .as_str()
    {
        "fp16" | "f16" | "float16" => Some("fp16"),
        "bf16" | "bfloat16" => Some("bf16"),
        "fp8" | "f8" | "float8" => Some("fp8"),
        "int8" | "i8" => Some("int8"),
        _ => None,
    }
}

fn model_dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::Fp16 => "fp16",
        DType::Bf16 => "bf16",
        DType::Fp8 => "fp8",
        DType::Int8 => "int8",
    }
}

fn steady_state_measurement_approximation(
    traffic: &ServingTraffic,
    simulation: &ServingSimulation,
) -> Option<SimulationApproximation> {
    if !simulation.measurement_window.steady_state_requested {
        return None;
    }

    let min_requests = traffic
        .measurement_steady_state_min_requests
        .unwrap_or(3)
        .max(1);
    let max_cv = traffic
        .measurement_steady_state_max_cv
        .unwrap_or(0.10)
        .max(0.0);
    let remediation = Some(
        "set measurement_start/end or warmup/cooldown for deterministic metric windows, or tune measurement_steady_state_min_requests and measurement_steady_state_max_cv"
            .to_string(),
    );

    if simulation.measurement_window.steady_state_applied {
        return Some(SimulationApproximation::new(
            "serving",
            "metrics",
            "measurement_window",
            "steady_state_measurement_window",
            format!(
                "Serving metrics use an auto-selected steady-state measurement window from {:.3}ms to {:.3}ms with {} measured completed requests (min_requests={}, max_cv={:.3}, selected_cv={}, worst_metric_cv={}).",
                simulation.measurement_window.start_s * 1_000.0,
                simulation.measurement_window.end_s * 1_000.0,
                simulation.metrics.measured_requests,
                min_requests,
                max_cv,
                simulation
                    .measurement_window
                    .steady_state_candidate_e2el_cv
                    .map(|cv| format!("{cv:.3}"))
                    .unwrap_or_else(|| "unknown".to_string()),
                simulation
                    .measurement_window
                    .steady_state_candidate_worst_cv
                    .map(|cv| format!(
                        "{}:{cv:.3}",
                        simulation
                            .measurement_window
                            .steady_state_candidate_worst_metric
                            .as_deref()
                            .unwrap_or("unknown")
                    ))
                    .unwrap_or_else(|| "unknown".to_string()),
            ),
            remediation,
        ));
    }

    if let (Some(start_s), Some(end_s)) = (
        simulation.measurement_window.steady_state_candidate_start_s,
        simulation.measurement_window.steady_state_candidate_end_s,
    ) {
        return Some(SimulationApproximation::new(
            "serving",
            "metrics",
            "measurement_window",
            "steady_state_measurement_window_config_overrides",
            format!(
                "Steady-state detection found a candidate window from {:.3}ms to {:.3}ms, but configured measurement bounds determine the reported metrics window.",
                start_s * 1_000.0,
                end_s * 1_000.0,
            ),
            remediation,
        ));
    }

    Some(SimulationApproximation::new(
        "serving",
        "metrics",
        "measurement_window",
        "steady_state_measurement_window_unavailable",
        format!(
            "Steady-state measurement was requested, but no completed-request latency window met min_requests={} and max_cv={:.3}; reported metrics use the configured/default measurement bounds.",
            min_requests, max_cv,
        ),
        remediation,
    ))
}

fn prefill_batching_label(batching: &ServingPrefillBatching) -> &'static str {
    match batching {
        ServingPrefillBatching::Independent => "independent",
        ServingPrefillBatching::Continuous { .. } => "continuous",
    }
}

fn decode_batching_label(batching: &ServingDecodeBatching) -> &'static str {
    match batching {
        ServingDecodeBatching::Independent => "independent",
        ServingDecodeBatching::Continuous { .. } => "continuous",
    }
}

fn is_disaggregated_pool(pool: &ResolvedServingPool) -> bool {
    let prefill: BTreeSet<_> = pool.prefill_nodes.iter().copied().collect();
    let decode: BTreeSet<_> = pool.decode_nodes.iter().copied().collect();
    prefill != decode
}

fn simulation_has_kv_handoff(simulation: &ServingSimulation) -> bool {
    simulation.request_observations.iter().any(|observation| {
        observation.kv_transfer_bytes > 0
            && node_set(&observation.prefill_route_nodes)
                != node_set(&observation.decode_route_nodes)
    })
}

fn node_set(nodes: &[NodeId]) -> BTreeSet<NodeId> {
    nodes.iter().copied().collect()
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

fn apply_serving_metric_calibration_fits(
    metrics: &mut ServingMetrics,
    calibration_profile: Option<&CalibrationProfileMetadata>,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
) -> Vec<CalibrationFitApplication> {
    if calibration_profile.is_none() {
        return Vec::new();
    }

    let baseline = *metrics;
    let mut applications = Vec::new();

    if let Some(application) = serving_metric_fit_application(
        calibration_profile,
        &[
            "ttft_ms",
            "ttft_latency_ms",
            "time_to_first_token_ms",
            "ttft_s",
            "time_to_first_token_s",
        ],
        "ttft",
        baseline.ttft_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    ) {
        apply_latency_metric_prediction(
            &mut metrics.ttft_s,
            &mut metrics.ttft_p50_s,
            &mut metrics.ttft_p90_s,
            &mut metrics.ttft_p95_s,
            &mut metrics.ttft_p99_s,
            &mut metrics.ttft_max_s,
            application.predicted_s,
        );
        applications.push(application);
    }

    if let Some(application) = serving_metric_fit_application(
        calibration_profile,
        &[
            "tpot_ms",
            "tpot_latency_ms",
            "time_per_output_token_ms",
            "tpot_s",
            "time_per_output_token_s",
        ],
        "tpot",
        baseline.tpot_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    ) {
        apply_latency_metric_prediction(
            &mut metrics.tpot_s,
            &mut metrics.tpot_p50_s,
            &mut metrics.tpot_p90_s,
            &mut metrics.tpot_p95_s,
            &mut metrics.tpot_p99_s,
            &mut metrics.tpot_max_s,
            application.predicted_s,
        );
        applications.push(application);
    }

    if let Some(application) = serving_metric_fit_application(
        calibration_profile,
        &[
            "e2el_ms",
            "e2el_latency_ms",
            "end_to_end_ms",
            "end_to_end_latency_ms",
            "e2el_s",
            "end_to_end_s",
        ],
        "e2el",
        baseline.e2el_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    ) {
        apply_latency_metric_prediction(
            &mut metrics.e2el_s,
            &mut metrics.e2el_p50_s,
            &mut metrics.e2el_p90_s,
            &mut metrics.e2el_p95_s,
            &mut metrics.e2el_p99_s,
            &mut metrics.e2el_max_s,
            application.predicted_s,
        );
        applications.push(application);
    }

    if let Some(application) = serving_metric_value_fit_application(
        calibration_profile,
        &[
            "throughput_tokens_per_s",
            "output_throughput_tokens_per_s",
            "throughput",
            "tokens_per_s",
        ],
        "throughput",
        baseline.throughput_tokens_per_s,
        &baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
        "throughput",
        Some("tokens/s"),
    ) {
        metrics.throughput_tokens_per_s = application.predicted_value;
        applications.push(application);
    }

    applications
}

#[allow(clippy::too_many_arguments)]
fn serving_metric_fit_application(
    calibration_profile: Option<&CalibrationProfileMetadata>,
    targets: &[&str],
    target_metric: &str,
    baseline_s: f64,
    baseline: &ServingMetrics,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
) -> Option<CalibrationFitApplication> {
    if !baseline_s.is_finite() || baseline_s <= 0.0 {
        return None;
    }
    let features = serving_metric_fit_features(
        target_metric,
        baseline_s,
        baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    );
    Solver::fitted_latency_from_features(
        calibration_profile,
        "serving",
        targets,
        &features,
        Some(baseline_s),
    )
}

#[allow(clippy::too_many_arguments)]
fn serving_metric_value_fit_application(
    calibration_profile: Option<&CalibrationProfileMetadata>,
    targets: &[&str],
    target_metric: &str,
    baseline_value: f64,
    baseline: &ServingMetrics,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
    prediction_kind: &str,
    prediction_unit: Option<&str>,
) -> Option<CalibrationFitApplication> {
    if !baseline_value.is_finite() || baseline_value <= 0.0 {
        return None;
    }
    let features = serving_metric_fit_features(
        target_metric,
        baseline_value,
        baseline,
        model,
        request,
        traffic,
        prefill_config,
        decode_config,
    );
    Solver::fitted_value_from_features(
        calibration_profile,
        "serving",
        targets,
        &features,
        Some(baseline_value),
        prediction_kind,
        prediction_unit,
    )
}

#[allow(clippy::too_many_arguments)]
fn serving_metric_fit_features(
    target_metric: &str,
    target_baseline_value: f64,
    baseline: &ServingMetrics,
    model: &ModelSpec,
    request: &InferenceRequest,
    traffic: &ServingTraffic,
    prefill_config: ParallelismConfig,
    decode_config: ParallelismConfig,
) -> BTreeMap<String, f64> {
    let mut features = BTreeMap::new();
    insert_serving_fit_feature(&mut features, "baseline_value", target_baseline_value);
    insert_serving_fit_feature(
        &mut features,
        &format!("{target_metric}_baseline_value"),
        target_baseline_value,
    );
    insert_serving_fit_feature(&mut features, "baseline_s", target_baseline_value);
    insert_serving_fit_feature(&mut features, "baseline_ms", target_baseline_value * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        &format!("{target_metric}_baseline_s"),
        target_baseline_value,
    );
    insert_serving_fit_feature(
        &mut features,
        &format!("{target_metric}_baseline_ms"),
        target_baseline_value * 1000.0,
    );
    if target_metric == "throughput" {
        insert_serving_fit_feature(
            &mut features,
            "baseline_throughput_tokens_per_s",
            target_baseline_value,
        );
        insert_serving_fit_feature(
            &mut features,
            "throughput_baseline_tokens_per_s",
            target_baseline_value,
        );
    }

    insert_serving_fit_feature(&mut features, "batch_size", f64::from(request.batch_size));
    insert_serving_fit_feature(
        &mut features,
        "prompt_tokens",
        f64::from(request.prompt_tokens),
    );
    insert_serving_fit_feature(
        &mut features,
        "decode_tokens",
        f64::from(request.decode_tokens),
    );
    insert_serving_fit_feature(
        &mut features,
        "sequence_tokens",
        f64::from(request.prompt_tokens.saturating_add(request.decode_tokens)),
    );
    insert_serving_fit_feature(
        &mut features,
        "max_sequence_tokens",
        f64::from(request.max_sequence_tokens),
    );
    insert_serving_fit_feature(&mut features, "model_layers", f64::from(model.layers));
    insert_serving_fit_feature(
        &mut features,
        "model_hidden_size",
        f64::from(model.hidden_size),
    );
    insert_serving_fit_feature(
        &mut features,
        "model_parameters_gb",
        model.parameters.as_gigabytes(),
    );
    insert_serving_fit_feature(
        &mut features,
        "model_parameter_count_billion",
        model.parameter_count_billion(),
    );
    insert_serving_fit_feature(
        &mut features,
        "request_count",
        f64::from(traffic.request_count.unwrap_or(baseline.scheduled_requests)),
    );
    insert_serving_fit_feature(
        &mut features,
        "scheduled_requests",
        f64::from(baseline.scheduled_requests),
    );
    insert_serving_fit_feature(
        &mut features,
        "admitted_requests",
        f64::from(baseline.admitted_requests),
    );
    insert_serving_fit_feature(
        &mut features,
        "completed_requests",
        f64::from(baseline.completed_requests),
    );
    insert_serving_fit_feature(
        &mut features,
        "measured_requests",
        f64::from(baseline.measured_requests),
    );
    if let Some(arrival_rate_per_s) = serving_arrival_rate_feature(traffic) {
        insert_serving_fit_feature(&mut features, "arrival_rate_per_s", arrival_rate_per_s);
        insert_serving_fit_feature(&mut features, "request_rate_per_s", arrival_rate_per_s);
    }

    insert_serving_fit_feature(
        &mut features,
        "prefill_tensor_ranks",
        f64::from(prefill_config.tensor_ranks),
    );
    insert_serving_fit_feature(
        &mut features,
        "prefill_pipeline_ranks",
        f64::from(prefill_config.pipeline_ranks),
    );
    insert_serving_fit_feature(
        &mut features,
        "decode_tensor_ranks",
        f64::from(decode_config.tensor_ranks),
    );
    insert_serving_fit_feature(
        &mut features,
        "decode_pipeline_ranks",
        f64::from(decode_config.pipeline_ranks),
    );

    insert_serving_fit_feature(&mut features, "simulated_ttft_s", baseline.ttft_s);
    insert_serving_fit_feature(&mut features, "simulated_ttft_ms", baseline.ttft_s * 1000.0);
    insert_serving_fit_feature(&mut features, "simulated_tpot_s", baseline.tpot_s);
    insert_serving_fit_feature(&mut features, "simulated_tpot_ms", baseline.tpot_s * 1000.0);
    insert_serving_fit_feature(&mut features, "simulated_e2el_s", baseline.e2el_s);
    insert_serving_fit_feature(&mut features, "simulated_e2el_ms", baseline.e2el_s * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        "throughput_tokens_per_s",
        baseline.throughput_tokens_per_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "simulated_throughput_tokens_per_s",
        baseline.throughput_tokens_per_s,
    );
    insert_serving_fit_feature(&mut features, "prefill_s", baseline.prefill_s);
    insert_serving_fit_feature(&mut features, "prefill_ms", baseline.prefill_s * 1000.0);
    insert_serving_fit_feature(&mut features, "decode_s", baseline.decode_s);
    insert_serving_fit_feature(&mut features, "decode_ms", baseline.decode_s * 1000.0);
    insert_serving_fit_feature(&mut features, "kv_transfer_s", baseline.kv_transfer_s);
    insert_serving_fit_feature(
        &mut features,
        "kv_transfer_ms",
        baseline.kv_transfer_s * 1000.0,
    );
    insert_serving_fit_feature(&mut features, "queue_s", baseline.queue_delay_s);
    insert_serving_fit_feature(&mut features, "queue_ms", baseline.queue_delay_s * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        "prefill_queue_s",
        baseline.prefill_worker_queue_s + baseline.prefill_resource_queue_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "prefill_queue_ms",
        (baseline.prefill_worker_queue_s + baseline.prefill_resource_queue_s) * 1000.0,
    );
    insert_serving_fit_feature(&mut features, "decode_queue_s", baseline.decode_queue_s);
    insert_serving_fit_feature(
        &mut features,
        "decode_queue_ms",
        baseline.decode_queue_s * 1000.0,
    );
    insert_serving_fit_feature(&mut features, "kv_queue_s", baseline.kv_queue_s);
    insert_serving_fit_feature(&mut features, "kv_queue_ms", baseline.kv_queue_s * 1000.0);
    insert_serving_fit_feature(
        &mut features,
        "kv_worker_queue_s",
        baseline.kv_worker_queue_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "kv_worker_queue_ms",
        baseline.kv_worker_queue_s * 1000.0,
    );
    insert_serving_fit_feature(
        &mut features,
        "kv_route_resource_queue_s",
        baseline.kv_resource_queue_s,
    );
    insert_serving_fit_feature(
        &mut features,
        "kv_route_resource_queue_ms",
        baseline.kv_resource_queue_s * 1000.0,
    );
    insert_serving_fit_feature(
        &mut features,
        "output_tokens",
        baseline.decode_iterations as f64 * f64::from(request.batch_size.max(1)),
    );
    insert_serving_fit_feature(
        &mut features,
        "effective_prefill_tokens",
        baseline.effective_prefill_tokens as f64,
    );

    features
}

fn serving_arrival_rate_feature(traffic: &ServingTraffic) -> Option<f64> {
    match traffic.arrival {
        ServingArrivalPattern::FixedGap => traffic
            .arrival_gap_s
            .filter(|gap_s| gap_s.is_finite() && *gap_s > 0.0)
            .map(|gap_s| 1.0 / gap_s),
        ServingArrivalPattern::Poisson { rate_per_s, .. }
        | ServingArrivalPattern::SelfSimilar { rate_per_s, .. } => {
            (rate_per_s.is_finite() && rate_per_s > 0.0).then_some(rate_per_s)
        }
        ServingArrivalPattern::Bursty {
            burst_size,
            burst_interval_s,
            ..
        } => (burst_interval_s.is_finite() && burst_interval_s > 0.0)
            .then_some(f64::from(burst_size) / burst_interval_s),
        ServingArrivalPattern::Diurnal {
            min_rate_per_s,
            max_rate_per_s,
            ..
        } => {
            let mean = (min_rate_per_s + max_rate_per_s) / 2.0;
            (mean.is_finite() && mean > 0.0).then_some(mean)
        }
        ServingArrivalPattern::TraceDerived => None,
    }
}

fn insert_serving_fit_feature(features: &mut BTreeMap<String, f64>, name: &str, value: f64) {
    if value.is_finite() {
        features.insert(name.to_string(), value);
    }
}

fn apply_latency_metric_prediction(
    mean_s: &mut f64,
    p50_s: &mut f64,
    p90_s: &mut f64,
    p95_s: &mut f64,
    p99_s: &mut f64,
    max_s: &mut f64,
    predicted_s: f64,
) {
    if !predicted_s.is_finite() || predicted_s <= 0.0 || !mean_s.is_finite() || *mean_s <= 0.0 {
        return;
    }
    let delta_s = predicted_s - *mean_s;
    *mean_s = predicted_s;
    for value in [p50_s, p90_s, p95_s, p99_s, max_s] {
        if value.is_finite() {
            *value = (*value + delta_s).max(0.0);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SteadyStateMeasurementSearch {
    selected: Option<SteadyStateMeasurementWindow>,
    sample_count: u32,
    matching_window_count: u32,
    min_requests: u32,
    max_cv: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct SteadyStateMeasurementWindow {
    start_s: f64,
    end_s: f64,
    request_count: u32,
    mean_e2el_s: f64,
    stddev_e2el_s: f64,
    cv: f64,
    std_error_s: f64,
    metric_count: u32,
    worst_metric: Option<String>,
    worst_cv: Option<f64>,
    output_tokens: u64,
    throughput_tokens_per_s: f64,
    metrics: Vec<ServingSteadyStateMetricObservation>,
}

fn effective_measurement_window(
    traffic: &ServingTraffic,
    observations: &[ServingRequestObservation],
    operations: &[ScheduledOperation],
    makespan_s: f64,
) -> MeasurementWindowSelection {
    let has_start = traffic.measurement_start_s.is_some() || traffic.measurement_warmup_s.is_some();
    let has_end = traffic.measurement_end_s.is_some() || traffic.measurement_cooldown_s.is_some();
    let mut start_s = traffic
        .measurement_start_s
        .or(traffic.measurement_warmup_s)
        .unwrap_or(0.0)
        .max(0.0);
    let mut end_s = traffic.measurement_end_s.unwrap_or_else(|| {
        traffic
            .measurement_cooldown_s
            .map(|cooldown_s| (makespan_s - cooldown_s.max(0.0)).max(start_s))
            .unwrap_or(makespan_s)
    });
    let mut steady_state_applied = false;
    let mut steady_state_candidate_start_s = None;
    let mut steady_state_candidate_end_s = None;
    let mut steady_state_min_requests = None;
    let mut steady_state_max_cv = None;
    let mut steady_state_sample_count = 0;
    let mut steady_state_matching_window_count = 0;
    let mut steady_state_candidate_request_count = None;
    let mut steady_state_candidate_e2el_mean_s = None;
    let mut steady_state_candidate_e2el_stddev_s = None;
    let mut steady_state_candidate_e2el_cv = None;
    let mut steady_state_candidate_e2el_std_error_s = None;
    let mut steady_state_candidate_metric_count = 0;
    let mut steady_state_candidate_worst_metric = None;
    let mut steady_state_candidate_worst_cv = None;
    let mut steady_state_candidate_output_tokens = None;
    let mut steady_state_candidate_throughput_tokens_per_s = None;
    let mut steady_state_candidate_metrics = Vec::new();
    let mut steady_state_candidate_utilization_count = 0;
    let mut steady_state_candidate_worst_utilization_resource = None;
    let mut steady_state_candidate_worst_utilization_cv = None;
    let mut steady_state_candidate_utilization = Vec::new();

    if traffic.measurement_steady_state {
        let steady_state_search = steady_state_measurement_window(
            observations,
            traffic.measurement_steady_state_min_requests,
            traffic.measurement_steady_state_max_cv,
        );
        steady_state_min_requests = Some(steady_state_search.min_requests);
        steady_state_max_cv = Some(steady_state_search.max_cv);
        steady_state_sample_count = steady_state_search.sample_count;
        steady_state_matching_window_count = steady_state_search.matching_window_count;
        if let Some(window) = steady_state_search.selected {
            steady_state_candidate_start_s = Some(window.start_s);
            steady_state_candidate_end_s = Some(window.end_s);
            steady_state_candidate_request_count = Some(window.request_count);
            steady_state_candidate_e2el_mean_s = Some(window.mean_e2el_s);
            steady_state_candidate_e2el_stddev_s = Some(window.stddev_e2el_s);
            steady_state_candidate_e2el_cv = Some(window.cv);
            steady_state_candidate_e2el_std_error_s = Some(window.std_error_s);
            steady_state_candidate_metric_count = window.metric_count;
            steady_state_candidate_worst_metric = window.worst_metric;
            steady_state_candidate_worst_cv = window.worst_cv;
            steady_state_candidate_output_tokens = Some(window.output_tokens);
            steady_state_candidate_throughput_tokens_per_s = Some(window.throughput_tokens_per_s);
            steady_state_candidate_metrics = window.metrics;
            steady_state_candidate_utilization = steady_state_candidate_utilization_diagnostics(
                observations,
                operations,
                traffic,
                window.start_s,
                window.end_s,
            );
            steady_state_candidate_utilization_count = steady_state_candidate_utilization
                .len()
                .min(u32::MAX as usize)
                as u32;
            if let Some(worst) = steady_state_candidate_utilization
                .iter()
                .filter(|utilization| utilization.utilization_cv.is_finite())
                .max_by(|left, right| left.utilization_cv.total_cmp(&right.utilization_cv))
            {
                steady_state_candidate_worst_utilization_resource = Some(worst.resource.clone());
                steady_state_candidate_worst_utilization_cv = Some(worst.utilization_cv);
            }
            if !has_start {
                start_s = start_s.max(window.start_s);
            }
            if !has_end {
                end_s = end_s.min(window.end_s);
            }
            steady_state_applied = !has_start || !has_end;
        }
    }

    MeasurementWindowSelection {
        start_s,
        end_s: end_s.max(start_s),
        configured_start_bound: has_start,
        configured_end_bound: has_end,
        steady_state_requested: traffic.measurement_steady_state,
        steady_state_applied,
        steady_state_candidate_start_s,
        steady_state_candidate_end_s,
        steady_state_min_requests,
        steady_state_max_cv,
        steady_state_sample_count,
        steady_state_matching_window_count,
        steady_state_candidate_request_count,
        steady_state_candidate_e2el_mean_s,
        steady_state_candidate_e2el_stddev_s,
        steady_state_candidate_e2el_cv,
        steady_state_candidate_e2el_std_error_s,
        steady_state_candidate_metric_count,
        steady_state_candidate_worst_metric,
        steady_state_candidate_worst_cv,
        steady_state_candidate_output_tokens,
        steady_state_candidate_throughput_tokens_per_s,
        steady_state_candidate_metrics,
        steady_state_candidate_utilization_count,
        steady_state_candidate_worst_utilization_resource,
        steady_state_candidate_worst_utilization_cv,
        steady_state_candidate_utilization,
    }
}

fn measurement_metric_source_counts(
    observations: &[ServingRequestObservation],
    measurement_start_s: f64,
    measurement_end_s: f64,
) -> Vec<ServingMeasurementMetricSourceCount> {
    if !measurement_start_s.is_finite() || !measurement_end_s.is_finite() {
        return Vec::new();
    }

    let mut counts = BTreeMap::<String, u32>::new();
    for observation in observations {
        if !observation.status.is_completed()
            || observation.arrival_s + 1e-12 < measurement_start_s
            || observation.arrival_s > measurement_end_s + 1e-12
        {
            continue;
        }
        let count = counts.entry(observation.metric_source.clone()).or_default();
        *count = count.saturating_add(1);
    }

    counts
        .into_iter()
        .map(
            |(metric_source, request_count)| ServingMeasurementMetricSourceCount {
                metric_source,
                request_count,
            },
        )
        .collect()
}

fn measurement_window_request_counts(
    observations: &[ServingRequestObservation],
    measurement_start_s: f64,
    measurement_end_s: f64,
) -> MeasurementWindowRequestCounts {
    if !measurement_start_s.is_finite() || !measurement_end_s.is_finite() {
        return MeasurementWindowRequestCounts::default();
    }

    let mut counts = MeasurementWindowRequestCounts::default();
    for observation in observations {
        if observation.arrival_s + 1e-12 < measurement_start_s
            || observation.arrival_s > measurement_end_s + 1e-12
        {
            continue;
        }

        counts.request_count = counts.request_count.saturating_add(1);
        if observation.status.is_completed() {
            counts.completed_request_count = counts.completed_request_count.saturating_add(1);
        } else {
            counts.failed_request_count = counts.failed_request_count.saturating_add(1);
        }
        match observation.status {
            ServingRequestStatus::RejectedAdmission => {
                counts.rejected_request_count = counts.rejected_request_count.saturating_add(1);
            }
            ServingRequestStatus::TimedOut => {
                counts.timed_out_request_count = counts.timed_out_request_count.saturating_add(1);
            }
            ServingRequestStatus::Cancelled => {
                counts.cancelled_request_count = counts.cancelled_request_count.saturating_add(1);
            }
            ServingRequestStatus::Pending | ServingRequestStatus::Completed => {}
        }
        if observation.deadline_s.is_some() {
            counts.deadline_constrained_request_count =
                counts.deadline_constrained_request_count.saturating_add(1);
        }
        if observation.deadline_missed {
            counts.deadline_missed_request_count =
                counts.deadline_missed_request_count.saturating_add(1);
        }
    }
    counts
}

fn steady_state_measurement_window(
    observations: &[ServingRequestObservation],
    min_requests: Option<u32>,
    max_cv: Option<f64>,
) -> SteadyStateMeasurementSearch {
    let samples = observations
        .iter()
        .filter(|observation| observation.status.is_completed())
        .filter(|observation| observation.arrival_s.is_finite() && observation.e2el_s.is_finite())
        .map(SteadyStateRequestSample::from_observation)
        .collect::<Vec<_>>();
    steady_state_measurement_window_from_request_samples(samples, min_requests, max_cv)
}

#[cfg(test)]
fn steady_state_measurement_window_from_samples(
    samples: Vec<(f64, f64)>,
    min_requests: Option<u32>,
    max_cv: Option<f64>,
) -> SteadyStateMeasurementSearch {
    let samples = samples
        .into_iter()
        .map(|(arrival_s, e2el_s)| SteadyStateRequestSample {
            arrival_s,
            completed_s: arrival_s + e2el_s.max(0.0),
            e2el_s,
            ttft_s: e2el_s,
            tpot_s: e2el_s,
            total_queue_s: 0.0,
            prefill_queue_s: 0.0,
            kv_queue_s: 0.0,
            decode_queue_s: 0.0,
            output_tokens: 1,
        })
        .collect();
    steady_state_measurement_window_from_request_samples(samples, min_requests, max_cv)
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct SteadyStateRequestSample {
    arrival_s: f64,
    completed_s: f64,
    e2el_s: f64,
    ttft_s: f64,
    tpot_s: f64,
    total_queue_s: f64,
    prefill_queue_s: f64,
    kv_queue_s: f64,
    decode_queue_s: f64,
    output_tokens: u64,
}

impl SteadyStateRequestSample {
    fn from_observation(observation: &ServingRequestObservation) -> Self {
        let prefill_queue_s =
            observation.prefill_worker_queue_s + observation.prefill_resource_queue_s;
        let total_queue_s = prefill_queue_s + observation.kv_queue_s + observation.decode_queue_s;
        Self {
            arrival_s: observation.arrival_s,
            completed_s: observation.last_decode_finish_s,
            e2el_s: observation.e2el_s,
            ttft_s: observation.ttft_s,
            tpot_s: observation.tpot_s,
            total_queue_s,
            prefill_queue_s,
            kv_queue_s: observation.kv_queue_s,
            decode_queue_s: observation.decode_queue_s,
            output_tokens: observation_output_tokens(observation),
        }
    }
}

fn observation_output_tokens(observation: &ServingRequestObservation) -> u64 {
    if !observation.status.is_completed() {
        return 0;
    }

    let decode_iterations = if observation.decode_iterations > 0 {
        observation.decode_iterations
    } else {
        observation.decode_tokens.max(1)
    };
    u64::from(observation.batch_size.max(1)).saturating_mul(u64::from(decode_iterations))
}

fn steady_state_measurement_window_from_request_samples(
    mut samples: Vec<SteadyStateRequestSample>,
    min_requests: Option<u32>,
    max_cv: Option<f64>,
) -> SteadyStateMeasurementSearch {
    samples.sort_by(|left, right| left.arrival_s.total_cmp(&right.arrival_s));

    let sample_count = samples.len();
    let requested_min_requests = min_requests.unwrap_or(3).max(1);
    let max_cv = max_cv.unwrap_or(0.10).max(0.0);
    if sample_count == 0 {
        return SteadyStateMeasurementSearch {
            selected: None,
            sample_count: 0,
            matching_window_count: 0,
            min_requests: requested_min_requests,
            max_cv,
        };
    }
    let effective_min_requests =
        requested_min_requests.min(sample_count.min(u32::MAX as usize) as u32) as usize;
    if sample_count < effective_min_requests {
        return SteadyStateMeasurementSearch {
            selected: None,
            sample_count: sample_count.min(u32::MAX as usize) as u32,
            matching_window_count: 0,
            min_requests: requested_min_requests,
            max_cv,
        };
    }

    let mut prefix_sum = vec![0.0; sample_count + 1];
    let mut prefix_sq_sum = vec![0.0; sample_count + 1];
    for (idx, sample) in samples.iter().copied().enumerate() {
        prefix_sum[idx + 1] = prefix_sum[idx] + sample.e2el_s;
        prefix_sq_sum[idx + 1] = prefix_sq_sum[idx] + sample.e2el_s * sample.e2el_s;
    }

    let mut matching_window_count = 0_u32;
    let mut best: Option<(usize, f64, f64, f64, usize, usize)> = None;
    for start_idx in 0..sample_count {
        for end_idx in start_idx + effective_min_requests - 1..sample_count {
            let count = end_idx - start_idx + 1;
            let sum = prefix_sum[end_idx + 1] - prefix_sum[start_idx];
            let sq_sum = prefix_sq_sum[end_idx + 1] - prefix_sq_sum[start_idx];
            let mean = sum / count as f64;
            if !mean.is_finite() || mean <= 0.0 {
                continue;
            }
            let variance = (sq_sum / count as f64 - mean * mean).max(0.0);
            let stddev = variance.sqrt();
            let cv = stddev / mean;
            if !cv.is_finite() || cv > max_cv + 1e-12 {
                continue;
            }
            matching_window_count = matching_window_count.saturating_add(1);

            let replace = best.is_none_or(|(best_count, best_cv, _, _, best_start, best_end)| {
                count > best_count
                    || (count == best_count
                        && (cv < best_cv
                            || ((cv - best_cv).abs() <= 1e-12
                                && centered_distance(start_idx, end_idx, sample_count)
                                    < centered_distance(best_start, best_end, sample_count))))
            });
            if replace {
                best = Some((count, cv, mean, stddev, start_idx, end_idx));
            }
        }
    }

    let selected = best.map(|(count, cv, mean, stddev, start_idx, end_idx)| {
        let diagnostics = steady_state_candidate_diagnostics(&samples, start_idx, end_idx);
        SteadyStateMeasurementWindow {
            start_s: samples[start_idx].arrival_s,
            end_s: samples[end_idx].arrival_s,
            request_count: count.min(u32::MAX as usize) as u32,
            mean_e2el_s: mean,
            stddev_e2el_s: stddev,
            cv,
            std_error_s: stddev / (count as f64).sqrt(),
            metric_count: diagnostics.metric_count,
            worst_metric: diagnostics.worst_metric,
            worst_cv: diagnostics.worst_cv,
            output_tokens: diagnostics.output_tokens,
            throughput_tokens_per_s: diagnostics.throughput_tokens_per_s,
            metrics: diagnostics.metrics,
        }
    });
    SteadyStateMeasurementSearch {
        selected,
        sample_count: sample_count.min(u32::MAX as usize) as u32,
        matching_window_count,
        min_requests: requested_min_requests,
        max_cv,
    }
}

struct SteadyStateCandidateDiagnostics {
    metric_count: u32,
    worst_metric: Option<String>,
    worst_cv: Option<f64>,
    output_tokens: u64,
    throughput_tokens_per_s: f64,
    metrics: Vec<ServingSteadyStateMetricObservation>,
}

fn steady_state_candidate_diagnostics(
    samples: &[SteadyStateRequestSample],
    start_idx: usize,
    end_idx: usize,
) -> SteadyStateCandidateDiagnostics {
    let window = &samples[start_idx..=end_idx];
    let mut metrics = Vec::new();
    push_steady_state_metric(
        &mut metrics,
        "e2el",
        "s",
        window.iter().map(|sample| sample.e2el_s),
    );
    push_steady_state_metric(
        &mut metrics,
        "ttft",
        "s",
        window.iter().map(|sample| sample.ttft_s),
    );
    push_steady_state_metric(
        &mut metrics,
        "tpot",
        "s",
        window.iter().map(|sample| sample.tpot_s),
    );
    push_steady_state_metric(
        &mut metrics,
        "total_queue",
        "s",
        window.iter().map(|sample| sample.total_queue_s),
    );
    push_steady_state_metric(
        &mut metrics,
        "prefill_queue",
        "s",
        window.iter().map(|sample| sample.prefill_queue_s),
    );
    push_steady_state_metric(
        &mut metrics,
        "kv_queue",
        "s",
        window.iter().map(|sample| sample.kv_queue_s),
    );
    push_steady_state_metric(
        &mut metrics,
        "decode_queue",
        "s",
        window.iter().map(|sample| sample.decode_queue_s),
    );
    push_steady_state_metric(
        &mut metrics,
        "request_throughput",
        "tokens/s",
        window.iter().filter_map(|sample| {
            (sample.e2el_s.is_finite() && sample.e2el_s > 0.0)
                .then_some(sample.output_tokens as f64 / sample.e2el_s)
        }),
    );

    let output_tokens = window.iter().fold(0_u64, |total, sample| {
        total.saturating_add(sample.output_tokens)
    });
    let first_arrival_s = window
        .iter()
        .map(|sample| sample.arrival_s)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let last_completion_s = window
        .iter()
        .filter_map(|sample| sample.completed_s.is_finite().then_some(sample.completed_s))
        .max_by(f64::total_cmp)
        .unwrap_or(first_arrival_s);
    let duration_s = (last_completion_s - first_arrival_s).max(0.0);
    let throughput_tokens_per_s = if duration_s > 0.0 {
        output_tokens as f64 / duration_s
    } else {
        0.0
    };

    let worst = metrics
        .iter()
        .filter(|metric| metric.cv.is_finite())
        .max_by(|left, right| left.cv.total_cmp(&right.cv))
        .map(|metric| (metric.metric.clone(), metric.cv));

    SteadyStateCandidateDiagnostics {
        metric_count: metrics.len().min(u32::MAX as usize) as u32,
        worst_metric: worst.as_ref().map(|(metric, _)| metric.clone()),
        worst_cv: worst.map(|(_, cv)| cv),
        output_tokens,
        throughput_tokens_per_s,
        metrics,
    }
}

fn push_steady_state_metric(
    metrics: &mut Vec<ServingSteadyStateMetricObservation>,
    metric: &str,
    unit: &str,
    values: impl IntoIterator<Item = f64>,
) {
    if let Some(observation) = steady_state_metric_observation(metric, unit, values) {
        metrics.push(observation);
    }
}

fn steady_state_metric_observation(
    metric: &str,
    unit: &str,
    values: impl IntoIterator<Item = f64>,
) -> Option<ServingSteadyStateMetricObservation> {
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }

    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / count as f64;
    let stddev = variance.max(0.0).sqrt();
    let cv = if mean.abs() > 1e-12 {
        stddev / mean.abs()
    } else if stddev <= 1e-12 {
        0.0
    } else {
        f64::INFINITY
    };
    Some(ServingSteadyStateMetricObservation {
        metric: metric.to_string(),
        unit: unit.to_string(),
        sample_count: count.min(u32::MAX as usize) as u32,
        mean,
        stddev,
        cv,
        std_error: stddev / (count as f64).sqrt(),
    })
}

const STEADY_STATE_UTILIZATION_BUCKETS: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SteadyStateUtilizationKey {
    source: String,
    phase: String,
    resource_kind: String,
    resource: String,
}

struct SteadyStateUtilizationAccumulator {
    buckets: Vec<f64>,
    event_count: u32,
}

impl SteadyStateUtilizationAccumulator {
    fn new(bucket_count: usize) -> Self {
        Self {
            buckets: vec![0.0; bucket_count],
            event_count: 0,
        }
    }

    fn into_observation(
        self,
        key: SteadyStateUtilizationKey,
    ) -> Option<ServingSteadyStateUtilizationObservation> {
        if self.buckets.is_empty() {
            return None;
        }
        let mean = self.buckets.iter().sum::<f64>() / self.buckets.len() as f64;
        let variance = self
            .buckets
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.buckets.len() as f64;
        let stddev = variance.max(0.0).sqrt();
        let utilization_cv = if mean.abs() > 1e-12 {
            stddev / mean.abs()
        } else if stddev <= 1e-12 {
            0.0
        } else {
            f64::INFINITY
        };
        Some(ServingSteadyStateUtilizationObservation {
            source: key.source,
            phase: key.phase,
            resource_kind: key.resource_kind,
            resource: key.resource,
            bucket_count: self.buckets.len().min(u32::MAX as usize) as u32,
            active_bucket_count: self
                .buckets
                .iter()
                .filter(|utilization| **utilization > 1e-12)
                .count()
                .min(u32::MAX as usize) as u32,
            event_count: self.event_count,
            mean_utilization: mean,
            max_utilization: max_value(&self.buckets),
            utilization_cv,
        })
    }
}

fn steady_state_candidate_utilization_diagnostics(
    observations: &[ServingRequestObservation],
    operations: &[ScheduledOperation],
    traffic: &ServingTraffic,
    start_s: f64,
    end_s: f64,
) -> Vec<ServingSteadyStateUtilizationObservation> {
    if !start_s.is_finite() || !end_s.is_finite() || end_s <= start_s {
        return Vec::new();
    }

    let bucket_count = STEADY_STATE_UTILIZATION_BUCKETS;
    let bucket_width_s = (end_s - start_s) / bucket_count as f64;
    if bucket_width_s <= 0.0 || !bucket_width_s.is_finite() {
        return Vec::new();
    }

    let mut accumulators = BTreeMap::new();
    accumulate_scheduled_resource_utilization(
        &mut accumulators,
        operations,
        start_s,
        bucket_width_s,
        bucket_count,
    );
    accumulate_worker_slot_utilization(
        &mut accumulators,
        observations,
        traffic,
        start_s,
        bucket_width_s,
        bucket_count,
    );
    accumulate_kv_residency_utilization(
        &mut accumulators,
        observations,
        traffic,
        start_s,
        bucket_width_s,
        bucket_count,
    );

    let mut diagnostics = accumulators
        .into_iter()
        .filter_map(|(key, accumulator)| accumulator.into_observation(key))
        .collect::<Vec<_>>();
    diagnostics.sort_by(|left, right| {
        right
            .max_utilization
            .total_cmp(&left.max_utilization)
            .then_with(|| right.utilization_cv.total_cmp(&left.utilization_cv))
            .then_with(|| left.source.cmp(&right.source))
            .then_with(|| left.phase.cmp(&right.phase))
            .then_with(|| left.resource_kind.cmp(&right.resource_kind))
            .then_with(|| left.resource.cmp(&right.resource))
    });
    diagnostics
}

fn accumulate_scheduled_resource_utilization(
    accumulators: &mut BTreeMap<SteadyStateUtilizationKey, SteadyStateUtilizationAccumulator>,
    operations: &[ScheduledOperation],
    window_start_s: f64,
    bucket_width_s: f64,
    bucket_count: usize,
) {
    for operation in operations {
        for resource in &operation.resources {
            let key = SteadyStateUtilizationKey {
                source: "scheduled_resource".to_string(),
                phase: operation_phase(&operation.name).to_string(),
                resource_kind: resource_kind(resource).to_string(),
                resource: resource.clone(),
            };
            accumulate_utilization_interval(
                accumulators,
                key,
                window_start_s,
                bucket_width_s,
                bucket_count,
                operation.start_s,
                operation.finish_s,
                1.0,
                1.0,
            );
        }
    }
}

fn accumulate_worker_slot_utilization(
    accumulators: &mut BTreeMap<SteadyStateUtilizationKey, SteadyStateUtilizationAccumulator>,
    observations: &[ServingRequestObservation],
    traffic: &ServingTraffic,
    window_start_s: f64,
    bucket_width_s: f64,
    bucket_count: usize,
) {
    let prefill_slots = prefill_worker_slots_per_gpu(traffic).min(u32::MAX as usize) as u32;
    let decode_slots = decode_worker_slots_per_gpu(traffic).min(u32::MAX as usize) as u32;
    let kv_transfer_slots =
        kv_transfer_worker_slots_per_gpu(traffic).map(|slots| slots.min(u32::MAX as usize) as u32);

    for observation in observations {
        for assignment in &observation.worker_assignments {
            let configured_slots = phase_worker_slots(
                &assignment.phase,
                prefill_slots,
                decode_slots,
                kv_transfer_slots,
            )
            .max(1);
            let key = SteadyStateUtilizationKey {
                source: "worker_slot".to_string(),
                phase: assignment.phase.clone(),
                resource_kind: "worker_slot".to_string(),
                resource: format!(
                    "node {} gpu {}",
                    assignment.node_id, assignment.local_gpu_id
                ),
            };
            accumulate_utilization_interval(
                accumulators,
                key,
                window_start_s,
                bucket_width_s,
                bucket_count,
                assignment.start_s,
                assignment.finish_s,
                1.0,
                f64::from(configured_slots),
            );
        }
    }
}

fn accumulate_kv_residency_utilization(
    accumulators: &mut BTreeMap<SteadyStateUtilizationKey, SteadyStateUtilizationAccumulator>,
    observations: &[ServingRequestObservation],
    traffic: &ServingTraffic,
    window_start_s: f64,
    bucket_width_s: f64,
    bucket_count: usize,
) {
    for observation in observations {
        for ownership in &observation.kv_block_ownership {
            accumulate_kv_residency_metric(
                accumulators,
                window_start_s,
                bucket_width_s,
                bucket_count,
                "resident_tokens",
                "global".to_string(),
                ownership.allocated_at_s,
                ownership.released_at_s,
                ownership.resident_tokens as f64,
                traffic.max_resident_tokens.map(|capacity| capacity as f64),
            );
            accumulate_kv_residency_metric(
                accumulators,
                window_start_s,
                bucket_width_s,
                bucket_count,
                "kv_blocks",
                "global".to_string(),
                ownership.allocated_at_s,
                ownership.released_at_s,
                ownership.kv_blocks as f64,
                traffic.max_kv_blocks.map(|capacity| capacity as f64),
            );
            accumulate_kv_residency_metric(
                accumulators,
                window_start_s,
                bucket_width_s,
                bucket_count,
                "resident_tokens_per_node",
                format!("node {}", ownership.owner.node_id),
                ownership.allocated_at_s,
                ownership.released_at_s,
                ownership.resident_tokens as f64,
                traffic
                    .max_resident_tokens_per_node
                    .map(|capacity| capacity as f64),
            );
            accumulate_kv_residency_metric(
                accumulators,
                window_start_s,
                bucket_width_s,
                bucket_count,
                "kv_blocks_per_node",
                format!("node {}", ownership.owner.node_id),
                ownership.allocated_at_s,
                ownership.released_at_s,
                ownership.kv_blocks as f64,
                traffic
                    .max_kv_blocks_per_node
                    .map(|capacity| capacity as f64),
            );
            accumulate_kv_residency_metric(
                accumulators,
                window_start_s,
                bucket_width_s,
                bucket_count,
                "resident_tokens_per_gpu",
                format!(
                    "node {} gpu {}",
                    ownership.owner.node_id, ownership.owner.local_gpu_id
                ),
                ownership.allocated_at_s,
                ownership.released_at_s,
                ownership.resident_tokens as f64,
                traffic
                    .max_resident_tokens_per_gpu
                    .map(|capacity| capacity as f64),
            );
            accumulate_kv_residency_metric(
                accumulators,
                window_start_s,
                bucket_width_s,
                bucket_count,
                "kv_blocks_per_gpu",
                format!(
                    "node {} gpu {}",
                    ownership.owner.node_id, ownership.owner.local_gpu_id
                ),
                ownership.allocated_at_s,
                ownership.released_at_s,
                ownership.kv_blocks as f64,
                traffic
                    .max_kv_blocks_per_gpu
                    .map(|capacity| capacity as f64),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_kv_residency_metric(
    accumulators: &mut BTreeMap<SteadyStateUtilizationKey, SteadyStateUtilizationAccumulator>,
    window_start_s: f64,
    bucket_width_s: f64,
    bucket_count: usize,
    resource_kind: &str,
    resource: String,
    start_s: f64,
    finish_s: f64,
    active_units: f64,
    capacity: Option<f64>,
) {
    let Some(capacity) = capacity else {
        return;
    };
    if capacity <= 0.0 || active_units <= 0.0 {
        return;
    }
    let key = SteadyStateUtilizationKey {
        source: "kv_residency".to_string(),
        phase: "decode".to_string(),
        resource_kind: resource_kind.to_string(),
        resource,
    };
    accumulate_utilization_interval(
        accumulators,
        key,
        window_start_s,
        bucket_width_s,
        bucket_count,
        start_s,
        finish_s,
        active_units,
        capacity,
    );
}

#[allow(clippy::too_many_arguments)]
fn accumulate_utilization_interval(
    accumulators: &mut BTreeMap<SteadyStateUtilizationKey, SteadyStateUtilizationAccumulator>,
    key: SteadyStateUtilizationKey,
    window_start_s: f64,
    bucket_width_s: f64,
    bucket_count: usize,
    start_s: f64,
    finish_s: f64,
    active_units: f64,
    capacity_units: f64,
) {
    if !start_s.is_finite()
        || !finish_s.is_finite()
        || finish_s <= start_s
        || active_units <= 0.0
        || capacity_units <= 0.0
    {
        return;
    }

    let window_end_s = window_start_s + bucket_width_s * bucket_count as f64;
    let clipped_start_s = start_s.max(window_start_s);
    let clipped_finish_s = finish_s.min(window_end_s);
    if clipped_finish_s <= clipped_start_s {
        return;
    }

    let first_bucket = utilization_bucket_idx(
        clipped_start_s,
        window_start_s,
        bucket_width_s,
        bucket_count,
    );
    let last_bucket = utilization_bucket_idx(
        (clipped_finish_s - f64::EPSILON).max(window_start_s),
        window_start_s,
        bucket_width_s,
        bucket_count,
    );
    let entry = accumulators
        .entry(key)
        .or_insert_with(|| SteadyStateUtilizationAccumulator::new(bucket_count));
    entry.event_count = entry.event_count.saturating_add(1);
    for bucket_idx in first_bucket..=last_bucket {
        let bucket_start_s = window_start_s + bucket_idx as f64 * bucket_width_s;
        let bucket_finish_s = bucket_start_s + bucket_width_s;
        let overlap_s =
            (clipped_finish_s.min(bucket_finish_s) - clipped_start_s.max(bucket_start_s)).max(0.0);
        if overlap_s > 0.0 {
            entry.buckets[bucket_idx] +=
                overlap_s * active_units / (bucket_width_s * capacity_units);
        }
    }
}

fn utilization_bucket_idx(
    time_s: f64,
    window_start_s: f64,
    bucket_width_s: f64,
    bucket_count: usize,
) -> usize {
    (((time_s - window_start_s) / bucket_width_s).floor() as usize)
        .min(bucket_count.saturating_sub(1))
}

fn centered_distance(start_idx: usize, end_idx: usize, sample_count: usize) -> usize {
    let window_center_twice = start_idx + end_idx;
    let sample_center_twice = sample_count.saturating_sub(1);
    window_center_twice.abs_diff(sample_center_twice)
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

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRequestObservation {
    pub request_idx: u32,
    pub request_id: Option<String>,
    pub tenant: Option<String>,
    pub model_id: Option<String>,
    pub traffic_class: Option<String>,
    pub shape_profile: Option<String>,
    pub cache_key: Option<String>,
    pub status: ServingRequestStatus,
    pub status_time_s: Option<f64>,
    pub failure_reason: Option<String>,
    pub rejection: Option<ServingRejection>,
    pub priority: i32,
    pub slo: ServingRequestSlo,
    pub ttft_slo_missed: bool,
    pub tpot_slo_missed: bool,
    pub itl_slo_missed: bool,
    pub e2el_slo_missed: bool,
    pub deadline_s: Option<f64>,
    pub deadline_missed: bool,
    pub cancellation_s: Option<f64>,
    pub prefill_node: NodeId,
    pub decode_node: NodeId,
    pub prefill_route_nodes: Vec<NodeId>,
    pub prefill_route_gpus: Vec<GpuAddr>,
    pub decode_route_nodes: Vec<NodeId>,
    pub decode_route_gpus: Vec<GpuAddr>,
    pub kv_cache_owner_gpus: Vec<GpuAddr>,
    pub routing_policy: ServingRoutingPolicy,
    pub routing_candidate_count: u32,
    pub routing_routable_candidate_count: u32,
    pub routing_estimated_e2el_s: f64,
    pub routing_estimated_kv_transfer_s: f64,
    pub routing_estimated_kv_resource_wait_s: f64,
    pub routing_estimated_prefill_wait_s: f64,
    pub routing_estimated_decode_wait_s: f64,
    pub routing_reason: String,
    pub routing_candidates: Vec<ServingRouteCandidateObservation>,
    pub arrival_s: f64,
    pub batch_size: u32,
    pub prompt_tokens: u32,
    pub prefix_cache_hit_tokens: u32,
    pub effective_prefill_tokens: u32,
    pub prefill_chunks: u32,
    pub prefill_token_start_s: Vec<f64>,
    pub prefill_token_finish_s: Vec<f64>,
    pub decode_tokens: u32,
    pub kv_block_tokens: u32,
    pub kv_cache_blocks: u64,
    pub kv_allocated_tokens: u64,
    pub kv_fragmentation_tokens: u64,
    pub kv_block_ownership: Vec<ServingKvBlockOwnershipObservation>,
    pub prefill_start_s: f64,
    pub prefill_finish_s: f64,
    pub kv_start_s: f64,
    pub kv_finish_s: f64,
    pub first_decode_start_s: f64,
    pub first_decode_finish_s: f64,
    pub last_decode_finish_s: f64,
    pub decode_iterations: u32,
    pub decode_token_start_s: Vec<f64>,
    pub decode_token_finish_s: Vec<f64>,
    pub inter_token_latency_s: Vec<f64>,
    pub ttft_s: f64,
    pub tpot_s: f64,
    pub itl_s: f64,
    pub e2el_s: f64,
    pub service_s: f64,
    pub queue_delay_s: f64,
    pub prefill_s: f64,
    pub prefill_worker_queue_s: f64,
    pub prefill_resource_queue_s: f64,
    pub kv_queue_s: f64,
    pub kv_worker_queue_s: f64,
    pub kv_resource_queue_s: f64,
    pub kv_transfer_bytes: u64,
    pub kv_transfer_bottlenecks: Vec<String>,
    pub kv_transfer_paths: Vec<ServingKvTransferPathObservation>,
    pub kv_transfer_resources: Vec<String>,
    pub kv_transfer_resource_dependencies: Vec<usize>,
    pub kv_transfer_s: f64,
    pub kv_transfer_fit: Option<CalibrationFitApplication>,
    pub decode_queue_s: f64,
    pub decode_worker_queue_s: f64,
    pub decode_resource_queue_s: f64,
    pub decode_s: f64,
    pub phase_breakdown: Vec<ServingRequestPhaseBreakdownObservation>,
    pub lifecycle_events: Vec<ServingRequestEventObservation>,
    pub metric_source: String,
    pub worker_summary: Vec<ServingRequestWorkerSummaryObservation>,
    pub worker_assignments: Vec<ServingWorkerAssignmentObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingRouteCandidateObservation {
    pub prefill_node: NodeId,
    pub decode_node: NodeId,
    pub prefill_route_nodes: Vec<NodeId>,
    pub prefill_route_gpus: Vec<GpuAddr>,
    pub decode_route_nodes: Vec<NodeId>,
    pub decode_route_gpus: Vec<GpuAddr>,
    pub selected: bool,
    pub routable: bool,
    pub rejection_reason: Option<String>,
    pub estimated_e2el_s: Option<f64>,
    pub estimated_kv_transfer_s: Option<f64>,
    pub estimated_kv_resource_wait_s: Option<f64>,
    pub estimated_prefill_wait_s: Option<f64>,
    pub estimated_decode_wait_s: Option<f64>,
    pub kv_transfer_bytes: u64,
    pub kv_transfer_bottlenecks: Vec<String>,
    pub kv_transfer_resources: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvTransferPathObservation {
    pub source: GpuAddr,
    pub destination: GpuAddr,
    pub latency_s: f64,
    pub bottleneck_bandwidth_gbps: f64,
    pub resources: Vec<String>,
    pub resource_details: Vec<ServingKvTransferPathResourceObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvTransferPathResourceObservation {
    pub kind: String,
    pub label: String,
    pub bandwidth_gbps: f64,
    pub latency_s: f64,
    pub rail_id: Option<u32>,
    pub from: Option<ServingKvTransferPathEndpointObservation>,
    pub to: Option<ServingKvTransferPathEndpointObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServingKvTransferPathEndpointObservation {
    pub kind: String,
    pub node_id: Option<NodeId>,
    pub local_gpu_id: Option<u32>,
    pub nic_id: Option<u32>,
    pub rail_id: Option<u32>,
}

impl ServingRequestObservation {
    fn from_decode_state(state: &DecodeRequestState) -> Self {
        let fallback_first_token_finish_s = state
            .first_decode_finish_s
            .unwrap_or(state.kv_finish_s.max(state.arrival_s));
        let fallback_first_decode_start_s = state
            .first_decode_start_s
            .unwrap_or(state.kv_finish_s.max(state.arrival_s));
        let fallback_last_token_finish_s = state
            .last_decode_finish_s
            .unwrap_or(fallback_first_token_finish_s);
        let fallback_tpot_s = if state.decode_tokens > 1 {
            (fallback_last_token_finish_s - fallback_first_token_finish_s)
                / f64::from(state.decode_tokens - 1)
        } else {
            (fallback_last_token_finish_s - state.kv_finish_s).max(0.0)
        };
        let status_time_s = state.status_time_s.or_else(|| {
            inferred_status_time_s(
                state,
                fallback_first_decode_start_s,
                fallback_first_token_finish_s,
            )
        });
        let lifecycle_events = request_lifecycle_events(state, status_time_s);
        let event_metrics = request_lifecycle_metric_summary(&lifecycle_events);
        let metric_source = if event_metrics.is_some() {
            "request_lifecycle_events"
        } else {
            "decode_state_fallback"
        };
        let first_decode_start_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.first_decode_start_s)
            .unwrap_or(fallback_first_decode_start_s);
        let first_token_finish_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.first_decode_finish_s)
            .unwrap_or(fallback_first_token_finish_s);
        let last_token_finish_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.last_decode_finish_s)
            .unwrap_or(fallback_last_token_finish_s);
        let decode_token_start_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_token_start_s.clone())
            .unwrap_or_else(|| state.decode_token_start_s.clone());
        let decode_token_finish_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_token_finish_s.clone())
            .unwrap_or_else(|| state.decode_token_finish_s.clone());
        let inter_token_latency_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.inter_token_latency_s.clone())
            .unwrap_or_else(|| inter_token_latencies(&decode_token_finish_s));
        let ttft_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.ttft_s)
            .unwrap_or_else(|| (first_token_finish_s - state.arrival_s).max(0.0));
        let tpot_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.tpot_s)
            .unwrap_or(fallback_tpot_s);
        let itl_s = mean(&inter_token_latency_s);
        let e2el_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.e2el_s)
            .unwrap_or_else(|| (last_token_finish_s - state.arrival_s).max(0.0));
        let service_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.service_s)
            .unwrap_or_else(|| (last_token_finish_s - state.prefill_start_s).max(0.0));
        let queue_delay_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.prefill_queue_s)
            .unwrap_or_else(|| (state.prefill_start_s - state.arrival_s).max(0.0));
        let prefill_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.prefill_s)
            .unwrap_or_else(|| (state.prefill_finish_s - state.prefill_start_s).max(0.0));
        let kv_queue_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.kv_queue_s)
            .unwrap_or_else(|| (state.kv_start_s - state.prefill_finish_s).max(0.0));
        let decode_queue_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_queue_s)
            .unwrap_or_else(|| (first_decode_start_s - state.kv_finish_s).max(0.0));
        let decode_s = event_metrics
            .as_ref()
            .map(|metrics| metrics.decode_s)
            .unwrap_or_else(|| (last_token_finish_s - first_decode_start_s).max(0.0));
        let completed = state.status.is_completed();
        let ttft_slo_missed = slo_missed(state.slo.ttft_s, ttft_s, completed);
        let tpot_slo_missed = slo_missed(state.slo.tpot_s, tpot_s, completed);
        let itl_slo_missed = slo_missed(state.slo.itl_s, itl_s, completed);
        let e2el_slo_missed = slo_missed(state.slo.e2el_s, e2el_s, completed);
        let deadline_missed = state.deadline_s.is_some_and(|deadline_s| {
            !state.status.is_completed() || last_token_finish_s > deadline_s + 1e-12
        });
        let phase_breakdown = request_phase_breakdown(
            state,
            status_time_s,
            first_decode_start_s,
            first_token_finish_s,
            last_token_finish_s,
        );
        let kv_block_ownership = request_kv_block_ownership(state);
        let worker_summary = request_worker_summary(state, &kv_block_ownership);
        let prefill_token_spans = prefill_capacity_spans(state);

        Self {
            request_idx: state.request_idx,
            request_id: state.request_id.clone(),
            tenant: state.tenant.clone(),
            model_id: state.model_id.clone(),
            traffic_class: state.traffic_class.clone(),
            shape_profile: state.shape_profile.clone(),
            cache_key: state.cache_key.clone(),
            status: state.status,
            status_time_s,
            failure_reason: state.failure_reason.clone(),
            rejection: state.failure_rejection.clone(),
            priority: state.priority,
            slo: state.slo,
            ttft_slo_missed,
            tpot_slo_missed,
            itl_slo_missed,
            e2el_slo_missed,
            deadline_s: state.deadline_s,
            deadline_missed,
            cancellation_s: state.cancellation_s,
            prefill_node: state.prefill_node,
            decode_node: state.decode_node,
            prefill_route_nodes: state.prefill_route_nodes.clone(),
            prefill_route_gpus: state.prefill_route_gpus.clone(),
            decode_route_nodes: state.decode_route_nodes.clone(),
            decode_route_gpus: state.decode_route_gpus.clone(),
            kv_cache_owner_gpus: state.decode_route_gpus.clone(),
            routing_policy: state.routing_policy,
            routing_candidate_count: state.routing_candidate_count,
            routing_routable_candidate_count: state.routing_routable_candidate_count,
            routing_estimated_e2el_s: state.routing_estimated_e2el_s,
            routing_estimated_kv_transfer_s: state.routing_estimated_kv_transfer_s,
            routing_estimated_kv_resource_wait_s: state.routing_estimated_kv_resource_wait_s,
            routing_estimated_prefill_wait_s: state.routing_estimated_prefill_wait_s,
            routing_estimated_decode_wait_s: state.routing_estimated_decode_wait_s,
            routing_reason: state.routing_reason.clone(),
            routing_candidates: state.routing_candidates.clone(),
            arrival_s: state.arrival_s,
            batch_size: state.batch_size,
            prompt_tokens: state.prompt_tokens,
            prefix_cache_hit_tokens: state.prefix_cache_hit_tokens,
            effective_prefill_tokens: state.effective_prefill_tokens,
            prefill_chunks: state.prefill_chunks,
            prefill_token_start_s: prefill_token_spans
                .iter()
                .map(|span| span.start_s)
                .collect(),
            prefill_token_finish_s: prefill_token_spans
                .iter()
                .map(|span| span.finish_s)
                .collect(),
            decode_tokens: state.decode_tokens,
            kv_block_tokens: state.kv_block_tokens,
            kv_cache_blocks: state.kv_cache_blocks,
            kv_allocated_tokens: state.kv_allocated_tokens,
            kv_fragmentation_tokens: state.kv_fragmentation_tokens,
            kv_block_ownership,
            prefill_start_s: state.prefill_start_s,
            prefill_finish_s: state.prefill_finish_s,
            kv_start_s: state.kv_start_s,
            kv_finish_s: state.kv_finish_s,
            first_decode_start_s,
            first_decode_finish_s: first_token_finish_s,
            last_decode_finish_s: last_token_finish_s,
            decode_iterations: decode_token_finish_s.len().min(u32::MAX as usize) as u32,
            decode_token_start_s,
            decode_token_finish_s,
            inter_token_latency_s,
            ttft_s,
            tpot_s,
            itl_s,
            e2el_s,
            service_s,
            queue_delay_s,
            prefill_s,
            prefill_worker_queue_s: state.prefill_worker_queue_s,
            prefill_resource_queue_s: state.prefill_resource_queue_s,
            kv_queue_s,
            kv_worker_queue_s: state.kv_worker_queue_s,
            kv_resource_queue_s: state.kv_resource_queue_s,
            kv_transfer_bytes: state.kv_transfer_bytes,
            kv_transfer_bottlenecks: state.kv_transfer_bottlenecks.clone(),
            kv_transfer_paths: state.kv_transfer_paths.clone(),
            kv_transfer_resources: state.kv_transfer_resources.clone(),
            kv_transfer_resource_dependencies: state.kv_transfer_resource_dependencies.clone(),
            kv_transfer_s: state.kv_transfer_s,
            kv_transfer_fit: state.kv_transfer_fit.clone(),
            decode_queue_s,
            decode_worker_queue_s: state.decode_worker_queue_s,
            decode_resource_queue_s: state.decode_resource_queue_s,
            decode_s,
            phase_breakdown,
            lifecycle_events,
            metric_source: metric_source.to_string(),
            worker_summary,
            worker_assignments: state.worker_assignments.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct RequestLifecycleMetricSummary {
    first_decode_start_s: f64,
    first_decode_finish_s: f64,
    last_decode_finish_s: f64,
    decode_token_start_s: Vec<f64>,
    decode_token_finish_s: Vec<f64>,
    inter_token_latency_s: Vec<f64>,
    ttft_s: f64,
    tpot_s: f64,
    e2el_s: f64,
    service_s: f64,
    prefill_queue_s: f64,
    prefill_s: f64,
    kv_queue_s: f64,
    decode_queue_s: f64,
    decode_s: f64,
}

fn request_lifecycle_metric_summary(
    events: &[ServingRequestEventObservation],
) -> Option<RequestLifecycleMetricSummary> {
    let arrival_s = first_lifecycle_event_time(events, ServingRequestEventKind::Arrived)?;
    let prefill_start_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::PrefillStarted)?;
    let prefill_finish_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::PrefillFinished)?;
    let kv_start_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::KvTransferStarted)?;
    let kv_finish_s =
        first_lifecycle_event_time(events, ServingRequestEventKind::KvTransferFinished)?;
    let decode_token_start_s =
        lifecycle_event_times(events, ServingRequestEventKind::DecodeIterationStarted);
    let decode_token_finish_s =
        lifecycle_event_times(events, ServingRequestEventKind::DecodeIterationFinished);
    let first_decode_start_s = decode_token_start_s.first().copied()?;
    let first_decode_finish_s = decode_token_finish_s.first().copied()?;
    let last_decode_finish_s = decode_token_finish_s.last().copied()?;
    let inter_token_latency_s = inter_token_latencies(&decode_token_finish_s);
    let tpot_s = if decode_token_finish_s.len() > 1 {
        (last_decode_finish_s - first_decode_finish_s) / (decode_token_finish_s.len() as f64 - 1.0)
    } else {
        (last_decode_finish_s - kv_finish_s).max(0.0)
    };

    Some(RequestLifecycleMetricSummary {
        first_decode_start_s,
        first_decode_finish_s,
        last_decode_finish_s,
        decode_token_start_s,
        decode_token_finish_s,
        inter_token_latency_s,
        ttft_s: (first_decode_finish_s - arrival_s).max(0.0),
        tpot_s,
        e2el_s: (last_decode_finish_s - arrival_s).max(0.0),
        service_s: (last_decode_finish_s - prefill_start_s).max(0.0),
        prefill_queue_s: (prefill_start_s - arrival_s).max(0.0),
        prefill_s: (prefill_finish_s - prefill_start_s).max(0.0),
        kv_queue_s: (kv_start_s - prefill_finish_s).max(0.0),
        decode_queue_s: (first_decode_start_s - kv_finish_s).max(0.0),
        decode_s: (last_decode_finish_s - first_decode_start_s).max(0.0),
    })
}

fn first_lifecycle_event_time(
    events: &[ServingRequestEventObservation],
    kind: ServingRequestEventKind,
) -> Option<f64> {
    lifecycle_event_times(events, kind).into_iter().next()
}

fn lifecycle_event_times(
    events: &[ServingRequestEventObservation],
    kind: ServingRequestEventKind,
) -> Vec<f64> {
    let mut times = events
        .iter()
        .filter(|event| event.kind == kind && event.at_s.is_finite())
        .map(|event| event.at_s)
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    times
}

fn inferred_status_time_s(
    state: &DecodeRequestState,
    first_decode_start_s: f64,
    first_token_finish_s: f64,
) -> Option<f64> {
    match state.status {
        ServingRequestStatus::Pending => None,
        ServingRequestStatus::Completed | ServingRequestStatus::TimedOut => {
            state.last_decode_finish_s.or(Some(first_token_finish_s))
        }
        ServingRequestStatus::RejectedAdmission => Some(
            [
                state.kv_finish_s,
                state.prefill_finish_s,
                state.prefill_start_s,
                state.arrival_s,
            ]
            .into_iter()
            .find(|value| value.is_finite())
            .unwrap_or(state.arrival_s),
        ),
        ServingRequestStatus::Cancelled => state.cancellation_s.or_else(|| {
            [
                state.last_decode_finish_s,
                Some(first_decode_start_s),
                Some(state.kv_finish_s),
                Some(state.prefill_finish_s),
                Some(state.arrival_s),
            ]
            .into_iter()
            .flatten()
            .find(|value| value.is_finite())
        }),
    }
}

fn request_phase_breakdown(
    state: &DecodeRequestState,
    status_time_s: Option<f64>,
    first_decode_start_s: f64,
    first_token_finish_s: f64,
    last_token_finish_s: f64,
) -> Vec<ServingRequestPhaseBreakdownObservation> {
    let terminal_s = status_time_s.unwrap_or(f64::INFINITY);
    let mut phases = Vec::new();
    let prefill_queue_finish_s = if state.prefill_start_s.is_finite() {
        state.prefill_start_s
    } else {
        terminal_s
    };
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "queued_for_prefill",
        ServingRequestPhaseCategory::Queue,
        state.arrival_s,
        prefill_queue_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "prefilling",
        ServingRequestPhaseCategory::Service,
        state.prefill_start_s,
        state.prefill_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "queued_for_kv_transfer",
        ServingRequestPhaseCategory::Queue,
        state.prefill_finish_s,
        state.kv_start_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "transferring_kv",
        ServingRequestPhaseCategory::Transfer,
        state.kv_start_s,
        state.kv_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "queued_for_decode",
        ServingRequestPhaseCategory::Queue,
        state.kv_finish_s,
        first_decode_start_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "first_decode_iteration",
        ServingRequestPhaseCategory::Service,
        first_decode_start_s,
        first_token_finish_s,
        true,
        true,
    );
    push_phase_breakdown(
        &mut phases,
        terminal_s,
        "decode_tail",
        ServingRequestPhaseCategory::Service,
        first_token_finish_s,
        last_token_finish_s,
        false,
        true,
    );
    phases
}

#[allow(clippy::too_many_arguments)]
fn push_phase_breakdown(
    phases: &mut Vec<ServingRequestPhaseBreakdownObservation>,
    terminal_s: f64,
    phase: &str,
    category: ServingRequestPhaseCategory,
    start_s: f64,
    finish_s: f64,
    contributes_to_ttft: bool,
    contributes_to_e2el: bool,
) {
    if !start_s.is_finite() || !finish_s.is_finite() || start_s > terminal_s + 1e-12 {
        return;
    }
    let finish_s = finish_s.min(terminal_s);
    if finish_s < start_s {
        return;
    }
    phases.push(ServingRequestPhaseBreakdownObservation {
        phase: phase.to_string(),
        category,
        start_s,
        finish_s,
        duration_s: (finish_s - start_s).max(0.0),
        contributes_to_ttft,
        contributes_to_e2el,
    });
}

fn request_kv_block_ownership(
    state: &DecodeRequestState,
) -> Vec<ServingKvBlockOwnershipObservation> {
    let Some((allocated_at_s, released_at_s)) = kv_residency_window_s(state) else {
        return Vec::new();
    };
    kv_owner_allocations(state)
        .into_iter()
        .map(|allocation| {
            let owner_worker_slots = decode_owner_worker_slots(state, allocation.owner);
            let worker_slot_ownership = kv_worker_slot_ownership(
                &allocation,
                &owner_worker_slots,
                state.batch_size.max(1),
                state.kv_block_tokens,
            );
            ServingKvBlockOwnershipObservation {
                allocation_id: allocation.allocation_id,
                owner: allocation.owner,
                owner_worker_slots,
                worker_slot_ownership,
                decode_operation_ids: decode_owner_operation_ids(state, allocation.owner),
                allocated_at_s,
                released_at_s,
                duration_s: (released_at_s - allocated_at_s).max(0.0),
                block_start: allocation.block_start,
                block_end: allocation.block_end,
                decode_sequences: state.batch_size.max(1),
                resident_tokens: allocation.resident_tokens,
                kv_blocks: allocation.kv_blocks,
                allocated_kv_tokens: allocation.allocated_kv_tokens,
                kv_fragmentation_tokens: allocation.kv_fragmentation_tokens,
                block_table_entries: allocation.block_table_entries,
                block_table_bytes: allocation.block_table_bytes,
            }
        })
        .collect()
}

fn kv_worker_slot_ownership(
    allocation: &KvOwnerAllocation,
    owner_worker_slots: &[u32],
    decode_sequences: u32,
    kv_block_tokens: u32,
) -> Vec<ServingKvWorkerSlotOwnershipObservation> {
    let mut slots = owner_worker_slots.to_vec();
    slots.sort_unstable();
    slots.dedup();
    if slots.is_empty() {
        return Vec::new();
    }

    let block_counts = partition_units(allocation.kv_blocks, slots.len());
    let resident_token_counts = partition_tokens_by_block_capacity(
        allocation.resident_tokens,
        &block_counts,
        u64::from(kv_block_tokens.max(1)),
    );
    let sequence_counts = partition_units(u64::from(decode_sequences), slots.len());
    let mut block_start = allocation.block_start;

    slots
        .into_iter()
        .zip(block_counts)
        .zip(resident_token_counts)
        .zip(sequence_counts)
        .filter_map(|(((slot, kv_blocks), resident_tokens), decode_sequences)| {
            let block_end = block_start.saturating_add(kv_blocks);
            let allocated_kv_tokens = kv_blocks.saturating_mul(u64::from(kv_block_tokens.max(1)));
            let slot_allocation = if kv_blocks == 0 && resident_tokens == 0 && decode_sequences == 0
            {
                None
            } else {
                Some(ServingKvWorkerSlotOwnershipObservation {
                    allocation_id: format!(
                        "{}:slot-{}:blocks-{}-{}",
                        allocation.allocation_id, slot, block_start, block_end
                    ),
                    slot,
                    block_start,
                    block_end,
                    decode_sequences: decode_sequences.min(u64::from(u32::MAX)) as u32,
                    resident_tokens,
                    kv_blocks,
                    allocated_kv_tokens,
                    kv_fragmentation_tokens: allocated_kv_tokens.saturating_sub(resident_tokens),
                    block_table_entries: kv_blocks,
                    block_table_bytes: kv_blocks.saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES),
                })
            };
            block_start = block_end;
            slot_allocation
        })
        .collect()
}

fn decode_owner_worker_slots(state: &DecodeRequestState, owner: GpuAddr) -> Vec<u32> {
    let mut slots = state
        .worker_assignments
        .iter()
        .filter(|assignment| {
            assignment.phase == "decode"
                && assignment.node_id == owner.node_id
                && assignment.local_gpu_id == owner.local_gpu_id
        })
        .map(|assignment| assignment.slot)
        .collect::<Vec<_>>();
    slots.sort_unstable();
    slots.dedup();
    slots
}

fn decode_owner_operation_ids(state: &DecodeRequestState, owner: GpuAddr) -> Vec<usize> {
    let mut operation_ids = state
        .worker_assignments
        .iter()
        .filter(|assignment| {
            assignment.phase == "decode"
                && assignment.node_id == owner.node_id
                && assignment.local_gpu_id == owner.local_gpu_id
        })
        .flat_map(|assignment| assignment.operation_ids.iter().copied())
        .collect::<Vec<_>>();
    operation_ids.sort_unstable();
    operation_ids.dedup();
    operation_ids
}

fn request_worker_summary(
    state: &DecodeRequestState,
    kv_block_ownership: &[ServingKvBlockOwnershipObservation],
) -> Vec<ServingRequestWorkerSummaryObservation> {
    let mut summary = Vec::new();
    summary.extend(worker_assignment_summary(
        &state.worker_assignments,
        "prefill",
        "prefill_source",
    ));
    summary.extend(worker_assignment_summary(
        &state.worker_assignments,
        "kv_transfer",
        "kv_transfer",
    ));
    summary.extend(worker_assignment_summary(
        &state.worker_assignments,
        "decode",
        "decode_owner",
    ));
    summary.extend(kv_block_ownership.iter().map(kv_cache_owner_worker_summary));
    summary
}

#[derive(Clone, Debug)]
struct WorkerSummaryAccumulator {
    worker_slots: Vec<u32>,
    operation_ids: Vec<usize>,
    assignment_count: u32,
    start_s: f64,
    finish_s: f64,
}

impl Default for WorkerSummaryAccumulator {
    fn default() -> Self {
        Self {
            worker_slots: Vec::new(),
            operation_ids: Vec::new(),
            assignment_count: 0,
            start_s: f64::INFINITY,
            finish_s: 0.0,
        }
    }
}

fn worker_assignment_summary(
    assignments: &[ServingWorkerAssignmentObservation],
    phase: &str,
    role: &str,
) -> Vec<ServingRequestWorkerSummaryObservation> {
    let mut grouped: BTreeMap<(NodeId, u32), WorkerSummaryAccumulator> = BTreeMap::new();
    for assignment in assignments
        .iter()
        .filter(|assignment| assignment.phase == phase)
    {
        let entry = grouped
            .entry((assignment.node_id, assignment.local_gpu_id))
            .or_default();
        entry.worker_slots.push(assignment.slot);
        entry
            .operation_ids
            .extend(assignment.operation_ids.iter().copied());
        entry.assignment_count = entry.assignment_count.saturating_add(1);
        entry.start_s = entry.start_s.min(assignment.start_s);
        entry.finish_s = entry.finish_s.max(assignment.finish_s);
    }

    grouped
        .into_iter()
        .map(|((node_id, local_gpu_id), mut entry)| {
            entry.worker_slots.sort_unstable();
            entry.worker_slots.dedup();
            entry.operation_ids.sort_unstable();
            entry.operation_ids.dedup();
            ServingRequestWorkerSummaryObservation {
                role: role.to_string(),
                phase: phase.to_string(),
                node_id,
                local_gpu_id,
                worker_slots: entry.worker_slots,
                operation_ids: entry.operation_ids,
                assignment_count: entry.assignment_count,
                start_s: if entry.start_s.is_finite() {
                    entry.start_s
                } else {
                    0.0
                },
                finish_s: entry.finish_s,
                resident_tokens: 0,
                kv_blocks: 0,
            }
        })
        .collect()
}

fn kv_cache_owner_worker_summary(
    ownership: &ServingKvBlockOwnershipObservation,
) -> ServingRequestWorkerSummaryObservation {
    ServingRequestWorkerSummaryObservation {
        role: "kv_cache_owner".to_string(),
        phase: "decode".to_string(),
        node_id: ownership.owner.node_id,
        local_gpu_id: ownership.owner.local_gpu_id,
        worker_slots: ownership.owner_worker_slots.clone(),
        operation_ids: ownership.decode_operation_ids.clone(),
        assignment_count: ownership.owner_worker_slots.len().min(u32::MAX as usize) as u32,
        start_s: ownership.allocated_at_s,
        finish_s: ownership.released_at_s,
        resident_tokens: ownership.resident_tokens,
        kv_blocks: ownership.kv_blocks,
    }
}

fn kv_residency_window_s(state: &DecodeRequestState) -> Option<(f64, f64)> {
    if !state.status.is_admitted() {
        return None;
    }
    let allocated_at_s = state.kv_finish_s;
    if !allocated_at_s.is_finite() {
        return None;
    }

    let mut released_at_s = state.last_decode_finish_s?;
    if state.status == ServingRequestStatus::Cancelled
        && let Some(cancellation_s) = state.cancellation_s.or(state.status_time_s)
    {
        released_at_s = released_at_s.min(cancellation_s);
    }
    if state.status == ServingRequestStatus::TimedOut
        && let Some(timeout_s) = state.request_timeout_s
    {
        released_at_s = released_at_s.min(state.arrival_s + timeout_s);
    }
    if state.status == ServingRequestStatus::TimedOut
        && let Some(status_time_s) = state.status_time_s
        && status_time_s > released_at_s
    {
        released_at_s = status_time_s;
    }
    if !released_at_s.is_finite() || released_at_s <= allocated_at_s {
        return None;
    }

    Some((allocated_at_s, released_at_s))
}

fn prefill_terminal_s(state: &DecodeRequestState) -> Option<f64> {
    if state.status == ServingRequestStatus::Cancelled {
        return state.cancellation_s.or(state.status_time_s);
    }
    if state.status == ServingRequestStatus::TimedOut
        && let Some(timeout_s) = state.request_timeout_s
    {
        return Some(state.arrival_s + timeout_s);
    }
    state.status_time_s
}

fn prefill_capacity_spans(state: &DecodeRequestState) -> Vec<PrefillTokenSpan> {
    if !state.status.is_admitted() {
        return Vec::new();
    }

    let terminal_s = prefill_terminal_s(state).unwrap_or(f64::INFINITY);
    let mut spans = state
        .prefill_token_spans
        .iter()
        .filter_map(|span| {
            let finish_s = span.finish_s.min(terminal_s);
            if span.tokens == 0
                || !span.start_s.is_finite()
                || !finish_s.is_finite()
                || finish_s <= span.start_s
            {
                return None;
            }
            Some(PrefillTokenSpan {
                start_s: span.start_s,
                finish_s,
                tokens: span.tokens,
            })
        })
        .collect::<Vec<_>>();
    if !spans.is_empty() {
        return spans;
    }

    let start_s = state.prefill_start_s;
    let finish_s = state.prefill_finish_s.min(terminal_s);
    let tokens = u64::from(state.batch_size.max(1))
        .saturating_mul(u64::from(state.effective_prefill_tokens));
    if tokens > 0 && start_s.is_finite() && finish_s.is_finite() && finish_s > start_s {
        spans.push(PrefillTokenSpan {
            start_s,
            finish_s,
            tokens,
        });
    }
    spans
}

fn prefill_owner_gpus(state: &DecodeRequestState) -> Vec<GpuAddr> {
    if state.prefill_route_gpus.is_empty() {
        vec![GpuAddr {
            node_id: state.prefill_node,
            local_gpu_id: 0,
        }]
    } else {
        state.prefill_route_gpus.clone()
    }
}

fn prefill_owner_nodes(state: &DecodeRequestState) -> Vec<NodeId> {
    if state.prefill_route_nodes.is_empty() {
        vec![state.prefill_node]
    } else {
        state.prefill_route_nodes.clone()
    }
}

fn decode_owner_gpus(state: &DecodeRequestState) -> Vec<GpuAddr> {
    if state.decode_route_gpus.is_empty() {
        vec![GpuAddr {
            node_id: state.decode_node,
            local_gpu_id: 0,
        }]
    } else {
        state.decode_route_gpus.clone()
    }
}

fn kv_ownership_event_message(state: &DecodeRequestState) -> String {
    let block_table_bytes = state
        .kv_cache_blocks
        .saturating_mul(KV_BLOCK_TABLE_ENTRY_BYTES);
    format!(
        "kv_blocks={} resident_tokens={} allocated_kv_tokens={} block_table_bytes={} owners={}",
        state.kv_cache_blocks,
        u64::from(state.batch_size.max(1)) * u64::from(state.max_sequence_tokens.max(1)),
        state.kv_allocated_tokens,
        block_table_bytes,
        decode_owner_gpus(state).len()
    )
}

fn request_lifecycle_events(
    state: &DecodeRequestState,
    status_time_s: Option<f64>,
) -> Vec<ServingRequestEventObservation> {
    let terminal_s = status_time_s.unwrap_or(f64::INFINITY);
    let mut events = Vec::new();
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::Arrived,
        "arrival",
        state.arrival_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::QueuedForPrefill,
        "prefill",
        state.arrival_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::PrefillStarted,
        "prefill",
        state.prefill_start_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::PrefillFinished,
        "prefill",
        state.prefill_finish_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::QueuedForKvTransfer,
        "kv_transfer",
        state.prefill_finish_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::KvTransferStarted,
        "kv_transfer",
        state.kv_start_s,
        None,
        None,
    );
    push_lifecycle_event(
        &mut events,
        terminal_s,
        ServingRequestEventKind::KvTransferFinished,
        "kv_transfer",
        state.kv_finish_s,
        None,
        None,
    );
    if let Some((allocated_at_s, released_at_s)) = kv_residency_window_s(state) {
        push_lifecycle_event(
            &mut events,
            terminal_s,
            ServingRequestEventKind::KvBlocksAllocated,
            "kv_cache",
            allocated_at_s,
            None,
            Some(kv_ownership_event_message(state)),
        );
        if state.first_decode_start_s.is_some() {
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::QueuedForDecode,
                "decode",
                state.kv_finish_s,
                None,
                None,
            );
        }
        for (idx, start_s) in state.decode_token_start_s.iter().copied().enumerate() {
            let decode_iteration = Some(idx.min(u32::MAX as usize) as u32);
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::DecodeIterationStarted,
                "decode",
                start_s,
                decode_iteration,
                None,
            );
            if let Some(finish_s) = state.decode_token_finish_s.get(idx).copied() {
                push_lifecycle_event(
                    &mut events,
                    terminal_s,
                    ServingRequestEventKind::DecodeIterationFinished,
                    "decode",
                    finish_s,
                    decode_iteration,
                    None,
                );
            }
        }
        push_lifecycle_event(
            &mut events,
            terminal_s,
            ServingRequestEventKind::KvBlocksReleased,
            "kv_cache",
            released_at_s,
            None,
            Some(kv_ownership_event_message(state)),
        );
    } else {
        if state.first_decode_start_s.is_some() {
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::QueuedForDecode,
                "decode",
                state.kv_finish_s,
                None,
                None,
            );
        }
        for (idx, start_s) in state.decode_token_start_s.iter().copied().enumerate() {
            let decode_iteration = Some(idx.min(u32::MAX as usize) as u32);
            push_lifecycle_event(
                &mut events,
                terminal_s,
                ServingRequestEventKind::DecodeIterationStarted,
                "decode",
                start_s,
                decode_iteration,
                None,
            );
            if let Some(finish_s) = state.decode_token_finish_s.get(idx).copied() {
                push_lifecycle_event(
                    &mut events,
                    terminal_s,
                    ServingRequestEventKind::DecodeIterationFinished,
                    "decode",
                    finish_s,
                    decode_iteration,
                    None,
                );
            }
        }
    }
    if let Some(status_time_s) = status_time_s
        && let Some(kind) = terminal_event_kind(state.status)
    {
        push_lifecycle_event(
            &mut events,
            f64::INFINITY,
            kind,
            "terminal",
            status_time_s,
            None,
            state.failure_reason.clone(),
        );
    }
    events
}

fn terminal_event_kind(status: ServingRequestStatus) -> Option<ServingRequestEventKind> {
    match status {
        ServingRequestStatus::Pending => None,
        ServingRequestStatus::Completed => Some(ServingRequestEventKind::Completed),
        ServingRequestStatus::RejectedAdmission => Some(ServingRequestEventKind::RejectedAdmission),
        ServingRequestStatus::TimedOut => Some(ServingRequestEventKind::TimedOut),
        ServingRequestStatus::Cancelled => Some(ServingRequestEventKind::Cancelled),
    }
}

fn push_lifecycle_event(
    events: &mut Vec<ServingRequestEventObservation>,
    terminal_s: f64,
    kind: ServingRequestEventKind,
    phase: &str,
    at_s: f64,
    decode_iteration: Option<u32>,
    message: Option<String>,
) {
    if !at_s.is_finite() || at_s > terminal_s + 1e-12 {
        return;
    }
    events.push(ServingRequestEventObservation {
        kind,
        phase: phase.to_string(),
        at_s,
        decode_iteration,
        message,
    });
}

fn cyclic_or_default(values: &[u32], idx: u32, default: u32) -> u32 {
    values
        .get(idx as usize % values.len().max(1))
        .copied()
        .unwrap_or(default)
        .max(1)
}

fn sample_or_cyclic(
    distribution: Option<&ServingValueDistribution>,
    cyclic_values: &[u32],
    seed: u64,
    stream: u64,
    idx: u32,
    default: u32,
) -> u32 {
    if let Some(distribution) = distribution {
        let mut rng = LcgRng::new(seed ^ stream ^ u64::from(idx).wrapping_mul(0x9E37_79B9));
        distribution.sample(&mut rng)
    } else {
        cyclic_or_default(cyclic_values, idx, default)
    }
}

impl ServingValueDistribution {
    fn sample(&self, rng: &mut LcgRng) -> u32 {
        match self {
            ServingValueDistribution::Uniform { min, max } => {
                let min = (*min).max(1);
                let max = (*max).max(min);
                let span = max.saturating_sub(min).saturating_add(1);
                min.saturating_add(rng.next_bounded_u32(span))
            }
            ServingValueDistribution::Weighted { values, weights } => {
                sample_weighted(values, weights, rng).unwrap_or(1)
            }
            ServingValueDistribution::LogNormal {
                median,
                sigma,
                min,
                max,
            } => {
                let min = (*min).max(1);
                let max = (*max).max(min);
                let z = rng.next_standard_normal();
                let sampled = (median.max(1.0) * (sigma.max(1e-9) * z).exp()).round();
                sampled.clamp(f64::from(min), f64::from(max)) as u32
            }
        }
    }
}

fn sample_weighted(values: &[u32], weights: &[f64], rng: &mut LcgRng) -> Option<u32> {
    if values.is_empty() || values.len() != weights.len() {
        return None;
    }
    let total = weights
        .iter()
        .copied()
        .filter(|weight| weight.is_finite() && *weight > 0.0)
        .sum::<f64>();
    if total <= 0.0 {
        return None;
    }

    let mut target = rng.next_open_unit_f64() * total;
    for (value, weight) in values.iter().copied().zip(weights.iter().copied()) {
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        if target <= weight {
            return Some(value.max(1));
        }
        target -= weight;
    }

    values.last().copied().map(|value| value.max(1))
}

fn poisson_arrival_times(request_count: u32, rate_per_s: f64, seed: u64) -> Vec<f64> {
    let rate_per_s = if rate_per_s.is_finite() && rate_per_s > 0.0 {
        rate_per_s
    } else {
        1.0
    };
    let mut rng = LcgRng::new(seed);
    let mut arrivals = Vec::with_capacity(request_count as usize);
    let mut next_arrival_s = 0.0;

    for idx in 0..request_count {
        if idx > 0 {
            let sample = rng.next_open_unit_f64();
            next_arrival_s += -sample.ln() / rate_per_s;
        }
        arrivals.push(next_arrival_s);
    }

    arrivals
}

fn bursty_arrival_times(
    request_count: u32,
    burst_size: u32,
    burst_interval_s: f64,
    intra_burst_gap_s: f64,
) -> Vec<f64> {
    let burst_size = burst_size.max(1);
    let burst_interval_s = if burst_interval_s.is_finite() && burst_interval_s > 0.0 {
        burst_interval_s
    } else {
        1.0
    };
    let intra_burst_gap_s = if intra_burst_gap_s.is_finite() && intra_burst_gap_s >= 0.0 {
        intra_burst_gap_s
    } else {
        0.0
    };

    (0..request_count)
        .map(|idx| {
            let burst_idx = idx / burst_size;
            let offset_idx = idx % burst_size;
            f64::from(burst_idx) * burst_interval_s + f64::from(offset_idx) * intra_burst_gap_s
        })
        .collect()
}

fn diurnal_arrival_times(
    request_count: u32,
    min_rate_per_s: f64,
    max_rate_per_s: f64,
    period_s: f64,
    phase_s: f64,
    seed: u64,
) -> Vec<f64> {
    let max_rate_per_s = if max_rate_per_s.is_finite() && max_rate_per_s > 0.0 {
        max_rate_per_s
    } else {
        1.0
    };
    let min_rate_per_s = if min_rate_per_s.is_finite() && min_rate_per_s >= 0.0 {
        min_rate_per_s.min(max_rate_per_s)
    } else {
        0.0
    };
    let period_s = if period_s.is_finite() && period_s > 0.0 {
        period_s
    } else {
        86_400.0
    };
    let phase_s = if phase_s.is_finite() { phase_s } else { 0.0 };
    let mut rng = LcgRng::new(seed);
    let mut arrivals = Vec::with_capacity(request_count as usize);
    let mut candidate_s = 0.0;

    while arrivals.len() < request_count as usize {
        let sample = rng.next_open_unit_f64();
        candidate_s += -sample.ln() / max_rate_per_s;
        let rate_s = diurnal_rate_at_s(
            candidate_s,
            min_rate_per_s,
            max_rate_per_s,
            period_s,
            phase_s,
        );
        if rng.next_open_unit_f64() <= rate_s / max_rate_per_s {
            arrivals.push(candidate_s);
        }
    }

    if let Some(first) = arrivals.first().copied() {
        for arrival in &mut arrivals {
            *arrival -= first;
        }
    }

    arrivals
}

fn diurnal_rate_at_s(
    time_s: f64,
    min_rate_per_s: f64,
    max_rate_per_s: f64,
    period_s: f64,
    phase_s: f64,
) -> f64 {
    let amplitude = (max_rate_per_s - min_rate_per_s).max(0.0) / 2.0;
    let midpoint = min_rate_per_s + amplitude;
    let angle = 2.0 * std::f64::consts::PI * ((time_s + phase_s) / period_s);
    (midpoint + amplitude * angle.sin()).clamp(min_rate_per_s, max_rate_per_s)
}

fn self_similar_arrival_times(
    request_count: u32,
    rate_per_s: f64,
    pareto_shape: f64,
    max_gap_s: Option<f64>,
    seed: u64,
) -> Vec<f64> {
    let rate_per_s = if rate_per_s.is_finite() && rate_per_s > 0.0 {
        rate_per_s
    } else {
        1.0
    };
    let pareto_shape = if pareto_shape.is_finite() && pareto_shape > 1.0 {
        pareto_shape
    } else {
        1.4
    };
    let max_gap_s = max_gap_s.and_then(|gap| (gap.is_finite() && gap > 0.0).then_some(gap));
    let mean_gap_s = 1.0 / rate_per_s;
    let pareto_scale_s = mean_gap_s * (pareto_shape - 1.0) / pareto_shape;
    let mut rng = LcgRng::new(seed);
    let mut arrivals = Vec::with_capacity(request_count as usize);
    let mut next_arrival_s = 0.0;

    for idx in 0..request_count {
        if idx > 0 {
            let sample = rng.next_open_unit_f64();
            let mut gap_s = pareto_scale_s / sample.powf(1.0 / pareto_shape);
            if let Some(max_gap_s) = max_gap_s {
                gap_s = gap_s.min(max_gap_s);
            }
            next_arrival_s += gap_s;
        }
        arrivals.push(next_arrival_s);
    }

    arrivals
}

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_open_unit_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = self.state >> 11;
        ((mantissa as f64) + 1.0) / ((1_u64 << 53) as f64 + 1.0)
    }

    fn next_bounded_u32(&mut self, upper_exclusive: u32) -> u32 {
        if upper_exclusive <= 1 {
            return 0;
        }
        (self.next_open_unit_f64() * f64::from(upper_exclusive)).floor() as u32
    }

    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_open_unit_f64();
        let u2 = self.next_open_unit_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn request_token_scale(
    batch_size: u32,
    tokens: u32,
    base_batch_size: u32,
    base_tokens: u32,
) -> f64 {
    let numerator = f64::from(batch_size.max(1)) * f64::from(tokens.max(1));
    let denominator = f64::from(base_batch_size.max(1)) * f64::from(base_tokens.max(1));
    numerator / denominator
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
