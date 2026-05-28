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
mod capacity;
mod hardware;
mod measurement;
mod memory;
mod metric_fits;
mod model;
mod observations;
mod operations;
mod pools;
mod ranking;
mod rejections;
mod reporting;
mod routing;
mod scheduling;
mod slo;
mod stats;
mod topology;
mod utilization;
use approximations::*;
use arrivals::*;
use calibration::*;
use capacity::*;
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
use operations::*;
use pools::*;
use ranking::*;
use rejections::*;
use reporting::*;
use routing::*;
use scheduling::*;
use slo::*;
use stats::*;
use topology::*;
use utilization::*;

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

#[cfg(test)]
mod tests;
