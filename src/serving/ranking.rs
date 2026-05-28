use super::*;

pub(super) fn serving_candidate_id(
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

pub(super) fn serving_bottleneck_summary(
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
pub(super) fn serving_objective_bottleneck_summaries(
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

pub(super) fn search_deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

pub(super) fn annotate_serving_pareto(results: &mut [ScoredServingConfig]) {
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

pub(super) fn sort_serving_results(
    results: &mut [ScoredServingConfig],
    objective: ServingObjective,
) {
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

pub(super) fn service_backpressure_penalty_score(
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

pub(super) fn topology_risk_penalty_score(
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

pub(super) fn slo_miss_penalty_components_from_metrics(
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

pub(super) fn traffic_class_slo_miss_penalties(
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
