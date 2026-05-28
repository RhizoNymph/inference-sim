use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn serving_approximations(
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

pub(super) fn serving_approximation_summary(
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
