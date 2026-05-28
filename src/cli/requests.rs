use super::*;

pub(super) fn write_request_observations<W: Write>(
    writer: &mut W,
    indent: &str,
    score: &ScoredServingConfig,
    limit: Option<usize>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let observations = &score.request_observations;
    let displayed_observations =
        limit.map_or(observations.len(), |limit| limit.min(observations.len()));
    writeln!(
        writer,
        "{indent}  \"request_observation_count\": {},",
        observations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"request_observations_truncated\": {},",
        displayed_observations < observations.len()
    )?;
    writeln!(writer, "{indent}  \"request_observations\": [")?;
    for (idx, observation) in observations.iter().take(displayed_observations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"request_idx\": {},",
            observation.request_idx
        )?;
        writeln!(
            writer,
            "{indent}      \"request_id\": {},",
            observation
                .request_id
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"tenant\": {},",
            observation
                .tenant
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"model_id\": {},",
            observation
                .model_id
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"traffic_class\": {},",
            observation
                .traffic_class
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"shape_profile\": {},",
            observation
                .shape_profile
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"cache_key\": {},",
            observation
                .cache_key
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"status\": {},",
            json_string(observation.status.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"status_time_ms\": {},",
            observation
                .status_time_s
                .map(json_ms)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"failure_reason\": {},",
            observation
                .failure_reason
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        write_request_rejection(writer, indent, observation.rejection.as_ref(), true)?;
        writeln!(
            writer,
            "{indent}      \"priority\": {},",
            observation.priority
        )?;
        write_request_slo(writer, indent, &observation.slo, true)?;
        writeln!(
            writer,
            "{indent}      \"ttft_slo_missed\": {},",
            observation.ttft_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_slo_missed\": {},",
            observation.tpot_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_slo_missed\": {},",
            observation.itl_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_slo_missed\": {},",
            observation.e2el_slo_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_ms\": {},",
            observation
                .deadline_s
                .map(json_ms)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"deadline_missed\": {},",
            observation.deadline_missed
        )?;
        writeln!(
            writer,
            "{indent}      \"cancellation_ms\": {},",
            observation
                .cancellation_s
                .map(json_ms)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_node\": {},",
            observation.prefill_node
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_route_nodes\": [{}],",
            u32_list(&observation.prefill_route_nodes)
        )?;
        write_gpu_addr_list(
            writer,
            indent,
            "prefill_route_gpus",
            &observation.prefill_route_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_node\": {},",
            observation.decode_node
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_route_nodes\": [{}],",
            u32_list(&observation.decode_route_nodes)
        )?;
        write_gpu_addr_list(
            writer,
            indent,
            "decode_route_gpus",
            &observation.decode_route_gpus,
            true,
        )?;
        write_gpu_addr_list(
            writer,
            indent,
            "kv_cache_owner_gpus",
            &observation.kv_cache_owner_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_policy\": {},",
            json_string(observation.routing_policy.as_str())
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_candidate_count\": {},",
            observation.routing_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_routable_candidate_count\": {},",
            observation.routing_routable_candidate_count
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_e2el_ms\": {},",
            json_ms(observation.routing_estimated_e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_kv_transfer_ms\": {},",
            json_ms(observation.routing_estimated_kv_transfer_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_kv_resource_wait_ms\": {},",
            json_ms(observation.routing_estimated_kv_resource_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_prefill_wait_ms\": {},",
            json_ms(observation.routing_estimated_prefill_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_estimated_decode_wait_ms\": {},",
            json_ms(observation.routing_estimated_decode_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"routing_reason\": {},",
            json_string(&observation.routing_reason)
        )?;
        write_routing_candidates(writer, indent, &observation.routing_candidates, true)?;
        writeln!(
            writer,
            "{indent}      \"batch_size\": {},",
            observation.batch_size
        )?;
        writeln!(
            writer,
            "{indent}      \"prompt_tokens\": {},",
            observation.prompt_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"prefix_cache_hit_tokens\": {},",
            observation.prefix_cache_hit_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"effective_prefill_tokens\": {},",
            observation.effective_prefill_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_chunks\": {},",
            observation.prefill_chunks
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_tokens\": {},",
            observation.decode_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_block_tokens\": {},",
            observation.kv_block_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_cache_blocks\": {},",
            observation.kv_cache_blocks
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_allocated_tokens\": {},",
            observation.kv_allocated_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_fragmentation_tokens\": {},",
            observation.kv_fragmentation_tokens
        )?;
        write_kv_block_ownership(writer, indent, observation, true)?;
        writeln!(
            writer,
            "{indent}      \"arrival_ms\": {},",
            json_ms(observation.arrival_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_start_ms\": {},",
            json_ms(observation.prefill_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_finish_ms\": {},",
            json_ms(observation.prefill_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_start_ms\": {},",
            json_ms(observation.kv_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_finish_ms\": {},",
            json_ms(observation.kv_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"first_decode_start_ms\": {},",
            json_ms(observation.first_decode_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"first_decode_finish_ms\": {},",
            json_ms(observation.first_decode_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"last_decode_finish_ms\": {},",
            json_ms(observation.last_decode_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_iterations\": {},",
            observation.decode_iterations
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_token_start_ms\": [{}],",
            ms_list(&observation.decode_token_start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_token_finish_ms\": [{}],",
            ms_list(&observation.decode_token_finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"inter_token_latency_ms\": [{}],",
            ms_list(&observation.inter_token_latency_s)
        )?;
        write_request_phase_spans(writer, indent, observation, true)?;
        write_request_phase_breakdown(writer, indent, observation, true)?;
        write_request_lifecycle_events(writer, indent, observation, true)?;
        writeln!(
            writer,
            "{indent}      \"metric_source\": {},",
            json_string(&observation.metric_source)
        )?;
        writeln!(
            writer,
            "{indent}      \"included_in_measurement_window\": {},",
            request_in_measurement_window(observation, score)
        )?;
        writeln!(
            writer,
            "{indent}      \"measurement_window_source\": {},",
            json_string(&score.measurement_window.source)
        )?;
        writeln!(
            writer,
            "{indent}      \"output_tokens\": {},",
            request_observation_output_tokens(observation)
        )?;
        write_request_metric_derivation(writer, indent, observation, score, true)?;
        write_request_worker_summary(writer, indent, &observation.worker_summary, true)?;
        write_worker_assignments(writer, indent, &observation.worker_assignments, true)?;
        writeln!(
            writer,
            "{indent}      \"ttft_ms\": {},",
            json_ms(observation.ttft_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"tpot_ms\": {},",
            json_ms(observation.tpot_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"itl_ms\": {},",
            json_ms(observation.itl_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"e2el_ms\": {},",
            json_ms(observation.e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"service_ms\": {},",
            json_ms(observation.service_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"queue_delay_ms\": {},",
            json_ms(observation.queue_delay_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_ms\": {},",
            json_ms(observation.prefill_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_worker_queue_ms\": {},",
            json_ms(observation.prefill_worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"prefill_resource_queue_ms\": {},",
            json_ms(observation.prefill_resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_queue_ms\": {},",
            json_ms(observation.kv_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_worker_queue_ms\": {},",
            json_ms(observation.kv_worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_resource_queue_ms\": {},",
            json_ms(observation.kv_resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_bytes\": {},",
            observation.kv_transfer_bytes
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_bottlenecks\": [{}],",
            string_list(&observation.kv_transfer_bottlenecks)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_resources\": [{}],",
            string_list(&observation.kv_transfer_resources)
        )?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_resource_dependencies\": [{}],",
            usize_list(&observation.kv_transfer_resource_dependencies)
        )?;
        write_kv_transfer_paths(writer, indent, observation, true)?;
        writeln!(
            writer,
            "{indent}      \"kv_transfer_ms\": {},",
            json_ms(observation.kv_transfer_s)
        )?;
        write_optional_calibration_fit_application(
            writer,
            indent,
            "kv_transfer_fit",
            observation.kv_transfer_fit.as_ref(),
            true,
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_queue_ms\": {},",
            json_ms(observation.decode_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_worker_queue_ms\": {},",
            json_ms(observation.decode_worker_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_resource_queue_ms\": {},",
            json_ms(observation.decode_resource_queue_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_ms\": {}",
            json_ms(observation.decode_s)
        )?;
        if idx + 1 < displayed_observations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

#[derive(Debug)]
pub(super) struct RequestMetricDerivation {
    pub(super) completed: bool,
    pub(super) event_sourced: bool,
    pub(super) terminal_event: Option<&'static str>,
    pub(super) terminal_event_s: Option<f64>,
    pub(super) metric_unavailable_reason: Option<String>,
    pub(super) ttft_end_event: Option<&'static str>,
    pub(super) tpot_start_event: Option<&'static str>,
    pub(super) tpot_end_event: Option<&'static str>,
    pub(super) e2el_end_event: Option<&'static str>,
    pub(super) throughput_duration_start_event: Option<&'static str>,
    pub(super) throughput_duration_end_event: Option<&'static str>,
    pub(super) decode_finish_event_count: usize,
    pub(super) tpot_sample_count: usize,
}

pub(super) fn request_metric_derivation(
    observation: &ServingRequestObservation,
) -> RequestMetricDerivation {
    let completed = observation.status.as_str() == "completed";
    let event_sourced = observation.metric_source == "request_lifecycle_events";
    let decode_finish_event_count = observation.decode_token_finish_s.len();
    let tpot_sample_count = if decode_finish_event_count > 1 {
        decode_finish_event_count - 1
    } else if completed && decode_finish_event_count == 1 {
        1
    } else {
        0
    };
    let first_decode_finish_event =
        (decode_finish_event_count > 0).then_some("decode_iteration_finished:first");
    let last_decode_finish_event =
        (decode_finish_event_count > 0).then_some("decode_iteration_finished:last");
    let tpot_end_event = if decode_finish_event_count > 1 {
        "decode_iteration_finished:last"
    } else {
        "decode_iteration_finished:first"
    };
    let tpot_event = (tpot_sample_count > 0).then_some(tpot_end_event);
    let terminal_event = match observation.status.as_str() {
        "pending" => None,
        status => Some(status),
    };
    let metric_unavailable_reason = if completed {
        None
    } else {
        Some(format!(
            "request_not_completed:{}",
            observation.status.as_str()
        ))
    };
    let e2el_end_event = if completed {
        last_decode_finish_event
    } else {
        terminal_event
    };

    RequestMetricDerivation {
        completed,
        event_sourced,
        terminal_event,
        terminal_event_s: observation.status_time_s,
        metric_unavailable_reason,
        ttft_end_event: first_decode_finish_event,
        tpot_start_event: tpot_event.map(|_| "decode_iteration_finished:first"),
        tpot_end_event: tpot_event,
        e2el_end_event,
        throughput_duration_start_event: completed.then_some("arrived"),
        throughput_duration_end_event: completed.then_some(last_decode_finish_event).flatten(),
        decode_finish_event_count,
        tpot_sample_count,
    }
}

fn write_request_metric_derivation<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    score: &ScoredServingConfig,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let derivation = request_metric_derivation(observation);

    writeln!(writer, "{indent}      \"metric_derivation\": {{")?;
    writeln!(
        writer,
        "{indent}        \"metric_source\": {},",
        json_string(&observation.metric_source)
    )?;
    writeln!(
        writer,
        "{indent}        \"event_sourced\": {},",
        derivation.event_sourced
    )?;
    writeln!(
        writer,
        "{indent}        \"completed\": {},",
        derivation.completed
    )?;
    writeln!(
        writer,
        "{indent}        \"terminal_event\": {},",
        json_optional_string(derivation.terminal_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"terminal_event_ms\": {},",
        derivation
            .terminal_event_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"metric_unavailable_reason\": {},",
        json_optional_string(derivation.metric_unavailable_reason.as_deref())
    )?;
    writeln!(
        writer,
        "{indent}        \"included_in_measurement_window\": {},",
        request_in_measurement_window(observation, score)
    )?;
    writeln!(
        writer,
        "{indent}        \"measurement_window_source\": {},",
        json_string(&score.measurement_window.source)
    )?;
    writeln!(
        writer,
        "{indent}        \"measurement_start_ms\": {},",
        json_ms(score.measurement_window.start_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"measurement_end_ms\": {},",
        json_ms(score.measurement_window.end_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"arrival_ms\": {},",
        json_ms(observation.arrival_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"prefill_start_ms\": {},",
        json_ms(observation.prefill_start_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"kv_finish_ms\": {},",
        json_ms(observation.kv_finish_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"first_decode_start_ms\": {},",
        json_ms(observation.first_decode_start_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"first_decode_finish_ms\": {},",
        json_ms(observation.first_decode_finish_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"last_decode_finish_ms\": {},",
        json_ms(observation.last_decode_finish_s)
    )?;
    writeln!(writer, "{indent}        \"ttft_start_event\": \"arrived\",")?;
    writeln!(
        writer,
        "{indent}        \"ttft_end_event\": {},",
        json_optional_string(derivation.ttft_end_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_start_event\": {},",
        json_optional_string(derivation.tpot_start_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_end_event\": {},",
        json_optional_string(derivation.tpot_end_event)
    )?;
    writeln!(writer, "{indent}        \"e2el_start_event\": \"arrived\",")?;
    writeln!(
        writer,
        "{indent}        \"e2el_end_event\": {},",
        json_optional_string(derivation.e2el_end_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"throughput_duration_start_event\": {},",
        json_optional_string(derivation.throughput_duration_start_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"throughput_duration_end_event\": {},",
        json_optional_string(derivation.throughput_duration_end_event)
    )?;
    writeln!(
        writer,
        "{indent}        \"decode_iteration_count\": {},",
        observation.decode_iterations
    )?;
    writeln!(
        writer,
        "{indent}        \"decode_finish_event_count\": {},",
        derivation.decode_finish_event_count
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_sample_count\": {},",
        derivation.tpot_sample_count
    )?;
    writeln!(
        writer,
        "{indent}        \"output_tokens\": {},",
        request_observation_output_tokens(observation)
    )?;
    writeln!(
        writer,
        "{indent}        \"request_output_tokens_per_s\": {},",
        json_optional_value(request_observation_output_tokens_per_s(observation))
    )?;
    writeln!(
        writer,
        "{indent}        \"ttft_ms\": {},",
        json_ms(observation.ttft_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_ms\": {},",
        json_ms(observation.tpot_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"itl_ms\": {},",
        json_ms(observation.itl_s)
    )?;
    writeln!(
        writer,
        "{indent}        \"e2el_ms\": {}",
        json_ms(observation.e2el_s)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_request_rejection<W: Write>(
    writer: &mut W,
    indent: &str,
    rejection: Option<&ServingRejection>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(rejection) = rejection else {
        writeln!(
            writer,
            "{indent}      \"rejection\": null{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    writeln!(writer, "{indent}      \"rejection\": {{")?;
    writeln!(
        writer,
        "{indent}        \"phase\": {},",
        json_string(&rejection.phase)
    )?;
    writeln!(
        writer,
        "{indent}        \"category\": {},",
        json_string(&rejection.category)
    )?;
    writeln!(
        writer,
        "{indent}        \"resource\": {},",
        json_string(&rejection.resource)
    )?;
    writeln!(
        writer,
        "{indent}        \"code\": {},",
        json_string(&rejection.code)
    )?;
    writeln!(
        writer,
        "{indent}        \"observed\": {},",
        json_optional_value(rejection.observed)
    )?;
    writeln!(
        writer,
        "{indent}        \"limit\": {},",
        json_optional_value(rejection.limit)
    )?;
    writeln!(
        writer,
        "{indent}        \"unit\": {},",
        rejection
            .unit
            .as_deref()
            .map(json_string)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"remediation\": {},",
        rejection
            .remediation
            .as_deref()
            .map(json_string)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"message\": {}",
        json_string(&rejection.message)
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}

fn write_routing_candidates<W: Write>(
    writer: &mut W,
    indent: &str,
    candidates: &[ServingRouteCandidateObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"routing_candidates\": [")?;
    for (idx, candidate) in candidates.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"prefill_node\": {},",
            candidate.prefill_node
        )?;
        writeln!(
            writer,
            "{indent}          \"prefill_route_nodes\": [{}],",
            u32_list(&candidate.prefill_route_nodes)
        )?;
        write_nested_gpu_addr_list(
            writer,
            &format!("{indent}          "),
            "prefill_route_gpus",
            &candidate.prefill_route_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_node\": {},",
            candidate.decode_node
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_route_nodes\": [{}],",
            u32_list(&candidate.decode_route_nodes)
        )?;
        write_nested_gpu_addr_list(
            writer,
            &format!("{indent}          "),
            "decode_route_gpus",
            &candidate.decode_route_gpus,
            true,
        )?;
        writeln!(
            writer,
            "{indent}          \"selected\": {},",
            candidate.selected
        )?;
        writeln!(
            writer,
            "{indent}          \"routable\": {},",
            candidate.routable
        )?;
        writeln!(
            writer,
            "{indent}          \"rejection_reason\": {},",
            candidate
                .rejection_reason
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_e2el_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_e2el_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_kv_transfer_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_kv_transfer_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_kv_resource_wait_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_kv_resource_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_prefill_wait_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_prefill_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"estimated_decode_wait_ms\": {},",
            json_optional_seconds_ms(candidate.estimated_decode_wait_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_transfer_bytes\": {},",
            candidate.kv_transfer_bytes
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_transfer_bottlenecks\": [{}],",
            string_list(&candidate.kv_transfer_bottlenecks)
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_transfer_resources\": [{}]",
            string_list(&candidate.kv_transfer_resources)
        )?;
        if idx + 1 < candidates.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_nested_gpu_addr_list<W: Write>(
    writer: &mut W,
    field_indent: &str,
    name: &str,
    gpus: &[GpuAddr],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{field_indent}\"{name}\": [")?;
    for (idx, gpu) in gpus.iter().enumerate() {
        writeln!(writer, "{field_indent}  {{")?;
        writeln!(writer, "{field_indent}    \"node_id\": {},", gpu.node_id)?;
        writeln!(
            writer,
            "{field_indent}    \"local_gpu_id\": {}",
            gpu.local_gpu_id
        )?;
        if idx + 1 < gpus.len() {
            writeln!(writer, "{field_indent}  }},")?;
        } else {
            writeln!(writer, "{field_indent}  }}")?;
        }
    }
    writeln!(writer, "{field_indent}]{}", comma(trailing_comma))
}

fn write_gpu_addr_list<W: Write>(
    writer: &mut W,
    indent: &str,
    name: &str,
    gpus: &[GpuAddr],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"{name}\": [")?;
    for (idx, gpu) in gpus.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(writer, "{indent}          \"node_id\": {},", gpu.node_id)?;
        writeln!(
            writer,
            "{indent}          \"local_gpu_id\": {}",
            gpu.local_gpu_id
        )?;
        if idx + 1 < gpus.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_kv_block_ownership<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"kv_block_ownership\": [")?;
    for (idx, ownership) in observation.kv_block_ownership.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"allocation_id\": {},",
            json_string(&ownership.allocation_id)
        )?;
        writeln!(writer, "{indent}          \"owner\": {{")?;
        writeln!(
            writer,
            "{indent}            \"node_id\": {},",
            ownership.owner.node_id
        )?;
        writeln!(
            writer,
            "{indent}            \"local_gpu_id\": {}",
            ownership.owner.local_gpu_id
        )?;
        writeln!(writer, "{indent}          }},")?;
        writeln!(
            writer,
            "{indent}          \"owner_worker_slots\": [{}],",
            u32_list(&ownership.owner_worker_slots)
        )?;
        write_kv_worker_slot_ownership(writer, indent, ownership, true)?;
        writeln!(
            writer,
            "{indent}          \"decode_operation_ids\": [{}],",
            usize_list(&ownership.decode_operation_ids)
        )?;
        writeln!(
            writer,
            "{indent}          \"allocated_at_ms\": {},",
            json_ms(ownership.allocated_at_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"released_at_ms\": {},",
            json_ms(ownership.released_at_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"duration_ms\": {},",
            json_ms(ownership.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"block_start\": {},",
            ownership.block_start
        )?;
        writeln!(
            writer,
            "{indent}          \"block_end\": {},",
            ownership.block_end
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_sequences\": {},",
            ownership.decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}          \"resident_tokens\": {},",
            ownership.resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_blocks\": {},",
            ownership.kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}          \"allocated_kv_tokens\": {},",
            ownership.allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_fragmentation_tokens\": {},",
            ownership.kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"block_table_entries\": {},",
            ownership.block_table_entries
        )?;
        writeln!(
            writer,
            "{indent}          \"block_table_bytes\": {}",
            ownership.block_table_bytes
        )?;
        if idx + 1 < observation.kv_block_ownership.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_kv_worker_slot_ownership<W: Write>(
    writer: &mut W,
    indent: &str,
    ownership: &ServingKvBlockOwnershipObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}          \"worker_slot_ownership\": [")?;
    for (idx, slot) in ownership.worker_slot_ownership.iter().enumerate() {
        writeln!(writer, "{indent}            {{")?;
        writeln!(
            writer,
            "{indent}              \"allocation_id\": {},",
            json_string(&slot.allocation_id)
        )?;
        writeln!(writer, "{indent}              \"slot\": {},", slot.slot)?;
        writeln!(
            writer,
            "{indent}              \"block_start\": {},",
            slot.block_start
        )?;
        writeln!(
            writer,
            "{indent}              \"block_end\": {},",
            slot.block_end
        )?;
        writeln!(
            writer,
            "{indent}              \"decode_sequences\": {},",
            slot.decode_sequences
        )?;
        writeln!(
            writer,
            "{indent}              \"resident_tokens\": {},",
            slot.resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}              \"kv_blocks\": {},",
            slot.kv_blocks
        )?;
        writeln!(
            writer,
            "{indent}              \"allocated_kv_tokens\": {},",
            slot.allocated_kv_tokens
        )?;
        writeln!(
            writer,
            "{indent}              \"kv_fragmentation_tokens\": {},",
            slot.kv_fragmentation_tokens
        )?;
        writeln!(
            writer,
            "{indent}              \"block_table_entries\": {},",
            slot.block_table_entries
        )?;
        writeln!(
            writer,
            "{indent}              \"block_table_bytes\": {}",
            slot.block_table_bytes
        )?;
        if idx + 1 < ownership.worker_slot_ownership.len() {
            writeln!(writer, "{indent}            }},")?;
        } else {
            writeln!(writer, "{indent}            }}")?;
        }
    }
    writeln!(writer, "{indent}          ]{}", comma(trailing_comma))
}

fn write_kv_transfer_paths<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"kv_transfer_paths\": [")?;
    for (idx, path) in observation.kv_transfer_paths.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(writer, "{indent}          \"source\": {{")?;
        writeln!(
            writer,
            "{indent}            \"node_id\": {},",
            path.source.node_id
        )?;
        writeln!(
            writer,
            "{indent}            \"local_gpu_id\": {}",
            path.source.local_gpu_id
        )?;
        writeln!(writer, "{indent}          }},")?;
        writeln!(writer, "{indent}          \"destination\": {{")?;
        writeln!(
            writer,
            "{indent}            \"node_id\": {},",
            path.destination.node_id
        )?;
        writeln!(
            writer,
            "{indent}            \"local_gpu_id\": {}",
            path.destination.local_gpu_id
        )?;
        writeln!(writer, "{indent}          }},")?;
        writeln!(
            writer,
            "{indent}          \"latency_ms\": {},",
            json_ms(path.latency_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"bottleneck_bandwidth_gbps\": {},",
            json_optional_f64(path.bottleneck_bandwidth_gbps)
        )?;
        writeln!(
            writer,
            "{indent}          \"resources\": [{}],",
            string_list(&path.resources)
        )?;
        write_kv_transfer_path_resource_details(writer, indent, &path.resource_details)?;
        if idx + 1 < observation.kv_transfer_paths.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_kv_transfer_path_resource_details<W: Write>(
    writer: &mut W,
    indent: &str,
    resources: &[ServingKvTransferPathResourceObservation],
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}          \"resource_details\": [")?;
    for (idx, resource) in resources.iter().enumerate() {
        writeln!(writer, "{indent}            {{")?;
        writeln!(
            writer,
            "{indent}              \"kind\": {},",
            json_string(&resource.kind)
        )?;
        writeln!(
            writer,
            "{indent}              \"label\": {},",
            json_string(&resource.label)
        )?;
        writeln!(
            writer,
            "{indent}              \"bandwidth_gbps\": {},",
            json_optional_f64(resource.bandwidth_gbps)
        )?;
        writeln!(
            writer,
            "{indent}              \"latency_ms\": {},",
            json_ms(resource.latency_s)
        )?;
        writeln!(
            writer,
            "{indent}              \"rail_id\": {},",
            json_optional_u32(resource.rail_id)
        )?;
        write_kv_transfer_path_endpoint(writer, indent, "from", resource.from.as_ref(), true)?;
        write_kv_transfer_path_endpoint(writer, indent, "to", resource.to.as_ref(), false)?;
        if idx + 1 < resources.len() {
            writeln!(writer, "{indent}            }},")?;
        } else {
            writeln!(writer, "{indent}            }}")?;
        }
    }
    writeln!(writer, "{indent}          ]")
}

pub(super) fn write_kv_transfer_path_endpoint<W: Write>(
    writer: &mut W,
    indent: &str,
    field: &str,
    endpoint: Option<&ServingKvTransferPathEndpointObservation>,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let Some(endpoint) = endpoint else {
        writeln!(
            writer,
            "{indent}              \"{field}\": null{}",
            comma(trailing_comma)
        )?;
        return Ok(());
    };

    writeln!(writer, "{indent}              \"{field}\": {{")?;
    writeln!(
        writer,
        "{indent}                \"kind\": {},",
        json_string(&endpoint.kind)
    )?;
    writeln!(
        writer,
        "{indent}                \"node_id\": {},",
        json_optional_u32(endpoint.node_id)
    )?;
    writeln!(
        writer,
        "{indent}                \"local_gpu_id\": {},",
        json_optional_u32(endpoint.local_gpu_id)
    )?;
    writeln!(
        writer,
        "{indent}                \"nic_id\": {},",
        json_optional_u32(endpoint.nic_id)
    )?;
    writeln!(
        writer,
        "{indent}                \"rail_id\": {}",
        json_optional_u32(endpoint.rail_id)
    )?;
    writeln!(writer, "{indent}              }}{}", comma(trailing_comma))
}

fn write_request_phase_spans<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let spans = request_phase_spans(observation);
    writeln!(writer, "{indent}      \"phase_spans\": [")?;
    for (idx, (phase, start_s, finish_s)) in spans.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(*start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(*finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"duration_ms\": {}",
            json_ms((finish_s - start_s).max(0.0))
        )?;
        if idx + 1 < spans.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn request_phase_spans(observation: &ServingRequestObservation) -> Vec<(&'static str, f64, f64)> {
    [
        (
            "queued_for_prefill",
            observation.arrival_s,
            observation.prefill_start_s,
        ),
        (
            "prefilling",
            observation.prefill_start_s,
            observation.prefill_finish_s,
        ),
        (
            "queued_for_kv_transfer",
            observation.prefill_finish_s,
            observation.kv_start_s,
        ),
        (
            "transferring_kv",
            observation.kv_start_s,
            observation.kv_finish_s,
        ),
        (
            "queued_for_decode",
            observation.kv_finish_s,
            observation.first_decode_start_s,
        ),
        (
            "decoding",
            observation.first_decode_start_s,
            observation.last_decode_finish_s,
        ),
    ]
    .into_iter()
    .filter(|(_, start_s, finish_s)| {
        start_s.is_finite() && finish_s.is_finite() && *finish_s >= *start_s
    })
    .collect()
}

fn write_request_phase_breakdown<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"phase_breakdown\": [")?;
    for (idx, phase) in observation.phase_breakdown.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&phase.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"category\": {},",
            json_string(phase.category.as_str())
        )?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(phase.start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(phase.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"duration_ms\": {},",
            json_ms(phase.duration_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"contributes_to_ttft\": {},",
            phase.contributes_to_ttft
        )?;
        writeln!(
            writer,
            "{indent}          \"contributes_to_e2el\": {}",
            phase.contributes_to_e2el
        )?;
        if idx + 1 < observation.phase_breakdown.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_request_lifecycle_events<W: Write>(
    writer: &mut W,
    indent: &str,
    observation: &ServingRequestObservation,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"lifecycle_events\": [")?;
    for (idx, event) in observation.lifecycle_events.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"event\": {},",
            json_string(event.kind.as_str())
        )?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&event.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"at_ms\": {},",
            json_ms(event.at_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"decode_iteration\": {},",
            event
                .decode_iteration
                .map(|iteration| iteration.to_string())
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}          \"message\": {}",
            event
                .message
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        if idx + 1 < observation.lifecycle_events.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_worker_assignments<W: Write>(
    writer: &mut W,
    indent: &str,
    assignments: &[ServingWorkerAssignmentObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"worker_assignments\": [")?;
    for (idx, assignment) in assignments.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&assignment.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"node_id\": {},",
            assignment.node_id
        )?;
        writeln!(
            writer,
            "{indent}          \"local_gpu_id\": {},",
            assignment.local_gpu_id
        )?;
        writeln!(writer, "{indent}          \"slot\": {},", assignment.slot)?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(assignment.start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(assignment.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"operation_ids\": [{}]",
            usize_list(&assignment.operation_ids)
        )?;
        if idx + 1 < assignments.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

fn write_request_worker_summary<W: Write>(
    writer: &mut W,
    indent: &str,
    summaries: &[ServingRequestWorkerSummaryObservation],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"worker_summary\": [")?;
    for (idx, summary) in summaries.iter().enumerate() {
        writeln!(writer, "{indent}        {{")?;
        writeln!(
            writer,
            "{indent}          \"role\": {},",
            json_string(&summary.role)
        )?;
        writeln!(
            writer,
            "{indent}          \"phase\": {},",
            json_string(&summary.phase)
        )?;
        writeln!(
            writer,
            "{indent}          \"node_id\": {},",
            summary.node_id
        )?;
        writeln!(
            writer,
            "{indent}          \"local_gpu_id\": {},",
            summary.local_gpu_id
        )?;
        writeln!(
            writer,
            "{indent}          \"worker_slots\": [{}],",
            u32_list(&summary.worker_slots)
        )?;
        writeln!(
            writer,
            "{indent}          \"operation_ids\": [{}],",
            usize_list(&summary.operation_ids)
        )?;
        writeln!(
            writer,
            "{indent}          \"assignment_count\": {},",
            summary.assignment_count
        )?;
        writeln!(
            writer,
            "{indent}          \"start_ms\": {},",
            json_ms(summary.start_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"finish_ms\": {},",
            json_ms(summary.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}          \"resident_tokens\": {},",
            summary.resident_tokens
        )?;
        writeln!(
            writer,
            "{indent}          \"kv_blocks\": {}",
            summary.kv_blocks
        )?;
        if idx + 1 < summaries.len() {
            writeln!(writer, "{indent}        }},")?;
        } else {
            writeln!(writer, "{indent}        }}")?;
        }
    }
    writeln!(writer, "{indent}      ]{}", comma(trailing_comma))
}

pub(super) fn write_decode_iterations<W: Write>(
    writer: &mut W,
    indent: &str,
    observations: &[ServingDecodeIterationObservation],
    limit: Option<usize>,
    include_operation_ids: bool,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let displayed_observations =
        limit.map_or(observations.len(), |limit| limit.min(observations.len()));
    writeln!(
        writer,
        "{indent}  \"decode_iteration_count\": {},",
        observations.len()
    )?;
    writeln!(
        writer,
        "{indent}  \"decode_iterations_truncated\": {},",
        displayed_observations < observations.len()
    )?;
    writeln!(writer, "{indent}  \"decode_iterations\": [")?;
    for (idx, observation) in observations.iter().take(displayed_observations).enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"iteration_idx\": {},",
            observation.iteration_idx
        )?;
        writeln!(
            writer,
            "{indent}      \"decode_nodes\": [{}],",
            u32_list(&observation.decode_nodes)
        )?;
        writeln!(
            writer,
            "{indent}      \"request_indices\": [{}],",
            u32_list(&observation.request_indices)
        )?;
        writeln!(
            writer,
            "{indent}      \"operation_count\": {},",
            observation.operation_ids.len()
        )?;
        if include_operation_ids {
            writeln!(
                writer,
                "{indent}      \"operation_ids\": [{}],",
                usize_list(&observation.operation_ids)
            )?;
        }
        writeln!(
            writer,
            "{indent}      \"start_ms\": {},",
            json_ms(observation.start_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"finish_ms\": {},",
            json_ms(observation.finish_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"latency_ms\": {},",
            json_ms(observation.latency_s)
        )?;
        writeln!(
            writer,
            "{indent}      \"batch_tokens\": {},",
            observation.batch_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"first_token_batch_tokens\": {},",
            observation.first_token_batch_tokens
        )?;
        writeln!(
            writer,
            "{indent}      \"tail_token_batch_tokens\": {}",
            observation.tail_token_batch_tokens
        )?;
        if idx + 1 < displayed_observations {
            writeln!(writer, "{indent}    }},")?;
        } else {
            writeln!(writer, "{indent}    }}")?;
        }
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

fn write_request_slo<W: Write>(
    writer: &mut W,
    indent: &str,
    slo: &crate::ServingRequestSlo,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}      \"slo\": {{")?;
    writeln!(
        writer,
        "{indent}        \"ttft_ms\": {},",
        slo.ttft_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"tpot_ms\": {},",
        slo.tpot_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"itl_ms\": {},",
        slo.itl_s.map(json_ms).unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(
        writer,
        "{indent}        \"e2el_ms\": {}",
        slo.e2el_s
            .map(json_ms)
            .unwrap_or_else(|| "null".to_string())
    )?;
    writeln!(writer, "{indent}      }}{}", comma(trailing_comma))
}
