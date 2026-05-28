use super::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct TraceWindow {
    pub(super) start_s: Option<f64>,
    pub(super) end_s: Option<f64>,
    pub(super) time_scale: f64,
    pub(super) arrival_offset_s: f64,
    pub(super) has_controls: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct TraceReplay {
    pub(super) repeat_count: u32,
    pub(super) repeat_interval_s: Option<f64>,
    pub(super) has_controls: bool,
}

pub(super) fn parse_trace_requests(
    requests: Option<Vec<ServingTraceRequestSection>>,
    trace_csv: Option<String>,
    trace_jsonl: Option<String>,
    base_dir: Option<&Path>,
) -> Result<Vec<ServingTraceRequest>, ConfigError> {
    let source_count = usize::from(requests.is_some())
        + usize::from(trace_csv.is_some())
        + usize::from(trace_jsonl.is_some());
    if source_count > 1 {
        return Err(ConfigError::new(
            "serving.traffic can set only one of inline requests, trace_csv, or trace_jsonl",
        ));
    }
    if let Some(trace_csv) = trace_csv {
        let trace_path = resolve_config_path(&trace_csv, base_dir);
        let contents = fs::read_to_string(&trace_path).map_err(|err| {
            ConfigError::new(format!(
                "failed to read serving.traffic.trace_csv {}: {err}",
                trace_path.display()
            ))
        })?;
        return parse_trace_csv(&contents, &trace_path);
    }
    if let Some(trace_jsonl) = trace_jsonl {
        let trace_path = resolve_config_path(&trace_jsonl, base_dir);
        let contents = fs::read_to_string(&trace_path).map_err(|err| {
            ConfigError::new(format!(
                "failed to read serving.traffic.trace_jsonl {}: {err}",
                trace_path.display()
            ))
        })?;
        return parse_trace_jsonl(&contents, &trace_path);
    }

    let Some(requests) = requests else {
        return Ok(Vec::new());
    };
    parse_trace_request_sections(requests)
}

pub(super) fn parse_trace_request_sections(
    requests: Vec<ServingTraceRequestSection>,
) -> Result<Vec<ServingTraceRequest>, ConfigError> {
    let mut parsed = Vec::with_capacity(requests.len());
    for (idx, request) in requests.into_iter().enumerate() {
        if request.batch_size == 0 {
            return Err(ConfigError::new(format!(
                "serving.traffic.requests[{idx}].batch_size must be greater than zero"
            )));
        }
        if request.prompt_tokens == 0 {
            return Err(ConfigError::new(format!(
                "serving.traffic.requests[{idx}].prompt_tokens must be greater than zero"
            )));
        }
        if request.decode_tokens == 0 {
            return Err(ConfigError::new(format!(
                "serving.traffic.requests[{idx}].decode_tokens must be greater than zero"
            )));
        }
        validate_optional_max_sequence_tokens(
            &format!("serving.traffic.requests[{idx}].max_sequence_tokens"),
            request.max_sequence_tokens,
            request.prompt_tokens,
            request.decode_tokens,
        )?;
        let prefix_cache_hit_rate = parse_cache_hit_rate(
            &format!("serving.traffic.requests[{idx}].prefix_cache_hit_rate"),
            request.prefix_cache_hit_rate,
        )?;
        if request.prefix_cache_hit_tokens.is_some() && prefix_cache_hit_rate.is_some() {
            return Err(ConfigError::new(format!(
                "serving.traffic.requests[{idx}] cannot set both prefix_cache_hit_tokens and prefix_cache_hit_rate"
            )));
        }
        if let Some(prefix_cache_hit_tokens) = request.prefix_cache_hit_tokens
            && prefix_cache_hit_tokens > request.prompt_tokens
        {
            return Err(ConfigError::new(format!(
                "serving.traffic.requests[{idx}].prefix_cache_hit_tokens must be less than or equal to prompt_tokens"
            )));
        }
        if request.arrival_s.is_some() && request.arrival_ms.is_some() {
            return Err(ConfigError::new(format!(
                "serving.traffic.requests[{idx}] cannot set both arrival_s and arrival_ms"
            )));
        }
        let arrival_s = match (request.arrival_s, request.arrival_ms) {
            (Some(arrival_s), None) => arrival_s,
            (None, Some(arrival_ms)) => arrival_ms / 1000.0,
            (None, None) => {
                return Err(ConfigError::new(format!(
                    "serving.traffic.requests[{idx}] requires arrival_s or arrival_ms"
                )));
            }
            (Some(_), Some(_)) => unreachable!(),
        };
        if !arrival_s.is_finite() || arrival_s < 0.0 {
            return Err(ConfigError::new(format!(
                "serving.traffic.requests[{idx}] arrival must be finite and non-negative"
            )));
        }
        let slo = parse_inline_request_slo(idx, &request)?;
        let deadline_s = parse_inline_deadline_s(idx, arrival_s, &request)?;
        let cancellation_s = parse_inline_cancellation_s(idx, arrival_s, &request)?;
        parsed.push(ServingTraceRequest {
            request_id: optional_nonempty_string(request.request_id),
            tenant: optional_nonempty_string(request.tenant),
            model_id: optional_nonempty_string(request.model_id),
            cache_key: optional_nonempty_string(request.cache_key),
            arrival_s,
            priority: request.priority.unwrap_or(0),
            batch_size: request.batch_size,
            prompt_tokens: request.prompt_tokens,
            decode_tokens: request.decode_tokens,
            max_sequence_tokens: request.max_sequence_tokens,
            prefix_cache_hit_tokens: request.prefix_cache_hit_tokens,
            prefix_cache_hit_rate,
            slo,
            deadline_s,
            cancellation_s,
        });
    }
    Ok(parsed)
}

pub(super) fn parse_inline_request_slo(
    idx: usize,
    request: &ServingTraceRequestSection,
) -> Result<ServingRequestSlo, ConfigError> {
    Ok(ServingRequestSlo {
        ttft_s: parse_inline_request_slo_value(
            idx,
            "ttft_slo",
            request.ttft_slo_s,
            request.ttft_slo_ms,
        )?,
        tpot_s: parse_inline_request_slo_value(
            idx,
            "tpot_slo",
            request.tpot_slo_s,
            request.tpot_slo_ms,
        )?,
        itl_s: parse_inline_request_slo_value(
            idx,
            "itl_slo",
            request.itl_slo_s,
            request.itl_slo_ms,
        )?,
        e2el_s: parse_inline_request_slo_value(
            idx,
            "e2el_slo",
            request.e2el_slo_s,
            request.e2el_slo_ms,
        )?,
    })
}

pub(super) fn parse_inline_request_slo_value(
    idx: usize,
    name: &str,
    seconds: Option<f64>,
    millis: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    let value = parse_optional_trace_seconds(
        &format!("serving.traffic.requests[].{name}_s"),
        seconds,
        &format!("serving.traffic.requests[].{name}_ms"),
        millis,
    )?;
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}].{name} must be positive"
        )));
    }
    Ok(value)
}

pub(super) fn parse_inline_deadline_s(
    idx: usize,
    arrival_s: f64,
    request: &ServingTraceRequestSection,
) -> Result<Option<f64>, ConfigError> {
    let deadline_s = parse_optional_trace_seconds(
        "serving.traffic.requests[].deadline_s",
        request.deadline_s,
        "serving.traffic.requests[].deadline_ms",
        request.deadline_ms,
    )?;
    let deadline_after_s = parse_optional_trace_seconds(
        "serving.traffic.requests[].deadline_after_s",
        request.deadline_after_s,
        "serving.traffic.requests[].deadline_after_ms",
        request.deadline_after_ms,
    )?;
    if deadline_s.is_some() && deadline_after_s.is_some() {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] cannot set both deadline and deadline_after"
        )));
    }
    if let Some(deadline_after_s) = deadline_after_s
        && deadline_after_s < 0.0
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] deadline_after must be non-negative"
        )));
    }
    let deadline_s = deadline_s.or_else(|| deadline_after_s.map(|value| arrival_s + value));
    if let Some(deadline_s) = deadline_s
        && !deadline_s.is_finite()
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] deadline must be finite"
        )));
    }
    if let Some(deadline_s) = deadline_s
        && deadline_s < arrival_s
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] deadline must be greater than or equal to arrival"
        )));
    }
    Ok(deadline_s)
}

pub(super) fn parse_inline_cancellation_s(
    idx: usize,
    arrival_s: f64,
    request: &ServingTraceRequestSection,
) -> Result<Option<f64>, ConfigError> {
    let cancellation_s = parse_optional_trace_seconds(
        "serving.traffic.requests[].cancellation_s",
        request.cancellation_s,
        "serving.traffic.requests[].cancellation_ms",
        request.cancellation_ms,
    )?;
    let cancel_after_s = parse_optional_trace_seconds(
        "serving.traffic.requests[].cancel_after_s",
        request.cancel_after_s,
        "serving.traffic.requests[].cancel_after_ms",
        request.cancel_after_ms,
    )?;
    if cancellation_s.is_some() && cancel_after_s.is_some() {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] cannot set both cancellation and cancel_after"
        )));
    }
    if let Some(cancel_after_s) = cancel_after_s
        && cancel_after_s < 0.0
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] cancel_after must be non-negative"
        )));
    }
    let cancellation_s = cancellation_s.or_else(|| cancel_after_s.map(|value| arrival_s + value));
    if let Some(cancellation_s) = cancellation_s
        && !cancellation_s.is_finite()
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] cancellation must be finite"
        )));
    }
    if let Some(cancellation_s) = cancellation_s
        && cancellation_s < arrival_s
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.requests[{idx}] cancellation must be greater than or equal to arrival"
        )));
    }
    Ok(cancellation_s)
}

pub(super) fn parse_trace_window(
    traffic: &ServingTrafficSection,
) -> Result<TraceWindow, ConfigError> {
    let start_s = parse_optional_trace_seconds(
        "serving.traffic.trace_start_s",
        traffic.trace_start_s,
        "serving.traffic.trace_start_ms",
        traffic.trace_start_ms,
    )?;
    let end_s = parse_optional_trace_seconds(
        "serving.traffic.trace_end_s",
        traffic.trace_end_s,
        "serving.traffic.trace_end_ms",
        traffic.trace_end_ms,
    )?;
    if let Some(start_s) = start_s
        && start_s < 0.0
    {
        return Err(ConfigError::new(
            "serving.traffic.trace_start must be non-negative",
        ));
    }
    if let Some(end_s) = end_s
        && end_s < 0.0
    {
        return Err(ConfigError::new(
            "serving.traffic.trace_end must be non-negative",
        ));
    }
    if let (Some(start_s), Some(end_s)) = (start_s, end_s)
        && end_s < start_s
    {
        return Err(ConfigError::new(
            "serving.traffic.trace_end must be greater than or equal to trace_start",
        ));
    }

    let time_scale = traffic.trace_time_scale.unwrap_or(1.0);
    if !time_scale.is_finite() || time_scale <= 0.0 {
        return Err(ConfigError::new(
            "serving.traffic.trace_time_scale must be finite and positive",
        ));
    }
    let arrival_offset_s = parse_optional_trace_seconds(
        "serving.traffic.trace_arrival_offset_s",
        traffic.trace_arrival_offset_s,
        "serving.traffic.trace_arrival_offset_ms",
        traffic.trace_arrival_offset_ms,
    )?
    .unwrap_or(0.0);

    Ok(TraceWindow {
        start_s,
        end_s,
        time_scale,
        arrival_offset_s,
        has_controls: start_s.is_some()
            || end_s.is_some()
            || traffic.trace_time_scale.is_some()
            || traffic.trace_arrival_offset_s.is_some()
            || traffic.trace_arrival_offset_ms.is_some(),
    })
}

pub(super) fn parse_trace_replay(
    traffic: &ServingTrafficSection,
) -> Result<TraceReplay, ConfigError> {
    let repeat_count = traffic.trace_repeat_count.unwrap_or(1);
    if repeat_count == 0 {
        return Err(ConfigError::new(
            "serving.traffic.trace_repeat_count must be greater than zero",
        ));
    }
    let repeat_interval_s = parse_optional_trace_seconds(
        "serving.traffic.trace_repeat_interval_s",
        traffic.trace_repeat_interval_s,
        "serving.traffic.trace_repeat_interval_ms",
        traffic.trace_repeat_interval_ms,
    )?;
    if let Some(repeat_interval_s) = repeat_interval_s
        && repeat_interval_s <= 0.0
    {
        return Err(ConfigError::new(
            "serving.traffic.trace_repeat_interval_s/trace_repeat_interval_ms must be positive",
        ));
    }
    Ok(TraceReplay {
        repeat_count,
        repeat_interval_s,
        has_controls: traffic.trace_repeat_count.is_some()
            || traffic.trace_repeat_interval_s.is_some()
            || traffic.trace_repeat_interval_ms.is_some(),
    })
}

pub(super) fn parse_optional_trace_seconds(
    seconds_name: &str,
    seconds: Option<f64>,
    millis_name: &str,
    millis: Option<f64>,
) -> Result<Option<f64>, ConfigError> {
    if seconds.is_some() && millis.is_some() {
        return Err(ConfigError::new(format!(
            "{seconds_name} and {millis_name} cannot both be set"
        )));
    }
    let value = match (seconds, millis) {
        (Some(seconds), None) => Some(seconds),
        (None, Some(millis)) => Some(millis / 1000.0),
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!(),
    };
    if let Some(value) = value
        && !value.is_finite()
    {
        return Err(ConfigError::new(format!(
            "{seconds_name}/{millis_name} must be finite"
        )));
    }
    Ok(value)
}

pub(super) fn optional_nonempty_string(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(super) fn apply_trace_window(
    requests: Vec<ServingTraceRequest>,
    window: TraceWindow,
) -> Result<Vec<ServingTraceRequest>, ConfigError> {
    if requests.is_empty() {
        if window.has_controls {
            return Err(ConfigError::new(
                "serving.traffic trace window controls require inline requests, trace_csv, or trace_jsonl",
            ));
        }
        return Ok(requests);
    }

    let start_s = window.start_s.unwrap_or(0.0);
    let end_s = window.end_s.unwrap_or(f64::INFINITY);
    let mut filtered = Vec::new();
    for mut request in requests {
        if request.arrival_s < start_s || request.arrival_s > end_s {
            continue;
        }
        request.arrival_s =
            (request.arrival_s - start_s) * window.time_scale + window.arrival_offset_s;
        if let Some(deadline_s) = request.deadline_s {
            request.deadline_s =
                Some((deadline_s - start_s) * window.time_scale + window.arrival_offset_s);
        }
        if let Some(cancellation_s) = request.cancellation_s {
            request.cancellation_s =
                Some((cancellation_s - start_s) * window.time_scale + window.arrival_offset_s);
        }
        if !request.arrival_s.is_finite() || request.arrival_s < 0.0 {
            return Err(ConfigError::new(
                "serving.traffic trace window produced a negative or non-finite arrival",
            ));
        }
        if let Some(cancellation_s) = request.cancellation_s
            && (!cancellation_s.is_finite() || cancellation_s < request.arrival_s)
        {
            return Err(ConfigError::new(
                "serving.traffic trace window produced an invalid cancellation time",
            ));
        }
        if let Some(deadline_s) = request.deadline_s
            && (!deadline_s.is_finite() || deadline_s < request.arrival_s)
        {
            return Err(ConfigError::new(
                "serving.traffic trace window produced an invalid deadline time",
            ));
        }
        filtered.push(request);
    }

    if filtered.is_empty() {
        return Err(ConfigError::new(
            "serving.traffic trace window removed all requests",
        ));
    }
    Ok(filtered)
}

pub(super) fn apply_trace_replay(
    requests: Vec<ServingTraceRequest>,
    replay: TraceReplay,
) -> Result<Vec<ServingTraceRequest>, ConfigError> {
    if requests.is_empty() {
        if replay.has_controls {
            return Err(ConfigError::new(
                "serving.traffic trace replay controls require inline requests, trace_csv, or trace_jsonl",
            ));
        }
        return Ok(requests);
    }
    if replay.repeat_count == 1 {
        return Ok(requests);
    }
    let repeat_interval_s = match replay.repeat_interval_s {
        Some(repeat_interval_s) => repeat_interval_s,
        None => default_trace_repeat_interval_s(&requests)?,
    };
    let base_len = requests.len();
    let mut repeated = Vec::with_capacity(base_len.saturating_mul(replay.repeat_count as usize));
    for repeat_idx in 0..replay.repeat_count {
        let offset_s = f64::from(repeat_idx) * repeat_interval_s;
        for request in &requests {
            let mut request = request.clone();
            if repeat_idx > 0
                && let Some(request_id) = request.request_id.as_mut()
            {
                request_id.push_str(&format!("#r{}", repeat_idx + 1));
            }
            request.arrival_s += offset_s;
            if let Some(deadline_s) = request.deadline_s.as_mut() {
                *deadline_s += offset_s;
            }
            if let Some(cancellation_s) = request.cancellation_s.as_mut() {
                *cancellation_s += offset_s;
            }
            repeated.push(request);
        }
    }
    Ok(repeated)
}

pub(super) fn validate_unique_trace_request_ids(
    requests: &[ServingTraceRequest],
) -> Result<(), ConfigError> {
    let mut seen = BTreeMap::new();
    for (idx, request) in requests.iter().enumerate() {
        let Some(request_id) = request.request_id.as_deref() else {
            continue;
        };
        if let Some(first_idx) = seen.insert(request_id.to_string(), idx) {
            return Err(ConfigError::new(format!(
                "serving.traffic request_id '{request_id}' is duplicated by trace request {idx}; first seen at trace request {first_idx}"
            )));
        }
    }
    Ok(())
}

pub(super) fn default_trace_repeat_interval_s(
    requests: &[ServingTraceRequest],
) -> Result<f64, ConfigError> {
    let mut arrivals: Vec<_> = requests.iter().map(|request| request.arrival_s).collect();
    arrivals.sort_by(f64::total_cmp);
    let first = arrivals[0];
    let last = arrivals[arrivals.len() - 1];
    let min_gap = arrivals
        .windows(2)
        .filter_map(|window| {
            let gap = window[1] - window[0];
            (gap > 0.0).then_some(gap)
        })
        .min_by(f64::total_cmp);
    let Some(min_gap) = min_gap else {
        return Err(ConfigError::new(
            "serving.traffic.trace_repeat_interval_s is required when repeating a single-arrival trace",
        ));
    };
    Ok((last - first) + min_gap)
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(super) struct MeasurementWindowConfig {
    pub(super) start_s: Option<f64>,
    pub(super) end_s: Option<f64>,
    pub(super) warmup_s: Option<f64>,
    pub(super) cooldown_s: Option<f64>,
}

pub(super) fn parse_measurement_window(
    traffic: &ServingTrafficSection,
) -> Result<MeasurementWindowConfig, ConfigError> {
    let start_s = parse_optional_trace_seconds(
        "serving.traffic.measurement_start_s",
        traffic.measurement_start_s,
        "serving.traffic.measurement_start_ms",
        traffic.measurement_start_ms,
    )?;
    let end_s = parse_optional_trace_seconds(
        "serving.traffic.measurement_end_s",
        traffic.measurement_end_s,
        "serving.traffic.measurement_end_ms",
        traffic.measurement_end_ms,
    )?;
    let warmup_s = parse_optional_trace_seconds(
        "serving.traffic.measurement_warmup_s",
        traffic.measurement_warmup_s,
        "serving.traffic.measurement_warmup_ms",
        traffic.measurement_warmup_ms,
    )?;
    let cooldown_s = parse_optional_trace_seconds(
        "serving.traffic.measurement_cooldown_s",
        traffic.measurement_cooldown_s,
        "serving.traffic.measurement_cooldown_ms",
        traffic.measurement_cooldown_ms,
    )?;
    if start_s.is_some() && warmup_s.is_some() {
        return Err(ConfigError::new(
            "serving.traffic must not set both measurement_start and measurement_warmup",
        ));
    }
    if end_s.is_some() && cooldown_s.is_some() {
        return Err(ConfigError::new(
            "serving.traffic must not set both measurement_end and measurement_cooldown",
        ));
    }
    if let Some(start_s) = start_s
        && start_s < 0.0
    {
        return Err(ConfigError::new(
            "serving.traffic.measurement_start must be non-negative",
        ));
    }
    if let Some(end_s) = end_s
        && end_s < 0.0
    {
        return Err(ConfigError::new(
            "serving.traffic.measurement_end must be non-negative",
        ));
    }
    if let Some(warmup_s) = warmup_s
        && warmup_s < 0.0
    {
        return Err(ConfigError::new(
            "serving.traffic.measurement_warmup must be non-negative",
        ));
    }
    if let Some(cooldown_s) = cooldown_s
        && cooldown_s < 0.0
    {
        return Err(ConfigError::new(
            "serving.traffic.measurement_cooldown must be non-negative",
        ));
    }
    if let (Some(start_s), Some(end_s)) = (start_s, end_s)
        && end_s < start_s
    {
        return Err(ConfigError::new(
            "serving.traffic.measurement_end must be greater than or equal to measurement_start",
        ));
    }
    Ok(MeasurementWindowConfig {
        start_s,
        end_s,
        warmup_s,
        cooldown_s,
    })
}

pub(super) fn parse_trace_jsonl(
    contents: &str,
    source_path: &Path,
) -> Result<Vec<ServingTraceRequest>, ConfigError> {
    let mut requests = Vec::new();
    for (line_idx, line) in contents.lines().enumerate() {
        let line_number = line_idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let request: ServingTraceRequestSection = serde_json::from_str(trimmed).map_err(|err| {
            ConfigError::new(format!(
                "serving.traffic.trace_jsonl {} line {line_number} has invalid JSON request: {err}",
                source_path.display()
            ))
        })?;
        requests.push(request);
    }
    if requests.is_empty() {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_jsonl {} contains no requests",
            source_path.display()
        )));
    }
    parse_trace_request_sections(requests)
}

pub(super) fn parse_trace_csv(
    contents: &str,
    source_path: &Path,
) -> Result<Vec<ServingTraceRequest>, ConfigError> {
    let mut data_lines = contents
        .lines()
        .enumerate()
        .filter(|(_, line)| trace_csv_is_data_line(line));
    let Some((_header_line_idx, header_line)) = data_lines.next() else {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} is empty",
            source_path.display()
        )));
    };
    let header_fields = trace_csv_fields(header_line);
    let headers = trace_csv_headers(&header_fields);
    let batch_size_idx = required_trace_csv_header(&headers, "batch_size", source_path)?;
    let prompt_tokens_idx = required_trace_csv_header(&headers, "prompt_tokens", source_path)?;
    let decode_tokens_idx = required_trace_csv_header(&headers, "decode_tokens", source_path)?;
    let request_id_idx = optional_trace_csv_header(&headers, &["request_id", "id"]);
    let tenant_idx = optional_trace_csv_header(&headers, &["tenant", "tenant_id"]);
    let model_id_idx = optional_trace_csv_header(&headers, &["model_id", "model"]);
    let cache_key_idx = optional_trace_csv_header(&headers, &["cache_key", "prefix_cache_key"]);
    let priority_idx = optional_trace_csv_header(&headers, &["priority"]);
    let arrival_s_idx = headers.get("arrival_s").copied();
    let arrival_ms_idx = headers.get("arrival_ms").copied();
    let ttft_slo_s_idx = headers.get("ttft_slo_s").copied();
    let ttft_slo_ms_idx = headers.get("ttft_slo_ms").copied();
    let tpot_slo_s_idx = headers.get("tpot_slo_s").copied();
    let tpot_slo_ms_idx = headers.get("tpot_slo_ms").copied();
    let itl_slo_s_idx = headers.get("itl_slo_s").copied();
    let itl_slo_ms_idx = headers.get("itl_slo_ms").copied();
    let e2el_slo_s_idx = headers.get("e2el_slo_s").copied();
    let e2el_slo_ms_idx = headers.get("e2el_slo_ms").copied();
    let deadline_s_idx = headers.get("deadline_s").copied();
    let deadline_ms_idx = headers.get("deadline_ms").copied();
    let deadline_after_s_idx = headers.get("deadline_after_s").copied();
    let deadline_after_ms_idx = headers.get("deadline_after_ms").copied();
    let cancellation_s_idx = headers.get("cancellation_s").copied();
    let cancellation_ms_idx = headers.get("cancellation_ms").copied();
    let cancel_after_s_idx = headers.get("cancel_after_s").copied();
    let cancel_after_ms_idx = headers.get("cancel_after_ms").copied();
    let max_sequence_tokens_idx = headers.get("max_sequence_tokens").copied();
    let prefix_cache_hit_tokens_idx = headers.get("prefix_cache_hit_tokens").copied();
    let prefix_cache_hit_rate_idx = optional_trace_csv_header(
        &headers,
        &["prefix_cache_hit_rate", "prefix_cache_hit_ratio"],
    );
    if arrival_s_idx.is_none() && arrival_ms_idx.is_none() {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} requires an arrival_s or arrival_ms column",
            source_path.display()
        )));
    }
    let mut parsed = Vec::new();
    for (line_idx, line) in data_lines {
        let fields = trace_csv_fields(line);
        let line_number = line_idx + 1;
        let batch_size = required_positive_trace_csv_u32(
            &fields,
            batch_size_idx,
            source_path,
            line_number,
            "batch_size",
        )?;
        let prompt_tokens = required_positive_trace_csv_u32(
            &fields,
            prompt_tokens_idx,
            source_path,
            line_number,
            "prompt_tokens",
        )?;
        let decode_tokens = required_positive_trace_csv_u32(
            &fields,
            decode_tokens_idx,
            source_path,
            line_number,
            "decode_tokens",
        )?;
        let max_sequence_tokens = optional_positive_trace_csv_u32(
            &fields,
            max_sequence_tokens_idx,
            source_path,
            line_number,
            "max_sequence_tokens",
        )?;
        validate_optional_max_sequence_tokens(
            &format!(
                "serving.traffic.trace_csv {} line {line_number} max_sequence_tokens",
                source_path.display()
            ),
            max_sequence_tokens,
            prompt_tokens,
            decode_tokens,
        )?;
        let prefix_cache_hit_tokens = optional_trace_csv_u32(
            &fields,
            prefix_cache_hit_tokens_idx,
            source_path,
            line_number,
            "prefix_cache_hit_tokens",
        )?;
        let prefix_cache_hit_rate = trace_csv_optional_rate(
            &fields,
            prefix_cache_hit_rate_idx,
            source_path,
            line_number,
            "prefix_cache_hit_rate",
        )?;
        if prefix_cache_hit_tokens.is_some() && prefix_cache_hit_rate.is_some() {
            return Err(ConfigError::new(format!(
                "serving.traffic.trace_csv {} line {line_number} cannot set both prefix_cache_hit_tokens and prefix_cache_hit_rate",
                source_path.display()
            )));
        }
        if let Some(prefix_cache_hit_tokens) = prefix_cache_hit_tokens
            && prefix_cache_hit_tokens > prompt_tokens
        {
            return Err(ConfigError::new(format!(
                "serving.traffic.trace_csv {} line {line_number} prefix_cache_hit_tokens must be less than or equal to prompt_tokens",
                source_path.display()
            )));
        }
        let arrival_s = trace_csv_arrival_s(
            &fields,
            arrival_s_idx,
            arrival_ms_idx,
            source_path,
            line_number,
        )?;
        let slo = trace_csv_request_slo(
            &fields,
            TraceCsvRequestSloColumns {
                ttft_slo_s_idx,
                ttft_slo_ms_idx,
                tpot_slo_s_idx,
                tpot_slo_ms_idx,
                itl_slo_s_idx,
                itl_slo_ms_idx,
                e2el_slo_s_idx,
                e2el_slo_ms_idx,
            },
            source_path,
            line_number,
        )?;
        let deadline_s = trace_csv_deadline_s(
            &fields,
            deadline_s_idx,
            deadline_ms_idx,
            deadline_after_s_idx,
            deadline_after_ms_idx,
            arrival_s,
            source_path,
            line_number,
        )?;
        let cancellation_s = trace_csv_cancellation_s(
            &fields,
            cancellation_s_idx,
            cancellation_ms_idx,
            cancel_after_s_idx,
            cancel_after_ms_idx,
            arrival_s,
            source_path,
            line_number,
        )?;

        parsed.push(ServingTraceRequest {
            request_id: trace_csv_optional_string(&fields, request_id_idx),
            tenant: trace_csv_optional_string(&fields, tenant_idx),
            model_id: trace_csv_optional_string(&fields, model_id_idx),
            cache_key: trace_csv_optional_string(&fields, cache_key_idx),
            arrival_s,
            priority: trace_csv_optional_i32(&fields, priority_idx, source_path, line_number)?
                .unwrap_or(0),
            batch_size,
            prompt_tokens,
            decode_tokens,
            max_sequence_tokens,
            prefix_cache_hit_tokens,
            prefix_cache_hit_rate,
            slo,
            deadline_s,
            cancellation_s,
        });
    }
    if parsed.is_empty() {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} contains no requests",
            source_path.display()
        )));
    }
    Ok(parsed)
}

pub(super) fn trace_csv_is_data_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.starts_with('#')
}

pub(super) fn trace_csv_fields(line: &str) -> Vec<String> {
    line.split(',')
        .map(|field| field.trim().to_string())
        .collect()
}

pub(super) fn trace_csv_headers(fields: &[String]) -> HashMap<String, usize> {
    let mut headers = HashMap::new();
    for (idx, field) in fields.iter().enumerate() {
        let header = normalize(field);
        if !header.is_empty() {
            headers.entry(header).or_insert(idx);
        }
    }
    headers
}

pub(super) fn required_trace_csv_header(
    headers: &HashMap<String, usize>,
    name: &str,
    source_path: &Path,
) -> Result<usize, ConfigError> {
    headers.get(name).copied().ok_or_else(|| {
        ConfigError::new(format!(
            "serving.traffic.trace_csv {} requires a {name} column",
            source_path.display()
        ))
    })
}

pub(super) fn optional_trace_csv_header(
    headers: &HashMap<String, usize>,
    names: &[&str],
) -> Option<usize> {
    names.iter().find_map(|name| headers.get(*name).copied())
}

pub(super) fn trace_csv_cell(fields: &[String], idx: Option<usize>) -> Option<&str> {
    idx.and_then(|idx| fields.get(idx))
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

pub(super) fn trace_csv_optional_string(fields: &[String], idx: Option<usize>) -> Option<String> {
    trace_csv_cell(fields, idx).map(ToString::to_string)
}

pub(super) fn trace_csv_optional_i32(
    fields: &[String],
    idx: Option<usize>,
    source_path: &Path,
    line_number: usize,
) -> Result<Option<i32>, ConfigError> {
    let Some(value) = trace_csv_cell(fields, idx) else {
        return Ok(None);
    };
    value.parse::<i32>().map(Some).map_err(|err| {
        ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} has invalid priority: {err}",
            source_path.display()
        ))
    })
}

pub(super) fn required_positive_trace_csv_u32(
    fields: &[String],
    idx: usize,
    source_path: &Path,
    line_number: usize,
    name: &str,
) -> Result<u32, ConfigError> {
    let value = trace_csv_cell(fields, Some(idx)).ok_or_else(|| {
        ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} requires {name}",
            source_path.display()
        ))
    })?;
    let parsed = value.parse::<u32>().map_err(|err| {
        ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} has invalid {name}: {err}",
            source_path.display()
        ))
    })?;
    if parsed == 0 {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} {name} must be greater than zero",
            source_path.display()
        )));
    }
    Ok(parsed)
}

pub(super) fn optional_positive_trace_csv_u32(
    fields: &[String],
    idx: Option<usize>,
    source_path: &Path,
    line_number: usize,
    name: &str,
) -> Result<Option<u32>, ConfigError> {
    let Some(value) = trace_csv_cell(fields, idx) else {
        return Ok(None);
    };
    let parsed = value.parse::<u32>().map_err(|err| {
        ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} has invalid {name}: {err}",
            source_path.display()
        ))
    })?;
    if parsed == 0 {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} {name} must be greater than zero",
            source_path.display()
        )));
    }
    Ok(Some(parsed))
}

pub(super) fn optional_trace_csv_u32(
    fields: &[String],
    idx: Option<usize>,
    source_path: &Path,
    line_number: usize,
    name: &str,
) -> Result<Option<u32>, ConfigError> {
    let Some(value) = trace_csv_cell(fields, idx) else {
        return Ok(None);
    };
    value.parse::<u32>().map(Some).map_err(|err| {
        ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} has invalid {name}: {err}",
            source_path.display()
        ))
    })
}

pub(super) fn trace_csv_arrival_s(
    fields: &[String],
    arrival_s_idx: Option<usize>,
    arrival_ms_idx: Option<usize>,
    source_path: &Path,
    line_number: usize,
) -> Result<f64, ConfigError> {
    let arrival_s = trace_csv_cell(fields, arrival_s_idx);
    let arrival_ms = trace_csv_cell(fields, arrival_ms_idx);
    if arrival_s.is_some() && arrival_ms.is_some() {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} cannot set both arrival_s and arrival_ms",
            source_path.display()
        )));
    }

    let arrival_s = match (arrival_s, arrival_ms) {
        (Some(value), None) => parse_trace_csv_f64(value, source_path, line_number, "arrival_s")?,
        (None, Some(value)) => {
            parse_trace_csv_f64(value, source_path, line_number, "arrival_ms")? / 1000.0
        }
        (None, None) => {
            return Err(ConfigError::new(format!(
                "serving.traffic.trace_csv {} line {line_number} requires arrival_s or arrival_ms",
                source_path.display()
            )));
        }
        (Some(_), Some(_)) => unreachable!(),
    };
    if !arrival_s.is_finite() || arrival_s < 0.0 {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} arrival must be finite and non-negative",
            source_path.display()
        )));
    }
    Ok(arrival_s)
}

#[derive(Copy, Clone)]
pub(super) struct TraceCsvRequestSloColumns {
    pub(super) ttft_slo_s_idx: Option<usize>,
    pub(super) ttft_slo_ms_idx: Option<usize>,
    pub(super) tpot_slo_s_idx: Option<usize>,
    pub(super) tpot_slo_ms_idx: Option<usize>,
    pub(super) itl_slo_s_idx: Option<usize>,
    pub(super) itl_slo_ms_idx: Option<usize>,
    pub(super) e2el_slo_s_idx: Option<usize>,
    pub(super) e2el_slo_ms_idx: Option<usize>,
}

pub(super) fn trace_csv_request_slo(
    fields: &[String],
    columns: TraceCsvRequestSloColumns,
    source_path: &Path,
    line_number: usize,
) -> Result<ServingRequestSlo, ConfigError> {
    Ok(ServingRequestSlo {
        ttft_s: trace_csv_slo_s(
            fields,
            columns.ttft_slo_s_idx,
            columns.ttft_slo_ms_idx,
            source_path,
            line_number,
            "ttft_slo_s",
            "ttft_slo_ms",
        )?,
        tpot_s: trace_csv_slo_s(
            fields,
            columns.tpot_slo_s_idx,
            columns.tpot_slo_ms_idx,
            source_path,
            line_number,
            "tpot_slo_s",
            "tpot_slo_ms",
        )?,
        itl_s: trace_csv_slo_s(
            fields,
            columns.itl_slo_s_idx,
            columns.itl_slo_ms_idx,
            source_path,
            line_number,
            "itl_slo_s",
            "itl_slo_ms",
        )?,
        e2el_s: trace_csv_slo_s(
            fields,
            columns.e2el_slo_s_idx,
            columns.e2el_slo_ms_idx,
            source_path,
            line_number,
            "e2el_slo_s",
            "e2el_slo_ms",
        )?,
    })
}

pub(super) fn trace_csv_slo_s(
    fields: &[String],
    seconds_idx: Option<usize>,
    millis_idx: Option<usize>,
    source_path: &Path,
    line_number: usize,
    seconds_name: &str,
    millis_name: &str,
) -> Result<Option<f64>, ConfigError> {
    let value = trace_csv_optional_seconds(
        fields,
        seconds_idx,
        millis_idx,
        source_path,
        line_number,
        seconds_name,
        millis_name,
    )?;
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} {seconds_name}/{millis_name} must be positive",
            source_path.display()
        )));
    }
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn trace_csv_deadline_s(
    fields: &[String],
    deadline_s_idx: Option<usize>,
    deadline_ms_idx: Option<usize>,
    deadline_after_s_idx: Option<usize>,
    deadline_after_ms_idx: Option<usize>,
    arrival_s: f64,
    source_path: &Path,
    line_number: usize,
) -> Result<Option<f64>, ConfigError> {
    let deadline_s = trace_csv_optional_seconds(
        fields,
        deadline_s_idx,
        deadline_ms_idx,
        source_path,
        line_number,
        "deadline_s",
        "deadline_ms",
    )?;
    let deadline_after_s = trace_csv_optional_seconds(
        fields,
        deadline_after_s_idx,
        deadline_after_ms_idx,
        source_path,
        line_number,
        "deadline_after_s",
        "deadline_after_ms",
    )?;
    if deadline_s.is_some() && deadline_after_s.is_some() {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} cannot set both deadline and deadline_after",
            source_path.display()
        )));
    }
    if let Some(deadline_after_s) = deadline_after_s
        && deadline_after_s < 0.0
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} deadline_after must be non-negative",
            source_path.display()
        )));
    }
    let deadline_s = deadline_s.or_else(|| deadline_after_s.map(|value| arrival_s + value));
    if let Some(deadline_s) = deadline_s
        && !deadline_s.is_finite()
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} deadline must be finite",
            source_path.display()
        )));
    }
    if let Some(deadline_s) = deadline_s
        && deadline_s < arrival_s
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} deadline must be greater than or equal to arrival",
            source_path.display()
        )));
    }
    Ok(deadline_s)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn trace_csv_cancellation_s(
    fields: &[String],
    cancellation_s_idx: Option<usize>,
    cancellation_ms_idx: Option<usize>,
    cancel_after_s_idx: Option<usize>,
    cancel_after_ms_idx: Option<usize>,
    arrival_s: f64,
    source_path: &Path,
    line_number: usize,
) -> Result<Option<f64>, ConfigError> {
    let cancellation_s = trace_csv_optional_seconds(
        fields,
        cancellation_s_idx,
        cancellation_ms_idx,
        source_path,
        line_number,
        "cancellation_s",
        "cancellation_ms",
    )?;
    let cancel_after_s = trace_csv_optional_seconds(
        fields,
        cancel_after_s_idx,
        cancel_after_ms_idx,
        source_path,
        line_number,
        "cancel_after_s",
        "cancel_after_ms",
    )?;
    if cancellation_s.is_some() && cancel_after_s.is_some() {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} cannot set both cancellation and cancel_after",
            source_path.display()
        )));
    }
    if let Some(cancel_after_s) = cancel_after_s
        && cancel_after_s < 0.0
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} cancel_after must be non-negative",
            source_path.display()
        )));
    }
    let cancellation_s = cancellation_s.or_else(|| cancel_after_s.map(|value| arrival_s + value));
    if let Some(cancellation_s) = cancellation_s
        && !cancellation_s.is_finite()
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} cancellation must be finite",
            source_path.display()
        )));
    }
    if let Some(cancellation_s) = cancellation_s
        && cancellation_s < arrival_s
    {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} cancellation must be greater than or equal to arrival",
            source_path.display()
        )));
    }
    Ok(cancellation_s)
}

pub(super) fn trace_csv_optional_seconds(
    fields: &[String],
    seconds_idx: Option<usize>,
    millis_idx: Option<usize>,
    source_path: &Path,
    line_number: usize,
    seconds_name: &str,
    millis_name: &str,
) -> Result<Option<f64>, ConfigError> {
    let seconds = trace_csv_cell(fields, seconds_idx);
    let millis = trace_csv_cell(fields, millis_idx);
    if seconds.is_some() && millis.is_some() {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} cannot set both {seconds_name} and {millis_name}",
            source_path.display()
        )));
    }
    let value = match (seconds, millis) {
        (Some(value), None) => Some(parse_trace_csv_f64(
            value,
            source_path,
            line_number,
            seconds_name,
        )?),
        (None, Some(value)) => {
            Some(parse_trace_csv_f64(value, source_path, line_number, millis_name)? / 1000.0)
        }
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!(),
    };
    Ok(value)
}

pub(super) fn trace_csv_optional_rate(
    fields: &[String],
    idx: Option<usize>,
    source_path: &Path,
    line_number: usize,
    name: &str,
) -> Result<Option<f64>, ConfigError> {
    let Some(value) = trace_csv_cell(fields, idx) else {
        return Ok(None);
    };
    let value = parse_trace_csv_f64(value, source_path, line_number, name)?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} {name} must be finite and between 0.0 and 1.0",
            source_path.display()
        )));
    }
    Ok(Some(value))
}

pub(super) fn parse_trace_csv_f64(
    value: &str,
    source_path: &Path,
    line_number: usize,
    name: &str,
) -> Result<f64, ConfigError> {
    value.parse::<f64>().map_err(|err| {
        ConfigError::new(format!(
            "serving.traffic.trace_csv {} line {line_number} has invalid {name}: {err}",
            source_path.display()
        ))
    })
}
