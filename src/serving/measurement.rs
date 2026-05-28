use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MeasurementWindowSelection {
    pub(super) start_s: f64,
    pub(super) end_s: f64,
    pub(super) configured_start_bound: bool,
    pub(super) configured_end_bound: bool,
    pub(super) steady_state_requested: bool,
    pub(super) steady_state_applied: bool,
    pub(super) steady_state_candidate_start_s: Option<f64>,
    pub(super) steady_state_candidate_end_s: Option<f64>,
    pub(super) steady_state_min_requests: Option<u32>,
    pub(super) steady_state_max_cv: Option<f64>,
    pub(super) steady_state_sample_count: u32,
    pub(super) steady_state_matching_window_count: u32,
    pub(super) steady_state_candidate_request_count: Option<u32>,
    pub(super) steady_state_candidate_e2el_mean_s: Option<f64>,
    pub(super) steady_state_candidate_e2el_stddev_s: Option<f64>,
    pub(super) steady_state_candidate_e2el_cv: Option<f64>,
    pub(super) steady_state_candidate_e2el_std_error_s: Option<f64>,
    pub(super) steady_state_candidate_metric_count: u32,
    pub(super) steady_state_candidate_worst_metric: Option<String>,
    pub(super) steady_state_candidate_worst_cv: Option<f64>,
    pub(super) steady_state_candidate_output_tokens: Option<u64>,
    pub(super) steady_state_candidate_throughput_tokens_per_s: Option<f64>,
    pub(super) steady_state_candidate_metrics: Vec<ServingSteadyStateMetricObservation>,
    pub(super) steady_state_candidate_utilization_count: u32,
    pub(super) steady_state_candidate_worst_utilization_resource: Option<String>,
    pub(super) steady_state_candidate_worst_utilization_cv: Option<f64>,
    pub(super) steady_state_candidate_utilization: Vec<ServingSteadyStateUtilizationObservation>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct MeasurementWindowRequestCounts {
    pub(super) request_count: u32,
    pub(super) completed_request_count: u32,
    pub(super) failed_request_count: u32,
    pub(super) rejected_request_count: u32,
    pub(super) timed_out_request_count: u32,
    pub(super) cancelled_request_count: u32,
    pub(super) deadline_constrained_request_count: u32,
    pub(super) deadline_missed_request_count: u32,
}

impl MeasurementWindowSelection {
    pub(super) fn rejected() -> Self {
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

    pub(super) fn into_observation(
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
pub(super) struct SteadyStateMeasurementSearch {
    pub(super) selected: Option<SteadyStateMeasurementWindow>,
    pub(super) sample_count: u32,
    pub(super) matching_window_count: u32,
    pub(super) min_requests: u32,
    pub(super) max_cv: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SteadyStateMeasurementWindow {
    pub(super) start_s: f64,
    pub(super) end_s: f64,
    pub(super) request_count: u32,
    pub(super) mean_e2el_s: f64,
    pub(super) stddev_e2el_s: f64,
    pub(super) cv: f64,
    pub(super) std_error_s: f64,
    pub(super) metric_count: u32,
    pub(super) worst_metric: Option<String>,
    pub(super) worst_cv: Option<f64>,
    pub(super) output_tokens: u64,
    pub(super) throughput_tokens_per_s: f64,
    pub(super) metrics: Vec<ServingSteadyStateMetricObservation>,
}

pub(super) fn effective_measurement_window(
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

pub(super) fn measurement_metric_source_counts(
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

pub(super) fn measurement_window_request_counts(
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
pub(super) fn steady_state_measurement_window_from_samples(
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

pub(super) fn observation_output_tokens(observation: &ServingRequestObservation) -> u64 {
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
