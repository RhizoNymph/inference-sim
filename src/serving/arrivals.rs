use super::*;

fn cyclic_or_default(values: &[u32], idx: u32, default: u32) -> u32 {
    values
        .get(idx as usize % values.len().max(1))
        .copied()
        .unwrap_or(default)
        .max(1)
}

pub(super) fn sample_or_cyclic(
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

pub(super) fn poisson_arrival_times(request_count: u32, rate_per_s: f64, seed: u64) -> Vec<f64> {
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

pub(super) fn bursty_arrival_times(
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

pub(super) fn diurnal_arrival_times(
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

pub(super) fn diurnal_rate_at_s(
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

pub(super) fn self_similar_arrival_times(
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

pub(super) struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub(super) fn next_open_unit_f64(&mut self) -> f64 {
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

pub(super) fn request_token_scale(
    batch_size: u32,
    tokens: u32,
    base_batch_size: u32,
    base_tokens: u32,
) -> f64 {
    let numerator = f64::from(batch_size.max(1)) * f64::from(tokens.max(1));
    let denominator = f64::from(base_batch_size.max(1)) * f64::from(base_tokens.max(1));
    numerator / denominator
}
