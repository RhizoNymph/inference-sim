pub(super) fn ratio_or_infinity(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        f64::INFINITY
    } else {
        f64::from(numerator) / f64::from(denominator)
    }
}

pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }

    values.iter().sum::<f64>() / values.len() as f64
}

pub(super) fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }

    values.sort_by(f64::total_cmp);
    let bounded = quantile.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * bounded).ceil() as usize;
    values[idx]
}

pub(super) fn max_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(f64::INFINITY)
}
