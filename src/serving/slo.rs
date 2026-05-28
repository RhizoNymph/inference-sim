use super::*;

pub(super) fn slo_missed(slo_s: Option<f64>, value_s: f64, completed: bool) -> bool {
    slo_s.is_some_and(|slo_s| !completed || !value_s.is_finite() || value_s > slo_s + 1e-12)
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(super) struct SloMissCounts {
    pub(super) constrained: u32,
    pub(super) missed: u32,
    pub(super) miss_rate: f64,
}

pub(super) fn observation_slo_miss_counts(
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

pub(super) fn slo_rejections(
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
