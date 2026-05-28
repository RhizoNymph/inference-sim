use super::*;

pub(super) fn collect_metric(
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

pub(super) fn phase_resource_utilization(
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

pub(super) fn operation_phase(name: &str) -> &'static str {
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

pub(super) fn resource_kind(resource: &str) -> &'static str {
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
