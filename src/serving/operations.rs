use super::*;

pub(super) fn schedule_trace(
    scheduler: &mut ResourceScheduler,
    prefix: &str,
    operations: &[SimOperation],
    earliest_start_s: f64,
    external_dependencies: &[usize],
) -> Vec<usize> {
    let mut scheduled_ids = Vec::with_capacity(operations.len());
    for operation in operations {
        let mut dependencies: Vec<_> = operation
            .dependencies
            .iter()
            .filter_map(|idx| scheduled_ids.get(*idx).copied())
            .collect();
        if operation.dependencies.is_empty() {
            dependencies.extend_from_slice(external_dependencies);
        }

        let scheduled_id = scheduler.schedule(
            format!("{prefix} {}", operation.name),
            operation.duration_s,
            earliest_start_s,
            &dependencies,
            operation.resources.clone(),
        );
        scheduled_ids.push(scheduled_id);
    }

    scheduled_ids
}

pub(super) fn preview_trace_span(
    scheduler: &ResourceScheduler,
    operations: &[SimOperation],
    earliest_start_s: f64,
    external_dependencies: &[usize],
) -> (f64, f64) {
    let mut preview_scheduler = scheduler.clone();
    let scheduled_ids = schedule_trace(
        &mut preview_scheduler,
        "preview",
        operations,
        earliest_start_s,
        external_dependencies,
    );
    operation_span(&preview_scheduler, &scheduled_ids)
}

pub(super) fn terminal_ids(operations: &[SimOperation], scheduled_ids: &[usize]) -> Vec<usize> {
    let mut depended_on = std::collections::HashSet::new();
    for operation in operations {
        for dependency in &operation.dependencies {
            depended_on.insert(*dependency);
        }
    }

    scheduled_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, scheduled_id)| {
            if depended_on.contains(&idx) {
                None
            } else {
                Some(*scheduled_id)
            }
        })
        .collect()
}

pub(super) fn operation_span(scheduler: &ResourceScheduler, ids: &[usize]) -> (f64, f64) {
    let mut start_s = f64::INFINITY;
    let mut finish_s = 0.0_f64;
    for id in ids {
        if let Some(operation) = scheduler.operation(*id) {
            start_s = start_s.min(operation.start_s);
            finish_s = finish_s.max(operation.finish_s);
        }
    }

    if start_s.is_finite() {
        (start_s, finish_s)
    } else {
        (0.0, 0.0)
    }
}

pub(super) fn operation_finish_s(scheduler: &ResourceScheduler, ids: &[usize]) -> f64 {
    ids.iter()
        .filter_map(|id| scheduler.operation(*id))
        .map(|operation| operation.finish_s)
        .fold(0.0, f64::max)
}

pub(super) fn scale_operations(operations: &mut [SimOperation], scale: f64) {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    for operation in operations {
        operation.duration_s *= scale;
    }
}

pub(super) fn single_placement_node(score: &ScoredParallelismConfig) -> Option<NodeId> {
    let nodes: BTreeSet<_> = placement_nodes(score).into_iter().collect();
    if nodes.len() == 1 {
        nodes.first().copied()
    } else {
        None
    }
}

pub(super) fn placement_nodes(score: &ScoredParallelismConfig) -> Vec<NodeId> {
    let mut nodes: Vec<_> = score
        .placement
        .rank_to_gpu
        .iter()
        .map(|addr| addr.node_id)
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

pub(super) fn routed_node_set(
    resource_base_node: Option<NodeId>,
    placement_nodes: &[NodeId],
    routed_node: NodeId,
) -> Vec<NodeId> {
    if resource_base_node.is_some() || placement_nodes.is_empty() {
        vec![routed_node]
    } else {
        placement_nodes.to_vec()
    }
}

pub(super) fn routed_gpu_set(
    resource_base_node: Option<NodeId>,
    placement_gpus: &[GpuAddr],
    routed_node: NodeId,
) -> Vec<GpuAddr> {
    let mut gpus = if resource_base_node.is_some() || placement_gpus.is_empty() {
        placement_gpus
            .iter()
            .map(|addr| GpuAddr {
                node_id: routed_node,
                local_gpu_id: addr.local_gpu_id,
            })
            .collect::<Vec<_>>()
    } else {
        placement_gpus.to_vec()
    };
    if gpus.is_empty() {
        gpus.push(GpuAddr {
            node_id: routed_node,
            local_gpu_id: 0,
        });
    }
    gpus.sort_unstable();
    gpus.dedup();
    gpus
}

pub(super) fn remap_operations_to_node(
    operations: &mut [SimOperation],
    base_node: Option<NodeId>,
    target_node: NodeId,
) {
    let Some(base_node) = base_node else {
        return;
    };
    if base_node == target_node {
        return;
    }

    for operation in operations {
        for resource in &mut operation.resources {
            *resource = remap_resource_to_node(resource, base_node, target_node);
        }
    }
}

fn remap_resource_to_node(resource: &str, base_node: NodeId, target_node: NodeId) -> String {
    let gpu_compute = format!("gpu compute node {base_node}");
    if resource == gpu_compute {
        return format!("gpu compute node {target_node}");
    }

    let gpu_hbm = format!("gpu HBM node {base_node}");
    if resource == gpu_hbm {
        return format!("gpu HBM node {target_node}");
    }

    let intra_node = format!("node {base_node} intra-node fabric");
    if resource == intra_node {
        return format!("node {target_node} intra-node fabric");
    }

    resource.to_string()
}
