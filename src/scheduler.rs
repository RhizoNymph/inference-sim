use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledOperation {
    pub id: usize,
    pub name: String,
    pub resources: Vec<String>,
    pub start_s: f64,
    pub finish_s: f64,
    pub explicit_dependencies: Vec<usize>,
    pub resource_dependencies: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceUtilization {
    pub resource: String,
    pub busy_s: f64,
    pub utilization: f64,
    pub operation_count: usize,
    pub first_start_s: f64,
    pub last_finish_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulePreview {
    pub resources: Vec<String>,
    pub start_s: f64,
    pub finish_s: f64,
    pub explicit_dependencies: Vec<usize>,
    pub resource_dependencies: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceOccupancySeries {
    pub resource: String,
    pub buckets: Vec<ResourceOccupancyBucket>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ResourceOccupancyBucket {
    pub bucket_idx: usize,
    pub start_s: f64,
    pub finish_s: f64,
    pub busy_s: f64,
    pub utilization: f64,
    pub operation_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CriticalPath {
    pub total_s: f64,
    pub steps: Vec<CriticalPathStep>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CriticalPathStep {
    pub operation_id: usize,
    pub name: String,
    pub resources: Vec<String>,
    pub start_s: f64,
    pub finish_s: f64,
    pub duration_s: f64,
    pub wait_s: f64,
    pub explicit_dependencies: Vec<usize>,
    pub resource_dependencies: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResourceScheduler {
    operations: Vec<ScheduledOperation>,
    reservations: HashMap<String, Vec<ResourceReservation>>,
}

impl ResourceScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn schedule(
        &mut self,
        name: impl Into<String>,
        duration_s: f64,
        earliest_start_s: f64,
        dependencies: &[usize],
        resources: Vec<String>,
    ) -> usize {
        let preview = self.preview(duration_s, earliest_start_s, dependencies, resources);
        let id = self.operations.len();

        for resource in &preview.resources {
            let reservations = self.reservations.entry(resource.clone()).or_default();
            reservations.push(ResourceReservation {
                start_s: preview.start_s,
                finish_s: preview.finish_s,
                operation_id: id,
            });
            reservations.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
        }

        self.operations.push(ScheduledOperation {
            id,
            name: name.into(),
            resources: preview.resources,
            start_s: preview.start_s,
            finish_s: preview.finish_s,
            explicit_dependencies: preview.explicit_dependencies,
            resource_dependencies: preview.resource_dependencies,
        });

        id
    }

    pub fn preview(
        &self,
        duration_s: f64,
        earliest_start_s: f64,
        dependencies: &[usize],
        resources: Vec<String>,
    ) -> SchedulePreview {
        let explicit_dependencies = normalized_dependency_ids(dependencies, self.operations.len());
        let dependency_ready_s = explicit_dependencies
            .iter()
            .filter_map(|id| self.operations.get(*id))
            .map(|operation| operation.finish_s)
            .fold(earliest_start_s.max(0.0), f64::max);
        let resources = normalized_resources(resources);
        let duration_s = duration_s.max(0.0);
        let start_s = self.find_earliest_start(dependency_ready_s, duration_s, &resources);
        let finish_s = start_s + duration_s;
        let resource_dependencies =
            self.resource_dependencies(dependency_ready_s, start_s, &resources);

        SchedulePreview {
            resources,
            start_s,
            finish_s,
            explicit_dependencies,
            resource_dependencies,
        }
    }

    pub fn operation(&self, id: usize) -> Option<&ScheduledOperation> {
        self.operations.get(id)
    }

    pub fn makespan_s(&self) -> f64 {
        self.operations
            .iter()
            .map(|operation| operation.finish_s)
            .fold(0.0, f64::max)
    }

    pub fn operations(&self) -> &[ScheduledOperation] {
        &self.operations
    }

    fn find_earliest_start(
        &self,
        earliest_start_s: f64,
        duration_s: f64,
        resources: &[String],
    ) -> f64 {
        if resources.is_empty() || duration_s == 0.0 {
            return earliest_start_s;
        }

        let mut start_s = earliest_start_s;
        loop {
            let finish_s = start_s + duration_s;
            let mut next_start = None;

            for resource in resources {
                for reservation in self.reservations.get(resource).into_iter().flatten() {
                    if overlaps(start_s, finish_s, reservation.start_s, reservation.finish_s) {
                        next_start = Some(next_start.unwrap_or(start_s).max(reservation.finish_s));
                    }
                }
            }

            if let Some(next_start_s) = next_start {
                if next_start_s <= start_s {
                    return start_s;
                }
                start_s = next_start_s;
            } else {
                return start_s;
            }
        }
    }

    fn resource_dependencies(
        &self,
        dependency_ready_s: f64,
        start_s: f64,
        resources: &[String],
    ) -> Vec<usize> {
        if start_s <= dependency_ready_s + f64::EPSILON {
            return Vec::new();
        }

        let mut dependencies = resources
            .iter()
            .filter_map(|resource| {
                self.reservations
                    .get(resource)
                    .into_iter()
                    .flatten()
                    .filter(|reservation| {
                        reservation.finish_s > dependency_ready_s + f64::EPSILON
                            && reservation.finish_s <= start_s + f64::EPSILON
                    })
                    .max_by(|left, right| left.finish_s.total_cmp(&right.finish_s))
                    .map(|reservation| reservation.operation_id)
            })
            .collect::<Vec<_>>();
        dependencies.sort_unstable();
        dependencies.dedup();
        dependencies
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct ResourceReservation {
    start_s: f64,
    finish_s: f64,
    operation_id: usize,
}

fn normalized_resources(mut resources: Vec<String>) -> Vec<String> {
    resources.sort();
    resources.dedup();
    resources
}

fn normalized_dependency_ids(dependencies: &[usize], operation_count: usize) -> Vec<usize> {
    let mut dependencies = dependencies
        .iter()
        .copied()
        .filter(|dependency| *dependency < operation_count)
        .collect::<Vec<_>>();
    dependencies.sort_unstable();
    dependencies.dedup();
    dependencies
}

fn overlaps(a_start: f64, a_finish: f64, b_start: f64, b_finish: f64) -> bool {
    a_start < b_finish && b_start < a_finish
}

pub fn resource_utilization(
    operations: &[ScheduledOperation],
    window_s: f64,
) -> Vec<ResourceUtilization> {
    let mut resources: HashMap<String, ResourceUtilizationAccumulator> = HashMap::new();

    for operation in operations {
        let duration_s = (operation.finish_s - operation.start_s).max(0.0);
        for resource in &operation.resources {
            let entry = resources
                .entry(resource.clone())
                .or_insert_with(ResourceUtilizationAccumulator::new);
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
    let mut utilization = resources
        .into_iter()
        .map(|(resource, accumulator)| ResourceUtilization {
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
        })
        .collect::<Vec<_>>();

    utilization.sort_by(|left, right| {
        right
            .utilization
            .total_cmp(&left.utilization)
            .then_with(|| right.busy_s.total_cmp(&left.busy_s))
            .then_with(|| left.resource.cmp(&right.resource))
    });
    utilization
}

pub fn resource_occupancy_buckets(
    operations: &[ScheduledOperation],
    window_s: f64,
    bucket_count: usize,
    resources: &[String],
) -> Vec<ResourceOccupancySeries> {
    let window_s = if window_s.is_finite() && window_s > 0.0 {
        window_s
    } else {
        return Vec::new();
    };
    if bucket_count == 0 || resources.is_empty() {
        return Vec::new();
    }

    let bucket_width_s = window_s / bucket_count as f64;
    let mut series = resources
        .iter()
        .map(|resource| ResourceOccupancySeries {
            resource: resource.clone(),
            buckets: (0..bucket_count)
                .map(|bucket_idx| {
                    let start_s = bucket_idx as f64 * bucket_width_s;
                    let finish_s = if bucket_idx + 1 == bucket_count {
                        window_s
                    } else {
                        (bucket_idx + 1) as f64 * bucket_width_s
                    };
                    ResourceOccupancyBucket {
                        bucket_idx,
                        start_s,
                        finish_s,
                        busy_s: 0.0,
                        utilization: 0.0,
                        operation_count: 0,
                    }
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let resource_to_series = resources
        .iter()
        .enumerate()
        .map(|(idx, resource)| (resource.as_str(), idx))
        .collect::<HashMap<_, _>>();

    for operation in operations {
        let operation_start_s = operation.start_s.max(0.0).min(window_s);
        let operation_finish_s = operation.finish_s.max(0.0).min(window_s);
        if operation_finish_s <= operation_start_s {
            continue;
        }

        let first_bucket = bucket_idx(operation_start_s, bucket_width_s, bucket_count);
        let last_bucket = bucket_idx(
            (operation_finish_s - f64::EPSILON).max(0.0),
            bucket_width_s,
            bucket_count,
        );

        for resource in &operation.resources {
            let Some(&series_idx) = resource_to_series.get(resource.as_str()) else {
                continue;
            };
            for bucket_idx in first_bucket..=last_bucket {
                let bucket = &mut series[series_idx].buckets[bucket_idx];
                let overlap_s = (operation_finish_s.min(bucket.finish_s)
                    - operation_start_s.max(bucket.start_s))
                .max(0.0);
                if overlap_s > 0.0 {
                    bucket.busy_s += overlap_s;
                    bucket.operation_count += 1;
                }
            }
        }
    }

    for series in &mut series {
        for bucket in &mut series.buckets {
            let width_s = (bucket.finish_s - bucket.start_s).max(0.0);
            bucket.utilization = if width_s > 0.0 {
                (bucket.busy_s / width_s).min(1.0)
            } else {
                0.0
            };
        }
    }

    series
}

pub fn critical_path(operations: &[ScheduledOperation]) -> CriticalPath {
    if operations.is_empty() {
        return CriticalPath {
            total_s: 0.0,
            steps: Vec::new(),
        };
    }

    let mut id_to_idx = HashMap::new();
    for (idx, operation) in operations.iter().enumerate() {
        id_to_idx.insert(operation.id, idx);
    }

    let mut elapsed_by_idx = vec![0.0_f64; operations.len()];
    let mut parent_by_idx = vec![None; operations.len()];

    for (idx, operation) in operations.iter().enumerate() {
        let dependencies = all_dependencies(operation);
        let best_parent = dependencies
            .iter()
            .filter_map(|dependency_id| id_to_idx.get(dependency_id).copied())
            .filter(|dependency_idx| *dependency_idx < idx)
            .max_by(|left, right| elapsed_by_idx[*left].total_cmp(&elapsed_by_idx[*right]));

        if let Some(parent_idx) = best_parent {
            let parent = &operations[parent_idx];
            elapsed_by_idx[idx] =
                elapsed_by_idx[parent_idx] + (operation.finish_s - parent.finish_s).max(0.0);
            parent_by_idx[idx] = Some(parent_idx);
        } else {
            elapsed_by_idx[idx] = operation.finish_s.max(0.0);
        }
    }

    let end_idx = (0..operations.len())
        .max_by(|left, right| elapsed_by_idx[*left].total_cmp(&elapsed_by_idx[*right]))
        .unwrap_or(0);
    let total_s = elapsed_by_idx[end_idx];
    let mut path_indices = Vec::new();
    let mut current_idx = Some(end_idx);
    while let Some(idx) = current_idx {
        path_indices.push(idx);
        current_idx = parent_by_idx[idx];
    }
    path_indices.reverse();

    let steps = path_indices
        .iter()
        .enumerate()
        .map(|(path_idx, operation_idx)| {
            let operation = &operations[*operation_idx];
            let wait_s = if path_idx == 0 {
                operation.start_s.max(0.0)
            } else {
                let previous = &operations[path_indices[path_idx - 1]];
                (operation.start_s - previous.finish_s).max(0.0)
            };
            CriticalPathStep {
                operation_id: operation.id,
                name: operation.name.clone(),
                resources: operation.resources.clone(),
                start_s: operation.start_s,
                finish_s: operation.finish_s,
                duration_s: (operation.finish_s - operation.start_s).max(0.0),
                wait_s,
                explicit_dependencies: operation.explicit_dependencies.clone(),
                resource_dependencies: operation.resource_dependencies.clone(),
            }
        })
        .collect();

    CriticalPath { total_s, steps }
}

fn all_dependencies(operation: &ScheduledOperation) -> Vec<usize> {
    let mut dependencies = operation.explicit_dependencies.clone();
    dependencies.extend_from_slice(&operation.resource_dependencies);
    dependencies.sort_unstable();
    dependencies.dedup();
    dependencies
}

fn bucket_idx(time_s: f64, bucket_width_s: f64, bucket_count: usize) -> usize {
    if bucket_count == 0 || bucket_width_s <= 0.0 {
        return 0;
    }
    ((time_s / bucket_width_s).floor() as usize).min(bucket_count - 1)
}

#[derive(Clone, Debug)]
struct ResourceUtilizationAccumulator {
    busy_s: f64,
    operation_count: usize,
    first_start_s: f64,
    last_finish_s: f64,
}

impl ResourceUtilizationAccumulator {
    fn new() -> Self {
        Self {
            busy_s: 0.0,
            operation_count: 0,
            first_start_s: f64::INFINITY,
            last_finish_s: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_independent_resources_in_parallel() {
        let mut scheduler = ResourceScheduler::new();
        let a = scheduler.schedule("a", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        let b = scheduler.schedule("b", 1.0, 0.0, &[], vec!["gpu1".to_string()]);

        assert_eq!(scheduler.operation(a).unwrap().start_s, 0.0);
        assert_eq!(scheduler.operation(b).unwrap().start_s, 0.0);
        assert_eq!(scheduler.makespan_s(), 1.0);
    }

    #[test]
    fn serializes_shared_resources() {
        let mut scheduler = ResourceScheduler::new();
        let a = scheduler.schedule("a", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        let b = scheduler.schedule("b", 1.0, 0.0, &[], vec!["gpu0".to_string()]);

        assert_eq!(scheduler.operation(a).unwrap().start_s, 0.0);
        assert_eq!(scheduler.operation(b).unwrap().start_s, 1.0);
        assert_eq!(
            scheduler.operation(b).unwrap().resource_dependencies,
            vec![a]
        );
        assert_eq!(scheduler.makespan_s(), 2.0);
    }

    #[test]
    fn preview_does_not_reserve_resources() {
        let mut scheduler = ResourceScheduler::new();
        let a = scheduler.schedule("a", 1.0, 0.0, &[], vec!["gpu0".to_string()]);

        let preview = scheduler.preview(1.0, 0.0, &[], vec!["gpu0".to_string()]);
        assert_eq!(preview.start_s, 1.0);
        assert_eq!(preview.finish_s, 2.0);
        assert_eq!(preview.resource_dependencies, vec![a]);
        assert_eq!(scheduler.operations().len(), 1);

        let b = scheduler.schedule("b", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        assert_eq!(scheduler.operation(b).unwrap().start_s, preview.start_s);
        assert_eq!(scheduler.operation(b).unwrap().finish_s, preview.finish_s);
    }

    #[test]
    fn honors_dependencies() {
        let mut scheduler = ResourceScheduler::new();
        let a = scheduler.schedule("a", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        let b = scheduler.schedule("b", 1.0, 0.0, &[a], vec!["gpu1".to_string()]);

        assert_eq!(scheduler.operation(b).unwrap().start_s, 1.0);
        assert_eq!(
            scheduler.operation(b).unwrap().explicit_dependencies,
            vec![a]
        );
    }

    #[test]
    fn summarizes_resource_utilization() {
        let mut scheduler = ResourceScheduler::new();
        scheduler.schedule("a", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        scheduler.schedule("b", 1.0, 0.0, &[], vec!["gpu1".to_string()]);
        scheduler.schedule("c", 1.0, 0.0, &[], vec!["gpu0".to_string()]);

        let utilization = resource_utilization(scheduler.operations(), scheduler.makespan_s());

        assert_eq!(utilization[0].resource, "gpu0");
        assert_eq!(utilization[0].busy_s, 2.0);
        assert_eq!(utilization[0].utilization, 1.0);
        assert_eq!(utilization[0].operation_count, 2);
        assert_eq!(utilization[1].resource, "gpu1");
        assert_eq!(utilization[1].utilization, 0.5);
    }

    #[test]
    fn buckets_resource_occupancy_over_time() {
        let mut scheduler = ResourceScheduler::new();
        scheduler.schedule("a", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        scheduler.schedule("b", 1.0, 0.0, &[], vec!["gpu1".to_string()]);
        scheduler.schedule("c", 1.0, 0.0, &[], vec!["gpu0".to_string()]);

        let occupancy = resource_occupancy_buckets(
            scheduler.operations(),
            scheduler.makespan_s(),
            2,
            &["gpu0".to_string(), "gpu1".to_string()],
        );

        assert_eq!(occupancy.len(), 2);
        assert_eq!(occupancy[0].resource, "gpu0");
        assert_eq!(occupancy[0].buckets[0].busy_s, 1.0);
        assert_eq!(occupancy[0].buckets[0].utilization, 1.0);
        assert_eq!(occupancy[0].buckets[1].busy_s, 1.0);
        assert_eq!(occupancy[1].resource, "gpu1");
        assert_eq!(occupancy[1].buckets[0].busy_s, 1.0);
        assert_eq!(occupancy[1].buckets[1].busy_s, 0.0);
    }

    #[test]
    fn computes_critical_path_with_resource_dependencies() {
        let mut scheduler = ResourceScheduler::new();
        let a = scheduler.schedule("a", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        let b = scheduler.schedule("b", 1.0, 0.0, &[], vec!["gpu0".to_string()]);
        let c = scheduler.schedule("c", 1.0, 0.0, &[b], vec!["gpu1".to_string()]);

        let path = critical_path(scheduler.operations());

        assert_eq!(path.total_s, 3.0);
        assert_eq!(
            path.steps
                .iter()
                .map(|step| step.operation_id)
                .collect::<Vec<_>>(),
            vec![a, b, c]
        );
        assert_eq!(path.steps[1].resource_dependencies, vec![a]);
        assert_eq!(path.steps[2].explicit_dependencies, vec![b]);
    }
}
