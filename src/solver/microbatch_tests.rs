//! Pipeline overlap across microbatches of one static batch (vLLM's batch
//! queue with pipeline parallelism).

use std::num::NonZeroU32;

use super::{
    microbatch::{MicrobatchPlan, PIPELINE_MICROBATCH_OVERLAP},
    *,
};
use crate::{
    calibration::PipelineMicrobatching,
    config::parse_cluster,
    workload::{DType, ParameterCountSource},
};

fn two_rtx3090_nodes() -> Cluster {
    parse_cluster(
        r#"
        schema_version = 1
        [cluster]
        preset = "custom"
        [interconnect]
        kind = "ethernet"
        variant = "10g"
        oversubscription = 1.0
        [[nodes]]
        id = 0
        gpu = "a100_40gb"
        gpu_count = 1
        hbm_gb = 24.0
        hbm_bandwidth_gb_s = 936.0
        peak_f16_tflops = 88.0
        intra = "pcie_gen4"
        nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
        [[nodes]]
        id = 1
        gpu = "a100_40gb"
        gpu_count = 1
        hbm_gb = 24.0
        hbm_bandwidth_gb_s = 936.0
        peak_f16_tflops = 88.0
        intra = "pcie_gen4"
        nics = { count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }
        "#,
    )
    .expect("two-node cluster parses")
}

fn qwen7b() -> ModelSpec {
    ModelSpec {
        layers: 28,
        hidden_size: 3584,
        attention_heads: 28,
        kv_heads: 4,
        vocab_size: 152_064,
        parameters: Bytes::from_gigabytes(15.23),
        parameter_count: Some(7.615e9),
        parameter_count_source: ParameterCountSource::ShapeWithFfnWidth,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn request(batch_size: u32, phase: InferencePhase) -> InferenceRequest {
    let decode_tokens = match phase {
        InferencePhase::Prefill => 1,
        InferencePhase::Decode | InferencePhase::EndToEnd => 128,
    };
    InferenceRequest {
        batch_size,
        prompt_tokens: 512,
        decode_tokens,
        max_sequence_tokens: 512 + decode_tokens,
        phase,
    }
}

fn pipeline(stages: u32) -> ParallelismConfig {
    ParallelismConfig {
        tensor_ranks: 1,
        pipeline_ranks: stages,
        expert_ranks: 1,
        data_ranks: 1,
    }
}

fn options(policy: PipelineMicrobatching) -> SolverOptions<'static> {
    SolverOptions {
        calibration: SimulationCalibration {
            compute_efficiency: 0.85,
            decode_memory_bandwidth_scale: 0.84,
            pipeline_microbatching: policy,
            ..SimulationCalibration::default()
        },
        ..SolverOptions::default()
    }
}

fn score(
    batch_size: u32,
    phase: InferencePhase,
    stages: u32,
    policy: PipelineMicrobatching,
) -> ScoredParallelismConfig {
    let score = Solver::score_config_with_options(
        &two_rtx3090_nodes(),
        &qwen7b(),
        &request(batch_size, phase),
        pipeline(stages),
        options(policy),
    );
    assert!(
        score.feasible,
        "batch {batch_size} {phase:?} pp={stages} infeasible: {:?}",
        score.rejected_reason
    );
    score
}

fn operation_index(score: &ScoredParallelismConfig, name: &str) -> usize {
    score
        .operations
        .iter()
        .position(|operation| operation.name == name)
        .unwrap_or_else(|| panic!("missing operation {name}"))
}

fn scheduled_finish(score: &ScoredParallelismConfig, name: &str) -> f64 {
    score.scheduled_operations[operation_index(score, name)].finish_s
}

fn scheduled_start(score: &ScoredParallelismConfig, name: &str) -> f64 {
    score.scheduled_operations[operation_index(score, name)].start_s
}

fn nonzero(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("nonzero")
}

#[test]
fn whole_batch_policy_never_splits() {
    for (batch, stages) in [(1, 1), (8, 1), (1, 2), (8, 2), (32, 4)] {
        assert_eq!(
            MicrobatchPlan::for_static_batch(PipelineMicrobatching::WholeBatch, batch, stages),
            MicrobatchPlan::WholeBatch,
            "batch {batch} pp={stages}"
        );
    }
}

#[test]
fn vllm_batch_queue_splits_first_request_from_the_rest_only_with_pipeline_stages() {
    let policy = PipelineMicrobatching::VllmBatchQueue;
    assert_eq!(
        MicrobatchPlan::for_static_batch(policy, 8, 1),
        MicrobatchPlan::WholeBatch
    );
    assert_eq!(
        MicrobatchPlan::for_static_batch(policy, 1, 2),
        MicrobatchPlan::WholeBatch
    );
    assert_eq!(
        MicrobatchPlan::for_static_batch(policy, 0, 2),
        MicrobatchPlan::WholeBatch
    );
    assert_eq!(
        MicrobatchPlan::for_static_batch(policy, 2, 2),
        MicrobatchPlan::FirstThenRest {
            first: nonzero(1),
            rest: nonzero(1)
        }
    );
    for stages in [2, 4] {
        assert_eq!(
            MicrobatchPlan::for_static_batch(policy, 8, stages),
            MicrobatchPlan::FirstThenRest {
                first: nonzero(1),
                rest: nonzero(7)
            },
            "pp={stages}"
        );
    }
}

#[test]
fn split_plan_sizes_sum_to_the_batch() {
    for batch in 2..=64 {
        let plan =
            MicrobatchPlan::for_static_batch(PipelineMicrobatching::VllmBatchQueue, batch, 2);
        let sizes = plan.microbatch_sizes(batch);
        assert_eq!(sizes.iter().map(|size| size.get()).sum::<u32>(), batch);
        assert_eq!(sizes.len(), 2);
    }
    assert_eq!(
        MicrobatchPlan::WholeBatch.microbatch_sizes(8),
        vec![nonzero(8)]
    );
}

#[test]
fn default_calibration_keeps_whole_batch_pipelines() {
    assert_eq!(
        SimulationCalibration::default().pipeline_microbatching,
        PipelineMicrobatching::WholeBatch
    );
}

#[test]
fn single_request_and_single_stage_are_unchanged_by_the_policy() {
    for phase in [
        InferencePhase::Prefill,
        InferencePhase::Decode,
        InferencePhase::EndToEnd,
    ] {
        for (batch, stages) in [(1, 2), (8, 1)] {
            let whole = score(batch, phase, stages, PipelineMicrobatching::WholeBatch);
            let queued = score(batch, phase, stages, PipelineMicrobatching::VllmBatchQueue);
            assert_eq!(
                whole.operations, queued.operations,
                "{phase:?} batch {batch} pp={stages}"
            );
            assert_eq!(whole.estimated_latency_s, queued.estimated_latency_s);
        }
    }
}

#[test]
fn decode_phase_is_never_microbatched() {
    let whole = score(
        8,
        InferencePhase::Decode,
        2,
        PipelineMicrobatching::WholeBatch,
    );
    let queued = score(
        8,
        InferencePhase::Decode,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    assert_eq!(whole.operations, queued.operations);
    assert_eq!(whole.estimated_latency_s, queued.estimated_latency_s);
}

#[test]
fn queued_prefill_overlaps_stages_and_beats_the_whole_batch() {
    let whole = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::WholeBatch,
    );
    let queued = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    let first_alone = score(
        1,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::WholeBatch,
    );
    let rest_alone = score(
        7,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::WholeBatch,
    );

    assert!(
        queued.estimated_latency_s < whole.estimated_latency_s,
        "queued {} >= whole {}",
        queued.estimated_latency_s,
        whole.estimated_latency_s
    );
    // Overlap hides at most the first microbatch, never the rest.
    assert!(queued.estimated_latency_s > rest_alone.estimated_latency_s);
    assert!(
        queued.estimated_latency_s
            < first_alone.estimated_latency_s + rest_alone.estimated_latency_s
    );

    let sends = queued
        .operations
        .iter()
        .filter(|operation| operation.name.ends_with("pipeline sendrecv edge 0"))
        .count();
    assert_eq!(sends, 2);
}

#[test]
fn second_microbatch_starts_on_stage_zero_while_the_first_is_on_stage_one() {
    let queued = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    // 28 layers over 2 stages: layers 0-13 on stage 0, 14-27 on stage 1.
    let second_starts = scheduled_start(&queued, "microbatch 1 layer 0 compute");
    let first_stage_one_ends = scheduled_finish(&queued, "microbatch 0 layer 27 compute");
    assert!(second_starts < first_stage_one_ends);
}

#[test]
fn second_microbatch_waits_for_the_first_microbatch_send_out_of_each_stage() {
    // vLLM's worker waits on the previous step's pipeline send before it
    // starts the next forward pass on that stage.
    let queued = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    let first_send = operation_index(&queued, "microbatch 0 pipeline sendrecv edge 0");
    let second_first_layer = operation_index(&queued, "microbatch 1 layer 0 compute");
    assert!(
        queued.operations[second_first_layer]
            .dependencies
            .contains(&first_send)
    );
    assert!(
        scheduled_start(&queued, "microbatch 1 layer 0 compute")
            >= scheduled_finish(&queued, "microbatch 0 pipeline sendrecv edge 0")
    );
}

#[test]
fn stages_never_run_two_microbatches_at_once() {
    let queued = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    let mut by_resource: BTreeMap<&str, Vec<(f64, f64)>> = BTreeMap::new();
    for operation in &queued.scheduled_operations {
        for resource in &operation.resources {
            if resource.starts_with("gpu compute") {
                by_resource
                    .entry(resource.as_str())
                    .or_default()
                    .push((operation.start_s, operation.finish_s));
            }
        }
    }
    for (resource, mut intervals) in by_resource {
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        for pair in intervals.windows(2) {
            assert!(
                pair[1].0 >= pair[0].1 - 1e-12,
                "{resource} overlaps: {pair:?}"
            );
        }
    }
}

#[test]
fn end_to_end_runs_whole_batch_decode_after_both_prefill_microbatches() {
    let queued_prefill = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    let queued_e2e = score(
        8,
        InferencePhase::EndToEnd,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    let whole_decode = score(
        8,
        InferencePhase::Decode,
        2,
        PipelineMicrobatching::WholeBatch,
    );
    let whole_e2e = score(
        8,
        InferencePhase::EndToEnd,
        2,
        PipelineMicrobatching::WholeBatch,
    );

    let decode_starts = scheduled_start(&queued_e2e, "decode layer 0 compute");
    assert!(decode_starts >= scheduled_finish(&queued_e2e, "microbatch 0 layer 27 compute"));
    assert!(decode_starts >= scheduled_finish(&queued_e2e, "microbatch 1 layer 27 compute"));

    let expected = queued_prefill.estimated_latency_s + whole_decode.estimated_latency_s;
    assert!(
        (queued_e2e.estimated_latency_s - expected).abs() < expected * 1e-9,
        "e2e {} != prefill + decode {expected}",
        queued_e2e.estimated_latency_s
    );
    assert!(queued_e2e.estimated_latency_s < whole_e2e.estimated_latency_s);
}

#[test]
fn queued_prefill_records_a_microbatching_approximation() {
    let queued = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::VllmBatchQueue,
    );
    assert!(
        queued
            .approximations
            .iter()
            .any(|approximation| approximation.code == PIPELINE_MICROBATCH_OVERLAP)
    );
    let whole = score(
        8,
        InferencePhase::Prefill,
        2,
        PipelineMicrobatching::WholeBatch,
    );
    assert!(
        !whole
            .approximations
            .iter()
            .any(|approximation| approximation.code == PIPELINE_MICROBATCH_OVERLAP)
    );
}

#[test]
fn four_stage_queued_prefill_still_overlaps() {
    // One GPU per node: the trace's compute resources are per node.
    let mut text = String::from(
        r#"
        schema_version = 1
        [cluster]
        preset = "custom"
        [interconnect]
        kind = "ethernet"
        variant = "10g"
        oversubscription = 1.0
        "#,
    );
    for id in 0..4 {
        text.push_str(&format!(
            r#"
        [[nodes]]
        id = {id}
        gpu = "a100_40gb"
        gpu_count = 1
        hbm_gb = 24.0
        hbm_bandwidth_gb_s = 936.0
        peak_f16_tflops = 88.0
        intra = "pcie_gen4"
        nics = {{ count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }}
            "#
        ));
    }
    let cluster = parse_cluster(&text).expect("four-node cluster parses");
    let run = |policy| {
        Solver::score_config_with_options(
            &cluster,
            &qwen7b(),
            &request(8, InferencePhase::Prefill),
            pipeline(4),
            options(policy),
        )
    };
    let whole = run(PipelineMicrobatching::WholeBatch);
    let queued = run(PipelineMicrobatching::VllmBatchQueue);
    assert!(whole.feasible && queued.feasible);
    assert!(queued.estimated_latency_s < whole.estimated_latency_s);
    let sends = queued
        .operations
        .iter()
        .filter(|operation| operation.name.contains("pipeline sendrecv"))
        .count();
    assert_eq!(sends, 6);
}
