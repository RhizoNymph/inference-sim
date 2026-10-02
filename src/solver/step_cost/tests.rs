use super::*;
use crate::{config::parse_cluster, workload::DType};

/// Three single-RTX-3090 nodes, as in lab-runs/2026-09-28-static-batch.
fn rtx3090_cluster() -> Cluster {
    let node = |id: u32| {
        format!(
            r#"
            [[nodes]]
            id = {id}
            gpu = "a100_40gb"
            gpu_count = 1
            hbm_gb = 24.0
            hbm_bandwidth_gb_s = 936.0
            peak_f16_tflops = 71.0
            intra = "pcie_gen4"
            nics = {{ count = 1, affinity = "uniform", bandwidth_gbps = 9.41, rail_count = 1 }}
            "#
        )
    };
    let text = format!(
        r#"
        schema_version = 1
        [cluster]
        preset = "custom"
        [interconnect]
        kind = "ethernet"
        variant = "10g"
        oversubscription = 1.0
        {}{}{}
        "#,
        node(0),
        node(1),
        node(2)
    );
    parse_cluster(&text).expect("rtx3090 test cluster parses")
}

/// Two-GPU node so tensor parallelism stays on one node.
fn dual_gpu_cluster() -> Cluster {
    Cluster::h100_sxm_nodes(
        1,
        crate::types::fabric::variants::ib::IbVariant::Ndr.default_profile(),
    )
}

fn qwen7b() -> ModelSpec {
    ModelSpec {
        layers: 28,
        hidden_size: 3584,
        attention_heads: 28,
        kv_heads: 4,
        vocab_size: 152_064,
        parameters: Bytes::from_gigabytes(15.23),
        parameter_count: None,
        parameter_count_source: crate::workload::ParameterCountSource::Explicit,
        dtype: DType::Bf16,
        kv_dtype: None,
        experts: None,
    }
}

fn fitted_3090_calibration() -> SimulationCalibration {
    SimulationCalibration {
        compute_efficiency: 0.850_699,
        decode_memory_bandwidth_scale: 0.838_309,
        ..SimulationCalibration::default()
    }
}

fn config(tensor_ranks: u32, pipeline_ranks: u32) -> ParallelismConfig {
    ParallelismConfig {
        tensor_ranks,
        pipeline_ranks,
        expert_ranks: 1,
        data_ranks: 1,
    }
}

fn request(
    batch_size: u32,
    prompt_tokens: u32,
    decode_tokens: u32,
    phase: InferencePhase,
) -> InferenceRequest {
    InferenceRequest {
        batch_size,
        prompt_tokens,
        decode_tokens,
        max_sequence_tokens: prompt_tokens + decode_tokens,
        phase,
    }
}

fn score(
    cluster: &Cluster,
    request: &InferenceRequest,
    config: ParallelismConfig,
    calibration: SimulationCalibration,
) -> ScoredParallelismConfig {
    let score = Solver::score_config_with_options(
        cluster,
        &qwen7b(),
        request,
        config,
        SolverOptions {
            calibration,
            ..SolverOptions::default()
        },
    );
    assert!(score.feasible, "{:?}", score.rejected_reason);
    score
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1e-12)
}

#[test]
fn pure_decode_step_matches_static_one_token_decode() {
    let cluster = rtx3090_cluster();
    let calibration = fitted_3090_calibration();
    for (batch, prompt) in [(1, 512), (8, 2048), (32, 512)] {
        let static_request = request(batch, prompt, 1, InferencePhase::Decode);
        let scored = score(&cluster, &static_request, config(1, 1), calibration);
        let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
        let mut work = StepWork::default();
        work.add_decode(batch, prompt + 1);
        let step = model.step_latency(&work);
        assert!(
            relative_error(step.total_s, scored.estimated_latency_s) < 1e-9,
            "batch {batch} prompt {prompt}: step {} vs static {}",
            step.total_s,
            scored.estimated_latency_s
        );
    }
}

#[test]
fn compute_bound_prefill_step_matches_static_prefill() {
    let cluster = rtx3090_cluster();
    let calibration = fitted_3090_calibration();
    for (batch, prompt) in [(1, 512), (1, 2048), (8, 512)] {
        let static_request = request(batch, prompt, 1, InferencePhase::Prefill);
        let scored = score(&cluster, &static_request, config(1, 1), calibration);
        let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
        let mut work = StepWork::default();
        work.add_prefill_chunk(batch, 0, prompt);
        let step = model.step_latency(&work);
        assert!(
            step.compute_s > step.memory_s,
            "prefill of {prompt} tokens is compute bound"
        );
        assert!(
            relative_error(step.total_s, scored.estimated_latency_s) < 1e-9,
            "batch {batch} prompt {prompt}: step {} vs static {}",
            step.total_s,
            scored.estimated_latency_s
        );
    }
}

#[test]
fn chunked_prefill_attention_sums_to_the_unchunked_prompt() {
    let mut whole = StepWork::default();
    whole.add_prefill_chunk(2, 0, 3000);
    let mut chunks = StepWork::default();
    chunks.add_prefill_chunk(2, 0, 2048);
    chunks.add_prefill_chunk(2, 2048, 952);
    assert_eq!(whole.prefill_tokens(), chunks.prefill_tokens());
    assert!(
        relative_error(
            chunks.prefill_attention_pairs,
            whole.prefill_attention_pairs
        ) < 1e-12
    );
    assert_eq!(chunks.prefill_context_tokens, 2 * 2048);
}

#[test]
fn decode_step_is_nearly_flat_in_batch_size_when_memory_bound() {
    let cluster = rtx3090_cluster();
    let calibration = fitted_3090_calibration();
    let scored = score(
        &cluster,
        &request(1, 512, 1, InferencePhase::Decode),
        config(1, 1),
        calibration,
    );
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
    let step = |model: &mut IterationCostModel<'_>, batch: u32| {
        let mut work = StepWork::default();
        work.add_decode(batch, 576);
        model.step_latency(&work).total_s
    };
    let one = step(&mut model, 1);
    let thirty_two = step(&mut model, 32);
    // Measured on the 3090: 19.3 ms at batch 1, 20.8 ms at batch 32.
    assert!((0.018..0.021).contains(&one), "batch-1 step {one}");
    assert!(thirty_two > one);
    assert!(
        thirty_two / one < 1.15,
        "batch-32 step {thirty_two} vs batch-1 {one}"
    );
}

#[test]
fn mixed_step_costs_at_least_its_parts_and_one_weight_read() {
    let cluster = rtx3090_cluster();
    let calibration = fitted_3090_calibration();
    let scored = score(
        &cluster,
        &request(1, 512, 1, InferencePhase::Decode),
        config(1, 1),
        calibration,
    );
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
    let mut decode = StepWork::default();
    decode.add_decode(16, 600);
    let mut prefill = StepWork::default();
    prefill.add_prefill_chunk(1, 0, 512);
    let mut mixed = decode;
    mixed.add_prefill_chunk(1, 0, 512);
    let decode_s = model.step_latency(&decode).total_s;
    let prefill_s = model.step_latency(&prefill).total_s;
    let mixed_s = model.step_latency(&mixed).total_s;
    assert!(mixed_s >= prefill_s && mixed_s >= decode_s);
    // One forward pass reads the weights once, so a mixed step is far cheaper
    // than running the prefill and the decode as two separate passes.
    assert!(mixed_s < prefill_s + decode_s * 0.5);
    assert_eq!(mixed.total_tokens(), 16 + 512);
}

#[test]
fn per_step_overhead_comes_from_scheduler_overhead_calibration() {
    let cluster = rtx3090_cluster();
    let calibration = SimulationCalibration {
        scheduler_overhead_us: 1_500.0,
        ..fitted_3090_calibration()
    };
    let scored = score(
        &cluster,
        &request(1, 512, 1, InferencePhase::Decode),
        config(1, 1),
        calibration,
    );
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
    let mut work = StepWork::default();
    work.add_decode(1, 513);
    let step = model.step_latency(&work);
    assert!((step.overhead_s - 0.0015).abs() < 1e-12);
    assert!((step.total_s - (step.compute_s.max(step.memory_s) + 0.0015)).abs() < 1e-12);
    assert!(relative_error(step.total_s, scored.estimated_latency_s) < 1e-9);
}

#[test]
fn tensor_parallel_steps_pay_two_all_reduces_per_layer_sized_by_step_tokens() {
    let cluster = dual_gpu_cluster();
    let calibration = SimulationCalibration::default();
    let scored = score(
        &cluster,
        &request(1, 512, 1, InferencePhase::Decode),
        config(2, 1),
        calibration,
    );
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
    let mut small = StepWork::default();
    small.add_decode(1, 513);
    let mut large = StepWork::default();
    large.add_prefill_chunk(1, 0, 2048);
    let small_step = model.step_latency(&small);
    let large_step = model.step_latency(&large);
    assert!(small_step.communication_s > 0.0);
    assert!(large_step.communication_s > small_step.communication_s);

    let call = CollectiveCall {
        kind: CollectiveKind::AllReduce,
        participants: scored.groups.tensor_groups[0].clone(),
        bytes_per_rank: Bytes::from_bytes(3584 * 2),
        dtype: DType::Bf16,
        reduction: Some(ReductionOp::Sum),
        root: None,
        phase: InferencePhase::Decode,
        algorithm: CollectiveAlgorithm::Hierarchical,
    };
    let one = Solver::estimate_collective_with_calibration(
        &cluster,
        &scored.placement,
        &call,
        calibration,
    );
    // Plus vLLM's once-per-step vocab-parallel collectives: one embedding
    // all-reduce at the same size, and a logits all-gather for the one
    // sampling sequence (each rank's shard of the 152064-token vocab).
    let logits = Solver::estimate_collective_with_calibration(
        &cluster,
        &scored.placement,
        &CollectiveCall {
            kind: CollectiveKind::AllGather,
            participants: scored.groups.tensor_groups[0].clone(),
            bytes_per_rank: Bytes::from_bytes(152064 / 2 * 2),
            dtype: DType::Bf16,
            reduction: None,
            root: None,
            phase: InferencePhase::Decode,
            algorithm: CollectiveAlgorithm::Ring,
        },
        calibration,
    );
    assert!(
        relative_error(
            small_step.communication_s,
            one.total_s * (2.0 * 28.0 + 1.0) + logits.total_s
        ) < 1e-9
    );
    // Without compute/communication overlap the step serializes them.
    assert!(
        (small_step.total_s
            - (small_step.compute_s.max(small_step.memory_s) + small_step.communication_s))
            .abs()
            < 1e-12
    );
}

#[test]
fn tensor_parallel_decode_step_matches_static_decode() {
    let cluster = dual_gpu_cluster();
    let calibration = SimulationCalibration::default();
    let static_request = request(4, 1024, 1, InferencePhase::Decode);
    let scored = score(&cluster, &static_request, config(2, 1), calibration);
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
    let mut work = StepWork::default();
    work.add_decode(4, 1025);
    let step = model.step_latency(&work);
    assert!(
        relative_error(step.total_s, scored.estimated_latency_s) < 1e-9,
        "step {} vs static {}",
        step.total_s,
        scored.estimated_latency_s
    );
}

#[test]
fn pipeline_parallel_steps_run_stages_in_sequence_like_the_static_solver() {
    let cluster = rtx3090_cluster();
    let calibration = fitted_3090_calibration();
    let static_request = request(1, 512, 1, InferencePhase::Decode);
    let scored = score(&cluster, &static_request, config(1, 2), calibration);
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
    let mut work = StepWork::default();
    work.add_decode(1, 513);
    let step = model.step_latency(&work);
    assert!(
        step.communication_s > 0.0,
        "stage boundary send/recv is priced"
    );
    assert!(
        relative_error(step.total_s, scored.estimated_latency_s) < 1e-9,
        "step {} vs static {}",
        step.total_s,
        scored.estimated_latency_s
    );
}

#[test]
fn empty_step_costs_nothing() {
    let cluster = rtx3090_cluster();
    let calibration = fitted_3090_calibration();
    let scored = score(
        &cluster,
        &request(1, 512, 1, InferencePhase::Decode),
        config(1, 1),
        calibration,
    );
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();
    assert_eq!(
        model.step_latency(&StepWork::default()),
        StepLatency::default()
    );
}

#[test]
fn unplaced_config_is_a_typed_error() {
    let cluster = rtx3090_cluster();
    let mut scored = score(
        &cluster,
        &request(1, 512, 1, InferencePhase::Decode),
        config(1, 1),
        SimulationCalibration::default(),
    );
    scored.placement.rank_to_gpu.clear();
    let result = IterationCostModel::new(
        &cluster,
        &qwen7b(),
        &scored,
        SimulationCalibration::default(),
    );
    assert!(matches!(result, Err(StepCostError::EmptyPlacement)));
}

#[test]
fn logits_all_gather_scales_with_sampling_sequences_not_tokens() {
    let cluster = dual_gpu_cluster();
    let calibration = SimulationCalibration::default();
    let scored = score(
        &cluster,
        &request(4, 1024, 1, InferencePhase::Decode),
        config(2, 1),
        calibration,
    );
    let mut model = IterationCostModel::new(&cluster, &qwen7b(), &scored, calibration).unwrap();

    let mut decode = StepWork::default();
    decode.add_decode(4, 1025);
    let mut partial_prefill = StepWork::default();
    partial_prefill.add_prefill_chunk(1, 0, 4);

    assert_eq!(decode.total_tokens(), partial_prefill.total_tokens());
    assert_eq!(decode.sampling_sequences(), 4);
    assert_eq!(partial_prefill.sampling_sequences(), 0);
    let decode_comm = model.step_latency(&decode).communication_s;
    let prefill_comm = model.step_latency(&partial_prefill).communication_s;
    assert!(
        decode_comm > prefill_comm,
        "decode {decode_comm} vs non-sampling prefill {prefill_comm}"
    );
}

#[test]
fn completing_prefill_chunk_samples_one_token_per_sequence() {
    let mut work = StepWork::default();
    work.add_prefill_chunk(2, 0, 512);
    assert_eq!(work.sampling_sequences(), 0);
    work.add_completing_prefill_chunk(2, 512, 128);
    assert_eq!(work.sampling_sequences(), 2);
    assert_eq!(work.prefill_tokens(), 2 * 512 + 2 * 128);
}
