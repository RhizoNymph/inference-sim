use super::*;

#[test]
fn parses_h100_cluster_config() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 2

            [interconnect]
            kind = "ib"
            variant = "ndr"
            oversubscription = 2.0

            [nics]
            count = 8
            affinity = "dedicated"
            bandwidth_gbps = 400.0
            "#,
    )
    .unwrap();

    assert_eq!(cluster.total_gpus(), 16);
    assert_eq!(cluster.node(0).unwrap().network.nic_count, 8);
}

#[test]
fn parses_workload_and_search_config() {
    let workload = parse_workload(
        r#"
            [model]
            id = "llama-70b"
            layers = 80
            hidden_size = 8192
            attention_heads = 64
            kv_heads = 8
            vocab_size = 128256
            parameters_gb = 140.0
            parameter_count_billion = 72.0
            dtype = "bf16"
            kv_dtype = "fp8"

            [request]
            batch_size = 4
            prompt_tokens = 1024
            decode_tokens = 128
            max_sequence_tokens = 2048
            phase = "end_to_end"

            [search]
            tensor_ranks = [1, 2, 4, 8]
            pipeline_ranks = [1, 2]
            expert_ranks = [1]
            data_ranks = [1, 2]
            "#,
    )
    .unwrap();

    assert_eq!(workload.model.dtype, DType::Bf16);
    assert_eq!(workload.model.kv_dtype, Some(DType::Fp8));
    assert_eq!(workload.model.parameter_count_billion(), 72.0);
    assert_eq!(workload.model_id.as_deref(), Some("llama-70b"));
    assert_eq!(workload.request.phase, InferencePhase::EndToEnd);
    assert_eq!(workload.search_space.tensor_ranks, vec![1, 2, 4, 8]);
}

#[test]
fn approximation_policy_presets_apply_defaults_and_field_overrides() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 4
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [approximation_policy]
            preset = "calibration-only"
            warn_categories = ["memory"]

            [[approximation_policy.metric_gates]]
            metrics = ["e2el", "maximize_throughput"]
            reject_categories = ["topology"]
            warn_codes = ["serving-queue-component-uncalibrated"]
            "#,
    )
    .unwrap();

    assert_eq!(
        workload.approximation_policy.preset,
        Some(ApproximationPolicyPreset::CalibrationOnly)
    );
    assert_eq!(
        workload.approximation_policy.default_action,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.approximation_policy.reject_categories,
        vec!["calibration", "runtime"]
    );
    assert_eq!(
        workload.approximation_policy.warn_categories,
        vec!["memory"]
    );
    assert_eq!(workload.approximation_policy.metric_gates.len(), 1);
    assert_eq!(
        workload.approximation_policy.metric_gates[0].metrics,
        vec!["e2el", "throughput"]
    );
    assert_eq!(
        workload.approximation_policy.metric_gates[0].reject_categories,
        vec!["topology"]
    );
    assert_eq!(
        workload.approximation_policy.metric_gates[0].warn_codes,
        vec!["serving_queue_component_uncalibrated"]
    );

    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 4
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [approximation_policy]
            preset = "production"
            default = "warn"
            "#,
    )
    .unwrap();

    assert_eq!(
        workload.approximation_policy.preset,
        Some(ApproximationPolicyPreset::ProductionRecommendation)
    );
    assert_eq!(
        workload.approximation_policy.default_action,
        CalibrationGateMode::Warn
    );
}

#[test]
fn rejects_invalid_model_and_request_shapes() {
    struct ShapeCase {
        parameters_gb: f64,
        hidden_size: u32,
        attention_heads: u32,
        kv_heads: u32,
        batch_size: u32,
        prompt_tokens: u32,
        decode_tokens: u32,
        max_sequence_tokens: u32,
    }

    impl Default for ShapeCase {
        fn default() -> Self {
            Self {
                parameters_gb: 16.0,
                hidden_size: 4096,
                attention_heads: 32,
                kv_heads: 8,
                batch_size: 1,
                prompt_tokens: 128,
                decode_tokens: 16,
                max_sequence_tokens: 256,
            }
        }
    }

    fn workload_toml(case: ShapeCase) -> String {
        let ShapeCase {
            parameters_gb,
            hidden_size,
            attention_heads,
            kv_heads,
            batch_size,
            prompt_tokens,
            decode_tokens,
            max_sequence_tokens,
        } = case;
        format!(
            r#"
                [model]
                layers = 4
                hidden_size = {hidden_size}
                attention_heads = {attention_heads}
                kv_heads = {kv_heads}
                vocab_size = 32000
                parameters_gb = {parameters_gb}
                dtype = "bf16"

                [request]
                batch_size = {batch_size}
                prompt_tokens = {prompt_tokens}
                decode_tokens = {decode_tokens}
                max_sequence_tokens = {max_sequence_tokens}
                phase = "end_to_end"

                [search]
                tensor_ranks = [1]
                pipeline_ranks = [1]
                expert_ranks = [1]
                data_ranks = [1]
                "#
        )
    }

    let err = parse_workload(&workload_toml(ShapeCase {
        hidden_size: 4097,
        ..ShapeCase::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("model.hidden_size must be divisible by model.attention_heads")
    );

    let err = parse_workload(&workload_toml(ShapeCase {
        kv_heads: 64,
        ..ShapeCase::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("model.kv_heads must be less than or equal to model.attention_heads")
    );

    let err = parse_workload(&workload_toml(ShapeCase {
        hidden_size: 3840,
        attention_heads: 30,
        ..ShapeCase::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("model.attention_heads must be divisible by model.kv_heads")
    );

    let err = parse_workload(&workload_toml(ShapeCase {
        batch_size: 0,
        ..ShapeCase::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("request.batch_size must be greater than zero")
    );

    let err = parse_workload(&workload_toml(ShapeCase {
        max_sequence_tokens: 100,
        ..ShapeCase::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("request.max_sequence_tokens must be greater than or equal")
    );

    let err = parse_workload(&workload_toml(ShapeCase {
        parameters_gb: 0.0,
        ..ShapeCase::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("model.parameters_gb must be finite and greater than zero")
    );
}

#[test]
fn rejects_invalid_serving_traffic_max_sequence_shapes() {
    fn serving_workload_with_traffic(traffic: &str) -> String {
        format!(
            r#"
                [model]
                layers = 4
                hidden_size = 4096
                attention_heads = 32
                kv_heads = 8
                vocab_size = 32000
                parameters_gb = 16.0
                dtype = "bf16"

                [request]
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 16
                max_sequence_tokens = 512
                phase = "end_to_end"

                [search]
                tensor_ranks = [1]
                pipeline_ranks = [1]
                expert_ranks = [1]
                data_ranks = [1]

                [serving]
                mode = "colocated"
                prefill_nodes = [0]
                decode_nodes = [0]

                [serving.traffic]
                request_count = 1
                {traffic}
                "#
        )
    }

    let err = parse_workload(&serving_workload_with_traffic(
        r#"
            [[serving.traffic.requests]]
            arrival_ms = 0.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 100
            "#,
    ))
    .unwrap_err();
    assert!(
        err.to_string().contains(
            "serving.traffic.requests[0].max_sequence_tokens must be greater than or equal"
        )
    );

    let err = parse_workload(&serving_workload_with_traffic(
        r#"
            [[serving.traffic.shape_profiles]]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 100
            "#,
    ))
    .unwrap_err();
    assert!(err.to_string().contains(
        "serving.traffic.shape_profiles[0].max_sequence_tokens must be greater than or equal"
    ));
}

#[test]
fn rejects_duplicate_serving_trace_request_ids() {
    fn serving_workload_with_traffic(traffic: &str) -> String {
        format!(
            r#"
                [model]
                layers = 4
                hidden_size = 4096
                attention_heads = 32
                kv_heads = 8
                vocab_size = 32000
                parameters_gb = 16.0
                dtype = "bf16"

                [request]
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 16
                max_sequence_tokens = 512
                phase = "end_to_end"

                [search]
                tensor_ranks = [1]
                pipeline_ranks = [1]
                expert_ranks = [1]
                data_ranks = [1]

                [serving]
                mode = "colocated"
                prefill_nodes = [0]
                decode_nodes = [0]

                [serving.traffic]
                request_count = 2
                {traffic}
                "#
        )
    }

    let err = parse_workload(&serving_workload_with_traffic(
        r#"
            [[serving.traffic.requests]]
            request_id = "req-dup"
            arrival_ms = 0.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16

            [[serving.traffic.requests]]
            request_id = "req-dup"
            arrival_ms = 1.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            "#,
    ))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("serving.traffic request_id 'req-dup' is duplicated")
    );
}

#[test]
fn parses_bursty_serving_arrivals() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 8
            arrival = "bursty"
            burst_size = 4
            burst_interval_ms = 10.0
            intra_burst_gap_ms = 0.5
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.request_count, Some(8));
    assert_eq!(
        traffic.arrival,
        ServingArrivalPattern::Bursty {
            burst_size: 4,
            burst_interval_s: 0.010,
            intra_burst_gap_s: 0.0005,
        }
    );
}

#[test]
fn parses_diurnal_serving_arrivals() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 8
            arrival = "diurnal"
            arrival_seed = 99
            diurnal_min_rate_per_s = 10.0
            diurnal_max_rate_per_s = 100.0
            diurnal_period_ms = 1000.0
            diurnal_phase_ms = 250.0
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.request_count, Some(8));
    assert_eq!(
        traffic.arrival,
        ServingArrivalPattern::Diurnal {
            min_rate_per_s: 10.0,
            max_rate_per_s: 100.0,
            period_s: 1.0,
            phase_s: 0.25,
            seed: 99,
        }
    );
}

#[test]
fn parses_self_similar_serving_arrivals() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 8
            arrival = "self_similar"
            arrival_seed = 123
            arrival_rate_per_s = 100.0
            self_similar_pareto_shape = 1.35
            self_similar_max_gap_ms = 200.0
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.request_count, Some(8));
    assert_eq!(
        traffic.arrival,
        ServingArrivalPattern::SelfSimilar {
            rate_per_s: 100.0,
            pareto_shape: 1.35,
            max_gap_s: Some(0.2),
            seed: 123,
        }
    );
}

#[test]
fn parses_trace_derived_serving_arrivals() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 2
            arrival = "trace_derived"
            batch_sizes = [4]
            prompt_tokens = [1024]
            decode_tokens = [64]

            [[serving.traffic.requests]]
            arrival_ms = 0.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8

            [[serving.traffic.requests]]
            arrival_ms = 2.5
            batch_size = 2
            prompt_tokens = 256
            decode_tokens = 16
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.request_count, Some(2));
    assert_eq!(traffic.arrival, ServingArrivalPattern::TraceDerived);
    assert_eq!(
        traffic
            .trace_requests
            .iter()
            .map(|request| request.arrival_s)
            .collect::<Vec<_>>(),
        vec![0.0, 0.0025]
    );
}

#[test]
fn parses_correlated_serving_shape_profiles() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 8
            shape_seed = 11

            [[serving.traffic.shape_profiles]]
            name = "tenant-a-small"
            weight = 3.0
            tenant = "tenant-a"
            model_id = "model-a"
            cache_key = "shared-prefix-a"
            priority = 9
            batch_size = 1
            prompt_tokens = 512
            decode_tokens = 32
            max_sequence_tokens = 1024
            prefix_cache_hit_tokens = 128
            ttft_slo_ms = 100.0
            tpot_slo_ms = 20.0
            e2el_slo_ms = 500.0
            request_timeout_ms = 750.0
            deadline_after_ms = 50.0

            [[serving.traffic.shape_profiles]]
            name = "tenant-b-large"
            weight = 1.0
            tenant = "tenant-b"
            model_id = "model-b"
            cache_key = "shared-prefix-b"
            priority = -1
            batch_size = 4
            prompt_tokens = 4096
            decode_tokens = 256
            max_sequence_tokens = 8192
            prefix_cache_hit_rate = 0.25
            ttft_slo_ms = 400.0
            tpot_slo_ms = 60.0
            itl_slo_ms = 60.0
            e2el_slo_ms = 2000.0
            request_timeout_ms = 3000.0
            cancel_after_ms = 10.0
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.shape_profiles.len(), 2);
    assert_eq!(traffic.shape_profiles[0].name, "tenant-a-small");
    assert_eq!(traffic.shape_profiles[0].weight, 3.0);
    assert_eq!(
        traffic.shape_profiles[0].tenant.as_deref(),
        Some("tenant-a")
    );
    assert_eq!(
        traffic.shape_profiles[0].model_id.as_deref(),
        Some("model-a")
    );
    assert_eq!(
        traffic.shape_profiles[0].cache_key.as_deref(),
        Some("shared-prefix-a")
    );
    assert_eq!(traffic.shape_profiles[0].priority, Some(9));
    assert_eq!(traffic.shape_profiles[0].prompt_tokens, 512);
    assert_eq!(traffic.shape_profiles[0].decode_tokens, 32);
    assert_eq!(traffic.shape_profiles[0].prefix_cache_hit_tokens, Some(128));
    assert_eq!(
        traffic.shape_profiles[0].slo,
        ServingRequestSlo {
            ttft_s: Some(0.100),
            tpot_s: Some(0.020),
            itl_s: None,
            e2el_s: Some(0.500),
        }
    );
    assert_eq!(traffic.shape_profiles[0].request_timeout_s, Some(0.750));
    assert_eq!(traffic.shape_profiles[0].deadline_after_s, Some(0.050));
    assert_eq!(traffic.shape_profiles[1].batch_size, 4);
    assert_eq!(traffic.shape_profiles[1].name, "tenant-b-large");
    assert_eq!(traffic.shape_profiles[1].max_sequence_tokens, Some(8192));
    assert_eq!(traffic.shape_profiles[1].prefix_cache_hit_rate, Some(0.25));
    assert_eq!(
        traffic.shape_profiles[1].slo,
        ServingRequestSlo {
            ttft_s: Some(0.400),
            tpot_s: Some(0.060),
            itl_s: Some(0.060),
            e2el_s: Some(2.000),
        }
    );
    assert_eq!(traffic.shape_profiles[1].request_timeout_s, Some(3.000));
    assert_eq!(traffic.shape_profiles[1].cancellation_after_s, Some(0.010));
}

#[test]
fn rejects_duplicate_serving_shape_profile_names() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 2

            [[serving.traffic.shape_profiles]]
            name = "tenant-a"
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16

            [[serving.traffic.shape_profiles]]
            label = "tenant-a"
            batch_size = 1
            prompt_tokens = 256
            decode_tokens = 16
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("serving.traffic.shape_profiles[1].name 'tenant-a' is duplicated")
    );
}

#[test]
fn parses_explicit_rank_placements() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [[placement.ranks]]
            rank = 0
            node = 1
            gpu = 2

            [[placement.ranks]]
            rank = 1
            node = 1
            gpu = 3

            [serving]
            mode = "fully_disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.prefill_search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving.decode_search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 0
            gpu = 4

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            local_gpu_id = 5
            "#,
    )
    .unwrap();

    assert_eq!(
        workload.placement.unwrap().rank_to_gpu,
        vec![
            GpuAddr {
                node_id: 1,
                local_gpu_id: 2
            },
            GpuAddr {
                node_id: 1,
                local_gpu_id: 3
            }
        ]
    );
    assert_eq!(
        workload.serving_prefill_placement.unwrap().rank_to_gpu,
        vec![GpuAddr {
            node_id: 0,
            local_gpu_id: 4
        }]
    );
    assert_eq!(
        workload.serving_decode_placement.unwrap().rank_to_gpu,
        vec![GpuAddr {
            node_id: 1,
            local_gpu_id: 5
        }]
    );
}

#[test]
fn validates_explicit_placement_against_cluster_and_search() {
    let cluster = Cluster::h100_sxm_nodes(1, IbVariant::Ndr.default_profile());
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [[placement.ranks]]
            rank = 0
            node = 0
            gpu = 0
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(err.to_string().contains(
        "placement.ranks defines 1 ranks but no search candidate has that total rank count"
    ));

    let unavailable = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 256
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [[placement.ranks]]
            rank = 0
            node = 9
            gpu = 0
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &unavailable).unwrap_err();
    assert!(
        err.to_string()
            .contains("placement.ranks[0] references unknown node id 9")
    );
}

#[test]
fn validates_model_dtype_against_gpu_capabilities() {
    let a100_cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            group = "a100"
            gpu = "a100_80gb"
            gpu_count = 2
            "#,
    )
    .unwrap();
    let fp8_workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "fp8"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&a100_cluster, &fp8_workload).unwrap_err();
    assert!(err.to_string().contains("model.dtype fp8"));
    assert!(err.to_string().contains("no available fp8-capable GPUs"));

    let h100_cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            group = "h100"
            gpu = "h100_sxm"
            gpu_count = 2
            "#,
    )
    .unwrap();
    validate_workload_for_cluster(&h100_cluster, &fp8_workload).unwrap();

    let mixed_cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            group = "h100"
            gpu = "h100_sxm"
            gpu_count = 1

            [[nodes]]
            id = 1
            group = "a100"
            gpu = "a100_80gb"
            gpu_count = 1
            "#,
    )
    .unwrap();
    let a100_placement = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "fp8"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [[placement.ranks]]
            rank = 0
            node = 1
            gpu = 0
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&mixed_cluster, &a100_placement).unwrap_err();
    assert!(err.to_string().contains("placement.ranks[0]"));
    assert!(err.to_string().contains("does not support model.dtype fp8"));

    let a100_decode_pool = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "fp8"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"

            [[serving.pool_candidates]]
            prefill_nodes = [0]
            decode_nodes = [1]
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&mixed_cluster, &a100_decode_pool).unwrap_err();
    assert!(
        err.to_string()
            .contains("serving.pool_candidates[0].decode_search for model.dtype fp8")
    );
}

#[test]
fn parses_custom_heterogeneous_cluster_config() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 8
            intra = "nvlink_v4"
            nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0 }

            [[nodes]]
            id = 1
            gpu = "a100_80gb"
            gpu_count = 4
            intra = "nvlink_v3"
            nics = { count = 4, affinity = "dedicated", bandwidth_gbps = 200.0 }
            "#,
    )
    .unwrap();

    assert_eq!(cluster.total_gpus(), 12);
    assert_eq!(
        cluster.gpu(crate::types::common::GpuAddr {
            node_id: 0,
            local_gpu_id: 0
        }),
        Some(Gpu::H100_SXM)
    );
    assert_eq!(
        cluster.gpu(crate::types::common::GpuAddr {
            node_id: 1,
            local_gpu_id: 0
        }),
        Some(Gpu::A100_80GB)
    );
    assert_eq!(cluster.node(1).unwrap().network.nic_count, 4);
}

#[test]
fn parses_custom_node_with_explicit_mixed_gpu_inventory() {
    let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            group = "mixed"
            intra = "pcie_gen5"
            nics = { count = 4, affinity = "shared", gpus_per_nic = 2, bandwidth_gbps = 400.0, rail_count = 4 }
            gpus = [
              { start_id = 0, count = 2, gpu = "h100_sxm" },
              { start_id = 2, count = 1, gpu = "h200_sxm" },
              { id = 3, gpu = "a100_80gb" },
            ]
            "#,
        )
        .unwrap();

    assert_eq!(cluster.total_gpus(), 4);
    assert_eq!(cluster.node_group("mixed"), Some(&[0][..]));
    assert_eq!(
        cluster.gpu(crate::types::common::GpuAddr {
            node_id: 0,
            local_gpu_id: 0
        }),
        Some(Gpu::H100_SXM)
    );
    assert_eq!(
        cluster.gpu(crate::types::common::GpuAddr {
            node_id: 0,
            local_gpu_id: 2
        }),
        Some(Gpu::H200_SXM)
    );
    assert_eq!(
        cluster.gpu(crate::types::common::GpuAddr {
            node_id: 0,
            local_gpu_id: 3
        }),
        Some(Gpu::A100_80GB)
    );
    assert_eq!(cluster.node(0).unwrap().network.nic_count, 4);
}

#[test]
fn parses_calibrated_gpu_profile_overrides() {
    let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[node_groups]]
            label = "calibrated"
            start_id = 0
            count = 1
            gpu = "a100_80gb"
            gpu_count = 3
            hbm_gb = 96.0
            hbm_bandwidth_gb_s = 2500.0
            peak_f16_tflops = 420.0
            peak_f8_tflops = 840.0
            gpu_profile_overrides = [
              { gpu = 1, hbm_gb = 48.0, peak_f16_tflops = 210.0 },
              { gpus = [2], hbm_bandwidth_gb_s = 1800.0 },
            ]

            [[nodes]]
            id = 10
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 200.0 }
            gpus = [
              { id = 0, gpu = "h100_sxm", hbm_gb = 72.0, hbm_bandwidth_gb_s = 3100.0, peak_f16_tflops = 900.0 },
              { id = 1, gpu = "a100_80gb", peak_f8_tflops = 700.0 },
            ]
            gpu_profile_overrides = [
              { gpu = 1, hbm_gb = 84.0, peak_f16_tflops = 330.0 },
            ]
            "#,
        )
        .unwrap();

    let group_profile = cluster
        .gpu_profile(GpuAddr {
            node_id: 0,
            local_gpu_id: 0,
        })
        .unwrap();
    assert!((group_profile.hbm_size.as_gigabytes() - 96.0).abs() < 1e-9);
    assert!((group_profile.hbm_bandwidth.as_gigabytes_per_sec() - 2500.0).abs() < 1e-9);
    assert!((group_profile.peak_f16_flops - 420.0).abs() < 1e-9);
    assert_eq!(group_profile.peak_f8_flops, Some(840.0));

    let per_gpu_memory_override = cluster
        .gpu_profile(GpuAddr {
            node_id: 0,
            local_gpu_id: 1,
        })
        .unwrap();
    assert!((per_gpu_memory_override.hbm_size.as_gigabytes() - 48.0).abs() < 1e-9);
    assert!((per_gpu_memory_override.hbm_bandwidth.as_gigabytes_per_sec() - 2500.0).abs() < 1e-9);
    assert!((per_gpu_memory_override.peak_f16_flops - 210.0).abs() < 1e-9);
    assert_eq!(per_gpu_memory_override.peak_f8_flops, Some(840.0));

    let per_gpu_bandwidth_override = cluster
        .gpu_profile(GpuAddr {
            node_id: 0,
            local_gpu_id: 2,
        })
        .unwrap();
    assert!((per_gpu_bandwidth_override.hbm_size.as_gigabytes() - 96.0).abs() < 1e-9);
    assert!(
        (per_gpu_bandwidth_override
            .hbm_bandwidth
            .as_gigabytes_per_sec()
            - 1800.0)
            .abs()
            < 1e-9
    );
    assert!((per_gpu_bandwidth_override.peak_f16_flops - 420.0).abs() < 1e-9);
    assert_eq!(per_gpu_bandwidth_override.peak_f8_flops, Some(840.0));

    let explicit_profile = cluster
        .gpu_profile(GpuAddr {
            node_id: 10,
            local_gpu_id: 0,
        })
        .unwrap();
    assert!((explicit_profile.hbm_size.as_gigabytes() - 72.0).abs() < 1e-9);
    assert!((explicit_profile.hbm_bandwidth.as_gigabytes_per_sec() - 3100.0).abs() < 1e-9);
    assert!((explicit_profile.peak_f16_flops - 900.0).abs() < 1e-9);
    assert_eq!(explicit_profile.peak_f8_flops, Some(1979.0));

    let a100_with_fp8 = cluster
        .gpu_profile(GpuAddr {
            node_id: 10,
            local_gpu_id: 1,
        })
        .unwrap();
    assert!((a100_with_fp8.hbm_size.as_gigabytes() - 84.0).abs() < 1e-9);
    assert!((a100_with_fp8.peak_f16_flops - 330.0).abs() < 1e-9);
    assert_eq!(a100_with_fp8.peak_f8_flops, Some(700.0));

    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "fp8"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
    )
    .unwrap();
    validate_workload_for_cluster(&cluster, &workload).unwrap();
}

#[test]
fn parses_explicit_gpu_nic_locality() {
    let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 4
            intra = "pcie_gen5"
            nics = { count = 4, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 4, nic_rail_map = [{ nic = 2, rail = 1 }, { nic_id = 3, rail_id = 0 }], gpu_nic_map = [{ gpu = 0, nics = [2, 3] }, { local_gpu_id = 1, nic = 1 }], gpu_numa_map = [{ gpus = [0, 1], domain = 0 }, { gpu = 2, socket = 1 }], nic_numa_map = [{ nics = [0, 1], numa = 0 }, { nic = 2, numa = 1 }, { nic = 3, socket = 1 }], cross_numa_bandwidth_scale = 0.5, cross_numa_latency_scale = 2.0, nic_bandwidth_overrides = [{ nic = 1, bandwidth_gbps = 200.0 }], nic_latency_scale_overrides = [{ nics = [1, 2], latency_scale = 2.5 }], gpu_nic_paths = [{ gpu = 0, nic = 2, label = "same_pcie", bandwidth_gbps = 300.0, latency_us = 1.5, gpudirect = true }, { gpu = 0, nic = 3, label = "host_staged", available = false, gpudirect = false }] }
            "#,
        )
        .unwrap();

    let network = &cluster.node(0).unwrap().network;
    assert_eq!(network.nic_rail_map.get(&2), Some(&1));
    assert_eq!(network.nic_rail_map.get(&3), Some(&0));
    assert_eq!(network.rail_id(2), 1);
    assert_eq!(network.rail_id(3), 0);
    assert_eq!(network.gpu_nic_map.get(&0), Some(&vec![2, 3]));
    assert_eq!(network.gpu_nic_map.get(&1), Some(&vec![1]));
    assert_eq!(network.gpu_numa_domain(0), Some(0));
    assert_eq!(network.gpu_numa_domain(1), Some(0));
    assert_eq!(network.gpu_numa_domain(2), Some(1));
    assert_eq!(network.nic_numa_domain(0), Some(0));
    assert_eq!(network.nic_numa_domain(1), Some(0));
    assert_eq!(network.nic_numa_domain(2), Some(1));
    assert_eq!(network.nic_numa_domain(3), Some(1));
    assert_eq!(network.cross_numa_bandwidth_scale, 0.5);
    assert_eq!(network.cross_numa_latency_scale, 2.0);
    assert!((network.nic_bandwidth(1).as_gigabits_per_sec() - 200.0).abs() < 1e-9);
    assert_eq!(network.nic_latency_scale(1), 2.5);
    assert_eq!(network.nic_latency_scale(2), 2.5);
    assert_eq!(network.nic_candidates_for_gpu(0), vec![2]);
    assert_eq!(network.nic_candidates_for_gpu(2), vec![0, 1, 2, 3]);
    let path = network.gpu_nic_path(0, 2).unwrap();
    assert_eq!(path.label.as_deref(), Some("same_pcie"));
    assert!((path.bandwidth.unwrap().as_gigabits_per_sec() - 300.0).abs() < 1e-9);
    assert!((path.latency.unwrap().to_us() - 1.5).abs() < 1e-9);
    assert_eq!(path.gpudirect, Some(true));
    assert!(!network.gpu_nic_path(0, 3).unwrap().available);
}

#[test]
fn parses_disabled_gpu_and_nic_inventory() {
    let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 4
            disabled_gpus = [3]
            gpu_states = [{ gpu = 1, state = "maintenance" }, { gpus = [2], state = "reserved" }]
            intra = "pcie_gen5"
            nics = { count = 4, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 4, disabled_nics = [2], nic_states = [{ nic = 1, state = "maintenance" }], gpu_nic_map = [{ gpu = 0, nics = [1, 2, 3] }] }
            "#,
        )
        .unwrap();

    let node = cluster.node(0).unwrap();
    assert!(node.disabled_gpus.contains(&3));
    assert!(node.disabled_gpus.contains(&1));
    assert!(node.disabled_gpus.contains(&2));
    assert_eq!(node.gpu_operational_state(1), OperationalState::Maintenance);
    assert_eq!(node.gpu_operational_state(2), OperationalState::Reserved);
    assert_eq!(node.gpu_operational_state(3), OperationalState::Disabled);
    assert!(!node.is_gpu_available(3));
    assert_eq!(node.available_gpu_count(), 1);
    assert!(node.network.disabled_nics.contains(&2));
    assert!(node.network.disabled_nics.contains(&1));
    assert_eq!(
        node.network.nic_operational_state(1),
        OperationalState::Maintenance
    );
    assert_eq!(
        node.network.nic_operational_state(2),
        OperationalState::Disabled
    );
    assert_eq!(node.network.active_nic_count(), 2);
    assert_eq!(node.network.nic_candidates_for_gpu(0), vec![3]);
    assert_eq!(cluster.available_gpus(), 1);
}

#[test]
fn parses_custom_node_groups_and_group_links() {
    let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from_group = "h100"
            to_group = "h100"
            kind = "ib"
            variant = "ndr"

            [[interconnect.links]]
            from_group = "h100"
            to_group = "a100"
            kind = "ib"
            variant = "hdr"

            [[node_groups]]
            label = "h100"
            start_id = 0
            count = 2
            gpu = "h100_sxm"
            gpu_count = 8
            intra = "nvlink_v4"
            nics = { count = 8, affinity = "dedicated", bandwidth_gbps = 400.0, rail_count = 8 }

            [[node_groups]]
            label = "a100"
            start_id = 10
            count = 1
            gpu = "a100_80gb"
            gpu_count = 4
            intra = "nvlink_v3"
            nics = { count = 4, affinity = "shared", gpus_per_nic = 2, bandwidth_gbps = 200.0, rail_count = 4 }
            "#,
        )
        .unwrap();

    assert_eq!(cluster.total_gpus(), 20);
    assert_eq!(
        cluster.gpu(crate::types::common::GpuAddr {
            node_id: 0,
            local_gpu_id: 0
        }),
        Some(Gpu::H100_SXM)
    );
    assert_eq!(
        cluster.gpu(crate::types::common::GpuAddr {
            node_id: 10,
            local_gpu_id: 0
        }),
        Some(Gpu::A100_80GB)
    );

    let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
        panic!("expected custom inter-node topology");
    };
    assert!(edges.contains_key(&UnorderedPair::new(0, 1)));
    assert!(edges.contains_key(&UnorderedPair::new(0, 10)));
    assert!(edges.contains_key(&UnorderedPair::new(1, 10)));
    assert_eq!(edges.len(), 3);
}

#[test]
fn parses_custom_interconnect_links_scoped_to_rails() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rails = [0, 2]

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ethernet"
            variant = "100g"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 4
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 4
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }
            "#,
    )
    .unwrap();

    let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
        panic!("expected custom inter-node topology");
    };
    let links = edges
        .get(&UnorderedPair::new(0, 1))
        .expect("missing custom link");
    assert_eq!(links.len(), 3);
    assert_eq!(links.iter().filter(|link| link.rail == Some(0)).count(), 1);
    assert_eq!(links.iter().filter(|link| link.rail == Some(1)).count(), 1);
    assert_eq!(links.iter().filter(|link| link.rail == Some(2)).count(), 1);
}

#[test]
fn parses_custom_interconnect_links_scoped_to_gpu_subsets() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpus = [0, 1]
            to_gpu = 2
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 4
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 4
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }
            "#,
    )
    .unwrap();

    let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
        panic!("expected custom inter-node topology");
    };
    let links = edges
        .get(&UnorderedPair::new(0, 1))
        .expect("missing custom link");
    assert_eq!(links.len(), 1);
    let endpoints = links[0].endpoints.as_ref().unwrap();
    assert_eq!(endpoints.from_node, 0);
    assert_eq!(endpoints.from_gpus, vec![0, 1]);
    assert_eq!(endpoints.to_node, 1);
    assert_eq!(endpoints.to_gpus, vec![2]);
}

#[test]
fn parses_custom_interconnect_links_scoped_to_gpu_types() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from_group = "mixed"
            to_group = "mixed"
            from_gpu_type = "h100_sxm"
            to_gpu_types = ["b200"]
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            group = "mixed"
            intra = "pcie_gen5"
            gpus = [
              { start_id = 0, gpu = "h100_sxm", count = 2 },
              { id = 2, gpu = "b200" },
            ]
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }

            [[nodes]]
            id = 1
            group = "mixed"
            intra = "pcie_gen5"
            gpus = [
              { id = 0, gpu = "a100_80gb" },
              { start_id = 1, gpu = "b200", count = 2 },
            ]
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }
            "#,
    )
    .unwrap();

    let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
        panic!("expected custom inter-node topology");
    };
    let links = edges
        .get(&UnorderedPair::new(0, 1))
        .expect("missing custom link");
    assert_eq!(links.len(), 1);
    let endpoints = links[0].endpoints.as_ref().unwrap();
    assert_eq!(endpoints.from_node, 0);
    assert_eq!(endpoints.from_gpus, vec![0, 1]);
    assert_eq!(endpoints.to_node, 1);
    assert_eq!(endpoints.to_gpus, vec![1, 2]);
}

#[test]
fn parses_custom_interconnect_links_scoped_to_gpu_tags() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu_tag = "socket-0"
            to_gpu_tags = ["decode-fast"]
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            intra = "pcie_gen5"
            gpus = [
              { start_id = 0, gpu = "h100_sxm", count = 2, labels = ["socket-0", "fast-nic"] },
              { start_id = 2, gpu = "h100_sxm", count = 2, labels = ["socket-1"] },
            ]
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }

            [[nodes]]
            id = 1
            intra = "pcie_gen5"
            gpus = [
              { id = 0, gpu = "a100_80gb", labels = ["decode-slow"] },
              { start_id = 1, gpu = "a100_80gb", count = 2, labels = ["decode-fast"] },
            ]
            nics = { count = 4, affinity = "dedicated", rail_count = 4 }
            "#,
    )
    .unwrap();

    let node = cluster.node(0).unwrap();
    assert!(node.gpu_labels(0).unwrap().contains("socket_0"));
    assert!(node.gpu_labels(0).unwrap().contains("fast_nic"));
    assert!(node.gpu_labels(2).unwrap().contains("socket_1"));

    let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
        panic!("expected custom inter-node topology");
    };
    let links = edges
        .get(&UnorderedPair::new(0, 1))
        .expect("missing custom link");
    assert_eq!(links.len(), 1);
    let endpoints = links[0].endpoints.as_ref().unwrap();
    assert_eq!(endpoints.from_node, 0);
    assert_eq!(endpoints.from_gpus, vec![0, 1]);
    assert_eq!(endpoints.to_node, 1);
    assert_eq!(endpoints.to_gpus, vec![1, 2]);
}

#[test]
fn parses_node_topology_metadata_and_domain_scoped_links() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from_rack = "rack-a"
            from_node_tag = "fast-prefill"
            to_failure_domain = "az-b"
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            group = "prefill"
            node_tags = ["fast-prefill", "power-a"]
            rack = "rack-a"
            island = "island-a"
            failure_domain = "az-a"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", rail_count = 2 }

            [[nodes]]
            id = 1
            group = "prefill"
            node_tag = "slow-prefill"
            rack = "rack-a"
            island = "island-a"
            failure_domain = "az-a"
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", rail_count = 2 }

            [[nodes]]
            id = 2
            group = "decode"
            node_tag = "decode"
            rack = "rack-b"
            island = "island-b"
            failure_domain = "az-b"
            gpu = "a100_80gb"
            gpu_count = 2
            nics = { count = 2, affinity = "dedicated", rail_count = 2 }
            "#,
    )
    .unwrap();

    let prefill = cluster.node(0).unwrap();
    assert!(prefill.topology.labels.contains("fast_prefill"));
    assert!(prefill.topology.labels.contains("power_a"));
    assert_eq!(prefill.topology.rack.as_deref(), Some("rack_a"));
    assert_eq!(prefill.topology.island.as_deref(), Some("island_a"));
    assert_eq!(prefill.topology.failure_domain.as_deref(), Some("az_a"));

    let InterNodeTopology::Custom(edges) = &cluster.inter_node_topology else {
        panic!("expected custom inter-node topology");
    };
    assert!(edges.contains_key(&UnorderedPair::new(0, 2)));
    assert!(!edges.contains_key(&UnorderedPair::new(1, 2)));
}

#[test]
fn rejects_unsupported_schema_versions() {
    let cluster_err = parse_cluster(
        r#"
            schema_version = 2

            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
    )
    .unwrap_err();
    assert!(
        cluster_err
            .to_string()
            .contains("unsupported cluster schema_version 2")
    );

    let workload_err = parse_workload(
        r#"
            schema_version = 2

            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
    )
    .unwrap_err();
    assert!(
        workload_err
            .to_string()
            .contains("unsupported workload schema_version 2")
    );

    let run_err = parse_run_config(
        r#"
            schema_version = 2
            cluster = "cluster.toml"
            workload = "workload.toml"
            "#,
    )
    .unwrap_err();
    assert!(
        run_err
            .to_string()
            .contains("unsupported run schema_version 2")
    );
}

#[test]
fn parses_run_config() {
    let run = parse_run_config(
        r#"
            schema_version = 1

            [run]
            cluster = "cluster.toml"
            workload = "workload.toml"

            [output]
            format = "json"
            top_k = 3
            output_dir = "artifacts/run"
            output_profile = "calibration"
            request_metrics_csv = "artifacts/request_metrics.csv"
            request_lifecycle_events_csv = "artifacts/request_lifecycle_events.csv"
            serving_metrics_csv = "artifacts/serving_metrics.csv"
            serving_metric_breakdowns_csv = "artifacts/serving_metric_breakdowns.csv"
            serving_services_csv = "artifacts/serving_services.csv"
            serving_utilization_csv = "artifacts/serving_utilization.csv"
            serving_memory_pressure_csv = "artifacts/serving_memory_pressure.csv"
            serving_timeline_csv = "artifacts/serving_timeline.csv"
            serving_occupancy_csv = "artifacts/serving_occupancy.csv"
            serving_placement_evidence_csv = "artifacts/serving_placement_evidence.csv"
            serving_worker_evidence_csv = "artifacts/serving_worker_evidence.csv"
            serving_rejections_csv = "artifacts/serving_rejections.csv"
            serving_route_paths_csv = "artifacts/serving_route_paths.csv"
            kv_route_resources_csv = "artifacts/kv_route_resources.csv"
            serving_bottlenecks_csv = "artifacts/serving_bottlenecks.csv"
            serving_phase_calibration_csv = "artifacts/serving_phase_calibration.csv"
            serving_approximations_csv = "artifacts/serving_approximations.csv"
            calibration_residuals_csv = "artifacts/calibration_residuals.csv"
            scenario_sensitivity_csv = "artifacts/scenario_sensitivity.csv"
            rank_sensitivity_csv = "artifacts/rank_sensitivity.csv"
            trace = true
            trace_limit = 0
            request_limit = 12
            occupancy = true
            occupancy_buckets = 8
            occupancy_resource_limit = 0
            critical_path = true
            critical_path_limit = 5

            [search]
            max_parallelism_candidates = 11
            max_prefill_candidates = 3
            max_decode_candidates = 4
            max_serving_pairs = 7
            max_runtime_ms = 250
            retain_rejected_candidates = false

            [[scenarios]]
            name = "baseline"
            request_count = 4
            arrival_rate_scale = 1.0

            [[scenarios]]
            name = "burst"
            requests = 8
            qps_scale = 2.0
            prompt_tokens_scale = 1.5
            decode_tokens_scale = 0.5
            calibration_profile = "scenario-profile.toml"

            [scenarios.calibration]
            decode_compute_scale = 1.4
            kv_transfer_scale = 1.2
            allow_compute_comm_overlap = false

            [scenarios.topology]
            interconnect_bandwidth_scale = 0.5
            interconnect_latency_scale = 2.0
            nic_bandwidth_scale = 0.75

            [[scenarios.topology.node_states]]
            group = "spare"
            node_tag = "warm-spare"
            rack = "rack-spare"
            state = "maintenance"

            [[scenarios.topology.disabled_gpus]]
            nodes = [0]
            node_tag = "gpu-maint"
            failure_domain = "az-a"
            gpus = [1, 2]

            [[scenarios.topology.disabled_nics]]
            group = "h100"
            rack = "rack-a"
            nics = [0]

            [[scenarios.topology.degraded_gpus]]
            node = 0
            island = "island-a"
            gpu = 3
            compute_scale = 0.5
            hbm_bandwidth_scale = 0.75
            hbm_capacity_scale = 0.9

            [[scenarios.topology.degraded_nics]]
            nodes = [0]
            failure_domain = "az-a"
            nics = [1]
            bandwidth_scale = 0.25
            latency_scale = 1.5

            [[scenarios.topology.degraded_rails]]
            groups = ["h100"]
            node_tag = "rail-scope"
            rails = [2]
            bandwidth_scale = 0.6
            latency_scale = 1.25

            [[scenarios.topology.degraded_links]]
            from = 0
            from_node_tag = "fast-prefill"
            from_rack = "rack-a"
            to_group = "h100"
            to_node_tag = "decode-pool"
            to_island = "island-b"
            to_failure_domain = "az-b"
            from_gpu = 0
            to_gpus = [1, 2]
            rails = [0, 1]
            bandwidth_scale = 0.4
            latency_scale = 1.5
            "#,
    )
    .unwrap();

    assert_eq!(run.cluster_path, Some(PathBuf::from("cluster.toml")));
    assert_eq!(run.workload_path, Some(PathBuf::from("workload.toml")));
    assert_eq!(run.output.format.as_deref(), Some("json"));
    assert_eq!(run.output.top_k, Some(3));
    assert_eq!(
        run.output.output_dir.as_deref(),
        Some(Path::new("artifacts/run"))
    );
    assert_eq!(run.output.output_profile.as_deref(), Some("calibration"));
    assert_eq!(
        run.output.request_metrics_csv_path.as_deref(),
        Some(Path::new("artifacts/request_metrics.csv"))
    );
    assert_eq!(
        run.output.request_lifecycle_events_csv_path.as_deref(),
        Some(Path::new("artifacts/request_lifecycle_events.csv"))
    );
    assert_eq!(
        run.output.serving_metrics_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_metrics.csv"))
    );
    assert_eq!(
        run.output.serving_metric_breakdowns_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_metric_breakdowns.csv"))
    );
    assert_eq!(
        run.output.serving_services_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_services.csv"))
    );
    assert_eq!(
        run.output.serving_utilization_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_utilization.csv"))
    );
    assert_eq!(
        run.output.serving_memory_pressure_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_memory_pressure.csv"))
    );
    assert_eq!(
        run.output.serving_timeline_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_timeline.csv"))
    );
    assert_eq!(
        run.output.serving_occupancy_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_occupancy.csv"))
    );
    assert_eq!(
        run.output.serving_placement_evidence_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_placement_evidence.csv"))
    );
    assert_eq!(
        run.output.serving_worker_evidence_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_worker_evidence.csv"))
    );
    assert_eq!(
        run.output.serving_rejections_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_rejections.csv"))
    );
    assert_eq!(
        run.output.serving_route_paths_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_route_paths.csv"))
    );
    assert_eq!(
        run.output.kv_route_resources_csv_path.as_deref(),
        Some(Path::new("artifacts/kv_route_resources.csv"))
    );
    assert_eq!(
        run.output.serving_bottlenecks_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_bottlenecks.csv"))
    );
    assert_eq!(
        run.output.serving_phase_calibration_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_phase_calibration.csv"))
    );
    assert_eq!(
        run.output.serving_approximations_csv_path.as_deref(),
        Some(Path::new("artifacts/serving_approximations.csv"))
    );
    assert_eq!(
        run.output.calibration_residuals_csv_path.as_deref(),
        Some(Path::new("artifacts/calibration_residuals.csv"))
    );
    assert_eq!(
        run.output.scenario_sensitivity_csv_path.as_deref(),
        Some(Path::new("artifacts/scenario_sensitivity.csv"))
    );
    assert_eq!(
        run.output.rank_sensitivity_csv_path.as_deref(),
        Some(Path::new("artifacts/rank_sensitivity.csv"))
    );
    assert_eq!(run.output.trace, Some(true));
    assert_eq!(run.output.trace_limit, Some(0));
    assert_eq!(run.output.request_limit, Some(12));
    assert_eq!(run.output.occupancy, Some(true));
    assert_eq!(run.output.occupancy_buckets, Some(8));
    assert_eq!(run.output.occupancy_resource_limit, Some(0));
    assert_eq!(run.output.critical_path, Some(true));
    assert_eq!(run.output.critical_path_limit, Some(5));
    assert_eq!(run.search_budget.max_parallelism_candidates, Some(11));
    assert_eq!(run.search_budget.max_prefill_candidates, Some(3));
    assert_eq!(run.search_budget.max_decode_candidates, Some(4));
    assert_eq!(run.search_budget.max_serving_pairs, Some(7));
    assert_eq!(run.search_budget.max_runtime_ms, Some(250));
    assert_eq!(run.search_budget.retain_rejected_candidates, Some(false));
    assert_eq!(run.scenarios.len(), 2);
    assert_eq!(run.scenarios[0].name, "baseline");
    assert_eq!(run.scenarios[0].request_count, Some(4));
    assert_eq!(run.scenarios[0].arrival_rate_scale, Some(1.0));
    assert_eq!(run.scenarios[1].name, "burst");
    assert_eq!(run.scenarios[1].request_count, Some(8));
    assert_eq!(run.scenarios[1].arrival_rate_scale, Some(2.0));
    assert_eq!(run.scenarios[1].prompt_tokens_scale, Some(1.5));
    assert_eq!(run.scenarios[1].decode_tokens_scale, Some(0.5));
    assert_eq!(
        run.scenarios[1].calibration_profile_path.as_deref(),
        Some(Path::new("scenario-profile.toml"))
    );
    assert_eq!(run.scenarios[1].calibration.decode_compute_scale, Some(1.4));
    assert_eq!(run.scenarios[1].calibration.kv_transfer_scale, Some(1.2));
    assert_eq!(
        run.scenarios[1].calibration.allow_compute_comm_overlap,
        Some(false)
    );
    assert_eq!(
        run.scenarios[1].topology.interconnect_bandwidth_scale,
        Some(0.5)
    );
    assert_eq!(
        run.scenarios[1].topology.interconnect_latency_scale,
        Some(2.0)
    );
    assert_eq!(run.scenarios[1].topology.nic_bandwidth_scale, Some(0.75));
    assert_eq!(run.scenarios[1].topology.node_states.len(), 1);
    assert_eq!(
        run.scenarios[1].topology.node_states[0].node_groups,
        vec!["spare".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.node_states[0].node_tags,
        vec!["warm_spare".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.node_states[0].racks,
        vec!["rack_spare".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.node_states[0].state,
        RunScenarioNodeState::Maintenance
    );
    assert_eq!(run.scenarios[1].topology.disabled_gpus.len(), 1);
    assert_eq!(run.scenarios[1].topology.disabled_gpus[0].node_ids, vec![0]);
    assert_eq!(
        run.scenarios[1].topology.disabled_gpus[0].node_tags,
        vec!["gpu_maint".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.disabled_gpus[0].failure_domains,
        vec!["az_a".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.disabled_gpus[0].gpu_ids,
        vec![1, 2]
    );
    assert_eq!(run.scenarios[1].topology.disabled_nics.len(), 1);
    assert_eq!(
        run.scenarios[1].topology.disabled_nics[0].node_groups,
        vec!["h100".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.disabled_nics[0].racks,
        vec!["rack_a".to_string()]
    );
    assert_eq!(run.scenarios[1].topology.disabled_nics[0].nic_ids, vec![0]);
    assert_eq!(run.scenarios[1].topology.degraded_gpus.len(), 1);
    assert_eq!(run.scenarios[1].topology.degraded_gpus[0].node_ids, vec![0]);
    assert_eq!(
        run.scenarios[1].topology.degraded_gpus[0].islands,
        vec!["island_a".to_string()]
    );
    assert_eq!(run.scenarios[1].topology.degraded_gpus[0].gpu_ids, vec![3]);
    assert_eq!(
        run.scenarios[1].topology.degraded_gpus[0].compute_scale,
        Some(0.5)
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_gpus[0].hbm_bandwidth_scale,
        Some(0.75)
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_gpus[0].hbm_capacity_scale,
        Some(0.9)
    );
    assert_eq!(run.scenarios[1].topology.degraded_nics.len(), 1);
    assert_eq!(run.scenarios[1].topology.degraded_nics[0].node_ids, vec![0]);
    assert_eq!(
        run.scenarios[1].topology.degraded_nics[0].failure_domains,
        vec!["az_a".to_string()]
    );
    assert_eq!(run.scenarios[1].topology.degraded_nics[0].nic_ids, vec![1]);
    assert_eq!(
        run.scenarios[1].topology.degraded_nics[0].bandwidth_scale,
        Some(0.25)
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_nics[0].latency_scale,
        Some(1.5)
    );
    assert_eq!(run.scenarios[1].topology.degraded_rails.len(), 1);
    assert_eq!(
        run.scenarios[1].topology.degraded_rails[0].node_groups,
        vec!["h100".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_rails[0].node_tags,
        vec!["rail_scope".to_string()]
    );
    assert_eq!(run.scenarios[1].topology.degraded_rails[0].rails, vec![2]);
    assert_eq!(
        run.scenarios[1].topology.degraded_rails[0].bandwidth_scale,
        Some(0.6)
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_rails[0].latency_scale,
        Some(1.25)
    );
    assert_eq!(run.scenarios[1].topology.degraded_links.len(), 1);
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].from_node_ids,
        vec![0]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].from_node_tags,
        vec!["fast_prefill".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].from_racks,
        vec!["rack_a".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].to_node_groups,
        vec!["h100".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].to_node_tags,
        vec!["decode_pool".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].to_islands,
        vec!["island_b".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].to_failure_domains,
        vec!["az_b".to_string()]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].from_gpus,
        vec![0]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].to_gpus,
        vec![1, 2]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].rails,
        vec![0, 1]
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].bandwidth_scale,
        Some(0.4)
    );
    assert_eq!(
        run.scenarios[1].topology.degraded_links[0].latency_scale,
        Some(1.5)
    );
}

#[test]
fn rejects_zero_run_search_budget() {
    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [search]
            max_serving_pairs = 0
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("search.max_serving_pairs must be greater than zero")
    );
}

#[test]
fn rejects_invalid_run_scenarios() {
    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "burst"
            arrival_rate_scale = 0.0
            "#,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("scenarios[0].arrival_rate_scale must be finite and greater than zero")
    );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "dup"

            [[scenarios]]
            name = "dup"
            "#,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("scenarios[1].name 'dup' is duplicated")
    );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-calibration"

            [scenarios.calibration]
            compute_efficiency = 1.1
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
            "scenarios[0].calibration.compute_efficiency must be finite and greater than 0.0 and less than or equal to 1.0"
        ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-topology"

            [scenarios.topology]
            interconnect_bandwidth_scale = 0.0
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
        "scenarios[0].topology.interconnect_bandwidth_scale must be finite and greater than zero"
    ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            disabled_gpus = [{ gpu = 0 }]
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
            "scenarios[0].topology.disabled_gpus[0] must specify at least one node, nodes, group, groups, node_tag, rack, island, or failure_domain selector"
        ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            disabled_nics = [{ node = 0 }]
            "#,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains(
            "scenarios[0].topology.disabled_nics[0] must specify at least one resource id"
        )
    );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            disabled_gpus = [{ node = 0, nodes = [0], gpu = 1 }]
            "#,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("scenarios[0].topology.disabled_gpus[0].nodes duplicates id 0")
    );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            degraded_nics = [{ node = 0, nic = 0 }]
            "#,
    )
    .unwrap_err();
    assert!(
            err.to_string()
                .contains("scenarios[0].topology.degraded_nics[0] must specify at least one of bandwidth_scale or latency_scale")
        );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            degraded_rails = [{ rail = 0 }]
            "#,
    )
    .unwrap_err();
    assert!(
            err.to_string()
                .contains("scenarios[0].topology.degraded_rails[0] must specify at least one of bandwidth_scale or latency_scale")
        );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            degraded_rails = [{ bandwidth_scale = 0.5 }]
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
        "scenarios[0].topology.degraded_rails[0] must specify at least one rail or rails selector"
    ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            node_states = [{ node = 0 }]
            "#,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("scenarios[0].topology.node_states[0].state is required")
    );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            node_states = [{ state = "maintenance" }]
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
            "scenarios[0].topology.node_states[0] must specify at least one node, nodes, group, groups, node_tag, rack, island, or failure_domain selector"
        ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            node_states = [{ node = 0, state = "quarantined" }]
            "#,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("scenarios[0].topology.node_states[0].state 'quarantined' is unsupported")
    );

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            degraded_gpus = [{ node = 0, gpu = 0 }]
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
            "scenarios[0].topology.degraded_gpus[0] must specify at least one of compute_scale, hbm_bandwidth_scale, or hbm_capacity_scale"
        ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            degraded_nics = [{ node = 0, nic = 0, bandwidth_scale = 0.0 }]
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
            "scenarios[0].topology.degraded_nics[0].bandwidth_scale must be finite and greater than zero"
        ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            degraded_links = [{ from = 0, to = 1 }]
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
            "scenarios[0].topology.degraded_links[0] must specify at least one of bandwidth_scale or latency_scale"
        ));

    let err = parse_run_config(
        r#"
            schema_version = 1
            cluster = "cluster.toml"
            workload = "workload.toml"

            [[scenarios]]
            name = "invalid-resource-overlay"

            [scenarios.topology]
            degraded_links = [{ to = 1, bandwidth_scale = 0.5 }]
            "#,
    )
    .unwrap_err();
    assert!(err.to_string().contains(
            "scenarios[0].topology.degraded_links[0] must specify at least one from, from_nodes, from_group, from_groups, from_node_tag, from_rack, from_island, or from_failure_domain selector"
        ));
}

#[test]
fn rejects_custom_interconnect_links_to_unknown_nodes() {
    let err = parse_cluster(
        r#"
            schema_version = 1

            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 7
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 8
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("interconnect.links[].to references unknown node id 7")
    );

    let err = parse_cluster(
        r#"
            schema_version = 1

            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            from_gpu = 7
            kind = "ib"
            variant = "ndr"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("interconnect.links[].from_gpus references node 0 local GPU id 7")
    );
}

#[test]
fn rejects_invalid_gpu_profile_overrides() {
    let unknown_gpu_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            gpu_profile_overrides = [{ gpu = 2, hbm_gb = 40.0 }]
            "#,
    )
    .unwrap_err();
    assert!(
        unknown_gpu_err
            .to_string()
            .contains("nodes[0].gpu_profile_overrides[0] references unknown local GPU id 2")
    );

    let duplicate_gpu_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            gpu_profile_overrides = [
              { gpu = 1, hbm_gb = 40.0 },
              { gpus = [1], peak_f16_tflops = 500.0 },
            ]
            "#,
    )
    .unwrap_err();
    assert!(
        duplicate_gpu_err
            .to_string()
            .contains("nodes[0].gpu_profile_overrides duplicates local GPU id 1")
    );

    let missing_profile_field_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            gpu_profile_overrides = [{ gpu = 1 }]
            "#,
    )
    .unwrap_err();
    assert!(
        missing_profile_field_err.to_string().contains(
            "nodes[0].gpu_profile_overrides[0] must specify at least one profile override"
        )
    );

    let invalid_value_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            gpu_profile_overrides = [{ gpu = 1, peak_f16_tflops = 0.0 }]
            "#,
    )
    .unwrap_err();
    assert!(invalid_value_err.to_string().contains(
        "nodes[0].gpu_profile_overrides[0].peak_f16_tflops must be finite and greater than zero"
    ));
}

#[test]
fn rejects_invalid_nic_rail_and_affinity_settings() {
    let rail_err = parse_cluster(
        r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"

            [nics]
            count = 4
            rail_count = 5
            affinity = "uniform"
            "#,
    )
    .unwrap_err();
    assert!(
        rail_err
            .to_string()
            .contains("nics.rail_count must be less than or equal to nics.count")
    );

    let affinity_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 8
            nics = { count = 4, affinity = "dedicated" }
            "#,
    )
    .unwrap_err();
    assert!(
        affinity_err
            .to_string()
            .contains("dedicated affinity requires nics.count >= gpu_count (8)")
    );

    let mixed_without_intra_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            nics = { count = 2, affinity = "uniform" }
            gpus = [
              { gpu = "h100_sxm" },
              { gpu = "a100_80gb" },
            ]
            "#,
    )
    .unwrap_err();
    assert!(
        mixed_without_intra_err
            .to_string()
            .contains("nodes[0].intra is required when nodes[0].gpus mixes GPU types")
    );

    let duplicate_gpu_id_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform" }
            gpus = [
              { id = 0, gpu = "h100_sxm" },
              { id = 0, gpu = "h100_sxm" },
            ]
            "#,
    )
    .unwrap_err();
    assert!(
        duplicate_gpu_id_err
            .to_string()
            .contains("nodes[0].gpus[1] duplicates local GPU id 0")
    );

    let invalid_link_rails_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "hdr"
            rail = 0
            rails = [1]

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 1
            "#,
    )
    .unwrap_err();
    assert!(
        invalid_link_rails_err
            .to_string()
            .contains("interconnect.links[].rail and rails cannot both be set")
    );

    let invalid_gpu_nic_map_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "uniform", gpu_nic_map = [{ gpu = 1, nics = [2] }] }
            "#,
    )
    .unwrap_err();
    assert!(
        invalid_gpu_nic_map_err
            .to_string()
            .contains("nodes[0].nics.gpu_nic_map for local GPU id 1 references NIC 2")
    );

    let invalid_nic_rail_map_err = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, rail_count = 2, affinity = "uniform", nic_rail_map = [{ nic = 1, rail = 2 }] }
            "#,
        )
        .unwrap_err();
    assert!(
        invalid_nic_rail_map_err
            .to_string()
            .contains("nodes[0].nics.nic_rail_map[0] references rail 2")
    );

    let invalid_gpu_nic_path_err = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "uniform", gpu_nic_paths = [{ gpu = 7, nic = 1, bandwidth_gbps = 100.0 }] }
            "#,
        )
        .unwrap_err();
    assert!(
        invalid_gpu_nic_path_err
            .to_string()
            .contains("nodes[0].nics.gpu_nic_paths references unknown local GPU id 7")
    );

    let invalid_nic_bandwidth_override_err = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "uniform", nic_bandwidth_overrides = [{ nic = 2, bandwidth_gbps = 100.0 }] }
            "#,
        )
        .unwrap_err();
    assert!(
        invalid_nic_bandwidth_override_err
            .to_string()
            .contains("nodes[0].nics.nic_bandwidth_overrides[0] references NIC 2")
    );

    let invalid_nic_latency_override_err = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "uniform", nic_latency_scale_overrides = [{ nic = 1, latency_scale = 0.0 }] }
            "#,
        )
        .unwrap_err();
    assert!(invalid_nic_latency_override_err.to_string().contains(
        "nodes[0].nics.nic_latency_scale_overrides[0].latency_scale must be finite and positive"
    ));

    let invalid_gpu_numa_map_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "uniform", gpu_numa_map = [{ gpu = 2, domain = 0 }] }
            "#,
    )
    .unwrap_err();
    assert!(
        invalid_gpu_numa_map_err
            .to_string()
            .contains("nodes[0].nics.gpu_numa_map references unknown local GPU id 2")
    );

    let invalid_nic_numa_map_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "uniform", nic_numa_map = [{ nic = 2, domain = 0 }] }
            "#,
    )
    .unwrap_err();
    assert!(
        invalid_nic_numa_map_err
            .to_string()
            .contains("nodes[0].nics.nic_numa_map[0] references NIC 2")
    );

    let invalid_cross_numa_scale_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            nics = { count = 2, affinity = "uniform", cross_numa_bandwidth_scale = 0.0 }
            "#,
    )
    .unwrap_err();
    assert!(
        invalid_cross_numa_scale_err.to_string().contains(
            "nodes[0].nics.cross_numa_bandwidth_scale must be finite and greater than zero"
        )
    );

    let disabled_gpu_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            disabled_gpus = [3]
            "#,
    )
    .unwrap_err();
    assert!(
        disabled_gpu_err
            .to_string()
            .contains("nodes[0].disabled_gpus references unknown local GPU id 3")
    );

    let disabled_nic_err = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", disabled_nics = [0] }
            "#,
    )
    .unwrap_err();
    assert!(
        disabled_nic_err
            .to_string()
            .contains("nodes[0].nics leaves local GPU id 0 without an enabled NIC path")
    );
}

#[test]
fn validates_serving_pools_against_loaded_cluster() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [9]
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(
        err.to_string()
            .contains("serving.pool_candidates[0].decode_nodes references unknown node id 9")
    );
}

#[test]
fn optional_routable_pool_validation_rejects_disconnected_serving_pools() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            require_routable_pools = true
            prefill_nodes = [0]
            decode_nodes = [1]
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(err.to_string().contains("has no routable KV transfer path"));

    let mut permissive = workload.clone();
    permissive.require_routable_serving_pools = false;
    validate_workload_for_cluster(&cluster, &permissive).unwrap();
}

#[test]
fn validates_kv_route_constraints_against_configured_pools() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1 }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1 }
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            min_kv_route_rail_count = 2
            prefill_nodes = [0]
            decode_nodes = [1]
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(err.to_string().contains(
        "no configured prefill/decode pool candidate satisfying configured KV route constraints"
    ));
    assert!(
        err.to_string()
            .contains("below configured serving.min_kv_route_rail_count 2")
    );

    let mut valid = workload;
    valid
        .serving
        .as_mut()
        .unwrap()
        .traffic
        .kv_route_constraints
        .min_inter_node_rail_count = Some(1);
    validate_workload_for_cluster(&cluster, &valid).unwrap();
}

#[test]
fn validates_gpudirect_kv_route_constraint_against_gpu_nic_paths() {
    let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 1
            intra = "pcie_gen5"
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1, gpu_nic_paths = [{ gpu = 0, nic = 0, label = "host_staged", gpudirect = false }] }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 1
            intra = "pcie_gen5"
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 1, gpu_nic_paths = [{ gpu = 0, nic = 0, label = "host_staged", gpudirect = false }] }
            "#,
        )
        .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            require_gpudirect_kv_paths = true
            prefill_nodes = [0]
            decode_nodes = [1]
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(
        err.to_string()
            .contains("host-staged or marked no-GPUDirect")
    );
}

#[test]
fn validates_kv_route_constraints_against_explicit_serving_placements() {
    let cluster = parse_cluster(
            r#"
            [cluster]
            preset = "custom"

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 0

            [[interconnect.links]]
            from = 0
            to = 1
            kind = "ib"
            variant = "ndr"
            rail = 1

            [[nodes]]
            id = 0
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 2, gpu_nic_map = [{ gpu = 0, nic = 0 }, { gpu = 1, nic = 1 }], gpu_nic_paths = [{ gpu = 0, nic = 0, label = "host_staged", gpudirect = false }, { gpu = 1, nic = 1, label = "gpudirect", gpudirect = true }] }

            [[nodes]]
            id = 1
            gpu = "h100_sxm"
            gpu_count = 2
            intra = "pcie_gen5"
            nics = { count = 2, affinity = "uniform", bandwidth_gbps = 400.0, rail_count = 2, gpu_nic_map = [{ gpu = 0, nic = 0 }, { gpu = 1, nic = 1 }], gpu_nic_paths = [{ gpu = 0, nic = 0, label = "host_staged", gpudirect = false }, { gpu = 1, nic = 1, label = "gpudirect", gpudirect = true }] }
            "#,
        )
        .unwrap();

    let direct = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            require_gpudirect_kv_paths = true
            prefill_nodes = [0]
            decode_nodes = [1]

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 0
            gpu = 1

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            gpu = 1
            "#,
    )
    .unwrap();
    validate_workload_for_cluster(&cluster, &direct).unwrap();

    let host_staged = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            require_gpudirect_kv_paths = true
            prefill_nodes = [0]
            decode_nodes = [1]

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 0
            gpu = 0

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            gpu = 0
            "#,
    )
    .unwrap();
    let err = validate_workload_for_cluster(&cluster, &host_staged).unwrap_err();
    assert!(
        err.to_string()
            .contains("host-staged or marked no-GPUDirect")
    );
}

#[test]
fn validates_explicit_serving_placements_fit_static_pools() {
    let cluster = Cluster::h100_sxm_nodes(2, IbVariant::Ndr.default_profile());
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 1
            gpu = 0

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            gpu = 1
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(err.to_string().contains(
        "serving explicit prefill/decode placements do not fit any configured serving pool"
    ));

    let valid = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 0
            gpu = 0

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            gpu = 1
            "#,
    )
    .unwrap();
    validate_workload_for_cluster(&cluster, &valid).unwrap();
}

#[test]
fn validates_explicit_serving_placements_fit_pool_search_candidates() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[node_groups]]
            label = "prefill"
            start_id = 0
            count = 1
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }

            [[node_groups]]
            label = "decode"
            start_id = 1
            count = 1
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"

            [serving.pool_search]
            prefill_groups = ["prefill"]
            decode_groups = ["decode"]
            prefill_node_counts = [1]
            decode_node_counts = [1]
            allow_overlap = false

            [[serving.prefill_placement.ranks]]
            rank = 0
            node = 1
            gpu = 0

            [[serving.decode_placement.ranks]]
            rank = 0
            node = 1
            gpu = 0
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(err.to_string().contains(
        "serving explicit prefill/decode placements do not fit any configured serving pool"
    ));
}

#[test]
fn optional_routable_pool_validation_checks_pool_search_candidates() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[node_groups]]
            label = "prefill"
            start_id = 0
            count = 1
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }

            [[node_groups]]
            label = "decode"
            start_id = 1
            count = 1
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "uniform", bandwidth_gbps = 400.0 }
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            require_routable_pools = true

            [serving.pool_search]
            prefill_groups = ["prefill"]
            decode_groups = ["decode"]
            prefill_node_counts = [1]
            decode_node_counts = [1]
            allow_overlap = false
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(err.to_string().contains(
        "serving.pool_search cannot generate any routable prefill/decode pool candidate"
    ));
}

#[test]
fn validates_pool_search_can_generate_a_candidate() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 1

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"

            [serving.pool_search]
            prefill_groups = ["all"]
            decode_groups = ["all"]
            prefill_node_counts = [1]
            decode_node_counts = [1]
            allow_overlap = false
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &workload).unwrap_err();
    assert!(
        err.to_string()
            .contains("serving.pool_search cannot generate any prefill/decode pool candidate")
    );
}

#[test]
fn validates_pool_search_topology_node_filters() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            group = "prefill"
            node_tag = "fast-prefill"
            rack = "rack-a"
            failure_domain = "az-a"
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }

            [[nodes]]
            id = 1
            group = "prefill"
            node_tag = "slow-prefill"
            rack = "rack-b"
            failure_domain = "az-a"
            gpu = "a100_80gb"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }

            [[nodes]]
            id = 2
            group = "decode"
            node_tag = "decode-east"
            rack = "rack-c"
            failure_domain = "az-b"
            gpu = "a100_80gb"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"

            [serving.pool_search]
            prefill_groups = ["prefill"]
            decode_groups = ["decode"]
            prefill_node_counts = [1]
            decode_node_counts = [1]
            prefill_node_tag = "fast-prefill"
            prefill_exclude_rack = "rack-b"
            decode_failure_domain = "az-b"
            decode_exclude_node_tag = "decode-unused"
            "#,
    )
    .unwrap();

    let pool_search = workload
        .serving
        .as_ref()
        .unwrap()
        .pool_search
        .as_ref()
        .unwrap();
    assert_eq!(
        pool_search.prefill_node_filter.node_labels,
        vec!["fast_prefill"]
    );
    assert_eq!(
        pool_search.prefill_node_filter.exclude_racks,
        vec!["rack_b"]
    );
    assert_eq!(pool_search.decode_node_filter.failure_domains, vec!["az_b"]);
    assert_eq!(
        pool_search.decode_node_filter.exclude_node_labels,
        vec!["decode_unused"]
    );
    validate_workload_for_cluster(&cluster, &workload).unwrap();
}

#[test]
fn validates_pool_candidate_topology_node_filters() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "custom"

            [[nodes]]
            id = 0
            group = "prefill"
            node_tag = "fast-prefill"
            rack = "rack-a"
            failure_domain = "az-a"
            gpu = "h100_sxm"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }

            [[nodes]]
            id = 1
            group = "prefill"
            node_tag = "slow-prefill"
            rack = "rack-b"
            failure_domain = "az-a"
            gpu = "a100_80gb"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }

            [[nodes]]
            id = 2
            group = "decode"
            node_tag = "decode-east"
            rack = "rack-c"
            failure_domain = "az-b"
            gpu = "a100_80gb"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }

            [[nodes]]
            id = 3
            group = "decode"
            node_tag = "decode-west"
            rack = "rack-d"
            failure_domain = "az-c"
            gpu = "a100_80gb"
            gpu_count = 1
            nics = { count = 1, affinity = "dedicated", rail_count = 1 }
            "#,
    )
    .unwrap();
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"

            [[serving.pool_candidates]]
            label = "filtered"
            prefill_groups = ["prefill"]
            decode_groups = ["decode"]
            prefill_node_tag = "fast-prefill"
            prefill_rack = "rack-a"
            prefill_exclude_rack = "rack-b"
            decode_failure_domain = "az-b"
            decode_exclude_failure_domain = "az-c"
            min_prefill_racks = 1
            min_decode_failure_domains = 1
            "#,
    )
    .unwrap();

    let candidate = &workload.serving.as_ref().unwrap().pool_candidates[0];
    assert_eq!(
        candidate.prefill_node_filter.node_labels,
        vec!["fast_prefill"]
    );
    assert_eq!(candidate.prefill_node_filter.racks, vec!["rack_a"]);
    assert_eq!(candidate.prefill_node_filter.exclude_racks, vec!["rack_b"]);
    assert_eq!(candidate.decode_node_filter.failure_domains, vec!["az_b"]);
    assert_eq!(
        candidate.decode_node_filter.exclude_failure_domains,
        vec!["az_c"]
    );
    assert_eq!(candidate.domain_spread.min_prefill_racks, Some(1));
    assert_eq!(candidate.domain_spread.min_decode_failure_domains, Some(1));
    validate_workload_for_cluster(&cluster, &workload).unwrap();

    let invalid = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"

            [[serving.pool_candidates]]
            label = "too-concentrated"
            prefill_groups = ["prefill"]
            decode_groups = ["decode"]
            min_prefill_failure_domains = 2
            "#,
    )
    .unwrap();

    let err = validate_workload_for_cluster(&cluster, &invalid).unwrap_err();
    assert!(
        err.to_string()
            .contains("does not satisfy configured topology-domain spread constraints")
    );
}

#[test]
fn validates_serving_deployment_modes() {
    let cluster = parse_cluster(
        r#"
            [cluster]
            preset = "h100_sxm"
            node_count = 2

            [interconnect]
            kind = "ib"
            variant = "ndr"
            "#,
    )
    .unwrap();

    let colocated = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "colocated"

            [serving.pool_search]
            prefill_groups = ["all"]
            decode_groups = ["all"]
            prefill_node_counts = [1]
            decode_node_counts = [1]
            "#,
    )
    .unwrap();
    let colocated_serving = colocated.serving.as_ref().unwrap();
    assert_eq!(
        colocated_serving.deployment_mode,
        ServingDeploymentMode::Colocated
    );
    assert!(
        colocated_serving
            .pool_search
            .as_ref()
            .unwrap()
            .allow_overlap
    );
    validate_workload_for_cluster(&cluster, &colocated).unwrap();

    let partial = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "partially_disaggregated"
            prefill_nodes = [0, 1]
            decode_nodes = [1]
            "#,
    )
    .unwrap();
    assert_eq!(
        partial.serving.as_ref().unwrap().deployment_mode,
        ServingDeploymentMode::PartiallyDisaggregated
    );
    validate_workload_for_cluster(&cluster, &partial).unwrap();

    let invalid = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "fully_disaggregated"
            prefill_nodes = [0]
            decode_nodes = [0]
            "#,
    )
    .unwrap();
    let err = validate_workload_for_cluster(&cluster, &invalid).unwrap_err();
    assert!(
        err.to_string()
            .contains("serving.pool_candidates[0] uses colocated prefill/decode nodes")
    );
}

#[test]
fn parses_serving_and_calibration_config() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 80
            hidden_size = 8192
            attention_heads = 64
            kv_heads = 8
            vocab_size = 128256
            parameters_gb = 140.0
            dtype = "bf16"

            [request]
            batch_size = 4
            prompt_tokens = 1024
            decode_tokens = 128
            max_sequence_tokens = 2048
            phase = "end_to_end"

            [calibration]
            compute_efficiency = 0.42
            decode_memory_bandwidth_scale = 0.75
            kv_transfer_scale = 1.3
            serving_memory_temporary_fraction = 0.40
            serving_memory_activation_communication_fraction = 0.20
            serving_memory_weight_communication_fraction = 0.010
            serving_memory_runtime_reserve_fraction = 0.07
            serving_memory_fragmentation_fraction = 0.04

            [calibration_policy]
            valid_shape = "reject"
            invalid_shape = "reject"
            coverage = "warn"
            fit_confidence = "reject"
            fit_extrapolation = "warn"
            fit_partially_bounded = "warn"
            fit_unbounded = "reject"
            fit_sample_count = "reject"
            fit_validation_sample_count = "warn"
            fit_source = "reject"
            fit_uncertainty = "warn"
            profile_source = "reject"
            profile_date = "warn"
            profile_runtime = "reject"
            min_coverage_score = 0.75
            min_fit_confidence_score = 0.80
            min_fit_confidence_level = 0.95
            min_fit_sample_count = 24
            min_fit_validation_sample_count = 6
            max_fit_relative_uncertainty_pct = 20.0
            max_fit_absolute_uncertainty_ms = 3.5
            min_serving_phase_coverage_fraction = 0.60
            uncertainty_ranking_weight = 1.5
            require_phase_coverage = false

            [approximation_policy]
            preset = "topology_sensitive"
            default = "warn"
            reject_categories = ["queueing", "routing"]
            reject_codes = ["node-set-kv-handoff"]
            warn_categories = ["memory"]
            warn_codes = ["component-memory-estimate"]

            [search]
            tensor_ranks = [1, 2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            serving_stack = "vLLM"
            runtime_features = ["Paged Attention", "cuda-graphs", "paged_attention"]
            objective = "memory_pressure"
            slo_miss_penalty_weight = 0.75
            ttft_slo_miss_penalty_weight = 1.25
            tpot_slo_miss_penalty_weight = 0.50
            itl_slo_miss_penalty_weight = 0.60
            e2el_slo_miss_penalty_weight = 2.00
            deadline_miss_penalty_weight = 3.00
            topology_risk_penalty_weight = 4.00
            max_memory_pressure_fraction = 0.82
            max_unique_gpus = 12
            min_throughput_tokens_per_s = 123.5
            max_ttft_ms = 250.0
            max_tpot_ms = 50.0
            max_itl_ms = 60.0
            max_e2el_s = 2.0
            min_kv_route_rail_count = 2
            require_kv_route_rail_metadata = true
            require_gpudirect_kv_paths = true
            prefill_nodes = [0]
            decode_nodes = [1]
            prefill_gpu_tags = ["fast-prefill"]
            decode_gpu_tag = "decode-fast"

            [serving.cost]
            default_gpu_hour_usd = 3.50
            node_hour_usd = 1.25
            kwh_usd = 0.12
            default_gpu_watts = 700.0
            node_watts = 1000.0

            [[serving.cost.gpu_rates]]
            gpu = "H100 SXM5"
            gpu_hour_usd = 4.00
            watts = 750.0

            [serving.services.prefill]
            health = "healthy"
            worker_scale = 2.0

            [serving.services.decode]
            health = "healthy"
            worker_scale = 1.5

            [serving.pool_search]
            prefill_groups = ["h100"]
            decode_groups = ["a100"]
            prefill_node_counts = [1, 2]
            decode_node_counts = [1]
            prefill_gpu_tag = "pool-search-prefill"
            decode_gpu_tags = ["pool-search-decode"]
            prefill_node_tags = ["pool-search-prefill-nodes"]
            decode_node_tag = "pool-search-decode-nodes"
            prefill_racks = ["rack-a", "rack-c"]
            decode_failure_domain = "az-b"
            min_prefill_racks = 2
            min_prefill_failure_domains = 2
            min_decode_racks = 1
            max_candidates = 8

            [[serving.pool_candidates]]
            label = "alternate"
            prefill_groups = ["H100"]
            decode_nodes = [1]
            prefill_gpu_tags = ["alternate-prefill"]
            decode_gpu_tags = ["alternate-decode"]
            prefill_node_tag = "pool-candidate-prefill"
            prefill_rack = "rack-a"
            decode_failure_domain = "az-b"
            min_prefill_racks = 1
            min_decode_failure_domains = 1

            [[serving.slo_policies]]
            group = "tenant"
            key = "gold"
            max_ttft_slo_miss_rate = 0.01
            max_e2el_slo_miss_rate = 0.02
            max_deadline_miss_rate = 0.03

            [[serving.slo_policies]]
            group = "model"
            key = "llama-70b"
            max_tpot_slo_miss_rate = 0.04
            max_itl_slo_miss_rate = 0.05

            [[serving.slo_policies]]
            group = "priority"
            key = "priority-10"
            max_e2el_slo_miss_rate = 0.06

            [[serving.traffic_classes]]
            name = "silver-tenant"
            group = "tenant"
            key = "silver"
            admission_priority = 7
            max_prefill_tokens = 2048
            max_decode_sequences = 3
            max_resident_tokens = 4096
            max_kv_blocks = 256
            ttft_slo_ms = 175.0
            tpot_slo_ms = 35.0
            max_queue_delay_ms = 25.0
            max_kv_queue_delay_ms = 35.0
            max_decode_queue_delay_ms = 45.0
            max_decode_iteration_queue_delay_ms = 55.0
            request_timeout_ms = 900.0
            slo_miss_penalty_weight = 0.10
            ttft_slo_miss_penalty_weight = 0.20
            e2el_slo_miss_penalty_weight = 0.30
            deadline_miss_penalty_weight = 0.40
            max_ttft_slo_miss_rate = 0.005
            max_e2el_slo_miss_rate = 0.015

            [serving.traffic]
            request_count = 12
            arrival = "poisson"
            arrival_rate_per_s = 250.0
            arrival_seed = 99
            arrival_gap_ms = 1.0
            routing_policy = "topology_aware"
            shape_seed = 123
            prefill_batching = "continuous"
            max_prefill_batch_tokens = 4096
            max_prefill_chunk_tokens = 512
            max_prefill_tokens = 8192
            max_prefill_tokens_per_node = 4096
            max_prefill_tokens_per_gpu = 1024
            max_prefill_worker_slots_per_gpu = 2
            decode_batching = "continuous"
            decode_capacity_policy = "request_reject"
            service_backpressure_penalty_weight = 0.25
            max_decode_batch_tokens = 32
            max_decode_sequences = 64
            max_resident_tokens = 262144
            max_decode_sequences_per_node = 32
            max_resident_tokens_per_node = 131072
            max_decode_sequences_per_gpu = 8
            max_decode_worker_slots_per_gpu = 2
            max_resident_tokens_per_gpu = 32768
            max_kv_transfer_worker_slots_per_gpu = 2
            kv_block_tokens = 16
            max_kv_blocks = 16384
            max_kv_blocks_per_node = 8192
            max_kv_blocks_per_gpu = 2048
            ttft_slo_ms = 250.0
            tpot_slo_ms = 40.0
            itl_slo_ms = 40.0
            e2el_slo_ms = 1000.0
            max_ttft_slo_miss_rate = 0.01
            max_tpot_slo_miss_rate = 0.02
            max_itl_slo_miss_rate = 0.03
            max_e2el_slo_miss_rate = 0.04
            max_deadline_miss_rate = 0.05
            measurement_start_ms = 5.0
            measurement_end_ms = 25.0
            max_queue_delay_ms = 10.0
            max_kv_queue_delay_ms = 20.0
            max_decode_queue_delay_ms = 30.0
            max_decode_iteration_queue_delay_ms = 40.0
            request_timeout_ms = 750.0
            prefix_cache_hit_rate = 0.25
            batch_sizes = [1, 4]
            prompt_tokens = [512, 1024]
            decode_tokens = [64, 128]

            [serving.traffic.services.kv_transfer]
            health = "draining"
            worker_scale = 0.5

            [serving.traffic.batch_size_distribution]
            kind = "weighted"
            values = [1, 2, 4]
            weights = [0.6, 0.3, 0.1]

            [serving.traffic.prompt_tokens_distribution]
            kind = "lognormal"
            median = 1024.0
            sigma = 0.7
            min = 128
            max = 4096

            [serving.traffic.decode_tokens_distribution]
            kind = "uniform"
            min = 16
            max = 256

            [serving.prefill_search]
            tensor_ranks = [2]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving.decode_search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
    )
    .unwrap();

    assert_eq!(workload.serving_stack.as_deref(), Some("vLLM"));
    assert_eq!(
        workload.serving_runtime_features,
        vec!["paged_attention", "cuda_graphs"]
    );
    let serving = workload.serving.unwrap();
    assert_eq!(workload.calibration.compute_efficiency, 0.42);
    assert_eq!(workload.calibration.decode_memory_bandwidth_scale, 0.75);
    assert_eq!(workload.calibration.serving_memory_temporary_fraction, 0.40);
    assert_eq!(
        workload
            .calibration
            .serving_memory_activation_communication_fraction,
        0.20
    );
    assert_eq!(
        workload
            .calibration
            .serving_memory_weight_communication_fraction,
        0.010
    );
    assert_eq!(
        workload.calibration.serving_memory_runtime_reserve_fraction,
        0.07
    );
    assert_eq!(
        workload.calibration.serving_memory_fragmentation_fraction,
        0.04
    );
    assert_eq!(
        workload.calibration_policy.valid_shape,
        CalibrationGateMode::Reject
    );
    assert_eq!(
        workload.calibration_policy.invalid_shape,
        CalibrationGateMode::Reject
    );
    assert_eq!(
        workload.calibration_policy.coverage,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.calibration_policy.fit_confidence,
        CalibrationGateMode::Reject
    );
    assert_eq!(
        workload.calibration_policy.fit_extrapolation,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.calibration_policy.fit_partially_bounded,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.calibration_policy.fit_unbounded,
        CalibrationGateMode::Reject
    );
    assert_eq!(
        workload.calibration_policy.fit_sample_count,
        CalibrationGateMode::Reject
    );
    assert_eq!(
        workload.calibration_policy.fit_validation_sample_count,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.calibration_policy.fit_source,
        CalibrationGateMode::Reject
    );
    assert_eq!(
        workload.calibration_policy.fit_uncertainty,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.calibration_policy.profile_source,
        CalibrationGateMode::Reject
    );
    assert_eq!(
        workload.calibration_policy.profile_date,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.calibration_policy.profile_runtime,
        CalibrationGateMode::Reject
    );
    assert_eq!(workload.calibration_policy.min_coverage_score, Some(0.75));
    assert_eq!(
        workload.calibration_policy.min_fit_confidence_score,
        Some(0.80)
    );
    assert_eq!(
        workload.calibration_policy.min_fit_confidence_level,
        Some(0.95)
    );
    assert_eq!(workload.calibration_policy.min_fit_sample_count, Some(24));
    assert_eq!(
        workload.calibration_policy.min_fit_validation_sample_count,
        Some(6)
    );
    assert_eq!(
        workload.calibration_policy.max_fit_relative_uncertainty_pct,
        Some(20.0)
    );
    assert_eq!(
        workload.calibration_policy.max_fit_absolute_uncertainty_s,
        Some(0.0035)
    );
    assert_eq!(
        workload
            .calibration_policy
            .min_serving_phase_coverage_fraction,
        Some(0.60)
    );
    assert_eq!(workload.calibration_policy.uncertainty_ranking_weight, 1.5);
    assert!(!workload.calibration_policy.require_phase_coverage);
    assert_eq!(
        workload.approximation_policy.preset,
        Some(ApproximationPolicyPreset::TopologySensitive)
    );
    assert_eq!(
        workload.approximation_policy.default_action,
        CalibrationGateMode::Warn
    );
    assert_eq!(
        workload.approximation_policy.reject_categories,
        vec!["queueing", "routing"]
    );
    assert_eq!(
        workload.approximation_policy.reject_codes,
        vec!["node_set_kv_handoff"]
    );
    assert_eq!(
        workload.approximation_policy.warn_categories,
        vec!["memory"]
    );
    assert_eq!(
        workload.approximation_policy.warn_codes,
        vec!["component_memory_estimate"]
    );
    assert_eq!(
        serving.deployment_mode,
        ServingDeploymentMode::FullyDisaggregated
    );
    assert_eq!(serving.prefill_nodes, vec![0]);
    assert_eq!(serving.objective, ServingObjective::MinimizeMemoryPressure);
    assert_eq!(serving.slo_miss_penalty_weight, 0.75);
    assert_eq!(serving.slo_miss_penalty_weights.aggregate, 0.75);
    assert_eq!(serving.slo_miss_penalty_weights.ttft, 1.25);
    assert_eq!(serving.slo_miss_penalty_weights.tpot, 0.50);
    assert_eq!(serving.slo_miss_penalty_weights.itl, 0.60);
    assert_eq!(serving.slo_miss_penalty_weights.e2el, 2.00);
    assert_eq!(serving.slo_miss_penalty_weights.deadline, 3.00);
    assert_eq!(serving.topology_risk_penalty_weight, 4.00);
    assert_eq!(serving.max_memory_pressure_fraction, Some(0.82));
    assert_eq!(serving.max_unique_gpus, Some(12));
    assert_eq!(serving.min_throughput_tokens_per_s, Some(123.5));
    assert_eq!(serving.traffic.metric_ceilings.max_ttft_s, Some(0.25));
    assert_eq!(serving.traffic.metric_ceilings.max_tpot_s, Some(0.05));
    assert_eq!(serving.traffic.metric_ceilings.max_itl_s, Some(0.06));
    assert_eq!(serving.traffic.metric_ceilings.max_e2el_s, Some(2.0));
    assert_eq!(
        serving
            .traffic
            .kv_route_constraints
            .min_inter_node_rail_count,
        Some(2)
    );
    assert!(
        serving
            .traffic
            .kv_route_constraints
            .require_inter_node_rail_metadata
    );
    assert!(serving.traffic.kv_route_constraints.require_gpudirect);
    assert_eq!(serving.cost_model.default_gpu_hour_usd, Some(3.50));
    assert_eq!(serving.cost_model.node_hour_usd, Some(1.25));
    assert_eq!(serving.cost_model.kwh_usd, Some(0.12));
    assert_eq!(serving.cost_model.default_gpu_watts, Some(700.0));
    assert_eq!(serving.cost_model.node_watts, Some(1000.0));
    assert_eq!(serving.cost_model.gpu_rates.len(), 1);
    assert_eq!(serving.cost_model.gpu_rates[0].gpu_label, "H100 SXM5");
    assert_eq!(serving.cost_model.gpu_rates[0].gpu_hour_usd, Some(4.00));
    assert_eq!(serving.cost_model.gpu_rates[0].watts, Some(750.0));
    assert_eq!(serving.search.prefill.tensor_ranks, vec![2]);
    assert_eq!(serving.search.decode.tensor_ranks, vec![1]);
    assert_eq!(serving.pool_candidates.len(), 2);
    assert_eq!(
        serving.pool_candidates[0].prefill_gpu_labels,
        vec!["fast_prefill"]
    );
    assert_eq!(
        serving.pool_candidates[0].decode_gpu_labels,
        vec!["decode_fast"]
    );
    assert_eq!(
        serving.pool_candidates[1].label.as_deref(),
        Some("alternate")
    );
    assert_eq!(serving.pool_candidates[1].prefill_groups, vec!["h100"]);
    assert_eq!(serving.pool_candidates[1].decode_nodes, vec![1]);
    assert_eq!(
        serving.pool_candidates[1].prefill_gpu_labels,
        vec!["alternate_prefill"]
    );
    assert_eq!(
        serving.pool_candidates[1].decode_gpu_labels,
        vec!["alternate_decode"]
    );
    assert_eq!(
        serving.pool_candidates[1].prefill_node_filter.node_labels,
        vec!["pool_candidate_prefill"]
    );
    assert_eq!(
        serving.pool_candidates[1].prefill_node_filter.racks,
        vec!["rack_a"]
    );
    assert_eq!(
        serving.pool_candidates[1]
            .decode_node_filter
            .failure_domains,
        vec!["az_b"]
    );
    assert_eq!(
        serving.pool_candidates[1].domain_spread.min_prefill_racks,
        Some(1)
    );
    assert_eq!(
        serving.pool_candidates[1]
            .domain_spread
            .min_decode_failure_domains,
        Some(1)
    );
    assert_eq!(serving.slo_policies.len(), 4);
    assert_eq!(serving.slo_policies[0].group, "tenant");
    assert_eq!(serving.slo_policies[0].key, "gold");
    assert_eq!(serving.slo_policies[0].max_ttft_slo_miss_rate, Some(0.01));
    assert_eq!(serving.slo_policies[0].max_e2el_slo_miss_rate, Some(0.02));
    assert_eq!(serving.slo_policies[0].max_deadline_miss_rate, Some(0.03));
    assert_eq!(serving.slo_policies[1].group, "model_id");
    assert_eq!(serving.slo_policies[1].key, "llama-70b");
    assert_eq!(serving.slo_policies[1].max_tpot_slo_miss_rate, Some(0.04));
    assert_eq!(serving.slo_policies[1].max_itl_slo_miss_rate, Some(0.05));
    assert_eq!(serving.slo_policies[2].group, "priority");
    assert_eq!(serving.slo_policies[2].key, "priority-10");
    assert_eq!(serving.slo_policies[2].max_e2el_slo_miss_rate, Some(0.06));
    assert_eq!(serving.slo_policies[3].group, "tenant");
    assert_eq!(serving.slo_policies[3].key, "silver");
    assert_eq!(serving.slo_policies[3].max_ttft_slo_miss_rate, Some(0.005));
    assert_eq!(serving.slo_policies[3].max_e2el_slo_miss_rate, Some(0.015));
    let pool_search = serving.pool_search.unwrap();
    assert_eq!(pool_search.prefill_groups, vec!["h100"]);
    assert_eq!(pool_search.decode_groups, vec!["a100"]);
    assert_eq!(pool_search.prefill_node_counts, vec![1, 2]);
    assert_eq!(pool_search.prefill_gpu_labels, vec!["pool_search_prefill"]);
    assert_eq!(pool_search.decode_gpu_labels, vec!["pool_search_decode"]);
    assert_eq!(
        pool_search.prefill_node_filter.node_labels,
        vec!["pool_search_prefill_nodes"]
    );
    assert_eq!(
        pool_search.decode_node_filter.node_labels,
        vec!["pool_search_decode_nodes"]
    );
    assert_eq!(
        pool_search.prefill_node_filter.racks,
        vec!["rack_a", "rack_c"]
    );
    assert_eq!(pool_search.decode_node_filter.failure_domains, vec!["az_b"]);
    assert_eq!(pool_search.domain_spread.min_prefill_racks, Some(2));
    assert_eq!(
        pool_search.domain_spread.min_prefill_failure_domains,
        Some(2)
    );
    assert_eq!(pool_search.domain_spread.min_decode_racks, Some(1));
    assert!(!pool_search.allow_overlap);
    assert_eq!(pool_search.max_candidates, 8);
    assert_eq!(serving.traffic.request_count, Some(12));
    assert_eq!(serving.traffic.arrival_gap_s, Some(0.001));
    assert_eq!(
        serving.traffic.arrival,
        ServingArrivalPattern::Poisson {
            rate_per_s: 250.0,
            seed: 99
        }
    );
    assert_eq!(
        serving.traffic.routing_policy,
        ServingRoutingPolicy::TopologyAware
    );
    assert_eq!(
        serving.traffic.prefill_batching,
        ServingPrefillBatching::Continuous {
            max_batch_tokens: Some(4096),
            chunk_tokens: Some(512)
        }
    );
    assert_eq!(
        serving.traffic.decode_batching,
        ServingDecodeBatching::Continuous {
            max_batch_tokens: Some(32)
        }
    );
    assert_eq!(
        serving.traffic.decode_capacity_policy,
        ServingDecodeCapacityPolicy::RequestReject
    );
    assert_eq!(
        serving.traffic.services.prefill.health,
        ServingServiceHealth::Healthy
    );
    assert_eq!(serving.traffic.services.prefill.worker_scale, 2.0);
    assert_eq!(
        serving.traffic.services.decode.health,
        ServingServiceHealth::Healthy
    );
    assert_eq!(serving.traffic.services.decode.worker_scale, 1.5);
    assert_eq!(
        serving.traffic.services.kv_transfer.health,
        ServingServiceHealth::Draining
    );
    assert_eq!(serving.traffic.services.kv_transfer.worker_scale, 0.5);
    assert_eq!(serving.traffic.max_prefill_tokens, Some(8192));
    assert_eq!(serving.traffic.max_prefill_tokens_per_node, Some(4096));
    assert_eq!(serving.traffic.max_prefill_tokens_per_gpu, Some(1024));
    assert_eq!(serving.traffic.max_prefill_worker_slots_per_gpu, Some(2));
    assert_eq!(serving.traffic.max_decode_sequences, Some(64));
    assert_eq!(serving.traffic.max_resident_tokens, Some(262144));
    assert_eq!(serving.traffic.max_decode_sequences_per_node, Some(32));
    assert_eq!(serving.traffic.max_resident_tokens_per_node, Some(131072));
    assert_eq!(serving.traffic.max_decode_sequences_per_gpu, Some(8));
    assert_eq!(serving.traffic.max_decode_worker_slots_per_gpu, Some(2));
    assert_eq!(serving.traffic.max_resident_tokens_per_gpu, Some(32768));
    assert_eq!(
        serving.traffic.max_kv_transfer_worker_slots_per_gpu,
        Some(2)
    );
    assert_eq!(serving.traffic.kv_block_tokens, Some(16));
    assert_eq!(serving.traffic.max_kv_blocks, Some(16384));
    assert_eq!(serving.traffic.max_kv_blocks_per_node, Some(8192));
    assert_eq!(serving.traffic.max_kv_blocks_per_gpu, Some(2048));
    assert_eq!(serving.traffic.ttft_slo_s, Some(0.25));
    assert_eq!(serving.traffic.tpot_slo_s, Some(0.04));
    assert_eq!(serving.traffic.itl_slo_s, Some(0.04));
    assert_eq!(serving.traffic.e2el_slo_s, Some(1.0));
    assert_eq!(serving.traffic.max_ttft_slo_miss_rate, Some(0.01));
    assert_eq!(serving.traffic.max_tpot_slo_miss_rate, Some(0.02));
    assert_eq!(serving.traffic.max_itl_slo_miss_rate, Some(0.03));
    assert_eq!(serving.traffic.max_e2el_slo_miss_rate, Some(0.04));
    assert_eq!(serving.traffic.max_deadline_miss_rate, Some(0.05));
    assert_eq!(serving.traffic.traffic_classes.len(), 1);
    assert_eq!(serving.traffic.traffic_classes[0].name, "silver-tenant");
    assert_eq!(serving.traffic.traffic_classes[0].group, "tenant");
    assert_eq!(serving.traffic.traffic_classes[0].key, "silver");
    assert_eq!(
        serving.traffic.traffic_classes[0].admission_priority,
        Some(7)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].max_prefill_tokens,
        Some(2048)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].max_decode_sequences,
        Some(3)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].max_resident_tokens,
        Some(4096)
    );
    assert_eq!(serving.traffic.traffic_classes[0].max_kv_blocks, Some(256));
    assert_eq!(
        serving.traffic.traffic_classes[0]
            .slo_miss_penalty_weights
            .aggregate,
        0.10
    );
    assert_eq!(
        serving.traffic.traffic_classes[0]
            .slo_miss_penalty_weights
            .ttft,
        0.20
    );
    assert_eq!(
        serving.traffic.traffic_classes[0]
            .slo_miss_penalty_weights
            .e2el,
        0.30
    );
    assert_eq!(
        serving.traffic.traffic_classes[0]
            .slo_miss_penalty_weights
            .deadline,
        0.40
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].slo,
        ServingRequestSlo {
            ttft_s: Some(0.175),
            tpot_s: Some(0.035),
            itl_s: None,
            e2el_s: None
        }
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].max_queue_delay_s,
        Some(0.025)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].max_kv_queue_delay_s,
        Some(0.035)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].max_decode_queue_delay_s,
        Some(0.045)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].max_decode_iteration_queue_delay_s,
        Some(0.055)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0].request_timeout_s,
        Some(0.900)
    );
    assert_eq!(
        serving.traffic.traffic_classes[0]
            .slo_miss_penalty_weights
            .aggregate,
        0.10
    );
    assert_eq!(
        serving.traffic.traffic_classes[0]
            .slo_miss_penalty_weights
            .ttft,
        0.20
    );
    assert_eq!(
        serving.traffic.traffic_classes[0]
            .slo_miss_penalty_weights
            .e2el,
        0.30
    );
    assert_eq!(serving.traffic.measurement_start_s, Some(0.005));
    assert_eq!(serving.traffic.measurement_end_s, Some(0.025));
    assert_eq!(serving.traffic.max_queue_delay_s, Some(0.010));
    assert_eq!(serving.traffic.max_kv_queue_delay_s, Some(0.020));
    assert_eq!(serving.traffic.max_decode_queue_delay_s, Some(0.030));
    assert_eq!(
        serving.traffic.max_decode_iteration_queue_delay_s,
        Some(0.040)
    );
    assert_eq!(serving.traffic.service_backpressure_penalty_weight, 0.25);
    assert_eq!(serving.traffic.request_timeout_s, Some(0.750));
    assert_eq!(serving.traffic.shape_seed, 123);
    assert_eq!(serving.traffic.prefix_cache_hit_rate, Some(0.25));
    assert_eq!(
        serving.traffic.batch_size_distribution,
        Some(ServingValueDistribution::Weighted {
            values: vec![1, 2, 4],
            weights: vec![0.6, 0.3, 0.1]
        })
    );
    assert_eq!(
        serving.traffic.prompt_tokens_distribution,
        Some(ServingValueDistribution::LogNormal {
            median: 1024.0,
            sigma: 0.7,
            min: 128,
            max: 4096
        })
    );
    assert_eq!(
        serving.traffic.decode_tokens_distribution,
        Some(ServingValueDistribution::Uniform { min: 16, max: 256 })
    );
    assert_eq!(serving.traffic.prompt_tokens, vec![512, 1024]);
}

#[test]
fn parses_serving_cost_energy_and_power_objectives() {
    for (objective, expected) in [
        ("cost", ServingObjective::MinimizeCost),
        ("minimize_energy_kwh", ServingObjective::MinimizeEnergy),
        ("average_power_watts", ServingObjective::MinimizePower),
    ] {
        let toml = format!(
            r#"
                [model]
                layers = 4
                hidden_size = 4096
                attention_heads = 32
                kv_heads = 8
                vocab_size = 32000
                parameters_gb = 16.0
                dtype = "bf16"

                [request]
                batch_size = 1
                prompt_tokens = 128
                decode_tokens = 16
                max_sequence_tokens = 256
                phase = "end_to_end"

                [search]
                tensor_ranks = [1]
                pipeline_ranks = [1]
                expert_ranks = [1]
                data_ranks = [1]

                [serving]
                objective = "{objective}"
                prefill_nodes = [0]
                decode_nodes = [1]
                "#
        );
        let workload = parse_workload(&toml).unwrap();
        assert_eq!(workload.serving.unwrap().objective, expected);
    }
}

#[test]
fn parses_serving_measurement_warmup_and_cooldown() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            request_count = 4
            arrival_gap_s = 1.0
            measurement_warmup_ms = 500.0
            measurement_cooldown_s = 1.25
            measurement_steady_state = true
            measurement_steady_state_min_requests = 3
            measurement_steady_state_max_cv = 0.20
            "#,
    )
    .unwrap();

    let serving = workload.serving.unwrap();
    assert_eq!(serving.traffic.measurement_start_s, None);
    assert_eq!(serving.traffic.measurement_end_s, None);
    assert_eq!(serving.traffic.measurement_warmup_s, Some(0.5));
    assert_eq!(serving.traffic.measurement_cooldown_s, Some(1.25));
    assert!(serving.traffic.measurement_steady_state);
    assert_eq!(
        serving.traffic.measurement_steady_state_min_requests,
        Some(3)
    );
    assert_eq!(serving.traffic.measurement_steady_state_max_cv, Some(0.20));
}

#[test]
fn rejects_invalid_serving_slo_miss_rate_limits() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]

            [serving.traffic]
            max_e2el_slo_miss_rate = 1.1
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string().contains(
            "serving.traffic.max_e2el_slo_miss_rate must be finite and between 0.0 and 1.0"
        )
    );
}

#[test]
fn rejects_invalid_serving_slo_miss_penalty_weight() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]
            e2el_slo_miss_penalty_weight = -1.0
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("serving.e2el_slo_miss_penalty_weight must be finite and non-negative")
    );
}

#[test]
fn rejects_invalid_serving_topology_risk_penalty_weight() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]
            topology_risk_penalty_weight = -1.0
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("serving.topology_risk_penalty_weight must be finite and non-negative")
    );
}

#[test]
fn rejects_invalid_serving_slo_policy_scope() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]

            [[serving.slo_policies]]
            group = "request_class"
            key = "gold"
            max_e2el_slo_miss_rate = 0.0
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("unsupported serving.slo_policies[0].group 'request_class'")
    );
}

#[test]
fn rejects_invalid_serving_traffic_class() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]

            [[serving.traffic_classes]]
            name = "bad"
            group = "prefill_node"
            key = "node-0"
            e2el_slo_ms = 1000.0
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("unsupported serving.traffic_classes[0].group 'prefill_node'")
    );
}

#[test]
fn rejects_duplicate_serving_traffic_class_selectors() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]

            [[serving.traffic_classes]]
            name = "gold-slo"
            group = "tenant"
            key = "gold"
            e2el_slo_ms = 1000.0

            [[serving.traffic_classes]]
            name = "gold-capacity"
            group = "tenant"
            key = "gold"
            max_decode_sequences = 8
            "#,
    )
    .unwrap_err();

    assert!(err.to_string().contains(
            "serving.traffic_classes[1] selector group='tenant' key='gold' duplicates traffic class 'gold-slo'"
        ));
}

#[test]
fn rejects_invalid_serving_traffic_class_capacity_limit() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]

            [[serving.traffic_classes]]
            name = "gold"
            group = "tenant"
            key = "gold"
            max_decode_sequences = 0
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("serving.traffic_classes[0].max_decode_sequences must be greater than zero")
    );
}

#[test]
fn rejects_invalid_serving_traffic_class_prefill_capacity_limit() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            prefill_nodes = [0]
            decode_nodes = [0]

            [[serving.traffic_classes]]
            name = "gold"
            group = "tenant"
            key = "gold"
            max_prefill_tokens = 0
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("serving.traffic_classes[0].max_prefill_tokens must be greater than zero")
    );
}

#[test]
fn parses_serving_trace_requests() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 2

            [[serving.traffic.requests]]
            request_id = "req-0"
            tenant = "tenant-a"
            model_id = "llama-70b"
            cache_key = "shared-prefix-a"
            arrival_ms = 0.0
            priority = 5
            ttft_slo_ms = 100.0
            tpot_slo_ms = 25.0
            itl_slo_ms = 25.0
            e2el_slo_ms = 500.0
            deadline_after_ms = 250.0
            batch_size = 1
            prompt_tokens = 256
            decode_tokens = 8
            max_sequence_tokens = 1024
            prefix_cache_hit_tokens = 128

            [[serving.traffic.requests]]
            arrival_ms = 2.5
            cancel_after_ms = 10.0
            batch_size = 2
            prompt_tokens = 512
            decode_tokens = 16
            prefix_cache_hit_rate = 0.5
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.request_count, Some(2));
    assert_eq!(traffic.trace_requests.len(), 2);
    assert_eq!(traffic.trace_requests[0].arrival_s, 0.0);
    assert_eq!(
        traffic.trace_requests[0].request_id.as_deref(),
        Some("req-0")
    );
    assert_eq!(
        traffic.trace_requests[0].tenant.as_deref(),
        Some("tenant-a")
    );
    assert_eq!(
        traffic.trace_requests[0].model_id.as_deref(),
        Some("llama-70b")
    );
    assert_eq!(traffic.trace_requests[0].priority, 5);
    assert_eq!(traffic.trace_requests[0].slo.ttft_s, Some(0.1));
    assert_eq!(traffic.trace_requests[0].slo.tpot_s, Some(0.025));
    assert_eq!(traffic.trace_requests[0].slo.itl_s, Some(0.025));
    assert_eq!(traffic.trace_requests[0].slo.e2el_s, Some(0.5));
    assert_eq!(traffic.trace_requests[0].deadline_s, Some(0.25));
    assert_eq!(traffic.trace_requests[0].prompt_tokens, 256);
    assert_eq!(traffic.trace_requests[0].max_sequence_tokens, Some(1024));
    assert_eq!(
        traffic.trace_requests[0].cache_key.as_deref(),
        Some("shared-prefix-a")
    );
    assert_eq!(traffic.trace_requests[0].prefix_cache_hit_tokens, Some(128));
    assert_eq!(traffic.trace_requests[1].arrival_s, 0.0025);
    assert_eq!(traffic.trace_requests[1].batch_size, 2);
    assert_eq!(traffic.trace_requests[1].prefix_cache_hit_rate, Some(0.5));
    assert_eq!(traffic.trace_requests[1].cancellation_s, Some(0.0125));
}

#[test]
fn loads_calibration_profile_and_applies_inline_overrides() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_calibration_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("profile.toml"),
        r#"
            [profile]
            name = "test-profile"
            hardware = "h100+a100"
            fabric = "ndr"
            model = "test-model"
            dtype = "bf16"
            serving_stack = "test-stack"
            runtime_features = ["Paged Attention", "cuda-graphs", "paged_attention"]
            serving_stack_version = "test-stack 1.2.3"
            gpu_driver_version = "550.54.15"
            cuda = "12.4"
            nccl = "2.21.5"
            ucx = "1.16.0"
            kernel_settings = ["cuda_graphs=true", "block_size=16"]
            environment_hash = "sha256:unit-test"
            source = "unit-test"
            date = "2026-05-25"

            [calibration]
            compute_efficiency = 0.25
            decode_compute_scale = 2.0
            kv_transfer_scale = 1.5

            [valid_shape]
            min_batch_size = 1
            max_batch_size = 8
            min_prompt_tokens = 128
            max_prompt_tokens = 4096
            min_decode_tokens = 1
            max_decode_tokens = 128
            min_sequence_tokens = 512
            max_sequence_tokens = 8192

            [[invalid_shapes]]
            name = "long-context"
            reason = "no benchmark coverage beyond 16k sequence tokens"
            min_sequence_tokens = 16385

            [[fits]]
            name = "decode-latency-fit"
            target = "decode_ms"
            phase = "decode"
            kind = "serving"
            model = "linear"
            unit = "ms"
            intercept = 1.25
            features = ["batch_size", "decode_tokens", "sequence_tokens"]
            coefficients = [0.5, 3.1, 0.002]
            feature_ranges = [
              { feature = "batch_size", min = 1, max = 8 },
              { feature = "decode_tokens", min = 1, max = 128 },
              { feature = "sequence_tokens", min = 512, max = 8192 },
            ]
            r_squared = 0.98
            adjusted_r_squared = 0.97
            rmse = 2.5
            rmse_pct = 3.0
            mean_abs_pct_error = 2.0
            max_abs_pct_error = 7.5
            validation_rmse = 3.5
            validation_rmse_pct = 4.0
            validation_mean_abs_pct_error = 3.0
            validation_max_abs_pct_error = 9.5
            confidence_interval = 4.25
            confidence_interval_pct = 5.0
            confidence_level = 0.95
            sample_count = 24
            validation_sample_count = 6
            source = "unit-test"

            [[benchmarks]]
            name = "decode-b4"
            kind = "serving"
            phase = "decode"
            hardware = "a100"
            fabric = "hdr"
            model = "test-model"
            dtype = "bf16"
            batch_size = 4
            prompt_tokens = 1024
            decode_tokens = 32
            sequence_tokens = 2048
            tensor_ranks = 4
            pipeline_ranks = 1
            expert_ranks = 1
            data_ranks = 1
            measured_ms = 12.5
            predicted_ms = 13.0
            throughput_tokens_per_s = 256.0
            command = "bench decode"
            source = "unit-test"
            "#,
    )
    .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [calibration_profile]
            path = "profile.toml"

            [calibration]
            compute_efficiency = 0.42

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
    )
    .unwrap();

    let workload = load_workload(&workload_path).unwrap();
    let profile = workload.calibration_profile.unwrap();
    assert!(profile.path.ends_with("profile.toml"));
    assert_eq!(profile.name.as_deref(), Some("test-profile"));
    assert_eq!(profile.hardware.as_deref(), Some("h100+a100"));
    assert_eq!(
        profile.serving_runtime_features,
        vec!["paged_attention", "cuda_graphs"]
    );
    assert_eq!(profile.backend_version.as_deref(), Some("test-stack 1.2.3"));
    assert_eq!(profile.driver_version.as_deref(), Some("550.54.15"));
    assert_eq!(profile.cuda_version.as_deref(), Some("12.4"));
    assert_eq!(profile.nccl_version.as_deref(), Some("2.21.5"));
    assert_eq!(profile.ucx_version.as_deref(), Some("1.16.0"));
    assert_eq!(
        profile.kernel_settings,
        vec!["cuda_graphs=true", "block_size=16"]
    );
    assert_eq!(
        profile.environment_hash.as_deref(),
        Some("sha256:unit-test")
    );
    let valid_shape = profile.valid_shape.as_ref().unwrap();
    assert_eq!(valid_shape.min_batch_size, Some(1));
    assert_eq!(valid_shape.max_sequence_tokens, Some(8192));
    assert_eq!(profile.invalid_shapes.len(), 1);
    assert_eq!(
        profile.invalid_shapes[0].name.as_deref(),
        Some("long-context")
    );
    assert_eq!(
        profile.invalid_shapes[0].shape.min_sequence_tokens,
        Some(16385)
    );
    assert_eq!(profile.fits.len(), 1);
    assert_eq!(profile.fits[0].name.as_deref(), Some("decode-latency-fit"));
    assert_eq!(profile.fits[0].target, "decode_ms");
    assert_eq!(profile.fits[0].phase.as_deref(), Some("decode"));
    assert_eq!(profile.fits[0].kind.as_deref(), Some("serving"));
    assert_eq!(profile.fits[0].model, "linear");
    assert_eq!(
        profile.fits[0].features,
        vec!["batch_size", "decode_tokens", "sequence_tokens"]
    );
    assert_eq!(profile.fits[0].coefficients, vec![0.5, 3.1, 0.002]);
    assert_eq!(profile.fits[0].feature_ranges.len(), 3);
    assert_eq!(profile.fits[0].feature_ranges[0].feature, "batch_size");
    assert_eq!(profile.fits[0].feature_ranges[0].min, Some(1.0));
    assert_eq!(profile.fits[0].feature_ranges[0].max, Some(8.0));
    assert_eq!(profile.fits[0].r_squared, Some(0.98));
    assert_eq!(profile.fits[0].validation_rmse, Some(3.5));
    assert_eq!(profile.fits[0].validation_rmse_pct, Some(4.0));
    assert_eq!(profile.fits[0].validation_mean_abs_pct_error, Some(3.0));
    assert_eq!(profile.fits[0].validation_max_abs_pct_error, Some(9.5));
    assert_eq!(profile.fits[0].confidence_interval, Some(4.25));
    assert_eq!(profile.fits[0].confidence_interval_pct, Some(5.0));
    assert_eq!(profile.fits[0].confidence_level, Some(0.95));
    assert_eq!(profile.fits[0].sample_count, Some(24));
    assert_eq!(profile.benchmarks.len(), 1);
    assert_eq!(profile.benchmarks[0].name.as_deref(), Some("decode-b4"));
    assert_eq!(profile.benchmarks[0].kind.as_deref(), Some("serving"));
    assert_eq!(profile.benchmarks[0].batch_size, Some(4));
    assert_eq!(profile.benchmarks[0].measured_ms, Some(12.5));
    assert_eq!(profile.benchmarks[0].throughput_tokens_per_s, Some(256.0));
    assert_eq!(workload.calibration.compute_efficiency, 0.42);
    assert_eq!(workload.calibration.decode_compute_scale, 2.0);
    assert_eq!(workload.calibration.kv_transfer_scale, 1.5);
    assert_eq!(
        workload.calibration_overrides.compute_efficiency,
        Some(0.42)
    );
    assert_eq!(workload.calibration_overrides.decode_compute_scale, None);
    let coverage = workload.calibration_coverage.as_ref().unwrap();
    assert_eq!(coverage.benchmark_count, 1);
    assert_eq!(coverage.complete_shape_benchmark_count, 1);
    assert_eq!(coverage.required_phases, vec!["prefill", "decode"]);
    assert_eq!(coverage.covered_phases, vec!["decode"]);
    assert_eq!(coverage.missing_phases, vec!["prefill"]);
    assert_eq!(coverage.status, "weak");
    assert!(workload.calibration_warnings.is_empty());
    assert!(workload.calibration_invalid_shape_warnings.is_empty());
    assert_eq!(workload.calibration_gate_violations.len(), 2);
    assert!(
        workload
            .calibration_gate_violations
            .iter()
            .any(|violation| violation.code == "coverage_score_below_min")
    );
    assert!(
        workload
            .calibration_gate_violations
            .iter()
            .any(|violation| {
                violation.code == "missing_required_calibration_phases"
                    && violation.action == CalibrationGateMode::Warn
            })
    );

    let _ = std::fs::remove_file(dir.join("profile.toml"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn rejects_malformed_calibration_fit_coefficients() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_calibration_fit_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("profile.toml"),
        r#"
            [[fits]]
            target = "decode_ms"
            model = "linear"
            features = ["batch_size", "decode_tokens"]
            coefficients = [0.5]
            "#,
    )
    .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [calibration_profile]
            path = "profile.toml"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
    )
    .unwrap();

    let err = load_workload(&workload_path).unwrap_err();
    assert!(
        err.to_string()
            .contains("fits[0].features and coefficients must have the same length")
    );

    let _ = std::fs::remove_file(dir.join("profile.toml"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn rejects_malformed_calibration_fit_feature_ranges() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_calibration_fit_range_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("profile.toml"),
        r#"
            [[fits]]
            target = "decode_ms"
            model = "linear"
            features = ["batch_size", "decode_tokens"]
            coefficients = [0.5, 3.1]
            feature_ranges = [
              { feature = "sequence_tokens", min = 512, max = 128 },
            ]
            "#,
    )
    .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [calibration_profile]
            path = "profile.toml"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]
            "#,
    )
    .unwrap();

    let err = load_workload(&workload_path).unwrap_err();
    assert!(
        err.to_string()
            .contains("fits[0].feature_ranges[0].feature must match")
    );

    let _ = std::fs::remove_file(dir.join("profile.toml"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn reports_calibration_warnings_for_shapes_outside_profile_range() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_calibration_warnings_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("profile.toml"),
        r#"
            [valid_shape]
            min_batch_size = 1
            max_batch_size = 2
            min_prompt_tokens = 128
            max_prompt_tokens = 256
            min_decode_tokens = 8
            max_decode_tokens = 16
            min_sequence_tokens = 512
            max_sequence_tokens = 1024

            [[invalid_shapes]]
            name = "long-context"
            reason = "outside validated context window"
            min_sequence_tokens = 2048
            "#,
    )
    .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8
            max_sequence_tokens = 512
            phase = "end_to_end"

            [calibration_profile]
            path = "profile.toml"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 1
            batch_sizes = [4]

            [serving.traffic.prompt_tokens_distribution]
            kind = "lognormal"
            median = 256.0
            sigma = 0.5
            min = 64
            max = 512

            [serving.traffic.decode_tokens_distribution]
            kind = "uniform"
            min = 4
            max = 32

            [[serving.traffic.requests]]
            arrival_ms = 0.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8
            max_sequence_tokens = 2048
            "#,
    )
    .unwrap();

    let workload = load_workload(&workload_path).unwrap();
    let coverage = workload.calibration_coverage.as_ref().unwrap();
    assert_eq!(coverage.status, "no_benchmarks");
    assert_eq!(workload.calibration_warnings.len(), 4);
    assert_eq!(workload.calibration_invalid_shape_warnings.len(), 1);
    assert_eq!(workload.calibration_gate_violations.len(), 10);
    assert!(
        workload
            .calibration_gate_violations
            .iter()
            .any(|violation| violation.code == "coverage_score_below_min")
    );
    assert!(
        workload
            .calibration_gate_violations
            .iter()
            .any(|violation| violation.code == "missing_required_calibration_phases")
    );
    assert!(
        workload
            .calibration_gate_violations
            .iter()
            .any(|violation| violation.code == "calibration_profile_source_unspecified")
    );
    assert!(
        workload
            .calibration_gate_violations
            .iter()
            .any(|violation| violation.code == "calibration_profile_date_unspecified")
    );
    assert!(
        workload
            .calibration_gate_violations
            .iter()
            .any(|violation| {
                violation.code == "calibration_profile_runtime_provenance_incomplete"
                    && violation
                        .message
                        .contains("backend_version, driver_version")
            })
    );
    assert_eq!(
        workload.calibration_invalid_shape_warnings[0]
            .name
            .as_deref(),
        Some("long-context")
    );
    assert!(
        workload.calibration_invalid_shape_warnings[0]
            .message
            .contains("outside validated context window")
    );
    let batch = workload
        .calibration_warnings
        .iter()
        .find(|warning| warning.field == "batch_size")
        .unwrap();
    assert_eq!(batch.observed_min, 1);
    assert_eq!(batch.observed_max, 4);
    assert_eq!(batch.calibrated_max, Some(2));
    let prompt = workload
        .calibration_warnings
        .iter()
        .find(|warning| warning.field == "prompt_tokens")
        .unwrap();
    assert_eq!(prompt.observed_min, 64);
    assert_eq!(prompt.observed_max, 512);
    assert_eq!(prompt.calibrated_min, Some(128));
    assert_eq!(prompt.calibrated_max, Some(256));
    let decode = workload
        .calibration_warnings
        .iter()
        .find(|warning| warning.field == "decode_tokens")
        .unwrap();
    assert_eq!(decode.observed_min, 4);
    assert_eq!(decode.observed_max, 32);
    let sequence = workload
        .calibration_warnings
        .iter()
        .find(|warning| warning.field == "sequence_tokens")
        .unwrap();
    assert_eq!(sequence.observed_min, 512);
    assert_eq!(sequence.observed_max, 2048);
    assert!(sequence.message.contains("512..2048"));

    let _ = std::fs::remove_file(dir.join("profile.toml"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn loads_serving_trace_requests_from_relative_csv() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_trace_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
            dir.join("requests.csv"),
            "\
request_id,tenant,model_id,cache_key,arrival_ms,priority,batch_size,prompt_tokens,decode_tokens,max_sequence_tokens,prefix_cache_hit_tokens,prefix_cache_hit_rate,ttft_slo_ms,tpot_slo_ms,itl_slo_ms,e2el_slo_ms,deadline_after_ms,cancel_after_ms
req-0,tenant-a,llama-70b,shared-prefix-a,0.0,7,1,256,8,1024,128,,100.0,25.0,25.0,500.0,250.0,
req-1,tenant-b,llama-70b,,2.5,-1,2,512,16,,,0.25,,,,,,10.0
",
        )
        .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 2
            trace_csv = "requests.csv"
            "#,
    )
    .unwrap();

    let workload = load_workload(&workload_path).unwrap();
    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.trace_requests.len(), 2);
    assert_eq!(traffic.trace_requests[0].arrival_s, 0.0);
    assert_eq!(
        traffic.trace_requests[0].request_id.as_deref(),
        Some("req-0")
    );
    assert_eq!(
        traffic.trace_requests[0].tenant.as_deref(),
        Some("tenant-a")
    );
    assert_eq!(
        traffic.trace_requests[0].model_id.as_deref(),
        Some("llama-70b")
    );
    assert_eq!(traffic.trace_requests[0].priority, 7);
    assert_eq!(
        traffic.trace_requests[0].cache_key.as_deref(),
        Some("shared-prefix-a")
    );
    assert_eq!(traffic.trace_requests[0].prefix_cache_hit_tokens, Some(128));
    assert_eq!(traffic.trace_requests[0].slo.ttft_s, Some(0.1));
    assert_eq!(traffic.trace_requests[0].slo.tpot_s, Some(0.025));
    assert_eq!(traffic.trace_requests[0].slo.itl_s, Some(0.025));
    assert_eq!(traffic.trace_requests[0].slo.e2el_s, Some(0.5));
    assert_eq!(traffic.trace_requests[0].deadline_s, Some(0.25));
    assert_eq!(traffic.trace_requests[0].max_sequence_tokens, Some(1024));
    assert_eq!(traffic.trace_requests[1].arrival_s, 0.0025);
    assert_eq!(traffic.trace_requests[1].priority, -1);
    assert_eq!(traffic.trace_requests[1].max_sequence_tokens, None);
    assert_eq!(traffic.trace_requests[1].prefix_cache_hit_rate, Some(0.25));
    assert_eq!(traffic.trace_requests[1].cancellation_s, Some(0.0125));

    let _ = std::fs::remove_file(dir.join("requests.csv"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn rejects_invalid_serving_trace_csv_max_sequence_shape() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_bad_trace_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("requests.csv"),
        "\
arrival_ms,batch_size,prompt_tokens,decode_tokens,max_sequence_tokens
0.0,1,128,16,100
",
    )
    .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 1
            trace_csv = "requests.csv"
            "#,
    )
    .unwrap();

    let err = load_workload(&workload_path).unwrap_err();
    assert!(err.to_string().contains("serving.traffic.trace_csv"));
    assert!(
        err.to_string()
            .contains("max_sequence_tokens must be greater than or equal")
    );

    let _ = std::fs::remove_file(dir.join("requests.csv"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn rejects_duplicate_serving_trace_csv_request_ids() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_dup_trace_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("requests.csv"),
        "\
request_id,arrival_ms,batch_size,prompt_tokens,decode_tokens
req-dup,0.0,1,128,16
req-dup,1.0,1,128,16
",
    )
    .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 2
            trace_csv = "requests.csv"
            "#,
    )
    .unwrap();

    let err = load_workload(&workload_path).unwrap_err();
    assert!(
        err.to_string()
            .contains("serving.traffic request_id 'req-dup' is duplicated")
    );

    let _ = std::fs::remove_file(dir.join("requests.csv"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn loads_serving_trace_requests_from_relative_jsonl() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("inference_sim_trace_jsonl_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
            dir.join("requests.jsonl"),
            r#"
{"id":"req-0","tenant_id":"tenant-a","model":"llama-70b","prefix_cache_key":"shared-prefix-a","arrival_time_ms":0.0,"priority":7,"batch":1,"input_tokens":256,"output_tokens":8,"sequence_tokens":1024,"prefix_cache_hit_tokens":128,"ttft_slo_ms":100.0,"tpot_slo_ms":25.0,"itl_slo_ms":25.0,"e2el_slo_ms":500.0,"deadline_after_ms":250.0}
{"request_id":"req-1","tenant":"tenant-b","model_id":"llama-70b","arrival_ms":2.5,"priority":-1,"batch_size":2,"prompt_tokens":512,"decode_tokens":16,"prefix_cache_hit_ratio":0.25,"cancel_after_ms":10.0}
"#,
        )
        .unwrap();
    let workload_path = dir.join("workload.toml");
    std::fs::write(
        &workload_path,
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 2
            trace_jsonl = "requests.jsonl"
            "#,
    )
    .unwrap();

    let workload = load_workload(&workload_path).unwrap();
    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.trace_requests.len(), 2);
    assert_eq!(traffic.trace_requests[0].arrival_s, 0.0);
    assert_eq!(
        traffic.trace_requests[0].request_id.as_deref(),
        Some("req-0")
    );
    assert_eq!(
        traffic.trace_requests[0].tenant.as_deref(),
        Some("tenant-a")
    );
    assert_eq!(
        traffic.trace_requests[0].model_id.as_deref(),
        Some("llama-70b")
    );
    assert_eq!(traffic.trace_requests[0].priority, 7);
    assert_eq!(
        traffic.trace_requests[0].cache_key.as_deref(),
        Some("shared-prefix-a")
    );
    assert_eq!(traffic.trace_requests[0].batch_size, 1);
    assert_eq!(traffic.trace_requests[0].prompt_tokens, 256);
    assert_eq!(traffic.trace_requests[0].decode_tokens, 8);
    assert_eq!(traffic.trace_requests[0].prefix_cache_hit_tokens, Some(128));
    assert_eq!(traffic.trace_requests[0].slo.ttft_s, Some(0.1));
    assert_eq!(traffic.trace_requests[0].slo.tpot_s, Some(0.025));
    assert_eq!(traffic.trace_requests[0].slo.itl_s, Some(0.025));
    assert_eq!(traffic.trace_requests[0].slo.e2el_s, Some(0.5));
    assert_eq!(traffic.trace_requests[0].deadline_s, Some(0.25));
    assert_eq!(traffic.trace_requests[0].max_sequence_tokens, Some(1024));
    assert_eq!(traffic.trace_requests[1].arrival_s, 0.0025);
    assert_eq!(traffic.trace_requests[1].priority, -1);
    assert_eq!(traffic.trace_requests[1].prefix_cache_hit_rate, Some(0.25));
    assert_eq!(traffic.trace_requests[1].cancellation_s, Some(0.0125));

    let _ = std::fs::remove_file(dir.join("requests.jsonl"));
    let _ = std::fs::remove_file(workload_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn applies_serving_trace_window_controls() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 2
            trace_start_ms = 5.0
            trace_end_ms = 10.0
            trace_time_scale = 2.0
            trace_arrival_offset_ms = 1.0

            [[serving.traffic.requests]]
            arrival_ms = 0.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8

            [[serving.traffic.requests]]
            arrival_ms = 5.0
            deadline_after_ms = 50.0
            cancel_after_ms = 5.0
            batch_size = 1
            prompt_tokens = 256
            decode_tokens = 8

            [[serving.traffic.requests]]
            arrival_ms = 10.0
            batch_size = 1
            prompt_tokens = 512
            decode_tokens = 16
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.trace_requests.len(), 2);
    assert!((traffic.trace_requests[0].arrival_s - 0.001).abs() < 1e-12);
    assert!((traffic.trace_requests[0].deadline_s.unwrap() - 0.101).abs() < 1e-12);
    assert!((traffic.trace_requests[0].cancellation_s.unwrap() - 0.011).abs() < 1e-12);
    assert_eq!(traffic.trace_requests[0].prompt_tokens, 256);
    assert!((traffic.trace_requests[1].arrival_s - 0.011).abs() < 1e-12);
    assert_eq!(traffic.trace_requests[1].prompt_tokens, 512);
}

#[test]
fn repeats_serving_trace_requests() {
    let workload = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            request_count = 4
            trace_repeat_count = 2
            trace_repeat_interval_ms = 10.0

            [[serving.traffic.requests]]
            request_id = "req-a"
            arrival_ms = 0.0
            deadline_after_ms = 50.0
            cancel_after_ms = 40.0
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 8

            [[serving.traffic.requests]]
            request_id = "req-b"
            arrival_ms = 2.0
            deadline_after_ms = 50.0
            batch_size = 1
            prompt_tokens = 256
            decode_tokens = 8
            "#,
    )
    .unwrap();

    let traffic = workload.serving.unwrap().traffic;
    assert_eq!(traffic.trace_requests.len(), 4);
    assert_eq!(
        traffic.trace_requests[0].request_id.as_deref(),
        Some("req-a")
    );
    assert_eq!(
        traffic.trace_requests[1].request_id.as_deref(),
        Some("req-b")
    );
    assert_eq!(
        traffic.trace_requests[2].request_id.as_deref(),
        Some("req-a#r2")
    );
    assert_eq!(
        traffic.trace_requests[3].request_id.as_deref(),
        Some("req-b#r2")
    );
    assert!((traffic.trace_requests[0].arrival_s - 0.0).abs() < 1e-12);
    assert!((traffic.trace_requests[1].arrival_s - 0.002).abs() < 1e-12);
    assert!((traffic.trace_requests[2].arrival_s - 0.010).abs() < 1e-12);
    assert!((traffic.trace_requests[3].arrival_s - 0.012).abs() < 1e-12);
    assert!((traffic.trace_requests[2].deadline_s.unwrap() - 0.060).abs() < 1e-12);
    assert!((traffic.trace_requests[2].cancellation_s.unwrap() - 0.050).abs() < 1e-12);
}

#[test]
fn rejects_trace_replay_controls_without_trace_requests() {
    let err = parse_workload(
        r#"
            [model]
            layers = 4
            hidden_size = 4096
            attention_heads = 32
            kv_heads = 8
            vocab_size = 32000
            parameters_gb = 16.0
            dtype = "bf16"

            [request]
            batch_size = 1
            prompt_tokens = 128
            decode_tokens = 16
            max_sequence_tokens = 512
            phase = "end_to_end"

            [search]
            tensor_ranks = [1]
            pipeline_ranks = [1]
            expert_ranks = [1]
            data_ranks = [1]

            [serving]
            mode = "disaggregated"
            prefill_nodes = [0]
            decode_nodes = [1]

            [serving.traffic]
            trace_repeat_count = 2
            "#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("trace replay controls require inline requests, trace_csv, or trace_jsonl")
    );
}
