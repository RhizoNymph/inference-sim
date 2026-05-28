use crate::types::common::Bytes;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Fp16,
    Bf16,
    Fp8,
    Int8,
}

impl DType {
    pub fn bytes_per_element(self) -> u64 {
        match self {
            DType::Fp16 | DType::Bf16 => 2,
            DType::Fp8 | DType::Int8 => 1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DType::Fp16 => "fp16",
            DType::Bf16 => "bf16",
            DType::Fp8 => "fp8",
            DType::Int8 => "int8",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InferencePhase {
    Prefill,
    Decode,
    EndToEnd,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExpertSpec {
    pub expert_count: u32,
    pub top_k: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    pub layers: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub vocab_size: u32,
    pub parameters: Bytes,
    pub parameter_count: Option<f64>,
    pub dtype: DType,
    pub kv_dtype: Option<DType>,
    pub experts: Option<ExpertSpec>,
}

impl ModelSpec {
    pub fn parameter_count(&self) -> f64 {
        self.parameter_count.unwrap_or_else(|| {
            self.parameters.as_bytes() as f64 / self.dtype.bytes_per_element() as f64
        })
    }

    pub fn parameter_count_billion(&self) -> f64 {
        self.parameter_count() / 1e9
    }

    pub fn kv_dtype(&self) -> DType {
        self.kv_dtype.unwrap_or(self.dtype)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct InferenceRequest {
    pub batch_size: u32,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub max_sequence_tokens: u32,
    pub phase: InferencePhase,
}
