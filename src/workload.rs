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
    /// Where `parameter_count` came from; drives the consistency check
    /// against `parameters` (see [`ModelSpec::parameter_count_mismatch`]).
    pub parameter_count_source: ParameterCountSource,
    pub dtype: DType,
    pub kv_dtype: Option<DType>,
    pub experts: Option<ExpertSpec>,
}

/// How a model's parameter count (which sets its FLOPs) was obtained.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ParameterCountSource {
    /// Given directly (`parameter_count_billion`), or implied by the weight
    /// bytes when no count is set.
    #[default]
    Explicit,
    /// Derived from layer shapes with an explicit MLP width (`ffn_hidden_size`).
    ShapeWithFfnWidth,
    /// Derived from layer shapes with the default MLP width of 4 x hidden size.
    ShapeWithDefaultFfnWidth,
}

/// Relative disagreement above which a shape-derived parameter count is
/// reported as inconsistent with the weight bytes.
pub const PARAMETER_COUNT_MISMATCH_TOLERANCE: f64 = 0.10;

/// A shape-derived parameter count that disagrees with the weight bytes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParameterCountMismatch {
    /// Parameter count used for FLOPs (derived from shapes).
    pub derived_count: f64,
    /// Parameter count implied by `parameters` bytes / dtype bytes.
    pub bytes_implied_count: f64,
    /// `(derived - bytes_implied) / bytes_implied`.
    pub relative_difference: f64,
}

impl ModelSpec {
    /// Reports when FLOPs come from a default MLP width and the resulting
    /// parameter count disagrees with the weight bytes by more than
    /// [`PARAMETER_COUNT_MISMATCH_TOLERANCE`]. Such a mismatch is otherwise
    /// silently absorbed by a fitted `compute_efficiency`.
    pub fn parameter_count_mismatch(&self) -> Option<ParameterCountMismatch> {
        if self.parameter_count_source != ParameterCountSource::ShapeWithDefaultFfnWidth {
            return None;
        }
        let derived_count = self.parameter_count?;
        let bytes_implied_count =
            self.parameters.as_bytes() as f64 / self.dtype.bytes_per_element() as f64;
        if !(derived_count.is_finite() && bytes_implied_count > 0.0) {
            return None;
        }
        let relative_difference = (derived_count - bytes_implied_count) / bytes_implied_count;
        (relative_difference.abs() > PARAMETER_COUNT_MISMATCH_TOLERANCE).then_some(
            ParameterCountMismatch {
                derived_count,
                bytes_implied_count,
                relative_difference,
            },
        )
    }

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
