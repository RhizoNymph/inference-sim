//! Which KV bytes move between which ranks.
//!
//! Pipeline stage `p` of `pp` holds layers `[p/pp, (p+1)/pp)` of the model
//! (as fractions, so uneven splits are spread proportionally). Tensor rank
//! `t` of `tp` holds KV heads `[t/tp, (t+1)/tp)` when `kv_heads >= tp`;
//! with fewer KV heads than tensor ranks vLLM replicates heads, so rank `t`
//! holds the single head `floor(t * kv_heads / tp)`. Expert ranks hold
//! identical (replicated) attention KV.
//!
//! Every destination rank needs every byte of its own shard, so replicated
//! destination shards are each transferred. A destination piece held by
//! several replicated source ranks is read from replica
//! `destination_rank % replica_count`, which spreads reads deterministically.

use std::collections::BTreeMap;

/// Shape of one placed model instance's KV sharding (data parallelism
/// excluded: a request lives in one replica).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::serving) struct KvShardLayout {
    tensor: u32,
    pipeline: u32,
    expert: u32,
    kv_heads: u32,
}

impl KvShardLayout {
    /// Zero dimensions are treated as 1.
    pub(in crate::serving) fn new(tensor: u32, pipeline: u32, expert: u32, kv_heads: u32) -> Self {
        Self {
            tensor: tensor.max(1),
            pipeline: pipeline.max(1),
            expert: expert.max(1),
            kv_heads: kv_heads.max(1),
        }
    }

    pub(in crate::serving) fn ranks(self) -> usize {
        (self.tensor as usize) * (self.pipeline as usize) * (self.expert as usize)
    }

    /// (pipeline stage, tensor rank) of a rank, matching the solver's rank
    /// order `(pipeline * expert + expert_idx) * tensor + tensor_idx`.
    fn coordinates(self, rank: usize) -> (u32, u32) {
        let rank = rank as u32;
        let tensor_idx = rank % self.tensor;
        let rest = rank / self.tensor;
        let pipeline_idx = (rest / self.expert) % self.pipeline;
        (pipeline_idx, tensor_idx)
    }

    fn layer_interval(self, stage: u32) -> (f64, f64) {
        let pipeline = f64::from(self.pipeline);
        (f64::from(stage) / pipeline, f64::from(stage + 1) / pipeline)
    }

    /// Head-space key and interval (fractions of all KV heads) of a tensor rank.
    fn head_piece(self, tensor_idx: u32) -> (u32, (f64, f64)) {
        if self.kv_heads >= self.tensor {
            let tensor = f64::from(self.tensor);
            (
                tensor_idx,
                (
                    f64::from(tensor_idx) / tensor,
                    f64::from(tensor_idx + 1) / tensor,
                ),
            )
        } else {
            let head = ((u64::from(tensor_idx) * u64::from(self.kv_heads)) / u64::from(self.tensor))
                as u32;
            let heads = f64::from(self.kv_heads);
            (head, (f64::from(head) / heads, f64::from(head + 1) / heads))
        }
    }
}

/// One distinct source shard (pipeline stage x head piece) and the source
/// ranks holding a copy of it.
struct SourcePiece {
    layers: (f64, f64),
    heads: (f64, f64),
    holders: Vec<usize>,
}

/// `fraction` of the request's full KV (all layers, all heads) moves from
/// source rank `src_rank` to destination rank `dst_rank`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(in crate::serving) struct ShardFlow {
    pub(in crate::serving) src_rank: usize,
    pub(in crate::serving) dst_rank: usize,
    pub(in crate::serving) fraction: f64,
}

fn overlap(left: (f64, f64), right: (f64, f64)) -> f64 {
    (left.1.min(right.1) - left.0.max(right.0)).max(0.0)
}

/// Shard-to-shard flows for moving a request's KV from `src` to `dst`,
/// sorted by (source rank, destination rank).
pub(in crate::serving) fn kv_shard_flows(src: KvShardLayout, dst: KvShardLayout) -> Vec<ShardFlow> {
    // Distinct source pieces (stage, head key) and the ranks holding each.
    let mut pieces: BTreeMap<(u32, u32), SourcePiece> = BTreeMap::new();
    for rank in 0..src.ranks() {
        let (stage, tensor_idx) = src.coordinates(rank);
        let (head_key, heads) = src.head_piece(tensor_idx);
        pieces
            .entry((stage, head_key))
            .or_insert_with(|| SourcePiece {
                layers: src.layer_interval(stage),
                heads,
                holders: Vec::new(),
            })
            .holders
            .push(rank);
    }
    let mut flows: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for dst_rank in 0..dst.ranks() {
        let (stage, tensor_idx) = dst.coordinates(dst_rank);
        let layers = dst.layer_interval(stage);
        let (_, heads) = dst.head_piece(tensor_idx);
        for SourcePiece {
            layers: src_layers,
            heads: src_heads,
            holders,
        } in pieces.values()
        {
            let fraction = overlap(layers, *src_layers) * overlap(heads, *src_heads);
            if fraction <= 0.0 || holders.is_empty() {
                continue;
            }
            let src_rank = holders[dst_rank % holders.len()];
            *flows.entry((src_rank, dst_rank)).or_insert(0.0) += fraction;
        }
    }
    flows
        .into_iter()
        .map(|((src_rank, dst_rank), fraction)| ShardFlow {
            src_rank,
            dst_rank,
            fraction,
        })
        .collect()
}
