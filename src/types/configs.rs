use crate::types::common::{GpuAddr, RankId};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelismConfig {
    pub tensor_ranks: u32,
    pub pipeline_ranks: u32,
    pub expert_ranks: u32,
    pub data_ranks: u32,
}

impl ParallelismConfig {
    pub fn total_ranks(self) -> u32 {
        self.tensor_ranks
            .saturating_mul(self.pipeline_ranks)
            .saturating_mul(self.expert_ranks)
            .saturating_mul(self.data_ranks)
    }

    pub fn validate_dimensions(self) -> Result<(), String> {
        if self.tensor_ranks == 0
            || self.pipeline_ranks == 0
            || self.expert_ranks == 0
            || self.data_ranks == 0
        {
            return Err("parallelism dimensions must be nonzero".to_string());
        }

        if self.total_ranks() == 0 {
            return Err("parallelism rank product overflowed".to_string());
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPlacement {
    pub rank_to_gpu: Vec<GpuAddr>,
}

impl RankPlacement {
    pub fn gpu_for_rank(&self, rank_id: RankId) -> Option<GpuAddr> {
        self.rank_to_gpu.get(rank_id as usize).copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParallelGroups {
    pub tensor_groups: Vec<Vec<RankId>>,
    pub pipeline_stages: Vec<Vec<RankId>>,
    pub expert_groups: Vec<Vec<RankId>>,
    pub data_groups: Vec<Vec<RankId>>,
}
