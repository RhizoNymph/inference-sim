use super::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum SearchDiagnosticsMode {
    Parallelism,
    Serving,
}

impl SearchDiagnosticsMode {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Parallelism => "parallelism",
            Self::Serving => "serving",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct SearchDiagnostics {
    pub(super) mode: SearchDiagnosticsMode,
    pub(super) candidate_space_count: Option<usize>,
    pub(super) prefill_candidate_space_count: Option<usize>,
    pub(super) decode_candidate_space_count: Option<usize>,
    pub(super) serving_pair_space_lower_bound_per_pool: Option<usize>,
    pub(super) max_runtime_ms: Option<u64>,
    pub(super) runtime_elapsed_ms: u128,
    pub(super) searched_candidate_count: usize,
    pub(super) reported_candidate_count: usize,
    pub(super) omitted_rejected_candidate_count: usize,
    pub(super) truncated_by_parallelism_budget: bool,
    pub(super) truncated_by_prefill_budget: bool,
    pub(super) truncated_by_decode_budget: bool,
    pub(super) truncated_by_serving_pair_budget: bool,
    pub(super) truncated_by_runtime_budget: bool,
    pub(super) serving_pair_budget_exhausted: bool,
    pub(super) retain_rejected_candidates: bool,
}

impl SearchDiagnostics {
    pub(super) fn truncated(self) -> bool {
        self.truncated_by_parallelism_budget
            || self.truncated_by_prefill_budget
            || self.truncated_by_decode_budget
            || self.truncated_by_serving_pair_budget
            || self.truncated_by_runtime_budget
    }
}

pub(super) fn parallelism_search_budget(budget: RunSearchBudgetConfig) -> RunSearchBudgetConfig {
    RunSearchBudgetConfig {
        max_parallelism_candidates: budget.max_parallelism_candidates,
        max_runtime_ms: budget.max_runtime_ms,
        retain_rejected_candidates: budget.retain_rejected_candidates,
        ..RunSearchBudgetConfig::default()
    }
}

pub(super) fn serving_search_budget(budget: RunSearchBudgetConfig) -> RunSearchBudgetConfig {
    RunSearchBudgetConfig {
        max_parallelism_candidates: budget.max_parallelism_candidates,
        max_prefill_candidates: budget
            .max_prefill_candidates
            .or(budget.max_parallelism_candidates),
        max_decode_candidates: budget
            .max_decode_candidates
            .or(budget.max_parallelism_candidates),
        max_serving_pairs: budget.max_serving_pairs,
        max_runtime_ms: budget.max_runtime_ms,
        retain_rejected_candidates: budget.retain_rejected_candidates,
    }
}

pub(super) fn retain_rejected_candidates(budget: RunSearchBudgetConfig) -> bool {
    budget.retain_rejected_candidates.unwrap_or(true)
}

pub(super) fn reported_parallelism_results(
    results: &[ScoredParallelismConfig],
    budget: RunSearchBudgetConfig,
) -> Vec<ScoredParallelismConfig> {
    if retain_rejected_candidates(budget) {
        return results.to_vec();
    }
    results
        .iter()
        .filter(|score| score.feasible)
        .cloned()
        .collect()
}

pub(super) fn reported_serving_results(
    results: &[ScoredServingConfig],
    budget: RunSearchBudgetConfig,
) -> Vec<ScoredServingConfig> {
    if retain_rejected_candidates(budget) {
        return results.to_vec();
    }
    results
        .iter()
        .filter(|score| score.feasible)
        .cloned()
        .collect()
}

pub(super) fn rejected_candidate_count<T>(results: &[T], feasible: impl Fn(&T) -> bool) -> usize {
    results.iter().filter(|score| !feasible(score)).count()
}

pub(super) fn search_space_candidate_count(search: &SearchSpace) -> usize {
    search
        .tensor_ranks
        .len()
        .saturating_mul(search.pipeline_ranks.len())
        .saturating_mul(search.expert_ranks.len())
        .saturating_mul(search.data_ranks.len())
}

pub(super) fn effective_budgeted_count(candidate_count: usize, budget: Option<usize>) -> usize {
    budget
        .map(|budget| candidate_count.min(budget))
        .unwrap_or(candidate_count)
}

#[derive(Copy, Clone, Debug)]
pub(super) struct SearchRuntime {
    pub(super) started_at: Instant,
    pub(super) deadline: Option<Instant>,
}

impl SearchRuntime {
    pub(super) fn start(budget: RunSearchBudgetConfig) -> Self {
        let started_at = Instant::now();
        let deadline = budget
            .max_runtime_ms
            .map(Duration::from_millis)
            .and_then(|duration| started_at.checked_add(duration));
        Self {
            started_at,
            deadline,
        }
    }

    pub(super) fn elapsed_ms(self) -> u128 {
        self.started_at.elapsed().as_millis()
    }

    pub(super) fn deadline_expired(self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    pub(super) fn truncated(
        self,
        searched_candidate_count: usize,
        budgeted_candidate_count: usize,
    ) -> bool {
        self.deadline_expired() && searched_candidate_count < budgeted_candidate_count
    }
}

pub(super) fn parallelism_search_diagnostics(
    search: &SearchSpace,
    budget: RunSearchBudgetConfig,
    searched_candidate_count: usize,
    reported_candidate_count: usize,
    omitted_rejected_candidate_count: usize,
    runtime_elapsed_ms: u128,
    truncated_by_runtime_budget: bool,
) -> SearchDiagnostics {
    let candidate_space_count = search_space_candidate_count(search);
    SearchDiagnostics {
        mode: SearchDiagnosticsMode::Parallelism,
        candidate_space_count: Some(candidate_space_count),
        prefill_candidate_space_count: None,
        decode_candidate_space_count: None,
        serving_pair_space_lower_bound_per_pool: None,
        max_runtime_ms: budget.max_runtime_ms,
        runtime_elapsed_ms,
        searched_candidate_count,
        reported_candidate_count,
        omitted_rejected_candidate_count,
        truncated_by_parallelism_budget: budget
            .max_parallelism_candidates
            .is_some_and(|budget| candidate_space_count > budget),
        truncated_by_prefill_budget: false,
        truncated_by_decode_budget: false,
        truncated_by_serving_pair_budget: false,
        truncated_by_runtime_budget,
        serving_pair_budget_exhausted: false,
        retain_rejected_candidates: retain_rejected_candidates(budget),
    }
}

pub(super) fn serving_search_diagnostics(
    serving: &DisaggregatedServingConfig,
    budget: RunSearchBudgetConfig,
    searched_candidate_count: usize,
    reported_candidate_count: usize,
    omitted_rejected_candidate_count: usize,
    runtime_elapsed_ms: u128,
    truncated_by_runtime_budget: bool,
) -> SearchDiagnostics {
    let prefill_candidate_space_count = search_space_candidate_count(&serving.search.prefill);
    let decode_candidate_space_count = search_space_candidate_count(&serving.search.decode);
    let serving_pair_space_lower_bound_per_pool = serving_pair_space_lower_bound(serving, budget);
    let serving_pair_budget_exhausted = budget
        .max_serving_pairs
        .is_some_and(|budget| searched_candidate_count >= budget);

    SearchDiagnostics {
        mode: SearchDiagnosticsMode::Serving,
        candidate_space_count: None,
        prefill_candidate_space_count: Some(prefill_candidate_space_count),
        decode_candidate_space_count: Some(decode_candidate_space_count),
        serving_pair_space_lower_bound_per_pool: Some(serving_pair_space_lower_bound_per_pool),
        max_runtime_ms: budget.max_runtime_ms,
        runtime_elapsed_ms,
        searched_candidate_count,
        reported_candidate_count,
        omitted_rejected_candidate_count,
        truncated_by_parallelism_budget: false,
        truncated_by_prefill_budget: budget
            .max_prefill_candidates
            .is_some_and(|budget| prefill_candidate_space_count > budget),
        truncated_by_decode_budget: budget
            .max_decode_candidates
            .is_some_and(|budget| decode_candidate_space_count > budget),
        truncated_by_serving_pair_budget: budget
            .max_serving_pairs
            .is_some_and(|budget| serving_pair_space_lower_bound_per_pool > budget),
        truncated_by_runtime_budget,
        serving_pair_budget_exhausted,
        retain_rejected_candidates: retain_rejected_candidates(budget),
    }
}

pub(super) fn serving_pair_space_lower_bound(
    serving: &DisaggregatedServingConfig,
    budget: RunSearchBudgetConfig,
) -> usize {
    let prefill_candidate_space_count = search_space_candidate_count(&serving.search.prefill);
    let decode_candidate_space_count = search_space_candidate_count(&serving.search.decode);
    let budgeted_prefill_count =
        effective_budgeted_count(prefill_candidate_space_count, budget.max_prefill_candidates);
    let budgeted_decode_count =
        effective_budgeted_count(decode_candidate_space_count, budget.max_decode_candidates);
    budgeted_prefill_count.saturating_mul(budgeted_decode_count)
}

pub(super) fn effective_serving_pair_budgeted_count(
    serving: &DisaggregatedServingConfig,
    budget: RunSearchBudgetConfig,
) -> usize {
    effective_budgeted_count(
        serving_pair_space_lower_bound(serving, budget),
        budget.max_serving_pairs,
    )
}

pub(super) fn write_markdown_search_diagnostics<W: Write>(
    writer: &mut W,
    diagnostics: SearchDiagnostics,
) -> Result<(), io::Error> {
    write_markdown_key_value_section(
        writer,
        "Search Diagnostics",
        &[
            ("Search Mode", diagnostics.mode.as_str().to_string()),
            (
                "Candidate Space",
                format_optional_usize(diagnostics.candidate_space_count),
            ),
            (
                "Prefill Candidate Space",
                format_optional_usize(diagnostics.prefill_candidate_space_count),
            ),
            (
                "Decode Candidate Space",
                format_optional_usize(diagnostics.decode_candidate_space_count),
            ),
            (
                "Serving Pair Space Lower Bound/Pool",
                format_optional_usize(diagnostics.serving_pair_space_lower_bound_per_pool),
            ),
            (
                "Max Runtime ms",
                format_optional_u64(diagnostics.max_runtime_ms),
            ),
            (
                "Runtime Elapsed ms",
                diagnostics.runtime_elapsed_ms.to_string(),
            ),
            ("Searched", diagnostics.searched_candidate_count.to_string()),
            ("Reported", diagnostics.reported_candidate_count.to_string()),
            (
                "Omitted Rejected",
                diagnostics.omitted_rejected_candidate_count.to_string(),
            ),
            ("Truncated", diagnostics.truncated().to_string()),
            (
                "Retain Rejected Candidates",
                diagnostics.retain_rejected_candidates.to_string(),
            ),
        ],
    )
}

pub(super) fn write_search_budget<W: Write>(
    writer: &mut W,
    budget: RunSearchBudgetConfig,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"search_budget\": {{")?;
    writeln!(
        writer,
        "{indent}  \"max_parallelism_candidates\": {},",
        json_optional_usize(budget.max_parallelism_candidates)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_prefill_candidates\": {},",
        json_optional_usize(budget.max_prefill_candidates)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_decode_candidates\": {},",
        json_optional_usize(budget.max_decode_candidates)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_serving_pairs\": {},",
        json_optional_usize(budget.max_serving_pairs)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_runtime_ms\": {},",
        json_optional_u64(budget.max_runtime_ms)
    )?;
    writeln!(
        writer,
        "{indent}  \"retain_rejected_candidates\": {}",
        retain_rejected_candidates(budget)
    )?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}

fn format_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

pub(super) fn write_search_diagnostics_text<W: Write>(
    writer: &mut W,
    diagnostics: SearchDiagnostics,
) -> Result<(), io::Error> {
    writeln!(
        writer,
        "search_diagnostics mode={} candidate_space={} prefill_candidate_space={} decode_candidate_space={} serving_pair_space_lower_bound_per_pool={} max_runtime_ms={} runtime_elapsed_ms={} searched={} reported={} omitted_rejected={} truncated={} truncated_parallelism={} truncated_prefill={} truncated_decode={} truncated_serving_pairs={} truncated_runtime={} serving_pair_budget_exhausted={} retain_rejected_candidates={}",
        diagnostics.mode.as_str(),
        format_optional_usize(diagnostics.candidate_space_count),
        format_optional_usize(diagnostics.prefill_candidate_space_count),
        format_optional_usize(diagnostics.decode_candidate_space_count),
        format_optional_usize(diagnostics.serving_pair_space_lower_bound_per_pool),
        format_optional_u64(diagnostics.max_runtime_ms),
        diagnostics.runtime_elapsed_ms,
        diagnostics.searched_candidate_count,
        diagnostics.reported_candidate_count,
        diagnostics.omitted_rejected_candidate_count,
        diagnostics.truncated(),
        diagnostics.truncated_by_parallelism_budget,
        diagnostics.truncated_by_prefill_budget,
        diagnostics.truncated_by_decode_budget,
        diagnostics.truncated_by_serving_pair_budget,
        diagnostics.truncated_by_runtime_budget,
        diagnostics.serving_pair_budget_exhausted,
        diagnostics.retain_rejected_candidates
    )
}

pub(super) fn write_search_diagnostics<W: Write>(
    writer: &mut W,
    diagnostics: SearchDiagnostics,
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}\"search_diagnostics\": {{")?;
    writeln!(
        writer,
        "{indent}  \"search_mode\": {},",
        json_string(diagnostics.mode.as_str())
    )?;
    writeln!(
        writer,
        "{indent}  \"candidate_space_count\": {},",
        json_optional_usize(diagnostics.candidate_space_count)
    )?;
    writeln!(
        writer,
        "{indent}  \"prefill_candidate_space_count\": {},",
        json_optional_usize(diagnostics.prefill_candidate_space_count)
    )?;
    writeln!(
        writer,
        "{indent}  \"decode_candidate_space_count\": {},",
        json_optional_usize(diagnostics.decode_candidate_space_count)
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_pair_space_lower_bound_per_pool\": {},",
        json_optional_usize(diagnostics.serving_pair_space_lower_bound_per_pool)
    )?;
    writeln!(
        writer,
        "{indent}  \"max_runtime_ms\": {},",
        json_optional_u64(diagnostics.max_runtime_ms)
    )?;
    writeln!(
        writer,
        "{indent}  \"runtime_elapsed_ms\": {},",
        diagnostics.runtime_elapsed_ms
    )?;
    writeln!(
        writer,
        "{indent}  \"searched_candidate_count\": {},",
        diagnostics.searched_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}  \"reported_candidate_count\": {},",
        diagnostics.reported_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}  \"omitted_rejected_candidate_count\": {},",
        diagnostics.omitted_rejected_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}  \"truncated\": {},",
        diagnostics.truncated()
    )?;
    writeln!(
        writer,
        "{indent}  \"truncated_by_parallelism_budget\": {},",
        diagnostics.truncated_by_parallelism_budget
    )?;
    writeln!(
        writer,
        "{indent}  \"truncated_by_prefill_budget\": {},",
        diagnostics.truncated_by_prefill_budget
    )?;
    writeln!(
        writer,
        "{indent}  \"truncated_by_decode_budget\": {},",
        diagnostics.truncated_by_decode_budget
    )?;
    writeln!(
        writer,
        "{indent}  \"truncated_by_serving_pair_budget\": {},",
        diagnostics.truncated_by_serving_pair_budget
    )?;
    writeln!(
        writer,
        "{indent}  \"truncated_by_runtime_budget\": {},",
        diagnostics.truncated_by_runtime_budget
    )?;
    writeln!(
        writer,
        "{indent}  \"serving_pair_budget_exhausted\": {},",
        diagnostics.serving_pair_budget_exhausted
    )?;
    writeln!(
        writer,
        "{indent}  \"retain_rejected_candidates\": {}",
        diagnostics.retain_rejected_candidates
    )?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}
