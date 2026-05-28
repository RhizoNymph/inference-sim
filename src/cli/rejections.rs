use super::*;

const REJECTION_SUMMARY_EXAMPLE_LIMIT: usize = 5;
const REJECTION_SUMMARY_TEXT_GROUP_LIMIT: usize = 3;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct ServingRejectionGroupKey {
    pub(super) phase: String,
    pub(super) category: String,
    pub(super) resource: String,
    pub(super) code: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ServingRejectionGroupSummary {
    pub(super) key: ServingRejectionGroupKey,
    pub(super) candidate_count: usize,
    pub(super) rejection_count: usize,
    pub(super) example_candidate_ids: Vec<String>,
    pub(super) candidate_ids_truncated: bool,
    pub(super) remediations: Vec<String>,
    pub(super) messages: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ServingRejectionSummary {
    pub(super) rejected_candidate_count: usize,
    pub(super) candidate_with_rejections_count: usize,
    pub(super) total_rejection_count: usize,
    pub(super) groups: Vec<ServingRejectionGroupSummary>,
}

#[derive(Clone, Debug, Default)]
struct ServingRejectionGroupAccumulator {
    rejection_count: usize,
    candidate_ids: BTreeSet<String>,
    remediations: BTreeSet<String>,
    messages: BTreeSet<String>,
}

pub(super) fn write_markdown_serving_rejection_summary<W: Write>(
    writer: &mut W,
    results: &[ScoredServingConfig],
) -> Result<(), io::Error> {
    let summary = serving_rejection_summary(results);
    if summary.total_rejection_count == 0 {
        return Ok(());
    }
    write_markdown_key_value_section(
        writer,
        "Rejection Summary",
        &[
            (
                "Rejected Candidates",
                summary.rejected_candidate_count.to_string(),
            ),
            (
                "Candidates With Rejections",
                summary.candidate_with_rejections_count.to_string(),
            ),
            (
                "Total Rejections",
                summary.total_rejection_count.to_string(),
            ),
        ],
    )?;
    writeln!(writer, "### Top Rejection Groups\n")?;
    write_markdown_row(
        writer,
        &[
            "Phase",
            "Category",
            "Resource",
            "Code",
            "Candidates",
            "Rejections",
            "Examples",
            "Remediations",
        ],
    )?;
    write_markdown_separator(writer, 8)?;
    for group in summary
        .groups
        .iter()
        .take(REJECTION_SUMMARY_TEXT_GROUP_LIMIT)
    {
        write_markdown_row(
            writer,
            &[
                group.key.phase.clone(),
                group.key.category.clone(),
                group.key.resource.clone(),
                group.key.code.clone(),
                group.candidate_count.to_string(),
                group.rejection_count.to_string(),
                if group.candidate_ids_truncated {
                    format!("{}, ...", group.example_candidate_ids.join(", "))
                } else {
                    group.example_candidate_ids.join(", ")
                },
                group.remediations.join("; "),
            ],
        )?;
    }
    writeln!(writer)
}

pub(super) fn write_serving_rejections<W: Write>(
    writer: &mut W,
    indent: &str,
    candidate_id: &str,
    rejections: &[ServingRejection],
    trailing_comma: bool,
) -> Result<(), io::Error> {
    writeln!(writer, "{indent}  \"rejections\": [")?;
    for (idx, rejection) in rejections.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"candidate_id\": {},",
            json_string(candidate_id)
        )?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&rejection.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&rejection.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&rejection.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&rejection.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"observed\": {},",
            json_optional_value(rejection.observed)
        )?;
        writeln!(
            writer,
            "{indent}      \"limit\": {},",
            json_optional_value(rejection.limit)
        )?;
        writeln!(
            writer,
            "{indent}      \"unit\": {},",
            rejection
                .unit
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"remediation\": {},",
            rejection
                .remediation
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string())
        )?;
        writeln!(
            writer,
            "{indent}      \"message\": {}",
            json_string(&rejection.message)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < rejections.len())
        )?;
    }
    writeln!(writer, "{indent}  ]{}", comma(trailing_comma))
}

pub(super) fn write_serving_rejection_summary_text<W: Write>(
    writer: &mut W,
    results: &[ScoredServingConfig],
) -> Result<(), io::Error> {
    let summary = serving_rejection_summary(results);
    if summary.total_rejection_count == 0 {
        return Ok(());
    }

    let top_groups = summary
        .groups
        .iter()
        .take(REJECTION_SUMMARY_TEXT_GROUP_LIMIT)
        .map(|group| {
            format!(
                "{}/{}/{}/{}:{}c/{}r",
                group.key.phase,
                group.key.category,
                group.key.resource,
                group.key.code,
                group.candidate_count,
                group.rejection_count
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        writer,
        "rejection_summary rejected_candidates={} candidates_with_rejections={} total_rejections={} top=[{}]",
        summary.rejected_candidate_count,
        summary.candidate_with_rejections_count,
        summary.total_rejection_count,
        top_groups
    )
}

pub(super) fn write_serving_rejection_summary_json<W: Write>(
    writer: &mut W,
    results: &[ScoredServingConfig],
    indent: &str,
    trailing_comma: bool,
) -> Result<(), io::Error> {
    let summary = serving_rejection_summary(results);
    writeln!(writer, "{indent}\"rejection_summary\": {{")?;
    writeln!(
        writer,
        "{indent}  \"rejected_candidate_count\": {},",
        summary.rejected_candidate_count
    )?;
    writeln!(
        writer,
        "{indent}  \"candidate_with_rejections_count\": {},",
        summary.candidate_with_rejections_count
    )?;
    writeln!(
        writer,
        "{indent}  \"total_rejection_count\": {},",
        summary.total_rejection_count
    )?;
    writeln!(writer, "{indent}  \"groups\": [")?;
    for (idx, group) in summary.groups.iter().enumerate() {
        writeln!(writer, "{indent}    {{")?;
        writeln!(
            writer,
            "{indent}      \"phase\": {},",
            json_string(&group.key.phase)
        )?;
        writeln!(
            writer,
            "{indent}      \"category\": {},",
            json_string(&group.key.category)
        )?;
        writeln!(
            writer,
            "{indent}      \"resource\": {},",
            json_string(&group.key.resource)
        )?;
        writeln!(
            writer,
            "{indent}      \"code\": {},",
            json_string(&group.key.code)
        )?;
        writeln!(
            writer,
            "{indent}      \"candidate_count\": {},",
            group.candidate_count
        )?;
        writeln!(
            writer,
            "{indent}      \"rejection_count\": {},",
            group.rejection_count
        )?;
        writeln!(
            writer,
            "{indent}      \"example_candidate_ids\": {},",
            json_string_array(&group.example_candidate_ids)
        )?;
        writeln!(
            writer,
            "{indent}      \"candidate_ids_truncated\": {},",
            group.candidate_ids_truncated
        )?;
        writeln!(
            writer,
            "{indent}      \"remediations\": {},",
            json_string_array(&group.remediations)
        )?;
        writeln!(
            writer,
            "{indent}      \"messages\": {}",
            json_string_array(&group.messages)
        )?;
        writeln!(
            writer,
            "{indent}    }}{}",
            comma(idx + 1 < summary.groups.len())
        )?;
    }
    writeln!(writer, "{indent}  ]")?;
    writeln!(writer, "{indent}}}{}", comma(trailing_comma))
}

pub(super) fn serving_rejection_summary(
    results: &[ScoredServingConfig],
) -> ServingRejectionSummary {
    let rejected_candidate_count = results.iter().filter(|score| !score.feasible).count();
    let mut summary = serving_rejection_summary_from_records(
        results
            .iter()
            .map(|score| (score.candidate_id.as_str(), score.rejections.as_slice())),
    );
    summary.rejected_candidate_count = rejected_candidate_count;
    summary
}

pub(super) fn serving_rejection_summary_from_records<'a, I>(records: I) -> ServingRejectionSummary
where
    I: IntoIterator<Item = (&'a str, &'a [ServingRejection])>,
{
    let mut candidate_with_rejections = BTreeSet::new();
    let mut groups = BTreeMap::<ServingRejectionGroupKey, ServingRejectionGroupAccumulator>::new();
    let mut total_rejection_count = 0_usize;

    for (candidate_id, rejections) in records {
        if !rejections.is_empty() {
            candidate_with_rejections.insert(candidate_id.to_string());
        }
        for rejection in rejections {
            total_rejection_count = total_rejection_count.saturating_add(1);
            let key = ServingRejectionGroupKey {
                phase: rejection.phase.clone(),
                category: rejection.category.clone(),
                resource: rejection.resource.clone(),
                code: rejection.code.clone(),
            };
            let group = groups.entry(key).or_default();
            group.rejection_count = group.rejection_count.saturating_add(1);
            group.candidate_ids.insert(candidate_id.to_string());
            if let Some(remediation) = rejection.remediation.as_ref() {
                group.remediations.insert(remediation.clone());
            }
            if !rejection.message.is_empty() {
                group.messages.insert(rejection.message.clone());
            }
        }
    }

    let mut groups = groups
        .into_iter()
        .map(|(key, group)| {
            let candidate_count = group.candidate_ids.len();
            let example_candidate_ids = group
                .candidate_ids
                .iter()
                .take(REJECTION_SUMMARY_EXAMPLE_LIMIT)
                .cloned()
                .collect::<Vec<_>>();
            let remediations = group
                .remediations
                .iter()
                .take(REJECTION_SUMMARY_EXAMPLE_LIMIT)
                .cloned()
                .collect::<Vec<_>>();
            let messages = group
                .messages
                .iter()
                .take(REJECTION_SUMMARY_EXAMPLE_LIMIT)
                .cloned()
                .collect::<Vec<_>>();
            ServingRejectionGroupSummary {
                key,
                candidate_count,
                rejection_count: group.rejection_count,
                example_candidate_ids,
                candidate_ids_truncated: candidate_count > REJECTION_SUMMARY_EXAMPLE_LIMIT,
                remediations,
                messages,
            }
        })
        .collect::<Vec<_>>();
    groups.sort_by(|left, right| {
        right
            .rejection_count
            .cmp(&left.rejection_count)
            .then_with(|| right.candidate_count.cmp(&left.candidate_count))
            .then_with(|| left.key.cmp(&right.key))
    });

    ServingRejectionSummary {
        rejected_candidate_count: candidate_with_rejections.len(),
        candidate_with_rejections_count: candidate_with_rejections.len(),
        total_rejection_count,
        groups,
    }
}
