# Effort analysis diagnostics

From the repository root, with the analysis virtual environment activated:

```bash
python scripts/analyze_diagnostics.py
```

If a dependency is missing, run `python -m pip install -r requirements-analysis.txt`.
The script reads existing executions in `experiment/results/` and reuses the
exporter's validators and pairing rules. **Every diagnostic report uses only
the models included in the main analysis**, as returned by
`analyze_results.load_valid_pairs()`. Models excluded by the existing eligibility
rule and configured models absent from that analysis do not enter any report,
count, denominator, or ranking. The current data contain 10 included models and
767 valid pairs. Quality and conversion reports also inspect invalid or missing
executions for these same models; effect reports use their valid pairs only.

Specification projects come from the original batch headers, identified by
project and batch number, in `data/raw/calikli_alhamed_2025_batches/`.
Reports are written to `analysis/outputs/diagnostics/`. When available, the main
analysis CSVs are checked against the reconstructed pairs and effects before
any report is written. A mismatch stops execution. Re-running the script replaces
the diagnostic reports, including any older reports that used a broader model
scope. It does not modify the main analysis outputs or experiment records.

## Reading the reports

| Indicator | CSV | Unit and interpretation |
|---|---|---|
| Q01, Q03, and part of Q06 | `quality_by_model.csv` | One row per main-analysis model. Coverage, pair classification, failed treatment, and median word counts for valid and invalid pairs. |
| Q02 | `invalid_reasons.csv` | One row per model, treatment, stage, and recorded reason. An execution may have multiple reasons; `n_invalid_executions_in_treatment` counts unique invalid executions. |
| Q04 | `invalid_specs.csv` | The ten specifications with the most invalid executions among main-analysis models. Ties use the number of affected models, then the specification ID. |
| Q05 and Q06 | `invalid_by_project.csv`, `invalid_by_length.csv` | Model counts and rates, stratified by project or word-count quartile. |
| E07 and E08 | `effect_distribution.csv` | Direction of WH relative to WD converted to hours, and the distribution of individual percentage deltas. |
| E09 | `extreme_specs.csv` | Up to five negative and five positive deltas per model, with differences in hours. Ties use the specification ID. |
| E10 | `leave_one_out_influence.csv` | Range of aggregate percentage differences when temporarily omitting one specification at a time. The largest absolute change is in percentage points. |
| E11 | `project_effects.csv` | Aggregate percentage difference and proportion with lower WH, by model and project. |
| E12 | `length_effect_association.csv` | Spearman coefficients between word count and signed or absolute delta, with average ranks for ties. Associations are `NA` with no variation or fewer than two pairs. |
| E13 | `story_overlap_summary.csv`, `story_overlap_pairs.csv` | Pairwise comparisons of the specification corpus used by the main-analysis models. Details list identical full specification texts or identical stories shared by specifications. |
| E14 | `conversion_factors.csv`, `conversion_factor_frequencies.csv` | Conversion-response validity and frequencies of valid factors. If multiple factors tie for the mode, the summary reports the smallest one and the number of modes. |
| E16 | `cross_model_patterns.csv` | Up to ten specifications for which the most models produced lower WH. Ties use the proportion among models with valid pairs, then the specification ID. |
| A20 | `order_audit.csv` | Planned order for main-analysis models only. Checks that each block has 44 specifications, without duplicates, overlap, or unknown IDs. |

For quality indicators, `n_planned` is the number of specifications in the
audited grid for each main-analysis model. The `analysis_scope` column is always
`included_in_main_analysis`. All reports use this same model scope.

Treatment validity comes from the existing validators. Q03 counts three mutually
exclusive categories among pairs with both treatments resolved: only WH invalid,
only WD invalid, or both invalid. Pairs with missing files enter
`n_missing_records`; pairs with existing files and a pending stage enter
`n_incomplete`. The `valid_rate_pct` denominator is the full planned grid;
`invalid_rate_pct` uses `n_resolved_pairs` (valid plus invalid pairs). Zero means
no occurrences; `NA` means unavailable, such as a rate with a zero denominator.
Terminal collection and processing failures use the logged error type. Recovered
attempt failures and conversions not yet requested are not invalid responses.

Text length counts Unicode letter and digit sequences in the full specification,
including the project description. Quartile cutoffs are calculated once over
the corpus; values equal to a cutoff enter the lower quartile. Exact cutoffs
appear in `manifest.json`. Word count measures text length without assuming
complexity or effort.

Each individual delta is `100 * (WH / WDh - 1)`, negative when WH is lower.
The aggregate percentage difference is `100 * (exp(mean(L)) - 1)`, where `L` is
the log difference from the canonical pairing. The consistency check compares
the ratio `R` and the absolute percentage difference with the main analysis,
whose `percent_difference` column is unsigned. In E07, equality uses the exact
paired hour values. In E16, `share_models_wh_lower` is a proportion from 0 to 1.

E13 normalizes text using Unicode NFC and collapsed whitespace, preserving case
and punctuation. Stories are delimited by the section header and `U1`–`U8`
markers; enclosing quotes are removed. These markers are local story positions
within a batch. A text match does not establish a shared issue identity or
statistical dependence.

E14 applies the conversion validator to WD records for the main-analysis models,
even when the WH–WD pair is invalid. Received conversion responses that fail
validation or processing enter `n_invalid_factors`. Collection failures without
a response, conversions not requested, and missing WD files have separate counts.
Frequency percentages use `n_valid_factors` as their denominator.

The manifest records the included model IDs, sources, counts, checks, and reports
without rows. All of its execution counts refer only to the included models.
E15 remains unavailable because no independent task-size measure is provided.
A20 does not certify observed attempt order, and A21 cannot be computed because
failed-attempt logs lack timestamps. These analyses are descriptive and
exploratory. Observed actual effort is unavailable for assessing accuracy.
