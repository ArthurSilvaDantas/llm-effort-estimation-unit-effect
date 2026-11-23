# Unit Effect on LLM-Generated Software Effort Estimates

A controlled experiment investigating whether the effort unit requested in the
prompt (**work-hours** vs. **workdays**) affects the magnitude of software
development effort estimates produced by general-purpose LLMs.

## Reproduce the results

The repository includes the collected responses. To regenerate all analyses and
figures, use Python 3.10 or newer on Linux or WSL. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python reproduce.py
```

After the environment is installed, reproduction requires only the last command.
The entry point runs both analysis stages in sequence and stops if either stage
fails. Existing analysis CSVs and figures are regenerated from the saved records.
An OpenRouter account, API key, or new model calls are not required for this path.

| Stage | Outputs |
|---|---|
| Per-model analysis | `analysis/outputs/`: valid pairs, normality, hypothesis tests, effects, and model figures. The effect table and ratio plot include pointwise two-sided 95% confidence intervals for R. |
| Exploratory diagnostics | `analysis/outputs/diagnostics/`: 16 diagnostic CSVs and a manifest. |

With the included data, the main analysis uses 767 valid pairs from 10 eligible
models. The confidence intervals are a post-collection addition and preserve the
original one-sided tests. See [the confidence-interval method](analysis/README_confidence_intervals.md)
and [the diagnostic definitions](analysis/README_diagnostics.md).

## Research question

> **RQ1**: For each evaluated model, does requesting the estimate in work-hours
> produce proportionally smaller total values than requesting it in workdays?

The question replicates, in LLMs, the framing effect already documented in
software professionals by Jørgensen & Halkjelsvik (2010).

## Experimental design

- **Factor**: requested effort unit, `T_WH` (work-hours) vs. `T_WD` (workdays), a crossover design paired by specification.
- **Experimental objects**: 88 software specifications (user stories plus project description), reused from the Çalıklı & Alhamed (2025) replication package, covering Spring XD, Hyperledger Fabric and Mule.
- **Unit of analysis**: the pair `(Y_WH, Y_WDh)` per specification and model, where `Y_WDh = Y_WD × C_im` (the work-hours-per-workday conversion factor reported by the model itself).
- **Models**: selected through purposive sampling from OpenRouter's monthly usage ranking, one representative per family. The preserved configuration lists 13 IDs; saved executions cover 11 models, and the primary analysis includes 10 eligible models.
- **Analysis**: a one-tailed paired t-test on the logarithmic differences, run separately per model, with Cohen's dz as effect size.

The full pre-registered protocol (constructs, prompts, JSON schemas, validation
criteria and analysis plan) is not included in this repository. It should be
preserved alongside the replication package separately.

## Repository layout

```
reproduce.py                Entry point for the complete analysis pipeline
config/models.yaml          Selected models, fixed provider, and execution config
data/specifications/        The 88 specifications (plain text, one file per spec)
data/raw/                   Original raw batches (Çalıklı & Alhamed 2025)
prompts/                    Prompt templates and JSON schemas (estimation and conversion)
pilot/                      Pilot study script and results (excluded from analysis)
experiment/
  run_experiment.py         Main collection runner
  randomize.py               Generates the T_WH-to-T_WD / T_WD-to-T_WH sequence per model
  randomization.json         Fixed sequence (seed 42), generated once per model before collection
  results/<model>/          One JSON per (specification, treatment) pair: raw and parsed data
  results_archive/           Pre-fix backups of executions reset for a retry, each folder
                              with a _manifest.json (timestamp, reason, file list)
scripts/
  openrouter_client.py       OpenRouter HTTP client (retry, error classification)
  prompt_builder.py          Builds prompts from the templates
  extract_specifications.py  Extracts specifications from the raw batches
  export_results_xlsx.py     Exports results to a spreadsheet
  analyze_results.py         Per-model tests, effects, R confidence intervals, and figures
  analyze_diagnostics.py     Exploratory diagnostics for the included models
```

## Collect new model responses

Collection runs the experiment against the external API and requires an
OpenRouter account and credits. It is separate from reproducing the analyses
of the saved responses above. Newly collected responses may differ from the
preserved responses because model availability and behavior can change.

Using the same virtual environment:

```bash
cp .env.example .env   # fill in OPENROUTER_API_KEY
```

Collect all models active in `config/models.yaml`:

```bash
.venv/bin/python experiment/run_experiment.py
```

Collect a single model (lets you run several in parallel, one process per model):

```bash
.venv/bin/python experiment/run_experiment.py --model <model_id>
```

Execution is incremental and resumable: each specification-treatment combination
is saved to disk as soon as it completes, at
`experiment/results/<model>/<spec>__<treatment>.json`. Re-running the same
command resumes from where it left off without re-collecting anything already
valid. Technical failures (network, parsing) automatically get a single retry,
following the protocol's retry policy; a response that was already received is
never collected again.

## Data validation

Each execution is validated against: JSON schema conformance, positive values,
`reported_total.most_likely_effort` equal to the sum of the activities, the
conversion factor `C_im` falling within `(0, 24]`, and correspondence to the
expected specification and treatment. A pair is valid only when both
corresponding executions (`T_WH` and `T_WD`, including the `C_im` collection)
are valid.
