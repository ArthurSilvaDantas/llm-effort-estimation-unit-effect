# Unit Effect on LLM-Generated Software Effort Estimates

A controlled experiment investigating whether the effort unit requested in the
prompt (**work-hours** vs. **workdays**) affects the magnitude of software
development effort estimates produced by general-purpose LLMs.

## Research question

> **RQ1**: For each evaluated model, does requesting the estimate in work-hours
> produce proportionally smaller total values than requesting it in workdays?

The question replicates, in LLMs, the framing effect already documented in
software professionals by Jørgensen & Halkjelsvik (2010).

## Experimental design

- **Factor**: requested effort unit, `T_WH` (work-hours) vs. `T_WD` (workdays), a crossover design paired by specification.
- **Experimental objects**: 88 software specifications (user stories plus project description), reused from the Çalıklı & Alhamed (2025) replication package, covering Spring XD, Hyperledger Fabric and Mule.
- **Unit of analysis**: the pair `(Y_WH, Y_WDh)` per specification and model, where `Y_WDh = Y_WD × C_im` (the work-hours-per-workday conversion factor reported by the model itself).
- **Models**: up to 12 LLMs selected through purposive sampling from OpenRouter's monthly usage ranking, one representative per family.
- **Analysis**: a one-tailed paired t-test on the logarithmic differences, run separately per model, with Cohen's dz as effect size.

The full pre-registered protocol (constructs, prompts, JSON schemas, validation
criteria and analysis plan) is not included in this repository. It should be
preserved alongside the replication package separately.

## Repository layout

```
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
```

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in OPENROUTER_API_KEY
```

Collect all models active in `config/models.yaml`:

```bash
python3 experiment/run_experiment.py
```

Collect a single model (lets you run several in parallel, one process per model):

```bash
python3 experiment/run_experiment.py --model <model_id>
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
