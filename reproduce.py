"""Regenerate the study's analysis outputs from the saved experiment records.

Run this entry point with the analysis environment's Python interpreter:
    python reproduce.py

The pipeline runs per-model analysis (including R confidence intervals) and
exploratory diagnostics in that order. Each stage uses its existing
script and validators. A failed stage stops the pipeline with a nonzero exit code.
Existing analysis CSVs and figures are regenerated. Collection is a separate
workflow in experiment/run_experiment.py and is not part of this pipeline.
"""

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = ("numpy", "pandas", "scipy", "matplotlib", "jsonschema", "yaml")
STAGES = (
    ("Per-model analysis and R confidence intervals", "scripts/analyze_results.py"),
    ("Exploratory diagnostics", "scripts/analyze_diagnostics.py"),
)


# region Analysis pipeline

def main() -> int:
    """Validate the local setup, then run all analysis stages sequentially."""
    missing = [name for name in REQUIRED_MODULES if find_spec(name) is None]
    if missing:
        print(f"Missing analysis dependencies: {', '.join(missing)}", file=sys.stderr)
        print(
            "Install them with: python -m pip install -r requirements-analysis.txt",
            file=sys.stderr,
        )
        return 1

    for _, script in STAGES:
        if not (ROOT / script).is_file():
            print(f"Required analysis script not found: {script}", file=sys.stderr)
            return 1
    if not any((ROOT / "experiment" / "results").glob("*/*.json")):
        print("No saved experiment records found in experiment/results/.", file=sys.stderr)
        return 1

    print("Reusing saved experiment responses.", flush=True)
    print("Existing analysis CSVs and figures will be regenerated.", flush=True)
    for index, (label, script) in enumerate(STAGES, start=1):
        print(f"\n[{index}/{len(STAGES)}] {label}", flush=True)
        try:
            # Use the current environment and resolve paths from the repository.
            subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            print(f"Analysis stopped at: {label} (exit code {error.returncode}).", file=sys.stderr)
            return error.returncode if error.returncode > 0 else 1

    print("\nReproduction completed. Results are in analysis/outputs/.", flush=True)
    return 0

# endregion


if __name__ == "__main__":
    raise SystemExit(main())
