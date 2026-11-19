"""
Describe existing results for main-analysis models (Q01–Q06, E07–E14, E16, A20).

Every report uses only the model IDs returned by the main analysis's valid-pair
loader. Reuse the existing validators and pairing rules without changing the
experiment or running additional hypothesis tests. Write diagnostic reports to
analysis/outputs/diagnostics/. See analysis/README_diagnostics.md for definitions,
denominators, and availability limits.

Usage: python scripts/analyze_diagnostics.py
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import yaml

from analyze_results import load_valid_pairs
from export_results_xlsx import (
    build_pairs_rows,
    get_conversion_factor,
    iter_results,
    validate_conversion,
    validate_estimation,
    validate_metadata,
)
from extract_specifications import DEST_DIR, PROJECT_PREFIX, SOURCE_DIR


ROOT = Path(__file__).resolve().parent.parent
PRIMARY_DIR = ROOT / "analysis" / "outputs"
OUTPUT_DIR = PRIMARY_DIR / "diagnostics"
MODELS_PATH = ROOT / "config" / "models.yaml"
ORDER_PATH = ROOT / "experiment" / "randomization.json"
TREATMENTS = ("WH", "WD")
PAIR_CLASSES = ("valid", "invalid", "missing_record", "incomplete")
FAILURE_CLASSES = ("only_wh_invalid", "only_wd_invalid", "both_invalid")
EXPECTED_ORDER_BLOCK_SIZE = 44  # Defined in experiment/randomize.py.
PAIR_METRICS = {
    "Y_WH": "wh_hours",
    "Y_WD": "wd_days",
    "C_im": "hours_per_day",
    "Y_WDh": "wdh_hours",
    "L_im": "log_ratio",
}
REPORTS = {
    "quality_by_model": ("Q01", "Q03", "Q06"),
    "invalid_reasons": ("Q02",),
    "invalid_specs": ("Q04",),
    "invalid_by_project": ("Q05",),
    "invalid_by_length": ("Q06",),
    "effect_distribution": ("E07", "E08"),
    "extreme_specs": ("E09",),
    "leave_one_out_influence": ("E10",),
    "project_effects": ("E11",),
    "length_effect_association": ("E12",),
    "story_overlap_summary": ("E13",),
    "story_overlap_pairs": ("E13",),
    "conversion_factors": ("E14",),
    "conversion_factor_frequencies": ("E14",),
    "cross_model_patterns": ("E16",),
    "order_audit": ("A20",),
}


# region Inputs and specification identification

def normalized_text(text: str) -> str:
    """Compare exact text while allowing Unicode and whitespace differences."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def story_texts(text: str) -> tuple[str, ...] | None:
    """Treat U1–U8 as local positions, not TAWOS issue identifiers."""
    _, marker, body = text.partition("The user stories (requirements) are:")
    matches = list(re.finditer(r"(?m)^[ \t]*U(\d+):[ \t]*", body))
    if not marker or [int(m[1]) for m in matches] != list(range(1, 9)):
        return None
    stories = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        story = body[match.end():end].strip()
        if story.startswith('"') and story.endswith('"'):
            story = story[1:-1]
        stories.append(normalized_text(story))
    return tuple(stories) if all(stories) else None


@dataclass(frozen=True)
class Specification:
    project: str
    words: int
    text: str
    stories: tuple[str, ...] | None


def load_specifications() -> dict[str, Specification]:
    projects = {}
    # Read the project from the original batch header, not the task wording.
    for path in sorted(SOURCE_DIR.glob("*.txt")):
        with path.open(encoding="utf-8-sig") as file:
            header = file.readline().strip()
        match = re.fullmatch(r"Project Name: (.+) Batch Number: (\d+)", header)
        if not match or match[1] not in PROJECT_PREFIX:
            raise ValueError(f"Unrecognized source metadata: {path.name}")
        spec_id = f"{PROJECT_PREFIX[match[1]]}-{int(match[2]):03d}"
        if spec_id in projects:
            raise ValueError(f"Multiple source batches for {spec_id}")
        projects[spec_id] = match[1]

    specs = {}
    for path in sorted(DEST_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8-sig")
        if path.stem not in projects or not text.strip():
            raise ValueError(f"Empty specification or unidentified source: {path.name}")
        specs[path.stem] = Specification(
            projects[path.stem], len(re.findall(r"[^\W_]+", text)),
            normalized_text(text), story_texts(text),
        )
    if not specs or specs.keys() != projects.keys():
        raise ValueError("Source batches and extracted specifications do not match.")
    return specs


def load_analysis_model_ids() -> list[str]:
    """Use the main analysis's model set for every diagnostic report."""
    config = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    configured_ids = [model["id"] for model in config["models"]]
    if not configured_ids or len(set(configured_ids)) != len(configured_ids):
        raise ValueError("The configuration must list unique model IDs.")

    # Reuse the selection itself so that quality, conversion, and order reports
    # cannot include models excluded from the main analysis or absent from it.
    model_ids = sorted(load_valid_pairs()["model_id"].unique())
    if not model_ids:
        raise ValueError("The main analysis has no models with valid pairs.")
    if not set(model_ids).issubset(configured_ids):
        raise ValueError("Main-analysis model IDs are missing from the configuration.")
    return model_ids


def load_analysis_records(model_ids):
    """Filter source records before building any diagnostic or denominator."""
    model_dirs = {model_id.replace("/", "__") for model_id in model_ids}
    return [record for record in iter_results() if record["_source_model_dir"] in model_dirs]


def index_records(records, model_ids, specs):
    """Assign metadata failures to expected file slots without repairing the JSON."""
    model_by_dir = {model.replace("/", "__"): model for model in model_ids}
    if len(model_by_dir) != len(model_ids):
        raise ValueError("Model IDs do not produce unique directory names.")
    indexed = {}
    declared_keys = set()
    for record in records:
        model_id = model_by_dir.get(record["_source_model_dir"])
        spec_id = record["_source_spec_id"]
        treatment = record["_source_treatment"]
        key = model_id, spec_id, treatment
        if model_id is None or spec_id not in specs or treatment not in TREATMENTS:
            raise ValueError(f"Execution file outside the recognized grid: {key}")
        declared = tuple(record.get(field, "") for field in ("model_id", "spec_id", "treatment"))
        if key in indexed or declared in declared_keys:
            raise ValueError(f"Duplicate execution: {key}, declared identifiers {declared}")
        indexed[key] = record
        declared_keys.add(declared)
    return indexed

# endregion


# region Execution diagnostics and paired grid construction

@dataclass(frozen=True, order=True)
class Failure:
    stage: str
    reason: str
    error_type: str = ""


@dataclass(frozen=True)
class Execution:
    status: str
    failures: tuple[Failure, ...] = ()
    factor_status: str | None = None
    factor: float | None = None


def stage_state(record, stage):
    return (record.get("collection") or {}).get(stage) or {}


def stage_pending(record, stage):
    state = stage_state(record, stage).get("status")
    processing = (record.get("processing") or {}).get(stage)
    return (
        stage not in record
        and state != "failed" and processing != "failed"
        and (state == "pending" or processing == "pending")
    )


def logged_failure(record, stage, log, reason):
    # Use terminal failures only; a recovered attempt does not invalidate a response.
    for event in reversed(record.get(log) or []):
        if event.get("stage") == stage:
            return Failure(stage, reason, event.get("error_type") or "")
    return Failure(stage, reason)


def stage_failures(record, stage, checks):
    payload = record.get(stage)
    if not isinstance(payload, dict):
        if stage_state(record, stage).get("status") == "failed":
            return [logged_failure(record, stage, "attempt_log", "collection_failed")]
        if (record.get("processing") or {}).get(stage) == "failed":
            return [logged_failure(record, stage, "processing_log", "processing_failed")]
        reason = "parsed_payload_missing" if payload is None else "parsed_payload_not_object"
        return [Failure(stage, reason)]

    reasons = []
    if not checks["schema_valid"]:
        reasons.append(Failure(stage, "schema_invalid"))
    if stage == "estimation":
        if not checks["positive_efforts"]:
            reasons.append(Failure(stage, "positive_efforts_not_validated"))
        elif not checks["total_matches_activities"]:
            reasons.append(Failure(stage, "total_does_not_match_activities"))
    elif get_conversion_factor(record) is None:
        reasons.append(Failure(stage, "factor_missing_or_non_numeric"))
    elif not checks["within_bounds"]:
        reasons.append(Failure(stage, "factor_out_of_bounds"))
    return reasons or [Failure(stage, "validator_rejected_without_detail")]


def conversion_status(record, validation):
    """Distinguish a rejected factor from a conversion without a response."""
    state = stage_state(record, "conversion")
    if validation["valid"]:
        return "valid", get_conversion_factor(record)
    received = state.get("status") == "received" or any(
        key in record for key in ("conversion", "conversion_raw", "conversion_response")
    )
    if stage_pending(record, "conversion"):
        if not received and state.get("status") == "pending" and state.get("attempts") == 0:
            return "not_requested", None
        return "pending_or_unknown", None
    if received:
        return "invalid", None
    if state.get("status") == "failed":
        return "failed_without_response", None
    return "pending_or_unknown", None


def inspect_execution(record, treatment):
    wd = treatment == "WD"
    if record is None:
        return Execution("missing_record", factor_status="missing_record" if wd else None)
    if validate_metadata(record)["valid"] is not True:
        return Execution(
            "invalid", (Failure("metadata", "metadata_mismatch"),),
            factor_status="metadata_invalid" if wd else None,
        )

    stages = {"estimation": validate_estimation(record)}
    factor_status, factor = None, None
    if wd:
        stages["conversion"] = validate_conversion(record)
        factor_status, factor = conversion_status(record, stages["conversion"])

    failures, pending = [], False
    for stage, checks in stages.items():
        if checks["valid"]:
            continue
        if stage_pending(record, stage):
            pending = True
        else:
            failures.extend(stage_failures(record, stage, checks))
    status = "invalid" if failures else "pending" if pending else "valid"
    return Execution(status, tuple(sorted(set(failures))), factor_status, factor)


def classify_pair(wh, wd):
    if "missing_record" in (wh, wd):
        return "missing_record", None
    if "pending" in (wh, wd):
        return "incomplete", None
    if wh == wd == "valid":
        return "valid", None
    if wh == wd == "invalid":
        return "invalid", "both_invalid"
    return "invalid", "only_wh_invalid" if wh == "invalid" else "only_wd_invalid"


def build_diagnostic_pairs(model_ids, specs, records):
    indexed = index_records(records, model_ids, specs)
    executions = {
        (model_id, spec_id, treatment): inspect_execution(
            indexed.get((model_id, spec_id, treatment)), treatment,
        )
        for model_id in model_ids for spec_id in specs for treatment in TREATMENTS
    }
    # Take effect values, including Y_WDh and L_im, from the canonical pairing.
    headers, raw_pairs = build_pairs_rows(records)
    canonical = {}
    for raw_pair in raw_pairs:
        pair = dict(zip(headers, raw_pair))
        if pair["pair_complete"]:
            canonical[pair["model_id"], pair["spec_id"]] = pair

    cuts = np.quantile([spec.words for spec in specs.values()], (0.25, 0.5, 0.75))
    rows = []
    for model_id in model_ids:
        for spec_id, spec in specs.items():
            wh = executions[model_id, spec_id, "WH"].status
            wd = executions[model_id, spec_id, "WD"].status
            pair_status, failure_class = classify_pair(wh, wd)
            original = canonical.get((model_id, spec_id))
            if (pair_status == "valid") != (original is not None):
                raise ValueError(f"Pairing-rule mismatch for {model_id}/{spec_id}")
            metrics = {col: original[key] if original else None for key, col in PAIR_METRICS.items()}
            delta, difference = None, None
            if original is not None:
                wh_hours, wdh_hours = original["Y_WH"], original["Y_WDh"]
                if not all(math.isfinite(x) and x > 0 for x in (wh_hours, wdh_hours)):
                    raise ValueError(f"Valid pair with nonfinite or nonpositive effort: {model_id}/{spec_id}")
                delta = 100 * (wh_hours / wdh_hours - 1)
                difference = wh_hours - wdh_hours
                if not math.isfinite(delta) or not math.isfinite(original["L_im"]):
                    raise ValueError(f"Nonfinite contrast: {model_id}/{spec_id}")
            rows.append({
                "model_id": model_id, "spec_id": spec_id, "project": spec.project,
                "spec_word_count": spec.words, "length_quartile": int(np.searchsorted(cuts, spec.words)) + 1,
                "wh_status": wh, "wd_status": wd, "pair_status": pair_status, "failure_class": failure_class,
                **metrics, "delta_pct": delta, "diff_hours": difference,
            })
    pairs = pd.DataFrame(rows)
    if len(canonical) != pairs["pair_status"].eq("valid").sum():
        raise ValueError("The valid-pair count differs from the canonical pairing.")
    return pairs, executions, [float(cut) for cut in cuts]

# endregion


# region Data quality (Q01–Q06)

def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def numeric_summary(value):
    return float(value) if pd.notna(value) else None


def pair_counts(frame):
    counts = frame["pair_status"].value_counts()
    n_valid, n_invalid, n_missing, n_incomplete = (int(counts.get(k, 0)) for k in PAIR_CLASSES)
    if n_valid + n_invalid + n_missing + n_incomplete != len(frame):
        raise ValueError("Grid classifications do not sum to the planned pair count.")
    return {
        "n_planned": len(frame), "n_valid": n_valid, "n_invalid": n_invalid,
        "n_missing_records": n_missing, "n_incomplete": n_incomplete,
        "n_resolved_pairs": n_valid + n_invalid,
        "invalid_rate_pct": percent(n_invalid, n_valid + n_invalid),
    }


def quality_by_model(pairs):
    rows = []
    for model_id, group in pairs.groupby("model_id", sort=True):
        counts = pair_counts(group)
        failure_counts = group["failure_class"].value_counts()
        failures = {k: int(failure_counts.get(k, 0)) for k in FAILURE_CLASSES}
        if sum(failures.values()) != counts["n_invalid"]:
            raise ValueError(f"Failure classifications do not sum to the invalid pairs for {model_id}")
        n_records = int(group["wh_status"].ne("missing_record").sum() + group["wd_status"].ne("missing_record").sum())
        rows.append({
            "model_id": model_id, "analysis_scope": "included_in_main_analysis",
            "n_recorded_executions": n_records, **counts,
            "valid_rate_pct": percent(counts["n_valid"], counts["n_planned"]), **failures,
            "median_words_valid_specs": numeric_summary(group.loc[group["pair_status"].eq("valid"), "spec_word_count"].median()),
            "median_words_invalid_specs": numeric_summary(group.loc[group["pair_status"].eq("invalid"), "spec_word_count"].median()),
        })
    return rows


def invalid_reasons(executions):
    denominators, reasons = Counter(), Counter()
    for (model_id, _, treatment), execution in executions.items():
        if execution.status == "invalid":
            denominators[model_id, treatment] += 1
            reasons.update((model_id, treatment, failure) for failure in execution.failures)
    return [
        {"model_id": model_id, "treatment": treatment, "stage": failure.stage,
         "invalid_reason": failure.reason, "logged_error_type": failure.error_type or None,
         "count": count, "n_invalid_executions_in_treatment": denominators[model_id, treatment]}
        for (model_id, treatment, failure), count in sorted(reasons.items())
    ]


def rank_invalid_specs(pairs):
    rows = []
    for (spec_id, project), group in pairs.groupby(["spec_id", "project"], sort=True):
        wh, wd = group["wh_status"].eq("invalid"), group["wd_status"].eq("invalid")
        n_invalid = int(wh.sum() + wd.sum())
        if n_invalid:
            rows.append({
                "spec_id": spec_id, "project": project,
                "n_recorded_executions": int(group["wh_status"].ne("missing_record").sum() + group["wd_status"].ne("missing_record").sum()),
                "total_invalid_executions": n_invalid, "affected_models": int((wh | wd).sum()),
                "invalid_wh": int(wh.sum()), "invalid_wd": int(wd.sum()),
            })
    rows.sort(key=lambda row: (-row["total_invalid_executions"], -row["affected_models"], row["spec_id"]))
    return [{"rank": rank, **row} for rank, row in enumerate(rows[:10], 1)]


def quality_by_stratum(pairs, stratum, strata):
    # Keep empty strata: zero resolved pairs means an unavailable invalid rate.
    return [
        {"model_id": model_id, stratum: value, **pair_counts(group.loc[group[stratum].eq(value)])}
        for model_id, group in pairs.groupby("model_id", sort=True) for value in strata
    ]

# endregion


# region Main-analysis pair effects (E07–E12, E16)

def aggregate_effect(mean_log_ratio):
    return 100 * math.expm1(mean_log_ratio)


def effect_distribution(group):
    delta = group["delta_pct"]
    # Compare canonical hours exactly, without introducing a tolerance for ties.
    lower = int(group["wh_hours"].lt(group["wdh_hours"]).sum())
    equal = int(group["wh_hours"].eq(group["wdh_hours"]).sum())
    higher = int(group["wh_hours"].gt(group["wdh_hours"]).sum())
    n = len(group)
    if lower + equal + higher != n or delta.isna().any():
        raise ValueError("Valid pairs must have a defined direction and delta.")
    return {
        "n_valid": n, "n_wh_lower": lower, "n_equal": equal, "n_wh_higher": higher,
        "pct_wh_lower": percent(lower, n),
        "median_delta_pct": float(delta.median()), "q25_delta_pct": float(delta.quantile(0.25)),
        "q75_delta_pct": float(delta.quantile(0.75)),
        "min_delta_pct": float(delta.min()), "max_delta_pct": float(delta.max()),
    }


def extreme_specs(model_id, group):
    rows = []
    for direction, mask, ascending in (
        ("negative", group["delta_pct"].lt(0), True),
        ("positive", group["delta_pct"].gt(0), False),
    ):
        candidates = group.loc[mask].sort_values(["delta_pct", "spec_id"], ascending=[ascending, True])
        for rank, row in enumerate(candidates.head(5).itertuples(index=False), 1):
            rows.append({
                "model_id": model_id, "direction": direction, "rank": rank,
                "n_valid": len(group), "n_in_direction": len(candidates),
                "spec_id": row.spec_id, "project": row.project,
                "wh_hours": row.wh_hours, "wdh_hours": row.wdh_hours,
                "diff_hours": row.diff_hours, "delta_pct": row.delta_pct,
            })
    return rows


def leave_one_out(group):
    values = group["log_ratio"].to_numpy(dtype=float)
    n = len(values)
    total = math.fsum(values)
    original = aggregate_effect(total / n)
    result = {"n_valid": n, "original_effect_pct": original, "n_loo": 0}
    if n < 2:
        return {**result, "min_loo_effect_pct": None, "max_loo_effect_pct": None,
                "most_influential_spec_id": None, "max_abs_change_pp": None}
    # Subtracting one log ratio recomputes the mean without that specification.
    recomputed = [aggregate_effect((total - value) / (n - 1)) for value in values]
    changes = [abs(value - original) for value in recomputed]
    specs = group["spec_id"].to_list()
    most = min(range(n), key=lambda index: (-changes[index], specs[index]))
    return {**result, "n_loo": n, "min_loo_effect_pct": min(recomputed), "max_loo_effect_pct": max(recomputed),
            "most_influential_spec_id": specs[most], "max_abs_change_pp": changes[most]}


def project_effects(model_id, group, projects):
    rows = []
    for project in projects:
        subgroup = group.loc[group["project"].eq(project)]
        n = len(subgroup)
        lower = int(subgroup["wh_hours"].lt(subgroup["wdh_hours"]).sum())
        rows.append({
            "model_id": model_id, "project": project, "n_valid": n,
            "subgroup_effect_pct": aggregate_effect(subgroup["log_ratio"].mean()) if n else None,
            "n_wh_lower": lower, "pct_wh_lower": percent(lower, n),
        })
    return rows


def spearman_coefficient(x, y):
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None
    # Spearman correlates ranks; tied values receive their average rank.
    return float(x.rank(method="average").corr(y.rank(method="average")))


def length_effect_association(group):
    words, delta = group["spec_word_count"], group["delta_pct"]
    return {
        "n_valid": len(group), "n_distinct_word_counts": words.nunique(),
        "spearman_words_signed_delta": spearman_coefficient(words, delta),
        "spearman_words_abs_delta": spearman_coefficient(words, delta.abs()),
    }


def cross_model_patterns(valid):
    rows = []
    n_models = valid["model_id"].nunique()
    for (spec_id, project), group in valid.groupby(["spec_id", "project"], sort=True):
        n = len(group)
        lower = int(group["wh_hours"].lt(group["wdh_hours"]).sum())
        rows.append({
            "spec_id": spec_id, "project": project, "n_models_in_analysis": n_models,
            "n_models_valid": n, "n_models_wh_lower": lower,
            "share_models_wh_lower": lower / n, "median_abs_delta_pct": float(group["delta_pct"].abs().median()),
        })
    rows.sort(key=lambda row: (-row["n_models_wh_lower"], -row["share_models_wh_lower"], row["spec_id"]))
    return [{"rank": rank, **row} for rank, row in enumerate(rows[:10], 1)]

# endregion


# region Text overlap, conversions, and planned order (E13, E14, A20)

def story_overlap(specs):
    detail = []
    full_matches, story_matches = [], []
    story_sets = {spec_id: set(spec.stories) for spec_id, spec in specs.items() if spec.stories is not None}
    for a, b in combinations(sorted(specs), 2):
        same_spec = specs[a].text == specs[b].text
        shared = len(story_sets[a] & story_sets[b]) if a in story_sets and b in story_sets else None
        if same_spec:
            full_matches.append((a, b))
        if shared:
            story_matches.append((a, b, shared))
        if same_spec or shared:
            detail.append({"spec_id_a": a, "project_a": specs[a].project,
                           "spec_id_b": b, "project_b": specs[b].project,
                           "identical_spec_text": same_spec, "n_shared_story_texts": shared})

    summary = []
    for method, n_checked, matches, max_shared in (
        ("exact_normalized_spec_text", len(specs), full_matches, None),
        ("exact_normalized_story_text", len(story_sets), story_matches,
         max((row[2] for row in story_matches), default=0) if len(story_sets) >= 2 else None),
    ):
        summary.append({
            "method": method, "n_specs_in_corpus": len(specs), "n_specs_checked": n_checked,
            "n_possible_spec_pairs": math.comb(n_checked, 2),
            "n_specs_with_overlap": len({spec_id for row in matches for spec_id in row[:2]}) if n_checked >= 2 else None,
            "n_pairs_of_specs_overlapping": len(matches) if n_checked >= 2 else None,
            "max_shared_stories": max_shared,
        })
    return summary, detail


def conversion_reports(executions):
    by_model = defaultdict(list)
    for (model_id, _, treatment), execution in executions.items():
        if treatment == "WD":
            by_model[model_id].append(execution)
    summary, frequencies = [], []
    for model_id, group in sorted(by_model.items()):
        states = Counter(entry.factor_status for entry in group)
        values = Counter(entry.factor for entry in group if entry.factor_status == "valid")
        n_valid = states["valid"]
        mode_count = max(values.values(), default=0)
        modes = sorted(value for value, n in values.items() if n == mode_count)
        summary.append({
            "model_id": model_id, "n_wd_planned": len(group), "n_valid_factors": n_valid,
            "n_invalid_factors": states["invalid"], "n_failed_without_response": states["failed_without_response"],
            "n_not_requested": states["not_requested"], "n_pending_or_unknown": states["pending_or_unknown"],
            "n_metadata_invalid": states["metadata_invalid"], "n_missing_wd_records": states["missing_record"],
            "mode_hours_per_day": modes[0] if modes else None, "n_modes": len(modes),
            "mode_share_pct": percent(mode_count, n_valid), "min_hours_per_day": min(values, default=None),
            "max_hours_per_day": max(values, default=None),
        })
        for value, n in sorted(values.items()):
            frequencies.append({"model_id": model_id, "hours_per_day": value, "count": n,
                                "n_valid_factors": n_valid, "frequency_pct": percent(n, n_valid)})
    return summary, frequencies


def audit_planned_order(model_ids, specs):
    if not ORDER_PATH.is_file():
        return []
    orders = json.loads(ORDER_PATH.read_text(encoding="utf-8"))["models"]
    known = set(specs)
    rows = []
    for model_id in model_ids:
        plan = orders.get(model_id)
        wh = (plan or {}).get("block_WH_first", [])
        wd = (plan or {}).get("block_WD_first", [])
        wh_set, wd_set = set(wh), set(wd)
        n_wh = len((wh_set - wd_set) & known)
        n_wd = len((wd_set - wh_set) & known)
        n_unknown = len(known) - n_wh - n_wd
        duplicates = len(wh) + len(wd) - len(wh_set) - len(wd_set)
        outside = len((wh_set | wd_set) - known)
        rows.append({
            "model_id": model_id, "n_planned": len(known), "randomization_available": plan is not None,
            "n_wh_first_planned": n_wh if plan is not None else None,
            "n_wd_first_planned": n_wd if plan is not None else None,
            "n_unknown": n_unknown,
            "n_duplicate_assignments_within_block": duplicates if plan is not None else None,
            "n_specs_in_both_blocks": len(wh_set & wd_set) if plan is not None else None,
            "n_unknown_spec_ids_in_blocks": outside if plan is not None else None,
            "conforms_to_planned_balance": (
                n_wh == n_wd == EXPECTED_ORDER_BLOCK_SIZE and n_unknown == duplicates == outside == 0
            ) if plan is not None else None,
        })
    return rows

# endregion


# region Consistency checks and output

def check_primary_outputs(valid):
    """Compare existing outputs without running additional inferential tests."""
    checks = {}
    keys = ["model_id", "spec_id"]
    path = PRIMARY_DIR / "valid_pairs.csv"
    if path.is_file():
        expected = pd.read_csv(path)[keys + list(PAIR_METRICS)].rename(columns=PAIR_METRICS)
        current = valid[keys + list(PAIR_METRICS.values())]
        pd.testing.assert_frame_equal(
            current.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True), check_dtype=False,
            check_exact=False, rtol=1e-10, atol=1e-10,
        )
        checks["valid_pairs_csv"] = "matched"
    else:
        checks["valid_pairs_csv"] = "not_available"

    path = PRIMARY_DIR / "effects.csv"
    if path.is_file():
        published = pd.read_csv(path).set_index("model_id")
        if not published.index.is_unique:
            raise ValueError("Duplicate model IDs in effects.csv.")
        groups = valid.groupby("model_id", sort=True)["log_ratio"]
        n = groups.size()
        if n.to_dict() != published["n_pairs"].to_dict():
            raise ValueError("Model pair counts differ from effects.csv.")
        ratios = np.exp(groups.mean())
        signed = (ratios - 1) * 100
        effects = published.loc[n.index]
        np.testing.assert_allclose(ratios, effects["R"], rtol=1e-10, atol=1e-10)
        # The primary CSV stores an unsigned percentage; R already checks direction.
        np.testing.assert_allclose(signed.abs(), effects["percent_difference"], rtol=1e-10, atol=1e-10)
        checks["effects_csv"] = "matched"
    else:
        checks["effects_csv"] = "not_available"
    return checks


def build_reports(pairs, executions, specs, model_ids, valid):
    projects = sorted({spec.project for spec in specs.values()})
    reports = {
        "quality_by_model": quality_by_model(pairs),
        "invalid_reasons": invalid_reasons(executions),
        "invalid_specs": rank_invalid_specs(pairs),
        "invalid_by_project": quality_by_stratum(pairs, "project", projects),
        "invalid_by_length": quality_by_stratum(pairs, "length_quartile", range(1, 5)),
        "effect_distribution": [], "extreme_specs": [], "leave_one_out_influence": [],
        "project_effects": [], "length_effect_association": [],
        "cross_model_patterns": cross_model_patterns(valid),
        "order_audit": audit_planned_order(model_ids, specs),
    }
    for model_id, group in valid.groupby("model_id", sort=True):
        reports["effect_distribution"].append({"model_id": model_id, **effect_distribution(group)})
        reports["extreme_specs"].extend(extreme_specs(model_id, group))
        reports["leave_one_out_influence"].append({"model_id": model_id, **leave_one_out(group)})
        reports["project_effects"].extend(project_effects(model_id, group, projects))
        reports["length_effect_association"].append({"model_id": model_id, **length_effect_association(group)})
    reports["story_overlap_summary"], reports["story_overlap_pairs"] = story_overlap(specs)
    reports["conversion_factors"], reports["conversion_factor_frequencies"] = conversion_reports(executions)
    return reports


def write_reports(reports, manifest):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest["reports"] = {}
    for name, indicators in REPORTS.items():
        rows = reports[name]
        path = OUTPUT_DIR / f"{name}.csv"
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False, na_rep="NA", lineterminator="\n")
        else:
            # Remove only a known report that no longer has rows.
            path.unlink(missing_ok=True)
        manifest["reports"][path.name] = {
            "indicators": list(indicators), "rows": len(rows), "written": bool(rows),
        }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def main():
    specs, model_ids = load_specifications(), load_analysis_model_ids()
    records = load_analysis_records(model_ids)
    pairs, executions, cuts = build_diagnostic_pairs(model_ids, specs, records)
    valid = pairs.loc[pairs["pair_status"].eq("valid")]
    if set(valid["model_id"]) != set(model_ids):
        raise ValueError("Diagnostic models differ from the main-analysis model set.")
    checks = check_primary_outputs(valid)
    reports = build_reports(pairs, executions, specs, model_ids, valid)
    quality = reports["quality_by_model"]
    manifest = {
        "analysis_type": "descriptive and exploratory, post-collection",
        "model_scope": "Only models included in the main analysis; applies to every report.",
        "model_selection": "Model IDs returned by analyze_results.load_valid_pairs().",
        "model_ids": model_ids,
        "sources": [
            "experiment/results/", "config/models.yaml", "data/specifications/",
            "data/raw/calikli_alhamed_2025_batches/", "experiment/randomization.json",
            "analysis/outputs/valid_pairs.csv", "analysis/outputs/effects.csv",
        ],
        "n_specs": len(specs),
        "project_counts": dict(sorted(Counter(spec.project for spec in specs.values()).items())),
        "n_models_in_diagnostics": len(model_ids), "n_raw_records": len(records),
        "n_models_with_records": sum(row["n_recorded_executions"] > 0 for row in quality),
        "n_models_in_main_analysis": valid["model_id"].nunique(), "n_valid_pairs_in_main_analysis": len(valid),
        "primary_output_checks": checks,
        "word_count": {"method": "Unicode letter and digit sequences in the full specification text",
                       "quartile_cutoffs": cuts, "boundary_rule": "Values equal to a cutoff enter the lower quartile."},
        "not_computed": {
            "E15": "independent_size_metric_unavailable",
            "A20_observed_order": "Randomization records planned order, not observed attempt order.",
            "A21": "Failed-attempt logs do not record timestamps.",
            "E13_story_ids": "U1–U8 are local positions; comparisons use exact normalized text only.",
        },
        "definitions": "analysis/README_diagnostics.md",
    }
    if not reports["order_audit"]:
        manifest["not_computed"]["A20_planned_order"] = "randomization_file_unavailable"
    write_reports(reports, manifest)
    n_reports = sum(bool(rows) for rows in reports.values())
    print(f"{n_reports} diagnostic CSVs and manifest.json saved to {OUTPUT_DIR.relative_to(ROOT)}/")
    print(f"Main-analysis valid pairs: {len(valid)} from {valid['model_id'].nunique()} models.")
    print(f"Consistency checks against existing outputs: {checks}")

# endregion


if __name__ == "__main__":
    main()
