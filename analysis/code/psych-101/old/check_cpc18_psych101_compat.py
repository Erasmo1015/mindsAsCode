#!/usr/bin/env python3
"""
Check whether CPC18 can be loaded from Psych-101-test instead of local Track II CSVs
without changing utils/tools/collect_participant_ids.py or Template_evo_non_strict.py.

Current CPC18 source (confirmed):
  - datasets/cpc18/raw-comp-set-data-Track-2.csv  (trial-level train data)
  - datasets/cpc18/Data-to-predict-Track-2.csv    (block B-rates for official MSE)
  - data_modules/cpc18.load_cpc18_track2_data()

Psych-101 CPC18 experiment:
  - plonsky2018when/exp1.csv  (Centaur paper "CPC18"; not a separate HF column name)

Writes a machine-readable report to analysis/data/psych-101/ and prints a short verdict.

Example:
  python analysis/code/psych-101/check_cpc18_psych101_compat.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "data" / "psych-101"
CPC18_EXPERIMENT = "plonsky2018when/exp1.csv"
LOCAL_CPC18_DIR = REPO_ROOT / "datasets" / "cpc18"
LOCAL_RAW = LOCAL_CPC18_DIR / "raw-comp-set-data-Track-2.csv"
LOCAL_TARGETS = LOCAL_CPC18_DIR / "Data-to-predict-Track-2.csv"
PSYCH_TEST_HF = "marcelbinz/Psych-101-test"
PSYCH_TRAIN_HF = "marcelbinz/Psych-101"


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_psych_split(hf_id: str, split_name: str):
    from datasets import load_dataset

    tok = _hf_token()
    kw = {"token": tok} if tok else {}
    ds = load_dataset(hf_id, **kw)
    if split_name not in ds:
        raise ValueError(f"{hf_id}: missing split {split_name!r}; have {list(ds.keys())}")
    return ds[split_name].filter(lambda ex: ex["experiment"] == CPC18_EXPERIMENT)


def _local_subj_ids() -> List[str]:
    ids: Set[str] = set()
    with open(LOCAL_RAW, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(str(int(row["SubjID"])))
    return sorted(ids, key=int)


def _local_per_participant_stats() -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(LOCAL_RAW)
    per = df.groupby("SubjID").agg(
        n_trials=("Trial", "count"),
        n_games=("GameID", "nunique"),
    )
    return {
        "n_participants": int(per.shape[0]),
        "trials_per_participant": sorted(per["n_trials"].unique().tolist()),
        "games_per_participant": sorted(per["n_games"].unique().tolist()),
        "trials_per_game": sorted(df.groupby(["SubjID", "GameID"]).size().unique().tolist()),
    }


def _psych_participant_stats(split_ds) -> Dict[str, Any]:
    participants = sorted(set(split_ds["participant"]), key=lambda x: int(x))
    press_counts: List[int] = []
    problem_counts: List[int] = []
    for row in split_ds:
        text = row["text"]
        press_counts.append(len(re.findall(r"You press <<[A-Z]>>", text)))
        problem_counts.append(len(re.findall(r"\n\nOption [A-Z] delivers", text)))
    return {
        "n_participants": len(participants),
        "participant_ids_sample": participants[:10],
        "participant_ids_min": participants[0] if participants else None,
        "participant_ids_max": participants[-1] if participants else None,
        "presses_per_row": {
            "min": min(press_counts) if press_counts else None,
            "max": max(press_counts) if press_counts else None,
            "unique": sorted(set(press_counts)),
        },
        "problems_per_row": {
            "min": min(problem_counts) if problem_counts else None,
            "max": max(problem_counts) if problem_counts else None,
            "unique": sorted(set(problem_counts)),
        },
    }


def _code_touch_points() -> Dict[str, Any]:
    """Files/lines that assume local CSV loading (focus: user-requested scripts)."""
    return {
        "data_modules_cpc18.py": {
            "role": "Primary loader; must change or add parallel Psych-101 loader here first.",
            "functions": [
                "load_cpc18_track2_data(data_path, participant_id) -> ParticipantData",
                "split_cpc18_trials / split_cpc18_trials_three_way",
            ],
            "hardcoded_files": [
                "raw-comp-set-data-Track-2.csv",
                "Data-to-predict-Track-2.csv",
            ],
        },
        "utils/tools/collect_participant_ids.py": {
            "modification_needed": True,
            "sites": [
                {
                    "function": "collect_cpc18",
                    "lines": "86-111",
                    "behavior": "Reads SubjID from raw-comp-set-data-Track-2.csv; calls load_cpc18_track2_data + split_cpc18_trials(cpc18_official_mse=True).",
                },
                {
                    "function": "main (cpc18 branch)",
                    "lines": "~265-277",
                    "behavior": "Default --data_path datasets/cpc18; writes valid_participant_ids.json with local id semantics.",
                },
            ],
            "note": "No Psych-101 / HF flags today (unlike collect_choice13k which uses --hf_dataset).",
        },
        "Template_evo_non_strict.py": {
            "modification_needed": True,
            "sites": [
                {
                    "function": "_load_loglik_train_val_test_trials",
                    "lines": "1798-1805",
                    "behavior": "cpc18 branch: load_cpc18_track2_data + split_cpc18_trials_three_way.",
                },
                {
                    "function": "run_template_evolution_loop (cpc18 branch)",
                    "lines": "4409-4454",
                    "behavior": "load_cpc18_track2_data; official MSE uses Data-to-predict block targets.",
                },
            ],
            "note": "No --local_dataset / HF path for cpc18 (choice13k uses get_choice13k_experiments from Psych-101-test).",
        },
        "other_repo_callers": [
            "baseline_methods/Centaur.py",
            "baseline_methods/prospect_theory.py",
            "baseline_methods/MLE.py",
            "te_dr.py",
            "te_aggregate.py",
            "Template_evo.py",
            "baseline_methods/run_openevolve.py",
        ],
    }


def _build_verdict(checks: Dict[str, Any]) -> Dict[str, Any]:
    blockers: List[str] = []

    if checks["participant_id_overlap"]["local_subj_vs_psych_test"] == 0:
        blockers.append(
            "Psych-101-test participant ids do not match local SubjID values; "
            "valid_participant_ids.json ordinals would not refer to the same people."
        )
    if checks["participant_id_overlap"]["local_subj_vs_psych_train"] == 0:
        blockers.append(
            "Psych-101 train participant ids are 0..215 ordinals, not SubjID; "
            "local CSV cohort is disjoint."
        )

    local_games = checks["local_csv"]["games_per_participant"]
    psych_probs = checks["psych_test"]["problems_per_row"]["unique"]
    if local_games != psych_probs:
        blockers.append(
            f"Problem count mismatch: local CSV has {local_games} games/participant; "
            f"Psych-101-test has {psych_probs} problems/participant."
        )

    local_trials = checks["local_csv"]["trials_per_participant"]
    psych_presses = checks["psych_test"]["presses_per_row"]["unique"]
    if local_trials != psych_presses:
        blockers.append(
            f"Trial count mismatch: local {local_trials} vs Psych-101 presses {psych_presses} per participant."
        )

    if not LOCAL_TARGETS.is_file():
        blockers.append("Local Data-to-predict-Track-2.csv missing (official MSE).")
    blockers.append(
        "Psych-101 rows are natural-language transcripts; pipeline expects structured "
        "ParticipantData (Ha, pHa, GameID, block_id, action 0/1, test_targets for MSE)."
    )
    blockers.append(
        "No load_cpc18_from_psych101 (or similar) exists in data_modules/cpc18.py."
    )

    can_swap_without_changes = len(blockers) == 0
    return {
        "can_swap_without_code_changes": can_swap_without_changes,
        "summary": (
            "No code changes needed."
            if can_swap_without_changes
            else "Code changes required — Psych-101-test is not a drop-in replacement for local Track II CSVs."
        ),
        "blockers": blockers,
        "recommended_work_if_migrating": [
            "Add data_modules/cpc18.py loader from Psych-101 (parse text -> ParticipantData or new trial schema).",
            "Map participant_id semantics (HF row ordinal vs raw participant string vs SubjID).",
            "Extend collect_participant_ids.py with HF/local switch (mirror choice13k).",
            "Wire Template_evo_non_strict.py cpc18 branches to the new loader (and --data_path / --hf_dataset flags).",
            "Regenerate datasets/cpc18/valid_participant_ids.json from Psych-101-test if that is the new source.",
            "Revisit cpc18_official_mse=True: block targets are not in Psych-101 transcripts.",
        ],
    }


def run_checks() -> Dict[str, Any]:
    if not LOCAL_RAW.is_file():
        raise FileNotFoundError(f"Missing local CPC18 raw file: {LOCAL_RAW}")

    local_subj = _local_subj_ids()
    local_stats = _local_per_participant_stats()

    psych_test = _load_psych_split(PSYCH_TEST_HF, "test")
    psych_train = _load_psych_split(PSYCH_TRAIN_HF, "train")
    psych_test_stats = _psych_participant_stats(psych_test)
    psych_train_stats = _psych_participant_stats(psych_train)

    psych_test_ids = set(psych_test["participant"])
    psych_train_ids = set(psych_train["participant"])
    local_set = set(local_subj)

    report: Dict[str, Any] = {
        "current_local_source": {
            "data_path": _relative(LOCAL_CPC18_DIR),
            "raw_comp_set": _relative(LOCAL_RAW),
            "data_to_predict": _relative(LOCAL_TARGETS),
            "loader": "data_modules.cpc18.load_cpc18_track2_data",
            "confirmed": (
                "Template_evo_non_strict.py and collect_participant_ids.py both use "
                "load_cpc18_track2_data() on datasets/cpc18/ (raw-comp-set-data-Track-2.csv "
                "+ Data-to-predict-Track-2.csv for official MSE)."
            ),
        },
        "psych101_source": {
            "train_hf": PSYCH_TRAIN_HF,
            "test_hf": PSYCH_TEST_HF,
            "experiment_column": CPC18_EXPERIMENT,
            "centaur_paper_name": "CPC18",
        },
        "local_csv": local_stats,
        "local_subj_ids_sample": local_subj[:5],
        "psych_train": psych_train_stats,
        "psych_test": psych_test_stats,
        "participant_id_overlap": {
            "local_subj_vs_psych_test": len(local_set & psych_test_ids),
            "local_subj_vs_psych_train": len(local_set & psych_train_ids),
            "psych_test_only_sample": sorted(psych_test_ids - local_set, key=int)[:10],
            "local_only_sample": sorted(local_set - psych_test_ids, key=int)[:10],
        },
        "collect_participant_ids.py": {
            "uses_psych101_today": False,
            "uses_local_raw_csv": True,
        },
        "Template_evo_non_strict.py": {
            "uses_psych101_today": False,
            "uses_load_cpc18_track2_data": True,
        },
        "code_touch_points": _code_touch_points(),
    }
    report["verdict"] = _build_verdict(report)
    return report


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    v = report["verdict"]
    lines = [
        "# CPC18 local CSV vs Psych-101-test compatibility",
        "",
        f"**Verdict:** {v['summary']}",
        "",
        "## Current source (confirmed)",
        "",
        f"- Directory: `{report['current_local_source']['data_path']}`",
        f"- Raw trials: `{report['current_local_source']['raw_comp_set']}`",
        f"- MSE targets: `{report['current_local_source']['data_to_predict']}`",
        f"- Loader: `{report['current_local_source']['loader']}`",
        "",
        "## Psych-101 CPC18",
        "",
        f"- Experiment id: `{report['psych101_source']['experiment_column']}`",
        f"- Train: `{report['psych101_source']['train_hf']}` "
        f"({report['psych_train']['n_participants']} participants)",
        f"- Test: `{report['psych101_source']['test_hf']}` "
        f"({report['psych_test']['n_participants']} participants)",
        "",
        "## Key mismatches",
        "",
    ]
    for b in v["blockers"]:
        lines.append(f"- {b}")
    lines.extend(
        [
            "",
            "## Files to change (minimum)",
            "",
            "- `data_modules/cpc18.py` (new loader / parser)",
            "- `utils/tools/collect_participant_ids.py` (`collect_cpc18`, cpc18 CLI branch)",
            "- `Template_evo_non_strict.py` (cpc18 branches ~1798, ~4409)",
            "",
            "See `cpc18_psych101_compat_report.json` for full stats.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Write reports here (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_checks()
    json_path = out_dir / "cpc18_psych101_compat_report.json"
    md_path = out_dir / "cpc18_psych101_compat_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)

    print(report["verdict"]["summary"])
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("\nBlockers:")
    for b in report["verdict"]["blockers"]:
        print(f"  - {b}")


if __name__ == "__main__":
    main()
