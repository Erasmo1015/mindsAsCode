#!/usr/bin/env python3
"""Generate per-trial held-out log-likelihood heatmap for one participant (or a range)."""

# use argument --participant_id 2 or 4 to plot the heatmap for participant 2 or 4 (default scope: single).
# use --participant_scope range --range_start_ordinal 0 --range_end_ordinal 9 to match te_dr/te_aggregate
# choice13k ordinal slicing into datasets/choice13k/valid_participant_ids.json (inclusive end).
# use argument --include_openevolve to include the openevolve heatmap. (default is False)
# use argument --pdf to save the heatmap as a PDF file. (default is False)
# default PNG/CSV output folder: analysis/analysis_plot/proposal/test_heatmap (CSV only with --write_trials_csv)
# default experiment folder: generated_outputs/choice13k/non_strict/run_260430_013702
# override runs/CSV with --ours-run-dir, --centaur-predictions-csv, --openevolve-run-dir (relative paths use repo root).
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


OURS_RUN_DIR = Path("generated_outputs/choice13k/non_strict/run_260430_013702")
CENTAUR_PRED_CSV = Path("generated_outputs/choice13k/centaur/run_260506_224505/log/predictions_vs_actual.csv")
OPENEVOLVE_RUN_DIR = Path("generated_outputs/choice13k/openevolve/run_260428_011306")
OUTPUT_DIR = Path("analysis/analysis_plot/proposal/test_heatmap")
FALLBACK_SPLIT_RATIO = 0.9
FALLBACK_SPLIT_SEED = 0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolved_experiment_path(cli_path: Optional[Path], default_relative: Path) -> Path:
    """Absolute path for a run directory or CSV: use CLI if set, else default; relative paths are under repo root."""
    p = Path(default_relative if cli_path is None else cli_path).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _load_choice13k_valid_participant_ids() -> List[int]:
    """Same ordering as te_dr / te_aggregate ``valid_participant_ids.json`` for choice13k."""
    path = REPO_ROOT / "datasets" / "choice13k" / "valid_participant_ids.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Generate with: python utils/tools/collect_participant_ids.py --dataset choice13k"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [int(x) for x in data["valid_participant_ids"]]


def _participant_ids_for_cli(
    *,
    participant_scope: str,
    participant_id: Optional[int],
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
) -> List[int]:
    if participant_scope == "single":
        if participant_id is None:
            raise ValueError("--participant_id is required when --participant_scope single.")
        return [int(participant_id)]
    if participant_scope != "range":
        raise ValueError(f"Unsupported --participant_scope {participant_scope!r}.")
    if range_start_ordinal is None or range_end_ordinal is None:
        raise ValueError(
            "--participant_scope range requires --range_start_ordinal and --range_end_ordinal (inclusive)."
        )
    valid = _load_choice13k_valid_participant_ids()
    a, b = int(range_start_ordinal), int(range_end_ordinal)
    if a < 0 or b >= len(valid) or a > b:
        raise ValueError(
            f"Invalid ordinal range [{a}, {b}] for valid list of length {len(valid)} (0-based inclusive end)."
        )
    return valid[a : b + 1]


def _clip_prob_b(p: float) -> float:
    return float(min(max(float(p), 1e-6), 1 - 1e-6))


def _binary_loglik(pred_prob_b: float, actual_action: int) -> float:
    p = _clip_prob_b(pred_prob_b)
    if int(actual_action) == 1:
        return float(math.log(p))
    return float(math.log(1.0 - p))


def _load_choose_fn(program_path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location("participant_best_program", str(program_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program module from: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    choose = getattr(module, "choose", None)
    if choose is None or not callable(choose):
        raise RuntimeError(f"'choose(problem, history)' function not found in: {program_path}")
    return choose


def _convert_choose_output_to_prob_b(raw_output: object) -> float:
    if isinstance(raw_output, bool):
        return 1.0 - 1e-6 if raw_output else 1e-6
    if isinstance(raw_output, (int, np.integer)):
        val = int(raw_output)
        if val == 0:
            return 1e-6
        if val == 1:
            return 1.0 - 1e-6
    if isinstance(raw_output, (float, int, np.floating)):
        return _clip_prob_b(float(raw_output))
    raise TypeError(
        "choose(problem, history) returned unsupported type. "
        f"Expected float or 0/1 hard action, got: {type(raw_output).__name__}"
    )


def _find_ours_prediction_csvs(run_dir: Path, participant_dir: Path) -> List[Path]:
    search_roots = [participant_dir, run_dir / "csvs", run_dir / "log"]
    candidates: List[Path] = []
    required = {"trial_index", "actual_action", "pred_prob_b"}
    for root in search_roots:
        if not root.exists():
            continue
        for csv_path in root.rglob("*.csv"):
            try:
                header = pd.read_csv(csv_path, nrows=0).columns
            except Exception:
                continue
            cols = {str(c).strip() for c in header}
            if required.issubset(cols):
                candidates.append(csv_path)
    candidates.sort(key=lambda p: str(p))
    return candidates


def _load_ours_predictions_from_existing_csv(csv_path: Path, participant_id: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "participant_id" in df.columns:
        df = df[df["participant_id"] == participant_id]
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"]
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == "choice13k"]
    needed = ["trial_index", "actual_action", "pred_prob_b"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    out = df[needed].copy()
    out["trial_index"] = out["trial_index"].astype(int)
    out["actual_action"] = out["actual_action"].astype(int)
    out["pred_prob_b"] = out["pred_prob_b"].astype(float)
    return out.sort_values("trial_index").reset_index(drop=True)


def _load_test_trials(participant_id: int) -> Tuple[Sequence[dict], Path]:
    root = repo_root()
    searched = [
        root / "analysis" / "data" / "choices13k" / f"participant_{participant_id}_test_trials.json",
        root / "analysis" / "code" / "choices13k" / f"participant_{participant_id}_test_trials.json",
    ]
    for p in searched:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), p

    # Fallback for participants without locally exported trial JSON:
    # reconstruct trials from canonical Choice13k dataset using the same
    # within-participant block split matching run evaluation defaults.
    try:
        from data_modules.choice13k import get_choice13k_experiments
    except Exception as exc:
        raise FileNotFoundError(
            "Could not find participant test trials JSON and fallback loader import failed. "
            "Searched:\n"
            + "\n".join(str(p) for p in searched)
            + f"\nFallback import error: {type(exc).__name__}: {exc}"
        ) from exc

    exps = get_choice13k_experiments(n_participants=participant_id + 1)
    if participant_id >= len(exps):
        raise FileNotFoundError(
            "Could not find participant test trials JSON and dataset fallback did not provide participant "
            f"{participant_id}. Searched:\n"
            + "\n".join(str(p) for p in searched)
        )
    exp = exps[participant_id]
    n_blocks = len(exp.blocks)
    if n_blocks < 2:
        raise FileNotFoundError(
            f"Could not build fallback trials for participant {participant_id}: only {n_blocks} blocks."
        )
    rng = np.random.default_rng(FALLBACK_SPLIT_SEED)
    perm = np.arange(n_blocks)
    rng.shuffle(perm)
    split_idx = int(n_blocks * FALLBACK_SPLIT_RATIO)
    split_idx = max(1, min(split_idx, n_blocks - 1))
    test_blocks = set(perm[split_idx:].tolist())
    test_trials = []
    for bi, block in enumerate(exp.blocks):
        if bi not in test_blocks:
            continue
        options = block.option_keys
        history_accum: List[dict] = []
        for trial in block.trials:
            test_trials.append(
                {
                    "problem": {
                        "gamble_A": {"probs": block.gamble_A.probs, "rewards": block.gamble_A.rewards},
                        "gamble_B": {"probs": block.gamble_B.probs, "rewards": block.gamble_B.rewards},
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": trial.action,
                }
            )
            history_accum.append({"action": trial.action, "feedback": trial.feedback})
    fallback_path = Path(f"[dataset-fallback] marcelbinz/Psych-101-test participant_ordinal={participant_id}")
    print(
        "[INFO] Local participant test JSON not found; using dataset fallback trials for "
        f"participant {participant_id} (split_ratio={FALLBACK_SPLIT_RATIO}, split_seed={FALLBACK_SPLIT_SEED})."
    )
    return test_trials, fallback_path


def _regenerate_ours_predictions(participant_id: int, participant_dir: Path) -> pd.DataFrame:
    searched_program_paths: List[Path] = []
    patterns = ["best_program*.py", "**/best_program*.py", "**/*best*program*.py"]
    best_program_path = None
    for pattern in patterns:
        matches = sorted(participant_dir.glob(pattern))
        searched_program_paths.append(participant_dir / pattern)
        if matches:
            best_program_path = matches[0]
            break
    if best_program_path is None:
        raise FileNotFoundError(
            f"Could not find final/best evolved program for participant {participant_id}. "
            "Searched patterns:\n" + "\n".join(str(p) for p in searched_program_paths)
        )

    choose = _load_choose_fn(best_program_path)
    test_trials, test_trials_path = _load_test_trials(participant_id)

    rows = []
    runtime_errors = 0
    float_prob = 0
    hard_action = 0
    other_numeric = 0
    for idx, trial in enumerate(test_trials):
        actual_action = int(trial["action"])
        raw_output = None
        pred_prob_b = 0.5
        was_clipped = False
        runtime_error = ""
        try:
            raw_output = choose(trial["problem"], trial.get("history", []))
            if isinstance(raw_output, bool):
                hard_action += 1
            elif isinstance(raw_output, (int, np.integer)) and int(raw_output) in (0, 1):
                hard_action += 1
            elif isinstance(raw_output, (float, np.floating)):
                float_prob += 1
            elif isinstance(raw_output, (int, np.integer)):
                other_numeric += 1

            converted = _convert_choose_output_to_prob_b(raw_output)
            if isinstance(raw_output, (float, int, np.floating, np.integer, bool)):
                raw_float = float(int(raw_output) if isinstance(raw_output, bool) else raw_output)
                was_clipped = abs(converted - raw_float) > 1e-12
            pred_prob_b = converted
        except Exception as exc:
            runtime_errors += 1
            runtime_error = f"{type(exc).__name__}: {exc}"
            raw_output = None
            pred_prob_b = 0.5

        rows.append(
            {
                "trial_index": int(idx),
                "actual_action": actual_action,
                "pred_prob_b": float(pred_prob_b),
                "ours_raw_output": "" if raw_output is None else str(raw_output),
                "ours_was_clipped": bool(was_clipped),
                "ours_runtime_error": runtime_error,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("trial_index").reset_index(drop=True)
    probs = out_df["pred_prob_b"].astype(float)
    print(f"[VERIFY] Ours final program path used: {best_program_path}")
    print(f"[VERIFY] Ours test trial source path used: {test_trials_path}")
    print(
        f"[VERIFY] choose() outputs -> float_probabilities={float_prob}, "
        f"hard_0_or_1_actions={hard_action}, other_numeric={other_numeric}"
    )
    print(
        f"[VERIFY] Ours predicted_prob_b stats -> min={probs.min():.6f}, "
        f"max={probs.max():.6f}, mean={probs.mean():.6f}"
    )
    print(
        "[VERIFY] Clipped boundary counts -> "
        f"at_1e-6={(probs == 1e-6).sum()}, at_0.999999={(probs == (1 - 1e-6)).sum()}"
    )
    print(f"[VERIFY] Ours runtime errors: {runtime_errors}")
    return out_df


def _regenerate_program_predictions(
    participant_id: int, participant_dir: Path, program_glob: str, label: str
) -> pd.DataFrame:
    matches = sorted(participant_dir.glob(program_glob))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {label} program for participant {participant_id} using pattern "
            f"'{program_glob}' in {participant_dir}"
        )
    program_path = matches[0]
    choose = _load_choose_fn(program_path)
    test_trials, test_trials_path = _load_test_trials(participant_id)

    rows = []
    runtime_errors = 0
    float_prob = 0
    hard_action = 0
    other_numeric = 0
    for idx, trial in enumerate(test_trials):
        actual_action = int(trial["action"])
        raw_output = None
        pred_prob_b = 0.5
        was_clipped = False
        runtime_error = ""
        try:
            raw_output = choose(trial["problem"], trial.get("history", []))
            if isinstance(raw_output, bool):
                hard_action += 1
            elif isinstance(raw_output, (int, np.integer)) and int(raw_output) in (0, 1):
                hard_action += 1
            elif isinstance(raw_output, (float, np.floating)):
                float_prob += 1
            elif isinstance(raw_output, (int, np.integer)):
                other_numeric += 1
            converted = _convert_choose_output_to_prob_b(raw_output)
            if isinstance(raw_output, (float, int, np.floating, np.integer, bool)):
                raw_float = float(int(raw_output) if isinstance(raw_output, bool) else raw_output)
                was_clipped = abs(converted - raw_float) > 1e-12
            pred_prob_b = converted
        except Exception as exc:
            runtime_errors += 1
            runtime_error = f"{type(exc).__name__}: {exc}"
            pred_prob_b = 0.5

        rows.append(
            {
                "trial_index": int(idx),
                "actual_action": actual_action,
                "pred_prob_b": float(pred_prob_b),
                f"{label.lower()}_raw_output": "" if raw_output is None else str(raw_output),
                f"{label.lower()}_was_clipped": bool(was_clipped),
                f"{label.lower()}_runtime_error": runtime_error,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("trial_index").reset_index(drop=True)
    probs = out_df["pred_prob_b"].astype(float)
    print(f"[VERIFY] {label} final program path used: {program_path}")
    print(f"[VERIFY] {label} test trial source path used: {test_trials_path}")
    print(
        f"[VERIFY] {label} choose() outputs -> float_probabilities={float_prob}, "
        f"hard_0_or_1_actions={hard_action}, other_numeric={other_numeric}"
    )
    print(
        f"[VERIFY] {label} predicted_prob_b stats -> min={probs.min():.6f}, "
        f"max={probs.max():.6f}, mean={probs.mean():.6f}"
    )
    print(
        f"[VERIFY] {label} clipped boundary counts -> "
        f"at_1e-6={(probs == 1e-6).sum()}, at_0.999999={(probs == (1 - 1e-6)).sum()}"
    )
    print(f"[VERIFY] {label} runtime errors: {runtime_errors}")
    return out_df


def load_ours_test_predictions(participant_id: int, *, ours_run_dir: Path) -> pd.DataFrame:
    participant_dir = ours_run_dir / f"participant_{participant_id}"
    if not participant_dir.exists():
        raise FileNotFoundError(f"Missing participant folder in Ours run: {participant_dir}")

    print("[INFO] Regenerating Ours predictions from final evolved program.")
    return _regenerate_ours_predictions(participant_id, participant_dir)


def load_centaur_test_predictions(participant_id: int, *, predictions_csv: Path) -> pd.DataFrame:
    csv_path = predictions_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Centaur predictions CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"participant_id", "split", "trial_index", "actual_action", "pred_prob_b"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Centaur CSV missing required columns: {missing}")

    df = df[(df["participant_id"] == participant_id) & (df["split"] == "test")].copy()
    df["trial_index"] = df["trial_index"].astype(int)
    df["actual_action"] = df["actual_action"].astype(int)
    df["pred_prob_b"] = df["pred_prob_b"].astype(float)
    return df.sort_values("trial_index").reset_index(drop=True)[["trial_index", "actual_action", "pred_prob_b"]]


def load_openevolve_test_predictions(participant_id: int, *, openevolve_run_dir: Path) -> pd.DataFrame:
    run_dir = openevolve_run_dir
    participant_dir = run_dir / f"participant_{participant_id}"
    if not participant_dir.exists():
        raise FileNotFoundError(f"Missing participant folder in OpenEvolve run: {participant_dir}")
    print("[INFO] Regenerating OpenEvolve predictions from final evolved program.")
    return _regenerate_program_predictions(
        participant_id=participant_id,
        participant_dir=participant_dir,
        program_glob="best_program.py",
        label="OpenEvolve",
    )


def build_aligned_dataframe(
    participant_id: int,
    ours: pd.DataFrame,
    centaur: pd.DataFrame,
    openevolve: pd.DataFrame | None = None,
    *,
    write_trials_csv: bool = False,
) -> pd.DataFrame:
    ours = ours.sort_values("trial_index").reset_index(drop=True)
    centaur = centaur.sort_values("trial_index").reset_index(drop=True)
    include_openevolve = openevolve is not None
    if include_openevolve:
        openevolve = openevolve.sort_values("trial_index").reset_index(drop=True)

    out_csv = repo_root() / OUTPUT_DIR / f"participant_{participant_id}_ours_vs_centaur_loglik_trials.csv"

    if len(ours) != len(centaur) or (include_openevolve and len(ours) != len(openevolve)):
        diagnostic = pd.merge(
            ours.rename(
                columns={
                    "actual_action": "ours_actual_action",
                    "pred_prob_b": "ours_pred_prob_b",
                }
            ),
            (
                pd.merge(
                    centaur.rename(
                        columns={
                            "actual_action": "centaur_actual_action",
                            "pred_prob_b": "centaur_pred_prob_b",
                        }
                    ),
                    openevolve.rename(
                        columns={
                            "actual_action": "openevolve_actual_action",
                            "pred_prob_b": "openevolve_pred_prob_b",
                        }
                    ),
                    on="trial_index",
                    how="outer",
                )
                if include_openevolve
                else centaur.rename(
                    columns={
                        "actual_action": "centaur_actual_action",
                        "pred_prob_b": "centaur_pred_prob_b",
                    }
                )
            ),
            on="trial_index",
            how="outer",
        ).sort_values("trial_index")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.to_csv(out_csv, index=False)
        length_msg = (
            f"Ours={len(ours)}, Centaur={len(centaur)}, OpenEvolve={len(openevolve)}. "
            if include_openevolve
            else f"Ours={len(ours)}, Centaur={len(centaur)}. "
        )
        raise ValueError(f"Length mismatch between model test trials. {length_msg}Saved diagnostic CSV: {out_csv}")

    action_match_centaur = ours["actual_action"].values == centaur["actual_action"].values
    action_match_openevolve = (
        ours["actual_action"].values == openevolve["actual_action"].values if include_openevolve else np.array([])
    )
    mismatch_count_centaur = int((~action_match_centaur).sum())
    mismatch_count_openevolve = int((~action_match_openevolve).sum()) if include_openevolve else 0
    if mismatch_count_centaur > 0 or (include_openevolve and mismatch_count_openevolve > 0):
        mismatch_trials = sorted(
            set(ours.loc[~action_match_centaur, "trial_index"].tolist())
            | (set(ours.loc[~action_match_openevolve, "trial_index"].tolist()) if include_openevolve else set())
        )
        if include_openevolve:
            print("!!! WARNING: actual_action sequences do not match across models !!!")
            print(
                "!!! Mismatch count: "
                f"vs Centaur={mismatch_count_centaur}, vs OpenEvolve={mismatch_count_openevolve}; "
                f"trial_index: {mismatch_trials} !!!"
            )
        else:
            print("!!! WARNING: Ours and Centaur actual_action sequences do not match !!!")
            print(f"!!! Mismatch count: {mismatch_count_centaur}; trial_index: {mismatch_trials} !!!")
    else:
        if include_openevolve:
            print("[VERIFY] Ours, Centaur, and OpenEvolve actual_action sequences match exactly.")
        else:
            print("[VERIFY] Ours and Centaur actual_action sequences match exactly.")

    actual_action = ours["actual_action"].astype(int)
    merged = pd.DataFrame(
        {
            "participant_id": participant_id,
            "trial_index": ours["trial_index"].astype(int),
            "problem_block": (ours["trial_index"].astype(int) // 5 + 1).astype(int),
            "trial_within_problem": (ours["trial_index"].astype(int) % 5 + 1).astype(int),
            "actual_action": actual_action,
            "actual_action_label": actual_action.map({0: "A", 1: "B"}),
            "ours_pred_prob_b": ours["pred_prob_b"].astype(float),
            "centaur_pred_prob_b": centaur["pred_prob_b"].astype(float),
            "ours_raw_output": ours["ours_raw_output"].astype(str),
            "ours_was_clipped": ours["ours_was_clipped"].astype(bool),
            "ours_runtime_error": ours["ours_runtime_error"].astype(str),
        }
    )
    if include_openevolve:
        merged["openevolve_pred_prob_b"] = openevolve["pred_prob_b"].astype(float)
        merged["openevolve_raw_output"] = openevolve["openevolve_raw_output"].astype(str)
        merged["openevolve_was_clipped"] = openevolve["openevolve_was_clipped"].astype(bool)
        merged["openevolve_runtime_error"] = openevolve["openevolve_runtime_error"].astype(str)
    merged["ours_loglik"] = merged.apply(
        lambda r: _binary_loglik(r["ours_pred_prob_b"], int(r["actual_action"])),
        axis=1,
    )
    merged["centaur_loglik"] = merged.apply(
        lambda r: _binary_loglik(r["centaur_pred_prob_b"], int(r["actual_action"])),
        axis=1,
    )
    if include_openevolve:
        merged["openevolve_loglik"] = merged.apply(
            lambda r: _binary_loglik(r["openevolve_pred_prob_b"], int(r["actual_action"])),
            axis=1,
        )
    merged["delta_loglik"] = merged["ours_loglik"] - merged["centaur_loglik"]
    if write_trials_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)
    return merged


def plot_heatmap(
    participant_id: int,
    df: pd.DataFrame,
    save_pdf: bool = False,
    *,
    norm_vcenter: float = -0.2,
) -> Path:
    def _display_loglik(value: float) -> str:
        if abs(value) < 0.005:
            return "0.00"
        return f"{value:.2f}"

    n_trials = len(df)
    include_openevolve = "openevolve_loglik" in df.columns
    if include_openevolve:
        heat = np.vstack([df["ours_loglik"].to_numpy(), df["centaur_loglik"].to_numpy(), df["openevolve_loglik"].to_numpy()])
    else:
        heat = np.vstack([df["ours_loglik"].to_numpy(), df["centaur_loglik"].to_numpy()])
    labels = df["actual_action_label"].to_numpy()
    all_loglik = (
        np.concatenate([df["ours_loglik"].to_numpy(), df["centaur_loglik"].to_numpy(), df["openevolve_loglik"].to_numpy()])
        if include_openevolve
        else np.concatenate([df["ours_loglik"].to_numpy(), df["centaur_loglik"].to_numpy()])
    )
    raw_min = float(np.min(all_loglik))
    vmin = max(-1.5, math.floor(raw_min * 10.0) / 10.0)
    vmax = 0.0
    if not (vmin < norm_vcenter < vmax):
        raise ValueError(
            f"norm_vcenter={norm_vcenter} must satisfy vmin < norm_vcenter < vmax "
            f"(here vmin={vmin}, vmax={vmax} from data)."
        )

    fig_w = max(8.0, 0.45 * n_trials)
    fig_h = 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=norm_vcenter, vmax=vmax)
    im = ax.imshow(heat, cmap="RdYlGn", norm=norm, aspect="auto")

    if n_trials <= 35:
        fs = 6.5
    elif n_trials <= 60:
        fs = 5.8
    else:
        fs = 4.5

    for row_idx in range(heat.shape[0]):
        for col_idx in range(n_trials):
            val = heat[row_idx, col_idx]
            text_color = "black" if val > -1.4 else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{_display_loglik(float(val))}\n{labels[col_idx]}",
                ha="center",
                va="center",
                fontsize=fs,
                color=text_color,
            )

    if include_openevolve:
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Ours", "Centaur", "OpenEvolve"])
    else:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Ours", "Centaur"])
    ax.set_xlabel("")
    ax.set_ylabel("")

    step = max(1, n_trials // 20)
    xticks = list(range(0, n_trials, step))
    if (n_trials - 1) not in xticks:
        xticks.append(n_trials - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=8)

    ours_mean = float(df["ours_loglik"].mean())
    centaur_mean = float(df["centaur_loglik"].mean())
    openevolve_mean = float(df["openevolve_loglik"].mean()) if include_openevolve else None

    ax.set_title(f"Participant {participant_id}: per-trial held-out log-likelihood", fontsize=10.5, pad=12)
    if include_openevolve:
        fig.text(
            0.5,
            0.03,
            f"Ours average: {ours_mean:.2f}   |   Centaur average: {centaur_mean:.2f}   |   OpenEvolve average: {openevolve_mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    else:
        fig.text(
            0.5,
            0.03,
            f"Ours average: {ours_mean:.2f}   |   Centaur average: {centaur_mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    candidate_ticks = [vmin, -1.0, -0.5, norm_vcenter, 0.0]
    cbar_ticks: List[float] = []
    for tick in candidate_ticks:
        if vmin <= tick <= vmax and tick not in cbar_ticks:
            cbar_ticks.append(float(tick))

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035, ticks=cbar_ticks)
    cbar.set_label("loglik")
    cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])

    out_path = repo_root() / OUTPUT_DIR / f"participant_{participant_id}_ours_vs_centaur_loglik_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.07, 0.99, 0.90])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate participant-level Ours vs Centaur vs OpenEvolve test-trial loglik heatmap."
    )
    parser.add_argument(
        "--participant_scope",
        choices=("single", "range"),
        default="single",
        help="single: one --participant_id. range: inclusive ordinals into valid_participant_ids.json "
        "(same as te_dr/te_aggregate --participant_scope range).",
    )
    parser.add_argument(
        "--participant_id",
        type=int,
        default=None,
        help="Raw participant id when --participant_scope single (e.g. 2 or 4).",
    )
    parser.add_argument(
        "--range_start_ordinal",
        type=int,
        default=None,
        help="0-based start index into datasets/choice13k/valid_participant_ids.json (inclusive).",
    )
    parser.add_argument(
        "--range_end_ordinal",
        type=int,
        default=None,
        help="0-based end index into valid_participant_ids.json (inclusive).",
    )
    parser.add_argument(
        "--include_openevolve",
        action="store_true",
        help="Include OpenEvolve as an additional row. Default: skip OpenEvolve.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save a PDF version of the heatmap.",
    )
    parser.add_argument(
        "--write_trials_csv",
        action="store_true",
        help="Write per-trial aligned loglik CSV under the output folder (default: PNG only).",
    )
    parser.add_argument(
        "--heatmap-vcenter",
        type=float,
        default=-0.2,
        metavar="V",
        help="Pivot loglik for the diverging colormap (TwoSlopeNorm vcenter); must lie between computed vmin and 0. "
        "Default: -0.2 (yellow-ish at this value).",
    )
    parser.add_argument(
        "--ours-run-dir",
        type=Path,
        default=None,
        help=f"Run directory with participant_<id>/ holding the evolved program (default: {OURS_RUN_DIR}). "
        "Relative paths are resolved under the repo root.",
    )
    parser.add_argument(
        "--centaur-predictions-csv",
        type=Path,
        default=None,
        help=f"Centaur-style predictions CSV with participant_id, split, trial_index, actual_action, pred_prob_b "
        f"(default: {CENTAUR_PRED_CSV}).",
    )
    parser.add_argument(
        "--openevolve-run-dir",
        type=Path,
        default=None,
        help=f"OpenEvolve run root when using --include_openevolve (default: {OPENEVOLVE_RUN_DIR}).",
    )
    args = parser.parse_args()

    try:
        participant_ids = _participant_ids_for_cli(
            participant_scope=args.participant_scope,
            participant_id=args.participant_id,
            range_start_ordinal=args.range_start_ordinal,
            range_end_ordinal=args.range_end_ordinal,
        )
    except ValueError as exc:
        parser.error(str(exc))

    ours_run_dir = _resolved_experiment_path(args.ours_run_dir, OURS_RUN_DIR)
    centaur_csv = _resolved_experiment_path(args.centaur_predictions_csv, CENTAUR_PRED_CSV)
    openevolve_run_dir = _resolved_experiment_path(args.openevolve_run_dir, OPENEVOLVE_RUN_DIR)

    print(f"[INFO] Ours run dir: {ours_run_dir}")
    print(f"[INFO] Centaur predictions CSV: {centaur_csv}")
    if args.include_openevolve:
        print(f"[INFO] OpenEvolve run dir: {openevolve_run_dir}")

    for pid in participant_ids:
        print(f"\n{'='*60}\nParticipant {pid}\n{'='*60}")
        ours_df = load_ours_test_predictions(pid, ours_run_dir=ours_run_dir)
        centaur_df = load_centaur_test_predictions(pid, predictions_csv=centaur_csv)
        openevolve_df = (
            load_openevolve_test_predictions(pid, openevolve_run_dir=openevolve_run_dir)
            if args.include_openevolve
            else None
        )
        aligned_df = build_aligned_dataframe(
            pid, ours_df, centaur_df, openevolve_df, write_trials_csv=args.write_trials_csv
        )
        png_path = plot_heatmap(
            pid, aligned_df, save_pdf=args.pdf, norm_vcenter=args.heatmap_vcenter
        )

        csv_path = repo_root() / OUTPUT_DIR / f"participant_{pid}_ours_vs_centaur_loglik_trials.csv"
        ours_mean = float(aligned_df["ours_loglik"].mean())
        centaur_mean = float(aligned_df["centaur_loglik"].mean())
        openevolve_mean = float(aligned_df["openevolve_loglik"].mean()) if args.include_openevolve else None

        print(f"Output PNG: {png_path}")
        if args.pdf:
            print(f"Output PDF: {png_path.with_suffix('.pdf')}")
        if args.write_trials_csv:
            print(f"Output CSV: {csv_path}")
        print(f"Number of aligned test trials: {len(aligned_df)}")
        print(f"Ours mean loglik: {ours_mean:.2f}")
        print(f"Centaur mean loglik: {centaur_mean:.2f}")
        if args.include_openevolve:
            print(f"OpenEvolve mean loglik: {openevolve_mean:.2f}")


if __name__ == "__main__":
    main()
