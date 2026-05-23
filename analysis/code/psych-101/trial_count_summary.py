#!/usr/bin/env python3
"""
Summarize parsed trial counts for TEH run participants across Psych-101 datasets.

Usage:
  python analysis/code/psych-101/trial_count_summary.py --all_in
  python analysis/code/psych-101/trial_count_summary.py --dataset 4wulff2018description

Uses the same dataset loading / participant ordinal logic as compare.py.
Participants come from the latest TEH run's participant_details_loglik.csv when
available (clamped to each dataset's supported ordinal range).

Outputs:
  analysis/data/psych101_trial_counts/trial_count_participants.csv
  analysis/data/psych101_trial_counts/trial_count_summary.csv
  analysis/data/psych101_trial_counts/report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    experiment_to_trial_dicts,
    get_filtered_psych101_split,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    parse_psych101_binary_row,
    split_psych_experiment,
)
from utils.teh.teh_datasets import is_mixed_gambles_dataset

from analysis.code.utils import compare as cmp

_DEFAULT_OUT_DIR = "analysis/data/psych101_trial_counts"
_FALLBACK_N_PARTICIPANTS = 50
_LOW_TRIAL_MEDIAN_TRAIN = 4
_LOW_TRIAL_MEDIAN_TOTAL = 20
_TEH_COMMAND_RE = re.compile(
    r"--split_mode\s+(?P<mode>\S+)"
    r"|--split_ratio\s+(?P<ratio>\S+)"
    r"|--split_seed\s+(?P<seed>\S+)"
)

_PARTICIPANT_FIELDS = (
    "dataset",
    "participant_id",
    "teh_run",
    "total_trials",
    "train_trials",
    "val_trials",
    "test_trials",
    "n_problem_groups",
    "max_group_size",
    "split_valid",
    "split_error",
)

_SUMMARY_FIELDS = (
    "dataset",
    "n_participants",
    "teh_run",
    "split_mode",
    "split_ratio",
    "split_seed",
    "split_settings_source",
    "total_trials_min",
    "total_trials_mean",
    "total_trials_median",
    "total_trials_max",
    "train_min",
    "train_mean",
    "train_median",
    "train_max",
    "val_min",
    "val_mean",
    "val_median",
    "val_max",
    "test_min",
    "test_mean",
    "test_median",
    "test_max",
    "participants_with_train_1",
    "participants_with_train_lt_4",
    "participants_with_train_lt_10",
    "low_trial_flag",
    "error",
)


@dataclass
class _ParticipantStats:
    dataset: str
    participant_id: int
    teh_run: str
    total_trials: int
    train_trials: int
    val_trials: int
    test_trials: int
    n_problem_groups: int
    max_group_size: int
    split_valid: bool
    split_error: str = ""


@dataclass
class _DatasetSummary:
    dataset: str
    n_participants: int = 0
    teh_run: str = ""
    split_mode: str = "within_participant"
    split_ratio: float = cmp._DEFAULT_SPLIT_RATIO
    split_seed: int = cmp._DEFAULT_SPLIT_SEED
    split_settings_source: str = ""
    low_trial_flag: bool = False
    error: str = ""
    participants: List[_ParticipantStats] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.participants is None:
            self.participants = []


def _repo_root() -> Path:
    return _REPO_ROOT


def _problem_signature(trial: Mapping[str, Any]) -> str:
    p = dict(trial.get("problem") or {})
    for k in ("dataset_alias", "experiment_id"):
        p.pop(k, None)
    return json.dumps(p, sort_keys=True, default=str)


def _unique_problem_groups(trials: Sequence[Mapping[str, Any]]) -> int:
    return len({_problem_signature(t) for t in trials})


def _max_group_size(trials: Sequence[Mapping[str, Any]]) -> int:
    if not trials:
        return 0
    counts: Counter[str] = Counter()
    for t in trials:
        counts[_problem_signature(t)] += 1
    return max(counts.values())


def _stat(values: Sequence[int]) -> Tuple[int, float, float, int]:
    if not values:
        return 0, 0.0, 0.0, 0
    return (
        min(values),
        float(statistics.mean(values)),
        float(statistics.median(values)),
        max(values),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_split_from_teh_command(run_dir: Path) -> Optional[Tuple[str, float, int, str]]:
    """Read split_mode/ratio/seed from TEH run log/command.txt when present."""
    cmd_path = run_dir / "log" / "command.txt"
    if not cmd_path.is_file():
        return None
    text = cmd_path.read_text(encoding="utf-8", errors="replace")
    mode = "within_participant"
    ratio: Optional[float] = None
    seed: Optional[int] = None
    for match in _TEH_COMMAND_RE.finditer(text):
        if match.group("mode"):
            mode = match.group("mode")
        elif match.group("ratio"):
            ratio = float(match.group("ratio"))
        elif match.group("seed"):
            seed = int(match.group("seed"))
    if ratio is None and seed is None and mode == "within_participant":
        return None
    return (
        mode,
        float(ratio if ratio is not None else cmp._DEFAULT_SPLIT_RATIO),
        int(seed if seed is not None else cmp._DEFAULT_SPLIT_SEED),
        str(cmd_path),
    )


def _resolve_split_settings(
    repo: Path,
    *,
    run_dir: Optional[Path],
    default_ratio: float,
    default_seed: int,
    split_ratio_override: Optional[float],
    split_seed_override: Optional[int],
) -> Tuple[float, int, str, str]:
    if run_dir is not None:
        parsed = _parse_split_from_teh_command(run_dir)
        if parsed is not None:
            mode, ratio, seed, source_path = parsed
            try:
                source = str(Path(source_path).resolve().relative_to(repo.resolve()))
            except ValueError:
                source = source_path
            if split_ratio_override is not None:
                ratio = float(split_ratio_override)
                source = f"{source} (ratio overridden)"
            if split_seed_override is not None:
                seed = int(split_seed_override)
                source = f"{source}; seed={seed}"
            return ratio, seed, mode, source

    ratio = float(split_ratio_override if split_ratio_override is not None else default_ratio)
    seed = int(split_seed_override if split_seed_override is not None else default_seed)
    source = "defaults (compare.py / TEH CLI)"
    if split_ratio_override is not None or split_seed_override is not None:
        source = f"CLI override (ratio={ratio}, seed={seed})"
    return ratio, seed, "within_participant", source


def _resolve_teh_roster(
    repo: Path,
    dataset: str,
    *,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> Tuple[List[int], str, Optional[Path], str]:
    """Return (participant_ids, run_name, run_dir, error_message)."""
    alias = cmp._normalize_compare_dataset(dataset)
    psych_split = cmp._effective_psych_dataset_split(alias, psych_dataset_split)
    teh_run = cmp._auto_discover_teh_run(
        repo, dataset=alias, psych_dataset_split=psych_split
    )
    if teh_run is None:
        root = cmp._teh_search_root(repo, alias, psych_split)
        ord_min, ord_max = cmp._dataset_participant_ordinal_bounds(
            alias,
            psych_dataset_split=psych_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
        n = min(_FALLBACK_N_PARTICIPANTS, ord_max - ord_min + 1)
        fallback = list(range(ord_min, ord_min + n))
        return (
            fallback,
            "",
            None,
            f"no TEH run under {root.relative_to(repo)}; using ordinals {ord_min}..{ord_min + n - 1}",
        )

    csv_path = cmp._resolve_loglik_csv(teh_run)
    run_dir = cmp._run_dir_from_path(teh_run)
    run_name = run_dir.name
    participant_ids = cmp._read_participant_ids_from_csv(csv_path)
    participant_ids, _, _ = cmp._clamp_participant_ids_to_dataset(
        participant_ids,
        dataset=alias,
        psych_dataset_split=psych_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
    )
    return participant_ids, run_name, run_dir, ""


def _psych101_participant_stats(
    alias: str,
    participant_id: int,
    *,
    filtered,
    split_ratio: float,
    split_seed: int,
) -> _ParticipantStats:
    split_error = ""
    all_trials: List[Dict[str, Any]] = []
    train_trials: List[Dict[str, Any]] = []
    val_trials: List[Dict[str, Any]] = []
    test_trials: List[Dict[str, Any]] = []
    n_blocks = 0

    try:
        raw = dict(filtered[int(participant_id)])
        exp = parse_psych101_binary_row(raw, alias)
        n_blocks = len(exp.blocks)
        all_trials = experiment_to_trial_dicts(exp, dataset_alias=alias)
        train_trials, val_trials, test_trials, _ = split_psych_experiment(
            exp, split_ratio=split_ratio, split_seed=split_seed
        )
    except Exception as exc:
        split_error = f"{type(exc).__name__}: {exc}"
        try:
            raw = dict(filtered[int(participant_id)])
            exp = parse_psych101_binary_row(raw, alias)
            n_blocks = len(exp.blocks)
            all_trials = experiment_to_trial_dicts(exp, dataset_alias=alias)
        except Exception:
            pass

    return _ParticipantStats(
        dataset=alias,
        participant_id=int(participant_id),
        teh_run="",
        total_trials=len(all_trials),
        train_trials=len(train_trials),
        val_trials=len(val_trials),
        test_trials=len(test_trials),
        n_problem_groups=n_blocks if n_blocks else _unique_problem_groups(all_trials),
        max_group_size=_max_group_size(all_trials),
        split_valid=split_error == "",
        split_error=split_error,
    )


def _mixed_gambles_participant_stats(
    participant_id: int,
    *,
    csv_path: str,
    filter_gain_loss_only: bool,
    split_ratio: float,
    split_seed: int,
) -> _ParticipantStats:
    split_error = ""
    all_trials: List[Dict[str, Any]] = []
    train_trials: List[Dict[str, Any]] = []
    val_trials: List[Dict[str, Any]] = []
    test_trials: List[Dict[str, Any]] = []

    try:
        train_trials, val_trials, test_trials, _ = load_mixed_gambles_trials(
            int(participant_id),
            csv_path=csv_path,
            filter_gain_loss_only=filter_gain_loss_only,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        all_trials = train_trials + val_trials + test_trials
    except Exception as exc:
        split_error = f"{type(exc).__name__}: {exc}"
        # Still count raw parsed rows for total / groups when split fails.
        import csv as csv_mod

        option_keys = [0, 1]
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                if int(row["subject"]) != int(participant_id):
                    continue
                if filter_gain_loss_only and row.get("gamble_type") != "gain_loss":
                    continue
                gain, loss, cert = float(row["gain"]), float(row["loss"]), float(row["cert"])
                took_gamble = int(row["took_gamble"])
                all_trials.append(
                    {
                        "problem": {
                            "gamble_A": {"rewards": [gain, loss], "probs": [0.5, 0.5]},
                            "gamble_B": {"rewards": [cert], "probs": [1.0]},
                            "option_keys": option_keys,
                        },
                        "action": 1 - took_gamble,
                    }
                )

    return _ParticipantStats(
        dataset="mixed_gambles",
        participant_id=int(participant_id),
        teh_run="",
        total_trials=len(all_trials),
        train_trials=len(train_trials),
        val_trials=len(val_trials),
        test_trials=len(test_trials),
        n_problem_groups=_unique_problem_groups(all_trials),
        max_group_size=_max_group_size(all_trials),
        split_valid=split_error == "",
        split_error=split_error,
    )


def _summarize_dataset(
    dataset: str,
    participants: Sequence[_ParticipantStats],
    *,
    teh_run: str,
    split_mode: str,
    split_ratio: float,
    split_seed: int,
    split_settings_source: str,
    error: str,
) -> _DatasetSummary:
    valid = [p for p in participants if p.split_valid]
    total_vals = [p.total_trials for p in participants]
    train_vals = [p.train_trials for p in valid]
    val_vals = [p.val_trials for p in valid]
    test_vals = [p.test_trials for p in valid]

    n_train_1 = sum(1 for p in valid if p.train_trials == 1)
    n_train_lt_4 = sum(1 for p in valid if p.train_trials < 4)
    n_train_lt_10 = sum(1 for p in valid if p.train_trials < 10)

    _, _, med_train, _ = _stat(train_vals)
    _, _, med_total, _ = _stat(total_vals)

    low_flag = False
    if valid:
        low_flag = (
            med_train <= _LOW_TRIAL_MEDIAN_TRAIN
            or med_total <= _LOW_TRIAL_MEDIAN_TOTAL
            or n_train_1 >= max(1, len(valid) // 2)
        )
    if dataset == "4wulff2018description" and valid and (med_train <= 2 or n_train_1 >= 10):
        low_flag = True

    summary = _DatasetSummary(
        dataset=dataset,
        n_participants=len(participants),
        teh_run=teh_run,
        split_mode=split_mode,
        split_ratio=split_ratio,
        split_seed=split_seed,
        split_settings_source=split_settings_source,
        low_trial_flag=low_flag,
        error=error,
        participants=list(participants),
    )
    summary._train_vals = train_vals  # type: ignore[attr-defined]
    summary._val_vals = val_vals  # type: ignore[attr-defined]
    summary._test_vals = test_vals  # type: ignore[attr-defined]
    summary._total_vals = total_vals  # type: ignore[attr-defined]
    summary._n_train_1 = n_train_1  # type: ignore[attr-defined]
    summary._n_train_lt_4 = n_train_lt_4  # type: ignore[attr-defined]
    summary._n_train_lt_10 = n_train_lt_10  # type: ignore[attr-defined]
    return summary


def _summary_to_csv_row(summary: _DatasetSummary) -> Dict[str, Any]:
    t_min, t_mean, t_med, t_max = _stat(summary._total_vals)  # type: ignore[attr-defined]
    tr_min, tr_mean, tr_med, tr_max = _stat(summary._train_vals)  # type: ignore[attr-defined]
    v_min, v_mean, v_med, v_max = _stat(summary._val_vals)  # type: ignore[attr-defined]
    te_min, te_mean, te_med, te_max = _stat(summary._test_vals)  # type: ignore[attr-defined]
    return {
        "dataset": summary.dataset,
        "n_participants": summary.n_participants,
        "teh_run": summary.teh_run,
        "split_mode": summary.split_mode,
        "split_ratio": summary.split_ratio,
        "split_seed": summary.split_seed,
        "split_settings_source": summary.split_settings_source,
        "total_trials_min": t_min,
        "total_trials_mean": round(t_mean, 2),
        "total_trials_median": t_med,
        "total_trials_max": t_max,
        "train_min": tr_min,
        "train_mean": round(tr_mean, 2),
        "train_median": tr_med,
        "train_max": tr_max,
        "val_min": v_min,
        "val_mean": round(v_mean, 2),
        "val_median": v_med,
        "val_max": v_max,
        "test_min": te_min,
        "test_mean": round(te_mean, 2),
        "test_median": te_med,
        "test_max": te_max,
        "participants_with_train_1": summary._n_train_1,  # type: ignore[attr-defined]
        "participants_with_train_lt_4": summary._n_train_lt_4,  # type: ignore[attr-defined]
        "participants_with_train_lt_10": summary._n_train_lt_10,  # type: ignore[attr-defined]
        "low_trial_flag": summary.low_trial_flag,
        "error": summary.error,
    }


def _participant_to_csv_row(p: _ParticipantStats) -> Dict[str, Any]:
    return {
        "dataset": p.dataset,
        "participant_id": p.participant_id,
        "teh_run": p.teh_run,
        "total_trials": p.total_trials,
        "train_trials": p.train_trials,
        "val_trials": p.val_trials,
        "test_trials": p.test_trials,
        "n_problem_groups": p.n_problem_groups,
        "max_group_size": p.max_group_size,
        "split_valid": p.split_valid,
        "split_error": p.split_error,
    }


def _analyze_dataset(
    repo: Path,
    dataset: str,
    *,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    split_ratio_override: Optional[float],
    split_seed_override: Optional[int],
    quiet: bool,
) -> _DatasetSummary:
    alias = cmp._normalize_compare_dataset(dataset)
    psych_split = cmp._effective_psych_dataset_split(alias, psych_dataset_split)

    roster, run_name, run_dir, roster_error = _resolve_teh_roster(
        repo,
        alias,
        psych_dataset_split=psych_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
    )
    split_ratio, split_seed, split_mode, split_source = _resolve_split_settings(
        repo,
        run_dir=run_dir,
        default_ratio=cmp._DEFAULT_SPLIT_RATIO,
        default_seed=cmp._DEFAULT_SPLIT_SEED,
        split_ratio_override=split_ratio_override,
        split_seed_override=split_seed_override,
    )
    if not quiet:
        run_label = run_name or "(none)"
        print(
            f"[{alias}] TEH run={run_label}, n_participants={len(roster)}, "
            f"split_mode={split_mode}, ratio={split_ratio}, seed={split_seed}"
        )
        if roster_error:
            print(f"  Warning: {roster_error}", file=sys.stderr)

    participants: List[_ParticipantStats] = []
    filtered = None
    if not is_mixed_gambles_dataset(alias):
        filtered = get_filtered_psych101_split(
            alias, split=psych_split, local_dataset=local_dataset
        )

    for pid in roster:
        if is_mixed_gambles_dataset(alias):
            stats = _mixed_gambles_participant_stats(
                pid,
                csv_path=mixed_gambles_csv,
                filter_gain_loss_only=filter_mixed_gambles,
                split_ratio=split_ratio,
                split_seed=split_seed,
            )
        else:
            stats = _psych101_participant_stats(
                alias,
                pid,
                filtered=filtered,
                split_ratio=split_ratio,
                split_seed=split_seed,
            )
        stats.teh_run = run_name
        participants.append(stats)

    return _summarize_dataset(
        alias,
        participants,
        teh_run=run_name,
        split_mode=split_mode,
        split_ratio=split_ratio,
        split_seed=split_seed,
        split_settings_source=split_source,
        error=roster_error,
    )


def _print_summary_table(summaries: Sequence[_DatasetSummary]) -> None:
    cols = [
        ("dataset", 28),
        ("n", 4),
        ("total_med", 10),
        ("train_med", 10),
        ("test_med", 9),
        ("tr=1", 5),
        ("tr<4", 5),
        ("flag", 5),
    ]
    header = " ".join(name.rjust(width) for name, width in cols)
    print("")
    print("Trial count summary (parsed trials after Psych-101 parser; TEH within-participant split)")
    print(header)
    print("-" * len(header))
    for s in summaries:
        if s.error and not s.participants:
            print(f"{s.dataset:<28} ERROR: {s.error}")
            continue
        row = _summary_to_csv_row(s)
        line = (
            f"{str(row['dataset']):<28}"
            f"{int(row['n_participants']):>4}"
            f"{float(row['total_trials_median']):>10.0f}"
            f"{float(row['train_median']):>10.0f}"
            f"{float(row['test_median']):>9.0f}"
            f"{int(row['participants_with_train_1']):>5}"
            f"{int(row['participants_with_train_lt_4']):>5}"
            f"{('YES' if row['low_trial_flag'] else ''):>5}"
        )
        print(line)
    print("")


def _build_report(
    repo: Path,
    summaries: Sequence[_DatasetSummary],
    *,
    psych_dataset_split: str,
) -> List[str]:
    lines: List[str] = []
    lines.append("PSYCH-101 / MIXED GAMBLES TRIAL COUNT SUMMARY")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Repo: {repo}")
    lines.append(f"psych_dataset_split: {psych_dataset_split}")
    lines.append("")
    lines.append(
        "Counts use parsed trials after the Psych-101 parser (or mixed_gambles loader), "
        "then TEH within-participant block/signature split."
    )
    lines.append(
        f"Default split when metadata missing: split_mode=within_participant, "
        f"split_ratio={cmp._DEFAULT_SPLIT_RATIO}, split_seed={cmp._DEFAULT_SPLIT_SEED}."
    )
    lines.append("")

    flagged = [s for s in summaries if s.low_trial_flag]
    if flagged:
        lines.append("=" * 72)
        lines.append("LOW-TRIAL WARNINGS")
        lines.append("=" * 72)
        for s in flagged:
            row = _summary_to_csv_row(s)
            lines.append(
                f"*** {s.dataset}: median train={row['train_median']}, "
                f"median total={row['total_trials_median']}, "
                f"train==1 for {row['participants_with_train_1']}/{row['n_participants']} participants"
            )
            if s.dataset == "4wulff2018description":
                lines.append(
                    "    4wulff2018description: first ~50 HF-row participants are often 3-block subjects; "
                    "with split_ratio=0.6 that yields train=1 trial for most TEH run participants. "
                    "This is expected block-split behavior, not a parser bug — but TEH search is severely "
                    "under-powered on train for those ids."
                )
        lines.append("")

    for s in summaries:
        lines.append("=" * 72)
        lines.append(s.dataset)
        lines.append("=" * 72)
        row = _summary_to_csv_row(s)
        lines.append(f"TEH run: {s.teh_run or '(not found)'}")
        if s.error:
            lines.append(f"Roster note: {s.error}")
        lines.append(
            f"Split: mode={s.split_mode}, ratio={s.split_ratio}, seed={s.split_seed} "
            f"(source: {s.split_settings_source})"
        )
        lines.append(f"Participants summarized: {s.n_participants}")
        lines.append(
            f"Total trials: min={row['total_trials_min']}, median={row['total_trials_median']}, "
            f"max={row['total_trials_max']}, mean={row['total_trials_mean']}"
        )
        lines.append(
            f"Train trials: min={row['train_min']}, median={row['train_median']}, "
            f"max={row['train_max']}, mean={row['train_mean']}"
        )
        lines.append(
            f"Val trials: min={row['val_min']}, median={row['val_median']}, "
            f"max={row['val_max']}, mean={row['val_mean']}"
        )
        lines.append(
            f"Test trials: min={row['test_min']}, median={row['test_median']}, "
            f"max={row['test_max']}, mean={row['test_mean']}"
        )
        lines.append(
            f"train==1: {row['participants_with_train_1']}; "
            f"train<4: {row['participants_with_train_lt_4']}; "
            f"train<10: {row['participants_with_train_lt_10']}"
        )
        invalid = [p for p in s.participants if not p.split_valid]
        if invalid:
            lines.append(f"Split-invalid participants: {len(invalid)}")
            for p in invalid[:5]:
                lines.append(f"  id={p.participant_id}: {p.split_error}")
            if len(invalid) > 5:
                lines.append(f"  ... and {len(invalid) - 5} more")
        lines.append("")

    return lines


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="Summarize parsed trial counts for TEH run participants."
    )
    p.add_argument(
        "--all_in",
        action="store_true",
        help="Process all train Psych-101 datasets plus mixed_gambles (compare.py roster).",
    )
    p.add_argument(
        "--dataset",
        choices=sorted(cmp._ALL_IN_DATASETS),
        default=None,
        help="Single dataset alias (ignored when --all_in).",
    )
    p.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=("train", "test"),
        help="Psych-101 HF corpus (default: train). Ignored for mixed_gambles.",
    )
    p.add_argument("--local_dataset", default=None, help="Optional local HF dataset path.")
    p.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV path for mixed_gambles.",
    )
    p.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        help="For mixed_gambles: keep gain_loss trials only.",
    )
    p.add_argument(
        "--split_ratio",
        type=float,
        default=None,
        help=f"Override split ratio (default: metadata or {cmp._DEFAULT_SPLIT_RATIO}).",
    )
    p.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help=f"Override split seed (default: metadata or {cmp._DEFAULT_SPLIT_SEED}).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(_DEFAULT_OUT_DIR),
        help="Output directory for CSVs and report.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-dataset progress lines.")
    args = p.parse_args()

    if args.all_in and args.dataset is not None:
        raise SystemExit("Use either --all_in or --dataset, not both.")
    if not args.all_in and args.dataset is None:
        args.all_in = True

    if args.split_ratio is not None and not (0.0 < args.split_ratio < 1.0):
        raise SystemExit(f"--split_ratio must be in (0, 1), got {args.split_ratio}.")

    datasets = list(cmp._ALL_IN_DATASETS) if args.all_in else [args.dataset]
    psych_split = normalize_psych_dataset_split(args.psych_dataset_split)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir.is_absolute()
        else (repo / args.out_dir).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[_DatasetSummary] = []
    all_participants: List[_ParticipantStats] = []

    for ds in datasets:
        summary = _analyze_dataset(
            repo,
            ds,
            psych_dataset_split=psych_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=str(args.mixed_gambles_csv),
            filter_mixed_gambles=bool(args.filter_mixed_gambles),
            split_ratio_override=args.split_ratio,
            split_seed_override=args.split_seed,
            quiet=bool(args.quiet),
        )
        summaries.append(summary)
        all_participants.extend(summary.participants)

    participant_rows = [_participant_to_csv_row(p) for p in all_participants]
    summary_rows = [_summary_to_csv_row(s) for s in summaries]

    participants_path = out_dir / "trial_count_participants.csv"
    summary_path = out_dir / "trial_count_summary.csv"
    report_path = out_dir / "report.txt"

    _write_csv(participants_path, participant_rows, _PARTICIPANT_FIELDS)
    _write_csv(summary_path, summary_rows, _SUMMARY_FIELDS)

    report_lines = _build_report(repo, summaries, psych_dataset_split=psych_split)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    _print_summary_table(summaries)
    print(f"Wrote {participants_path.relative_to(repo)}")
    print(f"Wrote {summary_path.relative_to(repo)}")
    print(f"Wrote {report_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
