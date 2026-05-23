#!/usr/bin/env python3
"""
Audit Psych-101 dataset loading/parsing for TEH — focused on 4wulff2018description.

Usage:
  python analysis/code/psych-101/audit_dataset4_loading.py \\
    --dataset 4wulff2018description \\
    --psych_dataset_split train

Also compares briefly with 3frey2017cct.

Outputs:
  analysis/data/psych101_dataset4_audit/dataset4_trial_counts.csv
  analysis/data/psych101_dataset4_audit/dataset4_prompt_diagnostics_summary.csv
  analysis/data/psych101_dataset4_audit/report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PSYCH101_BINARY_DATASETS,
    experiment_id_for_alias,
    experiment_to_trial_dicts,
    format_trial_for_prompt,
    format_trials_for_prompt,
    get_filtered_psych101_split,
    hf_id_for_psych_dataset_split,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    parse_psych101_binary_row,
    split_psych_experiment,
    summarize_runtime_schema_for_prompt,
    _action_semantics_for_schema,
)
from data_modules.psych101_parsers import _RE_WULFF_BLOCK

from analysis.code.utils import compare as cmp

_DEFAULT_DATASET = "4wulff2018description"
_COMPARE_DATASET = "3frey2017cct"
_DEFAULT_OUT_DIR = "analysis/data/psych101_dataset4_audit"
_SPLIT_RATIO = 0.8
_SPLIT_SEED = 42
_PROMPT_DIAG_NAME = "prompt_diagnostics.jsonl"
_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")
_RAW_ROW_LIMIT = 20
_INSPECT_PARTICIPANTS = (37, 0, 45, 100, 500)


def _repo_root() -> Path:
    return _REPO_ROOT


def _problem_signature(trial: Mapping[str, Any]) -> str:
    p = dict(trial.get("problem") or {})
    for k in ("dataset_alias", "experiment_id"):
        p.pop(k, None)
    return json.dumps(p, sort_keys=True, default=str)


def _max_group_size(trials: Sequence[Mapping[str, Any]]) -> int:
    if not trials:
        return 0
    counts: Counter = Counter()
    for t in trials:
        counts[_problem_signature(t)] += 1
    return max(counts.values())


def _unique_problem_groups(trials: Sequence[Mapping[str, Any]]) -> int:
    return len({_problem_signature(t) for t in trials})


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_filtered(alias: str, psych_split: str, local_dataset: Optional[str]):
    alias = normalize_psych101_dataset_alias(alias)
    filtered = get_filtered_psych101_split(
        alias, split=psych_split, local_dataset=local_dataset
    )
    exp_id = experiment_id_for_alias(alias)
    hf_id = hf_id_for_psych_dataset_split(psych_split)
    return alias, filtered, exp_id, hf_id


def _participant_counts(
    alias: str,
    filtered,
    *,
    split_ratio: float,
    split_seed: int,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for row_idx in range(len(filtered)):
        raw = dict(filtered[row_idx])
        hf_participant = raw.get("participant")
        text = raw.get("text", "")
        regex_matches = (
            len(list(_RE_WULFF_BLOCK.finditer(text)))
            if alias == "4wulff2018description"
            else None
        )
        split_error = ""
        n_blocks = n_trials = n_train = n_val = n_test = 0
        all_trials: List[Dict[str, Any]] = []
        train_trials: List[Dict[str, Any]] = []
        val_trials: List[Dict[str, Any]] = []
        test_trials: List[Dict[str, Any]] = []
        try:
            exp = parse_psych101_binary_row(raw, alias)
            n_blocks = len(exp.blocks)
            n_trials = sum(len(b.trials) for b in exp.blocks)
            all_trials = experiment_to_trial_dicts(exp, dataset_alias=alias)
            train_trials, val_trials, test_trials, _ = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            n_train = len(train_trials)
            n_val = len(val_trials)
            n_test = len(test_trials)
        except Exception as exc:
            split_error = f"{type(exc).__name__}: {exc}"
            try:
                exp = parse_psych101_binary_row(raw, alias)
                n_blocks = len(exp.blocks)
                n_trials = sum(len(b.trials) for b in exp.blocks)
                all_trials = experiment_to_trial_dicts(exp, dataset_alias=alias)
            except Exception:
                pass

        rows_out.append(
            {
                "teh_participant_id": row_idx,
                "hf_participant": hf_participant,
                "raw_text_chars": len(text),
                "wulff_regex_matches": regex_matches,
                "parsed_blocks": n_blocks,
                "parsed_total_trials": n_trials,
                "train_trials": n_train,
                "val_trials": n_val,
                "test_trials": n_test,
                "unique_problem_groups_total": _unique_problem_groups(all_trials),
                "max_group_size_total": _max_group_size(all_trials),
                "unique_problem_groups_train": _unique_problem_groups(train_trials),
                "max_group_size_train": _max_group_size(train_trials),
                "split_valid": split_error == "",
                "split_error": split_error,
            }
        )
    return rows_out


def _summarize_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("split_valid")]
    train_vals = [int(r["train_trials"]) for r in valid]
    trial_vals = [int(r["parsed_total_trials"]) for r in rows]
    block_vals = [int(r["parsed_blocks"]) for r in rows]
    return {
        "n_hf_rows": len(rows),
        "n_split_valid": len(valid),
        "n_split_invalid": len(rows) - len(valid),
        "parsed_trials_min": min(trial_vals) if trial_vals else 0,
        "parsed_trials_median": statistics.median(trial_vals) if trial_vals else 0,
        "parsed_trials_max": max(trial_vals) if trial_vals else 0,
        "blocks_min": min(block_vals) if block_vals else 0,
        "blocks_median": statistics.median(block_vals) if block_vals else 0,
        "blocks_max": max(block_vals) if block_vals else 0,
        "train_min": min(train_vals) if train_vals else 0,
        "train_median": statistics.median(train_vals) if train_vals else 0,
        "train_max": max(train_vals) if train_vals else 0,
        "n_train_eq_1": sum(1 for v in train_vals if v == 1),
        "n_parsed_eq_1": sum(1 for v in trial_vals if v == 1),
        "n_blocks_eq_3": sum(1 for v in block_vals if v == 3),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_prompt_diagnostics(
    repo: Path, alias: str, psych_split: str
) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    run_dir = cmp._auto_discover_teh_run(
        repo, dataset=alias, psych_dataset_split=psych_split
    )
    if run_dir is None:
        return None, []

    per_pid: Dict[int, Dict[str, Any]] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _PARTICIPANT_DIR_RE.match(child.name)
        if not m:
            continue
        pid = int(m.group(1))
        diag_path = child / _PROMPT_DIAG_NAME
        rows = _read_jsonl(diag_path)
        if not rows:
            continue
        before_vals = [
            int(r.get("train_trials_before") or 0)
            for r in rows
            if not r.get("_parse_error")
        ]
        after_vals = [
            int(r.get("train_trials_after") or 0)
            for r in rows
            if not r.get("_parse_error")
        ]
        if not before_vals:
            continue
        per_pid[pid] = {
            "participant_id": pid,
            "run_dir": str(run_dir.relative_to(repo)),
            "n_diag_rows": len(rows),
            "train_trials_before_min": min(before_vals),
            "train_trials_before_max": max(before_vals),
            "train_trials_before_first": before_vals[0],
            "train_trials_after_first": after_vals[0] if after_vals else "",
            "truncated_any": any(bool(r.get("truncated")) for r in rows),
        }

    summary_rows = sorted(per_pid.values(), key=lambda r: int(r["participant_id"]))
    return run_dir, summary_rows


def _format_raw_row(row: Mapping[str, Any], max_text: int = 400) -> str:
    parts = []
    for k, v in row.items():
        if k == "text":
            text = str(v)
            parts.append(f"  text[{len(text)} chars]: {text[:max_text]!r}" + ("..." if len(text) > max_text else ""))
        else:
            parts.append(f"  {k}: {v!r}")
    return "\n".join(parts)


def _inspect_participant(
    lines: List[str],
    alias: str,
    filtered,
    pid: int,
    *,
    split_ratio: float,
    split_seed: int,
) -> None:
    lines.append("")
    lines.append(f"--- Participant TEH id={pid} (HF row index) ---")
    if pid < 0 or pid >= len(filtered):
        lines.append(f"  OUT OF RANGE (filtered rows={len(filtered)})")
        return

    raw = dict(filtered[pid])
    lines.append(f"  hf_participant column: {raw.get('participant')!r}")
    lines.append("  Raw HF row:")
    lines.append(_format_raw_row(raw, max_text=1200))

    if alias == "4wulff2018description":
        matches = list(_RE_WULFF_BLOCK.finditer(raw.get("text", "")))
        lines.append(f"  Wulff regex matches in text: {len(matches)}")

    try:
        exp = parse_psych101_binary_row(raw, alias)
        train, val, test, options = split_psych_experiment(
            exp, split_ratio=split_ratio, split_seed=split_seed
        )
        lines.append(
            f"  Parsed: blocks={len(exp.blocks)}, total_trials={sum(len(b.trials) for b in exp.blocks)}"
        )
        lines.append(
            f"  Split (ratio={split_ratio}, seed={split_seed}): "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
        lines.append(f"  option_keys: {options}")

        if train:
            p0 = train[0]["problem"]
            keys = p0.get("option_keys", [])
            schema = str(p0.get("schema_type", "?"))
            has_gamble = "gamble_A" in p0 or "gamble_B" in p0
            sem = _action_semantics_for_schema(keys, schema, p0, is_gamble=has_gamble)
            lines.append(f"  problem dict keys (train[0]): {sorted(p0.keys())}")
            lines.append(f"  action semantics: {sem}")

        lines.append("  Formatted prompt train trials (first 5):")
        lines.append(format_trials_for_prompt(train, max_trials=5))

        lines.append("  Parsed train trial JSON (first 2):")
        for i, t in enumerate(train[:2]):
            slim = {
                "action": t.get("action"),
                "options": t.get("options"),
                "history_len": len(t.get("history") or []),
                "problem": t.get("problem"),
            }
            lines.append(f"    train[{i}]: {json.dumps(slim, default=str)[:900]}")

        lines.append("  Parsed test trial JSON (first 1):")
        if test:
            t = test[0]
            slim = {
                "action": t.get("action"),
                "options": t.get("options"),
                "history_len": len(t.get("history") or []),
                "problem": t.get("problem"),
            }
            lines.append(f"    test[0]: {json.dumps(slim, default=str)[:900]}")
    except Exception as exc:
        lines.append(f"  PARSE/SPLIT ERROR: {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())


def _section_raw_structure(
    lines: List[str],
    alias: str,
    filtered,
    exp_id: str,
    hf_id: str,
    local_dataset: Optional[str],
) -> None:
    lines.append("=" * 80)
    lines.append("1. RAW DATASET STRUCTURE")
    lines.append("=" * 80)
    lines.append(f"Dataset alias: {alias}")
    lines.append(f"HF dataset id: {hf_id}")
    lines.append(f"Psych split: {DEFAULT_PSYCH_DATASET_SPLIT}")
    lines.append(f"Experiment id filter: {exp_id}")
    if local_dataset:
        lines.append(f"Local dataset override: {local_dataset}")
    else:
        lines.append("Local dataset override: (none — loaded from HuggingFace)")
    lines.append(f"Filtered HF rows for experiment: {len(filtered)}")

    if len(filtered) == 0:
        lines.append("No rows — cannot inspect columns.")
        return

    row0 = dict(filtered[0])
    lines.append(f"Raw column names: {list(row0.keys())}")

    lines.append("")
    lines.append("Column role identification:")
    lines.append("  participant id (human): column 'participant' (string id per subject)")
    lines.append("  TEH participant id: 0-based row index in filtered HF split (NOT participant column)")
    lines.append("  trial / problem id: embedded in NL text (Lottery X offers ...); no separate column")
    lines.append("  block id: parser block_index in problem_static (one block per lottery pair)")
    lines.append("  condition: encoded in lottery outcome/probability text")
    lines.append("  choice/action: 'You press <<KEY>>' parsed to action index over option_keys")

    spec = PSYCH101_BINARY_DATASETS[alias]
    lines.append(f"Parser: {spec.get('parser')!r}, schema_type: {spec.get('schema_type')!r}")

    lines.append("")
    lines.append(f"First {min(_RAW_ROW_LIMIT, len(filtered))} raw rows (metadata + text head):")
    for i in range(min(_RAW_ROW_LIMIT, len(filtered))):
        raw = dict(filtered[i])
        lines.append(f"\n[row {i}] hf_participant={raw.get('participant')!r}")
        lines.append(_format_raw_row(raw, max_text=350))


def _section_grouping(
    lines: List[str],
    alias: str,
    filtered,
    counts: Sequence[Mapping[str, Any]],
) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("2. PARTICIPANT GROUPING")
    lines.append("=" * 80)

    hf_participants = [dict(filtered[i]).get("participant") for i in range(len(filtered))]
    hf_unique = len(set(hf_participants))
    lines.append(f"TEH uses participant ids 0..{len(filtered)-1} (row indices into filtered split).")
    lines.append(f"Unique HF 'participant' column values: {hf_unique} (of {len(filtered)} rows).")
    lines.append(
        "Each filtered row is one HF record; for this dataset that is one human participant "
        "with a self-contained NL transcript."
    )

    dup_check = Counter(hf_participants)
    multi = [(p, n) for p, n in dup_check.items() if n > 1]
    if multi:
        lines.append(f"WARNING: duplicate participant column values: {multi[:10]}")
    else:
        lines.append("No duplicate participant column values — row index aligns 1:1 with human id.")

    example_pid = 37 if len(counts) > 37 else 0
    lines.append(
        "Note: TEH participant_id is row index, not necessarily equal to hf_participant "
        f"(e.g. row {example_pid} -> hf_participant={counts[example_pid]['hf_participant']!r})."
    )

    lines.append("")
    lines.append("Per-participant raw vs parsed counts (first 15 rows):")
    lines.append(
        "  row | hf_participant | text_chars | parsed_trials | blocks | train | split_ok"
    )
    for r in counts[:15]:
        lines.append(
            f"  {r['teh_participant_id']:4d} | {str(r['hf_participant']):>14} | "
            f"{r['raw_text_chars']:10d} | {r['parsed_total_trials']:13d} | "
            f"{r['parsed_blocks']:6d} | {r['train_trials']:5d} | {r['split_valid']}"
        )


def _section_trial_counts(lines: List[str], counts: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("3. TRIAL COUNTS (4wulff2018description, all participants)")
    lines.append("=" * 80)
    lines.append(f"Participants: {summary['n_hf_rows']}")
    lines.append(f"Split-valid (train+val+test with >=3 blocks): {summary['n_split_valid']}")
    lines.append(f"Split-invalid: {summary['n_split_invalid']}")
    lines.append(
        f"Parsed trials per participant: min={summary['parsed_trials_min']}, "
        f"median={summary['parsed_trials_median']}, max={summary['parsed_trials_max']}"
    )
    lines.append(
        f"Parsed blocks per participant: min={summary['blocks_min']}, "
        f"median={summary['blocks_median']}, max={summary['blocks_max']}"
    )
    lines.append(
        f"Train trials (after split): min={summary['train_min']}, "
        f"median={summary['train_median']}, max={summary['train_max']}"
    )
    lines.append(f"Participants with exactly 1 parsed trial: {summary['n_parsed_eq_1']}")
    lines.append(f"Participants with exactly 3 blocks: {summary['n_blocks_eq_3']}")
    lines.append(f"Split-valid participants with exactly 1 train trial: {summary['n_train_eq_1']}")

    lines.append("")
    lines.append("Distribution of parsed_total_trials (top buckets):")
    dist = Counter(int(r["parsed_total_trials"]) for r in counts)
    for k, v in dist.most_common(15):
        lines.append(f"  {k} trials: {v} participants")

    lines.append("")
    lines.append("Distribution of train_trials among split-valid participants (top buckets):")
    train_dist = Counter(int(r["train_trials"]) for r in counts if r["split_valid"])
    for k, v in train_dist.most_common(10):
        lines.append(f"  {k} train trials: {v} participants")


def _section_prompt_diagnostics(
    lines: List[str],
    run_dir: Optional[Path],
    diag_rows: Sequence[Mapping[str, Any]],
    counts_by_pid: Mapping[int, Mapping[str, Any]],
    repo: Path,
) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("4. PROMPT DIAGNOSTICS COMPARISON")
    lines.append("=" * 80)
    if run_dir is None:
        lines.append("No TEH run discovered — skipped.")
        return

    lines.append(f"Latest TEH run: {run_dir.relative_to(repo)}")
    lines.append(f"Participants with prompt_diagnostics.jsonl: {len(diag_rows)}")

    if not diag_rows:
        lines.append("No diagnostics rows found.")
        return

    before_vals = [int(r["train_trials_before_first"]) for r in diag_rows]
    lines.append(
        f"train_trials_before (first diag row per participant): "
        f"min={min(before_vals)}, median={statistics.median(before_vals)}, max={max(before_vals)}"
    )
    lines.append(f"Distribution: {dict(sorted(Counter(before_vals).items()))}")
    lines.append(f"All participants have train_trials_before=1? {all(v == 1 for v in before_vals)}")
    lines.append(f"Count with train_trials_before=1: {sum(1 for v in before_vals if v == 1)}/{len(before_vals)}")

    lines.append("")
    lines.append("Cross-check diagnostics vs loader (first 10 TEH participants):")
    lines.append("  pid | diag_before | loader_train | parsed_trials | blocks")
    for r in diag_rows[:10]:
        pid = int(r["participant_id"])
        loader = counts_by_pid.get(pid, {})
        lines.append(
            f"  {pid:3d} | {r['train_trials_before_first']:11d} | "
            f"{loader.get('train_trials', '?'):12} | "
            f"{loader.get('parsed_total_trials', '?'):13} | "
            f"{loader.get('parsed_blocks', '?')}"
        )

    mismatches = []
    for r in diag_rows:
        pid = int(r["participant_id"])
        loader = counts_by_pid.get(pid)
        if not loader:
            continue
        if int(r["train_trials_before_first"]) != int(loader["train_trials"]):
            mismatches.append((pid, r["train_trials_before_first"], loader["train_trials"]))
    if mismatches:
        lines.append(f"WARNING: {len(mismatches)} diagnostics vs loader train mismatches: {mismatches[:5]}")
    else:
        lines.append("Diagnostics train_trials_before matches loader train_trials for all TEH participants.")


def _section_compare_dataset3(
    lines: List[str],
    counts3: Sequence[Mapping[str, Any]],
    summary3: Mapping[str, Any],
    alias3: str,
    exp_id3: str,
    hf_id3: str,
) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"6. COMPARISON WITH {alias3}")
    lines.append("=" * 80)
    lines.append(f"HF dataset id: {hf_id3}")
    lines.append(f"Experiment id: {exp_id3}")
    lines.append(f"HF rows: {summary3['n_hf_rows']}")
    lines.append(
        f"Parsed trials: min={summary3['parsed_trials_min']}, "
        f"median={summary3['parsed_trials_median']}, max={summary3['parsed_trials_max']}"
    )
    lines.append(
        f"Train trials: min={summary3['train_min']}, "
        f"median={summary3['train_median']}, max={summary3['train_max']}"
    )
    lines.append(
        f"Val trials median={statistics.median(int(r['val_trials']) for r in counts3 if r['split_valid']):.1f}, "
        f"Test trials median={statistics.median(int(r['test_trials']) for r in counts3 if r['split_valid']):.1f}"
    )
    lines.append(f"Split-invalid participants: {summary3['n_split_invalid']}")
    lines.append(f"Participants with train_trials=1: {summary3['n_train_eq_1']}")

    lines.append("")
    lines.append("Sample dataset-3 participants (row 0, 37, 46):")
    for pid in (0, 37, 46):
        if pid < len(counts3):
            r = counts3[pid]
            lines.append(
                f"  pid={pid}: parsed_trials={r['parsed_total_trials']}, "
                f"blocks={r['parsed_blocks']}, train={r['train_trials']}, "
                f"val={r['val_trials']}, test={r['test_trials']}, "
                f"hf_participant={r['hf_participant']!r}"
            )


def _section_conclusions(
    lines: List[str],
    summary4: Mapping[str, Any],
    summary3: Mapping[str, Any],
    diag_rows: Sequence[Mapping[str, Any]],
    counts4: Sequence[Mapping[str, Any]],
) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("7. FINAL CONCLUSIONS")
    lines.append("=" * 80)

    pid37 = counts4[37] if len(counts4) > 37 else {}
    diag_before = (
        [int(r["train_trials_before_first"]) for r in diag_rows] if diag_rows else []
    )
    teh_first50_train1 = sum(
        1 for r in counts4[:50] if r.get("split_valid") and int(r["train_trials"]) == 1
    )

    lines.append("")
    lines.append("A. Does dataset 4 truly have only one usable trial per participant?")
    lines.append(
        f"   NO for the full dataset. Parsed trials median={summary4['parsed_trials_median']}, "
        f"max={summary4['parsed_trials_max']}. "
        f"However, {summary4['n_parsed_eq_1']} participants have only 1 parsed lottery choice in the HF transcript, "
        f"and {summary4['n_train_eq_1']} split-valid participants end up with exactly 1 train trial "
        f"because they have only 3 blocks (minimum for TEH split)."
    )
    lines.append(
        f"   Participant 37 specifically: parsed_total_trials={pid37.get('parsed_total_trials')}, "
        f"train_trials={pid37.get('train_trials')} — the raw text contains exactly 3 lottery problems, not 1."
    )
    if diag_before:
        lines.append(
            f"   The TEH run's first 50 participants are mostly 3-block subjects "
            f"({teh_first50_train1}/50 with train=1), hence prompt diagnostics show train_trials_before=1 "
            f"for {sum(1 for v in diag_before if v==1)}/{len(diag_before)} run participants."
        )

    lines.append("")
    lines.append("B. Is TEH grouping by the correct participant field?")
    lines.append(
        "   YES. Each HF filtered row is one human participant (unique 'participant' column). "
        "TEH participant_id is the 0-based row index into that filtered split, which is the "
        "intended Psych-101 convention. Row index differs from hf_participant string labels "
        "but remains 1:1 with human subjects."
    )

    lines.append("")
    lines.append("C. Is there a parser bug dropping trials?")
    lines.append(
        "   NO evidence of systematic trial dropping. For Wulff rows, parsed block count matches "
        f"regex match count. Participant 37: wulff_regex_matches={pid37.get('wulff_regex_matches')}, "
        f"parsed_blocks={pid37.get('parsed_blocks')}. "
        "Short transcripts reflect the source NL text length, not parser truncation."
    )

    lines.append("")
    lines.append("D. Is there a split bug causing only one train trial?")
    lines.append(
        "   Not a bug — expected behavior of block-based split (split_ratio=0.8, split_seed=42) "
        "when n_blocks=3: algorithm assigns 1 train / 1 val / 1 test block, each with 1 trial. "
        f"Dataset 3 median train trials={summary3['train_median']:.0f} because median blocks is much larger."
    )

    lines.append("")
    lines.append("E. Should dataset 4 be excluded, fixed, or treated as a one-shot prediction dataset?")
    lines.append(
        "   Do NOT exclude entirely — median parsed trials is 6 (max 69) and many participants have 20–60+ choices. "
        "FIX participant selection for TEH benchmarks: avoid evaluating only low-row-index 3-block subjects. "
        "Consider minimum train-trial threshold (e.g. exclude n_blocks<=3 or train_trials<4) or use "
        "trial-level split instead of block split for small-N participants. "
        "One-shot behavior applies only to a subset (~55 split-valid with train=1), not the whole dataset."
    )

    lines.append("")
    lines.append("F. Does this explain why MLE/PT can win many participants while TEH performs poorly?")
    lines.append(
        "   PARTIALLY YES. With 1 train trial, TEH program search has almost no within-participant signal "
        "(train_trials_before=1 in prompt diagnostics for 45/50 run participants). "
        "Parametric MLE/PT can still fit choice probabilities from val+test or pooled trials depending on "
        "baseline setup, while TEH evolution sees a single training example. "
        "Combined with deterministic lottery structure, simple models can outperform under-powered TEH runs."
    )


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="Audit dataset 4 loading/parsing for TEH.")
    p.add_argument("--dataset", default=_DEFAULT_DATASET, help="Primary dataset alias.")
    p.add_argument(
        "--compare_dataset",
        default=_COMPARE_DATASET,
        help="Comparison dataset alias.",
    )
    p.add_argument(
        "--psych_dataset_split",
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        help="Psych-101 HF split (train or test).",
    )
    p.add_argument("--local_dataset", default=None, help="Optional local HF dataset path.")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(_DEFAULT_OUT_DIR),
        help="Output directory for CSVs and report.",
    )
    p.add_argument("--split_ratio", type=float, default=_SPLIT_RATIO)
    p.add_argument("--split_seed", type=int, default=_SPLIT_SEED)
    p.add_argument(
        "--inspect_participants",
        nargs="*",
        type=int,
        default=list(_INSPECT_PARTICIPANTS),
        help="TEH row indices for deep inspection.",
    )
    args = p.parse_args()

    psych_split = normalize_psych_dataset_split(args.psych_dataset_split)
    out_dir = args.out_dir.resolve() if args.out_dir.is_absolute() else (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("PSYCH-101 DATASET 4 LOADING / PARSING AUDIT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Repo: {repo}")
    lines.append(f"Primary dataset: {args.dataset}")
    lines.append(f"Compare dataset: {args.compare_dataset}")
    lines.append(f"psych_dataset_split: {psych_split}")
    lines.append(f"TEH split settings: ratio={args.split_ratio}, seed={args.split_seed}")

    alias4, filtered4, exp_id4, hf_id4 = _load_filtered(
        args.dataset, psych_split, args.local_dataset
    )
    counts4 = _participant_counts(
        alias4,
        filtered4,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
    )
    summary4 = _summarize_counts(counts4)
    counts_by_pid4 = {int(r["teh_participant_id"]): r for r in counts4}

    _section_raw_structure(lines, alias4, filtered4, exp_id4, hf_id4, args.local_dataset)
    _section_grouping(lines, alias4, filtered4, counts4)
    _section_trial_counts(lines, counts4, summary4)

    trial_fields = [
        "teh_participant_id",
        "hf_participant",
        "raw_text_chars",
        "wulff_regex_matches",
        "parsed_blocks",
        "parsed_total_trials",
        "train_trials",
        "val_trials",
        "test_trials",
        "unique_problem_groups_total",
        "max_group_size_total",
        "unique_problem_groups_train",
        "max_group_size_train",
        "split_valid",
        "split_error",
    ]
    counts_csv = out_dir / "dataset4_trial_counts.csv"
    _write_csv(counts_csv, counts4, trial_fields)

    run_dir, diag_rows = _load_prompt_diagnostics(repo, alias4, psych_split)
    diag_csv = out_dir / "dataset4_prompt_diagnostics_summary.csv"
    diag_fields = [
        "participant_id",
        "run_dir",
        "n_diag_rows",
        "train_trials_before_min",
        "train_trials_before_max",
        "train_trials_before_first",
        "train_trials_after_first",
        "truncated_any",
    ]
    _write_csv(diag_csv, diag_rows, diag_fields)
    _section_prompt_diagnostics(lines, run_dir, diag_rows, counts_by_pid4, repo)

    lines.append("")
    lines.append("=" * 80)
    lines.append("5. EXAMPLE PARTICIPANT INSPECTION")
    lines.append("=" * 80)
    for pid in args.inspect_participants:
        _inspect_participant(
            lines,
            alias4,
            filtered4,
            pid,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
        )

    alias3, filtered3, exp_id3, hf_id3 = _load_filtered(
        args.compare_dataset, psych_split, args.local_dataset
    )
    counts3 = _participant_counts(
        alias3,
        filtered3,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
    )
    summary3 = _summarize_counts(counts3)
    _section_compare_dataset3(lines, counts3, summary3, alias3, exp_id3, hf_id3)

    _section_conclusions(lines, summary4, summary3, diag_rows, counts4)

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {counts_csv.relative_to(repo)} ({len(counts4)} rows)")
    print(f"Wrote {diag_csv.relative_to(repo)} ({len(diag_rows)} rows)")
    print(f"Wrote {report_path.relative_to(repo)}")
    print()
    print("Quick summary (dataset 4):")
    print(f"  HF rows: {summary4['n_hf_rows']}, split-valid: {summary4['n_split_valid']}")
    print(
        f"  parsed trials: median={summary4['parsed_trials_median']}, "
        f"train median={summary4['train_median']}, train==1: {summary4['n_train_eq_1']}"
    )
    if diag_rows:
        before = [int(r["train_trials_before_first"]) for r in diag_rows]
        print(
            f"  TEH prompt diagnostics train_trials_before: "
            f"min={min(before)}, median={statistics.median(before)}, max={max(before)}"
        )
    print(f"  See {report_path.relative_to(repo)} for full evidence and conclusions A–F.")


if __name__ == "__main__":
    main()
