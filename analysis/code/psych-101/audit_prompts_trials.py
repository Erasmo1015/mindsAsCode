#!/usr/bin/env python3
"""
Audit TEH prompts and parsed trials with evidence-backed output.

Usage:
  python analysis/code/psych-101/audit_prompts_trials.py --all_in
  python analysis/code/psych-101/audit_prompts_trials.py \\
    --datasets 8flesch2018comparing 7hilbig2014generalized --n_participants 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    format_trial_for_prompt,
    get_psych101_binary_experiment,
    split_psych_experiment,
    summarize_runtime_schema_for_prompt,
    _action_semantics_for_schema,
    _schema_b_subtype,
)
from utils.teh.teh_datasets import is_mixed_gambles_dataset

from analysis.code.utils import compare as cmp

_ALL_IN_DATASETS = cmp._ALL_IN_DATASETS
_PRIORITY_DATASETS = (
    "7hilbig2014generalized",
    "8flesch2018comparing",
    "mixed_gambles",
    "4wulff2018description",
)
_DEFAULT_OUT = "analysis/data/psych101_prompt_trial_audit.txt"
_GRAND_FAILURE_CSV = "analysis/data/psych101_grand_analysis/grand_analysis_failure_cases.csv"
_GRAND_PARTICIPANTS_CSV = (
    "analysis/data/psych101_grand_analysis/grand_analysis_participants.csv"
)
_CONVERGENCE_CSV = "generated_outputs/psych101_train/teh/iteration_convergence.csv"
_PSYCH_SPLIT = DEFAULT_PSYCH_DATASET_SPLIT
_SPLIT_RATIO = 0.8
_SPLIT_SEED = 42
_RANDOM_LOGLIK = -math.log(2)
_SNIPPET_MAX = 2000
_TRIAL_JSON_MAX = 1200
_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")
_PROMPT_DIAG_NAME = "prompt_diagnostics.jsonl"
_CASE_GLOBS = (
    "*cases*.json",
    "*trials*.json",
    "train*.json",
    "val*.json",
    "test*.json",
)

_CONF_HIGH = "HIGH"
_CONF_MED = "MEDIUM"
_CONF_LOW = "LOW"


@dataclass
class _Issue:
    confidence: str
    dataset: str
    participant_id: Optional[int]
    path: str
    reason: str
    snippet: str
    manual_check: str
    matters: str = ""


@dataclass
class _AuditWriter:
    path: Path
    lines: List[str] = field(default_factory=list)
    issues: List[_Issue] = field(default_factory=list)

    def write(self, text: str = "") -> None:
        self.lines.append(text)

    def heading(self, title: str, level: int = 1) -> None:
        if level == 1:
            self.write()
            self.write("=" * 80)
            self.write(title)
            self.write("=" * 80)
        elif level == 2:
            self.write()
            self.write("-" * 72)
            self.write(title)
            self.write("-" * 72)
        else:
            self.write()
            self.write(f"### {title}")

    def snippet_block(self, label: str, content: str, *, max_len: int = _SNIPPET_MAX) -> None:
        self.write(f"  [{label}]")
        if not content:
            self.write("    (empty)")
            return
        clipped = content if len(content) <= max_len else content[: max_len - 20] + "\n... [truncated]"
        for ln in clipped.splitlines():
            self.write(f"    {ln}")

    def json_block(self, label: str, obj: Any, *, max_len: int = _TRIAL_JSON_MAX) -> None:
        try:
            text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            text = repr(obj)
        self.snippet_block(label, text, max_len=max_len)

    def add_issue(
        self,
        *,
        confidence: str,
        dataset: str,
        participant_id: Optional[int],
        path: str,
        reason: str,
        snippet: str,
        manual_check: str,
        matters: str = "",
    ) -> None:
        self.issues.append(
            _Issue(
                confidence=confidence,
                dataset=dataset,
                participant_id=participant_id,
                path=path,
                reason=reason,
                snippet=snippet[: _SNIPPET_MAX],
                manual_check=manual_check,
                matters=matters,
            )
        )

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    return _REPO_ROOT


def _normalize_dataset(name: str) -> str:
    return cmp._normalize_compare_dataset(name)


def _list_participant_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    if not run_dir.is_dir():
        return out
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _PARTICIPANT_DIR_RE.match(child.name)
        if m:
            out.append((int(m.group(1)), child))
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_parse_error": str(exc), "_line": i, "_raw": line[:200]})
    return rows


def _load_failure_cases(path: Path) -> Dict[str, List[Dict[str, str]]]:
    by_ds: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if not path.is_file():
        return by_ds
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row.get("dataset", "").strip()
            if ds:
                by_ds[ds].append(row)
    for ds in by_ds:
        by_ds[ds].sort(
            key=lambda r: float(r.get("gap_to_best_baseline") or 0), reverse=True
        )
    return by_ds


def _load_grand_participants(path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    out: Dict[Tuple[str, int], Dict[str, str]] = {}
    if not path.is_file():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row.get("dataset", "").strip()
            pid_raw = row.get("participant_id")
            if not ds or pid_raw is None or str(pid_raw).strip() == "":
                continue
            out[(ds, int(float(pid_raw)))] = row
    return out


def _load_convergence(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = str(row.get("dataset", "")).strip()
            pid_raw = row.get("participant_id")
            if not ds or pid_raw is None:
                continue
            try:
                final = float(row["final_train_loglik"])
            except (TypeError, ValueError):
                final = None
            out[(ds, int(float(pid_raw)))] = {
                "final_train_loglik": final,
                "probably_enough": row.get("probably_enough"),
            }
    return out


def _find_case_files(participant_dir: Path) -> List[Path]:
    found: List[Path] = []
    for pat in _CASE_GLOBS:
        found.extend(participant_dir.glob(pat))
        found.extend(participant_dir.glob(f"**/{pat}"))
    return sorted({p.resolve() for p in found if p.is_file()})


def _problem_signature(trial: Mapping[str, Any]) -> str:
    p = dict(trial.get("problem") or {})
    for k in ("dataset_alias", "experiment_id"):
        p.pop(k, None)
    if "problem_signature" in trial:
        return str(trial["problem_signature"])
    return json.dumps(p, sort_keys=True, default=str)


def _load_trials(
    dataset: str,
    participant_id: int,
    *,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Return train, val, test and optional error message."""
    try:
        if is_mixed_gambles_dataset(dataset):
            train, val, test, _ = load_mixed_gambles_trials(
                participant_id,
                csv_path=mixed_gambles_csv,
                filter_gain_loss_only=filter_mixed_gambles,
                split_ratio=_SPLIT_RATIO,
                split_seed=_SPLIT_SEED,
            )
            return train, val, test, None
        exp = get_psych101_binary_experiment(
            dataset,
            int(participant_id),
            split=_PSYCH_SPLIT,
            local_dataset=local_dataset,
        )
        train, val, test, _ = split_psych_experiment(
            exp, split_ratio=_SPLIT_RATIO, split_seed=_SPLIT_SEED
        )
        return train, val, test, None
    except Exception as exc:
        return [], [], [], f"{type(exc).__name__}: {exc}"


def _summarize_prompt_diag_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    max_ratio = 0.0
    worst_before = worst_after = 0
    token_trunc = False
    trial_cap = False
    per_problem = False
    max_tokens = 0
    token_cap: Optional[int] = None
    unique_before: Set[str] = set()
    unique_after: Set[str] = set()

    for row in rows:
        if row.get("_parse_error"):
            continue
        before = int(row.get("train_trials_before") or 0)
        after = int(row.get("train_trials_after") or 0)
        if before > 0:
            ratio = 1.0 - after / before
            if ratio > max_ratio:
                max_ratio = ratio
                worst_before = before
                worst_after = after
            if before > after:
                trial_cap = True
        if row.get("truncated"):
            token_trunc = True
        steps = row.get("truncation_steps") or []
        if steps:
            token_trunc = True
        if any("per_problem_cap" in str(s) for s in steps):
            per_problem = True
        tok_b = int(row.get("prompt_tokens_before_truncation") or 0)
        tok_a = int(row.get("prompt_tokens_after_truncation") or 0)
        max_tokens = max(max_tokens, tok_a, tok_b)
        if tok_b > tok_a:
            token_trunc = True
        cap = row.get("hard_prompt_token_cap")
        if cap is not None:
            try:
                token_cap = int(cap)
            except (TypeError, ValueError):
                pass
        for key in ("unique_problems_before", "unique_problems_after"):
            if key in row and row[key] is not None:
                try:
                    (unique_before if "before" in key else unique_after).add(str(row[key]))
                except TypeError:
                    pass

    return {
        "total_train_trials": worst_before,
        "included_train_trials": worst_after,
        "excluded_train_trials": max(0, worst_before - worst_after),
        "truncation_ratio": max_ratio,
        "included_fraction": (worst_after / worst_before) if worst_before else 1.0,
        "prompt_token_estimate": max_tokens,
        "prompt_char_count": max_tokens * 4,
        "token_truncated": token_trunc,
        "trial_cap_clipping": trial_cap,
        "per_problem_cap_clipping": per_problem,
        "severe_truncation": max_ratio > 0.5,
        "n_diag_rows": len(rows),
    }


def _action_distribution(trials: Sequence[Mapping[str, Any]]) -> Counter:
    c: Counter = Counter()
    for t in trials:
        a = t.get("action")
        if a is not None:
            c[int(a)] += 1
    return c


def _trial_sanity_issues(
    dataset: str,
    participant_id: int,
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    path_prefix: str,
) -> List[_Issue]:
    issues: List[_Issue] = []
    all_trials = list(train) + list(val) + list(test)

    for split_name, trials in (("train", train), ("val", val), ("test", test)):
        actions = [t.get("action") for t in trials]
        bad_actions = [
            a for a in actions if a not in (0, 1) and a is not None
        ]
        if bad_actions:
            issues.append(
                _Issue(
                    _CONF_HIGH,
                    dataset,
                    participant_id,
                    path_prefix,
                    f"action values not in {{0,1}} in {split_name}",
                    f"unique bad actions: {sorted(set(bad_actions))[:10]}",
                    "Verify parser maps choices to 0/1 only.",
                    "Invalid actions break log-lik evaluation.",
                )
            )
        ac = _action_distribution(trials)
        if trials and len(ac) == 1:
            issues.append(
                _Issue(
                    _CONF_MED,
                    dataset,
                    participant_id,
                    path_prefix,
                    f"all {split_name} actions identical (action={list(ac.keys())[0]})",
                    f"n_trials={len(trials)}, counts={dict(ac)}",
                    "Check label parsing or degenerate participant.",
                    "Constant actions can yield ~random loglik if model varies.",
                )
            )

    train_sigs = [_problem_signature(t) for t in train]
    test_sigs = [_problem_signature(t) for t in test]
    overlap = set(train_sigs) & set(test_sigs)
    if overlap and train and test:
        issues.append(
            _Issue(
                _CONF_MED,
                dataset,
                participant_id,
                path_prefix,
                f"{len(overlap)} problem signature(s) appear in both train and test",
                f"example signature: {list(overlap)[0][:300]}",
                "TEH splits by block/signature; overlap may be OK for pseudo-blocks but verify.",
                "Unexpected overlap could indicate split leakage.",
            )
        )

    rep = Counter(train_sigs)
    heavy = [(s, n) for s, n in rep.items() if n > 20]
    if heavy:
        issues.append(
            _Issue(
                _CONF_LOW,
                dataset,
                participant_id,
                path_prefix,
                f"train has problem(s) with >20 repeated trials",
                f"top repeat: n={heavy[0][1]}, sig={heavy[0][0][:200]}",
                "Confirm per-problem cap / history structure is intended.",
                "",
            )
        )

    for i, t in enumerate(all_trials[: min(50, len(all_trials))]):
        p = t.get("problem") or {}
        if "gamble_A" in p or "gamble_B" in p:
            for gname in ("gamble_A", "gamble_B"):
                g = p.get(gname) or {}
                probs = g.get("probs")
                if probs is not None and isinstance(probs, list):
                    try:
                        s = sum(float(x) for x in probs)
                        if probs and abs(s - 1.0) > 0.05 and all(x is not None for x in probs):
                            issues.append(
                                _Issue(
                                    _CONF_LOW,
                                    dataset,
                                    participant_id,
                                    path_prefix,
                                    f"probs sum != 1 for {gname} (trial index {i})",
                                    f"probs={probs}, sum={s}",
                                    "May be intentional for unknown probs.",
                                    "",
                                )
                            )
                    except (TypeError, ValueError):
                        issues.append(
                            _Issue(
                                _CONF_HIGH,
                                dataset,
                                participant_id,
                                path_prefix,
                                f"non-numeric probabilities in {gname}",
                                f"probs={probs!r}",
                                "Inspect parser output.",
                                "",
                            )
                        )
        if not p.get("option_keys") or len(p.get("option_keys", [])) < 2:
            issues.append(
                _Issue(
                    _CONF_HIGH,
                    dataset,
                    participant_id,
                    path_prefix,
                    "missing or short option_keys",
                    json.dumps(p, default=str)[:400],
                    "Prompt semantics require two options.",
                    "",
                )
            )
            break

    return issues


def _infer_action_mapping(
    dataset: str, sample_trials: Sequence[Mapping[str, Any]], prompt_text: str
) -> Dict[str, str]:
    lines: Dict[str, str] = {"dataset": dataset}
    if not sample_trials:
        lines["note"] = "no parsed trials available"
        return lines
    p0 = sample_trials[0]["problem"]
    keys = p0.get("option_keys", [])
    schema = str(p0.get("schema_type", "?"))
    subtype = _schema_b_subtype(p0) if schema == "B" else ""
    has_gamble = "gamble_A" in p0 or "gamble_B" in p0
    sem = _action_semantics_for_schema(keys, schema, p0, is_gamble=has_gamble)
    lines["schema_type"] = schema
    lines["schema_b_subtype"] = subtype or "(n/a)"
    lines["option_keys"] = str(keys)
    lines["action_semantics_from_parser"] = sem
    lines["program_contract"] = "choose(problem, history) -> P(action=1)"
    if is_mixed_gambles_dataset(dataset):
        lines["mixed_gambles_note"] = (
            "action=0 gamble/Option A (took_gamble=1); action=1 certain/Option B (took_gamble=0); "
            "return P(action=1)=P(reject/certain)"
        )
    m0 = re.search(r"action\s*=\s*0[^\n]{0,120}", prompt_text, re.I)
    m1 = re.search(r"action\s*=\s*1[^\n]{0,120}", prompt_text, re.I)
    if m0:
        lines["prompt_action_0_line"] = m0.group(0).strip()
    if m1:
        lines["prompt_action_1_line"] = m1.group(0).strip()
    if "P(action=1)" in prompt_text or "P(action = 1)" in prompt_text:
        lines["prompt_return_contract"] = "found P(action=1) in infer_single_choice.txt"
    else:
        lines["prompt_return_contract"] = "WARNING: P(action=1) not found in prompt"
    if has_gamble and schema != "A" and not is_mixed_gambles_dataset(dataset):
        lines["schema_naming"] = (
            "schema naming mismatch: gamble_A/gamble_B field names but schema may not be "
            "literal gamble task"
        )
    return lines


def _audit_action_mapping(
    w: _AuditWriter,
    dataset: str,
    run_dir: Path,
    sample_trials: Sequence[Mapping[str, Any]],
) -> None:
    w.heading(f"Action mapping — {dataset}", level=3)
    prompt_path = run_dir / "prompts" / "infer_single_choice.txt"
    prompt_text = ""
    if prompt_path.is_file():
        prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")
        w.write(f"  prompt path: {prompt_path}")
    else:
        w.write(f"  prompt path: MISSING ({prompt_path})")

    mapping = _infer_action_mapping(dataset, sample_trials, prompt_text)
    for k, v in mapping.items():
        w.write(f"  {k}: {v}")

    if sample_trials:
        ex = sample_trials[0]
        keys = ex["problem"].get("option_keys", [])
        act = ex.get("action")
        w.write(f"  example trial: action={act}, key_for_action={keys[act] if isinstance(act, int) and act < len(keys) else '?'}")

    if is_mixed_gambles_dataset(dataset) and sample_trials:
        t = sample_trials[0]
        ga, gb = t["problem"].get("gamble_A"), t["problem"].get("gamble_B")
        w.write(f"  example gamble_A rewards={ga.get('rewards') if ga else None}")
        w.write(f"  example gamble_B rewards={gb.get('rewards') if gb else None}")

    if mapping.get("prompt_return_contract", "").startswith("WARNING"):
        w.add_issue(
            confidence=_CONF_HIGH,
            dataset=dataset,
            participant_id=None,
            path=str(prompt_path),
            reason="Prompt missing explicit P(action=1) return contract",
            snippet=prompt_text[:500],
            manual_check="Open infer_single_choice.txt and confirm return semantics.",
            matters="Evaluator assumes P(action=1).",
        )


def _select_participants(
    dataset: str,
    participant_dirs: Sequence[Tuple[int, Path]],
    *,
    failure_rows: Sequence[Mapping[str, str]],
    convergence: Mapping[Tuple[str, int], Mapping[str, Any]],
    grand_rows: Mapping[Tuple[str, int], Mapping[str, str]],
    n_participants: int,
) -> List[Tuple[int, Path, str]]:
    """Return (pid, path, selection_reason) up to n_participants."""
    by_pid = {pid: pdir for pid, pdir in participant_dirs}
    chosen: List[Tuple[int, Path, str]] = []
    used: Set[int] = set()

    def _add(pid: int, reason: str) -> None:
        if pid in used or pid not in by_pid or len(chosen) >= n_participants:
            return
        chosen.append((pid, by_pid[pid], reason))
        used.add(pid)

    if failure_rows:
        try:
            pid = int(float(failure_rows[0]["participant_id"]))
            gap = failure_rows[0].get("gap_to_best_baseline", "")
            _add(pid, f"worst TEH gap in failure_cases (gap={gap})")
        except (KeyError, ValueError):
            pass

    for pid, _ in sorted(participant_dirs):
        conv = convergence.get((dataset, pid), {})
        final = conv.get("final_train_loglik")
        if final is not None and abs(final - _RANDOM_LOGLIK) < 0.02:
            _add(pid, f"final_train_loglik≈ln(2) random baseline ({final:.6f})")
        if final is not None and final > -0.05:
            _add(pid, f"near-perfect train loglik ({final:.6f})")

    for pid, _ in sorted(participant_dirs):
        if len(chosen) >= n_participants:
            break
        _add(pid, "default sample (first available)")

    return chosen[:n_participants]


def _write_participant_snapshot(
    w: _AuditWriter,
    dataset: str,
    run_dir: Path,
    pid: int,
    pdir: Path,
    reason: str,
    *,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> None:
    w.heading(f"{dataset} / participant_{pid} — {reason}", level=3)
    w.write(f"  participant_dir: {pdir}")

    diag_path = pdir / _PROMPT_DIAG_NAME
    diag_rows = _read_jsonl(diag_path)
    w.write(f"  prompt_diagnostics: {diag_path} ({'exists' if diag_path.is_file() else 'MISSING'}, {len(diag_rows)} rows)")
    if diag_rows:
        w.write("  prompt_diagnostics raw JSON (first 5 rows):")
        for row in diag_rows[:5]:
            w.json_block("diag", row, max_len=800)
        if len(diag_rows) > 5:
            w.write(f"    ... ({len(diag_rows) - 5} more rows omitted)")
        summ = _summarize_prompt_diag_rows(diag_rows)
        w.write(f"  aggregated diagnostics: {json.dumps(summ, default=str)}")

    case_files = _find_case_files(pdir)
    if case_files:
        w.write(f"  on-disk case files: {[str(p) for p in case_files]}")
    else:
        w.write("  on-disk case files: NONE (trials loaded from HF/CSV at runtime)")

    train, val, test, err = _load_trials(
        dataset,
        pid,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
    )
    if err:
        w.write(f"  trial load ERROR: {err}")
        w.add_issue(
            confidence=_CONF_HIGH,
            dataset=dataset,
            participant_id=pid,
            path=str(pdir),
            reason=f"Could not load parsed trials: {err}",
            snippet=traceback.format_exc()[-400:],
            manual_check="Verify HF dataset access and participant id range.",
            matters="Cannot validate trials without load.",
        )
        return

    w.write(f"  parsed trial counts: train={len(train)}, val={len(val)}, test={len(test)}")
    for split_name, trials in (("train", train), ("test", test)):
        w.write(f"  action distribution ({split_name}): {dict(_action_distribution(trials))}")
        w.write(f"  unique problems ({split_name}): {len({_problem_signature(t) for t in trials})}")

    w.write("  prompt-style train examples (format_trial_for_prompt, first 3):")
    for i, t in enumerate(train[:3]):
        w.snippet_block(f"train_prompt_line_{i+1}", format_trial_for_prompt(t, i + 1), max_len=500)

    w.write("  raw parsed train trials (first 3):")
    for i, t in enumerate(train[:3]):
        slim = {
            "action": t.get("action"),
            "options": t.get("options"),
            "problem": t.get("problem"),
            "history_len": len(t.get("history") or []),
        }
        w.json_block(f"train_trial_{i+1}", slim)

    w.write("  raw parsed test trials (first 3):")
    for i, t in enumerate(test[:3]):
        slim = {
            "action": t.get("action"),
            "options": t.get("options"),
            "problem": t.get("problem"),
            "history_len": len(t.get("history") or []),
        }
        w.json_block(f"test_trial_{i+1}", slim)

    prompt_path = run_dir / "prompts" / "infer_single_choice.txt"
    if prompt_path.is_file():
        w.snippet_block("infer_single_choice.txt (head)", prompt_path.read_text(encoding="utf-8", errors="replace")[:_SNIPPET_MAX])

    results_path = pdir / "results.json"
    if results_path.is_file():
        try:
            res = json.loads(results_path.read_text(encoding="utf-8"))
            w.json_block("results.json (subset)", {k: res.get(k) for k in ("baseline", "overall_best_train", "overall_best_test") if k in res})
        except json.JSONDecodeError as exc:
            w.write(f"  results.json parse error: {exc}")

    for iss in _trial_sanity_issues(dataset, pid, train, val, test, path_prefix=str(pdir)):
        w.issues.append(iss)


def _dataset_specific_audit(
    w: _AuditWriter,
    dataset: str,
    run_dir: Path,
    all_train: List[Mapping[str, Any]],
    all_test: List[Mapping[str, Any]],
    participant_summaries: Sequence[Mapping[str, Any]],
) -> None:
    w.heading(f"Dataset-specific — {dataset}", level=3)

    if dataset == "8flesch2018comparing":
        w.write("  Focus: ~-0.69 loglik floor, tree task, pseudo-block split.")
        train_actions = _action_distribution(all_train)
        test_actions = _action_distribution(all_test)
        w.write(f"  pooled train action counts: {dict(train_actions)}")
        w.write(f"  pooled test action counts: {dict(test_actions)}")
        if all_train:
            p0 = all_train[0]["problem"]
            w.write(f"  example problem keys: {sorted(p0.keys())}")
            tf = p0.get("tree_features")
            w.write(f"  example tree_features: {tf}")
        const_fields = 0
        for t in all_train[:200]:
            tf = (t.get("problem") or {}).get("tree_features")
            if tf == {"leafiness": 0, "branchiness": 0} or tf == {"leafiness": 1, "branchiness": 1}:
                const_fields += 1
        if all_train and const_fields > len(all_train) * 0.5:
            w.add_issue(
                confidence=_CONF_MED,
                dataset=dataset,
                participant_id=None,
                path=str(run_dir),
                reason=">50% sampled train trials share identical tree_features in spot check",
                snippet=f"const_like={const_fields}/{min(200,len(all_train))}",
                manual_check="Inspect whether tree_features vary across trials.",
                matters="Low feature diversity can cap predictability.",
            )
        n_random = sum(
            1
            for s in participant_summaries
            if s.get("test_loglik") is not None
            and abs(float(s["test_loglik"]) - _RANDOM_LOGLIK) < 0.02
        )
        w.write(f"  participants with test_loglik≈-0.693 (spot from results): {n_random}")
        if n_random >= 10:
            w.add_issue(
                confidence=_CONF_MED,
                dataset=dataset,
                participant_id=None,
                path=str(run_dir / "participant_details_loglik.csv"),
                reason=f"{n_random} participants have TEH test_loglik near random chance",
                snippet="See loglik_compare CSV / results.json baseline entries",
                manual_check="Compare action balance and whether model collapses to 0.5.",
                matters="Many -0.69 scores suggest uninformative predictions or labels.",
            )

    elif dataset == "7hilbig2014generalized":
        w.write("  Focus: Centaur strong, low BIR, product-choice schema.")
        if all_train:
            acts = _action_distribution(all_train)
            w.write(f"  pooled train actions: {dict(acts)}")
            p = all_train[0]["problem"]
            w.write(f"  ratings_A sample: {p.get('ratings_A')}")
            w.write(f"  ratings_B sample: {p.get('ratings_B')}")
        entropies: List[float] = []
        for s in participant_summaries:
            pa = s.get("train_action_0_frac")
            if pa is not None:
                p0 = float(pa)
                p1 = 1 - p0
                if 0 < p0 < 1:
                    ent = -p0 * math.log2(p0) - p1 * math.log2(p1)
                    entropies.append(ent)
        if entropies:
            w.write(f"  per-participant train action entropy: mean={statistics.mean(entropies):.3f}")
        low_ent = [s for s in participant_summaries if s.get("train_action_0_frac") in (0.0, 1.0)]
        if len(low_ent) >= 5:
            w.add_issue(
                confidence=_CONF_LOW,
                dataset=dataset,
                participant_id=None,
                path=str(run_dir),
                reason=f"{len(low_ent)} participants have deterministic train actions (all 0 or all 1)",
                snippet=str(low_ent[:5]),
                manual_check="See if simple rule beats TEH; Centaur may exploit population structure.",
                matters="Highly consistent actions are not bugs but affect TEH vs Centaur.",
            )

    elif dataset == "4wulff2018description":
        w.write("  Focus: MLE/PT spread, deterministic blocks, confidence in prompt.")
        if all_train:
            w.write(f"  schema sample keys: {sorted(all_train[0]['problem'].keys())}")
        for s in participant_summaries:
            tr = s.get("train_loglik")
            te = s.get("test_loglik")
            if tr is not None and te is not None:
                try:
                    if float(tr) > -0.1 and float(te) < -1.0:
                        w.add_issue(
                            confidence=_CONF_MED,
                            dataset=dataset,
                            participant_id=int(s["participant_id"]),
                            path=str(run_dir / f"participant_{s['participant_id']}"),
                            reason="train_loglik strong but test_loglik very poor",
                            snippet=f"train={tr}, test={te}",
                            manual_check="Distribution shift between splits?",
                            matters="Possible overfit or split mismatch.",
                        )
                except (TypeError, ValueError):
                    pass

    elif dataset == "mixed_gambles":
        w.write("  Focus: action 0=gamble A, action 1=certain B, P(action=1).")
        if all_train:
            t = all_train[0]
            w.write(f"  example: action={t.get('action')}, gamble_A={t['problem'].get('gamble_A')}")
            w.write(f"  example: gamble_B={t['problem'].get('gamble_B')}")
        gains = [
            (t.get("problem") or {}).get("gamble_A", {}).get("rewards", [None])[0]
            for t in all_train[:20]
            if (t.get("problem") or {}).get("gamble_A")
        ]
        w.write(f"  first 20 gamble_A gain values: {gains}")


def _audit_dataset(
    w: _AuditWriter,
    repo: Path,
    dataset: str,
    *,
    n_participants: int,
    failure_by_ds: Mapping[str, Sequence[Mapping[str, str]]],
    convergence: Mapping[Tuple[str, int], Mapping[str, Any]],
    grand_rows: Mapping[Tuple[str, int], Mapping[str, str]],
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> None:
    alias = _normalize_dataset(dataset)
    psych_split = _PSYCH_SPLIT
    w.heading(f"Dataset: {alias}", level=2)

    teh_run = cmp._auto_discover_teh_run(repo, dataset=alias, psych_dataset_split=psych_split)
    if teh_run is None:
        root = cmp._teh_search_root(repo, alias, psych_split)
        w.write(f"  ERROR: no TEH run under {root}")
        w.add_issue(
            confidence=_CONF_HIGH,
            dataset=alias,
            participant_id=None,
            path=str(root),
            reason="No TEH run discovered",
            snippet="",
            manual_check="Confirm generated_outputs path.",
            matters="",
        )
        return

    run_name = teh_run.name if teh_run.is_dir() else teh_run.parent.name
    w.write(f"  latest run: {teh_run}")
    w.write(f"  run_name: {run_name}")

    participants = _list_participant_dirs(teh_run)
    w.write(f"  participant folders: {len(participants)}")
    if participants:
        sample_ids = [pid for pid, _ in participants[:8]]
        w.write(f"  sample participant ids: {sample_ids}{'...' if len(participants) > 8 else ''}")

    flags = {
        "prompt_diagnostics.jsonl": 0,
        "wandb_metrics.jsonl": 0,
        "results.json": 0,
        "saved_prompts_dir": int((teh_run / "prompts" / "infer_single_choice.txt").is_file()),
        "case_files_any": 0,
    }
    diag_stats: List[Dict[str, Any]] = []
    diag_by_pid: Dict[int, Dict[str, Any]] = {}
    participant_summaries: List[Dict[str, Any]] = []
    pooled_train: List[Dict[str, Any]] = []
    pooled_test: List[Dict[str, Any]] = []

    for pid, pdir in participants:
        if (pdir / _PROMPT_DIAG_NAME).is_file():
            flags["prompt_diagnostics.jsonl"] += 1
            summ = _summarize_prompt_diag_rows(_read_jsonl(pdir / _PROMPT_DIAG_NAME))
            if summ:
                summ["participant_id"] = pid
                diag_stats.append(summ)
                diag_by_pid[pid] = summ
        if (pdir / "wandb_metrics.jsonl").is_file():
            flags["wandb_metrics.jsonl"] += 1
        if (pdir / "results.json").is_file():
            flags["results.json"] += 1
        if _find_case_files(pdir):
            flags["case_files_any"] += 1

        train, val, test, err = _load_trials(
            alias,
            pid,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
        if not err and train:
            ac = _action_distribution(train)
            n = len(train)
            p0_frac = ac.get(0, 0) / n if n else None
            participant_summaries.append(
                {
                    "participant_id": pid,
                    "train_n": n,
                    "test_n": len(test),
                    "train_action_0_frac": p0_frac,
                    "truncation_ratio": diag_by_pid.get(pid, {}).get("truncation_ratio"),
                }
            )
            if len(pooled_train) < 500:
                pooled_train.extend(train[: max(0, 500 - len(pooled_train))])
            if len(pooled_test) < 200:
                pooled_test.extend(test[: max(0, 200 - len(pooled_test))])

        ll_csv = teh_run / "participant_details_loglik.csv"
        if ll_csv.is_file():
            pass  # loaded below in batch if needed

    w.write("  artifact presence (counts across participants):")
    for k, v in flags.items():
        w.write(f"    {k}: {v}/{len(participants)}")

    w.heading("Prompt diagnostics summary", level=3)
    if diag_stats:
        fracs = [d["included_fraction"] for d in diag_stats]
        ratios = [d["truncation_ratio"] for d in diag_stats]
        w.write(f"  n_with_diagnostics: {len(diag_stats)}")
        w.write(f"  avg included_fraction: {statistics.mean(fracs):.4f}")
        w.write(f"  min/max included_fraction: {min(fracs):.4f} / {max(fracs):.4f}")
        w.write(f"  avg truncation_ratio: {statistics.mean(ratios):.4f}")
        severe = [d for d in diag_stats if d.get("severe_truncation")]
        none = [d for d in diag_stats if d.get("truncation_ratio", 0) < 0.01]
        w.write(f"  severe_truncation participants: {len(severe)}")
        w.write(f"  no_truncation participants: {len(none)}")
        trial_cap = sum(1 for d in diag_stats if d.get("trial_cap_clipping"))
        token_cap = sum(1 for d in diag_stats if d.get("token_truncated"))
        w.write(f"  participants with trial-cap clipping (before>after): {trial_cap}")
        w.write(f"  participants with token-level truncation signals: {token_cap}")
        w.write("  per-participant diagnostics table (truncation_ratio, included_fraction):")
        for d in sorted(diag_stats, key=lambda x: -x.get("truncation_ratio", 0))[:15]:
            w.write(
                f"    pid={d.get('participant_id')}: trunc={d.get('truncation_ratio', 0):.3f}, "
                f"included_frac={d.get('included_fraction', 0):.3f}, "
                f"trials {d.get('included_train_trials')}/{d.get('total_train_trials')}, "
                f"trial_cap={d.get('trial_cap_clipping')}, token_trunc={d.get('token_truncated')}"
            )
    else:
        w.write("  No prompt_diagnostics.jsonl found for any participant.")

    # Load loglik from run CSV for dataset-specific
    ll_path = cmp._resolve_loglik_csv(teh_run)
    if ll_path.is_file():
        with open(ll_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid_raw = row.get("participant_id")
                if pid_raw is None:
                    continue
                try:
                    pid = int(float(pid_raw))
                except ValueError:
                    continue
                for ps in participant_summaries:
                    if ps["participant_id"] == pid:
                        ps["test_loglik"] = row.get("test_loglik")
                        ps["train_loglik"] = row.get("train_loglik")

    w.heading("Raw evidence snapshots", level=3)
    selected = _select_participants(
        alias,
        participants,
        failure_rows=failure_by_ds.get(alias, []),
        convergence=convergence,
        grand_rows=grand_rows,
        n_participants=n_participants,
    )
    if not selected:
        w.write("  (no participants to snapshot)")
    else:
        sample_trials: List[Dict[str, Any]] = []
        for pid, pdir, reason in selected:
            _write_participant_snapshot(
                w,
                alias,
                teh_run,
                pid,
                pdir,
                reason,
                local_dataset=local_dataset,
                mixed_gambles_csv=mixed_gambles_csv,
                filter_mixed_gambles=filter_mixed_gambles,
            )
            train, _, _, err = _load_trials(
                alias,
                pid,
                local_dataset=local_dataset,
                mixed_gambles_csv=mixed_gambles_csv,
                filter_mixed_gambles=filter_mixed_gambles,
            )
            if not err and train:
                sample_trials = train
                break

    w.heading("Action mapping audit", level=3)
    _audit_action_mapping(w, alias, teh_run, sample_trials)

    w.heading("Parsed trial sanity (dataset pooled sample)", level=3)
    w.write(f"  pooled train trials in sample: {len(pooled_train)}")
    w.write(f"  pooled test trials in sample: {len(pooled_test)}")
    if pooled_train:
        w.write(f"  train action dist: {dict(_action_distribution(pooled_train))}")
        w.write(f"  unique train problems: {len({_problem_signature(t) for t in pooled_train})}")
        w.write("  schema summary from trials:")
        w.snippet_block("summarize_runtime_schema", summarize_runtime_schema_for_prompt(pooled_train[:20]))
    for iss in _trial_sanity_issues(alias, None, pooled_train, [], pooled_test, path_prefix=str(teh_run)):
        w.issues.append(iss)

    _dataset_specific_audit(w, alias, teh_run, pooled_train, pooled_test, participant_summaries)


def _write_issue_list(w: _AuditWriter) -> None:
    w.heading("Suspicious issue list", level=2)
    for conf in (_CONF_HIGH, _CONF_MED, _CONF_LOW):
        bucket = [i for i in w.issues if i.confidence == conf]
        w.write(f"\n{conf} confidence ({len(bucket)} items):")
        if not bucket:
            w.write("  (none)")
            continue
        for idx, iss in enumerate(bucket, 1):
            w.write(f"\n  [{idx}] {iss.dataset}" + (f" participant_{iss.participant_id}" if iss.participant_id is not None else ""))
            w.write(f"      path: {iss.path}")
            w.write(f"      reason: {iss.reason}")
            if iss.matters:
                w.write(f"      why it matters: {iss.matters}")
            w.write(f"      manual check: {iss.manual_check}")
            w.snippet_block("evidence", iss.snippet, max_len=600)


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="Audit TEH prompts and parsed trials with evidence.")
    p.add_argument("--all_in", action="store_true", help="Audit all train psych-101 + mixed_gambles.")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Dataset aliases (default priority: {list(_PRIORITY_DATASETS)}).",
    )
    p.add_argument(
        "--n_participants",
        type=int,
        default=5,
        help="Max participants per dataset for deep snapshots.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(_DEFAULT_OUT),
        help=f"Output text path (default: {_DEFAULT_OUT}).",
    )
    p.add_argument("--local_dataset", type=str, default=None)
    p.add_argument("--mixed_gambles_csv", type=str, default=DEFAULT_CSV_PATH)
    p.add_argument("--filter_mixed_gambles", action="store_true")
    args = p.parse_args()

    if args.all_in:
        datasets = list(_ALL_IN_DATASETS)
    elif args.datasets:
        datasets = [_normalize_dataset(d) for d in args.datasets]
    else:
        datasets = list(_PRIORITY_DATASETS)

    out_path = Path(args.out).expanduser()
    out_path = out_path.resolve() if out_path.is_absolute() else (repo / out_path).resolve()

    failure_path = repo / _GRAND_FAILURE_CSV
    grand_path = repo / _GRAND_PARTICIPANTS_CSV
    conv_path = repo / _CONVERGENCE_CSV
    failure_by_ds = _load_failure_cases(failure_path)
    grand_rows = _load_grand_participants(grand_path)
    convergence = _load_convergence(conv_path)

    w = _AuditWriter(path=out_path)
    w.heading("PSYCH-101 TEH PROMPT / TRIAL AUDIT", level=1)
    w.write(f"Generated: {datetime.now().isoformat()}")
    w.write(f"Repo: {repo}")
    w.write(f"Datasets: {', '.join(datasets)}")
    w.write(f"n_participants per dataset (snapshots): {args.n_participants}")
    w.write(f"Trial split: ratio={_SPLIT_RATIO}, seed={_SPLIT_SEED}, psych_split={_PSYCH_SPLIT}")
    if not failure_path.is_file():
        w.write(f"Note: failure cases CSV not found at {failure_path}")
    if not conv_path.is_file():
        w.write(f"Note: convergence CSV not found at {conv_path}")

    w.heading("1. Dataset / run summary & audits", level=2)
    for ds in datasets:
        try:
            _audit_dataset(
                w,
                repo,
                ds,
                n_participants=max(1, int(args.n_participants)),
                failure_by_ds=failure_by_ds,
                convergence=convergence,
                grand_rows=grand_rows,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=str(args.mixed_gambles_csv),
                filter_mixed_gambles=bool(args.filter_mixed_gambles),
            )
        except Exception as exc:
            w.write(f"ERROR auditing {ds}: {type(exc).__name__}: {exc}")
            w.write(traceback.format_exc())
            w.add_issue(
                confidence=_CONF_HIGH,
                dataset=ds,
                participant_id=None,
                path=str(ds),
                reason=f"Audit crashed: {exc}",
                snippet=traceback.format_exc()[-800:],
                manual_check="Re-run with smaller scope or fix environment.",
                matters="",
            )

    _write_issue_list(w)
    w.flush()

    by_conf = Counter(i.confidence for i in w.issues)
    top = sorted(w.issues, key=lambda i: (0 if i.confidence == _CONF_HIGH else 1 if i.confidence == _CONF_MED else 2))[:5]

    print(f"Wrote audit -> {out_path}")
    print(f"Datasets audited: {len(datasets)}")
    print(f"Issues: HIGH={by_conf[_CONF_HIGH]}, MEDIUM={by_conf[_CONF_MED]}, LOW={by_conf[_CONF_LOW]}")
    print("Top issues:")
    for iss in top:
        pid_s = f" p{iss.participant_id}" if iss.participant_id is not None else ""
        print(f"  [{iss.confidence}] {iss.dataset}{pid_s}: {iss.reason[:80]}")


if __name__ == "__main__":
    main()
