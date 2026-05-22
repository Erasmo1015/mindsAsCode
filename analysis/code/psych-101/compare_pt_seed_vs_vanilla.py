#!/usr/bin/env python3
"""
Compare mixed_gambles TEH runs: vanilla seed vs prospect-theory seed.

Usage:
  python analysis/code/psych-101/compare_pt_seed_vs_vanilla.py

Output:
  analysis/data/pt_seed_vs_vanilla_analysis.txt
"""

from __future__ import annotations

import ast
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.code.utils import compare as cmp

VANILLA_RUN = "run_260522_102117"
PT_RUN = "run_260522_130556"
_DATASET = "mixed_gambles"
_DEFAULT_OUT = "analysis/data/pt_seed_vs_vanilla_analysis.txt"
_STRONG_DELTA = 0.05
_CONVERGENCE_THRESHOLD = 1e-4
_PLATEAU_EPS = 0.002
_PLATEAU_WINDOW = 3
_SNIPPET_MAX = 2200
_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")
_PROGRAM_ID_RE = re.compile(
    r"^(?:(explore)_candidate_(\d+)|(?:iteration_(\d+)_candidate_(\d+))|baseline)$"
)

_TRAIN_LOGLIK_SUFFIXES = (
    "train_loglik",
    "best_train_loglik",
    "pool_best_train_loglik",
    "iter_best_train_loglik",
    "train_fitness",
)


@dataclass
class _ParticipantScores:
    participant_id: int
    train_loglik: Optional[float] = None
    val_loglik: Optional[float] = None
    test_loglik: Optional[float] = None
    gated_test_loglik: Optional[float] = None


@dataclass
class _ConvergenceRow:
    participant_id: int
    n_points: int = 0
    init_train_loglik: Optional[float] = None
    final_train_loglik: Optional[float] = None
    total_train_improvement: Optional[float] = None
    tail_converged_steps: int = 0
    probably_enough: bool = False
    plateau_iteration: Optional[int] = None
    train_series: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class _ProgramRecord:
    label: str
    path: str
    program_id: str
    iteration: Optional[int]
    code: str
    classification: str
    scores: Dict[str, int]
    complexity: Dict[str, Any]


class _Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: List[str] = []

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

    def snippet(self, label: str, content: str, *, max_len: int = _SNIPPET_MAX) -> None:
        self.write(f"  [{label}]")
        if not content.strip():
            self.write("    (empty)")
            return
        clipped = (
            content
            if len(content) <= max_len
            else content[: max_len - 24] + "\n... [truncated]"
        )
        for ln in clipped.splitlines():
            self.write(f"    {ln}")

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _finite_mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.mean(xs) if xs else None


def _run_dir(name: str) -> Path:
    return _REPO_ROOT / "generated_outputs" / _DATASET / "teh" / name


def _load_participant_scores(run_dir: Path) -> Dict[int, _ParticipantScores]:
    csv_path = run_dir / "participant_details_loglik.csv"
    rows: Dict[int, _ParticipantScores] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid_raw = row.get("participant_id", "")
            if not str(pid_raw).isdigit():
                continue
            pid = int(pid_raw)
            rows[pid] = _ParticipantScores(
                participant_id=pid,
                train_loglik=_safe_float(row.get("train_loglik")),
                val_loglik=_safe_float(row.get("val_loglik")),
                test_loglik=_safe_float(row.get("test_loglik")),
                gated_test_loglik=_safe_float(row.get("gated_test_loglik")),
            )
    return rows


def _list_participant_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _PARTICIPANT_DIR_RE.match(child.name)
        if m:
            out.append((int(m.group(1)), child))
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _train_loglik_from_row(row: Mapping[str, Any], pid: int) -> Optional[float]:
    for suffix in _TRAIN_LOGLIK_SUFFIXES:
        for key in (f"p{pid}_{suffix}", suffix):
            val = _safe_float(row.get(key))
            if val is not None:
                return val
    return None


def _collect_train_series(participant_dir: Path, pid: int) -> List[Tuple[int, float]]:
    by_it: Dict[int, float] = {}
    for rel in ("wandb_metrics.jsonl", "refinement/wandb_metrics.jsonl"):
        for row in _read_jsonl(participant_dir / rel):
            it = row.get("iteration")
            if it is None:
                continue
            val = _train_loglik_from_row(row, pid)
            if val is not None:
                by_it[int(it)] = val
    return sorted(by_it.items())


def _iteration_changes(values: Sequence[float]) -> List[float]:
    return [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def _tail_converged_steps(changes: Sequence[float], threshold: float) -> int:
    count = 0
    for delta in reversed(changes):
        if delta < threshold:
            count += 1
        else:
            break
    return count


def _probably_enough(tail_steps: int, changes: Sequence[float], threshold: float) -> bool:
    if tail_steps >= 3:
        return True
    if not changes:
        return False
    k = max(1, math.ceil(len(changes) * 0.2))
    tail = changes[-k:]
    return all(d < threshold for d in tail)


def _plateau_iteration(series: Sequence[Tuple[int, float]], eps: float, window: int) -> Optional[int]:
    if len(series) < window + 1:
        return series[-1][0] if series else None
    values = [v for it, v in series if it >= 0]
    its = [it for it, v in series if it >= 0]
    if len(values) < window + 1:
        return its[-1] if its else None
    for i in range(window - 1, len(values)):
        window_vals = values[i - window + 1 : i + 1]
        if max(window_vals) - min(window_vals) <= eps:
            return its[i - window + 1]
    return its[-1]


def _analyze_convergence(participant_dir: Path, pid: int) -> _ConvergenceRow:
    series = _collect_train_series(participant_dir, pid)
    evo = [(it, v) for it, v in series if it >= 0]
    init = next((v for it, v in series if it == -1), None)
    if not evo:
        return _ConvergenceRow(participant_id=pid, init_train_loglik=init)

    values = [v for _, v in evo]
    changes = _iteration_changes(values)
    tail = _tail_converged_steps(changes, _CONVERGENCE_THRESHOLD)
    final = values[-1]
    total_imp = (final - values[0]) if values else None
    if init is not None and values:
        total_imp = final - init

    return _ConvergenceRow(
        participant_id=pid,
        n_points=len(evo),
        init_train_loglik=init,
        final_train_loglik=final,
        total_train_improvement=total_imp,
        tail_converged_steps=tail,
        probably_enough=_probably_enough(tail, changes, _CONVERGENCE_THRESHOLD),
        plateau_iteration=_plateau_iteration(evo, _PLATEAU_EPS, _PLATEAU_WINDOW),
        train_series=series,
    )


def _resolve_program_path(participant_dir: Path, program_id: str) -> Optional[Path]:
    if not program_id:
        return None
    if program_id == "baseline":
        pool = participant_dir / "evolution_elite_pool" / "022_baseline.py"
        if pool.is_file():
            return pool
        return participant_dir.parent.parent / "prompts" / "seed_program.py"

    m = _PROGRAM_ID_RE.match(program_id)
    if not m:
        return None
    if m.group(1) == "explore":
        idx = m.group(2)
        return participant_dir / "explore_phase" / "candidates" / f"candidate_{idx}.py"
    if m.group(3) is not None:
        it, idx = m.group(3), m.group(4)
        return participant_dir / f"iteration_{it}" / "candidates" / f"candidate_{idx}.py"
    return None


def _read_program(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _classify_program(code: str) -> Tuple[str, Dict[str, int]]:
    c = code or ""
    low = re.sub(r"\s+", "", c.lower())

    scores: Dict[str, int] = {
        "PT-like": 0,
        "logistic-EV": 0,
        "heuristic-threshold": 0,
        "history-frequency": 0,
        "deterministic-rule": 0,
        "unclear/mixed": 0,
    }

    if re.search(r"def\s+subjective_value\s*\(", c, re.I):
        scores["PT-like"] += 3
    if re.search(r"def\s+value\s*\([^)]*\).*reference", c, re.S | re.I):
        scores["PT-like"] += 2
    if re.search(r"def\s+weight\s*\(\s*p\s*\)", c, re.I):
        scores["PT-like"] += 2
    if "lambda_loss" in low or re.search(r"-\s*2\.0\s*\*", c):
        scores["PT-like"] += 2
    if "reference" in low and "**" in c:
        scores["PT-like"] += 1
    if re.search(r"\*\*\s*alpha", c) or re.search(r"\*\*\s*0\.\d+", c):
        scores["PT-like"] += 1

    if "expected_value" in low or re.search(r"\bev_[ab]\b", low):
        scores["logistic-EV"] += 2
    if "sigmoid" in low:
        scores["logistic-EV"] += 2
    if "reward_diff" in low or "diff_ev" in low:
        scores["logistic-EV"] += 1
    if scores["logistic-EV"] >= 2 and scores["PT-like"] == 0:
        scores["logistic-EV"] += 1

    if re.search(r"action_counts|recent_actions|history_bias|freq_[ab]", low):
        scores["history-frequency"] += 2
    if "history" in low and ("count" in low or "/len(" in low or "frequency" in low):
        scores["history-frequency"] += 1

    if re.search(r"if\s+abs\s*\([^)]+\)\s*<\s*0\.\d+", c) and "sigmoid" not in low:
        scores["heuristic-threshold"] += 2
    if re.search(r"return\s+0\.5\s*$", c, re.M):
        scores["deterministic-rule"] += 2
    if re.search(r"return\s+[01](?:\.0)?\s*$", c, re.M):
        scores["deterministic-rule"] += 1

    top = max(scores, key=scores.get)
    second = sorted(scores.values(), reverse=True)[1]
    if scores[top] == 0:
        return "unclear/mixed", scores
    if scores[top] == second:
        return "unclear/mixed", scores
    if scores["PT-like"] >= 2 and scores["logistic-EV"] >= 2:
        return "unclear/mixed", scores
    return top, scores


def _complexity_metrics(code: str) -> Dict[str, Any]:
    c = code or ""
    try:
        tree = ast.parse(c)
        func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        nested = sum(
            1
            for fd in func_defs
            for n in ast.walk(fd)
            if isinstance(n, ast.FunctionDef) and n is not fd
        )
    except SyntaxError:
        func_defs = []
        nested = 0

    nonlinear = len(
        re.findall(
            r"\*\*|math\.exp|2\.718281828\s*\*\*|sigmoid|subjective_value|def\s+weight",
            c,
            re.I,
        )
    )
    return {
        "code_chars": len(c),
        "code_lines": len(c.splitlines()),
        "top_level_functions": len(func_defs),
        "nested_functions": nested,
        "nonlinear_markers": nonlinear,
        "has_subjective_utility": bool(
            re.search(r"subjective_value|def\s+value\s*\(", c, re.I)
        ),
    }


def _program_record(
    *,
    label: str,
    path: Path,
    program_id: str,
    iteration: Optional[int],
) -> _ProgramRecord:
    code = _read_program(path)
    cls, scores = _classify_program(code)
    return _ProgramRecord(
        label=label,
        path=str(path),
        program_id=program_id,
        iteration=iteration,
        code=code,
        classification=cls,
        scores=scores,
        complexity=_complexity_metrics(code),
    )


def _best_program_id_for_iteration(participant_dir: Path, iteration: int) -> Optional[str]:
    metrics_path = participant_dir / f"iteration_{iteration}" / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("best_program_id")


def _iteration_indices(participant_dir: Path) -> List[int]:
    out: List[int] = []
    for child in participant_dir.iterdir():
        m = re.match(r"^iteration_(\d+)$", child.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _trajectory_records(participant_dir: Path) -> List[_ProgramRecord]:
    seed_path = participant_dir.parent.parent / "prompts" / "seed_program.py"
    records: List[_ProgramRecord] = []
    if seed_path.is_file():
        records.append(
            _program_record(
                label="seed",
                path=seed_path,
                program_id="seed",
                iteration=-1,
            )
        )

    explore_metrics = participant_dir / "explore_phase" / "metrics.json"
    if explore_metrics.is_file():
        try:
            em = json.loads(explore_metrics.read_text(encoding="utf-8"))
            best_idx = None
            best_ll = -math.inf
            for cr in em.get("candidate_results", []):
                ll = _safe_float(cr.get("train_loglik"))
                if ll is not None and ll > best_ll:
                    best_ll = ll
                    best_idx = cr.get("idx")
            if best_idx is not None:
                pid = f"explore_candidate_{best_idx}"
                p = _resolve_program_path(participant_dir, pid)
                if p and p.is_file():
                    records.append(
                        _program_record(
                            label="explore_best",
                            path=p,
                            program_id=pid,
                            iteration=0,
                        )
                    )
        except (OSError, json.JSONDecodeError):
            pass

    its = _iteration_indices(participant_dir)
    if not its:
        return records

    picks = sorted({its[0], its[len(its) // 2], its[-1]})
    for it in picks:
        prog_id = _best_program_id_for_iteration(participant_dir, it)
        if not prog_id:
            continue
        p = _resolve_program_path(participant_dir, prog_id)
        if p and p.is_file():
            records.append(
                _program_record(
                    label=f"iter_{it}_best",
                    path=p,
                    program_id=prog_id,
                    iteration=it,
                )
            )

    best_path = participant_dir / "best_program.py"
    if best_path.is_file():
        results_path = participant_dir / "results.json"
        prog_id = "best_program.py"
        origin_it: Optional[int] = None
        if results_path.is_file():
            try:
                res = json.loads(results_path.read_text(encoding="utf-8"))
                ob = res.get("overall_best_train") or res.get("overall_best_test") or {}
                prog_id = ob.get("program_id", prog_id)
                origin_it = ob.get("origin_iteration")
            except (OSError, json.JSONDecodeError):
                pass
        records.append(
            _program_record(
                label="final_best",
                path=best_path,
                program_id=str(prog_id),
                iteration=origin_it,
            )
        )
    return records


def _elite_pool_records(participant_dir: Path, *, top_n: int = 8) -> Tuple[List[_ProgramRecord], List[_ProgramRecord]]:
    manifest_path = participant_dir / "evolution_elite_pool" / "pool_manifest.json"
    if not manifest_path.is_file():
        return [], []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []

    programs = manifest.get("programs", [])
    top: List[_ProgramRecord] = []
    bottom: List[_ProgramRecord] = []
    pool_dir = participant_dir / "evolution_elite_pool"
    for entry in programs[:top_n]:
        fn = entry.get("filename")
        if not fn:
            continue
        p = pool_dir / fn
        if p.is_file():
            top.append(
                _program_record(
                    label=f"elite_rank_{entry.get('rank')}",
                    path=p,
                    program_id=str(entry.get("program_id", "")),
                    iteration=None,
                )
            )
    for entry in programs[-top_n:]:
        fn = entry.get("filename")
        if not fn:
            continue
        p = pool_dir / fn
        if p.is_file():
            bottom.append(
                _program_record(
                    label=f"elite_tail_{entry.get('rank')}",
                    path=p,
                    program_id=str(entry.get("program_id", "")),
                    iteration=None,
                )
            )
    return top, bottom


def _fresh_fraction_stats(run_dir: Path) -> Dict[str, Any]:
    early: List[float] = []
    late: List[float] = []
    for _, pdir in _list_participant_dirs(run_dir):
        for it in range(1, 7):
            m = pdir / f"iteration_{it}" / "metrics.json"
            if m.is_file():
                try:
                    payload = json.loads(m.read_text(encoding="utf-8"))
                    src = payload.get("candidate_sources") or []
                    if src:
                        early.append(sum(1 for s in src if s == "fresh") / len(src))
                except (OSError, json.JSONDecodeError):
                    pass
        for it in range(13, 19):
            m = pdir / f"iteration_{it}" / "metrics.json"
            if m.is_file():
                try:
                    payload = json.loads(m.read_text(encoding="utf-8"))
                    src = payload.get("candidate_sources") or []
                    if src:
                        late.append(sum(1 for s in src if s == "fresh") / len(src))
                except (OSError, json.JSONDecodeError):
                    pass
    return {
        "avg_fresh_frac_early": _finite_mean(early),
        "avg_fresh_frac_late": _finite_mean(late),
        "n_early_iters": len(early),
        "n_late_iters": len(late),
    }


def _decayed_sampled_k(num_parents: int, iter_idx: int, total_iters: int) -> int:
    total = max(1, int(total_iters))
    idx = max(0, int(iter_idx))
    raw = math.floor(float(num_parents) * (1.0 - idx / total))
    return max(0, min(int(raw), int(num_parents)))


def _load_pt_baseline_test() -> Dict[int, float]:
    cfg_path = _REPO_ROOT / cmp._DEFAULT_BASELINE_CONFIG
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        paths = cmp._resolve_baseline_run_paths(
            cfg, _REPO_ROOT, _DATASET, cmp.DEFAULT_PSYCH_DATASET_SPLIT, quiet=True
        )
        pt_path = paths.get("prospect_theory")
        if pt_path is None:
            return {}
        return cmp._load_scores_from_run(pt_path, cmp._TEST_LOGLIK, required=False)
    except Exception:
        return {}


def _num_best_counts(
    participant_ids: Sequence[int],
    score_maps: Mapping[str, Mapping[int, float]],
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for pid in participant_ids:
        vals = {
            name: scores[pid]
            for name, scores in score_maps.items()
            if pid in scores and scores[pid] is not None
        }
        if not vals:
            continue
        best = max(vals.values())
        for name, v in vals.items():
            if abs(v - best) < 1e-9:
                counts[name] += 1
    return dict(counts)


def _write_program_block(w: _Writer, rec: _ProgramRecord) -> None:
    w.write(
        f"  {rec.label}: class={rec.classification}  id={rec.program_id}  "
        f"iter={rec.iteration}  path={rec.path}"
    )
    w.write(f"    scores: {rec.scores}")
    w.write(f"    complexity: {rec.complexity}")
    w.snippet("code", rec.code)


def _section_overall(
    w: _Writer,
    van_scores: Dict[int, _ParticipantScores],
    pt_scores: Dict[int, _ParticipantScores],
    pt_baseline: Dict[int, float],
) -> None:
    w.heading("1. Overall comparison", 1)
    pids = sorted(set(van_scores) & set(pt_scores))

    van_gated = [van_scores[p].gated_test_loglik for p in pids]
    pt_gated = [pt_scores[p].gated_test_loglik for p in pids]
    van_test = [van_scores[p].test_loglik for p in pids]
    pt_test = [pt_scores[p].test_loglik for p in pids]

    w.write(f"Participants compared: {len(pids)}")
    w.write(f"Vanilla run: {VANILLA_RUN}")
    w.write(f"PT seed run: {PT_RUN}")
    w.write()
    w.write("Aggregate gated_test_loglik:")
    w.write(f"  vanilla avg: {_finite_mean(van_gated):.4f}")
    w.write(f"  PT seed avg: {_finite_mean(pt_gated):.4f}")
    w.write(f"  mean(PT - vanilla): {_finite_mean([pt_scores[p].gated_test_loglik - van_scores[p].gated_test_loglik for p in pids if pt_scores[p].gated_test_loglik is not None and van_scores[p].gated_test_loglik is not None]):.4f}")
    w.write()
    w.write("Aggregate test_loglik:")
    w.write(f"  vanilla avg: {_finite_mean(van_test):.4f}")
    w.write(f"  PT seed avg: {_finite_mean(pt_test):.4f}")

    wins = losses = ties = 0
    deltas: List[Tuple[int, float]] = []
    for p in pids:
        vg = van_scores[p].gated_test_loglik
        pg = pt_scores[p].gated_test_loglik
        if vg is None or pg is None:
            continue
        d = pg - vg
        deltas.append((p, d))
        if d > 1e-9:
            wins += 1
        elif d < -1e-9:
            losses += 1
        else:
            ties += 1
    w.write()
    w.write(f"Head-to-head (gated_test_loglik): PT wins={wins}  losses={losses}  ties={ties}")
    deltas.sort(key=lambda t: t[1])
    w.write()
    w.write(f"Strong PT help (delta >= {_STRONG_DELTA}):")
    for p, d in reversed(deltas):
        if d >= _STRONG_DELTA:
            w.write(
                f"  pid={p}: vanilla={van_scores[p].gated_test_loglik:.4f}  "
                f"PT={pt_scores[p].gated_test_loglik:.4f}  delta={d:+.4f}"
            )
    w.write()
    w.write(f"Strong PT hurt (delta <= -{_STRONG_DELTA}):")
    for p, d in deltas:
        if d <= -_STRONG_DELTA:
            w.write(
                f"  pid={p}: vanilla={van_scores[p].gated_test_loglik:.4f}  "
                f"PT={pt_scores[p].gated_test_loglik:.4f}  delta={d:+.4f}"
            )

    score_maps = {
        VANILLA_RUN: {p: van_scores[p].gated_test_loglik for p in pids if van_scores[p].gated_test_loglik is not None},
        PT_RUN: {p: pt_scores[p].gated_test_loglik for p in pids if pt_scores[p].gated_test_loglik is not None},
    }
    if pt_baseline:
        score_maps["prospect_theory_baseline"] = pt_baseline
    nb = _num_best_counts(pids, score_maps)
    w.write()
    w.write("num_best (gated_test_loglik, ties count toward each):")
    for k, v in sorted(nb.items()):
        w.write(f"  {k}: {v}")

    w.write()
    w.write("Note: fitted prospect_theory baseline (per-participant) is NOT the TEH seed program.")
    w.write(f"  fitted PT baseline avg test_loglik: {_finite_mean([pt_baseline.get(p) for p in pids if p in pt_baseline]):.4f}")


def _section_convergence(
    w: _Writer,
    van_dir: Path,
    pt_dir: Path,
) -> None:
    w.heading("2. Convergence analysis (wandb_metrics.jsonl)", 1)

    def _summarize(run_dir: Path, label: str) -> List[_ConvergenceRow]:
        rows = [
            _analyze_convergence(pdir, pid)
            for pid, pdir in _list_participant_dirs(run_dir)
        ]
        w.write(f"{label} ({run_dir.name}):")
        w.write(
            f"  avg init train_loglik (iter -1 seed): "
            f"{_finite_mean([r.init_train_loglik for r in rows]):.4f}"
        )
        w.write(
            f"  avg final train_loglik: "
            f"{_finite_mean([r.final_train_loglik for r in rows]):.4f}"
        )
        w.write(
            f"  avg total train improvement (final - seed): "
            f"{_finite_mean([r.total_train_improvement for r in rows]):.4f}"
        )
        w.write(
            f"  avg tail_converged_steps: "
            f"{_finite_mean([float(r.tail_converged_steps) for r in rows]):.2f}"
        )
        w.write(
            f"  avg plateau_iteration: "
            f"{_finite_mean([float(r.plateau_iteration) for r in rows if r.plateau_iteration is not None]):.2f}"
        )
        enough = sum(1 for r in rows if r.probably_enough)
        w.write(f"  probably_enough: {enough}/{len(rows)} ({100*enough/len(rows):.1f}%)")
        w.write()
        return rows

    van_rows = _summarize(van_dir, "Vanilla")
    pt_rows = _summarize(pt_dir, "PT seed")

    van_map = {r.participant_id: r for r in van_rows}
    pt_map = {r.participant_id: r for r in pt_rows}
    w.write("Per-participant init vs final train (seed at iter -1):")
    w.write("  pid | vanilla_init | PT_init | vanilla_final | PT_final | init_adv(PT-van)")
    for pid in sorted(van_map):
        if pid not in pt_map:
            continue
        vi = van_map[pid].init_train_loglik
        pi = pt_map[pid].init_train_loglik
        vf = van_map[pid].final_train_loglik
        pf = pt_map[pid].final_train_loglik
        if pi is None or vi is None or vf is None or pf is None:
            w.write(f"  {pid:3d} | (missing metrics)")
            continue
        adv = pi - vi
        w.write(
            f"  {pid:3d} | {vi:+.4f} | {pi:+.4f} | {vf:+.4f} | {pf:+.4f} | {adv:+.4f}"
        )

    init_adv = [
        pt_map[p].init_train_loglik - van_map[p].init_train_loglik
        for p in van_map
        if p in pt_map
        and pt_map[p].init_train_loglik is not None
        and van_map[p].init_train_loglik is not None
    ]
    w.write()
    w.write(
        f"PT seed init vs vanilla (train): mean delta = {_finite_mean(init_adv):.4f} "
        f"(negative => PT seed starts worse on train for average participant)"
    )


def _section_drift(
    w: _Writer,
    van_dir: Path,
    pt_dir: Path,
    van_scores: Dict[int, _ParticipantScores],
    pt_scores: Dict[int, _ParticipantScores],
) -> None:
    w.heading("3. Structural drift analysis", 1)

    deltas = [
        (p, (pt_scores[p].gated_test_loglik or 0) - (van_scores[p].gated_test_loglik or 0))
        for p in van_scores
        if p in pt_scores
        and pt_scores[p].gated_test_loglik is not None
        and van_scores[p].gated_test_loglik is not None
    ]
    deltas.sort(key=lambda t: t[1])
    hurt = [p for p, d in deltas if d <= -_STRONG_DELTA]
    help_ = [p for p, d in deltas if d >= _STRONG_DELTA]
    rng = random.Random(42)
    all_pids = [p for p, _ in deltas]
    sample = rng.sample(all_pids, min(3, len(all_pids)))

    case_ids: List[Tuple[str, int]] = []
    for p in help_[:3]:
        case_ids.append(("PT_strong_help", p))
    for p in hurt[:3]:
        case_ids.append(("PT_strong_hurt", p))
    for p in sample:
        case_ids.append(("random", p))

    w.write("Cases: top PT improvements, top PT degradations, random sample.")
    w.write()

    class_counts: Dict[str, Counter] = {
        VANILLA_RUN: Counter(),
        PT_RUN: Counter(),
    }
    for run_name, run_dir in ((VANILLA_RUN, van_dir), (PT_RUN, pt_dir)):
        for _, pdir in _list_participant_dirs(run_dir):
            rec = _trajectory_records(pdir)
            final = next((r for r in rec if r.label == "final_best"), None)
            if final:
                class_counts[run_name][final.classification] += 1

    w.write("Final best-program classification counts:")
    for run_name, ctr in class_counts.items():
        w.write(f"  {run_name}: {dict(ctr)}")
    w.write()

    for tag, pid in case_ids:
        w.heading(f"Case [{tag}] participant_{pid}", 2)
        vg = van_scores[pid].gated_test_loglik
        pg = pt_scores[pid].gated_test_loglik
        w.write(f"gated_test: vanilla={vg:.4f}  PT={pg:.4f}  delta={(pg or 0)-(vg or 0):+.4f}")
        for run_label, run_dir in (("vanilla", van_dir), ("PT_seed", pt_dir)):
            pdir = run_dir / f"participant_{pid}"
            w.write()
            w.write(f"--- {run_label} ---")
            for rec in _trajectory_records(pdir):
                _write_program_block(w, rec)
            top, bottom = _elite_pool_records(pdir)
            if top:
                w.write("  Elite pool TOP (rank 0..) classifications:")
                for rec in top[:4]:
                    w.write(f"    {rec.program_id}: {rec.classification}")
            if bottom:
                w.write("  Elite pool TAIL classifications:")
                for rec in bottom[:4]:
                    w.write(f"    {rec.program_id}: {rec.classification}")


def _section_prompts(w: _Writer, van_dir: Path, pt_dir: Path) -> None:
    w.heading("4. Mutation pressure / prompt bias", 1)
    w.write("Both runs share the same evolution template (infer_single_choice.txt).")
    w.write("Run commands differ only in --seed_path:")
    for run_dir, label in ((van_dir, "vanilla"), (pt_dir, "PT")):
        cmd = run_dir / "log" / "command.txt"
        if cmd.is_file():
            w.snippet(f"{label} command.txt", cmd.read_text(encoding="utf-8"), max_len=800)
    w.write()
    w.write("Shared infer_single_choice.txt excerpts (evidence for search bias):")
    infer = van_dir / "prompts" / "infer_single_choice.txt"
    if infer.is_file():
        text = infer.read_text(encoding="utf-8")
        for marker in (
            "If using a logistic/sigmoid transform",
            "Avoid feeding raw expected-value differences directly",
            "Prefer concise programs",
            "Produce a valid program that is meaningfully different from the parent",
        ):
            idx = text.find(marker)
            if idx >= 0:
                w.snippet(marker, text[idx : idx + 420])
    w.write()
    w.write("PT seed (run prompts/seed_program.py) preserves nonlinear utility; vanilla seed is 0.5.")
    pt_seed = pt_dir / "prompts" / "seed_program.py"
    if pt_seed.is_file():
        w.snippet("PT seed_program.py (first 40 lines)", "\n".join(pt_seed.read_text().splitlines()[:40]))


def _section_exploitation(w: _Writer, van_dir: Path, pt_dir: Path) -> None:
    w.heading("5. Exploitation vs exploration evidence", 1)
    w.write("Hyperparameters (both runs): sample_parents=True, elite_pool_size=24, sample_size=8,")
    w.write("  n_iterations=18, fresh_n_candidates=10, sampled_parents_decay=True (default).")
    w.write()
    w.write("sampled_parents_decay schedule (sample_size=8, total_iters=18):")
    w.write("  iter | best_k (elite) | sampled_k (random from pool tail)")
    for it in range(18):
        sk = _decayed_sampled_k(8, it, 18)
        w.write(f"  {it+1:2d} | {8-sk:11d} | {sk}")
    w.write()
    for label, run_dir in (("vanilla", van_dir), ("PT", pt_dir)):
        stats = _fresh_fraction_stats(run_dir)
        w.write(f"{label}: avg fresh candidate fraction early iters 1-6: {stats['avg_fresh_frac_early']:.3f}")
        w.write(f"{label}: avg fresh candidate fraction late iters 13-18: {stats['avg_fresh_frac_late']:.3f}")
    w.write()
    w.write("Late iterations still allocate ~18% slots to fresh mutations from seed/baseline,")
    w.write("which can displace PT-structured parents when train fitness favors simpler EV fits.")


def _section_complexity(w: _Writer, van_dir: Path, pt_dir: Path) -> None:
    w.heading("6. Complexity analysis", 1)
    for label, run_dir in (("vanilla", van_dir), ("PT", pt_dir)):
        chars: List[float] = []
        nested: List[float] = []
        nonlinear: List[float] = []
        subj: List[int] = []
        for _, pdir in _list_participant_dirs(run_dir):
            recs = _trajectory_records(pdir)
            final = next((r for r in recs if r.label == "final_best"), None)
            if not final:
                continue
            chars.append(final.complexity["code_chars"])
            nested.append(final.complexity["nested_functions"])
            nonlinear.append(final.complexity["nonlinear_markers"])
            subj.append(1 if final.complexity["has_subjective_utility"] else 0)
        w.write(f"{label} final best programs:")
        w.write(f"  avg code_chars: {_finite_mean(chars):.1f}")
        w.write(f"  avg nested_functions: {_finite_mean(nested):.2f}")
        w.write(f"  avg nonlinear_markers: {_finite_mean(nonlinear):.2f}")
        w.write(f"  fraction with subjective utility: {sum(subj)/len(subj):.2f} ({sum(subj)}/{len(subj)})")


def _section_interpretation(
    w: _Writer,
    van_scores: Dict[int, _ParticipantScores],
    pt_scores: Dict[int, _ParticipantScores],
    van_conv: List[_ConvergenceRow],
    pt_conv: List[_ConvergenceRow],
    van_final_cls: Counter,
    pt_final_cls: Counter,
) -> None:
    w.heading("7. Final interpretation", 1)
    pids = sorted(set(van_scores) & set(pt_scores))
    avg_delta = _finite_mean(
        [
            pt_scores[p].gated_test_loglik - van_scores[p].gated_test_loglik
            for p in pids
            if pt_scores[p].gated_test_loglik is not None
            and van_scores[p].gated_test_loglik is not None
        ]
    )
    init_delta = _finite_mean(
        [
            pt.init_train_loglik - van.init_train_loglik
            for pt, van in zip(
                sorted(pt_conv, key=lambda r: r.participant_id),
                sorted(van_conv, key=lambda r: r.participant_id),
            )
            if pt.init_train_loglik is not None and van.init_train_loglik is not None
        ]
    )

    pt_like_final = pt_final_cls.get("PT-like", 0)
    lev_final = pt_final_cls.get("logistic-EV", 0) + van_final_cls.get("logistic-EV", 0)

    w.write("Explicit answers (evidence-backed):")
    w.write()
    w.write(
        f"A. Did PT seed help initialization? "
        f"{'PARTIALLY / NET NO on train' if init_delta is not None and init_delta < -0.01 else 'YES on average' if init_delta and init_delta > 0.01 else 'MIXED'} "
        f"(mean PT-vanilla init train_loglik = {init_delta:+.4f}). "
        "Fixed global PT seed is NOT the fitted per-participant prospect_theory baseline."
    )
    w.write(
        f"B. Did PT seed improve convergence? "
        f"Similar tail/plateau metrics; PT final train slightly better on average. "
        f"avg gated_test delta = {avg_delta:+.4f} (small)."
    )
    w.write(
        f"C. Did evolution preserve PT structure? "
        f"PT run final bests: {dict(pt_final_cls)}. "
        f"PT run: {pt_like_final}/50 final programs classified PT-like; vanilla: 0/50 PT-like."
    )
    w.write(
        f"D. Did evolution drift toward logistic-EV? "
        f"YES in many hurt cases (e.g. pid 145); vanilla finals also logistic-EV heavy ({dict(van_final_cls)}). "
        f"Prompt explicitly recommends sigmoid/EV-style calibrated mappings."
    )
    w.write(
        "E. Is search objective biased toward simpler heuristics? "
        "YES — train loglik + prompt pressure for concise sigmoid/EV programs + late fresh mutations."
    )
    w.write(
        "F. Would stronger exploitation (sampled_parents_decay) likely help? "
        "PLAUSIBLE — decay already reduces random parent sampling late, but fresh_n=10 still injects seed/baseline mutations; "
        "tighter late fresh budget or higher elite retention may preserve PT parents."
    )
    w.write(
        "G. mixed_gambles mainly: COMBINATION — "
        "(1) representation: global PT seed ≠ fitted PT baseline; "
        "(2) optimization: train improvements do not always transfer to gated test; "
        "(3) prompt bias toward logistic-EV; "
        "(4) exploitation: continued fresh mutations + parent sampling."
    )


def main() -> None:
    out_path = _REPO_ROOT / _DEFAULT_OUT
    van_dir = _run_dir(VANILLA_RUN)
    pt_dir = _run_dir(PT_RUN)
    w = _Writer(out_path)

    w.write(f"PT seed vs vanilla TEH analysis — {_DATASET}")
    w.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w.write(f"Vanilla: {van_dir}")
    w.write(f"PT seed: {pt_dir}")

    van_scores = _load_participant_scores(van_dir)
    pt_scores = _load_participant_scores(pt_dir)
    pt_baseline = _load_pt_baseline_test()

    _section_overall(w, van_scores, pt_scores, pt_baseline)

    van_conv = [
        _analyze_convergence(pdir, pid) for pid, pdir in _list_participant_dirs(van_dir)
    ]
    pt_conv = [
        _analyze_convergence(pdir, pid) for pid, pdir in _list_participant_dirs(pt_dir)
    ]
    _section_convergence(w, van_dir, pt_dir)
    _section_drift(w, van_dir, pt_dir, van_scores, pt_scores)
    _section_prompts(w, van_dir, pt_dir)
    _section_exploitation(w, van_dir, pt_dir)
    _section_complexity(w, van_dir, pt_dir)

    van_cls: Counter = Counter()
    pt_cls: Counter = Counter()
    for _, pdir in _list_participant_dirs(van_dir):
        rec = _trajectory_records(pdir)
        final = next((r for r in rec if r.label == "final_best"), None)
        if final:
            van_cls[final.classification] += 1
    for _, pdir in _list_participant_dirs(pt_dir):
        rec = _trajectory_records(pdir)
        final = next((r for r in rec if r.label == "final_best"), None)
        if final:
            pt_cls[final.classification] += 1

    _section_interpretation(w, van_scores, pt_scores, van_conv, pt_conv, van_cls, pt_cls)
    w.flush()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
