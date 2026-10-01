#!/usr/bin/env python3
"""Read-only mid-run check for one AAMAS v0 Slurm job.

Inspects Slurm state, the job log tail, saved policy/provenance, one prompt
example per reached phase, a few selection rows, and partial MEM scores.
Does not reconstruct prompts or re-tokenize them.

  python analysis/aamas_v0_midrun_sentinel.py --job_id 307707
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PACK = "aamas_v0_greedy_pack_v1"
POLICY = "aamas_v0_lossless_data_v1"
CAP = 14000
CONTEXT = 16384
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "generated_outputs" / "psych101_train" / "teh"
LOG_ROOT = Path("/careAIDrive/zichang/logs/x_log")
FATAL_RE = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|"
    r"vLLM died|vLLM timeout|EngineCore|RuntimeError:|Killed",
    re.I,
)
PID_IN_BLOCK = re.compile(r"^(?:unit|indep|session):(\d+)")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    partial = 0
    if not path.is_file():
        return rows, partial
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows, partial
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                partial += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows, partial


def _sample(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    if not rows:
        return []
    picks = [("early", rows[0])]
    if len(rows) > 2:
        picks.append(("middle", rows[len(rows) // 2]))
    if len(rows) > 1:
        picks.append(("latest", rows[-1]))
    return picks


def _slurm(job_id: str) -> Dict[str, str]:
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%M|%j|%R"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "UNKNOWN", "elapsed": "", "name": "", "reason": str(exc)}
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return {"state": "NOT_IN_QUEUE", "elapsed": "", "name": "", "reason": ""}
    parts = line[-1].split("|", 3)
    while len(parts) < 4:
        parts.append("")
    return {"state": parts[0], "elapsed": parts[1], "name": parts[2], "reason": parts[3]}


def _locate(job_id: str) -> List[Path]:
    found: List[Path] = []
    if not OUT_ROOT.is_dir():
        return found
    name = f"job_{job_id}"
    for dataset in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()):
        for kind in sorted(dataset.glob("pics_aamas_v0_*")):
            hit = kind / name
            if hit.is_dir():
                found.append(hit)
    return found


def _log_path(job_id: str) -> Optional[Path]:
    user = os.environ.get("USER", "")
    candidates = [
        LOG_ROOT / f"{user}.{job_id}.out",
        LOG_ROOT / f"zichangge.{job_id}.out",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


class Check:
    def __init__(self) -> None:
        self.level = "PASS"
        self.notes: List[str] = []

    def add(self, level: str, note: str) -> None:
        rank = {"PASS": 0, "PENDING": 1, "WARN": 2, "FAIL": 3}
        if rank[level] > rank[self.level]:
            self.level = level
        self.notes.append(f"{level}: {note}")


def _check_log(job_id: str, check: Check, slurm: Dict[str, str]) -> None:
    path = _log_path(job_id)
    check.notes.append(
        f"slurm {slurm['state']} elapsed {slurm['elapsed'] or '—'} name {slurm['name'] or '—'}"
    )
    if slurm["state"] in {"PENDING", "PD"}:
        check.add("PENDING", "job has not started")
        return
    if path is None:
        if slurm["state"] == "RUNNING":
            check.add("WARN", "no job log yet")
        return
    size_before = path.stat().st_size
    mtime = path.stat().st_mtime
    text = path.read_text(encoding="utf-8", errors="replace")
    tail = text[-200_000:]
    if FATAL_RE.search(tail):
        check.add("FAIL", "log tail matches a fatal error pattern")
    if tail.count("vLLM died") or tail.count("CUDA out of memory"):
        check.add("FAIL", "vLLM or CUDA failure in the log")
    if slurm["state"] == "RUNNING":
        time.sleep(2)
        size_after = path.stat().st_size
        age = time.time() - mtime
        if size_after > size_before:
            check.notes.append(f"log grew {size_before} -> {size_after}")
        elif age < 180:
            check.notes.append(f"log quiet for {int(age)}s; recent mtime")
        else:
            check.add("WARN", f"running job log has not grown and is {int(age)}s old")


def _argv_text(run: Path) -> str:
    path = run / "INTENDED_ARGV.txt"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _check_config(run: Path, job_id: str, check: Check) -> None:
    if f"job_{job_id}" not in run.name:
        check.add("FAIL", f"output dir {run} is not this job id")
    if "pics_aamas_v0_target_only" not in str(run) and "pics_aamas_v0_" not in str(run):
        check.add("FAIL", f"unexpected output root {run.parent.name}")
    policy = _load_json(run / "TRIAL_PROMPT_POLICY.json")
    if not policy:
        check.add("FAIL", "TRIAL_PROMPT_POLICY.json missing or unreadable")
    else:
        if policy.get("packing_implementation") != PACK:
            check.add("FAIL", f"packing stamp is {policy.get('packing_implementation')!r}")
        else:
            check.notes.append(f"stamp {PACK}")
        if policy.get("trial_prompt_policy") != POLICY:
            check.add("FAIL", f"policy is {policy.get('trial_prompt_policy')!r}")
    argv = _argv_text(run)
    if argv:
        if f"job_{job_id}" not in argv:
            check.add("FAIL", "argv output dir is not this job")
        if "--hard_prompt_token_cap 14000" not in argv:
            check.add("FAIL", "argv input cap is not 14000")
        if "--evolution_selection_score train_val" not in argv:
            check.add("FAIL", "selection score is not train_val")
        if "--split_seed 0" not in argv or "--pics_run_seed 0" not in argv:
            check.add("WARN", "seeds in argv are not the standard 0/0 pair")
    prov_path = run / "target_population" / "control" / "POPULATION_PROVENANCE.json"
    prov = _load_json(prov_path)
    if prov:
        if int(prov.get("hard_prompt_token_cap") or 0) != CAP:
            check.add("FAIL", f"provenance cap {prov.get('hard_prompt_token_cap')}")
        if int(prov.get("vllm_max_model_len") or 0) != CONTEXT:
            check.add("FAIL", f"provenance context {prov.get('vllm_max_model_len')}")
        if prov.get("test_used_for_selection") is True:
            check.add("FAIL", "provenance says test was used for selection")
        if prov.get("evolution_selection_score") not in (None, "train_val"):
            check.add("FAIL", f"provenance selection {prov.get('evolution_selection_score')}")
    elif (run / "target_population").is_dir():
        check.add("PENDING", "population provenance not written yet")


def _phase_status(run: Path) -> Dict[str, str]:
    """reached | pending | absent for the three LLM phases."""
    status = {"target_only_population": "absent", "participant_exploration": "absent", "participant_evolution": "absent"}
    global_dir = run / "target_population" / "control" / "global_phase"
    if (global_dir / "iteration_2").is_dir() or (run / "target_population" / "control" / "STAGE_COMPLETE.json").is_file():
        status["target_only_population"] = "reached"
    elif (global_dir / "iteration_1").is_dir():
        status["target_only_population"] = "pending"
    people = _people(run)
    if not people:
        return status
    rep = min(people)
    explore = run / "selected" / f"participant_{rep}" / "explore_phase"
    if (explore / "metrics.json").is_file():
        status["participant_exploration"] = "reached"
    elif (explore / "prompt_selection_map.json").is_file():
        payload = _load_json(explore / "prompt_selection_map.json") or {}
        steps = [int(row.get("selection_step") or -1) for row in payload.get("candidates") or []]
        status["participant_exploration"] = "reached" if steps and max(steps) >= 25 else "pending"
    evo = run / "selected" / f"participant_{rep}"
    if (evo / "iteration_2").is_dir():
        status["participant_evolution"] = "reached"
    elif (evo / "iteration_1").is_dir():
        status["participant_evolution"] = "pending"
    return status


def _people(run: Path) -> List[int]:
    selected = run / "selected"
    if not selected.is_dir():
        return []
    ids = []
    for path in selected.glob("participant_*"):
        suffix = path.name.split("_", 1)[-1]
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def _check_examples(run: Path, check: Check) -> None:
    folder = run / "prompt_examples"
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    by_arm = {path.stem: path for path in files}
    status = _phase_status(run)
    dataset = run.parents[1].name if len(run.parents) >= 2 else ""
    for arm, state in status.items():
        path = by_arm.get(arm)
        if state == "absent":
            if path:
                check.add("WARN", f"example {arm} exists before that phase has output")
            continue
        if state == "pending":
            if path:
                check.notes.append(f"{arm} example already saved")
                _check_one_example(path, dataset, check)
            else:
                check.add("PENDING", f"{arm} representative call not reached")
            continue
        if path is None:
            check.add("FAIL", f"missing prompt example for reached phase {arm}")
            continue
        _check_one_example(path, dataset, check)
        if arm == "participant_exploration":
            _check_explore_parent(run, path, check)
    extra = [stem for stem in by_arm if stem not in status]
    if extra:
        check.add("WARN", f"unexpected example files {extra}")
    if len(files) != len(set(by_arm)):
        check.add("FAIL", "duplicate prompt example names")


def _check_one_example(path: Path, dataset: str, check: Check) -> None:
    payload = _load_json(path)
    if not payload:
        check.add("FAIL", f"{path.name} is not valid JSON")
        return
    messages = payload.get("api_messages") or []
    if not messages or not isinstance(messages[0], dict) or "content" not in messages[0]:
        check.add("FAIL", f"{path.name} has no API message content")
        return
    content = str(messages[0].get("content") or "")
    chat = payload.get("qwen_chat_messages") or []
    user_parts = [str(item.get("content") or "") for item in chat if item.get("role") == "user"]
    if user_parts and user_parts[0] != content:
        check.add("FAIL", f"{path.name} chat user text differs from api_messages")
    tokens = payload.get("qwen_input_tokens")
    if not isinstance(tokens, int) or tokens > CAP:
        check.add("FAIL", f"{path.name} qwen tokens {tokens}")
    if payload.get("dataset") != dataset:
        check.add("FAIL", f"{path.name} dataset {payload.get('dataset')} != {dataset}")
    if payload.get("arm") != path.stem:
        check.add("FAIL", f"{path.name} arm field {payload.get('arm')}")
    parents = payload.get("parent_ids") or []
    programs = payload.get("parent_programs") or []
    if not parents or not programs or len(parents) != len(programs):
        check.add("FAIL", f"{path.name} parent ids and programs do not match")
    elif any("def choose" not in str(program) for program in programs):
        check.add("FAIL", f"{path.name} parent program is missing def choose")
    trials = payload.get("packed_trial_ids") or []
    if not trials:
        check.add("FAIL", f"{path.name} has no packed trials")
    else:
        missing = 0
        for trial in trials[:8]:
            block = str(trial.get("block_id"))
            if block not in content:
                missing += 1
        if missing:
            check.add("FAIL", f"{path.name} packed block ids are absent from the message")
    if "history_prefix" in content:
        check.add("FAIL", f"{path.name} contains history_prefix")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    check.notes.append(
        f"{path.stem} tokens={tokens} parents={parents[:2]} trials={len(trials)} hash={digest}"
    )


def _check_explore_parent(run: Path, path: Path, check: Check) -> None:
    payload = _load_json(path) or {}
    pid = payload.get("participant_id")
    programs = payload.get("parent_programs") or []
    if pid is None or not programs:
        return
    pool = _load_json(run / "selected" / f"participant_{pid}" / "initial_pool_from_global" / "pool_manifest.json")
    if not pool or not pool.get("programs"):
        check.add("PENDING", "explore parent pool is not saved yet")
        return
    rank = pool["programs"][0]
    filename = run / "selected" / f"participant_{pid}" / "initial_pool_from_global" / str(rank.get("filename"))
    if not filename.is_file():
        check.add("WARN", "rank-1 population program file is missing")
        return
    saved = filename.read_text(encoding="utf-8").strip()
    if saved != str(programs[0]).strip():
        check.add("FAIL", f"explore example parent is not {rank.get('program_id')}")
    else:
        check.notes.append(f"explore parent matches {rank.get('program_id')}")


def _trial_ids(selection: Dict[str, Any]) -> List[str]:
    trials = selection.get("trials") or []
    ids = []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        block = trial.get("block_id")
        pos = trial.get("trial_position")
        tid = trial.get("trial_id")
        if block is None or pos is None or tid is None:
            ids.append("")
        else:
            ids.append(str(tid))
    return ids


def _check_selection_row(label: str, row: Dict[str, Any], check: Check, native_pid: Optional[int]) -> None:
    selection = row.get("aamas_selection") if "aamas_selection" in row else row
    if not isinstance(selection, dict):
        check.add("WARN", f"{label} has no selection record")
        return
    if selection.get("packing_implementation") not in (None, PACK):
        check.add("FAIL", f"{label} packing {selection.get('packing_implementation')}")
    elif selection.get("packing_implementation") != PACK and "aamas_selection" in row:
        check.add("FAIL", f"{label} selection is missing the packing stamp")
    tokens = selection.get("prompt_tokens", row.get("final_chat_template_tokens"))
    if isinstance(tokens, int) and tokens > CAP:
        check.add("FAIL", f"{label} tokens {tokens} over cap")
    retained = row.get("train_trials_before")
    n_sup = selection.get("n_supervised")
    fits = bool(selection.get("full_retained_fits"))
    exceed = bool(selection.get("next_trial_would_exceed"))
    if fits and exceed:
        check.add("FAIL", f"{label} is marked both full-fit and next-trial overflow")
    if isinstance(n_sup, int) and isinstance(retained, int) and n_sup < retained and not fits and not exceed:
        check.add("FAIL", f"{label} stopped short with unseen trials still eligible")
    ids = _trial_ids(selection)
    if any(not item for item in ids):
        check.add("FAIL", f"{label} trial entry is missing block_id, position, or trial_id")
    if len(ids) != len(set(ids)):
        check.add("FAIL", f"{label} repeats a trial inside one prompt")
    if native_pid not in (None, 0):
        blocks = [str(trial.get("block_id")) for trial in selection.get("trials") or [] if isinstance(trial, dict)]
        native = [PID_IN_BLOCK.match(block) for block in blocks[:12]]
        parsed = [int(match.group(1)) for match in native if match]
        if parsed and all(pid == 0 for pid in parsed):
            check.add("FAIL", f"{label} block ids use participant 0 for native {native_pid}")


def _diagnostic_files(run: Path) -> List[Tuple[str, Path, Optional[int]]]:
    files: List[Tuple[str, Path, Optional[int]]] = []
    pop = run / "target_population" / "control" / "prompt_diagnostics.jsonl"
    if pop.is_file():
        files.append(("population", pop, None))
    people = _people(run)
    for pid in people[:3]:
        path = run / "selected" / f"participant_{pid}" / "prompt_diagnostics.jsonl"
        if path.is_file():
            files.append((f"participant_{pid}", path, pid))
    return files


def _check_packing(run: Path, check: Check) -> None:
    files = _diagnostic_files(run)
    if not files:
        check.add("PENDING", "no selection diagnostics yet")
        return
    for label, path, pid in files:
        rows, partial = _jsonl(path)
        if partial:
            check.notes.append(f"{label} ignored {partial} partial JSONL line")
        for name, row in _sample(rows):
            _check_selection_row(f"{label}/{name}", row, check, pid)
        check.notes.append(f"{label} sampled {len(_sample(rows))} of {len(rows)} rows")


def _score_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("record_type") == "candidate"]


def _summarize_scores(label: str, rows: Sequence[Dict[str, Any]], check: Check) -> None:
    if not rows:
        check.add("PENDING", f"{label} has no scored candidates yet")
        return
    ids = [str(row.get("candidate_id")) for row in rows]
    if len(ids) != len(set(ids)):
        check.add("FAIL", f"{label} has duplicate candidate ids")
    valid = [row for row in rows if row.get("runtime_valid") is True]
    invalid = len(rows) - len(valid)
    scores = [_finite(row.get("selection_score")) for row in valid]
    scores = [score for score in scores if score is not None]
    if len(rows) >= 10 and len(valid) / len(rows) < 0.5:
        check.add("FAIL", f"{label} valid {len(valid)}/{len(rows)}")
    elif invalid:
        check.add("WARN", f"{label} invalid {invalid}/{len(rows)}")
    if not scores:
        check.notes.append(f"{label} n={len(rows)} valid={len(valid)} no finite scores")
        return
    unique = len({round(score, 6) for score in scores})
    if len(scores) >= 8 and unique == 1:
        check.add("WARN", f"{label} selection scores are identical")
    best = max(scores)
    rank_row = max(valid, key=lambda row: _finite(row.get("selection_score")) or -1e100)
    parent = _finite(rank_row.get("reference_parent_score"))
    check.notes.append(
        f"{label} n={len(rows)} valid={len(valid)} "
        f"min={_fmt(min(scores))} med={_fmt(statistics.median(scores))} max={_fmt(best)} "
        f"rank1={rank_row.get('candidate_id')} parent={_fmt(parent)}"
    )
    if parent is not None and best < parent - 1e-9 and len(scores) >= 10:
        check.add("WARN", f"{label} rank-1 {_fmt(best)} is below parent {_fmt(parent)}")
    tests = [_finite(row.get("test_loglik")) for row in rows]
    if any(score is not None and score == _finite(row.get("selection_score")) for row, score in zip(rows, tests) if score is not None):
        check.add("FAIL", f"{label} selection score matches a test score")


def _check_scores(run: Path, check: Check) -> None:
    pop_mem = run / "target_population" / "control" / "global_phase" / "mem_trace.jsonl"
    rows, partial = _jsonl(pop_mem)
    if partial:
        check.notes.append("population mem ignored a partial final line")
    if rows:
        _summarize_scores("population", _score_rows(rows), check)
        iterations = sorted({int(row["iteration"]) for row in rows if row.get("record_type") == "iteration_context" and row.get("iteration") is not None})
        if iterations and iterations != list(range(iterations[0], iterations[-1] + 1)):
            check.add("FAIL", f"population iterations have a gap {iterations}")
    people = _people(run)
    for pid in people:
        person = run / "selected" / f"participant_{pid}"
        mem_rows, mem_partial = _jsonl(person / "mem_trace.jsonl")
        if mem_partial:
            check.notes.append(f"participant {pid} mem ignored a partial line")
        phase_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in _score_rows(mem_rows):
            phase_rows.setdefault(str(row.get("phase")), []).append(row)
        explore_metrics = _load_json(person / "explore_phase" / "metrics.json")
        if explore_metrics:
            test = explore_metrics.get("diagnostic_test") or {}
            if test and test.get("not_used_for_selection") is not True:
                check.add("FAIL", f"participant {pid} explore test is not marked passive")
            elif test.get("test_loglik") is not None:
                check.notes.append(f"participant {pid} passive explore test {_fmt(_finite(test.get('test_loglik')))}")
        if not phase_rows and not explore_metrics:
            continue
        for phase, group in phase_rows.items():
            _summarize_scores(f"p{pid}/{phase}", group, check)


def inspect(job_id: str) -> int:
    check = Check()
    slurm = _slurm(job_id)
    print(f"# job {job_id}")
    _check_log(job_id, check, slurm)
    runs = _locate(job_id)
    if not runs:
        if slurm["state"] in {"PENDING", "PD", "CONFIGURING"}:
            check.add("PENDING", "no output directory yet")
        else:
            check.add("FAIL", "no output directory for this job id")
    else:
        for run in runs:
            print(f"output {run}")
            _check_config(run, job_id, check)
            _check_examples(run, check)
            _check_packing(run, check)
            _check_scores(run, check)
    print(f"verdict {check.level}")
    for note in check.notes:
        print(f"- {note}")
    print()
    return {"PASS": 0, "PENDING": 0, "WARN": 0, "FAIL": 1}[check.level]


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only AAMAS v0 mid-run sentinel")
    parser.add_argument("--job_id", action="append", required=True, help="Slurm job id")
    args = parser.parse_args()
    code = 0
    for job_id in args.job_id:
        code = max(code, inspect(str(job_id)))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
