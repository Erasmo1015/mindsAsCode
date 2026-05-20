"""
One-shot TEH-style candidate generation for prompt-debug runs (no evolution/eval).
"""
from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from utils.prompt_flags import single_code_template_prompt_suffix
from utils.teh.prompt_sanitize import CANDIDATE_OUTPUT_RULES, sanitize_evolution_candidate_code


def _teh_helpers():
    """Lazy import TEH helpers (same code paths as evolution)."""
    import teh as teh_mod

    return teh_mod

DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_SPLIT_SEED = 42
DEFAULT_PARTICIPANT_ID = 0
DEFAULT_MAX_PROMPT_TRAIN_TRIALS = 5
DEFAULT_CANDIDATE_MAX_TOKENS = 800

_LEAKAGE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("gamble_A", re.compile(r"gamble_A", re.I)),
    ("gamble_B", re.compile(r"gamble_B", re.I)),
    ("expected value", re.compile(r"expected\s+value", re.I)),
    ("utility", re.compile(r"\butility\b", re.I)),
    ("risky/safe wording", re.compile(r"\brisky\b|\bsafe\b", re.I)),
    ("probability weighting", re.compile(r"probability\s+weight", re.I)),
    ("lottery/gamble assumptions", re.compile(r"\blottery\b|two gambles|two-option gamble", re.I)),
)


def load_debug_train_trials(
    dataset_alias: str,
    *,
    participant_id: int = DEFAULT_PARTICIPANT_ID,
    psych_dataset_split: str = "train",
    max_trials: int = DEFAULT_MAX_PROMPT_TRAIN_TRIALS,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> List[Dict[str, Any]]:
    train_trials, _, _ = _teh_helpers()._trials_for_loglik_participant(
        dataset_alias,
        participant_id,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
    )
    return train_trials[:max_trials]


def _dataset_type_for_trials(trials: List[Dict[str, Any]], dataset_alias: str) -> str:
    if trials and "problem" in trials[0]:
        prob0 = trials[0]["problem"]
        return str(prob0.get("dataset_alias") or dataset_alias)
    return dataset_alias


def _build_evolution_prompt(
    infer_prompt: str,
    parent_code: str,
    train_trials: List[Dict[str, Any]],
    dataset_alias: str,
) -> str:
    """Same structure as TEH generate_program_variants (single parent, capped train trials)."""
    dataset_type = _dataset_type_for_trials(train_trials, dataset_alias)
    state_text = _teh_helpers().format_trials_to_text(train_trials, dataset=dataset_type)
    parent_context = (
        f"\n\nReference program (parent):\n```python\n{parent_code}\n```\n\n"
        "Generate a variant that improves upon or explores alternatives to the parent program.\n"
    )
    return (
        f"{infer_prompt.rstrip()}\n{state_text}\n{parent_context}"
        f"{single_code_template_prompt_suffix('')}"
        f"\n{CANDIDATE_OUTPUT_RULES}\n"
    )


def write_leakage_report_file(
    dataset_dir: Path,
    dataset_alias: str,
    trials: List[Dict[str, Any]],
    schema_summary: str,
    *,
    infer_prompt: str = "",
    cleaned_code: str = "",
) -> None:
    """Write generated_candidate/leakage_report.txt from prompt and/or code text."""
    prompts_dir = dataset_dir / "prompts"
    cand_dir = dataset_dir / "generated_candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    if not infer_prompt and (prompts_dir / "infer_single_choice.txt").is_file():
        infer_prompt = (prompts_dir / "infer_single_choice.txt").read_text(encoding="utf-8")
    if not cleaned_code and (cand_dir / "cleaned_candidate.py").is_file():
        cleaned_code = (cand_dir / "cleaned_candidate.py").read_text(encoding="utf-8")
    leak_lines = (
        ["Prompt (infer_single_choice.txt):"]
        + _leakage_report(infer_prompt)
        + ["", "Code (cleaned_candidate.py):"]
        + (_leakage_report(cleaned_code) if cleaned_code.strip() else ["- (no cleaned code)"])
        + ["", "Schema metadata:"]
        + _schema_debug_lines(dataset_alias, trials, schema_summary)
    )
    (cand_dir / "leakage_report.txt").write_text("\n".join(leak_lines) + "\n", encoding="utf-8")


def _leakage_report(text: str) -> List[str]:
    lines = []
    for label, pattern in _LEAKAGE_PATTERNS:
        lines.append(f"- {label}: {'YES' if pattern.search(text) else 'no'}")
    return lines


def _schema_debug_lines(
    dataset_alias: str, trials: List[Dict[str, Any]], schema_summary: str
) -> List[str]:
    from data_modules.psych101_binary import (
        PSYCH101_BINARY_DATASETS,
        normalize_psych101_dataset_alias,
    )
    from utils.teh.teh_datasets import is_mixed_gambles_dataset
    from utils.teh.teh_runtime import _is_gamble_ab_task

    keys: set = set()
    for t in trials:
        for k in (t.get("problem") or {}):
            if k not in ("dataset_alias", "experiment_id"):
                keys.add(k)
    schema_type = "mixed_gambles"
    if not is_mixed_gambles_dataset(dataset_alias):
        spec = PSYCH101_BINARY_DATASETS.get(
            normalize_psych101_dataset_alias(dataset_alias), {}
        )
        schema_type = str(spec.get("schema_type", "?"))
    return [
        f"- dataset_alias: {dataset_alias}",
        f"- schema_type (registry): {schema_type}",
        f"- is_gamble_A/B_task (from trials): {_is_gamble_ab_task(trials)}",
        f"- runtime problem keys: {sorted(keys)}",
        "",
        schema_summary,
    ]


def _validate_candidate(code: str, trial: Optional[Dict[str, Any]]) -> List[str]:
    log: List[str] = []
    if not code:
        log.append("syntax: FAIL (empty after sanitization)")
        log.append("runtime: skipped")
        log.append("smoke_call: skipped")
        return log
    try:
        compile(code, "<candidate>", "exec")
        log.append("syntax: OK")
    except SyntaxError as exc:
        log.append(f"syntax: FAIL ({exc})")
        log.append("runtime: skipped")
        log.append("smoke_call: skipped")
        return log

    choose_fn = _teh_helpers().compile_program(code)
    if choose_fn is None:
        log.append("runtime: FAIL (no callable choose() after compile_program)")
        log.append("smoke_call: skipped")
        return log
    log.append("runtime: OK (choose callable extracted)")

    if trial is None:
        log.append("smoke_call: skipped (no trial)")
        return log
    try:
        out = choose_fn(trial["problem"], trial.get("history", []))
        log.append(f"smoke_call: OK (return type={type(out).__name__}, value={out!r})")
    except Exception as exc:
        log.append(f"smoke_call: FAIL ({exc})")
    return log


def generate_one_debug_candidate(
    client: OpenAI,
    model_name: str,
    dataset_alias: str,
    dataset_dir: Path,
    *,
    participant_id: int = DEFAULT_PARTICIPANT_ID,
    psych_dataset_split: str = "train",
    max_train_trials: int = DEFAULT_MAX_PROMPT_TRAIN_TRIALS,
    max_tokens: int = DEFAULT_CANDIDATE_MAX_TOKENS,
    schema_summary: str = "",
) -> Dict[str, Any]:
    """
    One TEH-style candidate: build evolution prompt from infer_single_choice.txt + seed + trials.
    """
    from utils.teh.teh_runtime import _runtime_schema_summary_for_prompt

    prompts_dir = dataset_dir / "prompts"
    out_dir = dataset_dir / "generated_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_candidate.txt"
    cleaned_path = out_dir / "cleaned_candidate.py"
    exec_path = out_dir / "execution_log.txt"
    raw_path.write_text("(pending)\n", encoding="utf-8")
    cleaned_path.write_text("", encoding="utf-8")
    exec_path.write_text("(pending)\n", encoding="utf-8")

    infer_path = prompts_dir / "infer_single_choice.txt"
    seed_path = prompts_dir / "seed_program.py"
    infer_prompt = infer_path.read_text(encoding="utf-8")
    parent_code = _teh_helpers().load_seed_program(str(seed_path))

    train_trials = load_debug_train_trials(
        dataset_alias,
        participant_id=participant_id,
        psych_dataset_split=psych_dataset_split,
        max_trials=max_train_trials,
    )
    if not schema_summary:
        schema_summary = _runtime_schema_summary_for_prompt(train_trials)
    (prompts_dir / "runtime_schema_summary.txt").write_text(
        schema_summary + "\n", encoding="utf-8"
    )

    evolution_prompt = _build_evolution_prompt(
        infer_prompt, parent_code, train_trials, dataset_alias
    )

    raw = ""
    cleaned = ""
    exec_log: List[str] = []
    error: Optional[str] = None
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": evolution_prompt}],
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        (out_dir / "raw_candidate.txt").write_text(raw + "\n", encoding="utf-8")

        cleaned = sanitize_evolution_candidate_code(raw)
        (out_dir / "cleaned_candidate.py").write_text(
            (cleaned + "\n") if cleaned else "", encoding="utf-8"
        )

        smoke_trial = train_trials[0] if train_trials else None
        exec_log = _validate_candidate(cleaned, smoke_trial)
        if raw and not cleaned:
            exec_log.append("extraction: WARN (sanitizer returned empty string)")
    except Exception:
        error = traceback.format_exc()
        exec_log.append(f"candidate_generation: FAIL\n{error}")
        if not raw:
            raw_path.write_text(f"(candidate generation failed)\n\n{error}\n", encoding="utf-8")
        if not cleaned:
            cleaned_path.write_text("", encoding="utf-8")

    exec_path.write_text("\n".join(exec_log) + "\n", encoding="utf-8")
    write_leakage_report_file(
        dataset_dir,
        dataset_alias,
        train_trials,
        schema_summary,
        infer_prompt=infer_prompt,
        cleaned_code=cleaned,
    )

    return {
        "schema_summary": schema_summary,
        "execution_log": exec_log,
        "error": error,
    }
