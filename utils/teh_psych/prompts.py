"""Prompt setup for categorical teh_psych runs (no LLM parse-plan)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    format_trials_for_prompt,
    summarize_runtime_schema_for_prompt,
)
from utils.teh.teh_runtime import REPO_ROOT, resolve_base_loglik_prompt_path
from utils.teh.prompt_sanitize import strip_embedded_choose_from_evolution_prompt

DEFAULT_CATEGORICAL_PROMPT = REPO_ROOT / "prompts" / "teh_psych" / "infer_single_choice.txt"
DEFAULT_PARSE_PLAN_PROMPT = REPO_ROOT / "prompts" / "teh_psych" / "utils" / "parse_plan.txt"


def format_categorical_trials_for_prompt(trials: List[Dict[str, Any]], max_trials: int = 8) -> str:
    """Summarize categorical trials for prompt injection."""
    lines: List[str] = []
    for i, t in enumerate(trials[:max_trials]):
        p = t.get("problem") or {}
        options = p.get("options") or []
        opt_summary = [
            {k: v for k, v in o.items() if k in ("action", "label", "gamble")}
            for o in options
            if isinstance(o, dict)
        ]
        lines.append(
            f"{i + 1}. options={opt_summary}; "
            f"stimulus_keys={sorted((p.get('stimulus') or {}).keys())}; "
            f"context_keys={sorted((p.get('context') or {}).keys())}; "
            f"target_action={t.get('target_action', t.get('action'))}; "
            f"history_len={len(t.get('history') or [])}"
        )
    return "\n".join(lines)


def merge_categorical_prompt_fallback(
    experiment_id: str,
    *,
    alias: Optional[str],
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    base_prompt_path: Optional[Path | str] = None,
) -> str:
    resolved = resolve_base_loglik_prompt_path(base_prompt_path or DEFAULT_CATEGORICAL_PROMPT)
    base = strip_embedded_choose_from_evolution_prompt(resolved.read_text(encoding="utf-8"))
    schema_summary = summarize_runtime_schema_for_prompt(sample_trials)
    if alias and alias in PSYCH101_BINARY_DATASETS:
        spec = PSYCH101_BINARY_DATASETS[alias]
        display = spec["display_name"]
        task_desc = spec["task_description"]
    else:
        display = experiment_id
        task_desc = instruction[:500] if instruction else "(no task description)"
    trial_text = format_categorical_trials_for_prompt(sample_trials)
    if not trial_text:
        trial_text = format_trials_for_prompt(sample_trials)
    extra = (
        f"\n\n## Experiment: {display} (`{experiment_id}`)\n\n"
        f"{task_desc}\n\n"
        f"### Runtime schema summary (from parsed categorical trials)\n\n{schema_summary}\n\n"
        f"### Task instructions (from Psych-101 transcript)\n\n{instruction[:1500]}\n\n"
        f"### Example categorical trials\n\n{trial_text}\n"
    )
    return base + extra


def setup_experiment_prompts(
    prompts_dir: Path,
    *,
    experiment_id: str,
    alias: Optional[str],
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    seed_program_path: Path,
    base_prompt_path: Path,
    parse_plan_prompt_path: Path,
) -> Path:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    infer_text = merge_categorical_prompt_fallback(
        experiment_id,
        alias=alias,
        instruction=instruction,
        sample_trials=sample_trials,
        base_prompt_path=base_prompt_path,
    )
    (prompts_dir / "infer_single_choice.txt").write_text(infer_text, encoding="utf-8")
    if parse_plan_prompt_path.is_file():
        shutil.copy2(parse_plan_prompt_path, prompts_dir / "parse_plan_prompt.txt")
    seed_src = seed_program_path.expanduser().resolve()
    if seed_src.is_file():
        shutil.copy2(seed_src, prompts_dir / "seed_program.py")
    meta: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "dataset_alias": alias,
        "llm_generated": False,
        "categorical_api": True,
        "base_prompt_path": str(base_prompt_path),
        "parse_plan_prompt_path": str(parse_plan_prompt_path),
        "seed_program_source": str(seed_src),
    }
    (prompts_dir / "prompt_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return prompts_dir
