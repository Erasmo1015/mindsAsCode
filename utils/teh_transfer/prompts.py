"""Transfer-phase prompt construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    format_trials_for_prompt,
    normalize_psych101_dataset_alias,
)
from utils.teh.teh_datasets import dataset_display_name, is_mixed_gambles_dataset


@dataclass
class SourceTransferContext:
    """Population-level source program and metadata for transfer prompts."""

    dataset_alias: str
    display_name: str
    task_description: str
    example_trial_text: str
    best_program_code: str
    best_loglik: float


def task_description_for_dataset(dataset_alias: str, *, instruction: str = "") -> str:
    """High-level task description for a dataset (Psych-101 metadata or mixed gambles)."""
    if is_mixed_gambles_dataset(dataset_alias):
        return instruction.strip() or (
            "Mixed gambles: Option A is a 50/50 gamble (gain/loss); Option B is certain. "
            "action=0 gamble, action=1 certain; choose(problem, history) returns P(action=1)."
        )
    alias = normalize_psych101_dataset_alias(dataset_alias)
    return PSYCH101_BINARY_DATASETS[alias]["task_description"]


def one_example_trial_text(
    trials: List[Dict[str, Any]],
    *,
    seed: int,
) -> str:
    """Serialize one randomly chosen trial for prompt injection."""
    if not trials:
        return "(no trials available)"
    rng = np.random.default_rng(int(seed))
    trial = trials[int(rng.integers(len(trials)))]
    return format_trials_for_prompt([trial], max_trials=1)


def build_transfer_source_suffix(sources: List[SourceTransferContext]) -> str:
    """Prompt section listing each source dataset, example trial, and best program."""
    if not sources:
        return ""
    lines = [
        "## Cross-task transfer context",
        "",
        "The following source datasets have population-level cognitive programs evolved "
        "on pooled train+validation trials across all participants. Adapt useful ideas to "
        "the target task described above.",
        "",
    ]
    for idx, src in enumerate(sources, start=1):
        lines.extend(
            [
                f"### Source dataset {idx}: {src.display_name} (`{src.dataset_alias}`)",
                "",
                src.task_description.strip(),
                "",
                "Example trial from source train+validation pool:",
                src.example_trial_text.strip(),
                "",
                f"Best population-level program (selection score / loglik: {src.best_loglik:.6f}):",
                f"```python\n{src.best_program_code.strip()}\n```",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def make_source_context(
    *,
    dataset_alias: str,
    example_trials: List[Dict[str, Any]],
    best_program_code: str,
    best_loglik: float,
    example_seed: int,
    instruction: str = "",
) -> SourceTransferContext:
    """Build one source context block for transfer prompts."""
    return SourceTransferContext(
        dataset_alias=dataset_alias,
        display_name=dataset_display_name(dataset_alias),
        task_description=task_description_for_dataset(dataset_alias, instruction=instruction),
        example_trial_text=one_example_trial_text(
            example_trials, seed=example_seed
        ),
        best_program_code=best_program_code,
        best_loglik=float(best_loglik),
    )


def write_debug_prompts_file(
    path: str,
    *,
    global_prompt: Optional[str],
    transfer_prompt: Optional[str],
) -> None:
    """Write global + transfer full prompts for one dataset."""
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: List[str] = []
    if global_prompt:
        chunks.extend(
            [
                "=" * 80,
                "GLOBAL PHASE — iteration 1 (full prompt)",
                "=" * 80,
                "",
                global_prompt.rstrip(),
                "",
            ]
        )
    if transfer_prompt:
        chunks.extend(
            [
                "=" * 80,
                "TRANSFER PHASE — iteration 1 (full prompt)",
                "=" * 80,
                "",
                transfer_prompt.rstrip(),
                "",
            ]
        )
    out.write_text("\n".join(chunks), encoding="utf-8")
