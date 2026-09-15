"""Build a Stage E explore prompt suffix from a live source rank-1 program."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from utils.teh_transfer.prompts import (
    build_transfer_source_suffix,
    make_source_context,
)


def _load_source_example_trials(
    source_dataset: str,
    *,
    split_seed: int,
    psych_dataset_split: str,
) -> List[dict]:
    try:
        from baseline_methods.MLE import trials_for_participant

        train, val, _test = trials_for_participant(
            source_dataset,
            0,
            split_ratio=0.6,
            split_seed=int(split_seed),
            filter_mixed_gambles=False,
            psych_dataset_split=psych_dataset_split,
            local_dataset=None,
            mixed_gambles_csv="",
        )
        trials: List[dict] = list(train or []) + list(val or [])
        return trials[:8]
    except Exception:
        return []


def build_rank1_explore_prompt_suffix(
    *,
    source_dataset: str,
    program_path: str,
    split_seed: int = 0,
    psych_dataset_split: str = "train",
    best_loglik: Optional[float] = None,
) -> str:
    """Cross-task block: source rank-1 code in the participant explore prompt."""
    path = Path(program_path).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Explore source program not found: {path}")
    code = path.read_text(encoding="utf-8")
    if "def choose(" not in code:
        raise ValueError(f"Explore source program has no choose(): {path}")
    loglik = float(best_loglik) if best_loglik is not None else 0.0
    example_trials: List[Any] = _load_source_example_trials(
        source_dataset,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
    )
    ctx = make_source_context(
        dataset_alias=source_dataset,
        example_trials=example_trials,
        best_program_code=code,
        best_loglik=loglik,
        example_seed=int(split_seed),
    )
    return build_transfer_source_suffix([ctx])
