"""Build a Stage E/G cross-task prompt suffix from a live source rank-1 program."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from data_modules.mixed_gambles import DEFAULT_CSV_PATH
from utils.teh.prompt_sanitize import _marker_present
from utils.teh_transfer.prompts import (
    build_transfer_source_suffix,
    make_source_context,
)


def _load_source_example_trials(
    source_dataset: str,
    *,
    split_seed: int,
    psych_dataset_split: str,
    split_ratio: float = 0.6,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    max_observed_trials_per_participant: Optional[int] = None,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    require: bool = False,
) -> List[dict]:
    def _load() -> List[dict]:
        from baseline_methods.MLE import trials_for_participant
        from utils.teh.participant_ids import load_valid_participant_ids
        from utils.teh.teh_datasets import emnlp_ordinal_range

        repo = Path(__file__).resolve().parents[2]
        pids = load_valid_participant_ids(
            source_dataset,
            repo,
            filter_mixed_gambles=bool(filter_mixed_gambles),
            split_ratio=float(split_ratio),
            split_seed=int(split_seed),
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv or DEFAULT_CSV_PATH,
            auto_prepare=True,
        )
        if not pids:
            raise ValueError(f"No valid participants for source dataset {source_dataset!r}")
        start, _end = emnlp_ordinal_range(source_dataset)
        if start >= len(pids):
            raise ValueError(
                f"Source {source_dataset!r} EMNLP start ordinal {start} is out of range "
                f"for {len(pids)} valid participants"
            )
        # First person in the source G.1 EMNLP slice, not valid_participant_ids[0].
        pid = int(pids[start])
        train, val, _test = trials_for_participant(
            source_dataset,
            pid,
            split_ratio=float(split_ratio),
            split_seed=int(split_seed),
            filter_mixed_gambles=bool(filter_mixed_gambles),
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv or DEFAULT_CSV_PATH,
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
        )
        del _test
        trials: List[dict] = list(train or []) + list(val or [])
        if not trials:
            raise ValueError(
                f"Source {source_dataset!r} participant {pid} has no train+val examples "
                "(test is never used)."
            )
        return trials[:8]

    if require:
        return _load()
    try:
        return _load()
    except Exception:
        return []


def _source_program_has_choose(code: str) -> bool:
    """Same bar as TEH candidate validity: a callable ``choose``, not a byte match on ``def choose(``.

    LLM best_program.py often has ``def choose (problem ,history ):`` (spaces).
    """
    if not _marker_present(code, "def choose("):
        return False
    ns: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, ns)
    except Exception:
        return False
    return callable(ns.get("choose"))


def build_rank1_explore_prompt_suffix(
    *,
    source_dataset: str,
    program_path: str,
    split_seed: int = 0,
    psych_dataset_split: str = "train",
    best_loglik: Optional[float] = None,
    split_ratio: float = 0.6,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    max_observed_trials_per_participant: Optional[int] = None,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    require_source_examples: bool = False,
) -> str:
    """Cross-task block: source rank-1 code plus obs (train+val) example trials.

    Example trials must use the same limited-data protocol as the source
    population (SA40 for T-PICS). Test is never loaded into the suffix.
    """
    path = Path(program_path).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Explore source program not found: {path}")
    code = path.read_text(encoding="utf-8")
    if not _source_program_has_choose(code):
        raise ValueError(f"Explore source program has no callable choose(): {path}")
    loglik = float(best_loglik) if best_loglik is not None else 0.0
    example_trials: List[Any] = _load_source_example_trials(
        source_dataset,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        split_ratio=split_ratio,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
        require=bool(require_source_examples),
    )
    if require_source_examples and not example_trials:
        raise ValueError(
            f"Required source train+val examples missing for {source_dataset!r} "
            f"(source test is never used)."
        )
    ctx = make_source_context(
        dataset_alias=source_dataset,
        example_trials=example_trials,
        best_program_code=code,
        best_loglik=loglik,
        example_seed=int(split_seed),
    )
    return build_transfer_source_suffix([ctx])
