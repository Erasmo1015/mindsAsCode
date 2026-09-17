"""Rescore initial_pool_from_global programs on a participant's train/val split.

Mirrors TEH handoff scoring (`_global_elite_to_participant_elite` +
`_evolution_selection_score`) without importing teh.py (heavy deps).

Never uses pooled/global loglik from the global-phase manifest as the
participant selection score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_CHOICE13K_LOGLIK_CLAMP_EPS = 1e-9
_CACHE_VERSION = 1


def prefer_hf_datasets_on_sys_path() -> None:
    """Ensure huggingface `datasets` wins over the repo `datasets/` data directory."""
    import site
    import sys

    for sp in [site.getusersitepackages(), *site.getsitepackages()]:
        if sp and sp not in sys.path:
            sys.path.insert(0, str(sp))


def compute_evolution_selection_score(
    train_loglik: float,
    val_loglik: Optional[float],
    n_train: int,
    n_val: int,
    *,
    mode: str = "train_val",
) -> float:
    """Same formula as teh._evolution_selection_score."""
    train_ll = float(train_loglik)
    normalized = str(mode).strip()
    if normalized == "train":
        return train_ll
    if val_loglik is None or int(n_val) <= 0:
        return train_ll
    try:
        val_ll = float(val_loglik)
    except (TypeError, ValueError):
        return train_ll
    if not math.isfinite(val_ll):
        return train_ll
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    denom = n_tr + n_vl
    if denom <= 0:
        return train_ll
    return (n_tr * train_ll + n_vl * val_ll) / denom


def compile_choose(code_str: str) -> Optional[Callable]:
    """Minimal TEH-compatible compile of choose(problem, history)."""
    from utils.teh.sandbox_builtins import compile_choose_with_error

    choose_fn, _err = compile_choose_with_error(code_str)
    return choose_fn


def _parse_choose_output(p_raw: Any) -> float:
    if isinstance(p_raw, bool) or (
        isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)
    ):
        return 1.0 if int(p_raw) == 1 else 0.0
    if isinstance(p_raw, float):
        if not (0.0 <= p_raw <= 1.0):
            raise ValueError(f"invalid probability: {p_raw!r}")
        return p_raw
    raise TypeError(f"choose must return float or 0/1, got {type(p_raw)}")


def _clamp_prob(p: float) -> float:
    return min(max(p, _CHOICE13K_LOGLIK_CLAMP_EPS), 1.0 - _CHOICE13K_LOGLIK_CLAMP_EPS)


def evaluate_binary_loglik(
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    *,
    n_seeds: int = 3,
) -> Dict[str, float]:
    """Choice13k-style Bernoulli loglik (matches teh.evaluate_choice13k_program)."""
    total = len(trials)
    seed_avg_accs: List[float] = []
    seed_avg_logliks: List[float] = []

    def _one_pass() -> Tuple[float, float]:
        loglik_acc = 0.0
        correct = 0
        for t in trials:
            y = int(t["action"])
            try:
                p_raw = choose_fn(t["problem"], t["history"])
                p_use = _parse_choose_output(p_raw)
                p = _clamp_prob(p_use)
            except Exception:
                p = _clamp_prob(0.5)
                p_raw = 0.5
            loglik_acc += y * np.log(p) + (1 - y) * np.log(1.0 - p)
            if isinstance(p_raw, float):
                pred = 1 if p_raw >= 0.5 else 0
            else:
                pred = 1 if int(p_raw) == 1 else 0
            correct += int(pred == y)
        avg_ll = loglik_acc / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_ll, acc

    for _ in range(max(1, int(n_seeds))):
        avg_ll, acc = _one_pass()
        seed_avg_logliks.append(avg_ll)
        seed_avg_accs.append(acc)

    return {
        "avg_loglik": float(np.mean(seed_avg_logliks)) if seed_avg_logliks else float("-inf"),
        "accuracy": float(np.mean(seed_avg_accs)) if seed_avg_accs else 0.0,
    }


def parse_split_hparams_from_command(command_txt: Path) -> Dict[str, Any]:
    """Parse TEH CLI flags used for participant splits / eval."""
    defaults: Dict[str, Any] = {
        "split_ratio": 0.6,
        "split_seed": 0,
        "n_eval_seeds": 3,
        "psych_dataset_split": "train",
        "evolution_selection_score": "train_val",
        "fitness_metric": "loglik",
    }
    if not command_txt.is_file():
        return defaults
    text = command_txt.read_text(encoding="utf-8", errors="ignore")

    def _float(flag: str, default: float) -> float:
        m = re.search(rf"--{flag}(?:\s+|=)([0-9.]+)", text)
        return float(m.group(1)) if m else default

    def _int(flag: str, default: int) -> int:
        m = re.search(rf"--{flag}(?:\s+|=)(-?\d+)", text)
        return int(m.group(1)) if m else default

    def _str(flag: str, default: str) -> str:
        m = re.search(rf"--{flag}(?:\s+|=)([^\s]+)", text)
        return m.group(1) if m else default

    return {
        "split_ratio": _float("split_ratio", defaults["split_ratio"]),
        "split_seed": _int("split_seed", defaults["split_seed"]),
        "n_eval_seeds": _int("n_eval_seeds", defaults["n_eval_seeds"]),
        "psych_dataset_split": _str("psych_dataset_split", defaults["psych_dataset_split"]),
        "evolution_selection_score": _str(
            "evolution_selection_score", defaults["evolution_selection_score"]
        ),
        "fitness_metric": _str("fitness_metric", defaults["fitness_metric"]),
    }


def _code_sha1(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


def default_rescore_cache_path(participant_dir: Path, cache_dir: Optional[Path] = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir) / participant_dir.name / "initial_pool_rescore_cache.json"
    return participant_dir / "initial_pool_from_global" / "participant_selection_scores.json"


def load_participant_train_val(
    *,
    dataset: str,
    participant_id: int,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str = "train",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reload the historical train/val split for one participant (no test)."""
    prefer_hf_datasets_on_sys_path()
    ds = str(dataset).strip()
    # Avoid importing utils.teh package __init__ (pulls openai).
    if ds.lower() in {"mixed_gambles", "mixed-gambles"}:
        from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials

        train_trials, val_trials, _test, _ = load_mixed_gambles_trials(
            int(participant_id),
            csv_path=DEFAULT_CSV_PATH,
            filter_gain_loss_only=False,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        return train_trials, val_trials

    # Import submodule directly to avoid data_modules/__init__.py (choice13k).
    import importlib

    psych101 = importlib.import_module("data_modules.psych101_binary")
    exp = psych101.get_psych101_binary_experiment(
        ds,
        int(participant_id),
        split=psych_dataset_split,
    )
    train_trials, val_trials, _test, _ = psych101.split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    return train_trials, val_trials


def score_program_on_splits(
    code: str,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    *,
    dataset: str = "",
    n_eval_seeds: int = 3,
    evolution_selection_score: str = "train_val",
) -> Dict[str, Any]:
    choose_fn = compile_choose(code)
    if choose_fn is None:
        return {
            "ok": False,
            "error": "compile_failed",
            "selection_score": None,
            "train_loglik": None,
            "val_loglik": None,
        }
    from utils.teh.teh_datasets import is_categorical_output_dataset

    if dataset and is_categorical_output_dataset(dataset):
        from utils.teh_psych.categorical_eval import evaluate_categorical_program

        train_eval = evaluate_categorical_program(
            choose_fn, train_trials, n_seeds=n_eval_seeds
        )
        val_eval = (
            evaluate_categorical_program(choose_fn, val_trials, n_seeds=n_eval_seeds)
            if val_trials
            else None
        )
    else:
        train_eval = evaluate_binary_loglik(choose_fn, train_trials, n_seeds=n_eval_seeds)
        val_eval = (
            evaluate_binary_loglik(choose_fn, val_trials, n_seeds=n_eval_seeds)
            if val_trials
            else None
        )
    train_ll = float(train_eval["avg_loglik"])
    val_ll = float(val_eval["avg_loglik"]) if val_eval is not None else None
    sel = compute_evolution_selection_score(
        train_ll,
        val_ll,
        len(train_trials),
        len(val_trials),
        mode=evolution_selection_score,
    )
    return {
        "ok": True,
        "error": None,
        "selection_score": float(sel),
        "train_loglik": train_ll,
        "val_loglik": val_ll,
        "n_train": len(train_trials),
        "n_val": len(val_trials),
    }


def list_initial_pool_programs(participant_dir: Path) -> List[Dict[str, Any]]:
    """Return [{program_id, filename, code}, ...] from initial_pool_from_global."""
    pool_dir = Path(participant_dir) / "initial_pool_from_global"
    manifest_path = pool_dir / "pool_manifest.json"
    out: List[Dict[str, Any]] = []
    if not pool_dir.is_dir():
        return out
    if manifest_path.is_file():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        for prog in man.get("programs") or []:
            if not isinstance(prog, dict):
                continue
            filename = str(prog.get("filename") or "")
            program_id = str(prog.get("program_id") or "")
            path = pool_dir / filename
            if not path.is_file():
                continue
            # Never treat global_* metrics on the manifest as participant scores.
            out.append(
                {
                    "program_id": program_id,
                    "filename": filename,
                    "code": path.read_text(encoding="utf-8"),
                }
            )
        return out
    for path in sorted(pool_dir.glob("*.py")):
        stem = path.stem
        # e.g. 000_global_baseline -> global_baseline
        pid = re.sub(r"^\d+_", "", stem)
        out.append({"program_id": pid, "filename": path.name, "code": path.read_text(encoding="utf-8")})
    return out


def rescore_initial_pool_for_participant(
    participant_dir: Path,
    *,
    dataset: str,
    split_ratio: float = 0.6,
    split_seed: int = 0,
    n_eval_seeds: int = 3,
    psych_dataset_split: str = "train",
    evolution_selection_score: str = "train_val",
    cache_path: Optional[Path] = None,
    force_rescore: bool = False,
) -> Dict[str, Any]:
    """
    Rescore every initial_pool_from_global program on this participant's train/val.

    Returns a cache document and writes it to cache_path.
    """
    participant_dir = Path(participant_dir)
    m = re.match(r"^participant_(\d+)$", participant_dir.name)
    if not m:
        raise ValueError(f"Not a participant dir: {participant_dir}")
    participant_id = int(m.group(1))
    cache_path = Path(cache_path) if cache_path is not None else default_rescore_cache_path(participant_dir)

    programs = list_initial_pool_programs(participant_dir)
    meta = {
        "cache_version": _CACHE_VERSION,
        "dataset": dataset,
        "participant_id": participant_id,
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
        "n_eval_seeds": int(n_eval_seeds),
        "psych_dataset_split": str(psych_dataset_split),
        "evolution_selection_score": str(evolution_selection_score),
        "scoring": "participant_train_val_recompute",
        "note": "Do not use global_phase pooled scores as participant selection_score.",
    }

    if cache_path.is_file() and not force_rescore:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("cache_version") == _CACHE_VERSION
            and cached.get("dataset") == dataset
            and int(cached.get("participant_id", -1)) == participant_id
            and float(cached.get("split_ratio", -1)) == float(split_ratio)
            and int(cached.get("split_seed", -1)) == int(split_seed)
            and int(cached.get("n_eval_seeds", -1)) == int(n_eval_seeds)
            and cached.get("psych_dataset_split") == psych_dataset_split
            and cached.get("evolution_selection_score") == evolution_selection_score
        ):
            # Ensure every current program_id+code_sha is present.
            by_id = {p["program_id"]: p for p in cached.get("programs") or [] if isinstance(p, dict)}
            complete = True
            for prog in programs:
                row = by_id.get(prog["program_id"])
                if row is None or row.get("code_sha1") != _code_sha1(prog["code"]):
                    complete = False
                    break
            if complete and by_id:
                return cached

    train_trials, val_trials = load_participant_train_val(
        dataset=dataset,
        participant_id=participant_id,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
    )

    scored_rows: List[Dict[str, Any]] = []
    for prog in programs:
        result = score_program_on_splits(
            prog["code"],
            train_trials,
            val_trials,
            dataset=dataset,
            n_eval_seeds=n_eval_seeds,
            evolution_selection_score=evolution_selection_score,
        )
        scored_rows.append(
            {
                "program_id": prog["program_id"],
                "filename": prog["filename"],
                "code_sha1": _code_sha1(prog["code"]),
                **result,
            }
        )

    doc = {**meta, "n_train": len(train_trials), "n_val": len(val_trials), "programs": scored_rows}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc
