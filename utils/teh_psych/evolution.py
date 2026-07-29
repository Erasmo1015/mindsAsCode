"""
Population-level evolution loop using categorical log-likelihood evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from teh import (
    BEST_PROGRAM_FILENAME,
    DEFAULT_ERROR_FEEDBACK_MODE,
    MAX_ERROR_MESSAGE_CHARS,
    MAX_INVALID_LINE_CHARS,
    _EARLY_STOP_MIN_IMPROVEMENT,
    _ErrorFeedbackStore,
    _build_past_error_prompt_section,
    _decayed_fresh_n_for_iteration,
    _evolution_selection_score,
    _generate_iteration_candidate_codes,
    _normalize_early_stop_iters,
    _normalize_error_feedback_mode,
    _normalize_error_message_for_grouping,
    _normalize_text,
    _record_invalid_program_error,
    _record_invalid_program_error_summary,
    _safe_float,
    _sanitize_llm_python_candidate,
    _select_parent_indices_from_elite_pool,
    _train_loglik_from_elite_tuple,
    _uses_train_val_evolution_selection,
    _write_iteration_error_prompt_file,
    compile_program,
    compile_program_with_error,
    load_seed_program,
)
from utils.teh_psych.categorical_eval import evaluate_categorical_program

EliteTuple = Tuple[Any, ...]


def _eval_error_grouping_key(error_type: str, error_message: str, invalid_line: str) -> str:
    """Same grouping key format as teh._build_invalid_program_error_entry."""
    norm_type = _normalize_text(error_type, 80)
    norm_msg = _normalize_error_message_for_grouping(error_message, MAX_ERROR_MESSAGE_CHARS)
    norm_line = _normalize_text(invalid_line, MAX_INVALID_LINE_CHARS)
    parts = (norm_type, norm_msg, norm_line) if norm_line else (norm_type, norm_msg)
    return "||".join(parts)


def _coerce_eval_error_entry(error_entry: Any) -> Optional[Dict[str, Any]]:
    if error_entry is None:
        return None
    if isinstance(error_entry, dict):
        return error_entry
    msg = str(error_entry).strip()
    if not msg:
        return None
    return {
        "normalized_key": _eval_error_grouping_key("EvalError", msg, ""),
        "error_message": msg,
        "error_type": "EvalError",
        "invalid_line": "",
    }


def _evaluate_categorical_loglik(
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    *,
    n_seeds: int = 1,
) -> Dict[str, Any]:
    return evaluate_categorical_program(choose_fn, trials, n_seeds=n_seeds)


def run_population_evolution(
    *,
    pooled_train: List[Dict[str, Any]],
    pooled_val: List[Dict[str, Any]],
    seed_program_path: str,
    n_iterations: int,
    n_candidates_per_iteration: int,
    fresh_n_candidates: int = 0,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool = True,
    elite_pool_size: Optional[int],
    model_name: str,
    client: Optional[OpenAI],
    max_prompt_train_trials: int = 1_000_000,
    max_prompt_trials_per_problem: int = 0,
    split_seed: int = 42,
    llm_max_tokens: int = 800,
    max_workers: int = 5,
    n_eval_seeds: int = 3,
    output_dir: Path,
    run_prompts_dir: str,
    early_stop_iters: Optional[int] = None,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    prompt_debug: bool = False,
    prompt_debug_on_no_valid: bool = True,
    prompt_debug_exit: bool = False,
    evolution_selection_score: str = "train_val",
    max_error_prompt_chars: int = 1200,
    error_feedback_mode: str = DEFAULT_ERROR_FEEDBACK_MODE,
    max_parent_chars: int = 6000,
    warn_parent_truncation_ratio: float = 0.5,
    dataset_label: str = "population",
    simple_log: bool = True,
) -> Dict[str, Any]:
    """
    Cross-participant population evolution on pooled categorical train trials.
    """
    error_feedback_mode = _normalize_error_feedback_mode(error_feedback_mode)
    if not pooled_train:
        raise ValueError("pooled_train is empty")
    evolution_selection_score = str(evolution_selection_score).strip().lower()
    use_train_val = _uses_train_val_evolution_selection(evolution_selection_score, "loglik")

    pop_dir = output_dir / "population_phase"
    pop_dir.mkdir(parents=True, exist_ok=True)

    seed_code = load_seed_program(seed_program_path)
    seed_fn = compile_program(seed_code)
    if seed_fn is None:
        raise RuntimeError(f"Failed to compile seed program: {seed_program_path}")

    baseline_eval = _evaluate_categorical_loglik(seed_fn, pooled_train, n_seeds=n_eval_seeds)
    baseline_ll = float(baseline_eval["avg_loglik"])
    baseline_val_ll: Optional[float] = None
    if pooled_val:
        baseline_val_eval = _evaluate_categorical_loglik(seed_fn, pooled_val, n_seeds=n_eval_seeds)
        baseline_val_ll = float(baseline_val_eval["avg_loglik"])
    baseline_fitness = (
        _evolution_selection_score(
            baseline_ll,
            baseline_val_ll,
            len(pooled_train),
            len(pooled_val),
            evolution_selection_score=evolution_selection_score,
            warn_key="population",
        )
        if use_train_val
        else baseline_ll
    )

    elite_parents: List[EliteTuple] = [
        (seed_code, baseline_fitness, None, "population_baseline", None, None, baseline_ll)
    ]
    early_stop_patience = _normalize_early_stop_iters(early_stop_iters)
    last_significant_best = baseline_fitness
    stagnant_iters = 0
    invalid_candidate_errors: List[Dict[str, Any]] = _ErrorFeedbackStore(
        error_feedback_mode
    )
    error_history_path = pop_dir / "error_history.jsonl"

    if elite_pool_size is None:
        pool_cap = max(sample_size * 2, 20)
    else:
        pool_cap = max(1, int(elite_pool_size))

    if n_iterations <= 0:
        best_path = pop_dir / BEST_PROGRAM_FILENAME
        best_path.write_text(seed_code, encoding="utf-8")
        return {
            "best_program_path": str(best_path),
            "best_code": seed_code,
            "baseline_train_loglik": baseline_ll,
            "baseline_val_loglik": baseline_val_ll,
            "pool_best_train_loglik": baseline_ll,
            "n_iterations_run": 0,
        }

    if client is None:
        raise RuntimeError("LLM client required for population evolution")

    for iteration in range(n_iterations):
        iteration_step = iteration + 1
        iter_dir = pop_dir / f"iteration_{iteration_step}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        if not simple_log:
            (iter_dir / "candidates").mkdir(exist_ok=True)

        pool_size = len(elite_parents)
        if sample_parents and pool_size > 0:
            rng = np.random.default_rng(int(split_seed) + 50_000 + iteration_step * 1_000_003)
            parent_idxs, _, _ = _select_parent_indices_from_elite_pool(
                pool_size,
                sample_size=sample_size,
                sample_parents=True,
                sampled_parents_decay=sampled_parents_decay,
                iter_idx=iteration,
                total_iters=n_iterations,
                rng=rng,
            )
            selected_parents = [elite_parents[int(j)] for j in parent_idxs]
        else:
            selected_parents = elite_parents[: min(sample_size, pool_size)]

        parent_codes = [p[0] for p in selected_parents]
        parent_train_lls = [_train_loglik_from_elite_tuple(p) for p in selected_parents]
        error_prompt_section = _build_past_error_prompt_section(
            invalid_candidate_errors,
            iteration=iteration_step,
            max_error_prompt_chars=max_error_prompt_chars,
            previous_n_candidates=n_candidates_per_iteration,
            error_feedback_mode=error_feedback_mode,
        )
        _write_iteration_error_prompt_file(iter_dir, error_prompt_section)
        fresh_n = _decayed_fresh_n_for_iteration(
            fresh_n_candidates, iteration, n_iterations, n_candidates_per_iteration
        )
        variant_kwargs = {
            "train_trials": pooled_train,
            "extra_prompt_trials": pooled_val if pooled_val else None,
            "max_tokens": llm_max_tokens,
            "dataset": dataset_label,
            "max_prompt_train_trials": max_prompt_train_trials,
            "max_prompt_trials_per_problem": max_prompt_trials_per_problem,
            "prompt_train_trials_seed": int(split_seed) + 60_000 + iteration_step,
            "fitness_metric": "loglik",
            "max_workers": max_workers,
            "run_prompts_dir": run_prompts_dir,
            "max_parent_chars": max_parent_chars,
            "warn_parent_truncation_ratio": warn_parent_truncation_ratio,
            "sample_size_for_warning": sample_size,
            "prompt_stats_path": iter_dir / "prompt_stats.json",
            "hard_prompt_token_cap": hard_prompt_token_cap,
            "strict_prompt_budget": strict_prompt_budget,
            "prompt_token_estimator": prompt_token_estimator,
            "prompt_diagnostics_dir": None if simple_log else output_dir,
            "phase": "population_evolution",
            "participant_id": None,
            "iteration": iteration_step,
            "prompt_debug": prompt_debug,
            "prompt_debug_exit": prompt_debug_exit,
            "past_invalid_program_errors": invalid_candidate_errors,
            "past_error_prompt_section": error_prompt_section,
            "max_error_prompt_chars": max_error_prompt_chars,
            "error_feedback_mode": error_feedback_mode,
        }
        candidate_codes, _ = _generate_iteration_candidate_codes(
            client=client,
            model_name=model_name,
            fresh_n_candidates=fresh_n,
            n_candidates=n_candidates_per_iteration,
            fresh_parent_programs=[seed_code],
            normal_parent_programs=parent_codes,
            variant_kwargs=variant_kwargs,
            fresh_parent_train_accuracies=[baseline_ll],
            normal_parent_train_accuracies=parent_train_lls,
        )

        for idx, code in enumerate(candidate_codes):
            if not simple_log:
                (iter_dir / "candidates" / f"candidate_{idx}.py").write_text(
                    code or "", encoding="utf-8"
                )
            code = _sanitize_llm_python_candidate(code, required_markers=("def choose(",))
            if not code:
                continue
            choose_fn, compile_error = compile_program_with_error(code)
            if choose_fn is None:
                _record_invalid_program_error(
                    invalid_candidate_errors,
                    code=code,
                    exc=compile_error,
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    eval_split="compile",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
                continue
            train_eval = _evaluate_categorical_loglik(choose_fn, pooled_train, n_seeds=n_eval_seeds)
            if train_eval.get("errors", 0) != 0:
                _record_invalid_program_error_summary(
                    invalid_candidate_errors,
                    _coerce_eval_error_entry(train_eval.get("first_error")),
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    quality_score=_safe_float(train_eval.get("avg_loglik")),
                    eval_split="train",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
                continue
            train_loglik = float(train_eval["avg_loglik"])
            val_loglik: Optional[float] = None
            if use_train_val and pooled_val:
                val_eval = _evaluate_categorical_loglik(choose_fn, pooled_val, n_seeds=n_eval_seeds)
                if val_eval.get("errors", 0) != 0:
                    _record_invalid_program_error_summary(
                        invalid_candidate_errors,
                        _coerce_eval_error_entry(val_eval.get("first_error")),
                        iteration=iteration_step,
                        participant_id=None,
                        candidate_id=f"candidate_{idx}",
                        quality_score=_safe_float(train_loglik),
                        eval_split="val",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                    continue
                val_loglik = float(val_eval["avg_loglik"])
            fitness = (
                _evolution_selection_score(
                    train_loglik,
                    val_loglik,
                    len(pooled_train),
                    len(pooled_val),
                    evolution_selection_score=evolution_selection_score,
                    warn_key="population",
                )
                if use_train_val
                else train_loglik
            )
            prog_id = f"iter{iteration_step}_cand{idx}"
            elite_parents.append(
                (code, fitness, None, prog_id, None, None, train_loglik)
            )

        elite_parents.sort(key=lambda x: float(x[1]), reverse=True)
        elite_parents = elite_parents[:pool_cap]
        pool_best_ll = _train_loglik_from_elite_tuple(elite_parents[0])
        pool_best_selection = float(elite_parents[0][1])
        (iter_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "iteration": iteration_step,
                    "pool_best_train_loglik": pool_best_ll,
                    "pool_best_selection_score": pool_best_selection,
                    "pool_size": len(elite_parents),
                    "error_feedback_mode": error_feedback_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if early_stop_patience is not None:
            improvement = pool_best_selection - float(last_significant_best)
            if improvement >= _EARLY_STOP_MIN_IMPROVEMENT:
                last_significant_best = pool_best_selection
                stagnant_iters = 0
            else:
                stagnant_iters += 1
                if stagnant_iters >= early_stop_patience:
                    break

    best_code = elite_parents[0][0] or ""
    best_path = pop_dir / BEST_PROGRAM_FILENAME
    best_path.write_text(best_code, encoding="utf-8")

    return {
        "best_program_path": str(best_path),
        "best_code": best_code,
        "baseline_train_loglik": baseline_ll,
        "baseline_val_loglik": baseline_val_ll,
        "pool_best_train_loglik": _train_loglik_from_elite_tuple(elite_parents[0]),
        "n_iterations_run": n_iterations,
    }
