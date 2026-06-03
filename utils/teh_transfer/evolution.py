"""Population global and transfer evolution loops (TEH transfer)."""
from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

import teh
from data_modules.mixed_gambles import DEFAULT_CSV_PATH
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    get_psych101_binary_experiment,
)
from utils.teh.teh_runtime import setup_teh_run_prompts

from utils.teh_transfer.config import TransferDatasetSpec
from utils.teh_transfer.prompts import (
    SourceTransferContext,
    build_transfer_source_suffix,
    make_source_context,
    write_debug_prompts_file,
)


@dataclass
class PopulationEvolutionResult:
    dataset_alias: str
    config_key: str
    psych_dataset_split: str
    output_dir: Path
    prompts_dir: Path
    best_loglik: float
    best_program_code: str
    best_program_id: str
    first_iter_best_loglik: Optional[float] = None
    participant_ids: Optional[List[int]] = None
    global_prompt_capture: Optional[str] = None
    transfer_prompt_capture: Optional[str] = None


def transfer_output_run_dir(timestamp: str) -> Path:
    return Path("generated_outputs_transfer") / "teh_transfer" / f"run_{timestamp}"


def _parallel_dataset_pool_sizes(
    max_workers: int, n_candidates: int, parallel_datasets: bool
) -> Tuple[int, int]:
    """Return (dataset_workers, candidate_workers_per_dataset)."""
    n_cand = max(1, int(n_candidates))
    if not parallel_datasets:
        return 1, max(1, int(max_workers))
    return max(1, int(max_workers) // n_cand), n_cand


def collect_pooled_train_val_trials(
    dataset: str,
    participant_ids: List[int],
    *,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pooled train and val trials across participants (same splits as TEH evolution)."""
    train = teh._collect_pooled_train_trials_for_participants(
        dataset,
        participant_ids,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    val = teh._collect_pooled_split_trials_for_participants(
        dataset,
        participant_ids,
        split="val",
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    return train, val


def _rename_global_phase_dir(dataset_output_dir: Path) -> Path:
    """Rename ``global_phase`` -> ``global`` to match transfer run layout."""
    src = dataset_output_dir / "global_phase"
    dst = dataset_output_dir / "global"
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        src.rename(dst)
        return dst
    return dst


def _read_phase_best_loglik(phase_dir: Path) -> Tuple[float, str, str]:
    """Read pool-best selection score from phase ``results.json``."""
    results_path = phase_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing phase results: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    score = payload.get("pool_best_selection_score")
    if score is None:
        score = payload.get("pool_best_global_train_loglik")
    if score is None:
        score = payload.get("pool_best_train_val_loglik")
    if score is None:
        raise KeyError(f"No pool-best loglik in {results_path}")
    program_id = str(payload.get("pool_best_program_id", "unknown"))
    pool_dir = phase_dir / "global_elite_pool"
    code = ""
    manifest = pool_dir / "pool_manifest.json"
    if manifest.is_file():
        programs = json.loads(manifest.read_text(encoding="utf-8")).get("programs", [])
        if programs:
            filename = programs[0]["filename"]
            code = (pool_dir / filename).read_text(encoding="utf-8")
    return float(score), program_id, code


def _first_iteration_best_loglik(phase_dir: Path) -> Optional[float]:
    iter_dir = phase_dir / "iteration_1"
    metrics_path = iter_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("pool_best_selection_score") is not None:
        return float(metrics["pool_best_selection_score"])
    if metrics.get("pool_best_global_train_loglik") is not None:
        return float(metrics["pool_best_global_train_loglik"])
    return None


def run_dataset_global_phase(
    spec: TransferDatasetSpec,
    *,
    run_dir: Path,
    client: OpenAI,
    participants: List[int],
    seed_program_path: Path,
    n_iterations: int,
    n_candidates: int,
    fresh_n_candidates: int,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool,
    elite_pool_size: Optional[int],
    model_name: str,
    split_ratio: float,
    split_seed: int,
    data_path: str,
    filter_mixed_gambles: bool,
    max_prompt_train_trials: int,
    max_prompt_trials_per_problem: int,
    llm_max_tokens: int,
    max_workers: int,
    n_eval_seeds: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    max_parent_chars: int,
    warn_parent_truncation_ratio: float,
    early_stop_iters: Optional[int],
    hard_prompt_token_cap: int,
    strict_prompt_budget: bool,
    prompt_token_estimator: str,
    evolution_selection_score: str,
    max_error_prompt_chars: int,
    use_llm_prompt: bool,
    base_prompt_path: Optional[str],
    debug_prompt: bool = False,
) -> PopulationEvolutionResult:
    """Population-level global evolution for one dataset."""
    dataset_output = run_dir / spec.config_key
    dataset_output.mkdir(parents=True, exist_ok=True)

    prompts_dir = setup_teh_run_prompts(
        dataset_output,
        spec.dataset_alias,
        seed_program_path,
        client=client,
        model_name=model_name,
        use_llm=use_llm_prompt,
        base_prompt_path=base_prompt_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=spec.psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    seed_path = str(prompts_dir / "seed_program.py")

    prompt_capture: Optional[str] = None

    if debug_prompt:
        pooled_train, pooled_val = collect_pooled_train_val_trials(
            spec.dataset_alias,
            participants,
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            psych_dataset_split=spec.psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        seed_code = teh.load_seed_program(seed_path)
        gen_debug_holder: Dict[str, Any] = {}
        teh.generate_program_variants(
            client=client,
            model_name=model_name,
            parent_programs=[seed_code],
            train_trials=pooled_train,
            n_variants=1,
            max_tokens=llm_max_tokens,
            parent_train_accuracies=[0.0],
            dataset=spec.dataset_alias,
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            prompt_train_trials_seed=split_seed + 60_001,
            fitness_metric="loglik",
            max_workers=1,
            extra_prompt_trials=pooled_val if pooled_val else None,
            run_prompts_dir=str(prompts_dir),
            max_parent_chars=max_parent_chars,
            warn_parent_truncation_ratio=warn_parent_truncation_ratio,
            sample_size_for_warning=sample_size,
            hard_prompt_token_cap=hard_prompt_token_cap,
            strict_prompt_budget=strict_prompt_budget,
            prompt_token_estimator=prompt_token_estimator,
            phase="global_evolution",
            iteration=1,
            generation_debug_out=gen_debug_holder,
        )
        prompt_capture = gen_debug_holder.get("prompt_text")

    teh.run_global_evolution_phase(
        dataset=spec.dataset_alias,
        participants=participants,
        seed_program_path=seed_path,
        n_iterations=n_iterations,
        n_candidates_per_iteration=n_candidates,
        fresh_n_candidates=fresh_n_candidates,
        sample_size=sample_size,
        sample_parents=sample_parents,
        sampled_parents_decay=sampled_parents_decay,
        elite_pool_size=elite_pool_size,
        model_name=model_name,
        client=client,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        max_prompt_train_trials=max_prompt_train_trials,
        max_prompt_trials_per_problem=max_prompt_trials_per_problem,
        llm_max_tokens=llm_max_tokens,
        max_workers=max_workers,
        n_eval_seeds=n_eval_seeds,
        output_dir=dataset_output,
        save_artifacts=True,
        wandb_module=None,
        run_prompts_dir=str(prompts_dir),
        psych_dataset_split=spec.psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_parent_chars=max_parent_chars,
        warn_parent_truncation_ratio=warn_parent_truncation_ratio,
        early_stop_iters=early_stop_iters,
        hard_prompt_token_cap=hard_prompt_token_cap,
        strict_prompt_budget=strict_prompt_budget,
        prompt_token_estimator=prompt_token_estimator,
        prompt_debug=False,
        prompt_debug_on_no_valid=False,
        prompt_debug_exit=False,
        evolution_selection_score=evolution_selection_score,
        max_error_prompt_chars=max_error_prompt_chars,
    )

    global_dir = _rename_global_phase_dir(dataset_output)
    best_loglik, program_id, code = _read_phase_best_loglik(global_dir)

    return PopulationEvolutionResult(
        dataset_alias=spec.dataset_alias,
        config_key=spec.config_key,
        psych_dataset_split=spec.psych_dataset_split,
        output_dir=dataset_output,
        prompts_dir=prompts_dir,
        best_loglik=best_loglik,
        best_program_code=code,
        best_program_id=program_id,
        participant_ids=list(participants),
        global_prompt_capture=prompt_capture,
    )


def run_transfer_evolution_phase(
    *,
    target: PopulationEvolutionResult,
    sources: List[SourceTransferContext],
    client: OpenAI,
    seed_program_path: str,
    n_iterations: int,
    n_candidates: int,
    fresh_n_candidates: int,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool,
    elite_pool_size: Optional[int],
    model_name: str,
    split_ratio: float,
    split_seed: int,
    data_path: str,
    filter_mixed_gambles: bool,
    max_prompt_train_trials: int,
    max_prompt_trials_per_problem: int,
    llm_max_tokens: int,
    max_workers: int,
    n_eval_seeds: int,
    max_parent_chars: int,
    warn_parent_truncation_ratio: float,
    early_stop_iters: Optional[int],
    hard_prompt_token_cap: int,
    strict_prompt_budget: bool,
    prompt_token_estimator: str,
    evolution_selection_score: str,
    max_error_prompt_chars: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    debug_prompt: bool = False,
) -> PopulationEvolutionResult:
    """
    Leave-one-dataset-out transfer evolution on pooled target train+val trials.

    Source dataset programs and metadata are injected via ``prompt_suffix``.
    """
    dataset = target.dataset_alias
    participant_ids = target.participant_ids or []
    pooled_train, pooled_val = collect_pooled_train_val_trials(
        dataset,
        participant_ids,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=target.psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    transfer_suffix = build_transfer_source_suffix(sources)
    transfer_dir = target.output_dir / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)

    seed_code = teh.load_seed_program(seed_program_path)
    seed_fn = teh.compile_program(seed_code)
    if seed_fn is None:
        raise RuntimeError(f"Failed to compile seed program: {seed_program_path}")
    baseline_eval = teh._evaluate_loglik_for_dataset(
        dataset, seed_fn, pooled_train, n_seeds=n_eval_seeds
    )
    baseline_ll = float(baseline_eval["avg_loglik"])
    baseline_val_ll: Optional[float] = None
    if pooled_val:
        baseline_val_eval = teh._evaluate_loglik_for_dataset(
            dataset, seed_fn, pooled_val, n_seeds=n_eval_seeds
        )
        baseline_val_ll = float(baseline_val_eval["avg_loglik"])
    use_train_val = teh._uses_train_val_evolution_selection(
        evolution_selection_score, "loglik"
    )
    baseline_fitness = (
        teh._evolution_selection_score(
            baseline_ll,
            baseline_val_ll,
            len(pooled_train),
            len(pooled_val),
            evolution_selection_score=evolution_selection_score,
            warn_key="transfer",
        )
        if use_train_val
        else baseline_ll
    )
    elite_parents: List[Tuple[Any, ...]] = [
        (
            seed_code,
            baseline_fitness,
            None,
            "transfer_baseline",
            None,
            None,
            baseline_ll if use_train_val else baseline_ll,
        )
    ]

    early_stop_patience = teh._normalize_early_stop_iters(early_stop_iters)
    last_significant_best = baseline_fitness
    stagnant_iters = 0
    invalid_candidate_errors: List[Dict[str, Any]] = []
    error_history_path = transfer_dir / "error_history.jsonl"
    first_iter_best: Optional[float] = None
    transfer_prompt_capture: Optional[str] = None

    for iteration in range(n_iterations):
        iteration_step = iteration + 1
        iter_dir = transfer_dir / f"iteration_{iteration_step}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "candidates").mkdir(exist_ok=True)

        pool_size = len(elite_parents)
        if sample_parents and pool_size > 0:
            rng = np.random.default_rng(
                int(split_seed) + 80_000 + int(iteration_step) * 1_000_003
            )
            parent_idxs, best_k, sampled_k = teh._select_parent_indices_from_elite_pool(
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
        parent_train_lls = [teh._train_loglik_from_elite_tuple(p) for p in selected_parents]

        fresh_n = teh._decayed_fresh_n_for_iteration(
            fresh_n_candidates, iteration, n_iterations, n_candidates
        )
        error_prompt_section = teh._build_past_error_prompt_section(
            invalid_candidate_errors,
            iteration=iteration_step,
            max_error_prompt_chars=max_error_prompt_chars,
        )
        teh._write_iteration_error_prompt_file(iter_dir, error_prompt_section)

        gen_debug: Dict[str, Any] = {}
        capture_prompt = debug_prompt and iteration == 0
        variant_kwargs = {
            "train_trials": pooled_train,
            "extra_prompt_trials": pooled_val if pooled_val else None,
            "max_tokens": llm_max_tokens,
            "dataset": dataset,
            "max_prompt_train_trials": max_prompt_train_trials,
            "max_prompt_trials_per_problem": max_prompt_trials_per_problem,
            "prompt_train_trials_seed": int(split_seed) + 90_000 + iteration_step,
            "fitness_metric": "loglik",
            "max_workers": max_workers,
            "run_prompts_dir": str(target.prompts_dir),
            "max_parent_chars": max_parent_chars,
            "warn_parent_truncation_ratio": warn_parent_truncation_ratio,
            "sample_size_for_warning": sample_size,
            "prompt_stats_path": iter_dir / "prompt_stats.json",
            "hard_prompt_token_cap": hard_prompt_token_cap,
            "strict_prompt_budget": strict_prompt_budget,
            "prompt_token_estimator": prompt_token_estimator,
            "prompt_diagnostics_dir": target.output_dir,
            "phase": "transfer_evolution",
            "participant_id": None,
            "iteration": iteration_step,
            "prompt_suffix": transfer_suffix,
            "past_invalid_program_errors": invalid_candidate_errors,
            "past_error_prompt_section": error_prompt_section,
            "max_error_prompt_chars": max_error_prompt_chars,
            "generation_debug_out": gen_debug if capture_prompt else None,
        }
        candidate_codes, candidate_sources = teh._generate_iteration_candidate_codes(
            client=client,
            model_name=model_name,
            fresh_n_candidates=fresh_n,
            n_candidates=n_candidates,
            fresh_parent_programs=[seed_code],
            normal_parent_programs=parent_codes,
            variant_kwargs=variant_kwargs,
            fresh_parent_train_accuracies=[baseline_ll],
            normal_parent_train_accuracies=parent_train_lls,
        )
        if capture_prompt and gen_debug.get("prompt_text"):
            transfer_prompt_capture = str(gen_debug["prompt_text"])

        selected_results: List[Dict[str, Any]] = []
        num_invalid_candidates = 0
        for idx, code in enumerate(candidate_codes):
            (iter_dir / "candidates" / f"candidate_{idx}.py").write_text(code or "")
            code = teh._sanitize_llm_python_candidate(code, required_markers=("def choose(",))
            if not code:
                continue
            choose_fn, compile_error = teh.compile_program_with_error(code)
            if choose_fn is None:
                num_invalid_candidates += 1
                teh._record_invalid_program_error(
                    invalid_candidate_errors,
                    code=code,
                    exc=compile_error,
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    history_path=error_history_path,
                )
                continue
            try:
                train_eval = teh._evaluate_loglik_for_dataset(
                    dataset, choose_fn, pooled_train, n_seeds=n_eval_seeds
                )
            except (AssertionError, TypeError, ValueError) as exc:
                num_invalid_candidates += 1
                teh._record_invalid_program_error(
                    invalid_candidate_errors,
                    code=code,
                    exc=exc,
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    history_path=error_history_path,
                )
                continue
            if train_eval.get("errors", 0) != 0:
                num_invalid_candidates += 1
                teh._record_invalid_program_error_summary(
                    invalid_candidate_errors,
                    train_eval.get("first_error"),
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    history_path=error_history_path,
                )
                continue
            train_loglik = float(train_eval["avg_loglik"])
            val_loglik: Optional[float] = None
            if use_train_val and pooled_val:
                val_eval = teh._evaluate_loglik_for_dataset(
                    dataset, choose_fn, pooled_val, n_seeds=n_eval_seeds
                )
                if val_eval.get("errors", 0) == 0:
                    val_loglik = float(val_eval["avg_loglik"])
            selection_score = teh._evolution_selection_score(
                train_loglik,
                val_loglik,
                len(pooled_train),
                len(pooled_val),
                evolution_selection_score=evolution_selection_score,
                warn_key="transfer" if use_train_val else None,
            )
            fitness = selection_score if use_train_val else train_loglik
            row: Dict[str, Any] = {
                "idx": idx,
                "code": code,
                "train_loglik": train_loglik,
                "fitness": fitness,
                "selection_score": selection_score,
            }
            if val_loglik is not None:
                row["val_loglik"] = val_loglik
            selected_results.append(row)

        if selected_results:
            selected_results.sort(key=lambda x: x["fitness"], reverse=True)
            for result in selected_results:
                program_id = f"transfer_iteration_{iteration_step}_candidate_{result['idx']}"
                elite_parents.append(
                    (
                        result["code"],
                        result["fitness"],
                        None,
                        program_id,
                        None,
                        None,
                        result["train_loglik"],
                    )
                )

        elite_parents.sort(key=lambda x: x[1], reverse=True)
        elite_cap = teh._elite_pool_capacity(sample_size, elite_pool_size)
        elite_parents = elite_parents[:elite_cap]
        pool_best_selection = float(elite_parents[0][1])

        if iteration == 0:
            first_iter_best = pool_best_selection

        metrics: Dict[str, Any] = {
            "iteration": iteration_step,
            "n_candidates": n_candidates,
            "n_runtime_valid": len(selected_results),
            "num_invalid_candidates": num_invalid_candidates,
            "pool_best_program_id": elite_parents[0][3],
            "pool_best_selection_score": pool_best_selection,
            "evolution_selection_score": evolution_selection_score,
        }
        metrics.update(
            teh._iteration_candidate_source_header(
                fresh_n_candidates,
                fresh_n,
                n_candidates,
                candidate_sources,
                iter_idx=iteration,
                total_iters=n_iterations,
            )
        )
        (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if early_stop_patience is not None:
            improvement = pool_best_selection - float(last_significant_best)
            if improvement >= teh._EARLY_STOP_MIN_IMPROVEMENT:
                last_significant_best = pool_best_selection
                stagnant_iters = 0
            else:
                stagnant_iters += 1
                if stagnant_iters >= early_stop_patience:
                    break

    teh._save_global_elite_pool(transfer_dir, elite_parents)
    pool_best_ll = teh._train_loglik_from_elite_tuple(
        elite_parents[0], evolution_selection_score=evolution_selection_score
    )
    transfer_results = {
        "phase": "transfer",
        "dataset": dataset,
        "config_key": target.config_key,
        "n_source_datasets": len(sources),
        "source_datasets": [s.dataset_alias for s in sources],
        "n_iterations": n_iterations,
        "n_pooled_train_trials": len(pooled_train),
        "n_pooled_val_trials": len(pooled_val),
        "pool_size": len(elite_parents),
        "pool_best_program_id": str(elite_parents[0][3]),
        "pool_best_global_train_loglik": pool_best_ll,
        "pool_best_selection_score": pool_best_selection,
        "evolution_selection_score": evolution_selection_score,
        "baseline_global_train_loglik": baseline_ll,
        "first_iteration_best_selection_score": first_iter_best,
    }
    (transfer_dir / "results.json").write_text(
        json.dumps(transfer_results, indent=2), encoding="utf-8"
    )

    best_code = elite_parents[0][0] or ""
    return PopulationEvolutionResult(
        dataset_alias=target.dataset_alias,
        config_key=target.config_key,
        psych_dataset_split=target.psych_dataset_split,
        output_dir=target.output_dir,
        prompts_dir=target.prompts_dir,
        best_loglik=pool_best_selection,
        best_program_code=best_code,
        best_program_id=str(elite_parents[0][3]),
        first_iter_best_loglik=first_iter_best,
        participant_ids=participant_ids,
        global_prompt_capture=target.global_prompt_capture,
        transfer_prompt_capture=transfer_prompt_capture,
    )


def build_source_contexts_for_target(
    target_key: str,
    global_results: Dict[str, PopulationEvolutionResult],
    *,
    split_ratio: float,
    split_seed: int,
    data_path: str,
    filter_mixed_gambles: bool,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    repo_root: Path,
) -> List[SourceTransferContext]:
    """Build source contexts from all datasets except the target."""
    contexts: List[SourceTransferContext] = []
    for key, result in global_results.items():
        if key == target_key:
            continue
        participants = result.participant_ids or teh.resolve_participants_for_scope(
            dataset=result.dataset_alias,
            repo_root=repo_root,
            participant_scope="all",
            single_participant_id=0,
            range_start_ordinal=None,
            range_end_ordinal=None,
            all_max_participants=None,
            participant_ordinals=None,
            filter_mixed_gambles=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=result.psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        train, val = collect_pooled_train_val_trials(
            result.dataset_alias,
            participants,
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            psych_dataset_split=result.psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        instruction = ""
        try:
            exp = get_psych101_binary_experiment(
                result.dataset_alias,
                0,
                split=result.psych_dataset_split,
                local_dataset=local_dataset,
            )
            instruction = exp.instruction
        except Exception:
            pass
        contexts.append(
            make_source_context(
                dataset_alias=result.dataset_alias,
                pooled_train_val_trials=train + val,
                best_program_code=result.best_program_code,
                best_loglik=result.best_loglik,
                example_seed=split_seed + hash(key) % 10_000,
                instruction=instruction,
            )
        )
    return contexts


def write_transfer_summary_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    """Write summary_csv/transfer.csv with global and transfer best logliks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "global_best", "transfer", "transfer_1st"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_global_phases_parallel(
    specs: Sequence[TransferDatasetSpec],
    worker_fn,
    *,
    max_workers: int,
    n_candidates: int,
    parallel_datasets: bool,
) -> Dict[str, PopulationEvolutionResult]:
    """Run population global evolution for each dataset (optional parallel)."""
    dataset_workers, candidate_workers = _parallel_dataset_pool_sizes(
        max_workers, n_candidates, parallel_datasets
    )
    results: Dict[str, PopulationEvolutionResult] = {}

    def _run_one(spec: TransferDatasetSpec) -> Tuple[str, PopulationEvolutionResult]:
        return spec.config_key, worker_fn(spec, candidate_workers)

    if dataset_workers <= 1 or len(specs) <= 1:
        for spec in tqdm(specs, desc="Global phase (datasets)"):
            key, result = _run_one(spec)
            results[key] = result
        return results

    with ThreadPoolExecutor(max_workers=dataset_workers) as pool:
        futures = {pool.submit(_run_one, spec): spec for spec in specs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Global phase (datasets)"):
            key, result = fut.result()
            results[key] = result
    return results


def run_transfer_phases_parallel(
    specs: Sequence[TransferDatasetSpec],
    global_results: Dict[str, PopulationEvolutionResult],
    worker_fn,
    *,
    max_workers: int,
    n_candidates: int,
    parallel_datasets: bool,
) -> Dict[str, PopulationEvolutionResult]:
    """Run leave-one-out transfer evolution for each target dataset."""
    dataset_workers, candidate_workers = _parallel_dataset_pool_sizes(
        max_workers, n_candidates, parallel_datasets
    )
    transfer_results: Dict[str, PopulationEvolutionResult] = {}

    def _run_one(spec: TransferDatasetSpec) -> Tuple[str, PopulationEvolutionResult]:
        return spec.config_key, worker_fn(spec, global_results, candidate_workers)

    if dataset_workers <= 1 or len(specs) <= 1:
        for spec in tqdm(specs, desc="Transfer phase (datasets)"):
            key, result = _run_one(spec)
            transfer_results[key] = result
        return transfer_results

    with ThreadPoolExecutor(max_workers=dataset_workers) as pool:
        futures = {pool.submit(_run_one, spec): spec for spec in specs}
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Transfer phase (datasets)"
        ):
            key, result = fut.result()
            transfer_results[key] = result
    return transfer_results
