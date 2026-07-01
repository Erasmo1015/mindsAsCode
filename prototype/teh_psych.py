#!/usr/bin/env python3
"""
prototype/teh_psych.py — Population-level categorical TEH over Psych-101 train experiments.

Pipeline per experiment:
  LLM parser plan → fixed parser engine → categorical trials → population program evolution
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import DEFAULT_PSYCH_DATASET_SPLIT, hf_id_for_psych_dataset_split
from teh import (
    BEST_PROGRAM_FILENAME,
    _normalize_early_stop_iters,
    _write_command_line_log,
    compile_program,
    load_seed_program,
)
from utils.teh_psych.action_id_normalization import normalize_categorical_trials_action_ids
from utils.teh_psych.adapters import pool_manual_parser_trials_from_rows
from utils.teh_psych.categorical_eval import evaluate_categorical_program
from utils.teh_psych.dataset_loop import (
    alias_for_experiment_id,
    discover_psych101_train_experiments,
    filter_rows_for_experiment,
    load_psych101_split,
    sample_row_indices,
    spec_for_experiment_id,
)
from utils.teh_psych.evolution import run_population_evolution
from utils.teh_psych.parser_plan import (
    CACHE_MISS_CLIENT_MSG,
    ParsePlanError,
    StateMachineNotImplementedError,
    default_parse_plan_prompt_path,
    run_parse_plan_pipeline,
)
from utils.teh_psych.prompts import setup_experiment_prompts
from utils.teh_psych.reporting import (
    DatasetResult,
    append_dataset_result_csv,
    append_dataset_result_jsonl,
    dataset_debug_dir,
    ensure_summary_dir,
    finalize_summaries,
    record_failure,
    write_json,
)
from utils.teh_psych.seed_program import resolve_seed_program_path
from utils.teh_psych.trial_split import split_pooled_categorical_trials
from utils.teh_psych.trial_validation import (
    summarize_trial_action_space,
    trial_filtering_summary,
    validate_categorical_trials,
    partition_pooled_trials,
)

TEH_PSYCH_WANDB_PROJECT = "teh_psych"

FAILURE_STAGES = frozenset({
    "discover_experiments",
    "load_rows",
    "sample_rows",
    "build_parse_plan_prompt",
    "generate_parse_plan",
    "validate_parse_plan",
    "execute_parse_plan",
    "validate_trials",
    "split_trials",
    "build_program_prompt",
    "compile_program",
    "evolve_program",
    "evaluate_program",
    "write_outputs",
    "unsupported_parser_missing",
    "unknown_error",
})


def _teh_psych_output_dir(timestamp: str, *, ablation: Optional[str] = None) -> Path:
    root = Path("generated_outputs_teh_psych")
    run_name = ablation if ablation else f"run_{timestamp}"
    return root / run_name


def _make_unique_run_dir(base_dir: Path) -> Path:
    for i in range(1000):
        candidate = base_dir if i == 0 else Path(f"{base_dir}_{i}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique run directory from {base_dir}")


def _build_client(args: argparse.Namespace) -> Optional[OpenAI]:
    if args.mode == "local":
        return OpenAI(base_url=args.llm_server_url, api_key=args.llm_api_key)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _parse_plan_model(args: argparse.Namespace) -> str:
    return args.parse_plan_model_name or args.model_name


def _task_description_for_experiment(experiment_id: str) -> str:
    spec = spec_for_experiment_id(experiment_id)
    if spec:
        return str(spec.get("task_description") or "")
    return ""


def _trials_via_parse_plan(
    *,
    client: Optional[OpenAI],
    experiment_id: str,
    rows: List[Dict[str, Any]],
    row_indices: List[int],
    debug_dir: Path,
    args: argparse.Namespace,
    cache_dir: Path,
    result: DatasetResult,
) -> List[Dict[str, Any]]:
    result.stage_reached = "build_parse_plan_prompt"
    parse_plan_template = Path(args.parse_plan_prompt_path)
    if not parse_plan_template.is_file():
        parse_plan_template = default_parse_plan_prompt_path()

    n_plan_rows = max(3, min(10, int(args.n_parse_plan_rows)))
    result.n_parse_plan_rows = min(n_plan_rows, len(rows))
    result.adapter_type = "parse_plan"
    result.parse_plan_model = _parse_plan_model(args)

    plan_run = run_parse_plan_pipeline(
        client=client,
        experiment_id=experiment_id,
        rows=rows,
        row_indices=row_indices,
        debug_dir=debug_dir,
        template_path=parse_plan_template,
        model_name=result.parse_plan_model,
        split_name=args.psych_dataset_split,
        task_description=_task_description_for_experiment(experiment_id),
        reuse_cache=bool(args.reuse_parse_plan_cache),
        cache_dir=cache_dir,
        parse_plan_max_tokens=args.parse_plan_max_tokens,
        n_parse_plan_rows=args.n_parse_plan_rows,
    )

    result.parse_plan_status = plan_run.status
    result.parse_plan_cached = plan_run.cached
    result.parse_plan_path = plan_run.plan_path
    result.parse_plan_human_review_required = plan_run.human_review_required
    result.parse_plan_raw_format_type = plan_run.raw_format_type
    result.parse_plan_failure_message = plan_run.failure_message

    if plan_run.failure_stage:
        result.stage_reached = plan_run.failure_stage

    if plan_run.status in ("prompt_failed", "generate_failed"):
        raise ParsePlanError(plan_run.failure_message or CACHE_MISS_CLIENT_MSG)
    if plan_run.status in ("validation_failed",):
        raise ParsePlanError(plan_run.failure_message or plan_run.validation_errors[0])
    if plan_run.status in ("execute_failed",):
        if plan_run.failure_message == "state_machine_not_implemented":
            raise StateMachineNotImplementedError("state_machine_not_implemented")
        raise ParsePlanError(plan_run.failure_message)

    return plan_run.trials


def _trials_via_manual_fallback(
    *,
    experiment_id: str,
    rows: List[Dict[str, Any]],
    row_indices: List[int],
    debug_dir: Path,
    result: DatasetResult,
) -> List[Dict[str, Any]]:
    alias = alias_for_experiment_id(experiment_id)
    if alias is None:
        raise RuntimeError(f"No manual parser fallback for experiment_id={experiment_id!r}")
    trials, status = pool_manual_parser_trials_from_rows(
        rows, alias=alias, experiment_id=experiment_id, row_indices=row_indices
    )
    write_json(debug_dir / "manual_fallback_status.json", status.__dict__)
    result.used_existing_parser_fallback = True
    result.adapter_type = "manual_fallback"
    result.notes = (result.notes + "; manual_parser_fallback").strip("; ")
    if not trials:
        msg = status.parse_errors[0] if status.parse_errors else "manual parser produced no trials"
        raise RuntimeError(msg)
    return trials


def process_one_experiment(
    experiment_id: str,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    summary_dir: Path,
    train_split_ds,
    client: Optional[OpenAI],
    parse_plan_cache_dir: Path,
) -> DatasetResult:
    result = DatasetResult(experiment_id=experiment_id, status="running")
    debug_dir = dataset_debug_dir(summary_dir, experiment_id)
    exp_run_dir = run_dir / "experiments" / debug_dir.name
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result.stage_reached = "load_rows"
        filtered = filter_rows_for_experiment(train_split_ds, experiment_id)
        result.n_rows_total = len(filtered)
        if result.n_rows_total == 0:
            raise ValueError(f"No train rows for experiment {experiment_id!r}")

        row_indices, sampling_note = sample_row_indices(
            result.n_rows_total,
            max_participants=args.max_participants_per_experiment,
            range_start_ordinal=args.range_start_ordinal,
            range_end_ordinal=args.range_end_ordinal,
        )
        result.n_rows_used = len(row_indices)
        result.notes = sampling_note
        if not row_indices:
            raise ValueError(f"No rows selected: {sampling_note}")

        result.stage_reached = "sample_rows"
        rows = [dict(filtered[i]) for i in row_indices]
        write_json(debug_dir / "sampled_rows.json", rows[:5])
        write_json(debug_dir / "row_sampling.json", {"indices": row_indices, "note": sampling_note})

        all_trials: List[Dict[str, Any]] = []
        try:
            all_trials = _trials_via_parse_plan(
                client=client,
                experiment_id=experiment_id,
                rows=rows,
                row_indices=row_indices,
                debug_dir=debug_dir,
                args=args,
                cache_dir=parse_plan_cache_dir,
                result=result,
            )
        except (ParsePlanError, StateMachineNotImplementedError) as exc:
            if args.allow_existing_parser_fallback and alias_for_experiment_id(experiment_id):
                result.parse_plan_failure_message = str(exc)
                all_trials = _trials_via_manual_fallback(
                    experiment_id=experiment_id,
                    rows=rows,
                    row_indices=row_indices,
                    debug_dir=debug_dir,
                    result=result,
                )
            else:
                raise

        all_trials = normalize_categorical_trials_action_ids(all_trials)

        result.n_parsed_trials = len(all_trials)
        prediction_trials, context_only_trials = partition_pooled_trials(all_trials)
        write_json(
            debug_dir / "trial_filtering.json",
            trial_filtering_summary(all_trials, prediction_trials),
        )
        if context_only_trials:
            write_json(
                debug_dir / "context_only_trial_preview.json",
                context_only_trials[:8],
            )

        result.stage_reached = "validate_trials"
        result.n_prediction_trials = len(prediction_trials)
        val_errors, _ = validate_categorical_trials(prediction_trials)
        action_summary = summarize_trial_action_space(prediction_trials)
        result.num_actions_min = action_summary["num_actions_min"]
        result.num_actions_max = action_summary["num_actions_max"]
        result.is_variable_k = bool(action_summary["is_variable_k"])
        if val_errors:
            write_json(debug_dir / "validation_errors.json", val_errors[:50])
            raise ValueError(val_errors[0])
        if result.n_prediction_trials < args.min_pooled_prediction_trials:
            raise ValueError(
                f"Only {result.n_prediction_trials} prediction trials; "
                f"need >= {args.min_pooled_prediction_trials}"
            )

        result.stage_reached = "split_trials"
        train_trials, val_trials, test_trials = split_pooled_categorical_trials(
            prediction_trials,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
        )
        write_json(
            debug_dir / "split_sizes.json",
            {"train": len(train_trials), "val": len(val_trials), "test": len(test_trials)},
        )

        result.stage_reached = "build_program_prompt"
        instruction = str(rows[0].get("instruction") or rows[0].get("text", "")[:500])
        alias = alias_for_experiment_id(experiment_id)
        seed_path = resolve_seed_program_path(args.seed_path)
        prompts_dir = setup_experiment_program_prompts(
            exp_run_dir / "prompts",
            experiment_id=experiment_id,
            alias=alias,
            instruction=instruction,
            sample_trials=prediction_trials[:8],
            seed_program_path=seed_path,
            base_prompt_path=Path(args.categorical_prompt_path),
        )
        seed_program_path = str(prompts_dir / "seed_program.py")

        result.stage_reached = "compile_program"
        seed_code = load_seed_program(seed_program_path)
        if compile_program(seed_code) is None:
            raise RuntimeError(f"Seed program failed to compile: {seed_program_path}")

        result.stage_reached = "evolve_program"
        if args.n_iterations <= 0:
            evo_out = run_population_evolution(
                pooled_train=train_trials,
                pooled_val=val_trials,
                seed_program_path=seed_program_path,
                n_iterations=0,
                n_candidates_per_iteration=args.n_candidates,
                fresh_n_candidates=0,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
                sampled_parents_decay=args.sampled_parents_decay,
                elite_pool_size=args.elite_pool_size,
                model_name=args.model_name,
                client=client,
                max_prompt_train_trials=args.max_prompt_train_trials,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                split_seed=args.split_seed,
                output_dir=exp_run_dir,
                run_prompts_dir=str(prompts_dir),
                dataset_label=experiment_id,
                simple_log=args.simple_log,
            )
        else:
            if client is None:
                raise RuntimeError("LLM client required for evolution (omit --eval_only).")
            evo_out = run_population_evolution(
                pooled_train=train_trials,
                pooled_val=val_trials,
                seed_program_path=seed_program_path,
                n_iterations=args.n_iterations,
                n_candidates_per_iteration=args.n_candidates,
                fresh_n_candidates=args.fresh_n_candidates,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
                sampled_parents_decay=args.sampled_parents_decay,
                elite_pool_size=args.elite_pool_size,
                model_name=args.model_name,
                client=client,
                max_prompt_train_trials=args.max_prompt_train_trials,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                split_seed=args.split_seed,
                llm_max_tokens=args.llm_max_tokens,
                max_workers=args.max_workers,
                n_eval_seeds=args.n_eval_seeds,
                output_dir=exp_run_dir,
                run_prompts_dir=str(prompts_dir),
                early_stop_iters=_normalize_early_stop_iters(args.early_stop_iters),
                hard_prompt_token_cap=args.hard_prompt_token_cap,
                strict_prompt_budget=args.strict_prompt_budget,
                prompt_token_estimator=args.prompt_token_estimator,
                prompt_debug=args.prompt_debug,
                prompt_debug_on_no_valid=args.prompt_debug_on_no_valid,
                prompt_debug_exit=args.prompt_debug_exit,
                evolution_selection_score=args.evolution_selection_score,
                max_error_prompt_chars=args.max_error_prompt_chars,
                max_parent_chars=args.max_parent_chars,
                warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
                dataset_label=experiment_id,
                simple_log=args.simple_log,
            )

        best_path = Path(evo_out["best_program_path"])
        result.best_program_path = str(best_path)

        result.stage_reached = "evaluate_program"
        best_fn = compile_program(evo_out["best_code"])
        if best_fn is None:
            raise RuntimeError("Best program failed to compile")
        train_eval = evaluate_categorical_program(best_fn, train_trials, n_seeds=args.n_eval_seeds)
        val_eval = evaluate_categorical_program(best_fn, val_trials, n_seeds=args.n_eval_seeds)
        test_eval = evaluate_categorical_program(best_fn, test_trials, n_seeds=args.n_eval_seeds)
        result.train_loglik = float(train_eval["avg_loglik"])
        result.val_loglik = float(val_eval["avg_loglik"])
        result.test_loglik = float(test_eval["avg_loglik"])

        write_json(
            debug_dir / "evaluation_metrics.json",
            {"train": train_eval, "val": val_eval, "test": test_eval},
        )
        if best_path.is_file():
            (debug_dir / BEST_PROGRAM_FILENAME).write_text(
                best_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

        result.stage_reached = "write_outputs"
        result.status = "success"
        result.failure_stage = ""
        result.failure_message = ""
        return result

    except StateMachineNotImplementedError as exc:
        record_failure(result, "execute_parse_plan", str(exc), exc)
        result.parse_plan_failure_message = str(exc)
        write_json(debug_dir / "failure.json", result.to_json())
        return result
    except ParsePlanError as exc:
        stage = result.stage_reached if result.stage_reached in FAILURE_STAGES else "generate_parse_plan"
        record_failure(result, stage, str(exc), exc)
        result.parse_plan_failure_message = str(exc)
        write_json(debug_dir / "failure.json", result.to_json())
        return result
    except Exception as exc:
        stage = result.stage_reached or "unknown_error"
        if stage not in FAILURE_STAGES:
            stage = "unknown_error"
        record_failure(result, stage, str(exc), exc)
        write_json(debug_dir / "failure.json", result.to_json())
        return result


def setup_experiment_program_prompts(
    prompts_dir: Path,
    *,
    experiment_id: str,
    alias: Optional[str],
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    seed_program_path: Path,
    base_prompt_path: Path,
) -> Path:
    return setup_experiment_prompts(
        prompts_dir,
        experiment_id=experiment_id,
        alias=alias,
        instruction=instruction,
        sample_trials=sample_trials,
        seed_program_path=seed_program_path,
        base_prompt_path=base_prompt_path,
        parse_plan_prompt_path=default_parse_plan_prompt_path(),
    )


def _parse_max_experiments(value: str) -> Optional[int]:
    text = str(value).strip().lower()
    if text in ("all", "none", ""):
        return None
    try:
        n = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max_experiments must be a positive integer or 'all', got {value!r}"
        ) from exc
    if n < 1:
        raise argparse.ArgumentTypeError(f"--max_experiments must be >= 1 or 'all', got {n}")
    return n


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TEH Psych: LLM parser plan + categorical population evolution over Psych-101 train."
    )
    parser.add_argument("--psych_dataset_split", type=str, default="train", choices=["train"])
    parser.add_argument("--local_dataset", type=str, default=None)
    parser.add_argument(
        "--max_experiments",
        type=_parse_max_experiments,
        default=None,
        help="Max experiments (integer or 'all').",
    )
    parser.add_argument("--experiment_ids", type=str, default=None)
    parser.add_argument("--range_start_ordinal", type=int, default=None)
    parser.add_argument("--range_end_ordinal", type=int, default=None)
    parser.add_argument("--max_participants_per_experiment", type=int, default=50)
    parser.add_argument("--min_pooled_prediction_trials", type=int, default=500)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prototype_summary_dir", type=str, default=None)
    parser.add_argument(
        "--allow_existing_parser_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reuse_parse_plan_cache", action="store_true", default=False)
    parser.add_argument("--parse_plan_cache_dir", type=str, default=None)
    parser.add_argument("--n_parse_plan_rows", type=int, default=5)
    parser.add_argument("--parse_plan_model_name", type=str, default=None)
    parser.add_argument("--parse_plan_max_tokens", type=int, default=4000)
    parser.add_argument(
        "--categorical_prompt_path",
        type=str,
        default="prompts/teh_psych/infer_single_choice.txt",
    )
    parser.add_argument(
        "--parse_plan_prompt_path",
        type=str,
        default="prompts/teh_psych/utils/parse_plan.txt",
    )
    parser.add_argument("--seed_path", type=str, default=None)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--n_iterations", type=int, default=5)
    parser.add_argument("--early_stop_iters", type=int, default=-1)
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--fresh_n_candidates", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--sample_parents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampled_parents_decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elite_pool_size", type=int, default=None)
    parser.add_argument("--n_eval_seeds", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--mode", type=str, default="default", choices=["default", "local"])
    parser.add_argument("--llm_server_url", type=str, default=os.getenv("VLLM_LOCAL_URL", "http://localhost:8000/v1"))
    parser.add_argument("--llm_api_key", type=str, default=os.getenv("VLLM_LOCAL_API_KEY", "EMPTY"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument(
        "--simple_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip per-iteration candidate program files and prompt_diagnostics.jsonl; "
            "keep per-iteration metrics.json, prompt_stats.json, summary CSVs, and "
            "population_phase/best_program.py."
        ),
    )
    parser.add_argument("--evolution_selection_score", type=str, default="train_val", choices=["train", "train_val"])
    parser.add_argument("--max_prompt_train_trials", type=int, default=1_000_000)
    parser.add_argument("--max_prompt_trials_per_problem", type=int, default=0)
    parser.add_argument("--llm_max_tokens", type=int, default=800)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--hard_prompt_token_cap", type=int, default=14000)
    parser.add_argument("--strict_prompt_budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt_token_estimator", type=str, default="char4")
    parser.add_argument("--prompt_debug", action="store_true")
    parser.add_argument("--prompt_debug_on_no_valid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt_debug_exit", action="store_true")
    parser.add_argument("--max_error_prompt_chars", type=int, default=1200)
    parser.add_argument("--max_parent_chars", type=int, default=6000)
    parser.add_argument("--warn_parent_truncation_ratio", type=float, default=0.5)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip program evolution only; still runs parser-plan generation/execution.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.psych_dataset_split != "train":
        print("Error: prototype uses Psych-101 train split only.")
        return 1
    if args.eval_only:
        args.n_iterations = 0

    experiment_filter = None
    if args.experiment_ids:
        experiment_filter = [x.strip() for x in args.experiment_ids.split(",") if x.strip()]

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _make_unique_run_dir(_teh_psych_output_dir(timestamp, ablation=args.ablation))
    summary_dir = ensure_summary_dir(run_dir, args.prototype_summary_dir)
    _write_command_line_log(run_dir)

    if args.parse_plan_cache_dir:
        parse_plan_cache_dir = Path(args.parse_plan_cache_dir).expanduser()
    else:
        parse_plan_cache_dir = run_dir / "parse_plan_cache"
    parse_plan_cache_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(project=TEH_PSYCH_WANDB_PROJECT, name=f"teh_psych_{timestamp}", config=vars(args))
        except Exception as exc:
            print(f"[teh_psych] wandb init failed ({exc}); continuing without logging.")

    print(f"Psych-101 corpus: train -> {hf_id_for_psych_dataset_split('train')}")
    print(f"Run directory: {run_dir}")
    print(f"Prototype summary: {summary_dir}")
    print(f"Parse-plan cache: {parse_plan_cache_dir}")

    train_split_ds = load_psych101_split("train", local_dataset=args.local_dataset)
    experiment_ids = discover_psych101_train_experiments(
        train_split_ds,
        experiment_ids=experiment_filter,
        max_experiments=args.max_experiments,
    )
    print(f"Discovered {len(experiment_ids)} experiment(s) to process.")

    client = _build_client(args)
    results: List[DatasetResult] = []

    for ordinal, experiment_id in enumerate(experiment_ids):
        print(f"\n{'=' * 80}\n[{ordinal + 1}/{len(experiment_ids)}] {experiment_id}\n{'=' * 80}")
        try:
            result = process_one_experiment(
                experiment_id,
                args=args,
                run_dir=run_dir,
                summary_dir=summary_dir,
                train_split_ds=train_split_ds,
                client=client,
                parse_plan_cache_dir=parse_plan_cache_dir,
            )
        except Exception as exc:
            result = DatasetResult(experiment_id=experiment_id)
            record_failure(result, "unknown_error", str(exc), exc)
        results.append(result)
        append_dataset_result_csv(summary_dir, result)
        append_dataset_result_jsonl(summary_dir, result)
        print(
            f"Result: {result.status} (stage={result.stage_reached}, "
            f"adapter={result.adapter_type or '-'}, failure={result.failure_stage or '-'})"
        )
        if wandb is not None:
            wandb.log(
                {
                    "experiment_ordinal": ordinal,
                    "status": 1 if result.status == "success" else 0,
                    "n_prediction_trials": result.n_prediction_trials,
                    "train_loglik": result.train_loglik,
                },
                step=ordinal,
            )
        if result.status != "success" and not args.continue_on_error:
            print("Stopping because --continue_on_error false.")
            break

    finalize_summaries(summary_dir, results)
    n_ok = sum(1 for r in results if r.status == "success")
    print(f"\nFinished: {n_ok}/{len(results)} experiments succeeded.")
    print(f"Summary: {summary_dir}")
    if wandb is not None:
        wandb.finish()
    return 0 if n_ok > 0 or args.continue_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
