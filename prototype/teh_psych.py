#!/usr/bin/env python3
"""
prototype/teh_psych.py — Population-level PICS/TEH prototype over Psych-101 train experiments.

One global categorical program per experiment; pooled trials across participants.
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
from utils.teh.teh_runtime import DEFAULT_SEED_PROGRAM
from utils.teh_psych.adapters import (
    UnsupportedParserError,
    pool_categorical_trials_from_rows,
)
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
from utils.teh_psych.prompts import (
    DEFAULT_CATEGORICAL_PROMPT,
    DEFAULT_PARSE_PLAN_PROMPT,
    setup_experiment_prompts,
)
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
from utils.teh_psych.trial_validation import summarize_trial_action_space, validate_categorical_trials

TEH_PSYCH_WANDB_PROJECT = "teh_psych"


def _teh_psych_output_dir(timestamp: str, *, ablation: Optional[str] = None) -> Path:
    root = "generated_outputs_ablation" if ablation else "generated_outputs"
    run_name = ablation if ablation else f"run_{timestamp}"
    return Path(root) / "psych101_train" / "teh_psych" / run_name


def _build_client(args: argparse.Namespace) -> Optional[OpenAI]:
    if args.eval_only or args.n_iterations <= 0:
        return None
    if args.mode == "local":
        return OpenAI(base_url=args.llm_server_url, api_key=args.llm_api_key)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def process_one_experiment(
    experiment_id: str,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    summary_dir: Path,
    train_split_ds,
    test_split_ds: Optional[Any],
    client: Optional[OpenAI],
) -> DatasetResult:
    result = DatasetResult(experiment_id=experiment_id, status="running")
    debug_dir = dataset_debug_dir(summary_dir, experiment_id)
    exp_run_dir = run_dir / "experiments" / debug_dir.name
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result.stage_reached = "load_rows"
        alias = alias_for_experiment_id(experiment_id)
        spec = spec_for_experiment_id(experiment_id)
        if alias is None or spec is None:
            raise UnsupportedParserError(
                f"No implemented Psych-101 parser for experiment_id={experiment_id!r}"
            )

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
            raise ValueError(f"No rows selected for experiment {experiment_id!r}: {sampling_note}")

        result.stage_reached = "sample_rows"
        rows = [dict(filtered[i]) for i in row_indices]
        write_json(debug_dir / "sampled_rows.json", rows[:5])
        write_json(
            debug_dir / "row_sampling.json",
            {"indices": row_indices, "note": sampling_note},
        )

        result.stage_reached = "parse_or_adapter"
        train_trials, val_trials, test_trials, adapter_status = pool_categorical_trials_from_rows(
            rows,
            alias=alias,
            experiment_id=experiment_id,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
        )
        write_json(debug_dir / "adapter_status.json", adapter_status.__dict__)
        if adapter_status.n_rows_parsed == 0:
            msg = adapter_status.parse_errors[0] if adapter_status.parse_errors else "no rows parsed"
            raise RuntimeError(msg)

        all_trials = train_trials + val_trials + test_trials
        result.n_parsed_trials = len(all_trials)
        write_json(
            debug_dir / "parsed_trial_preview.json",
            all_trials[:8],
        )

        result.stage_reached = "validate_trials"
        val_errors, n_prediction = validate_categorical_trials(all_trials)
        result.n_prediction_trials = n_prediction
        action_summary = summarize_trial_action_space(all_trials)
        result.num_actions_min = action_summary["num_actions_min"]
        result.num_actions_max = action_summary["num_actions_max"]
        result.is_variable_k = bool(action_summary["is_variable_k"])
        if val_errors:
            write_json(debug_dir / "validation_errors.json", val_errors[:50])
            raise ValueError(val_errors[0])
        if n_prediction < args.min_pooled_prediction_trials:
            raise ValueError(
                f"Only {n_prediction} prediction trials after pooling; "
                f"need >= {args.min_pooled_prediction_trials}"
            )

        result.stage_reached = "split_trials"
        if len(train_trials) < 1 or len(val_trials) < 1 or len(test_trials) < 1:
            raise ValueError(
                f"Insufficient split sizes: train={len(train_trials)}, "
                f"val={len(val_trials)}, test={len(test_trials)}"
            )

        result.stage_reached = "build_prompt"
        instruction = rows[0].get("instruction", "") if rows else ""
        seed_path = Path(args.seed_path) if args.seed_path else DEFAULT_SEED_PROGRAM
        base_prompt = Path(args.categorical_prompt_path)
        parse_plan = Path(args.parse_plan_prompt_path)
        prompts_dir = setup_experiment_prompts(
            exp_run_dir / "prompts",
            experiment_id=experiment_id,
            alias=alias,
            instruction=str(instruction or ""),
            sample_trials=all_trials[:8],
            seed_program_path=seed_path,
            base_prompt_path=base_prompt,
            parse_plan_prompt_path=parse_plan,
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
                client=client or OpenAI(api_key="offline"),
                max_prompt_train_trials=args.max_prompt_train_trials,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                split_seed=args.split_seed,
                output_dir=exp_run_dir,
                run_prompts_dir=str(prompts_dir),
                dataset_label=experiment_id,
            )
        else:
            if client is None:
                raise RuntimeError(
                    "LLM client unavailable (set OPENAI_API_KEY or use --mode local). "
                    "Use --eval_only to skip evolution."
                )
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
            )
        best_path = Path(evo_out["best_program_path"])
        result.best_program_path = str(best_path)

        result.stage_reached = "evaluate_program"
        best_fn = compile_program(evo_out["best_code"])
        if best_fn is None:
            raise RuntimeError("Best program failed to compile after evolution")
        train_eval = evaluate_categorical_program(best_fn, train_trials, n_seeds=args.n_eval_seeds)
        val_eval = evaluate_categorical_program(best_fn, val_trials, n_seeds=args.n_eval_seeds)
        test_eval = evaluate_categorical_program(best_fn, test_trials, n_seeds=args.n_eval_seeds)
        result.train_loglik = float(train_eval["avg_loglik"])
        result.val_loglik = float(val_eval["avg_loglik"])
        result.test_loglik = float(test_eval["avg_loglik"])

        ext_test_trials: List[Dict[str, Any]] = []
        if test_split_ds is not None:
            test_filtered = filter_rows_for_experiment(test_split_ds, experiment_id)
            if len(test_filtered) > 0:
                test_indices, _ = sample_row_indices(
                    len(test_filtered),
                    max_participants=args.max_participants_per_experiment,
                    range_start_ordinal=args.range_start_ordinal,
                    range_end_ordinal=args.range_end_ordinal,
                )
                test_rows = [dict(test_filtered[i]) for i in test_indices]
                _, _, ext_test, ext_status = pool_categorical_trials_from_rows(
                    test_rows,
                    alias=alias,
                    experiment_id=experiment_id,
                    split_ratio=args.split_ratio,
                    split_seed=args.split_seed,
                )
                ext_test_trials = ext_test
                write_json(debug_dir / "hf_test_adapter_status.json", ext_status.__dict__)
                if ext_test_trials:
                    ext_eval = evaluate_categorical_program(
                        best_fn, ext_test_trials, n_seeds=args.n_eval_seeds
                    )
                    result.extra["hf_test_loglik"] = float(ext_eval["avg_loglik"])
                    result.extra["hf_test_n_trials"] = len(ext_test_trials)
                    result.notes = (
                        f"{sampling_note}; hf_test_loglik={ext_eval['avg_loglik']:.4f} "
                        f"on {len(ext_test_trials)} trials"
                    )

        write_json(
            debug_dir / "evaluation_metrics.json",
            {
                "train": train_eval,
                "val": val_eval,
                "test": test_eval,
                "hf_test_n_trials": len(ext_test_trials),
                "hf_test_loglik": result.extra.get("hf_test_loglik"),
            },
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

    except UnsupportedParserError as exc:
        record_failure(result, "unsupported_parser_missing", str(exc), exc)
        return result
    except Exception as exc:
        stage = result.stage_reached or "unknown_error"
        if stage == "parse_or_adapter" and not isinstance(exc, UnsupportedParserError):
            stage = "parse_or_adapter"
        elif stage not in {
            "discover_experiments",
            "load_rows",
            "sample_rows",
            "parse_or_adapter",
            "validate_trials",
            "split_trials",
            "build_prompt",
            "compile_program",
            "evolve_program",
            "evaluate_program",
            "write_outputs",
            "skipped_non_categorical",
            "unsupported_parser_missing",
        }:
            stage = "unknown_error"
        record_failure(result, stage, str(exc), exc)
        write_json(debug_dir / "failure.json", result.to_json())
        return result


def _parse_max_experiments(value: str) -> Optional[int]:
    """Accept an integer cap or 'all' (case-insensitive) for no limit."""
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
        description="TEH Psych prototype: population-level categorical evolution over Psych-101 train experiments."
    )
    parser.add_argument("--psych_dataset_split", type=str, default=DEFAULT_PSYCH_DATASET_SPLIT, choices=["train", "test"])
    parser.add_argument("--local_dataset", type=str, default=None)
    parser.add_argument(
        "--max_experiments",
        type=_parse_max_experiments,
        default=None,
        help="Max experiments to process (default: all). Use an integer or 'all'.",
    )
    parser.add_argument("--experiment_ids", type=str, default=None, help="Comma-separated experiment id subset")
    parser.add_argument("--range_start_ordinal", type=int, default=None)
    parser.add_argument("--range_end_ordinal", type=int, default=None)
    parser.add_argument("--max_participants_per_experiment", type=int, default=50)
    parser.add_argument("--min_pooled_prediction_trials", type=int, default=500)
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prototype_summary_dir", type=str, default=None)
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
    parser.add_argument("--eval_only", action="store_true", help="Skip evolution (n_iterations=0)")
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.eval_only:
        args.n_iterations = 0
    if args.experiment_ids:
        experiment_filter = [x.strip() for x in args.experiment_ids.split(",") if x.strip()]
    else:
        experiment_filter = None

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else _teh_psych_output_dir(timestamp, ablation=args.ablation)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = ensure_summary_dir(run_dir, args.prototype_summary_dir)
    _write_command_line_log(run_dir)

    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(
                project=TEH_PSYCH_WANDB_PROJECT,
                name=f"teh_psych_{timestamp}",
                config=vars(args),
            )
        except Exception as exc:
            print(f"[teh_psych] wandb init failed ({exc}); continuing without logging.")

    print(f"Psych-101 corpus: {args.psych_dataset_split} -> {hf_id_for_psych_dataset_split(args.psych_dataset_split)}")
    print(f"Run directory: {run_dir}")
    print(f"Prototype summary: {summary_dir}")

    train_split_ds = load_psych101_split(args.psych_dataset_split, local_dataset=args.local_dataset)
    test_split_ds = None
    try:
        test_split_ds = load_psych101_split("test", local_dataset=args.local_dataset)
    except Exception as exc:
        print(f"[teh_psych] Could not load Psych-101-test split ({exc}); skipping HF test eval.")

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
                test_split_ds=test_split_ds,
                client=client,
            )
        except Exception as exc:
            result = DatasetResult(experiment_id=experiment_id)
            record_failure(result, "unknown_error", str(exc), exc)
        results.append(result)
        append_dataset_result_csv(summary_dir, result)
        append_dataset_result_jsonl(summary_dir, result)
        status = result.status
        print(f"Result: {status} (stage={result.stage_reached}, failure={result.failure_stage or '-'})")
        if wandb is not None:
            wandb.log(
                {
                    "experiment_ordinal": ordinal,
                    "status": 1 if status == "success" else 0,
                    "n_prediction_trials": result.n_prediction_trials,
                    "train_loglik": result.train_loglik,
                    "val_loglik": result.val_loglik,
                    "test_loglik": result.test_loglik,
                },
                step=ordinal,
            )
        if status != "success" and not args.continue_on_error:
            print("Stopping run because --continue_on_error false.")
            break

    finalize_summaries(summary_dir, results)
    n_ok = sum(1 for r in results if r.status == "success")
    print(f"\nFinished: {n_ok}/{len(results)} experiments succeeded.")
    print(f"Summary written to {summary_dir}")
    if wandb is not None:
        wandb.finish()
    return 0 if n_ok > 0 or args.continue_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
