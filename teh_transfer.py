#!/usr/bin/env python3
"""
teh_transfer.py — Population-level cross-task cognitive program transfer.

Runs global evolution on pooled train+val trials for each configured dataset, then
leave-one-dataset-out transfer using source programs from the other datasets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import teh
from data_modules.mixed_gambles import DEFAULT_CSV_PATH
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PSYCH101_BINARY_DATASETS,
    normalize_psych101_dataset_alias,
)
from utils.teh.teh_datasets import is_binary_loglik_dataset
from utils.teh.teh_runtime import DEFAULT_SEED_PROGRAM

from utils.teh_transfer.config import filter_transfer_specs, load_transfer_datasets
from utils.teh_transfer.participants import resolve_participants_for_transfer
from utils.teh_transfer.evolution import (
    build_source_contexts_for_target,
    run_dataset_global_phase,
    run_global_phases_parallel,
    run_transfer_evolution_phase,
    run_transfer_phases_parallel,
    transfer_output_run_dir,
    write_debug_prompts_file,
    write_transfer_summary_csv,
    write_run_train_val_summary_csv,
    write_run_test_loglik_summary_csv,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TEH transfer: population-level global evolution per dataset, then "
            "leave-one-dataset-out cross-task program transfer."
        )
    )
    parser.add_argument(
        "--transfer_config",
        type=str,
        default="configs/teh_transfer.yaml",
        help="YAML listing datasets for transfer (default: configs/teh_transfer.yaml).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="KEY",
        help="Optional subset of config keys or dataset aliases to run.",
    )
    parser.add_argument(
        "--transfer_iters",
        type=int,
        default=10,
        help="Transfer-phase evolution iterations per target dataset (default: 10).",
    )
    parser.add_argument(
        "--transfer_max_prompt_trials",
        type=int,
        default=1,
        help=(
            "Max target train+val trials injected into each transfer prompt "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--global_iters",
        type=int,
        default=10,
        help="Population global evolution iterations per dataset (default: 10).",
    )
    parser.add_argument(
        "--debug_prompt",
        action="store_true",
        default=False,
        help=(
            "Write run_xxx/debug/prompt_<dataset>.txt with one full global-iter and "
            "one full transfer-iter prompt per dataset."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override run directory (default: generated_outputs_transfer/teh_transfer/run_*).",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default=None,
        help="Seed program path (default: persona_code_example/te_vanilla/choices13k.py).",
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default="prompts/teh/infer_single_choice.txt",
        help="Base loglik prompt template for LLM prompt generation.",
    )
    parser.add_argument(
        "--no_llm_prompt",
        action="store_true",
        help="Skip LLM prompt generation; use merged fallback prompt.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Legacy TEH arg (Psych-101 uses Hugging Face).",
    )
    parser.add_argument(
        "--local_dataset",
        type=str,
        default=None,
        help="Optional datasets.load_from_disk path for Psych-101.",
    )
    parser.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV for mixed_gambles datasets.",
    )
    parser.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        default=False,
        help="For mixed_gambles: keep gain_loss trials only.",
    )
    parser.add_argument(
        "--participant_scope",
        type=str,
        default="all",
        choices=["single", "range", "ordinals", "all"],
        help="Participant selection for population pooling (default: all).",
    )
    parser.add_argument(
        "--single_participant_id",
        type=int,
        default=0,
        help="Raw participant id when --participant_scope single.",
    )
    parser.add_argument(
        "--range_start_ordinal",
        type=int,
        default=None,
        help=(
            "Start ordinal (inclusive) when --participant_scope range. "
            "Per dataset, ordinals index that dataset's valid_participant_ids.json."
        ),
    )
    parser.add_argument(
        "--range_end_ordinal",
        type=int,
        default=None,
        help=(
            "End ordinal (inclusive) when --participant_scope range. "
            "Auto-clamped per dataset when it exceeds the valid list length."
        ),
    )
    parser.add_argument(
        "--all_max_participants",
        type=int,
        default=None,
        help="Cap participants when --participant_scope all.",
    )
    parser.add_argument(
        "--ordinals",
        nargs="+",
        type=int,
        default=None,
        metavar="I",
        help="0-based ordinals when --participant_scope ordinals.",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=10,
        help="Candidate programs per iteration (global and transfer).",
    )
    parser.add_argument(
        "--fresh_n_candidates",
        type=int,
        default=0,
        help="Fresh seed-only candidates per iteration (decays over iterations).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of parent programs sampled per child generation.",
    )
    parser.add_argument(
        "--sample_parents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly sample parents from elite pool (default: True).",
    )
    parser.add_argument(
        "--sampled_parents_decay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decay random parent sampling over iterations (default: True).",
    )
    parser.add_argument(
        "--elite_pool_size",
        type=int,
        default=None,
        metavar="N",
        help="Max elite pool size after each iteration.",
    )
    parser.add_argument(
        "--n_eval_seeds",
        type=int,
        default=3,
        help="Evaluation runs averaged per program fitness.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        help="LLM model for candidate generation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "local"],
        help="LLM mode: default=OpenAI API; local=vLLM server.",
    )
    parser.add_argument(
        "--llm_server_url",
        type=str,
        default=os.getenv("VLLM_LOCAL_URL", "http://localhost:8000/v1"),
        help="vLLM base URL when --mode local.",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.getenv("VLLM_LOCAL_API_KEY", "EMPTY"),
        help="API key for local vLLM.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.6,
        help="Train ratio for within-participant splits (default: 0.6).",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Split subsampling seed (default: 0).",
    )
    parser.add_argument(
        "--max_prompt_train_trials",
        type=int,
        default=40,
        help=(
            "Max trials in global-phase LLM prompts (sampled from train+val union; "
            "default: 40)."
        ),
    )
    parser.add_argument(
        "--max_prompt_trials_per_problem",
        type=int,
        default=5,
        help="Trials per prompt block (0 = flat sampling).",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=800,
        help="Max output tokens per LLM generation.",
    )
    parser.add_argument(
        "--hard_prompt_token_cap",
        type=int,
        default=14000,
        help="Hard input token budget per LLM prompt.",
    )
    parser.add_argument(
        "--strict_prompt_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise on prompt budget overflow (default: True).",
    )
    parser.add_argument(
        "--prompt_token_estimator",
        type=str,
        default="char4",
        choices=("char4",),
        help="Token estimator for prompt budgeting.",
    )
    parser.add_argument(
        "--max_parent_chars",
        type=int,
        default=4500,
        help="Max characters per parent program in prompts.",
    )
    parser.add_argument(
        "--max_error_prompt_chars",
        type=int,
        default=1200,
        help="Max chars for past-error section in prompts.",
    )
    parser.add_argument(
        "--warn_parent_truncation_ratio",
        type=float,
        default=0.5,
        help="Warn when many parents are truncated for prompts.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help=(
            "Parallel LLM workers. With --parallel_datasets (default), "
            "dataset_workers=max_workers//n_candidates and each dataset uses "
            "n_candidates candidate workers."
        ),
    )
    parser.add_argument(
        "--parallel_datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run datasets in parallel during global and transfer phases (default: True).",
    )
    parser.add_argument(
        "--early_stop_iters",
        type=int,
        default=-1,
        help="Early stopping patience (<=0 disables; default: -1).",
    )
    parser.add_argument(
        "--evolution_selection_score",
        type=str,
        default="train_val",
        choices=["train", "train_val"],
        help="Fitness for pool ranking (default: train_val).",
    )
    parser.add_argument(
        "--fitness_metric",
        type=str,
        default="loglik",
        choices=["loglik"],
        help="Fitness metric (loglik only for transfer).",
    )
    parser.add_argument(
        "--skip_global",
        action="store_true",
        help="Skip global phase (load existing global results from --output_dir).",
    )
    parser.add_argument(
        "--skip_transfer",
        action="store_true",
        help="Run global phase only; skip transfer.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> bool:
    if not (0.0 < args.split_ratio < 1.0):
        print(f"Error: --split_ratio must be in (0,1), got {args.split_ratio}.")
        return False
    if args.max_workers < 1:
        print("Error: --max_workers must be >= 1.")
        return False
    if args.n_candidates < 1:
        print("Error: --n_candidates must be >= 1.")
        return False
    if args.fresh_n_candidates < 0 or args.fresh_n_candidates > args.n_candidates:
        print("Error: --fresh_n_candidates must satisfy 0 <= fresh_n_candidates <= n_candidates.")
        return False
    if args.global_iters < 1:
        print("Error: --global_iters must be >= 1.")
        return False
    if args.transfer_iters < 1:
        print("Error: --transfer_iters must be >= 1.")
        return False
    if args.max_prompt_train_trials < 0:
        print("Error: --max_prompt_train_trials must be >= 0.")
        return False
    if args.transfer_max_prompt_trials < 0:
        print("Error: --transfer_max_prompt_trials must be >= 0.")
        return False
    if args.elite_pool_size is not None and args.elite_pool_size < 1:
        print("Error: --elite_pool_size must be >= 1 when set.")
        return False
    return True


def _resolve_participants(
    args: argparse.Namespace,
    dataset: str,
    psych_dataset_split: str,
    *,
    config_key: Optional[str] = None,
) -> List[int]:
    return resolve_participants_for_transfer(
        dataset=dataset,
        repo_root=_REPO_ROOT,
        participant_scope=args.participant_scope,
        single_participant_id=args.single_participant_id,
        range_start_ordinal=args.range_start_ordinal,
        range_end_ordinal=args.range_end_ordinal,
        all_max_participants=args.all_max_participants,
        participant_ordinals=args.ordinals,
        filter_mixed_gambles=bool(args.filter_mixed_gambles),
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        config_key=config_key,
    )


def _make_client(args: argparse.Namespace) -> OpenAI:
    if args.mode == "local":
        return OpenAI(api_key=args.llm_api_key, base_url=args.llm_server_url)
    return OpenAI()


def _shared_evolution_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "fresh_n_candidates": args.fresh_n_candidates,
        "sample_size": args.sample_size,
        "sample_parents": args.sample_parents,
        "sampled_parents_decay": args.sampled_parents_decay,
        "elite_pool_size": args.elite_pool_size,
        "model_name": args.model_name,
        "split_ratio": args.split_ratio,
        "split_seed": args.split_seed,
        "data_path": args.data_path,
        "filter_mixed_gambles": bool(args.filter_mixed_gambles),
        "max_prompt_trials_per_problem": args.max_prompt_trials_per_problem,
        "llm_max_tokens": args.llm_max_tokens,
        "n_eval_seeds": args.n_eval_seeds,
        "local_dataset": args.local_dataset,
        "mixed_gambles_csv": args.mixed_gambles_csv,
        "max_parent_chars": args.max_parent_chars,
        "warn_parent_truncation_ratio": args.warn_parent_truncation_ratio,
        "early_stop_iters": args.early_stop_iters,
        "hard_prompt_token_cap": args.hard_prompt_token_cap,
        "strict_prompt_budget": args.strict_prompt_budget,
        "prompt_token_estimator": args.prompt_token_estimator,
        "evolution_selection_score": args.evolution_selection_score,
        "max_error_prompt_chars": args.max_error_prompt_chars,
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not _validate_args(args):
        return

    config_path = Path(args.transfer_config)
    if not config_path.is_absolute():
        config_path = (_REPO_ROOT / config_path).resolve()
    try:
        specs = load_transfer_datasets(config_path)
        specs = filter_transfer_specs(specs, only=args.datasets)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    for spec in specs:
        if not is_binary_loglik_dataset(spec.dataset_alias):
            print(f"Error: unsupported dataset {spec.dataset_alias!r} in transfer config.")
            return
        alias = normalize_psych101_dataset_alias(spec.dataset_alias)
        if alias in PSYCH101_BINARY_DATASETS and not PSYCH101_BINARY_DATASETS[alias].get(
            "implemented"
        ):
            print(f"Error: parser not implemented for {spec.dataset_alias!r}.")
            return

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else transfer_output_run_dir(timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd_log = teh._write_command_line_log(run_dir)
    print(f"Saved command log -> {cmd_log}")
    (run_dir / "transfer_config.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "datasets": [
                    {
                        "config_key": s.config_key,
                        "dataset_alias": s.dataset_alias,
                        "psych_dataset_split": s.psych_dataset_split,
                    }
                    for s in specs
                ],
                "cli_args": vars(args),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(
                project="teh_transfer",
                name=f"teh_transfer_{timestamp}",
                config=vars(args),
                reinit=False,
            )
        except Exception as exc:
            print(f"wandb logging disabled: {exc}")

    client = _make_client(args)
    seed_program_path = Path(args.seed_path or DEFAULT_SEED_PROGRAM).expanduser()
    if not seed_program_path.is_absolute():
        seed_program_path = (_REPO_ROOT / seed_program_path).resolve()
    if not seed_program_path.is_file():
        print(f"Error: seed program not found: {seed_program_path}")
        return

    shared = _shared_evolution_kwargs(args)
    global_results: Dict[str, Any] = {}

    def _global_worker(spec, candidate_workers: int):
        participants = _resolve_participants(
            args,
            spec.dataset_alias,
            spec.psych_dataset_split,
            config_key=spec.config_key,
        )
        print(
            f"\n[global] {spec.config_key} ({spec.dataset_alias}, split={spec.psych_dataset_split}): "
            f"{len(participants)} participant(s)"
        )
        return run_dataset_global_phase(
            spec,
            run_dir=run_dir,
            client=client,
            participants=participants,
            seed_program_path=seed_program_path,
            n_iterations=args.global_iters,
            n_candidates=args.n_candidates,
            max_workers=candidate_workers,
            max_prompt_train_trials=args.max_prompt_train_trials,
            use_llm_prompt=not args.no_llm_prompt,
            base_prompt_path=args.base_prompt,
            debug_prompt=args.debug_prompt,
            **shared,
        )

    if args.skip_global:
        from utils.teh_transfer.evolution import (
            PopulationEvolutionResult,
            _read_phase_avg_test_loglik,
            _read_phase_best_loglik,
        )

        for spec in specs:
            dataset_output = run_dir / spec.config_key
            global_dir = dataset_output / "global"
            if not global_dir.is_dir():
                global_dir = dataset_output / "global_phase"
            if not global_dir.is_dir():
                print(f"Error: missing global results for {spec.config_key} under {run_dir}")
                return
            best_loglik, program_id, code = _read_phase_best_loglik(global_dir)
            prompts_dir = dataset_output / "prompts"
            global_results[spec.config_key] = PopulationEvolutionResult(
                dataset_alias=spec.dataset_alias,
                config_key=spec.config_key,
                psych_dataset_split=spec.psych_dataset_split,
                output_dir=dataset_output,
                prompts_dir=prompts_dir,
                best_loglik=best_loglik,
                best_program_code=code,
                best_program_id=program_id,
                best_test_loglik=_read_phase_avg_test_loglik(global_dir),
                participant_ids=_resolve_participants(
                    args,
                    spec.dataset_alias,
                    spec.psych_dataset_split,
                    config_key=spec.config_key,
                ),
            )
    else:
        global_results = run_global_phases_parallel(
            specs,
            _global_worker,
            max_workers=args.max_workers,
            n_candidates=args.n_candidates,
            parallel_datasets=args.parallel_datasets,
        )

    transfer_results: Dict[str, Any] = {}

    if not args.skip_transfer:

        def _transfer_worker(spec, existing_global: Dict[str, Any], candidate_workers: int):
            target = existing_global[spec.config_key]
            sources = build_source_contexts_for_target(
                spec.config_key,
                existing_global,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                data_path=args.data_path,
                filter_mixed_gambles=bool(args.filter_mixed_gambles),
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
                repo_root=_REPO_ROOT,
            )
            print(
                f"\n[transfer] target={spec.config_key}, sources="
                f"{[s.dataset_alias for s in sources]}"
            )
            seed_path = str(target.prompts_dir / "seed_program.py")
            return run_transfer_evolution_phase(
                target=target,
                sources=sources,
                client=client,
                seed_program_path=seed_path,
                n_iterations=args.transfer_iters,
                n_candidates=args.n_candidates,
                max_workers=candidate_workers,
                max_prompt_train_trials=args.transfer_max_prompt_trials,
                debug_prompt=args.debug_prompt,
                **shared,
            )

        transfer_results = run_transfer_phases_parallel(
            specs,
            global_results,
            _transfer_worker,
            max_workers=args.max_workers,
            n_candidates=args.n_candidates,
            parallel_datasets=args.parallel_datasets,
        )

    summary_rows: List[Dict[str, Any]] = []
    test_summary_rows: List[Dict[str, Any]] = []
    for spec in specs:
        g = global_results[spec.config_key]
        t = transfer_results.get(spec.config_key)
        row = {
            "dataset": spec.config_key,
            "global_best": f"{g.best_loglik:.6f}",
            "transfer": f"{t.best_loglik:.6f}" if t else "",
            "transfer_1st": (
                f"{t.first_iter_best_loglik:.6f}"
                if t and t.first_iter_best_loglik is not None
                else ""
            ),
        }
        summary_rows.append(row)
        test_summary_rows.append(
            {
                "dataset": spec.config_key,
                "global_test": (
                    f"{g.best_test_loglik:.6f}"
                    if g.best_test_loglik is not None
                    else ""
                ),
                "transfer_test": (
                    f"{t.best_test_loglik:.6f}"
                    if t and t.best_test_loglik is not None
                    else ""
                ),
            }
        )

    summary_dir = run_dir / "summary_csv"
    train_val_path = summary_dir / "train_val_loglik.csv"
    test_path = summary_dir / "test_loglik.csv"
    write_run_train_val_summary_csv(train_val_path, summary_rows)
    write_run_test_loglik_summary_csv(test_path, test_summary_rows)
    print(f"\nWrote train+val selection summary -> {train_val_path}")
    print(f"Wrote test loglik summary -> {test_path}")

    if args.debug_prompt:
        debug_dir = run_dir / "debug"
        for spec in specs:
            g = global_results[spec.config_key]
            t = transfer_results.get(spec.config_key)
            write_debug_prompts_file(
                str(debug_dir / f"prompt_{spec.config_key}.txt"),
                global_prompt=g.global_prompt_capture,
                transfer_prompt=t.transfer_prompt_capture if t else None,
            )
        print(f"Wrote debug prompts -> {debug_dir}")

    if wandb is not None:
        for row in summary_rows:
            wandb.log(
                {
                    f"summary/{row['dataset']}/global_best": float(row["global_best"])
                    if row["global_best"]
                    else None,
                    f"summary/{row['dataset']}/transfer": float(row["transfer"])
                    if row.get("transfer")
                    else None,
                    f"summary/{row['dataset']}/transfer_1st": float(row["transfer_1st"])
                    if row.get("transfer_1st")
                    else None,
                }
            )
        for test_row in test_summary_rows:
            wandb.log(
                {
                    f"summary/{test_row['dataset']}/global_test": float(
                        test_row["global_test"]
                    )
                    if test_row.get("global_test")
                    else None,
                    f"summary/{test_row['dataset']}/transfer_test": float(
                        test_row["transfer_test"]
                    )
                    if test_row.get("transfer_test")
                    else None,
                }
            )
        wandb.finish()

    print(f"\nTransfer run complete -> {run_dir}")


if __name__ == "__main__":
    main()
