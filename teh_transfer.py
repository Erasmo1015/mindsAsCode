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
    load_global_results_from_source_run,
    run_dataset_global_phase,
    run_global_phases_parallel,
    run_transfer_evolution_phase,
    transfer_output_run_dir,
    warn_new_run_differs_from_global_source,
    write_debug_prompts_file,
    write_transfer_summary_csv,
    write_run_train_val_summary_csv,
    write_run_test_loglik_summary_csv,
    backfill_run_test_loglik_summary_csv,
)
from utils.teh_transfer.transfer_jobs import (
    TransferJobResult,
    append_single_transfer_result_jsonl,
    build_transfer_job_batches,
    ensure_single_transfer_matrix_csvs,
    make_transfer_job_worker,
    run_transfer_jobs_parallel,
    update_single_transfer_matrix_cell,
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
        help=(
            "Run datasets in parallel during global phase and transfer jobs in "
            "parallel during transfer phase (default: True)."
        ),
    )
    parser.add_argument(
        "--transfer_mode",
        type=str,
        default="multiple",
        choices=["multiple", "single"],
        help=(
            "Transfer source layout: multiple=all other datasets per target (default); "
            "single=one source dataset per transfer job (N*(N-1) jobs)."
        ),
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
        "--global_run_dir",
        type=str,
        default=None,
        help=(
            "Path to a completed run whose global-phase best programs seed this run's "
            "transfer phase. Creates a new run directory (unless --output_dir is set); "
            "does not modify the source run."
        ),
    )
    parser.add_argument(
        "--skip_global",
        action="store_true",
        help=(
            "Skip global phase (load existing global results from --output_dir). "
            "Mutually exclusive with --global_run_dir."
        ),
    )
    parser.add_argument(
        "--skip_transfer",
        action="store_true",
        help="Run global phase only; skip transfer.",
    )
    parser.add_argument(
        "--backfill-test-loglik-summary",
        action="store_true",
        help=(
            "Rebuild summary_csv/test_loglik.csv (including transfer_1st) from an "
            "existing run directory passed via --output_dir."
        ),
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
    if args.global_iters < 1 and not (args.skip_global or args.global_run_dir):
        print("Error: --global_iters must be >= 1.")
        return False
    if args.skip_global and args.global_run_dir:
        print("Error: --skip_global and --global_run_dir are mutually exclusive.")
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
    if args.transfer_mode not in {"multiple", "single"}:
        print(f"Error: unsupported --transfer_mode {args.transfer_mode!r}.")
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
    if args.backfill_test_loglik_summary:
        if not args.output_dir:
            print("Error: --backfill-test-loglik-summary requires --output_dir.")
            return
        run_dir = Path(args.output_dir)
        if not run_dir.is_dir():
            print(f"Error: run directory not found: {run_dir}")
            return
        try:
            out_path = backfill_run_test_loglik_summary_csv(run_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"Error: backfill failed: {exc}")
            return
        print(f"Wrote backfilled test loglik summary -> {out_path}")
        return
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

    global_source_dir: Optional[Path] = None
    if args.global_run_dir:
        global_source_dir = Path(args.global_run_dir).expanduser()
        if not global_source_dir.is_absolute():
            global_source_dir = (_REPO_ROOT / global_source_dir).resolve()
        if not global_source_dir.is_dir():
            print(f"Error: global source run not found: {global_source_dir}")
            return

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = (_REPO_ROOT / run_dir).resolve()
    else:
        run_dir = transfer_output_run_dir(timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd_log = teh._write_command_line_log(run_dir)
    print(f"Saved command log -> {cmd_log}")

    dataset_entries = [
        {
            "config_key": s.config_key,
            "dataset_alias": s.dataset_alias,
            "psych_dataset_split": s.psych_dataset_split,
        }
        for s in specs
    ]
    transfer_config_payload: Dict[str, Any] = {
        "config_path": str(config_path),
        "datasets": dataset_entries,
        "cli_args": vars(args),
    }
    if global_source_dir is not None:
        transfer_config_payload["global_source_run_dir"] = str(global_source_dir)
        warn_new_run_differs_from_global_source(
            source_run_dir=global_source_dir,
            new_cli_args=vars(args),
            new_dataset_entries=dataset_entries,
            repo_root=_REPO_ROOT,
        )
    (run_dir / "transfer_config.json").write_text(
        json.dumps(transfer_config_payload, indent=2) + "\n",
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

    def _participants_for_spec(spec) -> List[int]:
        return _resolve_participants(
            args,
            spec.dataset_alias,
            spec.psych_dataset_split,
            config_key=spec.config_key,
        )

    if global_source_dir is not None:
        try:
            global_results = load_global_results_from_source_run(
                source_run_dir=global_source_dir,
                dest_run_dir=run_dir,
                specs=specs,
                resolve_participants_fn=_participants_for_spec,
                copy_prompts=True,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return
        print(
            f"\n[global source] Loaded {len(global_results)} dataset global result(s) "
            f"from {global_source_dir} into {run_dir}"
        )
    elif args.skip_global:
        try:
            global_results = load_global_results_from_source_run(
                source_run_dir=run_dir,
                dest_run_dir=run_dir,
                specs=specs,
                resolve_participants_fn=_participants_for_spec,
                copy_prompts=False,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return
    else:
        global_results = run_global_phases_parallel(
            specs,
            _global_worker,
            max_workers=args.max_workers,
            n_candidates=args.n_candidates,
            parallel_datasets=args.parallel_datasets,
        )

    transfer_results: Dict[str, Any] = {}
    transfer_job_results: Dict[str, TransferJobResult] = {}

    if not args.skip_transfer:
        dataset_keys = [spec.config_key for spec in specs]
        job_batches = build_transfer_job_batches(dataset_keys, args.transfer_mode)
        n_jobs = sum(len(batch) for batch in job_batches)
        print(
            f"\n[transfer] mode={args.transfer_mode}: {n_jobs} job(s) "
            f"in {len(job_batches)} batch(es)"
        )

        single_test_matrix_path: Optional[Path] = None
        single_improve_matrix_path: Optional[Path] = None
        single_results_jsonl = run_dir / "debug" / "single_source_transfer_results.jsonl"
        if args.transfer_mode == "single":
            summary_dir = run_dir / "summary_csv"
            single_test_matrix_path, single_improve_matrix_path = (
                ensure_single_transfer_matrix_csvs(summary_dir, dataset_keys)
            )
            single_results_jsonl.parent.mkdir(parents=True, exist_ok=True)
            if single_results_jsonl.is_file():
                single_results_jsonl.unlink()

        def _on_job_complete(job_result: TransferJobResult) -> None:
            if args.transfer_mode != "single":
                return
            append_single_transfer_result_jsonl(
                single_results_jsonl,
                job_result.to_jsonl_record(),
            )
            if (
                job_result.status != "ok"
                or job_result.result is None
                or single_test_matrix_path is None
                or single_improve_matrix_path is None
            ):
                return
            source_key = job_result.job.source_keys[0]
            target_key = job_result.job.target_key
            test_loglik = job_result.result.best_test_loglik
            update_single_transfer_matrix_cell(
                single_test_matrix_path,
                dataset_keys,
                source_key=source_key,
                target_key=target_key,
                value=test_loglik,
            )
            global_test = global_results[target_key].best_test_loglik
            improve = None
            if test_loglik is not None and global_test is not None:
                improve = float(test_loglik) - float(global_test)
            update_single_transfer_matrix_cell(
                single_improve_matrix_path,
                dataset_keys,
                source_key=source_key,
                target_key=target_key,
                value=improve,
            )

        transfer_worker = make_transfer_job_worker(
            global_results=global_results,
            run_transfer_evolution_phase=run_transfer_evolution_phase,
            build_source_contexts_for_target=build_source_contexts_for_target,
            client=client,
            evolution_kwargs=shared,
            transfer_iters=args.transfer_iters,
            n_candidates=args.n_candidates,
            transfer_max_prompt_trials=args.transfer_max_prompt_trials,
            debug_prompt=args.debug_prompt,
            repo_root=_REPO_ROOT,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            data_path=args.data_path,
            filter_mixed_gambles=bool(args.filter_mixed_gambles),
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
        )
        transfer_job_results = run_transfer_jobs_parallel(
            job_batches,
            transfer_worker,
            max_workers=args.max_workers,
            n_candidates=args.n_candidates,
            parallel_jobs=args.parallel_datasets,
            on_job_complete=_on_job_complete,
            batch_desc=f"Transfer ({args.transfer_mode})",
        )

        if args.transfer_mode == "multiple":
            for job_id, job_result in transfer_job_results.items():
                if job_result.result is not None:
                    transfer_results[job_id] = job_result.result

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
                "transfer": (
                    f"{t.best_test_loglik:.6f}"
                    if t and t.best_test_loglik is not None
                    else ""
                ),
                "transfer_1st": (
                    f"{t.first_iter_best_test_loglik:.6f}"
                    if t and t.first_iter_best_test_loglik is not None
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
    if args.transfer_mode == "single" and not args.skip_transfer:
        single_dir = summary_dir / "single_transfer"
        print(f"Wrote single-source test loglik matrix -> {single_dir / 'test_loglik.csv'}")
        print(f"Wrote single-source improve matrix -> {single_dir / 'improve_test_loglik.csv'}")
        print(f"Wrote single-source job log -> {run_dir / 'debug' / 'single_source_transfer_results.jsonl'}")

    if args.debug_prompt and args.transfer_mode == "multiple":
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
                        test_row["transfer"]
                    )
                    if test_row.get("transfer")
                    else None,
                    f"summary/{test_row['dataset']}/transfer_1st_test": float(
                        test_row["transfer_1st"]
                    )
                    if test_row.get("transfer_1st")
                    else None,
                }
            )
        wandb.finish()

    print(f"\nTransfer run complete -> {run_dir}")


if __name__ == "__main__":
    main()
