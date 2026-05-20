#!/usr/bin/env python3
"""
Generate TEH prompts + one cheap candidate per dataset (prompt/code debug only).

Example:
  python analysis/code/generate_prompt.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# --- config (edit here) ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
USE_LLM = True
PSYCH_DATASET_SPLIT = "train"
PARTICIPANT_ID = 0
MAX_PROMPT_TRAIN_TRIALS = 5
CANDIDATE_MAX_TOKENS = 800
# Parallel datasets (LLM-bound; threads share one vLLM server). 0 = auto.
MAX_DATASET_WORKERS = 8
# ------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from data_modules.psych101_binary import get_psych101_binary_experiment
from utils.teh.prompt_debug_candidate import (
    generate_one_debug_candidate,
    load_debug_train_trials,
    write_leakage_report_file,
)
from utils.teh.teh_datasets import PARTICIPANT_DATASETS, is_mixed_gambles_dataset
from utils.teh.teh_runtime import (
    DEFAULT_SEED_PROGRAM,
    _runtime_schema_summary_for_prompt,
    build_prompt_generation_llm_user_content,
    setup_teh_run_prompts,
)

VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")

_SEPARATOR = "=" * 50
_SECTION = "-" * 18

LLM_INPUT_FALLBACK_MSG = "(LLM not used; fallback prompt generated)\n"
LLM_INPUT_FAILED_MSG = (
    "(LLM prompt generation failed; merge fallback used — reconstructed input below)\n\n"
)
CANDIDATE_SKIPPED_MSG = "(candidate generation skipped: USE_LLM=False or no API client)\n"


def _read_file(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dataset_instruction(dataset_alias: str, psych_dataset_split: str) -> str:
    if is_mixed_gambles_dataset(dataset_alias):
        return (
            "Mixed gambles: option_0 is a 50/50 risky option (gain/loss); option_1 is certain. "
            "action=0 selects option_0, action=1 selects option_1; choose returns P(action=1)."
        )
    exp = get_psych101_binary_experiment(
        dataset_alias, PARTICIPANT_ID, split=psych_dataset_split
    )
    return exp.instruction


def _ensure_llm_input_prompt(
    dataset_dir: Path,
    dataset_alias: str,
    *,
    use_llm: bool,
    psych_dataset_split: str,
    sample_trials: list,
) -> None:
    path = dataset_dir / "prompts" / "llm_input_prompt.txt"
    if path.is_file() and path.stat().st_size > 0:
        return
    if not use_llm:
        _write_text(path, LLM_INPUT_FALLBACK_MSG)
        return
    meta_path = dataset_dir / "prompts" / "prompt_meta.json"
    llm_generated = False
    if meta_path.is_file():
        llm_generated = bool(
            json.loads(meta_path.read_text(encoding="utf-8")).get("llm_generated", False)
        )
    if llm_generated:
        _write_text(path, "(LLM marked success but llm_input_prompt.txt was not saved)\n")
        return
    try:
        instruction = _dataset_instruction(dataset_alias, psych_dataset_split)
        trials = sample_trials or load_debug_train_trials(
            dataset_alias,
            participant_id=PARTICIPANT_ID,
            psych_dataset_split=psych_dataset_split,
            max_trials=MAX_PROMPT_TRAIN_TRIALS,
        )
        reconstructed = build_prompt_generation_llm_user_content(
            dataset_alias, instruction, trials[:10]
        )
        _write_text(path, LLM_INPUT_FAILED_MSG + reconstructed)
    except Exception as exc:
        _write_text(
            path,
            f"{LLM_INPUT_FALLBACK_MSG.strip()}\n(reconstruction error: {exc})\n",
        )


def _ensure_schema_summary(
    dataset_dir: Path,
    dataset_alias: str,
    psych_dataset_split: str,
) -> tuple[str, list]:
    path = dataset_dir / "prompts" / "runtime_schema_summary.txt"
    try:
        trials = load_debug_train_trials(
            dataset_alias,
            participant_id=PARTICIPANT_ID,
            psych_dataset_split=psych_dataset_split,
            max_trials=MAX_PROMPT_TRAIN_TRIALS,
        )
        summary = _runtime_schema_summary_for_prompt(trials)
        _write_text(path, summary + "\n")
        return summary, trials
    except Exception as exc:
        summary = f"(failed to build schema summary: {exc})\n"
        _write_text(path, summary)
        return summary, []


def _init_candidate_placeholders(cand_dir: Path, message: str) -> None:
    cand_dir.mkdir(parents=True, exist_ok=True)
    _write_text(cand_dir / "raw_candidate.txt", message)
    _write_text(cand_dir / "cleaned_candidate.py", "")
    _write_text(cand_dir / "execution_log.txt", message)


def _run_one_candidate(
    client: OpenAI,
    dataset_dir: Path,
    dataset_alias: str,
    *,
    schema_summary: str,
    trials: list,
    psych_dataset_split: str,
) -> str | None:
    """Run one cheap candidate; always leaves files under generated_candidate/."""
    cand_dir = dataset_dir / "generated_candidate"
    if client is None:
        _init_candidate_placeholders(cand_dir, CANDIDATE_SKIPPED_MSG)
        return "skipped (no client)"
    try:
        print(f"[prompt_debug]   candidate LLM call for {dataset_alias} ...")
        result = generate_one_debug_candidate(
            client,
            MODEL_NAME,
            dataset_alias,
            dataset_dir,
            participant_id=PARTICIPANT_ID,
            psych_dataset_split=psych_dataset_split,
            max_train_trials=MAX_PROMPT_TRAIN_TRIALS,
            max_tokens=CANDIDATE_MAX_TOKENS,
            schema_summary=schema_summary,
        )
        if result.get("error"):
            return "candidate LLM/API error (see execution_log.txt)"
        return None
    except Exception:
        err = traceback.format_exc()
        _init_candidate_placeholders(
            cand_dir,
            f"(candidate generation failed)\n\n{err}\n",
        )
        return err


def _write_leakage_if_missing(
    dataset_dir: Path,
    dataset_alias: str,
    schema_summary: str,
    trials: list,
) -> None:
    leak_path = dataset_dir / "generated_candidate" / "leakage_report.txt"
    if leak_path.is_file() and leak_path.stat().st_size > 0:
        return
    write_leakage_report_file(
        dataset_dir,
        dataset_alias,
        trials,
        schema_summary,
    )


def _write_summary(
    dataset_dir: Path,
    dataset_alias: str,
    *,
    candidate_ran: bool,
    candidate_note: str | None,
) -> None:
    meta_path = dataset_dir / "prompts" / "prompt_meta.json"
    llm_generated = False
    if meta_path.is_file():
        llm_generated = bool(
            json.loads(meta_path.read_text(encoding="utf-8")).get("llm_generated", False)
        )
    prompt_source = "LLM" if llm_generated else "merge fallback"
    lines = [
        f"dataset_alias: {dataset_alias}",
        "mode: prompt-debug (prompt generation + one candidate)",
        "evolution: not run",
        "evaluation: not run (smoke call on 1 trial only)",
        f"llm_generated: {llm_generated}",
        f"prompt_source: {prompt_source}",
        f"candidate_generation: {'attempted' if candidate_ran else 'skipped'}",
    ]
    if candidate_note:
        lines.append(f"candidate_note: {candidate_note}")
    (dataset_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_debug_block(dataset_dir: Path, dataset_alias: str) -> list[str]:
    prompts_dir = dataset_dir / "prompts"
    cand_dir = dataset_dir / "generated_candidate"

    def section(title: str, rel_path: Path) -> list[str]:
        content = _read_file(rel_path)
        if not content:
            content = f"(missing: {rel_path.name} not found at {rel_path})\n"
        return [
            _SECTION,
            f" {title} ",
            _SECTION,
            "",
            content,
            "" if content.endswith("\n") else "\n",
        ]

    blocks: list[str] = [
        _SEPARATOR,
        f"DATASET: {dataset_alias}",
        _SEPARATOR,
        "",
        *section("RUNTIME SCHEMA SUMMARY", prompts_dir / "runtime_schema_summary.txt"),
        *section("LLM INPUT PROMPT", prompts_dir / "llm_input_prompt.txt"),
        *section("GENERATED FINAL PROMPT", prompts_dir / "infer_single_choice.txt"),
        *section("RAW GENERATED CANDIDATE", cand_dir / "raw_candidate.txt"),
        *section("CLEANED GENERATED CANDIDATE", cand_dir / "cleaned_candidate.py"),
        *section("EXECUTION / VALIDATION LOG", cand_dir / "execution_log.txt"),
        *section("LEAKAGE CHECK", cand_dir / "leakage_report.txt"),
    ]
    error_path = dataset_dir / "error.txt"
    if error_path.is_file():
        blocks.extend(
            [
                _SECTION,
                " ERRORS ",
                _SECTION,
                "",
                _read_file(error_path),
                "",
            ]
        )
    blocks.append("")
    return blocks


def _make_vllm_client() -> OpenAI:
    return OpenAI(
        base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        api_key="EMPTY",
    )


def _dataset_worker_count() -> int:
    n = len(PARTICIPANT_DATASETS)
    if MAX_DATASET_WORKERS > 0:
        return min(MAX_DATASET_WORKERS, n)
    return min(n, os.cpu_count() or 4)


def _process_one_dataset(
    dataset_alias: str,
    output_root: Path,
    *,
    use_llm: bool,
    psych_dataset_split: str,
) -> tuple[str, bool]:
    """Process a single dataset; returns (alias, success)."""
    dataset_dir = output_root / dataset_alias
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prompt_debug] {dataset_alias} ...")
    candidate_note: str | None = None
    schema_summary = ""
    trials: list = []
    client: OpenAI | None = _make_vllm_client() if use_llm else None
    try:
        setup_teh_run_prompts(
            dataset_dir,
            dataset_alias,
            DEFAULT_SEED_PROGRAM,
            client=client,
            model_name=MODEL_NAME,
            use_llm=use_llm,
            psych_dataset_split=psych_dataset_split,
        )

        schema_summary, trials = _ensure_schema_summary(
            dataset_dir, dataset_alias, psych_dataset_split
        )
        _ensure_llm_input_prompt(
            dataset_dir,
            dataset_alias,
            use_llm=use_llm,
            psych_dataset_split=psych_dataset_split,
            sample_trials=trials,
        )

        if use_llm:
            candidate_note = _run_one_candidate(
                client,
                dataset_dir,
                dataset_alias,
                schema_summary=schema_summary,
                trials=trials,
                psych_dataset_split=psych_dataset_split,
            )
        else:
            _init_candidate_placeholders(
                dataset_dir / "generated_candidate", CANDIDATE_SKIPPED_MSG
            )
            candidate_note = "skipped (USE_LLM=False)"

        _write_leakage_if_missing(
            dataset_dir, dataset_alias, schema_summary, trials
        )
        _write_summary(
            dataset_dir,
            dataset_alias,
            candidate_ran=use_llm,
            candidate_note=candidate_note,
        )
        print(f"[prompt_debug] {dataset_alias} done")
        return dataset_alias, True
    except Exception:
        err = traceback.format_exc()
        (dataset_dir / "error.txt").write_text(err, encoding="utf-8")
        _write_text(
            dataset_dir / "prompts" / "runtime_schema_summary.txt",
            f"(dataset setup failed)\n{err}\n",
        )
        _write_text(
            dataset_dir / "prompts" / "llm_input_prompt.txt",
            f"(setup failed)\n{err}\n",
        )
        _init_candidate_placeholders(
            dataset_dir / "generated_candidate",
            f"(setup failed)\n{err}\n",
        )
        print(f"[prompt_debug] {dataset_alias} FAILED (see error.txt)")
        return dataset_alias, False


def _write_all_debug(output_root: Path) -> Path:
    out_path = output_root / "ALL_DEBUG.txt"
    blocks: list[str] = [
        "TEH prompt + candidate debug — combined export",
        f"run_dir: {output_root}",
        "",
    ]
    for dataset_alias in sorted(PARTICIPANT_DATASETS):
        blocks.extend(_dataset_debug_block(output_root / dataset_alias, dataset_alias))
    _write_text(out_path, "\n".join(blocks))
    return out_path


def main() -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_root = REPO_ROOT / "generated_outputs" / "prompt_debug" / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    if USE_LLM:
        print(f"[prompt_debug] vLLM -> http://{VLLM_HOST}:{VLLM_PORT}/v1")

    workers = _dataset_worker_count()
    datasets = sorted(PARTICIPANT_DATASETS)
    print(
        f"[prompt_debug] processing {len(datasets)} datasets "
        f"with {workers} parallel worker(s)"
    )

    succeeded: list[str] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_one_dataset,
                dataset_alias,
                output_root,
                use_llm=USE_LLM,
                psych_dataset_split=PSYCH_DATASET_SPLIT,
            ): dataset_alias
            for dataset_alias in datasets
        }
        for fut in as_completed(futures):
            alias, ok = fut.result()
            if ok:
                succeeded.append(alias)
            else:
                failed.append(alias)

    succeeded.sort()
    failed.sort()

    all_debug_path = _write_all_debug(output_root)
    print(f"combined: {all_debug_path}")

    print(f"\noutput_root: {output_root}")
    print(f"succeeded ({len(succeeded)}): {', '.join(succeeded) if succeeded else '(none)'}")
    print(f"failed ({len(failed)}): {', '.join(failed) if failed else '(none)'}")


if __name__ == "__main__":
    main()
