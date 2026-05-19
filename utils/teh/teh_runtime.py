"""
TEH run setup: prompts, output paths, WandB naming, valid participant id paths.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    experiment_id_for_alias,
    experiment_to_trial_dicts,
    format_trials_for_prompt,
    get_psych101_binary_experiment,
    normalize_psych101_dataset_alias,
    summarize_runtime_schema_for_prompt,
)
from utils.teh.teh_datasets import (
    dataset_display_name,
    is_mixed_gambles_dataset,
    teh_output_base_dir as _teh_output_base_dir,
    valid_participant_ids_path_with_filter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEH_WANDB_PROJECT = "teh"

BASE_LOGlik_PROMPT = (
    REPO_ROOT
    / "prompts"
    / "Template_evo"
    / "choice13k"
    / "non_strict"
    / "loglik"
    / "infer_single_choice.txt"
)
BASE_REFINE_PROMPT = (
    REPO_ROOT / "prompts" / "Template_evo" / "choice13k" / "refine" / "infer_single_choice.txt"
)
DEFAULT_SEED_PROGRAM = REPO_ROOT / "persona_code_example" / "te_vanilla" / "choices13k.py"

CONCISE_PROGRAM_GUIDANCE = (
    "Prefer concise programs. Avoid long repetitive helper code. "
    "Keep choose() compact, ideally under ~150 lines unless necessary."
)


def valid_participant_ids_path(
    dataset_alias: str,
    repo_root: Optional[Path] = None,
    *,
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = "train",
) -> Path:
    root = repo_root or REPO_ROOT
    return valid_participant_ids_path_with_filter(
        dataset_alias,
        root,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
    )


def teh_output_base_dir(
    dataset_alias: str,
    timestamp: str,
    *,
    psych_dataset_split: str = "train",
) -> str:
    return _teh_output_base_dir(
        dataset_alias, timestamp, psych_dataset_split=psych_dataset_split
    )


def teh_wandb_run_name(
    dataset_alias: str,
    timestamp: str,
    participant_scope: str,
    *,
    psych_dataset_split: str = "train",
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
    ordinals: Optional[Sequence[int]] = None,
) -> str:
    base = f"{dataset_alias}_teh_{psych_dataset_split}_{timestamp}"
    if participant_scope == "range" and range_start is not None and range_end is not None:
        return f"{base}_ordinals_{range_start}_to_{range_end}"
    if participant_scope == "ordinals" and ordinals:
        tag = "_".join(str(x) for x in ordinals)
        if len(tag) > 120:
            tag = tag[:120] + "_etc"
        return f"{base}_ordinals_{tag}"
    if participant_scope == "all":
        return f"{base}_all_valid"
    return base


def _format_trials_for_prompt(trials: List[Dict[str, Any]], max_trials: int = 8) -> str:
    """Schema-aware one-line summaries (gamble, CCT, weather, product, tree, bandit, …)."""
    return format_trials_for_prompt(trials, max_trials=max_trials)


def _runtime_schema_summary_for_prompt(trials: List[Dict[str, Any]]) -> str:
    return summarize_runtime_schema_for_prompt(trials)


def _merge_prompt_fallback(
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
) -> str:
    base = BASE_LOGlik_PROMPT.read_text(encoding="utf-8")
    if is_mixed_gambles_dataset(dataset_alias):
        display = dataset_display_name(dataset_alias)
        task_desc = instruction
    else:
        alias = normalize_psych101_dataset_alias(dataset_alias)
        spec = PSYCH101_BINARY_DATASETS[alias]
        display = spec["display_name"]
        task_desc = spec["task_description"]
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    extra = (
        f"\n\n## Dataset: {display} (`{dataset_alias}`)\n\n"
        f"{task_desc}\n\n"
        f"### Runtime schema summary (from parsed trials)\n\n{schema_summary}\n\n"
        f"### Task instructions (from Psych-101 transcript)\n\n{instruction[:1500]}\n\n"
        f"### Example parsed trials\n\n{_format_trials_for_prompt(sample_trials)}\n"
    )
    return base + extra


def _generate_prompt_via_llm(
    client: OpenAI,
    model_name: str,
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    *,
    max_tokens: int = 2048,
) -> str:
    base_prompt = BASE_LOGlik_PROMPT.read_text(encoding="utf-8")
    display = dataset_display_name(dataset_alias)
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    trial_examples = _format_trials_for_prompt(sample_trials, max_trials=10)
    if is_mixed_gambles_dataset(dataset_alias):
        task_description = instruction
    else:
        task_description = PSYCH101_BINARY_DATASETS[
            normalize_psych101_dataset_alias(dataset_alias)
        ]["task_description"]

    user_content = (
        f"Adapt the following evolution system prompt for dataset `{dataset_alias}` "
        f"({display}).\n\n"
        "Requirements:\n"
        "- Use the **Runtime schema summary** and **Parsed trial examples** sections below "
        "as the source of truth for the `problem` dict fields, `history` structure, and "
        "action semantics. Do not invent fields that are absent from those sections.\n"
        "- The base prompt is a shared template written for Choice13k (two-option gambles). "
        "Adapt it for this dataset:\n"
        "  - If this is NOT a gamble A/B task (runtime summary says is_gamble_A/B_task: False), "
        "remove all gamble-specific wording (gamble_A, gamble_B, lottery probabilities, "
        "'Option A/B' as gambles, etc.).\n"
        "  - Do NOT mention gamble_A or gamble_B unless they appear in the runtime schema summary.\n"
        "  - Preserve safety/API requirements from the base prompt that still apply: pure Python, "
        "no imports, deterministic, return a single float in (0, 1), clip to [1e-6, 1-1e-6], "
        "no randomness, no pow() (use **), helpers nested inside choose(), variables defined on "
        "all branches, no division by zero.\n"
        "  - Drop base-prompt task-specific Choice13k behavioral wording that does not apply "
        "(e.g. unknown gamble probs) when the schema summary shows a different task.\n"
        "- Executable API (required): `def choose(problem, history)` returning float in (0, 1) "
        "as P(action=1), where action=1 means the SECOND entry in option_keys (index 1).\n"
        "- Describe action=0 and action=1 using dataset-specific semantics from the schema "
        "summary and examples (participant-specific press keys), not generic 'Option B' labels.\n"
        "- Document `problem` keys using the exact names from the examples (e.g. round_id, "
        "current_score, cards_flipped for CCT — not renamed aliases).\n"
        f"- {CONCISE_PROGRAM_GUIDANCE}\n"
        "- Output ONLY the full prompt text (no markdown code fence).\n\n"
        f"## Runtime schema summary\n\n{schema_summary}\n\n"
        f"## Base prompt (shared template — adapt, do not copy gamble bias blindly)\n\n"
        f"{base_prompt}\n\n"
        f"## Task description (high-level)\n\n{task_description}\n\n"
        f"## Instruction excerpt from data\n\n{instruction[:2000]}\n\n"
        f"## Parsed trial examples\n\n{trial_examples}\n"
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You write prompts for program-evolution systems. Be precise and preserve APIs.",
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text or "def choose" not in text:
        raise ValueError("LLM prompt generation did not return a valid choose() prompt.")
    return text


def setup_teh_run_prompts(
    run_dir: Path,
    dataset_alias: str,
    seed_program_path: Path,
    *,
    client: Optional[OpenAI] = None,
    model_name: str = "gpt-4o-mini",
    use_llm: bool = True,
    n_sample_participants: int = 1,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = "train",
) -> Path:
    """
    Create run_dir/prompts/ with infer_single_choice.txt (generated), templates, refine, seed.

    Returns path to prompts directory.
    """
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    if is_mixed_gambles_dataset(dataset_alias):
        from utils.teh.participant_ids import load_valid_participant_ids

        valid_ids = load_valid_participant_ids(
            dataset_alias,
            REPO_ROOT,
            filter_mixed_gambles=filter_mixed_gambles,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        if not valid_ids:
            raise ValueError(f"No valid participant ids for mixed_gambles dataset {dataset_alias!r}")
        sample_pid = int(valid_ids[0])
        train_trials, _, _, _ = load_mixed_gambles_trials(
            sample_pid,
            csv_path=mixed_gambles_csv,
            filter_gain_loss_only=filter_mixed_gambles,
        )
        sample_trial_list = train_trials[:8]
        instruction = (
            "Mixed gambles: Option A is a 50/50 gamble (gain/loss); Option B is certain. "
            "action=0 gamble, action=1 certain; choose(problem, history) returns P(action=1)."
        )
    else:
        exp = get_psych101_binary_experiment(
            dataset_alias,
            0,
            split=psych_dataset_split,
            local_dataset=local_dataset,
        )
        sample_trial_list = experiment_to_trial_dicts(
            exp,
            dataset_alias=dataset_alias,
            experiment_id=experiment_id_for_alias(dataset_alias),
        )
        instruction = exp.instruction

    infer_path = prompts_dir / "infer_single_choice.txt"
    generated = False
    if use_llm and client is not None:
        try:
            infer_text = _generate_prompt_via_llm(
                client,
                model_name,
                dataset_alias,
                instruction,
                sample_trial_list,
            )
            infer_path.write_text(infer_text, encoding="utf-8")
            generated = True
            print(f"[TEH] Wrote LLM-generated prompt -> {infer_path}")
        except Exception as e:
            print(f"[TEH] LLM prompt generation failed ({e}); using merge fallback.")

    if not generated:
        infer_path.write_text(
            _merge_prompt_fallback(dataset_alias, instruction, sample_trial_list),
            encoding="utf-8",
        )
        print(f"[TEH] Wrote merged fallback prompt -> {infer_path}")

    shutil.copy2(BASE_REFINE_PROMPT, prompts_dir / "refine.txt")
    seed_src = seed_program_path.expanduser().resolve()
    if not seed_src.is_file():
        raise FileNotFoundError(f"Seed program not found: {seed_src}")
    shutil.copy2(seed_src, prompts_dir / "seed_program.py")

    meta: Dict[str, Any] = {
        "dataset_alias": dataset_alias,
        "llm_generated": generated,
        "seed_program_source": str(seed_src),
    }
    if is_mixed_gambles_dataset(dataset_alias):
        meta["mixed_gambles_csv"] = mixed_gambles_csv
        meta["filter_mixed_gambles"] = filter_mixed_gambles
    else:
        meta["experiment_id"] = experiment_id_for_alias(dataset_alias)
    (prompts_dir / "prompt_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return prompts_dir
