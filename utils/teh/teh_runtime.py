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
    get_psych101_binary_experiments,
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
    lines: List[str] = []
    for idx, t in enumerate(trials[:max_trials]):
        p = t["problem"]
        ga = p.get("gamble_A", {})
        gb = p.get("gamble_B", {})
        lines.append(
            f"{idx + 1}. gamble_A probs={ga.get('probs')} rewards={ga.get('rewards')}; "
            f"gamble_B probs={gb.get('probs')} rewards={gb.get('rewards')}; "
            f"option_keys={p.get('option_keys')}; has_feedback={p.get('has_feedback')}; "
            f"action={t['action']}; history_len={len(t.get('history', []))}"
        )
    return "\n".join(lines)


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
        spec = PSYCH101_BINARY_DATASETS[dataset_alias]
        display = spec["display_name"]
        task_desc = spec["task_description"]
    extra = (
        f"\n\n## Dataset: {display} (`{dataset_alias}`)\n\n"
        f"{task_desc}\n\n"
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
    user_content = (
        f"Adapt the following evolution system prompt for dataset `{dataset_alias}` "
        f"({display}).\n\n"
        "Requirements:\n"
        "- Preserve the executable API: def choose(problem, history) returning a float "
        "in (0,1) interpreted as P(action=1) where action=1 is the SECOND option in option_keys.\n"
        "- Keep all safety/requirement bullets from the base prompt that still apply.\n"
        "- Describe the problem dict fields accurately for this task.\n"
        "- Output ONLY the full prompt text (no markdown code fence).\n\n"
        f"## Base prompt\n\n{base_prompt}\n\n"
        f"## Task description\n\n"
        f"{PSYCH101_BINARY_DATASETS[dataset_alias]['task_description'] if not is_mixed_gambles_dataset(dataset_alias) else instruction}\n\n"
        f"## Instruction excerpt from data\n\n{instruction[:2000]}\n\n"
        f"## Parsed trial examples\n\n{_format_trials_for_prompt(sample_trials, max_trials=10)}\n"
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
        experiments = get_psych101_binary_experiments(
            dataset_alias,
            n_participants=max(1, n_sample_participants),
            split=psych_dataset_split,
            local_dataset=local_dataset,
        )
        exp = experiments[0]
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
