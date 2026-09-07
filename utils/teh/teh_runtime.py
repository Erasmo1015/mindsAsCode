"""
TEH run setup: prompts, output paths, WandB naming, valid participant id paths.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.external import (
    is_bergert_nosofsky_2007_dataset,
    is_guan_2020_stopping_dataset,
    is_steyvers_2009_bandit_dataset,
    is_external_dataset,
    load_external_loglik_trials,
    external_default_data_dir,
)
from data_modules.external.bergert_nosofsky_2007 import (
    TASK_DESCRIPTION as BERGERT_TASK_DESCRIPTION,
)
from data_modules.external.guan_2020_stopping import (
    TASK_DESCRIPTION as GUAN_TASK_DESCRIPTION,
)
from data_modules.external.steyvers_2009_bandit import (
    TASK_DESCRIPTION as STEYVERS_TASK_DESCRIPTION,
)
from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    experiment_id_for_alias,
    experiment_to_trial_dicts,
    format_trials_for_prompt,
    get_psych101_binary_experiment,
    normalize_psych101_dataset_alias,
    summarize_runtime_schema_for_prompt,
)
from utils.teh.prompt_sanitize import strip_embedded_choose_from_evolution_prompt
from utils.teh.teh_datasets import (
    dataset_display_name,
    is_categorical_output_dataset,
    is_mixed_gambles_dataset,
    teh_output_base_dir as _teh_output_base_dir,
    valid_participant_ids_path_with_filter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEH_WANDB_PROJECT = "teh"

DEFAULT_BASE_LOGlik_PROMPT = REPO_ROOT / "prompts" / "teh" / "infer_single_choice.txt"
BASE_LOGlik_PROMPT = DEFAULT_BASE_LOGlik_PROMPT


def resolve_base_loglik_prompt_path(base_prompt_path: Optional[Path | str] = None) -> Path:
    """Resolve base loglik prompt path (default: prompts/teh/infer_single_choice.txt)."""
    if base_prompt_path is None:
        return DEFAULT_BASE_LOGlik_PROMPT
    p = Path(base_prompt_path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


BASE_REFINE_PROMPT = (
    REPO_ROOT / "prompts" / "Template_evo" / "choice13k" / "refine" / "infer_single_choice.txt"
)
DEFAULT_SEED_PROGRAM = REPO_ROOT / "persona_code_example" / "te_vanilla" / "choices13k.py"

CONCISE_PROGRAM_GUIDANCE = (
    "Prefer concise programs. Avoid long repetitive helper code. "
    "Keep choose() compact, ideally under ~150 lines unless necessary."
)

_GENERIC_PROMPT_REQUIREMENTS = """Requirements:
- Pure Python, no imports, deterministic.
- Use only the provided problem and history.
- Do not call external APIs.
- Do not sample or use randomness.
- Return a single finite float probability of choosing action 1, i.e. P(action=1).
- The return value must be strictly inside (0, 1). If needed, clip to a safe range such as [1e-6, 1 - 1e-6].
- Higher returned values mean a higher P(action=1) (more likely to choose action 1).
- Avoid numerical errors such as division by zero, overflow, or invalid operations.
- Do not use `pow(...)`; use `**` for exponentiation.
- If using a logistic/sigmoid transform without imports, use:
  1 / (1 + 2.718281828 ** (-x))
  Do not use incorrect forms such as 1 / (1 + 1 / (1 + x)).
- Keep all helper logic used by `choose(...)` inside `choose(...)` (nested functions are allowed).
- Do not rely on top-level helper functions outside `choose(...)`.
- Ensure every variable used in expressions is defined on all branches."""

_GAMBLE_LEAK_RE = re.compile(
    r'problem\["gamble_[AB]"\]|problem\[\'gamble_[AB]\'\]|'
    r"gamble_A:|gamble_B:|- gamble_A|gamble_A/|/gamble_B|"
    r"unknown probabilit|\blottery\b|two gambles|two-option gamble",
    re.IGNORECASE,
)


def _is_gamble_ab_task(trials: List[Dict[str, Any]]) -> bool:
    """True when parsed trials include gamble_A/gamble_B problem fields."""
    for trial in trials:
        problem = trial.get("problem") or {}
        if "gamble_A" in problem or "gamble_B" in problem:
            return True
    return False


def _extract_action_semantics(schema_summary: str) -> str:
    for line in schema_summary.splitlines():
        if line.startswith("- action semantics:"):
            return line.split(":", 1)[1].strip()
    return "action=0 is first option; action=1 is second; return P(action=1)."


def _problem_keys_from_trials(trials: List[Dict[str, Any]]) -> List[str]:
    keys: set = set()
    for trial in trials:
        for key in (trial.get("problem") or {}):
            if key not in ("dataset_alias", "experiment_id"):
                keys.add(key)
    return sorted(keys)


def _build_schema_neutral_base_prompt(
    schema_summary: str,
    trials: List[Dict[str, Any]],
    *,
    categorical: bool = False,
) -> str:
    """Non-gamble base prompt: document observed problem/history keys only."""
    problem_keys = _problem_keys_from_trials(trials)
    action_sem = _extract_action_semantics(schema_summary)
    history_note = "(none observed)"
    for line in schema_summary.splitlines():
        if line.startswith("- history core keys:"):
            history_note = line.split(":", 1)[1].strip()
            break
    problem_doc = (
        "\n".join(f"        - {key}: (type/structure per parsed examples)" for key in problem_keys)
        or "        - (see parsed trial examples)"
    )
    if categorical:
        intro = (
            "You are given observations of human choices in multi-action decision problems.\n"
        )
        return_line = (
            "    return: dict[int, float] probabilities over every option['action'] "
            "in problem['options']\n"
        )
        requirements = """Requirements:
- Pure Python, no imports, deterministic.
- Use only the provided problem and history.
- Do not call external APIs.
- Do not sample or use randomness.
- Return a dict[int, float] over all action ids in problem["options"].
- Probabilities must be finite and non-negative (prefer summing to 1.0).
- Avoid numerical errors such as division by zero, overflow, or invalid operations.
- Do not use `pow(...)`; use `**` for exponentiation.
- Keep all helper logic used by `choose(...)` inside `choose(...)`.
"""
        behavioral = (
            "Behavioral requirements:\n"
            "- Do not return a near-uniform constant unless history truly provides no signal.\n"
            "- The distribution must depend meaningfully on the problem and/or history.\n"
            "- Respect history reset boundaries documented for this dataset.\n"
            "- The program should behave sensibly across different problems and histories.\n\n"
        )
    else:
        intro = (
            "You are given observations of human choices in binary decision problems.\n"
        )
        return_line = "    return: float, probability of choosing action 1, i.e. P(action=1)\n"
        requirements = f"{_GENERIC_PROMPT_REQUIREMENTS}\n"
        behavioral = (
            "Behavioral requirements:\n"
            "- Do not return constant or near-constant probabilities, such as always close to 0.5.\n"
            "- The probability must depend meaningfully on the problem inputs.\n"
            "- History may be used when helpful, but do not rely only on copying past actions.\n"
            "- The program should behave sensibly across different problems and histories.\n\n"
        )
    return (
        f"{intro}"
        "Each trial provides a `problem` dict and a `history` list. Use only fields "
        "present in the parsed data for this dataset.\n\n"
        "Write Python code that reproduces the observed behavior. You must generate "
        "a program implementing:\n\n"
        "def choose(problem, history):\n"
        '    """\n'
        "    Evaluated by log-likelihood.\n"
        "    Goal: maximize log-likelihood of the observed human choices.\n\n"
        "    problem: dict with keys (observed for this dataset):\n"
        f"{problem_doc}\n"
        f"    history: list of dicts (keys observed: {history_note})\n"
        f"    Action semantics: {action_sem}\n"
        f"{return_line}"
        '    """\n\n'
        f"{requirements}\n"
        f"{behavioral}"
        "Generation requirements:\n"
        f"- {CONCISE_PROGRAM_GUIDANCE}\n"
        "- Programs must implement choose(problem, history) as documented above.\n"
    )


def _prompt_has_gamble_leakage(text: str) -> bool:
    return bool(_GAMBLE_LEAK_RE.search(text))


def _sanitize_schema_summary_for_prompt(
    schema_summary: str, *, is_gamble: bool
) -> str:
    """Remove gamble_A/B wording from schema text embedded in non-gamble prompts."""
    if is_gamble:
        return schema_summary
    replacements = [
        ("- is_gamble_A/B_task: False", "- has_gamble_option_fields: False"),
        ("- is_gamble_A/B_task: True", "- has_gamble_option_fields: True"),
        (
            "- not a gamble task: do NOT document gamble_A/gamble_B (absent from examples).",
            "- task type: non-gamble; document only observed problem keys.",
        ),
        (
            "- gamble tasks: problem includes gamble_A/gamble_B dicts with probs/rewards; "
            "probs may be None for unknown probabilities.",
            "- two-option task with per-option outcome fields in `problem`.",
        ),
    ]
    out = schema_summary
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def _base_prompt_for_trials(
    trials: List[Dict[str, Any]],
    schema_summary: str,
    *,
    dataset_alias: str,
    base_prompt_path: Optional[Path | str] = None,
) -> str:
    if _is_gamble_ab_task(trials):
        return _choice13k_neutral_loglik_base(base_prompt_path)
    return _build_schema_neutral_base_prompt(
        schema_summary,
        trials,
        categorical=is_categorical_output_dataset(dataset_alias),
    )


def _apply_gamble_neutral_wording(text: str) -> str:
    """
    Use index-based option semantics in prompts (action 0/1).

    Keeps problem['gamble_A'] / problem['gamble_B'] dict keys unchanged; avoids
    Option A/B or Option P/U narrative that can be confused with gamble_A/gamble_B.
    """
    out = re.sub(
        r"Each problem presents two gambles: Option [A-Z] and Option [A-Z]\.",
        (
            "Each problem presents two options (option index 0 and option index 1). "
            "problem[\"gamble_A\"] stores option 0; problem[\"gamble_B\"] stores option 1. "
            "These are parsed schema field names and do not necessarily mean the task is a gamble problem."
        ),
        text,
        count=1,
    )
    out = re.sub(
        r"- action: int \(0 for [A-Z], 1 for [A-Z]\)",
        "- action: int (0 = first option / gamble_A; 1 = second option / gamble_B)",
        out,
        count=1,
    )
    out = re.sub(
        r"return: float, probability of choosing option 1 \(Option [A-Z]\)",
        (
            "return: float, P(action=1) — probability of choosing action 1 "
            "(option index 1; second gamble, gamble_B)"
        ),
        out,
        count=1,
    )
    out = re.sub(
        r"Return a single finite float probability of choosing Option [A-Z]\.",
        "Return a single finite float P(action=1): probability of choosing action 1 (second option).",
        out,
        count=1,
    )
    out = re.sub(
        r"Higher returned values mean the participant is more likely to choose Option [A-Z]\.",
        "Higher returned values mean a higher P(action=1) (more likely to choose action 1).",
        out,
        count=1,
    )
    out = re.sub(
        r'option_keys: e\.g\., \["[A-Z]"[,\s]*"[A-Z]"\]',
        (
            "option_keys: list of two press-key labels (e.g. [\"P\",\"U\"]); "
            "index 0/1 selects first/second gamble — do not match letters to gamble_A/gamble_B"
        ),
        out,
        count=1,
    )
    return out


def _choice13k_neutral_loglik_base(base_prompt_path: Optional[Path | str] = None) -> str:
    """Template loglik prompt with index-based gamble wording (matches evaluation)."""
    path = resolve_base_loglik_prompt_path(base_prompt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Base prompt not found: {path}")
    return _apply_gamble_neutral_wording(path.read_text(encoding="utf-8"))


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
    ablation: Optional[str] = None,
) -> str:
    return _teh_output_base_dir(
        dataset_alias,
        timestamp,
        psych_dataset_split=psych_dataset_split,
        ablation=ablation,
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
    raw = summarize_runtime_schema_for_prompt(trials)
    return _sanitize_schema_summary_for_prompt(
        raw, is_gamble=_is_gamble_ab_task(trials)
    )


def _merge_prompt_fallback(
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    *,
    base_prompt_path: Optional[Path | str] = None,
) -> str:
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    base = _base_prompt_for_trials(
        sample_trials,
        schema_summary,
        dataset_alias=dataset_alias,
        base_prompt_path=base_prompt_path,
    )
    if is_mixed_gambles_dataset(dataset_alias) or is_external_dataset(dataset_alias):
        display = dataset_display_name(dataset_alias)
        task_desc = instruction
    else:
        alias = normalize_psych101_dataset_alias(dataset_alias)
        spec = PSYCH101_BINARY_DATASETS[alias]
        display = spec["display_name"]
        task_desc = spec["task_description"]
    extra = (
        f"\n\n## Dataset: {display} (`{dataset_alias}`)\n\n"
        f"{task_desc}\n\n"
        f"### Runtime schema summary (from parsed trials)\n\n{schema_summary}\n\n"
        f"### Task instructions (from Psych-101 transcript)\n\n{instruction[:1500]}\n\n"
        f"### Example parsed trials\n\n{_format_trials_for_prompt(sample_trials)}\n"
    )
    return base + extra


def build_prompt_generation_llm_user_content(
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    *,
    base_prompt_path: Optional[Path | str] = None,
) -> str:
    """User message sent to the prompt-generation LLM (no API call)."""
    display = dataset_display_name(dataset_alias)
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    is_gamble = _is_gamble_ab_task(sample_trials)
    base_prompt = _base_prompt_for_trials(
        sample_trials,
        schema_summary,
        dataset_alias=dataset_alias,
        base_prompt_path=base_prompt_path,
    )
    trial_examples = _format_trials_for_prompt(sample_trials, max_trials=10)
    if is_mixed_gambles_dataset(dataset_alias) or is_external_dataset(dataset_alias):
        task_description = instruction
    else:
        task_description = PSYCH101_BINARY_DATASETS[
            normalize_psych101_dataset_alias(dataset_alias)
        ]["task_description"]

    if is_gamble:
        adapt_instructions = (
            "- The base prompt is for gamble_A/gamble_B tasks. Adapt wording only as needed "
            "for this dataset; keep gamble field names and P(action=1)=P(action on gamble_B).\n"
        )
        base_section_title = "## Base prompt (gamble_A/gamble_B task)\n\n"
        api_line = (
            "- Executable API: `def choose(problem, history)` returning float in (0, 1) as P(action=1).\n"
        )
        safety_line = (
            "- Preserve generic safety: pure Python, no imports, deterministic, clip to "
            "[1e-6, 1-1e-6], no randomness, no pow() (use **), helpers inside choose(), "
            "variables defined on all branches.\n"
        )
    elif is_categorical_output_dataset(dataset_alias):
        adapt_instructions = (
            "- The base prompt skeleton is schema-specific (categorical multi-action). Expand it "
            "into a complete evolution prompt using ONLY fields from the runtime schema summary.\n"
            "- Do NOT mention gamble_A, gamble_B, or Bernoulli P(action=1).\n"
            "- Document exact `problem` keys, history reset boundaries, and that choose() must "
            "return a probability distribution over all action ids in problem['options'].\n"
        )
        base_section_title = "## Base prompt skeleton (categorical)\n\n"
        api_line = (
            "- Executable API: `def choose(problem, history)` returning dict[int, float] "
            "probabilities over every option['action'] (renormalized if needed).\n"
        )
        safety_line = (
            "- Preserve generic safety: pure Python, no imports, deterministic, non-negative "
            "finite probabilities, no randomness, no pow() (use **), helpers inside choose(), "
            "variables defined on all branches.\n"
        )
    else:
        adapt_instructions = (
            "- The base prompt skeleton is schema-specific (non-gamble). Expand it into a "
            "complete evolution prompt using ONLY fields from the runtime schema summary.\n"
            "- Do NOT mention gamble_A, gamble_B, unknown probabilities, lottery, or Choice13k.\n"
            "- Document exact `problem` keys and action semantics from the schema summary.\n"
        )
        base_section_title = "## Base prompt skeleton (schema-specific)\n\n"
        api_line = (
            "- Executable API: `def choose(problem, history)` returning float in (0, 1) as P(action=1).\n"
        )
        safety_line = (
            "- Preserve generic safety: pure Python, no imports, deterministic, clip to "
            "[1e-6, 1-1e-6], no randomness, no pow() (use **), helpers inside choose(), "
            "variables defined on all branches.\n"
        )

    return (
        f"Write the evolution system prompt for dataset `{dataset_alias}` ({display}).\n\n"
        "Requirements:\n"
        "- Use the **Runtime schema summary** and **Parsed trial examples** sections below "
        "as the source of truth for `problem`, `history`, and action semantics.\n"
        f"{adapt_instructions}"
        f"{api_line}"
        f"{safety_line}"
        f"- {CONCISE_PROGRAM_GUIDANCE}\n"
        "- Output ONLY the evolution instruction prompt text (no markdown code fence).\n"
        "- Do NOT append a sample, reference, or complete choose() implementation.\n"
        "- You may document the choose() API in a short docstring block, but no executable code.\n\n"
        f"## Runtime schema summary\n\n{schema_summary}\n\n"
        f"{base_section_title}{base_prompt}\n\n"
        f"## Task description (high-level)\n\n{task_description}\n\n"
        f"## Instruction excerpt from data\n\n{instruction[:2000]}\n\n"
        f"## Parsed trial examples\n\n{trial_examples}\n"
    )


def _generate_prompt_via_llm(
    client: OpenAI,
    model_name: str,
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    *,
    max_tokens: int = 2048,
    save_llm_input_to: Optional[Path] = None,
    base_prompt_path: Optional[Path | str] = None,
) -> str:
    user_content = build_prompt_generation_llm_user_content(
        dataset_alias,
        instruction,
        sample_trials,
        base_prompt_path=base_prompt_path,
    )
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    is_gamble = _is_gamble_ab_task(sample_trials)
    if save_llm_input_to is not None:
        save_llm_input_to.parent.mkdir(parents=True, exist_ok=True)
        save_llm_input_to.write_text(user_content, encoding="utf-8")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write evolution instruction prompts (not Python solutions). "
                    "Be precise and preserve APIs."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("LLM prompt generation returned empty text.")
    text = strip_embedded_choose_from_evolution_prompt(text)
    if is_gamble:
        text = _apply_gamble_neutral_wording(text)
    elif _prompt_has_gamble_leakage(text):
        print(
            "[TEH] LLM prompt contained gamble-specific text for a non-gamble schema; "
            "using schema-neutral base prompt."
        )
        text = _build_schema_neutral_base_prompt(schema_summary, sample_trials)
    text = strip_embedded_choose_from_evolution_prompt(text)
    if not text:
        raise ValueError("LLM prompt was empty after removing embedded choose() code.")
    return text


def setup_teh_run_prompts(
    run_dir: Path,
    dataset_alias: str,
    seed_program_path: Path,
    *,
    client: Optional[OpenAI] = None,
    model_name: str = "gpt-4o-mini",
    use_llm: bool = True,
    base_prompt_path: Optional[Path | str] = None,
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
    resolved_base_prompt = resolve_base_loglik_prompt_path(base_prompt_path)
    if not resolved_base_prompt.is_file():
        raise FileNotFoundError(f"Base prompt not found: {resolved_base_prompt}")

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
    elif is_bergert_nosofsky_2007_dataset(dataset_alias):
        from utils.teh.participant_ids import load_valid_participant_ids

        valid_ids = load_valid_participant_ids(dataset_alias, REPO_ROOT)
        if not valid_ids:
            raise ValueError(f"No valid participant ids for dataset {dataset_alias!r}")
        sample_pid = int(valid_ids[0])
        train_trials, _, _, _ = load_external_loglik_trials(
            dataset_alias,
            sample_pid,
            data_dir=str(REPO_ROOT / external_default_data_dir(dataset_alias)),
        )
        sample_trial_list = train_trials[:8]
        instruction = BERGERT_TASK_DESCRIPTION
    elif is_guan_2020_stopping_dataset(dataset_alias):
        from utils.teh.participant_ids import load_valid_participant_ids

        valid_ids = load_valid_participant_ids(dataset_alias, REPO_ROOT)
        if not valid_ids:
            raise ValueError(f"No valid participant ids for dataset {dataset_alias!r}")
        sample_pid = int(valid_ids[0])
        train_trials, _, _, _ = load_external_loglik_trials(
            dataset_alias,
            sample_pid,
            data_dir=str(REPO_ROOT / external_default_data_dir(dataset_alias)),
        )
        sample_trial_list = train_trials[:8]
        instruction = GUAN_TASK_DESCRIPTION
    elif is_steyvers_2009_bandit_dataset(dataset_alias):
        from utils.teh.participant_ids import load_valid_participant_ids

        valid_ids = load_valid_participant_ids(dataset_alias, REPO_ROOT)
        if not valid_ids:
            raise ValueError(f"No valid participant ids for dataset {dataset_alias!r}")
        sample_pid = int(valid_ids[0])
        train_trials, _, _, _ = load_external_loglik_trials(
            dataset_alias,
            sample_pid,
            data_dir=str(REPO_ROOT / external_default_data_dir(dataset_alias)),
        )
        sample_trial_list = train_trials[:8]
        instruction = STEYVERS_TASK_DESCRIPTION
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
                save_llm_input_to=prompts_dir / "llm_input_prompt.txt",
                base_prompt_path=resolved_base_prompt,
            )
            infer_path.write_text(
                strip_embedded_choose_from_evolution_prompt(infer_text), encoding="utf-8"
            )
            generated = True
            print(f"[TEH] Wrote LLM-generated prompt -> {infer_path}")
        except Exception as e:
            print(f"[TEH] LLM prompt generation failed ({e}); using merge fallback.")

    if not generated:
        merged = _merge_prompt_fallback(
            dataset_alias,
            instruction,
            sample_trial_list,
            base_prompt_path=resolved_base_prompt,
        )
        if _is_gamble_ab_task(sample_trial_list):
            merged = _apply_gamble_neutral_wording(merged)
        merged = strip_embedded_choose_from_evolution_prompt(merged)
        infer_path.write_text(merged, encoding="utf-8")
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
        "base_prompt_path": str(resolved_base_prompt),
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