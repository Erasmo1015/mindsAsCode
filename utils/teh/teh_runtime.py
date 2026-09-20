"""
TEH run setup: prompts, output paths, WandB naming, valid participant id paths.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from data_modules.mixed_gambles import DEFAULT_CSV_PATH
from data_modules.external import (
    is_bergert_nosofsky_2007_dataset,
    is_guan_2020_stopping_dataset,
    is_steyvers_2009_bandit_dataset,
    is_external_dataset,
    external_default_data_dir,
    external_reference_prompt_path,
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
    format_trials_for_prompt,
    get_psych101_binary_experiment,
    normalize_psych101_dataset_alias,
)
from utils.teh.prompt_context import (
    DEFAULT_EXAMPLE_CHAR_BUDGET,
    DEFAULT_HISTORY_MAX_ENTRIES,
    DEFAULT_MAX_EXAMPLES,
    RUNTIME_CONTRACT_FILENAME,
    assert_no_generation_oracle_leak,
    attach_runtime_contract_to_prompt,
    build_deterministic_runtime_contract,
    history_keys_note_from_schema,
    infer_recursive_runtime_schema,
    problem_keys_note_from_schema,
    select_diverse_train_trials,
    serialize_train_trials_for_prompt_generation,
)
from utils.teh.limited_data_protocol import load_participant_limited_splits
from utils.teh.limited_data_registry import (
    LIMITED_DATA_PROTOCOL_OFF,
    normalize_limited_data_protocol,
)
from utils.teh.dataset_prompt_evolution import ensure_task_knowledge_in_prompt
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


def resolve_dataset_reference_prompt_path(dataset_alias: str) -> Optional[Path]:
    """
    Hand-written evolution prompt for a dataset, if registered.

    External datasets use EXTERNAL_DATASET_META['reference_prompt'].
    Psych-101 extensions (Schulz / Kool) use PSYCH101_BINARY_DATASETS['reference_prompt'].
    """
    rel: Optional[str] = None
    if is_external_dataset(dataset_alias):
        rel = external_reference_prompt_path(dataset_alias)
    else:
        try:
            alias = normalize_psych101_dataset_alias(dataset_alias)
            rel = PSYCH101_BINARY_DATASETS.get(alias, {}).get("reference_prompt")
        except Exception:
            rel = None
    if not rel:
        return None
    path = Path(str(rel)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    return path if path.is_file() else None


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
- Ensure every variable used in expressions is defined on all branches.
- History entries do not copy all problem keys. Use .get() for sometimes-present fields such as feedback.
- history[i]["action"] is an integer action id, never a press-key letter from option_keys (option_keys is problem-only).
- Guard every division (empty counts / missing feedback). Wrap dict.keys() in list() before indexing."""

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
        if line.startswith("- action / option_keys note:"):
            return line.split(":", 1)[1].strip()
    return "action=0 is first option; action=1 is second; return P(action=1)."


def _problem_keys_from_trials(trials: List[Dict[str, Any]]) -> List[str]:
    keys: set = set()
    for trial in trials:
        for key in (trial.get("problem") or {}):
            if key not in ("dataset_alias", "experiment_id"):
                keys.add(key)
    return sorted(keys)


def _task_description_for_prompt_gen(
    dataset_alias: str, instruction: str
) -> str:
    """High-level task description used in auto prompt-gen input (and retained in baseline)."""
    if is_mixed_gambles_dataset(dataset_alias) or is_external_dataset(dataset_alias):
        return instruction
    try:
        return PSYCH101_BINARY_DATASETS[
            normalize_psych101_dataset_alias(dataset_alias)
        ]["task_description"]
    except Exception:
        return instruction


def _build_schema_neutral_base_prompt(
    schema_summary: str,
    trials: List[Dict[str, Any]],
    *,
    categorical: bool = False,
) -> str:
    """Non-gamble base prompt: API/safety skeleton with concrete problem/history keys."""
    action_sem = _extract_action_semantics(schema_summary)
    history_note = history_keys_note_from_schema(schema_summary)
    problem_doc = problem_keys_note_from_schema(schema_summary)
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
- Guard every division (Laplace +1 or max(den, 1e-9)). Wrap dict.keys() in list() before indexing.
- History entries do not copy all problem keys; use .get() for sometimes-present fields.
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
    raw = infer_recursive_runtime_schema(trials)
    return _sanitize_schema_summary_for_prompt(
        raw, is_gamble=_is_gamble_ab_task(trials)
    )


def _merge_prompt_fallback(
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    *,
    base_prompt_path: Optional[Path | str] = None,
    example_char_budget: int = DEFAULT_EXAMPLE_CHAR_BUDGET,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
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
    trial_examples = serialize_train_trials_for_prompt_generation(
        sample_trials,
        char_budget=example_char_budget,
        history_max_entries=history_max_entries,
        max_examples=max_examples,
    )
    extra = (
        f"\n\n## Dataset: {display} (`{dataset_alias}`)\n\n"
        f"{task_desc}\n\n"
        f"### Runtime schema summary (from parsed trials)\n\n{schema_summary}\n\n"
        f"### Task instructions (from Psych-101 transcript)\n\n{instruction[:1500]}\n\n"
        f"### Example parsed trials\n\n{trial_examples}\n"
    )
    merged = base + extra
    return ensure_task_knowledge_in_prompt(
        merged,
        task_description=task_desc,
        instruction_excerpt=instruction,
    )


def build_prompt_generation_llm_user_content(
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    *,
    base_prompt_path: Optional[Path | str] = None,
    example_char_budget: int = DEFAULT_EXAMPLE_CHAR_BUDGET,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
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
    trial_examples = serialize_train_trials_for_prompt_generation(
        sample_trials,
        char_budget=example_char_budget,
        history_max_entries=history_max_entries,
        max_examples=max_examples,
    )
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
            "variables defined on all branches. Runtime builtins include "
            "set/frozenset/sorted/any/all/ord/next/map/filter/round.\n"
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
            "- Preserve generic safety: pure Python, no imports, deterministic, "
            "non-negative finite probabilities, no randomness, no pow() (use **), "
            "helpers inside choose(), variables defined on all branches. Runtime "
            "builtins include set/frozenset/sorted/any/all/ord/next/map/filter/round.\n"
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
            "variables defined on all branches. Runtime builtins include "
            "set/frozenset/sorted/any/all/ord/next/map/filter/round.\n"
        )

    return (
        f"Write the evolution system prompt for dataset `{dataset_alias}` ({display}).\n\n"
        "Requirements:\n"
        "- Use the **Runtime schema summary** and **Parsed trial examples** sections below "
        "as the source of truth for `problem`, `history`, and action semantics.\n"
        f"{adapt_instructions}"
        f"{api_line}"
        f"{safety_line}"
        "- Retain the **Task description** and **Instruction excerpt** content in the output "
        "prompt (task/domain facts must appear in the evolution prompt itself, not only here).\n"
        "- Prefer concrete modelling cues from the task description (e.g. cue validities, "
        "payoffs, stage structure) over vague behavioural platitudes.\n"
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
    example_char_budget: int = DEFAULT_EXAMPLE_CHAR_BUDGET,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    llm_decoding_seed: Optional[int] = None,
) -> str:
    user_content = build_prompt_generation_llm_user_content(
        dataset_alias,
        instruction,
        sample_trials,
        base_prompt_path=base_prompt_path,
        example_char_budget=example_char_budget,
        history_max_entries=history_max_entries,
        max_examples=max_examples,
    )
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    is_gamble = _is_gamble_ab_task(sample_trials)
    if save_llm_input_to is not None:
        save_llm_input_to.parent.mkdir(parents=True, exist_ok=True)
        save_llm_input_to.write_text(user_content, encoding="utf-8")
    create_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write evolution instruction prompts (not Python solutions). "
                    "Be precise and preserve APIs."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if llm_decoding_seed is not None:
        create_kwargs["seed"] = int(llm_decoding_seed)
    resp = client.chat.completions.create(**create_kwargs)
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
    text = ensure_task_knowledge_in_prompt(
        text,
        task_description=_task_description_for_prompt_gen(dataset_alias, instruction),
        instruction_excerpt=instruction,
    )
    return text


def _prompt_trial_fingerprint(trial: Dict[str, Any]) -> str:
    """Instance fingerprint for prompt-example leakage checks (problem + history)."""
    from utils.teh.prompt_snapshots import sanitize_problem_for_choose

    return json.dumps(
        {
            "problem": sanitize_problem_for_choose(trial.get("problem") or {}),
            "history": trial.get("history"),
        },
        sort_keys=True,
        default=str,
    )


def _prompt_sample_pid_and_instruction(
    dataset_alias: str,
    *,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = "train",
) -> Tuple[int, str]:
    """Dataset-level instruction plus the participant used for prompt examples."""
    from utils.teh.participant_ids import load_valid_participant_ids

    if is_mixed_gambles_dataset(dataset_alias):
        valid_ids = load_valid_participant_ids(
            dataset_alias,
            REPO_ROOT,
            filter_mixed_gambles=filter_mixed_gambles,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        if not valid_ids:
            raise ValueError(f"No valid participant ids for mixed_gambles dataset {dataset_alias!r}")
        instruction = (
            "Mixed gambles: Option A is a 50/50 gamble (gain/loss); Option B is certain. "
            "action=0 gamble, action=1 certain; choose(problem, history) returns P(action=1)."
        )
        return int(valid_ids[0]), instruction
    if is_bergert_nosofsky_2007_dataset(dataset_alias):
        valid_ids = load_valid_participant_ids(dataset_alias, REPO_ROOT)
        if not valid_ids:
            raise ValueError(f"No valid participant ids for dataset {dataset_alias!r}")
        return int(valid_ids[0]), BERGERT_TASK_DESCRIPTION
    if is_guan_2020_stopping_dataset(dataset_alias):
        valid_ids = load_valid_participant_ids(dataset_alias, REPO_ROOT)
        if not valid_ids:
            raise ValueError(f"No valid participant ids for dataset {dataset_alias!r}")
        return int(valid_ids[0]), GUAN_TASK_DESCRIPTION
    if is_steyvers_2009_bandit_dataset(dataset_alias):
        valid_ids = load_valid_participant_ids(dataset_alias, REPO_ROOT)
        if not valid_ids:
            raise ValueError(f"No valid participant ids for dataset {dataset_alias!r}")
        return int(valid_ids[0]), STEYVERS_TASK_DESCRIPTION
    exp = get_psych101_binary_experiment(
        dataset_alias,
        0,
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    return 0, exp.instruction


def _load_prompt_observation_trials(
    dataset_alias: str,
    sample_pid: int,
    *,
    split_ratio: float,
    split_seed: int,
    filter_mixed_gambles: bool = False,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    psych_dataset_split: str = "train",
    local_dataset: Optional[str] = None,
    speekenbrink_split: str = "chronological",
    data_dir: Optional[str] = None,
    limited_data_protocol: object = LIMITED_DATA_PROTOCOL_OFF,
    limited_train_val: Optional[int] = None,
    max_observed_trials_per_participant: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any]:
    """Observed train+val used for prompt examples.

    When the limited-data protocol is on, this is the exact retained subset
    used for scoring (same manifest/fingerprints). Test is never included.
    """
    resolved = data_dir
    if resolved is None and is_external_dataset(dataset_alias):
        resolved = str(REPO_ROOT / external_default_data_dir(dataset_alias))
    train, val, test, _audit, manifest = load_participant_limited_splits(
        dataset_alias,
        int(sample_pid),
        split_ratio=float(split_ratio),
        split_seed=int(split_seed),
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
        data_dir=resolved,
    )
    observed = list(train) + list(val)
    return observed, list(test), manifest


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
    example_char_budget: int = DEFAULT_EXAMPLE_CHAR_BUDGET,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    prefer_auto_llm_prompt: bool = False,
    dataset_prompt_file: Optional[Path | str] = None,
    split_ratio: float = 0.6,
    split_seed: int = 0,
    speekenbrink_split: str = "chronological",
    data_dir: Optional[str] = None,
    limited_data_protocol: object = LIMITED_DATA_PROTOCOL_OFF,
    limited_train_val: Optional[int] = None,
    max_observed_trials_per_participant: Optional[int] = None,
    require_auto_llm_prompt: bool = False,
    llm_decoding_seed: Optional[int] = None,
) -> Path:
    """
    Create run_dir/prompts/ with infer_single_choice.txt (generated), templates, refine, seed.

    Returns path to prompts directory.

    When prefer_auto_llm_prompt is True (T-PICS / dataset-prompt evolution), skip
    hand-written reference prompts so the run starts from the auto LLM prompt.

    When dataset_prompt_file is set, copy that file as infer_single_choice.txt (skips
    reference / LLM / merge). Used for hand-designed comparison prompts.

    When limited_data_protocol is enabled, parsed behavioral prompt examples are
    taken only from the retained limited-data train+val subset (same manifest as
    scoring). Task descriptions and schema notes remain dataset-level metadata.
    """
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    infer_path = prompts_dir / "infer_single_choice.txt"
    meta_path = prompts_dir / "prompt_meta.json"
    if require_auto_llm_prompt and (infer_path.exists() or meta_path.exists()):
        if not (infer_path.is_file() and meta_path.is_file()):
            raise RuntimeError(
                f"T-PICS gated found incomplete prompt artifacts under {prompts_dir}; "
                "refusing to silently regenerate a different target prompt."
            )
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"T-PICS gated prompt_meta.json is unreadable ({meta_path}): {exc}"
            ) from exc
        existing_text = infer_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(existing_text.encode("utf-8")).hexdigest()
        seed_copy = prompts_dir / "seed_program.py"
        if not (
            isinstance(existing_meta, dict)
            and existing_meta.get("prompt_mode") == "auto_llm"
            and existing_meta.get("llm_generated") is True
            and bool(existing_text.strip())
            and existing_meta.get("infer_prompt_sha256") == digest
            and seed_copy.is_file()
        ):
            raise RuntimeError(
                f"T-PICS gated found stale or invalid prompt artifacts under {prompts_dir}; "
                "refusing to silently regenerate a different target prompt."
            )
        print(f"[TEH] Reusing gated automatic target prompt -> {infer_path}")
        return prompts_dir

    resolved_base_prompt = resolve_base_loglik_prompt_path(base_prompt_path)
    if not resolved_base_prompt.is_file():
        raise FileNotFoundError(f"Base prompt not found: {resolved_base_prompt}")

    sample_pid, instruction = _prompt_sample_pid_and_instruction(
        dataset_alias,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
    )
    sample_trial_list, _test_trials, limited_manifest = _load_prompt_observation_trials(
        dataset_alias,
        sample_pid,
        split_ratio=float(split_ratio),
        split_seed=int(split_seed),
        filter_mixed_gambles=filter_mixed_gambles,
        mixed_gambles_csv=mixed_gambles_csv,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        speekenbrink_split=speekenbrink_split,
        data_dir=data_dir,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
    )

    infer_path = prompts_dir / "infer_single_choice.txt"
    generated = False
    used_reference = False
    used_dataset_prompt_file = False
    forced_prompt: Optional[Path] = None
    if dataset_prompt_file is not None:
        forced_prompt = Path(str(dataset_prompt_file)).expanduser()
        if not forced_prompt.is_absolute():
            forced_prompt = (REPO_ROOT / forced_prompt).resolve()
        else:
            forced_prompt = forced_prompt.resolve()
        if not forced_prompt.is_file():
            raise FileNotFoundError(f"dataset_prompt_file not found: {forced_prompt}")
        text = strip_embedded_choose_from_evolution_prompt(
            forced_prompt.read_text(encoding="utf-8")
        )
        infer_path.write_text(text, encoding="utf-8")
        used_dataset_prompt_file = True
        print(f"[TEH] Wrote dataset_prompt_file -> {infer_path}")
        print(f"[TEH]   source: {forced_prompt}")

    reference_prompt = resolve_dataset_reference_prompt_path(dataset_alias)
    if prefer_auto_llm_prompt or used_dataset_prompt_file or require_auto_llm_prompt:
        reference_prompt = None
    if not used_dataset_prompt_file and reference_prompt is not None:
        text = strip_embedded_choose_from_evolution_prompt(
            reference_prompt.read_text(encoding="utf-8")
        )
        infer_path.write_text(text, encoding="utf-8")
        used_reference = True
        print(f"[TEH] Wrote hand-written reference prompt -> {infer_path}")
        print(f"[TEH]   source: {reference_prompt}")
    elif not used_dataset_prompt_file and use_llm and client is not None:
        try:
            infer_text = _generate_prompt_via_llm(
                client,
                model_name,
                dataset_alias,
                instruction,
                sample_trial_list,
                save_llm_input_to=prompts_dir / "llm_input_prompt.txt",
                base_prompt_path=resolved_base_prompt,
                example_char_budget=example_char_budget,
                history_max_entries=history_max_entries,
                max_examples=max_examples,
                llm_decoding_seed=llm_decoding_seed,
            )
            infer_path.write_text(
                strip_embedded_choose_from_evolution_prompt(infer_text), encoding="utf-8"
            )
            generated = True
            print(f"[TEH] Wrote LLM-generated prompt -> {infer_path}")
        except Exception as e:
            if require_auto_llm_prompt:
                raise RuntimeError(
                    f"Automatic target-prompt generation failed for {dataset_alias!r}: {e}"
                ) from e
            print(f"[TEH] LLM prompt generation failed ({e}); using merge fallback.")

    if not generated and not used_reference and not used_dataset_prompt_file:
        if require_auto_llm_prompt:
            raise RuntimeError(
                f"T-PICS gated requires a valid automatic LLM target prompt for "
                f"{dataset_alias!r}; refusing merge fallback."
            )
        merged = _merge_prompt_fallback(
            dataset_alias,
            instruction,
            sample_trial_list,
            base_prompt_path=resolved_base_prompt,
            example_char_budget=example_char_budget,
            history_max_entries=history_max_entries,
            max_examples=max_examples,
        )
        if _is_gamble_ab_task(sample_trial_list):
            merged = _apply_gamble_neutral_wording(merged)
        merged = strip_embedded_choose_from_evolution_prompt(merged)
        infer_path.write_text(merged, encoding="utf-8")
        print(f"[TEH] Wrote merged fallback prompt -> {infer_path}")

    contract = build_deterministic_runtime_contract(sample_trial_list)
    assert_no_generation_oracle_leak(contract, context="runtime_contract")
    (prompts_dir / RUNTIME_CONTRACT_FILENAME).write_text(
        contract + "\n", encoding="utf-8"
    )
    infer_path.write_text(
        attach_runtime_contract_to_prompt(
            infer_path.read_text(encoding="utf-8"), contract
        ),
        encoding="utf-8",
    )
    print(f"[TEH] Appended deterministic runtime contract -> {infer_path}")

    shutil.copy2(BASE_REFINE_PROMPT, prompts_dir / "refine.txt")
    seed_src = seed_program_path.expanduser().resolve()
    if not seed_src.is_file():
        raise FileNotFoundError(f"Seed program not found: {seed_src}")
    shutil.copy2(seed_src, prompts_dir / "seed_program.py")

    infer_text_final = infer_path.read_text(encoding="utf-8")
    assert_no_generation_oracle_leak(
        infer_text_final, context=f"infer_single_choice:{infer_path}"
    )
    llm_input_path = prompts_dir / "llm_input_prompt.txt"
    if llm_input_path.is_file():
        assert_no_generation_oracle_leak(
            llm_input_path.read_text(encoding="utf-8"),
            context=f"llm_input_prompt:{llm_input_path}",
        )
    if require_auto_llm_prompt:
        if not generated or used_reference or used_dataset_prompt_file:
            raise RuntimeError(
                f"T-PICS gated requires prompt_mode=auto_llm for {dataset_alias!r}; "
                f"got generated={generated} reference={used_reference} "
                f"dataset_file={used_dataset_prompt_file}."
            )
        if not infer_text_final.strip():
            raise RuntimeError(
                f"T-PICS gated automatic target prompt is empty: {infer_path}"
            )
    if generated:
        prompt_mode = "auto_llm"
    elif used_reference:
        prompt_mode = "reference"
    elif used_dataset_prompt_file:
        prompt_mode = "dataset_prompt_file"
    else:
        prompt_mode = "merge_fallback"
    infer_sha256 = hashlib.sha256(infer_text_final.encode("utf-8")).hexdigest()

    meta: Dict[str, Any] = {
        "dataset_alias": dataset_alias,
        "llm_generated": generated,
        "prompt_mode": prompt_mode,
        "infer_prompt_path": str(infer_path.resolve()),
        "infer_prompt_sha256": infer_sha256,
        "used_reference_prompt": used_reference,
        "used_dataset_prompt_file": used_dataset_prompt_file,
        "dataset_prompt_file": str(forced_prompt) if used_dataset_prompt_file else None,
        "reference_prompt_source": str(reference_prompt) if used_reference else None,
        "seed_program_source": str(seed_src),
        "base_prompt_path": str(resolved_base_prompt),
        "dataset_prompt_evolved": False,
        "evolution_iterations": 0,
        "best_prompt_id": None,
        "example_char_budget": int(example_char_budget),
        "history_max_entries": int(history_max_entries),
        "max_examples": int(max_examples),
        "prefer_auto_llm_prompt": bool(prefer_auto_llm_prompt),
        "require_auto_llm_prompt": bool(require_auto_llm_prompt),
        "llm_decoding_seed": None if llm_decoding_seed is None else int(llm_decoding_seed),
        "n_prompt_example_trials": len(sample_trial_list),
        "prompt_examples_exclude_test": True,
        "runtime_contract_appended": True,
        "limited_data_protocol": normalize_limited_data_protocol(limited_data_protocol),
        "limited_train_val": None if limited_train_val is None else int(limited_train_val),
        "prompt_examples_from_retained_observed": (
            normalize_limited_data_protocol(limited_data_protocol) != LIMITED_DATA_PROTOCOL_OFF
        ),
        "prompt_sample_pid": int(sample_pid),
        "observed_subset_fingerprint": getattr(limited_manifest, "subset_fingerprint", None),
        "observed_train_fingerprint": getattr(limited_manifest, "train_fingerprint", None),
        "observed_val_fingerprint": getattr(limited_manifest, "val_fingerprint", None),
        "observed_test_fingerprint": getattr(limited_manifest, "test_fingerprint", None),
        "retained_n_train": getattr(limited_manifest, "retained_n_train", None),
        "retained_n_val": getattr(limited_manifest, "retained_n_val", None),
        "selected_example_fingerprints": [
            _prompt_trial_fingerprint(t)
            for t in select_diverse_train_trials(
                sample_trial_list, max_examples=int(max_examples)
            )
        ],
    }
    if is_mixed_gambles_dataset(dataset_alias):
        meta["mixed_gambles_csv"] = mixed_gambles_csv
        meta["filter_mixed_gambles"] = filter_mixed_gambles
    elif is_external_dataset(dataset_alias):
        meta["experiment_id"] = None
        meta["execution_source"] = "external"
        meta["default_data_dir"] = external_default_data_dir(dataset_alias)
    else:
        meta["experiment_id"] = experiment_id_for_alias(dataset_alias)
    (prompts_dir / "prompt_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return prompts_dir