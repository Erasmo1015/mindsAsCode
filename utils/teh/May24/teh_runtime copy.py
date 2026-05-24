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
    "Prefer concise programs and avoid long repetitive helper code. "
    "Do not simplify away useful behavioral structure just to make the code shorter. "
    "Keep choose() reasonably compact unless additional logic improves behavioral fit."
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


def _prompt_schema_audit_from_trials(
    dataset_alias: str,
    trials: List[Dict[str, Any]],
    instruction: str,
    schema_summary: str,
) -> Dict[str, Any]:
    """Audit runtime-observed prompt schema from parsed train/prompt trials only."""
    observed_problem_keys: set[str] = set()
    observed_history_keys: set[str] = set()
    schema_types: set[str] = set()
    has_feedback_ever_true = False
    has_feedback_ever_false = False
    total_trial_count = len(trials)
    empty_history_trial_count = 0
    total_history_entries = 0
    hist_action_key_present = 0
    hist_action_key_missing = 0
    hist_feedback_key_present = 0
    hist_feedback_key_missing = 0
    hist_feedback_none = 0
    hist_feedback_non_none = 0
    max_history_len = 0

    for trial in trials:
        problem = trial.get("problem") if isinstance(trial.get("problem"), dict) else {}
        for key in problem:
            if key not in ("dataset_alias", "experiment_id"):
                observed_problem_keys.add(key)
        has_feedback_val = problem.get("has_feedback")
        if has_feedback_val is True:
            has_feedback_ever_true = True
        if has_feedback_val is False:
            has_feedback_ever_false = True
        schema_type = problem.get("schema_type")
        if schema_type not in (None, "", "?"):
            schema_types.add(str(schema_type))

        history = trial.get("history") if isinstance(trial.get("history"), list) else []
        max_history_len = max(max_history_len, len(history))
        if not history:
            empty_history_trial_count += 1
            continue
        for entry in history:
            total_history_entries += 1
            if not isinstance(entry, dict):
                hist_action_key_missing += 1
                hist_feedback_key_missing += 1
                continue
            observed_history_keys.update(entry.keys())
            if "action" in entry:
                hist_action_key_present += 1
            else:
                hist_action_key_missing += 1
            if "feedback" in entry:
                hist_feedback_key_present += 1
                if entry.get("feedback") is None:
                    hist_feedback_none += 1
                else:
                    hist_feedback_non_none += 1
            else:
                hist_feedback_key_missing += 1

    has_gamble_a = "gamble_A" in observed_problem_keys
    has_gamble_b = "gamble_B" in observed_problem_keys
    feedback_key_may_be_absent = hist_feedback_key_missing > 0
    feedback_may_be_none = hist_feedback_none > 0
    feedback_observed_non_none = hist_feedback_non_none > 0

    return {
        "dataset_alias": dataset_alias,
        "instruction_excerpt": instruction[:500],
        "schema_summary_excerpt": schema_summary[:1200],
        "total_trial_count": total_trial_count,
        "empty_history_trial_count": empty_history_trial_count,
        "max_history_len": max_history_len,
        "total_history_entries": total_history_entries,
        "observed_problem_keys": sorted(observed_problem_keys),
        "observed_history_keys": sorted(observed_history_keys),
        "has_gamble_a": has_gamble_a,
        "has_gamble_b": has_gamble_b,
        "has_gamble_ab": has_gamble_a and has_gamble_b,
        "hist_action_key_present": hist_action_key_present,
        "hist_action_key_missing": hist_action_key_missing,
        "hist_feedback_key_present": hist_feedback_key_present,
        "hist_feedback_key_missing": hist_feedback_key_missing,
        "hist_feedback_none": hist_feedback_none,
        "hist_feedback_non_none": hist_feedback_non_none,
        "has_feedback_ever_true": has_feedback_ever_true,
        "has_feedback_ever_false": has_feedback_ever_false,
        "feedback_key_may_be_absent": feedback_key_may_be_absent,
        "feedback_may_be_none": feedback_may_be_none,
        "feedback_observed_non_none": feedback_observed_non_none,
        "schema_types": sorted(schema_types),
    }


def _history_feedback_stats(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize history/feedback signals from parsed sample trials."""
    audit = _prompt_schema_audit_from_trials("", trials, "", "")
    return {
        "has_feedback_true": audit["has_feedback_ever_true"],
        "has_feedback_false_all": not audit["has_feedback_ever_true"],
        "all_hist_empty": audit["empty_history_trial_count"] == audit["total_trial_count"],
        "max_hist": audit["max_history_len"],
        "hist_actions": audit["hist_action_key_present"],
        "hist_feedback_non_none": audit["hist_feedback_non_none"],
        "hist_feedback_key_present": audit["hist_feedback_key_present"],
        "hist_feedback_key_missing": audit["hist_feedback_key_missing"],
        "hist_feedback_none": audit["hist_feedback_none"],
        "hist_action_key_present": audit["hist_action_key_present"],
        "hist_action_key_missing": audit["hist_action_key_missing"],
        "is_bandit": "C" in set(audit["schema_types"]),
        "schema_types": set(audit["schema_types"]),
    }


def _infer_history_feedback_guidance(
    trials: List[Dict[str, Any]],
    *,
    dataset_alias: str,
    instruction: str = "",
    schema_summary: str = "",
) -> List[str]:
    """Minimal history/feedback availability guidance from parsed examples."""
    audit = _prompt_schema_audit_from_trials(
        dataset_alias, trials, instruction, schema_summary
    )
    lines: List[str] = []
    history_empty = audit["empty_history_trial_count"] == audit["total_trial_count"]
    no_history_entries = audit["total_history_entries"] == 0
    has_actions = audit["hist_action_key_present"] > 0
    feedback_any_non_none = audit["hist_feedback_non_none"] > 0
    feedback_absent_or_none = not feedback_any_non_none

    if history_empty or no_history_entries:
        lines.append("History is empty in parsed examples; rely on current problem fields.")
        lines.append("Non-None feedback is not observed in parsed examples.")
    elif has_actions and feedback_absent_or_none:
        lines.append("Parsed examples contain prior actions.")
        lines.append("Non-None feedback is not observed in parsed examples.")
    elif has_actions and feedback_any_non_none:
        lines.append("Parsed examples contain prior actions.")
        lines.append("Some feedback values are observed; feedback may be missing or None.")
    else:
        lines.append("History entries are present; use only observed history keys.")

    deduped: List[str] = []
    seen = set()
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)
    return deduped


def _format_history_feedback_section(guidance_lines: Sequence[str]) -> str:
    bullets = "\n".join(f"- {line}" for line in guidance_lines)
    return f"History and feedback:\n{bullets}"


def _format_guidance_section(title: str, lines: Sequence[str]) -> str:
    bullets = "\n".join(f"- {line}" for line in lines)
    return f"{title}\n{bullets}"


def _dedupe_lines_preserve_order(lines: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.strip()
        if not key:
            continue
        low_key = " ".join(key.lower().split())
        if low_key.startswith(
            "history is empty in parsed examples; rely on current problem fields"
        ):
            key = "History is empty in parsed examples; rely on current problem fields."
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line.strip())
    return deduped


def _strip_prompt_generation_meta_instructions(text: str) -> str:
    out_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            out_lines.append(raw_line)
            continue
        low = line.lower()
        if "include this section verbatim" in low:
            continue
        if "use the following points when writing" in low:
            continue
        if "output only the evolution instruction prompt text" in low:
            continue
        if "your task is not to rewrite the full base prompt" in low:
            continue
        out_lines.append(raw_line)
    return "\n".join(out_lines).strip()


def _dedupe_history_empty_guidance_lines(text: str) -> str:
    out_lines: List[str] = []
    seen_history_empty = False
    for raw_line in text.splitlines():
        normalized = " ".join(raw_line.strip().lower().split())
        if normalized.startswith(
            "- history is empty in parsed examples; rely on current problem fields"
        ):
            if seen_history_empty:
                continue
            seen_history_empty = True
        out_lines.append(raw_line)
    return "\n".join(out_lines).strip()


def _filter_unsupported_behavioral_priors(lines: Sequence[str]) -> List[str]:
    banned = (
        "prospect theory",
        "loss aversion",
        "subjective utility",
        "win-stay",
        "lose-shift",
        "exploration/exploitation",
        "participant bias",
        "weighted-sum",
        "weighted sum",
    )
    kept: List[str] = []
    for line in lines:
        low = line.lower()
        if any(term in low for term in banned):
            continue
        kept.append(line)
    return kept


def _render_dataset_schema_history_section(
    dataset_guidance_lines: Sequence[str],
    history_guidance_lines: Sequence[str],
) -> str:
    dataset_lines = _dedupe_lines_preserve_order(
        _filter_unsupported_behavioral_priors(dataset_guidance_lines)
    )
    history_lines = _dedupe_lines_preserve_order(
        _filter_unsupported_behavioral_priors(history_guidance_lines)
    )
    section_parts: List[str] = []
    if dataset_lines:
        section_parts.append(_format_guidance_section("Dataset/task-format guidance:", dataset_lines))
    if history_lines:
        section_parts.append(_format_history_feedback_section(history_lines))
    return "\n\n".join(section_parts).strip()


def _extract_section_block(prompt_text: str, section_title: str) -> str:
    lines = prompt_text.splitlines()
    section_titles = {
        "Requirements:",
        "History and feedback safety:",
        "Behavioral requirements:",
        "Parent comparison and improvement requirements:",
        "Generation requirements:",
    }
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip() == section_title:
            start_idx = idx
            break
    if start_idx is None:
        return ""

    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip() in section_titles:
            end_idx = idx
            break
    return "\n".join(lines[start_idx:end_idx]).strip()


def _ensure_invariant_base_sections(
    final_prompt: str,
    base_prompt: str,
    *,
    audit: Dict[str, Any],
) -> tuple[str, List[str]]:
    out = final_prompt
    warnings: List[str] = []
    required_sections = [
        "Requirements:",
        "History and feedback safety:",
        "Behavioral requirements:",
        "Parent comparison and improvement requirements:",
        "Generation requirements:",
    ]
    if "def choose(problem, history)" not in out and "def choose(problem, history)" in base_prompt:
        warnings.append("Missing `def choose(problem, history)` signature; restored from base prompt.")
        out = out.rstrip() + "\n\n" + "def choose(problem, history):" + "\n"

    for title in required_sections:
        if title in out:
            continue
        block = _extract_section_block(base_prompt, title)
        if block:
            warnings.append(f"Missing invariant section `{title}`; restored from base prompt.")
            out = out.rstrip() + "\n\n" + block + "\n"

    final_rule = "Provide only the code for `choose(...)` as a complete function."
    if final_rule not in out and final_rule in base_prompt:
        warnings.append("Missing final code-only rule; restored from base prompt.")
        out = out.rstrip() + "\n\n" + final_rule + "\n"

    if audit.get("has_gamble_ab"):
        out_l = out.lower()
        has_action0 = (
            'action 0 -> problem["gamble_a"]' in out_l
            or 'action 0 -> `problem["gamble_a"]`' in out_l
        )
        has_action1 = (
            'action 1 -> problem["gamble_b"]' in out_l
            or 'action 1 -> `problem["gamble_b"]`' in out_l
        )
        has_return = "return p(action=1)" in out_l
        if not (has_action0 and has_action1 and has_return):
            action_mapping_lines = [
                '- action 0 -> `problem["gamble_A"]`',
                '- action 1 -> `problem["gamble_B"]`',
                "- return P(action=1)",
            ]
            warnings.append("Missing gamble action mapping; restored canonical mapping.")
            out = (
                out.rstrip()
                + "\n\nDataset/task-format guidance:\n"
                + "\n".join(action_mapping_lines)
                + "\n"
            )

    return out, warnings


def _assemble_prompt_with_invariant_base(
    base_prompt: str,
    dataset_schema_history_section: str,
    *,
    audit: Dict[str, Any],
) -> tuple[str, List[str]]:
    section = _strip_prompt_generation_meta_instructions(dataset_schema_history_section)
    section_lines = _dedupe_lines_preserve_order(section.splitlines())
    section = "\n".join(section_lines).strip()
    if section:
        prompt = section + "\n\n" + base_prompt.strip()
    else:
        prompt = base_prompt.strip()
    prompt, invariant_warnings = _ensure_invariant_base_sections(prompt, base_prompt, audit=audit)
    prompt = _strip_prompt_generation_meta_instructions(prompt)
    prompt = _dedupe_history_empty_guidance_lines(prompt)
    return prompt.strip() + "\n", invariant_warnings


def _history_safety_requirements_from_audit(audit: Dict[str, Any]) -> str:
    lines: List[str] = [
        "Code must work when history is empty.",
        "History entries may not all contain the same keys.",
        'Do not use h["feedback"] or history[-1]["feedback"] unless checking key existence.',
        'Prefer h.get("feedback") and explicitly check whether the value is None.',
    ]
    if audit["hist_action_key_missing"] > 0:
        lines.append(
            'If using historical actions, use only entries that contain "action" or guard key access safely.'
        )
    if (
        audit["total_history_entries"] > 0
        and audit["hist_action_key_present"] > 0
        and not audit["feedback_observed_non_none"]
    ):
        lines.append(
            "History actions may be available, but feedback is absent or None; do not create feedback-based rules."
        )
    if (
        audit["hist_feedback_key_present"] == 0
        or not audit["feedback_observed_non_none"]
    ):
        lines.append(
            "Do not invent feedback-based rules when feedback is absent, always None, or unavailable in the parsed examples."
        )
    lines.append(
        "Feedback logic should be consistent with which action produced the feedback."
    )
    if audit["total_history_entries"] == 0:
        lines.append(
            "History is empty in parsed examples; rely on current problem fields instead of creating artificial history terms."
        )
    return _format_guidance_section("History safety requirements:", lines)


def _numerical_stability_requirements() -> str:
    return _format_guidance_section(
        "Numerical stability:",
        [
            "If using sigmoid/logistic/exponential mappings, clip the score to a safe range before exponentiation, e.g. [-50, 50].",
            "Avoid unbounded 10 ** (-score) or 2.718281828 ** (-score) when score can be large.",
            "Always return a finite float strictly inside (0, 1).",
        ],
    )


def _dataset_task_format_guidance(
    dataset_alias: str,
    trials: List[Dict[str, Any]],
    instruction: str,
    schema_summary: str,
    audit: Dict[str, Any],
) -> List[str]:
    _ = instruction
    lines: List[str] = []
    if audit["has_gamble_ab"]:
        lines.extend(
            [
                'problem["gamble_A"] corresponds to action 0.',
                'problem["gamble_B"] corresponds to action 1.',
                "choose(problem, history) returns P(action=1).",
                "Use only fields observed in parsed examples/schema.",
            ]
        )
        return lines

    action_sem = _extract_action_semantics(schema_summary)
    lines.extend(
        [
            f"Action semantics from schema summary: {action_sem}",
            "Use only fields observed in the runtime schema summary and parsed examples.",
        ]
    )
    if _is_gamble_ab_task(trials):
        lines.append("Return convention remains P(action=1).")
    else:
        lines.append(
            f'Observed problem keys: {", ".join(audit["observed_problem_keys"]) or "(none observed)"}'
        )
        lines.append(
            f'Observed history keys: {", ".join(audit["observed_history_keys"]) or "(none observed)"}'
        )
    return lines


def _ensure_history_feedback_guidance(text: str, guidance_lines: Sequence[str]) -> str:
    """Ensure a minimal History and feedback section exists."""
    if not guidance_lines:
        return text
    if "History and feedback:" in text:
        return text
    return text.rstrip() + "\n\n" + _format_history_feedback_section(guidance_lines) + "\n"


def _ensure_section(text: str, section_text: str, section_title: str) -> str:
    if section_title in text:
        return text
    return text.rstrip() + "\n\n" + section_text + "\n"


def _has_history_safety_content(text: str) -> bool:
    lower = text.lower()
    has_empty_history = (
        "history may be empty" in lower
        or "history is empty" in lower
        or "code must work when history is empty" in lower
        or "code must work when `history` is empty" in lower
    )
    has_feedback_absence_guard = (
        'h.get("feedback")' in text
        or "feedback key may be absent" in lower
        or "feedback may be absent" in lower
    )
    return has_empty_history and has_feedback_absence_guard


def _has_numerical_stability_content(text: str) -> bool:
    lower = text.lower()
    has_clip = (
        "clip the score to a safe range before exponentiation" in lower
        or "clip the score to a safe range" in lower
        or "clip the score before exponentiation" in lower
        or "clip to a safe range" in lower
    )
    has_exp_or_overflow = (
        "exponentiation" in lower
        or "exponential" in lower
        or "overflow" in lower
        or "unbounded" in lower
    )
    has_exp_guard = (
        "avoid unbounded 10 ** (-score)" in lower
        or "avoid unbounded 2.718281828 ** (-score)" in lower
        or ("avoid unbounded" in lower and "exponent" in lower)
        or "overflow" in lower
    )
    return has_clip and (has_exp_guard or has_exp_or_overflow)


def _ensure_compact_prompt_safety_additions(text: str, audit: Dict[str, Any]) -> str:
    out = text.rstrip()
    if not _has_history_safety_content(out):
        history_additions: List[str] = [
            "History may be empty.",
            'Prefer h.get("feedback"); feedback key may be absent.',
        ]
        if audit.get("total_history_entries", 0) == 0:
            history_additions.append(
                "History is empty in parsed examples; rely on current problem fields."
            )
        out = (
            out
            + "\n\n"
            + _format_guidance_section("History safety additions:", history_additions)
            + "\n"
        )

    if not _has_numerical_stability_content(out):
        out = (
            out
            + "\n\n"
            + _format_guidance_section(
                "Numerical stability additions:",
                [
                    "Clip the score to a safe range before exponentiation (e.g., [-50, 50]).",
                    "Avoid unbounded exponentiation / overflow (e.g., unbounded 10 ** (-score)).",
                ],
            )
            + "\n"
        )

    return out


def _ensure_prompt_safety_content(text: str, audit: Dict[str, Any]) -> str:
    return _ensure_compact_prompt_safety_additions(text, audit)


_CCT_PROBLEM_KEYS = frozenset(
    {
        "current_score",
        "n_cards_remaining",
        "n_loss_cards",
        "gain_amount",
        "loss_amount",
        "cards_flipped",
    }
)
_PRODUCT_RATING_KEYS = frozenset({"ratings_A", "ratings_B"})
_CCT_DATASET_ALIAS = "3frey2017cct"
_PRODUCT_RATING_DATASET_ALIAS = "7hilbig2014generalized"


def _is_cct_task(trials: List[Dict[str, Any]], *, dataset_alias: str = "") -> bool:
    _ = dataset_alias
    problem_keys = set(_problem_keys_from_trials(trials))
    return len(problem_keys & _CCT_PROBLEM_KEYS) >= 3


def _is_product_rating_task(trials: List[Dict[str, Any]], *, dataset_alias: str = "") -> bool:
    _ = dataset_alias
    problem_keys = set(_problem_keys_from_trials(trials))
    return _PRODUCT_RATING_KEYS.issubset(problem_keys)


def _cct_specific_guidance(action_sem: str) -> str:
    return (
        "CCT-specific guidance:\n"
        "- This is a stop/continue risk task.\n"
        "- Build rules around comparing stopping now against continuing.\n"
        "- A useful structure is:\n"
        "  ev_stop = current_score\n"
        "  p_loss = n_loss_cards / n_cards_remaining if n_cards_remaining > 0 else 0\n"
        "  ev_continue = current_score + (1 - p_loss) * gain_amount - p_loss * loss_amount\n"
        "  decision_score = ev_stop - ev_continue\n"
        "- Higher decision_score should increase P(stop), according to the dataset action semantics.\n"
        f"- Action semantics for this dataset: {action_sem}\n"
        "- Use current_score, remaining cards, loss-card risk, gain amount, and loss amount.\n"
        "- Avoid generic raw expected-value rules that ignore the stop-vs-continue structure.\n"
        "- If the stop/continue rule consistently fits observed choices, use appropriately confident probabilities; otherwise stay calibrated."
    )


def _product_rating_specific_guidance() -> str:
    return (
        "Product-rating guidance:\n"
        "- Parsed fields indicate a product/cue-rating choice task.\n"
        "- `ratings_A` and `ratings_B` are observed problem fields.\n"
        "- Compare and use observed rating fields in ways consistent with dataset action semantics.\n"
        "- Do not describe this task as a gamble task."
    )


def _schema_neutral_behavioral_sections(
    trials: List[Dict[str, Any]],
    *,
    dataset_alias: str,
    action_sem: str,
) -> str:
    sections = [
        "Behavioral requirements:",
        "- Do not return constant or near-constant probabilities.",
        "- The probability must depend meaningfully on the problem inputs.",
        "- When the learned rule clearly favors action 1, return probability above 0.5.",
        "- When the learned rule clearly favors action 0, return probability below 0.5.",
        "- If a good rule predicts the correct direction but is too close to 0.5, move the "
        "probability farther from 0.5 in the same direction.",
        "- If the rule is uncertain or conflicting, return a probability closer to 0.5.",
        "- Do not add history terms unless history improves behavioral fit.",
        "- Avoid history/idiosyncratic terms that make a strong problem-based rule worse.",
        "",
        "Parent comparison:",
        "- When parent programs and scores are provided, preserve components from better-scoring parents.",
        "- Do not replace a strong parent with a generic simpler rule.",
        "- Remove or weaken code, parameters, or history terms that appear in worse parents "
        "but not in better parents.",
        "- Prefer targeted changes to scaling, thresholds, confidence, feature weights, "
        "or small behavioral components.",
    ]
    if _is_cct_task(trials, dataset_alias=dataset_alias):
        sections.extend(["", _cct_specific_guidance(action_sem)])
    if _is_product_rating_task(trials, dataset_alias=dataset_alias):
        sections.extend(["", _product_rating_specific_guidance()])
    return "\n".join(sections)


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
    dataset_alias: str = "",
    instruction: str = "",
    history_feedback_guidance: Optional[Sequence[str]] = None,
) -> str:
    """Non-gamble base prompt: document observed problem/history keys only."""
    problem_keys = _problem_keys_from_trials(trials)
    action_sem = _extract_action_semantics(schema_summary)
    observed_history_keys: set[str] = set()
    for trial in trials:
        history = trial.get("history") if isinstance(trial.get("history"), list) else []
        for entry in history:
            if isinstance(entry, dict):
                observed_history_keys.update(entry.keys())
    history_note = ", ".join(sorted(observed_history_keys)) if observed_history_keys else "(none observed)"
    guidance_lines = list(
        history_feedback_guidance
        or _infer_history_feedback_guidance(
            trials,
            dataset_alias=dataset_alias,
            instruction=instruction,
            schema_summary=schema_summary,
        )
    )
    problem_doc = (
        "\n".join(f"        - {key}: (type/structure per parsed examples)" for key in problem_keys)
        or "        - (see parsed trial examples)"
    )
    behavioral_sections = _schema_neutral_behavioral_sections(
        trials,
        dataset_alias=dataset_alias,
        action_sem=action_sem,
    )
    return (
        "You are given observations of human choices in binary decision problems.\n"
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
        "    return: float, probability of choosing action 1, i.e. P(action=1)\n"
        '    """\n\n'
        f"{_GENERIC_PROMPT_REQUIREMENTS}\n\n"
        f"{behavioral_sections}\n\n"
        f"{_format_history_feedback_section(guidance_lines)}\n\n"
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


def _prompt_schema_sanity_warnings(
    prompt_text: str,
    *,
    dataset_alias: str,
    audit: Dict[str, Any],
) -> List[str]:
    warnings: List[str] = []
    is_gamble = bool(audit.get("has_gamble_ab"))
    text_l = prompt_text.lower()

    if is_gamble:
        if "def choose(problem, history)" not in prompt_text:
            warnings.append("Prompt missing explicit `def choose(problem, history)` API signature.")
        if "p(action=1)" not in text_l and "probability of choosing action 1" not in text_l:
            warnings.append("Prompt does not clearly document return convention as P(action=1).")
        if not re.search(r"action\s*0.*gamble_a", text_l):
            warnings.append("Prompt missing explicit mapping: action 0 -> gamble_A.")
        if not re.search(r"action\s*1.*gamble_b", text_l):
            warnings.append("Prompt missing explicit mapping: action 1 -> gamble_B.")

        if audit.get("feedback_key_may_be_absent"):
            rigid_pattern = re.compile(
                r"history\s*:\s*list of dicts with keys action and feedback", re.IGNORECASE
            )
            optional_pattern = re.compile(
                r"feedback key may be absent|optional|guard.*feedback|h\.get\([\"']feedback[\"']\)",
                re.IGNORECASE,
            )
            if rigid_pattern.search(prompt_text) and not optional_pattern.search(prompt_text):
                warnings.append(
                    "Prompt states rigid history keys (action+feedback) without documenting optional/missing feedback."
                )
            if not optional_pattern.search(prompt_text):
                warnings.append(
                    "Prompt does not mention safe feedback access (`h.get(\"feedback\")` or key-absence guarding)."
                )
    else:
        if not is_gamble and re.search(
            r"gamble_A|gamble_B|unknown probabilit|\blottery\b|choice13k", prompt_text, re.IGNORECASE
        ):
            warnings.append(
                f"Non-gamble dataset `{dataset_alias}` prompt contains gamble-specific terminology."
            )

    return warnings


def _base_prompt_for_trials(
    trials: List[Dict[str, Any]],
    schema_summary: str,
    *,
    dataset_alias: str,
    instruction: str = "",
    base_prompt_path: Optional[Path | str] = None,
) -> str:
    if _is_gamble_ab_task(trials):
        return _choice13k_neutral_loglik_base(base_prompt_path)
    return _build_schema_neutral_base_prompt(
        schema_summary,
        trials,
        dataset_alias=dataset_alias,
        instruction=instruction,
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
            "Each problem presents two options: option 0 and option 1. "
            "problem[\"gamble_A\"] stores option 0; problem[\"gamble_B\"] stores option 1."
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
) -> tuple[str, str, List[str]]:
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    audit = _prompt_schema_audit_from_trials(
        dataset_alias, sample_trials, instruction, schema_summary
    )
    guidance_lines = _infer_history_feedback_guidance(
        sample_trials,
        dataset_alias=dataset_alias,
        instruction=instruction,
        schema_summary=schema_summary,
    )
    dataset_guidance_lines = _dataset_task_format_guidance(
        dataset_alias,
        sample_trials,
        instruction,
        schema_summary,
        audit,
    )
    base = _base_prompt_for_trials(
        sample_trials,
        schema_summary,
        dataset_alias=dataset_alias,
        instruction=instruction,
        base_prompt_path=base_prompt_path,
    )
    dataset_schema_history_section = _render_dataset_schema_history_section(
        dataset_guidance_lines, guidance_lines
    )
    merged, invariant_warnings = _assemble_prompt_with_invariant_base(
        base,
        dataset_schema_history_section,
        audit=audit,
    )
    merged = _ensure_history_feedback_guidance(merged, guidance_lines)
    merged = _ensure_prompt_safety_content(merged, audit)
    merged = _dedupe_history_empty_guidance_lines(merged)
    return merged, dataset_schema_history_section, invariant_warnings


def _combine_sample_trials_from_participants(
    trial_lists: Sequence[List[Dict[str, Any]]],
    *,
    max_total: int = 8,
) -> List[Dict[str, Any]]:
    """Merge a few trials from each participant list (deterministic, capped)."""
    lists = [trials for trials in trial_lists if trials]
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0][:max_total]
    per = max(1, max_total // len(lists))
    combined: List[Dict[str, Any]] = []
    for trials in lists:
        combined.extend(trials[:per])
    return combined[:max_total]


def _psych101_sample_trials_for_prompts(
    dataset_alias: str,
    *,
    n_sample_participants: int,
    psych_dataset_split: str,
    local_dataset: Optional[str],
) -> tuple[str, List[Dict[str, Any]], List[int]]:
    """Instruction and merged trial examples from multiple valid participant row indices."""
    from utils.teh.participant_ids import load_valid_participant_ids

    n_want = max(1, int(n_sample_participants))
    sample_row_indices: List[int]
    try:
        valid_ids = load_valid_participant_ids(
            dataset_alias,
            REPO_ROOT,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
        )
        sample_row_indices = [int(x) for x in valid_ids[:n_want]] if valid_ids else [0]
    except Exception:
        sample_row_indices = [0]

    trial_lists: List[List[Dict[str, Any]]] = []
    instruction = ""
    exp_id = experiment_id_for_alias(dataset_alias)
    for row_idx in sample_row_indices:
        exp = get_psych101_binary_experiment(
            dataset_alias,
            row_idx,
            split=psych_dataset_split,
            local_dataset=local_dataset,
        )
        if not instruction:
            instruction = exp.instruction
        trial_lists.append(
            experiment_to_trial_dicts(
                exp,
                dataset_alias=dataset_alias,
                experiment_id=exp_id,
            )
        )
    return (
        instruction,
        _combine_sample_trials_from_participants(trial_lists, max_total=8),
        sample_row_indices,
    )


def _mixed_gambles_sample_trials_for_prompts(
    dataset_alias: str,
    *,
    n_sample_participants: int,
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> tuple[List[Dict[str, Any]], List[int]]:
    """Merged train-trial examples from multiple valid mixed-gambles participant ids."""
    from utils.teh.participant_ids import load_valid_participant_ids

    valid_ids = load_valid_participant_ids(
        dataset_alias,
        REPO_ROOT,
        filter_mixed_gambles=filter_mixed_gambles,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    if not valid_ids:
        raise ValueError(f"No valid participant ids for mixed_gambles dataset {dataset_alias!r}")

    n_want = max(1, int(n_sample_participants))
    sample_pids = [int(x) for x in valid_ids[:n_want]]
    trial_lists: List[List[Dict[str, Any]]] = []
    for pid in sample_pids:
        train_trials, _, _, _ = load_mixed_gambles_trials(
            pid,
            csv_path=mixed_gambles_csv,
            filter_gain_loss_only=filter_mixed_gambles,
        )
        trial_lists.append(train_trials)
    return (
        _combine_sample_trials_from_participants(trial_lists, max_total=8),
        sample_pids,
    )


def build_prompt_generation_llm_user_content(
    dataset_alias: str,
    instruction: str,
    sample_trials: List[Dict[str, Any]],
    *,
    base_prompt_path: Optional[Path | str] = None,
) -> str:
    """User message for generating only dataset/schema/history adaptation text."""
    display = dataset_display_name(dataset_alias)
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    audit = _prompt_schema_audit_from_trials(
        dataset_alias, sample_trials, instruction, schema_summary
    )
    is_gamble = _is_gamble_ab_task(sample_trials)
    guidance_lines = _infer_history_feedback_guidance(
        sample_trials,
        dataset_alias=dataset_alias,
        instruction=instruction,
        schema_summary=schema_summary,
    )
    dataset_guidance_lines = _dataset_task_format_guidance(
        dataset_alias,
        sample_trials,
        instruction,
        schema_summary,
        audit,
    )
    base_prompt = _base_prompt_for_trials(
        sample_trials,
        schema_summary,
        dataset_alias=dataset_alias,
        instruction=instruction,
        base_prompt_path=base_prompt_path,
    )
    trial_examples = _format_trials_for_prompt(sample_trials, max_trials=10)
    if is_mixed_gambles_dataset(dataset_alias):
        task_description = instruction
    else:
        task_description = PSYCH101_BINARY_DATASETS[
            normalize_psych101_dataset_alias(dataset_alias)
        ]["task_description"]

    adaptation_shape = (
        "Dataset/task-format guidance:\n"
        "- <bullet 1>\n"
        "- <bullet 2>\n\n"
        "History and feedback:\n"
        "- <bullet 1>\n"
        "- <bullet 2>\n"
    )
    if is_gamble:
        adapt_constraints = (
            "- Preserve action mapping: action 0 -> `problem[\"gamble_A\"]`, action 1 -> `problem[\"gamble_B\"]`, return P(action=1).\n"
            "- Do not add unsupported behavioral priors (Prospect Theory, loss aversion, subjective utility, etc.) unless directly present in parsed schema/instruction.\n"
        )
    else:
        adapt_constraints = (
            "- Document only observed non-gamble fields and action semantics from parsed examples/schema.\n"
            "- Do not introduce gamble-only field assumptions unless present in parsed schema.\n"
        )

    return (
        f"Generate only a concise dataset/schema/history adaptation section for dataset `{dataset_alias}` ({display}).\n\n"
        "Your task is NOT to rewrite the full base prompt. Generate only the dataset-specific schema/history section that should be inserted before the invariant base prompt.\n"
        "Use runtime schema summary and parsed trial examples as source of truth for:\n"
        "- observed problem keys,\n"
        "- observed history keys,\n"
        "- optional/missing keys,\n"
        "- feedback availability,\n"
        "- action semantics,\n"
        "- return convention.\n\n"
        "Do not remove, summarize, or weaken invariant base prompt requirements. The invariant base prompt already contains API, implementation, safety, behavioral-search, parent-comparison, and generation rules.\n"
        "Output only the adaptation section text (no code fence, no Python code).\n"
        "Keep it concise and non-repetitive; deduplicate repeated lines.\n"
        "Do not include prompt-generation meta-instructions such as \"Include this section verbatim\".\n"
        "Do not force unsupported dataset-specific behavioral priors.\n"
        f"{adapt_constraints}"
        "Prefer these exact section titles in output:\n"
        f"{adaptation_shape}\n"
        "When history is unavailable, state that plainly and rely on current problem fields.\n"
        "When action history exists but feedback is unavailable, do not claim history is unavailable.\n\n"
        f"## Runtime schema summary\n\n{schema_summary}\n\n"
        "## Prompt schema audit (from parsed trials)\n\n"
        f"{json.dumps({k: audit[k] for k in ('observed_problem_keys', 'observed_history_keys', 'total_trial_count', 'empty_history_trial_count', 'total_history_entries', 'hist_action_key_present', 'hist_action_key_missing', 'hist_feedback_key_present', 'hist_feedback_key_missing', 'hist_feedback_none', 'hist_feedback_non_none', 'has_feedback_ever_true', 'has_feedback_ever_false', 'feedback_key_may_be_absent', 'feedback_may_be_none', 'feedback_observed_non_none', 'schema_types')}, indent=2)}\n\n"
        f"{_format_guidance_section('Dataset/task-format guidance:', dataset_guidance_lines)}\n\n"
        f"{_format_history_feedback_section(guidance_lines)}\n\n"
        f"## Invariant base prompt (do not rewrite)\n\n{base_prompt}\n\n"
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
) -> tuple[str, str, List[str]]:
    user_content = build_prompt_generation_llm_user_content(
        dataset_alias,
        instruction,
        sample_trials,
        base_prompt_path=base_prompt_path,
    )
    schema_summary = _runtime_schema_summary_for_prompt(sample_trials)
    audit = _prompt_schema_audit_from_trials(
        dataset_alias, sample_trials, instruction, schema_summary
    )
    guidance_lines = _infer_history_feedback_guidance(
        sample_trials,
        dataset_alias=dataset_alias,
        instruction=instruction,
        schema_summary=schema_summary,
    )
    dataset_guidance_lines = _dataset_task_format_guidance(
        dataset_alias,
        sample_trials,
        instruction,
        schema_summary,
        audit,
    )
    base_prompt = _base_prompt_for_trials(
        sample_trials,
        schema_summary,
        dataset_alias=dataset_alias,
        instruction=instruction,
        base_prompt_path=base_prompt_path,
    )
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
    text = _strip_prompt_generation_meta_instructions(text)
    llm_lines = _dedupe_lines_preserve_order(text.splitlines())
    llm_lines = _filter_unsupported_behavioral_priors(llm_lines)
    canonical_section = _render_dataset_schema_history_section(
        dataset_guidance_lines,
        guidance_lines,
    )
    if llm_lines:
        llm_section = "\n".join(llm_lines).strip()
        if "Dataset/task-format guidance:" not in llm_section:
            llm_section = canonical_section + "\n\n" + llm_section
        dataset_schema_history_section = llm_section.strip()
    else:
        dataset_schema_history_section = canonical_section

    final_prompt, invariant_warnings = _assemble_prompt_with_invariant_base(
        base_prompt,
        dataset_schema_history_section,
        audit=audit,
    )
    if _is_gamble_ab_task(sample_trials):
        final_prompt = _apply_gamble_neutral_wording(final_prompt)
    elif _prompt_has_gamble_leakage(final_prompt):
        print(
            "[TEH] Non-gamble prompt contained gamble-specific text after assembly; "
            "falling back to schema-neutral base prompt with canonical schema/history section."
        )
        schema_neutral_base = _build_schema_neutral_base_prompt(
            schema_summary,
            sample_trials,
            dataset_alias=dataset_alias,
            instruction=instruction,
            history_feedback_guidance=guidance_lines,
        )
        final_prompt, fallback_warnings = _assemble_prompt_with_invariant_base(
            schema_neutral_base,
            canonical_section,
            audit=audit,
        )
        invariant_warnings.extend(fallback_warnings)
    final_prompt = _ensure_history_feedback_guidance(final_prompt, guidance_lines)
    final_prompt = _ensure_section(
        final_prompt,
        _format_guidance_section("Dataset/task-format guidance:", dataset_guidance_lines),
        "Dataset/task-format guidance:",
    )
    final_prompt = _ensure_section(
        final_prompt, _format_history_feedback_section(guidance_lines), "History and feedback:"
    )
    final_prompt = _ensure_prompt_safety_content(final_prompt, audit)
    final_prompt = _dedupe_history_empty_guidance_lines(final_prompt)
    final_prompt = strip_embedded_choose_from_evolution_prompt(final_prompt)
    final_prompt = _strip_prompt_generation_meta_instructions(final_prompt)
    if not final_prompt:
        raise ValueError("LLM prompt was empty after removing embedded choose() code.")
    return final_prompt, dataset_schema_history_section, invariant_warnings


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
        sample_trial_list, sample_participant_ids = _mixed_gambles_sample_trials_for_prompts(
            dataset_alias,
            n_sample_participants=n_sample_participants,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
        instruction = (
            "Mixed gambles: Option A is a 50/50 gamble (gain/loss); Option B is certain. "
            "action=0 gamble, action=1 certain; choose(problem, history) returns P(action=1)."
        )
    else:
        instruction, sample_trial_list, sample_participant_ids = _psych101_sample_trials_for_prompts(
            dataset_alias,
            n_sample_participants=n_sample_participants,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
        )

    infer_path = prompts_dir / "infer_single_choice.txt"
    schema_summary = _runtime_schema_summary_for_prompt(sample_trial_list)
    audit = _prompt_schema_audit_from_trials(
        dataset_alias, sample_trial_list, instruction, schema_summary
    )
    history_guidance = _infer_history_feedback_guidance(
        sample_trial_list,
        dataset_alias=dataset_alias,
        instruction=instruction,
        schema_summary=schema_summary,
    )
    dataset_guidance = _dataset_task_format_guidance(
        dataset_alias,
        sample_trial_list,
        instruction,
        schema_summary,
        audit,
    )
    generated_dataset_schema_history_section = _render_dataset_schema_history_section(
        dataset_guidance, history_guidance
    )
    audit_warnings: List[str] = []
    generated = False
    final_prompt_text = ""
    if use_llm and client is not None:
        try:
            infer_text, generated_dataset_schema_history_section, invariant_warnings = _generate_prompt_via_llm(
                client,
                model_name,
                dataset_alias,
                instruction,
                sample_trial_list,
                save_llm_input_to=prompts_dir / "llm_input_prompt.txt",
                base_prompt_path=resolved_base_prompt,
            )
            final_prompt_text = strip_embedded_choose_from_evolution_prompt(infer_text)
            infer_path.write_text(final_prompt_text, encoding="utf-8")
            audit_warnings.extend(invariant_warnings)
            generated = True
            print(f"[TEH] Wrote LLM-generated prompt -> {infer_path}")
        except Exception as e:
            print(f"[TEH] LLM prompt generation failed ({e}); using merge fallback.")

    if not generated:
        merged, generated_dataset_schema_history_section, invariant_warnings = _merge_prompt_fallback(
            dataset_alias,
            instruction,
            sample_trial_list,
            base_prompt_path=resolved_base_prompt,
        )
        if _is_gamble_ab_task(sample_trial_list):
            merged = _apply_gamble_neutral_wording(merged)
        merged = strip_embedded_choose_from_evolution_prompt(merged)
        final_prompt_text = merged
        infer_path.write_text(final_prompt_text, encoding="utf-8")
        audit_warnings.extend(invariant_warnings)
        print(f"[TEH] Wrote merged fallback prompt -> {infer_path}")

    if final_prompt_text:
        audit_warnings.extend(
            _prompt_schema_sanity_warnings(
                final_prompt_text,
                dataset_alias=dataset_alias,
                audit=audit,
            )
        )
    for warning in audit_warnings:
        print(f"[TEH][prompt-warning] {warning}")

    audit_path = prompts_dir / "prompt_schema_audit.json"
    audit_payload: Dict[str, Any] = {
        "dataset_alias": dataset_alias,
        "sample_participant_ids": sample_participant_ids,
        "observed_problem_keys": audit["observed_problem_keys"],
        "observed_history_keys": audit["observed_history_keys"],
        "has_gamble_a": audit["has_gamble_a"],
        "has_gamble_b": audit["has_gamble_b"],
        "has_gamble_ab": audit["has_gamble_ab"],
        "total_trial_count": audit["total_trial_count"],
        "empty_history_trial_count": audit["empty_history_trial_count"],
        "total_history_entries": audit["total_history_entries"],
        "hist_action_key_present": audit["hist_action_key_present"],
        "hist_action_key_missing": audit["hist_action_key_missing"],
        "hist_feedback_key_present": audit["hist_feedback_key_present"],
        "hist_feedback_key_missing": audit["hist_feedback_key_missing"],
        "hist_feedback_none": audit["hist_feedback_none"],
        "hist_feedback_non_none": audit["hist_feedback_non_none"],
        "has_feedback_ever_true": audit["has_feedback_ever_true"],
        "has_feedback_ever_false": audit["has_feedback_ever_false"],
        "feedback_key_may_be_absent": audit["feedback_key_may_be_absent"],
        "feedback_may_be_none": audit["feedback_may_be_none"],
        "feedback_observed_non_none": audit["feedback_observed_non_none"],
        "schema_types": audit["schema_types"],
        "generated_dataset_task_format_guidance": dataset_guidance,
        "generated_history_feedback_guidance": history_guidance,
        "generated_dataset_schema_history_section": generated_dataset_schema_history_section,
        "warnings": audit_warnings,
        "schema_source_of_truth": True,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

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
        "n_sample_participants": int(n_sample_participants),
        "sample_participant_ids": sample_participant_ids,
        "prompt_schema_audit_path": str(audit_path),
        "schema_source_of_truth": True,
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