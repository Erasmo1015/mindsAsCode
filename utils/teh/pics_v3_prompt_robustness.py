"""PICS v3 post-adaptive history reminder (structure_aware_v3 only).

Default / legacy: a single generic HISTORY_ROBUSTNESS_BLOCK_V3 for all datasets.

Prospective policy ``dataset_keyed_post_adaptive_v2``: for named datasets only,
replace that block body with a concise dataset-keyed reminder. All other datasets
keep the byte-identical legacy block.

v2 revises Steyvers / Badham / Speekenbrink guidance. CPC18
(``2plonsky2018when``) stays on the legacy generic block (centaur-gap CPC18
prompt/example changes reverted after ``job_294982`` did not improve).
Kool / Schulz / Guan bodies stay byte-identical to v1.
"""
from __future__ import annotations

import threading
from typing import Dict, Mapping, Optional

HISTORY_ROBUSTNESS_MARKER = "HISTORY_ROBUSTNESS_BLOCK_V3"

# Official prospective policy id for new structure_aware_v3 main jobs.
DATASET_KEYED_POST_ADAPTIVE_POLICY_ID = "dataset_keyed_post_adaptive_v2"
# Historical policy id (reminder_v1 jobs); bodies for kool/schulz/guan unchanged.
DATASET_KEYED_POST_ADAPTIVE_POLICY_ID_V1 = "dataset_keyed_post_adaptive_v1"
LEGACY_GENERIC_REMINDER_POLICY_ID = "generic_history_robustness_v0"

# Immutable legacy body (byte-stable for unaffected datasets and reproduction).
HISTORY_ROBUSTNESS_BLOCK = (
    f"[{HISTORY_ROBUSTNESS_MARKER}]\n"
    "`history` may be empty, and different history entries may contain different "
    "fields. Never assume optional fields such as `feedback`, `reward`, or "
    "outcome fields exist or are non-null. Check for a key or use `.get(...)` "
    "before reading it. Only fields explicitly required by the current "
    "task/API contract may be accessed directly.\n"
    f"[/{HISTORY_ROBUSTNESS_MARKER}]"
)

# Datasets that receive a keyed override under dataset_keyed_post_adaptive_v2.
DATASET_KEYED_REMINDER_ALIASES = frozenset(
    {
        "5speekenbrink2008learning",
        "14kool2016when",
        "steyvers_2009_bandit",
        "13schulz2020finding",
        "guan_2020_stopping",
        "12badham2017deficits",
    }
)

# Bodies only (no markers). Kept factual from loaders / frozen contracts.
_DATASET_KEYED_REMINDER_BODIES: Dict[str, str] = {
    "5speekenbrink2008learning": (
        "`history` may be empty; use `.get` for optional history fields. "
        "Do not read current-trial weather/correctness from `problem`.\n"
        "Primary current cue: `problem['cards']`. Learn participant-specific "
        "cue→weather associations from past `cards` with `weather_outcome` / "
        "`was_correct` (and optional `feedback`). Prefer empirical association "
        "counts over fixed hard-coded card-set lookup tables when history "
        "supplies evidence. Smooth/bound probabilities; if evidence is sparse "
        "or uninformative, stay near ~0.5. Do not invent unavailable fields."
    ),
    "14kool2016when": (
        "`history` may be empty; use `.get` for stage-conditional fields. "
        "Never read current-trial `reward`/`treasure` from `problem`.\n"
        "Stage 1 (`stage==1`): integer action 0/1 over spaceship "
        "`option_keys`/`spaceship_options`; learn from `stage==1` history only.\n"
        "Stage 2 (`stage==2`): condition on `planet`, `alien_options`/"
        "`option_keys`, and available `spaceship`/`stage1_action`; learn from "
        "`stage==2` history using `feedback`/`reward` when present. Never "
        "`option_keys.index(letter)` for actions. Clip probs; avoid extremes "
        "from raw counts."
    ),
    "steyvers_2009_bandit": (
        "`history` may be empty; when present it has `action` and `reward` "
        "(use `.get`—reward may be absent in defensive edge cases).\n"
        "`reward` is the primary learning signal. Map actions through the "
        "current valid option keys/`problem['options']`—do not treat action "
        "ids as raw list indices into a fixed array. Initialize every legal "
        "arm explicitly. Use positive smoothing so denominators cannot be "
        "zero. Return a finite probability for every legal arm (full K-way "
        "dict over `option['action']`, K=4). Normalize safely; use a uniform "
        "fallback only when history is empty or yields no usable reward "
        "signal—non-empty usable history must not collapse to uniform. "
        "Avoid undefined helper variables and fragile manual renormalization."
    ),
    "13schulz2020finding": (
        "`history` may be empty; when present it has `action` and `reward`. "
        "Use `.get` for safety—do not assume reward is usually missing.\n"
        "`reward` is the primary learning signal. Return a full K-way dict "
        "over every `option['action']` in `problem['options']` (K=8). Empty "
        "history → neutral prior; non-empty history must not collapse to "
        "uniform. Smooth sparse counts; renormalize finite non-negative probs."
    ),
    "guan_2020_stopping": (
        "`history` may be empty. No `feedback`/`reward` interface.\n"
        "Use `position`, `values_observed` (prefix through current position), "
        "`sequence_length`, `environment`. Actions: 0=continue, 1=stop; "
        "return P(stop). At the final position, stop is the only remaining "
        "decision. Optional history: prior continues as "
        "`{action,position,value}`. Do not invent outcome fields."
    ),
    "12badham2017deficits": (
        "`history` may be empty; use `.get` for nested feedback.\n"
        "Use current `stimulus_features`; learn within the current "
        "`rule_block_id` (reset/separate learning across blocks). "
        "Binary feedback: when `feedback.is_correct` is True the chosen "
        "action is the correct category; when False the other binary "
        "category is correct—both correct and incorrect past trials provide "
        "learning information. Past `feedback.correct_category` may appear "
        "in history only—never read a current-trial correct category from "
        "`problem`. Use smoothed/bounded scores; do not accumulate unbounded "
        "feature-count logits. Return calibrated P(action=1)."
    ),
}

_tls = threading.local()


def using_pics_v3_prompt_robustness() -> bool:
    return bool(getattr(_tls, "enabled", False))


def using_pics_v3_legacy_generic_reminder() -> bool:
    return bool(getattr(_tls, "force_legacy", False))


def configure_pics_v3_legacy_generic_reminder(enabled: bool) -> None:
    """Process-level opt-in to force the legacy generic block for all datasets."""
    _tls.force_legacy = bool(enabled)


def current_pics_v3_reminder_dataset() -> Optional[str]:
    raw = getattr(_tls, "dataset", None)
    return str(raw) if raw else None


def normalize_reminder_dataset_alias(dataset: Optional[str]) -> Optional[str]:
    if dataset is None:
        return None
    alias = str(dataset).strip()
    return alias or None


def dataset_keyed_reminder_body(dataset: Optional[str]) -> Optional[str]:
    alias = normalize_reminder_dataset_alias(dataset)
    if alias is None:
        return None
    return _DATASET_KEYED_REMINDER_BODIES.get(alias)


def wrap_history_robustness_body(body: str) -> str:
    text = (body or "").strip()
    return f"[{HISTORY_ROBUSTNESS_MARKER}]\n{text}\n[/{HISTORY_ROBUSTNESS_MARKER}]"


def resolve_history_robustness_policy_id(
    dataset: Optional[str] = None,
    *,
    force_legacy: Optional[bool] = None,
) -> str:
    """Policy id that would be used for this dataset under the current/default rules."""
    legacy = (
        bool(force_legacy)
        if force_legacy is not None
        else using_pics_v3_legacy_generic_reminder()
    )
    if legacy:
        return LEGACY_GENERIC_REMINDER_POLICY_ID
    alias = normalize_reminder_dataset_alias(dataset)
    if alias is None:
        alias = current_pics_v3_reminder_dataset()
    if alias in DATASET_KEYED_REMINDER_ALIASES:
        return DATASET_KEYED_POST_ADAPTIVE_POLICY_ID
    return LEGACY_GENERIC_REMINDER_POLICY_ID


def resolve_history_robustness_block(
    dataset: Optional[str] = None,
    *,
    force_legacy: Optional[bool] = None,
) -> str:
    """Return the exact reminder block for this dataset (markers included)."""
    legacy = (
        bool(force_legacy)
        if force_legacy is not None
        else using_pics_v3_legacy_generic_reminder()
    )
    if legacy:
        return HISTORY_ROBUSTNESS_BLOCK
    alias = normalize_reminder_dataset_alias(dataset)
    if alias is None:
        alias = current_pics_v3_reminder_dataset()
    body = dataset_keyed_reminder_body(alias)
    if body is None:
        return HISTORY_ROBUSTNESS_BLOCK
    return wrap_history_robustness_body(body)


def all_dataset_keyed_reminder_bodies() -> Mapping[str, str]:
    """Read-only view of keyed bodies (tests / docs)."""
    return dict(_DATASET_KEYED_REMINDER_BODIES)


class pics_v3_prompt_robustness_scope:
    """Thread-local: inject history robustness into candidate prompts."""

    def __init__(
        self,
        enabled: bool,
        *,
        dataset: Optional[str] = None,
        force_legacy: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.dataset = normalize_reminder_dataset_alias(dataset)
        self.force_legacy = bool(force_legacy)
        self._prev_enabled = False
        self._prev_dataset: Optional[str] = None
        self._prev_legacy = False

    def __enter__(self) -> "pics_v3_prompt_robustness_scope":
        self._prev_enabled = using_pics_v3_prompt_robustness()
        self._prev_dataset = current_pics_v3_reminder_dataset()
        self._prev_legacy = using_pics_v3_legacy_generic_reminder()
        _tls.enabled = self.enabled
        _tls.dataset = self.dataset
        # Preserve process-level legacy configure unless this scope opts in.
        _tls.force_legacy = bool(self.force_legacy) or bool(self._prev_legacy)
        return self

    def __exit__(self, *exc: object) -> None:
        _tls.enabled = self._prev_enabled
        _tls.dataset = self._prev_dataset
        _tls.force_legacy = self._prev_legacy


def ensure_history_robustness_block(
    text: str,
    dataset: Optional[str] = None,
    *,
    force_legacy: Optional[bool] = None,
) -> str:
    """Insert or replace the robustness/reminder block; do not invent trial fields.

    If a block is already present, replace it with the policy-resolved block so
    new ``structure_aware_v3`` writes always match the active keyed policy
    (or legacy when forced / for non-override datasets). Unaffected datasets
    resolve to the byte-identical legacy string.
    """
    import re

    body = text or ""
    block = resolve_history_robustness_block(
        dataset, force_legacy=force_legacy
    ).strip()
    if HISTORY_ROBUSTNESS_MARKER in body:
        pat = re.compile(
            r"\[HISTORY_ROBUSTNESS_BLOCK_V3\][\s\S]*?\[/HISTORY_ROBUSTNESS_BLOCK_V3\]",
            re.M,
        )
        new_body, n = pat.subn(block, body, count=1)
        if n:
            return new_body
        return body
    return body.rstrip() + "\n\n" + block + "\n"


def maybe_attach_history_robustness_after_task_description(
    base_prompt: str,
    dataset: Optional[str] = None,
    *,
    force_legacy: Optional[bool] = None,
) -> str:
    """Attach after dataset-adaptive task text when the v3 prompt scope is active."""
    if not using_pics_v3_prompt_robustness():
        return base_prompt
    return ensure_history_robustness_block(
        base_prompt, dataset=dataset, force_legacy=force_legacy
    )


# ---------------------------------------------------------------------------
# Optional family reminder v3 (default OFF). When the sequential-RL flag is set
# for a routed dataset, this block REPLACES the keyed/generic
# HISTORY_ROBUSTNESS_BLOCK_V3 (preserving necessary v2 interface rules inside
# the v3 text). Not activated by structure_aware_v3 alone. Reminder-v2 source
# bodies in _DATASET_KEYED_REMINDER_BODIES are unchanged for default/main runs.
# ---------------------------------------------------------------------------

SEQUENTIAL_RL_REMINDER_V3_MARKER = "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V3"
SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER = "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V3_KOOL"
FEEDBACK_LEARNING_REMINDER_V3_MARKER = "FEEDBACK_LEARNING_BEHAVIOR_REMINDER_V3"

SEQUENTIAL_RL_REMINDER_V3_ALIASES = frozenset(
    {
        "steyvers_2009_bandit",
        "13schulz2020finding",
        "14kool2016when",
    }
)
# Intended feedback family (Speekenbrink / Badham). No body registered:
# FEEDBACK_REMINDER_V3_NOT_JUSTIFIED (family_prompt_v3/FEEDBACK_LEARNING_AUDIT.md).
FEEDBACK_LEARNING_REMINDER_V3_ALIASES = frozenset(
    {
        "5speekenbrink2008learning",
        "12badham2017deficits",
    }
)

# Behavioral prediction + necessary former v2 interface rules (K-way, .get/None,
# option-key mapping, anti-uniform, smoothing). Omits "reward is the primary
# learning signal" (exploit-framing bias from keyed v2).
_SEQUENTIAL_RL_BANDIT_BODY = (
    "This program predicts human choices; it does not compute a reward-maximizing "
    "policy. Even when participants are instructed to seek reward, their choices may "
    "combine exploitation of learned rewards with uncertainty-sensitive exploration, "
    "preference for relatively untried options, recency-weighted updating, "
    "perseveration or switching, and participant-specific stochasticity. In a "
    "repeated finite-horizon task, an exploratory choice may sacrifice immediate "
    "expected reward to gain information that can improve later choices. The strength "
    "and timing of this trade-off may differ across participants and trial positions. "
    "Infer from the observed training history whether exploration is weak or strong "
    "and whether it decreases, persists, or changes over the horizon; do not impose a "
    "fixed explore-then-exploit schedule. Infer a plausible combination from the "
    "observed training history; do not assume either purely greedy choice or "
    "exploration for its own sake, and do not hard-code a single true action.\n"
    "\n"
    "`history` may be empty; when present it has `action` and `reward` (use `.get`—"
    "reward may be absent or None). Treat missing or None reward as no observed "
    "outcome for that step: skip it or handle it safely, and never add None to a "
    "numeric accumulator. Map actions through the current valid option keys/"
    "`problem['options']`—do not treat action ids as raw list indices into a fixed "
    "array. Initialize every legal arm explicitly. Return a finite probability for "
    "every legal arm (full K-way dict over `option['action']`). Use positive "
    "smoothing so denominators cannot be zero; normalize safely. Use a uniform "
    "fallback only when history is empty or yields no usable reward signal—"
    "non-empty usable history must not collapse to uniform."
)

_SEQUENTIAL_RL_KOOL_BODY = (
    "This program predicts human choices in a two-step task; it does not compute a "
    "treasure-maximizing policy. Preserve the distinction between stage-1 spaceship "
    "choices and stage-2 alien choices. Human behavior may combine model-based use of "
    "learned transitions, model-free reward tracking, uncertainty-sensitive "
    "exploration, recency, perseveration or switching, and participant-specific "
    "stochasticity. Infer a plausible combination from the observed training history "
    "rather than prescribing an optimal action.\n"
    "\n"
    "Because choices repeat across a finite horizon, behavior may reflect both "
    "immediate reward and information or transition knowledge useful for later "
    "choices. Infer from the observed training history whether and how this balance "
    "changes across days and stages; do not impose a fixed exploration schedule.\n"
    "\n"
    "`history` may be empty; use `.get` for stage-conditional fields. Never read "
    "current-trial `reward`/`treasure` from `problem`. Stage 1 (`stage==1`): integer "
    "action 0/1 over spaceship `option_keys`/`spaceship_options`; learn from "
    "`stage==1` history only. Stage 2 (`stage==2`): condition on `planet`, "
    "`alien_options`/`option_keys`, and available `spaceship`/`stage1_action`; learn "
    "from `stage==2` history using `feedback`/`reward` when present. Never "
    "`option_keys.index(letter)` for actions. Treat missing or None outcomes as "
    "unobserved and never add None to numeric accumulators. Clip probs; avoid "
    "extremes from raw counts; return finite, smoothed, calibrated action "
    "probabilities."
)

_SEQUENTIAL_RL_REMINDER_V3_BLOCKS: Dict[str, str] = {
    "steyvers_2009_bandit": (
        f"[{SEQUENTIAL_RL_REMINDER_V3_MARKER}]\n"
        f"{_SEQUENTIAL_RL_BANDIT_BODY}\n"
        f"[/{SEQUENTIAL_RL_REMINDER_V3_MARKER}]"
    ),
    "13schulz2020finding": (
        f"[{SEQUENTIAL_RL_REMINDER_V3_MARKER}]\n"
        f"{_SEQUENTIAL_RL_BANDIT_BODY}\n"
        f"[/{SEQUENTIAL_RL_REMINDER_V3_MARKER}]"
    ),
    "14kool2016when": (
        f"[{SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER}]\n"
        f"{_SEQUENTIAL_RL_KOOL_BODY}\n"
        f"[/{SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER}]"
    ),
}


def using_pics_v3_sequential_rl_reminder_v3() -> bool:
    return bool(getattr(_tls, "sequential_rl_reminder_v3", False))


def using_pics_v3_feedback_learning_reminder_v3() -> bool:
    return bool(getattr(_tls, "feedback_learning_reminder_v3", False))


def configure_pics_v3_family_reminder_v3(
    *,
    sequential_rl: bool = False,
    feedback_learning: bool = False,
) -> None:
    """Process-level opt-in for family reminder v3 (both default False)."""
    _tls.sequential_rl_reminder_v3 = bool(sequential_rl)
    _tls.feedback_learning_reminder_v3 = bool(feedback_learning)


def sequential_rl_reminder_v3_block(dataset: Optional[str]) -> Optional[str]:
    """Exact tagged block for a sequential-RL dataset, or None if not routed."""
    alias = normalize_reminder_dataset_alias(dataset)
    if alias is None:
        return None
    return _SEQUENTIAL_RL_REMINDER_V3_BLOCKS.get(alias)


def feedback_learning_reminder_v3_block(dataset: Optional[str]) -> Optional[str]:
    """No body registered (audit NOT_JUSTIFIED). Always returns None."""
    del dataset  # routing checked by caller
    return None


def _family_reminder_v3_already_present(text: str) -> bool:
    return (
        SEQUENTIAL_RL_REMINDER_V3_MARKER in text
        or SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER in text
        or FEEDBACK_LEARNING_REMINDER_V3_MARKER in text
    )


def maybe_attach_family_reminder_v3(
    text: str,
    dataset: Optional[str] = None,
) -> str:
    """Replace HISTORY reminder v2 with Sequential-RL reminder v3 when flagged.

    Default (both flags false): return ``text`` unchanged (byte-identical).
    Sequential-RL flag: for Steyvers/Schulz/Kool, replace the existing
    ``[HISTORY_ROBUSTNESS_BLOCK_V3]`` with the family v3 block (complete
    replacement; necessary v2 interface rules live inside the v3 body).
    Unrelated datasets print a diagnostic and are not mutated.
    Feedback-learning flag: no body is registered — intended family datasets
    raise; unrelated datasets print a diagnostic and are not mutated.
    """
    import re

    seq = using_pics_v3_sequential_rl_reminder_v3()
    fb = using_pics_v3_feedback_learning_reminder_v3()
    if not seq and not fb:
        return text

    alias = normalize_reminder_dataset_alias(dataset)
    if alias is None:
        alias = current_pics_v3_reminder_dataset()

    if fb:
        if alias in FEEDBACK_LEARNING_REMINDER_V3_ALIASES:
            raise RuntimeError(
                "--pics_v3_feedback_learning_reminder_v3 is set but no "
                "feedback-learning reminder v3 body is registered "
                "(FEEDBACK_REMINDER_V3_NOT_JUSTIFIED; see "
                "analysis_2026Sep/Sep20_V3/others/family_prompt_v3/"
                "FEEDBACK_LEARNING_AUDIT.md). Refusing prompt mutation."
            )
        print(
            f"[TEH] --pics_v3_feedback_learning_reminder_v3 ignored for "
            f"dataset={alias!r} (not in feedback-learning family / no body); "
            f"no prompt mutation"
        )

    if not seq:
        return text

    block = sequential_rl_reminder_v3_block(alias)
    if block is None:
        print(
            f"[TEH] --pics_v3_sequential_rl_reminder_v3 ignored for "
            f"dataset={alias!r} (not in sequential-RL family); no prompt mutation"
        )
        return text

    body = text or ""
    if _family_reminder_v3_already_present(body):
        # Already replaced (or injected); do not duplicate.
        if HISTORY_ROBUSTNESS_MARKER in body:
            # Stale v2 still present alongside v3 — strip leftover v2.
            pat = re.compile(
                r"\[HISTORY_ROBUSTNESS_BLOCK_V3\][\s\S]*?\[/HISTORY_ROBUSTNESS_BLOCK_V3\]",
                re.M,
            )
            body, _ = pat.subn("", body, count=1)
            return body
        return body

    if HISTORY_ROBUSTNESS_MARKER in body:
        pat = re.compile(
            r"\[HISTORY_ROBUSTNESS_BLOCK_V3\][\s\S]*?\[/HISTORY_ROBUSTNESS_BLOCK_V3\]",
            re.M,
        )
        new_body, n = pat.subn(block, body, count=1)
        if n:
            return new_body
    return body.rstrip() + "\n\n" + block + "\n"
