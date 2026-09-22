"""PICS v3 post-adaptive history reminder (structure_aware_v3 only).

Default / legacy: a single generic HISTORY_ROBUSTNESS_BLOCK_V3 for all datasets.

Prospective policy ``dataset_keyed_post_adaptive_v2``: for named datasets only,
replace that block body with a concise dataset-keyed reminder. All other datasets
keep the byte-identical legacy block.

v2 revises Steyvers / Badham / Speekenbrink guidance and adds CPC18
(``2plonsky2018when``). Kool / Schulz / Guan bodies stay byte-identical to v1.
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
        "2plonsky2018when",
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
    "2plonsky2018when": (
        "`history` may be empty; use `.get` for optional fields. "
        "Within each problem, history is chronological over up to 25 repeats.\n"
        "Structured history exposes `action` and, when present, a scalar "
        "`feedback` = the realized payoff of the **chosen** option only. "
        "Do not assume a forgone/unchosen outcome is available in `history` "
        "(it is not represented). Early trials often have `feedback is None`; "
        "later trials may include chosen-option feedback. "
        "`problem['has_feedback']` is block-level metadata—prefer checking "
        "`history[i].get('feedback')` for the actual regime. "
        "Combine description (`gamble_*` probs/rewards) with experiential "
        "chosen feedback when present. Return calibrated P(action=1)."
    ),
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
