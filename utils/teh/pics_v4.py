"""PICS v4: structured snapshots, one uniform prompt, slot-stable panels.

Trial text is ``format_snapshot_examples``. Dataset-keyed reminders are not
inserted. Each absolute candidate slot keeps one shuffled-block panel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from utils.teh.pics_aamas_v0 import (
    BEHAVIOR_OBJECTIVE_BLOCK,
    BEHAVIOR_OBJECTIVE_MARKER,
    CANONICAL_DATASETS,
)
from utils.teh.pics_v3_prompt_robustness import (
    HISTORY_ROBUSTNESS_MARKER,
    SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER,
    SEQUENTIAL_RL_REMINDER_V3_MARKER,
    SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER,
    _DATASET_KEYED_REMINDER_BODIES,
)

METHOD_VERSION = "pics_v4"
PICS_V4_TRIAL_POLICY = "structured_snapshot_v2"
REFUSED_TRIAL_POLICIES = frozenset(
    {
        "format_trial_for_prompt",
        "compact_faithful_trial_v1",
        "compact_faithful_trial_v2",
    }
)
# pics_run_seed orders retained trials, blocks, and participants.
# LLM requests and parent sampling stay on the historical split_seed formulas.
SEARCH_RNG_POLICY = "legacy_split_seed_formulas"
PICS_RUN_SEED_SCOPE = "trial_block_participant_schedule"
ADDITIONAL_PROMPT_POLICY_ID = "uniform_additional_prompt_v3"
PREVIOUS_ADDITIONAL_PROMPT_POLICY_ID = "uniform_additional_prompt_v2"
# Hash of uniform_additional_prompt_v2. Resume refuses this exact digest.
PREVIOUS_UNIFORM_PROMPT_SHA256 = (
    "764647b048fe35a9f05596e3e044b5861d46591c36756cacc4adee25f2240170"
)
# 15360 + llm_max_tokens 1024 = 16384, the real Qwen context limit.
HARD_PROMPT_TOKEN_CAP = 15_360
KIND_TARGET_ONLY = "pics_v4_target_only"
UNIFORM_MARKER = "PICS_V4_UNIFORM_ADDITIONAL_PROMPT"
TEXT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "teh"
    / "additional_prompt"
    / "pics_v4_uniform_additional_prompt_v3.txt"
)
PREVIOUS_TEXT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "teh"
    / "additional_prompt"
    / "pics_v4_uniform_additional_prompt_v2.txt"
)

_PREFACE = (
    "Apply only the section whose task type matches the task described above. "
    "Ignore every section that does not match."
)

# Historical generic guidance, without the old marker tags. Same bytes for every dataset.
GENERIC_HISTORY_GUIDANCE = (
    "`history` may be empty, and different history entries may contain different "
    "fields. Never assume optional fields such as `feedback`, `reward`, or "
    "outcome fields exist or are non-null. Check for a key or use `.get(...)` "
    "before reading it. Only fields explicitly required by the current "
    "task/API contract may be accessed directly."
)

_BANDIT_SHARED = """\
Sequential multi-armed bandit with repeated choices and reward feedback.
This program predicts human choices; it does not compute an optimal reward-maximizing policy. `history` may be empty; when present it has `action` and `reward` (use `.get`—reward may be absent or None). Treat missing or None reward as no observed outcome; never add None to a numeric accumulator. Return a finite probability for every legal arm (full K-way dict over every `option['action']` in `problem['options']`). Use positive smoothing; normalize safely so probabilities stay finite.

Four-armed bandit (K=4).
Across a finite horizon, participants may trade immediate reward against information useful for later choices. Exploration strength may vary by participant and trial position—do not impose a fixed schedule such as always explore early then exploit. Allowed simple participant mechanisms (use a small combination supported by observed behavior; do not implement every mechanism): smoothed reward learning; uncertainty-sensitive exploration; recency; perseveration/switching; stochastic choice. Avoid extreme probabilities and unnecessary complexity. Skip a missing or None reward or handle it safely; `.get('reward', 0)` is insufficient when the key exists with value None. Map actions through current valid option keys/`problem['options']`—do not treat action ids as raw list indices into a fixed array. Initialize every legal arm explicitly. Use a uniform fallback only when history is empty or yields no usable reward signal—non-empty usable history must not collapse to uniform.

Eight-armed bandit (K=8).
Across a finite horizon, participants may briefly trade reward for information; prefer a simple calibrated rule supported by observed behavior rather than stacking many weakly supported mechanisms. Prefer smooth probabilities over highly confident action selection. Participant stochasticity, recency, or perseveration may be used only when supported. Do not invent spatial coordinates, arm relationships, or other features absent from `problem` and `history`. Do not force nonuniform predictions merely because history is nonempty—weak or uninformative evidence may appropriately remain close to uniform."""

_process_enabled = False


def uniform_additional_prompt_text() -> str:
    """Exact block inserted for every dataset.

    ``uniform_additional_prompt_v3`` keeps the v2 task sections byte-for-byte
    and places the historical generic history paragraph once before them.
    """
    kool = _DATASET_KEYED_REMINDER_BODIES["14kool2016when"].strip()
    badham = _DATASET_KEYED_REMINDER_BODIES["12badham2017deficits"].strip()
    guan = _DATASET_KEYED_REMINDER_BODIES["guan_2020_stopping"].strip()
    body = "\n\n".join(
        (
            _BANDIT_SHARED,
            "Two-stage task involving spaceships, planets, aliens, transitions, and rewards.\n"
            + kool,
            "Feedback-based category learning involving stimulus features, category keys, "
            "rule blocks, and correctness feedback.\n"
            + badham,
            "Sequential sampling or stopping involving observed values, conditions or "
            "environments, and continue-versus-stop choices.\n"
            + guan,
        )
    )
    return (
        f"[{UNIFORM_MARKER}]\n"
        f"{_PREFACE}\n"
        f"\n"
        f"{GENERIC_HISTORY_GUIDANCE}\n"
        f"\n"
        f"{body}\n"
        f"[/{UNIFORM_MARKER}]"
    )


def uniform_additional_prompt_sha256() -> str:
    return hashlib.sha256(uniform_additional_prompt_text().encode("utf-8")).hexdigest()


def configure_pics_v4(enabled: bool) -> None:
    global _process_enabled
    _process_enabled = bool(enabled)


def using_pics_v4() -> bool:
    return bool(_process_enabled)


def _strip_tagged_block(text: str, marker: str) -> str:
    import re

    pat = re.compile(
        rf"\[{re.escape(marker)}\][\s\S]*?\[/{re.escape(marker)}\]",
        re.M,
    )
    out, _ = pat.subn("", text or "", count=1)
    return out


def apply_pics_v4_prompt_body(text: str) -> str:
    """Append the unchanged objective and the uniform block. Dataset is ignored."""
    body = text or ""
    for marker in (
        HISTORY_ROBUSTNESS_MARKER,
        SEQUENTIAL_RL_REMINDER_V3_MARKER,
        SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER,
        "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V4_STEYVERS",
        "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V4_SCHULZ",
        SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER,
        BEHAVIOR_OBJECTIVE_MARKER,
        UNIFORM_MARKER,
    ):
        body = _strip_tagged_block(body, marker)
    block = uniform_additional_prompt_text()
    return (
        body.rstrip()
        + "\n\n"
        + BEHAVIOR_OBJECTIVE_BLOCK
        + "\n\n"
        + block
        + "\n"
    )


def dataset_names_in_uniform_block() -> list[str]:
    text = uniform_additional_prompt_text().lower()
    names = list(CANONICAL_DATASETS) + [
        "steyvers",
        "schulz",
        "kool",
        "badham",
        "guan",
        "speekenbrink",
        "enkavi",
        "frey",
        "peterson",
        "plonsky",
        "wulff",
        "hilbig",
        "bergert",
        "gambles",
        "choice13k",
        "cpc18",
    ]
    return [name for name in names if name.lower() in text]


def assert_pics_v4_output(path: Path) -> None:
    text = str(Path(path))
    if f"/{KIND_TARGET_ONLY}/" not in text and not text.rstrip("/").endswith(
        f"/{KIND_TARGET_ONLY}"
    ):
        raise RuntimeError(f"PICS v4 output is not under {KIND_TARGET_ONLY}: {text}")
    for marker in (
        "/pics_v3/",
        "/pics_v3_g1/",
        "/pics_aamas_v0/",
        "/pics_aamas_v0_target_only/",
        "/pics_aamas_v0_population/",
        "/pics_v3_ablation_",
    ):
        if marker in text:
            raise RuntimeError(f"PICS v4 refusing historical path marker {marker!r}: {text}")


def _elite_policy_id() -> str:
    from utils.teh.elite_sha import ELITE_POLICY_SHA256_UNIQUE_V1

    return ELITE_POLICY_SHA256_UNIQUE_V1


def _elite_policy_fields() -> dict[str, Any]:
    from utils.teh.elite_sha import elite_policy_provenance

    return elite_policy_provenance()


def pics_v4_marker_payload() -> dict[str, Any]:
    from utils.teh.aamas_v0_lossless_trials import (
        pics_recorded_split_seed,
        pics_run_seed,
    )
    from utils.teh.pics_v4_panels import (
        CONTINUATION_POLICY_ID,
        PACKING_IMPLEMENTATION_ID,
        PANEL_BANK_POLICY_ID,
        PANEL_POLICY_ID,
        PARENT_ENVELOPE_POLICY_ID,
        SLOT_ASSIGNMENT_POLICY_ID,
    )

    payload: dict[str, Any] = {
        "method_version": METHOD_VERSION,
        "trial_prompt_policy": PICS_V4_TRIAL_POLICY,
        "prompt_policy": PICS_V4_TRIAL_POLICY,
        "additional_prompt_policy": ADDITIONAL_PROMPT_POLICY_ID,
        "additional_prompt_sha256": uniform_additional_prompt_sha256(),
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "packing_implementation": PACKING_IMPLEMENTATION_ID,
        "panel_policy": PANEL_POLICY_ID,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "panel_bank_policy": PANEL_BANK_POLICY_ID,
        "parent_envelope_policy": PARENT_ENVELOPE_POLICY_ID,
        "slot_assignment": SLOT_ASSIGNMENT_POLICY_ID,
        "search_rng": SEARCH_RNG_POLICY,
        "pics_run_seed_scope": PICS_RUN_SEED_SCOPE,
        **_elite_policy_fields(),
    }
    run_seed = pics_run_seed()
    split_seed = pics_recorded_split_seed()
    if run_seed is not None:
        payload["pics_run_seed"] = int(run_seed)
    if split_seed is not None:
        payload["split_seed"] = int(split_seed)
    return payload


def write_pics_v4_marker(output_dir: Any) -> None:
    root = Path(output_dir)
    assert_pics_v4_output(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "TRIAL_PROMPT_POLICY.json"
    if path.is_file():
        assert_pics_v4_resume(root)
        return
    path.write_text(json.dumps(pics_v4_marker_payload(), indent=2) + "\n", encoding="utf-8")


def assert_pics_v4_resume(output_dir: Any) -> None:
    """Fail closed unless the directory is this pics_v4 prompt and seed."""
    root = Path(output_dir)
    assert_pics_v4_output(root)
    path = root / "TRIAL_PROMPT_POLICY.json"
    if not path.is_file():
        from utils.teh.aamas_v0_lossless_trials import assert_legacy_output_not_resumed

        assert_legacy_output_not_resumed(root)
        if any(root.rglob("STAGE_COMPLETE.json")) or any(root.rglob("prompt_stats.json")):
            raise RuntimeError(
                f"Refusing to resume {root} as {METHOD_VERSION}: no PICS v4 policy marker."
            )
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("method_version") or "") != METHOD_VERSION:
        raise RuntimeError(
            f"Refusing to resume {root} as {METHOD_VERSION}; "
            f"method_version is {payload.get('method_version')!r}."
        )
    found_trial = str(payload.get("trial_prompt_policy") or payload.get("prompt_policy") or "")
    if found_trial in REFUSED_TRIAL_POLICIES or found_trial != PICS_V4_TRIAL_POLICY:
        raise RuntimeError(
            f"Refusing to resume {root}: trial policy {found_trial!r} "
            f"is not {PICS_V4_TRIAL_POLICY}."
        )
    found_policy = str(payload.get("additional_prompt_policy") or "")
    found = str(payload.get("additional_prompt_sha256") or "")
    if (
        found_policy == PREVIOUS_ADDITIONAL_PROMPT_POLICY_ID
        or found == PREVIOUS_UNIFORM_PROMPT_SHA256
    ):
        raise RuntimeError(
            f"Refusing to resume {root}: previous uniform prompt "
            f"{found_policy} {found}."
        )
    if found_policy != ADDITIONAL_PROMPT_POLICY_ID:
        raise RuntimeError(
            f"Refusing to resume {root}: additional prompt policy is "
            f"{found_policy!r}, not {ADDITIONAL_PROMPT_POLICY_ID}."
        )
    expected = uniform_additional_prompt_sha256()
    if found != expected:
        raise RuntimeError(
            f"Refusing to resume {root}: uniform prompt hash {found} != {expected}."
        )
    found_cap = payload.get("hard_prompt_token_cap")
    if int(found_cap or 0) != HARD_PROMPT_TOKEN_CAP:
        raise RuntimeError(
            f"Refusing to resume {root}: hard_prompt_token_cap is {found_cap!r}, "
            f"not {HARD_PROMPT_TOKEN_CAP}."
        )
    from utils.teh.pics_v4_panels import (
        CONTINUATION_POLICY_ID,
        PACKING_IMPLEMENTATION_ID,
        PANEL_BANK_POLICY_ID,
        PANEL_POLICY_ID,
        PARENT_ENVELOPE_POLICY_ID,
        REFUSED_PANEL_BANK_POLICY_IDS,
        REFUSED_PANEL_POLICY_IDS,
        REFUSED_PARENT_ENVELOPE_POLICY_IDS,
        SLOT_ASSIGNMENT_POLICY_ID,
    )

    found_pack = str(payload.get("packing_implementation") or "")
    found_panel = str(payload.get("panel_policy") or "")
    if found_pack in REFUSED_PANEL_POLICY_IDS or found_panel in REFUSED_PANEL_POLICY_IDS:
        refused = found_panel if found_panel in REFUSED_PANEL_POLICY_IDS else found_pack
        raise RuntimeError(
            f"Refusing to resume {root}: panel policy {refused} "
            f"does not implement within-block continuation. "
            f"Required panel policy is {PANEL_POLICY_ID}."
        )
    if found_pack != PACKING_IMPLEMENTATION_ID:
        raise RuntimeError(
            f"Refusing to resume {root}: packing_implementation is {found_pack!r}, "
            f"not {PACKING_IMPLEMENTATION_ID}."
        )
    if found_panel != PANEL_POLICY_ID:
        raise RuntimeError(
            f"Refusing to resume {root}: panel_policy is {found_panel!r}, "
            f"not {PANEL_POLICY_ID}."
        )
    found_bank_policy = str(payload.get("panel_bank_policy") or "")
    found_envelope = str(payload.get("parent_envelope_policy") or "")
    if found_bank_policy in REFUSED_PANEL_BANK_POLICY_IDS:
        raise RuntimeError(
            f"Refusing to resume {root}: panel bank policy {found_bank_policy} "
            f"is not {PANEL_BANK_POLICY_ID}."
        )
    if found_envelope in REFUSED_PARENT_ENVELOPE_POLICY_IDS:
        raise RuntimeError(
            f"Refusing to resume {root}: parent envelope {found_envelope} "
            "uses the longest programs. "
            f"Required parent envelope is {PARENT_ENVELOPE_POLICY_ID}."
        )
    for field_name, expected in (
        ("continuation_policy", CONTINUATION_POLICY_ID),
        ("panel_bank_policy", PANEL_BANK_POLICY_ID),
        ("parent_envelope_policy", PARENT_ENVELOPE_POLICY_ID),
    ):
        found_field = str(payload.get(field_name) or "")
        if found_field != expected:
            raise RuntimeError(
                f"Refusing to resume {root}: {field_name} is {found_field!r}, "
                f"not {expected}."
            )
    found_slots = str(payload.get("slot_assignment") or "")
    if found_slots != SLOT_ASSIGNMENT_POLICY_ID:
        raise RuntimeError(
            f"Refusing to resume {root}: slot_assignment is {found_slots!r}, "
            f"not {SLOT_ASSIGNMENT_POLICY_ID}."
        )
    from utils.teh.aamas_v0_lossless_trials import _assert_recorded_seeds_match

    _assert_recorded_seeds_match(root, payload)
    found_rng = str(payload.get("search_rng") or "")
    if found_rng != SEARCH_RNG_POLICY:
        raise RuntimeError(
            f"Refusing to resume {root}: search_rng is {found_rng!r}, "
            f"not {SEARCH_RNG_POLICY}."
        )
    found_scope = str(payload.get("pics_run_seed_scope") or "")
    if found_scope != PICS_RUN_SEED_SCOPE:
        raise RuntimeError(
            f"Refusing to resume {root}: pics_run_seed_scope is {found_scope!r}, "
            f"not {PICS_RUN_SEED_SCOPE}."
        )
    expected_policy = _elite_policy_fields()
    found_elite = str(payload.get("elite_policy") or "")
    if found_elite != expected_policy["elite_policy"]:
        raise RuntimeError(
            f"Refusing to resume {root}: elite_policy is {found_elite!r}, "
            f"not {expected_policy['elite_policy']}."
        )
    if "participant_population_unique_cap" not in payload or int(
        payload.get("participant_population_unique_cap")
    ) != int(expected_policy["participant_population_unique_cap"]):
        raise RuntimeError(
            f"Refusing to resume {root}: participant_population_unique_cap is "
            f"{payload.get('participant_population_unique_cap')!r}, "
            f"not {expected_policy['participant_population_unique_cap']}."
        )
    if payload.get("elite_pool_size_is_upper_bound") is not True:
        raise RuntimeError(
            f"Refusing to resume {root}: elite_pool_size_is_upper_bound is "
            f"{payload.get('elite_pool_size_is_upper_bound')!r}."
        )
    if payload.get("duplicate_backfill") is not False:
        raise RuntimeError(
            f"Refusing to resume {root}: duplicate_backfill is "
            f"{payload.get('duplicate_backfill')!r}."
        )
