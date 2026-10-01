"""PICS v4: compact_faithful_trial_v1 plus one uniform additional prompt.

The trial serializer is unchanged. This module only replaces dataset-keyed
reminder selection with one block appended for every dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from utils.teh.aamas_v0_lossless_trials import TRIAL_PROMPT_POLICY_ID
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
ADDITIONAL_PROMPT_POLICY_ID = "uniform_additional_prompt_v2"
KIND_TARGET_ONLY = "pics_v4_target_only"
UNIFORM_MARKER = "PICS_V4_UNIFORM_ADDITIONAL_PROMPT"
TEXT_PATH = (
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

    ``uniform_additional_prompt_v2`` keeps the Kool, Badham, and Guan bodies
    byte-for-byte and states the shared sequential-bandit rules once.
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


def pics_v4_marker_payload() -> dict[str, Any]:
    from utils.teh.aamas_v0_lossless_trials import (
        PACKING_IMPLEMENTATION_ID,
        pics_recorded_split_seed,
        pics_run_seed,
    )

    payload: dict[str, Any] = {
        "method_version": METHOD_VERSION,
        "trial_prompt_policy": TRIAL_PROMPT_POLICY_ID,
        "prompt_policy": TRIAL_PROMPT_POLICY_ID,
        "additional_prompt_policy": ADDITIONAL_PROMPT_POLICY_ID,
        "additional_prompt_sha256": uniform_additional_prompt_sha256(),
        "packing_implementation": PACKING_IMPLEMENTATION_ID,
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
    if str(payload.get("trial_prompt_policy") or payload.get("prompt_policy") or "") != TRIAL_PROMPT_POLICY_ID:
        raise RuntimeError(
            f"Refusing to resume {root}: trial policy is not {TRIAL_PROMPT_POLICY_ID}."
        )
    found_policy = str(payload.get("additional_prompt_policy") or "")
    if found_policy != ADDITIONAL_PROMPT_POLICY_ID:
        raise RuntimeError(
            f"Refusing to resume {root}: additional prompt policy is "
            f"{found_policy!r}, not {ADDITIONAL_PROMPT_POLICY_ID}."
        )
    expected = uniform_additional_prompt_sha256()
    found = str(payload.get("additional_prompt_sha256") or "")
    if found != expected:
        raise RuntimeError(
            f"Refusing to resume {root}: uniform prompt hash {found} != {expected}."
        )
    from utils.teh.aamas_v0_lossless_trials import _assert_recorded_seeds_match

    _assert_recorded_seeds_match(root, payload)
