"""PICS v4: sparse-history snapshots, one uniform prompt, slot-stable panels.

Trial text is ``format_sparse_history_examples``
(``structured_snapshot_sparse_history_v1``). Dataset-keyed reminders are not
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
)

METHOD_VERSION = "pics_v4"
PICS_V4_TRIAL_POLICY = "structured_snapshot_sparse_history_v1"
REFUSED_TRIAL_POLICIES = frozenset(
    {
        "format_trial_for_prompt",
        "compact_faithful_trial_v1",
        "compact_faithful_trial_v2",
        "structured_snapshot_v2",
    }
)
# pics_run_seed orders retained trials, blocks, and participants.
# LLM requests and parent sampling stay on the historical split_seed formulas.
SEARCH_RNG_POLICY = "legacy_split_seed_formulas"
PICS_RUN_SEED_SCOPE = "trial_block_participant_schedule"
ADDITIONAL_PROMPT_POLICY_ID = "uniform_additional_prompt_v8"
PREVIOUS_ADDITIONAL_PROMPT_POLICY_ID = "uniform_additional_prompt_v2"
# Hash of uniform_additional_prompt_v2. Resume refuses this exact digest.
PREVIOUS_UNIFORM_PROMPT_SHA256 = (
    "764647b048fe35a9f05596e3e044b5861d46591c36756cacc4adee25f2240170"
)
# Hash of uniform_additional_prompt_v3, the full-history block without calibration text.
UNIFORM_V3_PROMPT_SHA256 = (
    "12d8a94bf524ce1b985e4bf4b08c08c8d78e5a7f8a7437026516486433416e5b"
)
# Hash of uniform_additional_prompt_v4, which added the sparse-history paragraph.
SPARSE_UNIFORM_PROMPT_SHA256 = (
    "342297e3d814b8dcb565d31ac2b71a63f29d0b79e37e8aceb16c2df2240a9930"
)
# Draft v5 hashes from the calibration audit. They were never submitted.
OBSOLETE_V5_PROMPT_SHA256 = frozenset(
    {
        "c045321e6222ee7f968b9ca71e01da30f80db8b69c333128212a1839bfd38964",
        "52bc0a340b5f17760f061ae64c290d8ac76aee7ab3d68cd228ab6857bed609f1",
    }
)
# Submitted uniform v5. Its jobs were cancelled. Resume refuses this digest.
UNIFORM_V5_PROMPT_SHA256 = (
    "6df73ea2757bfbf4c4f80e0fc8a6098b3d9a5a8466180917421ddeef7905dabc"
)
# Draft v6 hash registered before the override-sentence audit. Never submitted.
OBSOLETE_V6_PROMPT_SHA256 = frozenset(
    {
        "8a84c26a191dabd1e047280a9489797ef3b20295df2efd7041901fc5a6694ebd",
    }
)
# Submitted uniform v6. Its jobs were cancelled. Resume refuses this digest.
UNIFORM_V6_PROMPT_SHA256 = (
    "3edb24f80e251b145493821018956100479ba812dc41b68bdd8004663c0b5612"
)
# Submitted uniform v7. Resume refuses this digest.
UNIFORM_V7_PROMPT_SHA256 = (
    "470363dcb340a3c96a7a33b937499a9d314cf635aed11bbe34f4b6cc7e754a9d"
)
UNIFORM_PROMPT_SHA256 = (
    "9663c810440fcb295bcf046afea31ca3f527e310c5afe61d66264943afb1470b"
)
REFUSED_ADDITIONAL_PROMPT_POLICY_IDS = frozenset(
    {
        "uniform_additional_prompt_v2",
        "uniform_additional_prompt_v3",
        "uniform_additional_prompt_v4",
        "uniform_additional_prompt_v5",
        "uniform_additional_prompt_v6",
        "uniform_additional_prompt_v7",
    }
)
REFUSED_UNIFORM_PROMPT_SHA256 = frozenset(
    {
        PREVIOUS_UNIFORM_PROMPT_SHA256,
        UNIFORM_V3_PROMPT_SHA256,
        SPARSE_UNIFORM_PROMPT_SHA256,
        UNIFORM_V5_PROMPT_SHA256,
        UNIFORM_V6_PROMPT_SHA256,
        UNIFORM_V7_PROMPT_SHA256,
        *OBSOLETE_V5_PROMPT_SHA256,
        *OBSOLETE_V6_PROMPT_SHA256,
    }
)
# 15360 + llm_max_tokens 1024 = 16384, the real Qwen context limit.
HARD_PROMPT_TOKEN_CAP = 15_360
KIND_TARGET_ONLY = "pics_v4_target_only"
KIND_TRANSFER_BASED_ONLY = "pics_v4_transfer_based_only"
KIND_OFFICIAL_GATE = "pics_v4_official_gate"
UNIFORM_MARKER = "PICS_V4_UNIFORM_ADDITIONAL_PROMPT"
TEXT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "teh"
    / "additional_prompt"
    / "pics_v4_uniform_additional_prompt_v8.txt"
)
PREVIOUS_TEXT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "teh"
    / "additional_prompt"
    / "pics_v4_uniform_additional_prompt_v2.txt"
)

SPARSE_HISTORY_CLARIFICATION = (
    "`PROMPT_ONLY_METADATA`, `history_entries_omitted`, "
    "`actual_runtime_history_length`, and `displayed_history_entries` are "
    "prompt annotations only. They never exist in the runtime `problem` or "
    "`history`, and generated code must not access them."
)
HARD_PROBABILITY_RULE = (
    "MANDATORY PROBABILITY OUTPUT RULE — NO EXCEPTIONS.\n\n"
    "The generated program is incorrect unless every possible return path from "
    "`choose(problem, history)` assigns every legal action a probability inside "
    "the closed interval [0.01, 0.99]. This requirement applies on every trial, "
    "including terminal trials, regardless of the retained examples, "
    "train/validation likelihood, apparent certainty, task-specific instructions, "
    "or parent/reference programs.\n\n"
    "For binary tasks, compute any raw score or raw probability first, but every "
    "return path must finally return "
    "`min(0.99, max(0.01, float(raw_probability)))`.\n\n"
    "For binary tasks, an affine expression such as `0.01 + 0.98 * x` is not "
    "an accepted substitute for the final clamp, even if the program assumes "
    "that `x` lies in [0, 1]. Do not rely on assumptions about the sign or "
    "range of any intermediate value. Every binary return path must either "
    "literally return `min(0.99, max(0.01, float(raw_probability)))`, or "
    "return a variable that was assigned by that exact clamp immediately "
    "before the return.\n\n"
    "The following binary code patterns are strictly prohibited:\n\n"
    "- `return 0` or `return 1`;\n"
    "- returning any literal below 0.01 or above 0.99;\n"
    "- returning a raw sigmoid, softmax component, empirical frequency, threshold "
    "result, score, or probability variable without applying the final "
    "[0.01, 0.99] bound;\n"
    "- an early return from any branch that bypasses the final bound.\n\n"
    "Do not use `1e-6`, `1-1e-6`, or any value outside [0.01, 0.99] as a "
    "returned probability or final probability bound. Small epsilon constants "
    "may be used only for internal numerical safety, such as preventing "
    "division by zero, `log(0)`, overflow, underflow, or a zero normalization "
    "denominator. An internal epsilon is not an output probability and never "
    "replaces the mandatory final [0.01, 0.99] transformation.\n\n"
    "For categorical tasks with K legal actions, every return path must include "
    "every legal action. First construct a finite nonnegative normalized "
    "distribution `q`, then return `0.01 + (1 - 0.01 * K) * q[action]` for "
    "every legal action. The returned dictionary must sum to 1. Returning raw "
    "or merely normalized categorical probabilities without this transformation "
    "is strictly prohibited. If the finite nonnegative raw weights sum to zero, "
    "use the uniform distribution `q[action] = 1 / K` before applying "
    "`0.01 + (1 - 0.01 * K) * q[action]`.\n\n"
    "Before outputting the program, inspect every `return` statement in "
    "`choose`. If any return path can bypass these rules, revise the code "
    "before responding. Parent or reference programs may violate this rule; "
    "never copy their probability-return code.\n\n"
    "This rule is absolute. There are no forced-action, terminal-action, "
    "high-score, strong-evidence, zero-count, or task-specific exceptions."
)
_INTERNAL_EPSILON_ALLOWANCE = (
    "Do not use `1e-6`, `1-1e-6`, or any value outside [0.01, 0.99] as a "
    "returned probability or final probability bound."
)

# Historical generic guidance, without the old marker tags. Same bytes for every dataset.
GENERIC_HISTORY_GUIDANCE = (
    "`history` may be empty, and different history entries may contain different "
    "fields. Never assume optional fields such as `feedback`, `reward`, or "
    "outcome fields exist or are non-null. Check for a key or use `.get(...)` "
    "before reading it. Only fields explicitly required by the current "
    "task/API contract may be accessed directly."
)

_process_enabled = False


def uniform_additional_prompt_text() -> str:
    """Exact ``uniform_additional_prompt_v8`` block inserted for every dataset."""
    text = TEXT_PATH.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != UNIFORM_PROMPT_SHA256:
        raise RuntimeError(
            f"uniform_additional_prompt_v8 hash {digest} != {UNIFORM_PROMPT_SHA256}."
        )
    if text.count(SPARSE_HISTORY_CLARIFICATION) != 1:
        raise RuntimeError("uniform v8 sparse-history clarification is missing or repeated.")
    if text.count(HARD_PROBABILITY_RULE) != 1:
        raise RuntimeError("uniform v8 hard probability rule is missing or repeated.")
    if "The only exception" in text or "position == sequence_length" in text:
        raise RuntimeError("uniform v8 still contains a probability-rule exception.")
    if "only remaining" in text:
        raise RuntimeError("uniform v8 still contains a terminal-action exception.")
    if "1e-6" in text.replace(_INTERNAL_EPSILON_ALLOWANCE, ""):
        raise RuntimeError(
            "uniform v8 uses 1e-6 outside the internal-epsilon allowance."
        )
    return text


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


_PICS_V4_INSTRUCTION_REWRITES = (
    (
        "- The return value must be strictly inside (0, 1). If needed, clip to a safe range such as [1e-6, 1 - 1e-6].",
        "- The return value must lie in [0.01, 0.99]. Every return path must finally return min(0.99, max(0.01, float(raw_probability))).",
    ),
    (
        "and continue returning a valid probability in (0, 1).",
        "and continue returning a valid probability in [0.01, 0.99].",
    ),
    (
        "- Probabilities must be finite and non-negative (prefer summing to 1.0).",
        "- Each legal-action probability must lie in [0.01, 0.99], and the returned dictionary must sum to 1.",
    ),
    (
        "- Output: a single finite float P(action=1) strictly inside (0, 1); clip to [1e-6, 1-1e-6] if needed.",
        "- Output: a single finite float P(action=1) inside [0.01, 0.99]. Every return path must finally return min(0.99, max(0.01, float(raw_probability))).",
    ),
    (
        "returning float in (0, 1) as P(action=1).",
        "returning float in [0.01, 0.99] as P(action=1).",
    ),
)


def _rewrite_pics_v4_probability_instructions(text: str) -> str:
    """Replace active clip instructions. Parent source is appended after this step."""
    out = text or ""
    for old, new in _PICS_V4_INSTRUCTION_REWRITES:
        out = out.replace(old, new)
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
    body = _rewrite_pics_v4_probability_instructions(body)
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
    from utils.teh.pics_v4_transfer import KINDS, current_identity
    identity = current_identity()
    expected_kind = identity['kind'] if identity else KIND_TARGET_ONLY
    if identity:
        text = str(Path(path).resolve())
    if identity and f"/{identity['target']}/{expected_kind}/" not in text:
        raise RuntimeError('PICS v4 output target/mode mismatch')
    if f"/{expected_kind}/" not in text and not text.rstrip('/').endswith(f"/{expected_kind}"):
        raise RuntimeError(f"PICS v4 output is not under {expected_kind}: {text}")
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
    assert_pics_v4_resume(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "TRIAL_PROMPT_POLICY.json"
    if path.is_file():
        assert_pics_v4_resume(root)
        return
    from utils.teh.pics_v4_transfer import current_identity
    payload = pics_v4_marker_payload()
    if current_identity() is not None:
        payload['run_identity'] = current_identity()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def assert_pics_v4_resume(output_dir: Any, *, expected_identity=None) -> None:
    """Fail closed unless the directory is this pics_v4 prompt and seed."""
    root = Path(output_dir)
    from utils.teh.pics_v4_transfer import current_identity, assert_identity
    expected = expected_identity or current_identity()
    if expected_identity is not None:
        if f"/{expected_identity['kind']}/" not in str(root):
            raise RuntimeError('PICS v4 materialization KIND mismatch')
    else:
        assert_pics_v4_output(root)
    path = root / "TRIAL_PROMPT_POLICY.json"
    if not path.is_file():
        from utils.teh.aamas_v0_lossless_trials import assert_legacy_output_not_resumed

        assert_legacy_output_not_resumed(root)
        if expected is not None and root.exists() and any(p.name not in {'OUTPUT_DIR.txt', 'TRACK_STATUS.txt', 'INTENDED_ARGV.txt'} for p in root.iterdir()):
            raise RuntimeError('refusing unbound partial transfer/gate output')
        if any(root.rglob("STAGE_COMPLETE.json")) or any(root.rglob("prompt_stats.json")):
            raise RuntimeError(
                f"Refusing to resume {root} as {METHOD_VERSION}: no PICS v4 policy marker."
            )
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected is not None:
        from utils.teh.pics_v3 import G2_PAIRED_PACK_FILENAME
        if (root / G2_PAIRED_PACK_FILENAME).exists() or (root / 'gate' / G2_PAIRED_PACK_FILENAME).exists():
            raise RuntimeError('v4 transfer refuses historical paired-pack freezes')
        assert_identity(payload.get('run_identity'), expected)
        from utils.teh.pics_v4_transfer import assert_participant_resume, configure_identity, source_suffix
        saved_identity = current_identity()
        if expected_identity is not None:
            saved_suffix = source_suffix()
            configure_identity(expected, saved_suffix)
        try:
            from utils.teh.pics_v4_recovery import participant_directories
            for participant_path in participant_directories(root / 'selected'):
                assert_participant_resume(participant_path, int(participant_path.name.split('_')[-1]))
        finally:
            if expected_identity is not None:
                configure_identity(saved_identity, saved_suffix)
        for record_path in (root / 'selected/STAGE_COMPLETE.json' , root / 'gate/gate_record.json'):
            if record_path.is_file():
                assert_identity(json.loads(record_path.read_text()).get('run_identity'), expected)
    elif payload.get('run_identity') or any(payload.get(k) for k in ('source_dataset', 'source_program_sha256', 'source_map_sha256')):
        raise RuntimeError('target-only marker contains transfer provenance')
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
        found_policy in REFUSED_ADDITIONAL_PROMPT_POLICY_IDS
        or found in REFUSED_UNIFORM_PROMPT_SHA256
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
        REFUSED_PARENT_ENVELOPE_POLICY_IDS,
        SLOT_ASSIGNMENT_POLICY_ID,
        refuse_historical_panel_policy,
    )

    found_pack = str(payload.get("packing_implementation") or "")
    found_panel = str(payload.get("panel_policy") or "")
    found_continuation = str(payload.get("continuation_policy") or "")
    refuse_historical_panel_policy(
        found_panel,
        found_pack,
        found_continuation,
        where=str(root),
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
