"""PICS AAMAS v0 — isolated method (fresh population bank + transfer pipeline).

Does not change historical PICS v3, ablation A–J, FGH, or reminder-v1/v2/v3/v4
rendering. Reminder bodies are reused by reference; this module only chooses
which existing body, if any, is attached under policy ``pics_aamas_v0``.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.teh.aamas_v0_lossless_trials import (
    TRIAL_PROMPT_POLICY_ID,
    pics_recorded_split_seed,
    pics_run_seed,
)

_tls = threading.local()
_process_enabled = False

PROMPT_POLICY_ID = "pics_aamas_v0"
# Superseded KINDs. Launchers that write these must fail closed.
KIND_POPULATION = "pics_aamas_v0_population"
KIND_DOWNSTREAM = "pics_aamas_v0"
KIND_TARGET_ONLY = "pics_aamas_v0_target_only"
KIND_TRANSFER_BASED_ONLY = "pics_aamas_v0_transfer_based_only"
KIND_OFFICIAL_GATE = "pics_aamas_v0_official_gate"
TRACK_TARGET_ONLY = "target_only"
TRACK_TRANSFER_BASED_ONLY = "transfer_based_only"
TRACK_OFFICIAL_GATE = "official_gate"
TRACK_MODES = (TRACK_OFFICIAL_GATE, TRACK_TARGET_ONLY, TRACK_TRANSFER_BASED_ONLY)
KIND_BY_TRACK = {
    TRACK_TARGET_ONLY: KIND_TARGET_ONLY,
    TRACK_TRANSFER_BASED_ONLY: KIND_TRANSFER_BASED_ONLY,
    TRACK_OFFICIAL_GATE: KIND_OFFICIAL_GATE,
}
WANDB_PROJECT = "teh_pics_aamas_v0"
WANDB_GROUP_POPULATION = "pics_aamas_v0_population"
WANDB_GROUP_DOWNSTREAM = "pics_aamas_v0"
POPULATION_STAGE = "pics_aamas_v0_target_population"
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
VLLM_MAX_MODEL_LEN = 16384
LLM_MAX_TOKENS = 1024
BEHAVIOR_OBJECTIVE_MARKER = "AAMAS_V0_BEHAVIOR_OBJECTIVE"
HARD_PROMPT_TOKEN_CAP = 14000
PARENT_CHAR_CAP = 5000
PAIRED_PARENT_RESERVE = 8
GLOBAL_ITERS = 10
N_CANDIDATES = 10
EXPLORE_CANDIDATES = 50
PERSON_ITERS = 10
SOURCE_SELECTOR = "occurrence_eb_schema5_iter10_pics_v3"

CANONICAL_DATASETS: tuple[str, ...] = (
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "12badham2017deficits",
    "mixed_gambles",
    "bergert_nosofsky_2007",
    "guan_2020_stopping",
    "steyvers_2009_bandit",
    "13schulz2020finding",
    "14kool2016when",
)

POPULATION_PACKS: Mapping[str, tuple[str, ...]] = {
    "pack1": (
        "2plonsky2018when",
        "bergert_nosofsky_2007",
        "5speekenbrink2008learning",
        "11enkavi2019recentprobes",
    ),
    "pack2": (
        "10frey2017risk",
        "14kool2016when",
        "7hilbig2014generalized",
    ),
    "pack3": (
        "3frey2017cct",
        "4wulff2018description",
        "13schulz2020finding",
        "12badham2017deficits",
    ),
    "pack4": (
        "1peterson2021using",
        "mixed_gambles",
        "guan_2020_stopping",
        "steyvers_2009_bandit",
    ),
}

# Output-path fragments that must never be loaded as an AAMAS source program.
_FORBIDDEN_PATH_MARKERS = (
    "/pics_v3/",
    "/pics_v3_g1/",
    "/pics_v3_independent/",
    "/pics_v3_ablation_",
    "/pics_v3_allocation_",
    "/pics_v3_family_prompt_",
    "/pics_v3_reminder_",
    "/pics_v3_centaur_",
    "/t_pics_source_pop",
    "/pics_aamas_v0_population/",
    "/pics_aamas_v0/",
)

BEHAVIOR_OBJECTIVE_BLOCK = (
    f"[{BEHAVIOR_OBJECTIVE_MARKER}]\n"
    "Predict observed participant choices and maximize behavioral log-likelihood. "
    "Do not treat the task as merely computing an optimal reward policy.\n"
    f"[/{BEHAVIOR_OBJECTIVE_MARKER}]"
)


def configure_pics_aamas_v0_prompt(enabled: bool) -> None:
    """Process-level opt-in. Default off, so historical generation is unchanged.

    The flag is process-wide. Participant evolution runs in worker threads, and
    a thread-local flag would send those threads back to the one-line serializer.
    """
    global _process_enabled
    _process_enabled = bool(enabled)
    _tls.enabled = bool(enabled)


def using_pics_aamas_v0_prompt() -> bool:
    return bool(_process_enabled or getattr(_tls, "enabled", False))


@dataclass(frozen=True)
class ReminderDecision:
    dataset: str
    reminder_id: str
    """``none`` or a stable identity for the attached block."""
    block: Optional[str]
    """Exact tagged block, or None when AAMAS v0 attaches no separate reminder."""


def reminder_decision(dataset: Optional[str]) -> ReminderDecision:
    """Frozen AAMAS v0 reminder routing. Does not edit historical bodies."""
    from utils.teh.pics_v3_prompt_robustness import (
        dataset_keyed_reminder_body,
        sequential_rl_reminder_v4_block,
        wrap_history_robustness_body,
    )

    alias = str(dataset or "").strip()
    if alias in ("steyvers_2009_bandit", "13schulz2020finding"):
        block = sequential_rl_reminder_v4_block(alias)
        if not block:
            raise RuntimeError(f"missing Sequential-RL v4 block for {alias}")
        suffix = "steyvers" if alias.startswith("steyvers") else "schulz"
        return ReminderDecision(
            dataset=alias,
            reminder_id=f"sequential_rl_reminder_v4_{suffix}",
            block=block,
        )
    if alias == "14kool2016when":
        body = dataset_keyed_reminder_body(alias)
        if not body:
            raise RuntimeError("missing Kool keyed reminder body")
        return ReminderDecision(
            dataset=alias,
            reminder_id="dataset_keyed_kool_v1",
            block=wrap_history_robustness_body(body),
        )
    if alias == "12badham2017deficits":
        body = dataset_keyed_reminder_body(alias)
        if not body:
            raise RuntimeError("missing Badham keyed reminder body")
        return ReminderDecision(
            dataset=alias,
            reminder_id="dataset_keyed_badham_v2",
            block=wrap_history_robustness_body(body),
        )
    if alias == "guan_2020_stopping":
        body = dataset_keyed_reminder_body(alias)
        if not body:
            raise RuntimeError("missing Guan keyed reminder body")
        return ReminderDecision(
            dataset=alias,
            reminder_id="dataset_keyed_guan",
            block=wrap_history_robustness_body(body),
        )
    return ReminderDecision(dataset=alias, reminder_id="none", block=None)


def _strip_tagged_block(text: str, marker: str) -> str:
    import re

    pat = re.compile(
        rf"\[{re.escape(marker)}\][\s\S]*?\[/{re.escape(marker)}\]",
        re.M,
    )
    out, _ = pat.subn("", text or "", count=1)
    return out


def apply_aamas_v0_prompt_body(text: str, dataset: Optional[str]) -> str:
    """Insert the shared objective and at most one dataset reminder.

    Strips any pre-existing history / sequential-RL / objective blocks so a
    reminder cannot stack. Does not modify the registered description.
    """
    from utils.teh.pics_v3_prompt_robustness import (
        HISTORY_ROBUSTNESS_MARKER,
        SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER,
        SEQUENTIAL_RL_REMINDER_V3_MARKER,
        SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER,
        SEQUENTIAL_RL_REMINDER_V4_SCHULZ_MARKER,
        SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER,
    )

    body = text or ""
    for marker in (
        HISTORY_ROBUSTNESS_MARKER,
        SEQUENTIAL_RL_REMINDER_V3_MARKER,
        SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER,
        SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER,
        SEQUENTIAL_RL_REMINDER_V4_SCHULZ_MARKER,
        SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER,
        BEHAVIOR_OBJECTIVE_MARKER,
    ):
        body = _strip_tagged_block(body, marker)
    decision = reminder_decision(dataset)
    parts = [body.rstrip(), "", BEHAVIOR_OBJECTIVE_BLOCK]
    if decision.block:
        parts.extend(["", decision.block])
    return "\n".join(parts).rstrip() + "\n"


def char4_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def population_parent_reserve_tokens() -> int:
    return PAIRED_PARENT_RESERVE * char4_tokens("x" * PARENT_CHAR_CAP)


def prompt_budget_report(infer_text: str) -> Dict[str, Any]:
    """Static packing report for a rendered infer prompt (char4)."""
    import re

    tokens = char4_tokens(infer_text)
    reserve = population_parent_reserve_tokens()
    m = re.search(
        r"showing (\d+) complete train trial example",
        infer_text or "",
    )
    n_inlined = int(m.group(1)) if m else 0
    over_base = tokens > HARD_PROMPT_TOKEN_CAP
    overflow = max(0, tokens + reserve - HARD_PROMPT_TOKEN_CAP)
    return {
        "infer_char4_tokens": tokens,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "parent_reserve_tokens": reserve,
        "base_under_cap": not over_base,
        "headroom_after_parent_reserve": HARD_PROMPT_TOKEN_CAP - tokens - reserve,
        "extra_trials_displaced_by_parent_reserve": overflow > 0,
        "displaced_token_estimate": overflow,
        "n_inlined_examples": n_inlined,
    }


def population_run_root(
    repo_root: Path,
    dataset: str,
    *,
    psych_split: str = "train",
    kind: str = KIND_TARGET_ONLY,
) -> Path:
    return (
        Path(repo_root)
        / "generated_outputs"
        / f"psych101_{psych_split}"
        / "teh"
        / str(dataset)
        / str(kind)
    )


def assert_aamas_population_path(path: Path) -> None:
    text = str(Path(path).resolve())
    if f"/{KIND_TARGET_ONLY}/" not in text and not text.rstrip("/").endswith(
        f"/{KIND_TARGET_ONLY}"
    ):
        raise RuntimeError(
            f"AAMAS v0 source path is not under {KIND_TARGET_ONLY}: {text}"
        )
    for marker in _FORBIDDEN_PATH_MARKERS:
        if marker in text:
            raise RuntimeError(
                f"AAMAS v0 refusing historical/ablation path marker {marker!r}: {text}"
            )


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def assert_aamas_track_output(path: Path, kind: str) -> None:
    text = str(Path(path).resolve())
    if f"/{kind}/" not in text and not text.rstrip("/").endswith(f"/{kind}"):
        raise RuntimeError(f"AAMAS v0 path is not under {kind}: {text}")
    for marker in _FORBIDDEN_PATH_MARKERS:
        if marker in text:
            raise RuntimeError(
                f"AAMAS v0 refusing historical/ablation path marker {marker!r}: {text}"
            )


def write_population_completion(
    run_dir: Path,
    *,
    dataset: str,
    prompt_meta: Mapping[str, Any],
    global_iters: int,
    n_candidates: int,
    range_start_ordinal: Optional[int] = None,
    range_end_ordinal: Optional[int] = None,
    track_mode: str = TRACK_TARGET_ONLY,
    kind: Optional[str] = None,
    model_name: str = MODEL_NAME,
    hard_prompt_token_cap: int = HARD_PROMPT_TOKEN_CAP,
    llm_max_tokens: int = LLM_MAX_TOKENS,
    max_parent_chars: int = PARENT_CHAR_CAP,
    vllm_max_model_len: int = VLLM_MAX_MODEL_LEN,
    source_dataset: Optional[str] = None,
    source_rank1_sha256: Optional[str] = None,
    provisional: Optional[bool] = None,
) -> Path:
    """Write provenance + strict STAGE_COMPLETE.json after a successful population.

    Refuses to mark complete when the rank-1 program is missing or invalid.
    """
    from utils.teh.t_pics_gated_transfer import program_has_valid_choose

    run_dir = Path(run_dir)
    best = run_dir / "global_phase" / "best_program.py"
    results_path = run_dir / "global_phase" / "results.json"
    if not best.is_file():
        raise RuntimeError(f"AAMAS v0 population missing rank-1: {best}")
    if not results_path.is_file():
        raise RuntimeError(f"AAMAS v0 population missing results: {results_path}")
    resolved_kind = str(kind or kind_for_track(track_mode))
    if provisional is None:
        provisional = track_mode != TRACK_OFFICIAL_GATE
    assert_aamas_track_output(run_dir, resolved_kind)
    results = _read_json(results_path)
    valid, valid_reason = program_has_valid_choose(best)
    program_bytes = best.read_bytes()
    program_sha = hashlib.sha256(program_bytes).hexdigest()
    from utils.teh.pics_v4 import using_pics_v4

    decision = reminder_decision(dataset)
    policy_id = TRIAL_PROMPT_POLICY_ID
    if using_pics_v4():
        from utils.teh.pics_v4 import PICS_V4_TRIAL_POLICY

        policy_id = PICS_V4_TRIAL_POLICY
    reminder_id = decision.reminder_id
    if using_pics_v4():
        from utils.teh.pics_v4 import (
            ADDITIONAL_PROMPT_POLICY_ID,
            METHOD_VERSION,
            uniform_additional_prompt_sha256,
        )

        reminder_id = ADDITIONAL_PROMPT_POLICY_ID
    score = results.get("pool_best_selection_score")
    provenance: Dict[str, Any] = {
        "schema": "pics_aamas_v0_population_provenance_v1",
        "dataset": str(dataset),
        "kind": resolved_kind,
        "track_mode": str(track_mode),
        "provisional_experimental_track": bool(provisional),
        "official_main_result": track_mode == TRACK_OFFICIAL_GATE and not provisional,
        "prompt_policy": policy_id,
        "trial_prompt_policy": policy_id,
        "prompt_mode": prompt_meta.get("prompt_mode"),
        "rendered_prompt_sha256": prompt_meta.get("infer_prompt_sha256"),
        "reminder_id": reminder_id,
        "sa40_fingerprint": prompt_meta.get("observed_subset_fingerprint"),
        "observed_train_fingerprint": prompt_meta.get("observed_train_fingerprint"),
        "observed_val_fingerprint": prompt_meta.get("observed_val_fingerprint"),
        "participant_ids": results.get("participant_ids"),
        "range_start_ordinal": range_start_ordinal,
        "range_end_ordinal": range_end_ordinal,
        "limited_data_protocol": prompt_meta.get("limited_data_protocol"),
        "limited_train_val": prompt_meta.get("limited_train_val"),
        "global_iters": int(global_iters),
        "n_candidates": int(n_candidates),
        "evolution_selection_score": results.get("evolution_selection_score"),
        "rank1_program": str(best.resolve()),
        "rank1_sha256": program_sha,
        "pooled_train_val_score": score,
        "runtime_valid": bool(valid),
        "runtime_valid_reason": valid_reason,
        "test_used_for_selection": False,
        "model_name": str(model_name),
        "hard_prompt_token_cap": int(hard_prompt_token_cap),
        "llm_max_tokens": int(llm_max_tokens),
        "max_parent_chars": int(max_parent_chars),
        "vllm_max_model_len": int(vllm_max_model_len),
        "source_dataset": source_dataset,
        "source_rank1_sha256": source_rank1_sha256,
        "arm_role": "transfer" if source_dataset else "target_only",
        "split_seed": pics_recorded_split_seed(),
        "pics_run_seed": pics_run_seed(),
    }
    if using_pics_v4():
        from utils.teh.pics_v4 import PICS_RUN_SEED_SCOPE, SEARCH_RNG_POLICY

        provenance["method_version"] = METHOD_VERSION
        provenance["additional_prompt_policy"] = ADDITIONAL_PROMPT_POLICY_ID
        provenance["additional_prompt_sha256"] = uniform_additional_prompt_sha256()
        from utils.teh.elite_sha import elite_policy_provenance

        provenance["search_rng"] = SEARCH_RNG_POLICY
        provenance["pics_run_seed_scope"] = PICS_RUN_SEED_SCOPE
        provenance.update(elite_policy_provenance())
    from utils.teh.pics_v4_transfer import current_identity, assert_arm_resume
    identity = current_identity() if using_pics_v4() else None
    if identity is not None:
        role = 'transfer' if source_dataset else 'control'
        if role == 'transfer' and (source_dataset != identity['source']['source_dataset'] or source_rank1_sha256 != identity['source']['source_program_sha256']):
            raise RuntimeError('population completion source identity mismatch')
        assert_arm_resume(run_dir, role=role)
        if prompt_meta.get('infer_prompt_sha256') != identity['base_prompt_sha256']:
            raise RuntimeError('PICS v4 registered prompt SHA changed')
        for field, meta_field in (('sa40_fingerprint', 'observed_subset_fingerprint'), ('observed_train_fingerprint', 'observed_train_fingerprint'), ('observed_val_fingerprint', 'observed_val_fingerprint')):
            if prompt_meta.get(meta_field) != identity['cohort'][field]:
                raise RuntimeError(f'PICS v4 target observed cohort changed: {field}')
        if sorted(results.get('participant_ids') or []) != sorted(identity['cohort']['participant_ids']):
            raise RuntimeError('PICS v4 participant cohort changed')
        if not valid:
            raise RuntimeError('PICS v4 completion rank-1 is not runtime-valid')
        provenance['rank1_program_id'] = results['pool_best_program_id']
        provenance['run_identity'] = identity
        provenance['arm_role'] = role
        provenance['panel_bank_sha256'] = {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run_dir / 'global_phase/pics_v4_panel_banks').glob('*.json'))}
        provenance.update(identity['policies'])
    prov_path = run_dir / "POPULATION_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if not valid:
        raise RuntimeError(
            f"AAMAS v0 population rank-1 failed runtime check ({valid_reason}): {best}"
        )
    allowed_modes = {"aamas_v0_registered", "pics_v4_registered"}
    if provenance["prompt_mode"] not in allowed_modes:
        raise RuntimeError(
            f"AAMAS v0 population prompt_mode={provenance['prompt_mode']!r}"
        )
    if str(provenance["limited_data_protocol"]) != "structure_aware_v3":
        raise RuntimeError("AAMAS v0 population protocol is not structure_aware_v3")
    if int(provenance["limited_train_val"] or 0) != 40:
        raise RuntimeError("AAMAS v0 population limited_train_val is not 40")
    if str(results.get("evolution_selection_score") or "") != "train_val":
        raise RuntimeError("AAMAS v0 population selection score is not train_val")
    marker = {
        "stage": POPULATION_STAGE,
        "dataset": str(dataset),
        "kind": resolved_kind,
        "track_mode": str(track_mode),
        "prompt_policy": policy_id,
        "trial_prompt_policy": policy_id,
        "reminder_id": reminder_id,
        "sa40_fingerprint": provenance["sa40_fingerprint"],
        "global_iters": int(global_iters),
        "n_candidates": int(n_candidates),
        "rank1_program": str(best.resolve()),
        "rank1_sha256": program_sha,
        "pooled_train_val_score": score,
        "split_seed": provenance["split_seed"],
        "pics_run_seed": provenance["pics_run_seed"],
        "runtime_valid": True,
        "provenance_path": str(prov_path.resolve()),
    }
    if using_pics_v4():
        marker["search_rng"] = provenance["search_rng"]
        marker["pics_run_seed_scope"] = provenance["pics_run_seed_scope"]
        marker["elite_policy"] = provenance["elite_policy"]
        marker["participant_population_unique_cap"] = provenance[
            "participant_population_unique_cap"
        ]
        marker["elite_pool_size_is_upper_bound"] = provenance["elite_pool_size_is_upper_bound"]
        marker["duplicate_backfill"] = provenance["duplicate_backfill"]
    if identity is not None:
        marker['rank1_program_id'] = provenance['rank1_program_id']
        marker['run_identity'] = identity
        marker['arm_role'] = provenance['arm_role']
        marker['panel_bank_sha256'] = provenance['panel_bank_sha256']
    complete = run_dir / "STAGE_COMPLETE.json"
    complete.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return complete


def source_identity_for_target(
    target: str,
    *,
    config_path: Optional[Path] = None,
) -> str:
    """Dataset identity only. Ignores historical rank-1 paths in the YAML."""
    from utils.teh.t_pics_gated_transfer import load_frozen_transfer_config
    from utils.teh.t_pics_sources import normalize_t_pics_dataset

    cfg = load_frozen_transfer_config(config_path)
    from utils.teh.pics_v4 import using_pics_v4
    allowed_selectors = {'occurrence_eb_uniform_v8_primary'} if using_pics_v4() else {SOURCE_SELECTOR, "pics_aamas_v0_source_map"}
    if str(cfg.selector_name) not in allowed_selectors:
        raise RuntimeError(
            f"AAMAS v0 source map selector {cfg.selector_name!r} "
            f"not in {sorted(allowed_selectors)}"
        )
    key = normalize_t_pics_dataset(str(target))
    source = cfg.sources.get(key)
    if not source:
        raise RuntimeError(f"AAMAS v0 source identity unresolved for target {key}")
    source = normalize_t_pics_dataset(str(source))
    if source == key:
        raise RuntimeError(f"AAMAS v0 refusing self-source for {key}")
    entry = cfg.targets.get(key)
    if entry is not None and normalize_t_pics_dataset(entry.selected_source) != source:
        raise RuntimeError(
            f"AAMAS v0 source map disagreement for {key}: "
            f"sources={source} targets.selected_source={entry.selected_source}"
        )
    return source


def load_source_job_manifest(path: Path) -> Dict[str, str]:
    """Map source dataset -> job id. JSON or YAML. No program paths."""
    raw_path = Path(path)
    text = raw_path.read_text(encoding="utf-8")
    if raw_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"source-job manifest must be a mapping: {raw_path}")
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), dict) else payload
    if not isinstance(jobs, dict):
        raise RuntimeError(f"source-job manifest has no jobs mapping: {raw_path}")
    out: Dict[str, str] = {}
    for key, value in jobs.items():
        if str(key) in {"schema", "selector", "sources", "targets"}:
            continue
        if isinstance(value, dict):
            value = value.get("job_id")
        if value is None:
            continue
        out[str(key)] = str(value)
    return out


def _completed_population_arm(job_dir: Path) -> Optional[Path]:
    arm = population_arm_dir(job_dir)
    if (arm / "STAGE_COMPLETE.json").is_file() and (arm / "POPULATION_PROVENANCE.json").is_file():
        return arm
    return None


def resolve_aamas_v0_source_program(
    *,
    target_dataset: str,
    source_dataset: str,
    repo_root: Path,
    psych_split: str = "train",
    config_path: Optional[Path] = None,
    job_id: Optional[str] = None,
    job_manifest: Optional[Path] = None,
) -> Path:
    """Fail-closed load of a target-only population rank-1.

    ``source_dataset`` must match the source-map identity. The program path is
    taken only from a completed ``pics_aamas_v0_target_only`` population arm.
    """
    from utils.teh.pics_v4 import using_pics_v4
    if using_pics_v4():
        from utils.teh.pics_v4_transfer import official_inputs, resolve_source
        import os
        override = os.environ.get("PICS_V4_ABLATION_SOURCE_RANK1", "").strip()
        if override:
            sources, _artifacts = official_inputs(config_path)
            if sources[target_dataset] != source_dataset:
                raise RuntimeError("PICS v4 selected source mismatch")
            path = Path(override)
            if not path.is_file():
                raise FileNotFoundError(f"ablation source rank-1 missing: {path}")
            return path
        if psych_split != 'train' or job_id or job_manifest:
            raise RuntimeError('PICS v4 source-bank overrides are forbidden')
        record = resolve_source(target_dataset, config_path=config_path)
        if source_dataset != record['source_dataset']:
            raise RuntimeError('PICS v4 selected source mismatch')
        return Path(record['source_program_path'])
    from utils.teh.t_pics_sources import normalize_t_pics_dataset

    expected = source_identity_for_target(target_dataset, config_path=config_path)
    got = normalize_t_pics_dataset(str(source_dataset))
    if got != expected:
        raise RuntimeError(
            f"AAMAS v0 source identity mismatch for {target_dataset}: "
            f"got {got}, map requires {expected}"
        )
    root = population_run_root(repo_root, expected, psych_split=psych_split)
    manifest_job = None
    if job_manifest is not None:
        manifest_job = load_source_job_manifest(Path(job_manifest)).get(expected)
    chosen = str(job_id or manifest_job or "").strip() or None
    if chosen:
        candidates = [root / f"job_{chosen}"]
    else:
        candidates = sorted(p for p in root.glob("job_*") if p.is_dir()) if root.is_dir() else []
    completes: List[Path] = []
    for run in candidates:
        arm = _completed_population_arm(run)
        if arm is not None:
            completes.append(arm)
    if not completes:
        raise FileNotFoundError(
            f"AAMAS v0 target-only population artifact missing for source {expected} "
            f"under {root} (target_population/control completion required)"
        )
    if len(completes) != 1:
        raise RuntimeError(
            f"AAMAS v0 found {len(completes)} completed target-only populations for "
            f"{expected}; set AAMAS_V0_POPULATION_JOB_ID or a source-job manifest: "
            + ", ".join(str(p) for p in completes)
        )
    arm = completes[0]
    assert_aamas_population_path(arm)
    marker = _read_json(arm / "STAGE_COMPLETE.json")
    prov = _read_json(arm / "POPULATION_PROVENANCE.json")
    if marker.get("stage") != POPULATION_STAGE:
        raise RuntimeError(f"AAMAS v0 bad completion stage: {marker.get('stage')!r}")
    if marker.get("kind") != KIND_TARGET_ONLY or prov.get("kind") != KIND_TARGET_ONLY:
        raise RuntimeError("AAMAS v0 source population kind is not target_only")
    if str(prov.get("track_mode") or "") != TRACK_TARGET_ONLY:
        raise RuntimeError("AAMAS v0 source population track_mode is not target_only")
    if (
        marker.get("prompt_policy") != TRIAL_PROMPT_POLICY_ID
        or prov.get("prompt_policy") != TRIAL_PROMPT_POLICY_ID
    ):
        raise RuntimeError(
            "AAMAS v0 population prompt policy mismatch "
            f"(expected {TRIAL_PROMPT_POLICY_ID})"
        )
    if str(marker.get("dataset") or "") != expected or str(prov.get("dataset") or "") != expected:
        raise RuntimeError("AAMAS v0 population dataset key mismatch")
    if str(prov.get("limited_data_protocol") or "") != "structure_aware_v3":
        raise RuntimeError("AAMAS v0 population protocol mismatch")
    if int(prov.get("limited_train_val") or 0) != 40:
        raise RuntimeError("AAMAS v0 population SA40 budget mismatch")
    if not prov.get("sa40_fingerprint") or prov.get("sa40_fingerprint") != marker.get(
        "sa40_fingerprint"
    ):
        raise RuntimeError("AAMAS v0 population fingerprint missing or mismatched")
    if int(prov.get("global_iters") or 0) != GLOBAL_ITERS or int(prov.get("n_candidates") or 0) != N_CANDIDATES:
        raise RuntimeError("AAMAS v0 population budget mismatch")
    if prov.get("runtime_valid") is not True or prov.get("test_used_for_selection") is not False:
        raise RuntimeError("AAMAS v0 population rank-1 is not a valid train/val selection")
    if prov.get("arm_role") not in (None, "target_only"):
        raise RuntimeError("AAMAS v0 refusing a non-target-only arm as a source program")
    rank1 = Path(str(prov.get("rank1_program") or ""))
    if not rank1.is_file():
        raise FileNotFoundError(f"AAMAS v0 rank-1 missing: {rank1}")
    assert_aamas_population_path(rank1)
    digest = hashlib.sha256(rank1.read_bytes()).hexdigest()
    if digest != prov.get("rank1_sha256") or digest != marker.get("rank1_sha256"):
        raise RuntimeError("AAMAS v0 rank-1 SHA mismatch")
    return rank1.resolve()


def packs_cover_canonical_once() -> None:
    seen: List[str] = []
    for name, datasets in POPULATION_PACKS.items():
        if name not in ("pack1", "pack2", "pack3", "pack4"):
            raise RuntimeError(f"unexpected pack {name}")
        seen.extend(datasets)
    if len(POPULATION_PACKS) != 4:
        raise RuntimeError("AAMAS v0 requires exactly four population packs")
    if sorted(seen) != sorted(CANONICAL_DATASETS) or len(seen) != len(set(seen)):
        raise RuntimeError(
            "AAMAS v0 packs must cover the 15 canonical datasets exactly once"
        )


def iter_population_packs() -> Sequence[tuple[str, tuple[str, ...]]]:
    """Superseded 15-dataset population packs. Not a launch path."""
    packs_cover_canonical_once()
    return [(name, POPULATION_PACKS[name]) for name in ("pack1", "pack2", "pack3", "pack4")]


# Temporary experimental track. Not the official main result.
BATCH1_TARGET_ONLY_DATASETS: tuple[str, ...] = (
    "2plonsky2018when",
    "3frey2017cct",
    "bergert_nosofsky_2007",
    "11enkavi2019recentprobes",
    "5speekenbrink2008learning",
    "guan_2020_stopping",
)
# Packed by ICLR wall-clock hours, longer dataset first inside a job.
# Batch 1 keeps a 2/2/1/1 count. Estimated hours: 3.4, 3.1, 3.4, 4.5.
BATCH1_TARGET_ONLY_PACKS: Mapping[str, tuple[str, ...]] = {
    "pack1": ("guan_2020_stopping", "11enkavi2019recentprobes"),
    "pack2": ("bergert_nosofsky_2007", "5speekenbrink2008learning"),
    "pack3": ("3frey2017cct",),
    "pack4": ("2plonsky2018when",),
}
# Estimated hours: 5.4, 5.1, 5.0, 4.8.
BATCH2_TARGET_ONLY_PACKS: Mapping[str, tuple[str, ...]] = {
    "pack1": (
        "4wulff2018description",
        "7hilbig2014generalized",
        "12badham2017deficits",
    ),
    "pack2": ("14kool2016when", "steyvers_2009_bandit"),
    "pack3": ("10frey2017risk", "1peterson2021using"),
    "pack4": ("mixed_gambles", "13schulz2020finding"),
}
# All 15. Estimated hours: 8.7, 8.7, 8.7, 8.6.
BATCH5_TRANSFER_PACKS: Mapping[str, tuple[str, ...]] = {
    "pack1": (
        "2plonsky2018when",
        "4wulff2018description",
        "1peterson2021using",
    ),
    "pack2": (
        "14kool2016when",
        "bergert_nosofsky_2007",
        "steyvers_2009_bandit",
        "12badham2017deficits",
    ),
    "pack3": (
        "10frey2017risk",
        "3frey2017cct",
        "7hilbig2014generalized",
    ),
    "pack4": (
        "mixed_gambles",
        "13schulz2020finding",
        "guan_2020_stopping",
        "11enkavi2019recentprobes",
        "5speekenbrink2008learning",
    ),
}

_PROVENANCE_MATCH_FIELDS = (
    "prompt_policy",
    "prompt_mode",
    "rendered_prompt_sha256",
    "sa40_fingerprint",
    "limited_data_protocol",
    "limited_train_val",
    "global_iters",
    "n_candidates",
    "model_name",
    "hard_prompt_token_cap",
    "llm_max_tokens",
    "max_parent_chars",
    "vllm_max_model_len",
    "range_start_ordinal",
    "range_end_ordinal",
)


def kind_for_track(track_mode: str) -> str:
    key = str(track_mode or "").strip()
    if key not in KIND_BY_TRACK:
        raise ValueError(
            f"pics_aamas_v0_track_mode must be one of {TRACK_MODES}, got {track_mode!r}"
        )
    return KIND_BY_TRACK[key]


def aamas_v0_prompt_enabled(args: Any) -> bool:
    return str(getattr(args, "pics_aamas_v0_track_mode", "") or "").strip() in TRACK_MODES


def population_arm_dir(job_dir: Path) -> Path:
    """Target-only population arm inside a track job directory."""
    return Path(job_dir) / "target_population" / "control"


def iter_named_packs(
    packs: Mapping[str, tuple[str, ...]],
) -> Sequence[tuple[str, tuple[str, ...]]]:
    return [(name, packs[name]) for name in ("pack1", "pack2", "pack3", "pack4")]


def assert_packs_partition(packs: Mapping[str, Sequence[str]], expected: Sequence[str]) -> None:
    seen: List[str] = []
    if tuple(packs) != ("pack1", "pack2", "pack3", "pack4") and set(packs) != {
        "pack1",
        "pack2",
        "pack3",
        "pack4",
    }:
        raise RuntimeError(f"expected exactly four packs, got {sorted(packs)}")
    if len(packs) != 4:
        raise RuntimeError("expected exactly four packs")
    for name in ("pack1", "pack2", "pack3", "pack4"):
        seen.extend(packs[name])
    if len(seen) != len(set(seen)):
        raise RuntimeError(f"pack datasets repeat: {seen}")
    if sorted(seen) != sorted(expected):
        raise RuntimeError(
            f"pack coverage mismatch: got {sorted(seen)} expected {sorted(expected)}"
        )


def decide_aamas_v0_gate(
    *,
    target_only_score: float,
    transfer_score: float,
) -> Dict[str, Any]:
    """Official gate. Transfer wins only when its pooled train∪val score is strictly greater.

    Exact ties select target-only. Test scores are not an input.
    """
    import math

    try:
        target_ll = float(target_only_score)
        transfer_ll = float(transfer_score)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("official gate scores must be finite train∪val log-likelihoods") from exc
    if not math.isfinite(target_ll) or not math.isfinite(transfer_ll):
        raise RuntimeError("official gate scores must be finite train∪val log-likelihoods")
    diff = transfer_ll - target_ll
    if transfer_ll > target_ll:
        selected, reason = "transfer", "transfer_strictly_greater"
    elif transfer_ll == target_ll:
        selected, reason = "target_only", "exact_tie_select_target_only"
    else:
        selected, reason = "target_only", "target_only_greater"
    return {
        "schema": "pics_aamas_v0_official_gate_v1",
        "selected_arm": selected,
        "reason": reason,
        "target_only_pooled_train_val_loglik": target_ll,
        "transfer_pooled_train_val_loglik": transfer_ll,
        "score_difference_transfer_minus_target_only": diff,
        "score_field": "pooled_train_val_loglik",
        "observed_splits": ["train", "val"],
        "test_used_for_gate": False,
        "tie_policy": "exact_tie_selects_target_only",
        "strictly_greater_required": True,
    }


def _population_arm_from_run(run: Path, role: str) -> Path:
    run = Path(run)
    if (run / "POPULATION_PROVENANCE.json").is_file() and (run / "global_phase").is_dir():
        return run
    sub_name = "control" if role == "target_only" else "transfer"
    arm = run / "target_population" / sub_name
    if (arm / "POPULATION_PROVENANCE.json").is_file():
        return arm
    raise FileNotFoundError(
        f"AAMAS v0 {role} population provenance missing under {run}"
    )


def _require_matching_provenance(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if str(left.get("dataset") or "") != str(right.get("dataset") or ""):
        raise RuntimeError(
            "official-gate materialize dataset mismatch: "
            f"{left.get('dataset')!r} vs {right.get('dataset')!r}"
        )
    for field in _PROVENANCE_MATCH_FIELDS:
        if left.get(field) != right.get(field):
            raise RuntimeError(
                f"official-gate materialize mismatch on {field}: "
                f"{left.get(field)!r} vs {right.get(field)!r}"
            )
    left_ids = sorted(int(x) for x in (left.get("participant_ids") or []))
    right_ids = sorted(int(x) for x in (right.get("participant_ids") or []))
    if left_ids != right_ids:
        raise RuntimeError("official-gate materialize participant_ids mismatch")
    if left.get("test_used_for_selection") is not False or right.get("test_used_for_selection") is not False:
        raise RuntimeError("official-gate materialize refused a population that used test for selection")
    if str(left.get("prompt_policy") or "") != TRIAL_PROMPT_POLICY_ID:
        raise RuntimeError(
            "official-gate materialize prompt policy is not "
            f"{TRIAL_PROMPT_POLICY_ID}"
        )


def materialize_official_gate(
    *,
    target_only_run: Path,
    transfer_run: Path,
    output_dir: Path,
    target_dataset: str,
    evaluate_score,
) -> Dict[str, Any]:
    """Build the official gate from completed target-only and transfer-only populations.

    ``evaluate_score(program_path, provenance)`` must return the pooled train∪val
    log-likelihood and must not consult test. Selection uses only those two scores.
    """
    target_arm = _population_arm_from_run(Path(target_only_run), "target_only")
    transfer_arm = _population_arm_from_run(Path(transfer_run), "transfer")
    target_prov = _read_json(target_arm / "POPULATION_PROVENANCE.json")
    transfer_prov = _read_json(transfer_arm / "POPULATION_PROVENANCE.json")
    v4 = str(target_prov.get('kind')) == 'pics_v4_target_only'
    if v4:
        from utils.teh.pics_v4_transfer import validate_materialization
        target_prov, transfer_prov = validate_materialization(target_arm, transfer_arm, target_dataset, output_dir)
    else:
        if str(target_prov.get("dataset") or "") != str(target_dataset):
            raise RuntimeError("target-only run dataset does not match the official target")
        if str(target_prov.get("kind") or "") != KIND_TARGET_ONLY or str(target_prov.get("track_mode") or "") != TRACK_TARGET_ONLY:
            raise RuntimeError("materialize target arm is not a target_only population")
        if str(transfer_prov.get("kind") or "") != KIND_TRANSFER_BASED_ONLY or str(
            transfer_prov.get("track_mode") or ""
        ) != TRACK_TRANSFER_BASED_ONLY:
            raise RuntimeError("materialize transfer arm is not a transfer_based_only population")
        if str(transfer_prov.get("arm_role") or "") != "transfer":
            raise RuntimeError("materialize transfer provenance is not a transfer arm")
        _require_matching_provenance(target_prov, transfer_prov)
        if not transfer_prov.get("source_dataset") or not transfer_prov.get("source_rank1_sha256"):
            raise RuntimeError("transfer population is missing source identity provenance")
    target_rank1 = Path(str(target_prov["rank1_program"]))
    transfer_rank1 = Path(str(transfer_prov["rank1_program"]))
    target_score = float(evaluate_score(target_rank1, target_prov))
    transfer_score = float(evaluate_score(transfer_rank1, transfer_prov))
    decision = decide_aamas_v0_gate(
        target_only_score=target_score,
        transfer_score=transfer_score,
    )
    winner = target_arm if decision["selected_arm"] == "target_only" else transfer_arm
    retained_pool = winner / "global_phase" / "global_elite_pool"
    if not retained_pool.is_dir():
        raise FileNotFoundError(f"winning elite pool missing: {retained_pool}")
    out = Path(output_dir)
    assert_aamas_track_output(out, 'pics_v4_official_gate' if v4 else KIND_OFFICIAL_GATE)
    gate_dir = out / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    selected = out / "selected"
    selected.mkdir(parents=True, exist_ok=True)
    record = {
        **decision,
        "target": str(target_dataset),
        "materialized_from_tracks": True,
        "target_only_run": str(target_arm.resolve()),
        "transfer_run": str(transfer_arm.resolve()),
        "target_only_rank1": str(target_rank1.resolve()),
        "transfer_rank1": str(transfer_rank1.resolve()),
        "source_dataset": transfer_prov.get("source_dataset"),
        "source_rank1_sha256": transfer_prov.get("source_rank1_sha256"),
        "retained_pool_path": str(retained_pool.resolve()),
        "prompt_policy": PROMPT_POLICY_ID,
        "sa40_fingerprint": target_prov.get("sa40_fingerprint"),
    }
    if v4:
        from utils.teh.pics_v4_transfer import current_identity
        record['run_identity'] = current_identity()
        record['prompt_policy'] = target_prov['prompt_policy']
        record['target_only_rank1_sha256'] = target_prov['rank1_sha256']
        record['transfer_rank1_sha256'] = transfer_prov['rank1_sha256']
        record['prompt_exposures_matched'] = False
    gate_path = gate_dir / "gate_record.json"
    gate_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    link = selected / "retained_global_elite_pool"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        import shutil

        shutil.rmtree(link)
    try:
        link.symlink_to(retained_pool.resolve())
    except OSError:
        import shutil

        shutil.copytree(retained_pool, link)
    (selected / "SELECTED_ARM.txt").write_text(decision["selected_arm"] + "\n", encoding="utf-8")
    (out / "TRACK_STATUS.txt").write_text(
        "official_method\ntrack_mode=official_gate\nmaterialized_from_completed_tracks\n",
        encoding="utf-8",
    )
    record["gate_record_path"] = str(gate_path.resolve())
    return record
