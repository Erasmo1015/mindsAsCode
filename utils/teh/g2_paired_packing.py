"""G.2 paired target-example packing (PICS v3).

Control and transfer freeze the same target-example IDs under the *transfer*
token condition (source suffix + runtime contract + reserved parent copies).
Transfer then receives the suffix; control does not. Parent *count* is paired
under the stricter (transfer) budget; parent *identities* may differ by arm.
Existing parent-copy compaction (per-parent ``max_parent_chars`` head/tail) is
preserved. At freeze time, examples are chosen under the transfer budget so
runtime G.2 (``freeze_examples=True``) only parent-trims; it must not drop the
frozen example set. Non-frozen prompts drop whole trials before extra parents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.teh.prompt_context import attach_runtime_contract_to_prompt
from utils.teh.prompt_snapshots import format_snapshot_examples, prompt_participant_id
from utils.teh.prompt_units import (
    OUTPUT_RESERVE,
    PROMPT_DISPLAY_CEILING,
    QWEN_INPUT_CEILING,
    VLLM_CONTEXT,
    infer_prompt_dataset_alias,
    qwen_user_prompt_token_count,
    select_structure_aware_prompt_examples,
)
from utils.teh.pics_v3 import G2_PAIRED_PACK_VERSION

SOURCE_SUFFIX_MARKERS = (
    "## Cross-task transfer context",
    "SOURCE-dataset context only",
    "Best population-level program",
    "SOURCE example from source train+validation only",
)


class PairedPackingFitError(RuntimeError):
    """Required instruction + one parent + suffix cannot fit the 14k budget."""


def stable_target_example_id(trial: Dict[str, Any], *, selection_index: Optional[int] = None) -> str:
    """Stable ID for a prompt example. Selection index guarantees uniqueness."""
    problem = trial.get("problem") or {}
    ldp = trial.get("_ldp") or {}
    origin = ldp.get("origin_index")
    if origin is None:
        origin = ldp.get("session_index")
    unit = (
        ldp.get("unit_id")
        if ldp.get("unit_id") is not None
        else problem.get("problem_id")
        if problem.get("problem_id") is not None
        else problem.get("presented_day")
        if problem.get("presented_day") is not None
        else problem.get("round")
        if problem.get("round") is not None
        else problem.get("game")
        if problem.get("game") is not None
        else problem.get("balloon_id")
        if problem.get("balloon_id") is not None
        else problem.get("block_index")
    )
    parts = [
        str(trial.get("_prompt_participant_id")),
        str(ldp.get("origin_split") or ""),
        str(origin if origin is not None else ""),
        str(unit if unit is not None else ""),
        str(problem.get("stage") if problem.get("stage") is not None else ""),
        str(trial.get("action") if trial.get("action") is not None else ""),
    ]
    if selection_index is not None:
        parts.insert(0, str(int(selection_index)))
    return "|".join(parts)


def origin_splits_in_trials(trials: Iterable[Dict[str, Any]]) -> List[str]:
    found = []
    for trial in trials:
        split = str((trial.get("_ldp") or {}).get("origin_split") or "")
        if split and split not in found:
            found.append(split)
    return found


def assert_no_test_in_trials(trials: Iterable[Dict[str, Any]], *, label: str) -> None:
    for i, trial in enumerate(trials):
        if str((trial.get("_ldp") or {}).get("origin_split") or "") == "test":
            raise AssertionError(f"{label} contains a test trial at index {i}")


def pad_parent_to_max_chars(code: str, max_parent_chars: int) -> str:
    body = (code or "def choose(problem, history):\n    return 0.5\n").rstrip() + "\n"
    n = int(max_parent_chars)
    if n <= 0:
        return body
    if len(body) >= n:
        return body[:n]
    pad_line = "    _PAD = '" + ("x" * 80) + "'\n"
    while len(body) + len(pad_line) <= n:
        body += pad_line
    need = n - len(body)
    if need > 0:
        body += " " * need
    return body[:n]


def one_parent_context(code: str) -> str:
    return (
        f"\n\nReference program (parent):\n```python\n{code}\n```\n\n"
        "Generate a variant that improves upon or explores alternatives to the parent program.\n"
    )


def freeze_reservation_parent_context(code: str) -> str:
    """One max-length parent plus dummy loglik lines matching production parent context."""
    return (
        f"\n\nReference program (parent):\n```python\n{code}\n```\n\n"
        "Parent performance: train_loglik: -9.9999 val_loglik: -9.9999 "
        "overall_loglik: -9.9999\n\n"
        "Generate a variant that improves upon or explores alternatives to the parent program.\n"
    )


def n_parent_context(codes: Sequence[str]) -> str:
    codes = list(codes)
    if len(codes) <= 1:
        return one_parent_context(codes[0] if codes else "def choose(problem, history):\n    return 0.5\n")
    parts = [f"\n\nReference parent programs ({len(codes)} elite programs):\n"]
    for i, program in enumerate(codes):
        parts.append(f"\nParent {i + 1}:\n```python\n{program}\n```\n")
    parts.append(
        "\nGenerate a variant that improves upon or explores alternatives to these parent programs.\n"
    )
    return "".join(parts)


def assemble_candidate_prompt(
    *,
    instruction: str,
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    parent_context: str,
    code_template_suffix: str,
    candidate_output_rules: str,
    runtime_contract: str = "",
    extra_prompt_trials_label: str = "Validation observations",
) -> str:
    state_text = format_snapshot_examples(list(train_trials)) if train_trials else ""
    extra_state_text = ""
    if val_trials:
        extra_state_text = (
            f"\n\n{extra_prompt_trials_label}:\n"
            f"{format_snapshot_examples(list(val_trials))}\n"
        )
    text = (
        f"{instruction}\n{state_text}{extra_state_text}\n{parent_context}"
        f"{code_template_suffix}\n{candidate_output_rules}\n"
    )
    if runtime_contract and runtime_contract.strip():
        text = attach_runtime_contract_to_prompt(text, runtime_contract.strip())
    return text


def final_prompt_tokens(prompt: str) -> int:
    return int(qwen_user_prompt_token_count(prompt))


def fits_input_and_context(
    tokens: int,
    *,
    input_ceiling: int = QWEN_INPUT_CEILING,
    output_reserve: int = OUTPUT_RESERVE,
    vllm_context: int = VLLM_CONTEXT,
) -> bool:
    """True when tokens fit the input ceiling and leave room for completion.

    ``output_reserve`` should match ``--llm_max_tokens`` (default
    ``OUTPUT_RESERVE`` = PICS v3 1024).
    """
    return int(tokens) <= int(input_ceiling) and (
        int(tokens) + int(output_reserve) <= int(vllm_context)
    )


def split_union_into_train_val(
    selected: Sequence[Dict[str, Any]],
    train_pool: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_ids = {id(t) for t in train_pool}
    train = [t for t in selected if id(t) in train_ids]
    val = [t for t in selected if id(t) not in train_ids]
    return train, val


def _instruction_with_suffix(infer_text: str, suffix: Optional[str]) -> str:
    base = (infer_text or "").rstrip()
    if suffix and suffix.strip():
        return f"{base}\n\n{suffix.strip()}\n"
    return base


def freeze_g2_target_examples(
    *,
    dataset: str,
    pooled_train: Sequence[Dict[str, Any]],
    pooled_val: Sequence[Dict[str, Any]],
    infer_text: str,
    source_suffix: str,
    runtime_contract: str,
    seed_code: str,
    code_template_suffix: str,
    candidate_output_rules: str,
    max_parent_chars: int,
    max_prompt_train_trials: int,
    prompt_train_trials_seed: int,
    hard_prompt_token_cap: int = QWEN_INPUT_CEILING,
    sample_size: int = 8,
    output_reserve: int = OUTPUT_RESERVE,
) -> Dict[str, Any]:
    """Choose the largest target-example prefix that fits the transfer condition.

    Reserves up to ``sample_size`` parent copies (each capped at ``max_parent_chars``)
    under the transfer suffix so control cannot exploit freed suffix space for
    extra parents. Records ``paired_parent_count`` for both arms.
    """
    train_list = list(pooled_train)
    val_list = list(pooled_val)
    union = train_list + val_list
    assert_no_test_in_trials(union, label="G.2 freeze union")
    alias = infer_prompt_dataset_alias(union) if union else dataset
    pids = {t.get("_prompt_participant_id") for t in union if t.get("_prompt_participant_id") is not None}
    max_trials = min(int(max_prompt_train_trials or PROMPT_DISPLAY_CEILING), PROMPT_DISPLAY_CEILING)
    selected, sel_diag = select_structure_aware_prompt_examples(
        union,
        dataset=alias,
        max_trials=max_trials,
        subsample_seed=int(prompt_train_trials_seed),
        pooled=len(pids) > 1,
    )
    from utils.teh.pics_v3_cpc18_prompt import ensure_cpc18_feedback_regime_examples

    selected, cpc18_diag = ensure_cpc18_feedback_regime_examples(
        selected, union, dataset=alias
    )
    sel_diag = dict(sel_diag)
    sel_diag["cpc18_feedback_regime"] = cpc18_diag
    reservation_parent = pad_parent_to_max_chars(seed_code, int(max_parent_chars))
    transfer_instruction = _instruction_with_suffix(infer_text, source_suffix)
    n_reserve = max(1, min(int(sample_size), 8))
    out_reserve = int(output_reserve)

    def parent_ctx(n: int) -> str:
        codes = [reservation_parent] * max(1, int(n))
        if len(codes) == 1:
            return freeze_reservation_parent_context(codes[0])
        return n_parent_context(codes)

    def _fits(n_tok: int) -> bool:
        return fits_input_and_context(
            n_tok,
            input_ceiling=hard_prompt_token_cap,
            output_reserve=out_reserve,
        )

    required = assemble_candidate_prompt(
        instruction=transfer_instruction,
        train_trials=[],
        val_trials=[],
        parent_context=parent_ctx(1),
        code_template_suffix=code_template_suffix,
        candidate_output_rules=candidate_output_rules,
        runtime_contract=runtime_contract,
    )
    required_tokens = final_prompt_tokens(required)
    if not _fits(required_tokens):
        raise PairedPackingFitError(
            "G.2 required instruction + one parent + source suffix + runtime contract "
            f"is {required_tokens} tokens (cap {hard_prompt_token_cap}; "
            f"input+{out_reserve} vs vLLM {VLLM_CONTEXT})."
        )

    # Prefer reserving sample_size parents when selecting examples (stricter transfer arm).
    reserved_n = 1
    for n in range(n_reserve, 0, -1):
        probe = assemble_candidate_prompt(
            instruction=transfer_instruction,
            train_trials=[],
            val_trials=[],
            parent_context=parent_ctx(n),
            code_template_suffix=code_template_suffix,
            candidate_output_rules=candidate_output_rules,
            runtime_contract=runtime_contract,
        )
        if _fits(final_prompt_tokens(probe)):
            reserved_n = n
            break

    def pack_prefix(k: int) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], int]:
        kept = list(selected[:k])
        tr, va = split_union_into_train_val(kept, train_list)
        prompt = assemble_candidate_prompt(
            instruction=transfer_instruction,
            train_trials=tr,
            val_trials=va,
            parent_context=parent_ctx(reserved_n),
            code_template_suffix=code_template_suffix,
            candidate_output_rules=candidate_output_rules,
            runtime_contract=runtime_contract,
        )
        return prompt, tr, va, final_prompt_tokens(prompt)

    lo, hi = 0, len(selected)
    best_k = 0
    best_tokens = required_tokens
    best_train: List[Dict[str, Any]] = []
    best_val: List[Dict[str, Any]] = []
    while lo <= hi:
        mid = (lo + hi) // 2
        _prompt, tr, va, n_tok = pack_prefix(mid)
        if _fits(n_tok):
            best_k = mid
            best_tokens = n_tok
            best_train, best_val = tr, va
            lo = mid + 1
        else:
            hi = mid - 1

    # After freezing examples, largest parent count that still fits transfer.
    paired_parent_count = 1
    for n in range(n_reserve, 0, -1):
        prompt = assemble_candidate_prompt(
            instruction=transfer_instruction,
            train_trials=best_train,
            val_trials=best_val,
            parent_context=parent_ctx(n),
            code_template_suffix=code_template_suffix,
            candidate_output_rules=candidate_output_rules,
            runtime_contract=runtime_contract,
        )
        if _fits(final_prompt_tokens(prompt)):
            paired_parent_count = n
            best_tokens = final_prompt_tokens(prompt)
            break

    example_ids = []
    train_ids = []
    val_ids = []
    train_id_set = {id(t) for t in best_train}
    for i, trial in enumerate(selected[:best_k]):
        eid = stable_target_example_id(trial, selection_index=i)
        example_ids.append(eid)
        if id(trial) in train_id_set:
            train_ids.append(eid)
        else:
            val_ids.append(eid)

    suffix_tokens = 0
    if source_suffix and source_suffix.strip():
        suffix_tokens = final_prompt_tokens(
            _instruction_with_suffix(infer_text, source_suffix)
        ) - final_prompt_tokens(_instruction_with_suffix(infer_text, None))

    return {
        "version": G2_PAIRED_PACK_VERSION,
        "dataset": dataset,
        "prompt_dataset_alias": alias,
        "max_prompt_train_trials": max_trials,
        "prompt_train_trials_seed": int(prompt_train_trials_seed),
        "hard_prompt_token_cap": int(hard_prompt_token_cap),
        "output_reserve": out_reserve,
        "max_parent_chars_reserved": int(max_parent_chars),
        "sample_size_reserved": int(n_reserve),
        "parents_reserved_at_example_freeze": int(reserved_n),
        "paired_parent_count": int(paired_parent_count),
        "n_selected_available": len(selected),
        "n_examples_included": best_k,
        "n_train_included": len(best_train),
        "n_val_included": len(best_val),
        "example_ids": example_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "transfer_required_tokens": required_tokens,
        "transfer_packed_tokens_at_freeze": best_tokens,
        "source_suffix_tokens": int(suffix_tokens),
        "selection_diag": sel_diag,
        "origin_splits": origin_splits_in_trials(selected[:best_k]),
        "test_examples_included": False,
    }


def materialize_frozen_trials(
    freeze: Dict[str, Any],
    *,
    pooled_train: Sequence[Dict[str, Any]],
    pooled_val: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuild frozen train/val lists in freeze order from the live pools."""
    union = list(pooled_train) + list(pooled_val)
    train_obj = {id(t) for t in pooled_train}
    selected, _ = select_structure_aware_prompt_examples(
        union,
        dataset=str(freeze.get("prompt_dataset_alias") or freeze.get("dataset")),
        max_trials=int(freeze.get("max_prompt_train_trials") or PROMPT_DISPLAY_CEILING),
        subsample_seed=int(freeze.get("prompt_train_trials_seed") or 0),
        pooled=True,
    )
    wanted = list(freeze.get("example_ids") or [])
    by_id: Dict[str, Dict[str, Any]] = {}
    for i, trial in enumerate(selected):
        by_id[stable_target_example_id(trial, selection_index=i)] = trial
    ordered: List[Dict[str, Any]] = []
    missing = []
    for eid in wanted:
        trial = by_id.get(eid)
        if trial is None:
            missing.append(eid)
        else:
            ordered.append(trial)
    if missing:
        raise RuntimeError(f"G.2 paired-pack freeze IDs not in live pool: {missing[:8]}")
    train = [t for t in ordered if id(t) in train_obj]
    val = [t for t in ordered if id(t) not in train_obj]
    return train, val


def write_paired_pack_freeze(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_paired_pack_freeze(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"G.2 paired-pack freeze is not an object: {path}")
    version = str(payload.get("version") or "")
    if version != G2_PAIRED_PACK_VERSION:
        raise ValueError(
            f"G.2 paired-pack freeze version {version!r} != {G2_PAIRED_PACK_VERSION!r}"
        )
    return payload


def prompt_contains_source_suffix(text: str) -> bool:
    blob = text or ""
    return any(marker in blob for marker in SOURCE_SUFFIX_MARKERS)


def trim_reason(steps: Sequence[str]) -> str:
    return ",".join(steps) if steps else "none_under_cap"


def drop_parents_until_fit(
    *,
    instruction: str,
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    parent_programs: Sequence[str],
    parent_ids: Sequence[str],
    code_template_suffix: str,
    candidate_output_rules: str,
    runtime_contract: str,
    max_parent_chars: int,
    hard_prompt_token_cap: int,
    token_count: Callable[[str], int],
    compress_instruction: Optional[Callable[[str], str]] = None,
    strip_parent: Optional[Callable[[str], str]] = None,
    truncate_parent: Optional[Callable[[str, int], Tuple[str, bool]]] = None,
    output_reserve: int = OUTPUT_RESERVE,
) -> Tuple[str, Dict[str, Any], List[str]]:
    """Keep frozen examples; drop extra parents / slice parent chars if needed."""
    steps: List[str] = []
    base = instruction
    parents = list(parent_programs)
    ids = list(parent_ids) if parent_ids else [f"parent_{i}" for i in range(len(parents))]
    if len(ids) < len(parents):
        ids = ids + [f"parent_{i}" for i in range(len(ids), len(parents))]
    n_before = len(parents)
    out_reserve = int(output_reserve)

    def assemble(cur_parents: List[str]) -> str:
        return assemble_candidate_prompt(
            instruction=base,
            train_trials=train_trials,
            val_trials=val_trials,
            parent_context=n_parent_context(cur_parents),
            code_template_suffix=code_template_suffix,
            candidate_output_rules=candidate_output_rules,
            runtime_contract=runtime_contract,
        )

    prompt = assemble(parents)
    tokens_before = token_count(prompt)
    if compress_instruction is not None and tokens_before > hard_prompt_token_cap:
        base = compress_instruction(base)
        steps.append("compress_instruction_whitespace")
        prompt = assemble(parents)
    if strip_parent is not None:
        parents = [strip_parent(p) for p in parents]
        steps.append("strip_parent_comments")
        prompt = assemble(parents)
    while len(parents) > 1 and token_count(assemble(parents)) > hard_prompt_token_cap:
        parents = parents[:-1]
        ids = ids[: len(parents)]
        steps.append("drop_extra_parent")
        prompt = assemble(parents)
    if (
        truncate_parent is not None
        and max_parent_chars > 0
        and token_count(assemble(parents)) > hard_prompt_token_cap
    ):
        new_parents = []
        changed = False
        for parent in parents:
            clipped, was = truncate_parent(parent, max_parent_chars)
            new_parents.append(clipped)
            changed = changed or was
        if changed:
            parents = new_parents
            steps.append("parent_char_truncation")
            prompt = assemble(parents)
    tokens_after = token_count(prompt)
    if not fits_input_and_context(
        tokens_after,
        input_ceiling=hard_prompt_token_cap,
        output_reserve=out_reserve,
    ):
        raise PairedPackingFitError(
            "G.2 packed prompt still exceeds budget after parent drops with frozen "
            f"target examples: {tokens_after} tokens (cap {hard_prompt_token_cap}, "
            f"output_reserve={out_reserve}). "
            f"trim={trim_reason(steps)}"
        )
    diag = {
        "truncated": bool(steps) or tokens_before > tokens_after,
        "prompt_tokens_before_truncation": tokens_before,
        "prompt_tokens_after_truncation": tokens_after,
        "final_chat_template_tokens": tokens_after,
        "parents_before": n_before,
        "parents_after": len(parents),
        "parent_ids_after": list(ids[: len(parents)]),
        "parent_retention_differed": len(parents) != n_before,
        "trim_reason": trim_reason(steps),
        "truncation_steps": list(steps),
        "frozen_examples": True,
    }
    return prompt, diag, steps
