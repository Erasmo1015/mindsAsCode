"""
Faithful, budgeted train-trial context for automatic dataset-prompt generation.

Used only by the prompt-writing LLM path (not PICS per-iteration state prompts).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_EXAMPLE_CHAR_BUDGET = 10_000
DEFAULT_HISTORY_MAX_ENTRIES = 8
DEFAULT_MAX_EXAMPLES = 8

_SKIP_PROBLEM_META = frozenset({"dataset_alias", "experiment_id"})


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _merge_type_sets(dst: Set[str], src: Iterable[str]) -> None:
    dst.update(src)


def _walk_value_schema(
    value: Any,
    *,
    key_stats: Dict[str, Dict[str, Any]],
    n_parents: int,
    path_prefix: str = "",
) -> None:
    """Accumulate nested key presence/types under path_prefix into key_stats."""
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        path = f"{path_prefix}.{key}" if path_prefix else str(key)
        st = key_stats.setdefault(
            path,
            {"present": 0, "types": set(), "list_elem_types": set(), "children_seen": 0},
        )
        st["present"] += 1
        st["types"].add(_json_type_name(child))
        if isinstance(child, dict):
            _walk_value_schema(child, key_stats=key_stats, n_parents=n_parents, path_prefix=path)
        elif isinstance(child, list) and child:
            for elem in child[:5]:
                st["list_elem_types"].add(_json_type_name(elem))
                if isinstance(elem, dict):
                    _walk_value_schema(
                        elem, key_stats=key_stats, n_parents=n_parents, path_prefix=f"{path}[]"
                    )


def _format_key_stats(key_stats: Dict[str, Dict[str, Any]], n_examples: int) -> List[str]:
    lines: List[str] = []
    for path in sorted(key_stats):
        st = key_stats[path]
        present = int(st["present"])
        freq = "always" if present >= n_examples else "sometimes"
        types = ", ".join(sorted(st["types"])) or "?"
        extra = ""
        if st.get("list_elem_types"):
            extra = f"; list_elem={', '.join(sorted(st['list_elem_types']))}"
        lines.append(f"  - {path}: {types} ({freq} present, {present}/{n_examples}){extra}")
    return lines


def _partition_trials_by_stage(
    trials: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group trials by problem['stage'] when present; else single bucket 'all'."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    has_stage = any(
        isinstance(t.get("problem"), dict) and "stage" in t["problem"] for t in trials
    )
    for t in trials:
        p = t.get("problem") or {}
        if has_stage:
            stage = p.get("stage")
            key = f"stage={stage}" if stage is not None else "stage=<missing>"
        else:
            key = "all"
        buckets[key].append(t)
    return dict(buckets)


def infer_recursive_runtime_schema(trials: Sequence[Dict[str, Any]]) -> str:
    """
    Recursive runtime schema from parsed train trials.

    Marks always/sometimes keys and, when ``stage`` varies, documents
    stage-conditional problem key availability separately.
    """
    if not trials:
        return "- (no parsed trial examples provided)"

    lines: List[str] = []
    schemas = sorted(
        {
            str((t.get("problem") or {}).get("schema_type", "?"))
            for t in trials
        }
    )
    lines.append(f"- schema_type(s): {', '.join(schemas)}")
    has_gamble = any(
        "gamble_A" in (t.get("problem") or {}) or "gamble_B" in (t.get("problem") or {})
        for t in trials
    )
    lines.append(f"- is_gamble_A/B_task: {has_gamble}")

    buckets = _partition_trials_by_stage(trials)
    for bucket_name, bucket_trials in sorted(buckets.items()):
        n = len(bucket_trials)
        problem_stats: Dict[str, Dict[str, Any]] = {}
        history_stats: Dict[str, Dict[str, Any]] = {}
        history_lens: List[int] = []
        for t in bucket_trials:
            p = dict(t.get("problem") or {})
            for meta in _SKIP_PROBLEM_META:
                p.pop(meta, None)
            _walk_value_schema(p, key_stats=problem_stats, n_parents=n)
            hist = t.get("history") or []
            history_lens.append(len(hist) if isinstance(hist, list) else 0)
            if isinstance(hist, list):
                for entry in hist:
                    if isinstance(entry, dict):
                        _walk_value_schema(entry, key_stats=history_stats, n_parents=n)

        header = (
            f"- problem schema ({bucket_name}, n_examples={n}):"
            if bucket_name != "all"
            else f"- problem schema (n_examples={n}):"
        )
        lines.append(header)
        lines.extend(_format_key_stats(problem_stats, n))
        if history_stats:
            lines.append(
                f"- history entry schema ({bucket_name}, "
                f"len range {min(history_lens)}..{max(history_lens)}):"
                if bucket_name != "all"
                else (
                    f"- history entry schema "
                    f"(len range {min(history_lens)}..{max(history_lens)}):"
                )
            )
            lines.extend(_format_key_stats(history_stats, n))
        else:
            lines.append(f"- history: empty or no dict entries ({bucket_name})")

    # Action semantics hint from first trial with option_keys
    for t in trials:
        p = t.get("problem") or {}
        keys = p.get("option_keys")
        if isinstance(keys, list) and len(keys) >= 2:
            schema = str(p.get("schema_type", "?"))
            lines.append(
                f"- action / option_keys note: option_keys example={list(keys)}; "
                f"schema_type={schema}; use Parsed trial examples for exact coding."
            )
            break

    if has_gamble:
        lines.append(
            "- gamble tasks: problem includes gamble_A/gamble_B dicts with probs/rewards; "
            "probs may be None for unknown probabilities."
        )
    else:
        lines.append(
            "- not a gamble task: do NOT document gamble_A/gamble_B (absent from examples)."
        )
    lines.append(
        "- Nested fields: access via full paths shown above "
        "(e.g. option_A.cues.cue1 means problem['option_A']['cues']['cue1']). "
        "Keys marked 'sometimes' are conditional — do not assume they exist on every trial."
    )
    return "\n".join(lines)


def _trial_structure_fingerprint(trial: Dict[str, Any]) -> str:
    p = trial.get("problem") or {}
    keys = tuple(sorted(str(k) for k in p.keys() if k not in _SKIP_PROBLEM_META))
    stage = p.get("stage", "")
    schema = p.get("schema_type", "")
    hist = trial.get("history") or []
    hist_keys: Set[str] = set()
    if isinstance(hist, list):
        for entry in hist[:3]:
            if isinstance(entry, dict):
                hist_keys.update(str(k) for k in entry.keys())
    return f"schema={schema}|stage={stage}|keys={keys}|hist={tuple(sorted(hist_keys))}"


def _truncate_history_entries(
    history: Any,
    *,
    max_entries: int,
) -> Tuple[Any, bool, int]:
    if not isinstance(history, list):
        return history, False, 0
    original_len = len(history)
    if original_len <= max_entries or max_entries <= 0:
        return history, False, original_len
    # Keep earliest and most recent complete entries.
    if max_entries == 1:
        return [history[-1]], True, original_len
    keep_head = max(1, max_entries // 2)
    keep_tail = max_entries - keep_head
    truncated = list(history[:keep_head]) + list(history[-keep_tail:])
    return truncated, True, original_len


def _trial_to_example_dict(
    trial: Dict[str, Any],
    index: int,
    *,
    history_max_entries: int,
) -> Dict[str, Any]:
    problem = trial.get("problem") or {}
    # Drop heavy meta aliases that are not needed for structure (keep schema_type).
    hist, was_trunc, orig_len = _truncate_history_entries(
        trial.get("history") or [],
        max_entries=history_max_entries,
    )
    out: Dict[str, Any] = {
        "index": index,
        "problem": problem,
        "action": trial.get("action"),
        "history": hist,
    }
    if was_trunc:
        out["history_truncated"] = True
        out["history_original_len"] = orig_len
        out["history_truncation_note"] = (
            f"Displayed {len(hist)} of {orig_len} history entries "
            f"(earliest + most recent); each shown entry is complete."
        )
    return out


def select_diverse_train_trials(
    trials: Sequence[Dict[str, Any]],
    *,
    max_examples: int,
) -> List[Dict[str, Any]]:
    """Deterministic selection covering distinct structure fingerprints first."""
    if not trials or max_examples <= 0:
        return []
    indexed = list(enumerate(trials))
    # Stable groups by fingerprint, ordered by first occurrence.
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    order: List[str] = []
    for i, t in indexed:
        fp = _trial_structure_fingerprint(t)
        if fp not in groups:
            groups[fp] = []
            order.append(fp)
        groups[fp].append((i, t))

    selected: List[Dict[str, Any]] = []
    # Round-robin across fingerprints.
    ptr = {fp: 0 for fp in order}
    while len(selected) < max_examples:
        progress = False
        for fp in order:
            if len(selected) >= max_examples:
                break
            idx = ptr[fp]
            bucket = groups[fp]
            if idx < len(bucket):
                selected.append(bucket[idx][1])
                ptr[fp] = idx + 1
                progress = True
        if not progress:
            break
    return selected


def serialize_train_trials_for_prompt_generation(
    trials: Sequence[Dict[str, Any]],
    *,
    char_budget: int = DEFAULT_EXAMPLE_CHAR_BUDGET,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> str:
    """
    Serialize complete JSON train-trial examples within a character budget.

    Never truncates mid-JSON; skips an example entirely if it would exceed budget.
    """
    if not trials:
        return "(no train trial examples)"
    if char_budget <= 0:
        return "(trial example budget is 0)"

    candidates = select_diverse_train_trials(trials, max_examples=max(max_examples, 1))
    blocks: List[str] = []
    used = 0
    sep = "\n\n"
    for i, trial in enumerate(candidates, start=1):
        example = _trial_to_example_dict(
            trial, i, history_max_entries=history_max_entries
        )
        block = json.dumps(example, indent=2, ensure_ascii=False, default=str)
        # Validate structural round-trip.
        json.loads(block)
        extra = len(sep) if blocks else 0
        if used + extra + len(block) > char_budget:
            if not blocks:
                # Single example too large even alone: still try with more aggressive history.
                if history_max_entries > 2:
                    return serialize_train_trials_for_prompt_generation(
                        [trial],
                        char_budget=char_budget,
                        history_max_entries=max(2, history_max_entries // 2),
                        max_examples=1,
                    )
                return (
                    "(unable to fit even one complete trial example in the configured budget; "
                    "increase dataset_prompt_example_char_budget)"
                )
            break
        blocks.append(block)
        used += extra + len(block)

    header = (
        f"(showing {len(blocks)} complete train trial example(s); "
        f"nested problem/history preserved; char_budget={char_budget})\n\n"
    )
    body = sep.join(blocks)
    # Header is outside packing budget accounting for clarity; keep body within budget.
    return header + body


def build_prompt_generation_trial_and_schema_sections(
    sample_trials: Sequence[Dict[str, Any]],
    *,
    char_budget: int = DEFAULT_EXAMPLE_CHAR_BUDGET,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> Tuple[str, str]:
    """Return (schema_summary, trial_examples_text) for prompt-generation user content."""
    schema = infer_recursive_runtime_schema(sample_trials)
    trials_text = serialize_train_trials_for_prompt_generation(
        sample_trials,
        char_budget=char_budget,
        history_max_entries=history_max_entries,
        max_examples=max_examples,
    )
    return schema, trials_text
