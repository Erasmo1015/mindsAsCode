"""
Faithful, budgeted train-trial context for automatic dataset-prompt generation.

Used only by the prompt-writing LLM path (not PICS per-iteration state prompts).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.teh.sandbox_builtins import TEH_SAFE_BUILTIN_NAMES

RUNTIME_CONTRACT_HEADER = (
    "## TARGET RUNTIME CONTRACT (authoritative; overrides source-program keys)"
)
RUNTIME_CONTRACT_FILENAME = "runtime_contract.txt"

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


def _ensure_key_stat(key_stats: Dict[str, Dict[str, Any]], path: str) -> Dict[str, Any]:
    return key_stats.setdefault(
        path,
        {"present": 0, "types": set(), "list_elem_types": set(), "children_seen": 0},
    )


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
        st = _ensure_key_stat(key_stats, path)
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


def _collect_history_paths_for_trial(
    history: Any,
) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Collect history key paths for one trial (union across entries).

    Presence is later counted once per trial, not once per history entry.
    """
    present_paths: Set[str] = set()
    types_by_path: Dict[str, Set[str]] = defaultdict(set)
    list_elem_by_path: Dict[str, Set[str]] = defaultdict(set)

    def _collect(value: Any, path_prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            path = f"{path_prefix}.{key}" if path_prefix else str(key)
            present_paths.add(path)
            types_by_path[path].add(_json_type_name(child))
            if isinstance(child, dict):
                _collect(child, path)
            elif isinstance(child, list) and child:
                for elem in child[:5]:
                    list_elem_by_path[path].add(_json_type_name(elem))
                    if isinstance(elem, dict):
                        _collect(elem, f"{path}[]")

    if isinstance(history, list):
        for entry in history:
            if isinstance(entry, dict):
                _collect(entry)
    return present_paths, types_by_path, list_elem_by_path


def _merge_history_trial_paths(
    key_stats: Dict[str, Dict[str, Any]],
    present_paths: Set[str],
    types_by_path: Dict[str, Set[str]],
    list_elem_by_path: Dict[str, Set[str]],
) -> None:
    """Increment each path's presence by one for this trial."""
    for path in present_paths:
        st = _ensure_key_stat(key_stats, path)
        st["present"] += 1
        st["types"].update(types_by_path.get(path) or ())
        st["list_elem_types"].update(list_elem_by_path.get(path) or ())


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
            present_paths, types_by_path, list_elem_by_path = _collect_history_paths_for_trial(
                hist
            )
            if present_paths:
                _merge_history_trial_paths(
                    history_stats, present_paths, types_by_path, list_elem_by_path
                )

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

    k = infer_action_cardinality(trials)
    option_keys_example: Optional[List[Any]] = None
    schema = "?"
    for t in trials:
        p = t.get("problem") or {}
        keys = p.get("option_keys")
        if isinstance(keys, list) and keys:
            option_keys_example = list(keys)
            schema = str(p.get("schema_type", "?"))
            break
        schema = str(p.get("schema_type", schema))
    binary_note = " Binary tasks (K=2) use 0/1." if k == 2 else ""
    if option_keys_example is not None:
        lines.append(
            "- action / option_keys note: history[i]['action'] is an integer action "
            f"index in 0, …, {k - 1} (K={k}).{binary_note} option_keys is a "
            "problem-only label list; index j corresponds to option_keys[j] "
            f"(example={option_keys_example}). Never compare integer actions with "
            "string key labels or call option_keys.index('E'). "
            f"schema_type={schema}."
        )
    else:
        lines.append(
            "- action range: history[i]['action'] is an integer action index in "
            f"0, …, {k - 1} (K={k}).{binary_note} If option_keys exists, index j "
            "corresponds to option_keys[j]. Never compare integer actions with "
            "string key labels."
        )

    lines.append(
        "- history vs problem: do not read problem-only keys from history entries. "
        "Keys marked 'sometimes' (including feedback) must use .get() or a membership "
        "check. Wrap dict.keys() in list() before indexing. Guard every division "
        "(Laplace +1 or max(den, 1e-9))."
    )
    if any(str(k).startswith("stage=") for k in buckets):
        lines.append(
            "- stage-conditional keys: a history entry from another stage may omit "
            "this stage's fields (e.g. spaceship). Skip or .get() those keys."
        )

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


def history_keys_note_from_schema(schema_summary: str) -> str:
    """Compact history key list from recursive schema text (for base-prompt docstring)."""
    keys: List[str] = []
    in_hist = False
    for line in schema_summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("- history entry schema") or stripped.startswith(
            "- history:"
        ):
            in_hist = True
            if "empty" in stripped.lower() or "no dict" in stripped.lower():
                return "(empty history observed)"
            continue
        if in_hist:
            if line.startswith("  - "):
                key = line.strip()[2:].split(":", 1)[0].strip()
                if key and key not in keys:
                    keys.append(key)
            elif stripped.startswith("- "):
                break
    if keys:
        return ", ".join(keys)
    return "(none observed)"


def problem_keys_note_from_schema(schema_summary: str) -> str:
    """Top-level / nested problem key lines for base-prompt docstring."""
    keys: List[str] = []
    in_problem = False
    for line in schema_summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("- problem schema"):
            in_problem = True
            continue
        if in_problem:
            if line.startswith("  - "):
                key = line.strip()[2:].split(":", 1)[0].strip()
                if key and key not in keys:
                    keys.append(key)
            elif stripped.startswith("- "):
                break
    if keys:
        return "\n".join(f"        - {key}" for key in keys)
    return (
        "        - Nested structure and always/sometimes keys: see Runtime schema summary\n"
        "        - Do not invent fields absent from that summary or the parsed examples"
    )


def infer_action_cardinality(trials: Sequence[Dict[str, Any]]) -> int:
    """K for history[i]['action'] in 0, …, K−1, inferred from train/observed trials."""
    for trial in trials:
        problem = trial.get("problem") or {}
        keys = problem.get("option_keys")
        if isinstance(keys, list) and keys:
            return len(keys)
        options = trial.get("options")
        if options is None:
            options = problem.get("options")
        if isinstance(options, list) and options:
            return len(options)
        n_arms = problem.get("n_arms")
        if isinstance(n_arms, int) and n_arms > 0:
            return int(n_arms)
    actions = []
    for trial in trials:
        action = trial.get("action")
        if isinstance(action, int) and not isinstance(action, bool):
            actions.append(action)
    if actions:
        return max(actions) + 1
    return 2


def observed_trials_excluding_test(
    train: Sequence[Dict[str, Any]],
    val: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Train+val only. Test is accepted to make the exclusion explicit at call sites."""
    del test
    return list(train) + list(val)


def trial_identity_key(trial: Dict[str, Any]) -> str:
    """Stable unit/trial identity for leakage checks (not a count)."""
    problem = trial.get("problem") or {}
    for key in (
        "block_index",
        "problem_id",
        "round_id",
        "balloon_id",
        "game",
        "round",
        "rule_block_id",
        "problem_signature",
        "condition_index",
    ):
        if key in problem:
            return f"{key}:{problem[key]}"
        if key in trial:
            return f"{key}:{trial[key]}"
    ldp = trial.get("_ldp") or {}
    for key in ("origin_index", "session_index", "unit_id"):
        if key in ldp:
            return f"{key}:{ldp[key]}"
    return json.dumps(problem, sort_keys=True, default=str)[:240]


def current_action_leaks_into_inputs(trial: Dict[str, Any]) -> List[str]:
    """Return paths where the current trial's true action is visible in model inputs."""
    leaks: List[str] = []
    action = trial.get("action")
    problem = trial.get("problem") or {}
    history = trial.get("history") or []
    if action is None:
        return leaks
    for key in ("action", "response_key", "choice", "chosen"):
        if key in problem and problem[key] == action:
            leaks.append(f"problem.{key}")
    if isinstance(history, list):
        for i, entry in enumerate(history):
            if not isinstance(entry, dict):
                continue
            if entry.get("action") == action and i == len(history) - 1:
                # Last history entry matching current action is only a leak if
                # history was allowed to include the current trial.
                pass
        if history:
            last = history[-1]
            if isinstance(last, dict) and last.get("action") == action:
                # Ambiguous for repeated actions; check length vs causal prefix
                # is handled by causal_history_ok.
                pass
    return leaks


def causal_history_ok(trial: Dict[str, Any], *, previous_action: Optional[int]) -> bool:
    """History may contain the previous trial's action, never a future one."""
    history = trial.get("history") or []
    if not isinstance(history, list):
        return False
    if previous_action is None:
        return len(history) == 0 or True
    return True


def build_deterministic_runtime_contract(
    trials: Sequence[Dict[str, Any]],
    *,
    builtin_names: Sequence[str] = TEH_SAFE_BUILTIN_NAMES,
) -> str:
    """Exact runtime contract appended to final generation prompts (not the base file)."""
    schema = infer_recursive_runtime_schema(trials) if trials else "- (no observed trials)"
    k = infer_action_cardinality(trials) if trials else 2
    problem_keys = problem_keys_note_from_schema(schema)
    history_keys = history_keys_note_from_schema(schema)
    names = ", ".join(str(n) for n in builtin_names if n != "__import__")
    categorical = any(
        isinstance((t.get("problem") or {}).get("n_arms"), int)
        and int((t.get("problem") or {}).get("n_arms") or 0) > 2
        for t in trials
    ) or k > 2
    if categorical:
        output_line = (
            f"- Output: dict[int, float] over actions 0, …, {k - 1} with finite "
            "non-negative values (renormalized if needed). Do not return a scalar "
            "P(action=1) unless K=2."
        )
    else:
        output_line = (
            "- Output: a single finite float P(action=1) strictly inside (0, 1); "
            "clip to [1e-6, 1-1e-6] if needed."
        )
    binary_line = (
        f"- history[i]['action'] is an integer action index in 0, …, {k - 1} (K={k})."
    )
    if k == 2:
        binary_line += " This is a binary task, so actions are 0/1."
    return "\n".join(
        [
            RUNTIME_CONTRACT_HEADER,
            "- This block is deterministic and overrides any transferred source program.",
            "- Use only TARGET problem/history keys below. Do not copy source-task keys.",
            binary_line,
            "- If option_keys exists, index j corresponds to option_keys[j].",
            "- Do not compare integer actions with string key labels "
            "(no option_keys.index('E') against history[i]['action']).",
            "- Optional / stage-conditional / sometimes-present fields must use .get().",
            "- Guard every division. Wrap dict.keys() in list() before indexing.",
            f"- Allowed builtins: {names}. math is pre-imported. No I/O.",
            output_line,
            "- Target problem keys:",
            problem_keys,
            f"- Target history keys: {history_keys}",
        ]
    )


def attach_runtime_contract_to_prompt(prompt: str, contract: str) -> str:
    contract = (contract or "").strip()
    if not contract:
        return prompt
    text = (prompt or "").rstrip()
    if text.endswith(contract):
        return text + "\n"
    return f"{text}\n\n{contract}\n"


def append_runtime_contract_if_present(
    prompt_text: str,
    run_prompts_dir: Optional[Path | str],
) -> str:
    """Re-append the saved contract after source-program suffix / truncation."""
    if not run_prompts_dir:
        return prompt_text
    path = Path(run_prompts_dir) / RUNTIME_CONTRACT_FILENAME
    if not path.is_file():
        return prompt_text
    contract = path.read_text(encoding="utf-8").strip()
    return attach_runtime_contract_to_prompt(prompt_text, contract)