"""
LLM parser-plan generation, validation, and fixed execution engine.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from utils.teh_psych.dataset_loop import safe_experiment_id_for_path

SCHEMA_VERSION = "psych101_parser_plan_v1"
CACHE_MISS_CLIENT_MSG = "LLM client required because parse plan cache was not available"


class ParsePlanError(Exception):
    """Parser plan generation/validation/execution error."""


class StateMachineNotImplementedError(ParsePlanError):
    """Raised when plan requires state_machine execution."""


@dataclass
class ParsePlanRunResult:
    experiment_id: str
    status: str = "pending"
    plan: Optional[Dict[str, Any]] = None
    plan_path: str = ""
    cached: bool = False
    model_name: str = ""
    raw_response: str = ""
    prompt_text: str = ""
    validation_errors: List[str] = field(default_factory=list)
    human_review_required: bool = False
    raw_format_type: str = ""
    failure_message: str = ""
    failure_stage: str = ""
    n_rows_executed: int = 0
    trials: List[Dict[str, Any]] = field(default_factory=list)
    execution_errors: List[str] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_parse_plan_prompt_path() -> Path:
    return _repo_root() / "prompts" / "teh_psych" / "utils" / "parse_plan.txt"


def _row_metadata(row: Dict[str, Any], row_index: int, split_name: str) -> Dict[str, Any]:
    return {
        "row_index": row_index,
        "participant": row.get("participant", row.get("subject", row.get("id"))),
        "experiment": row.get("experiment"),
        "split": split_name,
        "response_fields": {
            k: row.get(k)
            for k in sorted(row.keys())
            if k not in ("text", "instruction")
        },
    }


def build_parser_plan_user_content(
    experiment_id: str,
    rows: List[Dict[str, Any]],
    *,
    row_indices: List[int],
    split_name: str = "train",
    task_description: str = "",
) -> str:
    """Build user message with representative raw rows for parser-plan LLM."""
    parts = [
        f"experiment_id: {experiment_id}",
        f"split: {split_name}",
        "",
        "## Task description",
        task_description or "(not provided)",
        "",
        "## Representative raw rows",
    ]
    for i, (row_idx, row) in enumerate(zip(row_indices, rows)):
        meta = _row_metadata(row, row_idx, split_name)
        text = str(row.get("text") or "")
        instruction = str(row.get("instruction") or "")
        if instruction and instruction not in text[:200]:
            transcript = f"INSTRUCTION:\n{instruction}\n\nTRANSCRIPT:\n{text}"
        else:
            transcript = text
        preview = transcript[:4000]
        parts.append(f"### Row {i} (dataset row_index={row_idx})")
        parts.append(f"metadata: {json.dumps(meta, default=str)}")
        parts.append("transcript:")
        parts.append(preview)
        parts.append("")
    parts.append(
        "Output only valid JSON for schema psych101_parser_plan_v1. "
        "Do not include markdown fences or commentary."
    )
    return "\n".join(parts)


def build_parser_plan_prompt(
    experiment_id: str,
    rows: List[Dict[str, Any]],
    *,
    row_indices: List[int],
    template_path: Path,
    split_name: str = "train",
    task_description: str = "",
) -> str:
    template = template_path.read_text(encoding="utf-8")
    user = build_parser_plan_user_content(
        experiment_id,
        rows,
        row_indices=row_indices,
        split_name=split_name,
        task_description=task_description,
    )
    return f"{template}\n\n---\n\n{user}"


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ParsePlanError("empty LLM response")
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParsePlanError(f"invalid JSON in LLM response: {exc}") from exc
    if not isinstance(obj, dict):
        raise ParsePlanError("parser plan must be a JSON object")
    return obj


def validate_parser_plan(plan: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if not plan.get("experiment_id"):
        errors.append("missing experiment_id")
    if plan.get("task_type") != "categorical_choice":
        errors.append("task_type must be categorical_choice")
    line_classifier = plan.get("line_classifier")
    if not isinstance(line_classifier, dict) or not line_classifier.get("line_types"):
        errors.append("line_classifier.line_types required")
    trial_extraction = plan.get("trial_extraction")
    if not isinstance(trial_extraction, dict):
        errors.append("trial_extraction required")
    else:
        if not trial_extraction.get("boundary_strategy"):
            errors.append("trial_extraction.boundary_strategy required")
        action_rule = trial_extraction.get("action_rule")
        if not isinstance(action_rule, dict):
            errors.append("trial_extraction.action_rule required")
    option_norm = plan.get("option_normalization")
    if not isinstance(option_norm, dict) or not option_norm.get("strategy"):
        errors.append("option_normalization.strategy required")
    if "validation_expectations" not in plan:
        errors.append("validation_expectations required")
    sm = plan.get("state_machine") or {}
    if isinstance(sm, dict) and sm.get("enabled"):
        errors.append("state_machine.enabled=true (not supported by engine v1)")
    return errors


def parse_plan_cache_path(cache_dir: Path, experiment_id: str) -> Path:
    return cache_dir / safe_experiment_id_for_path(experiment_id) / "parse_plan.json"


def load_cached_parse_plan(cache_dir: Path, experiment_id: str) -> Optional[Dict[str, Any]]:
    path = parse_plan_cache_path(cache_dir, experiment_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_parse_plan_cache(cache_dir: Path, experiment_id: str, plan: Dict[str, Any]) -> Path:
    path = parse_plan_cache_path(cache_dir, experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path


def generate_parser_plan_via_llm(
    client: OpenAI,
    *,
    model_name: str,
    prompt_text: str,
    max_tokens: int = 4000,
) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _regex_flags(flags: str) -> int:
    out = 0
    if "i" in (flags or ""):
        out |= re.I
    if "m" in (flags or ""):
        out |= re.M
    if "s" in (flags or ""):
        out |= re.S
    return out


def _line_matches_detection(line: str, detection: Dict[str, Any]) -> bool:
    if not isinstance(detection, dict):
        return False
    if detection.get("regex"):
        try:
            return bool(
                re.search(
                    detection["regex"],
                    line,
                    _regex_flags(str(detection.get("flags", "i"))),
                )
            )
        except re.error:
            return False
    if detection.get("prefix") and line.startswith(detection["prefix"]):
        return True
    if detection.get("contains") and detection["contains"] in line:
        return True
    return False


def classify_transcript_lines(text: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    lc = plan.get("line_classifier") or {}
    line_types = lc.get("line_types") or []
    join_mode = lc.get("multiline_chunk_join") or "line"
    raw_lines = text.splitlines() if join_mode == "line" else re.split(r"\n\s*\n", text)
    classified: List[Dict[str, Any]] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        type_id = "trial_stimulus"
        for spec in line_types:
            det = spec.get("detection") or {}
            if _line_matches_detection(line, det):
                type_id = spec.get("type_id", type_id)
                break
        classified.append({"line": line, "type_id": type_id})
    return classified


def _split_blocks(classified: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    rule = plan.get("block_boundary_rule") or {}
    strategy = rule.get("strategy", "single_block")
    if strategy == "single_block":
        return [classified]
    if strategy == "regex_header" and rule.get("regex"):
        blocks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        pat = re.compile(rule["regex"], _regex_flags("i"))
        for item in classified:
            if pat.search(item["line"]) and current:
                blocks.append(current)
                current = [item]
            else:
                current.append(item)
        if current:
            blocks.append(current)
        return blocks or [classified]
    if strategy == "keyword_header" and rule.get("keyword"):
        kw = str(rule["keyword"])
        blocks = []
        current = []
        for item in classified:
            if item["line"].startswith(kw) and current:
                blocks.append(current)
                current = [item]
            else:
                current.append(item)
        if current:
            blocks.append(current)
        return blocks or [classified]
    if strategy == "blank_line_paragraphs":
        return [classified]
    return [classified]


def _capture_from_rule(line: str, capture: Dict[str, Any]) -> Optional[str]:
    if not capture:
        return None
    regex = capture.get("regex")
    if not regex:
        return None
    group = capture.get("group", 1)
    try:
        group_idx = int(group) if str(group).isdigit() else 1
    except (TypeError, ValueError):
        group_idx = 1
    m = re.search(regex, line, _regex_flags("i"))
    if not m:
        return None
    try:
        return m.group(group_idx)
    except IndexError:
        return m.group(1) if m.lastindex else m.group(0)


def _normalize_key_list(
    keys: List[str],
    action_id_order: str,
) -> List[str]:
    keys = [str(k) for k in keys if k is not None]
    if action_id_order == "sorted_raw_key":
        return sorted(keys)
    if action_id_order == "first_seen_in_transcript":
        seen: List[str] = []
        for k in keys:
            if k not in seen:
                seen.append(k)
        return seen
    return keys


def _option_map_from_plan(
    plan: Dict[str, Any],
    *,
    instruction: str,
    block_lines: List[Dict[str, Any]],
    trial_lines: List[Dict[str, Any]],
) -> Dict[str, int]:
    norm = plan.get("option_normalization") or {}
    strategy = norm.get("strategy", "fixed_keys_from_instruction")
    action_id_order = norm.get("action_id_order", "instruction_order")
    explicit = norm.get("raw_response_to_action_id_mapping") or {}
    if explicit:
        return {str(k): int(v) for k, v in explicit.items()}

    keys: List[str] = []
    if strategy == "fixed_keys_from_instruction":
        instr_re = norm.get("instruction_regex") or r"press\s+([A-Z])"
        keys = re.findall(instr_re, instruction, re.I)
    elif strategy == "per_block_key_list":
        for item in block_lines:
            if item["type_id"] == "block_header":
                keys.extend(re.findall(r"<<([A-Z])>>|press\s+([A-Z])", item["line"], re.I))
                flat = [g for pair in keys for g in pair if g]
                keys = flat
        keys = _normalize_key_list(keys, action_id_order)
    elif strategy == "per_trial_available_keys":
        for item in trial_lines:
            keys.extend(re.findall(r"<<([A-Z])>>", item["line"]))
    elif strategy == "numeric_range_from_context":
        keys = [str(i) for i in range(2)]

    keys = _normalize_key_list(keys, action_id_order)
    if len(keys) < 2:
        keys = ["0", "1"]
    return {k: i for i, k in enumerate(keys)}


def _parse_feedback(line: str, plan: Dict[str, Any]) -> Any:
    fb_rule = (plan.get("trial_extraction") or {}).get("feedback_rule") or {}
    if not fb_rule.get("regex"):
        return None
    m = re.search(fb_rule["regex"], line, _regex_flags("i"))
    if not m:
        return None
    value_type = fb_rule.get("value_type", "points")
    if value_type == "points":
        for g in m.groups():
            if g is not None:
                try:
                    return float(g)
                except ValueError:
                    continue
        return None
    if value_type == "boolean":
        text = m.group(0).lower()
        if "correct" in text or "right" in text:
            return True
        if "incorrect" in text or "wrong" in text:
            return False
        return None
    if value_type == "string":
        return m.group(1) if m.lastindex else m.group(0)
    return None


def _is_context_only_trial(trial_lines: List[Dict[str, Any]], plan: Dict[str, Any]) -> bool:
    rule = (plan.get("trial_extraction") or {}).get("context_only_trial_rule") or {}
    regex_any = rule.get("regex_any") or []
    line_type = rule.get("line_type")
    for item in trial_lines:
        if line_type and item.get("type_id") == line_type:
            return True
        for rx in regex_any:
            if re.search(rx, item["line"], re.I):
                return True
        if re.search(r"You are instructed to press", item["line"], re.I):
            return True
    return False


def _extract_stimulus_fields(trial_lines: List[Dict[str, Any]], plan: Dict[str, Any]) -> Dict[str, Any]:
    stimulus: Dict[str, Any] = {}
    fields = (plan.get("trial_extraction") or {}).get("stimulus_fields") or []
    blob = "\n".join(item["line"] for item in trial_lines)
    for spec in fields:
        name = spec.get("field_name")
        regex = spec.get("regex")
        if not name or not regex:
            continue
        m = re.search(regex, blob, _regex_flags("i"))
        if not m:
            continue
        group = spec.get("group", 1)
        try:
            raw = m.group(int(group) if str(group).isdigit() else 1)
        except (IndexError, ValueError):
            raw = m.group(0)
        cast = spec.get("cast", "str")
        if cast == "int":
            stimulus[name] = int(raw)
        elif cast == "float":
            stimulus[name] = float(raw)
        elif cast == "list_int":
            stimulus[name] = [int(x) for x in re.findall(r"-?\d+", raw)]
        else:
            stimulus[name] = raw
    return stimulus


def _extract_context_fields(block_lines: List[Dict[str, Any]], plan: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    fields = (plan.get("trial_extraction") or {}).get("context_fields") or []
    blob = "\n".join(item["line"] for item in block_lines)
    for spec in fields:
        name = spec.get("field_name")
        if not name:
            continue
        if spec.get("source") == "constant":
            context[name] = spec.get("value")
            continue
        regex = spec.get("regex")
        if regex:
            m = re.search(regex, blob, _regex_flags("i"))
            if m:
                context[name] = m.group(1) if m.lastindex else m.group(0)
    return context


def _trials_from_block(
    block_lines: List[Dict[str, Any]],
    plan: Dict[str, Any],
    *,
    instruction: str,
    row_index: int,
    participant: Any,
    block_id: int,
) -> List[Dict[str, Any]]:
    te = plan.get("trial_extraction") or {}
    boundary = te.get("boundary_strategy", "one_press_per_trial")
    action_rule = te.get("action_rule") or {}
    source_type = action_rule.get("source_line_type", "action_press")
    capture = action_rule.get("capture") or {"regex": r"<<([A-Z])>>", "group": 1}
    history_rule = plan.get("history_rule") or {}
    history_scope = history_rule.get("scope", "block")
    history_fields = history_rule.get("fields") or ["action", "feedback"]
    include_context = bool(history_rule.get("include_context_only", False))

    raw_trials: List[List[Dict[str, Any]]] = []
    if boundary == "one_press_per_trial":
        for item in block_lines:
            if item["type_id"] == source_type or re.search(
                capture.get("regex", r"<<([A-Z])>>"), item["line"], re.I
            ):
                raw_trials.append([item])
    elif boundary == "stimulus_then_press":
        buf: List[Dict[str, Any]] = []
        for item in block_lines:
            buf.append(item)
            if item["type_id"] == source_type or re.search(
                capture.get("regex", r"<<([A-Z])>>"), item["line"], re.I
            ):
                raw_trials.append(buf)
                buf = []
        if buf and raw_trials:
            raw_trials[-1] = raw_trials[-1] + buf
    elif boundary in ("game_segment", "round_state_machine", "study_then_test_pair"):
        raise ParsePlanError(f"boundary_strategy {boundary!r} not implemented in parser engine v1")
    else:
        raise ParsePlanError(f"unknown boundary_strategy {boundary!r}")

    context = _extract_context_fields(block_lines, plan)
    context["block_id"] = block_id
    trials: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    for trial_lines in raw_trials:
        press_line = trial_lines[-1]["line"]
        raw_key = _capture_from_rule(press_line, capture)
        if raw_key is None:
            m = re.search(r"<<([A-Z])>>", press_line)
            raw_key = m.group(1) if m else None
        if raw_key is None:
            continue

        key_map = _option_map_from_plan(
            plan,
            instruction=instruction,
            block_lines=block_lines,
            trial_lines=trial_lines,
        )
        if raw_key not in key_map:
            upper = str(raw_key).upper()
            if upper in key_map:
                raw_key = upper
            elif str(raw_key) in key_map:
                pass
            else:
                continue
        target_action = int(key_map[str(raw_key).upper()] if str(raw_key).upper() in key_map else key_map[raw_key])

        options = [
            {"action": i, "label": k}
            for k, i in sorted(key_map.items(), key=lambda kv: kv[1])
        ]
        stimulus = _extract_stimulus_fields(trial_lines, plan)
        feedback = None
        for item in trial_lines:
            fb = _parse_feedback(item["line"], plan)
            if fb is not None:
                feedback = fb

        is_context_only = _is_context_only_trial(trial_lines, plan)
        is_prediction = not is_context_only

        trial = {
            "problem": {
                "options": options,
                "stimulus": stimulus,
                "context": dict(context),
                "experiment_id": plan.get("experiment_id"),
            },
            "history": list(history) if history_scope in ("block", "game", "participant") else [],
            "action": target_action,
            "target_action": target_action,
            "feedback": feedback,
            "is_prediction_target": is_prediction,
            "_meta": {
                "row_index": row_index,
                "participant": participant,
                "block_id": block_id,
                "raw_key": raw_key,
            },
        }
        trials.append(trial)

        hist_entry = {k: trial.get(k) for k in history_fields if k in trial}
        hist_entry["action"] = target_action
        if feedback is not None:
            hist_entry["feedback"] = feedback
        if include_context or is_prediction:
            history.append(hist_entry)

    return trials


def execute_parser_plan_on_row(
    plan: Dict[str, Any],
    row: Dict[str, Any],
    *,
    row_index: int = 0,
) -> List[Dict[str, Any]]:
    sm = plan.get("state_machine") or {}
    if isinstance(sm, dict) and sm.get("enabled"):
        raise StateMachineNotImplementedError("state_machine_not_implemented")

    text = str(row.get("text") or "")
    instruction = str(row.get("instruction") or "")
    if instruction and instruction not in text:
        full_text = instruction + "\n\n" + text
    else:
        full_text = text
    participant = row.get("participant", row.get("subject"))

    classified = classify_transcript_lines(full_text, plan)
    blocks = _split_blocks(classified, plan)
    out: List[Dict[str, Any]] = []
    for block_id, block_lines in enumerate(blocks):
        out.extend(
            _trials_from_block(
                block_lines,
                plan,
                instruction=instruction,
                row_index=row_index,
                participant=participant,
                block_id=block_id,
            )
        )
    return out


def execute_parser_plan_on_rows(
    plan: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    row_indices: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    indices = row_indices if row_indices is not None else list(range(len(rows)))
    all_trials: List[Dict[str, Any]] = []
    errors: List[str] = []
    for row_idx, row in zip(indices, rows):
        try:
            trials = execute_parser_plan_on_row(plan, row, row_index=row_idx)
            all_trials.extend(trials)
        except StateMachineNotImplementedError as exc:
            raise
        except Exception as exc:
            errors.append(f"row {row_idx}: {type(exc).__name__}: {exc}")
    return all_trials, errors


def select_parse_plan_example_rows(
    rows: List[Dict[str, Any]],
    row_indices: List[int],
    n_parse_plan_rows: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    n = max(3, min(10, int(n_parse_plan_rows)))
    n = min(n, len(rows))
    return rows[:n], row_indices[:n]


def run_parse_plan_pipeline(
    *,
    client: Optional[OpenAI],
    experiment_id: str,
    rows: List[Dict[str, Any]],
    row_indices: List[int],
    debug_dir: Path,
    template_path: Path,
    model_name: str,
    split_name: str = "train",
    task_description: str = "",
    reuse_cache: bool = False,
    cache_dir: Optional[Path] = None,
    parse_plan_max_tokens: int = 4000,
    n_parse_plan_rows: int = 5,
) -> ParsePlanRunResult:
    result = ParsePlanRunResult(experiment_id=experiment_id, model_name=model_name)
    debug_dir.mkdir(parents=True, exist_ok=True)
    result.failure_stage = "build_parse_plan_prompt"
    try:
        example_rows, example_indices = select_parse_plan_example_rows(
            rows, row_indices, n_parse_plan_rows
        )
        result.prompt_text = build_parser_plan_prompt(
            experiment_id,
            example_rows,
            row_indices=example_indices,
            template_path=template_path,
            split_name=split_name,
            task_description=task_description,
        )
        (debug_dir / "parse_plan_prompt.txt").write_text(result.prompt_text, encoding="utf-8")
    except Exception as exc:
        result.status = "prompt_failed"
        result.failure_stage = "build_parse_plan_prompt"
        result.failure_message = str(exc)
        return result

    plan: Optional[Dict[str, Any]] = None
    if reuse_cache and cache_dir is not None:
        cached = load_cached_parse_plan(cache_dir, experiment_id)
        if cached is not None:
            cache_errors = validate_parser_plan(cached)
            if not cache_errors:
                plan = cached
                result.cached = True
                result.status = "cached"
            else:
                (debug_dir / "parse_plan_cache_validation_errors.json").write_text(
                    json.dumps(cache_errors, indent=2) + "\n", encoding="utf-8"
                )

    if plan is None:
        result.failure_stage = "generate_parse_plan"
        if client is None:
            result.status = "generate_failed"
            result.failure_message = CACHE_MISS_CLIENT_MSG
            return result
        try:
            result.raw_response = generate_parser_plan_via_llm(
                client,
                model_name=model_name,
                prompt_text=result.prompt_text,
                max_tokens=parse_plan_max_tokens,
            )
        except Exception as exc:
            result.status = "generate_failed"
            result.failure_message = str(exc)
            return result
        (debug_dir / "parse_plan_raw_response.txt").write_text(
            result.raw_response, encoding="utf-8"
        )
        try:
            plan = extract_json_from_llm_response(result.raw_response)
        except ParsePlanError as exc:
            result.status = "generate_failed"
            result.failure_message = str(exc)
            return result
        result.status = "generated"
        if cache_dir is not None:
            result.plan_path = str(save_parse_plan_cache(cache_dir, experiment_id, plan))

    result.plan = plan
    (debug_dir / "parse_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    result.plan_path = result.plan_path or str(debug_dir / "parse_plan.json")

    result.failure_stage = "validate_parse_plan"
    validation_errors = validate_parser_plan(plan)
    result.validation_errors = validation_errors
    if validation_errors:
        (debug_dir / "parse_plan_validation_errors.json").write_text(
            json.dumps(validation_errors, indent=2) + "\n", encoding="utf-8"
        )
        result.status = "validation_failed"
        result.failure_message = validation_errors[0]
        return result

    result.human_review_required = bool(plan.get("human_review_required"))
    sa = plan.get("source_assessment") or {}
    result.raw_format_type = str(sa.get("raw_format_type") or "")

    result.failure_stage = "execute_parse_plan"
    try:
        trials, exec_errors = execute_parser_plan_on_rows(
            plan, rows, row_indices=row_indices
        )
    except StateMachineNotImplementedError:
        result.status = "execute_failed"
        result.failure_message = "state_machine_not_implemented"
        return result

    result.execution_errors = exec_errors
    result.trials = trials
    result.n_rows_executed = len(rows)
    if not trials:
        result.status = "execute_failed"
        result.failure_message = exec_errors[0] if exec_errors else "no trials parsed"
        return result

    preview = [t for t in trials if t.get("is_prediction_target", True)][:12]
    (debug_dir / "adapter_trials_preview.json").write_text(
        json.dumps(preview, indent=2, default=str) + "\n", encoding="utf-8"
    )
    result.status = "executed"
    result.failure_stage = "execute_parse_plan"
    return result
