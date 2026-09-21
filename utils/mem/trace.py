"""Pure helpers for PICS MEM tracing and offline annotation validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.mem.annotation_context import (
    ANNOTATION_SAFETY_MARGIN_TOKENS,
    ANNOTATION_VLLM_MAX_MODEL_LEN,
    PARTICIPANT_DEFAULT_MAX_TOKENS,
)

# Schema v1 flat taxonomy (legacy annotations only). Prefer utils.mem.schema_v2.
MOTIF_TAXONOMY = (
    "value_or_expected_value",
    "probability_or_risk",
    "history_or_memory",
    "feedback_or_reward",
    "learning_or_adaptation",
    "gating_or_conditional",
    "aggregation_or_combination",
    "nonlinear_transformation",
    "parameter_or_threshold_change",
    "simplification_or_removal",
    "other",
    "no_meaningful_change",
)

_MOTIF_SET = frozenset(MOTIF_TAXONOMY)

# Re-export schema v2 constants for convenience.
from utils.mem.schema_v2 import (  # noqa: E402
    BEHAVIORAL_MOTIFS_V2,
    SCHEMA_VERSION,
    STRUCTURAL_OPERATIONS_V2,
    validate_annotation_response_v2,
)
_TEST_METRIC_KEYS = frozenset(
    {
        "test_loglik",
        "test_acc",
        "test_accuracy",
        "test_mse",
        "gated_test_loglik",
    }
)

# Slim traces omit embedded program text; annotate resolves code from run artifacts.
# Legacy fat traces with a non-empty "code" field remain supported.
MEM_TRACE_FORMAT = "slim_path_ref_v1"
DEFAULT_EMBED_CODE = False


def mem_trace_path(participant_dir: Path | str) -> Path:
    return Path(participant_dir) / "mem_trace.jsonl"


def candidate_code_relpath(
    *,
    phase: str,
    iteration: int,
    candidate_idx: int,
    source: Optional[str] = None,
) -> str:
    """Participant-relative path where TEH writes candidate_*.py."""
    if str(phase) == "explore" or str(source) == "explore":
        return f"explore_phase/candidates/candidate_{int(candidate_idx)}.py"
    return f"iteration_{int(iteration)}/candidates/candidate_{int(candidate_idx)}.py"


def resolve_program_code(
    participant_dir: Path | str,
    *,
    program_id: Optional[str] = None,
    code_path: Optional[str] = None,
    phase: Optional[str] = None,
    iteration: Optional[Any] = None,
    candidate_idx: Optional[Any] = None,
    source: Optional[str] = None,
    embedded_code: Optional[str] = None,
) -> Optional[str]:
    """Load program source for annotation.

    Preference order:
      1) non-empty embedded_code (legacy fat traces)
      2) explicit code_path (relative to participant_dir, or absolute)
      3) phase/iteration/candidate_idx layout
      4) program_id heuristics (baseline / explore_candidate_N / iteration_X_candidate_Y /
         elite or initial pool filenames containing the id)
    """
    if isinstance(embedded_code, str) and embedded_code.strip():
        return embedded_code

    root = Path(participant_dir)
    candidates: List[Path] = []

    if code_path:
        p = Path(code_path)
        candidates.append(p if p.is_absolute() else root / p)

    if candidate_idx is not None and (phase is not None or source is not None):
        rel = candidate_code_relpath(
            phase=str(phase or ""),
            iteration=int(iteration or 0),
            candidate_idx=int(candidate_idx),
            source=source,
        )
        candidates.append(root / rel)

    pid = str(program_id) if program_id is not None else ""
    if pid:
        if pid in ("baseline", "global_baseline"):
            pool = root / "initial_pool_from_global"
            for name in (
                "000_global_baseline.py",
                "001_global_baseline.py",
                "000_baseline.py",
            ):
                candidates.append(pool / name)
            if pool.is_dir():
                candidates.extend(sorted(pool.glob("*baseline*.py")))
        elif pid.startswith("explore_candidate_"):
            try:
                idx = int(pid.rsplit("_", 1)[-1])
                candidates.append(root / f"explore_phase/candidates/candidate_{idx}.py")
            except ValueError:
                pass
        elif pid.startswith("iteration_") and "_candidate_" in pid:
            # iteration_{N}_candidate_{K}
            try:
                mid = pid[len("iteration_") :]
                it_s, cand_s = mid.split("_candidate_", 1)
                candidates.append(
                    root / f"iteration_{int(it_s)}/candidates/candidate_{int(cand_s)}.py"
                )
            except ValueError:
                pass

        for pool_name in ("evolution_elite_pool", "initial_pool_from_global"):
            pool = root / pool_name
            if not pool.is_dir():
                continue
            # Ranked files are typically NNN_<program_id>.py
            candidates.extend(sorted(pool.glob(f"*_{pid}.py")))
            candidates.extend(sorted(pool.glob(f"*{pid}*.py")))

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text
    return None


def hydrate_trace_code_fields(
    rec: Dict[str, Any],
    participant_dir: Path | str,
) -> Dict[str, Any]:
    """Return a shallow copy with ``code`` filled when missing (slim traces)."""
    out = dict(rec)
    code = resolve_program_code(
        participant_dir,
        program_id=out.get("candidate_id") or out.get("program_id"),
        code_path=out.get("code_path"),
        phase=out.get("phase"),
        iteration=out.get("iteration"),
        candidate_idx=out.get("candidate_idx"),
        source=out.get("source"),
        embedded_code=out.get("code"),
    )
    if code is not None:
        out["code"] = code
    return out


def hydrate_parent_record(
    parent: Dict[str, Any],
    participant_dir: Path | str,
) -> Dict[str, Any]:
    """Fill parent ``code`` from disk when absent."""
    out = dict(parent)
    code = resolve_program_code(
        participant_dir,
        program_id=out.get("program_id"),
        code_path=out.get("code_path"),
        embedded_code=out.get("code"),
    )
    if code is not None:
        out["code"] = code
    return out


def json_safe_value(value: Any) -> Any:
    """Convert values for UTF-8 JSONL (non-finite floats -> null)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe_value(value.item())
        except Exception:
            pass
    return value


def append_mem_trace_record(path: Path | str, record: Dict[str, Any]) -> None:
    """Append one JSONL record (UTF-8). Safe for distinct participant files."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe_value(record)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    with out.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()


def selection_score_from_elite_tuple(parent_tuple: Sequence[Any]) -> Optional[float]:
    """Pool-ranking score stored at elite tuple index 1 (train or train_val)."""
    if len(parent_tuple) < 2:
        return None
    try:
        score = float(parent_tuple[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def parent_record_from_elite_tuple(
    parent_tuple: Sequence[Any],
    *,
    val_loglik: Optional[float] = None,
    train_loglik: Optional[float] = None,
    embed_code: bool = DEFAULT_EMBED_CODE,
    code_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize one selected parent for iteration_context (no test metrics).

    Default is slim: program_id + scores only (no embedded code).
    """
    program_id = str(parent_tuple[3]) if len(parent_tuple) > 3 else ""
    code = parent_tuple[0] if parent_tuple else ""
    selection_score = selection_score_from_elite_tuple(parent_tuple)
    train_ll = train_loglik
    if train_ll is None and len(parent_tuple) > 6 and parent_tuple[6] is not None:
        try:
            train_ll = float(parent_tuple[6])
        except (TypeError, ValueError):
            train_ll = None
    if train_ll is None and selection_score is not None and val_loglik is None:
        # train-only ranking: index 1 is train loglik
        train_ll = selection_score
    record: Dict[str, Any] = {
        "program_id": program_id,
        "selection_score": selection_score,
        "train_loglik": train_ll if train_ll is not None and math.isfinite(float(train_ll)) else None,
        "val_loglik": (
            float(val_loglik)
            if val_loglik is not None and math.isfinite(float(val_loglik))
            else None
        ),
    }
    if code_path:
        record["code_path"] = str(code_path)
    if embed_code:
        record["code"] = code if isinstance(code, str) else ("" if code is None else str(code))
    return record


def best_reference_parent(
    parent_records: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Highest selection_score among selected parents; ties keep first occurrence."""
    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for rec in parent_records:
        score = rec.get("selection_score")
        if score is None:
            continue
        try:
            s = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(s):
            continue
        if s > best_score:
            best_score = s
            best = rec
    return best


def compute_delta_f(
    candidate_score: Optional[float],
    reference_score: Optional[float],
) -> Optional[float]:
    """ΔF = S(candidate) - S(reference); null if either score missing/nonfinite."""
    if candidate_score is None or reference_score is None:
        return None
    try:
        c = float(candidate_score)
        r = float(reference_score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(c) or not math.isfinite(r):
        return None
    return c - r


def record_contains_test_metrics(record: Dict[str, Any]) -> bool:
    """True if any forbidden test metric key appears at top level."""
    return any(k in _TEST_METRIC_KEYS for k in record.keys())


def estimate_tokens_char4(text: str) -> int:
    """Project token count with the repo's char/4 convention."""
    return max(0, (len(text) + 3) // 4)


def allowed_annotation_input_tokens(
    *,
    max_model_len: int,
    reserved_output_tokens: int,
    safety_margin_tokens: int,
    max_input_tokens: Optional[int] = None,
) -> int:
    """Return the max chat-input tokens under the canonical packing inequality."""
    if int(max_model_len) <= 0:
        raise ValueError("max_model_len must be positive")
    if int(reserved_output_tokens) < 0 or int(safety_margin_tokens) < 0:
        raise ValueError("reserved_output_tokens / safety_margin_tokens must be >= 0")
    allowed = int(max_model_len) - int(reserved_output_tokens) - int(safety_margin_tokens)
    if allowed <= 0:
        raise ValueError(
            f"no room for input tokens: max_model_len={max_model_len} "
            f"reserved_output={reserved_output_tokens} margin={safety_margin_tokens}"
        )
    if max_input_tokens is not None and int(max_input_tokens) > 0:
        allowed = min(allowed, int(max_input_tokens))
    return allowed


def pack_items_under_chat_budget(
    items: Sequence[Any],
    *,
    estimate_chat_tokens: Any,
    max_items_per_batch: int,
    max_model_len: int,
    reserved_output_tokens: int,
    safety_margin_tokens: int,
    max_input_tokens: Optional[int] = None,
) -> Tuple[List[List[Any]], List[Tuple[Any, int]]]:
    """Greedy pack items so chat_input + reserved + margin ≤ max_model_len.

    ``estimate_chat_tokens(batch)`` must return the full system+user token count
    for that batch (exact tokenizer preferred). Never truncates item payloads.

    Returns ``(batches, oversized_singletons)`` where each oversized entry is
    ``(item, solo_chat_tokens)``. Callers must log/fail those; do not silently drop.
    """
    if max_items_per_batch <= 0:
        raise ValueError("max_items_per_batch must be positive")
    allowed = allowed_annotation_input_tokens(
        max_model_len=max_model_len,
        reserved_output_tokens=reserved_output_tokens,
        safety_margin_tokens=safety_margin_tokens,
        max_input_tokens=max_input_tokens,
    )
    batches: List[List[Any]] = []
    current: List[Any] = []
    oversized: List[Tuple[Any, int]] = []

    for item in items:
        solo = [item]
        solo_tokens = int(estimate_chat_tokens(solo))
        if solo_tokens > allowed:
            oversized.append((item, solo_tokens))
            continue
        trial = current + [item]
        if len(trial) > max_items_per_batch or int(estimate_chat_tokens(trial)) > allowed:
            if current:
                batches.append(current)
            current = [item]
        else:
            current = list(trial)
    if current:
        batches.append(current)
    return batches, oversized


def split_annotation_batches(
    candidates: Sequence[Dict[str, Any]],
    *,
    reference_code: str,
    base_prompt_chars: int = 0,
    max_input_tokens: Optional[int] = None,
    max_candidates_per_batch: int = 10,
    system_prompt: str = "",
    build_user_prompt: Optional[Any] = None,
    token_counter: Optional[Any] = None,
    max_model_len: int = ANNOTATION_VLLM_MAX_MODEL_LEN,
    reserved_output_tokens: int = PARTICIPANT_DEFAULT_MAX_TOKENS,
    safety_margin_tokens: int = ANNOTATION_SAFETY_MARGIN_TOKENS,
) -> List[List[Dict[str, Any]]]:
    """
    Split candidates into batches that fit the token budget without truncating code.

    Preferred budget (when ``token_counter`` + ``build_user_prompt`` are set):

      count(system + user) + reserved_output_tokens + safety_margin_tokens
          <= max_model_len

    Legacy mode (tests / callers without a tokenizer): char/4 on
    ``("x" * base_prompt_chars) + payload_json`` vs ``max_input_tokens``.

    Raises ValueError if a single candidate cannot fit even alone.
    """
    if max_candidates_per_batch <= 0:
        raise ValueError("max_candidates_per_batch must be positive")

    use_exact = token_counter is not None and build_user_prompt is not None
    if use_exact:

        def _est(cands: Sequence[Dict[str, Any]]) -> int:
            user = build_user_prompt(reference_code, cands)  # type: ignore[misc]
            return int(token_counter(str(system_prompt) + str(user)))  # type: ignore[misc]

        batches, oversized = pack_items_under_chat_budget(
            list(candidates),
            estimate_chat_tokens=_est,
            max_items_per_batch=max_candidates_per_batch,
            max_model_len=max_model_len,
            reserved_output_tokens=reserved_output_tokens,
            safety_margin_tokens=safety_margin_tokens,
            max_input_tokens=max_input_tokens,
        )
        if oversized:
            item, solo_tokens = oversized[0]
            allowed = allowed_annotation_input_tokens(
                max_model_len=max_model_len,
                reserved_output_tokens=reserved_output_tokens,
                safety_margin_tokens=safety_margin_tokens,
                max_input_tokens=max_input_tokens,
            )
            raise ValueError(
                f"Candidate {item.get('candidate_id')!r} alone exceeds "
                f"allowed_input_tokens={allowed} (est={solo_tokens}); "
                "refusing to truncate code."
            )
        return batches

    if max_input_tokens is None or int(max_input_tokens) <= 0:
        raise ValueError("max_input_tokens must be positive in legacy char/4 mode")
    allowed_input = int(max_input_tokens)

    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    def _batch_tokens(cands: Sequence[Dict[str, Any]]) -> int:
        payload = {
            "reference_program": reference_code,
            "candidates": [
                {"candidate_id": c["candidate_id"], "code": c["code"]} for c in cands
            ],
        }
        body = json.dumps(payload, ensure_ascii=False)
        return estimate_tokens_char4("x" * int(base_prompt_chars) + body)

    for cand in candidates:
        solo = [cand]
        solo_tokens = _batch_tokens(solo)
        if solo_tokens > allowed_input:
            raise ValueError(
                f"Candidate {cand.get('candidate_id')!r} alone exceeds "
                f"allowed_input_tokens={allowed_input} (est={solo_tokens}); "
                "refusing to truncate code."
            )
        trial = current + [cand]
        if len(trial) > max_candidates_per_batch or _batch_tokens(trial) > allowed_input:
            if current:
                batches.append(current)
            current = [cand]
        else:
            current = list(trial)
    if current:
        batches.append(current)
    return batches


def validate_annotation_response(
    payload: Any,
    *,
    expected_ids: Sequence[str],
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Validate annotator JSON.

    Returns (ok, error_message, rows). On failure rows may be partial/empty.
    """
    expected = list(expected_ids)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        return False, "expected_ids contains duplicates", []

    if isinstance(payload, dict) and "annotations" in payload:
        rows = payload["annotations"]
    else:
        rows = payload
    if not isinstance(rows, list):
        return False, "response must be a JSON list (or object with 'annotations' list)", []

    seen: set[str] = set()
    cleaned: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            return False, f"annotation[{i}] is not an object", []
        cid = row.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            return False, f"annotation[{i}] missing candidate_id", []
        if cid not in expected_set:
            return False, f"unknown candidate_id {cid!r}", []
        if cid in seen:
            return False, f"duplicate candidate_id {cid!r}", []
        seen.add(cid)

        for key in ("added_motifs", "removed_motifs", "modified_motifs"):
            motifs = row.get(key)
            if not isinstance(motifs, list):
                return False, f"{cid}: {key} must be a list", []
            for m in motifs:
                if m not in _MOTIF_SET:
                    return False, f"{cid}: invalid motif {m!r} in {key}", []

        primary = row.get("primary_edit")
        if primary not in _MOTIF_SET:
            return False, f"{cid}: invalid primary_edit {primary!r}", []

        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
            return False, f"{cid}: evidence must be a list of strings", []

        conf = row.get("confidence")
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            return False, f"{cid}: confidence must be a float", []
        if not (0.0 <= conf_f <= 1.0):
            return False, f"{cid}: confidence must be in [0,1]", []

        cleaned.append(
            {
                "candidate_id": cid,
                "added_motifs": list(row["added_motifs"]),
                "removed_motifs": list(row["removed_motifs"]),
                "modified_motifs": list(row["modified_motifs"]),
                "primary_edit": primary,
                "evidence": list(evidence),
                "confidence": conf_f,
            }
        )

    missing = [cid for cid in expected if cid not in seen]
    if missing:
        return False, f"missing candidate_id(s): {missing}", cleaned
    return True, "", cleaned


def build_iteration_context_record(
    *,
    dataset: str,
    participant_id: Any,
    run_id: str,
    split_seed: int,
    phase: str,
    iteration: int,
    evolution_selection_score: str,
    selected_parents: Sequence[Dict[str, Any]],
    best_selected_parent_id: Optional[str],
    mem_trace_format: str = MEM_TRACE_FORMAT,
) -> Dict[str, Any]:
    return {
        "record_type": "iteration_context",
        "mem_trace_format": mem_trace_format,
        "dataset": dataset,
        "participant_id": participant_id,
        "run_id": run_id,
        "split_seed": int(split_seed),
        "phase": phase,
        "iteration": int(iteration),
        "evolution_selection_score": evolution_selection_score,
        "selected_parents": list(selected_parents),
        "best_selected_parent_id": best_selected_parent_id,
    }


def build_candidate_record(
    *,
    dataset: str,
    participant_id: Any,
    run_id: str,
    split_seed: int,
    phase: str,
    iteration: int,
    candidate_id: str,
    candidate_idx: int,
    source: str,
    code: str = "",
    runtime_valid: bool,
    train_loglik: Optional[float],
    val_loglik: Optional[float],
    selection_score: Optional[float],
    reference_parent_id: Optional[str],
    reference_parent_score: Optional[float],
    reference_kind: str,
    delta_f: Optional[float],
    survived_elite_truncation: bool,
    evolution_selection_score: str,
    reference_type: Optional[str] = None,
    reference_id: Optional[str] = None,
    reference_is_exact: Optional[bool] = None,
    prompted_parent_ids: Optional[Sequence[str]] = None,
    embed_code: bool = DEFAULT_EMBED_CODE,
    code_path: Optional[str] = None,
    mem_trace_format: str = MEM_TRACE_FORMAT,
) -> Dict[str, Any]:
    from utils.mem.reference_types import enrich_candidate_reference_fields

    ref_type = str(reference_type or reference_kind)
    ref_id = reference_id if reference_id is not None else reference_parent_id
    rel = code_path or candidate_code_relpath(
        phase=phase,
        iteration=iteration,
        candidate_idx=candidate_idx,
        source=source,
    )
    base: Dict[str, Any] = {
        "record_type": "candidate",
        "mem_trace_format": mem_trace_format,
        "dataset": dataset,
        "participant_id": participant_id,
        "run_id": run_id,
        "split_seed": int(split_seed),
        "phase": phase,
        "iteration": int(iteration),
        "evolution_selection_score": evolution_selection_score,
        "candidate_id": candidate_id,
        "candidate_idx": int(candidate_idx),
        "source": source,
        "code_path": rel,
        "runtime_valid": bool(runtime_valid),
        "train_loglik": train_loglik,
        "val_loglik": val_loglik,
        "selection_score": selection_score,
        "survived_elite_truncation": bool(survived_elite_truncation),
    }
    if embed_code:
        base["code"] = code
    if prompted_parent_ids is not None:
        base["prompted_parent_ids"] = [str(x) for x in prompted_parent_ids]
    return enrich_candidate_reference_fields(
        base,
        reference_type=ref_type,
        reference_id=ref_id,
        reference_score=reference_parent_score,
        delta_f=delta_f,
        reference_is_exact=reference_is_exact,
    )


def iter_jsonl_records(paths: Iterable[Path | str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL record must be object at {p}:{line_no}")
                yield obj
