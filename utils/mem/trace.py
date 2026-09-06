"""Pure helpers for PICS MEM tracing and offline annotation validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
_TEST_METRIC_KEYS = frozenset(
    {
        "test_loglik",
        "test_acc",
        "test_accuracy",
        "test_mse",
        "gated_test_loglik",
    }
)


def mem_trace_path(participant_dir: Path | str) -> Path:
    return Path(participant_dir) / "mem_trace.jsonl"


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
) -> Dict[str, Any]:
    """Serialize one selected parent for iteration_context (no test metrics)."""
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
    record = {
        "program_id": program_id,
        "selection_score": selection_score,
        "train_loglik": train_ll if train_ll is not None and math.isfinite(float(train_ll)) else None,
        "val_loglik": (
            float(val_loglik)
            if val_loglik is not None and math.isfinite(float(val_loglik))
            else None
        ),
        "code": code if isinstance(code, str) else ("" if code is None else str(code)),
    }
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


def split_annotation_batches(
    candidates: Sequence[Dict[str, Any]],
    *,
    reference_code: str,
    base_prompt_chars: int,
    max_input_tokens: int = 12000,
    max_candidates_per_batch: int = 10,
) -> List[List[Dict[str, Any]]]:
    """
    Split candidates into batches that fit the token budget without truncating code.

    Raises ValueError if a single candidate cannot fit even alone.
    """
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if max_candidates_per_batch <= 0:
        raise ValueError("max_candidates_per_batch must be positive")

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
        if solo_tokens > max_input_tokens:
            raise ValueError(
                f"Candidate {cand.get('candidate_id')!r} alone exceeds "
                f"max_input_tokens={max_input_tokens} (est={solo_tokens}); "
                "refusing to truncate code."
            )
        trial = current + [cand]
        if len(trial) > max_candidates_per_batch or _batch_tokens(trial) > max_input_tokens:
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
) -> Dict[str, Any]:
    return {
        "record_type": "iteration_context",
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
    code: str,
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
) -> Dict[str, Any]:
    return {
        "record_type": "candidate",
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
        "code": code,
        "runtime_valid": bool(runtime_valid),
        "train_loglik": train_loglik,
        "val_loglik": val_loglik,
        "selection_score": selection_score,
        "reference_parent_id": reference_parent_id,
        "reference_parent_score": reference_parent_score,
        "reference_kind": reference_kind,
        "delta_f": delta_f,
        "survived_elite_truncation": bool(survived_elite_truncation),
    }


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
