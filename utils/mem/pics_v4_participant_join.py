"""Fail-closed, one-to-one join for stored uniform-v8 transition identities."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.mem.pics_v4_uniform_v8_reference import (
    REFERENCE_POLICY, TRANSITION_IDENTITY_POLICY, PicsV4ReferenceError,
    resolve_uniform_v8_reference, uniform_v8_annotation_key,
)
from utils.mem.schema_participant_transition_v5 import is_schema_v5_row
from utils.mem.trace import record_contains_test_metrics


class V8JoinError(ValueError):
    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        counts = {name: len(audit[name]) for name in (
            "unmatched_trace_rows", "unmatched_annotations", "duplicate_annotation_keys",
            "duplicate_trace_keys", "sha_mismatches", "invalid_annotations", "invalid_traces",
        )}
        super().__init__(f"uniform-v8 one-to-one join refused: {counts}")


class V8AnnotationIndex(dict):
    """Keep duplicate/invalid evidence instead of silently overwriting labels."""
    def __init__(self, records: Iterable[Mapping[str, Any]]):
        super().__init__()
        self.duplicates: list[Any] = []
        self.invalid: list[Any] = []
        self.sha_mismatches: list[Any] = []
        self.n_rows = 0
        for record in records:
            self.n_rows += 1
            try:
                if not is_schema_v5_row(record) or record_contains_test_metrics(dict(record)):
                    raise PicsV4ReferenceError("invalid schema or passive test fields")
                key = uniform_v8_annotation_key(record)
            except (PicsV4ReferenceError, TypeError, ValueError) as exc:
                detail = {"row": self.n_rows, "reason": str(exc)}
                self.invalid.append(detail)
                if "SHA" in str(exc):
                    self.sha_mismatches.append(detail)
                continue
            if key in self:
                self.duplicates.append(list(key))
                continue
            self[key] = dict(record)


def _event_key(key: tuple[str, ...]) -> tuple[str, ...]:
    # Diagnostic grouping only. Never used to match labels.
    return key[:8] + key[9:11]


def validate_v8_join(
    traces: Iterable[dict[str, Any]], annotations: V8AnnotationIndex,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Traces are the annotation-eligible slice, with their participant path.

    Resolve the actual stored reference and hash both artifacts with the same
    resolver used by annotation. Compare the entire key with the stored key.
    Any join discrepancy refuses the whole result, before CSV writing.
    """
    audit: dict[str, Any] = {
        "reference_policy": REFERENCE_POLICY,
        "transition_identity_policy": TRANSITION_IDENTITY_POLICY,
        "n_annotation_rows": annotations.n_rows,
        "n_trace_rows": 0, "n_matched": 0,
        "unmatched_trace_rows": [], "unmatched_annotations": [],
        "duplicate_annotation_keys": annotations.duplicates,
        "duplicate_trace_keys": [], "sha_mismatches": list(annotations.sha_mismatches),
        "invalid_annotations": annotations.invalid, "invalid_traces": [],
    }
    prepared = []
    used: set[tuple[str, ...]] = set()
    counts: Counter = Counter()
    annotation_events: dict[Any, list[Any]] = {}
    for key in annotations:
        annotation_events.setdefault(_event_key(key), []).append(key)
    for record in traces:
        audit["n_trace_rows"] += 1
        try:
            resolved = resolve_uniform_v8_reference(
                record, participant_dir=Path(record["_participant_dir"])
            )
        except (PicsV4ReferenceError, KeyError, OSError, ValueError) as exc:
            detail = {"candidate_id": record.get("candidate_id"), "reason": str(exc)}
            audit["invalid_traces"].append(detail)
            audit["unmatched_trace_rows"].append(detail)
            if "SHA" in str(exc):
                audit["sha_mismatches"].append(detail)
            continue
        key = tuple(resolved["resume_key"])
        counts[key] += 1
        if counts[key] > 1:
            audit["duplicate_trace_keys"].append(list(key))
        if key not in annotations:
            audit["unmatched_trace_rows"].append(list(key))
            alternatives = annotation_events.get(_event_key(key), [])
            if alternatives:
                audit["sha_mismatches"].append({
                    "trace_key": list(key), "annotation_keys": [list(k) for k in alternatives],
                })
            continue
        used.add(key)
        audit["n_matched"] += 1
        prepared.append({**record, "_uniform_v8_key": key, "_v8_reference": resolved})
    audit["unmatched_annotations"] = [list(k) for k in annotations if k not in used]
    audit["ok"] = not any(audit[name] for name in (
        "unmatched_trace_rows", "unmatched_annotations", "duplicate_annotation_keys",
        "duplicate_trace_keys", "sha_mismatches", "invalid_annotations", "invalid_traces",
    ))
    if not audit["ok"]:
        raise V8JoinError(audit)
    return prepared, audit
