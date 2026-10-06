"""Snapshot-bound provenance adapter; frozen annotation semantics stay unchanged."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from utils.mem.pics_v4_uniform_v8_reference import (
    PicsV4ReferenceError, REFERENCE_POLICY, TRANSITION_IDENTITY_POLICY,
    uniform_v8_annotation_key,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_bytes(path: Path, data: bytes) -> None:
    """Commit raw-response cache before exposing it to the annotation worker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def strict_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise PicsV4ReferenceError(
            f"unterminated annotation JSONL tail: {path}; quarantine/repair explicitly before restart"
        )
    rows = []
    for index, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise PicsV4ReferenceError(f"invalid annotation JSONL line {index}: {path}") from exc
        if not isinstance(row, dict):
            raise PicsV4ReferenceError("annotation JSONL row is not an object")
        rows.append(row)
    return rows


def repair_tail(path: Path, scope) -> Path:
    """Explicit quarantine repair, confined to the new annotation namespace.

    Never automatically repairs a journal. Validates every retained row first;
    corruption in the prefix or duplicate completed rows refuses repair.
    """
    path = Path(path).resolve()
    expected = (Path(scope.ledger["annotation_output_base"]) /
                scope.ledger["official_files"]["provenance.json"]["sha256"] /
                scope.dataset / {("explore", "explore"): "exploration",
                                 ("evolution", "normal"): "evolution_normal",
                                 ("evolution", "fresh"): "evolution_fresh"}[(scope.phase, scope.source)])
    if path.parent != expected.resolve() or path.name != "annotations_v5.jsonl":
        raise PicsV4ReferenceError("tail repair outside selected annotation output refused")
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if not lines:
        raise PicsV4ReferenceError("no annotation tail to repair")
    if data.endswith(b"\n"):
        try:
            json.loads(lines[-1])
        except (ValueError, UnicodeError):
            pass
        else:
            raise PicsV4ReferenceError("valid completed tail will not be removed")
    prefix = b"".join(lines[:-1])
    quarantine = path.with_name(path.name + ".quarantine." + digest(data))
    prefix_path = path.with_name(path.name + ".repair-prefix")
    atomic_bytes(prefix_path, prefix)
    try:
        scope.completed(prefix_path)
    finally:
        prefix_path.unlink(missing_ok=True)
    atomic_bytes(quarantine, data)
    atomic_bytes(path, prefix)
    return quarantine


class SelectedScope:
    """Consume a certified ledger, never discover or select a replacement track."""

    def __init__(self, ledger: Path, expected_sha: str, dataset: str, phase: str, source: str):
        self.path = Path(ledger).resolve()
        data = self.path.read_bytes()
        if digest(data) != expected_sha:
            raise PicsV4ReferenceError("selected annotation ledger SHA mismatch")
        self.ledger_sha = expected_sha
        self.ledger = json.loads(data)
        self.dataset, self.phase, self.source = dataset, phase, source
        for name, artifact in self.ledger["official_files"].items():
            if digest(Path(artifact["path"]).read_bytes()) != artifact["sha256"]:
                raise PicsV4ReferenceError("official-gate snapshot drift: " + name)
        for path, expected in self.ledger["method_hashes"].items():
            if digest(Path(path).read_bytes()) != expected:
                raise PicsV4ReferenceError("frozen annotation method drift: " + path)
        match = [d for d in self.ledger["datasets"] if d["dataset"] == dataset]
        if len(match) != 1 or not match[0]["annotation_ready"]:
            raise PicsV4ReferenceError("dataset lacks selected-track annotation certification")
        self.meta = match[0]
        self.root = Path(self.meta["selected_root"]).resolve()
        self.events = {}
        self.keys = {}
        for row in self.meta["transitions"]:
            if row["status"] != "eligible" or row["phase"] != phase or row["source"] != source:
                continue
            event = tuple(row["event_key"])
            key = tuple(row["uniform_v8_resume_key"])
            if event in self.events or key in self.keys:
                raise PicsV4ReferenceError("duplicate canonical ledger identity")
            self.events[event] = row
            self.keys[key] = row
        if not self.events:
            raise PicsV4ReferenceError("selected slice contains no eligible transitions")
        # Certify all trace and artifact bytes again before output creation or resume.
        for artifact in self.meta["resume_inputs"]:
            if digest(Path(artifact["path"]).read_bytes()) != artifact["sha256"]:
                raise PicsV4ReferenceError("selected-track input drift: " + artifact["path"])
        expected_traces = {Path(p["path"]).resolve() for p in self.meta["traces"]}
        actual_traces = {p.resolve() for p in self.root.glob("participant_*/mem_trace.jsonl")}
        if actual_traces != expected_traces:
            raise PicsV4ReferenceError("selected participant trace coverage drift")

    def event(self, rec: dict, participant_dir: Path) -> dict:
        from utils.mem.trace import record_contains_test_metrics
        if record_contains_test_metrics(rec):
            raise PicsV4ReferenceError("passive test fields on selected transition")
        participant_dir = Path(participant_dir).resolve()
        if participant_dir.parent != self.root:
            raise PicsV4ReferenceError("non-selected track/root refused")
        if rec.get("dataset") != self.dataset or participant_dir.name != f"participant_{rec.get('participant_id')}":
            raise PicsV4ReferenceError("selected dataset/participant identity mismatch")
        key = tuple(str(rec.get(k, "")) for k in (
            "dataset", "run_id", "participant_id", "phase", "source", "iteration", "candidate_id"
        ))
        row = self.events.get(key)
        if row is None:
            raise PicsV4ReferenceError("non-selected, excluded, or uncertified event refused")
        for field in ("code_path", "reference_id", "reference_type", "selection_score", "reference_score", "delta_f"):
            if rec.get(field) != row["stored_record"].get(field):
                raise PicsV4ReferenceError("stored transition drift: " + field)
        return row

    def resolve(self, rec: dict, *, participant_dir: Path, **_: Any) -> dict:
        row = self.event(rec, participant_dir)
        cand, ref = row["candidate_artifact"], row["reference_artifact"]
        cb, rb = Path(cand["path"]).read_bytes(), Path(ref["path"]).read_bytes()
        if digest(cb) != cand["sha256"] or digest(rb) != ref["sha256"]:
            raise PicsV4ReferenceError("candidate/reference SHA mismatch")
        if rec.get("code") and digest(str(rec["code"]).encode()) != cand["sha256"]:
            raise PicsV4ReferenceError("hydrated candidate SHA mismatch")
        return dict(reference_policy=REFERENCE_POLICY, reference_id=row["reference_id"],
                    reference_type=row["reference_type"], reference_score=rec["reference_score"],
                    candidate_code=cb.decode(), reference_code=rb.decode(),
                    candidate_code_path=cand["path"], reference_code_path=ref["path"],
                    candidate_sha256=cand["sha256"], reference_sha256=ref["sha256"],
                    job_id=self.meta["canonical_output_job"],
                    resume_key=row["uniform_v8_resume_key"],
                    resolution_mode="official_gate_selected_stored_reference", gate_used=False,
                    fallback_used=False)

    def stamp(self, annotation: dict) -> dict:
        key = uniform_v8_annotation_key(annotation)
        row = self.keys.get(key)
        if row is None:
            raise PicsV4ReferenceError("annotation is outside selected canonical scope")
        out = dict(annotation)
        out.update(official_gate_selected_key=row["official_gate_selected_key"],
                   official_gate_provenance_sha256=self.ledger["official_files"]["provenance.json"]["sha256"],
                   chosen_track=self.meta["chosen_track"], selected_ledger_sha256=self.ledger_sha,
                   selected_execution_lineage=self.meta["lineage"])
        return out

    def completed(self, path: Path) -> set[tuple]:
        from utils.mem.schema_participant_transition_v5 import is_schema_v5_row
        from utils.mem.trace import record_contains_test_metrics
        done = set()
        for row in strict_rows(path):
            if (not is_schema_v5_row(row) or record_contains_test_metrics(row)
                    or row.get("annotation_kind") != "participant_program_motif_transition"
                    or not isinstance(row.get("reference_motif_state"), list)
                    or not isinstance(row.get("candidate_motif_state"), list)
                    or row.get("error") or row.get("success") is False):
                raise PicsV4ReferenceError("failure/non-schema/passive row cannot mark annotation success")
            key = uniform_v8_annotation_key(row)
            expected = self.stamp(row)
            for field in ("official_gate_selected_key", "official_gate_provenance_sha256", "chosen_track", "selected_ledger_sha256", "selected_execution_lineage"):
                if row.get(field) != expected[field]:
                    raise PicsV4ReferenceError("resume selected snapshot/scope mismatch: " + field)
            if key in done:
                raise PicsV4ReferenceError("duplicate completed selected annotation")
            done.add(key)
        return done


class RawCache:
    """Cache the actual LLM wire response; schema/finalizers are still the worker's."""
    def __init__(self, directory: Path, binding: Any):
        self.directory, self.binding = Path(directory), binding

    def response(self, request: dict, generate):
        key = digest(encoded([self.binding, request]))
        path = self.directory / (key + ".json")
        if path.exists():
            row = json.loads(path.read_bytes())
            if row.get("request_sha256") != key or digest(row["raw"].encode()) != row["raw_sha256"]:
                raise PicsV4ReferenceError("raw-response cache corruption")
            return row["raw"]
        raw = generate()
        atomic_bytes(path, encoded(dict(request_sha256=key, raw=raw, raw_sha256=digest(raw.encode()))) + b"\n")
        return raw
