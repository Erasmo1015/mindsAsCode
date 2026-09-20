#!/usr/bin/env python3
"""Annotate population programs for construct presence + parent→candidate transitions.

Schema v3: single-program presence only.
Schema v4: six-construct reference state + modified + per-construct candidate details.
Schema v5: five-construct final taxonomy (history, value, probability_used, feedback,
learning); same transition/detail contract as v4 without explicit risk.

Candidate presence inventories are derived from complete motif_details.

Resume keys: ``dataset|run_id|iteration|candidate|parent``.
Existing successful annotations are skipped (append-only resume).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from openai import OpenAI
    try:
        from openai import APIStatusError, InternalServerError
    except ImportError:  # older openai SDKs
        APIStatusError = Exception  # type: ignore[misc, assignment]
        InternalServerError = Exception  # type: ignore[misc, assignment]
except Exception:  # noqa: BLE001 — login-node site-packages can be broken
    OpenAI = None  # type: ignore[misc, assignment]
    APIStatusError = Exception  # type: ignore[misc, assignment]
    InternalServerError = Exception  # type: ignore[misc, assignment]

from utils.mem import schema_population_motif as schema_v3
from utils.mem import schema_population_motif_v4 as schema_v4
from utils.mem import schema_population_motif_v5 as schema_v5

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO
    / "analysis_2026Sep/mem/population_motif_pilot_emnlp_r1/program_manifest.json"
)
FAILURES_NAME = "annotation_failures.jsonl"
SUMMARY_NAME = "annotation_summary.json"

# Serializes appends to shared jsonl outputs when --n_workers > 1.
_IO_LOCK = threading.Lock()

# Job script treats this exit code as "restart vLLM and resume".
EXIT_VLLM_DEAD = 75

DEFAULT_MAX_TOKENS_V4 = 8192
DEFAULT_MAX_TOKENS_V5 = 8192
DEFAULT_MAX_TOKENS_V3 = 2048
DEFAULT_SERVER_RETRIES = 5
DEFAULT_SERVER_RETRY_SLEEP_SEC = 15.0


class VLLMServerDeadError(RuntimeError):
    """Raised after bounded InternalServerError retries are exhausted."""


def _schema_mod(schema_version: int):
    if int(schema_version) == 5:
        return schema_v5
    if int(schema_version) == 4:
        return schema_v4
    if int(schema_version) == 3:
        return schema_v3
    raise ValueError(
        f"unsupported schema_version={schema_version} (expected 3, 4, or 5)"
    )


def _is_transition_schema(schema_version: int) -> bool:
    return int(schema_version) in (4, 5)


def _out_name(schema_version: int) -> str:
    return f"annotations_population_v{int(schema_version)}.jsonl"


_SYSTEM_V3 = """You annotate behavioral motif *presence* in a single Python decision program.
Report which motifs from the allowed taxonomy are present in the program's logic.
Do NOT invent edits, transitions, added/removed/modified labels, or fitness changes.
Return ONLY a JSON array matching the requested schema."""

_SYSTEM_V4 = """You annotate behavioral constructs for a parent→candidate program pair.
Report construct presence in the reference (parent), which shared constructs were
modified, and per-construct details for the CANDIDATE (presence, applicability,
confidence, rationale, code_evidence) covering EVERY allowed construct.
Candidate presence inventories are taken from motif_details.presence — do not emit
a separate candidate_motif_state list.
Do NOT invent fitness/ΔF values.
Distinguish carefully:
- probability_used vs explicit_risk_mechanism (linear p*x is probability_used only)
- feedback vs learning (learning requires an across-trial update rule)
Return ONLY a JSON array matching the requested schema."""

_SYSTEM_V5 = """You annotate behavioral constructs for a parent→candidate program pair.
Report construct presence in the reference (parent), which shared constructs were
modified, and per-construct details for the CANDIDATE (presence, applicability,
confidence, rationale, code_evidence) covering EVERY allowed construct.
Candidate presence inventories are taken from motif_details.presence — do not emit
a separate candidate_motif_state list.
Do NOT invent fitness/ΔF values.
Distinguish carefully:
- probability_used means reading/using problem probability fields (including linear p*x);
  empirical rates from feedback history alone are feedback/learning, not probability_used
- feedback vs learning (learning requires an across-trial update rule)
Return ONLY a JSON array matching the requested schema."""


def _parse_json_payload(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _is_vllm_server_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("InternalServerError", "APIStatusError", "APIConnectionError"):
        code = getattr(exc, "status_code", None)
        if code is None and name == "InternalServerError":
            return True
        try:
            return int(code or 0) >= 500
        except (TypeError, ValueError):
            return name == "InternalServerError"
    if isinstance(exc, InternalServerError) and InternalServerError is not Exception:
        return True
    if isinstance(exc, APIStatusError) and APIStatusError is not Exception:
        try:
            return int(getattr(exc, "status_code", 0) or 0) >= 500
        except (TypeError, ValueError):
            return False
    return False


def _build_user_prompt_v3(batch: Sequence[Dict[str, Any]], *, schema_mod: Any) -> str:
    blocks = []
    for item in batch:
        blocks.append(
            f"### program_id={item['program_id']}\n"
            f"```python\n{item['code']}\n```"
        )
    return (
        "Allowed behavioral motifs (use only these names):\n"
        f"{schema_mod.motif_definitions_block()}\n\n"
        "Label mechanisms implemented in the program code, not dataset subject matter.\n"
        "For each program, return an object with:\n"
        "- program_id (exact)\n"
        "- program_motif_state: list of present motifs (subset of allowed names)\n"
        "- evidence: short code-grounded strings\n"
        "- confidence: 0..1\n\n"
        "Programs:\n\n" + "\n\n".join(blocks)
    )


def _build_user_prompt_v4(batch: Sequence[Dict[str, Any]], *, schema_mod: Any) -> str:
    blocks = []
    for item in batch:
        blocks.append(
            f"### candidate_id={item['candidate_id']}\n"
            f"parent_id={item.get('parent_id')}\n"
            f"source={item.get('source')} reference_kind={item.get('reference_kind')}\n"
            f"#### reference (parent) code\n```python\n{item['parent_code']}\n```\n"
            f"#### candidate code\n```python\n{item['code']}\n```"
        )
    return (
        "Allowed behavioral motifs (use only these names):\n"
        f"{schema_mod.motif_definitions_block()}\n\n"
        f"{schema_mod.applicability_block()}\n\n"
        "Label mechanisms implemented in the program code, not dataset subject matter.\n"
        "For each pair, return an object with:\n"
        "- candidate_id (exact)\n"
        "- reference_motif_state: motifs present in the parent\n"
        "- modified_motifs: subset of motifs present in BOTH parent and candidate "
        "whose implementation changed (parameter / functional form / update rule)\n"
        "- motif_details: one object per allowed motif for the CANDIDATE with "
        "presence, applicability, confidence, rationale, code_evidence "
        "(candidate presence is read from these presence flags)\n"
        "- confidence: overall 0..1\n\n"
        "Pairs:\n\n" + "\n\n".join(blocks)
    )


def _load_completed(path: Path, *, schema_version: int) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("schema_version", -1) or -1) != int(schema_version):
                continue
            kind = row.get("annotation_kind")
            if _is_transition_schema(schema_version):
                if kind not in (
                    "population_program_motif_transition",
                    "population_program_motif_state",
                ):
                    continue
            elif kind != "population_program_motif_state":
                continue
            key = row.get("resume_key")
            if key:
                done.add(str(key))
    return done


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _IO_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def _annotate_batch(
    client: OpenAI,
    *,
    model_name: str,
    batch: List[Dict[str, Any]],
    use_guided_json: bool,
    temperature: float,
    max_tokens: int,
    schema_mod: Any,
    repair_hint: str = "",
    server_retries: int = DEFAULT_SERVER_RETRIES,
    server_retry_sleep_sec: float = DEFAULT_SERVER_RETRY_SLEEP_SEC,
) -> Tuple[List[Dict[str, Any]], str, str]:
    if _is_transition_schema(schema_mod.SCHEMA_VERSION):
        expected_ids = [str(x["candidate_id"]) for x in batch]
        user = _build_user_prompt_v4(batch, schema_mod=schema_mod)
        system = _SYSTEM_V5 if schema_mod.SCHEMA_VERSION == 5 else _SYSTEM_V4
    else:
        expected_ids = [str(x["program_id"]) for x in batch]
        user = _build_user_prompt_v3(batch, schema_mod=schema_mod)
        system = _SYSTEM_V3
    if repair_hint:
        user += (
            "\n\nPREVIOUS RESPONSE FAILED VALIDATION. Fix the JSON to satisfy:\n"
            + repair_hint
            + "\nReturn ONLY the corrected JSON array."
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs: Dict[str, Any] = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if use_guided_json:
        kwargs["extra_body"] = {
            "guided_json": schema_mod.guided_json_schema_for_programs(expected_ids),
            "guided_decoding_backend": "xgrammar",
        }
    print(
        f"[pop-annotate] LLM n={len(batch)} ids={expected_ids} guided={use_guided_json} "
        f"schema={schema_mod.SCHEMA_VERSION} max_tokens={max_tokens}",
        flush=True,
    )
    raw = ""
    last_server_exc: Optional[BaseException] = None
    for srv_attempt in range(max(1, int(server_retries))):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            last_server_exc = None
            break
        except Exception as exc:  # noqa: BLE001 — classify below
            if _is_vllm_server_error(exc):
                last_server_exc = exc
                wait = float(server_retry_sleep_sec) * (1 + srv_attempt)
                print(
                    f"[pop-annotate] vLLM server error ({type(exc).__name__}: {exc}); "
                    f"retry {srv_attempt + 1}/{server_retries} sleep={wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
                continue
            return [], raw, f"LLM request error: {type(exc).__name__}: {exc}"
    if last_server_exc is not None:
        raise VLLMServerDeadError(
            f"vLLM server errors exhausted after {server_retries} tries: {last_server_exc}"
        ) from last_server_exc
    try:
        payload = _parse_json_payload(raw)
    except json.JSONDecodeError as exc:
        return [], raw, f"JSON parse error: {exc}"
    ok, err, rows = schema_mod.validate_program_motif_response(
        payload,
        expected_ids=expected_ids,
        prompt_version=schema_mod.PROMPT_VERSION,
    )
    if not ok:
        return [], raw, err
    return rows, raw, ""


def _write_success_rows(
    *,
    batch: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    schema_mod: Any,
    out_jsonl: Path,
) -> int:
    id_key = (
        "candidate_id" if _is_transition_schema(schema_mod.SCHEMA_VERSION) else "program_id"
    )
    by_id = {r[id_key]: r for r in rows}
    n_ok = 0
    for p in batch:
        lookup = p.get("candidate_id") or p["program_id"]
        rec = dict(by_id[lookup])
        present = set(rec.get("candidate_motif_state") or rec.get("program_motif_state") or [])
        rec.update(
            {
                "dataset": p["dataset"],
                "run_id": p["run_id"],
                "family": p.get("family"),
                "code_path": p["code_path"],
                "parent_code_path": p.get("parent_code_path"),
                "code_sha256": p.get("code_sha256"),
                "parent_code_sha256": p.get("parent_code_sha256"),
                "global_fitness": p.get("global_fitness"),
                "selection_score": p.get("selection_score"),
                "iteration": p.get("iteration"),
                "candidate": p.get("candidate"),
                "candidate_idx": p.get("candidate_idx"),
                "parent_id": p.get("parent_id"),
                "parent": p.get("parent") or p.get("parent_id"),
                "source": p.get("source"),
                "reference_kind": p.get("reference_kind"),
                "reference_type": p.get("reference_type") or p.get("reference_kind"),
                "reference_is_exact": p.get("reference_is_exact"),
                "reference_is_proxy": p.get("reference_is_proxy"),
                "resume_key": p["resume_key"],
                "lineage_reconstructed": p.get("lineage_reconstructed"),
                "has_motifs": {m: (m in present) for m in schema_mod.BEHAVIORAL_MOTIFS},
            }
        )
        _append_jsonl(out_jsonl, rec)
        n_ok += 1
    return n_ok


def _annotate_one_unit(
    client: OpenAI,
    *,
    bi: int,
    batch: List[Dict[str, Any]],
    model_name: str,
    use_guided_json: bool,
    max_attempts: int,
    max_tokens: int,
    schema_mod: Any,
    raw_dir: Path,
    out_jsonl: Path,
    failures_path: Path,
    server_retries: int,
    server_retry_sleep_sec: float,
    allow_singleton_split: bool,
) -> Tuple[int, int]:
    """Annotate one batch; on failure with n>1, split to singletons."""
    last_err = ""
    rows: List[Dict[str, Any]] = []
    raw = ""
    err = "not attempted"
    for attempt in range(max_attempts):
        rows, raw, err = _annotate_batch(
            client,
            model_name=model_name,
            batch=batch,
            use_guided_json=use_guided_json,
            temperature=0.0,
            max_tokens=max_tokens,
            schema_mod=schema_mod,
            repair_hint=last_err if attempt > 0 else "",
            server_retries=server_retries,
            server_retry_sleep_sec=server_retry_sleep_sec,
        )
        tag = "batch" if len(batch) > 1 else "single"
        (raw_dir / f"{tag}_{bi}_attempt{attempt}.txt").write_text(raw, encoding="utf-8")
        if rows:
            break
        last_err = err

    if rows:
        return _write_success_rows(
            batch=batch, rows=rows, schema_mod=schema_mod, out_jsonl=out_jsonl
        ), 0

    if allow_singleton_split and len(batch) > 1:
        print(
            f"[pop-annotate] batch_{bi} failed ({err}); retrying {len(batch)} singletons",
            flush=True,
        )
        ok_c = fail_c = 0
        for j, prog in enumerate(batch):
            a, b = _annotate_one_unit(
                client,
                bi=bi * 1000 + j,
                batch=[prog],
                model_name=model_name,
                use_guided_json=use_guided_json,
                max_attempts=max_attempts,
                max_tokens=max_tokens,
                schema_mod=schema_mod,
                raw_dir=raw_dir,
                out_jsonl=out_jsonl,
                failures_path=failures_path,
                server_retries=server_retries,
                server_retry_sleep_sec=server_retry_sleep_sec,
                allow_singleton_split=False,
            )
            ok_c += a
            fail_c += b
        return ok_c, fail_c

    for p in batch:
        _append_jsonl(
            failures_path,
            {
                "resume_key": p["resume_key"],
                "dataset": p["dataset"],
                "run_id": p["run_id"],
                "program_id": p["program_id"],
                "parent_id": p.get("parent_id"),
                "error": err,
            },
        )
    return 0, len(batch)


def annotate_programs(
    client: OpenAI,
    *,
    programs: List[Dict[str, Any]],
    model_name: str,
    out_jsonl: Path,
    failures_path: Path,
    raw_dir: Path,
    use_guided_json: bool,
    max_attempts: int,
    batch_size: int,
    n_workers: int,
    schema_version: int = 4,
    max_tokens: Optional[int] = None,
    server_retries: int = DEFAULT_SERVER_RETRIES,
    server_retry_sleep_sec: float = DEFAULT_SERVER_RETRY_SLEEP_SEC,
) -> Dict[str, Any]:
    schema_mod = _schema_mod(schema_version)
    if max_tokens is None:
        if schema_mod.SCHEMA_VERSION == 5:
            max_tokens = DEFAULT_MAX_TOKENS_V5
        elif schema_mod.SCHEMA_VERSION == 4:
            max_tokens = DEFAULT_MAX_TOKENS_V4
        else:
            max_tokens = DEFAULT_MAX_TOKENS_V3
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = _load_completed(out_jsonl, schema_version=schema_mod.SCHEMA_VERSION)
    todo = [p for p in programs if p["resume_key"] not in done]
    print(
        f"[pop-annotate] schema={schema_mod.SCHEMA_VERSION}/{schema_mod.PROMPT_VERSION} "
        f"total={len(programs)} done={len(done)} todo={len(todo)} "
        f"max_tokens={max_tokens}",
        flush=True,
    )
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        loaded = []
        for p in chunk:
            code = (REPO / p["code_path"]).read_text(encoding="utf-8")
            item = {**p, "code": code}
            if _is_transition_schema(schema_mod.SCHEMA_VERSION):
                parent_rel = p.get("parent_code_path")
                if not parent_rel:
                    raise FileNotFoundError(
                        f"missing parent_code_path for {p.get('resume_key')}"
                    )
                item["parent_code"] = (REPO / parent_rel).read_text(encoding="utf-8")
                item["candidate_id"] = p.get("candidate_id") or p["program_id"]
            loaded.append(item)
        batches.append(loaded)

    n_ok = 0
    n_fail = 0

    def _one(bi: int, batch: List[Dict[str, Any]]) -> Tuple[int, int]:
        return _annotate_one_unit(
            client,
            bi=bi,
            batch=batch,
            model_name=model_name,
            use_guided_json=use_guided_json,
            max_attempts=max_attempts,
            max_tokens=int(max_tokens),
            schema_mod=schema_mod,
            raw_dir=raw_dir,
            out_jsonl=out_jsonl,
            failures_path=failures_path,
            server_retries=server_retries,
            server_retry_sleep_sec=server_retry_sleep_sec,
            allow_singleton_split=True,
        )

    try:
        if n_workers <= 1:
            for bi, batch in enumerate(batches):
                a, b = _one(bi, batch)
                n_ok += a
                n_fail += b
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = {
                    ex.submit(_one, bi, batch): bi for bi, batch in enumerate(batches)
                }
                for fut in as_completed(futs):
                    a, b = fut.result()
                    n_ok += a
                    n_fail += b
    except VLLMServerDeadError:
        # Persist partial progress summary before propagating.
        summary = {
            "n_programs_manifest": len(programs),
            "n_already_done": len(done),
            "n_todo": len(todo),
            "n_ok_this_run": n_ok,
            "n_fail_this_run": n_fail,
            "schema_version": schema_mod.SCHEMA_VERSION,
            "prompt_version": schema_mod.PROMPT_VERSION,
            "annotation_kind": getattr(schema_mod, "ANNOTATION_KIND", None),
            "out_jsonl": str(out_jsonl),
            "vllm_dead": True,
        }
        (out_jsonl.parent / SUMMARY_NAME).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        raise

    summary = {
        "n_programs_manifest": len(programs),
        "n_already_done": len(done),
        "n_todo": len(todo),
        "n_ok_this_run": n_ok,
        "n_fail_this_run": n_fail,
        "schema_version": schema_mod.SCHEMA_VERSION,
        "prompt_version": schema_mod.PROMPT_VERSION,
        "annotation_kind": getattr(schema_mod, "ANNOTATION_KIND", None),
        "out_jsonl": str(out_jsonl),
        "max_tokens": int(max_tokens),
        "server_retries": int(server_retries),
    }
    return summary


def _demo_v4_payload(candidate_id: str = "demo") -> List[Dict[str, Any]]:
    details = []
    present = {"history", "value", "probability_used"}
    for m in schema_v4.BEHAVIORAL_MOTIFS:
        details.append(
            {
                "motif": m,
                "presence": m in present,
                "applicability": "applicable",
                "confidence": 0.85,
                "rationale": f"demo rationale for {m}",
                "code_evidence": [f"demo evidence for {m}"] if m in present else [],
            }
        )
    return [
        {
            "candidate_id": candidate_id,
            "reference_motif_state": ["history", "value"],
            "modified_motifs": ["value"],
            "motif_details": details,
            "confidence": 0.85,
        }
    ]


def _demo_v5_payload(candidate_id: str = "demo") -> List[Dict[str, Any]]:
    details = []
    present = {"history", "value", "probability_used"}
    for m in schema_v5.BEHAVIORAL_MOTIFS:
        details.append(
            {
                "motif": m,
                "presence": m in present,
                "applicability": "applicable",
                "confidence": 0.85,
                "rationale": f"demo rationale for {m}",
                "code_evidence": [f"demo evidence for {m}"] if m in present else [],
            }
        )
    return [
        {
            "candidate_id": candidate_id,
            "reference_motif_state": ["history", "value"],
            "modified_motifs": ["value"],
            "motif_details": details,
            "confidence": 0.85,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--llm_server_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--llm_api_key", type=str, default="EMPTY")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--server_retries", type=int, default=DEFAULT_SERVER_RETRIES)
    parser.add_argument(
        "--server_retry_sleep_sec",
        type=float,
        default=DEFAULT_SERVER_RETRY_SLEEP_SEC,
    )
    parser.add_argument("--no_guided_json", action="store_true")
    parser.add_argument(
        "--schema_version",
        type=int,
        default=4,
        choices=(3, 4, 5),
        help="Population construct schema (3, 4, or 5; default 4).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load manifest, print counts, validate schema helpers; no LLM.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset filter.",
    )
    args = parser.parse_args()

    schema_mod = _schema_mod(args.schema_version)
    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    programs = list(payload["programs"])
    if args.datasets:
        keep = {x.strip() for x in args.datasets.split(",") if x.strip()}
        programs = [p for p in programs if p["dataset"] in keep]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / _out_name(args.schema_version)

    if args.dry_run:
        if _is_transition_schema(args.schema_version):
            demo_fn = (
                _demo_v5_payload if args.schema_version == 5 else _demo_v4_payload
            )
            ok, err, rows = schema_mod.validate_program_motif_response(
                demo_fn("demo"),
                expected_ids=["demo"],
            )
            assert ok and rows, err
            assert rows[0]["candidate_motif_state"] == [
                "history",
                "value",
                "probability_used",
            ]
            assert rows[0]["added_motifs"] == ["probability_used"]
            mismatched = demo_fn("demo2")
            mismatched[0]["candidate_motif_state"] = ["learning"]  # ignored
            ok2, err2, rows2 = schema_mod.validate_program_motif_response(
                mismatched, expected_ids=["demo2"]
            )
            assert ok2 and rows2, err2
            assert "learning" not in rows2[0]["candidate_motif_state"]
            bad = demo_fn("demo3")
            bad[0]["motif_details"] = bad[0]["motif_details"][:3]
            bad_ok, bad_err, _ = schema_mod.validate_program_motif_response(
                bad, expected_ids=["demo3"]
            )
            assert not bad_ok and "missing motifs" in bad_err
            if args.schema_version == 5:
                # Reject explicit_risk if the model invents it.
                bad_risk = demo_fn("demo4")
                bad_risk[0]["motif_details"].append(
                    {
                        "motif": "explicit_risk_mechanism",
                        "presence": False,
                        "applicability": "applicable",
                        "confidence": 0.5,
                        "rationale": "should be rejected",
                        "code_evidence": [],
                    }
                )
                risk_ok, risk_err, _ = schema_mod.validate_program_motif_response(
                    bad_risk, expected_ids=["demo4"]
                )
                assert not risk_ok
                assert "explicit_risk" in risk_err or "invalid motif" in risk_err
            rk = schema_mod.resume_key("ds", "job_1", 2, "candidate_0", "global_baseline")
            assert rk == "ds|job_1|2|candidate_0|global_baseline"
            done = _load_completed(out_jsonl, schema_version=args.schema_version)
            assert isinstance(done, set)
        else:
            ok, err, rows = schema_mod.validate_program_motif_response(
                [
                    {
                        "program_id": "demo",
                        "program_motif_state": ["history", "value"],
                        "evidence": ["uses past outcomes"],
                        "confidence": 0.8,
                    }
                ],
                expected_ids=["demo"],
            )
            assert ok and rows, err
        summary = {
            "dry_run": True,
            "schema_version": schema_mod.SCHEMA_VERSION,
            "prompt_version": schema_mod.PROMPT_VERSION,
            "motifs": list(schema_mod.BEHAVIORAL_MOTIFS),
            "n_datasets": len({p["dataset"] for p in programs}),
            "n_programs": len(programs),
            "n_already_done": len(_load_completed(out_jsonl, schema_version=args.schema_version)),
            "per_dataset_counts": {
                ds: sum(1 for p in programs if p["dataset"] == ds)
                for ds in sorted({p["dataset"] for p in programs})
            },
            "manifest": str(manifest_path),
            "out_jsonl": str(out_jsonl),
            "schema_ok": True,
            "default_max_tokens_v4": DEFAULT_MAX_TOKENS_V4,
            "resume_key_sample": (
                programs[0]["resume_key"] if programs else None
            ),
        }
        (out_dir / "dry_run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    if OpenAI is None:
        print("[error] openai package unavailable in this environment", flush=True)
        sys.exit(2)
    client = OpenAI(base_url=args.llm_server_url, api_key=args.llm_api_key)
    t0 = time.time()
    try:
        summary = annotate_programs(
            client,
            programs=programs,
            model_name=args.model_name,
            out_jsonl=out_jsonl,
            failures_path=out_dir / FAILURES_NAME,
            raw_dir=out_dir / "raw_responses",
            use_guided_json=not args.no_guided_json,
            max_attempts=args.max_attempts,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            schema_version=args.schema_version,
            max_tokens=args.max_tokens,
            server_retries=args.server_retries,
            server_retry_sleep_sec=args.server_retry_sleep_sec,
        )
    except VLLMServerDeadError as exc:
        print(f"[pop-annotate] FATAL vLLM dead: {exc}", flush=True)
        sys.exit(EXIT_VLLM_DEAD)
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
