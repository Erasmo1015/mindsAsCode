#!/usr/bin/env python3
"""Annotate final population-pool programs for motif *presence* only (schema pilot).

Reads program_manifest.json from the population motif pilot tree.
Writes annotations_population_v3.jsonl with resume keys dataset|run_id|program_id.

Does NOT invent reference→candidate transitions or ΔF. No schema-v2 reuse.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import OpenAI

from utils.mem.schema_population_motif import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    guided_json_schema_for_programs,
    motif_definitions_block,
    resume_key,
    validate_program_motif_response,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO
    / "analysis_2026Sep/mem/population_motif_pilot_emnlp_r1/program_manifest.json"
)
OUT_NAME = "annotations_population_v3.jsonl"
FAILURES_NAME = "annotation_failures.jsonl"
SUMMARY_NAME = "annotation_summary.json"

_SYSTEM = """You annotate behavioral motif *presence* in a single Python decision program.
Report which motifs from the allowed taxonomy are present in the program's logic.
Do NOT invent edits, transitions, added/removed/modified labels, or fitness changes.
Return ONLY a JSON array matching the requested schema."""


def _parse_json_payload(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _build_user_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for item in batch:
        blocks.append(
            f"### program_id={item['program_id']}\n"
            f"```python\n{item['code']}\n```"
        )
    return (
        "Allowed behavioral motifs (use only these names):\n"
        f"{motif_definitions_block()}\n\n"
        "For each program, return an object with:\n"
        "- program_id (exact)\n"
        "- program_motif_state: list of present motifs (subset of allowed names)\n"
        "- evidence: short code-grounded strings\n"
        "- confidence: 0..1\n\n"
        "Programs:\n\n" + "\n\n".join(blocks)
    )


def _load_completed(path: Path) -> Set[str]:
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
            if int(row.get("schema_version", -1) or -1) != SCHEMA_VERSION:
                continue
            if row.get("annotation_kind") != "population_program_motif_state":
                continue
            key = row.get("resume_key")
            if key:
                done.add(str(key))
    return done


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _annotate_batch(
    client: OpenAI,
    *,
    model_name: str,
    batch: List[Dict[str, Any]],
    use_guided_json: bool,
    temperature: float,
    max_tokens: int,
    repair_hint: str = "",
) -> Tuple[List[Dict[str, Any]], str, str]:
    expected_ids = [str(x["program_id"]) for x in batch]
    user = _build_user_prompt(batch)
    if repair_hint:
        user += (
            "\n\nPREVIOUS RESPONSE FAILED VALIDATION. Fix the JSON to satisfy:\n"
            + repair_hint
            + "\nReturn ONLY the corrected JSON array."
        )
    messages = [
        {"role": "system", "content": _SYSTEM},
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
            "guided_json": guided_json_schema_for_programs(expected_ids),
            "guided_decoding_backend": "xgrammar",
        }
    print(
        f"[pop-annotate] LLM n={len(batch)} ids={expected_ids} guided={use_guided_json}",
        flush=True,
    )
    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    try:
        payload = _parse_json_payload(raw)
    except json.JSONDecodeError as exc:
        return [], raw, f"JSON parse error: {exc}"
    ok, err, rows = validate_program_motif_response(
        payload, expected_ids=expected_ids, prompt_version=PROMPT_VERSION
    )
    if not ok:
        return [], raw, err
    return rows, raw, ""


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
) -> Dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = _load_completed(out_jsonl)
    todo = [p for p in programs if p["resume_key"] not in done]
    print(
        f"[pop-annotate] total={len(programs)} done={len(done)} todo={len(todo)}",
        flush=True,
    )
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        loaded = []
        for p in chunk:
            code = (REPO / p["code_path"]).read_text(encoding="utf-8")
            loaded.append({**p, "code": code})
        batches.append(loaded)

    n_ok = 0
    n_fail = 0

    def _one(bi: int, batch: List[Dict[str, Any]]) -> Tuple[int, int]:
        ok_c = fail_c = 0
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
                max_tokens=2048,
                repair_hint=last_err if attempt > 0 else "",
            )
            (raw_dir / f"batch_{bi}_attempt{attempt}.txt").write_text(
                raw, encoding="utf-8"
            )
            if rows:
                break
            last_err = err
        if not rows:
            for p in batch:
                _append_jsonl(
                    failures_path,
                    {
                        "resume_key": p["resume_key"],
                        "dataset": p["dataset"],
                        "run_id": p["run_id"],
                        "program_id": p["program_id"],
                        "error": err,
                    },
                )
                fail_c += 1
            return ok_c, fail_c
        by_id = {r["program_id"]: r for r in rows}
        for p in batch:
            rec = dict(by_id[p["program_id"]])
            rec.update(
                {
                    "dataset": p["dataset"],
                    "run_id": p["run_id"],
                    "family": p.get("family"),
                    "code_path": p["code_path"],
                    "code_sha256": p.get("code_sha256"),
                    "global_fitness": p.get("global_fitness"),
                    "selection_score": p.get("selection_score"),
                    "resume_key": p["resume_key"],
                    "has_motifs": {
                        m: (m in set(rec["program_motif_state"]))
                        for m in (
                            "history",
                            "value",
                            "risk",
                            "feedback",
                            "learning",
                            "other_behavioral",
                        )
                    },
                }
            )
            _append_jsonl(out_jsonl, rec)
            ok_c += 1
        return ok_c, fail_c

    if n_workers <= 1:
        for bi, batch in enumerate(batches):
            a, b = _one(bi, batch)
            n_ok += a
            n_fail += b
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_one, bi, batch): bi for bi, batch in enumerate(batches)}
            for fut in as_completed(futs):
                a, b = fut.result()
                n_ok += a
                n_fail += b

    summary = {
        "n_programs_manifest": len(programs),
        "n_already_done": len(done),
        "n_todo": len(todo),
        "n_ok_this_run": n_ok,
        "n_fail_this_run": n_fail,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "out_jsonl": str(out_jsonl),
    }
    return summary


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
    parser.add_argument("--no_guided_json", action="store_true")
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

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    programs = list(payload["programs"])
    if args.datasets:
        keep = {x.strip() for x in args.datasets.split(",") if x.strip()}
        programs = [p for p in programs if p["dataset"] in keep]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Schema smoke: empty state + one motif
        ok, err, rows = validate_program_motif_response(
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
        bad_ok, bad_err, _ = validate_program_motif_response(
            [
                {
                    "program_id": "demo",
                    "program_motif_state": ["not_a_motif"],
                    "evidence": [],
                    "confidence": 0.1,
                }
            ],
            expected_ids=["demo"],
        )
        assert not bad_ok and "invalid motif" in bad_err
        summary = {
            "dry_run": True,
            "n_datasets": len({p["dataset"] for p in programs}),
            "n_programs": len(programs),
            "per_dataset_counts": {
                ds: sum(1 for p in programs if p["dataset"] == ds)
                for ds in sorted({p["dataset"] for p in programs})
            },
            "manifest": str(manifest_path),
            "schema_ok": True,
        }
        (out_dir / "dry_run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    client = OpenAI(base_url=args.llm_server_url, api_key=args.llm_api_key)
    t0 = time.time()
    summary = annotate_programs(
        client,
        programs=programs,
        model_name=args.model_name,
        out_jsonl=out_dir / OUT_NAME,
        failures_path=out_dir / FAILURES_NAME,
        raw_dir=out_dir / "raw_responses",
        use_guided_json=not args.no_guided_json,
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
    )
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
