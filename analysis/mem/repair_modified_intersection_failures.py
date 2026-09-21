#!/usr/bin/env python3
"""Offline repair: normalize modified∉∩ from annotation_failures.jsonl into annotations_v5.

Does not call an LLM. For the Badham pilot systematic failure only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    PROMPT_VERSION,
    SCHEMA_VERSION,
    global_candidate_id,
    validate_annotation_response_v5,
)

NORMALIZATIONS_NAME = "annotation_normalizations.jsonl"
ANNOTATIONS_NAME = "annotations_v5.jsonl"
FAILURES_NAME = "annotation_failures.jsonl"


def _parse_json_payload(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _load_done_cids(ann_path: Path) -> set:
    done = set()
    if not ann_path.is_file():
        return done
    for line in ann_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("schema_version", -1) or -1) != SCHEMA_VERSION:
            continue
        cid = row.get("candidate_id")
        if isinstance(cid, str):
            done.add(cid)
    return done


def repair_failures(
    *,
    out_dir: Path,
    candidate_ids: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    fail_path = out_dir / FAILURES_NAME
    ann_path = out_dir / ANNOTATIONS_NAME
    norm_path = out_dir / NORMALIZATIONS_NAME
    done = _load_done_cids(ann_path)
    want = set(candidate_ids) if candidate_ids else None

    repaired = []
    skipped = []
    still_bad = []
    if not fail_path.is_file():
        return {"error": f"missing {fail_path}", "repaired": []}

    for line in fail_path.open():
        line = line.strip()
        if not line:
            continue
        fail = json.loads(line)
        cid = fail.get("candidate_id")
        if not isinstance(cid, str):
            continue
        if want is not None and cid not in want:
            continue
        if cid in done:
            skipped.append({"candidate_id": cid, "reason": "already_in_annotations"})
            continue
        raws = fail.get("raw_responses") or []
        if not raws:
            still_bad.append({"candidate_id": cid, "reason": "no_raw"})
            continue
        # Prefer last attempt.
        payload = None
        last_err = ""
        norms: List[Dict[str, Any]] = []
        for raw in reversed(raws):
            try:
                payload = _parse_json_payload(raw)
            except json.JSONDecodeError as exc:
                last_err = f"JSON parse: {exc}"
                continue
            norms = []
            ok, err, rows = validate_annotation_response_v5(
                payload,
                expected_ids=[cid],
                prompt_version=PROMPT_VERSION,
                normalize_modified_outside_intersection=True,
                normalizations_out=norms,
            )
            if ok and rows:
                row = rows[0]
                # Carry trace metadata from sibling annotations if present —
                # failures file alone lacks dataset/run; leave enrichment to caller
                # via a template from annotations_v5 or require fields on failure.
                meta = {
                    k: fail.get(k)
                    for k in (
                        "dataset",
                        "run_id",
                        "participant_id",
                        "phase",
                        "iteration",
                        "reference_id",
                        "reference_parent_id",
                        "reference_type",
                        "delta_f",
                        "selection_score",
                        "reference_score",
                        "source",
                    )
                    if fail.get(k) is not None
                }
                enriched = {**row, **meta, "schema_version": SCHEMA_VERSION}
                enriched["prompt_version"] = PROMPT_VERSION
                enriched["repair_source"] = "offline_normalize_modified_intersection"
                if norms:
                    enriched["modified_motifs_normalization"] = {
                        "dropped": norms[0].get("dropped_modified_motifs"),
                        "kept": norms[0].get("kept_modified_motifs"),
                    }
                repaired.append({"row": enriched, "norms": norms, "failure": fail})
                break
            last_err = err
            payload = None
        else:
            still_bad.append({"candidate_id": cid, "reason": last_err or "validate_failed"})

    if dry_run:
        return {
            "dry_run": True,
            "n_repaired": len(repaired),
            "repaired_ids": [r["row"]["candidate_id"] for r in repaired],
            "skipped": skipped,
            "still_bad": still_bad,
        }

    # Enrich from an existing annotation template (same participant) when meta missing.
    template = None
    if ann_path.is_file():
        for line in ann_path.open():
            try:
                template = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    with ann_path.open("a", encoding="utf-8") as af:
        for item in repaired:
            row = item["row"]
            fail = item["failure"]
            if template:
                for k in (
                    "dataset",
                    "run_id",
                    "participant_id",
                    "phase",
                    "annotation_model",
                    "reference_type",
                    "reference_kind",
                ):
                    if row.get(k) is None and template.get(k) is not None:
                        row[k] = template[k]
            # Pull iteration from candidate_id when possible.
            if row.get("iteration") is None and row["candidate_id"].startswith("iteration_"):
                try:
                    row["iteration"] = int(row["candidate_id"].split("_")[1])
                except (IndexError, ValueError):
                    pass
            if row.get("phase") is None:
                row["phase"] = "evolution"
            if row.get("source") is None:
                row["source"] = "normal"
            # Reference / scores: leave if unknown; build_dataset will need them from trace merge.
            row["global_candidate_id"] = global_candidate_id(
                row.get("dataset"),
                row.get("run_id"),
                row.get("participant_id"),
                row.get("iteration"),
                row["candidate_id"],
                phase=row.get("phase"),
            )
            af.write(json.dumps(row, ensure_ascii=False) + "\n")
            for note in item["norms"]:
                with norm_path.open("a", encoding="utf-8") as nf:
                    nf.write(
                        json.dumps(
                            {
                                **note,
                                "repair": "offline_normalize_modified_intersection",
                                "from_failure_batch_tag": fail.get("batch_tag"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    return {
        "n_repaired": len(repaired),
        "repaired_ids": [r["row"]["candidate_id"] for r in repaired],
        "skipped": skipped,
        "still_bad": still_bad,
        "annotations": str(ann_path),
        "normalizations": str(norm_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--candidate_ids", default="")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    cids = [x.strip() for x in args.candidate_ids.split(",") if x.strip()] or None
    result = repair_failures(
        out_dir=Path(args.output_dir), candidate_ids=cids, dry_run=args.dry_run
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
