#!/usr/bin/env python3
"""
Ad-hoc report fixer for TE non-strict Choice13k loglik runs.

Safety-first behavior:
- Reads only existing run artifacts.
- Never overwrites any existing run file.
- Writes all outputs into a fresh subfolder inside the run directory.

Fix logic:
- For each participant, prefer `participant_*/results.json`.
- Fallback for legacy OpenEvolve folders:
  `participant_*/openevolve_output/best/best_program_info.json`.
- Build fixed participant and summary CSVs from train/test loglik values.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ParticipantLoglikRow:
    participant_id: int
    train_loglik: Optional[float]
    test_loglik: Optional[float]
    source_program_id: str
    source_results_json: str


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt2(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.2f}"


def _infer_participant_id(path: Path) -> int:
    parent_name = path.parent.name
    if parent_name.startswith("participant_"):
        return int(parent_name.split("_", 1)[1])
    # Legacy best_program_info.json lives deeper:
    # participant_123/openevolve_output/best/best_program_info.json
    for ancestor in path.parents:
        name = ancestor.name
        if name.startswith("participant_"):
            return int(name.split("_", 1)[1])
    raise ValueError(f"Cannot infer participant_id from {path}")


def _load_row_from_results_json(path: Path) -> ParticipantLoglikRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_train = payload.get("overall_best_train_accuracy", {})

    participant_id = payload.get("participant_id", None)
    if participant_id is None:
        participant_id = _infer_participant_id(path)

    return ParticipantLoglikRow(
        participant_id=int(participant_id),
        train_loglik=_safe_float(best_train.get("train_loglik")),
        test_loglik=_safe_float(best_train.get("test_loglik")),
        source_program_id=str(best_train.get("program_id", "")),
        source_results_json=str(path),
    )


def _load_row_from_legacy_best_program_info(path: Path) -> ParticipantLoglikRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    participant_id = _infer_participant_id(path)
    return ParticipantLoglikRow(
        participant_id=int(participant_id),
        train_loglik=_safe_float(payload.get("train_loglik")),
        test_loglik=_safe_float(payload.get("test_loglik")),
        source_program_id=str(payload.get("program_id", "")),
        source_results_json=str(path),
    )


def _write_participant_details_csv(path: Path, rows: List[ParticipantLoglikRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["participant_id", "train_loglik", "test_loglik"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "participant_id": r.participant_id,
                    "train_loglik": _fmt2(r.train_loglik),
                    "test_loglik": _fmt2(r.test_loglik),
                }
            )


def _write_summary_csv(path: Path, rows: List[ParticipantLoglikRow]) -> None:
    tr = [r.train_loglik for r in rows if r.train_loglik is not None]
    te = [r.test_loglik for r in rows if r.test_loglik is not None]
    avg_tr = (sum(tr) / len(tr)) if tr else None
    avg_te = (sum(te) / len(te)) if te else None

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["num_of_participants", "avg_train_loglik", "avg_test_loglik"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_of_participants": len(rows),
                "avg_train_loglik": _fmt2(avg_tr),
                "avg_test_loglik": _fmt2(avg_te),
            }
        )


def _write_audit_csv(path: Path, rows: List[ParticipantLoglikRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "source_program_id",
                "source_results_json",
                "train_loglik",
                "test_loglik",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "participant_id": r.participant_id,
                    "source_program_id": r.source_program_id,
                    "source_results_json": r.source_results_json,
                    "train_loglik": _fmt2(r.train_loglik),
                    "test_loglik": _fmt2(r.test_loglik),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a safe ad-hoc fixed loglik report from participant results.json files."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run folder path, e.g. generated_outputs/choice13k/non_strict/run_260427_112812",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="adhoc_fix_report",
        help="Prefix for the new output subfolder inside run_dir.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir is not a directory: {run_dir}")

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = run_dir / f"{args.output_prefix}_{ts}"
    out_dir.mkdir(parents=False, exist_ok=False)

    # Build fixed rows from participant_*/results.json with a legacy fallback.
    rows: List[ParticipantLoglikRow] = []
    for pdir in sorted(run_dir.glob("participant_*")):
        if not pdir.is_dir():
            continue
        results_path = pdir / "results.json"
        legacy_best_path = pdir / "openevolve_output" / "best" / "best_program_info.json"
        if results_path.exists():
            rows.append(_load_row_from_results_json(results_path))
        elif legacy_best_path.exists():
            rows.append(_load_row_from_legacy_best_program_info(legacy_best_path))

    if not rows:
        raise RuntimeError(
            "No participant metrics files found under "
            f"{run_dir} (expected results.json or legacy best_program_info.json)"
        )

    rows.sort(key=lambda r: r.participant_id)

    _write_participant_details_csv(out_dir / "fixed_participant_details_loglik.csv", rows)
    _write_summary_csv(out_dir / "fixed_summary_loglik.csv", rows)
    _write_audit_csv(out_dir / "fixed_participant_loglik_audit.csv", rows)

    readme = out_dir / "README.txt"
    readme.write_text(
        (
            "Ad-hoc fixed report (read-only source, new outputs only)\n"
            "\n"
            "Method:\n"
            "- For each participant, prefer participant_*/results.json\n"
            "- Legacy fallback: participant_*/openevolve_output/best/best_program_info.json\n"
            "- Emit fixed participant and summary CSVs from those paired values\n"
            "\n"
            f"Run directory: {run_dir}\n"
            f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n"
        ),
        encoding="utf-8",
    )

    print(f"[OK] Wrote ad-hoc fixed report to: {out_dir}")
    print("[OK] Files:")
    for name in [
        "fixed_participant_details_loglik.csv",
        "fixed_summary_loglik.csv",
        "fixed_participant_loglik_audit.csv",
        "README.txt",
    ]:
        p = out_dir / name
        if p.exists():
            print(f"  - {p}")


if __name__ == "__main__":
    main()

