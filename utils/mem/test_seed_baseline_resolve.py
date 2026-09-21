#!/usr/bin/env python3
"""Tests for strict seed-baseline artifact resolution (no openai)."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_annotate_edits():
    sys.modules.setdefault("openai", ModuleType("openai")).OpenAI = object  # type: ignore[attr-defined]
    path = _REPO / "analysis/mem/annotate_edits.py"
    spec = importlib.util.spec_from_file_location("annotate_edits_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(
    Path(
        "/careAIDrive/zichang/misc/mindAsCode/generated_outputs/psych101_train/"
        "teh/bergert_nosofsky_2007/pics_v3/job_271247/selected/participant_1/mem_trace.jsonl"
    ).is_file(),
    "bergert gated run not available on this host",
)
class TestSeedBaselineResolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_annotate_edits()
        cls.run_dir = Path(
            "/careAIDrive/zichang/misc/mindAsCode/generated_outputs/psych101_train/"
            "teh/bergert_nosofsky_2007/pics_v3/job_271247"
        )
        cls.pdir = cls.run_dir / "selected/participant_1"
        cls.rec = None
        with (cls.pdir / "mem_trace.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                if o.get("record_type") == "candidate" and o.get("reference_id") == "baseline":
                    cls.rec = o
                    break
        assert cls.rec is not None
        cls.rec = dict(cls.rec)
        cls.rec["_participant_dir"] = str(cls.pdir)

    def test_artifact_resolve(self) -> None:
        parent, rid, mode = self.mod._resolve_seed_baseline_artifact(
            self.rec, participant_dir=self.pdir
        )
        self.assertEqual(mode, "official_seed_baseline_artifact")
        self.assertEqual(rid, "baseline")
        assert parent is not None
        self.assertIn("return 0.5", parent["code"])
        self.assertEqual(parent["selection_score"], self.rec.get("reference_score"))

    def test_strict_resolve_uses_artifact_not_selected_parents(self) -> None:
        ctx = {
            "selected_parents": [{"program_id": "explore_candidate_17"}],
            "_participant_dir": str(self.pdir),
        }
        parent, rid, mode = self.mod._resolve_reference_for_candidate(
            self.rec, ctx, strict_reference=True, run_dir=self.run_dir
        )
        self.assertEqual(mode, "official_seed_baseline_artifact")
        self.assertEqual(rid, "baseline")
        self.assertIsNotNone(parent)

    def test_non_baseline_still_unresolved(self) -> None:
        rec = dict(self.rec)
        rec["reference_id"] = "missing_parent_xyz"
        ctx = {"selected_parents": [], "_participant_dir": str(self.pdir)}
        parent, rid, mode = self.mod._resolve_reference_for_candidate(
            rec, ctx, strict_reference=True, run_dir=self.run_dir
        )
        self.assertIsNone(parent)
        self.assertEqual(mode, "unresolved_official_id_not_in_selected_parents")

if __name__ == "__main__":
    unittest.main()
