#!/usr/bin/env python3
"""Production integration: annotate write path, MEM CSV stamps, fitter exclusion."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_annotate_edits():
    sys.modules.setdefault("openai", ModuleType("openai")).OpenAI = object  # type: ignore[attr-defined]
    path = _REPO / "analysis/mem/annotate_edits.py"
    spec = importlib.util.spec_from_file_location("annotate_edits_prod_wire", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from analysis.mem.build_dataset import (  # noqa: E402
    _fieldnames_for_schema,
    build_rows_v5,
)
from utils.mem.participant_semantic_postprocess_v5 import (  # noqa: E402
    NMC_ADJUDICATION_UNRESOLVED,
    STATUS_NMC_NEEDS_ADJUDICATION,
    STATUS_RESOLVED,
    TRANSITION_UNRESOLVED,
    filter_frame_for_construct_effect_fitting,
    finalize_v5_annotation_for_write,
    postprocess_participant_annotation,
)
from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    state_and_eligibility_flags,
)


class TestUnresolvedNmcStatus(unittest.TestCase):
    def test_generic_nmc_clear_stamps_unresolved(self) -> None:
        a = "def choose(problem, history):\n    x = 0.5\n    return x\n"
        b = "def choose(problem, history):\n    x = 0.7\n    return x\n"
        ann = {
            "schema_version": 5,
            "dataset": "12badham2017deficits",
            "reference_motif_state": ["value"],
            "candidate_motif_state": ["value"],
            "modified_motifs": [],
            "structural_operations": [],
            "no_meaningful_change": True,
            "evidence": [],
        }
        fixed, _, meta = postprocess_participant_annotation(
            ann, reference_code=a, candidate_code=b, dataset="12badham2017deficits"
        )
        self.assertIsNotNone(meta["needs_nmc_adjudication"])
        self.assertEqual(
            fixed["semantic_resolution_status"], STATUS_NMC_NEEDS_ADJUDICATION
        )
        self.assertEqual(fixed["nmc_adjudication_status"], NMC_ADJUDICATION_UNRESOLVED)
        self.assertTrue(fixed["exclude_from_construct_effect_fitting"])
        self.assertFalse(fixed["no_meaningful_change"])
        self.assertNotIn("value", fixed["modified_motifs"])
        self.assertEqual(
            fixed["transition_by_construct"]["value"], TRANSITION_UNRESOLVED
        )

    def test_finalize_preserves_raw_llm_fields(self) -> None:
        a = "def choose(problem, history):\n    x = 0.5\n    return x\n"
        b = "def choose(problem, history):\n    x = 0.7\n    return x\n"
        ann = {
            "schema_version": 5,
            "dataset": "12badham2017deficits",
            "reference_motif_state": ["value"],
            "candidate_motif_state": ["value"],
            "modified_motifs": [],
            "structural_operations": [],
            "no_meaningful_change": True,
            "evidence": ["raw claim"],
            "confidence": 0.9,
        }
        fixed, _, _ = finalize_v5_annotation_for_write(
            ann, reference_code=a, candidate_code=b, dataset="12badham2017deficits"
        )
        raw = fixed["raw_llm_annotation"]
        self.assertTrue(raw["no_meaningful_change"])
        self.assertEqual(raw["evidence"], ["raw claim"])
        self.assertFalse(fixed["no_meaningful_change"])

    def test_eligibility_flags_emit_unresolved_transition(self) -> None:
        ann = {
            "schema_version": 5,
            "annotation_kind": "participant_program_motif_transition",
            "prompt_version": "x",
            "candidate_id": "c1",
            "reference_motif_state": ["value"],
            "candidate_motif_state": ["value"],
            "added_motifs": [],
            "removed_motifs": [],
            "modified_motifs": [],
            "structural_operations": ["parameter_change"],
            "no_meaningful_change": False,
            "evidence": [],
            "confidence": 0.5,
            "transition_by_construct": {"value": "unresolved"},
            "semantic_resolution_status": STATUS_NMC_NEEDS_ADJUDICATION,
            "nmc_adjudication_status": NMC_ADJUDICATION_UNRESOLVED,
            "exclude_from_construct_effect_fitting": True,
        }
        flags = state_and_eligibility_flags(ann)
        self.assertEqual(flags["transition_value"], "unresolved")
        self.assertEqual(flags["value_modified"], 0)
        self.assertEqual(flags["retained_unmodified_value"], 0)


class TestAnnotateWritePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_annotate_edits()

    def test_write_runs_postprocess_and_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            out = tmp_p / "annotations_v5.jsonl"
            corr = tmp_p / "semantic_corrections.jsonl"
            adj = tmp_p / "nmc_adjudication_queue.jsonl"
            ref = "def choose(problem, history):\n    x = 0.5\n    return x\n"
            cand = "def choose(problem, history):\n    x = 0.7\n    return x\n"
            batch = [
                {
                    "candidate_id": "cand_1",
                    "code": cand,
                    "delta_f": 0.1,
                    "source": "normal",
                }
            ]
            rows = [
                {
                    "candidate_id": "cand_1",
                    "reference_motif_state": ["value"],
                    "candidate_motif_state": ["value"],
                    "modified_motifs": [],
                    "structural_operations": [],
                    "no_meaningful_change": True,
                    "evidence": [],
                    "confidence": 0.8,
                    "added_motifs": [],
                    "removed_motifs": [],
                }
            ]
            completed: set = set()
            n = self.mod._write_annotation_rows(
                out,
                rows=rows,
                batch=batch,
                key=("run", "12badham2017deficits", 1, "evolution", 0),
                ref_id="ref0",
                ref_type="seed_baseline",
                completed=completed,
                reference_code=ref,
                schema_version=5,
                prompt_version="participant_transition_v5_calibrated_2026Sep22",
                model_name="test",
                corrections_path=corr,
                adjudication_path=adj,
            )
            self.assertEqual(n, 1)
            written = json.loads(out.read_text(encoding="utf-8").strip())
            self.assertIn("raw_llm_annotation", written)
            self.assertTrue(written["raw_llm_annotation"]["no_meaningful_change"])
            self.assertEqual(
                written["semantic_resolution_status"], STATUS_NMC_NEEDS_ADJUDICATION
            )
            self.assertTrue(written["exclude_from_construct_effect_fitting"])
            self.assertTrue(corr.is_file())
            self.assertTrue(adj.is_file())
            adj_row = json.loads(adj.read_text(encoding="utf-8").strip())
            self.assertEqual(
                adj_row["nmc_adjudication_status"], NMC_ADJUDICATION_UNRESOLVED
            )


class TestBuildAndFitExclusion(unittest.TestCase):
    def test_build_fieldnames_include_status(self) -> None:
        cols = _fieldnames_for_schema(5)
        for c in (
            "semantic_resolution_status",
            "nmc_adjudication_status",
            "exclude_from_construct_effect_fitting",
            "raw_llm_annotation",
        ):
            self.assertIn(c, cols)

    def test_filter_frame_excludes_unresolved(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "candidate_id": "a",
                    "exclude_from_construct_effect_fitting": 0,
                    "semantic_resolution_status": STATUS_RESOLVED,
                },
                {
                    "candidate_id": "b",
                    "exclude_from_construct_effect_fitting": 1,
                    "semantic_resolution_status": STATUS_NMC_NEEDS_ADJUDICATION,
                },
            ]
        )
        out, n_excl = filter_frame_for_construct_effect_fitting(df)
        self.assertEqual(n_excl, 1)
        self.assertEqual(list(out["candidate_id"]), ["a"])

    def test_build_rows_v5_propagates_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            # Minimal mem_trace with one eligible candidate.
            trace = tmp_p / "mem_trace.jsonl"
            rec = {
                "record_type": "candidate",
                "dataset": "12badham2017deficits",
                "run_id": "r1",
                "participant_id": 1,
                "iteration": 0,
                "candidate_id": "c1",
                "phase": "evolution",
                "source": "normal",
                "runtime_valid": True,
                "delta_f": 0.2,
                "selection_score": 1.2,
                "reference_score": 1.0,
                "reference_parent_score": 1.0,
                "reference_id": "ref0",
                "reference_type": "parent",
                "reference_parent_id": "ref0",
            }
            trace.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            # Put a fake nested layout: build_dataset scans for mem_trace under run_dir
            run_dir = tmp_p / "run"
            run_dir.mkdir()
            (run_dir / "mem_trace.jsonl").write_text(
                json.dumps(rec) + "\n", encoding="utf-8"
            )

            from utils.mem.schema_participant_transition_v5 import (
                annotation_resume_key as ann_key_v5,
            )

            key = ann_key_v5(
                rec["dataset"],
                rec["run_id"],
                rec["participant_id"],
                rec["iteration"],
                rec["candidate_id"],
                reference_id=rec["reference_id"],
                reference_type=rec["reference_type"],
                phase=rec["phase"],
            )
            ann = {
                "schema_version": 5,
                "annotation_kind": "participant_program_motif_transition",
                "prompt_version": "x",
                "candidate_id": "c1",
                "participant_id": 1,
                "dataset": "12badham2017deficits",
                "run_id": "r1",
                "iteration": 0,
                "phase": "evolution",
                "reference_id": "ref0",
                "reference_type": "parent",
                "reference_motif_state": ["value"],
                "candidate_motif_state": ["value"],
                "added_motifs": [],
                "removed_motifs": [],
                "modified_motifs": [],
                "structural_operations": ["parameter_change"],
                "no_meaningful_change": False,
                "evidence": [],
                "confidence": 0.5,
                "transition_by_construct": {"value": "unresolved"},
                "semantic_postprocess_version": "participant_semantic_v5_2026Sep22",
                "semantic_resolution_status": STATUS_NMC_NEEDS_ADJUDICATION,
                "nmc_adjudication_status": NMC_ADJUDICATION_UNRESOLVED,
                "exclude_from_construct_effect_fitting": True,
                "raw_llm_annotation": {"no_meaningful_change": True},
            }
            rows, excl = build_rows_v5(
                run_dir=run_dir,
                annotations={key: ann},
                phase="evolution",
                source="normal",
            )
            self.assertEqual(len(rows), 1, msg=dict(excl))
            row = rows[0]
            self.assertEqual(
                row["semantic_resolution_status"], STATUS_NMC_NEEDS_ADJUDICATION
            )
            self.assertEqual(row["exclude_from_construct_effect_fitting"], 1)
            self.assertEqual(row["transition_value"], "unresolved")
            self.assertIsNotNone(row["raw_llm_annotation"])


if __name__ == "__main__":
    unittest.main()
