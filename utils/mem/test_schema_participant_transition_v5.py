#!/usr/bin/env python3
"""Unit tests for participant_transition_v5 schema + resume keys."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    BEHAVIORAL_MOTIFS,
    PRIMARY_FOCAL_CANDIDATES,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    annotation_resume_key,
    assert_transition_identities,
    derive_directional_motifs,
    global_candidate_id,
    state_and_eligibility_flags,
    validate_annotation_response_v5,
)


class TestConstructs(unittest.TestCase):
    def test_exactly_five_constructs(self) -> None:
        self.assertEqual(
            list(BEHAVIORAL_MOTIFS),
            ["history", "value", "probability_used", "feedback", "learning"],
        )
        self.assertNotIn("risk", BEHAVIORAL_MOTIFS)
        self.assertNotIn("other_behavioral", BEHAVIORAL_MOTIFS)
        self.assertNotIn("explicit_risk", BEHAVIORAL_MOTIFS)

    def test_primary_focal_candidates_preserved(self) -> None:
        for name in (
            "history_modified",
            "value_modified",
            "feedback_added",
            "feedback_modified",
        ):
            self.assertIn(name, PRIMARY_FOCAL_CANDIDATES)


class TestDirectionsAndEligibility(unittest.TestCase):
    def test_derive_added_removed_modified(self) -> None:
        added, removed, modified = derive_directional_motifs(
            ["history", "value"],
            ["value", "feedback"],
            ["value"],
        )
        self.assertEqual(added, ["feedback"])
        self.assertEqual(removed, ["history"])
        self.assertEqual(modified, ["value"])

    def test_validate_and_eligibility_flags(self) -> None:
        payload = [
            {
                "candidate_id": "c0",
                "reference_motif_state": ["history", "value"],
                "candidate_motif_state": ["value", "feedback"],
                "modified_motifs": ["value"],
                "structural_operations": [],
                "no_meaningful_change": False,
                "evidence": ["value window changed", "feedback term added"],
                "confidence": 0.8,
            }
        ]
        ok, err, rows = validate_annotation_response_v5(
            payload, expected_ids=["c0"], prompt_version=PROMPT_VERSION
        )
        self.assertTrue(ok, err)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["schema_version"], SCHEMA_VERSION)
        self.assertEqual(row["added_motifs"], ["feedback"])
        self.assertEqual(row["removed_motifs"], ["history"])
        self.assertEqual(row["modified_motifs"], ["value"])
        assert_transition_identities(row)
        flags = state_and_eligibility_flags(row)
        self.assertEqual(flags["history_removed"], 1)
        self.assertEqual(flags["feedback_added"], 1)
        self.assertEqual(flags["value_modified"], 1)
        self.assertEqual(flags["eligible_history_added"], 0)
        self.assertEqual(flags["eligible_history_removed"], 1)
        self.assertEqual(flags["eligible_feedback_added"], 1)
        self.assertEqual(flags["eligible_value_modified"], 1)
        self.assertEqual(flags["eligible_feedback_modified"], 0)
        # Retained-construct set ≠ broader modification opportunity.
        self.assertEqual(flags["modification_opportunity_value"], 1)
        self.assertEqual(flags["modification_opportunity_feedback"], 0)
        self.assertEqual(flags["retained_unmodified_value"], 0)
        self.assertEqual(flags["transition_history"], "removed")
        self.assertEqual(flags["transition_value"], "modified")
        self.assertEqual(flags["transition_feedback"], "added")
        self.assertEqual(flags["transition_learning"], "absent")

    def test_retained_construct_risk_set_vs_opportunity(self) -> None:
        from utils.mem.schema_participant_transition_v5 import risk_set_definition

        d = risk_set_definition("history_modified")
        self.assertEqual(d["risk_set_name"], "retained_construct_modification_risk_set")
        self.assertIn("AND", d["rule"])
        self.assertIn("retained_unmodified", d["focal_contrast"])
        self.assertIn("modification_opportunity", d["broader_pre_transition_opportunity"])

    def test_delta_f_consistency(self) -> None:
        from utils.mem.schema_participant_transition_v5 import verify_delta_f_consistency

        ok, _ = verify_delta_f_consistency(
            delta_f=-1.5, candidate_score=-2.0, reference_score=-0.5
        )
        self.assertTrue(ok)
        ok, err = verify_delta_f_consistency(
            delta_f=-1.0, candidate_score=-2.0, reference_score=-0.5
        )
        self.assertFalse(ok)
        self.assertIn("inconsistent", err)

    def test_no_meaningful_change_allowed(self) -> None:
        payload = [
            {
                "candidate_id": "c0",
                "reference_motif_state": ["history", "value"],
                "candidate_motif_state": ["history", "value"],
                "modified_motifs": [],
                "structural_operations": [],
                "no_meaningful_change": True,
                "evidence": ["cosmetic rename only"],
                "confidence": 0.9,
            }
        ]
        ok, err, rows = validate_annotation_response_v5(payload, expected_ids=["c0"])
        self.assertTrue(ok, err)
        self.assertTrue(rows[0]["no_meaningful_change"])

    def test_reject_risk_and_normalize_modified(self) -> None:
        bad_risk = [
            {
                "candidate_id": "c0",
                "reference_motif_state": ["risk"],
                "candidate_motif_state": ["risk"],
                "modified_motifs": [],
                "structural_operations": ["parameter_change"],
                "no_meaningful_change": False,
                "evidence": ["x"],
                "confidence": 0.5,
            }
        ]
        ok, err, _ = validate_annotation_response_v5(bad_risk, expected_ids=["c0"])
        self.assertFalse(ok)
        self.assertIn("banned", err.lower())

        # Added-as-modified: deterministically drop non-intersection entries.
        bad_mod = [
            {
                "candidate_id": "c0",
                "reference_motif_state": ["history", "value"],
                "candidate_motif_state": ["history", "value", "feedback", "learning"],
                "modified_motifs": ["feedback", "learning"],
                "structural_operations": [],
                "no_meaningful_change": False,
                "evidence": ["x"],
                "confidence": 0.5,
            }
        ]
        norms: list = []
        ok, err, rows = validate_annotation_response_v5(
            bad_mod,
            expected_ids=["c0"],
            normalizations_out=norms,
        )
        self.assertTrue(ok, err)
        self.assertEqual(rows[0]["added_motifs"], ["feedback", "learning"])
        self.assertEqual(rows[0]["modified_motifs"], [])
        self.assertEqual(norms[0]["dropped_modified_motifs"], ["feedback", "learning"])

        ok, err, _ = validate_annotation_response_v5(
            bad_mod,
            expected_ids=["c0"],
            normalize_modified_outside_intersection=False,
        )
        self.assertFalse(ok)
        self.assertIn("intersection", err)

    def test_empty_array_rejected_for_nonempty_batch(self) -> None:
        ok, err, _ = validate_annotation_response_v5([], expected_ids=["c0", "c1"])
        self.assertFalse(ok)
        self.assertIn("empty annotation array", err)

    def test_guided_schema_minmax_items(self) -> None:
        from utils.mem.schema_participant_transition_v5 import (
            guided_json_schema_for_batch_v5,
        )

        schema = guided_json_schema_for_batch_v5(["a", "b"])
        self.assertEqual(schema["minItems"], 2)
        self.assertEqual(schema["maxItems"], 2)

    def test_resume_key_includes_phase(self) -> None:
        k_evo = annotation_resume_key(
            "ds", "job", 0, 2, "cand",
            reference_id="p", reference_type="best_prompted_parent", phase="evolution",
        )
        k_exp = annotation_resume_key(
            "ds", "job", 0, 2, "cand",
            reference_id="p", reference_type="population_program", phase="explore",
        )
        self.assertNotEqual(k_evo, k_exp)
        self.assertEqual(k_evo[3], "evolution")
        g = global_candidate_id("ds", "job", 0, 2, "cand", phase="explore")
        self.assertIn("|explore|", g)


class TestUniqueIdsAndResume(unittest.TestCase):
    def test_global_candidate_id_and_resume_key(self) -> None:
        g1 = global_candidate_id("dsA", "job_1", 0, 2, "iteration_2_candidate_9")
        g2 = global_candidate_id("dsB", "job_1", 0, 2, "iteration_2_candidate_9")
        self.assertNotEqual(g1, g2)

        k1 = annotation_resume_key(
            "dsA", "job_1", 0, 2, "iteration_2_candidate_9",
            reference_id="parent_a", reference_type="best_prompted_parent",
        )
        k2 = annotation_resume_key(
            "dsB", "job_1", 0, 2, "iteration_2_candidate_9",
            reference_id="parent_a", reference_type="best_prompted_parent",
        )
        self.assertNotEqual(k1, k2)

        # Same local cid + participant across iterations must not collide.
        k_iter2 = annotation_resume_key(
            "dsA", "job_1", 0, 2, "cand_x",
            reference_id="p", reference_type="best_prompted_parent",
        )
        k_iter3 = annotation_resume_key(
            "dsA", "job_1", 0, 3, "cand_x",
            reference_id="p", reference_type="best_prompted_parent",
        )
        self.assertNotEqual(k_iter2, k_iter3)

        # Different reference pairing must not resume.
        k_ref_b = annotation_resume_key(
            "dsA", "job_1", 0, 2, "iteration_2_candidate_9",
            reference_id="parent_b", reference_type="best_prompted_parent",
        )
        self.assertNotEqual(k1, k_ref_b)

    def test_resume_skips_completed_global_keys(self) -> None:
        """Simulate annotate resume: completed keys skip re-annotation."""
        done = {
            annotation_resume_key(
                "dsA", "job_1", 0, 2, "iteration_2_candidate_9",
                reference_id="parent_a", reference_type="best_prompted_parent",
            )
        }
        pending = annotation_resume_key(
            "dsA", "job_1", 0, 2, "iteration_2_candidate_9",
            reference_id="parent_a", reference_type="best_prompted_parent",
        )
        other = annotation_resume_key(
            "dsA", "job_1", 0, 2, "iteration_2_candidate_10",
            reference_id="parent_a", reference_type="best_prompted_parent",
        )
        self.assertIn(pending, done)
        self.assertNotIn(other, done)


if __name__ == "__main__":
    unittest.main()
