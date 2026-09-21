#!/usr/bin/env python3
"""Unit tests for participant semantic postprocess (offline, no GPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.mem.participant_semantic_postprocess_v5 import (  # noqa: E402
    RULE_BERGERT_ACTION_MEANS_SOLE,
    RULE_BERGERT_CUE_WEIGHT_VALUE_MOD,
    RULE_NMC_AST_PARAM_OR_STRUCTURE,
    RULE_NMC_NEEDS_ADJUDICATION,
    RULE_SEED_BASELINE_REF_ABSENT,
    RULE_UNUSED_HISTORY_ABSENT,
    compare_ast_semantics,
    history_runtime_used,
    is_seed_baseline_constant_program,
    postprocess_participant_annotation,
)


BASELINE = "def choose(problem, history):\n    return 0.5\n"

VALUE_CUES = """def choose(problem, history):
    cues_A = problem['option_A']['cues']
    cues_B = problem['option_B']['cues']
    score_A = sum(cues_A.values())
    score_B = sum(cues_B.values())
    return 1.0 / (1.0 + (2.718 ** -(score_A - score_B)))
"""

VALUE_AM_BIAS = """def choose(problem, history):
    cues_A = problem['option_A']['cues']
    cues_B = problem['option_B']['cues']
    weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    score_A = sum(c * w for c, w in zip(cues_A.values(), weights))
    score_B = sum(c * w for c, w in zip(cues_B.values(), weights))
    diff = score_A - score_B + problem.get('action_means_option_A_when_1', 0) * 0.5
    return 1.0 / (1.0 + (2.718 ** -diff))
"""

HISTORY_USED = """def choose(problem, history):
    if len(history) > 0:
        last = history[-1]
    cues_A = problem['option_A']['cues']
    return 0.5
"""

HISTORY_ALIAS = """def choose(problem, history):
    past = history
    if len(past) > 0:
        _ = past[-1]
    return 0.5
"""


class TestSeedBaseline(unittest.TestCase):
    def test_detect_constant(self) -> None:
        self.assertTrue(is_seed_baseline_constant_program(BASELINE))
        self.assertFalse(is_seed_baseline_constant_program(VALUE_CUES))

    def test_force_ref_absent(self) -> None:
        ann = {
            "schema_version": 5,
            "dataset": "bergert_nosofsky_2007",
            "reference_motif_state": ["history", "value"],
            "candidate_motif_state": ["value"],
            "modified_motifs": [],
            "structural_operations": [],
            "no_meaningful_change": False,
            "evidence": [],
            "added_motifs": ["value"],
            "removed_motifs": ["history"],
        }
        fixed, corrs, _ = postprocess_participant_annotation(
            ann, reference_code=BASELINE, candidate_code=VALUE_CUES, dataset="bergert_nosofsky_2007"
        )
        self.assertEqual(fixed["reference_motif_state"], [])
        self.assertIn("value", fixed["candidate_motif_state"])
        self.assertNotIn("history", fixed["removed_motifs"])  # rederived: history never on ref
        self.assertTrue(any(c.rule_id == RULE_SEED_BASELINE_REF_ABSENT for c in corrs))


class TestHistoryUse(unittest.TestCase):
    def test_unused_signature(self) -> None:
        self.assertFalse(history_runtime_used(BASELINE))
        self.assertFalse(history_runtime_used(VALUE_CUES))

    def test_used_and_alias(self) -> None:
        self.assertTrue(history_runtime_used(HISTORY_USED))
        self.assertTrue(history_runtime_used(HISTORY_ALIAS))

    def test_strip_unused(self) -> None:
        ann = {
            "schema_version": 5,
            "dataset": "bergert_nosofsky_2007",
            "reference_motif_state": ["history", "value"],
            "candidate_motif_state": ["history", "value"],
            "modified_motifs": [],
            "structural_operations": [],
            "no_meaningful_change": False,
            "evidence": [],
        }
        fixed, corrs, _ = postprocess_participant_annotation(
            ann, reference_code=VALUE_CUES, candidate_code=VALUE_CUES, dataset="bergert_nosofsky_2007"
        )
        self.assertNotIn("history", fixed["reference_motif_state"])
        self.assertNotIn("history", fixed["candidate_motif_state"])
        self.assertTrue(any(c.rule_id == RULE_UNUSED_HISTORY_ABSENT for c in corrs))


class TestActionMeans(unittest.TestCase):
    def test_strip_prob_keep_value(self) -> None:
        ann = {
            "schema_version": 5,
            "dataset": "bergert_nosofsky_2007",
            "reference_motif_state": ["value", "probability_used"],
            "candidate_motif_state": ["value", "probability_used"],
            "modified_motifs": ["value"],
            "structural_operations": ["parameter_change"],
            "no_meaningful_change": False,
            "evidence": [
                "weights = [0.9, 0.8]",
                "diff += problem.get('action_means_option_A_when_1', 0) * 0.5",
            ],
        }
        fixed, corrs, _ = postprocess_participant_annotation(
            ann,
            reference_code=VALUE_CUES,
            candidate_code=VALUE_AM_BIAS,
            dataset="bergert_nosofsky_2007",
        )
        self.assertIn("value", fixed["candidate_motif_state"])
        self.assertNotIn("probability_used", fixed["candidate_motif_state"])
        # ref VALUE_CUES has no AM — probability_used on ref also unsupported → but
        # BERGERT rule requires AM involvement on that side; ref may keep or lose via
        # independent_support=false only when AM involved. Ref has no AM → keep unless
        # we also strip unsupported; current rule requires AM. So ref may still have prob.
        # Candidate must strip.
        self.assertTrue(
            any(
                c.rule_id == RULE_BERGERT_ACTION_MEANS_SOLE and c.construct == "probability_used"
                for c in corrs
            )
        )

    def test_do_not_strip_value_with_cues(self) -> None:
        ann = {
            "schema_version": 5,
            "dataset": "bergert_nosofsky_2007",
            "reference_motif_state": ["value"],
            "candidate_motif_state": ["value"],
            "modified_motifs": ["value"],
            "structural_operations": [],
            "no_meaningful_change": False,
            "evidence": ["action_means_option_A_when_1"],
        }
        fixed, corrs, _ = postprocess_participant_annotation(
            ann,
            reference_code=VALUE_AM_BIAS,
            candidate_code=VALUE_AM_BIAS,
            dataset="bergert_nosofsky_2007",
        )
        self.assertIn("value", fixed["candidate_motif_state"])
        self.assertFalse(
            any(
                c.rule_id == RULE_BERGERT_ACTION_MEANS_SOLE and c.construct == "value"
                for c in corrs
            )
        )


class TestNmcAst(unittest.TestCase):
    def test_rename_equivalent_keeps_nmc(self) -> None:
        a = "def choose(problem, history):\n    x = 0.5\n    return x\n"
        b = "def choose(problem, history):\n    y = 0.5\n    return y\n"
        self.assertTrue(compare_ast_semantics(a, b).equivalent)
        ann = {
            "schema_version": 5,
            "dataset": "bergert_nosofsky_2007",
            "reference_motif_state": [],
            "candidate_motif_state": [],
            "modified_motifs": [],
            "structural_operations": [],
            "no_meaningful_change": True,
            "evidence": [],
        }
        fixed, corrs, meta = postprocess_participant_annotation(
            ann, reference_code=a, candidate_code=b, dataset="bergert_nosofsky_2007"
        )
        self.assertTrue(fixed["no_meaningful_change"])
        self.assertFalse(any(c.rule_id == RULE_NMC_AST_PARAM_OR_STRUCTURE for c in corrs))
        self.assertIsNone(meta["needs_nmc_adjudication"])

    def test_param_change_clears_nmc_marks_value(self) -> None:
        a = VALUE_CUES
        b = VALUE_CUES.replace("sum(cues_A.values())", "0.9 * sum(cues_A.values())")
        ann = {
            "schema_version": 5,
            "dataset": "bergert_nosofsky_2007",
            "reference_motif_state": ["value"],
            "candidate_motif_state": ["value"],
            "modified_motifs": [],
            "structural_operations": [],
            "no_meaningful_change": True,
            "evidence": [],
        }
        fixed, corrs, meta = postprocess_participant_annotation(
            ann, reference_code=a, candidate_code=b, dataset="bergert_nosofsky_2007"
        )
        self.assertFalse(fixed["no_meaningful_change"])
        self.assertIn("value", fixed["modified_motifs"])
        self.assertTrue(
            any(c.rule_id == RULE_BERGERT_CUE_WEIGHT_VALUE_MOD for c in corrs)
        )
        self.assertIsNone(meta["needs_nmc_adjudication"])

    def test_non_bergert_nmc_does_not_auto_value(self) -> None:
        # Avoid exact seed-baseline constant (return 0.5) so REF_ABSENT does not fire.
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
        fixed, corrs, meta = postprocess_participant_annotation(
            ann, reference_code=a, candidate_code=b, dataset="12badham2017deficits"
        )
        self.assertIsNotNone(meta["needs_nmc_adjudication"])
        self.assertFalse(fixed["no_meaningful_change"])
        self.assertNotIn("value", fixed["modified_motifs"])
        self.assertTrue(any(c.rule_id == RULE_NMC_NEEDS_ADJUDICATION for c in corrs))
        self.assertEqual(
            fixed["semantic_resolution_status"], "nmc_needs_adjudication"
        )
        self.assertTrue(fixed["exclude_from_construct_effect_fitting"])
        self.assertEqual(fixed["transition_by_construct"]["value"], "unresolved")
        self.assertFalse(
            any(c.rule_id.startswith("BERGERT_") for c in corrs)
        )

    def test_bergert_rules_gated_off_other_dataset_with_am_string(self) -> None:
        """Even if code mentions action_means, non-Bergert dataset must not fire Bergert rules."""
        code = VALUE_AM_BIAS
        ann = {
            "schema_version": 5,
            "dataset": "12badham2017deficits",
            "reference_motif_state": ["value", "probability_used"],
            "candidate_motif_state": ["value", "probability_used"],
            "modified_motifs": ["value"],
            "structural_operations": [],
            "no_meaningful_change": False,
            "evidence": ["action_means_option_A_when_1"],
        }
        fixed, corrs, _ = postprocess_participant_annotation(
            ann, reference_code=code, candidate_code=code, dataset="12badham2017deficits"
        )
        self.assertIn("probability_used", fixed["candidate_motif_state"])
        self.assertFalse(any(c.rule_id.startswith("BERGERT_") for c in corrs))


if __name__ == "__main__":
    unittest.main()
