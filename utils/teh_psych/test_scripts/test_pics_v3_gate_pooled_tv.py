"""PICS v3 gate: count-pooled TV (not equal-person) chooses G.3 parent."""
from __future__ import annotations

from pathlib import Path

from utils.teh.explore_handoff import select_explore_handoff_parents
from utils.teh.t_pics_gated_transfer import (
    DEFAULT_EXPLORE_POPULATION_TOP_K,
    GATE_RECORD_SCHEMA,
    GATE_SCORE_FIELD,
    decide_gate,
    gate_record_payload,
    plan_g3_explore_parents,
)


def _equal_person_mean(person_scores):
    return sum(person_scores) / float(len(person_scores))


def _count_pooled(person_scores, person_ns):
    total_n = sum(person_ns)
    return sum(s * n for s, n in zip(person_scores, person_ns)) / float(total_n)


def test_gate_score_field_is_pooled_train_val():
    assert GATE_SCORE_FIELD == "pooled_train_val_loglik"
    assert GATE_RECORD_SCHEMA == "t_pics_gated_transfer_gate_v3"


def test_pooled_and_equal_person_disagree_v3_picks_pooled_rank1_for_g3(tmp_path: Path):
    """Unequal trial counts: equal-person prefers transfer; pooled prefers control.

    Control: many trials at mild score + one bad tiny person.
    Transfer: many trials worse + one tiny person that is excellent.
    """
    # Per-person TV scores and trial counts (same people for both arms).
    n = [100, 1]
    control_person = [-0.40, -2.00]
    transfer_person = [-1.00, -0.05]

    equal_control = _equal_person_mean(control_person)
    equal_transfer = _equal_person_mean(transfer_person)
    pooled_control = _count_pooled(control_person, n)
    pooled_transfer = _count_pooled(transfer_person, n)

    assert equal_transfer > equal_control  # equal-person would pick transfer
    assert pooled_control > pooled_transfer  # pooled prefers control
    assert equal_transfer != pooled_transfer or equal_control != pooled_control

    # Equal-person decision would select transfer.
    equal_decision = decide_gate(
        control_score=equal_control,
        transfer_score=equal_transfer,
    )
    assert equal_decision.selected_arm == "transfer"

    # PICS v3 gate uses pooled scores → control.
    pooled_decision = decide_gate(
        control_score=pooled_control,
        transfer_score=pooled_transfer,
    )
    assert pooled_decision.selected_arm == "control"
    assert pooled_decision.score_field == GATE_SCORE_FIELD

    control_rank1 = tmp_path / "control_best.py"
    transfer_rank1 = tmp_path / "transfer_best.py"
    control_rank1.write_text(
        "def choose(problem, history):\n    return 0.25\n", encoding="utf-8"
    )
    transfer_rank1.write_text(
        "def choose(problem, history):\n    return 0.75\n", encoding="utf-8"
    )
    control_code = control_rank1.read_text(encoding="utf-8")
    transfer_code = transfer_rank1.read_text(encoding="utf-8")

    # Elite pools ordered by pooled TV (G.2 ranking); rank-1 is index 0.
    control_elite = [
        (control_code, pooled_control, None, "control_rank1", None, None),
        ("other_c", pooled_control - 0.1, None, "control_other", None, None),
    ]
    transfer_elite = [
        (transfer_code, pooled_transfer, None, "transfer_rank1", None, None),
        ("other_t", pooled_transfer - 0.1, None, "transfer_other", None, None),
    ]
    retained = (
        transfer_elite if pooled_decision.selected_arm == "transfer" else control_elite
    )
    assert retained is control_elite

    plan = plan_g3_explore_parents(retained, n_explore=50)
    assert plan["n_explore_from_rank1"] == 50
    programs, pinned = select_explore_handoff_parents(
        retained, enabled=True, top_k=DEFAULT_EXPLORE_POPULATION_TOP_K
    )
    assert list(pinned) == ["control_rank1"]
    assert len(programs) == 1
    parent_code, parent_id = programs[0]
    assert parent_id == "control_rank1"
    assert parent_code == control_code
    assert parent_code != transfer_code
    assert plan["explore_parent_ids"] == ["control_rank1"]

    record = gate_record_payload(
        target="toy_target",
        selected_source="toy_source",
        decision=pooled_decision,
        control_rank1=control_rank1,
        transfer_rank1=transfer_rank1,
        retained_pool=tmp_path / "control_pool",
        config_path=tmp_path / "cfg.yaml",
        selector_name="unit_test",
        selected_source_rank1=tmp_path / "source_rank1.py",
    )
    assert record["schema"] == GATE_RECORD_SCHEMA
    assert record["score_field"] == "pooled_train_val_loglik"
    assert record["participant_weighting"] == "pooled_trial"
    assert record["never_used_equal_person_mean_for_gate"] is True
    assert record["control_pooled_train_val_loglik"] == pooled_control
    assert record["transfer_pooled_train_val_loglik"] == pooled_transfer
    assert "control_mean_train_val_loglik" not in record
    assert "transfer_mean_train_val_loglik" not in record
    assert record["selected_arm"] == "control"
    assert record["g3_explore_parent"] == "winning_arm_pooled_tv_rank1"
    assert record["control_rank1_path"] == str(control_rank1)


def test_tie_still_keeps_control():
    d = decide_gate(control_score=-0.5, transfer_score=-0.5)
    assert d.selected_arm == "control"
    d2 = decide_gate(control_score=-0.5, transfer_score=-0.5 + 1e-13)
    assert d2.selected_arm == "control"
    d3 = decide_gate(control_score=-0.5, transfer_score=-0.4)
    assert d3.selected_arm == "transfer"
