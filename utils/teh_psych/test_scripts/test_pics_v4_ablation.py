"""Focused checks for frozen PICS v4 A–E controls."""
from types import SimpleNamespace

from utils.teh.pics_v4 import BEHAVIOR_OBJECTIVE_BLOCK, apply_pics_v4_prompt_body
from utils.teh.pics_v4_ablation import SPECS, bind, iteration_controls, launch_readiness, packed_commands, plan_dataset


def test_compensation_uses_top_parents_and_no_fresh():
    bind(SimpleNamespace(pics_v4_ablation="A", pics_v4_omit_uniform_block=False,
                         pics_v4_compensation_from_iteration=11))
    assert iteration_controls(10, sample_parents=True, fresh_n=99, fresh_n_max=10, n_candidates=10) == (True, 1)
    assert iteration_controls(11, sample_parents=True, fresh_n=4, fresh_n_max=10, n_candidates=10) == (False, 0)


def test_d_disables_fresh_and_keeps_sampling():
    bind(SimpleNamespace(pics_v4_ablation="D", pics_v4_omit_uniform_block=False,
                         pics_v4_compensation_from_iteration=None))
    assert iteration_controls(1, sample_parents=True, fresh_n=10) == (True, 0)
    assert SPECS["D"]["person_iters"] == 10


def test_e_drops_only_uniform_block():
    bind(SimpleNamespace(pics_v4_ablation="E", pics_v4_omit_uniform_block=True,
                         pics_v4_compensation_from_iteration=None))
    text = apply_pics_v4_prompt_body("registered task")
    assert BEHAVIOR_OBJECTIVE_BLOCK in text
    assert "registered task" in text
    assert "PICS_V4_UNIFORM_ADDITIONAL_PROMPT" not in text


def test_unresolved_gate_is_reported():
    row = {
        "dataset": "7hilbig2014generalized",
        "chosen_track": "unresolved",
        "result_status": "population_gate_unresolved",
        "target_status": "complete",
        "target_only_population_status": "complete",
        "transfer_based_only_population_status": "incomplete",
        "target_only_output_root": "/tmp/target",
        "transfer_based_only_output_root": "",
    }
    assert "official_gate_unresolved" in plan_dataset(row, "D")["missing"]
    assert plan_dataset(row, "A")["track"] == "target_only"


def test_launch_pack_is_three_transfer_datasets():
    report = launch_readiness()
    assert report["ready"]
    assert [row["chosen_track"] for row in report["datasets"]] == ["transfer_based_only"] * 3
    commands = packed_commands()
    assert len(commands) == 5
    assert "bergert_nosofsky_2007" not in commands[0]
    assert "PICS_V4_ABLATION=A" in commands[0]
    assert "AAMAS_PERSON_ITERS=30" in commands[1]
    assert "AAMAS_EXPLORE_CANDIDATES=0" in commands[2]
    assert "AAMAS_FRESH_N=0" in commands[3]
