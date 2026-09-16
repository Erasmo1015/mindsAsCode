"""T-PICS official source map, SA40 suffix wiring, and CLI guards."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from utils.teh.limited_data_registry import LIMITED_DATA_REGISTRY
from utils.teh.t_pics_sources import (
    DEFAULT_T_PICS_SOURCE_CONFIG,
    T_PICS_OFFICIAL_SOURCES,
    UNFILTERED_T_PICS_SOURCE_CONFIG,
    _FALLBACK_T_PICS_SOURCES,
    load_t_pics_official_sources,
    official_t_pics_source,
    resolve_t_pics_source_participant_ids,
)

_PRE_FIX_SOURCES = {
    "2plonsky2018when",
    "5speekenbrink2008learning",
    "10frey2017risk",
    "12badham2017deficits",
}
from utils.teh.teh_datasets import PARTICIPANT_DATASETS

REPO = Path(__file__).resolve().parents[1]


def test_official_map_covers_all_15_and_never_self():
    mapping = load_t_pics_official_sources()
    assert mapping == T_PICS_OFFICIAL_SOURCES
    assert mapping == _FALLBACK_T_PICS_SOURCES
    assert DEFAULT_T_PICS_SOURCE_CONFIG.name == "t_pics_score_weighted_temp_fix.yaml"
    assert DEFAULT_T_PICS_SOURCE_CONFIG.is_file()
    assert set(mapping) == set(LIMITED_DATA_REGISTRY)
    for target, source in mapping.items():
        assert target != source
        assert source in LIMITED_DATA_REGISTRY
        assert source in PARTICIPANT_DATASETS
        assert source not in _PRE_FIX_SOURCES
        assert official_t_pics_source(target) == source
    assert set(mapping.values()) == {
        "11enkavi2019recentprobes",
        "1peterson2021using",
        "7hilbig2014generalized",
    }


def test_unfiltered_yaml_still_has_pre_fix_tops():
    unfiltered = load_t_pics_official_sources(UNFILTERED_T_PICS_SOURCE_CONFIG)
    assert unfiltered["3frey2017cct"] == "10frey2017risk"
    assert unfiltered["5speekenbrink2008learning"] == "12badham2017deficits"
    assert unfiltered["7hilbig2014generalized"] == "2plonsky2018when"
    assert unfiltered["12badham2017deficits"] == "5speekenbrink2008learning"


def test_official_choice13k_source_is_enkavi():
    assert official_t_pics_source("1peterson2021using") == "11enkavi2019recentprobes"


def test_temp_fix_skips_pre_fix_sources():
    assert official_t_pics_source("3frey2017cct") == "11enkavi2019recentprobes"
    assert official_t_pics_source("5speekenbrink2008learning") == "11enkavi2019recentprobes"
    assert official_t_pics_source("7hilbig2014generalized") == "11enkavi2019recentprobes"
    assert official_t_pics_source("12badham2017deficits") == "11enkavi2019recentprobes"


def test_source_range_clamps_with_patched_valid_ids():
    with patch(
        "utils.teh.participant_ids.load_valid_participant_ids",
        return_value=list(range(23)),
    ):
        pids, note = resolve_t_pics_source_participant_ids(
            source_dataset="5speekenbrink2008learning",
            repo_root=REPO,
            participant_scope="range",
            single_participant_id=0,
            range_start_ordinal=0,
            range_end_ordinal=49,
            all_max_participants=None,
            participant_ordinals=None,
            filter_mixed_gambles=False,
            split_ratio=0.6,
            split_seed=0,
            psych_dataset_split="train",
            local_dataset=None,
            mixed_gambles_csv="",
        )
    assert pids == list(range(23))
    assert "clamped" in note


def test_suffix_forwards_sa40_to_trial_loader(tmp_path: Path):
    from utils.teh.explore_source_prompt import build_rank1_explore_prompt_suffix

    prog = tmp_path / "best_program.py"
    prog.write_text("def choose(problem, history):\n    return 0.5\n", encoding="utf-8")
    captured = {}

    def _fake_trials(dataset, pid, **kwargs):
        captured.update(kwargs)
        captured["dataset"] = dataset
        captured["pid"] = pid
        dummy = {
            "problem": {"schema_type": "binary", "option_keys": ["j", "f"]},
            "history": [],
            "action": 0,
            "options": ["j", "f"],
        }
        return ([dummy], [dummy], [dummy])

    with patch("baseline_methods.MLE.trials_for_participant", side_effect=_fake_trials):
        suffix = build_rank1_explore_prompt_suffix(
            source_dataset="11enkavi2019recentprobes",
            program_path=str(prog),
            split_seed=0,
            limited_data_protocol="structure_aware",
            limited_train_val=40,
            split_ratio=0.6,
        )
    assert captured["limited_data_protocol"] == "structure_aware"
    assert captured["limited_train_val"] == 40
    assert captured["pid"] == 0
    assert "Cross-task transfer context" in suffix
    assert "def choose(" in suffix


def _run_teh(
    *extra: str, dataset: str = "1peterson2021using"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "teh.py"),
            "--dataset",
            dataset,
            "--fitness_metric",
            "loglik",
            "--no_log",
            *extra,
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_help_exposes_t_pics_source():
    proc = subprocess.run(
        [sys.executable, str(REPO / "teh.py"), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "--t_pics_source" in proc.stdout
    assert "--t_pics" in proc.stdout
    assert "--t_pics_source_config" in proc.stdout
    assert "t_pics_score_weighted_temp_fix.yaml" in proc.stdout


def test_cli_rejects_t_pics_source_without_global_phase():
    proc = _run_teh(
        "--t_pics_source",
        "11enkavi2019recentprobes",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "requires --global_phase" in combined


def test_cli_rejects_t_pics_source_with_existing_program_path():
    proc = _run_teh(
        "--global_phase",
        "--t_pics_source",
        "11enkavi2019recentprobes",
        "--global_prompt_source_program",
        "missing.py",
        "--global_prompt_source_dataset",
        "11enkavi2019recentprobes",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "do not also pass --global_prompt_source_program" in combined


def test_cli_rejects_t_pics_self_source():
    proc = _run_teh(
        "--global_phase",
        "--t_pics_source",
        "1peterson2021using",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "must differ from --dataset" in combined


def test_cli_rejects_t_pics_without_no_refinement():
    proc = _run_teh(
        "--global_phase",
        "--t_pics_source",
        "11enkavi2019recentprobes",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "requires --no-refinement_phase" in combined


def test_cli_rejects_t_pics_without_explore_top_k_1():
    proc = _run_teh(
        "--global_phase",
        "--t_pics_source",
        "11enkavi2019recentprobes",
        "--explore_from_population_parents",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "requires --explore_population_top_k 1" in combined


def test_cli_t_pics_flag_auto_selects_enkavi_for_choice13k():
    proc = _run_teh(
        "--t_pics",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "auto source for 1peterson2021using -> 11enkavi2019recentprobes" in combined
    assert "requires --global_phase" in combined


def test_cli_t_pics_source_without_value_auto_selects():
    proc = _run_teh(
        "--t_pics_source",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "auto source for 1peterson2021using -> 11enkavi2019recentprobes" in combined
    assert "requires --global_phase" in combined


def test_cli_t_pics_source_config_unfiltered_uses_pre_fix_cct_source():
    proc = _run_teh(
        "--t_pics",
        "--t_pics_source_config",
        "analysis/config/transfer_source/t_pics_score_weighted.yaml",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
        dataset="3frey2017cct",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "auto source for 3frey2017cct -> 10frey2017risk" in combined
    assert "requires --global_phase" in combined


def test_cli_t_pics_flag_cct_defaults_to_enkavi_not_frey_risk():
    proc = _run_teh(
        "--t_pics",
        "--explore_from_population_parents",
        "--explore_population_top_k",
        "1",
        "--explore_candidates",
        "50",
        "--no-refinement_phase",
        dataset="3frey2017cct",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "auto source for 3frey2017cct -> 11enkavi2019recentprobes" in combined
    assert "10frey2017risk" not in combined
    assert "requires --global_phase" in combined
