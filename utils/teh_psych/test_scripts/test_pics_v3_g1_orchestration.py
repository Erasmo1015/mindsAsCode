"""Bounded regression: PICS v3 G.1 orchestration (global-only, no gated/YAML)."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.teh.teh_datasets import emnlp_ordinal_range

DATASETS = [
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "12badham2017deficits",
    "mixed_gambles",
    "bergert_nosofsky_2007",
    "guan_2020_stopping",
    "steyvers_2009_bandit",
    "13schulz2020finding",
    "14kool2016when",
]

EXPECTED_RANGES = {
    "4wulff2018description": (1290, 1319),
    "5speekenbrink2008learning": (0, 22),
    "12badham2017deficits": (0, 9),
}

FORBIDDEN_TOKENS = (
    "--t_pics_gated_independent",
    "--t_pics_gated_transfer",
    "--t_pics_source_config",
    "--output_kind",
    "Transfer_source/v2/",
    "occurrence_eb_schema4_iter10_sa40_v2",
    "selected_source",
)

# Standalone --t_pics (not a prefix of longer flags) must not appear.
FORBIDDEN_EXACT = {"--t_pics"}


def _emit_g1_commands() -> list[list[str]]:
    script = r"""
set -euo pipefail
REPO_ROOT="$(pwd)"
source cluster/v2/ours/Qwen/_common.sh
source "$(t_pics_v2_origami_common)"
# Quiet stdout chatter from helpers (keep argv lines clean).
append_local_psych_args() {
  local -n _out="$1"
  local path="${LOCAL_PSYCH_DATASET:-datasets/downloaded/Psych-101}"
  if [[ -d "${path}" ]]; then
    _out+=(--local_dataset "${path}")
  fi
}
export REPO="${REPO_ROOT}"
export PSYCH_DATASET_SPLIT=train
export PSYCH_SPLIT=train
export KIND=pics_v3_g1
export LIMITED_DATA_PROTOCOL=structure_aware_v3
export LIMITED_TRAIN_VAL=40
export GLOBAL_ITERS=10
export WANDB_PROJECT_NAME=teh_pics_v3
export MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct
export LLM_URL=http://localhost:0/v1
export LOCAL_PSYCH_DATASET=
DATASETS=(
  1peterson2021using 2plonsky2018when 3frey2017cct 4wulff2018description
  5speekenbrink2008learning 7hilbig2014generalized 10frey2017risk
  11enkavi2019recentprobes 12badham2017deficits mixed_gambles
  bergert_nosofsky_2007 guan_2020_stopping steyvers_2009_bandit
  13schulz2020finding 14kool2016when
)
for ds in "${DATASETS[@]}"; do
  unset RANGE_START_ORDINAL RANGE_END_ORDINAL
  export DATASET="${ds}"
  SEED_PATH="$(t_pics_v2_seed_path "${ds}")"
  export SEED_PATH
  t_pics_v2_resolve_emnlp_range "${ds}" >/dev/null
  OUT_DIR="$(t_pics_v2_out_root "${ds}" "${KIND}")/job_<id>"
  export OUT_DIR
  t_pics_v3_fill_g1_args
  printf '%q ' "${PICS_ARGS[@]}"
  printf '\n'
done
"""
    out = subprocess.check_output(
        ["bash", "-c", script],
        cwd=str(REPO),
        text=True,
        stderr=subprocess.DEVNULL,
    )
    lines = [
        ln.strip()
        for ln in out.splitlines()
        if ln.strip() and not ln.strip().startswith("[INFO]") and "--dataset" in ln
    ]
    assert len(lines) == 15, f"expected 15 G.1 argv lines, got {len(lines)}:\n{out}"
    return [shlex.split(ln) for ln in lines]


def _flag_value(argv: list[str], flag: str) -> str:
    try:
        i = argv.index(flag)
    except ValueError as exc:
        raise AssertionError(f"missing {flag}") from exc
    assert i + 1 < len(argv), f"{flag} missing value"
    return argv[i + 1]


_TEH_PARSER = None


def _teh_parser():
    """Build the real teh.py ArgumentParser once (without running a job)."""
    global _TEH_PARSER
    if _TEH_PARSER is not None:
        return _TEH_PARSER
    import teh as teh_mod

    captured: dict = {}

    def _capture(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", _capture):
        with mock.patch.object(sys, "argv", ["teh.py"]):
            with pytest.raises(SystemExit):
                teh_mod.main()
    assert "parser" in captured
    _TEH_PARSER = captured["parser"]
    return _TEH_PARSER


def _parse_teh_argv(argv: list[str]):
    """Accept argv via the real teh.py ArgumentParser without running the job."""
    return _teh_parser().parse_args(argv)


@pytest.fixture(scope="module")
def g1_commands() -> list[list[str]]:
    return _emit_g1_commands()


def test_pics_v3_g1_fifteen_commands_and_ranges(g1_commands):
    assert len(g1_commands) == 15
    seen = []
    for argv in g1_commands:
        ds = _flag_value(argv, "--dataset")
        seen.append(ds)
        start = int(_flag_value(argv, "--range_start_ordinal"))
        end = int(_flag_value(argv, "--range_end_ordinal"))
        assert _flag_value(argv, "--participant_scope") == "range"
        expected = EXPECTED_RANGES.get(ds)
        if expected is None:
            expected = (0, 29)
        assert (start, end) == expected, ds
        assert (start, end) == emnlp_ordinal_range(ds), ds
    assert seen == DATASETS


def test_pics_v3_g1_required_settings_and_auto_prompt(g1_commands):
    for argv in g1_commands:
        joined = " ".join(argv)
        assert "--prefer_auto_llm_prompt" in argv
        # Not a teh.py CLI flag; fail-closed is wired in main() (see test below).
        assert "--require_auto_llm_prompt" not in argv
        assert "--dataset_prompt_file" not in argv
        assert _flag_value(argv, "--limited_data_protocol") == "structure_aware_v3"
        assert _flag_value(argv, "--hard_prompt_token_cap") == "14000"
        assert _flag_value(argv, "--llm_max_tokens") == "1024"
        assert _flag_value(argv, "--max_parent_chars") == "5000"
        assert _flag_value(argv, "--n_eval_seeds") == "1"
        assert _flag_value(argv, "--global_iters") == "10"
        assert _flag_value(argv, "--n_iterations") == "0"
        assert _flag_value(argv, "--explore_candidates") == "0"
        assert _flag_value(argv, "--n_candidates") == "10"
        assert _flag_value(argv, "--fresh_n_candidates") == "10"
        assert _flag_value(argv, "--sample_size") == "8"
        assert _flag_value(argv, "--elite_pool_size") == "50"
        assert "--global_phase" in argv
        assert _flag_value(argv, "--wandb_project") == "teh_pics_v3"
        out = _flag_value(argv, "--output_dir")
        assert "/pics_v3_g1/" in out.replace("\\", "/")
        assert "pics_v3_g1" in out
        for tok in FORBIDDEN_TOKENS:
            assert tok not in joined, tok
        assert FORBIDDEN_EXACT.isdisjoint(argv)


def test_require_auto_llm_prompt_is_not_a_cli_flag():
    parser = _teh_parser()
    opt_strings = {
        opt
        for action in parser._actions
        for opt in (action.option_strings or [])
    }
    assert "--prefer_auto_llm_prompt" in opt_strings
    assert "--require_auto_llm_prompt" not in opt_strings


def _g1_require_auto_predicate(ns) -> bool:
    """Mirror teh.py main() g1_require_auto (PICS v3 only)."""
    return bool(
        getattr(ns, "prefer_auto_llm_prompt", False)
        and getattr(ns, "global_phase", False)
        and int(getattr(ns, "n_iterations", 0) or 0) == 0
        and str(getattr(ns, "limited_data_protocol", "") or "") == "structure_aware_v3"
    )


def test_pics_v3_g1_fail_closed_auto_prompt_wiring(g1_commands):
    """PICS v3 G.1 enables require_auto; v1/v2 G.1-shaped runs do not."""
    teh_src = (REPO / "teh.py").read_text(encoding="utf-8")
    assert "g1_require_auto" in teh_src
    assert '== "structure_aware_v3"' in teh_src
    assert "require_auto_llm_prompt=bool(t_pics_gated or g1_require_auto)" in teh_src
    runtime_src = (REPO / "utils/teh/teh_runtime.py").read_text(encoding="utf-8")
    assert "if prefer_auto_llm_prompt or used_dataset_prompt_file or require_auto_llm_prompt:" in runtime_src
    assert "reference_prompt = None" in runtime_src
    assert "refusing merge fallback" in runtime_src
    assert "Automatic target-prompt generation failed" in runtime_src

    for argv in g1_commands:
        ns = _parse_teh_argv(argv)
        assert bool(ns.prefer_auto_llm_prompt) is True
        assert bool(ns.global_phase) is True
        assert int(ns.n_iterations) == 0
        assert ns.limited_data_protocol == "structure_aware_v3"
        assert getattr(ns, "dataset_prompt_file", None) in (None, "")
        assert _g1_require_auto_predicate(ns) is True

    # Historical v1 / preliminary-v2 G.1 shape: same prefer/global/n_iter=0, different protocol.
    for proto in ("structure_aware", "structure_aware_v2"):
        hist = argparse.Namespace(
            prefer_auto_llm_prompt=True,
            global_phase=True,
            n_iterations=0,
            limited_data_protocol=proto,
        )
        assert _g1_require_auto_predicate(hist) is False, proto


def test_setup_teh_run_prompts_require_auto_skips_reference_and_refuses_fallback(tmp_path):
    """Internal require_auto_llm_prompt guard: no registered prompt, no merge fallback."""
    from utils.teh import teh_runtime

    alias = "13schulz2020finding"
    ref = teh_runtime.resolve_dataset_reference_prompt_path(alias)
    assert ref is not None and ref.is_file()

    seed = REPO / "persona_code_example/teh/categorical_uniform.py"
    run_dir = tmp_path / "g1_auto"

    def _boom(*args, **kwargs):
        raise RuntimeError("llm unavailable")

    with mock.patch.object(
        teh_runtime,
        "_prompt_sample_pid_and_instruction",
        return_value=(0, "instruction"),
    ), mock.patch.object(
        teh_runtime,
        "_load_prompt_observation_trials",
        return_value=([{"trial_id": 0, "choice": "A"}], None, None),
    ), mock.patch.object(
        teh_runtime, "_generate_prompt_via_llm", side_effect=_boom
    ):
        with pytest.raises(
            RuntimeError,
            match="Automatic target-prompt generation failed|refusing merge fallback",
        ):
            teh_runtime.setup_teh_run_prompts(
                run_dir,
                alias,
                seed,
                client=object(),
                model_name="dummy",
                use_llm=True,
                prefer_auto_llm_prompt=True,
                require_auto_llm_prompt=True,
                limited_data_protocol="structure_aware_v3",
                limited_train_val=40,
                split_ratio=0.6,
                split_seed=0,
            )

    meta = run_dir / "prompts" / "prompt_meta.json"
    assert not meta.is_file(), "fail-closed must not finalize prompt_meta as reference/merge"
    # Reference file must not have been copied through as the evolution prompt.
    infer = run_dir / "prompts" / "infer_single_choice.txt"
    if infer.is_file():
        assert infer.read_text(encoding="utf-8") != ref.read_text(encoding="utf-8")


def test_pics_v3_g1_accepted_by_teh_parser(g1_commands):
    for argv in g1_commands:
        ns = _parse_teh_argv(argv)
        assert ns.limited_data_protocol == "structure_aware_v3"
        assert int(ns.hard_prompt_token_cap) == 14000
        assert int(ns.llm_max_tokens) == 1024
        assert int(ns.max_parent_chars) == 5000
        assert int(ns.n_eval_seeds) == 1
        assert bool(ns.prefer_auto_llm_prompt) is True
        assert ns.participant_scope == "range"
        assert int(ns.global_iters) == 10
        assert int(ns.n_iterations) == 0
        assert not getattr(ns, "t_pics_gated_transfer", False)
        assert not getattr(ns, "t_pics_gated_independent", False)
        assert getattr(ns, "t_pics_source_config", None) in (None, "")


def test_pics_v3_dry_run_prints_g1_without_sbatch_or_gated():
    import os

    env = dict(os.environ)
    env["SKIP_PYTEST"] = "1"
    env.pop("RANGE_START_ORDINAL", None)
    env.pop("RANGE_END_ORDINAL", None)
    proc = subprocess.run(
        ["bash", "analysis/config/T-PICS/pics_v3/dry_run_commands.sh"],
        cwd=str(REPO),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    out = proc.stdout or ""
    assert "This script did not invoke sbatch." in out
    assert "printed 15 G.1 commands" in out
    cmd_lines = [
        ln
        for ln in out.splitlines()
        if ("DRY_RUN:" in ln or "CONFIRM:" in ln) and "/teh.py" in ln
    ]
    assert len(cmd_lines) == 15
    joined_cmds = "\n".join(cmd_lines)
    assert joined_cmds.count("--prefer_auto_llm_prompt") == 15
    assert joined_cmds.count("--hard_prompt_token_cap") == 15
    assert joined_cmds.count("pics_v3_g1") == 15
    assert "--t_pics_gated_independent" not in joined_cmds
    assert "--t_pics_gated_transfer" not in joined_cmds
    assert "--t_pics_source_config" not in joined_cmds
    assert "--output_kind" not in joined_cmds
    assert "occurrence_eb_schema4_iter10_sa40_v2" not in joined_cmds
    assert "Transfer_source/v2/" not in joined_cmds
