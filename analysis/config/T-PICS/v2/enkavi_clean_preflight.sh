#!/usr/bin/env bash
# CPU-only Enkavi G.1 prompt preflight (no sbatch / no GPU).
# Regenerates automatic dataset prompt into a NEW directory and asserts
# probe_in_set is absent from every generation-time artifact.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
PY="${PY:-/careAIDrive/zichang/conda_envs/evo312/bin/python}"

OUT_KIND="t_pics_g1_sa40_v2_enkavi_clean_preflight"
OUT_DIR="generated_outputs/psych101_train/teh/11enkavi2019recentprobes/${OUT_KIND}/preflight_cpu"
PREPARE_DIR="analysis_2026Sep/Sep20_V2/others/source_selection/enkavi_rerun_prepare"
mkdir -p "$PREPARE_DIR"

"$PY" <<'PY'
"""Regenerate Enkavi prompts (CPU merge fallback) + assemble first G.1 generation prompt."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from utils.teh.limited_data_registry import LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2
from utils.teh.prompt_context import assert_no_generation_oracle_leak
from utils.teh.prompt_snapshots import sanitize_problem_for_choose
from utils.teh.teh_runtime import (
    _load_prompt_observation_trials,
    _prompt_sample_pid_and_instruction,
    _runtime_schema_summary_for_prompt,
    build_prompt_generation_llm_user_content,
    setup_teh_run_prompts,
)

ALIAS = "11enkavi2019recentprobes"
OUT = Path(
    "generated_outputs/psych101_train/teh/11enkavi2019recentprobes/"
    "t_pics_g1_sa40_v2_enkavi_clean_preflight/preflight_cpu"
)
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

seed = ROOT / "persona_code_example/te_vanilla/choices13k.py"
prompts_dir = setup_teh_run_prompts(
    OUT,
    ALIAS,
    seed,
    client=None,
    use_llm=False,
    prefer_auto_llm_prompt=False,
    require_auto_llm_prompt=False,
    limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
    limited_train_val=40,
    max_observed_trials_per_participant=40,
    split_ratio=0.6,
    split_seed=0,
)

# Also materialize the prompt-generation LLM user content (what auto_llm would see).
sample_pid, instruction = _prompt_sample_pid_and_instruction(ALIAS)
obs, _te, _man = _load_prompt_observation_trials(
    ALIAS,
    sample_pid,
    split_ratio=0.6,
    split_seed=0,
    limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
    limited_train_val=40,
    max_observed_trials_per_participant=40,
)
llm_user = build_prompt_generation_llm_user_content(ALIAS, instruction, obs)
(prompts_dir / "llm_input_prompt_cpu_rebuild.txt").write_text(llm_user + "\n", encoding="utf-8")
schema = _runtime_schema_summary_for_prompt(obs)
(prompts_dir / "runtime_schema_summary.txt").write_text(schema + "\n", encoding="utf-8")

# Assemble first G.1-style evolution prompt: infer + seed parent + sanitized trials.
infer = (prompts_dir / "infer_single_choice.txt").read_text(encoding="utf-8")
seed_code = (prompts_dir / "seed_program.py").read_text(encoding="utf-8")
# Mirror TEH candidate packing: show a few sanitized observed trials.
from utils.teh.prompt_context import serialize_train_trials_for_prompt_generation

trial_block = serialize_train_trials_for_prompt_generation(obs, max_examples=8)
first_gen = (
    infer.rstrip()
    + "\n\n## Parent program (seed)\n\n```python\n"
    + seed_code.rstrip()
    + "\n```\n\n## Observed train/val examples (sanitized)\n\n"
    + trial_block
    + "\n"
)
(prompts_dir / "first_g1_generation_prompt.txt").write_text(first_gen, encoding="utf-8")

artifacts = [
    prompts_dir / "infer_single_choice.txt",
    prompts_dir / "runtime_contract.txt",
    prompts_dir / "seed_program.py",
    prompts_dir / "runtime_schema_summary.txt",
    prompts_dir / "llm_input_prompt_cpu_rebuild.txt",
    prompts_dir / "first_g1_generation_prompt.txt",
]
# llm_input_prompt.txt may be absent when use_llm=False; check if present.
optional = prompts_dir / "llm_input_prompt.txt"
if optional.is_file():
    artifacts.append(optional)

for path in artifacts:
    text = path.read_text(encoding="utf-8")
    assert_no_generation_oracle_leak(text, context=str(path))
    # Extra safety: no oracle in any sanitized problem dump.
    for t in obs[:5]:
        assert "probe_in_set" not in sanitize_problem_for_choose(t.get("problem") or {})

# Contaminated job 258518 must not be reused.
contam = Path(
    "generated_outputs/psych101_train/teh/11enkavi2019recentprobes/"
    "t_pics_g1_sa40_v2/job_258518/prompts/infer_single_choice.txt"
)
assert contam.is_file()
assert "probe_in_set" in contam.read_text(encoding="utf-8")
assert prompts_dir.resolve() != contam.parent.resolve()

prepare = Path("analysis_2026Sep/Sep20_V2/others/source_selection/enkavi_rerun_prepare")
prepare.mkdir(parents=True, exist_ok=True)
plan = {
    "verdict": "READY_TO_RERUN_ENKAVI_G1",
    "preflight_prompt_dir": str(prompts_dir),
    "do_not_reuse": {
        "contaminated_g1": "generated_outputs/psych101_train/teh/11enkavi2019recentprobes/t_pics_g1_sa40_v2/job_258518",
        "contaminated_annotations": "analysis_2026Sep/mem/t_pics_g1_sa40_v2_schema_v4/annotations/by_dataset/11enkavi2019recentprobes",
        "current_v2_yaml": "analysis/config/T-PICS/Transfer_source/v2/occurrence_eb_schema4_iter10_sa40_v2.yaml",
    },
    "planned_new_paths": {
        "g1_output_kind": "t_pics_g1_sa40_v2_enkavi_rerun",
        "g1_command": (
            "python teh.py --dataset 11enkavi2019recentprobes --psych_dataset_split train "
            "--t_pics_gated_independent --t_pics_gated_transfer "
            "--limited_data_protocol structure_aware_v2 --limited_train_val 40 "
            "--split_ratio 0.6 --split_seed 0 "
            "--output_kind t_pics_g1_sa40_v2_enkavi_rerun"
        ),
        "annotations_root": "analysis_2026Sep/mem/t_pics_g1_sa40_v2_schema_v4_enkavi_rerun",
        "merged_panel_note": (
            "Merge unchanged 14 datasets from t_pics_g1_sa40_v2_schema_v4 with ONLY "
            "new Enkavi rows under schema_v4_enkavi_rerun; never overwrite contaminated Enkavi."
        ),
        "occurrence_eb_yaml": (
            "analysis/config/T-PICS/Transfer_source/v2/"
            "occurrence_eb_schema4_iter10_sa40_v2_enkavi_fix.yaml"
        ),
        "occurrence_eb_freeze_dir": (
            "analysis/config/T-PICS/Transfer_source/v2/"
            "schema4_occurrence_eb_freeze_enkavi_fix"
        ),
    },
    "submit": False,
    "notes": [
        "CPU preflight used merge-fallback infer prompt (no LLM). "
        "Production G.1 should still prefer_auto_llm_prompt; hard assert rejects probe_in_set.",
        "Bergert/Kool and other 14 G.1 jobs are not rerun.",
    ],
}
(prepare / "rerun_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "prompts_dir": str(prompts_dir), "plan": str(prepare / "rerun_plan.json")}, indent=2))
PY

echo "# Planned (NOT submitted) Enkavi-only G.1:"
echo "python teh.py --dataset 11enkavi2019recentprobes --psych_dataset_split train \\"
echo "  --t_pics_gated_independent --t_pics_gated_transfer \\"
echo "  --limited_data_protocol structure_aware_v2 --limited_train_val 40 \\"
echo "  --split_ratio 0.6 --split_seed 0 \\"
echo "  --output_kind t_pics_g1_sa40_v2_enkavi_rerun"
