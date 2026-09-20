#!/usr/bin/env bash
# ICLR PICS v3 chronological commands. Default is dry-run.
# Prints the exact 15 global-only G.1 commands via t_pics_v3_fill_g1_args.
# This script never calls sbatch / never launches GPU jobs.
set -euo pipefail
CONFIRM=0
if [[ "${1:-}" == "--confirm" ]]; then
  CONFIRM=1
fi
prefix="echo DRY_RUN:"
if [[ "$CONFIRM" -eq 1 ]]; then
  prefix="echo CONFIRM:"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/cluster/v2/ours/Qwen/_common.sh"
# shellcheck source=/dev/null
source "$(t_pics_v2_origami_common)"

PY="$(t_pics_v2_python)"
export REPO="${REPO_ROOT}"
export PSYCH_DATASET_SPLIT="${PSYCH_DATASET_SPLIT:-train}"
export PSYCH_SPLIT="${PSYCH_SPLIT:-${PSYCH_DATASET_SPLIT}}"
export KIND="${KIND:-pics_v3_g1}"
export LIMITED_DATA_PROTOCOL="${LIMITED_DATA_PROTOCOL:-structure_aware_v3}"
export LIMITED_TRAIN_VAL="${LIMITED_TRAIN_VAL:-40}"
export GLOBAL_ITERS="${GLOBAL_ITERS:-10}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-teh_pics_v3}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-32B-Instruct}"
export LLM_URL="${LLM_URL:-http://localhost:0/v1}"
export LOCAL_PSYCH_DATASET="${LOCAL_PSYCH_DATASET:-}"

DATASETS=(
  1peterson2021using 2plonsky2018when 3frey2017cct 4wulff2018description
  5speekenbrink2008learning 7hilbig2014generalized 10frey2017risk
  11enkavi2019recentprobes 12badham2017deficits mixed_gambles
  bergert_nosofsky_2007 guan_2020_stopping steyvers_2009_bandit
  13schulz2020finding 14kool2016when
)

if [[ "${SKIP_PYTEST:-0}" != "1" ]]; then
  $prefix pytest -q \
    utils/teh_psych/test_pics_v3_protocol.py \
    utils/teh_psych/test_pics_v3_g1_orchestration.py \
    utils/teh_psych/test_tpics_v2_protocol.py \
    utils/teh_psych/test_g2_paired_packing.py \
    utils/teh_psych/test_iclr_baseline_guards.py
fi

echo "# GPU templates (vLLM Qwen2.5-Coder-32B-Instruct --max-model-len 32768):"
echo "#   H100 NVL: TP=1 | 2xL40S: TP=2 | 4xRTX3090: TP=4"
echo "# 1) CPU preflight (above)."
echo "# 2) Exact 15 PICS v3 G.1 commands (global-only source population;"
echo "#    KIND=pics_v3_g1 sets output_dir only; not a teh.py flag; no gated/YAML):"

# Quiet helper chatter so the 15 G.1 lines stay clean.
append_local_psych_args() {
  local -n _out="$1"
  local path="${LOCAL_PSYCH_DATASET:-datasets/downloaded/Psych-101}"
  if [[ -d "${path}" ]]; then
    _out+=(--local_dataset "${path}")
  fi
}

g1_count=0
for ds in "${DATASETS[@]}"; do
  unset RANGE_START_ORDINAL RANGE_END_ORDINAL
  export DATASET="${ds}"
  export SEED_PATH
  SEED_PATH="$(t_pics_v2_seed_path "${ds}")"
  t_pics_v2_resolve_emnlp_range "${ds}" >/dev/null
  export OUT_DIR
  OUT_DIR="$(t_pics_v2_out_root "${ds}" "${KIND}")/job_<id>"
  t_pics_v3_fill_g1_args
  # Print one shell-ready line; KIND is only in --output_dir.
  cmd=( "$PY" "$REPO_ROOT/teh.py" "${PICS_ARGS[@]}" )
  $prefix "${cmd[@]@Q}"
  g1_count=$((g1_count + 1))
done
echo "# printed ${g1_count} G.1 commands (expected 15)"

echo "# 3) schema-v4 annotate new G.1 programs -> pics_v3_g1_schema_v4"
$prefix echo "annotate pics_v3_g1 programs into analysis_2026Sep/mem/pics_v3_g1_schema_v4/"

echo "# 4) Occurrence-EB refit (unchanged formulas) -> freeze YAML"
$prefix echo "write analysis/config/T-PICS/Transfer_source/pics_v3/occurrence_eb_schema4_iter10_pics_v3.yaml"

echo "# 5) validate source map (no transfer/test/G.2 peek)"
$prefix echo "validate pics_v3 source YAML"

echo "# 6) regenerate G.2 paired-packing audit with selected pics_v3 sources"
$prefix echo "audit g2_paired_pack_pics_v3 under 30000/5000"

echo "# 7) 15 pics_v3 gated target jobs (only after steps 4-6; not G.1)"
$prefix echo "gated pics_v3 targets after Occurrence-EB freeze (separate from G.1)"

echo "# 8) baselines share structure_aware_v3 context; OpenEvolve is not pics_v3"
$prefix echo "Centaur/LM/PT/OE with --limited_data_protocol structure_aware_v3 (report, do not launch here)"

echo "# This script did not invoke sbatch."
