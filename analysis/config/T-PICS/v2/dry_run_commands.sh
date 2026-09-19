#!/usr/bin/env bash
# ICLR T-PICS v2 chronological commands. Default is dry-run.
# Pass --confirm to print the same commands without the dry-run prefix.
# This script never calls sbatch.
set -euo pipefail
CONFIRM=0
if [[ "${1:-}" == "--confirm" ]]; then
  CONFIRM=1
fi
prefix="echo DRY_RUN:"
if [[ "$CONFIRM" -eq 1 ]]; then
  prefix="echo CONFIRM:"
fi

DATASETS=(
  1peterson2021using 2plonsky2018when 3frey2017cct 4wulff2018description
  5speekenbrink2008learning 7hilbig2014generalized 10frey2017risk
  11enkavi2019recentprobes 12badham2017deficits mixed_gambles
  bergert_nosofsky_2007 guan_2020_stopping steyvers_2009_bandit
  13schulz2020finding 14kool2016when
)

$prefix pytest -q utils/teh_psych/test_tpics_v2_protocol.py utils/teh_psych/test_iclr_baseline_guards.py

echo "# 1) CPU preflight (above). 2) 15 v2 G.1 source-population jobs:"
for ds in "${DATASETS[@]}"; do
  $prefix python teh.py --dataset "$ds" --psych_dataset_split train \
    --t_pics_gated_independent --t_pics_gated_transfer \
    --limited_data_protocol structure_aware_v2 --limited_train_val 40 \
    --split_ratio 0.6 --split_seed 0 \
    --output_kind t_pics_g1_sa40_v2
done

echo "# 3) schema-v4 annotate new G.1 programs (same motif code; new output dir)"
$prefix echo "annotate v2 G.1 programs into analysis/config/T-PICS/Transfer_source/v2/"

echo "# 4) Occurrence-EB refit (unchanged formulas) -> freeze YAML"
$prefix echo "write analysis/config/T-PICS/Transfer_source/v2/occurrence_eb_schema4_iter10_sa40_v2.yaml"

echo "# 5) validate source map (no transfer peek)"
$prefix echo "validate v2 source YAML"

echo "# 6) 15 v2 gated target jobs"
for ds in "${DATASETS[@]}"; do
  $prefix python teh.py --dataset "$ds" --psych_dataset_split train \
    --t_pics_gated_transfer \
    --limited_data_protocol structure_aware_v2 --limited_train_val 40 \
    --t_pics_source_config analysis/config/T-PICS/Transfer_source/v2/occurrence_eb_schema4_iter10_sa40_v2.yaml \
    --output_kind t_pics_gated_sa40_v2
done

echo "# 7) baselines into v2 output dirs (Centaur: kool, speekenbrink, hilbig, enkavi; LM/PT: same plus any observed-set tables; OE: all future jobs)"
$prefix echo "Centaur/LM/PT/OE with --limited_data_protocol structure_aware_v2"

echo "# This script did not invoke sbatch."
