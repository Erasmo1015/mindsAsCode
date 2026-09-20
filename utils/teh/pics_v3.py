"""ICLR PICS v3 — final method constants.

Supersedes preliminary T-PICS v2 (``t_pics_*_sa40_v2``) for future ICLR runs.
Does not overwrite v1 or preliminary-v2 artifacts.

Concepts (do not conflate):
- Method / output kind: ``pics_v3`` / ``pics_v3_g1`` / ``pics_v3_independent``
- Shared data protocol: ``structure_aware_v3``
- Annotation taxonomy: ``schema_v5`` (five constructs; final). Schema-v4 artifacts
  remain frozen and must not be overwritten.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Final PICS v3 kinds ---
KIND = "pics_v3"
G1_KIND = "pics_v3_g1"
INDEPENDENT_KIND = "pics_v3_independent"
RUN_TAG = "g5e50p30_occurrence_eb_pics_v3"
ANNOTATION_PACKAGE = "pics_v3_g1_schema_v4"  # legacy completed package
ANNOTATION_PACKAGE_V5 = "pics_v3_g1_schema_v5"

LIMITED_DATA_PROTOCOL = "structure_aware_v3"
LIMITED_TRAIN_VAL = 40

# Qwen2.5-Coder-32B-Instruct production context (fully chat-templated).
# 16k-class ceiling (preliminary-v2 pair): 14000 input + 1024 out ≤ 16384.
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
VLLM_MAX_MODEL_LEN = 16_384
HARD_PROMPT_TOKEN_CAP = 14_000
LLM_MAX_TOKENS = 1_024
MAX_PARENT_CHARS = 5_000
MAX_PROMPT_TRAIN_TRIALS = 60
MAX_PROMPT_TRIALS_PER_PROBLEM = 5
SAMPLE_SIZE = 8

G2_PAIRED_PACK_VERSION = "g2_paired_pack_pics_v3"
G2_PAIRED_PACK_FILENAME = "g2_paired_packing.json"

SOURCE_DIR = _REPO_ROOT / "analysis/config/T-PICS/Transfer_source/pics_v3"
# Active runtime map for currently submitted G.2 jobs (do not change mid-flight).
SOURCE_YAML = SOURCE_DIR / "occurrence_eb_schema4_iter10_pics_v3.yaml"
# Five-construct selector freeze (schema-v4 annotations, construct subset).
SOURCE_YAML_5CONSTRUCT = SOURCE_DIR / "occurrence_eb_schema4_5construct_iter10_pics_v3.yaml"
# Schema-v5 freeze (written only after v5 reannotation + identical-map check).
SOURCE_YAML_SCHEMA5 = SOURCE_DIR / "occurrence_eb_schema5_iter10_pics_v3.yaml"
FREEZE_DIR = SOURCE_DIR / "schema4_occurrence_eb_freeze"
FREEZE_DIR_SCHEMA5 = SOURCE_DIR / "schema5_occurrence_eb_freeze"
ANNOTATIONS_ROOT = (
    _REPO_ROOT / "analysis_2026Sep/mem/pics_v3_g1_schema_v4/annotations"
)
ANNOTATIONS_ROOT_V5 = (
    _REPO_ROOT / "analysis_2026Sep/mem/pics_v3_g1_schema_v5/annotations"
)
DOCS = _REPO_ROOT / "analysis/config/T-PICS/docs/Documentation_pics_v3.md"

# --- Frozen non-final predecessors (do not overwrite) ---
V1_SOURCE_YAML = (
    _REPO_ROOT / "analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml"
)
V1_G1_JOBS = tuple(range(257174, 257189))
V1_OCCURRENCE_EB_JOB = 257756

PRELIMINARY_V2_KIND = "t_pics_gated_sa40_v2"
PRELIMINARY_V2_G1_KIND = "t_pics_g1_sa40_v2"
PRELIMINARY_V2_INDEPENDENT_KIND = "t_pics_gated_independent_sa40_v2"
PRELIMINARY_V2_SOURCE_YAML = (
    _REPO_ROOT
    / "analysis/config/T-PICS/Transfer_source/v2/occurrence_eb_schema4_iter10_sa40_v2.yaml"
)
PRELIMINARY_V2_HARD_PROMPT_TOKEN_CAP = 14_000
PRELIMINARY_V2_VLLM_MAX_MODEL_LEN = 16_384
PRELIMINARY_V2_MAX_PARENT_CHARS = 3_500
PRELIMINARY_V2_ENKAVI_JOB = 258518

# GPU layout templates (documentation / dry-run only; not launched here).
GPU_TEMPLATES = {
    "h100_nvl": {"gpus": 1, "tensor_parallel": 1, "dtype": "bfloat16"},
    "2x_l40s": {"gpus": 2, "tensor_parallel": 2, "dtype": "bfloat16"},
    "4x_rtx_3090": {"gpus": 4, "tensor_parallel": 4, "dtype": "float16"},
}
