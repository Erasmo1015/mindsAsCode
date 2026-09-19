"""ICLR T-PICS v2 path/kind constants. Does not overwrite v1 frozen artifacts."""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

KIND_V2 = "t_pics_gated_sa40_v2"
INDEPENDENT_KIND_V2 = "t_pics_gated_independent_sa40_v2"
G1_KIND_V2 = "t_pics_g1_sa40_v2"
RUN_TAG_V2 = "g5e50p10_occurrence_eb_sa40_v2"

V1_FROZEN_SOURCE_YAML = (
    _REPO_ROOT / "analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml"
)
V2_SOURCE_YAML = (
    _REPO_ROOT
    / "analysis/config/T-PICS/Transfer_source/v2/occurrence_eb_schema4_iter10_sa40_v2.yaml"
)
V2_FREEZE_DIR = (
    _REPO_ROOT / "analysis/config/T-PICS/Transfer_source/v2/schema4_occurrence_eb_freeze"
)
V2_DOCS = _REPO_ROOT / "analysis/config/T-PICS/docs/Documentation_v2.md"

V1_G1_JOBS = tuple(range(257174, 257189))
V1_OCCURRENCE_EB_JOB = 257756

LIMITED_DATA_PROTOCOL_V2 = "structure_aware_v2"
LIMITED_TRAIN_VAL_V2 = 40
