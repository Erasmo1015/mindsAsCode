"""EMNLP participant ordinals for the 15 live in teh_datasets.yaml."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from utils.teh.limited_data_registry import LIMITED_DATA_REGISTRY
from utils.teh.teh_datasets import (
    TEH_DATASETS_YAML,
    emnlp_ordinal_range,
    load_teh_dataset_registry,
    teh_dataset_row,
)

REPO = Path(__file__).resolve().parents[1]
_BASH_RANGES = REPO / "cluster/2026Sep_Sparse_Data/_emnlp_lm_pt_datasets.sh"

_EXCEPTIONS = {
    "4wulff2018description": (1290, 1339),
    "5speekenbrink2008learning": (0, 22),
    "12badham2017deficits": (0, 9),
}


def test_registry_covers_all_15():
    rows = load_teh_dataset_registry()
    aliases = {str(r["alias"]) for r in rows}
    assert aliases == set(LIMITED_DATA_REGISTRY)
    for row in rows:
        start, end = int(row["range_start_ordinal"]), int(row["range_end_ordinal"])
        assert 0 <= start <= end


def test_emnlp_exceptions_and_defaults():
    assert emnlp_ordinal_range("4wulff2018description") == (1290, 1339)
    assert emnlp_ordinal_range("5speekenbrink2008learning") == (0, 22)
    assert emnlp_ordinal_range("12badham2017deficits") == (0, 9)
    assert emnlp_ordinal_range("20bergert") == (0, 49)
    assert emnlp_ordinal_range("bergert_nosofsky_2007") == (0, 49)
    for alias in (
        "1peterson2021using",
        "mixed_gambles",
        "guan_2020_stopping",
        "steyvers_2009_bandit",
        "13schulz2020finding",
        "14kool2016when",
    ):
        assert emnlp_ordinal_range(alias) == (0, 49)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown TEH dataset"):
        teh_dataset_row("9wilson2014humans")


def test_yaml_matches_emnlp_lm_pt_bash_table():
    text = _BASH_RANGES.read_text(encoding="utf-8")
    block = re.search(
        r"EMNLP_LM_PT_RANGES=\((.*?)\)",
        text,
        flags=re.S,
    )
    assert block is not None
    bash = {}
    for alias, start, end in re.findall(
        r'"(\S+)\s+(\d+)\s+(\d+)"', block.group(1)
    ):
        bash[alias] = (int(start), int(end))
    yaml_map = {
        str(row["alias"]): emnlp_ordinal_range(str(row["alias"]))
        for row in load_teh_dataset_registry()
    }
    assert bash == yaml_map
    for alias, expected in _EXCEPTIONS.items():
        assert bash[alias] == expected
    assert TEH_DATASETS_YAML.is_file()


def test_centaur_sa40_submit_uses_yaml_ordinals():
    text = (
        REPO / "cluster/2026Sep17_Centaur_SA40/submit_sa40.sh"
    ).read_text(encoding="utf-8")
    aliases = re.findall(
        r"^\s+(5speekenbrink2008learning|bergert_nosofsky_2007|guan_2020_stopping|"
        r"steyvers_2009_bandit|13schulz2020finding|14kool2016when)\s*$",
        text,
        flags=re.M,
    )
    assert aliases == [
        "5speekenbrink2008learning",
        "bergert_nosofsky_2007",
        "guan_2020_stopping",
        "steyvers_2009_bandit",
        "13schulz2020finding",
        "14kool2016when",
    ]
    assert emnlp_ordinal_range("5speekenbrink2008learning") == (0, 22)
    for alias in aliases[1:]:
        assert emnlp_ordinal_range(alias) == (0, 49)
