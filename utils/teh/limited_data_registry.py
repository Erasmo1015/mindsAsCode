"""Structure-aware limited-data registry for the 15 PICS datasets.

Classifications follow loader/task semantics (history reset, unit identity),
not the legacy TEH pseudo-block split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from data_modules.psych101_binary import normalize_psych101_dataset_alias
from utils.teh.teh_datasets import MIXED_GAMBLES, is_external_dataset, is_mixed_gambles_dataset

INDEPENDENT_TRIAL = "independent_trial"
RESETTING_UNIT = "resetting_unit"
CONTINUOUS_SESSION = "continuous_session"

LIMITED_DATA_PROTOCOL_OFF = "off"
LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE = "structure_aware"
LIMITED_DATA_PROTOCOLS = (
    LIMITED_DATA_PROTOCOL_OFF,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
)


@dataclass(frozen=True)
class LimitedDataSpec:
    dataset: str
    display_name: str
    category: str
    unit_type: str
    unit_keys: Tuple[str, ...]
    history_resets_at_unit: bool
    prefix_valid: bool
    chronological_split: bool
    notes: str


def _spec(
    dataset: str,
    display_name: str,
    category: str,
    unit_type: str,
    unit_keys: Tuple[str, ...],
    *,
    history_resets_at_unit: bool,
    prefix_valid: bool,
    chronological_split: bool,
    notes: str,
) -> LimitedDataSpec:
    return LimitedDataSpec(
        dataset=dataset,
        display_name=display_name,
        category=category,
        unit_type=unit_type,
        unit_keys=unit_keys,
        history_resets_at_unit=history_resets_at_unit,
        prefix_valid=prefix_valid,
        chronological_split=chronological_split,
        notes=notes,
    )


LIMITED_DATA_REGISTRY: Dict[str, LimitedDataSpec] = {
    "1peterson2021using": _spec(
        "1peterson2021using",
        "Choice13k",
        RESETTING_UNIT,
        "problem",
        ("block_index",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="5-trial problems; history accumulates within a problem and resets between problems.",
    ),
    "2plonsky2018when": _spec(
        "2plonsky2018when",
        "CPC18",
        RESETTING_UNIT,
        "problem",
        ("block_index",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="25-trial feedback problems; history resets between problems.",
    ),
    "3frey2017cct": _spec(
        "3frey2017cct",
        "Frey CCT",
        RESETTING_UNIT,
        "round",
        ("round_id", "round"),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="Sequential card flips within a round; history resets each round.",
    ),
    "4wulff2018description": _spec(
        "4wulff2018description",
        "Wulff description",
        INDEPENDENT_TRIAL,
        "problem",
        ("block_index",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="One description-based choice per lottery pair; history is empty.",
    ),
    "5speekenbrink2008learning": _spec(
        "5speekenbrink2008learning",
        "Speekenbrink learning",
        CONTINUOUS_SESSION,
        "session",
        (),
        history_resets_at_unit=False,
        prefix_valid=True,
        chronological_split=True,
        notes="Single 200-trial weather-learning sequence; history does not reset. "
        "Default split is chronological (contiguous test suffix). "
        "--speekenbrink_split legacy restores shuffled TEH pseudo-blocks.",
    ),
    "7hilbig2014generalized": _spec(
        "7hilbig2014generalized",
        "Hilbig generalized",
        INDEPENDENT_TRIAL,
        "trial",
        (),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="IID product-pair choices with empty history; parser emits one block.",
    ),
    "10frey2017risk": _spec(
        "10frey2017risk",
        "Frey balloon risk",
        RESETTING_UNIT,
        "balloon",
        ("balloon_id",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="Sequential pumps within a balloon; history resets each balloon.",
    ),
    "11enkavi2019recentprobes": _spec(
        "11enkavi2019recentprobes",
        "Enkavi recent probes",
        INDEPENDENT_TRIAL,
        "trial",
        (),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="Each probe is a new memory-set recognition trial; history is empty.",
    ),
    "12badham2017deficits": _spec(
        "12badham2017deficits",
        "Badham category learning",
        RESETTING_UNIT,
        "rule_block",
        ("rule_block_id",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="Feedback category learning within a rule block; history resets across rule blocks.",
    ),
    MIXED_GAMBLES: _spec(
        MIXED_GAMBLES,
        "mixed gambles",
        INDEPENDENT_TRIAL,
        "trial",
        (),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="History is always empty. Repeated (gain, loss, cert) rows are still independent choices.",
    ),
    "bergert_nosofsky_2007": _spec(
        "bergert_nosofsky_2007",
        "Bergert & Nosofsky 2007",
        INDEPENDENT_TRIAL,
        "problem",
        ("problem_id",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="40 pairwise cue problems; history is empty.",
    ),
    "guan_2020_stopping": _spec(
        "guan_2020_stopping",
        "Guan 2020 stopping",
        RESETTING_UNIT,
        "stopping_problem",
        ("condition_index", "problem_id"),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="Continue/stop positions within one stopping problem; history resets between problems.",
    ),
    "steyvers_2009_bandit": _spec(
        "steyvers_2009_bandit",
        "Steyvers 2009 bandit",
        RESETTING_UNIT,
        "game",
        ("game",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="15-trial 4-arm games; history resets every game.",
    ),
    "13schulz2020finding": _spec(
        "13schulz2020finding",
        "Schulz 2020 exp4",
        RESETTING_UNIT,
        "round",
        ("round",),
        history_resets_at_unit=True,
        prefix_valid=True,
        chronological_split=False,
        notes="10 pulls per round; options and history reset each round.",
    ),
    "14kool2016when": _spec(
        "14kool2016when",
        "Kool 2016 two-step",
        CONTINUOUS_SESSION,
        "day",
        ("presented_day",),
        history_resets_at_unit=False,
        prefix_valid=True,
        chronological_split=True,
        notes="History carries across days. Split is contiguous usable days (seed unused). "
        "A day prefix (stage-1 only) is valid; a stage-2 suffix without stage-1 is not.",
    ),
}


def normalize_limited_dataset_alias(dataset: str) -> str:
    alias = str(dataset).strip()
    if alias in LIMITED_DATA_REGISTRY:
        return alias
    if is_mixed_gambles_dataset(alias) or is_external_dataset(alias):
        return alias
    return normalize_psych101_dataset_alias(alias)


def limited_data_spec(dataset: str) -> LimitedDataSpec:
    alias = normalize_limited_dataset_alias(dataset)
    spec = LIMITED_DATA_REGISTRY.get(alias)
    if spec is None:
        raise KeyError(
            f"No structure-aware limited-data spec for {dataset!r}. "
            f"Known: {sorted(LIMITED_DATA_REGISTRY)}"
        )
    return spec


def normalize_limited_data_protocol(value: object) -> str:
    if value is None:
        return LIMITED_DATA_PROTOCOL_OFF
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "none", "off", "legacy"}:
        return LIMITED_DATA_PROTOCOL_OFF
    if text in {"structure_aware", "structure-aware", "1", "true", "yes", "on"}:
        return LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE
    raise ValueError(
        f"limited_data_protocol must be 'off' or 'structure_aware', got {value!r}"
    )
