"""T-PICS source map (score-weighted cosine, self excluded).

Default runtime lookup:
``analysis/config/transfer_source/t_pics_score_weighted_temp_fix.yaml``
(pre-fix CPC18 / Speekenbrink / Frey Risk / Badham programs skipped).
Unfiltered ranking: ``t_pics_score_weighted.yaml``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from utils.teh.limited_data_registry import (
    LIMITED_DATA_REGISTRY,
    normalize_limited_dataset_alias,
)
from utils.teh.teh_datasets import PARTICIPANT_DATASETS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRANSFER_SOURCE_DIR = _REPO_ROOT / "analysis/config/transfer_source"
DEFAULT_T_PICS_SOURCE_CONFIG = _TRANSFER_SOURCE_DIR / "t_pics_score_weighted_temp_fix.yaml"
UNFILTERED_T_PICS_SOURCE_CONFIG = _TRANSFER_SOURCE_DIR / "t_pics_score_weighted.yaml"
_AUTO_SOURCE_TOKENS = frozenset({"", "auto", "__auto__"})

# Fallback if the yaml is missing; must match DEFAULT_T_PICS_SOURCE_CONFIG.
_FALLBACK_T_PICS_SOURCES = {
    "1peterson2021using": "11enkavi2019recentprobes",
    "2plonsky2018when": "11enkavi2019recentprobes",
    "3frey2017cct": "11enkavi2019recentprobes",
    "4wulff2018description": "11enkavi2019recentprobes",
    "5speekenbrink2008learning": "11enkavi2019recentprobes",
    "7hilbig2014generalized": "11enkavi2019recentprobes",
    "10frey2017risk": "11enkavi2019recentprobes",
    "11enkavi2019recentprobes": "1peterson2021using",
    "12badham2017deficits": "11enkavi2019recentprobes",
    "mixed_gambles": "11enkavi2019recentprobes",
    "bergert_nosofsky_2007": "11enkavi2019recentprobes",
    "guan_2020_stopping": "11enkavi2019recentprobes",
    "steyvers_2009_bandit": "7hilbig2014generalized",
    "13schulz2020finding": "11enkavi2019recentprobes",
    "14kool2016when": "1peterson2021using",
}


def _normalize_source_map(raw: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for target, source in raw.items():
        t_alias = normalize_limited_dataset_alias(str(target))
        s_alias = normalize_limited_dataset_alias(str(source))
        out[t_alias] = s_alias
    return out


def load_t_pics_official_sources(
    config_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Load target → source aliases from the transfer_source config yaml."""
    path = Path(config_path) if config_path is not None else DEFAULT_T_PICS_SOURCE_CONFIG
    if path.is_file():
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load T-PICS source config "
                f"({path}). pip install pyyaml"
            ) from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = payload.get("sources") if isinstance(payload, dict) else None
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"T-PICS source config has no sources mapping: {path}")
        return _normalize_source_map({str(k): str(v) for k, v in raw.items()})
    return dict(_FALLBACK_T_PICS_SOURCES)


# Imported by tests; always the yaml map when the file exists.
T_PICS_OFFICIAL_SOURCES = load_t_pics_official_sources()


def is_auto_t_pics_source_token(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _AUTO_SOURCE_TOKENS


def normalize_t_pics_dataset(dataset: str) -> str:
    alias = normalize_limited_dataset_alias(dataset)
    if alias not in LIMITED_DATA_REGISTRY and alias not in PARTICIPANT_DATASETS:
        raise ValueError(f"Unknown T-PICS dataset {dataset!r}")
    return alias


def official_t_pics_source(
    target_dataset: str,
    config_path: Optional[Path] = None,
) -> str:
    alias = normalize_t_pics_dataset(target_dataset)
    mapping = load_t_pics_official_sources(config_path)
    source = mapping.get(alias)
    if source is None:
        cfg = Path(config_path) if config_path is not None else DEFAULT_T_PICS_SOURCE_CONFIG
        raise KeyError(
            f"No official T-PICS source for {alias!r}. "
            f"Add it to {cfg} (known targets: {sorted(mapping)})"
        )
    return source


def resolve_t_pics_source_participant_ids(
    *,
    source_dataset: str,
    repo_root: Path,
    participant_scope: str,
    single_participant_id: int,
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
    all_max_participants: Optional[int],
    participant_ordinals: Optional[Sequence[int]],
    filter_mixed_gambles: bool,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
) -> Tuple[List[int], str]:
    """Same ordinal slice as the target command, clamped to the source valid list.

    Source and target do not share raw participant ids. Ordinals index each
    dataset's own valid_participant_ids.json. If the requested range overshoots
    (e.g. 0-49 on a 23-person source), keep every source person and note it.
    """
    from utils.teh.participant_ids import load_valid_participant_ids

    source_alias = normalize_t_pics_dataset(source_dataset)
    valid = [
        int(x)
        for x in load_valid_participant_ids(
            source_alias,
            repo_root,
            filter_mixed_gambles=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            auto_prepare=True,
        )
    ]
    if not valid:
        raise ValueError(f"No valid participants for T-PICS source {source_alias!r}")
    n = len(valid)
    note = ""
    scope = str(participant_scope).strip().lower()
    if scope == "single":
        if int(single_participant_id) in valid:
            return [int(single_participant_id)], note
        note = (
            f"source {source_alias} has no raw id {int(single_participant_id)}; "
            f"using source ordinal 0 (raw id {valid[0]})"
        )
        return [valid[0]], note
    if scope == "range":
        if range_start_ordinal is None or range_end_ordinal is None:
            raise ValueError(
                "T-PICS source range requires --range_start_ordinal and --range_end_ordinal"
            )
        start = max(0, int(range_start_ordinal))
        end = int(range_end_ordinal)
        if start >= n:
            raise ValueError(
                f"T-PICS source {source_alias} has {n} valid participants; "
                f"requested ordinal start {start} is out of range"
            )
        clamped_end = min(end, n - 1)
        if clamped_end != end or start != int(range_start_ordinal):
            note = (
                f"clamped source ordinals [{start}, {clamped_end}] "
                f"(requested [{range_start_ordinal}, {end}]; source n={n})"
            )
        return valid[start : clamped_end + 1], note
    if scope == "ordinals":
        if not participant_ordinals:
            raise ValueError("T-PICS source ordinals requires --ordinals")
        out: List[int] = []
        seen: set[int] = set()
        skipped: List[int] = []
        for raw_o in participant_ordinals:
            oi = int(raw_o)
            if oi < 0 or oi >= n:
                skipped.append(oi)
                continue
            pid = valid[oi]
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        if not out:
            raise ValueError(
                f"T-PICS source {source_alias} n={n}: none of ordinals "
                f"{list(participant_ordinals)} are in range"
            )
        if skipped:
            note = f"dropped source ordinals {skipped} (source n={n})"
        return out, note
    if scope == "all":
        if all_max_participants is not None:
            return valid[: max(0, int(all_max_participants))], note
        return list(valid), note
    raise ValueError(f"Unknown participant_scope for T-PICS source: {participant_scope!r}")
