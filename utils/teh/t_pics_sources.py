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
DEFAULT_T_PICS_RUN_CONFIG = (
    _REPO_ROOT / "analysis/config/misc/Sep17_T-PICS/config_T-PICS.yaml"
)
_AUTO_SOURCE_TOKENS = frozenset({"", "auto", "__auto__"})
_STEP1_META_KEYS = frozenset({"protocol", "limited_train_val", "kind"})

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


def _resolve_config_path(config_path: Optional[Path], default: Path) -> Path:
    path = Path(config_path) if config_path is not None else default
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _read_yaml_mapping(path: Path) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            f"PyYAML is required to load T-PICS config ({path}). pip install pyyaml"
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"T-PICS config is not a mapping: {path}")
    return payload


def _repo_path(value: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def load_t_pics_official_sources(
    config_path: Optional[Path] = None,
    *,
    _seen: Optional[frozenset] = None,
) -> Dict[str, str]:
    """Load target → source aliases from a transfer_source yaml or a run config.

    A run config may set ``sources:`` directly, or ``source_map:`` pointing at
    ``t_pics_score_weighted_temp_fix.yaml``.
    """
    path = _resolve_config_path(config_path, DEFAULT_T_PICS_SOURCE_CONFIG)
    if not path.is_file():
        return dict(_FALLBACK_T_PICS_SOURCES)
    payload = _read_yaml_mapping(path)
    raw = payload.get("sources")
    if isinstance(raw, dict) and raw:
        return _normalize_source_map({str(k): str(v) for k, v in raw.items()})
    nested = payload.get("source_map")
    if nested:
        nested_path = _repo_path(str(nested)).resolve()
        seen = _seen or frozenset()
        resolved = path.resolve()
        if nested_path in seen or nested_path == resolved:
            raise ValueError(f"T-PICS source_map cycle at {path}")
        return load_t_pics_official_sources(
            nested_path, _seen=seen | {resolved}
        )
    raise ValueError(f"T-PICS source config has no sources mapping: {path}")


def load_t_pics_step1_source_pops(
    config_path: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    """Load Step 1 source-pop job dirs / rank-1 paths from a T-PICS run config."""
    path = _resolve_config_path(config_path, DEFAULT_T_PICS_RUN_CONFIG)
    if not path.is_file():
        raise FileNotFoundError(f"T-PICS run config not found: {path}")
    payload = _read_yaml_mapping(path)
    step1 = payload.get("step1") or payload.get("step1_source_pops") or {}
    if not isinstance(step1, dict) or not step1:
        raise ValueError(f"T-PICS run config has no step1.source_pops: {path}")
    raw_pops = step1.get("source_pops")
    if not isinstance(raw_pops, dict) or not raw_pops:
        raw_pops = {
            k: v
            for k, v in step1.items()
            if str(k) not in _STEP1_META_KEYS and isinstance(v, dict)
        }
    if not raw_pops:
        raise ValueError(f"T-PICS run config has no step1.source_pops: {path}")
    out: Dict[str, Dict[str, str]] = {}
    for alias, entry in raw_pops.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Step 1 source pop {alias!r} is not a mapping in {path}")
        source = normalize_limited_dataset_alias(str(alias))
        run_dir = str(entry.get("run_dir") or "").strip()
        best = str(entry.get("best_program") or "").strip()
        job_id = str(entry.get("job_id") or "").strip()
        if not best and run_dir:
            best = str(Path(run_dir) / "global_phase" / "best_program.py")
        if not job_id or not best:
            raise ValueError(
                f"Step 1 source pop {source!r} needs job_id and best_program in {path}"
            )
        out[source] = {
            "job_id": job_id,
            "run_dir": run_dir,
            "best_program": best,
        }
    return out


def has_t_pics_step1_source_pops(config_path: Optional[Path] = None) -> bool:
    try:
        return bool(load_t_pics_step1_source_pops(config_path))
    except (OSError, ValueError):
        return False


def resolve_t_pics_reuse_source(
    target_dataset: str,
    config_path: Optional[Path] = None,
) -> Tuple[str, Path, str]:
    """Official source alias, absolute rank-1 path, and Step 1 job id."""
    source_cfg = (
        _resolve_config_path(config_path, DEFAULT_T_PICS_SOURCE_CONFIG)
        if config_path is not None
        else DEFAULT_T_PICS_SOURCE_CONFIG
    )
    pop_cfg = (
        _resolve_config_path(config_path, DEFAULT_T_PICS_RUN_CONFIG)
        if config_path is not None
        else DEFAULT_T_PICS_RUN_CONFIG
    )
    source = official_t_pics_source(target_dataset, config_path=source_cfg)
    pops = load_t_pics_step1_source_pops(pop_cfg)
    entry = pops.get(source)
    if entry is None:
        raise KeyError(
            f"No Step 1 source-pop path for {source!r} in {pop_cfg} "
            f"(known: {sorted(pops)})"
        )
    return source, _repo_path(entry["best_program"]), entry["job_id"]


def t_pics_step1_best_program(
    source_dataset: str,
    config_path: Optional[Path] = None,
) -> Path:
    """Absolute rank-1 path for a Step 1 source dataset (not a target lookup)."""
    alias = normalize_t_pics_dataset(source_dataset)
    pops = load_t_pics_step1_source_pops(
        config_path if config_path is not None else DEFAULT_T_PICS_RUN_CONFIG
    )
    entry = pops.get(alias)
    if entry is None:
        raise KeyError(
            f"No Step 1 source-pop path for {alias!r} (known: {sorted(pops)})"
        )
    return _repo_path(entry["best_program"])


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


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="utils.teh.t_pics_sources")
    parser.add_argument("command", choices=("reuse", "step1-best"))
    parser.add_argument("dataset")
    parser.add_argument("config", nargs="?", default=None)
    args = parser.parse_args(argv)
    cfg = Path(args.config) if args.config else None
    if args.command == "reuse":
        src, best, job_id = resolve_t_pics_reuse_source(args.dataset, cfg)
        print(src)
        print(best)
        print(job_id)
        return 0
    print(t_pics_step1_best_program(args.dataset, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
