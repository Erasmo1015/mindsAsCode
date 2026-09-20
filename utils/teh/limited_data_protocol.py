"""Optional structure-aware limited-data protocol shared by PICS, LM, and PT.

CLI default is ``structure_aware_v3`` with train+val budget 40 (ICLR PICS v3).
Pass ``--limited_data_protocol off`` for the legacy TEH split and
``apply_max_observed_trials`` (random train+val cap, test untouched).

``--limited_data_protocol structure_aware`` (v1) with budget 40 means at most 40
combined train+validation observations per participant. Test is reserved first
and is never counted toward the 40. v1 also rebuilds continuous-session test
histories from the retained suffix and may retain 41 Kool trials.

``--limited_data_protocol structure_aware_v3`` is the final ICLR PICS v3 protocol
(same SA40/history semantics as preliminary ``structure_aware_v2``).
``--limited_data_protocol structure_aware_v2`` remains the preliminary T-PICS v2 protocol:
the same train+val cap, original test histories, and a hard Kool cap of 40.

Speekenbrink uses a chronological session split by default (full data and SA40).
Pass ``--speekenbrink_split legacy`` for the old shuffled pseudo-blocks.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_modules.mixed_gambles import (
    DEFAULT_CSV_PATH,
    load_mixed_gambles_trials,
    three_way_unit_counts,
)
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    experiment_to_trial_dicts,
    get_psych101_binary_experiment,
    split_psych_experiment,
)
from utils.teh.limited_data_registry import (
    CONTINUOUS_SESSION,
    INDEPENDENT_TRIAL,
    LIMITED_DATA_PROTOCOL_OFF,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3,
    LIMITED_DATA_PROTOCOLS,
    RESETTING_UNIT,
    is_structure_aware_protocol,
    limited_data_protocol_revision,
    limited_data_spec,
    normalize_limited_data_protocol,
    normalize_limited_dataset_alias,
    uses_training_only_sa40,
)
from utils.teh.sparse_observations import (
    SparseObservationAudit,
    allocate_train_val_counts,
    apply_max_observed_trials,
    normalize_max_observed_trials,
    subset_fingerprint as sparse_subset_fingerprint,
)
from utils.teh.teh_datasets import (
    external_default_data_dir,
    is_external_dataset,
    is_mixed_gambles_dataset,
    is_psych101_dataset,
)

from data_modules.external import load_external_loglik_trials

LIMITED_DATA_MANIFEST_FILENAME = "limited_data_manifest.json"
LIMITED_DATA_MANIFEST_JSONL_FILENAME = "limited_data_manifest.jsonl"
LIMITED_DATA_MANIFEST_CSV_FILENAME = "limited_data_manifest.csv"

SPEEKENBRINK_ALIAS = "5speekenbrink2008learning"
KOOL_ALIAS = "14kool2016when"
SPEEKENBRINK_SPLIT_CHRONOLOGICAL = "chronological"
SPEEKENBRINK_SPLIT_LEGACY = "legacy"
SPEEKENBRINK_SPLITS = (
    SPEEKENBRINK_SPLIT_CHRONOLOGICAL,
    SPEEKENBRINK_SPLIT_LEGACY,
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_speekenbrink_split(value: object) -> str:
    raw = str(value if value is not None else SPEEKENBRINK_SPLIT_CHRONOLOGICAL).strip().lower()
    if raw in {"", "default", "chrono", "chronological"}:
        return SPEEKENBRINK_SPLIT_CHRONOLOGICAL
    if raw in {"legacy", "pseudo", "pseudo_block", "emnlp"}:
        return SPEEKENBRINK_SPLIT_LEGACY
    raise ValueError(
        f"Unknown --speekenbrink_split {value!r}; expected one of {SPEEKENBRINK_SPLITS}"
    )


def use_speekenbrink_chronological_split(speekenbrink_split: object = None) -> bool:
    return normalize_speekenbrink_split(speekenbrink_split) == SPEEKENBRINK_SPLIT_CHRONOLOGICAL


@dataclass
class LimitedDataManifest:
    dataset: str
    participant_id: int
    protocol: str
    split_seed: int
    split_ratio: float
    budget: Optional[int]
    structural_category: str
    unit_type: str
    split_kind: str
    original_n_train: int
    original_n_val: int
    original_n_test: int
    retained_n_train: int
    retained_n_val: int
    retained_n_test: int
    retained_unit_ids_train: Tuple[str, ...] = ()
    retained_unit_ids_val: Tuple[str, ...] = ()
    retained_unit_ids_test: Tuple[str, ...] = ()
    retained_train_indices: Tuple[int, ...] = ()
    retained_val_indices: Tuple[int, ...] = ()
    retained_test_indices: Tuple[int, ...] = ()
    retained_train_chrono: Tuple[int, ...] = ()
    retained_val_chrono: Tuple[int, ...] = ()
    retained_test_chrono: Tuple[int, ...] = ()
    used_partial_unit: bool = False
    fallback_reason: str = ""
    history_consistency: str = "not_checked"
    test_set_differs_from_legacy: bool = False
    test_diff_reason: str = ""
    subset_fingerprint: str = ""
    train_fingerprint: str = ""
    val_fingerprint: str = ""
    test_fingerprint: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    def csv_row(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "participant_id": int(self.participant_id),
            "protocol": self.protocol,
            "structural_category": self.structural_category,
            "unit_type": self.unit_type,
            "split_kind": self.split_kind,
            "split_seed": int(self.split_seed),
            "budget": "" if self.budget is None else int(self.budget),
            "original_n_train": self.original_n_train,
            "original_n_val": self.original_n_val,
            "original_n_test": self.original_n_test,
            "retained_n_train": self.retained_n_train,
            "retained_n_val": self.retained_n_val,
            "retained_n_test": self.retained_n_test,
            "retained_n_observed": self.retained_n_train + self.retained_n_val,
            "used_partial_unit": int(self.used_partial_unit),
            "fallback_reason": self.fallback_reason,
            "history_consistency": self.history_consistency,
            "test_set_differs_from_legacy": int(self.test_set_differs_from_legacy),
            "test_diff_reason": self.test_diff_reason,
            "subset_fingerprint": self.subset_fingerprint,
            "train_fingerprint": self.train_fingerprint,
            "val_fingerprint": self.val_fingerprint,
            "test_fingerprint": self.test_fingerprint,
            "retained_unit_ids_train": ",".join(self.retained_unit_ids_train),
            "retained_unit_ids_val": ",".join(self.retained_unit_ids_val),
            "retained_unit_ids_test": ",".join(self.retained_unit_ids_test),
        }


def add_limited_data_cli_arguments(parser: Any) -> None:
    parser.add_argument(
        "--limited_data_protocol",
        type=str,
        default=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3,
        metavar="NAME",
        help=(
            "Observation-selection protocol after the train/val/test split. "
            "Default 'structure_aware_v3' is final ICLR PICS v3: hard cap 40, "
            "original continuous/resetting test histories, independent history=[], "
            "train+val histories rebuilt only from retained observations. "
            "'structure_aware_v2' is preliminary T-PICS v2 (same SA40 semantics). "
            "'structure_aware' is frozen v1 (may rebuild continuous test histories; "
            "Kool may retain 41). 'off' keeps the legacy random train+val cap via "
            "--max_observed_trials_per_participant. Speekenbrink's session split is "
            "chronological by default (see --speekenbrink_split), independent of "
            "this protocol."
        ),
    )
    parser.add_argument(
        "--speekenbrink_split",
        type=str,
        default=SPEEKENBRINK_SPLIT_CHRONOLOGICAL,
        choices=list(SPEEKENBRINK_SPLITS),
        help=(
            "Speekenbrink train/val/test split. 'chronological' (default) is a "
            "contiguous session suffix for test, used for full data and SA40. "
            "'legacy' restores shuffled TEH pseudo-blocks."
        ),
    )
    parser.add_argument(
        "--limited_train_val",
        type=int,
        default=40,
        metavar="N",
        help=(
            "Under --limited_data_protocol structure_aware, structure_aware_v2, or "
            "structure_aware_v3, "
            "keep at most N combined train+validation observations per participant "
            "(test is reserved first and never counted). Default 40. Ignored when "
            "protocol is 'off'. If omitted under a structure-aware protocol, "
            "--max_observed_trials_per_participant is used as the same budget."
        ),
    )


def resolve_limited_data_budget(
    *,
    protocol: object,
    max_observed_trials_per_participant: Optional[int],
    limited_train_val: Optional[int],
) -> Tuple[str, Optional[int]]:
    proto = normalize_limited_data_protocol(protocol)
    n_max = normalize_max_observed_trials(max_observed_trials_per_participant)
    n_tv = normalize_max_observed_trials(limited_train_val)
    if proto == LIMITED_DATA_PROTOCOL_OFF:
        # CLI default --limited_train_val 40 is ignored for full-data reruns.
        return proto, n_max
    if not is_structure_aware_protocol(proto):
        raise ValueError(f"unsupported limited_data_protocol {proto!r}")
    if n_tv is not None and n_max is not None and n_tv != n_max:
        raise ValueError(
            "--limited_train_val and --max_observed_trials_per_participant both set "
            f"and differ ({n_tv} vs {n_max})"
        )
    budget = n_tv if n_tv is not None else n_max
    if budget is None:
        budget = 40
    return proto, budget


def _json_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _trial_identity_payload(trial: Dict[str, Any]) -> Dict[str, Any]:
    meta = trial.get("_ldp") or {}
    return {
        "unit_id": meta.get("unit_id"),
        "chrono": meta.get("chrono"),
        "session_index": meta.get("session_index"),
        "action": trial.get("action"),
        "problem": trial.get("problem"),
    }


def split_fingerprint(trials: Sequence[Dict[str, Any]]) -> str:
    return _json_hash([_trial_identity_payload(t) for t in trials])


def structure_aware_subset_fingerprint(
    *,
    dataset: str,
    participant_id: int,
    split_seed: int,
    budget: Optional[int],
    protocol: str,
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    test_trials: Sequence[Dict[str, Any]],
) -> str:
    payload = {
        "protocol": str(protocol),
        "dataset": str(dataset),
        "participant_id": int(participant_id),
        "split_seed": int(split_seed),
        "budget": None if budget is None else int(budget),
        "train": [_trial_identity_payload(t) for t in train_trials],
        "val": [_trial_identity_payload(t) for t in val_trials],
        "test": [_trial_identity_payload(t) for t in test_trials],
    }
    return _json_hash(payload)


def _history_len(trial: Dict[str, Any]) -> int:
    hist = trial.get("history")
    return len(hist) if isinstance(hist, list) else 0


def _unit_id_from_trial(spec, trial: Dict[str, Any], fallback: int) -> str:
    problem = trial.get("problem") or {}
    if spec.category == CONTINUOUS_SESSION:
        if spec.unit_type == "day" and "presented_day" in problem:
            return f"day:{problem['presented_day']}"
        return "session"
    if spec.dataset == "guan_2020_stopping":
        cond = problem.get("condition_index")
        pid = problem.get("problem_id")
        if cond is not None or pid is not None:
            return f"stopping_problem:{cond}:{pid}"
    for key in spec.unit_keys:
        if key in problem and problem[key] is not None:
            return f"{spec.unit_type}:{problem[key]}"
    if "gamble_A" in problem or "gamble_B" in problem:
        sig = json.dumps(
            {"A": problem.get("gamble_A"), "B": problem.get("gamble_B")},
            sort_keys=True,
            default=str,
        )
        return f"problem:{sig}"
    if "problem_id" in problem:
        return f"problem:{problem['problem_id']}"
    return f"{spec.unit_type}:{fallback}"


def _tag_trial(
    trial: Dict[str, Any],
    *,
    unit_id: str,
    chrono: int,
    session_index: int,
    origin_split: str,
    origin_index: int,
) -> Dict[str, Any]:
    tagged = copy.deepcopy(trial)
    tagged["_ldp"] = {
        "unit_id": str(unit_id),
        "chrono": int(chrono),
        "session_index": int(session_index),
        "origin_split": origin_split,
        "origin_index": int(origin_index),
    }
    return tagged


def _strip_ldp(trial: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(trial)
    out.pop("_ldp", None)
    return out


def _strip_ldp_list(trials: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_strip_ldp(t) for t in trials]


def tag_split_trials(
    dataset: str,
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    test_trials: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    spec = limited_data_spec(dataset)
    tagged_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    session_index = 0
    unit_fallback = 0
    chrono_by_unit: Dict[str, int] = {}
    for split_name, src in (
        ("train", train_trials),
        ("val", val_trials),
        ("test", test_trials),
    ):
        prev_unit = None
        for i, trial in enumerate(src):
            if spec.category == INDEPENDENT_TRIAL:
                unit_id = f"trial:{split_name}:{i}"
                chrono = 0
            elif spec.category == CONTINUOUS_SESSION:
                unit_id = _unit_id_from_trial(spec, trial, 0)
                if spec.unit_type == "day":
                    chrono = chrono_by_unit.get(unit_id, 0)
                    chrono_by_unit[unit_id] = chrono + 1
                else:
                    chrono = session_index
            else:
                if _history_len(trial) == 0 or prev_unit is None:
                    unit_id = _unit_id_from_trial(spec, trial, unit_fallback)
                    unit_fallback += 1
                    chrono = 0
                    prev_unit = unit_id
                else:
                    unit_id = prev_unit
                    chrono = chrono_by_unit.get(unit_id, 0)
            chrono_by_unit[unit_id] = chrono + 1
            tagged_splits[split_name].append(
                _tag_trial(
                    trial,
                    unit_id=unit_id,
                    chrono=chrono,
                    session_index=session_index,
                    origin_split=split_name,
                    origin_index=i,
                )
            )
            session_index += 1
    return tagged_splits["train"], tagged_splits["val"], tagged_splits["test"]


def group_resetting_units(trials: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    units: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    for trial in trials:
        uid = str((trial.get("_ldp") or {}).get("unit_id"))
        if not current:
            current = [trial]
            current_id = uid
            continue
        if uid != current_id:
            units.append(current)
            current = [trial]
            current_id = uid
        else:
            current.append(trial)
    if current:
        units.append(current)
    return units


def select_complete_then_prefix(
    units: Sequence[Sequence[Dict[str, Any]]],
    budget: int,
    *,
    prefix_valid: bool,
) -> Tuple[List[Dict[str, Any]], bool, str]:
    """Take complete units, then a chronological prefix of the next unit."""
    retained: List[Dict[str, Any]] = []
    used_partial = False
    reason = ""
    if budget <= 0:
        return retained, False, "zero_budget"
    for unit in units:
        remaining = int(budget) - len(retained)
        if remaining <= 0:
            break
        unit_list = list(unit)
        if len(unit_list) <= remaining:
            retained.extend(unit_list)
            continue
        if prefix_valid and remaining > 0:
            retained.extend(unit_list[:remaining])
            used_partial = True
            reason = "chronological_prefix"
        else:
            reason = "prefix_invalid_stop_below_budget"
        break
    if not retained and units and not prefix_valid:
        reason = reason or "prefix_invalid_stop_below_budget"
    return retained, used_partial, reason


def _entries_produced(
    prev: Dict[str, Any], nxt: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    prev_hist = prev.get("history") if isinstance(prev.get("history"), list) else []
    if nxt is not None and isinstance(nxt.get("history"), list):
        nxt_hist = nxt["history"]
        if len(nxt_hist) > len(prev_hist):
            return [copy.deepcopy(item) for item in nxt_hist[len(prev_hist) :]]
        if len(nxt_hist) == len(prev_hist) + 1:
            return [copy.deepcopy(nxt_hist[-1])]
    entry: Dict[str, Any] = {"action": prev.get("action")}
    return [entry]


def _entry_fingerprint(entry: Any) -> str:
    return json.dumps(entry, sort_keys=True, default=str, separators=(",", ":"))


def rebuild_histories_in_scopes(
    scopes: Sequence[Sequence[Dict[str, Any]]],
    original_scopes: Sequence[Sequence[Dict[str, Any]]],
) -> Tuple[List[List[Dict[str, Any]]], str]:
    """Rebuild history from retained earlier observations in each scope."""
    rebuilt_scopes: List[List[Dict[str, Any]]] = []
    errors: List[str] = []
    for scope, original in zip(scopes, original_scopes):
        produced_by_session: Dict[int, List[Dict[str, Any]]] = {}
        orig_list = list(original)
        for i, trial in enumerate(orig_list):
            meta = trial.get("_ldp") or {}
            nxt = orig_list[i + 1] if i + 1 < len(orig_list) else None
            produced_by_session[int(meta.get("session_index", i))] = _entries_produced(
                trial, nxt
            )
        retained_session = {
            int((t.get("_ldp") or {}).get("session_index", -1)) for t in scope
        }
        history: List[Dict[str, Any]] = []
        history_sources: List[int] = []
        out: List[Dict[str, Any]] = []
        for trial in scope:
            new_trial = copy.deepcopy(trial)
            new_trial["history"] = [copy.deepcopy(h) for h in history]
            bad = [sid for sid in history_sources if sid not in retained_session]
            if bad:
                errors.append(
                    f"history source session {bad[:4]} not retained in unit "
                    f"{(trial.get('_ldp') or {}).get('unit_id')}"
                )
            out.append(new_trial)
            sid = int((trial.get("_ldp") or {}).get("session_index", -1))
            produced = produced_by_session.get(sid, [{"action": trial.get("action")}])
            history.extend(copy.deepcopy(x) for x in produced)
            history_sources.extend([sid] * len(produced))
        rebuilt_scopes.append(out)
    status = "pass" if not errors else "fail: " + "; ".join(errors[:8])
    return rebuilt_scopes, status


def rebuild_resetting_split(
    retained: Sequence[Dict[str, Any]],
    original: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    orig_by_unit: Dict[str, List[Dict[str, Any]]] = {}
    for trial in original:
        uid = str((trial.get("_ldp") or {}).get("unit_id"))
        orig_by_unit.setdefault(uid, []).append(trial)
    out: List[Dict[str, Any]] = []
    status = "pass"
    for unit in group_resetting_units(retained):
        uid = str((unit[0].get("_ldp") or {}).get("unit_id"))
        rebuilt, unit_status = rebuild_histories_in_scopes(
            [unit], [orig_by_unit.get(uid, unit)]
        )
        out.extend(rebuilt[0])
        if unit_status != "pass":
            status = unit_status
    return out, status


def assert_no_split_overlap(
    train: Sequence[Dict[str, Any]],
    val: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
) -> None:
    def keys(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Any, Any, Any]]:
        out = []
        for t in rows:
            meta = t.get("_ldp") or {}
            out.append(
                (meta.get("unit_id"), meta.get("chrono"), meta.get("session_index"))
            )
        return out

    seen = set()
    for name, rows in (("train", train), ("val", val), ("test", test)):
        for key in keys(rows):
            if key in seen:
                raise AssertionError(f"{name} overlaps another split at {key}")
            seen.add(key)


def _history_contains_omitted(
    trials: Sequence[Dict[str, Any]], omitted_fps: Sequence[str]
) -> bool:
    omitted = set(omitted_fps)
    for trial in trials:
        for item in trial.get("history") or []:
            if _entry_fingerprint(item) in omitted:
                return True
    return False


def _kool_is_stage2(trial: Dict[str, Any]) -> bool:
    problem = trial.get("problem") or {}
    return int(problem.get("stage", 1)) == 2


def _contiguous_tv_suffix(
    tv_trials: Sequence[Dict[str, Any]],
    budget: int,
    *,
    kool: bool,
    exact_40: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    tv = list(tv_trials)
    if budget <= 0 or len(tv) <= budget:
        reason = "insufficient_train_val" if len(tv) < budget else ""
        return tv, reason
    start = len(tv) - int(budget)
    reason = ""
    if kool and (not exact_40) and start < len(tv) and _kool_is_stage2(tv[start]):
        if start > 0:
            start -= 1
            reason = "kool_include_matching_stage1"
        else:
            reason = "kool_stage2_at_session_start"
    return tv[start:], reason


def _split_contiguous_segment(
    segment: Sequence[Dict[str, Any]],
    n_train_keep: int,
    *,
    kool: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    rows = list(segment)
    n_train_keep = max(0, min(int(n_train_keep), len(rows)))
    extra = ""
    if not kool:
        return rows[:n_train_keep], rows[n_train_keep:], extra
    days: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_id = None
    for trial in rows:
        uid = (trial.get("_ldp") or {}).get("unit_id")
        if current and uid != current_id:
            days.append(current)
            current = [trial]
            current_id = uid
        else:
            current.append(trial)
            current_id = uid
    if current:
        days.append(current)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for day in days:
        if len(train) < n_train_keep:
            train.extend(day)
            if len(train) > n_train_keep:
                extra = "kool_day_boundary_overshoot_train"
        else:
            val.extend(day)
    if not val and train and len(days) >= 2:
        last_id = (train[-1].get("_ldp") or {}).get("unit_id")
        moved = []
        kept = []
        for trial in train:
            if (trial.get("_ldp") or {}).get("unit_id") == last_id:
                moved.append(trial)
            else:
                kept.append(trial)
        if kept and moved:
            train, val = kept, moved
            extra = (extra + ";" if extra else "") + "kool_moved_last_day_to_val"
    return train, val, extra


def split_continuous_session_chronological(
    trials: Sequence[Dict[str, Any]],
    split_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(trials)
    n_train, n_val, n_test = three_way_unit_counts(len(rows), split_ratio)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    if len(test) != n_test:
        raise AssertionError("chronological split test count mismatch")
    return train, val, test


def load_raw_participant_splits(
    dataset: str,
    participant_id: int,
    *,
    split_ratio: float,
    split_seed: int,
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    limited_data_protocol: object = LIMITED_DATA_PROTOCOL_OFF,
    speekenbrink_split: object = SPEEKENBRINK_SPLIT_CHRONOLOGICAL,
    data_dir: Optional[str] = None,
    filtered_split: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Load the pre-cap split. Speekenbrink is chronological by default."""
    alias = normalize_limited_dataset_alias(dataset)
    split_kind = "legacy_unit_shuffle"
    if is_mixed_gambles_dataset(alias):
        train, val, test, _ = load_mixed_gambles_trials(
            int(participant_id),
            csv_path=mixed_gambles_csv or DEFAULT_CSV_PATH,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        split_kind = "legacy_signature_shuffle"
    elif is_external_dataset(alias):
        resolved = data_dir or str(_REPO_ROOT / external_default_data_dir(alias))
        train, val, test, _ = load_external_loglik_trials(
            alias,
            int(participant_id),
            data_dir=resolved,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        split_kind = "legacy_unit_shuffle"
    elif is_psych101_dataset(alias):
        exp = get_psych101_binary_experiment(
            alias,
            int(participant_id),
            split=psych_dataset_split,
            local_dataset=local_dataset,
            filtered_split=filtered_split,
        )
        if alias == SPEEKENBRINK_ALIAS and use_speekenbrink_chronological_split(
            speekenbrink_split
        ):
            all_trials = experiment_to_trial_dicts(exp)
            train, val, test = split_continuous_session_chronological(
                all_trials, split_ratio
            )
            split_kind = "chronological_session"
        else:
            train, val, test, _ = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            if alias == KOOL_ALIAS:
                split_kind = "legacy_chronological_days"
            elif alias == SPEEKENBRINK_ALIAS:
                split_kind = "legacy_pseudo_block_shuffle"
            else:
                split_kind = "legacy_unit_shuffle"
    else:
        raise ValueError(f"Unsupported TEH dataset for limited-data split: {dataset!r}")
    return list(train), list(val), list(test), split_kind


def _legacy_result(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    audit: SparseObservationAudit,
    *,
    dataset: str,
    split_kind: str,
    split_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SparseObservationAudit, LimitedDataManifest]:
    spec = limited_data_spec(dataset)
    manifest = LimitedDataManifest(
        dataset=str(dataset),
        participant_id=int(audit.participant_id),
        protocol=LIMITED_DATA_PROTOCOL_OFF,
        split_seed=int(audit.split_seed),
        split_ratio=float(split_ratio),
        budget=audit.max_observed_trials_per_participant,
        structural_category=spec.category,
        unit_type=spec.unit_type,
        split_kind=split_kind,
        original_n_train=audit.original_n_train,
        original_n_val=audit.original_n_val,
        original_n_test=audit.original_n_test,
        retained_n_train=audit.retained_n_train,
        retained_n_val=audit.retained_n_val,
        retained_n_test=audit.retained_n_test,
        retained_train_indices=audit.selected_train_indices,
        retained_val_indices=audit.selected_val_indices,
        retained_test_indices=tuple(range(audit.retained_n_test)),
        used_partial_unit=False,
        fallback_reason="",
        history_consistency="legacy_not_rebuilt",
        test_set_differs_from_legacy=(
            spec.dataset == SPEEKENBRINK_ALIAS and "chronological" in str(split_kind)
        ),
        test_diff_reason=(
            "speekenbrink_chronological_split_replaces_legacy_pseudo_blocks"
            if spec.dataset == SPEEKENBRINK_ALIAS and "chronological" in str(split_kind)
            else ""
        ),
        subset_fingerprint=audit.subset_fingerprint,
        train_fingerprint=sparse_subset_fingerprint(
            dataset=dataset,
            participant_id=audit.participant_id,
            split_seed=audit.split_seed,
            budget=audit.max_observed_trials_per_participant,
            selected_train_indices=audit.selected_train_indices,
            selected_val_indices=(),
        ),
        val_fingerprint=sparse_subset_fingerprint(
            dataset=dataset,
            participant_id=audit.participant_id,
            split_seed=audit.split_seed,
            budget=audit.max_observed_trials_per_participant,
            selected_train_indices=(),
            selected_val_indices=audit.selected_val_indices,
        ),
        test_fingerprint=_json_hash({"n_test": audit.retained_n_test}),
    )
    return train, val, test, audit, manifest


def apply_structure_aware_protocol(
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    test_trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    participant_id: int,
    split_seed: int,
    split_ratio: float,
    budget: int,
    split_kind: str,
    revision: str = "v1",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SparseObservationAudit, LimitedDataManifest]:
    spec = limited_data_spec(dataset)
    alias = spec.dataset
    rev = str(revision).strip().lower()
    # v2 and v3 share the training-only SA40 path (exact-40, original test histories).
    is_v2 = rev in ("v2", "v3")
    if rev == "v3":
        protocol_name = LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3
    elif rev == "v2":
        protocol_name = LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2
    else:
        protocol_name = LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE
    tagged_train, tagged_val, tagged_test = tag_split_trials(
        alias, train_trials, val_trials, test_trials
    )
    original_n_train = len(tagged_train)
    original_n_val = len(tagged_val)
    original_n_test = len(tagged_test)
    fallback = ""
    used_partial = False
    test_differs = alias == SPEEKENBRINK_ALIAS and "chronological" in split_kind
    test_diff_reason = (
        "speekenbrink_chronological_split_replaces_legacy_pseudo_blocks"
        if test_differs
        else ""
    )

    if spec.category == INDEPENDENT_TRIAL:
        sparse_train, sparse_val, sparse_test, sparse_audit = apply_max_observed_trials(
            tagged_train,
            tagged_val,
            tagged_test,
            max_observed_trials_per_participant=budget,
            dataset=alias,
            participant_id=int(participant_id),
            split_seed=int(split_seed),
        )
        new_train, new_val, new_test = sparse_train, sparse_val, sparse_test
        if original_n_train + original_n_val < budget:
            fallback = "insufficient_train_val"
        new_train, new_val, new_test = sparse_train, sparse_val, sparse_test
        # Independent-trial semantics: each decision is a new unit. Loader-
        # accumulated cross-trial history is an artifact and must not appear in
        # train, val, or test (v1 and v2). Continuous/resetting tasks are
        # unchanged below.
        for row in new_train + new_val + new_test:
            row["history"] = []
        hist_status = "independent_empty_history"
        train_idx = sparse_audit.selected_train_indices
        val_idx = sparse_audit.selected_val_indices
    elif spec.category == RESETTING_UNIT:
        t_keep, v_keep = allocate_train_val_counts(
            original_n_train, original_n_val, int(budget)
        )
        train_units = group_resetting_units(tagged_train)
        val_units = group_resetting_units(tagged_val)
        new_train, part_t, reason_t = select_complete_then_prefix(
            train_units, t_keep, prefix_valid=spec.prefix_valid
        )
        new_val, part_v, reason_v = select_complete_then_prefix(
            val_units, v_keep, prefix_valid=spec.prefix_valid
        )
        used_partial = bool(part_t or part_v)
        reasons = [r for r in (reason_t, reason_v) if r]
        if original_n_train + original_n_val < budget:
            fallback = "insufficient_train_val"
            new_train, new_val = tagged_train, tagged_val
            used_partial = False
        elif reasons:
            fallback = ";".join(reasons)
        new_test = list(tagged_test)
        new_train, hist_t = rebuild_resetting_split(new_train, tagged_train)
        new_val, hist_v = rebuild_resetting_split(new_val, tagged_val)
        if is_v2:
            for tagged, orig in zip(new_test, test_trials):
                tagged["history"] = copy.deepcopy(orig.get("history") or [])
            hist_te = "original_test_histories_preserved"
        else:
            new_test, hist_te = rebuild_resetting_split(new_test, tagged_test)
        hist_status = "pass"
        for part in (hist_t, hist_v, hist_te):
            if part != "pass":
                hist_status = part
                break
        train_idx = tuple(
            int((t.get("_ldp") or {}).get("origin_index", i))
            for i, t in enumerate(new_train)
        )
        val_idx = tuple(
            int((t.get("_ldp") or {}).get("origin_index", i))
            for i, t in enumerate(new_val)
        )
    else:
        tv = tagged_train + tagged_val
        new_test = list(tagged_test)
        segment, suffix_reason = _contiguous_tv_suffix(
            tv, int(budget), kool=(alias == KOOL_ALIAS), exact_40=is_v2
        )
        if original_n_train + original_n_val < budget:
            fallback = "insufficient_train_val"
        elif suffix_reason:
            fallback = suffix_reason
        t_keep, v_keep = allocate_train_val_counts(
            original_n_train, original_n_val, len(segment)
        )
        new_train, new_val, split_extra = _split_contiguous_segment(
            segment, t_keep, kool=(alias == KOOL_ALIAS)
        )
        if split_extra:
            fallback = (fallback + ";" if fallback else "") + split_extra
        if is_v2:
            rebuilt, hist_status = rebuild_histories_in_scopes(
                [new_train + new_val],
                [tagged_train + tagged_val],
            )
            combined = rebuilt[0]
            n_tr, n_va = len(new_train), len(new_val)
            new_train = combined[:n_tr]
            new_val = combined[n_tr : n_tr + n_va]
            new_test = list(tagged_test)
            for tagged, orig in zip(new_test, test_trials):
                tagged["history"] = copy.deepcopy(orig.get("history") or [])
            hist_status = (
                "original_test_histories_preserved"
                if hist_status == "pass"
                else hist_status
            )
        else:
            rebuilt, hist_status = rebuild_histories_in_scopes(
                [new_train + new_val + new_test],
                [tagged_train + tagged_val + tagged_test],
            )
            combined = rebuilt[0]
            n_tr, n_va = len(new_train), len(new_val)
            new_train = combined[:n_tr]
            new_val = combined[n_tr : n_tr + n_va]
            new_test = combined[n_tr + n_va :]
        train_idx = tuple(
            int((t.get("_ldp") or {}).get("session_index", i)) for i, t in enumerate(new_train)
        )
        val_idx = tuple(
            int((t.get("_ldp") or {}).get("session_index", i)) for i, t in enumerate(new_val)
        )

    assert_no_split_overlap(new_train, new_val, new_test)
    n_obs = len(new_train) + len(new_val)
    if n_obs > int(budget):
        if is_v2:
            raise AssertionError(
                f"{alias} v2 retained train+val={n_obs} exceeds budget {budget}"
            )
        if fallback != "kool_include_matching_stage1" and "overshoot" not in fallback:
            if alias == KOOL_ALIAS and (
                "kool_include_matching_stage1" in fallback
                or "overshoot" in fallback
            ):
                pass
            elif alias == KOOL_ALIAS:
                pass
            else:
                raise AssertionError(
                    f"{alias} retained train+val={n_obs} exceeds budget {budget}"
                )
    if len(new_test) != original_n_test and not test_differs:
        raise AssertionError("structure-aware protocol changed the test count")
    if n_obs > int(budget):
        if "kool" not in fallback:
            fallback = (fallback + ";" if fallback else "") + "budget_overshoot"

    assert_no_split_overlap(new_train, new_val, new_test)
    if hist_status not in (
        "pass",
        "original_test_histories_preserved",
        "independent_empty_history",
    ):
        raise AssertionError(hist_status)

    fp = structure_aware_subset_fingerprint(
        dataset=alias,
        participant_id=int(participant_id),
        split_seed=int(split_seed),
        budget=int(budget),
        protocol=protocol_name,
        train_trials=new_train,
        val_trials=new_val,
        test_trials=new_test,
    )
    train_fp = split_fingerprint(new_train)
    val_fp = split_fingerprint(new_val)
    test_fp = split_fingerprint(new_test)

    def _units(rows: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:
        seen = []
        for t in rows:
            uid = str((t.get("_ldp") or {}).get("unit_id"))
            if uid not in seen:
                seen.append(uid)
        return tuple(seen)

    def _chrono(rows: Sequence[Dict[str, Any]]) -> Tuple[int, ...]:
        return tuple(int((t.get("_ldp") or {}).get("chrono", 0)) for t in rows)

    audit = SparseObservationAudit(
        dataset=str(alias),
        participant_id=int(participant_id),
        split_seed=int(split_seed),
        max_observed_trials_per_participant=int(budget),
        applied=len(new_train) + len(new_val) < original_n_train + original_n_val,
        original_n_train=original_n_train,
        original_n_val=original_n_val,
        original_n_test=original_n_test,
        retained_n_train=len(new_train),
        retained_n_val=len(new_val),
        retained_n_test=len(new_test),
        selected_train_indices=tuple(train_idx),
        selected_val_indices=tuple(val_idx),
        subset_fingerprint=fp,
    )
    manifest = LimitedDataManifest(
        dataset=str(alias),
        participant_id=int(participant_id),
        protocol=protocol_name,
        split_seed=int(split_seed),
        split_ratio=float(split_ratio),
        budget=int(budget),
        structural_category=spec.category,
        unit_type=spec.unit_type,
        split_kind=split_kind,
        original_n_train=original_n_train,
        original_n_val=original_n_val,
        original_n_test=original_n_test,
        retained_n_train=len(new_train),
        retained_n_val=len(new_val),
        retained_n_test=len(new_test),
        retained_unit_ids_train=_units(new_train),
        retained_unit_ids_val=_units(new_val),
        retained_unit_ids_test=_units(new_test),
        retained_train_indices=tuple(train_idx),
        retained_val_indices=tuple(val_idx),
        retained_test_indices=tuple(
            int((t.get("_ldp") or {}).get("origin_index", i))
            for i, t in enumerate(new_test)
        ),
        retained_train_chrono=_chrono(new_train),
        retained_val_chrono=_chrono(new_val),
        retained_test_chrono=_chrono(new_test),
        used_partial_unit=used_partial,
        fallback_reason=fallback,
        history_consistency=hist_status,
        test_set_differs_from_legacy=test_differs,
        test_diff_reason=test_diff_reason,
        subset_fingerprint=fp,
        train_fingerprint=train_fp,
        val_fingerprint=val_fp,
        test_fingerprint=test_fp,
        extra={
            "prefix_valid": spec.prefix_valid,
            "chronological_split": spec.chronological_split,
            "sa40_revision": rev if rev in ("v2", "v3") else "v1",
            "test_history_policy": (
                "independent_empty_all_splits"
                if spec.category == INDEPENDENT_TRIAL
                else (
                    "original_pre_choice"
                    if is_v2
                    else "v1_rebuilt_or_emptied"
                )
            ),
        },
    )
    return (
        _strip_ldp_list(new_train),
        _strip_ldp_list(new_val),
        _strip_ldp_list(new_test),
        audit,
        manifest,
    )


def apply_limited_data_protocol(
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    test_trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    participant_id: int,
    split_seed: int,
    split_ratio: float = 0.6,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: object = LIMITED_DATA_PROTOCOL_OFF,
    limited_train_val: Optional[int] = None,
    split_kind: str = "legacy_unit_shuffle",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SparseObservationAudit, LimitedDataManifest]:
    proto, budget = resolve_limited_data_budget(
        protocol=limited_data_protocol,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_train_val=limited_train_val,
    )
    if proto == LIMITED_DATA_PROTOCOL_OFF:
        train, val, test, audit = apply_max_observed_trials(
            train_trials,
            val_trials,
            test_trials,
            max_observed_trials_per_participant=budget,
            dataset=dataset,
            participant_id=int(participant_id),
            split_seed=int(split_seed),
        )
        return _legacy_result(
            train,
            val,
            test,
            audit,
            dataset=dataset,
            split_kind=split_kind,
            split_ratio=split_ratio,
        )
    return apply_structure_aware_protocol(
        train_trials,
        val_trials,
        test_trials,
        dataset=dataset,
        participant_id=int(participant_id),
        split_seed=int(split_seed),
        split_ratio=float(split_ratio),
        budget=int(budget),
        split_kind=split_kind,
        revision=limited_data_protocol_revision(proto),
    )


def load_participant_limited_splits(
    dataset: str,
    participant_id: int,
    *,
    split_ratio: float,
    split_seed: int,
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: object = LIMITED_DATA_PROTOCOL_OFF,
    limited_train_val: Optional[int] = None,
    speekenbrink_split: object = SPEEKENBRINK_SPLIT_CHRONOLOGICAL,
    data_dir: Optional[str] = None,
    filtered_split: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SparseObservationAudit, LimitedDataManifest]:
    train, val, test, split_kind = load_raw_participant_splits(
        dataset,
        int(participant_id),
        split_ratio=split_ratio,
        split_seed=split_seed,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        limited_data_protocol=limited_data_protocol,
        speekenbrink_split=speekenbrink_split,
        data_dir=data_dir,
        filtered_split=filtered_split,
    )
    return apply_limited_data_protocol(
        train,
        val,
        test,
        dataset=dataset,
        participant_id=int(participant_id),
        split_seed=int(split_seed),
        split_ratio=float(split_ratio),
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        split_kind=split_kind,
    )


def write_limited_data_manifest(path: Path, manifest: LimitedDataManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def append_limited_data_manifest_jsonl(path: Path, manifest: LimitedDataManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest.to_dict(), sort_keys=True) + "\n")


def write_limited_data_manifests_csv(
    path: Path, manifests: Sequence[LimitedDataManifest]
) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "participant_id",
        "protocol",
        "structural_category",
        "unit_type",
        "split_kind",
        "split_seed",
        "budget",
        "original_n_train",
        "original_n_val",
        "original_n_test",
        "retained_n_train",
        "retained_n_val",
        "retained_n_test",
        "retained_n_observed",
        "used_partial_unit",
        "fallback_reason",
        "history_consistency",
        "test_set_differs_from_legacy",
        "test_diff_reason",
        "subset_fingerprint",
        "train_fingerprint",
        "val_fingerprint",
        "test_fingerprint",
        "retained_unit_ids_train",
        "retained_unit_ids_val",
        "retained_unit_ids_test",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for manifest in manifests:
            writer.writerow(manifest.csv_row())
    return path


def rewrite_limited_data_csv_from_jsonl(
    jsonl_path: Path, csv_path: Optional[Path] = None
) -> Optional[Path]:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        return None
    manifests: List[LimitedDataManifest] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        manifests.append(LimitedDataManifest(**_manifest_kwargs_from_dict(json.loads(line))))
    out = Path(csv_path) if csv_path is not None else jsonl_path.with_name(
        LIMITED_DATA_MANIFEST_CSV_FILENAME
    )
    return write_limited_data_manifests_csv(out, manifests)


def _manifest_kwargs_from_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    known = LimitedDataManifest.__dataclass_fields__
    kwargs: Dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            continue
        if key.startswith("retained_") and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return kwargs


def should_persist_limited_data_manifest(manifest: LimitedDataManifest) -> bool:
    return is_structure_aware_protocol(manifest.protocol)
