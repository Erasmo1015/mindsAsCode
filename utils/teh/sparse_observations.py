"""Deterministic post-split train+val observation budgets for sparse-data TEH.

Split first; never touch the test set. Limit the combined train+val count to at
most ``N`` per participant, sampling proportionally from each split.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SPARSE_AUDIT_FILENAME = "sparse_observations.json"
SPARSE_AUDIT_JSONL_FILENAME = "sparse_observations.jsonl"
SPARSE_AUDIT_CSV_FILENAME = "sparse_observations.csv"

_TRAIN_STREAM = 0
_VAL_STREAM = 1


@dataclass(frozen=True)
class SparseObservationAudit:
    """Per-participant record of the original vs retained observation counts."""

    dataset: str
    participant_id: int
    split_seed: int
    max_observed_trials_per_participant: Optional[int]
    applied: bool
    original_n_train: int
    original_n_val: int
    original_n_test: int
    retained_n_train: int
    retained_n_val: int
    retained_n_test: int
    selected_train_indices: Tuple[int, ...]
    selected_val_indices: Tuple[int, ...]
    subset_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["selected_train_indices"] = list(self.selected_train_indices)
        payload["selected_val_indices"] = list(self.selected_val_indices)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SparseObservationAudit":
        budget = payload.get("max_observed_trials_per_participant")
        return cls(
            dataset=str(payload["dataset"]),
            participant_id=int(payload["participant_id"]),
            split_seed=int(payload["split_seed"]),
            max_observed_trials_per_participant=(
                None if budget in (None, "") else int(budget)
            ),
            applied=_as_bool_flag(payload.get("applied", False)),
            original_n_train=int(payload.get("original_n_train", 0)),
            original_n_val=int(payload.get("original_n_val", 0)),
            original_n_test=int(payload.get("original_n_test", 0)),
            retained_n_train=int(payload.get("retained_n_train", 0)),
            retained_n_val=int(payload.get("retained_n_val", 0)),
            retained_n_test=int(payload.get("retained_n_test", 0)),
            selected_train_indices=parse_selected_indices(
                payload.get("selected_train_indices")
            ),
            selected_val_indices=parse_selected_indices(
                payload.get("selected_val_indices")
            ),
            subset_fingerprint=str(payload.get("subset_fingerprint") or ""),
        )

    def csv_row(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "participant_id": int(self.participant_id),
            "split_seed": int(self.split_seed),
            "max_observed_trials_per_participant": (
                ""
                if self.max_observed_trials_per_participant is None
                else int(self.max_observed_trials_per_participant)
            ),
            "applied": int(self.applied),
            "original_n_train": self.original_n_train,
            "original_n_val": self.original_n_val,
            "original_n_test": self.original_n_test,
            "retained_n_train": self.retained_n_train,
            "retained_n_val": self.retained_n_val,
            "retained_n_test": self.retained_n_test,
            "original_n_observed": self.original_n_train + self.original_n_val,
            "retained_n_observed": self.retained_n_train + self.retained_n_val,
            "subset_fingerprint": self.subset_fingerprint,
            "selected_train_indices": ",".join(
                str(i) for i in self.selected_train_indices
            ),
            "selected_val_indices": ",".join(str(i) for i in self.selected_val_indices),
        }


def parse_selected_indices(value: Any) -> Tuple[int, ...]:
    """Parse index lists from JSON (list) or CSV (comma-separated string)."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(i) for i in value)
    if isinstance(value, int) and not isinstance(value, bool):
        return (int(value),)
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        parsed = json.loads(text)
        return tuple(int(i) for i in parsed)
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def _as_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "none"}:
        return False
    if text in {"1", "true", "yes"}:
        return True
    raise ValueError(f"Cannot parse boolean flag {value!r}")


def normalize_max_observed_trials(value: Optional[int]) -> Optional[int]:
    """Return a positive budget, or None when the cap is disabled."""
    if value is None:
        return None
    n = int(value)
    if n <= 0:
        return None
    return n


def sparse_budget_for_dataset(
    *,
    dataset_alias: str,
    config_key: Optional[str] = None,
    max_observed_trials_per_participant: Optional[int],
    max_observed_trials_datasets: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """Resolve the per-dataset budget (None = full data).

    When ``max_observed_trials_datasets`` is empty/None and a budget is set, the
    cap applies to every dataset. Otherwise only listed aliases/config keys.
    """
    budget = normalize_max_observed_trials(max_observed_trials_per_participant)
    if budget is None:
        return None
    if not max_observed_trials_datasets:
        return budget
    wanted = {str(x).strip() for x in max_observed_trials_datasets if str(x).strip()}
    if not wanted:
        return budget
    alias = str(dataset_alias).strip()
    key = str(config_key or "").strip()
    if alias in wanted or key in wanted:
        return budget
    return None


def dataset_seed_uint32(dataset: str) -> int:
    digest = hashlib.sha256(str(dataset).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def subset_fingerprint(
    *,
    dataset: str,
    participant_id: int,
    split_seed: int,
    budget: Optional[int],
    selected_train_indices: Sequence[int],
    selected_val_indices: Sequence[int],
) -> str:
    payload = {
        "dataset": str(dataset),
        "participant_id": int(participant_id),
        "split_seed": int(split_seed),
        "budget": None if budget is None else int(budget),
        "selected_train_indices": [int(i) for i in selected_train_indices],
        "selected_val_indices": [int(i) for i in selected_val_indices],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def allocate_train_val_counts(n_train: int, n_val: int, budget: int) -> Tuple[int, int]:
    """Choose how many train vs val items to keep for a combined budget ``N``.

    Keeps at least one item from each non-empty split when ``N >= 2``.
    """
    n_train = max(0, int(n_train))
    n_val = max(0, int(n_val))
    available = n_train + n_val
    if budget <= 0 or available <= budget:
        return n_train, n_val
    n_keep = int(budget)

    if n_train == 0:
        return 0, min(n_keep, n_val)
    if n_val == 0:
        return min(n_keep, n_train), 0
    if n_keep == 1:
        return (1, 0) if n_train >= n_val else (0, 1)

    raw_train = n_keep * n_train / available
    raw_val = n_keep * n_val / available
    t_keep = int(math.floor(raw_train))
    v_keep = int(math.floor(raw_val))
    remainder = n_keep - t_keep - v_keep
    frac_t = raw_train - t_keep
    frac_v = raw_val - v_keep
    if remainder >= 2:
        t_keep += 1
        v_keep += 1
        remainder -= 2
    if remainder == 1:
        if frac_t > frac_v or (frac_t == frac_v and n_train >= n_val):
            t_keep += 1
        else:
            v_keep += 1

    t_keep = min(n_train, t_keep)
    v_keep = min(n_val, v_keep)
    if t_keep < 1:
        t_keep = 1
        v_keep = min(n_val, n_keep - 1)
    if v_keep < 1:
        v_keep = 1
        t_keep = min(n_train, n_keep - 1)
    t_keep = min(n_train, max(0, t_keep))
    v_keep = min(n_val, max(0, v_keep))

    total = t_keep + v_keep
    if total < n_keep:
        extra = n_keep - total
        add_t = min(n_train - t_keep, extra)
        t_keep += add_t
        extra -= add_t
        v_keep += min(n_val - v_keep, extra)
    elif total > n_keep:
        need = total - n_keep
        if t_keep >= v_keep:
            cut = min(need, max(0, t_keep - 1))
            t_keep -= cut
            need -= cut
            v_keep -= min(need, max(0, v_keep - 1))
        else:
            cut = min(need, max(0, v_keep - 1))
            v_keep -= cut
            need -= cut
            t_keep -= min(need, max(0, t_keep - 1))
    return t_keep, v_keep


def _rng_for_split(
    *,
    dataset: str,
    participant_id: int,
    split_seed: int,
    budget: int,
    stream: int,
) -> np.random.Generator:
    seed_seq = np.random.SeedSequence(
        [
            int(split_seed) & 0xFFFFFFFF,
            int(participant_id) & 0xFFFFFFFF,
            int(budget) & 0xFFFFFFFF,
            dataset_seed_uint32(dataset),
            int(stream) & 0xFFFFFFFF,
        ]
    )
    return np.random.default_rng(seed_seq)


def _sample_indices(
    n_items: int,
    n_keep: int,
    rng: np.random.Generator,
) -> List[int]:
    if n_keep <= 0:
        return []
    if n_keep >= n_items:
        return list(range(n_items))
    chosen = rng.choice(n_items, size=n_keep, replace=False)
    return sorted(int(i) for i in chosen)


def apply_max_observed_trials(
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    test_trials: Sequence[Dict[str, Any]],
    *,
    max_observed_trials_per_participant: Optional[int],
    dataset: str,
    participant_id: int,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SparseObservationAudit]:
    """Subsample train+val after the split. Test is returned unchanged."""
    train_list = list(train_trials)
    val_list = list(val_trials)
    test_list = list(test_trials)
    budget = normalize_max_observed_trials(max_observed_trials_per_participant)
    n_train = len(train_list)
    n_val = len(val_list)
    n_test = len(test_list)
    available = n_train + n_val

    if budget is None or available <= budget:
        train_idx = tuple(range(n_train))
        val_idx = tuple(range(n_val))
        audit = SparseObservationAudit(
            dataset=str(dataset),
            participant_id=int(participant_id),
            split_seed=int(split_seed),
            max_observed_trials_per_participant=budget,
            applied=False,
            original_n_train=n_train,
            original_n_val=n_val,
            original_n_test=n_test,
            retained_n_train=n_train,
            retained_n_val=n_val,
            retained_n_test=n_test,
            selected_train_indices=train_idx,
            selected_val_indices=val_idx,
            subset_fingerprint=subset_fingerprint(
                dataset=dataset,
                participant_id=participant_id,
                split_seed=split_seed,
                budget=budget,
                selected_train_indices=train_idx,
                selected_val_indices=val_idx,
            ),
        )
        return train_list, val_list, test_list, audit

    n_train_keep, n_val_keep = allocate_train_val_counts(n_train, n_val, budget)
    train_rng = _rng_for_split(
        dataset=dataset,
        participant_id=participant_id,
        split_seed=split_seed,
        budget=budget,
        stream=_TRAIN_STREAM,
    )
    val_rng = _rng_for_split(
        dataset=dataset,
        participant_id=participant_id,
        split_seed=split_seed,
        budget=budget,
        stream=_VAL_STREAM,
    )
    train_idx = _sample_indices(n_train, n_train_keep, train_rng)
    val_idx = _sample_indices(n_val, n_val_keep, val_rng)
    sparse_train = [train_list[i] for i in train_idx]
    sparse_val = [val_list[i] for i in val_idx]
    audit = SparseObservationAudit(
        dataset=str(dataset),
        participant_id=int(participant_id),
        split_seed=int(split_seed),
        max_observed_trials_per_participant=budget,
        applied=True,
        original_n_train=n_train,
        original_n_val=n_val,
        original_n_test=n_test,
        retained_n_train=len(sparse_train),
        retained_n_val=len(sparse_val),
        retained_n_test=n_test,
        selected_train_indices=tuple(train_idx),
        selected_val_indices=tuple(val_idx),
        subset_fingerprint=subset_fingerprint(
            dataset=dataset,
            participant_id=participant_id,
            split_seed=split_seed,
            budget=budget,
            selected_train_indices=train_idx,
            selected_val_indices=val_idx,
        ),
    )
    return sparse_train, sparse_val, test_list, audit


def write_sparse_audit(path: Path, audit: SparseObservationAudit) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def write_sparse_audits_payload(path: Path, audits: Sequence[SparseObservationAudit]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_participants": len(audits),
        "participants": [a.to_dict() for a in audits],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def sparse_audit_csv_fieldnames() -> List[str]:
    return [
        "dataset",
        "participant_id",
        "split_seed",
        "max_observed_trials_per_participant",
        "applied",
        "original_n_train",
        "original_n_val",
        "original_n_test",
        "retained_n_train",
        "retained_n_val",
        "retained_n_test",
        "original_n_observed",
        "retained_n_observed",
        "subset_fingerprint",
        "selected_train_indices",
        "selected_val_indices",
    ]


def write_sparse_audits_csv(path: Path, audits: Sequence[SparseObservationAudit]) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sparse_audit_csv_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            writer.writerow(audit.csv_row())
    return path


def append_sparse_audit_jsonl(path: Path, audit: SparseObservationAudit) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict(), sort_keys=True) + "\n")


def should_persist_sparse_audit(audit: SparseObservationAudit) -> bool:
    """Persist audits only when a sparse budget was requested."""
    return audit.max_observed_trials_per_participant is not None


def rewrite_sparse_audit_csv_from_jsonl(
    jsonl_path: Path,
    csv_path: Optional[Path] = None,
) -> Optional[Path]:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        return None
    audits: List[SparseObservationAudit] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        audits.append(SparseObservationAudit.from_dict(json.loads(line)))
    out = Path(csv_path) if csv_path is not None else jsonl_path.with_name(
        SPARSE_AUDIT_CSV_FILENAME
    )
    return write_sparse_audits_csv(out, audits)


_AUDIT_MATCH_FIELDS = (
    "dataset",
    "participant_id",
    "split_seed",
    "max_observed_trials_per_participant",
    "original_n_train",
    "original_n_val",
    "original_n_test",
    "retained_n_train",
    "retained_n_val",
    "retained_n_test",
    "selected_train_indices",
    "selected_val_indices",
    "subset_fingerprint",
)


def load_sparse_audits(path: Path) -> List[SparseObservationAudit]:
    """Load audits from a run directory, JSON/JSONL payload, or CSV."""
    path = Path(path)
    if path.is_dir():
        json_path = path / SPARSE_AUDIT_FILENAME
        csv_path = path / SPARSE_AUDIT_CSV_FILENAME
        jsonl_path = path / SPARSE_AUDIT_JSONL_FILENAME
        if json_path.is_file():
            return load_sparse_audits(json_path)
        if csv_path.is_file():
            return load_sparse_audits(csv_path)
        if jsonl_path.is_file():
            return load_sparse_audits(jsonl_path)
        raise FileNotFoundError(
            f"No {SPARSE_AUDIT_FILENAME}, {SPARSE_AUDIT_CSV_FILENAME}, or "
            f"{SPARSE_AUDIT_JSONL_FILENAME} under {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Sparse audit path not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        import csv

        with path.open(newline="", encoding="utf-8") as f:
            return [SparseObservationAudit.from_dict(row) for row in csv.DictReader(f)]
    if suffix == ".jsonl":
        audits: List[SparseObservationAudit] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            audits.append(SparseObservationAudit.from_dict(json.loads(line)))
        return audits
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "participants" in payload:
        return [SparseObservationAudit.from_dict(row) for row in payload["participants"]]
    if isinstance(payload, list):
        return [SparseObservationAudit.from_dict(row) for row in payload]
    if isinstance(payload, dict):
        return [SparseObservationAudit.from_dict(payload)]
    raise ValueError(f"Unrecognized sparse audit payload in {path}")


def load_reference_audits_by_participant(
    path: Path,
) -> Dict[int, SparseObservationAudit]:
    audits = load_sparse_audits(path)
    if not audits:
        raise ValueError(f"No sparse audits in {path}")
    by_pid: Dict[int, SparseObservationAudit] = {}
    for audit in audits:
        pid = int(audit.participant_id)
        if pid in by_pid:
            raise ValueError(f"Duplicate participant_id={pid} in {path}")
        by_pid[pid] = audit
    return by_pid


def audit_match_errors(
    got: SparseObservationAudit,
    expected: SparseObservationAudit,
) -> List[str]:
    errors: List[str] = []
    for field in _AUDIT_MATCH_FIELDS:
        left = getattr(got, field)
        right = getattr(expected, field)
        if left != right:
            errors.append(
                f"participant {got.participant_id} {field}: got {left!r} != expected {right!r}"
            )
    return errors


def require_audit_matches_reference(
    audit: SparseObservationAudit,
    reference_by_pid: Dict[int, SparseObservationAudit],
) -> None:
    expected = reference_by_pid.get(int(audit.participant_id))
    if expected is None:
        raise ValueError(
            f"Sparse subset mismatch: participant {audit.participant_id} is not in the "
            "Stage D reference audit."
        )
    errors = audit_match_errors(audit, expected)
    if errors:
        raise ValueError(
            "Sparse subset mismatch vs Stage D reference:\n" + "\n".join(errors)
        )


def require_all_reference_participants_seen(
    reference_by_pid: Dict[int, SparseObservationAudit],
    seen_pids: Sequence[int],
) -> None:
    seen = {int(pid) for pid in seen_pids}
    expected = set(reference_by_pid)
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(
            "Sparse subset mismatch vs Stage D reference participants: "
            f"missing={missing} extra={extra}"
        )


def replay_sparse_audit_from_original_counts(
    expected: SparseObservationAudit,
) -> SparseObservationAudit:
    """Re-apply the shared cap using only the original split sizes (no trial content)."""
    train = [{"i": i} for i in range(expected.original_n_train)]
    val = [{"i": i} for i in range(expected.original_n_val)]
    test = [{"i": i} for i in range(expected.original_n_test)]
    _, _, _, audit = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=expected.max_observed_trials_per_participant,
        dataset=expected.dataset,
        participant_id=expected.participant_id,
        split_seed=expected.split_seed,
    )
    return audit


def distribute_explore_budget(n_explore: int, n_parents: int) -> List[int]:
    """Split a fixed explore-candidate budget across parents (no multiplication)."""
    n_explore = max(0, int(n_explore))
    n_parents = int(n_parents)
    if n_parents <= 0:
        return []
    base, remainder = divmod(n_explore, n_parents)
    return [base + (1 if i < remainder else 0) for i in range(n_parents)]


def initial_program_id_from_path(path: Path, used_ids: Iterable[str]) -> str:
    """Stable distinct program id from a filename stem."""
    import re

    stem = re.sub(r"[^\w.\-]+", "_", Path(path).stem) or "program"
    program_id = stem if stem.startswith("global_") else f"global_{stem}"
    used = set(used_ids)
    if program_id not in used:
        return program_id
    n = 2
    while f"{program_id}_{n}" in used:
        n += 1
    return f"{program_id}_{n}"
