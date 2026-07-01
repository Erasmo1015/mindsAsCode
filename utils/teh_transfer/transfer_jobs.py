"""Transfer job definitions, scheduling, and single-mode summary matrices."""
from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from utils.teh_transfer.evolution import (
    PopulationEvolutionResult,
    _parallel_dataset_pool_sizes,
)


@dataclass(frozen=True)
class TransferJob:
    """One transfer evolution job (one target, one or more source datasets)."""

    target_key: str
    source_keys: Tuple[str, ...]
    transfer_mode: str  # "multiple" | "single"

    @property
    def job_name(self) -> str:
        """Human-readable job identifier."""
        if self.transfer_mode == "single":
            return f"target={self.target_key}/source={self.source_keys[0]}"
        return f"target={self.target_key}"

    @property
    def job_id(self) -> str:
        """Unique key for result dictionaries."""
        if self.transfer_mode == "single":
            return f"{self.target_key}<-{self.source_keys[0]}"
        return self.target_key

    def relative_output_path(self) -> str:
        """
        Output path relative to the target dataset directory.

        Multiple mode keeps the legacy layout ``transfer/``.
        Single mode nests one directory per source under ``transfer/``.
        """
        if self.transfer_mode == "single":
            if len(self.source_keys) != 1:
                raise ValueError(
                    f"single-mode job must have exactly one source, got {self.source_keys!r}"
                )
            return f"transfer/source={self.source_keys[0]}"
        return "transfer"

    def output_dir(self, target_output_dir: Path) -> Path:
        return target_output_dir / self.relative_output_path()


@dataclass
class TransferJobResult:
    job: TransferJob
    result: Optional[PopulationEvolutionResult]
    status: str
    error: Optional[str]
    runtime_seconds: float
    best_program_path: Optional[Path]

    @property
    def source_dataset(self) -> str:
        if self.transfer_mode == "single":
            return self.job.source_keys[0]
        return ""

    @property
    def target_dataset(self) -> str:
        return self.job.target_key

    @property
    def transfer_mode(self) -> str:
        return self.job.transfer_mode

    def to_jsonl_record(self) -> Dict[str, Any]:
        test_loglik = None
        first_iter_test_loglik = None
        train_val_score = None
        if self.result is not None:
            test_loglik = self.result.best_test_loglik
            first_iter_test_loglik = self.result.first_iter_best_test_loglik
            train_val_score = self.result.best_loglik
        return {
            "transfer_mode": self.job.transfer_mode,
            "source_dataset": self.job.source_keys[0] if self.job.source_keys else "",
            "source_datasets": list(self.job.source_keys),
            "target_dataset": self.job.target_key,
            "score": train_val_score,
            "test_loglik": test_loglik,
            "first_iter_test_loglik": first_iter_test_loglik,
            "best_program_path": str(self.best_program_path) if self.best_program_path else "",
            "status": self.status,
            "error": self.error or "",
            "runtime_seconds": round(self.runtime_seconds, 3),
            "job_name": self.job.job_name,
        }


def build_transfer_jobs(
    dataset_keys: Sequence[str],
    transfer_mode: str,
) -> List[TransferJob]:
    """Build transfer jobs for ``multiple`` or ``single`` mode."""
    if transfer_mode not in {"multiple", "single"}:
        raise ValueError(f"Unsupported transfer_mode: {transfer_mode!r}")
    keys = [str(k) for k in dataset_keys]
    if transfer_mode == "multiple":
        return [
            TransferJob(
                target_key=target,
                source_keys=tuple(k for k in keys if k != target),
                transfer_mode="multiple",
            )
            for target in keys
        ]

    jobs: List[TransferJob] = []
    for target in keys:
        for source in keys:
            if source == target:
                continue
            jobs.append(
                TransferJob(
                    target_key=target,
                    source_keys=(source,),
                    transfer_mode="single",
                )
            )
    return jobs


def build_transfer_job_batches(
    dataset_keys: Sequence[str],
    transfer_mode: str,
) -> List[List[TransferJob]]:
    """
    Group jobs into batches for execution.

    Multiple mode: one batch containing all target jobs (parallel across targets).
    Single mode: one batch per target (parallel across sources within a target).
    """
    jobs = build_transfer_jobs(dataset_keys, transfer_mode)
    if transfer_mode == "multiple":
        return [jobs]

    batches: List[List[TransferJob]] = []
    by_target: Dict[str, List[TransferJob]] = {}
    for job in jobs:
        by_target.setdefault(job.target_key, []).append(job)
    for target in dataset_keys:
        batch = by_target.get(str(target))
        if batch:
            batches.append(batch)
    return batches


def run_transfer_jobs_parallel(
    job_batches: Sequence[Sequence[TransferJob]],
    worker_fn: Callable[[TransferJob, int], TransferJobResult],
    *,
    max_workers: int,
    n_candidates: int,
    parallel_jobs: bool,
    on_job_complete: Optional[Callable[[TransferJobResult], None]] = None,
    batch_desc: str = "Transfer jobs",
) -> Dict[str, TransferJobResult]:
    """
    Run transfer jobs with shared worker budget.

    ``job_workers = max(1, max_workers // n_candidates)`` jobs run concurrently;
    each job uses ``candidate_workers = n_candidates`` for LLM candidate generation.
    """
    job_workers, candidate_workers = _parallel_dataset_pool_sizes(
        max_workers, n_candidates, parallel_jobs
    )
    all_results: Dict[str, TransferJobResult] = {}

    def _run_one(job: TransferJob) -> TransferJobResult:
        return worker_fn(job, candidate_workers)

    n_batches = len(job_batches)
    for batch_idx, batch in enumerate(job_batches):
        if not batch:
            continue
        desc = batch_desc
        if n_batches > 1:
            target_label = batch[0].target_key
            desc = f"{batch_desc} target={target_label} ({batch_idx + 1}/{n_batches})"

        if job_workers <= 1 or len(batch) <= 1:
            for job in tqdm(batch, desc=desc):
                job_result = _run_one(job)
                all_results[job_result.job.job_id] = job_result
                if on_job_complete is not None:
                    on_job_complete(job_result)
            continue

        with ThreadPoolExecutor(max_workers=job_workers) as pool:
            futures = {pool.submit(_run_one, job): job for job in batch}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                job_result = fut.result()
                all_results[job_result.job.job_id] = job_result
                if on_job_complete is not None:
                    on_job_complete(job_result)

    return all_results


def _read_transfer_matrix(
    path: Path,
    dataset_keys: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    keys = [str(k) for k in dataset_keys]
    matrix = {target: {source: "" for source in keys} for target in keys}
    if not path.is_file():
        return matrix
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            return matrix
        col_keys = [str(k) for k in header[1:]]
        for row in reader:
            if not row:
                continue
            target = str(row[0])
            if target not in matrix:
                continue
            for idx, source in enumerate(col_keys):
                if source not in matrix[target]:
                    continue
                value_col = idx + 1
                if value_col < len(row):
                    matrix[target][source] = row[value_col]
    return matrix


def _write_transfer_matrix(
    path: Path,
    dataset_keys: Sequence[str],
    matrix: Dict[str, Dict[str, str]],
) -> None:
    keys = [str(k) for k in dataset_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + keys)
        for target in keys:
            writer.writerow([target] + [matrix[target].get(source, "") for source in keys])


SINGLE_TRANSFER_MATRIX_TEST_PATH = "matrix_test_loglik.csv"
SINGLE_TRANSFER_MATRIX_IMPROVE_PATH = "matrix_improve_test_loglik.csv"
SINGLE_TRANSFER_SUMMARY_PATH = "transfer_summary.csv"
SINGLE_TRANSFER_1ST_ITER_SUBDIR = "transfer_1st_iter"


def _ensure_single_transfer_matrix_pair(
    output_dir: Path,
    dataset_keys: Sequence[str],
) -> Tuple[Path, Path]:
    test_path = output_dir / SINGLE_TRANSFER_MATRIX_TEST_PATH
    improve_path = output_dir / SINGLE_TRANSFER_MATRIX_IMPROVE_PATH
    empty = {t: {s: "" for s in dataset_keys} for t in dataset_keys}
    if not test_path.is_file():
        _write_transfer_matrix(test_path, dataset_keys, empty)
    if not improve_path.is_file():
        _write_transfer_matrix(improve_path, dataset_keys, empty)
    return test_path, improve_path


def ensure_single_transfer_matrix_csvs(
    summary_dir: Path,
    dataset_keys: Sequence[str],
) -> Tuple[Path, Path, Path, Path]:
    """Create empty source x target matrices for best and iteration-1 results."""
    single_dir = summary_dir / "single_transfer"
    test_path, improve_path = _ensure_single_transfer_matrix_pair(single_dir, dataset_keys)
    first_iter_dir = single_dir / SINGLE_TRANSFER_1ST_ITER_SUBDIR
    first_iter_test_path, first_iter_improve_path = _ensure_single_transfer_matrix_pair(
        first_iter_dir,
        dataset_keys,
    )
    return test_path, improve_path, first_iter_test_path, first_iter_improve_path


def _parse_matrix_float(value: str) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_single_transfer_summary_csv(
    path: Path,
    dataset_keys: Sequence[str],
    *,
    test_matrix_path: Path,
    improve_matrix_path: Path,
) -> None:
    """Aggregate per-target single-source transfer stats from matrix CSVs."""
    keys = [str(k) for k in dataset_keys]
    test_matrix = _read_transfer_matrix(test_matrix_path, keys)
    improve_matrix = _read_transfer_matrix(improve_matrix_path, keys)
    metric_names = [
        "best_single_source",
        "best_single_source_test_loglik",
        "best_single_source_improve_vs_global",
        "num_sources_better_than_global",
        "mean_single_source_improve_vs_global",
    ]
    rows: List[List[str]] = []
    for target in keys:
        improves: List[float] = []
        best_source = ""
        best_improve: Optional[float] = None
        best_test: Optional[float] = None
        for source in keys:
            if source == target:
                continue
            improve = _parse_matrix_float(improve_matrix[target].get(source, ""))
            if improve is None:
                continue
            improves.append(improve)
            if best_improve is None or improve > best_improve:
                best_improve = improve
                best_source = source
                best_test = _parse_matrix_float(test_matrix[target].get(source, ""))
        num_better = sum(1 for value in improves if value > 0.0)
        mean_improve = sum(improves) / len(improves) if improves else None
        rows.append(
            [
                target,
                best_source,
                f"{best_test:.6f}" if best_test is not None else "",
                f"{best_improve:.6f}" if best_improve is not None else "",
                str(num_better),
                f"{mean_improve:.6f}" if mean_improve is not None else "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + metric_names)
        writer.writerows(rows)


def update_single_transfer_matrix_cell(
    path: Path,
    dataset_keys: Sequence[str],
    *,
    source_key: str,
    target_key: str,
    value: Optional[float],
    fmt: str = "{:.6f}",
) -> None:
    matrix = _read_transfer_matrix(path, dataset_keys)
    if value is None:
        matrix[str(target_key)][str(source_key)] = ""
    else:
        matrix[str(target_key)][str(source_key)] = fmt.format(float(value))
    _write_transfer_matrix(path, dataset_keys, matrix)


def update_single_transfer_matrix_result(
    test_matrix_path: Path,
    improve_matrix_path: Path,
    dataset_keys: Sequence[str],
    *,
    source_key: str,
    target_key: str,
    test_loglik: Optional[float],
    global_test_loglik: Optional[float],
) -> None:
    """Write test loglik and improve-vs-global cells for one source-target pair."""
    update_single_transfer_matrix_cell(
        test_matrix_path,
        dataset_keys,
        source_key=source_key,
        target_key=target_key,
        value=test_loglik,
    )
    improve = None
    if test_loglik is not None and global_test_loglik is not None:
        improve = float(test_loglik) - float(global_test_loglik)
    update_single_transfer_matrix_cell(
        improve_matrix_path,
        dataset_keys,
        source_key=source_key,
        target_key=target_key,
        value=improve,
    )


def append_single_transfer_result_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def make_transfer_job_worker(
    *,
    global_results: Dict[str, PopulationEvolutionResult],
    run_transfer_evolution_phase: Callable[..., PopulationEvolutionResult],
    build_source_contexts_for_target: Callable[..., List[Any]],
    client: Any,
    evolution_kwargs: Dict[str, Any],
    transfer_iters: int,
    n_candidates: int,
    transfer_max_prompt_trials: int,
    debug_prompt: bool,
    repo_root: Path,
    split_ratio: float,
    split_seed: int,
    data_path: str,
    filter_mixed_gambles: bool,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    explain: bool = False,
) -> Callable[[TransferJob, int], TransferJobResult]:
    """Factory for the per-job worker used by ``run_transfer_jobs_parallel``."""

    def _worker(job: TransferJob, candidate_workers: int) -> TransferJobResult:
        import teh

        start = time.monotonic()
        target = global_results[job.target_key]
        best_program_path: Optional[Path] = None
        try:
            sources = build_source_contexts_for_target(
                job.target_key,
                global_results,
                source_keys=list(job.source_keys),
                split_ratio=split_ratio,
                split_seed=split_seed,
                data_path=data_path,
                filter_mixed_gambles=filter_mixed_gambles,
                local_dataset=local_dataset,
                mixed_gambles_csv=mixed_gambles_csv,
                repo_root=repo_root,
            )
            print(
                f"\n[transfer] mode={job.transfer_mode} target={job.target_key}, "
                f"sources={[s.dataset_alias for s in sources]}",
                flush=True,
            )
            transfer_dir = job.output_dir(target.output_dir)
            seed_path = str(target.prompts_dir / "seed_program.py")
            result = run_transfer_evolution_phase(
                target=target,
                sources=sources,
                client=client,
                seed_program_path=seed_path,
                n_iterations=transfer_iters,
                n_candidates=n_candidates,
                max_workers=candidate_workers,
                max_prompt_train_trials=transfer_max_prompt_trials,
                debug_prompt=debug_prompt,
                transfer_dir=transfer_dir,
                transfer_mode=job.transfer_mode,
                source_config_keys=list(job.source_keys),
                explain=explain,
                **evolution_kwargs,
            )
            candidate_best = transfer_dir / teh.BEST_PROGRAM_FILENAME
            if candidate_best.is_file():
                best_program_path = candidate_best
            status = "ok"
            error: Optional[str] = None
        except Exception as exc:
            result = None
            status = "error"
            error = str(exc)
            print(
                f"[transfer] ERROR job={job.job_name}: {exc}",
                flush=True,
            )
        runtime_seconds = time.monotonic() - start
        return TransferJobResult(
            job=job,
            result=result,
            status=status,
            error=error,
            runtime_seconds=runtime_seconds,
            best_program_path=best_program_path,
        )

    return _worker
