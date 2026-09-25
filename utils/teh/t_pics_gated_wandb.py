"""Reporting-only W&B helper for gated T-PICS.

Never imported by fitness, ranking, gating, resume completeness, or argv builders.
Failures here must not change local scientific artifacts or method decisions.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.teh.t_pics_gated_transfer import (
    DEFAULT_EXPLORE_CANDIDATES,
    DEFAULT_GLOBAL_ITERS,
    DEFAULT_N_CANDIDATES_PER_ITER,
    DEFAULT_N_ITERATIONS,
    KIND,
    RUN_TAG,
    participant_run_is_complete,
    participant_train_val_loglik,
    program_has_valid_choose,
)

GROUP = "t_pics_gated_main"
JOB_TYPE = "t_pics_gated"
TAGS = ("ICLR", "SA40", "gated_t_pics")
WANDB_RUN_FILENAME = "wandb_run.json"
_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")
# Dynamic per-person Runs-table columns (one key per participant). Never upload.
# Matches p0/test_loglik and p12_train_acc only — not fixed ``participant/*``.
_DYNAMIC_PARTICIPANT_SCALAR_KEY = re.compile(r"^p\d+(?:/|_)")


def is_dynamic_participant_wandb_key(key: Any) -> bool:
    """True for per-participant scalar keys that explode the W&B Runs table."""
    return bool(_DYNAMIC_PARTICIPANT_SCALAR_KEY.match(str(key)))


def strip_dynamic_participant_wandb_keys(
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Drop ``p{pid}/*`` / ``p{pid}_*`` keys; keep fixed contract (incl. ``participant/*``)."""
    return {
        str(k): v
        for k, v in data.items()
        if not is_dynamic_participant_wandb_key(k)
    }


def suppress_dynamic_participant_wandb_scalars() -> bool:
    """PICS v3 jobs (G.1 or gated) should not upload per-person scalar columns.

    Detection is env-based (cluster workers export ``KIND`` / ``WANDB_PROJECT_NAME``).
    Does not affect Centaur/OpenEvolve.
    """
    kind = str(os.environ.get("KIND") or "").strip()
    if kind.startswith("pics_v3"):
        return True
    project = str(
        os.environ.get("WANDB_PROJECT_NAME")
        or os.environ.get("WANDB_PROJECT")
        or ""
    ).strip()
    return project == "teh_pics_v3"


def _warn(message: str) -> None:
    print(f"[T-PICS gated wandb] {message}", flush=True)


def _git_commit(repo: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        text = out.decode("utf-8", errors="replace").strip()
        return text or None
    except Exception:
        return None


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _is_nonfinite_loglik(value: Any) -> bool:
    """True when a reported loglik is present but not finite (e.g. -inf)."""
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not (number == number) or number in (float("inf"), float("-inf"))


def count_inf_test_loglik_participants(rows: Sequence[Mapping[str, Any]]) -> int:
    """Completed people whose frozen best program has non-finite final test loglik."""
    n = 0
    for row in rows:
        if str(row.get("completion_status")) != "complete":
            continue
        if bool(row.get("final_test_is_nonfinite")):
            n += 1
    return int(n)


def deterministic_wandb_run_id(
    *,
    dataset: str,
    output_root: Path,
    method: str = KIND,
    run_tag: str = RUN_TAG,
) -> str:
    """Stable W&B id bound to method + dataset + this job output directory."""
    payload = f"{method}|{run_tag}|{dataset}|{Path(output_root).resolve()}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    alias = _ID_SAFE.sub("_", str(dataset))[:24].strip("_") or "dataset"
    run_id = f"tpg_{alias}_{digest}"
    return run_id[:64]


def persist_wandb_run_identity(
    output_root: Path,
    *,
    run_id: str,
    project: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the chosen W&B run id before expensive work. Never overwrites a different id."""
    path = Path(output_root) / WANDB_RUN_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            existing = loaded
    prior = str(existing.get("wandb_run_id") or "").strip()
    if prior and prior != str(run_id):
        _warn(
            f"keeping persisted wandb_run_id={prior} "
            f"(computed {run_id} for this directory)"
        )
        run_id = prior
    payload = dict(existing)
    payload.update(
        {
            "wandb_run_id": str(run_id),
            "project": str(project),
            "group": GROUP,
            "job_type": JOB_TYPE,
            "tags": list(TAGS),
            "method": KIND,
            "run_tag": RUN_TAG,
        }
    )
    if extra:
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_persisted_wandb_run_id(output_root: Path) -> Optional[str]:
    path = Path(output_root) / WANDB_RUN_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    run_id = str(payload.get("wandb_run_id") or "").strip()
    return run_id or None


def gated_wandb_run_name(dataset: str, slurm_job_id: Optional[str]) -> str:
    job = str(slurm_job_id or os.environ.get("SLURM_JOB_ID") or "local").strip() or "local"
    alias = _ID_SAFE.sub("_", str(dataset))
    return f"{alias}_t_pics_gated_{job}"


def build_gated_wandb_config(
    *,
    args: Any,
    output_root: Path,
    selected_source: str,
    source_job_id: str,
    source_rank1: Path,
    source_config_path: Path,
    source_config_sha256: Optional[str],
    prompt_mode: Optional[str],
    git_commit: Optional[str],
    slurm_job_id: Optional[str],
) -> Dict[str, Any]:
    rank1 = Path(source_rank1)

    def _int_or_default(attr: str, default: int) -> int:
        raw = getattr(args, attr, None)
        if raw is None:
            return int(default)
        return int(raw)

    cfg: Dict[str, Any] = {
        "method": KIND,
        "target_dataset": str(getattr(args, "dataset", "")),
        "limited_data_protocol": str(getattr(args, "limited_data_protocol", "")),
        "limited_train_val": getattr(args, "limited_train_val", None),
        "split_seed": int(getattr(args, "split_seed", 0) or 0),
        "evolution_selection_score": str(
            getattr(args, "evolution_selection_score", "train_val")
        ),
        "frozen_source_config": str(source_config_path),
        "frozen_source_config_sha256": source_config_sha256,
        "selected_source_dataset": str(selected_source),
        "source_g1_job_id": str(source_job_id),
        "source_rank1_program": str(rank1),
        "source_rank1_program_sha256": _sha256_file(rank1) if rank1.is_file() else None,
        "independent_source_population": bool(
            getattr(args, "t_pics_gated_independent", False)
        ),
        "target_control_iterations": _int_or_default(
            "global_iters", DEFAULT_GLOBAL_ITERS
        ),
        "target_transfer_iterations": _int_or_default(
            "global_iters", DEFAULT_GLOBAL_ITERS
        ),
        "exploration_candidates": _int_or_default(
            "explore_candidates", DEFAULT_EXPLORE_CANDIDATES
        ),
        "participant_iterations": _int_or_default(
            "n_iterations", DEFAULT_N_ITERATIONS
        ),
        "n_candidates": _int_or_default(
            "n_candidates", DEFAULT_N_CANDIDATES_PER_ITER
        ),
        "fresh_n_candidates": getattr(args, "fresh_n_candidates", None),
        "sample_size": getattr(args, "sample_size", None),
        "sample_parents": bool(getattr(args, "sample_parents", True)),
        "sampled_parents_decay": bool(getattr(args, "sampled_parents_decay", True)),
        "elite_pool_size": getattr(args, "elite_pool_size", None),
        "model_name": str(getattr(args, "model_name", "") or ""),
        "automatic_prompt_mode": prompt_mode,
        "slurm_job_id": slurm_job_id or os.environ.get("SLURM_JOB_ID"),
        "output_root": str(Path(output_root).resolve()),
        "git_commit": git_commit,
        "run_tag": RUN_TAG,
        "kind": KIND,
        "group": GROUP,
    }
    try:
        from utils.teh.pics_v3_ablation import (
            ABLATION_WANDB_GROUP,
            ablation_metadata_payload,
        )

        ablation_meta = ablation_metadata_payload(args)
        if ablation_meta.get("ablation_id"):
            cfg["group"] = ABLATION_WANDB_GROUP
            cfg["ablation"] = ablation_meta
            cfg["kind"] = ablation_meta.get("ablation_kind") or KIND
            if bool(getattr(args, "t_pics_gated_control_only", False)):
                cfg["target_transfer_iterations"] = 0
    except Exception:
        pass
    try:
        from utils.teh.pics_v3 import (
            FAMILY_PROMPT_V3_CONTROL_KIND,
            FAMILY_PROMPT_V3_RUN_TAG,
            FAMILY_PROMPT_V3_TREATMENT_KIND,
            FAMILY_PROMPT_V3_WANDB_GROUP,
            FAMILY_PROMPT_V4_KIND,
            FAMILY_PROMPT_V4_RUN_TAG,
            FAMILY_PROMPT_V4_WANDB_GROUP,
        )

        seq_v3 = bool(getattr(args, "pics_v3_sequential_rl_reminder_v3", False))
        fb_v3 = bool(getattr(args, "pics_v3_feedback_learning_reminder_v3", False))
        seq_v4 = bool(getattr(args, "pics_v3_sequential_rl_reminder_v4", False))
        out_root_s = str(Path(output_root).resolve())
        if seq_v4 or FAMILY_PROMPT_V4_KIND in out_root_s:
            cfg["group"] = FAMILY_PROMPT_V4_WANDB_GROUP
            cfg["run_tag"] = FAMILY_PROMPT_V4_RUN_TAG
            cfg["kind"] = FAMILY_PROMPT_V4_KIND
            cfg["family_prompt_v4"] = {
                "sequential_rl_reminder_v4": seq_v4,
                "policy_id": "sequential_rl_reminder_v4",
            }
        elif seq_v3 or fb_v3 or FAMILY_PROMPT_V3_TREATMENT_KIND in out_root_s:
            cfg["group"] = FAMILY_PROMPT_V3_WANDB_GROUP
            cfg["run_tag"] = FAMILY_PROMPT_V3_RUN_TAG
            cfg["kind"] = FAMILY_PROMPT_V3_TREATMENT_KIND
            cfg["family_prompt_v3"] = {
                "sequential_rl_reminder_v3": seq_v3,
                "feedback_learning_reminder_v3": fb_v3,
            }
        elif FAMILY_PROMPT_V3_CONTROL_KIND in out_root_s:
            cfg["group"] = FAMILY_PROMPT_V3_WANDB_GROUP
            cfg["run_tag"] = FAMILY_PROMPT_V3_RUN_TAG
            cfg["kind"] = FAMILY_PROMPT_V3_CONTROL_KIND
            cfg["family_prompt_v3"] = {
                "sequential_rl_reminder_v3": False,
                "feedback_learning_reminder_v3": False,
            }
    except Exception:
        pass
    return cfg


def collect_final_participant_rows(
    selected_dir: Path,
    expected_participant_ids: Sequence[int],
    *,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    explore_candidates: int = DEFAULT_EXPLORE_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Authoritative per-person rows from local result files (paper inputs)."""
    rows: List[Dict[str, Any]] = []
    for ordinal, pid in enumerate(expected_participant_ids):
        person = Path(selected_dir) / f"participant_{int(pid)}"
        results_path = person / "results.json"
        program_path = person / "best_program.py"
        complete = participant_run_is_complete(
            person,
            expected_n_iterations=int(n_iterations),
            expected_explore_candidates=int(explore_candidates),
        )
        payload: Dict[str, Any] = {}
        if results_path.is_file():
            try:
                loaded = json.loads(results_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                payload = loaded
        best = payload.get("overall_best_train") or {}
        test_block = payload.get("overall_best_test") or best
        train_val = _finite(best.get("selection_score"))
        if train_val is None:
            train_val = participant_train_val_loglik(
                _finite(best.get("train_loglik")),
                _finite(best.get("val_loglik")),
                1 if _finite(best.get("train_loglik")) is not None else 0,
                1 if _finite(best.get("val_loglik")) is not None else 0,
            )
        program_ok, _msg = (
            program_has_valid_choose(program_path) if program_path.is_file() else (False, "")
        )
        status = "complete" if complete and program_ok else "incomplete"
        raw_test_ll = test_block.get("test_loglik")
        rows.append(
            {
                "participant_ordinal": int(ordinal),
                "participant_id": int(pid),
                "final_program_id": best.get("program_id") or test_block.get("program_id"),
                "final_program_sha256": _sha256_file(program_path) if program_path.is_file() else None,
                "final_train_val_loglik": train_val,
                "final_test_loglik": _finite(raw_test_ll),
                "final_test_is_nonfinite": _is_nonfinite_loglik(raw_test_ll),
                "completion_status": status,
            }
        )
    return rows


def mean_from_complete_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> Optional[float]:
    if not rows:
        return None
    if any(str(row.get("completion_status")) != "complete" for row in rows):
        return None
    values: List[float] = []
    for row in rows:
        number = _finite(row.get(field))
        if number is None:
            return None
        values.append(number)
    if len(values) != len(rows):
        return None
    return float(sum(values) / float(len(values)))


def _backfill_g2_points(arm_dir: Path) -> List[Dict[str, Any]]:
    global_dir = Path(arm_dir) / "global_phase"
    points: List[Dict[str, Any]] = []
    if not global_dir.is_dir():
        return points
    for child in sorted(global_dir.glob("iteration_*")):
        metrics_path = child / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metrics, dict):
            continue
        iteration = metrics.get("iteration")
        try:
            iteration_i = int(iteration)
        except (TypeError, ValueError):
            continue
        diagnostic = metrics.get("diagnostic_test") or {}
        points.append(
            {
                "iteration": iteration_i,
                "best_train_val_loglik": _finite(
                    metrics.get("pool_best_selection_score")
                    or metrics.get("pool_best_global_train_loglik")
                ),
                "diagnostic_test_loglik": _finite(diagnostic.get("test_loglik")),
            }
        )
    return points


class GatedWandbReporter:
    """Drop-in wandb-module stand-in: remaps logs, swallows W&B errors, never reads back."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wandb: Any = None
        self._run: Any = None
        self._enabled = False
        self._finished = False
        self._g2_arm: Optional[str] = None
        self._participant_ordinals: Dict[int, int] = {}
        self._expected_n = 0
        self._n_person_iters = DEFAULT_N_ITERATIONS
        self._explore_candidates = DEFAULT_EXPLORE_CANDIDATES
        self._selected_dir: Optional[Path] = None
        self._expected_pids: List[int] = []
        self._output_root: Optional[Path] = None
        self.run_id: Optional[str] = None
        self.project: Optional[str] = None
        self._last_milestone: Optional[str] = None
        self._milestone_path: Optional[Path] = None

    def __bool__(self) -> bool:
        return True

    def _milestone(self, message: str) -> None:
        """Short status line for humans (summary + Files/milestones.log).

        Console capture stays off so TEH/vLLM stdout does not flood the W&B Logs tab.
        """
        text = " ".join(str(message).split())
        if not text or text == self._last_milestone:
            return
        self._last_milestone = text
        line = f"[pics_v3] {text}"
        print(line, flush=True)
        self._safe_summary({"status/last_message": text})
        if not self._enabled or self._run is None or self._wandb is None:
            return
        try:
            if self._milestone_path is None:
                raw_dir = str(getattr(self._run, "dir", "") or "").strip()
                # Path("") is_dir() is True (cwd) — do not write into the repo.
                if not raw_dir:
                    return
                run_dir = Path(raw_dir)
                if not run_dir.is_dir():
                    return
                files_dir = run_dir / "files"
                files_dir.mkdir(parents=True, exist_ok=True)
                self._milestone_path = files_dir / "milestones.log"
            with self._milestone_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            # Live-upload the small milestone file (not full TEH logs).
            self._wandb.save(
                str(self._milestone_path),
                base_path=str(self._milestone_path.parent),
                policy="live",
            )
        except Exception as exc:
            _warn(f"milestone write failed (ignored): {exc}")

    def attach_context(
        self,
        *,
        expected_participant_ids: Sequence[int],
        selected_dir: Path,
        n_iterations: int,
        explore_candidates: int,
        output_root: Path,
    ) -> None:
        pids = [int(p) for p in expected_participant_ids]
        self._expected_pids = pids
        self._expected_n = len(pids)
        self._participant_ordinals = {pid: i for i, pid in enumerate(pids)}
        self._selected_dir = Path(selected_dir)
        self._n_person_iters = int(n_iterations)
        self._explore_candidates = int(explore_candidates)
        self._output_root = Path(output_root)

    def set_g2_arm(self, arm: str) -> None:
        self._g2_arm = str(arm)
        self._safe_summary(
            {
                "progress/stage": f"g2_{arm}",
                "status/last_completed_stage": f"g2_{arm}_running",
                "status/state": "running",
            }
        )
        self._milestone(f"G.2 {arm} arm started")

    def init(
        self,
        *,
        args: Any,
        output_root: Path,
        project: str,
        selected_source: str,
        source_job_id: str,
        source_rank1: Path,
        source_config_path: Path,
        source_config_sha256: Optional[str],
        prompt_mode: Optional[str],
        expected_participant_ids: Sequence[int],
        selected_dir: Path,
        repo_root: Path,
        enabled: bool,
    ) -> "GatedWandbReporter":
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        computed_id = deterministic_wandb_run_id(
            dataset=str(args.dataset),
            output_root=output_root,
        )
        persist_wandb_run_identity(
            output_root,
            run_id=computed_id,
            project=str(project),
            extra={"dataset": str(args.dataset), "slurm_job_id": slurm_job_id},
        )
        run_id = load_persisted_wandb_run_id(output_root) or computed_id
        self.run_id = run_id
        self.project = str(project)
        self.attach_context(
            expected_participant_ids=expected_participant_ids,
            selected_dir=selected_dir,
            n_iterations=int(getattr(args, "n_iterations", DEFAULT_N_ITERATIONS) or DEFAULT_N_ITERATIONS),
            explore_candidates=int(
                getattr(args, "explore_candidates", DEFAULT_EXPLORE_CANDIDATES)
                or DEFAULT_EXPLORE_CANDIDATES
            ),
            output_root=output_root,
        )
        if not enabled:
            _warn("W&B disabled (--no_log); local files remain authoritative.")
            return self
        git_commit = _git_commit(repo_root)
        config = build_gated_wandb_config(
            args=args,
            output_root=output_root,
            selected_source=selected_source,
            source_job_id=source_job_id,
            source_rank1=source_rank1,
            source_config_path=source_config_path,
            source_config_sha256=source_config_sha256,
            prompt_mode=prompt_mode,
            git_commit=git_commit,
            slurm_job_id=slurm_job_id,
        )
        os.environ.setdefault("WANDB_DISABLE_CODE", "true")
        os.environ.setdefault("WANDB_DISABLE_GIT", "true")
        try:
            import wandb as wandb_mod

            settings = None
            try:
                settings = wandb_mod.Settings(
                    save_code=False,
                    disable_git=True,
                    console="off",
                )
            except TypeError:
                try:
                    settings = wandb_mod.Settings(save_code=False)
                except Exception:
                    settings = None
            init_kwargs: Dict[str, Any] = {
                "project": str(project),
                "name": gated_wandb_run_name(str(args.dataset), slurm_job_id),
                "id": run_id,
                "resume": "allow",
                "group": GROUP,
                "job_type": JOB_TYPE,
                "tags": list(TAGS),
                "config": config,
                "reinit": False,
                "allow_val_change": True,
            }
            if settings is not None:
                init_kwargs["settings"] = settings
            run = wandb_mod.init(**init_kwargs)
            self._wandb = wandb_mod
            self._run = run
            self._enabled = True
            self._define_step_metrics()
            self._safe_summary(
                {
                    "status/state": "running",
                    "status/source_dataset": selected_source,
                    "status/failure_reason": None,
                    "progress/expected_participants": self._expected_n,
                    "progress/completed_participants": 0,
                    "progress/failed_participants": 0,
                    "progress/stage": "started",
                    "dataset": str(args.dataset),
                }
            )
            _warn(f"run id={run_id} project={project} resume=allow")
            self._milestone(
                f"started dataset={args.dataset} source={selected_source} "
                f"persons={self._expected_n} job={slurm_job_id or 'local'}"
            )
        except Exception as exc:
            self._enabled = False
            self._wandb = None
            self._run = None
            _warn(f"init failed (experiment continues): {exc}")
        return self

    def _define_step_metrics(self) -> None:
        if self._wandb is None:
            return
        specs = (
            ("g2/control/iteration", "g2/control/*"),
            ("g2/transfer/iteration", "g2/transfer/*"),
            ("participant/step", "participant/*"),
            ("progress/event", "progress/*"),
        )
        for step_name, pattern in specs:
            try:
                self._wandb.define_metric(step_name)
                self._wandb.define_metric(pattern, step_metric=step_name)
            except Exception as exc:
                _warn(f"define_metric failed: {exc}")

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _safe_log(self, data: Mapping[str, Any], **kwargs: Any) -> None:
        if not self._enabled or self._wandb is None or not data:
            return
        payload = strip_dynamic_participant_wandb_keys(data)
        if not payload:
            return
        try:
            with self._lock:
                self._wandb.log(dict(payload), **kwargs)
        except Exception as exc:
            _warn(f"log failed (ignored): {exc}")

    def _safe_summary(self, fields: Mapping[str, Any]) -> None:
        if not fields:
            return
        if not self._enabled or self._run is None:
            return
        cleaned = strip_dynamic_participant_wandb_keys(fields)
        if not cleaned:
            return
        try:
            with self._lock:
                for key, value in cleaned.items():
                    self._run.summary[key] = value
        except Exception as exc:
            _warn(f"summary failed (ignored): {exc}")

    def log(self, data: Optional[Mapping[str, Any]] = None, step: Any = None, **kwargs: Any) -> None:
        if not data:
            return
        remapped = self._remap(dict(data), step=step)
        if remapped:
            self._safe_log(remapped)

    def _remap(self, data: Dict[str, Any], step: Any = None) -> Dict[str, Any]:
        keys = set(data)
        if any(
            str(k).startswith(("g2/", "participant/", "progress/", "gate/", "final/", "status/"))
            for k in keys
        ):
            # Fixed-contract payloads: keep non-None fixed keys only (never p{pid}/*).
            return {
                k: v
                for k, v in strip_dynamic_participant_wandb_keys(data).items()
                if v is not None
            }
        if "global/selection_score" in data or "global/train_loglik" in data:
            arm = self._g2_arm or "control"
            iteration = data.get("global/iteration", step)
            out: Dict[str, Any] = {
                f"g2/{arm}/iteration": iteration,
                f"g2/{arm}/best_train_val_loglik": data.get(
                    "global/selection_score", data.get("global/train_loglik")
                ),
                f"g2/{arm}/diagnostic_test_loglik": data.get("global/diagnostic_test_loglik"),
            }
            return {k: v for k, v in out.items() if v is not None}
        pid, ordinal, iteration = self._participant_coords(data, step)
        if pid is None:
            # Drop unknown / dynamic-only payloads rather than forwarding p{pid} scalars.
            return {}
        if data.get(f"p{pid}_is_baseline") == 1 or data.get(f"p{pid}/is_baseline") == 1:
            return {}
        train_val = data.get(f"p{pid}/selection_score")
        if train_val is None:
            train_val = data.get(f"p{pid}_selection_score")
        if train_val is None:
            train_val = data.get(f"p{pid}/train_loglik", data.get(f"p{pid}_train_loglik"))
        test_ll = data.get(f"p{pid}/test_loglik", data.get(f"p{pid}_test_loglik"))
        person_step = int(ordinal or 0) * 100 + int(iteration or 0)
        out = {
            "participant/step": person_step,
            "participant/ordinal": ordinal,
            "participant/iteration": iteration,
            "participant/best_train_val_loglik": train_val,
            "participant/diagnostic_test_loglik": test_ll,
        }
        return {k: v for k, v in out.items() if v is not None}

    def _participant_coords(
        self, data: Mapping[str, Any], step: Any
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        pid: Optional[int] = None
        iteration: Optional[int] = None
        for key in data:
            match = re.match(r"p(\d+)(?:_|/)", str(key))
            if match:
                pid = int(match.group(1))
                break
        if pid is None:
            return None, None, None
        step_key = f"p{pid}_step"
        if step_key in data:
            try:
                iteration = int(data[step_key])
            except (TypeError, ValueError):
                iteration = None
        elif step is not None:
            try:
                iteration = int(step)
            except (TypeError, ValueError):
                iteration = None
        ordinal = self._participant_ordinals.get(int(pid))
        return pid, ordinal, iteration

    def publish_gate_record(self, gate_record_path: Path) -> None:
        path = Path(gate_record_path)
        if not path.is_file():
            _warn(f"gate record missing: {path}")
            return
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _warn(f"could not read gate record: {exc}")
            return
        if not isinstance(record, dict):
            return
        self._safe_summary(
            {
                "gate/control_pooled_train_val_loglik": record.get(
                    "control_pooled_train_val_loglik"
                ),
                "gate/transfer_pooled_train_val_loglik": record.get(
                    "transfer_pooled_train_val_loglik"
                ),
                "gate/score_field": record.get("score_field"),
                "gate/score_difference": record.get("score_difference"),
                "gate/selected_arm": record.get("selected_arm"),
                "gate/reason": record.get("reason"),
                "status/selected_arm": record.get("selected_arm"),
                "status/source_dataset": record.get("selected_source"),
                "status/last_completed_stage": "gate",
                "progress/stage": "gate",
            }
        )
        self._milestone(
            f"gate selected_arm={record.get('selected_arm')} "
            f"ctrl={record.get('control_pooled_train_val_loglik')} "
            f"xfer={record.get('transfer_pooled_train_val_loglik')}"
        )

    def maybe_backfill_g2_arm(self, arm: str, arm_dir: Path) -> None:
        self._g2_arm = str(arm)
        for point in _backfill_g2_points(arm_dir):
            self._safe_log(
                {
                    f"g2/{arm}/iteration": point["iteration"],
                    f"g2/{arm}/best_train_val_loglik": point["best_train_val_loglik"],
                    f"g2/{arm}/diagnostic_test_loglik": point["diagnostic_test_loglik"],
                }
            )

    def sync_progress_from_disk(self) -> None:
        if self._selected_dir is None:
            return
        rows = collect_final_participant_rows(
            self._selected_dir,
            self._expected_pids,
            n_iterations=self._n_person_iters,
            explore_candidates=self._explore_candidates,
        )
        completed = sum(1 for row in rows if row["completion_status"] == "complete")
        failed = max(0, self._expected_n - completed)
        n_inf_test = count_inf_test_loglik_participants(rows)
        self._safe_summary(
            {
                "progress/completed_participants": completed,
                "progress/expected_participants": self._expected_n,
                "progress/failed_participants": failed,
                "progress/stage": "person" if completed < self._expected_n else "aggregating",
                # Cumulative count of finished people whose best program has -inf test LL.
                # Recomputed from disk only here / in publish_final — not during eval loops.
                "final/n_inf_test_loglik": n_inf_test,
            }
        )
        self._safe_log(
            {
                "progress/event": completed,
                "progress/completed_participants": completed,
            }
        )
        # Sparse person-progress milestones (not every sync).
        if completed > 0 and (
            completed == self._expected_n
            or completed % max(1, self._expected_n // 5) == 0
            or completed in {1, 5, 10, 25}
        ):
            self._milestone(
                f"person progress {completed}/{self._expected_n} complete "
                f"n_inf_test={n_inf_test}"
            )

    def publish_final(self) -> bool:
        """Write final/* only when every expected participant is complete. Returns is_complete."""
        if self._selected_dir is None:
            return False
        rows = collect_final_participant_rows(
            self._selected_dir,
            self._expected_pids,
            n_iterations=self._n_person_iters,
            explore_candidates=self._explore_candidates,
        )
        completed = sum(1 for row in rows if row["completion_status"] == "complete")
        is_complete = bool(rows) and completed == self._expected_n and self._expected_n > 0
        n_inf_test = count_inf_test_loglik_participants(rows)
        self._upload_table(rows)
        self._safe_summary(
            {
                "progress/completed_participants": completed,
                "progress/expected_participants": self._expected_n,
                "progress/failed_participants": max(0, self._expected_n - completed),
                "final/completed_participants": completed,
                "final/expected_participants": self._expected_n,
                "final/is_complete": bool(is_complete),
                "final/n_inf_test_loglik": n_inf_test,
            }
        )
        if not is_complete:
            self._safe_summary(
                {
                    "progress/stage": "incomplete",
                    "status/state": "running",
                    "status/last_completed_stage": "person_partial",
                }
            )
            return False
        mean_tv = mean_from_complete_rows(rows, field="final_train_val_loglik")
        mean_te = mean_from_complete_rows(rows, field="final_test_loglik")
        csv_path = self._selected_dir / "summary_loglik.csv"
        if csv_path.is_file():
            csv_mean_te = _read_summary_avg_test(csv_path)
            if csv_mean_te is not None:
                mean_te = csv_mean_te
        self._safe_summary(
            {
                "final/mean_train_val_loglik": mean_tv,
                "final/mean_test_loglik": mean_te,
                "final/n_inf_test_loglik": n_inf_test,
                "final/is_complete": True,
                "status/state": "completed",
                "status/failure_reason": None,
                "status/last_completed_stage": "completed",
                "progress/stage": "completed",
            }
        )
        self._milestone(
            f"completed mean_test_loglik={mean_te} "
            f"mean_train_val_loglik={mean_tv} "
            f"n_inf_test_loglik={n_inf_test} "
            f"persons={completed}/{self._expected_n}"
        )
        return True

    def _upload_table(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not self._enabled or self._wandb is None or not rows:
            return
        try:
            table = self._wandb.Table(
                columns=[
                    "participant_ordinal",
                    "final_program_id",
                    "final_program_sha256",
                    "final_train_val_loglik",
                    "final_test_loglik",
                    "completion_status",
                ],
                data=[
                    [
                        row.get("participant_ordinal"),
                        row.get("final_program_id"),
                        row.get("final_program_sha256"),
                        row.get("final_train_val_loglik"),
                        row.get("final_test_loglik"),
                        row.get("completion_status"),
                    ]
                    for row in rows
                ],
            )
            self._safe_log({"final/participant_table": table})
        except Exception as exc:
            _warn(f"table upload failed (ignored): {exc}")

    def mark_failed(self, reason: str) -> None:
        text = str(reason or "unknown")[:500]
        self._safe_summary(
            {
                "status/state": "failed",
                "status/failure_reason": text,
                "final/is_complete": False,
                "progress/stage": "failed",
            }
        )

    def finish(
        self,
        *,
        completed: Optional[bool] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if failure_reason:
            self.mark_failed(failure_reason)
        elif completed is True:
            self.publish_final()
        elif completed is False:
            self._safe_summary(
                {
                    "status/state": "interrupted",
                    "final/is_complete": False,
                }
            )
            self.sync_progress_from_disk()
        if not self._enabled or self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception as exc:
            _warn(f"finish failed (ignored): {exc}")


def _read_summary_avg_test(summary_csv: Path) -> Optional[float]:
    import csv

    try:
        with Path(summary_csv).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
    except (OSError, csv.Error, StopIteration):
        return None
    if not row:
        return None
    return _finite(row.get("avg_test_loglik"))


def init_gated_wandb_reporter(
    *,
    args: Any,
    output_root: Path,
    project: str,
    selected_source: str,
    source_job_id: str,
    source_rank1: Path,
    source_config_path: Path,
    source_config_sha256: Optional[str],
    prompt_mode: Optional[str],
    expected_participant_ids: Sequence[int],
    selected_dir: Path,
    repo_root: Path,
    enabled: bool,
) -> GatedWandbReporter:
    reporter = GatedWandbReporter()
    return reporter.init(
        args=args,
        output_root=output_root,
        project=project,
        selected_source=selected_source,
        source_job_id=source_job_id,
        source_rank1=source_rank1,
        source_config_path=source_config_path,
        source_config_sha256=source_config_sha256,
        prompt_mode=prompt_mode,
        expected_participant_ids=expected_participant_ids,
        selected_dir=selected_dir,
        repo_root=repo_root,
        enabled=enabled,
    )
