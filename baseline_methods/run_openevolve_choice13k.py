"""
OpenEvolve baseline runner for Choice13k with TE/Centaur-compatible participant selection.

This script is intentionally strict:
- No silent fallbacks for evaluation failures
- Fatal evaluator failures are printed clearly and can stop the run
- CSV outputs mirror existing Centaur/TE conventions
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import shlex
import socket
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_modules.choice13k import Experiment, get_choice13k_experiments  # noqa: E402


def _ensure_openevolve_importable() -> None:
    """Allow running without pip install by importing from reference_repos/openevolve."""
    try:
        import openevolve  # noqa: F401
        return
    except ImportError:
        local_repo = REPO_ROOT / "reference_repos" / "openevolve"
        if local_repo.is_dir():
            sys.path.insert(0, str(local_repo))
            import openevolve  # noqa: F401
            return
        raise


def _to_builtin(x: Any) -> Any:
    """Convert numpy scalars/containers into JSON-serializable builtin types."""
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_builtin(v) for v in x]
    if isinstance(x, tuple):
        return [_to_builtin(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def experiment_to_trials(exp: Experiment) -> Tuple[List[Dict[str, Any]], list]:
    options = exp.blocks[0].option_keys
    all_trials: List[Dict[str, Any]] = []
    history_accum: List[Dict[str, Any]] = []
    for block in exp.blocks:
        for trial in block.trials:
            history_entry = {"action": trial.action, "feedback": trial.feedback}
            all_trials.append(
                {
                    "problem": {
                        "gamble_A": {
                            "probs": block.gamble_A.probs,
                            "rewards": block.gamble_A.rewards,
                        },
                        "gamble_B": {
                            "probs": block.gamble_B.probs,
                            "rewards": block.gamble_B.rewards,
                        },
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": trial.action,
                }
            )
            history_accum.append(history_entry)
    return all_trials, options


def trials_from_blocks_chronological(exp: Experiment, block_indices: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(exp.blocks):
        if bi not in block_indices:
            continue
        options = block.option_keys
        history_accum: List[Dict[str, Any]] = []
        for trial in block.trials:
            out.append(
                {
                    "problem": {
                        "gamble_A": {
                            "probs": block.gamble_A.probs,
                            "rewards": block.gamble_A.rewards,
                        },
                        "gamble_B": {
                            "probs": block.gamble_B.probs,
                            "rewards": block.gamble_B.rewards,
                        },
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": trial.action,
                }
            )
            history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def split_trials(
    exp: Experiment,
    split_ratio: float = 0.9,
    split_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], list]:
    n_blocks = len(exp.blocks)
    if n_blocks < 2:
        raise ValueError(
            f"Choice13k within-participant split requires at least 2 problems (blocks); got {n_blocks}."
        )
    rng = np.random.default_rng(split_seed)
    perm = np.arange(n_blocks)
    rng.shuffle(perm)
    split_idx = int(n_blocks * split_ratio)
    split_idx = max(1, min(split_idx, n_blocks - 1))
    train_blocks = set(perm[:split_idx].tolist())
    test_blocks = set(perm[split_idx:].tolist())
    train_trials = trials_from_blocks_chronological(exp, train_blocks)
    test_trials = trials_from_blocks_chronological(exp, test_blocks)
    options = exp.blocks[0].option_keys
    return train_trials, test_trials, options


def load_valid_participant_ids_from_json(
    dataset: str, repo_root: Path, filter_mixed_gambles: bool
) -> List[int]:
    if dataset == "choice13k":
        path = repo_root / "datasets" / "choice13k" / "valid_participant_ids.json"
    elif dataset == "cpc18":
        path = repo_root / "datasets" / "cpc18" / "valid_participant_ids.json"
    elif dataset == "mixed_gambles":
        name = (
            "valid_participant_ids_gain_loss.json"
            if filter_mixed_gambles
            else "valid_participant_ids.json"
        )
        path = repo_root / "datasets" / "mixed_gambles" / name
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing valid participant list: {path}. Generate with "
            f"`python utils/tools/collect_participant_ids.py --dataset {dataset}`"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["valid_participant_ids"])


def resolve_participants_for_scope(
    *,
    dataset: str,
    repo_root: Path,
    participant_scope: str,
    single_participant_id: int,
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
    all_max_participants: Optional[int],
    filter_mixed_gambles: bool,
) -> List[int]:
    valid = load_valid_participant_ids_from_json(dataset, repo_root, filter_mixed_gambles)
    if participant_scope == "single":
        if single_participant_id not in valid:
            raise ValueError(
                f"--single_participant_id={single_participant_id} not in valid list ({len(valid)} ids)."
            )
        return [single_participant_id]
    if participant_scope == "range":
        if range_start_ordinal is None or range_end_ordinal is None:
            raise ValueError("range scope requires --range_start_ordinal and --range_end_ordinal.")
        if range_start_ordinal < 0 or range_end_ordinal >= len(valid) or range_start_ordinal > range_end_ordinal:
            raise ValueError(
                f"Invalid ordinal range [{range_start_ordinal}, {range_end_ordinal}] "
                f"for list length {len(valid)}."
            )
        return valid[range_start_ordinal : range_end_ordinal + 1]
    if participant_scope == "all":
        if all_max_participants is not None:
            return valid[: max(0, int(all_max_participants))]
        return list(valid)
    raise ValueError(f"Unknown participant_scope: {participant_scope!r}")


def _round_floats_for_csv_row(row: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (float, np.floating)):
            x = float(v)
            out[k] = round(x, ndigits) if math.isfinite(x) else x
        else:
            out[k] = v
    return out


def _round_floats_for_csv_rows(rows: List[Dict[str, Any]], ndigits: int = 4) -> List[Dict[str, Any]]:
    return [_round_floats_for_csv_row(r, ndigits) for r in rows]


def _write_all_mode_csvs(
    base: Path,
    participant_details: List[Dict[str, Any]],
    participant_loglik: List[Dict[str, Any]],
) -> None:
    base.mkdir(parents=True, exist_ok=True)
    details_file = base / "participants_details.csv"
    summary_file = base / "summary.csv"
    details_loglik_file = base / "participant_details_loglik.csv"
    summary_loglik_file = base / "summary_loglik.csv"

    with open(details_file, "w", newline="", encoding="utf-8") as f:
        fn = [
            "participant_id",
            "train_fitness",
            "test_fitness",
            "total_runtime",
            "seed_program_train_fitness",
            "seed_program_test_fitness",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(_round_floats_for_csv_rows(participant_details))

    avg_train = float(np.mean([d["train_fitness"] for d in participant_details]))
    avg_test = float(np.mean([d["test_fitness"] for d in participant_details]))
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["num_of_participants", "avg_train_fitness", "avg_test_fitness"]
        )
        w.writeheader()
        w.writerow(
            _round_floats_for_csv_row(
                {
                    "num_of_participants": len(participant_details),
                    "avg_train_fitness": avg_train,
                    "avg_test_fitness": avg_test,
                }
            )
        )

    with open(details_loglik_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["participant_id", "train_loglik", "test_loglik"])
        w.writeheader()
        w.writerows(_round_floats_for_csv_rows(participant_loglik))

    tr_vals = [d["train_loglik"] for d in participant_loglik if d["train_loglik"] is not None]
    te_vals = [d["test_loglik"] for d in participant_loglik if d["test_loglik"] is not None]
    with open(summary_loglik_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["num_of_participants", "avg_train_loglik", "avg_test_loglik"]
        )
        w.writeheader()
        w.writerow(
            _round_floats_for_csv_row(
                {
                    "num_of_participants": len(participant_loglik),
                    "avg_train_loglik": float(np.mean(tr_vals)) if tr_vals else None,
                    "avg_test_loglik": float(np.mean(te_vals)) if te_vals else None,
                }
            )
        )


def _write_command_line_log(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "log"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "command.txt"
    cmd = shlex.join([sys.executable, *sys.argv])
    stamp = datetime.now().isoformat(timespec="seconds")
    body = f"# saved {stamp}\n# cwd: {os.getcwd()}\n# host: {socket.gethostname()}\n{cmd}\n"
    path.write_text(body, encoding="utf-8")
    return path


def _build_participant_evaluator_code(dataset_json_path: Path) -> str:
    return textwrap.dedent(
        f"""
        import importlib.util
        import json
        import math
        import traceback
        from pathlib import Path

        DATA_PATH = Path(r\"{str(dataset_json_path)}\")


        def _load_program(program_path: str):
            spec = importlib.util.spec_from_file_location("candidate_module", program_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create module spec from {{program_path}}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod


        def _eval_trials(choose_fn, trials):
            total = len(trials)
            if total == 0:
                raise ValueError("No trials provided to evaluator.")
            ll_sum = 0.0
            correct = 0
            for i, t in enumerate(trials):
                y = int(t["action"])
                p_raw = choose_fn(t["problem"], t["history"])
                if not isinstance(p_raw, float):
                    raise TypeError(f"trial={{i}} expected float prob, got {{type(p_raw)}}")
                if not (0.0 <= p_raw <= 1.0):
                    raise ValueError(f"trial={{i}} probability out of [0,1]: {{p_raw}}")
                p = min(max(float(p_raw), 1e-9), 1.0 - 1e-9)
                ll_sum += y * math.log(p) + (1 - y) * math.log(1.0 - p)
                pred = 1 if p_raw >= 0.5 else 0
                correct += int(pred == y)
            return (ll_sum / total), (correct / total), total


        def evaluate(program_path: str):
            try:
                payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
                train_trials = payload["train_trials"]
                test_trials = payload["test_trials"]

                mod = _load_program(program_path)
                choose_fn = getattr(mod, "choose", None)
                if choose_fn is None:
                    raise AttributeError("Candidate program must define choose(problem, history).")

                train_ll, train_acc, train_n = _eval_trials(choose_fn, train_trials)
                test_ll, test_acc, test_n = _eval_trials(choose_fn, test_trials)
                return {{
                    "combined_score": float(train_ll),
                    "train_loglik": float(train_ll),
                    "test_loglik": float(test_ll),
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                    "train_n": float(train_n),
                    "test_n": float(test_n),
                    "fatal_failure": 0.0,
                }}
            except Exception as e:
                print("[FATAL] evaluator failure:", repr(e), flush=True)
                traceback.print_exc()
                # Make failure unmissable in logs and OpenEvolve metrics.
                return {{
                    "combined_score": -1.0e9,
                    "train_loglik": -1.0e9,
                    "test_loglik": -1.0e9,
                    "train_acc": 0.0,
                    "test_acc": 0.0,
                    "fatal_failure": 1.0,
                }}
        """
    ).strip() + "\n"


def _build_openevolve_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode == "local":
        api_base = args.llm_server_url
        api_key = args.llm_api_key
    else:
        api_base = args.api_base if args.api_base else "https://api.openai.com/v1"
        api_key = args.api_key if args.api_key else "${OPENAI_API_KEY}"

    return {
        "max_iterations": int(args.n_iterations),
        "checkpoint_interval": max(1, min(20, int(args.n_iterations))),
        "log_level": "INFO",
        "diff_based_evolution": False,
        "max_code_length": 20000,
        "llm": {
            "models": [{"name": args.model_name, "weight": 1.0}],
            "evaluator_models": [{"name": args.model_name, "weight": 1.0}],
            "api_base": api_base,
            "api_key": api_key,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096,
            "timeout": int(args.llm_timeout_sec),
            "retries": 2,
            "retry_delay": 3,
        },
        "prompt": {
            "system_message": (
                "You are improving a Choice13k choose(problem, history) policy. "
                "Return valid Python code preserving the choose signature."
            ),
            "evaluator_system_message": "You are a strict code evaluator.",
            "num_top_programs": 3,
            "num_diverse_programs": 2,
            "include_artifacts": True,
        },
        "database": {
            "in_memory": True,
            "log_prompts": True,
            "population_size": max(30, int(args.n_candidates) * 3),
            "archive_size": max(10, int(args.n_candidates)),
            "num_islands": 1,
            "migration_interval": 1000,
            "migration_rate": 0.1,
            "elite_selection_ratio": 0.2,
            "exploration_ratio": 0.2,
            "exploitation_ratio": 0.6,
            "feature_dimensions": ["complexity", "diversity"],
            "feature_bins": 10,
        },
        "evaluator": {
            "timeout": int(args.eval_timeout_sec),
            "max_retries": 0,
            "cascade_evaluation": False,
            "parallel_evaluations": 1,
            "use_llm_feedback": False,
            "llm_feedback_weight": 0.0,
        },
        "evolution_trace": {"enabled": False},
    }


def _run_one_openevolve(
    *,
    seed_code: str,
    train_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
    participant_tag: str,
    run_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    _ensure_openevolve_importable()
    from openevolve import OpenEvolve
    from openevolve.config import load_config

    part_dir = run_root / f"participant_{participant_tag}"
    part_dir.mkdir(parents=True, exist_ok=True)

    initial_program_path = part_dir / "initial_program.py"
    evaluator_path = part_dir / "evaluator.py"
    data_path = part_dir / "dataset.json"
    config_path = part_dir / "config.yaml"

    initial_program_path.write_text(seed_code, encoding="utf-8")
    data_payload = {
        "train_trials": _to_builtin(train_trials),
        "test_trials": _to_builtin(test_trials),
    }
    data_path.write_text(json.dumps(data_payload), encoding="utf-8")
    evaluator_path.write_text(_build_participant_evaluator_code(data_path), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(_build_openevolve_config_dict(args), sort_keys=False), encoding="utf-8")

    config = load_config(str(config_path))
    output_dir = part_dir / "openevolve_output"
    controller = OpenEvolve(
        initial_program_path=str(initial_program_path),
        evaluation_file=str(evaluator_path),
        config=config,
        output_dir=str(output_dir),
    )

    t0 = datetime.now()
    best_program = asyncio.run(controller.run(iterations=args.n_iterations))
    runtime = (datetime.now() - t0).total_seconds()

    if best_program is None:
        raise RuntimeError(f"[FATAL] OpenEvolve returned no best program for participant {participant_tag}.")

    best_code_path = part_dir / "best_program.py"
    best_code_path.write_text(best_program.code, encoding="utf-8")

    metrics = dict(best_program.metrics or {})
    train_ll = float(metrics.get("train_loglik", -1e9))
    test_ll = float(metrics.get("test_loglik", -1e9))
    train_acc = float(metrics.get("train_acc", 0.0))
    test_acc = float(metrics.get("test_acc", 0.0))
    fatal_flag = float(metrics.get("fatal_failure", 0.0))

    hard_fail = fatal_flag >= 0.5 or train_ll <= -1e8 or test_ll <= -1e8
    if hard_fail:
        msg = (
            f"[FATAL] participant {participant_tag} evolution failed. "
            f"metrics={metrics} output_dir={output_dir}"
        )
        print(msg, flush=True)
        if not args.allow_failure:
            raise RuntimeError(msg)

    print(
        f"[INFO] participant {participant_tag}: train_loglik={train_ll:.6f}, "
        f"test_loglik={test_ll:.6f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
        f"fatal_failure={fatal_flag:.1f}"
    )

    return {
        "participant_id": participant_tag,
        "train_loglik": train_ll,
        "test_loglik": test_ll,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_fitness": train_ll,
        "test_fitness": test_ll,
        "seed_program_train_fitness": train_ll,
        "seed_program_test_fitness": test_ll,
        "total_runtime": runtime,
        "fatal_failure": fatal_flag,
        "metrics_raw": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEvolve baseline on Choice13k (TE-compatible participant scope).")
    parser.add_argument("--dataset", type=str, default="choice13k", choices=["choice13k"])
    parser.add_argument("--seed_path", type=str, required=True, help="Path to seed Python program containing choose().")
    parser.add_argument("--n_iterations", type=int, default=20)
    parser.add_argument("--n_candidates", type=int, default=10, help="Mapped to OpenEvolve population sizing.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "default"])
    parser.add_argument("--llm_server_url", type=str, default=os.getenv("VLLM_LOCAL_URL", "http://localhost:8000/v1"))
    parser.add_argument("--llm_api_key", type=str, default=os.getenv("VLLM_LOCAL_API_KEY", "EMPTY"))
    parser.add_argument("--api_base", type=str, default=None, help="Used only when --mode default.")
    parser.add_argument("--api_key", type=str, default=None, help="Used only when --mode default.")
    parser.add_argument("--participant_scope", type=str, default="single", choices=["single", "range", "all"])
    parser.add_argument("--single_participant_id", type=int, default=0)
    parser.add_argument("--range_start_ordinal", type=int, default=None)
    parser.add_argument("--range_end_ordinal", type=int, default=None)
    parser.add_argument("--all_max_participants", type=int, default=None)
    parser.add_argument("--filter_mixed_gambles", action="store_true", default=False)
    parser.add_argument("--fitness_metric", type=str, default="loglik", choices=["loglik"])
    parser.add_argument("--split_mode", type=str, default="within_participant", choices=["within_participant", "across_participants"])
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--allow_failure", action="store_true", help="Continue run after a participant-level fatal failure.")
    parser.add_argument("--eval_timeout_sec", type=int, default=300)
    parser.add_argument("--llm_timeout_sec", type=int, default=120)
    args = parser.parse_args()

    if not (0.0 < args.split_ratio < 1.0):
        print("Error: --split_ratio must be in (0,1).")
        sys.exit(1)
    if args.split_mode == "across_participants" and args.participant_scope == "single":
        print("Error: across_participants needs at least two participants; use range or all.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_run_dir = Path(
        args.output_dir
        if args.output_dir
        else str(REPO_ROOT / "generated_outputs" / "choice13k" / "openevolve" / f"run_{timestamp}")
    )
    base_run_dir.mkdir(parents=True, exist_ok=True)
    cmd_log = _write_command_line_log(base_run_dir)
    print(f"Wrote full command line to {cmd_log}")

    seed_path = Path(args.seed_path)
    if not seed_path.is_file():
        print(f"Error: --seed_path not found: {seed_path}")
        sys.exit(1)
    seed_code = seed_path.read_text(encoding="utf-8")
    if "def choose(" not in seed_code:
        print(f"Error: seed program must define choose(problem, history): {seed_path}")
        sys.exit(1)

    mixed = bool(args.filter_mixed_gambles)
    try:
        participants = resolve_participants_for_scope(
            dataset=args.dataset,
            repo_root=REPO_ROOT,
            participant_scope=args.participant_scope,
            single_participant_id=args.single_participant_id,
            range_start_ordinal=args.range_start_ordinal,
            range_end_ordinal=args.range_end_ordinal,
            all_max_participants=args.all_max_participants,
            filter_mixed_gambles=mixed,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = f"choice13k_openevolve_{timestamp}_{args.participant_scope}"
            wandb.init(project="ROTE_evo", name=run_name, config=vars(args), reinit=False)
        except Exception as e:
            print(f"[WARN] wandb logging disabled: {e}")
            wandb = None

    try:
        # across_participants: one pooled run
        if args.split_mode == "across_participants":
            if len(participants) < 2:
                raise RuntimeError("across_participants requires >=2 selected participants.")
            rng = np.random.default_rng(args.split_seed)
            shuffled = list(participants)
            rng.shuffle(shuffled)
            split_idx = int(len(shuffled) * args.split_ratio)
            split_idx = max(1, min(split_idx, len(shuffled) - 1))
            train_p = shuffled[:split_idx]
            test_p = shuffled[split_idx:]

            max_pid = max(participants)
            experiments = get_choice13k_experiments(n_participants=max_pid + 1)
            train_trials: List[Dict[str, Any]] = []
            test_trials: List[Dict[str, Any]] = []
            for pid in train_p:
                tr, _ = experiment_to_trials(experiments[pid])
                train_trials.extend(tr)
            for pid in test_p:
                tr, _ = experiment_to_trials(experiments[pid])
                test_trials.extend(tr)

            print(f"[INFO] Across-participants trials: train={len(train_trials)}, test={len(test_trials)}")
            result = _run_one_openevolve(
                seed_code=seed_code,
                train_trials=train_trials,
                test_trials=test_trials,
                participant_tag="0",
                run_root=base_run_dir,
                args=args,
            )
            row = {
                "participant_id": 0,
                "train_fitness": result["train_loglik"],
                "test_fitness": result["test_loglik"],
                "total_runtime": result["total_runtime"],
                "seed_program_train_fitness": result["train_loglik"],
                "seed_program_test_fitness": result["test_loglik"],
            }
            row_ll = {"participant_id": 0, "train_loglik": result["train_loglik"], "test_loglik": result["test_loglik"]}
            _write_all_mode_csvs(base_run_dir, [row], [row_ll])
            print(f"Wrote CSVs under {base_run_dir}")
            return

        # all mode: all-mode csvs
        if args.participant_scope == "all":
            participant_details: List[Dict[str, Any]] = []
            participant_loglik: List[Dict[str, Any]] = []
            for pid in tqdm(participants, desc="Participants"):
                experiments = get_choice13k_experiments(n_participants=pid + 1)
                exp = experiments[pid]
                train_trials, test_trials, _ = split_trials(exp, split_ratio=args.split_ratio, split_seed=args.split_seed)
                result = _run_one_openevolve(
                    seed_code=seed_code,
                    train_trials=train_trials,
                    test_trials=test_trials,
                    participant_tag=str(pid),
                    run_root=base_run_dir,
                    args=args,
                )
                participant_details.append(
                    {
                        "participant_id": pid,
                        "train_fitness": result["train_fitness"],
                        "test_fitness": result["test_fitness"],
                        "total_runtime": result["total_runtime"],
                        "seed_program_train_fitness": result["seed_program_train_fitness"],
                        "seed_program_test_fitness": result["seed_program_test_fitness"],
                    }
                )
                participant_loglik.append(
                    {"participant_id": pid, "train_loglik": result["train_loglik"], "test_loglik": result["test_loglik"]}
                )
                _write_all_mode_csvs(base_run_dir, participant_details, participant_loglik)
                if wandb is not None:
                    wandb.log(
                        {
                            "participant_id": pid,
                            "train_loglik": result["train_loglik"],
                            "test_loglik": result["test_loglik"],
                            "train_acc": result["train_acc"],
                            "test_acc": result["test_acc"],
                            "fatal_failure": result["fatal_failure"],
                        }
                    )
            print(f"Wrote CSVs under {base_run_dir}")
            return

        # single / range mode: participants_summary + loglik files
        participants_summary: List[Dict[str, Any]] = []
        participants_loglik_summary: List[Dict[str, Any]] = []
        summary_file = base_run_dir / "participants_summary.csv"
        summary_loglik_file = base_run_dir / "summary_loglik.csv"
        details_loglik_file = base_run_dir / "participant_details_loglik.csv"

        for pid in tqdm(participants, desc="Participants"):
            experiments = get_choice13k_experiments(n_participants=pid + 1)
            exp = experiments[pid]
            train_trials, test_trials, _ = split_trials(exp, split_ratio=args.split_ratio, split_seed=args.split_seed)
            result = _run_one_openevolve(
                seed_code=seed_code,
                train_trials=train_trials,
                test_trials=test_trials,
                participant_tag=str(pid),
                run_root=base_run_dir,
                args=args,
            )
            summ = {
                "participant_id": pid,
                "train_acc": result["train_acc"],
                "test_acc": result["test_acc"],
                "train_loglik": result["train_loglik"],
                "test_loglik": result["test_loglik"],
                "train_fitness": result["train_fitness"],
                "test_fitness": result["test_fitness"],
                "seed_program_train_fitness": result["seed_program_train_fitness"],
                "seed_program_test_fitness": result["seed_program_test_fitness"],
            }
            participants_summary.append(summ)
            participants_loglik_summary.append(
                {"participant_id": pid, "train_loglik": result["train_loglik"], "test_loglik": result["test_loglik"]}
            )

            with open(summary_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summ.keys()))
                w.writeheader()
                w.writerows(_round_floats_for_csv_rows(participants_summary))

            with open(details_loglik_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["participant_id", "train_loglik", "test_loglik"])
                w.writeheader()
                w.writerows(_round_floats_for_csv_rows(participants_loglik_summary))

            tr_vals = [d["train_loglik"] for d in participants_loglik_summary if d["train_loglik"] is not None]
            te_vals = [d["test_loglik"] for d in participants_loglik_summary if d["test_loglik"] is not None]
            with open(summary_loglik_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["num_of_participants", "avg_train_loglik", "avg_test_loglik"]
                )
                w.writeheader()
                w.writerow(
                    _round_floats_for_csv_row(
                        {
                            "num_of_participants": len(participants_loglik_summary),
                            "avg_train_loglik": float(np.mean(tr_vals)) if tr_vals else None,
                            "avg_test_loglik": float(np.mean(te_vals)) if te_vals else None,
                        }
                    )
                )

            if wandb is not None:
                wandb.log(
                    {
                        "participant_id": pid,
                        "train_loglik": result["train_loglik"],
                        "test_loglik": result["test_loglik"],
                        "train_acc": result["train_acc"],
                        "test_acc": result["test_acc"],
                        "fatal_failure": result["fatal_failure"],
                    }
                )

        print(f"Wrote {summary_file} and loglik summaries under {base_run_dir}")
    finally:
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
