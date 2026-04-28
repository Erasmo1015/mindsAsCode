"""
OpenEvolve baseline runner for Choice13k/CPC18/Mixed Gambles with TE-compatible participant selection.

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
import re
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
from data_modules.cpc18 import load_cpc18_track2_data, split_cpc18_trials  # noqa: E402


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


def _patch_openevolve_for_gemma_system_role() -> None:
    """
    Runtime patch in wrapper layer only:
    for Gemma models, merge system prompt into first user message and avoid
    sending a separate system-role message. Non-Gemma models are unchanged.
    """
    import openevolve.llm.openai as oe_openai  # type: ignore

    if getattr(oe_openai.OpenAILLM, "_mindsascode_gemma_patch", False):
        return

    original_call_api = oe_openai.OpenAILLM._call_api

    def _extract_context_limits(exc: Exception) -> Optional[Tuple[int, int]]:
        """
        Parse context-overflow details from vLLM/OpenAI-style errors.
        Returns (max_context, input_tokens) when available.
        """
        text = str(exc)
        m = re.search(
            r"maximum context length is (\d+) tokens.*prompt contains at least (\d+) input tokens",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _normalize_gemma_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        system_chunks: List[str] = []
        for msg in messages:
            role = str(msg.get("role", "user")).lower()
            content = str(msg.get("content", ""))
            if role == "system":
                if content.strip():
                    system_chunks.append(content)
                continue
            if role == "assistant":
                normalized.append({"role": "model", "content": content})
                continue
            normalized.append({"role": str(msg.get("role", "user")), "content": content})

        merged_system = "\n\n".join(system_chunks).strip()
        if merged_system:
            first_user_idx = next(
                (i for i, m in enumerate(normalized) if str(m.get("role", "")).lower() == "user"),
                None,
            )
            if first_user_idx is None:
                normalized.insert(0, {"role": "user", "content": merged_system})
            else:
                first_content = str(normalized[first_user_idx].get("content", ""))
                normalized[first_user_idx]["content"] = (
                    f"{merged_system}\n\n{first_content}" if first_content else merged_system
                )
        return normalized

    async def _patched_call_api(self, params):
        model_name = str(getattr(self, "model", "")).lower()
        if "gemma" not in model_name:
            return await original_call_api(self, params)

        patched_params = dict(params)
        orig_messages = patched_params.get("messages", [])
        if isinstance(orig_messages, list):
            final_messages = _normalize_gemma_messages(orig_messages)
            final_roles = [str(m.get("role", "")).lower() for m in final_messages]
            print(f"[GemmaCompat] final message roles: {final_roles}", flush=True)
            assert all(r not in {"system", "assistant"} for r in final_roles), (
                f"Gemma payload contains invalid roles: {final_roles}"
            )
            patched_params["messages"] = final_messages
        request_params = dict(patched_params)
        for _ in range(4):
            try:
                return await original_call_api(self, request_params)
            except Exception as exc:
                parsed = _extract_context_limits(exc)
                if parsed is None:
                    raise
                max_context, input_tokens = parsed
                # Larger buffer to survive tokenization drift between attempts.
                # Allow very small outputs in emergency mode to avoid hard failure loops.
                safe_max_tokens = max(32, max_context - input_tokens - 256)
                current_max_tokens = int(
                    request_params.get("max_tokens", getattr(self, "max_tokens", 512))
                )
                if safe_max_tokens >= current_max_tokens:
                    raise
                request_params = dict(request_params)
                request_params["max_tokens"] = safe_max_tokens
                print(
                    f"[GemmaCompat] context overflow detected; retrying with max_tokens="
                    f"{safe_max_tokens} (was {current_max_tokens}, input_tokens={input_tokens}, "
                    f"max_context={max_context})",
                    flush=True,
                )
        return await original_call_api(self, request_params)

    oe_openai.OpenAILLM._call_api = _patched_call_api
    oe_openai.OpenAILLM._mindsascode_gemma_patch = True


def _patch_openevolve_for_minimal_prompt() -> None:
    """
    Wrapper-layer runtime patch:
    build a compact mutation prompt that keeps only essential context:
    - objective/system instruction
    - current program
    - top parent programs (num_top_programs)
    This avoids OpenEvolve template overhead (history blocks, repeated metrics prose,
    artifact rendering) that frequently causes context overflow.
    """
    import openevolve.prompt.sampler as oe_sampler  # type: ignore

    if getattr(oe_sampler.PromptSampler, "_mindsascode_min_prompt_patch", False):
        return

    def _resolve_system_message(self) -> str:
        if getattr(self, "system_template_override", None):
            key = str(self.system_template_override)
            try:
                return str(self.template_manager.get_template(key))
            except Exception:
                return key
        msg = str(getattr(self.config, "system_message", "") or "")
        if msg in getattr(self.template_manager, "templates", {}):
            try:
                return str(self.template_manager.get_template(msg))
            except Exception:
                return msg
        return msg

    def _patched_build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Any]] = None,
        feature_dimensions: Optional[List[str]] = None,
        current_changes_description: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        del parent_program, previous_programs, inspirations, evolution_round
        del diff_based_evolution, template_key, program_artifacts
        del feature_dimensions, current_changes_description, kwargs

        system_message = _resolve_system_message(self).strip()
        if not system_message:
            system_message = (
                "You are improving choose(problem, history). "
                "Return only runnable Python code."
            )

        n_top = int(getattr(self.config, "num_top_programs", 3))
        selected_top = list(top_programs[: max(0, n_top)])

        lines: List[str] = []
        lines.append("# Task")
        lines.append(
            "Improve the policy while preserving the exact signature `choose(problem, history)`."
        )
        if isinstance(program_metrics, dict):
            score = program_metrics.get("combined_score", None)
            if isinstance(score, (int, float)):
                lines.append(f"Current combined_score: {float(score):.6f}")
        lines.append("")

        if selected_top:
            lines.append("# Parent Programs")
            for i, prog in enumerate(selected_top, start=1):
                code = str(prog.get("code", "") or "").strip()
                if not code:
                    continue
                lines.append(f"## Parent {i}")
                lines.append(f"```{language}")
                lines.append(code)
                lines.append("```")
                lines.append("")

        lines.append("# Current Program")
        lines.append(f"```{language}")
        lines.append(str(current_program or ""))
        lines.append("```")
        lines.append("")

        lines.append("# Output Rules")
        lines.append("- Output only complete runnable Python code.")
        lines.append("- Do not include markdown fences or explanations.")
        lines.append("- Keep `def choose(problem, history):`.")

        user_message = "\n".join(lines).strip() + "\n"
        return {"system": system_message, "user": user_message}

    oe_sampler.PromptSampler.build_prompt = _patched_build_prompt
    oe_sampler.PromptSampler._mindsascode_min_prompt_patch = True


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


def load_mixed_gambles_trials(
    csv_path: str,
    participant_id: int,
    *,
    filter_gain_loss_only: bool,
    split_ratio: float,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    option_keys = [0, 1]  # 0 = gamble, 1 = certain
    all_trials: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["subject"]) != participant_id:
                continue
            if filter_gain_loss_only and row.get("gamble_type") != "gain_loss":
                continue
            gain, loss, cert = float(row["gain"]), float(row["loss"]), float(row["cert"])
            took_gamble = int(row["took_gamble"])
            action = 1 - took_gamble
            all_trials.append(
                {
                    "problem": {
                        "gamble_A": {"rewards": [gain, loss], "probs": [0.5, 0.5]},
                        "gamble_B": {"rewards": [cert], "probs": [1.0]},
                        "option_keys": option_keys,
                        "has_feedback": False,
                    },
                    "history": [],
                    "options": option_keys,
                    "action": action,
                    "problem_signature": (gain, loss, cert),
                }
            )

    if len(all_trials) == 0:
        raise ValueError(f"No rows found for subject {participant_id} in {csv_path}")

    signatures = sorted({t["problem_signature"] for t in all_trials})
    if len(signatures) < 2:
        raise ValueError(
            f"mixed_gambles participant {participant_id} has <2 unique problems; cannot build disjoint train/test."
        )
    rng = np.random.default_rng(int(split_seed))
    shuffled = list(signatures)
    rng.shuffle(shuffled)
    split_point = int(len(shuffled) * float(split_ratio))
    split_point = max(1, min(split_point, len(shuffled) - 1))
    train_sigs = set(shuffled[:split_point])
    test_sigs = set(shuffled[split_point:])
    train_trials = [t for t in all_trials if t["problem_signature"] in train_sigs]
    test_trials = [t for t in all_trials if t["problem_signature"] in test_sigs]
    for t in train_trials:
        t.pop("problem_signature", None)
    for t in test_trials:
        t.pop("problem_signature", None)
    return train_trials, test_trials, option_keys


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


def _compile_choose_from_code(code: str):
    namespace: Dict[str, Any] = {}
    exec(compile(code, "<openevolve-best-program>", "exec"), namespace, namespace)
    choose_fn = namespace.get("choose", None)
    if choose_fn is None:
        raise AttributeError("Candidate program must define choose(problem, history).")
    return choose_fn


def _eval_trials_loglik(choose_fn, trials: List[Dict[str, Any]]) -> Tuple[float, float, int]:
    total = len(trials)
    if total == 0:
        raise ValueError("No trials provided to evaluator.")
    ll_sum = 0.0
    correct = 0
    for i, t in enumerate(trials):
        y = int(t["action"])
        p_raw = choose_fn(t["problem"], t["history"])
        if isinstance(p_raw, (bool, int)) and int(p_raw) in (0, 1):
            p_raw = 1.0 if int(p_raw) == 1 else 0.0
        if not isinstance(p_raw, float):
            raise TypeError(f"trial={i} expected float prob, got {type(p_raw)}")
        if not (0.0 <= p_raw <= 1.0):
            raise ValueError(f"trial={i} probability out of [0,1]: {p_raw}")
        p = min(max(float(p_raw), 1e-9), 1.0 - 1e-9)
        ll_sum += y * math.log(p) + (1 - y) * math.log(1.0 - p)
        pred = 1 if p_raw >= 0.5 else 0
        correct += int(pred == y)
    return (ll_sum / total), (correct / total), total


def _eval_action_trials(choose_fn, trials: List[Dict[str, Any]]) -> Tuple[float, int]:
    total = len(trials)
    if total == 0:
        raise ValueError("No trials provided to evaluator.")
    correct = 0
    for t in trials:
        try:
            pred = choose_fn(t["problem"], t["history"])
        except Exception:
            pred = None
        if pred is not None and int(pred) == int(t["action"]):
            correct += 1
    return (correct / total), total


def _eval_cpc18_mse(
    choose_fn,
    trials: List[Dict[str, Any]],
    observed_blocks: Dict[str, Any],
) -> Tuple[float, bool]:
    problems: Dict[int, List[Dict[str, Any]]] = {}
    for tr in trials:
        pid = int(tr["problem_id"])
        problems.setdefault(pid, []).append(tr)

    all_mse: List[float] = []
    for problem_id, problem_trials in problems.items():
        if str(problem_id) in observed_blocks:
            obs_rates = observed_blocks[str(problem_id)]
        elif problem_id in observed_blocks:
            obs_rates = observed_blocks[problem_id]
        else:
            continue

        blocks: Dict[int, List[Dict[str, Any]]] = {}
        for tr in problem_trials:
            bid = int(tr["block_id"])
            blocks.setdefault(bid, []).append(tr)

        pred_rates = [0.0] * 5
        for block_id in range(1, 6):
            block_trials = blocks.get(block_id, [])
            if not block_trials:
                continue
            preds: List[int] = []
            for tr in block_trials:
                try:
                    pred = choose_fn(tr["problem"], tr["history"])
                    preds.append(int(pred == 1))
                except Exception:
                    continue
            if not preds:
                return float("inf"), False
            pred_rates[block_id - 1] = sum(preds) / len(preds)
        mse = 100.0 * sum((float(pred_rates[i]) - float(obs_rates[i])) ** 2 for i in range(5)) / 5.0
        all_mse.append(float(mse))

    if not all_mse:
        return float("inf"), False
    return float(sum(all_mse) / len(all_mse)), True


def _evaluate_program_code_metrics(
    *,
    code: str,
    dataset: str,
    fitness_metric: str,
    cpc18_official_mse: bool,
    train_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
    observed_blocks: Optional[Dict[str, Any]],
    n_eval_seeds: int,
) -> Dict[str, Any]:
    choose_fn = _compile_choose_from_code(code)
    n_rep = max(1, int(n_eval_seeds))
    metrics: Dict[str, Any] = {"fatal_failure": 0.0}
    if dataset in {"choice13k", "mixed_gambles"}:
        train_lls: List[float] = []
        test_lls: List[float] = []
        train_accs: List[float] = []
        test_accs: List[float] = []
        for _ in range(n_rep):
            train_ll, train_acc, train_n = _eval_trials_loglik(choose_fn, train_trials)
            test_ll, test_acc, test_n = _eval_trials_loglik(choose_fn, test_trials)
            train_lls.append(float(train_ll))
            test_lls.append(float(test_ll))
            train_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
        metrics.update(
            {
                "train_loglik": float(sum(train_lls) / len(train_lls)),
                "test_loglik": float(sum(test_lls) / len(test_lls)),
                "train_acc": float(sum(train_accs) / len(train_accs)),
                "test_acc": float(sum(test_accs) / len(test_accs)),
                "train_n": float(train_n),
                "test_n": float(test_n),
            }
        )
        metrics["combined_score"] = (
            float(metrics["train_loglik"]) if fitness_metric == "loglik" else float(metrics["train_acc"])
        )
        return metrics

    if dataset == "cpc18" and cpc18_official_mse:
        train_accs = []
        test_accs = []
        train_mses = []
        test_mses = []
        obs = observed_blocks or {}
        for _ in range(n_rep):
            train_acc, train_n = _eval_action_trials(choose_fn, train_trials)
            test_acc, test_n = _eval_action_trials(choose_fn, test_trials)
            train_mse, train_ok = _eval_cpc18_mse(choose_fn, train_trials, obs)
            test_mse, test_ok = _eval_cpc18_mse(choose_fn, test_trials, obs)
            if (not train_ok) or (not test_ok):
                raise RuntimeError("Invalid CPC18 MSE evaluation (missing/invalid block predictions).")
            train_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
            train_mses.append(float(train_mse))
            test_mses.append(float(test_mse))
        metrics.update(
            {
                "train_loglik": None,
                "test_loglik": None,
                "train_acc": float(sum(train_accs) / len(train_accs)),
                "test_acc": float(sum(test_accs) / len(test_accs)),
                "train_mse": float(sum(train_mses) / len(train_mses)),
                "test_mse": float(sum(test_mses) / len(test_mses)),
                "train_n": float(train_n),
                "test_n": float(test_n),
            }
        )
        metrics["combined_score"] = -float(metrics["train_mse"])
        return metrics

    if dataset == "cpc18" and (not cpc18_official_mse):
        train_lls = []
        test_lls = []
        train_accs = []
        test_accs = []
        for _ in range(n_rep):
            train_ll, train_acc, train_n = _eval_trials_loglik(choose_fn, train_trials)
            test_ll, test_acc, test_n = _eval_trials_loglik(choose_fn, test_trials)
            train_lls.append(float(train_ll))
            test_lls.append(float(test_ll))
            train_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
        metrics.update(
            {
                "train_loglik": float(sum(train_lls) / len(train_lls)),
                "test_loglik": float(sum(test_lls) / len(test_lls)),
                "train_acc": float(sum(train_accs) / len(train_accs)),
                "test_acc": float(sum(test_accs) / len(test_accs)),
                "train_mse": 0.0,
                "test_mse": 0.0,
                "train_n": float(train_n),
                "test_n": float(test_n),
            }
        )
        metrics["combined_score"] = (
            float(metrics["train_loglik"]) if fitness_metric == "loglik" else float(metrics["train_acc"])
        )
        return metrics

    train_accs = []
    test_accs = []
    for _ in range(n_rep):
        train_acc, train_n = _eval_action_trials(choose_fn, train_trials)
        test_acc, test_n = _eval_action_trials(choose_fn, test_trials)
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
    metrics.update(
        {
            "train_loglik": None,
            "test_loglik": None,
            "train_acc": float(sum(train_accs) / len(train_accs)),
            "test_acc": float(sum(test_accs) / len(test_accs)),
            "train_n": float(train_n),
            "test_n": float(test_n),
            "combined_score": float(sum(train_accs) / len(train_accs)),
        }
    )
    return metrics


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


def _format_choice13k_train_trials_for_prompt(
    trials: List[Dict[str, Any]],
    *,
    max_trials: int,
    rng_seed: int,
    max_history_items_per_trial: int,
) -> str:
    """
    Build human-readable train-split text for OpenEvolve mutation prompts (mirrors TE-style summaries).

    Prompt summary for mutation. Keep compact to stay within context limits.
    """
    if not trials:
        return "(no train trials)\n"

    n = len(trials)
    if max_trials > 0 and n > max_trials:
        rng = np.random.default_rng(int(rng_seed))
        perm = rng.permutation(n)
        order = perm[:max_trials].tolist()
    else:
        order = list(range(n))

    lines = [
        "Choice13k TRAIN split — reference for improving choose(problem, history).",
        "Fitness uses mean Bernoulli log-likelihood of P(action=1) on these trials only.",
        "action 0 = option A, action 1 = option B.",
        "",
    ]
    for j, i in enumerate(order):
        t = trials[i]
        prob = t["problem"]
        prob_a = prob["gamble_A"]["probs"]
        rew_a = prob["gamble_A"]["rewards"]
        prob_b = prob["gamble_B"]["probs"]
        rew_b = prob["gamble_B"]["rewards"]
        has_fb = prob.get("has_feedback", False)
        action = t["action"]
        lines.append(f"--- Train trial {j + 1} (index {i} in train split) ---")
        lines.append(
            f"Problem: Option A probs {prob_a} rewards {rew_a}; "
            f"Option B probs {prob_b} rewards {rew_b}; has_feedback={has_fb}"
        )
        lines.append(f"Observed human action: {action}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_cpc18_train_trials_for_prompt(
    trials: List[Dict[str, Any]],
    *,
    max_trials: int,
    rng_seed: int,
    max_history_items_per_trial: int,
) -> str:
    if not trials:
        return "(no train trials)\n"

    n = len(trials)
    if max_trials > 0 and n > max_trials:
        rng = np.random.default_rng(int(rng_seed))
        order = rng.permutation(n)[:max_trials].tolist()
    else:
        order = list(range(n))

    lines = [
        "CPC18 TRAIN trials — improve choose(problem, history).",
        "Primary fitness is block-level MSE (lower is better); OpenEvolve maximizes combined_score = -train_mse.",
        "action 0 = option A (L), action 1 = option B (R).",
        "",
    ]
    for j, i in enumerate(order):
        t = trials[i]
        p = t["problem"]
        lines.append(f"--- Train trial {j + 1} (index {i}) ---")
        lines.append(
            "Problem: "
            f"Ha={p['Ha']} pHa={p['pHa']} La={p['La']} | "
            f"Hb={p['Hb']} pHb={p['pHb']} Lb={p['Lb']} | "
            f"Amb={p['Amb']} Corr={p['Corr']}"
        )
        lines.append(f"Observed human action: {int(t['action'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_mixed_gambles_train_trials_for_prompt(
    trials: List[Dict[str, Any]],
    *,
    fitness_metric: str,
    max_trials: int,
    rng_seed: int,
    max_history_items_per_trial: int,  # kept for shared signature
) -> str:
    del max_history_items_per_trial
    if not trials:
        return "(no train trials)\n"

    n = len(trials)
    if max_trials > 0 and n > max_trials:
        rng = np.random.default_rng(int(rng_seed))
        order = rng.permutation(n)[:max_trials].tolist()
    else:
        order = list(range(n))

    if fitness_metric == "loglik":
        objective_line = (
            "Fitness uses mean Bernoulli log-likelihood of P(action=1) on train trials "
            "(higher is better)."
        )
    else:
        objective_line = "Optimize train accuracy (action 0/1)."
    lines = [
        "Mixed Gambles TRAIN trials — improve choose(problem, history).",
        objective_line,
        "action 0 = gamble option, action 1 = certain option.",
        "",
    ]
    for j, i in enumerate(order):
        t = trials[i]
        p = t["problem"]
        lines.append(f"--- Train trial {j + 1} (index {i}) ---")
        lines.append(
            f"Problem: gamble rewards={p['gamble_A']['rewards']} probs={p['gamble_A']['probs']}; "
            f"certain rewards={p['gamble_B']['rewards']} probs={p['gamble_B']['probs']}"
        )
        lines.append(f"Observed human action: {int(t['action'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_train_trials_prompt_file(
    *,
    dataset: str,
    fitness_metric: str,
    path: Path,
    train_trials: List[Dict[str, Any]],
    max_prompt_train_trials: int,
    prompt_train_trials_seed: int,
    max_history_items_per_trial: int,
    max_prompt_trials_per_problem: int,
) -> None:
    if max_prompt_trials_per_problem > 0:
        by_problem: Dict[Any, List[Dict[str, Any]]] = {}
        for t in train_trials:
            if "problem_id" in t:
                k = ("problem_id", t["problem_id"])
            else:
                p = t.get("problem", {})
                ga = p.get("gamble_A", {})
                gb = p.get("gamble_B", {})
                ga_probs = ga.get("probs", [])
                gb_probs = gb.get("probs", [])
                if ga_probs is None:
                    ga_probs = []
                if gb_probs is None:
                    gb_probs = []
                k = (
                    "problem_sig",
                    tuple(ga.get("rewards", [])),
                    tuple(ga_probs),
                    tuple(gb.get("rewards", [])),
                    tuple(gb_probs),
                )
            by_problem.setdefault(k, []).append(t)
        capped: List[Dict[str, Any]] = []
        for _, rows in by_problem.items():
            capped.extend(rows[:max_prompt_trials_per_problem])
        train_trials = capped

    if dataset == "choice13k":
        body = _format_choice13k_train_trials_for_prompt(
            train_trials,
            max_trials=max_prompt_train_trials,
            rng_seed=prompt_train_trials_seed,
            max_history_items_per_trial=max_history_items_per_trial,
        )
    elif dataset == "cpc18":
        body = _format_cpc18_train_trials_for_prompt(
            train_trials,
            max_trials=max_prompt_train_trials,
            rng_seed=prompt_train_trials_seed,
            max_history_items_per_trial=max_history_items_per_trial,
        )
    elif dataset == "mixed_gambles":
        body = _format_mixed_gambles_train_trials_for_prompt(
            train_trials,
            fitness_metric=fitness_metric,
            max_trials=max_prompt_train_trials,
            rng_seed=prompt_train_trials_seed,
            max_history_items_per_trial=max_history_items_per_trial,
        )
    else:
        raise ValueError(f"Unsupported dataset for prompt formatting: {dataset!r}")
    path.write_text(body, encoding="utf-8")


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


def _write_hyperparameters_log(run_dir: Path, args: argparse.Namespace) -> Path:
    """Record CLI and effective OpenEvolve config under run_dir/log/ (repro / comparison)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "log"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "hyperparameters.yaml"
    payload: Dict[str, Any] = {
        "meta": {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "cwd": os.getcwd(),
            "host": socket.gethostname(),
            "python": sys.executable,
            "script": str(Path(__file__).resolve()),
        },
        "cli": _to_builtin(vars(args)),
        "openevolve_config": _to_builtin(_build_openevolve_config_dict(args)),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _build_participant_evaluator_code(
    dataset_json_path: Path,
    train_trials_prompt_path: Path,
    dataset: str,
    fitness_metric: str,
    n_eval_seeds: int,
) -> str:
    return textwrap.dedent(
        f"""
        import ast
        import importlib.util
        import json
        import math
        import re
        import traceback
        from pathlib import Path

        from openevolve.evaluation_result import EvaluationResult

        DATA_PATH = Path(r\"{str(dataset_json_path)}\")
        TRAIN_PROMPT_PATH = Path(r\"{str(train_trials_prompt_path)}\")
        DATASET = {dataset!r}
        FITNESS_METRIC = {fitness_metric!r}
        N_EVAL_SEEDS = {int(n_eval_seeds)}


        def _extract_python_candidate(text: str) -> str:
            norm = (
                text.replace("’", "'")
                .replace("“", '"')
                .replace("”", '"')
            )
            m = re.search(r"```(?:python)?\\s*(.*?)```", norm, flags=re.IGNORECASE | re.DOTALL)
            if m:
                candidate = m.group(1).strip()
            else:
                candidate = norm.strip()

            # Handle truncated/unterminated fenced outputs.
            candidate = re.sub(r"^```(?:python)?\\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\\s*```\\s*$", "", candidate, flags=re.IGNORECASE)

            idx = candidate.find("def choose(")
            if idx >= 0:
                candidate = candidate[idx:]
            else:
                idx2 = norm.find("def choose(")
                if idx2 >= 0:
                    candidate = norm[idx2:].strip()
            return candidate


        def _guard_candidate_file(program_path: str) -> str:
            def _choose_has_explicit_final_return(src: str) -> bool:
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    return False
                choose_node = None
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "choose":
                        choose_node = node
                        break
                if choose_node is None or not choose_node.body:
                    return False
                # Conservative check: require an explicit trailing return in choose().
                return isinstance(choose_node.body[-1], ast.Return)

            def _append_choose_fallback_return(src: str) -> str:
                lines = src.splitlines()
                def_line_idx = None
                def_indent = ""
                for i, line in enumerate(lines):
                    stripped = line.lstrip()
                    if stripped.startswith("def choose("):
                        def_line_idx = i
                        def_indent = line[: len(line) - len(stripped)]
                        break
                if def_line_idx is None:
                    return src

                body_indent = def_indent + "    "
                fallback_line = body_indent + "return 0.5"
                # Avoid duplicate fallback if already present at end.
                if lines and lines[-1].strip() == "return 0.5":
                    return src
                if lines and lines[-1].strip() != "":
                    lines.append("")
                lines.append(fallback_line)
                return "\\n".join(lines).rstrip() + "\\n"

            def _salvage_by_trimming_tail(src: str) -> str | None:
                lines = src.splitlines()
                # Trim broken trailing fragments until compile succeeds.
                for keep in range(len(lines), 0, -1):
                    trial = "\\n".join(lines[:keep]).strip()
                    if not trial:
                        continue
                    if "def choose(" not in trial:
                        continue
                    try:
                        compile(trial, "<candidate-trim>", "exec")
                        return trial
                    except SyntaxError:
                        continue
                return None

            p = Path(program_path)
            raw = p.read_text(encoding="utf-8")
            candidate = _extract_python_candidate(raw)
            if not candidate:
                return program_path
            try:
                compile(candidate, str(p), "exec")
            except SyntaxError:
                # Final fallback: drop any standalone fence lines and retry.
                fallback = "\\n".join(
                    line for line in candidate.splitlines() if not line.strip().startswith("```")
                ).strip()
                if not fallback:
                    return program_path
                try:
                    compile(fallback, str(p), "exec")
                    candidate = fallback
                except SyntaxError:
                    salvaged = _salvage_by_trimming_tail(fallback)
                    if salvaged is None:
                        return program_path
                    candidate = salvaged
            if candidate != raw:
                p.write_text(candidate, encoding="utf-8")
            # Ensure choose() cannot silently fall through and return None.
            if not _choose_has_explicit_final_return(candidate):
                candidate2 = _append_choose_fallback_return(candidate)
                try:
                    compile(candidate2, str(p), "exec")
                    candidate = candidate2
                    p.write_text(candidate, encoding="utf-8")
                except SyntaxError:
                    # Keep original candidate if fallback insertion somehow breaks syntax.
                    pass
            return program_path


        def _load_program(program_path: str):
            program_path = _guard_candidate_file(program_path)
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
                if isinstance(p_raw, (bool, int)) and int(p_raw) in (0, 1):
                    p_raw = 1.0 if int(p_raw) == 1 else 0.0
                if not isinstance(p_raw, float):
                    raise TypeError(f"trial={{i}} expected float prob, got {{type(p_raw)}}")
                if not (0.0 <= p_raw <= 1.0):
                    raise ValueError(f"trial={{i}} probability out of [0,1]: {{p_raw}}")
                p = min(max(float(p_raw), 1e-9), 1.0 - 1e-9)
                ll_sum += y * math.log(p) + (1 - y) * math.log(1.0 - p)
                pred = 1 if p_raw >= 0.5 else 0
                correct += int(pred == y)
            return (ll_sum / total), (correct / total), total


        def _eval_action_trials(choose_fn, trials):
            total = len(trials)
            if total == 0:
                raise ValueError("No trials provided to evaluator.")
            correct = 0
            for t in trials:
                try:
                    pred = choose_fn(t["problem"], t["history"])
                except Exception:
                    pred = None
                if pred is not None and int(pred) == int(t["action"]):
                    correct += 1
            return (correct / total), total


        def _eval_cpc18_mse(choose_fn, trials, observed_blocks):
            problems = {{}}
            for tr in trials:
                pid = int(tr["problem_id"])
                problems.setdefault(pid, []).append(tr)

            all_mse = []
            for problem_id, problem_trials in problems.items():
                if str(problem_id) in observed_blocks:
                    obs_rates = observed_blocks[str(problem_id)]
                elif problem_id in observed_blocks:
                    obs_rates = observed_blocks[problem_id]
                else:
                    continue

                blocks = {{}}
                for tr in problem_trials:
                    bid = int(tr["block_id"])
                    blocks.setdefault(bid, []).append(tr)

                pred_rates = [0.0] * 5
                for block_id in range(1, 6):
                    block_trials = blocks.get(block_id, [])
                    if not block_trials:
                        continue
                    preds = []
                    for tr in block_trials:
                        try:
                            pred = choose_fn(tr["problem"], tr["history"])
                            preds.append(int(pred == 1))
                        except Exception:
                            continue
                    if not preds:
                        return float("inf"), False
                    pred_rates[block_id - 1] = sum(preds) / len(preds)
                mse = 100.0 * sum((float(pred_rates[i]) - float(obs_rates[i])) ** 2 for i in range(5)) / 5.0
                all_mse.append(float(mse))

            if not all_mse:
                return float("inf"), False
            return float(sum(all_mse) / len(all_mse)), True


        def _train_artifact():
            try:
                return TRAIN_PROMPT_PATH.read_text(encoding="utf-8")
            except OSError:
                return ""


        def evaluate(program_path: str):
            art = {{"choice13k_train_trials": _train_artifact()}}
            try:
                payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
                train_trials = payload["train_trials"]
                test_trials = payload["test_trials"]
                observed_blocks = payload.get("test_observed_blocks", {{}})
                cpc18_official = bool(payload.get("cpc18_official_mse", True))

                mod = _load_program(program_path)
                choose_fn = getattr(mod, "choose", None)
                if choose_fn is None:
                    raise AttributeError("Candidate program must define choose(problem, history).")

                if DATASET in {"choice13k", "mixed_gambles"}:
                    train_lls, train_accs = [], []
                    train_n = float(len(train_trials))
                    test_n = float(len(test_trials))
                    for seed in range(max(1, int(N_EVAL_SEEDS))):
                        train_ll, train_acc, _ = _eval_trials(choose_fn, train_trials)
                        train_lls.append(float(train_ll))
                        train_accs.append(float(train_acc))
                    train_ll = float(sum(train_lls) / len(train_lls))
                    train_acc = float(sum(train_accs) / len(train_accs))
                    combined = float(train_ll) if FITNESS_METRIC == "loglik" else float(train_acc)
                    if FITNESS_METRIC == "loglik":
                        test_ll = None
                        test_acc = None
                    else:
                        test_lls, test_accs = [], []
                        for seed in range(max(1, int(N_EVAL_SEEDS))):
                            test_ll, test_acc, _ = _eval_trials(choose_fn, test_trials)
                            test_lls.append(float(test_ll))
                            test_accs.append(float(test_acc))
                        test_ll = float(sum(test_lls) / len(test_lls))
                        test_acc = float(sum(test_accs) / len(test_accs))
                    metrics = {{
                        "combined_score": combined,
                        "train_loglik": float(train_ll),
                        "test_loglik": float(test_ll) if test_ll is not None else None,
                        "train_acc": float(train_acc),
                        "test_acc": float(test_acc) if test_acc is not None else None,
                        "train_n": float(train_n),
                        "test_n": float(test_n),
                        "fatal_failure": 0.0,
                    }}
                elif DATASET == "cpc18" and cpc18_official:
                    train_accs, test_accs = [], []
                    train_mses, test_mses = [], []
                    train_n = float(len(train_trials))
                    test_n = float(len(test_trials))
                    for seed in range(max(1, int(N_EVAL_SEEDS))):
                        train_acc, _ = _eval_action_trials(choose_fn, train_trials)
                        test_acc, _ = _eval_action_trials(choose_fn, test_trials)
                        train_mse, train_mse_valid = _eval_cpc18_mse(choose_fn, train_trials, observed_blocks)
                        test_mse, test_mse_valid = _eval_cpc18_mse(choose_fn, test_trials, observed_blocks)
                        if (not train_mse_valid) or (not test_mse_valid):
                            raise RuntimeError("Invalid CPC18 MSE evaluation (missing/invalid block predictions).")
                        train_accs.append(float(train_acc))
                        test_accs.append(float(test_acc))
                        train_mses.append(float(train_mse))
                        test_mses.append(float(test_mse))
                    train_acc = float(sum(train_accs) / len(train_accs))
                    test_acc = float(sum(test_accs) / len(test_accs))
                    train_mse = float(sum(train_mses) / len(train_mses))
                    test_mse = float(sum(test_mses) / len(test_mses))
                    combined = -float(train_mse)
                    metrics = {{
                        "combined_score": float(combined),
                        "train_loglik": None,
                        "test_loglik": None,
                        "train_acc": float(train_acc),
                        "test_acc": float(test_acc),
                        "train_mse": float(train_mse),
                        "test_mse": float(test_mse),
                        "train_n": float(train_n),
                        "test_n": float(test_n),
                        "fatal_failure": 0.0,
                    }}
                elif DATASET == "cpc18" and (not cpc18_official):
                    train_lls, train_accs = [], []
                    train_n = float(len(train_trials))
                    test_n = float(len(test_trials))
                    for seed in range(max(1, int(N_EVAL_SEEDS))):
                        train_ll, train_acc, _ = _eval_trials(choose_fn, train_trials)
                        train_lls.append(float(train_ll))
                        train_accs.append(float(train_acc))
                    train_ll = float(sum(train_lls) / len(train_lls))
                    train_acc = float(sum(train_accs) / len(train_accs))
                    if FITNESS_METRIC == "loglik":
                        test_ll = None
                        test_acc = None
                    else:
                        test_lls, test_accs = [], []
                        for seed in range(max(1, int(N_EVAL_SEEDS))):
                            test_ll, test_acc, _ = _eval_trials(choose_fn, test_trials)
                            test_lls.append(float(test_ll))
                            test_accs.append(float(test_acc))
                        test_ll = float(sum(test_lls) / len(test_lls))
                        test_acc = float(sum(test_accs) / len(test_accs))
                    combined = float(train_ll) if FITNESS_METRIC == "loglik" else float(train_acc)
                    metrics = {{
                        "combined_score": float(combined),
                        "train_loglik": float(train_ll),
                        "test_loglik": float(test_ll) if test_ll is not None else None,
                        "train_acc": float(train_acc),
                        "test_acc": float(test_acc) if test_acc is not None else None,
                        "train_mse": 0.0,
                        "test_mse": 0.0,
                        "train_n": float(train_n),
                        "test_n": float(test_n),
                        "fatal_failure": 0.0,
                    }}
                else:
                    train_accs, test_accs = [], []
                    train_n = float(len(train_trials))
                    test_n = float(len(test_trials))
                    for seed in range(max(1, int(N_EVAL_SEEDS))):
                        train_acc, _ = _eval_action_trials(choose_fn, train_trials)
                        test_acc, _ = _eval_action_trials(choose_fn, test_trials)
                        train_accs.append(float(train_acc))
                        test_accs.append(float(test_acc))
                    train_acc = float(sum(train_accs) / len(train_accs))
                    test_acc = float(sum(test_accs) / len(test_accs))
                    metrics = {{
                        "combined_score": float(train_acc),
                        "train_loglik": None,
                        "test_loglik": None,
                        "train_acc": float(train_acc),
                        "test_acc": float(test_acc),
                        "train_n": float(train_n),
                        "test_n": float(test_n),
                        "fatal_failure": 0.0,
                    }}
                return EvaluationResult(metrics=metrics, artifacts=art)
            except Exception as e:
                print("[FATAL] evaluator failure:", repr(e), flush=True)
                traceback.print_exc()
                _pl = None
                try:
                    _pl = json.loads(DATA_PATH.read_text(encoding="utf-8"))
                except OSError:
                    _pl = {{}}
                cpc18_off = bool(_pl.get("cpc18_official_mse", True))
                _tl = -1.0e9 if (DATASET in {"choice13k", "mixed_gambles"} or (DATASET == "cpc18" and (not cpc18_off))) else None
                metrics = {{
                    "combined_score": -1.0e9,
                    "train_loglik": _tl,
                    "test_loglik": _tl,
                    "train_acc": 0.0,
                    "test_acc": 0.0,
                    "train_mse": float("inf") if DATASET == "cpc18" and cpc18_off else (0.0 if DATASET == "cpc18" else None),
                    "test_mse": float("inf") if DATASET == "cpc18" and cpc18_off else (0.0 if DATASET == "cpc18" else None),
                    "fatal_failure": 1.0,
                }}
                return EvaluationResult(metrics=metrics, artifacts=art)
        """
    ).strip() + "\n"


def _build_openevolve_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode == "local":
        api_base = args.llm_server_url
        api_key = args.llm_api_key
    else:
        api_base = args.api_base if args.api_base else "https://api.openai.com/v1"
        api_key = args.api_key if args.api_key else "${OPENAI_API_KEY}"

    if args.dataset in {"choice13k", "mixed_gambles"}:
        if args.fitness_metric == "loglik":
            objective = "increase train fitness (combined_score / mean log-likelihood)."
        else:
            objective = "increase train accuracy fitness (combined_score)."
    elif args.dataset == "cpc18":
        if getattr(args, "cpc18_official_mse", False):
            objective = "increase train fitness where combined_score = -train_mse (lower MSE is better)."
        elif args.fitness_metric == "loglik":
            objective = "increase train fitness (combined_score = mean log-likelihood on train trials)."
        else:
            objective = "increase train accuracy fitness (combined_score)."
    else:
        objective = "increase train accuracy fitness (combined_score)."

    prompt_cfg: Dict[str, Any] = {
        "num_top_programs": int(args.prompt_num_top_programs),
        "num_diverse_programs": int(args.prompt_num_diverse_programs),
        "include_artifacts": False,
        "max_artifact_bytes": int(args.max_artifact_bytes),
    }

    system_message = (
        f"You are improving a {args.dataset} choose(problem, history) policy. "
        f"Use provided code context to {objective} "
        "Output ONLY runnable Python code (no explanations, no markdown fences, no preamble), "
        "preserving the choose(problem, history) signature."
    )
    if args.fitness_metric == "loglik":
        prompt_dataset = None
        if args.dataset == "choice13k":
            prompt_dataset = "choice13k"
        elif args.dataset == "mixed_gambles":
            prompt_dataset = "mixed_gambles"
        elif args.dataset == "cpc18" and not getattr(args, "cpc18_official_mse", False):
            prompt_dataset = "cpc18"
        if prompt_dataset is not None:
            prompt_root = REPO_ROOT / "prompts" / "Template_evo" / prompt_dataset / "non_strict" / "loglik"
            infer_path = prompt_root / "infer_single_choice.txt"
            template_path = prompt_root / "single_code_template.txt"
            try:
                infer_text = infer_path.read_text(encoding="utf-8").strip()
                template_text = template_path.read_text(encoding="utf-8").strip()
                system_message = (
                    f"{system_message}\n\n"
                    "# Dataset-specific loglik prompt guidance\n"
                    f"{infer_text}\n\n"
                    "# Dataset-specific loglik code template\n"
                    f"{template_text}"
                )
            except OSError as e:
                print(
                    f"[WARN] Could not load dataset-specific loglik prompts from {prompt_root}: {e}",
                    flush=True,
                )
    prompt_cfg["system_message"] = system_message
    prompt_cfg["evaluator_system_message"] = "You are a strict code evaluator."

    return {
        "max_iterations": int(args.n_iterations),
        # Keep per-iteration checkpoints so we can emit W&B time series points
        # for each participant without relying on OpenEvolve internal callbacks.
        "checkpoint_interval": 1,
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
            "max_tokens": int(args.llm_max_tokens),
            "timeout": int(args.llm_timeout_sec),
            "retries": 2,
            "retry_delay": 3,
        },
        "prompt": prompt_cfg,
        "database": {
            "in_memory": True,
            "log_prompts": True,
            "population_size": max(30, int(args.n_candidates) * 3),
            "archive_size": max(10, int(args.n_candidates)),
            "num_islands": 1,
            "migration_interval": 1000,
            "migration_rate": 0.1,
            "elite_selection_ratio": 0.5,
            "exploration_ratio": 0.05,
            "exploitation_ratio": 0.9,
            "feature_dimensions": ["complexity"],
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


def _load_trials_for_participant(
    *,
    args: argparse.Namespace,
    participant_id: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if args.dataset == "choice13k":
        experiments = get_choice13k_experiments(n_participants=participant_id + 1)
        exp = experiments[participant_id]
        train_trials, test_trials, _ = split_trials(
            exp, split_ratio=args.split_ratio, split_seed=args.split_seed
        )
        return train_trials, test_trials, None

    if args.dataset == "cpc18":
        cpc18_data_path = args.data_path if args.data_path != "data" else "datasets/cpc18"
        participant_data = load_cpc18_track2_data(
            data_path=cpc18_data_path, participant_id=participant_id
        )
        train_trials, test_trials, test_observed_blocks = split_cpc18_trials(
            participant_data,
            train_ratio=0.8,
            cpc18_official_mse=bool(getattr(args, "cpc18_official_mse", False)),
            split_ratio=float(getattr(args, "split_ratio", 0.9)),
            split_seed=int(getattr(args, "split_seed", 0)),
        )
        observed = {str(k): [float(x) for x in v] for k, v in (test_observed_blocks or {}).items()}
        return train_trials, test_trials, observed

    if args.dataset == "mixed_gambles":
        train_trials, test_trials, _ = load_mixed_gambles_trials(
            args.mixed_gambles_csv,
            participant_id,
            filter_gain_loss_only=bool(args.filter_mixed_gambles),
            split_ratio=float(args.split_ratio),
            split_seed=int(args.split_seed),
        )
        return train_trials, test_trials, None

    raise ValueError(f"Unsupported dataset: {args.dataset!r}")


def _run_one_openevolve(
    *,
    seed_code: str,
    train_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
    test_observed_blocks: Optional[Dict[str, Any]],
    participant_tag: str,
    run_root: Path,
    args: argparse.Namespace,
    wandb: Optional[Any] = None,
) -> Dict[str, Any]:
    _ensure_openevolve_importable()
    _patch_openevolve_for_gemma_system_role()
    _patch_openevolve_for_minimal_prompt()
    from openevolve import OpenEvolve
    from openevolve.config import load_config

    part_dir = run_root / f"participant_{participant_tag}"
    part_dir.mkdir(parents=True, exist_ok=True)

    initial_program_path = part_dir / "initial_program.py"
    evaluator_path = part_dir / "evaluator.py"
    data_path = part_dir / "dataset.json"
    train_trials_prompt_path = part_dir / "train_trials_prompt.txt"
    config_path = part_dir / "config.yaml"

    initial_program_path.write_text(seed_code, encoding="utf-8")
    data_payload = {
        "train_trials": _to_builtin(train_trials),
        "test_trials": _to_builtin(test_trials),
        "test_observed_blocks": _to_builtin(test_observed_blocks) if test_observed_blocks is not None else {},
        "cpc18_official_mse": bool(getattr(args, "cpc18_official_mse", False)),
    }
    data_path.write_text(json.dumps(data_payload), encoding="utf-8")
    _write_train_trials_prompt_file(
        dataset=args.dataset,
        fitness_metric=args.fitness_metric,
        path=train_trials_prompt_path,
        train_trials=train_trials,
        max_prompt_train_trials=int(args.max_prompt_train_trials),
        prompt_train_trials_seed=int(args.split_seed),
        max_history_items_per_trial=int(args.max_history_items_per_trial),
        max_prompt_trials_per_problem=int(args.max_prompt_trials_per_problem),
    )
    print(
        f"[INFO] participant {participant_tag}: wrote train trial prompt for OpenEvolve artifacts: "
        f"{train_trials_prompt_path} ({train_trials_prompt_path.stat().st_size} bytes)"
    )
    evaluator_path.write_text(
        _build_participant_evaluator_code(
            data_path,
            train_trials_prompt_path,
            dataset=args.dataset,
            fitness_metric=args.fitness_metric,
            n_eval_seeds=int(args.n_eval_seeds),
        ),
        encoding="utf-8",
    )
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

    # Gather checkpoint-level pool-best programs and metrics.
    ckpt_dir = output_dir / "checkpoints"
    checkpoint_rows: List[Dict[str, Any]] = []
    if ckpt_dir.exists():
        for cp in ckpt_dir.iterdir():
            if (not cp.is_dir()) or (not cp.name.startswith("checkpoint_")):
                continue
            info_path = cp / "best_program_info.json"
            if not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            iter_idx = info.get("current_iteration", None)
            if iter_idx is None:
                continue
            try:
                iter_idx_int = int(iter_idx)
            except (TypeError, ValueError):
                continue
            prog_id = info.get("id", None)
            prog_code = None
            if prog_id is not None:
                prog_json = cp / "programs" / f"{prog_id}.json"
                if prog_json.exists():
                    try:
                        prog_payload = json.loads(prog_json.read_text(encoding="utf-8"))
                        prog_code = prog_payload.get("code", None)
                    except Exception:
                        prog_code = None
            checkpoint_rows.append(
                {
                    "iter_idx": iter_idx_int,
                    "program_id": prog_id,
                    "metrics": info.get("metrics", {}) or {},
                    "program_code": prog_code,
                }
            )
    checkpoint_rows.sort(key=lambda x: int(x["iter_idx"]))

    # Evaluate only pool-best programs per iteration in loglik mode.
    is_loglik_mode = (
        args.fitness_metric == "loglik"
        and (args.dataset in {"choice13k", "mixed_gambles"} or (args.dataset == "cpc18" and not args.cpc18_official_mse))
    )
    eval_cache: Dict[str, Dict[str, Any]] = {}
    for row in checkpoint_rows:
        row["paired_metrics"] = None
        if (not is_loglik_mode) or (not row.get("program_code")):
            continue
        pid = str(row.get("program_id", ""))
        if pid in eval_cache:
            row["paired_metrics"] = eval_cache[pid]
            continue
        try:
            pm = _evaluate_program_code_metrics(
                code=str(row["program_code"]),
                dataset=args.dataset,
                fitness_metric=args.fitness_metric,
                cpc18_official_mse=bool(args.cpc18_official_mse),
                train_trials=train_trials,
                test_trials=test_trials,
                observed_blocks=test_observed_blocks,
                n_eval_seeds=int(args.n_eval_seeds),
            )
        except Exception:
            pm = None
        if pm is not None:
            eval_cache[pid] = pm
            row["paired_metrics"] = pm

    # Final best must be from the last updated pool (last checkpoint), then evaluated as a pair.
    final_row = checkpoint_rows[-1] if checkpoint_rows else None
    if final_row is not None and final_row.get("program_code"):
        final_best_code = str(final_row["program_code"])
        final_prog_id = str(final_row.get("program_id", "unknown"))
        final_iter = int(final_row.get("iter_idx", -1))
        final_metrics = dict(final_row.get("paired_metrics") or {})
        if not final_metrics:
            final_metrics = _evaluate_program_code_metrics(
                code=final_best_code,
                dataset=args.dataset,
                fitness_metric=args.fitness_metric,
                cpc18_official_mse=bool(args.cpc18_official_mse),
                train_trials=train_trials,
                test_trials=test_trials,
                observed_blocks=test_observed_blocks,
                n_eval_seeds=int(args.n_eval_seeds),
            )
    else:
        final_best_code = str(best_program.code)
        final_prog_id = str(getattr(best_program, "id", "unknown"))
        final_iter = int(args.n_iterations)
        final_metrics = dict(best_program.metrics or {})
        if is_loglik_mode:
            final_metrics = _evaluate_program_code_metrics(
                code=final_best_code,
                dataset=args.dataset,
                fitness_metric=args.fitness_metric,
                cpc18_official_mse=bool(args.cpc18_official_mse),
                train_trials=train_trials,
                test_trials=test_trials,
                observed_blocks=test_observed_blocks,
                n_eval_seeds=int(args.n_eval_seeds),
            )

    best_code_path = part_dir / "best_program.py"
    best_code_path.write_text(final_best_code, encoding="utf-8")
    best_from_iter_path = part_dir / f"best_program_fr_iter{final_iter}_cand{final_prog_id[:8]}.py"
    best_from_iter_path.write_text(final_best_code, encoding="utf-8")

    if wandb is not None:
        pid_key = f"p{participant_tag}"
        iter_key = f"{pid_key}_iter"
        wandb.define_metric(iter_key)
        wandb.define_metric(f"{pid_key}_*", step_metric=iter_key)
        for row in checkpoint_rows:
            iter_idx = int(row["iter_idx"])
            m = dict(row.get("metrics", {}) or {})
            paired = row.get("paired_metrics") if is_loglik_mode else None
            src = dict(paired) if paired else m
            train_fitness_i = src.get("combined_score", None)
            test_fitness_i = src.get("test_loglik", None)
            if args.dataset in {"choice13k", "mixed_gambles"}:
                if args.fitness_metric == "loglik":
                    train_fitness_i = src.get("train_loglik", train_fitness_i)
                    test_fitness_i = src.get("test_loglik", test_fitness_i)
                else:
                    train_fitness_i = src.get("train_acc", train_fitness_i)
                    test_fitness_i = src.get("test_acc", test_fitness_i)
            elif args.dataset == "cpc18":
                if args.cpc18_official_mse:
                    tm = src.get("test_mse", None)
                    test_fitness_i = (-float(tm)) if tm is not None else None
                    train_fitness_i = src.get("combined_score", train_fitness_i)
                elif args.fitness_metric == "loglik":
                    train_fitness_i = src.get("train_loglik", train_fitness_i)
                    test_fitness_i = src.get("test_loglik", test_fitness_i)
                else:
                    train_fitness_i = src.get("combined_score", train_fitness_i)
                    test_fitness_i = src.get("test_acc", test_fitness_i)
            payload = {
                iter_key: iter_idx,
                f"{pid_key}_train_fitness": train_fitness_i,
                f"{pid_key}_test_fitness": test_fitness_i,
                f"{pid_key}_train_loglik": src.get("train_loglik", None),
                f"{pid_key}_test_loglik": src.get("test_loglik", None),
                f"{pid_key}_train_accuracy": src.get("train_acc", None),
                f"{pid_key}_test_accuracy": src.get("test_acc", None),
                f"{pid_key}_n_valid": src.get("train_n", None),
            }
            wandb.log(payload)

    metrics = dict(final_metrics or {})
    raw_train_ll = metrics.get("train_loglik", None)
    raw_test_ll = metrics.get("test_loglik", None)
    train_ll = float(raw_train_ll) if raw_train_ll is not None else None
    test_ll = float(raw_test_ll) if raw_test_ll is not None else None
    train_acc = float(metrics.get("train_acc", 0.0))
    test_acc = float(metrics.get("test_acc", 0.0))
    train_mse = float(metrics.get("train_mse", float("inf"))) if metrics.get("train_mse", None) is not None else None
    test_mse = float(metrics.get("test_mse", float("inf"))) if metrics.get("test_mse", None) is not None else None
    combined_score = float(metrics.get("combined_score", -1e9))
    fatal_flag = float(metrics.get("fatal_failure", 0.0))

    if args.dataset in {"choice13k", "mixed_gambles"} or (args.dataset == "cpc18" and not args.cpc18_official_mse):
        hard_fail = (
            fatal_flag >= 0.5
            or train_ll is None
            or test_ll is None
            or train_ll <= -1e8
            or test_ll <= -1e8
        )
    else:
        hard_fail = fatal_flag >= 0.5 or combined_score <= -1e8
    if hard_fail:
        msg = (
            f"[FATAL] participant {participant_tag} evolution failed. "
            f"metrics={metrics} output_dir={output_dir}"
        )
        print(msg, flush=True)
        if not args.allow_failure:
            raise RuntimeError(msg)

    if args.dataset in {"choice13k", "mixed_gambles"}:
        print(
            f"[INFO] participant {participant_tag}: train_loglik={float(train_ll):.6f}, "
            f"test_loglik={float(test_ll):.6f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
            f"fatal_failure={fatal_flag:.1f}"
        )
        if args.fitness_metric == "loglik":
            train_fitness = float(train_ll)
            test_fitness = float(test_ll)
        else:
            train_fitness = float(train_acc)
            test_fitness = float(test_acc)
    elif args.dataset == "cpc18":
        if args.cpc18_official_mse:
            print(
                f"[INFO] participant {participant_tag}: train_mse={train_mse}, "
                f"test_mse={test_mse}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
                f"combined_score={combined_score:.6f}, fatal_failure={fatal_flag:.1f}"
            )
            train_fitness = combined_score
            test_fitness = -float(test_mse) if test_mse is not None and math.isfinite(test_mse) else -1e9
        else:
            print(
                f"[INFO] participant {participant_tag}: train_loglik={float(train_ll):.6f}, "
                f"test_loglik={float(test_ll):.6f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
                f"combined_score={combined_score:.6f}, fatal_failure={fatal_flag:.1f}"
            )
            if args.fitness_metric == "loglik":
                train_fitness = float(train_ll) if train_ll is not None else -1e9
                test_fitness = float(test_ll) if test_ll is not None else -1e9
            else:
                train_fitness = combined_score
                test_fitness = float(test_acc) if test_acc is not None else 0.0
    else:
        print(
            f"[INFO] participant {participant_tag}: train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}, combined_score={combined_score:.6f}, fatal_failure={fatal_flag:.1f}"
        )
        train_fitness = combined_score
        test_fitness = float(test_acc)

    results_payload = {
        "overall_best_train": {
            "program_id": final_prog_id,
            "origin_iteration": final_iter,
            "program_file": best_from_iter_path.name,
            "train_loglik": train_ll,
            "test_loglik": test_ll,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_fitness": train_fitness,
            "test_fitness": test_fitness,
        },
        # Keep paired values from the same best-train program.
        "overall_best_test": {
            "program_id": final_prog_id,
            "origin_iteration": final_iter,
            "program_file": best_from_iter_path.name,
            "train_loglik": train_ll,
            "test_loglik": test_ll,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_fitness": train_fitness,
            "test_fitness": test_fitness,
        },
    }
    (part_dir / "results.json").write_text(
        json.dumps(_to_builtin(results_payload), indent=2),
        encoding="utf-8",
    )

    return {
        "participant_id": participant_tag,
        "train_loglik": train_ll,
        "test_loglik": test_ll,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "combined_score": combined_score,
        "train_fitness": train_fitness,
        "test_fitness": test_fitness,
        "seed_program_train_fitness": train_fitness,
        "seed_program_test_fitness": test_fitness,
        "total_runtime": runtime,
        "fatal_failure": fatal_flag,
        "metrics_raw": metrics,
        "overall_best_train": results_payload["overall_best_train"],
        "overall_best_test": results_payload["overall_best_test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEvolve baseline (TE-compatible participant scope).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="choice13k",
        choices=["choice13k", "cpc18", "mixed_gambles"],
    )
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
    parser.add_argument("--fitness_metric", type=str, default="accuracy", choices=["loglik", "accuracy"])
    parser.add_argument("--split_mode", type=str, default="within_participant", choices=["within_participant", "across_participants"])
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="For cpc18: directory containing Track II files (default datasets/cpc18 when value is 'data').",
    )
    parser.add_argument(
        "--cpc18_official_mse",
        action="store_true",
        help="CPC18: official all-trials block MSE protocol. Default: held-out train/test split (set --fitness_metric loglik for log-likelihood).",
    )
    parser.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default="datasets/mixed_gambles/data_all_2021-01-08.csv",
        help="CSV path for mixed_gambles dataset.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--allow_failure", action="store_true", help="Continue run after a participant-level fatal failure.")
    parser.add_argument("--eval_timeout_sec", type=int, default=300)
    parser.add_argument("--llm_timeout_sec", type=int, default=120)
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=1024,
        help="Max output tokens per mutation request (smaller value reduces context-overflow failures).",
    )
    parser.add_argument(
        "--n_eval_seeds",
        type=int,
        default=3,
        help="Number of repeated evaluation passes to average for each candidate.",
    )
    parser.add_argument(
        "--prompt_num_top_programs",
        type=int,
        default=3,
        help="Top-performing programs included in OpenEvolve prompt context.",
    )
    parser.add_argument(
        "--prompt_num_diverse_programs",
        type=int,
        default=0,
        help="Diverse programs included in OpenEvolve prompt context.",
    )
    parser.add_argument(
        "--max_prompt_train_trials",
        type=int,
        default=120,
        help="Max train trials to include in the mutation-prompt artifact (0 = all). Subsample is random with --split_seed.",
    )
    parser.add_argument(
        "--max_history_items_per_trial",
        type=int,
        default=12,
        help="For each serialized train trial, include at most this many most-recent history items (0 = all).",
    )
    parser.add_argument(
        "--max_prompt_trials_per_problem",
        type=int,
        default=0,
        help="Cap serialized train trials per problem in prompt artifacts (0 = no per-problem cap).",
    )
    parser.add_argument(
        "--max_artifact_bytes",
        type=int,
        default=131072,
        help="OpenEvolve truncates each prompt artifact to this many bytes (default 128 KiB).",
    )
    args = parser.parse_args()

    if args.max_prompt_train_trials < 0:
        print("Error: --max_prompt_train_trials must be >= 0.")
        sys.exit(1)
    if args.max_history_items_per_trial < 0:
        print("Error: --max_history_items_per_trial must be >= 0.")
        sys.exit(1)
    if args.max_prompt_trials_per_problem < 0:
        print("Error: --max_prompt_trials_per_problem must be >= 0.")
        sys.exit(1)
    if args.max_artifact_bytes < 1024:
        print("Error: --max_artifact_bytes must be at least 1024.")
        sys.exit(1)
    if args.llm_max_tokens < 64:
        print("Error: --llm_max_tokens must be >= 64.")
        sys.exit(1)
    if args.n_eval_seeds < 1:
        print("Error: --n_eval_seeds must be >= 1.")
        sys.exit(1)
    if args.prompt_num_top_programs < 1:
        print("Error: --prompt_num_top_programs must be >= 1.")
        sys.exit(1)
    if args.prompt_num_diverse_programs < 0:
        print("Error: --prompt_num_diverse_programs must be >= 0.")
        sys.exit(1)

    if not (0.0 < args.split_ratio < 1.0):
        print("Error: --split_ratio must be in (0,1).")
        sys.exit(1)
    if args.fitness_metric == "loglik" and args.dataset not in {"choice13k", "mixed_gambles"} and not (
        args.dataset == "cpc18" and not args.cpc18_official_mse
    ):
        print(
            "Error: --fitness_metric loglik is only supported for --dataset choice13k/mixed_gambles, "
            "or cpc18 without --cpc18_official_mse."
        )
        sys.exit(1)
    if args.split_mode == "across_participants" and args.dataset != "choice13k":
        print("Error: --split_mode across_participants is only supported with --dataset choice13k.")
        sys.exit(1)
    if args.split_mode == "across_participants" and args.participant_scope == "single":
        print("Error: across_participants needs at least two participants; use range or all.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_run_dir = Path(
        args.output_dir
        if args.output_dir
        else str(REPO_ROOT / "generated_outputs" / args.dataset / "openevolve" / f"run_{timestamp}")
    )
    base_run_dir.mkdir(parents=True, exist_ok=True)
    cmd_log = _write_command_line_log(base_run_dir)
    hyp_log = _write_hyperparameters_log(base_run_dir, args)
    print(f"Wrote full command line to {cmd_log}")
    print(f"Wrote hyperparameters to {hyp_log}")

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
            run_name = f"{args.dataset}_openevolve_{timestamp}_{args.participant_scope}"
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
                test_observed_blocks=None,
                participant_tag="0",
                run_root=base_run_dir,
                args=args,
                wandb=wandb,
            )
            row = {
                "participant_id": 0,
                "train_fitness": result["train_fitness"],
                "test_fitness": result["test_fitness"],
                "total_runtime": result["total_runtime"],
                "seed_program_train_fitness": result["seed_program_train_fitness"],
                "seed_program_test_fitness": result["seed_program_test_fitness"],
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
                train_trials, test_trials, test_observed_blocks = _load_trials_for_participant(
                    args=args, participant_id=pid
                )
                result = _run_one_openevolve(
                    seed_code=seed_code,
                    train_trials=train_trials,
                    test_trials=test_trials,
                    test_observed_blocks=test_observed_blocks,
                    participant_tag=str(pid),
                    run_root=base_run_dir,
                    args=args,
                    wandb=wandb,
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
                            "final/participant_id": pid,
                            "final/train_fitness": result["train_fitness"],
                            "final/test_fitness": result["test_fitness"],
                            "final/train_loglik": result["train_loglik"],
                            "final/test_loglik": result["test_loglik"],
                            "final/train_acc": result["train_acc"],
                            "final/test_acc": result["test_acc"],
                            "final/train_mse": result["train_mse"],
                            "final/test_mse": result["test_mse"],
                            "final/fatal_failure": result["fatal_failure"],
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
            train_trials, test_trials, test_observed_blocks = _load_trials_for_participant(
                args=args, participant_id=pid
            )
            result = _run_one_openevolve(
                seed_code=seed_code,
                train_trials=train_trials,
                test_trials=test_trials,
                test_observed_blocks=test_observed_blocks,
                participant_tag=str(pid),
                run_root=base_run_dir,
                args=args,
                wandb=wandb,
            )
            summ = {
                "participant_id": pid,
                "train_acc": result["train_acc"],
                "test_acc": result["test_acc"],
                "train_loglik": result["train_loglik"],
                "test_loglik": result["test_loglik"],
                "train_mse": result["train_mse"],
                "test_mse": result["test_mse"],
                "combined_score": result["combined_score"],
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
                        "final/participant_id": pid,
                        "final/train_fitness": result["train_fitness"],
                        "final/test_fitness": result["test_fitness"],
                        "final/train_loglik": result["train_loglik"],
                        "final/test_loglik": result["test_loglik"],
                        "final/train_acc": result["train_acc"],
                        "final/test_acc": result["test_acc"],
                        "final/train_mse": result["train_mse"],
                        "final/test_mse": result["test_mse"],
                        "final/fatal_failure": result["fatal_failure"],
                    }
                )

        print(f"Wrote {summary_file} and loglik summaries under {base_run_dir}")
    finally:
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
