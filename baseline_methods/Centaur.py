"""
Centaur (Llama 3.1 + Psych-101-style prompts) baseline for Choice13k/CPC18/Mixed Gambles.

Run from repository root (same as Template_evo_non_strict.py):
  python baseline_methods/Centaur.py --dataset choice13k --fitness_metric loglik \\
    --participant_scope range --range_start_ordinal 0 --range_end_ordinal 2

Requires: conda env `centaur` (or torch + unsloth + transformers on a GPU node).

If Psych-101-test is gated for your account, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
in the job environment (see `data_modules/choice13k.py`). Model download uses the same
vars via Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Repo root on sys.path (for `data_modules` imports when executing this file directly)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_modules.choice13k import Experiment, get_choice13k_experiments  # noqa: E402
from data_modules.cpc18 import load_cpc18_track2_data, split_cpc18_trials  # noqa: E402


def experiment_to_trials(exp: Experiment) -> Tuple[List[Dict[str, Any]], list]:
    """Same as Template_evo_non_strict.experiment_to_trials (avoid importing TE / JAX)."""
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


def trials_from_blocks_chronological(
    exp: Experiment, block_indices: set
) -> List[Dict[str, Any]]:
    """Same as Template_evo_non_strict.trials_from_blocks_chronological."""
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
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """Same as Template_evo_non_strict.split_trials (block-level train/test)."""
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
    option_keys = [0, 1]  # 0 = gamble option, 1 = certain option
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
    """Same as Template_evo_non_strict.load_valid_participant_ids_from_json (no JAX import)."""
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
            f"Missing valid participant list: {path}. "
            f"Generate with: python utils/tools/collect_participant_ids.py --dataset {dataset}"
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
    """Same semantics as Template_evo_non_strict.resolve_participants_for_scope."""
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


def evaluate_centaur_on_trials(
    chooser: "CentaurChooser",
    trials: List[Dict[str, Any]],
    *,
    verbose: bool = False,
    n_seeds: int = 1,
    debug_prob: bool = False,
    debug_limit: int = 5,
) -> Dict[str, Any]:
    """
    Same aggregate metrics as Template_evo_non_strict.evaluate_choice13k_program, but uses
    indexed prompts so past-trial key labels match each block's option_keys.
    """
    total = len(trials)
    seed_avg_accs: List[float] = []
    seed_avg_logliks: List[float] = []
    first_seed_errors = 0
    first_seed_neutral = 0
    first_seed_neutral_reasons: Dict[str, int] = {}

    def _one_pass(seed_idx: int) -> Tuple[float, float, int, int, Dict[str, int]]:
        loglik_acc = 0.0
        correct = 0
        errors = 0
        neutral = 0
        neutral_reasons: Dict[str, int] = {}
        for i in range(total):
            y = int(trials[i]["action"])
            try:
                p_raw = chooser.prob_choose_second_option(trials, i)
            except Exception as e:
                errors += 1
                if verbose and errors <= 3 and seed_idx == 0:
                    print(f"  Evaluation error: {e}")
                p = 0.5
                p_clamped = min(max(p, 1e-9), 1.0 - 1e-9)
                loglik_acc += y * np.log(p_clamped) + (1 - y) * np.log(1.0 - p_clamped)
                pred = 1 if p >= 0.5 else 0
                correct += int(pred == y)
                if debug_prob and seed_idx == 0 and errors <= debug_limit:
                    print(f"[DEBUG] trial={i} exception fallback to p=0.5 -> {e!r}")
                continue

            if not isinstance(p_raw, float):
                raise TypeError(f"expected float, got {type(p_raw)}")
            if not (0.0 <= p_raw <= 1.0):
                raise ValueError(f"expected p in [0,1], got {p_raw}")

            dbg = getattr(chooser, "last_prob_debug", {})
            if dbg.get("fallback_source") == "invalid_denom":
                neutral += 1
                reason = str(dbg.get("fallback_reason", "invalid_denom"))
                neutral_reasons[reason] = neutral_reasons.get(reason, 0) + 1
                if debug_prob and seed_idx == 0 and neutral <= debug_limit:
                    print(
                        f"[DEBUG] trial={i} invalid-denom fallback p=0.5 "
                        f"(lp0={dbg.get('lp0')}, lp1={dbg.get('lp1')}, "
                        f"s0_reason={dbg.get('suffix0_reason')}, s1_reason={dbg.get('suffix1_reason')})"
                    )

            p = min(max(p_raw, 1e-9), 1.0 - 1e-9)
            loglik_acc += y * np.log(p) + (1 - y) * np.log(1.0 - p)
            pred = 1 if p_raw >= 0.5 else 0
            correct += int(pred == y)

        avg_ll = loglik_acc / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_ll, acc, errors, neutral, neutral_reasons

    for seed in range(n_seeds):
        avg_ll, acc, errs, neutral, neutral_reasons = _one_pass(seed)
        seed_avg_logliks.append(avg_ll)
        seed_avg_accs.append(acc)
        if seed == 0:
            first_seed_errors = errs
            first_seed_neutral = neutral
            first_seed_neutral_reasons = neutral_reasons

    if debug_prob and n_seeds == 1:
        print(
            f"[DEBUG] eval summary: total={total}, exception_fallbacks={first_seed_errors}, "
            f"invalid_denom_fallbacks={first_seed_neutral}, reasons={first_seed_neutral_reasons}"
        )

    avg_acc = float(np.mean(seed_avg_accs)) if seed_avg_accs else 0.0
    avg_loglik = float(np.mean(seed_avg_logliks)) if seed_avg_logliks else float("-inf")
    correct = int(round(avg_acc * total))
    return {
        "accuracy": avg_acc,
        "avg_loglik": avg_loglik,
        "total": total,
        "correct": correct,
        "errors": first_seed_errors if n_seeds == 1 else 0,
        "neutral_fallbacks": first_seed_neutral if n_seeds == 1 else 0,
        "neutral_reasons": first_seed_neutral_reasons if n_seeds == 1 else {},
    }


PETERSON_INTRO = (
    "You will encounter a series of gambling problems where you have to select between two options.\n"
    "You can select an option by pressing the corresponding key.\n"
    "For some problems, you are told the points you received and missed out on after each selection, "
    "while for others this information is suppressed.\n"
    "In cases where the probabilities are unknown, they sum up to one and remain constant within a problem.\n"
)


def _format_option_line(letter: str, rewards: List[float], probs: Optional[List[float]]) -> str:
    if probs is None:
        parts = [f"either {float(r)} points with unknown chance" for r in rewards]
        if len(parts) == 1:
            body = parts[0]
        else:
            body = ", ".join(parts[:-1]) + ", or " + parts[-1]
        return f"Option {letter} delivers {body}."
    chunks = [
        f"{float(rewards[i])} points with {float(probs[i]) * 100:.1f}% chance"
        for i in range(len(rewards))
    ]
    if len(chunks) == 1:
        inner = chunks[0]
    else:
        inner = ", ".join(chunks[:-1]) + ", or " + chunks[-1]
    return f"Option {letter} delivers {inner}."


def _format_current_problem(problem: Dict[str, Any]) -> str:
    keys = problem["option_keys"]
    ga = problem["gamble_A"]
    gb = problem["gamble_B"]
    line_a = _format_option_line(keys[0], list(ga["rewards"]), ga.get("probs"))
    line_b = _format_option_line(keys[1], list(gb["rewards"]), gb.get("probs"))
    return f"{line_a}\n{line_b}"


def _one_history_line(past_problem: Dict[str, Any], h: Dict[str, Any]) -> str:
    keys = past_problem["option_keys"]
    act = int(h["action"])
    letter = keys[act]
    line = f"You press <<{letter}>>."
    if past_problem.get("has_feedback") and h.get("feedback") is not None:
        line += f" You receive {float(h['feedback'])} points by selecting this option."
    return line


def build_centaur_prompt_prefix_indexed(trials: List[Dict[str, Any]], trial_index: int) -> str:
    """
    Build prefix for trials[trial_index] using TE history convention.
    Past key labels come from each prior trial's problem (blocks may use different letters).

    History entries always refer to the len(history) trials immediately before this one in
    ``trials`` (works for split_trials lists and for across-participant concatenation).
    """
    cur = trials[trial_index]["problem"]
    hist = trials[trial_index]["history"]
    L = len(hist)
    start = trial_index - L
    if start < 0:
        raise ValueError(
            f"trial_index={trial_index} has len(history)={L} but not enough prior trials in list"
        )
    lines_hist: List[str] = []
    for j, h in enumerate(hist):
        lines_hist.append(_one_history_line(trials[start + j]["problem"], h))
    hist_txt = "\n".join(lines_hist)
    opt_txt = _format_current_problem(cur)
    parts = [PETERSON_INTRO.rstrip()]
    if hist_txt:
        parts.append(hist_txt)
    parts.append(opt_txt)
    parts.append("You press ")
    return "\n\n".join(parts)


class CentaurChooser:
    """Loads Centaur once; choose(problem, history) -> P(second option key), i.e. P(action==1)."""

    def __init__(
        self,
        model_name: str,
        *,
        max_seq_length: int = 32768,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self.last_prob_debug: Dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from unsloth import FastLanguageModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load Centaur (Unsloth probes GPU at import/load).")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer

    def _suffix_logprob(self, prefix: str, suffix: str) -> float:
        score, _ = self._suffix_logprob_detailed(prefix, suffix)
        return score

    def _suffix_logprob_detailed(self, prefix: str, suffix: str) -> Tuple[float, Optional[str]]:
        import torch

        self._ensure_loaded()
        model = self._model
        tokenizer = self._tokenizer
        device = next(model.parameters()).device

        text = prefix + suffix
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        pref = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        plen = int(pref["input_ids"].shape[1])
        if plen >= input_ids.shape[1]:
            return float("-inf"), "empty_suffix_after_tokenization"

        labels = input_ids.clone()
        labels[:, :plen] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if loss is None:
                return float("-inf"), "loss_none"
            if not torch.isfinite(loss):
                return float("nan"), f"loss_non_finite:{float(loss.item())}"
            n = int((labels != -100).sum().item())
            if n <= 0:
                return float("-inf"), "no_suffix_tokens_after_masking"
            score = float(-loss.item() * n)
            if not math.isfinite(score):
                return score, f"logprob_non_finite:{score}"
            return score, None

    def prob_choose_second_option(self, trials: List[Dict[str, Any]], trial_index: int) -> float:
        """P(action==1) for trials[trial_index] with correct cross-block key labels in history."""
        problem = trials[trial_index]["problem"]
        keys = problem["option_keys"]
        if len(keys) != 2:
            raise ValueError(f"Expected two option keys, got {keys!r}")

        prefix = build_centaur_prompt_prefix_indexed(trials, trial_index)
        s0 = f"<<{keys[0]}>>."
        s1 = f"<<{keys[1]}>>."
        lp0, r0 = self._suffix_logprob_detailed(prefix, s0)
        lp1, r1 = self._suffix_logprob_detailed(prefix, s1)
        m = max(lp0, lp1)
        t0 = math.exp(lp0 - m)
        t1 = math.exp(lp1 - m)
        denom = t0 + t1
        if denom <= 0 or not math.isfinite(denom):
            self.last_prob_debug = {
                "fallback_source": "invalid_denom",
                "fallback_reason": "denom_non_finite_or_non_positive",
                "lp0": lp0,
                "lp1": lp1,
                "suffix0_reason": r0,
                "suffix1_reason": r1,
            }
            return 0.5
        p1 = t1 / denom
        self.last_prob_debug = {
            "fallback_source": None,
            "lp0": lp0,
            "lp1": lp1,
            "suffix0_reason": r0,
            "suffix1_reason": r1,
        }
        return float(min(max(p1, 1e-9), 1.0 - 1e-9))


def _convert_cpc18_problem_to_choice_style(problem: Dict[str, Any]) -> Dict[str, Any]:
    p_ha = float(problem["pHa"])
    p_hb = float(problem["pHb"])
    return {
        "gamble_A": {
            "probs": [p_ha, max(0.0, 1.0 - p_ha)],
            "rewards": [float(problem["Ha"]), float(problem["La"])],
        },
        "gamble_B": {
            "probs": [p_hb, max(0.0, 1.0 - p_hb)],
            "rewards": [float(problem["Hb"]), float(problem["Lb"])],
        },
        "option_keys": ["A", "B"],
        "has_feedback": True,
    }


def _prepare_trials_for_centaur(dataset: str, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if dataset == "choice13k":
        return trials

    out: List[Dict[str, Any]] = []
    for t in trials:
        nt = dict(t)
        if dataset == "cpc18":
            nt["problem"] = _convert_cpc18_problem_to_choice_style(t["problem"])
        else:
            p = dict(t["problem"])
            p["option_keys"] = ["A", "B"]
            nt["problem"] = p
        out.append(nt)
    return out


def _eval_accuracy_from_probs(chooser: "CentaurChooser", trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trials)
    correct = 0
    errors = 0
    for i, t in enumerate(trials):
        y = int(t["action"])
        try:
            p_raw = chooser.prob_choose_second_option(trials, i)
            pred = 1 if p_raw >= 0.5 else 0
        except Exception:
            errors += 1
            pred = 0
        correct += int(pred == y)
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": float(acc), "total": int(total), "correct": int(correct), "errors": int(errors)}


def _eval_cpc18_mse_from_probs(
    chooser: "CentaurChooser",
    trials: List[Dict[str, Any]],
    observed_blocks: Dict[int, Any],
) -> Dict[str, Any]:
    pred_by_index: Dict[int, int] = {}
    for i in range(len(trials)):
        try:
            p_raw = chooser.prob_choose_second_option(trials, i)
            pred_by_index[i] = int(p_raw >= 0.5)
        except Exception:
            continue

    problems_dict: Dict[int, List[int]] = {}
    for i, tr in enumerate(trials):
        pid = int(tr["problem_id"])
        problems_dict.setdefault(pid, []).append(i)

    all_mse: List[float] = []
    valid = True
    for problem_id, problem_trial_indices in problems_dict.items():
        obs_rates = observed_blocks.get(problem_id)
        if obs_rates is None:
            obs_rates = observed_blocks.get(str(problem_id))
        if obs_rates is None:
            continue

        blocks: Dict[int, List[int]] = {}
        for i in problem_trial_indices:
            tr = trials[i]
            bid = int(tr["block_id"])
            blocks.setdefault(bid, []).append(i)

        pred_rates = np.zeros(5, dtype=np.float64)
        for block_id in range(1, 6):
            block_indices = blocks.get(block_id, [])
            if not block_indices:
                continue
            preds: List[int] = []
            for i in block_indices:
                if i not in pred_by_index:
                    continue
                preds.append(pred_by_index[i])
            if not preds:
                valid = False
                break
            pred_rates[block_id - 1] = float(np.mean(preds))
        if not valid:
            break

        obs_arr = np.asarray(obs_rates, dtype=np.float64)
        mse = 100.0 * float(np.mean((pred_rates - obs_arr) ** 2))
        all_mse.append(mse)

    if not valid or not all_mse:
        return {"mse": float("inf"), "valid": False, "n_problems": 0}
    return {"mse": float(np.mean(all_mse)), "valid": True, "n_problems": len(all_mse)}

def _participant_summary_row(
    participant_id: int,
    dataset: str,
    fitness_metric: str,
    train_eval: Dict[str, Any],
    test_eval: Dict[str, Any],
    train_mse_eval: Optional[Dict[str, Any]] = None,
    test_mse_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if dataset in {"choice13k", "mixed_gambles"}:
        tr_ll = train_eval["avg_loglik"]
        te_ll = test_eval["avg_loglik"]
        train_fit = tr_ll if fitness_metric == "loglik" else train_eval["accuracy"]
        test_fit = te_ll if fitness_metric == "loglik" else test_eval["accuracy"]
        return {
            "participant_id": participant_id,
            "train_acc": train_eval["accuracy"],
            "test_acc": test_eval["accuracy"],
            "train_loglik": tr_ll,
            "test_loglik": te_ll,
            "train_mse": None,
            "test_mse": None,
            "train_fitness": train_fit,
            "test_fitness": test_fit,
            "seed_program_train_fitness": train_fit,
            "seed_program_test_fitness": test_fit,
        }

    if dataset == "cpc18" and train_mse_eval is not None and test_mse_eval is not None:
        tr_mse = float(train_mse_eval["mse"])
        te_mse = float(test_mse_eval["mse"])
        tr_fit = -tr_mse if math.isfinite(tr_mse) else float("-inf")
        te_fit = -te_mse if math.isfinite(te_mse) else float("-inf")
        return {
            "participant_id": participant_id,
            "train_acc": train_eval["accuracy"],
            "test_acc": test_eval["accuracy"],
            "train_loglik": None,
            "test_loglik": None,
            "train_mse": tr_mse,
            "test_mse": te_mse,
            "train_fitness": tr_fit,
            "test_fitness": te_fit,
            "seed_program_train_fitness": tr_fit,
            "seed_program_test_fitness": te_fit,
        }
    if dataset == "cpc18":
        tr_ll = train_eval["avg_loglik"]
        te_ll = test_eval["avg_loglik"]
        train_fit = tr_ll if fitness_metric == "loglik" else train_eval["accuracy"]
        test_fit = te_ll if fitness_metric == "loglik" else test_eval["accuracy"]
        return {
            "participant_id": participant_id,
            "train_acc": train_eval["accuracy"],
            "test_acc": test_eval["accuracy"],
            "train_loglik": tr_ll,
            "test_loglik": te_ll,
            "train_mse": None,
            "test_mse": None,
            "train_fitness": train_fit,
            "test_fitness": test_fit,
            "seed_program_train_fitness": train_fit,
            "seed_program_test_fitness": test_fit,
        }

    return {
        "participant_id": participant_id,
        "train_acc": train_eval["accuracy"],
        "test_acc": test_eval["accuracy"],
        "train_loglik": None,
        "test_loglik": None,
        "train_mse": None,
        "test_mse": None,
        "train_fitness": train_eval["accuracy"],
        "test_fitness": test_eval["accuracy"],
        "seed_program_train_fitness": train_eval["accuracy"],
        "seed_program_test_fitness": test_eval["accuracy"],
    }


def _round_floats_for_csv_row(row: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    """Round finite floats for CSV output; keep ints, None, bools, and str keys unchanged."""
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
    """Persist interpreter + argv under run_dir/log/command.txt (path also printed for SLURM logs)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "log"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "command.txt"
    cmd = shlex.join([sys.executable, *sys.argv])
    stamp = datetime.now().isoformat(timespec="seconds")
    body = f"# saved {stamp}\n# cwd: {os.getcwd()}\n# host: {socket.gethostname()}\n{cmd}\n"
    path.write_text(body, encoding="utf-8")
    return path


def _load_trials_for_participant(
    *,
    args: argparse.Namespace,
    participant_id: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[int, Any]]]:
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
        return train_trials, test_trials, test_observed_blocks

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


def _evaluate_participant(
    *,
    chooser: "CentaurChooser",
    args: argparse.Namespace,
    participant_id: int,
) -> Dict[str, Any]:
    train_trials_raw, test_trials_raw, test_observed_blocks = _load_trials_for_participant(
        args=args, participant_id=participant_id
    )
    train_trials = _prepare_trials_for_centaur(args.dataset, train_trials_raw)
    test_trials = _prepare_trials_for_centaur(args.dataset, test_trials_raw)

    if args.dataset == "choice13k":
        train_eval = evaluate_centaur_on_trials(
            chooser,
            train_trials,
            n_seeds=args.n_eval_seeds,
            debug_prob=args.debug_prob,
            debug_limit=args.debug_prob_limit,
        )
        test_eval = evaluate_centaur_on_trials(
            chooser,
            test_trials,
            n_seeds=args.n_eval_seeds,
            debug_prob=args.debug_prob,
            debug_limit=args.debug_prob_limit,
        )
        return _participant_summary_row(
            participant_id=participant_id,
            dataset=args.dataset,
            fitness_metric=args.fitness_metric,
            train_eval=train_eval,
            test_eval=test_eval,
        )

    if args.dataset == "cpc18" and bool(getattr(args, "cpc18_official_mse", False)):
        train_eval = _eval_accuracy_from_probs(chooser, train_trials)
        test_eval = _eval_accuracy_from_probs(chooser, test_trials)
        train_mse_eval = _eval_cpc18_mse_from_probs(
            chooser, train_trials, test_observed_blocks or {}
        )
        test_mse_eval = _eval_cpc18_mse_from_probs(
            chooser, test_trials, test_observed_blocks or {}
        )
        return _participant_summary_row(
            participant_id=participant_id,
            dataset=args.dataset,
            fitness_metric=args.fitness_metric,
            train_eval=train_eval,
            test_eval=test_eval,
            train_mse_eval=train_mse_eval,
            test_mse_eval=test_mse_eval,
        )
    if args.dataset == "cpc18":
        train_eval = evaluate_centaur_on_trials(
            chooser,
            train_trials,
            n_seeds=args.n_eval_seeds,
            debug_prob=args.debug_prob,
            debug_limit=args.debug_prob_limit,
        )
        test_eval = evaluate_centaur_on_trials(
            chooser,
            test_trials,
            n_seeds=args.n_eval_seeds,
            debug_prob=args.debug_prob,
            debug_limit=args.debug_prob_limit,
        )
        return _participant_summary_row(
            participant_id=participant_id,
            dataset=args.dataset,
            fitness_metric=args.fitness_metric,
            train_eval=train_eval,
            test_eval=test_eval,
            train_mse_eval=None,
            test_mse_eval=None,
        )

    train_eval = _eval_accuracy_from_probs(chooser, train_trials)
    test_eval = _eval_accuracy_from_probs(chooser, test_trials)
    return _participant_summary_row(
        participant_id=participant_id,
        dataset=args.dataset,
        fitness_metric=args.fitness_metric,
        train_eval=train_eval,
        test_eval=test_eval,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Centaur baseline (TE-compatible data args).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="choice13k",
        choices=["choice13k", "cpc18", "mixed_gambles"],
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="For cpc18: directory containing Track II files (default datasets/cpc18 when value is 'data').",
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default=None,
        help="Ignored (TE compatibility). Centaur does not use a seed program file.",
    )
    parser.add_argument("--n_iterations", type=int, default=1, help="Ignored (TE compatibility).")
    parser.add_argument("--n_candidates", type=int, default=1, help="Ignored (TE compatibility).")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Ignored (TE compatibility). Use --centaur_model for the HF Centaur checkpoint.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "local"],
        help="Ignored (TE compatibility).",
    )
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
        "--mixed_gambles_csv",
        type=str,
        default="datasets/mixed_gambles/data_all_2021-01-08.csv",
        help="CSV path for mixed_gambles dataset.",
    )
    parser.add_argument(
        "--cpc18_official_mse",
        action="store_true",
        help="CPC18: use official all-trials block MSE. Default: per-participant held-out split (use --fitness_metric loglik).",
    )
    parser.add_argument("--n_eval_seeds", type=int, default=1)
    parser.add_argument(
        "--debug_prob",
        action="store_true",
        default=False,
        help="Print why p=0.5 fallbacks happen (exception vs invalid denominator).",
    )
    parser.add_argument(
        "--debug_prob_limit",
        type=int,
        default=5,
        help="Max number of per-trial debug lines for each fallback type (default: 5).",
    )
    parser.add_argument(
        "--centaur_model",
        type=str,
        default="marcelbinz/Llama-3.1-Centaur-70B-adapter",
        help="Hugging Face model id (Unsloth adapter or compatible).",
    )
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if not (0.0 < args.split_ratio < 1.0):
        print("Error: --split_ratio must be in (0,1).")
        sys.exit(1)
    if args.fitness_metric == "loglik" and args.dataset not in {"choice13k", "mixed_gambles"} and not (
        args.dataset == "cpc18" and not args.cpc18_official_mse
    ):
        print("Error: --fitness_metric loglik needs choice13k/mixed_gambles, or cpc18 without --cpc18_official_mse.")
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
        else str(REPO_ROOT / "generated_outputs" / args.dataset / "centaur" / f"run_{timestamp}")
    )

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

    cmd_log = _write_command_line_log(base_run_dir)
    print(f"Wrote full command line to {cmd_log}")

    chooser = CentaurChooser(args.centaur_model, max_seq_length=args.max_seq_length)

    # ----- across_participants (same construction as TE) -----
    if args.split_mode == "across_participants":
        if len(participants) < 2:
            print("Error: across_participants requires >= 2 selected participants.")
            sys.exit(1)
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
        print(f"Across-participants: train trials={len(train_trials)}, test trials={len(test_trials)}")
        t0 = datetime.now()
        train_eval = evaluate_centaur_on_trials(
            chooser,
            train_trials,
            n_seeds=args.n_eval_seeds,
            debug_prob=args.debug_prob,
            debug_limit=args.debug_prob_limit,
        )
        test_eval = evaluate_centaur_on_trials(
            chooser,
            test_trials,
            n_seeds=args.n_eval_seeds,
            debug_prob=args.debug_prob,
            debug_limit=args.debug_prob_limit,
        )
        runtime = (datetime.now() - t0).total_seconds()
        base_run_dir.mkdir(parents=True, exist_ok=True)
        summ = _participant_summary_row(
            participant_id=0,
            dataset="choice13k",
            fitness_metric=args.fitness_metric,
            train_eval=train_eval,
            test_eval=test_eval,
        )
        row = {
            "participant_id": 0,
            "train_fitness": summ["train_fitness"],
            "test_fitness": summ["test_fitness"],
            "total_runtime": runtime,
            "seed_program_train_fitness": summ["seed_program_train_fitness"],
            "seed_program_test_fitness": summ["seed_program_test_fitness"],
        }
        row_ll = {
            "participant_id": 0,
            "train_loglik": summ["train_loglik"],
            "test_loglik": summ["test_loglik"],
        }
        _write_all_mode_csvs(base_run_dir, [row], [row_ll])
        print(f"Wrote CSVs under {base_run_dir}")
        return

    # ----- participant_scope=all -----
    if args.participant_scope == "all":
        participant_details: List[Dict[str, Any]] = []
        participant_loglik: List[Dict[str, Any]] = []
        for pid in tqdm(participants, desc="Participants"):
            t0 = datetime.now()
            summ = _evaluate_participant(chooser=chooser, args=args, participant_id=pid)
            runtime = (datetime.now() - t0).total_seconds()
            participant_details.append(
                {
                    "participant_id": pid,
                    "train_fitness": summ["train_fitness"],
                    "test_fitness": summ["test_fitness"],
                    "total_runtime": runtime,
                    "seed_program_train_fitness": summ["seed_program_train_fitness"],
                    "seed_program_test_fitness": summ["seed_program_test_fitness"],
                }
            )
            participant_loglik.append(
                {"participant_id": pid, "train_loglik": summ["train_loglik"], "test_loglik": summ["test_loglik"]}
            )
            _write_all_mode_csvs(base_run_dir, participant_details, participant_loglik)
        print(f"Wrote CSVs under {base_run_dir}")
        return

    # ----- single / range: participants_summary + loglik aggregates -----
    participants_summary: List[Dict[str, Any]] = []
    participants_loglik_summary: List[Dict[str, Any]] = []
    base_run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = base_run_dir / "participants_summary.csv"
    summary_loglik_file = base_run_dir / "summary_loglik.csv"
    details_loglik_file = base_run_dir / "participant_details_loglik.csv"

    for pid in tqdm(participants, desc="Participants"):
        summ = _evaluate_participant(chooser=chooser, args=args, participant_id=pid)
        participants_summary.append(summ)
        participants_loglik_summary.append(
            {"participant_id": pid, "train_loglik": summ["train_loglik"], "test_loglik": summ["test_loglik"]}
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

    print(f"Wrote {summary_file} and loglik summaries under {base_run_dir}")


if __name__ == "__main__":
    main()
