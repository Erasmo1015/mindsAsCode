"""Generate and evaluate fast, leakage-free programs for three repaired TEH datasets."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teh import (
    _trials_for_loglik_participant,
    compile_program,
    evaluate_choice13k_program,
)


DATASETS = {
    "5speekenbrink2008learning": REPO_ROOT
    / "generated_outputs_old/psych101_train/teh/5speekenbrink2008learning/run_260706_223004",
    "10frey2017risk": REPO_ROOT
    / "generated_outputs_old/psych101_train/teh/10frey2017risk/run_260706_224551",
    "12badham2017deficits": REPO_ROOT
    / "generated_outputs_old/psych101_train/teh/12badham2017deficits/run_260706_232606",
}


def _evaluate(source: str, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    choose_fn = compile_program(source)
    if choose_fn is None:
        raise RuntimeError("Generated program did not compile")
    result = evaluate_choice13k_program(choose_fn, trials, n_seeds=1)
    if result["errors"]:
        raise RuntimeError(f"Generated program had {result['errors']} evaluation errors")
    return result


def _participant_ids(run_dir: Path) -> List[int]:
    csv_path = run_dir / "participant_details_loglik.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [int(row["participant_id"]) for row in csv.DictReader(handle)]


def _cards_key(cards: Iterable[Any]) -> str:
    return ",".join(str(int(card)) for card in sorted(cards))


def _weather_source(
    train_trials: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    history_weight: float,
) -> str:
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for trial in train_trials:
        key = _cards_key(trial["problem"].get("cards") or [])
        action = int(trial["action"])
        counts[key][action] += 1
        total[action] += 1
    prior = (total[1] + 1.0) / (sum(total) + 2.0)
    frozen = {key: value for key, value in sorted(counts.items())}
    return f"""def choose(problem, history):
    cards = [int(x) for x in problem.get("cards", [])]
    cards.sort()
    key = ",".join(str(x) for x in cards)
    counts = {frozen!r}
    base = counts.get(key, [0, 0])
    n0 = float(base[0])
    n1 = float(base[1])
    history_weight = {history_weight!r}
    if history_weight:
        for entry in history:
            past_cards = [int(x) for x in entry.get("cards", [])]
            past_cards.sort()
            if past_cards == cards:
                if int(entry.get("action", -1)) == 1:
                    n1 += history_weight
                elif int(entry.get("action", -1)) == 0:
                    n0 += history_weight
    alpha = {alpha!r}
    prior = {prior!r}
    p = (n1 + alpha * prior) / (n0 + n1 + alpha)
    return max(1e-6, min(1.0 - 1e-6, p))
"""


def _risk_source(
    train_trials: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    monotone: bool,
) -> str:
    counts: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for trial in train_trials:
        pump_count = int(trial["problem"].get("pump_count_before", 0))
        action = int(trial["action"])
        counts[pump_count][action] += 1
        total[action] += 1
    prior = (total[1] + 1.0) / (sum(total) + 2.0)
    probs = {
        count: (actions[1] + alpha * prior) / (sum(actions) + alpha)
        for count, actions in counts.items()
    }
    if monotone:
        running = 0.0
        for count in sorted(probs):
            running = max(running, probs[count])
            probs[count] = running
    frozen = {key: value for key, value in sorted(probs.items())}
    return f"""def choose(problem, history):
    pump_count = int(problem.get("pump_count_before", 0))
    probabilities = {frozen!r}
    if pump_count in probabilities:
        p = probabilities[pump_count]
    else:
        lower = [k for k in probabilities if k <= pump_count]
        p = probabilities[max(lower)] if lower else {prior!r}
    return max(1e-6, min(1.0 - 1e-6, float(p)))
"""


def _badham_source(
    train_trials: Sequence[Dict[str, Any]],
    *,
    confidence: float,
    fallback: str,
) -> str:
    n1 = sum(int(trial["action"]) for trial in train_trials)
    prior = (n1 + 1.0) / (len(train_trials) + 2.0)
    fallback_probability = 0.5 if fallback == "chance" else prior
    return f"""def choose(problem, history):
    stimulus = problem.get("stimulus_features") or {{}}
    keys = [str(x).upper() for x in problem.get("option_keys", [])]
    confidence = {confidence!r}
    for entry in reversed(history):
        if (entry.get("stimulus_features") or {{}}) != stimulus:
            continue
        feedback = entry.get("feedback")
        correct = feedback.get("correct_category") if isinstance(feedback, dict) else None
        if correct is None:
            correct = entry.get("correct_category")
        correct = str(correct).upper() if correct is not None else ""
        if len(keys) >= 2 and correct == keys[1]:
            return confidence
        if len(keys) >= 2 and correct == keys[0]:
            return 1.0 - confidence
    return {fallback_probability!r}
"""


def _candidate_sources(
    dataset: str,
    train_trials: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if dataset == "5speekenbrink2008learning":
        for alpha in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            for history_weight in (0.0, 0.5, 1.0, 2.0):
                label = f"cue_table_alpha={alpha}_history_weight={history_weight}"
                candidates.append(
                    (
                        label,
                        _weather_source(
                            train_trials, alpha=alpha, history_weight=history_weight
                        ),
                    )
                )
    elif dataset == "10frey2017risk":
        for alpha in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            for monotone in (False, True):
                label = f"pump_hazard_alpha={alpha}_monotone={monotone}"
                candidates.append(
                    (
                        label,
                        _risk_source(train_trials, alpha=alpha, monotone=monotone),
                    )
                )
    elif dataset == "12badham2017deficits":
        for confidence in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98):
            for fallback in ("chance", "train_prior"):
                label = f"causal_feedback_lookup_conf={confidence}_fallback={fallback}"
                candidates.append(
                    (
                        label,
                        _badham_source(
                            train_trials, confidence=confidence, fallback=fallback
                        ),
                    )
                )
    else:
        raise ValueError(dataset)
    return candidates


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, output_name: str) -> Path:
    old_run = DATASETS[dataset]
    output_dir = old_run / output_name
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    rows: List[Dict[str, Any]] = []
    for participant_id in _participant_ids(old_run):
        train_trials, val_trials, test_trials = _trials_for_loglik_participant(
            dataset,
            participant_id,
            split_ratio=0.6,
            split_seed=0,
            psych_dataset_split="train",
        )
        scored: List[Tuple[float, str, str, Dict[str, Any]]] = []
        for label, source in _candidate_sources(dataset, train_trials):
            val_result = _evaluate(source, val_trials)
            scored.append((float(val_result["avg_loglik"]), label, source, val_result))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        val_loglik, label, source, val_result = scored[0]

        train_result = _evaluate(source, train_trials)
        test_result = _evaluate(source, test_trials)
        participant_dir = output_dir / f"participant_{participant_id}"
        participant_dir.mkdir()
        (participant_dir / "best_program.py").write_text(source, encoding="utf-8")
        result_record = {
            "participant_id": participant_id,
            "selection": label,
            "train": train_result,
            "val": val_result,
            "test": test_result,
            "selection_used_test": False,
        }
        (participant_dir / "results.json").write_text(
            json.dumps(result_record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "participant_id": participant_id,
                "train_loglik": round(float(train_result["avg_loglik"]), 6),
                "val_loglik": round(val_loglik, 6),
                "test_loglik": round(float(test_result["avg_loglik"]), 6),
                "gated_test_loglik": round(float(test_result["avg_loglik"]), 6),
                "train_acc": round(float(train_result["accuracy"]), 6),
                "val_acc": round(float(val_result["accuracy"]), 6),
                "test_acc": round(float(test_result["accuracy"]), 6),
                "selection": label,
            }
        )
        print(
            f"{dataset} participant {participant_id}: {label}; "
            f"val={val_loglik:.4f}, test={float(test_result['avg_loglik']):.4f}"
        )

    detail_fields = [
        "participant_id",
        "train_loglik",
        "val_loglik",
        "test_loglik",
        "gated_test_loglik",
        "train_acc",
        "val_acc",
        "test_acc",
        "selection",
    ]
    _write_csv(output_dir / "participant_details_loglik.csv", detail_fields, rows)
    summary = {
        "num_of_participants": len(rows),
        "avg_train_loglik": round(mean(row["train_loglik"] for row in rows), 6),
        "avg_val_loglik": round(mean(row["val_loglik"] for row in rows), 6),
        "avg_test_loglik": round(mean(row["test_loglik"] for row in rows), 6),
        "avg_gated_test_loglik": round(
            mean(row["gated_test_loglik"] for row in rows), 6
        ),
        "avg_test_acc": round(mean(row["test_acc"] for row in rows), 6),
    }
    _write_csv(output_dir / "summary_loglik.csv", list(summary), [summary])
    metadata = {
        "dataset": dataset,
        "source_run": str(old_run.relative_to(REPO_ROOT)),
        "method": "hand-authored causal heuristic family",
        "selection_protocol": "parameters fit on train; candidate selected by val loglik; test evaluated after selection",
        "split_ratio": 0.6,
        "split_seed": 0,
        "current_problem_post_choice_fields_used": False,
        "summary": summary,
    }
    (output_dir / "README.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--output_name", default="manual_causal_eval_260730")
    args = parser.parse_args()

    for dataset in args.datasets:
        path = run_dataset(dataset, args.output_name)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
