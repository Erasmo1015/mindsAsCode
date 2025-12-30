import numpy as np
from typing import List, Dict, Any, Callable


def evaluate_program(choose_fn: Callable, trials: List[Dict[str, Any]]) -> Dict[str, float]:
    correct = 0
    total = 0
    for t in trials:
        try:
            pred = choose_fn(t["problem"], t["history"])
        except Exception:
            pred = None
        total += 1
        if pred is not None and pred == t["action"]:
            correct += 1
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "total": total}


def aggregate_predictions(programs: List[Callable], weights: np.ndarray, trials: List[Dict[str, Any]]) -> float:
    correct = 0
    total = len(trials)
    for t in trials:
        option_count = len(t["problem"]["option_keys"])
        votes = np.zeros(option_count)
        for fn, w in zip(programs, weights):
            try:
                pred = fn(t["problem"], t["history"])
                if pred is not None and isinstance(pred, int) and 0 <= pred < option_count:
                    votes[pred] += w
            except Exception:
                continue
        pred_idx = int(np.argmax(votes)) if votes.sum() > 0 else -1
        if pred_idx == t["action"]:
            correct += 1
    return correct / total if total > 0 else 0.0

