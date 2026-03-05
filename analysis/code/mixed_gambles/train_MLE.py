#!/usr/bin/env python3
"""
Fit the logistic gain–loss model by MLE for all participants in the mixed gambles dataset.

Model:
  utility = G - omega * L
  P(accept) = sigmoid(lambda * utility) = 1 / (1 + exp(-lambda * utility))

Fits (omega, lambda) per participant. Saves results to analysis/models/mixed_gambles/logistic_MLE_results.csv.
Optional --filter_mixed_gambles: keep only gain_loss trial type (default: disabled, use all trial types).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# Paths relative to repo root (script is analysis/code/mixed_gambles/train_MLE.py)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = REPO_ROOT / "datasets/mixed_gambles/data_all_2021-01-08.csv"
OUT_DIR = REPO_ROOT / "analysis/models/mixed_gambles"
OUT_CSV = OUT_DIR / "logistic_MLE_results.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_all_participant_trials(csv_path: Path, filter_gain_loss_only: bool = False):
    """Load CSV; return dict participant_id -> list of (G, L, y). G=gain, L=abs(loss), y=took_gamble (0/1).
    If filter_gain_loss_only is True, keep only gamble_type == 'gain_loss' trials (Section 4.2). Default: False (use all).
    """
    trials_by_participant = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filter_gain_loss_only and row.get("gamble_type") != "gain_loss":
                continue
            pid = int(row["subject"])
            G = float(row["gain"])
            L = abs(float(row["loss"]))
            y = int(row["took_gamble"])
            if pid not in trials_by_participant:
                trials_by_participant[pid] = []
            trials_by_participant[pid].append((G, L, y))
    return trials_by_participant


def nll(params: np.ndarray, G: np.ndarray, L: np.ndarray, y: np.ndarray) -> float:
    """Negative log likelihood. params = [omega, lambda]."""
    omega, lam = params[0], params[1]
    utility = G - omega * L
    p = sigmoid(lam * utility)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def fit_participant(G: np.ndarray, L: np.ndarray, y: np.ndarray):
    """Fit (omega, lambda) by MLE. Returns (omega_hat, lambda_hat, nll, accuracy)."""
    bounds = [(1e-5, 10.0), (1e-5, 20.0)]
    res = minimize(
        lambda p: nll(p, G, L, y),
        x0=[1.0, 1.0],
        method="L-BFGS-B",
        bounds=bounds,
    )
    omega_hat, lam_hat = res.x[0], res.x[1]
    utility = G - omega_hat * L
    p = sigmoid(lam_hat * utility)
    pred = (p >= 0.5).astype(np.float64)
    acc = np.mean(pred == y)
    return omega_hat, lam_hat, float(res.fun), float(acc)


def main():
    parser = argparse.ArgumentParser(description="Fit logistic gain-loss MLE for mixed gambles")
    parser.add_argument("--filter_mixed_gambles", action="store_true", help="Keep only gain_loss trial type (Section 4.2). Default: disabled (use all trial types).")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    trials_by_participant = load_all_participant_trials(CSV_PATH, filter_gain_loss_only=args.filter_mixed_gambles)
    if args.filter_mixed_gambles:
        print("[Mixed Gambles] Using gain_loss trials only.")
    participant_ids = sorted(trials_by_participant.keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in participant_ids:
        triples = trials_by_participant[pid]
        G = np.array([t[0] for t in triples], dtype=np.float64)
        L = np.array([t[1] for t in triples], dtype=np.float64)
        y = np.array([t[2] for t in triples], dtype=np.float64)
        omega_hat, lam_hat, nll_val, acc = fit_participant(G, L, y)
        rows.append({
            "participant_id": pid,
            "omega": omega_hat,
            "lambda": lam_hat,
            "nll": nll_val,
            "accuracy": acc,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["participant_id", "omega", "lambda", "nll", "accuracy"])
        w.writeheader()
        w.writerows(rows)

    print(f"Fitted MLE logistic model for {len(participant_ids)} participants.")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
