#!/usr/bin/env python3
"""Generate concise Choices13k participant evidence CSV (no plots)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_trials(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def problem_key(trial: Dict[str, Any]) -> str:
    return json.dumps(trial["problem"], sort_keys=True)


def split_blocks(trials: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for t in trials:
        k = problem_key(t)
        if k not in grouped:
            grouped[k] = []
            order.append(k)
        grouped[k].append(t)
    return [grouped[k] for k in order]


def rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else float("nan")


def compute_behavior_stats(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    blocks = split_blocks(trials)
    total_pairs = 0
    repeats = 0
    switches = 0
    neg_total = 0
    neg_repeats = 0
    no_fb_pairs = 0
    no_fb_repeats = 0

    for block in blocks:
        if len(block) < 2:
            continue
        has_feedback = bool(block[0]["problem"].get("has_feedback", False))
        for i in range(1, len(block)):
            prev = block[i - 1]
            cur = block[i]
            prev_a = int(prev["action"])
            cur_a = int(cur["action"])
            total_pairs += 1
            if cur_a == prev_a:
                repeats += 1
            else:
                switches += 1

            if not has_feedback:
                no_fb_pairs += 1
                if cur_a == prev_a:
                    no_fb_repeats += 1

            prev_feedback = None
            if cur.get("history"):
                prev_feedback = cur["history"][-1].get("feedback")
            if prev_feedback is not None and float(prev_feedback) < 0:
                neg_total += 1
                if cur_a == prev_a:
                    neg_repeats += 1

    return {
        "repeat_rate": rate(repeats, total_pairs),
        "switch_count": float(switches),
        "after_negative_repeat_rate": rate(neg_repeats, neg_total),
        "no_feedback_repeat_rate": rate(no_fb_repeats, no_fb_pairs),
    }


def fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.4f}"


def pick_json(base_name: str) -> Path:
    root = repo_root()
    candidates = [
        root / "analysis" / "code" / "choices13k" / base_name,
        root / "analysis" / "data" / "choices13k" / base_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing required JSON: {base_name}")


def main() -> None:
    rows_spec = {
        2: {
            "heldout_delta": 0.7038,
            "ours_test_loglik": -0.0839,
            "centaur_test_loglik": -0.7877,
            "program_mechanism": "Recent-action persistence: last-5 majority ±0.3; last action ±0.2; feedback ±0.15",
            "main_interpretation": "Strong within-problem persistence; recent actions can dominate EV.",
        },
        4: {
            "heldout_delta": 0.1428,
            "ours_test_loglik": -0.0645,
            "centaur_test_loglik": -0.2073,
            "program_mechanism": "Choice-count persistence: past action counts used twice; no feedback variable used",
            "main_interpretation": "Self-reinforcing choice habit; repeats choices even after bad outcomes.",
        },
    }

    out_path = repo_root() / "analysis" / "analysis_plot" / "proposal" / "choices13k_program_evidence.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "participant_id",
        "heldout_delta",
        "ours_test_loglik",
        "centaur_test_loglik",
        "train_repeat_rate",
        "test_repeat_rate",
        "train_switch_count",
        "test_switch_count",
        "train_after_negative_repeat_rate",
        "test_after_negative_repeat_rate",
        "train_no_feedback_repeat_rate",
        "test_no_feedback_repeat_rate",
        "program_mechanism",
        "main_interpretation",
    ]

    rows: List[Dict[str, str]] = []
    for pid, meta in rows_spec.items():
        train_trials = load_trials(pick_json(f"participant_{pid}_train_trials.json"))
        test_trials = load_trials(pick_json(f"participant_{pid}_test_trials.json"))
        tr = compute_behavior_stats(train_trials)
        te = compute_behavior_stats(test_trials)
        rows.append(
            {
                "participant_id": str(pid),
                "heldout_delta": f"{meta['heldout_delta']:+.4f}",
                "ours_test_loglik": f"{meta['ours_test_loglik']:.4f}",
                "centaur_test_loglik": f"{meta['centaur_test_loglik']:.4f}",
                "train_repeat_rate": fmt(tr["repeat_rate"]),
                "test_repeat_rate": fmt(te["repeat_rate"]),
                "train_switch_count": f"{int(tr['switch_count'])}",
                "test_switch_count": f"{int(te['switch_count'])}",
                "train_after_negative_repeat_rate": fmt(tr["after_negative_repeat_rate"]),
                "test_after_negative_repeat_rate": fmt(te["after_negative_repeat_rate"]),
                "train_no_feedback_repeat_rate": fmt(tr["no_feedback_repeat_rate"]),
                "test_no_feedback_repeat_rate": fmt(te["no_feedback_repeat_rate"]),
                "program_mechanism": meta["program_mechanism"],
                "main_interpretation": meta["main_interpretation"],
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
