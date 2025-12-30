from typing import List, Dict, Any


def format_trials_to_text(trials: List[Dict[str, Any]]) -> str:
    """Convert Choice13k trials to numbered text similar to gridROTE formatting."""
    lines = []
    for idx, t in enumerate(trials):
        prob_a = t["problem"]["gamble_A"]["probs"]
        rew_a = t["problem"]["gamble_A"]["rewards"]
        prob_b = t["problem"]["gamble_B"]["probs"]
        rew_b = t["problem"]["gamble_B"]["rewards"]
        has_fb = t["problem"].get("has_feedback", False)
        action = t["action"]
        lines.append(
            f"{idx+1}. Problem: Option A probs {prob_a} rewards {rew_a}; "
            f"Option B probs {prob_b} rewards {rew_b}; has_feedback={has_fb}; "
            f"Observed action: {action}"
        )
    return "\n".join(lines)

