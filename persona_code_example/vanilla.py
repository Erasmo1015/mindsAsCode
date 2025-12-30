def choose(problem, history):
    """
    problem: dict with gamble_A/gamble_B (probs, rewards), option_keys, has_feedback
    history: list of dicts with keys action (int) and feedback (float or None)
    return: int index (0 for Option A, 1 for Option B)
    """
    # Check if there are any high-value outcomes in Option B's reward distribution
    option_b_rewards = problem['gamble_B']['rewards']
    high_value_outcomes = [reward for reward in option_b_rewards if reward > 20]

    # If there are high-value outcomes in Option B, prefer Option B
    if high_value_outcomes:
        return 1
    else:
        # Otherwise, prefer Option A
        return 0