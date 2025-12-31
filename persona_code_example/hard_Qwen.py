def choose(problem, history):
    """
    problem: dict with keys
        - gamble_A: {"probs": [...], "rewards": [...]}
        - gamble_B: {"probs": [...], "rewards": [...]}
        - option_keys
        - has_feedback: bool
    history: list of dicts with keys
        - action: int (0 or 1)
        - feedback: float or None
    return: int (0 or 1)
    """

    # ----------------------------
    # Parameters (fixed)
    # ----------------------------
    alpha = 0.88
    beta = 0.75
    theta = 0.9
    epsilon = 0.1
    delta = 0.5
    kappa = 0.2

    # ----------------------------
    # Reference point
    # ----------------------------
    reference = 0.0
    if problem.get("has_feedback", False):
        for h in history:
            if h.get("feedback") is not None:
                reference = (1 - kappa) * reference + kappa * h["feedback"]

    # ----------------------------
    # Subjective value computation
    # ----------------------------
    def subjective_value(outcomes, probs):
        sv = 0.0
        for x, p in zip(outcomes, probs):
            sv += p * ((x - reference) ** alpha if x - reference >= 0 else -beta * (reference - x) ** alpha)
        return sv

    # ----------------------------
    # Extract gambles
    # ----------------------------
    A = problem["gamble_A"]
    B = problem["gamble_B"]

    A_probs = A["probs"]
    A_rewards = A["rewards"]

    B_probs = B["probs"]
    B_rewards = B["rewards"]

    # ----------------------------
    # Compute subjective values
    # ----------------------------
    sv_A = subjective_value(A_rewards, A_probs)
    sv_B = subjective_value(B_rewards, B_probs)

    # ----------------------------
    # Satisficing adjustment
    # ----------------------------
    if abs(sv_A - sv_B) < delta:
        # Prefer the option with less risk
        var_A = sum((x - sum(A_rewards) / len(A_rewards)) ** 2 for x in A_rewards)
        var_B = sum((x - sum(B_rewards) / len(B_rewards)) ** 2 for x in B_rewards)
        if var_A <= var_B:
            sv_B += epsilon
        else:
            sv_A += epsilon

    # ----------------------------
    # Deterministic decision
    # ----------------------------
    if sv_B > sv_A:
        return 1
    return 0