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
    lambda_loss = 2.0
    gamma = 0.65
    kappa = 0.2
    delta = 0.5
    phi = 0.2

    # ----------------------------
    # Reconstruct reference point
    # ----------------------------
    reference = 0.0
    if problem.get("has_feedback", False):
        for h in history:
            if h.get("feedback") is not None:
                reference = (1 - kappa) * reference + kappa * h["feedback"]

    # ----------------------------
    # Prospect value
    # ----------------------------
    def value(x):
        if x >= reference:
            return (x - reference) ** alpha
        else:
            return -lambda_loss * (reference - x) ** alpha

    # ----------------------------
    # Probability weighting
    # ----------------------------
    def weight(p):
        if p <= 0.0:
            p = 1e-10
        if p >= 1.0:
            p = 1.0
        return (-(-__import__("math").log(p)) ** gamma).__rpow__(__import__("math").e)

    # ----------------------------
    # Subjective value
    # ----------------------------
    def subjective_value(outcomes, probs):
        sv = 0.0
        for x, p in zip(outcomes, probs):
            sv += weight(p) * value(x)
        return sv

    # ----------------------------
    # Extract gambles
    # ----------------------------
    A = problem["gamble_A"]
    B = problem["gamble_B"]

    A_outcomes = A["rewards"]
    B_outcomes = B["rewards"]

    if A["probs"] is None:
        A_probs = [1.0 / len(A_outcomes)] * len(A_outcomes)
    else:
        A_probs = A["probs"]

    if B["probs"] is None:
        B_probs = [1.0 / len(B_outcomes)] * len(B_outcomes)
    else:
        B_probs = B["probs"]

    # ----------------------------
    # Compute subjective values
    # ----------------------------
    sv_A = subjective_value(A_outcomes, A_probs)
    sv_B = subjective_value(B_outcomes, B_probs)

    # ----------------------------
    # Satisficing adjustment
    # ----------------------------
    if abs(sv_A - sv_B) < delta:
        # Treat option with smaller outcome variance as "safe"
        var_A = sum((x - sum(A_outcomes) / len(A_outcomes)) ** 2 for x in A_outcomes)
        var_B = sum((x - sum(B_outcomes) / len(B_outcomes)) ** 2 for x in B_outcomes)
        if var_A <= var_B:
            sv_A += phi
        else:
            sv_B += phi

    # ----------------------------
    # Deterministic decision
    # ----------------------------
    if sv_B > sv_A:
        return 1
    return 0
