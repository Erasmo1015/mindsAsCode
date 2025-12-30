def choose(problem, history):
    # Parameters
    alpha = 0.85
    lam = 2.0
    kappa = 0.15

    # Reconstruct reference point
    reference = 0.0
    if problem.get("has_feedback", False):
        for h in history:
            if h.get("feedback") is not None:
                reference = (1 - kappa) * reference + kappa * h["feedback"]

    def value(x):
        if x >= reference:
            return (x - reference) ** alpha
        else:
            return -lam * (reference - x) ** alpha

    def expected_value(outcomes, probs):
        total = 0.0
        for o, p in zip(outcomes, probs):
            total += p * value(o)
        return total

    A = problem["gamble_A"]
    B = problem["gamble_B"]

    A_outcomes = A["rewards"]
    B_outcomes = B["rewards"]
    B_outcomes = B["rewards"]

    if A["probs"] is None:
        A_probs = [1.0 / len(A_outcomes)] * len(A_outcomes)
    else:
        A_probs = A["probs"]

    if B["probs"] is None:
        B_probs = [1.0 / len(B_outcomes)] * len(B_outcomes)
    else:
        B_probs = B["probs"]

    ua = expected_value(A_outcomes, A_probs)
    ub = expected_value(B_outcomes, B_probs)

    return 1 if ub > ua else 0
