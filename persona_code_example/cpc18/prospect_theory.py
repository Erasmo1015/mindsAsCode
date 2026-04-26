def choose(problem, history):
    """
    CPC18-compatible decision program (Track II).

    problem: dict with keys
        - Ha: float
        - pHa: float
        - La: float
        - LotShapeA: str
        - LotNumA: int

        - Hb: float
        - pHb: float
        - Lb: float
        - LotShapeB: str
        - LotNumB: int

        - Amb: int
        - Corr: int

    history: list of dicts with keys
        - action: int (0 = A, 1 = B)
        - feedback: float or None

    return: float, probability of choosing option 1 (B)
    """

    import math

    # ----------------------------
    # Parameters (fixed)
    # ----------------------------
    alpha = 0.88
    lambda_loss = 2.0
    gamma = 0.65
    kappa = 0.2
    delta = 0.5
    phi = 0.2
    beta = 1.0   # NEW: soft decision temperature

    # ----------------------------
    # Reconstruct reference point
    # ----------------------------
    reference = 0.0
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
        p = min(max(p, 1e-10), 1.0)
        return math.exp(-(-math.log(p)) ** gamma)

    # ----------------------------
    # Construct lottery outcomes
    # ----------------------------
    def build_lottery(H, pH, L, shape, n):
        if shape == '-' or n <= 2:
            return [H, L], [pH, 1 - pH]

        outcomes = []
        probs = []

        for i in range(n):
            t = i / (n - 1)
            x = L + t * (H - L)
            outcomes.append(x)

            if shape == 'Symm':
                probs.append(1.0)
            elif shape == 'L-skew':
                probs.append(math.exp(-2 * t))
            elif shape == 'R-skew':
                probs.append(math.exp(-2 * (1 - t)))
            else:
                probs.append(1.0)

        Z = sum(probs)
        probs = [p / Z for p in probs]

        return outcomes, probs

    # ----------------------------
    # Build gambles A and B
    # ----------------------------
    A_outcomes, A_probs = build_lottery(
        problem["Ha"], problem["pHa"], problem["La"],
        problem["LotShapeA"], problem["LotNumA"]
    )

    B_outcomes, B_probs = build_lottery(
        problem["Hb"], problem["pHb"], problem["Lb"],
        problem["LotShapeB"], problem["LotNumB"]
    )

    # ----------------------------
    # Subjective values
    # ----------------------------
    def subjective_value(outcomes, probs):
        return sum(weight(p) * value(x) for x, p in zip(outcomes, probs))

    sv_A = subjective_value(A_outcomes, A_probs)
    sv_B = subjective_value(B_outcomes, B_probs)

    # ----------------------------
    # Satisficing adjustment
    # ----------------------------
    if abs(sv_A - sv_B) < delta:
        mean_A = sum(A_outcomes) / len(A_outcomes)
        mean_B = sum(B_outcomes) / len(B_outcomes)

        var_A = sum((x - mean_A) ** 2 for x in A_outcomes)
        var_B = sum((x - mean_B) ** 2 for x in B_outcomes)

        if var_A <= var_B:
            sv_A += phi
        else:
            sv_B += phi

    # ----------------------------
    # Probabilistic decision (NEW)
    # ----------------------------
    diff = sv_B - sv_A

    try:
        p_choose_1 = 1.0 / (1.0 + math.exp(-beta * diff))
    except OverflowError:
        p_choose_1 = 1.0 if diff > 0 else 0.0

    # Clamp for numerical stability in loglik
    if p_choose_1 < 1e-9:
        p_choose_1 = 1e-9
    elif p_choose_1 > 1.0 - 1e-9:
        p_choose_1 = 1.0 - 1e-9

    return p_choose_1