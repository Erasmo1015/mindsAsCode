def choose(problem, history):
    """
    Logistic gain–loss seed (probabilistic boundary).

    Interpretable rule:
        utility_raw = G - omega * L
        score = lam * utility_raw + bias
        choose gamble with probability sigmoid(score)

    problem: dict with keys
        - gamble_A: {"probs": [...], "rewards": [...]}   # risky gamble (gain/loss)
        - gamble_B: {"probs": [...], "rewards": [...]}   # certain outcome
        - option_keys
        - has_feedback: bool
    history: unused
    return: float — probability of choosing option 1 (TE option index: 1 = gamble_B, certain / reject gamble)
    """

    # ----------------------------
    # Parameters (fixed)
    # ----------------------------
    omega = 1.0
    lam = 1.0
    bias = 0.0
    beta = 1.0

    # ----------------------------
    # Extract gamble magnitudes G and L from Option A
    # ----------------------------
    A = problem["gamble_A"]
    rewards = A["rewards"]

    # Gain magnitude G: largest positive outcome (or 0 if none)
    G = max([r for r in rewards if r > 0.0], default=0.0)

    # Loss magnitude L: absolute value of most negative outcome (or 0 if none)
    L = abs(min([r for r in rewards if r < 0.0], default=0.0))

    # ----------------------------
    # Probabilistic decision boundary (logistic-style score)
    # ----------------------------
    utility_raw = G - omega * L
    score = lam * utility_raw + bias

    # Option A is the risky gamble; Option B is the certain outcome
    # Original deterministic rule:
    #   if score >= 0.0: return 0
    #   else: return 1
    # Therefore probability of choosing option 1 should decrease as score increases.
    try:
        p_choose_1 = 1.0 / (1.0 + __import__("math").exp(beta * score))
    except OverflowError:
        p_choose_1 = 0.0 if score > 0.0 else 1.0

    if p_choose_1 < 1e-9:
        p_choose_1 = 1e-9
    elif p_choose_1 > 1.0 - 1e-9:
        p_choose_1 = 1.0 - 1e-9

    return p_choose_1