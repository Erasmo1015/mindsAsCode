def choose(problem, history):
    """
    Logistic gain–loss seed (deterministic boundary).

    Interpretable rule:
        utility_raw = G - omega * L
        score = lam * utility_raw + bias
        choose gamble iff score >= 0

    problem: dict with keys
        - gamble_A: {"probs": [...], "rewards": [...]}   # risky gamble (gain/loss)
        - gamble_B: {"probs": [...], "rewards": [...]}   # certain outcome
        - option_keys
        - has_feedback: bool
    history: unused
    return: int — TE option index: 0 = gamble_A (accept risky gamble), 1 = gamble_B (certain / reject gamble)
    """

    # ----------------------------
    # Parameters (fixed)
    # ----------------------------
    omega = 1.0
    lam = 1.0
    bias = 0.0

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
    # Deterministic decision boundary (logistic-style score)
    # ----------------------------
    utility_raw = G - omega * L
    score = lam * utility_raw + bias

    # Option A is the risky gamble; Option B is the certain outcome
    if score >= 0.0:
        return 0
    else:
        return 1