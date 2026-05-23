def choose(problem, history):
    """
    Logistic gain-loss seed for mixed gambles.

    problem: dict with keys
        - gamble_A: {"probs": [...], "rewards": [...]}   # risky gamble / accept
        - gamble_B: {"probs": [...], "rewards": [...]}   # certain option / reject
        - option_keys
        - has_feedback: bool

    history: unused

    return: float, probability of choosing option 1
            option 1 = gamble_B / certain option / reject gamble
    """

    import math

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

    # Gain magnitude G: largest positive outcome
    G = max([r for r in rewards if r > 0.0], default=0.0)

    # Loss magnitude L: absolute value of most negative outcome
    L = abs(min([r for r in rewards if r < 0.0], default=0.0))

    # ----------------------------
    # Logistic gain-loss score
    # ----------------------------
    utility_raw = G - omega * L
    score = lam * utility_raw + bias

    # score > 0 favors option 0 = gamble_A / accept
    # therefore P(option 1) should increase when score is negative
    diff = -score

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