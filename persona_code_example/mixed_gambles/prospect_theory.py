def choose(problem, history):
    alpha = 0.88
    beta_loss = 0.88
    loss_aversion = 2.0
    gamma = 0.65
    decision_scale = 0.15

    def clip(p):
        if p < 1e-6:
            return 1e-6
        if p > 1.0 - 1e-6:
            return 1.0 - 1e-6
        return p

    def value(x):
        if x >= 0:
            return x ** alpha
        return -loss_aversion * ((-x) ** beta_loss)

    def weight(p):
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        pg = p ** gamma
        qg = (1.0 - p) ** gamma
        return pg / ((pg + qg) ** (1.0 / gamma))

    def probs_for(gamble):
        rewards = gamble["rewards"]
        probs = gamble.get("probs")
        if probs is None:
            return [1.0 / len(rewards)] * len(rewards)
        return probs

    def subjective_value(gamble):
        rewards = gamble["rewards"]
        probs = probs_for(gamble)
        total = 0.0
        for p, r in zip(probs, rewards):
            total += weight(p) * value(r)
        return total

    sv_A = subjective_value(problem["gamble_A"])
    sv_B = subjective_value(problem["gamble_B"])

    score = decision_scale * (sv_B - sv_A)

    if score > 50:
        p = 1.0
    elif score < -50:
        p = 0.0
    else:
        p = 1.0 / (1.0 + 2.718281828 ** (-score))

    return clip(p)