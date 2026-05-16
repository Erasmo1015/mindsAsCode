def choose(problem, history):
    def expected_utility(gamble):
        return sum(p * r for p, r in zip(gamble["probs"], gamble["rewards"]))
    
    def variance(gamble):
        ev = expected_utility(gamble)
        return sum(p * (r - ev) ** 2 for p, r in zip(gamble["probs"], gamble["rewards"]))
    
    # Calculate expected utilities and variances for both options
    util_A = expected_utility(problem["gamble_A"])
    util_B = expected_utility(problem["gamble_B"])
    var_A = variance(problem["gamble_A"])
    var_B = variance(problem["gamble_B"])
    
    # Initial probability based on expected utility difference
    base_prob_B = 0.5 + 0.5 * (util_B - util_A) / (abs(util_A) + abs(util_B) + 1e-6)
    
    # Adjust probability based on variance
    variance_adjustment = (var_A - var_B) / (var_A + var_B + 1e-6)  # Avoid division by zero
    base_prob_B += 0.2 * variance_adjustment  # Aggressive adjustment
    
    # Use history to slightly adjust the probability
    if history:
        recent_actions = [h['action'] for h in history[-5:]]  # Consider last 5 actions
        count_B = recent_actions.count(1)
        count_A = recent_actions.count(0)
        
        # Bias towards the most recent choice
        if count_B > count_A:
            base_prob_B += 0.3  # Stronger bias
        elif count_A > count_B:
            base_prob_B -= 0.3  # Stronger bias
        
        # Further adjustment based on the most recent action
        if history[-1]['action'] == 1:
            base_prob_B += 0.2
        elif history[-1]['action'] == 0:
            base_prob_B -= 0.2
    
    # If feedback is available, adjust the probability accordingly
    if problem["has_feedback"] and history:
        last_feedback = history[-1]['feedback']
        if last_feedback is not None:
            if last_feedback > 0:
                base_prob_B += 0.15  # Larger adjustment
            else:
                base_prob_B -= 0.15  # Larger adjustment
    
    # Additional adjustment based on the sign of the expected utility
    if util_B < util_A:
        base_prob_B *= 0.8  # Reduce probability if B is worse
    else:
        base_prob_B *= 1.2  # Increase probability if B is better
    
    # Ensure probability is within safe bounds
    prob_B = max(1e-6, min(1 - 1e-6, base_prob_B))
    
    return prob_B






