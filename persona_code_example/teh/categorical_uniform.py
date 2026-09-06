def choose(problem, history):
    options = problem["options"]
    p = 1.0 / len(options)
    return {option["action"]: p for option in options}
