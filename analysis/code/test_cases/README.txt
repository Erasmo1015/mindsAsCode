Usage:

From this folder:
  cd test_cases/
  python run_test_cases.py

Files (keep this folder together when copying to another server):
  run_test_cases.py    — loads files from the same directory as this script
  evolved_program.py   — evolved choose(problem, history)
  test_cases.json      — all 10 participant-2 test trials
  

Choice13k gamble problems (PSDD test slice)

problem
  gamble_A, gamble_B: each gamble has
    probs: list of outcome probabilities (may be None if unknown)
    rewards: list of outcome values (same length as probs when probs is set)
  has_feedback: whether feedback was shown on this block

history
  List of past trials on the same problem, in order. Each entry:
    action: 0 = Option A, 1 = Option B
    feedback: float or null

Returned value
  choose(problem, history) returns P(choose Option B), a float in (0, 1).
  Higher values mean the model predicts the participant is more likely to pick B.
