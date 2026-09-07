# Steyvers, Lee, & Wagenmakers (2009) bandit

Steyvers, M., Lee, M. D., & Wagenmakers, E.-J. (2009). A Bayesian analysis of human decision-making on bandit problems. *Journal of Mathematical Psychology*, 53, 168–179.

Four-armed bandit. Each of 451 people played 20 games of 15 trials. Reward rates are fixed across people and given in `gameRewardRates.csv`.

## Files

- `trials.csv` — 135,300 rows
- `gameRewardRates.csv` — 20 games × 4 arms

## Columns (`trials.csv`)

- `participant` 1–451
- `game` 1–20
- `trial` 1–15
- `choice` 1–4 (arm)
- `reward` 0/1
- `rewardRateChosen` — generating probability of the chosen arm on that game
