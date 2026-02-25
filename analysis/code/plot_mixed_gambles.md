# Mixed Gambles Plotting Script: Figure Descriptions

This document explains each figure produced by `plot_mixed_gambles.py`, which replicates the style of **Twelve Angry Models** (Michael Lee), Section 4.2 (Mixed Gambles). The script plots **one participant’s behavioral data** and **one evolved program’s predictions** in Gain–Loss (G–L) space.

---

## Data convention (matches Template_evo_non_strict)

- **Gamble A (risky):** 50/50 chance of **gain** or **loss**; `gamble_A["rewards"] = [gain, loss]` with `loss` typically negative in the CSV; `probs = [0.5, 0.5]`.
- **Gamble B (certain):** Sure outcome `cert`; `gamble_B["rewards"] = [cert]`, `probs = [1.0]`.
- **Action encoding:** `action = 1` means **accept the gamble** (choose A); `action = 0` means **take the certain outcome** (choose B).
- **G–L space:** For each trial we use **Gain** = first outcome of the risky gamble (positive), and **Loss** = magnitude of the second outcome (positive), so axes are (Loss, Gain) with both non-negative.

---

## Figure 4.3–like: Behavioral data scatter (G–L space)

**Filename suffix:** `_fig43_behavior.png`

### Purpose

Summarize the **participant’s choices** in the two-dimensional space defined by potential gain and potential loss. Each trial is one point; the plot shows where the participant accepted vs rejected the gamble.

### Book reference

> “Summary of behavioral data... each trial is characterized by two numbers, G and L... visualized as a two-dimensional plot.”

Interpretation examples from the book: *“Participant B always accepts gambles for which the potential gain is as large or greater than the potential loss... Participant L is highly and consistently risk averse... losses loom large.”*

### What is plotted

- **Axes:** **x = Loss** (magnitude, 0 to `loss_max`), **y = Gain** (0 to `gain_max`). Equal aspect ratio.
- **Green circles:** Trials where the **participant accepted** the gamble (`action == 1`). Drawn as open circles (no fill, green edge).
- **Red crosses (x):** Trials where the **participant rejected** the gamble (`action == 0`).
- **Dashed line:** **Gain = Loss** (diagonal). In the book this is a reference boundary (e.g. “accept if gain ≥ loss”); points above the line have gain &gt; loss, below have loss &gt; gain.

### How to read it

- Dots above the diagonal are “favorable” gambles (gain &gt; loss); dots below are “unfavorable.”
- A participant who mostly accepts above the line and rejects below is close to a “gain ≥ loss” rule; many red x’s above the line or green circles below suggest risk aversion or risk seeking relative to that rule.

---

## Figure 4.5–like: Program prediction at design points (descriptive adequacy)

**Filename suffix:** `_fig45_design_heatmap.png`

### Purpose

Show the **evolved program’s predictions** at every **unique (G, L) design point** in the chosen split (train or test). The plot compares the program’s accept/reject pattern to the participant’s **observed** accepts. There is no Bayesian posterior here; the “probability” is the **program’s deterministic output** (0 or 1) treated as P(accept) ∈ {0, 1}.

### Book reference

> “At each combination of gain and loss that defines a trial, the color of the square represents the posterior predictive probability of accepting the gamble... black markers are overlaid on the color squares for those gambles that were accepted by the participant.”

We replace “posterior predictive probability” with **program predicted probability** (0 or 1).

### What is plotted

- **Axes:** Same as Fig 4.3: **x = Loss**, **y = Gain**.
- **Colored squares:** One per **unique (G, L)** in the data. Color = program’s prediction:
  - **Red:** program predicts **reject** (P(accept) = 0).
  - **Green:** program predicts **accept** (P(accept) = 1).
  Colormap is RdYlGn with values in [0, 1].
- **Black circles:** Overlaid on the squares for trials where the **participant actually accepted** the gamble. So black circles on green squares = agreement (program and participant both accept); black circles on red squares = program underpredicts acceptance.

### How to read it

- **Agreement:** Black circles on green = correct accept; no black circle on red = correct reject (at that design point, participant rejected).
- **Mismatch:** Black circle on red = participant accepted but program predicted reject; green square with no black circle = program accepts but participant rejected at that (G, L). This gives a quick visual of descriptive adequacy over the **observed design**.

---

## Figure 4.6–like: Generalization heatmap (full stimulus space)

**Filename suffix:** `_fig46_generalization_heatmap.png`

### Purpose

Show the **program’s decision rule over a broader G–L space**, not only at the trials that were actually run. In the book this is the “posterior predictive” over a full grid; here it is the **program’s predicted accept/reject over a dense grid** (or, in `grid_mode=design`, only at design points). Participant data are overlaid for reference.

### Book reference

> “Posterior predictive distribution is now inferred for all gains from $1 to $16 combined with all losses from $1 to $32... colors represent probability of accepting... black markers indicate gambles accepted in experiment.”

Again we use **program predicted probability** (0/1) instead of a Bayesian posterior.

### What is plotted

- **Axes:** Same as above: **x = Loss**, **y = Gain** (0 to `loss_max` and `gain_max`).
- **Heatmap:**
  - **`grid_mode=full`:** Dense grid over **G = 1, 2, …, gain_max** and **L = 1, 2, …, loss_max**. For each (G, L) we build a synthetic `problem` (risky [G, −L] with 0.5/0.5, certain 0) and call `choose(problem, [])`. The result (0 or 1) is plotted as red (reject) or green (accept). So the heatmap is the **program’s decision boundary** over the full grid.
  - **`grid_mode=design`:** Same idea but only at the **unique (G, L)** design points from the data (scatter of colored squares, no full grid).
- **Black circles:** Participant’s **observed accepted** trials overlaid, so you can see where real choices fall in the program’s predicted landscape.
- **Dashed line:** **Gain = Loss**, as in the other figures.

### How to read it

- **Shape of the green region:** Shows where the program accepts; e.g. a diagonal band suggests a “gain ≥ loss”–like rule; a small green region suggests risk aversion.
- **Black dots in green:** Participant accepted there and program agrees.
- **Black dots in red:** Participant accepted there but program predicts reject (underprediction of acceptance).
- Useful for comparing different evolved programs (e.g. different iterations/candidates) on the same participant.

---

## Summary table

| Figure | Content | Red | Green | Black overlay | Reference line |
|--------|--------|-----|-------|----------------|----------------|
| **4.3** | Participant behavior | Rejected gambles (x) | Accepted gambles (o) | — | Gain = Loss |
| **4.5** | Program at design points | Program predicts reject | Program predicts accept | Participant accepted | — |
| **4.6** | Program over grid + data | Program predicts reject | Program predicts accept | Participant accepted | Gain = Loss |

---

## Output files and metadata

- **PNGs:** `mixed_gambles_p{id}_iter{iter}_cand{cand}[_train|_test]_fig43_behavior.png` (and `_fig45_`, `_fig46_`). When `--split both`, you get separate train and test versions for each figure.
- **Metadata JSON:** `..._metadata.json` records participant_id, split, program_path, iteration/candidate ids, dataset info (probs, accept_action, gain_max, loss_max), and counts/accept rates (participant and program) for reproducibility.

All figures use the same axis convention (**Loss** on x, **Gain** on y) and, where applicable, the same **Gain = Loss** reference line and **red → green** colormap for program predictions.
