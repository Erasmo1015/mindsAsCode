# ICLR paper terminology: implementation to manuscript

This guide follows the current Method section edited on 23 September 2026. Use the manuscript's wording in the paper and figures; use G.1/G.2 and code names only to locate implementation artifacts. Older EMNLP and PICS v3 notes do not override the current Method text.

## Current paper vocabulary

| Implementation or older wording | Current manuscript wording | Meaning |
| --- | --- | --- |
| `pics_v3`, T-PICS | **Population-to-Individual Cognitive Synthesis (PICS)** | Name of the complete method. |
| Dataset-adaptive or dataset-specific prompt | **Dataset-adapted prompt** or **dataset-adapted prompt construction** | An LLM constructs the prompt from the task description and representative training trials. |
| Source population / population, when referring to code | **Dataset-focused population program**, **population program**, or **population program pool** | Say “program” when the object is executable code; use “pool” for the set of candidate programs. |
| G.1 source population job | **Single dataset-focused population program evolution** | Constructs candidate population programs independently from each dataset's pooled training observations. These programs supply the source library and cognitive-construct annotations. |
| Motif labels | **Cognitive constructs** | The five paper labels are **history, value, probability, feedback, and learning**. The code label `probability_used` becomes **probability** in prose and figures. |
| Occurrence-EB source map | **Cognitive construct profile** and **similar dataset selection** | A five-dimensional empirical-Bayes profile represents each dataset; cosine similarity selects another dataset as a source. |
| G.2 control arm; older “target-only track” | **Separately evolved dataset-focused population programs** | A fresh dataset-focused search on target training observations supplies the choice gate's comparison. It is distinct from the G.1 library-building run. |
| G.2 transfer arm; older “transfer-conditioned track” | **Transfer-based population program evolution** | A separate search on target training observations, with the selected similar dataset's program, task schema, and one training demonstration in the generation prompt. The source program is context, not a seed or a target-pool parent. |
| Transfer gate / training-based transfer gate | **Choice gate** | Retains the transfer-based program pool only if its highest-ranked program has strictly higher trial-pooled training log-likelihood than the separately evolved dataset-focused comparison; otherwise retains the dataset-focused pool. |
| G.3 exploration | **Participant-level exploration** | Generates participant-level candidates from the retained highest-ranked population program. |
| G.4 participant evolution | **Participant-level multi-parent program evolution** | Starts from the retained elite pool merged with participant-level candidates and evolves individual cognitive programs. |
| `rank-1`, `selected/` | **Highest-ranked population program**, **retained population program pool** | The highest-ranked retained program is the sole population parent for participant-level exploration; the complete retained elite pool initializes participant-level evolution. |

The conceptual paper narrative is **dataset-adapted prompt → dataset-focused population programs → similar dataset selection → transfer-based population programs and choice gate → participant-level exploration → participant-level multi-parent program evolution**. This narrative does not say the two target searches run simultaneously. In the implementation, G.1 library construction precedes the two fresh G.2 target searches; the G.2 searches are executed sequentially.

### Two uses of “dataset-focused”

The current manuscript uses **dataset-focused population programs** both for the G.1 source library and for the separately evolved G.2 comparison at the choice gate. These use the same broad approach—search using one dataset's own training observations—but are **different runs**. G.1 uses 10 iterations per dataset; the two G.2 searches each use 5 iterations on the target dataset. Keep this distinction explicit in implementation-facing notes and appendix budgets. In the main paper, the phrase **“separately evolved dataset-focused population programs”** distinguishes the gate comparison without introducing G.1/G.2 labels.

The G.1 target program is not an input to the reported choice gate. That gate compares the two G.2 highest-ranked programs on target training observations. Do not describe it as selecting the best program across G.1 and G.2.

## Cognitive constructs and source selection

| Paper construct | Implementation label | Presence means the program... |
| --- | --- | --- |
| History | `history` | Uses prior choices or trial records to affect a current decision. |
| Value | `value` | Scores or compares options by attractiveness, utility, or payoff. |
| Probability | `probability_used` | Uses task probability, likelihood, or odds information. |
| Feedback | `feedback` | Uses realized past outcomes to affect a later choice. |
| Learning | `learning` | Updates or reconstructs an internal estimate or decision rule from experience. |

The paper should state that an LLM annotates **1,414 candidate dataset-focused population programs across 15 datasets** (schema-v5 / g5e50p30; not the historical schema-v4 count of 1,445). Their construct occurrence estimates, with empirical-Bayes shrinkage, form the five-dimensional profile \(\mathbf v_d\). The paper uses \(s^\star(t)\) for the non-target dataset with the highest profile cosine similarity. In the current Method, \(\mathcal S\) is defined as the 15 evaluated datasets and \(s\ne t\) leaves 14 possible sources; the **runtime freeze** used by final mains still restricts selection to the **six-source allowlist** in `occurrence_eb_schema5_iter10_pics_v3.yaml`. Treat the frozen YAML as the implementation authority for which sources are eligible.

Do not call source selection a participant mixed-effects model. The empirical-Bayes construct profiles summarize candidate population programs; the separate joint analysis of participant candidate edits uses different observations and notation.

## Data, scores, and program notation

| Implementation | Paper term or symbol | Interpretation |
| --- | --- | --- |
| `structure_aware_v3`, grouped splitter | **Training and held-out test sets in an 8:2 ratio** | Repeated trials from the same problem or block are grouped before splitting. The paper's Experiment Setup explains the grouping. |
| `limited_train_val=40`, SA40 | **At most 40 training observations per participant** | The held-out test set is reserved for evaluation. |
| Participant range | **Up to 30 participants per dataset** | Dataset-specific exceptions can be stated in Experiment Setup. |
| `train_val`, pooled train∪val | **Training data**, **training score**, **training behavioral log-likelihood** | `train_val` is an internal field name; do not describe a validation set or refinement stage in this paper. |
| `pool_best_selection_score`, gate `pooled_train_val_loglik` | **Trial-pooled training log-likelihood** | Mean log-likelihood across pooled target training trials; used for population program ranking and the choice gate. |
| Per-person final test score | **Held-out per-trial log-likelihood** | Final performance is averaged over participants as stated in the Experiments section. Test observations do not decide source selection, the choice gate, or program selection. |
| `seed_program.py` | **Uniform seed program** \(P_0\) | Returns a uniform probability distribution over the available actions, including multi-action tasks. |
| `problem`, `history`, `action` | **Decision context** \(x_t\), **behavioral history** \(h_t\), **observed action** \(a_t\) | Inputs and observed response for a trial. |
| `choose(problem, history)` | **Individual cognitive program** | An executable program mapping \((x_t,h_t)\) to \(p_P(\cdot\mid x_t,h_t)\), a predictive distribution over available actions. |
| Returned scalar/vector/dict | \(p_P(a\mid x_t,h_t)\in[0,1]\), \(\sum_{a\in\mathcal A_t}p_P(a\mid x_t,h_t)=1\) | Paper representation of a valid choice distribution. |
| `sample_parents`, `fresh_n_candidates` | **Multiple parent programs**; **seed-based exploratory candidates**; **exploration-to-exploitation schedule** | Early iterations generate more candidates from \(P_0\), while later iterations use high-performing programs in the pool. |
| `runtime_valid` | **Executability filtering** | Invalid programs are excluded before selection. LLM token likelihood is not the behavioral fitness. |

## Search budgets: appendix-facing facts

| Implementation stage | Candidate generations | Scope |
| --- | ---: | --- |
| G.1 dataset-focused library construction | \(10\times10=100\) | Per dataset; built separately from the target comparison. |
| G.2 separately evolved dataset-focused target search | \(5\times10=50\) | Per target dataset. |
| G.2 transfer-based target search | \(5\times10=50\) | Per target dataset; matched to the G.2 dataset-focused search. |
| Participant-level exploration | 50 | Per participant. |
| Participant-level multi-parent program evolution | \(10\times10=100\) | Per participant. |

The commonly cited \(100+50+50+50+100=350\) counts one G.1 block, both G.2 searches, and one participant's stages **once each**. It is a candidate-generation accounting convention, not 350 iterations or the full job cost over all participants. Population program searches are shared across participants; participant searches repeat for each person. The final runs use a maximum model context length of 16,384 tokens. Under prompt overflow, training-trial examples are truncated before parent-program context.

## Exact paper wording for an implementation-sensitive distinction

The current Method uses this wording for the choice gate:

> A \emph{choice gate} retains the transfer-based population program pool produced by the transfer-based approach only when its highest-ranked program achieves strictly higher training log-likelihood than \zc{that of the separately evolved dataset-focused population programs}; otherwise, it retains the dataset-focused population pool.

This wording identifies the G.2 dataset-focused comparison. The earlier dataset-focused source-library search is a distinct run, even though the paper applies the same name to its population programs.

## Terms to avoid in manuscript prose

- PICS v3, T-PICS, G.1/G.2/G.3/G.4, `train_val`, TV, `rank-1`, suffix, paired pack, schema filenames, or job paths, except where implementation detail is explicitly needed in an appendix.
- Dataset-adaptive or dataset-specific **prompt**; use **dataset-adapted prompt**.
- Probability use as the paper construct label; use **probability**. Keep `probability_used` as the code label.
- Dual-track target-population evolution, target-only track, transfer-conditioned track, or control/transfer arm as component headings. Use the current Method's dataset-focused, transfer-based, and choice-gate language.
- Training-based transfer gate as the official gate name; use **choice gate**.
- Validation-based selection or refinement stage. The paper presents an 8:2 train–test split.
- Uniform *binary* choice probability when the method covers multiple actions; use **uniform probability distribution over the available actions**.

The current Method draft still contains one old phrase, **“dual-track target-population evolution,”** immediately after the source-selection equation. Replace it in the manuscript before treating that paragraph as finalized. Keep Arunesh's source-selection heading and his intended contrast between dataset-focused and transfer-based population program evolution.


[1]: https://nlp.cs.berkeley.edu/pubs/Liang-Jordan-Klein_2010_Programs_paper.pdf "Learning Programs: A Hierarchical Bayesian Approach"
[2]: https://www.sciencedirect.com/science/article/pii/S0031320324010409 "Semantics-guided multi-task genetic programming for multi-output regression"
[3]: https://arxiv.org/abs/1409.2677 "Two Modeling Strategies for Empirical Bayes Estimation"
[4]: https://sites.socsci.uci.edu/~mdlee/Lee2011.pdf "How cognitive modeling can benefit from hierarchical Bayesian models"
[5]: https://www.sciencedirect.com/science/article/pii/S0031320317302881 "On automated source selection for transfer learning in convolutional neural networks"
[6]: https://aclanthology.org/2021.emnlp-main.689/ "To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning"
[7]: https://arxiv.org/abs/2005.14165 "Language Models are Few-Shot Learners"
[8]: https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.html "Characterizing and Avoiding Negative Transfer"
[9]: https://dl.acm.org/doi/pdf/10.1162/106365600568086 "An Architecture for Evolving Coadapted Subcomponents"