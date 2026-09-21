# ICLR paper terminology

## Finalized naming rules

These rules supersede older wording in historical notes and implementation documentation.

1. In the manuscript, call the complete method **Population-to-Individual Cognitive Synthesis (PICS)**. In professor-facing progress reports, **ICLR Method** is also acceptable. Do not write “PICS v3” unless distinguishing implementation versions in an internal note.
2. Describe the data protocol as an **8:2 train--test split** with at most **40 training observations per participant** and at most **30 participants per dataset**. The test set remains untouched until final evaluation.
3. Treat `train_val` only as a legacy implementation field name. In the paper, write **training data**, **training score**, or **training behavioral log-likelihood**. Do not introduce a validation set or a refinement stage. Paper **training data** \(D^{\mathrm{train}}\) ≡ implementation **`train_val`** (retained `train∪val` after the loader’s 60/20/20 split). The manuscript presents this as the 80% training half of an **8:2 train--test** protocol; the internal `val` slice is not a validation set for model selection. The budget \(B=40\) (`limited_train_val`) caps that paper-training / `train_val` set. Implementation-only per-split `train` / `val` diagnostics are not paper-facing sets.
4. Call the paired target-population component **dual-track target-population evolution**. Its two tracks are the **target-only track** and the **transfer-conditioned track**. “Dual-track” describes the two matched search pathways; it does not assert simultaneous hardware execution.
5. Call the gate the **training-based transfer gate**. It selects the winning target-population track using trial-pooled training log-likelihood.
6. The gate retains the **complete winning elite program pool**. Its highest-scoring program is the sole population parent for participant-level exploration, while the complete retained pool initializes participant-level evolution.
7. Programs return a **predictive probability distribution over the available actions**. For each available action, \(p_P(a\mid x_t,h_t)\in[0,1]\), and the probabilities sum to one. For binary tasks, this reduces to the scalar \(p_P(x_t,h_t)\in[0,1]\) for action 1.
8. Reuse EMNLP terminology whenever the scientific concept is unchanged. Hide implementation labels such as G.1/G.2/G.3, TV, `rank-1`, `suffix`, `KIND`, and schema filenames from main-paper prose.

## 1. Existing concepts: reuse the EMNLP terminology

| Impl Name | Paper Name | Meaning | Source of Paper Name |
| --- | --- | --- | --- |
| `pics_v3` | **Population-to-Individual Cognitive Synthesis (PICS)**; **ICLR Method** in professor-facing reports | The complete finalized ICLR approach. | PICS/EMNLP paper + finalized ICLR naming |
| `choose(problem, history)` | **individual cognitive program** \(P_i\) | An executable program mapping the current decision context and behavioral history to a predictive choice distribution. | PICS/EMNLP paper |
| `problem`, `history`, `action` | **decision context** \(x_t\), **behavioral history** \(h_t\), **observed action** \(a_t\) | The information available before trial \(t\), the preceding behavioral record, and the participant’s observed choice. | PICS/EMNLP paper |
| Returned scalar/vector/dictionary | **predictive probability distribution** \(p_P(\cdot\mid x_t,h_t)\) | A distribution over the available action set \(\mathcal A_t\), with \(p_P(a\mid x_t,h_t)\in[0,1]\) and \(\sum_{a\in\mathcal A_t}p_P(a\mid x_t,h_t)=1\). For binary tasks, \(p_P(x_t,h_t)\) denotes the probability of action 1. | Generalized PICS representation for binary and multi-action datasets |
| `seed_program.py` | **seed program** \(P_0\) | The neutral initial program, which returns a uniform probability distribution over the available actions. | PICS/EMNLP paper + finalized notation |
| `candidate`, `parent`, `child` | **candidate program**, **parent program**, **child program** | Programs evaluated during evolutionary search and their generation relationships. | PICS/EMNLP paper; standard evolutionary-computation terminology |
| `global_phase`, G.1 | **source-population synthesis** | Evolutionary search over training observations pooled within a dataset to construct reusable source population programs. | PICS population phase, specialized for the ICLR transfer pipeline |
| `global_phase/best_program.py`, `rank-1` | **highest-scoring source population program** \(P_d^{\mathrm{pop}}\) | The source population candidate with the highest trial-pooled training log-likelihood for dataset \(d\). | PICS/EMNLP paper; `rank-1` remains implementation language |
| `global_elite_pool`, `elite_pool_size` | **elite program pool** \(\mathcal E\) | The retained set of high-scoring candidate programs available for parent selection or downstream initialization. | PICS/EMNLP paper |
| `sample_parents`, `sample_size` | **multi-parent evolutionary search** | The LLM generates child programs conditioned jointly on multiple selected parent programs. | PICS/EMNLP paper |
| `fresh_n_candidates` and its decay | **seed-based exploration** within an **exploration-to-exploitation schedule** | Early iterations generate more candidates from \(P_0\); later iterations increasingly use high-scoring programs from the evolving pool. | PICS/EMNLP paper |
| Automatic dataset prompt | **dataset-specific program-generation prompt** | An automatically constructed instruction specifying the dataset interface, action convention, available information, and executable-program requirements. | PICS/EMNLP paper |
| `prefer_auto_llm_prompt` | **dataset-adaptive prompting** | Automatic adaptation of the program-generation prompt to each dataset using task metadata and representative training trials. | PICS/EMNLP paper |
| `runtime_valid`, invalid candidate | **executability filtering** | Compilation, runtime, and probability-distribution checks applied before candidate selection. | PICS/EMNLP paper |
| G.3, `explore_candidates=50`, `rank-1 only` | **best-program-conditioned participant exploration** | Generates diverse participant-conditioned candidates from the gate-winning highest-scoring target-population program before iterative participant evolution. | PICS participant-level exploration + finalized ICLR conditioning rule |
| Participant TEH, `n_iterations=10` | **participant-level multi-parent evolution** | Iteratively adapts the retained target-population candidates to one participant using that participant’s training observations. | PICS/EMNLP paper |
| `selection_score`, internal `train_val` score | **training behavioral log-likelihood** or **training fitness** \(S_{\mathrm{train}}(P)\) | The score used to rank and select programs during synthesis. The internal field name is not paper terminology. | Finalized 8:2 train--test protocol |
| Internal `TV` data | **training data** \(D_i^{\mathrm{train}}\) | All participant observations permitted to influence prompt construction, synthesis, gating, and selection. Do not write “TV” in the paper. | Finalized 8:2 train--test protocol |
| Internal `pooled TV` | **pooled training observations** \(D_d^{\mathrm{train}}=\bigcup_iD_{di}^{\mathrm{train}}\) | The union of training trials across participants in dataset \(d\). | PICS pooled population evaluation + finalized protocol |
| `final/mean_test_loglik` | **average held-out per-trial log-likelihood** | Final predictive performance averaged across participants. | PICS/EMNLP paper |
| Final participant `best_program.py` | **selected individual cognitive program** \(P_{di}^{\star}\) | The final program selected for participant \(i\) in dataset \(d\) using training fitness. | PICS/EMNLP paper + finalized protocol |
| `test reporting only` | **held-out test evaluation** | Test observations evaluate an already selected program and never affect synthesis, source selection, gating, or program selection. | PICS/EMNLP paper |

## 2. New ICLR concepts: paper-facing terminology

| Impl Name | Paper Name | Meaning | Source of Paper Name |
| --- | --- | --- | --- |
| `structure_aware_v3` | **structure-aware sparse-observation protocol** | Preserves each dataset’s problem, block, episode, or temporal organization when constructing train--test splits and behavioral histories. | New descriptive PICS term |
| `SA40`, `limited_train_val=40` | **40-observation training setting** or **training-observation budget \(B=40\)** | Retains at most 40 training observations per participant after the grouped 8:2 train--test split; the test set remains unchanged. “SA40” may be used only as a compact appendix or table label after definition. | Finalized sparse-observation protocol |
| Participant cap | **up to 30 participants per dataset** | The finalized ICLR evaluation includes at most 30 eligible participants from each dataset. | Finalized experimental protocol |
| Independent / Resetting / Continuous families | **independent-trial**, **episodic**, and **continuous sequential** task structures | Distinguishes history-free trials, units with history resets, and uninterrupted temporal sequences. | Standard sequential-decision terminology adapted to the data organizations |
| `sanitize_problem_for_choose` | **pre-decision information set** | Program input restricted to information available before the current choice. | Standard decision-modeling terminology |
| Removed current/future outcome fields | **information-leakage control** | Excludes labels, future outcomes, correctness, and oracle variables from program inputs and prompts. | Standard evaluation terminology |
| `prompt_snapshots`, JSON snapshots | **structured behavioral context** or **structured training examples** | Serialized decision contexts, bounded behavioral histories, and observed actions supplied to the LLM. | New descriptive PICS term |
| `schema_v4`, historical six-construct artifacts | **historical cognitive-construct annotation schema** | Earlier annotations include explicit risk. Preserve their provenance; do not relabel them as the final schema. | Historical PICS artifact terminology |
| `schema_v5`, five constructs | **five-construct cognitive vocabulary** \(\mathcal C\) | The finalized source-selection vocabulary: history, value, probability use, feedback, and learning. | Finalized ICLR taxonomy |
| Motif presence fields | **program-level construct indicators** \(X_{dkc}\in\{0,1\}\) | Indicates whether source-population program \(k\) from dataset \(d\) explicitly instantiates construct \(c\). | Finalized source-selection formulation |
| Candidate/reference annotation | **evolutionary program-transition annotation** | Identifies cognitive constructs introduced, removed, or modified by a structural program edit. Use for participant-evolution analysis when applicable. | New descriptive PICS term; related to structural transfer analysis in genetic programming ([ScienceDirect][2]) |
| Construct counts \(k/n\) | **construct occurrence rate** | Dataset-level frequency with which a construct appears among annotated candidate population programs. | New descriptive term |
| \(\widetilde p=(k+0.5)/(n+1)\) | **Jeffreys-smoothed construct occurrence rate** | Stabilized Bernoulli occurrence estimate before cross-dataset comparison. | Standard Jeffreys-prior terminology |
| MoM \(\tau^2\), shrinkage toward \(\alpha\) | **empirical-Bayes shrinkage** or **partial pooling of construct occurrence rates** | Dataset-specific occurrence estimates borrow strength from the cross-dataset distribution while preserving reliable dataset-specific variation. | Standard empirical-Bayes terminology ([arXiv][3]); hierarchical cognitive modeling uses related logic ([Lee][4]) |
| `Occurrence-EB` | **empirical-Bayes construct-profile source selection** | Represents each dataset using a five-dimensional shrinkage-adjusted construct-occurrence profile and selects a source by profile similarity. It is not the participant mixed-effects model. | New PICS name combining empirical-Bayes shrinkage with source selection |
| 1,445 annotations across 15 datasets | **1,445 annotated candidate population programs across 15 datasets** | The evidence used to estimate the finalized five-dimensional task representations. | Finalized ICLR source-selection analysis |
| Six-source allowlist | **candidate source-task set** \(\mathcal S\) | The predefined datasets eligible to provide cross-task knowledge. | Standard transfer-learning terminology |
| Self-exclusion | **leave-target-out source selection** | Excludes the target dataset itself from its candidate source-task set. | New descriptive term |
| `argmax cosine` | **construct-profile similarity** | Cosine similarity between empirical-Bayes construct profiles determines the selected source task. | Task-similarity-based source selection in transfer learning ([ScienceDirect][5]) |
| Frozen source YAML, no transfer peek | **transfer-outcome-independent source selection** | The source map is selected and frozen without observing target-transfer or test performance. | New descriptive PICS term; related source-selection framing ([ACL Anthology][6]) |
| G.2 overall | **dual-track target-population evolution** | Evolves target-population programs along two matched search tracks and applies a training-based transfer gate. | Finalized ICLR component name |
| G.2 control arm/branch | **target-only track** \(P_t^{\mathrm{target},\star}\) | Evolves target-population programs from \(P_0\) using target training data, the target prompt, and the common search budget. | Finalized dual-track terminology |
| G.2 transfer arm/branch | **transfer-conditioned track** \(P_t^{\mathrm{transfer},\star}\) | Uses the same target training data, initialization, and search budget while additionally conditioning generation on the selected source-task context. | Finalized dual-track terminology |
| `source suffix` | **source-task context** | The selected source population program, source-task schema, transfer instruction, and one source training demonstration included in the generation prompt. | New descriptive PICS term |
| One source `TV` example | **one-shot source-task training demonstration** | A single labeled source training example included in the LLM context. | Standard in-context-learning terminology ([Brown et al.][7]) |
| Source is context, not seed/parent | **prompt-mediated cross-task transfer** | Source knowledge conditions target program generation without inserting the source program into the target candidate pool. | New PICS term distinguishing the method from parameter, instance, and direct program transfer |
| `g2_paired_pack_pics_v3` | **context-budget-matched dual-track search** | Both tracks use the same target training examples, initialization, parent-context cardinality, and search budget; source-task context is the designed difference. | Finalized matched dual-track design |
| `paired_parent_count` | **matched parent-context cardinality** | Holds the number of target parent programs constant across the two tracks while allowing their identities to differ. | New descriptive term |
| Internal `count-pooled train_val` | **trial-pooled training log-likelihood** \(S_{\mathrm{train}}(P;D_t^{\mathrm{train}})\) | Mean log-likelihood over the pooled target training trials, weighting participants in proportion to their numbers of retained trials. | Standard pooled-likelihood description + finalized protocol |
| Equal-person mean | **participant-averaged held-out log-likelihood** | Computes each participant’s held-out mean first and then averages participants equally; used for final reporting, not gating. | Standard participant-level aggregation description |
| G.2 `gate` | **training-based transfer gate** | Retains the transfer-conditioned track only when its highest-scoring program has strictly higher trial-pooled training log-likelihood than the target-only track; otherwise retains the target-only track. | Finalized PICS term; selective-transfer safeguard against negative transfer ([Wang et al.][8]) |
| `S_tr`, `S_ctl` | **transfer-conditioned training score** \(S_t^{\mathrm{transfer}}\) and **target-only training score** \(S_t^{\mathrm{target}}\) | Scores compared by the training-based transfer gate. Avoid abbreviated `tr`, which can be confused with “training.” | Finalized PICS notation |
| Tie/failure → control | **conservative target-only fallback** | Retains the target-only track unless the transfer-conditioned track is valid and strictly better by training log-likelihood. | Negative-transfer prevention terminology ([Wang et al.][8]) |
| Winner track’s rank-1 program | **gate-winning highest-scoring target-population program** \(P_t^\star\) | The highest-scoring program in the retained track. It is the sole population parent for participant-level exploration. | Finalized PICS search design |
| Winner track’s full elite pool | **retained target-population elite pool** \(\mathcal E_t^\star\) | The complete elite pool from the gate-winning track. It initializes participant-level multi-parent evolution. | Finalized PICS search design |
| G.3 `rank-1 only` | **best-program-conditioned participant exploration** | Participant exploration is conditioned only on \(P_t^\star\). | Extension of the PICS participant-exploration terminology |
| No source suffix after G.2 | **target-only participant specialization** | Cross-task context is converted into target-population programs before participant search; participant synthesis subsequently uses target programs and target training observations only. | New descriptive PICS term |
| Paper “350” schedule | **nominal candidate-generation budget of 350 programs** | Counts 100 source-population candidates, 100 candidates across both target-population tracks, 50 participant-exploration candidates, and 100 participant-evolution candidates. Never call this “350 iterations.” | Finalized search-budget description |
| `max_model_len=16384` | **maximum context length of 16,384 tokens** | Context limit used by the finalized experiments. | Finalized implementation setting |
| Overflow packing order | **training-trial-first context truncation** | When the prompt exceeds the context budget, PICS truncates training-trial examples before truncating parent-program context. | Finalized prompt-packing policy |
| “co-evolutionary search” | **Do not use; write dual-track target-population evolution** | The tracks do not coadapt or define one another’s fitness, so cooperative-coevolution terminology is incorrect. | Standard cooperative-coevolution definition ([Potter and De Jong][9]) |

## Core ICLR component names

The three central new component names are:

1. **Empirical-Bayes construct-profile source selection**
2. **Dual-track target-population evolution**
3. **Training-based transfer gate**

The participant stages retain the PICS terminology **best-program-conditioned participant exploration** and **participant-level multi-parent evolution**.

## Terms to avoid in the manuscript

- PICS v3, T-PICS, G.1, G.2, or G.3, except in internal implementation notes.
- Training-validation, observed training-validation, TV, or `train_val` as paper-facing data terminology.
- Validation set, validation-guided selection, or refinement stage.
- Control arm, transfer arm, control branch, or transfer branch.
- Observed-data transfer gate; use **training-based transfer gate**.
- Gated target-population synthesis, when referring to the full component; use **dual-track target-population evolution**.
- Co-evolutionary search.
- Rank-1, suffix, paired pack, allowlist, or implementation filenames in main-paper prose.
- Uniform choice probability when multiple actions are possible; use **uniform probability distribution over the available actions**.

[1]: https://nlp.cs.berkeley.edu/pubs/Liang-Jordan-Klein_2010_Programs_paper.pdf "Learning Programs: A Hierarchical Bayesian Approach"
[2]: https://www.sciencedirect.com/science/article/pii/S0031320324010409 "Semantics-guided multi-task genetic programming for multi-output regression"
[3]: https://arxiv.org/abs/1409.2677 "Two Modeling Strategies for Empirical Bayes Estimation"
[4]: https://sites.socsci.uci.edu/~mdlee/Lee2011.pdf "How cognitive modeling can benefit from hierarchical Bayesian models"
[5]: https://www.sciencedirect.com/science/article/pii/S0031320317302881 "On automated source selection for transfer learning in convolutional neural networks"
[6]: https://aclanthology.org/2021.emnlp-main.689/ "To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning"
[7]: https://arxiv.org/abs/2005.14165 "Language Models are Few-Shot Learners"
[8]: https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.html "Characterizing and Avoiding Negative Transfer"
[9]: https://dl.acm.org/doi/pdf/10.1162/106365600568086 "An Architecture for Evolving Coadapted Subcomponents"
