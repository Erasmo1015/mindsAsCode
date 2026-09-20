# ICLR paper terminology

## Finalized overrides

These overrides take precedence over older wording elsewhere in this file and in
historical notes:

1. Professor-facing reports call the complete approach the **ICLR Method** (not
   “PICS v3” or other implementation labels).
2. The **observed-data transfer gate** retains **one** highest-scoring
   target-population program, not a full elite pool.
3. That single program initializes **best-program-conditioned participant
   exploration**.

The main naming rule is: **reuse the EMNLP term whenever the scientific concept
is unchanged; hide implementation labels such as G.1/G.2/G.3, TV, `rank-1`,
`suffix`, and `KIND` from the main paper.**

## 1. Existing concepts: reuse the EMNLP terminology

| Implementation term | Paper-facing term | Meaning/usage | Origin/rationale |
| --- | --- | --- | --- |
| `pics_v3` | **ICLR Method** (professor-facing); **Population-to-Individual Cognitive Synthesis (PICS)** in manuscript drafts | The complete ICLR approach. Do not write “PICS v3” in professor-facing or main-paper prose unless distinguishing implementation versions in the appendix. | Finalized override + EMNLP paper |
| `choose(problem, history)` | **individual cognitive program** \(P_i\) | An executable program mapping the current decision context and behavioral history to a choice probability. | EMNLP paper |
| `problem`, `history`, `action` | **decision context** \(x_t\), **behavioral history** \(h_t\), and **observed action** \(a_t\) | The information available before trial \(t\), the preceding behavioral record, and the participant’s observed choice. | EMNLP paper |
| Returned scalar | **choice probability** \(p_P(x_t,h_t)\) | Probability assigned by program \(P\) to action 1. | EMNLP paper |
| `global_phase`, G.1 | **population phase** or **population-level program synthesis** | Evolutionary search over observations pooled across participants to discover reusable population-level program structures. | EMNLP paper; hierarchical program learning describes related-task sharing as sharing statistical strength ([nlp.cs.berkeley.edu][1]) |
| `global_phase/best_program.py`, `rank-1` | **highest-scoring population program** \(P_d^{\mathrm{pop}}\) | The population candidate with the highest pooled observed-data fitness for dataset \(d\). | EMNLP paper used “highest train-validation behavioral log-likelihood”; `rank-1` remains implementation language |
| `seed_program.py` | **seed program** \(P_{\mathrm{seed}}\) | The simple initial program from which evolutionary search begins. | EMNLP paper |
| `candidate`, `parent`, `child` | **candidate program**, **parent program**, **child program** | Programs evaluated during evolutionary search and their generation relationships. | EMNLP paper; standard evolutionary-computation terminology |
| `global_elite_pool`, `elite_pool_size` | **elite program pool** \(\mathcal E\) | Retained high-fitness candidate programs available for later parent selection. For the transfer gate, only the single highest-scoring program is retained (see Finalized overrides). | EMNLP paper + finalized override |
| `sample_parents`, `sample_size` | **multi-parent evolutionary search** | The LLM generates child programs conditioned on multiple selected parent programs. | EMNLP paper |
| `fresh_n_candidates` and its decay | **seed-based exploration** within an **exploration-to-exploitation schedule** | Early iterations generate more candidates from the seed; later iterations increasingly exploit strong programs. | EMNLP paper |
| Automatic dataset prompt | **dataset-specific program-generation prompt** | An automatically constructed instruction specifying the dataset interface, action convention, available information, and program requirements. | EMNLP paper |
| `prefer_auto_llm_prompt` | **dataset-adaptive prompting** | Automatic adaptation of the program-generation prompt to each dataset. | EMNLP paper |
| `runtime_valid`, invalid candidate | **executability filtering** | Compilation, runtime, and probability-output checks applied before candidate selection. | EMNLP paper |
| G.3, `explore_candidates=50` | **best-program-conditioned participant exploration** | Generation of diverse participant-conditioned candidates from the single gate-winning highest-scoring target-population program before iterative participant evolution. | EMNLP paper + finalized override |
| Participant TEH, `n_iterations=10` | **target-only participant specialization** / **participant-level multi-parent evolutionary search** | Iterative specialization after exploration, using only target programs and target observations. | EMNLP paper + finalized override |
| `train_val`, `selection_score` | **training-validation behavioral log-likelihood** | Fitness used to rank and select programs from observations available during synthesis. | EMNLP paper |
| `TV` | **observed training-validation data** \(D_i^{\mathrm{obs}}=D_i^{\mathrm{tr}}\cup D_i^{\mathrm{val}}\) | All observations for participant \(i\) that may influence synthesis and selection. Avoid “TV” in the main text. | EMNLP paper; \(D_i^{\mathrm{obs}}\) is the cleaner ICLR symbol |
| `pooled TV` | **pooled population observations** \(D_d^{\mathrm{obs}}=\bigcup_iD_{di}^{\mathrm{obs}}\) | The union of observed training-validation trials across participants in dataset \(d\). | EMNLP paper used “behavioral observations pooled across participants” |
| `final/mean_test_loglik` | **average held-out test behavioral log-likelihood** | Final predictive performance averaged across participants. | EMNLP paper |
| Final participant `best_program.py` | **selected individual cognitive program** \(P_{di}^{\star}\) | The final program selected for participant \(i\) in dataset \(d\). | EMNLP paper |
| `test reporting only` | **held-out test evaluation** | Test observations are used only to evaluate already selected programs and never affect synthesis or selection. | EMNLP paper |

## 2. New ICLR concepts: paper-facing terminology

| Implementation term | Paper-facing term | Meaning/usage | Origin/rationale |
| --- | --- | --- | --- |
| `structure_aware_v3` | **task-structure-aware sparse-observation protocol** | Constructs the observed set while preserving the dataset’s trial, episode, or temporal organization. | New descriptive PICS term; clearer than exposing the versioned protocol name |
| `SA40`, `limited_train_val=40` | **40-observation sparse-data setting** or **observation budget \(B=40\)** | Each participant contributes at most 40 combined training-validation observations; the test set is reserved separately. | New descriptive term. Use “SA40” only as a compact appendix/table label after defining \(B=40\) |
| Independent / Resetting / Continuous families | **independent-trial**, **episodic**, and **continuous sequential** task structures | Distinguishes history-free trials, units with history resets, and uninterrupted temporal sequences. | “Episodic” and “continuing” are standard sequential-decision terms; adapted here to the three data organizations |
| `sanitize_problem_for_choose` | **pre-decision information set** | The program input containing only information available before the current choice. | Standard decision-modeling concept; “sanitized problem” is implementation language |
| Removed current/future outcome fields | **information-leakage control** | Excludes labels, future outcomes, correctness, and oracle variables from program inputs and prompts. | Standard evaluation terminology |
| `prompt_snapshots`, full JSON snapshots | **structured behavioral context** or **structured trial examples** | Serialized decision contexts, bounded behavioral histories, and observed actions supplied to the LLM. | New descriptive term; avoids implementation-specific “snapshot contract” |
| `schema_v4`, six motifs | **cognitive-construct vocabulary** \(\mathcal C\) (schema-v4) | Schema-v4 annotations record six constructs including explicit risk. Frozen artifacts remain valid. | Existing Persona/PICS terminology |
| `schema_v5`, five constructs | **cognitive-construct vocabulary** \(\mathcal C\) (final) | Final ICLR Method taxonomy: history, value, probability use, feedback, learning. Explicit risk removed. Prompt `population_transition_v5`. Do not relabel v4 artifacts as v5. | Finalized five-construct schema |
| `modified_motifs`, motif presence | **program-level construct indicators** \(m_k(P)\in\{0,1\}\) | Whether cognitive construct \(k\) is explicitly instantiated in program \(P\). | New PICS formalization based on the cognitive-construct vocabulary |
| Candidate versus reference-parent annotation | **evolutionary program-transition annotation** | Identifies which cognitive constructs are introduced, removed, or modified by a structural program edit. | New descriptive term. Multi-task GP literature similarly studies knowledge transfer through program structure, but our construct annotation is distinct ([ScienceDirect][2]) |
| Motif counts \(k/n\) | **construct occurrence rate** | Dataset-level frequency with which a construct is present in annotated candidate population programs (\(X_{dje}\)). | New descriptive term |
| \(\tilde p=(k+0.5)/(n+1)\) | **Jeffreys-smoothed construct occurrence rate** | Stabilized Bernoulli occurrence estimate before cross-dataset comparison. | Standard Jeffreys-prior terminology |
| MoM \(\tau^2\), shrink \(b\) toward \(\alpha\) | **empirical-Bayes shrinkage** or **partial pooling of construct occurrence rates** | Dataset-specific occurrence estimates borrow strength from the cross-dataset distribution, with greater shrinkage for less informative datasets. | Standard empirical-Bayes terminology ([arxiv.org][3]); hierarchical cognitive modeling uses related logic ([sites.socsci.uci.edu][4]) |
| `Occurrence-EB` | **empirical-Bayes construct-profile source selection** | Represents every dataset by its shrunk construct-occurrence profile and selects a source using profile similarity. | New PICS name combining standard **empirical-Bayes shrinkage** with standard **source selection**. Do not call it a mixed-effects model: the finalized selector is not the earlier proposed MEM |
| Six-source allowlist | **candidate source-task set** \(\mathcal S\) | Predefined datasets eligible to provide cross-task knowledge. | Standard transfer-learning term “source task/domain” |
| Self-exclusion | **leave-target-out source selection** | Excludes the target dataset itself from its candidate source-task set. | New descriptive term |
| `argmax cosine` | **construct-profile similarity** | Cosine similarity between empirical-Bayes construct profiles determines the selected source task. | Task-similarity-based source selection is standard in transfer learning ([ScienceDirect][5]) |
| Frozen source YAML, no transfer peek | **transfer-outcome-independent source selection** | Source tasks are selected and frozen without observing gated-branch or test transfer performance. | New descriptive PICS term ([ACL Anthology][6]) |
| G.2 overall | **gated target-population synthesis** | Constructs two target-population programs and selects whether cross-task context should be retained. | New PICS component name |
| G.2 control arm | **target-only population synthesis branch** \(P_d^{\mathrm{base}}\) | Evolves the target population program using only target data and the target dataset prompt. | New descriptive term |
| G.2 transfer arm | **source-conditioned target-population synthesis branch** \(P_d^{\mathrm{src}}\) | Evolves a new target population program while conditioning candidate generation on selected source-task context. | New descriptive term ([nlp.cs.berkeley.edu][1]) |
| `source suffix` | **source-task context** | The selected source population program, source-task schema, transfer instruction, and one observed source example placed in the generation prompt. | New descriptive PICS term |
| One source TV example | **one-shot source-task demonstration** | A single labeled source training-validation example included in the LLM context. | “One-shot demonstration” follows standard in-context-learning terminology ([arxiv.org][7]) |
| Source is context, not seed/parent | **prompt-mediated cross-task transfer** | Source knowledge conditions target program generation without directly inserting the source program into the target candidate pool. | New PICS term; distinguishes our method from parameter, instance, or direct program transfer |
| `g2_paired_pack_pics_v3` | **prompt-budget-matched dual-branch search** | Both target-only and source-conditioned branches use the same target examples and parent count under a common context budget. | New PICS term; “paired packing” is implementation language |
| `paired_parent_count` | **matched parent-context cardinality** | Holds the number of parent programs constant across the two gated branches while allowing parent identities to differ. | New descriptive term |
| `count-pooled train_val` | **trial-pooled observed-data log-likelihood** \(S(P;D_d^{\mathrm{obs}})\) | Mean log-likelihood over the pooled union of observed target trials, equivalently weighting participants by their numbers of observed trials. | Standard pooled-likelihood description |
| Equal-person mean | **participant-averaged held-out log-likelihood** | Computes each participant’s mean first and then averages participants equally; used for final paper reporting, not the gate. | Standard participant-level aggregation description |
| G.2 `gate` | **observed-data transfer gate** | Selects the source-conditioned program only when it achieves higher trial-pooled observed-data log-likelihood than the target-only program; retains that single highest-scoring program. | Existing Persona-preferred term; selective-transfer safeguard against negative transfer ([openaccess.thecvf.com][8]) |
| `S_tr`, `S_ctl` | **source-conditioned score** \(S_d^{\mathrm{src}}\) and **target-only score** \(S_d^{\mathrm{base}}\) | Scores compared by the observed-data transfer gate. Avoid `tr`, which can be confused with “training.” | New notation proposed for PICS |
| Tie/failure → control | **conservative no-transfer fallback** / **conservative target-only fallback** | Retains the target-only branch unless the source-conditioned branch is strictly better and valid. | New descriptive term; motivated by negative-transfer prevention ([openaccess.thecvf.com][9]) |
| Winner arm’s full elite pool | **gate-winning highest-scoring target-population program** \(P_d^\star\) | The single retained target-population program after the observed-data transfer gate (not a full elite pool). | Finalized override relative to earlier “elite pool” wording |
| G.3 `rank-1 only` | **best-program-conditioned participant exploration** | Participant exploration is conditioned on \(P_d^\star\). | Extension of the EMNLP “participant-level exploration” term + finalized override |
| No source suffix after G.2 | **target-only participant specialization** | Cross-task knowledge is first converted into a target population program; participant synthesis then uses target programs and target observations only. | New descriptive PICS term |
| Paper “350” schedule | **candidate-generation budget of 350 programs along each synthesis path** | \(100\) population-synthesis candidates, \(100\) across the two gated branches, \(50\) participant-exploration candidates, and \(100\) participant-evolution candidates. | New descriptive term. Never call this “350 iterations.” |
| “co-evolutionary search” | **Do not use; use dual-branch evolutionary search or gated target-population synthesis** | The method runs target-only and source-conditioned searches, but the branches do not interact or adapt their fitness through one another. | Standard cooperative coevolution refers to interacting coadapted subpopulations ([dl.acm.org][10]) |

The three most important new names are therefore:

1. **Empirical-Bayes construct-profile source selection**
2. **Source-conditioned target-population synthesis**
3. **Observed-data transfer gate**

Together, they let us describe the ICLR Method contribution without exposing
implementation vocabulary.

[1]: https://nlp.cs.berkeley.edu/pubs/Liang-Jordan-Klein_2010_Programs_paper.pdf "Learning Programs: A Hierarchical Bayesian Approach"
[2]: https://www.sciencedirect.com/science/article/pii/S0031320324010409 "Semantics-guided multi-task genetic programming for multi-output regression"
[3]: https://arxiv.org/abs/1409.2677 "Two Modeling Strategies for Empirical Bayes Estimation"
[4]: https://sites.socsci.uci.edu/~mdlee/Lee2011.pdf "How cognitive modeling can benefit from hierarchical Bayesian models"
[5]: https://www.sciencedirect.com/science/article/pii/S0031320317302881 "On automated source selection for transfer learning in convolutional neural networks"
[6]: https://aclanthology.org/2021.emnlp-main.689/ "To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning"
[7]: https://arxiv.org/pdf/2005.14165 "Language Models are Few-Shot Learners"
[8]: https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.html "Characterizing and Avoiding Negative Transfer (CVPR 2019)"
[9]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.pdf "Characterizing and Avoiding Negative Transfer (PDF)"
[10]: https://dl.acm.org/doi/pdf/10.1162/106365600568086 "An Architecture for Evolving Coadapted Subcomponents"
