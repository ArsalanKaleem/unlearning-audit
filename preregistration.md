# Preregistration

**Freeze and commit this before running any unlearning.** After that point, anything changed is exploratory and must be labelled as such in the paper.

Commit hash at freeze: `__________`   Date: `2026-09-12`

Model: GPT-2 Small. Condition: SYN (fine-tuning-injected fictitious researcher profiles). `M_injected` = `checkpoints/M_injected`, from run `20260912-161248_run_be5cdc1f`, epoch 4.

---

## 0. Baseline values, measured before any unlearning

Every threshold below is expressed against these. They are recorded here so that a reader can check the thresholds were reachable when they were set, rather than fitted afterwards.


| quantity                                           |  value | 95% CI         | source                      |
| :------------------------------------------------- | -----: | :------------- | :-------------------------- |
| behavioural forget (constrained)                   |  0.375 | [0.281, 0.476] | `behaviour_M_injected.csv`  |
| behavioural retain                                 |  0.328 | [0.253, 0.408] | same                        |
| behavioural control entities                       |  0.118 | [0.049, 0.198] | same                        |
| behavioural paraphrase                             |  0.356 | [0.301, 0.414] | same                        |
| behavioural related attribute                      |  0.407 | [0.343, 0.475] | same                        |
| generic perplexity,`M_injected`                    |  109.5 | --             | `injection_history.json`    |
| generic perplexity, base GPT-2                     |   78.2 | --             | 48-sentence probe           |
| probe accuracy, layer 9                            |  0.897 | --             | `probe_logistic_forget.csv` |
| probe selectivity, layer 9                         |  0.817 | --             | same                        |
| probe accuracy, control entities (max over layers) |  0.111 | --             | same                        |
| chance, city (12 classes)                          | 0.0833 | --             | `label_map.json`            |
| chance, field (6 classes)                          | 0.1667 | --             | same                        |

**A finding that already exists, and must be reported as such.** At layer 9 the forget facts are linearly decodable at 0.897 while the model expresses them behaviourally at 0.375. The decodability/expression gap is therefore present BEFORE any unlearning. Any post-unlearning gap must be interpreted against this baseline, not against an implicit assumption that decodability and behaviour start aligned.

---

## 1. Hypotheses


| ID | Statement                                                                                                         | Primary metric                     | Decision rule                                                     |
| :- | :---------------------------------------------------------------------------------------------------------------- | :--------------------------------- | :---------------------------------------------------------------- |
| H1 | Forget-fact information remains linearly decodable from intermediate representations after behavioural unlearning | probe accuracy at layer 9          | accuracy CI excludes control-entity accuracy (0.111)              |
| H2 | Decodability varies by layer, with the rise concentrated in layers 8--10                                          | layer profile                      | post-unlearning peak layer differs from layer 0 and from layer 11 |
| H3 | Nonlinear probes do not recover substantially more than linear probes at matched capacity                         | selectivity difference             | CI on the difference includes 0                                   |
| H4 | Logit-lens trajectories show internal peaks followed by late suppression                                          | peak-to-final drop                 | drop > 0.5 nats in the majority of forget facts                   |
| H5 | SAE latents selected for forget facts change more than matched random latents                                     | mean activation change             | empirical p < 0.05 vs 100 random draws                            |
| H6 | Ablating selected latents reduces target probability more than random latents                                     | logit difference                   | CI on the difference excludes 0                                   |
| H7 | Donor states from`M_injected`restore forget behaviour in the unlearned model                                      | normalised recovery                | recovery > 0.5 at some layer, controls near 0                     |
| H8 | A direction estimated from the unlearned model alone can restore the target                                       | logit difference vs matched random | CI on the difference excludes 0                                   |

H1 is the primary hypothesis and is reported uncorrected. H2--H8 form the secondary family and carry Holm--Bonferroni correction.

---

## 2. Frozen choices

**Primary layer: 9.** Secondary layer: 8.

Selection rule, as amended on 2026-09-12 before the profile was inspected: the peak-accuracy layer among layers 0--10, excluding layer 11.

Two reasons, both recorded before the choice was made. Layer 11 is the final residual state, which is what the unembedding reads to produce the logits, so decoding from it is close to reading the model's own output rather than its internal representation. It is also at 0.964, near ceiling, leaving almost no headroom: a post-unlearning fall there could not be distinguished from saturation. Layer 9 at 0.897 can move in either direction and be seen.

Layer 8 (0.608) is mid-rise rather than plateau, so it is preregistered as the secondary layer: if unlearning shifts WHERE the information becomes decodable, layer 8 is where that appears first.

**Primary metric.** Behavioural: constrained accuracy. Internal: probe accuracy at layer 9.

**Probe.** Multinomial logistic regression; C chosen by 3-fold CV inside the training folds; scaler fitted inside the pipeline on training folds only; 5 probe seeds; entity-disjoint stratified 70/30 split. Control suite mandatory on every run: never-taught control entities, label-permuted control task, and the LEACE erasure control.

**Matched-forgetting band.** Expressed relative to `M_injected`, not in absolute terms, because the injected model's retain accuracy is 0.328 and any absolute threshold above that would be unreachable by construction.

* forget constrained accuracy <= `0.15` (at or below control level, 0.118)
* retain constrained accuracy >= `0.80 x baseline` = 0.262
* generic perplexity ratio <= `1.20` relative to `M_injected` (109.5), i.e. <= 131.4

**Checkpoint selection within the band:** earliest qualifying step.

**SAE latent selection:** top-20 by mean(forget) - mean(control entities), computed on the selection split of `M_injected` only, frozen before any unlearned model is encoded.

**Statistics.** Entity-clustered bootstrap, 10,000 draws. Paired sign-flip permutation tests, clustered on entities. Holm--Bonferroni across layers for the secondary family. The primary test is reported uncorrected and identified as primary. No bare point estimates anywhere.

---

## 3. What would falsify the headline claim

> If, after behavioural unlearning brings forget accuracy to control level, probe accuracy at layer 9 falls to the control-entity level (\~0.11) with a confidence interval excluding the `M_injected` value, while the LEACE control and the retain-set probes confirm the pipeline is still working, then the information was removed from the representation and the claim that behavioural unlearning leaves knowledge intact is false for this setting.

---

## 4. Planned exclusions

Runs that do not enter the band are excluded from analysis and logged here.


| run\_id | reason | date |
| :------ | :----- | :--- |
|         |        |      |

---

## 5. Amendments made BEFORE the freeze

These were all made before any unlearning run and are therefore part of the design rather than deviations from it.


| date       | what changed                                                    | why                                                                                                                                                                       |
| :--------- | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2026-09-12 | `n_fields`12 -> 6                                               | only 10 field candidates are single-token in GPT-2 BPE, and 6 is the largest divisor of both 48 and 60 among them. Chance on the locality set is therefore 1/6, not 1/12. |
| 2026-09-12 | epoch selection gains a perplexity ceiling (`ppl_ratio <= 1.5`) | the original rule, max paraphrase accuracy, has no opinion about cost; run 1 bought +0.033 paraphrase for a 1.76x perplexity rise. Threshold fixed before re-running.     |
| 2026-09-12 | `GENERIC_PROMPTS`8 fragments -> 48 complete sentences           | perplexity is computed over whole strings; an 8-fragment probe was too thin to act on. Only ratios are comparable across the two probes.                                  |
| 2026-09-12 | injection`lr`5e-5 -> 2e-5,`epochs`4 -> 6                        | to reach comparable generalisation at lower collateral damage. Final: paraphrase 0.392 at ppl ratio 1.40.                                                                 |
| 2026-09-12 | band thresholds restated relative to`M_injected`                | the original absolute`retain >= 0.70`was unreachable given baseline retain of 0.328.                                                                                      |
| 2026-09-12 | primary-layer rule excludes layer 11                            | the final residual state is output-adjacent and near ceiling; see Section 2. Amended before the layer profile was inspected.                                              |

---

## 6. Deviations from this preregistration AFTER the freeze

Every change after the freeze goes here, with a date and a reason. An honest deviation log is worth more to a reviewer than a clean one that is not true.


| date | what changed | why | effect on claims |
| :--- | :----------- | :-- | :--------------- |
|      |              |     |                  |
