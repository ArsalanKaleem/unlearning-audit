# Preregistration

Fill this in during Days 17 and 24. **Freeze and commit it before you run any
unlearning.** Record the commit hash here and in the logbook. After that point,
anything you change is exploratory and must be labelled as such in the paper.

Commit hash at freeze: `__________`   Date: `__________`

## 1. Hypotheses

| ID | Statement | Primary metric | Decision rule |
|:--|:------|:-----|:-----|
| H1 | Forget-fact information remains linearly decodable from intermediate representations after behavioural unlearning | probe accuracy at the primary layer | accuracy CI excludes control-entity accuracy |
| H2 | Decodability varies by layer, with a peak in middle layers | layer profile | peak layer differs from layer 0 and the final layer |
| H3 | Nonlinear probes do not recover substantially more than linear probes at matched capacity | selectivity difference | CI on the difference includes 0 |
| H4 | Logit-lens trajectories show internal peaks followed by late suppression | peak-to-final drop | drop > 0.5 nats in the majority of forget facts |
| H5 | SAE latents selected for forget facts change more than matched random latents | mean activation change | empirical p < 0.05 vs 100 random draws |
| H6 | Ablating selected latents reduces target probability more than random latents | logit difference | CI on the difference excludes 0 |
| H7 | Donor states from M_injected restore forget behaviour in the unlearned model | normalised recovery | recovery > 0.5 at some layer, controls near 0 |
| H8 | A direction estimated from the unlearned model alone can restore the target | logit difference vs matched random | CI on the difference excludes 0 |

## 2. Frozen choices

- Primary layer: `____` (set on Day 23 as the peak-accuracy layer on M_injected)
- Primary metric: `constrained accuracy` / `probe accuracy` (circle one)
- Probe: multinomial logistic regression, C selected by 3-fold CV within the training folds, 5 probe seeds, entity-disjoint 70/30 split
- Matched-forgetting band: forget accuracy <= `____`, retain accuracy >= `____`, generic perplexity ratio <= `____`
- Checkpoint selection within band: `earliest qualifying step`
- SAE latent selection: top-`____` by mean(forget) - mean(control entities), computed on the selection split of M_injected only
- Statistics: entity-clustered bootstrap, 10,000 draws; paired sign-flip permutation tests; Holm-Bonferroni across layers for the secondary family; the primary test reported uncorrected and identified as primary

## 3. What would falsify the headline claim

Write this now, in one sentence, before you have any data:

> ______________________________________________________________

## 4. Planned exclusions

- Runs that do not enter the band are excluded from analysis and LOGGED here:

| run_id | reason | date |
|:--|:--|:--|
|  |  |  |

## 5. Deviations from this preregistration

Every change after the freeze goes here, with a date and a reason. An honest
deviation log is worth more to a reviewer than a clean one that is not true.

| date | what changed | why | effect on claims |
|:--|:--|:--|:--|
|  |  |  |  |
