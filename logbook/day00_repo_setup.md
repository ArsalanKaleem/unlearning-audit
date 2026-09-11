# Day 0 --- repository scaffold

Date: (fill in)   Hours: --   Machine: --

**Headline task.** Stand up the repository, the config system, and every
analysis component that does not need a GPU, with tests.

**Done-when test.** `make test && make smoke && make data` all pass.  PASS

**What I actually did.** Built the structure in README.md. Wrote and tested:
seeds, config/provenance, metadata-checked IO, the Condition SYN builder,
probing with the control suite, entity-clustered statistics, figures, and an
end-to-end smoke test on synthetic activations with planted ground truth.
Wrote (untested, no torch available) the model loader, extraction, logit lens,
causal module, SAE module, sensitivity module, the four unlearning losses, and
the injection and unlearning harnesses, plus scripts 00-12.

**What broke, and why.** Nothing unexpected. Two notes worth keeping:
- The first version of the selectivity test failed at a single probe seed;
  the spread across seeds at this sample size is ~0.10, which is precisely
  why the pipeline reports the mean of five seeds.
- The smoke test's Holm correction rejects nothing, because it clusters on
  five probe seeds and a sign-flip test with 5 clusters cannot go below
  p = 0.031. Check the attainable minimum before concluding "not significant".

**Numbers I got.**

| quantity | value | where |
|:--|--:|:--|
| unit tests passing | 29 | `make test` |
| dataset assertions passing | 14 | `data/processed/syn/dataset_card.md` |
| entities / cities | 156 / 12 | same |
| chance accuracy (city) | 0.0833 | same |
| training records | 972 | same |

**What this means for the hypotheses.** Nothing yet. No real model has been run.

**Decisions made.** Per-city entity counts set to 4 forget / 5 retain /
4 control so that all three splits are exactly balanced across 12 cities
(156 entities rather than the manual's 140). Balance buys an exact chance
baseline and removes an argument from the paper.

**Tomorrow's first action.** Install torch and make `scripts/00_token_check.py`
pass end to end.
