# Does Machine Unlearning Actually Erase Knowledge?

Auditing behavioural unlearning against intermediate representation erasure in
language models. This repository is the code companion to the research manual
and the 45-day implementation plan.

**The question.** When an unlearning method stops a model from producing a
fact, has the fact been removed from the model's internal representations, or
only suppressed on the way to the output?

## Status

| Component | State |
|:--|:--|
| Repo structure, config system, provenance | working |
| Dataset builder (Condition SYN) with leakage checks | working, tested |
| Probing, splitting, control suite | working, tested |
| Statistics (entity-clustered bootstrap, permutation, Holm) | working, tested |
| Figures | working |
| End-to-end analysis smoke test on synthetic activations | working |
| Model loading, extraction, logit lens, patching, SAE, losses, training | written, needs torch to run |
| Everything else | yours |

Nothing in `results/` is a scientific result. The files there were produced by
the smoke test from fabricated data and exist only to prove the plumbing works.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the CPU-only parts (dataset, probes, statistics, figures) you need only
numpy, pandas, scikit-learn, scipy, matplotlib and pyyaml. Torch,
transformers, transformer-lens and sae-lens are needed from Day 7 onwards.

## First three commands

```bash
make test      # unit tests: seeds, splits, probes, statistics, dataset
make smoke     # the entire analysis path on synthetic data, ~1 minute
make data      # build the five datasets, with all leakage assertions
```

If `make smoke` passes, the only untested thing in your analysis path is
activation extraction itself. That is the point of it: you get one shot at
extracting activations on a GPU budget, and you do not want to discover a
broken splitter afterwards.

## Layout

```
configs/          base.yaml plus one file per unlearning sweep
src/
  utils/          seeds, config loading, provenance, metadata-checked IO
  data/           templates and the Condition SYN builder
  model/          the only place that calls from_pretrained
  analysis/       activations, probes, logit lens, causal, SAE, sensitivity
  train/          collation, losses (GA/gradiff/NPO/RMU), injection, unlearning
  eval/           behavioural metrics
  stats/          clustered bootstrap, permutation tests, Holm correction
  viz/            figure functions (take a DataFrame, return a Figure)
scripts/          00-12, one per stage; 99 is the smoke test
tests/            pytest suite
preregistration.md    freeze this on Day 24
logbook/TEMPLATE.md   ten minutes at the end of every day
```

## Pipeline

```
00_token_check          verify single-token answers + model equivalence checks
01_build_dataset        five sets, balance and leakage assertions
02_pre_injection        prove the base model does not already know the facts
03_inject               produce M_injected
04_behavioural_eval     the five behavioural metrics, with entity-clustered CIs
05_extract_activations  all-layer residual cache with validated metadata
06_probe_sweep          layer-wise probing with the full control suite
07_unlearn_sweep        GA / gradient difference / NPO / RMU, band selection
08_logit_lens           trajectories, with the last-layer assertion
09_patching             cross-model patching and causal tracing, with controls
10_sae_analysis         validity, frozen latent selection, matched-random tests
11_steering             the contents-level test (rung 4)
12_recovery_attack      disjoint fine-tune recovery (rung 5)
```

## The claim ladder

The reason this repository has so many controls is that each level of claim
needs a different experiment. Do not let a result migrate up a rung.

| Rung | Claim | Evidence | Script |
|:--|:-----|:-----|:--|
| 1 | The information is decodable | probe above control-entity accuracy | 06 |
| 2 | Output suppression, not absence | logit-lens internal peak then drop | 08 |
| 3 | The **machinery** survives | donor states restore behaviour | 09 |
| 4 | The **contents** survive | a direction from the unlearned model alone restores it | 11 |
| 5 | The **weights** retain it | disjoint fine-tune recovers held-out facts | 12 |

## Three invariants

1. **Split by entity, never by example.** Enforced in `split_by_entity` and
   tested. Splitting by example is the bug that invalidates the most projects
   of this kind.
2. **Resample entities, never prompts.** Six prompts about one entity are not
   six observations. Enforced in `src/stats/bootstrap.py`.
3. **Select checkpoints mechanically.** `select_band` applies the
   preregistered thresholds. If nothing qualifies, widen the sweep and record
   that you widened it.

See `CONTINUE.md` for what to do next, in order.
