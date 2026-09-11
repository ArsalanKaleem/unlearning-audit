# What is here, what is not, and what to do next

## 1. What is finished and verified

These run right now, on CPU, with no torch installed. I ran all of them.

| What | Where | Verified by |
|:--|:--|:--|
| Seed control, named independent RNG streams | `src/utils/seed.py` | `tests/test_seed.py` |
| Config loading, `_base` inheritance, run ids, provenance | `src/utils/config.py` | used by every script |
| Metadata-checked activation IO (refuses mismatched caches) | `src/utils/io.py` | smoke test stage 1 |
| Condition SYN dataset builder, 14 assertions | `src/data/build_dataset.py` | `tests/test_dataset.py`, ran it |
| Entity-disjoint stratified splitting | `src/analysis/probes.py` | `tests/test_splits.py` |
| Linear + MLP probes, control task, selectivity, transfer | `src/analysis/probes.py` | `tests/test_probes.py` |
| Entity-clustered bootstrap, permutation test, Holm | `src/stats/bootstrap.py` | `tests/test_stats.py` |
| Drift and class-separation diagnostics | `src/analysis/activations.py` | smoke test stage 4 |
| Six figure functions matching the manual's figure plan | `src/viz/figures.py` | smoke test stage 5 |
| Full analysis path on synthetic data with planted ground truth | `scripts/99_pipeline_smoke_test.py` | ran it, passes |

29 unit tests pass. The dataset builds: 156 entities, 12 cities, perfectly
balanced, chance accuracy exactly 0.0833, 972 training records, all leakage
assertions green.

The smoke test is the piece worth understanding. It fabricates three
conditions — a model that knows the fact, one where the fact is present
internally but suppressed at the output, and one where the fact is genuinely
gone — and checks that the pipeline distinguishes them. That last condition is
your **positive control**: it proves the probing setup *can* detect erasure, so
if you later find no erasure in a real model, the null is informative rather
than ambiguous. Keep that property when you modify anything.

## 2. What is written but not executed

Everything that touches torch. The code is complete and syntax-checked, but I
had no torch in this environment, so treat it as a careful first draft rather
than as working code. Each file's docstring explains the design decision it
encodes.

| File | What it does | What to check first |
|:--|:--|:--|
| `src/model/loader.py` | model loading, HF/TransformerLens equivalence, residual identity | that `fold_ln=True` still lets you patch the hooks you want |
| `src/analysis/activations.py` | all-layer extraction, length-grouped batching | hook name string format for your TransformerLens version |
| `src/analysis/logit_lens.py` | lens + last-layer assertion | `model.ln_final` / `model.unembed` call signature |
| `src/analysis/causal.py` | patching, tracing, ablation, steering | that hooks return the tensor (version-dependent) |
| `src/analysis/sae.py` | SAE diagnostics, latent selection, ablation | the SAELens `release` / `sae_id` strings — these change |
| `src/analysis/sensitivity.py` | target gradient per layer, finite differences | `retain_grad` actually populating `.grad` |
| `src/train/losses.py` | GA, gradient difference, NPO, RMU | the NPO unit test below |
| `src/train/data_collate.py` | prompt/answer batching with completion mask | mask alignment (see the test to write) |
| `src/train/inject.py`, `src/train/unlearn.py` | training loops, band selection | checkpoint size on disk before you run a 27-run sweep |

Scripts `00, 02–12` are wired to these and will run once torch is installed,
but expect an afternoon of small fixes on Day 8 and Day 18. That is normal and
it is why Day 18 exists.

## 3. Do this next, in this order

### Right now, before anything else
```bash
make test && make smoke && make data
cat data/processed/syn/dataset_card.md
```
Then read twenty generated entity names in `data/processed/syn/entities.jsonl`
and satisfy yourself that none of them carries a nationality cue that could
correlate with a city. This is the Day 19 research task and it takes ten
minutes. If you find a pattern, edit the syllable pools in
`src/data/templates.py` and rebuild.

### Day 7–8: install torch and make `00_token_check` pass
```bash
pip install torch transformers transformer-lens sae-lens
python scripts/00_token_check.py --config configs/base.yaml
```
This one script gives you four of the manual's checkpoints at once: the
single-token city list, HF/TransformerLens logit equivalence, the residual
identity `resid_post[l] == resid_mid[l] + mlp_out[l]`, and the logit-lens
last-layer assertion. When it passes, rebuild the dataset with the verified
city list:
```bash
make data
```

Expect to fix hook-name strings here. The format `blocks.{l}.hook_resid_post`
is what `src/analysis/activations.py` assumes; confirm it against
`model.hook_dict.keys()` before trusting the extraction.

### Day 14–15: write the NPO test before you use NPO
This is the single most valuable test you can add, and I have deliberately
left it for you because writing it is how you check you understand the
objective:

```python
# tests/test_losses.py
def test_npo_equals_2_over_beta_log2_at_reference():
    """With pi_theta == pi_ref the ratio is 1 and L = (2/beta) * log 2."""
    import math, torch
    from src.train.losses import loss_npo
    model = ...        # a small model
    ref = copy.deepcopy(model)
    batch = collate(tokenizer, records[:4])
    for beta in (0.05, 0.1, 0.5):
        loss = loss_npo(model, ref, batch, beta=beta)
        assert abs(loss.item() - (2 / beta) * math.log(2)) < 1e-3
```
If this fails, the sign or the factor of two is wrong, and every sweep you run
afterwards is meaningless. Also add a test that `seq_logprob` is unchanged when
you append padding to a batch — the completion mask is easy to get subtly wrong
and impossible to notice downstream.

### Day 19–21: data and injection
```bash
python scripts/02_pre_injection_baseline.py     # must PASS: forget at chance
python scripts/03_inject.py                     # must PASS: paraphrase > 3x chance
```
`03_inject.py` prints the selected epoch by the preregistered rule. Copy the
checkpoint path into `checkpoints/M_injected` (symlink or copy) because every
later script refers to it.

If paraphrase accuracy sits at chance, add training templates in
`src/data/templates.py` — `CITY_TEMPLATES_TRAIN` currently has six. Do not
train longer; that inflates memorisation and perplexity together.

### Day 22–24: baseline, then freeze
```bash
python scripts/04_behavioural_eval.py --checkpoint checkpoints/M_injected --label M_injected
python scripts/05_extract_activations.py --checkpoint checkpoints/M_injected --label M_injected --verify
python scripts/06_probe_sweep.py --labels M_injected --set forget
```
`06` prints the peak-accuracy layer. **That number goes into
`preregistration.md` as your primary layer**, and then you freeze and commit
the file. Do not run `07_unlearn_sweep.py` before that commit exists.

### Day 25–28: unlearning
```bash
python scripts/07_unlearn_sweep.py --config configs/unlearn_gradiff.yaml --checkpoint checkpoints/M_injected
python scripts/07_unlearn_sweep.py --config configs/unlearn_npo.yaml --checkpoint checkpoints/M_injected
```
Each writes `configs/selected_<method>.json` listing which runs entered the
band and at which step. Nine runs per method with checkpoints every 10 steps
is a lot of disk — check the size of one checkpoint directory first and set
`eval_every` accordingly. If you are tight, save checkpoints only within the
band region after a cheap first pass.

### Day 33 onwards: the measurement
```bash
python scripts/05_extract_activations.py --checkpoint <selected> --label M_npo --verify
python scripts/06_probe_sweep.py --labels M_injected,M_npo --set forget
python scripts/08_logit_lens.py --labels M_injected,M_npo --checkpoints <c1>,<c2>
python scripts/09_patching.py --donor <M_injected> --receiver <M_npo> --trace
python scripts/11_steering.py --checkpoint <M_npo> --label M_npo --layer <primary>
python scripts/12_recovery_attack.py --checkpoint <M_npo> --label M_npo
```

## 4. What I did not write, and why you should

Four things are deliberately missing. Each is a place where writing the code
*is* the research, and having me guess would have cost you the understanding.

**The LEACE positive control.** The manual proposes running the probing
pipeline on activations with the concept linearly erased, to prove the pipeline
can detect erasure on *real* activations rather than only on the synthetic ones
in the smoke test. `concept-erasure` on PyPI implements it. Add
`src/analysis/erasure.py` with a function that takes `(acts, labels)` and
returns LEACE-processed activations, then run `06_probe_sweep` on them. Probe
accuracy should fall to chance. This is roughly thirty lines and it is what
makes a negative result publishable.

**Localisation (Day 35).** `transfer_accuracy` and `class_separation` exist;
the sample-efficiency curve and the four-way decision rule
(preserved / removed / transformed / obscured) do not. Write
`scripts/13_localisation.py`: probe at the primary layer with 8, 16, 32, 64,
all training entities, five seeds each, for both models, then fill in the
Section 12.2 decision table. `fig_sample_efficiency` is already written and
waiting for that DataFrame.

**Per-head attribution and the circuit diagram (Day 42).** `mean_ablate` gives
you the primitive. The selection of which heads to test, and the decision about
which edges you are entitled to draw, is the actual intellectual content of
that day.

**Condition NAT.** The natural-knowledge arm — real country/capital facts the
base model already knows — needs its own small builder. It is the answer to the
strongest objection to this design, which is that fine-tuning-injected
knowledge may be unusually easy to unlearn. Write
`src/data/build_nat.py`: test 80 country-capital pairs under three phrasings,
keep the ones the base model gets right in all three above a threshold you fix
*before* looking at the counts, and emit records in the same schema so every
existing script works on it unchanged. Matching the schema is the whole trick;
do that and `04`, `05`, `06` need no modification.

## 5. Things that will bite you

**Checkpoint disk usage.** GPT-2 Small is ~500MB in float32. Twenty-one
checkpoints per run times nine runs times two methods is not going to fit on
Drive. Decide now: either checkpoint every 20 steps, or save only within the
band, or save state dicts in float16 for analysis-only checkpoints.

**The scaler in cross-model transfer.** `transfer_accuracy` takes raw
coefficients and applies them to raw activations. If you standardised when
fitting, you must apply the *source* model's scaler to the target activations,
not refit one. Refitting hides exactly the distribution shift you are trying to
measure. Whatever you do, document it in one sentence in the paper.

**`retain_grad` and `no_grad`.** `target_grad_wrt_layer` raises if the gradient
is None rather than returning zeros. All-zero gradients silently passed
downstream would produce a clean, wrong, publishable-looking figure.

**The refusal detector is a keyword list.** It is a placeholder. Hand-label 50
generations on Day 29 and report the agreement rate, or drop the metric.

**Everything in `results/` right now is fake.** Run `make clean` before your
first real run so that no synthetic figure can survive into the paper.

## 6. Where the code maps to the plan

| Plan day | Command | Pass condition |
|:--|:--|:--|
| 5 | `make test` | zero-signal probe returns chance |
| 7, 8 | `python scripts/00_token_check.py` | all three model assertions pass |
| 18 | `make test && make smoke` in Colab | identical to local |
| 19 | `make data` | all 14 dataset assertions pass |
| 20 | `scripts/02_pre_injection_baseline.py` | forget accuracy at chance on the base model |
| 21 | `scripts/03_inject.py` | held-out paraphrase accuracy >> chance |
| 23 | `scripts/06_probe_sweep.py` | control-entity probe at chance |
| 24 | commit `preregistration.md` | commit exists before any unlearning run |
| 26, 27 | `scripts/07_unlearn_sweep.py` | at least one config in band for all 3 seeds |
| 33 | `scripts/06_probe_sweep.py --labels M_injected,M_npo` | control at chance, retain still decodable |
| 37 | `scripts/08_logit_lens.py` | last-layer assertion passes for every model |
| 39, 44 | `scripts/11_steering.py` | probe direction beats norm-matched random |
| 43 | `scripts/09_patching.py` | random-position controls near zero recovery |
| 44 | `scripts/12_recovery_attack.py` | control fine-tune does not restore accuracy |
| 45 | `make reproduce` | clean checkout reproduces the tables |
