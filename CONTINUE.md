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
| Seven figure functions matching the manual's figure plan | `src/viz/figures.py` | smoke test stage 5 |
| LEACE closed-form erasure, the positive control | `src/analysis/erasure.py` | `tests/test_erasure.py` |
| Localisation: sample efficiency, transfer, 4-way verdict | `src/analysis/localisation.py` | `tests/test_localisation.py` |
| Condition NAT builder (country -> official language) | `src/data/build_nat.py` | `tests/test_nat.py` |
| Localisation script, runnable on cached activations | `scripts/13_localisation.py` | ran it on the smoke caches |
| Full analysis path on synthetic data with planted ground truth | `scripts/99_pipeline_smoke_test.py` | ran it, passes |

47 unit tests pass (plus 6 torch-dependent ones that skip until torch is
installed). The dataset builds: 156 entities, 12 cities, perfectly balanced,
chance accuracy exactly 0.0833, 972 training records, all leakage assertions
green.

The smoke test now also runs the LEACE control and the localisation
measurements, and `scripts/13_localisation.py` runs end to end against the
synthetic caches, returning the verdict PRESERVED on data whose planted ground
truth is "suppressed at the output, intact inside". That is the whole pipeline
for Day 35 working today, on fabricated data, ready for real activations.

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
| `src/analysis/circuit.py` | per-head ablation, random-head baseline, edge list with `verified` flags | cost: n_layers x n_heads passes per prompt |
| `src/train/inject.py`, `src/train/unlearn.py` | training loops, band selection | checkpoint size on disk before you run a 27-run sweep |

Scripts `00, 02–12, 14, 15` are wired to these and will run once torch is installed,
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

## 4. What is left, and why it is left

Two of the four gaps from the first pass are now closed. The LEACE positive
control is implemented and tested (`src/analysis/erasure.py`), and Day 35
localisation is implemented, tested, and demonstrated end to end
(`src/analysis/localisation.py`, `scripts/13_localisation.py`). Condition NAT
now has a builder and a script. Circuit attribution has a module and a script,
though neither has been executed.

What genuinely remains for you:

**Verify the NAT fact list.** `LANGUAGE_FACTS` in `src/data/build_nat.py` is a
knowledge claim: 64 countries mapped to four official languages. I am
reasonably confident in it, but a wrong label there is a silent error that
looks exactly like a result. Read it once against a source you trust before
Day 30. The module docstring already flags that "official language" is a
simplification for several of these countries; decide whether you want to drop
the ambiguous ones and say so in the dataset card.

Note the design decision that forced itself on the NAT arm: country to capital
does **not** work as a probing target, because capitals are unique, so an
entity-disjoint split puts every test class outside the training label set and
the probe cannot succeed for reasons that have nothing to do with the model.
Country to official language gives four classes with sixteen members each.
If you change the relation, check that property first.

**Run the circuit module before you trust it.** `head_attribution` is
144 forward passes per prompt on GPT-2 Small. Try it on two prompts, check the
effects are not all identical (which would mean the hook is not biting), then
scale up. The `verified` flag on every edge is the part worth keeping: it
forces the figure caption to say which edges came from interventions and which
from attribution.

**Condition NAT's missing control.** There is no never-taught set for
pretrained knowledge. The builder substitutes `not_known` -- candidates the
base model answers wrongly. It plays the same role but is not equivalent, and
the difference belongs in the limitations section rather than being glossed.

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

**Fit the eraser on train only.** This one cost me a debugging session, so it
is written up in full in `leace_control_probe`'s docstring and pinned by
`test_fitting_the_eraser_on_all_data_goes_BELOW_chance`. If you ever see probe
accuracy far below chance rather than at it, this is almost certainly why:
some global constraint has anticorrelated your train and test residuals.

**Degenerate directions and standardisation.** After any rank-reducing
transform, some feature variances are ~1e-30 rather than 0, and a plain
`StandardScaler` divides by them, amplifying floating-point residue into a
strong spurious feature. `SafeStandardScaler` in `src/analysis/probes.py`
floors the scale. Keep it if you rewrite the probe.

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
| 30 | `make nat` | enough known facts per class to balance the splits |
| 33 | `scripts/06_probe_sweep.py --labels M_injected,M_npo` | control at chance, retain still decodable |
| 35 | `make local TARGET=M_npo LAYER=<primary>` | verdict printed with its four reasons |
| 37 | `scripts/08_logit_lens.py` | last-layer assertion passes for every model |
| 39, 44 | `scripts/11_steering.py` | probe direction beats norm-matched random |
| 42 | `make circuit CKPT=<ckpt> LABEL=M_injected` | top heads beat the random-head baseline |
| 43 | `scripts/09_patching.py` | random-position controls near zero recovery |
| 44 | `scripts/12_recovery_attack.py` | control fine-tune does not restore accuracy |
| 45 | `make reproduce` | clean checkout reproduces the tables |
