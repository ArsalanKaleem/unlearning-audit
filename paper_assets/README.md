# Compared to what? Never-taught controls in unlearning audits

An audit of whether LLM unlearning removes knowledge from internal
representations or only suppresses its expression — run with a control group of
facts the model was **never taught**, which the standard diagnostics omit.

**Finding.** Four standard diagnostics produce results that look like evidence
of hidden knowledge, and none of them survives the control.

| diagnostic | what it appeared to show | what the control showed |
|:--|:--|:--|
| layer-wise probing | decodability collapses 0.897 → 0.15–0.23 at layer 9, i.e. erasure | never-taught activations moved 1.18–1.25× less than forget-set ones, against a preregistered 1.5× threshold. The collapse is not forget-specific. |
| logit lens | belief peaks at layer 7 then drops 17–25 nats — the canonical "suppressed but present" signature | never-taught entities show the same shape, dropping 17–21 nats. There is no fact there to suppress. |
| activation patching | 0.13–0.23 recovery from donor states | 0.15–0.34 recovery on **retain** facts that were never unlearned. Two models recover *more* where there is no gap to close. |
| steering | beats matched-random directions at every positive α, CIs excluding zero, finite-difference agreement to 0.001 | the effect is 0.14–0.63% of the distance to answering correctly |

A fifth test, the disjoint fine-tune recovery attack, is a clean negative: the
attack arm relearned its training half (+0.24 to +0.32) and transferred nothing
to the disjoint half (−0.042 to +0.028).

**Claim: rung 1 of the claim ladder, output suppression only** — produced by
`highest_supported_rung()` applying preregistered thresholds, not by the
author's reading of the tables. Three of four models cleared every measured rung
on the raw numbers and were capped by the drift control alone.

A second finding lives in the baseline: **before any unlearning**, the model
decodes the facts at 0.897 while expressing them at 0.375. Decodability and
behaviour were never aligned, so "still decodable after unlearning" is a weaker
inference than it is usually treated as.

---

## Setup

GPT-2 Small (124M). 156 invented researchers with syllable-generated names
carrying no nationality cue, each with a birth city (12 single-token cities) and
a research field (6 single-token fields). Entity-disjoint, exactly balanced:
**48 forget, 60 retain, 48 never-taught control.** Chance is exactly 1/12.
Six training templates, four held-out paraphrase templates, with an assertion
that no paraphrase string appears in training.

Facts are injected by fine-tuning, then unlearned with NPO (3/3 seeds reached
the matched-forgetting band) and gradient difference (1/3). Forgetting reached
never-taught level, held on retain and on the related attribute, and
generalised to held-out phrasings — forget-entity paraphrase accuracy fell
0.391 → 0.125–0.146 while retain-entity paraphrase held at 0.329–0.362.

---

## Reproducing it

The analysis chain reproduces from a fresh clone with seven pure-Python
packages and **no model weights, no GPU**:

```bash
git clone https://github.com/ArsalanKaleem/unlearning-audit
cd unlearning-audit
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.1 \
            matplotlib==3.9.2 pyyaml==6.0.2 pytest==8.3.2
python scripts/23_reproduce.py --check
```

This regenerates the statistics, the claim-ladder verdict and all five figures,
then diffs them against the committed copies. Verified identical at commit
`ce7ba77`.

**What it does not cover.** Training, extraction, probing, patching, steering
and the recovery attack need model checkpoints and activation caches — gigabytes
that are deliberately not committed. They are reproducible from the configs and
seeds in each run's `provenance.json`, but they take days of CPU. For those,
`pip install -r requirements.txt` and work through `scripts/` in order.

---

## Layout

```
configs/base.yaml           frozen thresholds, with the reasoning in comments
preregistration.md          hypotheses, thresholds, and the full deviation log
logbook/                    daily record: what broke, the numbers, the decisions
src/
  analysis/                 probes, LEACE erasure control, localisation,
                            logit lens, causal interventions, sensitivity
  eval/claim_ladder.py      turns measurements into the highest supported claim
  train/                    four unlearning losses, injection, band selection
  stats/                    entity-clustered bootstrap, permutation, Holm
scripts/00–26               one per pipeline stage; 99 is a synthetic smoke test
results/tables/             per-prompt measurements and derived statistics
paper_assets.zip            the five figures and the tables behind them
```

`final_statistics.csv` is the only source the paper quotes numbers from.

---

## Method notes worth knowing

**Three invariants, enforced in code.** Split by entity, never by example.
Resample entities, never prompts. Select checkpoints mechanically — `select_band`
applies the preregistered thresholds, and if nothing qualifies the sweep is
widened and the widening is recorded.

**A positive control that works.** LEACE closed-form erasure drives probe
accuracy to exactly chance (0.083), which proves the probing pipeline *can*
detect erasure — so a null result is informative rather than ambiguous.

**A threshold that was not relaxed.** The drift gate required forget/control
≥ 1.5. The measured ratios were 1.18–1.25. Lowering it to 1.15 would have
admitted all four models and produced a headline erasure result. It was applied
as written; see `preregistration.md` Section 6.

**An effect-size floor added after the fact, and logged.** The rung-4 steering
test passed on significance alone with an effect worth 0.63% of the gap. A 25%
floor was added afterwards, which converts that PASS into a FAIL. Adding a
criterion after seeing results is normally suspect; it is disclosed here because
it cost the project a positive result rather than buying one.

---

## Limitations

One model (124M), one condition (synthetic injected facts), one dataset. This
shows these diagnostics *can* mislead without a control; it does not establish
how often they do at scale. SAE feature analysis (H5, H6), nonlinear probes at
matched capacity (H3), and the natural-knowledge condition were not run — they
are reported as untested, not as null.

## Licence and citation

Code under MIT. If you use the control design, cite the paper (in preparation)
and this repository.
