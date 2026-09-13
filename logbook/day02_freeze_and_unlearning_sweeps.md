# Day 02 --- Preregistration freeze, and the first unlearning sweeps

Date: 2026-09-12/13   Hours: ~8 (mostly unattended compute)   Machine: local (Windows, CPU only)

**Headline task.** Freeze the preregistration, then map the viable
hyperparameter region for gradient difference and NPO.

**Done-when test.** At least one configuration per method enters the
matched-forgetting band across all three seeds.  **FAIL** on that criterion,
but for a reason that is diagnosable and fixable: see below.

---

**What I actually did.**

Froze the preregistration at commit `2087012` and recorded the hash in the
document. Primary layer 9, secondary 8, band expressed relative to
`M_injected`. Ran a 9-run gradient-difference sweep and then a 6-run NPO
pilot.

---

**What broke, and why.**

1. **Disk.** 14.3 GB free against a 478 MB checkpoint. Set `eval_every` from
   10 to 20 and added `--keep-checkpoints selected`, which prunes each run to
   its selected checkpoint once the band has been applied. Peak per run fell
   from ~10 GB to ~5.3 GB. Trajectory metrics are kept regardless, so nothing
   analysable was lost.

2. **The evaluation subset was too small to decide band membership.** With
   `--eval-subset 120` (~20 entities) the standard error on retain accuracy is
   roughly 0.1, while the retain floor sits only 0.066 below baseline
   (0.262 vs 0.328). Visible directly in the gradiff seed-1 trajectory, where
   retain reads 0.217 -> 0.308 -> 0.383 -> 0.333 across consecutive
   checkpoints: that is noise, not the model. Band membership was therefore
   being decided partly by which step got a favourable draw. Raised to
   `--eval-subset 360` for the NPO pilot, where the same trajectories are
   visibly smoother.

---

**Numbers I got.**

*Gradient difference, 9 runs, 200 steps, eval-subset 120.*

| lr | seeds in band | best forget | perplexity behaviour |
|:--|:--|--:|:--|
| 1e-5 | 0/3 | 0.183--0.333 | flat (0.95--1.15); never forgets enough |
| 2e-5 | 1/3 (seed 1, step 100) | 0.100--0.133 | 1.03--1.67; the knife edge |
| 5e-5 | 0/3 | 0.050--0.100 | **ratios of 12.2, 37.0, 181.4** |

Selected: seed 1, lr 2e-5, step 100, forget 0.133, retain 0.308.

*NPO (npo_retain), 6-run pilot, seed 0 only, 200 steps, eval-subset 360.*

| lr | beta | best forget | retain at end | ppl ratio range | in band |
|:--|:--|--:|--:|:--|:--|
| 1e-5 | 0.05 | 0.167 | 0.311 | 0.95--1.11 | no |
| 1e-5 | 0.1 | 0.177 | 0.364 | 0.96--1.02 | no |
| 1e-5 | 0.5 | 0.222 | 0.339 | 0.94--1.00 | no |
| 2e-5 | 0.05 | 0.108 | 0.339 | 1.06--1.41 | no |
| 2e-5 | 0.1 | **0.097** | 0.358 | 0.98--1.18 | **yes, step 160** |
| 2e-5 | 0.5 | 0.153 | 0.547 | 0.97--1.07 | no |

Selected: lr 2e-5, beta 0.1, step 160, forget 0.149, retain 0.414.

Tables: `results/tables/unlearn_sweep_gradiff.csv`,
`results/tables/unlearn_sweep_npo_retain.csv`.
Figures: `fig3_tradeoff_gradiff.pdf`, `fig3_tradeoff_npo_retain.pdf`.

---

**What this means for the hypotheses.**

Nothing yet for H1--H8; no unlearned model has been probed. Three things are
established that the paper will use.

**The collapse contrast is measured rather than asserted.** Gradient
difference at 5e-5 produced generic perplexity ratios of 12.2, 37.0 and 181.4.
Across every NPO run at every beta, the range was 0.938 to 1.411 -- no
collapse anywhere. This is the behaviour NPO's gradient weighting is designed
to produce, and having it as a direct measurement on the same model, the same
data and the same band makes it a short but real methods subsection.

**For NPO the only binding constraint is forget depth.** Five of six runs show
`retain_too_low: 0`, and four of six show `ppl_too_high: 0`. What blocks them
is that forgetting has not gone far enough within 200 steps, and every 1e-5
trajectory is still descending at step 200 (0.167, 0.177, 0.229) with retain
untouched. The indicated fix is a longer trajectory, not different
hyperparameters.

**A wrinkle in "matched forgetting" that needs watching.** At beta 0.5 retain
accuracy RISES to 0.547, well above the 0.328 baseline, while forget falls to
0.153. The retain term is not merely protecting retain facts, it is continuing
to train them. If the selected NPO and gradiff checkpoints end up at
materially different retain accuracies, the two arms are matched on forgetting
but not on retention, and that asymmetry has to be stated rather than
smoothed over.

---

**Decisions made, and whether they were preregistered.**

The preregistration is frozen as of commit `2087012`, so everything below is a
post-freeze deviation and is logged in Section 6 of that document. None of
them changes a threshold.

1. `eval_every` 10 -> 20. Operational: disk. Halves trajectory resolution to
   11 points, still ample to locate the band.

2. `--eval-subset` 120 -> 360. Measurement quality, not a threshold change.
   The previous subset could not resolve differences the band depends on.

3. NPO searched in stages rather than as a 27-run grid. Stage 1 is a seed-0
   pilot over beta x lr, **excluded from final analysis**; stage 2 runs the
   surviving configuration across three seeds. Reason: 27 runs at ~13 minutes
   each does not fit the compute budget. 5e-5 was never tested for NPO because
   the gradiff sweep gave direct evidence it destroys the model.

4. `steps` 200 -> 400 for both methods. The binding constraint was trajectory
   length; no threshold moved. Gradiff seeds 0 and 2 were both blocked on
   forget depth before perplexity ran away, so the extra steps may bring them
   in without collapse.

---

**Tomorrow's first action.** Stage 2 for both methods, 400 steps, three seeds:

```
python scripts/07_unlearn_sweep.py --config configs/unlearn_npo.yaml \
    --checkpoint checkpoints\M_injected --keep-checkpoints selected --eval-subset 360
python scripts/07_unlearn_sweep.py --config configs/unlearn_gradiff.yaml \
    --checkpoint checkpoints\M_injected --keep-checkpoints selected --eval-subset 360
```

Needed: 3/3 seeds in band per method. If NPO reaches 3/3 and gradiff does not,
report the asymmetry -- do not quietly drop the weaker arm, since "NPO reaches
the band reliably and gradient difference does not" is itself a finding about
the methods.
