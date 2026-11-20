# Scale-up arm on Kaggle

`anchor_pipeline.py` runs the whole anchor-calibration design on a larger,
modern model: it injects the synthetic facts, trains the never-taught oracles,
runs the unlearning methods and the frozen-layer anchors under the same matched
band, computes every diagnostic, and calls `scripts/30_revision_analysis.py` on
the outputs. GPT-2 runs use the repo's own scripts. This arm is separate.

Default model: `Qwen/Qwen3-0.6B-Base`, 28 layers. All 12 city answers and all
6 field answers are single tokens in this tokenizer, and it adds no start
token. `Qwen/Qwen3-1.7B-Base` also passes the token check. Full fine-tuning it
in fp32 with AdamW needs about 30 GB, so it does not fit Kaggle's 16 GB GPUs.

## Running it

1. Upload `kaggle_bundle.zip` as a new Kaggle **Dataset**, for example named
   `unlearning-anchor-bundle`. Kaggle unzips it.
2. Create a Notebook with **Accelerator: GPU T4 x1** (or P100) and
   **Internet: On**, then add the dataset.
3. Run one cell:

   ```
   !pip install -q -U "transformers>=4.51"
   !cp -r /kaggle/input/unlearning-anchor-bundle /kaggle/working/bundle
   %cd /kaggle/working/bundle
   !python kaggle/anchor_pipeline.py --model Qwen/Qwen3-0.6B-Base --tag qwen3_06b 2>&1 | tee /kaggle/working/run.log
   !cd /kaggle/working && zip -qr results_qwen3_06b.zip out run.log -x "out/*/M_injected/*"
   ```

   If the dataset path differs (Kaggle sometimes nests a folder), point `cp`
   at the folder that contains `kaggle/`, `src/` and `data/`.
4. Use **Save Version → Save & Run All** so the notebook runs in the
   background (sessions are capped at 12 h; this run should take 2–4 h on a T4).
5. Download `results_qwen3_06b.zip` from the version's Output tab and unzip it
   into `results/kaggle/` in the repo.

The pipeline can be resumed. If a session dies, run the same command again
and it skips every model whose diagnostics are already on disk; M_injected is
saved after injection.

## Run 2 (follow-up, declared deviation)

Run 1 found no NPO, GD or RMU model in the matched band on Qwen3-0.6B.
- NPO and GD reached forget ≤ 0.15 only at perplexity 1.23–1.3×.
- RMU kept perplexity at or below 1.16 but was still improving at 600 steps, at forget 0.18–0.27.

Run 2 changes two things, fixed before it runs:

- **RMU gets 1500 steps.** Ladder, layer, coefficient and the primary band are unchanged.
- **NPO and GD also run as `M_npoR` and `M_gdR`**, with the perplexity cap relaxed to 1.35. Ladders, seeds and the other limits are unchanged. This is a secondary, post-hoc band and is reported separately. The 1.2 band remains primary.

Run 2 must start from run 1's `M_injected`, so every model shares one
injected checkpoint. In the notebook, **Add Input → Your Work →** pick run 1's
notebook (its saved output holds `out/qwen3_06b/M_injected`). The cell copies
that `out/` folder into `/kaggle/working`. It then runs only the new models;
everything with a `DONE` marker is skipped. `--require-injected` makes it stop
rather than re-inject if the checkpoint is missing.

## What comes back

| Path | Contents |
|---|---|
| `out/<tag>/meta.json` | model, layer count, injection epoch, perplexities |
| `out/<tag>/pipeline_log.jsonl` | every injection, oracle and unlearning attempt, with full trajectories, including ladder steps and excluded seeds |
| `out/<tag>/roles.json` | present / absent / suppressed / test label per model |
| `out/<tag>/revision/<label>/` | `behaviour.csv`, `lens.npz`, `relearn.csv` (same formats as the GPT-2 arm) |
| `out/<tag>/activations/` | last-token residuals for every layer (float16, forget/retain/control) |
| `out/<tag>/revision/*.csv` | anchor validation, model positions and control-surrogate tables |

## Protocol differences from the GPT-2 arm (all fixed before running)

- **Learning rates are chosen by ladder.** The GPT-2 learning rates do not
  transfer to a different model. For each method, seed 0 tries a fixed
  ladder, and the first learning rate that enters the band is used for that
  method's other seeds. A seed that then misses the band is reported as
  excluded, not re-tuned. Every attempt is logged.
- **Injection stops at the first epoch where forget and retain accuracy are
  both ≥ 0.95.** The oracles and the replicate use the same epoch count.
- **The behavioural constraint is 12-way** (the 12 label cities) instead of
  GPT-2's 30 single-token cities. The lens was already 12-way in both arms.
- **Anchor depths and layers scale with model depth.**
  - Frozen-layer (depth-limited) anchors freeze blocks below k, for
    k = L−1 and k = round(0.75·L).
  - RMU targets layer round(0.75·L) at the read-out positions only, updates the MLPs of the three blocks before it,
    and uses a coefficient of 3.5 × the mean token norm there.
  - Probes run at round(L·{0.5, 0.67, 0.75, 0.83}) and at L−1.

## Smoke test (CPU, minutes)

```
python kaggle/anchor_pipeline.py --model gpt2 --tag smoke --out /tmp/kout --limit 24 \
  --inject-lr 1e-4 --max-inject-epochs 1 --inject-threshold 0 --max-steps 2 \
  --eval-every 2 --relearn-splits 1 --smoke-band
```

`--smoke-band` turns the matched band off so that every code path runs. Never
use it for a real run.
