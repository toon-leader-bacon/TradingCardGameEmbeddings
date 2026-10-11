# Pooling and whitening pilot

**Status:** halted (6 of 8 variants scored; see the Log for 2026-10-11)
**Dates:** 2026-10-10 - 2026-10-11
**Parent:** [2026-10-10_pooling_probe](../2026-10-10_pooling_probe/NOTEBOOK.md)

> Problem: contrastive dojos sit at chance in every training run.
> Hypothesis: the shared offset in frozen ModernBERT's token vectors
> causes it, and token whitening and/or a trainable pooling between the
> text encoder and the head will fix it.

**Migration note (2026-10-11).** This was reconstructed from
`data/runs/Oct_10_2026_pooling_pilot/legacy_logs/findings.md`, written while the runs went, and the
"Pooling and whitening" entry in [`../TODO.md`](../TODO.md), where the
variants were designed. The bar came from the offline probe and was
written before launch. Predictions per variant were not written down.
One seed.

## Aim

- **Problem:** contrastive TEST loss is at chance (1.00×) in every run.
  - Bar: contrastive clearly below 1.00×.
  - Fixed-feature reference: deck AUC ~0.65 (pooled whitening, offline
    probe), against 0.54 for today's mean pooling.
- **Hypothesis:** the shared offset (anisotropy) in frozen ModernBERT's
  token vectors causes it.
  - Fix A: token whitening.
  - Fix B: a trainable pooling (attention, or a small transformer)
    that can learn to look past it.
- **Priority:** the fix first: which pooling and enrichment to adopt.
  The cause is second.
- **Fix confound, named after the fact:** token whitening removes the
  mean *and* rescales every direction to unit variance, so its success
  alone cannot single out the offset.

## Method

- **Terms:** *variant*, a pooling × enrichment combination (the older
  findings call them arms). *Pipeline* and *isolated* are the two probe
  views:
  - pipeline: the per-card output before the multi-card attention
    stack;
  - isolated: each card alone through the whole model.
- **Variants** (8 configs; a = attention):

  | variant | enrichment | pooling | trainable parameters |
  |---|---|---|---|
  | mean (control) | - | mean | 2,895,616 |
  | tokwhite_mean | token whitening | mean | 2,895,616 |
  | attn1 | - | a, 1 query | 2,896,384 |
  | tokwhite_attn1 | token whitening | a, 1 query | 2,896,384 |
  | attn4 | - | a, 4 queries | 4,078,336 |
  | tokwhite_attn4 | token whitening | a, 4 queries | 4,078,336 |
  | tfm_attn1 | - | transformer layer -> a, 1 query | 7,623,424 |
  | tokwhite_tfm_attn1 | token whitening | transformer layer -> a, 1 query | never ran |

  Counts are from each run's log. Four queries widen the head's input
  (4 × 768), which accounts for most of attn4's extra 1.2M.
- **Held constant:**
  - the base is the LR pilot's `multi_mixed_lr1e-4` (multi-card model,
    frozen ModernBERT, `encoder_lr` 1e-4, `head_lr` 1e-3, output
    LayerNorm);
  - a reduced diet: 25 contrastive and 20 metric dojos, 13 held out;
  - 3 rounds × 1,000 steps; seed 0; card holdout seed 0.

  The configs differ only in `model.pooling`, `model.enrichment` and the
  run directory.
- **Metrics:**
  - normalized TEST loss (1.00 = chance for contrastive; 1.0 = mean
    predictor for metric);
  - the probe from
    [2026-10-10_pooling_probe](../2026-10-10_pooling_probe/NOTEBOOK.md),
    on the same 3,415 cards and 150 decks: cos, linear R² and
    classification, kNN field, deck AUC (0.5 = chance).
- **References:**
  - untrained `init_mean`: deck AUC 0.541;
  - offline token whitening (768): 0.613;
  - offline pooled whitening: 0.637.

## Recipe

- **Whitening stats:** `data/whitening/modernbert_tokens_holdout_seed0.pt`
  (2026-10-10 16:43), from `scripts/fit_whitening.py`, fitted on the
  TRAIN-tier cards of the run's holdout. The exact command wasn't
  recorded. Its documented form:
  `PYTHONPATH=. python scripts/fit_whitening.py --holdout-from experiments/2026-10-10_pooling_pilot/configs/mean.yaml --out data/whitening/modernbert_tokens_holdout_seed0.pt`.
- **Training** (ROCm venv), one variant after another:
  `PYTHONPATH=. python scripts/run_training.py experiments/2026-10-10_pooling_pilot/configs/<variant>.yaml`.
  - The configs are the ones each run recorded, with dojo patterns
    expanded.
  - `tokwhite_tfm_attn1.yaml` never ran, so it is still the launch
    config.
  - To re-run, pass `--set run_directory=...`: the configs still point
    at the original run directories.
- **Probe:**
  `PYTHONPATH=. python experiments/2026-10-10_pooling_probe/analysis/probe_checkpoint.py data/runs/Oct_10_2026_pooling_pilot/<variant>`.
  It appends to `data/runs/2026-10-10_pooling_probe/legacy_logs/results_checkpoints.csv`, which was
  copied to [`results/probe_checkpoints.csv`](results/probe_checkpoints.csv).
- **TEST table:** [`../2026-10-10_lr_pilot/analysis/compare.py`](../2026-10-10_lr_pilot/analysis/compare.py)
  on `data/runs/Oct_10_2026_pooling_pilot/legacy_logs/<variant>.log` produces
  [`results/compare_test_loss.txt`](results/compare_test_loss.txt).
- **Raw data:**
  - runs, 2.4-2.5 GB each: `data/runs/Oct_10_2026_pooling_pilot/<variant>/`;
  - console logs: `data/runs/Oct_10_2026_pooling_pilot/legacy_logs/`.
- **Cost:** about 71 min per variant (84 for tfm_attn1). The probe takes
  minutes per variant.

## Log

### 2026-10-10 16:47 · Variants launched in sequence

mean, tokwhite_mean, attn1, tokwhite_attn1, attn4, tokwhite_attn4 and
tfm_attn1 each finished with exit 0. The last ended 2026-10-11 01:34
([`results/timeline.txt`](results/timeline.txt)). Each finished variant
was probed.

### 2026-10-10 (run ended 17:58) · mean: training made per-card vectors less informative

The pipeline output became far less anisotropic (cos 0.97 -> 0.57), but
every probe dropped against the untrained model:
- linear classification 0.78 -> 0.60;
- R² 0.39 -> 0.32;
- kNN 0.52 -> 0.43;
- deck AUC 0.541 -> 0.528.

TEST barely moved.

### 2026-10-10 (run ended 19:11) · tokwhite_mean: first variant off chance

Contrastive went 0.91 -> 0.89 -> 0.82 over the three rounds; the mean
variant stayed at 1.000 every round. Deck AUC is 0.678, above every
reference, so training on whitened tokens *added* deck signal. Field
probes are a little below the untrained model's.

### 2026-10-10 (runs ended 20:25, 21:40) · attn1 behaves like mean; tokwhite_attn1 works but no better than tokwhite_mean

A learned query over unwhitened tokens does not escape chance.

### 2026-10-10 (run ended 22:54) · attn4: best numeric-field probe, still at chance

Linear R² is 0.57, against 0.39 untrained and ≤ 0.34 for every other
variant. Four queries keep numeric detail (costs, stats). Contrastive
0.99.

### 2026-10-11 (run ended 00:08) · tokwhite_attn4: best deck AUC, but attn4's numeric gain is gone

Deck AUC 0.685 / 0.697 and contrastive 0.81, both the best. But R² falls
to 0.27. A plausible reading: the unwhitened queries read numeric
fields from the high-variance leading directions, which whitening
scales down.

### 2026-10-11 ~01:35 · Incident: probing tfm_attn1 exhausted system memory (1st)

tfm_attn1 trained (exit 0, checkpoint saved). Claude Code then stopped
the background job for critically low system memory during
`probe_checkpoint.py` on it. No score was written.
tokwhite_tfm_attn1 never started.

### 2026-10-11 ~01:37 · Re-run stopped again (2nd)

Re-run with 19.5 GB free. It was still running at ~09:09, when the other
variants' probes had taken minutes. No score was written, and the job
was stopped for low memory again. Free RAM afterwards: 19.6 of 31.9 GB.

Every non-transformer probe and the transformer *training* worked, so
the fault is likely specific to TransformerPooling in the probe path
(fp16 autocast eval, or `isolated_embeddings`). Not diagnosed. The
experiment was halted pending a machine check.

### 2026-10-11 · Correction to the findings file

`data/runs/Oct_10_2026_pooling_pilot/legacy_logs/findings.md` says "the only difference from the mean
arm is the fixed whitening, so the shared offset was what held
contrastive at chance". That overclaims, for two reasons:
- whitening removes the offset *and* rescales every direction;
- the parent probe found that mean-centering alone barely helps
  (0.543 -> 0.566), while whitening reaches 0.637.

The pilot shows that whitening fixes the problem. It does not show
that the offset is the cause. See Conclusions.

## Results

TEST, final round ([`results/compare_test_loss.txt`](results/compare_test_loss.txt),
which also has rounds 0-1; 20 trained metric dojos):

| variant | contrastive | metric | held out | metric < 1× | worst metric |
|---|---|---|---|---|---|
| mean | 1.000 | 0.97 | 1.10 | 11 | 1.9 |
| tokwhite_mean | 0.820 | 0.91 | 1.11 | 12 | 2.0 |
| attn1 | 1.000 | 0.98 | 1.08 | 10 | 2.1 |
| tokwhite_attn1 | 0.860 | 0.89 | 1.10 | 13 | 2.0 |
| attn4 | 0.990 | 0.97 | 1.06 | 11 | 1.7 |
| tokwhite_attn4 | **0.810** | 0.92 | 1.12 | 14 | 2.7 |
| tfm_attn1 | 1.000 | 0.96 | 1.09 | 13 | 2.0 |

Probe, final checkpoint, pipeline / isolated views
([`results/probe_checkpoints.csv`](results/probe_checkpoints.csv)):

| variant | cos | lin R² | lin cls | kNN field | deck AUC |
|---|---|---|---|---|---|
| init_mean (untrained) | 0.971 / 0.972 | 0.392 / 0.386 | 0.782 / 0.770 | 0.516 / 0.501 | 0.541 / 0.539 |
| mean | 0.568 / 0.944 | 0.320 / 0.298 | 0.596 / 0.565 | 0.433 / 0.438 | 0.528 / 0.532 |
| tokwhite_mean | 0.272 / 0.537 | 0.338 / 0.324 | 0.701 / 0.672 | 0.502 / 0.463 | 0.678 / 0.680 |
| attn1 | 0.608 / 0.965 | 0.292 / 0.290 | 0.618 / 0.607 | 0.445 / 0.442 | 0.533 / 0.544 |
| tokwhite_attn1 | 0.265 / 0.547 | 0.290 / 0.260 | 0.641 / 0.616 | 0.496 / 0.463 | 0.655 / 0.665 |
| attn4 | 0.679 / 0.942 | **0.570** / 0.542 | 0.649 / 0.616 | 0.468 / 0.452 | 0.550 / 0.559 |
| tokwhite_attn4 | 0.292 / 0.596 | 0.273 / 0.267 | 0.678 / 0.650 | 0.510 / 0.474 | **0.685 / 0.697** |
| tfm_attn1 | not probed | | | | |

## Conclusions

- **Problem: largely met, by whitening.** Every whitened variant leaves
  chance (final contrastive 0.81-0.86), and every unwhitened one stays
  at 0.99-1.00 whatever the pooling. Whitened deck AUC is 0.66-0.70,
  above the fixed-feature reference (0.637).
- **Hypothesis (the offset is the cause): not established.** Whitening
  removes the offset and also equalizes variance. The parent probe's
  centering-only result (0.566) suggests the rescaling matters at least
  as much. Separating the two needs a mean-subtraction-only variant.
- **Fix B (trainable pooling): little benefit at this budget.**
  - tokwhite_attn4 edges tokwhite_mean (deck AUC 0.685 vs 0.678,
    contrastive 0.81 vs 0.82) for 1.2M more parameters. That gap is
    within single-seed noise.
  - One query is slightly worse than mean.
  - The transformer variant was not scored.
- **Numeric detail is lost under whitening.** Only unwhitened attn4 keeps
  it (R² 0.57); whitened variants fall to 0.27-0.34.
- **Scope:** one seed, 3,000 steps, a reduced diet, a multi-card model,
  frozen ModernBERT. Held-out dojos stay ~1.10× in every variant.
- **Promoted to:** root `TODO.md`, "Adopt token whitening; test raw +
  whitened tokens together": whitening by default, whitened mean as the
  cheap baseline, whitened 4-query attention for longer runs.

## Open questions

- **Finish the variants:** probe tfm_attn1 (diagnose the memory fault
  first, on a small card sample in the foreground); train and probe
  tokwhite_tfm_attn1.
- **Separate the cause:** a mean-subtraction-only variant against full
  whitening.
- **Keep both signals:** give the pooling raw and whitened tokens
  together, to keep deck signal and numeric detail.
- **Noise floor:** run the same config with 3 training seeds, so gaps
  like 0.685 vs 0.678 can be judged.
