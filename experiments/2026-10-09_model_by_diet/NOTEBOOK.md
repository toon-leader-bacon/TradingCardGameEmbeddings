# Series 1: model × diet

**Status:** halted
**Dates:** 2026-10-09 - 2026-10-10
**Parent:** [2026-09-30_first_train_shakeout](../2026-09-30_first_train_shakeout/NOTEBOOK.md)

> Problem: which model (single-card or multi-card) and which diet
> (contrastive only, metric dojos only, or 50/50 mixed) trains the best
> general card embedding?

**Migration note (2026-10-11).** This was reconstructed from the series 1
entry in `experiments/TODO.md` (then `EXPERIMENTS_TODO.md`), the
pre-launch design in [`configs/README.md`](configs/README.md), and the
run's findings file, now archived at
`logs/archive/2026-10-10_multi_mixed_before_layernorm/multi_mixed_findings.md`.
No predictions or bar were written before launch. Only one of the six
cells ran, and it was stopped at round 10 of 20.

## Aim

A 2 × 3 comparison. Single-card and multi-card models, each trained on
three diets:
- contrastive dojos only;
- metric dojos only;
- 50/50 mixed, with per-dojo diet weights so the contrastive keys
  together are drawn half the time in expectation.

The first shakeout showed that contrastive training alone collapses
the embedding. The question was whether a mixed diet keeps more card
information, and whether the multi-card model's group attention helps.
This aim was a problem without a stated bar or hypothesis.

## Method

The full pre-launch design (2026-10-09) is
[`configs/README.md`](configs/README.md). In short:

- **Terms:** a *cell* is one model × diet combination. There are 6 cells
  and 2 untrained controls (planned, never built).
- **Model:**
  - frozen ModernBERT-base;
  - `residual_mlp` embedding head with 256-wide embeddings;
  - about 2.9M trainable parameters in both models (2,898,176 for
    `single_mixed`);
  - multi-card: 3 MLP blocks plus 2 pre-norm group-attention layers.
- **Dojos:**
  - 25 contrastive dojos: 5 styles × 5 games, Gwent excluded;
  - 124 metric dojos, weighted by TRAIN count ** 0.3;
  - 13 held-out dojos, never trained.
- **Holdout:** Gwent is a held-out game. Card seed 0, tiers 8/1/1.
- **Budget:** 20 rounds × 1,000 steps. Saturation was disabled, so the
  diet never changes. `max_batch_cost` 256, fp16.
- **Optimizer:** `encoder_lr` and `head_lr` 1e-3.
- **Metric:** TEST loss divided by each dojo's baseline (1.0 = the mean
  predictor; for contrastive, 1.0 = chance), 128 examples per dojo per
  round.

## Recipe

- Data: every metric parquet and split the catalog keys need, and
  passing `run_training.py --check`.
- Command (ROCm venv):
  `PYTHONPATH=. python scripts/run_training.py experiments/2026-10-09_model_by_diet/configs/<cell>.yaml`.
  - `configs/multi_mixed.yaml` is the config the run recorded, with its
    dojo patterns expanded.
  - The five cells that never ran keep their launch configs, whose
    comments give the pre-migration path `configs/training/Oct_9_2026/`.
  - To re-run, pass `--set run_directory=...`: the configs still point
    at the original run directories.
- Raw data:
  - the console log, the run records and the diagnostics are in
    `logs/archive/2026-10-10_multi_mixed_before_layernorm/`;
  - the checkpoints were deleted.
- Analysis:
  - [`analysis/round_summary.py`](analysis/round_summary.py) takes
    `rounds.csv`;
  - [`analysis/embedding_scale.py`](analysis/embedding_scale.py) takes a
    config and a run directory, and needs checkpoints;
  - the comparison table comes from
    [`../2026-10-10_lr_pilot/analysis/compare.py`](../2026-10-10_lr_pilot/analysis/compare.py)
    on the console log.

## Log

### 2026-10-09 22:23 · `--check` on `single_mixed.yaml` failed

The re-split of the metric data failed with `pyarrow ArrowInvalid:
offset overflow while concatenating arrays`
(`logs/series1_check/single_mixed_resplit_failed1.log`).

Cause: bucket-shuffle sizing. Parquet's dictionary-encoded byte count
underestimated the 17lands draft file about 30×. The fix sizes buckets
from a sample of decoded rows.

### 2026-10-10 00:15 · `--check` passed; `multi_mixed` launched

162 of 162 dojos passed preflight
(`logs/series1_check/single_mixed_resplit.log`). The re-split took about
1 h 50 min; calibration took 27 min (61 before the fix).

The full `multi_mixed` run started at 00:15. Startup took about 28 min.
Each round took about 37 min: 23 min for 1,000 steps at about 1.4 s per
step, plus about 14 min of TEST over 162 dojos.

Resources were flat: about 4.4 GiB of GPU memory and about 7.5 GiB of
private RAM ([`results/multi_mixed_resources.csv`](results/multi_mixed_resources.csv)).
Round 0 had 7 fp16 overflow skips.

### 2026-10-10 (~02:00, after round 1) · Big issue: the embeddings are a growing shared vector

Measured with `analysis/embedding_scale.py` (12 card texts from 5 games):

| checkpoint | final norm | shared-mean norm | card-specific spread | mean pairwise cos |
|---|---|---|---|---|
| init | 20.6 | 20.4 | 2.4 | 0.985 |
| round 0 | 79.1 | 78.7 | 7.5 | 0.990 |
| round 1 | 107.0 | 106.4 | 11.6 | 0.986 |

- About 90% of every embedding is one vector shared by all cards. This
  is already true at init: ModernBERT's pooled output is anisotropic,
  and nothing centers or normalizes it.
- The embedding grew about 5× in two rounds. Nothing bounds its scale:
  the head ends in a bare Linear, and the pre-norm attention stack has
  no final LayerNorm.

That fits every symptom in TEST:
- **Contrastive at 0.96-1.00×.** The loss is cosine-based, and every pair
  of cards sits at cos ~0.99.
- **Metric dojos swing wildly** (cardvault cost_regression 0.83× -> 7.5×).
  Each head sees inputs on a new scale every round.
- **Held-out heads blow up too.** A random head's logits scale with its
  input.

No code was changed. The run was left going, to see whether the growth
continued.

### 2026-10-10 07:51 · Stopped by request after round 10

After round 2 the norm stopped growing and oscillated between 61 and
124. Cos drifted down from 0.97 to 0.86, and the card-specific spread
grew from 16 to 30
([`results/multi_mixed_scale_by_round.txt`](results/multi_mixed_scale_by_round.txt)).
The metric losses never settled (see Results).

Fix taken: a LayerNorm without gain or bias at the end of every
EmbeddingHead and of MultiCardModel's attention stack. The series
paused, and an `encoder_lr` pilot with the norm came next:
[2026-10-10_lr_pilot](../2026-10-10_lr_pilot/NOTEBOOK.md).

The checkpoints were deleted. The other five cells never ran.

## Results

Median normalized TEST loss per round
([`results/compare_test_loss.txt`](results/compare_test_loss.txt), from
the console log; 149 trained dojos, of which 124 are metric):

| round | contrastive | metric | held out | metric < 1× | metric > 5× | worst metric |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 2.02 | 1.48 | 5 | 19 | 40.7 |
| 1 | 0.990 | 2.10 | 1.82 | 9 | 30 | 112.8 |
| 2 | 1.000 | 2.34 | 1.66 | 5 | 25 | 81.3 |
| 5 | 0.990 | 2.53 | 1.33 | 5 | 42 | 80.8 |
| 9 | 0.970 | 1.45 | 1.30 | 8 | 17 | 34.9 |
| 10 | 0.970 | 1.96 | 1.63 | 7 | 30 | 82.4 |

The full table, rounds 0-10, is in the results file. Round timing is in
[`results/round_summary.txt`](results/round_summary.txt): 36-38 min per
round, 407.5 min for 11 rounds. That output is computed from
`rounds.csv`, so its medians differ slightly from the console log's
rounded values.

## Conclusions

- **The problem was not answered.** The cell comparison never ran.
  `multi_mixed` showed that the setup itself was unstable, so comparing
  cells would have measured that instability.
- **Finding (one run, one seed):** with a frozen, anisotropic
  ModernBERT and no normalization at the output, the embedding is
  dominated by one shared, unbounded vector. That destabilizes metric
  heads and holds cosine contrastive losses at chance.
- **Promoted to:**
  - the output LayerNorm, now in `EmbeddingHead` and `MultiCardModel`
    (`src/encoder_model/`);
  - the follow-up [2026-10-10_lr_pilot](../2026-10-10_lr_pilot/NOTEBOOK.md).

## Open questions

- Re-run the six cells once the embedding is stable and contrastive
  leaves chance. The pooling and whitening work
  ([2026-10-10_pooling_pilot](../2026-10-10_pooling_pilot/NOTEBOOK.md))
  is the current blocker.
- The 14-minute TEST pass is about 38% of each round; see
  "Measuring training" in [`../TODO.md`](../TODO.md).
