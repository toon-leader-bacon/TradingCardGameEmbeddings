# Frozen-feature pooling probe

**Status:** concluded
**Dates:** 2026-10-10
**Parent:** [2026-10-10_lr_pilot](../2026-10-10_lr_pilot/NOTEBOOK.md)

> Problem: before any training, how much same-deck signal can cosine see
> in frozen text-encoder features, and which encoder, serialization,
> pooling and post-processing expose the most of it?

**Migration note (2026-10-11).** This was reconstructed from
`logs/pooling_probe/findings.md`, written as the probe ran, and moved
here with its scripts and result CSVs. No bar or predictions were
written beforehand. The working idea, from the LR pilot, was that
anisotropy hides card differences from cosine.

## Aim

- **Problem:** contrastive dojos sit at chance in every training run so
  far. Measure the frozen-feature ceiling for the contrastive task
  without training, and find the best fixed starting vector. This gives
  later trainable poolings a bar to beat.
- **Hypothesis (implicit):** frozen ModernBERT's anisotropy (mean
  pairwise cos ~0.96) hides card differences from cosine, so removing it
  should raise same-deck AUC.
  - Refuted if centering or whitening leaves deck AUC near today's
    0.54.

## Method

- **Cards:** 3,415 cards, plus 150 decks sampled from 7 games (25 decks
  per game, except Hearthstone, which has no decks and contributes 200
  random cards). Seed 0.
- **Compared:** 3 encoders × 2 serializations × 3 poolings × 4
  post-processing steps = 72 rows:
  - **encoders:** `answerdotai/ModernBERT-base` (ours),
    `nomic-ai/modernbert-embed-base`, `Alibaba-NLP/gte-modernbert-base`;
  - **serializations:** compact JSON (today's) and `key: value` lines;
  - **poolings:** mean, value-only mean, CLS;
  - **post-processing:** raw, mean-center, remove the top 8 directions,
    PCA-whiten to 256 dims.
- **Control:** ModernBERT, JSON, mean, raw: what training used at the
  time.
- **Metrics:**
  - anisotropy: mean pairwise cos, and the top principal component's
    share of variance;
  - linear probes on 17 card fields: 5-fold ridge R² and logistic
    balanced accuracy;
  - cosine 10-NN balanced accuracy on the same fields and on game;
  - **deck AUC:** P(cos to a deck-mate > cos to a same-game card outside
    the deck). 0.5 = chance. This is the contrastive dojos' task with no
    training.
- **Follow-up (same day):** token whitening before mean pooling, fitted
  on every real token's state.
- **Confounds:**
  - one seed and 25 decks per game, so AUC differences under ~0.01 are
    noise;
  - whitening is fitted on the same cards it is scored on (slightly
    optimistic);
  - the probe labels appear in the card text, so field probes measure
    how readable a field is, not inference.

## Recipe

- GPU (the CPU was too slow to finish one encoder). Encoding takes ~25 s
  per encoder and serialization.
- Commands (ROCm venv):
  - [`analysis/probe.py`](analysis/probe.py):
    `PYTHONPATH=. python experiments/2026-10-10_pooling_probe/analysis/probe.py`;
  - then [`analysis/probe_token_whitening.py`](analysis/probe_token_whitening.py),
    which reads `probe.py`'s cache.
- **Paths:** the scripts are kept as run. They still write to
  `logs/pooling_probe/`: `cache/` (pooled vectors, 181 MB, ignored),
  `results.csv`, `results_token_whitening.csv` and the console logs. The
  CSVs were copied into [`results/`](results/).
- [`analysis/probe_checkpoint.py`](analysis/probe_checkpoint.py) lives
  here because it imports `probe.py` from its own folder. It scores
  *trained* checkpoints with the same cards and metrics, and is used by
  [2026-10-10_pooling_pilot](../2026-10-10_pooling_pilot/NOTEBOOK.md).

## Log

### 2026-10-10 15:33 · `probe.py` done: 72 rows

[`results/results.csv`](results/results.csv); the console log is
`logs/pooling_probe/probe.log`. The headline result is in Results.

### 2026-10-10 16:07 · Token whitening before mean pooling

Mean pooling is linear, so whitening every token and then averaging
equals the cached mean vector, centered by the token mean and projected
by the token PCA. One extra encoding pass per encoder and serialization
collected the token statistics.
[`results/results_token_whitening.csv`](results/results_token_whitening.csv).

Why: a trainable pooling sees tokens, not pooled vectors. So the
whitening we would actually build into the model is token-level, and its
fixed-feature score is the "token whitening + mean" variant of the
pooling pilot.

## Results

Ranking by deck AUC ([`results/results.csv`](results/results.csv)):

| encoder | text | pooling | post | cos | lin R² | kNN field | deck AUC |
|---|---|---|---|---|---|---|---|
| gte-modernbert | kv | mean | whiten256 | 0.00 | 0.45 | 0.66 | **0.652** |
| nomic-embed | kv | value | whiten256 | 0.00 | 0.43 | 0.68 | 0.649 |
| modernbert | kv | mean | whiten256 | 0.00 | **0.53** | **0.69** | 0.643 |
| modernbert | json | mean | whiten256 | 0.00 | 0.50 | 0.67 | 0.637 |
| nomic-embed | kv | mean | raw | 0.69 | 0.39 | 0.66 | 0.621 |
| gte-modernbert | kv | mean | raw | 0.55 | 0.45 | 0.59 | 0.584 |
| **modernbert (control)** | json | mean | raw | 0.96 | 0.50 | 0.54 | 0.543 |

Per game, control -> best: MTG 0.58 -> 0.66, Pokemon 0.50 -> 0.69,
Gwent 0.59 -> 0.69, StS2 0.53 -> 0.65, FaB 0.54 -> 0.65, Dominion 0.51
-> 0.58.

Token whitening before mean pooling, deck AUC
([`results/results_token_whitening.csv`](results/results_token_whitening.csv)):

| encoder | text | keep 256 | keep 768 | pooled whiten256 | none |
|---|---|---|---|---|---|
| modernbert | json | 0.593 | 0.613 | 0.637 | 0.543 |
| modernbert | kv | 0.601 | 0.624 | 0.643 | 0.548 |
| nomic-embed | json | 0.631 | 0.640 | - | 0.607 |
| nomic-embed | kv | 0.640 | 0.651 | 0.643 | 0.621 |
| gte-modernbert | json | 0.628 | 0.638 | 0.635 | 0.579 |
| gte-modernbert | kv | 0.640 | **0.655** | 0.652 | 0.584 |

Observations:
- **Post-processing matters more than the encoder.** Whitening is the
  best post step for every encoder, and the three whitened encoders land
  within 0.01 of each other (0.643-0.652).
  - Mean-centering alone barely helps (0.543 -> 0.566).
  - Removing the top 8 directions gets about half the gain.
- **Embedding-trained encoders win only before whitening** (raw 0.58-0.62
  vs 0.54), and they keep less field detail (linear R² 0.39-0.45 vs
  0.50-0.53).
- **Pooling:** mean ≈ value-only mean > CLS.
- **Serialization:** `key: value` beats JSON slightly everywhere, and
  costs ~7% fewer tokens (101 vs 108 per card).
- **The information is there; cosine can't see it.** Linear classifiers
  read card fields at ~0.86 balanced accuracy in every variant, raw
  included.
- **Token whitening:** for our ModernBERT it recovers most of the gain
  (0.543 -> 0.613, keeping all 768 components), but stays below pooled
  whitening (0.637). Keeping 256 components loses 0.01-0.02 every time.

## Conclusions

- **Problem: answered.** The frozen-feature ceiling for deck AUC is
  ~0.65, reached by whitening. That is the bar for trainable poolings.
  Today's control is 0.543.
- **Hypothesis: supported, for the geometry.** Removing anisotropy
  raises deck AUC from 0.54 to 0.64-0.65 with the same information
  (linear probes are unchanged). But plain centering does little (0.566).
  So the gain comes from equalizing variance across directions, not just
  from removing the shared mean. "The offset hides the signal" is
  therefore too narrow: it is the offset *and* the dominant high-variance
  directions.
- **Not worth switching encoders.** Whitened ModernBERT matches the
  embedding-trained ones and keeps the best field probes.
- **Promoted to:**
  - the token whitening enrichment and `scripts/fit_whitening.py`, which
    keeps all components by default (`src/encoder_model/whitening/`);
  - the pooling series design in [`../TODO.md`](../TODO.md), and
    [2026-10-10_pooling_pilot](../2026-10-10_pooling_pilot/NOTEBOOK.md).

## Open questions

- `key: value` serialization is slightly better and cheaper than JSON.
  It is not adopted yet.
- Whitening fitted on held-out cards versus the scored cards: the
  optimism is small, but it hasn't been measured.
