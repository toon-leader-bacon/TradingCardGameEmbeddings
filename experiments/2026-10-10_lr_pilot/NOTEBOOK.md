# encoder_lr pilot with the output LayerNorm

**Status:** concluded
**Dates:** 2026-10-10
**Parent:** [2026-10-09_model_by_diet](../2026-10-09_model_by_diet/NOTEBOOK.md)

> Problem: metric TEST losses swing wildly in series 1. Hypothesis: the
> new output LayerNorm fixes this on its own, so lowering `encoder_lr`
> isn't needed.

**Migration note (2026-10-11).** This was reconstructed from
`logs/lr_pilot/findings.md`, written while the runs went, and moved here
with its scripts and outputs. The hypothesis was stated as a question
("the norm or the LR?") before launch, but no bar was written. One seed.

## Aim

- **Problem:** in `multi_mixed`, the median metric TEST loss swung
  between 1.45× and 2.53× the mean predictor, and 17-42 trained metric
  dojos were above 5× in every round.
  - Bar (implicit, set after the fact): no dojo above 5×, and the median
    near or below 1×.
- **Hypothesis:** the output LayerNorm (no gain, no bias) bounds the
  embedding scale and stabilizes the heads by itself.
  - Refuted if the norm at `encoder_lr` 1e-3 is still unstable, while
    lower LRs are stable.
- **Priority:** the fix first. Choose an `encoder_lr` for the next runs.

## Method

- **Terms:** three *pilots*, named by `encoder_lr`: 3e-4, 1e-4 and 1e-3.
- **Configs:** copies of series 1's `multi_mixed`. Diffed: they differ
  from it only in `encoder_lr`, `max_rounds` (3 instead of 20) and the
  run directory. `head_lr` is 1e-3 in all three.
- **Code difference from the baseline:** the LayerNorm at the end of
  every EmbeddingHead and of the attention stack.
- **Baseline:** the archived pre-LayerNorm `multi_mixed` run at
  `encoder_lr` 1e-3 (series 1). Its first three rounds compare directly;
  its later rounds are context.
- **Budget:** 3 rounds × 1,000 steps each.
- **Metrics:**
  - normalized TEST loss medians and counts per round (1.0 = mean
    predictor; contrastive 1.0 = chance);
  - embedding geometry on 12 fixed cards from 5 games: final norm,
    mean pairwise cos, and card-specific spread (the norm left after
    subtracting the 12 cards' mean).
- **Confounds:**
  - one seed;
  - 128 TEST examples per dojo, so single dojos are noisy, and medians
    and counts are compared instead;
  - the baseline differs in code (the norm), which is the variable under
    test.

## Recipe

- Data: as series 1.
- Commands (ROCm venv), run one after another:
  `PYTHONPATH=. python scripts/run_training.py experiments/2026-10-10_lr_pilot/configs/multi_mixed_lr<lr>.yaml`.
  The configs are the ones each run recorded, with dojo patterns
  expanded. To re-run, pass `--set run_directory=...`: they still point
  at the original run directories.
- Raw data:
  - runs: `data/runs/Oct_10_2026_lr_pilot/<pilot>/`;
  - console logs: `logs/lr_pilot/<pilot>.log`.
- Analysis:
  - [`analysis/compare.py`](analysis/compare.py) on the console logs
    produces [`results/compare_test_loss.txt`](results/compare_test_loss.txt);
  - [`../2026-10-09_model_by_diet/analysis/round_summary.py`](../2026-10-09_model_by_diet/analysis/round_summary.py)
    and `embedding_scale.py` from the same folder produce
    [`results/round_and_scale_summary.txt`](results/round_and_scale_summary.txt);
  - the centered-geometry table in the Log came from a scratch script
    (`bert_geometry.py`) that wasn't kept.

## Log

### 2026-10-10 07:51 · Pilots launched in sequence

3e-4 ran 07:51-10:09, 1e-4 ran 10:10-12:24, 1e-3 ran 12:24-14:46, all
exit 0 ([`results/timeline.txt`](results/timeline.txt)). Rounds took
about 36 min each.

### 2026-10-10 (time not recorded) · Round 0 with the LayerNorm: no metric blowups

The worst dojo was 4× (baseline 41×), the metric median 1.13 (baseline
2.02), and 22 trained metric dojos were already below the mean predictor
(baseline 5). Contrastive was unchanged at chance.

### 2026-10-10 (time not recorded) · The final embeddings are dominated by one shared offset

At round 1 (3e-4), raw cos is 0.999 across five games. After
subtracting the 12 cards' mean, the geometry is meaningful, and *better*
than at init (scratch `bert_geometry.py`; "ctr" = mean-centered):

| stage | raw cos | same-game cos (ctr) | cross-game cos (ctr) |
|---|---|---|---|
| ModernBERT mean-pool (frozen) | 0.976 | 0.31 | -0.15 |
| ModernBERT CLS | 0.988 | 0.21 | -0.14 |
| embedding head, init | 0.984 | 0.28 | -0.15 |
| embedding head, round 1 | 0.974 | **0.54** | -0.15 |
| after attention stack, round 1 (alone) | 0.999 | 0.55 | -0.16 |
| after attention stack, round 1 (12 together) | 1.000 | 0.55 | -0.16 |

So:
- frozen ModernBERT is strongly anisotropic, but informative once
  centered;
- training sharpened game structure in the head (same-game 0.28 ->
  0.54);
- the pre-norm attention stack then adds a large constant vector, and
  the no-gain LayerNorm leaves the card-specific part small.

An earlier note at this point called this a collapse. It is not: the
centered structure survives, and the metric heads (linear and MLP) are
indifferent to a constant offset.

### 2026-10-10 (time not recorded) · Round 2 (3e-4) and the final 1e-4 checkpoint

- **3e-4, round 2:** the offset now reaches the head too (head raw cos
  0.974 -> 0.993; final 1.000). Centered: same-game 0.49 (head) and 0.54
  (final, embedded apart) vs cross-game -0.14 / -0.15. Game structure
  is kept but not improving past round 1.
- **1e-4, final checkpoint:** the same centered structure as 3e-4, with
  less offset (head raw cos 0.975 vs 0.993).

### 2026-10-10 14:46 · All three done

## Results

Median normalized TEST loss
([`results/compare_test_loss.txt`](results/compare_test_loss.txt)).
The baseline is pre-LayerNorm at `encoder_lr` 1e-3:

| run | round | contrastive | metric | held out | metric < 1× | metric > 5× | worst |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 1.000 | 2.02 | 1.48 | 5 | 19 | 40.7 |
| baseline | 1 | 0.990 | 2.10 | 1.82 | 9 | 30 | 112.8 |
| baseline | 2 | 1.000 | 2.34 | 1.66 | 5 | 25 | 81.3 |
| baseline, rounds 3-10 | | 0.97-0.99 | 1.45-2.53 | 1.30-1.74 | 5-8 | 17-42 | 33-147 |
| LN + 3e-4 | 2 | 1.000 | 1.06 | 1.08 | 31 | 0 | 3.6 |
| LN + 1e-4 | 2 | 1.000 | 1.08 | 1.08 | 33 | 0 | 3.6 |
| LN + 1e-3 | 2 | 1.000 | 1.08 | 1.10 | 31 | 0 | 4.0 |

Rounds 0-1 of each pilot are in the results file.

Final embedding geometry after the attention stack
([`results/round_and_scale_summary.txt`](results/round_and_scale_summary.txt);
12 cards, norm 16 in all):

| pilot | raw cos | card-specific spread |
|---|---|---|
| init (all) | 0.985 | 1.83 |
| 3e-4 | 1.000 | 0.20 |
| 1e-4 | 0.997 | 0.76 |
| 1e-3 | 1.000 | 0.00 |

Centered geometry is the same in all three (same-game ~0.53-0.56,
cross-game ~-0.15).

The consistently worst dojos in all three are the `sts2_runs` per-card
run statistics, at 2.4-4.5×.

## Conclusions

- **Problem: met.** With the norm, no trained metric dojo is above 5×
  in any round (baseline: 17-42), and the final metric medians are
  1.06-1.08 (baseline: 1.45-2.53).
- **Hypothesis: supported (one seed).** At the baseline's own
  `encoder_lr` of 1e-3, the norm alone removes the blowups. The LR
  barely matters for the metrics over 3 rounds: final medians 1.06,
  1.08 and 1.08 are within the noise of 128-example TEST.
- **Further finding:**
  - the LR does matter for the shared offset: card-specific spread is
    0.00 at 1e-3, 0.20 at 3e-4 and 0.76 at 1e-4;
  - the offset carries no information (centered geometry is identical);
  - contrastive stays at chance at every LR, so the LR is not the fix
    for contrastive.
- **Promoted to:**
  - `encoder_lr` 1e-4 as the default for the next runs, until the offset
    itself is removed (then re-check 3e-4);
  - the offset problem became
    [2026-10-10_pooling_probe](../2026-10-10_pooling_probe/NOTEBOOK.md)
    and [2026-10-10_pooling_pilot](../2026-10-10_pooling_pilot/NOTEBOOK.md);
  - the centering ideas and the worst-dojo note went to
    [`../TODO.md`](../TODO.md).

## Open questions

- The `sts2_runs` per-card statistics are the worst dojos at every LR.
  They are candidates for label transforms or a Huber loss.
- Contrastive was also at 0.97-0.99 in the baseline, where cos was
  0.86-0.97. So the offset may not be the whole story: the same-deck task
  may simply be hard at ~20 steps per contrastive dojo per round.
