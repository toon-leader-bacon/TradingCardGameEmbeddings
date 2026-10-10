# Experiments TODO

Training experiments to run, roughly in order. Code work an experiment
needs lives in the relevant container's `TODO.md` or a `plans/` file;
this list is about runs and comparisons.

## Fixed for every series unless an entry says otherwise

- Same seed and `HoldoutSpec` across all runs in a series.
- Equal step budget per run; saturation does not end a run early.
- **Gwent is held out** from every training diet (its contrastive and
  metric dojos alike), as the unseen game for evaluation.
  `cross_game.rarity_tier` is held out too: it is the intrinsic "medium"
  label set. About 10 more keys spread across games and cells are held
  out for extrinsic evaluation.
- Untrained single-card and multi-card models as controls.

## Series

- [ ] **1. Model × diet (first real series).** Single-card and multi-card
  models, each trained on three diets: contrastive only, metric dojos
  only, and 50/50 (per-dojo diet weights, so the contrastive keys
  together are drawn half the time in expectation). ModernBERT frozen,
  `residual_mlp` head, both models about 2.9M trainable parameters (sizes
  in `src/training/TODO.md`, section D). Both models get the same
  contrastive dojos: all five training styles, each style equally often
  (series 2 then separates them). Configs: `configs/training/Oct_9_2026/`. 6 runs
  plus 2 controls, evaluated together (8 encoders fit the learning-curve
  plots).
- [ ] **2. Contrastive styles against each other.** Single-card,
  cross-slice card match, slice match, missing card, odd one out
  (`plans/multi_card_contrastive_dojos.md`). Card in contexts is not an
  arm: it is an evaluation probe (`src/evaluation/TODO.md`).
- [ ] **3. Unfrozen ModernBERT.** Repeat the useful arms of series 1 with
  the language model trainable; needs VRAM measurement and likely
  gradient checkpointing (`src/training/TODO.md`, section E).

## Ablations, when a series raises the question

- [ ] Contrastive share of the mixed diet (e.g. 25 / 50 / 75%).
- [ ] Game balance: today's per-dojo diet vs. a per-game two-level diet
  (`src/training/TODO.md`, section C).
- [ ] MSE vs. Huber regression loss (`src/training/TODO.md`, section C).
- [ ] Staple subsampling in contrastive pairs, arms t = inf / 0.1 / 0.01
  (`src/training/TODO.md`, section C).
- [ ] Curriculum: a contrastive phase, then a metric-dojo phase, against
  the mixed diet.

## Ideas, not yet scheduled

Each idea is marked config-only or code. When one is picked up, it
moves to an ablation above or to a series, and any code it needs moves
to the relevant `TODO.md`.

### What the runs so far show

From the first `multi_mixed` run (archived in
`logs/archive/2026-10-10_multi_mixed_before_layernorm/`) and the
LayerNorm + `encoder_lr` pilot (`logs/lr_pilot/findings.md`):

- **Most metric heads barely train.** One optimizer step trains one
  batch from one dojo. With 149 trained dojos and the 50/50 diet
  (metric dojos weighted by TRAIN count ** 0.3), steps per dojo per
  1,000-step round:

  | steps per round | dojos |
  |---|---|
  | < 1 | 41 |
  | 1-3 | 42 |
  | 3-10 | 31 |
  | >= 10 | 35 |

  A dojo with a 0.09% share gets about 18 steps in a 20,000-step run.
  Its head (~130k parameters) stays close to its initialization. Its
  TEST loss then mostly measures head noise, and its gradients reach the
  encoder through a near-random head.
- **Adam on a rarely stepped head is close to sign descent.** With bias
  correction, Adam's first steps move every weight by about `lr`,
  whatever the gradient size. A head stepped a few dozen times never
  leaves that regime, so `head_lr` 1e-3 is in effect a fixed 1e-3 step
  per weight each time the head is visited.
- **The embedding scale mattered most.** Without a norm, about 90% of
  each card embedding was one direction shared by all cards, with norm
  60-124. Metric TEST loss swung between 1.5x and 2.5x the mean
  predictor, and 17-42 dojos were above 5x every round. A final
  LayerNorm (no gain, no bias) fixed the scale. Round 0 of the pilot:
  median 1.13x and no dojo above 5x.
- **TEST is expensive.** About 14 of each round's ~37 minutes go to
  scoring all 162 dojos on 128 examples each.

### Getting more steps to each head

- **Head warm-up phase** (config-only). A first phase with
  `encoder_trainable: false` trains the heads alone, maybe for a few
  thousand steps; then the joint phase. The encoder's first gradients
  then come through heads that already fit something. The warm-up
  steps are cheaper too, since they skip the encoder's backward pass.
  This is the linear-probe-then-fine-tune (LP-FT) idea.
- **More steps per round** (config-only). This does not change any
  dojo's share of steps. It gives more steps between TEST passes, so:
  - less of the run goes to TEST: with 5,000-step rounds, about 14 of
    ~130 minutes instead of 14 of 37;
  - each per-round change in TEST loss reflects more training;
  - the saturation tracker makes fewer, coarser decisions.

  On a fixed budget, the total steps a head gets depends on its share,
  not on round length. Giving a head more training means more total
  steps, a bigger share, or fewer dojos sharing the budget.
- **Flatter metric diet** (config-only). `within: {rule: temperature,
  alpha: 0}` makes the metric half uniform: about 4 steps per dojo per
  round, instead of 0.9-31. A per-dojo floor would need code
  (`src/training/diet/table_diet.py`).
- **Fewer metric dojos per run** (config-only). Drop near-duplicates,
  such as the per-card run statistics that both `sts2_runs` and `sts_gg`
  have, or rotate subsets of dojos across phases.
- **Several dojos per optimizer step** (code). Accumulate gradients
  from k dojos' batches before one step. Every step then touches k
  heads, and each encoder update averages over k tasks instead of
  jumping between tasks. This is standard multi-task practice. It costs
  k forward/backward passes per step (`src/training/trainer.py`,
  `_take_step`).

### Head resets

- **Reset decoder heads occasionally** (code, small). Re-initialize
  every head, and its Adam state, every N rounds or once before the
  final rounds. Why it could help:
  - Heads fitted to an older encoder can pin the encoder in place, and
    fresh heads remove that pull (cf. "primacy bias" / plasticity loss
    in RL, where periodic head resets help).
  - The TEST loss right after a reset shows what the encoder alone
    carries, as the held-out dojos do today.

  The risk is junk gradients reaching the encoder right after a reset.
  To avoid that, follow each reset with a short heads-only stretch, like
  the warm-up phase above. Possible config: a phase flag
  `reset_heads: true`, or `reset_heads_every_rounds: N`.

### Curriculum: easy to hard

- **Start with easier dojos, move to harder ones** (config-only with
  hand-picked phases; code for a diet that shifts gradually). Likely a
  series 2 or 3 experiment. A possible order:
  1. contrastive dojos and single-card field masks and regressions
     (cost, type, rarity: readable straight off the card text);
  2. single-card play statistics (win rates, pick rates);
  3. multi-card and game-outcome dojos (deck win prediction, next buy,
     pool-conditioned pick).

  Hand-written phases work with today's phase list. A smooth version
  would move diet weight from easy dojos to hard ones over the run.
  Difficulty could be measured instead of hand-labeled: each dojo's
  normalized TEST loss after the head warm-up phase, or a self-paced
  diet driven by live TEST loss. A narrower version is already listed
  under Ablations: a contrastive phase, then a metric phase.

### Optimizer

- **LR schedule** (code, small). The trainer has no warmup or decay.
  Linear warmup over a few hundred steps, then cosine decay, is the
  default for transformers. It limits the two risky ends: large early
  steps through random heads, and thrashing late in the run.
- **`head_lr` sweep** (config-only). Because of the near-sign-descent
  effect above, `head_lr` 3e-4 vs. 1e-3 may matter as much as
  `encoder_lr`.
- **Weight decay** on the encoder group (code, small). Today it is
  AdamW's default, 0.01.

### Architecture

- **Remove the shared direction** (code). The LayerNorm fixes the scale
  but does not stop all cards sharing one direction. One fix: center
  the pooled ModernBERT features with a running mean before the head,
  as in whitening / "all-but-the-top". Track it with mean pairwise cos
  (`logs/lr_pilot/embedding_scale.py`).
- **Token-level head; thawing ModernBERT's top layers**: see
  `src/encoder_model/TODO.md` and series 3.

### Measuring training

- **Cheaper, less noisy TEST** (config-only). Use fewer, longer rounds
  (above), or raise `eval_examples_per_dojo` for the final round only.
  At 128 examples, one dojo's ratio can swing by tens of percent from
  sampling alone. Compare single rounds by medians and counts across
  dojos instead, as `logs/lr_pilot/compare.py` does.
