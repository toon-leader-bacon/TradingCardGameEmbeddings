# Experiments TODO

Training experiments to run, roughly in order, plus ideas not yet
scheduled. Experiments already started have their own folder and
notebook, listed in [`README.md`](README.md). Code work an experiment
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
  in the series notebook). Both models get the same
  contrastive dojos: all five training styles, each style equally often
  (series 2 then separates them). Configs and notebook: [2026-10-09_model_by_diet](2026-10-09_model_by_diet/NOTEBOOK.md). 6 runs
  plus 2 controls, evaluated together (8 encoders fit the learning-curve
  plots).
- [ ] **2. Contrastive styles against each other.** Single-card,
  cross-slice card match, slice match, missing card, odd one out
  (`plans/multi_card_contrastive_dojos.md`). Card in contexts is not an
  arm: it is an evaluation probe (`src/evaluation/TODO.md`).
- [ ] **3. Unfrozen ModernBERT.** Repeat the useful arms of series 1 with
  the language model trainable; needs VRAM measurement and likely
  gradient checkpointing (`src/encoder_model/TODO.md`).

- [ ] **Pooling and whitening** (code built: `model.pooling` / `model.enrichment`
  in `src/training/README.md`; stats from `scripts/fit_whitening.py`).
  Fixed-feature bar from the offline probe
  ([2026-10-10_pooling_probe](2026-10-10_pooling_probe/NOTEBOOK.md)): same-deck AUC ~0.65 (today's
  mean-pooled JSON: 0.54). Arms, all with the same text encoder, head
  and diet:

  | arm | enrichment | pooling | trained? |
  |---|---|---|---|
  | 1 | - | mean (today) | no |
  | 2 | token whitening | mean | no |
  | 3 | - | attention, 1 query | yes |
  | 4 | - | attention, 4-8 queries | yes |
  | 5 | - | transformer (1-2 layers) -> attention | yes |
  | 6 | token whitening | attention, 1 query | yes |
  | 7 | token whitening | attention, 4-8 queries | yes |
  | 8 | token whitening | transformer -> attention | yes |

  Arms 1-2 can be scored offline first (the 0.65 bar is pooled-vector
  whitening; token whitening is not yet measured). To
  decide: diet (contrastive + a few field dojos keeps TEST cheap),
  single- vs. multi-card model (single-card isolates pooling from the
  attention stack's offset), rounds, and the scoring (probe
  `--checkpoint` mode plus TEST loss). Report trainable parameter counts:
  multi-query concatenation widens the head's input.

  **Pilot 2026-10-10** (multi-card, reduced mixed diet, 3 x 1000 steps,
  4 queries; notebook
  [2026-10-10_pooling_pilot](2026-10-10_pooling_pilot/NOTEBOOK.md)): arms 1-4, 6, 7 done; arm 5
  trained but not probed (the probe exhausted system memory twice); arm
  8 not run. Whitening is what moves contrastive off chance (0.81-0.86
  vs 1.00) and lifts deck AUC (0.66-0.70 vs 0.53-0.56); trainable
  pooling adds little on top. Unwhitened 4-query attention alone keeps
  numeric fields (R^2 0.57). Next: arms 5 and 8, then raw + whitened
  tokens (root `TODO.md`).

## From model_by_diet_r2 (concluded 2026-10-14)

[Notebook](2026-10-11_model_by_diet_r2/NOTEBOOK.md), Conclusions.
Premises for the next suite (one seed, so pilot strength):
- metric-only training on a trainable encoder collapses the embedding;
- contrastive is the pre-training direction, but it trades card-text
  fields for deck structure;
- no trained cell met the bar.

The reference encoder is the head-free frozen whitened mean. The trained
pilot default is `single_mixed`.

The goal (human, 2026-10-14): useful embeddings, judged by the
evaluation module, with transfer to unseen cards and games as a major
goal. Field recovery is a nice-to-have. **First decide which evaluation
metrics count as "useful"**: fields that bear on card use, not every
recoverable field.

Evaluation gaps to close first (minutes each, on existing checkpoints and
the feature cache):
- [ ] The frozen whitened mean as an encoder in the held-out and Gwent
  fresh heads and in the intrinsic suite. It was never measured there.
- [ ] Held-out fresh heads on control seeds 1-2 and 2+ head seeds, with
  paired differences against the control mean.
- [ ] A longer head budget for the deck dojos and the four dojos no
  encoder learned. About half the held-out heads peaked in the last of 6 rounds at 3,000 steps.

Proposed experiments:
- [ ] **A. A head that keeps the frozen features.**
  - First, an offline probe of untrained variants: a linear projection, a
    residual skip or a concatenation of the frozen feature, 512/768 wide,
    no output LayerNorm.
  - Then a contrastive pilot, with an anti-collapse term (VICReg, or
    uniformity) as an option.
  - The idea: the contrastive part learns deck structure, and the frozen
    part keeps the text.
  - A learned variant (human, 2026-10-14): an attention head over both
    the whitened text embedding and a contrastive-trained refinement of
    it, instead of a fixed skip or concatenation.
- [ ] **B. Curriculum (multi-stage diet).** Contrastive pre-training, then (i) metric heads on
  the frozen encoder, (ii) a full metric fine-tune, (iii) mixed from the
  start, at equal steps.
- [ ] **C. Why metric training collapses.** Smaller heads; per-card
  against run- and deck-level metric dojos, trained separately.
- [ ] **D. Train from cached frozen features**, for throughput. Check the
  results match an uncached run.
- [ ] **E. Seeds** of `single_contrastive`, `multi_contrastive` and
  `single_mixed`, before any model or diet ranking is trusted.
- [ ] Contrastive-only at 10k steps, to separate "metric hurts" from
  "mixed gets half the contrastive steps".
- [ ] Probe the per-round checkpoints: is the trade between deck AUC and
  linear-cls monotone over training?
- [ ] Centered-cosine deck AUC and nearest-centroid, to separate offset
  from collapse.
- [ ] How deck splits interact across dojos built from the same source
  (contrastive decks against FaB hero-from-deck VALIDATION decks), before
  deck-level results are read as transfer.

## Ablations, when a series raises the question

- [ ] Contrastive share of the mixed diet (e.g. 25 / 50 / 75%).
- [ ] Game balance: today's per-dojo diet vs. a per-game two-level diet
  (`src/training/TODO.md`).
- [ ] MSE vs. Huber regression loss (`src/training/TODO.md`).
- [ ] Staple subsampling in contrastive pairs, arms t = inf / 0.1 / 0.01
  (`src/training/TODO.md`).
- [ ] Curriculum: a contrastive phase, then a metric-dojo phase, against
  the mixed diet.

## Ideas, not yet scheduled

Each idea is marked config-only or code. When one is picked up, it
moves to an ablation above or to a series, and any code it needs moves
to the relevant `TODO.md`.

### What the runs so far show

From the first `multi_mixed` run (archived in
[2026-10-09_model_by_diet](2026-10-09_model_by_diet/NOTEBOOK.md)) and the
LayerNorm + `encoder_lr` pilot ([2026-10-10_lr_pilot](2026-10-10_lr_pilot/NOTEBOOK.md)):

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
- **...but one shared offset still dominates.** Frozen ModernBERT is
  anisotropic: mean-pooled features of cards from five games are at cos
  ~0.98. Once mean-centered they carry real structure (same-game cos
  0.31 vs cross-game -0.15), and by round 1 of the pilot the trained head
  has sharpened it (same-game 0.54). The multi-card attention stack then
  adds a large constant vector: final cos 0.999, with the card-specific
  part 0.40 of norm 16. Metric heads don't care about a constant offset;
  cosine-based contrastive losses do, and every contrastive dojo sits at
  chance. Details: [2026-10-10_lr_pilot](2026-10-10_lr_pilot/NOTEBOOK.md).
- **TEST is expensive.** About 14 of each round's ~37 minutes go to
  scoring all 162 dojos on 128 examples each.

### The shared offset and the contrastive dojos

- **Contrastive-only control first** (config-only). `multi_contrastive`
  with the LayerNorm, 2-3 rounds. If contrastive cannot leave chance
  even alone, mixing is not the cause; if it can, the mixed diet is
  starving it (20 steps per contrastive dojo per round today).
- **Center the frozen features** (code, small). ModernBERT is frozen, so
  its mean pooled vector over the card binder is a constant: compute it
  once and subtract it (optionally whiten, "all-but-the-top") before the
  head. Removes the anisotropy at the source. **Built** as token
  whitening (`model.enrichment`, before pooling) and measured in the
  pooling pilot; whitening the pooled vector instead is still open
  (`src/encoder_model/TODO.md`).
- **Center before the cosine** (code, small). Subtract the batch mean
  (or a running mean, as DINO does) from the embeddings inside the
  contrastive losses, so a shared offset added later (the attention
  stack's) cannot hide card differences from them.
- **Learnable contrastive temperature** (code, small). CLIP's
  `logit_scale`: the loss can sharpen itself when cosines are bunched.
- **Track geometry every round** (code, small). Log raw and centered
  cos, same-game vs. cross-game, over a fixed card set at each TEST pass
  (`2026-10-09_model_by_diet/analysis/embedding_scale.py` does it offline).

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

- **LR schedule** (config-only; built as a phase's `lr_schedule:`,
  `src/training/README.md`). Linear warmup over a few hundred steps, then
  cosine decay, is the default for transformers. It limits the two risky
  ends: large early steps through random heads, and thrashing late in the
  run. Not yet tried in a run.
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
  (`2026-10-09_model_by_diet/analysis/embedding_scale.py`). Token
  whitening is built; see "Center the frozen features" above.
- **Token-level head; thawing ModernBERT's top layers**: see
  `src/encoder_model/TODO.md` and series 3.

### Measuring training

- **Cheaper, less noisy TEST** (config-only). Use fewer, longer rounds
  (above), or raise `eval_examples_per_dojo` for the final round only.
  At 128 examples, one dojo's ratio can swing by tens of percent from
  sampling alone. Compare single rounds by medians and counts across
  dojos instead, as `2026-10-10_lr_pilot/analysis/compare.py` does.
