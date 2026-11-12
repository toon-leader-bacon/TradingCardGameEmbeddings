# Teacher-neighbour pairs against real-deck pairs for contrastive pre-training

**Status:** concluded
**Dates:** 2026-11-11 - 2026-11-12
**Parent:** [2026-10-11_model_by_diet_r2](../2026-10-11_model_by_diet_r2/NOTEBOOK.md) (contrastive as the pre-training direction); teacher smoke test in `plans/teacher_student_training.md` · **Branch:** `feature/teacher-cooccurrence`

> At an equal step budget, does a single-card encoder trained contrastively
> on teacher-neighbour pairs (a per-game PPMI-SVD co-occurrence teacher)
> learn a better card embedding, and learn it faster, than the same encoder
> trained on pairs drawn from real decks?

## Aim

**Hypothesis (priority: understanding; academic).** Replacing real-deck
pairs with teacher-neighbour pairs improves contrastive pre-training of
the single-card student, in quality and in speed.

**Primary metric:** same-deck retrieval AUC on a fixed set of VALIDATION
real decks, as the diet-weighted mean over the 15 trained games (the
weights in the diet table below). Δ = T − R, with a 95% bootstrap CI
over decks (per game, then combined with the same weights).

- **Quality** (final checkpoints), three outcomes:
  - **supported:** Δ ≥ +0.02, its CI excludes 0, **and** T's card-field
    probe (linear-cls) is no more than 0.02 below R's;
  - **refuted:** Δ ≤ −0.02 with its CI excluding 0, **or** T's
    linear-cls is more than 0.02 below R's;
  - **no clear difference:** anything else.
- **Speed** (matched-step checkpoints; both arms share the schedule
  shape and step count, so checkpoint N sits at the same point of each
  arm's warmup and decay), three outcomes:
  - **supported:** Δ ≥ +0.02 at every round checkpoint in the first half
    of the run;
  - **refuted:** Δ ≤ −0.02 at any round checkpoint in the first half;
  - **no clear difference:** anything else.
  - Descriptive, not a verdict: the step at which T first reaches R's
    final AUC. R's final checkpoint has decayed to 0.1× LR and T's
    midpoint has not, so this number is confounded by the schedule.
- 0.02 is carried over from r2 and has never been measured against seed
  noise. The bootstrap CI covers deck sampling, not training seeds.
  One verdict per part.

**What the treatment changes besides "the teacher".** The two arms
share the model, the loss (in-batch InfoNCE over 2-card units), batch,
augmentations, diet and budget. The pair source differs in three ways
at once, and this experiment does not separate them:
1. **Smoothing:** a pair is an anchor and one of its 32 nearest teacher
   neighbours, not two cards that happened to share one deck.
2. **Anchor distribution:** T draws anchors uniformly over confident
   cards (in at least 5 TRAIN decks); R draws decks uniformly, so cards
   appear in proportion to deck popularity and staples dominate.
3. **Reweighting:** PPMI down-weights staples' co-occurrence; real-deck
   sampling does not.
A verdict for T is a verdict for the bundle. The ablations that
separate the parts are under Open questions.

**Metric training is out of scope.** The direction (human, 2026-11-11)
is contrastive pre-training, then fine-tuning on metric dojos after.
Both arms train contrastive only.

## Background

- [model_by_diet_r2](../2026-10-11_model_by_diet_r2/NOTEBOOK.md)
  (one seed). For the single-card model, contrastive-only beat mixed on
  deck AUC (0.767 vs 0.711) with no clear linear-cls difference (0.615 vs
  0.617). Metric-only collapsed the embedding. Contrastive trades
  card-text fields for deck structure. The untrained 256-d head already
  loses most of the frozen feature's field information (linear-cls 0.863
  for the frozen whitened mean, against ~0.67).
- [lr_pilot](../2026-10-10_lr_pilot/NOTEBOOK.md) promoted `encoder_lr`
  1e-4 "until the offset itself is removed (then re-check 3e-4)". Token
  whitening removed it
  ([pooling_pilot](../2026-10-10_pooling_pilot/NOTEBOOK.md)). The
  re-check never ran.
- **Teacher smoke test** (2026-11-05, `plans/teacher_student_training.md`):
  MTG and FaB, 2 × 500 steps. TEST loss (chance 6.24): MTG 4.00 → 3.71,
  FaB 2.94 → 2.82, still falling. Gradients were clipped (max norm 1.0)
  on 494 of 500 steps in round 1, against 0 of 2,000 in a real-deck run.
- **Readiness work** (2026-11-11, `plans/teacher_full_test_readiness.md`):
  - teacher dojos for 15 games;
  - per-game augmentation masks measured from same-deck field sharing;
  - Archidekt, Moxfield and Deckstats decks merged into the MTG box,
    which lifted confident MTG TRAIN cards from 8,878 to 25,518.
- [throughput_shakeout](../2026-10-29_throughput_shakeout/NOTEBOOK.md):
  - one intra-op thread gives +34%, now the `torch_cpu_threads` default;
  - ~77 ms per 512-card step for real-deck contrastive;
  - bounded mod variants are encoded live (~200 texts/s) until they are
    cached.
- **Prior art for a content model trained toward co-occurrence vectors:**
  - van den Oord, Dieleman & Schrauwen (2013), *Deep content-based music
    recommendation*, NeurIPS: audio → collaborative-filtering latent
    factors, for cold-start items;
  - Levy & Goldberg (2014), *Neural word embedding as implicit matrix
    factorization*, NeurIPS: SGNS ≈ shifted-PMI factorization, which
    motivates PPMI-SVD as the teacher;
  - Barkan & Koenigstein (2016), *Item2Vec* (arXiv:1603.04259);
  - knowledge distillation: Hinton, Vinyals & Dean (2015),
    arXiv:1503.02531.

## Method

**Terms.**
- *Arm*: a pair source.
  - **R**: real decks, catalog keys `contrastive.<game>`.
  - **T**: teacher neighbours, `teacher_contrastive.<game>`.
- *Pilot run*: a short LR-pilot run.
- *Main run*: one arm's long run.
- *Control*: an encoder evaluated without training.

**Data.**
- 15 games, with Gwent held out as the unseen game:
  - MTG (7.15M decks), StS2, Hearthstone, FaB, Dominion, Pokémon,
    Yu-Gi-Oh;
  - One Piece, Digimon, Gundam, DB Fusion World, DB Masters, SWU, BSS
    and Lorcana (Limitless).
- `HoldoutSpec(seed=0, tier_ratios=(8, 1, 1), held_out_games={gwent})`.
- Deck snapshots as rebuilt on 2026-11-11 (MTG after the crawl ingest).
- Teacher: PPMI-SVD, dim 128, top-32 neighbours, `deck_size_rule:
  inverse_size`. Each deck's pairs share one unit, so MTG's Commander
  decks (3% of decks, most of the pair mass under `uniform`) weigh per
  deck, as R samples them.
- Default teacher views:
  - TRAIN: TRAIN decks and TRAIN cards;
  - TEST: TRAIN + TEST decks and TRAIN + TEST cards.
- VALIDATION decks and VALIDATION-tier cards are seen by neither arm.

**Model (held constant).** Single-card:
- frozen ModernBERT-base;
- token whitening, refitted for this experiment over the 15 games'
  TRAIN-tier cards (`data/runs/2026-11-11_teacher_vs_real_pairs/whitening_stats.pt`);
- mean pooling;
- `residual_mlp` head: 3 blocks, hidden 512, output 256, final
  LayerNorm.

Everything before the head is frozen, so the feature cache stores pooled
vectors (`PooledCut`): VRAM 3 GiB, RAM 4 GiB, a disk tier shared by both
arms.

**Held constant across arms.**
- seed 0;
- batch: 512 cards (256 two-card units; chance InfoNCE ≈ ln 511 = 6.24);
- fp16; `torch_cpu_threads` 1;
- game-default augmentations at `variants: 4`;
- `eval_examples_per_dojo` 128;
- equal steps (saturation effectively off: patience 1,000 rounds);
- LR schedule shape: warmup, then cosine decay to 0.1×.
- **Diet:** a per-game table, identical in both arms, weight =
  round(√ teacher TRAIN anchors):

  | game | weight | share |
  |---|---|---|
  | MTG | 160 | 22.7% |
  | Yu-Gi-Oh | 106 | 15.0% |
  | Pokémon | 78 | 11.0% |
  | Hearthstone | 66 | 9.3% |
  | FaB | 62 | 8.8% |
  | Digimon | 54 | 7.6% |
  | One Piece | 40 | 5.7% |
  | StS2 | 21 | 3.0% |
  | SWU | 20 | 2.8% |
  | DB Masters | 19 | 2.7% |
  | Gundam | 19 | 2.7% |
  | DB Fusion World | 18 | 2.5% |
  | Lorcana | 16 | 2.3% |
  | BSS | 16 | 2.3% |
  | Dominion | 12 | 1.7% |

  (Generated by `analysis/make_configs.py`.)

**Cross-scoring.** Each arm lists the other arm's 15 dojos as held-out
dojos. Contrastive dojos are head-free (`src/dojos/README.md`), so every
round scores both tasks on both arms without training the other. That
gives learning curves on both tasks for free. The arms' own TEST losses
are not comparable to each other (different tasks). Only the shared
held-out scores and the offline evaluation are.

**Step 1: LR pilot.** Each arm at `encoder_lr` 1e-4, 3e-4 and 1e-3:
3 × 1,000 steps, 500 warmup steps, cosine decay over the pilot (6
runs). `head_lr` 1e-3 is inert, since no trained dojo has a head.
- **Rule, fixed now:** each arm takes the LR with the lowest median
  own-task normalized TEST loss at the end of round 3. The median is
  over the 7 games with at least 128 T TEST anchors (MTG, Yu-Gi-Oh,
  Pokémon, Hearthstone, FaB, Digimon, One Piece), for both arms. The
  small games have 15-55 TEST anchors, too few for a stable score.
- If two LRs are within 2% of each other, take the larger. Cosine decay
  of a long run makes a larger peak LR safer than in a short pilot.
- The pilot also measures ms/step per arm, which sets the main budget.
  Only the last pilot runs' final rounds are used, since the first runs
  pay for live variant encoding.

**Step 2: main runs.**
- R then T, 10,000-step rounds, a checkpoint every round.
- `max_rounds` chosen after the pilot so that the slower arm fits about
  4.5 h; both arms get the same number of steps.
- 2,000 warmup steps, then cosine decay over the whole run.

**Step 3: offline evaluation.** The same for every encoder: **every**
round checkpoint of both arms (the speed verdict needs matched steps),
plus the controls. The script (`analysis/evaluate.py`) is written and
smoke-tested on the pilot checkpoints **before** the main runs. Its
VALIDATION deck sample per game is fixed once (seeded) and shared by
every encoder.
1. **Primary:** same-deck retrieval AUC on VALIDATION decks, per game
   and as the diet-weighted mean, with bootstrap CIs. 0.5 = chance.
   The unweighted mean is also reported.
2. **Card-field probe (linear-cls)** as in r2: the text/deck trade.
3. **Gwent:** zero-shot same-deck AUC and kNN, plus fresh heads on its
   dojos (transfer, the project's main goal).
4. **VALIDATION-tier cards** (trained on by neither): deck AUC over
   VALIDATION decks restricted to them, for cold start.
5. **Collapse checks:** effective rank and mean pairwise cosine.

**Controls.**
- The untrained student (seed 0 initialization).
- The head-free frozen whitened mean (768-d), r2's reference encoder.
- r2's `single_contrastive` final checkpoint, as a historical point only.
  Its data, whitening and head depth (9 blocks) differ, so it is not
  a controlled comparison.

**Predictions (before any run).**
- R and T both beat the untrained student on deck AUC by far more than
  0.02.
- **If T wins on quality,** expect the per-game margin to grow with
  staple concentration: the share of a game's TRAIN deck slots taken by
  its top 1% of cards, computed from the snapshots by the evaluation
  script. Where real-deck pairs are mostly staples, PPMI's weighting
  against chance helps most. Checked by the rank correlation across the
  15 games.
- **If T loses on quality,** the likely reason is that 32 neighbours
  are a narrower notion of "goes together" than whole decks, and R's
  pair distribution matches the evaluation.
- **Speed:** T's curve should rise faster early, because neighbour
  pairs carry less noise per pair than two random cards from one deck.
- **Field probe:** both arms lose against the frozen mean, as in r2. T
  may lose less, since uniform anchors give rare cards more steps.
- **Gwent:** zero-shot near chance for both, as in r2 (0.99-1.02 on the
  contrastive score).
- R's TEST loss on T's dojos falls over training. That would show that
  real-deck training learns the teacher's neighbourhoods too.

**Known confounds.**
- One seed; the pilot is one seed per LR.
- **Anchor coverage:** R trains on every card in its decks, T only on
  confident cards (MTG 25,518 of 27,759 TRAIN cards).
- The three-part bundle under Aim.
- **Wall-clock and run order:** the second arm inherits the first arm's
  cached variant texts. Speed is therefore judged in steps, and
  wall-clock is reported only for steady-state rounds.
- **Repeats on small games:** at ~150k steps, Dominion's 144 T anchors
  are each drawn thousands of times. Per-game results for small games
  may show overfitting in T that R (531k Dominion decks) does not.
- **The evaluation favours R:** VALIDATION-deck AUC is R's own training
  task on unseen decks.
- **LR chosen per arm** (human, 2026-11-11), so each method runs at its
  own best setting.
- **Gradient clipping.** In the smoke test T clipped at max norm 1.0 on
  494 of 500 steps, against 0 of 2,000 for R. When most steps clip, the
  update size is set by the clip, not the LR. Under Adam the effect
  should be small (it normalizes per weight), but it bears on speed.
  The clip fraction is logged per round for both arms and reported
  next to the curves.
- **Small-game TEST:** T's TEST anchors for the small games number
  15-55, below `eval_examples_per_dojo` 128, so their training-time TEST
  losses are noisy. The verdicts use the offline evaluation, not
  training-time TEST.

**Cost.**
- Prep (whitening, teacher rebuild with `inverse_size`, MTG mask
  re-measure, `--check`): ~1 h.
- LR pilot: ~1 h.
- Main runs: ~9 h.
- Offline evaluation: ~1 h.
- **Total ≈ 12 h**, under the 24 h gate.
- Disk: ~1.2 GB per checkpoint (including frozen ModernBERT), up to ~16
  per main run, so ~40 GB in all; G: has 926 GB free.

## Recipe

ROCm venv (`G:/Projects/venvs/tcg-rocm/Scripts/python.exe`), project
root, `PYTHONPATH=.`.

1. Data stages: card binders, deck boxes and Arrow snapshots for the 15
   games (`scripts/build_training_snapshots.py --all`).
2. Configs: `python experiments/2026-11-11_teacher_vs_real_pairs/analysis/make_configs.py pilot`
   (and `main --lr-r … --lr-t … --max-rounds …` after the pilot).
3. Whitening: `scripts/fit_whitening.py --holdout-from <a pilot config> --games <the 15> --out data/runs/2026-11-11_teacher_vs_real_pairs/whitening_stats.pt`.
4. Teachers: `scripts/build_teachers.py --config <a pilot config>`.
5. Each run: `scripts/run_training.py <config>`; check first with `--check`.
6. Evaluation: `analysis/evaluate.py` (every round encoder plus the
   controls; ~35 min frozen-feature pass once, then ~5 min per encoder),
   then `analysis/round_curves.py` for `results/round_compare.csv` and
   the figure. `analysis/compare.py <A> <B>` gives one paired comparison.
7. Probes A, B, C (added after the main runs): `analysis/teacher_probes.py`,
   ~45 min on the GPU.
8. Probes C2 and E (offline distillation on cached features):
   `analysis/distill_probes.py [--only c2|e] [--arms …]`, seconds to
   ~10 min per arm.
9. Real-trainer ablations: `configs/ablation_T_{noaug,noclip}.yaml` with
   step 5, stopped once `round0000/` is written, then step 6.

Outputs: `data/runs/2026-11-11_teacher_vs_real_pairs/<run>/`; evaluations
and probe logs under `data/evaluations/2026-11-11_teacher_vs_real_pairs/`.

## Log

### 2026-11-11 · Design (PI, with the human)

What: aim, arms, diet, LR pilot, evaluation and budget, as above.

Why:
- The human's question: does the teacher-student regime improve speed
  and/or quality over the earlier regime?
- Framed as a pair-source comparison on the same single-card model.
  The teacher is not a second model tower, only a source of pairs.

Human decisions:
1. Contrastive only, both arms: pre-training first, metric fine-tuning
   later.
2. Each arm at its own best LR from the pilot.
3. `inverse_size` teacher, so both arms weigh MTG decks equally.
4. One seed, long runs.

### 2026-11-11 · Design review (experiment-critic) and changes

Five concerns, each answered in the Aim and Method above:
1. **Speed tilted by the cosine schedule:** speed is now judged at
   matched-step checkpoints, and every round checkpoint is evaluated.
   "Steps to reach R's final" is kept as a descriptive number only.
2. **Quality outcomes overlapped:** now three outcomes (supported, no
   clear difference, refuted), with a bootstrap CI.
3. **No noise estimate for 0.02, and the evaluation unwritten:** the
   primary is now a diet-weighted mean with bootstrap CIs over a fixed
   VALIDATION deck sample. The script is written and smoke-tested on
   pilot checkpoints before the main runs. Seed noise is still
   unmeasured and is stated as such.
4. **Clipping as a confound:** declared, and logged per round.
5. **Small-game TEST noise in the pilot rule:** the rule now uses the 7
   games with at least 128 TEST anchors.

Minor points:
- the archetype prediction is now operational (staple concentration);
- `staple_subsampling` was added to Open questions;
- the pilot's first runs are excluded from step-time sizing.

### 2026-11-11 12:00 · Prep: whitening, teachers, MTG field sharing (PI)

What:
- Committed the readiness work and this design (a07c936, deb6939).
- Refitted token whitening over the 15 games' TRAIN tier (13,873
  cards, 768 → 768; `data/runs/2026-11-11_teacher_vs_real_pairs/fit_whitening.log`).
- Rebuilt every teacher with `deck_size_rule: inverse_size` (both
  views; confident TRAIN anchors unchanged, since deck counts come from
  presence; `build_teachers.log`).
- Measured MTG field sharing in both arms' pair distributions, to check
  the MTG masks after the crawl merge (plan steps 2.6-2.7).

Seen: share of pairs with equal values (lift over random pairs).

| field | R: same real deck | T `uniform` teacher | T `inverse_size` teacher |
|---|---|---|---|
| set | 60% (17.8×) | 36% (64×) | 36% (59×) |
| color_identity | 31% (2.4×) | 48% (4.4×) | 50% (4.5×) |
| colors | 22% (2.5×) | 44% (4.2×) | 46% (4.4×) |
| legalities | 14% (2.7×) | 13% (6.8×) | 14% (6.9×) |
| keywords | 1% (4.5×) | 5% (11×) | 4% (9×) |

- R: 19,975 random decks (scratch `field_sharing.py`).
- T: one uniform neighbour per confident TRAIN anchor, 4 draws each
  (`analysis/neighbour_field_sharing.py`).

Decided:
- `inverse_size` barely moves the neighbours. PPMI already discounts
  the Commander pair mass. It is kept, so both arms weigh MTG per deck.
- **The MTG masks stay unchanged.** They are identical in both arms, so
  they are not an arm confound. `set` is the artefact (17lands draft
  pools) and is masked half the time. Colour sharing is higher for T,
  but colour is a real reason cards go together, not an artefact.
  Recorded here because T's colour lift (4.5×) means T may lean on
  colour more than R does. The intrinsic suite's colour-label agreement
  will show it.

Deferred to later experiments:
- head architecture (TODO experiment A: keep the frozen features);
- thawing ModernBERT (series 3);
- metric fine-tuning (experiment B).

### 2026-11-11 14:15 · LR pilot and evaluation dry run (PI)

What:
- Ran the six-run LR pilot (`analysis/run_pilot.sh`, 11:51-13:30, all
  exit 0), 3 × 1,000 steps per run.
- Applied the pick rule (`analysis/pick_lr.py` → `results/lr_pilot.csv`).
- Dry-ran `analysis/evaluate.py` and the new `analysis/compare.py` on
  the two 3e-4 pilot checkpoints and the two controls, to test the
  evaluation code before the main runs.

Seen, pick rule: median own-task normalized TEST loss over the 7 big
games, last round (other arm's task in brackets).

| encoder_lr | R | T | ms/step R / T |
|---|---|---|---|
| 1e-4 | 0.882 (0.851) | 0.753 (0.965) | 159 / 185 |
| **3e-4** | **0.855** (0.832) | **0.690** (0.964) | 151 / 185 |
| 1e-3 | 0.892 (0.875) | 0.731 (0.984) | 151 / 183 |

- Both arms pick **3e-4**. It is the clear minimum in both, so the
  tie rule did not apply.
- Cross-task losses are asymmetric: R training lowers T's loss (0.83),
  but T training barely lowers R's (0.96).

Seen, evaluation dry run (`results/eval_summary.csv`; pilot checkpoints
after only 3k steps, so not a result):

| encoder | weighted deck AUC | Gwent | linear-cls | eff. rank | mean cos |
|---|---|---|---|---|---|
| frozen whitened mean | 0.687 | 0.633 | 0.927 | 272 | 0.02 |
| untrained head | 0.640 | 0.586 | 0.739 | 119 | 0.52 |
| R pilot 3e-4 | 0.796 | 0.600 | 0.727 | 82 | 0.54 |
| T pilot 3e-4 | 0.821 | 0.609 | 0.759 | 66 | 0.39 |

- `compare.py` gives T − R = +0.025 [+0.023, +0.026] weighted deck
  AUC, and +0.032 linear-cls. By game: T leads on Dominion (+0.16),
  Pokémon (+0.09), MTG (+0.06) and Hearthstone (+0.05). R leads on
  SWU (−0.04), Lorcana (−0.03), Digimon (−0.02) and FaB (−0.02).
- The evaluation works end to end. The cached head path matches the
  real model (min cosine 0.9998). Each encoder takes about 4 min after
  a one-off ~35 min frozen-feature pass over 102,479 cards.

Interpretation, provisional:
- The deck-AUC ranking runs opposite to the cross-task losses. R's pairs
  are what the deck AUC asks about, yet T scores higher there. Training
  losses on the other arm's pairs are not the measure for the Aim; the
  deck AUC is.
- **The bootstrap CI covers deck sampling only.** With one seed it is
  much narrower than the run-to-run spread would be. That is the
  declared one-seed confound. A Δ near the ±0.02 margin should be read
  with it in mind. Ask the human whether a second seed of the final
  checkpoints is worth it if the main Δ lands close.
- The frozen whitened mean keeps far more field information
  (linear-cls 0.93) than any head (0.73-0.76), with 3-4× the effective
  rank. That supports TODO experiment A, a head that keeps the frozen
  features. Out of scope here.

Decided:
- Main configs: `analysis/make_configs.py main --lr-r 3e-4 --lr-t 3e-4
  --max-rounds 8`. That is 8 × 10k steps, warmup 2,000, cosine. At the
  pilot's ms/step it comes to about 3.4 h for R and 4.1 h for T,
  including per-round TEST.
- The pilot eval rows stay in `results/` under `pilot*` labels. The
  controls (`frozen_mean`, `untrained`) are final and are not re-run.

### 2026-11-11 17:44 · Main queue launched early (PI)

- The human halted the MTG crawls at 17:38 to free the machine. They
  had been at about 1 deck/min since 09:36: IPv6 is dropped on this
  network, fixed with `run_data_retrieval.py --ipv4` (92c6d97). So the
  main runs start now instead of after 21:30. The machine is idle for
  both arms: no crawl load in either, which removes one wall-clock
  difference between them.
- Handed to `lab_technician`: main_R, then main_T, at commit 92c6d97
  (clean tree). Pre-flight: 21.7 GB RAM free, 17.0 GB VRAM free, 910 GB
  disk; `--check` passed for both main configs. Queue:
  `data/runs/2026-11-11_teacher_vs_real_pairs/queue.csv`. Estimated
  ~3.4 h + ~4.1 h, so done around 01:30.

- 2026-11-11 19:50 · lab: incident, main_R kept only its best checkpoint; added `keep_round_encoders`, retrying main_R ([bench note](bench_notes/2026-11-11_1950_incident_round_checkpoints_pruned.md))

- 2026-11-11 21:50 · lab: main_R done, 118 min, all 8 round encoders kept; main_T launched ([bench note](bench_notes/2026-11-11_2150_run_main_R_done.md))

- 2026-11-11 23:45 · lab: main_T done, 114 min, all 8 round encoders kept; evaluation of all round encoders started ([bench note](bench_notes/2026-11-11_2345_run_main_T_done.md))

- 2026-11-12 01:25 · lab: wrap-up; queue done, all round encoders evaluated, awaiting review ([bench note](bench_notes/2026-11-12_0125_wrap_up.md))

### 2026-11-12 · First reading of the results, with the human (PI)

Results as the Aim's rules give them (`results/round_compare.csv`,
`results/deck_auc_by_round.png`): speed "no clear difference" (T ahead
by +0.010 to +0.015 in every first-half round, under the +0.02 bar);
quality "refuted" (final deck AUC delta -0.003 [-0.004, -0.002], inside
the margin, but T's linear-cls is 0.108 below R's against an allowed
0.02). Conclusions are not written yet: the probes below come first.

**The human's reading** (recorded as said, not yet tested):
- *Training removes field information.* Frozen whitened mean 0.927
  linear-cls, untrained head 0.739, R about 0.70 throughout, T falling
  from 0.713 to 0.591. Two causes are possible, and both may hold:
  1. the input is a "muddled" mean-pooled text vector, so little field
     detail reaches the head in a usable form (although the untrained
     head still keeps 0.739);
  2. the contrastive-only diet. Even with token embeddings as input, a
     diet that only asks "are these cards deck partners" would likely
     lose field detail too. That suggests the exact keywords, cost and
     other fields matter little for deck partnership.
- *Two remedies for the second cause:* a mixed diet from the start, or
  post-training on these embeddings to bring attention back to the card
  fields (the original plan's "contrastive pre-training, metric
  fine-tuning").
- *Collapse may be a feature.* One reading is that training compresses
  card-field information efficiently and keeps only what neighbour
  matching needs, which may be useful in general. Open until a
  downstream task says otherwise.
- *Next direction for the encoder:* stop treating the text side as
  fixed. Fine-tune the underlying BERT, try other text embedding models,
  and drop pooling and whitening in favour of the raw token embeddings.
- *Keep in mind* (from the PI's list, the human's pick as most
  important): R trains on real decks and is scored on real decks. T
  also learns from real decks, but indirectly: through co-occurrence
  counts, PPMI-SVD vectors and top-k neighbours. R has the home
  advantage, and given enough steps it should be expected to win.

**PI hypotheses for why T was not faster** (for the probes to test):
1. The student never sees the teacher's vectors, only top-32 neighbour
   pairs. Distances and everything past rank 32 are thrown away.
2. The input bottleneck: no head over one pooled text vector can
   separate cards whose text is near-identical but whose deck roles
   differ.
3. T overfits a fixed target: its own TEST loss bottoms at round 3 and
   then rises, and its effective rank falls from 53 to 33.
4. R's home advantage, as above.

The current logs cannot tell under-fitting from over-fitting: the
trainer records TEST loss per round, not train loss.

**Probes chosen** (the human: A, B, C, in that order; D, AUC split by
staples and near-identical text, deferred). None of them trains
anything: they use the frozen features, the round encoders and the
teacher tables already on disk. Script:
`analysis/teacher_probes.py`; output `results/probe_*.csv`.

- **A. The teacher's ceiling.** Score each game's teacher table (the
  TRAIN-tier table main_T trained on, fitted on TRAIN decks only, so
  the VALIDATION decks are clean) with the same deck AUC, decks and diet
  weights. A teacher table covers only cards seen in TRAIN decks, so
  every encoder in this probe is re-scored on the teacher's cards only:
  teacher, R_r7, T_r7, R_r1, T_r1 and the frozen mean. Gwent has no
  teacher and is left out.
  - Teacher at 0.92 or more: the teacher knows much more than the
    student learned, which points at hypotheses 1 and 2.
  - Teacher at 0.88 or less: the teacher is near the student's level
    and is a ceiling for teacher-pair training.
  - In between: both are partly true.
- **B. TRAIN against VALIDATION decks.** The same encoders scored on
  500 TRAIN decks per game, sampled the same way. TRAIN minus VALIDATION
  is the generalisation gap.
  - Gap under 0.01 for R_r7: R is not over-fitting its decks. The limit
    is capacity or the input, and more steps or more data of the same
    kind will not help much.
  - Gap over 0.03: over-fitting. Regularisation or more decks would
    help, not a bigger model.
  - The teacher's own gap shows how much of its score comes from having
    counted those exact decks.
- **C. Can pooled text predict the teacher?** For each game, regress
  the teacher vectors on the frozen whitened mean features with ridge
  and with a small MLP the size of the student head. Five folds by
  card, so every card gets a prediction from a model that never saw it.
  Report held-out R² and cosine, and score the out-of-fold predictions
  with the A deck AUC.
  - Deck AUC of the predictions close to the teacher's: pooled text
    carries what the teacher knows, so the training signal (hypothesis
    1) is the bottleneck, not the input.
  - Close to R_r7 or below: pooled text cannot reach the teacher,
    whatever the head (hypothesis 2). That supports moving to token
    embeddings or a different text encoder.
  - Ridge against MLP shows whether head capacity matters at all.

### 2026-11-12 09:10 · Probes A, B, C results (PI)

`analysis/teacher_probes.py`, 45 min on the GPU; log at
`data/evaluations/2026-11-11_teacher_vs_real_pairs/probes/probes.log`.
Tables: `results/probe_deck_auc.csv`, `results/probe_text_to_teacher.csv`.
Every encoder is scored on the teacher's cards only: 81,068 of 101,219
cards in the 15 games, 23,586 deck cards skipped per split. So R_r7
scores 0.892 here, not the 0.867 of the full-card evaluation.

Two fixes before the run that counted, neither changing a number: the
cache check used a different 64-card sample and missed the 0.999 bar
(0.9979, fp16 noise), so it now uses evaluate.py's own sample (0.99968).
The MLP's R² was computed on raw outputs, which is meaningless for a
model trained on cosine, so R² is now on unit-length predictions.

Diet-weighted deck AUC, teacher cards only:

| encoder | VALIDATION decks (5,339) | TRAIN decks (7,287) | TRAIN - VALIDATION |
|---|---|---|---|
| teacher | 0.9444 | 0.9506 | +0.006 |
| R_r7 | 0.8924 | 0.8942 | +0.002 |
| T_r7 | 0.8866 | 0.8875 | +0.001 |
| T_r1 | 0.8750 | 0.8762 | +0.001 |
| R_r1 | 0.8664 | 0.8660 | -0.000 |
| text_ridge (out of fold) | 0.8326 | 0.8318 | -0.001 |
| text_mlp (out of fold) | 0.8317 | 0.8312 | -0.001 |
| frozen_mean | 0.6868 | 0.6861 | -0.001 |

Against the predictions written above:
- **A:** the teacher is at 0.944, above the 0.92 line. The teacher knows
  much more than either student learned: 0.052 above R_r7. That points
  at hypotheses 1 and 2. The gap is widest on Star Wars Unlimited
  (0.933 against 0.827), Pokemon, MTG, Lorcana and Gundam, and nearly
  zero on Slay the Spire 2 and Dragon Ball Masters.
- **B:** R_r7's gap is +0.002, under the 0.01 line. No student
  over-fits its decks, at any round. So hypothesis 3, as worded about
  decks, is not supported: T's rising TEST loss is on its own
  pseudo-pairs and does not show up as a deck-level gap. The teacher's
  own gap is +0.006, so its 0.944 is not from having counted those
  exact decks.
- **C:** text predictions for cards the regressor never saw reach 0.832,
  below R_r7 (0.892) and far below the teacher (0.944). Ridge and MLP
  are equal on deck AUC (0.833 against 0.832), although the MLP's
  held-out cosine is higher on the big games (MTG 0.71 against 0.53).
  Held-out cosine falls to 0.3-0.45 on the small games. By the rule
  written above this reads as "pooled text cannot reach the teacher".
  **Caveat, found when reading:** C asks for unseen cards, while the
  students are scored on cards they trained on (TRAIN-tier cards), so
  C is a harder test than A's student rows. C shows that pooled text
  generalises the teacher's geometry to new cards poorly. It does not
  yet show whether a head over pooled text can *fit* the teacher on
  cards it trains on. That is the capacity question, and it needs an
  in-sample fit (proposed as C2 below).
- **Not predicted:** R_r7 scores 0.430 on Dominion, below chance and
  below the frozen mean (0.613). T_r7 scores 0.645. R's training made
  Dominion worse than no training. Its small diet weight (12) means this
  barely moves the weighted score.

**Proposed next probe, C2 (not run):** fit the same MLP and a wider one
on all confident cards and score the in-sample predictions. If they
reach about 0.94, pooled text plus a head of this size can hold the
teacher's knowledge, and the limit is the training signal (hypothesis
1). If they stop near 0.89, the input or the head size is the limit.

### 2026-11-12 · Discussion: rotation, learning rate, schedule; probes C2 and E (PI, with the human)

**Why pseudo-decks, and would direct distillation break?** The human's
reason for pseudo-decks: each game's teacher space is fixed only up to
a rotation, and the spaces of different games have no relation to one
another. Their analogy is per-language word vectors: "cat" and "chat"
have no shared coordinates, but similar neighbourhoods. The PI's
answer: regressing one shared student onto every game's *coordinates*
would be a problem. It would spend capacity on arbitrary rotations and
block cross-game sharing. A *rotation-invariant* signal does not have
that problem, and neither does a per-game linear adapter. Examples of a
rotation-invariant signal: matching the teacher's within-game
similarities, as in relational knowledge distillation (Park et al.
2019, arXiv:1904.05068) and similarity-preserving distillation (Tung &
Mori 2019, arXiv:1907.09682). On languages: Mikolov et al. 2013
(arXiv:1309.4168) and Conneau et al. 2018 (MUSE, arXiv:1710.04087)
found that separately trained word spaces differ by a near-orthogonal
map. Pseudo-decks are also rotation-invariant, but they reduce the
teacher to a yes/no top-32 signal.

**Learning rate and length.** The LR pilot found 1e-3 worse than 3e-4
in both arms, so a higher rate is unlikely to help. The schedule is
warm-up then cosine to a 0.1 floor at step 80k; the PI's earlier "near
zero" was wrong. Decaying is standard and usually helps. The problem is
that the schedule is tied to the planned length, so late flattening
cannot be told apart from the decay. Proposal for later experiments:
warmup-stable-decay (Hu et al. 2024, MiniCPM, arXiv:2404.06395), which
needs a small addition to `src/training/learning_rate_schedule.py`.
Probe B shows no over-fitting, so longer training is safe, but likely
worth about +0.01 next to the teacher's 0.05 lead. Each arm took about
2 h of a 12 h budget.

**The human asked to keep probing.** Two probes, both offline on the
cached frozen features (no augmentation, no text encoder); script
`analysis/distill_probes.py`:

- **C2. In-sample capacity, per game.** For each game, fit
  `ResidualMlpEmbeddingHead` (the student's head class) to the unit
  teacher vectors (1 - cosine) on all confident cards, for a fixed 4,000
  steps with no held-out data. Widths 512 (the student's), 1024 and
  2048, with 3 blocks each. Score the predictions on the probe A deck
  AUC (VALIDATION decks, teacher cards).
  - Width 512 already at 0.93 or more: the head and the pooled input
    can hold the teacher's knowledge of seen cards, so size is not the
    limit.
  - Rising with width: size matters.
  - Every width at 0.90 or less: the pooled input is the limit, even
    for seen cards.
- **E. The training signal, isolated.** One shared head across the 15
  games, as in the student: `ResidualMlpEmbeddingHead(768, 256)`. Same
  optimiser and schedule as the main runs: AdamW at 3e-4, 2,000 warm-up
  steps, cosine to 0.1 at 80k steps. Same batches too. Each step picks a
  game by the diet weights, then 64 confident anchors and 7 of each
  anchor's top-32 teacher neighbours, 512 cards in all. Only the target
  differs:
  - **E_pairs** (the current signal): for each card, a uniform target
    over the batch cards in its teacher top-32.
  - **E_kl** (similarity distillation): a softmax over the teacher's
    cosines to the batch cards. Both use temperature 0.05, the
    student's softmax over its own cosines, a KL loss, and self and
    same-card pairs masked out.
  - **E_kl_wide:** E_kl with hidden width 2048.

  Each is scored with evaluate.py's full evaluation (all cards, so
  directly comparable to `results/round_compare.csv`) at 10k, 20k, 40k
  and 80k steps.
  - E_pairs within about 0.01 of T at matched steps: the offline
    replica is faithful enough to stand in for T. If not, the replica's
    differences (no augmentation, other batches) matter, and E is read
    only within itself.
  - E_kl at least 0.02 above E_pairs at 80k, and above R_r7's 0.867:
    the yes/no signal is what held T back, and similarity distillation
    is the next arm to test properly.
  - E_kl_wide at least 0.01 above E_kl: head size matters for the
    shared student.
  - Watch linear-cls: does a graded target keep more card-field
    information than a yes/no one?

### 2026-11-12 13:00 · Probes C2 and E results; replica and width control added (PI)

`analysis/distill_probes.py`; logs `data/evaluations/2026-11-11_teacher_vs_real_pairs/probes/distill*.log`.
Tables: `results/probe_c2_capacity.csv`, `results/probe_e_curves.csv`,
E rows (`E_*_s<k>k`) in `results/eval_summary.csv`, and
`results/probe_e_covered.csv` for the teacher-card view.

**C2** (per game, in-sample, VALIDATION decks, teacher cards):

| width | parameters per game head | in-sample cosine | deck AUC |
|---|---|---|---|
| 512 | 1.25M | 0.992 | 0.9191 |
| 1024 | 4.07M | 0.993 | 0.9195 |
| 2048 | 14.4M | 0.996 | 0.9207 |

Against the prediction: width 512 already memorises the teacher on the
cards it fits (cosine 0.99), and width adds nothing, so a per-game head
of the student's size is large enough. The deck AUC stops at 0.92, not
0.944. A likely reason, not checked: the cards the teacher holds but
marks as not confident (seen in fewer than 5 decks) were not fitted, so
their predictions are generalisation, as in C.

**E** (one shared head across the 15 games; all cards, so comparable
to `results/round_compare.csv`). Diet-weighted deck AUC, with seconds of
training time to reach each step count:

| arm | 10k | 20k | 40k | 80k | linear-cls at 80k | rank at 80k |
|---|---|---|---|---|---|---|
| main_T (real run) | 0.842 | 0.860 | 0.866 | 0.864 | 0.591 | 33 |
| main_R (real run) | 0.828 | 0.846 | 0.860 | 0.867 | 0.700 | 88 |
| E_t_replica (main_T's batch shape) | 0.867 (25 s) | 0.870 | 0.870 | 0.869 (192 s) | 0.515 | 30 |
| E_pairs | 0.873 (63 s) | 0.875 | 0.871 | 0.871 (500 s) | 0.557 | 41 |
| E_kl | 0.868 (24 s) | 0.865 | 0.873 | 0.874 (189 s) | 0.588 | 40 |
| E_kl_wide (width 2048) | 0.879 (68 s) | 0.887 | 0.892 | **0.897** (543 s) | 0.604 | 85 |

Against the predictions:
- **Replica faithfulness.** E_pairs did not stay within 0.01 of T: it
  was 0.031 ahead at 10k. So **E_t_replica** was added: main_T's batch
  shape exactly (256 random anchor-neighbour pairs, one positive each,
  other neighbours as negatives, temperature 0.07).
  - It matches T's end point: 0.869 against 0.864 (teacher cards: 0.891
    against 0.887).
  - It reproduces T's collapse: linear-cls 0.515, rank 30.
  - It is faster early: 0.867 against 0.842 at 10k.

  So the offline setup reproduces T where it ends. The real pipeline
  costs early speed, not the end point. (The PI's first remark to the
  human, "something in the real pipeline is holding training back",
  was too strong for the end point.)
- **KL against pairs at width 512:** 0.874 against 0.871 at 80k, under
  the 0.02 line. At the student's size the graded target is not better
  on deck AUC. It keeps a little more field information (linear-cls
  0.588 against 0.557).
- **Width:** E_kl_wide is 0.023 above E_kl and still rising at 80k, and
  its rank stays at 85. It is the first encoder in this experiment
  above R's 0.867 by a clear margin. The width control **E_pairs_wide**
  (pairs target, width 2048) is queued, to tell whether this is width
  alone or width with the KL target.
- **Collapse comes from the signal, not the pipeline.** The replica, fed
  only teacher pairs, loses field information and rank just as T did.
  At width 512, every teacher-signal arm ends with linear-cls of
  0.51-0.59.
- **Throughput:** offline, 10k steps take 25-70 s. The real runs took
  about 10-25 min per 10k steps, at about 99% feature-cache hits. That
  is an engineering question, not part of this experiment's aim (see
  the hardware-utilisation memory: throughput work goes through
  architect).

**Ablations in the real trainer, queued** (configs
`configs/ablation_T_{noaug,noclip}.yaml`, commit 5c5cb39). Each is
main_T with one change, run until its round-0 encoder (10k steps) is
written and then stopped. The 80k cosine plan is kept, so the learning
rate at 10k matches main_T's round 0. Scored with evaluate.py.
- `ablation_T_noaug`: `default_augmentation.probability: 0.0`. The
  default masks the field that nearly names the deck (set, class,
  faction) about half the time. The offline features are never masked.
- `ablation_T_noclip`: `max_grad_norm: 1e6`. main_T clipped 15-35% of
  steps (`run.log` clip tallies). The replica never clips.

Predictions, against main_T round 0 (0.842) and the replica (0.867):
- An ablation at 0.86 or more explains most of the early lag.
- Both near 0.842: the lag is elsewhere: fp16 autocast, the per-dojo
  loss weighting, or the pseudo-deck dealer.
- A caveat on augmentation, written now: masking the set or class field
  removes a shortcut that deck AUC rewards (two cards of one MTG deck
  share a set 62% of the time). A faster no-augmentation run would show
  the cost of the masking, not that the masking is wrong. Whether that
  shortcut is wanted is the human's call, and it depends on the
  transfer goal for unseen games.

### 2026-11-12 15:10 · Width control and real-trainer ablations: results (PI)

**Width control, E_pairs_wide** (pairs target, width 2048): 0.879,
0.883, 0.885 and 0.874 at 10k, 20k, 40k and 80k (all cards). It peaks
mid-run and falls, as main_T did. Teacher cards at 80k: 0.893.

| at 80k, all cards (teacher cards) | width 512 | width 2048 |
|---|---|---|
| pairs target | 0.871 (0.897) | 0.874 (0.893) |
| KL target | 0.874 (0.900) | **0.897 (0.914)** |

Neither width nor the graded target helps alone (about +0.003 each);
together they give +0.026, and E_kl_wide is still rising at 80k. A
reading, not tested: a yes/no target has little to fit, so a big head
over-fits it, while a graded target has enough structure to use the
extra capacity. On teacher cards, E_kl_wide at 0.914 closes about 40%
of the gap between R_r7 (0.892) and the teacher (0.944), with C2's
per-game in-sample fit at 0.92 as a reference.

**Real-trainer ablations** (inline: about 8 and 31 min of training;
queue script in the session scratchpad; console logs
`data/runs/2026-11-11_teacher_vs_real_pairs/logs/ablation_T_*.log`;
evaluation log `data/evaluations/2026-11-11_teacher_vs_real_pairs/ablation_eval.log`,
cache checks 0.99968 and 0.99981; rows `abl_T_*_r0` in
`results/eval_summary.csv`):

| round 0 (10k steps) | deck AUC | linear-cls | rank | clipped steps | wall-clock |
|---|---|---|---|---|---|
| main_T | 0.8417 | 0.713 | 53 | 2,111 | ~30 min |
| ablation_T_noclip | 0.8442 | 0.723 | 53 | 0 | ~31 min |
| ablation_T_noaug | **0.8666** | 0.680 | 47 | 535 | **~7.5 min** |
| E_t_replica, 10k (offline) | 0.8672 | 0.672 | 47 | none | 25 s |

Against the predictions: **augmentation explains all of the early lag**
(0.867, matching the replica to 0.001). Clipping explains none of it
(+0.003). The real trainer without augmentation is the replica. The
4x wall-clock comes from the same cause: augmented card text misses the
feature cache and must go through ModernBERT (main_T encoded about 300k
new texts in round 0, `samples.csv`).

Per game (`results/deck_auc_by_game.csv`), turning augmentation off
helps most where a masked field nearly names the deck: MTG +0.056 (set),
Lorcana +0.051, Hearthstone +0.032, Gundam +0.031. It **hurts the
unseen game**, Gwent, by 0.020 (0.595 with augmentation, 0.575 without),
and Dominion by 0.015. That fits the masking's purpose
(`src/dojos/augmentation_defaults.py`): denying the model the field
that nearly names the deck, which a new game will not share. So the
early "slowness" is mostly the price of that masking, measured on an
evaluation that rewards the shortcut. At 80k the cost on trained games
is small (replica 0.869 against main_T 0.864). One seed each, so a
difference of 0.02 on one game is suggestive, not established.

**Where the investigation stands** (for the human; not Conclusions yet):
1. "Why wasn't T faster?" On trained games, T *was* about 0.025 faster
   at 10k (0.842 against R's 0.828), and augmentation took about as
   much away from both arms. T plateaus at about 0.87 because a yes/no
   top-32 target on a 1.3M head saturates by about 20k steps, offline
   and in the real trainer alike, and over-fits after.
2. The teacher holds much more: 0.944 against 0.89 on its own cards.
   A graded similarity target with a wider head recovers about 40% of
   that gap (0.914), and is still rising.
3. Field information and rank collapse come from the teacher-pair
   signal (the replica reproduces both). A graded target and a wide
   head keep more rank (85 against 30), but linear-cls still falls to
   about 0.60. The human's mixed-diet or field post-training idea is
   still the candidate remedy.
4. Pooled text generalises the teacher poorly to unseen cards (C: 0.83),
   which backs the move to token embeddings or a better text side.

### 2026-11-12 · Reading checked with the human; direction for the next experiment (PI, with the human)

**Two corrections to the 15:10 entry:**
- Item 1 says T was "about 0.025 faster at 10k (0.842 against R's
  0.828)". The paired delta is +0.014 [+0.012, …]
  (`results/round_compare.csv`, round 0). The 0.025 is the LR pilot's
  number (above, 14:15 entry).
- The "4× wall-clock" from augmentation holds for round 0 only.
  Augmentation draws at most `variants: 4` texts per card and mod, and
  they persist in the on-disk feature cache, so the cost amortises.
  Round times from `rounds.csv`:
  - main_T: 1,826 s (round 0), 953 s (round 1), then falling to 596 s
    (round 7);
  - main_R: 1,436 s, 915 s, then falling to 728 s;
  - ablation_T_noaug: 407 s for round 0.
  The steady-state cost is about 1.5×, not 4×.

**The human's decisions:**
- Keep the masking augmentation. Transfer to unseen games is valued, and
  the one-seed Gwent and Dominion drops without it are reason enough
  for now.
- The field-information fade is not a problem to solve yet. linear-cls
  stays as an evaluation metric; recovering fields is nice-to-have
  (matches the project's embedding-quality goal).
- Less interested in pooled text features. Token embeddings are the
  obvious candidate, but the options are to be weighed first.
- Focus of the next experiment: a better teacher-to-student signal, and
  a richer student (deeper, somewhat wider).

**PI's proposal, under discussion:** rank the candidate signals and
candidate text inputs offline first, with the probe harness
(`analysis/distill_probes.py`, minutes per arm; probe C's held-out-card
score for text inputs), then take the winners into one real-trainer
run.

### 2026-11-12 · Closing (PI, with the human)

The human asked to record the next-step ideas (under Open questions)
and close the experiment; further work becomes a new experiment.

**One more correction to the 15:10 entry:** item 1 says augmentation
"took about as much away from both arms". Only T was ablated. R without
augmentation was never run, so how much masking costs R is unmeasured,
and whether T beats R with neither arm masked is open.

**Corrections from the conclusion-mode review (experiment-critic):**
- The 13:00 entry says a per-game head of width 512 "memorises the
  teacher (cosine 0.99), and width adds nothing". That holds for the
  mean over games, not for MTG, the largest: fit cosine 0.960 at width
  512 and 0.994 at 2048, deck AUC 0.966 → 0.973
  (`results/probe_c2_capacity.csv`). Width matters per game where a game
  has many cards.
- The 09:10 entry reads R_r7's below-chance Dominion score (0.430) as
  "R's training made Dominion worse". Unrelated encoders are below
  chance on Dominion too: text_ridge 0.346, E_kl_wide 0.334, and C2's
  in-sample fit 0.24 at fit cosine 0.996, while the teacher itself
  scores 0.817 (`results/probe_deck_auc.csv`, `probe_e_covered.csv`,
  `probe_c2_capacity.csv`). Only 144 of Dominion's 654 teacher cards are
  confident, so most of its scored cards are predicted, not fitted. The
  cause is unchecked; Dominion's deck AUC is treated as suspect.

**Planned evaluation items not done** (Method, step 3 and Controls):
- Gwent kNN and fresh heads on Gwent's dojos (step 3.3): never
  implemented in `analysis/evaluate.py`. Transfer is judged on zero-shot
  deck AUC alone.
- r2's `single_contrastive` checkpoint as a historical point: not
  evaluated. It was a non-controlled reference, and the untrained head
  and frozen mean already bracket the arms.
- Cold start (step 3.4) and mean pairwise cosine (step 3.5) were
  computed but not reported until Results below.

**Ablation provenance:** the two ablations were short inline runs by
the PI, recorded in the 15:10 Log entry rather than in bench notes.
Their queue script is now `analysis/run_ablations.sh` (copied from the
session scratchpad; the one line that waited on a session job is
replaced by a comment).

## Results

All deck AUCs are the diet-weighted same-deck retrieval AUC on fixed
VALIDATION decks (500 per game, fewer for small games; 0.5 = chance),
one seed. Training is deterministic at a fixed seed here
(`main_R_attempt1` against `main_R` round 7: Δ = 0.0000), so the
bootstrap CIs cover deck sampling only; seed-to-seed spread is
unmeasured.

**Main comparison** (`results/round_compare.csv`,
`results/deck_auc_by_round.png`; wrap-up
[bench note](bench_notes/2026-11-12_0125_wrap_up.md)):

| round (steps) | R | T | Δ T − R [95% CI] | R linear-cls | T linear-cls |
|---|---|---|---|---|---|
| 0 (10k) | 0.828 | 0.842 | +0.014 [+0.012, +0.015] | 0.712 | 0.713 |
| 1 (20k) | 0.846 | 0.860 | +0.015 [+0.013, +0.016] | 0.701 | 0.676 |
| 2 (30k) | 0.853 | 0.863 | +0.010 [+0.008, +0.011] | 0.703 | 0.655 |
| 3 (40k) | 0.860 | 0.866 | +0.006 [+0.005, +0.008] | 0.705 | 0.629 |
| 7 (80k) | 0.867 | 0.864 | −0.003 [−0.004, −0.002] | 0.700 | 0.591 |

Controls (`results/eval_summary.csv`): untrained head 0.640 deck AUC,
0.739 linear-cls; frozen whitened mean (no head) 0.687, 0.927. Gwent
(unseen game) at 80k: R 0.584, T 0.545; untrained head 0.586, frozen
mean 0.633. Effective rank at 80k: R 88, T 33.

Also in `results/eval_summary.csv`:

| encoder | cold start: deck AUC on VALIDATION-tier anchors | mean pairwise cosine |
|---|---|---|
| untrained head | 0.617 | 0.52 |
| frozen whitened mean | 0.657 | 0.02 |
| R_r0 / R_r7 | 0.742 / 0.755 | 0.53 / 0.40 |
| T_r0 / T_r7 | 0.766 / 0.751 | 0.37 / 0.11 |
| E_kl_wide, 80k (offline) | 0.793 | 0.03 |

Neither arm collapses to one direction (mean cosine falls during
training); T's loss is in effective rank, not spread.

Training-time cross-task losses (bench notes; 1.00 = untrained, 128
TEST examples per dojo, noisy): R's median on T's held-out teacher
dojos ends at 0.819; T's median on R's held-out real-deck dojos rises
from 0.91 to 1.03, above the untrained baseline.

**Probes** (details and predictions in the Log, 09:10 to 15:10):

| question | result | file |
|---|---|---|
| A. Teacher's own deck AUC (teacher cards only) | 0.944, against R_r7 0.892, T_r7 0.887 | `results/probe_deck_auc.csv` |
| B. Over-fitting to TRAIN decks | TRAIN − VALIDATION gaps of −0.001 to +0.006 for every encoder | `results/probe_deck_auc.csv` |
| C. Pooled text → teacher, unseen cards | 0.833 (ridge), 0.832 (MLP) | `results/probe_text_to_teacher.csv` |
| C2. Per-game head fit, seen cards | mean cosine 0.99 at widths 512-2048 (MTG 0.960 → 0.994); deck AUC 0.919-0.921 | `results/probe_c2_capacity.csv` |
| E. Training signal offline, 80k steps | see below | `results/probe_e_curves.csv`, `results/probe_e_covered.csv` |
| Ablations, 10k steps in the real trainer | no augmentation 0.867, no clipping 0.844, main_T 0.842 | `results/eval_summary.csv`, `results/deck_auc_by_game.csv` |

Probe E at 80k steps, all cards (teacher cards in brackets):

| | width 512 (student) | width 2048 |
|---|---|---|
| yes/no top-32 target | 0.871 (0.897) | 0.874 (0.893); peaked at 0.885 at 40k |
| teacher-similarity KL target | 0.874 (0.900) | **0.897 (0.914)**, still rising |
| main_T's exact batch shape (replica) | 0.869 (0.891) | - |

## Conclusions

**Speed hypothesis: no clear difference** (by the Aim's rule). T led R
in every first-half round, with CIs excluding 0, but by +0.006 to
+0.015, never reaching +0.02 (`results/round_compare.csv`). One seed.

**Quality hypothesis: refuted** (by the Aim's rule). The final deck-AUC
difference, −0.003 [−0.004, −0.002], is inside the ±0.02 margin, but
T's linear-cls is 0.108 below R's (0.591 against 0.700), against an
allowed 0.02. On deck AUC alone the arms end level. One seed.

**Against the predictions in Method:**
- Both arms beat the untrained head by far more than 0.02 (0.867 and
  0.864 against 0.640): as predicted.
- T rises faster early: as predicted in direction, under the bar in
  size.
- Field probe: both lose against the frozen mean, as predicted, but T
  loses more, not less (0.591 against 0.700).
- Gwent: neither arm is near chance (R 0.584, T 0.545, chance 0.5), but
  neither beats the untrained head (0.586) or the frozen mean (0.633).
  On zero-shot deck AUC (the only transfer measure run; kNN and fresh
  heads were not), training on the 15 games does not yet transfer to
  the unseen one.
- R's TEST loss on T's held-out teacher dojos ends at 0.819 (median;
  1.00 = untrained), so real-deck training does learn part of the
  teacher's neighbourhoods, as predicted (main_R
  [bench note](bench_notes/2026-11-11_2150_run_main_R_done.md); the
  per-round values are too noisy to show a trend). The mirror was not
  predicted: T's loss on R's real-deck dojos rises from 0.91 to 1.03,
  ending above untrained (main_T
  [bench note](bench_notes/2026-11-11_2345_run_main_T_done.md)), while
  T's deck AUC stays near R's. Training-time TEST is 128 examples per
  dojo, so this is weak evidence.
- The staple-concentration correlation was only to be checked if T won
  on quality, so it was not computed.

**Further findings**, each one seed, frozen ModernBERT-base with mean
pooling and whitening, 15 games:

1. **The student absorbs only part of what the teacher knows.** The
   teacher scores 0.944 on its own cards against 0.892 for R and 0.887
   for T (probe A). No encoder over-fits its training decks (probe B,
   gaps ≤ 0.006), so the gap is not memorisation.
2. **A yes/no top-32 target on a 512-wide head plateaus at about
   0.87, and neither change alone lifts it.** Trained offline on cached
   features, it levels off by 20k steps with main_T's batch shape
   (replica 0.869 at 80k) and with richer batches (0.871). A 4× wider
   head lifts the yes/no target only mid-run (0.885 at 40k) and it falls
   back to 0.874 by 80k; the graded target at width 512 ends at 0.874.
   The replica also reproduces T's loss of field information and rank
   (linear-cls 0.515, rank 30). So T's end point and its collapse come
   from the signal and head together, not from the training pipeline.
3. **A graded similarity target on a wider head does better, but only
   both together.** KL distillation of the teacher's within-game
   similarities with a 2048-wide head reaches 0.897 (all cards) and
   0.914 (teacher cards), still rising at 80k, and closes about 40% of
   the gap to the teacher. Either change alone adds about 0.003. Scope:
   offline, on cached features, without augmentation; not yet run in
   the real trainer.
4. **Head size is mostly not the limit within one game.** A per-game
   head the student's size fits the teacher's vectors to a mean cosine
   of 0.99 over games, and widening barely moves the deck AUC (0.919 to
   0.921; probe C2). The exception is MTG, the largest game: cosine
   0.960 at width 512 and 0.994 at 2048 (deck AUC 0.966 to 0.973). For
   the one shared head, width mattered with a graded target
   (finding 3).
5. **Pooled text is a weak predictor of the teacher for unseen
   cards.** Out-of-fold predictions score 0.83 (probe C), below the
   students' 0.89 on seen cards. Whether the limit is the pooling or the
   text model is not separated.
6. **Augmentation explains T's early lag behind the offline replica**
   (T still led R). Without the default field masking, T reaches 0.867
   at 10k steps, against 0.842 with it, matching the replica (0.867).
   Clipping explains none of it (0.844). The masking hides fields that
   nearly name a deck (MTG +0.056, Lorcana +0.051 without it), and
   removing it hurt the unseen game, Gwent (0.595 → 0.575): suggestive,
   one seed (Dominion also fell, by 0.015, but its metric is suspect;
   see the closing Log entry). Its wall-clock cost is about 4× in
   round 0 and about 1.5× once the cache holds the masked variants
   (`data/runs/2026-11-11_teacher_vs_real_pairs/<run>/rounds.csv`). R
   was not ablated.
7. **Throughput.** The offline loop runs 10k steps in 25-70 s; the real
   trainer took about 7 min (no augmentation) to 30 min (round 0 with
   augmentation) for the same. An engineering question, outside this
   aim.
8. **Dominion's deck AUC looks unreliable.** Several unrelated
   encoders score below chance on it, including a near-exact in-sample
   fit of the teacher (0.24) while the teacher scores 0.817 (closing Log
   entry). The cause is unchecked. Its diet weight is small, so it
   barely moves the weighted score, but per-game claims about Dominion
   should wait for a look at the metric.
9. **Cold start:** on VALIDATION-tier anchors, cards neither arm
   trained on, the arms end level (R 0.755, T 0.751), above the frozen
   mean (0.657); the offline E_kl_wide reaches 0.793.

**Decisions taken from this experiment (the human):** keep the masking
augmentation; treat linear-cls as an evaluation metric, not a goal;
focus the next experiment on the teacher-to-student signal, the text
input and a richer student.

**Promoted to:**
- `experiments/TODO.md`, Teacher-student: the next experiment (signal,
  text input and head comparison, offline first), R without
  augmentation, seeds, warmup-stable-decay, and the throughput gap for
  an `architect` session.
- `experiments/README.md`: index row, and the measured run times under
  Typical environment.
- `keep_round_encoders` (the round-checkpoint fix, incident
  [bench note](bench_notes/2026-11-11_1950_incident_round_checkpoints_pruned.md))
  is already in `src/training/`.

## Open questions

- **Which part of the bundle matters**, if T and R differ:
  - T with uniform anchors replaced by deck-popularity anchors;
  - R with anchors drawn uniformly over cards (a partner from one of the
    anchor's decks);
  - T with raw co-occurrence counts instead of PPMI;
  - R with `staple_subsampling` at a finite t (already built,
    `src/training/run_config.py`), a cheap way to isolate the
    reweighting part.
- Neighbour count k (8 / 32 / 128) and multi-neighbour pseudo-decks
  (`student.max_neighbours_per_deck`).
- Seeds of the winning arm.
- **R without augmentation**, to tell whether T beats R when neither
  arm is masked (only T was ablated).
- Probe D (deferred): deck AUC split by staples and by cards with
  near-identical text.
- **Why Dominion's deck AUC is below chance** for unrelated encoders
  (finding 8): check the metric (shared base cards, low confident
  coverage) before trusting per-game Dominion numbers.
- Gwent kNN and fresh heads on Gwent's dojos, the transfer measures
  planned here but never built.

### Ideas for the next experiment (2026-11-12, PI with the human)

The human's direction: a better teacher-to-student signal, a better
text input than one pooled vector, and a richer student (deeper,
somewhat wider). Masking stays on. The PI's proposal: rank the options
offline first, on the probe harness (`analysis/distill_probes.py`,
minutes per arm, and probe C's held-out-card score for text inputs),
then take the winners into one real-trainer run against R.

**Teaching signals** (the current best is finding 3: in-batch KL on the
teacher's within-game cosine softmax, temperature 0.05):
1. **Per-game adapter regression** (FitNets, Romero et al. 2014,
   arXiv:1412.6550): the student vector passes through a learned linear
   map per game and is regressed onto that card's teacher vector. The
   densest signal (a full vector per card, not relations within one
   batch); the map absorbs the rotation between game spaces. C2 shows a
   per-game fit of cosine 0.99 is possible. Risk: the adapters do too
   much of the work.
2. **A queue of anchors instead of one batch** (SEED, Fang et al. 2021,
   arXiv:2101.04731; CompRess, Abbasi Koohpayegani et al. 2020,
   arXiv:2010.14713): KL over similarities to thousands of stored
   same-game teacher vectors, so each step carries more graded
   information.
3. **Ranking distillation** (DarkRank, Chen et al. 2018,
   arXiv:1707.01220; RankDistil, Reddi et al. 2021): learn the order of
   the teacher's nearest neighbours rather than their exact
   similarities. Deck AUC is itself a ranking metric.
4. **Contrastive representation distillation** (CRD, Tian et al. 2019,
   arXiv:1910.10699): a student card and its own teacher vector as a
   positive pair against other cards; close to 1 with negatives.
5. **Whole-batch structure matching** (CKA, Kornblith et al. 2019,
   arXiv:1905.00414): match the batch's similarity matrix, invariant to
   rotation and scale. Cheap, likely weaker than KL.
6. **Past the teacher:** fit the PPMI co-occurrence directly, GloVe-style
   (Pennington et al. 2014), since the truncated SVD throws information
   away. Later.
Also: the KL temperature, and combinations (adapter regression + KL).

**Text input** (today: ModernBERT-base, a masked-language model not
trained for sentence embeddings, mean-pooled and whitened):
- **Token embeddings** with learned attention pooling or a small
  transformer: most information; storage roughly 20 GB per text variant
  (a rough figure, ~100k cards × ~128 tokens × 768 dims × 2 bytes),
  about 100 GB with the 4 masking variants; slower steps.
- **One vector per card field** (name, type line, rules text, stats …),
  with the student attending over the set: ~10 vectors per card instead
  of ~128, and masking becomes dropping field vectors in training, so it
  needs no re-encoding and no cache. The encoder loses context between
  fields. The PI's pick, alongside token embeddings.
- **A frozen embedding model** (for example an embedding-tuned
  ModernBERT; to verify which exist and their licences): the cheapest
  test, a re-encode and probe C.
- **A mix of several hidden layers** (ELMo-style, Peters et al. 2018,
  arXiv:1802.05365): cheap; likely a small gain.
- **Partial fine-tuning** (cache a lower layer's output, train the top
  layers or LoRA, Hu et al. 2021, arXiv:2106.09685): most expressive
  while keeping some caching; heavy engineering. Full fine-tuning
  later.
Probe C (predict teacher vectors for cards held out of the fit) can
rank the inputs before any training.

**Student:** width (1024 / 2048 / 4096) against depth (3 / 6 / 9
blocks), offline. If the input becomes a set of token or field
vectors, the head changes shape (attention over the set), so the input
is chosen first and the head sized after.

**Schedule:** warmup-stable-decay (MiniCPM, Hu et al. 2024,
arXiv:2404.06395), so a plateau can be told apart from the decay.
