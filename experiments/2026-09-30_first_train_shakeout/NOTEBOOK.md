# First end-to-end train + evaluate shakeout (2026-09-30)

**Status:** concluded
**Dates:** 2026-09-30 (the section 7 follow-ups came later; their date wasn't recorded)
**Parent:** none · **Code:** commit `8e3f382` (branch `first-training-shakeout`)

> Problem: can we train a single-card model on GPU and evaluate it
> intrinsically and extrinsically, end to end, with everything logged?

**Migration note (2026-10-11).** This notebook was written before
`EXPERIMENT_PRINCIPLES.md` existed, as `plans/archive/first_train_shakeout.md`,
and was moved here unchanged apart from this header. Read it as a bench
log plus write-up in one file. What it lacks under today's rules:

- no predictions written before the runs;
- a single seed;
- no committed `results/`: its tables were typed from the run outputs at
  the time.

**The raw artifacts it cites are no longer on disk:** `models/single_card/`,
`data/evaluations/gwent_contrastive_v1/` and `logs/first_train_shakeout/`.
The config was `configs/training/gwent_contrastive.yaml`, which has
changed since, so the version used is in commit `8e3f382`. The "This log"
row below gives the old path.

**Verdict.**
- **Problem: met.** The pipeline trains, checkpoints and evaluates end to
  end. The bugs found were fixed (section 7).
- **Main finding:** deck-contrastive training alone collapses the
  embedding (effective rank 48 -> 10). It keeps faction and squeezes out
  rarity. That argued for a mixed diet, which became
  [2026-10-09_model_by_diet](../2026-10-09_model_by_diet/NOTEBOOK.md).

---

Goal (from the user): train a single-card embedding model on a contrastive
dojo diet (8/1/1 train/test/validation), log everything to files, save the
model under `models/single_card/`, then run intrinsic + extrinsic
evaluation on it with logged output. Small bugs: fix and record here. Big
bugs needing design: stop and record here for discussion.

## Where everything is

| What | Where |
|---|---|
| This log | `plans/first_train_shakeout.md` |
| Training console log | `logs/first_train_shakeout/train.log` |
| Trained model (checkpoints, rounds.csv, config copy) | `models/single_card/gwent_contrastive_v1/` |
| Evaluation console log | `logs/first_train_shakeout/evaluate.log` |
| Evaluation results | `data/evaluations/gwent_contrastive_v1/` |

`models/` was added to `.gitignore` (checkpoints hold the full LM, several
hundred MB each); `logs/` was already ignored.

## Pre-flight findings

- **Deck box ingestion is NOT needed.** `data/final/decks/gwent.db`
  (2026-09-27 13:58) postdates the Gwent card binder (13:52), and
  `--check` on `gwent_contrastive.yaml` already passed against it (48,157
  TRAIN decks, 6,020 TEST). FaB and STS2 deck boxes are the same age.
- **Splits are 8/1/1 twice over:** decks by `DeckBoxDealer`'s default
  `split_ratios=[8, 1, 1]`, cards by the config's
  `holdout.tier_ratios: [8, 1, 1]` (cards in the TEST/VALIDATION tiers are
  never trained on).
- **ROCm venv lacked the evaluation deps** (matplotlib, scikit-learn),
  both already in `requirements.txt`.

## Log

(entries appended as the run proceeds)

### 1. Training refused to start: "phase 'frozen' has no trainable parameters" (FIXED)

- **Cause (config bug, mine):** `Phase.encoder_trainable` covers every
  card-model parameter that is trainable as built. The frozen ModernBERT is
  excluded when the Trainer is constructed, so for `LinearProjectionCardModel`
  the "encoder" is just the projection head. `gwent_contrastive.yaml` set
  `encoder_trainable: false`, which froze the only trainable part, and a
  contrastive dojo has no head of its own.
- **Also affected `gpu_smoke.yaml`:** it had the same setting, so the earlier
  gwent-mask GPU/CPU smoke runs trained only the dojos' decoder heads on top of
  a frozen, random projection. Those runs proved the plumbing but taught the
  embedding nothing.
- **Fix:** both configs now use `encoder_trainable: true`,
  `encoder_lr: 1.0e-3`, with a comment explaining what "encoder" means here.
- **Fix:** `--check` missed this because it never built a Trainer. It now
  also builds the model and constructs a `Trainer` (the constructor validates
  the plan against the model and the dojos), without a checkpointer or
  listeners. `--check` now also loads ModernBERT, so it takes a few seconds
  longer.
- **Discussion item (not changed):** the name `encoder_trainable` / `encoder_lr`
  is misleading now that the LM is frozen by the model and the "encoder" is the
  card model's own head. Consider renaming it, e.g. `card_model_trainable`, or
  splitting LM and head learning rates. See also
  `src/encoder_model/TODO.md` (thawed-encoder model).
- The failed attempt left only `run_config.yaml` in the run directory; I
  removed it so the rerun could use the same directory.

### 2. Training run (relaunched 2026-09-30)

Command (ROCm venv):
```
PYTHONPATH=. G:/Projects/venvs/tcg-rocm/Scripts/python scripts/run_training.py \
    configs/training/gwent_contrastive.yaml \
    --set run_directory=models/single_card/gwent_contrastive_v1 \
    --set phases.0.max_rounds=7
```
7 rounds x 200 steps x 32 cards (16 decks x 2), fp16, frozen ModernBERT +
linear head (embed_dim 256), Gwent default augmentations.

**Result: success, exit 0, no quarantines, no stop from saturation.** Log:
`logs/first_train_shakeout/train.log`; per-round CSV:
`models/single_card/gwent_contrastive_v1/rounds.csv`.

| Round | Step | Elapsed (s) | TEST InfoNCE loss |
|---|---|---|---|
| 0 | 200 | 56 | 3.403 |
| 1 | 400 | 347 | 3.102 |
| 2 | 599 | 590 | 2.784 |
| 3 | 798 | 815 | 2.766 |
| 4 | 998 | 1053 | **2.572** (best) |
| 5 | 1198 | ~1290 | 2.598 |
| 6 | 1398 | ~1530 | 2.610 |

- The chance level for 32 items is ln 32 ≈ 3.47, so the linear head learned
  something (about 0.9 nats below chance). It leveled off after round 4;
  more rounds of the same setup look like diminishing returns.
- **Best model:** `models/single_card/gwent_contrastive_v1/frozen_round0004/`,
  containing `encoder.pt`, `state.pt`, `manifest.json` and `holdout.json`.
  Load it with `load_encoder_weights(dir, LinearProjectionCardModel(embed_dim=256))`.

#### Findings from the run

- **Checkpoints are 1.2 GB each; 6 GB for this run, for a 197k-parameter
  linear head. Discussion, not fixed.** `encoder.pt` (597 MB) is the full
  model state dict, frozen ModernBERT included, and `state.pt` (598 MB)
  apparently holds it again, alongside the optimizer state. A new best is
  saved every round and nothing is ever deleted. This is the open "checkpoint
  retention" TODO item, now with numbers. Options:
  - save only trainable parameters plus the LM checkpoint name when the LM
    is frozen;
  - keep the last and best checkpoints only;
  - don't duplicate the encoder weights in `state.pt`.
- **Rounds 1-6 took about 4 minutes each; round 0 took 56 s. Probably caused
  by me, not a bug.** An evaluation `--smoke` run was going on the CPU at the
  same time, from round 1 onward, and CPU-side work (tokenization, mods,
  batching) competed with it. Not confirmed. The next training run alone on
  the machine will show whether round 0's pace holds. If it doesn't, check
  for Windows VRAM spill (see TODO section A).
- **2 optimizer steps were skipped silently.** The rounds end at steps 599,
  798 and so on instead of 600 and 800. That is almost certainly the fp16
  GradScaler skipping an overflowed step, which is by design. But nothing is
  logged, so it is invisible except in the step column. Small follow-up
  (not done): count skipped steps in the round report or CSV.
- **No log line says what is being trained.** It would have made issue 1
  obvious. Small follow-up (not done): print the trainable parameter count
  per group at startup.

### 3. Evaluation smoke test (CPU, while training ran)

`scripts/run_evaluation.py --smoke` (untrained encoder only, 200 cards),
output under the session scratchpad; log
`logs/first_train_shakeout/evaluate_smoke.log`.

- Stages 1-4 (binder, encoders/labels/dojos, embed, intrinsic analyses)
  ran clean. The `holdout_tier` analyses were skipped as expected, because
  with no checkpoint everything is in the TRAIN tier.
- Stage 5 (extrinsic) got through round 0 (5 steps + eval: 593 s on a CPU
  shared with training) and was then killed by my own 25-minute `timeout`.
  Not a code failure. The GPU run (below) exercises the same path.
- **Fix (small): `run_evaluation.py` had no device option**, so every
  encoder, and the extrinsic dojo heads with it, ran on the CPU. I added
  `--device` (default `cpu`). Each model moves there after its weights
  load, and `Trainer.run` moves the extrinsic heads to follow it.
  `isolated_embeddings` already returns CPU tensors, so the embedding
  table code needed no change.
- The ROCm venv needed `matplotlib` and `scikit-learn`, which were
  installed (both are already in `requirements.txt`).

### 4. Real evaluation (GPU, fp16)

```
PYTHONPATH=. G:/Projects/venvs/tcg-rocm/Scripts/python scripts/run_evaluation.py \
    --name gwent_contrastive_v1 \
    --single-checkpoint models/single_card/gwent_contrastive_v1/frozen_round0004 \
    --device cuda --precision fp16
```
Log: `logs/first_train_shakeout/evaluate.log`; results:
`data/evaluations/gwent_contrastive_v1/` (embedding tables, `intrinsic/`
scalars and plots, `extrinsic/` CSVs and learning-curve plots).
Encoders: `single` (the trained checkpoint) vs `untrained` (same
architecture, random linear head).

#### Intrinsic (all 1,260 Gwent cards)

| Label | Metric | untrained | **single (trained)** | baseline |
|---|---|---|---|---|
| faction (7) | cluster ARI | 0.041 | **0.710** | 0 |
| faction | cluster NMI | 0.061 | **0.725** | 0 |
| faction | silhouette | 0.003 | **0.191** | 0 |
| faction | nearest-centroid acc. | 0.567 | **0.876** | 0.193 (majority) |
| rarity (4) | cluster ARI | 0.019 | -0.001 | 0 |
| rarity | nearest-centroid acc. | 0.568 | **0.340** | 0.322 (majority) |
| holdout tier (3) | nearest-centroid acc. | 0.359 | 0.258 | 0.662 (majority) |

Reading:
- **Faction: a big, real gain.** The contrastive positives are "two cards
  from the same deck", and a Gwent deck is single-faction, so the model
  learned faction structure strongly, and that structure is now what the
  embedding's clusters are.
- **Rarity got worse** (0.57 → 0.34, down to the majority baseline). The
  256-d embedding reorganized around the deck signal, and rarity (which
  cuts across decks) was squeezed out. That's expected for a
  single-objective contrastive head, and it's the argument for a mixed diet
  (masks + contrastive) in the next run.
- **Holdout tier:** no structure in either encoder, which is correct
  (tiers are random). The trained model does *not* separate seen from unseen
  cards, a mild sign that it isn't memorizing individual cards.
- Shortcut caveat: the faction mask was applied to ~35% of training cards,
  but the other ~65% still showed faction in plain text, so part of this
  gain may be "read the faction field" rather than richer card semantics.
  Evaluating with faction masked in the input would separate the two.
- The two `projection` KL values per encoder are identical across label
  sources. That is expected: t-SNE's KL doesn't depend on labels, and the
  sample is the same.

#### Extrinsic: DISCUSSION NEEDED. The heads overfit, and saturation keeps them training

The extrinsic stage runs (GPU: ~41 s per 200-step round at 64 cards per
step, versus 593 s per 5 steps on the contended CPU). But the curves show a
design problem (`data/evaluations/gwent_contrastive_v1/extrinsic/single/rounds.csv`):

| round | faction_mask TEST | rarity_mask TEST |
|---|---|---|
| 0 | 1.155 | 1.001 |
| 1 | **1.122** | 0.925 |
| 2 | 1.155 | **0.873** |
| 4 | 1.244 (saturated) | 0.928 |
| 7 | 1.401 | 1.020 |
| 11 | 1.770 | 1.198 |
| 13 | 1.708 | 1.260 |

1. **The heads overfit within 1-2 rounds.** Each dojo has about 1k
   TRAIN-tier cards, and a round is 200 steps x 64 cards, so each round is
   several epochs over the same cards. TEST loss bottoms out in round 1-2
   and then climbs steadily.
2. **Saturation amplifies it instead of stopping it.** In
   `SaturationTracker` (`src/training/diet/saturation_tracker.py`) a
   SATURATED dojo is reactivated when its loss rises above
   best + `reactivation_delta`, and its best is then *reset to the
   regressed loss*. An overfitting head's rising TEST loss is read as
   "regressed, needs more training". So the dojo trains more, gets worse,
   saturates after 3 rounds, is reactivated, and so on. The log shows
   `saturated`/`active` flip-flopping with the loss climbing. `phase_done`
   is never reached, so each encoder runs all 50 rounds (~35 min each).
   Reactivation was designed for multi-dojo encoder training, where
   *another* dojo's updates move a shared encoder. With a frozen encoder
   and one head per dojo, the only thing that can raise a dojo's loss is
   its own overfitting.
3. **The reported extrinsic score comes from the last round's heads, the
   most overfit ones.** `run_extrinsic` passes `checkpointer=None`, and the
   validation losses are computed at the end, so an encoder's extrinsic
   number is "how badly did the head overfit by round 50", not "how well
   can a head read this embedding". The learning-curve plots still show the
   minimum, which is the useful signal for now.

Options to discuss (I have not changed anything):
- For extrinsic runs, turn reactivation off, or make it relative to the
  all-time best rather than resetting best to the regressed value;
- early-stop on TEST and report validation at the best-TEST round (keep the
  best heads in memory; no checkpointer needed);
- shrink the extrinsic budget (e.g. 50 steps per round), and/or add weight
  decay or dropout on the extrinsic heads;
- the main training loop has the same reactivation semantics. It's
  harmless with one contrastive dojo and no head, but it matters as soon as
  a masking dojo with a head trains on a frozen LM (the `gpu_smoke.yaml`
  setup).

Decision taken for this shakeout: let the evaluation finish its 50 rounds
as designed, so the pipeline is proven end to end and every artifact (CSVs,
plots, validation numbers) exists. Read the curves' minimum, not the final
validation numbers.

**Update: fixed at the user's request, and extrinsic re-run.** The first
extrinsic run was stopped at round 17 of 50 for `single` (the `untrained`
encoder hadn't started). Its partial results are kept for comparison in
`data/evaluations/gwent_contrastive_v1/extrinsic_v0_reactivating/` and
`logs/first_train_shakeout/evaluate_v0_reactivating.log`.

Fixes, all in `scripts/run_evaluation.py` (no core `Trainer` /
`SaturationTracker` / `run_extrinsic` change):
- **Extrinsic saturation never reactivates** (`reactivation_delta=math.inf`).
  A saturated head stays out of the diet, and the phase ends once every
  dojo is saturated.
- **Shorter rounds:** `steps_per_round` 200 → 50 (and `max_rounds` 50 → 60).
  A head now trains at most 3 x 50 = 150 steps past its best TEST round,
  instead of 600, and the best round is found at 50-step resolution.

Still open (follow-ups, not done):
- **The VALIDATION score is still taken from the final heads,** which are
  now at most 150 steps past their best rather than hundreds. A proper fix
  keeps each dojo's head at its best-TEST round, for example a RunListener
  in `run_extrinsic` that snapshots a head's state_dict whenever that dojo's
  TEST loss improves, and restores the snapshots before the VALIDATION pass.
- **The main `Trainer` has the same reactivation semantics.** For a
  frozen-LM phase where the dojos have their own heads (the `gpu_smoke.yaml`
  setup) it would show the same flip-flop. Options: skip reactivation when
  the phase's encoder is frozen, or measure reactivation against the
  all-time best rather than resetting best to the regressed loss.

### 5. New: `scripts/plot_learning_curves.py` (at the user's request)

Plotting only ever depended on rounds CSVs (`plot_learning_curves` takes
label -> CSV path). The new script exposes that directly, so a crashed or
stopped run can still be analysed on a best-effort basis:
```
PYTHONPATH=. python scripts/plot_learning_curves.py \
    --curve single=data/evaluations/gwent_contrastive_v1/extrinsic/single/rounds.csv \
    --curve untrained=data/evaluations/gwent_contrastive_v1/extrinsic/untrained/rounds.csv \
    --output-dir data/evaluations/gwent_contrastive_v1/extrinsic/plots
```
It works on any CsvRunListener rounds CSV. Plots made with it:
- `data/evaluations/gwent_contrastive_v1/extrinsic_v0_reactivating/plots/`:
  the stopped first extrinsic run; the overfitting climb is plain to see.
- `models/single_card/gwent_contrastive_v1/plots/contrastive.gwent.png`:
  the training run's own TEST-loss curve.

Tests: `tests/scripts/test_plot_learning_curves.py` (argument parsing). Like
the other evaluation tests it needs matplotlib, so it runs in the ROCm venv
but not the project `venv/`.

### 6. Extrinsic re-run: results (complete, exit 0)

Log: `logs/first_train_shakeout/evaluate.log`. Plots:
`data/evaluations/gwent_contrastive_v1/extrinsic/plots/`. `single` ended after
15 rounds (~3 min) and `untrained` after 26 (~5 min), each with every dojo
saturated. Before the fix the projection was about 35 minutes per encoder.

| Dojo (cross-entropy) | untrained best TEST | single best TEST | untrained VALIDATION | single VALIDATION |
|---|---|---|---|---|
| faction_mask (7) | 1.401 | **1.105** | 1.325 | **0.937** |
| rarity_mask (4) | **0.745** | 0.876 | **0.847** | 1.082 |

The trained model is better at inferring a card's **faction** with the
faction field masked, and worse at inferring **rarity** with rarity masked.
The user flagged this as unexpected.

#### Why rarity got worse: diagnosis

A scratch diagnostic over the two embedding tables (1,261 unmasked cards)
gave the following:

| | untrained | single |
|---|---|---|
| effective rank (entropy of the singular-value spectrum) | 48.1 | **10.3** |
| variance in top 5 directions | 42% | **80%** |
| linear probe, faction (5-fold logistic regression) | 0.944 | 0.946 |
| linear probe, rarity | 0.971 | 0.924 |

1. **Dimensional collapse.** Contrastive training squeezed the 256-d
   embedding into about 10 effective dimensions. That is a well-known
   failure mode of InfoNCE with small batches (32 items, 31 negatives) and a
   loss computed directly on the representation.
2. **The objective has no reason to keep rarity.** The task is "which deck
   is this card from". Every deck mixes rarities, so rarity doesn't help
   tell decks apart, and InfoNCE pushes it into the low-variance directions.
   Faction *does* tell decks apart (Gwent decks are single-faction), so it
   gets the dominant directions. The faction-mask augmentation (~35% of
   training cards) also taught the model to infer faction from the other
   fields, which is exactly what the faction_mask dojo measures.
3. **The information isn't gone, it's squashed.** A linear probe still
   reads rarity at 92% (vs 97%) from unmasked cards. But distance-based
   metrics (clusters, centroids) and a small head on *masked* cards (which
   must infer rarity from subtler cues such as provision cost) mostly see
   the dominant directions, so they drop toward baseline.

So this is the expected behaviour of a single-objective, deck-contrastive
linear head, not a pipeline bug. It is the strongest argument yet for:
- **A mixed diet** (contrastive plus the gwent_one masking dojos). This
  needs the loss-normalization TODO first.
- **A SimCLR-style projector:** compute InfoNCE on a small MLP stacked on
  the embedding, and use the pre-projector embedding downstream. This is
  the standard fix for exactly this collapse-and-discard effect.
- **More negatives per batch** (larger `decks_per_sample`) and/or a tuned
  InfoNCE temperature.
- **Track effective rank** as an intrinsic analysis (cheap, and it flags
  collapse immediately).

## Summary for review

**Done end to end:** trained (GPU, fp16, ~25 min), saved to
`models/single_card/gwent_contrastive_v1/` (best checkpoint
`frozen_round0004`), evaluated intrinsically and extrinsically against the
untrained baseline. Every result and log is on disk (see "Where everything
is"). Deck box ingestion was not needed.

**Small fixes made:**
1. `gwent_contrastive.yaml` and `gpu_smoke.yaml` froze the only trainable
   part. Both now use `encoder_trainable: true` with a comment.
2. `run_training.py --check` now builds the model and constructs a
   `Trainer`, so plan errors surface without training.
3. `run_evaluation.py` gets `--device`, so evaluation can use the GPU.
4. `run_evaluation.py` extrinsic stage: no reactivation, 50-step rounds,
   ending about 12x sooner and no longer training overfit heads further.
5. New `scripts/plot_learning_curves.py` for plotting from CSVs on a
   best-effort basis (+ tests, + evaluation README).
6. `models/` added to `.gitignore`. The ROCm venv now has matplotlib and
   scikit-learn.

**For discussion (not changed):**
- Checkpoints are 1.2 GB each (the frozen LM, saved twice, every round).
  Retention and what to save.
- The main Trainer's reactivation rule has the same overfit flip-flop with
  a frozen LM and per-dojo heads.
- Extrinsic VALIDATION is scored on the final heads, not the best-TEST
  heads; a per-dojo best-head snapshot would fix it.
- The `encoder_trainable` / `encoder_lr` names are misleading now that the
  "encoder" is the card model's own head.
- Skipped fp16 steps and the trainable-parameter counts are not logged.
- Training rounds 1-6 were ~4x slower than round 0, probably CPU
  contention from my concurrent smoke run. To re-measure on a quiet machine.
- The rarity result: collapse (effective rank 48 → 10). Next-run design
  choices are listed in section 6.

## 7. Follow-up fixes (decided with the user after review)

- **Reactivation measured from the saturation point**
  (`SaturationTracker`): a SATURATED dojo re-enters the diet only if its
  TEST loss rises more than `reactivation_delta` above the loss *when it
  saturated*. Before, the comparison was against its best. With a frozen
  encoder, a paused head can't get worse, so the flip-flop is gone
  everywhere (main Trainer included). A shared encoder moved by other dojos
  still triggers reactivation. The user chose this over "vs all-time best",
  which would still flip-flop because an overfit head's frozen loss sits
  above its best, and over a moving average (no evidence of noisy losses
  yet).
- **Checkpoint retention** (`DirectoryCheckpointer`): keeps only the best
  checkpoint per phase (a new best deletes the phase's previous one once
  the new one is fully written) plus `latest/`, which the Trainer rewrites
  every round. The checkpoints CSV still lists every best ever written.
  Still open: checkpoints hold the full model, frozen ModernBERT included,
  in both `state.pt` and `encoder.pt` (1.2 GB each).
- **Trainer logging:** each phase logs its trainable parameter counts
  (encoder / dojo heads, with learning rates). Each round logs how many
  fp16 steps the GradScaler skipped, if any.
- **`EffectiveRank` intrinsic analysis**
  (`src/evaluation/analyses/effective_rank.py`), run by `run_evaluation.py`
  once per encoder under `intrinsic/<encoder>/unlabeled/effective_rank/`.
  On this run: single 10.28, untrained 48.06, matching the scratch
  diagnostic.
- **Extrinsic runs score each dojo's best head** (`BestHeadKeeper`): a
  RunListener snapshots each head at its best TEST round and restores it
  before the VALIDATION pass. `validation.json` gains `best_test_rounds`.
  Earlier extrinsic results were moved aside: `extrinsic_v0_reactivating/`
  (the original, stopped) and `extrinsic_v1_final_heads/` (the fixed
  saturation, final heads).
- **Flaky test fixed:** `test_stops_the_phase_when_saturated` assumed a
  zero learning rate makes the loss constant, but `FakeDojo` batches are
  `torch.randn`. It is now seeded. It failed only when new test files
  shifted the global RNG.

### Extrinsic re-run with best-head scoring (final numbers)

| VALIDATION (best-TEST head) | untrained | single |
|---|---|---|
| faction_mask | 1.440 (round 20) | **0.952** (round 7) |
| rarity_mask | **0.847** (round 18) | 1.031 (round 11) |

Same conclusion as before. The untrained faction head scores *worse* on
VALIDATION at its best-TEST round (1.440) than its final head did (1.325):
TEST and VALIDATION are small (~126 cards each), so single-round
selection is noisy at this scale. Committed as `8e3f382` on branch
`first-training-shakeout`.
