# Training at scale: where a multi-card mixed-diet run spends its time, with and without bounded mods

**Status:** proposed
**Dates:** 2026-10-24 -
**Parent:** [2026-10-21_cache_shakeout](../2026-10-21_cache_shakeout/NOTEBOOK.md) · **Branch:** `experiment/2026-10-24_scale_profile`

> At scale, a multi-card model on a mixed contrastive and metric diet
> trains at some rate set by its slowest parts; we measure those parts,
> with augmentation off and with it bounded to 4 cacheable variants, and
> project how long one contrastive pass over every card of every MTG deck
> would take.

## Aim

**Problem (priority: the fix, i.e. industry R&D; measurement, no bar).**
The shake-out (parent) measured a single-card model for 2 × 1,000 steps.
This experiment measures the current code over hours, on the model and
diet we expect to train (multi-card, r2's mixed diet), and answers:

1. **Bottlenecks:** where a step's time goes (waiting for the batch,
   forward + loss, the rest; and within those, mods, serialization,
   cache lookups and live encoding), and which dojos are slowest to
   build and to train.
2. **Memory pressure:** peak and steady VRAM (torch allocated and
   reserved, and the process's dedicated VRAM), process and free system
   RAM, and the feature cache's fill per tier.
3. **Cache hit rate:** the share of training texts not encoded, over
   time, and per tier; and whether evictions happen.
4. **GPU utilization:** the Windows compute-engine counter over the run.
5. **The at-scale estimate:** how long one contrastive pass that uses
   every known card of every TRAIN MTG deck as an anchor would take,
   without mods and with mods at 4 variants.

The answer is a ranked list of costs with a size for each, plus the
projection, to decide the next throughput work.

**Hypotheses** (predictions, second priority):

- **H1 (bounded mods are cacheable).** With `variants: 4`, after the
  first pass over the modded texts, training is about as fast as with
  mods off. Judged in steady state (the second half of training time),
  `mixed_n4` against `mixed_n0`. *Refuted if* `mixed_n4`'s hit rate is
  more than 3 percentage points below `mixed_n0`'s, or its cards per
  second in training steps is under 0.85× `mixed_n0`'s. (The bar is
  relative because task-mod texts and TEST passes count in both arms'
  hit rates.)
- **H2 (CPU-bound).** With mods off and a warm cache, the training
  thread spends most of each step waiting for its batch, and the GPU is
  mostly idle. Judged on `mixed_n0` in steady state, with GPU
  utilization taken over the same training window only. *Refuted if*
  the median share of step time spent waiting is under 30%, or the
  median GPU compute utilization is over 60%. `mtg_n0` is reported
  beside it, not judged.
- **H3 (a full MTG pass takes more than a day).** At the measured
  steady MTG-contrastive rate (wall-clock cards per second over the
  steady window), one pass of every TRAIN-visible card of every TRAIN
  MTG deck as an anchor takes more than 24 h, with or without mods.
  *Refuted if* either projection is under 24 h. The visit count is
  estimated (Method) and its uncertainty reported; if the bar falls
  inside that uncertainty, H3 is "not established".

## Background

- **Parent (2026-10-21_cache_shakeout):** single-card model, mixed diet,
  2 × 1,000 steps. With the then-default random mods, live encoding was
  82% of a cached step (1.22 s per step); mods off, 0.255 s per step,
  the GPU busy 21%, batch wait 13%. A bigger batch (512 vs 256 cards)
  gave only 9% more cards/s: the step is per-card bound. CardBinder's
  deep copies cost about 22 min of startup.
- **Since then:**
  - PR #12 merged the parent's speed work into main.
  - PR #13 removed CardBinder's deep copies, after a test showed no mod
    changes a binder card.
  - PR #14 (open, branched from here) added `variants:` to card-field
    mods: N > 0 gives each card at most N modified texts. They are
    picked from 100 fixed seeds per (variant, card, mod class), so they
    are cacheable.
- **`src/training/ENV_SIZING.md`** (2026-10-22): 512-card budget,
  mods off, warm cache: about 2,100 cards/s, 78% of the step waiting
  for the batch. Encoding runs at about 200 MTG cards/s. The whole
  corpus (65,626 texts) is 96 MiB at fp16.
- **MTG deck box:** 6,911,380 decks in `data/final/decks/mtg.db`
  (17lands limited decks); counts of cards per deck are in Method.

## Method

- **Terms.** An *arm* is one training run; arms differ only in the
  settings named below. *Training thread* and *worker* (the prefetch
  thread) as in the parent. *Anchor visit*: one card of one deck used
  once as an InfoNCE anchor; one `contrastive.mtg` step anchors every
  item card in its batch. *Hit rate*: the share of texts the training
  path asked the feature cache for that it did not have to encode,
  1 - Δencoded / Δtexts between samples (`FeatureCacheStats`).
  *Steady state*: the second half of an arm's training time.
- **Recipe held constant:** r2's recorded `multi_mixed` config
  (`../2026-10-11_model_by_diet_r2/configs/multi_mixed.yaml`):
  - multi-card model: frozen ModernBERT-base, token whitening
    (`data/whitening/modernbert_tokens_holdout_seed0.pt`), mean pooling,
    `residual_mlp` head (3 blocks, 512 hidden, 256 out), 2 group
    attention layers (4 heads, ffn 1024);
  - its 162 dojos (149 trained, 13 held out), its 50/50 table diet
    (contrastive styles vs metric sources), holdout seed 0, tiers 8/1/1,
    Gwent held out;
  - seed 0, fp16, Huber, `eval_examples_per_dojo` 128, encoder_lr 1e-4,
    head_lr 1e-3.
- **Changed from r2 in every arm** (the throughput settings):
  - `max_batch_cost` 512 (r2: 256); `prefetch_depth` 4 (r2: none);
  - feature cache: fp16 rows; VRAM 3 GiB, LFU (about 2.1M rows, so the
    whole corpus and every bounded variant fit); pinned RAM 4 GiB, LRU;
    a new SQLite disk file per arm (`write: warm_only`), so every arm's
    startup warms the same 7 games from cold;
  - `default_augmentation: {probability: 0.1, variants: N}`: every game
    default card-field mod (the weighted field mask and the random key
    mask) acts on about 10% of cards. Task mods (the metric dojos' target
    masks) are not augmentations and run in every arm.
  - The run is profiled (`scripts/run_training.py --profile`, commit
    38e80d8 or later; see Instrument).
- **Arms** (configs written by [`analysis/make_configs.py`](analysis/make_configs.py)):

  | arm | dojos | augmentation | role |
  |---|---|---|---|
  | `pilot` | mixed (162) | N=4 | 2 × 300 steps: checks the profiler, calibrates cost |
  | `mixed_n0` | mixed (162) | N=0 (off) | **control**: augmentation off |
  | `mixed_n4` | mixed (162) | N=4 | H1: bounded mods at scale |
  | `mtg_n0` | `contrastive.mtg` only | N=0 | the MTG anchor-pass rate, no mods (H3) |
  | `mtg_n4` | `contrastive.mtg` only | N=4 | the same with bounded mods (H3) |

  Step counts are set from the pilot (Log). The mixed arms have equal
  steps; so do the MTG arms.
- **Instrument** (new in `src`, see `plans/run_profiling.md` and the
  Log): `--profile` writes into `<run>/profile/`:
  - `steps.csv`: per step, the wait for the batch, forward + loss, the
    remainder, cards, and the training thread's time in mods,
    serialization, cache lookups and live encoding;
  - `builds.csv` and `reserves.csv`: per batch built (worker) and
    reserved in the cache, with the same stage split;
  - `evaluation.csv`: per TEST pass per dojo;
  - `marks.csv`: startup stages and round boundaries;
  - `samples.csv` (every 10 s): VRAM allocated, reserved and peak, the
    cache's totals and per-tier hits, rows, pins and evictions.
  - The experiment's Windows counter samplers add free RAM and the
    process's working set and dedicated VRAM (every 30 s), and GPU engine
    utilization and Python CPU use (every 15 s), in
    `data/runs/2026-10-24_scale_profile/logs/`.
- **The MTG projection.** `mtg.db` holds 6,911,380 decks with
  161,289,978 (deck, distinct card) rows (23.3 per deck) and 277,275,615
  copies (40.1 per deck); one-off scan,
  `data/runs/2026-10-24_scale_profile/mtg_deck_card_totals.txt`. A TRAIN
  deck shows only TRAIN-tier cards (the card tiers are a separate 8/1/1
  hash per card, `src/schema/holdout.py`), and the Unknown sentinel and
  cards the binder lacks are never anchors. So the visits are about 0.8
  (TRAIN decks) × 0.8 (TRAIN-tier cards, if held-out cards are as common
  in decks as any other) × 161.3M ≈ 103M; the upper bound, ignoring card
  tiers, is 129M. A sample of decks, scanned with the run's HoldoutSpec
  after the runs (`analysis/mtg_train_visits.py`), replaces the 0.8 card
  factor with a measured one. Hours = visits / the steady wall-clock
  anchor cards per second of the `mtg_*` arm. Today's single-card style samples 2 cards per deck per pass, so
  covering every card by random draws takes many more deck passes than
  12; the projection is the ideal (each anchor visited once), a lower
  bound for today's sampler.
- **Predictions:** H1-H3 in Aim. Also expected:
  - `mixed_n4` is slower than `mixed_n0` in its first hour only: about
    10% of draws need a modded text, at most 4 per card per mod, so
    at most about 600k distinct modded texts are encoded once (about
    50 min at 200 cards/s if all of them occur, in practice fewer);
  - no evictions in either arm (everything fits the VRAM tier);
  - the dojos slowest to build per batch (builds.csv, ms per card) are
    the single-card metric dojos over whole games and the replay dojos
    (parent). Startup is measured per stage only (build, preflight,
    model, warm); per-dojo startup times come only from the
    calibration lines in run.log, where they exist;
  - peak VRAM under 8 GiB (3 GiB slab + the model + steps).
- **Noise:** one seed per arm. The parent found timing of two seeds of
  one config within 2%; differences under 10% are "no clear
  difference". Distributions are reported as medians and p90.
- **Known confounds:**
  - Profiling synchronizes the GPU around the forward pass and live
    encoding; on the worker, a sync also waits for the trainer's queued
    kernels. Steps may be slightly slower than unprofiled.
  - The pilot ran commit d92e038, before the critic's fixes (38e80d8:
    the profile decides what is recorded, a reservation is recorded
    outside its try, a phase column). The fixes change what is written,
    not what is trained; the pilot is calibration only. While the pilot
    built its dojos, `./scripts/check.sh` ran (about 1 min of 4 CPU
    workers), so its startup time is slightly inflated.
  - The machine runs the usual desktop apps but is otherwise idle; the
    user's other workload was stopped before the experiment.
  - A fixed probability and variants also change the data each arm
    trains on; losses are not compared, only throughput.
- **Cost:** estimated after the pilot (Log). Budget from the human:
  about 4-5 h per mixed arm, about 10 h in all. Disk: about 2.5 GB per
  arm (checkpoints include ModernBERT) plus up to a 0.5 GB SQLite file
  each.

## Recipe

- Data: every metric parquet, split file and deck box r2's
  `multi_mixed` used; the whitening stats above.
- `python experiments/2026-10-24_scale_profile/analysis/make_configs.py`
  writes the configs.
- Queue: `data/runs/2026-10-24_scale_profile/queue.csv`; run with
  `bash experiments/2026-10-24_scale_profile/analysis/run_queue.sh` from
  the repo root (ROCm venv `G:\Projects\venvs\tcg-rocm`). Each run goes
  through `analysis/run_one.sh`: provenance (commit, git status, pip
  freeze, hardware), the two Windows counter samplers, and
  `PYTHONPATH=. python scripts/run_training.py <config> --profile`.
- Outputs: `data/runs/2026-10-24_scale_profile/<arm>/` (run directory,
  with `profile/`), `.../logs/` (console and sampler CSVs),
  `.../provenance/<arm>/`.
- `python experiments/2026-10-24_scale_profile/analysis/profile_report.py`
  writes `results/`.

## Log

### 2026-10-24 · Design brief from the human

What: the user handed over the workstation for a multi-hour,
industry-style profiling experiment and gave the PI experimental
authority while away. Asked for:
- two full runs: N=0 (augmentation mods off, task mods kept) and N=4
  variants, with mods firing on about 10% of cards;
- a smart eviction policy, good cache sizes and good batch sizes;
- logging and timing added "throughout";
- a new experiment branch.

Answers to the PI's questions before they left:
- branch off PR #14;
- timing code goes permanently into `src`;
- about 4-5 h per arm, about 10 h in all;
- add a `probability` to every augmentation mod.

### 2026-10-24 08:30-10:15 · Instrument and design (PI)

What:
- **Profiling code in `src`** (the human waived human skeleton review
  for this slice). Plan note: `plans/run_profiling.md` (ignored by git;
  summarized in Method). Went through `design-recipe-skeleton`; the
  file-critic pre-pass returned REVISE (6 points, all taken: rates that
  check themselves, declared spec fields, complete stage totals, GPU
  sync on entry and exit, one CSV writer, one time base). Commit
  d92e038. The full pass returned REVISE (3 points, all taken: the
  profile, not the global clock, decides what is recorded; profiling
  can never throw into training, including a reservation-pin leak if a
  record failed; a phase column). Commit 38e80d8. `./scripts/check.sh`:
  4,350 passed.
- **Augmentation rates:** every card-field mod takes a probability
  gate, rolled on the mod's own state before the variant is picked; the
  run-wide `default_augmentation` rewrites the game defaults' rates.
  With N=4 and probability 0.1 a card is modified on about 10% of its
  draws, into one of at most 4 texts per mod.
- **A 1-minute GPU smoke run** (Gwent, 2 × 60 steps, N=4, cache,
  prefetch 2) wrote every CSV; nothing was stamped uncacheable, as
  bounded variants should be.

Why these design choices:
- **Batch 512, prefetch 4:** the parent found the step per-card bound,
  so a bigger batch gains little (+9%), but it is what ENV_SIZING
  recommends and the multi-card model packs fewer real cards per padded
  slot. Depth 4 gives the single worker more room over slow steps.
- **LFU VRAM tier at 3 GiB:** the human asked for a smart eviction
  policy. At the pooled cut everything fits (about 2.1M rows against at
  most about 0.7M texts), so eviction should never trigger; the eviction
  counter checks that rather than assuming it. LFU keeps staples if a
  later, bigger corpus does not fit.
- **A new SQLite file per arm,** so both arms pay the same cold warm-up.
- **The MTG probes** (`mtg_n0`, `mtg_n4`) answer the at-scale question
  directly: in the mixed diet `contrastive.mtg` is only about 2% of
  steps, and its rate there is shaped by every other dojo sharing the
  worker.

### 2026-10-24 10:14 · Pilot launched

`pilot` (2 × 300 steps, N=4), commit d92e038, through the queue runner.

### 2026-10-24 10:25 · Design review (experiment-critic, design mode)

Stands in for the human's approval (non-interactive mode). No blocker
in the run code. The points and responses:
1. H3's visits ignored the per-card holdout tier (about 20% too high):
   Method now uses 0.8 × 0.8 × 161.3M ≈ 103M as the estimate, 129M as
   the upper bound, and a deck sample (`analysis/mtg_train_visits.py`,
   run after the arms) measures the card factor. H3 is "not
   established" if the bar falls inside the uncertainty.
2. H1's absolute 95% hit-rate bar mixed in task-mod and TEST misses:
   now relative to `mixed_n0` (within 3 points), and "last hour" was
   replaced by the steady state used everywhere else.
3. H2 named no arm: now `mixed_n0` in steady state.
4. GPU, CPU and RAM medians included startup: `profile_report.py` now
   clips the Windows counters to the steady training window.
5. Wait share over all steps: a steady-state version added.
6. The MTG projection used step time only: it now uses wall-clock
   cards per second over the steady window (TEST and gaps included);
   decks per card is measured from `builds.csv`, not assumed to be 1/2.
7. Per-dojo startup build time is not recorded: the prediction was
   reworded to batch build cost; startup is reported per stage.
8. Hourly hit rate lost the change across hour boundaries: now summed
   from sample-to-sample differences.
These edits were made before any main-arm result existed; the pilot
had produced no steps yet.

## Results

## Conclusions

## Open questions
