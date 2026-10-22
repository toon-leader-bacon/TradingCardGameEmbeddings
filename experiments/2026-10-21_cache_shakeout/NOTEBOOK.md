# Shake-out of the frozen feature cache on a full mixed diet

**Status:** running
**Dates:** 2026-10-21 -
**Parent:** [2026-10-11_model_by_diet_r2](../2026-10-11_model_by_diet_r2/NOTEBOOK.md) (its `single_mixed` recipe) · **Branch:** `experiment/2026-10-21_cache_shakeout`

> With the frozen feature cache, prefetch and the vectorized contrastive
> loss in place, where does a full-diet training run now spend its time,
> and what should the next throughput and data work target?

## Aim

**Problem (priority: the fix, i.e. what to work on next).** Training
throughput was the stated bottleneck of series 1 (r2's `single_mixed`:
about 1.3 s per step including TEST, 478 min for 20,000 steps). The
feature cache work (plan `plans/frozen_feature_cache.md`, v1-v1.2) was
meant to remove the frozen ModernBERT pass from most steps. This
experiment measures, on a realistic run rather than a contrastive-only
microbenchmark:

1. **Where a step's time goes**, split into waiting for the batch,
   forward + loss, backward + optimizer, and within those: live encoding
   of cache misses, cache lookups, serialization, mods and CardBinder
   deep copies. Split by dojo family (contrastive styles, metric dojos
   by source).
2. **How much the default mods cost**: the share of cards encoded live
   and the time that takes.
3. **Batch building cost for metric dojos**, per card, against the
   contrastive dojos' (0.27 ms per card, `src/training/ENV_SIZING.md`).
4. **CardBinder's deep copy:** its share of build time, and whether
   training mutates any binder card when the copy is skipped.
5. **The TEST pass's cost**, per round and per dojo, now that steps are
   cheap.
6. **Cached against uncached**, same seed and steps: speed, and whether
   TEST losses match (the cheap half of `experiments/TODO.md` D).
7. **Startup**: dojo building, preflight, model build, cache warming.

There is no bar: this is a measurement. Its answer is a ranked list of
the costs, with a size for each, which decides the next engineering
work and which data to prioritise.

**Hypotheses** (predictions, second priority):
- **H1 (mods).** With the default mods on, live encoding of
  do-not-cache cards is the largest single cost of a cached training
  step. *Refuted if* the training thread's live-encode time is under
  30% of its step time in `cached`, or `cached_nomods` is less than
  1.3x faster per card than `cached` (per card, since removing mods can
  change batch contents). Do-not-cache texts are encoded only on the
  training thread: the worker's reserve skips them
  (`PooledFeatureCache.reserve`), so encodes during reserve are cold
  misses of cacheable texts, reported apart.
- **H2 (TEST).** At r2's `eval_examples_per_dojo: 128` over 162 dojos,
  one TEST pass now costs more wall-clock time than 1,000 cached
  training steps. *Refuted if* a round's TEST pass takes less than its
  1,000 training steps.
- **H3 (deep copy).** CardBinder's deep copies are 10% or more of TRAIN
  batch building time, and skipping them changes no binder card.
  *Refuted* on the cost if under 5%; *on safety* if any binder card
  serializes differently after training (`binder_check.txt`). Losses are
  not used for H3: `cached_nocopy` is also H4's noise yardstick, so it
  can't be judged against itself.
- **H4 (cache equivalence).** The cached run's TEST losses after 2,000
  steps match the uncached run's within run-to-run noise (median
  absolute difference of normalized TEST loss across dojos no larger
  than between `cached` and `cached_nocopy`, or than 0.02 if that pair
  is closer, since it may be bit-identical), and the cached run is at
  least 3x faster per training step. The cache stores fp16 rows while
  `uncached` uses the live fp16-autocast output, so a small difference
  is expected even with no bug.

## Background

- r2's resource table
  ([`resources_by_cell.csv`](../2026-10-11_model_by_diet_r2/results/resources_by_cell.csv)):
  `single_mixed` took 478 min (startup 30 min, rounds of 111-114 min for
  5,000 steps + TEST), GPU compute busy a median 68%, Python about 2.7 of
  16 threads, peak VRAM 5.0 GB, RAM 4.1 GB. Commit ced7ba7: no feature
  cache, no prefetch, the per-anchor InfoNCE loop.
- Since then (commits 951302a, 3f69a78, f4872cf, 3595994): the pooled
  feature cache with VRAM, RAM and SQLite tiers, prefetch with cache
  reservation, a vectorized InfoNCE loss, and contrastive batches sized
  by the card budget alone.
- Microbenchmarks, contrastive only (`src/training/ENV_SIZING.md`):
  warm cache, mods off, 512 cards: 182 ms per step, nearly all of it
  waiting for the CPU to build the batch (about 0.27 ms per card). With
  the default mods, 40% of cards were encoded live and the step took
  642 ms. `copy.deepcopy` of one card: about 45 us (2026-10-21,
  `src/dojos/TODO.md`).
- r2's zero-shot contrastive analysis found contrastive VALIDATION
  passes slow on the CPU (8-10 min per encoder for 30 dojos), not yet
  profiled (`src/dojos/TODO.md`).

## Method

- **Terms:** a *run* is one training run under the profiler; runs differ
  only in the settings named below. *Training thread* is the trainer's
  thread; *worker* is the prefetch thread that builds and reserves
  batches. *Live encoding* is the frozen stages (ModernBERT, whitening,
  mean pooling) run on texts not served by the cache: do-not-cache
  modded texts, misses, or every text without a cache.
- **Recipe held constant** (r2's recorded `single_mixed` config,
  `../2026-10-11_model_by_diet_r2/configs/single_mixed.yaml`): single-card
  model, frozen ModernBERT-base, token whitening
  (`data/whitening/modernbert_tokens_holdout_seed0.pt`), mean pooling,
  `residual_mlp` head (9 blocks, 512 hidden, 256 out); the same 162
  dojos (149 trained, 13 held out) and 50/50 table diet; holdout seed 0,
  tiers 8/1/1, Gwent held out; seed 0; `max_batch_cost` 256; fp16;
  Huber; `eval_examples_per_dojo` 128; encoder_lr 1e-4, head_lr 1e-3.
- **Changed from r2 in every run:** 2 rounds of 1,000 steps (TEST after
  each), and `hardware.prefetch_depth: 2`.
- **Runs** (configs in [`configs/`](configs/), written by
  [`analysis/make_configs.py`](analysis/make_configs.py)):

  | run | feature cache | default mods | CardBinder copy | role |
  |---|---|---|---|---|
  | `pilot` | yes | yes | yes | 2 × 100 steps; checks the profiler, calibrates cost |
  | `cached` | yes | yes | yes | **control**: the new default way to train |
  | `cached_nomods` | yes | off (every dojo `mods: []`; metric task mods stay) | yes | H1 |
  | `cached_nocopy` | yes | yes | skipped (profiler flag) | H3; and, since it computes the same thing as `cached`, the run-to-run noise for H4 |
  | `uncached` | no | yes | yes | H4 |

  Feature cache in the cached runs: fp16 rows; VRAM 2 GiB LRU; RAM
  4 GiB; SQLite disk tier `data/feature_cache/shakeout_2026-10-21.sqlite`
  (a new file, so the first cached run measures a cold warm-up and later
  ones a disk refill), `write: warm_only`; every card of the 7 games
  warmed before training.
- **Instrument:** [`analysis/profiled_training.py`](analysis/profiled_training.py)
  runs `scripts/run_training.py` unchanged, with timers wrapped around the
  hot paths (no source change). Per step: wait for the batch, forward +
  loss, backward + optimizer, and the live-encode, cache-lookup,
  serialization, mod and deep-copy time spent on that thread during
  the step. Per built batch (worker): build time with its mod and
  deep-copy time, and reserve time with the texts it encoded. Per TEST
  dojo per round: time, examples, cards and the same split. Startup
  stages. torch VRAM every 10 s. RAM, VRAM and GPU engine use from the
  Windows counters, as in r2 (`sample_*.ps1` in
  `data/runs/2026-10-21_cache_shakeout/`).
- **Predictions:** H1-H4 in Aim. Also expected, from ENV_SIZING: batch
  building at about 0.27 ms per card for contrastive dojos; the cache
  avoids nearly all encodes of unmodded cards (it is exact on the
  warmed corpus).
- **Noise:** one seed per run. `cached` and `cached_nocopy` should
  compute the same batches (same seeds; the copy changes no value if H3
  holds), so their TEST-loss difference is the run-to-run noise
  (GPU nondeterminism) that H4 is judged against. Timing noise: the
  per-step distributions are reported as medians and p10/p90, and
  differences under 10% are "no clear difference".
- **Known confounds:**
  - The profiler synchronizes the CUDA stream around the stages it times.
    The training thread already synchronizes once per step, so the cost
    is small there; on the worker it blocks only the worker. The total
    step rate may still be slightly lower than an unprofiled run's.
  - The first cached run (`pilot`) warms the cache from scratch; every
    later cached run refills it from disk, so their startup times differ
    by design.
  - Two rounds of 1,000 steps cannot show training quality; losses are
    compared only between these runs, never to r2's 20,000 steps.
  - The desktop is otherwise idle overnight, but not isolated.
  - `backward_opt_s` in `steps.csv` is the step's remainder (total minus
    wait minus forward + loss): backward, clipping, the optimizer, the
    end-of-step sync and bookkeeping. Results call it "remainder".
  - Cached rows are stored as fp16 (see H4).
  - The whitening stats path differs from r2's recorded one, but the
    files are byte-identical (checked with `cmp`).
- **Cost:** startup about 30 min per run (r2). Step time unknown; the
  pilot calibrates it. Guess: cached about 0.3-0.6 s per step, uncached
  1-2 s, TEST 10-20 min per pass. So about 1-1.5 h per run, 5-6 h for
  the queue, inside the ~9 h window. Disk: about 2.4 GB per run
  (checkpoints include ModernBERT), about 12 GB.
- **Free-form follow-ups,** if time remains, chosen from what the
  profile shows (logged as they are decided): for example a larger
  `max_batch_cost`, a deeper prefetch, or a lower
  `eval_examples_per_dojo`.

## Recipe

- Data: every metric parquet and split file r2's `single_mixed` used;
  the whitening stats above.
- `python experiments/2026-10-21_cache_shakeout/analysis/make_configs.py`
  writes the configs.
- Each run: `bash data/runs/2026-10-21_cache_shakeout/run_one.sh <run>`
  from the repo root (ROCm venv `G:\Projects\venvs\tcg-rocm`). It records
  provenance (commit, git status, pip freeze, hardware), starts the
  resource samplers, and runs
  `PYTHONPATH=. python experiments/2026-10-21_cache_shakeout/analysis/profiled_training.py --profile-out data/runs/2026-10-21_cache_shakeout/profile/<run> [--no-binder-copy] -- <config>`.
  The run queue is `data/runs/2026-10-21_cache_shakeout/queue.csv`.
- Outputs: `data/runs/2026-10-21_cache_shakeout/<run>/` (the run
  directory), `.../profile/<run>/` (profiler CSVs), `.../logs/` (console
  and sampler CSVs), `.../provenance/<run>/`.

## Log

### 2026-10-21 21:30 · Design

What: the design above, written by the PI without a live discussion. The
human set the questions (shake-out of the new caching, r2-like config,
1-2k steps, timing by dojo type, resources, the five questions in Aim)
and an overnight window of about 9 h of GPU, and asked for an
autonomous plan with free-form profiling afterwards.
Why: throughput work is engineering, but the user asked for it to be
recorded as an experiment with notes and critical reading as results
come in.
Decisions:
- 2 × 1,000 steps rather than 1 × 2,000, so the TEST pass is measured
  twice.
- Prefetch depth 2 everywhere (the new protocol); r2 had none.
- The deep-copy question is answered with a profiler flag, not a code
  change, and a binder fingerprint check before and after training.
- `cached_nocopy` doubles as the noise run for the cache comparison.

### 2026-10-21 21:34 · Pilot launched

`pilot` (2 × 100 steps), commit a8d7063. After launch the profiler gained
two columns in `builds.csv` (`examples`, `pass_id`, to count passes over
each dojo's TRAIN data); the pilot ran the version without them.

### 2026-10-21 22:00 · Design review (experiment-critic, design mode)

What: one design-mode pass, before `cached` launched. Its eight points
and the responses:
1. H4's noise pair may be bit-identical, and fp16 row storage was not
   named: added a 0.02 absolute fallback and the confound.
2. H3's loss half was circular (`cached_nocopy` is also H4's noise
   yardstick): dropped; H3's safety is the binder check alone.
3. Which thread encodes modded texts: only the training thread (the
   worker's reserve skips stamped texts). Stated under H1.
4. The profiler change after the pilot launched was uncommitted:
   committed as d099a95 (with `analysis/profile_report.py`). `pilot` ran
   a8d7063's profiler; every later run runs d099a95's or later.
5. The whitening path differs from r2's: byte-identical; noted.
6. `cached_nomods` changes the data: H1's speed ratio is per card.
7. Queue order puts H4 at risk: decided after the pilot calibrates.
8. `backward_opt_s` is a remainder: named so under confounds.
