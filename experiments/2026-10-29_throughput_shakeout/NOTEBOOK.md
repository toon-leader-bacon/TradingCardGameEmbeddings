# Throughput shake-out: three games, baked key shuffle, snapshots

**Status:** awaiting review
**Dates:** 2026-10-29 -
**Parent:** 2026-10-24_scale_profile (notebook on branch `experiment/2026-10-24_scale_profile-notebook`, commit edec447) · **Branch:** `new_pokemon_decks` (commit 666ce89 for every run)

> With the Arrow training snapshots, baked binder mods and three games in
> one run, how fast does a contrastive-only run train once the cache is
> warm, what limits it, and what does that predict for a full pass over
> every TRAIN deck?

## Aim

**Problem (priority: the fix; this is a shake-out, smaller than a
training experiment and bigger than a smoke test).** The human wants a
rough estimate of how long a real contrastive pre-training run will take,
and the infrastructure exercised on several games at once before that
run. The last measurement (scale_profile `mtg_n0`, 2026-10-24, before the
snapshots) was about **2,150 cards/s**, with 71% of each step spent
waiting for the CPU to build the batch. Questions:

1. **Throughput:** steady-state cards/s (and decks/s) once the cache is
   warm, on current code.
2. **Bottleneck:** is the step bound by CPU batch building or by the GPU?
3. **Resources:** VRAM (torch peak and reserved, process dedicated), RAM,
   python CPU, GPU compute utilization.
4. **Mods:** what bounded dynamic mods (4 variants per mod) cost on top of
   a baked binder with no dynamic mods; how many variants the cache can
   hold.
5. **Batch size:** does a 2048-card budget raise cards/s?
6. **Infrastructure:** do binder mods (`shuffle_keys` at probability 1.0,
   baked and saved once), snapshots and three games train together
   without errors?

No bar: the output is measured numbers plus a projection for one pass
(1 sample per TRAIN deck) and 60 samples per deck.

## Background

- cache_shakeout (2026-10-21): with the feature cache warm and mods off,
  the step was per-card bound (512 → 1,024 cards/step gave +9% cards/s);
  default mods left most texts uncacheable (fixed since by bounded
  variants).
- scale_profile (2026-10-24, not merged): `mtg_n0` 2,150 cards/s and
  `mtg_n4` 1,841 cards/s on `contrastive.mtg` (single-card style, 2 cards
  per deck), 512 cards per step, 0.23 s per step, wait 71%. A step whose
  batch was ready took about 50 ms on the GPU (forward 38 ms + rest 13 ms),
  so the GPU ceiling is about 10,000 cards/s at this model.
- Since then (merged 2026-10-28): training snapshots stage 1 (Arrow card
  and deck snapshots, `DeckSplitIndex`, snapshot-backed dealer; plan
  `plans/training_snapshots.md` projected about 50 ms to build a 512-card
  batch, not measured), binder without deep copies, augmentation rates.
  Today: binder mods (commit 666ce89).
- Deck counts (deck snapshots, 2026-10-29): MTG 6,911,380; FaB 794,360
  (rebuilt today after the Talishar ingest); Dominion 531,675.

## Method

**Terms.** An *arm* is one configuration; each arm is one *run* (one
seed: a shake-out, not a result).

| arm | games | binder mods | dojo mods | `max_batch_cost` | steps |
|---|---|---|---|---|---|
| `control_mtg_n0` | MTG | none | off (variants 0) | 512 | 1 × 6,000 |
| `three_baked_n0` | MTG, FaB, Dominion | `shuffle_keys` p=1.0, baked | off (variants 0) | 512 | 4 × 2,000 |
| `three_baked_n4` | MTG, FaB, Dominion | same bake | game defaults (weighted field mask + random key mask 0.1), variants 4 | 512 | 4 × 2,000 |
| `three_baked_n0_b2048` | MTG, FaB, Dominion | same bake | off | 2048 | 4 × 500 |

- **Control.** `control_mtg_n0` is scale_profile's `mtg_n0` config
  unchanged (bar the paths), on current code: the comparison with
  2,150 cards/s measures what the code since 2026-10-24 changed.
  `three_baked_n0` is the control for `three_baked_n4` and
  `three_baked_n0_b2048`.
- **Held constant:** model (multi-card residual MLP head, group attention
  2 × 4 heads, mean pooling, whitening, ModernBERT-base frozen, about 3M
  trainable), feature cache tiers (VRAM 3 GiB LFU, RAM 4 GiB, disk
  SQLite warm-only), fp16, prefetch depth 4, holdout seed 0 tiers 8/1/1
  with Gwent held out, training seed 0, `eval_examples_per_dojo` 128,
  uniform diet, single-card contrastive style (`contrastive.<game>`).
  The whitening stats are scale_profile's (same card tiers).
- **Configs:** `configs/<arm>.yaml`, written by `analysis/make_configs.py`.
- **Recorded per run:** `rounds.csv` (step and elapsed time per round),
  `samples.csv` (torch VRAM and feature cache tiers every 10 s), `run.log`,
  plus the Windows samplers (`analysis/sample_*.ps1`: free RAM, python
  working set and dedicated VRAM every 30 s; GPU engine utilization and
  python CPU every 15 s) and provenance (`analysis/run_one.sh`).
- **Bottleneck without a profiler.** The per-step profiler from
  scale_profile was not merged, so the step is not split. Instead:
  `analysis/batch_build_bench.py` times each dojo's TRAIN batch building
  alone on the CPU (single thread, no model or cache), after the runs.
  If that time per batch is close to the step time and the GPU compute
  counter is well below 100%, the run is CPU bound; if it is well below
  the step time and the GPU is busy, it is GPU bound.

**Predictions (written before any run):**

- **P1 (snapshots help):** `control_mtg_n0` reaches 4,000-10,000 cards/s
  (stage 1's projected 50 ms build makes the GPU, at about 50 ms per step,
  the limit). At or below about 2,500 cards/s means stage 1 did not move
  the bottleneck.
- **P2 (bounded mods are cheap once warm):** `three_baked_n4`'s
  steady-state cards/s (rounds 2-4) is within 15% of `three_baked_n0`'s.
  Its first round is slower, because variant texts miss the cache and are
  encoded live.
- **P3 (batch size):** if the step is GPU bound per card,
  `three_baked_n0_b2048` gains at most 20% cards/s over `three_baked_n0`;
  a gain over 50% means a fixed per-step overhead dominates.
- **P4 (memory):** peak torch VRAM under 6 GiB in every arm; the cache
  holds about 40.7k rows in the n0 arms and at most about 16× that in n4
  (two mods, 4 variants each, plus the unmodded text); free RAM stays
  above 12 GB.

**Known confounds.** One seed per arm. The three-game arms share one disk
tier, so the first (`three_baked_n0`) pays the cold encode and the later
ones load from disk (startup only; training rates are compared in steady
state). Two data downloads (Limitless, HearthstoneTopDecks) keep running
in the background, rate-limited and network bound (a little CPU).
Whitening stats were fitted on clean cards, not the baked ones (harmless
for throughput).

**Cost.** About 30-45 min per arm including startup, about 2.5 h in all;
disk about 2.4 GB per run (checkpoints) plus a few hundred MB of cache.

## Recipe

- Data that must exist: card binders and deck boxes for MTG, FaB and
  Dominion; current snapshots (`scripts/build_training_snapshots.py
  --game flesh_and_blood` was re-run 2026-10-29 for the new FaB box);
  scale_profile's whitening stats
  (`data/runs/2026-10-24_scale_profile/mtg_n0/whitening_stats.pt`, or
  refit with `scripts/fit_whitening.py --holdout-from <config>`).
- Configs: `python experiments/2026-10-29_throughput_shakeout/analysis/make_configs.py`.
- Bake (optional, also happens in the first run):
  `PYTHONPATH=. python scripts/run_training.py configs/three_baked_n0.yaml --check`.
- Queue: `data/runs/2026-10-29_throughput_shakeout/queue.csv`, run with
  `bash experiments/2026-10-29_throughput_shakeout/analysis/run_queue.sh`
  (ROCm venv `G:\Projects\venvs\tcg-rocm`).
- Batch building: `PYTHONPATH=. python
  experiments/2026-10-29_throughput_shakeout/analysis/batch_build_bench.py
  <config> --batches 300`.
- Outputs: `data/runs/2026-10-29_throughput_shakeout/<arm>/`, logs and
  sampler CSVs in `.../logs/`, provenance in `.../provenance/<arm>/`.

## Results

Seven runs, all exit 0, one seed each (a shake-out: every comparison is
one run against one run). Per-run numbers: [`results/runs.csv`](results/runs.csv)
(`analysis/summarize_runs.py`); batch building:
[`results/batch_build_three_baked_n0.csv`](results/batch_build_three_baked_n0.csv),
[`..._n4.csv`](results/batch_build_three_baked_n4.csv); projection:
[`results/projection.csv`](results/projection.csv) (`analysis/projection.py`).
Cards/s is steady state (every round after the first, TEST included),
TRAIN cards = steps × `max_batch_cost`.

| run | cards/s | ms/step | vs `three_baked_n0` | python CPU | VRAM peak (GiB) | cache rows |
|---|---|---|---|---|---|---|
| `control_mtg_n0` | 5,181 | 98.8 | (MTG only) | 1,503% | 4.46 | 34,666 |
| `three_baked_n0` | 4,961 | 103.2 | - | 1,504% | 4.46 | 40,670 |
| `three_baked_n4` | 4,083 | 125.4 | -18% (rounds 2-4) | 1,491% | 4.53 | 103,430 |
| `three_baked_n0_b2048` | 6,035 | 339.4 | +22% | 1,235% | 3.90 | 40,670 |
| `three_baked_n0_omp2` | 6,471 | 79.1 | +30% | 346% | 3.72 | 40,670 |
| `three_baked_n0_omp1` | 6,667 | 76.8 | +34% | 224% | 3.72 | 40,670 |
| `three_baked_n0_omp2_b2048` | 5,941 | 344.8 | +20% | 375% | 3.90 | 40,670 |

Against the predictions:
- **P1 (snapshots help): met.** `control_mtg_n0` ran at 5,181 cards/s,
  inside 4,000-10,000; 98.8 ms per step against scale_profile `mtg_n0`'s
  238 ms (2.4×). Three games instead of one cost about 4% (4,961).
- **P2 (bounded mods cheap once warm): not met in this window.**
  `three_baked_n4` rounds 2-4 were 18% slower than `three_baked_n0` (the bar
  was 15%), and round 1 was 43% slower. Live encodes were still falling
  when the run ended (bench note), so the steady state was not reached in
  8,000 steps; the batch-build benchmark adds that the mods themselves cost
  about 15-19 ms per batch (MTG 27.8 → 46.5 ms).
- **P3 (batch size): at the bar, not a fixed overhead.** 2048 gave +22% at
  default threads (bar: at most 20% for per-card bound, over 50% for a
  fixed per-step overhead). With the thread count fixed it gave nothing:
  `omp2_b2048` 5,941 against `omp2` 6,471 at 512.
- **P4 (memory): met.** Torch peak 3.7-4.5 GiB, process dedicated VRAM at
  most 6.85 GB of 16; no evictions anywhere; free system RAM 11.4-12.4 GB
  during training (process working set 5.8-6.6 GB), so RAM, not VRAM, is
  the tighter resource on this desktop.
- **Infrastructure: worked.** Bake once (about 4 s for MTG), saved and
  reloaded by every later run with a matching stamp; snapshots and three
  games trained together; 0 uncacheable texts in every run.

Unplanned, from the free-form probes: **python CPU at about 1,500% was
mostly torch's intra-op threads, and they slowed training.**
`OMP_NUM_THREADS=1` gave +34% at 224% CPU.

**Where the time goes (no step profiler, so an inference).** With one OMP
thread a 512-card step takes 76.8 ms, and building the batch alone takes
22.7-31.2 ms on one core (no dojo mods). The prefetch worker should hide
that behind the step, but the step is about the batch build plus about
50 ms, which is what one GIL shared by the worker and the training thread
predicts. Per card, every run that fixed the threads lands near 0.15 ms
(76.8 ms / 512 = 0.150; 344.8 ms / 2048 = 0.168), so the remaining limit
scales with cards, not steps: per-card Python work (batch building,
serialization, cache lookups) on one core. The GPU's share cannot be
split out from these data; the Windows compute counter reads 2-25% and
does not track the ROCm kernels.

## Interim reading (for the human's review)

- **Projection** (`results/projection.csv`, at 6,667 cards/s, the best
  measured configuration): one sample per TRAIN deck takes about **0.55 h**
  for these three games (6.59M decks) and about **0.9 h** for about 10.4M
  TRAIN decks across every game; 60 samples per deck about **1.4 days**
  (three games) and **2.2 days** (every game); about 23 samples (every
  distinct card of an MTG deck once) about 0.5 and 0.8 days. With today's
  default threads, multiply by about 1.34; with 4-variant dojo mods, by
  about 1.2 more until the variant texts are cached. Startup: about 3.5
  min with a cold cache, about 30 s from a warm disk tier.
- **Cheapest fix found:** run with `OMP_NUM_THREADS=1` (or
  `torch.set_num_threads(1)` at startup). Not yet promoted: it is two
  short runs, and its effect on the TEST pass or on the metric dojos was
  not measured.
- **Batch size:** no reason to raise it for speed once threads are fixed.
  A larger batch also changes the contrastive task (more in-batch
  negatives), a training-quality question, not a throughput one.
- **Mod variants:** memory is not the limit. The VRAM tier (3 GiB) holds
  about 2.1M fp16 rows and held at most 103k; even 25 texts for each of
  the about 66k cards of every game (about 1.65M rows, about 2.5 GiB)
  fits, and the default 512 MiB tier would not. The cost is encoding:
  ModernBERT encodes about 200 cards/s, so one new variant text costs
  about as much time as about 33 training cards. The warm step encodes only
  the base texts (`three_baked_n4` warmed 40,670, none of the variants),
  so variants are paid for live, on the training path.
- **Next throughput work, by size of the prize:** threads (done by an env
  var); moving per-card Python work out of the GIL (a process-based
  prefetch worker, or index-native batches as in
  `plans/training_snapshots.md` stage 2); warming the bounded variants up
  front; and a step profiler to replace the inference above with a
  measurement.

## Open questions

- Does `OMP_NUM_THREADS=1` slow anything else (TEST, metric dojos,
  warming, multi-card styles)?
- How much of the remaining ~50 ms per step is GPU time? (Needs the step
  profiler, or `torch.profiler` on a few hundred steps.)
- Where does `three_baked_n4` settle once every variant it draws is cached?

## Log

### 2026-10-29 · Design (PI, with the human)

What the human asked for: a ~30 min probe per arm, not a full
experiment; MTG, FaB and Dominion together; contrastive only; the binder
mod pipeline set to shuffle keys at probability 1.0 and nothing else; two
mod arms, no dynamic mods vs dynamic mods bounded at 4 variants; uniform
diet ("orthogonal" to throughput); single-card style only.
Decisions made with the human:
- **Binder mods were a skeleton** (stubs on the training path, so the
  working tree could not train). The human chose to build them first:
  done through design-recipe-implement (file-critic APPROVED, small notes
  taken; `check.sh` all passed, 4,626 tests), committed with the rest of
  the working tree as 666ce89 for provenance.
- **Added by the PI:** `control_mtg_n0` (direct comparison with
  scale_profile) and `three_baked_n0_b2048` (the batch-size question).
- **FaB snapshot** rebuilt for the new deck box (794,360 decks, 84 s).

### 2026-10-29 13:05 · `--check` of `three_baked_n0`: bake and startup

`scripts/run_training.py configs/three_baked_n0.yaml --check` at 666ce89
(log: `data/runs/2026-10-29_throughput_shakeout/logs/check_three_baked_n0.log`):
exit 0 in 1 min 45 s, from launch to "Check passed" (scale_profile's
startup took about 19 min).
- **Bake:** MTG baked in about 4 s after launch, then FaB and Dominion as
  their dojos were built; all three saved under
  `data/cache/binder_mods/throughput_shakeout_shuffle_keys/` (binder,
  alias ledger, stamp each).
- **Dojos:** TRAIN decks: MTG 5,529,170, FaB 635,615, Dominion 425,658
  (6,590,443 in all); all three passed preflight. The dojo mods report
  "changed 0/512" as expected with `variants: 0`.
- **Model:** 2,895,616 trainable parameters.

### 2026-10-29 13:10 · Queue launched

Order: `three_baked_n0`, `three_baked_n4`, `three_baked_n0_b2048`,
`control_mtg_n0` (`data/runs/2026-10-29_throughput_shakeout/queue.csv`).
`batch_build_bench.py` runs after the queue, so it does not compete with
training for the CPU.
- 2026-10-29 13:22 · run three_baked_n0 done (exit 0, 17 min, about 5,000 cards/s) → [bench note](bench_notes/2026-10-29_1322_three_baked_n0_done.md)

### 2026-10-29 13:30 · PI: python CPU is saturated; a thread-count probe added

`three_baked_n0`'s python process used about 15 of 16 hardware threads
for the whole training window, and scale_profile's `mtg_n0` used about
13. Two readings predict different things:
- **Real batch-building work** (several threads building batches, or
  pyarrow and NumPy's own pools): fewer threads would make steps slower.
- **Spin-waiting** (torch's OpenMP intra-op threads busy-waiting while the
  training thread waits on the GPU or the next batch): fewer threads would
  leave steps as fast and the CPU lower.

**Probe:** `three_baked_n0_omp2`, `three_baked_n0` with `OMP_NUM_THREADS=2`
set in the environment (no config or code change), 2 rounds, queued last.
The control is `three_baked_n0` (5,000 cards/s, CPU 1,503%). Prediction:
within 10% of 5,000 cards/s with a much lower CPU means spin-waiting;
clearly slower means real CPU work, which the batch-build benchmark then
sizes.
- 2026-10-29 13:41 · run three_baked_n4 done (exit 0, 19 min; rounds 2-4 21% slower than n0, live encoding still falling) → [bench note](bench_notes/2026-10-29_1341_three_baked_n4_done.md)
- 2026-10-29 13:53 · run three_baked_n0_b2048 done (exit 0, 12 min, about 6,050 cards/s, +21%) → [bench note](bench_notes/2026-10-29_1353_three_baked_n0_b2048_done.md)
- 2026-10-29 14:06 · run control_mtg_n0 done (exit 0, 13 min, about 5,180 cards/s; 2.4× scale_profile's mtg_n0 per step) → [bench note](bench_notes/2026-10-29_1406_control_mtg_n0_done.md)
- 2026-10-29 14:12 · run three_baked_n0_omp2 done (exit 0, 6 min, about 6,470 cards/s, +29% with a quarter of the CPU) → [bench note](bench_notes/2026-10-29_1412_three_baked_n0_omp2_done.md)

### 2026-10-29 14:15 · PI: spin-waiting confirmed; two more thread probes

The prediction for `three_baked_n0_omp2` was "within 10% at a much lower
CPU means spin-waiting". It came out faster by 29% at a quarter of the
CPU: torch's intra-op threads were not doing useful work, and they slowed
the step, most likely by contending with the training thread and the
prefetch worker for cores. Two short probes to size the fix:
- **`three_baked_n0_omp1`** (`OMP_NUM_THREADS=1`, 2 rounds): does one
  thread gain more, or start to starve CPU-side tensor work?
- **`three_baked_n0_omp2_b2048`** (`OMP_NUM_THREADS=2`, `max_batch_cost`
  2048, 2 rounds of 500): do the thread fix and the batch-size gain add up?
- 2026-10-29 14:19 · run three_baked_n0_omp1 done (exit 0, 6 min, about 6,670 cards/s, CPU 224%) → [bench note](bench_notes/2026-10-29_1419_three_baked_n0_omp1_done.md)
- 2026-10-29 14:25 · run three_baked_n0_omp2_b2048 done (exit 0, 6 min, about 5,940 cards/s: no gain over default threads at 2048) → [bench note](bench_notes/2026-10-29_1425_three_baked_n0_omp2_b2048_done.md)
- 2026-10-29 14:35 · batch-build benchmark done (27 ms per 512-card batch without dojo mods, 42 ms with) → [bench note](bench_notes/2026-10-29_1435_batch_build_bench.md)
- 2026-10-29 14:56 · py-spy runs done (both exit 0 on the second attempt; no measurable overhead) → [bench note](bench_notes/2026-10-29_1456_pyspy_runs.md)

### 2026-10-29 15:00 · PI: py-spy replaces the GIL inference

The human asked for a measurement before choosing between fewer Python
objects (snapshot stage 2) and worker processes. What it shows:
- **The GIL is not saturated.** It is held about 48% of the time, so the
  two Python threads are not mainly waiting on each other. The Results
  section's "one GIL shared by worker and training thread" reading was
  wrong; it stays there as written, corrected here.
- **The worker is not the bottleneck either.** It is idle 35% of the time.
- **The training thread is, and half of its time is one function.**
  `clique_positive_mask` loops over every card index and reads and writes a
  torch tensor element by element. Each element access is a full torch
  operation (dispatch, a 0-d tensor, a bool conversion), mostly spent in
  C++ with the GIL released, which is why the GIL looks free while the
  thread is busy. It runs once per batch in the worker (to check the
  baseline is usable) and again in the training thread (loss weighting's
  baseline), on identical inputs.
- **Text serialization and hashing are small** (JSON 2.4% of the training
  thread, cache lookup 6%). The earlier suggestion that they dominate was
  wrong.
- **The thread result fits:** the 512 × 512 mask operations around the loop
  can be split over OpenMP threads, so with 8 threads each tiny operation
  paid for waking and spinning them.

Candidate fix (not applied): build `clique_of` as a Python list and convert
it once (or with NumPy), and compute each batch's baseline once and carry
it on the batch. If the main thread's 49% mostly disappears, the step
would fall from about 77 ms toward 40 ms. That would be a source change in
the middle of the experiment, so it is the human's call.
