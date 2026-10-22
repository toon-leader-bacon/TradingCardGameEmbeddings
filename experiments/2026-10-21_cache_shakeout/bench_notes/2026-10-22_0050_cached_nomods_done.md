# 2026-10-22 00:50 · run cached_nomods done

**Run:** cached_nomods · **Config:** `configs/cached_nomods.yaml` (now the recorded `run_config.yaml`) · **Exit:** 0 · **Duration:** 44 min, 23:56:45 to 00:40:37 (estimate 65–75 min)

Same as `cached`, except no dojo applies mods (`mods: {…: []}`). That
makes every card cacheable. Numbers come from
[`results/run_summary.csv`](../results/run_summary.csv),
[`results/step_by_family.csv`](../results/step_by_family.csv) and
[`results/test_by_family.csv`](../results/test_by_family.csv), rows
`cached_nomods`, and from `results/data_exposure_cached_nomods.csv`. The
`cached` values in brackets come from the same files.

## Key numbers

- **Startup: 32 min.** `build_dojos` took 1,856 s (1,813), preflight
  65 s, and the cache warm-up 13 s.
- **Steps (2,000):**
  - Mean 0.255 s, median 0.205 s, p90 0.419 s (1.22 / 1.01 / 1.96).
  - **4.8× faster per step**, with 965 cards/s (202).
  - About 14,100 steps per hour (2,956).
  - Training plus TEST took 11.3 min (43 min).
- **Where a step's time goes now:**
  - forward + loss: 53% (93%)
  - remainder (backward, optimizer): 34% (7%)
  - **batch wait: 13% (0.6%)**
  - feature lookup: 18% of the step, all from cache hits, with no live
    encoding
  - serialization: 0.9%
- **Cache:** 1,003,344 texts looked up and 21,434 encoded. 97.9% of
  lookups were avoided, and none were uncacheable. The VRAM tier held
  87,060 rows. The RAM tier again had 0 hits.
- **Batch building (worker):** 0.28 ms per card (0.33). Mods were 0.2% of
  it and deep copies 34%.
- **TEST pass:** 85 s in round 0 and 71 s in round 1, unchanged from
  `cached`. One pass now costs about **280–330 steps**.
- **GPU busy:** median 21%, p90 25% (61% / 77%). Python CPU median was
  383% (321%), from 21 samples.
- **Memory:** peak Python RAM 7.9 GB, and free RAM never fell below 13.1 GB.
  Torch peaked at 3.7 GiB allocated and 6.5 GiB reserved. Process VRAM
  peaked at 6.9 GB.

## Where the data is

- Run directory: `data/runs/2026-10-21_cache_shakeout/cached_nomods/`
- Profiler output: `data/runs/2026-10-21_cache_shakeout/profile/cached_nomods/`
- Console log and samplers: `data/runs/2026-10-21_cache_shakeout/logs/cached_nomods.*`
- Provenance: `data/runs/2026-10-21_cache_shakeout/provenance/cached_nomods/`

## First observations

- **With every card cached, the GPU sits mostly idle.** It is 21% busy.
  The time goes to the backward pass and optimizer, to waiting for
  batches, and to fixed per-step overhead.
- **The batch wait comes from card-field metric dojos.** Their median
  wait per step:

  | dojo family | median wait |
  |---|---|
  | `scryfall` | 840 ms |
  | `pokemon_tcg` | 677 ms |
  | `cardvault_fabtcg` | 288 ms |
  | `hearthstonejson` | 270 ms |
  | `dominiontabs` | 211 ms |
  | `spire_codex` | 177 ms |
  | contrastive and deck-level metric dojos | 0.3–40 ms |

  - The wait is twice as large in round 0 as in round 1 (44 s vs 23 s).
  - A likely cause: the worker's reserve step encodes these dojos'
    target-masked texts live, because the warm-up did not include them,
    and the training thread waits on the reserve. This fits the round 0
    TEST pattern from `cached`, but it is not yet confirmed from
    `builds.csv`.
- **The slowest dojo is the same as in `cached`.** The
  `seventeenlands_replay_data` family has a median step of 912 ms with
  no live encoding, the slowest in this run. Its forward pass is
  323 ms, against 30–240 ms for the other families. That fits long
  serialized texts in `attacker_blocker_combat_outcome`.
- **Startup is now 73% of this run's wall time** (32 of 44 min), and a
  single TEST pass costs as much as about 300 steps. At this speed, the
  run's fixed costs are startup and TEST, not training.
- **Loss comparison is not done yet.** The cached vs nomods TEST losses
  are in `results/loss_pairs.csv`. They are only context for H4, since
  this pair changes the training data. They are not read here.

## Resources and timing

`uncached` started at 00:40, estimated at about 1 h 45 min. Then comes
`cached_nocopy`, about 75 min. Queue ETA is about 03:40.
