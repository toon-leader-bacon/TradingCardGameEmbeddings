# 2026-10-22 03:30 · run cached_nocopy done

**Run:** cached_nocopy · **Config:** `configs/cached_nocopy.yaml` (now the recorded `run_config.yaml`) · **Profiler flag:** `--no-binder-copy` · **Exit:** 0 · **Duration:** 52 min, 02:30:34 to 03:22:42 (estimate about 75 min)

`cached_nocopy` is the same as `cached`. The only difference is that the
profiler replaces `copy` in `card_binder` with a shim, so CardBinder
returns its stored cards instead of deep copies. The binders' cards are
fingerprinted before and after training.

Sources:
- `profile/cached_nocopy/binder_check.txt`
- [`results/run_summary.csv`](../results/run_summary.csv)
- [`results/test_by_family.csv`](../results/test_by_family.csv)
- [`results/loss_pairs.csv`](../results/loss_pairs.csv)

Values in brackets are from `cached`.

## Key numbers

- **`binder_check.txt`:** 7 binders checked, 0 changed. Result: unchanged.
- **Startup:**
  - `build_dojos` took 514 s, against 1,813 s, about 72% less (−1,299 s).
    In `cached`, `cached_nomods` and `uncached`, `build_dojos` took
    1,813–1,856 s.
  - Preflight took 48 s (70 s), and the warm-up took 18 s.
- **Steps:**
  - mean 1.20 s (1.22), median 0.99 s (1.01)
  - live share 82.5% (82.4%)
  - So no measurable change on the training thread.
- **Batch building on the worker:**
  - 0.22 ms per card (0.33), about 32% less.
  - Deep copies take 0.2% of build time (29%).
  - The worker's total is 108 s (160 s). It was already hidden behind
    the prefetch.
- **TEST pass:** 64 s in round 0 and 48 s in round 1 (87 / 70 s),
  about 30% faster.
  - The deck-level metric families gained the most in round 1:
    - `isotropic`: 10.5 s (16.1)
    - `sts2_runs`: 8.4 s (13.2)
    - `final_decks`: 4.1 s (6.7)
    - `seventeenlands_game_data`: 2.2 s (4.4)
- **TEST loss vs `cached`:** across all 162 dojos in the last round,
  the median absolute normalized difference is 0.035. The p90 is 0.26
  and the max is 2.93. For contrastive dojos the median is 0.011; for
  metric dojos it is 0.041.
  - So the two runs are not bit-identical: training is not
    deterministic between runs here.
- **Memory:** peak Python RAM 7.75 GB (7.89). Peak process VRAM 7.55 GB.

## Where the data is

- Run directory: `data/runs/2026-10-21_cache_shakeout/cached_nocopy/`
- Profiler output: `data/runs/2026-10-21_cache_shakeout/profile/cached_nocopy/`.
  This includes `binder_check.txt`.
- Logs: `data/runs/2026-10-21_cache_shakeout/logs/cached_nocopy.*`
- Provenance: `data/runs/2026-10-21_cache_shakeout/provenance/cached_nocopy/`

## First observations

- **Removing the deep copies mostly helps startup, not training.**
  - They cost about 22 of the 30 startup minutes.
  - They cost about 30% of each TEST pass.
  - They cost about a third of worker build time.
  - Training-step time does not change, because live encoding
    dominates it and the worker is already hidden.
- **No binder card was mutated in this run.** That covers the code
  paths this run exercises: default mods, the training dojos, TEST and
  validation. It does not prove that no other caller mutates cards.
- **The `cached` vs `cached_nocopy` loss gap (0.035) is about the same
  size as `cached` vs `uncached` (0.032).** That makes run-to-run
  noise around 0.03 at this budget. `cached_seed1` will add a second
  noise reading.

## Resources and timing

`cached_seed1` started at 03:22 and is estimated at about 75 min, so it
should finish about 04:40. After that the planned queue is empty.
