# 2026-10-22 04:45 · run cached_seed1 done

**Run:** cached_seed1 · **Config:** `configs/cached_seed1.yaml` (now the recorded `run_config.yaml`) · **Exit:** 0 · **Duration:** 75 min, 03:22:43 to 04:38:10 (estimate about 75 min)

This is `cached` with training seed 1. The holdout seed stays 0, so the
splits are the same. It is a free-form run, added at 02:50 (PI Log). The
numbers come from [`results/run_summary.csv`](../results/run_summary.csv)
and [`results/loss_pairs.csv`](../results/loss_pairs.csv). Values in
brackets are from `cached`.

## Key numbers

- **Startup:** `build_dojos` took 1,857 s (1,813).
- **Steps:**
  - mean 1.20 s (1.22), median 1.01 s (1.01)
  - live encoding took 82.6% of a step (82.4%)
  - TEST took 86 s in round 0 and 71 s in round 1 (87 / 70).
  - The timing reproduces within 2%.
- **TEST loss gaps, last round.** Each entry is the median absolute
  normalized difference over 162 dojos (contrastive / metric in
  brackets):

  | pair | median gap |
  |---|---|
  | `cached` vs `cached_seed1` (seed only) | 0.056 (0.004 / 0.079) |
  | `cached` vs `cached_nocopy` (no change to the computation) | 0.035 (0.011 / 0.041) |
  | `cached` vs `uncached` | 0.032 (0.0075 / 0.034) |
  | `uncached` vs `cached_seed1` | 0.053 |
  | `cached_nocopy` vs `cached_seed1` | 0.068 |

- **Memory:** peak Python RAM 8.0 GB. Peak process VRAM 6.9 GB.

## Where the data is

- Run directory: `data/runs/2026-10-21_cache_shakeout/cached_seed1/`
- Profiler output: `data/runs/2026-10-21_cache_shakeout/profile/cached_seed1/`
- Logs: `data/runs/2026-10-21_cache_shakeout/logs/cached_seed1.*`
- Provenance: `data/runs/2026-10-21_cache_shakeout/provenance/cached_seed1/`

## First observations

- **Noise sets the scale for the cache comparison.**
  - Two runs with identical computation and the same seed differ by
    about 0.035.
  - A seed change adds to that: about 0.05–0.07.
  - The cached vs uncached gap (0.032) is at the bottom of this range.
- **Most of the noise is in the metric dojos.**
  - Metric pairs differ by 0.04–0.08 at the median, with p90 up to 0.6.
  - Contrastive pairs differ by 0.004–0.011.
  - A single 2,000-step run cannot rank configurations by metric-dojo
    TEST loss unless the gap is well above about 0.08.

## Resources and timing

The planned queue and the free-form run are done (queue empty at
04:38). About 1 h 50 min of the GPU window remains.
