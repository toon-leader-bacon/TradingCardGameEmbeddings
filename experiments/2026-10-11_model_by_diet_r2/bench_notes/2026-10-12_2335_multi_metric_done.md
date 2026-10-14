# 2026-10-12 23:35 · run multi_metric done

**Run:** multi_metric (batch 2, run 2 of 4) · **Config:** `configs/multi_metric.yaml` (now the recorded `run_config.yaml`) · **Exit:** 0 · **Duration:** 511 min, 14:58:32 to 23:29:32 (estimate 520)

## Key numbers

Normalized TEST loss, median over dojos, with the interquartile range
([`results/test_loss_by_round.csv`](../results/test_loss_by_round.csv)):

| kind | n dojos | round 0 | round 3 | IQR, round 3 | below 1.0, round 3 | max, round 3 |
|---|---|---|---|---|---|---|
| trained metric (1.0 = mean predictor) | 124 | 1.030 | 0.936 | 0.770-1.041 | 77 | 2.43 |
| held out (1.0 = mean predictor) | 13 | 1.109 | 1.111 | 1.045-1.206 | 0 | 1.87 |

- There are no contrastive dojos in this cell: 124 metric plus the 13
  held out.
- No trained metric dojo went above 5× in any round (max 2.43).
- `checkpoints.csv` marks round 3 (step 19,988) as the best checkpoint,
  so `train_round0003/` and `latest/` hold the same final weights.
- 2,895,616 trainable parameters (config check).
- The probe was not run; it comes after all the cells.

## Where the data is

- Raw data: `data/runs/2026-10-11_model_by_diet_r2/multi_metric/` (2.8 GB)
- Console capture: `data/runs/2026-10-11_model_by_diet_r2/logs/multi_metric.log`
- Project log: `data/runs/2026-10-11_model_by_diet_r2/multi_metric/run.log`
- Provenance: `data/runs/2026-10-11_model_by_diet_r2/provenance/multi_metric/` (commit 80877a5, clean tree; the code matches 467d0e1, see the [incident note](2026-10-12_1500_incident_launcher_rerun.md))
- Samples: `logs/multi_metric.resources.csv` (60 s), `logs/multi_metric.utilization.csv` (30 s)

## First observations

- The trained-metric median dropped below 1.0 from round 1 on and fell in
  every round. The IQR narrowed from the top (q75 1.139 to 1.041), and
  the worst dojo stayed between 2.1 and 2.4.
- The held-out dojos were flat at 1.11 in all four rounds, with 0 of 13
  below 1.0 by round 2. Their worst case stays under 1.9.
- Stable: 12 fp16 overflow steps were skipped, with no NaNs and no errors.
  The only warnings were the three usual startup ones (HF Hub, ModernBERT
  load report, SDPA kernel).

## Resources and utilization

The first run sampled from start to finish.

- **Memory:** peak Python RAM 3.9 GB, lowest free system RAM 16.0 GB.
  Peak Python VRAM 3.6 GB, with 4.7 GB in use on the 16 GB card
  (494 samples).
- **GPU:** in training (after startup, 903 samples), the compute engine
  was busy a median of **76%** of the time (p10 57%, p90 89%, max 95%).
  The 3D and copy engines were idle (0% and 0.3%).
- **CPU:** Python used a median of **280%** of one core (p10 263%, p90
  292%), about 2.8 of 16 threads, and very steady.
- **Reading this:**
  - VRAM is less than a quarter used.
  - The GPU's compute engine has some idle time, about a quarter in the
    median.
  - The CPU side is flat at about 2.8 threads, which fits a mostly
    single-threaded Python loop (batch building, dojo dispatch) that
    the GPU waits on part of the time.
  - Windows' "utilization" counts the time the engine has any work. It
    doesn't measure how full each kernel is, so 76% busy can still mean
    small, under-filled kernels.
  - Both readings point the same way: larger batches (`max_batch_cost`),
    or moving batch building off the main thread, could raise
    throughput. This experiment can't test that, since all cells hold
    `max_batch_cost` 256. It is a candidate for a short throughput pilot.

## Timing

- Startup: 26 min (14:58:32 to 15:24:19).
- Rounds, each 5,000 steps plus TEST: 128, 120, 118, 119 min.
- Next run: `single_contrastive`, launched 23:29:52, estimate 400 min,
  so ETA about 06:10. After it, `single_metric`, estimate 520 min, so
  queue ETA about 14:55 on 2026-10-13. Single-card step time is
  unmeasured.
