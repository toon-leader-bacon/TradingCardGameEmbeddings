# 2026-10-12 15:05 · run multi_contrastive done

**Run:** multi_contrastive (batch 2, run 1 of 2) · **Config:** `configs/multi_contrastive.yaml` (now the recorded `run_config.yaml`) · **Exit:** not recorded; training completed normally (see [incident](2026-10-12_1500_incident_launcher_rerun.md)) · **Duration:** 399 min, 08:18:28 to 14:57:01 (estimate 480)

## Key numbers

Normalized TEST loss, median over dojos, with the interquartile range
([`results/test_loss_by_round.csv`](../results/test_loss_by_round.csv)):

| kind | n dojos | round 0 | round 3 | IQR, round 3 | below 1.0, round 3 | max, round 3 |
|---|---|---|---|---|---|---|
| contrastive (1.00 = chance) | 25 | 0.649 | 0.532 | 0.423-0.795 | 25 | 1.00 |
| held out (1.0 = mean predictor) | 13 | 1.137 | 1.134 | 1.033-1.232 | 1 | 2.52 |

- There are no trained metric dojos in this cell: 25 contrastive plus the
  13 held out.
- `checkpoints.csv` marks round 2 (step 14,996) as the best checkpoint,
  and `train_round0002/` is kept. Round 3's weights, the final ones, are
  in `latest/`. For the probe, "final checkpoint" means `latest/`.
- 2,895,616 trainable parameters, the same as `multi_mixed`.
- The probe was not run; it comes after all the cells.

## Where the data is

- Raw data: `data/runs/2026-10-11_model_by_diet_r2/multi_contrastive/` (2.4 GB)
- Project log: `data/runs/2026-10-11_model_by_diet_r2/multi_contrastive/run.log` (complete)
- Script log: `data/runs/2026-10-11_model_by_diet_r2/logs/multi_contrastive.script_log_copy.log` (complete, INFO level)
- Console capture: `data/runs/2026-10-11_model_by_diet_r2/logs/multi_contrastive.log`, overwritten (see the incident)
- Provenance: `data/runs/2026-10-11_model_by_diet_r2/provenance/multi_contrastive/` (commit 4d97bdf, clean tree)

## First observations

- Contrastive TEST fell in every round, and fastest in rounds 1 to 2. By
  round 3 all 25 dojos were below chance; the worst was exactly at 1.00.
- The held-out dojos stayed flat at about 1.13 (1 of 13 below 1.0 in
  every round). Training only on contrastive did not move the
  metric heads of the held-out fields.
- Stable: 6 fp16 overflow steps were skipped, with no NaNs and no errors.
  1,554 "Skipping deck" warnings, about twice `multi_mixed`'s 810, as
  expected with twice the contrastive steps.

## Resources and timing

- Startup: 3 min (08:18:28 to 08:21:26). There are no metric dojos to
  calibrate, against 26 min for `multi_mixed`.
- Rounds, each 5,000 steps plus TEST: 99, 99, 100, 98 min.
- **Resource use, partial.** The sample files were lost (see the
  incident). What remains are two readings taken during the run and
  quoted to the human at the time:
  - 09:00, from `resources.csv`: peak Python RAM 1.9 GB, peak Python VRAM
    2.3 GB (3.0 GB in use on the 16 GB card), lowest free system RAM
    19.1 GB;
  - 09:03, one utilization sample: GPU compute engine 50% busy, Python
    CPU about 280% of one core (about 2.8 of 16 threads).

  Even partial, this is notable: the run used under a fifth of the VRAM
  and a small share of the CPU, and the GPU's compute engine was about
  half busy. One sample isn't enough to say where the bottleneck is.
  `multi_metric` is sampled throughout, with the fixed samplers.
- **Next run:** `multi_metric`, launched 14:58:32, estimate 520 min, so
  ETA about 23:40. Queue ETA: the same; it is the last run in batch 2.
