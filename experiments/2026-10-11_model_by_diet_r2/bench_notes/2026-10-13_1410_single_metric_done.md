# 2026-10-13 14:10 · run single_metric done

**Run:** single_metric (batch 2, run 4 of 4) · **Config:** `configs/single_metric.yaml` (now the recorded `run_config.yaml`) · **Exit:** 0 · **Duration:** 499 min, 05:49:59 to 14:09:11 (estimate 520, revised to 490)

## Key numbers

Normalized TEST loss, median over dojos, with the interquartile range
([`results/test_loss_by_round.csv`](../results/test_loss_by_round.csv)):

| kind | n dojos | round 0 | round 1 | round 2 | round 3 | IQR, round 3 | below 1.0, round 3 | max, round 3 |
|---|---|---|---|---|---|---|---|---|
| trained metric (1.0 = mean predictor) | 124 | 1.029 | 1.004 | 1.002 | 0.997 | 0.853-1.097 | 63 | 2.33 |
| held out (1.0 = mean predictor) | 13 | 1.135 | 1.141 | 1.134 | 1.132 | 1.038-1.227 | 0 | 2.43 |

- No trained metric dojo went above 5× in any round (max 2.36).
- `checkpoints.csv` marks round 3 (step 19,990) as the best checkpoint,
  so `train_round0003/` and `latest/` hold the same final weights.
- 2,898,176 trainable parameters.
- The probe was not run; it comes after all the cells.

## Where the data is

- Raw data: `data/runs/2026-10-11_model_by_diet_r2/single_metric/` (2.8 GB)
- Console capture: `data/runs/2026-10-11_model_by_diet_r2/logs/single_metric.log`
- Project log: `data/runs/2026-10-11_model_by_diet_r2/single_metric/run.log`
- Provenance: `data/runs/2026-10-11_model_by_diet_r2/provenance/single_metric/` (commit 1e27409, clean tree; the code matches 467d0e1)
- Samples: `logs/single_metric.resources.csv` (60 s), `logs/single_metric.utilization.csv` (30 s)

## First observations

- The trained-metric median fell in round 1, then barely moved (1.004,
  1.002, 0.997). By round 3 about half the dojos (63 of 124) are below
  the mean predictor. The top of the IQR stays at about 1.10.
- The held-out dojos were flat at 1.13-1.14, with none below 1.0 in any
  round.
- Stable: 10 fp16 overflow steps were skipped, with no NaNs and no errors.
  The only warnings were the three usual startup ones.

## Resources and utilization

From [`results/resources_by_cell.csv`](../results/resources_by_cell.csv):
- **Memory:** peak Python RAM 4.0 GB, lowest free system RAM 17.4 GB.
  Peak Python VRAM 3.6 GB, with 4.5 GB in use on the 16 GB card.
- **GPU:** in training, the compute engine was busy a median of 75.7% of
  the time (p10 58%, p90 89%; 882 samples).
- **CPU:** Python used a median of 268% of one core, about 2.7 of 16
  threads.

## Timing

- Startup: 26 min. Rounds, each 5,000 steps plus TEST: 121, 117, 117, 118 min.
- The queue is empty. `single_mixed` is held by the human and not queued.
