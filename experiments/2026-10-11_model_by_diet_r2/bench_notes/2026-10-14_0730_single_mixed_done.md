# 2026-10-14 07:30 · run single_mixed done

**Run:** single_mixed (batch 3, run 1 of 1) · **Config:** `configs/single_mixed.yaml` (now the recorded `run_config.yaml`) · **Exit:** 0 · **Duration:** 478 min, 23:22:27 to 07:20:21 (estimate 500)

## Key numbers

Normalized TEST loss, median over dojos, with the interquartile range
([`results/test_loss_by_round.csv`](../results/test_loss_by_round.csv)):

| kind | n dojos | round 0 | round 1 | round 2 | round 3 | IQR, round 3 | below 1.0, round 3 | max, round 3 |
|---|---|---|---|---|---|---|---|---|
| contrastive (1.00 = chance) | 25 | 0.977 | 0.825 | 0.786 | 0.745 | 0.559-0.920 | 23 | 1.03 |
| trained metric (1.0 = mean predictor) | 124 | 1.034 | 0.989 | 0.985 | 0.947 | 0.774-1.144 | 78 | 2.30 |
| held out (1.0 = mean predictor) | 13 | 1.098 | 1.108 | 1.078 | 1.093 | 1.068-1.159 | 1 | 3.26 |

- No dojo went above 5× in any round. The held-out maximum is
  `sts2_runs.card_reward_pick` (2.34, 2.30, 3.40, 3.26 across rounds 0
  to 3; `multi_mixed` had 4.20 to 4.73 on the same dojo).
- `checkpoints.csv` marks every round as a new best; round 3 (step
  19,990) is the last, so `train_round0003/` and `latest/` hold the same
  final weights.
- 2,898,176 trainable encoder parameters.
- The probe was not run; it comes after all the cells.

## Where the data is

- Raw data: `data/runs/2026-10-11_model_by_diet_r2/single_mixed/` (2.8 GB)
- Console capture: `data/runs/2026-10-11_model_by_diet_r2/logs/single_mixed.log`
- Project log: `data/runs/2026-10-11_model_by_diet_r2/single_mixed/run.log`
- Provenance: `data/runs/2026-10-11_model_by_diet_r2/provenance/single_mixed/` (commit ced7ba7, clean tree)
- Samples: `logs/single_mixed.resources.csv` (60 s), `logs/single_mixed.utilization.csv` (30 s)

## First observations

- All three kinds fell in every round except held out, which stayed at
  1.08-1.11.
- Trained metric fell most in the last round (0.985 to 0.947), and the
  contrastive median was still falling at round 3 (0.786 to 0.745).
- Stable: 10 fp16 overflow steps were skipped in total (5, 1, 2, 2), with
  no NaNs and no errors. The only warnings besides the contrastive
  pair-constructor deck skips were the three usual startup ones.

## Resources and utilization

From [`results/resources_by_cell.csv`](../results/resources_by_cell.csv):
- **Memory:** peak Python RAM 4.1 GB, lowest free system RAM 18.7 GB.
  Peak Python VRAM 5.0 GB, with 5.6 GB in use on the 16 GB card.
- **GPU:** in training, the compute engine was busy a median of 68.4% of
  the time (p10 49%, p90 88%; 836 samples).
- **CPU:** Python used a median of 269% of one core, about 2.7 of 16
  threads.

## Timing

- Startup: 30 min. Rounds, each 5,000 steps plus TEST: 114, 113, 111, 111 min.
- The queue is empty; wrap-up follows.
