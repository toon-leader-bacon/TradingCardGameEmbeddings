# 2026-10-13 05:55 · run single_contrastive done

**Run:** single_contrastive (batch 2, run 3 of 4) · **Config:** `configs/single_contrastive.yaml` (now the recorded `run_config.yaml`) · **Exit:** 0 · **Duration:** 380 min, 2026-10-12 23:29:52 to 2026-10-13 05:49:41 (estimate 400)

## Key numbers

Normalized TEST loss, median over dojos, with the interquartile range
([`results/test_loss_by_round.csv`](../results/test_loss_by_round.csv)):

| kind | n dojos | round 0 | round 1 | round 2 | round 3 | IQR, round 3 | below 1.0, round 3 | max, round 3 |
|---|---|---|---|---|---|---|---|---|
| contrastive (1.00 = chance) | 25 | 0.767 | 0.732 | 0.647 | 0.670 | 0.495-0.826 | 24 | 1.06 |
| held out (1.0 = mean predictor) | 13 | 1.154 | 1.169 | 1.164 | 1.153 | 1.059-1.411 | 0 | 3.68 |

- `checkpoints.csv` marks round 2 (step 14,996) as the best checkpoint,
  and `train_round0002/` is kept. Round 3's final weights are in
  `latest/`.
- 2,898,176 trainable parameters, from the run's own log. That is 2,560
  (0.09%) more than the multi-card cells' 2,895,616.
- The probe was not run; it comes after all the cells.

## Where the data is

- Raw data: `data/runs/2026-10-11_model_by_diet_r2/single_contrastive/` (2.4 GB)
- Console capture: `data/runs/2026-10-11_model_by_diet_r2/logs/single_contrastive.log`
- Project log: `data/runs/2026-10-11_model_by_diet_r2/single_contrastive/run.log`
- Provenance: `data/runs/2026-10-11_model_by_diet_r2/provenance/single_contrastive/` (commit 2d29488, clean tree; the code matches 467d0e1)
- Samples: `logs/single_contrastive.resources.csv` (60 s), `logs/single_contrastive.utilization.csv` (30 s)

## First observations

- Contrastive TEST improved through round 2 (0.767 to 0.647), then rose
  slightly in round 3 (0.670). It is the first cell whose median didn't
  fall every round. One step of 0.023 is about the size of the
  experiment's noise threshold (0.02), so it can't yet be told apart
  from a plateau.
- The held-out dojos were flat at about 1.15-1.17, and none were below
  1.0 in any round. The worst one reached 3.68 in round 3.
- Stable: 7 fp16 overflow steps were skipped, with no NaNs and no errors.
  1,566 "Skipping deck" warnings, as expected for a contrastive-only
  cell. The only other warnings were the three usual startup ones.

## Resources and utilization

- **Memory:** peak Python RAM 2.0 GB, lowest free system RAM 19.2 GB.
  Peak Python VRAM 2.9 GB, with 3.8 GB in use on the 16 GB card
  (368 samples).
- **GPU:** in training (after 23:34, 700 samples), the compute engine was
  busy a median of **57%** of the time (p10 45%, p90 74%). The copy engine
  was idle (0.4%).
- **CPU:** Python used a median of **262%** of one core (p10 243%, p90
  294%), about 2.6 of 16 threads.
- The GPU was idle a larger share of the time than in `multi_metric`,
  with a similar CPU profile. The CPU-bound pattern is clearer here than
  in that run.

## Timing

- Startup: 4 min (23:29:52 to 23:33:44).
- Rounds, each 5,000 steps plus TEST: 94, 94, 95, 93 min. Single-card
  step time is now measured.
- Next run: `single_metric`, launched 05:50, which is also the queue's
  last run. Its estimate drops from 520 to about 490 min, so its ETA is
  about 14:00.
