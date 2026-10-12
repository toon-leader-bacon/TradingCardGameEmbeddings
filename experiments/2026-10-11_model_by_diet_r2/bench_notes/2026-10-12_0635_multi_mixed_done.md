# 2026-10-12 06:35 · run multi_mixed done

**Run:** multi_mixed (batch 1 of 6) · **Config:** `configs/multi_mixed.yaml` (now the recorded `run_config.yaml`) · **Exit:** 0 · **Duration:** 518 min, 2026-10-11 21:51:52 to 2026-10-12 06:30:13 (estimate 552)

## Key numbers

Normalized TEST loss after round 3 (step 19,990), median over dojos, with the interquartile range
([`results/test_loss_by_round.csv`](../results/test_loss_by_round.csv)):

| kind | n dojos | round 0 | round 3 | IQR, round 3 | below 1.0, round 3 | max, round 3 |
|---|---|---|---|---|---|---|
| contrastive (1.00 = chance) | 25 | 0.824 | 0.662 | 0.523-0.910 | 24 | 1.04 |
| trained metric (1.0 = mean predictor) | 124 | 1.022 | 0.942 | 0.758-1.103 | 76 | 2.70 |
| held out (1.0 = mean predictor) | 13 | 1.120 | 1.083 | 1.053-1.288 | 0 | 4.73 |

- Pilot reference for contrastive: 0.81-0.86 after 3,000 steps on a
  reduced diet (pooling_pilot).
- No trained metric dojo went above 5× in any round (max 3.09, in round 1).
- `../2026-10-10_lr_pilot/analysis/compare.py` on the console log gives
  the same medians. Its below-1 counts differ by 1-4 (50/64/68/76 against
  54/65/69/76), because it reads the log's 3-decimal ratios, where a
  value like 0.9996 prints as 1.000.
- The probe (deck AUC, linear-cls) was not run; it comes after all the
  cells, as the Recipe says.

## Where the data is

- Raw data: `data/runs/2026-10-11_model_by_diet_r2/multi_mixed/` (2.8 GB;
  `rounds.csv`, `checkpoints.csv`, `train_round0003/` = final and best
  checkpoint, `latest/`)
- Console capture: `data/runs/2026-10-11_model_by_diet_r2/logs/multi_mixed.log`
- Project log: `data/runs/2026-10-11_model_by_diet_r2/multi_mixed/run.log`
- Provenance: `data/runs/2026-10-11_model_by_diet_r2/provenance/multi_mixed/`
  (commit 467d0e1, clean tree; pip freeze; torch 2.10.0+rocm7.13, RX 6800)
- Config check: `data/runs/2026-10-11_model_by_diet_r2/logs/multi_mixed.check.log`
  (passed; 2,895,616 trainable parameters)
- Resource samples: `data/runs/2026-10-11_model_by_diet_r2/logs/multi_mixed.resources.csv`

## First observations

- Contrastive TEST left chance and kept falling across all four rounds.
  It did not plateau at the pilot's 0.81-0.86. The IQR is wide: one
  dojo is at 1.04 in round 3, and the top quartile starts at 0.91.
- The trained-metric median crossed below 1.0 only from round 1 on, and
  is still falling at round 3. Almost half (48 of 124) are still at or
  above the mean predictor.
- The held-out dojos never beat the mean predictor (0 of 13 below 1.0 in
  any round). Their median stays flat at 1.08-1.12, and the worst one
  reaches 4.73.
- Stable: 10 fp16 overflow steps were skipped across 20,000 steps (the
  GradScaler halved the scale each time; the last was at 05:49), and
  there were no NaNs or errors.
- 810 "Skipping deck … known card(s), need N" warnings from the
  contrastive pair builder. These are expected data filtering: series 1's
  `multi_mixed` console log has 596 of the same kind.

## Resources and timing

- Startup (calibration and preflight): 26 min (21:51:52 to 22:18:04).
- Rounds, each 5,000 steps plus TEST: 125, 124, 121, 122 min. The TEST
  share wasn't split out.
- Peak Python RAM 4.2 GB, lowest free system RAM 17.1 GB. Peak Python
  VRAM 5.0 GB (5.6 GB in use on the card), from 502 one-minute samples.
- Queue: empty. This was the batch's only run; the next batch waits for
  the human.
