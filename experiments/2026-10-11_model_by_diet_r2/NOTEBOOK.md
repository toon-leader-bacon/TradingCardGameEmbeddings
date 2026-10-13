# Series 1, round 2: model × diet on a stable embedding

**Status:** running
**Dates:** 2026-10-11 -
**Parent:** [2026-10-09_model_by_diet](../2026-10-09_model_by_diet/NOTEBOOK.md) · **Branch:** `experiment/2026-10-11_model_by_diet_r2`

> Now that the embedding is stable (output LayerNorm, token whitening,
> mean pooling), which model (single-card or multi-card) and which diet
> (contrastive, metric or 50/50) trains the most useful general card
> embedding, and does any of them beat the untrained whitened encoder?

## Aim

**Problem (priority: the fix).** Pick the default model and diet for
series 2 and 3.
- **Metric:** the per-card probe from
  [2026-10-10_pooling_probe](../2026-10-10_pooling_probe/NOTEBOOK.md), on
  the "isolated" view (each card alone through the whole model):
  - same-deck AUC (0.5 = chance);
  - linear-classification balanced accuracy over card fields.
- **Current value:** not measured on this setup. The nearest is the
  pooling pilot's `tokwhite_mean`: deck AUC 0.680 and linear-cls 0.672,
  after 3,000 steps on a reduced diet. The unwhitened untrained model
  scores 0.539 and 0.770
  ([pilot probe CSV](../2026-10-10_pooling_pilot/results/probe_checkpoints.csv)).
- **Bar:** the best cell beats its own untrained whitened-mean control by
  at least **+0.03 deck AUC**, and loses no more than **0.03 linear-cls
  balanced accuracy** against that control. The bar must hold twice:
  - on the full probe, which is comparable with the pilot;
  - on the out-of-sample slice: Gwent's deck AUC, and linear-cls on
    non-TRAIN-tier cards over fields no dojo trained.

  If it holds on the full probe but not on the slice, it is reported as
  "met in sample only".
- **Ranking and choosing:**
  - Cells that pass the linear-cls limit are ranked by deck AUC.
  - Cells within 0.02 of the top cell count as tied. Among tied cells,
    mixed beats the single-objective diets, and then single-card beats
    multi-card (it is cheaper and simpler).
  - If no cell meets the bar, the default stays the untrained whitened
    encoder plus whichever cell loses the least linear-cls. The next
    experiment then targets the information loss.
  - One seed makes this a pilot-strength choice
    (`EXPERIMENT_PRINCIPLES.md` §1).

Training can cost per-card information. In the pilot, training lowered
linear-cls from 0.77 (untrained) to 0.67 while deck AUC rose. That is
why the bar compares against an untrained control and covers both
measures.

**Hypotheses** (second priority, understanding):
- **H1, diet.** The 50/50 mixed diet gives the best balance:
  - contrastive-only wins deck AUC but loses field probes, as in the
    shakeout's collapse
    ([2026-09-30_first_train_shakeout](../2026-09-30_first_train_shakeout/NOTEBOOK.md));
  - metric-only keeps the field probes but gains little deck AUC.

  *Refuted if* either single-objective diet matches or beats mixed on
  both deck AUC and linear-cls, within the same model. It is judged
  separately for each model. If it holds for one model and not the
  other, the verdict is "model-dependent".
- **H2, model.** On per-card probes there is no clear difference between
  single-card and multi-card. Group attention should help a card in
  context, not the card alone. *Refuted if* either model leads the other
  by 0.02 or more deck AUC on the same diet, in either direction.

## Background

Series 1 asked this same question on 2026-10-09 but never answered it.
Its only cell, `multi_mixed`, showed that the setup itself was unstable.
Three follow-ups fixed that:
- **[lr_pilot](../2026-10-10_lr_pilot/NOTEBOOK.md):**
  - an output LayerNorm (no gain or bias) ended the metric blowups;
  - `encoder_lr` 1e-4 kept the most card-specific spread;
  - contrastive stayed at chance.
- **[pooling_probe](../2026-10-10_pooling_probe/NOTEBOOK.md):** frozen
  ModernBERT features are anisotropic. Whitening raises the fixed-feature
  deck AUC from 0.54 to about 0.65; centering alone barely helps.
- **[pooling_pilot](../2026-10-10_pooling_pilot/NOTEBOOK.md):**
  - token whitening moves contrastive off chance (0.81-0.86, against
    1.00 without it);
  - trainable pooling adds little over whitened mean pooling;
  - whitening loses numeric field detail (R² 0.27-0.34, against 0.57
    for unwhitened 4-query attention).

This round reruns the series-1 design with those fixes, and with fewer
and longer rounds. In series 1, TEST took 38% of each 1,000-step round.

## Method

- **Terms:**
  - a *cell* is one model × diet combination;
  - a *control* is an untrained model of the same shape;
  - a *batch* is one queued run of the lab technician (here, one cell).
- **Cells** (configs in [`configs/`](configs/); they differ only in
  `model` and the diet):

  | | contrastive only | metric only | 50/50 mixed |
  |---|---|---|---|
  | **single-card** | `single_contrastive` | `single_metric` | `single_mixed` |
  | **multi-card** | `multi_contrastive` | `multi_metric` | `multi_mixed` |

  - trained dojos: 25 contrastive (5 styles × 5 games), 124 metric, or
    149 for mixed; plus the same 13 held-out dojos in every cell
    (counted by parsing the configs against today's catalog);
  - diets as in series 1: contrastive styles weighted equally, then
    games equally; metric dojos weighted by TRAIN count ** 0.3; mixed is
    half and half.
- **Controls:** untrained single-card and multi-card models, built after
  the training runs and probed the same way:
  - each is built by `build_model` from `configs/single_mixed.yaml` or
    `configs/multi_mixed.yaml` under seed 0;
  - its weights are saved as `latest/encoder.pt`, next to a copy of the
    config, in `data/runs/2026-10-11_model_by_diet_r2/control_<model>/`;
  - it should then match the trained cells' starting weights. Building
    them includes checking that the RNG order reproduces the round-0
    initial weights.

  A second, head-free reference is the whitened mean-pooled ModernBERT
  feature alone. Offline it scored 0.613-0.637 deck AUC
  ([pooling_probe](../2026-10-10_pooling_probe/NOTEBOOK.md)).
- **Probe split** (decided before results, from the design review):
  - The probe's cards and decks are not filtered by holdout tier.
    Contrastive dojos train on decks from 5 games, and metric dojos
    train some probe fields directly (MTG, Hearthstone and Dominion
    type, Spire rarity). So the full probe is partly in-sample, and in
    a different way for each diet.
  - The probe is therefore also reported split by card tier (TRAIN
    against VALIDATION/TEST under holdout seed 0), with its fields split
    into trained ones and held-out ones (class, pitch, STS2 type).
  - Gwent's deck AUC is the clean deck number.
  - Output goes to this experiment's `results/`, labelled
    `r2_<cell>` / `r2_control_<model>`. It does not go to the pooling
    probe's shared CSV.
- **Held constant:**
  - **model:** frozen ModernBERT-base; token whitening
    (`data/whitening/modernbert_tokens_holdout_seed0.pt`); mean pooling;
    `residual_mlp` head, 256-wide embeddings, `mlp_hidden_dim` 512;
    output LayerNorm (in the code since the lr_pilot);
  - **architectures:** single-card has 9 MLP blocks. Multi-card has 3
    blocks plus 2 pre-norm group-attention layers (4 heads, FFN 1024,
    dropout 0.1);
  - **optimizer:** `encoder_lr` 1e-4, `head_lr` 1e-3. The config's
    "encoder" (`encoder_trainable: true`) is the card model's trainable
    part: pooling, embedding head and group attention, about 2.9M
    parameters. ModernBERT stays frozen;
  - **budget:** 4 rounds × 5,000 steps = 20,000 steps. Saturation is off
    (`patience_rounds` 1000), so the diet never changes;
  - **data:** training seed 0; holdout seed 0, tiers 8/1/1, Gwent held
    out (every Gwent card is VALIDATION-tier, and no Gwent dojo is
    trained);
  - **batches:** Huber regression loss; `max_batch_cost` 256; fp16;
    128 TEST examples per dojo per round.
- **Metrics:**
  - **Primary, after training:** the probe on each cell's final
    checkpoint and on each control (deck AUC, linear-cls bacc, linear
    R², kNN). The pipeline view is reported too.
  - **Secondary:**
    - median normalized TEST loss per round, split into contrastive,
      metric and held out (1.00 = chance for contrastive; 1.0 = mean
      predictor for metric);
    - Gwent's per-game deck AUC, since Gwent is the unseen game.
  - **Later, outside this aim:** the evaluation tools
    (`scripts/run_evaluation.py`, intrinsic analyses and extrinsic
    fresh-head runs on the held-out dojos), on the checkpoints this
    experiment keeps.
- **Predictions** (written before launch):
  - the bar above, and H1 and H2 as stated in Aim;
  - contrastive TEST leaves chance in the contrastive and mixed cells
    (pilot: 0.81-0.86 after 3,000 steps). Staying at or above 0.95
    after 20,000 steps would mean the pilot's result does not carry over
    to the full diet;
  - no trained metric dojo goes above 5× in any round (lr_pilot: 0).
- **Noise:** one seed. A gap under 0.02 deck AUC, or under 0.02
  linear-cls, is reported as "no clear difference". For scale, the
  pilot's whitened variants spread over 0.655-0.697 deck AUC.
- **Known confounds:**
  - Trainable parameter counts differ slightly: about 2.9M for each
    model in series 1. Whitening and mean pooling add no trainable
    parameters. `--check` reports the exact counts.
  - TEST covers only a cell's own dojos plus the 13 held-out ones. That
    is 38 for contrastive-only, 137 for metric-only and 162 for mixed.
    So only the held-out dojos' TEST numbers compare across all six
    cells, and the cells' run times differ while their step budgets
    stay equal.
  - The probe is partly in-sample, and in a different way for each diet
    (see Probe split).
  - The whitening stats were fitted on TRAIN-tier cards of this
    holdout. The probe's cards may overlap them; the parent probe
    judged the optimism small but did not measure it.
- **Cost** (from `experiments/README.md`; full mixed diet, multi-card):
  - startup about 28 min;
  - about 117 min of training per 5,000-step round (about 1.4 s per
    step), plus about 14 min of TEST;
  - so about **9 h per mixed or metric cell**, and somewhat less for
    contrastive-only. About 50-55 h in total, run as six batches of one
    cell each (human decision 2026-10-11). Single-card step time is not
    measured yet.
  - Disk: about 2.4 GB per run, so about 15 GB in total. The probe and
    controls take minutes each.

## Recipe

- **Data:** every metric parquet and split the catalog keys need.
  `PYTHONPATH=. python scripts/run_training.py <config> --check` passes on
  `configs/single_mixed.yaml` (it holds every dojo any cell uses).
- **Whitening stats:**
  `data/whitening/modernbert_tokens_holdout_seed0.pt` (2026-10-10 16:43),
  from the pooling pilot:
  `PYTHONPATH=. python scripts/fit_whitening.py --holdout-from <any config here> --out data/whitening/modernbert_tokens_holdout_seed0.pt`.
- **Training** (ROCm venv `G:\Projects\venvs\tcg-rocm`), one cell per
  batch:
  `PYTHONPATH=. python scripts/run_training.py experiments/2026-10-11_model_by_diet_r2/configs/<cell>.yaml`.
  Outputs land in `data/runs/2026-10-11_model_by_diet_r2/<cell>/`.
- **Probe:** an `analysis/` script, written after training, that wraps
  `experiments/2026-10-10_pooling_probe/analysis/probe_checkpoint.py`
  with the tier and field split (see Probe split). It writes to this
  experiment's `results/`.
- **TEST table:**
  [`../2026-10-10_lr_pilot/analysis/compare.py`](../2026-10-10_lr_pilot/analysis/compare.py)
  on each cell's console log.
- **Configs:** the six files in `configs/` are launch configs, generated
  from series 1's launch configs (patterns kept). Each is replaced by its
  run's recorded `run_config.yaml` once it runs.

## Log

### 2026-10-11 · Design decisions

What:
- whitening + mean pooling + `encoder_lr` 1e-4 in every cell;
- 4 × 5,000 steps instead of 20 × 1,000;
- one cell per batch;
- final checkpoints kept for the probe and for later evaluation;
- controls built at the end.

Why:
- the three follow-ups above settled the model block;
- the human chose not to re-check `encoder_lr` 3e-4 (the lr_pilot's
  "re-check once the offset is removed"), and not to pilot the
  single-card model separately, since both cost hours and would not
  change the design;
- longer rounds cut the TEST share from about 38% to about 11%.

Seen: the configs parse against today's catalog with the same dojo
counts as series 1.

### 2026-10-11 · Design review (experiment-critic) and changes

What: one design-mode pass found no blocker to launch. Its concerns,
and the response to each:
1. **The probe is partly in-sample, unequally by diet.** Added the
   probe split, and a second, out-of-sample reading of the bar.
2. **The control had no recipe.** Defined it (`build_model`, seed 0,
   same starting weights), and added the head-free whitened feature as
   a second reference.
3. **No rule for picking a cell.** Added the ranking, the tie-break and
   the fallback if no cell meets the bar.
4. **H2 could be refuted in one direction only, and H1 had no
   per-model verdict.** Both fixed.
5. **The TEST confound was misstated.** Each cell scores only its own
   dojos plus the held-out ones; corrected.
6. **Probe output went to the shared pooling-probe CSV.** It now goes
   to this experiment's `results/`, labelled `r2_*`.
7. **Provenance.** The experiment folder is committed on its branch
   before launch, and the frozen-ModernBERT wording is clarified.

Why: every change is to how the results will be read, so it is fixed
now, before any result is seen. No config changed.

### 2026-10-12 · Batch 1 (lab_technician)

- [multi_mixed done](bench_notes/2026-10-12_0635_multi_mixed_done.md): exit 0, 518 min; TEST medians after round 3: contrastive 0.662 (1.00 = chance), trained metric 0.942, held out 1.083 (1.0 = mean predictor).
- [Batch 1 wrap-up](bench_notes/2026-10-12_0640_batch1_wrap_up.md): no incidents; results in `results/test_loss_by_round.*`.

### 2026-10-12 · Batch 2 queued (human decision)

What: batch 2 runs the two remaining multi-card cells back to back:
`multi_contrastive`, then `multi_metric`. They are estimated at about
8 h and 8.7 h, so about 17 h in total, under the ~24 h gate.

Why: the human reviewed batch 1 and read it as "the training is
working". Running the other multi-card diets next completes the
multi-card row of the grid.

### 2026-10-12 · Batch 2, run 1 (lab_technician)

- [Incident: launcher re-ran training](bench_notes/2026-10-12_1500_incident_launcher_rerun.md): small issue 1 of 5. An edit to the launcher mid-run lost `multi_contrastive`'s console capture and resource samples. Training data intact; no confound.
- [multi_contrastive done](bench_notes/2026-10-12_1505_multi_contrastive_done.md): 399 min; TEST medians after round 3: contrastive 0.532 (1.00 = chance), held out 1.134 (1.0 = mean predictor).

### 2026-10-12 · Batch 2 extended (human decision)

What: `single_contrastive` and `single_metric` were added to batch 2,
after `multi_metric`. Projected end: about 15:00 on 2026-10-13, so about
17 h from the decision (21:47) and about 31 h for batch 2 as a whole
(from 08:18 on 2026-10-12), past the ~24 h gate. The human asked for this
explicitly, and that is the approval.

The estimates reuse the multi-card figures: single-card step time has
not been measured. No separate `--check` is run for these two, so that
check jobs don't overlap `multi_metric`'s resource samples. Each run's
own startup preflight serves as the check, and a failure there is
handled as a small issue.

## Results

## Conclusions

## Open questions

- **A much longer training run once the diet is chosen** (human,
  2026-10-12). `multi_mixed` was still improving at the end of its
  20,000 steps (about 9 h): contrastive TEST median 0.824 → 0.662 and
  trained-metric median 1.022 → 0.942 from round 0 to round 3, both
  still falling
  ([bench note](bench_notes/2026-10-12_0635_multi_mixed_done.md)).
  Once this experiment picks the default model and diet, a follow-up
  should train that cell for much longer, to find where it saturates.
  Open points: whether the held-out dojos, flat at about 1.08 so far,
  also improve with more steps; and the ~24 h gate, which a longer run
  needs explicit approval to pass.
