# Series 1, round 2: model × diet on a stable embedding

**Status:** awaiting review
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

### 2026-10-12 · Batch 2, run 2 (lab_technician)

- [multi_metric done](bench_notes/2026-10-12_2335_multi_metric_done.md): exit 0, 511 min; TEST medians after round 3: trained metric 0.936, held out 1.111 (1.0 = mean predictor). The first fully sampled run: VRAM 3.6 GB of 16, GPU compute busy 76% (median), Python CPU about 2.8 of 16 threads.

### 2026-10-13 · Batch 2, run 3 (lab_technician)

- [single_contrastive done](bench_notes/2026-10-13_0555_single_contrastive_done.md): exit 0, 380 min; TEST medians after round 3: contrastive 0.670 (best in round 2, 0.647; 1.00 = chance), held out 1.153 (1.0 = mean predictor). GPU compute busy 57% (median).

### 2026-10-13 · Batch 2, run 4 and wrap-up (lab_technician)

- [single_metric done](bench_notes/2026-10-13_1410_single_metric_done.md): exit 0, 499 min; TEST medians after round 3: trained metric 0.997, held out 1.132 (1.0 = mean predictor).
- [Batch 2 wrap-up](bench_notes/2026-10-13_1415_batch2_wrap_up.md): 5 of 6 cells done; `single_mixed` held by the human. On TEST, multi-card beats single-card on the same diet, and no diet moves the held-out dojos. Resource and timing table in `results/resources_by_cell.csv`.

### 2026-10-13 · Batch 3 queued (human decision)

What: batch 3 is the last cell, `single_mixed`, estimated at about
500 min (26 min startup plus 4 rounds of about 118 min, from
`single_metric`), so well under the ~24 h gate. The human held it on
2026-10-13 after batch 2 and released it the same evening.

Pre-flight: free RAM 23.9 GB, VRAM 17.0 of 17.2 GB free with no other
Python process, 1.1 TB free on G:. No separate `--check`; as for the
other single-card cells, the run's startup preflight covers it. Its
launch config differs from `multi_mixed` only in the model block.

### 2026-10-14 · Batch 3, run 1 and wrap-up (lab_technician)

- [single_mixed done](bench_notes/2026-10-14_0730_single_mixed_done.md): exit 0, 478 min; TEST medians after round 3: contrastive 0.745 (1.00 = chance), trained metric 0.947, held out 1.093 (1.0 = mean predictor).
- [Batch 3 wrap-up](bench_notes/2026-10-14_0735_batch3_wrap_up.md): all 6 training cells done. On TEST, multi-card is lower than single-card on every diet (the mixed trained-metric gap is under 0.02); mixed matches or beats metric-only on trained metric; no cell moves the held-out dojos below 1.08. The probe and controls are next (principal investigator).

### 2026-10-14 · Step 2: probe, controls and fresh-head runs (principal investigator)

What: the human asked for step 2 (evaluate every trained model) and then
a review of all findings. Three pieces, all from `analysis/`:
1. **Per-card probe** (`analysis/probe_cells.py`): the pooling probe's
   card pool and metrics, plus the Method's probe split. Results in
   `results/probe_summary.csv`, `probe_by_field.csv` and
   `probe_deck_by_game.csv`.
2. **Probe tables and figures** (`analysis/probe_report.py`):
   `results/probe_vs_control.csv` (the bar, per cell) and four figures.
3. **Fresh-head runs on the 13 held-out dojos**
   (`analysis/extrinsic_held_out.py`, `src/evaluation/extrinsic`): each
   encoder frozen, the held-out dojos' heads trained from identical
   starting weights (6 rounds × 500 steps, uniform diet, head lr 1e-3,
   seed 0), then scored on VALIDATION at each head's best TEST round.
   Encoders: the six cells and one untrained control per model shape.
   Output: `results/extrinsic_validation.csv`.

Deviations from the Method, decided before the fresh-head results:
- **Controls cannot match the cells' starting weights.** Training never
  seeds the model's initial weights (`scripts/run_training.py` calls no
  `torch.manual_seed`; only the dojos and diet are seeded), so the
  Method's check "the RNG order reproduces the round-0 initial weights"
  cannot pass. Instead each model shape has three untrained controls
  (`torch.manual_seed` 0, 1, 2), which also shows how much a random
  initialization moves the probe. The bar is reported against control
  seed 0 and against the mean of the three.
- **The held-out TEST numbers in the bench notes do not measure transfer.**
  The trainer evaluates held-out dojos every round but never trains
  their heads (`src/training/README.md`: "evaluated every round but
  never trained on"), so their loss is that of a randomly initialized
  head. Flat values of 1.08-1.15 are expected whatever the encoder
  learned. The wrap-up observation "no diet moved the held-out dojos" is
  correct as a number but says nothing about the embedding
  ([batch 3 wrap-up](bench_notes/2026-10-14_0735_batch3_wrap_up.md),
  [batch 2 wrap-up](bench_notes/2026-10-13_1415_batch2_wrap_up.md)). The
  fresh-head runs replace it as the transfer measure. The Method listed
  them as "later, outside this aim"; they are run now at the human's
  request and reported as secondary evidence, not as part of the bar.
- **Speed: the frozen features are computed once per card.** ModernBERT,
  token whitening and mean pooling have no trainable weights, and all six
  runs used byte-identical whitening stats, so both scripts compute the
  pooled 768-d vector once per card and run only each model's head and
  attention. `probe_cells.py` checks this against the real model on 64
  cards per cell: minimum cosine 1.00000 in all six.
- **The out-of-sample linear-cls slice uses every card of the three
  games** (Hearthstone, Flesh and Blood, Slay the Spire 2), not just the
  probe pool, so that the non-TRAIN-tier subset is large enough: 1,272
  Hearthstone class, 860 pitch and 101 STS2 type cards, 5-fold CV.

Why: the probe is the Aim's decision metric; the fresh-head runs are the
only measure of transfer to dojos the encoder never trained on.

### 2026-10-14 · PI review, draft for discussion (principal investigator)

This is a draft. Nothing here is a Conclusion yet: the verdicts below are
proposed, for discussion with the human. Every number is in Results.

**Proposed verdicts** (judged against the predictions written before
launch):

| part of the aim | proposed verdict | evidence |
|---|---|---|
| Problem: the bar | **not met.** Every cell fails the linear-cls limit, and all cells are below their controls on the slice. Under the fallback rule, the default stays the untrained whitened encoder, plus the cell that loses the least linear-cls: `single_mixed` (-0.036 vs seed 0) and `single_contrastive` (-0.038) are tied (under 0.02). | `probe_vs_control.csv` |
| H1, diet | **model-dependent.** Single-card: refuted. Contrastive-only beats mixed on deck AUC (0.767 against 0.711), with no clear linear-cls difference (0.615 against 0.617). Multi-card: not refuted. Contrastive-only leads on deck AUC (+0.027) but loses linear-cls (-0.023), and metric-only is worse on both. The predicted mechanism was wrong: metric-only does not keep the field probes, it collapses the embedding. | `probe_summary.csv` |
| H2, model | **refuted**, in single-card's favour. On the contrastive diet single-card leads by 0.030 deck AUC, and the bootstrap intervals don't overlap. Mixed shows no clear difference (0.711 against 0.710). The metric-only gap (+0.022 to multi) is between two collapsed embeddings near chance, so it carries little weight. | `probe_summary.csv` |

**Key findings**

1. **TEST and the per-card probe disagree about the model.** Multi-card
   wins TEST on every diet, but single-card matches or beats it on the
   isolated per-card probe. Two observations point the same way:
   multi-card's TEST advantage lives in context, and running group
   attention over a group of one costs card information. The second
   shows in multi-card's pre-attention view, which keeps 0.05-0.08 more
   linear-cls than its isolated view.
2. **Metric training with a trainable encoder collapses the card
   embedding while its TEST loss improves.** Effective rank falls to 2.7
   and 5.6. A likely cause (a hypothesis, not tested):
   - the metric heads total about 17M parameters, against the encoder's
     2.9M, so they can decode tiny differences, and nothing pushes the
     encoder to keep its geometry spread out;
   - the metrics that learn best are run-level ones (e.g. STS2 character
     at 0.06), which a few directions solve.

   Falling metric TEST is therefore not evidence of a better embedding.
3. **Contrastive training is the only source of the deck signal.** The
   signal holds for non-TRAIN-tier cards in the seen games (0.725
   against about 0.57), but not for the unseen game: Gwent deck AUC is
   below the controls in every cell. It also helps a deck-level held-out
   task strongly (FaB hero from deck: 0.24-0.29 against 0.92-1.01).
4. **Every trained cell loses card-field information**, and most on the
   fields no dojo trains (slice: -0.09 to -0.20).
5. **The untrained head loses the most information of all.** The frozen
   whitened mean beats every trained cell and every control on
   linear-cls (0.863), the slice (0.956) and Gwent (0.639). The 256-d
   residual MLP head loses much of that before any training. This is
   the largest effect in the experiment, and the cheapest to study: an
   offline probe of untrained head variants needs no training.
6. **The mixed diet transfers best to held-out dojos with fresh heads**
   (median 0.856 single and 0.814 multi, against 0.938 and 0.915 for the
   controls; 7 of 13 dojos better than control). Contrastive-only and
   metric-only show no clear median difference from their controls.
   This is secondary, one head seed and a short head budget, and 4 of 13
   dojos don't learn at all.
7. **Small per-card statistic dojos overfit.** About 450 TRAIN cards
   each, sampled throughout 20,000 steps, leaves their TEST worse than
   the mean predictor (up to 2.4×).

**The human's questions**
- **Is multi-card the better model?** Not for per-card embeddings, on
  this evidence. It wins when cards are scored together, which is
  exactly where its attention helps. On the isolated view it is tied or
  worse, and it costs about 5% more time per step. If downstream use is
  per-card retrieval or storage (what evaluation stores), single-card is
  the pick for now. If downstream use is in-context (scoring a deck), it
  is multi-card. One seed; a second seed would firm this up.
- **Contrastive pre-training, then metrics as post-training?** Partly
  supported:
  - contrastive is clearly the right pre-training signal;
  - metric training on a trainable encoder is destructive, so metric
    "post-training" should leave the encoder frozen (heads only), or use
    a guarded fine-tune. Related work: Kumar et al. 2022, *Fine-Tuning
    can Distort Pretrained Features*
    ([arXiv:2202.10054](https://arxiv.org/abs/2202.10054)), whose
    linear-probe-then-fine-tune recipe fits this case.

  Against a pure curriculum: mixed transferred best to fresh heads. So
  some metric signal during pre-training may help, as long as the
  contrastive loss keeps the geometry from collapsing. That is a
  testable question (proposed experiment B).
- **More contrastive or more metric data?** The data says the bottleneck
  right now is the model, not the data:
  - the frozen features already hold most of the card-field information
    that training loses (finding 5);
  - metric TEST improvements did not make better embeddings (finding 2).

  Within that, what the data does suggest:
  - **Contrastive:** more decks for the small games (Pokémon has 150
    TRAIN decks, and its probe deck AUC of 0.93 is probably partly
    memorized). The Gwent result also says that deck signal does not yet
    cross games, so more game variety may matter more than more MTG
    decks.
  - **Metric:** prefer metrics with many examples per target (run-level,
    pick-level, draft-level). Per-card statistic dojos with a few
    hundred cards overfit as currently weighted, and need either
    pooling across games or far fewer steps.
  - Which metrics best preserve card fields can't be answered from this
    grid. It needs an ablation by metric family (proposed experiment C).
- **Faster training.** The frozen-feature cache works exactly (minimum
  cosine 1.00000 against the real model). With it, a fresh-head run
  covered 3,000 steps in about 4 min per encoder, once 51.5k cards were
  cached. While ModernBERT stays frozen, training could read cached
  features instead of running the language model every step. That is
  likely the biggest throughput gain available, ahead of batch-size
  tuning. Not measured for training yet.

**Weaknesses to discuss**
- One training seed, and one head seed for the fresh-head runs.
- The controls can't reproduce the cells' starting weights.
- The probe pool's decks are partly TRAIN decks: deck AUC is partly in
  sample, as the Method expected.
- The fresh-head budget is short.

**Proposed next experiments** (for discussion; not yet in
`experiments/TODO.md`):
- **A. Head that preserves the frozen features.** First, an offline
  probe of untrained variants (minutes, no training):
  - a linear projection;
  - a residual skip from the frozen feature;
  - 512 or 768 wide;
  - no output LayerNorm.

  Then a short training pilot with contrastive and an anti-collapse
  term. Candidates are VICReg's variance and covariance terms (Bardes
  et al. 2022, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)) or
  the uniformity loss (Wang & Isola 2020,
  [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)). See also Jing
  et al. 2022 on dimensional collapse
  ([arXiv:2110.09348](https://arxiv.org/abs/2110.09348)).
- **B. Curriculum.** Contrastive pre-training, then (i) metric heads on
  the frozen encoder, against (ii) a full metric fine-tune, against
  (iii) mixed from the start, at equal total steps. This tests the
  human's idea directly.
- **C. Why metric training collapses.** Smaller dojo heads, and per-card
  metric dojos against run- and deck-level ones, trained separately.
- **D. Throughput.** Train from cached frozen features. Measure steps/s
  and check that the results match an uncached run.
- **E. Seeds.** Two more seeds of `single_contrastive`, `single_mixed`
  and `multi_mixed`, cheaper once D lands.

Small code fixes these runs exposed (for the owning TODOs):
- seed the model's initial weights in `run_training.py`;
- log training loss per round;
- either drop held-out dojos from the training-time TEST, or label them
  "untrained head" in the logs.

### 2026-10-14 · Intrinsic suite with src/evaluation (principal investigator, human request)

The human asked to run the evaluation module's own intrinsic analyses, so
the numbers compare across experiments. The per-card probe above is
experiment code from the pooling probe, not `src/evaluation`. The Method had
listed the module's tools as "later, outside this aim".

- **Script:** `analysis/intrinsic_suite.py`, summarized by
  `analysis/intrinsic_report.py`. It uses the module as built:
  - **Embedding:** one `EmbeddingTable` per encoder, isolated view, over
    every card of all 7 games (65,634 cards).
  - **Label-free:** `EffectiveRank` on the whole table.
  - **Per label source:** `ClusterAgreement`, `LabelCompactness` and
    `ProjectionPlot`, with `PerLabelCap(500)` and seed 0.
  - **Learning curves:** `plot_learning_curves` over the fresh-head runs'
    rounds CSVs.
- **Encoders:** the 6 cells and the 6 untrained controls (seeds 0-2 per
  shape). The head-free whitened mean is not included, since it is not a
  model the module can embed with.
- **Label sources:**
  - every card: `game` and `holdout_tier` (the runs' holdout);
  - held out in every cell (`MetricParquetLabels`): rarity tier across
    games, Hearthstone class, Flesh and Blood pitch, STS2 card type;
  - trained in the metric and mixed cells: MTG card type, Flesh and Blood
    class, Pokémon stage;
  - the unseen game: Gwent faction.
- **Frozen features:** computed once for all 65,634 cards, about 4 min. Each
  trained cell was checked against the real model on 64 cards: minimum
  cosine 0.99975 to 0.99999.
- **Time:** 10:03 to 10:34. No analysis was skipped.
- **Output:** `data/evaluations/2026-10-11_model_by_diet_r2/`, summarized in
  `results/intrinsic_*` and `results/learning_curves/`.

The human also asked for two TODOs, added to `src/evaluation/TODO.md`:
- deck AUC as an intrinsic analysis;
- `run_evaluation.py` driving an experiment's held-out dojos, cards and
  games.

## Results

All numbers are from one training seed (pilot strength). "No clear
difference" means a gap under 0.02, as set in the Method.

### Training (TEST loss)

From [`results/test_loss_by_round.csv`](results/test_loss_by_round.csv)
and [`test_loss_by_round.png`](results/test_loss_by_round.png). These are
medians at round 3 (step about 20,000).

| cell | contrastive (25; 1.00 = chance) | trained metric (124; 1.0 = mean predictor) |
|---|---|---|
| single_contrastive | 0.670 | - |
| single_metric | - | 0.997 |
| single_mixed | 0.745 | 0.947 |
| multi_contrastive | 0.532 | - |
| multi_metric | - | 0.936 |
| multi_mixed | 0.662 | 0.942 |

- Contrastive TEST left chance in every cell that trained it, as
  predicted.
- No trained metric dojo went above 5× in any round, as predicted.
- Multi-card has the lower TEST on every diet. The one exception is mixed
  trained-metric (0.942 against 0.947), which is no clear difference.
- **Correction:** the held-out TEST numbers in the bench notes come from
  heads that were never trained (see the 2026-10-14 step 2 Log entry).
  They don't measure transfer; the fresh-head runs below do.
- **TEST by metric dojo size** (`multi_metric`, round 3, from
  `test_loss_by_dojo.csv` and the TRAIN counts in `diet_shares.csv`):
  - The dojos that learn best are large run-level and deck-level ones.
    For example `sts2_runs.character_prediction` (2.1M TRAIN examples)
    reaches 0.06, and `sts2_runs.floors_cleared` reaches 0.33.
  - The worst are per-card statistics with about 440-450 TRAIN cards,
    such as `sts_gg.card_deck_size` (2.43) and
    `sts2_runs.card_relic_count` (1.93). Their heads end worse than the
    mean predictor.

### Per-card probe (the Aim's metric)

From [`results/probe_summary.csv`](results/probe_summary.csv) and
[`results/probe_vs_control.csv`](results/probe_vs_control.csv):
- isolated view;
- 3,415 cards and 150 decks (25 for each of 6 games);
- 95% bootstrap intervals over decks.

The "untrained" columns give the range over the three control seeds. Deck
AUC: 0.5 = chance.

| model | deck AUC [95%] | Gwent deck AUC (unseen game) | linear-cls bacc | slice linear-cls (held-out fields, non-TRAIN cards) | effective rank |
|---|---|---|---|---|---|
| frozen whitened mean (768-d, no head) | 0.618 [0.607-0.628] | 0.639 | 0.863 | 0.956 | 232.6 |
| untrained single-card (3 seeds) | 0.567-0.578 | 0.581-0.608 | 0.653-0.688 | 0.683-0.697 | 106-108 |
| untrained multi-card (3 seeds) | 0.570-0.572 | 0.588-0.612 | 0.652-0.675 | 0.672-0.692 | 90-93 |
| single_contrastive | **0.767** [0.757-0.776] | 0.568 | 0.615 | 0.596 | 30.1 |
| single_metric | 0.500 [0.492-0.508] | 0.504 | 0.582 | 0.581 | 2.7 |
| single_mixed | 0.711 [0.700-0.721] | 0.552 | 0.617 | 0.538 | 13.2 |
| multi_contrastive | 0.737 [0.728-0.745] | 0.551 | 0.596 | 0.566 | 25.6 |
| multi_metric | 0.522 [0.512-0.531] | 0.530 | 0.545 | 0.497 | 5.6 |
| multi_mixed | 0.710 [0.700-0.720] | 0.537 | 0.619 | 0.557 | 13.2 |

![deck AUC against linear-cls](results/probe_tradeoff.png)

**Against the bar** (`probe_vs_control.csv`):
- No cell meets it, against either control seed 0 or the mean of the
  three seeds, on the full probe or on the slice.
- **Deck AUC passes easily for the four cells with contrastive dojos:**
  +0.14 to +0.20 over the controls. The metric-only cells fall below
  them, by 0.05 to 0.07.
- **Every cell fails the linear-cls limit:**
  - against control seed 0, losses range from 0.036 (`single_mixed`) to
    0.123 (`multi_metric`);
  - against the control mean, from 0.046 to 0.120.
  - The control seeds alone spread 0.035 on linear-cls, which is as wide
    as the limit itself.
- **On the slice, every cell is below its controls on both measures:**
  - Gwent deck AUC is lower by 0.01 to 0.09;
  - held-out-field linear-cls is lower by 0.09 to 0.20.

**Other probe readings:**
- **Deck AUC by anchor tier.** The contrastive gain holds for non-TRAIN
  cards in the seen games. `single_contrastive` scores 0.725 on
  non-TRAIN anchors and 0.818 on TRAIN anchors, against about 0.57 for
  the controls. It does not carry to Gwent
  ([`probe_deck_by_game.png`](results/probe_deck_by_game.png)).
- **Metric-only training collapses the isolated embedding.** Mean cosine
  between cards is 0.99 (single) and 0.94 (multi), effective rank is 2.7
  and 5.6, and kNN game accuracy is 0.54 and 0.47, against about 0.89
  untrained ([`probe_effective_rank.png`](results/probe_effective_rank.png)).
- **Multi-card's pre-attention ("pipeline") view keeps more field
  information than its isolated view.** Linear-cls is 0.656 against
  0.596 (contrastive), 0.623 against 0.545 (metric) and 0.666 against
  0.619 (mixed). The untrained multi-card controls show the same gap,
  about 0.025.
- **The untrained 256-d head already loses information.** Going from the
  frozen whitened mean to the untrained controls, linear-cls falls 0.863
  to about 0.67, slice linear-cls 0.956 to about 0.69, and effective
  rank 233 to about 100. On the slice, Flesh and Blood pitch falls from
  0.926 to 0.55-0.62
  ([`probe_held_out_fields.png`](results/probe_held_out_fields.png)).

### Fresh heads on the 13 held-out dojos (secondary)

From [`results/extrinsic_summary.csv`](results/extrinsic_summary.csv) and
[`extrinsic_validation.csv`](results/extrinsic_validation.csv). Values
are normalized VALIDATION loss (1.0 = mean predictor or prior; lower is
better). The setup:
- one control per model shape, seed 0;
- 3,000 head steps in total (about 230 per dojo);
- one head seed.

| encoder | median over 13 dojos | dojos below 1.0 | median paired difference vs its control | dojos better / worse than control by > 0.02 |
|---|---|---|---|---|
| control_single_s0 | 0.938 | 7 | - | - |
| single_contrastive | 0.876 | 10 | +0.004 | 3 / 5 |
| single_metric | 0.964 | 7 | +0.001 | 4 / 4 |
| single_mixed | 0.856 | 10 | -0.047 | 7 / 4 |
| control_multi_s0 | 0.915 | 8 | - | - |
| multi_contrastive | 0.964 | 10 | +0.011 | 4 / 6 |
| multi_metric | 0.965 | 8 | +0.003 | 5 / 6 |
| multi_mixed | 0.814 | 10 | -0.027 | 7 / 4 |

![held-out dojos](results/extrinsic_by_dojo.png)

- **Deck-level gain.** `fabtcg_decklists.hero_masked_from_deck` drops
  from 0.92-1.01 (controls and metric-only) to 0.24-0.29 with any
  contrastive-trained encoder.
- **Card-field losses.**
  - `spire_codex.card_type_mask`: controls 0.33-0.35, trained cells
    0.48-0.98.
  - `pokemon_tcg.hp_regression`: controls 0.36-0.43, trained cells
    0.46-0.78.
- **Dojos no encoder learns in this budget.** Two stay at about 1.0:
  `seventeenlands_game_data.deck_win_prediction` and
  `cardvault_fabtcg.pitch_mask`. Two end above 1.0 for every encoder:
  `sts2_runs.card_reward_pick` and `sts2_runs.card_win_rate_at_act2`.
- **Speed.** With the cached frozen features, each encoder's 3,000 steps
  plus TEST and VALIDATION took about 4-9 min. The cache grew to 51,523
  cards. The log covers 07:32 to 08:09 for all eight encoders.

### Intrinsic suite from src/evaluation (standard comparison)

From [`results/intrinsic_summary.csv`](results/intrinsic_summary.csv)
(every scalar: [`intrinsic_scalars.csv`](results/intrinsic_scalars.csv)).
The setup:
- isolated view;
- 65,634 cards;
- `PerLabelCap(500)`, seed 0.

Nearest-centroid accuracy is leave-one-out; its majority baseline is in
the CSV. The "untrained" columns give the range over the three control
seeds.

| label source | untrained single | single contr. / metric / mixed | untrained multi | multi contr. / metric / mixed | majority baseline |
|---|---|---|---|---|---|
| game | 0.83-0.86 | 0.69 / 0.34 / 0.49 | 0.82-0.84 | 0.59 / 0.35 / 0.55 | 0.14 |
| held out: rarity tier | 0.47-0.51 | 0.34 / 0.33 / 0.35 | 0.45-0.51 | 0.32 / 0.37 / 0.31 | 0.19 |
| held out: HS class | 0.49-0.52 | 0.29 / 0.15 / 0.24 | 0.46-0.50 | 0.28 / 0.17 / 0.24 | 0.10 |
| held out: FaB pitch | 0.53-0.56 | 0.42 / 0.39 / 0.40 | 0.53-0.56 | 0.41 / 0.38 / 0.38 | 0.33 |
| held out: STS2 type | 0.85-0.87 | 0.46 / 0.52 / 0.42 | 0.83-0.86 | 0.47 / 0.62 / 0.46 | 0.43 |
| trained: MTG type | 0.72-0.74 | 0.43 / 0.30 / 0.33 | 0.69-0.73 | 0.41 / 0.39 / 0.36 | 0.15 |
| trained: FaB class | 0.57-0.60 | **0.72** / 0.15 / **0.68** | 0.53-0.58 | **0.71** / 0.19 / **0.69** | 0.11 |
| trained: Pokémon stage | 0.69-0.70 | 0.41 / 0.46 / 0.42 | 0.69-0.71 | 0.39 / 0.47 / 0.38 | 0.26 |
| unseen: Gwent faction | 0.61-0.66 | 0.40 / 0.22 / 0.31 | 0.60-0.64 | 0.38 / 0.27 / 0.33 | 0.19 |
| holdout tier | 0.36-0.37 | 0.37 / 0.34 / 0.38 | 0.34-0.36 | 0.34 / 0.37 / 0.37 | 0.33 |

![per label source](results/intrinsic_by_label.png)

**Effective rank** (`EffectiveRank`, all cards):

| model | effective rank | top-1 variance share |
|---|---|---|
| untrained single (3 seeds) | 110-111 | 0.04 |
| untrained multi (3 seeds) | 93-96 | 0.04-0.05 |
| single contrastive / metric / mixed | 28.7 / 2.7 / 13.1 | 0.12 / 0.76 / 0.20 |
| multi contrastive / metric / mixed | 24.4 / 6.3 / 12.6 | 0.12 / 0.33 / 0.20 |

- **These agree with the probe, by different methods.** No fitted
  classifier is involved, and the corpus is every card rather than the
  probe pool. Every trained cell is below every control on nearest-centroid
  accuracy for every card field, with one exception (holdout tier sits at chance for all of them).
- **The exception is Flesh and Blood class**, under the contrastive and
  mixed diets:
  - nearest-centroid accuracy 0.68-0.72 against 0.53-0.60 for the controls;
  - NMI 0.55-0.58 against about 0.14;
  - silhouette 0.11-0.15 against about 0.
  - The metric-only cells, which train a FaB class dojo, are far below
    the controls (0.15-0.19). So the class structure comes from the deck
    signal, not from the class dojo. A likely reason, not tested: Flesh
    and Blood decks are built around one hero class, so deck-mates share a
    class.
- **The fields metric training targets are not more clustered after
  metric training.**
  - MTG type and Pokémon stage are lower in the metric cells than in the
    controls (0.30-0.47 against 0.69-0.74).
  - The heads decode these fields from directions that cosine geometry
    does not show. This fits the collapse reading (top-1 variance share
    0.76 for `single_metric`).
- **Holdout tier stays at chance in every encoder** (0.34-0.38 against
  0.33). Nothing in the geometry separates training cards from held-out
  cards.
- **Projections**
  ([game](results/intrinsic_projection_game.png),
  [HS class](results/intrinsic_projection_heldout_hs_class.png),
  [MTG type](results/intrinsic_projection_trained_mtg_card_type.png),
  [holdout tier](results/intrinsic_projection_holdout_tier.png)), for
  illustration only:
  - the controls separate games into large clouds;
  - the contrastive cells break them into many small, tight clusters, which
    look like deck neighbourhoods;
  - the metric cells mix all games together.
- **Learning curves** of the fresh-head runs are in
  [`results/learning_curves/`](results/learning_curves/), one per held-out
  dojo. They show, for example, the contrastive-trained encoders' early
  lead on `fabtcg_decklists.hero_masked_from_deck`.

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

- **Head that preserves the frozen features** (PI review 2026-10-14,
  proposed experiment A): the untrained 256-d head loses most of the
  frozen feature's card information before any training.
- **Contrastive pre-training, then metrics** (human, 2026-10-14;
  proposed experiment B): frozen-encoder metric heads against a full
  fine-tune against mixed from the start.
- **Why metric training collapses the embedding** (proposed experiment
  C).
- **Training from cached frozen features** (proposed experiment D), for
  throughput.
- **More seeds** of the leading cells (proposed experiment E).
- **What data to collect next**: no ranking of data types is possible
  from this grid; see the PI review's answer and experiment C.
