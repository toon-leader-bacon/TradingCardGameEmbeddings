# TODO

- [ ] **A project glossary (`TERMS.md` at the root).** One place that
  defines the project's vocabulary and how the terms relate:
  - metric vs. dojo, one feeding the other;
  - dojo vs. dojo cell;
  - card binder and deck box;
  - batch, step, round, phase, diet, saturation and epoch;
  - TRAIN / TEST / VALIDATION splits vs. holdout tiers;
  - baseline and normalized loss;
  - held-out dojos and held-out games.

  Parts are already explained in `src/training/README.md`,
  `src/dojos/README.md` and `src/data_refinement/`. The glossary should
  link to those READMEs for depth, not repeat them. Low priority.
- [ ] **Adopt token whitening; test raw + whitened tokens together.**
  From the 2026-10-10 pooling pilot (`experiments/2026-10-10_pooling_pilot/NOTEBOOK.md`):
  make token whitening the default, with mean pooling as the cheap
  baseline and whitened 4-query attention as the candidate for longer
  runs. The next idea worth testing is giving the pooling both raw and
  whitened tokens, to try to keep both the deck signal (whitening:
  contrastive off chance, deck AUC ~0.68-0.70) and the numeric detail
  (unwhitened 4-query attention: numeric-field R^2 0.57, lost under
  whitening).
- Use the test loss to adjust the loss function temperature in an
  intelligent way.
- Explore a "just-in-time" metric-generation architecture where a dojo
  drives its associated metric generator directly against the raw data
  at split time, instead of always going through an intermediate
  per-metric output file (parquet/JSONL) written by a separate scan
  step. A possible future alternative, not a planned change.
