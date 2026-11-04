# evaluation/extrinsic

Extrinsic evaluation trains fresh dojo heads on a frozen encoder by
reusing `Trainer`, and compares the loss curves. Most of its complexity
lives in `dojos/` and `training/`, so this container stays flat.

## Files

- [run_extrinsic.py](run_extrinsic.py): `run_extrinsic`, one frozen
  `Trainer` phase (named `"extrinsic"`) over every given dojo, no
  checkpointer, a `CsvRunListener` writing
  `<output_dir>/extrinsic/<encoder_label>/rounds.csv`, then one VALIDATION
  pass over each head restored to its best TEST round (`validation.json`:
  raw `losses` and `normalized_losses`, i.e. loss / the dojo's baseline;
  skipped if the run stopped early). Regression dojos' raw losses are in
  label standard deviations (their labels are z-scored from TRAIN
  statistics). `ExtrinsicSpec` holds the phase settings and seed;
  `ExtrinsicResult` reports what was written. Everything is validated
  before anything touches disk (label, directory, a trainable head per
  dojo, then `Phase`/`TrainingPlan`/`Trainer` construction); after
  training starts nothing raises - a failed VALIDATION pass or write is
  logged, so a finished run is never lost. Reuse the *same* dojo objects
  for every encoder: each run calls `reset_head()` and reseeds
  `random`/`torch`, so every encoder starts from identical heads.
- [best_head_keeper.py](best_head_keeper.py): `BestHeadKeeper`, a
  `RunListener` that snapshots each dojo's head at its lowest normalized
  TEST loss and restores those snapshots before the VALIDATION pass.
- [learning_curves.py](learning_curves.py):
  `plot_learning_curves(curves, output_dir, x_axis)`: encoder label ->
  rounds CSV in, one `<dojo>.png` per dojo out, every encoder's normalized
  TEST loss (with a reference line at 1.0, "learned nothing") against
  `step` or `elapsed_seconds`; a chart with any row from a rounds CSV
  older than the `normalized_test_loss` column plots raw TEST loss
  instead. Reads CSVs only through `read_rounds_csv`; reads, groups and
  names every plot (refusing two dojos that would share a file name)
  before writing any. The order of `curves` fixes each encoder's color on
  every chart; up to eight encoders (the validated line colors).
- [curve_renderer.py](curve_renderer.py): `CurveRenderer` draws one chart
  in a `CurveStyle` (on `charts/`'s `ChartCanvas`): 2px lines ending in a
  ringed dot, a legend for two or more encoders (one encoder: the title
  names it), direct end labels when there are at most four and their ends
  don't crowd, hairline gridlines. The rounds CSVs are the table view.

## How to run

One run per encoder on the same dojos:

```python
from src.evaluation.extrinsic.learning_curves import plot_learning_curves
from src.evaluation.extrinsic.run_extrinsic import ExtrinsicSpec, run_extrinsic

curves = {}
spec = ExtrinsicSpec(diet_rule=Uniform(), head_lr=1e-3, steps_per_round=200,
                     max_rounds=50, saturation=saturation, eval_examples_per_dojo=512,
                     seed=0)
for label, model in (("single_v1", trained), ("untrained", fresh)):
    result = run_extrinsic(model, label, dojos, spec, hardware,
                           Path("data/evaluations/run_a"))
    print(label, result.validation_losses)
    curves[label] = result.rounds_csv
plot_learning_curves(curves, Path("data/evaluations/run_a/extrinsic/plots"))
```
