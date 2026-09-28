# extrinsic

How useful the frozen embedding is as input to an actual downstream
task: unlike `intrinsic/`, a new model *is* trained on top, and its
training is itself the measurement.

## The one workflow: fresh-head training curve

Freeze the encoder checkpoint under test. Take a `Dojo` - either one the
encoder never trained on at all, or one it did train on but with
`reset_head()` called first, so the head itself starts fresh either way.
Train only `dojo.trainable_parameters()` (never the encoder) and record
loss over time (both wall-clock and step/epoch count, since the two can
diverge across hardware or batch size). Reuse
`src/training/round_evaluation.py`'s `evaluate_test_losses` for the
TEST-side number at each checkpoint of this mini-run, the same way the
real `Trainer` does.

Report, per run:
- time-to-a-given-loss (wall clock and step count)
- deepest loss reached before plateauing

A `Dojo` is already a cards-and-labels source with splits, holdout and a
loss bolted on, so "which cards" for this workflow is mostly already
handled by which `Dojo`/`HoldoutSpec` is passed in - see `../README.md`.

## Extensibility (not v1, but the shape should allow it later)

- More than one dojo per evaluation run (a battery, not a single task).
- Restricting to a subset of cards (e.g. only a game's VALIDATION-tier
  cards) rather than the dojo's full TEST pass.
- More statistics beyond time-to-loss and floor loss (e.g. loss variance
  across seeds, as a stand-in for representational stability).
- Comparing models: single-card vs. multi-card, different architectures,
  or different training seeds of the same architecture, all through the
  same harness, since it only depends on `Dojo` and `TrainableEncoder`
  Protocols, not on which concrete model produced the embeddings.
  Multi-card model comparisons have an open wrinkle (an embedding is
  contextualized by whatever list it's evaluated with - what list to use
  at eval time isn't decided yet); punted for now, noted in `../TODO.md`.

## Status

Design only; no code yet.
