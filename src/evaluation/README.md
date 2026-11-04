# evaluation

Post-training evaluation of encoder checkpoints, run offline by hand from
scripts. `Trainer` never calls this package.

Status: in progress. Built: the intrinsic pipeline (corpus selection, the
embedding table, card labels, evaluation decks, and five analyses),
extrinsic runs, and learning-curve plots. Open work is in
[`TODO.md`](TODO.md).

## Containers

- **[`intrinsic/`](intrinsic/README.md)** - fits no model: embeds cards
  once into an `EmbeddingTable`, then analyzes the stored vectors (label
  agreement, projection plots, effective rank, same-deck retrieval AUC).
  Output: `<run>/embeddings/<encoder>.db` and
  `<run>/intrinsic/<encoder>/<analysis>/`.
- **[`extrinsic/`](extrinsic/README.md)** - trains fresh dojo heads on a
  frozen encoder by reusing `Trainer`, and compares the loss curves.
  Output: `<run>/extrinsic/<encoder>/` and `<run>/extrinsic/plots/`.
- **[`charts/`](charts/README.md)** - `ChartTheme` and `ChartCanvas`, the
  shared colors and matplotlib scaffolding every figure on both sides is
  drawn with. Imports neither.

`intrinsic/` and `extrinsic/` never import each other; `charts/` is the
only code they share.

## How to run

The whole flow - load checkpoints, embed and analyze per encoder, extrinsic
runs on shared dojos, learning curves - is
[`scripts/run_evaluation.py`](../../scripts/run_evaluation.py) (a prototype:
corpus, labels and dojos are hard-coded there as a worked example; it is
re-runnable, skipping finished work):

```bash
PYTHONPATH=. python3 scripts/run_evaluation.py --name gwent_v1 \
    --single-checkpoint runs/single/pretrain_round0007 \
    --multi-checkpoint runs/multi/pretrain_round0005
PYTHONPATH=. python3 scripts/run_evaluation.py --name smoke --smoke  # wiring check
```

It writes `EffectiveRank` under `intrinsic/<encoder>/unlabeled/`. It
samples deck retrieval AUC's VALIDATION decks once per run from each
corpus game's deck box (`data/final/decks/<game>.db`) under the
checkpoint's holdout, so every encoder is scored on the same decks; a
missing or unreadable box, or a holdout with no deck split, skips that
analysis, not the run. Per-deck results pair only within one run: an
untrained-only run falls back to `HoldoutSpec.no_holdout()` and samples
different decks. The corpus must hold the decks' cards (a full-game corpus
does).

[`scripts/plot_learning_curves.py`](../../scripts/plot_learning_curves.py)
plots any rounds CSVs on their own (`--curve label=path`, repeatable): an
extrinsic run that stopped partway, or a training run's own `rounds.csv`.

Each container's README shows its pieces composed by hand.
