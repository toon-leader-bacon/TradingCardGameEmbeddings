# nocab_card_embedding

A personal/portfolio project to build a generic card embedding model
that represents a card from any trading card game (MTG, Pokemon,
Yu-Gi-Oh, Hearthstone, etc.) in a shared embedding space. The model is
trained via a suite of pluggable auxiliary tasks ("dojos" and "metrics").
The primary artifact of
value is the embedding model itself, not
performance on any one training task.

This file is a map: one paragraph per top-level container, its current
status, and a link to that container's own `README.md` for depth. Data
produced or consumed anywhere in this project lives under this
project's `data/` directory.

## How the pieces fit together

Two data flows run through this project, both landing on disk between
containers rather than passing objects in memory directly:

1. **Card ingestion:** `data_retrieval/` writes a raw dump to
   `data/raw/`, and `card_binder/build.py`'s
   `build_or_update_card_binder()` hands it to a `CardIngestionStage`
   implementation, which writes
   directly into a `CardBinder` (owning its own duplicate-detection
   and collision-resolution against it), saved to
   `data/final/cards/<game>.jsonl`. There is a similar flow for "deck" data
   that is stored in a "Deck Box", mirroring the "Card Binder" flow described.
2. **Metric → dojo:** `data_retrieval/` writes a raw dump, and one of
   `data_refinement/metrics/`'s metric classes (either the
   `Metric[RawRowT]` accumulator/streaming shape or the
   `CorpusScanMetric` shape - see
   [`data_refinement/metrics/README.md`](data_refinement/metrics/README.md))
   resolves card identity through the *same* `CardBinder` file flow 1
   produced and writes a per-card/per-deck training-data parquet file
   under `data/metrics/<source>/`. A `Dojo` (`dojos/`) then reads that
   parquet file as its (training input, true label) source, and separately reads the
   `CardBinder` file again to turn each label's `nocab_uuid` into an
   actual card.

Flow 2 depends on flow 1 having already run for whichever game a
metric's raw data belongs to — a dojo can't resolve a card it has no
binder entry for.

## Containers

- **`schema/`** — *stable.* The shared, project-wide data structures
  (`GameId`, `GenericCard`, `GenericDeck`, `Provenance`, `DataSource`)
  every other container depends on. Plain dataclasses/enums, no
  behavior — no README of its own (nothing to document beyond the
  types themselves).

- **`data_retrieval/`** — Collects raw card/deck data
  from external sources and writes it to `data/raw`. These classes may 'unwrap'
  the raw data, but typically logic related to cleaning, validating, transforming
  or merging the data should be done as a dedicated metric class. See
  [`data_retrieval/README.md`](data_retrieval/README.md).

- **`data_refinement/`** — There are 3 main types of data refinement: card_binder (extract card data from raw data),
  deck_box (extract deck lists from raw data), and metrics (extrat training data inputs and true labels for consumption by a dojo later)
  [`data_refinement/README.md`](data_refinement/README.md).

- **`encoder_model/`** — Defines the
  embedding network architecture(s) themselves (PyTorch model code
  only)

- **`dojos/`** — Houses pluggable auxiliary training
  tasks, one dojo per metric. Dojos consume training data produces by metrics, split the data into
  train/test/validate sets, apply modifications (as appropriate), yield the data
  to the training loop, receive the embedding vectors back from the encoder_model,
  and compute the loss. To convert the card(s) embeddings(s) into a loss, typically
  the dojo will use a dedicated shallow decoder head model, and the loss signal
  will flow through that into the actual embedding model we're training.
  [`dojos/README.md`](dojos/README.md) for current state.

- **`training/`** — `demo_training_loop.py` is a
  sketch of how encoder_model` + a dojo + an optimizer would be driven through a
  train/test/validate loop.

- **`evaluation/`** — This is NOT the typical Train/Test/Validation concept that
  the dojo is responsible for. This evaluation is for post-training to determine
  how effective the embedding model is. This may include intrinsic validation
  (measuring how much of the embedding space is actually used by the model;
  attempting to identify reasonable clusters of card embeddings; what parts of
  the inputs actually activate certain regions of the model; etc) and extrinsic
  validation (evaluate how the embedding works for new dojos not present durring training at all;
  evaluate how the embedding works for new cards not present during training;
  evaluate how the embedding works for new card games not present during training; etc.)

## A naming note: `Dojo` vs. `Gym`

This project names a pluggable training task by its *shape*, not just
by convention: **`Dojo`** is a non-episodic task — one example in, one
label, one loss out, the ordinary supervised/multi-task learning shape
every task in this project uses today. **`Gym`** is deliberately
reserved, unused, for a task with the OpenAI-Gym-style reinforcement
learning shape instead — an episodic `reset()`/`step()` loop, an
observation-action cycle, a reward signal — a genuinely different kind
of training loop, not a relabeling of the same thing. No `Gym` exists
in this codebase; the name is held in reserve so that if a real
RL-shaped task is ever added, it gets a name that tells a reader which
loop it needs at a glance, instead of stretching "gym" (or "dojo") to
silently cover two different shapes.
