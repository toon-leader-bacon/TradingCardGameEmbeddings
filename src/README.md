# nocab_card_embedding

A personal/portfolio project to build a generic card embedding model
that represents a card from any trading card game (MTG, Pokemon,
Yu-Gi-Oh, Hearthstone, etc.) in a shared embedding space. The model is
trained via a suite of pluggable auxiliary tasks ("dojos") and
evaluated independently of any single dojo. The primary artifact of
value is the embedding model itself (possibly published), not
performance on any one training task — breadth of dojos/data sources
is a deliberate feature of this project, not scope creep to be
trimmed.

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
   implementation (e.g.
   `data_refinement/card_binder/scryfall/ScryfallCardIngestionStage`,
   `.../pokemon_tcg/PokemonTcgCardIngestionStage`), which writes
   directly into a `CardBinder` (owning its own duplicate-detection
   and collision-resolution against it), saved to
   `data/final/cards/<game>.jsonl`.
2. **Metric → dojo:** `data_retrieval/` writes a raw dump, and one of
   `data_refinement/metrics/`'s metric classes (either the
   `Metric[RawRowT]` accumulator/streaming shape or the
   `CorpusScanMetric` shape - see
   [`data_refinement/metrics/README.md`](data_refinement/metrics/README.md))
   resolves card identity through the *same* `CardBinder` file flow 1
   produced and writes a per-card/per-deck training-data parquet file
   under `data/metrics/<source>/`. A `Dojo` (`dojos/`) then reads that
   parquet file as its label source, and separately reads the
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

- **`data_retrieval/`** — *in progress.* Collects raw card/deck data
  from external sources and writes it untouched to `data/raw`. Four
  sources implemented and tested today; more are planned. See
  [`data_retrieval/README.md`](data_retrieval/README.md).

- **`data_refinement/`** — *in progress.* Converts raw data into the
  standardized `GenericCard`/`GenericDeck` schema and 17lands-derived
  per-card metrics, landing results under `data/final`. See
  [`data_refinement/README.md`](data_refinement/README.md).

- **`image_processing/`** — *not started.* Optional, lower-priority
  effort to extract structured text from card images for sources that
  only provide images. No implementation exists yet. See
  [`image_processing/README.md`](image_processing/README.md).

- **`encoder_model/`** — *in progress, no README yet.* Defines the
  embedding network architecture(s) themselves (PyTorch model code
  only) — no knowledge of dojos, training loops, or data loading.
  `SingleCardModel` and `MultiCardModel` both exist, dispatching on
  `src/schema/type_hints.py`'s `InputShape`; `MultiCardModel`'s
  internal embedder is a placeholder (a hashed `nn.Embedding` + a
  self-attention stand-in, not a real card-content encoder yet), and
  `SingleCardModel`'s is unimplemented (`internal_model = None`).

- **`dojos/`** — *in progress.* Houses pluggable auxiliary training
  tasks. Two generations side by side: `seventeenlands/`'s two
  hand-copied per-metric dojo packages (v1, legacy, not migrated), and
  a growing set of generic `(input shape, task shape)` dojo cells under
  `generic/` plus thin per-metric wrapper packages (v2, in progress —
  five cells, `single_card_regression`,
  `single_card_fixed_classification`, `multi_card_regression`,
  `multi_card_binary_classification`, and
  `multi_card_fixed_classification`, and four metric families,
  `CardAverageMetric`, `MaskedFieldMetric`, `DeckLabelMetric` (fully
  covered except `KilledByMetric`, skipped as dead data), and
  `DeckCardMaskMetric` (`LeaderMaskedFromDeckMetric`, its one concrete
  subclass so far), implemented so far). See
  [`dojos/README.md`](dojos/README.md) for current state and
  `plans/dojo_v2.md` for the remaining cells/families still to build.

- **`training/`** — *not started.* `demo_training_loop.py` is a
  non-functional sketch (imports classes that don't exist) of how
  `encoder_model` + a dojo + an optimizer would be driven through a
  train/test/validate loop — no real trainer is implemented yet.

- **`evaluation/`** — *not started.* Will score a trained encoder
  against dojos' held-out splits without backpropagating, plus
  dojo-independent embedding-space exploration. No evaluation runner
  exists yet. See [`evaluation/README.md`](evaluation/README.md).

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
