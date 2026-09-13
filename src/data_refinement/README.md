# data_refinement

Converts raw data written by `data_retrieval` (and `image_processing`,
for image-only sources) into the standardized `GenericCard`/
`GenericDeck` schema (`src/schema/`), and computes derived per-card
metrics from 17lands' raw data. Outputs land under `data/final/`.

Internally, this is expected to be a pipeline of small, individually
typed stages — each stage has an explicit input schema and output
schema, so a stage can be tested and reasoned about in isolation. The
order stages get chained in is a per-experiment/per-source concern,
not something fixed by this container's structure.

## Containers

- **`card_binder/`** the standardized, multi-game card
  store every other stage (and `training`, downstream) reads from and
  writes into. We only need one data source per game to ingest into dedicated
  cards.
  See [`card_binder/README.md`](card_binder/README.md).
- **`deck_box/`** the standardized, multi-game deck
  store: `DeckBox` (mirroring `card_binder`'s CRUD-by-uuid shape) plus the
  `DeckExtractionStage` Strategy Protocol every raw deck source will
  implement. Not ever data source provides deck data to be consumed here.
  See [`deck_box/README.md`](deck_box/README.md).
- **metrics/** Each data source should have dedicated, game-specific or data-source
  specific metrics associated with it. The logic for transforming raw data into
  these more rich game-specific metrics are to be stored here. One directory
  per data source.
