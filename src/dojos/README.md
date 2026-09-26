# dojos

Pluggable auxiliary training tasks for the card embedding model. Every
dojo presents one interface to the trainer, the `Dojo` Protocol
(`dojo.py`): `batches(split, budget, max_examples)` yields `DojoBatch`es
(`inputs` for the encoder, `__len__` = example count) whose card cost
fits the trainer's `BatchBudget`; `compute_loss(embeddings, batch)`
returns a scalar per-example mean loss; `example_count(split)`,
`trainable_parameters()` (the dojo's own decoder head, never the
encoder) and `reset_head()` round it out. A dojo never owns its batch
size - the budget is the trainer's.

Most dojos read (input, label) rows from a parquet file one of
`../data_refinement/metrics/`'s metrics wrote. The contrastive dojo
reads decks straight from a `DeckBox` instead.

Card holdout: each dojo is built with a `CardLookup` and a `HoldoutSpec`
(`src/schema/holdout.py`) and reads cards through one `VisibleCardLookup`
per split, so a TRAIN example never contains a TEST- or VALIDATION-tier
card (TEST sees TRAIN+TEST; VALIDATION sees all). Row splits (8/1/1 by
row, or by deck for contrastive) layer under it.

## Files

- `dojo.py` - the `Dojo`, `DojoBatch` and `BatchBudget` contract.
- `batch.py` - `Batch`, the (inputs, labels) container generic dojos
  yield.
- `budgeted_batching.py` - `group_by_budget`, packs examples into
  batches within a `BatchBudget`.

## Subdirectories

- **[`generic/`](generic/README.md)** - the reusable dojo cells, one per
  `(input shape, task shape)`, plus the `DataConstructor`s that feed
  them and the bases per-metric wrappers build on.
- **`loss/`** - the `NocabLoss` Protocol and the row-wise losses the
  cells use (`MseLoss`, `BceLoss`, `FixedClassificationLoss`,
  `SoftClassificationLoss`, `MaskedVectorRegressionLoss`,
  `PickPredictionCrossEntropyLoss`). A cell builds its own loss; callers
  never construct one.
- **`mods/`** - `Mod`/`ModPipeline`, input transformations applied
  before batching (`MaskTargetKeyMod`, `ShuffleDeckMod`, `NoOpMod`). A
  mod declares `train_only`: augmentations run on TRAIN only, while a
  mod the task depends on (masking the field a dojo predicts) is built
  with `train_only=False` so it applies on every split.
- **`file_managers/`** - split management. `FileManagerParquet` splits a
  metric's parquet into train/test/validation files and streams them in
  chunks (`splits_exist()` lets a repeat construction reuse them).
  `DeckBoxDealer` does the same for a `DeckBox`: a seeded, exact-ratio
  split assignment kept in its own small SQLite index, and fixed-size
  deck samples per split read straight from SQLite.
- **`contrastive/`** - `ContrastiveDojo`, below.
- **Per-source wrappers** - `gwent_one/`, `dominiontabs/`, `play_gwent/`,
  `sts_gg/`, `seventeenlands/{draft_data,game_data,replay_data}/`: one
  thin generic-cell subclass per metric (see `generic/README.md`'s
  "Per-metric wrappers"). Every implemented metric has one.

## `contrastive/` - InfoNCE over deck co-occurrence

Trains embeddings directly on "these cards appear in the same deck",
sourced from a `DeckBox` through a `DeckBoxDealer`: no metric, no parquet
file, no learned decoder head. It doesn't fit the generic cells because
InfoNCE needs every item's embedding in a batch jointly, not a
row-independent `(output, label) -> loss`.

- `pair_constructor.py` - `ContrastivePairConstructor` Strategy: one deck
  sample -> one `ContrastiveBatch`, deciding what counts as a positive
  pair. This is the research surface. `SingleCardPairConstructor`
  samples single cards (every same-deck card is a positive);
  `MultiCardPairConstructor` samples fixed-size groups of cards.
- `contrastive_batch.py` - `ContrastiveBatch`: a flat pool of `inputs`
  (every item is both anchor and candidate), per-item card
  `identities` (so exact duplicate cards are excluded from an anchor's
  negatives), and `positive_cliques` (index sets that are mutually
  positive, one per source deck).
- `contrastive_loss.py` - `ContrastiveLoss` Strategy (batch-level, unlike
  `NocabLoss`). `SingleCardInfoNCELoss`: InfoNCE over one cosine
  similarity matrix of the whole pool. `MultiCardInfoNCELoss`: the same,
  but a card's own item is excluded from both its positives and
  negatives. Each validates its expected item shape and raises on a
  mismatch.
- `dojo.py` - `ContrastiveDojo`: wires dealer + pair constructor + card
  lookup into `Dojo`. An example is one source deck, so the budget
  becomes a deck count per batch; a batch with no positive clique of
  size >= 2 is skipped and logged. It has no trainable parameters.
  Defaults to `SingleCardInfoNCELoss`; the pair constructor and loss
  must agree on item shape.

Not built yet: multi-positive SupCon, a pooling `ContrastiveLoss`
Decorator, and mixed contrastive + label-based training in one step.

## How to run

```python
from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import BatchBudget
from src.dojos.sts_gg.card_average_dojos import CardWinRateDojo
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

binder = CardBinder.load([CardBinder.default_output_path(GameId.SLAY_THE_SPIRE_2)])
dojo = CardWinRateDojo(binder, HoldoutSpec.no_holdout(), card_embedding_size=32)
for batch in dojo.batches(Split.TRAIN, BatchBudget(32, lambda card: 1)):
    loss = dojo.compute_loss(model(batch.inputs), batch)
```

`scripts/smoke_test_training_loop.py` drives real dojos through the
`Trainer` end to end.
