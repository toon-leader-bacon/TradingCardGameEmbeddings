# dojos

Pluggable auxiliary training tasks for the card embedding model. Every
dojo presents one interface to the trainer, the `Dojo` Protocol
(`dojo.py`): `batches(split, budget, max_examples)` yields `DojoBatch`es
(`inputs` for the encoder, `__len__` = example count) whose card cost
fits the trainer's `BatchBudget`; `compute_loss(embeddings, batch)`
returns a scalar per-example mean loss; `baseline_loss(batch)` returns
the loss an input-ignoring predictor gets on that batch, never depending
on the encoder or head; `example_count(split)`, `trainable_parameters()`
(the dojo's own decoder head, never the encoder) and `reset_head()`
round it out. A dojo never owns its batch size - the budget is the
trainer's.

Normalized loss: dojo losses sit on unrelated scales (nats over a few
classes, nats over a batch's items, squared label units), so the
trainer judges each dojo by loss / baseline_loss: 1.0 means the dojo
learned nothing, 0.0 is perfect. Generic dojos fit one baseline to
their TRAIN split at construction and z-score regression labels from
the same TRAIN sample (`generic/README.md`, "Loss calibration"); the
contrastive dojo computes its baseline per batch, as below.

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
  never construct one. Also the loss calibration that fits a cell's loss
  to its TRAIN split: `LossCalibration` Strategy and its `CalibratedLoss`
  result (`loss_calibration.py`, with `StandardizedRegressionCalibration`
  for the regression cells), `LabelStats` (TRAIN mean and population
  std, `label_stats.py`), the `StandardizedLabelLoss` Decorator that
  z-scores a regression loss's labels, and `prior_baseline_calibrations.py`
  (a Template Method base, one subclass per non-regression loss family,
  each measuring its baseline: class-prior entropy, binary entropy, soft
  target entropy, per-position mean MSE, mean ln(option count)).
- **`mods/`** - `Mod`/`ModPipeline`, input transformations applied
  before batching (`MaskTargetKeyMod`, `ShuffleDeckMod`, `NoOpMod` in
  `common_mods.py`; the card-field augmentations below). A
  mod declares `train_only`: augmentations run on TRAIN only, while a
  mod the task depends on (masking the field a dojo predicts) is built
  with `train_only=False` so it applies on every split. **Mods never
  mutate their input.** A mod returns new cards and lists and never
  assigns into, appends to, or shuffles what it was given. To edit a
  card field, use `GenericCardFactory.with_field` (`src/schema/card_factory.py`, path
  copying: only the containers on the path are copied). For lists, use
  the non-mutating expression forms (`[*xs, x]`, slicing, `sorted`,
  `rng.sample`), never `append`/`insert`/`remove`/`pop`/`sort`/
  `random.shuffle`/`+=`.
  `card_field_mods.py` holds best-effort, train-only augmentations that
  edit every card of any input shape (one card, a list, groups):
  `ShuffleKeysMod` reorders top-level `raw_content` keys,
  `RandomKeyMaskMod` masks one uniformly chosen top-level key (with a
  probability), and `WeightedFieldMaskMod` pulls a `FieldMask` (a set of
  `FieldPath`s, empty meaning "nothing") from a `DropTable`
  (`src/utils/drop_table.py`) filtered to the masks that fit the card
  (empty, or at least one of its paths present; a partly present mask
  masks the paths that exist), so weight meant for fields a card lacks
  moves to masks that do something.
  Best effort: while applying, a card the mod cannot edit passes through
  unchanged and training continues; constructor arguments are validated.
  Each such mod keeps a `ModTally` (cards seen, changed, failed, and the
  first failure) so a mod that never fires or keeps failing is visible.
  `deck_mods.py` holds train-only deck mods that thin the card lists of a
  multi-card or multi-group input (a `DeckThinningMod` Template Method
  base): `CardDropoutMod` drops each card with a probability,
  `CardSubsampleMod` keeps `max(1, round(f * n))` of a group's cards, and
  `DuplicateCollapseMod` keeps the first copy of each card. Each is built
  with the group indexes it may thin (a multi-card input is group 0); other
  groups pass through as the same lists. A non-empty group always keeps at
  least one card, kept cards keep their order, and a datum the mod cannot
  read (a single card, a group index past its groups) passes through and is
  counted. Their `ModTally` counts the cards in thinned groups as seen and
  the cards removed as changed. Deck mods are opt-in per dojo, never a
  default: they change deck size and copy counts, so they must never reach
  a dojo whose label depends on those. `DECK_MOD_GROUPS`
  (`src/training/dojo_catalog.py`) lists the dojos that take them and which
  groups; an option-selection dojo's options group is never listed (its
  label indexes it). A run config's `mods:` attaches them, after the dojo's
  own task mods (`GenericDojo.append_mods`); `mods:` replaces the dojo's
  default augmentations, so list those too to keep them.
  `MASK_TOKEN` (`mod.py`) is the one mask string every masking mod writes,
  and `ModTally` lives there too (`Mod.tally` is None for mods that keep
  none). `mod_specs.py` holds `ModSpec`s: frozen, shareable recipes, one
  per card-field or deck mod (`DeckModSpec` for the latter, carrying its
  `groups`), that each dojo builds into its own mods (a mod's
  random state and tally are per dojo, so mod objects are never shared).
  Per-game default specs are in `augmentation_defaults.py`; see
  `contrastive/` below.
- **`file_managers/`** - split management. `FileManagerParquet` splits a
  metric's parquet into train/test/validation files and streams them in
  chunks (`splits_exist()` lets a repeat construction reuse them).
  `DeckBoxDealer` does the same for a `DeckBox`: a seeded, exact-ratio
  split assignment kept in its own small SQLite index, and fixed-size
  deck samples per split read straight from SQLite.
- **`contrastive/`** - `ContrastiveDojo`, below.
- **Per-source wrappers** - `gwent_one/`, `dominiontabs/`, `play_gwent/`,
  `sts_gg/`, `seventeenlands/{draft_data,game_data,replay_data}/`,
  `isotropic/`, `final_decks/`, `scryfall/`, `pokemon_tcg/`, `cardvault_fabtcg/`,
  `hearthstonejson/`, `spire_codex/`, `fabtcg_decklists/`: one thin generic-cell subclass per metric (see
  `generic/README.md`'s "Per-metric wrappers"). Every implemented metric
  has one, except two isotropic ones: `CopiesBoughtDistributionMetric`
  (raw samples of the mean `AverageCopiesBoughtDojo` already learns) and
  `MultiplayerPlacementMetric` (a ranking over 3-4 decks; no cell ranks
  groups). Its kingdom and partial-deck metrics key rows by
  `kingdom_uuid`/`partial_deck_uuid`, so `isotropic/` wraps
  `DeckLabelDataConstructor` in a column-renaming Decorator
  (`renamed_column_data_constructor.py`). Its two-group and "which
  card(s)" metrics use `isotropic/`'s own constructors: a `CardGroup`
  Strategy (`card_groups.py`) reads each group from a row column (a
  DeckBox group, optionally distinct or with extra cards such as the
  base supply, or a single card), `GroupLabelDataConstructor` builds
  `([group_0, group_1], float)` for the multi-group binary/regression
  cells, and `GroupPickDataConstructor` builds one option-selection
  datum per picked card.

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
  mismatch. `constant_logit_loss(identities, positive_cliques)` is the
  loss with every similarity equal, computed from the batch shape alone:
  each anchor with a positive scores ln(1 + its valid negatives), averaged
  over anchors (single-card, no duplicates, N items in cliques of k_c:
  sum k_c ln(N - k_c + 1) / sum k_c).
- `dojo.py` - `ContrastiveDojo`: wires dealer + pair constructor + card
  lookup into `Dojo`. An example is one source deck, so the budget
  becomes a deck count per batch; a batch with no positive clique of
  size >= 2 is skipped and logged. It has no trainable parameters.
  Defaults to `SingleCardInfoNCELoss`; the pair constructor and loss
  must agree on item shape. An optional `ModPipeline` runs over every
  item card after the pair constructor builds a batch (identities and
  positive cliques are kept: a modded card is still the same card);
  train-only mods run on TRAIN batches only. `mod_tallies()` exposes the
  mods' tallies, as every `Dojo` does. `baseline_loss(batch)` is the
  loss's `constant_logit_loss` for that batch: per batch, since the
  item count follows the trainer's budget and each deck's visible cards,
  and duplicate cards shrink a batch's negatives.

Augmentation defaults: `augmentation_defaults.py` (at the top of
`dojos/`) holds each game's default train-only augmentations as
`ModSpec`s (`mods/mod_specs.py`): a key shuffle, a `DropTable` of
`FieldMask`s, and a small random-key mask. The tables name each game's
`raw_content` keys, so they follow the game's card-binder ingestion stage;
each mainly masks the field that nearly identifies a card's deck (Gwent
`faction`, STS2 `color`), so same-deck positives cannot be matched on that
field alone. A `ModSpec` is a frozen recipe; each dojo builds its own mods
from it, with its own seed and tally. Every catalog dojo takes its game's
defaults (`src/training/dojo_catalog.py`): a contrastive dojo as its whole
pipeline, a metric dojo after its own task mods (`GenericDojo.append_mods`),
so a task's mask (`train_only=False`) stays first and in order. Card-field
mods only remove information and never move a card, so a mask stays masked
and an option-selection dojo keeps its groups and option order. A run
config's `mods:` replaces a dojo's defaults (an empty list turns them off).

Not built yet: multi-positive SupCon, a pooling `ContrastiveLoss`
Decorator, and mixed contrastive + label-based training in one step.

## How to run

```python
from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import BatchBudget
from src.dojos.sts_gg.card_average_dojos import CardUpgradeRateDojo
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

binder = CardBinder.load([CardBinder.default_output_path(GameId.SLAY_THE_SPIRE_2)])
dojo = CardUpgradeRateDojo(binder, HoldoutSpec.no_holdout(), card_embedding_size=32)
for batch in dojo.batches(Split.TRAIN, BatchBudget(32, lambda card: 1)):
    loss = dojo.compute_loss(model(batch.inputs), batch)
```

`scripts/smoke_test_training_loop.py` drives real dojos through the
`Trainer` end to end.
