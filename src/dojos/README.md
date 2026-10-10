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
contrastive dojo computes its baseline per batch, as below. The trainer
also divides each step's loss by this baseline (times a per-dojo weight)
before backward(), so the gradient scale matches the judged scale
(`../training/README.md`, "Loss weighting").

Most dojos read (input, label) rows from a parquet file one of
`../data_refinement/metrics/`'s metrics wrote. The contrastive dojo
reads decks straight from a `DeckBox` instead.

Card holdout: each dojo is built with a `CardLookup` and a `HoldoutSpec`
(`src/schema/holdout.py`) and reads cards through one `VisibleCardLookup`
per split, so a TRAIN example never contains a TEST- or VALIDATION-tier
card (TEST sees TRAIN+TEST; VALIDATION sees all). Row splits (8/1/1)
layer under it. They are by row, by deck for contrastive, or by
`DojoConfig.split_group_column` for a metric with several rows per deck
or kingdom (the held-out-card dojos and four isotropic dojos), so those
rows never straddle TRAIN and TEST.

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
  cells use (`MseLoss`, `HuberLoss`, `BceLoss`, `FixedClassificationLoss`,
  `SoftClassificationLoss`, `MaskedVectorRegressionLoss`,
  `PickPredictionCrossEntropyLoss`). A cell builds its own loss; callers
  never construct one. Also the loss calibration that fits a cell's loss
  to its TRAIN split: `LossCalibration` Strategy and its `CalibratedLoss`
  result (`loss_calibration.py`, with `StandardizedRegressionCalibration`
  and `HuberRegressionCalibration` for the regression cells; a
  `RegressionObjective` bundles each loss with its matching calibration),
  `LabelStats` (TRAIN mean and population
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
  `per_game_mod.py` holds `PerGameMod`, which applies the `ModPipeline`
  of each datum's card's game (a game with no entry passes through); the
  inner mods run whenever it does, so its own `train_only` decides the
  splits. It reports the inner mods' tallies through `Mod.child_tallies()`,
  which `ModPipeline.mod_tallies` lists under the mod's key
  (`1:PerGameMod/gwent/0:ShuffleKeysMod`).
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
  The rows are in uniformly random order across the whole file, so any
  prefix of a split is a fair sample (calibration reads one). Metric
  files are often sorted (17lands by set), and a split too big for memory
  is shuffled in two passes (`bucket_shuffle.py`): rows are scattered at
  random into bucket files of about 256 MB in memory (estimated from a
  sample of decoded rows, not parquet's dictionary-encoded size), then each bucket
  is shuffled whole and appended to the splits. Temporaries live in
  `<split directory>/.shuffle_tmp/`, and a split only gets its final
  name once every split is complete. Each split file carries the schema
  metadata `nocab_split_order: uniform_shuffle`; `splits_exist()` treats
  a file without it as missing, so splits written before this rule are
  rebuilt on the next construction.
  `DeckBoxDealer` does the same for a `DeckBox`: a seeded, exact-ratio
  split assignment kept in its own small SQLite index, and fixed-size
  deck samples per split read straight from SQLite.
  `GameBalancedChunkReader` reads one split file as a pass of rows drawn
  evenly across the values of a column (a game), optionally dropping rows
  first (`keep_row`); a dojo returns it from `GenericDojo._chunks` to
  change what it trains on.
- **`cross_game/`** - `RarityTierDojo`: one head over the shared
  rarity ladder (`TIER_1`..`TIER_4` and `SPECIAL`), trained on the cards of
  six games at once. It reads a `MultiGameCardLookup`
  (`../data_refinement/card_binder/`) and the cross-game rarity metric
  (`../data_refinement/metrics/cross_game/README.md`), whose file carries
  a binder version per game (`GenericDojo` checks each). What it masks
  differs per game, so the mask is a `PerGameMod` over each game's
  translator `masked_paths()` (every split); the catalog's per-game
  augmentations are one train-only `PerGameMod` too. TRAIN rows are drawn
  evenly across games (and the OTHER rows dropped first), through
  `GenericDojo._chunks`, for training and for the loss calibration alike,
  so TRAIN passes are random; TEST and VALIDATION are read in file order.
  Its `RarityTierDataConstructor` drops OTHER rows. Catalog key
  `cross_game.rarity_tier`, built by `MultiGameCardDojoRecipe`.
- **`contrastive/`** - `ContrastiveDojo`, below.
- **Per-source wrappers** - `gwent_one/`, `dominiontabs/`, `play_gwent/`,
  `sts_gg/`, `sts2_runs/` (the card-reward, shop-purchase, card-removal and card-upgrade picks), `seventeenlands/{draft_data,game_data,replay_data}/`,
  `isotropic/`, `final_decks/`, `scryfall/`, `pokemon_tcg/`, `cardvault_fabtcg/`,
  `hearthstonejson/`, `spire_codex/`, `fabtcg_decklists/`: one thin generic-cell subclass per metric (see
  `generic/README.md`'s "Per-metric wrappers"). The 17lands
  wrappers take a `data_slice` (sets and formats) and train on that
  slice's file (`seventeenlands/sliced_dojos.py`; see
  `../data_refinement/metrics/seventeenlands/README.md`, "Partitions and
  slices"). Every implemented metric
  has one, except isotropic `MultiplayerPlacementMetric` (a ranking over
  3-4 decks; no cell ranks groups) and the support-only 17lands
  `DeckOccurrenceCountMetric`. `CopiesBoughtDistributionDojo` classifies a
  card's copy count in a finished deck (1 to 9, 10+), so it learns the
  whole distribution where `AverageCopiesBoughtDojo` learns the mean. Its kingdom and partial-deck metrics key rows by
  `kingdom_uuid`/`partial_deck_uuid`, so `isotropic/` wraps
  `DeckLabelDataConstructor` in a column-renaming Decorator
  (`generic/renamed_column_data_constructor.py` - reused by
  `fabtcg_decklists/` and `play_gwent/` too, so it lives in `generic/`
  rather than here). Its two-group and "which
  card(s)" metrics use `isotropic/`'s own constructors: a `CardGroup`
  Strategy (`card_groups.py`) reads each group from a row column (a
  DeckBox group, optionally distinct or with extra cards such as the
  base supply, or a single card), `GroupLabelDataConstructor` builds
  `([group_0, group_1], float)` for the multi-group binary/regression
  cells, and `GroupPickDataConstructor` builds one option-selection
  datum per picked card.

## `contrastive/` - InfoNCE over deck co-occurrence

Trains embeddings directly on how cards sit together in decks, sourced
from a `DeckBox` through a `DeckBoxDealer`: no metric, no parquet file, no
learned decoder head. It doesn't fit the generic cells because InfoNCE
needs every item's embedding in a batch jointly, not a row-independent
`(output, label) -> loss`.

Each **style** is a matched pair constructor and loss, with one catalog
key per style and game (`<prefix>.<game>`, for the six games with a deck
box). They teach different lessons about how cards sit in a deck:

| Style | Key prefix | What attends together | Compared unit | Teaches |
|---|---|---|---|---|
| Single card | `contrastive` | nothing | each card | "these cards share a deck" |
| Missing card | `contrastive_missing_card` | the slice; the missing card alone | slice mean vs. lone card | "which card is missing from this slice" |
| Odd one out | `contrastive_odd_one_out` | the slice, intruder included | each card vs. the rest | "which card doesn't belong here" |
| Cross-slice card match | `contrastive_cross_slice` | each slice | each card | "my card and your cards share a deck" |
| Slice match | `contrastive_slice_match` | each slice | slice mean | "these two slices share a deck" |
| Card in contexts | `contrastive_card_in_contexts` | each slice, same anchor in each | the anchor in each slice | "a card stays itself in any context" |

The examples use two decks, A = {a1 … a6} and B = {b1 … b6}. Brackets
mark what is embedded together: only cards inside the same bracket attend
to each other. A real batch holds more decks (up to 16, fewer when the
batch budget is tight).

**Single card** (2 cards per deck). Every card is embedded alone:

```
[a1]  [a4]  [b2]  [b5]      → a1: positive a4; negatives b2 b5
```

**Missing card** (slice 8, +1 card). Per deck, one card is picked and a
slice drawn with every copy of it removed. The slice attends and is
averaged; the missing card is embedded alone, the same path as the stored
card embeddings:

```
[a1 a2 a3] → context_A      [a4] → card_A
[b1 b2 b3] → context_B      [b4] → card_B
```

|  | card_A | card_B |
|---|---|---|
| **context_A** | positive | negative |
| **context_B** | negative | positive |

Both directions count, CLIP-style. Contexts are never compared with each
other, nor cards with cards, and two decks missing the same card are each
other's right answers, not negatives. A card from deck B can still be in
deck A outside A's slice and count as a negative: the batch carries no
deck membership, so that noise is accepted.

**Odd one out** (7 cards + 1 intruder). Per deck, a slice plus one card
from another deck in the batch that the host deck doesn't hold at all.
Each card's fit is the cosine of its contextual embedding with the mean of
the item's other cards; the loss is the cross-entropy of the intruder
fitting worst (baseline ln 8):

```
[a1 a2 a3 b3]   intruder: b3        [b1 b2 b4 a5]   intruder: a5
```

**Cross-slice card match** (2 disjoint slices of 4). A card's positives
are the other slice's cards; its own slice's cards are ignored:

```
[a1 a2 a3]  [a4 a5 a6]  [b1 b2 b3]  [b4 b5 b6]
a1: positives a4 a5 a6; ignored a2 a3; negatives b1 … b6
```

**Slice match** (the same batches). Each slice is averaged after
attention and compared like a single card: `[a1 a2 a3] → A1`, positive
A2, negatives B1 B2. Two slices of the same cards in another order count
as one identity, so neither is the other's negative.

**Card in contexts** (an anchor + 7 cards, twice). The same anchor card
fronts two disjoint slices of its deck; only its output is kept:

```
[a1 a2 a3] → a1 (context 1)    [a1 a4 a5] → a1 (context 2)
a1 (context 1): positive a1 (context 2); negatives both b1s
```

It is primarily an evaluation dojo, not for training diets: attention
that ignores context wins it outright, and for `SingleCardModel` it is
trivially solved. Run on TEST batches, it measures identity retention
(see `src/evaluation/TODO.md`).

None of these teach copy counts (how many of a card a deck should run):
the missing card and the intruder are never cards still in the slice or
host deck. Count quality is a claim about winning, left to the
outcome-labelled deck dojos.

Batch budget: a deck costs its style's cards per deck (single card 2,
missing card 9, odd one out 8, cross-slice and slice match 8, card in
contexts 16), and `ContrastiveDojo` fits as many decks as `max_batch_cost`
allows, up to 16 and never fewer than 2.

Files:

- `contrastive_style.py` - the `ContrastiveStyle` Protocol (Abstract
  Factory): `pair_constructor(rng_seed, staple_subsampling)` and
  `contrastive_loss()`, built as a matched set so the two can't be mixed
  up. The catalog's `ContrastiveDojoRecipe` takes a style; its
  `_CONTRASTIVE_STYLES` maps each key prefix to one.
- `styles/` - one module per style, each holding its pair constructor,
  its loss and its style: `single_card.py`, `missing_card.py`,
  `odd_one_out.py`, `cross_slice_card.py`, `slice_match.py`,
  `card_in_contexts.py`, and `deck_slices.py` (`DeckSlicesPairConstructor`,
  shared by cross-slice card match and slice match). Slice match's loss
  and card in contexts' loss are Decorators over `SingleCardInfoNCELoss`
  (`PooledItemsLoss` averages each item, `AnchorCardLoss` keeps each
  item's anchor). Odd one out's `OddOneOutLoss` is the only loss that
  isn't InfoNCE: its candidates are positions within one item.
- `pair_constructor.py` - the `ContrastivePairConstructor` Protocol
  (Strategy: one deck sample -> one `ContrastiveBatch`) and what every
  style shares. `SliceSampler`: a deck's known cards, staple thinning,
  draws without replacement (optionally leaving out every copy of some
  cards), `full_draw` (a draw that can't come up short, or a
  `RuntimeError`), `held_out_draw` (pick a card, draw others around it),
  and the
  skip logging (warning for too few known cards, debug for too few after
  thinning or once a pick's copies are out). `consecutive_slices` cuts one
  draw into disjoint slices. `contrastive_batch_from_deck_items` turns each
  deck's item card ids into a batch, the deck's items one clique in order.
  `known_card_uuids`, `card_for_known_uuid`, `cards_for_known_uuids` look
  cards up.
- `staple_subsampling.py` - optional staple thinning, applied by
  `SliceSampler` for every style. Before sampling, each card occurrence is
  kept with probability `min(1, sqrt(t / df))` (word2vec subsampling),
  where `df` is the card's share of decks (`DocumentFrequency`).
  `df` is counted once over the first 20,000 TRAIN decks, in the
  dealer's seeded order. It is cached as JSON under `data/splits/` and
  keyed by the box's CardBinder version and the sample size. The deck box
  is only read. A deck thinned below what its style needs is skipped. No
  subsampling (`t = inf`, the default) is the old path and draws nothing
  extra from the RNG. A run config's `staple_subsampling:` sets `t` per
  contrastive dojo.
- `contrastive_batch.py` - `ContrastiveBatch`: a flat pool of `inputs`
  (every item is both anchor and candidate), per-item card
  `identities` (so exact duplicate cards are excluded from an anchor's
  negatives), and `positive_cliques` (index sets that are mutually
  positive, one per source deck). A style's loss may read meaning into
  the order within a clique (missing card: `[context, card]`) or a card's
  position within an item (odd one out: the intruder is last; card in
  contexts: the anchor is first). That leaks nothing because every
  encoder is permutation-equivariant within a group (pinned by
  `tests/encoder_model`'s `test_a_group_is_permutation_equivariant`) and
  contrastive dojos take no deck mods.
- `contrastive_loss.py` - the `ContrastiveLoss` Protocol (Strategy,
  batch-level, unlike `NocabLoss`), `DegenerateBatchError` (a well-formed
  batch whose loss is undefined: no positive, or no negative), and the
  shared InfoNCE math every InfoNCE style reduces to:
  `pairwise_cosine_similarity`, `identity_negative_mask`, `anchor_loss`,
  `mean_anchor_loss`, `mean_item_embeddings`, `check_item_embeddings` and
  `mean_constant_logit_loss`. Each loss's
  `constant_logit_loss(identities, positive_cliques)` is its loss with
  every similarity equal, computed from the batch shape alone: each
  anchor with a positive scores ln(1 + its valid negatives), averaged over
  anchors (single card, no duplicates, N items in cliques of k_c:
  sum k_c ln(N - k_c + 1) / sum k_c; missing card with P pairs: ln P;
  odd one out: ln(item size)).
- `dojo.py` - `ContrastiveDojo`: wires dealer + pair constructor + card
  lookup into `Dojo`. An example is one source deck, so the budget
  becomes a deck count per batch. A batch whose loss is undefined (the
  loss raises `DegenerateBatchError`, e.g. one surviving deck) is skipped
  and logged, so training and evaluation never see one; a malformed batch
  (any other error from the loss) still raises. It has no trainable
  parameters. Defaults to `SingleCardInfoNCELoss`. An optional
  `ModPipeline` runs over every item card after the pair constructor
  builds a batch (identities and positive cliques are kept: a modded card
  is still the same card); train-only mods run on TRAIN batches only.
  `mod_tallies()` exposes the mods' tallies, as every `Dojo` does.
  `baseline_loss(batch)` is the loss's `constant_logit_loss` for that
  batch: per batch, since the item count follows the trainer's budget and
  each deck's visible cards, and duplicate cards shrink a batch's
  negatives.

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

Not built yet: multi-positive SupCon, and mixed contrastive + label-based
training in one step.

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
