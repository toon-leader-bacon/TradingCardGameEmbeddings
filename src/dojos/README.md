# dojos

Pluggable auxiliary training tasks for the card embedding model. Most
dojos present the same contract to a training loop - `training_data()`/
`test_data()`/`validation_data()` generators yielding `Batch` (see
`[batch.py](batch.py)`), plus `compute_loss(embeddings, labels)` - while
reading its own labels from a parquet file produced by one of
`../data_refinement/metrics/`'s metric classes. An "exotic" dojo whose
task shape genuinely doesn't fit that contract (e.g. `[contrastive/](contrastive/)`,
below) presents its own split-generator + loss shape instead, sourced
directly from `../data_refinement/`'s stores rather than a metric's
parquet output.

A small set of reusable generic dojo classes, one per `(input shape, task shape)` cell, plus
thin per-metric subclasses that configure and inject a
metric-family-specific `DataConstructor` into whichever generic cell
that family's task shape needs.

## Shared plumbing (used by both generations)

- `[batch.py](batch.py)` - `Batch`, the (inputs, labels) container every
dojo's split generators yield.
- `[mods/](mods/)` - `Mod`/`ModPipeline`, composable training-data
transformations (e.g. masking) applied before batching. Each `Mod`
declares its own `train_only: bool` at construction (default `True`)
  - a plain augmentation mod (shuffling a deck, reordering json keys)
  only runs during training; a mod that's a structural requirement of
  the task itself (e.g. `MaskTargetKeyMod`, which must keep a field
  masked at test/validation time too, or the model could just read the
  answer off the input) is constructed with `train_only=False`.
- `[file_managers/FileManagerParquet.py](file_managers/FileManagerParquet.py)` -
`FileManagerParquet`/`ParquetChunkReader`, splits a metric's output
parquet file into train/test/validation files and streams each in
chunks.
- `[loss/](loss/)` - `NocabLoss` Protocol plus concrete losses  
(`MseLoss`, `FixedClassificationLoss`, `SoftClassificationLoss`,  
`MaskedVectorRegressionLoss`, `BceLoss`, `PickPredictionCrossEntropyLoss`).
A generic dojo builds whichever loss its task shape needs internally;
callers never construct one directly.

## `generic/` - the generic dojo "cell"

`[generic/data_constructor.py](generic/data_constructor.py)` defines  
the `DataConstructor` Protocol every metric-family-specific constructor  
satisfies: `build(chunk: pd.DataFrame) -> List[TrainingDatum]`, converting  
one chunk of a metric's raw output rows into (input, label) pairs. A  
generic dojo takes a `DataConstructor` as an injected collaborator  
rather than owning one, so the same generic dojo class serves every  
metric family that shares its `(input shape, task shape)` cell.

- `CardAverageDataConstructor` - for `CardAverageMetric`'s family.
  - Single card training datum input, regression (float) output.
  - `uuid_column` (default `"nocab_uuid"`) is configurable the same way
    `label_column` is, for a metric whose id column is named
    differently because a row isn't "the" card in the usual
    single-card-per-row sense - e.g. `TutorTargetPoolMetric`'s
    `"pool_card_uuid"` (`TutorTargetPoolDojo`).
- `MaskedFieldDataConstructor` - for `MaskedFieldMetric`'s family
  - Single card training datum, where the specified card data json field is masked out before returning. Classification output (guess what the field was masked out, like rarity, card type, power, cost, etc).
- `DeckLabelDataConstructor` - for `DeckLabelMetric`'s family
  - A multi-card input (a deck of some type) with a classification output
- `DeckCardMaskDataConstructor` - for `DeckCardMaskMetric`'s family
  - A multi-card input where one card is targeted for exclusion. The training task is to guess what the missing card is.
- `PickNumberDecayCurveDataConstructor` - for `PickNumberDecayCurveMetric`
  - Single card input, a sparse `{bucket_index: take_rate}` output - one entry per pick-number bucket with enough samples to trust (a configurable `min_sample_count` threshold), rather than every bucket. Paired with `single_card_fixed_classification`'s cell (see below) via a non-default `loss_factory`.
- `PackToPickChoiceSetDataConstructor` / `PoolConditionedPickDataConstructor` -
  for `PackToPickChoiceSetMetric` / `PoolConditionedPickMetric`
  - A ragged pack of option cards input (plus a second, optional pool
    card group for the latter), the picked option's POSITION within
    that row's own option list as output - a label shape no other
    `DataConstructor` needed before. Skips the whole row if ANY option
    fails to resolve (unlike `DeckLabelDataConstructor`'s "drop
    individual unresolved cards, keep the row" convention - here the
    label is a position that would otherwise go stale). Feed
    `multi_card_option_selection`/`multi_group_option_selection` (see
    below).
- `AttackerBlockerCombatOutcomeDataConstructor` - for
  `AttackerBlockerCombatOutcomeMetric`
  - Two card groups (attackers, blockers) input, a signed net-kill-count
    delta (float) output - a genuine regression over two groups, not a
    position/selection label, so unlike the pair above it just reuses
    `_cards_for_uuids()` for both groups directly. Skips a row only if
    the attacker group is empty after resolution (mirrors
    `DeckLabelDataConstructor`'s "empty result -> skip the row"
    convention); an empty blocker group (an unblocked attack) is a
    valid, expected result, never a skip condition. Feeds
    `multi_group_regression` (see below).
- Other custom data constructors, etc.

### `generic/single_card_regression/` - single card in, scalar out

`SingleCardRegressionDojo` (`dojo.py`) is the generic cell for
`(SingleCardInput, Regression)`: single card in, one float label out.
Takes `path_to_training_data`, an injected `DataConstructor`,
`card_embedding_size`, and optionally a `ModPipeline`/`rng_seed`.
Internally builds `MseLoss` and its own
`SingleCardRegressionDecoderHead` (`decoder_head.py` - a small MLP down
to one scalar) - callers never construct either directly.

### `generic/single_card_fixed_classification/` - single card in, closed-vocabulary class out

`SingleCardFixedClassificationDojo` (`dojo.py`) is the generic cell for  
`(SingleCardInput, Fixed classification)`: single card in, one class  
label out of a closed, per-metric vocabulary. Same constructor/method  
shape as `SingleCardRegressionDojo`, plus `label_values: Sequence[str]`

For single card masking tasks, this is what is typically used where the masking is implemented by a masking Mod which is applied to the training, test and validation data (typically trying to predict the masked out item)

`loss_factory` (default `FixedClassificationLoss`) lets a caller swap in
a differently-scored loss over this same decoder head and split/batching
plumbing, for a label shape that isn't one hard class per example:
`SoftClassificationLoss` (a probability distribution over `label_values`
that sums to 1 - `CardCharacterPredictionDojo`) and
`MaskedVectorRegressionLoss` (independent per-position probabilities
that don't sum to 1, some positions masked out per example -
`PickNumberDecayCurveDojo`) are the two consumers today.

### `generic/pooling.py` - shared multi-card pooling strategy

Every multi-card generic cell faces the same problem before it can run
its MLP: a deck is a *variable-length* list of card embeddings, not
one fixed-size vector like a single card.  `MeanEmbeddingPooler` is the only concrete pooler implemented so far.

### `generic/option_scoring.py` - shared per-option scoring strategy

Shared by `multi_card_option_selection`/`multi_group_option_selection`
(below) for their "score each option in a ragged pack" task. Strategy
(PATTERNS.md) - `OptionScoringHead` scores one example's ragged option
embeddings, optionally conditioned on a single pooled context vector,
producing one logit per option. Deliberately swappable rather than
fixed: a low-capacity scorer forces more of the "why is this card good
here" reasoning into the shared card embeddings themselves; a
high-capacity one (e.g. full self-attention across the pack) can solve
the task by absorbing that reasoning into its own weights instead,
leaving embeddings comparatively under-constrained - swapping capacity
for research should mean swapping this one collaborator, not
rewriting either cell. `BilinearOptionScoringHead` is the only
concrete implementation today - a compatibility score between each
option and its optional context, low-capacity enough to still require
real embedding structure to solve the task.

### `generic/multi_card_regression/` - whole deck in, scalar out

`MultiCardRegressionDojo` (`dojo.py`) is the generic cell for
`(MultiCardInput, Regression)`: whole deck in, one float label out.

### `generic/multi_card_binary_classification/` - whole deck in, single logit out

`MultiCardBinaryClassificationDojo` (`dojo.py`) is the generic cell for
`(MultiCardInput, Binary classification)`: whole deck in, one raw logit out. Similar to MultiCardRegressionDojo, but uses the bce_loss only expecting decoder head to output in the range [0.0, 1.0].

### `generic/multi_card_fixed_classification/` - whole deck in, closed-vocabulary class out

`MultiCardFixedClassificationDojo` (`dojo.py`) is the generic cell for  
`(MultiCardInput, Fixed classification)`: whole deck in, one class  
label out of a closed, per-metric vocabulary.

### `generic/multi_card_option_selection/` - ragged pack of options in, which one was picked out

`MultiCardOptionSelectionDojo` (`dojo.py`) is the generic cell for a
ragged pack of option cards in (`MultiCardInput`), one logit per
option out - unlike every other multi-card cell above, this never
pools the input down to a single vector; the whole point is a
per-option score, not one score for the whole set. Delegates scoring
to an injected `OptionScoringHead` (`option_scoring.py`, above; `None`
defaults to `BilinearOptionScoringHead`). Loss is always
`PickPredictionCrossEntropyLoss` (`src/dojos/loss/`) - parameterless,
so unlike `single_card_fixed_classification` there's nothing
per-metric to build it from. Today's only consumer:
`PackToPickChoiceSetDojo`.

### `generic/multi_group_option_selection/` - ragged pack of options + a conditioning group in, which option was picked out

`MultiGroupOptionSelectionDojo` (`dojo.py`) is the sibling cell for
`MultiGroupInput = [pack_option_cards, pool_cards]` - the same
per-option selection task, plus a second card group pooled (via an
injected `EmbeddingPooler`, `pooling.py`) into a single context vector
that conditions each option's score. **Group order is load-bearing**:
`src/schema/type_hints.py`'s `input_shape_of()` classifies shape by
peeking group 0 only and raises on an empty list there - pack options
(never empty) must stay index 0, the conditioning group (legitimately
empty, e.g. a drafter's pool on the first pick of a draft) must stay
index 1. Same loss as `multi_card_option_selection`. Today's only
consumer: `PoolConditionedPickDojo`.

### `generic/multi_group_regression/` - two card groups in, scalar regression out

`MultiGroupRegressionDojo` (`dojo.py`) is the sibling cell to
`multi_group_option_selection` for a genuine two-group *regression*
task - both groups contribute symmetrically to one aggregate label,
so there's no "which one" being scored. `MultiGroupRegressionDecoderHead`
pools each group to a vector via a single shared `EmbeddingPooler`
(one instance for both groups, not one per group), concatenates the
pair (`[group_0_vector, group_1_vector]`, fixed order - no separator/
segment token needed, since a plain `Linear` layer already has
separate learned weights per input dimension, unlike a self-attention
sequence that would need one), and runs a small MLP down to one
scalar. When group 1 is empty, a learned `nn.Parameter` placeholder
stands in for its pooled vector rather than calling the pooler on an
empty list (undefined - see `EmbeddingPooler`) or assuming a plain
zero vector is semantically neutral. Same group-0-must-be-non-empty
ordering rule as `multi_group_option_selection` (`input_shape_of()`
peeks group 0 only). Loss is always `MseLoss`. Today's only consumer:
`AttackerBlockerCombatOutcomeDojo`.

## `contrastive/` - exotic dojo for CLIP/InfoNCE-style contrastive training

Trains the card embedding model directly on co-occurrence-in-a-deck,
sourced straight from `../data_refinement/deck_box/`'s `DeckBox` and
`../data_refinement/card_binder/`'s `CardLookup` - no metric, no
parquet file, no learned decoder head. Doesn't fit the generic dojo
cell pattern above: contrastive loss (InfoNCE) needs every item's
embedding in a batch jointly (to build the similarity comparison
across in-batch positives and negatives), not the row-independent
`decoder_output, labels -> loss` shape every generic cell assumes.

- `[deck_box_dealer.py](contrastive/deck_box_dealer.py)` - `DeckBoxDealer`,
  a `FileManagerParquet`-equivalent for `DeckBox`: partitions one
  game's decks into a seeded train/test/validation split at
  construction, and `training_decks()`/`test_decks()`/`validation_decks()`
  yield fixed-size groups of raw, unmodified `GenericDeck` objects per
  split (a "deck sample" - deliberately not "batch," which is reserved
  for `ContrastiveBatch` below). Mechanical, non-swappable.
- `[pair_constructor.py](contrastive/pair_constructor.py)` - `ContrastivePairConstructor`
  Strategy: turns one deck sample into one `ContrastiveBatch`, deciding
  what counts as a positive pair and how items are sampled. This is the
  actual research surface - a different positive-pair definition (same-
  color, same-archetype, etc.) or item shape (single-card vs. multi-
  card group) is a new class here, not a change to `ContrastiveDojo`.
  `SingleCardPairConstructor` is the only concrete implementation so
  far: single-card items, `items_per_deck` sampled per deck (skipping
  and logging a deck with fewer known cards than that), every same-deck
  item marked mutually positive.
- `[contrastive_batch.py](contrastive/contrastive_batch.py)` - `ContrastiveBatch`,
  the `(items, identities, positive_cliques)` container a
  `ContrastivePairConstructor` builds: `items` is one flat pool (every
  item acts as both anchor and candidate for every other item in the
  same batch), `identities` is a tuple of card uuid(s) per item -
  positionally parallel to that item's own card order, not deduplicated
  or reordered - so exact-duplicate cards can be excluded from the
  negative pool at loss time without being precomputed or stored, and
  `positive_cliques` is index groups - each entry a set of item indices
  that are all mutually positive (one entry per surviving source deck
  for `SingleCardPairConstructor`). Enforces one shared `InputShape`
  across `items` the same way `Batch` does today.
- `[contrastive_loss.py](contrastive/contrastive_loss.py)` - `ContrastiveLoss`
  Strategy, sibling to but distinct from `NocabLoss` (batch-level, not
  row-independent). Takes `item_embeddings: BatchedModelOutput` -
  a concrete implementation validates its own expected per-item shape
  (via `src/schema/type_hints.py`'s `output_shape_of()`) immediately and
  raises `ValueError` on a mismatch, rather than a caller pooling to one
  fixed shape first. `SingleCardInfoNCELoss` is the only concrete
  implementation so far: single-positive InfoNCE via a numerically
  stable `log_softmax` over cosine similarity, computed once across the
  whole flat item pool (so every pairwise deck comparison falls out of
  one similarity matrix, never special-cased per deck pair), excluding
  exact-identity duplicates from a given anchor's negative pool.
- `[dojo.py](contrastive/dojo.py)` - `ContrastiveDojo`, wiring a
  `DeckBoxDealer` + `ContrastivePairConstructor` + `CardLookup` into
  `training_data()`/`test_data()`/`validation_data()` generators
  yielding `ContrastiveBatch` (skipping and logging a degenerate batch
  - no positive clique of size >= 2 - rather than letting it fail deep
  inside the loss), plus `compute_loss(item_embeddings, batch)` - a
  batch-level entry point, not `Dojo`'s `compute_loss(embeddings,
  labels)`, since this task shape has no per-example label at all.
  Owns its `ContrastiveLoss` internally (an injected collaborator with a
  sensible default), same convention as every generic dojo cell. A
  future item-embedding-pooling need is a `ContrastiveLoss` Decorator
  (PATTERNS.md) wrapping another `ContrastiveLoss`, not a separate
  strategy `ContrastiveDojo` owns - there is no `ItemEmbeddingStrategy`
  here.

No training loop drives this yet - `../training/demo_training_loop.py`
is stale and unrelated (it targets `Dojo`'s row-independent shape).
Single-card items only; a multi-card item shape, alternate
`ContrastivePairConstructor`/`ContrastiveLoss` implementations
(multi-positive SupCon, a pooling `ContrastiveLoss` Decorator), and a
mixed contrastive + label-based training loop are all future work, not
yet built.

## Thin per-metric wrapper convention

A per-source package (`gwent_one/`, `play_gwent/`, `sts_gg/`, etc.) holds thin generic-dojo subclasses, one per concrete metric.
This is the prefer style for developing new dojos per metric, but exotic metrics may need a fully custom dojo that dose not fit neatly into an generic dojo cell/ style/ family.

`src/dojos/seventeenlands/{draft_data,game_data,replay_data}/` follows
this convention for all 26 of 17lands' metrics - every metric in that
source has a thin wrapper dojo today, one file per metric module.
