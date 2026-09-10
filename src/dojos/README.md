# dojos

Pluggable auxiliary training tasks for the card embedding model. Each
dojo presents the same contract to a training loop - `training_data()`/
`test_data()`/`validation_data()` generators yielding `Batch` (see
`[batch.py](batch.py)`), plus `compute_loss(embeddings, labels)` - while
reading its own labels from a parquet file produced by one of
`../data_refinement/metrics/`'s metric classes.

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
`BceLoss`, `PickPredictionCrossEntropyLoss`). A generic dojo builds whichever  
loss its task shape needs internally; callers never construct one  
directly.

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
- `MaskedFieldDataConstructor` - for `MaskedFieldMetric`'s family
  - Single card training datum, where the specified card data json field is masked out before returning. Classification output (guess what the field was masked out, like rarity, card type, power, cost, etc).
- `DeckLabelDataConstructor` - for `DeckLabelMetric`'s family
  - A multi-card input (a deck of some type) with
- `DeckCardMaskDataConstructor` - for `DeckCardMaskMetric`'s family
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

### `generic/pooling.py` - shared multi-card pooling strategy

Every multi-card generic cell faces the same problem before it can run
its MLP: a deck is a *variable-length* list of card embeddings, not
one fixed-size vector like a single card.  `MeanEmbeddingPooler` is the only concrete pooler implemented so far.

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

## Thin per-metric wrapper convention

A per-source package (`gwent_one/`, `play_gwent/`, `sts_gg/`, etc.) holds thin generic-dojo subclasses, one per concrete metric.
This is the prefer style for developing new dojos per metric, but exotic metrics may need a fully custom dojo that dose not fit neatly into an generic dojo cell/ style/ family.
