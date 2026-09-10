"""Generic dojo for the (whole deck in, fixed classification out) task
shape.

Sibling to MultiCardRegressionDojo (src/dojos/generic/
multi_card_regression/dojo.py) - same constructor/method shape plus
label_values, same relationship SingleCardFixedClassificationDojo
(src/dojos/generic/single_card_fixed_classification/dojo.py) has to
its own regression sibling. Today's only concrete consumer is
CharacterPredictionMetric (src/dojos/sts_gg/deck_label_dojos.py's
CharacterDojo wrapper).
"""

import random
from pathlib import Path
from typing import Generator, List, Sequence

import torch

from src.dojos.batch import Batch
from src.dojos.file_managers.FileManagerParquet import (
    MAX_INT,
    FileManagerParquet,
    ParquetChunkReader,
)
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.multi_card_fixed_classification.decoder_head import (
    MultiCardFixedClassificationDecoderHead,
)
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.type_hints import BatchedModelOutput, Label, TrainingDatum


class MultiCardFixedClassificationDojo:
    """Wires a DataConstructor, ModPipeline, decoder head, and loss into
    the train/test/validation split + loss-computation contract every
    dojo presents to a training loop. Callers never construct the
    decoder head or loss themselves - both are this dojo's own
    implementation detail, built from label_values (fixed
    classification always uses FixedClassificationLoss, same as
    SingleCardFixedClassificationDojo). The one addition over that
    single-card sibling is an optional pooler, forwarded straight
    through to the decoder head it builds - same pooler-forwarding
    relationship MultiCardRegressionDojo has to
    SingleCardRegressionDojo."""

    def __init__(
        self,
        path_to_training_data: Path,
        data_constructor: DataConstructor,
        label_values: Sequence[str],
        card_embedding_size: int,
        mod_pipeline: ModPipeline | None = None,
        pooler: EmbeddingPooler | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """
        Inputs:
            path_to_training_data: a metric's own output parquet file
                (e.g. CharacterPredictionMetric's DEFAULT_OUTPUT_PATH).
            data_constructor: converts raw rows from that file into
                (MultiCardInput, str) TrainingDatum pairs - the
                metric-family-specific collaborator (e.g.
                DeckLabelDataConstructor with label_caster=str) a thin
                per-metric wrapper configures and injects.
            label_values: this metric's full OBSERVED label vocabulary,
                fixed order (index i is class i) - see
                FixedClassificationLoss's docstring for why this must
                be the full observed set (OTHER sentinel included where
                applicable), not just a metric's "normal" values.
            card_embedding_size: dimensionality of the card embeddings
                this dojo's decoder head will receive.
            mod_pipeline: transformations applied before batching. Runs
                on every split, not just training - see
                SingleCardFixedClassificationDojo's own mod_pipeline
                docstring for why.
            pooler: the EmbeddingPooler strategy forwarded to this
                dojo's decoder head (see decoder_head.py). None
                (default) uses MeanEmbeddingPooler.
            rng_seed: seed for split shuffling; None means
                non-deterministic.
        Output: none (constructor).
        Side effects: see FileManagerParquet.make_splits() - creates/
            overwrites this dojo's split files under data/splits.
        Exceptions: whatever FileManagerParquet raises for a missing/
            malformed path_to_training_data.
        """
        self.card_embedding_size = card_embedding_size
        self.label_values = list(label_values)
        self.rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        self.file_manager = FileManagerParquet(
            path_to_training_data,
            output_directory=Path("data/splits"),
            output_file_prefix=path_to_training_data.stem,
            seed=self.rng.randint(0, MAX_INT),
        )
        self.file_iterators = self.file_manager.make_splits(
            split_ratios=[8, 1, 1],
            shuffle=True,
        )

        self.data_constructor = data_constructor
        self.data_mod_pipeline = mod_pipeline or ModPipeline([])

        self.loss_calculator = FixedClassificationLoss(self.label_values)
        self.decoder_head = MultiCardFixedClassificationDecoderHead(
            card_embedding_size, len(self.label_values), pooler=pooler
        )

    def training_data(self) -> Generator[Batch, None, None]:
        """Yield the training split's data as Batches, with this dojo's
        mod pipeline applied.

        Inputs: none.
        Output: a generator of Batch, one per chunk of the training
            split.
        Side effects: none beyond reading the training split file.
        Exceptions: whatever ParquetChunkReader raises.
        """
        train_file_iterator = self.file_iterators[0]
        yield from self._data_iterator(train_file_iterator, is_training=True)

    def test_data(self) -> Generator[Batch, None, None]:
        """Yield the test split's data as Batches. This dojo's mod
        pipeline still runs here - see __init__'s mod_pipeline
        docstring.

        Inputs: none.
        Output: a generator of Batch, one per chunk of the test split.
        Side effects: none beyond reading the test split file.
        Exceptions: whatever ParquetChunkReader raises.
        """
        test_file_iterator = self.file_iterators[1]
        yield from self._data_iterator(test_file_iterator, is_training=False)

    def validation_data(self) -> Generator[Batch, None, None]:
        """Yield the validation split's data as Batches. See
        test_data()'s docstring for why the mod pipeline still runs
        here.

        Inputs: none.
        Output: a generator of Batch, one per chunk of the validation
            split.
        Side effects: none beyond reading the validation split file.
        Exceptions: whatever ParquetChunkReader raises.
        """
        validation_file_iterator = self.file_iterators[2]
        yield from self._data_iterator(validation_file_iterator, is_training=False)

    def compute_loss(
        self, embeddings: BatchedModelOutput, labels: List[Label]
    ) -> torch.Tensor:
        """Compute the classification loss between predicted embeddings
        and true labels.

        Inputs:
            embeddings: one deck's card embeddings per training
                example, same order as labels. len(embeddings) must
                equal len(labels).
            labels: the ground-truth raw string label for each deck, as
                produced by this dojo's data_constructor. Every label
                MUST be a member of self.label_values (see
                FixedClassificationLoss).
        Output: a scalar loss tensor.
        Side effects: none.
        Exceptions: ValueError if len(embeddings) != len(labels).
            ValueError if any label isn't in self.label_values.
        """
        result: torch.Tensor

        if len(embeddings) != len(labels):
            raise ValueError(
                f"The number of embeddings ({len(embeddings)}) does not match "
                f"the number of labels ({len(labels)})"
            )

        decoder_output = self.decoder_head(embeddings)
        result = self.loss_calculator.calculate(decoder_output, labels)
        return result

    def _data_iterator(
        self, file_iterator: ParquetChunkReader, is_training: bool = False
    ) -> Generator[Batch, None, None]:
        """Shared chunk-to-Batch conversion for training_data/test_data/
        validation_data. Identical shape to
        MultiCardRegressionDojo._data_iterator() - see that method's
        docstring; copied rather than shared, same near-identical-logic
        reasoning SingleCardFixedClassificationDojo._data_iterator()
        already documents.

        Private helper - single set of callers are the three split
        methods above.

        Inputs:
            file_iterator: a split's chunked parquet reader.
            is_training: whether this is the training split. Passed
                through to self.data_mod_pipeline.apply() on every
                split - each mod decides for itself (via its own
                train_only) whether it actually runs for a non-training
                split.
        Output: a generator of Batch, one per chunk read from
            file_iterator.
        Side effects: none beyond reading file_iterator.
        Exceptions: whatever self.data_constructor.build() raises.
        """
        for chunk in file_iterator:
            data: List[TrainingDatum] = self.data_constructor.build(chunk)
            data = self.data_mod_pipeline.apply(data, is_training=is_training)
            yield Batch.from_training_data(data)
