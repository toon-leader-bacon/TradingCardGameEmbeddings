"""Generic dojo for the (ragged pack-options + a pool conditioning
group in, which option was picked out) task shape. See
multi_card_option_selection/dojo.py for the sibling with no
conditioning group, and that module's docstring plus
src/dojos/generic/option_scoring.py for why these two cells share the
same OptionScoringHead Strategy rather than each hardcoding their own
scoring architecture.

Today's only consumer is PoolConditionedPickMetric, via the thin
wrapper at src/dojos/seventeenlands/draft_data/
pool_conditioned_pick_dojo.py.
"""

import random
from pathlib import Path
from typing import Generator, List

import torch

from src.dojos.batch import Batch
from src.dojos.file_managers.FileManagerParquet import (
    MAX_INT,
    FileManagerParquet,
    ParquetChunkReader,
)
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.multi_group_option_selection.decoder_head import (
    MultiGroupOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import OptionScoringHead
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.type_hints import BatchedModelOutput, Label, TrainingDatum


class MultiGroupOptionSelectionDojo:
    """Wires a DataConstructor, ModPipeline, decoder head, and loss into
    the train/test/validation split + loss-computation contract every
    dojo presents to a training loop. Callers never construct the
    decoder head or loss themselves - loss is always
    PickPredictionCrossEntropyLoss, same as
    MultiCardOptionSelectionDojo's sibling cell. scoring_head/pooler are
    this cell's swappable seams, forwarded straight through to the
    decoder head it builds."""

    def __init__(
        self,
        path_to_training_data: Path,
        data_constructor: DataConstructor,
        card_embedding_size: int,
        mod_pipeline: ModPipeline | None = None,
        scoring_head: OptionScoringHead | None = None,
        pooler: EmbeddingPooler | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """
        Inputs:
            path_to_training_data: a metric's own output parquet file
                (e.g. PoolConditionedPickMetric.DEFAULT_OUTPUT_PATH).
            data_constructor: converts raw rows from that file into
                (MultiGroupInput, int) TrainingDatum pairs - a
                [pack_option_cards, pool_cards] group pair plus the
                picked option's index within pack_option_cards (e.g.
                PoolConditionedPickDataConstructor, src/dojos/generic/
                data_constructors.py - see its docstring for why group
                order is fixed).
            card_embedding_size: dimensionality of the card embeddings
                this dojo's decoder head will receive.
            mod_pipeline: transformations applied to training (not
                test/validation) data before batching. Defaults to an
                empty pipeline when not given.
            scoring_head: the OptionScoringHead strategy forwarded to
                this dojo's decoder head (see option_scoring.py). None
                (default) uses BilinearOptionScoringHead.
            pooler: the EmbeddingPooler strategy forwarded to this
                dojo's decoder head for collapsing the pool group. None
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

        self.loss_calculator = PickPredictionCrossEntropyLoss()
        self.decoder_head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size, scoring_head=scoring_head, pooler=pooler
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
        """Yield the test split's data as Batches, unmodded.

        Inputs: none.
        Output: a generator of Batch, one per chunk of the test split.
        Side effects: none beyond reading the test split file.
        Exceptions: whatever ParquetChunkReader raises.
        """
        test_file_iterator = self.file_iterators[1]
        yield from self._data_iterator(test_file_iterator, is_training=False)

    def validation_data(self) -> Generator[Batch, None, None]:
        """Yield the validation split's data as Batches, unmodded.

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
        """Compute the pick-selection loss between predicted per-option
        logits and the true picked-option index.

        Inputs:
            embeddings: one example's [option_embeddings,
                pool_embeddings] group pair per training example, same
                order as labels. len(embeddings) must equal
                len(labels).
            labels: the ground-truth picked-option index for each
                example (an int, indexing into that same example's own
                option_embeddings), as produced by this dojo's
                data_constructor.
        Output: a scalar loss tensor.
        Side effects: none.
        Exceptions: ValueError if len(embeddings) != len(labels).
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
        validation_data. Identical shape to MultiCardOptionSelectionDojo's
        sibling _data_iterator - see that method's docstring for the
        near-identical-logic tradeoff this repeats.

        Private helper - single set of callers are the three split
        methods above.

        Inputs:
            file_iterator: a split's chunked parquet reader.
            is_training: whether this is the training split. Passed
                through to self.data_mod_pipeline.apply() on every
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
