"""Dojo for the pick-prediction metric.

Multi-group in ([pack_cards, pool_cards]), single classification label out
(which pack card was picked): trains the card embedding model to predict
the drafter's actual pick, given what's in the pack and what's already in
the pool (see
src/data_refinement/seventeenlands/draft_game_metrics/picked_v_held_v_pack_metric.py
for how that label is produced). Rough sketch, marshaling the pieces in
this package - details/edge cases still to be discussed.
"""

import random
from pathlib import Path
from typing import Generator, List

import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.batch import Batch
from src.dojos.file_managers.FileManagerParquet import (
    MAX_INT,
    FileManagerParquet,
    ParquetChunkReader,
)
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.seventeenlands.draft_game_metrics.pick_prediction.data_constructor import (
    PickPredictionDataConstructor,
)
from src.dojos.seventeenlands.draft_game_metrics.pick_prediction.decoder_head import (
    PickPredictionDecoderHead,
)
from src.schema.type_hints import BatchedModelOutput, Label, TrainingDatum


class PickPredictionDojo:
    def __init__(
        self,
        path_to_training_data: Path,
        card_binder: CardBinder,
        card_embedding_size: int,
        rng_seed: int | None = None,
    ):
        self.card_embedding_size = card_embedding_size
        self.rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        self.file_manager = FileManagerParquet(
            path_to_training_data,
            output_directory=Path("data/splits"),
            output_file_prefix="pick_prediction_",
            seed=self.rng.randint(0, MAX_INT),
        )
        self.file_iterators = self.file_manager.make_splits(
            split_ratios=[8, 1, 1],
            shuffle=True,
        )

        # Takes raw data (picked_card_uuid, pack_cards_uuids,
        # pool_cards_uuids) rows and converts them into
        # (MultiGroupInput, Label) TrainingDatum pairs.
        self.data_constructor = PickPredictionDataConstructor(card_binder)
        # Takes (inputs, labels) and applies any necessary transformations to
        # enrich/augment the data for training.
        self.data_mod_pipeline = ModPipeline([])

        # Takes the decoder head's output and the labels, computes the loss.
        self.loss_calculator = PickPredictionCrossEntropyLoss()
        self.decoder_head = PickPredictionDecoderHead(card_embedding_size)

    def training_data(self) -> Generator[Batch, None, None]:
        train_file_iterator = self.file_iterators[0]
        yield from self._data_iterator(train_file_iterator, apply_mod_pipeline=True)

    def test_data(self) -> Generator[Batch, None, None]:
        test_file_iterator = self.file_iterators[1]
        yield from self._data_iterator(test_file_iterator, apply_mod_pipeline=False)

    def validation_data(self) -> Generator[Batch, None, None]:
        validation_file_iterator = self.file_iterators[2]
        yield from self._data_iterator(
            validation_file_iterator, apply_mod_pipeline=False
        )

    def _data_iterator(
        self, file_iterator: ParquetChunkReader, apply_mod_pipeline: bool = False
    ) -> Generator[Batch, None, None]:
        for chunk in file_iterator:
            data: List[TrainingDatum] = self.data_constructor.build(chunk)
            if apply_mod_pipeline:
                data = self.data_mod_pipeline.apply(data)
            yield Batch.from_training_data(data)

    def compute_loss(
        self, embeddings: BatchedModelOutput, labels: List[Label]
    ) -> torch.Tensor:
        """Compute the loss between the embeddings and the labels.

        We expect len(embeddings) == len(labels), in matching order - the
        trainer passes the labels back to us alongside the embeddings it
        produced from the inputs we handed it.
        """
        if len(embeddings) != len(labels):
            raise ValueError(
                f"The number of embeddings ({len(embeddings)}) does not match "
                f"the number of labels ({len(labels)})"
            )

        decoder_output = self.decoder_head(embeddings)
        return self.loss_calculator.calculate(decoder_output, labels)
