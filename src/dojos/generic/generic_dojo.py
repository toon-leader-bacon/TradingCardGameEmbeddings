"""Shared base for every generic dojo cell.

A cell (e.g. SingleCardRegressionDojo) supplies only what differs between
task shapes - its decoder head and loss - and inherits the rest of the
`Dojo` contract (src/dojos/dojo.py): split files, budgeted batching, holdout
filtering, example counts and head management.
"""

import copy
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, List

import torch
from torch import nn

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.card_binder.visible_card_lookup import VisibleCardLookup
from src.dojos.batch import Batch
from src.dojos.budgeted_batching import group_by_budget
from src.dojos.dojo import BatchBudget, DojoBatch
from src.dojos.file_managers.FileManagerParquet import MAX_INT, FileManagerParquet
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.loss.nocab_loss import NocabLoss
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import BatchedModelOutput, TrainingDatum

# Rows converted to examples at a time; independent of the batch budget.
_ROWS_PER_CHUNK = 256


class GenericDojo:
    """Implements `Dojo` for a parquet-backed (input, label) task.

    Inputs (constructor):
        path_to_training_data: a metric's output parquet file. Its stem is
            this dojo's `name`.
        data_constructor: turns rows into TrainingDatum pairs, looking cards
            up through the split's holdout-filtered lookup.
        card_lookup: the full card store; each split sees it through a
            VisibleCardLookup.
        holdout: which cards each split may see. The trainer requires every
            dojo to share the plan's spec.
        decoder_head: this cell's private head, mapping encoder output to
            loss inputs.
        loss_calculator: scores decoder_head's output against labels.
        mod_pipeline: augmentations; None means none.
        rng_seed: seed for split shuffling; None means non-deterministic.
    Output: n/a.
    Side effects: creates/overwrites this dojo's split files under
        data/splits (see FileManagerParquet.make_splits).
    Exceptions: whatever FileManagerParquet raises for a missing or
        malformed path_to_training_data.
    """

    def __init__(
        self,
        path_to_training_data: Path,
        data_constructor: DataConstructor,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        decoder_head: nn.Module,
        loss_calculator: NocabLoss[Any, Any],
        mod_pipeline: ModPipeline | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.name = path_to_training_data.stem
        self.holdout = holdout
        self.rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        # TODO: an output_file_prefix that stays unique per metric sharing a
        # cell (the stem is unique per metric file, but not per directory).
        self.file_manager = FileManagerParquet(
            path_to_training_data,
            output_directory=Path("data/splits"),
            output_file_prefix=path_to_training_data.stem,
            seed=self.rng.randint(0, MAX_INT),
        )
        self.file_manager.make_splits(split_ratios=[8, 1, 1], shuffle=True)

        self.data_constructor = data_constructor
        self.data_mod_pipeline = mod_pipeline or ModPipeline([])
        self.loss_calculator = loss_calculator
        self.decoder_head = decoder_head
        self._initial_head_state = copy.deepcopy(decoder_head.state_dict())
        self._lookups = {
            split: VisibleCardLookup(card_lookup, holdout, split) for split in Split
        }

    def batches(
        self, split: Split, budget: BatchBudget, max_examples: int | None = None
    ) -> Iterator[Batch]:
        """Yield one split's examples as Batches within `budget`.

        Inputs: split (Split), budget (BatchBudget), max_examples (int |
            None): keep only the first N examples in the split file's fixed
            order, so a capped pass is deterministic.
        Output: iterator of Batch, each costing at most budget.max_cost.
        Side effects: reads the split's parquet file.
        Exceptions: ValueError if one example alone exceeds the budget;
            whatever data_constructor.build() raises.

        Example:
            >>> next(dojo.batches(Split.TEST, BatchBudget(64, lambda c: 1)))
        """
        for group in group_by_budget(self._examples(split, max_examples), budget):
            yield Batch.from_training_data(group)

    def example_count(self, split: Split) -> int:
        """Approximate example count: rows in the split file (before skips)."""
        return self.file_manager.row_count(split)

    def compute_loss(
        self, embeddings: BatchedModelOutput, batch: DojoBatch
    ) -> torch.Tensor:
        """Scalar per-example mean loss for one batch's embeddings.

        Inputs: embeddings (one per example, in batch order), batch (a Batch
            yielded by this dojo).
        Output: scalar tensor.
        Side effects: none.
        Exceptions: TypeError if batch isn't a Batch; ValueError if
            len(embeddings) != len(batch); whatever the loss raises for an
            out-of-vocabulary label.
        """
        if not isinstance(batch, Batch):
            raise TypeError(f"{self.name} needs a Batch, got {type(batch).__name__}")
        if len(embeddings) != len(batch):
            raise ValueError(
                f"The number of embeddings ({len(embeddings)}) does not match "
                f"the number of examples ({len(batch)})"
            )
        decoder_output: Any = self.decoder_head(embeddings)
        loss: torch.Tensor = self.loss_calculator.calculate(
            decoder_output, batch.labels
        )
        return loss

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """This dojo's decoder-head parameters (never the encoder's)."""
        return self.decoder_head.parameters()

    def reset_head(self) -> None:
        """Restore the decoder head to its state at construction."""
        self.decoder_head.load_state_dict(self._initial_head_state)

    def _examples(
        self, split: Split, max_examples: int | None
    ) -> Iterator[TrainingDatum]:
        """Stream a split's examples in file order, modded and truncated."""
        lookup = self._lookups[split]
        emitted = 0
        for chunk in self.file_manager.reader_for(split, _ROWS_PER_CHUNK):
            data: List[TrainingDatum] = self.data_constructor.build(chunk, lookup)
            data = self.data_mod_pipeline.apply(data, is_training=split == Split.TRAIN)
            for datum in data:
                if max_examples is not None and emitted >= max_examples:
                    return
                emitted += 1
                yield datum
