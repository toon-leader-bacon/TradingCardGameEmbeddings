"""Shared base for every generic dojo cell.

A cell (e.g. SingleCardRegressionDojo) supplies only what differs between
task shapes - its decoder head and loss - and inherits the rest of the
`Dojo` contract (src/dojos/dojo.py): split files, budgeted batching, holdout
filtering, example counts and head management.
"""

import copy
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, List

import torch
from torch import nn

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.card_binder.visible_card_lookup import VisibleCardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.version_metadata import metadata_from_schema
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

_logger = logging.getLogger(__name__)

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
        deck_box: only required when path_to_training_data's metric was
            built from a DeckBox (its embedded metadata says
            requires_deck_box) - used to verify that box was minted
            from the same CardBinder version as the metric itself
            (DeckBox.card_binder_version_for()). None otherwise.
        strict_version_check: when True (default), path_to_training_data's
            embedded CardBinder version (see
            src/data_refinement/metrics/version_metadata.py) is checked
            against card_lookup (and, when requires_deck_box, deck_box's
            own recorded CardBinder version) before splitting, and a
            mismatch (or missing metadata) raises. False is the explicit
            escape hatch for "proceed anyway" - it skips the check
            entirely and logs a warning instead.
    Output: n/a.
    Side effects: creates/overwrites this dojo's split files under
        data/splits (see FileManagerParquet.make_splits).
    Exceptions: whatever FileManagerParquet raises for a missing or
        malformed path_to_training_data. ValueError if
        strict_version_check is True and path_to_training_data's
        version metadata is missing, doesn't match card_lookup, or
        needs a deck_box that wasn't given or whose own recorded
        CardBinder version doesn't match.
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
        deck_box: DeckBox | None = None,
        strict_version_check: bool = True,
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
        self._check_metric_version(
            path_to_training_data, card_lookup, deck_box, strict_version_check
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

    def _check_metric_version(
        self,
        path_to_training_data: Path,
        card_lookup: CardLookup,
        deck_box: DeckBox | None,
        strict_version_check: bool,
    ) -> None:
        """Raise if path_to_training_data's embedded version metadata
        doesn't match card_lookup/deck_box - see this class's own
        docstring for strict_version_check's exact contract.

        Private helper - single caller is __init__, called before
        make_splits() so a stale metric is caught before spending time
        re-splitting it.

        Inputs: same as __init__'s own deck_box/strict_version_check,
            plus path_to_training_data and card_lookup.
        Output: none.
        Side effects: none of its own (self.file_manager.schema is a
            pure read of already-loaded state) - logs one
            logging.warning() when strict_version_check is False.
        Exceptions: ValueError - see __init__'s docstring.
        """
        if not strict_version_check:
            _logger.warning(
                "%s: skipping metric version check for %s (strict_version_check=False)",
                self.name,
                path_to_training_data,
            )
            return

        metadata = metadata_from_schema(self.file_manager.schema)
        if metadata is None:
            raise ValueError(
                f"{self.name}: no version metadata found in "
                f"{path_to_training_data} - was this metric regenerated "
                "since this check was added?"
            )

        actual_card_binder_version = card_lookup.version_for(metadata.game)
        if actual_card_binder_version != metadata.card_binder_version:
            raise ValueError(
                f"{self.name}: {path_to_training_data} was built from "
                f"CardBinder version {metadata.card_binder_version!r}, but "
                f"the current one for {metadata.game} is "
                f"{actual_card_binder_version!r} - regenerate this metric"
            )

        if not metadata.requires_deck_box:
            return
        if deck_box is None:
            raise ValueError(
                f"{self.name}: {path_to_training_data} requires a deck_box "
                "to verify, none was given"
            )

        actual_deck_box_binder_version = deck_box.card_binder_version_for(metadata.game)
        if actual_deck_box_binder_version != metadata.card_binder_version:
            raise ValueError(
                f"{self.name}: {path_to_training_data} was built from "
                f"CardBinder version {metadata.card_binder_version!r}, but "
                f"the given deck_box was minted from "
                f"{actual_deck_box_binder_version!r} for {metadata.game} - "
                "regenerate the deck box or this metric"
            )
