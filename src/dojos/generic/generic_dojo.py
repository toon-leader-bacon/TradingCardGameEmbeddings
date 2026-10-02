"""Shared base for every generic dojo cell.

A cell (e.g. SingleCardRegressionDojo) supplies only what differs between
task shapes - its decoder head and loss - and inherits the rest of the
`Dojo` contract (src/dojos/dojo.py): split files, budgeted batching, holdout
filtering, example counts and head management.
"""

import copy
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping

import torch
from torch import nn

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.card_binder.visible_card_lookup import VisibleCardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.version_metadata import metadata_from_schema
from src.dojos.batch import Batch
from src.dojos.budgeted_batching import group_by_budget
from src.dojos.dojo import BatchBudget, DojoBatch
from src.dojos.file_managers.file_manager_parquet import MAX_INT, FileManagerParquet
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.loss.label_stats import LabelStats
from src.dojos.loss.loss_calibration import CalibratedLoss, LossCalibration
from src.dojos.loss.nocab_loss import NocabLoss
from src.dojos.mods.mod import ModTally
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import BatchedModelOutput, TrainingDatum

_logger = logging.getLogger(__name__)

# Rows converted to examples at a time; independent of the batch budget.
_ROWS_PER_CHUNK = 256

# Most TRAIN rows the calibration pass converts to examples: an evenly
# strided sample across the whole TRAIN file (not a prefix - make_splits
# only shuffles within 10k-row blocks, so a prefix inherits the source
# file's order). Small dojos use every TRAIN row.
_CALIBRATION_SAMPLE_CAP = 20_000


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
        loss_calculator: scores decoder_head's output against labels, as
            the cell built it (before calibration).
        calibration: fits loss_calculator to the TRAIN split (see
            src/dojos/loss/loss_calibration.py). The result, one
            CalibratedLoss kept as self._calibrated, is what compute_loss
            scores with (regression: labels z-scored), what baseline_loss()
            returns, and where label_stats is read from.
        mod_pipeline: augmentations; None means none.
        deck_box: only required when path_to_training_data's metric was
            built from a DeckBox (its embedded metadata says
            requires_deck_box) - used to verify that box was minted
            from the same CardBinder version as the metric itself
            (DeckBox.card_binder_version_for()). None otherwise.
        config: this dojo's identity/split-management/version-check
            configuration (see DojoConfig, src/dojos/generic/dojo_config.py)
            - defaults to DojoConfig() (every field at its default).
            A None field falls back as follows (the only place this
            fallback logic lives):
              - self.name = config.name or path_to_training_data.stem.
              - the split-file prefix passed to FileManagerParquet is
                config.output_file_prefix or self.name (so setting
                config.name alone already avoids a split-file collision,
                without also needing output_file_prefix).
              - self.rng is seeded from config.rng_seed (None means
                unseeded).
              - make_splits() runs only when config.force_resplit is
                True or the split files don't already exist
                (FileManagerParquet.splits_exist()) - otherwise the
                existing files are reused untouched.
              - _check_metric_version's strictness is
                config.strict_version_check.
    Output: n/a.
    Side effects: reads a strided sample of the TRAIN split (at most
        _CALIBRATION_SAMPLE_CAP rows, unmodded) through data_constructor
        to calibrate the loss. Creates this dojo's split files under
        config.output_directory (see FileManagerParquet.make_splits)
        unless they already exist and config.force_resplit is False, in
        which case the existing files are left untouched and reused.
    Exceptions: whatever FileManagerParquet raises for a missing or
        malformed path_to_training_data. ValueError if
        config.strict_version_check is True and path_to_training_data's
        version metadata is missing, doesn't match card_lookup, or
        needs a deck_box that wasn't given or whose own recorded
        CardBinder version doesn't match. ValueError from calibration
        if the TRAIN sample is empty, a regression label has zero std, or
        the baseline is zero (see LossCalibration.calibrate).
    """

    def __init__(
        self,
        path_to_training_data: Path,
        data_constructor: DataConstructor,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        decoder_head: nn.Module,
        loss_calculator: NocabLoss[Any, Any],
        calibration: LossCalibration,
        mod_pipeline: ModPipeline | None = None,
        deck_box: DeckBox | None = None,
        config: DojoConfig = DojoConfig(),
    ) -> None:
        self.name = config.name or path_to_training_data.stem
        self.holdout = holdout
        self.rng = (
            random.Random(config.rng_seed)
            if config.rng_seed is not None
            else random.Random()
        )

        self.file_manager = FileManagerParquet(
            path_to_training_data,
            output_directory=config.output_directory,
            output_file_prefix=config.output_file_prefix or self.name,
            seed=self.rng.randint(0, MAX_INT),
        )
        self._check_metric_version(
            path_to_training_data, card_lookup, deck_box, config.strict_version_check
        )
        if config.force_resplit or not self.file_manager.splits_exist():
            self.file_manager.make_splits(split_ratios=[8, 1, 1], shuffle=True)

        self.data_constructor = data_constructor
        self.data_mod_pipeline = mod_pipeline or ModPipeline([])
        self.decoder_head = decoder_head
        self._initial_head_state = copy.deepcopy(decoder_head.state_dict())
        self._lookups = {
            split: VisibleCardLookup(card_lookup, holdout, split) for split in Split
        }
        # Fit the loss to TRAIN, now that the split files and TRAIN lookup exist
        self._calibrated: CalibratedLoss = self._calibrated_loss(
            loss_calculator, calibration
        )

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
        loss: torch.Tensor = self._calibrated.loss.calculate(
            decoder_output, batch.labels
        )
        return loss

    def baseline_loss(self, batch: DojoBatch) -> float:
        """See Dojo.baseline_loss: one constant for every batch, the
        calibration's TRAIN baseline (1.0 for a z-scored regression).

        Inputs: batch (a Batch yielded by this dojo; its contents are not
            read - the baseline is a property of the TRAIN split).
        Output: float, finite and > 0.
        Side effects: none.
        Exceptions: TypeError if batch isn't a Batch.

        Example:
            >>> dojo.baseline_loss(next(dojo.batches(Split.TEST, budget)))
            1.0
        """
        if not isinstance(batch, Batch):
            raise TypeError(f"{self.name} needs a Batch, got {type(batch).__name__}")
        return self._calibrated.baseline_loss

    @property
    def label_stats(self) -> LabelStats | None:
        """TRAIN label stats, for mapping a regression prediction back to
        label units (LabelStats.to_label_units); None for non-regression
        cells. Read off the calibrated loss (CalibratedLoss.label_stats).

        Side effects: none. Exceptions: none.

        Example:
            >>> dojo.label_stats.to_label_units(0.5)  # a regression dojo
            512.3
        """
        return self._calibrated.label_stats

    @property
    def loss_calculator(self) -> NocabLoss[Any, Any]:
        """The loss compute_loss scores with: the cell's loss after
        calibration (a regression cell's is wrapped in StandardizedLabelLoss).

        Side effects: none. Exceptions: none.

        Example:
            >>> isinstance(dojo.loss_calculator, FixedClassificationLoss)
            True
        """
        return self._calibrated.loss

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """This dojo's decoder-head parameters (never the encoder's)."""
        return self.decoder_head.parameters()

    def mod_tallies(self) -> Mapping[str, ModTally]:
        """See Dojo.mod_tallies: its data_mod_pipeline's tallies."""
        return self.data_mod_pipeline.mod_tallies()

    def move_head_to(self, device: torch.device) -> None:
        """Move the decoder head to device, in place."""
        self.decoder_head.to(device)

    def reset_head(self) -> None:
        """Restore the decoder head to its state at construction (on
        whatever device it is on now)."""
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

    def _calibration_sample(self) -> List[TrainingDatum]:
        """An evenly strided, unmodded sample of the TRAIN split's examples.

        Private helper - single caller is __init__. Reads the TRAIN split
        file in chunks, keeps every k-th row (k = ceil(TRAIN rows /
        _CALIBRATION_SAMPLE_CAP), counted across chunks), and builds only
        the kept rows through data_constructor with the TRAIN lookup, so the
        cost is bounded by the cap, not the split size (213k rows -> ~20k
        builds). No mods run: labels are what calibration needs, and a mod
        would advance its RNG and tally before training starts.

        Inputs: none (reads self.file_manager, self.data_constructor and
            self._lookups[Split.TRAIN], all set before the call).
        Output: List[TrainingDatum], in file order; rows the constructor
            skips (unknown or held-out cards) are simply absent.
        Side effects: reads the TRAIN split file.
        Exceptions: ValueError if no TRAIN example survives (an empty or
            fully held-out TRAIN split cannot be calibrated); whatever
            data_constructor.build raises.
        """
        result: List[TrainingDatum] = []
        stride = _calibration_stride(self.file_manager.row_count(Split.TRAIN))
        lookup = self._lookups[Split.TRAIN]

        # Keep every stride-th row, counting rows across chunk boundaries
        rows_before_chunk = 0
        for chunk in self.file_manager.reader_for(Split.TRAIN, _ROWS_PER_CHUNK):
            first_kept = -rows_before_chunk % stride
            kept_rows = chunk.iloc[first_kept::stride]
            rows_before_chunk += len(chunk)
            if len(kept_rows):
                result.extend(self.data_constructor.build(kept_rows, lookup))

        if not result:
            raise ValueError(
                f"{self.name}: no TRAIN example to calibrate the loss on "
                "(empty split, or every row skipped by the data constructor)"
            )
        return result

    def _calibrated_loss(
        self, loss_calculator: NocabLoss[Any, Any], calibration: LossCalibration
    ) -> CalibratedLoss:
        """calibration fit to a TRAIN sample, with one INFO line timing it.

        Private helper - single caller is __init__ (its last step).
        Inputs: the cell's loss and calibration.
        Output: CalibratedLoss.
        Side effects: reads the TRAIN split file; logs the sample size,
            baseline and elapsed seconds at INFO.
        Exceptions: whatever _calibration_sample and calibration.calibrate
            raise (see __init__).
        """
        started = time.monotonic()
        sample = self._calibration_sample()
        result = calibration.calibrate(loss_calculator, sample)
        _logger.info(
            "%s: calibrated loss on %d TRAIN examples in %.2fs (baseline %.4g%s)",
            self.name,
            len(sample),
            time.monotonic() - started,
            result.baseline_loss,
            _label_stats_note(result.label_stats),
        )
        return result

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


def _calibration_stride(train_rows: int) -> int:
    """Keep every k-th TRAIN row so at most _CALIBRATION_SAMPLE_CAP remain.

    Inputs: train_rows (int >= 0).
    Output: k = ceil(train_rows / cap), at least 1.
    Side effects: none. Exceptions: none.
    """
    return max(1, math.ceil(train_rows / _CALIBRATION_SAMPLE_CAP))


def _label_stats_note(label_stats: LabelStats | None) -> str:
    """ ", label mean ... std ..." for the calibration log line, or "".

    Inputs: label_stats. Output: str. Side effects: none. Exceptions: none.
    """
    if label_stats is None:
        return ""
    return f", label mean {label_stats.mean:.4g} std {label_stats.std:.4g}"
