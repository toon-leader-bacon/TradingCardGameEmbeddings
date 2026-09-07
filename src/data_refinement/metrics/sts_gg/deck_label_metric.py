"""Template Method base for streaming metrics whose training input is
one sts_gg run's final deck (referenced by deck_uuid, never embedded -
see ../hash_utils.py and card_upgrade_rate_metric.py's module docstring
"Deck references" discussion, also folded into ../README.md) paired
with a single scalar label that's already present on that same run's
raw row - no cross-run aggregation, no per-run lookup beyond the row
itself.

Eleven concrete metrics (deck_label_metrics.py) share this exact
sequence of steps - resolve the final deck's cards, hash them, write
the deck into a private DeckBox, pull one scalar off the row, write
one output row - and differ only in which field of the row that scalar
comes from and what type/column name it's written under. That's a
textbook Template Method (PATTERNS.md): this class owns every shared
step; a subclass only fixes LABEL_COLUMN/LABEL_TYPE/DEFAULT_OUTPUT_PATH
and implements _label_for_run().

ascension_prediction_metric.py's AscensionPredictionMetric is the same
shape and subclasses this base too, even though it predates it - it
was hand-written before this Template Method existed, and folding it
in here removes what would otherwise be its own independent copy of
this exact sequence.

CARD RESOLUTION: same "CARD."-prefix-strip + CardBinder.get_by_alias
rule duplicated (deliberately, per card_upgrade_rate_metric.py's module
docstring) by every other metric in this container.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.schema.card import GenericDeck
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_STS_GG_CARD_ID_PREFIX = "CARD."


class DeckLabelMetric(ABC):
    """One run's final deck -> (deck_uuid, label), one row per run,
    written as soon as accumulate() sees it.

    Structural Metric[dict] (../metric.py) - every concrete subclass
    satisfies it, since accumulate()/finalize() are defined here and
    inherited unchanged.
    """

    LABEL_COLUMN: ClassVar[str]
    LABEL_TYPE: ClassVar[pa.DataType]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve sts_gg's "CARD.<id>"
                references against - must already have spire_codex's
                cards ingested (this class never writes to it).
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into - shared with any other
                metric in the same scan pass that also takes a
                DeckBox, so identical final decks dedupe against each
                other. Never the published deck_box/sts_gg/ box.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a pyarrow.parquet.ParquetWriter held
            open for the lifetime of this instance - callers MUST call
            finalize() when done, or the file is left incomplete.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
        """
        self._card_binder = card_binder
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = pa.schema(
            [
                ("run_id", pa.string()),
                ("deck_uuid", pa.string()),
                (self.LABEL_COLUMN, self.LABEL_TYPE),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Convert one parsed sts_gg run into a single output row and
        write it immediately.

        Inputs:
            row: one parsed JSON object from sts_gg's runs.jsonl,
                carrying at least "id" (str), "deck" (list of {"id":
                str, ...} card entries - the run's final deck), plus
                whatever field _label_for_run() reads.
        Output: none.
        Side effects: writes exactly one row to the open
            ParquetWriter. Writes this run's final deck into
            self._deck_box via create_if_absent() (a no-op if an
            identical deck was already written by this or another
            metric sharing the same box). Emits one logging.error()
            per deck entry that fails to resolve against card_binder.
        Exceptions: raises if row is missing "id" or "deck", or
            whatever _label_for_run() raises for missing fields of its
            own.

        Example:
            >>> metric = SomeDeckLabelMetric(card_binder, deck_box)
            >>> metric.accumulate(json.loads(next(runs_file)))
            >>> metric.finalize()
        """
        run_id = row["id"]
        label = self._label_for_run(row)

        card_nocab_uuids: list[UUID] = []
        for card_entry in row["deck"]:
            card_uuid = self._card_uuid(card_entry["id"])
            if card_uuid is None:
                _logger.error(
                    "%s: run %s references unresolved card id %r - "
                    "excluding it from this run's final deck",
                    type(self).__name__,
                    run_id,
                    card_entry["id"],
                )
                continue
            card_nocab_uuids.append(card_uuid)

        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        self._deck_box.create_if_absent(
            GenericDeck(
                nocab_uuid=deck_uuid,
                source_game=GameId.SLAY_THE_SPIRE_2,
                name=f"sts_gg run {run_id} final deck",
                card_nocab_uuids=card_nocab_uuids,
            )
        )

        output_row = pa.Table.from_pydict(
            {
                "run_id": [run_id],
                "deck_uuid": [str(deck_uuid)],
                self.LABEL_COLUMN: [label],
            },
            schema=self._output_schema,
        )
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        A true no-op relative to data - every row this instance will
        ever write was already written by accumulate(). Idempotent: a
        second call is a no-op, so scan_runs_jsonl's unconditional
        finalize() call is always safe. Does NOT save self._deck_box -
        that's the calling driver's own responsibility, since the box
        is shared across metrics and only the driver knows when every
        metric sharing it is done.

        Inputs: none.
        Output: output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts_gg/some_label.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    @abstractmethod
    def _label_for_run(self, row: dict) -> Any:
        """Pull this metric's one scalar label off a raw run row.

        The only step of accumulate()'s sequence a subclass overrides -
        every other step (deck resolution, hashing, DeckBox write,
        output row write) is shared in this base class.

        Inputs:
            row: one parsed sts_gg run - same row accumulate() received.
        Output: the label value for self.LABEL_COLUMN - must already
            match self.LABEL_TYPE's Python representation (e.g. a
            plain int for pa.int64(), a str or None for a nullable
            pa.string()).
        Side effects: implementation-defined (expected: none - a plain
            field read).
        Exceptions: implementation-defined (expected: raises if row is
            missing the field this subclass reads).
        """
        raise NotImplementedError

    def _card_uuid(self, raw_card_id: str) -> UUID | None:
        """Look up the nocab_uuid for one sts_gg deck entry's card id.

        Private helper - single consumer is accumulate(). Same
        "CARD." prefix-strip + alias lookup rule as every other metric
        in this container (see module docstring).

        Inputs:
            raw_card_id: one deck entry's "id" field, e.g.
                "CARD.GRAND_FINALE".
        Output: the matching nocab_uuid, or None if unresolved.
        Side effects: none.
        Exceptions: none.
        """
        stripped_id = raw_card_id.removeprefix(_STS_GG_CARD_ID_PREFIX)
        card = self._card_binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, stripped_id
        )
        return card.nocab_uuid if card is not None else None
