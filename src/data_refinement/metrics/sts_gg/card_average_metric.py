"""Template Method base for accumulation metrics that tally, per card
seen anywhere in a run's final deck, the running average of one scalar
already present on that same run's raw row (e.g. average relicCount
across every run a card appeared in).

Mirrors deck_label_metric.py's relationship to deck_label_metrics.py,
one level down: several per-card metrics
(card_average_metrics.py) are otherwise identical except for which
field of the row feeds the average and what its output column is
named - this class owns the shared sequence (per-copy iteration,
card resolution, running sum/count, final division), and a subclass
only fixes LABEL_COLUMN/DEFAULT_OUTPUT_PATH and implements
_value_for_run().

WIN RATE IS AN AVERAGE: CardWinRateMetric (card_average_metrics.py)
folds win/loss through this same machinery by treating a win as 1.0
and a loss as 0.0 - a rate is just the average of an indicator
variable, so it needs no separate accumulation shape.

ACCUMULATION, NOT STREAMING: like CardUpgradeRateMetric/
CardWinRateAtAct2Metric, a card's average is only knowable after
scanning every run it appears in, so accumulate() only tallies
in-memory per-card (value_sum, total_count) pairs and finalize() does
the real division and write.

COPY COUNTING / CARD RESOLUTION: same rules as every other metric in
this container - a card appearing twice in one run's final deck is
tallied as two independent samples, and card ids are resolved via the
same "CARD."-prefix-strip + CardBinder.get_by_alias rule (both
deliberately duplicated per card_upgrade_rate_metric.py's module
docstring, not shared via a cross-cutting helper yet).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_STS_GG_CARD_ID_PREFIX = "CARD."


class CardAverageMetric(ABC):
    """Per-card running average of one run-level scalar, across every
    run a card appears in.

    Satisfies the Metric[dict] Protocol (../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox | None = None,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve sts_gg's "CARD.<id>"
                references against - must already have spire_codex's
                cards ingested (this class never writes to it).
            deck_box: accepted only for constructor-shape consistency
                with this container's deck-input metrics (e.g.
                DeckLabelMetric subclasses) - see ../README.md. This
                metric's output is per-card, not per-deck, so it never
                reads from or writes into deck_box.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none - no I/O happens until finalize() is
            called.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._total_count: dict[UUID, int] = {}
        self._value_sum: dict[UUID, float] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every card copy in one run's final deck toward this
        metric's running per-card (value_sum, total_count) pairs.

        Inputs:
            row: one parsed sts_gg run, carrying at least "deck" (list
                of {"id": str, ...} card entries - the run's final
                deck) plus whatever field _value_for_run() reads.
        Output: none.
        Side effects: updates self._value_sum/_total_count in place,
            once per physical copy in row["deck"] (see module
            docstring's COPY COUNTING section). Emits one
            logging.error() per deck entry that fails to resolve.
        Exceptions: raises if row is missing "deck", or whatever
            _value_for_run() raises for a missing field of its own.

        Example:
            >>> metric = SomeCardAverageMetric(card_binder)
            >>> metric.accumulate(json.loads(next(runs_file)))
            >>> metric.finalize()
        """
        value = float(self._value_for_run(row))

        for card_entry in row["deck"]:
            card_uuid = self._card_uuid(card_entry["id"])
            if card_uuid is None:
                _logger.error(
                    "%s: unresolved card id %r - excluding it from this " "run's tally",
                    type(self).__name__,
                    card_entry["id"],
                )
                continue

            self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
            self._value_sum[card_uuid] = self._value_sum.get(card_uuid, 0.0) + value

    def finalize(self) -> Path:
        """Compute every seen card's average and write one row per
        card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, self.LABEL_COLUMN: float,
            sample_count: int - one row per card seen at least once).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts_gg/some_average.parquet')
        """
        result: list[dict] = [
            self._average_row(card_uuid) for card_uuid in self._total_count
        ]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    @abstractmethod
    def _value_for_run(self, row: dict) -> Any:
        """Pull this metric's one run-level scalar off a raw run row.

        The only step of accumulate()'s sequence a subclass overrides.

        Inputs:
            row: one parsed sts_gg run - same row accumulate() received.
        Output: a value convertible to float (an int, a float, or a
            bool - see module docstring's WIN RATE IS AN AVERAGE
            section for why a bool is a legitimate return here).
        Side effects: implementation-defined (expected: none - a plain
            field read).
        Exceptions: implementation-defined (expected: raises if row is
            missing the field this subclass reads).
        """
        raise NotImplementedError

    def _average_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._total_count.
        Output: a dict with keys "nocab_uuid" (str), self.LABEL_COLUMN
            (float, self._value_sum[card_uuid] divided by
            self._total_count[card_uuid]), and "sample_count" (int,
            self._total_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        return {
            "nocab_uuid": str(card_uuid),
            self.LABEL_COLUMN: self._value_sum[card_uuid] / total,
            "sample_count": total,
        }

    def _card_uuid(self, raw_card_id: str) -> UUID | None:
        """Look up the nocab_uuid for one sts_gg deck entry's card id.

        Private helper - single consumer is accumulate(). See module
        docstring's CARD RESOLUTION section.

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
