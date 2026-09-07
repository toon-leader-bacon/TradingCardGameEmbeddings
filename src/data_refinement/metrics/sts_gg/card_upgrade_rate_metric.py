"""Accumulation metric: how often a card that's in a run's final deck
also ends that run upgraded.

CARD RESOLUTION: duplicates legacy/sts_gg/deck_outcome_metric.py's own
"CARD."-prefix-strip + CardBinder.get_by_alias(GameId.SLAY_THE_SPIRE_2,
DataSource.SPIRE_CODEX, ...) rule rather than sharing a helper with it
- see plans/sts_gg_metrics.md's Explicitly out of scope section; this
is the project's 4th/5th independent site with this exact logic,
deliberately left duplicated pending one later comprehensive
cross-cutting dedup pass, not extracted per-addition. On a miss: log
loudly and skip just that one deck entry, never drop the whole run.

COPY COUNTING: a card appearing twice in one run's final deck (e.g.
two upgraded GRAND_FINALE copies) is tallied as two independent
samples, not deduplicated to one-per-run - see plans/sts_gg_metrics.md.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_STS_GG_CARD_ID_PREFIX = "CARD."


class CardUpgradeRateMetric:
    """Accumulation metric: P(card upgraded by run end | card in final
    deck).

    Satisfies the Metric[dict] Protocol (src/data_refinement/metrics/
    metric.py) structurally. accumulate() only tallies in-memory
    per-card counts; finalize() does the real per-card division and
    write - unlike a streaming metric, no output exists until
    finalize() runs.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/sts_gg/card_upgrade_rate.parquet"
    )

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
                with this container's multi-card metrics (e.g.
                AscensionPredictionMetric) - see plans/sts_gg_metrics.md.
                This metric's output is per-card, not per-deck, so it
                never reads from or writes into deck_box.
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
        self._upgraded_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every card copy in one run's final deck toward this
        metric's running per-card upgrade counts.

        Inputs:
            row: one parsed sts_gg run, carrying at least "deck" (list
                of {"id": str, "upgraded": bool, ...} card entries -
                other per-entry fields like "floor" aren't used by
                this metric).
        Output: none.
        Side effects: updates self._total_count/_upgraded_count in
            place, once per physical copy in row["deck"] (see module
            docstring's COPY COUNTING section). Emits one
            logging.error() per deck entry that fails to resolve.
        Exceptions: raises if row is missing "deck".

        Example:
            >>> metric = CardUpgradeRateMetric(card_binder)
            >>> metric.accumulate(json.loads(next(runs_file)))
            >>> metric.finalize()
        """
        # Tally every physical card copy as its own independent
        # sample.
        for card_entry in row["deck"]:
            self._tally_card_entry(card_entry)

    def finalize(self) -> Path:
        """Compute every seen card's upgrade rate and write one row
        per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, upgrade_rate: float, sample_count:
            int - one row per card seen at least once).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts_gg/card_upgrade_rate.parquet')
        """
        result: list[dict] = []

        # Build one output row per card seen at least once.
        for card_uuid in self._total_count:
            result.append(self._upgrade_rate_row(card_uuid))

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _tally_card_entry(self, card_entry: dict) -> None:
        """Resolve one deck entry's card id and fold it into this
        metric's running per-card counts.

        Private helper - single consumer is accumulate().

        Inputs:
            card_entry: one row["deck"] entry, carrying "id" (str) and
                "upgraded" (bool).
        Output: none.
        Side effects: increments self._total_count[card_uuid], and
            self._upgraded_count[card_uuid] if card_entry["upgraded"]
            is true, when the card resolves; otherwise emits one
            logging.error() call and updates nothing.
        Exceptions: none.
        """
        card_uuid = self._card_uuid(card_entry["id"])
        if card_uuid is None:
            _logger.error(
                "CardUpgradeRateMetric: unresolved card id %r - excluding "
                "it from this run's tally",
                card_entry["id"],
            )
            return

        self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
        if card_entry["upgraded"]:
            self._upgraded_count[card_uuid] = self._upgraded_count.get(card_uuid, 0) + 1

    def _upgrade_rate_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._total_count.
        Output: a dict with keys "nocab_uuid" (str), "upgrade_rate"
            (float, self._upgraded_count.get(card_uuid, 0) divided by
            self._total_count[card_uuid]), and "sample_count" (int,
            self._total_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        upgraded = self._upgraded_count.get(card_uuid, 0)
        return {
            "nocab_uuid": str(card_uuid),
            "upgrade_rate": upgraded / total,
            "sample_count": total,
        }

    def _card_uuid(self, raw_card_id: str) -> UUID | None:
        """Look up the nocab_uuid for one sts_gg deck entry's card id.

        Private helper - single consumer is _tally_card_entry(). See
        module docstring's CARD RESOLUTION section: strips sts_gg's
        "CARD." prefix and looks the remainder up directly against
        spire_codex's own registered aliases.

        Inputs:
            raw_card_id: one deck entry's "id" field, e.g.
                "CARD.GRAND_FINALE".
        Output: the matching nocab_uuid, or None if no card is
            registered under spire_codex's alias for the stripped id.
        Side effects: none.
        Exceptions: none.
        """
        stripped_id = raw_card_id.removeprefix(_STS_GG_CARD_ID_PREFIX)
        card = self._card_binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, stripped_id
        )
        return card.nocab_uuid if card is not None else None
