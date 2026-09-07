"""Accumulation metric: how often a run is won, among runs where a
given card was in the deck as of act 2's start.

ACT-2 BOUNDARY: sts_gg run records do not have a fixed floor-to-act
mapping - confirmed by direct inspection (925 sampled runs showed 4
distinct floor-per-actIdx patterns, e.g. act 2 starting at floor 18 in
most runs but floor 17 in others, almost certainly from Neow
bonuses/skipped floors). "Deck at start of act 2" is therefore computed
per-run from hpPerFloor's actIdx field (the minimum floor with
actIdx == 1), never a hardcoded floor cutoff - see
plans/sts_gg_metrics.md. A run with no actIdx == 1 entry in
hpPerFloor never reached act 2 and contributes nothing to any card's
tally - not an error, just no snapshot for that run to fold in.

CARD RESOLUTION / COPY COUNTING: identical rule and decision to
card_upgrade_rate_metric.py's module docstring, independently
duplicated per plans/sts_gg_metrics.md's Explicitly out of scope
section rather than shared.
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
_ACT_2_INDEX = 1


class CardWinRateAtAct2Metric:
    """Accumulation metric: P(win | card in deck_at_start_of_act_2).

    Satisfies the Metric[dict] Protocol (src/data_refinement/metrics/
    metric.py) structurally. accumulate() only tallies in-memory
    per-card (win_count, total_count) pairs; finalize() does the real
    per-card division and write - unlike a streaming metric, no
    output exists until finalize() runs.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/sts_gg/card_win_rate_at_act2.parquet"
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
        self._win_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every card copy present in this run's deck as of act
        2's start toward this metric's running per-card
        (win_count, total_count) pairs.

        Inputs:
            row: one parsed sts_gg run, carrying at least "id" (str),
                "win" (bool), "deck" (list of {"id", "upgraded",
                "floor"} entries), and "hpPerFloor" (list of {"floor",
                "actIdx", ...} entries).
        Output: none.
        Side effects: updates self._total_count/_win_count in place,
            once per physical copy present by act 2's start (see
            module docstrings' ACT-2 BOUNDARY and COPY COUNTING
            sections) - nothing updates if this run never reached act
            2. Emits one logging.error() per deck entry that fails to
            resolve.
        Exceptions: raises if row is missing "id", "win", "deck", or
            "hpPerFloor".

        Example:
            >>> metric = CardWinRateAtAct2Metric(card_binder)
            >>> metric.accumulate(json.loads(next(runs_file)))
            >>> metric.finalize()
        """
        act_2_start_floor = self._act_2_start_floor(row["hpPerFloor"])
        if act_2_start_floor is None:
            # This run never reached act 2 - no snapshot to tally.
            return

        # Tally every card copy present by act 2's start as its own
        # independent sample.
        for card_entry in row["deck"]:
            if card_entry["floor"] > act_2_start_floor:
                continue
            self._tally_card_entry(card_entry, row["win"])

    def finalize(self) -> Path:
        """Compute every seen card's win rate and write one row per
        card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, win_rate: float, sample_count:
            int - one row per card seen at least once).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts_gg/card_win_rate_at_act2.parquet')
        """
        result: list[dict] = []

        # Build one output row per card seen at least once.
        for card_uuid in self._total_count:
            result.append(self._win_rate_row(card_uuid))

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _act_2_start_floor(self, hp_per_floor: list[dict]) -> int | None:
        """Find the earliest floor at which this run's hpPerFloor
        entries record act 2 (actIdx == _ACT_2_INDEX), if any.

        Private helper - single consumer is accumulate(). See module
        docstring's ACT-2 BOUNDARY section.

        Inputs:
            hp_per_floor: one run's "hpPerFloor" list.
        Output: the minimum floor with actIdx == _ACT_2_INDEX, or None
            if this run never reached act 2.
        Side effects: none.
        Exceptions: none.
        """
        act_2_floors = [
            entry["floor"] for entry in hp_per_floor if entry["actIdx"] == _ACT_2_INDEX
        ]
        if not act_2_floors:
            return None
        return min(act_2_floors)

    def _tally_card_entry(self, card_entry: dict, win: bool) -> None:
        """Resolve one deck entry's card id and fold it into this
        metric's running per-card (win_count, total_count) pairs.

        Private helper - single consumer is accumulate().

        Inputs:
            card_entry: one row["deck"] entry, carrying at least "id".
            win: this run's own "win" field.
        Output: none.
        Side effects: increments self._total_count[card_uuid], and
            self._win_count[card_uuid] if win is true, when the card
            resolves; otherwise emits one logging.error() call and
            updates nothing.
        Exceptions: none.
        """
        card_uuid = self._card_uuid(card_entry["id"])
        if card_uuid is None:
            _logger.error(
                "CardWinRateAtAct2Metric: unresolved card id %r - "
                "excluding it from this run's tally",
                card_entry["id"],
            )
            return

        self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
        if win:
            self._win_count[card_uuid] = self._win_count.get(card_uuid, 0) + 1

    def _win_rate_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._total_count.
        Output: a dict with keys "nocab_uuid" (str), "win_rate"
            (float, self._win_count.get(card_uuid, 0) divided by
            self._total_count[card_uuid]), and "sample_count" (int,
            self._total_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        wins = self._win_count.get(card_uuid, 0)
        return {
            "nocab_uuid": str(card_uuid),
            "win_rate": wins / total,
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
