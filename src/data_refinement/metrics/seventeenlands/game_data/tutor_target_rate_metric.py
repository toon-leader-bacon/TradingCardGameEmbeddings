"""TutorTargetRateMetric - BRAINSTORM.md's single-card metric "Tutor
Target Rate": P(card in tutored_<name> | card in deck_<name>) - among
games where a card was in the deck, how often did a tutor effect
actually fetch it that game.

Standalone accumulation, structurally close to draft_data's take-rate
shape (a ratio of two per-card tallies,
draft_data/pack_card_tally_metric.py's PackCardTallyMetric) but NOT a
subclass of that class - PackCardTallyMetric's shape is specific to
draft_data's pack/pick concept (row["pick"] equality against a pack
option); this metric's numerator is set membership (was this card
present in tutored_<name> this game), a different accumulate() entirely,
and reaching into draft_data/ for a base would be exactly the kind of
cross-cousin-directory import PRINCIPLES.md flags as a smell.

tutored_<name> is rare (~8.5% of games have any hit at all per
BRAINSTORM.md) - sample_count is written alongside the rate so a
consumer can filter out noisy near-zero estimates from cards seen in
few games, per BRAINSTORM.md's own flag for this metric.
"""

from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/tutor_target_rate.parquet"
)


class TutorTargetRateMetric:
    """Card -> P(tutored | in deck).

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's deck_<name>/
                tutored_<name> column suffixes against - assumed
                already fully populated for source_game. Never queried
                directly by this class - only through the
                GameCardColumns this constructor builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own GameCardColumns.
            source_game: which game's cards header names are matched
                against.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none beyond building this instance's own
            GameCardColumns from card_binder/header - no further I/O
            happens until finalize() is called.
        Exceptions: none.
        """
        self._game_columns = GameCardColumns.from_header(
            header, card_binder, source_game
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._times_in_deck: dict[UUID, int] = {}
        self._times_tutored: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every deck-present card's (times_in_deck,
        times_tutored) toward this metric's running state.

        Inputs:
            row: one game_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: updates self._times_in_deck/_times_tutored in
            place, once per card present (count > 0) in deck_<name> on
            this row.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = TutorTargetRateMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        # Build this row's tutored-uuid set once, rather than
        # re-querying present_uuids() per deck card.
        tutored_uuids = set(
            self._game_columns.present_uuids(row, self._game_columns.tutored_columns)
        )

        for card_uuid in self._game_columns.present_uuids(
            row, self._game_columns.deck_columns
        ):
            self._times_in_deck[card_uuid] = self._times_in_deck.get(card_uuid, 0) + 1
            if card_uuid in tutored_uuids:
                self._times_tutored[card_uuid] = (
                    self._times_tutored.get(card_uuid, 0) + 1
                )

    def finalize(self) -> Path:
        """Compute every seen card's tutor target rate and write one
        row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, tutor_target_rate: float,
            sample_count: int - one row per card seen in deck_<name> at
            least once).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/tutor_target_rate.parquet')
        """
        result: list[dict] = [
            self._tutor_rate_row(card_uuid) for card_uuid in self._times_in_deck
        ]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _tutor_rate_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._times_in_deck.
        Output: a dict with keys "nocab_uuid" (str), "tutor_target_rate"
            (float, self._times_tutored.get(card_uuid, 0) /
            self._times_in_deck[card_uuid]), "sample_count" (int,
            self._times_in_deck[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        times_in_deck = self._times_in_deck[card_uuid]
        return {
            "nocab_uuid": str(card_uuid),
            "tutor_target_rate": self._times_tutored.get(card_uuid, 0) / times_in_deck,
            "sample_count": times_in_deck,
        }
