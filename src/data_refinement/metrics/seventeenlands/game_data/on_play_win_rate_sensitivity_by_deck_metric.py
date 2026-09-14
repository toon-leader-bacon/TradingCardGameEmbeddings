"""OnPlayWinRateSensitivityByDeckMetric - BRAINSTORM.md's multi-card
metric "On-Play Win-Rate Sensitivity by Deck": the deck-level mirror of
on_play_win_rate_delta_metric.py's OnPlayWinRateDeltaMetric - per deck,
P(won | on_play) - P(won | on_draw), aggregated across every game
sharing an identical deck.

ACCUMULATION, NOT STREAMING: unlike game_deck_label_metric.py's
GameDeckLabelMetric family, this metric's label needs cross-row
aggregation (every game with an identical deck contributes to that
deck's own delta), so it does NOT subclass GameDeckLabelMetric - that
class's whole shape assumes the label is already fully known from one
row alone. This duplicates GameDeckLabelMetric.accumulate()'s deck-
identification-and-hashing steps (present_uuids over deck_columns,
deck_uuid_from_cards(), DeckBox.create_if_absent()) rather than sharing
them across the streaming/accumulation split - the same call
draft_data's pack_to_pick_choice_set_metric.py/
pool_conditioned_pick_metric.py already made for a similar near-
duplicate pair (see that container's README "Open questions"
precedent).

NULLABLE OUTPUT: same on-play/on-draw-side-must-both-have-samples rule
as OnPlayWinRateDeltaMetric - if a deck was never seen on one side, its
delta is written as None.
"""

from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck.parquet"
)


class OnPlayWinRateSensitivityByDeckMetric:
    """Deck -> P(won | on_play) - P(won | on_draw), aggregated across
    every game sharing that exact deck.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's deck_<name>
                column suffixes against - assumed already fully
                populated for source_game. Never queried directly by
                this class - only through the GameCardColumns this
                constructor builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own GameCardColumns.
            source_game: which game's cards header names are matched
                against.
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into - shared with any other
                metric in the same scan pass that also takes a
                deck_box, so identical decks dedupe against each other.
                Never the published deck_box/ box.
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
        self._source_game = source_game
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._win_count: dict[tuple[UUID, bool], int] = {}
        self._total_count: dict[tuple[UUID, bool], int] = {}

    def accumulate(self, row: dict) -> None:
        """Identify this row's deck, write it into self._deck_box, and
        tally its (win_count, total_count) keyed by (deck_uuid,
        on_play).

        Inputs:
            row: one game_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: updates self._win_count/_total_count in place.
            Writes this row's deck into self._deck_box via
            create_if_absent() (a no-op if an identical deck was
            already written by this or another metric sharing the same
            box).
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = OnPlayWinRateSensitivityByDeckMetric(
            ...     card_binder, header, GameId.MTG, deck_box
            ... )
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        card_nocab_uuids = self._game_columns.present_uuids(
            row, self._game_columns.deck_columns
        )
        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        self._deck_box.create_if_absent(
            self._deck_for_row(row, deck_uuid, card_nocab_uuids)
        )

        on_play = bool(row["on_play"])
        won = bool(row["won"])
        key = (deck_uuid, on_play)
        self._total_count[key] = self._total_count.get(key, 0) + 1
        if won:
            self._win_count[key] = self._win_count.get(key, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen deck's on-play/on-draw win rate delta and
        write one row per deck to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns deck_uuid: str, on_play_win_rate_sensitivity: float
            | None, sample_count: int - one row per deck seen at least
            once on either side).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck.parquet')
        """
        result: list[dict] = [
            self._sensitivity_row(deck_uuid) for deck_uuid in self._distinct_decks()
        ]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _deck_for_row(
        self, row: dict, deck_uuid: UUID, card_nocab_uuids: list[UUID]
    ) -> GenericDeck:
        """Build the GenericDeck this row's deck_<name> multiset
        represents, for writing into self._deck_box.

        Private helper - single consumer is accumulate(). Same shape as
        game_deck_label_metric.py's GameDeckLabelMetric._deck_for_row()
        - deliberately duplicated, not shared, per this module's own
        docstring.

        Inputs:
            row: one game_data CSV row, dict-like.
            deck_uuid: this deck's already-hashed identity.
            card_nocab_uuids: this row's already-matched deck_<name>
                present cards.
        Output: a GenericDeck (src/schema/card.py).
        Side effects: none.
        Exceptions: none expected.
        """
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=self._source_game,
            name=(
                f"game_data {row['draft_id']}/{row['match_number']}/"
                f"{row['game_number']} deck"
            ),
            card_nocab_uuids=card_nocab_uuids,
        )

    def _distinct_decks(self) -> list[UUID]:
        """Every distinct deck_uuid tallied at least once, on either
        side of on_play.

        Private helper - single consumer is finalize().

        Inputs: none (uses accumulated state).
        Output: every distinct key[0] across self._total_count, order
            not guaranteed.
        Side effects: none.
        Exceptions: none.
        """
        return list({deck_uuid for deck_uuid, _ in self._total_count})

    def _win_rate(self, deck_uuid: UUID, on_play: bool) -> float | None:
        """This deck's win rate on one side of on_play, or None if it
        was never seen on that side.

        Private helper - single consumer is _sensitivity_row().

        Inputs:
            deck_uuid: deck to look up.
            on_play: which side to compute the rate for.
        Output: self._win_count[(deck_uuid, on_play)] /
            self._total_count[(deck_uuid, on_play)], or None if
            self._total_count has no entry for that key.
        Side effects: none.
        Exceptions: none.
        """
        key = (deck_uuid, on_play)
        if key not in self._total_count:
            return None
        return self._win_count.get(key, 0) / self._total_count[key]

    def _sensitivity_row(self, deck_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied deck.

        Private helper - single consumer is finalize().

        Inputs:
            deck_uuid: a deck seen in self._total_count on at least one
                side.
        Output: a dict with keys "deck_uuid" (str),
            "on_play_win_rate_sensitivity" (float | None - None if
            either side's _win_rate() is None), "sample_count" (int,
            this deck's total_count summed across both sides).
        Side effects: none.
        Exceptions: none.
        """
        on_play_rate = self._win_rate(deck_uuid, True)
        on_draw_rate = self._win_rate(deck_uuid, False)

        sensitivity = (
            on_play_rate - on_draw_rate
            if on_play_rate is not None and on_draw_rate is not None
            else None
        )
        sample_count = self._total_count.get(
            (deck_uuid, True), 0
        ) + self._total_count.get((deck_uuid, False), 0)

        return {
            "deck_uuid": str(deck_uuid),
            "on_play_win_rate_sensitivity": sensitivity,
            "sample_count": sample_count,
        }
