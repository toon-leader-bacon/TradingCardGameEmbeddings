"""TurnsToGameEndAfterCastMetric - plans/replay_data_metrics.md's
"Turns-To-Game-End After Cast": row["num_turns"] minus the turn a card
was FIRST cast, for a card matched out of creatures_cast/
non_creatures_cast (either actor - not deck-conditioned).

Standalone - not ReplayTurnEventRateMetric-shaped (its value is a
delta, not a hit/miss ratio) and not AverageTurnCastMetric-shaped
either (it needs the FIRST occurrence only, plus a second row-level
value, num_turns, that AverageTurnCastMetric never reads).

OPEN QUESTION SETTLED (plans/replay_data_metrics.md's "Open
questions" #1): verified directly against
data/raw/17lands/replay_data/MSH.PremierDraft.csv (a 25-row sample,
cross-referencing num_turns against the last turn number with any
populated per-turn action column - lands_played/creatures_cast/
non_creatures_cast/cards_drawn/creatures_attacked/creatures_blocking -
for each actor). Result: num_turns matches max(last real user turn,
last real oppo turn) in every sampled row (never their SUM) - i.e.
num_turns is on the SAME per-actor turn-number scale as
user_turn_N/oppo_turn_N (roughly "the highest turn number either
player reached before the game ended"), not a combined elapsed-turn
count across both players. first_cast_turn is therefore directly
comparable to num_turns with no unit conversion - the plan's formula
is used as written.
"""

from pathlib import Path
from typing import ClassVar, Iterable, Literal
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/turns_to_game_end_after_cast.parquet"
)


class TurnsToGameEndAfterCastMetric:
    """Card -> average (num_turns - first cast turn), across every
    game it's cast in at all.

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
            card_binder: registry to match this CSV's per-turn
                creatures_cast/non_creatures_cast Arena-ID cells
                against - assumed already fully populated for
                source_game. Never queried directly by this class -
                only through the ReplayCardColumns this constructor
                builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own ReplayCardColumns.
            source_game: which game's cards header names are matched
                against.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none beyond building this instance's own
            ReplayCardColumns from card_binder/header.
        Exceptions: none.
        """
        self._replay_columns = ReplayCardColumns.from_header(
            header, card_binder, source_game
        )
        self._version_metadata = MetricVersionMetadata(
            game=source_game, card_binder_version=card_binder.version_for(source_game)
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._delta_sum: dict[UUID, float] = {}
        self._occurrence_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every cast card's (num_turns - first_cast_turn) delta
        toward this metric's running per-card state.

        Inputs:
            row: one replay_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: for every card cast at least once this game
            (either actor), finds that card's first_cast_turn via
            _first_cast_turn() and updates self._delta_sum/
            self._occurrence_count in place.
        Exceptions: implementation-defined (expected: none for a
            well-formed row).

        Example:
            >>> metric = TurnsToGameEndAfterCastMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        num_turns = row["num_turns"]
        for card_uuid in self._every_cast_card(row):
            first_cast_turn = self._first_cast_turn(row, card_uuid)
            if first_cast_turn is None:
                continue

            self._delta_sum[card_uuid] = self._delta_sum.get(card_uuid, 0.0) + (
                num_turns - first_cast_turn
            )
            self._occurrence_count[card_uuid] = (
                self._occurrence_count.get(card_uuid, 0) + 1
            )

    def finalize(self) -> Path:
        """Compute every seen card's average
        turns-to-game-end-after-cast and write one row per card to
        self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str,
            turns_to_game_end_after_cast: float, sample_count: int).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/turns_to_game_end_after_cast.parquet')
        """
        result = [self._delta_row(card_uuid) for card_uuid in self._occurrence_count]

        write_dataframe_with_version_metadata(
            pd.DataFrame(result), self._output_path, self._version_metadata
        )
        return self._output_path

    def _first_cast_turn(self, row: dict, card_uuid: UUID) -> int | None:
        """The smallest turn (searched ascending, either actor) at
        which card_uuid appears in that half-turn's creatures_cast/
        non_creatures_cast on this row.

        Private helper - single consumer is accumulate().

        Inputs:
            row: one replay_data CSV row, dict-like.
            card_uuid: card to search for.
        Output: the first matching turn number, or None if card_uuid
            never appears in either actor's creatures_cast/
            non_creatures_cast this game. See module docstring's
            settled open question on what "first" means when actor
            turn-counters aren't a shared elapsed-turn index - this
            returns the smallest turn NUMBER seen on either actor,
            not a globally-ordered occurrence.
        Side effects: none.
        Exceptions: none expected.
        """
        candidate_turns = []
        for actor in ReplayCardColumns.ACTORS:
            turn_numbers = (
                self._replay_columns.user_turn_numbers
                if actor == "user"
                else self._replay_columns.oppo_turn_numbers
            )
            for turn in turn_numbers:
                cast_uuids = self._cast_uuids_for_turn(row, actor, turn)
                if card_uuid in cast_uuids:
                    candidate_turns.append(turn)
                    break
        return min(candidate_turns) if candidate_turns else None

    def _cast_uuids_for_turn(
        self, row: dict, actor: Literal["user", "oppo"], turn: int
    ) -> set[UUID]:
        """Every card matched out of this half-turn's creatures_cast
        union non_creatures_cast cell.

        Private helper - single consumer is _first_cast_turn()/
        _every_cast_card().

        Inputs:
            row: one replay_data CSV row, dict-like.
            actor: which half-turn - "user" or "oppo".
            turn: that actor's own turn-number counter.
        Output: every matched card in creatures_cast union
            non_creatures_cast for this half-turn.
        Side effects: none.
        Exceptions: none expected.
        """
        creatures = self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "creatures_cast")]
        )
        non_creatures = self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "non_creatures_cast")]
        )
        return set(creatures) | set(non_creatures)

    def _every_cast_card(self, row: dict) -> set[UUID]:
        """Every card matched out of any actor's creatures_cast/
        non_creatures_cast cell anywhere on this row.

        Private helper - single consumer is accumulate().

        Inputs:
            row: one replay_data CSV row, dict-like.
        Output: the union of every matched card across every
            half-turn's creatures_cast/non_creatures_cast cell, either
            actor.
        Side effects: none.
        Exceptions: none expected.
        """
        card_uuids: set[UUID] = set()
        for actor in ReplayCardColumns.ACTORS:
            turn_numbers = (
                self._replay_columns.user_turn_numbers
                if actor == "user"
                else self._replay_columns.oppo_turn_numbers
            )
            for turn in turn_numbers:
                card_uuids.update(self._cast_uuids_for_turn(row, actor, turn))
        return card_uuids

    def _delta_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._occurrence_count.
        Output: a dict with keys "nocab_uuid" (str),
            "turns_to_game_end_after_cast" (float,
            self._delta_sum[card_uuid] /
            self._occurrence_count[card_uuid]), "sample_count" (int,
            self._occurrence_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        count = self._occurrence_count[card_uuid]
        return {
            "nocab_uuid": str(card_uuid),
            "turns_to_game_end_after_cast": self._delta_sum[card_uuid] / count,
            "sample_count": count,
        }
