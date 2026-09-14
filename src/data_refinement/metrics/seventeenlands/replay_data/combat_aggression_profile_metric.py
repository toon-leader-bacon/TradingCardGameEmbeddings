"""CombatAggressionProfileMetric - plans/replay_data_metrics.md's "Full
Deck -> Combat-Aggression-Profile Prediction": given the user's full
deck_<name> list, predict an aggregate combat-tempo scalar derived from
this same game's per-turn columns - the average len(creatures_attacked)
per user half-turn that had any attack at all.

Streaming - one row already carries a complete example. Structurally a
GameDeckLabelMetric-style class (deck identification + hashing +
DeckBox write + one output row), except the label isn't already
sitting on the row - it's computed via this class's own turn loop.
Given that one real difference, this duplicates GameDeckLabelMetric's
deck-handling steps directly rather than subclassing across the
draft_data/game_data/replay_data sibling-container boundary
(PRINCIPLES.md section 3's cross-cousin-directory caution) - mirroring
how game_data.OnPlayWinRateSensitivityByDeckMetric already made the
same call against its own accumulation/streaming split.

deck_box is a required constructor parameter (genuinely used) -
following the "required only where used" convention every metric in
this container otherwise follows.
"""

from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/combat_aggression_profile.parquet"
)

_OUTPUT_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("match_number", pa.int64()),
        ("game_number", pa.int64()),
        ("deck_uuid", pa.string()),
        ("combat_aggression_profile", pa.float64()),
    ]
)


class CombatAggressionProfileMetric:
    """One game's constructed (user) deck -> (deck_uuid, average
    attackers-per-attacking-turn), one row per game, written as soon as
    accumulate() sees it.

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
                this class - only through the ReplayCardColumns this
                constructor builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own ReplayCardColumns.
            source_game: which game's cards header names are matched
                against.
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into - shared with any other
                metric in the same scan pass that also takes a
                deck_box, so identical decks dedupe against each other.
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
        self._replay_columns = ReplayCardColumns.from_header(
            header, card_binder, source_game
        )
        self._source_game = source_game
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, _OUTPUT_SCHEMA)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Convert one game_data row into a single output row and write
        it immediately.

        Inputs:
            row: one replay_data CSV row, dict-like - carrying at least
                draft_id/match_number/game_number and this row's
                deck_<name>/per-turn creatures_attacked columns.
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter.
            Writes this row's deck into self._deck_box via
            create_if_absent().
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = CombatAggressionProfileMetric(card_binder, header, GameId.MTG, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        card_nocab_uuids = self._replay_columns.present_uuids(
            row, self._replay_columns.deck_columns
        )
        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        self._deck_box.create_if_absent(
            self._deck_for_row(row, deck_uuid, card_nocab_uuids)
        )

        profile = self._aggression_profile(row)
        output_row = self._output_row(row, deck_uuid, profile)
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        A true no-op relative to data - every row this instance will
        ever write was already written by accumulate(). Idempotent: a
        second call is a no-op. Does NOT save self._deck_box - that's
        the calling driver's own responsibility.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/combat_aggression_profile.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _deck_for_row(
        self, row: dict, deck_uuid: UUID, card_nocab_uuids: list[UUID]
    ) -> GenericDeck:
        """Build the GenericDeck this row's deck_<name> multiset
        represents, for writing into self._deck_box.

        Private helper - single consumer is accumulate(). Same shape as
        game_data.GameDeckLabelMetric._deck_for_row().

        Inputs:
            row: one replay_data CSV row, dict-like.
            deck_uuid: this deck's already-hashed identity.
            card_nocab_uuids: this row's already-matched deck_<name>
                present cards.
        Output: a GenericDeck (src/schema/card.py) named e.g.
            f"replay_data {row['draft_id']}/{row['match_number']}/
            {row['game_number']} deck", with source_game and
            card_nocab_uuids set.
        Side effects: none.
        Exceptions: none expected.
        """
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=self._source_game,
            name=(
                f"replay_data {row['draft_id']}/{row['match_number']}/"
                f"{row['game_number']} deck"
            ),
            card_nocab_uuids=card_nocab_uuids,
        )

    def _aggression_profile(self, row: dict) -> float:
        """Average len(creatures_attacked) per user half-turn that had
        any attack at all, on this row.

        Private helper - single consumer is accumulate().

        Inputs:
            row: one replay_data CSV row, dict-like.
        Output: the average count of attacking creatures per
            user_turn_N with a non-empty creatures_attacked, across
            every N in self._replay_columns.user_turn_numbers. 0.0 if
            the user never attacked.
        Side effects: none.
        Exceptions: none expected.
        """
        attacker_counts = []
        for turn in self._replay_columns.user_turn_numbers:
            attackers = self._replay_columns.arena_uuids(
                row[ReplayCardColumns.turn_column("user", turn, "creatures_attacked")]
            )
            if attackers:
                attacker_counts.append(len(attackers))

        if not attacker_counts:
            return 0.0
        return sum(attacker_counts) / len(attacker_counts)

    def _output_row(self, row: dict, deck_uuid: UUID, profile: float) -> pa.Table:
        """Build one single-row pa.Table matching _OUTPUT_SCHEMA.

        Private helper - single consumer is accumulate().

        Inputs:
            row: the same row accumulate() received.
            deck_uuid: this row's already-hashed deck identity.
            profile: this row's already-computed
                _aggression_profile(row).
        Output: a one-row pa.Table matching _OUTPUT_SCHEMA:
            draft_id/match_number/game_number read straight off row,
            deck_uuid stringified, profile under
            "combat_aggression_profile".
        Side effects: none.
        Exceptions: none expected.
        """
        return pa.Table.from_pydict(
            {
                "draft_id": [row["draft_id"]],
                "match_number": [row["match_number"]],
                "game_number": [row["game_number"]],
                "deck_uuid": [str(deck_uuid)],
                "combat_aggression_profile": [profile],
            },
            schema=_OUTPUT_SCHEMA,
        )
