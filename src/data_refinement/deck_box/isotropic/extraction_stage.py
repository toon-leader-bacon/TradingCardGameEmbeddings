"""Translates isotropic's Dominion game summaries into final decks.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface.

RAW SHAPE: data/raw/isotropic/*-summary.tar.bz2, each holding one
games-YYYYMMDD.json JSONL member per day, one game per line (see
metrics/isotropic/BRAINSTORM.md). Each players[] entry that reached the
end of the game carries end.deck: {card name: copies}. Resigned players
have no "end" block, so no final deck.

WHAT IS KEPT: every non-resigned player's complete final deck. A deck
with any card name the binder does not know is dropped whole (logged),
never stored short, since the box holds only full decks. The metrics'
private box (data/metrics/isotropic/deck_box.db) also holds partial
decks and kingdoms; this stage writes final decks only.

IDENTITY: content-addressed (deck_ids.deck_uuid_from_cards, via
row_utils.deck_for_player), so a re-run is a no-op, identical decks from
different games are stored once, and a final deck here has the same
uuid as in the metrics' box.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Iterator
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    deck_for_player,
    eligible_player_entries,
)
from src.data_refinement.metrics.isotropic.summary.scanner import iter_summary_rows
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_SUMMARY_ARCHIVE_GLOB = "*-summary.tar.bz2"


class IsotropicDeckExtractionStage:
    """isotropic game summaries -> final GenericDecks (Dominion)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.DOMINION
    DEFAULT_RAW_PATH: ClassVar[Path] = Path("data/raw/isotropic")

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every non-resigned player's final deck from the summary
        archives at raw_path.

        Inputs: raw_path (one *-summary.tar.bz2, or a directory of them;
            DEFAULT_RAW_PATH when None), box (written to), card_lookup
            (the Dominion binder).
        Output: nocab_uuid of every deck this call newly created; a deck
            already on box (same card multiset) is not listed.
        Side effects: reads the archives; creates decks on box; logs how
            many decks were dropped for unknown card names.
        Exceptions: FileNotFoundError if raw_path does not exist or holds
            no summary archive; KeyError if a game row lacks "players"
            or a player's "end" block lacks "deck"; whatever tarfile or
            json raise on a malformed archive.

        Example:
            >>> IsotropicDeckExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        dropped = 0

        # One game per row; one final deck per player who did not resign
        for archive_path in self._archive_paths(raw_path or self.DEFAULT_RAW_PATH):
            for row in iter_summary_rows(archive_path):
                for player in eligible_player_entries(row):
                    deck = self._full_deck(player, card_lookup)
                    if deck is None:
                        dropped += 1
                        continue
                    if box.get_by_uuid(deck.nocab_uuid) is None:
                        box.create(deck)
                        result.append(deck.nocab_uuid)

        if dropped:
            _logger.warning(
                "IsotropicDeckExtractionStage: dropped %d decks with unknown "
                "card names",
                dropped,
            )
        return result

    @staticmethod
    def _archive_paths(raw_path: Path) -> Iterator[Path]:
        """raw_path itself if it is a file, else its summary archives in
        name order.

        Inputs: raw_path. Output: iterator of archive paths.
        Side effects: lists the directory.
        Exceptions: FileNotFoundError if raw_path is missing or a
            directory with no summary archive.
        """
        if raw_path.is_file():
            yield raw_path
            return
        archives = sorted(raw_path.glob(_SUMMARY_ARCHIVE_GLOB))
        if not archives:
            raise FileNotFoundError(
                f"no {_SUMMARY_ARCHIVE_GLOB} archives at {raw_path}; run "
                "scripts/run_data_retrieval.py --source isotropic"
            )
        yield from archives

    def _full_deck(self, player: dict, card_lookup: CardLookup) -> GenericDeck | None:
        """player's final deck with provenance, or None if any card name
        did not map to a card (deck_for_player drops those copies).

        Inputs: player (a players[] entry with an "end" block),
            card_lookup.
        Output: GenericDeck or None. Side effects: none beyond
            deck_for_player's per-name error log.
        Exceptions: KeyError if player["end"] lacks "deck".
        """
        deck = deck_for_player(card_lookup, player)
        if len(deck.card_nocab_uuids) != sum(player["end"]["deck"].values()):
            return None
        result = GenericDeck(
            nocab_uuid=deck.nocab_uuid,
            source_game=self.SOURCE_GAME,
            name=deck.name,
            card_nocab_uuids=deck.card_nocab_uuids,
            provenance=Provenance(
                data_source=DataSource.ISOTROPIC,
                source_id=str(deck.nocab_uuid),
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        return result
