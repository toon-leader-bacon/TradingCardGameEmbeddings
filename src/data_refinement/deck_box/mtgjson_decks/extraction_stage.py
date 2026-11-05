"""Translates MTGJSON's AllDeckFiles (Wizards-printed precons) into the
MTG DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface, and
src/data_retrieval/mtgjson_decks/downloader.py for the source.

RAW SHAPE: raw_path is a build's decks/ directory, one JSON file per
deck, {"meta": ..., "data": {"name", "code", "type", "releaseDate",
"mainBoard", "commander", "sideBoard", "displayCommander", "tokens",
"planes", "schemes", ...}}. Each board is a list of full MTGJSON card
objects carrying "name", "count" and "identifiers"."scryfallOracleId".
A board can list one card twice (a foil and a non-foil row: 83 of 3,135
decks on 2026-11-05); the copies add up.

WHICH DECKS: a deck is stored unless its "type" is in _NOT_DECK_TYPES
(Secret Lair drops, MTGO redemption entries and bundle land packs are
products, not decks) or it has no cards. 2,044 decks remain on the
2026-11-05 build. The type filter is a denylist, so the summary log
counts decks per type: a new MTGJSON product type shows up there.

WHICH CARDS: "mainBoard" plus "commander". The sideboard, tokens,
planes, schemes and displayCommander (a display duplicate) are not part
of the played deck and are skipped.

CARD MATCHING: by Scryfall oracle id, the MTG binder's primary alias
(card_lookup.get_by_alias(MTG, SCRYFALL, oracle_id)), so no name
matching. A card with no oracle id or no binder match maps to the
Unknown sentinel; the most common misses are logged by name.

IDENTITY: deck uuid = deck_uuid_from_cards() over the copy-count
multiset, stored once via store_multiset_deck(), so a reprinted precon
or a list 17lands also holds is one deck. source_id is the file's stem
(e.g. "BoggartFeast_LRW"). Type, set code and release date stay in the
raw files for metrics.

DEFAULT RAW PATH: MTGJSON lands one dated build directory per day, so
the default is chosen at call time (the downloader's latest complete
build), not fixed as a ClassVar like the siblings' DEFAULT_RAW_PATH.
Multiset identity keeps a re-run over a newer build idempotent.

RETURN VALUE: only decks this call newly created.

PRECONDITION: card_lookup holds MTG's Unknown sentinel
(CardBinder.ensure_unknown_card()); raises RuntimeError otherwise,
before any file is read.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.mtgjson_decks.downloader import MtgjsonDeckFilesDownloader
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

# Products MTGJSON lists as decks that are not played decks
_NOT_DECK_TYPES = frozenset({"Secret Lair Drop", "MTGO Redemption", "Bundle Land Pack"})
# The boards that make up the played deck
_PLAYED_BOARDS = ("mainBoard", "commander")
_UNMATCHED_LOG_LIMIT = 30


@dataclass(frozen=True)
class MtgjsonCardRow:
    """One board row of a deck file: a card and its copy count."""

    name: str
    oracle_id: str | None  # identifiers.scryfallOracleId; absent on some rows
    count: int


@dataclass(frozen=True)
class MtgjsonDeck:
    """One deck file, parsed: what this stage reads and nothing more."""

    source_id: str  # the file's stem
    name: str
    deck_type: str
    rows: tuple[MtgjsonCardRow, ...]  # over _PLAYED_BOARDS, in file order


@dataclass
class _ExtractionTally:
    """Counts one extract() call logs at its end."""

    decks_by_type: Counter[str] = field(default_factory=Counter)
    skipped_by_type: int = 0
    empty: int = 0
    already_stored: int = 0
    created: int = 0


@dataclass
class _OracleIdMatcher:
    """Oracle id -> binder uuid, with the Unknown sentinel for a miss and
    a count of misses by card name for the summary log."""

    card_lookup: CardLookup
    unknown_uuid: UUID
    copies: int = 0
    unmatched_names: Counter[str] = field(default_factory=Counter)

    def card_uuids_of(self, deck: MtgjsonDeck) -> list[UUID]:
        """One nocab_uuid per copy of every row of deck.

        Inputs: deck. Output: list of UUIDs (unknown_uuid for a miss).
        Side effects: counts copies and misses.
        Exceptions: none.
        """
        raise NotImplementedError

    def _uuid_of(self, row: MtgjsonCardRow) -> UUID | None:
        """The binder card for row's oracle id, or None (no oracle id, or
        no binder alias).

        Inputs: row. Output: UUID or None. Side effects: none.
        Exceptions: none.
        """
        raise NotImplementedError


class MtgjsonDeckExtractionStage:
    """MTGJSON precon deck files -> GenericDecks (MTG)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG

    @staticmethod
    def default_raw_path() -> Path:
        """The latest complete MTGJSON build's decks/ directory.

        Inputs: none. Output: Path.
        Side effects: lists the downloader's raw directory.
        Exceptions: FileNotFoundError if no complete build is landed.
        """
        return MtgjsonDeckFilesDownloader.latest_build_decks_dir()

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store one deck per playable precon file under raw_path.

        Inputs: raw_path (a build's decks/ directory; default_raw_path()
            when None), box (written to), card_lookup (MTG binder with
            the Unknown sentinel).
        Output: uuid of every deck this call newly created.
        Side effects: reads every *.json under raw_path (sorted by name);
            creates decks on box; logs the tally and the top misses.
        Exceptions: RuntimeError if the Unknown sentinel is missing;
            FileNotFoundError if raw_path is not a directory; as
            _read_deck for a malformed file.

        Example:
            >>> MtgjsonDeckExtractionStage().extract(None, box, binder)
            [UUID('...'), ...]
        """
        result: list[UUID] = []
        tally = _ExtractionTally()

        # Validate inputs
        matcher = _OracleIdMatcher(
            card_lookup, require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        )
        directory = self.default_raw_path() if raw_path is None else raw_path
        if not directory.is_dir():
            raise FileNotFoundError(f"no MTGJSON decks directory at {directory}")

        # One file per deck
        for path in sorted(directory.glob("*.json")):
            deck = _read_deck(path)
            tally.decks_by_type[deck.deck_type] += 1
            if deck.deck_type in _NOT_DECK_TYPES:
                tally.skipped_by_type += 1
                continue
            card_nocab_uuids = matcher.card_uuids_of(deck)
            if not card_nocab_uuids:
                tally.empty += 1
                continue
            created = store_multiset_deck(
                box,
                self.SOURCE_GAME,
                deck.name,
                card_nocab_uuids,
                DataSource.MTGJSON_DECKS,
                deck.source_id,
            )
            if created is None:
                tally.already_stored += 1
                continue
            tally.created += 1
            result.append(created)

        _log_summary(tally, matcher)
        return result


def _read_deck(path: Path) -> MtgjsonDeck:
    """Parse one deck file: its name, type and _PLAYED_BOARDS rows.

    Inputs: path. Output: MtgjsonDeck (source_id = path's stem).
    Side effects: reads path.
    Exceptions: json.JSONDecodeError; KeyError if "data", its "name" or
        "type", or a row's "name" or "count" is absent; ValueError for a
        count that is not a positive int.
    """
    raise NotImplementedError


def _log_summary(tally: _ExtractionTally, matcher: _OracleIdMatcher) -> None:
    """Log one summary line for an extract() call, the decks per type,
    and (at WARNING) the _UNMATCHED_LOG_LIMIT most common unmatched names.

    Inputs: tally, matcher. Output: none. Side effects: logs.
    Exceptions: none.
    """
    raise NotImplementedError
