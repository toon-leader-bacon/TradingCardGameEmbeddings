"""Translates the Zenodo "Hearthstone Decks Dataset" into the Hearthstone
DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface, and
src/data_retrieval/zenodo_hearthstone_decks/downloader.py for the
dataset, its citation and its provenance.

RAW SHAPE: raw_path is the directory holding the two ";"-separated
files:
    cards.csv: idDeck;idCard, one row per card copy, idCard being the
        card's English name (sometimes with a trailing space). A deck's
        rows are contiguous (checked on the full file: 25,231,435 rows,
        852,028 decks); a deck id seen again after another deck's rows
        raises, since the deck would be split in two.
    decks.csv: idDeck;...;DeckType;DeckArchetype;... (see the
        downloader). Only the archetype is read, for the deck's name.
        12 rows hold extra leading ids and can't be told apart; they
        are skipped and those decks are named by id.
Decks with no cards (in decks.csv only) never reach this stage.

CARD MATCHING (CardNameMatcher): by exact name after stripping
whitespace (measured 2026-10-28: 99.99% of copies match; the misses are
cards since removed from the game, e.g. Mistress of Pain, and tokens or
heroes such as Jaina Proudmoore). A name can name several binder cards,
one per version of a rebalanced card. Per the user, a card is taken as
its latest version:
    1. drop set VANILLA versions: Classic format's 2014 snapshot,
       never a card's current version (unless a name has only VANILLA
       versions, then those are kept);
    2. of the rest, take the one whose provenance dbfId is highest.
       The binder's provenance is a version's lowest dbfId, so this is
       the version printed most recently (CORE_HIDDEN's 2023 Spectral
       Pillager over ICECROWN's 2017 one).
A miss maps to the Unknown sentinel and is counted.

DECK SIZE: any non-empty deck is stored, as in hearthstonetopdecks/.
Arena, Tavern Brawl and Adventure decks are finished decks of their
own modes. The stage logs the size counts.

IDENTITY: deck uuid = deck_uuid_from_cards() over the copy-count
multiset (src/data_refinement/deck_ids.py), stored once with
create_if_absent(), so a list also posted to HearthstoneTopDecks is one
deck. A stored deck's name and provenance come from the first source
that stored it. source_id is idDeck. The deck type, archetype, cost,
date and patch stay in decks.csv for metrics.

RETURN VALUE: only decks this call newly created.

PRECONDITION: card_lookup holds Hearthstone's Unknown sentinel
(CardBinder.ensure_unknown_card()); raises RuntimeError otherwise.
"""

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import ClassVar, Iterator
from uuid import UUID

from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.zenodo_hearthstone_decks.downloader import (
    ZenodoHearthstoneDecksDownloader,
)
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_DELIMITER = ";"
_UNKNOWN_ARCHETYPE = "Unknown"
_CLASSIC_SNAPSHOT_SET = "VANILLA"
_DECKS_COLUMN_COUNT = 9
_ARCHETYPE_COLUMN = 5
_UNMATCHED_LOG_LIMIT = 30


class ZenodoDeckOutcome(str, Enum):
    """What extract() did with one deck (for its summary log)."""

    CREATED = "created"
    ALREADY_STORED = "already_stored"


@dataclass(frozen=True)
class ZenodoDeck:
    """One deck's card rows from cards.csv: its id and card names, one
    per copy, stripped."""

    deck_id: str
    card_names: tuple[str, ...]


class CardNameMatcher:
    """Maps Hearthstone card names to binder cards, each as its latest
    version (see the module docstring), the Unknown sentinel standing in
    for a miss, and counts the misses.

    Matches are cached: 25M copies use ~2,600 distinct names.
    """

    def __init__(self, card_lookup: CardLookup, unknown_uuid: UUID) -> None:
        """
        Inputs: card_lookup (Hearthstone binder), unknown_uuid.
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self.card_lookup = card_lookup
        self.unknown_uuid = unknown_uuid
        self.unmatched_counts: Counter[str] = Counter()
        self._uuid_by_name: dict[str, UUID | None] = {}

    def match_card_nocab_uuids(self, card_names: tuple[str, ...]) -> list[UUID]:
        """One binder uuid per copy, in input order.

        Inputs: card_names (one per copy, stripped).
        Output: uuids (len == len(card_names)).
        Side effects: fills the cache; counts each unmatched copy in
            unmatched_counts.
        Exceptions: none.
        """
        result: list[UUID] = []
        for name in card_names:
            if name not in self._uuid_by_name:
                self._uuid_by_name[name] = self._latest_version_uuid(name)
            nocab_uuid = self._uuid_by_name[name]
            if nocab_uuid is None:
                self.unmatched_counts[name] += 1
                nocab_uuid = self.unknown_uuid
            result.append(nocab_uuid)
        return result

    def _latest_version_uuid(self, name: str) -> UUID | None:
        """The uuid of name's latest version, or None if no card has
        that name.

        Inputs: name. Output: UUID | None. Side effects: none.
        Exceptions: none.
        """
        # Only cards from the build: the Unknown sentinel is named
        # "Unknown" too, and its provenance holds no dbfId
        versions = [
            card
            for card in self.card_lookup.get_by_name(GameId.HEARTHSTONE, name)
            if card.provenance.data_source == DataSource.HEARTHSTONEJSON
        ]
        # set is the version's original printing's set
        # (card_binder/hearthstonejson/ingestion_stage.py)
        current = [
            card
            for card in versions
            if card.raw_content.get("set") != _CLASSIC_SNAPSHOT_SET
        ] or versions
        if not current:
            return None
        return max(current, key=_provenance_dbf_id).nocab_uuid


class ZenodoHearthstoneDecksExtractionStage:
    """Zenodo Hearthstone Decks Dataset -> GenericDecks (Hearthstone).

    Single-consumer to src/data_refinement/deck_box/.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        ZenodoHearthstoneDecksDownloader.DEFAULT_RAW_DATA_DIR
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every distinct deck in raw_path's cards.csv.

        Inputs: raw_path (the directory with cards.csv and decks.csv;
            DEFAULT_RAW_PATH when None), box (written to), card_lookup
            (Hearthstone binder holding the Unknown sentinel).
        Output: nocab_uuid of every deck newly created this call.
        Side effects: reads both files; creates decks on box; shows a
            tqdm bar; logs a summary (outcomes, deck sizes, skipped
            decks.csv rows, unmatched names).
        Exceptions: FileNotFoundError if either file is missing;
            ValueError if a deck's cards.csv rows aren't contiguous;
            RuntimeError if the Unknown sentinel is missing.

        Example:
            >>> ZenodoHearthstoneDecksExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        raw_dir = raw_path or self.DEFAULT_RAW_PATH
        decks_path = raw_dir / ZenodoHearthstoneDecksDownloader.DECKS_FILE.filename
        cards_path = raw_dir / ZenodoHearthstoneDecksDownloader.CARDS_FILE.filename
        for path in (decks_path, cards_path):
            if not path.is_file():
                raise FileNotFoundError(f"no Zenodo Hearthstone file at {path}")
        matcher = CardNameMatcher(
            card_lookup, require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        )
        archetypes = self._read_archetypes(decks_path)
        outcomes: Counter[ZenodoDeckOutcome] = Counter()
        deck_sizes: Counter[int] = Counter()

        # Store each deck once
        for deck in self._read_decks(cards_path):
            deck_sizes[len(deck.card_names)] += 1
            card_nocab_uuids = matcher.match_card_nocab_uuids(deck.card_names)
            name = archetypes.get(deck.deck_id) or deck.deck_id
            created = store_multiset_deck(
                box,
                self.SOURCE_GAME,
                name,
                card_nocab_uuids,
                DataSource.ZENODO_HEARTHSTONE_DECKS,
                deck.deck_id,
            )
            if created is None:
                outcomes[ZenodoDeckOutcome.ALREADY_STORED] += 1
            else:
                outcomes[ZenodoDeckOutcome.CREATED] += 1
                result.append(created)

        self._log_summary(outcomes, deck_sizes, matcher.unmatched_counts)
        return result

    def _read_archetypes(self, decks_path: Path) -> dict[str, str]:
        """Each deck's archetype, when it has a known one.

        Inputs: decks_path.
        Output: {idDeck: archetype}; decks with archetype "Unknown" and
            rows without exactly _DECKS_COLUMN_COUNT fields are left out.
        Side effects: reads decks_path; logs the count of malformed
            rows at INFO.
        Exceptions: OSError if the file can't be read.
        """
        result: dict[str, str] = {}
        malformed_rows = 0
        with open(decks_path, encoding="utf-8", newline="") as decks_file:
            rows = csv.reader(decks_file, delimiter=_DELIMITER)
            next(rows, None)  # header
            for row in rows:
                if len(row) != _DECKS_COLUMN_COUNT:
                    malformed_rows += 1
                    continue
                archetype = row[_ARCHETYPE_COLUMN].strip()
                if archetype and archetype != _UNKNOWN_ARCHETYPE:
                    result[row[0]] = archetype
        _logger.info("Zenodo decks.csv: skipped %d malformed rows", malformed_rows)
        return result

    def _read_decks(self, cards_path: Path) -> Iterator[ZenodoDeck]:
        """Every deck in cards.csv, in file order.

        Inputs: cards_path.
        Output: an iterator of decks.
        Side effects: reads the file lazily; shows a tqdm bar.
        Exceptions: ValueError if a deck id recurs after another deck's
            rows, or a row hasn't two fields.
        """
        seen_deck_ids: set[str] = set()
        with open(cards_path, encoding="utf-8", newline="") as cards_file:
            rows = csv.reader(cards_file, delimiter=_DELIMITER)
            next(rows, None)  # header
            for deck_id, deck_rows in groupby(
                tqdm(rows, desc="Zenodo HS card rows", unit="row"),
                key=_deck_id_of,
            ):
                if deck_id in seen_deck_ids:
                    raise ValueError(
                        f"{cards_path.name}: deck {deck_id}'s rows aren't contiguous"
                    )
                seen_deck_ids.add(deck_id)
                yield ZenodoDeck(
                    deck_id=deck_id,
                    card_names=tuple(row[1].strip() for row in deck_rows),
                )

    def _log_summary(
        self,
        outcomes: Counter[ZenodoDeckOutcome],
        deck_sizes: Counter[int],
        unmatched_names: Counter[str],
    ) -> None:
        """Log outcomes and deck sizes at INFO, and the most common
        unmatched names at WARNING.

        Inputs: outcomes, deck_sizes, unmatched_names. Output: none.
        Side effects: up to two log lines. Exceptions: none.
        """
        _logger.info(
            "Zenodo Hearthstone decks: %s; deck sizes: %s",
            {outcome.value: count for outcome, count in outcomes.items()},
            dict(deck_sizes.most_common()),
        )
        if unmatched_names:
            _logger.warning(
                "Zenodo Hearthstone card names matched to Unknown (copies), most "
                "common first: %s",
                unmatched_names.most_common(_UNMATCHED_LOG_LIMIT),
            )


def _deck_id_of(row: list[str]) -> str:
    """A cards.csv row's idDeck, checking the row has two fields.

    Inputs: row. Output: idDeck. Side effects: none.
    Exceptions: ValueError if the row hasn't exactly two fields.
    """
    if len(row) != 2:
        raise ValueError(f"cards.csv row without two fields: {row!r}")
    return row[0]


def _provenance_dbf_id(card: GenericCard) -> int:
    """The dbfId a Hearthstone binder card's provenance names (its
    version's lowest dbfId).

    Inputs: card. Output: the dbfId. Side effects: none.
    Exceptions: ValueError if source_id isn't an integer.
    """
    return int(card.provenance.source_id)
