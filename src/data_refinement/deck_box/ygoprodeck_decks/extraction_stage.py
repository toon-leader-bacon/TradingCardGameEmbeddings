"""Translates YGOPRODeck's published decks into the Yu-Gi-Oh DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface, and src/data_retrieval/
huggingface_datasets/ygoprodeck_repos.py for the dataset
(feregrino/yugioh-decks, CC-BY-SA-3.0: a published box derived from it
carries the same licence and credits YGOPRODeck).

RAW SHAPE: raw_path is the yugioh-decks mirror directory, complete
(HuggingFaceDatasetDownloader.is_complete_mirror), holding NNNNNNNN.csv
shards of 10,000 deck_num each, read in name order, rows in file order.
Columns read: deck_num, deck_name, main_deck, extra_deck (JSON lists of
passcode strings, one entry per copy). side_deck is dropped, as
hearthstonetopdecks drops sideboards. format, dates, user and
description stay in the CSVs for metrics.

FULL DECKS ONLY: main 40-60 cards and extra 0-15, the game's
construction rules; any other row is skipped and counted (theorycraft
stubs, Speed Duel's 20-30-card decks, empty decks). A row whose
deck_num or a deck column can't be read is skipped as malformed.

CARD MATCHING (PasscodeMatcher): each passcode through the binder's
DataSource.YGOPRODECK alias (card_binder/ygoprodeck/, ids spelled by
passcode_id), cached per passcode; a miss is the Unknown sentinel,
counted per passcode. Measured 2026-11-06 on shard 73: 95.2% of main +
extra copies match.

IDENTITY: deck uuid = deck_uuid_from_cards() over the copy-count
multiset, stored once with store_multiset_deck(), so re-uploads and
"Copy of" decks are one deck and a re-run is a no-op. The stored
deck's name is its deck_name (deck_num when blank); source_id is
deck_num; data source DataSource.YGOPRODECK_DECKS.

RETURN VALUE: only decks this call newly created.

PRECONDITION: card_lookup holds Yu-Gi-Oh's Unknown sentinel
(CardBinder.ensure_unknown_card()); raises RuntimeError otherwise.
"""

import csv
import json
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Iterator, Mapping
from uuid import UUID

from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.card_binder.ygoprodeck.passcode import passcode_id
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.huggingface_datasets.downloader import (
    HuggingFaceDatasetDownloader,
)
from src.data_retrieval.huggingface_datasets.ygoprodeck_repos import (
    YGOPRODECK_DECKS_REPO_ID,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_SHARD_GLOB = "[0-9]" * 8 + ".csv"
# Above csv's default 131,072-character field limit: deck_description is
# user HTML of any length, and one long one would stop the whole read.
# 2**31 - 1, not sys.maxsize, which overflows a C long on Windows.
_CSV_FIELD_SIZE_LIMIT = 2**31 - 1
_MAIN_DECK_SIZES = range(40, 61)
_EXTRA_DECK_SIZES = range(0, 16)
_UNMATCHED_LOG_LIMIT = 30


class YgoprodeckDeckOutcome(str, Enum):
    """What extract() did with one CSV row (for its summary log)."""

    CREATED = "created"
    ALREADY_STORED = "already_stored"
    SKIPPED_MAIN_SIZE = "skipped_main_size"
    SKIPPED_EXTRA_SIZE = "skipped_extra_size"
    SKIPPED_MALFORMED = "skipped_malformed"


@dataclass(frozen=True)
class YgoprodeckDeck:
    """One shard row, parsed: its deck_num, name (deck_num when blank),
    and main and extra decks as alias ids (passcode_id), one per copy."""

    deck_num: str
    name: str
    main_deck: tuple[str, ...]
    extra_deck: tuple[str, ...]


class PasscodeMatcher:
    """Passcode alias id -> binder card uuid, the Unknown sentinel
    standing in for a miss; counts the misses. Cached: ~400k decks use
    ~14k distinct passcodes.

    Single-consumer to YgoprodeckDecksExtractionStage.
    """

    def __init__(self, card_lookup: CardLookup, unknown_uuid: UUID) -> None:
        """
        Inputs: card_lookup (Yu-Gi-Oh binder), unknown_uuid.
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self.card_lookup = card_lookup
        self.unknown_uuid = unknown_uuid
        self.unmatched_counts: Counter[str] = Counter()
        self._uuid_by_passcode: dict[str, UUID | None] = {}

    def match_card_nocab_uuids(self, passcodes: tuple[str, ...]) -> list[UUID]:
        """One binder uuid per copy, in input order.

        Inputs: passcodes (alias ids, one per copy).
        Output: uuids (len == len(passcodes)).
        Side effects: fills the cache; counts each unmatched copy in
            unmatched_counts.
        Exceptions: none.

        Example:
            >>> matcher.match_card_nocab_uuids(("46986414", "46986414"))
            [UUID('...'), UUID('...')]
        """
        result: list[UUID] = []
        for passcode in passcodes:
            if passcode not in self._uuid_by_passcode:
                card = self.card_lookup.get_by_alias(
                    GameId.YUGIOH, DataSource.YGOPRODECK, passcode
                )
                self._uuid_by_passcode[passcode] = card.nocab_uuid if card else None
            nocab_uuid = self._uuid_by_passcode[passcode]
            if nocab_uuid is None:
                self.unmatched_counts[passcode] += 1
                nocab_uuid = self.unknown_uuid
            result.append(nocab_uuid)
        return result


class YgoprodeckDecksExtractionStage:
    """YGOPRODeck deck shards -> GenericDecks (Yu-Gi-Oh).

    Single-consumer to src/data_refinement/deck_box/.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.YUGIOH
    DEFAULT_RAW_PATH: ClassVar[Path] = HuggingFaceDatasetDownloader.raw_data_dir_for(
        YGOPRODECK_DECKS_REPO_ID
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every full, distinct deck in raw_path's shards.

        Inputs: raw_path (the yugioh-decks mirror directory;
            DEFAULT_RAW_PATH when None), box (written to), card_lookup
            (Yu-Gi-Oh binder holding the Unknown sentinel and the
            YGOPRODECK aliases).
        Output: nocab_uuid of every deck newly created this call.
        Side effects: reads every shard; creates decks on box; shows a
            tqdm bar; logs a summary (row outcomes, unmatched
            passcodes).
        Exceptions: FileNotFoundError if raw_path isn't a complete
            mirror or holds no shard; RuntimeError if the Unknown
            sentinel is missing.

        Example:
            >>> YgoprodeckDecksExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        raw_dir = raw_path or self.DEFAULT_RAW_PATH
        shard_paths = self._shard_paths(raw_dir)
        matcher = PasscodeMatcher(
            card_lookup, require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        )
        outcomes: Counter[YgoprodeckDeckOutcome] = Counter()

        # Store each full deck once
        for deck in self._read_decks(shard_paths):
            if deck is None:
                outcomes[YgoprodeckDeckOutcome.SKIPPED_MALFORMED] += 1
                continue
            size_outcome = _size_outcome(deck)
            if size_outcome is not None:
                outcomes[size_outcome] += 1
                continue
            created = store_multiset_deck(
                box,
                self.SOURCE_GAME,
                deck.name,
                matcher.match_card_nocab_uuids(deck.main_deck + deck.extra_deck),
                DataSource.YGOPRODECK_DECKS,
                deck.deck_num,
            )
            if created is None:
                outcomes[YgoprodeckDeckOutcome.ALREADY_STORED] += 1
            else:
                outcomes[YgoprodeckDeckOutcome.CREATED] += 1
                result.append(created)

        self._log_summary(outcomes, matcher.unmatched_counts)
        return result

    def _shard_paths(self, raw_dir: Path) -> list[Path]:
        """The mirror's shards, in name order.

        Inputs: raw_dir. Output: shard paths (non-empty).
        Side effects: lists raw_dir.
        Exceptions: FileNotFoundError if raw_dir isn't a complete
            mirror (HuggingFaceDatasetDownloader.require_complete_mirror)
            or holds no shard.
        """
        HuggingFaceDatasetDownloader.require_complete_mirror(raw_dir)
        result = sorted(raw_dir.glob(_SHARD_GLOB))
        if not result:
            raise FileNotFoundError(f"no YGOPRODeck deck shard in {raw_dir}")
        return result

    def _read_decks(self, shard_paths: list[Path]) -> Iterator[YgoprodeckDeck | None]:
        """Every row of every shard, parsed, in order; None for a row
        that can't be read (extract() counts it).

        Inputs: shard_paths. Output: decks or None, lazily.
        Side effects: reads the shards; shows a tqdm bar over shards;
            raises csv's field size limit while reading (restored after).
        Exceptions: OSError if a shard can't be read; csv.Error if a
            shard isn't CSV.
        """
        with _csv_field_size_limit(_CSV_FIELD_SIZE_LIMIT):
            for shard_path in tqdm(
                shard_paths, desc="YGOPRODeck deck shards", unit="shard"
            ):
                with open(shard_path, encoding="utf-8", newline="") as shard_file:
                    for row in csv.DictReader(shard_file):
                        yield _deck_from_row(row)

    def _log_summary(
        self,
        outcomes: Counter[YgoprodeckDeckOutcome],
        unmatched_passcodes: Counter[str],
    ) -> None:
        """Log row outcomes at INFO and the most common unmatched
        passcodes at WARNING.

        Inputs: outcomes, unmatched_passcodes. Output: none.
        Side effects: log lines. Exceptions: none.
        """
        _logger.info(
            "YGOPRODeck decks: %s",
            {outcome.value: count for outcome, count in outcomes.most_common()},
        )
        if unmatched_passcodes:
            _logger.warning(
                "YGOPRODeck passcodes matched to Unknown (%d copies), most common "
                "first: %s",
                sum(unmatched_passcodes.values()),
                unmatched_passcodes.most_common(_UNMATCHED_LOG_LIMIT),
            )


def _deck_from_row(row: Mapping[str | None, str | None]) -> YgoprodeckDeck | None:
    """Parse one shard row; None if deck_num is blank or missing (a
    short row's cells are None) or main_deck or extra_deck isn't a JSON
    list of passcode strings (an empty cell is an empty list).

    Inputs: row (a csv.DictReader row: a long row's extra cells sit
        under the None key, a short row's missing cells are None).
    Output: YgoprodeckDeck | None.
    Side effects: none. Exceptions: none.
    """
    deck_num = (row.get("deck_num") or "").strip()
    main_deck = _passcodes_from_cell(row.get("main_deck"))
    extra_deck = _passcodes_from_cell(row.get("extra_deck"))
    if not deck_num or main_deck is None or extra_deck is None:
        return None
    return YgoprodeckDeck(
        deck_num=deck_num,
        name=(row.get("deck_name") or "").strip() or deck_num,
        main_deck=main_deck,
        extra_deck=extra_deck,
    )


def _passcodes_from_cell(cell: str | None) -> tuple[str, ...] | None:
    """A deck column's passcodes as alias ids, one per copy; () for an
    empty cell; None if the cell isn't a JSON list of passcodes (strings
    or ints).

    Inputs: cell (None for a short row's missing cell).
    Output: tuple[str, ...] | None.
    Side effects: none. Exceptions: none.
    """
    if cell is None:
        return None
    if not cell.strip():
        return ()
    try:
        values = json.loads(cell)
        if not isinstance(values, list):
            return None
        return tuple(passcode_id(str(value)) for value in values)
    except ValueError:  # json.JSONDecodeError is a ValueError, as is passcode_id's
        return None


@contextmanager
def _csv_field_size_limit(limit: int) -> Iterator[None]:
    """Set csv.field_size_limit(limit) for the block, then restore the
    previous limit (it is process-global).

    Inputs: limit. Output: a context manager.
    Side effects: changes csv's limit for the block's duration.
    Exceptions: OverflowError if limit doesn't fit a C long.
    """
    previous = csv.field_size_limit(limit)
    try:
        yield
    finally:
        csv.field_size_limit(previous)


def _size_outcome(deck: YgoprodeckDeck) -> YgoprodeckDeckOutcome | None:
    """The skip outcome for a deck outside the construction rules
    (_MAIN_DECK_SIZES, _EXTRA_DECK_SIZES), or None for a full deck.

    Inputs: deck. Output: YgoprodeckDeckOutcome | None.
    Side effects: none. Exceptions: none.
    """
    if len(deck.main_deck) not in _MAIN_DECK_SIZES:
        return YgoprodeckDeckOutcome.SKIPPED_MAIN_SIZE
    if len(deck.extra_deck) not in _EXTRA_DECK_SIZES:
        return YgoprodeckDeckOutcome.SKIPPED_EXTRA_SIZE
    return None
