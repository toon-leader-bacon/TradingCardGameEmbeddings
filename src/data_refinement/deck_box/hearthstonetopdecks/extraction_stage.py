"""Translates HearthstoneTopDecks deck pages into the Hearthstone
DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface.

RAW SHAPE: data/raw/hearthstonetopdecks/decks.jsonl, written by
src/data_retrieval/hearthstonetopdecks/downloader.py: one line per deck
page, {"url", "title", "deck_code", "details"}. deck_code is the page's
deck code (deckstring.py decodes it) or null. A crash-then-resume can
repeat a line; identity (below) makes that harmless.

CARD MATCHING: a deck code lists dbfIds, and the Hearthstone binder
registers every collectible dbfId of a card (reprints included) as a
DataSource.HEARTHSTONEJSON alias, so each dbfId maps to one binder
card, else to the Unknown sentinel. The binder holds the newest build's
cards, so an old deck's card is taken as its current version (a
rebalanced card is not told apart). Heroes and sideboard cards (see
deckstring.py) are not deck cards and are dropped, so two lists that
differ only in E.T.C.'s band or Zilliax's modules are one deck. The
stage logs the most common unmatched dbfIds at the end of a run.

DECK SIZE: any non-empty deck is stored. Every list here is a finished
deck for its own mode: constructed decks are 30 cards (40 with Prince
Renathal), and old Duels or Tavern Brawl lists can differ. The stage
logs the size counts so odd sizes can be looked at.

IDENTITY: deck uuid = deck_uuid_from_cards() over the full copy-count
multiset (src/data_refinement/deck_ids.py), stored once with
create_if_absent(): many pages post the same list. A stored deck's name
and provenance come from the first page seen with it; source_id is the
page's URL slug. The page's details (format, class, author, rating)
stay in the raw file for metrics.

RETURN VALUE: only decks this call newly created. A re-run over the
same file returns [].

PRECONDITION: card_lookup holds Hearthstone's Unknown sentinel
(CardBinder.ensure_unknown_card()); raises RuntimeError otherwise.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Iterator
from uuid import UUID

from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.hearthstonetopdecks.deckstring import (
    Deckstring,
    decode_deckstring,
)
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.hearthstonetopdecks.downloader import (
    HearthstoneTopDecksDownloader,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_UNMATCHED_LOG_LIMIT = 30


class DeckPageOutcome(str, Enum):
    """What extract() did with one deck page (for its summary log)."""

    CREATED = "created"
    ALREADY_STORED = "already_stored"
    SKIPPED_NO_CODE = "skipped_no_code"
    SKIPPED_BAD_CODE = "skipped_bad_code"
    SKIPPED_EMPTY = "skipped_empty"


@dataclass(frozen=True)
class TopDecksDeckPage:
    """The fields of one decks.jsonl row this stage reads.

    source_id is the URL's slug ("murloc-paladin-77"), stable per page.
    """

    source_id: str
    title: str | None
    deck_code: str | None

    @classmethod
    def from_row(cls, row: dict) -> "TopDecksDeckPage":
        """Parse one decks.jsonl row.

        Inputs: row ({"url", "title", "deck_code", ...}).
        Output: the page. Side effects: none.
        Exceptions: KeyError if "url" is missing.
        """
        return cls(
            source_id=_slug_from_url(row["url"]),
            title=row.get("title") or None,
            deck_code=row.get("deck_code") or None,
        )


class DbfIdCardMatcher:
    """Maps dbfIds to Hearthstone binder cards through their
    HEARTHSTONEJSON alias, the Unknown sentinel standing in for a miss,
    and counts the misses.
    """

    def __init__(self, card_lookup: CardLookup, unknown_uuid: UUID) -> None:
        """
        Inputs: card_lookup (Hearthstone binder), unknown_uuid.
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self.card_lookup = card_lookup
        self.unknown_uuid = unknown_uuid
        self.unmatched_counts: Counter[int] = Counter()

    def match_card_nocab_uuids(self, deckstring: Deckstring) -> list[UUID]:
        """The deck's copy-count multiset as binder uuids.

        Inputs: deckstring.
        Output: one uuid per copy (len == deckstring.card_count).
        Side effects: counts each unmatched copy in unmatched_counts.
        Exceptions: none.
        """
        result: list[UUID] = []
        for dbf_id, copies in deckstring.card_counts.items():
            card = self.card_lookup.get_by_alias(
                GameId.HEARTHSTONE, DataSource.HEARTHSTONEJSON, str(dbf_id)
            )
            if card is None:
                self.unmatched_counts[dbf_id] += copies
                result.extend([self.unknown_uuid] * copies)
            else:
                result.extend([card.nocab_uuid] * copies)
        return result


class HearthstoneTopDecksDeckExtractionStage:
    """HearthstoneTopDecks deck pages -> GenericDecks (Hearthstone).

    Single-consumer to src/data_refinement/deck_box/.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        HearthstoneTopDecksDownloader.DEFAULT_RAW_DATA_DIR
        / HearthstoneTopDecksDownloader.DECKS_FILENAME
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every distinct, decodable deck in raw_path.

        Inputs: raw_path (decks.jsonl; DEFAULT_RAW_PATH when None), box
            (written to), card_lookup (Hearthstone binder holding the
            Unknown sentinel).
        Output: nocab_uuid of every deck newly created this call.
        Side effects: reads raw_path; creates decks on box; shows a tqdm
            bar; logs a summary (page outcomes, deck sizes, unmatched
            dbfIds).
        Exceptions: FileNotFoundError if raw_path is missing;
            json.JSONDecodeError / KeyError on a malformed line;
            RuntimeError if the Unknown sentinel is missing.

        Example:
            >>> HearthstoneTopDecksDeckExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        decks_path = raw_path or self.DEFAULT_RAW_PATH
        if not decks_path.is_file():
            raise FileNotFoundError(f"no HearthstoneTopDecks decks at {decks_path}")
        matcher = DbfIdCardMatcher(
            card_lookup, require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        )
        outcomes: Counter[DeckPageOutcome] = Counter()
        deck_sizes: Counter[int] = Counter()

        # Store each decodable, non-empty deck once
        for page in self._read_pages(decks_path):
            if page.deck_code is None:
                outcomes[DeckPageOutcome.SKIPPED_NO_CODE] += 1
                continue
            deckstring = self._decode_or_none(page.source_id, page.deck_code)
            if deckstring is None:
                outcomes[DeckPageOutcome.SKIPPED_BAD_CODE] += 1
                continue
            if deckstring.card_count == 0:
                outcomes[DeckPageOutcome.SKIPPED_EMPTY] += 1
                continue
            deck_sizes[deckstring.card_count] += 1
            card_nocab_uuids = matcher.match_card_nocab_uuids(deckstring)
            created = store_multiset_deck(
                box,
                self.SOURCE_GAME,
                page.title or page.source_id,
                card_nocab_uuids,
                DataSource.HEARTHSTONETOPDECKS,
                page.source_id,
            )
            if created is None:
                outcomes[DeckPageOutcome.ALREADY_STORED] += 1
            else:
                outcomes[DeckPageOutcome.CREATED] += 1
                result.append(created)

        self._log_summary(outcomes, deck_sizes, matcher.unmatched_counts)
        return result

    def _read_pages(self, decks_path: Path) -> Iterator[TopDecksDeckPage]:
        """Every row of decks.jsonl, in file order.

        Inputs: decks_path. Output: an iterator of pages.
        Side effects: reads the file lazily; shows a tqdm bar.
        Exceptions: json.JSONDecodeError / KeyError on a malformed line.
        """
        with open(decks_path, encoding="utf-8") as decks_file:
            for line in tqdm(decks_file, desc="HSTD deck pages", unit="deck"):
                if line.strip():
                    yield TopDecksDeckPage.from_row(json.loads(line))

    def _decode_or_none(self, source_id: str, deck_code: str) -> Deckstring | None:
        """deck_code decoded, or None (logged at DEBUG) if it doesn't
        decode.

        Inputs: source_id (for the log line), deck_code.
        Output: Deckstring | None.
        Side effects: a debug log line on failure. Exceptions: none.
        """
        try:
            return decode_deckstring(deck_code)
        except ValueError as error:
            _logger.debug("HSTD deck %s: bad deck code: %s", source_id, error)
            return None

    def _log_summary(
        self,
        outcomes: Counter[DeckPageOutcome],
        deck_sizes: Counter[int],
        unmatched_dbf_ids: Counter[int],
    ) -> None:
        """Log page outcomes and deck sizes at INFO, and the most common
        unmatched dbfIds at WARNING.

        Inputs: outcomes, deck_sizes, unmatched_dbf_ids. Output: none.
        Side effects: up to two log lines. Exceptions: none.
        """
        _logger.info(
            "HearthstoneTopDecks pages: %s; deck sizes: %s",
            {outcome.value: count for outcome, count in outcomes.items()},
            dict(deck_sizes.most_common()),
        )
        if unmatched_dbf_ids:
            _logger.warning(
                "HearthstoneTopDecks dbfIds matched to Unknown (copies), most "
                "common first: %s",
                unmatched_dbf_ids.most_common(_UNMATCHED_LOG_LIMIT),
            )


def _slug_from_url(url: str) -> str:
    """The last path segment of a deck page URL
    (".../decks/murloc-paladin-77/" -> "murloc-paladin-77").

    Inputs: url. Output: the slug. Side effects: none.
    Exceptions: none.
    """
    return url.rstrip("/").rsplit("/", 1)[-1]
