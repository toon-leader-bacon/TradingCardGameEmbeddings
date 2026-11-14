"""The loop every MTG deck-site crawl shares (Archidekt, Moxfield,
Deckstats, MTGTop8): crawl shards -> the MTG DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface. Each site's stage (its own package,
e.g. archidekt_decks/) subclasses CrawledMtgDeckExtractionStage,
setting DATA_SOURCE and DEFAULT_RAW_PATH and handing in its parser.

RAW SHAPE: a crawl's raw dir of JsonlShards (decks-NNNNN.jsonl[.gz]),
read with read_shard_rows() (MTGTop8: a root of one such dir per
format, through the shard_rows hook); one row per deck, in the site's
own schema, which only the site's parser reads.

PER ROW: parser.parse(row) gives a CrawledDeck (only the cards that
count as the deck) or a DeckSkip. A CrawledDeck outside DECK_SIZES is
skipped. The rest are matched (MtgCardMatcher, Unknown sentinel for a
miss) and stored with store_multiset_deck(). Each outcome is counted
for the summary log.

IDENTITY: deck uuid = the copy-count multiset hash, stored once with
create_if_absent(). A deck posted twice, or on two sites, keeps the
name and provenance of the first row seen. A card that matches Unknown
changes the hash, so run this after the MTG binder carries
SCRYFALL_PRINTING aliases, or ids won't reproduce on a later run.

RETURN VALUE: only decks this call newly created; a re-run returns [].

PRECONDITIONS: card_lookup holds MTG's Unknown sentinel (RuntimeError
otherwise); the site's crawl is stopped (see read_shard_rows).
"""

import logging
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Iterator, Protocol
from uuid import UUID

from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.card_matching import card_uuids_per_copy
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.deck_sizes import DeckSizes
from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    DeckSkip,
    DeckSkipReason,
)
from src.data_refinement.deck_box.mtg_crawls.mtg_card_matcher import MtgCardMatcher
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.resumable_crawl.crawl_files import read_shard_rows
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.json_types import JsonObject

_logger = logging.getLogger(__name__)

_UNMATCHED_LOG_LIMIT = 30


class CrawledDeckOutcome(StrEnum):
    """What extract() did with one raw row (for its summary log)."""

    CREATED = "created"
    ALREADY_STORED = "already_stored"
    WRONG_SIZE = "wrong_size"
    SKIPPED_BY_PARSER = "skipped_by_parser"  # see the DeckSkipReason counts


class CrawledDeckParser(Protocol):
    """Strategy: one site's raw row -> a CrawledDeck or a DeckSkip."""

    def parse(self, row: JsonObject) -> CrawledDeck | DeckSkip:
        """Inputs: row (one crawl row). Output: the deck, or why not.
        Side effects: none.
        Exceptions: ValueError naming the deck id on a malformed row."""
        ...


class CrawledMtgDeckExtractionStage:
    """Crawl shards -> GenericDecks (MTG), for one site.

    A site's subclass sets DATA_SOURCE and DEFAULT_RAW_PATH and hands its
    parser (a Strategy) to __init__. The one step it may override is
    shard_rows (Template Method hook), for a raw layout other than one
    shard directory.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    DATA_SOURCE: ClassVar[DataSource]
    DEFAULT_RAW_PATH: ClassVar[Path]
    DECK_SIZES: ClassVar[DeckSizes] = DeckSizes(20, 256)

    def __init__(self, parser: CrawledDeckParser) -> None:
        """
        Inputs: parser (the site's CrawledDeckParser).
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self._parser = parser

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every full, distinct deck in the crawl's shards.

        Inputs: raw_path (the crawl's raw dir; DEFAULT_RAW_PATH when
            None), box (written to), card_lookup (the MTG binder holding
            the Unknown sentinel).
        Output: nocab_uuid of every deck newly created this call.
        Side effects: reads the shards; creates decks on box; shows a
            tqdm bar; logs a summary (outcomes, skip reasons, card match
            counts, most common unmatched cards).
        Exceptions: FileNotFoundError if raw_path is missing;
            json.JSONDecodeError / TypeError on a malformed line;
            ValueError from the parser on a malformed deck;
            RuntimeError if the Unknown sentinel is missing.

        Example:
            >>> MoxfieldDeckExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        shards_dir = raw_path or self.DEFAULT_RAW_PATH
        if not shards_dir.is_dir():
            raise FileNotFoundError(f"no crawl shards at {shards_dir}")
        unknown_uuid = require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        matcher = MtgCardMatcher(card_lookup)
        outcomes: Counter[CrawledDeckOutcome] = Counter()
        skip_reasons: Counter[DeckSkipReason] = Counter()

        # Store each full deck once
        rows = self.shard_rows(shards_dir)
        for row in tqdm(rows, desc=f"{self.DATA_SOURCE.value} decks", unit="deck"):
            parsed = self._parser.parse(row)
            if isinstance(parsed, DeckSkip):
                outcomes[CrawledDeckOutcome.SKIPPED_BY_PARSER] += 1
                skip_reasons[parsed.reason] += 1
                continue
            if not self.DECK_SIZES.fits(parsed.card_count):
                outcomes[CrawledDeckOutcome.WRONG_SIZE] += 1
                continue
            created = store_multiset_deck(
                box,
                self.SOURCE_GAME,
                parsed.name or parsed.source_id,
                card_uuids_per_copy(parsed.entries, matcher, unknown_uuid),
                self.DATA_SOURCE,
                parsed.source_id,
            )
            if created is None:
                outcomes[CrawledDeckOutcome.ALREADY_STORED] += 1
            else:
                outcomes[CrawledDeckOutcome.CREATED] += 1
                result.append(created)

        self._log_summary(outcomes, skip_reasons, matcher)
        return result

    def shard_rows(self, shards_dir: Path) -> Iterator[JsonObject]:
        """Template Method hook: the crawl's raw rows. Default: the one
        JsonlShards directory (read_shard_rows); MTGTop8, which keeps
        one directory per format, overrides it.

        Inputs: shards_dir (an existing directory).
        Output: a lazy iterator of rows.
        Side effects: reads the shard files.
        Exceptions: as read_shard_rows.

        Example:
            >>> next(MoxfieldDeckExtractionStage().shard_rows(Path("data/raw/moxfield")))
            {'publicId': ..., ...}
        """
        return read_shard_rows(shards_dir)

    def _log_summary(
        self,
        outcomes: Counter[CrawledDeckOutcome],
        skip_reasons: Counter[DeckSkipReason],
        matcher: MtgCardMatcher,
    ) -> None:
        """Log outcomes, skip reasons and match counts at INFO, and the
        most common unmatched card names at WARNING (summed over their
        printings, so one missing card is one line).

        Inputs: outcomes, skip_reasons, matcher. Output: none.
        Side effects: up to two log lines. Exceptions: none.
        """
        _logger.info(
            "%s decks: %s; skipped by parser: %s; card entries matched by: %s",
            self.DATA_SOURCE.value,
            {outcome.value: count for outcome, count in outcomes.items()},
            {reason.value: count for reason, count in skip_reasons.items()},
            {kind.value: count for kind, count in matcher.match_counts.items()},
        )
        unmatched_names: Counter[str] = Counter()
        for card, count in matcher.unmatched_counts.items():
            unmatched_names[card.name] += count
        if unmatched_names:
            _logger.warning(
                "%s entries matched to Unknown, most common first: %s",
                self.DATA_SOURCE.value,
                unmatched_names.most_common(_UNMATCHED_LOG_LIMIT),
            )
