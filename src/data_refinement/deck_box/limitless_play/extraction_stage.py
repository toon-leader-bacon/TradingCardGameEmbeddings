"""Translates Limitless tournament decklists into DeckBoxes.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface.

LimitlessDeckExtractionStage is the loop every Limitless game shares;
a subclass supplies the game, raw path, deck sizes and a card matcher.
LimitlessPlayDeckExtractionStage (below) is Pokemon's;
tcg_extraction_stages.py holds the other games'.

RAW SHAPE: data/raw/limitless_play/<game>/standings.jsonl, written by
src/data_retrieval/limitless_play/downloader.py: one line per
tournament, {"tournament_id", "standings": [row, ...]}; each row's
decklist is parsed by limitless_decklist.py (which drops side decks).
A crash-then-resume can repeat a tournament's line; identity (below)
makes that harmless.

CARD MATCHING: the subclass's CachedCardMatcher (deck_box/card_matching.py).
The stage logs how many entries each match step took at the end of a
run.

FULL DECKS ONLY: a decklist whose copies don't fit the subclass's
DECK_SIZES is skipped and counted, never stored.

IDENTITY: deck uuid = deck_uuid_from_cards() over the full copy-count
multiset (src/data_refinement/deck_ids.py), stored once with
create_if_absent(): many players submit identical lists. A stored
deck's name and provenance come from the first decklist seen with it.
How often each list was played, its placings and records are metrics'
job (keyed by this uuid), not box fields.

RETURN VALUE: only decks this call newly created, unlike
seventeenlands_game_data's stage (which also lists decks it saw that
were already stored). A re-run over the same file returns [].

PRECONDITION: card_lookup holds the game's Unknown sentinel
(CardBinder.ensure_unknown_card()); raises RuntimeError otherwise.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import ClassVar, Iterator
from uuid import UUID

from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.limitless_play.card_index import (
    LimitlessCardIndex,
)
from src.data_refinement.deck_box.card_matching import (
    CachedCardMatcher,
    card_uuids_per_copy,
)
from src.data_refinement.deck_box.deck_sizes import DeckSizes
from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardRef,
    LimitlessDecklist,
    parse_standings_row,
)
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.limitless_play.downloader import (
    LimitlessGame,
    LimitlessPlayDownloader,
)
from src.data_retrieval.pokemon_tcg.downloader import PokemonTcgDataDownloader
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_UNMATCHED_LOG_LIMIT = 30


class DecklistOutcome(str, Enum):
    """What extract() did with one decklist (for its summary log)."""

    CREATED = "created"
    ALREADY_STORED = "already_stored"
    SKIPPED_WRONG_SIZE = "skipped_wrong_size"


class LimitlessDeckExtractionStage(ABC):
    """Template Method: Limitless decklists -> GenericDecks for one game.

    A subclass sets SOURCE_GAME, DEFAULT_RAW_PATH and DECK_SIZES and
    builds the card matcher in _build_card_matcher.
    """

    SOURCE_GAME: ClassVar[GameId]
    DEFAULT_RAW_PATH: ClassVar[Path]
    DECK_SIZES: ClassVar[DeckSizes]

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every full, distinct decklist in raw_path.

        Inputs: raw_path (standings.jsonl; DEFAULT_RAW_PATH when None),
            box (written to), card_lookup (SOURCE_GAME's binder holding
            the Unknown sentinel).
        Output: nocab_uuid of every deck newly created this call.
        Side effects: reads raw_path (and whatever _build_card_matcher
            reads); creates decks on box; shows a tqdm bar; logs a
            summary (decklist outcomes, card match counts).
        Exceptions: FileNotFoundError if raw_path is missing;
            json.JSONDecodeError / KeyError / TypeError / ValueError on
            a malformed line or decklist entry; RuntimeError if the
            Unknown sentinel is missing; whatever _build_card_matcher
            raises.

        Example:
            >>> LimitlessPlayDeckExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        standings_path = raw_path or self.DEFAULT_RAW_PATH
        if not standings_path.is_file():
            raise FileNotFoundError(f"no Limitless standings at {standings_path}")
        unknown_uuid = require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        card_matcher = self._build_card_matcher(card_lookup)
        outcomes: Counter[DecklistOutcome] = Counter()

        # Store each full decklist once
        for decklist in self._read_decklists(standings_path):
            if not self.DECK_SIZES.fits(decklist.card_count):
                outcomes[DecklistOutcome.SKIPPED_WRONG_SIZE] += 1
                continue
            card_nocab_uuids = card_uuids_per_copy(
                decklist.entries, card_matcher, unknown_uuid
            )
            created = store_multiset_deck(
                box,
                self.SOURCE_GAME,
                decklist.archetype_name or decklist.source_id,
                card_nocab_uuids,
                DataSource.LIMITLESS_PLAY,
                decklist.source_id,
            )
            if created is None:
                outcomes[DecklistOutcome.ALREADY_STORED] += 1
            else:
                outcomes[DecklistOutcome.CREATED] += 1
                result.append(created)

        self._log_summary(outcomes, card_matcher)
        return result

    @abstractmethod
    def _build_card_matcher(
        self, card_lookup: CardLookup
    ) -> CachedCardMatcher[LimitlessCardRef]:
        """The matcher for this game's entries.

        Inputs: card_lookup (SOURCE_GAME's binder, read-only).
        Output: a CachedCardMatcher[LimitlessCardRef].
        Side effects: as the subclass documents (e.g. reads a file).
        Exceptions: as the subclass documents.
        """

    def _read_decklists(self, standings_path: Path) -> Iterator[LimitlessDecklist]:
        """Every decklist in standings.jsonl, in file order.

        Inputs: standings_path. Output: an iterator of decklists.
        Side effects: reads the file lazily; shows a tqdm bar over its
            lines (tournaments).
        Exceptions: json.JSONDecodeError / KeyError / TypeError /
            ValueError on a malformed line or decklist entry.
        """
        with open(standings_path, encoding="utf-8") as standings_file:
            for line in tqdm(
                standings_file,
                desc=f"Limitless {self.SOURCE_GAME.value} tournaments",
                unit="t",
            ):
                if not line.strip():
                    continue
                tournament = json.loads(line)
                tournament_id = tournament["tournament_id"]
                for row_index, row in enumerate(tournament["standings"]):
                    yield from parse_standings_row(tournament_id, row_index, row)

    def _log_summary(
        self,
        outcomes: Counter[DecklistOutcome],
        card_matcher: CachedCardMatcher[LimitlessCardRef],
    ) -> None:
        """Log decklist outcomes and card match counts at INFO, and the
        most common unmatched entries at WARNING.

        Inputs: outcomes, card_matcher. Output: none.
        Side effects: up to two log lines. Exceptions: none.
        """
        _logger.info(
            "Limitless %s decklists: %s; card entries matched by: %s",
            self.SOURCE_GAME.value,
            {outcome.value: count for outcome, count in outcomes.items()},
            {kind.value: count for kind, count in card_matcher.match_counts.items()},
        )
        if card_matcher.unmatched_counts:
            _logger.warning(
                "Limitless %s entries matched to Unknown, most common first: %s",
                self.SOURCE_GAME.value,
                card_matcher.unmatched_counts.most_common(_UNMATCHED_LOG_LIMIT),
            )


class LimitlessPlayDeckExtractionStage(LimitlessDeckExtractionStage):
    """Limitless play tournament decklists -> GenericDecks (Pokemon).

    Every Limitless PTCG format, retro ones included, plays 60 cards.
    Cards are matched by card_index.py, which needs pokemon-tcg-data's
    sets/en.json (extracted by src/data_retrieval/pokemon_tcg/).
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        LimitlessPlayDownloader.raw_data_dir_for(LimitlessGame.POKEMON_TCG)
        / LimitlessPlayDownloader.STANDINGS_FILENAME
    )
    DECK_SIZES: ClassVar[DeckSizes] = DeckSizes(60, 60)
    DEFAULT_SETS_PATH: ClassVar[Path] = (
        PokemonTcgDataDownloader.DEFAULT_RAW_DATA_DIR / "sets" / "en.json"
    )

    def __init__(self, sets_path: Path | None = None) -> None:
        """
        Inputs: sets_path (pokemon-tcg-data's sets/en.json;
            DEFAULT_SETS_PATH when None).
        Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        self.sets_path = sets_path or self.DEFAULT_SETS_PATH

    def _build_card_matcher(
        self, card_lookup: CardLookup
    ) -> CachedCardMatcher[LimitlessCardRef]:
        """LimitlessCardIndex over self.sets_path.

        Inputs: card_lookup. Output: LimitlessCardIndex.
        Side effects: reads self.sets_path.
        Exceptions: FileNotFoundError, json.JSONDecodeError, KeyError /
            ValueError on a malformed set row.
        """
        return LimitlessCardIndex.from_sets_file(self.sets_path, card_lookup)
