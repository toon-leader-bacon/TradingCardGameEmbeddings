"""Translates Limitless tournament decklists into the Pokemon DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface.

RAW SHAPE: data/raw/limitless_play/standings.jsonl, written by
src/data_retrieval/limitless_play/downloader.py: one line per
tournament, {"tournament_id", "standings": [row, ...]}; each row's
decklist is parsed by limitless_decklist.py. A crash-then-resume can
repeat a tournament's line; identity (below) makes that harmless.

CARD MATCHING: card_index.py (printing, then name, then the Unknown
sentinel). It needs pokemon-tcg-data's sets/en.json (extracted by
src/data_retrieval/pokemon_tcg/downloader.py). The stage logs how many
entries each match step took at the end of a run.

FULL DECKS ONLY: a decklist whose copies don't total 60 (every
Limitless PTCG format, retro ones included, plays 60) is skipped and
counted, never stored.

IDENTITY: deck uuid = deck_uuid_from_cards() over the full copy-count
multiset (src/data_refinement/deck_ids.py), stored once with
create_if_absent(): many players submit identical lists. A stored
deck's name and provenance come from the first decklist seen with it.
How often each list was played, its placings and records are metrics'
job (keyed by this uuid), not box fields.

RETURN VALUE: only decks this call newly created, unlike
seventeenlands_game_data's stage (which also lists decks it saw that
were already stored). A re-run over the same file returns [].

PRECONDITION: card_lookup holds Pokemon's Unknown sentinel
(CardBinder.ensure_unknown_card()); raises RuntimeError otherwise.
"""

import json
import logging
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
from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessDecklist,
    parse_standings_row,
)
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.limitless_play.downloader import LimitlessPlayDownloader
from src.data_retrieval.pokemon_tcg.downloader import PokemonTcgDataDownloader
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_DECK_SIZE = 60
_UNMATCHED_LOG_LIMIT = 30


class DecklistOutcome(str, Enum):
    """What extract() did with one decklist (for its summary log)."""

    CREATED = "created"
    ALREADY_STORED = "already_stored"
    SKIPPED_NOT_60 = "skipped_not_60"


class LimitlessPlayDeckExtractionStage:
    """Limitless play tournament decklists -> GenericDecks (Pokemon).

    Single-consumer to src/data_refinement/deck_box/.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        LimitlessPlayDownloader.DEFAULT_RAW_DATA_DIR
        / LimitlessPlayDownloader.STANDINGS_FILENAME
    )
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

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every full, distinct decklist in raw_path.

        Inputs: raw_path (standings.jsonl; DEFAULT_RAW_PATH when None),
            box (written to), card_lookup (Pokemon binder holding the
            Unknown sentinel).
        Output: nocab_uuid of every deck newly created this call.
        Side effects: reads raw_path and self.sets_path; creates decks
            on box; shows a tqdm bar; logs a summary (decklist outcomes,
            card match counts).
        Exceptions: FileNotFoundError if raw_path or sets_path is
            missing; json.JSONDecodeError / KeyError / TypeError /
            ValueError on a malformed line or decklist entry;
            RuntimeError if the Unknown sentinel is missing.

        Example:
            >>> LimitlessPlayDeckExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        standings_path = raw_path or self.DEFAULT_RAW_PATH
        if not standings_path.is_file():
            raise FileNotFoundError(f"no Limitless standings at {standings_path}")
        unknown_uuid = require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        card_index = LimitlessCardIndex.from_sets_file(self.sets_path, card_lookup)
        outcomes: Counter[DecklistOutcome] = Counter()

        # Store each full decklist once
        for decklist in self._read_decklists(standings_path):
            if decklist.card_count != _DECK_SIZE:
                outcomes[DecklistOutcome.SKIPPED_NOT_60] += 1
                continue
            card_nocab_uuids = self._match_card_nocab_uuids(
                decklist, card_index, unknown_uuid
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

        self._log_summary(outcomes, card_index)
        return result

    def _read_decklists(self, standings_path: Path) -> Iterator[LimitlessDecklist]:
        """Every decklist in standings.jsonl, in file order.

        Inputs: standings_path. Output: an iterator of decklists.
        Side effects: reads the file lazily; shows a tqdm bar over its
            lines (tournaments).
        Exceptions: json.JSONDecodeError / KeyError / TypeError /
            ValueError on a malformed line or decklist entry.
        """
        with open(standings_path, encoding="utf-8") as standings_file:
            for line in tqdm(standings_file, desc="Limitless tournaments", unit="t"):
                if not line.strip():
                    continue
                tournament = json.loads(line)
                tournament_id = tournament["tournament_id"]
                for row_index, row in enumerate(tournament["standings"]):
                    yield from parse_standings_row(tournament_id, row_index, row)

    def _match_card_nocab_uuids(
        self,
        decklist: LimitlessDecklist,
        card_index: LimitlessCardIndex,
        unknown_uuid: UUID,
    ) -> list[UUID]:
        """The decklist's copy-count multiset as binder uuids, the
        Unknown sentinel standing in for an unmatched entry.

        Inputs: decklist, card_index, unknown_uuid.
        Output: one uuid per copy (len == decklist.card_count).
        Side effects: as card_index.match_card_uuid.
        Exceptions: none.
        """
        result: list[UUID] = []
        for entry in decklist.entries:
            card_uuid = card_index.match_card_uuid(entry)
            result.extend([card_uuid or unknown_uuid] * entry.count)
        return result

    def _log_summary(
        self, outcomes: Counter[DecklistOutcome], card_index: LimitlessCardIndex
    ) -> None:
        """Log decklist outcomes and card match counts at INFO, and the
        most common unmatched entries at WARNING.

        Inputs: outcomes, card_index. Output: none.
        Side effects: up to two log lines. Exceptions: none.
        """
        _logger.info(
            "Limitless decklists: %s; card entries matched by: %s",
            {outcome.value: count for outcome, count in outcomes.items()},
            {kind.value: count for kind, count in card_index.match_counts.items()},
        )
        if card_index.unmatched_counts:
            _logger.warning(
                "Limitless entries matched to Unknown, most common first: %s",
                card_index.unmatched_counts.most_common(_UNMATCHED_LOG_LIMIT),
            )
