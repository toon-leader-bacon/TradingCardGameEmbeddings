"""Translates 17Lands' game_data CSVs into full constructed decks, directly into a DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface this implements.

RAW SHAPE: data/raw/17lands/game_data/<Expansion>.<FormatCode>.csv. Each
row is one game. The header carries one "deck_<CardName>" column per
card legal in that set and format; a row's value under it is that card's
copy count in the deck played that game (a row's deck_ counts sum to
the deck size, usually 40).

ONE PARSER FOR DECKS AND METRICS: rows are read with the same
GameDataChunkParser (src/data_refinement/seventeenlands/game_data/) the
game_data metrics use, and each row's deck comes from its ChunkDecks
(src/data_refinement/seventeenlands/chunk_decks.py). So a deck_uuid a
metric writes is, by construction, a deck this stage stores: same card
matching, same Unknown-sentinel substitution for an unmatched deck_
column (at that column's copy count), same multiset hash
(src/data_refinement/deck_ids.py's deck_uuid_from_cards()).

EVERY DISTINCT DECKLIST PLAYED: a draft plays 3-7 games and may change
its deck between them (sideboarding, rebuilds); every game's decklist is
a full deck that was played, so each distinct one is stored. Identity is
the content hash, so a decklist shared by several games or drafts is
stored once (create_if_absent()); how many drafts played each one is
the DeckOccurrenceCountMetric's job, not the box's. A stored deck's name
and provenance are those of the first game seen with it.

UNMATCHED CARDS: an unmatched deck_ column counts as that many copies
of GameId.MTG's Unknown sentinel card (CardBinder.unknown_card_uuid()).
PRECONDITION: card_lookup must already hold that sentinel
(CardBinder.ensure_unknown_card()); this stage stays read-only and raises
RuntimeError if it is missing. Unmatched names are logged once per CSV.

FRESH BOX REQUIRED AFTER AN IDENTITY CHANGE: an mtg.db written under an
earlier identity scheme (per-game or per-draft uuid5) must be deleted
before re-running; none of those ids match deck_uuid_from_cards(), so a
re-run would add content-addressed decks next to the stale ones.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import pyarrow.csv as pa_csv
from tqdm import tqdm

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.seventeenlands.chunk_decks import ChunkDecks, GameKeys
from src.data_refinement.seventeenlands.csv_header import read_csv_header
from src.data_refinement.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)
from src.data_retrieval.seventeenlands.downloader import SeventeenLandsDownloader
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

# CSV bytes per record batch: an I/O knob only, never changes which
# decks are stored
_BLOCK_SIZE = 64 << 20


class SeventeenLandsGameDataDeckExtractionStage:
    """Translates 17Lands' game_data CSVs directly into a DeckBox.

    Single-consumer to src/data_refinement/deck_box/ — no other
    container depends on this class directly.

    Runtime: about 3 h 20 min for all 133 game_data CSVs (85 GB, about 7 MB/s
    on average) into a fresh box: 6,911,380 decks, a 40 GB mtg.db. Later files
    run slower as the box grows (measured 2026-10-08).
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data"
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every distinct decklist played in raw_path's games on box.

        Inputs:
            raw_path: one game_data CSV, or a directory of them (its
                *.csv files in sorted order). None means
                DEFAULT_RAW_PATH.
            box: the DeckBox to create decks on, under SOURCE_GAME.
            card_lookup: populated for SOURCE_GAME, Unknown sentinel
                included.
        Output: every deck_uuid this call stored or found already
            stored, each once, in order of first sight.
        Side effects: reads every CSV; box.create_if_absent() once per
            distinct deck; logs each CSV's unmatched card names; one tqdm
            progress bar per CSV on stderr.
        Exceptions: RuntimeError if the Unknown sentinel is not on
            card_lookup; ValueError if a CSV lacks a required scalar
            column or a batch holds a null scalar; whatever pyarrow or
            DeckBox raise.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> box = DeckBox.load([Path("data/final/decks/mtg.db")])
            >>> stage = SeventeenLandsGameDataDeckExtractionStage()
            >>> deck_uuids = stage.extract(None, box, binder)
        """
        # Validate inputs: unmatched columns stand for the sentinel
        _require_unknown_card(card_lookup, self.SOURCE_GAME)

        path = raw_path if raw_path is not None else self.DEFAULT_RAW_PATH
        csv_paths = sorted(path.glob("*.csv")) if path.is_dir() else [path]

        # Each CSV in turn; a deck already handled (this call) is skipped
        stored: dict[UUID, None] = {}
        for csv_path in csv_paths:
            self._extract_csv(csv_path, box, card_lookup, stored)
        return list(stored)

    def _extract_csv(
        self,
        csv_path: Path,
        box: DeckBox,
        card_lookup: CardLookup,
        stored: dict[UUID, None],
    ) -> None:
        """Store one CSV's distinct decks not already in stored.

        Inputs: csv_path, box, card_lookup, stored (deck uuids this
            extract() call already handled, in order; mutated).
        Output: none.
        Side effects: reads csv_path in record batches; adds each new
            deck to box and to stored; logs unmatched card names.
        Exceptions: as extract().
        """
        header = read_csv_header(csv_path)
        parser = GameDataChunkParser.from_header(header, card_lookup, self.SOURCE_GAME)
        _log_unmatched_names(csv_path, parser)
        convert_options = pa_csv.ConvertOptions(
            include_columns=parser.needed_columns(),
            column_types=parser.column_types(),
        )
        read_options = pa_csv.ReadOptions(block_size=_BLOCK_SIZE)
        label = csv_path.stem  # "<Expansion>.<FormatCode>"

        with open(csv_path, "rb") as raw_file, tqdm(
            total=csv_path.stat().st_size,
            unit="B",
            unit_scale=True,
            desc=f"seventeenlands_game_data extract: {csv_path.name}",
        ) as progress:
            reader = pa_csv.open_csv(
                raw_file, read_options=read_options, convert_options=convert_options
            )
            for batch in reader:
                chunk = parser.parse(batch)
                self._store_new_decks(chunk.decks, chunk.keys, label, box, stored)
                progress.update(raw_file.tell() - progress.n)

    def _store_new_decks(
        self,
        decks: ChunkDecks,
        keys: GameKeys,
        label: str,
        box: DeckBox,
        stored: dict[UUID, None],
    ) -> None:
        """create_if_absent() each of a chunk's decks not yet in stored,
        named after (and sourced from) its first row's game.

        Inputs: decks and keys (one chunk's), label (the CSV's
            "<Expansion>.<FormatCode>"), box, stored (mutated).
        Output: none.
        Side effects: box.create_if_absent() per new deck; adds its uuid
            to stored.
        Exceptions: DeckBox's batch-wide errors.
        """
        deck_indexes, first_rows = np.unique(decks.row_deck, return_index=True)
        fetched_at = datetime.now(timezone.utc)
        for deck_index, row in zip(deck_indexes, first_rows):
            deck = decks.decks[deck_index]
            if deck.nocab_uuid in stored:
                continue
            game = (
                f"{keys.draft_id[row]}:{keys.match_number[row]}:{keys.game_number[row]}"
            )
            box.create_if_absent(
                GenericDeck(
                    nocab_uuid=deck.nocab_uuid,
                    source_game=self.SOURCE_GAME,
                    name=f"17lands game_data {label} {game} deck",
                    card_nocab_uuids=deck.card_nocab_uuids,
                    provenance=Provenance(
                        data_source=DataSource.SEVENTEENLANDS_GAME_DATA,
                        source_id=game,
                        fetched_at=fetched_at,
                    ),
                )
            )
            stored[deck.nocab_uuid] = None


def _require_unknown_card(card_lookup: CardLookup, source_game: GameId) -> None:
    """Raise unless source_game's Unknown sentinel card is on card_lookup.

    Inputs: card_lookup, source_game. Output: none. Side effects: none.
    Exceptions: RuntimeError naming the missing seeding step.
    """
    if card_lookup.get_by_uuid(CardBinder.unknown_card_uuid(source_game)) is None:
        raise RuntimeError(
            f"SeventeenLandsGameDataDeckExtractionStage: {source_game!r}'s "
            "Unknown sentinel card is not seeded — call "
            "CardBinder.ensure_unknown_card() before extract()"
        )


def _log_unmatched_names(csv_path: Path, parser: GameDataChunkParser) -> None:
    """Log one error line naming csv_path's unmatched deck_ columns.

    Inputs: csv_path, parser (built from csv_path's header).
    Output: none. Side effects: one logging.error() when any column is
        unmatched. Exceptions: none.
    """
    unmatched = parser.unmatched_deck_columns()
    if unmatched:
        _logger.error(
            "SeventeenLandsGameDataDeckExtractionStage: %s: %d unmatched deck_ "
            "columns counted as the Unknown sentinel card: %s",
            csv_path.name,
            len(unmatched),
            ", ".join(unmatched),
        )
