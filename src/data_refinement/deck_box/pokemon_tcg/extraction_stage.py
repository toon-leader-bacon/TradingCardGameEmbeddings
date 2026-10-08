"""Translates pokemon-tcg-data's theme decks into a DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface, and spire_codex_runs/extraction_stage.py for the
sibling this follows (deterministic deck uuids, create-or-update, the
Unknown sentinel on a card miss).

RAW SHAPE: data/raw/pokemon_tcg/decks/<set id>.json, written by
src/data_retrieval/pokemon_tcg/downloader.py - one JSON list per file of
{"id": "d-base1-1", "name", "types", "cards": [{"id": "base1-47",
"name", "rarity", "count"}]}. Card ids are pokemon_tcg printing ids, the
same ids the card binder registers as POKEMON_TCG aliases. These are
official prefab theme and starter decks (60 cards; a couple list 61),
not player-built ones.

IDENTITY: deck uuid = uuid5(_DECK_NAMESPACE, deck "id"), so a re-run
updates rather than duplicates.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid5

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_retrieval.pokemon_tcg.downloader import PokemonTcgDataDownloader
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

# Fixed, arbitrary - never regenerate (deck identity depends on it)
_DECK_NAMESPACE = UUID("0f6a3c1e-7d2b-4e59-9a84-5c1b2d3e4f60")


class PokemonTcgDeckExtractionStage:
    """pokemon-tcg-data theme decks -> GenericDecks (Pokemon)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        PokemonTcgDataDownloader.DEFAULT_RAW_DATA_DIR / "decks"
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Create or update one deck per theme deck in every *.json file
        under raw_path.

        Inputs: raw_path (the decks directory; DEFAULT_RAW_PATH when
            None), box (written to), card_lookup (must hold Pokemon's
            Unknown sentinel card; see CardBinder.ensure_unknown_card).
        Output: nocab_uuid of every deck created or whose card multiset
            changed; a re-seen, unchanged deck is not listed.
        Side effects: reads the files (sorted by name); creates/updates
            decks on box; logs one error per card id that falls back to
            the Unknown sentinel.
        Exceptions: FileNotFoundError if raw_path is not a directory;
            json.JSONDecodeError / KeyError on a malformed file or deck;
            RuntimeError if the Unknown sentinel is missing.

        Example:
            >>> PokemonTcgDeckExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        directory = raw_path or self.DEFAULT_RAW_PATH
        if not directory.is_dir():
            raise FileNotFoundError(f"no pokemon_tcg decks directory at {directory}")

        # One file per set, each a list of decks
        for path in sorted(directory.glob("*.json")):
            for raw_deck in json.loads(path.read_text(encoding="utf-8")):
                changed = self._extract_deck(raw_deck, box, card_lookup)
                if changed is not None:
                    result.append(changed)
        return result

    def _extract_deck(
        self, raw_deck: dict, box: DeckBox, card_lookup: CardLookup
    ) -> UUID | None:
        """Create or update one theme deck.

        Inputs: raw_deck (one entry of a decks file), box, card_lookup.
        Output: the deck's uuid if it was created or its cards changed,
            else None.
        Side effects: creates/updates the deck on box.
        Exceptions: KeyError if raw_deck lacks "id" or "cards", or a
            card entry lacks "id" or "count"; as _card_uuid.
        """
        deck_id = raw_deck["id"]
        deck_uuid = uuid5(_DECK_NAMESPACE, deck_id)
        # A deck lists each card once with its copy count
        card_nocab_uuids = [
            self._card_uuid(entry["id"], card_lookup)
            for entry in raw_deck["cards"]
            for _ in range(entry["count"])
        ]
        provenance = Provenance(
            data_source=DataSource.POKEMON_TCG,
            source_id=deck_id,
            fetched_at=datetime.now(timezone.utc),
        )

        existing = box.get_by_uuid(deck_uuid)
        if existing is None:
            box.create(
                GenericDeck(
                    nocab_uuid=deck_uuid,
                    source_game=self.SOURCE_GAME,
                    name=raw_deck.get("name", deck_id),
                    card_nocab_uuids=card_nocab_uuids,
                    provenance=provenance,
                )
            )
            return deck_uuid
        if Counter(existing.card_nocab_uuids) != Counter(card_nocab_uuids):
            box.update(
                deck_uuid, card_nocab_uuids=card_nocab_uuids, provenance=provenance
            )
            return deck_uuid
        return None

    def _card_uuid(self, card_id: str, card_lookup: CardLookup) -> UUID:
        """The nocab_uuid for a pokemon_tcg card id, or the Unknown
        sentinel's on a miss.

        Inputs: card_id (e.g. "base1-47"), card_lookup.
        Output: UUID. Side effects: logs an error on a miss.
        Exceptions: RuntimeError if the Unknown sentinel is not seeded.
        """
        card = card_lookup.get_by_alias(
            self.SOURCE_GAME, DataSource.POKEMON_TCG, card_id
        )
        if card is not None:
            return card.nocab_uuid

        _logger.error(
            "PokemonTcgDeckExtractionStage: unknown card id %r; substituting "
            "the Unknown sentinel card",
            card_id,
        )
        unknown_card = card_lookup.get_by_name_single(
            self.SOURCE_GAME, CardBinder.UNKNOWN_CARD_NAME, strict=False
        )
        if unknown_card is None:
            raise RuntimeError(
                f"{self.SOURCE_GAME!r}'s Unknown sentinel card is not seeded; "
                "call CardBinder.ensure_unknown_card() before extract()"
            )
        return unknown_card.nocab_uuid
