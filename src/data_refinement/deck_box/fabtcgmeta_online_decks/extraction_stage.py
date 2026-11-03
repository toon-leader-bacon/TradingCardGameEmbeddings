"""Translates fabtcgmeta.com's online (Talishar) decklists into the Flesh
and Blood DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface, and
src/data_retrieval/fabtcgmeta_online_decks/downloader.py for the source
and its shard layout.

RAW SHAPE: raw_path is the directory holding archive/ and recent/, each
with <prefix>.json shards. A shard is {deck hash: deck}, a deck being
    {"deckName": <hash>,
     "deckInfo": {"heroFormat": "Briar, Warden of Thorns Silver Age-O",
                  "hero": ..., "format": "Silver Age-O", "date": ...},
     "cards": {"heroWeaponEquipment": [entry, ...],
               "redPitchCards": [...], "yellowPitchCards": [...],
               "bluePitchCards": [...]}}
and an entry {"name": "Harness Lightning", "count": 2, "pitch": "1", ...}.
Every group is read: the hero, weapons and equipment are part of the
deck, as in fabtcg_decklists/. The group itself is dropped (GenericDeck
has no slots); the entry's pitch is kept only to pick the printing.
A deck without a "cards" object, with a line lacking a positive
integer count, or with no lines at all is skipped and counted.

CARD MATCHING (FabCardMatcher): the entry's name, stripped, looked up
    1. exactly, then
    2. by name key (casefolded, letters and digits only), which absorbs
       what differs from cardvault's names: " // " where the binder has
       "||" ("Arcane Seeds // Life"), letter case ("Consign To Cosmos"),
       dropped apostrophes ("Swordmasters Path"). A two-faced card is
       keyed by its front face too (raw_content's name: "A Drop in the
       Ocean" for the binder's "A Drop in the Ocean // Inner Chi").
A name can name several printings, one per pitch ("Harness Lightning"
red, yellow and blue): the one with the entry's pitch is taken. Hero,
weapon and equipment entries carry pitch "0" where the binder has none,
so when no printing has the entry's pitch, all of them stay candidates.
Of the remaining candidates, the lowest cardvault id is taken, so a
re-run picks the same card. A miss maps to the Unknown sentinel and is
counted; measured on 1,844 decks (2026-10-28), 99.8% of copies match,
the misses being cards newer than the binder and empty names.

IDENTITY: deck uuid = deck_uuid_from_cards() over the copy-count
multiset (src/data_refinement/deck_ids.py), stored once with
create_if_absent(). The name is deckInfo's heroFormat (hero plus
format); source_id is the deck hash. The date stays in the raw shard.

RETURN VALUE: only decks this call newly created.

PRECONDITION: card_lookup holds Flesh and Blood's Unknown sentinel
(CardBinder.ensure_unknown_card()); raises RuntimeError otherwise.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Iterator
from uuid import UUID

from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.multiset_deck import store_multiset_deck
from src.data_refinement.deck_box.unknown_card import require_unknown_card_uuid
from src.data_retrieval.fabtcgmeta_online_decks.downloader import (
    FabtcgmetaOnlineDecksDownloader,
)
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_SHARD_GLOB = "*/*.json"
_UNKNOWN_FORMAT = "unknown"
_UNMATCHED_LOG_LIMIT = 30


class OnlineDeckOutcome(str, Enum):
    """What extract() did with one raw deck (for its summary log)."""

    CREATED = "created"
    ALREADY_STORED = "already_stored"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class DeckCardEntry:
    """One card line of a deck: name (stripped), pitch as given ("1"-"3",
    "0" for heroes and gear) and copy count."""

    name: str
    pitch: str
    count: int


@dataclass(frozen=True)
class OnlineDeck:
    """One parsed online deck: its hash, its heroFormat label (the deck's
    name), its format ("CC-O", "Silver Age-O", ...) and its card lines."""

    deck_hash: str
    hero_format: str
    deck_format: str
    entries: tuple[DeckCardEntry, ...]


class FabCardMatcher:
    """Maps (name, pitch) card lines to Flesh and Blood binder cards (see
    the module docstring), the Unknown sentinel standing in for a miss,
    and counts the misses.

    Matches are cached: ~800k decks use a few thousand distinct lines.
    """

    def __init__(self, card_lookup: CardLookup, unknown_uuid: UUID) -> None:
        """
        Inputs: card_lookup (Flesh and Blood binder), unknown_uuid.
        Output: none (constructor).
        Side effects: reads every Flesh and Blood card once, to index
            them by name key.
        Exceptions: none.
        """
        self.card_lookup = card_lookup
        self.unknown_uuid = unknown_uuid
        self.unmatched_counts: Counter[str] = Counter()
        self._uuid_by_line: dict[tuple[str, str], UUID | None] = {}
        self._cards_by_key: dict[str, list[GenericCard]] = defaultdict(list)
        for card in card_lookup.all_cards(GameId.FLESH_AND_BLOOD):
            if not _is_real_card(card):
                continue
            # A two-faced card is also found by its front face, which
            # raw_content's name holds
            for key in {
                _name_key(card.name),
                _name_key(card.raw_content.get("name", card.name)),
            }:
                self._cards_by_key[key].append(card)

    def match_card_nocab_uuids(self, entries: tuple[DeckCardEntry, ...]) -> list[UUID]:
        """One binder uuid per copy, in entry order.

        Inputs: entries.
        Output: uuids (len == total of the entries' counts).
        Side effects: fills the cache; counts each unmatched copy in
            unmatched_counts.
        Exceptions: none.
        """
        result: list[UUID] = []
        for entry in entries:
            line = (entry.name, entry.pitch)
            if line not in self._uuid_by_line:
                self._uuid_by_line[line] = self._printing_uuid(entry)
            nocab_uuid = self._uuid_by_line[line]
            if nocab_uuid is None:
                self.unmatched_counts[entry.name] += entry.count
                nocab_uuid = self.unknown_uuid
            result.extend([nocab_uuid] * entry.count)
        return result

    def _printing_uuid(self, entry: DeckCardEntry) -> UUID | None:
        """The uuid of the printing entry names, or None if no card has
        that name or name key.

        Inputs: entry. Output: UUID | None. Side effects: none.
        Exceptions: none.
        """
        candidates = [
            card
            for card in self.card_lookup.get_by_name(GameId.FLESH_AND_BLOOD, entry.name)
            if _is_real_card(card)
        ] or self._cards_by_key.get(_name_key(entry.name), [])
        pitched = [
            card for card in candidates if card.raw_content.get("pitch") == entry.pitch
        ] or candidates
        if not pitched:
            return None
        return min(pitched, key=lambda card: card.provenance.source_id).nocab_uuid


class FabtcgmetaOnlineDecksExtractionStage:
    """fabtcgmeta online decklist shards -> GenericDecks (Flesh and Blood).

    Single-consumer to src/data_refinement/deck_box/.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        FabtcgmetaOnlineDecksDownloader.DEFAULT_RAW_DATA_DIR
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Store every distinct deck in raw_path's shards.

        Inputs: raw_path (the directory with archive/ and recent/;
            DEFAULT_RAW_PATH when None), box (written to), card_lookup
            (Flesh and Blood binder holding the Unknown sentinel).
        Output: nocab_uuid of every deck newly created this call.
        Side effects: reads every shard; creates decks on box; shows a
            tqdm bar; logs a summary (outcomes, formats, unmatched names).
        Exceptions: FileNotFoundError if raw_path holds no shards;
            ValueError if a shard isn't JSON; RuntimeError if
            the Unknown sentinel is missing.

        Example:
            >>> FabtcgmetaOnlineDecksExtractionStage().extract(None, box, binder)
        """
        result: list[UUID] = []
        raw_dir = raw_path or self.DEFAULT_RAW_PATH
        shard_paths = sorted(raw_dir.glob(_SHARD_GLOB))
        if not shard_paths:
            raise FileNotFoundError(f"no fabtcgmeta shards under {raw_dir}")
        matcher = FabCardMatcher(
            card_lookup, require_unknown_card_uuid(card_lookup, self.SOURCE_GAME)
        )
        outcomes: Counter[OnlineDeckOutcome] = Counter()
        formats: Counter[str] = Counter()

        # Store each deck once
        for deck in self._read_decks(shard_paths, outcomes):
            formats[deck.deck_format] += 1
            created = store_multiset_deck(
                box,
                self.SOURCE_GAME,
                deck.hero_format,
                matcher.match_card_nocab_uuids(deck.entries),
                DataSource.FABTCGMETA_ONLINE_DECKS,
                deck.deck_hash,
            )
            if created is None:
                outcomes[OnlineDeckOutcome.ALREADY_STORED] += 1
            else:
                outcomes[OnlineDeckOutcome.CREATED] += 1
                result.append(created)

        self._log_summary(outcomes, formats, matcher.unmatched_counts)
        return result

    def _read_decks(
        self, shard_paths: list[Path], outcomes: Counter[OnlineDeckOutcome]
    ) -> Iterator[OnlineDeck]:
        """Every well-formed deck in shard_paths, shard by shard.

        Inputs: shard_paths; outcomes (MALFORMED is counted into it).
        Output: an iterator of decks.
        Side effects: reads each shard; shows a tqdm bar.
        Exceptions: ValueError naming the shard if one isn't JSON.
        """
        for shard_path in tqdm(shard_paths, desc="fabtcgmeta shards", unit="shard"):
            with open(shard_path, encoding="utf-8") as shard_file:
                try:
                    shard = json.load(shard_file)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{shard_path} isn't JSON (a truncated download? "
                        "delete it and re-run the downloader)"
                    ) from error
            for deck_hash, raw_deck in shard.items():
                deck = _parse_deck(deck_hash, raw_deck)
                if deck is None:
                    outcomes[OnlineDeckOutcome.MALFORMED] += 1
                    continue
                yield deck

    def _log_summary(
        self,
        outcomes: Counter[OnlineDeckOutcome],
        formats: Counter[str],
        unmatched_names: Counter[str],
    ) -> None:
        """Log outcomes and formats at INFO, and the most common
        unmatched names at WARNING.

        Inputs: outcomes, formats, unmatched_names. Output: none.
        Side effects: up to two log lines. Exceptions: none.
        """
        _logger.info(
            "fabtcgmeta online decks: %s; formats: %s",
            {outcome.value: count for outcome, count in outcomes.items()},
            dict(formats.most_common()),
        )
        if unmatched_names:
            _logger.warning(
                "fabtcgmeta card names matched to Unknown (copies), most common "
                "first: %s",
                unmatched_names.most_common(_UNMATCHED_LOG_LIMIT),
            )


def _parse_deck(deck_hash: str, raw_deck: object) -> OnlineDeck | None:
    """One raw shard deck as an OnlineDeck, or None if it has no card
    groups, a malformed line, or no copies at all.

    Inputs: deck_hash (the shard key), raw_deck (its JSON value).
    Output: OnlineDeck | None.
    Side effects: none.
    Exceptions: none; a malformed deck or entry is returned as None.
    """
    entries: list[DeckCardEntry] = []
    if not isinstance(raw_deck, dict) or not isinstance(raw_deck.get("cards"), dict):
        return None
    deck_info = raw_deck.get("deckInfo")
    if not isinstance(deck_info, dict):
        deck_info = {}

    # Parse each group's card lines
    for group in raw_deck["cards"].values():
        if not isinstance(group, list):
            return None
        for raw_entry in group:
            entry = _parse_entry(raw_entry)
            if entry is None:
                return None
            entries.append(entry)

    if not entries:
        return None
    return OnlineDeck(
        deck_hash=deck_hash,
        hero_format=str(deck_info.get("heroFormat") or deck_hash),
        deck_format=str(deck_info.get("format") or _UNKNOWN_FORMAT),
        entries=tuple(entries),
    )


def _parse_entry(raw_entry: object) -> DeckCardEntry | None:
    """One raw card line as a DeckCardEntry, or None if it isn't an
    object with a positive integer count.

    Inputs: raw_entry. Output: DeckCardEntry | None.
    Side effects: none. Exceptions: none.
    """
    if not isinstance(raw_entry, dict):
        return None
    count = raw_entry.get("count")
    # bool is an int subclass; a count must be a real number of copies
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return None
    return DeckCardEntry(
        name=str(raw_entry.get("name", "")).strip(),
        pitch=str(raw_entry.get("pitch", "")),
        count=count,
    )


def _is_real_card(card: GenericCard) -> bool:
    """False for the Unknown sentinel (minted by the binder, not cardvault).

    Inputs: card. Output: bool. Side effects: none. Exceptions: none.
    """
    return card.provenance.data_source != DataSource.SYSTEM


def _name_key(name: str) -> str:
    """name casefolded, letters and digits only ("Arcane Seeds // Life"
    and "Arcane Seeds||Life" share "arcaneseedslife").

    Inputs: name. Output: the key. Side effects: none. Exceptions: none.
    """
    return "".join(character for character in name.casefold() if character.isalnum())
