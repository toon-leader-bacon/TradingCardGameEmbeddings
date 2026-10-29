"""Matches Limitless decklist entries to Pokemon binder cards.

The Pokemon binder (card_binder/pokemon_tcg/) registers every printing
under its pokemon-tcg-data id, "<set id>-<number>" (e.g. "sv4-57"), as a
DataSource.POKEMON_TCG alias. Limitless names a printing by PTCGL set
code and number ("PAR", "57"). pokemon-tcg-data's sets/en.json maps set
id to that code (its "ptcgoCode"), so most entries match exactly.

MATCH ORDER (first hit wins), measured on three sampled tournaments
(Expanded, GLC, Base-Fossil; 2026-10-28): 97% matched at step 1.

    1. PRINTING: every set id carrying the entry's set code, tried as
       "<set id>-<number>". A code can name two sets (HIF is Hidden
       Fates and its Shiny Vault); both are tried, in sets/en.json
       order. Codes Limitless spells differently from ptcgoCode
       (promos, energies) come from _LIMITLESS_SET_CODES.
    2. NAME: the binder's cards with the entry's exact name, taking the
       one from the newest set (sets/en.json releaseDate; ties broken
       by set id). The binder keys a card by (name, set), and
       reprints keep their rules text, so for Trainers and Energies any
       same-name card is right. A Pokemon with that name in another
       set may be a different card; the newest is the likeliest one
       an online player used. This catches basic energies Limitless
       files under sets pokemon-tcg-data lacks (e.g. "MEE").
    3. UNMATCHED: None; the caller substitutes the Unknown sentinel.

Matches are cached by (set code, number, name): a run sees the same
few thousand printings across hundreds of thousands of decks.
"""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardEntry,
)
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# Limitless set codes that differ from pokemon-tcg-data's ptcgoCode.
# Observed in sampled standings: SVP, SMP, WP. The other promo codes
# follow the same "<era>P" pattern; a wrong guess only falls to step 2.
_LIMITLESS_SET_CODES: dict[str, str] = {
    "WP": "basep",
    "NP": "np",
    "DPP": "dpp",
    "HSP": "hsp",
    "BWP": "bwp",
    "XYP": "xyp",
    "SMP": "smp",
    "SWSHP": "swshp",
    "SVP": "svp",
}


class CardMatchKind(str, Enum):
    """Which match step found a card (for the stage's summary log)."""

    PRINTING = "printing"
    NAME = "name"
    UNMATCHED = "unmatched"


@dataclass(frozen=True)
class PokemonSet:
    """One sets/en.json row, as far as matching needs it."""

    set_id: str
    ptcgo_code: str | None
    release_date: date


class LimitlessCardIndex:
    """Limitless (set code, number, name) -> binder nocab_uuid, cached.

    Single-consumer to LimitlessPlayDeckExtractionStage.
    """

    _GAME: ClassVar[GameId] = GameId.POKEMON

    def __init__(self, sets: list[PokemonSet], card_lookup: CardLookup) -> None:
        """
        Inputs: sets (every set, in sets/en.json order), card_lookup
            (the Pokemon binder, read-only).
        Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._set_ids_by_code = _index_set_ids_by_code(sets)
        self._release_date_by_set_id = {s.set_id: s.release_date for s in sets}
        self._match_cache: dict[
            tuple[str, str, str], tuple[UUID | None, CardMatchKind]
        ] = {}
        self.match_counts: Counter[CardMatchKind] = Counter()
        # (set code, number, name) of each unmatched entry, per entry
        self.unmatched_counts: Counter[tuple[str, str, str]] = Counter()

    @classmethod
    def from_sets_file(
        cls, sets_path: Path, card_lookup: CardLookup
    ) -> "LimitlessCardIndex":
        """Build an index from pokemon-tcg-data's sets/en.json.

        Inputs: sets_path, card_lookup.
        Output: LimitlessCardIndex.
        Side effects: reads sets_path.
        Exceptions: FileNotFoundError, json.JSONDecodeError, KeyError /
            ValueError on a malformed set row.

        Example:
            >>> LimitlessCardIndex.from_sets_file(
            ...     Path("data/raw/pokemon_tcg/sets/en.json"), binder
            ... )
        """
        raw_sets = json.loads(sets_path.read_text(encoding="utf-8"))
        return cls([_parse_set(raw_set) for raw_set in raw_sets], card_lookup)

    def match_card_uuid(self, entry: LimitlessCardEntry) -> UUID | None:
        """The binder card an entry names, or None if no step matches.

        Inputs: entry. Output: nocab_uuid or None.
        Side effects: fills the cache; counts each lookup's match kind
            in match_counts, and each unmatched key in unmatched_counts
            (cache hits included, so the counts are per decklist entry).
        Exceptions: none.

        Example:
            >>> index.match_card_uuid(LimitlessCardEntry(4, "PAR", "57", "..."))
        """
        key = (entry.set_code, entry.number, entry.name)
        if key not in self._match_cache:
            self._match_cache[key] = self._match(entry)
        result, match_kind = self._match_cache[key]
        self.match_counts[match_kind] += 1
        if match_kind is CardMatchKind.UNMATCHED:
            self.unmatched_counts[key] += 1
        return result

    def _match(self, entry: LimitlessCardEntry) -> tuple[UUID | None, CardMatchKind]:
        """Steps 1 then 2 of the module docstring's MATCH ORDER.

        Inputs: entry. Output: (nocab_uuid or None, the step that hit).
        Side effects: none (card_lookup reads only).
        Exceptions: none.
        """
        card = self._card_by_printing(entry)
        if card is not None:
            return card.nocab_uuid, CardMatchKind.PRINTING
        card = self._newest_card_named(entry.name)
        if card is not None:
            return card.nocab_uuid, CardMatchKind.NAME
        return None, CardMatchKind.UNMATCHED

    def _card_by_printing(self, entry: LimitlessCardEntry) -> GenericCard | None:
        """Step 1: the first set id for entry.set_code whose
        "<set id>-<number>" is a POKEMON_TCG alias.

        Inputs: entry. Output: GenericCard or None.
        Side effects: none. Exceptions: none.
        """
        for set_id in self._set_ids_by_code.get(entry.set_code, ()):
            card = self._card_lookup.get_by_alias(
                self._GAME, DataSource.POKEMON_TCG, f"{set_id}-{entry.number}"
            )
            if card is not None:
                return card
        return None

    def _newest_card_named(self, name: str) -> GenericCard | None:
        """Step 2: the same-name binder card from the newest set.

        A card whose stored "set" isn't in sets/en.json sorts oldest.

        Inputs: name. Output: GenericCard or None.
        Side effects: none. Exceptions: none.
        """
        candidates = self._card_lookup.get_by_name(self._GAME, name)
        if not candidates:
            return None
        return max(candidates, key=self._release_order)

    def _release_order(self, card: GenericCard) -> tuple[date, str]:
        """Sort key: the card's set release date, then set id.

        Inputs: card (a Pokemon binder card; raw_content["set"] is its
            set id).
        Output: (release date, set id); date.min for a set not in
            sets/en.json.
        Side effects: none. Exceptions: none.
        """
        set_id = str(card.raw_content.get("set", ""))
        return self._release_date_by_set_id.get(set_id, date.min), set_id


def _index_set_ids_by_code(sets: list[PokemonSet]) -> dict[str, tuple[str, ...]]:
    """Set code -> set ids: every ptcgoCode, plus _LIMITLESS_SET_CODES.

    Inputs: sets. Output: dict (set ids in sets order).
    Side effects: none. Exceptions: none.
    """
    result: dict[str, list[str]] = {}
    for pokemon_set in sets:
        if pokemon_set.ptcgo_code:
            result.setdefault(pokemon_set.ptcgo_code, []).append(pokemon_set.set_id)
    for limitless_code, set_id in _LIMITLESS_SET_CODES.items():
        set_ids = result.setdefault(limitless_code, [])
        if set_id not in set_ids:
            set_ids.append(set_id)
    return {code: tuple(set_ids) for code, set_ids in result.items()}


def _parse_set(raw_set: dict) -> PokemonSet:
    """One sets/en.json row -> PokemonSet ("releaseDate" is
    "YYYY/MM/DD").

    Inputs: raw_set. Output: PokemonSet. Side effects: none.
    Exceptions: KeyError / ValueError on a malformed row.
    """
    return PokemonSet(
        set_id=raw_set["id"],
        ptcgo_code=raw_set.get("ptcgoCode") or None,
        release_date=datetime.strptime(raw_set["releaseDate"], "%Y/%m/%d").date(),
    )
