"""Raw playgwent.com guide rows, read as typed GuideDecks whose cards come
from the published Gwent deck box (data/final/decks/gwent.db).

The box already holds every guide's deck under
PlayGwentDeckExtractionStage.deck_uuid_for_guide(guide id) (a pure uuid5
of the guide id), so a metric here never re-resolves card ids and never
writes a deck: published deck boxes are read-only for metrics. (The
older LeaderMaskedFromDeckMetric does write its decks, through
extract_one(); see scripts/run_metrics.py's play_gwent family.)

A deck's faction is its leader card's binder faction, looked up through
the guide's own "leaderId" exactly as LeaderMaskedFromDeckMetric does,
so it uses the binder's spelling ("monster", "northern_realms"), not
the guide's slug ("monsters", "northernrealms").
"""

from dataclasses import dataclass
from typing import TypeGuard
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.play_gwent.extraction_stage import (
    PlayGwentDeckExtractionStage,
)
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# The six factions a deck can be, in the binder's spelling
GWENT_FACTIONS: tuple[str, ...] = (
    "monster",
    "nilfgaard",
    "northern_realms",
    "scoiatael",
    "skellige",
    "syndicate",
)
_NEUTRAL = "neutral"
_LEADER_COLOR = "leader"


@dataclass(frozen=True)
class GuideDeck:
    """One guide's deck, as published.

    guide_id: the guide's "id". deck_uuid: its uuid in the published
    box. faction: one of GWENT_FACTIONS. votes: the guide's net votes
    (may be negative). card_uuids: the deck's distinct cards, leader
    excluded.
    """

    guide_id: int
    deck_uuid: UUID
    faction: str
    votes: int
    card_uuids: frozenset[UUID]


class PublishedGuideDecks:
    """Reads raw guide rows as GuideDecks against the published box."""

    def __init__(self, deck_box: DeckBox, card_lookup: CardLookup) -> None:
        """
        Inputs:
            deck_box: the published Gwent box; only read.
            card_lookup: the Gwent binder the box was built from.
        Output: none (constructor). Side effects: none.
        Exceptions: none.

        Example:
            >>> box = DeckBox.load([DeckBox.default_output_path(GameId.GWENT)])
            >>> guide_decks = PublishedGuideDecks(box, binder)
        """
        self._deck_box = deck_box
        self._card_lookup = card_lookup

    def guide_deck_for_row(self, row: dict) -> GuideDeck | None:
        """Parse one raw guide row.

        Inputs: row (one parsed guides.jsonl object, carrying "id",
            "leaderId" and "votes").
        Output: its GuideDeck, or None when the box has no deck for the
            guide, the leader does not resolve to a faction card, or
            votes is not an int (rare data-quality cases: skip the row;
            every metric here skips such a guide, even GuideVotesMetric,
            which needs no faction: none of the 60,197 guides hits it).
        Side effects: reads the box (one get_by_uuid).
        Exceptions: KeyError if row has no "id".

        Example:
            >>> PublishedGuideDecks(box, binder).guide_deck_for_row(row).faction
            'nilfgaard'
        """
        guide_id = row["id"]
        votes = row.get("votes")
        # bool is an int subclass, but never an id or a vote count
        if not _is_plain_int(guide_id) or not _is_plain_int(votes):
            return None
        deck_uuid = PlayGwentDeckExtractionStage.deck_uuid_for_guide(guide_id)
        deck = self._deck_box.get_by_uuid(deck_uuid)
        leader = self._leader_for_row(row)
        if deck is None or leader is None:
            return None
        faction = leader.raw_content.get("faction")
        if faction not in GWENT_FACTIONS:
            return None
        return GuideDeck(
            guide_id=guide_id,
            deck_uuid=deck_uuid,
            faction=str(faction),
            votes=votes,
            card_uuids=frozenset(deck.card_nocab_uuids) - {leader.nocab_uuid},
        )

    def _leader_for_row(self, row: dict) -> GenericCard | None:
        """The binder card for row's "leaderId", or None.

        Inputs: row. Output: GenericCard | None.
        Side effects: none. Exceptions: none.
        """
        leader_id = row.get("leaderId")
        if leader_id is None:
            return None
        return self._card_lookup.get_by_alias(
            GameId.GWENT, DataSource.GWENT_ONE, str(leader_id)
        )


def is_leader(card: GenericCard) -> bool:
    """Whether card is a leader ability (gwent.one color "leader").

    Inputs: card. Output: bool. Side effects: none. Exceptions: none.

    Example:
        >>> is_leader(binder.get_by_name_single(GameId.GWENT, "Blood Scent"))
        True
    """
    return card.raw_content.get("color") == _LEADER_COLOR


def legal_factions(card: GenericCard) -> frozenset[str]:
    """The factions whose decks may include card.

    A neutral card fits every faction; a faction card fits its own, and a
    dual-faction card ("faction-duo", e.g. "syndicate_monster") fits both.

    Inputs: card (a Gwent card). Output: frozenset of GWENT_FACTIONS
        members; empty for a card with no known faction.
    Side effects: none. Exceptions: none.

    Example:
        >>> legal_factions(tatterwing)
        frozenset({'syndicate', 'monster'})
    """
    faction = card.raw_content.get("faction")
    if faction == _NEUTRAL:
        return frozenset(GWENT_FACTIONS)
    duo = str(card.raw_content.get("faction-duo", ""))
    # "syndicate_northern_realms": match whole faction names, since a
    # faction name can itself contain "_"
    return frozenset(name for name in GWENT_FACTIONS if name == faction or name in duo)


def _is_plain_int(value: object) -> TypeGuard[int]:
    """Whether value is an int and not a bool.

    Inputs: value. Output: bool. Side effects: none. Exceptions: none.
    """
    return isinstance(value, int) and not isinstance(value, bool)
