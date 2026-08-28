from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.column_lookup import (
    ColumnResolution,
    _build_card_column_set,
    _discover_card_names,
    find_card_columns,
)
from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str, source_game: GameId = GameId.MTG) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=source_game,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestDiscoverCardNames:
    def test_extracts_names_from_deck_columns_in_order(self) -> None:
        header = [
            "expansion",
            "deck_Bolt",
            "opening_hand_Bolt",
            "deck_Shock",
            "won",
        ]

        assert _discover_card_names(header) == ["Bolt", "Shock"]

    def test_no_deck_columns_returns_empty_list(self) -> None:
        assert _discover_card_names(["expansion", "won"]) == []

    def test_ignores_non_deck_columns_with_similar_names(self) -> None:
        header = ["opening_hand_Bolt", "drawn_Bolt", "deck_Bolt", "sideboard_Bolt"]

        assert _discover_card_names(header) == ["Bolt"]


# The exact-match/regex-fallback resolution POLICY tested here moved to
# the shared name_lookup.find_uuid_by_name() (see tmp/REFACTOR.md §1) —
# its own dedicated tests are in
# tests/data_refinement/seventeenlands/test_name_lookup.py, not
# duplicated here.


class TestBuildCardColumnSet:
    def test_builds_all_five_column_names(self) -> None:
        uuid = uuid4()

        result = _build_card_column_set("Bolt", uuid)

        assert result == CardColumnSet(
            nocab_uuid=uuid,
            opening_hand="opening_hand_Bolt",
            drawn="drawn_Bolt",
            tutored="tutored_Bolt",
            deck="deck_Bolt",
            sideboard="sideboard_Bolt",
        )


class TestResolveCardColumns:
    def test_resolves_and_splits_unresolved(self) -> None:
        registry = CardBinder()
        bolt = _card("Bolt")
        registry.add(bolt)
        header = ["expansion", "deck_Bolt", "deck_Nonexistent", "won"]

        resolution = find_card_columns(header, registry, GameId.MTG)

        assert resolution == ColumnResolution(
            resolved=[
                CardColumnSet(
                    nocab_uuid=bolt.nocab_uuid,
                    opening_hand="opening_hand_Bolt",
                    drawn="drawn_Bolt",
                    tutored="tutored_Bolt",
                    deck="deck_Bolt",
                    sideboard="sideboard_Bolt",
                )
            ],
            unresolved_names=["Nonexistent"],
        )

    def test_no_card_columns_returns_empty_resolution(self) -> None:
        registry = CardBinder()

        resolution = find_card_columns(["expansion", "won"], registry, GameId.MTG)

        assert resolution == ColumnResolution(resolved=[], unresolved_names=[])
