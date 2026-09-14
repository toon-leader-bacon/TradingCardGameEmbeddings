"""Skeleton-stage test stubs for game_card_columns.py's GameCardColumns
- structure only, mirroring
tests/data_refinement/metrics/seventeenlands/draft_data/test_pack_pool_columns.py's
coverage shape one level over for game_data's five column prefixes.
Bodies are filled in by design-recipe-implement.
"""

from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _make_card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder_with_cards(names: list[str]) -> CardBinder:
    binder = CardBinder()
    for name in names:
        binder.create(_make_card(name))
    return binder


class TestFromHeader:
    def test_matches_all_five_column_prefixes_by_exact_name(self) -> None:
        """opening_hand_/drawn_/tutored_/deck_/sideboard_ columns for a
        matched name should each land in their own *_columns list."""
        binder = _binder_with_cards(["Owlbear"])
        header = [
            "opening_hand_Owlbear",
            "drawn_Owlbear",
            "tutored_Owlbear",
            "deck_Owlbear",
            "sideboard_Owlbear",
        ]

        game_columns = GameCardColumns.from_header(header, binder, GameId.MTG)

        assert [c for c, _ in game_columns.opening_hand_columns] == [
            "opening_hand_Owlbear"
        ]
        assert [c for c, _ in game_columns.drawn_columns] == ["drawn_Owlbear"]
        assert [c for c, _ in game_columns.tutored_columns] == ["tutored_Owlbear"]
        assert [c for c, _ in game_columns.deck_columns] == ["deck_Owlbear"]
        assert [c for c, _ in game_columns.sideboard_columns] == ["sideboard_Owlbear"]

    def test_non_prefixed_columns_are_ignored(self) -> None:
        """Top-level columns (expansion, rank, won, etc.) should never
        appear in any of the five *_columns lists."""
        binder = _binder_with_cards([])
        header = ["expansion", "event_type", "rank", "won", "num_turns"]

        game_columns = GameCardColumns.from_header(header, binder, GameId.MTG)

        assert game_columns.opening_hand_columns == []
        assert game_columns.drawn_columns == []
        assert game_columns.tutored_columns == []
        assert game_columns.deck_columns == []
        assert game_columns.sideboard_columns == []

    def test_unmatched_column_name_is_absent_from_every_list(self) -> None:
        binder = _binder_with_cards([])
        header = ["deck_Nonexistent Card"]

        game_columns = GameCardColumns.from_header(header, binder, GameId.MTG)

        assert game_columns.deck_columns == []
        assert "Nonexistent Card" in game_columns.unmatched_names

    def test_split_card_falls_back_to_front_face_regex(self) -> None:
        binder = CardBinder()
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.MTG,
            name="Bruce Banner // Hulk",
            raw_content={},
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id="bruce-banner",
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder.create(card)
        header = ["deck_Bruce Banner"]

        game_columns = GameCardColumns.from_header(header, binder, GameId.MTG)

        assert game_columns.deck_columns == [("deck_Bruce Banner", card.nocab_uuid)]

    def test_ambiguous_exact_match_falls_back_and_then_gives_up_if_still_ambiguous(
        self,
    ) -> None:
        binder = CardBinder()
        binder.create(_make_card("Strike"))
        binder.create(_make_card("Strike"))
        header = ["deck_Strike"]

        game_columns = GameCardColumns.from_header(header, binder, GameId.MTG)

        assert game_columns.deck_columns == []
        assert "Strike" in game_columns.unmatched_names


class TestUuidForName:
    def test_caches_result_across_calls(self) -> None:
        binder = _binder_with_cards(["Owlbear"])
        game_columns = GameCardColumns(binder, GameId.MTG)

        first = game_columns.uuid_for_name("Owlbear")
        second = game_columns.uuid_for_name("Owlbear")

        assert first is not None
        assert first == second

    def test_unmatched_name_caches_none(self) -> None:
        binder = _binder_with_cards([])
        game_columns = GameCardColumns(binder, GameId.MTG)

        result = game_columns.uuid_for_name("Nonexistent Card")

        assert result is None
        assert "Nonexistent Card" in game_columns.unmatched_names


class TestPresentUuids:
    def test_only_columns_with_a_positive_count_are_present(self) -> None:
        """A column set here is a copy COUNT (e.g. deck_<name> summing
        to 40), not a per-copy list entry - present_uuids() should
        still only report presence once per qualifying card, matching
        draft_data's own present_uuids() semantics."""
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        header = ["deck_Owlbear", "deck_Goblin Morningstar"]
        game_columns = GameCardColumns.from_header(header, binder, GameId.MTG)
        owlbear_uuid = game_columns.uuid_for_name("Owlbear")
        row = {"deck_Owlbear": 4, "deck_Goblin Morningstar": 0}

        result = game_columns.present_uuids(row, game_columns.deck_columns)

        assert result == [owlbear_uuid]

    def test_nan_cell_is_treated_as_absent(self) -> None:
        binder = _binder_with_cards(["Owlbear"])
        header = ["deck_Owlbear"]
        game_columns = GameCardColumns.from_header(header, binder, GameId.MTG)
        row = {"deck_Owlbear": float("nan")}

        result = game_columns.present_uuids(row, game_columns.deck_columns)

        assert result == []


def test_unmatched_names_only_lists_names_that_never_matched() -> None:
    binder = _binder_with_cards(["Owlbear"])
    game_columns = GameCardColumns(binder, GameId.MTG)

    game_columns.uuid_for_name("Owlbear")
    game_columns.uuid_for_name("Nonexistent Card")

    assert game_columns.unmatched_names == ["Nonexistent Card"]
