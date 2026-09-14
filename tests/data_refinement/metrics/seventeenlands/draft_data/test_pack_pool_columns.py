from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_pool_columns import (
    DraftCardColumns,
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
    def test_matches_pack_card_and_pool_columns_by_exact_name(self) -> None:
        binder = _binder_with_cards(["Lightning Bolt", "Owlbear"])
        header = [
            "draft_id",
            "pack_card_Lightning Bolt",
            "pool_Owlbear",
            "pick",
        ]

        draft_columns = DraftCardColumns.from_header(header, binder, GameId.MTG)

        assert [column for column, _ in draft_columns.pack_columns] == [
            "pack_card_Lightning Bolt"
        ]
        assert [column for column, _ in draft_columns.pool_columns] == ["pool_Owlbear"]

    def test_non_pack_pool_columns_are_ignored(self) -> None:
        binder = _binder_with_cards([])
        header = ["draft_id", "pack_number", "pick_number", "pick", "rank"]

        draft_columns = DraftCardColumns.from_header(header, binder, GameId.MTG)

        assert draft_columns.pack_columns == []
        assert draft_columns.pool_columns == []

    def test_unmatched_column_name_is_absent_from_both_lists(self) -> None:
        binder = _binder_with_cards([])
        header = ["pack_card_Nonexistent Card"]

        draft_columns = DraftCardColumns.from_header(header, binder, GameId.MTG)

        assert draft_columns.pack_columns == []
        assert "Nonexistent Card" in draft_columns.unmatched_names

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
        header = ["pack_card_Bruce Banner"]

        draft_columns = DraftCardColumns.from_header(header, binder, GameId.MTG)

        assert draft_columns.pack_columns == [
            ("pack_card_Bruce Banner", card.nocab_uuid)
        ]

    def test_ambiguous_exact_match_falls_back_and_then_gives_up_if_still_ambiguous(
        self,
    ) -> None:
        binder = CardBinder()
        binder.create(_make_card("Strike"))
        binder.create(_make_card("Strike"))
        header = ["pack_card_Strike"]

        draft_columns = DraftCardColumns.from_header(header, binder, GameId.MTG)

        assert draft_columns.pack_columns == []
        assert "Strike" in draft_columns.unmatched_names


class TestUuidForName:
    def test_caches_result_across_calls(self) -> None:
        binder = _binder_with_cards(["Owlbear"])
        draft_columns = DraftCardColumns(binder, GameId.MTG)

        first = draft_columns.uuid_for_name("Owlbear")
        second = draft_columns.uuid_for_name("Owlbear")

        assert first is not None
        assert first == second

    def test_unmatched_name_caches_none(self) -> None:
        binder = _binder_with_cards([])
        draft_columns = DraftCardColumns(binder, GameId.MTG)

        result = draft_columns.uuid_for_name("Nonexistent Card")

        assert result is None
        assert "Nonexistent Card" in draft_columns.unmatched_names


class TestPresentUuids:
    def test_only_columns_with_a_positive_count_are_present(self) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        header = ["pack_card_Owlbear", "pack_card_Goblin Morningstar"]
        draft_columns = DraftCardColumns.from_header(header, binder, GameId.MTG)
        owlbear_uuid = draft_columns.uuid_for_name("Owlbear")
        row = {"pack_card_Owlbear": 1, "pack_card_Goblin Morningstar": 0}

        result = draft_columns.present_uuids(row, draft_columns.pack_columns)

        assert result == [owlbear_uuid]

    def test_nan_cell_is_treated_as_absent(self) -> None:
        binder = _binder_with_cards(["Owlbear"])
        header = ["pack_card_Owlbear"]
        draft_columns = DraftCardColumns.from_header(header, binder, GameId.MTG)
        row = {"pack_card_Owlbear": float("nan")}

        result = draft_columns.present_uuids(row, draft_columns.pack_columns)

        assert result == []


def test_unmatched_names_only_lists_names_that_never_matched() -> None:
    binder = _binder_with_cards(["Owlbear"])
    draft_columns = DraftCardColumns(binder, GameId.MTG)

    draft_columns.uuid_for_name("Owlbear")
    draft_columns.uuid_for_name("Nonexistent Card")

    assert draft_columns.unmatched_names == ["Nonexistent Card"]
