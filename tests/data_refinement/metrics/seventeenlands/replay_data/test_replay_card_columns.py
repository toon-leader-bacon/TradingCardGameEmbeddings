"""Tests for replay_card_columns.py's ReplayCardColumns - structure
mirrors
tests/data_refinement/metrics/seventeenlands/game_data/test_game_card_columns.py's
coverage shape, extended for this class's second (Arena-ID) matching
mechanism and per-actor turn-number discovery.
"""

from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
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
    def test_matches_deck_and_sideboard_columns_by_exact_name(self) -> None:
        """deck_/sideboard_ columns for a matched name should each land
        in their own *_columns list - only these two prefixes exist in
        this container (no opening_hand_<name>/drawn_<name>/
        tutored_<name> the way game_data has)."""
        binder = _binder_with_cards(["Owlbear"])
        header = ["deck_Owlbear", "sideboard_Owlbear"]

        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)

        assert [c for c, _ in replay_columns.deck_columns] == ["deck_Owlbear"]
        assert [c for c, _ in replay_columns.sideboard_columns] == ["sideboard_Owlbear"]

    def test_non_prefixed_columns_are_ignored(self) -> None:
        """Top-level columns (expansion, num_turns, won, per-turn
        event columns, etc.) should never appear in deck_columns/
        sideboard_columns."""
        binder = _binder_with_cards([])
        header = ["expansion", "num_turns", "won", "user_turn_1_creatures_cast"]

        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)

        assert replay_columns.deck_columns == []
        assert replay_columns.sideboard_columns == []

    def test_unmatched_column_name_is_absent_from_every_list(self) -> None:
        binder = _binder_with_cards([])
        header = ["deck_Nonexistent Card"]

        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)

        assert replay_columns.deck_columns == []
        assert "Nonexistent Card" in replay_columns.unmatched_names

    def test_split_card_falls_back_to_front_face_regex(self) -> None:
        """Same _match_uuid() fallback policy as DraftCardColumns/
        GameCardColumns."""
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

        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)

        assert replay_columns.deck_columns == [("deck_Bruce Banner", card.nocab_uuid)]

    def test_ambiguous_exact_match_falls_back_and_then_gives_up_if_still_ambiguous(
        self,
    ) -> None:
        binder = CardBinder()
        binder.create(_make_card("Strike"))
        binder.create(_make_card("Strike"))
        header = ["deck_Strike"]

        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)

        assert replay_columns.deck_columns == []
        assert "Strike" in replay_columns.unmatched_names

    def test_discovers_user_and_oppo_turn_numbers_from_header(self) -> None:
        """A header with user_turn_1_*/user_turn_2_*/oppo_turn_1_*
        columns should yield user_turn_numbers == [1, 2],
        oppo_turn_numbers == [1] - derived from header, never a
        hardcoded range."""
        binder = _binder_with_cards([])
        header = [
            "user_turn_1_creatures_cast",
            "user_turn_2_creatures_cast",
            "oppo_turn_1_creatures_cast",
        ]

        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)

        assert replay_columns.user_turn_numbers == [1, 2]
        assert replay_columns.oppo_turn_numbers == [1]

    def test_turn_numbers_are_sorted_and_deduplicated(self) -> None:
        binder = _binder_with_cards([])
        header = [
            "user_turn_3_creatures_cast",
            "user_turn_1_creatures_cast",
            "user_turn_3_creatures_attacked",
        ]

        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)

        assert replay_columns.user_turn_numbers == [1, 3]


class TestUuidForName:
    def test_caches_result_across_calls(self) -> None:
        binder = _binder_with_cards(["Owlbear"])
        replay_columns = ReplayCardColumns(binder, GameId.MTG)

        first = replay_columns.uuid_for_name("Owlbear")
        second = replay_columns.uuid_for_name("Owlbear")

        assert first is not None
        assert first == second

    def test_unmatched_name_caches_none(self) -> None:
        binder = _binder_with_cards([])
        replay_columns = ReplayCardColumns(binder, GameId.MTG)

        result = replay_columns.uuid_for_name("Nonexistent Card")

        assert result is None
        assert "Nonexistent Card" in replay_columns.unmatched_names


class TestUuidForArenaId:
    def test_caches_result_across_calls(self) -> None:
        binder = _binder_with_cards(["Owlbear"])
        card_uuid = binder.get_by_name(GameId.MTG, "Owlbear")[0].nocab_uuid
        binder.register_alias(GameId.MTG, DataSource.ARENA, "104936", card_uuid)
        replay_columns = ReplayCardColumns(binder, GameId.MTG)

        first = replay_columns.uuid_for_arena_id("104936")
        second = replay_columns.uuid_for_arena_id("104936")

        assert first == card_uuid
        assert first == second

    def test_unmatched_arena_id_caches_none(self) -> None:
        binder = _binder_with_cards([])
        replay_columns = ReplayCardColumns(binder, GameId.MTG)

        result = replay_columns.uuid_for_arena_id("999999")

        assert result is None
        assert "999999" in replay_columns.unmatched_arena_ids

    def test_uses_a_cache_separate_from_uuid_for_name(self) -> None:
        """An Arena id string and a bare card name are different key
        spaces - a collision between the two (e.g. a card literally
        named "104936") must not cross-contaminate either cache."""
        binder = _binder_with_cards(["104936"])
        replay_columns = ReplayCardColumns(binder, GameId.MTG)

        name_result = replay_columns.uuid_for_name("104936")
        arena_result = replay_columns.uuid_for_arena_id("104936")

        assert name_result is not None
        assert arena_result is None
        assert "104936" in replay_columns.unmatched_arena_ids
        assert "104936" not in replay_columns.unmatched_names


class TestArenaUuids:
    def _replay_columns_with_alias(self, name: str, arena_id: str) -> ReplayCardColumns:
        binder = _binder_with_cards([name])
        card_uuid = binder.get_by_name(GameId.MTG, name)[0].nocab_uuid
        binder.register_alias(GameId.MTG, DataSource.ARENA, arena_id, card_uuid)
        return ReplayCardColumns(binder, GameId.MTG)

    def test_matches_a_pipe_delimited_cell_to_every_matched_card(self) -> None:
        binder = CardBinder()
        owlbear = _make_card("Owlbear")
        wasp = _make_card("The Wondrous Wasp")
        binder.create(owlbear)
        binder.create(wasp)
        binder.register_alias(GameId.MTG, DataSource.ARENA, "1", owlbear.nocab_uuid)
        binder.register_alias(GameId.MTG, DataSource.ARENA, "2", wasp.nocab_uuid)
        replay_columns = ReplayCardColumns(binder, GameId.MTG)

        result = replay_columns.arena_uuids("1|2")

        assert set(result) == {owlbear.nocab_uuid, wasp.nocab_uuid}

    def test_nan_cell_yields_empty_list(self) -> None:
        replay_columns = self._replay_columns_with_alias("Owlbear", "104936")

        assert replay_columns.arena_uuids(float("nan")) == []
        assert replay_columns.arena_uuids(None) == []

    def test_normalizes_a_bare_float_cell_before_lookup(self) -> None:
        """The dtype trap this class exists to close: a cell arriving
        as the Python float 104936.0 (pandas' single-id-column
        inference) must match the same card as the string "104936"
        would - str(int(float(token))) normalization, not a naive
        str() call."""
        replay_columns = self._replay_columns_with_alias("Owlbear", "104936")

        result = replay_columns.arena_uuids(104936.0)

        assert result == replay_columns.arena_uuids("104936")
        assert len(result) == 1

    def test_normalizes_a_float_suffixed_string_token_before_lookup(self) -> None:
        """A "|"-split piece that itself looks like "104936.0" (not
        just a bare float cell) must normalize the same way."""
        replay_columns = self._replay_columns_with_alias("Owlbear", "104936")

        result = replay_columns.arena_uuids("104936.0")

        assert len(result) == 1

    def test_unmatched_token_is_dropped_not_raised(self) -> None:
        replay_columns = self._replay_columns_with_alias("Owlbear", "104936")

        result = replay_columns.arena_uuids("104936|999999")

        assert len(result) == 1
        assert "999999" in replay_columns.unmatched_arena_ids


class TestPresentUuids:
    def test_only_columns_with_a_positive_count_are_present(self) -> None:
        binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
        header = ["deck_Owlbear", "deck_Goblin Morningstar"]
        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)
        owlbear_uuid = replay_columns.uuid_for_name("Owlbear")
        row = {"deck_Owlbear": 4, "deck_Goblin Morningstar": 0}

        result = replay_columns.present_uuids(row, replay_columns.deck_columns)

        assert result == [owlbear_uuid]

    def test_nan_cell_is_treated_as_absent(self) -> None:
        binder = _binder_with_cards(["Owlbear"])
        header = ["deck_Owlbear"]
        replay_columns = ReplayCardColumns.from_header(header, binder, GameId.MTG)
        row = {"deck_Owlbear": float("nan")}

        result = replay_columns.present_uuids(row, replay_columns.deck_columns)

        assert result == []


class TestTurnColumn:
    def test_builds_the_literal_column_name(self) -> None:
        """ReplayCardColumns.turn_column("oppo", 5, "creatures_attacked")
        == "oppo_turn_5_creatures_attacked"."""
        assert (
            ReplayCardColumns.turn_column("oppo", 5, "creatures_attacked")
            == "oppo_turn_5_creatures_attacked"
        )
        assert (
            ReplayCardColumns.turn_column("user", 3, "creatures_cast")
            == "user_turn_3_creatures_cast"
        )


def test_unmatched_names_only_lists_names_that_never_matched() -> None:
    binder = _binder_with_cards(["Owlbear"])
    replay_columns = ReplayCardColumns(binder, GameId.MTG)

    replay_columns.uuid_for_name("Owlbear")
    replay_columns.uuid_for_name("Nonexistent Card")

    assert replay_columns.unmatched_names == ["Nonexistent Card"]


def test_unmatched_arena_ids_only_lists_ids_that_never_matched() -> None:
    binder = _binder_with_cards(["Owlbear"])
    card_uuid = binder.get_by_name(GameId.MTG, "Owlbear")[0].nocab_uuid
    binder.register_alias(GameId.MTG, DataSource.ARENA, "104936", card_uuid)
    replay_columns = ReplayCardColumns(binder, GameId.MTG)

    replay_columns.uuid_for_arena_id("104936")
    replay_columns.uuid_for_arena_id("999999")

    assert replay_columns.unmatched_arena_ids == ["999999"]
