from pathlib import Path

from src.data_refinement.deck_box.limitless_play.card_index import (
    CardMatchKind,
    LimitlessCardIndex,
)
from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardEntry,
)
from tests.data_refinement.deck_box.limitless_play.fixtures import (
    card_uuid,
    make_binder,
    write_sets,
)


class TestLimitlessCardIndex:
    def test_matches_printing_by_set_code_and_number(self, tmp_path: Path) -> None:
        binder = make_binder([("sv4-57", "Pikachu")])
        index = LimitlessCardIndex.from_sets_file(write_sets(tmp_path), binder)

        result = index.match_card_uuid(LimitlessCardEntry(1, "PAR", "57", "Pikachu"))

        assert result == card_uuid(binder, "sv4-57")
        assert index.match_counts == {CardMatchKind.PRINTING: 1}

    def test_tries_every_set_sharing_a_code(self, tmp_path: Path) -> None:
        binder = make_binder([("sma-SV49", "Charizard-GX")])
        index = LimitlessCardIndex.from_sets_file(write_sets(tmp_path), binder)

        result = index.match_card_uuid(LimitlessCardEntry(1, "HIF", "SV49", "x"))

        assert result == card_uuid(binder, "sma-SV49")

    def test_limitless_promo_code_maps_to_promo_set(self, tmp_path: Path) -> None:
        binder = make_binder([("svp-155", "Wooper")])
        index = LimitlessCardIndex.from_sets_file(write_sets(tmp_path), binder)

        result = index.match_card_uuid(LimitlessCardEntry(1, "SVP", "155", "Wooper"))

        assert result == card_uuid(binder, "svp-155")

    def test_falls_back_to_newest_same_name_card(self, tmp_path: Path) -> None:
        binder = make_binder(
            [("base1-99", "Fire Energy"), ("sv1-2", "Fire Energy"), ("sv4-3", "X")]
        )
        index = LimitlessCardIndex.from_sets_file(write_sets(tmp_path), binder)

        result = index.match_card_uuid(LimitlessCardEntry(8, "MEE", "2", "Fire Energy"))

        assert result == card_uuid(binder, "sv1-2")
        assert index.match_counts == {CardMatchKind.NAME: 1}

    def test_unmatched_is_none_and_counted_per_entry(self, tmp_path: Path) -> None:
        index = LimitlessCardIndex.from_sets_file(write_sets(tmp_path), make_binder([]))
        entry = LimitlessCardEntry(1, "ZZZ", "1", "Nobody")

        assert index.match_card_uuid(entry) is None
        assert index.match_card_uuid(entry) is None
        assert index.match_counts == {CardMatchKind.UNMATCHED: 2}
        assert index.unmatched_counts == {("ZZZ", "1", "Nobody"): 2}
