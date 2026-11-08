from src.data_refinement.deck_box.card_matching import CardMatchKind
from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardRef,
)
from src.data_refinement.deck_box.limitless_play.tcg_card_index import (
    LimitlessTcgCardIndex,
    fold_card_name,
    index_names,
)
from src.schema.game_id import GameId
from tests.data_refinement.deck_box.limitless_play.tcg_fixtures import (
    card_uuid_named,
    make_tcg_binder,
)

_DIGIMON = GameId.DIGIMON
_SWU = GameId.STAR_WARS_UNLIMITED


class TestLimitlessTcgCardIndex:
    def test_matches_a_printed_number_before_the_name(self) -> None:
        binder = make_tcg_binder(
            _DIGIMON, [("Agumon", "BT1-010"), ("Greymon", "BT1-020")]
        )
        index = LimitlessTcgCardIndex.from_card_lookup(_DIGIMON, binder, True)

        # The number names Agumon, whatever name the entry carries
        matched = index.match_card_uuid(LimitlessCardRef("BT1", "010", "Greymon"))

        assert matched == card_uuid_named(binder, _DIGIMON, "Agumon")
        assert index.match_counts == {CardMatchKind.PRINTING: 1}

    def test_number_hit_and_name_fallback_are_counted(self) -> None:
        binder = make_tcg_binder(_DIGIMON, [("Agumon", "BT1-010"), ("Gabumon", None)])
        index = LimitlessTcgCardIndex.from_card_lookup(_DIGIMON, binder, True)

        by_number = index.match_card_uuid(LimitlessCardRef("BT1", "010", "x"))
        by_name = index.match_card_uuid(LimitlessCardRef("P", "9", "Gabumon"))

        assert by_number == card_uuid_named(binder, _DIGIMON, "Agumon")
        assert by_name == card_uuid_named(binder, _DIGIMON, "Gabumon")
        assert index.match_counts == {CardMatchKind.PRINTING: 1, CardMatchKind.NAME: 1}

    def test_a_name_several_cards_share_matches_nothing(self) -> None:
        binder = make_tcg_binder(
            _DIGIMON, [("Agumon", "BT1-010"), ("Agumon", "BT5-001")]
        )
        index = LimitlessTcgCardIndex.from_card_lookup(_DIGIMON, binder, True)
        key = LimitlessCardRef("EX9", "001", "Agumon")

        assert index.match_card_uuid(key) is None
        assert index.match_card_uuid(key) is None
        assert index.match_counts == {CardMatchKind.UNMATCHED: 2}
        assert index.unmatched_counts == {LimitlessCardRef("EX9", "001", "Agumon"): 2}

    def test_a_name_only_game_ignores_printed_numbers(self) -> None:
        binder = make_tcg_binder(
            _SWU,
            [("Sabine Wren - Explosives Artist", "SOR-142"), ("Other", "SOR-001")],
        )
        index = LimitlessTcgCardIndex.from_card_lookup(_SWU, binder, False)

        matched = index.match_card_uuid(
            LimitlessCardRef("SOR", "001", "Sabine Wren, Explosives Artist")
        )

        assert matched == card_uuid_named(
            binder, _SWU, "Sabine Wren - Explosives Artist"
        )
        assert index.match_counts == {CardMatchKind.NAME: 1}

    def test_the_unknown_sentinel_is_never_matched_by_name(self) -> None:
        binder = make_tcg_binder(_SWU, [("Other", None)])
        index = LimitlessTcgCardIndex.from_card_lookup(_SWU, binder, False)

        assert index.match_card_uuid(LimitlessCardRef("X", "1", "Unknown")) is None


class TestIndexNames:
    def test_unique_and_shared_names(self) -> None:
        binder = make_tcg_binder(
            _DIGIMON,
            [("Agumon", "BT1-010"), ("Agumon", "BT5-001"), ("Gabumon", None)],
        )

        names = index_names(
            card for card in binder.all_cards(_DIGIMON) if card.name != "Unknown"
        )

        assert names == {
            "agumon": None,
            "gabumon": card_uuid_named(binder, _DIGIMON, "Gabumon"),
        }

    def test_an_empty_binder(self) -> None:
        assert index_names([]) == {}


class TestFoldCardName:
    def test_subtitle_separators_agree(self) -> None:
        assert fold_card_name("Sabine Wren, Explosives Artist") == fold_card_name(
            "Sabine Wren - Explosives Artist"
        )

    def test_diacritics_zero_width_spaces_and_quotes_fold(self) -> None:
        assert fold_card_name("Te Kā - Heartless") == "te ka - heartless"
        assert fold_card_name("Airframe​ Seizure") == "airframe seizure"
        assert fold_card_name("Jabba’s  Palace ") == "jabba's palace"

    def test_hyphens_inside_a_word_stay(self) -> None:
        assert fold_card_name("Spider-Man") == "spider-man"
