import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.data_refinement.card_binder.tcgjson.catalog import TcgjsonCatalog, read_catalog
from src.data_refinement.card_binder.tcgjson.game_spec import (
    DIGIMON_SPEC,
    LEADING_CARD_NUMBER,
    LORCANA_SPEC,
    NAME_ENDING_IN_CARD_NUMBER,
    ONE_PIECE_SPEC,
    UNION_ARENA_SPEC,
    TcgjsonGameSpec,
)
from src.data_refinement.card_binder.tcgjson.printing_groups import (
    PrintingGroup,
    base_name,
    build_card_content,
    group_printings,
    representative_printing,
)
from tests.data_refinement.card_binder.tcgjson.fixtures import product, write_catalog


def _catalog(
    tmp_path: Path, products: list[dict[str, Any]], **kwargs: Any
) -> TcgjsonCatalog:
    return read_catalog(write_catalog(tmp_path / "c.json.gz", products, **kwargs))


def _only_group(catalog: TcgjsonCatalog) -> PrintingGroup:
    (group,) = group_printings(catalog, LORCANA_SPEC)
    return group


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Elsa - Snow Queen", "Elsa - Snow Queen"),
        ("Elsa - Snow Queen (Epic)", "Elsa - Snow Queen"),
        ("Elsa - Snow Queen (Epic) (Foil)", "Elsa - Snow Queen"),
        ("Stitch (Version 1 of 2) ", "Stitch"),
        ("(Promo)", "(Promo)"),
        ("Usopp (3-on-3 Cup) [Winner]", "Usopp"),
    ],
)
def test_base_name(name: str, expected: str) -> None:
    assert base_name(name) == expected


def test_base_name_strips_the_suffix_after_the_tags() -> None:
    suffix = re.compile(r"\s+//.*$")
    assert base_name("Capital City // Shield (Foil)", suffix) == "Capital City"
    assert base_name("// Shield", suffix) == "// Shield"


class TestGroupPrintings:
    def test_printings_of_one_card_group_and_non_cards_are_left_out(
        self, tmp_path: Path
    ) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(30, "Elsa - Snow Queen (Epic)", rarity="Epic"),
                product(10, "Elsa - Snow Queen"),
                product(20, "Ariel - On Human Legs"),
                product(40, "Elsa Puzzle Insert (Set of 4)", card_types=[]),
                product(50, "Anna - Ensnared Sister", set_id=3, rarity="None"),
            ],
        )
        groups = group_printings(catalog, LORCANA_SPEC)

        assert [g.name for g in groups] == [
            "Elsa - Snow Queen",
            "Ariel - On Human Legs",
        ]
        assert [p.product_id for p in groups[0].products] == [10, 30]

    def test_an_empty_catalog_has_no_groups(self, tmp_path: Path) -> None:
        assert group_printings(_catalog(tmp_path, []), LORCANA_SPEC) == []


class TestRepresentativePrinting:
    def test_lowest_ladder_rarity_wins_over_promo_and_higher(
        self, tmp_path: Path
    ) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(1, "Mickey", set_id=2, rarity="Promo"),
                product(2, "Mickey (Enchanted)", rarity="Enchanted"),
                product(3, "Mickey", rarity="Legendary"),
            ],
        )
        assert (
            representative_printing(
                _only_group(catalog), catalog, LORCANA_SPEC
            ).product_id
            == 3
        )

    def test_equal_rarity_goes_to_the_earliest_set_then_lowest_id(
        self, tmp_path: Path
    ) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(5, "Elsa", set_id=1, rarity="Promo"),
                product(9, "Elsa", set_id=2, rarity="Promo"),
                product(7, "Elsa", set_id=2, rarity="Promo"),
            ],
        )
        group = _only_group(catalog)
        assert representative_printing(group, catalog, LORCANA_SPEC).product_id == 7

    def test_a_set_without_a_date_ranks_last(self, tmp_path: Path) -> None:
        sets = [
            {"setId": 1, "name": "Undated"},
            {"setId": 2, "name": "Dated", "releaseDate": "2030-01-01T00:00:00"},
        ]
        catalog = _catalog(
            tmp_path,
            [product(1, "Elsa", set_id=1), product(2, "Elsa", set_id=2)],
            sets=sets,
        )
        assert (
            representative_printing(
                _only_group(catalog), catalog, LORCANA_SPEC
            ).product_id
            == 2
        )

    def test_an_empty_group_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            representative_printing(
                PrintingGroup("X", ()), _catalog(tmp_path, []), LORCANA_SPEC
            )


class TestBuildCardContent:
    def test_lean_content_in_order(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(
                    1,
                    "Mickey Mouse - Brave Little Tailor",
                    rarity="Legendary",
                    rules_text="<strong>Evasive</strong> <em>(Only Evasive.)</em>",
                    costInk="8",
                    inkwellIcononCard="No",
                    strength="5",
                    willpower="5",
                    loreValue="4",
                    classification="Dreamborn;Hero",
                    characterVersion="Brave Little Tailor",
                    property="Standard Characters",
                    releaseDate="2022-09-11T00:00:00Z",
                    detailNote="Presale item.",
                )
            ],
        )
        content = build_card_content(_only_group(catalog), catalog, LORCANA_SPEC)

        assert content == {
            "name": "Mickey Mouse - Brave Little Tailor",
            "type": "Character",
            "ink": "Ruby",
            "cost": 8,
            "inkwell": False,
            "strength": 5,
            "willpower": 5,
            "lore": 4,
            "classifications": ["Dreamborn", "Hero"],
            "rarity": "Legendary",
            "set": "The First Chapter",
            "text": "Evasive (Only Evasive.)",
        }
        assert list(content) == list(
            ["name", "type", "ink", "cost", "inkwell", "strength", "willpower"]
            + ["lore", "classifications", "rarity", "set", "text"]
        )

    def test_the_majority_outvotes_a_mistyped_printing(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(
                    1, "Vanellope", inkwellIcononCard="Yes", rules_text="Shift 4 *"
                ),
                product(
                    2, "Vanellope", inkwellIcononCard="No", rules_text="Shift 4 Ink"
                ),
                product(
                    3,
                    "Vanellope (Enchanted)",
                    inkwellIcononCard="Yes",
                    rules_text="Shift 4 Ink",
                ),
            ],
        )
        content = build_card_content(_only_group(catalog), catalog, LORCANA_SPEC)
        assert content["inkwell"] is True
        assert content["text"] == "Shift 4 Ink"

    def test_a_tie_goes_to_the_newest_printing(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(1, "Robin", rules_text="Old wording."),
                product(2, "Robin", rules_text="New wording."),
            ],
        )
        content = build_card_content(_only_group(catalog), catalog, LORCANA_SPEC)
        assert content["text"] == "New wording."

    def test_errata_printings_outvote_the_rest(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(1, "Bucky", rules_text="Old.", costInk="2"),
                product(2, "Bucky", rules_text="Old.", costInk="2"),
                product(3, "Bucky (Errata Version)", rules_text="New.", costInk="3"),
            ],
        )
        content = build_card_content(_only_group(catalog), catalog, LORCANA_SPEC)
        assert (content["text"], content["cost"]) == ("New.", 3)

    def test_missing_values_are_left_out(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(
                    1,
                    "Agrabah - Marketplace",
                    card_types=["Location"],
                    rules_text="<br>",
                    moveCost="1",
                )
            ],
        )
        content = build_card_content(_only_group(catalog), catalog, LORCANA_SPEC)
        assert content == {
            "name": "Agrabah - Marketplace",
            "type": "Location",
            "ink": "Ruby",
            "cost": 3,
            "move_cost": 1,
            "rarity": "Common",
            "set": "The First Chapter",
        }

    def test_several_card_types_stay_a_list(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path, [product(1, "Song", card_types=["Action", "Song"])]
        )
        content = build_card_content(_only_group(catalog), catalog, LORCANA_SPEC)
        assert content["type"] == ["Action", "Song"]


@pytest.mark.parametrize(
    "spec, number, expected",
    [
        # numbers as each catalog writes them (weekly-20260831)
        (DIGIMON_SPEC, "BT5-086 SR", "BT5-086"),
        (DIGIMON_SPEC, "ST9-05", "ST9-05"),
        (ONE_PIECE_SPEC, "OP01-024", "OP01-024"),
        (ONE_PIECE_SPEC, "P-001 ", "P-001"),
        (UNION_ARENA_SPEC, "UE01BT/BLC-1-041", "BLC-1-041"),
        (UNION_ARENA_SPEC, "UEPR/BLC-1-041 SR", "BLC-1-041"),
        (UNION_ARENA_SPEC, "AP01", None),
        (DIGIMON_SPEC, "", None),
    ],
)
def test_each_card_number_spec_reads_its_catalogs_numbers(
    spec: TcgjsonGameSpec, number: str, expected: str | None
) -> None:
    assert spec.card_number is not None
    match = spec.card_number.search(number)
    assert (match.group(0) if match else None) == expected


def test_a_name_ending_in_its_card_number_loses_it() -> None:
    assert base_name("Upamon - BT1-003", NAME_ENDING_IN_CARD_NUMBER) == "Upamon"
    assert base_name("Dot-Matrix - Eyes", NAME_ENDING_IN_CARD_NUMBER) == (
        "Dot-Matrix - Eyes"
    )


class TestIdentity:
    def test_case_and_subtitle_separator_do_not_split_a_card(
        self, tmp_path: Path
    ) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(1, "Jayce, Man of Progress"),
                product(2, "Jayce - Man of Progress"),
                product(3, "Jayce, man of  progress (Foil)"),
            ],
        )
        (group,) = group_printings(catalog, LORCANA_SPEC)
        # the newest wins the 1-1-1 tie among spellings
        assert group.name == "Jayce, man of progress"
        assert [p.product_id for p in group.products] == [1, 2, 3]

    def test_a_card_number_spec_groups_by_number_and_votes_the_name(
        self, tmp_path: Path
    ) -> None:
        spec = replace(
            LORCANA_SPEC,
            card_number=LEADING_CARD_NUMBER,
            name_suffix=NAME_ENDING_IN_CARD_NUMBER,
        )
        catalog = _catalog(
            tmp_path,
            [
                product(1, "Monkey.D.Luffy", number="OP01-024"),
                product(2, "Monkey.D.Luffy", number="OP05-119 SR"),
                product(3, "Monkey.D.Luffy - OP01-024 (Parallel)", number="OP01-024"),
                product(4, "Monkey.D.Luffy [Winner]", number="OP05-119"),
                product(5, "Roronoa Zoro"),
            ],
        )
        groups = group_printings(catalog, spec)

        assert [(g.name, [p.product_id for p in g.products]) for g in groups] == [
            ("Monkey.D.Luffy", [1, 3]),
            ("Monkey.D.Luffy", [2, 4]),
            ("Roronoa Zoro", [5]),
        ]
        # a name-keyed card has no numbers; the suffix "SR" is not one
        assert [g.card_numbers for g in groups] == [
            ("OP01-024",),
            ("OP05-119",),
            (),
        ]

    def test_card_numbers_list_each_full_printed_number_after_the_key(
        self, tmp_path: Path
    ) -> None:
        catalog = _catalog(
            tmp_path,
            [
                product(1, "Ichigo", number="UE01BT/BLC-1-041"),
                product(2, "Ichigo (R*)", number="UE01BT/BLC-1-041"),
                product(3, "Ichigo", number="UEPR/BLC-1-041"),
            ],
        )
        (group,) = group_printings(catalog, UNION_ARENA_SPEC)
        assert group.card_numbers == (
            "BLC-1-041",
            "UE01BT/BLC-1-041",
            "UEPR/BLC-1-041",
        )

    def test_a_type_attribute_spec_reads_the_type_and_card_ness_from_it(
        self, tmp_path: Path
    ) -> None:
        spec = replace(LORCANA_SPEC, type_attribute="cardTypeB")
        catalog = _catalog(
            tmp_path,
            [
                product(1, "Dark Magician", card_types=[], cardTypeB="Normal Monster"),
                product(2, "Dark Magician (UR)", cardTypeB="Normal Monster"),
                product(3, "Token", cardTypeB="None"),
            ],
        )
        (group,) = group_printings(catalog, spec)

        assert [p.product_id for p in group.products] == [1, 2]
        assert build_card_content(group, catalog, spec)["type"] == "Normal Monster"

    def test_excluded_types_are_not_cards(self, tmp_path: Path) -> None:
        spec = replace(LORCANA_SPEC, excluded_types=("DON!!",))
        catalog = _catalog(
            tmp_path,
            [
                product(1, "DON!! Card", card_types=["DON!!"]),
                product(2, "Odd Pair", card_types=["Character", "DON!!"]),
                product(3, "Nami"),
            ],
        )
        assert [g.name for g in group_printings(catalog, spec)] == ["Nami"]
