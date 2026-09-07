from pathlib import Path

import pytest

from src.data_refinement.metrics.legacy.fabtcg_decklists.fragment_parsing import (
    iter_card_quantities_and_names,
)

_REAL_FRAGMENT_PATH = Path("data/tmp/list_view.html")


def _fragment(*groups: tuple[str, list[str]]) -> str:
    """Build a synthetic decklist fragment: groups is
    (header_text, [card_name_div_html, ...]) pairs."""
    group_html = "".join(
        f'<div class="list-view-container">'
        f"<h3>{header}</h3>"
        f'<ul class="cards-container">{"".join(items)}</ul>'
        f"</div>"
        for header, items in groups
    )
    return f'<section class="decklist-list-view block hidden">{group_html}</section>'


def _card_item(quantity_and_name: str) -> str:
    return (
        '<li class="card-item group">'
        f'<div class="card-name"><span>{quantity_and_name}</span></div>'
        "</li>"
    )


class TestIterCardQuantitiesAndNames:
    def test_walks_hero_and_pitch_groups_and_preserves_quantities(self) -> None:
        fragment = _fragment(
            ("Hero / Weapon / Equipment", [_card_item("1x Dorinthea Ironsong")]),
            ("Pitch 1", [_card_item("3x Enlightened Strike")]),
        )

        pairs = list(iter_card_quantities_and_names(fragment))

        assert pairs == [(1, "Dorinthea Ironsong"), (3, "Enlightened Strike")]

    def test_single_digit_quantity(self) -> None:
        fragment = _fragment(("Hero", [_card_item("1x Dorinthea Ironsong")]))
        assert list(iter_card_quantities_and_names(fragment)) == [
            (1, "Dorinthea Ironsong")
        ]

    def test_multi_digit_quantity(self) -> None:
        fragment = _fragment(("Hero", [_card_item("12x Some Card")]))
        assert list(iter_card_quantities_and_names(fragment)) == [(12, "Some Card")]

    def test_raises_on_non_matching_text(self) -> None:
        fragment = _fragment(("Hero", [_card_item("no quantity prefix here")]))
        with pytest.raises(ValueError):
            list(iter_card_quantities_and_names(fragment))

    def test_raises_when_card_item_missing_name_div(self) -> None:
        fragment = (
            '<section class="decklist-list-view block hidden">'
            '<div class="list-view-container"><h3>Hero</h3>'
            '<ul class="cards-container">'
            '<li class="card-item group"><div class="card-image"></div></li>'
            "</ul></div></section>"
        )
        with pytest.raises(ValueError):
            list(iter_card_quantities_and_names(fragment))

    def test_empty_fragment_yields_nothing(self) -> None:
        assert list(iter_card_quantities_and_names(_fragment())) == []

    def test_against_real_fixture(self) -> None:
        if not _REAL_FRAGMENT_PATH.exists():
            pytest.skip(f"{_REAL_FRAGMENT_PATH} fixture not present")

        fragment_html = _REAL_FRAGMENT_PATH.read_text(encoding="utf-8")
        pairs = list(iter_card_quantities_and_names(fragment_html))

        names = [name for _, name in pairs]
        assert "Dorinthea Ironsong" in names
        assert any(name.startswith("Enlightened Strike") for name in names)
