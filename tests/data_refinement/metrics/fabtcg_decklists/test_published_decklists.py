from pathlib import Path

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.fabtcg_decklists.published_decklists import (
    PublishedDecklists,
    count_decks_per_hero,
)
from src.schema.game_id import GameId
from tests.data_refinement.metrics.fabtcg_decklists.fab_fixtures import (
    fab_binder,
    fab_card,
    published_box,
)

HERO = fab_card("Dorinthea Ironsong", "Warrior Hero")
OTHER_HERO = fab_card("Katsu, the Wanderer", "Ninja Hero")
ATTACK = fab_card("Sharpen Steel", "Warrior Action")
BINDER = fab_binder([HERO, OTHER_HERO, ATTACK])


class TestPublishedDecklists:
    def test_indexes_a_single_hero_deck_by_slug(self) -> None:
        box = published_box({"slug-a": [HERO, ATTACK, ATTACK]})

        decklist = PublishedDecklists(box, BINDER).decklist_for_slug("slug-a")

        assert decklist is not None
        assert decklist.hero == HERO
        assert decklist.card_uuids == frozenset({ATTACK.nocab_uuid})
        assert decklist.deck_uuid == next(box.all_uuids())

    def test_decks_with_no_or_two_heroes_have_no_decklist(self) -> None:
        box = published_box({"none": [ATTACK], "two": [HERO, OTHER_HERO, ATTACK]})

        decklists = PublishedDecklists(box, BINDER)

        assert decklists.decklist_for_slug("none") is None
        assert decklists.decklist_for_slug("two") is None
        assert list(decklists.all_decklists()) == []

    def test_unknown_slug_has_no_decklist(self) -> None:
        box = published_box({"slug-a": [HERO]})

        assert PublishedDecklists(box, BINDER).decklist_for_slug("missing") is None

    def test_reading_never_writes_the_box_file(self, tmp_path: Path) -> None:
        box_path = tmp_path / "flesh_and_blood.db"
        published_box({"slug-a": [HERO, ATTACK]}).save(
            box_path, GameId.FLESH_AND_BLOOD, "v1"
        )
        before = box_path.read_bytes()

        box = DeckBox.load([box_path])
        PublishedDecklists(box, BINDER)
        box.flush()

        assert box_path.read_bytes() == before
        assert box.card_binder_version_for(GameId.FLESH_AND_BLOOD) == "v1"


class TestCountDecksPerHero:
    def test_counts_by_hero_name(self) -> None:
        box = published_box({"a": [HERO], "b": [HERO, ATTACK], "c": [OTHER_HERO]})

        counts = count_decks_per_hero(PublishedDecklists(box, BINDER).all_decklists())

        assert counts == {HERO.name: 2, OTHER_HERO.name: 1}
