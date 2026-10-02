import pytest

from src.data_refinement.metrics.fabtcg_decklists.hero_legality import (
    is_hero,
    is_legal_for_hero,
)
from src.schema.card import GenericCard
from tests.data_refinement.metrics.fabtcg_decklists.fab_fixtures import fab_card

DORINTHEA = fab_card("Dorinthea Ironsong", "Warrior Hero")
BRIAR = fab_card("Briar, Warden of Thorns", "Elemental Runeblade Hero")
ARAKNI = fab_card("Arakni, Marionette", "Chaos Assassin Hero")
KAYO = fab_card("Kayo, Underhanded Cheat", "Reviled Brute Hero")
BOLTYN = fab_card("Ser Boltyn, Breaker of Dawn // Ser Boltyn", "Light Warrior Hero")


class TestIsHero:
    def test_hero_typebox(self) -> None:
        assert is_hero(DORINTHEA)

    def test_demi_hero_and_actions_are_not_heroes(self) -> None:
        assert not is_hero(fab_card("Demi", "Demi-Hero"))
        assert not is_hero(fab_card("Heroic Pose", "Revered Action"))

    def test_card_without_typebox_is_not_a_hero(self) -> None:
        card = fab_card("x", "")
        assert not is_hero(card)


class TestIsLegalForHero:
    @pytest.mark.parametrize(
        "typebox, hero",
        [
            ("Generic Action - Attack", DORINTHEA),
            ("Warrior Action - Attack", DORINTHEA),
            ("Lightning Instant", BRIAR),  # Elemental grants Lightning
            ("Earth Runeblade Action", BRIAR),
            ("Assassin / Ninja Action", ARAKNI),  # a hybrid needs one class
            ("Revered Action", DORINTHEA),
            ("Reviled Action", KAYO),
        ],
    )
    def test_legal(self, typebox: str, hero: GenericCard) -> None:
        assert is_legal_for_hero(fab_card("c", typebox), hero)

    @pytest.mark.parametrize(
        "typebox, hero",
        [
            ("Ninja Action", DORINTHEA),
            ("Light Warrior Action", DORINTHEA),  # missing talent
            ("Pirate Necromancer Action", fab_card("P", "Pirate Hero")),
            ("Revered Action", KAYO),
            ("Shadow Runeblade Action", BRIAR),
        ],
    )
    def test_illegal(self, typebox: str, hero: GenericCard) -> None:
        assert not is_legal_for_hero(fab_card("c", typebox), hero)

    def test_specialization_needs_the_named_hero(self) -> None:
        card = fab_card("V", "Light Warrior Action", "**Boltyn Specialization**")
        other_warrior = fab_card("Victor", "Light Warrior Hero")
        assert is_legal_for_hero(card, BOLTYN)
        assert not is_legal_for_hero(card, other_warrior)

    def test_legendary_specialization(self) -> None:
        card = fab_card("S", "Warrior Action", "**Legendary Dorinthea Specialization**")
        assert is_legal_for_hero(card, DORINTHEA)
