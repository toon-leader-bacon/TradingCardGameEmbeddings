"""Which Flesh and Blood cards a hero may put in its deck, read off the
cardvault_fabtcg binder's "typebox" and "textbox" strings.

The binder has no class/talent keys: a typebox reads "<talents>
<classes> <type> - <subtypes>" ("Light Warrior Action - Attack"), see
../cardvault_fabtcg/card_mask_metrics.py. The rule, checked against the
published FaB deck box (only 242 of 129,845 deck card slots, 0.19%,
break it, mostly name-resolution noise in the box):

  - every talent on the card must be one of the hero's (an Elemental
    hero also has Earth, Ice and Lightning);
  - every class on the card must be one of the hero's, except a hybrid
    card ("Assassin / Ninja"), which needs any one of them;
  - a Revered card is illegal for a Reviled hero, and the reverse;
  - a "**X Specialization**" card needs X in the hero's name.

Generic cards name no class or talent, so they are legal for every hero.
"""

import re

from src.schema.card import GenericCard

_CLASSES = frozenset(
    {
        "Adjudicator",
        "Assassin",
        "Bard",
        "Brute",
        "Guardian",
        "Illusionist",
        "Mechanologist",
        "Merchant",
        "Necromancer",
        "Ninja",
        "Pirate",
        "Ranger",
        "Runeblade",
        "Shapeshifter",
        "Thief",
        "Warrior",
        "Wizard",
    }
)
_TALENTS = frozenset(
    {
        "Chaos",
        "Draconic",
        "Earth",
        "Elemental",
        "Ice",
        "Light",
        "Lightning",
        "Mystic",
        "Royal",
        "Shadow",
    }
)
_ELEMENTAL_TALENTS = frozenset({"Earth", "Ice", "Lightning"})
_REVERED = "Revered"
_REVILED = "Reviled"
_HERO_TYPE = "Hero"
# typebox words are space separated; split cards join faces with "||"
# and hybrids name two classes with " / "
_TYPEBOX_WORD_SEPARATOR = re.compile(r"[\s|/]+")
_SPECIALIZATION = re.compile(r"\*\*(?:Legendary )?([^*]+?) Specialization\*\*")


def is_hero(card: GenericCard) -> bool:
    """Whether card is a hero (its typebox, before " - ", names Hero).

    Inputs: card (GenericCard). Output: bool.
    Side effects: none. Exceptions: none.

    Example:
        >>> is_hero(dorinthea)  # typebox "Warrior Hero"
        True
    """
    return _HERO_TYPE in _typebox_words(card)


def is_legal_for_hero(card: GenericCard, hero: GenericCard) -> bool:
    """Whether hero may include card in its deck, per the module rule.

    Inputs:
        card: any Flesh and Blood card.
        hero: a card for which is_hero() holds.
    Output: bool.
    Side effects: none. Exceptions: none.

    Example:
        >>> is_legal_for_hero(lightning_instant, iyslander)  # Elemental
        True
    """
    hero_words = _hero_supertypes(hero)
    card_words = _typebox_words(card)
    if not (card_words & _TALENTS) <= hero_words:
        return False
    if not _classes_fit(card, card_words & _CLASSES, hero_words):
        return False
    if _REVERED in card_words and _REVILED in hero_words:
        return False
    if _REVILED in card_words and _REVERED in hero_words:
        return False
    return _specialization_fits(card, hero)


def _typebox_words(card: GenericCard) -> frozenset[str]:
    """The words of card's typebox before its " - " subtypes.

    Inputs: card. Output: frozenset[str], empty without a typebox.
    Side effects: none. Exceptions: none.
    """
    typebox = str(card.raw_content.get("typebox", ""))
    return frozenset(_TYPEBOX_WORD_SEPARATOR.split(typebox.split(" - ")[0])) - {""}


def _hero_supertypes(hero: GenericCard) -> frozenset[str]:
    """hero's classes and talents, plus Revered/Reviled and the talents
    Elemental grants.

    Inputs: hero. Output: frozenset[str].
    Side effects: none. Exceptions: none.
    """
    words = _typebox_words(hero)
    if "Elemental" in words:
        return words | _ELEMENTAL_TALENTS
    return words


def _classes_fit(
    card: GenericCard, card_classes: frozenset[str], hero_words: frozenset[str]
) -> bool:
    """Whether card's classes allow hero: all of them, or any one for a
    " / " hybrid; a classless card always fits.

    Inputs: card, card_classes (its typebox classes), hero_words
        (_hero_supertypes of the hero). Output: bool.
    Side effects: none. Exceptions: none.
    """
    if not card_classes:
        return True
    if " / " in str(card.raw_content.get("typebox", "")).split(" - ")[0]:
        return bool(card_classes & hero_words)
    return card_classes <= hero_words


def _specialization_fits(card: GenericCard, hero: GenericCard) -> bool:
    """Whether card is not a specialization, or is hero's.

    Inputs: card, hero. Output: bool.
    Side effects: none. Exceptions: none.
    """
    match = _SPECIALIZATION.search(str(card.raw_content.get("textbox", "")))
    if match is None:
        return True
    return re.search(rf"\b{re.escape(match.group(1))}\b", hero.name) is not None
