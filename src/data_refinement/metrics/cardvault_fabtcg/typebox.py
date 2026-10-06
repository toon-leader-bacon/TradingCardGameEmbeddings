"""Shared parsing of the cardvault_fabtcg binder's "typebox" string.

A Flesh and Blood typebox reads "<talents> <classes> <type> -
<subtypes>" (e.g. "Light Warrior Action - Attack"): talent(s), class(es)
and card type are all space-separated words before " - ", subtypes
after. Both ../cardvault_fabtcg/card_mask_metrics.py (masking the whole
typebox to predict class/card type) and
../fabtcg_decklists/hero_legality.py (reading a card's talents/classes
to check hero legality) parse this same string the same way - this
module holds that shared primitive. The class/talent/type vocabularies
each file applies on top are not shared: they differ in purpose and
membership (e.g. hero_legality has no use for "Generic").
"""

import re

# typebox words are space separated; split cards join faces with "||"
# and hybrids name two classes with " / "
TYPEBOX_WORD_SEPARATOR = re.compile(r"[\s|/]+")


def typebox_head(typebox: str) -> str:
    """The typebox before its " - " subtypes: talent, class and type.

    Inputs: typebox (str). Output: str. Side effects: none.
    Exceptions: none.

    Example:
        >>> typebox_head("Light Warrior Action - Attack")
        'Light Warrior Action'
    """
    return typebox.split(" - ")[0]


def typebox_words(typebox: str) -> frozenset[str]:
    """The words of typebox's head (see typebox_head()).

    Inputs: typebox (str). Output: frozenset[str], empty for a blank
        typebox. Side effects: none. Exceptions: none.

    Example:
        >>> typebox_words("Light Warrior Action - Attack")
        frozenset({'Light', 'Warrior', 'Action'})
    """
    return frozenset(TYPEBOX_WORD_SEPARATOR.split(typebox_head(typebox))) - {""}
