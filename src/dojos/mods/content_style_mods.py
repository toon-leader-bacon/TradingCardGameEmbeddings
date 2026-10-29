"""Style augmentations: mods that rewrite how a card's raw_content is
written (key casing, nesting, value types, letter case) while keeping
what it says. Aimed at transfer to unseen games, whose schemas differ
from ours in exactly these ways. Not every rewrite can be undone:
ValueCaseMod and KeyCaseMod drop letter case ("HPValue" -> "hp_value"),
ListToStringMod blurs "3" vs 3, and FlattenKeysMod blurs a key that
already holds the separator.

Each mod applies ONE style to the whole card, at every depth: a card
under KeyCaseMod gets every key in camel case (say), never a mix. The
style is picked per card from the mod's `styles` with the card's random
state, so `variants` and `probability` work as for every CardFieldMod
(card_field_mods.py). A mod with a single style has nothing to draw, so
its cards are deterministic whatever `variants` says: never stamped,
always cached. With probability < 1 a card then has two texts (the
original and the one restyle).

"styles" here means a mod's rewrite options; it is unrelated to the
contrastive pair styles (src/dojos/contrastive/styles/).

KeyCaseMod and FlattenKeysMod rename and move keys. Anything that finds
a field by name or path must run before them in the pipeline (see
ModPipeline). The value mods keep keys but change value types, so a
reader expecting an int under "power" must also run before them. Order
matters among these mods too: KeyCaseMod after FlattenKeysMod leaves
joined keys alone when the separator is not "_", "-" or a space (those
it re-cases), and ListToStringMod writes different text before and
after the bool/number mods.

A card a mod cannot rewrite (two keys collide after renaming, odd data)
passes through unchanged and is counted in the tally, as for every
CardFieldMod.
"""

import dataclasses
import json
import random
import re
from abc import abstractmethod
from enum import StrEnum
from typing import Callable, ClassVar, Generic, TypeVar

from src.dojos.mods.card_field_mods import UNBOUNDED_VARIANTS, CardFieldMod
from src.dojos.mods.content_rewrite import (
    ContentRewrite,
    flattened_content,
    rewritten_content,
)
from src.dojos.mods.mod import MASK_TOKEN
from src.schema.card import GenericCard
from src.schema.json_types import JsonObject, JsonScalar, JsonValue

StyleT = TypeVar("StyleT")


class KeyCase(StrEnum):
    """How KeyCaseMod writes every key, shown for "mana cost"."""

    SNAKE = "snake"  # mana_cost
    UPPER_SNAKE = "upper_snake"  # MANA_COST
    CAMEL = "camel"  # manaCost
    PASCAL = "pascal"  # ManaCost
    KEBAB = "kebab"  # mana-cost
    TITLE = "title"  # Mana Cost


class ValueCase(StrEnum):
    """How ValueCaseMod writes every string value."""

    UPPER = "upper"
    LOWER = "lower"


class BoolSpelling(StrEnum):
    """How BoolToStringMod writes True / False."""

    LOWER = "lower"  # "true" / "false"
    TITLE = "title"  # "True" / "False"
    YES_NO = "yes_no"  # "yes" / "no"


class ContentRewriteMod(CardFieldMod):
    """Template Method base for mods that rebuild raw_content as a whole:
    modified_card is shared, subclasses implement new_raw_content.
    Constructor: as CardFieldMod.
    """

    def modified_card(self, card: GenericCard, rng: random.Random) -> GenericCard:
        """A new card with raw_content replaced by new_raw_content, or
        card itself when the rewrite changes nothing.

        Inputs: card, rng. Output: GenericCard. Side effects: may advance rng.
        Exceptions: as new_raw_content (caught by the best-effort guard).

        Example:
            >>> NumberToStringMod().modified_card(card, rng).raw_content
            {'name': 'Bolt', 'power': '3'}
        """
        content = self.new_raw_content(card.raw_content, rng)
        if content == card.raw_content:
            return card
        return dataclasses.replace(card, raw_content=content)

    @abstractmethod
    def new_raw_content(self, content: JsonObject, rng: random.Random) -> JsonObject:
        """A new raw_content; content is never changed.

        Inputs: content, rng (the only random state to draw from).
        Output: JsonObject. Side effects: may advance rng.
        Exceptions: any (caught by the caller).
        """
        ...


class StyleChoiceMod(ContentRewriteMod, Generic[StyleT]):
    """A ContentRewriteMod that picks one of its styles per card and
    applies it to the whole card. Subclasses set STYLE_TYPE (every style
    must be an instance of it), DEFAULT_STYLES (the one place a mod's
    default lives; specs and config pass None to use it) and implement
    restyled_content.
    """

    # Bare tuple: a ClassVar cannot use StyleT; each subclass annotates
    # its own concrete tuple[..., ...]
    DEFAULT_STYLES: ClassVar[tuple]
    STYLE_TYPE: ClassVar[type]

    def __init__(
        self,
        styles: tuple[StyleT, ...] | None = None,
        train_only: bool = True,
        rng_seed: int | None = None,
        variants: int = UNBOUNDED_VARIANTS,
        probability: float = 1.0,
    ) -> None:
        """Inputs: styles (non-empty, no repeats; one is drawn per card;
        None: DEFAULT_STYLES), train_only, rng_seed, variants, probability
        (see CardFieldMod).
        Output: none. Side effects: none.
        Exceptions: TypeError if styles is not a tuple or holds a
            non-STYLE_TYPE; ValueError for an empty or repeating styles, or
            as CardFieldMod."""
        super().__init__(
            train_only=train_only,
            rng_seed=rng_seed,
            variants=variants,
            probability=probability,
        )
        self.styles: tuple[StyleT, ...] = (
            type(self).DEFAULT_STYLES if styles is None else styles
        )
        _check_styles(self.styles, type(self).STYLE_TYPE)

    @property
    def deterministic_cards(self) -> bool:
        """True when variants bounds the texts, or there is one style (so
        nothing is drawn). Inputs: none. Output: bool. Side effects: none.
        Exceptions: none."""
        return super().deterministic_cards or len(self.styles) == 1

    def new_raw_content(self, content: JsonObject, rng: random.Random) -> JsonObject:
        """content restyled with one style drawn from rng (no draw when
        there is only one).

        Inputs: content, rng. Output: JsonObject.
        Side effects: advances rng when there are several styles.
        Exceptions: as restyled_content.

        Example:
            >>> KeyCaseMod((KeyCase.CAMEL,)).new_raw_content({"mana_cost": 1}, rng)
            {'manaCost': 1}
        """
        style = self.styles[0] if len(self.styles) == 1 else rng.choice(self.styles)
        return self.restyled_content(content, style)

    @abstractmethod
    def restyled_content(self, content: JsonObject, style: StyleT) -> JsonObject:
        """content with style applied throughout.

        Inputs: content, style. Output: JsonObject. Side effects: none.
        Exceptions: any (caught by the best-effort guard).
        """
        ...


class SeparatorChoiceMod(StyleChoiceMod[str]):
    """A StyleChoiceMod whose styles are separator strings; none may be
    empty (an empty one would glue words or path steps together).
    Constructor: as StyleChoiceMod; Exceptions: also ValueError for an
    empty separator.
    """

    STYLE_TYPE: ClassVar[type] = str

    def __init__(
        self,
        styles: tuple[str, ...] | None = None,
        train_only: bool = True,
        rng_seed: int | None = None,
        variants: int = UNBOUNDED_VARIANTS,
        probability: float = 1.0,
    ) -> None:
        """See the class docstring."""
        super().__init__(styles, train_only, rng_seed, variants, probability)
        # Validate inputs: every separator non-empty
        if any(not separator for separator in self.styles):
            raise ValueError(f"{type(self).__name__} separators must be non-empty")


class KeyCaseMod(StyleChoiceMod[KeyCase]):
    """Rewrite every key, at every depth, in one KeyCase. A key is split
    into words at "_", "-", spaces and lower-to-upper case changes
    ("manaCost" -> mana, cost). A key holding any other character ("{R}",
    "stats.hp") is left as it is, so no character is ever lost. Keys that
    are really data (energy types as keys, "Fire") are re-cased too. Keys
    colliding after the rewrite pass the card through unchanged.
    DEFAULT_STYLES: every KeyCase. Constructor: as StyleChoiceMod.
    """

    STYLE_TYPE: ClassVar[type] = KeyCase
    DEFAULT_STYLES: ClassVar[tuple[KeyCase, ...]] = tuple(KeyCase)

    def restyled_content(self, content: JsonObject, style: KeyCase) -> JsonObject:
        """See StyleChoiceMod.

        Example:
            >>> KeyCaseMod().restyled_content({"mana_cost": {"red": 1}}, KeyCase.PASCAL)
            {'ManaCost': {'Red': 1}}
        """
        return rewritten_content(content, _KeyCaseRewrite(style))


class FlattenKeysMod(SeparatorChoiceMod):
    """Move every nested object up to the top level, joining its path with
    one separator ({"stats": {"hp": 60}} -> {"stats.hp": 60}); see
    content_rewrite.flattened_content for lists and empty objects.
    DEFAULT_STYLES: (".",). Constructor: as SeparatorChoiceMod.
    """

    DEFAULT_STYLES: ClassVar[tuple[str, ...]] = (".",)

    def restyled_content(self, content: JsonObject, style: str) -> JsonObject:
        """See StyleChoiceMod.

        Example:
            >>> FlattenKeysMod().restyled_content({"stats": {"hp": 60}}, "/")
            {'stats/hp': 60}
        """
        return flattened_content(content, style)


class ListToStringMod(SeparatorChoiceMod):
    """Join every list of scalars, at every depth, into one string with a
    separator (["R", "G"] -> "R, G"). A list holding an object or a list
    is kept and descended into. A non-string item is written as JSON
    (true, null, 3), so ["3"] and [3] both become "3", and an item holding
    the separator is ambiguous. An empty list is kept.
    DEFAULT_STYLES: (", ",). Constructor: as SeparatorChoiceMod.
    """

    DEFAULT_STYLES: ClassVar[tuple[str, ...]] = (", ",)

    def restyled_content(self, content: JsonObject, style: str) -> JsonObject:
        """See StyleChoiceMod.

        Example:
            >>> ListToStringMod().restyled_content({"colors": ["R", "G"]}, " / ")
            {'colors': 'R / G'}
        """
        return rewritten_content(content, _ListJoinRewrite(style))


class ValueCaseMod(StyleChoiceMod[ValueCase]):
    """Upper- or lower-case every string value at every depth (keys are
    untouched; KeyCaseMod does those). MASK_TOKEN is left as it is, so a
    mask written by an earlier mod stays the one mask string.
    DEFAULT_STYLES: every ValueCase. Constructor: as StyleChoiceMod.
    """

    STYLE_TYPE: ClassVar[type] = ValueCase
    DEFAULT_STYLES: ClassVar[tuple[ValueCase, ...]] = tuple(ValueCase)

    def restyled_content(self, content: JsonObject, style: ValueCase) -> JsonObject:
        """See StyleChoiceMod.

        Example:
            >>> ValueCaseMod().restyled_content({"name": "Bolt"}, ValueCase.UPPER)
            {'name': 'BOLT'}
        """
        return rewritten_content(content, _ValueCaseRewrite(style))


class BoolToStringMod(StyleChoiceMod[BoolSpelling]):
    """Write every bool value, at every depth, as a string in one
    BoolSpelling. DEFAULT_STYLES: (BoolSpelling.LOWER,).
    Constructor: as StyleChoiceMod.
    """

    STYLE_TYPE: ClassVar[type] = BoolSpelling
    DEFAULT_STYLES: ClassVar[tuple[BoolSpelling, ...]] = (BoolSpelling.LOWER,)

    def restyled_content(self, content: JsonObject, style: BoolSpelling) -> JsonObject:
        """See StyleChoiceMod.

        Example:
            >>> BoolToStringMod().restyled_content({"legal": True}, BoolSpelling.YES_NO)
            {'legal': 'yes'}
        """
        return rewritten_content(content, _BoolSpellingRewrite(style))


class NumberToStringMod(ContentRewriteMod):
    """Write every int and float value, at every depth, as a string
    (3 -> "3"); bools are not numbers here. One fixed spelling, so its
    cards are always deterministic. Constructor: as CardFieldMod.
    """

    @property
    def deterministic_cards(self) -> bool:
        """Always True: nothing is drawn. Inputs: none. Output: bool.
        Side effects: none. Exceptions: none."""
        return True

    def new_raw_content(self, content: JsonObject, rng: random.Random) -> JsonObject:
        """See ContentRewriteMod; rng is unused.

        Example:
            >>> NumberToStringMod().new_raw_content({"hp": 60}, rng)
            {'hp': '60'}
        """
        return rewritten_content(content, _NumberSpellingRewrite())


class _KeyCaseRewrite(ContentRewrite):
    """Renames every key into one KeyCase (see KeyCaseMod)."""

    def __init__(self, case: KeyCase) -> None:
        """Inputs: case. Side effects: none. Exceptions: none."""
        self.case = case

    def key(self, key: str) -> str:
        """key split into words and joined in self.case; key itself when
        _key_words finds no words in it.
        Inputs: key. Output: str. Side effects: none. Exceptions: none."""
        words = _key_words(key)
        if words is None:
            return key
        return _KEY_CASE_JOINERS[self.case](words)


class _ValueCaseRewrite(ContentRewrite):
    """Re-cases every string leaf except MASK_TOKEN (see ValueCaseMod)."""

    def __init__(self, case: ValueCase) -> None:
        """Inputs: case. Side effects: none. Exceptions: none."""
        self.case = case

    def leaf(self, value: JsonScalar) -> JsonValue:
        """See ContentRewrite. Inputs: value. Output: JsonValue.
        Side effects: none. Exceptions: none."""
        if not isinstance(value, str) or value == MASK_TOKEN:
            return value
        return _VALUE_CASE_FUNCTIONS[self.case](value)


class _BoolSpellingRewrite(ContentRewrite):
    """Writes every bool leaf in one BoolSpelling (see BoolToStringMod)."""

    def __init__(self, spelling: BoolSpelling) -> None:
        """Inputs: spelling. Side effects: none. Exceptions: none."""
        self.spelling = spelling

    def leaf(self, value: JsonScalar) -> JsonValue:
        """See ContentRewrite. Inputs: value. Output: JsonValue.
        Side effects: none. Exceptions: none."""
        if not isinstance(value, bool):
            return value
        true_text, false_text = _BOOL_SPELLINGS[self.spelling]
        return true_text if value else false_text


class _ListJoinRewrite(ContentRewrite):
    """Joins every list of scalars with one separator (see ListToStringMod)."""

    def __init__(self, separator: str) -> None:
        """Inputs: separator. Side effects: none. Exceptions: none."""
        self.separator = separator

    def list_replacement(self, items: list[JsonValue]) -> JsonValue | None:
        """See ContentRewrite. Inputs: items. Output: str, or None for an
        empty list or one holding a container.
        Side effects: none. Exceptions: none."""
        if not items or any(isinstance(item, (dict, list)) for item in items):
            return None
        return self.separator.join(
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in items
        )


class _NumberSpellingRewrite(ContentRewrite):
    """Writes every int and float leaf (not bool) as str (see NumberToStringMod)."""

    def leaf(self, value: JsonScalar) -> JsonValue:
        """See ContentRewrite. Inputs: value. Output: JsonValue.
        Side effects: none. Exceptions: none."""
        # bool subclasses int, and is BoolToStringMod's to spell
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        return str(value)


def _check_styles(styles: tuple[object, ...], style_type: type) -> None:
    """Reject a styles tuple the mod does not accept, so a bad style fails
    at construction rather than on every card.

    Inputs: styles, style_type (the mod's STYLE_TYPE).
    Output: none. Side effects: none.
    Exceptions: TypeError if styles is not a tuple (a bare str would pass
        as one style per character) or holds a non-style_type (a plain
        "camel" is not a KeyCase); ValueError if it is empty or repeats a
        style.
    """
    if not isinstance(styles, tuple):
        raise TypeError(f"styles must be a tuple, got {type(styles).__name__}")
    if not styles:
        raise ValueError("styles must not be empty")
    for style in styles:
        if not isinstance(style, style_type):
            raise TypeError(f"style {style!r} is not a {style_type.__name__}")
    if len(set(styles)) != len(styles):
        raise ValueError(f"styles must not repeat, got {styles}")


def _key_words(key: str) -> list[str] | None:
    """key split into lower-case words at "_", "-", spaces, lower-to-upper
    case changes, letter-digit changes and the end of an acronym
    ("HPValue" -> hp, value), or None when key holds any other character
    (or has no words), so the caller keeps it as it is.

    Inputs: key. Output: list[str] or None.
    Side effects: none. Exceptions: none.

    Example:
        >>> _key_words("manaCost_total")
        ['mana', 'cost', 'total']
    """
    if not _KEY_CHARACTERS.fullmatch(key):
        return None
    words = [word.lower() for word in _KEY_WORD.findall(key)]
    return words or None


# A key KeyCaseMod may rewrite: only word characters and word separators
_KEY_CHARACTERS = re.compile(r"[A-Za-z0-9_\- ]+")
# One word: an acronym (not swallowing the capital that starts the next
# word), a capitalized or lower-case word, or a run of digits
_KEY_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def _capitalized(words: list[str]) -> list[str]:
    """Each word with its first letter upper-cased.
    Inputs: words (lower case). Output: list[str]. Side effects: none.
    Exceptions: none."""
    return [word[:1].upper() + word[1:] for word in words]


# Each KeyCase's way of joining lower-case words (a table, not an if-chain;
# a test checks its keys are exactly set(KeyCase))
_KEY_CASE_JOINERS: dict[KeyCase, Callable[[list[str]], str]] = {
    KeyCase.SNAKE: "_".join,
    KeyCase.UPPER_SNAKE: lambda words: "_".join(words).upper(),
    KeyCase.CAMEL: lambda words: words[0] + "".join(_capitalized(words[1:])),
    KeyCase.PASCAL: lambda words: "".join(_capitalized(words)),
    KeyCase.KEBAB: "-".join,
    KeyCase.TITLE: lambda words: " ".join(_capitalized(words)),
}

_VALUE_CASE_FUNCTIONS: dict[ValueCase, Callable[[str], str]] = {
    ValueCase.UPPER: str.upper,
    ValueCase.LOWER: str.lower,
}

# (true text, false text) per spelling
_BOOL_SPELLINGS: dict[BoolSpelling, tuple[str, str]] = {
    BoolSpelling.LOWER: ("true", "false"),
    BoolSpelling.TITLE: ("True", "False"),
    BoolSpelling.YES_NO: ("yes", "no"),
}
