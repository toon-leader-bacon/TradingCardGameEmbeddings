"""The run config's `teacher:` section -> TeacherSettings.

    teacher:
      games: [mtg, flesh_and_blood]
      deck_splits: [train]            # optional, default [train]
      card_tiers: [train]             # optional, default [train]
      cooccurrence:                   # optional
        copy_count_rule: presence     # presence | copies | log_copies
        deck_size_rule: uniform       # uniform | inverse_size
      method: {kind: ppmi_svd, dim: 128, context_smoothing: 0.75}
      neighbours: {kind: top_k, k: 32}
        # or {kind: cosine_threshold, min_cosine: 0.4, max_neighbours: 64}
      min_deck_count: 5               # optional
      directory: data/teachers        # optional

`method` and `neighbours` are `kind`-tagged (parse_by_kind): a new method
or rule adds one parser to its table below.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.schema.game_id import GameId
from src.schema.holdout import CardTier
from src.schema.splits import Split
from src.teacher.neighbour_rules import (
    CosineThresholdNeighbours,
    NeighbourRule,
    TopKNeighbours,
)
from src.teacher.teacher_settings import (
    CooccurrenceOptions,
    CopyCountRule,
    DeckSizeRule,
    PpmiSvdOptions,
    TeacherMethodOptions,
    TeacherSettings,
)
from src.training.config_parsing.config_section import (
    ConfigSection,
    construct_at,
    parse_by_kind,
)

_EnumT = TypeVar("_EnumT", bound=StrEnum)


def parse_teacher(section: ConfigSection) -> TeacherSettings:
    """A `teacher:` section -> TeacherSettings; an absent optional key
    keeps TeacherSettings' default (only present keys are passed on).

    Inputs: section ("teacher").
    Output: TeacherSettings.
    Side effects: none.
    Exceptions: ValueError naming the location for a missing, unknown or
        wrongly typed key, an unknown game, split, tier or kind, or what
        the settings' own validation rejects.

    Example:
        >>> parse_teacher(top.required_section("teacher")).method.dim
        128
    """
    # The required keys
    options: dict[str, Any] = {
        "games": _enum_values(section, "games", GameId),
        "method": parse_by_kind(section.required_section("method"), _METHOD_PARSERS),
        "neighbours": parse_by_kind(
            section.required_section("neighbours"), _NEIGHBOUR_PARSERS
        ),
    }

    # The optional keys, only when present
    if section.has("deck_splits"):
        options["deck_splits"] = frozenset(_enum_values(section, "deck_splits", Split))
    if section.has("card_tiers"):
        options["card_tiers"] = frozenset(_enum_values(section, "card_tiers", CardTier))
    if section.has("cooccurrence"):
        options["cooccurrence"] = _parse_cooccurrence(
            section.required_section("cooccurrence")
        )
    if section.has("min_deck_count"):
        options["min_deck_count"] = section.required("min_deck_count", int)
    if section.has("directory"):
        options["directory"] = Path(section.required("directory", str))
    section.reject_unread_keys()
    return construct_at(section.where, lambda: TeacherSettings(**options))


def _enum_values(
    section: ConfigSection, key: str, enum_type: type[_EnumT]
) -> tuple[_EnumT, ...]:
    """section[key], a list of strings, as enum_type members in order.

    Private helper - callers are parse_teacher (games, splits, tiers).
    Inputs: section, key, enum_type (a StrEnum). Output: tuple of members.
    Side effects: marks key read.
    Exceptions: ValueError naming the location and the valid values for a
        string no member has.
    """
    location = section.location_of(key)
    result: list[_EnumT] = []
    for text in section.required_str_list(key):
        try:
            result.append(enum_type(text))
        except ValueError:
            wanted = [member.value for member in enum_type]
            raise ValueError(f"{location}: {text!r} is not one of {wanted}") from None
    return tuple(result)


def _parse_cooccurrence(section: ConfigSection) -> CooccurrenceOptions:
    """`{[copy_count_rule], [deck_size_rule]}` -> CooccurrenceOptions
    (optional_choice per rule, defaults from CooccurrenceOptions).

    Private helper - single caller is parse_teacher.
    Inputs: section (the cooccurrence subsection). Output: CooccurrenceOptions.
    Side effects: marks keys read. Exceptions: ValueError for an unknown
        rule or key.
    """
    defaults = CooccurrenceOptions()
    copy_count_rule = section.optional_choice(
        "copy_count_rule", CopyCountRule, defaults.copy_count_rule
    )
    deck_size_rule = section.optional_choice(
        "deck_size_rule", DeckSizeRule, defaults.deck_size_rule
    )
    section.reject_unread_keys()
    return CooccurrenceOptions(copy_count_rule, deck_size_rule)


def _parse_ppmi_svd(section: ConfigSection) -> PpmiSvdOptions:
    """`{kind: ppmi_svd, dim, [context_smoothing]}` -> PpmiSvdOptions.

    Private helper - single caller is the _METHOD_PARSERS table.
    Inputs: section. Output: PpmiSvdOptions. Side effects: marks keys read.
    Exceptions: ValueError for a wrongly typed value.
    """
    return PpmiSvdOptions(
        dim=section.required("dim", int),
        context_smoothing=section.optional(
            "context_smoothing", float, PpmiSvdOptions.context_smoothing
        ),
    )


def _parse_top_k(section: ConfigSection) -> TopKNeighbours:
    """`{kind: top_k, k}` -> TopKNeighbours.

    Private helper - single caller is the _NEIGHBOUR_PARSERS table.
    Inputs: section. Output: TopKNeighbours. Side effects: marks keys read.
    Exceptions: ValueError for a wrongly typed value.
    """
    return TopKNeighbours(k=section.required("k", int))


def _parse_cosine_threshold(section: ConfigSection) -> CosineThresholdNeighbours:
    """`{kind: cosine_threshold, min_cosine, max_neighbours}` ->
    CosineThresholdNeighbours.

    Private helper - single caller is the _NEIGHBOUR_PARSERS table.
    Inputs: section. Output: CosineThresholdNeighbours.
    Side effects: marks keys read. Exceptions: ValueError for a wrongly
        typed value.
    """
    return CosineThresholdNeighbours(
        min_cosine=section.required("min_cosine", float),
        max_neighbours=section.required("max_neighbours", int),
    )


# Each teacher method's parser, by its config kind
_METHOD_PARSERS: dict[str, Callable[[ConfigSection], TeacherMethodOptions]] = {
    PpmiSvdOptions.KIND: _parse_ppmi_svd,
}

# Each neighbour rule's parser, by its config kind
_NEIGHBOUR_PARSERS: dict[str, Callable[[ConfigSection], NeighbourRule]] = {
    TopKNeighbours.KIND: _parse_top_k,
    CosineThresholdNeighbours.KIND: _parse_cosine_threshold,
}
