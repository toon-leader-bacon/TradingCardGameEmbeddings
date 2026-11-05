"""The run config's `teacher:` section -> TeacherViews.

    teacher:
      games: [mtg, flesh_and_blood]
      cooccurrence:                   # optional
        copy_count_rule: presence     # presence | copies | log_copies
        deck_size_rule: uniform       # uniform | inverse_size
      method: {kind: ppmi_svd, dim: 128, context_smoothing: 0.75}
      neighbours: {kind: top_k, k: 32}
        # or {kind: cosine_threshold, min_cosine: 0.4, max_neighbours: 64}
      min_deck_count: 5               # optional
      directory: data/teachers        # optional
      views:                          # optional; these are the defaults
        train: {deck_splits: [train], card_tiers: [train]}
        test: {deck_splits: [train, test], card_tiers: [train, test]}

One teacher per view: each student split deals pseudo-decks from its own
view's teacher (NeighbourDealer). `method` and `neighbours` are
`kind`-tagged (parse_by_kind): a new method or rule adds one parser to
its table below.
"""

from pathlib import Path
from dataclasses import replace
from typing import Callable

from src.schema.game_id import GameId
from src.schema.holdout import CardTier
from src.schema.splits import Split
from src.teacher.neighbour_rules import (
    CosineThresholdNeighbours,
    NeighbourRule,
    TopKNeighbours,
)
from src.teacher.teacher_settings import (
    DEFAULT_TEACHER_DIRECTORY,
    CooccurrenceOptions,
    CopyCountRule,
    DeckSizeRule,
    PpmiSvdOptions,
    TeacherMethodOptions,
    TeacherSettings,
    TeacherViews,
)
from src.training.config_parsing.config_section import (
    ConfigSection,
    choice_of,
    construct_at,
    parse_by_kind,
)

# One view: the deck splits counted and the card tiers kept
_View = tuple[frozenset[Split], frozenset[CardTier]]

# The views when the section names none: the student trains on a
# TRAIN-only teacher and is tested on TEST cards of a TRAIN + TEST teacher
_DEFAULT_VIEWS: dict[Split, _View] = {
    Split.TRAIN: (frozenset({Split.TRAIN}), frozenset({CardTier.TRAIN})),
    Split.TEST: (
        frozenset({Split.TRAIN, Split.TEST}),
        frozenset({CardTier.TRAIN, CardTier.TEST}),
    ),
}


def parse_teacher(section: ConfigSection) -> TeacherViews:
    """A `teacher:` section -> TeacherViews: the shared keys once, then
    one TeacherSettings per view (default: _DEFAULT_VIEWS). An absent
    optional key keeps TeacherSettings' default.

    Inputs: section ("teacher").
    Output: TeacherViews.
    Side effects: marks keys read.
    Exceptions: ValueError naming the location for a missing, unknown or
        wrongly typed key, an unknown game, split, tier or kind, a views
        section without train, or what the settings' own validation
        rejects.

    Example:
        >>> parse_teacher(top.required_section("teacher")).by_split[Split.TEST].card_tiers
        frozenset({<CardTier.TRAIN: 'train'>, <CardTier.TEST: 'test'>})
    """
    # The fit every view shares, at the default scope
    base = _parse_shared_settings(section)

    # Each view's deck splits and card tiers over that fit
    views = (
        _parse_views(section.required_section("views"))
        if section.has("views")
        else _DEFAULT_VIEWS
    )
    section.reject_unread_keys()
    by_split = {
        split: _view_settings(base, view, f"{section.where}.views.{split.value}")
        for split, view in views.items()
    }
    return construct_at(section.where, lambda: TeacherViews(by_split))


def _parse_shared_settings(section: ConfigSection) -> TeacherSettings:
    """The section's keys every view shares, as TeacherSettings at the
    default scope (each view replaces deck_splits and card_tiers).

    Private helper - single caller is parse_teacher.
    Inputs: section ("teacher"). Output: TeacherSettings.
    Side effects: marks the shared keys read.
    Exceptions: ValueError for a missing, unknown or wrongly typed key or
        what TeacherSettings rejects.
    """
    games = section.required_choice_list("games", GameId)
    method = parse_by_kind(section.required_section("method"), _METHOD_PARSERS)
    neighbours = parse_by_kind(
        section.required_section("neighbours"), _NEIGHBOUR_PARSERS
    )
    cooccurrence = _parse_cooccurrence(section.optional_section("cooccurrence"))
    min_deck_count = section.optional(
        "min_deck_count", int, TeacherSettings.min_deck_count
    )
    directory = Path(section.optional("directory", str, str(DEFAULT_TEACHER_DIRECTORY)))
    return construct_at(
        section.where,
        lambda: TeacherSettings(
            games,
            method,
            neighbours,
            cooccurrence=cooccurrence,
            min_deck_count=min_deck_count,
            directory=directory,
        ),
    )


def _view_settings(base: TeacherSettings, view: _View, where: str) -> TeacherSettings:
    """One view's TeacherSettings: base with the view's scope.

    Private helper - single caller is parse_teacher.
    Inputs: base (the shared fit), view (deck splits, card tiers), where
        (the view's location, for messages).
    Output: TeacherSettings. Side effects: none.
    Exceptions: ValueError naming where, from TeacherSettings' validation.
    """
    deck_splits, card_tiers = view
    return construct_at(
        where,
        lambda: replace(base, deck_splits=deck_splits, card_tiers=card_tiers),
    )


def _parse_views(section: ConfigSection) -> dict[Split, _View]:
    """`views: {<split>: {deck_splits: [...], card_tiers: [...]}}` -> each
    named split's (deck splits, card tiers).

    Private helper - single caller is parse_teacher.
    Inputs: section (the views subsection). Output: dict[Split, _View].
    Side effects: marks keys read.
    Exceptions: ValueError for an unknown split, a view missing a key, or
        an unknown split or tier inside one.
    """
    result: dict[Split, _View] = {}
    for key in section.keys():
        split = choice_of(key, Split, section.where)
        view = section.required_section(key)
        result[split] = (
            frozenset(view.required_choice_list("deck_splits", Split)),
            frozenset(view.required_choice_list("card_tiers", CardTier)),
        )
        view.reject_unread_keys()
    section.reject_unread_keys()
    return result


def _parse_cooccurrence(section: ConfigSection) -> CooccurrenceOptions:
    """`{[copy_count_rule], [deck_size_rule]}` -> CooccurrenceOptions
    (optional_choice per rule, defaults from CooccurrenceOptions).

    Private helper - single caller is _parse_shared_settings.
    Inputs: section (the cooccurrence subsection, empty when absent). Output: CooccurrenceOptions.
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
