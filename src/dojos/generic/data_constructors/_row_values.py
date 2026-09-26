"""Helpers that turn one raw metric-row value into a typed label or
card(s), shared by more than one DataConstructor in this package (e.g.
_card_for_uuid backs constructors for both single_card_regression and
single_card_fixed_classification).
"""

import math
from typing import List, Tuple, cast
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.card import GenericCard
from src.schema.type_hints import MultiCardInput


def _cast_to_float(raw_label: object) -> float:
    """DeckLabelDataConstructor's default label_caster - float() typed
    against object rather than its builtin overloads, so it satisfies
    Callable[[object], Label]."""
    return float(raw_label)  # type: ignore[arg-type]


def _label_as_float(raw_label: object) -> float | None:
    """Parse one row's raw label cell as a float, rejecting NaN.

    Shared by CardAverageDataConstructor and
    MaskedFieldRegressionDataConstructor - both families read a plain
    float64 label column and need the same NaN handling: a metric's own
    nullable-output convention (e.g. CardAverageMetric's None for a
    card never observed) round-trips through a float64 parquet column
    as NaN, not None, so it must be treated the same as an unparseable
    label rather than becoming a NaN training label. Unlike
    _cast_to_float above (DeckLabelDataConstructor's caster, raises on
    a bad value by design), every failure here collapses to None so a
    build() loop can skip the row.

    Inputs:
        raw_label: a row's label-column cell (float, int, or bool - see
            CardAverageMetric's WIN RATE IS AN AVERAGE note for why a
            bool is a legitimate input here).
    Output: raw_label as a float, or None if it isn't a float/int/bool,
        or is NaN.
    Side effects: none.
    Exceptions: none - all failures collapse to None.
    """
    if isinstance(raw_label, (float, int, bool)):
        value = float(raw_label)
        return None if math.isnan(value) else value
    return None


def _card_for_uuid(lookup: CardLookup, raw_nocab_uuid: object) -> GenericCard | None:
    """Look up a card for one row's raw nocab_uuid value.

    Shared by every DataConstructor in this package that resolves a
    single card by nocab_uuid (CardAverageDataConstructor,
    MaskedFieldDataConstructor, CardCharacterPredictionDataConstructor,
    PickNumberDecayCurveDataConstructor).

    Inputs:
        lookup: registry to resolve raw_nocab_uuid against. Never
            written to.
        raw_nocab_uuid: a row's "nocab_uuid" cell, expected to be a str
            parseable as a UUID.
    Output: the matching GenericCard, or None if raw_nocab_uuid doesn't
        parse as a UUID or lookup has no card for it.
    Side effects: none.
    Exceptions: none - all failures collapse to None.
    """
    try:
        card_uuid = UUID(str(raw_nocab_uuid))
    except (TypeError, ValueError):
        return None
    return lookup.get_by_uuid(card_uuid)


def _cards_for_uuids(lookup: CardLookup, raw_uuids: object) -> List[GenericCard]:
    """Look up cards for a raw list of uuid strings, dropping any that
    don't match a card.

    Shared by PoolConditionedPickDataConstructor for its pool_uuids
    column and AttackerBlockerCombatOutcomeDataConstructor for its
    attacker_uuids/blocker_uuids columns - NOT used for either option-
    selection metric's pack_option_uuids column, see
    _option_cards_and_pick_index()'s docstring for why that side can't
    tolerate silently dropping an unmatched entry.

    Inputs:
        lookup: registry to resolve each uuid against. Never
            written to.
        raw_uuids: a row's list[str] cell (e.g. "pool_uuids").
    Output: every uuid in raw_uuids that resolves to a card, same
        relative order; unparseable or unresolvable entries are
        silently dropped. Empty is a valid, expected output (e.g. a
        drafter's pool on the first pick of a draft) - unlike every
        other multi-card DataConstructor's "empty card list -> skip the
        row" convention, callers here must NOT treat an empty result as
        a reason to skip.
    Side effects: none.
    Exceptions: none.
    """
    cards: List[GenericCard] = []
    for raw_uuid in cast(List[str], raw_uuids):
        card = _card_for_uuid(lookup, raw_uuid)
        if card is None:
            continue
        cards.append(card)
    return cards


def _option_cards_and_pick_index(
    lookup: CardLookup,
    raw_pack_option_uuids: object,
    raw_pick_uuid: object,
) -> Tuple[MultiCardInput, int] | None:
    """Resolve one row's pack option list plus which position in it was
    picked.

    Shared by PackToPickChoiceSetDataConstructor and
    PoolConditionedPickDataConstructor.build() - both need the exact
    same "which option was picked" resolution, differing only in
    whether a second (pool) group is also attached to the result.

    Inputs:
        lookup: registry to resolve every option's uuid against.
            Never written to.
        raw_pack_option_uuids: a row's "pack_option_uuids" cell
            (list[str]).
        raw_pick_uuid: a row's "pick_uuid" cell (str, nullable - both
            metrics' own schemas allow an unmatched pick to round-trip
            as null - see their _output_row() docstrings).
    Output: (resolved option cards, index of the picked option within
        that list), cards in raw_pack_option_uuids' own order - or None
        if raw_pick_uuid is null/unparseable, doesn't appear in
        raw_pack_option_uuids, or ANY option uuid in
        raw_pack_option_uuids fails to parse or to resolve against
        lookup. That last condition is a deliberate
        divergence from DeckLabelDataConstructor's convention of
        dropping individual unresolved cards and keeping the row: here
        the label is a POSITION in this exact list, so silently
        dropping one card would shift every later index and corrupt the
        label instead of just shrinking the input - the whole row is
        skipped rather than risk that.
    Side effects: none.
    Exceptions: none - all failures collapse to None.
    """
    try:
        pick_uuid = UUID(str(raw_pick_uuid))
    except (TypeError, ValueError):
        return None

    option_uuids: List[UUID] = []
    for raw_option_uuid in cast(List[str], raw_pack_option_uuids):
        try:
            option_uuids.append(UUID(str(raw_option_uuid)))
        except (TypeError, ValueError):
            return None

    try:
        pick_index = option_uuids.index(pick_uuid)
    except ValueError:
        return None

    option_cards: MultiCardInput = []
    for option_uuid in option_uuids:
        card = lookup.get_by_uuid(option_uuid)
        if card is None:
            return None
        option_cards.append(card)

    return option_cards, pick_index
