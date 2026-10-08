"""Helpers that turn one raw metric-row value into a typed label or
card(s), shared by DataConstructors in this package and per-source ones such as
src/dojos/isotropic/ (e.g.
card_for_uuid backs constructors for both single_card_regression and
single_card_fixed_classification).
"""

import math
from typing import List, Tuple, cast
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericCard
from src.schema.type_hints import MultiCardInput


def cast_to_float(raw_label: object) -> float:
    """DeckLabelDataConstructor's default label_caster - float() typed
    against object rather than its builtin overloads, so it satisfies
    Callable[[object], Label]."""
    return float(raw_label)  # type: ignore[arg-type]


def label_as_float(raw_label: object) -> float | None:
    """Parse one row's raw label cell as a float, rejecting NaN.

    Shared by CardAverageDataConstructor and
    MaskedFieldRegressionDataConstructor - both families read a plain
    float64 label column and need the same NaN handling: a metric's own
    nullable-output convention (e.g. CardAverageMetric's None for a
    card never observed) round-trips through a float64 parquet column
    as NaN, not None, so it must be treated the same as an unparseable
    label rather than becoming a NaN training label. Unlike
    cast_to_float above (DeckLabelDataConstructor's caster, raises on
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


def parsed_uuid(raw_uuid: object) -> UUID | None:
    """One row's raw uuid cell as a UUID.

    Shared by every helper here that parses a uuid cell.

    Inputs: raw_uuid (expected a str parseable as a UUID).
    Output: the UUID, or None if raw_uuid does not parse.
    Side effects: none. Exceptions: none.
    """
    try:
        return UUID(str(raw_uuid))
    except (TypeError, ValueError):
        return None


def deck_cards_excluding(
    deck_box: DeckBox,
    lookup: CardLookup,
    raw_deck_uuid: object,
    target_uuid: UUID | None,
) -> MultiCardInput:
    """A row's deck as the split sees it, with every copy of the target
    removed.

    Shared by DeckCardMaskDataConstructor and
    HeldOutDeckCardDataConstructor. The target is filtered out BEFORE
    card lookup (filter first, look up second), so no masking Mod is
    needed for the multi-card case.

    Inputs:
        deck_box: the box raw_deck_uuid points into; only read.
        lookup: the split's holdout-filtered card lookup.
        raw_deck_uuid: a row's "deck_uuid" cell.
        target_uuid: the card to remove (every copy); None removes
            nothing.
    Output: the deck's remaining cards that lookup can see, in the
        deck's own order. Empty if raw_deck_uuid does not parse, the box
        has no such deck, or nothing visible remains: callers treat
        empty as "skip the row".
    Side effects: reads deck_box. Exceptions: none.
    """
    result: MultiCardInput = []
    deck_uuid = parsed_uuid(raw_deck_uuid)
    if deck_uuid is None:
        return result
    deck = deck_box.get_by_uuid(deck_uuid)
    if deck is None:
        return result

    # Filter the target first, then look each remaining card up
    for card_uuid in deck.card_nocab_uuids:
        if card_uuid == target_uuid:
            continue
        card = lookup.get_by_uuid(card_uuid)
        if card is not None:
            result.append(card)
    return result


def card_for_uuid(lookup: CardLookup, raw_nocab_uuid: object) -> GenericCard | None:
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
    card_uuid = parsed_uuid(raw_nocab_uuid)
    if card_uuid is None:
        return None
    return lookup.get_by_uuid(card_uuid)


def cards_for_uuids(lookup: CardLookup, raw_uuids: object) -> List[GenericCard]:
    """Look up cards for a raw list of uuid strings, dropping any that
    don't match a card.

    Shared by PoolConditionedPickDataConstructor for its pool_uuids
    column and AttackerBlockerCombatOutcomeDataConstructor for its
    attacker_uuids/blocker_uuids columns - NOT used for either option-
    selection metric's pack_option_uuids column, see
    option_cards_and_pick_index()'s docstring for why that side can't
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
        card = card_for_uuid(lookup, raw_uuid)
        if card is None:
            continue
        cards.append(card)
    return cards


def option_cards_for_uuids(
    lookup: CardLookup, raw_option_uuids: object
) -> MultiCardInput | None:
    """A row's option list as cards, all or nothing.

    Shared by option_cards_and_pick_index() and the datum builders whose
    pick may be "none" (CardRewardPickDataConstructor). Unlike
    cards_for_uuids(), one bad option voids the row: the label is a
    POSITION in this list, so dropping a card would shift it.

    Inputs: lookup (never written to), raw_option_uuids (a list[str]
        cell).
    Output: the option cards in the cell's order, or None if any uuid
        fails to parse or to look up.
    Side effects: none. Exceptions: none.

    Example:
        >>> option_cards_for_uuids(binder, [str(card_a.nocab_uuid)])
        [<GenericCard>]
    """
    result: MultiCardInput = []
    for raw_option_uuid in cast(List[str], raw_option_uuids):
        option_uuid = parsed_uuid(raw_option_uuid)
        if option_uuid is None:
            return None
        card = lookup.get_by_uuid(option_uuid)
        if card is None:
            return None
        result.append(card)
    return result


def pick_position(option_cards: MultiCardInput, pick_uuid: UUID | None) -> int | None:
    """Where a picked card sits in a row's option list.

    Inputs: option_cards (the row's options), pick_uuid (the picked
        card's uuid, or None).
    Output: the index of the first option with that uuid; None if
        pick_uuid is None or not among the options.
    Side effects: none. Exceptions: none.

    Example:
        >>> pick_position([card_a, card_b], card_b.nocab_uuid)
        1
    """
    for position, card in enumerate(option_cards):
        if card.nocab_uuid == pick_uuid:
            return position
    return None


def option_cards_and_pick_index(
    lookup: CardLookup,
    raw_pack_option_uuids: object,
    raw_pick_uuid: object,
) -> Tuple[MultiCardInput, int] | None:
    """One row's pack option list as cards, plus which position in it was
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
    pick_uuid = parsed_uuid(raw_pick_uuid)
    option_cards = option_cards_for_uuids(lookup, raw_pack_option_uuids)
    if pick_uuid is None or option_cards is None:
        return None

    pick_index = pick_position(option_cards, pick_uuid)
    return None if pick_index is None else (option_cards, pick_index)
