"""Derives a player's PARTIAL DECK as of just before some turn T - the
"mid-game partial deck" concept BRAINSTORM.md's "New candidates raised
during human review" section names as the shared prerequisite for every
*(Flavor B only)* mid-game metric (Next Buy Prediction, Next Trashed
Card Prediction, Next-Turn Action Count Prediction, Eventual Win
Probability, Deck Pair -> Winner Prediction).

CARD-QUANTITY EXPANSION LIVES HERE, NOT IN game_log_parser.py: a
game_log_parser.CardQuantity carries isotropic's own rendered text
("3 Coppers"), still English-pluralized when count > 1 - the same
depluralization ambiguity header_parser.py's
GameHeader.exhausted_pile_names already has (Wharf -> "Wharves", not a
plain "+s"), which row_utils.pile_card_uuid_for_name() already solves
via a CardBinder-verified depluralization-candidate search rather than
a single guessed string rule. expand_card_quantity() below reuses that
same idea (not the same function - see its own docstring) to turn one
CardQuantity into `count` individual matched nocab_uuids. Keeping this
CardBinder-dependent step out of game_log_parser.py preserves that
module's existing boundary (it never touches a CardBinder at all,
matching header_parser.py's own convention).

STARTING DECK IS A GAME-RULES CONSTANT, NOT READ OFF THE LOG: a Flavor
B log's own "({nick}'s first hand: ...)" line only shows the 5 cards
actually drawn into that opening hand, never the other 5 of the real
10-card vanilla Dominion starting deck (7 Copper, 3 Estate) - so
partial_deck_card_uuids() below always starts from
_STARTING_DECK rather than parsing that line at all.
"""

import logging
from uuid import UUID
from weakref import WeakKeyDictionary

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.isotropic.games.game_log_parser import (
    CardQuantity,
    GameLog,
)
from src.data_refinement.metrics.isotropic.games.row_utils import (
    pile_card_uuid_for_name,
)

_logger = logging.getLogger(__name__)

# Vanilla Dominion's fixed starting deck - see module docstring's
# STARTING DECK note for why this isn't read off the log itself.
_STARTING_DECK: tuple[CardQuantity, ...] = (
    CardQuantity("Copper", 7),
    CardQuantity("Estate", 3),
)


# Per-lookup memo of card-name text -> nocab_uuid (None cached too). Weak keys
# so a discarded binder (e.g. one per test) is not kept alive by the cache.
_NAME_CACHES: "WeakKeyDictionary[CardLookup, dict[str, UUID | None]]" = (
    WeakKeyDictionary()
)


def _cached_card_uuid(card_lookup: CardLookup, card_name_text: str) -> UUID | None:
    """Memoized row_utils.pile_card_uuid_for_name(), logging each
    unmatched name once per lookup.

    Private helper - single consumer is card_uuid_for_quantity_name().
    Every CardBinder name lookup deep-copies the card (measured at ~70%
    of a scan's runtime), and a partial deck is rebuilt from scratch for
    every player-turn of every metric, so the same few hundred names are
    looked up millions of times. Logging on a cache miss (not per call)
    also keeps one unmatched name from logging once per later turn.
    Hazard: a None result is cached too, so a binder must be fully
    ingested before it is first queried through this module - metrics
    only ever read a finished binder. card_lookup must be hashable and
    weak-referenceable (CardBinder is).

    Inputs:
        card_lookup: already-populated registry.
        card_name_text: singular or English-pluralized card name text.
    Output: the matching nocab_uuid, or None if no candidate matches.
    Side effects: fills the cache; emits one logging.error() the first
        time a name is found unmatched for this card_lookup.
    Exceptions: whatever pile_card_uuid_for_name() raises.
    """
    cache = _NAME_CACHES.setdefault(card_lookup, {})
    if card_name_text not in cache:
        card_uuid = pile_card_uuid_for_name(card_lookup, card_name_text)
        if card_uuid is None:
            _logger.error("partial_deck: unmatched card name %r", card_name_text)
        cache[card_name_text] = card_uuid
    return cache[card_name_text]


def partial_deck_card_uuids(
    card_lookup: CardLookup,
    game_log: GameLog,
    player_nick: str,
    before_turn_number: int,
) -> list[UUID]:
    """Compute one player's partial deck as of just before their own
    turn `before_turn_number` (module docstring's per-player turn-count
    convention - see game_log_parser.py's own module docstring).

    Inputs:
        card_lookup: already-populated registry - must already have
            dominiontabs' cards ingested (this function never writes to
            it).
        game_log: one parsed GameLog (game_log_parser.py).
        player_nick: whose partial deck to compute - must match one of
            game_log.header.players' own nicks.
        before_turn_number: only that player's turns strictly before
            this turn number are folded in - e.g. before_turn_number=1
            yields just _STARTING_DECK, unmodified.
    Output: every matched nocab_uuid currently in that partial deck,
        as a flat list (one entry per physical card - duplicates
        expected and meaningful, e.g. 7 Copper entries at the start).
        Order is not meaningful.
    Side effects: none. Unmatched names are logged once per lookup (see
        _cached_card_uuid()) and excluded from the result, not raised on.
    Exceptions: none expected beyond a malformed game_log.

    Example:
        >>> partial_deck_card_uuids(card_binder, game_log, "Strategyst", 1)
        [UUID(...), UUID(...), ...]  # 7 Copper + 3 Estate uuids
    """
    result: list[UUID] = []
    for quantity in _STARTING_DECK:
        result.extend(expand_card_quantity(card_lookup, quantity))

    # Fold in every earlier turn this player took, in turn order.
    earlier_turns = [
        turn
        for turn in game_log.turns
        if turn.player_nick == player_nick and turn.turn_number < before_turn_number
    ]
    for turn in earlier_turns:
        for quantity in turn.cards_gained:
            result.extend(expand_card_quantity(card_lookup, quantity))
        for quantity in turn.cards_trashed:
            for card_uuid in expand_card_quantity(card_lookup, quantity):
                if card_uuid in result:
                    result.remove(card_uuid)

    return result


def expand_card_quantity(card_lookup: CardLookup, quantity: CardQuantity) -> list[UUID]:
    """Expand one CardQuantity (isotropic's own rendered, possibly
    plural, card-name text plus a count) into `quantity.count`
    individual matched nocab_uuids of the same card.

    Inputs:
        card_lookup: already-populated registry - must already have
            dominiontabs' cards ingested (this function never writes to
            it).
        quantity: one CardQuantity from a Turn's cards_bought/
            cards_gained/cards_trashed/cards_played.
    Output: quantity.count copies of the same matched nocab_uuid, or
        an empty list if quantity.card_name_text matches no
        depluralization candidate (see card_uuid_for_quantity_name() -
        a real "unmatched name" outcome, not an exception).
    Side effects: whatever card_uuid_for_quantity_name() does.
    Exceptions: none.

    Example:
        >>> expand_card_quantity(card_binder, CardQuantity("Coppers", 3))
        [UUID('...'), UUID('...'), UUID('...')]
    """
    card_uuid = card_uuid_for_quantity_name(card_lookup, quantity)
    if card_uuid is None:
        return []
    return [card_uuid] * quantity.count


def card_uuid_for_quantity_name(
    card_lookup: CardLookup, quantity: CardQuantity
) -> UUID | None:
    """Look up one CardQuantity's card_name_text to a single nocab_uuid,
    independent of quantity.count (a distinct-card-identity question,
    for callers that want "which card" rather than "how many copies" -
    e.g. a metric's variable-set label over which cards were bought,
    not how many of each).

    NOT a plain card_uuid_for_name() call when quantity.count > 1 - see
    module docstring's CARD-QUANTITY EXPANSION note; reuses
    row_utils.pile_card_uuid_for_name()'s candidate-search idea against
    quantity.card_name_text, rather than card_uuid_for_name() directly,
    for exactly the cases that function's own docstring documents
    (Wharf -> "Wharves", "Ironworks" unchanged, etc.) - card_name_text
    is already singular and a plain card_uuid_for_name() call would
    suffice whenever quantity.count == 1, but the candidate search is
    harmless to run unconditionally (it tries the unchanged name
    first).

    Inputs:
        card_lookup: already-populated registry.
        quantity: one CardQuantity.
    Output: the matching nocab_uuid, or None if unmatched.
    Side effects: see _cached_card_uuid() (logs each unmatched name once).
    Exceptions: none.
    """
    # pile_card_uuid_for_name() already does the CardBinder-verified
    # depluralization search this function's docstring describes.
    return _cached_card_uuid(card_lookup, quantity.card_name_text)


def distinct_card_uuids(
    card_lookup: CardLookup, quantities: tuple[CardQuantity, ...]
) -> list[UUID]:
    """Look up a tuple of CardQuantity to their DISTINCT card
    identities - one nocab_uuid per distinct card named, not one per
    physical copy (contrast expand_card_quantity(), which is
    count-aware).

    Shared by any metric that needs "which card(s)," not "how many" -
    e.g. mid_game_next_buy_metric.py's own next_buy_card_uuids label and
    mid_game_next_trashed_card_metric.py's own
    next_trashed_card_uuids label both reduce to exactly this
    transform, with zero per-metric row-shape variation - the same
    situation card_resolution.py's own module docstring documents for
    why card_uuid_for_name() was promoted out of a single per-subpackage
    row_utils.py once a second, byte-identical consumer appeared. This
    lives in partial_deck.py (not row_utils.py) since both of its
    current consumers are already partial_deck.py's own dependents.

    Inputs:
        card_lookup: already-populated registry - must already have
            dominiontabs' cards ingested (this function never writes to
            it).
        quantities: e.g. one Turn.cards_bought or Turn.cards_trashed.
    Output: one nocab_uuid per distinct matchable card named across
        quantities, in first-seen order, never repeated even if
        quantities names the same card more than once (whether via one
        CardQuantity with count > 1, or two separate CardQuantity
        entries for the same card). Excludes (does not raise for) any
        quantity that has no match.
    Side effects: see _cached_card_uuid() (logs each unmatched name
        once).
    Exceptions: none.

    Example:
        >>> distinct_card_uuids(
        ...     card_binder,
        ...     (CardQuantity("Coppers", 3), CardQuantity("Silver", 1)),
        ... )
        [UUID('...copper...'), UUID('...silver...')]
    """
    result: list[UUID] = []
    for quantity in quantities:
        card_uuid = card_uuid_for_quantity_name(card_lookup, quantity)
        if card_uuid is not None and card_uuid not in result:
            result.append(card_uuid)
    return result
