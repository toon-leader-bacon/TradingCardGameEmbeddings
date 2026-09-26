"""Shared, stateless helpers for reading one Flavor A ("summary" JSONL)
row - see BRAINSTORM.md for the confirmed row shape. Every metric in
this container reads the same handful of row fields (the kingdom, the
winner, whether this game's kingdom was isotropic-generator-constrained)
the same way, so those reads live here once rather than being
re-derived per metric the way sts_gg's per-metric "CARD."-prefix
resolution deliberately is (see ../../sts_gg/deck_label_metric.py's module
docstring for that project's opposite call on a similar-looking
question) - isotropic has no equivalent plan document making that same
call, and the isotropic-specific rules below ("natural kingdom"
filtering, winner/eligibility lookups) are non-trivial enough that
duplicating them per metric would be a real PRINCIPLES.md violation,
not a stylistic preference.

card_uuid_for_name() itself is re-exported from ../card_names.py,
not defined here - see that module's own docstring for why (it became
a byte-identical second consumer once games/row_utils.py needed the
exact same lookup, which is what pushed it past the "duplicate per
container" convention the rest of this file's docstring argues for).

None of these raise on a malformed row - each documents its own
"can't determine this" return value (None, [], False) and leaves the
decision of whether that's fatal to the calling metric, matching
BRAINSTORM.md's "Known biases" section (e.g. every player resigned, so
there is no winner; this is a real, expected shape, not corrupt data).

Typed against CardLookup, not CardBinder: every function here only
ever reads (see ../../../card_binder/card_lookup.py's own module
docstring - "any consumer that only needs to read... should type
against CardLookup, not CardBinder, even when a real CardBinder is
what gets passed in"). A CardBinder instance satisfies CardLookup
structurally, so every caller in this container that holds a
self._card_binder: CardBinder keeps working unchanged; this is also
what lets deck_card_mask_metric.py's WinningDeckMaskedCardMetric (which
only ever has a CardLookup, per ../../generic/deck_card_mask_metric.py's
own contract) call these same functions directly instead of
duplicating card resolution a second time.
"""

import logging
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.card_names import (
    card_uuid_for_name as card_uuid_for_name,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_KINGDOM_CONSTRAINT_FLAGS: tuple[str, ...] = (
    "constraints",
    "required",
    "prohibited",
    "force_big_cards",
    "force_equal_start",
    "force_banes",
)


def winner_entry(row: dict) -> dict | None:
    """Find the row's rank == 1 player entry with a real final deck.

    Inputs:
        row: one parsed Flavor A summary row (see BRAINSTORM.md).
    Output: the players[] entry with rank == 1, or None if either no
        player has rank == 1 (never observed, but not assumed
        impossible) or that entry has no "end" block (a rank-1 player
        can still be resigned in principle - BRAINSTORM.md's
        "resignation" bias note - though whether that combination is
        ever actually observed is confirmed/refuted at implement time,
        not here).
    Side effects: none.
    Exceptions: none.

    Example:
        >>> winner_entry(row)["nick"]
        'unlikely'
    """
    for player in row["players"]:
        if player.get("rank") == 1 and "end" in player:
            return player
    return None


def kingdom_card_names(row: dict) -> list[str]:
    """Pull this row's dealt kingdom as a plain list of card names.

    Inputs:
        row: one parsed Flavor A summary row.
    Output: row["board"]["supply"], unchanged (10, rarely 11, card
        names) - callers resolve each name themselves via
        card_uuid_for_name(), since some callers need the raw name
        alongside the uuid (e.g. to index into end.deck) and some
        don't.
    Side effects: none.
    Exceptions: none.
    """
    return row["board"]["supply"]


def is_natural_kingdom(row: dict) -> bool:
    """Whether this row's kingdom was isotropic's plain random deal,
    not one of its own curated/challenge generator knobs.

    See BRAINSTORM.md's "Known biases" section: board.constraints/
    required/prohibited/force_big_cards/force_equal_start/force_banes
    are isotropic's own kingdom-generator settings, not vanilla
    Dominion - a metric computing a "natural" per-card/per-kingdom
    baseline should filter to rows where this returns True.
    Deliberately NOT checking board.point_tracker/black_market/bane:
    those reflect which CARDS are in the kingdom (Colony/Platinum-style
    scoring, Black Market, Young Witch's bane), not an artificial
    constraint on WHICH TEN cards got dealt - a real natural game can
    have any of those three.

    Inputs:
        row: one parsed Flavor A summary row.
    Output: True if none of _KINGDOM_CONSTRAINT_FLAGS is present as a
        truthy key of row["board"].
    Side effects: none.
    Exceptions: none.
    """
    board = row["board"]
    return not any(board.get(flag) for flag in _KINGDOM_CONSTRAINT_FLAGS)


def eligible_player_entries(row: dict) -> list[dict]:
    """This row's players[] entries that reached a real game end.

    Inputs:
        row: one parsed Flavor A summary row.
    Output: every entry of row["players"] that has an "end" block -
        excludes resigned players (BRAINSTORM.md's "resignation" bias
        note: a resigned player's final deck was never recorded).
        Order preserved from row["players"].
    Side effects: none.
    Exceptions: none.
    """
    return [player for player in row["players"] if "end" in player]


def deck_card_uuids(card_lookup: CardLookup, end_deck: dict[str, int]) -> list[UUID]:
    """Expand one player's end.deck (name -> copy count) into a flat
    card_nocab_uuids multiset, resolving every name.

    Shared by every metric that needs a full resolved deck (as opposed
    to kingdom_card_names()'s unresolved kingdom list) - e.g. building
    a GenericDeck for DeckBox, or computing per-card copy counts.

    Inputs:
        card_lookup: same registry as card_uuid_for_name().
        end_deck: one player's row["players"][i]["end"]["deck"] dict.
    Output: one nocab_uuid per physical copy (a card with count 3
        appears 3 times) - unresolved names are skipped, not raised on
        (mirrors every ../../sts_gg metric's own "log and exclude"
        convention for a card id that fails to resolve), so this
        function's own docstring does NOT promise len(output) ==
        sum(end_deck.values()).
    Side effects: implementation-defined logging (expected: one log
        line per unresolved name).
    Exceptions: none.
    """
    card_nocab_uuids: list[UUID] = []
    for card_name, count in end_deck.items():
        card_uuid = card_uuid_for_name(card_lookup, card_name)
        if card_uuid is None:
            _logger.error(
                "deck_card_uuids: unresolved card name %r - excluding it "
                "from this deck",
                card_name,
            )
            continue
        card_nocab_uuids.extend([card_uuid] * count)
    return card_nocab_uuids


def deck_for_player(card_lookup: CardLookup, player_entry: dict) -> GenericDeck:
    """Build the GenericDeck for one eligible player's end.deck.

    Shared by every metric whose training input is one player's full
    final deck - full_deck_win_prediction_metric.py,
    deck_pair_winner_metric.py, and multiplayer_placement_metric.py
    each reached this exact sequence independently; every one of their
    skeletons flagged "a third real consumer would be the trigger to
    share this" - this is that third consumer, centralized here rather
    than duplicated a third time. Each of those files keeps its own
    thin _deck_for_player() wrapper delegating to this function, so
    their approved skeleton's method surface is unchanged.

    Inputs:
        card_lookup: same registry as card_uuid_for_name().
        player_entry: one players[] entry with a real "end" block.
    Output: a GenericDeck with source_game=GameId.DOMINION, content-
        addressed via hash_utils.deck_uuid_from_cards() (so two
        players' identical final decks - even across different games -
        collapse to the same deck_uuid). provenance is left at its
        default (None) - a private, metrics-only DeckBox entry, not a
        canonical ingested deck, mirroring
        ../../sts_gg/deck_label_metric.py's own GenericDeck
        construction (which likewise omits provenance entirely).
    Side effects: none (the caller writes it into its own deck_box).
    Exceptions: none.

    Example:
        >>> deck_for_player(card_binder, winner_entry(row))
        GenericDeck(...)
    """
    end_deck = player_entry["end"]["deck"]
    card_nocab_uuids = deck_card_uuids(card_lookup, end_deck)
    deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
    return GenericDeck(
        nocab_uuid=deck_uuid,
        source_game=GameId.DOMINION,
        name=f"isotropic {player_entry.get('nick', 'unknown')} final deck",
        card_nocab_uuids=card_nocab_uuids,
    )
