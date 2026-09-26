"""Shared, stateless helpers for reading one already-parsed GameHeader
(header_parser.py) - the games/ analogue of ../summary/row_utils.py.

card_uuid_for_name() is re-exported from ../card_names.py, not
defined here - see that module's own docstring for why: it became a
byte-identical second consumer of what was originally
../summary/row_utils.py's own private function, which is what pushed
the lookup past this project's usual per-container duplication
convention (contrast ../../sts_gg/deck_label_metric.py's module
docstring, which duplicates ITS OWN per-metric card-id resolution
deliberately - that case has a real per-metric row-shape difference to
justify keeping copies separate; this one never did).
"""

from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.isotropic.card_names import (
    card_uuid_for_name as card_uuid_for_name,
)
from src.data_refinement.metrics.isotropic.games.header_parser import (
    GameHeader,
    GameHeaderPlayer,
)


def winner_player(header: GameHeader) -> GameHeaderPlayer:
    """Find header's own winner among its players.

    Inputs:
        header: one parsed GameHeader.
    Output: the players[] entry whose nick == header.winner_nick.
    Side effects: none.
    Exceptions: raises ValueError if no player entry matches
        header.winner_nick - a genuine internal-consistency violation
        (header_parser.py's own parse_game_header() guarantees the
        winner nick it sets always has a matching player block), not a
        real "expected" outcome the way a Flavor A resignation is.

    Example:
        >>> winner_player(header).nick == header.winner_nick
        True
    """
    for player in header.players:
        if player.nick == header.winner_nick:
            return player
    raise ValueError(
        f"winner_player: {header.winner_nick!r} has no matching player block"
    )


def is_single_pile_ending(header: GameHeader) -> bool:
    """Whether header's game ended via exactly one exhausted pile (a
    Province/Colony-style ending) rather than a multi-pile ending.

    Inputs:
        header: one parsed GameHeader.
    Output: len(header.exhausted_pile_names) == 1.
    Side effects: none.
    Exceptions: none.
    """
    return len(header.exhausted_pile_names) == 1


def pile_card_uuid_for_name(card_lookup: CardLookup, plural_name: str) -> UUID | None:
    """Resolve one exhausted-pile name (English-pluralized, as
    header_parser.py extracts it) against the dominiontabs CardBinder.

    NOT a plain card_uuid_for_name() call: GameHeader.exhausted_pile_names
    entries are rendered in their PLURAL form ("Provinces", "Duchies",
    "Wharves" for Wharf), and Dominion's own card-name vocabulary has
    enough exceptions that no single depluralization rule inverts every
    observed case losslessly - confirmed live: "Ironworks" is NOT
    further pluralized at all (already reads as a plural-shaped name),
    while "Wharf" pluralizes irregularly to "Wharves" (a plain "strip
    trailing s" gives "Wharve", wrong). Rather than pick one rule and
    accept silent mis-resolution on whichever real card it's wrong for,
    this tries several depluralization candidates against the real
    CardBinder and returns the first that actually resolves - a
    correctness question this function can answer authoritatively
    (whether a given string is a real dominiontabs card name), unlike a
    pure string transformation guessing in the dark.

    Inputs:
        card_lookup: already-populated registry - must already have
            dominiontabs' cards ingested (this function never writes to
            it).
        plural_name: one GameHeader.exhausted_pile_names entry.
    Output: the matching nocab_uuid, or None if no candidate resolves
        (a real "this name doesn't match any known depluralization
        pattern" outcome - not assumed impossible, just not yet
        observed in this project's own live sampling).
    Side effects: none.
    Exceptions: none - card_uuid_for_name()'s own ambiguous-name
        exception is allowed to propagate for any candidate that
        triggers it, same as a direct caller of that function would see.

    Example:
        >>> pile_card_uuid_for_name(card_binder, "Provinces")
        UUID('...')
        >>> pile_card_uuid_for_name(card_binder, "Wharves")
        UUID('...')
        >>> pile_card_uuid_for_name(card_binder, "Ironworks")
        UUID('...')
    """
    for candidate in _depluralization_candidates(plural_name):
        card_uuid = card_uuid_for_name(card_lookup, candidate)
        if card_uuid is not None:
            return card_uuid
    return None


def _depluralization_candidates(plural_name: str) -> list[str]:
    """Every plausible singular form of one English-pluralized pile name.

    Private helper - single consumer is pile_card_uuid_for_name().
    Order matters only in that the unchanged name is tried first (the
    cheap, common case once a card is actually already singular-shaped,
    e.g. "Ironworks") - every candidate is still tried regardless of
    whether an earlier one "looks" more likely to be right, since only
    an actual CardBinder lookup can tell.

    Inputs:
        plural_name: one GameHeader.exhausted_pile_names entry.
    Output: candidate strings, in order: the name unchanged; "ies" ->
        "y" (if the name ends with "ies"); "ves" -> "f" (if the name
        ends with "ves"); "es" -> "is" (confirmed live: "Oases" ->
        "Oasis"); the name with a trailing "es" stripped (if
        present - confirmed live necessary: "Witches" -> "Witch", not
        the wrong "Witche" a plain single-"s" strip would give, since
        "ch"/"sh"/"ss"/"x"/"z"-ending words pluralize with "+es" in
        English); the name with one trailing "s" stripped (if present -
        the plain case, e.g. "Provinces" -> "Province"). Never fewer
        than one candidate (the unchanged name). More than one
        candidate is often tried even when only one is linguistically
        "correct" for a given name (e.g. both "es"- and "s"-stripped
        forms are generated for "Provinces") - harmless, since
        pile_card_uuid_for_name() only returns the first candidate that
        actually resolves against the real CardBinder. Only the head
        noun before " of " is depluralized (confirmed live: "Horns of
        Plenty" -> "Horn of Plenty").
    Side effects: none.
    Exceptions: none.
    """
    # "Horns of Plenty" pluralizes only its head noun, before " of ".
    head, separator, tail = plural_name.partition(" of ")
    return [form + separator + tail for form in _plural_word_singular_forms(head)]


def _plural_word_singular_forms(plural_text: str) -> list[str]:
    """Every plausible singular form of one English-pluralized phrase,
    the unchanged text first.

    Private helper - single consumer is _depluralization_candidates().

    Inputs:
        plural_text: a pluralized card name, or just its head noun.
    Output: candidates in order: unchanged; "ies" -> "y"; "ves" -> "f";
        "es" -> "is" ("Oases" -> "Oasis"); "es" stripped; "s" stripped.
    Side effects: none.
    Exceptions: none.
    """
    candidates = [plural_text]
    if plural_text.endswith("ies"):
        candidates.append(plural_text[: -len("ies")] + "y")
    if plural_text.endswith("ves"):
        candidates.append(plural_text[: -len("ves")] + "f")
    if plural_text.endswith("es"):
        candidates.append(plural_text[: -len("es")] + "is")
        candidates.append(plural_text[: -len("es")])
    if plural_text.endswith("s"):
        candidates.append(plural_text[: -len("s")])
    return candidates
