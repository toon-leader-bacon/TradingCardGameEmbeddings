"""The per-player action-line and continuation-line grammar (play/buy/
trash/gain/replay) for one isotropic Flavor B turn block - split out of
game_log_parser.py (see that module's docstring for the turn-block/
header-parsing half of this source's shape, and its Turn/GameLog for
the data model these matchers feed). CardQuantity - the one piece of
that data model these matchers themselves build - lives here instead,
and game_log_parser.py imports it back for Turn's own fields.

Single consumer: game_log_parser._parse_turn_block(). See that
function's own docstring (SCOPE NARROWED TO THIS BLOCK'S OWN OWNER) for
the cross-player-attribution policy every matcher below implements via
_line_belongs_to_owner().

ACTION LINES CONFIRMED (all belong to the turn's own owner, never an
explicit different nick - only CONTINUATION lines, below, are ever
attributed to someone else): "{nick} plays {a Cardname|N
Cardnames}." (0+ per turn, one per distinct card/count played),
"{nick} buys {a Cardname|N Cardnames}." (0+ per turn - never assumed
exactly one), "{nick} resigns from the game[...]." and "{nick} wins!" (the
terminal line, redundant with header_parser.py's own winner_nick) -
neither is parsed: a game containing any resignation is already
rejected by header_parser.parse_game_header() before this module runs,
and neither line matches a captured shape.

CONTINUATION LINES CONFIRMED (prefixed "... ", one level of indentation
deeper than the action line whose effect they resolve; MAY name an
explicit different nick than the turn's own owner - confirmed live for
attack/reaction effects, e.g. a Militia forcing an opponent to discard):
"... getting +1 action." / "... drawing 1 card and getting +$1." (no
card name - ignored), "... revealing {card list}." (ignored - see
SCOPE above), "... trashing {a Cardname} for +${N}." (a same-owner
trash), "... {other_nick} trashes {N} Cardnames, gaining a {Cardname}
in hand." (an explicit-other-nick trash-AND-gain combined in one
sentence - both effects attributed to other_nick, not the turn's
owner), "... gaining {a Cardname}." / "... {other_nick} gains {a
Cardname} and {a Cardname}." (both same-owner and explicit-other-nick
gain shapes confirmed live), "... and plays {a Cardname} again." / "...
and plays the {Cardname} a third time." (Throne-Room/King's-Court
replay narration - same owner, no new nick - counted as a play of that
card, same as a normal "plays" action line), every other confirmed
continuation shape (discards, draws, shuffles, "puts back", "returning
... to the supply") - ignored, per SCOPE above.
"""

import re
from dataclasses import dataclass

# One card phrase: "a Copper", "an Estate", "the Copper", "another Silver",
# "the trashed Silver" (Thief), or "2 Coppers" (module docstring's
# CardQuantity note on plurals).
_CARD_ITEM_PATTERN = re.compile(
    r"^(?:(?:an?|the|another) (?:trashed )?|(?P<count>\d+) )(?P<name>.+)$"
)
_LIST_SEPARATOR_PATTERN = re.compile(r", and |, | and ")

_CONTINUATION_PREFIX = r"^(?:\.\.\. )+"
_HAND_OR_DECK_SUFFIX = (
    r"(?: in (?:the )?hand| on top of the deck| on the deck"
    r"| and put(?:ting)? it on the deck| to replace it)?"
)
_TRASH_SUFFIX = r"(?: for \+.*| and gets \+.*| from (?:the )?(?:hand|play area))?"

# "... [{nick} [discards {cards} and ]gains|[revealing {cards} and ]gaining] {cards}"
_GAIN_PATTERN = re.compile(
    _CONTINUATION_PREFIX
    + r"(?:revealing .+? and )?"
    + r"(?:(?P<nick>.+?) (?:discards .+? and )?gains|gaining) "
    + r"(?P<cards>.+?)"
    + _HAND_OR_DECK_SUFFIX
    + r"\.$"
)
# Mint: "... revealing a Silver and gaining another one." - a gain of a
# copy of the revealed card, always the turn's own owner (no nick).
_REVEAL_AND_COPY_PATTERN = re.compile(
    _CONTINUATION_PREFIX + r"revealing (?P<cards>.+?) and gaining another one\.$"
)
_TRASH_PATTERN = re.compile(
    _CONTINUATION_PREFIX
    + r"(?:revealing .+? and )?(?:(?P<nick>.+?) trashes|trashing) (?P<cards>.+?)"
    + _TRASH_SUFFIX
    + r"\.$"
)
_REVEAL_AND_TRASH_PATTERN = re.compile(
    _CONTINUATION_PREFIX
    + r"(?P<nick>.+?) (?:reveals|turns up) (?P<cards>.+?) and trashes it\.$"
)
_TRASH_AND_GAIN_PATTERN = re.compile(
    _CONTINUATION_PREFIX
    + r"(?:(?P<nick>.+?) trashes|trashing) (?P<trashed>.+?)(?:, | and )gaining "
    + r"(?P<gained>.+?)(?: in (?:the )?hand)?\.$"
)
_REPLAY_PATTERN = re.compile(
    _CONTINUATION_PREFIX
    + r"and plays (?P<card>(?:an?|the) .+?)"
    + r"(?: again| an? \w+ time)?\.$"
)


@dataclass(frozen=True)
class CardQuantity:
    """One phrase's card-name text and how many copies it refers to.

    card_name_text is English-pluralized exactly as isotropic renders
    it when count > 1 ("Coppers"), and singular when count == 1
    ("Copper") - the SAME depluralization ambiguity
    header_parser.GameHeader.exhausted_pile_names' own field comment
    documents (e.g. "Wharves" for Wharf, "Ironworks" unchanged),
    deliberately not resolved to individual singular card names at
    parse time, since doing that correctly needs a real CardBinder
    lookup this module has no access to (matching header_parser.py's
    own established boundary: this parser never touches a CardBinder,
    only partial_deck.py's downstream helpers do). See
    partial_deck.expand_card_quantity() for where count > 1 actually
    gets expanded into count individual singular card names.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    card_name_text: str
    count: int


def _parse_card_quantity_list(list_text: str) -> tuple[CardQuantity, ...]:
    """Parse an "and"/Oxford-comma-joined list of "{a Cardname}"/"{an
    Cardname}"/"{N Cardnames}" phrases into one CardQuantity per phrase
    - the SHARED GRAMMAR behind every ACTION/CONTINUATION line shape
    below that names cards at all.

    Private helper - consumed by every one of this module's line
    matchers, via _parse_owner_action_line() for plays/buys,
    _parse_trash_and_gain_continuation_line() (both its trashed and
    gained clauses), _parse_trash_continuation_line(),
    _parse_gain_continuation_line(), and
    _parse_replay_continuation_line() (whose phrase is always a single
    card, the one-item case of this same grammar). Centralized here
    rather than re-derived per matcher, since every matcher shares the
    identical "and"/Oxford-comma phrase-list shape - the same near-identical-logic
    situation card_names.card_uuid_for_name() and this module's own
    sibling partial_deck.distinct_card_uuids() were already promoted
    out of per-consumer copies for.

    Inputs:
        list_text: the object-list portion of a line, already stripped
            of its leading verb/nick and trailing punctuation by the
            calling matcher - e.g. "a Silver and 4 Coppers",
            "2 Estates", "an Estate, 2 Coppers, and a Mountebank".
    Output: one CardQuantity per phrase, in list_text's own order -
        "a"/"an" imply count=1 (already-singular card_name_text); "N
        Cardnames" implies count=N (still-plural card_name_text,
        depluralized downstream - see CardQuantity's own docstring). An
        EMPTY tuple if any phrase isn't a card phrase at all ("nothing",
        "+$2", "another one") - callers treat that as "line not
        matched", not as zero cards.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> _parse_card_quantity_list("a Silver and 4 Coppers")
        (CardQuantity('Silver', 1), CardQuantity('Coppers', 4))
    """
    quantities: list[CardQuantity] = []
    for item in _LIST_SEPARATOR_PATTERN.split(list_text):
        match = _CARD_ITEM_PATTERN.match(item)
        # "another one" names no card ("revealing a Silver and gaining
        # another one"), so the whole phrase list is treated as unparseable.
        if match is None or match.group("name") == "one":
            return ()
        count = int(match.group("count")) if match.group("count") else 1
        quantities.append(CardQuantity(match.group("name"), count))
    return tuple(quantities)


def _line_belongs_to_owner(nick_in_line: str | None, owner_nick: str) -> bool:
    """Whether a CONTINUATION line's own optional named nick should be
    attributed to owner_nick - the SHARED GATING CHECK behind
    game_log_parser._parse_turn_block()'s own SCOPE NARROWED convention
    (see that function's docstring).

    Private helper - consumed by _parse_trash_and_gain_continuation_line(),
    _parse_trash_continuation_line(), and
    _parse_gain_continuation_line() - the three CONTINUATION-line
    matchers whose own regex may or may not capture an explicit nick
    (_parse_replay_continuation_line() never names one).
    ACTION-line matchers (_parse_play_line(), _parse_buy_line()) do NOT
    use this helper: an action line always names an explicit nick
    (module docstring), so those two only need a plain equality check
    against the nick they unconditionally capture, not this
    three-way (none / same / different) gate.

    Inputs:
        nick_in_line: the nick a continuation-line regex captured, or
            None if the line named no nick at all (the plain, no-nick
            shape - implicitly the turn's own owner).
        owner_nick: this block's own turn owner.
    Output: True if nick_in_line is None or equals owner_nick (accept -
        attribute to owner_nick); False if nick_in_line names a
        different, explicit nick (reject - out of this pass's scope,
        per game_log_parser._parse_turn_block()'s own documented
        interim decision).
    Side effects: none.
    Exceptions: none.
    """
    return nick_in_line is None or nick_in_line == owner_nick


def _parse_owner_action_line(
    line: str, owner_nick: str, verb: str
) -> tuple[CardQuantity, ...] | None:
    """Match "[... ]{owner_nick} {verb} {card list}." - the shared shape
    behind the play and buy matchers.

    Private helper - consumed by _parse_play_line() and
    _parse_buy_line(). The nick is matched literally (re.escape) rather
    than captured, since nicks contain spaces and punctuation and the
    owner is already known; a line naming any other nick simply fails to
    match.

    Inputs:
        line: one raw line from a turn block.
        owner_nick: this block's own turn owner.
        verb: "plays" or "buys".
    Output: _parse_card_quantity_list()'s result, or None if the line
        doesn't match or lists no parseable cards.
    Side effects: none.
    Exceptions: none.
    """
    pattern = rf"^(?:\.\.\. )*{re.escape(owner_nick)} {verb} (?P<cards>.+)\.$"
    match = re.match(pattern, line.strip())
    if match is None:
        return None
    return _parse_card_quantity_list(match.group("cards")) or None


def _parse_play_line(line: str, owner_nick: str) -> tuple[CardQuantity, ...] | None:
    """Match "{owner_nick} plays {a Cardname|N Cardnames}[ and ...]."
    (module docstring's ACTION LINES).

    Private helper - single consumer is
    game_log_parser._parse_turn_block(). A play line always names the
    turn's own owner explicitly (module docstring: "all belong to the
    turn's own owner, never an explicit different nick") - unlike the
    continuation-line matchers below, this one checks its captured nick
    against owner_nick directly, not via _line_belongs_to_owner() (which
    exists for the three-way none/same/different gate a continuation
    line needs, not this shape's plain two-way check). Delegates the
    actual card-list text to _parse_card_quantity_list().

    Inputs:
        line: one raw line from a turn block (not yet known to match
            this shape).
        owner_nick: this block's own turn owner.
    Output: _parse_card_quantity_list()'s own result over the played
        card-list text (module docstring: "can play multiple different
        card groups on separate lines" - AND a single line can itself
        list more than one, e.g. "plays a Silver and 4 Coppers."), or
        None if line doesn't match this shape, or matches but names a
        nick other than owner_nick.
    Side effects: none.
    Exceptions: none.
    """
    return _parse_owner_action_line(line, owner_nick, "plays")


def _parse_buy_line(line: str, owner_nick: str) -> tuple[CardQuantity, ...] | None:
    """Match "{owner_nick} buys {a Cardname|N Cardnames}[ and ...]."
    (module docstring's ACTION LINES - "never assumed exactly one" buy
    verb per turn, and a single buy line can itself list more than one
    card, same as _parse_play_line()).

    Private helper - single consumer is
    game_log_parser._parse_turn_block(). Same plain-equality nick check
    as _parse_play_line() (not _line_belongs_to_owner() - see that
    function's own docstring for why). Delegates the card-list text to
    _parse_card_quantity_list().

    Inputs:
        line: one raw line from a turn block.
        owner_nick: this block's own turn owner.
    Output: _parse_card_quantity_list()'s own result over the bought
        card-list text, or None if line doesn't match this shape, or
        names a different nick.
    Side effects: none.
    Exceptions: none.
    """
    return _parse_owner_action_line(line, owner_nick, "buys")


def _parse_trash_and_gain_continuation_line(
    line: str, owner_nick: str
) -> tuple[tuple[CardQuantity, ...], tuple[CardQuantity, ...]] | None:
    """Match "... [{owner_nick} ]trashes {a Cardname|N Cardnames},
    gaining {a Cardname} in hand." (module docstring's CONTINUATION
    LINES - one combined trash-and-gain sentence).

    Private helper - single consumer is
    game_log_parser._parse_turn_block(). Tried BEFORE
    _parse_trash_continuation_line() in that function's own dispatch
    order, since a plain trash matcher would otherwise partially match
    this line's own leading "trashes ..." clause and silently drop its
    trailing ", gaining ... in hand" clause. Gates its captured (or
    absent) nick via _line_belongs_to_owner(); delegates both the
    trashed-card-list and the gained-card text to
    _parse_card_quantity_list().

    Inputs:
        line: one raw "... " continuation line.
        owner_nick: this block's own turn owner - see
            _line_belongs_to_owner()'s own docstring for the accept/
            reject rule this applies.
    Output: (trashed_quantities, gained_quantities), or None if line
        doesn't match this shape, or _line_belongs_to_owner() rejects
        its captured nick.
    Side effects: none.
    Exceptions: none.
    """
    match = _TRASH_AND_GAIN_PATTERN.match(line.strip())
    if match is None or not _line_belongs_to_owner(match.group("nick"), owner_nick):
        return None
    trashed = _parse_card_quantity_list(match.group("trashed"))
    gained = _parse_card_quantity_list(match.group("gained"))
    if not trashed or not gained:
        return None
    return trashed, gained


def _parse_trash_continuation_line(
    line: str, owner_nick: str
) -> tuple[CardQuantity, ...] | None:
    """Match "... [{owner_nick} ]trash(es|ing) {a Cardname|N
    Cardnames}[ for +${N}]." (module docstring's CONTINUATION LINES -
    the plain trash shape, without a combined gain clause - see
    _parse_trash_and_gain_continuation_line() for that one).

    Private helper - single consumer is
    game_log_parser._parse_turn_block(). Gates its captured (or absent)
    nick via _line_belongs_to_owner(); delegates the card-list text to
    _parse_card_quantity_list().

    Inputs:
        line: one raw "... " continuation line.
        owner_nick: this block's own turn owner - see
            _line_belongs_to_owner()'s own docstring.
    Output: _parse_card_quantity_list()'s own result over the trashed
        card-list text, or None if line doesn't match this shape, or
        _line_belongs_to_owner() rejects its captured nick.
    Side effects: none.
    Exceptions: none.
    """
    stripped = line.strip()
    # Both patterns can partially match one line (a "reveals ... and
    # trashes it" line also fits _TRASH_PATTERN with the wrong nick), so
    # each is tried in turn rather than stopping at the first match.
    for pattern in (_REVEAL_AND_TRASH_PATTERN, _TRASH_PATTERN):
        match = pattern.match(stripped)
        if match is None or not _line_belongs_to_owner(match.group("nick"), owner_nick):
            continue
        quantities = _parse_card_quantity_list(match.group("cards"))
        if quantities:
            return quantities
    return None


def _parse_gain_continuation_line(
    line: str, owner_nick: str
) -> tuple[CardQuantity, ...] | None:
    """Match "... [{owner_nick} ]gain(s|ing) {a Cardname|N Cardnames}[
    and ...]." (module docstring's CONTINUATION LINES - a plain gain).

    Private helper - single consumer is
    game_log_parser._parse_turn_block(). Gates its captured (or absent)
    nick via _line_belongs_to_owner(); delegates the card-list text to
    _parse_card_quantity_list().

    Inputs:
        line: one raw "... " continuation line.
        owner_nick: this block's own turn owner - see
            _line_belongs_to_owner()'s own docstring.
    Output: _parse_card_quantity_list()'s own result over the gained
        card-list text, or None if line doesn't match this shape, or
        _line_belongs_to_owner() rejects its captured nick.
    Side effects: none.
    Exceptions: none.
    """
    stripped = line.strip()
    copy_match = _REVEAL_AND_COPY_PATTERN.match(stripped)
    if copy_match is not None:
        return _parse_card_quantity_list(copy_match.group("cards")) or None
    match = _GAIN_PATTERN.match(stripped)
    if match is None or not _line_belongs_to_owner(match.group("nick"), owner_nick):
        return None
    # "gains a Pirate Ship token" is a token, not a card.
    if match.group("cards").endswith(" token"):
        return None
    return _parse_card_quantity_list(match.group("cards")) or None


def _parse_replay_continuation_line(line: str) -> tuple[CardQuantity, ...] | None:
    """Match "... and plays {a Cardname} again." / "... and plays the
    {Cardname} a third time." (module docstring's CONTINUATION LINES -
    Throne-Room/King's-Court replay narration - always the turn's own
    owner; no sampled line has ever named a nick, so no owner check).

    Private helper - single consumer is
    game_log_parser._parse_turn_block(). Counted as a play of that card
    (same field, cards_played, as _parse_play_line()'s own matches) -
    this pass makes no attempt to distinguish an "original" play from a
    replay.

    Inputs:
        line: one raw "... " continuation line.
    Output: _parse_card_quantity_list()'s result over the single card
        phrase (count=1), or None if line doesn't match this shape.
    Side effects: none.
    Exceptions: none.
    """
    match = _REPLAY_PATTERN.match(line.strip())
    if match is None:
        return None
    return _parse_card_quantity_list(match.group("card")) or None
