"""Parses the turn-by-turn `<hr/><b>Game log</b>` section of one
isotropic Flavor B game-log HTML file - the part header_parser.py's own
module docstring explicitly leaves untouched. Composes on top of
header_parser.parse_game_header() (NOT modified by this module) rather
than re-deriving anything header_parser.py already extracts (winner,
kingdom, pile exhaustion, per-player score/opening buy).

SCOPE: BRAINSTORM.md's "New candidates raised during human review" #3-#7
(Next Buy Prediction, Next Trashed Card Prediction, Next-Turn Action
Count Prediction, Eventual Win Probability, Deck Pair -> Winner
Prediction - resignation metrics are a separate, still-deferred family)
only need, per player-turn: which cards were bought, which were gained
(a superset of bought - includes attack-forced gains, e.g. a Witch's
Curse), which were trashed, and which were played at all (this module
does NOT itself decide which of those are Action-typed - see
partial_deck.py's own module docstring for why that filtering happens
downstream, with real CardBinder access, not here). Every other
observed continuation-line shape (reveals, discard/draw/shuffle
bookkeeping, "puts back", Throne-Room-style replay narration) is
deliberately NOT captured by this pass's Turn dataclass - out of scope,
not silently mis-parsed. See this module's own confirmed-shape notes
below, gathered by direct sampling of 150 real files from
data/raw/isotropic/2013_20130315.tar.bz2 via the same
BeautifulSoup(html_text, "html.parser").find("pre").get_text()
technique header_parser.py already uses.

CONFIRMED GAME-LOG BODY SHAPE (after header_parser.py's own second
"----------------------" separator, "trash: ..." and "league game:
.../no" lines, then a blank line, then):

    Game log

    Turn order is {nick} and then {nick}[, {nick}, and {nick}].

    ({nick}'s first hand: {Oxford-comma card list}.)
    [one such line per player]

    — {nick}'s turn {N} —
    {action line}
    [... continuation line, 0+ per action line]
    [more action lines]

       — {other_nick}'s turn {N} —
       {action line}
    ...

Turn blocks repeat in turn order until the game ends. EACH PLAYER'S
WHOLE TURN BLOCK IS INDENTED BY 3 SPACES PER TURN-ORDER POSITION AFTER
THE FIRST (confirmed live: the second player's entire block, header
line included, is prefixed "   ", not just re-stated once) - purely
cosmetic, redundant with the nick already present in the turn-header
line itself, so _parse_turn_header() strips leading whitespace rather
than tracking position/indentation depth as meaningful state.

TWO ERAS (confirmed live, both surviving days): 2013 files use the
"— {nick}'s turn {N} —" header, an em dash on both sides, with "({nick}'s
first hand: ...)" lines before turn 1. 2010 files use "--- {nick}'s
turn ---" (literal dashes, NO turn number), no first-hand lines, and a
1- or 4-space indent. parse_game_log() therefore numbers each player's
turns by counting their blocks rather than reading a number off the
header; the counts equal header_parser's GameHeaderPlayer.turns for every
sampled game of both eras (0 mismatches over ~2,500 games).

MEASURED ACCURACY (final-deck check): each file's header also lists every
player's final deck ("[25 cards] 1 Bridge, ..."). Rebuilding it from
7 Coppers + 3 Estates + gained - trashed matches exactly for ~55% of
players (~65% ignoring Curses), across 1,500 2013 games. Nearly all the
misses are the documented interim drops below (cross-player gains and
trashes - see TODO.md) plus non-trash removals (Ambassador
returns) and Trader replacements; no matcher-level bug remains in the
sample. Do not "fix" the drops without revisiting that TODO.

`{N}` IN "— {nick}'s turn {N} —" IS THAT PLAYER'S OWN PERSONAL TURN
COUNT (confirmed live: "Strategyst's turn 1", "luzifer851's turn 1",
"Strategyst's turn 2", ... - each player's own counter, matching
header_parser.py's GameHeaderPlayer.turns field, NOT a shared
across-all-players round index) - this is what makes
partial_deck.py's "before turn T" framing well-defined per player
without needing to interleave both players' turns by wall-clock order.

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

from bs4 import BeautifulSoup

from src.data_refinement.metrics.isotropic.games.header_parser import (
    GameHeader,
    parse_game_header,
)

_GAME_LOG_MARKER_LINE = "Game log"

# "— {nick}'s turn {N} —" (2013 era) or "--- {nick}'s turn ---" (2010 era,
# no number) - matched against an already-stripped line. Nicks may hold
# spaces and punctuation, so the nick group is a greedy "anything".
_TURN_HEADER_PATTERN = re.compile(r"^(?:—|---) (?P<nick>.+)'s turn(?: \d+)? (?:—|---)$")

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


@dataclass(frozen=True)
class Turn:
    """One player's single turn, as parsed from their turn block (see
    module docstring's confirmed shape) - deliberately narrower than
    everything a turn block actually contains (see module docstring's
    SCOPE section for exactly what's dropped).

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    player_nick: str
    turn_number: int  # this player's OWN turn count - see module docstring.
    cards_bought: tuple[CardQuantity, ...]
    # Superset of cards_bought - every gain, bought or not (module
    # docstring's ACTION/CONTINUATION LINES notes).
    cards_gained: tuple[CardQuantity, ...]
    cards_trashed: tuple[CardQuantity, ...]
    # Every card played this turn, of ANY type (treasures included) -
    # see partial_deck.py's own module docstring for why Action-type
    # filtering is this field's consumer's job, not this parser's.
    cards_played: tuple[CardQuantity, ...]


@dataclass(frozen=True)
class GameLog:
    """One game's parsed header plus its full turn-by-turn body.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    header: GameHeader
    turns: tuple[Turn, ...]


def parse_game_log(html_text: str) -> GameLog | None:
    """Parse one Flavor B HTML file's content into a GameLog.

    Inputs:
        html_text: the full raw content of one Flavor B HTML file (see
            header_parser.parse_game_header()'s own docstring for where
            these live).
    Output: a GameLog, or None under any condition
        header_parser.parse_game_header() itself already returns None
        for (module docstring there - no winner, no <pre>, a
        partial-resignation pile-exhaustion line), OR if no "Game log"
        marker line is found at all - a real "not this pass's shape"
        outcome, not a parse error, matching header_parser.py's own
        None-not-exception convention for out-of-scope files.
    Side effects: none.
    Exceptions: none expected for a well-formed file of either known
        era; a genuinely malformed file may raise from list indexing or
        a private helper's own assertion - not caught here, matching
        header_parser.parse_game_header()'s own documented expectation
        that a caller driving many files (scanner.py) isolates this
        per-file, not this function itself.

    Example:
        >>> game_log = parse_game_log(open("game-...html").read())
        >>> game_log.turns[0].player_nick
        'Strategyst'
        >>> game_log.turns[0].cards_bought
        (CardQuantity(card_name_text='Silver', count=1),)
    """
    header = parse_game_header(html_text)
    if header is None:
        return None

    soup = BeautifulSoup(html_text, "html.parser")
    pre = soup.find("pre")
    if pre is None:
        return None
    lines = pre.get_text().splitlines()

    log_start = _find_game_log_start_index(lines)
    if log_start is None:
        return None

    # Number each player's turns by counting their blocks - the 2013-era
    # header carries an explicit number but the 2010-era one does not
    # (module docstring), so counting is the one rule that works for both.
    turns: list[Turn] = []
    turn_counts: dict[str, int] = {}
    for block in _split_into_turn_blocks(lines[log_start + 1 :]):
        owner_nick = _parse_turn_header(block[0])
        if owner_nick is None:
            continue
        turn_counts[owner_nick] = turn_counts.get(owner_nick, 0) + 1
        turn = _parse_turn_block(block, turn_counts[owner_nick])
        if turn is not None:
            turns.append(turn)
    return GameLog(header=header, turns=tuple(turns))


def _find_game_log_start_index(lines: list[str]) -> int | None:
    """Find the index of the "Game log" marker line.

    Private helper - single consumer is parse_game_log().

    Inputs:
        lines: pre.get_text(), split into lines.
    Output: the index of the (stripped) line equal to
        _GAME_LOG_MARKER_LINE, or None if not found - a real
        "not this pass's shape" outcome, not a parse error.
    Side effects: none.
    Exceptions: none.
    """
    for index, line in enumerate(lines):
        if line.strip() == _GAME_LOG_MARKER_LINE:
            return index
    return None


def _split_into_turn_blocks(lines: list[str]) -> list[list[str]]:
    """Split the game-log body (everything after the "Game log" marker
    line) into one block of lines per player-turn.

    Private helper - single consumer is parse_game_log(). Drops the
    leading "Turn order is ..." and "({nick}'s first hand: ...)"
    preamble lines (module docstring) - neither carries per-turn state
    this pass captures. Each returned block's own first line is its
    "— {nick}'s turn {N} —" header line (whitespace-stripped - module
    docstring's indentation note); a block ends at the next such header
    line or end of input.

    Inputs:
        lines: every line from just after "Game log" to the end of the
            file.
    Output: one block per player-turn, in log order (turn order, not
        grouped by player).
    Side effects: none.
    Exceptions: none.
    """
    blocks: list[list[str]] = []
    for line in lines:
        if _parse_turn_header(line) is not None:
            blocks.append([line])
        elif blocks and line.strip():
            blocks[-1].append(line)
    return blocks


def _parse_turn_header(line: str) -> str | None:
    """Parse one "— {nick}'s turn {N} —" (2013 era) or "--- {nick}'s
    turn ---" (2010 era, no number) line (module docstring), after
    stripping leading indentation.

    Private helper - consumed by _split_into_turn_blocks() (to find
    block boundaries) and parse_game_log() (to read each block's owner
    nick). The explicit turn number is deliberately NOT returned: the
    2010 era has none, so parse_game_log() numbers turns by counting
    each nick's blocks instead.

    Inputs:
        line: one raw line, possibly indented (module docstring's
            per-turn-order-position indentation note).
    Output: the owner nick, or None if line doesn't match this shape
        at all (an ordinary non-header line, not an error).
    Side effects: none.
    Exceptions: none.
    """
    match = _TURN_HEADER_PATTERN.match(line.strip())
    return match.group("nick") if match else None


def _parse_turn_block(block_lines: list[str], turn_number: int) -> Turn | None:
    """Parse one player-turn's full block (header line plus every
    action/continuation line under it) into a Turn.

    SCOPE NARROWED TO THIS BLOCK'S OWN OWNER - NOT EVERY NICK A
    CONTINUATION LINE NAMES: the module docstring's own CONTINUATION
    LINES section documents that some continuation lines name an
    EXPLICIT DIFFERENT nick than the turn's owner (e.g. an attack
    forcing an opponent to gain a Curse) - but a Turn only has ONE
    player_nick, and attributing that other player's gain/trash to
    THEIR OWN eventual turn correctly would need tracking a global,
    cross-player event ordering this pass's Turn/GameLog shape doesn't
    have. THIS IS A DOCUMENTED INTERIM DECISION, RAISED TO THE HUMAN
    EXPLICITLY RATHER THAN SILENTLY BAKED IN - revisit if the Turn/
    GameLog data model changes to support cross-player attribution
    (e.g. giving each CardQuantity its own subject nick plus a global
    event-sequence index), not a placeholder pending an answer before
    this shape can be implemented as-is. Given that, every per-line
    matcher below (_parse_*_line()) only returns a real result when a
    line names NO explicit nick, or explicitly names THIS block's own
    owner (see _line_belongs_to_owner()) - a line naming a different
    explicit nick is deliberately treated the same as an out-of-scope
    continuation shape (reveals, discards, etc.): skipped, not
    misattributed to the wrong player. Concretely, this means
    attack-forced gains onto an opponent (the single biggest real-world
    case - Witch's Curse) are NOT reflected in that opponent's partial
    deck under this pass, a real accuracy gap worth naming explicitly
    rather than silently accepting.

    Private helper - single consumer is parse_game_log().

    Inputs:
        block_lines: one _split_into_turn_blocks() entry -
            block_lines[0] is the turn header line.
        turn_number: this owner's own 1-based turn count, assigned by
            parse_game_log().
    Output: a Turn, or None if block_lines[0] itself doesn't match
        _parse_turn_header()'s shape (a genuinely malformed block).
    Side effects: none.
    Exceptions: none expected for a well-formed block.
    """
    owner_nick = _parse_turn_header(block_lines[0])
    if owner_nick is None:
        return None

    cards_bought: list[CardQuantity] = []
    cards_gained: list[CardQuantity] = []
    cards_trashed: list[CardQuantity] = []
    cards_played: list[CardQuantity] = []

    # Try every confirmed line shape (module docstring's ACTION LINES /
    # CONTINUATION LINES notes) against each line under the header, in
    # a fixed order - the shapes are mutually exclusive by construction
    # (each has a distinguishing verb/prefix), so at most one matcher
    # ever accepts a given line.
    for line in block_lines[1:]:
        played = _parse_play_line(line, owner_nick)
        if played is not None:
            cards_played.extend(played)
            continue

        bought = _parse_buy_line(line, owner_nick)
        if bought is not None:
            cards_bought.extend(bought)
            cards_gained.extend(bought)  # every buy is also a gain
            continue

        trashed_and_gained = _parse_trash_and_gain_continuation_line(line, owner_nick)
        if trashed_and_gained is not None:
            trashed_quantities, gained_quantities = trashed_and_gained
            cards_trashed.extend(trashed_quantities)
            cards_gained.extend(gained_quantities)
            continue

        trashed = _parse_trash_continuation_line(line, owner_nick)
        if trashed is not None:
            cards_trashed.extend(trashed)
            continue

        gained = _parse_gain_continuation_line(line, owner_nick)
        if gained is not None:
            cards_gained.extend(gained)
            continue

        replayed = _parse_replay_continuation_line(line)
        if replayed is not None:
            cards_played.extend(replayed)
            continue

        # Every other confirmed shape (reveals, discards, draws,
        # shuffles, "puts back", an explicit-OTHER-nick gain/trash - see
        # this function's own docstring) - module docstring's SCOPE
        # section - is deliberately ignored, not an error.

    return Turn(
        player_nick=owner_nick,
        turn_number=turn_number,
        cards_bought=tuple(cards_bought),
        cards_gained=tuple(cards_gained),
        cards_trashed=tuple(cards_trashed),
        cards_played=tuple(cards_played),
    )


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
    situation card_resolution.card_uuid_for_name() and this module's own
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
    _parse_turn_block()'s own SCOPE NARROWED convention (see that
    function's docstring).

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
        per _parse_turn_block()'s own documented interim decision).
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

    Private helper - single consumer is _parse_turn_block(). A play
    line always names the turn's own owner explicitly (module
    docstring: "all belong to the turn's own owner, never an explicit
    different nick") - unlike the continuation-line matchers below,
    this one checks its captured nick against owner_nick directly, not
    via _line_belongs_to_owner() (which exists for the three-way none/
    same/different gate a continuation line needs, not this shape's
    plain two-way check). Delegates the actual card-list text to
    _parse_card_quantity_list().

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

    Private helper - single consumer is _parse_turn_block(). Same
    plain-equality nick check as _parse_play_line() (not
    _line_belongs_to_owner() - see that function's own docstring for
    why). Delegates the card-list text to _parse_card_quantity_list().

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

    Private helper - single consumer is _parse_turn_block(). Tried
    BEFORE _parse_trash_continuation_line() in _parse_turn_block()'s
    own dispatch order, since a plain trash matcher would otherwise
    partially match this line's own leading "trashes ..." clause and
    silently drop its trailing ", gaining ... in hand" clause. Gates its
    captured (or absent) nick via _line_belongs_to_owner(); delegates
    both the trashed-card-list and the gained-card text to
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

    Private helper - single consumer is _parse_turn_block(). Gates its
    captured (or absent) nick via _line_belongs_to_owner(); delegates
    the card-list text to _parse_card_quantity_list().

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

    Private helper - single consumer is _parse_turn_block(). Gates its
    captured (or absent) nick via _line_belongs_to_owner(); delegates
    the card-list text to _parse_card_quantity_list().

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

    Private helper - single consumer is _parse_turn_block(). Counted as
    a play of that card (same field, cards_played, as
    _parse_play_line()'s own matches) - this pass makes no attempt to
    distinguish an "original" play from a replay.

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
