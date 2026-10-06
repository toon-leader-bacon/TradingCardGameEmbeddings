"""Parses the turn-by-turn `<hr/><b>Game log</b>` section of one
isotropic Flavor B game-log HTML file - the part header_parser.py's own
module docstring explicitly leaves untouched. Composes on top of
header_parser.parse_game_header() (NOT modified by this module) rather
than re-deriving anything header_parser.py already extracts (winner,
kingdom, pile exhaustion, per-player score/opening buy).

TURN-BLOCK SPLITTING/HEADER PARSING AND THE PUBLIC parse_game_log()
ENTRY POINT LIVE HERE; THE PER-PLAYER ACTION/CONTINUATION-LINE GRAMMAR
LIVES IN ITS OWN SIBLING MODULE: _continuation_line_parsers.py holds
the play/buy/trash/gain/replay line matchers (_parse_play_line()
through _parse_replay_continuation_line()) plus CardQuantity, the data
holder they all build and Turn's own fields carry - split out purely
to keep this file to one concern (splitting the log into turn blocks
and driving them) rather than also carrying that line-level grammar.

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

ACTION/CONTINUATION LINE GRAMMAR: see _continuation_line_parsers.py's
own module docstring for exactly which action-line and continuation-
line shapes are captured (plays, buys, trashes, gains, replays) versus
deliberately ignored (reveals, discards, draws, shuffles, "puts back",
an explicit-other-nick gain/trash) - that detail lives there now, next
to the matchers that implement it.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from src.data_refinement.metrics.isotropic.games._continuation_line_parsers import (
    CardQuantity,
    _parse_buy_line,
    _parse_gain_continuation_line,
    _parse_play_line,
    _parse_replay_continuation_line,
    _parse_trash_and_gain_continuation_line,
    _parse_trash_continuation_line,
)
from src.data_refinement.metrics.isotropic.games.header_parser import (
    GameHeader,
    parse_game_header,
)

_GAME_LOG_MARKER_LINE = "Game log"

# "— {nick}'s turn {N} —" (2013 era) or "--- {nick}'s turn ---" (2010 era,
# no number) - matched against an already-stripped line. Nicks may hold
# spaces and punctuation, so the nick group is a greedy "anything".
_TURN_HEADER_PATTERN = re.compile(r"^(?:—|---) (?P<nick>.+)'s turn(?: \d+)? (?:—|---)$")


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
