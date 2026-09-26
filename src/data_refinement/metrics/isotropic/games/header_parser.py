"""Parses the HEADER block of one isotropic Flavor B game-log HTML file
(see ../BRAINSTORM.md's "Flavor B" section) into a small, strongly
typed GameHeader - deliberately NOT the turn-by-turn `<hr/><b>Game
log</b>` section below it, which game_log_parser.py parses on top of this
module (this module is not modified by that one). The six header metrics
(opening-buy, pile-exhaustion/game-ending) need nothing from the log body.

USES BEAUTIFULSOUP, like every other HTML raw source in this project
(../../../deck_box/fabtcg_decklists/fragment_parsing.py,
../../../card_binder/gwent_one/ingestion_stage.py). Hand-rolled regex
tag stripping would mis-parse real files: they contain HTML entities
(e.g. `&mdash;` as the 2013-era turn-boundary marker; apostrophes in
card names like "Fool's Gold" are literal). `Tag.get_text()` decodes
entities and strips every nested tag uniformly, in one call, regardless
of which of the two confirmed span-attribute shapes below a given file
uses.

CONFIRMED HEADER SHAPE (direct sampling, both surviving days -
2010-10-11's 108 games and 2013-03-15's 10,815 games - via
BeautifulSoup(html_text, "html.parser").find("pre").get_text()):

    {winner_nick} wins!
    {pile-exhaustion sentence}

    cards in supply: {Oxford-comma card list}
    {0+ optional lines: "Constraint(s) used: ...", "The point tracker
    will be available.", "All players get identical starting hands.",
    "Default card selection was used." (the last one confirmed as the
    natural-kingdom complement of "Constraint(s) used:" - present when
    NO constraint applied; already correctly ignored by
    _CONSTRAINT_LINE_PREFIXES since it matches none of those prefixes)}
    ----------------------

    {player block}
    [blank line between multiple player blocks]
    {player block}

    ----------------------

    trash: ...
    league game: yes/no

    <hr/><b>Game log</b>
    ...

TWO CLIENT ERAS, ONE PARSER, NO SPECIAL-CASING NEEDED: 2013's files
sometimes tag card spans with a `cardname="X"` attribute (e.g. `<span
cardname="Cellar" ...>Cellar</span>`) that 2010's files never have, and
2013's player-block header line is sometimes prefixed `#N ` (rank)
before the nick (`<b>#1 babykmama</b>:`) where 2010's is always bare
(`<b>undertoe</b>:`) - confirmed inconsistent even within 2013's own
files (the SAME file's kingdom line can carry `cardname=` while its
very next `opening:` line doesn't). None of this matters to the parsing
logic below: `get_text()` already discards every tag and its
attributes uniformly, so `_parse_player_blocks()`'s own regex only ever
sees plain text either way - it still has to handle the OPTIONAL `#N `
rank prefix, but never has to know or care whether the surrounding
markup carried a `cardname=` attribute.

RESIGNED/NO-WINNER GAMES ARE OUT OF SCOPE FOR THIS PASS: a game that
ends in resignation has no "{nick} wins!" first line (BRAINSTORM.md's
"More Flavor B-only candidates" section covers resignation metrics
separately, not built in this pass) - parse_game_header() returns None
for any file whose first line doesn't match that shape, a real "not
this pass's shape" outcome, not a parse error.

KINGDOM-CONSTRAINT DETECTION, FLAVOR-B STYLE: unlike Flavor A's
structured `board.constraints`/`required`/`prohibited`/etc. booleans,
Flavor B renders the same generator settings as free-text lines
between "cards in supply:" and the first "----------------------"
separator - confirmed: "Constraint(s) used: required: ...",
"Constraint(s) used: prohibited: ...", "All players get identical
starting hands.". "The point tracker will be available." is also one
of these lines but, mirroring ../summary/row_utils.py's
is_natural_kingdom() treatment of board.point_tracker, is NOT treated
as an unnatural-kingdom marker - it reflects which CARDS are in the
kingdom, not a constraint on which ten got dealt.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

_CONSTRAINT_LINE_PREFIXES: tuple[str, ...] = (
    "Constraint(s) used:",
    "All players get identical starting hands.",
)

_SEPARATOR_LINE = "----------------------"
_BANE_MARKER = "♦"

# Optional "#N " rank prefix (module docstring's TWO CLIENT ERAS
# section) before the nick, then "{nick}: {score} points (...);
# {turns} turns" - the "(...)" VP breakdown is deliberately not
# captured, since no metric in this pass reads it. Score can be
# negative (a heavily-cursed loser, confirmed live: "-2 points") and
# "point"/"turn" singularize when the count is exactly 1 (confirmed
# live: "1 point (an Estate); 14 turns" alongside plenty of "1 turn"
# occurrences too) - both plurals are optional, not assumed.
_PLAYER_LINE_PATTERN = re.compile(
    r"^(?:#\d+ )?(.+): (-?\d+) points? \(.*\); (\d+) turns?$"
)
_OPENING_LINE_PATTERN = re.compile(r"^opening: (.+) / (.+)$")

# A non-winning player who resigned mid-game - confirmed live to
# appear even in a game with a genuine winner (see
# _parse_player_blocks()'s own docstring); this player's block is
# skipped, not parsed for a score that was never recorded.
_RESIGNED_PLAYER_LINE_PATTERN = re.compile(
    r"^(?:#\d+ )?.+: resigned \(.+\); \d+ turns?$"
)


@dataclass(frozen=True)
class GameHeaderPlayer:
    """One player's header block - see module docstring's confirmed shape.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nick: str
    score: int
    turns: int
    # A slot is None for "opening: X / nothing" AND "opening: nothing / X"
    # (a $2 first turn buys nothing, confirmed live in both eras) -
    # verbatim string "nothing", never a card name meaning "no buy".
    opening_buy_names: tuple[str | None, str | None]


@dataclass(frozen=True)
class GameHeader:
    """One game's parsed header - see module docstring's confirmed shape.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    winner_nick: str
    # Populated by the parser itself (see module docstring's KINGDOM-
    # CONSTRAINT DETECTION section) - there is no separate row_utils
    # function to derive this after the fact the way
    # ../summary/row_utils.py's is_natural_kingdom() does, since Flavor
    # B has no structured board dict to re-inspect; the parser already
    # visits this exact text region once, so it sets the flag there.
    is_natural_kingdom: bool
    # 1 card (a single Province/Colony-style pile exhaustion) or more
    # (a multi-pile ending - confirmed always exactly 3 in a large
    # live sample, but this field does not assume that count).
    # ENGLISH-PLURALIZED, NOT PRE-SINGULARIZED: isotropic's own "X, Y,
    # and Z are all gone"/"All X are gone" sentence renders each pile
    # by its plural card name ("Provinces", "Duchies", "Wharves" for
    # Wharf), and a large live sample confirmed at least one card name
    # ("Ironworks") that ISN'T further pluralized at all - a plain
    # "strip trailing s" or "-y/-ies" rule cannot invert every observed
    # case losslessly (Wharves needs "-ves"->"f"; Ironworks needs no
    # change at all). Resolving one of these names to a nocab_uuid is
    # therefore NOT a plain row_utils.card_uuid_for_name() call - see
    # row_utils.pile_card_uuid_for_name(), which tries several
    # depluralization candidates against the real CardBinder rather
    # than guessing once.
    exhausted_pile_names: tuple[str, ...]
    kingdom_card_names: tuple[str, ...]
    players: tuple[GameHeaderPlayer, ...]


def parse_game_header(html_text: str) -> GameHeader | None:
    """Parse one Flavor B HTML file's content into a GameHeader.

    Inputs:
        html_text: the full raw content of one
            `data/raw/isotropic/201010_11_all/game-*.html` or
            `.../20130315/game-*.html` file (see ../BRAINSTORM.md for
            where these live - two tar.bz2 archives, one HTML member
            per game).
    Output: a GameHeader, or None if html_text has no `<pre>` element
        at all, its first line doesn't match "{nick} wins!" (module
        docstring's RESIGNED/NO-WINNER GAMES note), its second line
        isn't a real pile-exhaustion sentence (confirmed live: a game
        WITH a real winner can still show "All but one player has
        resigned." here instead, when a losing player quit mid-game
        even though the eventual winner played it out - a partial-
        resignation case just as out of this pass's scope as a full
        one), or the expected separator structure isn't found - all
        real "not this pass's shape" outcomes, not exceptions.
    Side effects: none.
    Exceptions: none expected for a well-formed file of either known
        era; a genuinely malformed file may raise from list indexing
        inside a private helper, or from BeautifulSoup itself on
        thoroughly broken markup - not caught here, since a caller
        driving many files (../scanner.py) is expected to isolate that
        per-file the same way ../../sts_gg/scanner.py isolates a
        per-row accumulate() failure, not this function itself.

    Example:
        >>> header = parse_game_header(open("game-...html").read())
        >>> header.winner_nick
        'undertoe'
        >>> header.exhausted_pile_names
        ('Chapel', 'Monument', 'Peddler')
    """
    soup = BeautifulSoup(html_text, "html.parser")
    pre = soup.find("pre")
    if pre is None:
        return None

    lines = pre.get_text().splitlines()

    winner_nick = _parse_winner_nick(lines[0]) if lines else None
    if winner_nick is None:
        return None

    exhausted_pile_names = _parse_exhausted_pile_names(lines[1])
    if exhausted_pile_names is None:
        return None

    separators = _find_separator_line_indices(lines)
    if separators is None:
        return None
    first_separator, second_separator = separators

    # lines[2] is always the blank line between the pile-exhaustion
    # sentence and "cards in supply:" - confirmed live, never absent.
    kingdom_card_names, is_natural_kingdom = _parse_kingdom_section(
        lines[3:first_separator]
    )
    players = _parse_player_blocks(lines[first_separator + 1 : second_separator])

    return GameHeader(
        winner_nick=winner_nick,
        is_natural_kingdom=is_natural_kingdom,
        exhausted_pile_names=exhausted_pile_names,
        kingdom_card_names=kingdom_card_names,
        players=players,
    )


def _parse_winner_nick(first_line: str) -> str | None:
    """Extract the winner's nick from the header's first line.

    Private helper - single consumer is parse_game_header().

    Inputs:
        first_line: lines[0] of pre.get_text(), e.g. "undertoe wins!"
            or, for a resigned-out game, something else entirely (e.g.
            "All but one player has resigned.").
    Output: the nick, or None if first_line doesn't match the
        "{nick} wins!" shape (module docstring's RESIGNED/NO-WINNER
        GAMES note - a real, expected non-match for this pass's scope,
        not a parse error).
    Side effects: none.
    Exceptions: none.
    """
    match = re.match(r"^(.+) wins!$", first_line)
    return match.group(1) if match else None


def _parse_exhausted_pile_names(pile_line: str) -> tuple[str, ...] | None:
    """Extract the exhausted pile name(s) from the second header line.

    Private helper - single consumer is parse_game_header(). Handles
    both confirmed sentence shapes (module docstring): "All {card} are
    gone." (exactly one pile - a Province/Colony-style ending) and
    "{card}, {card}, and {card} are all gone." (an Oxford-comma list of
    two or more piles) - a shape-aware split on already-plain text (no
    tag stripping needed here; pre.get_text() already handled it),
    never assuming exactly 3 items even though that's what a large live
    sample always showed.

    Inputs:
        pile_line: lines[1] of pre.get_text().
    Output: one or more pile names, in the order the sentence lists
        them (order carries no meaning - a set, not a ranking - but a
        tuple is used for GameHeader field consistency with
        kingdom_card_names) - still English-pluralized, exactly as
        rendered (see GameHeader.exhausted_pile_names' own field
        comment for why this deliberately does not singularize) - or
        None if pile_line matches neither shape, confirmed live to be
        a real outcome: a game with a genuine winner can still show
        "All but one player has resigned." here when some OTHER player
        quit mid-game (see parse_game_header()'s own docstring) - not
        a pile-exhaustion ending at all, out of this pass's scope.
    Side effects: none.
    Exceptions: none.
    """
    if pile_line.startswith("All ") and pile_line.endswith(" are gone."):
        card_name = pile_line[len("All ") : -len(" are gone.")]
        return (card_name,)

    if pile_line.endswith(" are all gone."):
        card_list_text = pile_line[: -len(" are all gone.")]
        return _split_oxford_comma_list(card_list_text)

    return None


def _find_separator_line_indices(lines: list[str]) -> tuple[int, int] | None:
    """Find the indices of the first two `----------------------`
    lines - the boundary between the kingdom section and the player
    blocks, and between the player blocks and the trailing
    trash/league-game/log lines.

    Private helper - single consumer is parse_game_header(). Note this
    is a run of literal dash characters rendered as plain text inside
    `<pre>`, NOT the real `<hr/>` element that separates the header
    from the `<b>Game log</b>` section further down - the two are
    unrelated boundaries, and this function only ever looks for the
    text one.

    Inputs:
        lines: pre.get_text(), split into lines.
    Output: (first_separator_index, second_separator_index), or None
        if fewer than two separator lines are found - a real "this
        file doesn't have the shape this pass expects" outcome.
    Side effects: none.
    Exceptions: none.
    """
    separator_indices = [
        index for index, line in enumerate(lines) if line == _SEPARATOR_LINE
    ]
    if len(separator_indices) < 2:
        return None
    return separator_indices[0], separator_indices[1]


def _parse_kingdom_section(section_lines: list[str]) -> tuple[tuple[str, ...], bool]:
    """Parse the kingdom-card list and detect any generator-constraint
    marker line, from the lines between the pile-exhaustion line and
    the first separator.

    Private helper - single consumer is parse_game_header(). See
    module docstring's KINGDOM-CONSTRAINT DETECTION section for the
    exact marker line prefixes this checks (_CONSTRAINT_LINE_PREFIXES)
    and why "The point tracker will be available."/"Default card
    selection was used." are deliberately NOT among them.

    Inputs:
        section_lines: lines[3:first_separator_index] - starts with
            "cards in supply: ..." and may be followed by 0 or more
            constraint/point-tracker/identical-hands/default-selection
            marker lines.
    Output: (kingdom_card_names, is_natural_kingdom) - the Oxford-
        comma card list off the "cards in supply:" line, and whether
        none of section_lines' remaining lines start with a
        constraint marker prefix.
    Side effects: none.
    Exceptions: none.
    """
    kingdom_line = section_lines[0]
    card_list_text = kingdom_line[len("cards in supply: ") :]
    # A trailing "♦" marks the Young Witch bane pile (confirmed live);
    # it is part of the rendering, not the card name.
    kingdom_card_names = tuple(
        name.removesuffix(_BANE_MARKER)
        for name in _split_oxford_comma_list(card_list_text)
    )

    is_natural_kingdom = not any(
        line.startswith(prefix)
        for line in section_lines[1:]
        for prefix in _CONSTRAINT_LINE_PREFIXES
    )
    return kingdom_card_names, is_natural_kingdom


def _parse_player_blocks(section_lines: list[str]) -> tuple[GameHeaderPlayer, ...]:
    """Parse every player's block from the lines between the two
    separators.

    Private helper - single consumer is parse_game_header(). Handles
    both the bare (`{nick}: {score} points (...); {turns} turns`) and
    rank-prefixed (`#{N} {nick}: ...`) header-line shapes confirmed
    across the two eras (module docstring) - already plain text by
    this point (pre.get_text() already dropped the `<b>...</b>`
    wrapping either shape came in), so this only needs a regex over
    plain text, never markup. The rank number itself is discarded, not
    stored on GameHeaderPlayer, since it's redundant with score
    ordering and unused by every metric this pass builds.

    A RESIGNED PLAYER'S BLOCK IS SKIPPED, NOT PARSED: confirmed live -
    a game can have a genuine winner (parse_game_header() already
    verified a real "{nick} wins!" line and a real pile-exhaustion
    sentence) while a DIFFERENT, non-winning player still resigned
    along the way; that player's own header line reads
    "{nick}: resigned ({place}); {turns} turns" instead of the normal
    "... points (...); N turns" shape, with no score to record at all
    - matching ../summary/row_utils.py's eligible_player_entries()'s
    own "drop a resigned player, keep the rest of the game" choice for
    Flavor A, this drops just that one player's block rather than
    rejecting the whole game (which parse_game_header() already
    confirmed has a valid winner and ending).

    Inputs:
        section_lines: lines[first_separator_index + 1
            : second_separator_index] - one blank-line-separated block
            per player, each block's own "opening: X / Y-or-nothing"
            line immediately following its "... points (...); N turns"
            (or "... resigned (...); N turns") line.
    Output: one GameHeaderPlayer per non-resigned player block, in the
        blocks' own order (not sorted by score/rank) - may be shorter
        than the number of blocks in section_lines if any player
        resigned.
    Side effects: none.
    Exceptions: raises AssertionError (with the offending line) if a
        non-resigned block's first or second line doesn't match either
        expected pattern - a genuinely malformed block, not one of the
        two confirmed shapes this function otherwise handles; the
        assertions exist to give mypy (and a future debugger) a named
        failure point instead of a bare AttributeError on `None`.
    """
    players: list[GameHeaderPlayer] = []
    for block in _split_into_blocks(section_lines):
        if _RESIGNED_PLAYER_LINE_PATTERN.match(block[0]):
            continue

        player_match = _PLAYER_LINE_PATTERN.match(block[0])
        assert player_match is not None, f"malformed player line: {block[0]!r}"
        nick = player_match.group(1)
        score = int(player_match.group(2))
        turns = int(player_match.group(3))

        opening_match = _OPENING_LINE_PATTERN.match(block[1].strip())
        assert opening_match is not None, f"malformed opening line: {block[1]!r}"
        first_buy_name = _none_if_nothing(opening_match.group(1))
        second_buy_name = _none_if_nothing(opening_match.group(2))

        players.append(
            GameHeaderPlayer(
                nick=nick,
                score=score,
                turns=turns,
                opening_buy_names=(first_buy_name, second_buy_name),
            )
        )
    return tuple(players)


def _none_if_nothing(opening_slot_text: str) -> str | None:
    """Map isotropic's literal "nothing" opening-buy slot text to None.

    Private helper - single consumer is _parse_player_blocks().

    Inputs:
        opening_slot_text: one side of "opening: X / Y".
    Output: None for "nothing", otherwise the text unchanged.
    Side effects: none.
    Exceptions: none.
    """
    return None if opening_slot_text == "nothing" else opening_slot_text


def _split_oxford_comma_list(text: str) -> tuple[str, ...]:
    """Split an Oxford-comma-joined English list into its items.

    Private helper - consumed by _parse_exhausted_pile_names() (the
    multi-pile branch) and _parse_kingdom_section().

    Inputs:
        text: e.g. "A, B, and C", "A and B", or a single "A" (no
            conjunction at all).
    Output: ("A", "B", "C"), ("A", "B"), or ("A",) respectively.
    Side effects: none.
    Exceptions: none.
    """
    if ", and " in text:
        head, last_item = text.rsplit(", and ", 1)
        return tuple(head.split(", ")) + (last_item,)
    if " and " in text:
        first_item, last_item = text.rsplit(" and ", 1)
        return (first_item, last_item)
    return (text,)


def _split_into_blocks(lines: list[str]) -> list[list[str]]:
    """Split a list of lines into blank-line-separated blocks.

    Private helper - single consumer is _parse_player_blocks().

    Inputs:
        lines: e.g. player-block section lines, some blank ("")
            entries separating one player's block from the next.
    Output: one list per non-empty run of consecutive non-blank lines,
        in original order - blank lines themselves are dropped, never
        appear inside a returned block.
    Side effects: none.
    Exceptions: none.
    """
    blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        current_block.append(line)
    if current_block:
        blocks.append(current_block)
    return blocks
