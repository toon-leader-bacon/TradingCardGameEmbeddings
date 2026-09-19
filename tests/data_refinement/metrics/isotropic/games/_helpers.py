"""Shared, non-test helpers for isotropic/games' test files.

Reuses summary/'s own binder_from_card_names()/card_uuid() - building a
real dominiontabs CardBinder is a Flavor-A-agnostic concern, so there's
no reason for games/ tests to duplicate it.
"""

from src.data_refinement.metrics.isotropic.games.game_log_parser import (
    CardQuantity,
    GameLog,
    Turn,
)
from src.data_refinement.metrics.isotropic.games.header_parser import (
    GameHeader,
    GameHeaderPlayer,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (  # noqa: F401
    binder_from_card_names,
    card_uuid,
)


def game_header_player(
    nick: str,
    score: int,
    turns: int,
    opening_buy_names: tuple[str | None, str | None] = ("Silver", None),
) -> GameHeaderPlayer:
    return GameHeaderPlayer(
        nick=nick, score=score, turns=turns, opening_buy_names=opening_buy_names
    )


def game_header(
    winner_nick: str,
    kingdom_card_names: tuple[str, ...],
    players: tuple[GameHeaderPlayer, ...],
    exhausted_pile_names: tuple[str, ...] = ("Provinces",),
    is_natural_kingdom: bool = True,
) -> GameHeader:
    return GameHeader(
        winner_nick=winner_nick,
        is_natural_kingdom=is_natural_kingdom,
        exhausted_pile_names=exhausted_pile_names,
        kingdom_card_names=kingdom_card_names,
        players=players,
    )


def _oxford_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_header_html(
    winner_nick: str,
    pile_sentence: str,
    kingdom_card_names: list[str],
    players: list[tuple[str, int, int, str | None, str | None]],
    extra_kingdom_lines: list[str] = (),
) -> str:
    """Build a minimal, plain-text (no <span>/<b> markup - see
    header_parser.py's own module docstring for why bs4's get_text()
    makes that markup irrelevant to every parsing function this
    fixture exercises) Flavor B HTML fixture through the second
    "----------------------" separator - real files continue with
    trash:/league game:/<hr/><b>Game log</b> after that point, but
    nothing under test here ever reads past the second separator.

    Inputs:
        winner_nick: e.g. "undertoe".
        pile_sentence: the raw second line, e.g. "All Provinces are
            gone." or "Chapel, Monument, and Peddler are all gone."
        kingdom_card_names: the kingdom, in supply-list order.
        players: (nick, score, turns, first_buy, second_buy_or_None)
            tuples, one per player block, in order.
        extra_kingdom_lines: optional constraint/point-tracker marker
            lines to insert between the kingdom line and the separator.
    Output: a full `<html>...<pre>...</pre>...</html>` string.
    """
    lines = [
        f"{winner_nick} wins!",
        pile_sentence,
        "",
        f"cards in supply: {_oxford_join(kingdom_card_names)}",
        *extra_kingdom_lines,
        "----------------------",
        "",
    ]
    for nick, score, turns, first_buy, second_buy in players:
        first_buy_text = first_buy if first_buy is not None else "nothing"
        second_buy_text = second_buy if second_buy is not None else "nothing"
        lines.append(f"{nick}: {score} points (details); {turns} turns")
        lines.append(f"opening: {first_buy_text} / {second_buy_text}")
        lines.append("[N cards] details")
        lines.append("")
    lines.append("----------------------")
    lines.append("")
    lines.append("trash: details")
    lines.append("league game: no")

    body = "\n".join(lines)
    return f"<html><head></head><body><pre>{body}</pre></body></html>"


def build_game_log_html(
    log_lines: list[str],
    winner_nick: str = "a",
    players: list[tuple[str, int, int, str | None, str | None]] | None = None,
    kingdom_card_names: list[str] | None = None,
) -> str:
    """Build a full Flavor B fixture: build_header_html()'s header, then a
    "Game log" section made of log_lines (already-formatted text lines)."""
    if players is None:
        players = [("a", 40, 20, "Witch", None), ("b", 30, 20, "Silver", None)]
    header_html = build_header_html(
        winner_nick,
        "All Provinces are gone.",
        kingdom_card_names or ["Witch", "Moat"],
        players,
    )
    log_body = "\n\nGame log\n\n" + "\n".join(log_lines) + "\n"
    return header_html.replace("</pre>", log_body + "</pre>")


def card_quantity(card_name_text: str, count: int = 1) -> CardQuantity:
    return CardQuantity(card_name_text=card_name_text, count=count)


def turn(
    player_nick: str,
    turn_number: int,
    bought: tuple[CardQuantity, ...] = (),
    gained: tuple[CardQuantity, ...] | None = None,
    trashed: tuple[CardQuantity, ...] = (),
    played: tuple[CardQuantity, ...] = (),
) -> Turn:
    """A Turn whose cards_gained defaults to its cards_bought (every buy is
    also a gain, as the real parser produces)."""
    return Turn(
        player_nick=player_nick,
        turn_number=turn_number,
        cards_bought=bought,
        cards_gained=bought if gained is None else gained,
        cards_trashed=trashed,
        cards_played=played,
    )


def game_log(header: GameHeader, turns: tuple[Turn, ...]) -> GameLog:
    return GameLog(header=header, turns=turns)
