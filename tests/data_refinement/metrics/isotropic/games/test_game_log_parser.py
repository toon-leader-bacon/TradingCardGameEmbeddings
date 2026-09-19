from src.data_refinement.metrics.isotropic.games.game_log_parser import (
    CardQuantity,
    _line_belongs_to_owner,
    _parse_card_quantity_list,
    _parse_turn_block,
    _parse_turn_header,
    parse_game_log,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    build_game_log_html,
    card_quantity,
)


def _two_player_log(a_lines: list[str], b_lines: list[str] | None = None):
    lines = [
        "Turn order is a and then b.",
        "",
        "— a's turn 1 —",
        *a_lines,
        "",
        "   — b's turn 1 —",
        *(b_lines or ["   b buys a Silver."]),
    ]
    game_log = parse_game_log(build_game_log_html(lines))
    assert game_log is not None
    return game_log


def test_none_when_header_is_out_of_scope() -> None:
    html = "<html><body><pre>All but one player has resigned.\n</pre></body></html>"
    assert parse_game_log(html) is None


def test_none_when_no_game_log_marker() -> None:
    html = build_game_log_html([]).replace("Game log", "Something else")
    assert parse_game_log(html) is None


def test_parses_turns_in_log_order_with_per_player_numbers() -> None:
    lines = [
        "— a's turn 1 —",
        "a buys a Witch.",
        "",
        "   — b's turn 1 —",
        "   b buys a Silver.",
        "",
        "— a's turn 2 —",
        "a buys a Moat.",
    ]

    game_log = parse_game_log(build_game_log_html(lines))

    assert game_log is not None
    assert [(t.player_nick, t.turn_number) for t in game_log.turns] == [
        ("a", 1),
        ("b", 1),
        ("a", 2),
    ]
    assert game_log.header.winner_nick == "a"


def test_2010_era_headers_without_numbers_are_counted_per_player() -> None:
    lines = [
        " --- a's turn ---",
        " a plays 5 Coppers.",
        " a buys a Witch.",
        "",
        "    --- b's turn ---",
        "    b buys a Silver.",
        "",
        " --- a's turn ---",
        " a buys a Moat.",
    ]

    game_log = parse_game_log(build_game_log_html(lines))

    assert game_log is not None
    assert [(t.player_nick, t.turn_number) for t in game_log.turns] == [
        ("a", 1),
        ("b", 1),
        ("a", 2),
    ]
    assert game_log.turns[0].cards_bought == (card_quantity("Witch"),)


def test_buys_are_recorded_as_bought_and_gained() -> None:
    game_log = _two_player_log(["a plays 3 Coppers.", "a buys a Silver and 2 Golds."])

    turn = game_log.turns[0]

    expected = (card_quantity("Silver"), card_quantity("Golds", 2))
    assert turn.cards_bought == expected
    assert turn.cards_gained == expected
    assert turn.cards_played == (card_quantity("Coppers", 3),)


def test_play_line_lists_several_card_groups() -> None:
    game_log = _two_player_log(["a plays a Gold, a Silver, and 2 Coppers."])

    assert game_log.turns[0].cards_played == (
        card_quantity("Gold"),
        card_quantity("Silver"),
        card_quantity("Coppers", 2),
    )


def test_nick_with_spaces_and_apostrophe() -> None:
    lines = [
        "— King Crimson's Court's turn 1 —",
        "King Crimson's Court buys a Festival.",
    ]

    game_log = parse_game_log(
        build_game_log_html(
            lines,
            players=[
                ("King Crimson's Court", 40, 1, "Witch", None),
                ("b", 30, 1, "Silver", None),
            ],
            winner_nick="King Crimson's Court",
        )
    )

    assert game_log is not None
    assert game_log.turns[0].player_nick == "King Crimson's Court"
    assert game_log.turns[0].cards_bought == (card_quantity("Festival"),)


def test_continuation_gain_for_owner_with_and_without_nick() -> None:
    game_log = _two_player_log(
        [
            "a plays a Workshop.",
            "... gaining a Silver in hand.",
            "... a gains a Gold on the deck.",
            "... gaining nothing.",
        ]
    )

    assert game_log.turns[0].cards_gained == (
        card_quantity("Silver"),
        card_quantity("Gold"),
    )
    assert game_log.turns[0].cards_bought == ()


def test_continuation_gain_naming_another_player_is_dropped() -> None:
    game_log = _two_player_log(
        ["a plays a Witch.", "... b gains a Curse.", "... gaining a Copper."]
    )

    assert game_log.turns[0].cards_gained == (card_quantity("Copper"),)


def test_gain_token_is_not_a_card() -> None:
    game_log = _two_player_log(
        ["a plays a Pirate Ship.", "... a gains a Pirate Ship token."]
    )

    assert game_log.turns[0].cards_gained == ()


def test_reveal_and_gain_another_one_gains_the_revealed_card() -> None:
    game_log = _two_player_log(
        ["a plays a Mint.", "... revealing a Silver and gaining another one."]
    )

    assert game_log.turns[0].cards_gained == (card_quantity("Silver"),)


def test_trash_shapes() -> None:
    game_log = _two_player_log(
        [
            "a plays a Chapel.",
            "... trashing a Copper for +$3.",
            "... a trashes an Estate and gets +1 ▼.",
            "... trashing 2 Coppers from hand.",
            "... a reveals a Silver and trashes it.",
            "... trashing nothing.",
        ]
    )

    assert game_log.turns[0].cards_trashed == (
        card_quantity("Copper"),
        card_quantity("Estate"),
        card_quantity("Coppers", 2),
        card_quantity("Silver"),
    )


def test_trash_naming_another_player_is_dropped() -> None:
    game_log = _two_player_log(["a plays a Swindler.", "... b trashes a Gold."])

    assert game_log.turns[0].cards_trashed == ()


def test_trash_and_gain_in_one_line() -> None:
    game_log = _two_player_log(
        ["a plays a Trader.", "... a trashes 2 Estates, gaining a Silver in hand."]
    )

    assert game_log.turns[0].cards_trashed == (card_quantity("Estates", 2),)
    assert game_log.turns[0].cards_gained == (card_quantity("Silver"),)


def test_trash_and_gain_joined_by_and() -> None:
    game_log = _two_player_log(
        ["a plays a Mine.", "... trashing a Copper and gaining a Silver."]
    )

    assert game_log.turns[0].cards_trashed == (card_quantity("Copper"),)
    assert game_log.turns[0].cards_gained == (card_quantity("Silver"),)


def test_replay_lines_count_as_plays() -> None:
    game_log = _two_player_log(
        [
            "a plays a Throne Room.",
            "... and plays the Steward again.",
            "... ... and plays the Steward a third time.",
        ]
    )

    assert game_log.turns[0].cards_played == (
        card_quantity("Throne Room"),
        card_quantity("Steward"),
        card_quantity("Steward"),
    )


def test_ignored_lines_do_not_affect_turn() -> None:
    game_log = _two_player_log(
        [
            "a plays a Militia.",
            "... getting +$2.",
            "... b discards 2 cards.",
            "(a draws: an Estate and 4 Coppers.)",
            "(a reshuffles.)",
        ]
    )

    turn = game_log.turns[0]
    assert turn.cards_played == (card_quantity("Militia"),)
    assert turn.cards_bought == ()
    assert turn.cards_gained == ()
    assert turn.cards_trashed == ()


def test_buy_and_play_naming_another_nick_are_ignored() -> None:
    game_log = _two_player_log(
        ["b plays a Witch.", "b buys a Gold.", "a buys a Silver."]
    )

    turn = game_log.turns[0]
    assert turn.cards_bought == (card_quantity("Silver"),)
    assert turn.cards_played == ()


def test_owner_nick_with_regex_metacharacters() -> None:
    lines = ["— C++'s turn 1 —", "C++ buys a Silver.", "aC buys a Gold."]

    game_log = parse_game_log(
        build_game_log_html(
            lines,
            winner_nick="C++",
            players=[("C++", 40, 1, "Witch", None), ("b", 30, 1, "Silver", None)],
        )
    )

    assert game_log is not None
    assert game_log.turns[0].cards_bought == (card_quantity("Silver"),)


def test_nick_that_is_a_suffix_of_another_does_not_match() -> None:
    game_log = _two_player_log(["a buys a Silver.", "xa buys a Gold."])

    assert game_log.turns[0].cards_bought == (card_quantity("Silver"),)


def test_resign_and_win_lines_are_ignored() -> None:
    game_log = _two_player_log(
        ["a buys a Silver.", "b resigns from the game.", "a wins!"]
    )

    assert game_log.turns[0].cards_bought == (card_quantity("Silver"),)


def test_turn_block_with_malformed_header_is_none() -> None:
    assert _parse_turn_block(["not a header", "a buys a Silver."], 1) is None


class TestParseTurnHeader:
    def test_2013_shape(self) -> None:
        assert _parse_turn_header("— a's turn 3 —") == "a"

    def test_indented_2013_shape(self) -> None:
        assert _parse_turn_header("   — b's turn 3 —") == "b"

    def test_2010_shape(self) -> None:
        assert (
            _parse_turn_header("    --- quartersmostly's turn ---") == "quartersmostly"
        )

    def test_non_header_line(self) -> None:
        assert _parse_turn_header("a buys a Silver.") is None


class TestParseCardQuantityList:
    def test_single_article(self) -> None:
        assert _parse_card_quantity_list("a Silver") == (CardQuantity("Silver", 1),)

    def test_an_article_and_count(self) -> None:
        assert _parse_card_quantity_list("an Estate and 2 Coppers") == (
            CardQuantity("Estate", 1),
            CardQuantity("Coppers", 2),
        )

    def test_oxford_comma_list(self) -> None:
        assert _parse_card_quantity_list("a Gold, 2 Silvers, and a Copper") == (
            CardQuantity("Gold", 1),
            CardQuantity("Silvers", 2),
            CardQuantity("Copper", 1),
        )

    def test_the_and_another_articles(self) -> None:
        assert _parse_card_quantity_list("the Copper") == (CardQuantity("Copper", 1),)
        assert _parse_card_quantity_list("another Silver") == (
            CardQuantity("Silver", 1),
        )

    def test_the_trashed_article_is_dropped(self) -> None:
        assert _parse_card_quantity_list("the trashed Silver") == (
            CardQuantity("Silver", 1),
        )

    def test_unparseable_phrase_yields_empty(self) -> None:
        assert _parse_card_quantity_list("nothing") == ()
        assert _parse_card_quantity_list("another one") == ()
        assert _parse_card_quantity_list("+$2") == ()


class TestLineBelongsToOwner:
    def test_no_nick_belongs_to_owner(self) -> None:
        assert _line_belongs_to_owner(None, "a") is True

    def test_same_nick_belongs_to_owner(self) -> None:
        assert _line_belongs_to_owner("a", "a") is True

    def test_other_nick_does_not(self) -> None:
        assert _line_belongs_to_owner("b", "a") is False
