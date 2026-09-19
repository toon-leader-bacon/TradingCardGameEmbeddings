from src.data_refinement.metrics.isotropic.games.header_parser import (
    GameHeaderPlayer,
    parse_game_header,
)
from tests.data_refinement.metrics.isotropic.games._helpers import build_header_html


def test_parses_single_pile_ending() -> None:
    html = build_header_html(
        "undertoe",
        "All Provinces are gone.",
        ["Witch", "Village", "Moat"],
        [
            ("undertoe", 40, 20, "Witch", "Silver"),
            ("Dave", 30, 20, "Moat", None),
        ],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.winner_nick == "undertoe"
    assert header.exhausted_pile_names == ("Provinces",)
    assert header.kingdom_card_names == ("Witch", "Village", "Moat")
    assert header.is_natural_kingdom is True
    assert header.players == (
        GameHeaderPlayer(
            nick="undertoe", score=40, turns=20, opening_buy_names=("Witch", "Silver")
        ),
        GameHeaderPlayer(
            nick="Dave", score=30, turns=20, opening_buy_names=("Moat", None)
        ),
    )


def test_parses_multi_pile_ending() -> None:
    html = build_header_html(
        "brog",
        "Duke, Monument, and Duchy are all gone.",
        ["Witch", "Village", "Moat", "Duke", "Monument"],
        [("brog", 50, 22, "Witch", None)],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.exhausted_pile_names == ("Duke", "Monument", "Duchy")


def test_parses_two_item_pile_list_without_oxford_comma() -> None:
    html = build_header_html(
        "brog",
        "Duke and Monument are all gone.",
        ["Witch"],
        [("brog", 50, 22, "Witch", None)],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.exhausted_pile_names == ("Duke", "Monument")


def test_negative_score_and_singular_point_and_turn() -> None:
    html = build_header_html(
        "Jonestown",
        "All Provinces are gone.",
        ["Witch"],
        [
            ("Jonestown", 48, 1, "Witch", None),
            ("Go Canucks", -2, 25, "Witch", None),
        ],
    )
    # Rewrite the singular "1 turn"/"-2 points" cases directly, since
    # build_header_html always writes plural "points"/"turns" - real
    # isotropic text singularizes these (confirmed live) so the parser
    # must too.
    html = html.replace("48 points (details); 1 turns", "48 points (details); 1 turn")
    html = html.replace("-2 points (details); 25 turns", "-2 point (details); 25 turns")

    header = parse_game_header(html)

    assert header is not None
    assert header.players[0].turns == 1
    assert header.players[1].score == -2


def test_rank_prefixed_player_line() -> None:
    html = build_header_html(
        "sakebomb",
        "All Provinces are gone.",
        ["Witch"],
        [("sakebomb", 28, 29, "Witch", None), ("jwpepa", 10, 29, "Moat", None)],
    )
    html = html.replace("sakebomb: 28 points", "#1 sakebomb: 28 points")
    html = html.replace("jwpepa: 10 points", "#2 jwpepa: 10 points")

    header = parse_game_header(html)

    assert header is not None
    assert [p.nick for p in header.players] == ["sakebomb", "jwpepa"]


def test_skips_resigned_players_final_deck_kept() -> None:
    html = build_header_html(
        "sakebomb",
        "All Provinces are gone.",
        ["Witch"],
        [("sakebomb", 28, 29, "Witch", None)],
    )
    # Append a resigned player's block manually (build_header_html has
    # no direct support for this shape - confirmed live format).
    html = html.replace(
        "----------------------\n\ntrash:",
        "jwpepa: resigned (1st); 29 turns\n"
        "opening: nothing / Saboteur\n"
        "[33 cards] details\n\n"
        "----------------------\n\ntrash:",
    )

    header = parse_game_header(html)

    assert header is not None
    assert [p.nick for p in header.players] == ["sakebomb"]


def test_none_when_no_pre_tag() -> None:
    assert parse_game_header("<html><body>no pre here</body></html>") is None


def test_none_when_no_winner_line() -> None:
    html = "<html><body><pre>All but one player has resigned.\n</pre></body></html>"
    assert parse_game_header(html) is None


def test_none_when_winner_exists_but_pile_line_is_resignation_summary() -> None:
    # Confirmed live: a game CAN have a real winner line while the
    # second line is a resignation summary instead of a pile-exhaustion
    # sentence, when a different player resigned mid-game.
    html = (
        "<html><body><pre>sakebomb wins!\n"
        "All but one player has resigned.\n"
        "\ncards in supply: Witch\n"
        "----------------------\n\n"
        "sakebomb: 28 points (details); 29 turns\n"
        "opening: Witch / nothing\n\n"
        "----------------------\n</pre></body></html>"
    )
    assert parse_game_header(html) is None


def test_none_when_you_have_resigned_pile_line() -> None:
    html = "<html><body><pre>solo wins!\nYou have resigned.\n</pre></body></html>"
    assert parse_game_header(html) is None


def test_none_when_missing_second_separator() -> None:
    html = (
        "<html><body><pre>undertoe wins!\n"
        "All Provinces are gone.\n"
        "\ncards in supply: Witch\n"
        "----------------------\n\n"
        "undertoe: 40 points (details); 20 turns\n"
        "opening: Witch / nothing\n"
        "</pre></body></html>"
    )
    assert parse_game_header(html) is None


def test_natural_kingdom_false_when_constraint_marker_present() -> None:
    html = build_header_html(
        "undertoe",
        "All Provinces are gone.",
        ["Witch"],
        [("undertoe", 40, 20, "Witch", None)],
        extra_kingdom_lines=["Constraint(s) used: required: Witch."],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.is_natural_kingdom is False


def test_natural_kingdom_true_with_point_tracker_line() -> None:
    html = build_header_html(
        "undertoe",
        "All Provinces are gone.",
        ["Witch"],
        [("undertoe", 40, 20, "Witch", None)],
        extra_kingdom_lines=["The point tracker will be available."],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.is_natural_kingdom is True


def test_natural_kingdom_true_with_default_selection_line() -> None:
    html = build_header_html(
        "undertoe",
        "All Provinces are gone.",
        ["Witch"],
        [("undertoe", 40, 20, "Witch", None)],
        extra_kingdom_lines=["Default card selection was used."],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.is_natural_kingdom is True


def test_opening_buy_nothing_in_first_slot_maps_to_none() -> None:
    html = build_header_html(
        "a",
        "All Provinces are gone.",
        ["Witch"],
        [("a", 40, 20, None, "Witch")],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.players[0].opening_buy_names == (None, "Witch")


def test_bane_marker_stripped_from_kingdom_card_name() -> None:
    html = build_header_html(
        "a",
        "All Provinces are gone.",
        ["Young Witch", "Tunnel\u2666"],
        [("a", 40, 20, "Witch", None)],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.kingdom_card_names == ("Young Witch", "Tunnel")


def test_opening_buy_nothing_in_both_slots() -> None:
    html = build_header_html(
        "a",
        "All Provinces are gone.",
        ["Witch"],
        [("a", 40, 20, None, None)],
    )

    header = parse_game_header(html)

    assert header is not None
    assert header.players[0].opening_buy_names == (None, None)
