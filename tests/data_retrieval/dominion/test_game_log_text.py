from src.data_retrieval.dominion.game_log_text import game_log_text


def _line(padding: str, text: str) -> str:
    return f'<div style="padding-left: {padding}%"><span>{text}</span></div>'


class TestGameLogText:
    def test_new_turn_line_starts_after_a_blank_line(self) -> None:
        log_html = '<div class="new-turn-line"></div>' + _line("0", "Turn 1")

        assert game_log_text(log_html) == "\n\n\nTurn 1"

    def test_indents_four_spaces_per_padding_level_in_order_seen(self) -> None:
        log_html = (
            _line("0", "A plays a Village.")
            + _line("2.5", "A draws a card.")
            + _line("5", "A trashes a Copper.")
            + _line("2.5", "A gets +2 Actions.")
        )

        assert game_log_text(log_html) == (
            "\nA plays a Village."
            "\n    A draws a card."
            "\n        A trashes a Copper."
            "\n    A gets +2 Actions."
        )

    def test_other_classes_and_newlines_are_ignored(self) -> None:
        log_html = '<div class="log-line">\n' + _line("0", "Game over.") + "\n</div>"

        assert game_log_text(log_html) == "\nGame over."

    def test_empty_html_gives_empty_text(self) -> None:
        assert game_log_text("") == ""
