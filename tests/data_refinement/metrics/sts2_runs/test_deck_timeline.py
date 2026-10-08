import pytest

from src.data_refinement.metrics.sts2_runs.deck_timeline import (
    DeckTimeline,
    TimedCard,
)


def _timeline() -> DeckTimeline:
    return DeckTimeline(
        (
            TimedCard("CARD.A", floor_added=1, floor_left=None),
            TimedCard("CARD.B", floor_added=1, floor_left=4),
            TimedCard("CARD.C", floor_added=3, floor_left=None),
        )
    )


class TestRawCardIdsBefore:
    def test_the_run_start_deck_is_what_was_added_on_floor_1(self) -> None:
        assert _timeline().raw_card_ids_before(2) == ["CARD.A", "CARD.B"]

    def test_a_copy_added_on_the_floor_is_not_yet_held(self) -> None:
        assert _timeline().raw_card_ids_before(3) == ["CARD.A", "CARD.B"]
        assert _timeline().raw_card_ids_before(4) == ["CARD.A", "CARD.B", "CARD.C"]

    def test_a_copy_is_held_on_arrival_at_the_floor_that_removes_it(self) -> None:
        assert "CARD.B" in _timeline().raw_card_ids_before(4)

    def test_a_copy_is_gone_after_the_floor_that_removed_it(self) -> None:
        assert "CARD.B" not in _timeline().raw_card_ids_before(5)

    def test_floor_1_has_an_empty_deck(self) -> None:
        assert _timeline().raw_card_ids_before(1) == []

    def test_a_floor_below_1_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _timeline().raw_card_ids_before(0)
