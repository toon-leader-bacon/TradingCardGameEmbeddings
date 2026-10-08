import pytest

from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice
from src.data_refinement.metrics.sts2_runs.player_history import PlayerHistory
from tests.data_refinement.metrics.sts2_runs._points import player, point, u


class TestBuild:
    def test_floors_count_points_the_player_has_no_stats_for(self) -> None:
        points = [
            {"rooms": [{"room_type": "monster"}], "player_stats": []},
            point("shop"),
        ]

        history = PlayerHistory.build(player(("CARD.A", 1)), points)

        assert [(p.floor, p.room_type) for p in history.points] == [(2, "shop")]

    def test_the_timeline_holds_the_final_deck_and_the_removed_copies(self) -> None:
        removal = {"id": "CARD.OLD", "floor_added_to_deck": 1}
        points = [point("event"), point("shop", cards_removed=[removal])]

        history = PlayerHistory.build(player(("CARD.A", 1)), points)

        assert history.timeline.raw_card_ids_before(2) == ["CARD.A", "CARD.OLD"]
        assert history.timeline.raw_card_ids_before(3) == ["CARD.A"]


class TestUpgradeCountsBefore:
    def test_counts_upgrades_on_earlier_floors_in_any_room(self) -> None:
        points = [
            point("event", upgraded_cards=["CARD.A", "CARD.A"]),
            point("rest_site", upgraded_cards=["CARD.B"]),
            point("rest_site", upgraded_cards=["CARD.A"]),
        ]
        history = PlayerHistory.build(player(("CARD.A", 1)), points)

        counts = history.upgrade_counts_before(3)

        assert counts == {"CARD.A": 2, "CARD.B": 1}

    def test_a_floor_below_one_is_rejected(self) -> None:
        history = PlayerHistory.build(player(("CARD.A", 1)), [point()])

        with pytest.raises(ValueError):
            history.upgrade_counts_before(0)


class TestPickChoice:
    def test_a_pick_outside_the_options_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            PickChoice(1, (), (u("CARD.A"),), picked_index=1)

    def test_a_decline_is_allowed(self) -> None:
        assert PickChoice(1, (), (u("CARD.A"),), picked_index=None).picked_index is None
