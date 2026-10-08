from src.data_refinement.metrics.sts2_runs.card_removal_reader import (
    CardRemovalReader,
)
from src.data_refinement.metrics.sts2_runs.card_upgrade_reader import (
    CardUpgradeReader,
)
from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice
from src.data_refinement.metrics.sts2_runs.player_history import PlayerHistory
from src.data_refinement.metrics.sts2_runs.shop_purchase_reader import (
    ShopPurchaseReader,
)
from tests.data_refinement.metrics.sts2_runs._points import (
    UNKNOWN,
    player,
    point,
    u,
    uuid_of,
)

_STOCK = ["CARD.S1", "CARD.S2", "CARD.S3", "CARD.S4", "CARD.S5", "CARD.S6", "CARD.S7"]


def _shop_stats(bought: list[str], **extra: object) -> dict:
    unsold = [card for card in _STOCK if card not in bought]
    return {
        "card_choices": [{"card": {"id": c}, "was_picked": False} for c in unsold],
        "cards_gained": [{"id": c} for c in bought],
        **extra,
    }


def _history(deck: list[tuple[str, int]], points: list[dict]) -> PlayerHistory:
    return PlayerHistory.build(player(*deck), points)


class TestShopPurchaseReader:
    def _read(self, history: PlayerHistory) -> tuple[PickChoice, ...]:
        return ShopPurchaseReader(uuid_of).read(history)

    def test_each_purchase_is_a_pick_from_the_stock_less_earlier_buys(self) -> None:
        points = [point("event"), point("shop", **_shop_stats(["CARD.S6", "CARD.S2"]))]

        first, second = self._read(_history([("CARD.A", 1)], points))

        sorted_stock = tuple(u(c) for c in _STOCK)
        assert first.offered == sorted_stock
        assert first.picked_index is not None
        assert first.offered[first.picked_index] == u("CARD.S6")
        assert first.deck_before == (u("CARD.A"),)
        # S6 is gone from the stock and now in the deck
        assert second.offered == tuple(u(c) for c in _STOCK if c != "CARD.S6")
        assert second.picked_index is not None
        assert second.offered[second.picked_index] == u("CARD.S2")
        assert second.deck_before == (u("CARD.A"), u("CARD.S6"))

    def test_the_option_order_does_not_depend_on_what_was_bought(self) -> None:
        points = [point("event"), point("shop", **_shop_stats(["CARD.S7"]))]
        other = [point("event"), point("shop", **_shop_stats(["CARD.S1"]))]

        (a,) = self._read(_history([("CARD.A", 1)], points))
        (b,) = self._read(_history([("CARD.A", 1)], other))

        assert a.offered == b.offered
        assert a.picked_index != b.picked_index

    def test_a_visit_with_no_purchase_writes_nothing(self) -> None:
        points = [point("event"), point("shop", **_shop_stats([]))]

        assert self._read(_history([("CARD.A", 1)], points)) == ()

    def test_a_visit_flagged_with_was_picked_is_not_usable(self) -> None:
        stats = _shop_stats(["CARD.S1"])
        stats["card_choices"][0]["was_picked"] = True
        points = [point("event"), point("shop", **stats)]

        assert self._read(_history([("CARD.A", 1)], points)) == ()

    def test_a_visit_whose_stock_does_not_add_up_to_seven_is_not_usable(self) -> None:
        stats = _shop_stats(["CARD.S1"])
        stats["card_choices"].pop()
        points = [point("event"), point("shop", **stats)]

        assert self._read(_history([("CARD.A", 1)], points)) == ()

    def test_an_unaliased_card_for_sale_stays_in_the_options_as_none(self) -> None:
        stats = _shop_stats(["CARD.S1"])
        stats["card_choices"][0]["card"]["id"] = UNKNOWN
        points = [point("event"), point("shop", **stats)]

        (choice,) = self._read(_history([("CARD.A", 1)], points))

        assert None in choice.offered

    def test_only_shops_count(self) -> None:
        points = [point("event"), point("monster", **_shop_stats(["CARD.S1"]))]

        assert self._read(_history([("CARD.A", 1)], points)) == ()


class TestCardRemovalReader:
    def _read(self, history: PlayerHistory) -> tuple[PickChoice, ...]:
        return CardRemovalReader(uuid_of).read(history)

    def test_the_options_are_the_distinct_deck_cards(self) -> None:
        removal = {"id": "CARD.A", "floor_added_to_deck": 1}
        # The removed copy is not in the final deck, only in the removal
        deck = [("CARD.B", 1), ("CARD.B", 1), ("CARD.C", 1)]
        points = [point("event"), point("shop", cards_removed=[removal])]

        (choice,) = self._read(_history(deck, points))

        assert choice.floor == 2
        assert choice.offered == tuple(sorted({u("CARD.A"), u("CARD.B"), u("CARD.C")}))
        assert choice.picked_index is not None
        assert choice.offered[choice.picked_index] == u("CARD.A")
        # The removed copy is still in the deck on arrival, copies and all
        assert len(choice.deck_before) == 4

    def test_two_removals_at_one_visit_are_not_read(self) -> None:
        removal = {"id": "CARD.A", "floor_added_to_deck": 1}
        points = [point("event"), point("shop", cards_removed=[removal, removal])]

        assert self._read(_history([("CARD.A", 1)], points)) == ()

    def test_a_removal_outside_a_shop_is_not_read(self) -> None:
        removal = {"id": "CARD.A", "floor_added_to_deck": 1}
        points = [point("event"), point("event", cards_removed=[removal])]

        assert self._read(_history([("CARD.A", 1)], points)) == ()

    def test_an_unaliased_removed_card_writes_nothing(self) -> None:
        removal = {"id": UNKNOWN, "floor_added_to_deck": 1}
        points = [point("event"), point("shop", cards_removed=[removal])]

        assert self._read(_history([(UNKNOWN, 1), ("CARD.A", 1)], points)) == ()

    def test_a_card_bought_and_removed_at_the_same_shop_writes_nothing(self) -> None:
        # Added on floor 2 itself, so it was not in the deck on arrival
        removal = {"id": "CARD.NEW", "floor_added_to_deck": 2}
        points = [point("event"), point("shop", cards_removed=[removal])]

        assert self._read(_history([("CARD.A", 1)], points)) == ()


class TestCardUpgradeReader:
    def _read(self, history: PlayerHistory) -> tuple[PickChoice, ...]:
        return CardUpgradeReader(uuid_of).read(history)

    def _smith(self, card: str) -> dict:
        return point("rest_site", rest_site_choices=["SMITH"], upgraded_cards=[card])

    def test_the_options_are_the_deck_cards_not_yet_upgraded(self) -> None:
        deck = [("CARD.A", 1), ("CARD.B", 1)]
        points = [point("event"), self._smith("CARD.A"), self._smith("CARD.B")]

        first, second = self._read(_history(deck, points))

        assert set(first.offered) == {u("CARD.A"), u("CARD.B")}
        # A was upgraded on floor 2, so floor 3 can only upgrade B
        assert second.offered == (u("CARD.B"),)
        assert second.picked_index == 0

    def test_another_copy_keeps_the_card_an_option(self) -> None:
        deck = [("CARD.A", 1), ("CARD.A", 1)]
        points = [point("event"), self._smith("CARD.A"), self._smith("CARD.A")]

        first, second = self._read(_history(deck, points))

        assert first.offered == (u("CARD.A"),)
        assert second.offered == (u("CARD.A"),)

    def test_an_upgrade_elsewhere_counts_against_the_options(self) -> None:
        points = [
            point("event"),
            point("event", upgraded_cards=["CARD.A"]),
            self._smith("CARD.B"),
        ]

        (choice,) = self._read(_history([("CARD.A", 1), ("CARD.B", 1)], points))

        assert choice.offered == (u("CARD.B"),)

    def test_a_rest_without_a_smith_is_not_read(self) -> None:
        points = [
            point("event"),
            point("rest_site", rest_site_choices=["HEAL"], upgraded_cards=["CARD.A"]),
        ]

        assert self._read(_history([("CARD.A", 1)], points)) == ()

    def test_a_smith_upgrading_two_cards_is_not_read(self) -> None:
        points = [
            point("event"),
            point(
                "rest_site",
                rest_site_choices=["SMITH"],
                upgraded_cards=["CARD.A", "CARD.B"],
            ),
        ]

        assert self._read(_history([("CARD.A", 1), ("CARD.B", 1)], points)) == ()

    def test_an_unaliased_upgraded_card_writes_nothing(self) -> None:
        points = [point("event"), self._smith(UNKNOWN)]

        assert self._read(_history([(UNKNOWN, 1), ("CARD.A", 1)], points)) == ()

    def test_an_upgrade_of_a_card_outside_the_options_writes_nothing(self) -> None:
        points = [point("event"), self._smith("CARD.GHOST")]

        assert self._read(_history([("CARD.A", 1)], points)) == ()
