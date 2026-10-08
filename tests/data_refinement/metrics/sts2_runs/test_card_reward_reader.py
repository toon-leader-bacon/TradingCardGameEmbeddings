from uuid import NAMESPACE_OID, UUID, uuid5

from src.data_refinement.metrics.sts2_runs.card_reward_reader import (
    CardRewardReader,
)
from src.data_refinement.metrics.sts2_runs.run_record import CardRewardChoice

_UNKNOWN = "CARD.UNKNOWN"


def _uuid_of(raw_id: str) -> UUID | None:
    return None if raw_id == _UNKNOWN else uuid5(NAMESPACE_OID, raw_id)


def _u(raw_id: str) -> UUID:
    return uuid5(NAMESPACE_OID, raw_id)


def _options(picked: str | None, *ids: str) -> list[dict]:
    return [{"card": {"id": i}, "was_picked": i == picked} for i in ids]


def _point(room_type: str = "monster", **stats: object) -> dict:
    return {
        "rooms": [{"room_type": room_type}],
        "player_stats": [{"player_id": 1, **stats}],
    }


def _player(*deck: tuple[str, int]) -> dict:
    return {
        "id": 1,
        "deck": [{"id": i, "floor_added_to_deck": f} for i, f in deck],
    }


def _read(player: dict, points: list[dict]) -> tuple[CardRewardChoice, ...]:
    return CardRewardReader(_uuid_of).read(player, points)


class TestRead:
    def test_a_pick_records_the_deck_before_and_the_offer(self) -> None:
        player = _player(("CARD.A", 1), ("CARD.B", 2))
        points = [
            _point("event"),
            _point(card_choices=_options("CARD.B", "CARD.B", "CARD.C")),
        ]

        (choice,) = _read(player, points)

        assert choice.floor == 2
        # CARD.B was added by this very reward, so it is not in the deck yet
        assert choice.deck_before == (_u("CARD.A"),)
        assert choice.offered == (_u("CARD.B"), _u("CARD.C"))
        assert choice.picked_index == 0

    def test_a_skip_has_no_picked_index(self) -> None:
        points = [_point(card_choices=_options(None, "CARD.B", "CARD.C"))]

        (choice,) = _read(_player(("CARD.A", 1)), points)

        assert choice.picked_index is None

    def test_the_picked_index_is_the_position_in_the_offer(self) -> None:
        options = _options("CARD.D", "CARD.B", "CARD.C", "CARD.D")

        (choice,) = _read(_player(("CARD.A", 1)), [_point(card_choices=options)])

        assert choice.picked_index == 2

    def test_a_reward_with_two_picks_is_not_a_choice(self) -> None:
        options = [
            {"card": {"id": "CARD.B"}, "was_picked": True},
            {"card": {"id": "CARD.C"}, "was_picked": True},
        ]

        assert _read(_player(("CARD.A", 1)), [_point(card_choices=options)]) == ()

    def test_only_combat_rooms_count(self) -> None:
        options = _options("CARD.B", "CARD.B", "CARD.C")
        points = [
            _point("shop", card_choices=options),
            _point("event", card_choices=options),
            _point("elite", card_choices=options),
            _point("boss", card_choices=options),
        ]

        floors = [c.floor for c in _read(_player(("CARD.A", 1)), points)]

        assert floors == [3, 4]

    def test_a_combat_room_with_no_card_choices_is_skipped(self) -> None:
        points = [_point(), _point(card_choices=[])]

        assert _read(_player(("CARD.A", 1)), points) == ()

    def test_a_removed_card_is_in_the_deck_until_its_floor_passes(self) -> None:
        player = _player(("CARD.A", 1))
        removal = {"id": "CARD.OLD", "floor_added_to_deck": 1}
        points = [
            _point("event"),
            _point(card_choices=_options(None, "CARD.X")),
            _point("shop", cards_removed=[removal]),
            _point(card_choices=_options(None, "CARD.X")),
        ]

        before_shop, after_shop = _read(player, points)

        assert before_shop.deck_before == (_u("CARD.A"), _u("CARD.OLD"))
        assert after_shop.deck_before == (_u("CARD.A"),)

    def test_a_transformed_original_leaves_and_the_final_card_stays(self) -> None:
        player = _player(("CARD.A", 1), ("CARD.NEW", 2))
        transform = {
            "original_card": {"id": "CARD.OLD", "floor_added_to_deck": 1},
            "final_card": {"id": "CARD.NEW", "floor_added_to_deck": 2},
        }
        points = [
            _point("event"),
            _point("event", cards_transformed=[transform]),
            _point(card_choices=_options(None, "CARD.X")),
        ]

        (choice,) = _read(player, points)

        assert choice.deck_before == (_u("CARD.A"), _u("CARD.NEW"))

    def test_an_unaliased_card_is_kept_as_none(self) -> None:
        player = _player((_UNKNOWN, 1))
        points = [
            _point("event"),
            _point(card_choices=_options(None, _UNKNOWN, "CARD.X")),
        ]

        (choice,) = _read(player, points)

        assert choice.deck_before == (None,)
        assert choice.offered == (None, _u("CARD.X"))

    def test_floors_count_points_the_player_has_no_stats_for(self) -> None:
        points = [
            {"rooms": [{"room_type": "monster"}], "player_stats": []},
            _point(card_choices=_options(None, "CARD.X")),
        ]

        (choice,) = _read(_player(("CARD.A", 1)), points)

        assert choice.floor == 2

    def test_another_players_stats_are_ignored(self) -> None:
        point = _point(card_choices=_options("CARD.X", "CARD.X"))
        point["player_stats"].insert(
            0, {"player_id": 2, "card_choices": _options(None, "CARD.Y")}
        )

        (choice,) = _read(_player(("CARD.A", 1)), [point])

        assert choice.offered == (_u("CARD.X"),)
