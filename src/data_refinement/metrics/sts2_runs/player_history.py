"""A player's run as the per-floor readers see it: every map point with
that player's own stats, and the deck as it stood on each floor.

Extracted from CardRewardReader so the four pick readers (card reward,
shop purchase, card removal, card upgrade) share one parse of the raw
player instead of each rebuilding it.

RAW SHAPE (extends run_parser.py's): each map point's "rooms" holds the
room_type; its "player_stats" holds one entry per player, matched by
"player_id". The stats keys the readers use:
    "card_choices", "cards_gained", "cards_removed", "cards_transformed",
    "upgraded_cards", "rest_site_choices".
A player's "deck" entries carry "floor_added_to_deck".
"""

from collections import Counter
from dataclasses import dataclass

from src.data_refinement.metrics.sts2_runs.deck_timeline import (
    DeckTimeline,
    TimedCard,
)


@dataclass(frozen=True)
class PlayerPoint:
    """One map point as one player saw it.

    floor: the point's 1-based position across all acts.
    room_type: the raw room_type of its first room.
    stats: the raw player_stats entry for this player. The readers index
        it by the raw key names above (accepted debt: the raw-schema
        knowledge is spread over the readers, not parsed once here).
    """

    floor: int
    room_type: str
    stats: dict


@dataclass(frozen=True)
class PlayerHistory:
    """One player's points and deck timeline.

    points: every point with a room and this player's stats, in floor
        order (a point missing either is left out; its floor number
        still counts).
    timeline: every copy the player ever held (deck_timeline.py).
    """

    points: tuple[PlayerPoint, ...]
    timeline: DeckTimeline

    @classmethod
    def build(cls, player: dict, map_points: list[dict]) -> "PlayerHistory":
        """Pair the player with their points and rebuild their timeline.

        Inputs:
            player: one raw player dict ("id", "deck").
            map_points: every act's map points, flattened, in order.
        Output: PlayerHistory.
        Side effects: none.
        Exceptions: KeyError / TypeError if a point or the player lacks a
            field the module docstring names.

        Example:
            >>> PlayerHistory.build(raw_player, flat_map_points).points[0].floor
            1
        """
        points: list[PlayerPoint] = []

        # Pair each map point with this player's own stats
        for floor, point in enumerate(map_points, start=1):
            own_stats = [
                stats
                for stats in point["player_stats"]
                if stats["player_id"] == player["id"]
            ]
            if not point["rooms"] or not own_stats:
                continue
            points.append(
                PlayerPoint(floor, point["rooms"][0]["room_type"], own_stats[0])
            )

        # The final deck: still held at run end
        cards = [
            TimedCard(entry["id"], entry["floor_added_to_deck"], None)
            for entry in player["deck"]
        ]

        # Copies that left: removed at a shop, or transformed into another
        for seen in points:
            for removed in seen.stats.get("cards_removed", []):
                cards.append(
                    TimedCard(removed["id"], removed["floor_added_to_deck"], seen.floor)
                )
            for transform in seen.stats.get("cards_transformed", []):
                original = transform["original_card"]
                cards.append(
                    TimedCard(
                        original["id"], original["floor_added_to_deck"], seen.floor
                    )
                )
        return cls(tuple(points), DeckTimeline(tuple(cards)))

    def upgrade_counts_before(self, floor: int) -> Counter[str]:
        """How many times each raw card id was upgraded before floor, over
        every room type (rest-site smiths, events, and so on).

        Inputs: floor (int, >= 1).
        Output: Counter of raw card id -> upgrades on floors < floor.
        Side effects: none.
        Exceptions: ValueError if floor < 1.

        Example:
            >>> history.upgrade_counts_before(10)["CARD.STRIKE_REGENT"]
            1
        """
        result: Counter[str] = Counter()

        # Validate inputs
        if floor < 1:
            raise ValueError(f"floor must be >= 1, got {floor}")

        # Count every upgrade on an earlier floor, whatever the room
        for point in self.points:
            if point.floor < floor:
                result.update(point.stats.get("upgraded_cards", []))
        return result
