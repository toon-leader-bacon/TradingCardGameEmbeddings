"""Shared builders for the pick-reader tests: raw map points and players in
the spire_codex schema, and a card-id -> uuid function."""

from uuid import NAMESPACE_OID, UUID, uuid5

UNKNOWN = "CARD.UNKNOWN"


def uuid_of(raw_id: str) -> UUID | None:
    """A stable uuid per raw id; UNKNOWN has no alias."""
    return None if raw_id == UNKNOWN else uuid5(NAMESPACE_OID, raw_id)


def u(raw_id: str) -> UUID:
    return uuid5(NAMESPACE_OID, raw_id)


def point(room_type: str = "monster", **stats: object) -> dict:
    return {
        "rooms": [{"room_type": room_type}],
        "player_stats": [{"player_id": 1, **stats}],
    }


def player(*deck: tuple[str, int]) -> dict:
    return {
        "id": 1,
        "deck": [{"id": i, "floor_added_to_deck": f} for i, f in deck],
    }
