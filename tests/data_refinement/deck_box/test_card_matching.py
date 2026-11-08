from dataclasses import dataclass
from uuid import UUID

from src.data_refinement.deck_box.card_matching import (
    CachedCardMatcher,
    CardMatchKind,
    card_uuids_per_copy,
)

_BOLT = UUID("00000000-0000-0000-0000-000000000001")
_UNKNOWN = UUID("00000000-0000-0000-0000-0000000000ff")


@dataclass(frozen=True)
class _NameRef:
    name: str


@dataclass(frozen=True)
class _Entry:
    count: int
    card: _NameRef


class _NameMatcher(CachedCardMatcher[_NameRef]):
    """Knows one card; records each uncached lookup."""

    def __init__(self) -> None:
        super().__init__()
        self.lookups: list[str] = []

    def _match(self, key: _NameRef) -> tuple[UUID | None, CardMatchKind]:
        self.lookups.append(key.name)
        if key.name == "Bolt":
            return _BOLT, CardMatchKind.NAME
        return None, CardMatchKind.UNMATCHED


class TestCachedCardMatcher:
    def test_each_key_is_looked_up_once_and_counted_per_call(self) -> None:
        matcher = _NameMatcher()

        results = [
            matcher.match_card_uuid(_NameRef(name))
            for name in "Bolt Bolt X X X".split()
        ]

        assert results == [_BOLT, _BOLT, None, None, None]
        assert matcher.lookups == ["Bolt", "X"]
        assert matcher.match_counts == {
            CardMatchKind.NAME: 2,
            CardMatchKind.UNMATCHED: 3,
        }
        assert matcher.unmatched_counts == {_NameRef("X"): 3}


class TestCardUuidsPerCopy:
    def test_one_uuid_per_copy_with_unknown_for_misses(self) -> None:
        entries = [_Entry(2, _NameRef("Bolt")), _Entry(3, _NameRef("X"))]

        result = card_uuids_per_copy(entries, _NameMatcher(), _UNKNOWN)

        assert result == [_BOLT, _BOLT, _UNKNOWN, _UNKNOWN, _UNKNOWN]

    def test_no_entries(self) -> None:
        assert card_uuids_per_copy([], _NameMatcher(), _UNKNOWN) == []
