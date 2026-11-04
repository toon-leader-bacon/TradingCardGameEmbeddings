import pytest

from src.data_refinement.snapshots.stamps import (
    CardSnapshotStamp,
    DeckSnapshotStamp,
    StaleSnapshotError,
)

_CARD_STAMP = CardSnapshotStamp(card_binder_version="binder-v1", format_version=1)
_DECK_STAMP = DeckSnapshotStamp(
    card_snapshot=_CARD_STAMP, deck_box_version="box-v1", format_version=1
)


class TestRoundTrip:
    def test_a_card_stamp_round_trips(self) -> None:
        metadata = _CARD_STAMP.to_schema_metadata()
        assert CardSnapshotStamp.from_schema_metadata(metadata) == _CARD_STAMP

    def test_a_deck_stamp_round_trips_with_its_card_stamp(self) -> None:
        metadata = _DECK_STAMP.to_schema_metadata()
        assert DeckSnapshotStamp.from_schema_metadata(metadata) == _DECK_STAMP

    def test_a_deck_stamps_metadata_also_reads_as_its_card_stamp(self) -> None:
        metadata = _DECK_STAMP.to_schema_metadata()
        assert CardSnapshotStamp.from_schema_metadata(metadata) == _CARD_STAMP


class TestUnreadable:
    def test_no_metadata_is_stale(self) -> None:
        with pytest.raises(StaleSnapshotError, match="build_training_snapshots"):
            CardSnapshotStamp.from_schema_metadata(None)

    def test_a_missing_key_is_stale(self) -> None:
        with pytest.raises(StaleSnapshotError):
            DeckSnapshotStamp.from_schema_metadata(_CARD_STAMP.to_schema_metadata())

    def test_a_non_integer_format_is_stale(self) -> None:
        metadata = _CARD_STAMP.to_schema_metadata()
        metadata[b"card_snapshot_format_version"] = b"one"
        with pytest.raises(StaleSnapshotError, match="not an integer"):
            CardSnapshotStamp.from_schema_metadata(metadata)

    def test_non_utf8_text_is_stale(self) -> None:
        metadata = _CARD_STAMP.to_schema_metadata()
        metadata[b"card_binder_version"] = b"\xff\xfe"
        with pytest.raises(StaleSnapshotError, match="unreadable"):
            CardSnapshotStamp.from_schema_metadata(metadata)
