from pathlib import Path

import pyarrow as pa
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.snapshots import build as build_module
from src.data_refinement.snapshots.arrow_files import write_table_atomically
from src.data_refinement.snapshots.build import build_card_snapshot
from src.data_refinement.snapshots.card_snapshot import (
    CARD_SNAPSHOT_FILE_NAME,
    CardSnapshot,
)
from src.data_refinement.snapshots.stamps import (
    CardSnapshotStamp,
    StaleSnapshotError,
)
from src.schema.game_id import GameId
from tests.snapshot_fixtures import binder_with_cards, card


class TestBuildAndOpen:
    def test_one_row_per_card_plus_the_sentinel_in_byte_order(
        self, tmp_path: Path
    ) -> None:
        binder = binder_with_cards(10)

        stamp = build_card_snapshot(binder, GameId.MTG, tmp_path)
        snapshot = CardSnapshot.open(tmp_path, binder, GameId.MTG)

        expected = set(binder.all_uuids(GameId.MTG)) | {
            CardBinder.unknown_card_uuid(GameId.MTG)
        }
        assert snapshot.stamp == stamp
        assert snapshot.card_count == 11
        assert set(snapshot.card_uuids) == expected
        assert snapshot.card_uuids == sorted(expected, key=lambda u: u.bytes)

    def test_unknown_index_names_the_sentinel(self, tmp_path: Path) -> None:
        binder = binder_with_cards(5)
        build_card_snapshot(binder, GameId.MTG, tmp_path)

        snapshot = CardSnapshot.open(tmp_path, binder, GameId.MTG)

        assert snapshot.card_uuids[
            snapshot.unknown_index
        ] == CardBinder.unknown_card_uuid(GameId.MTG)

    def test_a_binder_holding_its_sentinel_keeps_one_row(self, tmp_path: Path) -> None:
        binder = binder_with_cards(3)
        binder.ensure_unknown_card(GameId.MTG)
        build_card_snapshot(binder, GameId.MTG, tmp_path)

        assert CardSnapshot.open(tmp_path, binder, GameId.MTG).card_count == 4

    def test_an_empty_game_holds_only_the_sentinel(self, tmp_path: Path) -> None:
        binder = CardBinder()
        build_card_snapshot(binder, GameId.MTG, tmp_path)

        snapshot = CardSnapshot.open(tmp_path, binder, GameId.MTG)

        assert snapshot.card_count == 1
        assert snapshot.unknown_index == 0

    def test_only_the_given_games_cards(self, tmp_path: Path) -> None:
        binder = binder_with_cards(2)
        binder.create(card(GameId.POKEMON))
        build_card_snapshot(binder, GameId.MTG, tmp_path)

        assert CardSnapshot.open(tmp_path, binder, GameId.MTG).card_count == 3

    def test_too_many_cards_for_uint16_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(build_module, "_MAX_CARD_ROWS", 3)
        with pytest.raises(ValueError, match="uint32"):
            build_card_snapshot(binder_with_cards(3), GameId.MTG, tmp_path)
        assert not (tmp_path / CARD_SNAPSHOT_FILE_NAME).exists()

    def test_default_directory_is_per_game(self) -> None:
        assert CardSnapshot.default_directory(GameId.MTG) == Path(
            "data/derived/card_snapshots/mtg"
        )


class TestStale:
    def test_a_missing_file_names_the_build_command(self, tmp_path: Path) -> None:
        with pytest.raises(StaleSnapshotError, match="build_training_snapshots"):
            CardSnapshot.open(tmp_path, CardBinder(), GameId.MTG)

    def test_a_changed_binder_is_stale(self, tmp_path: Path) -> None:
        binder = binder_with_cards(3)
        build_card_snapshot(binder, GameId.MTG, tmp_path)
        binder.create(card())

        with pytest.raises(StaleSnapshotError, match="another card binder version"):
            CardSnapshot.open(tmp_path, binder, GameId.MTG)

    def test_another_format_version_is_stale(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        old_stamp = CardSnapshotStamp(binder.version_for(GameId.MTG), format_version=0)
        _write_card_table(tmp_path, [b"\x00" * 16], [True], old_stamp)

        with pytest.raises(StaleSnapshotError, match="format 0"):
            CardSnapshot.open(tmp_path, binder, GameId.MTG)

    def test_no_sentinel_row_is_stale(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        stamp = CardSnapshotStamp(binder.version_for(GameId.MTG), format_version=1)
        _write_card_table(tmp_path, [b"\x00" * 16], [False], stamp)

        with pytest.raises(StaleSnapshotError, match="0 Unknown sentinel rows"):
            CardSnapshot.open(tmp_path, binder, GameId.MTG)


def _write_card_table(
    directory: Path,
    uuid_bytes: list[bytes],
    is_unknown: list[bool],
    stamp: CardSnapshotStamp,
) -> None:
    schema = pa.schema(
        [("nocab_uuid", pa.binary(16)), ("is_unknown", pa.bool_())],
        metadata=stamp.to_schema_metadata(),
    )
    table = pa.Table.from_arrays(
        [pa.array(uuid_bytes, type=pa.binary(16)), pa.array(is_unknown)],
        schema=schema,
    )
    write_table_atomically(table, directory / CARD_SNAPSHOT_FILE_NAME)
