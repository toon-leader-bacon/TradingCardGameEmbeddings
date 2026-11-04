from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pyarrow as pa
import pytest

from src.data_refinement.snapshots.arrow_files import (
    open_mapped_table,
    write_table_atomically,
)
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.dojos.file_managers import deck_split_index as deck_split_index_module
from src.dojos.file_managers.deck_split_index import DeckSplitIndex, deck_split_tag
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec, UnknownDeckSplitError
from src.schema.splits import Split
from tests.snapshot_fixtures import binder_with_cards, built_snapshots, stamped_deck_box


def _holdout(
    deck_split_ratios: tuple[float, float, float] | None = (1, 1, 1),
    held_out_games: frozenset[GameId] = frozenset(),
) -> HoldoutSpec:
    return HoldoutSpec(
        seed=1,
        tier_ratios=(1, 0, 0),
        held_out_games=held_out_games,
        deck_split_ratios=deck_split_ratios,
    )


def _snapshot(tmp_path: Path, deck_count: int) -> DeckSnapshot:
    binder = binder_with_cards(1)
    decks = [
        GenericDeck(uuid4(), GameId.MTG, "deck", list(binder.all_uuids(GameId.MTG)))
        for _ in range(deck_count)
    ]
    box = stamped_deck_box(decks, binder, tmp_path)
    return built_snapshots(binder, box, tmp_path)


def _deck_uuid(snapshot: DeckSnapshot, row: int) -> UUID:
    return UUID(bytes=snapshot.deck_uuid_bytes[row].tobytes())


class TestAssignment:
    def test_every_row_lands_in_its_holdout_split(self, tmp_path: Path) -> None:
        snapshot = _snapshot(tmp_path, 40)
        holdout = _holdout()

        index = DeckSplitIndex.load_or_build(snapshot, holdout, tmp_path / "splits")

        for split in Split:
            expected = [
                row
                for row in range(40)
                if holdout.deck_split_of(_deck_uuid(snapshot, row), GameId.MTG) == split
            ]
            assert index.rows_of(split).tolist() == expected

    def test_the_splits_partition_the_rows(self, tmp_path: Path) -> None:
        snapshot = _snapshot(tmp_path, 30)

        index = DeckSplitIndex.load_or_build(snapshot, _holdout(), tmp_path / "s")

        rows = np.concatenate([index.rows_of(split) for split in Split])
        assert sorted(rows.tolist()) == list(range(30))

    def test_rows_are_read_only_int64(self, tmp_path: Path) -> None:
        index = DeckSplitIndex.load_or_build(
            _snapshot(tmp_path, 5), _holdout(), tmp_path / "s"
        )
        rows = index.rows_of(Split.TRAIN)
        assert rows.dtype == np.int64
        assert not rows.flags.writeable

    def test_a_held_out_games_decks_are_all_validation(self, tmp_path: Path) -> None:
        holdout = _holdout(held_out_games=frozenset({GameId.MTG}))

        index = DeckSplitIndex.load_or_build(
            _snapshot(tmp_path, 10), holdout, tmp_path / "s"
        )

        assert len(index.rows_of(Split.VALIDATION)) == 10

    def test_an_unknown_deck_split_raises_before_writing(self, tmp_path: Path) -> None:
        snapshot = _snapshot(tmp_path, 2)
        with pytest.raises(UnknownDeckSplitError):
            DeckSplitIndex.load_or_build(
                snapshot, _holdout(deck_split_ratios=None), tmp_path / "s"
            )
        assert not (tmp_path / "s").exists()

    def test_an_empty_snapshot_has_empty_splits(self, tmp_path: Path) -> None:
        index = DeckSplitIndex.load_or_build(
            _snapshot(tmp_path, 0), _holdout(), tmp_path / "s"
        )
        assert all(len(index.rows_of(split)) == 0 for split in Split)


class TestCache:
    def test_writes_one_file_named_by_game_and_split_tag(self, tmp_path: Path) -> None:
        holdout = _holdout()
        DeckSplitIndex.load_or_build(_snapshot(tmp_path, 3), holdout, tmp_path / "s")

        tag = deck_split_tag(holdout.deck_split_key(GameId.MTG))
        assert [p.name for p in (tmp_path / "s").iterdir()] == [
            f"mtg.{tag}.deck_splits.arrow"
        ]

    def test_a_current_cache_is_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = _snapshot(tmp_path, 10)
        first = DeckSplitIndex.load_or_build(snapshot, _holdout(), tmp_path / "s")

        def _fail(*args: object) -> None:
            raise AssertionError("a current cache must not be rebuilt")

        monkeypatch.setattr(deck_split_index_module, "_assign_split_codes", _fail)
        reused = DeckSplitIndex.load_or_build(snapshot, _holdout(), tmp_path / "s")

        for split in Split:
            assert reused.rows_of(split).tolist() == first.rows_of(split).tolist()
        assert reused.snapshot_stamp == snapshot.stamp

    def test_another_snapshot_rebuilds_the_cache(self, tmp_path: Path) -> None:
        DeckSplitIndex.load_or_build(
            _snapshot(tmp_path / "a", 5), _holdout(), tmp_path / "s"
        )
        grown = _snapshot(tmp_path / "b", 8)

        index = DeckSplitIndex.load_or_build(grown, _holdout(), tmp_path / "s")

        assert index.snapshot_stamp == grown.stamp
        assert sum(len(index.rows_of(split)) for split in Split) == 8

    def test_another_deck_split_gets_its_own_file(self, tmp_path: Path) -> None:
        snapshot = _snapshot(tmp_path, 6)
        DeckSplitIndex.load_or_build(snapshot, _holdout((1, 0, 0)), tmp_path / "s")

        index = DeckSplitIndex.load_or_build(
            snapshot, _holdout((0, 1, 0)), tmp_path / "s"
        )

        assert len(index.rows_of(Split.TEST)) == 6
        assert len(list((tmp_path / "s").iterdir())) == 2

    def test_another_cache_format_is_a_miss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = _snapshot(tmp_path, 4)
        DeckSplitIndex.load_or_build(snapshot, _holdout((1, 0, 0)), tmp_path / "s")
        (path,) = (tmp_path / "s").iterdir()
        _overwrite_codes_and_format(path, code=1, format_version=b"0")

        index = DeckSplitIndex.load_or_build(
            snapshot, _holdout((1, 0, 0)), tmp_path / "s"
        )

        assert len(index.rows_of(Split.TRAIN)) == 4

    def test_an_unreadable_cache_is_rebuilt(self, tmp_path: Path) -> None:
        snapshot = _snapshot(tmp_path, 4)
        holdout = _holdout((1, 0, 0))
        DeckSplitIndex.load_or_build(snapshot, holdout, tmp_path / "s")
        (path,) = (tmp_path / "s").iterdir()
        path.write_bytes(b"junk")

        index = DeckSplitIndex.load_or_build(snapshot, holdout, tmp_path / "s")

        assert len(index.rows_of(Split.TRAIN)) == 4


def _overwrite_codes_and_format(path: Path, code: int, format_version: bytes) -> None:
    """Rewrite a cache file with every code set to code and another format."""
    table = open_mapped_table(path)
    metadata = dict(table.schema.metadata)
    metadata[b"deck_split_cache_format_version"] = format_version
    codes = pa.array(np.full(table.num_rows, code, dtype=np.uint8))
    schema = pa.schema([("split_code", pa.uint8())], metadata=metadata)
    rewritten = pa.Table.from_arrays([codes], schema=schema)
    del table
    write_table_atomically(rewritten, path)
