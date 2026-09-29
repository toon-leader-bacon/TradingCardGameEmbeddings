import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from src.evaluation.embedding_table import (
    CardRow,
    EmbeddingTable,
    EmbeddingTableMetadata,
    NonFiniteEmbeddingError,
    require_finite_vectors,
)
from src.schema.game_id import GameId

_WIDTH = 3


def _metadata(**overrides: object) -> EmbeddingTableMetadata:
    fields: dict = dict(
        encoder_label="single_v1",
        checkpoint_dir=Path("runs/a/pretrain_round0007"),
        embedding_dim=_WIDTH,
        binder_versions={GameId.MTG: "mtg-v1", GameId.GWENT: "gwent-v1"},
        created_at=datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc),
    )
    fields.update(overrides)
    return EmbeddingTableMetadata(**fields)


def _rows(count: int, game: GameId = GameId.MTG) -> list[CardRow]:
    return [CardRow(uuid4(), game) for _ in range(count)]


def _vectors(count: int, start: float = 0.0) -> np.ndarray:
    return np.arange(count * _WIDTH, dtype=np.float32).reshape(count, _WIDTH) + start


class TestMetadata:
    @pytest.mark.parametrize(
        "overrides",
        [{"encoder_label": ""}, {"embedding_dim": 0}, {"binder_versions": {}}],
    )
    def test_invalid_fields_are_rejected(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            _metadata(**overrides)

    def test_binder_versions_is_a_read_only_copy(self) -> None:
        versions = {GameId.MTG: "mtg-v1"}
        metadata = _metadata(binder_versions=versions)
        versions[GameId.GWENT] = "gwent-v1"
        assert metadata.games == frozenset({GameId.MTG})
        with pytest.raises(TypeError):
            metadata.binder_versions[GameId.GWENT] = "x"  # type: ignore[index]


class TestCreateAndOpen:
    def test_metadata_round_trips_through_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "a.db"
        with EmbeddingTable.create(path, _metadata()):
            pass
        with EmbeddingTable.open(path) as table:
            assert table.metadata == _metadata()

    def test_an_untrained_encoder_has_no_checkpoint_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "a.db"
        EmbeddingTable.create(path, _metadata(checkpoint_dir=None)).close()
        with EmbeddingTable.open(path) as table:
            assert table.metadata.checkpoint_dir is None

    def test_create_refuses_an_existing_path(self, tmp_path: Path) -> None:
        path = tmp_path / "a.db"
        path.write_text("")
        with pytest.raises(FileExistsError):
            EmbeddingTable.create(path, _metadata())

    def test_open_refuses_a_missing_path_without_creating_it(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            EmbeddingTable.open(tmp_path / "absent.db")
        assert not (tmp_path / "absent.db").exists()

    def test_open_rejects_a_file_that_is_not_sqlite(self, tmp_path: Path) -> None:
        path = tmp_path / "a.db"
        path.write_text("not a database at all, just some text " * 10)
        with pytest.raises(ValueError, match="not an EmbeddingTable"):
            EmbeddingTable.open(path)

    def test_open_rejects_another_sqlite_file(self, tmp_path: Path) -> None:
        path = tmp_path / "deck_box.db"
        connection = sqlite3.connect(str(path))
        connection.execute("CREATE TABLE decks (id TEXT)")
        connection.commit()
        connection.close()
        with pytest.raises(ValueError, match="not an EmbeddingTable"):
            EmbeddingTable.open(path)

    def test_open_rejects_unreadable_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "a.db"
        EmbeddingTable.create(path, _metadata()).close()
        connection = sqlite3.connect(str(path))
        connection.execute("UPDATE table_info SET value = '{}' WHERE key = 'metadata'")
        connection.commit()
        connection.close()
        with pytest.raises(ValueError, match="unreadable metadata"):
            EmbeddingTable.open(path)

    def test_a_closed_table_refuses_further_use(self, tmp_path: Path) -> None:
        table = EmbeddingTable.create(tmp_path / "a.db", _metadata())
        table.close()
        table.close()  # safe twice
        with pytest.raises(sqlite3.ProgrammingError):
            table.rows()


class TestAddAndRead:
    def test_added_vectors_read_back_in_the_requested_order(
        self, tmp_path: Path
    ) -> None:
        rows = _rows(3)
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            table.add(rows, _vectors(3))
            wanted = [rows[2], rows[0], rows[2]]  # reordered, with a repeat
            result = table.vectors_for(wanted)
        expected = _vectors(3)[[2, 0, 2]]
        assert result.dtype == np.float32
        assert np.array_equal(result, expected)

    def test_rows_are_sorted_by_uuid_whatever_the_insertion_order(
        self, tmp_path: Path
    ) -> None:
        rows = _rows(20)
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as first:
            first.add(rows, _vectors(20))
            forward = first.rows()
        with EmbeddingTable.create(tmp_path / "b.db", _metadata()) as second:
            second.add(rows[::-1], _vectors(20))
            backward = second.rows()
        assert forward == backward == sorted(rows, key=lambda row: row.nocab_uuid)

    def test_an_already_stored_card_is_left_unchanged(self, tmp_path: Path) -> None:
        rows = _rows(1)
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            table.add(rows, _vectors(1))
            table.add(rows, _vectors(1, start=100.0))
            assert np.array_equal(table.vectors_for(rows), _vectors(1))
            assert len(table.rows()) == 1

    def test_has_card(self, tmp_path: Path) -> None:
        rows = _rows(1)
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            assert not table.has_card(rows[0].nocab_uuid)
            table.add(rows, _vectors(1))
            assert table.has_card(rows[0].nocab_uuid)

    def test_adds_survive_reopening(self, tmp_path: Path) -> None:
        path = tmp_path / "a.db"
        rows = _rows(2, GameId.GWENT)
        with EmbeddingTable.create(path, _metadata()) as table:
            table.add(rows, _vectors(2))
        with EmbeddingTable.open(path) as table:
            assert set(table.rows()) == set(rows)

    def test_empty_reads(self, tmp_path: Path) -> None:
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            assert table.rows() == []
            assert table.vectors_for([]).shape == (0, _WIDTH)

    def test_an_unknown_card_raises_key_error(self, tmp_path: Path) -> None:
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            with pytest.raises(KeyError):
                table.vectors_for(_rows(1))

    def test_reads_more_cards_than_one_sqlite_statement_can_bind(
        self, tmp_path: Path
    ) -> None:
        rows = _rows(2100)
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            table.add(rows, _vectors(2100))
            assert np.array_equal(table.vectors_for(rows), _vectors(2100))

    @pytest.mark.parametrize(
        "vectors",
        [
            np.zeros((2, _WIDTH), dtype=np.float32),  # too few rows
            np.zeros((3, _WIDTH + 1), dtype=np.float32),  # too wide
            np.zeros(3 * _WIDTH, dtype=np.float32),  # flat
        ],
    )
    def test_a_shape_mismatch_writes_nothing(
        self, tmp_path: Path, vectors: np.ndarray
    ) -> None:
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            with pytest.raises(ValueError, match="shape"):
                table.add(_rows(3), vectors)
            assert table.rows() == []

    def test_a_game_the_table_was_not_created_for_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        metadata = _metadata(binder_versions={GameId.MTG: "mtg-v1"})
        rows = _rows(1) + _rows(1, GameId.GWENT)
        with EmbeddingTable.create(tmp_path / "a.db", metadata) as table:
            with pytest.raises(ValueError, match="gwent"):
                table.add(rows, _vectors(2))
            assert table.rows() == []

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e39])
    def test_non_finite_values_write_nothing(self, tmp_path: Path, bad: float) -> None:
        # 1e39 is finite as float64 but infinite once stored as float32
        vectors = np.zeros((2, _WIDTH), dtype=np.float64)
        vectors[1, 2] = bad
        with EmbeddingTable.create(tmp_path / "a.db", _metadata()) as table:
            with pytest.raises(NonFiniteEmbeddingError):
                table.add(_rows(2), vectors)
            assert table.rows() == []


def test_require_finite_vectors_accepts_finite_values() -> None:
    require_finite_vectors(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(NonFiniteEmbeddingError, match="1 NaN"):
        require_finite_vectors(np.array([[0.0, float("nan")]]))


def test_a_failed_create_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.embedding_table as module

    def failing_write(*args: object) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(module, "_write_table_info", failing_write)
    path = tmp_path / "a.db"
    with pytest.raises(sqlite3.OperationalError):
        EmbeddingTable.create(path, _metadata())
    assert not path.exists()


def test_a_naive_created_at_round_trips(tmp_path: Path) -> None:
    metadata = _metadata(created_at=datetime(2026, 9, 29, 12, 0))
    EmbeddingTable.create(tmp_path / "a.db", metadata).close()
    with EmbeddingTable.open(tmp_path / "a.db") as table:
        assert table.metadata == metadata
