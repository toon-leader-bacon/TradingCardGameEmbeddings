import gzip
import json
from pathlib import Path

import pytest

from src.data_retrieval.resumable_crawl.crawl_files import (
    Cursor,
    CursorFile,
    JsonlShards,
    RejectLog,
    append_line,
    cut_torn_last_line,
    gzip_and_remove,
    refuse_orphan_cursor,
    write_text_atomically,
)


class TestCutTornLastLine:
    def test_missing_file_is_zero_lines(self, tmp_path: Path) -> None:
        assert cut_torn_last_line(tmp_path / "none.jsonl") == 0

    def test_whole_lines_are_counted_and_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_bytes(b"a\nb\n")
        assert cut_torn_last_line(path) == 2
        assert path.read_bytes() == b"a\nb\n"

    def test_a_torn_last_line_is_cut(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_bytes(b'a\nb\n{"par')
        assert cut_torn_last_line(path) == 2
        assert path.read_bytes() == b"a\nb\n"

    def test_a_file_with_only_a_torn_line_is_emptied(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_bytes(b'{"par')
        assert cut_torn_last_line(path) == 0
        assert path.read_bytes() == b""


def _gz_text(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8")


class TestDeckShards:
    def test_a_full_shard_is_gzipped_and_the_next_started(self, tmp_path: Path) -> None:
        shards = JsonlShards(tmp_path, rows_per_shard=2)
        for row in ["1", "2", "3"]:
            shards.append(row)
        assert _gz_text(tmp_path / "decks-00000.jsonl.gz") == "1\n2\n"
        assert not (tmp_path / "decks-00000.jsonl").exists()
        assert (tmp_path / "decks-00001.jsonl").read_bytes().decode() == "3\n"

    def test_reopening_continues_the_newest_shard_after_a_torn_line(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "decks-00000.jsonl.gz").write_bytes(gzip.compress(b"1\n2\n"))
        (tmp_path / "decks-00001.jsonl").write_bytes(b"3\n4")  # torn
        shards = JsonlShards(tmp_path, rows_per_shard=2)
        shards.append("5")
        shards.append("6")
        assert _gz_text(tmp_path / "decks-00001.jsonl.gz") == "3\n5\n"
        assert (tmp_path / "decks-00002.jsonl").read_bytes().decode() == "6\n"

    def test_reopening_after_a_crash_mid_compression_finishes_it(
        self, tmp_path: Path
    ) -> None:
        # Older plain shard never compressed; newest compressed, plain left
        (tmp_path / "decks-00000.jsonl").write_bytes(b"1\n2\n")
        (tmp_path / "decks-00001.jsonl.gz").write_bytes(gzip.compress(b"3\n4\n"))
        (tmp_path / "decks-00001.jsonl").write_bytes(b"3\n4\n")
        shards = JsonlShards(tmp_path, rows_per_shard=2)
        shards.append("5")
        assert _gz_text(tmp_path / "decks-00000.jsonl.gz") == "1\n2\n"
        assert not (tmp_path / "decks-00000.jsonl").exists()
        assert not (tmp_path / "decks-00001.jsonl").exists()
        assert (tmp_path / "decks-00002.jsonl").read_bytes().decode() == "5\n"

    def test_rows_per_shard_below_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            JsonlShards(tmp_path, rows_per_shard=0)

    def test_a_full_plain_shard_is_gzipped_on_reopen(self, tmp_path: Path) -> None:
        (tmp_path / "decks-00000.jsonl").write_bytes(b"1\n2\n")
        shards = JsonlShards(tmp_path, rows_per_shard=2)
        shards.append("3")
        assert _gz_text(tmp_path / "decks-00000.jsonl.gz") == "1\n2\n"
        assert (tmp_path / "decks-00001.jsonl").read_bytes().decode() == "3\n"


class TestCursorFile:
    def _file(self, tmp_path: Path) -> CursorFile:
        return CursorFile(tmp_path / "cursor.json", "ab12", 10)

    def test_absent_reads_as_the_start(self, tmp_path: Path) -> None:
        assert self._file(tmp_path).read() == Cursor(0, 0)

    def test_round_trips(self, tmp_path: Path) -> None:
        self._file(tmp_path).write(3, failures=2)
        assert self._file(tmp_path).read() == Cursor(3, 2)
        assert json.loads((tmp_path / "cursor.json").read_text()) == {
            "position": 3,
            "failures": 2,
            "plan": "ab12",
        }

    def test_missing_failures_read_as_zero(self, tmp_path: Path) -> None:
        (tmp_path / "cursor.json").write_text('{"position": 4, "plan": "ab12"}')
        assert self._file(tmp_path).read() == Cursor(4, 0)

    @pytest.mark.parametrize(
        "body",
        [
            "not json",
            "[]",
            '{"position": 1}',
            '{"position": "1", "plan": "ab12"}',
            '{"position": true, "plan": "ab12"}',
            '{"position": 1, "plan": "other"}',
            '{"position": 11, "plan": "ab12"}',
            '{"position": -1, "plan": "ab12"}',
        ],
    )
    def test_bad_cursors_raise(self, tmp_path: Path, body: str) -> None:
        (tmp_path / "cursor.json").write_text(body)
        with pytest.raises(ValueError):
            self._file(tmp_path).read()

    def test_the_end_position_is_valid(self, tmp_path: Path) -> None:
        self._file(tmp_path).write(10)
        assert self._file(tmp_path).read() == Cursor(10, 0)


class TestWriteTextAtomically:
    def test_replaces_with_lf_bytes_and_leaves_no_temp(self, tmp_path: Path) -> None:
        path = tmp_path / "x.tsv"
        path.write_text("old")
        write_text_atomically(path, "a\tb\nc\n")
        assert path.read_bytes() == b"a\tb\nc\n"
        assert list(tmp_path.iterdir()) == [path]


class TestRefuseOrphanCursor:
    def test_no_cursor_passes(self, tmp_path: Path) -> None:
        refuse_orphan_cursor(tmp_path, "plan.json")

    def test_a_cursor_raises(self, tmp_path: Path) -> None:
        (tmp_path / "cursor.json").write_text("{}")
        with pytest.raises(RuntimeError, match="plan.json"):
            refuse_orphan_cursor(tmp_path, "plan.json")


class TestRejectLog:
    def test_appends_rows_after_cutting_a_torn_line(self, tmp_path: Path) -> None:
        path = tmp_path / "rejects.jsonl"
        path.write_bytes(b'{"position": 0, "id": 9, "reason": "x"}\n{"pos')
        log = RejectLog(path)
        log.append(7, 4242, "HTTP 410")
        log.append(8, "abc", "HTTP 403")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows == [
            {"position": 0, "id": 9, "reason": "x"},
            {"position": 7, "id": 4242, "reason": "HTTP 410"},
            {"position": 8, "id": "abc", "reason": "HTTP 403"},
        ]


class TestGzipAndRemove:
    def test_round_trips_and_deletes_the_plain_file(self, tmp_path: Path) -> None:
        plain = tmp_path / "decks-00000.jsonl"
        plain.write_bytes(b"1\n2\n")
        gzip_and_remove(plain)
        assert _gz_text(tmp_path / "decks-00000.jsonl.gz") == "1\n2\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["decks-00000.jsonl.gz"]

    def test_a_missing_plain_file_is_a_no_op(self, tmp_path: Path) -> None:
        gzip_and_remove(tmp_path / "decks-00000.jsonl")
        assert list(tmp_path.iterdir()) == []


class TestAppendLine:
    def test_appends_lf_terminated_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        append_line(path, "a")
        append_line(path, "b")
        assert path.read_bytes() == b"a\nb\n"
