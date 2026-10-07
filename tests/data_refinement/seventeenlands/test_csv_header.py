"""Tests for csv_header.py's read_csv_header."""

from pathlib import Path

import pytest

from src.data_refinement.seventeenlands.csv_header import read_csv_header


def test_the_header_is_read_from_the_first_line(tmp_path: Path) -> None:
    csv_path = tmp_path / "KTK.TradDraft.csv"
    csv_path.write_text('a,"b,c",d\n1,2,3\n', encoding="utf-8")

    assert read_csv_header(csv_path) == ("a", "b,c", "d")


def test_a_byte_order_mark_is_dropped(tmp_path: Path) -> None:
    csv_path = tmp_path / "KTK.TradDraft.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8-sig")

    assert read_csv_header(csv_path) == ("a", "b")


def test_an_empty_csv_has_no_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "KTK.TradDraft.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        read_csv_header(csv_path)
