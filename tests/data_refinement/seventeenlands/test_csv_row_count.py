from pathlib import Path

from src.data_refinement.seventeenlands.csv_row_count import count_data_rows


class TestCountDataRows:
    def test_counts_data_rows_excluding_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")

        assert count_data_rows(csv_path) == 3

    def test_header_only_file_returns_zero(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n")

        assert count_data_rows(csv_path) == 0

    def test_empty_file_returns_zero_not_negative(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("")

        assert count_data_rows(csv_path) == 0
