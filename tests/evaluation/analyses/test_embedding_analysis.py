import json
from pathlib import Path

import pytest

from src.evaluation.analyses.embedding_analysis import (
    require_fresh_output_dir,
    write_analysis_result,
)


def test_a_missing_or_empty_output_dir_is_fresh(tmp_path: Path) -> None:
    require_fresh_output_dir(tmp_path / "absent")
    (tmp_path / "empty").mkdir()
    require_fresh_output_dir(tmp_path / "empty")


def test_an_output_dir_holding_anything_is_refused(tmp_path: Path) -> None:
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "scalars.json").write_text("{}")
    (tmp_path / "a_file").write_text("")
    for path in (tmp_path / "full", tmp_path / "a_file"):
        with pytest.raises(FileExistsError):
            require_fresh_output_dir(path)


def test_writes_scalars_json_and_lists_every_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "analysis"
    plot = tmp_path / "plot.png"
    result = write_analysis_result(out, {"b": 2.0, "a": 0.5}, extra_files=(plot,))
    assert result.files == (plot, out / "scalars.json")
    assert json.loads((out / "scalars.json").read_text()) == {"a": 0.5, "b": 2.0}
    with pytest.raises(TypeError):
        result.scalars["a"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_scalar_writes_nothing(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError):
        write_analysis_result(tmp_path / "out", {"score": bad})
    assert not (tmp_path / "out").exists()
