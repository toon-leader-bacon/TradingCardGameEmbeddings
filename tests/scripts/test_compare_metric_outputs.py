"""Tests for scripts/compare_metric_outputs.py's parity rule."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_SCRIPT = Path("scripts/compare_metric_outputs.py")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_metric_outputs_script", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, frame: pd.DataFrame, version: str = "v1") -> None:
    write_dataframe_with_version_metadata(
        frame, path, MetricVersionMetadata(game=GameId.MTG, card_binder_version=version)
    )


def _frame(
    label: float = 0.5, count: int = 2, uuids: tuple = ("a", "b")
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nocab_uuid": list(uuids),
            "rate": [label] * len(uuids),
            "sample_count": [count] * len(uuids),
        }
    )


def _compare(script: ModuleType, tmp_path: Path) -> dict[str, str]:
    results = script.compare_trees(tmp_path / "ref", tmp_path / "cand", 1e-9)
    return {str(r.relative_path): r.status.name for r in results}


def test_identical_files_in_any_row_order_match(
    script: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "ref/m.parquet", _frame())
    _write(tmp_path / "cand/m.parquet", _frame().iloc[::-1])

    assert _compare(script, tmp_path) == {"m.parquet": "MATCH"}


def test_floats_within_tolerance_match(script: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path / "ref/m.parquet", _frame(label=0.5))
    _write(tmp_path / "cand/m.parquet", _frame(label=0.5 + 1e-12))

    assert _compare(script, tmp_path) == {"m.parquet": "MATCH"}


def test_a_float_beyond_tolerance_is_a_mismatch(
    script: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "ref/m.parquet", _frame(label=0.5))
    _write(tmp_path / "cand/m.parquet", _frame(label=0.51))

    assert _compare(script, tmp_path) == {"m.parquet": "MISMATCH"}


def test_a_different_count_is_a_mismatch(script: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path / "ref/m.parquet", _frame(count=2))
    _write(tmp_path / "cand/m.parquet", _frame(count=3))

    assert _compare(script, tmp_path) == {"m.parquet": "MISMATCH"}


def test_a_different_key_set_is_a_mismatch(script: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path / "ref/m.parquet", _frame(uuids=("a", "b")))
    _write(tmp_path / "cand/m.parquet", _frame(uuids=("a", "c")))

    assert _compare(script, tmp_path) == {"m.parquet": "MISMATCH"}


def test_files_on_one_side_only_are_reported(
    script: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "ref/only_ref.parquet", _frame())
    _write(tmp_path / "cand/only_cand.parquet", _frame())

    assert _compare(script, tmp_path) == {
        "only_cand.parquet": "MISSING_REFERENCE",
        "only_ref.parquet": "MISSING_CANDIDATE",
    }


def test_a_different_binder_version_is_a_stale_reference(
    script: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "ref/m.parquet", _frame(), version="old")
    _write(tmp_path / "cand/m.parquet", _frame(), version="new")

    assert _compare(script, tmp_path) == {"m.parquet": "STALE_REFERENCE"}


def test_a_columnless_empty_reference_matches_an_empty_candidate(
    script: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "ref/m.parquet", pd.DataFrame([]))
    _write(tmp_path / "cand/m.parquet", _frame().iloc[0:0])

    assert _compare(script, tmp_path) == {"m.parquet": "MATCH"}
