"""Tests for partition.py: SeventeenLandsPartition and find_partitions."""

from pathlib import Path

import pytest

from src.data_refinement.metrics.seventeenlands.partition import (
    SeventeenLandsPartition,
    find_partitions,
)
from src.data_retrieval.seventeenlands.refs import DataType, Expansion, FormatCode

_KTK_TRAD = SeventeenLandsPartition(
    DataType.GAME, "drawn_win_rate", Expansion.KTK, FormatCode.TradDraft
)


def test_path_follows_the_partition_layout() -> None:
    assert _KTK_TRAD.path() == Path(
        "data/metrics/seventeenlands/game_data/drawn_win_rate/KTK/TradDraft.parquet"
    )


def test_path_under_another_root(tmp_path: Path) -> None:
    assert _KTK_TRAD.path(tmp_path).is_relative_to(tmp_path)


def test_from_path_inverts_path(tmp_path: Path) -> None:
    assert SeventeenLandsPartition.from_path(_KTK_TRAD.path(tmp_path)) == _KTK_TRAD


def test_from_path_reads_a_set_with_punctuation(tmp_path: Path) -> None:
    cube = SeventeenLandsPartition(
        DataType.DRAFT, "card_take_rate", Expansion.Powered_Cube, FormatCode.Sealed
    )

    assert SeventeenLandsPartition.from_path(cube.path(tmp_path)) == cube


@pytest.mark.parametrize(
    "path",
    [
        "data/metrics/seventeenlands/game_data/drawn_win_rate/KTK/TradDraft.csv",
        "data/metrics/other/game_data/drawn_win_rate/KTK/TradDraft.parquet",
        "data/metrics/seventeenlands/game_data/drawn_win_rate/XXX/TradDraft.parquet",
        "data/metrics/seventeenlands/game_data/drawn_win_rate/KTK/Nope.parquet",
        "TradDraft.parquet",
    ],
)
def test_from_path_rejects_other_shapes(path: str) -> None:
    with pytest.raises(ValueError):
        SeventeenLandsPartition.from_path(Path(path))


@pytest.mark.parametrize("stem", ["", "slices", "a.b", "a/b"])
def test_rejects_a_stem_that_cannot_be_a_metric_directory(stem: str) -> None:
    with pytest.raises(ValueError, match="not a valid metric stem"):
        SeventeenLandsPartition(DataType.GAME, stem, Expansion.KTK, FormatCode.Sealed)


def test_find_partitions_lists_every_file_sorted(tmp_path: Path) -> None:
    partitions = [
        SeventeenLandsPartition(DataType.GAME, "m", expansion, fmt)
        for expansion, fmt in [
            (Expansion.MSH, FormatCode.PremierDraft),
            (Expansion.KTK, FormatCode.TradDraft),
            (Expansion.KTK, FormatCode.Sealed),
        ]
    ]
    for partition in partitions:
        partition.path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        partition.path(tmp_path).write_bytes(b"")

    found = find_partitions(DataType.GAME, "m", tmp_path)

    assert [(p.expansion, p.format) for p in found] == [
        (Expansion.KTK, FormatCode.Sealed),
        (Expansion.KTK, FormatCode.TradDraft),
        (Expansion.MSH, FormatCode.PremierDraft),
    ]


def test_find_partitions_is_empty_for_a_metric_with_no_output(tmp_path: Path) -> None:
    assert find_partitions(DataType.GAME, "missing", tmp_path) == []


def test_find_partitions_rejects_a_stray_file(tmp_path: Path) -> None:
    stray = tmp_path / "seventeenlands/game_data/m/KTK/Nope.parquet"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"")

    with pytest.raises(ValueError):
        find_partitions(DataType.GAME, "m", tmp_path)
