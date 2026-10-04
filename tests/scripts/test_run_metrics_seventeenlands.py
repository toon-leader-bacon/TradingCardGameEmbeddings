"""Tests for scripts/run_metrics.py's seventeenlands driver helpers:
output re-rooting, the family spec check, and metric construction."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    WinRateWhenInDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)
from src.data_retrieval.seventeenlands.refs import DataType, Expansion, format_code
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    HEADER,
    VERSION,
    binder_with_cards,
)

_SCRIPT = Path("scripts/run_metrics.py")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_metrics_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family(script: ModuleType, specs: tuple, deck_box_path: Path | None):
    return script._SeventeenLandsFamily(
        name="test_family",
        data_type=DataType.GAME,
        family_dir=Path("unused"),
        metric_specs=specs,
        scan_for_csv=Mock(),
        deck_box_output_path=deck_box_path,
    )


def _context(script: ModuleType, deck_box: DeckBox | None) -> object:
    return script._CsvMetricContext(
        version_metadata=VERSION,
        expansion=Expansion.KTK,
        format_code=format_code.TradDraft,
        output_root=Path("scratch"),
        deck_box=deck_box,
    )


def test_output_paths_are_partition_paths(script: ModuleType) -> None:
    family = _family(script, (), None)
    spec = script.ChunkMetricSpec(WinRateWhenInDeckMetric)
    context = script._CsvMetricContext(
        **{**vars(_context(script, None)), "output_root": None}
    )

    path = script._partition_path(family, spec, context)

    assert path == Path(
        "data/metrics/seventeenlands/game_data/win_rate_when_in_deck/KTK/TradDraft.parquet"
    )


def test_an_output_root_re_roots_the_partition_path(script: ModuleType) -> None:
    family = _family(script, (), None)
    spec = script.ChunkMetricSpec(WinRateWhenInDeckMetric)

    path = script._partition_path(family, spec, _context(script, None))

    assert path == Path(
        "scratch/seventeenlands/game_data/win_rate_when_in_deck/KTK/TradDraft.parquet"
    )


def test_csv_names_parse_into_set_and_format(script: ModuleType) -> None:
    assert script._parse_expansion_format(Path("x/Cube_-_Powered.Sealed.csv")) == (
        Expansion.Powered_Cube,
        format_code.Sealed,
    )


@pytest.mark.parametrize("name", ["KTK.csv", "XXX.Sealed.csv", "KTK.Nope.csv"])
def test_unknown_csv_names_are_rejected(script: ModuleType, name: str) -> None:
    with pytest.raises(ValueError):
        script._parse_expansion_format(Path(name))


def test_an_output_root_also_re_roots_the_deck_box(script: ModuleType) -> None:
    box = Path("data/metrics/seventeenlands/game_data/deck_box.db")

    assert script._deck_box_path(box, None) == box
    assert script._deck_box_path(box, Path("scratch")) == Path(
        "scratch/seventeenlands/game_data/deck_box.db"
    )
    assert script._deck_box_path(None, Path("scratch")) is None


@pytest.mark.parametrize("metrics_name", ["_GAME_DATA_METRICS", "_REPLAY_DATA_METRICS"])
def test_a_family_with_a_deck_box_path_passes_the_spec_check(
    script: ModuleType, metrics_name: str
) -> None:
    family = _family(script, getattr(script, metrics_name), Path("box.db"))

    script._check_family_specs(family)


def test_a_family_without_deck_box_metrics_needs_no_deck_box_path(
    script: ModuleType,
) -> None:
    script._check_family_specs(_family(script, script._DRAFT_DATA_METRICS, None))


def test_a_deck_box_metric_without_a_deck_box_path_is_rejected(
    script: ModuleType,
) -> None:
    family = _family(script, script._GAME_DATA_METRICS, None)

    with pytest.raises(ValueError, match="no deck box path"):
        script._check_family_specs(family)


def test_metrics_are_built_in_spec_order(script: ModuleType) -> None:
    specs = (
        script.ChunkMetricSpec(script.WinRateWhenInDeckMetric),
        script.DeckBoxChunkMetricSpec(script.DeckWinPredictionMetric),
    )
    family = _family(script, specs, Path("box.db"))

    metrics = script._namespaced_metrics(family, _context(script, DeckBox()))

    assert [type(metric) for metric in metrics] == [
        script.WinRateWhenInDeckMetric,
        script.DeckWinPredictionMetric,
    ]


def test_a_deck_box_chunk_metric_gets_the_family_box(
    script: ModuleType, tmp_path: Path
) -> None:
    context = _context(script, deck_box=DeckBox())
    spec = script.DeckBoxChunkMetricSpec(script.OnPlayWinRateSensitivityByDeckMetric)

    metric = script._build_chunk_metric(spec, context, tmp_path / "out.parquet")

    assert isinstance(metric, script.OnPlayWinRateSensitivityByDeckMetric)
    assert metric._deck_box is context.deck_box


def test_a_deck_box_chunk_metric_without_a_box_is_rejected(
    script: ModuleType,
) -> None:
    spec = script.DeckBoxChunkMetricSpec(script.DeckWinPredictionMetric)

    with pytest.raises(ValueError, match="needs the family DeckBox"):
        script._build_chunk_metric(
            spec, _context(script, deck_box=None), Path("unused.parquet")
        )


def test_a_chunk_scan_binds_a_parser_built_from_the_header(
    script: ModuleType,
) -> None:
    scan = script._GAME_DATA_SCAN(HEADER, binder_with_cards(["Owlbear"]))

    assert isinstance(scan.keywords["parser"], GameDataChunkParser)


def test_the_header_is_read_from_the_first_line(
    script: ModuleType, tmp_path: Path
) -> None:
    csv_path = tmp_path / "KTK.TradDraft.csv"
    csv_path.write_text('a,"b,c",d\n1,2,3\n', encoding="utf-8")

    assert script._read_header(csv_path) == ("a", "b,c", "d")


def test_an_empty_csv_has_no_header(script: ModuleType, tmp_path: Path) -> None:
    csv_path = tmp_path / "KTK.TradDraft.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        script._read_header(csv_path)
