"""Tests for scripts/run_metrics.py's seventeenlands driver helpers:
output re-rooting, the family spec check, and metric construction."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.rowwise_metric import RowwiseMetric
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


class _FakeRowMetric:
    """A row metric class: built from (binder, header, game, output_path)."""

    DEFAULT_OUTPUT_PATH = Path("data/metrics/seventeenlands/x/fake.parquet")

    def __init__(self, binder, header, source_game, output_path=None) -> None:
        self.output_path = output_path

    def accumulate(self, row: dict) -> None:
        return

    def finalize(self) -> Path:
        return Path("unused")


class _FakeDeckBoxChunkMetric:
    """A chunk metric class that also takes the family deck box: built
    from (version_metadata, deck_box, output_path)."""

    DEFAULT_OUTPUT_PATH = Path("data/metrics/seventeenlands/x/fake_deck_box.parquet")

    def __init__(self, version_metadata, deck_box, output_path=None) -> None:
        self._deck_box = deck_box
        self.output_path = output_path

    def accumulate(self, chunk) -> None:
        return

    def finalize(self) -> Path:
        return Path("unused")


def _family(
    script: ModuleType, specs: tuple, scanning: object, deck_box_path: Path | None
):
    return script._SeventeenLandsFamily(
        name="test_family",
        family_dir=Path("unused"),
        metric_specs=specs,
        scanning=scanning,
        deck_box_output_path=deck_box_path,
    )


def _row_scanning(script: ModuleType) -> object:
    return script.RowScanning(lambda path, metrics: None)


def _chunk_scanning(script: ModuleType, frame_of: object = None) -> object:
    return script.ChunkScanning(Mock(), frame_of=frame_of)


def _context(script: ModuleType, deck_box: DeckBox | None) -> object:
    return script._CsvMetricContext(
        binder=binder_with_cards(["Owlbear"]),
        header=pd.Index(HEADER),
        source_game=VERSION.game,
        version_metadata=VERSION,
        expansion="KTK",
        format_code="TradDraft",
        output_root=Path("scratch"),
        deck_box=deck_box,
    )


def test_output_paths_are_namespaced_by_expansion_and_format(
    script: ModuleType,
) -> None:
    default = Path(
        "data/metrics/seventeenlands/game_data/win_rate_when_in_deck.parquet"
    )

    path = script._namespaced_output_path(default, "KTK", "TradDraft", None)

    assert path == Path(
        "data/metrics/seventeenlands/game_data/KTK/TradDraft/win_rate_when_in_deck.parquet"
    )


def test_an_output_root_re_roots_the_namespaced_path(script: ModuleType) -> None:
    default = Path(
        "data/metrics/seventeenlands/game_data/win_rate_when_in_deck.parquet"
    )

    path = script._namespaced_output_path(default, "KTK", "TradDraft", Path("scratch"))

    assert path == Path(
        "scratch/seventeenlands/game_data/KTK/TradDraft/win_rate_when_in_deck.parquet"
    )


def test_an_output_root_also_re_roots_the_deck_box(script: ModuleType) -> None:
    box = Path("data/metrics/seventeenlands/game_data/deck_box.db")

    assert script._deck_box_path(box, None) == box
    assert script._deck_box_path(box, Path("scratch")) == Path(
        "scratch/seventeenlands/game_data/deck_box.db"
    )
    assert script._deck_box_path(None, Path("scratch")) is None


def test_game_data_is_all_chunk_metrics_and_keeps_no_source_frame(
    script: ModuleType,
) -> None:
    family = _family(
        script,
        script._GAME_DATA_METRICS,
        _chunk_scanning(script),
        Path("data/metrics/x/deck_box.db"),
    )

    assert script._check_family_specs(family) is False


def test_a_chunk_family_still_wrapping_row_metrics_keeps_a_source_frame(
    script: ModuleType,
) -> None:
    specs = (script.RowMetricSpec(_FakeRowMetric),)
    family = _family(script, specs, _chunk_scanning(script, frame_of=Mock()), None)

    assert script._check_family_specs(family) is True


def test_a_row_family_never_keeps_a_source_frame(script: ModuleType) -> None:
    family = _family(script, script._DRAFT_DATA_METRICS, _row_scanning(script), None)

    assert script._check_family_specs(family) is False


def test_a_chunk_metric_in_a_row_family_is_rejected(script: ModuleType) -> None:
    family = _family(
        script,
        script._GAME_DATA_METRICS,
        _row_scanning(script),
        Path("data/metrics/x/box.db"),
    )

    with pytest.raises(ValueError, match="chunk metric"):
        script._check_family_specs(family)


def test_a_row_metric_in_a_chunk_family_without_frame_of_is_rejected(
    script: ModuleType,
) -> None:
    specs = (script.RowMetricSpec(_FakeRowMetric),)
    family = _family(script, specs, _chunk_scanning(script), None)

    with pytest.raises(ValueError, match="no frame_of"):
        script._check_family_specs(family)


def test_a_deck_box_metric_without_a_deck_box_path_is_rejected(
    script: ModuleType,
) -> None:
    specs = (script.DeckBoxChunkMetricSpec(_FakeDeckBoxChunkMetric),)
    family = _family(script, specs, _chunk_scanning(script), None)

    with pytest.raises(ValueError, match="no deck box path"):
        script._check_family_specs(family)


def test_row_metrics_are_wrapped_only_in_a_chunk_family(script: ModuleType) -> None:
    context = _context(script, deck_box=None)
    row_spec = script.RowMetricSpec(_FakeRowMetric)
    chunk_spec = script.ChunkMetricSpec(script.WinRateWhenInDeckMetric)

    chunk_family = _family(
        script, (chunk_spec, row_spec), _chunk_scanning(script, Mock()), None
    )
    chunk_metric, wrapped = script._namespaced_metrics(chunk_family, context)
    row_family = _family(script, (row_spec,), _row_scanning(script), None)
    (unwrapped,) = script._namespaced_metrics(row_family, context)

    assert isinstance(chunk_metric, script.WinRateWhenInDeckMetric)
    assert isinstance(wrapped, RowwiseMetric)
    assert isinstance(unwrapped, _FakeRowMetric)


def test_a_deck_box_chunk_metric_gets_the_family_box(
    script: ModuleType, tmp_path: Path
) -> None:
    context = _context(script, deck_box=DeckBox())
    spec = script.DeckBoxChunkMetricSpec(_FakeDeckBoxChunkMetric)

    metric = script._build_chunk_metric(spec, context, tmp_path / "out.parquet")

    assert isinstance(metric, _FakeDeckBoxChunkMetric)
    assert metric._deck_box is context.deck_box


def test_a_deck_box_chunk_metric_without_a_box_is_rejected(
    script: ModuleType,
) -> None:
    spec = script.DeckBoxChunkMetricSpec(_FakeDeckBoxChunkMetric)

    with pytest.raises(ValueError, match="needs the family DeckBox"):
        script._build_chunk_metric(
            spec, _context(script, deck_box=None), Path("unused.parquet")
        )


def test_game_data_scans_refuse_a_source_frame(script: ModuleType) -> None:
    with pytest.raises(ValueError, match="no source frame"):
        script._scan_game_data_csv(
            pd.Index(HEADER), binder_with_cards(["Owlbear"]), True
        )


def test_a_row_scanning_family_scans_every_csv_the_same_way(
    script: ModuleType,
) -> None:
    def scan(path: Path, metrics: list) -> None:
        return None

    scanning = script.RowScanning(scan)

    assert scanning.scan_for_csv(pd.Index(HEADER), Mock(), False) is scan
