"""Tests for scripts/run_metrics.py's seventeenlands driver helpers:
output re-rooting, the family spec check, and metric construction."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

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


def _family(
    script: ModuleType, specs: tuple, frame_of: object, deck_box_path: Path | None
):
    return script._SeventeenLandsFamily(
        name="test_family",
        family_dir=Path("unused"),
        metric_specs=specs,
        scan_for_csv=script._same_scan_for_every_csv(lambda path, metrics: None),
        frame_of=frame_of,
        deck_box_output_path=deck_box_path,
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


def test_game_data_keeps_a_source_frame_while_row_metrics_remain(
    script: ModuleType,
) -> None:
    family = _family(
        script, script._GAME_DATA_METRICS, Mock(), Path("data/metrics/x/deck_box.db")
    )

    assert script._check_family_specs(family) is True


def test_a_chunk_only_family_needs_no_source_frame(script: ModuleType) -> None:
    chunk_specs = tuple(
        spec
        for spec in script._GAME_DATA_METRICS
        if isinstance(spec, script.ChunkMetricSpec)
    )

    assert (
        script._check_family_specs(_family(script, chunk_specs, Mock(), None)) is False
    )


def test_a_row_family_never_keeps_a_source_frame(script: ModuleType) -> None:
    family = _family(script, script._DRAFT_DATA_METRICS, None, None)

    assert script._check_family_specs(family) is False


def test_a_chunk_metric_in_a_row_family_is_rejected(script: ModuleType) -> None:
    family = _family(
        script, script._GAME_DATA_METRICS, None, Path("data/metrics/x/box.db")
    )

    with pytest.raises(ValueError, match="chunk metric"):
        script._check_family_specs(family)


def test_a_deck_box_metric_without_a_deck_box_path_is_rejected(
    script: ModuleType,
) -> None:
    family = _family(script, script._GAME_DATA_METRICS, Mock(), None)

    with pytest.raises(ValueError, match="no deck box path"):
        script._check_family_specs(family)


def test_row_metrics_are_wrapped_only_in_a_chunk_family(script: ModuleType) -> None:
    binder = binder_with_cards(["Owlbear"])
    context = script._CsvMetricContext(
        binder=binder,
        header=pd.Index(HEADER),
        source_game=VERSION.game,
        version_metadata=VERSION,
        expansion="KTK",
        format_code="TradDraft",
        output_root=Path("scratch"),
        deck_box=None,
    )
    row_spec = script.RowMetricSpec(script.OnPlayWinRateDeltaMetric)
    chunk_spec = script.ChunkMetricSpec(script.WinRateWhenInDeckMetric)

    chunk_family = _family(script, (chunk_spec, row_spec), Mock(), None)
    chunk_metric, wrapped = script._namespaced_metrics(chunk_family, context)
    row_family = _family(script, (row_spec,), None, None)
    (unwrapped,) = script._namespaced_metrics(row_family, context)

    assert isinstance(chunk_metric, script.WinRateWhenInDeckMetric)
    assert isinstance(wrapped, RowwiseMetric)
    assert isinstance(unwrapped, script.OnPlayWinRateDeltaMetric)


def test_a_deck_box_row_metric_without_a_box_is_rejected(script: ModuleType) -> None:
    context = script._CsvMetricContext(
        binder=binder_with_cards(["Owlbear"]),
        header=pd.Index(HEADER),
        source_game=VERSION.game,
        version_metadata=VERSION,
        expansion="KTK",
        format_code="TradDraft",
        output_root=None,
        deck_box=None,
    )
    spec = script.DeckBoxRowMetricSpec(script.DeckWinPredictionMetric)

    with pytest.raises(ValueError, match="needs the family DeckBox"):
        script._build_row_metric(spec, context, Path("unused.parquet"))
