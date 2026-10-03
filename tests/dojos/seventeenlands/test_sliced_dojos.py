"""Tests for sliced_dojos.py: seventeenlands_training_path and the
17lands intermediate dojo bases, run in an isolated working directory
(the slice files and split files land under tmp_path)."""

from pathlib import Path

import numpy as np
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.count_table import write_count_table
from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.data_refinement.metrics.seventeenlands.partition import (
    SeventeenLandsPartition,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_retrieval.seventeenlands.refs import DataType, Expansion, format_code
from src.dojos.seventeenlands.game_data.tutor_target_rate_dojo import (
    TutorTargetRateDojo,
)
from src.dojos.seventeenlands.sliced_dojos import seventeenlands_training_path
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder data, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")

_VERSION = MetricVersionMetadata(game=GameId.MTG, card_binder_version="v1")
_SEALED = SeventeenLandsSlice(formats=frozenset({format_code.Sealed}))


@pytest.fixture
def partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two TutorTargetRateMetric partitions under ./data/metrics, with
    tmp_path as the working directory."""
    monkeypatch.chdir(tmp_path)
    for fmt in (format_code.Sealed, format_code.TradDraft):
        partition = SeventeenLandsPartition(
            DataType.GAME, TutorTargetRateMetric.OUTPUT_STEM, Expansion.KTK, fmt
        )
        write_count_table(
            partition.path(),
            TutorTargetRateMetric,
            {"nocab_uuid": ["00000000-0000-0000-0000-000000000000"]},
            {"in_deck": np.array([4.0]), "tutored": np.array([1.0])},
            _VERSION,
        )


def test_an_explicit_path_wins_and_builds_nothing(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.parquet"

    result = seventeenlands_training_path(TutorTargetRateMetric, _SEALED, explicit)

    assert result == explicit
    assert not (tmp_path / "data").exists()


@pytest.mark.usefixtures("partitions")
def test_without_a_path_the_slice_file_is_built() -> None:
    result = seventeenlands_training_path(TutorTargetRateMetric, _SEALED, None)

    assert result == Path(
        "data/metrics/seventeenlands/game_data/slices/"
        "tutor_target_rate.formats-Sealed.parquet"
    )
    assert result.exists()


@pytest.mark.usefixtures("partitions")
def test_a_dojo_defaults_to_the_all_slice_and_is_named_by_it() -> None:
    dojo = TutorTargetRateDojo(
        CardBinder(), HoldoutSpec.no_holdout(), 4, strict_version_check=False
    )

    assert dojo.name == "tutor_target_rate.all"


@pytest.mark.usefixtures("partitions")
def test_two_slices_of_one_metric_get_different_names() -> None:
    every = TutorTargetRateDojo(
        CardBinder(), HoldoutSpec.no_holdout(), 4, strict_version_check=False
    )
    sealed = TutorTargetRateDojo(
        CardBinder(),
        HoldoutSpec.no_holdout(),
        4,
        data_slice=_SEALED,
        strict_version_check=False,
    )

    assert every.name != sealed.name
