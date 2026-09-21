import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.metrics.isotropic.summary.veto_rate_metric import (
    VetoRateMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def test_tallies_veto_rate_per_kingdom_card(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village", "Moat"], tmp_path)
    metric = VetoRateMetric(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(
        summary_row(
            ["Witch", "Village", "Moat"],
            [player_entry("a", 1, {"Copper": 7})],
            vetoed=["Witch"],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    assert df.loc[str(card_uuid(binder, "Witch")), "veto_rate"] == 1.0
    assert df.loc[str(card_uuid(binder, "Village")), "veto_rate"] == 0.0


def test_skips_generator_constrained_kingdoms(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = VetoRateMetric(binder, output_path=tmp_path / "out.parquet")

    row = summary_row(
        ["Witch"], [player_entry("a", 1, {"Copper": 7})], vetoed=["Witch"]
    )
    row["board"]["required"] = True
    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_row_with_no_vetoes_still_tallies_total(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = VetoRateMetric(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(summary_row(["Witch"], [player_entry("a", 1, {"Copper": 7})]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.iloc[0]["veto_rate"] == 0.0
    assert df.iloc[0]["sample_count"] == 1


def test_unresolved_kingdom_card_is_excluded_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = VetoRateMetric(binder, output_path=tmp_path / "out.parquet")

    with caplog.at_level(logging.ERROR):
        metric.accumulate(
            summary_row(["Witch", "Not A Card"], [player_entry("a", 1, {"Copper": 7})])
        )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    assert any("Not A Card" in record.getMessage() for record in caplog.records)


def test_default_output_path() -> None:
    assert VetoRateMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/veto_rate.parquet"
    )
