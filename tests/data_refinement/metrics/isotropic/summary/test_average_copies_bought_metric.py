from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.isotropic.summary.average_copies_bought_metric import (
    AverageCopiesBoughtMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def test_averages_copies_across_games(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = AverageCopiesBoughtMetric(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(
        summary_row(["Witch"], [player_entry("a", 1, {"Witch": 4, "Copper": 3})])
    )
    metric.accumulate(
        summary_row(["Witch"], [player_entry("b", 1, {"Witch": 2, "Copper": 3})])
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    row = df.loc[str(card_uuid(binder, "Witch"))]
    assert row["average_copies_bought"] == 3.0
    assert row["sample_count"] == 2


def test_default_output_path() -> None:
    assert AverageCopiesBoughtMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/average_copies_bought.parquet"
    )
