from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.isotropic.summary.copies_bought_distribution_metric import (
    CopiesBoughtDistributionMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def test_writes_one_row_per_distinct_card_per_eligible_player(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Copper"], tmp_path)
    metric = CopiesBoughtDistributionMetric(
        binder, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch"],
            [
                player_entry("winner", 1, {"Witch": 3, "Copper": 2}),
                player_entry("resigned", 2, resigned=True),
            ],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 2
    witch_row = df[df["nocab_uuid"] == str(card_uuid(binder, "Witch"))].iloc[0]
    assert witch_row["copies"] == 3


def test_default_output_path() -> None:
    assert CopiesBoughtDistributionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/copies_bought_distribution.parquet"
    )
