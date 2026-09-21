from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.metrics.isotropic.summary.turn_count_association_metric import (
    TurnCountAssociationMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def test_turn_count_delta_relative_to_corpus_baseline(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village"], tmp_path)
    metric = TurnCountAssociationMetric(binder, output_path=tmp_path / "out.parquet")

    # Baseline over both games: (30 + 10) / 2 = 20.
    metric.accumulate(
        summary_row(["Witch"], [player_entry("a", 1, {"Copper": 7}, turns=30)])
    )
    metric.accumulate(
        summary_row(["Village"], [player_entry("b", 1, {"Copper": 7}, turns=10)])
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    assert df.loc[str(card_uuid(binder, "Witch")), "turn_count_delta"] == pytest.approx(
        10.0
    )
    assert df.loc[
        str(card_uuid(binder, "Village")), "turn_count_delta"
    ] == pytest.approx(-10.0)


def test_skips_generator_constrained_kingdoms(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = TurnCountAssociationMetric(binder, output_path=tmp_path / "out.parquet")

    row = summary_row(["Witch"], [player_entry("a", 1, {"Copper": 7}, turns=30)])
    row["board"]["required"] = True
    metric.accumulate(row)

    with pytest.raises(ZeroDivisionError):
        metric.finalize()


def test_no_resolvable_winner_contributes_nothing(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    metric = TurnCountAssociationMetric(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(summary_row(["Witch"], [player_entry("solo", 1, resigned=True)]))

    with pytest.raises(ZeroDivisionError):
        metric.finalize()


def test_default_output_path() -> None:
    assert TurnCountAssociationMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/turn_count_association.parquet"
    )
