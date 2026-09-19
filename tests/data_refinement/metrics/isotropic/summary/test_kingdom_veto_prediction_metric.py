from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.kingdom_veto_prediction_metric import (
    KingdomVetoPredictionMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def test_writes_row_for_vetoed_game(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Village", "Moat"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomVetoPredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch", "Village"],
            [player_entry("a", 1, {"Copper": 7})],
            vetoed=["Moat"],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    assert df.iloc[0]["vetoed_card_uuids"].tolist() == [str(card_uuid(binder, "Moat"))]


def test_no_vetoes_writes_nothing(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomVetoPredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(summary_row(["Witch"], [player_entry("a", 1, {"Copper": 7})]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_generator_constrained_kingdom_writes_nothing(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = KingdomVetoPredictionMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    row = summary_row(
        ["Witch"], [player_entry("a", 1, {"Copper": 7})], vetoed=["Witch"]
    )
    row["board"]["required"] = True
    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_default_output_path() -> None:
    assert KingdomVetoPredictionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/kingdom_veto_prediction.parquet"
    )
