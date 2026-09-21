from pathlib import Path

import pandas as pd

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.deck_card_set_copy_count_metric import (
    DeckCardSetCopyCountMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def test_writes_count_per_distinct_card_per_eligible_player(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Copper"], tmp_path)
    deck_box = DeckBox()
    metric = DeckCardSetCopyCountMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(["Witch"], [player_entry("a", 1, {"Witch": 2, "Copper": 7})])
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("card_uuid")
    assert df.loc[str(card_uuid(binder, "Witch")), "count"] == 2
    assert df.loc[str(card_uuid(binder, "Copper")), "count"] == 7


def test_default_output_path() -> None:
    assert DeckCardSetCopyCountMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/deck_card_set_copy_count.parquet"
    )
