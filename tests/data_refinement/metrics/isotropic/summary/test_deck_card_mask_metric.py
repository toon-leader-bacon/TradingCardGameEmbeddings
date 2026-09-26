from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.deck_card_mask_metric import (
    WinningDeckMaskedCardMetric,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    player_entry,
    summary_row,
)


def test_masks_a_non_basic_card_from_the_winning_deck(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch", "Copper", "Estate"], tmp_path)
    deck_box = DeckBox()
    metric = WinningDeckMaskedCardMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(
            ["Witch"],
            [player_entry("winner", 1, {"Witch": 2, "Copper": 5, "Estate": 3})],
        )
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    assert df.iloc[0]["label"] == "Witch"


def test_basics_only_deck_has_no_valid_masking_target(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Copper", "Estate"], tmp_path)
    deck_box = DeckBox()
    metric = WinningDeckMaskedCardMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        summary_row(["Witch"], [player_entry("winner", 1, {"Copper": 7, "Estate": 3})])
    )
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 0


def test_no_resolvable_winner_raises(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_box = DeckBox()
    metric = WinningDeckMaskedCardMetric(
        binder, deck_box, output_path=tmp_path / "out.parquet"
    )

    # Per this metric's own contract (a deck is required for every row
    # it's fed): the shared scanner's per-row isolation is what's
    # expected to absorb this in real use, not this metric itself.
    with pytest.raises(ValueError):
        metric.accumulate(
            summary_row(["Witch"], [player_entry("solo", 1, resigned=True)])
        )
    metric.finalize()


def test_default_output_path() -> None:
    assert WinningDeckMaskedCardMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/isotropic/winning_deck_masked_card.parquet"
    )


@pytest.mark.skipif(
    not Path("data/final/cards/dominion.jsonl").exists(),
    reason="needs the real, already-ingested Dominion CardBinder",
)
def test_label_values_excludes_basics_and_includes_real_kingdom_cards() -> None:
    # LABEL_VALUES is populated once, at class-definition time, from
    # the real (already-ingested) data/final/cards/dominion.jsonl - not
    # a per-test fixture, so this asserts against the live project data.
    assert "Witch" in WinningDeckMaskedCardMetric.LABEL_VALUES
    assert "Copper" not in WinningDeckMaskedCardMetric.LABEL_VALUES
    assert "Estate" not in WinningDeckMaskedCardMetric.LABEL_VALUES
