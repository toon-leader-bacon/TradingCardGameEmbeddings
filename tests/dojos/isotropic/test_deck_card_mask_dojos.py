import pytest
from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.deck_card_mask_metric import (
    WinningDeckMaskedCardMetric,
)
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.isotropic.deck_card_mask_dojos import WinningDeckMaskedCardDojo
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")


def _write_source(path: Path) -> None:
    df = pd.DataFrame(
        {
            "deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
            "target_card_uuid": ["00000000-0000-0000-0000-000000000001"] * 10,
            "label": ["Chapel"] * 10,
        }
    )
    df.to_parquet(path, index=False)


class TestWinningDeckMaskedCardDojo:
    def test_wires_deck_card_mask_constructor_and_metric_label_values(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "isotropic_masked_card_source.parquet"
        _write_source(source)
        deck_box = DeckBox()

        dojo = WinningDeckMaskedCardDojo(
            CardBinder(),
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, MultiCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, DeckCardMaskDataConstructor)
        assert dojo.data_constructor._deck_box is deck_box
        assert dojo.label_values == list(WinningDeckMaskedCardMetric.LABEL_VALUES)
