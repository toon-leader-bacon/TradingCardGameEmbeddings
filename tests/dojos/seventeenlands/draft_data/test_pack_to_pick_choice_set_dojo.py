from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import PackToPickChoiceSetDataConstructor
from src.dojos.generic.multi_card_option_selection.dojo import (
    MultiCardOptionSelectionDojo,
)
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.seventeenlands.draft_data.pack_to_pick_choice_set_dojo import (
    PackToPickChoiceSetDojo,
)


def _write_source(path: Path) -> None:
    df = pd.DataFrame(
        {
            "pack_option_uuids": [["00000000-0000-0000-0000-000000000000"]] * 10,
            "pick_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
        }
    )
    df.to_parquet(path, index=False)


class TestPackToPickChoiceSetDojo:
    def test_wires_data_constructor_and_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = PackToPickChoiceSetDojo(
            card_binder,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardOptionSelectionDojo)
        assert isinstance(dojo.data_constructor, PackToPickChoiceSetDataConstructor)
        assert isinstance(dojo.loss_calculator, PickPredictionCrossEntropyLoss)
