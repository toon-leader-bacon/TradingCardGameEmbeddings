from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.seventeenlands.replay_data.combat_aggression_profile_dojo import (
    CombatAggressionProfileDojo,
)
from src.schema.holdout import HoldoutSpec


def _write_source(path: Path) -> None:
    df = pd.DataFrame({"deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df["combat_aggression_profile"] = 2.5
    df.to_parquet(path, index=False)


class TestCombatAggressionProfileDojo:
    def test_wires_data_constructor_to_label_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = CombatAggressionProfileDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardRegressionDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert dojo.data_constructor._label_column == "combat_aggression_profile"
