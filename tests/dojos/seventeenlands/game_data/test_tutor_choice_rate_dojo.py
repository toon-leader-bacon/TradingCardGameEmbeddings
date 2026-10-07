import pytest
from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.seventeenlands.game_data.tutor_choice_rate_dojo import (
    TutorChoiceRateDojo,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")


def _write_source(path: Path) -> None:
    df = pd.DataFrame({"nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df["tutor_choice_rate"] = 0.5
    df["sample_count"] = 2
    df.to_parquet(path, index=False)


class TestTutorChoiceRateDojo:
    def test_wires_data_constructor_to_the_rate_column(self, tmp_path: Path) -> None:
        source = tmp_path / "tutor_choice_rate_source.parquet"
        _write_source(source)

        dojo = TutorChoiceRateDojo(
            CardBinder(),
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, SeventeenLandsCardLabelDojo)
        assert isinstance(dojo.data_constructor, CardAverageDataConstructor)
        assert dojo.data_constructor._label_column == "tutor_choice_rate"
        assert dojo.data_constructor._uuid_column == "nocab_uuid"
