from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.attacker_blocker_combat_outcome_metric import (  # noqa: E501
    AttackerBlockerCombatOutcomeMetric,
)
from src.dojos.generic.data_constructors import (
    AttackerBlockerCombatOutcomeDataConstructor,
)
from src.dojos.generic.multi_group_regression.dojo import MultiGroupRegressionDojo
from src.dojos.seventeenlands.replay_data.attacker_blocker_combat_outcome_dojo import (
    AttackerBlockerCombatOutcomeDojo,
)


def _write_source(path: Path) -> None:
    df = pd.DataFrame(
        {
            "attacker_uuids": [["00000000-0000-0000-0000-000000000000"]] * 10,
            "blocker_uuids": [[]] * 10,
            "net_kill_delta": [0] * 10,
        }
    )
    df.to_parquet(path, index=False)


class TestAttackerBlockerCombatOutcomeDojo:
    def test_wires_data_constructor(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = AttackerBlockerCombatOutcomeDojo(
            card_binder,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiGroupRegressionDojo)
        assert isinstance(
            dojo.data_constructor, AttackerBlockerCombatOutcomeDataConstructor
        )

    def test_defaults_to_metrics_own_output_path(self) -> None:
        card_binder = CardBinder()

        with pytest.raises(FileNotFoundError) as exc_info:
            AttackerBlockerCombatOutcomeDojo(card_binder, card_embedding_size=4)

        assert str(AttackerBlockerCombatOutcomeMetric.DEFAULT_OUTPUT_PATH) in str(
            exc_info.value
        )
