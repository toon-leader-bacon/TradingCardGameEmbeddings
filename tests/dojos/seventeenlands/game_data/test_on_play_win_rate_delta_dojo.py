from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.seventeenlands.game_data.on_play_win_rate_delta_dojo import (
    OnPlayWinRateDeltaDojo,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card() -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="Strike",
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id="strike",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _write_source(path: Path) -> None:
    df = pd.DataFrame({"nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df["on_play_win_rate_delta"] = 0.05
    df["sample_count"] = 10
    df.to_parquet(path, index=False)


class TestOnPlayWinRateDeltaDojo:
    def test_wires_data_constructor_to_label_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = OnPlayWinRateDeltaDojo(
            card_binder,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, SingleCardRegressionDojo)
        assert isinstance(dojo.data_constructor, CardAverageDataConstructor)
        assert dojo.data_constructor._label_column == "on_play_win_rate_delta"

    def test_defaults_to_metrics_own_output_path(self) -> None:
        card_binder = CardBinder()

        with pytest.raises(FileNotFoundError) as exc_info:
            OnPlayWinRateDeltaDojo(card_binder, card_embedding_size=4)

        assert str(OnPlayWinRateDeltaMetric.DEFAULT_OUTPUT_PATH) in str(exc_info.value)

    def test_nan_label_is_skipped_not_trained_on(self) -> None:
        # A card never seen on one side of on_play writes None for its
        # delta - round-tripped through parquet as NaN. Confirms the
        # data_constructors.py NaN guard actually protects this dojo's
        # own wiring, not just the constructor in isolation.
        card_binder = CardBinder()
        card = _card()
        card_binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "on_play_win_rate_delta": [float("nan")],
                "sample_count": [3],
            }
        )

        result = CardAverageDataConstructor(
            card_binder, "on_play_win_rate_delta"
        ).build(chunk)

        assert result == []
