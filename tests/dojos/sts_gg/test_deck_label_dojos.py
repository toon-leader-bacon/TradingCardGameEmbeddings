from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
    ElitesKilledMetric,
    FloorsClearedMetric,
    RelicCountMetric,
    TotalCardsPickedMetric,
    TotalCardsSkippedMetric,
    TotalCombatsMetric,
    TotalDamageTakenMetric,
    TotalTurnsMetric,
    WinMetric,
)
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.sts_gg.deck_label_dojos import (
    CharacterDojo,
    DeckElitesKilledDojo,
    DeckFloorsClearedDojo,
    DeckRelicCountDojo,
    DeckTotalCardsPickedDojo,
    DeckTotalCardsSkippedDojo,
    DeckTotalCombatsDojo,
    DeckTotalDamageTakenDojo,
    DeckTotalTurnsDojo,
    WinDojo,
)

_CASES = [
    (DeckRelicCountDojo, RelicCountMetric),
    (DeckTotalDamageTakenDojo, TotalDamageTakenMetric),
    (DeckTotalCardsPickedDojo, TotalCardsPickedMetric),
    (DeckTotalCardsSkippedDojo, TotalCardsSkippedMetric),
    (DeckTotalTurnsDojo, TotalTurnsMetric),
    (DeckElitesKilledDojo, ElitesKilledMetric),
    (DeckFloorsClearedDojo, FloorsClearedMetric),
    (DeckTotalCombatsDojo, TotalCombatsMetric),
]


def _write_source(path: Path, label_column: str) -> None:
    df = pd.DataFrame({"deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df[label_column] = 1
    df.to_parquet(path, index=False)


class TestDeckLabelDojoWrappers:
    @pytest.mark.parametrize("dojo_cls,metric_cls", _CASES)
    def test_wires_data_constructor_to_metrics_label_column(
        self, dojo_cls, metric_cls, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, metric_cls.LABEL_COLUMN)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = dojo_cls(
            card_binder,
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardRegressionDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert dojo.data_constructor._label_column == metric_cls.LABEL_COLUMN

    @pytest.mark.parametrize("dojo_cls,metric_cls", _CASES)
    def test_defaults_to_metrics_own_output_path(self, dojo_cls, metric_cls) -> None:
        # No DEFAULT_OUTPUT_PATH file exists on disk in a test environment,
        # so confirm the wrapper *attempts* to use it (FileManagerParquet
        # raises FileNotFoundError against that exact path) rather than
        # actually constructing a dojo against it.
        card_binder = CardBinder()
        deck_box = DeckBox()

        with pytest.raises(FileNotFoundError) as exc_info:
            dojo_cls(card_binder, deck_box, card_embedding_size=4)

        assert str(metric_cls.DEFAULT_OUTPUT_PATH) in str(exc_info.value)


class TestWinDojo:
    def test_wires_data_constructor_to_win_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, WinMetric.LABEL_COLUMN)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = WinDojo(
            card_binder,
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardBinaryClassificationDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert dojo.data_constructor._label_column == WinMetric.LABEL_COLUMN

    def test_defaults_to_win_metrics_own_output_path(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()

        with pytest.raises(FileNotFoundError) as exc_info:
            WinDojo(card_binder, deck_box, card_embedding_size=4)

        assert str(WinMetric.DEFAULT_OUTPUT_PATH) in str(exc_info.value)


class TestCharacterDojo:
    def test_wires_data_constructor_to_character_column_with_str_caster(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, CharacterPredictionMetric.LABEL_COLUMN)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = CharacterDojo(
            card_binder,
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert dojo.data_constructor._label_column == (
            CharacterPredictionMetric.LABEL_COLUMN
        )
        assert dojo.data_constructor._label_caster is str

    def test_label_values_is_characters_own_label_values(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, CharacterPredictionMetric.LABEL_COLUMN)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = CharacterDojo(
            card_binder,
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert dojo.label_values == list(CharacterPredictionMetric.LABEL_VALUES)

    def test_defaults_to_characters_own_output_path(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()

        with pytest.raises(FileNotFoundError) as exc_info:
            CharacterDojo(card_binder, deck_box, card_embedding_size=4)

        assert str(CharacterPredictionMetric.DEFAULT_OUTPUT_PATH) in str(exc_info.value)
