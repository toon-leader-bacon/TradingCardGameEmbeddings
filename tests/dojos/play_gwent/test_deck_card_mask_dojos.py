from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.play_gwent.deck_card_mask_dojos import LeaderMaskedFromDeckDojo


def _write_source(path: Path) -> None:
    df = pd.DataFrame(
        {
            "deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
            "target_card_uuid": ["00000000-0000-0000-0000-000000000001"] * 10,
            "label": ["Geralt"] * 10,
        }
    )
    df.to_parquet(path, index=False)


class TestLeaderMaskedFromDeckDojo:
    def test_wires_data_constructor_to_card_binder_and_deck_box(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = LeaderMaskedFromDeckDojo(
            card_binder,
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, DeckCardMaskDataConstructor)
        assert dojo.data_constructor._card_binder is card_binder
        assert dojo.data_constructor._deck_box is deck_box

    def test_label_values_is_leader_metrics_own_label_values(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = LeaderMaskedFromDeckDojo(
            card_binder,
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert dojo.label_values == list(LeaderMaskedFromDeckMetric.LABEL_VALUES)

    def test_defaults_to_leader_metrics_own_output_path(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()

        with pytest.raises(FileNotFoundError) as exc_info:
            LeaderMaskedFromDeckDojo(card_binder, deck_box, card_embedding_size=4)

        assert str(LeaderMaskedFromDeckMetric.DEFAULT_OUTPUT_PATH) in str(
            exc_info.value
        )
