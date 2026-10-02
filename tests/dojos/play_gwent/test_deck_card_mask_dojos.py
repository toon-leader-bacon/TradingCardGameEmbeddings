import pytest
from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.play_gwent.deck_card_mask_dojos import LeaderMaskedFromDeckDojo
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")


def _write_source(path: Path) -> None:
    df = pd.DataFrame(
        {
            "deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
            "target_card_uuid": ["00000000-0000-0000-0000-000000000001"] * 10,
            "label": ["Geralt"] * 10,
        }
    )
    df.to_parquet(path, index=False)


def _source_path(tmp_path: Path) -> Path:
    result = tmp_path / "unversioned.parquet"
    _write_source(result)
    return result


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

    def test_label_values_is_leader_metrics_own_label_values(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = LeaderMaskedFromDeckDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert dojo.label_values == list(LeaderMaskedFromDeckMetric.LABEL_VALUES)

    def test_hands_its_deck_box_to_the_metric_version_check(
        self, tmp_path: Path
    ) -> None:
        # The metric needs a deck box to verify; a strict check without it raises
        card_binder = CardBinder()
        version = card_binder.version_for(GameId.GWENT)
        deck_box_path = tmp_path / "gwent.db"
        DeckBox().save(deck_box_path, GameId.GWENT, version)
        source = tmp_path / "source.parquet"
        write_dataframe_with_version_metadata(
            pd.read_parquet(_source_path(tmp_path)),
            source,
            MetricVersionMetadata(GameId.GWENT, version, requires_deck_box=True),
        )

        dojo = LeaderMaskedFromDeckDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            DeckBox.load([deck_box_path]),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardFixedClassificationDojo)
