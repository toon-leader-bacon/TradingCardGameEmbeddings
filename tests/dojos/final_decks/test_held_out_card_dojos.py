from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.dojos.final_decks import held_out_card_dojos
from src.dojos.final_decks.held_out_card_dojos import GwentHeldOutCardDojo
from src.dojos.generic.data_constructors import HeldOutDeckCardDataConstructor
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.generic.paired_metric_dojos import HeldOutDeckCardMetricDojo
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")

_WRAPPERS = [
    cls
    for cls in vars(held_out_card_dojos).values()
    if isinstance(cls, type)
    and issubclass(cls, HeldOutDeckCardMetricDojo)
    and cls is not HeldOutDeckCardMetricDojo
]


def _write_source(path: Path, metadata: MetricVersionMetadata) -> None:
    df = pd.DataFrame(
        {
            "deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
            "target_card_uuid": ["00000000-0000-0000-0000-000000000001"] * 10,
            "candidate_uuids": [["00000000-0000-0000-0000-000000000001"]] * 10,
        }
    )
    write_dataframe_with_version_metadata(df, path, metadata)


class TestHeldOutCardDojos:
    def test_wires_the_constructor_and_passes_the_box_to_the_version_check(
        self, tmp_path: Path
    ) -> None:
        # A strict check on a requires_deck_box metric raises without the box
        card_binder = CardBinder()
        version = card_binder.version_for(GameId.GWENT)
        box_path = tmp_path / "gwent.db"
        DeckBox().save(box_path, GameId.GWENT, version)
        source = tmp_path / "held_out_card_gwent.parquet"
        _write_source(
            source, MetricVersionMetadata(GameId.GWENT, version, requires_deck_box=True)
        )
        deck_box = DeckBox.load([box_path])

        dojo = GwentHeldOutCardDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiGroupOptionSelectionDojo)
        assert isinstance(dojo.data_constructor, HeldOutDeckCardDataConstructor)
        assert dojo.data_constructor._deck_box is deck_box

    def test_there_is_one_wrapper_per_published_box(self) -> None:
        assert len(_WRAPPERS) == 6

    @pytest.mark.parametrize("dojo_class", _WRAPPERS)
    def test_each_wrapper_reads_its_metrics_output(self, dojo_class: type) -> None:
        path = dojo_class.METRIC.DEFAULT_OUTPUT_PATH
        assert path.parent == Path("data/metrics/final_decks")
        assert path.stem == f"held_out_card_{dojo_class.METRIC.SOURCE_GAME.value}"
