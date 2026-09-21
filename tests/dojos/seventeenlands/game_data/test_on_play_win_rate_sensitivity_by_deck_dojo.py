from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_dojo import (
    OnPlayWinRateSensitivityByDeckDojo,
)
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec


def _write_source(path: Path) -> None:
    df = pd.DataFrame({"deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df["on_play_win_rate_sensitivity"] = 0.05
    df["sample_count"] = 10
    df.to_parquet(path, index=False)


class TestOnPlayWinRateSensitivityByDeckDojo:
    def test_wires_data_constructor_to_label_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = OnPlayWinRateSensitivityByDeckDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, MultiCardRegressionDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert dojo.data_constructor._label_column == "on_play_win_rate_sensitivity"

    def test_nan_label_is_skipped_not_trained_on(self) -> None:
        # A deck never seen on one side of on_play writes None for its
        # sensitivity - round-tripped through parquet as NaN. The deck
        # itself resolves cleanly, so a non-skip would produce a real
        # (non-empty) result - isolating the NaN label as the cause.
        card_binder = CardBinder()
        deck_box = DeckBox()
        card = GenericCard(
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
        card_binder.create(card)
        deck = GenericDeck(
            nocab_uuid=uuid4(),
            source_game=GameId.MTG,
            name="test deck",
            card_nocab_uuids=[card.nocab_uuid],
        )
        deck_box.create(deck)
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(deck.nocab_uuid)],
                "on_play_win_rate_sensitivity": [float("nan")],
                "sample_count": [3],
            }
        )

        result = DeckLabelDataConstructor(
            deck_box, "on_play_win_rate_sensitivity"
        ).build(chunk, card_binder)

        assert result == []
