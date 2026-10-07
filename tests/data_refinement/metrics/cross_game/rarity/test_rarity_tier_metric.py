"""Tests for rarity_tier_metric.py's RarityTierMetric."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.cross_game.rarity.rarity_tier_metric import (
    RarityTierMetric,
)
from src.data_refinement.metrics.cross_game.rarity.rarity_translator import (
    FieldRarityTranslator,
    RarityTranslator,
    UnmappedRarityError,
)
from src.data_refinement.metrics.version_metadata import (
    multi_game_versions_from_schema,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.rarity_tier import RarityTier

_TRANSLATORS: dict[GameId, RarityTranslator] = {
    GameId.GWENT: FieldRarityTranslator(
        GameId.GWENT,
        "rarity",
        {"common": RarityTier.TIER_1, "legendary": RarityTier.TIER_4},
    ),
    GameId.SLAY_THE_SPIRE_2: FieldRarityTranslator(
        GameId.SLAY_THE_SPIRE_2,
        "rarity",
        {"Common": RarityTier.TIER_1, "Curse": RarityTier.OTHER},
    ),
}
_COLUMNS = ["nocab_uuid", "source_game", "raw_rarity", "label"]


def _card(game: GameId, name: str, raw_content: dict[str, Any]) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name=name,
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder(*cards: GenericCard) -> CardBinder:
    binder = CardBinder()
    for card in cards:
        binder.create(card)
    return binder


def _scan(binder: CardBinder, tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "out" / "rarity_tier.parquet"
    RarityTierMetric(binder, _TRANSLATORS, path).scan()
    return pd.read_parquet(path)


class TestScan:
    def test_writes_one_row_per_card_with_a_rarity(self, tmp_path: Path) -> None:
        ember = _card(GameId.GWENT, "Ember", {"rarity": "legendary"})
        strike = _card(GameId.SLAY_THE_SPIRE_2, "Strike", {"rarity": "Common"})

        frame = _scan(_binder(ember, strike), tmp_path)

        rows = {
            row.nocab_uuid: (row.source_game, row.raw_rarity, row.label)
            for row in frame.itertuples()
        }
        assert rows == {
            str(ember.nocab_uuid): ("gwent", "legendary", "tier_4"),
            str(strike.nocab_uuid): ("slay_the_spire_2", "Common", "tier_1"),
        }

    def test_a_card_without_a_rarity_and_the_unknown_sentinel_get_no_row(
        self, tmp_path: Path
    ) -> None:
        binder = _binder(_card(GameId.GWENT, "Plain", {"faction": "x"}))
        binder.ensure_unknown_card(GameId.GWENT)

        assert _scan(binder, tmp_path).empty

    def test_a_game_without_a_translator_gets_no_rows(self, tmp_path: Path) -> None:
        binder = _binder(_card(GameId.DOMINION, "Village", {"rarity": "common"}))

        assert _scan(binder, tmp_path).empty

    def test_other_rows_are_written(self, tmp_path: Path) -> None:
        curse = _card(GameId.SLAY_THE_SPIRE_2, "Regret", {"rarity": "Curse"})

        frame = _scan(_binder(curse), tmp_path)

        assert list(frame["label"]) == ["other"]

    def test_an_unmapped_value_raises_before_writing_anything(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "rarity_tier.parquet"
        binder = _binder(
            _card(GameId.GWENT, "Fine", {"rarity": "common"}),
            _card(GameId.GWENT, "Odd", {"rarity": "mythic"}),
        )

        with pytest.raises(UnmappedRarityError, match="mythic"):
            RarityTierMetric(binder, _TRANSLATORS, path).scan()

        assert not path.exists()

    def test_the_file_is_stamped_with_each_games_binder_version(
        self, tmp_path: Path
    ) -> None:
        binder = _binder(_card(GameId.GWENT, "Ember", {"rarity": "common"}))
        path = tmp_path / "rarity_tier.parquet"

        RarityTierMetric(binder, _TRANSLATORS, path).scan()

        stamp = multi_game_versions_from_schema(pq.ParquetFile(path).schema_arrow)
        assert stamp is not None
        assert dict(stamp.card_binder_versions) == {
            game: binder.version_for(game) for game in _TRANSLATORS
        }

    def test_an_empty_corpus_still_writes_the_columns(self, tmp_path: Path) -> None:
        frame = _scan(CardBinder(), tmp_path)

        assert list(frame.columns) == _COLUMNS
        assert frame.empty


def test_label_values_cover_every_tier() -> None:
    assert set(RarityTierMetric.LABEL_VALUES) == {tier.value for tier in RarityTier}


def test_nocab_uuids_round_trip_as_uuids(tmp_path: Path) -> None:
    card = _card(GameId.GWENT, "Ember", {"rarity": "common"})

    frame = _scan(_binder(card), tmp_path)

    assert UUID(frame["nocab_uuid"].iloc[0]) == card.nocab_uuid
