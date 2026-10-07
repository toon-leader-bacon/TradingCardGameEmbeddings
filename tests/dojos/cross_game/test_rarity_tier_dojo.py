"""Tests for rarity_tier_dojo.py and rarity_tier_data_constructor.py."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.cross_game.rarity.rarity_tier_metric import (
    RarityTierMetric,
)
from src.data_refinement.metrics.cross_game.rarity.translator_tables import (
    RARITY_TRANSLATORS,
)
from src.dojos.cross_game.rarity_tier_data_constructor import (
    RarityTierDataConstructor,
)
from src.dojos.cross_game.rarity_tier_dojo import RarityTierDojo, _masking_pipelines
from src.dojos.dojo import BatchBudget
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

_BUDGET = BatchBudget(max_cost=8, cost_of=lambda card: 1)
_GWENT_RARITIES = ["common", "rare", "epic", "legendary"]


def _card(game: GameId, name: str, rarity: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name=name,
        raw_content={"rarity": rarity, "name": name},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder() -> CardBinder:
    """60 Gwent cards over four rarities, 10 StS2 Commons and 10 StS2 Curses."""
    binder = CardBinder()
    for index in range(60):
        rarity = _GWENT_RARITIES[index % 4]
        binder.create(_card(GameId.GWENT, f"g{index}", rarity))
    for index in range(10):
        binder.create(_card(GameId.SLAY_THE_SPIRE_2, f"s{index}", "Common"))
        binder.create(_card(GameId.SLAY_THE_SPIRE_2, f"c{index}", "Curse"))
    return binder


@pytest.fixture
def binder() -> CardBinder:
    return _binder()


@pytest.fixture
def metric_path(binder: CardBinder, tmp_path: Path) -> Path:
    path = tmp_path / "rarity_tier.parquet"
    RarityTierMetric(binder, output_path=path).scan()
    return path


def _dojo(binder: CardBinder, metric_path: Path, **kwargs: Any) -> RarityTierDojo:
    return RarityTierDojo(
        binder,
        HoldoutSpec.no_holdout(),
        card_embedding_size=4,
        path_to_training_data=metric_path,
        rng_seed=0,
        **kwargs,
    )


def _examples(dojo: RarityTierDojo, split: Split) -> Iterator[tuple[GenericCard, Any]]:
    for batch in dojo.batches(split, _BUDGET):
        yield from zip(batch.inputs, batch.labels)  # type: ignore[arg-type]


class TestRarityTierDojo:
    def test_trains_five_classes_tier_1_to_4_and_special(
        self, binder: CardBinder, metric_path: Path
    ) -> None:
        dojo = _dojo(binder, metric_path)

        assert dojo.label_values == [
            "tier_1",
            "tier_2",
            "tier_3",
            "tier_4",
            "special",
        ]

    @pytest.mark.parametrize("split", [Split.TRAIN, Split.TEST, Split.VALIDATION])
    def test_the_rarity_field_is_masked_on_every_split_for_every_game(
        self, binder: CardBinder, metric_path: Path, split: Split
    ) -> None:
        dojo = _dojo(binder, metric_path)

        cards = [card for card, _ in _examples(dojo, split)]

        assert cards
        assert all(card.raw_content["rarity"] == "[MASK]" for card in cards)
        # Other fields are left alone
        assert all(card.raw_content["name"] != "[MASK]" for card in cards)

    def test_train_rows_are_drawn_evenly_across_games(
        self, binder: CardBinder, metric_path: Path
    ) -> None:
        dojo = _dojo(binder, metric_path)

        games = Counter(card.source_game for card, _ in _examples(dojo, Split.TRAIN))

        # About 48 Gwent and 8 StS2 rows are trainable: unbalanced would be ~85% Gwent
        share = games[GameId.SLAY_THE_SPIRE_2] / sum(games.values())
        assert 0.3 < share < 0.7

    def test_other_rows_are_never_trained_on(
        self, binder: CardBinder, metric_path: Path
    ) -> None:
        dojo = _dojo(binder, metric_path)

        for split in (Split.TRAIN, Split.TEST, Split.VALIDATION):
            examples = list(_examples(dojo, split))
            assert all(label in dojo.label_values for _, label in examples)
            assert not any(card.name.startswith("c") for card, _ in examples)

    def test_test_and_validation_are_read_in_file_order(
        self, binder: CardBinder, metric_path: Path
    ) -> None:
        dojo = _dojo(binder, metric_path)

        first = [card.nocab_uuid for card, _ in _examples(dojo, Split.TEST)]
        second = [card.nocab_uuid for card, _ in _examples(dojo, Split.TEST)]

        assert first == second

    def test_the_loss_baseline_is_calibrated_on_the_balanced_draw(
        self, binder: CardBinder, metric_path: Path
    ) -> None:
        dojo = _dojo(binder, metric_path)
        batch = next(dojo.batches(Split.TRAIN, _BUDGET))

        assert 0 < dojo.baseline_loss(batch) < 2.0

    def test_a_stale_binder_version_for_any_game_raises(
        self, binder: CardBinder, metric_path: Path
    ) -> None:
        binder.create(_card(GameId.SLAY_THE_SPIRE_2, "new", "Common"))

        with pytest.raises(ValueError, match="regenerate this metric"):
            _dojo(binder, metric_path)

    def test_the_stale_check_can_be_switched_off(
        self, binder: CardBinder, metric_path: Path
    ) -> None:
        binder.create(_card(GameId.GWENT, "new", "common"))

        assert _dojo(binder, metric_path, strict_version_check=False)


class TestMaskingPipelines:
    def test_one_mask_per_masked_path_of_each_games_translator(self) -> None:
        pipelines = _masking_pipelines(RARITY_TRANSLATORS)

        assert set(pipelines) == set(RARITY_TRANSLATORS)
        for game, pipeline in pipelines.items():
            paths = RARITY_TRANSLATORS[game].masked_paths()
            assert [mod.key for mod in pipeline.mods] == list(paths)  # type: ignore[attr-defined]
            assert all(not mod.train_only for mod in pipeline.mods)


class TestRarityTierDataConstructor:
    def _chunk(self, cards: list[GenericCard], labels: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {"nocab_uuid": [str(card.nocab_uuid) for card in cards], "label": labels}
        )

    def test_drops_other_rows_and_builds_the_rest(self, binder: CardBinder) -> None:
        cards = list(binder.all_cards(GameId.SLAY_THE_SPIRE_2))[:2]
        chunk = self._chunk(cards, ["tier_1", "other"])

        data = RarityTierDataConstructor().build(chunk, binder)

        assert [(card.nocab_uuid, label) for card, label in data] == [
            (cards[0].nocab_uuid, "tier_1")
        ]

    def test_skips_a_card_the_lookup_does_not_hold(self, binder: CardBinder) -> None:
        stranger = _card(GameId.GWENT, "stranger", "common")
        chunk = self._chunk([stranger], ["tier_1"])

        assert RarityTierDataConstructor().build(chunk, binder) == []
