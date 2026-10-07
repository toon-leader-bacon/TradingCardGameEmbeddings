"""Tests for per_game_mod.py's PerGameMod."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from src.dojos.mods.card_field_mods import RandomKeyMaskMod
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.per_game_mod import PerGameMod
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(game: GameId, raw_content: dict[str, Any]) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name="Card",
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id="x",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _mask(key: str) -> ModPipeline:
    return ModPipeline([MaskTargetKeyMod(key=key, train_only=False)])


class TestPerGameMod:
    def test_applies_the_pipeline_of_the_cards_game(self) -> None:
        mod = PerGameMod(
            {GameId.GWENT: _mask("rarity"), GameId.MTG: _mask("faction")},
            train_only=False,
        )
        card = _card(GameId.GWENT, {"rarity": "epic", "faction": "x"})

        masked, label = mod.apply_single((card, "tier_3"))

        assert isinstance(masked, GenericCard)
        assert masked.raw_content == {"rarity": "[MASK]", "faction": "x"}
        assert label == "tier_3"

    def test_a_game_without_a_pipeline_passes_through_unchanged(self) -> None:
        mod = PerGameMod({GameId.GWENT: _mask("rarity")}, train_only=False)
        datum = (_card(GameId.MTG, {"rarity": "rare"}), "tier_3")

        assert mod.apply_single(datum) is datum

    def test_never_mutates_its_input(self) -> None:
        mod = PerGameMod({GameId.GWENT: _mask("rarity")}, train_only=False)
        card = _card(GameId.GWENT, {"rarity": "epic"})

        mod.apply_single((card, "tier_3"))

        assert card.raw_content == {"rarity": "epic"}

    def test_a_non_single_card_datum_raises(self) -> None:
        mod = PerGameMod({}, train_only=False)

        with pytest.raises(AssertionError, match="single-card"):
            mod.apply_single(([], "x"))  # type: ignore[arg-type]

    def test_inner_tallies_are_reported_under_the_game(self) -> None:
        inner = RandomKeyMaskMod(probability=1.0, rng_seed=0)
        mod = PerGameMod({GameId.GWENT: ModPipeline([inner])}, train_only=True)

        mod.apply_single((_card(GameId.GWENT, {"a": 1}), "x"))

        tallies = ModPipeline([mod]).mod_tallies()
        assert list(tallies) == ["0:PerGameMod/gwent/0:RandomKeyMaskMod"]
        assert tallies["0:PerGameMod/gwent/0:RandomKeyMaskMod"].cards_seen == 1

    def test_train_only_decides_which_splits_it_runs_on(self) -> None:
        card = _card(GameId.GWENT, {"rarity": "epic"})
        pipelines = {GameId.GWENT: _mask("rarity")}
        train_only = ModPipeline([PerGameMod(pipelines, train_only=True)])
        every_split = ModPipeline([PerGameMod(pipelines, train_only=False)])

        eval_datum = train_only.apply([(card, "x")], is_training=False)
        masked_eval = every_split.apply([(card, "x")], is_training=False)

        assert eval_datum[0][0] is card
        assert masked_eval[0][0] is not card
