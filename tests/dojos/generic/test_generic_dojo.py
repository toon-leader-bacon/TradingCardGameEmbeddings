from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import UUID, uuid4

import pandas as pd
import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.batch import Batch
from src.dojos.dojo import BatchBudget, Dojo
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import TrainingDatum

_BUDGET = BatchBudget(max_cost=8, cost_of=lambda card: 1)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(DataSource.SCRYFALL, name, datetime.now(timezone.utc)),
    )


class _CardPerRowConstructor:
    """Row i -> the card whose uuid the row names, label = i; a card the
    lookup can't see skips the row (the real constructors' convention)."""

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        data: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            card = lookup.get_by_uuid(UUID(row["nocab_uuid"]))
            if card is not None:
                data.append((card, float(row["label"])))
        return data


def _dojo(
    tmp_path: Path, cards: list[GenericCard], holdout: HoldoutSpec, rows: int = 100
) -> SingleCardRegressionDojo:
    binder = CardBinder()
    for card in cards:
        binder.create(card)
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "nocab_uuid": [str(cards[i % len(cards)].nocab_uuid) for i in range(rows)],
            "label": [float(i) for i in range(rows)],
        }
    ).to_parquet(source, index=False)
    return SingleCardRegressionDojo(
        path_to_training_data=source,
        data_constructor=_CardPerRowConstructor(),
        card_lookup=binder,
        holdout=holdout,
        card_embedding_size=4,
        rng_seed=0,
    )


def test_satisfies_the_dojo_protocol(tmp_path: Path) -> None:
    dojo: Dojo = _dojo(tmp_path, [_card("a")], HoldoutSpec.no_holdout())

    assert dojo.name == "source"
    assert dojo.example_count(Split.TRAIN) == 80


def test_every_batch_respects_the_budget(tmp_path: Path) -> None:
    dojo = _dojo(tmp_path, [_card("a")], HoldoutSpec.no_holdout())

    batches = list(dojo.batches(Split.TRAIN, _BUDGET))

    assert all(len(b) <= 8 for b in batches)
    assert sum(len(b) for b in batches) == 80


def test_max_examples_truncates_deterministically(tmp_path: Path) -> None:
    dojo = _dojo(tmp_path, [_card("a")], HoldoutSpec.no_holdout())

    first = list(dojo.batches(Split.TEST, _BUDGET, max_examples=5))
    second = list(dojo.batches(Split.TEST, _BUDGET, max_examples=5))

    assert sum(len(b) for b in first) == 5
    assert [b.labels for b in first] == [b.labels for b in second]


def test_batches_can_be_replayed(tmp_path: Path) -> None:
    dojo = _dojo(tmp_path, [_card("a")], HoldoutSpec.no_holdout())

    assert sum(len(b) for b in dojo.batches(Split.TRAIN, _BUDGET)) == sum(
        len(b) for b in dojo.batches(Split.TRAIN, _BUDGET)
    )


def test_held_out_cards_are_invisible_to_the_train_split_only(tmp_path: Path) -> None:
    holdout = HoldoutSpec(seed=1, tier_ratios=(1, 1, 1), held_out_games=frozenset())
    cards = [_card(f"c{i}") for i in range(60)]
    by_tier = {
        tier: {
            c.nocab_uuid
            for c in cards
            if holdout.tier_of(c.nocab_uuid, c.source_game) == tier
        }
        for tier in CardTier
    }
    assert all(by_tier.values()), "fixture needs cards in every tier"
    dojo = _dojo(tmp_path, cards, holdout, rows=600)

    def seen(split: Split) -> set:
        return {
            card.nocab_uuid
            for batch in dojo.batches(split, _BUDGET)
            for card in batch.inputs
        }

    assert seen(Split.TRAIN) <= by_tier[CardTier.TRAIN]
    assert not seen(Split.TEST) & by_tier[CardTier.VALIDATION]
    assert seen(Split.VALIDATION) & by_tier[CardTier.VALIDATION]


def test_reset_head_restores_initial_parameters(tmp_path: Path) -> None:
    dojo = _dojo(tmp_path, [_card("a")], HoldoutSpec.no_holdout())
    before = [p.detach().clone() for p in dojo.trainable_parameters()]
    with torch.no_grad():
        for p in dojo.trainable_parameters():
            p.add_(1.0)

    dojo.reset_head()

    after = list(dojo.trainable_parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_compute_loss_rejects_a_foreign_batch(tmp_path: Path) -> None:
    dojo = _dojo(tmp_path, [_card("a")], HoldoutSpec.no_holdout())
    card = _card("x")

    with pytest.raises(TypeError):
        dojo.compute_loss([torch.randn(4)], object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dojo.compute_loss([torch.randn(4)], Batch([card, card], [1.0, 2.0]))
