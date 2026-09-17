from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.contrastive.pair_constructor import SingleCardPairConstructor
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(binder: CardBinder, name: str) -> GenericCard:
    card = GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    binder.create(card)
    return card


def _deck(*card_uuids: object, name: str = "deck") -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        card_nocab_uuids=list(card_uuids),  # type: ignore[arg-type]
        provenance=None,
    )


class TestInit:
    def test_raises_on_non_positive_items_per_deck(self) -> None:
        with pytest.raises(ValueError):
            SingleCardPairConstructor(items_per_deck=0)


class TestBuild:
    def test_samples_items_per_deck_from_each_deck(self) -> None:
        binder = CardBinder()
        cards = [_card(binder, f"card{i}") for i in range(10)]
        deck_a = _deck(*[c.nocab_uuid for c in cards[:6]], name="a")
        deck_b = _deck(*[c.nocab_uuid for c in cards[4:10]], name="b")
        constructor = SingleCardPairConstructor(items_per_deck=3, rng_seed=1)

        batch = constructor.build([deck_a, deck_b], binder)

        assert len(batch.items) == 6
        assert len(batch.identities) == 6
        assert batch.positive_cliques == [[0, 1, 2], [3, 4, 5]]

    def test_skips_a_deck_with_too_few_known_cards(self) -> None:
        binder = CardBinder()
        cards = [_card(binder, f"card{i}") for i in range(10)]
        small_deck = _deck(*[c.nocab_uuid for c in cards[:2]], name="small")
        big_deck = _deck(*[c.nocab_uuid for c in cards[2:10]], name="big")
        constructor = SingleCardPairConstructor(items_per_deck=3, rng_seed=1)

        batch = constructor.build([small_deck, big_deck], binder)

        assert len(batch.items) == 3
        assert batch.positive_cliques == [[0, 1, 2]]

    def test_excludes_an_unresolvable_card_uuid(self) -> None:
        binder = CardBinder()
        cards = [_card(binder, f"card{i}") for i in range(3)]
        unknown_uuid = uuid4()
        deck = _deck(*([c.nocab_uuid for c in cards] + [unknown_uuid]), name="deck")
        constructor = SingleCardPairConstructor(items_per_deck=3, rng_seed=1)

        batch = constructor.build([deck], binder)

        assert len(batch.items) == 3
        sampled_uuids = {identity[0] for identity in batch.identities}
        assert unknown_uuid not in sampled_uuids

    def test_returns_an_empty_batch_when_every_deck_is_skipped(self) -> None:
        binder = CardBinder()
        cards = [_card(binder, f"card{i}") for i in range(1)]
        tiny_deck = _deck(*[c.nocab_uuid for c in cards], name="tiny")
        constructor = SingleCardPairConstructor(items_per_deck=5, rng_seed=1)

        batch = constructor.build([tiny_deck], binder)

        assert batch.items == []
        assert batch.identities == []
        assert batch.positive_cliques == []

    def test_is_deterministic_given_the_same_seed(self) -> None:
        binder = CardBinder()
        cards = [_card(binder, f"card{i}") for i in range(10)]
        deck = _deck(*[c.nocab_uuid for c in cards], name="deck")

        batch_a = SingleCardPairConstructor(items_per_deck=4, rng_seed=99).build(
            [deck], binder
        )
        batch_b = SingleCardPairConstructor(items_per_deck=4, rng_seed=99).build(
            [deck], binder
        )

        assert batch_a.identities == batch_b.identities
