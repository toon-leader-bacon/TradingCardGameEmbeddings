from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str, source_id: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _deck(cards: list[GenericCard]) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name="test deck",
        card_nocab_uuids=[card.nocab_uuid for card in cards],
    )


class TestDeckLabelDataConstructorBuild:
    def test_builds_one_datum_per_resolvable_row(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        card1, card2 = _card("Strike", "strike"), _card("Defend", "defend")
        card_binder.create(card1)
        card_binder.create(card2)
        deck = _deck([card1, card2])
        deck_box.create(deck)
        chunk = pd.DataFrame({"deck_uuid": [str(deck.nocab_uuid)], "relic_count": [3]})
        constructor = DeckLabelDataConstructor(card_binder, deck_box, "relic_count")

        result = constructor.build(chunk)

        assert len(result) == 1
        deck_cards, label = result[0]
        assert {card.nocab_uuid for card in deck_cards} == {
            card1.nocab_uuid,
            card2.nocab_uuid,
        }
        assert label == 3.0

    def test_drops_unresolved_card_but_keeps_the_row(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        resolvable_card = _card("Strike", "strike")
        card_binder.create(resolvable_card)
        unresolvable_card = _card("Ghost Card", "ghost")  # never registered
        deck = _deck([resolvable_card, unresolvable_card])
        deck_box.create(deck)
        chunk = pd.DataFrame({"deck_uuid": [str(deck.nocab_uuid)], "relic_count": [3]})
        constructor = DeckLabelDataConstructor(card_binder, deck_box, "relic_count")

        result = constructor.build(chunk)

        assert len(result) == 1
        deck_cards, label = result[0]
        assert [card.nocab_uuid for card in deck_cards] == [resolvable_card.nocab_uuid]
        assert label == 3.0

    def test_skips_row_when_every_card_in_the_deck_is_unresolved(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        unresolvable_card = _card("Ghost Card", "ghost")
        deck = _deck([unresolvable_card])
        deck_box.create(deck)
        chunk = pd.DataFrame({"deck_uuid": [str(deck.nocab_uuid)], "relic_count": [3]})
        constructor = DeckLabelDataConstructor(card_binder, deck_box, "relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_unresolvable_deck_uuid(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        chunk = pd.DataFrame({"deck_uuid": [str(uuid4())], "relic_count": [3]})
        constructor = DeckLabelDataConstructor(card_binder, deck_box, "relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_malformed_deck_uuid(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        chunk = pd.DataFrame({"deck_uuid": ["not-a-uuid"], "relic_count": [3]})
        constructor = DeckLabelDataConstructor(card_binder, deck_box, "relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_empty_chunk_returns_empty_list(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        chunk = pd.DataFrame({"deck_uuid": [], "relic_count": []})
        constructor = DeckLabelDataConstructor(card_binder, deck_box, "relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_label_caster_defaults_to_float(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        card = _card("Strike", "strike")
        card_binder.create(card)
        deck = _deck([card])
        deck_box.create(deck)
        chunk = pd.DataFrame({"deck_uuid": [str(deck.nocab_uuid)], "win": [True]})
        constructor = DeckLabelDataConstructor(card_binder, deck_box, "win")

        _, label = constructor.build(chunk)[0]

        assert label == 1.0
        assert isinstance(label, float)

    def test_label_caster_can_be_overridden_for_string_labels(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        card = _card("Strike", "strike")
        card_binder.create(card)
        deck = _deck([card])
        deck_box.create(deck)
        chunk = pd.DataFrame(
            {"deck_uuid": [str(deck.nocab_uuid)], "character": ["CHARACTER.SILENT"]}
        )
        constructor = DeckLabelDataConstructor(
            card_binder, deck_box, "character", label_caster=str
        )

        _, label = constructor.build(chunk)[0]

        assert label == "CHARACTER.SILENT"
