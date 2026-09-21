from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
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


class TestDeckCardMaskDataConstructorBuild:
    def test_excludes_target_card_keeps_others(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        target, other = _card("Geralt", "geralt"), _card("Yennefer", "yennefer")
        card_binder.create(target)
        card_binder.create(other)
        deck = _deck([target, other])
        deck_box.create(deck)
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(deck.nocab_uuid)],
                "target_card_uuid": [str(target.nocab_uuid)],
                "label": ["Geralt"],
            }
        )
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        result = constructor.build(chunk, card_binder)

        assert len(result) == 1
        deck_cards, label = result[0]
        assert [card.nocab_uuid for card in deck_cards] == [other.nocab_uuid]
        assert label == "Geralt"

    def test_label_passed_through_unchanged(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        target, other = _card("Geralt", "geralt"), _card("Yennefer", "yennefer")
        card_binder.create(target)
        card_binder.create(other)
        deck = _deck([target, other])
        deck_box.create(deck)
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(deck.nocab_uuid)],
                "target_card_uuid": [str(target.nocab_uuid)],
                "label": ["OTHER"],
            }
        )
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        _, label = constructor.build(chunk, card_binder)[0]

        assert label == "OTHER"
        assert isinstance(label, str)

    def test_skips_row_with_unresolvable_deck_uuid(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(uuid4())],
                "target_card_uuid": [str(uuid4())],
                "label": ["Geralt"],
            }
        )
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        result = constructor.build(chunk, card_binder)

        assert result == []

    def test_malformed_target_uuid_excludes_nothing_but_keeps_the_row(self) -> None:
        # Unlike a malformed deck_uuid (which drops the row entirely), a
        # malformed target_card_uuid degrades gracefully to "exclude
        # nothing" - see _deck_cards_excluding_target()'s docstring for
        # why (defensive path; DeckCardMaskMetric's own schema always
        # writes a valid uuid string here in practice).
        card_binder = CardBinder()
        deck_box = DeckBox()
        card1, card2 = _card("Geralt", "geralt"), _card("Yennefer", "yennefer")
        card_binder.create(card1)
        card_binder.create(card2)
        deck = _deck([card1, card2])
        deck_box.create(deck)
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(deck.nocab_uuid)],
                "target_card_uuid": ["not-a-uuid"],
                "label": ["Geralt"],
            }
        )
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        result = constructor.build(chunk, card_binder)

        assert len(result) == 1
        deck_cards, _ = result[0]
        assert {card.nocab_uuid for card in deck_cards} == {
            card1.nocab_uuid,
            card2.nocab_uuid,
        }

    def test_skips_row_when_every_non_target_card_is_unresolved(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        target = _card("Geralt", "geralt")
        card_binder.create(target)
        unresolvable = _card("Ghost Card", "ghost")  # never registered
        deck = _deck([target, unresolvable])
        deck_box.create(deck)
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(deck.nocab_uuid)],
                "target_card_uuid": [str(target.nocab_uuid)],
                "label": ["Geralt"],
            }
        )
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        result = constructor.build(chunk, card_binder)

        assert result == []

    def test_drops_unresolved_non_target_card_but_keeps_the_row(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        target = _card("Geralt", "geralt")
        resolvable = _card("Yennefer", "yennefer")
        card_binder.create(target)
        card_binder.create(resolvable)
        unresolvable = _card("Ghost Card", "ghost")  # never registered
        deck = _deck([target, resolvable, unresolvable])
        deck_box.create(deck)
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(deck.nocab_uuid)],
                "target_card_uuid": [str(target.nocab_uuid)],
                "label": ["Geralt"],
            }
        )
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        result = constructor.build(chunk, card_binder)

        assert len(result) == 1
        deck_cards, _ = result[0]
        assert [card.nocab_uuid for card in deck_cards] == [resolvable.nocab_uuid]

    def test_processes_multiple_rows_independently(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        target1, other1 = _card("Geralt", "geralt"), _card("Yennefer", "yennefer")
        target2, other2 = _card("Ciri", "ciri"), _card("Vesemir", "vesemir")
        for card in (target1, other1, target2, other2):
            card_binder.create(card)
        deck1 = _deck([target1, other1])
        deck2 = _deck([target2, other2])
        deck_box.create(deck1)
        deck_box.create(deck2)
        chunk = pd.DataFrame(
            {
                "deck_uuid": [str(deck1.nocab_uuid), str(deck2.nocab_uuid)],
                "target_card_uuid": [str(target1.nocab_uuid), str(target2.nocab_uuid)],
                "label": ["Geralt", "Ciri"],
            }
        )
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        result = constructor.build(chunk, card_binder)

        assert len(result) == 2
        (cards1, label1), (cards2, label2) = result
        assert [card.nocab_uuid for card in cards1] == [other1.nocab_uuid]
        assert label1 == "Geralt"
        assert [card.nocab_uuid for card in cards2] == [other2.nocab_uuid]
        assert label2 == "Ciri"

    def test_empty_chunk_returns_empty_list(self) -> None:
        card_binder = CardBinder()
        deck_box = DeckBox()
        chunk = pd.DataFrame({"deck_uuid": [], "target_card_uuid": [], "label": []})
        constructor = DeckCardMaskDataConstructor(deck_box, "label")

        result = constructor.build(chunk, card_binder)

        assert result == []
