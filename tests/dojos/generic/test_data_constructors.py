from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors import (
    CardAverageDataConstructor,
    CardCharacterPredictionDataConstructor,
    DeckCardMaskDataConstructor,
    DeckLabelDataConstructor,
    MaskedFieldDataConstructor,
)
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


class TestCardAverageDataConstructorBuild:
    def test_builds_one_datum_per_resolvable_row(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "average_relic_count": [1.5],
                "sample_count": [10],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == [(card, 1.5)]

    def test_skips_row_with_unresolvable_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(uuid4())],
                "average_relic_count": [1.5],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_malformed_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": ["not-a-uuid"],
                "average_relic_count": [1.5],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_unparseable_label(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "average_relic_count": ["not-a-float"],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_reads_bool_label_column_as_float(self) -> None:
        # CardWinRateMetric's label column holds bools (win/loss) - see
        # CardAverageMetric's WIN RATE IS AN AVERAGE docstring note.
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "win_rate": [True],
            }
        )
        constructor = CardAverageDataConstructor(binder, "win_rate")

        result = constructor.build(chunk)

        assert result == [(card, 1.0)]

    def test_empty_chunk_returns_empty_list(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame({"nocab_uuid": [], "average_relic_count": []})
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []


class TestMaskedFieldDataConstructorBuild:
    def test_builds_one_datum_per_resolvable_row(self) -> None:
        binder = CardBinder()
        card = _card("Geralt of Rivia", "geralt")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "masked_field": [["faction"]],
                "label": ["northern_realms"],
            }
        )
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == [(card, "northern_realms")]

    def test_skips_row_with_unresolvable_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(uuid4())],
                "masked_field": [["faction"]],
                "label": ["northern_realms"],
            }
        )
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_malformed_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": ["not-a-uuid"],
                "masked_field": [["faction"]],
                "label": ["northern_realms"],
            }
        )
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == []

    def test_empty_chunk_returns_empty_list(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame({"nocab_uuid": [], "masked_field": [], "label": []})
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == []


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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        result = constructor.build(chunk)

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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        _, label = constructor.build(chunk)[0]

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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        result = constructor.build(chunk)

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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        result = constructor.build(chunk)

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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        result = constructor.build(chunk)

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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        result = constructor.build(chunk)

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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        result = constructor.build(chunk)

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
        constructor = DeckCardMaskDataConstructor(card_binder, deck_box, "label")

        result = constructor.build(chunk)

        assert result == []


class TestCardCharacterPredictionDataConstructorBuild:
    def test_single_character_row_builds_single_entry_dict(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "characters": [["CHARACTER.SILENT"]],
                "probabilities": [[1.0]],
                "sample_count": [1],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [(card, {"CHARACTER.SILENT": 1.0})]

    def test_multi_character_row_zips_into_one_dict(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "characters": [["CHARACTER.SILENT", "CHARACTER.REGENT"]],
                "probabilities": [[0.6, 0.4]],
                "sample_count": [5],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [(card, {"CHARACTER.SILENT": 0.6, "CHARACTER.REGENT": 0.4})]

    def test_skips_row_with_unresolvable_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(uuid4())],
                "characters": [["CHARACTER.SILENT"]],
                "probabilities": [[1.0]],
                "sample_count": [1],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == []

    def test_processes_multiple_rows_independently(self) -> None:
        binder = CardBinder()
        card1 = _card("Strike", "strike")
        card2 = _card("Defend", "defend")
        binder.create(card1)
        binder.create(card2)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card1.nocab_uuid), str(card2.nocab_uuid)],
                "characters": [["CHARACTER.SILENT"], ["CHARACTER.REGENT"]],
                "probabilities": [[1.0], [1.0]],
                "sample_count": [1, 1],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [
            (card1, {"CHARACTER.SILENT": 1.0}),
            (card2, {"CHARACTER.REGENT": 1.0}),
        ]

    def test_empty_chunk_returns_empty_list(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [],
                "characters": [],
                "probabilities": [],
                "sample_count": [],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == []
