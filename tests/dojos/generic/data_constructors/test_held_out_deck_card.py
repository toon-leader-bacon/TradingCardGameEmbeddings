from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors import HeldOutDeckCardDataConstructor
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _setup(
    deck_cards: list[GenericCard], visible: list[GenericCard]
) -> tuple[CardBinder, DeckBox, str]:
    """A binder holding only `visible` (a hidden card reads as None, as
    through a VisibleCardLookup) and a box holding one deck."""
    binder = CardBinder()
    for card in visible:
        binder.create(card)
    box = DeckBox()
    deck = GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name="deck",
        card_nocab_uuids=[card.nocab_uuid for card in deck_cards],
    )
    box.create(deck)
    return binder, box, str(deck.nocab_uuid)


def _chunk(deck_uuid: str, target: GenericCard, candidates: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "deck_uuid": [deck_uuid],
            "target_card_uuid": [str(target.nocab_uuid)],
            "candidate_uuids": [[str(card.nocab_uuid) for card in candidates]],
        }
    )


class TestHeldOutDeckCardDataConstructorBuild:
    def test_builds_candidates_then_deck_minus_every_target_copy(self) -> None:
        target, other, decoy_a, decoy_b = (_card(n) for n in "TODE")
        binder, box, deck_uuid = _setup(
            [target, other, target], [target, other, decoy_a, decoy_b]
        )
        chunk = _chunk(deck_uuid, target, [decoy_a, target, decoy_b])

        result = HeldOutDeckCardDataConstructor(box).build(chunk, binder)

        assert result == [([[decoy_a, target, decoy_b], [other]], 1)]

    def test_hidden_target_skips_the_row(self) -> None:
        target, other, decoy = (_card(n) for n in "TOD")
        binder, box, deck_uuid = _setup([target, other], [other, decoy])

        result = HeldOutDeckCardDataConstructor(box).build(
            _chunk(deck_uuid, target, [target, decoy]), binder
        )

        assert result == []

    def test_hidden_decoy_is_dropped_and_the_index_follows(self) -> None:
        target, other, hidden, decoy = (_card(n) for n in "TOHD")
        binder, box, deck_uuid = _setup([target, other], [target, other, decoy])

        result = HeldOutDeckCardDataConstructor(box).build(
            _chunk(deck_uuid, target, [hidden, decoy, target]), binder
        )

        assert result == [([[decoy, target], [other]], 1)]

    def test_no_visible_decoy_skips_the_row(self) -> None:
        target, other, hidden = (_card(n) for n in "TOH")
        binder, box, deck_uuid = _setup([target, other], [target, other])

        result = HeldOutDeckCardDataConstructor(box).build(
            _chunk(deck_uuid, target, [target, hidden]), binder
        )

        assert result == []

    def test_hidden_context_cards_are_dropped_and_empty_context_skips(self) -> None:
        target, hidden, decoy = (_card(n) for n in "THD")
        binder, box, deck_uuid = _setup([target, hidden], [target, decoy])

        result = HeldOutDeckCardDataConstructor(box).build(
            _chunk(deck_uuid, target, [target, decoy]), binder
        )

        assert result == []

    def test_unknown_or_malformed_deck_skips_the_row(self) -> None:
        target, other, decoy = (_card(n) for n in "TOD")
        binder, box, _ = _setup([target, other], [target, other, decoy])
        constructor = HeldOutDeckCardDataConstructor(box)

        assert (
            constructor.build(_chunk(str(uuid4()), target, [target, decoy]), binder)
            == []
        )
        assert (
            constructor.build(_chunk("not-a-uuid", target, [target, decoy]), binder)
            == []
        )

    def test_target_missing_from_candidates_skips_the_row(self) -> None:
        target, other, decoy_a, decoy_b = (_card(n) for n in "TOAB")
        binder, box, deck_uuid = _setup(
            [target, other], [target, other, decoy_a, decoy_b]
        )

        result = HeldOutDeckCardDataConstructor(box).build(
            _chunk(deck_uuid, target, [decoy_a, decoy_b]), binder
        )

        assert result == []
