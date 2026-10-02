import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.contrastive.pair_constructor import SingleCardPairConstructor
from src.dojos.contrastive.staple_subsampling import (
    DocumentFrequency,
    StapleSubsampling,
    cached_document_frequency,
)
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.splits import Split

_STAPLE, _RARE = uuid4(), uuid4()


def _deck(*card_uuids: UUID) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.DOMINION,
        name="deck",
        card_nocab_uuids=list(card_uuids),
        provenance=None,
    )


def _table(shares: dict[UUID, float], deck_count: int = 100) -> DocumentFrequency:
    return DocumentFrequency("v1", deck_count, shares)


class TestDocumentFrequency:
    def test_counts_decks_not_copies(self) -> None:
        other = uuid4()
        decks = [_deck(_STAPLE, _STAPLE, _STAPLE), _deck(_STAPLE, other)]

        table = DocumentFrequency.from_decks(decks, "v1")

        assert table.deck_count == 2
        assert table.share_of(_STAPLE) == 1.0
        assert table.share_of(other) == 0.5
        assert table.share_of(uuid4()) == 0.0

    def test_json_round_trips(self) -> None:
        table = _table({_STAPLE: 0.9, _RARE: 0.01})
        assert DocumentFrequency.from_json(table.to_json()) == table

    def test_an_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError):
            DocumentFrequency.from_decks([], "v1")

    @pytest.mark.parametrize(
        ("deck_count", "share"), [(0, 0.5), (10, 0.0), (10, 1.5), (10, math.nan)]
    )
    def test_validates(self, deck_count: int, share: float) -> None:
        with pytest.raises(ValueError):
            DocumentFrequency("v1", deck_count, {_STAPLE: share})


class TestStapleSubsampling:
    def test_keep_probability_is_the_word2vec_formula(self) -> None:
        subsampling = StapleSubsampling(0.25, _table({_STAPLE: 1.0, _RARE: 0.1}))
        assert subsampling.keep_probability(_STAPLE) == pytest.approx(0.5)
        assert subsampling.keep_probability(_RARE) == 1.0  # df <= t
        assert subsampling.keep_probability(uuid4()) == 1.0  # never seen

    @pytest.mark.parametrize("threshold", [0.0, -1.0, math.inf, math.nan])
    def test_threshold_must_be_finite_and_positive(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            StapleSubsampling(threshold, _table({}))

    def test_kept_thins_staples_at_their_keep_rate(self) -> None:
        subsampling = StapleSubsampling(0.25, _table({_STAPLE: 1.0}))
        cards = [_STAPLE] * 10_000 + [_RARE] * 100
        before = list(cards)

        kept = subsampling.kept(cards, random.Random(0))

        assert cards == before  # only read
        assert kept.count(_RARE) == 100
        assert kept.count(_STAPLE) / 10_000 == pytest.approx(0.5, abs=0.02)


class _FakeDealer:
    """A DeckBoxDealer stand-in: a version and a fixed TRAIN deck list."""

    def __init__(self, decks: list[GenericDeck], version: str | None = "v1") -> None:
        self.decks = decks
        self.card_binder_version = version
        self.reads = 0

    def deck_count(self, split: Split) -> int:
        return len(self.decks) if split == Split.TRAIN else 0

    def decks_for(
        self, split: Split, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        assert split == Split.TRAIN and decks_per_sample == 1 and not shuffle
        for deck in self.decks:
            self.reads += 1
            yield [deck]


class TestCachedDocumentFrequency:
    def test_counts_the_first_decks_then_reuses_the_cache(self, tmp_path: Path) -> None:
        dealer = _FakeDealer([_deck(_STAPLE), _deck(_STAPLE, _RARE), _deck(_RARE)])
        cache = tmp_path / "sub" / "df.json"

        first = cached_document_frequency(dealer, cache, 2)  # type: ignore[arg-type]
        reads = dealer.reads
        second = cached_document_frequency(dealer, cache, 2)  # type: ignore[arg-type]

        assert first.deck_count == 2 and first.share_of(_STAPLE) == 1.0
        assert second == first and dealer.reads == reads
        assert cache.exists()

    def test_a_new_binder_version_recounts(self, tmp_path: Path) -> None:
        cache = tmp_path / "df.json"
        decks = [_deck(_STAPLE)]
        old_dealer: Any = _FakeDealer(decks, "v1")
        new_dealer: Any = _FakeDealer(decks, "v2")
        cached_document_frequency(old_dealer, cache, 5)

        table = cached_document_frequency(new_dealer, cache, 5)

        assert table.card_binder_version == "v2"

    def test_an_unreadable_cache_is_recounted(self, tmp_path: Path) -> None:
        cache = tmp_path / "df.json"
        cache.write_text("not json", encoding="utf-8")
        dealer: Any = _FakeDealer([_deck(_RARE)])
        table = cached_document_frequency(dealer, cache, 5)
        assert table.share_of(_RARE) == 1.0

    def test_a_box_without_a_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="CardBinder version"):
            cached_document_frequency(
                _FakeDealer([_deck(_RARE)], None), tmp_path / "df.json", 5  # type: ignore[arg-type]
            )


def _binder_cards(binder: CardBinder, count: int) -> list[GenericCard]:
    cards = []
    for index in range(count):
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.DOMINION,
            name=f"card{index}",
            raw_content={},
            provenance=Provenance(
                DataSource.DOMINIONTABS, str(index), datetime.now(timezone.utc)
            ),
        )
        binder.create(card)
        cards.append(card)
    return cards


class TestPairConstructorKnob:
    def test_no_subsampling_samples_exactly_as_before(self) -> None:
        binder = CardBinder()
        cards = _binder_cards(binder, 12)
        decks = [_deck(*[c.nocab_uuid for c in cards[i : i + 6]]) for i in (0, 6)]

        before = SingleCardPairConstructor(2, rng_seed=3).build(decks, binder)
        after = SingleCardPairConstructor(2, rng_seed=3, staple_subsampling=None).build(
            decks, binder
        )

        assert after.identities == before.identities

    def test_staples_are_thinned_from_positive_pairs(self) -> None:
        binder = CardBinder()
        staple, *kingdom = _binder_cards(binder, 11)
        decks = [
            _deck(
                *[staple.nocab_uuid] * 8,
                kingdom[i].nocab_uuid,
                kingdom[i + 5].nocab_uuid,
            )
            for i in range(5)
        ]
        table = _table({staple.nocab_uuid: 1.0})  # every kingdom card is rare
        constructor = SingleCardPairConstructor(
            2, rng_seed=0, staple_subsampling=StapleSubsampling(0.01, table)
        )

        staple_items = total = 0
        for _ in range(200):
            batch = constructor.build(decks, binder)
            total += len(batch.identities)
            staple_items += sum(i == (staple.nocab_uuid,) for i in batch.identities)

        # Unthinned, 8 of 10 slots are the staple; at keep 0.1 about 0.8 of
        # ~2.8 surviving cards are
        assert staple_items / total < 0.4

    def test_a_deck_thinned_below_items_per_deck_is_skipped(self) -> None:
        binder = CardBinder()
        staple, rare_a, rare_b, rare_c = _binder_cards(binder, 4)
        all_staples = _deck(*[staple.nocab_uuid] * 5)
        kept_deck = _deck(rare_a.nocab_uuid, rare_b.nocab_uuid, rare_c.nocab_uuid)
        table = _table({staple.nocab_uuid: 1.0})
        subsampling = StapleSubsampling(1e-9, table)  # keeps ~0 staples
        constructor = SingleCardPairConstructor(
            2, rng_seed=0, staple_subsampling=subsampling
        )

        batch = constructor.build([all_staples, kept_deck], binder)

        assert batch.positive_cliques == [[0, 1]]
        assert all(identity[0] != staple.nocab_uuid for identity in batch.identities)

    def test_the_unknown_sentinel_stays_excluded(self) -> None:
        binder = CardBinder()
        cards = _binder_cards(binder, 3)
        unknown = CardBinder.unknown_card_uuid(GameId.DOMINION)
        deck = _deck(*[c.nocab_uuid for c in cards], *[unknown] * 20)
        table = _table({unknown: 0.001})  # rare, so never thinned
        constructor = SingleCardPairConstructor(
            3, rng_seed=0, staple_subsampling=StapleSubsampling(0.5, table)
        )

        batch = constructor.build([deck], binder)

        assert unknown not in {identity[0] for identity in batch.identities}
