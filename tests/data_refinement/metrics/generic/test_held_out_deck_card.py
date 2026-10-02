from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.generic.held_out_deck_card.candidate_sampler import (
    CandidateSampler,
    _staple_weights,
)
from src.data_refinement.metrics.generic.held_out_deck_card.deck_sample import (
    DeckSample,
)
from src.data_refinement.metrics.generic.held_out_deck_card.metric import (
    HeldOutDeckCardMetric,
)
from src.data_refinement.metrics.generic.held_out_deck_card.sampling import (
    HeldOutCardSampling,
)
from src.data_refinement.metrics.version_metadata import read_version_metadata
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_GAME = GameId.GWENT
_UNKNOWN = CardBinder.unknown_card_uuid(_GAME)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=_GAME,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _deck(card_uuids: list[UUID]) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=_GAME,
        name="deck",
        card_nocab_uuids=card_uuids,
    )


def _sampling(**overrides: object) -> HeldOutCardSampling:
    fields: dict = dict(max_decks=None, targets_per_deck=2, decoy_count=3, seed=0)
    fields.update(overrides)
    return HeldOutCardSampling(**fields)


def _box_of(decks: list[list[UUID]]) -> DeckBox:
    box = DeckBox()
    for cards in decks:
        box.create(_deck(cards))
    return box


def _two_clusters(cards_per_cluster: int = 6, decks_per_cluster: int = 10):
    """Two disjoint card pools ("factions"); every deck draws 4 cards from
    one pool, plus a staple in every deck and an Unknown in some."""
    rng = np.random.default_rng(1)
    pools = [[uuid4() for _ in range(cards_per_cluster)] for _ in range(2)]
    staple = uuid4()
    decks: list[list[UUID]] = []
    for pool in pools:
        for index in range(decks_per_cluster):
            picked = [pool[int(i)] for i in rng.choice(len(pool), 4, replace=False)]
            extra = [_UNKNOWN] if index % 2 == 0 else []
            decks.append([*picked, picked[0], staple, *extra])
    return pools, staple, decks


class TestHeldOutCardSampling:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_decks": 0},
            {"targets_per_deck": 0},
            {"decoy_count": 0},
            {"cooccurrence_decoy_share": -0.1},
            {"cooccurrence_decoy_share": 1.5},
            {"staple_threshold": 0.0},
        ],
    )
    def test_rejects_out_of_range_fields(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            _sampling(**overrides)

    def test_accepts_boundary_values(self) -> None:
        sampling = _sampling(max_decks=1, cooccurrence_decoy_share=1.0)
        assert sampling.max_decks == 1


class TestDeckSample:
    def test_indexes_distinct_cards_and_inverts_them(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        box = _box_of([[a, a, b], [b, c]])

        sample = DeckSample.from_deck_box(box, _GAME, None, 0, _UNKNOWN)

        assert sample.deck_count == 2
        assert sample.card_count == 3
        for deck in range(sample.deck_count):
            cards = sample.cards_of(deck)
            assert list(cards) == sorted(set(cards.tolist()))
            for card in cards:
                assert deck in sample.decks_containing(int(card))
        frequencies = dict(zip(sample.card_uuids, sample.document_frequencies()))
        assert frequencies == {a: 1, b: 2, c: 1}

    def test_never_indexes_the_excluded_card_and_drops_thin_decks(self) -> None:
        a, b = uuid4(), uuid4()
        box = _box_of([[a, _UNKNOWN, _UNKNOWN], [a, a], [a, b, _UNKNOWN]])

        sample = DeckSample.from_deck_box(box, _GAME, None, 0, _UNKNOWN)

        assert sample.deck_count == 1
        assert _UNKNOWN not in sample.card_uuids
        assert set(sample.card_uuids) == {a, b}

    def test_max_decks_caps_the_sample(self) -> None:
        _, _, decks = _two_clusters()
        box = _box_of(decks)

        sample = DeckSample.from_deck_box(box, _GAME, 5, 0, _UNKNOWN)

        assert sample.deck_count == 5

    def test_empty_box_gives_an_empty_sample(self) -> None:
        sample = DeckSample.from_deck_box(DeckBox(), _GAME, None, 0, _UNKNOWN)
        assert sample.deck_count == 0
        assert sample.card_count == 0

    def test_out_of_range_indices_raise(self) -> None:
        sample = DeckSample.from_deck_box(
            _box_of([[uuid4(), uuid4()]]), _GAME, None, 0, _UNKNOWN
        )
        with pytest.raises(IndexError):
            sample.cards_of(1)
        with pytest.raises(IndexError):
            sample.decks_containing(-1)


class TestStapleWeights:
    def test_down_weights_only_cards_above_the_threshold(self) -> None:
        weights = _staple_weights(np.array([1, 25, 100]), 100, 0.25)
        assert weights == pytest.approx([1.0, 1.0, 0.5])


class TestCandidateSampler:
    def _sample(self) -> tuple[DeckSample, list[list[UUID]], UUID]:
        pools, staple, decks = _two_clusters()
        box = _box_of(decks)
        return DeckSample.from_deck_box(box, _GAME, None, 0, _UNKNOWN), pools, staple

    def test_every_row_holds_out_a_deck_card_against_outside_decoys(self) -> None:
        sample, _, _ = self._sample()
        sampler = CandidateSampler(sample, _sampling())

        for deck in range(sample.deck_count):
            deck_cards = {sample.card_uuids[int(c)] for c in sample.cards_of(deck)}
            rows = sampler.rows_for(deck)
            assert 1 <= len(rows) <= 2
            assert len({row.target_card_uuid for row in rows}) == len(rows)
            for row in rows:
                assert row.deck_uuid == sample.deck_uuids[deck]
                assert row.target_card_uuid in deck_cards
                candidates = list(row.candidate_uuids)
                assert len(candidates) == len(set(candidates)) <= 4
                assert candidates.count(row.target_card_uuid) == 1
                decoys = set(candidates) - {row.target_card_uuid}
                assert decoys and not decoys & deck_cards
                assert _UNKNOWN not in candidates

    def test_cooccurrence_decoys_share_a_deck_with_the_target(self) -> None:
        sample, _, _ = self._sample()
        sampler = CandidateSampler(sample, _sampling(staple_threshold=1.0))
        found = 0

        for deck in range(sample.deck_count):
            deck_cards = sample.cards_of(deck)
            for target in deck_cards:
                decoy = sampler._cooccurrence_decoy(deck, deck_cards, int(target), [])
                if decoy is None:
                    continue
                found += 1
                assert decoy not in deck_cards
                shared = set(sample.decks_containing(int(target)).tolist()) & set(
                    sample.decks_containing(decoy).tolist()
                )
                assert shared - {deck}
        assert found > 0

    def test_same_seed_gives_the_same_rows(self) -> None:
        sample, _, _ = self._sample()

        first = [CandidateSampler(sample, _sampling()).rows_for(d) for d in range(3)]
        second = [CandidateSampler(sample, _sampling()).rows_for(d) for d in range(3)]

        assert first == second

    def test_a_deck_with_no_possible_decoy_yields_no_row(self) -> None:
        a, b = uuid4(), uuid4()
        sample = DeckSample.from_deck_box(_box_of([[a, b]]), _GAME, None, 0, _UNKNOWN)

        assert CandidateSampler(sample, _sampling()).rows_for(0) == []

    def test_empty_sample_raises(self) -> None:
        sample = DeckSample.from_deck_box(DeckBox(), _GAME, None, 0, _UNKNOWN)
        with pytest.raises(ValueError):
            CandidateSampler(sample, _sampling())


class _GwentHeldOutCardMetric(HeldOutDeckCardMetric):
    SOURCE_GAME = _GAME
    DEFAULT_OUTPUT_PATH = Path("unused.parquet")
    SAMPLING = HeldOutCardSampling(max_decks=None, targets_per_deck=2, decoy_count=3)


class TestHeldOutDeckCardMetric:
    def _stamped_box(self, tmp_path: Path, binder: CardBinder) -> DeckBox:
        _, _, decks = _two_clusters()
        path = tmp_path / "box.db"
        _box_of(decks).save(path, _GAME, binder.version_for(_GAME))
        return DeckBox.load([path])

    def test_scan_writes_rows_with_deck_box_version_metadata(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        box = self._stamped_box(tmp_path, binder)
        output = tmp_path / "out" / "held_out_card_gwent.parquet"

        path = _GwentHeldOutCardMetric(binder, box, output_path=output).scan()

        table = pq.read_table(path)
        assert path == output
        assert table.column_names == [
            "deck_uuid",
            "target_card_uuid",
            "candidate_uuids",
        ]
        assert 20 <= table.num_rows <= 40
        row = table.slice(0, 1).to_pylist()[0]
        assert row["target_card_uuid"] in row["candidate_uuids"]
        metadata = read_version_metadata(path)
        assert metadata is not None
        assert metadata.requires_deck_box
        assert metadata.card_binder_version == binder.version_for(_GAME)

    def test_scan_is_reproducible(self, tmp_path: Path) -> None:
        binder = CardBinder()
        box = self._stamped_box(tmp_path, binder)

        first = _GwentHeldOutCardMetric(binder, box, tmp_path / "a.parquet").scan()
        second = _GwentHeldOutCardMetric(binder, box, tmp_path / "b.parquet").scan()

        assert pq.read_table(first).equals(pq.read_table(second))

    def test_sampling_override_applies(self, tmp_path: Path) -> None:
        binder = CardBinder()
        box = self._stamped_box(tmp_path, binder)
        sampling = _sampling(max_decks=3, targets_per_deck=1)

        path = _GwentHeldOutCardMetric(
            binder, box, tmp_path / "o.parquet", sampling
        ).scan()

        assert pq.read_table(path).num_rows == 3

    def test_unstamped_box_raises(self, tmp_path: Path) -> None:
        _, _, decks = _two_clusters()

        with pytest.raises(ValueError, match="re-ingest"):
            _GwentHeldOutCardMetric(
                CardBinder(), _box_of(decks), tmp_path / "o.parquet"
            ).scan()
