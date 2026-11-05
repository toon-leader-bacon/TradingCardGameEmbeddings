import math
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from src.schema.splits import Split
from src.teacher import cooccurrence as cooccurrence_module
from src.teacher.cooccurrence import CooccurrenceCounts, count_cooccurrence
from src.teacher.teacher_settings import (
    CooccurrenceOptions,
    CopyCountRule,
    DeckSizeRule,
)
from tests.teacher.teacher_fixtures import binder_and_cards, teacher_inputs

_TRAIN = frozenset({Split.TRAIN})
_TRAIN_TIER = frozenset({CardTier.TRAIN})


def _pair(counts: CooccurrenceCounts, first: UUID, second: UUID) -> float:
    rows = {card_uuid: row for row, card_uuid in enumerate(counts.card_uuids)}
    return float(counts.pair_counts[rows[first], rows[second]])


def _deck_count(counts: CooccurrenceCounts, card: UUID) -> int:
    return int(counts.deck_counts[counts.card_uuids.index(card)])


def _counted(
    tmp_path: Path, deck_cards: list[list[UUID]], binder_cards: tuple, **options: object
) -> CooccurrenceCounts:
    binder, _ = binder_cards
    inputs = teacher_inputs(tmp_path, binder, deck_cards)
    return count_cooccurrence(
        inputs, _TRAIN, _TRAIN_TIER, CooccurrenceOptions(**options)  # type: ignore[arg-type]
    )


class TestPresence:
    def test_counts_shared_decks_per_pair(self, tmp_path: Path) -> None:
        binder_cards = binder_and_cards(3)
        a, b, c = binder_cards[1]

        counts = _counted(tmp_path, [[a, b], [a, b], [a, c]], binder_cards)

        assert _pair(counts, a, b) == 2
        assert _pair(counts, a, c) == 1
        assert _pair(counts, b, c) == 0
        assert counts.deck_count == 3
        assert [_deck_count(counts, card) for card in (a, b, c)] == [3, 2, 1]

    def test_matrix_is_symmetric_with_zero_diagonal(self, tmp_path: Path) -> None:
        binder_cards = binder_and_cards(3)
        a, b, c = binder_cards[1]

        counts = _counted(tmp_path, [[a, b, c], [a, a, b]], binder_cards)

        dense = counts.pair_counts.toarray()
        assert np.array_equal(dense, dense.T)
        assert np.all(np.diag(dense) == 0)

    def test_copies_count_once(self, tmp_path: Path) -> None:
        binder_cards = binder_and_cards(2)
        a, b = binder_cards[1]

        counts = _counted(tmp_path, [[a, a, a, b]], binder_cards)

        assert _pair(counts, a, b) == 1


class TestCopyCountRules:
    @pytest.mark.parametrize(
        ("rule", "expected"),
        [
            (CopyCountRule.COPIES, 3.0),
            (CopyCountRule.LOG_COPIES, 1.0 + math.log(3)),
        ],
    )
    def test_weighs_copies(
        self, tmp_path: Path, rule: CopyCountRule, expected: float
    ) -> None:
        binder_cards = binder_and_cards(2)
        a, b = binder_cards[1]

        counts = _counted(tmp_path, [[a, a, a, b]], binder_cards, copy_count_rule=rule)

        assert _pair(counts, a, b) == pytest.approx(expected)


class TestDeckSizeRules:
    def test_inverse_size_gives_each_deck_one_unit(self, tmp_path: Path) -> None:
        binder_cards = binder_and_cards(3)
        a, b, c = binder_cards[1]

        counts = _counted(
            tmp_path,
            [[a, b], [a, b, c]],
            binder_cards,
            deck_size_rule=DeckSizeRule.INVERSE_SIZE,
        )

        # [a, b]: one unordered pair, half the unit each way; [a, b, c]: 1/6 each
        assert _pair(counts, a, b) == pytest.approx(0.5 + 1 / 6)
        assert _pair(counts, b, c) == pytest.approx(1 / 6)
        assert counts.pair_counts.sum() == pytest.approx(2.0)

    def test_weighting_never_moves_deck_counts(self, tmp_path: Path) -> None:
        binder_cards = binder_and_cards(2)
        a, b = binder_cards[1]
        decks = [[a], [a, b]]

        uniform = _counted(tmp_path / "u", decks, binder_cards)
        inverse = _counted(
            tmp_path / "i",
            decks,
            binder_cards,
            deck_size_rule=DeckSizeRule.INVERSE_SIZE,
        )

        assert np.array_equal(uniform.deck_counts, inverse.deck_counts)
        assert uniform.deck_count == inverse.deck_count == 2


class TestSelection:
    def test_drops_cards_outside_the_tiers(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(30)
        holdout = HoldoutSpec(
            seed=0,
            tier_ratios=(1, 1, 0),
            held_out_games=frozenset(),
            deck_split_ratios=(1, 0, 0),
        )
        inputs = teacher_inputs(tmp_path, binder, [cards], holdout)

        counts = count_cooccurrence(inputs, _TRAIN, _TRAIN_TIER, CooccurrenceOptions())

        expected = {
            card
            for card in cards
            if holdout.tier_of(card, GameId.MTG) == CardTier.TRAIN
        }
        assert 0 < len(expected) < len(cards)
        assert set(counts.card_uuids) == expected

    def test_counts_only_the_chosen_deck_splits(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(2)
        holdout = HoldoutSpec(
            seed=0,
            tier_ratios=(1, 0, 0),
            held_out_games=frozenset(),
            deck_split_ratios=(1, 1, 0),
        )
        inputs = teacher_inputs(tmp_path, binder, [cards] * 40, holdout)

        counts = count_cooccurrence(inputs, _TRAIN, _TRAIN_TIER, CooccurrenceOptions())

        train_decks = len(inputs.deck_rows_by_split[Split.TRAIN])
        assert 0 < train_decks < 40
        assert counts.deck_count == train_decks

    def test_no_selected_card_raises(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(2)
        inputs = teacher_inputs(tmp_path, binder, [cards])

        with pytest.raises(ValueError, match="card tiers"):
            count_cooccurrence(
                inputs, _TRAIN, frozenset({CardTier.TEST}), CooccurrenceOptions()
            )

    def test_no_selected_deck_raises(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(2)
        inputs = teacher_inputs(tmp_path, binder, [cards])

        with pytest.raises(ValueError, match="deck splits"):
            count_cooccurrence(
                inputs, frozenset({Split.TEST}), _TRAIN_TIER, CooccurrenceOptions()
            )


class TestChunking:
    def test_one_deck_chunks_match_one_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binder, cards = binder_and_cards(4)
        a, b, c, d = cards
        inputs = teacher_inputs(tmp_path, binder, [[a, b], [b, c, c], [a, d], [d]])
        whole = count_cooccurrence(inputs, _TRAIN, _TRAIN_TIER, CooccurrenceOptions())

        monkeypatch.setattr(cooccurrence_module, "_DECKS_PER_CHUNK", 1)
        chunked = count_cooccurrence(inputs, _TRAIN, _TRAIN_TIER, CooccurrenceOptions())

        assert np.array_equal(
            whole.pair_counts.toarray(), chunked.pair_counts.toarray()
        )
        assert np.array_equal(whole.deck_counts, chunked.deck_counts)
        assert whole.deck_count == chunked.deck_count == 4
