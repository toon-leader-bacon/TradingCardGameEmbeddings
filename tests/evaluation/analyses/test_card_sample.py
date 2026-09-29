from uuid import uuid4

import pytest

from src.evaluation.analyses.card_sample import AllCards, PerLabelCap
from src.evaluation.card_row import CardRow
from src.schema.game_id import GameId
from tests.evaluation.analyses.tables import DictLabels

_ROWS = sorted(
    (CardRow(uuid4(), GameId.MTG) for _ in range(30)), key=lambda r: r.nocab_uuid
)
# 20 "a", 8 "b", 2 unlabeled
_LABELS = DictLabels(
    {row.nocab_uuid: ("a" if i < 20 else "b") for i, row in enumerate(_ROWS[:28])}
)


def test_all_cards_keeps_every_row() -> None:
    assert AllCards().select(_ROWS, None, seed=0) == _ROWS
    assert AllCards().select(_ROWS, _LABELS, seed=5) == _ROWS


def test_per_label_cap_caps_each_label_and_skips_unlabeled() -> None:
    chosen = PerLabelCap(5).select(_ROWS, _LABELS, seed=0)
    labels = [_LABELS.label_of(row) for row in chosen]
    assert labels.count("a") == 5 and labels.count("b") == 5
    assert None not in labels


def test_a_cap_above_a_labels_size_keeps_the_whole_label() -> None:
    chosen = PerLabelCap(10).select(_ROWS, _LABELS, seed=0)
    assert [_LABELS.label_of(row) for row in chosen].count("b") == 8


def test_the_choice_keeps_input_order_and_is_seeded() -> None:
    first = PerLabelCap(5).select(_ROWS, _LABELS, seed=1)
    assert first == [row for row in _ROWS if row in set(first)]
    assert PerLabelCap(5).select(_ROWS, _LABELS, seed=1) == first
    assert PerLabelCap(5).select(_ROWS, _LABELS, seed=2) != first


def test_per_label_cap_needs_labels_and_a_positive_cap() -> None:
    with pytest.raises(ValueError):
        PerLabelCap(5).select(_ROWS, None, seed=0)
    with pytest.raises(ValueError):
        PerLabelCap(0)
