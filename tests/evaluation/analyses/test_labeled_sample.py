from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from src.evaluation.analyses.card_sample import AllCards
from src.evaluation.analyses.labeled_sample import LabeledSample, draw_labeled_sample
from src.evaluation.card_row import CardRow
from src.schema.game_id import GameId
from tests.evaluation.analyses.tables import labeled_table


def test_unlabeled_cards_are_dropped_and_vectors_align(tmp_path: Path) -> None:
    vectors = np.arange(12, dtype=np.float32).reshape(4, 3)
    table, labels = labeled_table(tmp_path / "t.db", vectors, ["b", None, "a", "a"])
    with table:
        sample = draw_labeled_sample(table, labels, AllCards(), seed=0)
        stored = table.vectors_for(list(sample.rows))
    assert len(sample.rows) == 3
    assert np.array_equal(sample.vectors, stored)
    assert [labels.label_of(row) for row in sample.rows] == list(sample.labels)
    assert sample.distinct_labels == ("a", "b")
    assert sample.size_scalars() == {"n_cards": 3.0, "n_labels": 2.0}


@pytest.mark.parametrize(
    ("labels", "match"),
    [(["a", None, None], "2 labeled cards"), (["a", "a", "a"], "2 distinct labels")],
)
def test_an_unmeasurable_sample_is_refused(
    tmp_path: Path, labels: list, match: str
) -> None:
    table, card_labels = labeled_table(tmp_path / "t.db", np.eye(3), labels)
    with table, pytest.raises(ValueError, match=match):
        draw_labeled_sample(table, card_labels, AllCards(), seed=0)


def test_unit_vectors_have_length_one_and_zero_stays_zero() -> None:
    sample = LabeledSample(
        rows=(CardRow(uuid4(), GameId.MTG), CardRow(uuid4(), GameId.MTG)),
        vectors=np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32),
        labels=("a", "b"),
    )
    unit = sample.unit_vectors()
    assert np.allclose(unit[0], [0.6, 0.8])
    assert np.array_equal(unit[1], [0.0, 0.0])
    assert np.array_equal(sample.vectors[0], [3.0, 4.0])  # original untouched


@pytest.mark.parametrize(
    ("row_count", "vectors", "label_count"),
    [
        (2, np.zeros((3, 2)), 2),  # vectors too long
        (2, np.zeros((2, 2)), 1),  # labels too short
        (2, np.zeros(2), 2),  # vectors not 2-D
    ],
)
def test_misaligned_samples_are_refused(
    row_count: int, vectors: np.ndarray, label_count: int
) -> None:
    rows = tuple(CardRow(uuid4(), GameId.MTG) for _ in range(row_count))
    with pytest.raises(ValueError):
        LabeledSample(rows=rows, vectors=vectors, labels=("a",) * label_count)
