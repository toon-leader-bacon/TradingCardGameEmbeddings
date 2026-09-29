import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.analyses.card_sample import AllCards, PerLabelCap
from src.evaluation.analyses.label_compactness import LabelCompactness
from tests.evaluation.analyses.tables import labeled_table, separated_clusters


def _at_degrees(*angles: float) -> np.ndarray:
    radians = np.deg2rad(angles)
    return np.stack([np.cos(radians), np.sin(radians)], axis=1)


def test_separated_labels_are_compact(tmp_path: Path) -> None:
    vectors, labels = separated_clusters()
    table, card_labels = labeled_table(tmp_path / "t.db", vectors, labels)
    with table:
        scalars = (
            LabelCompactness(card_labels, AllCards(), seed=0)
            .run(table, tmp_path / "out")
            .scalars
        )
    assert scalars["silhouette"] > 0.8
    assert scalars["nearest_centroid_accuracy"] == 1.0
    assert scalars["majority_baseline"] == pytest.approx(1 / 3)
    assert scalars["n_scored_cards"] == 60
    assert scalars["n_cards"] == 60 and scalars["n_labels"] == 3


def test_a_card_never_votes_for_itself(tmp_path: Path) -> None:
    # "a" cards sit at 0 and 180 degrees; "b" cards at 10 and 20. With
    # leave-one-out, both "a" cards are misclassified (0.5 accuracy); a
    # centroid including the card itself would rescue the 180 one (0.75).
    vectors = _at_degrees(0, 180, 10, 20)
    table, card_labels = labeled_table(tmp_path / "t.db", vectors, ["a", "a", "b", "b"])
    with table:
        scalars = (
            LabelCompactness(card_labels, AllCards(), seed=0)
            .run(table, tmp_path / "out")
            .scalars
        )
    assert scalars["nearest_centroid_accuracy"] == 0.5


def test_single_card_labels_are_left_out_of_the_centroid_score(
    tmp_path: Path,
) -> None:
    vectors = _at_degrees(0, 5, 90, 95, 200)
    table, card_labels = labeled_table(
        tmp_path / "t.db", vectors, ["a", "a", "b", "b", "c"]
    )
    with table:
        scalars = (
            LabelCompactness(card_labels, AllCards(), seed=0)
            .run(table, tmp_path / "out")
            .scalars
        )
    assert scalars["n_scored_cards"] == 4 and scalars["n_cards"] == 5
    assert scalars["majority_baseline"] == 0.5
    assert all(math.isfinite(value) for value in scalars.values())


def test_a_single_card_label_still_competes_as_a_centroid(tmp_path: Path) -> None:
    # "c" (one card, at 3 degrees) is not scored, but it sits nearer each
    # "a" card than that card's leave-one-out "a" centroid, so both "a"
    # cards are misclassified: 0.5. Ignoring "c" as a candidate would give 1.
    vectors = _at_degrees(0, 5, 90, 95, 3)
    table, card_labels = labeled_table(
        tmp_path / "t.db", vectors, ["a", "a", "b", "b", "c"]
    )
    with table:
        scalars = (
            LabelCompactness(card_labels, AllCards(), seed=0)
            .run(table, tmp_path / "out")
            .scalars
        )
    assert scalars["nearest_centroid_accuracy"] == 0.5


def test_a_sample_of_one_card_per_label_is_refused(tmp_path: Path) -> None:
    vectors, labels = separated_clusters(per_label=4)
    table, card_labels = labeled_table(tmp_path / "t.db", vectors, labels)
    out = tmp_path / "out"
    with table, pytest.raises(ValueError, match="at least 2 sampled cards"):
        LabelCompactness(card_labels, PerLabelCap(1), seed=0).run(table, out)
    assert not out.exists()
