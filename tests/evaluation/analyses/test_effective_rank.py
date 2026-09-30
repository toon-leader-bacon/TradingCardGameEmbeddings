import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.analyses.effective_rank import EffectiveRank
from tests.evaluation.analyses.tables import labeled_table


def _run(tmp_path: Path, vectors: np.ndarray) -> dict[str, float]:
    table, _ = labeled_table(tmp_path / "t.db", vectors, [None] * len(vectors))
    with table:
        result = EffectiveRank().run(table, tmp_path / "out")
    return dict(result.scalars)


def test_isotropic_vectors_use_nearly_every_direction(tmp_path: Path) -> None:
    vectors = np.random.default_rng(0).standard_normal((2000, 8))
    scalars = _run(tmp_path, vectors)
    assert scalars["effective_rank"] == pytest.approx(8, rel=0.05)
    assert scalars["effective_rank_fraction"] == pytest.approx(1, rel=0.05)
    assert scalars["embedding_dim"] == 8 and scalars["n_cards"] == 2000


def test_vectors_on_a_line_have_rank_one(tmp_path: Path) -> None:
    direction = np.array([1.0, 2.0, 0.0, -1.0])
    vectors = np.outer(np.linspace(-1, 1, 50), direction) + 3.0  # offset: centered
    scalars = _run(tmp_path, vectors)
    assert scalars["effective_rank"] == pytest.approx(1, abs=1e-3)
    assert scalars["top1_variance_share"] == pytest.approx(1, abs=1e-6)


def test_two_equal_directions_have_rank_two(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    vectors = np.zeros((4000, 6))
    vectors[:, :2] = rng.choice([-1.0, 1.0], size=(4000, 2))
    scalars = _run(tmp_path, vectors)
    assert scalars["effective_rank"] == pytest.approx(2, rel=0.02)
    assert scalars["top5_variance_share"] == pytest.approx(1, abs=1e-6)


def test_scalars_are_written(tmp_path: Path) -> None:
    _run(tmp_path, np.random.default_rng(2).standard_normal((10, 3)))
    written = json.loads((tmp_path / "out" / "scalars.json").read_text())
    assert "effective_rank" in written


@pytest.mark.parametrize(
    "vectors",
    [np.ones((1, 4)), np.ones((5, 4)), np.full((7, 4), 0.1)],
    ids=["one card", "no variance", "no variance, inexact mean"],
)
def test_degenerate_tables_are_refused_and_nothing_is_written(
    tmp_path: Path, vectors: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        _run(tmp_path, vectors)
    assert not (tmp_path / "out").exists()
