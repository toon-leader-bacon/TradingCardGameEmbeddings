import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.analyses.card_sample import AllCards, PerLabelCap
from src.evaluation.analyses.cluster_agreement import ClusterAgreement
from tests.evaluation.analyses.tables import labeled_table, separated_clusters


def test_separated_labels_agree_almost_perfectly(tmp_path: Path) -> None:
    vectors, labels = separated_clusters()
    table, card_labels = labeled_table(tmp_path / "t.db", vectors, labels)
    out = tmp_path / "cluster_agreement"
    with table:
        result = ClusterAgreement(card_labels, AllCards(), seed=0).run(table, out)
    assert result.scalars["ari"] > 0.95 and result.scalars["nmi"] > 0.95
    assert result.scalars["n_cards"] == 60 and result.scalars["n_labels"] == 3
    assert result.files == (out / "scalars.json",)
    assert json.loads((out / "scalars.json").read_text()) == dict(result.scalars)


def test_shuffled_labels_agree_about_as_well_as_chance(tmp_path: Path) -> None:
    vectors, labels = separated_clusters()
    shuffled = list(np.random.default_rng(1).permutation(labels))
    table, card_labels = labeled_table(tmp_path / "t.db", vectors, shuffled)
    with table:
        result = ClusterAgreement(card_labels, AllCards(), seed=0).run(
            table, tmp_path / "out"
        )
    assert abs(result.scalars["ari"]) < 0.2


def test_the_same_seed_gives_the_same_scores(tmp_path: Path) -> None:
    vectors, labels = separated_clusters(seed=3)
    table, card_labels = labeled_table(tmp_path / "t.db", vectors, labels)
    analysis = ClusterAgreement(card_labels, PerLabelCap(10), seed=7)
    with table:
        first = analysis.run(table, tmp_path / "a").scalars
        second = analysis.run(table, tmp_path / "b").scalars
    assert first == second


def test_refusals_write_nothing(tmp_path: Path) -> None:
    vectors, _ = separated_clusters(per_label=3)
    table, one_label = labeled_table(tmp_path / "t.db", vectors, ["x"] * 9)
    out = tmp_path / "out"
    with table:
        with pytest.raises(ValueError, match="distinct labels"):
            ClusterAgreement(one_label, AllCards(), seed=0).run(table, out)
        assert not out.exists()
        out.mkdir()
        (out / "old.json").write_text("{}")
        with pytest.raises(FileExistsError):
            ClusterAgreement(one_label, AllCards(), seed=0).run(table, out)
    with pytest.raises(ValueError):
        ClusterAgreement(one_label, AllCards(), seed=0, n_init=0)
