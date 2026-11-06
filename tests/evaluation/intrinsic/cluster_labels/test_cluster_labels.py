from src.evaluation.intrinsic.cluster_labels.cluster_labels import (
    GameClusterLabels,
    HoldoutTierClusterLabels,
)
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from tests.evaluation.intrinsic.fakes import make_row


def test_game_labels_are_the_game_value() -> None:
    labels = GameClusterLabels()
    assert labels.grouping == "game"
    assert labels.cluster_label_of(make_row(GameId.GWENT)) == "gwent"
    assert labels.cluster_label_of(make_row(GameId.MTG)) == "mtg"


def test_holdout_tier_labels_agree_with_the_spec() -> None:
    spec = HoldoutSpec(
        seed=4, tier_ratios=(1, 1, 1), held_out_games=frozenset({GameId.GWENT})
    )
    labels = HoldoutTierClusterLabels(spec)
    assert labels.grouping == "holdout_tier"
    rows = [make_row() for _ in range(30)]
    assert [labels.cluster_label_of(row) for row in rows] == [
        spec.tier_of(row.nocab_uuid, row.source_game).value for row in rows
    ]
    assert labels.cluster_label_of(make_row(GameId.GWENT)) == "validation"
