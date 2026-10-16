"""WhiteningStats: shape checks, the checkpoint guard, save/load round trip."""

from pathlib import Path

import pytest
import torch

from src.encoder_model.whitening.whitening_stats import WhiteningStats
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

HOLDOUT = HoldoutSpec(
    seed=3, tier_ratios=(8, 1, 1), held_out_games=frozenset({GameId.GWENT})
)


def _stats(input_dim: int = 4, output_dim: int = 3) -> WhiteningStats:
    return WhiteningStats(
        checkpoint="ckpt",
        holdout=HOLDOUT,
        card_count=10,
        mean=torch.arange(input_dim, dtype=torch.float32),
        projection=torch.ones(input_dim, output_dim),
    )


def test_widths_come_from_the_projection() -> None:
    stats = _stats(4, 3)
    assert (stats.input_dim, stats.output_dim) == (4, 3)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "stats.pt"
    _stats().save(path)
    loaded = WhiteningStats.load(path)
    assert loaded.checkpoint == "ckpt"
    assert loaded.holdout == HOLDOUT
    assert loaded.card_count == 10
    assert torch.equal(loaded.mean, _stats().mean)
    assert torch.equal(loaded.projection, _stats().projection)


def test_a_file_from_before_deck_splits_loads_with_an_unknown_deck_split(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.pt"
    _stats().save(path)
    raw = torch.load(path, weights_only=True)
    raw["holdout"] = {"seed": 3, "tier_ratios": [8, 1, 1], "held_out_games": ["gwent"]}
    torch.save(raw, path)

    loaded = WhiteningStats.load(path)

    assert loaded.holdout.deck_split_ratios is None
    assert loaded.holdout.card_tier_key() == HOLDOUT.card_tier_key()


def test_an_unreadable_holdout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    _stats().save(path)
    raw = torch.load(path, weights_only=True)
    raw["holdout"] = 7
    torch.save(raw, path)

    with pytest.raises(ValueError, match="holdout"):
        WhiteningStats.load(path)


def test_a_file_that_is_not_stats_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2)}, path)
    with pytest.raises(ValueError, match="not a whitening stats file"):
        WhiteningStats.load(path)


def test_stats_refuse_another_checkpoint() -> None:
    _stats().require_fitted_on("ckpt")
    with pytest.raises(ValueError, match="fitted on 'ckpt'"):
        _stats().require_fitted_on("other")


@pytest.mark.parametrize(
    "mean, projection, card_count",
    [
        (torch.zeros(4, 1), torch.ones(4, 3), 10),
        (torch.zeros(4), torch.ones(5, 3), 10),
        (torch.zeros(4), torch.ones(4, 3), 0),
    ],
)
def test_inconsistent_stats_are_refused(
    mean: torch.Tensor, projection: torch.Tensor, card_count: int
) -> None:
    with pytest.raises(ValueError):
        WhiteningStats("ckpt", HOLDOUT, card_count, mean, projection)


def test_a_stats_file_with_non_tensor_values_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    _stats().save(path)
    raw = torch.load(path, weights_only=True)
    raw["mean"] = [0.0, 1.0, 2.0, 3.0]
    torch.save(raw, path)
    with pytest.raises(ValueError, match="must be tensors"):
        WhiteningStats.load(path)
