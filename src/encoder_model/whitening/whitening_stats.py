"""WhiteningStats: a fitted whitening (mean and projection) plus where it
came from, saved to and loaded from a .pt file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

_KEYS = ("checkpoint", "holdout", "card_count", "mean", "projection")


@dataclass(frozen=True)
class WhiteningStats:
    """x -> (x - mean) @ projection, fitted on one text encoder's token
    states.

    checkpoint: the text encoder the statistics came from; a pipeline
        built on another encoder must refuse them (require_fitted_on).
    holdout: the HoldoutSpec whose TRAIN tier was sampled; a run under a
        different holdout must refuse them (the fit may have seen its TEST
        or VALIDATION cards).
    card_count: how many cards were encoded for the fit.
    mean: (input_dim,) float32, the token mean.
    projection: (input_dim, output_dim) float32, the PCA whitening matrix.
    """

    checkpoint: str
    holdout: HoldoutSpec
    card_count: int
    mean: torch.Tensor
    projection: torch.Tensor

    def __post_init__(self) -> None:
        """Exceptions: ValueError if mean is not 1-D, projection not 2-D,
        their input widths differ, or card_count < 1."""
        if self.mean.dim() != 1 or self.projection.dim() != 2:
            raise ValueError(
                f"mean must be 1-D and projection 2-D, got shapes "
                f"{tuple(self.mean.shape)} and {tuple(self.projection.shape)}"
            )
        if self.mean.shape[0] != self.projection.shape[0]:
            raise ValueError(
                f"mean width {self.mean.shape[0]} != projection input width "
                f"{self.projection.shape[0]}"
            )
        if self.card_count < 1:
            raise ValueError(f"card_count must be >= 1, got {self.card_count}")

    @property
    def input_dim(self) -> int:
        """The token width read. Inputs: none. Output: int. Side effects:
        none. Exceptions: none."""
        return int(self.projection.shape[0])

    @property
    def output_dim(self) -> int:
        """The token width returned (kept components). Inputs: none.
        Output: int. Side effects: none. Exceptions: none."""
        return int(self.projection.shape[1])

    def require_fitted_on(self, checkpoint: str) -> None:
        """Refuse use with another text encoder than the one fitted on.

        Inputs: checkpoint (the text encoder about to use these stats).
        Output: none. Side effects: none.
        Exceptions: ValueError if checkpoint != self.checkpoint.

        Example:
            >>> stats.require_fitted_on("answerdotai/ModernBERT-base")
        """
        if checkpoint != self.checkpoint:
            raise ValueError(
                f"whitening stats were fitted on {self.checkpoint!r}, "
                f"not {checkpoint!r}"
            )

    def save(self, path: Path) -> None:
        """Write to path (parents created), loadable with load().

        Inputs: path. Output: none.
        Side effects: creates path's parent directory; writes path.
        Exceptions: OSError on a write failure.

        Example:
            >>> stats.save(Path("data/whitening/modernbert_tokens.pt"))
        """
        # Plain types and tensors only, so load() can use weights_only
        holdout = {
            "seed": self.holdout.seed,
            "tier_ratios": list(self.holdout.tier_ratios),
            "held_out_games": sorted(
                game.value for game in self.holdout.held_out_games
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint": self.checkpoint,
                "holdout": holdout,
                "card_count": self.card_count,
                "mean": self.mean.detach().cpu(),
                "projection": self.projection.detach().cpu(),
            },
            path,
        )

    @staticmethod
    def load(path: Path) -> "WhiteningStats":
        """Read a file save() wrote.

        Inputs: path. Output: WhiteningStats.
        Side effects: reads path.
        Exceptions: FileNotFoundError if absent; ValueError if it is not a
            whitening stats file (missing keys) or fails __post_init__.

        Example:
            >>> WhiteningStats.load(Path("data/whitening/modernbert_tokens.pt")).output_dim
            768
        """
        raw: Any = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or any(key not in raw for key in _KEYS):
            raise ValueError(
                f"{path} is not a whitening stats file (want keys {_KEYS})"
            )
        tensors = (raw["mean"], raw["projection"])
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError(f"{path}: mean and projection must be tensors")
        holdout = raw["holdout"]
        holdout_keys = ("seed", "tier_ratios", "held_out_games")
        if not isinstance(holdout, dict) or any(
            key not in holdout for key in holdout_keys
        ):
            raise ValueError(f"{path}: holdout must be a mapping with {holdout_keys}")
        return WhiteningStats(
            checkpoint=raw["checkpoint"],
            holdout=HoldoutSpec(
                seed=holdout["seed"],
                tier_ratios=tuple(holdout["tier_ratios"]),
                held_out_games=frozenset(GameId(g) for g in holdout["held_out_games"]),
            ),
            card_count=raw["card_count"],
            mean=raw["mean"],
            projection=raw["projection"],
        )
