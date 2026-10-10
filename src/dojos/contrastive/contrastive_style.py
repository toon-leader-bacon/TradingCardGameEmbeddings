"""ContrastiveStyle: one contrastive training style's matched pair constructor and loss.

Abstract Factory (PATTERNS.md): a pair constructor and a loss must agree
on item shape and on what positive_cliques mean (ContrastiveDojo's wiring
invariant), so each style builds both together and they can't be mixed
up. The catalog's ContrastiveDojoRecipe takes a style; everything else
about a contrastive dojo (dealer, mods, staple subsampling) is shared.
Each concrete style lives in styles/, beside its own constructor and loss.
"""

from typing import Protocol

from src.dojos.contrastive.contrastive_loss import ContrastiveLoss
from src.dojos.contrastive.pair_constructor import ContrastivePairConstructor
from src.dojos.contrastive.staple_subsampling import StapleSubsampling


class ContrastiveStyle(Protocol):
    """Builds one style's pair constructor and its matching loss."""

    def pair_constructor(
        self, rng_seed: int, staple_subsampling: StapleSubsampling | None
    ) -> ContrastivePairConstructor:
        """This style's pair constructor.

        Inputs: rng_seed (its sampling seed), staple_subsampling (or None
            for t = inf).
        Output: a ContrastivePairConstructor whose batches this style's
            contrastive_loss() reads.
        Side effects: none.
        Exceptions: ValueError from the constructor on a bad setting.
        """
        ...

    def contrastive_loss(self) -> ContrastiveLoss:
        """This style's loss, matching pair_constructor()'s batches.

        Inputs: none. Output: ContrastiveLoss. Side effects: none.
        Exceptions: none.
        """
        ...
