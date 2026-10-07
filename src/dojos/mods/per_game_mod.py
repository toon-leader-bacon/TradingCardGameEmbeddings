"""PerGameMod: a mod whose behavior depends on the card's game.

A cross-game dojo trains on cards from several games, and both the field
a task masks and the default augmentations differ per game. This mod
holds one ModPipeline per game and applies the pipeline of each datum's
card's game, so the dojo's pipeline stays one flat list of mods.
"""

from typing import Mapping

from src.dojos.mods.mod import Mod, ModTally
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.card import GenericCard
from src.schema.game_id import GameId
from src.schema.type_hints import TrainingDatum


class PerGameMod(Mod):
    """Applies the pipeline of the datum's card's game (Strategy per game).

    pipelines: game -> the mods to run on that game's cards. A game with
        no entry passes through unchanged.
    train_only: as Mod. The inner mods always run when this mod does: this
        mod's own train_only decides the splits, so build masking pipelines
        with train_only=False here and augmentation pipelines train_only=True.

    Inputs (constructor): pipelines, train_only.
    Side effects: none; the input datum is never mutated.
    Exceptions: none from the constructor.

    Example:
        >>> mod = PerGameMod({GameId.GWENT: ModPipeline([MaskTargetKeyMod("rarity")])},
        ...                  train_only=False)
        >>> mod.apply_single((gwent_card, "tier_1"))
        (GenericCard(... raw_content={..., 'rarity': '[MASK]'}), 'tier_1')
    """

    def __init__(
        self, pipelines: Mapping[GameId, ModPipeline], train_only: bool
    ) -> None:
        super().__init__(train_only=train_only)
        self._pipelines = dict(pipelines)

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        """A new datum run through its card's game's pipeline.

        Inputs: data, a single-card TrainingDatum (GenericCard, label).
        Output: the pipeline's result, or data itself if that game has no
            pipeline.
        Side effects: none (the inner mods' tallies update).
        Exceptions: AssertionError if data is not a single-card datum;
            whatever the inner mods raise.

        Example:
            >>> PerGameMod({}, train_only=True).apply_single((card, 3)) == (card, 3)
            True
        """
        card, _ = data
        assert isinstance(card, GenericCard), "PerGameMod expects a single-card datum"

        # The datum's own game picks the pipeline; no entry means no-op
        pipeline = self._pipelines.get(card.source_game)
        if pipeline is None:
            return data
        return pipeline.apply_single(data, is_training=True)

    def child_tallies(self) -> dict[str, ModTally]:
        """The inner mods' tallies, keyed "<game>/<position>:<class name>".

        Inputs: none. Output: the live ModTally objects.
        Side effects: none. Exceptions: none.
        """
        return {
            f"{game.value}/{key}": tally
            for game, pipeline in self._pipelines.items()
            for key, tally in pipeline.mod_tallies().items()
        }
