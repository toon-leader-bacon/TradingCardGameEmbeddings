from typing import List

from src.dojos.mods.mod import Mod, ModTally
from src.schema.type_hints import TrainingDatum


class ModPipeline:
    """
    A ModPipeline is a list of Mods that are applied to the data in order.

    is_training selects which mods actually run: a mod whose own
    train_only is True is skipped when is_training=False, so a caller
    can run the same pipeline unconditionally across every split (see
    each Mod's own train_only for which case it falls into) without the
    caller having to sort mods into separate lists itself. Defaults to
    True so an existing caller that never passes is_training keeps
    running every mod, matching this pipeline's behavior before
    train_only existed.
    """

    def __init__(self, mods: List[Mod]):
        self.mods = mods

    def mod_tallies(self) -> dict[str, ModTally]:
        """The ModTally of every mod that keeps one, keyed
        "<position>:<class name>" so two mods of one class stay apart.

        Inputs: none. Output: dict[str, ModTally] (the live objects).
        Side effects: none. Exceptions: none.

        Example:
            >>> ModPipeline([ShuffleKeysMod()]).mod_tallies()
            {'0:ShuffleKeysMod': ModTally(cards_seen=0, ...)}
        """
        result: dict[str, ModTally] = {}
        for position, mod in enumerate(self.mods):
            key = f"{position}:{type(mod).__name__}"
            if mod.tally is not None:
                result[key] = mod.tally

            # A mod holding other mods reports theirs under its own key
            for child_key, child_tally in mod.child_tallies().items():
                result[f"{key}/{child_key}"] = child_tally
        return result

    def apply_single(
        self, data: TrainingDatum, is_training: bool = True
    ) -> TrainingDatum:
        for mod in self.mods:
            if not is_training and mod.train_only:
                continue
            data = mod.apply_single(data)
        return data

    def apply(
        self, data: List[TrainingDatum], is_training: bool = True
    ) -> List[TrainingDatum]:
        for mod in self.mods:
            if not is_training and mod.train_only:
                continue
            data = mod.apply(data)
        return data
