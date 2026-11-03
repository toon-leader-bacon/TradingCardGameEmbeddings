from typing import List

from src.dojos.mods.mod import Mod, ModTally
from src.schema.type_hints import LabelT, TrainingDatum


class ModPipeline:
    """
    A ModPipeline is a list of Mods that are applied to the data in order.

    ORDER MATTERS - RENAMING MODS GO LAST. KeyCaseMod and FlattenKeysMod
    (content_style_mods.py) rename and move keys; the value mods there
    (NumberToStringMod, BoolToStringMod, ListToStringMod, ValueCaseMod)
    change value types. Anything that finds fields by name or path
    (MaskTargetKeyMod, WeightedFieldMaskMod's FieldPath, label extraction
    in a dojo) must run BEFORE them in the pipeline. Run it after and it
    finds nothing: the best-effort mods pass the card through unchanged,
    so a mask silently fails to fire and the label leaks into the input.
    A dojo's own task mods come first and run-config mods are appended
    after them (GenericDojo.append_mods), which keeps this order; watch
    the masking mods' ModTally.cards_changed after adding a renaming mod.

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
        self, data: TrainingDatum[LabelT], is_training: bool = True
    ) -> TrainingDatum[LabelT]:
        for mod in self.mods:
            if not is_training and mod.train_only:
                continue
            data = mod.apply_single(data)
        return data

    def apply(
        self, data: List[TrainingDatum[LabelT]], is_training: bool = True
    ) -> List[TrainingDatum[LabelT]]:
        for mod in self.mods:
            if not is_training and mod.train_only:
                continue
            data = mod.apply(data)
        return data
