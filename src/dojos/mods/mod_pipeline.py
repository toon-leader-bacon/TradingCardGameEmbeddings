from typing import List

from src.dojos.mods.mod import Mod
from src.schema.type_hints import TrainingDatum


class ModPipeline:
    """
    A ModPipeline is a list of Mods that are applied to the data in order.
    """

    def __init__(self, mods: List[Mod]):
        self.mods = mods

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        for mod in self.mods:
            data = mod.apply_single(data)
        return data

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        for mod in self.mods:
            data = mod.apply(data)
        return data
