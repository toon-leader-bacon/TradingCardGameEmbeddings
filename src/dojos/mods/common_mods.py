import random
from typing import List

from src.dojos.mods.mod import Mod
from src.schema.card import GenericCard
from src.schema.type_hints import TrainingDatum


class NoOpMod(Mod):
    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        return data

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        return data


class MaskTargetKeyMod(Mod):
    def __init__(self, key: str):
        self.key = key

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        card = data[0]
        assert isinstance(
            card, GenericCard
        ), "MaskTargetKeyMod expects a single-card TrainingDatum"
        card.raw_content[self.key] = "[MASK]"
        return (card, data[1])

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        return [self.apply_single(datum) for datum in data]


class ShuffleDeckMod(Mod):
    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        deck = data[0]
        assert isinstance(deck, list) and all(
            isinstance(card, GenericCard) for card in deck
        ), "ShuffleDeckMod expects a multi-card TrainingDatum"
        random.shuffle(deck)
        return (deck, data[1])

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        return [self.apply_single(datum) for datum in data]
