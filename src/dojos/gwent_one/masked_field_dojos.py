"""Thin per-metric wrappers over MaskedFieldMetric's eight concrete
gwent_one subclasses
(src/data_refinement/metrics/gwent_one/{faction,color,rarity,set,type,
armor,provision,power}_mask_metric.py).

Each wrapper is a thin SingleCardFixedClassificationDojo subclass - it
adds no behavior of its own, only configuration: it pulls its paired
metric class's own LABEL_VALUES/MASKED_FIELD/DEFAULT_OUTPUT_PATH
ClassVars by reference (never duplicated as a literal), passes them to
SingleCardFixedClassificationDojo.__init__ via a
MaskedFieldDataConstructor, and builds the mod that actually masks the
field out of every split's input.

MASKING MOD, train_only=False: every wrapper injects its own
MaskTargetKeyMod(key=<Metric>.MASKED_FIELD[-1], train_only=False) - the
masked field must be masked on test/validation input too, not just
training, or the model could just read the answer off raw_content
instead of predicting it (see plans/dojo_v2.md and this session's mod-
scope discussion). train_only=False is what makes that hold on every
split; ModPipeline/Mod's own train_only default (True) is for the
opposite case, augmentation a model shouldn't see at eval time.

KNOWN LIMITATION, not fixed by this dojo: MaskTargetKeyMod.key masks
one top-level raw_content key
(card.raw_content[self.key] = "[MASK]") - it does not walk a multi-
segment MASKED_FIELD path. Every concrete MaskedFieldMetric today has a
single-element MASKED_FIELD (["faction"], ["armor"], etc - confirmed by
reading all 8 gwent_one subclasses), so MASKED_FIELD[-1] is correct for
every wrapper below. A future MaskedFieldMetric subclass with a
genuinely nested MASKED_FIELD (e.g. ["stat_block", "attack"]) would
need MaskTargetKeyMod extended to walk a path first - out of scope for
this slice.

LABEL_VALUES, no OTHER_LABEL appending needed: none of these 8
wrappers append masked_field_metric.OTHER_LABEL to the LABEL_VALUES
they pass through. FactionMaskMetric/ColorMaskMetric/RarityMaskMetric/
SetMaskMetric/TypeMaskMetric never fall back to it at all. ArmorMask-
Metric/ProvisionMaskMetric/PowerMaskMetric do fall back to it, but each
already bakes OTHER_LABEL into its own LABEL_VALUES tuple as a literal
element (confirmed by reading all three files) - appending it again
here would double the sentinel.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.gwent_one.armor_mask_metric import ArmorMaskMetric
from src.data_refinement.metrics.gwent_one.color_mask_metric import ColorMaskMetric
from src.data_refinement.metrics.gwent_one.faction_mask_metric import FactionMaskMetric
from src.data_refinement.metrics.gwent_one.power_mask_metric import PowerMaskMetric
from src.data_refinement.metrics.gwent_one.provision_mask_metric import (
    ProvisionMaskMetric,
)
from src.data_refinement.metrics.gwent_one.rarity_mask_metric import RarityMaskMetric
from src.data_refinement.metrics.gwent_one.set_mask_metric import SetMaskMetric
from src.data_refinement.metrics.gwent_one.type_mask_metric import TypeMaskMetric
from src.dojos.generic.data_constructors import MaskedFieldDataConstructor
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline


class FactionMaskDojo(SingleCardFixedClassificationDojo):
    """Card, faction masked -> predicted faction (FactionMaskMetric).
    label_values=FactionMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or FactionMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=FactionMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=FactionMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )


class ColorMaskDojo(SingleCardFixedClassificationDojo):
    """Card, color masked -> predicted color (ColorMaskMetric).
    label_values=ColorMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or ColorMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=ColorMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=ColorMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )


class RarityMaskDojo(SingleCardFixedClassificationDojo):
    """Card, rarity masked -> predicted rarity (RarityMaskMetric).
    label_values=RarityMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or RarityMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=RarityMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=RarityMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )


class SetMaskDojo(SingleCardFixedClassificationDojo):
    """Card, set masked -> predicted set (SetMaskMetric).
    label_values=SetMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or SetMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=SetMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [MaskTargetKeyMod(key=SetMaskMetric.MASKED_FIELD[-1], train_only=False)]
            ),
            rng_seed=rng_seed,
        )


class TypeMaskDojo(SingleCardFixedClassificationDojo):
    """Card, type masked -> predicted type (TypeMaskMetric).
    label_values=TypeMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or TypeMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=TypeMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=TypeMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )


class ArmorMaskDojo(SingleCardFixedClassificationDojo):
    """Card, armor masked -> predicted armor (ArmorMaskMetric).
    label_values=ArmorMaskMetric.LABEL_VALUES unchanged - OTHER_LABEL is
    already a literal element of that tuple (this metric falls back to
    it for values outside the observed dense range)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or ArmorMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=ArmorMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=ArmorMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )


class ProvisionMaskDojo(SingleCardFixedClassificationDojo):
    """Card, provision masked -> predicted provision (ProvisionMaskMetric).
    label_values=ProvisionMaskMetric.LABEL_VALUES unchanged - OTHER_LABEL
    is already a literal element of that tuple (this metric falls back
    to it for values outside the observed dense range)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or ProvisionMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=ProvisionMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=ProvisionMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )


class PowerMaskDojo(SingleCardFixedClassificationDojo):
    """Card, power masked -> predicted power (PowerMaskMetric).
    label_values=PowerMaskMetric.LABEL_VALUES unchanged - OTHER_LABEL is
    already a literal element of that tuple (this metric falls back to
    it for values outside the observed dense range)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or PowerMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=PowerMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=PowerMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )
