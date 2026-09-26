"""Per-metric wrappers over MaskedFieldMetric's eight concrete
gwent_one subclasses
(src/data_refinement/metrics/gwent_one/{faction,color,rarity,set,type,
armor,provision,power}_mask_metric.py).

Each is a MaskedFieldMetricDojo (../generic/paired_metric_dojos.py) that
names its paired metric; that base masks the field on every split.

FactionMaskDojo ALSO masks "faction-duo" (present on 15 of 1260 cards):
its value always starts with the true faction ("syndicate_monster" for
faction "syndicate"), so leaving it unmasked would let the model read
the masked faction straight back off it. Masking it on every card, not
only the 15 that have it, is harmless (the other cards gain a
"faction-duo": "[MASK]" key) and keeps this dojo's input shape uniform.
No other wrapper's field has a correlated key in gwent.one's raw data
(the one found, category vs. color, is dropped at ingestion instead -
see card_binder/gwent_one/ingestion_stage.py).

LABEL_VALUES are passed through unchanged. FactionMaskMetric/
ColorMaskMetric/RarityMaskMetric/SetMaskMetric/TypeMaskMetric never
fall back to OTHER_LABEL; ArmorMaskMetric/ProvisionMaskMetric/
PowerMaskMetric do, and already include OTHER_LABEL in their own
LABEL_VALUES, so appending it here would double the sentinel.
"""

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
from src.dojos.generic.paired_metric_dojos import MaskedFieldMetricDojo


class FactionMaskDojo(MaskedFieldMetricDojo):
    """Card, faction masked -> predicted faction (FactionMaskMetric).
    label_values=FactionMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL. Also masks "faction-duo" - see this
    module's docstring for why."""

    METRIC = FactionMaskMetric
    EXTRA_MASKED_KEYS = ("faction-duo",)


class ColorMaskDojo(MaskedFieldMetricDojo):
    """Card, color masked -> predicted color (ColorMaskMetric).
    label_values=ColorMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    METRIC = ColorMaskMetric


class RarityMaskDojo(MaskedFieldMetricDojo):
    """Card, rarity masked -> predicted rarity (RarityMaskMetric).
    label_values=RarityMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    METRIC = RarityMaskMetric


class SetMaskDojo(MaskedFieldMetricDojo):
    """Card, set masked -> predicted set (SetMaskMetric).
    label_values=SetMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    METRIC = SetMaskMetric


class TypeMaskDojo(MaskedFieldMetricDojo):
    """Card, type masked -> predicted type (TypeMaskMetric).
    label_values=TypeMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL."""

    METRIC = TypeMaskMetric


class ArmorMaskDojo(MaskedFieldMetricDojo):
    """Card, armor masked -> predicted armor (ArmorMaskMetric).
    label_values=ArmorMaskMetric.LABEL_VALUES unchanged - OTHER_LABEL is
    already a literal element of that tuple (this metric falls back to
    it for values outside the observed dense range)."""

    METRIC = ArmorMaskMetric


class ProvisionMaskDojo(MaskedFieldMetricDojo):
    """Card, provision masked -> predicted provision (ProvisionMaskMetric).
    label_values=ProvisionMaskMetric.LABEL_VALUES unchanged - OTHER_LABEL
    is already a literal element of that tuple (this metric falls back
    to it for values outside the observed dense range)."""

    METRIC = ProvisionMaskMetric


class PowerMaskDojo(MaskedFieldMetricDojo):
    """Card, power masked -> predicted power (PowerMaskMetric).
    label_values=PowerMaskMetric.LABEL_VALUES unchanged - OTHER_LABEL is
    already a literal element of that tuple (this metric falls back to
    it for values outside the observed dense range)."""

    METRIC = PowerMaskMetric
