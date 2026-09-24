from pathlib import Path

import pandas as pd
import pytest

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
from src.dojos.gwent_one.masked_field_dojos import (
    ArmorMaskDojo,
    ColorMaskDojo,
    FactionMaskDojo,
    PowerMaskDojo,
    ProvisionMaskDojo,
    RarityMaskDojo,
    SetMaskDojo,
    TypeMaskDojo,
)
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.schema.holdout import HoldoutSpec

_CASES = [
    (FactionMaskDojo, FactionMaskMetric),
    (ColorMaskDojo, ColorMaskMetric),
    (RarityMaskDojo, RarityMaskMetric),
    (SetMaskDojo, SetMaskMetric),
    (TypeMaskDojo, TypeMaskMetric),
    (ArmorMaskDojo, ArmorMaskMetric),
    (ProvisionMaskDojo, ProvisionMaskMetric),
    (PowerMaskDojo, PowerMaskMetric),
]


def _write_source(path: Path) -> None:
    df = pd.DataFrame(
        {
            "nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
            "masked_field": [["faction"]] * 10,
            "label": ["northern_realms"] * 10,
        }
    )
    df.to_parquet(path, index=False)


class TestMaskedFieldDojoWrappers:
    @pytest.mark.parametrize("dojo_cls,metric_cls", _CASES)
    def test_wires_data_constructor_and_label_values(
        self, dojo_cls, metric_cls, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = dojo_cls(
            card_binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, SingleCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, MaskedFieldDataConstructor)
        assert dojo.label_values == list(metric_cls.LABEL_VALUES)

    @pytest.mark.parametrize("dojo_cls,metric_cls", _CASES)
    def test_wires_a_train_only_false_masking_mod_for_its_own_field(
        self, dojo_cls, metric_cls, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = dojo_cls(
            card_binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        mods = dojo.data_mod_pipeline.mods
        assert isinstance(mods[0], MaskTargetKeyMod)
        assert mods[0].key == metric_cls.MASKED_FIELD[-1]
        assert mods[0].train_only is False

    def test_every_wrapper_except_faction_masks_only_its_own_field(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        for dojo_cls, _ in _CASES:
            if dojo_cls is FactionMaskDojo:
                continue
            dojo = dojo_cls(
                card_binder,
                HoldoutSpec.no_holdout(),
                card_embedding_size=4,
                path_to_training_data=source,
                rng_seed=0,
                strict_version_check=False,
            )
            assert len(dojo.data_mod_pipeline.mods) == 1

    def test_faction_dojo_also_masks_the_correlated_faction_duo_field(
        self, tmp_path: Path
    ) -> None:
        # faction-duo always contains the true faction as its own prefix
        # (confirmed against the live corpus); left unmasked, it would let
        # the model read the masked faction straight back off it.
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = FactionMaskDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        mods = dojo.data_mod_pipeline.mods
        assert len(mods) == 2
        assert mods[1].key == "faction-duo"
        assert mods[1].train_only is False
