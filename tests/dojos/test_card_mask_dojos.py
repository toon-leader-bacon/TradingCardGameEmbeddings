"""Wiring tests for the single-card masked-field wrappers of the games
added after gwent_one/dominiontabs (scryfall, pokemon_tcg,
cardvault_fabtcg, spire_codex): each builds its cell with its metric's
path, constructor and labels, and masks the field plus its leaking keys
on every split."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.cardvault_fabtcg import card_mask_dojos as fabtcg
from src.dojos.generic.data_constructors import (
    MaskedFieldDataConstructor,
    MaskedFieldMultiLabelDataConstructor,
    MaskedFieldRegressionDataConstructor,
)
from src.dojos.generic.paired_metric_dojos import (
    MaskedFieldMetricDojo,
    MaskedFieldMultiLabelMetricDojo,
    MaskedFieldRegressionMetricDojo,
)
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.pokemon_tcg import card_mask_dojos as pokemon
from src.dojos.scryfall import card_mask_dojos as scryfall
from src.dojos.spire_codex import card_mask_dojos as sts2
from src.schema.card_factory import MissingPathPolicy
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")

_WRAPPERS = [
    scryfall.CmcRegressionDojo,
    scryfall.CardTypeMaskDojo,
    scryfall.RarityMaskDojo,
    scryfall.ColorsMaskDojo,
    scryfall.PowerRegressionDojo,
    scryfall.ToughnessRegressionDojo,
    pokemon.HpRegressionDojo,
    pokemon.TypesMaskDojo,
    pokemon.StageMaskDojo,
    pokemon.RetreatCostRegressionDojo,
    pokemon.WeaknessMaskDojo,
    fabtcg.PitchMaskDojo,
    fabtcg.CostRegressionDojo,
    fabtcg.PowerRegressionDojo,
    fabtcg.DefenseRegressionDojo,
    fabtcg.ClassMaskDojo,
    fabtcg.CardTypeMaskDojo,
    sts2.CostMaskDojo,
    sts2.CardTypeMaskDojo,
    sts2.RarityMaskDojo,
    sts2.ColorMaskDojo,
]

_CONSTRUCTOR_BY_BASE = {
    MaskedFieldMetricDojo: MaskedFieldDataConstructor,
    MaskedFieldRegressionMetricDojo: MaskedFieldRegressionDataConstructor,
    MaskedFieldMultiLabelMetricDojo: MaskedFieldMultiLabelDataConstructor,
}


def _build(dojo_cls, tmp_path: Path):
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {"nocab_uuid": ["x"], "masked_field": [["f"]], "label": ["1"]}
    ).to_parquet(source, index=False)
    return dojo_cls(
        CardBinder(),
        HoldoutSpec.no_holdout(),
        card_embedding_size=4,
        path_to_training_data=source,
        rng_seed=0,
        strict_version_check=False,
    )


@pytest.mark.parametrize("dojo_cls", _WRAPPERS, ids=lambda c: c.__qualname__)
def test_wires_its_base_constructor(dojo_cls, tmp_path: Path) -> None:
    dojo = _build(dojo_cls, tmp_path)

    (base,) = [b for b in _CONSTRUCTOR_BY_BASE if issubclass(dojo_cls, b)]
    assert isinstance(dojo.data_constructor, _CONSTRUCTOR_BY_BASE[base])
    if base is not MaskedFieldRegressionMetricDojo:
        assert dojo.label_values == list(dojo_cls.METRIC.LABEL_VALUES)


@pytest.mark.parametrize("dojo_cls", _WRAPPERS, ids=lambda c: c.__qualname__)
def test_masks_the_field_and_its_leaks_on_every_split(dojo_cls, tmp_path: Path) -> None:
    mods = _build(dojo_cls, tmp_path).data_mod_pipeline.mods

    assert all(isinstance(mod, MaskTargetKeyMod) for mod in mods)
    assert all(mod.train_only is False for mod in mods)
    assert [mod.key for mod in mods] == [
        dojo_cls.METRIC.MASKED_FIELD[-1],
        *dojo_cls.EXTRA_MASKED_KEYS,
        *dojo_cls.EXTRA_MASKED_PATHS,
    ]


def test_nested_leak_paths_skip_cards_without_them(tmp_path: Path) -> None:
    mods = _build(pokemon.TypesMaskDojo, tmp_path).data_mod_pipeline.mods

    path_mods = [mod for mod in mods if isinstance(mod.key, tuple)]
    assert [mod.key for mod in path_mods] == [
        ("attacks", index, "cost") for index in range(4)
    ]
    assert all(mod.missing is MissingPathPolicy.PASS for mod in path_mods)
    assert mods[0].missing is MissingPathPolicy.ADD
