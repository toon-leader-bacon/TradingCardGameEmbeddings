from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.gwent_one.ingestion_stage import (
    GwentOneCardIngestionStage,
)
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
from src.schema.game_id import GameId

# Real gwent.one card-wrap shapes, trimmed to the attributes these
# metrics read - mirrors tests/data_refinement/card_binder/gwent_one/
# test_ingestion_stage.py's fixture style.

_STRATAGEM_CARD = """
<div class="card-wrap card-data" data-id="1" data-power="0" data-armor="0"
     data-provision="0" data-faction="skellige" data-set="merchants of ofir"
     data-color="gold" data-type="stratagem" data-rarity="legendary">
    <div class="card-head">
        <div class="card-name"><a href="#">Mask of Uroboros</a></div>
        <div class="card-category">Location</div>
    </div>
    <div class="card-body"><div class="card-body-ability"></div></div>
</div>
"""

_UNIT_CARD = """
<div class="card-wrap card-data" data-id="2" data-power="5" data-armor="2"
     data-provision="8" data-faction="monster" data-set="baseset"
     data-color="bronze" data-type="unit" data-rarity="common">
    <div class="card-head">
        <div class="card-name"><a href="#">Werewolf</a></div>
        <div class="card-category">&nbsp;</div>
    </div>
    <div class="card-body"><div class="card-body-ability"></div></div>
</div>
"""

_ARTIFACT_CARD = """
<div class="card-wrap card-data" data-id="3" data-power="0" data-armor="0"
     data-provision="6" data-faction="neutral" data-set="baseset"
     data-color="bronze" data-type="artifact" data-rarity="rare">
    <div class="card-head">
        <div class="card-name"><a href="#">Horn</a></div>
        <div class="card-category">&nbsp;</div>
    </div>
    <div class="card-body"><div class="card-body-ability"></div></div>
</div>
"""

# A unit whose power/provision fall outside every metric's known
# LABEL_VALUES range - exercises the OTHER fallback.
_OUTLIER_UNIT_CARD = """
<div class="card-wrap card-data" data-id="4" data-power="99" data-armor="50"
     data-provision="200" data-faction="syndicate" data-set="novigrad"
     data-color="gold" data-type="unit" data-rarity="epic">
    <div class="card-head">
        <div class="card-name"><a href="#">Impossible Beast</a></div>
        <div class="card-category">&nbsp;</div>
    </div>
    <div class="card-body"><div class="card-body-ability"></div></div>
</div>
"""


def _binder_from_cards(card_htmls: list[str], tmp_path: Path) -> CardBinder:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "page_1.html").write_text("".join(card_htmls), encoding="utf-8")
    binder = CardBinder()
    GwentOneCardIngestionStage().ingest(raw_dir, binder)
    return binder


def _labels_by_name(df: pd.DataFrame, binder: CardBinder) -> dict[str, str]:
    uuid_to_name = {
        str(card.nocab_uuid): card.name for card in binder.all_cards(GameId.GWENT)
    }
    return {uuid_to_name[row["nocab_uuid"]]: row["label"] for _, row in df.iterrows()}


class TestFixedStringFields:
    """Faction/color/type/rarity/set: every card eligible, label is a
    direct raw_content lookup - one representative test each."""

    def test_faction_mask_metric(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_STRATAGEM_CARD, _UNIT_CARD], tmp_path)
        metric = FactionMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert len(df) == 2
        labels = _labels_by_name(df, binder)
        assert labels["Mask of Uroboros"] == "skellige"
        assert labels["Werewolf"] == "monster"
        assert all(list(field) == ["faction"] for field in df["masked_field"])

    def test_color_mask_metric(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_UNIT_CARD], tmp_path)
        metric = ColorMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert df.iloc[0]["label"] == "bronze"

    def test_type_mask_metric(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_ARTIFACT_CARD], tmp_path)
        metric = TypeMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert df.iloc[0]["label"] == "artifact"

    def test_rarity_mask_metric(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_STRATAGEM_CARD], tmp_path)
        metric = RarityMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert df.iloc[0]["label"] == "legendary"

    def test_set_mask_metric(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_STRATAGEM_CARD], tmp_path)
        metric = SetMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert df.iloc[0]["label"] == "merchants of ofir"

    @pytest.mark.parametrize(
        "metric_cls",
        [
            FactionMaskMetric,
            ColorMaskMetric,
            TypeMaskMetric,
            RarityMaskMetric,
            SetMaskMetric,
        ],
    )
    def test_every_card_is_eligible(self, metric_cls, tmp_path: Path) -> None:
        binder = _binder_from_cards(
            [_STRATAGEM_CARD, _UNIT_CARD, _ARTIFACT_CARD], tmp_path
        )
        metric = metric_cls(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert len(df) == 3


class TestProvisionMaskMetric:
    def test_stratagem_excluded(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_STRATAGEM_CARD, _UNIT_CARD], tmp_path)
        metric = ProvisionMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1
        assert df.iloc[0]["label"] == "8"

    def test_out_of_range_value_falls_back_to_other(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_OUTLIER_UNIT_CARD], tmp_path)
        metric = ProvisionMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert df.iloc[0]["label"] == "OTHER"


class TestPowerMaskMetric:
    def test_only_units_eligible(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_ARTIFACT_CARD, _UNIT_CARD], tmp_path)
        metric = PowerMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1
        assert df.iloc[0]["label"] == "5"

    def test_out_of_range_value_falls_back_to_other(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_OUTLIER_UNIT_CARD], tmp_path)
        metric = PowerMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert df.iloc[0]["label"] == "OTHER"


class TestArmorMaskMetric:
    def test_only_units_eligible(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_ARTIFACT_CARD, _UNIT_CARD], tmp_path)
        metric = ArmorMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1
        assert df.iloc[0]["label"] == "2"

    def test_zero_armor_is_a_valid_label_for_a_unit(self, tmp_path: Path) -> None:
        # _UNIT_CARD's own armor is nonzero, so this uses a second unit
        # with armor="0" to confirm zero stays a real, non-excluded label.
        zero_armor_unit = _UNIT_CARD.replace('data-id="2"', 'data-id="5"').replace(
            'data-armor="2"', 'data-armor="0"'
        )
        binder = _binder_from_cards([zero_armor_unit], tmp_path)
        metric = ArmorMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert len(df) == 1
        assert df.iloc[0]["label"] == "0"

    def test_out_of_range_value_falls_back_to_other(self, tmp_path: Path) -> None:
        binder = _binder_from_cards([_OUTLIER_UNIT_CARD], tmp_path)
        metric = ArmorMaskMetric(binder, output_path=tmp_path / "out.parquet")

        df = pd.read_parquet(metric.scan())

        assert df.iloc[0]["label"] == "OTHER"


@pytest.mark.parametrize(
    "metric_cls,expected_path",
    [
        (FactionMaskMetric, Path("data/metrics/gwent_one/faction_mask.parquet")),
        (ColorMaskMetric, Path("data/metrics/gwent_one/color_mask.parquet")),
        (TypeMaskMetric, Path("data/metrics/gwent_one/type_mask.parquet")),
        (RarityMaskMetric, Path("data/metrics/gwent_one/rarity_mask.parquet")),
        (SetMaskMetric, Path("data/metrics/gwent_one/set_mask.parquet")),
        (ProvisionMaskMetric, Path("data/metrics/gwent_one/provision_mask.parquet")),
        (PowerMaskMetric, Path("data/metrics/gwent_one/power_mask.parquet")),
        (ArmorMaskMetric, Path("data/metrics/gwent_one/armor_mask.parquet")),
    ],
)
def test_default_output_path(metric_cls, expected_path: Path) -> None:
    assert metric_cls.DEFAULT_OUTPUT_PATH == expected_path
