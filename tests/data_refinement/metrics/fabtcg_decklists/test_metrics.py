import math
from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.metrics.fabtcg_decklists.card_inclusion_metrics import (
    CardInclusionRateMetric,
    HeroConditionedInclusionMetric,
)
from src.data_refinement.metrics.fabtcg_decklists.hero_labels import HERO_NAMES
from src.data_refinement.metrics.fabtcg_decklists.hero_masked_from_deck_metric import (
    HeroMaskedFromDeckMetric,
)
from src.data_refinement.metrics.fabtcg_decklists.scanner import scan_decklist_files
from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.version_metadata import read_version_metadata
from src.schema.game_id import GameId
from tests.data_refinement.metrics.fabtcg_decklists.fab_fixtures import (
    fab_binder,
    fab_card,
    published_box,
)

# Two labeled heroes (HERO_NAMES members) and one rare one
WARRIOR = fab_card("Dorinthea Ironsong", "Warrior Hero")
NINJA = fab_card("Katsu, the Wanderer", "Ninja Hero")
RARE = fab_card("Hala, Bladesaint of the Vow", "Warrior Hero")
GENERIC = fab_card("Sink Below", "Generic Defense Reaction")
WARRIOR_ATTACK = fab_card("Sharpen Steel", "Warrior Action")
BINDER = fab_binder([WARRIOR, NINJA, RARE, GENERIC, WARRIOR_ATTACK])


def _box_of_21_decks():
    """10 warrior decks (all with the attack, 5 with Sink Below), 10 ninja
    decks (all with Sink Below), 1 rare-hero deck (with the attack)."""
    decks = {
        f"w{i}": [WARRIOR, WARRIOR_ATTACK, *([GENERIC] * (i < 5))] for i in range(10)
    }
    decks.update({f"n{i}": [NINJA, GENERIC] for i in range(10)})
    decks["rare"] = [RARE, WARRIOR_ATTACK]
    return published_box(decks)


def _scan(metric_rows: list, slugs: list[str]) -> None:
    for slug in slugs:
        for metric in metric_rows:
            metric.accumulate({"slug": slug})
    for metric in metric_rows:
        metric.finalize()


SLUGS = [*(f"w{i}" for i in range(10)), *(f"n{i}" for i in range(10)), "rare"]


class TestHeroMaskedFromDeckMetric:
    def test_one_row_per_single_hero_deck_with_hero_label(self, tmp_path: Path) -> None:
        box = _box_of_21_decks()
        out = tmp_path / "hero.parquet"

        _scan([HeroMaskedFromDeckMetric(BINDER, box, out)], [*SLUGS, "missing"])

        df = pd.read_parquet(out)
        assert len(df) == 21
        labels = dict(zip(df.target_card_uuid, df.label))
        assert labels[str(WARRIOR.nocab_uuid)] == WARRIOR.name
        assert labels[str(RARE.nocab_uuid)] == OTHER_LABEL
        assert set(df.deck_uuid) == {str(uuid) for uuid in box.all_uuids()}

    def test_label_values_are_hero_names_then_other(self) -> None:
        assert HeroMaskedFromDeckMetric.LABEL_VALUES == (*HERO_NAMES, OTHER_LABEL)
        assert {WARRIOR.name, NINJA.name} <= set(HERO_NAMES)
        assert RARE.name not in HERO_NAMES

    def test_output_requires_the_deck_box(self, tmp_path: Path) -> None:
        out = tmp_path / "hero.parquet"
        _scan([HeroMaskedFromDeckMetric(BINDER, _box_of_21_decks(), out)], SLUGS)

        metadata = read_version_metadata(out)

        assert metadata is not None and metadata.requires_deck_box


class TestCardInclusionRateMetric:
    def test_rate_over_legal_decks_only(self, tmp_path: Path) -> None:
        out = tmp_path / "rate.parquet"

        _scan([CardInclusionRateMetric(BINDER, _box_of_21_decks(), out)], SLUGS)

        df = pd.read_parquet(out).set_index("nocab_uuid")
        # Sink Below: legal in all 21, held by 5 + 10
        generic = df.loc[str(GENERIC.nocab_uuid)]
        assert generic.legal_deck_count == 21
        assert generic.inclusion_rate == pytest.approx(15 / 21)
        # The warrior attack is illegal for the ninja: 11 legal decks < 20
        assert str(WARRIOR_ATTACK.nocab_uuid) not in df.index
        # Heroes are never rows
        assert str(WARRIOR.nocab_uuid) not in df.index

    def test_stamps_the_binder_version(self, tmp_path: Path) -> None:
        out = tmp_path / "rate.parquet"
        _scan([CardInclusionRateMetric(BINDER, _box_of_21_decks(), out)], SLUGS)

        metadata = read_version_metadata(out)

        assert metadata is not None
        assert metadata.card_binder_version == BINDER.version_for(
            GameId.FLESH_AND_BLOOD
        )
        assert not metadata.requires_deck_box


class TestHeroConditionedInclusionMetric:
    def test_rates_per_hero_name_with_illegal_positions_zeroed(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "by_hero.parquet"

        _scan([HeroConditionedInclusionMetric(BINDER, _box_of_21_decks(), out)], SLUGS)

        df = pd.read_parquet(out).set_index("nocab_uuid")
        warrior_at = HERO_NAMES.index(WARRIOR.name)
        ninja_at = HERO_NAMES.index(NINJA.name)
        attack = df.loc[str(WARRIOR_ATTACK.nocab_uuid)]
        assert attack.legal_deck_count_by_hero[warrior_at] == 10
        assert attack.legal_deck_count_by_hero[ninja_at] == 0
        assert attack.inclusion_rate_by_hero[warrior_at] == pytest.approx(1.0)
        assert math.isnan(attack.inclusion_rate_by_hero[ninja_at])
        generic = df.loc[str(GENERIC.nocab_uuid)]
        assert generic.inclusion_rate_by_hero[warrior_at] == pytest.approx(0.5)
        assert generic.inclusion_rate_by_hero[ninja_at] == pytest.approx(1.0)
        assert sum(generic.legal_deck_count_by_hero) == 20  # the rare hero has no slot


class TestScanDecklistFiles:
    def test_feeds_one_slug_row_per_html_file_then_finalizes(
        self, tmp_path: Path
    ) -> None:
        raw_dir = tmp_path / "decklists"
        raw_dir.mkdir()
        for slug in ("b-deck", "a-deck"):
            (raw_dir / f"{slug}.html").write_text("<section/>", encoding="utf-8")
        (raw_dir / "notes.txt").write_text("", encoding="utf-8")
        recorder = _RecordingMetric()

        scan_decklist_files(raw_dir, [recorder])

        assert recorder.rows == [{"slug": "a-deck"}, {"slug": "b-deck"}]
        assert recorder.finalized

    def test_one_failing_metric_does_not_stop_the_others(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "decklists"
        raw_dir.mkdir()
        (raw_dir / "a.html").write_text("", encoding="utf-8")
        recorder = _RecordingMetric()

        scan_decklist_files(raw_dir, [_FailingMetric(), recorder])

        assert recorder.rows == [{"slug": "a"}]
        assert recorder.finalized

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan_decklist_files(tmp_path / "missing", [])


class _RecordingMetric:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.finalized = False

    def accumulate(self, row: dict) -> None:
        self.rows.append(row)

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused")


class _FailingMetric:
    def accumulate(self, row: dict) -> None:
        raise ValueError("boom")

    def finalize(self) -> Path:
        raise ValueError("boom")
