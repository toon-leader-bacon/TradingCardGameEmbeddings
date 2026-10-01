"""Eligibility and labels of the single-card masking metrics over the
scryfall, pokemon_tcg, cardvault_fabtcg, spire_codex and hearthstonejson
binders, on raw_content shaped like each game's leaned ingestion output."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.metrics.cardvault_fabtcg import card_mask_metrics as fabtcg
from src.data_refinement.metrics.hearthstonejson import (
    card_mask_metrics as hearthstone,
)
from src.data_refinement.metrics.pokemon_tcg import card_mask_metrics as pokemon
from src.data_refinement.metrics.scryfall import card_mask_metrics as scryfall
from src.data_refinement.metrics.spire_codex import card_mask_metrics as sts2
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class _FakeCardLookup:
    def __init__(self, cards: list[GenericCard]) -> None:
        self._cards = cards

    def all_cards(self, source_game: GameId) -> list[GenericCard]:
        return [card for card in self._cards if card.source_game == source_game]

    def version_for(self, source_game: GameId) -> str:
        return "test-version"


def _labels(metric_cls: Any, contents: list[dict], tmp_path: Path) -> list[Any]:
    """Scan one card per raw_content and return the labels in input order
    (None for a card the metric skipped)."""
    cards = [
        GenericCard(
            nocab_uuid=uuid4(),
            source_game=metric_cls.SOURCE_GAME,
            name=f"card {index}",
            raw_content=content,
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id=str(index),
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        for index, content in enumerate(contents)
    ]
    metric = metric_cls(_FakeCardLookup(cards), output_path=tmp_path / "out.parquet")
    df = pd.read_parquet(metric.scan())
    by_uuid = dict(zip(df["nocab_uuid"], df["label"]))
    labels = [by_uuid.get(str(card.nocab_uuid)) for card in cards]
    return [
        (
            list(label)
            if label is not None and not isinstance(label, (str, float))
            else label
        )
        for label in labels
    ]


class TestScryfall:
    def test_cmc_needs_a_mana_cost_one_face_and_a_sane_value(
        self, tmp_path: Path
    ) -> None:
        labels = _labels(
            scryfall.CmcRegressionMetric,
            [
                {"cmc": 3, "mana_cost": "{2}{G}"},
                {"cmc": 0, "type_line": "Land"},
                {"cmc": 4, "mana_cost": "{1}{R} // {2}", "card_faces": [{}]},
                {"cmc": 1000000, "mana_cost": "{1000000}"},
            ],
            tmp_path,
        )
        assert labels == [3.0, None, None, None]

    def test_card_type_follows_priority(self, tmp_path: Path) -> None:
        labels = _labels(
            scryfall.CardTypeMaskMetric,
            [
                {"type_line": "Artifact Creature — Golem"},
                {"type_line": "Artifact Land"},
                {"type_line": "Legendary Enchantment"},
                {"type_line": "Battle — Siege"},
            ],
            tmp_path,
        )
        assert labels == ["Creature", "Land", "Enchantment", "OTHER"]

    def test_rarity_folds_rare_values_into_other(self, tmp_path: Path) -> None:
        labels = _labels(
            scryfall.RarityMaskMetric,
            [{"rarity": "mythic"}, {"rarity": "special"}],
            tmp_path,
        )
        assert labels == ["mythic", "OTHER"]

    def test_colors_are_multi_label_and_colorless_is_empty(
        self, tmp_path: Path
    ) -> None:
        labels = _labels(
            scryfall.ColorsMaskMetric,
            [{"colors": ["G", "U"]}, {"name": "Sol Ring"}],
            tmp_path,
        )
        assert labels == [["U", "G"], []]

    @pytest.mark.parametrize(
        "metric_cls,key",
        [
            (scryfall.PowerRegressionMetric, "power"),
            (scryfall.ToughnessRegressionMetric, "toughness"),
        ],
    )
    def test_power_and_toughness_take_whole_numbers_only(
        self, metric_cls: Any, key: str, tmp_path: Path
    ) -> None:
        labels = _labels(
            metric_cls,
            [{key: "3"}, {key: "*"}, {key: "99"}, {"name": "Shock"}],
            tmp_path,
        )
        assert labels == [3.0, None, None, None]


class TestPokemon:
    _POKEMON = {"supertype": "Pokémon"}

    def test_hp_covers_pokemon_only(self, tmp_path: Path) -> None:
        labels = _labels(
            pokemon.HpRegressionMetric,
            [{**self._POKEMON, "hp": "120"}, {"supertype": "Trainer", "hp": "60"}],
            tmp_path,
        )
        assert labels == [120.0, None]

    def test_types_are_multi_label(self, tmp_path: Path) -> None:
        labels = _labels(
            pokemon.TypesMaskMetric,
            [{**self._POKEMON, "types": ["Darkness", "Grass"]}],
            tmp_path,
        )
        assert labels == [["Grass", "Darkness"]]

    def test_stage_reads_the_ladder_out_of_subtypes(self, tmp_path: Path) -> None:
        labels = _labels(
            pokemon.StageMaskMetric,
            [
                {**self._POKEMON, "subtypes": ["Stage 2", "ex"]},
                {**self._POKEMON, "subtypes": ["VMAX"]},
            ],
            tmp_path,
        )
        assert labels == ["Stage 2", "OTHER"]

    def test_retreat_cost_skips_pokemon_without_it(self, tmp_path: Path) -> None:
        labels = _labels(
            pokemon.RetreatCostRegressionMetric,
            [{**self._POKEMON, "convertedRetreatCost": 2}, dict(self._POKEMON)],
            tmp_path,
        )
        assert labels == [2.0, None]

    def test_weakness_is_the_first_weakness_type(self, tmp_path: Path) -> None:
        labels = _labels(
            pokemon.WeaknessMaskMetric,
            [
                {
                    **self._POKEMON,
                    "weaknesses": [
                        {"type": "Fire", "value": "×2"},
                        {"type": "Water", "value": "×2"},
                    ],
                },
                dict(self._POKEMON),
            ],
            tmp_path,
        )
        assert labels == ["Fire", None]


class TestFleshAndBlood:
    def test_pitch_takes_one_to_three(self, tmp_path: Path) -> None:
        labels = _labels(
            fabtcg.PitchMaskMetric,
            [{"pitch": "2"}, {"pitch": "4"}, {"name": "Hero"}],
            tmp_path,
        )
        assert labels == ["2", None, None]

    @pytest.mark.parametrize(
        "metric_cls,key",
        [
            (fabtcg.CostRegressionMetric, "cost"),
            (fabtcg.PowerRegressionMetric, "power"),
            (fabtcg.DefenseRegressionMetric, "defense"),
        ],
    )
    def test_numeric_stats_take_whole_numbers_only(
        self, metric_cls: Any, key: str, tmp_path: Path
    ) -> None:
        labels = _labels(metric_cls, [{key: "3"}, {key: "X"}, {key: "99"}], tmp_path)
        assert labels == [3.0, None, None]

    def test_two_faced_cards_are_skipped(self, tmp_path: Path) -> None:
        back = {"typebox": "Warrior Action", "pitch": "1"}
        labels = _labels(
            fabtcg.PitchMaskMetric,
            [{"pitch": "1", "back_face": back}, {"pitch": "1"}],
            tmp_path,
        )
        assert labels == [None, "1"]

    def test_class_reads_the_typebox(self, tmp_path: Path) -> None:
        labels = _labels(
            fabtcg.ClassMaskMetric,
            [
                {"typebox": "Light Warrior Action - Attack"},
                {"typebox": "Shadow Action"},
                {"typebox": "Assassin / Ninja Action"},
                {"typebox": "Bard Action"},
                {"name": "no typebox"},
            ],
            tmp_path,
        )
        assert labels == ["Warrior", "Classless", "Multiclass", "OTHER", None]

    def test_card_type_reads_the_typebox(self, tmp_path: Path) -> None:
        labels = _labels(
            fabtcg.CardTypeMaskMetric,
            [
                {"typebox": "Warrior Attack Reaction"},
                {"typebox": "Ninja Action - Attack"},
                {"typebox": "Generic Equipment - Head"},
                {"typebox": "Event"},
            ],
            tmp_path,
        )
        assert labels == ["Attack Reaction", "Action", "Equipment", "OTHER"]


class TestSlayTheSpire2:
    def test_cost_skips_unplayable_and_x_cost_cards(self, tmp_path: Path) -> None:
        labels = _labels(
            sts2.CostMaskMetric,
            [{"cost": 1}, {"cost": 5}, {"cost": -1}, {"cost": 0, "is_x_cost": True}],
            tmp_path,
        )
        assert labels == ["1", "OTHER", None, None]

    @pytest.mark.parametrize(
        "metric_cls,key,kept,skipped",
        [
            (sts2.CardTypeMaskMetric, "type", "Power", "Curse"),
            (sts2.RarityMaskMetric, "rarity", "Basic", "Event"),
            (sts2.ColorMaskMetric, "color", "regent", "status"),
        ],
    )
    def test_categories_are_skipped(
        self, metric_cls: Any, key: str, kept: str, skipped: str, tmp_path: Path
    ) -> None:
        labels = _labels(metric_cls, [{key: kept}, {key: skipped}], tmp_path)
        assert labels == [kept, None]


class TestHearthstone:
    def test_cost_skips_oddities(self, tmp_path: Path) -> None:
        labels = _labels(
            hearthstone.CostRegressionMetric, [{"cost": 3}, {"cost": 100}], tmp_path
        )
        assert labels == [3.0, None]

    @pytest.mark.parametrize(
        "metric_cls,key",
        [
            (hearthstone.AttackRegressionMetric, "attack"),
            (hearthstone.HealthRegressionMetric, "health"),
        ],
    )
    def test_stats_cover_minions_only(
        self, metric_cls: Any, key: str, tmp_path: Path
    ) -> None:
        labels = _labels(
            metric_cls,
            [{"type": "MINION", key: 4}, {"type": "HERO", key: 30}],
            tmp_path,
        )
        assert labels == [4.0, None]

    def test_class_marks_multiclass_cards(self, tmp_path: Path) -> None:
        labels = _labels(
            hearthstone.ClassMaskMetric,
            [
                {"cardClass": "MAGE"},
                {"cardClass": "NEUTRAL", "classes": ["MAGE", "PRIEST"]},
                {"name": "classless"},
            ],
            tmp_path,
        )
        assert labels == ["MAGE", "MULTICLASS", None]

    def test_races_expand_all_and_allow_none(self, tmp_path: Path) -> None:
        labels = _labels(
            hearthstone.RacesMaskMetric,
            [
                {"type": "MINION", "races": ["MURLOC", "BEAST"]},
                {"type": "MINION", "races": ["ALL"]},
                {"type": "MINION"},
                {"type": "SPELL"},
            ],
            tmp_path,
        )
        assert labels[0] == ["BEAST", "MURLOC"]
        assert len(labels[1]) == 12
        assert labels[2:] == [[], None]

    def test_spell_school_labels_schoolless_spells_none(self, tmp_path: Path) -> None:
        labels = _labels(
            hearthstone.SpellSchoolMaskMetric,
            [
                {"type": "SPELL", "spellSchool": "FIRE"},
                {"type": "SPELL"},
                {"type": "MINION"},
            ],
            tmp_path,
        )
        assert labels == ["FIRE", "NONE", None]
