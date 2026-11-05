from pathlib import Path
from typing import Any

import pytest

from src.schema.game_id import GameId
from src.schema.holdout import CardTier
from src.schema.splits import Split
from src.teacher.neighbour_rules import CosineThresholdNeighbours, TopKNeighbours
from src.teacher.teacher_settings import (
    DEFAULT_TEACHER_DIRECTORY,
    CooccurrenceOptions,
    CopyCountRule,
    DeckSizeRule,
    PpmiSvdOptions,
)
from src.training.config_parsing.config_section import ConfigSection
from src.training.config_parsing.teacher_parsing import parse_teacher


def _section(**overrides: Any) -> ConfigSection:
    raw: dict[str, Any] = {
        "games": ["mtg", "flesh_and_blood"],
        "method": {"kind": "ppmi_svd", "dim": 128},
        "neighbours": {"kind": "top_k", "k": 32},
    }
    raw.update(overrides)
    return ConfigSection(raw, "teacher")


class TestParseTeacher:
    def test_minimal_section_takes_the_defaults(self) -> None:
        settings = parse_teacher(_section())

        assert settings.games == (GameId.MTG, GameId.FLESH_AND_BLOOD)
        assert settings.method == PpmiSvdOptions(dim=128)
        assert settings.neighbours == TopKNeighbours(32)
        assert settings.deck_splits == frozenset({Split.TRAIN})
        assert settings.card_tiers == frozenset({CardTier.TRAIN})
        assert settings.cooccurrence == CooccurrenceOptions()
        assert settings.min_deck_count == 5
        assert settings.directory == DEFAULT_TEACHER_DIRECTORY

    def test_every_option(self) -> None:
        settings = parse_teacher(
            _section(
                deck_splits=["train", "test"],
                card_tiers=["train", "test"],
                cooccurrence={
                    "copy_count_rule": "log_copies",
                    "deck_size_rule": "inverse_size",
                },
                method={"kind": "ppmi_svd", "dim": 64, "context_smoothing": 1.0},
                neighbours={
                    "kind": "cosine_threshold",
                    "min_cosine": 0.4,
                    "max_neighbours": 64,
                },
                min_deck_count=2,
                directory="elsewhere",
            )
        )

        assert settings.deck_splits == frozenset({Split.TRAIN, Split.TEST})
        assert settings.card_tiers == frozenset({CardTier.TRAIN, CardTier.TEST})
        assert settings.cooccurrence == CooccurrenceOptions(
            CopyCountRule.LOG_COPIES, DeckSizeRule.INVERSE_SIZE
        )
        assert settings.method == PpmiSvdOptions(dim=64, context_smoothing=1.0)
        assert settings.neighbours == CosineThresholdNeighbours(0.4, 64)
        assert settings.min_deck_count == 2
        assert settings.directory == Path("elsewhere")

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"method": {"kind": "word2vec", "dim": 8}}, "word2vec"),
            ({"neighbours": {"kind": "top_k"}}, "k"),
            (
                {"neighbours": {"kind": "top_k", "k": 3, "min_cosine": 0.1}},
                "min_cosine",
            ),
            ({"games": ["not_a_game"]}, "not_a_game"),
            ({"card_tiers": ["middle"]}, "middle"),
            ({"cooccurrence": {"copy_count_rule": "squared"}}, "squared"),
            ({"typo": 1}, "typo"),
            ({"games": []}, "empty"),
        ],
    )
    def test_rejects_bad_sections(self, overrides: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_teacher(_section(**overrides))
