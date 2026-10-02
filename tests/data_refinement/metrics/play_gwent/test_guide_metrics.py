import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.play_gwent.extraction_stage import (
    PlayGwentDeckExtractionStage,
)
from src.data_refinement.metrics.play_gwent.card_inclusion_metrics import (
    CardInclusionRateMetric,
    FactionConditionedInclusionMetric,
)
from src.data_refinement.metrics.play_gwent.guide_votes_metric import (
    GuideVotesMetric,
    signed_log_votes,
)
from src.data_refinement.metrics.play_gwent.published_guide_decks import (
    GWENT_FACTIONS,
    PublishedGuideDecks,
    legal_factions,
)
from src.data_refinement.metrics.version_metadata import read_version_metadata
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_NEXT_ID = iter(range(1000, 10_000))


def _card(name: str, faction: str, color: str = "bronze", duo: str = "") -> GenericCard:
    raw_content = {"name": name, "faction": faction, "color": color}
    if duo:
        raw_content["faction-duo"] = duo
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name=name,
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=str(next(_NEXT_ID)),
            fetched_at=datetime.now(timezone.utc),
        ),
    )


MONSTER_LEADER = _card("Blood Scent", "monster", color="leader")
NILF_LEADER = _card("Double Cross", "nilfgaard", color="leader")
NEUTRAL = _card("Tactical Advantage", "neutral")
MONSTER_UNIT = _card("Ekimmara", "monster")
DUO = _card("Tatterwing", "syndicate", duo="syndicate_monster")
CARDS = [MONSTER_LEADER, NILF_LEADER, NEUTRAL, MONSTER_UNIT, DUO]


def _binder() -> CardBinder:
    binder = CardBinder()
    for card in CARDS:
        binder.create(card)
        binder.register_alias(
            GameId.GWENT,
            DataSource.GWENT_ONE,
            card.provenance.source_id,
            card.nocab_uuid,
        )
    return binder


def _guide(
    guide_id: int, leader: GenericCard, cards: list[GenericCard], votes: object
) -> tuple[dict, GenericDeck]:
    """A raw guide row and its published deck (leader included)."""
    row = {"id": guide_id, "leaderId": int(leader.provenance.source_id), "votes": votes}
    deck = GenericDeck(
        nocab_uuid=PlayGwentDeckExtractionStage.deck_uuid_for_guide(guide_id),
        source_game=GameId.GWENT,
        name=f"guide {guide_id}",
        card_nocab_uuids=[leader.nocab_uuid, *(card.nocab_uuid for card in cards)],
    )
    return row, deck


def _guides() -> tuple[list[dict], DeckBox]:
    """20 monster guides (all with the unit, 10 with the neutral), 20
    nilfgaard guides (all with the neutral), votes = guide index - 5."""
    pairs = [
        _guide(i, MONSTER_LEADER, [MONSTER_UNIT, *([NEUTRAL] * (i < 10))], i - 5)
        for i in range(20)
    ]
    pairs += [_guide(100 + i, NILF_LEADER, [NEUTRAL], 0) for i in range(20)]
    box = DeckBox()
    for _, deck in pairs:
        box.create(deck)
    return [row for row, _ in pairs], box


def _run(metric, rows: list[dict]) -> Path:
    for row in rows:
        metric.accumulate(row)
    return metric.finalize()


class TestLegalFactions:
    def test_neutral_fits_every_faction(self) -> None:
        assert legal_factions(NEUTRAL) == frozenset(GWENT_FACTIONS)

    def test_faction_card_fits_its_own(self) -> None:
        assert legal_factions(MONSTER_UNIT) == {"monster"}

    def test_dual_faction_card_fits_both(self) -> None:
        assert legal_factions(DUO) == {"syndicate", "monster"}
        realms = _card("Temple Guard", "syndicate", duo="syndicate_northern_realms")
        assert legal_factions(realms) == {"syndicate", "northern_realms"}


class TestPublishedGuideDecks:
    def test_parses_a_published_guide(self) -> None:
        rows, box = _guides()

        guide_deck = PublishedGuideDecks(box, _binder()).guide_deck_for_row(rows[0])

        assert guide_deck is not None
        assert guide_deck.faction == "monster"
        assert guide_deck.votes == -5
        assert guide_deck.card_uuids == {MONSTER_UNIT.nocab_uuid, NEUTRAL.nocab_uuid}

    @pytest.mark.parametrize(
        "change", [{"id": 99_999}, {"leaderId": 1}, {"votes": None}, {"votes": True}]
    )
    def test_unpublished_unled_or_voteless_guide_is_skipped(self, change: dict) -> None:
        rows, box = _guides()

        guide_decks = PublishedGuideDecks(box, _binder())

        assert guide_decks.guide_deck_for_row({**rows[0], **change}) is None

    def test_reading_never_writes_the_box_file(self, tmp_path: Path) -> None:
        rows, box = _guides()
        box_path = tmp_path / "gwent.db"
        box.save(box_path, GameId.GWENT, "v1")
        before = box_path.read_bytes()
        published = DeckBox.load([box_path])

        _run(GuideVotesMetric(_binder(), published, tmp_path / "votes.parquet"), rows)
        published.flush()

        assert box_path.read_bytes() == before


class TestCardInclusionRateMetric:
    def test_rate_over_faction_legal_decks(self, tmp_path: Path) -> None:
        rows, box = _guides()

        out = _run(
            CardInclusionRateMetric(_binder(), box, tmp_path / "r.parquet"), rows
        )

        df = pd.read_parquet(out).set_index("nocab_uuid")
        neutral = df.loc[str(NEUTRAL.nocab_uuid)]
        assert neutral.legal_deck_count == 40
        assert neutral.inclusion_rate == pytest.approx(30 / 40)
        unit = df.loc[str(MONSTER_UNIT.nocab_uuid)]
        assert unit.legal_deck_count == 20
        assert unit.inclusion_rate == pytest.approx(1.0)
        # Leaders are never rows; a card in no deck is no row
        assert str(MONSTER_LEADER.nocab_uuid) not in df.index
        assert str(DUO.nocab_uuid) not in df.index


class TestFactionConditionedInclusionMetric:
    def test_rate_per_faction_with_illegal_positions_zeroed(
        self, tmp_path: Path
    ) -> None:
        rows, box = _guides()
        metric = FactionConditionedInclusionMetric(
            _binder(), box, tmp_path / "f.parquet"
        )

        df = pd.read_parquet(_run(metric, rows)).set_index("nocab_uuid")

        monster = GWENT_FACTIONS.index("monster")
        nilfgaard = GWENT_FACTIONS.index("nilfgaard")
        neutral = df.loc[str(NEUTRAL.nocab_uuid)]
        assert neutral.inclusion_rate_by_faction[monster] == pytest.approx(0.5)
        assert neutral.inclusion_rate_by_faction[nilfgaard] == pytest.approx(1.0)
        # Factions with no decks have a 0 count even where the card is legal
        assert sum(neutral.legal_deck_count_by_faction) == 40
        unit = df.loc[str(MONSTER_UNIT.nocab_uuid)]
        assert list(unit.legal_deck_count_by_faction) == [20, 0, 0, 0, 0, 0]
        assert math.isnan(unit.inclusion_rate_by_faction[nilfgaard])


class TestGuideVotesMetric:
    def test_one_signed_log_row_per_published_guide(self, tmp_path: Path) -> None:
        rows, box = _guides()

        out = _run(GuideVotesMetric(_binder(), box, tmp_path / "v.parquet"), rows)

        df = pd.read_parquet(out).set_index("guide_id")
        assert len(df) == 40
        assert df.loc[0].signed_log_votes == pytest.approx(-math.log(6))
        assert df.loc[0].deck_uuid == str(
            PlayGwentDeckExtractionStage.deck_uuid_for_guide(0)
        )
        metadata = read_version_metadata(out)
        assert metadata is not None and metadata.requires_deck_box

    @pytest.mark.parametrize(
        "votes, expected", [(0, 0.0), (1, math.log(2)), (-120, -math.log(121))]
    )
    def test_signed_log_votes(self, votes: int, expected: float) -> None:
        assert signed_log_votes(votes) == pytest.approx(expected)
