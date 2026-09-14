"""Skeleton-stage test stubs for game_deck_label_metric.py's
GameDeckLabelMetric, covered through its three
game_deck_label_metrics.py concretes (DeckWinPredictionMetric,
DeckGameLengthPredictionMetric, DeckRankTierPredictionMetric) - the
same "cover the shared base through its concretes" convention
tests/data_refinement/metrics/seventeenlands/draft_data/test_pack_card_tally_metrics.py
uses for PackCardTallyMetric. Bodies are filled in by
design-recipe-implement.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _make_card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder_with_cards(names: list[str]) -> CardBinder:
    binder = CardBinder()
    for name in names:
        binder.create(_make_card(name))
    return binder


_HEADER = ["draft_id", "match_number", "game_number", "won", "num_turns", "rank"]
_DECK_HEADER = _HEADER + ["deck_Owlbear"]


def _row(
    won: bool = True,
    num_turns: int = 8,
    rank: str = "gold",
    owlbear_deck: int = 4,
    draft_id: str = "draft1",
    match_number: int = 0,
    game_number: int = 0,
) -> dict:
    return {
        "draft_id": draft_id,
        "match_number": match_number,
        "game_number": game_number,
        "won": won,
        "num_turns": num_turns,
        "rank": rank,
        "deck_Owlbear": owlbear_deck,
    }


class TestDeckWinPredictionMetric:
    def test_writes_one_row_per_game_with_deck_uuid_and_won_label(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = DeckWinPredictionMetric(
            binder,
            _DECK_HEADER,
            GameId.MTG,
            DeckBox(),
            output_path=tmp_path / "out.parquet",
        )

        metric.accumulate(_row(won=True))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        rows = table.to_pylist()
        assert len(rows) == 1
        assert rows[0]["won"] is True
        assert rows[0]["deck_uuid"] is not None

    def test_identical_decks_across_games_dedupe_in_the_shared_deck_box(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        deck_box = DeckBox()
        metric = DeckWinPredictionMetric(
            binder,
            _DECK_HEADER,
            GameId.MTG,
            deck_box,
            output_path=tmp_path / "out.parquet",
        )

        metric.accumulate(_row(owlbear_deck=4, match_number=0, game_number=0))
        metric.accumulate(_row(owlbear_deck=4, match_number=0, game_number=1))
        metric.finalize()

        assert len(list(deck_box.all_uuids(GameId.MTG))) == 1

    def test_output_row_carries_draft_id_match_number_game_number(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = DeckWinPredictionMetric(
            binder,
            _DECK_HEADER,
            GameId.MTG,
            DeckBox(),
            output_path=tmp_path / "out.parquet",
        )

        metric.accumulate(_row(draft_id="draft42", match_number=3, game_number=1))
        metric.finalize()

        row = pq.read_table(tmp_path / "out.parquet").to_pylist()[0]
        assert row["draft_id"] == "draft42"
        assert row["match_number"] == 3
        assert row["game_number"] == 1


class TestDeckGameLengthPredictionMetric:
    def test_writes_one_row_per_game_with_num_turns_label(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = DeckGameLengthPredictionMetric(
            binder,
            _DECK_HEADER,
            GameId.MTG,
            DeckBox(),
            output_path=tmp_path / "out.parquet",
        )

        metric.accumulate(_row(num_turns=11))
        metric.finalize()

        row = pq.read_table(tmp_path / "out.parquet").to_pylist()[0]
        assert row["num_turns"] == 11


class TestDeckRankTierPredictionMetric:
    def test_known_rank_tier_is_written_as_is(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = DeckRankTierPredictionMetric(
            binder,
            _DECK_HEADER,
            GameId.MTG,
            DeckBox(),
            output_path=tmp_path / "out.parquet",
        )

        metric.accumulate(_row(rank="mythic"))
        metric.finalize()

        row = pq.read_table(tmp_path / "out.parquet").to_pylist()[0]
        assert row["rank"] == "mythic"

    def test_unknown_rank_value_falls_back_to_other_label(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Owlbear"])
        metric = DeckRankTierPredictionMetric(
            binder,
            _DECK_HEADER,
            GameId.MTG,
            DeckBox(),
            output_path=tmp_path / "out.parquet",
        )

        metric.accumulate(_row(rank="unranked_prerelease"))
        metric.finalize()

        row = pq.read_table(tmp_path / "out.parquet").to_pylist()[0]
        assert row["rank"] == OTHER_LABEL


def test_finalize_closes_writer_and_is_idempotent(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = DeckWinPredictionMetric(
        binder,
        _DECK_HEADER,
        GameId.MTG,
        DeckBox(),
        output_path=tmp_path / "out.parquet",
    )
    metric.accumulate(_row())

    first = metric.finalize()
    second = metric.finalize()

    assert first == second == tmp_path / "out.parquet"
