"""Tests for game_deck_label_metric.py's GameDeckLabelMetric, covered
through its three game_deck_label_metrics.py concretes and driven
through scan_game_csv, as a real run drives them."""

from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.deck_ids import deck_uuid_from_cards
from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import scan_game_csv
from src.data_refinement.metrics.version_metadata import read_version_metadata
from src.schema.game_id import GameId
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    parser_for,
    row,
    uuid_for,
    write_csv,
)


def _scan_rows(
    tmp_path: Path, rows: list[dict], metric_class: type, deck_box: DeckBox
) -> list[dict]:
    """Scan rows through one metric_class instance; its output rows."""
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    metric = metric_class(VERSION, deck_box, output_path=tmp_path / "out.parquet")
    csv_path = write_csv(tmp_path / "games.csv", rows)
    scan_game_csv(csv_path, [metric], parser_for(binder), block_size=256)
    return pq.read_table(tmp_path / "out.parquet").to_pylist()


class TestDeckWinPredictionMetric:
    def test_writes_one_row_per_game_with_deck_uuid_and_won_label(
        self, tmp_path: Path
    ) -> None:
        rows = _scan_rows(
            tmp_path,
            [row(won=True, owlbear_deck=4), row(won=False, owlbear_deck=4)],
            DeckWinPredictionMetric,
            DeckBox(),
        )

        assert [r["won"] for r in rows] == [True, False]
        assert all(r["deck_uuid"] is not None for r in rows)

    def test_deck_uuid_is_the_row_implementations_hash(self, tmp_path: Path) -> None:
        binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
        metric = DeckWinPredictionMetric(
            VERSION, DeckBox(), output_path=tmp_path / "out.parquet"
        )
        csv_path = write_csv(
            tmp_path / "games.csv", [row(won=True, owlbear_deck=4, morningstar_deck=1)]
        )
        scan_game_csv(csv_path, [metric], parser_for(binder))

        (output,) = pq.read_table(metric.finalize()).to_pylist()
        # 4 copies of Owlbear, 1 of Morningstar - the full multiset, not
        # merely which columns are present.
        expected = deck_uuid_from_cards(
            [uuid_for(binder, OWLBEAR)] * 4 + [uuid_for(binder, MORNINGSTAR)]
        )
        assert output["deck_uuid"] == str(expected)

    def test_identical_decks_across_games_dedupe_in_the_shared_deck_box(
        self, tmp_path: Path
    ) -> None:
        deck_box = DeckBox()

        _scan_rows(
            tmp_path,
            [
                row(won=True, owlbear_deck=4, game_number=1),
                row(won=True, owlbear_deck=4, game_number=2),
            ],
            DeckWinPredictionMetric,
            deck_box,
        )

        assert len(list(deck_box.all_uuids(GameId.MTG))) == 1

    def test_a_deck_is_named_after_its_first_game(self, tmp_path: Path) -> None:
        deck_box = DeckBox()

        first, _ = _scan_rows(
            tmp_path,
            [
                row(won=True, owlbear_deck=4, draft_id="d7", match_number=2),
                row(won=True, owlbear_deck=4, draft_id="d8"),
            ],
            DeckWinPredictionMetric,
            deck_box,
        )

        deck = deck_box.get_by_uuid(UUID(first["deck_uuid"]))
        assert deck is not None
        assert deck.name == "game_data d7/2/1 deck"

    def test_output_row_carries_draft_id_match_number_game_number(
        self, tmp_path: Path
    ) -> None:
        (output,) = _scan_rows(
            tmp_path,
            [row(won=True, draft_id="draft42", match_number=3, game_number=1)],
            DeckWinPredictionMetric,
            DeckBox(),
        )

        assert output["draft_id"] == "draft42"
        assert output["match_number"] == 3
        assert output["game_number"] == 1

    def test_output_requires_the_deck_box(self, tmp_path: Path) -> None:
        _scan_rows(tmp_path, [row(won=True)], DeckWinPredictionMetric, DeckBox())

        metadata = read_version_metadata(tmp_path / "out.parquet")
        assert metadata is not None
        assert metadata.requires_deck_box


class TestDeckGameLengthPredictionMetric:
    def test_writes_one_row_per_game_with_num_turns_label(self, tmp_path: Path) -> None:
        (output,) = _scan_rows(
            tmp_path,
            [row(won=True, num_turns=11)],
            DeckGameLengthPredictionMetric,
            DeckBox(),
        )

        assert output["num_turns"] == 11


class TestDeckRankTierPredictionMetric:
    def test_known_rank_tier_is_written_as_is(self, tmp_path: Path) -> None:
        (output,) = _scan_rows(
            tmp_path,
            [row(won=True, rank="mythic")],
            DeckRankTierPredictionMetric,
            DeckBox(),
        )

        assert output["rank"] == "mythic"

    def test_unknown_rank_value_falls_back_to_other_label(self, tmp_path: Path) -> None:
        (output,) = _scan_rows(
            tmp_path,
            [row(won=True, rank="unranked_prerelease")],
            DeckRankTierPredictionMetric,
            DeckBox(),
        )

        assert output["rank"] == OTHER_LABEL

    def test_an_empty_rank_is_other_label(self, tmp_path: Path) -> None:
        # Trad and Sealed events leave rank empty in every row
        rows = _scan_rows(
            tmp_path,
            [row(won=True, rank=""), row(won=True, rank="gold")],
            DeckRankTierPredictionMetric,
            DeckBox(),
        )

        assert [r["rank"] for r in rows] == [OTHER_LABEL, "gold"]


def test_finalize_closes_writer_and_is_idempotent(tmp_path: Path) -> None:
    metric = DeckWinPredictionMetric(
        VERSION, DeckBox(), output_path=tmp_path / "out.parquet"
    )

    first = metric.finalize()
    second = metric.finalize()

    assert first == second == tmp_path / "out.parquet"
    assert pq.read_table(first).num_rows == 0
