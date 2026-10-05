"""Tests for deck_occurrence_count_metric.py's DeckOccurrenceCountMetric,
driven through scan_game_csv as a real run drives it."""

from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.deck_ids import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.game_data.deck_occurrence_count_metric import (  # noqa: E501
    DeckOccurrenceCountMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import scan_game_csv
from src.data_refinement.metrics.version_metadata import read_version_metadata
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

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR])


def _scan(tmp_path: Path, rows: list[dict], block_size: int = 1 << 20) -> pd.DataFrame:
    """Scan rows through the metric; its output indexed by deck_uuid."""
    metric = DeckOccurrenceCountMetric(VERSION, output_path=tmp_path / "out.parquet")
    csv_path = write_csv(tmp_path / "games.csv", rows)
    scan_game_csv(csv_path, [metric], parser_for(_BINDER), block_size=block_size)
    return pd.read_parquet(metric.finalize()).set_index("deck_uuid")


def test_multiple_games_of_one_draft_count_once(tmp_path: Path) -> None:
    df = _scan(
        tmp_path,
        [
            row(won=True, owlbear_deck=4, draft_id="d1", game_number=1),
            row(won=False, owlbear_deck=4, draft_id="d1", game_number=2),
            row(won=True, owlbear_deck=4, draft_id="d1", game_number=3),
        ],
    )

    deck_uuid = str(deck_uuid_from_cards([uuid_for(_BINDER, OWLBEAR)] * 4))
    assert list(df.index) == [deck_uuid]
    assert df.loc[deck_uuid, "occurrence_count"] == 1


def test_two_drafts_with_the_same_decklist_count_as_two(tmp_path: Path) -> None:
    df = _scan(
        tmp_path,
        [
            row(won=True, owlbear_deck=4, draft_id="d1", game_number=1),
            row(won=False, owlbear_deck=4, draft_id="d1", game_number=2),
            row(won=True, owlbear_deck=4, draft_id="d2", game_number=1),
        ],
    )

    deck_uuid = str(deck_uuid_from_cards([uuid_for(_BINDER, OWLBEAR)] * 4))
    assert list(df.index) == [deck_uuid]
    assert df.loc[deck_uuid, "occurrence_count"] == 2


def test_two_drafts_with_different_decklists_get_their_own_rows(
    tmp_path: Path,
) -> None:
    df = _scan(
        tmp_path,
        [
            row(won=True, owlbear_deck=4, draft_id="d1"),
            row(won=True, owlbear_deck=2, draft_id="d2"),
        ],
    )

    owlbear_x4 = str(deck_uuid_from_cards([uuid_for(_BINDER, OWLBEAR)] * 4))
    owlbear_x2 = str(deck_uuid_from_cards([uuid_for(_BINDER, OWLBEAR)] * 2))
    assert sorted(df.index) == sorted([owlbear_x4, owlbear_x2])
    assert df.loc[owlbear_x4, "occurrence_count"] == 1
    assert df.loc[owlbear_x2, "occurrence_count"] == 1


def test_a_deck_spanning_chunks_tallies_like_one_chunk(tmp_path: Path) -> None:
    rows = [
        row(
            won=i % 3 == 0,
            owlbear_deck=4,
            draft_id=f"d{i % 7}",
            game_number=(i // 7) + 1,
        )
        for i in range(30)
    ]

    one = _scan(tmp_path, rows)
    many = _scan(tmp_path, rows, block_size=256)

    assert len(one) == 1
    assert one.iloc[0]["occurrence_count"] == 7
    pd.testing.assert_frame_equal(one.sort_index(), many.sort_index())


def test_identical_decks_across_games_share_one_deck_uuid(
    tmp_path: Path,
) -> None:
    df = _scan(
        tmp_path,
        [
            row(won=True, owlbear_deck=4, draft_id="d1", game_number=1),
            row(won=False, owlbear_deck=4, draft_id="d1", game_number=2),
        ],
    )

    assert len(df) == 1


def test_requires_deck_box_is_stamped_on_the_output(tmp_path: Path) -> None:
    metric = DeckOccurrenceCountMetric(VERSION, output_path=tmp_path / "out.parquet")
    csv_path = write_csv(tmp_path / "games.csv", [row(won=True, owlbear_deck=4)])
    scan_game_csv(csv_path, [metric], parser_for(_BINDER))
    output_path = metric.finalize()

    metadata = read_version_metadata(output_path)
    assert metadata is not None
    assert metadata.requires_deck_box is True
