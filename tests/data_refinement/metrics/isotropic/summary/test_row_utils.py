import logging
from pathlib import Path

import pytest

from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    deck_card_uuids,
    deck_for_player,
    eligible_player_entries,
    is_natural_kingdom,
    kingdom_card_names,
    winner_entry,
)
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def test_card_uuid_for_name_resolves(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    assert card_uuid_for_name(binder, "Witch") == card_uuid(binder, "Witch")


def test_card_uuid_for_name_unresolved_returns_none(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    assert card_uuid_for_name(binder, "Not A Card") is None


def test_winner_entry_finds_rank_one_with_end() -> None:
    row = summary_row(
        ["Witch"],
        [
            player_entry("loser", 2, {"Copper": 7}),
            player_entry("winner", 1, {"Copper": 7, "Witch": 1}),
        ],
    )
    winner = winner_entry(row)
    assert winner is not None
    assert winner["nick"] == "winner"


def test_winner_entry_none_when_rank_one_player_resigned() -> None:
    row = summary_row(["Witch"], [player_entry("solo", 1, resigned=True)])
    assert winner_entry(row) is None


def test_winner_entry_none_when_no_rank_one() -> None:
    row = summary_row(["Witch"], [player_entry("only", 2, {"Copper": 7})])
    assert winner_entry(row) is None


def test_kingdom_card_names_returns_supply() -> None:
    row = summary_row(["Witch", "Village"], [])
    assert kingdom_card_names(row) == ["Witch", "Village"]


def test_is_natural_kingdom_true_by_default() -> None:
    row = summary_row(["Witch"], [])
    assert is_natural_kingdom(row) is True


@pytest.mark.parametrize(
    "flag",
    [
        "constraints",
        "required",
        "prohibited",
        "force_big_cards",
        "force_equal_start",
        "force_banes",
    ],
)
def test_is_natural_kingdom_false_when_constraint_flag_set(flag: str) -> None:
    row = summary_row(["Witch"], [])
    row["board"][flag] = True
    assert is_natural_kingdom(row) is False


def test_is_natural_kingdom_ignores_non_constraint_flags() -> None:
    row = summary_row(["Witch"], [])
    row["board"]["point_tracker"] = True
    row["board"]["black_market"] = True
    row["board"]["bane"] = "Moat"
    assert is_natural_kingdom(row) is True


def test_eligible_player_entries_excludes_resigned() -> None:
    row = summary_row(
        ["Witch"],
        [
            player_entry("finished", 1, {"Copper": 7}),
            player_entry("quit", 2, resigned=True),
        ],
    )
    eligible = eligible_player_entries(row)
    assert [p["nick"] for p in eligible] == ["finished"]


def test_deck_card_uuids_expands_copies(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    uuids = deck_card_uuids(binder, {"Witch": 3})
    assert uuids == [card_uuid(binder, "Witch")] * 3


def test_deck_card_uuids_skips_and_logs_unresolved(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    with caplog.at_level(logging.ERROR):
        uuids = deck_card_uuids(binder, {"Witch": 1, "Not A Card": 2})
    assert uuids == [card_uuid(binder, "Witch")]
    assert any("Not A Card" in record.getMessage() for record in caplog.records)


def test_deck_for_player_hashes_content(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck = deck_for_player(binder, player_entry("nick", 1, {"Witch": 2, "Copper": 5}))
    assert set(deck.card_nocab_uuids) == {card_uuid(binder, "Witch")}
    assert deck.card_nocab_uuids.count(card_uuid(binder, "Witch")) == 2
    assert deck.provenance is None


def test_deck_for_player_same_content_same_uuid(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    deck_a = deck_for_player(binder, player_entry("a", 1, {"Witch": 1}))
    deck_b = deck_for_player(binder, player_entry("b", 2, {"Witch": 1}))
    assert deck_a.nocab_uuid == deck_b.nocab_uuid
