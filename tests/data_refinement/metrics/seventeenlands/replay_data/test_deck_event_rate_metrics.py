"""Tests for the DeckEventRateMetric concretes (deck_event_rate_metric.py):
CastRateMetric, DiscardRateMetric and TutorTargetRateMetric share one
rule - per deck card, games the user's FIELDS named it over games in the
deck - so each case runs against all three."""

from pathlib import Path

import pytest

from src.data_refinement.metrics.seventeenlands.replay_data.cast_rate_metric import (
    CastRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.deck_event_rate_metric import (
    DeckEventRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.discard_rate_metric import (
    DiscardRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    CAST_FIELDS,
    Actor,
    ReplayField,
)
from src.data_refinement.metrics.seventeenlands.replay_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from tests.data_refinement.metrics.seventeenlands.replay_data._replay_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    OWLBEAR_ID,
    VERSION,
    binder_with_cards,
    field_column,
    row,
    scan_into_frame,
    uuid_for,
)

_METRICS = [CastRateMetric, DiscardRateMetric, TutorTargetRateMetric]


def _scan(tmp_path: Path, metric_class: type[DeckEventRateMetric], rows):
    binder = binder_with_cards()
    metric = metric_class(VERSION, tmp_path / "out.parquet")
    return binder, scan_into_frame(tmp_path, binder, rows, metric)


def _user_cell(metric_class: type[DeckEventRateMetric], turn: int = 1) -> str:
    return field_column(Actor.USER, turn, metric_class.FIELDS[0])


def test_each_metric_reads_its_own_fields() -> None:
    assert CastRateMetric.FIELDS == CAST_FIELDS
    assert DiscardRateMetric.FIELDS == (ReplayField.CARDS_DISCARDED,)
    assert TutorTargetRateMetric.FIELDS == (ReplayField.CARDS_TUTORED,)


@pytest.mark.parametrize("metric_class", _METRICS)
def test_rate_is_games_named_over_games_in_deck(
    tmp_path: Path, metric_class: type[DeckEventRateMetric]
) -> None:
    named_twice = {
        _user_cell(metric_class, 1): OWLBEAR_ID,
        _user_cell(metric_class, 2): OWLBEAR_ID,
    }
    rows = [row(named_twice, owlbear_deck=1), row(owlbear_deck=2)]

    binder, frame = _scan(tmp_path, metric_class, rows)

    owlbear = str(uuid_for(binder, OWLBEAR))
    assert frame.loc[owlbear, "sample_count"] == 2
    assert frame.loc[owlbear, metric_class.LABEL_COLUMN] == 0.5


@pytest.mark.parametrize("metric_class", _METRICS)
def test_a_deck_card_never_named_gets_zero(
    tmp_path: Path, metric_class: type[DeckEventRateMetric]
) -> None:
    binder, frame = _scan(tmp_path, metric_class, [row(morningstar_deck=1)])

    morningstar = str(uuid_for(binder, MORNINGSTAR))
    assert frame.loc[morningstar, metric_class.LABEL_COLUMN] == 0.0


@pytest.mark.parametrize("metric_class", [CastRateMetric, DiscardRateMetric])
def test_only_the_users_half_turns_count(
    tmp_path: Path, metric_class: type[DeckEventRateMetric]
) -> None:
    cells = {field_column(Actor.OPPO, 1, metric_class.FIELDS[0]): OWLBEAR_ID}

    binder, frame = _scan(tmp_path, metric_class, [row(cells, owlbear_deck=1)])

    assert frame.loc[str(uuid_for(binder, OWLBEAR)), metric_class.LABEL_COLUMN] == 0


@pytest.mark.parametrize("metric_class", _METRICS)
def test_a_named_card_not_in_the_deck_is_not_counted(
    tmp_path: Path, metric_class: type[DeckEventRateMetric]
) -> None:
    rows = [row({_user_cell(metric_class): OWLBEAR_ID}, morningstar_deck=1)]

    binder, frame = _scan(tmp_path, metric_class, rows)

    assert str(uuid_for(binder, OWLBEAR)) not in frame.index


def test_cast_rate_reads_non_creature_casts_too(tmp_path: Path) -> None:
    cells = {field_column(Actor.USER, 1, ReplayField.NON_CREATURES_CAST): OWLBEAR_ID}

    binder, frame = _scan(tmp_path, CastRateMetric, [row(cells, owlbear_deck=1)])

    assert frame.loc[str(uuid_for(binder, OWLBEAR)), "cast_rate"] == 1.0
