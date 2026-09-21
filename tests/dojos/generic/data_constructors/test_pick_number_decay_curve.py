from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import PickNumberDecayCurveDataConstructor
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str, source_id: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestPickNumberDecayCurveDataConstructorBuild:
    def test_buckets_at_or_above_threshold_are_included(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "take_rate_by_pick_number": [[0.2, 0.9]],
                "sample_count_by_pick_number": [[10, 5]],
            }
        )
        constructor = PickNumberDecayCurveDataConstructor(min_sample_count=10)

        result = constructor.build(chunk, binder)

        # bucket 0 clears the threshold (10 >= 10); bucket 1 (5 samples)
        # does not, so it's absent from the dict entirely, not included
        # with its own (unreliable) take_rate.
        assert result == [(card, {0: 0.2})]

    def test_unobserved_bucket_is_excluded_same_as_below_threshold(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "take_rate_by_pick_number": [[0.2, None]],
                "sample_count_by_pick_number": [[10, 0]],
            }
        )
        constructor = PickNumberDecayCurveDataConstructor(min_sample_count=10)

        result = constructor.build(chunk, binder)

        assert result == [(card, {0: 0.2})]

    def test_skips_row_when_every_bucket_is_below_threshold(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "take_rate_by_pick_number": [[0.2, 0.9]],
                "sample_count_by_pick_number": [[3, 5]],
            }
        )
        constructor = PickNumberDecayCurveDataConstructor(min_sample_count=10)

        result = constructor.build(chunk, binder)

        assert result == []

    def test_skips_row_with_unresolvable_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(uuid4())],
                "take_rate_by_pick_number": [[0.2]],
                "sample_count_by_pick_number": [[10]],
            }
        )
        constructor = PickNumberDecayCurveDataConstructor(min_sample_count=10)

        result = constructor.build(chunk, binder)

        assert result == []

    def test_empty_chunk_returns_empty_list(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [],
                "take_rate_by_pick_number": [],
                "sample_count_by_pick_number": [],
            }
        )
        constructor = PickNumberDecayCurveDataConstructor(min_sample_count=10)

        result = constructor.build(chunk, binder)

        assert result == []

    def test_raises_on_min_sample_count_below_one(self) -> None:
        with pytest.raises(ValueError):
            PickNumberDecayCurveDataConstructor(min_sample_count=0)
