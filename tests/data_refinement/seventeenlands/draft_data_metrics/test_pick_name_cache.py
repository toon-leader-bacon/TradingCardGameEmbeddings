from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.draft_data_metrics.pick_name_cache import (
    PickNameCache,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
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


class TestGetUuids:
    def test_exact_name_resolution(self) -> None:
        registry = CardBinder()
        bolt = registry.add(_card("Bolt")).stored_card
        cache = PickNameCache(registry, GameId.MTG)

        resolved = cache.get_uuids(pd.Series(["Bolt"]))

        assert resolved.tolist() == [bolt.nocab_uuid]

    def test_repeated_name_uses_cache_not_repeated_registry_queries(self) -> None:
        registry = CardBinder()
        bolt = registry.add(_card("Bolt")).stored_card
        cache = PickNameCache(registry, GameId.MTG)

        resolved = cache.get_uuids(pd.Series(["Bolt", "Bolt", "Bolt"]))

        assert resolved.tolist() == [bolt.nocab_uuid] * 3

        # Poison the registry so a second query would return something
        # different — proves the second and third occurrences were
        # served from the cache, not re-queried.
        registry._uuid_by_name = {}
        resolved_again = cache.get_uuids(pd.Series(["Bolt"]))
        assert resolved_again.tolist() == [bolt.nocab_uuid]

    def test_mdfc_regex_fallback_resolves_single_match(self) -> None:
        registry = CardBinder()
        hulk = registry.add(_card("Bruce Banner // The Incredible Hulk")).stored_card
        cache = PickNameCache(registry, GameId.MTG)

        resolved = cache.get_uuids(pd.Series(["Bruce Banner"]))

        assert resolved.tolist() == [hulk.nocab_uuid]

    def test_ambiguous_regex_fallback_is_unresolved(self) -> None:
        registry = CardBinder()
        registry.add(_card("Bruce Banner // The Incredible Hulk"))
        registry.add(_card("Bruce Banner // Something Else"))
        cache = PickNameCache(registry, GameId.MTG)

        resolved = cache.get_uuids(pd.Series(["Bruce Banner"]))

        assert resolved.tolist() == [None]
        assert cache.unresolved_names == ["Bruce Banner"]

    def test_no_match_at_all_is_unresolved(self) -> None:
        registry = CardBinder()
        cache = PickNameCache(registry, GameId.MTG)

        resolved = cache.get_uuids(pd.Series(["Nonexistent Card"]))

        assert resolved.tolist() == [None]
        assert cache.unresolved_names == ["Nonexistent Card"]

    def test_unresolved_names_accumulate_and_dedupe_across_calls(self) -> None:
        registry = CardBinder()
        cache = PickNameCache(registry, GameId.MTG)

        cache.get_uuids(pd.Series(["Foo", "Bar"]))
        cache.get_uuids(pd.Series(["Foo", "Baz"]))

        assert cache.unresolved_names == ["Bar", "Baz", "Foo"]

    def test_get_uuids_preserves_index_alignment(self) -> None:
        registry = CardBinder()
        bolt = registry.add(_card("Bolt")).stored_card
        cache = PickNameCache(registry, GameId.MTG)
        pick_names = pd.Series(["Bolt", "Missing"], index=[5, 9])

        resolved = cache.get_uuids(pick_names)

        assert resolved.index.tolist() == [5, 9]
        assert resolved.loc[5] == bolt.nocab_uuid
        assert resolved.loc[9] is None
