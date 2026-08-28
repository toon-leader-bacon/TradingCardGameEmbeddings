from uuid import uuid4

from src.data_refinement.seventeenlands.lookup_cache import LookupCache


class TestGet:
    def test_resolves_via_lookup_callback(self) -> None:
        uuid = uuid4()
        cache = LookupCache(lambda key: uuid if key == "hit" else None)

        assert cache.get("hit") == uuid

    def test_unresolvable_key_returns_none(self) -> None:
        cache = LookupCache(lambda key: None)

        assert cache.get("miss") is None

    def test_repeated_key_does_not_call_lookup_again(self) -> None:
        uuid = uuid4()
        calls = []

        def lookup(key: str):
            calls.append(key)
            return uuid

        cache = LookupCache(lookup)

        assert cache.get("a") == uuid
        assert cache.get("a") == uuid
        assert cache.get("a") == uuid
        assert calls == ["a"]

    def test_cached_miss_does_not_call_lookup_again(self) -> None:
        calls = []

        def lookup(key: str):
            calls.append(key)
            return None

        cache = LookupCache(lookup)

        assert cache.get("miss") is None
        assert cache.get("miss") is None
        assert calls == ["miss"]

    def test_distinct_keys_are_resolved_independently(self) -> None:
        uuid_a = uuid4()
        uuid_b = uuid4()
        cache = LookupCache(lambda key: {"a": uuid_a, "b": uuid_b}.get(key))

        assert cache.get("a") == uuid_a
        assert cache.get("b") == uuid_b


class TestUnresolvedKeys:
    def test_empty_when_nothing_resolved_yet(self) -> None:
        cache = LookupCache(lambda key: None)

        assert cache.unresolved_keys == frozenset()

    def test_tracks_only_unresolvable_keys(self) -> None:
        uuid = uuid4()
        cache = LookupCache(lambda key: uuid if key == "hit" else None)

        cache.get("hit")
        cache.get("miss1")
        cache.get("miss2")

        assert cache.unresolved_keys == frozenset({"miss1", "miss2"})

    def test_is_cumulative_across_calls(self) -> None:
        cache = LookupCache(lambda key: None)

        cache.get("a")
        assert cache.unresolved_keys == frozenset({"a"})
        cache.get("b")
        assert cache.unresolved_keys == frozenset({"a", "b"})
