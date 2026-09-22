import copy

import pytest

from src.data_refinement.card_binder.lean_content import (
    drop_repeated_strings,
    map_strings,
    order_keys,
    strip_noise,
)


class TestStripNoise:
    def test_drops_url_uuid_and_date_shaped_values_whatever_the_key(self) -> None:
        content = {
            "name": "Bolt",
            "art": "https://cards.example.com/a.jpg",
            "ref": "eca27964-accc-46cc-8aff-a06183e61e9c",
            "when": "2026-09-12",
            "stamp": "2026-09-12T01:55:50Z",
        }

        assert strip_noise(content) == {"name": "Bolt"}

    def test_keeps_text_that_merely_contains_a_url_or_date(self) -> None:
        content = {
            "text": "See https://example.com for rules",
            "note": "Since 2026-09-12",
        }

        assert strip_noise(content) == content

    def test_drops_id_uri_and_url_named_keys_but_not_lookalikes(self) -> None:
        content = {
            "id": 1,
            "card_id": 2,
            "multiverse_ids": [3],
            "scryfall_uri": "x",
            "image_uris": {"small": "y"},
            "thumbnail_url": "z",
            "grid": "keep me",
            "valid": "keep me too",
        }

        assert strip_noise(content) == {"grid": "keep me", "valid": "keep me too"}

    def test_drops_empty_values_and_containers_that_become_empty(self) -> None:
        content = {
            "a": None,
            "b": "",
            "c": [],
            "d": {},
            "e": False,
            "f": {"only_noise": "https://x.io/y", "id": 4},
            "g": ["https://x.io/y", None],
        }

        assert strip_noise(content) == {}

    def test_keeps_zero_and_true(self) -> None:
        content = {"power": 0, "armor": "0", "reserved": True, "cost": 0.5}

        assert strip_noise(content) == content

    def test_whole_floats_become_ints_and_fractions_are_kept(self) -> None:
        result = strip_noise({"cmc": 3.0, "half": 0.5, "nested": {"x": 2.0}})

        assert result == {"cmc": 3, "half": 0.5, "nested": {"x": 2}}
        assert isinstance(result["cmc"], int)

    def test_extra_noise_keys_are_dropped_at_every_depth(self) -> None:
        content = {
            "artist": "A",
            "card_faces": [{"name": "F", "artist": "B", "object": "card_face"}],
        }

        result = strip_noise(content, frozenset({"artist", "object"}))

        assert result == {"card_faces": [{"name": "F"}]}

    def test_does_not_mutate_its_input(self) -> None:
        content = {"name": "Bolt", "id": "1", "faces": [{"id": "2", "text": "t"}]}
        original = copy.deepcopy(content)

        strip_noise(content)

        assert content == original

    def test_empty_content_returns_empty_dict(self) -> None:
        assert strip_noise({}) == {}


class TestDropRepeatedStrings:
    def test_later_equal_string_is_dropped_first_is_kept(self) -> None:
        content = {"name": "Preserve Tradition", "true_name": "Preserve Tradition"}

        assert drop_repeated_strings(content) == {"name": "Preserve Tradition"}

    def test_short_and_numeric_strings_are_not_treated_as_repeats(self) -> None:
        content = {
            "a": "Red",
            "b": "Red",
            "c": "0000",
            "d": "0000",
            "e": "12345",
            "f": "12345",
        }

        assert drop_repeated_strings(content) == content

    def test_repeats_are_tracked_per_object_not_globally(self) -> None:
        content = {
            "name": "Bolt",
            "faces": [{"title": "Bolt"}, {"title": "Bolt"}],
        }

        assert drop_repeated_strings(content) == content

    def test_equal_non_string_values_are_kept(self) -> None:
        content = {"colors": ["R"], "color_identity": ["R"]}

        assert drop_repeated_strings(content) == content

    def test_does_not_mutate_its_input(self) -> None:
        content = {"a": "abcd", "b": "abcd", "c": [{"x": "wxyz", "y": "wxyz"}]}
        original = copy.deepcopy(content)

        drop_repeated_strings(content)

        assert content == original


class TestOrderKeys:
    def test_leading_first_trailing_last_middle_keeps_order(self) -> None:
        content = {"text": "t", "b": 2, "name": "n", "a": 1}

        result = order_keys(content, leading=["name"], trailing=["text"])

        assert list(result) == ["name", "b", "a", "text"]

    def test_leading_follows_the_order_given_not_the_content_order(self) -> None:
        content = {"b": 2, "a": 1}

        assert list(order_keys(content, leading=["a", "b"])) == ["a", "b"]

    def test_named_keys_missing_from_content_are_skipped(self) -> None:
        result = order_keys({"a": 1}, leading=["missing"], trailing=["gone"])

        assert result == {"a": 1}

    def test_same_items_as_input(self) -> None:
        content = {"z": 1, "a": 2, "m": 3}

        assert order_keys(content, leading=["m"]) == content

    def test_key_in_both_leading_and_trailing_raises(self) -> None:
        with pytest.raises(ValueError):
            order_keys({"a": 1}, leading=["a"], trailing=["a"])


class TestMapStrings:
    def test_transforms_string_values_at_every_depth(self) -> None:
        content = {"a": "x", "b": ["x", {"c": "x"}], "d": {"e": "x"}}

        assert map_strings(content, str.upper) == {
            "a": "X",
            "b": ["X", {"c": "X"}],
            "d": {"e": "X"},
        }

    def test_keys_and_non_strings_are_untouched(self) -> None:
        content = {"key": 3, "flag": True, "none": None, "half": 0.5}

        assert map_strings(content, str.upper) == content

    def test_does_not_mutate_its_input(self) -> None:
        content = {"a": "x", "b": [{"c": "x"}]}
        original = copy.deepcopy(content)

        map_strings(content, str.upper)

        assert content == original
