import random
from datetime import datetime, timezone
from typing import Any

import pytest

from src.dojos.mods.content_rewrite import (
    ContentRewrite,
    KeyCollisionError,
    flattened_content,
    rewritten_content,
)
from src.dojos.mods.content_style_mods import (
    _KEY_CASE_JOINERS,
    BoolSpelling,
    BoolToStringMod,
    FlattenKeysMod,
    KeyCase,
    KeyCaseMod,
    ListToStringMod,
    NumberToStringMod,
    ValueCase,
    ValueCaseMod,
    _key_words,
)
from src.dojos.mods.mod import MASK_TOKEN
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.dojos.mods.card_ids import content_uuid


def _card(raw_content: dict[str, Any]) -> GenericCard:
    return GenericCard(
        nocab_uuid=content_uuid(raw_content),
        source_game=GameId.POKEMON,
        name="Test Card",
        raw_content=raw_content,
        provenance=Provenance(
            DataSource.POKEMON_TCG, "src-id", datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def _pokemon_content() -> dict[str, Any]:
    return {
        "name": "Pikachu",
        "hp": 60,
        "is_basic": True,
        "types": ["Lightning"],
        "attacks": [{"name": "Thunder Jolt", "damage": 30, "cost": {"lightning": 2}}],
        "retreat": 1.5,
        "rules": None,
    }


class TestContentRewrite:
    def test_identity_rewrite_copies_every_container(self) -> None:
        content = _pokemon_content()

        result = rewritten_content(content, ContentRewrite())

        assert result == content
        assert result is not content
        assert result["attacks"] is not content["attacks"]
        assert result["attacks"][0] is not content["attacks"][0]

    def test_keys_are_rewritten_at_every_depth(self) -> None:
        class Upper(ContentRewrite):
            def key(self, key: str) -> str:
                return key.upper()

        result = rewritten_content({"a": {"b": 1}, "c": [{"d": 2}]}, Upper())

        assert result == {"A": {"B": 1}, "C": [{"D": 2}]}

    def test_colliding_keys_raise(self) -> None:
        class Lower(ContentRewrite):
            def key(self, key: str) -> str:
                return key.lower()

        with pytest.raises(KeyCollisionError):
            rewritten_content({"a": 1, "A": 2}, Lower())

    def test_input_is_unchanged(self) -> None:
        content = _pokemon_content()
        before = repr(content)

        rewritten_content(content, ContentRewrite())
        flattened_content(content, ".")

        assert repr(content) == before


class TestFlattenedContent:
    def test_nested_objects_move_to_the_top_level(self) -> None:
        result = flattened_content({"name": "Pikachu", "stats": {"hp": 60}}, ".")

        assert result == {"name": "Pikachu", "stats.hp": 60}

    def test_deep_paths_join_every_step(self) -> None:
        assert flattened_content({"a": {"b": {"c": 1}}}, "/") == {"a/b/c": 1}

    def test_lists_stay_and_their_objects_flatten_in_place(self) -> None:
        content = {"attacks": [{"cost": {"fire": 1}}, [{"x": {"y": 2}}], 3]}

        result = flattened_content(content, ".")

        assert result == {"attacks": [{"cost.fire": 1}, [{"x.y": 2}], 3]}

    def test_empty_object_stays_as_a_value(self) -> None:
        assert flattened_content({"a": {}, "b": 1}, ".") == {"a": {}, "b": 1}

    def test_key_order_follows_the_walk(self) -> None:
        result = flattened_content({"a": 1, "b": {"x": 2, "y": 3}, "c": 4}, ".")

        assert list(result) == ["a", "b.x", "b.y", "c"]

    def test_a_joined_path_equal_to_an_existing_key_raises(self) -> None:
        with pytest.raises(KeyCollisionError):
            flattened_content({"stats.hp": 1, "stats": {"hp": 2}}, ".")


class TestKeyWords:
    @pytest.mark.parametrize(
        ("key", "words"),
        [
            ("manaCost_total", ["mana", "cost", "total"]),
            ("mana_cost", ["mana", "cost"]),
            ("mana-cost", ["mana", "cost"]),
            ("Mana Cost", ["mana", "cost"]),
            ("ManaCost", ["mana", "cost"]),
            ("MANA_COST", ["mana", "cost"]),
            ("HPValue", ["hp", "value"]),
            ("hp", ["hp"]),
            ("attack2Damage", ["attack", "2", "damage"]),
        ],
    )
    def test_splits_every_case(self, key: str, words: list[str]) -> None:
        assert _key_words(key) == words

    @pytest.mark.parametrize("key", ["{R}", "stats.hp", "faction:duo", "", "__"])
    def test_other_characters_or_no_words_keep_the_key(self, key: str) -> None:
        assert _key_words(key) is None

    def test_every_key_case_has_a_joiner(self) -> None:
        assert set(_KEY_CASE_JOINERS) == set(KeyCase)


class TestKeyCaseMod:
    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            (KeyCase.SNAKE, "mana_cost"),
            (KeyCase.UPPER_SNAKE, "MANA_COST"),
            (KeyCase.CAMEL, "manaCost"),
            (KeyCase.PASCAL, "ManaCost"),
            (KeyCase.KEBAB, "mana-cost"),
            (KeyCase.TITLE, "Mana Cost"),
        ],
    )
    def test_every_case(self, case: KeyCase, expected: str) -> None:
        result = KeyCaseMod().restyled_content({"manaCost": 1}, case)

        assert result == {expected: 1}

    def test_applies_one_case_at_every_depth(self) -> None:
        content = {"mana_cost": {"red": 1}, "faces": [{"card_name": "x"}]}

        result = KeyCaseMod().restyled_content(content, KeyCase.PASCAL)

        assert result == {"ManaCost": {"Red": 1}, "Faces": [{"CardName": "x"}]}

    def test_keeps_keys_with_other_characters(self) -> None:
        result = KeyCaseMod().restyled_content({"{R}": 1, "a_b": 2}, KeyCase.CAMEL)

        assert result == {"{R}": 1, "aB": 2}

    def test_values_are_untouched(self) -> None:
        result = KeyCaseMod().restyled_content({"name": "mana_cost"}, KeyCase.CAMEL)

        assert result == {"name": "mana_cost"}

    def test_colliding_keys_pass_the_card_through(self) -> None:
        card = _card({"mana_cost": 1, "manaCost": 2})
        mod = KeyCaseMod((KeyCase.CAMEL,), rng_seed=0)

        result, _ = mod.apply_single((card, 0))

        assert result is card
        assert mod.tally.cards_failed == 1


class TestFlattenKeysMod:
    def test_flattens_with_the_separator(self) -> None:
        result = FlattenKeysMod().restyled_content({"stats": {"hp": 60}}, "/")

        assert result == {"stats/hp": 60}

    def test_default_separator_is_a_dot(self) -> None:
        assert FlattenKeysMod().styles == (".",)

    def test_a_flat_card_is_passed_through(self) -> None:
        card = _card({"name": "x", "hp": 1})

        result, _ = FlattenKeysMod(rng_seed=0).apply_single((card, 0))

        assert result is card

    @pytest.mark.parametrize("mod_class", [FlattenKeysMod, ListToStringMod])
    def test_rejects_an_empty_separator(self, mod_class: type) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            mod_class(("", "."))


class TestValueCaseMod:
    @pytest.mark.parametrize(
        ("case", "expected"),
        [(ValueCase.UPPER, "THUNDER JOLT"), (ValueCase.LOWER, "thunder jolt")],
    )
    def test_every_case_at_every_depth(self, case: ValueCase, expected: str) -> None:
        content = {"attacks": [{"name": "Thunder Jolt"}], "names": ["Thunder Jolt"]}

        result = ValueCaseMod().restyled_content(content, case)

        assert result == {"attacks": [{"name": expected}], "names": [expected]}

    def test_keys_and_non_strings_are_untouched(self) -> None:
        content = {"Name": "x", "hp": 60, "basic": True, "rules": None}

        result = ValueCaseMod().restyled_content(content, ValueCase.UPPER)

        assert result == {"Name": "X", "hp": 60, "basic": True, "rules": None}

    def test_mask_token_is_kept(self) -> None:
        result = ValueCaseMod().restyled_content({"name": MASK_TOKEN}, ValueCase.LOWER)

        assert result == {"name": MASK_TOKEN}


class TestBoolToStringMod:
    @pytest.mark.parametrize(
        ("spelling", "true_text", "false_text"),
        [
            (BoolSpelling.LOWER, "true", "false"),
            (BoolSpelling.TITLE, "True", "False"),
            (BoolSpelling.YES_NO, "yes", "no"),
        ],
    )
    def test_every_spelling(
        self, spelling: BoolSpelling, true_text: str, false_text: str
    ) -> None:
        content = {"legal": True, "nested": {"banned": False}, "flags": [True]}

        result = BoolToStringMod().restyled_content(content, spelling)

        assert result == {
            "legal": true_text,
            "nested": {"banned": false_text},
            "flags": [true_text],
        }

    def test_numbers_are_not_bools(self) -> None:
        result = BoolToStringMod().restyled_content(
            {"hp": 1, "x": 0}, BoolSpelling.LOWER
        )

        assert result == {"hp": 1, "x": 0}


class TestListToStringMod:
    def test_joins_a_list_of_scalars(self) -> None:
        result = ListToStringMod().restyled_content({"colors": ["R", "G"]}, " / ")

        assert result == {"colors": "R / G"}

    def test_non_strings_are_written_as_json(self) -> None:
        result = ListToStringMod().restyled_content(
            {"x": [3, True, None, 1.5, "é"]}, ", "
        )

        assert result == {"x": "3, true, null, 1.5, é"}

    def test_lists_holding_containers_are_descended_into(self) -> None:
        content = {"attacks": [{"cost": ["Fire", "Fire"]}], "grid": [[1, 2], [3]]}

        result = ListToStringMod().restyled_content(content, ", ")

        assert result == {"attacks": [{"cost": "Fire, Fire"}], "grid": ["1, 2", "3"]}

    def test_empty_list_is_kept(self) -> None:
        assert ListToStringMod().restyled_content({"x": []}, ", ") == {"x": []}

    def test_singleton_list_becomes_its_item(self) -> None:
        assert ListToStringMod().restyled_content({"x": ["a"]}, ", ") == {"x": "a"}


class TestNumberToStringMod:
    def test_ints_and_floats_at_every_depth(self) -> None:
        content = {"hp": 60, "retreat": 1.5, "attacks": [{"damage": 30}], "x": [2]}

        result = NumberToStringMod().new_raw_content(content, random.Random(0))

        assert result == {
            "hp": "60",
            "retreat": "1.5",
            "attacks": [{"damage": "30"}],
            "x": ["2"],
        }

    def test_bools_and_strings_are_untouched(self) -> None:
        content = {"basic": True, "power": "6", "rules": None}

        result = NumberToStringMod().new_raw_content(content, random.Random(0))

        assert result == content

    def test_is_always_deterministic(self) -> None:
        assert NumberToStringMod().deterministic_cards


class TestStyleChoice:
    def test_default_styles(self) -> None:
        assert KeyCaseMod().styles == tuple(KeyCase)
        assert ValueCaseMod().styles == tuple(ValueCase)
        assert BoolToStringMod().styles == (BoolSpelling.LOWER,)
        assert ListToStringMod().styles == (", ",)

    def test_one_style_is_deterministic_and_never_stamped(self) -> None:
        mod = KeyCaseMod((KeyCase.CAMEL,), rng_seed=0)
        card = _card({"mana_cost": 1})

        result, _ = mod.apply_single((card, 0))

        assert mod.deterministic_cards
        assert result.raw_content == {"manaCost": 1}
        assert result.cacheable

    def test_several_unbounded_styles_are_stamped(self) -> None:
        mod = KeyCaseMod((KeyCase.CAMEL, KeyCase.KEBAB), rng_seed=0)

        result, _ = mod.apply_single((_card({"mana_cost": 1}), 0))

        assert not mod.deterministic_cards
        assert not result.cacheable

    def test_several_bounded_styles_are_deterministic(self) -> None:
        mod = KeyCaseMod((KeyCase.CAMEL, KeyCase.KEBAB), rng_seed=0, variants=2)

        result, _ = mod.apply_single((_card({"mana_cost": 1}), 0))

        assert mod.deterministic_cards
        assert result.cacheable

    def test_one_variant_gives_one_text_per_card(self) -> None:
        card = _card({"mana_cost": 1})
        texts = set()

        # Many draws, one variant: always the same style for this card
        mod = KeyCaseMod(rng_seed=0, variants=1)
        for _ in range(50):
            result, _ = mod.apply_single((card, 0))
            texts.add(next(iter(result.raw_content)))

        assert len(texts) == 1

    def test_several_styles_are_all_drawn(self) -> None:
        mod = KeyCaseMod(rng_seed=0)
        card = _card({"mana_cost": 1})
        keys = set()

        for _ in range(200):
            result, _ = mod.apply_single((card, 0))
            keys.add(next(iter(result.raw_content)))

        assert keys == {
            "mana_cost",
            "MANA_COST",
            "manaCost",
            "ManaCost",
            "mana-cost",
            "Mana Cost",
        }

    def test_probability_zero_never_changes_a_card(self) -> None:
        card = _card(_pokemon_content())

        result, _ = NumberToStringMod(probability=0.0, rng_seed=0).apply_single(
            (card, 0)
        )

        assert result is card

    def test_unchanged_content_returns_the_card_itself(self) -> None:
        card = _card({"name": "x"})

        result, _ = NumberToStringMod(rng_seed=0).apply_single((card, 0))

        assert result is card

    @pytest.mark.parametrize(
        ("styles", "error"),
        [
            ((), ValueError),
            ((KeyCase.CAMEL, KeyCase.CAMEL), ValueError),
            ([KeyCase.CAMEL], TypeError),
        ],
    )
    def test_rejects_bad_styles(self, styles: Any, error: type) -> None:
        with pytest.raises(error):
            KeyCaseMod(styles)

    @pytest.mark.parametrize(
        ("mod_class", "styles"),
        [
            (KeyCaseMod, ("camel",)),
            (ValueCaseMod, (KeyCase.CAMEL,)),
            (FlattenKeysMod, (1,)),
            (ListToStringMod, (None,)),
        ],
    )
    def test_rejects_styles_of_the_wrong_type(
        self, mod_class: type, styles: tuple[Any, ...]
    ) -> None:
        with pytest.raises(TypeError):
            mod_class(styles)

    def test_input_card_is_unchanged(self) -> None:
        card = _card(_pokemon_content())
        before = repr(card)

        for mod in (
            KeyCaseMod(rng_seed=0),
            FlattenKeysMod(rng_seed=0),
            ValueCaseMod(rng_seed=0),
            BoolToStringMod(rng_seed=0),
            ListToStringMod(rng_seed=0),
            NumberToStringMod(rng_seed=0),
        ):
            mod.apply_single((card, 0))

        assert repr(card) == before
