"""GenericCardFactory.with_field: returns a new card with the edit applied,
leaves the input untouched, and shares everything off the edited path
instead of copying it.

Every edit goes through _edit, which checks that the input card is
unchanged against a deep-copied snapshot, whether the edit returns or
raises.
"""

import copy
import random
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

import pytest

from src.schema.card import GenericCard, Provenance
from src.schema.card_factory import FieldPath, GenericCardFactory, MissingPathPolicy
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

ALL_POLICIES = list(MissingPathPolicy)


def _card(raw_content: dict) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="Test Card",
        raw_content=raw_content,
        provenance=Provenance(
            DataSource.SCRYFALL, "src-id", datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def _bolt() -> GenericCard:
    """A card with dicts, lists, dicts inside lists, None, and scalars."""
    return _card(
        {
            "name": "Lightning Bolt",
            "costs": {"mana": "{R}", "cmc": 1},
            "rulings": ["r1", "r2"],
            "card_faces": [
                {"oracle_text": "Deal 3.", "types": ["Instant"]},
                {"oracle_text": "Back.", "types": ["Sorcery"]},
            ],
            "empty_list": [],
            "empty_dict": {},
            "nothing": None,
        }
    )


def _get(raw_content: Any, path: FieldPath) -> Any:
    for step in path:
        raw_content = raw_content[step]
    return raw_content


def _edit(
    card: GenericCard, path: FieldPath, value: object, **kwargs: Any
) -> GenericCard:
    """with_field, asserting the input card is unchanged afterwards - also
    when with_field raises."""
    snapshot = copy.deepcopy(card.raw_content)
    try:
        return GenericCardFactory.with_field(card, path, value, **kwargs)
    finally:
        assert card.raw_content == snapshot, "with_field mutated its input card"


# --- happy path: existing slots ---------------------------------------------


@pytest.mark.parametrize("policy", ALL_POLICIES)
@pytest.mark.parametrize(
    "path",
    [
        ("name",),
        ("costs", "mana"),
        ("rulings", 0),
        ("rulings", 1),
        ("card_faces", 1, "oracle_text"),
        ("card_faces", 0, "types", 0),
        ("nothing",),  # a slot holding None exists; it is not missing
    ],
)
def test_replacing_an_existing_slot_sets_it_under_every_policy(
    path: FieldPath, policy: MissingPathPolicy
) -> None:
    card = _bolt()

    result = _edit(card, path, "[MASK]", missing=policy)

    assert _get(result.raw_content, path) == "[MASK]"
    assert result is not card


def test_default_policy_is_strict() -> None:
    with pytest.raises(KeyError):
        _edit(_bolt(), ("missing",), 1)


def test_identity_fields_are_preserved() -> None:
    card = _bolt()

    result = _edit(card, ("costs", "mana"), "[MASK]")

    assert result.nocab_uuid == card.nocab_uuid
    assert result.source_game == card.source_game
    assert result.name == card.name
    assert result.provenance is card.provenance


def test_value_is_stored_as_is_not_copied() -> None:
    value = {"nested": ["x"]}

    result = _edit(_bolt(), ("costs",), value)

    assert result.raw_content["costs"] is value


def test_replacing_a_whole_subtree_replaces_it() -> None:
    result = _edit(_bolt(), ("card_faces",), "[MASK]")

    assert result.raw_content["card_faces"] == "[MASK]"


def test_replacing_with_the_same_value_still_returns_a_new_card() -> None:
    card = _bolt()

    result = _edit(card, ("name",), card.raw_content["name"])

    assert result is not card
    assert result.raw_content is not card.raw_content
    assert result.raw_content == card.raw_content


def test_list_edit_keeps_length_and_order() -> None:
    result = _edit(_bolt(), ("rulings", 0), "X")

    assert result.raw_content["rulings"] == ["X", "r2"]


def test_the_last_index_of_a_list_is_editable() -> None:
    card = _bolt()
    last = len(card.raw_content["card_faces"]) - 1

    result = _edit(card, ("card_faces", last, "oracle_text"), "X")

    assert result.raw_content["card_faces"][last]["oracle_text"] == "X"


def test_consecutive_edits_compose_and_leave_earlier_cards_unchanged() -> None:
    card = _bolt()

    first = _edit(card, ("costs", "mana"), "A")
    second = _edit(first, ("costs", "cmc"), "B")

    assert second.raw_content["costs"] == {"mana": "A", "cmc": "B"}
    assert first.raw_content["costs"] == {"mana": "A", "cmc": 1}
    assert card.raw_content["costs"] == {"mana": "{R}", "cmc": 1}


# --- structural sharing ------------------------------------------------------


def test_containers_on_the_path_are_new_and_everything_off_it_is_shared() -> None:
    card = _bolt()
    old = card.raw_content

    new = _edit(card, ("card_faces", 1, "oracle_text"), "X").raw_content

    # Copied: every container on the path
    assert new is not old
    assert new["card_faces"] is not old["card_faces"]
    assert new["card_faces"][1] is not old["card_faces"][1]
    # Shared: every sibling at every level
    assert new["costs"] is old["costs"]
    assert new["rulings"] is old["rulings"]
    assert new["card_faces"][0] is old["card_faces"][0]
    assert new["card_faces"][1]["types"] is old["card_faces"][1]["types"]


def test_mutating_the_output_off_the_path_would_be_visible_in_the_input() -> None:
    """Documents why the no-mutation rule matters: sharing is real, so
    both cards see a change made through a shared object."""
    card = _bolt()
    result = GenericCardFactory.with_field(card, ("name",), "X")

    result.raw_content["rulings"].append("r3")

    assert card.raw_content["rulings"] == ["r1", "r2", "r3"]


def test_mutating_the_output_on_the_path_is_not_visible_in_the_input() -> None:
    card = _bolt()
    result = GenericCardFactory.with_field(card, ("costs", "mana"), "X")

    result.raw_content["costs"]["cmc"] = 99

    assert card.raw_content["costs"]["cmc"] == 1


# --- missing slots: STRICT ---------------------------------------------------


@pytest.mark.parametrize(
    "path, error",
    [
        (("missing",), KeyError),  # last step, dict
        (("missing", "deeper"), KeyError),  # intermediate step, dict
        (("costs", "missing"), KeyError),  # last step of a nested dict
        (("rulings", 2), IndexError),  # last step, list
        (("empty_list", 0), IndexError),
        (("card_faces", 5, "oracle_text"), IndexError),  # intermediate list
    ],
)
def test_strict_raises_on_a_missing_slot(path: FieldPath, error: type) -> None:
    with pytest.raises(error):
        _edit(_bolt(), path, 1, missing=MissingPathPolicy.STRICT)


# --- missing slots: ADD ------------------------------------------------------


def test_add_adds_a_missing_top_level_key() -> None:
    result = _edit(_bolt(), ("faction-duo",), "[MASK]", missing=MissingPathPolicy.ADD)

    assert result.raw_content["faction-duo"] == "[MASK]"


def test_add_extends_the_path_with_new_dicts() -> None:
    card = _bolt()

    result = _edit(card, ("a", "b", "c"), 7, missing=MissingPathPolicy.ADD)

    assert result.raw_content["a"] == {"b": {"c": 7}}
    assert result.raw_content["costs"] is card.raw_content["costs"]


def test_add_extends_below_an_existing_dict() -> None:
    result = _edit(_bolt(), ("costs", "new", "leaf"), 1, missing=MissingPathPolicy.ADD)

    assert result.raw_content["costs"] == {"mana": "{R}", "cmc": 1, "new": {"leaf": 1}}


def test_add_extends_inside_a_dict_in_a_list() -> None:
    result = _edit(
        _bolt(), ("card_faces", 0, "flavor"), "X", missing=MissingPathPolicy.ADD
    )

    assert result.raw_content["card_faces"][0]["flavor"] == "X"


def test_add_into_an_empty_dict() -> None:
    result = _edit(_bolt(), ("empty_dict", "k"), 1, missing=MissingPathPolicy.ADD)

    assert result.raw_content["empty_dict"] == {"k": 1}


@pytest.mark.parametrize(
    "path",
    [
        ("rulings", 2),  # list index past the end, last step
        ("empty_list", 0),
        ("card_faces", 2, "oracle_text"),  # list index past the end, mid-path
        ("new_key", 0),  # would need to index a new dict with an int
        ("costs", "new", 3, "x"),
    ],
)
def test_add_cannot_create_list_slots(path: FieldPath) -> None:
    with pytest.raises(IndexError):
        _edit(_bolt(), path, 1, missing=MissingPathPolicy.ADD)


def test_added_dicts_are_not_shared_between_results() -> None:
    card = _bolt()

    first = _edit(card, ("a", "b"), 1, missing=MissingPathPolicy.ADD)
    second = _edit(card, ("a", "b"), 2, missing=MissingPathPolicy.ADD)

    assert first.raw_content["a"] is not second.raw_content["a"]
    assert first.raw_content["a"] == {"b": 1}


# --- missing slots: PASS -----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ("missing",),
        ("missing", "deeper", "still"),
        ("costs", "missing"),
        ("rulings", 2),
        ("empty_list", 0),
        ("card_faces", 5, "oracle_text"),
        ("card_faces", 0, "missing"),
    ],
)
def test_pass_returns_the_input_card_itself_when_the_path_runs_out(
    path: FieldPath,
) -> None:
    card = _bolt()

    result = _edit(card, path, 1, missing=MissingPathPolicy.PASS)

    assert result is card


# --- malformed paths: same error under every policy -------------------------


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_empty_path_raises_value_error(policy: MissingPathPolicy) -> None:
    with pytest.raises(ValueError):
        _edit(_bolt(), (), 1, missing=policy)


@pytest.mark.parametrize("policy", ALL_POLICIES)
@pytest.mark.parametrize("path", [("rulings", -1), ("card_faces", -2, "oracle_text")])
def test_negative_index_raises_value_error(
    path: FieldPath, policy: MissingPathPolicy
) -> None:
    with pytest.raises(ValueError):
        _edit(_bolt(), path, 1, missing=policy)


@pytest.mark.parametrize("policy", ALL_POLICIES)
@pytest.mark.parametrize(
    "path",
    [
        ("rulings", True),  # bool is not an int step
        ("rulings", False),
        ("costs", 1.5),  # neither str nor int
        (None,),
        (("tuple",),),
    ],
)
def test_bad_step_type_raises_type_error(
    path: tuple[Any, ...], policy: MissingPathPolicy
) -> None:
    with pytest.raises(TypeError):
        _edit(_bolt(), path, 1, missing=policy)


@pytest.mark.parametrize("policy", ALL_POLICIES)
@pytest.mark.parametrize(
    "path",
    [
        ("rulings", "first"),  # str step into a list
        ("costs", 0),  # int step into a dict
        (0,),  # int step into raw_content itself
        ("name", "x"),  # step into a str
        ("costs", "cmc", "x"),  # step into an int
        ("nothing", "x"),  # step into None
        ("card_faces", 0, "oracle_text", 0),  # int step into a str
    ],
)
def test_step_type_mismatch_raises_type_error_under_every_policy(
    path: FieldPath, policy: MissingPathPolicy
) -> None:
    with pytest.raises(TypeError):
        _edit(_bolt(), path, 1, missing=policy)


def test_malformed_path_is_rejected_even_when_an_earlier_step_is_missing() -> None:
    """The path is checked before the card is read, so PASS can't hide it."""
    with pytest.raises(ValueError):
        _edit(_bolt(), ("missing", -1), 1, missing=MissingPathPolicy.PASS)


# --- randomized invariants ---------------------------------------------------


def _random_content(rng: random.Random, depth: int) -> Any:
    """A random JSON-like value: dicts and lists nested up to `depth`."""
    kind = rng.choice(["dict", "list", "leaf"] if depth > 0 else ["leaf"])
    if kind == "dict":
        return {
            f"k{i}": _random_content(rng, depth - 1) for i in range(rng.randint(0, 4))
        }
    if kind == "list":
        return [_random_content(rng, depth - 1) for _ in range(rng.randint(0, 4))]
    return rng.choice([0, 1.5, "s", None, True])


def _all_paths(raw_content: Any) -> Iterator[FieldPath]:
    """Every existing slot's path, found with an explicit stack (no recursion)."""
    stack: list[tuple[Any, FieldPath]] = [(raw_content, ())]
    while stack:
        node, prefix = stack.pop()
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for step, child in items:
            path = (*prefix, step)
            yield path
            if isinstance(child, (dict, list)):
                stack.append((child, path))


@pytest.mark.parametrize("seed", range(25))
def test_every_existing_path_edits_exactly_one_slot_and_shares_the_rest(
    seed: int,
) -> None:
    rng = random.Random(seed)
    raw = {f"top{i}": _random_content(rng, depth=4) for i in range(5)}
    card = _card(raw)
    sentinel = object()

    for path in _all_paths(card.raw_content):
        result = _edit(card, path, sentinel)

        # The edited slot holds the value
        assert _get(result.raw_content, path) is sentinel
        # Every other slot not below the edit is unchanged, and any slot
        # off the path is the very same object
        for other in _all_paths(card.raw_content):
            if other[: len(path)] == path:
                continue  # at or below the edited slot
            if path[: len(other)] == other:
                continue  # an ancestor container on the path: a new copy
            assert _get(result.raw_content, other) is _get(card.raw_content, other)


def _container_paths(raw_content: dict) -> Iterator[FieldPath]:
    """The path of raw_content itself and of every dict or list inside it."""
    yield ()
    for path in _all_paths(raw_content):
        if isinstance(_get(raw_content, path), (dict, list)):
            yield path


@pytest.mark.parametrize("seed", range(25))
def test_every_missing_path_follows_its_policy(seed: int) -> None:
    """Hang a missing tail off every container. A dict gets one or two new
    str keys; a list gets its first out-of-range index. Check all three
    policies on each."""
    rng = random.Random(seed)
    card = _card({f"top{i}": _random_content(rng, depth=4) for i in range(5)})

    for prefix in _container_paths(card.raw_content):
        container = _get(card.raw_content, prefix)
        if isinstance(container, dict):
            tails: list[FieldPath] = [("zz_new",), ("zz_new", "deeper")]
        else:
            tails = [(len(container),)]
        for tail in tails:
            path = (*prefix, *tail)
            is_list_slot = isinstance(tail[0], int)

            # PASS: the card itself, untouched
            assert _edit(card, path, 1, missing=MissingPathPolicy.PASS) is card

            # STRICT: KeyError for a dict key, IndexError for a list index
            with pytest.raises(IndexError if is_list_slot else KeyError):
                _edit(card, path, 1, missing=MissingPathPolicy.STRICT)

            # ADD: dict tails are created; list slots can't be
            if is_list_slot:
                with pytest.raises(IndexError):
                    _edit(card, path, 1, missing=MissingPathPolicy.ADD)
            else:
                added = _edit(card, path, 1, missing=MissingPathPolicy.ADD)
                assert _get(added.raw_content, path) == 1


# --- edge values -------------------------------------------------------------


def test_none_can_be_stored_as_the_value() -> None:
    result = _edit(_bolt(), ("costs", "mana"), None)

    assert result.raw_content["costs"] == {"mana": None, "cmc": 1}


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_editing_an_empty_raw_content(policy: MissingPathPolicy) -> None:
    card = _card({})

    if policy is MissingPathPolicy.STRICT:
        with pytest.raises(KeyError):
            _edit(card, ("a", "b"), 1, missing=policy)
    elif policy is MissingPathPolicy.PASS:
        assert _edit(card, ("a", "b"), 1, missing=policy) is card
    else:
        assert _edit(card, ("a", "b"), 1, missing=policy).raw_content == {"a": {"b": 1}}


def test_a_missing_step_before_a_mismatch_is_handled_by_the_policy() -> None:
    """("missing", 0) under PASS: the missing key is found first, so the
    card comes back unchanged. The int step is never checked against the
    dict ADD would have created (see MissingPathPolicy)."""
    card = _bolt()

    assert _edit(card, ("missing", 0), 1, missing=MissingPathPolicy.PASS) is card
    with pytest.raises(KeyError):
        _edit(card, ("missing", 0), 1, missing=MissingPathPolicy.STRICT)
