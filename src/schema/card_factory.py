"""GenericCardFactory: builds new GenericCards from existing ones without
mutating them.

Edits use path copying. Only the containers on the path from raw_content
down to the edited slot are copied, one shallow copy per level. Every
other object is shared with the input card, so an edit costs
O(depth x width of the containers on the path), not O(size of the card)
like copy.deepcopy.

Sharing is safe only because nothing mutates a card's raw_content in
place (GenericCard's rule). To build a new card from an old one, use this
factory or Python's non-mutating expressions directly: {**d, k: v},
[*xs, x], sorted(xs), rng.sample(xs, len(xs)), and so on.
"""

import dataclasses
from enum import Enum
from typing import cast

from src.schema.card import GenericCard

FieldPath = tuple[str | int, ...]
"""Steps from raw_content down to one slot. A str step indexes a dict and
an int step indexes a list, e.g. ("card_faces", 1, "oracle_text"). List
indices must be non-negative (no -1 counting from the end), and bool is
not accepted as an int step, so each step names exactly one slot."""


class MissingPathPolicy(Enum):
    """What GenericCardFactory.with_field does when a step names a slot
    that doesn't exist: a dict key that is absent, or a list index past
    the end.

    STRICT: raise KeyError (dict) or IndexError (list).
    ADD: create each missing dict key. Before the last step the new key
        holds a new empty dict; at the last step it holds the value. A
        list index can't be created, so it still raises IndexError, as
        does an int step that would have to index a newly created dict.
    PASS: return the input card unchanged (the same object).

    A step whose type doesn't match its container is a malformed path,
    not a missing one. That includes a str step into a list, an int step
    into a dict, and any step into a non-container. It raises TypeError
    under every policy. Steps are followed in order, so a mismatch is only
    found if no earlier step was missing: after a missing step, PASS
    returns and STRICT raises before the mismatch is reached. Malformed
    steps (a bool, a negative index, a type that is neither str nor int)
    are rejected before the card is read, whatever else the path holds.
    """

    STRICT = "strict"
    ADD = "add"
    PASS = "pass"


# Marks a slot that doesn't exist, as distinct from a slot that holds None
_ABSENT = object()


class GenericCardFactory:
    """Stateless factory. Every method returns a new GenericCard and
    leaves its input unchanged. Not meant to be instantiated; call the
    static methods."""

    @staticmethod
    def with_field(
        card: GenericCard,
        path: FieldPath,
        value: object,
        missing: MissingPathPolicy = MissingPathPolicy.STRICT,
    ) -> GenericCard:
        """A new card whose raw_content has `value` at `path`; `card` is unchanged.

        Inputs:
            card: GenericCard to derive from. Never mutated.
            path: FieldPath with at least one step.
            value: object stored at that slot as is (not copied).
            missing: MissingPathPolicy for a step whose slot doesn't
                exist. The default, STRICT, raises.
        Output: GenericCard. It shares every raw_content object off the
            path with `card`, and each container on the path is a new
            shallow copy. Under PASS, when the path runs out, the output
            is `card` itself.
        Side effects: none.
        Exceptions:
            ValueError: path is empty or has a negative list index.
            TypeError: a step is not a str or int, or is a bool; a step's
                type doesn't match its container; or a step before the
                last reaches something that is neither a dict nor a list.
            KeyError: a dict key is missing under STRICT.
            IndexError: a list index is past the end under STRICT or ADD,
                or ADD would have to index a new dict with an int step.

        Example:
            >>> masked = GenericCardFactory.with_field(card, ("costs", "mana"), "[MASK]")
            >>> card.raw_content["costs"]["mana"]      # input untouched
            '{R}'
            >>> masked.raw_content["rulings"] is card.raw_content["rulings"]
            True
        """
        # Validate inputs; malformed paths fail before the card is read
        GenericCardFactory._check_path(path)

        # Walk down the path, keeping each container it passes through
        containers = GenericCardFactory._containers_along(
            card.raw_content, path, missing
        )
        if containers is None:
            return card  # PASS: the path ran out

        # Rebuild bottom-up: each level becomes a copy of its container with
        # one slot pointing at the new child built by the level below it.
        # A loop, not recursion (PRINCIPLES section 4).
        new_child: object = value
        for container, step in zip(reversed(containers), reversed(path)):
            new_child = GenericCardFactory._copy_with_slot(container, step, new_child)

        return dataclasses.replace(card, raw_content=cast(dict, new_child))

    @staticmethod
    def _check_path(path: FieldPath) -> None:
        """Reject a path that is malformed whatever card it is used on.

        Inputs: path (FieldPath).
        Output: None.
        Side effects: none.
        Exceptions: ValueError if path is empty or has a negative int
            step. TypeError if a step is a bool or is neither a str nor
            an int.
        """
        if not path:
            raise ValueError("path must have at least one step")
        for step in path:
            # bool is a subclass of int; rule it out before the int check
            if isinstance(step, bool) or not isinstance(step, (str, int)):
                raise TypeError(f"path step {step!r} must be a str or an int")
            if isinstance(step, int) and step < 0:
                raise ValueError(f"path step {step} must be a non-negative index")

    @staticmethod
    def _containers_along(
        raw_content: dict, path: FieldPath, missing: MissingPathPolicy
    ) -> list[dict | list] | None:
        """The container each step of `path` indexes into, top-down.

        Also checks that the last step's slot exists, or applies `missing`
        if it doesn't.

        Inputs: raw_content (dict); path (FieldPath, already checked);
            missing (MissingPathPolicy).
        Output: a list the same length as path, where element i is the
            container path[i] indexes and element 0 is raw_content. Under
            ADD, containers for missing keys are new empty dicts. Returns
            None under PASS when a slot is missing.
        Side effects: none. Reads only; created dicts are new objects.
        Exceptions: as in with_field (TypeError, KeyError, IndexError).
        """
        result: list[dict | list] = []
        container: object = raw_content

        # Follow each step, one container per step
        for index, step in enumerate(path):
            if not isinstance(container, (dict, list)):
                raise TypeError(
                    f"path step {step!r} indexes a {type(container).__name__}, "
                    "not a dict or list"
                )
            result.append(container)
            child = GenericCardFactory._child_at(container, step)
            is_last = index == len(path) - 1

            if child is _ABSENT:
                if missing is MissingPathPolicy.PASS:
                    return None
                GenericCardFactory._raise_unless_addable(
                    container,
                    step,
                    missing,
                    next_step=None if is_last else path[index + 1],
                )
                child = {}  # ADD: only reached for a dict key
            container = child
        return result

    @staticmethod
    def _child_at(container: dict | list, step: str | int) -> object:
        """container[step], or _ABSENT if that slot doesn't exist.

        Inputs: container (dict or list); step (str or non-negative int).
        Output: the child object, or _ABSENT.
        Side effects: none.
        Exceptions: TypeError if step's type doesn't match the container.
        """
        if isinstance(container, dict):
            if not isinstance(step, str):
                raise TypeError(f"int step {step!r} cannot index a dict")
            return container.get(step, _ABSENT)
        if not isinstance(step, int):
            raise TypeError(f"str step {step!r} cannot index a list")
        return container[step] if step < len(container) else _ABSENT

    @staticmethod
    def _raise_unless_addable(
        container: dict | list,
        step: str | int,
        missing: MissingPathPolicy,
        next_step: str | int | None,
    ) -> None:
        """Raise unless ADD can create the missing slot container[step].

        Inputs: container (dict or list); step (its missing slot); missing
            (STRICT or ADD, never PASS); next_step (the following path
            step, or None if step is the last).
        Output: None, meaning ADD may create a dict key here.
        Side effects: none.
        Exceptions: KeyError for a missing dict key under STRICT.
            IndexError for a missing list index (STRICT or ADD), and
            under ADD for an int next_step, since the new container
            would be a dict.
        """
        if isinstance(container, list):
            raise IndexError(
                f"list index {step} out of range (length {len(container)})"
            )
        if missing is MissingPathPolicy.STRICT:
            raise KeyError(step)
        if isinstance(next_step, int):
            raise IndexError(
                f"cannot add key {step!r}: the next step {next_step} is a list "
                "index, and ADD only creates dicts"
            )

    @staticmethod
    def _copy_with_slot(
        container: dict | list, step: str | int, value: object
    ) -> dict | list:
        """A shallow copy of `container` with container[step] = value.

        Inputs: container (dict or list); step (a str for a dict, an
            in-range int for a list, both already checked by
            _containers_along); value (object).
        Output: a new container of the same type. Every other slot keeps
            its object reference, and a missing dict key is added.
        Side effects: none; `container` is unchanged. Only the new copy
            is written to.
        Exceptions: none for a checked step.
        """
        if isinstance(container, dict):
            return {**container, cast(str, step): value}
        copied = list(container)
        copied[cast(int, step)] = value
        return copied
