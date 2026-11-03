"""Reproducible card ids for mod tests.

A bounded-variant mod seeds each card's random state from its nocab_uuid
(src/dojos/mods/variant_seeds.py), so a test card built with uuid4()
makes the test draw differently on every run. content_uuid gives the
same card the same id every run, whatever order the tests run in.
"""

from typing import Any
from uuid import UUID, uuid5

# Fixed namespace for test card ids; never edit (draws depend on it)
_TEST_CARD_NAMESPACE = UUID("6f1d3c2a-8b4e-4f7a-9c1d-2e5b7a9f0c34")


def content_uuid(raw_content: Any) -> UUID:
    """A fixed UUID for a test card, from its content.

    Inputs: raw_content (anything with a stable repr; cards with equal
        content share an id, so give distinct cards distinct content).
    Output: UUID. Side effects: none. Exceptions: none.

    Example:
        >>> content_uuid({"name": "x"}) == content_uuid({"name": "x"})
        True
    """
    return uuid5(_TEST_CARD_NAMESPACE, repr(raw_content))
