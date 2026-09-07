"""Content-addressed id schemes shared across every metrics/<source>
container.

See plans/sts_gg_metrics.md's "New shared components" section for the
design this implements: a metric whose training input is a whole deck
mints this module's deck_uuid_from_cards() rather than embedding the
deck's full card list in its own output row, and writes the deck once
(via DeckBox.create_if_absent()) into a private, metrics-only DeckBox
instance that every metric in the same scan pass shares — so two
metrics seeing the same deck actually dedupe against each other,
which only works if they all mint the SAME id from the SAME content.
That's why this hashing lives in one shared file rather than each
metric rolling its own.

A data source whose deck identity needs source-specific logic gets its
own metrics/<source>/hash_utils.py instead of adding cases here — this
file only ever holds hashing that's meaningful across every source.
"""

from uuid import UUID, uuid5

# Fixed, arbitrary - never regenerate. Distinct from every other uuid5
# namespace already minted in this codebase (e.g.
# deck_box/sts_gg/extraction_stage.py's _DECK_NAMESPACE) - collision
# with that namespace is moot regardless, since its input (a raw run
# id string) and this module's input (a joined, sorted uuid string)
# can never coincide, but a fresh constant is minted anyway as this
# project's convention for a new deterministic-id scheme.
_DECK_CONTENT_NAMESPACE = UUID("6a2f8e3d-6a51-4b8c-9d3e-1f7c2b9a4e6d")


def deck_uuid_from_cards(card_nocab_uuids: list[UUID]) -> UUID:
    """Deterministic content-addressed id for a deck's card multiset.

    Sorts card_nocab_uuids before hashing (it's a multiset, not an
    ordered list - see GenericDeck's own docstring in
    src/schema/card.py), so two calls with the same multiset in a
    different order produce the same id. Duplicates are preserved by
    the sort, not collapsed - two copies of one card hash differently
    from one copy.

    Inputs:
        card_nocab_uuids: a deck's card multiset.
    Output: uuid5(_DECK_CONTENT_NAMESPACE, <sorted uuids joined>) -
        deterministic for a given multiset, stable across runs and
        across process restarts.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> deck_uuid_from_cards([uuid1, uuid2]) == deck_uuid_from_cards(
        ...     [uuid2, uuid1]
        ... )
        True
    """
    sorted_ids = sorted(str(card_uuid) for card_uuid in card_nocab_uuids)
    return uuid5(_DECK_CONTENT_NAMESPACE, ",".join(sorted_ids))
