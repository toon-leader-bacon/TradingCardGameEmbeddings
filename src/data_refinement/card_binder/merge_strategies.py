"""Reusable collision-resolution policies for CardIngestionStage implementations.

See plans/card_binder_v2.md — each CardIngestionStage decides for
itself, per raw row, whether it has found a duplicate of an
already-stored card and, if so, which of these policies (or a custom
one of its own) to apply. CardBinder itself has no opinion on any of
this; these functions exist so more than one stage can share a named
policy instead of reimplementing one privately.

Hard contract every function in this module MUST satisfy: the
returned GenericCard always carries existing.nocab_uuid, never
candidate's. These functions decide CONTENT only (raw_content,
provenance, name) — never identity. candidate.nocab_uuid is freshly
minted by whichever stage produced it and is not yet known to be
canonical; returning it as-is (e.g. `return candidate`) would silently
let a card's nocab_uuid change on a later run, which
plans/card_binder_v2.md requires never happens once a uuid is minted.
"""

import json
from dataclasses import replace

from src.schema.card import GenericCard


def keep_existing(existing: GenericCard, candidate: GenericCard) -> GenericCard:
    """Prior entry wins, unconditionally.

    Inputs:
        existing: the card currently stored under this identity.
        candidate: the newly-seen card being considered — ignored
            entirely by this policy.
    Output: existing, unchanged.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> keep_existing(existing_card, candidate_card) == existing_card
        True
    """
    return existing


def keep_incoming(existing: GenericCard, candidate: GenericCard) -> GenericCard:
    """New entry wins, unconditionally.

    Still returns existing.nocab_uuid — see this module's docstring.
    Only raw_content/provenance/name move from candidate.

    Inputs:
        existing: the card currently stored under this identity.
        candidate: the newly-seen card whose content should win.
    Output: existing with raw_content/provenance/name replaced by
        candidate's.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> result = keep_incoming(existing_card, candidate_card)
        >>> result.nocab_uuid == existing_card.nocab_uuid
        True
        >>> result.raw_content == candidate_card.raw_content
        True
    """
    return replace(
        existing,
        raw_content=candidate.raw_content,
        provenance=candidate.provenance,
        name=candidate.name,
    )


def keep_longer_content(existing: GenericCard, candidate: GenericCard) -> GenericCard:
    """Whichever card's raw_content serializes to more JSON bytes wins.

    An exact tie favors existing. Ported from the pre-card_binder_v2
    CardBinder.add()'s richness comparison (_is_richer) — same
    behavior, relocated here so more than one stage can share it by
    name.

    Inputs:
        existing: the card currently stored under this identity.
        candidate: the newly-seen card being compared against it.
    Output: keep_incoming(existing, candidate) if candidate's
        raw_content is strictly larger when serialized, else existing
        unchanged.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> result = keep_longer_content(existing_card, candidate_card)
        >>> result.nocab_uuid == existing_card.nocab_uuid
        True
    """
    candidate_size = len(json.dumps(candidate.raw_content, sort_keys=True))
    existing_size = len(json.dumps(existing.raw_content, sort_keys=True))
    if candidate_size > existing_size:
        return keep_incoming(existing, candidate)
    return existing
