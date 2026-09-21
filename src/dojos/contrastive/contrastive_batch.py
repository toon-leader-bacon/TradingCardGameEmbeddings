"""The (inputs, identities, positive_cliques) container a ContrastivePairConstructor builds.

See plans/contrastive_dojo.md's "ContrastiveBatch" section. Deliberately
not a two-list anchor/candidate split: every item acts as both an
anchor and a candidate for every other item in the same batch. Reuses
BatchedTrainingInput exactly (no new input representation) - the same
encoder forward() call every other dojo already makes.
"""

from dataclasses import dataclass
from uuid import UUID

from src.schema.type_hints import BatchedTrainingInput, input_shape_of


@dataclass
class ContrastiveBatch:
    """One contrastive training step's flat pool of items.

    inputs: every sampled item, one flat pool (not split by source deck).
    identities: parallel to items - one tuple of card uuid(s) per item,
        positionally parallel to that item's own card order (a 1-tuple
        for a single-card item; one uuid per card, same order as the
        item's cards/embeddings, for a multi-card item). This ordering
        is load-bearing: a multi-card ContrastiveLoss recovers a given
        card's own uuid by zipping its position in the item's embedding
        list against the same position in identities[i]. Duplicate card
        counts are preserved, never deduplicated or reordered.
    positive_cliques: each entry is a list of indices into items that
        are all mutually positive (every current positive-pair
        definition - same-deck co-occurrence, and same-color/same-
        archetype floated for later - is a partition of items into
        disjoint mutually-positive groups, so this is the direct,
        non-redundant representation rather than a derived (i, j) edge
        list). A group of size 1 contributes no positive pair at all -
        an accepted efficiency cost, not a correctness one (see
        pair_constructor.py's skip-a-too-small-deck policy).
    """

    inputs: BatchedTrainingInput
    identities: list[tuple[UUID, ...]]
    positive_cliques: list[list[int]]

    def __len__(self) -> int:
        """Example count: one per surviving source deck (positive clique)."""
        return len(self.positive_cliques)

    def __post_init__(self) -> None:
        """Enforce this batch's structural invariants.

        Mirrors Batch._validate_input_structure()'s one-shared-
        InputShape check (via input_shape_of()), plus an equal-length
        inputs/identities check this dataclass adds.

        Inputs: none (runs on the fields already set by __init__).
        Output: none.
        Side effects: none.
        Exceptions: ValueError if inputs and identities aren't the same
            length, if items is non-empty and its elements don't all
            share one InputShape, or if any positive_cliques entry
            indexes outside items.
        """
        if len(self.inputs) != len(self.identities):
            raise ValueError(
                f"inputs ({len(self.inputs)}) and identities "
                f"({len(self.identities)}) must be the same length"
            )
        if self.inputs:
            self._validate_shared_input_shape()
        self._validate_positive_cliques()

    def _validate_shared_input_shape(self) -> None:
        """Raise ValueError unless every item in self.inputs shares one InputShape.

        Private helper - single caller is __post_init__.

        Inputs: none.
        Output: none.
        Side effects: none.
        Exceptions: ValueError on a shape mismatch, or whatever
            input_shape_of() itself raises (e.g. on an empty item).
        """
        first_shape = input_shape_of(self.inputs[0])
        for item in self.inputs:
            if input_shape_of(item) != first_shape:
                raise ValueError(
                    "ContrastiveBatch.inputs must all share one InputShape"
                )

    def _validate_positive_cliques(self) -> None:
        """Raise ValueError if any positive_cliques entry indexes outside items.

        Private helper - single caller is __post_init__.

        Inputs: none.
        Output: none.
        Side effects: none.
        Exceptions: ValueError on an out-of-range index, or on a
            duplicate index within one group.
        """
        item_count = len(self.inputs)
        for group in self.positive_cliques:
            seen_indices: set[int] = set()
            for index in group:
                if not 0 <= index < item_count:
                    raise ValueError(
                        f"positive_cliques index {index} is out of range "
                        f"for {item_count} items"
                    )
                if index in seen_indices:
                    raise ValueError(
                        f"positive_cliques group contains duplicate index {index}"
                    )
                seen_indices.add(index)
