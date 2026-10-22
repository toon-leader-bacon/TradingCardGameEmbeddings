"""PaddedEmbeddings: the model's output for a batch of single-card or
multi-card examples, as one padded tensor plus a mask.

Both src/encoder_model/ (which builds it) and src/dojos/ (whose heads and
losses read it) use it, so it lives in src/schema/ beside type_hints.py.

Layout invariants every consumer may rely on:

- cards is (examples, max_cards, d); mask is (examples, max_cards) bool,
  True for a real card.
- Left-aligned, in input order: example b's cards sit at positions
  0 .. lengths[b] - 1, padding only at the end of a row. Styles that read
  a fixed position rely on this (the anchor first, the intruder last).
- Padding slots are exactly zero, so a sum over max_cards needs no mask.
- Row-major flattening: cards[mask] lists every real card in input order,
  the order iter_cards(inputs) yields.
- max_cards is the longest example in this batch, never a fixed constant;
  a single-card batch has max_cards == 1.
- An example may have zero cards (an all-False mask row, all-zero cards
  row).
- lengths lives on the host, so consumers check sizes and pick positions
  without a device sync. They never read a length back from mask.
"""

from dataclasses import dataclass
from typing import Sequence, Union, cast

import torch

from src.schema.type_hints import (
    BatchedMultiCardEmbedding,
    BatchedMultiGroupEmbedding,
    BatchedSingleCardEmbedding,
    InputShape,
)

_PADDED_SHAPES = (InputShape.SINGLE_CARD, InputShape.MULTI_CARD)


@dataclass(frozen=True)
class PaddedEmbeddings:
    """One batch's card embeddings, padded to the longest example.

    cards: (examples, max_cards, d); padding slots are 0.
    mask: (examples, max_cards) bool on cards' device, True = real card.
    shape: SINGLE_CARD or MULTI_CARD, what each example is.
    lengths: real cards per example, host ints; len == examples.
    """

    cards: torch.Tensor
    mask: torch.Tensor
    shape: InputShape
    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check the fields agree on every size the layout fixes.

        Inputs: the dataclass fields.
        Output: none.
        Side effects: none. Reads only tensor sizes, never values, so no
            device sync.
        Exceptions: ValueError if shape is not SINGLE_CARD or MULTI_CARD,
            if cards is not 3-D, if mask's size is not cards' first two
            dims, if len(lengths) differs from the example count, if a
            length is negative or over max_cards, or if a SINGLE_CARD
            batch has max_cards != 1.
        """
        # Validate the shape and the tensors' ranks
        _require_padded_shape(self.shape)
        if self.cards.dim() != 3:
            raise ValueError(f"cards must be 3-D, got shape {tuple(self.cards.shape)}")
        examples, max_cards = self.cards.shape[0], self.cards.shape[1]
        if tuple(self.mask.shape) != (examples, max_cards):
            raise ValueError(
                f"mask shape {tuple(self.mask.shape)} does not match cards "
                f"{(examples, max_cards)}"
            )

        # Validate the host lengths against the padded width
        if len(self.lengths) != examples:
            raise ValueError(f"{len(self.lengths)} lengths for {examples} examples")
        if any(length < 0 or length > max_cards for length in self.lengths):
            raise ValueError(f"lengths {self.lengths} out of range 0..{max_cards}")
        if self.shape is InputShape.SINGLE_CARD and examples and max_cards != 1:
            raise ValueError(f"a SINGLE_CARD batch has max_cards 1, got {max_cards}")

    @property
    def width(self) -> int:
        """d, the width of one card embedding.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return int(self.cards.shape[2])

    def __len__(self) -> int:
        """The example count (len(lengths)).

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return len(self.lengths)

    def require_shape(self, expected: InputShape, reader: str) -> None:
        """Raise unless this batch has the shape its reader expects.

        A head reading cards[:, 0] from a MULTI_CARD batch would silently
        use each example's first card; this host-side check (no device
        sync) catches that.
        Inputs: expected (SINGLE_CARD or MULTI_CARD), reader (named in
            the message).
        Output: none. Side effects: none.
        Exceptions: ValueError if self.shape is not expected.

        Example:
            >>> embeddings.require_shape(InputShape.SINGLE_CARD, "MyHead")
        """
        if self.shape is not expected:
            raise ValueError(
                f"{reader} expects {expected.name} embeddings, got {self.shape.name}"
            )

    def has_cards(self) -> torch.Tensor:
        """Which examples hold at least one real card.

        Inputs: none.
        Output: (examples,) bool on cards' device, built from host lengths
            (no device sync).
        Side effects: one host-to-device copy. Exceptions: none.

        Example:
            >>> PaddedEmbeddings.from_nested([[a], []], InputShape.MULTI_CARD, 8).has_cards()
            tensor([ True, False])
        """
        return torch.tensor(
            [length > 0 for length in self.lengths],
            dtype=torch.bool,
            device=self.cards.device,
        )

    def mean_per_example(self) -> torch.Tensor:
        """Each example's mean card embedding.

        Inputs: none.
        Output: (examples, d): cards summed over max_cards (padding is 0)
            and divided by max(length, 1). An empty example gives zeros;
            a head that must treat "empty" specially checks lengths.
        Side effects: none beyond autograd.
        Exceptions: none.

        Example:
            >>> PaddedEmbeddings.from_nested(
            ...     [[torch.ones(2), torch.zeros(2)]], InputShape.MULTI_CARD, 2
            ... ).mean_per_example()
            tensor([[0.5000, 0.5000]])
        """
        # Divisors come from host lengths, clamped so an empty row gives 0
        divisors = torch.tensor(
            [max(length, 1) for length in self.lengths],
            dtype=self.cards.dtype,
            device=self.cards.device,
        )
        return self.cards.sum(dim=1) / divisors.unsqueeze(1)

    def flat_cards(self) -> torch.Tensor:
        """Every real card, in input order (row-major).

        Inputs: none. Output: (sum(lengths), d), cards[mask].
        Side effects: none beyond autograd. Exceptions: none.

        Example:
            >>> embeddings.flat_cards().shape
            torch.Size([5, 256])
        """
        return self.cards[self.mask]

    def example_of_card(self) -> torch.Tensor:
        """Which example each flat card belongs to, built from host lengths.

        Inputs: none.
        Output: (sum(lengths),) long on cards' device, parallel to
            flat_cards().
        Side effects: none. Exceptions: none.

        Example:
            >>> embeddings.lengths, embeddings.example_of_card()
            ((2, 0, 1), tensor([0, 0, 2]))
        """
        counts = torch.tensor(self.lengths, dtype=torch.long)
        examples = torch.arange(len(self.lengths), dtype=torch.long)
        return torch.repeat_interleave(examples, counts).to(self.cards.device)

    @staticmethod
    def from_single_cards(units: torch.Tensor) -> "PaddedEmbeddings":
        """A SINGLE_CARD batch whose example i is units[i].

        For composite contrastive losses, which hand their inner loss one
        unit per item, and for builders of single-card stand-ins.
        Inputs: units, (examples, d).
        Output: PaddedEmbeddings with cards (examples, 1, d), an all-True
            mask and every length 1.
        Side effects: none beyond autograd (cards is a view of units).
        Exceptions: ValueError if units is not 2-D.

        Example:
            >>> PaddedEmbeddings.from_single_cards(torch.randn(3, 8)).lengths
            (1, 1, 1)
        """
        if units.dim() != 2:
            raise ValueError(f"units must be 2-D, got shape {tuple(units.shape)}")
        count = units.shape[0]
        mask = torch.ones((count, 1), dtype=torch.bool, device=units.device)
        return PaddedEmbeddings(
            units.unsqueeze(1), mask, InputShape.SINGLE_CARD, (1,) * count
        )

    @staticmethod
    def from_nested(
        embeddings: Union[BatchedSingleCardEmbedding, BatchedMultiCardEmbedding],
        shape: InputShape,
        width: int,
        device: torch.device | None = None,
    ) -> "PaddedEmbeddings":
        """Pack per-card tensors into the padded layout.

        Used by multi-group heads to batch one group across examples, by
        the stand-in builders, and by tests.
        Inputs:
            embeddings: one (d,) tensor per example (SINGLE_CARD), or one
                list of (d,) tensors per example (MULTI_CARD); examples
                may be empty lists.
            shape: SINGLE_CARD or MULTI_CARD, how to read embeddings.
            width: d, used when there is no card at all to read it from.
            device: where to build an all-empty batch; ignored otherwise
                (the cards' own device is used). None means the CPU.
        Output: PaddedEmbeddings meeting the module's invariants.
        Side effects: none beyond autograd (one stack, one scatter).
        Exceptions: ValueError if shape is not SINGLE_CARD or MULTI_CARD,
            or a tensor's width is not width.

        Example:
            >>> PaddedEmbeddings.from_nested(
            ...     [[a, b], [], [c]], InputShape.MULTI_CARD, 8
            ... ).lengths
            (2, 0, 1)
        """
        # Validate inputs
        _require_padded_shape(shape)

        # Read each example as a list of cards, and record its length
        rows = _rows_of(embeddings, shape)
        lengths = tuple(len(row) for row in rows)
        flat = [card for row in rows for card in row]

        # Scatter the stacked cards into a zeroed padded tensor
        target = flat[0].device if flat else (device or torch.device("cpu"))
        mask = mask_from_lengths(lengths, target)
        cards = _zeros_like_mask(mask, width, flat)
        if flat:
            cards[mask] = torch.stack(flat)
        return PaddedEmbeddings(cards, mask, shape, lengths)

    def to_nested(self) -> Union[BatchedSingleCardEmbedding, BatchedMultiCardEmbedding]:
        """The inverse of from_nested: per-card tensors nested by shape.

        For multi-group regrouping in CardEncoderModel.forward, and tests.
        Inputs: none.
        Output: SINGLE_CARD -> one (d,) tensor per example; MULTI_CARD ->
            one list of (d,) tensors per example (empty for an empty
            example). Views into cards, so gradients flow back.
        Side effects: none beyond autograd. Exceptions: none.

        Example:
            >>> [len(row) for row in padded.to_nested()]
            [2, 0, 1]
        """
        rows = [
            list(self.cards[example, :length].unbind(0))
            for example, length in enumerate(self.lengths)
        ]
        if self.shape is InputShape.SINGLE_CARD:
            return [row[0] for row in rows]
        return rows


# What CardEncoderModel.forward returns and Dojo.compute_loss reads. The
# multi-group arm stays nested until multi-group dojos get a padded type
# of their own. Defined here, not in type_hints.py, because this module
# already imports type_hints.
ModelEmbeddings = Union[PaddedEmbeddings, BatchedMultiGroupEmbedding]


def mask_from_lengths(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    """The left-aligned mask for these host lengths.

    Shared by PaddedEmbeddings.from_nested and CardEncoderModel.forward.
    Inputs: lengths (host ints, >= 0), device.
    Output: (len(lengths), max(lengths, default 0)) bool on device; row b
        is True at positions 0 .. lengths[b] - 1. Built on the CPU and
        moved once.
    Side effects: one host-to-device copy. Exceptions: none.

    Example:
        >>> mask_from_lengths([2, 0, 1], torch.device("cpu"))
        tensor([[ True,  True], [False, False], [ True, False]])
    """
    max_cards = max(lengths, default=0)
    positions = torch.arange(max_cards).unsqueeze(0)
    counts = torch.tensor(list(lengths), dtype=torch.long).unsqueeze(1)
    return (positions < counts).to(device)


def _require_padded_shape(shape: InputShape) -> None:
    """Private helper - callers are __post_init__ and from_nested.
    Inputs: shape. Output: none. Side effects: none.
    Exceptions: ValueError unless shape is SINGLE_CARD or MULTI_CARD.
    """
    if shape not in _PADDED_SHAPES:
        raise ValueError(
            f"PaddedEmbeddings holds SINGLE_CARD or MULTI_CARD, not {shape.name}"
        )


def _rows_of(
    embeddings: Union[BatchedSingleCardEmbedding, BatchedMultiCardEmbedding],
    shape: InputShape,
) -> list[list[torch.Tensor]]:
    """Private helper - single caller is from_nested.
    Inputs: embeddings, shape (already checked).
    Output: one list of cards per example; a SINGLE_CARD example becomes
        a one-card list.
    Side effects: none. Exceptions: none.
    """
    if shape is InputShape.SINGLE_CARD:
        return [[card] for card in cast(BatchedSingleCardEmbedding, embeddings)]
    return [list(row) for row in cast(BatchedMultiCardEmbedding, embeddings)]


def _zeros_like_mask(
    mask: torch.Tensor, width: int, flat: list[torch.Tensor]
) -> torch.Tensor:
    """Private helper - single caller is from_nested.
    Inputs: mask, width, flat (to take the dtype from; empty -> float32).
    Output: (examples, max_cards, width) zeros on mask's device.
    Side effects: none.
    Exceptions: ValueError if the first flat card's width is not width.
    """
    # torch.stack rejects mixed widths, so checking one card suffices
    if flat and tuple(flat[0].shape) != (width,):
        raise ValueError(
            f"a card embedding has shape {tuple(flat[0].shape)}, want ({width},)"
        )
    dtype = flat[0].dtype if flat else torch.float32
    return torch.zeros((*mask.shape, width), dtype=dtype, device=mask.device)
