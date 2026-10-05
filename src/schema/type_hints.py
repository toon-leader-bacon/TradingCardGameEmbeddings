"""Shared type aliases for card/embedding shapes used across dojos and models.

Every training input and model output in this project takes one of three
shapes - single card, multi card (an ordered list), or multi group (a list
of card lists) - per src/README.md's model taxonomy. These aliases name
that taxonomy once so dojos, encoder models, and training code all refer
to the same shapes instead of each re-deriving them.
"""

from enum import Enum
from typing import Any, Callable, Iterator, List, Tuple, TypeAlias, Union, cast

import torch

from src.schema.card import GenericCard

# region Inputs
SingleCardInput = GenericCard
BatchedSingleCardInput = List[SingleCardInput]

MultiCardInput = List[GenericCard]
BatchedMultiCardInput = List[MultiCardInput]

MultiGroupInput = List[MultiCardInput]
BatchedMultiGroupInput = List[MultiGroupInput]

TrainingInput = Union[
    SingleCardInput,  # GenericCard
    MultiCardInput,  # List[GenericCard]
    MultiGroupInput,  # List[List[GenericCard]]
]
BatchedTrainingInput = Union[
    BatchedSingleCardInput,  # List[GenericCard]
    BatchedMultiCardInput,  # List[List[GenericCard]]
    BatchedMultiGroupInput,  # List[List[List[GenericCard]]]
]
# endregion Inputs


class InputShape(Enum):
    """Which of the nested-list-of-GenericCard shapes above a value has.

    SingleCardInput/MultiCardInput/MultiGroupInput (and their Batched*
    counterparts one list-level up) are all just plain `list`/`GenericCard`
    at runtime - the type parameters are erased, so `isinstance(x,
    MultiCardInput)` isn't just unhelpful, it's a TypeError (Python refuses
    isinstance against a subscripted generic). Nesting depth is the only
    thing that actually distinguishes these shapes at runtime, so
    `input_shape_of` counts it instead.

    Depth alone is ambiguous at 1 and 2: a batch of single cards looks
    like one multi-card input, and a batch of multi-card inputs looks
    like one multi-group input. Code that knows it holds a batch, such as
    an encoder's forward() or Batch, classifies one example instead:
    batched_input_shape_of().
    """

    SINGLE_CARD = 0
    MULTI_CARD = 1  # also BatchedSingleCardInput - same runtime shape
    MULTI_GROUP = 2  # also BatchedMultiCardInput - same runtime shape
    BATCHED_MULTI_GROUP = 3


def input_shape_of(
    x: Union[TrainingInput, BatchedMultiGroupInput],
) -> InputShape:
    """Classify a value by GenericCard-list nesting depth.

    Inputs: x, a GenericCard optionally nested in 0-3 levels of list.
    Output: the matching InputShape.
    Side effects: none.
    Exceptions: ValueError if x bottoms out in something other than a
        GenericCard, or an empty list is encountered before reaching one
        (depth is then ambiguous).

    Example:
        >>> input_shape_of([some_card, other_card])
        <InputShape.MULTI_CARD: 1>
    """
    depth = 0
    probe: Any = x
    while isinstance(probe, list):
        if not probe:
            raise ValueError("Cannot classify the shape of an empty list input")
        probe = probe[0]
        depth += 1

    if not isinstance(probe, GenericCard):
        raise ValueError(f"Not a valid TrainingInput shape: {x}")
    return InputShape(depth)


def batched_input_shape_of(x: BatchedTrainingInput) -> InputShape:
    """The shape of each example in a batch: the unambiguous reading of x.

    Treats x as a batch, a list of examples, and classifies its first
    example with input_shape_of. So [card, card] is SINGLE_CARD (a batch
    of single cards), not MULTI_CARD, and [[card], [card]] is MULTI_CARD
    (a batch of multi-card inputs). Like Batch, only the first example is
    classified; Batch itself checks that every example shares it.

    Inputs: x (BatchedTrainingInput), a non-empty list of examples.
    Output: InputShape of one example: SINGLE_CARD, MULTI_CARD or
        MULTI_GROUP for a well-formed batch. BATCHED_MULTI_GROUP means x
        is nested one level too deep.
    Side effects: none.
    Exceptions: TypeError if x is not a list (e.g. a bare GenericCard,
        which is not a batch). ValueError if x is empty, or as
        input_shape_of raises for its first example.

    Example:
        >>> batched_input_shape_of([some_card, other_card])
        <InputShape.SINGLE_CARD: 0>
    """
    if not isinstance(x, list):
        raise TypeError(f"A batch must be a list of examples, got {type(x).__name__}")
    if not x:
        raise ValueError("Cannot classify the shape of an empty batch")
    return input_shape_of(x[0])


def iter_cards(x: Union[TrainingInput, BatchedTrainingInput]) -> Iterator[GenericCard]:
    """Yield every GenericCard in a (possibly nested, possibly batched) input.

    Inputs: x, a GenericCard or any depth of list nesting of them.
    Output: iterator of GenericCard, depth-first, in order; duplicates kept.
    Side effects: none.
    Exceptions: none (an empty list yields nothing).

    Example:
        >>> [c.name for c in iter_cards([[card_a], [card_b, card_a]])]
        ['A', 'B', 'A']
    """
    if isinstance(x, GenericCard):
        yield x
        return
    for element in x:
        yield from iter_cards(element)


def map_cards(
    x: TrainingInput, transform: Callable[[GenericCard], GenericCard]
) -> TrainingInput:
    """x rebuilt with transform applied to every card, same nesting and
    order: the shape-preserving counterpart of iter_cards.

    Inputs: x (a GenericCard or list nesting of them), transform.
    Output: TrainingInput of the same shape; new lists, never x's own.
    Side effects: whatever transform does (called once per card, depth
        first, in order).
    Exceptions: whatever transform raises.

    Example:
        >>> map_cards([card_a, card_b], lambda c: c)
        [card_a, card_b]
    """
    if isinstance(x, GenericCard):
        return transform(x)
    # Recursive like iter_cards: the nesting is at most two lists deep
    return cast(TrainingInput, [map_cards(element, transform) for element in x])


Label = Any  # Typically a single scaler value, but could be a list of values
TrainingDatum = Tuple[TrainingInput, Label]


# region Outputs
Embedding: TypeAlias = torch.Tensor

SingleCardEmbedding: TypeAlias = Embedding
BatchedSingleCardEmbedding = List[SingleCardEmbedding]

MultiCardEmbedding = List[Embedding]
BatchedMultiCardEmbedding = List[MultiCardEmbedding]

MultiGroupEmbedding = List[MultiCardEmbedding]
BatchedMultiGroupEmbedding = List[MultiGroupEmbedding]

ModelOutput = Union[SingleCardEmbedding, MultiCardEmbedding, MultiGroupEmbedding]
BatchedModelOutput = Union[
    BatchedSingleCardEmbedding, BatchedMultiCardEmbedding, BatchedMultiGroupEmbedding
]
# endregion Outputs


def output_shape_of(
    x: Union[ModelOutput, BatchedMultiGroupEmbedding],
) -> InputShape:
    """Classify a value by torch.Tensor-list nesting depth.

    Symmetric to input_shape_of, for the output/embedding side of the
    same three-shape taxonomy (bottoms out on torch.Tensor instead of
    GenericCard). Called on one representative element (e.g. one item's
    embedding), the same way input_shape_of is - not on a whole batched
    list, unless that list is itself the thing being classified.

    Inputs: x, a torch.Tensor optionally nested in 0-3 levels of list.
    Output: the matching InputShape.
    Side effects: none.
    Exceptions: ValueError if x bottoms out in something other than a
        torch.Tensor, or an empty list is encountered before reaching one
        (depth is then ambiguous).

    Example:
        >>> output_shape_of([torch.zeros(4), torch.zeros(4)])
        <InputShape.MULTI_CARD: 1>
    """
    depth = 0
    probe: Any = x
    while isinstance(probe, list):
        if not probe:
            raise ValueError("Cannot classify the shape of an empty list output")
        probe = probe[0]
        depth += 1

    if not isinstance(probe, torch.Tensor):
        raise ValueError(f"Not a valid ModelOutput shape: {x}")
    return InputShape(depth)
