"""Shared per-option scoring strategy for the multi_card_option_selection
and multi_group_option_selection generic dojo cells (see
src/dojos/generic/README.md).

Both cells convert a padded batch of pack-option card embeddings into
one logit per option (see PickPredictionCrossEntropyLoss, src/dojos/loss/
pick_prediction_cross_entropy_loss.py - the loss both cells consume
unchanged). How an option's embedding becomes its logit - and how much
it's allowed to be nudged by an optional pooled conditioning context
(the drafter's pool-so-far, for multi_group_option_selection only) - is
deliberately a swappable Strategy (PATTERNS.md), not baked into either
cell's decoder head: a low-capacity scoring function forces more of the
"why is this card good here" reasoning into the shared card embeddings
themselves (closer to a linear-probe read of embedding quality); a
high-capacity one (e.g. full self-attention across the pack) can solve
the task by absorbing that reasoning into its own weights instead,
leaving embeddings comparatively under-constrained. Swapping capacity
for either metric later should mean swapping this one collaborator,
not rewriting either cell - mirrors src/dojos/generic/pooling.py's
EmbeddingPooler seam.

BilinearOptionScoringHead is the one concrete implementation shipped
today - a compatibility score between each option and its (optional)
context, low-capacity enough to still require real embedding structure
to solve the task, unlike a full attention stack. A future
self-attention or plain-per-option-MLP implementation is a new class in
this same file, not a change to either cell or its owning dojo.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class OptionScoringHead(Protocol):
    """Strategy: scores a padded batch of pack-option embeddings, each
    example optionally conditioned on one pooled context vector.

    A concrete implementation that holds learnable weights (like
    BilinearOptionScoringHead) must also subclass torch.nn.Module, so
    whichever decoder head owns it - assigning it as a plain attribute
    in __init__, the same way MultiCardBinaryClassificationDecoderHead
    already owns an EmbeddingPooler - registers it as a submodule and
    its weights reach the optimizer. This Protocol itself doesn't
    enforce that (a future parameter-free strategy wouldn't need it),
    the same way EmbeddingPooler's own Protocol doesn't."""

    def score(
        self,
        options: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Inputs:
            options: (examples, max_options, d) option embeddings,
                left-aligned; every example has at least one real option
                (a pick always comes from a non-empty pack).
            mask: (examples, max_options) bool, True = real option.
            context: (examples, d) pooled conditioning vectors (e.g. the
                drafter's pool-so-far), or None when no example is
                conditioned (every multi_card_option_selection call).
            context_mask: (examples,) bool, True = that example's context
                row is real. None = every row is real. A False row is
                scored exactly as if context were None for that example
                (e.g. a first pick with an empty pool), whatever values
                the row holds. Ignored when context is None.
        Output: (examples, max_options) logits; padding slots are -inf.
        Side effects: implementation-defined (expected: none beyond
            ordinary nn.Module forward-pass gradient tracking).
        Exceptions: implementation-defined.
        """
        ...


class BilinearOptionScoringHead(nn.Module):
    """Default OptionScoringHead: a bilinear compatibility score between
    each option and its (optional) pooled context, plus a plain linear
    option-only term so an option still gets a meaningful score with no
    context at all (multi_card_option_selection's use case). See this
    module's docstring for why bilinear - not a full attention stack -
    is the chosen default capacity."""

    def __init__(self, card_embedding_size: int) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive (both option and context
                vectors share this width).
        Output: none (constructor).
        Side effects: registers this module's layers (nn.Module state).
        Exceptions: none expected.
        """
        super().__init__()
        # A per-option linear term (always applied) plus a bilinear
        # option-context interaction term (added only where a context
        # row is real) - see score()'s docstring for how the two combine.
        self.option_term = nn.Linear(card_embedding_size, 1, bias=True)
        self.context_projection = nn.Linear(
            card_embedding_size, card_embedding_size, bias=False
        )

    def score(
        self,
        options: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """self.option_term(option) + option . self.context_projection(context)
        where the example's context is real, else self.option_term(option)
        alone, for every option; -inf at padding.

        Inputs: see OptionScoringHead.score().
        Output: (examples, max_options) logits, -inf at padding.
        Side effects: none beyond ordinary autograd tracking.
        Exceptions: none expected.

        Example:
            >>> head = BilinearOptionScoringHead(card_embedding_size=4)
            >>> head.score(torch.randn(2, 3, 4), mask, None, None).shape
            torch.Size([2, 3])
        """
        logits = self.option_term(options).squeeze(-1)  # (examples, max_options)

        # Add the bilinear term only on examples whose context is real
        if context is not None:
            interaction = torch.einsum(
                "eod,ed->eo", options, self.context_projection(context)
            )
            logits = logits + _masked_interaction(interaction, context_mask)

        # Padding can never be picked
        return logits.masked_fill(~mask, float("-inf"))


def _masked_interaction(
    interaction: torch.Tensor, context_mask: torch.Tensor | None
) -> torch.Tensor:
    """interaction with the rows of unreal contexts zeroed.

    Private helper - single caller is BilinearOptionScoringHead.score.
    Inputs: interaction (examples, max_options); context_mask (examples,)
        bool or None (every row real).
    Output: (examples, max_options); a False row is exactly 0, so neither
        its value nor its gradient depends on that context row. Uses
        torch.where, not multiplication, so a non-finite garbage row
        cannot leak NaN.
    Side effects: none. Exceptions: none.
    """
    if context_mask is None:
        return interaction
    return torch.where(
        context_mask.unsqueeze(1), interaction, torch.zeros_like(interaction)
    )
