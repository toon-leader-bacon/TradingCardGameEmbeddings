"""Shared per-option scoring strategy for the multi_card_option_selection
and multi_group_option_selection generic dojo cells (plans/
seventeen_lands_dojos.md).

Both cells convert a ragged set of pack-option card embeddings into one
logit per option (see PickPredictionCrossEntropyLoss, src/dojos/loss/
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

from src.schema.type_hints import MultiCardEmbedding


@runtime_checkable
class OptionScoringHead(Protocol):
    """Strategy: scores one example's ragged pack-option embeddings,
    optionally conditioned on a single pooled context vector.

    A concrete implementation that holds learnable weights (like
    BilinearOptionScoringHead) must also subclass torch.nn.Module, so
    whichever decoder head owns it - assigning it as a plain attribute
    in __init__, the same way MultiCardBinaryClassificationDecoderHead
    already owns an EmbeddingPooler - registers it as a submodule and
    its weights reach the optimizer. This Protocol itself doesn't
    enforce that (a future parameter-free strategy wouldn't need it),
    the same way EmbeddingPooler's own Protocol doesn't."""

    def score(
        self, option_embeddings: MultiCardEmbedding, context: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Inputs:
            option_embeddings: one example's pack-option card
                embeddings, ragged length (varies example to example,
                never empty - a pick always comes from a non-empty
                pack).
            context: a single pooled conditioning vector (e.g. the
                drafter's pool-so-far, already mean-pooled - see
                multi_group_option_selection/decoder_head.py), or None
                for a cell/example with no conditioning group (e.g.
                every multi_card_option_selection call).
        Output: one 1-D tensor of per-option logits, same length and
            order as option_embeddings.
        Side effects: implementation-defined (expected: none beyond
            ordinary nn.Module forward-pass gradient tracking).
        Exceptions: implementation-defined (expected: none for a
            non-empty option_embeddings - callers never pass an empty
            pack).
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
        # option-context interaction term (added only when a context
        # vector is given) - see score()'s docstring for how the two
        # combine.
        self.option_term = nn.Linear(card_embedding_size, 1, bias=True)
        self.context_projection = nn.Linear(
            card_embedding_size, card_embedding_size, bias=False
        )

    def score(
        self, option_embeddings: MultiCardEmbedding, context: torch.Tensor | None
    ) -> torch.Tensor:
        """self.option_term(option) + option . self.context_projection(context)
        when context is given, else just self.option_term(option) alone,
        for every option in option_embeddings.

        Inputs:
            option_embeddings: see OptionScoringHead.score().
            context: see OptionScoringHead.score() - None for
                multi_card_option_selection (no conditioning group).
        Output: a (len(option_embeddings),) tensor of logits, one per
            option, same order as option_embeddings.
        Side effects: none beyond ordinary autograd tracking.
        Exceptions: none expected for a non-empty option_embeddings.

        Example:
            >>> head = BilinearOptionScoringHead(card_embedding_size=4)
            >>> head.score([torch.randn(4), torch.randn(4)], context=None).shape
            torch.Size([2])
        """
        stacked_options = torch.stack(option_embeddings)
        logits = self.option_term(stacked_options).squeeze(-1)
        if context is not None:
            projected_context = self.context_projection(context)
            logits = logits + stacked_options @ projected_context
        return logits
