"""Decoder head for the pick-prediction metric.

Multi-group in - [pack_embeddings, pool_embeddings] per training datum -
one logits vector out per datum, one logit per candidate card in that
datum's pack. Rough sketch, pending discussion of the scoring
architecture itself.
"""

from typing import List

import torch
import torch.nn as nn

from src.schema.type_hints import BatchedMultiGroupEmbedding


class PickPredictionDecoderHead(nn.Module):
    """concat each
    candidate with the pool's averaged context, score with a small MLP -
    but pads every pack up to a fixed max_pack_size instead of leaving it
    ragged.

    Because every training datum's output is now the same length, the
    whole batch can be re-stacked into one rectangular
    (batch_size, max_pack_size) tensor here in the decoder head, unlike
    V2 - a direct application of this project's decoder-head rule of
    thumb (re-stack on the way out when the output happens to be uniform).
    A plain nn.CrossEntropyLoss can score the
    whole batch in one call. Padded slots are set to -inf before returning
    so they always lose the softmax regardless of what the loss does with
    the extra positions - not wired up to a dedicated loss file in this
    pass, just noted here as the natural follow-up.

    default max_pack_size=16 matches an MTG draft booster pack.
    """

    def __init__(self, card_embedding_size: int, max_pack_size: int = 16):
        super().__init__()
        self.card_embedding_size = card_embedding_size
        self.max_pack_size = max_pack_size
        self.scorer = nn.Sequential(
            nn.Linear(card_embedding_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, embeddings: BatchedMultiGroupEmbedding) -> torch.Tensor:
        """One (max_pack_size,) logits row per training datum, stacked into
        a single (batch_size, max_pack_size) tensor. Unlike V2, this is
        NOT a List[torch.Tensor] - padding to a fixed size is exactly what
        makes re-stacking possible (and worthwhile) here.
        """
        batch_logits: List[torch.Tensor] = []
        for pack_embeddings, pool_embeddings in embeddings:
            num_pack_cards = len(pack_embeddings)
            if num_pack_cards > self.max_pack_size:
                raise ValueError(
                    f"pack has {num_pack_cards} cards, exceeds max_pack_size={self.max_pack_size}")

            pack = torch.stack(pack_embeddings)  # (num_pack_cards, embedding_dim)
            context = (
                torch.stack(pool_embeddings).mean(dim=0)
                if pool_embeddings
                else torch.zeros(self.card_embedding_size)
            )  # (embedding_dim,)
            context = context.unsqueeze(0).expand(
                num_pack_cards, -1)  # (num_pack_cards, embedding_dim)

            combined = torch.cat([pack, context], dim=-1)  # (num_pack_cards, embedding_dim * 2)
            logits = self.scorer(combined).squeeze(-1)  # (num_pack_cards,)

            # Pad up to max_pack_size with -inf, so padded slots always lose
            # the softmax regardless of what the loss does with them.
            padding = torch.full((self.max_pack_size - num_pack_cards,), float("-inf"))
            batch_logits.append(torch.cat([logits, padding]))

        return torch.stack(batch_logits)  # (batch_size, max_pack_size)


class PickPredictionDecoderHeadV2(nn.Module):
    """BERT-style variant: card embeddings go through learned self-attention
    encoders (one for pool, one for pack) instead of naive mean-pooling and
    concat-then-MLP scoring.

    The pool is collapsed into a single context embedding via a
    self-attention encoder with a learned CLS token (the standard BERT
    pooling trick). The pack is NOT collapsed - self-attention lets each
    candidate's embedding become contextualized by the other candidates in
    the pack, but every candidate keeps its own output row. Collapsing the
    pack to one vector would make it impossible for any fixed-width dense
    layer downstream to ever produce a variable-length (num_pack_cards,)
    output again - a dense layer's output width is baked into its weight
    matrix at construction time, it can't scale with however many
    candidates happen to be in a given pack.

    Scoring is a query/key dot product - the score half of scaled
    dot-product attention, without the value/weighted-sum half: the pool's
    single context embedding becomes a query, each pack candidate's
    contextualized embedding becomes a key, and their dot product is that
    candidate's logit. This naturally produces one logit per candidate no
    matter how many candidates there are, since nothing here has a
    fixed-width output - the exact property we lost if we tried to pool
    the pack too.
    """

    def __init__(self, card_embedding_size: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.card_embedding_size = card_embedding_size

        # card_embedding_size must be divisible by num_heads - a
        # nn.TransformerEncoderLayer constraint (each head gets an equal
        # card_embedding_size / num_heads slice).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_embedding_size,
            nhead=num_heads,
            batch_first=True,
        )
        self.pool_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pack_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learned "start of sequence" token, prepended to the pool before
        # encoding - its output position becomes the pooled context
        # embedding, same trick BERT uses for its [CLS] token.
        self.pool_cls_token = nn.Parameter(torch.randn(1, card_embedding_size))

        self.query_proj = nn.Linear(card_embedding_size, card_embedding_size)
        self.key_proj = nn.Linear(card_embedding_size, card_embedding_size)

    def forward(self, embeddings: BatchedMultiGroupEmbedding) -> List[torch.Tensor]:
        """One logits tensor per training datum, one logit per candidate
        card in that datum's pack - same output contract as
        PickPredictionDecoderHead (V1): ragged across the batch, so this
        stays a List[torch.Tensor] rather than a single stacked tensor.
        """
        results: List[torch.Tensor] = []
        for pack_embeddings, pool_embeddings in embeddings:
            pack = (
                torch.stack(pack_embeddings)
                if pack_embeddings
                else torch.zeros(0, self.card_embedding_size)
            )  # (num_pack_cards, embedding_dim)

            pool_cards = (
                torch.stack(pool_embeddings)
                if pool_embeddings
                else torch.zeros(0, self.card_embedding_size)
            )  # (num_pool_cards, embedding_dim)

            # (num_pool_cards + 1, embedding_dim)
            pool_with_cls = torch.cat([self.pool_cls_token, pool_cards], dim=0)

            # nn.TransformerEncoder expects a batch dim; treat each training
            # datum as its own batch of size 1, then drop it back off.
            pool_contextualized = self.pool_encoder(pool_with_cls.unsqueeze(0)).squeeze(0)
            pooled_context = pool_contextualized[0]  # the CLS token's own output row

            pack_contextualized = self.pack_encoder(
                pack.unsqueeze(0)
            ).squeeze(0)  # (num_pack_cards, embedding_dim)

            query = self.query_proj(pooled_context)  # (embedding_dim,)
            keys = self.key_proj(pack_contextualized)  # (num_pack_cards, embedding_dim)

            logits = (keys @ query) / (self.card_embedding_size ** 0.5)  # (num_pack_cards,)
            results.append(logits)
        return results
