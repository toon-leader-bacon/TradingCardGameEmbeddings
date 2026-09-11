"""A handful of concrete SingleCardModel presets, each fixing one
TextEncoder + EmbeddingHead pair, to demonstrate how the two Strategy
seams (text_encoder.py, embedding_head.py) mix and match.

These are reference examples, not the primary way to construct a
SingleCardModel - a real experiment should inject a TextEncoder/
EmbeddingHead pair directly, not add a new preset class per combination
tried. SingleCardModel itself still takes both collaborators injected;
these subclasses just fix a specific choice of both in their own
__init__ for convenience/demonstration.
"""

from src.encoder_model.embedding_head import (
    AttentionPoolingEmbeddingHead,
    LinearEmbeddingHead,
    ResidualMlpEmbeddingHead,
)
from src.encoder_model.single_card_model import SingleCardModel
from src.encoder_model.text_encoder import (
    PretrainedTextEncoder,
    StaticEmbeddingTextEncoder,
)

_DEFAULT_PRETRAINED_CHECKPOINT = "distilbert-base-uncased"


class LinearProjectionCardModel(SingleCardModel):
    """Floor baseline: a frozen pretrained text encoder + a single
    linear projection head. Any richer combination should beat this."""

    def __init__(
        self, checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT, embed_dim: int = 256
    ):
        text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
        hidden_dim = text_encoder.model.config.hidden_size
        embedding_head = LinearEmbeddingHead(input_dim=hidden_dim, embed_dim=embed_dim)
        super().__init__(text_encoder=text_encoder, embedding_head=embedding_head)


class ResidualMlpCardModel(SingleCardModel):
    """A frozen pretrained text encoder + a deeper residual-MLP head."""

    def __init__(
        self,
        checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_blocks: int = 3,
    ):
        text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
        encoder_hidden_dim = text_encoder.model.config.hidden_size
        embedding_head = ResidualMlpEmbeddingHead(
            input_dim=encoder_hidden_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        super().__init__(text_encoder=text_encoder, embedding_head=embedding_head)


class AttentionPoolingCardModel(SingleCardModel):
    """A frozen pretrained text encoder + a learned attention-pooling
    head, instead of the other two presets' naive mean pooling."""

    def __init__(
        self, checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT, embed_dim: int = 256
    ):
        text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
        hidden_dim = text_encoder.model.config.hidden_size
        embedding_head = AttentionPoolingEmbeddingHead(
            input_dim=hidden_dim, embed_dim=embed_dim
        )
        super().__init__(text_encoder=text_encoder, embedding_head=embedding_head)


class FromScratchCardModel(SingleCardModel):
    """No pretrained weights anywhere: a from-scratch-trainable static
    token embedding table + a residual-MLP head, both trained purely
    from this project's own dojos. The cheapest fully-from-scratch
    starting point - swap in a richer TextEncoder later for real
    cross-token contextualization."""

    def __init__(
        self,
        tokenizer_checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT,
        token_embedding_dim: int = 256,
        embed_dim: int = 256,
    ):
        text_encoder = StaticEmbeddingTextEncoder(
            tokenizer_checkpoint=tokenizer_checkpoint,
            embedding_dim=token_embedding_dim,
        )
        embedding_head = ResidualMlpEmbeddingHead(
            input_dim=token_embedding_dim, embed_dim=embed_dim
        )
        super().__init__(text_encoder=text_encoder, embedding_head=embedding_head)
