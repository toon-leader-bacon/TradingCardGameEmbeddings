"""A handful of concrete MultiCardModel presets, each fixing one
TextEncoder + EmbeddingHead pair, to demonstrate how the two Strategy
seams (text_encoder.py, embedding_head.py) mix and match for the
self-attention-over-a-card-group model - mirrors reference_models.py's
presets for SingleCardModel.

These are reference examples, not the primary way to construct a
MultiCardModel - a real experiment should inject a TextEncoder/
EmbeddingHead pair directly, not add a new preset class per combination
tried. MultiCardModel itself still takes both collaborators injected;
these subclasses just fix a specific choice of both (plus the
self-attention sizing) in their own __init__ for convenience/
demonstration.
"""

from src.encoder_model.embedding_head import (
    AttentionPoolingEmbeddingHead,
    LinearEmbeddingHead,
    ResidualMlpEmbeddingHead,
)
from src.encoder_model.multi_card_model import MultiCardModel
from src.encoder_model.text_encoder import (
    PretrainedTextEncoder,
    StaticEmbeddingTextEncoder,
)

_DEFAULT_PRETRAINED_CHECKPOINT = "distilbert-base-uncased"


class LinearProjectionMultiCardModel(MultiCardModel):
    """Floor baseline: a frozen pretrained text encoder + a single
    linear projection head feeding the self-attention group encoder."""

    def __init__(
        self,
        checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
        hidden_dim = text_encoder.model.config.hidden_size
        embedding_head = LinearEmbeddingHead(
            input_dim=hidden_dim, embed_dim=card_embedding_size
        )
        super().__init__(
            text_encoder=text_encoder,
            embedding_head=embedding_head,
            card_embedding_size=card_embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )


class ResidualMlpMultiCardModel(MultiCardModel):
    """A frozen pretrained text encoder + a deeper residual-MLP head
    feeding the self-attention group encoder."""

    def __init__(
        self,
        checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
        encoder_hidden_dim = text_encoder.model.config.hidden_size
        embedding_head = ResidualMlpEmbeddingHead(
            input_dim=encoder_hidden_dim,
            embed_dim=card_embedding_size,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        super().__init__(
            text_encoder=text_encoder,
            embedding_head=embedding_head,
            card_embedding_size=card_embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )


class AttentionPoolingMultiCardModel(MultiCardModel):
    """A frozen pretrained text encoder + a learned attention-pooling
    head, instead of the other presets' naive mean pooling, feeding the
    self-attention group encoder."""

    def __init__(
        self,
        checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
        hidden_dim = text_encoder.model.config.hidden_size
        embedding_head = AttentionPoolingEmbeddingHead(
            input_dim=hidden_dim, embed_dim=card_embedding_size
        )
        super().__init__(
            text_encoder=text_encoder,
            embedding_head=embedding_head,
            card_embedding_size=card_embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )


class FromScratchMultiCardModel(MultiCardModel):
    """No pretrained weights anywhere: a from-scratch-trainable static
    token embedding table + a residual-MLP head feeding the
    self-attention group encoder, both trained purely from this
    project's own dojos. The cheapest fully-from-scratch starting
    point - swap in a richer TextEncoder later for real cross-token
    contextualization."""

    def __init__(
        self,
        tokenizer_checkpoint: str = _DEFAULT_PRETRAINED_CHECKPOINT,
        token_embedding_dim: int = 256,
        card_embedding_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        text_encoder = StaticEmbeddingTextEncoder(
            tokenizer_checkpoint=tokenizer_checkpoint,
            embedding_dim=token_embedding_dim,
        )
        embedding_head = ResidualMlpEmbeddingHead(
            input_dim=token_embedding_dim, embed_dim=card_embedding_size
        )
        super().__init__(
            text_encoder=text_encoder,
            embedding_head=embedding_head,
            card_embedding_size=card_embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )
