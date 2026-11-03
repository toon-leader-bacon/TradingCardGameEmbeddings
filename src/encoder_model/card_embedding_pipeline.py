"""CardEmbeddingPipeline: cards -> one embedding each, through four injected stages.

text encoder -> optional token enrichment -> token pooling -> embedding
head. SingleCardModel and MultiCardModel each own one; this is the only
place the stages are chained. See README.md.
"""

from typing import Any, List, Sequence

import torch
import torch.nn as nn

from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.embedding_head import EmbeddingHead
from src.encoder_model.enrichment.token_enrichment import TokenEnrichment
from src.encoder_model.feature_cache.frozen_feature_cache import (
    FeatureReservation,
    FrozenFeatureCache,
)
from src.encoder_model.pipeline_cut import PipelineCut, PooledCut, build_pipeline_cut
from src.encoder_model.pooling.token_pooling import TokenPooling
from src.encoder_model.precision import Precision, autocast_for, parameter_device
from src.encoder_model.text_encoder import TextEncoder
from src.schema.card import GenericCard

# Cards per warm() chunk: large enough to amortize a forward pass, small
# enough that one chunk's token states (~150 MB at 384 tokens, fp16) fit
# beside training. The text encoder buckets each chunk by length itself
_WARM_CHUNK_CARDS = 256


class CardEmbeddingPipeline(nn.Module):
    """Every card on its own: serialized to text, encoded to token states,
    optionally enriched, pooled to one vector, projected by the head. No
    card sees another.

    Strategy + dependency injection: all four stages are handed in, so any
    one can be swapped without touching this class or the models.

    With a feature_cache, the stages before the cache's cut run only on
    texts the cache has not seen: for a pooled cut, everything before the
    head; for a token cut, the text encoder through the cut's layer (the
    rest runs live, see pipeline_cut.py). The head always runs. The cache
    is a plain attribute, not a submodule, so state_dict() and
    checkpoints are the same with or without it.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        enrichment: TokenEnrichment | None,
        pooling: TokenPooling,
        embedding_head: EmbeddingHead,
        feature_cache: FrozenFeatureCache[Any] | None = None,
    ) -> None:
        """Inputs: text_encoder, enrichment (None: tokens go straight to
            the pooling), pooling, embedding_head, feature_cache (None:
            every stage runs on every card; its fingerprint's cut point
            says where the pipeline is cut).
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if adjacent widths do not chain:
            text_encoder.output_dim -> enrichment -> pooling.input_dim, and
            pooling.output_dim -> embedding_head.input_dim; or if a
            feature_cache is given while a stage before its cut has a
            trainable parameter (cached features would go stale), or its
            cut is inside a text encoder that cannot be split.

        Example:
            >>> encoder = PretrainedTextEncoder("answerdotai/ModernBERT-base")
            >>> pipeline = CardEmbeddingPipeline(encoder, None, MeanPooling(768),
            ...                         LinearEmbeddingHead(768, 256))
            >>> pipeline.output_dim
            256
        """
        super().__init__()
        # Validate that each stage's output width is the next one's input
        _check_widths_chain(text_encoder, enrichment, pooling, embedding_head)
        # The cut follows the cache; without one, the whole live path
        cut: PipelineCut[Any] = PooledCut(text_encoder, enrichment, pooling)
        if feature_cache is not None:
            cut = build_pipeline_cut(
                feature_cache.fingerprint.cut, text_encoder, enrichment, pooling
            )
            cut.check_frozen()
        self.text_encoder: TextEncoder = text_encoder
        self.enrichment: TokenEnrichment | None = enrichment
        self.pooling: TokenPooling = pooling
        self.embedding_head: EmbeddingHead = embedding_head
        # Plain attribute (not an nn.Module), so it stays out of state_dict
        self._feature_cache: FrozenFeatureCache[Any] | None = feature_cache
        # Plain attribute too: it only refers to the stages above
        self._cut: PipelineCut[Any] = cut

    @property
    def output_dim(self) -> int:
        """The embedding width: the head's output_dim. Inputs: none.
        Output: int. Side effects: none. Exceptions: none."""
        return self.embedding_head.output_dim

    @property
    def feature_cache(self) -> FrozenFeatureCache[Any] | None:
        """The cache of the stages before the cut, or None. Inputs: none.
        Output: FrozenFeatureCache or None. Side effects: none.
        Exceptions: none."""
        return self._feature_cache

    def embed(self, cards: List[GenericCard]) -> torch.Tensor:
        """One embedding per card, same order, each independent of the
        others.

        Inputs: cards (List[GenericCard]), non-empty, from any game.
        Output: (len(cards), output_dim) tensor.
        Side effects: none directly (a trainable text encoder accumulates
            gradients as a normal side effect of autograd). With a cache,
            unseen texts are encoded, and those of unstamped cards
            (GenericCard.cacheable) offered to it.
        Exceptions: whatever the injected stages raise; ValueError from the
            cache if the active autocast is not its compute precision.

        Example:
            >>> pipeline.embed([card_a, card_b]).shape
            torch.Size([2, 256])
        """
        # Serialize every card, then run up to the cut: through the cache
        # when there is one
        texts = [serialize_card_to_json(card) for card in cards]
        cut = self._cut
        if self._feature_cache is None:
            features = cut.encode_before_cut(texts)
        else:
            features = self._feature_cache.features(
                texts,
                cut.encode_before_cut,
                parameter_device(self),
                cacheable=_cacheable_of(cards),
            )
        # The rest of the way to the pooled vector, then the head
        pooled = cut.pool_after_cut(features)
        return self.embedding_head(pooled)

    def warm(
        self,
        cards: Sequence[GenericCard],
        precision: Precision,
        durable: bool | None = True,
    ) -> None:
        """Encode cards into the cache ahead of time, in large chunks, so
        training steps find them; every tier keeps them durably. Does
        nothing without a cache.

        Safe to call from a worker thread while another thread trains: it
        changes no module's train/eval mode, and no_grad and autocast are
        thread-local.

        Inputs: cards (any number, from any game), precision (the autocast
            the cache was built for), durable (True: every tier keeps
            them; None: the cache's training-miss setting, as prefetch
            uses; False: never the disk).
        Output: none.
        Side effects: runs the frozen stages (always in eval mode with a
            cache, see train) on the cards the cache lacks, without
            autograd; the store may admit them.
        Exceptions: ValueError from the cache if precision is not its
            compute precision; whatever the stages or the cache raise.

        Example:
            >>> pipeline.warm(list(binder.all_cards(GameId.GWENT)), "fp16")
        """
        cache = self._feature_cache
        if cache is None:
            return
        device = parameter_device(self)

        # One cache call per chunk; only the cache keeps the features
        with torch.no_grad(), autocast_for(self, precision):
            for start in range(0, len(cards), _WARM_CHUNK_CARDS):
                chunk = cards[start : start + _WARM_CHUNK_CARDS]
                texts = [serialize_card_to_json(card) for card in chunk]
                cache.features(
                    texts,
                    self._cut.encode_before_cut,
                    device,
                    durable=durable,
                    cacheable=_cacheable_of(chunk),
                )

    def reserve(
        self,
        cards: Sequence[GenericCard],
        precision: Precision,
        durable: bool | None = None,
    ) -> FeatureReservation:
        """Pin one batch's cards in the cache's memory tiers, then warm
        them, so they stay until release. An empty reservation without a
        cache. Thread-safe like warm: a prefetch worker calls it while
        another thread trains.

        Inputs: cards (one batch), precision (the cache's compute
            precision), durable (None: the cache's training-miss setting;
            see warm).
        Output: FeatureReservation, to hand to release once the batch's
            step has looked its features up.
        Side effects: see FrozenFeatureCache.reserve; the frozen stages run
            without autograd under precision's autocast.
        Exceptions: see FrozenFeatureCache.reserve.

        Example:
            >>> reservation = pipeline.reserve(batch_cards, "fp16")
        """
        cache = self._feature_cache
        if cache is None:
            return FeatureReservation(())
        device = parameter_device(self)

        # One cache call, so the whole batch is one reservation
        with torch.no_grad(), autocast_for(self, precision):
            texts = [serialize_card_to_json(card) for card in cards]
            result = cache.reserve(
                texts,
                self._cut.encode_before_cut,
                device,
                durable=durable,
                cacheable=_cacheable_of(cards),
            )
        return result

    def release(self, reservation: FeatureReservation) -> None:
        """Remove a reservation's pins. Any thread.

        Inputs: reservation (from this pipeline's reserve). Output: none.
        Side effects: see FrozenFeatureCache.release.
        Exceptions: see FrozenFeatureCache.release; ValueError for a
            non-empty reservation when there is no cache.

        Example:
            >>> pipeline.release(reservation)
        """
        cache = self._feature_cache
        if cache is not None:
            cache.release(reservation)
        elif reservation.keys:
            raise ValueError("a reservation with keys, but this pipeline has no cache")

    def train(self, mode: bool = True) -> "CardEmbeddingPipeline":
        """Set train/eval mode; with a cache, the stages before its cut
        stay in eval, so cached features are always eval-mode features.

        Inputs: mode (bool). Output: self.
        Side effects: sets every submodule's mode, then returns the stages
            before the cache's cut to eval when there is a cache.
        Exceptions: none.

        Example:
            >>> pipeline.train().text_encoder.training
            False
        """
        super().train(mode)
        if self._feature_cache is not None:
            self._cut.keep_cached_stages_in_eval()
        return self


def _cacheable_of(cards: Sequence[GenericCard]) -> list[bool]:
    """Each card's cacheable flag, in order (False: the do-not-cache stamp).

    Private helper - consumers are CardEmbeddingPipeline.embed, warm and
    reserve.
    Inputs: cards. Output: list[bool]. Side effects: none. Exceptions: none.
    """
    return [card.cacheable for card in cards]


def _check_widths_chain(
    text_encoder: TextEncoder,
    enrichment: TokenEnrichment | None,
    pooling: TokenPooling,
    embedding_head: EmbeddingHead,
) -> None:
    """Private helper - single consumer is CardEmbeddingPipeline.__init__.

    Inputs: the four stages, as CardEmbeddingPipeline.__init__.
    Output: none. Side effects: none.
    Exceptions: ValueError naming the first pair of stages whose widths
        differ (e.g. "text encoder output_dim 768 != pooling input_dim 256").
    """
    # Each (producer name, width out, consumer name, width in), in order
    links: list[tuple[str, int, str, int]] = []
    token_source, token_width = "text encoder", text_encoder.output_dim
    if enrichment is not None:
        links.append((token_source, token_width, "enrichment", enrichment.input_dim))
        token_source, token_width = "enrichment", enrichment.output_dim
    links.append((token_source, token_width, "pooling", pooling.input_dim))
    links.append(
        ("pooling", pooling.output_dim, "embedding head", embedding_head.input_dim)
    )

    for producer, width_out, consumer, width_in in links:
        if width_out != width_in:
            raise ValueError(
                f"{producer} output_dim {width_out} != {consumer} input_dim {width_in}"
            )
