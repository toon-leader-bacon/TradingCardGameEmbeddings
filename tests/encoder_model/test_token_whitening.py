"""TokenWhitening: the fitted map on every token, in float32, as buffers."""

import torch

from src.encoder_model.enrichment.token_whitening import TokenWhitening
from src.encoder_model.text_encoder import TokenEncoding
from src.encoder_model.whitening.whitening_stats import WhiteningStats
from src.schema.holdout import HoldoutSpec


def _stats() -> WhiteningStats:
    generator = torch.Generator().manual_seed(0)
    return WhiteningStats(
        checkpoint="ckpt",
        holdout=HoldoutSpec.no_holdout(),
        card_count=5,
        mean=torch.randn(4, generator=generator),
        projection=torch.randn(4, 3, generator=generator),
    )


def _encoding(dtype: torch.dtype = torch.float32) -> TokenEncoding:
    hidden = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(1))
    return TokenEncoding(
        hidden_states=hidden.to(dtype), attention_mask=torch.ones(2, 5)
    )


def test_every_token_is_centered_and_projected() -> None:
    stats = _stats()
    encoding = _encoding()
    whitened = TokenWhitening(stats)(encoding)
    expected = (encoding.hidden_states - stats.mean) @ stats.projection
    assert torch.allclose(whitened.hidden_states, expected)
    assert torch.equal(whitened.attention_mask, encoding.attention_mask)


def test_widths_come_from_the_stats() -> None:
    whitening = TokenWhitening(_stats())
    assert (whitening.input_dim, whitening.output_dim) == (4, 3)


def test_low_precision_tokens_are_whitened_in_float32() -> None:
    whitening = TokenWhitening(_stats())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        whitened = whitening(_encoding(torch.bfloat16))
    assert whitened.hidden_states.dtype == torch.float32


def test_the_fit_is_saved_with_the_model_and_never_trained() -> None:
    whitening = TokenWhitening(_stats())
    assert set(whitening.state_dict()) == {"mean", "projection"}
    assert list(whitening.parameters()) == []
