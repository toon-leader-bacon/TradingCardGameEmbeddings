"""TokenPooling strategies: widths, padding, and each pooling's own rule."""

import pytest
import torch

from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.pooling.token_pooling import TokenPooling
from src.encoder_model.pooling.transformer_pooling import TransformerPooling
from src.encoder_model.text_encoder import TokenEncoding

WIDTH = 8


def _encoding(seed: int = 0) -> TokenEncoding:
    """Two rows: the first has 3 real tokens of 4, the second all 4."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, 4, WIDTH, generator=generator)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    return TokenEncoding(hidden_states=hidden, attention_mask=mask)


def _with_new_padding(encoding: TokenEncoding, value: float) -> TokenEncoding:
    """The same encoding with value at its padding position."""
    hidden = encoding.hidden_states.clone()
    hidden[0, 3] = value
    return TokenEncoding(hidden_states=hidden, attention_mask=encoding.attention_mask)


def _poolings() -> list[TokenPooling]:
    torch.manual_seed(0)
    return [
        MeanPooling(WIDTH),
        AttentionPooling(WIDTH, num_queries=1),
        AttentionPooling(WIDTH, num_queries=3),
        TransformerPooling(WIDTH, inner=MeanPooling(WIDTH), num_heads=2).eval(),
        TransformerPooling(WIDTH, inner=AttentionPooling(WIDTH, 2), num_heads=2).eval(),
    ]


@pytest.mark.parametrize("pooling", _poolings())
def test_every_pooling_returns_one_vector_of_its_output_dim_per_row(
    pooling: TokenPooling,
) -> None:
    assert pooling.input_dim == WIDTH
    assert pooling(_encoding()).shape == (2, pooling.output_dim)


@pytest.mark.parametrize("value", [1000.0, float("nan"), float("inf")])
@pytest.mark.parametrize("pooling", _poolings())
def test_every_pooling_ignores_padding_whatever_it_holds(
    pooling: TokenPooling, value: float
) -> None:
    encoding = _encoding()
    with torch.no_grad():
        before = pooling(encoding)
        after = pooling(_with_new_padding(encoding, value))
    assert torch.allclose(before, after, atol=1e-5)


@pytest.mark.parametrize("pooling", _poolings())
def test_every_pooling_runs_under_bf16_autocast(pooling: TokenPooling) -> None:
    encoding = _encoding()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        pooled = pooling(encoding)
    assert pooled.shape == (2, pooling.output_dim)
    assert torch.isfinite(pooled).all()


def test_mean_pooling_averages_the_real_tokens() -> None:
    encoding = _encoding()
    pooled = MeanPooling(WIDTH)(encoding)
    assert torch.allclose(pooled[0], encoding.hidden_states[0, :3].mean(dim=0))
    assert torch.allclose(pooled[1], encoding.hidden_states[1].mean(dim=0))


def test_attention_pooling_concatenates_one_average_per_query() -> None:
    assert AttentionPooling(WIDTH, num_queries=4).output_dim == 4 * WIDTH


def test_attention_pooling_of_identical_tokens_is_that_token() -> None:
    token = torch.randn(WIDTH)
    encoding = TokenEncoding(
        hidden_states=token.expand(1, 3, WIDTH).clone(),
        attention_mask=torch.ones(1, 3),
    )
    pooled = AttentionPooling(WIDTH, num_queries=2)(encoding)
    assert torch.allclose(pooled, torch.cat([token, token]).unsqueeze(0), atol=1e-5)


def test_attention_pooling_learns_its_queries() -> None:
    pooling = AttentionPooling(WIDTH, num_queries=2)
    pooling(_encoding()).sum().backward()
    assert pooling.queries.grad is not None


def test_transformer_pooling_takes_its_inner_pooling_width() -> None:
    inner = AttentionPooling(WIDTH, num_queries=3)
    assert TransformerPooling(WIDTH, inner=inner, num_heads=2).output_dim == 3 * WIDTH


def test_transformer_pooling_trains_its_layers() -> None:
    pooling = TransformerPooling(WIDTH, inner=MeanPooling(WIDTH), num_heads=2)
    pooling(_encoding()).sum().backward()
    assert all(p.grad is not None for p in pooling.layers.parameters())


@pytest.mark.parametrize(
    "build",
    [
        lambda: TransformerPooling(WIDTH, inner=MeanPooling(WIDTH + 1), num_heads=2),
        lambda: TransformerPooling(WIDTH, inner=MeanPooling(WIDTH), num_heads=3),
        lambda: TransformerPooling(WIDTH, inner=MeanPooling(WIDTH), num_layers=0),
        lambda: AttentionPooling(WIDTH, num_queries=0),
        lambda: MeanPooling(0),
    ],
)
def test_impossible_sizes_are_refused(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]
