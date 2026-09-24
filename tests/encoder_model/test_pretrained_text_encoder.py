from types import SimpleNamespace
from typing import Callable

import pytest
import torch
from torch import nn

from src.encoder_model import text_encoder as module
from src.encoder_model.text_encoder import PretrainedTextEncoder


class _TinyLm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.weight = nn.Parameter(torch.zeros(1))


class _Loader:
    @staticmethod
    def from_pretrained(checkpoint: str) -> object:
        return _TinyLm() if "model" in checkpoint else object()


@pytest.fixture(autouse=True)
def _fake_hugging_face(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "AutoTokenizer", _Loader)
    monkeypatch.setattr(module, "AutoModel", _Loader)


def test_a_frozen_lm_stays_in_eval_when_the_parent_is_set_to_train() -> None:
    encoder = PretrainedTextEncoder("model", trainable=False)
    encoder.train(True)
    assert encoder.model.training is False


def test_a_trainable_lm_follows_the_requested_mode() -> None:
    encoder = PretrainedTextEncoder("model", trainable=True)
    encoder.train(True)
    assert encoder.model.training is True
    encoder.train(False)
    assert encoder.model.training is False


def test_model_kwargs_are_forwarded_to_auto_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs = {}

    class _RecordingAutoModel:
        @staticmethod
        def from_pretrained(checkpoint: str, **kwargs: object) -> object:
            seen_kwargs.update(kwargs)
            return _TinyLm()

    monkeypatch.setattr(module, "AutoModel", _RecordingAutoModel)

    PretrainedTextEncoder("model", model_kwargs={"attn_implementation": "sdpa"})

    assert seen_kwargs == {"attn_implementation": "sdpa"}


def test_model_kwargs_default_to_none_forwarded_as_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs = {"sentinel": "untouched"}

    class _RecordingAutoModel:
        @staticmethod
        def from_pretrained(checkpoint: str, **kwargs: object) -> object:
            seen_kwargs.clear()
            seen_kwargs.update(kwargs)
            return _TinyLm()

    monkeypatch.setattr(module, "AutoModel", _RecordingAutoModel)

    PretrainedTextEncoder("model")

    assert seen_kwargs == {}


# --- fakes for the length-bucketing tests below: a real enough tokenizer
# and model to exercise encode() end to end without a network call. Token
# j of a text is id j (clamped to max_length), so a real model's fake
# per-token output can be checked against known values.


class _FakeBatchEncoding(dict):
    def to(self, device: torch.device) -> "_FakeBatchEncoding":
        return _FakeBatchEncoding(
            {key: value.to(device) for key, value in self.items()}
        )


class _FakeTokenizer:
    def __call__(
        self,
        texts: list,
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> _FakeBatchEncoding:
        lengths = [min(len(text), max_length or len(text)) for text in texts]
        width = max(lengths) if lengths else 0
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), width), dtype=torch.long)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(length)
            attention_mask[row, :length] = 1
        return _FakeBatchEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )


class _FakeModel(nn.Module):
    """last_hidden_state's single feature is each token's own id, scaled by
    a trainable parameter (1.0 initially, so it doesn't change values in
    tests that don't care about gradients)."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask  # masking is exercised via the returned mask, not this fake
        return SimpleNamespace(
            last_hidden_state=input_ids.unsqueeze(-1).float() * self.scale
        )


def _make_encoder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    length_bucket_size: int | None,
    trainable: bool = False
) -> PretrainedTextEncoder:
    monkeypatch.setattr(
        module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda c: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        module,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda c, **kw: _FakeModel()),
    )
    return PretrainedTextEncoder(
        "fake",
        trainable=trainable,
        max_length=100,
        length_bucket_size=length_bucket_size,
    )


def _counting(encoder: PretrainedTextEncoder) -> tuple[Callable[[list], object], list]:
    """Wraps encoder._encode_batch to record each call's batch size."""
    calls: list = []
    original = encoder._encode_batch

    def counted(texts: list) -> object:
        calls.append(len(texts))
        return original(texts)

    encoder._encode_batch = counted  # type: ignore[method-assign]
    return counted, calls


class TestLengthBucketing:
    def test_batch_at_or_under_bucket_size_is_encoded_in_one_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        encoder = _make_encoder(monkeypatch, length_bucket_size=8)
        _, calls = _counting(encoder)

        encoder.encode(["a", "bb", "ccc"])

        assert calls == [3]

    def test_larger_batch_splits_into_sorted_length_buckets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        encoder = _make_encoder(monkeypatch, length_bucket_size=2)
        _, calls = _counting(encoder)

        encoder.encode(["a", "bb", "ccc", "dddd", "e"])  # 5 texts -> chunks of 2, 2, 1

        assert calls == [2, 2, 1]

    def test_bucket_size_none_never_splits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        encoder = _make_encoder(monkeypatch, length_bucket_size=None)
        _, calls = _counting(encoder)

        encoder.encode(["a" * i for i in range(1, 30)])

        assert calls == [29]

    def test_bucketed_and_unbucketed_encode_produce_the_same_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        texts = ["ccc", "a", "dddddd", "bb", "eeeeeeeee", "f", "gg", "hhhh"]
        bucketed = _make_encoder(monkeypatch, length_bucket_size=3)
        unbucketed = _make_encoder(monkeypatch, length_bucket_size=None)

        bucketed_result = bucketed.encode(texts)
        unbucketed_result = unbucketed.encode(texts)

        assert torch.equal(
            bucketed_result.attention_mask, unbucketed_result.attention_mask
        )
        assert torch.equal(
            bucketed_result.hidden_states, unbucketed_result.hidden_states
        )

    def test_original_row_order_and_content_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        texts = ["ccc", "a", "dddd", "bb"]  # unsorted lengths: 3, 1, 4, 2
        encoder = _make_encoder(monkeypatch, length_bucket_size=2)

        result = encoder.encode(texts)

        assert result.hidden_states.shape == (4, 4, 1)  # padded to the longest text (4)
        for row, text in enumerate(texts):
            real_positions = result.attention_mask[row].nonzero().flatten()
            assert len(real_positions) == len(text)
            assert torch.equal(
                result.hidden_states[row, real_positions, 0],
                torch.arange(len(text)).float(),
            )
            assert result.hidden_states[row, len(text) :, 0].eq(0).all()

    def test_padding_beyond_each_row_is_masked_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        encoder = _make_encoder(
            monkeypatch, length_bucket_size=1
        )  # every row its own bucket

        result = encoder.encode(["a", "bbbbb"])

        assert result.attention_mask.tolist() == [[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]

    def test_gradients_reach_the_model_through_bucketed_reassembly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        encoder = _make_encoder(monkeypatch, length_bucket_size=2, trainable=True)

        result = encoder.encode(["a", "bb", "ccc", "dddd", "e"])
        result.hidden_states.sum().backward()

        assert encoder.model.scale.grad is not None
        assert encoder.model.scale.grad.item() != 0

    def test_truncation_still_applies_per_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            module,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=lambda c: _FakeTokenizer()),
        )
        monkeypatch.setattr(
            module,
            "AutoModel",
            SimpleNamespace(from_pretrained=lambda c, **kw: _FakeModel()),
        )
        encoder = PretrainedTextEncoder("fake", max_length=3, length_bucket_size=1)

        result = encoder.encode(["a", "bbbbbbbbbb"])

        assert result.attention_mask[1].sum().item() == 3  # truncated to max_length
