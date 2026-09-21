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
