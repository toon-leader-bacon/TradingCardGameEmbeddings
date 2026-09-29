import contextlib

import pytest
import torch
from torch import nn

from src.encoder_model.precision import autocast_for, device_type_of


def test_device_type_is_the_first_parameters_device() -> None:
    assert device_type_of(nn.Linear(2, 2)) == "cpu"


def test_a_model_without_parameters_reports_cpu() -> None:
    assert device_type_of(nn.ReLU()) == "cpu"


def test_fp32_is_a_no_op_context() -> None:
    assert isinstance(autocast_for(nn.Linear(2, 2), "fp32"), contextlib.nullcontext)


@pytest.mark.parametrize(
    ("precision", "dtype"), [("bf16", torch.bfloat16), ("fp16", torch.float16)]
)
def test_lower_precisions_autocast_on_the_models_device(
    precision: str, dtype: torch.dtype
) -> None:
    context = autocast_for(nn.Linear(2, 2), precision)  # type: ignore[arg-type]
    assert isinstance(context, torch.autocast)
    assert context.device == "cpu"
    assert context.fast_dtype == dtype


def test_weights_stay_float32_while_ops_run_in_bf16() -> None:
    model = nn.Linear(4, 4)
    with autocast_for(model, "bf16"):
        output = model(torch.randn(2, 4))
    assert output.dtype == torch.bfloat16
    assert model.weight.dtype == torch.float32
