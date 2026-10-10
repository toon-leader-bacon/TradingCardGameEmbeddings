"""Every loss builds its targets on the decoder output's device, not the CPU.

Uses PyTorch's "meta" device as a stand-in for a GPU: a CPU-built target
mixed with a meta-device prediction raises just as it would on CUDA/ROCm,
so this runs on any machine.
"""

from uuid import uuid4

import torch

from src.dojos.contrastive.contrastive_loss import identity_negative_mask
from src.dojos.loss.bce_loss import BceLoss
from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss
from src.dojos.loss.mse_loss import MseLoss
from src.dojos.loss.nocab_loss import device_of
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.loss.soft_classification_loss import SoftClassificationLoss

META = torch.device("meta")


def test_mse_loss_follows_the_decoder_output_device() -> None:
    loss = MseLoss().calculate(torch.zeros(2, device=META), [1.0, 2.0])  # type: ignore[arg-type]
    assert loss.device == META


def test_bce_loss_follows_the_decoder_output_device() -> None:
    loss = BceLoss().calculate(torch.zeros(2, device=META), [1.0, 0.0])  # type: ignore[arg-type]
    assert loss.device == META


def test_fixed_classification_loss_follows_the_decoder_output_device() -> None:
    loss = FixedClassificationLoss(["a", "b"]).calculate(
        torch.zeros(2, 2, device=META), ["a", "b"]  # type: ignore[arg-type]
    )
    assert loss.device == META


def test_soft_classification_loss_follows_the_decoder_output_device() -> None:
    loss = SoftClassificationLoss(["a", "b"]).calculate(
        torch.zeros(1, 2, device=META), [{"a": 1.0}]  # type: ignore[arg-type]
    )
    assert loss.device == META


def test_masked_vector_regression_loss_follows_the_decoder_output_device() -> None:
    loss = MaskedVectorRegressionLoss(["a", "b"]).calculate(
        torch.zeros(1, 2, device=META), [{0: 0.5}]  # type: ignore[arg-type]
    )
    assert loss.device == META


def test_pick_prediction_loss_follows_each_logits_device() -> None:
    loss = PickPredictionCrossEntropyLoss().calculate(
        [torch.zeros(3, device=META), torch.zeros(2, device=META)], [0, 1]
    )
    assert loss.device == META


def test_contrastive_negative_mask_is_built_on_the_similarity_device() -> None:
    # The full loss reads mask entries back as Python bools, which meta
    # tensors cannot do; the mask was its only CPU-built tensor
    mask = identity_negative_mask([uuid4(), uuid4()], META)
    assert mask.device == META


def test_device_of_reads_a_tensor_or_a_list_and_defaults_to_cpu() -> None:
    assert device_of(torch.zeros(2, device=META)) == META
    assert device_of([torch.zeros(2, device=META)]) == META
    assert device_of([]) == torch.device("cpu")
