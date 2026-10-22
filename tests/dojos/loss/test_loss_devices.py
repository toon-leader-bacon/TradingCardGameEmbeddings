"""Every loss builds its targets on the decoder output's device, not the CPU.

Uses PyTorch's "meta" device as a stand-in for a GPU: a CPU-built target
mixed with a meta-device prediction raises just as it would on CUDA/ROCm,
so this runs on any machine.
"""

from uuid import uuid4

import torch

from src.dojos.contrastive.contrastive_loss import (
    clique_positive_mask,
    info_nce_loss,
    info_nce_masks,
)
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


def test_pick_prediction_loss_follows_the_logits_device() -> None:
    loss = PickPredictionCrossEntropyLoss().calculate(
        torch.zeros(2, 3, device=META), [0, 1]
    )
    assert loss.device == META


def test_info_nce_loss_follows_the_units_device() -> None:
    # The masks are built on the CPU and moved once; nothing is read back
    # from the device, which meta tensors could not do
    identities = [(uuid4(),) for _ in range(4)]
    masks = info_nce_masks(clique_positive_mask([[0, 1], [2, 3]], 4), identities, None)
    loss = info_nce_loss(torch.zeros(4, 3, device=META), masks, 0.07)
    assert loss.device == META


def test_device_of_reads_a_tensor_or_a_list_and_defaults_to_cpu() -> None:
    assert device_of(torch.zeros(2, device=META)) == META
    assert device_of([torch.zeros(2, device=META)]) == META
    assert device_of([]) == torch.device("cpu")
