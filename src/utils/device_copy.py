"""Small host-to-device copies that do not stall the host.

torch.tensor(values, device=cuda) and tensor.to(cuda) are blocking
copies: the host waits for every kernel already queued before it can
queue the next one. In a training step that is bound by kernel launches,
each one costs a full pipeline drain. Building on the CPU and copying with
non_blocking=True stages pageable memory without that wait.
"""

from typing import Sequence

import torch


def host_tensor(
    values: Sequence[float | int | bool], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A small tensor of host values on device, without a device sync.

    Inputs: values (host numbers), dtype, device.
    Output: (len(values),) tensor of dtype on device.
    Side effects: one non-blocking host-to-device copy. Exceptions: none.

    Example:
        >>> host_tensor([2, 0, 1], torch.long, torch.device("cpu"))
        tensor([2, 0, 1])
    """
    return to_device(torch.tensor(list(values), dtype=dtype), device)


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """tensor (built on the CPU) on device, without a device sync.

    Inputs: tensor, device.
    Output: the same values on device (tensor itself if already there).
    Side effects: one non-blocking host-to-device copy. Exceptions: none.

    Example:
        >>> to_device(torch.ones(2), torch.device("cpu"))
        tensor([1., 1.])
    """
    return tensor.to(device, non_blocking=True)
