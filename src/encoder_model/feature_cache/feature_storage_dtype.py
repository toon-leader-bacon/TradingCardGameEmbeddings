"""FeatureStorageDtype: the dtype a feature cache stores its rows in.

Storage rounding only. It is not the autocast compute precision
(precision.Precision): cached features are always handed back as float32.
"""

from enum import StrEnum

import torch

_BYTES_PER_MIB = 2**20


class FeatureStorageDtype(StrEnum):
    """How a cached feature row is stored: fp32 (exact), or fp16/bf16 for
    half the VRAM at a rounding cost."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def torch_dtype(self) -> torch.dtype:
        """The matching torch dtype. Inputs: none. Output: torch.dtype.
        Side effects: none. Exceptions: none.

        Example:
            >>> FeatureStorageDtype.FP16.torch_dtype
            torch.float16
        """
        return _TORCH_DTYPES[self]

    def rows_in(self, capacity_mib: int, width: int) -> int:
        """How many width-wide rows of this dtype fit in capacity_mib MiB of
        payload (keys and indexes not counted). Shared by every tier.

        Inputs: capacity_mib, width (>= 1). Output: int (>= 1).
        Side effects: none.
        Exceptions: ValueError if width < 1 or the budget holds no row.

        Example:
            >>> FeatureStorageDtype.FP32.rows_in(3, 768)
            1024
        """
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        rows = capacity_mib * _BYTES_PER_MIB // (width * self.torch_dtype.itemsize)
        if rows < 1:
            raise ValueError(
                f"capacity_mib {capacity_mib} holds no {width}-wide {self.value} row"
            )
        return rows


_TORCH_DTYPES: dict[FeatureStorageDtype, torch.dtype] = {
    FeatureStorageDtype.FP32: torch.float32,
    FeatureStorageDtype.FP16: torch.float16,
    FeatureStorageDtype.BF16: torch.bfloat16,
}
