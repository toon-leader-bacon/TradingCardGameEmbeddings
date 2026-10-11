"""CovarianceAccumulator: streaming mean and covariance of vectors."""

import torch


class CovarianceAccumulator:
    """Sums of vectors and of their outer products, in float64 on the CPU,
    so the mean and covariance of far more vectors than fit in memory at
    once can be read off at the end. Knows nothing about what the vectors
    are (token states, pooled cards), which is what makes it reusable for
    later recalibrations."""

    def __init__(self, dim: int) -> None:
        """Inputs: dim (>= 1), the vector width.
        Output: none (constructor). Side effects: allocates the sums.
        Exceptions: ValueError if dim < 1.

        Example:
            >>> CovarianceAccumulator(768).count
            0
        """
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self._dim = dim
        self._count = 0
        self._sum = torch.zeros(dim, dtype=torch.float64)
        self._outer_sum = torch.zeros(dim, dim, dtype=torch.float64)

    @property
    def count(self) -> int:
        """How many vectors have been added. Inputs: none. Output: int.
        Side effects: none. Exceptions: none."""
        return self._count

    def add(self, vectors: torch.Tensor) -> None:
        """Accumulate a batch.

        Inputs: vectors, (n, dim), any float dtype and device; n may be 0.
        Output: none.
        Side effects: updates the running sums (moved to CPU float64).
        Exceptions: ValueError if vectors is not (n, dim).

        Example:
            >>> accumulator.add(torch.randn(100, 768))
        """
        if vectors.dim() != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"vectors must be (n, {self._dim}), got {tuple(vectors.shape)}"
            )
        batch = vectors.detach().to(device="cpu", dtype=torch.float64)
        self._count += batch.shape[0]
        self._sum += batch.sum(dim=0)
        self._outer_sum += batch.T @ batch

    def mean(self) -> torch.Tensor:
        """The mean of every vector added.

        Inputs: none. Output: (dim,) float64.
        Side effects: none. Exceptions: ValueError if count is 0.
        """
        if self._count == 0:
            raise ValueError("no vectors added")
        return self._sum / self._count

    def covariance(self) -> torch.Tensor:
        """The (population) covariance of every vector added.

        Inputs: none. Output: (dim, dim) float64, symmetric.
        Side effects: none. Exceptions: ValueError if count < 2.
        """
        if self._count < 2:
            raise ValueError(f"need at least 2 vectors, have {self._count}")
        mean = self.mean()
        covariance = self._outer_sum / self._count - torch.outer(mean, mean)
        # Symmetric by construction; average away float round-off
        return (covariance + covariance.T) / 2
