"""TeacherMethod: the Strategy that turns one game's co-occurrence
counts into one vector per counted card. Its options type picks it
(teacher_method_for), so a run config swaps methods by one `kind` key.

A new method is one options class in teacher_settings.py, one module
implementing TeacherMethod, and one entry in _METHOD_BUILDERS.
"""

from typing import Callable, Protocol

import numpy as np

from src.teacher.cooccurrence import CooccurrenceCounts
from src.teacher.ppmi_svd import PpmiSvdMethod
from src.teacher.teacher_settings import PpmiSvdOptions, TeacherMethodOptions


class TeacherMethod(Protocol):
    """Fits teacher vectors from one game's counts."""

    def fit(self, counts: CooccurrenceCounts) -> np.ndarray:
        """One vector per counted card.

        Inputs: counts. Output: np.ndarray float32 (len(counts.card_uuids),
            dim), rows aligned with counts.card_uuids.
        Side effects: none. Exceptions: ValueError if counts are too small
            to fit.
        """
        ...


# Each options type's method; a new method adds its entry here
_METHOD_BUILDERS: dict[type, Callable[[TeacherMethodOptions], TeacherMethod]] = {
    PpmiSvdOptions: PpmiSvdMethod,
}


def teacher_method_for(options: TeacherMethodOptions) -> TeacherMethod:
    """The method options configure.

    Inputs: options. Output: TeacherMethod.
    Side effects: none.
    Exceptions: ValueError for an options type with no method.

    Example:
        >>> teacher_method_for(PpmiSvdOptions(dim=64)).fit(counts).shape[1]
        64
    """
    builder = _METHOD_BUILDERS.get(type(options))
    if builder is None:
        raise ValueError(f"no teacher method for {type(options).__name__}")
    return builder(options)
