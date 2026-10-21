"""FrozenStageFingerprint: what produced a set of cached feature rows.

Two caches may share rows only when every stage before the cut, and the
precision it ran at, are the same. The fingerprint's digest names a row
set in the disk tier, and the cache checks its compute precision on every
call.
"""

import hashlib
import json
from dataclasses import dataclass

from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.precision import Precision
from src.encoder_model.text_encoder import TextEncoderSource


@dataclass(frozen=True)
class FrozenStageFingerprint:
    """Everything that fixes a cached row's numbers, given its text.

    text_encoder: the text encoder's source, or None for one with no fixed
        source (no disk tier may use such a fingerprint).
    enrichment: the whitening stats file's sha256, or "none".
    pooling: the pooling kind, e.g. "mean".
    compute_precision: the autocast the frozen stages run under.
    width: the cached rows' width.
    storage_dtype: the dtype rows are stored in.
    """

    text_encoder: TextEncoderSource | None
    enrichment: str
    pooling: str
    compute_precision: Precision
    width: int
    storage_dtype: FeatureStorageDtype

    def digest(self) -> str:
        """The sha256 hex of this fingerprint's canonical JSON: equal
        fingerprints always give equal digests.

        Inputs: none. Output: str (64 hex characters). Side effects: none.
        Exceptions: none.

        Example:
            >>> len(fingerprint.digest())
            64
        """
        canonical = json.dumps(
            self._as_mapping(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        """One readable line, stored beside the row set on disk.

        Inputs: none. Output: str. Side effects: none. Exceptions: none.

        Example:
            >>> fingerprint.describe()
            'answerdotai/ModernBERT-base@8949b90 len=384 mean fp16 -> 768 fp32'
        """
        source = self.text_encoder
        encoder = "no source"
        if source is not None:
            revision = (source.revision or "local")[:7]
            encoder = f"{source.checkpoint}@{revision} len={source.max_length}"
        enrichment = (
            "" if self.enrichment == "none" else f" whitened {self.enrichment[:7]}"
        )
        return (
            f"{encoder}{enrichment} {self.pooling} {self.compute_precision} "
            f"-> {self.width} {self.storage_dtype.value}"
        )

    def _as_mapping(self) -> dict[str, object]:
        """Every field as JSON-ready plain values.

        Private helper - single consumer is digest.
        Inputs: none. Output: dict. Side effects: none. Exceptions: none.
        """
        source = self.text_encoder
        return {
            "text_encoder": (
                None
                if source is None
                else {
                    "checkpoint": source.checkpoint,
                    "revision": source.revision,
                    "max_length": source.max_length,
                    "model_kwargs": [list(pair) for pair in source.model_kwargs],
                }
            ),
            "enrichment": self.enrichment,
            "pooling": self.pooling,
            "compute_precision": self.compute_precision,
            "width": self.width,
            "storage_dtype": self.storage_dtype.value,
        }
