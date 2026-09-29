"""Build small EmbeddingTables with chosen vectors and labels for analysis
tests."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.evaluation.card_row import CardRow
from src.evaluation.embedding.embedding_table import EmbeddingTable
from src.evaluation.embedding.embedding_table_metadata import EmbeddingTableMetadata
from src.schema.game_id import GameId


class DictLabels:
    """CardLabels over a fixed nocab_uuid -> label dict."""

    def __init__(self, labels: dict, name: str = "fixed") -> None:
        self.name = name
        self._labels = labels

    def label_of(self, row: CardRow) -> str | None:
        return self._labels.get(row.nocab_uuid)


def labeled_table(
    path: Path, vectors: np.ndarray, labels: list[str | None]
) -> tuple[EmbeddingTable, DictLabels]:
    """A new table holding one card per vector, labeled labels[i]."""
    metadata = EmbeddingTableMetadata(
        encoder_label="synthetic",
        checkpoint_dir=None,
        embedding_dim=vectors.shape[1],
        binder_versions={GameId.MTG: "mtg-v1"},
        created_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )
    rows = [CardRow(uuid4(), GameId.MTG) for _ in range(len(vectors))]
    table = EmbeddingTable.create(path, metadata)
    table.add(rows, vectors.astype(np.float32))
    by_uuid = {row.nocab_uuid: label for row, label in zip(rows, labels)}
    return table, DictLabels(by_uuid)


def separated_clusters(
    per_label: int = 20, label_count: int = 3, width: int = 8, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """label_count tight clusters around orthogonal directions."""
    rng = np.random.default_rng(seed)
    vectors, labels = [], []
    for index in range(label_count):
        center = np.zeros(width)
        center[index] = 1.0
        vectors.append(center + 0.02 * rng.standard_normal((per_label, width)))
        labels += [f"label{index}"] * per_label
    return np.vstack(vectors), labels
